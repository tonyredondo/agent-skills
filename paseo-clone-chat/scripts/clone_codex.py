#!/usr/bin/env python3
"""Fork a finished Codex history prefix, or verify an existing fork. No model turns."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid


def save_state(path, state):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as out:
        json.dump(state, out, indent=2)
        out.write("\n")
        temporary = Path(out.name)
    os.replace(temporary, path)
    if json.loads(path.read_text()) != state:
        raise RuntimeError("State write could not be verified")


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def history_proof(turns):
    digest = hashlib.sha256()
    for turn in turns:
        digest.update(json.dumps(turn, sort_keys=True, separators=(",", ":")).encode())
    return {"turn_count": len(turns), "item_count": sum(len(t.get("items", [])) for t in turns),
            "sha256": digest.hexdigest(), "last_turn_id": turns[-1]["id"] if turns else None}


def finished_prefix(turns, through_turn=None, snapshot_finished=False):
    if through_turn:
        for index, turn in enumerate(turns):
            if turn["id"] == through_turn:
                prefix = turns[:index + 1]
                break
        else:
            raise RuntimeError("The requested cutoff is absent from the source history")
    else:
        prefix = list(turns)
        # A read from another app-server process reports a live unfinished turn
        # as interrupted. Paseo's live running status selects this snapshot mode.
        while snapshot_finished and prefix and prefix[-1].get("status") in ("inProgress", "interrupted"):
            prefix.pop()
    if not prefix:
        raise RuntimeError("No finished source turn exists; no copy created")
    if any(t.get("status") == "inProgress" for t in prefix):
        raise RuntimeError("The selected history includes an in-progress turn")
    return prefix


class AppServer:
    def __init__(self, cwd):
        self.cwd = cwd
        self.process = None
        self.request_id = 0

    async def __aenter__(self):
        self.process = await asyncio.create_subprocess_exec(
            "codex", "app-server", "--stdio", cwd=self.cwd,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, limit=512 * 1024 * 1024)
        return self

    async def __aexit__(self, *_):
        self.process.stdin.close()
        try:
            await asyncio.wait_for(self.process.wait(), 5)
        except asyncio.TimeoutError:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()

    async def send(self, message):
        self.process.stdin.write((json.dumps(message) + "\n").encode())
        await self.process.stdin.drain()

    async def rpc(self, method, params):
        self.request_id += 1
        request_id = self.request_id
        await self.send({"id": request_id, "method": method, "params": params})
        while True:
            line = await self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"App-server closed before responding to {method}")
            message = json.loads(line)
            if message.get("id") == request_id and "method" not in message:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message["result"]
            if "id" in message and "method" in message:
                await self.send({"id": message["id"], "error": {
                    "code": -32601, "message": "Only session duplication is authorized"}})

    async def read(self, thread_id, include_turns):
        result = await self.rpc("thread/read", {"threadId": thread_id, "includeTurns": include_turns})
        if result["thread"]["id"] != thread_id:
            raise RuntimeError("App-server returned an unexpected thread")
        return result["thread"]


async def execute(args, path, state):
    async with AppServer(args.cwd) as server:
        await server.rpc("initialize", {
            "clientInfo": {"name": "paseo_clone_chat", "version": "1.0.0"},
            "capabilities": {"experimentalApi": True}})
        await server.send({"method": "initialized", "params": {}})
        source = await server.read(args.source, False)
        if Path(source["cwd"]).resolve() != Path(args.cwd).resolve():
            raise RuntimeError("Source cwd does not match the selected workspace")
        source_path = source.get("path")
        before_hash = file_hash(source_path) if source_path else None
        source = await server.read(args.source, True)
        prefix = finished_prefix(source["turns"], args.through_turn, args.snapshot_finished)
        proof = history_proof(prefix)
        state.update({"source_proof": proof, "through_turn": proof["last_turn_id"],
                      "excluded_turns": len(source["turns"]) - len(prefix),
                      "source_path": source_path, "source_sha256": before_hash})
        del source, prefix
        save_state(path, state)
        if args.verify_copy:
            state["copy_id"] = args.verify_copy
        else:
            state["status"] = "fork_requested"
            save_state(path, state)
            params = {"threadId": args.source, "cwd": args.cwd, "excludeTurns": True}
            if args.snapshot_finished or args.through_turn:
                params["lastTurnId"] = state["through_turn"]
            if args.model:
                params["model"] = args.model
            if args.thinking:
                params["config"] = {"model_reasoning_effort": args.thinking}
            result = await server.rpc("thread/fork", params)
            state.update({"copy_id": result["thread"]["id"], "status": "fork_created"})
        save_state(path, state)
        if state["copy_id"] == args.source:
            raise RuntimeError("Fork must have a different thread ID")
        copy = await server.read(state["copy_id"], True)
        if copy.get("forkedFromId") != args.source:
            raise RuntimeError("Copy does not identify the requested source as its parent")
        state["copy_proof"] = history_proof(copy["turns"])
        del copy
        if state["source_proof"] != state["copy_proof"]:
            raise RuntimeError("Copy history differs from the selected source prefix")
        unchanged = bool(source_path) and before_hash == file_hash(source_path)
        state["source_file_unchanged"] = unchanged
        if not unchanged:
            # The source may be running this workflow. Appended turns do not
            # invalidate the frozen prefix, which must still match exactly.
            current = await server.read(args.source, True)
            actual = history_proof(finished_prefix(current["turns"], state["through_turn"]))
            if actual != proof:
                raise RuntimeError("The source prefix changed during verification")
        state["status"] = "history_verified"
        save_state(path, state)
        print(json.dumps({"state_file": str(path), **state}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=lambda v: str(uuid.UUID(v)))
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--model")
    parser.add_argument("--thinking")
    parser.add_argument("--verify-copy", type=lambda v: str(uuid.UUID(v)))
    parser.add_argument("--through-turn")
    parser.add_argument("--snapshot-finished", action="store_true",
                        help="Source is running: exclude unfinished trailing turns")
    parser.add_argument("--state", type=Path, help="New state file; existing files are never overwritten")
    args = parser.parse_args()
    args.cwd = str(Path(args.cwd).resolve(strict=True))
    path = args.state or Path(tempfile.mkdtemp(prefix="paseo-clone-chat-")) / "state.json"
    # Reserve before any mutation. Reusing this path cannot silently fork twice.
    with path.open("x"):
        pass
    os.chmod(path, 0o600)
    state = {"status": "prepared", "source_id": args.source, "cwd": args.cwd}
    save_state(path, state)
    print(json.dumps({"state_file": str(path), "status": "prepared"}), flush=True)
    try:
        asyncio.run(asyncio.wait_for(execute(args, path, state), 180))
    except Exception as error:
        state["error"] = f"{type(error).__name__}: {error}"
        save_state(path, state)
        print(json.dumps({"state_file": str(path), **state}), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

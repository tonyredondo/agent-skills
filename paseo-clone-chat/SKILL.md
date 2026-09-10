---
name: paseo-clone-chat
description: Clone a Codex conversation and import the verified copy into Paseo with the source workspace and settings. Use for requests such as "clona este chat", "duplica esta conversación", or a Codex session ID to fork in Paseo. Applies to Codex chats, not Git repositories or other providers.
---

# Clone a Codex chat in Paseo

Create one independent conversation in the source agent's Paseo workspace, verify its history, and leave it idle for the user. Workspace files remain shared. A request to clone authorizes the fork, import, and restoration of source settings; do not ask again once the source is unambiguous. A question about whether cloning is possible is informational.

## Identify the source

- Accept a Codex UUID, `codex resume <UUID>`, or a Paseo agent ID. These are different identifiers; never run the supplied `resume` command to identify the chat.
- For "this chat", read only the `PASEO_AGENT_ID` environment variable and confirm it with Paseo `get_agent_status`. Otherwise use `list_agents` filtered to the known directory and inspect candidate statuses. Ask if the source or daemon remains ambiguous; never substitute the newest session.
- Record the source agent ID, `persistence.sessionId`, `cwd`, `workspaceId`, model, effective thinking option, mode, and editable toggles from the live snapshot.
- Run the helpers on the machine/container owning that Codex session and the selected Paseo daemon. The client device may be elsewhere. Use existing authentication; do not transfer sessions or credentials between machines.
- A running source is allowed: pass `--snapshot-finished` to snapshot through its last finished turn. Say this briefly. Do not interrupt the source. For the current chat, always use this flag. No finished history means stop without creating a copy. A separate app-server can label the active turn `interrupted`, so its reconstructed status alone cannot establish that the source is idle.

## Create and verify one fork

Use source values; omit absent model/thinking options rather than guessing:

```bash
python3 <skill-dir>/scripts/clone_codex.py \
  --source <codex-thread-id> --cwd <source-cwd> \
  --model <source-model> --thinking <source-thinking>
```

The helper prints its private temporary state path before mutation and saves the copy ID immediately after `thread/fork` returns. Keep this recovery key. It uses native JSON-RPC, sends no model turn, and has a 180-second total limit and a 512 MiB response limit. Use yielding execution to keep progress visible. Inspect the reason before proposing larger limits.

The helper compares effective histories returned by `thread/read`: finished turn IDs, item counts, and canonical SHA-256. A native fork can store a reference to parent history and have a tiny rollout file; file size or raw JSONL equality does not verify history. This is a conversation fork, not a self-contained backup; leave parent session files intact.

Only `status: history_verified` may proceed to import. On timeout/failure, inspect the saved state and reconcile existing sessions before any retry. Do not fork again when a copy ID exists or the outcome is uncertain. Verify an existing copy without creating another:

```bash
python3 <skill-dir>/scripts/clone_codex.py \
  --source <source-id> --cwd <source-cwd> --verify-copy <copy-id> \
  --through-turn <saved-cutoff-turn-id>
```

## Import into the same workspace

Find the installed `@getpaseo/cli` root from the resolved `paseo` executable or `npm root -g`; verify its `package.json`. Do not scan the filesystem or install/update packages as part of cloning.

```bash
node <skill-dir>/scripts/import_paseo.mjs \
  --cli-root <installed-@getpaseo/cli-directory> \
  --host <verified-daemon-endpoint> --source-agent <paseo-agent-id> \
  --state <verified-state.json> --title '<source-title> (copia)'
```

The helper reuses Paseo's installed CLI connection helper and `importAgent` API with explicit `workspaceId`. The plain `paseo import` CLI in 0.7.2 lacks that option. If this installed API is unavailable, inspect the current supported interface; do not copy or edit Paseo storage files.

The imported agent ID is saved before configuration. The helper copies the source model, effective thinking option, mode, and editable toggles, then verifies the new session/workspace mapping and idle status. It never creates a worktree or sends an initial prompt.

On partial import, reconcile the copy's Codex ID against Paseo snapshots. Rerunning with a recorded `new_agent_id` finishes that agent instead of importing another. If `import_requested` was saved without an agent ID, locate the matching imported agent first and pass `--existing-agent <id>`; do not blindly retry. Use `--check --existing-agent <id>` with the same state and title for read-only verification of an existing import.

## Finish

Verify the original still maps to its original session and the copy is idle, in the intended workspace, with source settings. Report its Paseo title and `codex resume <new-id>`. For a running source, mention the finished-turn cutoff. Do not start work, archive chats, delete sessions, restart daemons, or change source settings.

Validated against Codex CLI 0.153.4 and Paseo 0.7.2. For protocol drift, prefer the installed Codex JSON schema and Paseo client types. Docs: [Codex app-server](https://learn.chatgpt.com/docs/app-server) and [Codex in Paseo](https://paseo.sh/docs/codex.md).

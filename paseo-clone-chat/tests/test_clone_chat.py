"""Exercise native history preservation and recovery without user sessions or models."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import uuid


SKILL = Path(__file__).resolve().parents[1]


class NativeForkTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(shutil.which("codex"), "Native tests require the Codex CLI")
        self.temporary = tempfile.TemporaryDirectory(prefix="paseo-clone-native-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cwd = self.root / "workspace"
        self.cwd.mkdir()
        self.codex_home = self.root / "codex"
        self.env = dict(os.environ, CODEX_HOME=str(self.codex_home))
        self.source_id = str(uuid.uuid4())

    def write_source(self, finished=2, active=False):
        now = datetime.datetime.now(datetime.timezone.utc)
        timestamp = now.isoformat().replace("+00:00", "Z")
        folder = self.codex_home / "sessions" / now.strftime("%Y/%m/%d")
        folder.mkdir(parents=True)
        self.rollout = folder / f"rollout-{now:%Y-%m-%dT%H-%M-%S}-{self.source_id}.jsonl"
        records = []

        def add(kind, payload):
            records.append({"timestamp": timestamp, "type": kind, "payload": payload})

        add("session_meta", {
            "id": self.source_id, "session_id": self.source_id,
            "timestamp": timestamp, "cwd": str(self.cwd), "originator": "codex_cli_rs",
            "cli_version": "0.153.4", "source": "cli", "model_provider": "openai",
            "base_instructions": {"text": "Synthetic fixture. Never run a model turn."},
            "history_mode": "legacy",
        })
        for number in range(finished + int(active)):
            turn_id = str(uuid.uuid4())
            add("event_msg", {"type": "task_started", "turn_id": turn_id,
                              "model_context_window": None, "collaboration_mode_kind": "default"})
            add("event_msg", {"type": "user_message", "message": f"Fixture request {number}",
                              "images": [], "local_images": [], "text_elements": []})
            if number < finished:
                add("event_msg", {"type": "agent_message", "message": f"Fixture answer {number}",
                                  "phase": "final_answer"})
                add("event_msg", {"type": "task_complete", "turn_id": turn_id,
                                  "last_agent_message": f"Fixture answer {number}"})
        self.rollout.write_text("".join(json.dumps(r) + "\n" for r in records))

    def run_helper(self, *options, state="proof.json"):
        command = ["python3", str(SKILL / "scripts/clone_codex.py"),
                   "--source", self.source_id, "--cwd", str(self.cwd),
                   "--state", str(self.root / state), *options]
        return subprocess.run(command, env=self.env, text=True, capture_output=True, timeout=40)

    def test_idle_fork_preserves_history_and_refuses_duplicate_state(self):
        self.write_source()
        before = hashlib.sha256(self.rollout.read_bytes()).hexdigest()
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        proof = json.loads((self.root / "proof.json").read_text())
        self.assertEqual(proof["status"], "history_verified")
        self.assertEqual(proof["source_proof"], proof["copy_proof"])
        self.assertEqual(proof["copy_proof"]["turn_count"], 2)
        self.assertEqual(proof["copy_proof"]["item_count"], 4)
        self.assertNotEqual(proof["copy_id"], self.source_id)
        self.assertEqual(hashlib.sha256(self.rollout.read_bytes()).hexdigest(), before)
        paths = set(self.codex_home.rglob("*.jsonl"))
        self.assertNotEqual(self.run_helper().returncode, 0)
        self.assertEqual(set(self.codex_home.rglob("*.jsonl")), paths)

    def test_running_source_copies_only_the_finished_prefix(self):
        self.write_source(active=True)
        result = self.run_helper("--snapshot-finished")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        proof = json.loads((self.root / "proof.json").read_text())
        self.assertEqual(proof["source_proof"], proof["copy_proof"])
        self.assertEqual(proof["copy_proof"]["turn_count"], 2)
        self.assertEqual(proof["copy_proof"]["item_count"], 4)
        self.assertEqual(proof["excluded_turns"], 1)
        self.assertTrue(proof["source_file_unchanged"])

    def test_unfinished_only_source_and_invalid_cutoff_create_no_fork(self):
        self.write_source(finished=0, active=True)
        paths = set(self.codex_home.rglob("*.jsonl"))
        self.assertNotEqual(self.run_helper("--snapshot-finished").returncode, 0)
        self.assertNotEqual(self.run_helper("--through-turn", "absent", state="invalid.json").returncode, 0)
        self.assertEqual(set(self.codex_home.rglob("*.jsonl")), paths)


class PaseoImportTests(unittest.TestCase):
    def test_partial_import_recovers_without_duplicates_or_source_mutation(self):
        self.assertIsNotNone(shutil.which("node"), "Import tests require Node.js")
        with tempfile.TemporaryDirectory(prefix="paseo-clone-import-test-") as directory:
            root = Path(directory)
            package = root / "cli"
            (package / "dist/utils").mkdir(parents=True)
            (package / "package.json").write_text(json.dumps({"name": "@getpaseo/cli", "type": "module"}))
            (package / "dist/utils/client.js").write_text(CLIENT_FIXTURE)
            source = {
                "id": "source-agent", "provider": "codex", "cwd": "/fixture",
                "workspaceId": "workspace", "persistence": {"sessionId": "source-thread"},
                "status": "idle", "model": "fixture-model", "effectiveThinkingOptionId": "xhigh",
                "currentModeId": "full-access",
                "features": [{"id": "fast_mode", "type": "toggle", "value": True}],
            }
            fixture = root / "fixture.json"
            fixture.write_text(json.dumps({"source": source, "imports": 0, "copy": None, "fail": True}))
            proof = {"turn_count": 2, "item_count": 4, "sha256": "fixture-hash", "last_turn_id": "turn-two"}
            state = root / "state.json"
            state.write_text(json.dumps({"status": "history_verified", "source_id": "source-thread",
                                         "copy_id": "copy-thread", "cwd": "/fixture",
                                         "source_proof": proof, "copy_proof": proof}))
            command = ["node", str(SKILL / "scripts/import_paseo.mjs"), "--cli-root", str(package),
                       "--host", "fixture", "--source-agent", "source-agent", "--state", str(state),
                       "--title", "Fixture copy"]

            def run(*options):
                return subprocess.run(command + list(options), env=dict(os.environ, FIXTURE_STATE=str(fixture)),
                                      text=True, capture_output=True, timeout=15)

            first = run()
            self.assertNotEqual(first.returncode, 0)
            self.assertIn("injected configuration failure", first.stderr)
            partial = json.loads(state.read_text())
            self.assertEqual(partial["status"], "imported")
            self.assertEqual(partial["new_agent_id"], "copy-agent")
            second = run()
            self.assertEqual(second.returncode, 0, second.stderr)
            result = json.loads(fixture.read_text())
            self.assertEqual(result["imports"], 1)
            self.assertEqual(result["source"], source)
            self.assertEqual(json.loads(state.read_text())["status"], "complete")
            checked = run("--check")
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertEqual(json.loads(fixture.read_text()), result)
            uncertain = json.loads(state.read_text())
            uncertain["status"] = "import_requested"
            uncertain.pop("new_agent_id")
            state.write_text(json.dumps(uncertain))
            self.assertNotEqual(run().returncode, 0)
            self.assertEqual(json.loads(fixture.read_text())["imports"], 1)


CLIENT_FIXTURE = """
import fs from 'node:fs';
import assert from 'node:assert/strict';
const file = process.env.FIXTURE_STATE;
const read = () => JSON.parse(fs.readFileSync(file, 'utf8'));
const save = state => fs.writeFileSync(file, JSON.stringify(state));
export async function connectToDaemon({ host }) {
  assert.equal(host, 'fixture');
  return {
    async fetchAgent({ agentId }) {
      const state = read();
      assert.ok(['source-agent', 'copy-agent'].includes(agentId));
      return { agent: agentId === 'source-agent' ? state.source : state.copy };
    },
    async importAgent(input) {
      const state = read();
      assert.equal(input.sessionId, 'copy-thread');
      assert.equal(input.workspaceId, 'workspace');
      state.imports++;
      state.copy = { id: 'copy-agent', provider: 'codex', cwd: '/fixture',
        workspaceId: 'workspace', persistence: { sessionId: 'copy-thread' },
        status: 'idle', features: [{ id: 'fast_mode', type: 'toggle', value: false }] };
      save(state);
      return state.copy;
    },
    async applyAgentConfig(id, config) {
      assert.equal(id, 'copy-agent');
      const state = read();
      if (state.fail) {
        state.fail = false;
        save(state);
        throw Error('injected configuration failure');
      }
      state.copy.model = config.modelId;
      state.copy.effectiveThinkingOptionId = config.thinkingOptionId;
      state.copy.currentModeId = config.modeId;
      state.copy.features = [{ id: 'fast_mode', type: 'toggle', value: config.featureValues.fast_mode }];
      save(state);
    },
    async updateAgent(id, updates) {
      assert.equal(id, 'copy-agent');
      const state = read();
      state.copy.title = updates.name;
      save(state);
    },
    async close() {},
  };
}
"""


if __name__ == "__main__":
    unittest.main()

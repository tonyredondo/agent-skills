# paseo-clone-chat

Clone a Codex conversation and import the verified copy into the original Paseo workspace. The copy is left idle, ready for the user to continue. Workspace files remain shared.

## Install on another machine

Ask Codex:

```text
Use $skill-installer to install https://github.com/tonyredondo/agent-skills/tree/main/paseo-clone-chat
```

The skill installer copies this directory and its helper scripts into your Codex skills directory. It does not transfer conversations, configure accounts, or install Paseo.

## Requirements

- Python 3.11 or newer, including the standard library; no pip dependencies.
- Node.js 20 or newer.
- An installed Codex CLI and an existing, authenticated Paseo daemon on the machine that owns the source conversation.
- The installed `@getpaseo/cli` package, including its `dist/utils/client.js` connection helper.

The live workflow was verified on Linux with Codex CLI 0.153.4 and Paseo 0.7.2. Other operating systems have not been validated. Run the helpers on the daemon machine or inside its container, even when the Paseo client runs elsewhere.

## Usage

Ask the agent:

```text
Clona este chat en Paseo.
```

Or provide an explicit source:

```text
$paseo-clone-chat clona esta conversación: codex resume <thread-id>
```

The agent resolves the source, verifies the native fork, imports it into the same workspace, and restores its settings. When cloning a running conversation, the copy ends at the last finished turn. No new model turn is started.

## Implementation and recovery

`scripts/clone_codex.py` communicates with `codex app-server` over stdio. It compares effective histories from `thread/read` using a canonical SHA-256, turn counts, and item counts. Native forks may reference their parent's history; a small fork file is expected and the parent session files must remain intact.

`scripts/import_paseo.mjs` reuses Paseo's authenticated CLI connection and imports with an explicit workspace ID. It records the new agent ID before applying settings so an interrupted operation can resume without another import.

Both scripts expose `--help`. See [SKILL.md](SKILL.md) for invocation, verification-only modes, time limits, and recovery from ambiguous outcomes. Keep the state path printed by the first helper until the operation is complete. State files contain identifiers and hashes, not conversation content.

## Validation

From the repository root:

```bash
python3 -m unittest discover -s paseo-clone-chat/tests -p 'test_*.py'
node --check paseo-clone-chat/scripts/import_paseo.mjs
```

Tests exercise real native forks with synthetic sessions in a temporary `CODEX_HOME`, and import/recovery behavior against an isolated Paseo client fixture. They do not call a model, connect to a real Paseo daemon, or modify your conversations. The native tests require the Codex CLI; the import tests require Node.js.

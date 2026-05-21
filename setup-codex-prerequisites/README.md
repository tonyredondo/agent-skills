# setup-codex-prerequisites

Skill for preparing a Windows, macOS, or Debian/Ubuntu Linux workstation with the baseline tools Codex usually needs for development, repository inspection, package management, linting, light auditing, and builds.

Initial Linux support is limited to Debian and Ubuntu. Other distributions should stop without installing anything until they have a dedicated bootstrap path.

## What It Installs

- `winget` bootstrap on Windows and Homebrew bootstrap on macOS when they are missing.
- `apt` support for Debian/Ubuntu.
- Node.js 22 on Debian/Ubuntu through NodeSource when the system Node.js version is too old for `pnpm`.
- Python 3.13 on Windows when neither `python` nor `py` is available; `python3` on macOS/Linux.
- Python tools: `uv`, `uvx`, `pipx`, `PyYAML`.
- Isolated CLIs through `pipx`: `ruff`, `pytest`, `mypy`, `pre-commit`, `pip-audit`.
- Baseline tools: `git`, `gh`, `rg`, `pwsh`, `node`, `npm`, `pnpm`.
- Utilities: `jq`, `yq`, `fd`, `fzf`, `bat`, `delta`, `7z`, `just`.
- Build and audit tools: `cmake`, `ninja`, `gitleaks`, `shellcheck`, `shfmt`, `hadolint`.

## Local Usage

From the skill root:

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-codex-prerequisites.ps1
```

macOS/Linux:

```bash
./scripts/install.sh
```

The script is idempotent: if a tool is already available and passes its version check, it is not reinstalled. If a command exists but does not pass its check, the script tries to install or repair the corresponding package.

## Verification

The installer finishes by verifying that each expected command resolves by name from `PATH` and responds with version or help output. If a terminal was open before the `PATH` changes, it may need to be restarted.

The Linux path was tested on WSL with Ubuntu 24.04. When the user does not have non-interactive `sudo`, the script fails fast with a clear message instead of waiting for a password.

## Security

This skill does not configure secrets, API keys, or service-specific integrations. It only installs general environment tools.

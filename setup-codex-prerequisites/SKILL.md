---
name: setup-codex-prerequisites
description: Install and verify a Codex workstation baseline on Windows, macOS, and Debian/Ubuntu Linux. Use when setting up a new computer, repairing a developer PATH, preparing Python and non-Python CLI tools for Codex, installing uv/pipx-managed Python tools, installing lightweight audit tools, installing winget/Homebrew/apt developer tools, or validating that Codex prerequisites are available by command name.
---

# Setup Codex Prerequisites

## Overview

Use this skill to bootstrap Windows, macOS, and Debian/Ubuntu Linux machines with the CLI tools Codex commonly needs for coding, repository inspection, package management, linting, auditing, archives, and build workflows.

The bundled script installs the current baseline from this skill and verifies every command by name before reporting success.

The bootstrap order is platform-specific and deliberate: first make the native package manager available, then install Python support tools and native CLI tools.

## Workflow

1. Choose the installer for the host OS. Use Windows PowerShell on Windows, `install.sh` on macOS/Linux, or the direct platform script when needed.

2. On Windows, run:

```powershell
$skillPath = Join-Path $env:USERPROFILE ".codex\skills\setup-codex-prerequisites"
$installer = Join-Path $skillPath "scripts\install-codex-prerequisites.ps1"
powershell -ExecutionPolicy Bypass -File $installer
```

3. On macOS or Linux, run from the skill root:

```bash
./scripts/install.sh
```

4. On Linux, only Debian and Ubuntu are supported. On other distros, stop and report that the distro needs a separate bootstrap.

5. After installing, tell the user which tools were installed and which checks passed. Mention that already-open terminals may need to be restarted to inherit PATH changes.

## Installed Baseline

The script installs missing native tools, upgrades pipx-managed Python tools when possible, and verifies every command by name:

- Windows package manager bootstrap: `winget` / App Installer registration or repair
- macOS package manager bootstrap: Homebrew when missing
- Linux package manager support: `apt` on Debian/Ubuntu
- Linux Node.js support: NodeSource Node.js 22 when the distro Node.js is too old for current `pnpm`
- Python runtime: Python 3.13 on Windows when no `python` or `py` command is available; `python3` on macOS/Linux
- Python environment tools: `uv`, `uvx`, `pipx`
- Python support libraries: `PyYAML` for Codex skill validation scripts
- pipx-managed tools: `ruff`, `pytest`, `mypy`, `pre-commit`, `pip-audit`
- Audit/security tools: `gitleaks`, `pip-audit`, `shellcheck`, `shfmt`, `hadolint`
- Core Codex tools: `git`, `gh`, `rg`, `pwsh`
- JavaScript tools: `node`, `npm`, `pnpm`
- General CLI tools: `jq`, `yq`, `fd`, `fzf`, `bat`, `delta`, `7z`, `just`, `cmake`, `ninja`

The scripts also add common user PATH entries when the target directory exists or must be created:

- The active Python user `Scripts` directory, discovered from Python itself
- `%USERPROFILE%\.local\bin`
- `C:\Program Files\7-Zip`
- `C:\Program Files\Git\cmd`
- `C:\Program Files\nodejs`
- `%APPDATA%\npm`
- `C:\Program Files\PowerShell\7`
- `$HOME/.local/bin` on macOS/Linux

PowerShell may also resolve through the WindowsApps alias path on Windows, depending on how it is installed; the scripts verify `pwsh` by command name either way.

## Safety Notes

- Do not retrieve or print secrets while using this skill.
- Do not configure API keys or service-specific integrations in this installer.
- Use Microsoft's App Installer registration first when `winget` is missing; fall back to `Microsoft.WinGet.Client` repair only if registration does not make `winget` available.
- Use Homebrew on macOS and `apt` only on Debian/Ubuntu Linux.
- Use `pipx` for Python CLI tools so each app stays isolated from the user's base Python.
- Use `winget` for native Windows CLI tools when available.

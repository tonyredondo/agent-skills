---
name: setup-codex-prerequisites
description: Install and verify a Windows workstation baseline for Codex work. Use when setting up a new computer, repairing a developer PATH, preparing Python and non-Python CLI tools for Codex, installing uv/pipx-managed Python tools, installing lightweight audit tools, installing common winget developer tools, or validating that Codex prerequisites are available by command name.
---

# Setup Codex Prerequisites

## Overview

Use this skill to bootstrap a Windows machine with the CLI tools Codex commonly needs for coding, repository inspection, package management, linting, auditing, archives, and build workflows.

This skill is Windows-only. If the current machine is not Windows, do not attempt installation; report that this skill cannot run on the current OS and that a separate macOS/Linux bootstrap script is needed.

The bundled script installs the current baseline from this skill and verifies every command by name before reporting success.

The bootstrap order is deliberate: first make `winget` available, then install Python if missing, then install Python support tools and native CLI tools.

## Workflow

1. Confirm the host OS is Windows before running the installer. On non-Windows hosts, stop and report that the skill is Windows-only.

2. Prefer the bundled installer:

```powershell
$skillPath = Join-Path $env:USERPROFILE ".codex\skills\setup-codex-prerequisites"
$installer = Join-Path $skillPath "scripts\install-codex-prerequisites.ps1"
powershell -ExecutionPolicy Bypass -File $installer
```

3. After installing, tell the user which tools were installed and which checks passed. Mention that already-open terminals may need to be restarted to inherit PATH changes.

## Installed Baseline

The script installs missing native tools, upgrades pipx-managed Python tools when possible, and verifies every command by name:

- Windows package manager bootstrap: `winget` / App Installer registration or repair
- Python runtime: Python 3.13 when no `python` or `py` command is available
- Python environment tools: `uv`, `uvx`, `pipx`
- Python support libraries: `PyYAML` for Codex skill validation scripts
- pipx-managed tools: `ruff`, `pytest`, `mypy`, `pre-commit`, `pip-audit`
- Audit/security tools: `gitleaks`, `pip-audit`, `shellcheck`, `shfmt`, `hadolint`
- Core Codex tools: `git`, `gh`, `rg`, `pwsh`
- JavaScript tools: `node`, `npm`, `pnpm`
- General CLI tools: `jq`, `yq`, `fd`, `fzf`, `bat`, `delta`, `7z`, `just`, `cmake`, `ninja`

The script also adds common user PATH entries when the target directory exists or must be created:

- The active Python user `Scripts` directory, discovered from Python itself
- `%USERPROFILE%\.local\bin`
- `C:\Program Files\7-Zip`
- `C:\Program Files\Git\cmd`
- `C:\Program Files\nodejs`
- `%APPDATA%\npm`
- `C:\Program Files\PowerShell\7`

PowerShell may also resolve through the WindowsApps alias path, depending on how it is installed; the script verifies `pwsh` by command name either way.

## Safety Notes

- Do not retrieve or print secrets while using this skill.
- Do not configure API keys or service-specific integrations in this installer.
- Use Microsoft's App Installer registration first when `winget` is missing; fall back to `Microsoft.WinGet.Client` repair only if registration does not make `winget` available.
- Use `pipx` for Python CLI tools so each app stays isolated from the user's base Python.
- Use `winget` for native Windows CLI tools when available.

#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "setup-codex-prerequisites macOS installer only runs on macOS. Nothing was installed." >&2
  exit 2
fi

ensure_path_entry() {
  case ":$PATH:" in
    *":$1:"*) ;;
    *) export PATH="$1:$PATH" ;;
  esac
}

ensure_homebrew() {
  if command -v brew >/dev/null 2>&1; then
    return
  fi

  echo "Homebrew was not found. Installing Homebrew."
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
}

command_healthy() {
  local command_name="$1"
  shift

  command -v "$command_name" >/dev/null 2>&1 || return 1
  if [[ "$#" -gt 0 ]]; then
    "$command_name" "$@" >/dev/null 2>&1 || return 1
  fi
}

brew_install_if_needed() {
  local package_name="$1"
  local command_name="$2"
  shift 2

  if command_healthy "$command_name" "$@"; then
    echo "brew tool already healthy: $command_name"
    return
  fi

  echo "Installing or repairing Homebrew package: $package_name"
  brew install "$package_name"
}

brew_install_cask_if_needed() {
  local package_name="$1"
  local command_name="$2"
  shift 2

  if command_healthy "$command_name" "$@"; then
    echo "Homebrew cask tool already healthy: $command_name"
    return
  fi

  echo "Installing or repairing Homebrew cask: $package_name"
  brew install --cask "$package_name"
}

pipx_install_or_upgrade() {
  local package_name="$1"
  local command_name="$2"

  if command -v "$command_name" >/dev/null 2>&1; then
    echo "pipx tool already available, upgrading: $package_name"
    pipx upgrade "$package_name" || echo "pipx upgrade did not complete for $package_name; leaving existing command in place."
    return
  fi

  echo "Installing pipx package: $package_name"
  pipx install "$package_name"
}

install_pyyaml() {
  python3 -m pip install --user --upgrade PyYAML || \
    python3 -m pip install --user --break-system-packages --upgrade PyYAML
}

verify_tool() {
  local command_name="$1"
  shift

  local command_path
  command_path="$(command -v "$command_name")"
  echo "Verified command: $command_name -> $command_path"

  if [[ "$#" -gt 0 ]]; then
    "$command_name" "$@" | head -n 3
  else
    "$command_name" | head -n 3
  fi
}

ensure_homebrew
brew update

ensure_path_entry "$HOME/.local/bin"

brew_install_if_needed python@3.13 python3 --version
brew_install_if_needed uv uv --version
brew_install_if_needed pipx pipx --version
brew_install_if_needed git git --version
brew_install_if_needed gh gh --version
brew_install_if_needed ripgrep rg --version
brew_install_cask_if_needed powershell pwsh --version
brew_install_if_needed node node --version
brew_install_if_needed node npm --version
brew_install_if_needed jq jq --version
brew_install_if_needed yq yq --version
brew_install_if_needed fd fd --version
brew_install_if_needed fzf fzf --version
brew_install_if_needed bat bat --version
brew_install_if_needed git-delta delta --version
brew_install_if_needed p7zip 7z
brew_install_if_needed just just --version
brew_install_if_needed shellcheck shellcheck --version
brew_install_if_needed shfmt shfmt --version
brew_install_if_needed hadolint hadolint --version
brew_install_if_needed cmake cmake --version
brew_install_if_needed ninja ninja --version
brew_install_if_needed pnpm pnpm --version
brew_install_if_needed gitleaks gitleaks version

install_pyyaml
pipx ensurepath
ensure_path_entry "$HOME/.local/bin"

pipx_install_or_upgrade ruff ruff
pipx_install_or_upgrade pytest pytest
pipx_install_or_upgrade mypy mypy
pipx_install_or_upgrade pre-commit pre-commit
pipx_install_or_upgrade pip-audit pip-audit

echo "Verifying installed commands"
verify_tool uv --version
verify_tool uvx --version
verify_tool pipx --version
verify_tool python3 --version
python3 -c "import yaml; print('PyYAML ' + yaml.__version__)"
verify_tool ruff --version
verify_tool pytest --version
verify_tool mypy --version
verify_tool pre-commit --version
verify_tool pip-audit --version
verify_tool git --version
verify_tool gh --version
verify_tool rg --version
verify_tool pwsh --version
verify_tool node --version
verify_tool npm --version
verify_tool pnpm --version
verify_tool jq --version
verify_tool yq --version
verify_tool fd --version
verify_tool fzf --version
verify_tool bat --version
verify_tool delta --version
verify_tool 7z
verify_tool just --version
verify_tool shellcheck --version
verify_tool shfmt --version
verify_tool hadolint --version
verify_tool cmake --version
verify_tool ninja --version
verify_tool gitleaks version

echo "Codex prerequisite setup completed for macOS. Restart already-open terminals to inherit PATH changes."

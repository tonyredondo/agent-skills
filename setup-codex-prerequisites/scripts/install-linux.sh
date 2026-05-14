#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "setup-codex-prerequisites Linux installer only runs on Linux. Nothing was installed." >&2
  exit 2
fi

if [[ ! -r /etc/os-release ]]; then
  echo "Cannot detect Linux distribution because /etc/os-release is missing. Nothing was installed." >&2
  exit 2
fi

# shellcheck disable=SC1091
. /etc/os-release

if [[ "${ID:-}" != "ubuntu" && "${ID:-}" != "debian" ]]; then
  echo "This installer currently supports Debian/Ubuntu Linux only. Detected: ${PRETTY_NAME:-unknown Linux}." >&2
  exit 2
fi

ensure_path_entry() {
  case ":$PATH:" in
    *":$1:"*) ;;
    *) export PATH="$1:$PATH" ;;
  esac
}

ensure_local_bin() {
  mkdir -p "$HOME/.local/bin"
  ensure_path_entry "$HOME/.local/bin"
}

sudo_cmd() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

ensure_sudo_access() {
  if [[ "$(id -u)" -eq 0 ]]; then
    return
  fi

  if sudo -n true >/dev/null 2>&1; then
    return
  fi

  if [[ -t 0 ]]; then
    sudo -v
    return
  fi

  echo "This Debian/Ubuntu installer needs sudo access for apt packages, but sudo requires an interactive password prompt. Re-run it in an interactive terminal or as root." >&2
  exit 3
}

command_healthy() {
  local command_name="$1"
  shift

  command -v "$command_name" >/dev/null 2>&1 || return 1
  if [[ "$#" -gt 0 ]]; then
    "$command_name" "$@" >/dev/null 2>&1 || return 1
  fi
}

apt_install() {
  sudo_cmd apt-get install -y "$@"
}

apt_package_exists() {
  apt-cache show "$1" >/dev/null 2>&1
}

apt_install_if_available() {
  local package_name="$1"
  if apt_package_exists "$package_name"; then
    apt_install "$package_name"
    return 0
  fi
  return 1
}

download_to_local_bin() {
  local url="$1"
  local output_name="$2"
  local temp_file
  temp_file="$(mktemp)"
  curl -fsSL "$url" -o "$temp_file"
  install -m 0755 "$temp_file" "$HOME/.local/bin/$output_name"
  rm -f "$temp_file"
}

install_tar_binary_to_local_bin() {
  local url="$1"
  local binary_name="$2"
  local archive_name
  local temp_dir
  archive_name="$(mktemp)"
  temp_dir="$(mktemp -d)"

  curl -fsSL "$url" -o "$archive_name"
  tar -xzf "$archive_name" -C "$temp_dir"
  local found_binary
  found_binary="$(find "$temp_dir" -type f -name "$binary_name" | head -n 1)"
  if [[ -z "$found_binary" ]]; then
    echo "Could not find $binary_name inside $url" >&2
    rm -rf "$temp_dir" "$archive_name"
    exit 1
  fi
  install -m 0755 "$found_binary" "$HOME/.local/bin/$binary_name"
  rm -rf "$temp_dir" "$archive_name"
}

install_gh_if_needed() {
  if command_healthy gh --version; then
    echo "apt tool already healthy: gh"
    return
  fi

  echo "Installing GitHub CLI apt repository."
  sudo_cmd mkdir -p -m 755 /etc/apt/keyrings
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo_cmd tee /etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null
  sudo_cmd chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo_cmd tee /etc/apt/sources.list.d/github-cli.list >/dev/null
  sudo_cmd apt-get update
  apt_install gh
}

install_powershell_if_needed() {
  if command_healthy pwsh --version; then
    echo "apt tool already healthy: pwsh"
    return
  fi

  echo "Installing PowerShell apt repository."
  local deb_url="https://packages.microsoft.com/config/${ID}/${VERSION_ID}/packages-microsoft-prod.deb"
  local deb_file
  deb_file="$(mktemp --suffix=.deb)"
  curl -fsSL "$deb_url" -o "$deb_file"
  sudo_cmd dpkg -i "$deb_file"
  rm -f "$deb_file"
  sudo_cmd apt-get update
  apt_install powershell
}

node_major_version() {
  if ! command -v node >/dev/null 2>&1; then
    echo 0
    return
  fi

  node --version | sed -E 's/^v([0-9]+).*/\1/'
}

install_node_if_needed() {
  local major
  major="$(node_major_version)"

  if [[ "$major" -ge 22 ]] && command_healthy npm --version; then
    echo "Node.js already healthy: $(node --version)"
    return
  fi

  echo "Installing Node.js 22 from NodeSource because current Node.js major version is $major."
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo_cmd bash -
  apt_install nodejs
}

install_uv_if_needed() {
  if command_healthy uv --version && command_healthy uvx --version; then
    echo "uv already healthy"
    return
  fi

  echo "Installing uv."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ensure_path_entry "$HOME/.local/bin"
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

install_pnpm_if_needed() {
  if command_healthy pnpm --version; then
    echo "npm tool already healthy: pnpm"
    return
  fi

  echo "Installing pnpm with npm user prefix."
  npm config set prefix "$HOME/.local"
  npm install -g pnpm
  ensure_path_entry "$HOME/.local/bin"
}

yq_mikefarah_healthy() {
  command -v yq >/dev/null 2>&1 || return 1
  yq --version 2>/dev/null | grep -Eqi 'mikefarah|version v[0-9]+'
}

install_direct_tools_if_needed() {
  if ! command_healthy hadolint --version; then
    echo "Installing hadolint."
    download_to_local_bin "https://github.com/hadolint/hadolint/releases/download/v2.14.0/hadolint-Linux-x86_64" hadolint
  fi

  if ! command_healthy gitleaks version; then
    echo "Installing gitleaks."
    install_tar_binary_to_local_bin "https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz" gitleaks
  fi

  if ! command_healthy delta --version; then
    if ! apt_install_if_available git-delta; then
      echo "Installing delta."
      install_tar_binary_to_local_bin "https://github.com/dandavison/delta/releases/download/0.19.2/delta-0.19.2-x86_64-unknown-linux-gnu.tar.gz" delta
    fi
  fi

  if ! command_healthy just --version; then
    if ! apt_install_if_available just; then
      echo "Installing just."
      install_tar_binary_to_local_bin "https://github.com/casey/just/releases/download/1.51.0/just-1.51.0-x86_64-unknown-linux-musl.tar.gz" just
    fi
  fi

  if ! yq_mikefarah_healthy; then
    echo "Installing Mike Farah yq."
    download_to_local_bin "https://github.com/mikefarah/yq/releases/download/v4.53.2/yq_linux_amd64" yq
  fi
}

install_fd_bat_aliases() {
  if ! command -v fd >/dev/null 2>&1 && command -v fdfind >/dev/null 2>&1; then
    ln -sf "$(command -v fdfind)" "$HOME/.local/bin/fd"
  fi

  if ! command -v bat >/dev/null 2>&1 && command -v batcat >/dev/null 2>&1; then
    ln -sf "$(command -v batcat)" "$HOME/.local/bin/bat"
  fi
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

ensure_local_bin
ensure_sudo_access

echo "Updating apt package metadata."
sudo_cmd apt-get update

apt_install ca-certificates curl wget gnupg lsb-release apt-transport-https software-properties-common unzip tar gzip
apt_install python3 python3-pip python3-venv python3-yaml pipx
apt_install git jq fd-find fzf bat ripgrep p7zip-full shellcheck shfmt cmake ninja-build
apt_install_if_available yq || true
apt_install_if_available just || true
apt_install_if_available git-delta || true

install_fd_bat_aliases
install_node_if_needed
install_gh_if_needed
install_powershell_if_needed
install_uv_if_needed
install_pnpm_if_needed
install_direct_tools_if_needed

python3 -c "import yaml; print('PyYAML ' + yaml.__version__)"
pipx ensurepath
ensure_local_bin

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

echo "Codex prerequisite setup completed for Debian/Ubuntu Linux. Restart already-open terminals to inherit PATH changes."

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
os_name="$(uname -s)"

case "$os_name" in
  Darwin)
    exec bash "$script_dir/install-macos.sh" "$@"
    ;;
  Linux)
    exec bash "$script_dir/install-linux.sh" "$@"
    ;;
  *)
    echo "setup-codex-prerequisites supports Windows, macOS, and Debian/Ubuntu Linux only. Detected: $os_name" >&2
    exit 2
    ;;
esac

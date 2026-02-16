#!/usr/bin/env bash
set -euo pipefail

# Wrapper: portable launcher for podcast maker Python entrypoint.
# Usage:
#   make_podcast.sh /path/to/script.json /output/dir episode_name

SCRIPT_FILE="${1:?script.json required}"
OUTDIR="${2:?output dir required}"
BASENAME="${3:-episode}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

exec "${SCRIPT_DIR}/make_podcast.py" "$SCRIPT_FILE" "$OUTDIR" "$BASENAME"

#!/usr/bin/env bash
set -euo pipefail

# Wrapper: portable launcher for podcast maker Python entrypoint.
# Usage:
#   make_podcast.sh /path/to/script.json /output/dir [episode_name] [extra make_podcast.py flags...]

if [[ "$#" -lt 2 ]]; then
  echo "usage: make_podcast.sh <script.json> <outdir> [basename] [extra args...]" >&2
  exit 2
fi

SCRIPT_FILE="$1"
OUTDIR="$2"
shift 2

BASENAME="episode"
if [[ "$#" -gt 0 && "${1:-}" != --* ]]; then
  BASENAME="$1"
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

exec "${SCRIPT_DIR}/make_podcast.py" "$SCRIPT_FILE" "$OUTDIR" "$BASENAME" "$@"

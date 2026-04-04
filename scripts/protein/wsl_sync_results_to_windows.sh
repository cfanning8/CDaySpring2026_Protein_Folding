#!/usr/bin/env bash
# Copy results/protein/ from the ext4 WSL working copy to the Windows-mounted repo (IDE, Explorer).
#
# Usage (from ext4 repo, e.g. ~/CDaySpring2026_Protein_Folding-main):
#   bash scripts/protein/wsl_sync_results_to_windows.sh
#
# Destination resolution (first match wins):
#   1. WSL_WINDOWS_REPO if set
#   2. First line of .wsl_windows_repo (written by wsl_sync_minimal_to_home.sh)
#   3. Fallback path (override per machine via WSL_WINDOWS_REPO)
#
# Skip when running smoke: WSL_SKIP_RESULTS_SYNC=1 bash scripts/protein/wsl_run_smoke.sh ...

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

case "$ROOT" in
  /mnt/[a-z]/*)
    echo "Run this from your ext4 repo copy under \\\$HOME, not from /mnt/c/..." >&2
    exit 1
    ;;
esac

if [[ ! -d "$ROOT/results/protein" ]]; then
  echo "No results/protein at $ROOT — nothing to sync." >&2
  exit 1
fi

DEST=""
if [[ -n "${WSL_WINDOWS_REPO:-}" ]]; then
  DEST="$WSL_WINDOWS_REPO"
elif [[ -f "$ROOT/.wsl_windows_repo" ]]; then
  DEST="$(head -1 "$ROOT/.wsl_windows_repo" | tr -d '\r')"
fi
if [[ -z "$DEST" ]]; then
  DEST="/mnt/c/Users/auror/dev/2026/Spring_2026/C_Day/CDaySpring2026_Protein_Folding-main/CDaySpring2026_Protein_Folding-main"
fi

if [[ -z "$DEST" ]] || [[ "$DEST" != /mnt/* ]]; then
  echo "Invalid Windows repo path: ${DEST:-"(empty)"}" >&2
  echo "Set WSL_WINDOWS_REPO to your /mnt/c/.../repo root." >&2
  exit 1
fi

parent="$(dirname "$DEST")"
if [[ ! -d "$parent" ]]; then
  echo "Parent of destination missing (is /mnt/c mounted?): $parent" >&2
  exit 1
fi

mkdir -p "$DEST/results/protein"
echo "Syncing results/protein/ -> $DEST/results/protein/" >&2
rsync -a "$ROOT/results/protein/" "$DEST/results/protein/"
echo "Done." >&2

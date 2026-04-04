#!/usr/bin/env bash
# Copy ColabFold output from WSL ext4 (cf_smoke_corpus50) into repo results/protein/output/colabfold_smoke.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${COLABFOLD_SMOKE_SRC:-$HOME/cf_smoke_corpus50}"
DST="$REPO/results/protein/output/colabfold_smoke"
if [[ ! -d "$SRC" ]]; then
  echo "missing source: $SRC" >&2
  exit 1
fi
mkdir -p "$DST"
rsync -a --delete "$SRC/" "$DST/"
echo "synced $SRC -> $DST"

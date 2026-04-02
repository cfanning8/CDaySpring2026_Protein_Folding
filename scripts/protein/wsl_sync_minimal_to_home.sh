#!/usr/bin/env bash
# Copy only what is needed for protein end-to-end smoke + ColabFold setup from a Windows-mounted
# repo to $HOME (ext4). Does NOT clone the full repository (no epi track, notebooks, TEMPORARY_CODE,
# .venv, large results, etc.).
#
# Includes:
#   src/__init__.py  src/protein/  src/topology/
#   scripts/protein/
#   requirements-base.txt  requirements-protein.txt
#   data/protein/mmcif/     (mmCIF inputs; does not delete extra files you added)
#   empty dirs: data/processed/protein  results/protein/{output,figures,tables/smoke}
#
# Usage (inside WSL, from the repo on /mnt/c):
#   bash scripts/protein/wsl_sync_minimal_to_home.sh
#
# Then:
#   cd "${WSL_HOME_REPO:-$HOME/2_Protein_Folding}"
#   bash scripts/protein/wsl_setup_protein_venv.sh
#   bash scripts/protein/wsl_install_localcolabfold.sh
#   bash scripts/protein/wsl_run_smoke.sh

set -euo pipefail

# Resolve repo root from this file (works when invoked as `bash /mnt/c/.../wsl_sync_minimal_to_home.sh`).
TOP="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ ! -f "$TOP/requirements-base.txt" ]] || [[ ! -d "$TOP/src/protein" ]]; then
  echo "Expected C-DAY protein repo root at $TOP (missing requirements-base.txt or src/protein)." >&2
  exit 1
fi

case "$TOP" in
  /mnt/[a-z]/*) ;;
  *)
    echo "This script copies FROM a repo on a Windows drive (/mnt/c/...) TO \$HOME (ext4)." >&2
    echo "Current repo: $TOP" >&2
    echo "If you are already on ext4, run setup here (no sync needed):" >&2
    echo "  bash scripts/protein/wsl_setup_protein_venv.sh" >&2
    exit 0
    ;;
esac

NAME="$(basename "$TOP")"
DEST="${WSL_HOME_REPO:-$HOME/$NAME}"

echo "Minimal sync: $TOP -> $DEST" >&2

mkdir -p "$DEST"
mkdir -p "$DEST/scripts"
mkdir -p "$DEST/data/processed/protein"
mkdir -p "$DEST/results/protein/output" "$DEST/results/protein/figures" "$DEST/results/protein/tables/smoke"
mkdir -p "$DEST/data/protein/mmcif"

# Source tree (protein + shared topology helpers; excludes src/models, epi loaders, etc.)
mkdir -p "$DEST/src"
install -m0644 "$TOP/src/__init__.py" "$DEST/src/__init__.py"
rsync -a --delete "$TOP/src/protein/" "$DEST/src/protein/"
rsync -a --delete "$TOP/src/topology/" "$DEST/src/topology/"

# Entry points and WSL helpers
rsync -a --delete "$TOP/scripts/protein/" "$DEST/scripts/protein/"

for f in requirements-base.txt requirements-protein.txt; do
  install -m0644 "$TOP/$f" "$DEST/$f"
done

# mmCIF inputs (append/update; do not --delete user-added structures)
if [[ -d "$TOP/data/protein/mmcif" ]]; then
  rsync -a "$TOP/data/protein/mmcif/" "$DEST/data/protein/mmcif/"
fi

echo "" >&2
echo "Done. Next (from ext4 tree):" >&2
echo "  cd $DEST" >&2
echo "  bash scripts/protein/wsl_setup_protein_venv.sh" >&2
echo "  bash scripts/protein/wsl_install_localcolabfold.sh" >&2
echo "  bash scripts/protein/wsl_run_smoke.sh" >&2

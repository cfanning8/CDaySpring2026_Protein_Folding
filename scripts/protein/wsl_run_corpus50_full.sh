#!/usr/bin/env bash
# Full corpus-50 pipeline: Layers 1–2 + ColabFold (LocalColabFold on PATH). Run from WSL2 at repo root or via:
#   bash scripts/protein/wsl_run_corpus50_full.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
if [[ ! -d .venv-wsl-full ]]; then
  echo "Create venv first: python3 -m venv .venv-wsl-full && . .venv-wsl-full/bin/activate && pip install -r requirements-protein.txt" >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv-wsl-full/bin/activate
export LOCALCOLABFOLD_ROOT="${LOCALCOLABFOLD_ROOT:-$HOME/localcolabfold}"
# Prepend ColabFold after venv so `python` stays the project venv (Pixi env may also ship `python`).
export PATH="$PATH:$LOCALCOLABFOLD_ROOT/.pixi/envs/default/bin"
export TF_FORCE_UNIFIED_MEMORY="${TF_FORCE_UNIFIED_MEMORY:-1}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-4.0}"
# JAX + huge I/O on /mnt/c/ (9p) can SIGSEGV; keep ColabFold output on ext4.
CF_OUT="${COLABFOLD_SMOKE_OUT:-$HOME/cf_smoke_corpus50}"
mkdir -p "$CF_OUT"
"${VIRTUAL_ENV}/bin/python" -u scripts/protein/run_smoke_pipeline.py \
  --corpus-dir data/protein/mmcif_corpus50 \
  --graph-mode ca_legacy \
  --skip-structure-figures \
  --colabfold-out-dir "$CF_OUT" \
  --colabfold-sequential \
  --colabfold-max-structures 50 \
  --colabfold-extra-args "--num-recycle 1 --num-models 1"
MIRROR="$REPO/results/protein/output/colabfold_smoke"
export COLABFOLD_SMOKE_SRC="$CF_OUT"
bash "$REPO/scripts/protein/wsl_sync_cf_smoke_to_repo.sh"
echo "ColabFold artifacts: $CF_OUT (synced to $MIRROR). On Windows, rerun: python scripts/protein/run_smoke_pipeline.py --corpus-dir data/protein/mmcif_corpus50 --graph-mode ca_legacy --skip-structure-figures --no-colabfold-smoke"

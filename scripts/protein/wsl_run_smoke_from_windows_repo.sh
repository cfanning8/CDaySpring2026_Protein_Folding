#!/usr/bin/env bash
# Run smoke using WSL venv at $HOME/CDaySpring2026_Protein_Folding-main/.venv-wsl-gpu against this repo
# (often checked out on /mnt/c/...). LocalColabFold on PATH.
set -euo pipefail
WIN_MOUNT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_HOME="${WSL_PROTEIN_VENV:-$HOME/CDaySpring2026_Protein_Folding-main/.venv-wsl-gpu}"
LCF="${LOCALCOLABFOLD_ROOT:-$HOME/localcolabfold}"
if [[ -x "${LCF}/.pixi/envs/default/bin/colabfold_batch" ]]; then
  export PATH="${LCF}/.pixi/envs/default/bin:${PATH}"
fi
# shellcheck source=/dev/null
source "${VENV_HOME}/bin/activate"
cd "$WIN_MOUNT_REPO"
export PYTHONPATH="$WIN_MOUNT_REPO"
MAX="${1:-1}"
exec python -u scripts/protein/run_smoke_pipeline.py --skip-structure-figures --colabfold-max-structures "${MAX}"

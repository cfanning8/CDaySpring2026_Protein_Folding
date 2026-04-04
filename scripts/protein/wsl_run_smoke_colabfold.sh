#!/usr/bin/env bash
# Run protein smoke with ColabFold on PATH (WSL2). Uses repo .venv-wsl-gpu when present.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LCF="${LOCALCOLABFOLD_ROOT:-$HOME/localcolabfold}"
if [[ -x "${LCF}/.pixi/envs/default/bin/colabfold_batch" ]]; then
  export PATH="${LCF}/.pixi/envs/default/bin:${PATH}"
fi

if [[ -f "${ROOT}/.venv-wsl-gpu/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv-wsl-gpu/bin/activate"
fi

MAX_STRUCTS="${1:-1}"
exec python -u scripts/protein/run_smoke_pipeline.py \
  --skip-structure-figures \
  --colabfold-max-structures "${MAX_STRUCTS}"

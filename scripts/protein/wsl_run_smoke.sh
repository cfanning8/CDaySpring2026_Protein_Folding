#!/usr/bin/env bash
# Run full protein smoke inside WSL2 with GPU validation (ColabFold path).
# Prereq: bash scripts/protein/wsl_setup_protein_venv.sh && bash scripts/protein/wsl_install_localcolabfold.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source "${ROOT}/scripts/protein/wsl_lib.sh"
wsl_refuse_windows_mount_repo "$ROOT"

LCF="${LOCALCOLABFOLD_ROOT:-$HOME/localcolabfold}"
if [[ -x "${LCF}/.pixi/envs/default/bin/colabfold_batch" ]]; then
  export PATH="${LCF}/.pixi/envs/default/bin:${PATH}"
fi

export PYTHONPATH="$ROOT"
# shellcheck source=/dev/null
source "${ROOT}/.venv-wsl-gpu/bin/activate"
python -u scripts/protein/run_smoke_pipeline.py "$@"
rc=$?

if [[ "$rc" -eq 0 ]] && [[ "${WSL_SKIP_RESULTS_SYNC:-0}" != "1" ]] && [[ -f "${ROOT}/scripts/protein/wsl_sync_results_to_windows.sh" ]]; then
  bash "${ROOT}/scripts/protein/wsl_sync_results_to_windows.sh" || rc=$?
fi
exit "$rc"

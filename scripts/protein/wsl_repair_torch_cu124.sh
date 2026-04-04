#!/usr/bin/env bash
# Fix a venv that got torch built for CUDA 13 (cu130) when WSL reports CUDA 12.x — cuda.is_available() stays False.
# Run inside WSL from repo root:
#   bash scripts/protein/wsl_repair_torch_cu124.sh
#
# Default: **ColabFold publication path** — does **not** install legacy OpenFold3 deps.
# To also reinstall legacy stack: ENABLE_LEGACY_OPENFOLD3_REPAIR=1 bash scripts/protein/wsl_repair_torch_cu124.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source "${ROOT}/scripts/protein/wsl_lib.sh"
wsl_refuse_windows_mount_repo "$ROOT"

# shellcheck source=/dev/null
source "${ROOT}/.venv-wsl-gpu/bin/activate"

nvidia-smi
pip uninstall -y torch torchvision 2>/dev/null || true
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

export DS_BUILD_OPS=0
export DS_BUILD_AIO=0
if [[ "${ENABLE_LEGACY_OPENFOLD3_REPAIR:-}" == "1" ]]; then
  pip install -r requirements-openfold3-smoke.txt
fi

export PYTHONPATH="$ROOT"
python3 - <<'PY'
import torch

print("CUDA OK:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "Repair done (torch cu124). ColabFold: use wsl_install_localcolabfold.sh"
echo "Legacy OpenFold3 repair only if ENABLE_LEGACY_OPENFOLD3_REPAIR=1 was set."

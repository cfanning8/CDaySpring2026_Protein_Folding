#!/usr/bin/env bash
# Legacy / embargoed: non-ColabFold venv (see legacy/README.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source "${ROOT}/scripts/protein/wsl_lib.sh"
wsl_refuse_windows_mount_repo "$ROOT"

if [[ ! -f "requirements-base.txt" ]]; then
  echo "Run this script from the repository root (inside WSL)." >&2
  exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
  echo "nvidia-smi not found. Install NVIDIA Windows driver + WSL CUDA support, then retry." >&2
  exit 1
fi
nvidia-smi

VENV="${ROOT}/.venv-wsl-gpu"
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

pip install -U pip wheel
pip install pandas numpy gudhi scikit-learn matplotlib tqdm
pip uninstall -y torch torchvision 2>/dev/null || true
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-protein.txt

export DS_BUILD_OPS=0
export DS_BUILD_AIO=0
pip install -r requirements-openfold3-smoke.txt

export PYTHONPATH="$ROOT"
python3 - <<'PY'
import torch

from src.protein.cuda_guard import ensure_cuda_for_openfold

ensure_cuda_for_openfold(purpose="legacy wsl_setup_openfold3_gpu.sh post-install")
print("CUDA OK:", torch.cuda.get_device_name(0))
PY

echo ""
echo "Venv:  $VENV"
echo "Legacy stack only — ColabFold path: wsl_install_localcolabfold.sh"

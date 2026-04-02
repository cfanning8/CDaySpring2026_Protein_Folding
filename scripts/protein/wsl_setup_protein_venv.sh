#!/usr/bin/env bash
# Project venv for protein Layers 1–2 (topology, tables, py3Dmol) + PyTorch CUDA for dev checks.
# Layer 3 structure prediction uses ColabFold in a separate LocalColabFold install — see wsl_install_localcolabfold.sh
#
# Usage (from repo root on ext4 inside WSL):
#   bash scripts/protein/wsl_setup_protein_venv.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source "${ROOT}/scripts/protein/wsl_lib.sh"
wsl_refuse_windows_mount_repo "$ROOT"

if [[ ! -f "requirements-base.txt" ]]; then
  echo "Run this script from the repository root (inside WSL)." >&2
  exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
  echo "nvidia-smi not found (optional for CPU-only topology; GPU needed for ColabFold smoke)." >&2
else
  nvidia-smi
fi

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

export PYTHONPATH="$ROOT"
python3 - <<'PY'
import torch

assert torch.cuda.is_available(), "torch CUDA not available — check driver + cu124 wheel"
print("CUDA OK:", torch.cuda.get_device_name(0))
PY

echo ""
echo "Venv:  $VENV"
echo "Next:  bash scripts/protein/wsl_install_localcolabfold.sh   # ColabFold (separate from this venv)"
echo "Then:  bash scripts/protein/wsl_run_smoke.sh --skip-structure-figures"

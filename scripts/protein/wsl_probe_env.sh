#!/usr/bin/env bash
# Inspect WSL + GPU + venv state (no writes). Run inside WSL:
#   bash scripts/protein/wsl_probe_env.sh
# From Windows:
#   wsl -e bash /mnt/c/Users/<you>/.../scripts/protein/wsl_probe_env.sh
set -euo pipefail
echo "=== uname ==="
uname -a
echo
echo "=== /etc/os-release (head) ==="
head -5 /etc/os-release 2>/dev/null || true
echo
echo "=== df home vs /mnt/c ==="
df -hT "$HOME" 2>/dev/null | head -2
df -hT /mnt/c 2>/dev/null | head -2 || echo "(no /mnt/c)"
echo
# Override with: WSL_PROBE_REPO_WIN=/mnt/c/Users/you/.../repo
REPO_WIN="${WSL_PROBE_REPO_WIN:-/mnt/c/Users/auror/dev/2026/Spring_2026/C_Day/CDaySpring2026_Protein_Folding-main/CDaySpring2026_Protein_Folding-main}"
REPO_HOME="${WSL_HOME_REPO:-$HOME/CDaySpring2026_Protein_Folding-main}"
echo "=== paths ==="
for p in "$REPO_WIN" "$REPO_HOME"; do
  if [[ -d "$p" ]]; then
    echo "EXISTS $p"
    if [[ -d "$p/.venv-wsl-gpu" ]]; then
      echo "  .venv-wsl-gpu: exists (size skipped; du on /mnt/c is slow)"
      # shellcheck source=/dev/null
      if [[ -f "$p/.venv-wsl-gpu/bin/activate" ]]; then
        # shellcheck source=/dev/null
        source "$p/.venv-wsl-gpu/bin/activate"
        echo "  python: $(command -v python)"
        python -c "import sys; print('  py_version', sys.version.split()[0])" 2>/dev/null || true
        python -c "import torch; print('  torch', torch.__version__, 'cuda_avail', torch.cuda.is_available())" 2>/dev/null || echo "  torch: not importable"
        pip show openfold3 2>/dev/null | head -2 || echo "  openfold3: not installed (legacy path only; see scripts/protein/legacy/)"
        command -v run_openfold >/dev/null && echo "  run_openfold (legacy): $(command -v run_openfold)" || echo "  run_openfold: not on PATH"
      fi
    else
      echo "  (no .venv-wsl-gpu)"
    fi
  else
    echo "MISSING $p"
  fi
done
echo
LCF="${LOCALCOLABFOLD_ROOT:-$HOME/localcolabfold}"
echo "=== LocalColabFold ==="
if [[ -x "${LCF}/.pixi/envs/default/bin/colabfold_batch" ]]; then
  echo "colabfold_batch: ${LCF}/.pixi/envs/default/bin/colabfold_batch"
else
  echo "colabfold_batch: not found (set LOCALCOLABFOLD_ROOT or run wsl_install_localcolabfold.sh)"
fi
echo
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 || echo "nvidia-smi failed"
echo
echo "=== nvcc (CUDA toolkit; LocalColabFold uses JAX — check version vs their README) ==="
if command -v nvcc &>/dev/null; then
  nvcc --version 2>&1 | head -5
else
  echo "(nvcc not on PATH — install CUDA toolkit in WSL if colabfold_batch fails with CUDA/JAX errors)"
fi
echo
echo "=== git (Windows worktree) ==="
if [[ -d "$REPO_WIN/.git" ]]; then
  git -C "$REPO_WIN" rev-parse --show-toplevel 2>&1
  git -C "$REPO_WIN" status -sb 2>&1 | head -1
else
  echo "(no .git at $REPO_WIN)"
fi

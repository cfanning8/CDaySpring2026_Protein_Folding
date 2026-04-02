#!/usr/bin/env bash
# Install LocalColabFold (ColabFold + AlphaFold2-class weights on first run) under $HOME (ext4).
# See: https://github.com/YoshitakaMo/localcolabfold
# Upstream ColabFold: https://github.com/sokrypton/colabfold
#
# Does not use the project .venv — ColabFold ships its own JAX stack via pixi.
# After install, colabfold_batch lives at:
#   ${LOCALCOLABFOLD_ROOT:-$HOME/localcolabfold}/.pixi/envs/default/bin/colabfold_batch
#
# Usage (WSL2, from any directory):
#   bash scripts/protein/wsl_install_localcolabfold.sh
#
# Prereqs: git, curl, wget (Ubuntu: sudo apt install -y curl git wget).
# CUDA: LocalColabFold docs say to verify the *toolkit* with `nvcc --version` (11.8+ in older docs; GPU JAX
# stacks often want 12.1+ / cudnn — see their README), not only `nvidia-smi` (driver).
# Install only on ext4 ($HOME), not /mnt/c — avoids slow I/O and symlink/case issues. If you ever clone on a
# Windows path and hit symlink errors, LocalColabFold suggests enabling NTFS case sensitivity from
# Windows PowerShell: fsutil file SetCaseSensitiveInfo <path> enable
# Alternative entry: install_colabbatch_linux.sh in the same repo (see upstream README).

set -euo pipefail

wsl_refuse_mnt_c_dest() {
  case "${1:-}" in
    /mnt/[a-z]/*)
      echo "Refusing LOCALCOLABFOLD_ROOT on Windows mount (slow / symlink issues): $1" >&2
      echo "Unset LOCALCOLABFOLD_ROOT to use \$HOME/localcolabfold on ext4." >&2
      exit 1
      ;;
  esac
}

DEST="${LOCALCOLABFOLD_ROOT:-$HOME/localcolabfold}"
wsl_refuse_mnt_c_dest "$DEST"

if ! command -v git &>/dev/null; then
  echo "Install git first (e.g. sudo apt install -y git)." >&2
  exit 1
fi

if [[ ! -d "$DEST/.git" ]]; then
  echo "Cloning LocalColabFold -> $DEST" >&2
  git clone https://github.com/YoshitakaMo/localcolabfold.git "$DEST"
else
  echo "Using existing clone: $DEST" >&2
fi

cd "$DEST"

if ! command -v pixi &>/dev/null; then
  echo "Installing pixi..." >&2
  curl -fsSL https://pixi.sh/install.sh | sh
  # shellcheck disable=SC1091
  [[ -f "${HOME}/.pixi/bin/pixi" ]] && export PATH="${HOME}/.pixi/bin:${PATH}"
fi
command -v pixi &>/dev/null || {
  echo "pixi not on PATH after install; add ~/.pixi/bin to PATH and re-run." >&2
  exit 1
}

echo "Running pixi install && pixi run setup (network + time; model weights on first colabfold_batch run)..." >&2
pixi install && pixi run setup

BIN="$DEST/.pixi/envs/default/bin/colabfold_batch"
if [[ ! -x "$BIN" ]]; then
  echo "Expected $BIN missing after setup." >&2
  exit 1
fi

echo "" >&2
echo "Done. Export for smoke runs:" >&2
echo "  export LOCALCOLABFOLD_ROOT=$DEST" >&2
echo "  export PATH=\"$DEST/.pixi/envs/default/bin:\$PATH\"" >&2
echo "Then: bash scripts/protein/wsl_run_smoke.sh --skip-structure-figures" >&2

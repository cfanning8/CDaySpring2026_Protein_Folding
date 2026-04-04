#!/usr/bin/env bash
# If the repo is on /mnt/c/... (Windows disk), git-clone the *entire* repo to $HOME/<basename> (ext4).
# Prefer scripts/protein/wsl_sync_minimal_to_home.sh when you only need the protein + topology track
# (smaller, faster copy: src/protein, src/topology, scripts/protein, requirements pins, mmcif).
#
# You still use a normal .venv inside that tree — nothing is installed "globally".
#
# Usage (inside WSL, from anywhere under this git repo):
#   bash scripts/protein/wsl_git_clone_to_home.sh
#
# Then:
#   cd "$HOME/$(basename "$(git rev-parse --show-toplevel)")"
#   bash scripts/protein/wsl_setup_protein_venv.sh && bash scripts/protein/wsl_install_localcolabfold.sh

set -euo pipefail

TOP="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$TOP" ]]; then
  echo "Not inside a git repository." >&2
  exit 1
fi

case "$TOP" in
  /mnt/[a-z]/*) ;; # Windows mount — proceed
  *)
    echo "Repository is already on the Linux filesystem:" >&2
    echo "  $TOP" >&2
    echo "No clone needed. Run setup from here:" >&2
    echo "  bash scripts/protein/wsl_setup_protein_venv.sh" >&2
    exit 0
    ;;
esac

NAME="$(basename "$TOP")"
DEST="${WSL_HOME_REPO:-$HOME/$NAME}"

if [[ -d "$DEST/.git" ]]; then
  echo "Clone already exists: $DEST" >&2
  echo "  cd $DEST && git pull && bash scripts/protein/wsl_setup_protein_venv.sh" >&2
  exit 0
fi

if [[ -e "$DEST" ]]; then
  echo "Path exists but is not a git repo: $DEST" >&2
  echo "Remove or rename it, or set WSL_HOME_REPO to a different directory." >&2
  exit 1
fi

echo "Cloning (local) to ext4 worktree: $DEST" >&2
git clone "$TOP" "$DEST"

echo "" >&2
echo "Next commands:" >&2
echo "  cd $DEST" >&2
echo "  bash scripts/protein/wsl_setup_protein_venv.sh && bash scripts/protein/wsl_install_localcolabfold.sh" >&2

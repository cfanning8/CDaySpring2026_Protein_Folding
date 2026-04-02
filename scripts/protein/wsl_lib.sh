#!/usr/bin/env bash
# Shared helpers for WSL protein scripts (source from repo root only).
# shellcheck shell=bash

# Exit with instructions if the repo lives on a Windows drive mount (/mnt/X/...).
# Pip + CUDA wheels perform poorly there; use scripts/protein/wsl_git_clone_to_home.sh once.
wsl_refuse_windows_mount_repo() {
  local root="$1"
  case "$root" in
    /mnt/[a-z]/*)
      echo "This repository is under a Windows drive mount (slow I/O for large pip installs)." >&2
      echo "Run once from WSL, inside the Windows-mounted repo:" >&2
      echo "  bash scripts/protein/wsl_sync_minimal_to_home.sh   # recommended (src/scripts/data only)" >&2
      echo "  bash scripts/protein/wsl_setup_protein_venv.sh && bash scripts/protein/wsl_install_localcolabfold.sh" >&2
      echo "  # optional full mirror: bash scripts/protein/wsl_git_clone_to_home.sh" >&2
      echo "Then continue from the path it prints (under \\\$HOME, ext4)." >&2
      exit 1
      ;;
  esac
}

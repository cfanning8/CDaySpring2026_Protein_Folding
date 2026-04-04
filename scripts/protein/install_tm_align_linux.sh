#!/usr/bin/env bash
# Fetch and compile TM-align (Yang & Zhang) for CASP-style TM-scores. Run in WSL2/Linux; needs gfortran.
set -euo pipefail
DEST="${1:-$HOME/localbin}"
mkdir -p "$DEST"
WORKDIR="$(mktemp -d)"
cd "$WORKDIR"
curl -fsSL -o TMalign.f "https://zhanggroup.org/TM-align/TMalign.f"
gfortran -O3 -ffast-math TMalign.f -o TMalign
cp -f TMalign "$DEST/TMalign"
chmod +x "$DEST/TMalign"
rm -rf "$WORKDIR"
echo "Installed: $DEST/TMalign"
echo "Add to PATH or: export TMALIGN_BIN=$DEST/TMalign"

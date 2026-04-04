#!/usr/bin/env bash
# Deprecated stub — real installer: scripts/protein/legacy/wsl_setup_openfold3_gpu.sh
set -euo pipefail
echo "DEPRECATED: ColabFold-only publication path uses wsl_install_localcolabfold.sh" >&2
echo "Legacy non-ColabFold installer:" >&2
echo "  bash scripts/protein/legacy/wsl_setup_openfold3_gpu.sh" >&2
exec bash "$(dirname "${BASH_SOURCE[0]}")/legacy/wsl_setup_openfold3_gpu.sh" "$@"

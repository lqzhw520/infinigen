#!/bin/bash
# Thin wrapper around the PhysNAP experiment launcher.

set -euo pipefail

PROJECT_ROOT="/mnt/afs2/zhuhaowu/infinigen"
CONDA_SH="/root/anaconda3/etc/profile.d/conda.sh"

if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: Conda init script not found at $CONDA_SH"
    exit 1
fi

source "$CONDA_SH"
conda activate infinigen
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$PROJECT_ROOT"
exec python scripts/physnap_experiment_launcher.py "$@"

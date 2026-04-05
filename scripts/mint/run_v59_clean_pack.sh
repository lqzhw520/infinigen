#!/bin/bash
# V59: Clean dataset repack — pack V58+P4 NPZ files into clean LeRobot dataset
# Archive timestamp: 2026-04-05T14:35+08:00
# Source: artifacts/v58_physics_legal_rollouts/ (227 files) + p4_physics_legal_rollouts/ (13 files)
# Output: experiments/mint/mint_drawer_v1/dataset/
# This is a CLEAN pack — old dataset/ was archived before this run.

set -e
cd /mnt/afs2/zhuhaowu/infinigen

echo "[V59 pack] $(date '+%Y-%m-%d %H:%M:%S') Starting clean dataset pack"
echo "[V59 pack] GPU memory: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"

# Activate conda
source /root/anaconda3/etc/profile.d/conda.sh
conda activate infinigen

# Run the pack
python scripts/mint/run_v58_lerobot_pack.py 2>&1 | tee "experiments/mint/mint_drawer_v1/outputs/v59_lerobot_pack.log"

echo "[V59 pack] $(date '+%Y-%m-%d %H:%M:%S') Pack completed"

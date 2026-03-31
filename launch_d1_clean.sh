#!/bin/bash
# Clean launch script for D1

set -e

cd /mnt/afs2/zhuhaowu/infinigen

# Clean up
rm -rf experiments/mint/mint_drawer_v1/outputs_d1_seed_002* 2>/dev/null || true
rm -rf experiments/mint/mint_drawer_v1/dataset_d1_seed_002* 2>/dev/null || true

# Activate conda
source /root/anaconda3/etc/profile.d/conda.sh
conda activate mint

# Update controller
python -c "
import sys
sys.path.insert(0, 'scripts/mint')
from mint_common import load_state, save_state, release_controller_lease, acquire_controller_lease, read_controller_lease
import os
current = read_controller_lease()
if current.get('controller_id'):
    release_controller_lease(current['controller_id'], current.get('run_id', ''))
state = load_state()
state['controller'] = {'controller_id': 'd1_clean', 'run_id': 'd1_clean_main', 'heartbeat_at': '2026-03-30T16:00:00+08:00', 'owner': 'manual_fix'}
save_state(state)
success, lease = acquire_controller_lease('d1_clean', 'd1_clean_main', owner='manual_fix', pid=os.getpid())
print('Controller ready:', success)
"

# Launch
cd /mnt/afs2/zhuhaowu/infinigen
exec env \
    MINT_CONTROLLER_ID=d1_clean \
    MINT_RUN_ID=d1_clean_main \
    PYTHONUNBUFFERED=1 \
    /root/anaconda3/envs/mint/bin/python scripts/mint/run_d1_single_rollout_overfit.py

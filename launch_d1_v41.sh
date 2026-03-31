#!/bin/bash
# Launch D1 with proper controller setup

cd /mnt/afs2/zhuhaowu/infinigen

# Clean up stale outputs
rm -rf experiments/mint/mint_drawer_v1/outputs_d1_seed_002_episode_01_multi_episode_overfit* 2>/dev/null
rm -rf experiments/mint/mint_drawer_v1/dataset_d1_seed_002_episode_01_multi_episode_overfit* 2>/dev/null

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
state['controller'] = {'controller_id': 'd1_v41', 'run_id': 'd1_v41_main', 'heartbeat_at': '2026-03-30T15:05:00+08:00', 'owner': 'manual_fix'}
save_state(state)
success, lease = acquire_controller_lease('d1_v41', 'd1_v41_main', owner='manual_fix', pid=os.getpid())
print('Controller ready:', success)
"

# Launch D1
export MINT_CONTROLLER_ID=d1_v41
export MINT_RUN_ID=d1_v41_main
export PYTHONUNBUFFERED=1

nohup python scripts/mint/run_d1_single_rollout_overfit.py > experiments/mint/mint_drawer_v1/artifacts/d1_v41_run.log 2>&1 &
echo "D1 v41 launched, PID=$!"

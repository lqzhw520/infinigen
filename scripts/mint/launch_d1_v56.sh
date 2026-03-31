#!/bin/bash
# Launch script for D1 v56 with fixes
set -e
cd /mnt/afs2/zhuhaowu/infinigen
# Activate conda
source /root/anaconda3/etc/profile.d/conda.sh
conda activate mint
# Update controller
python3 -c "
import sys
sys.path.insert(0, 'scripts/mint')
from mint_common import load_state, save_state
from datetime import datetime

state = load_state()
state['controller'] = {
    'controller_id': 'd1_v56',
    'run_id': 'd1_v56_main',
    'heartbeat_at': datetime.now().isoformat(),
    'owner': 'manual_fix'
}
save_state(state)
print('Controller updated to d1_v56')
"
# Launch D1
exec env \
    MINT_CONTROLLER_ID=d1_v56 \
    MINT_RUN_ID=d1_v56_main \
    PYTHONUNBUFFERED=1 \
    /root/anaconda3/envs/mint/bin/python scripts/mint/run_d1_single_rollout_overfit.py
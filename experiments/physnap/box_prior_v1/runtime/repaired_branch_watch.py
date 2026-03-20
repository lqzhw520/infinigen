import json
import subprocess
import sys
import time
from pathlib import Path

repo = Path('/mnt/afs2/zhuhaowu/infinigen')
status_cmd = [str(repo / 'scripts' / 'train_physnap_infinigen.sh'), 'status']
launch_cmd = [str(repo / 'scripts' / 'train_physnap_infinigen.sh'), 'finetune_k10_fixed', '--background']

while True:
    payload = json.loads(subprocess.check_output(status_cmd, text=True))
    active = payload.get('active_train_job')
    experiments = payload.get('experiments', {})
    scratch = experiments.get('infinigen_k10_fixed', {})
    finetune = experiments.get('finetune_k10_fixed', {})
    if active:
        time.sleep(120)
        continue
    scratch_status = scratch.get('status')
    if scratch_status in {'failed', 'stopped'}:
        print(f'scratch status={scratch_status}; not launching finetune', flush=True)
        sys.exit(1)
    if scratch_status != 'completed':
        time.sleep(120)
        continue
    if finetune.get('status') in {'running', 'completed'}:
        print(f"finetune already {finetune.get('status')}; exiting", flush=True)
        sys.exit(0)
    launch_out = subprocess.check_output(launch_cmd, text=True)
    print('launched finetune_k10_fixed', flush=True)
    print(launch_out, flush=True)
    sys.exit(0)

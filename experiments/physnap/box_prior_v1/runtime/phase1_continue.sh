#!/bin/bash
set -euo pipefail
TARGET_PID=2438227
while kill -0 "$TARGET_PID" 2>/dev/null; do
  sleep 30
done
cd /mnt/afs2/zhuhaowu/infinigen
export INFINIGEN_CAMPAIGN_DIR=/mnt/afs2/zhuhaowu/infinigen/experiments/physnap/box_prior_v1
exec /root/anaconda3/envs/infinigen/bin/python scripts/physnap_auto_review_loop.py --run-to-completion --poll-seconds 120 >> experiments/physnap/box_prior_v1/runtime/phase1_auto_loop.log 2>&1

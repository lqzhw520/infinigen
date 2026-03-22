#!/usr/bin/env bash
# Sequential export → guided → eval for Phase 2 cheap ablations (sandbox only).
set -uo pipefail
ROOT="/mnt/afs2/zhuhaowu/infinigen"
CAMPAIGN="$ROOT/experiments/physnap/_overnight_sandbox/campaign_phase2_ablations"
RUN_DIR="$ROOT/experiments/physnap/_overnight_sandbox/runs/phase2_ablations_20260320"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/output.log") 2>&1

source /root/anaconda3/etc/profile.d/conda.sh
conda activate infinigen
cd "$ROOT"

# Note: do not name this GROUPS — bash reserves GROUPS (array of process group IDs).
ABLATION_GROUPS=(
  ablation_best_single_view
  ablation_fixed_state_multiview
  ablation_pts_500
  ablation_pts_1000
  ablation_pts_2000
  ablation_pts_5000
)

echo "=== Phase 0: exports ($(date -Iseconds)) ==="
set -e
for g in "${ABLATION_GROUPS[@]}"; do
  echo "--- EXPORT $g ---"
  python scripts/build_conditioning_v2_dataset.py --campaign-dir "$CAMPAIGN" export --group-id "$g"
done
set +e

echo "=== Phase 1: guided runs ($(date -Iseconds)) ==="
FAILS=0
for g in "${ABLATION_GROUPS[@]}"; do
  echo "--- GUIDED $g ---"
  if ! python scripts/run_conditioning_v2_group.py --campaign-dir "$CAMPAIGN" --group-id "$g" --force; then
    echo "FAILED guided: $g" | tee -a "$RUN_DIR/failures.txt"
    FAILS=$((FAILS + 1))
  fi
done

echo "=== Phase 2: evaluation ($(date -Iseconds)) ==="
python "$ROOT/experiments/physnap/_overnight_sandbox/scripts/evaluate_overnight_ablations.py" --campaign-dir "$CAMPAIGN" || true

echo "=== Done failures=$FAILS ($(date -Iseconds)) ==="

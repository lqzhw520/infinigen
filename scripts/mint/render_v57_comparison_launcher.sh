#!/bin/bash
# Launch D1/V57 comparison video rendering on A800
#
# Usage:
#   bash scripts/mint/render_v57_comparison_launcher.sh [mode]
#
# Modes:
#   preview    - Seed 2: pretrained + finetuned + random (fast, ~15min)
#   full_eval  - Seeds 11-15: all 3 policies (slow, ~1h)
#   quick      - Seed 2: pretrained only (fastest, ~5min)
#
# Output: experiments/mint/mint_drawer_v1/videos/

MODE=${1:-preview}
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/experiments/mint/mint_drawer_v1/videos"
FINETUNED_PATH="$PROJECT_ROOT/experiments/mint/mint_drawer_v1/outputs_d1_seed_002_episode_01_multi_episode_overfit_1774878439_c895350f/checkpoints/003000/pretrained_model"
LOG_DIR="$PROJECT_ROOT/experiments/mint/mint_drawer_v1/runtime"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/render_v57_${MODE}_${TS}.log"

# Activate conda env — use `mint` env (has torch + lerobot + lerobot_policy_mint)
# NOT `infinigen` (that env is for Blender/Infinigen asset generation)
source /root/anaconda3/etc/profile.d/conda.sh && conda activate mint

echo "[$(date)] Launching D1/V57 video render: mode=$MODE"
echo "  Working dir: $SCRIPT_DIR"
echo "  Finetuned path: $FINETUNED_PATH"
echo "  Output dir: $OUTPUT_DIR"
echo "  Log: $LOG"

cd "$SCRIPT_DIR"

case "$MODE" in
    preview)
        echo "  Mode: preview (seed 2, pretrained+finetuned+random, ~15min)"
        python render_v57_comparison.py \
            --finetuned_path "$FINETUNED_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --preview \
            --fps 10 \
            --max_steps 96 \
            2>&1 | tee "$LOG"
        ;;
    full_eval)
        echo "  Mode: full_eval (seeds 11-15, all 3 policies, ~1h)"
        python render_v57_comparison.py \
            --finetuned_path "$FINETUNED_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --full_eval \
            --fps 10 \
            --max_steps 96 \
            2>&1 | tee "$LOG"
        ;;
    quick)
        echo "  Mode: quick (seed 2, pretrained only, ~5min)"
        python render_v57_comparison.py \
            --finetuned_path "$FINETUNED_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --seed 2 \
            --policy pretrained \
            --fps 10 \
            --max_steps 96 \
            2>&1 | tee "$LOG"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [preview|full_eval|quick]"
        exit 1
        ;;
esac

echo "[$(date)] Done. Log: $LOG"
echo ""
echo "Videos saved to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.mp4 2>/dev/null | tail -10
ls -lh "$OUTPUT_DIR"/*_d1_v57_summary.csv 2>/dev/null

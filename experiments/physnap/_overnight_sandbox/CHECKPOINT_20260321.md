# Overnight Ablation Checkpoint — 2026-03-21 20:10

## Reason for pause
MINT integration task requires GPU. Overnight ablations paused mid-run.

## Completed experiments (with stats.json)
1. ablation_best_single_view — DONE
2. ablation_fixed_state_multiview — DONE
3. ablation_pts_500 — DONE
4. ablation_pts_1000 — DONE

## Paused experiment
- ablation_pts_2000: 72/96 objects generated (~75%)
  - Partial outputs: external/physnap/log/overnight_box_cond_v2__pts_2000/
  - To resume: re-run with --force (run_guided.py has no mid-run checkpoint)

## Not started
- ablation_pts_5000

## Resume command
conda activate infinigen
cd /mnt/afs2/zhuhaowu/infinigen
python scripts/run_conditioning_v2_group.py --campaign-dir experiments/physnap/_overnight_sandbox/campaign_phase2_ablations --group-id ablation_pts_2000 --force
python scripts/run_conditioning_v2_group.py --campaign-dir experiments/physnap/_overnight_sandbox/campaign_phase2_ablations --group-id ablation_pts_5000 --force

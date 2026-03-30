# MINT Drawer Campaign Dashboard

**Updated**: 2026-03-30T00:00:00+08:00
**Campaign**: `mint_drawer_v1`
**Phase**: `mint_robot_trajectory_claim_push`
**Gate**: `data_distribution_fix_required`
**Claim**: `Infinigen-generated AnyGrasp-conditioned robot-arm drawer trajectories improve MINT success on held-out drawer variants in simulation relative to pretrained MINT.`
**Revision**: `robot_revision_v3_claim_push`

## Queue

- `u1_asset_geometry_audit`: completed
- `u2_joint_semantics_audit`: completed
- `u3_handle_region_audit`: completed
- `u4_export_consistency_audit`: completed
- `u5_seed_outlier_audit`: completed
- `a1_asset_scene_audit`: completed
- `a2_frame_transform_audit`: completed
- `b1_oracle_scripted_baseline`: completed
- `b2_anygrasp_scripted_baseline`: completed
- `c1_teacher_native_rollout_rebuild`: completed
- `c2_action_contract_repair`: completed
- `c3_single_rollout_replay_gate`: completed
- `d1_single_rollout_overfit`: **failed** (attempt #2)
  - V21-V27 deep dive: all engineering bugs fixed
  - Core issue: **Data Distribution Shift confirmed**
- `d2_single_seed_overfit`: pending
- `d3_train_seed_probe`: pending
- `e1_heldout_eval`: pending
- `write_claim_memo`: pending

## V21-V27 Deep Dive Summary

After ~6 hours of iterations (V21-V27), all known engineering bugs are fixed:

### Engineering Fixes Completed
| Fix | Status |
|-----|--------|
| `mint_utils.py` in-place ops (F.silu, f_hat.add_) | FIXED |
| VQ-VAE decoder unfreeze | FIXED (140/153 weights update) |
| `direct_grip_head` dtype mismatch | FIXED |
| `find_latest_checkpoint` bug | FIXED |
| Reconstruction loss in training | ADDED |

### Core Finding: Data Distribution Shift
| Issue | Detail |
|-------|--------|
| **Gripper signal mismatch** | Infinigen: continuous-gradual (1.0→-1.0 over ~5 steps) vs MINT: discrete binary (-1/+1) |
| **grasp_steps mismatch** | Teacher data: 1-2 steps vs `strict_rollout_criteria`: >=5 steps |
| **direct_grip_head learns OPEN** | Bias=+0.008, sigmoid→0.5, always predicts OPEN gripper |

### Positive Signals
- Decoder trained (140/153 weights changed)
- Finetuned EEF motion: **16.3m vs pretrained 1.9m** (8.6x more motion)
- Quantizer updated (codebook entries 369/419/124/179 changed)
- Model learns motion but NOT gripper close

## Data Distribution Fix Options
| Option | Description | Time |
|--------|-------------|------|
| **方案A: Dataset Wrapper** | Gripper binarization, action scaling to match LIBERO | ~10 min |
| **方案B: AnyGrasp Regeneration** | Force gripper -1.0 for >=5 steps at contact | 2-3h GPU |
| **方案C: Criteria Adjustment** | Lower `grasp_steps_gte` from 5 to 1-2 | ~5 min |

## Recent History
- 2026-03-29T14:00:00+08:00: d1_deep_dive_started | V21-V27 deep dive complete
- 2026-03-29T14:00:00+08:00: step_reconciled | d1_single_rollout_overfit => failed (attempt #2)

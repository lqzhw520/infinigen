# Infinigen x PhysNAP Phase 1 Transfer Diagnosis Campaign Dashboard

**Updated**: 2026-03-19T16:04:13+08:00
**Phase**: `phase1_diagnostics`
**Gate**: `auto_handoff_ready`
**Claim**: `Phase 1 diagnoses whether forgetting, over-updating, or parameterization limits dominate repaired transfer behavior.`

## Queue

- `prepare_k10`: completed
- `train_infinigen_k10`: completed
- `train_finetune_k10`: completed
- `prepare_mixed_replay`: completed
- `train_mixed_finetune_k10`: completed
- `train_partial_finetune_k10`: completed
- `evaluate_all`: completed

## Runtime

- Watcher PID: `2515638`
- Supervisor PID: `2501842`
- Worker PID: `None`
- CPU Worker PID: `None`
- Current Queue Step: `evaluate_all`
- Current Evaluation Stage: `write_report`
- Watcher Status: `running`
- Active GPU Step: `evaluate_all`
- Active CPU Step: `None`
- Active Runtime Step: `None`
- Active Runtime Worker Kind: `None`
- Active Runtime Worker PID: `None`
- Active Runtime Attempt: `None`
- Last Repair Action: `None`
- Last Heartbeat: `2026-03-19T16:04:11+08:00`
- Latest Artifact Produced: `/mnt/afs2/zhuhaowu/infinigen/experiments/physnap/box_prior_v1/evaluation/comparison_report.md`
- Blocking Error: `None`
- Phase 2 Base Experiment: `mixed_finetune_k10`
- Phase 2 Base Checkpoint: `/mnt/afs2/zhuhaowu/infinigen/external/physnap/log/v6.1_diffusion_mixed_finetune_infinigen_k10/checkpoint/425_latest.pt`
- Auto Handoff Status: `ready`

## Evaluation Progress

- Backend: `torch_fallback`
- Status: `completed`
- Current Stage: `write_report`
- Message: Evaluation summary and report are ready
- `export_box_ref`: `completed` | updated `2026-03-18T20:58:35+08:00` | Reference `infinigen_box_k10_val_eval32` ready for box metrics
- `export_partnet_ref`: `completed` | updated `2026-03-18T20:58:35+08:00` | Reference `GTval_eval64` ready
- `generate_baseline`: `completed` | updated `2026-03-18T20:58:35+08:00` | Generation ready for `baseline`
- `generate_scratch`: `completed` | updated `2026-03-18T20:58:35+08:00` | Generation ready for `infinigen_k10`
- `generate_finetune`: `completed` | updated `2026-03-18T20:58:35+08:00` | Generation ready for `finetune_k10`
- `generate_mixed_finetune`: `completed` | updated `2026-03-18T21:01:16+08:00` | Generation ready for `mixed_finetune_k10`
- `generate_partial_finetune`: `completed` | updated `2026-03-18T21:03:22+08:00` | Generation ready for `partial_finetune_k10`
- `compute_box_metrics`: `completed` | updated `2026-03-18T21:12:17+08:00` | Box-domain metrics complete
- `compute_partnet_metrics`: `completed` | updated `2026-03-18T21:26:18+08:00` | PartNet sanity metrics complete
- `audit_conversion`: `completed` | updated `2026-03-18T21:26:40+08:00` | Conversion audit complete
- `write_report`: `completed` | updated `2026-03-18T21:26:40+08:00` | Evaluation summary and report are ready

## Experiment Status

### baseline

- Status: `reference`
- Checkpoint: `/mnt/afs2/zhuhaowu/infinigen/external/physnap/log/v6.1_diffusion_adapted/checkpoint/5455.pt`
- Log dir: `v6.1_diffusion_adapted`

### infinigen_k10

- Status: `completed`
- PID: `2222811`
- Launched: `2026-03-17T11:42:26+08:00`
- Finished: `2026-03-17T14:53:42+08:00`
- Log dir: `v6.1_diffusion_infinigen_k10`

### finetune_k10

- Status: `completed`
- PID: `2272456`
- Launched: `2026-03-17T15:00:09+08:00`
- Finished: `2026-03-17T16:25:49+08:00`
- Log dir: `v6.1_diffusion_finetune_infinigen_k10`

### mixed_finetune_k10

- Status: `completed`
- PID: `2438644`
- Launched: `2026-03-18T17:10:12+08:00`
- Finished: `2026-03-18T18:32:56+08:00`
- Log dir: `v6.1_diffusion_mixed_finetune_infinigen_k10`

### partial_finetune_k10

- Status: `completed`
- PID: `2475979`
- Launched: `2026-03-18T19:35:03+08:00`
- Finished: `2026-03-18T20:58:32+08:00`
- Log dir: `v6.1_diffusion_partial_finetune_infinigen_k10`

## Review

- Verdict: `phase1_diagnosis_ready`
- Decision: `advance_phase`
- Blocker Type: `none`
- Evidence Score: `6`/10
- Workflow Score: `10`/10
- Diagnosis: `forgetting-dominant`
- Claim assessment: The repaired bridge is healthy, and the dominant remaining failure signal is catastrophic forgetting during pure fine-tune.
- Weakness: Fine-tune box-domain MMD is worse than baseline.
- Weakness: PartNet sanity shows a notable MMD regression after fine-tuning.
- Strength: Diagnostics decision: `healthy`.
- Strength: Baseline, scratch, and fine-tune arms are present.
- Strength: Fine-tune maintains or improves box-domain coverage relative to scratch.
- Strength: Fine-tune improves or matches baseline penetration error.
- Strength: Fine-tune improves or matches baseline mobility error.
- Strength: Mixed replay diagnostic arm is present.
- Strength: Partial fine-tune diagnostic arm is present.
- Strength: Dominant Phase 1 diagnosis: `forgetting-dominant`.
- Strength: Selected Phase 2 base checkpoint: `mixed_finetune_k10`.
- Next: Write the Phase 1 decision memo and selected Phase 2 base checkpoint.
- Next: Auto-bootstrap `box_conditioning_v2` immediately after Phase 1 completes.

## Recent History

- 2026-03-18T17:00:31+08:00: phase1_bootstrap | Archived prior repaired-branch report and reset queue for mixed/partial diagnostics.
- 2026-03-18T17:10:01+08:00: prepare_mixed_replay | Completed prepare_mixed_replay.
- 2026-03-18T17:10:17+08:00: train_mixed_finetune_k10 | Launched mixed_finetune_k10 in background.
- 2026-03-18T18:33:01+08:00: train_partial_finetune_k10 | Launched partial_finetune_k10 in background.
- 2026-03-18T19:33:16+08:00: auto_repair | Queued retry for train_partial_finetune_k10 after detected failure `optimizer_key_mismatch`.
- 2026-03-18T19:34:52+08:00: manual_retry_reset | Reset partial fine-tune and evaluate queue items after patching initialization semantics and auto-repair recovery.
- 2026-03-18T19:35:03+08:00: train_partial_finetune_k10 | Launched partial_finetune_k10 in background.
- 2026-03-18T21:27:03+08:00: evaluate_all | Evaluation completed.

## Evaluation Summary

Evaluation artifacts are present.

## Diagnostics

- Conversion audit decision: `healthy`
- GT pen mean: `0.0`
- GT mob mean: `4.3893143768514165e-06`

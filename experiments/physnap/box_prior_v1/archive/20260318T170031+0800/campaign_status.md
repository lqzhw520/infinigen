# Box Prior V1 Campaign Dashboard

**Updated**: 2026-03-18T16:05:14+08:00
**Claim**: `multi-state procedural assets improve articulated generative robustness beyond zero-articulation training`

## Queue

- `prepare_k10`: completed
- `train_infinigen_k10`: completed
- `train_finetune_k10`: completed
- `evaluate_all`: completed

## Runtime

- Watcher PID: `2437450`
- Worker PID: `None`
- Current Queue Step: `None`
- Current Evaluation Stage: `None`
- Watcher Status: `starting`
- Last Heartbeat: `2026-03-18T16:05:13+08:00`
- Latest Artifact Produced: `None`
- Blocking Error: `None`

## Evaluation Progress

- Backend: `torch_fallback`
- Status: `completed`
- Current Stage: `write_report`
- Message: Evaluation summary and report are ready
- `export_box_ref`: `completed` | updated `2026-03-18T15:16:13+08:00` | Reference `infinigen_box_k10_val_eval32` ready for box metrics
- `export_partnet_ref`: `completed` | updated `2026-03-18T15:20:41+08:00` | Reference `GTval_eval64` ready
- `generate_baseline`: `completed` | updated `2026-03-18T15:23:04+08:00` | Generation ready for `baseline`
- `generate_scratch`: `completed` | updated `2026-03-18T15:25:18+08:00` | Generation ready for `infinigen_k10`
- `generate_finetune`: `completed` | updated `2026-03-18T15:27:18+08:00` | Generation ready for `finetune_k10`
- `generate_mixed_finetune`: `pending` | updated `None` | 
- `generate_partial_finetune`: `pending` | updated `None` | 
- `compute_box_metrics`: `completed` | updated `2026-03-18T15:40:53+08:00` | Box-domain metrics complete
- `compute_partnet_metrics`: `completed` | updated `2026-03-18T16:02:31+08:00` | PartNet sanity metrics complete
- `audit_conversion`: `completed` | updated `2026-03-18T16:02:54+08:00` | Conversion audit complete
- `write_report`: `completed` | updated `2026-03-18T16:02:54+08:00` | Evaluation summary and report are ready

## What Was Implemented

- K=10 box-domain data preparation, training launcher, evaluator, and auto-review loop.
- True fine-tune initialization from PartNet `5455.pt` with `p_pe` ignored for the K=10 adaptation.
- Runtime sidecars under `experiments/physnap/box_prior_v1/runtime/` to survive PhysNAP log-directory resets.
- Project-memory iteration recording and STATUS regeneration compatibility fix for legacy `bugs_fixed` string entries.

## Issues Solved

- Old fine-tune path was scratch training because no initialization checkpoint was wired in.
- PhysNAP startup deleted prewritten `run_meta.json` and resolved configs inside `log/<exp>`.
- Headless visualization warnings are tolerated and do not block training progression.

## Experiment Status

### baseline

- Status: `reference`
- Latest checkpoint: `5455.pt` (241.1 MB)
- Checkpoint updated: `2026-03-10T23:17:20+08:00`
- Batch loss: first `3.654698610305786` -> last `0.01945587433874607` (min `0.01415217760950327`)
- Epoch trace: first `1` -> last `5455`
- Non-blocking warnings: render `120`, mesh `0`

### infinigen_k10

- Status: `completed`
- Launched: `2026-03-17T11:42:26+08:00`
- Finished: `2026-03-17T14:53:42+08:00`
- Latest checkpoint: `2400_latest.pt` (241.2 MB)
- Checkpoint updated: `2026-03-17T23:52:38+08:00`
- Batch loss: first `2.83164381980896` -> last `0.011710312217473984` (min `0.005593461915850639`)
- Epoch trace: first `1` -> last `2401`
- Non-blocking warnings: render `60`, mesh `143`

### finetune_k10

- Status: `completed`
- Launched: `2026-03-17T15:00:09+08:00`
- Finished: `2026-03-17T16:25:49+08:00`
- Latest checkpoint: `1200_latest.pt` (241.2 MB)
- Checkpoint updated: `2026-03-18T01:16:38+08:00`
- Batch loss: first `0.15172429382801056` -> last `0.00880079623311758` (min `0.006165142171084881`)
- Epoch trace: first `1` -> last `1201`
- Non-blocking warnings: render `30`, mesh `70`

## Review

- Verdict: `claim_not_supported`
- Evidence Score: `6`/10
- Workflow Score: `10`/10
- Claim assessment: The current implementation does not support the main claim. Static structure is not the main failure signal; the weak link is articulation/physics transfer and/or catastrophic forgetting.
- Weakness: Fine-tune box-domain MMD is worse than the baseline reference.
- Weakness: PartNet sanity shows a notable MMD regression after fine-tuning.
- Strength: All three comparison arms are present.
- Strength: Fine-tune maintains or improves box-domain coverage relative to scratch.
- Strength: Fine-tune improves or matches baseline penetration error.
- Strength: Fine-tune improves or matches baseline mobility error.
- Strength: Conversion audit completed with decision `healthy`.
- Next: Run `mixed_finetune_k10` to test whether pure fine-tuning is causing catastrophic forgetting.
- Next: Run `partial_finetune_k10` to test whether full fine-tuning is destroying the pretrained prior.
- Next: Inspect comparison_report.md and conversion_audit.md together to separate data fidelity issues from transfer-recipe issues.
- Next: Only lower the claim after mixed and partial fine-tune diagnostics also fail without any implementation blocker.

## Gate Decision

- `infinigen_k10` has a full-duration checkpoint and is eligible for the finetune stage.

## Diagnostics

- Conversion audit decision: `healthy`
- GT self-eval pen mean: `0.0`
- GT self-eval mob mean: `4.3893143768514165e-06`

## Evaluation Summary

Evaluation artifacts are present.

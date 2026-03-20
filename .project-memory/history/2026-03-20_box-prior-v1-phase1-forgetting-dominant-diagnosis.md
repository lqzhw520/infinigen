# Campaign History: box_prior_v1 phase1 forgetting-dominant diagnosis

**Archived**: 2026-03-20T14:24:59+08:00
**Campaign**: `box_prior_v1`
**Phase**: `phase1_diagnostics`
**Gate**: `auto_handoff_ready`
**Verdict**: `phase1_diagnosis_ready`
**Decision**: `advance_phase`

## Claim

`Phase 1 diagnoses whether forgetting, over-updating, or parameterization limits dominate repaired transfer behavior.`

## Review

- Claim assessment: The repaired bridge is healthy, and the dominant remaining failure signal is catastrophic forgetting during pure fine-tune.
- Diagnosis: `forgetting-dominant`
- Selected Phase 2 base experiment: `mixed_finetune_k10`
- Selected Phase 2 base checkpoint: `/mnt/afs2/zhuhaowu/infinigen/external/physnap/log/v6.1_diffusion_mixed_finetune_infinigen_k10/checkpoint/425_latest.pt`

## Queue

- `prepare_k10`: completed
- `train_infinigen_k10`: completed
- `train_finetune_k10`: completed
- `prepare_mixed_replay`: completed
- `train_mixed_finetune_k10`: completed
- `train_partial_finetune_k10`: completed
- `evaluate_all`: completed

## Strengths

- Diagnostics decision: `healthy`.
- Baseline, scratch, and fine-tune arms are present.
- Fine-tune maintains or improves box-domain coverage relative to scratch.
- Fine-tune improves or matches baseline penetration error.
- Fine-tune improves or matches baseline mobility error.
- Mixed replay diagnostic arm is present.
- Partial fine-tune diagnostic arm is present.
- Dominant Phase 1 diagnosis: `forgetting-dominant`.
- Selected Phase 2 base checkpoint: `mixed_finetune_k10`.

## Weaknesses

- Fine-tune box-domain MMD is worse than baseline.
- PartNet sanity shows a notable MMD regression after fine-tuning.

## Next Actions

- Write the Phase 1 decision memo and selected Phase 2 base checkpoint.
- Auto-bootstrap `box_conditioning_v2` immediately after Phase 1 completes.

## Recent Campaign History

- 2026-03-17T21:10:30+08:00: representation_gate_passed | Regenerated box data with fixed converter; GT self-eval healthy; launched infinigen_k10_fixed.
- 2026-03-18T17:00:31+08:00: phase1_bootstrap | Archived prior repaired-branch report and reset queue for mixed/partial diagnostics.
- 2026-03-18T17:10:01+08:00: prepare_mixed_replay | Completed prepare_mixed_replay.
- 2026-03-18T17:10:17+08:00: train_mixed_finetune_k10 | Launched mixed_finetune_k10 in background.
- 2026-03-18T18:33:01+08:00: train_partial_finetune_k10 | Launched partial_finetune_k10 in background.
- 2026-03-18T19:33:16+08:00: auto_repair | Queued retry for train_partial_finetune_k10 after detected failure `optimizer_key_mismatch`.
- 2026-03-18T19:34:52+08:00: manual_retry_reset | Reset partial fine-tune and evaluate queue items after patching initialization semantics and auto-repair recovery.
- 2026-03-18T19:35:03+08:00: train_partial_finetune_k10 | Launched partial_finetune_k10 in background.
- 2026-03-18T21:27:03+08:00: evaluate_all | Evaluation completed.

## Evidence Snapshot

- diagnostics decision: `healthy`

## Canonical Files

- `/mnt/afs2/zhuhaowu/infinigen/experiments/physnap/box_prior_v1/state.json`
- `/mnt/afs2/zhuhaowu/infinigen/experiments/physnap/box_prior_v1/review.json`
- `/mnt/afs2/zhuhaowu/infinigen/experiments/physnap/box_prior_v1/decision_memo.md`
- `/mnt/afs2/zhuhaowu/infinigen/experiments/physnap/box_prior_v1/campaign_status.md`

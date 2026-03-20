# Campaign History: box_conditioning_v2 phase2 multistate rerun progress

**Archived**: 2026-03-20T14:24:59+08:00
**Campaign**: `box_conditioning_v2`
**Phase**: `phase2_conditioning_design`
**Gate**: `await_phase2_execution`
**Verdict**: `waiting_for_results`
**Decision**: `run_experiments`

## Claim

`Multi-state, multi-view procedural articulated observations can outperform PhysNAP's original zero-articulation single-observation conditioning regime.`

## Review

- Claim assessment: No evaluation evidence yet.

## Queue

- `prepare_conditioning_v2`: completed
- `export_cond_zero_singleview`: completed
- `export_cond_zero_multiview`: completed
- `export_cond_multistate_singleview`: completed
- `export_cond_multistate_multiview`: completed
- `run_guided_zero_singleview`: completed
- `run_guided_zero_multiview`: completed
- `run_guided_multistate_singleview`: running
- `run_guided_multistate_multiview`: pending
- `evaluate_conditioning_v2`: pending
- `write_phase2_claim_memo`: pending

## Active Runtime

- step: `run_guided_multistate_singleview`
- worker_kind: `gpu_guided`
- worker_pid: `2642746`
- launched_at: `2026-03-20T10:12:37+08:00`
- heartbeat_at: `2026-03-20T14:24:50+08:00`

## Weaknesses

- No evaluation summary exists yet.

## Next Actions

- Wait for active queue step `run_guided_multistate_singleview` to finish.

## Recent Campaign History

- 2026-03-18T20:53:59+0800: phase2_prestage_completed | export_cond_multistate_multiview finished with exit code 0.
- 2026-03-18T21:27:32+0800: auto_handoff_from_phase1 | Handoff from box_prior_v1 using base mixed_finetune_k10.
- 2026-03-19T10:17:24+08:00: phase2_auto_repair | Repaired Phase 2 base checkpoint/config for `mixed_finetune_k10` and reset `run_guided_zero_singleview`.
- 2026-03-19T15:17:37+0800: supervisor_inner_restart | Restarted inner loop after stagnant dead watcher; restart 1/6.
- 2026-03-19T16:03:10+0800: supervisor_inner_restart | Restarted inner loop after stagnant dead watcher; restart 2/6.
- 2026-03-20T01:26:16+08:00: run_guided_multistate_singleview | Completed run_guided_multistate_singleview.
- 2026-03-20T06:00:41+08:00: run_guided_multistate_multiview | Completed run_guided_multistate_multiview.
- 2026-03-20T06:01:22+08:00: evaluate_conditioning_v2 | Evaluation failed with exit code 1.
- 2026-03-20T06:02:03+08:00: evaluate_conditioning_v2 | Evaluation failed with exit code 1.
- 2026-03-20T06:02:22+08:00: auto_repair | Queued retry for evaluate_conditioning_v2 after detected failure `invalid_phase2_summary`.
- 2026-03-20T06:02:53+08:00: evaluate_conditioning_v2 | Evaluation failed with exit code 1.
- 2026-03-20T10:12:37+08:00: phase2_auto_repair | Reset `run_guided_multistate_singleview` because guided outputs for `multistate_singleview` were incomplete.

## Evidence Snapshot


## Canonical Files

- `/mnt/afs2/zhuhaowu/infinigen/experiments/physnap/box_conditioning_v2/state.json`
- `/mnt/afs2/zhuhaowu/infinigen/experiments/physnap/box_conditioning_v2/review.json`
- `/mnt/afs2/zhuhaowu/infinigen/experiments/physnap/box_conditioning_v2/decision_memo.md`
- `/mnt/afs2/zhuhaowu/infinigen/experiments/physnap/box_conditioning_v2/campaign_status.md`

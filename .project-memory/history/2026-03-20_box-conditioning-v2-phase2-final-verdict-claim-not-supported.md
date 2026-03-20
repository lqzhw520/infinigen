# Campaign History: box_conditioning_v2 phase2 final verdict claim_not_supported

**Archived**: 2026-03-20T21:20:50+08:00
**Campaign**: `box_conditioning_v2`
**Phase**: `phase2_conditioning_design`
**Gate**: `ready_for_writeup`
**Verdict**: `claim_not_supported`
**Decision**: `revise_claim`

## Claim

`Multi-state, multi-view procedural articulated observations can outperform PhysNAP's original zero-articulation single-observation conditioning regime.`

## Review

- Claim assessment: All bounded Phase 2 conditioning recipes completed, but none produced a consistent gain over the zero-state single-view anchor.
- Strongest true claim: Mixed replay is the strongest validated transfer improvement, but richer conditioning under the current PhysNAP parameterization does not consistently beat the zero-state single-view anchor.

## Queue

- `prepare_conditioning_v2`: completed
- `export_cond_zero_singleview`: completed
- `export_cond_zero_multiview`: completed
- `export_cond_multistate_singleview`: completed
- `export_cond_multistate_multiview`: completed
- `run_guided_zero_singleview`: completed
- `run_guided_zero_multiview`: completed
- `run_guided_multistate_singleview`: completed
- `run_guided_multistate_multiview`: completed
- `evaluate_conditioning_v2`: completed
- `write_phase2_claim_memo`: completed

## Weaknesses

- Baseline, scratch, and fine-tune comparison arms are incomplete.
- `zero_multiview` does not consistently beat the zero-state single-view anchor.
- `multistate_singleview` does not consistently beat the zero-state single-view anchor.
- `multistate_multiview` does not consistently beat the zero-state single-view anchor.
- Claim ladder `multi-state + multi-view beats zero-state single-view` via `multistate_multiview` => not supported.
- Claim ladder `multi-view only beats zero-state single-view` via `zero_multiview` => not supported.
- Claim ladder `multi-state only beats zero-state single-view` via `multistate_singleview` => not supported.

## Next Actions

- Write the strongest true claim memo for Phase 2 instead of the original strong claim.
- Record which conditioning groups failed to improve the anchor and why.

## Recent Campaign History

- 2026-03-20T06:00:41+08:00: run_guided_multistate_multiview | Completed run_guided_multistate_multiview.
- 2026-03-20T06:01:22+08:00: evaluate_conditioning_v2 | Evaluation failed with exit code 1.
- 2026-03-20T06:02:03+08:00: evaluate_conditioning_v2 | Evaluation failed with exit code 1.
- 2026-03-20T06:02:22+08:00: auto_repair | Queued retry for evaluate_conditioning_v2 after detected failure `invalid_phase2_summary`.
- 2026-03-20T06:02:53+08:00: evaluate_conditioning_v2 | Evaluation failed with exit code 1.
- 2026-03-20T10:12:37+08:00: phase2_auto_repair | Reset `run_guided_multistate_singleview` because guided outputs for `multistate_singleview` were incomplete.
- 2026-03-20T14:50:51+08:00: run_guided_multistate_singleview | Completed run_guided_multistate_singleview with --force.
- 2026-03-20T14:51:17+08:00: run_guided_multistate_multiview | Completed run_guided_multistate_multiview.
- 2026-03-20T14:51:28+08:00: phase2_auto_repair | Reset `run_guided_multistate_multiview` because guided outputs for `multistate_multiview` were incomplete.
- 2026-03-20T19:35:13+08:00: run_guided_multistate_multiview | Completed run_guided_multistate_multiview with --force.
- 2026-03-20T19:35:54+08:00: evaluate_conditioning_v2 | Evaluation completed.
- 2026-03-20T19:36:20+08:00: write_phase2_claim_memo | Completed write_phase2_claim_memo.

## Evidence Snapshot

- strongest_true_claim: Richer conditioning did not consistently beat the zero-state single-view anchor under the bounded Phase 2 recipes.
- zero_singleview: mmd=0.5589997228235006 cov=0.0908203125 1NN=0.9250000044703484 E_pen=0.0007139240296964999 E_mob=0.0005797943085781299 cond=0.0022055536246625707
- zero_multiview: mmd=0.5496429605409503 cov=0.08984375 1NN=0.9241071473807096 E_pen=0.0007759372820146382 E_mob=0.0005648964070132934 cond=0.0014058477245271206
- multistate_singleview: mmd=0.5471300091594458 cov=0.0908203125 1NN=0.9250000044703484 E_pen=0.0022372949169948697 E_mob=0.0006206683465279639 cond=0.0013572014092157285
- multistate_multiview: mmd=0.5368264261633158 cov=0.0888671875 1NN=0.9214285761117935 E_pen=0.0091311283952867 E_mob=0.010760995797075642 cond=0.0014382785884663463

## Canonical Files

- `/mnt/afs2/zhuhaowu/infinigen/experiments/physnap/box_conditioning_v2/state.json`
- `/mnt/afs2/zhuhaowu/infinigen/experiments/physnap/box_conditioning_v2/review.json`
- `/mnt/afs2/zhuhaowu/infinigen/experiments/physnap/box_conditioning_v2/decision_memo.md`
- `/mnt/afs2/zhuhaowu/infinigen/experiments/physnap/box_conditioning_v2/campaign_status.md`

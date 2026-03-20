# Infinigen x PhysNAP Phase 2 Conditioning Redesign Campaign Dashboard

**Updated**: 2026-03-20T19:36:21+08:00
**Phase**: `phase2_conditioning_design`
**Gate**: `ready_for_writeup`
**Claim**: `Multi-state, multi-view procedural articulated observations can outperform PhysNAP's original zero-articulation single-observation conditioning regime.`

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

## Runtime

- Watcher PID: `2642735`
- Supervisor PID: `2515830`
- Worker PID: `None`
- CPU Worker PID: `None`
- Current Queue Step: `write_phase2_claim_memo`
- Current Evaluation Stage: `write_phase2_claim_memo`
- Watcher Status: `completed`
- Active GPU Step: `evaluate_conditioning_v2`
- Active CPU Step: `None`
- Active Runtime Step: `None`
- Active Runtime Worker Kind: `None`
- Active Runtime Worker PID: `None`
- Active Runtime Attempt: `None`
- Last Repair Action: `evaluate_conditioning_v2:invalid_phase2_summary`
- Last Heartbeat: `2026-03-20T19:36:20+08:00`
- Latest Artifact Produced: `/mnt/afs2/zhuhaowu/infinigen/experiments/physnap/box_conditioning_v2/evaluation/comparison_report.md`
- Blocking Error: `None`
- Phase 2 Base Experiment: `mixed_finetune_k10`
- Phase 2 Base Checkpoint: `/mnt/afs2/zhuhaowu/infinigen/external/physnap/log/v6.1_diffusion_mixed_finetune_infinigen_k10/checkpoint/425_latest.pt`
- Auto Handoff Status: `bootstrapped`

## Conditioning Groups

- `zero_singleview`: output `box_conditioning_v2__zero_singleview` | mmd `0.5589997228235006` | cov `0.0908203125` | pen `0.0007139240296964999` | mob `0.0005797943085781299`
- `zero_multiview`: output `box_conditioning_v2__zero_multiview` | mmd `0.5496429605409503` | cov `0.08984375` | pen `0.0007759372820146382` | mob `0.0005648964070132934`
- `multistate_singleview`: output `box_conditioning_v2__multistate_singleview` | mmd `0.5471300091594458` | cov `0.0908203125` | pen `0.0022372949169948697` | mob `0.0006206683465279639`
- `multistate_multiview`: output `box_conditioning_v2__multistate_multiview` | mmd `0.5368264261633158` | cov `0.0888671875` | pen `0.0091311283952867` | mob `0.010760995797075642`
## Review

- Verdict: `claim_not_supported`
- Decision: `revise_claim`
- Blocker Type: `transfer`
- Evidence Score: `5`/10
- Workflow Score: `10`/10
- Strongest True Claim: Mixed replay is the strongest validated transfer improvement, but richer conditioning under the current PhysNAP parameterization does not consistently beat the zero-state single-view anchor.
- Claim assessment: All bounded Phase 2 conditioning recipes completed, but none produced a consistent gain over the zero-state single-view anchor.
- Weakness: Baseline, scratch, and fine-tune comparison arms are incomplete.
- Weakness: `zero_multiview` does not consistently beat the zero-state single-view anchor.
- Weakness: `multistate_singleview` does not consistently beat the zero-state single-view anchor.
- Weakness: `multistate_multiview` does not consistently beat the zero-state single-view anchor.
- Weakness: Claim ladder `multi-state + multi-view beats zero-state single-view` via `multistate_multiview` => not supported.
- Weakness: Claim ladder `multi-view only beats zero-state single-view` via `zero_multiview` => not supported.
- Weakness: Claim ladder `multi-state only beats zero-state single-view` via `multistate_singleview` => not supported.
- Next: Write the strongest true claim memo for Phase 2 instead of the original strong claim.
- Next: Record which conditioning groups failed to improve the anchor and why.

## Recent History

- 2026-03-20T06:02:53+08:00: evaluate_conditioning_v2 | Evaluation failed with exit code 1.
- 2026-03-20T10:12:37+08:00: phase2_auto_repair | Reset `run_guided_multistate_singleview` because guided outputs for `multistate_singleview` were incomplete.
- 2026-03-20T14:50:51+08:00: run_guided_multistate_singleview | Completed run_guided_multistate_singleview with --force.
- 2026-03-20T14:51:17+08:00: run_guided_multistate_multiview | Completed run_guided_multistate_multiview.
- 2026-03-20T14:51:28+08:00: phase2_auto_repair | Reset `run_guided_multistate_multiview` because guided outputs for `multistate_multiview` were incomplete.
- 2026-03-20T19:35:13+08:00: run_guided_multistate_multiview | Completed run_guided_multistate_multiview with --force.
- 2026-03-20T19:35:54+08:00: evaluate_conditioning_v2 | Evaluation completed.
- 2026-03-20T19:36:20+08:00: write_phase2_claim_memo | Completed write_phase2_claim_memo.

## Evaluation Summary

Evaluation artifacts are present.

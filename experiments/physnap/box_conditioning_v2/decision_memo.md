# Infinigen x PhysNAP Phase 2 Conditioning Redesign Decision Memo

**Updated**: 2026-03-20T19:36:21+08:00
**Phase**: `phase2_conditioning_design`
**Gate**: `ready_for_writeup`
**Verdict**: `claim_not_supported`
**Decision**: `revise_claim`
**Workflow Score**: `10`/10
**Evidence Score**: `5`/10

## Claim Assessment

All bounded Phase 2 conditioning recipes completed, but none produced a consistent gain over the zero-state single-view anchor.

## Scientific Position

- Strongest true claim: Mixed replay is the strongest validated transfer improvement, but richer conditioning under the current PhysNAP parameterization does not consistently beat the zero-state single-view anchor.

## Strengths

- None recorded.

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

## Phase Handoff

- phase2_base_experiment: `mixed_finetune_k10`
- phase2_base_checkpoint: `/mnt/afs2/zhuhaowu/infinigen/external/physnap/log/v6.1_diffusion_mixed_finetune_infinigen_k10/checkpoint/425_latest.pt`
- phase2_base_config: `/mnt/afs2/zhuhaowu/infinigen/external/physnap/configs/nap/v6.1_diffusion_mixed_finetune_infinigen_k10.yaml`
- auto_handoff_status: `bootstrapped`

## Evidence Snapshot

- zero_singleview: mmd=0.5589997228235006 cov=0.0908203125 1NN=0.9250000044703484 E_pen=0.0007139240296964999 E_mob=0.0005797943085781299 cond=0.0022055536246625707
- zero_multiview: mmd=0.5496429605409503 cov=0.08984375 1NN=0.9241071473807096 E_pen=0.0007759372820146382 E_mob=0.0005648964070132934 cond=0.0014058477245271206
- multistate_singleview: mmd=0.5471300091594458 cov=0.0908203125 1NN=0.9250000044703484 E_pen=0.0022372949169948697 E_mob=0.0006206683465279639 cond=0.0013572014092157285
- multistate_multiview: mmd=0.5368264261633158 cov=0.0888671875 1NN=0.9214285761117935 E_pen=0.0091311283952867 E_mob=0.010760995797075642 cond=0.0014382785884663463

## Claim Ladder

- multi-state + multi-view beats zero-state single-view: `multistate_multiview` => not supported
- multi-view only beats zero-state single-view: `zero_multiview` => not supported
- multi-state only beats zero-state single-view: `multistate_singleview` => not supported

# Infinigen x PhysNAP Phase 1 Transfer Diagnosis Decision Memo

**Updated**: 2026-03-19T16:04:13+08:00
**Phase**: `phase1_diagnostics`
**Gate**: `auto_handoff_ready`
**Verdict**: `phase1_diagnosis_ready`
**Decision**: `advance_phase`
**Workflow Score**: `10`/10
**Evidence Score**: `6`/10

## Claim Assessment

The repaired bridge is healthy, and the dominant remaining failure signal is catastrophic forgetting during pure fine-tune.

## Scientific Position

- Phase 1 dominant diagnosis: `forgetting-dominant`

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

## Phase Handoff

- phase2_base_experiment: `mixed_finetune_k10`
- phase2_base_checkpoint: `/mnt/afs2/zhuhaowu/infinigen/external/physnap/log/v6.1_diffusion_mixed_finetune_infinigen_k10/checkpoint/425_latest.pt`
- phase2_base_config: `/mnt/afs2/zhuhaowu/infinigen/external/physnap/configs/nap/v6.1_diffusion_mixed_finetune_infinigen_k10.yaml`
- auto_handoff_status: `ready`

## Evidence Snapshot

- diagnostics decision: `healthy`
- baseline: MMD=0.36308541893959045 COV=0.53125 E_pen=0.44212023355066776 E_mob=0.4545806972309947
- infinigen_k10: MMD=0.4774954915046692 COV=0.28125 E_pen=0.005582073703408241 E_mob=0.009745195508003235
- finetune_k10: MMD=0.4406281113624573 COV=0.375 E_pen=0.014981623739004135 E_mob=0.004752624940010719
- mixed_finetune_k10: MMD=0.32110756635665894 COV=0.5625 E_pen=0.13090578466653824 E_mob=0.10306754242628813
- partial_finetune_k10: MMD=0.48090338706970215 COV=0.3125 E_pen=0.007567312642350998 E_mob=0.004471115302294493

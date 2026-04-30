# Reset Geometry And Model Builder Repair Closeout

Closeout: `RESET_GEOMETRY_REPAIRED_PREACTION_SMOKE_PASSED`

This phase repaired reset/preaction physical plausibility for V11-G4 under the unchanged GOC-v3 exact-ID authority. It did not run rollout, render, training, or bounded candidate generation.

## Key Evidence

- Harness preflight passed: `True`
- Lock bound to repair spec: `True`
- Model-builder patch: `scripts/mint/merged_model_builder.py`
- Selected robot base offset: `[-0.9, 0.0, 0.0]`
- Forbidden contacts at reset: `0`
- Max reset penetration: `0.0` m
- Max reset contact force: `0.0` N
- Exact GOC-v3 IDs emitted: `True`
- Body-based 31/27 rejected: `True`
- Legacy name-only rejected: `True`

## Boundary

No GOC-v3 authority IDs were changed. No forbidden contact was accepted. No threshold was lowered. No direct-qpos drawer opening, rollout, render, teacher generation, training, or current truth mutation was performed.

## Next Gate

`V11_G4_GOC_V3_COMPOSITE_PHASE1H_BOUNDED_AUTONOMOUS_RUN`

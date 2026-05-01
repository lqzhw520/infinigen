# GOC-v4 Robust Contact Pull Export Local Replay Overnight Closeout

Closeout: `ROBUST_LAYER4R_CONTACT_DYNAMICS_FAILED`

This phase is deliberately stricter than the prior single-instance Layer 4 result. It does not promote the earlier 7-frame contact probe into rollout eligibility unless the robust Layer 4R gate passes across the available drawer seeds and perturbations.

Key results:
- harness preflight passed: `None`
- robust Layer 4R passed: `False`
- available drawer seeds tested: `[1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15]`
- Layer 4R test cases: `117`
- Layer 4R failed cases: `90`
- bounded pull attempted: `False`
- pull attempts used: `0`
- strict candidate found: `False`
- strict candidate drawer fraction: `0.0`
- local replay/render passed: `False`

The phase never mutates `current_truth.json` or `next_actions.json`, never uses body-based 31/27 authority, never treats forbidden or broad-link contact as target, never directly opens the drawer qpos, and never claims MINT or visual success without the required gates.

Next gate: `ROBUST_CONTACT_DYNAMICS_REPAIR_WITH_OFFENDING_SEED_EVIDENCE`

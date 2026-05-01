
# V11-G4 GOC-v3 Contact-Aware Teacher Synthesis

Closeout: **COLLISION_GEOMETRY_PAD_MODEL_INSUFFICIENT**

Prior dynamic failure mode: `C_IK_SETPOINT_OR_CLOSE_COMMAND_STARTS_INSIDE_HANDLE`.
Contact geometry quality: `broad_link_collision`.

A guarded contact mode controller and force/penetration-limited probe controller were built. CEM/MPPI-style optimization was attempted with `5` iterations and `16` samples per iteration.

Best probe:

- target_contact_frames: `150`
- target_contact_max_consecutive_frames: `150`
- forbidden_contact_frames: `0`
- max_penetration_m: `0.3127174777741097`
- max_force_n: `297106.5508951408`
- max_drawer_fraction: `0.0`

Bounded rollout attempted: `False`.
Strict candidate found: `False`.

Current truth modified: `false`. Next actions modified: `false`. GOC-v3 authority changed: `false`.

Next gate: **GOC_V4_DEDICATED_FINGER_PAD_COLLISION_MODEL_REPAIR**

## Post-push Verification

- Evidence commit: `049016a8ef77de84911f85f7e397c8d88d0a84cf`
- Remote ref: `my-origin/feature/mint-env-reformulation-v1-visual-fidelity`
- Required evidence origin-visible: `True`
- Post-push verification artifact: `experiments/mint/mint_drawer_v1/runtime/v11_g4_goc_v3_contact_aware_teacher_synthesis_20260501T134626Z/post_push_verification.json`
- Closeout classification: `COLLISION_GEOMETRY_PAD_MODEL_INSUFFICIENT`
- Next gate: `GOC_V4_DEDICATED_FINGER_PAD_COLLISION_MODEL_REPAIR`

Metadata note: `closeout_decision.json` records the evidence commit that was already pushed before this metadata update. The final branch HEAD after committing this metadata can be verified from git/origin.



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

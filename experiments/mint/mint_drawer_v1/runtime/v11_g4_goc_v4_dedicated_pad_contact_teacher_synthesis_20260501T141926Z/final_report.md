
# V11-G4 GOC-v4 Dedicated Finger-Pad Contact-Aware Teacher Synthesis

Closeout: **DEDICATED_PAD_CONTACT_AWARE_PROBE_FAILED**

Prior dynamic failure mode: `C_IK_SETPOINT_OR_CLOSE_COMMAND_STARTS_INSIDE_HANDLE`.
Contact geometry quality: `dedicated_finger_pad_collision`. GOC-v4 generated: `True`. Dedicated pad IDs: `[98, 99]`.

A guarded contact mode controller and force/penetration-limited probe controller were built. CEM/MPPI-style optimization was attempted with `5` iterations and `16` samples per iteration.

Best probe:

- target_contact_frames: `19`
- target_contact_max_consecutive_frames: `19`
- forbidden_contact_frames: `13`
- max_penetration_m: `0.05002065754481153`
- max_force_n: `17030.84100901885`
- max_drawer_fraction: `0.0`

Bounded rollout attempted: `False`.
Strict candidate found: `False`.

Current truth modified: `false`. Next actions modified: `false`. GOC-v4 authority changed: `false`.

Next gate: **ACTUATOR_INTERFACE_OR_OPERATIONAL_SPACE_CONTROL_REPAIR**

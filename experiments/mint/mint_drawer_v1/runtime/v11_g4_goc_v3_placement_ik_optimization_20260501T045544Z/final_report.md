# V11-G4 GOC-v3 Placement IK Optimization

Closeout: **PLACEMENT_IK_DYNAMIC_CONTACT_PROBE_FAILED**

The prior best row ambiguity is resolved: `prior_best_row_reset_ok=False`. The prior row had reset counts `{'forbidden': 29, 'handle_nonlegal': 0, 'max_contact_force_n': 3.6419876285087264e+17, 'max_penetration_m': 0.34626260236300876, 'other': 6, 'target': 9}`, so the earlier best IK values were not jointly feasible with reset clearance.

This phase evaluated `720` continuous/local placement plus initial-qpos candidates under per-instance GOC bindings. It found `1` reset-clean joint-feasible candidate:

- seed: `13`
- base_pos: `[-0.875, 0.05, 0.0]`
- yaw_deg: `-10.0`
- qpos_index: `4`
- legal_pad_to_handle_ik_m: `0.022194251967419466`
- eef_to_handle_ik_m: `0.03989097505439188`
- reset_contact_counts: `{'forbidden': 0, 'handle_nonlegal': 0, 'max_contact_force_n': 0.0, 'max_penetration_m': 0.0, 'other': 0, 'target': 0}`

A short dynamic contact probe was attempted. It produced exact GOC-v3 target contact frames, with no forbidden contact frames, but failed the physical plausibility gate because penetration exceeded the 0.02 m threshold:

- target_contact_frame_count: `103`
- forbidden_contact_frame_count: `0`
- max_contact_force_n: `302624.86704762443`
- max_penetration_m: `0.31483595862277625`
- dynamic_probe_passed: `False`

Bounded rollout was not attempted. Runtime patch applied: `false`. Current truth modified: `false`. Next actions modified: `false`.

Next gate: **FINGER_PAD_DYNAMIC_ROUTE_POLICY_REPAIR**

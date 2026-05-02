# V11-G4 GOC-v4 Model Placement/Keepout Overnight Closeout

Closeout: `MODEL_PLACEMENT_KEEP_OUT_REPAIR_STALLED_WITH_SCHEMA_AWARE_CERTIFICATE`

This campaign was bound to the overnight model/placement/keepout repair spec and ran from the clean schema-aware infeasibility point.
It repaired/search-expanded the coupled pre-contact feasibility layer before any promotion to corridor, dynamic probe, Layer4R, or pull rollout.

- harness preflight passed: `None`
- task spec lock bound: `None`
- schema-aware binding preserved: `True`
- model/placement repair cycles run: `4`
- endpoint feasible candidate found: `False`
- feasible seed/case count: `0` / `0`
- best reset-clean two-pad residual m: `0.0011419128005339195`
- remaining endpoint failure clusters: `{'contact_hold_forbidden_contact': 2860, 'contact_hold_ik_or_contact_constraints_failed': 2860, 'contact_hold_penetration_gt_threshold': 2860, 'guarded_forbidden_contact': 2860, 'guarded_ik_or_contact_constraints_failed': 2860, 'guarded_penetration_gt_threshold': 2860, 'pregrasp_forbidden_contact': 2860, 'pregrasp_ik_or_contact_constraints_failed': 2860, 'pregrasp_penetration_gt_threshold': 2860}`
- approach corridor verified: `False`
- dynamic probe attempted/passed: `False` / `False`
- Layer4R matrix attempted: `False`
- Layer4R passed/failed: `0` / `0`
- next gate: `MODEL_OR_PLACEMENT_REPAIR_WITH_SCHEMA_AWARE_INFEASIBILITY_CERTIFICATE`

## Post-Push Verification
- evidence_commit_hash: 7ab8f067ff341cfa8a06587808a6d3cb9cbfbe3a
- origin_head_at_verification: 7ab8f067ff341cfa8a06587808a6d3cb9cbfbe3a
- origin_matches_local_head: True
- all_required_paths_origin_visible: True

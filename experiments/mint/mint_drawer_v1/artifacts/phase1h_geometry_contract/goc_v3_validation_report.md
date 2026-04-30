# GOC-v3 Validation Report

Generated UTC: `2026-04-30T12:45:25Z`

All invariants passed: `True`

## Summary Counts

- legal_gripper_surface_count: 3
- forbidden_robot_surface_count: 5
- drawer_handle_surface_count: 9
- unknown_contact_relevant_geom_count: 0
- visual_only_or_noncontact_count: 50

## Invariants

- I1: PASS. Every contact-relevant geom has exactly one non-unknown primary_owner. Offenders: []
- I2: PASS. Every target contact geom is an exact current geom_id. Offenders: []
- I3: PASS. legal_gripper_surface and forbidden_robot_surface are disjoint. Offenders: []
- I4: PASS. drawer_handle and robot_arm are disjoint. Offenders: []
- I5: PASS. drawer_handle and gripper are disjoint. Offenders: []
- I6: PASS. visual_only_or_noncontact geoms cannot be legal target contact. Offenders: []
- I7: PASS. unknown geoms cannot participate in target contact. Offenders: []
- I8: PASS. unknown geoms cannot participate in forbidden contact classification. Offenders: []
- I9: PASS. Target contact is exactly legal_gripper_surface_geom_ids <-> drawer_handle_geom_ids by contract rule. Offenders: []
- I10: PASS. Forbidden contact includes forbidden_robot_surface against drawer_handle/drawer_body_or_cabinet by contract rule. Offenders: []
- I11: PASS. drawer_body_or_cabinet cannot be counted as gripper contact. Offenders: []
- I12: PASS. Robot link body/collision cannot be counted as drawer handle. Offenders: []
- I13: PASS. GOC-v3 exact-ID sets explain GOC-v2 29/26 versus runtime 31/27. Offenders: []
- I14: PASS. GOC-v3 contains no count-only authority in exact sets. Offenders: []
- I15: PASS. GOC-v3 is generated from current model inventory, not copied from old artifacts. Offenders: []

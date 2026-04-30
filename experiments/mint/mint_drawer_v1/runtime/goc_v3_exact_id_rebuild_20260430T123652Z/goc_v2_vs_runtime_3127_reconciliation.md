# GOC-v2 29/26 vs Runtime 31/27 Reconciliation

Generated UTC: `2026-04-30T12:45:25Z`

GOC-v2 stored count/body-level expectations while the runtime derived body-owned sets from the current model. GOC-v3 resolves the ambiguity by exact current geom IDs and contact roles.

## Exact Deltas Requested

- legal IDs in 29 not 31: `[]`
- legal IDs in 31 not 29: `[63, 81]`
- forbidden IDs in 26 not 27: `[]`
- forbidden IDs in 27 not 26: `[59]`

## Current GOC-v3 Authority

- legal_gripper_surface_geom_ids: `[63, 81, 90]`
- forbidden_robot_surface_geom_ids: `[45, 47, 49, 54, 59]`
- drawer_handle_geom_ids: `[0, 1, 2, 3, 4, 5, 6, 7, 8]`
- drawer_body_or_cabinet_geom_ids: `[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]`

## Cause Classification

`['B_visual_or_noncontact_geom_inclusion', 'C_stale_artifact', 'E_classifier_bug_count_only_body_semantics']`

The old count fields are not future authority. Exact-ID GOC-v3 is the authority for the next V11-G4 task rewrite.

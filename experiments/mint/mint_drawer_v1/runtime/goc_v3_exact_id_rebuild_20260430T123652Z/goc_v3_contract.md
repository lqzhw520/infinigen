# GOC-v3 Exact-ID Geometry Ownership Contract

Contract ID: `GOC_V3_EXACT_ID_GEOMETRY_OWNERSHIP_CONTRACT`
Version: `v3`
Generated UTC: `2026-04-30T12:45:25Z`
Source model commit: `7a7d5c43748822d9c38d948c6d1d39ec2895fe87`

## What GOC-v3 Is

GOC-v3 is the exact per-geom authority for future V11-G4 contact classification. It binds target contact and forbidden contact to concrete MuJoCo `geom_id` sets from the current merged Panda + Infinigen drawer model.

It is not a rollout result, not a visual artifact claim, not a training eligibility claim, and not a MINT success claim.

## Exact Contact Authority

- Legal gripper target surfaces: `[63, 81, 90]`
- Drawer handle target surfaces: `[0, 1, 2, 3, 4, 5, 6, 7, 8]`
- Forbidden robot contact surfaces: `[45, 47, 49, 54, 59]`
- Drawer body/cabinet surfaces: `[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]`
- Visual-only/noncontact geoms: `[33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 48, 50, 51, 52, 53, 55, 56, 57, 58, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89]`
- Unknown geoms: `[]`

Future target contact is valid only for:

```text
legal_gripper_surface_geom_ids <-> drawer_handle_geom_ids
```

Forbidden contact includes:

```text
forbidden_robot_surface_geom_ids <-> drawer_handle_geom_ids
forbidden_robot_surface_geom_ids <-> drawer_body_or_cabinet_geom_ids
nonlegal robot contact surfaces <-> drawer_handle/body/cabinet
```

## Why GOC-v2 Was Insufficient

GOC-v2 described ownership with body-level counts: legal_pad=29, forbidden=26, handle=9. The current runtime body-based derivation sees legal=31 and forbidden=27 because it includes every geom on the named robot bodies. That body-count logic is not exact contact authority.

GOC-v3 resolves this by listing exact IDs and separating contact-capable geoms from visual-only/noncontact geoms.

## 29/26 vs 31/27 Reconciliation

- Runtime legal body-based 31: `[60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]`
- Diagnostic GOC-v2 legal 29 reconstruction: `[60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90]`
- IDs in 31 but not 29: `[63, 81]`
- IDs in 29 but not 31: `[]`

- Runtime forbidden body-based 27: `[33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]`
- Diagnostic GOC-v2 forbidden 26 reconstruction: `[33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]`
- IDs in 27 but not 26: `[59]`
- IDs in 26 but not 27: `[]`

The old artifacts also contain stale/count inconsistencies, so these diagnostic reconstructions are not future authority. The future authority is the exact GOC-v3 contact-role sets above.

## Validation

All invariants passed: `True`

## Future Runtime Requirement

Future V11-G4 runtime/contact reports must load `artifacts/phase1h_geometry_contract/goc_v3_contract.json` and classify target/forbidden contacts using exact geom IDs. It may not use GOC-v2 counts and may not use body-based 31/27 without exact-ID role classification.

## Still Not Proven

- No Phase 1H rollout was run.
- No render or visual artifact was generated.
- No teacher rollout, replay, training, fine-tuning, or evaluation success is claimed.
- This contract only proves exact current model geometry ownership for future task rewrite and contact-report validation.

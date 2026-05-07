# Controller-Preserving Reference Topology Boundary Report

- source_head: 259388bb8ee565b3d1e5c703f96438f6b619440d
- topology target: solid drawer front, short-stub spherical knob, complete moving drawer box/tray, visible guide/support semantics
- controller_algorithm_modified: false
- result: reference topology oracle passed, but frozen controller migration did not pass fast_guarded_contact
- next_gate: REFERENCE_TOPOLOGY_CONTROLLER_PULL_WORK_REPAIR_UNDER_TOPOLOGY_LOCK

## Key Evidence
- high_opening_but_forbidden: reference_aligned_solid_front_c35_fg_cycle2_05_d0p9_02_padpatch_00_current_single_pad_baseline_v3_pull_trajectory_sh_c00_s00 fraction=0.938685986006102 bilateral_pull=0 forbidden=1334 penetration=0.0056358686336666595
- best_latest_legal_opening: reference_aligned_solid_front_c43_fg_cycle2_05_d0p9_02_padpatch_00_current_single_pad_baseline_v3_pull_trajectory_sh_c00_s00 fraction=0.10804562958860969 bilateral_pull=0 forbidden=0 penetration=0.001031794098186073
- best_latest_exact_pull_legal: reference_aligned_solid_front_c73_fg_cycle2_05_d0p9_02_padpatch_00_current_single_pad_baseline_v3_pull_trajectory_sh_c00_s00 fraction=0.10201934319190892 bilateral_pull=155 forbidden=0 penetration=0.004413283460008466

## Interpretation
The topology-only phase no longer supports blaming hollow/front-frame geometry: selected candidates satisfy the reference topology oracle. The remaining blocker is migration of pull work and keep-out posture for the already established round-knob exact latch-pull family on the restored solid-front/tray topology. Continuing with action-only replay or further random topology-only sweeps would be scientifically mis-scoped.

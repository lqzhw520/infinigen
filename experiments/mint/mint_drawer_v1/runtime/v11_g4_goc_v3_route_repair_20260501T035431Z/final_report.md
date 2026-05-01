# V11-G4 GOC-v3 Route Repair Closeout

closeout_classification: `ROUTE_STILL_NO_GOC_V3_TARGET_CONTACT`

## Harness

- route repair spec: `experiments/mint/mint_drawer_v1/sovereign/experiment_specs/v11_g4_goc_v3_route_repair_autonomous.yaml`
- production lock bound to route repair spec: true
- preflight passed with no bypass: true
- source_worktree_head for telemetry: `251714b2b52f8f8b296997e131c5ce2893b359a1`
- runtime patch applied: false

## Telemetry answers

1. EEF trajectory approached handle: False
2. Minimum EEF-to-handle distance: 0.134439 m
3. Minimum legal-pad-to-handle distance: 0.142450 m
4. Diagnostic direct robot-qpos IK best EEF-to-handle distance: 0.077817 m
5. Diagnostic direct robot-qpos IK best legal-pad-to-handle distance: 0.069338 m
6. Target contact found in route telemetry: False
7. Forbidden contact found in route telemetry: False

## Prior mv2 force explanation

The previous mv2 high force was replayed from its qpos trace. The max-force pair is not legal-pad-to-handle contact. It is robot self/contact geometry, e.g. `{'category': 'other', 'contact_index': 0, 'dist_m': -0.002043706532265573, 'geom1': 45, 'geom1_body': 'link0', 'geom1_name': 'link0_collision', 'geom2': 90, 'geom2_body': 'link7', 'geom2_name': 'link7_collision', 'normal_force_n': 1784.6540649391907}`. This explains why a 2500N-class force/penetration did not count as GOC-v3 target contact: the pair is not `[63,81,90] <-> [0..8]` and does not involve the drawer handle exact target set.

## Seed/model-instance exact-ID issue

The fixed GOC-v3 ID set is not semantically stable across the attempted drawer seeds. Seed 1 maps ID 90 to `link7_collision`, but seeds 11/12/13 map `link7_collision` to different IDs and fixed ID 90 becomes unnamed noncontact. This does not mutate GOC-v3 authority in this phase, but it blocks honest positive candidate generation across multiple model instances.

```json
[
  {
    "seed": 1,
    "ngeom": 91,
    "fixed_id_90_name": "link7_collision",
    "fixed_id_90_contype": 1,
    "fixed_id_90_conaffinity": 1,
    "link7_collision_actual_id": 90
  },
  {
    "seed": 11,
    "ngeom": 97,
    "fixed_id_90_name": "",
    "fixed_id_90_contype": 0,
    "fixed_id_90_conaffinity": 0,
    "link7_collision_actual_id": 96
  },
  {
    "seed": 12,
    "ngeom": 94,
    "fixed_id_90_name": "",
    "fixed_id_90_contype": 0,
    "fixed_id_90_conaffinity": 0,
    "link7_collision_actual_id": 93
  },
  {
    "seed": 13,
    "ngeom": 98,
    "fixed_id_90_name": "",
    "fixed_id_90_contype": 0,
    "fixed_id_90_conaffinity": 0,
    "link7_collision_actual_id": 97
  }
]
```

## Closeout

No strict candidate is claimed. No runtime patch was made. No current_truth or next_actions mutation was made. The next gate is `ROUTE_CONTROLLER_AND_MODEL_INSTANCE_ID_BINDING_REPAIR`: repair the route/controller binding and either bind GOC-v3 to a specific model instance/seed or add a governed per-instance exact-ID remapping before positive candidate attempts.

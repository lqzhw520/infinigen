
# V11-G4 GOC-v3 Instance-Bound Route Synthesis

The phase generated per-model-instance semantic GOC bindings and rejected fixed global ID reuse unless the instance proved the same semantic surfaces. It then searched robot base placement/yaw variants with reset contact and diagnostic IK gates before any dynamic probe.

Closeout: **MODEL_INSTANCE_REACHABILITY_BLOCKER**

Best diagnostic IK legal-pad-to-handle distance: `0.008132436607010393` m. Best EEF-to-handle distance: `0.027836743078149474` m.

Runtime patch applied: `false`. GOC-v3 authority changed: `false`. Direct current_truth/next_actions mutation: `false`.

Next gate: **MODEL_INSTANCE_PLACEMENT_OR_KINEMATIC_CHAIN_REPAIR**

Post-push evidence commit: `6258567b13843d3d178591d20ad9f4981d488fd3`. Origin-visible required evidence: `True`.

## Best Row Gate Audit

The best IK row was not feasible because `reset_ok=false`: 29 forbidden contacts, max penetration `0.34626260236300876` m, and max reset contact force `3.6419876285087264e+17` N. Recomputed feasible rows: `0`. The refined interpretation is: IK closeness and reset clearance were not jointly feasible under the searched base/yaw grid.

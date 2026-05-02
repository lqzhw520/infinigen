# V11-G4 GOC-v4 Joint Placement / Reset / IK / Keepout Feasibility Closeout

Closeout: `PLACEMENT_RESET_IK_JOINT_FEASIBILITY_FAILED`

This phase tested a coupled feasibility claim before any Layer5 pull rollout:
base placement, reset qpos, two-pad handle-frame IK, and approach-corridor
keepout must be jointly satisfiable under GOC-v4 dedicated finger-pad authority.

- harness preflight passed: `True`
- task spec lock bound: `True`
- semantic bindings generated: `True`
- handle frames generated: `True`
- joint feasibility solver built: `True`
- feasible seed count: `0`
- feasible case count: `0`
- best reset-clean two-pad residual m: `0.017613888435584424`
- best overall two-pad residual m: `0.017613888435584424`
- approach corridor verified: `False`
- dynamic probe attempted: `False`
- dynamic probe passed: `False`
- Layer4R full matrix attempted: `False`
- Layer4R passed/failed: `0` / `0`
- remaining failure clusters: `{}`
- next gate: `MODEL_OR_PLACEMENT_REPAIR_WITH_INFEASIBILITY_CERTIFICATE`

Layer5 pull rollout, export, local replay/render, and MINT training were not run.

## Post-Push Verification

- evidence commit: `26145df2ff1c3601c7f2e726362b01341ecdf77b`
- origin-visible evidence: `True`
- finalization commit containing this post-push note is created after this file is updated.


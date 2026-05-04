# V11-G4 GOC-v4 Physical Accessibility Instance Pool Closeout

Closeout: `ACCESSIBLE_INSTANCE_POOL_READY_FOR_LAYER4R`

This phase did not run Layer4R, teacher rollout, export, local replay/render, or
MINT training. It built and certified a physically accessible instance pool for
the future GOC-v4 line.

Key results:

- harness preflight passed: `True`
- task spec lock bound: `True`
- physical accessibility invariant written: `True`
- accessibility oracle built: `True`
- candidate instances total: `31`
- accepted accessible instances: `5`
- rejected inaccessible instances: `26`
- accepted IDs: `['generated_knob_drawer_accessible_001', 'generated_knob_drawer_accessible_002', 'generated_knob_drawer_accessible_003', 'generated_knob_drawer_accessible_004', 'generated_knob_drawer_accessible_005']`
- rejection histogram: `{'two_pad_ik_residual_blocks_layout': 22, 'ACCEPTED': 5, 'two_pad_reachable_but_forbidden_keepout_blocks_layout': 4}`
- existing/repaired/generated accept counts: `0` / `0` / `5`
- fake collision demotion used: `False`
- next gate: `LAYER4R_ON_ACCESSIBLE_INSTANCE_POOL`

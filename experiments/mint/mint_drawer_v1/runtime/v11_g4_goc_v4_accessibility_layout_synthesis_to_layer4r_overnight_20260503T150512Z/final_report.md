# V11-G4 GOC-v4 Accessibility Layout Synthesis To Layer4R Closeout

Closeout: `ACCESSIBILITY_REJECTION_INVARIANT_READY`

This phase attempted to clear the structural predecessor to Layer4R by testing whether a physically legal full-robot layout exists for reset clearance, dedicated two-pad handle-frame endpoint IK, forbidden keepout, approach corridor, short dynamic contact, and Layer4R certification.

- harness preflight passed: `True`
- task spec lock bound: `True`
- prior evidence ingested: `True`
- seeds total: `13`
- endpoint candidates evaluated: `19760`
- endpoint feasible cases/seeds: `0` / `0`
- best reset-clean two-pad residual m: `0.00014165405097371712`
- best overall two-pad residual m: `0.00014165405097371712`
- rejection invariants: `{'two_pad_ik_residual_blocks_layout': 10, 'two_pad_reachable_but_forbidden_keepout_blocks_layout': 3}`
- corridor attempted/passed: `False` / `False`
- dynamic attempted/passed: `False` / `False`
- Layer4R full matrix attempted: `False`
- Layer4R passed/failed: `0` / `0`
- next gate: `MODEL_OR_LAYOUT_REDESIGN_WITH_PHYSICAL_ACCESSIBILITY_INVARIANT`

Layer5 pull rollout, export bundle, local replay/render, and MINT training were not run.

## Commit And Push

- committed: `true`
- pushed_to_origin: `true`
- science_evidence_commit_hash: `101fd0313f1501b691fc16bd00b795eb5930f8b2`
- post_push_verification: `experiments/mint/mint_drawer_v1/runtime/v11_g4_goc_v4_accessibility_layout_synthesis_to_layer4r_overnight_20260503T150512Z/post_push_verification.json`

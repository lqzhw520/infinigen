# V11-G4 GOC-v4 Schema-Aware Model Placement Feasibility Repair Closeout

Closeout: `SCHEMA_AWARE_STAGE3_MODEL_PLACEMENT_INFEASIBLE`

This phase repaired the model-instance joint/qpos/qvel/actuator binding issue that previously polluted seed 7 and seed 15. The Stage3 feasibility matrix was rerun from committed schema-aware runtime code across 1040 endpoint candidates.

Key results:

- harness preflight passed: `True`
- task spec lock bound: `True`
- schema-aware joint binding repaired: `true`
- all mandatory seeds Stage3 without indexing errors: `True`
- schema error count: `0`
- feasible seed count: `0`
- feasible case count: `0`
- best reset-clean two-pad residual m: `0.017613888435584424`
- best overall two-pad residual m: `0.017613888435584424`
- approach corridor verified: `False`
- dynamic probe attempted: `False`
- Layer4R full matrix attempted: `False`
- next gate: `MODEL_OR_PLACEMENT_REPAIR_WITH_SCHEMA_AWARE_INFEASIBILITY_CERTIFICATE`

Interpretation: the prior schema/indexing evidence blocker is fixed. The remaining blocker is not helper indexing; it is a clean schema-aware model/placement/keepout infeasibility signal under the bounded Stage3 search. The best near-residual candidates still require forbidden contact and excessive penetration with zero target contact, so corridor, dynamic probe, Layer4R, Layer5 rollout, export, local replay/render, and MINT training remain blocked.

Post-push verification:

- evidence commit hash: `1b7c31a2c86b32993fe4ca14434921f72feedfc5`
- all origin-visible: `True`

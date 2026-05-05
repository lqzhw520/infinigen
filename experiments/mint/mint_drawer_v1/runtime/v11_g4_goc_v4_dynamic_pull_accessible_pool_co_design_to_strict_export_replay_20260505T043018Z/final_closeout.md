# Dynamic Pull Accessible Pool Co-Design Closeout

- closeout_classification: `TARGETED_DYNAMIC_PULL_POOL_SHARD_FAILED`
- next_gate: `ANALYZE_DYNAMIC_PULL_POOL_CO_DESIGN_DEFECT_CERTIFICATES`
- candidate_pool_total: `32`
- candidate_pool_admitted_count: `0`
- targeted_shard: `0/0` because no candidate passed O1-O6 admission
- full30_attempted: `False`
- strict_export_complete: `False`
- local_replay_render_passed: `False`

Best overall dynamic opening was `0.986625089646256` on `dynamic_generated_low_y_offset_003`, but it failed strict admission with `33` forbidden-contact frames.

Best keepout-legal opening was `0.5253545174648184` on `dynamic_keepout_old002_yneg`, but it still failed strict admission as `TWO_PAD_RETENTION_FAILED`.

Dominant candidate failure histogram: `{'FULL_BODY_KEEP_OUT_FAILED': 4, 'GUARDED_IK_INFEASIBLE': 5, 'APPROACH_CORRIDOR_BLOCKED': 4, 'PULL_WRENCH_INSUFFICIENT': 3, 'TWO_PAD_RETENTION_FAILED': 13, 'DRAWER_QPOS_NONMONOTONIC': 3}`.

The prior fixed pool was used only as a negative baseline; it was not modified and was not promoted to strict export.

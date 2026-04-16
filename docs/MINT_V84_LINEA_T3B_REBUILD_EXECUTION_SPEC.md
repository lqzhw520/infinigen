Current authoritative status:
- Sovereign top-level head: `fea75fb436b40bd5f854d7a74063b8ecff663cd4`.
- `P1B` is frozen and no longer an active redesign target.
- `T3A` fidelity is contract-backed and currently exact relative to the frozen reference.
- `T3B` rebuild round 3 improves materially over rounds 1-2: seed2/4 rise to `0.465/0.386`, `attach_persistence` rises to `64/65`, and `hybrid_open_hold` roughly doubles, but `matched_superiority_over_t3a` still fails and Line A remains unresolved.
- The current causal diagnosis is now sharper: `T3B` open-phase step authority is already strong enough; the unresolved object is partially-open continuation, i.e. reattach/relock failure after the first opening burst.

# MINT v8.4 Line A — T3B Rebuild Execution Spec

**Question**
> Can reset teachers on hard seeds approach the frozen hard-seed frontier under the current regime, specifically by restoring continuation after the first partially-open burst?

## Frozen objects
- Freeze latest validated `P1B` as frontier reference.
- Freeze frozen `v8.3 X2` reset baseline as `T3A` reference.
- Do not change acceptance thresholds.
- Do not touch truth-contract or runtime logic in this line.

## Required repo-level work
1. Patch `scripts/mint/run_v84_teacher_abstraction_full.py`
   - add precise `T3A` fidelity comparison against frozen reference with epsilon tolerance
   - add frozen frontier loading and `gap_to_frontier` fields
   - mark pairwise comparison invalid if `T3A` fidelity fails
2. Patch `scripts/mint/drawer_robot_env_mujoco.py`
   - keep `T3A` and `T3B` matched except for opening law
   - rebuild `T3B` as a true interaction-frame hybrid controller

## Minimal faithful T3B structure
- explicit local frame: `t`, `n`, `b`
- explicit controller states: `s_t`, `s_n`, `s_b`, `lock_score`, `reseat_budget_remaining`
- explicit phases:
  - `pregrasp`
  - `contact`
  - `close`
  - `grasp_seat`
  - `interaction_lock`
  - `hybrid_open`
  - `reseat_once`
  - `hybrid_open_final`
  - `retreat`
- target form:
  - `x_target = handle + s_t * t + s_n * n + s_b * b + z_offset`

## Required telemetry
- `t3b_s_t_trace`
- `t3b_s_n_trace`
- `t3b_s_b_trace`
- `t3b_lock_score_trace`
- `t3b_phase_trace`
- `t3b_reseat_budget_used`
- `t3b_plateau_reason`
- `gap_to_frontier_seed2`
- `gap_to_frontier_seed4`
- `gap_closed_fraction_seed2`
- `gap_closed_fraction_seed4`

## Required run order
```bash
python scripts/mint/run_v84_teacher_abstraction_full.py --phase p1a
python scripts/mint/run_v84_teacher_abstraction_full.py --phase p1b
python scripts/mint/run_v84_teacher_abstraction_full.py --phase t3a
python scripts/mint/run_v84_teacher_abstraction_full.py --phase t3b --resume-from t3a
```

## Allowed verdicts
1. `RESET_ABSTRACTION_STILL_PRIMARY`
2. `RESET_TO_FRONTIER_MOSTLY_CLOSED_BUT_FRONTIER_BELOW_ACCEPTANCE`
3. `RESET_SUPPLY_RESTORED_FOR_HARD_SEEDS`

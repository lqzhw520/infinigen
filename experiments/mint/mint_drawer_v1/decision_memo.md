# MINT Drawer Robot-Trajectory Campaign Decision Memo

**Updated**: 2026-03-26T20:00:00+08:00
**Phase**: `mint_robot_trajectory_claim_push`
**Gate**: `await_train_reproducibility`
**Verdict**: `mainline_recovery_in_progress`

## Claim Assessment

The current blocker is no longer the old evaluator bug. The live risk is a combination of:
- contaminated D1 evidence from concurrent control,
- weak upstream handle semantics,
- and mainline D1 selection needing clean grasp-rich candidates instead of pull-dominant ones.

The automatic route is still alive, but only under a clean single-controller rerun.

## Risk Register

### Confirmed Fixed

- `Layer 0` evaluator attach-state restoration bug is fixed; D1 is no longer structurally impossible.
- The old `D2/D3/E1` gate drift has been diagnosed and should not be reused as final claim evidence.

### Confirmed Open

- Concurrent control contamination: multiple loops / D1 wrappers / `lerobot-train` runs have overlapped and invalidated post-`2026-03-26T18:26+08:00` D1 evidence.
- `u3` handle semantics remain weak: missing handle part labels and large heuristic offsets still destabilize attach behavior.
- Strict replay remains fully failing; learnability can proceed via fallback, but scientific cleanliness is still incomplete.
- Mainline D1 still needs clean staged validation on grasp-rich candidates (`seed_010_episode_01`, `seed_002_episode_04`).
- Historical rollout drift: the archived diagnostic `seed_010_episode_04` that showed positive D1 signal is not the same rollout as the current `c2_replay_valid_rollouts/seed_010_episode_04.npz`. The archived copy is `96` steps with phase counts `{grasp: 15, hold_close: 10, pull: 7}` and `handle_distance_min ~= 0.0997`, while the current copy is `80` steps with `{grasp: 1, hold_close: 10, pull: 2}` and `handle_distance_min ~= 0.0250`.
- Missing rollout-lineage contract: prior plans treated `(seed, episode_index)` as a stable teacher identity across rebuilds. That was wrong. Historical comparisons, D1 ranking discussions, and checkpoint reuse arguments must be keyed by immutable rollout fingerprint, not by seed/episode name alone.

### Watch But Not Primary

- Render / camera report quality is still weak and should not be confused with task success.
- Diagnostic winners from `seed_009_*` remain useful evidence, but they are not valid mainline seeds for D2.
- Handle-distance-only ranking is not yet proven. The strongest apparent support mixed metrics from different versions of `seed_010_episode_04`, so `handle_distance_min` is currently treated as an important attachment signal, not a standalone learnability proof.

## Engineering Failure Modes

- Loop alive/dead mismatch: supervisor can die while queue still looks active.
- Artifact truth vs dashboard truth drift: artifact may finish while `state/watch/campaign_status` stay stale.
- Concurrent controller drift: a second loop can restart D1 and contaminate live evidence unless a lease is enforced.
- Gate misuse: diagnostic D1 winners or single-rollout seeds can be pushed into D2 if hard gating is not enforced.

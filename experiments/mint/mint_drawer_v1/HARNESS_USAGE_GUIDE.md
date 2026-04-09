# Harness v2 — mint_drawer_v1
**GENERATED FROM current_truth.json — NOT CANONICAL**

- Generated at: `2026-04-09T16:45:04+08:00`
- Source: `sovereign/current_truth.json`
- Stale policy: regenerate with `python scripts/harness/sovereign_cli.py go`

## At A Glance
- Verdict: `V59_MUJOCO_PILOT_ESTABLISHED — OFFICIAL_LIBERO_BASELINE_REPRO_NEXT`
- Phase: `v59_MUJOCO_PILOT_PHASE3 — official LIBERO drawer baseline reproduction`
- Decision: `V59_MUJOCO_PILOT_ESTABLISHED — OFFICIAL_LIBERO_BASELINE_REPRO_NEXT`
- Human read: We are in `v59_MUJOCO_PILOT_PHASE3 — official LIBERO drawer baseline reproduction`. The current top priority is `libero_goal_drawer_official_baseline_repro`: Reproduce official MINT LIBERO drawer baseline on an in-distribution drawer/cabinet task
- Dataset anchor: `v59_20260405` / `240` episodes / `19701` frames / loads=`True`
- Workspace: branch `feature/mint-integration` @ `4253ba45f3951c828c3482cb42c118e409702125` / dirty files `10`
- MINT fidelity: `F2` — Runnable but semantically drifted

## Claims Driving This Phase
- No current-driving claims were classified.

## Historical Context Claims
- Historical context claims currently suppressed from the main dashboard: `15`
- `C_PHYSICS_LEGAL_TEACHER_HISTORICAL_SUMMARY` [supported] / scope `historical_context_only`
- `C_STATE_CONTRACT_HISTORICAL_SUMMARY` [supported] / scope `historical_context_only`
- `C_VISION_ALIGNMENT_HISTORICAL_SUMMARY` [supported] / scope `historical_context_only`
- `C_IMAGE_NEAR_WHITE` [supported] / scope `v58_vision_pipeline`
- `C_IMAGE_NEAR_WHITE@2` [supported] / scope `v58_vision_pipeline`
- ... plus `10` more historical-context claims

## What Is Settled Right Now
- `E034`: Forward-pass recheck after E033 shows mixed, nonzero state sensitivity on real LIBERO observations; strong state-invariant/image-only diagnosis is not supported.
- `E033`: MINT key-remap fix verified. Large 165 missing / 164 unexpected mismatch collapses to 1 intentional tied-weight miss; forward-pass diagnostics must be rechecked afterward.
- `E031`: Against PyBullet dataset: raw_joint_pos mean_l2_delta=4.23, pi_minus_joint_pos=9.77. pi_minus NOT better. Reference is PyBullet dataset (wrong baseline for LIBERO comparison).
- `E030`: MINT on LIBERO drawer: 0% success for both raw_joint_pos and pi_minus_joint_pos. action_l2≈1.0 dominated by gripper=-1. Pipeline not clean: LM weights missing, image_size mismatch.
- `E029`: State-space oracle 3/3. Direct qpos manipulation drives drawer from closed to open_qpos. Does NOT prove action-space reachability.

## What Is Still Open
- Open gate `gate_b_teacher_replayability` [P0]: Verify teacher actions replay successfully in DrawerRobotEnv (top-10 episodes)
- No canonical blockers are registered.

## What To Read First In A New Session
1. `python scripts/harness/sovereign_cli.py go`
2. `sovereign/current_truth.json`
3. The latest verified evidence listed in `current_truth.json`
4. The spec for the current next action, if the next action is spec-backed

## Tonight / Next Safe Move
- Canonical next action: `MINT_LIBERO_OFFICIAL_BASELINE_REPRO` / `libero_goal_drawer_official_baseline_repro` / priority `P0`
- Target: Reproduce official MINT LIBERO drawer baseline on an in-distribution drawer/cabinet task
- Scope: Use the official LeRobot LIBERO evaluation path on an official MINT evaluation-suite drawer task before drawing capability conclusions from the MuJoCo pilot.

## Canonical Rules
- Tier 0 files are authoritative; generated docs are convenience views only.
- If a generated doc conflicts with `current_truth.json`, regenerate and trust `current_truth.json`.
- `state.json` is runtime-only. It is not the place to carry long scientific narrative.
- Only spec-backed deterministic experiments are allowed to auto-publish scientific updates.

## What Not To Trust
- Do not treat bootstrap/handoff/usage guide as canonical truth.
- Do not promote inconclusive raw artifacts to eliminated/confirmed narrative.
- Do not read old Cursor summaries as current project truth.
- Do not interpret the current campaign as evaluating upstream-faithful MINT; check `model_load_fidelity.json` first.

## Quick Files
- `sovereign/current_truth.json` — single generated truth surface for current status
- `sovereign/next_actions.json` — canonical action queue
- `sovereign/model_load_fidelity.json` — whether current MINT runtime is scientifically comparable
- `sovereign/run_ledger.yaml` — append-only process log
- `sovereign/experiment_specs/` — what night runner is allowed to auto-publish

## Claim Debt To Clean Later
- Claims with debt flags: `5`
- `C_IMAGE_NEAR_WHITE` — scope_not_current_phase
- `C_IMAGE_NEAR_WHITE@2` — scope_not_current_phase
- `C_DATA_INTEGRITY_AND_P0B_PRIORITY` — scope_not_current_phase
- `C_GRIPPER_JOINT_NOT_FROZEN_IN_V58` — scope_not_current_phase
- `C_V58_TRAIN_vs_LIVE_DATA` — scope_not_current_phase

## Claim Lifecycle Review Queue
- Canonical cleanup candidates now: `0`
- none

## Generated Doc Health
- Stale derived docs: `0`
- none

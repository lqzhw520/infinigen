# Harness v2 — mint_drawer_v1
**GENERATED FROM current_truth.json — NOT CANONICAL**

- Generated at: `2026-04-09T16:42:29+08:00`
- Source: `sovereign/current_truth.json`
- Stale policy: regenerate with `python scripts/harness/sovereign_cli.py go`

## At A Glance
- Verdict: `V59_MUJOCO_PILOT_ESTABLISHED — OFFICIAL_LIBERO_BASELINE_REPRO_NEXT`
- Phase: `v59_MUJOCO_PILOT_PHASE3 — official LIBERO drawer baseline reproduction`
- Decision: `V59_MUJOCO_PILOT_ESTABLISHED — OFFICIAL_LIBERO_BASELINE_REPRO_NEXT`
- Human read: We are in `v59_MUJOCO_PILOT_PHASE3 — official LIBERO drawer baseline reproduction`. The current top priority is `gate_b_teacher_replayability`: Verify teacher actions replay successfully in DrawerRobotEnv (top-10 episodes)
- Dataset anchor: `v59_20260405` / `240` episodes / `19701` frames / loads=`True`
- Workspace: branch `feature/mint-integration` @ `0ac4f14d77ca6a1c0df23a413f728a2f3cbf8c98` / dirty files `61`
- MINT fidelity: `F2` — Runnable but semantically drifted

## Claims Driving This Phase
- `C_V59_OFFLINE_OVERFIT_SIGNAL` [supported]: V59 finetuned checkpoint shows better offline action imitation than pretrained on the training subset. Evidence: 200 steps on 494 frames, loss 6.520→3.741 (-42.7%), action RMSE +51.7%, gripper MAE +78.0%. HOWEVER: task-level overfit has NOT been verified in DrawerRobotEnv. No evidence of: (1) drawer success rate in simulation, (2) attach/open sequence correctness, (3) drawer fraction improvement. "Pipeline confirmed" is OVERSTATED — "offline imitation improved" is the accurate claim. Env-level overfit gate (finetuned vs pretrained in env) required before claiming task success.
- `C_V59_ENV_GATE_FAILED` [supported]: V59 env-level overfit gate: NEGATIVE_GATE. Both pretrained and finetuned MINT achieve 0.000 drawer_fraction on all 6 simulation rollouts (3 seeds × 2 episodes). Offline action RMSE improvement (+51.7%) does NOT translate to task success. CORRECTED DIAGNOSIS: E023 (RCA1) proved all 7/7 teacher episodes are task-successful (mean_max_drawer=0.989). Teacher demos ARE successful. MINT improved at imitating them offline, but imitated actions still do not open the drawer in simulation. Root cause must be: action normalization mismatch, state representation gap, or sim-to-sim gap (LIBERO→Infinigen). Insufficient training data is NOT the sole cause. Full V59 retrain without fixing action/state contract will repeat V58 failure. P0b (colored reroll): OPEN QUESTION — pretrained failure in white images is consistent with multiple hypotheses.
- `C_RCA1_TEACHER_NOT_ROOT_CAUSE` [supported]: All 7/7 V59 overfit teacher episodes are task-successful. RCA1 verified: episodes 0,1,2,3,4,5,49 (LeRobot indices) map to physics-legal NPZ rollouts with success=True, mean_max_drawer_fraction=0.989, attach_rate=1.0. Teacher quality is NOT the root cause of V58/V59 failure. Root cause must be in MINT training pipeline: action normalization, state representation, or sim-to-sim gap. METHODOLOGY: LeRobot episode_index ≠ NPZ filename episode index. Must trace through V59 pack log sorted file order. E023 confirms.
- `C_V59_ACTION_NORMALIZATION_NOT_ROOT_CAUSE` [supported]: For the 7 V59 overfit episodes, action normalization is NOT the root cause of
task failure in DrawerRobotEnv. Dataset actions are in [-1, 1] range (position/rotation)
and gripper uses ±1.0 convention, matching DrawerRobotEnv's expected input contract.
Teacher action replay is inconclusive (attachment_local estimation error) but
static analysis independently proves no action format mismatch.

- `C_V59_STATE_REPRESENTATION_NOT_ROOT_CAUSE` [supported]: State representation is not the root cause for the 7 overfit episodes.

## Historical Context Claims
- Historical context claims currently suppressed from the main dashboard: `10`
- `C_PHYSICS_LEGAL_TEACHER_HISTORICAL_SUMMARY` [supported] / scope `historical_context_only`
- `C_STATE_CONTRACT_HISTORICAL_SUMMARY` [supported] / scope `historical_context_only`
- `C_VISION_ALIGNMENT_HISTORICAL_SUMMARY` [supported] / scope `historical_context_only`
- `C_IMAGE_NEAR_WHITE` [supported] / scope `v58_vision_pipeline`
- `C_IMAGE_NEAR_WHITE@2` [supported] / scope `v58_vision_pipeline`
- ... plus `5` more historical-context claims

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
- Canonical next action: `learnability_audit_gate` / `gate_b_teacher_replayability` / priority `P0`
- Target: Verify teacher actions replay successfully in DrawerRobotEnv (top-10 episodes)
- Scope: Gate A reveals: all 240 episodes have strong action quality and extremely weak within-episode visual variation. Gate B answers a narrower question: can teacher actions that succeed in the source simulator achieve attach/open when replayed inside DrawerRobotEnv? If not, simulator physics/geometry compatibility becomes the stronger candidate. If yes, learnability remains open and needs separate gates.

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

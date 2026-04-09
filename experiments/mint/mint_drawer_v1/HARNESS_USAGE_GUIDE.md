# Harness v2 — mint_drawer_v1
**GENERATED FROM current_truth.json — NOT CANONICAL**

- Generated at: `2026-04-09T20:52:11+08:00`
- Source: `sovereign/current_truth.json`
- Stale policy: regenerate with `python scripts/harness/sovereign_cli.py go`

## At A Glance
- Verdict: `V59_UPSTREAM_MINT_BASELINE_RESTORED — PATCH_REGRESSION_ISOLATION_NEXT`
- Phase: `v59_MUJOCO_PILOT_PHASE4 — upstream baseline restored; isolate patch regression`
- Decision: `V59_UPSTREAM_MINT_BASELINE_RESTORED — PATCH_REGRESSION_ISOLATION_NEXT`
- Human read: We are in `v59_MUJOCO_PILOT_PHASE4 — upstream baseline restored; isolate patch regression`. The current top priority is `mint_patch_regression_isolation`: Isolate the minimal local patch delta that breaks the restored official LIBERO drawer baseline
- Dataset anchor: `v59_20260405` / `240` episodes / `19701` frames / loads=`True`
- Workspace: branch `feature/mint-integration` @ `8fd7e83659983fd205ca7c1b318261a40b703b83` / dirty files `1`
- MINT fidelity: `F1` — Compatibility-patched but high-fidelity

## Claims Driving This Phase
- `C_MINT_RELEASE_BASELINE_RESTORED` [supported]: Under the release-era runtime and official LIBERO drawer baseline conditions, upstream external/MINT commit 4eab579 restores a valid MINT-libero control baseline: libero_goal/open_the_middle_drawer_of_the_cabinet succeeds 3/3 with 256x256 observations and policy.n_action_steps=4.
- `C_MINT_LOCAL_PATCH_REGRESSION` [supported]: On the same release-era runtime, official LIBERO drawer task, checkpoint, tokenizer, and evaluation path, the local patched MINT variant regresses relative to upstream 4eab579: upstream succeeds 3/3, while patched b5eabd4 fails before rollout with a GemmaModel.has-no-attribute-model error. This supports a local patch regression claim under the matched A/B scope.

## Historical Context Claims
- Historical context claims currently suppressed from the main dashboard: `15`
- `C_PHYSICS_LEGAL_TEACHER_HISTORICAL_SUMMARY` [supported] / scope `historical_context_only`
- `C_STATE_CONTRACT_HISTORICAL_SUMMARY` [supported] / scope `historical_context_only`
- `C_VISION_ALIGNMENT_HISTORICAL_SUMMARY` [supported] / scope `historical_context_only`
- `C_IMAGE_NEAR_WHITE` [supported] / scope `v58_vision_pipeline`
- `C_IMAGE_NEAR_WHITE@2` [supported] / scope `v58_vision_pipeline`
- ... plus `10` more historical-context claims

## What Is Settled Right Now
- `E036`: After rolling external/MINT back to upstream 4eab579 on feature/mint-integration, the official LIBERO drawer baseline is restored: libero_goal/open_the_middle_drawer_of_the_cabinet succeeds 3/3 under the release-era runtime with 256x256 observations and n_action_steps=4.
- `E035`: Release-runtime matched A/B on official LIBERO drawer baseline: upstream 4eab579 succeeds 3/3, while local patched b5eabd4 fails under identical task/checkpoint/runtime conditions with GemmaModel.has-no-attribute-model. Supports a local patch regression claim, not an upstream capability failure.
- `E034`: Forward-pass recheck after E033 shows mixed, nonzero state sensitivity on real LIBERO observations; strong state-invariant/image-only diagnosis is not supported.
- `E033`: MINT key-remap fix verified. Large 165 missing / 164 unexpected mismatch collapses to 1 intentional tied-weight miss; forward-pass diagnostics must be rechecked afterward.
- `E031`: Against PyBullet dataset: raw_joint_pos mean_l2_delta=4.23, pi_minus_joint_pos=9.77. pi_minus NOT better. Reference is PyBullet dataset (wrong baseline for LIBERO comparison).

## What Is Still Open
- Open gate `gate_b_teacher_replayability` [P0]: Verify teacher actions replay successfully in DrawerRobotEnv (top-10 episodes)
- No canonical blockers are registered.

## What To Read First In A New Session
1. `python scripts/harness/sovereign_cli.py go`
2. `sovereign/current_truth.json`
3. The latest verified evidence listed in `current_truth.json`
4. The spec for the current next action, if the next action is spec-backed

## Tonight / Next Safe Move
- Canonical next action: `MINT_PATCH_REGRESSION_ISOLATION` / `mint_patch_regression_isolation` / priority `P0`
- Target: Isolate the minimal local patch delta that breaks the restored official LIBERO drawer baseline
- Scope: Use the restored 4eab579 release baseline as the control and re-introduce local MINT patches one change at a time until the regression reappears.

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

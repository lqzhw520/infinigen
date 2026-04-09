# SESSION_BOOTSTRAP.20260409
**GENERATED FROM current_truth.json — NOT CANONICAL**

- Generated at: `2026-04-09T16:45:04+08:00`
- Source: `sovereign/current_truth.json`
- Stale policy: regenerate with `python scripts/harness/sovereign_cli.py go`

## Current Canonical Truth
- Verdict: `V59_MUJOCO_PILOT_ESTABLISHED — OFFICIAL_LIBERO_BASELINE_REPRO_NEXT`
- Phase: `v59_MUJOCO_PILOT_PHASE3 — official LIBERO drawer baseline reproduction`
- Phase gate: `MUJOCO_PILOT_OFFICIAL_BASELINE_PENDING`
- Decision: `V59_MUJOCO_PILOT_ESTABLISHED — OFFICIAL_LIBERO_BASELINE_REPRO_NEXT`

## Canonical Sources
- `sovereign/claims.yaml`
- `sovereign/evidence/index.json`
- `sovereign/evidence/*.yaml`
- `sovereign/next_actions.json`
- `artifacts/current_dataset_manifest.json`
- `sovereign/workspace_manifest.json`
- `sovereign/model_load_fidelity.json`
- `sovereign/current_truth.json`

## Dataset Anchor
- Version: `v59_20260405`
- Episodes: `240`
- Frames: `19701`
- Loads: `True`

## Workspace Snapshot
- Branch: `feature/mint-integration`
- HEAD: `4253ba45f3951c828c3482cb42c118e409702125`
- Dirty files: `10`
- external/MINT dirty files: `0`

## Model Load Fidelity
- Grade: `F2`
- Summary: Runnable but semantically drifted

## Next Action
- Type: `MINT_LIBERO_OFFICIAL_BASELINE_REPRO`
- Id: `libero_goal_drawer_official_baseline_repro`
- Priority: `P0`
- Target: Reproduce official MINT LIBERO drawer baseline on an in-distribution drawer/cabinet task

## Stale / Unsafe Docs
- none

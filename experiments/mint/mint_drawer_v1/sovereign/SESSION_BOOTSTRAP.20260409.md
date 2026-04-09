# SESSION_BOOTSTRAP.20260409
**GENERATED FROM current_truth.json — NOT CANONICAL**

- Generated at: `2026-04-09T20:50:15+08:00`
- Source: `sovereign/current_truth.json`
- Stale policy: regenerate with `python scripts/harness/sovereign_cli.py go`

## Current Canonical Truth
- Verdict: `V59_UPSTREAM_MINT_BASELINE_RESTORED — PATCH_REGRESSION_ISOLATION_NEXT`
- Phase: `v59_MUJOCO_PILOT_PHASE4 — upstream baseline restored; isolate patch regression`
- Phase gate: `MUJOCO_PILOT_UPSTREAM_BASELINE_RESTORED`
- Decision: `V59_UPSTREAM_MINT_BASELINE_RESTORED — PATCH_REGRESSION_ISOLATION_NEXT`

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
- HEAD: `8ddcc44e2736fb8f186090e8fa3af7009e870772`
- Dirty files: `17`
- external/MINT dirty files: `1`

## Model Load Fidelity
- Grade: `F1`
- Summary: Compatibility-patched but high-fidelity

## Next Action
- Type: `MINT_PATCH_REGRESSION_ISOLATION`
- Id: `mint_patch_regression_isolation`
- Priority: `P0`
- Target: Isolate the minimal local patch delta that breaks the restored official LIBERO drawer baseline

## Stale / Unsafe Docs
- none

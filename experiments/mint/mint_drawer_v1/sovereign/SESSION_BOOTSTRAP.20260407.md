# SESSION_BOOTSTRAP.20260407
**GENERATED FROM current_truth.json — NOT CANONICAL**

- Generated at: `2026-04-07T15:19:53+08:00`
- Source: `sovereign/current_truth.json`
- Stale policy: regenerate with `python scripts/harness/sovereign_cli.py go`

## Current Canonical Truth
- Verdict: `V59_GATE_A_COMPLETED — GATE_B_NOW_PRIMARY`
- Phase: `v59_LEARNABILITY_AUDIT_GATE_B`
- Phase gate: `v59_GATE_A_COMPLETED`
- Decision: `V59_GATE_A_COMPLETED — GATE_B_NOW_PRIMARY`

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
- HEAD: `78f807833d5fbb17734b6cf83f5b8430f7b018ce`
- Dirty files: `25`
- external/MINT dirty files: `0`

## Model Load Fidelity
- Grade: `F2`
- Summary: Runnable but semantically drifted

## Next Action
- Type: `learnability_audit_gate`
- Id: `gate_b_teacher_replayability`
- Priority: `P0`
- Target: Verify teacher actions replay successfully in DrawerRobotEnv (top-10 episodes)

## Stale / Unsafe Docs
- none

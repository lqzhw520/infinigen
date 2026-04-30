# GOC_V3_EXACT_ID_CONTRACT_REBUILD_AND_VALIDATE Execution Plan

- task_id: GOC_V3_EXACT_ID_CONTRACT_REBUILD_AND_VALIDATE
- task_type: CONTRACT_REBUILD_NOT_RUNTIME_EXECUTION
- authoritative_worktree: /mnt/afs2/zhuhaowu/infinigen
- branch: feature/mint-env-reformulation-v1-visual-fidelity
- max_duration: 8h

## Hard Boundaries
- No rollout, render, train, replay, teacher generation, evaluation rollout, or Phase 1H execution.
- No runtime patch and no harness patch.
- No mutation of current_truth.json or next_actions.json.
- Only proposed sovereign deltas may be written.
- No user prompts mid-run.

## Allowed Outputs
- Canonical GOC-v3 exact-ID artifacts under artifacts/phase1h_geometry_contract/.
- Run evidence under this run_dir.
- Proposed V11-G4 task rewrite and proposed sovereign deltas.

## Validation Invariants
- Every contact-relevant geom has exactly one primary owner.
- Target contact is only legal_gripper_surface_geom_ids <-> drawer_handle_geom_ids.
- Forbidden contact covers nonlegal robot contact surfaces against handle/body/cabinet.
- Unknown geoms cannot participate in target or forbidden contact.
- GOC-v3 uses exact geom IDs from the current static model inventory, never count-only authority.
- GOC-v2 29/26 versus runtime 31/27 must be reconciled with exact current IDs and stale/count-only metadata called out explicitly.

## Closeout States
- SUCCESS: GOC_V3_READY_FOR_V11_G4_TASK_REWRITE
- NON-SUCCESS: HARNESS_PREFLIGHT_FAILED, WRONG_WORKTREE, WRONG_BRANCH, REMOTE_NOT_CONFIGURED, DIRTY_FORBIDDEN_SCOPE, GOC_V3_MODEL_LOAD_BLOCKED, GOC_V3_FULL_ROBOT_MODEL_UNAVAILABLE, GOC_V3_MODEL_INVENTORY_FAILED, GOC_V3_AUTHORITY_AMBIGUOUS, GOC_V3_INVARIANTS_FAILED, GOC_V3_ARTIFACT_WRITE_FAILED, GOC_V3_COMMIT_PUSH_FAILED, TIME_BUDGET_EXHAUSTED, EXECUTION_FAILED

## Execution
After writing this plan, run the static model inventory and exact-ID GOC-v3 builder automatically.

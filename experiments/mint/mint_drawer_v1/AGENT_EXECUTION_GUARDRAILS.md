# Agent Execution Guardrails — MINT Drawer V11 Campaign

**Version**: v3.1 (campaign-native)
**Effective**: 2026-04-29
**Changelog**: v3.1 ports repo-root guardrails to campaign-native paths. Same 8 rules, but paths point to `experiments/mint/mint_drawer_v1/` campaign root. Learned from Phase 1H FSM v2 authority drift and grasp_pose crash misclassification.
**Scope**: All agent sessions (Opus, Codex, Claude Code, and any future agent) working on V11/MINT campaign
**Persistence**: Long-lived. Remains valid after V11 closes. Active goal contract changes from V11 to V12 but these rules persist.
**Read by**: Every agent session MUST read this file at session start.
**Campaign Root**: `experiments/mint/mint_drawer_v1/`

---

## The 8 Rules

### Rule 1: Never Change Acceptance Criteria During Execution
- `autopilot/v11_hard_goal_contract.json` and GOC artifact authority are **immutable** once execution begins.
- The FSM/runtime is **NOT** allowed to discover runtime truth and then retroactively update the acceptance criteria.
- If runtime and artifact disagree, **STOP** with `AUTHORITY_MISMATCH`.

### Rule 2: Never Modify active_goal_contract / GOC Authority Inside Runtime Task
- No agent may edit GOC artifacts during task execution.
- If GOC and runtime disagree, the agent MUST NOT patch runtime or GOC during the same task.

### Rule 3: If Runtime and Artifact Disagree, Stop with AUTHORITY_MISMATCH
- Runtime geometry classification is **not** the authoritative truth.
- GOC artifact authority is the authoritative truth.
- `scripts/harness/validators/validate_task_authority.py` MUST run at S0 and V03 gates.

### Rule 4: Crash Is NOT Route Failure
- `Traceback`, `ValueError`, `AttributeError`, or missing `result.json` → `INFRASTRUCTURE_BLOCKED`.
- `ROUTE_NEVER_REACHES_HANDLE` / `ROLLOUT_EXHAUSTED` only valid when all seeds completed without crashing and route was actually evaluated.
- `grasp_pose` crash → `ROLLOUT_COMMAND_BROKEN` → `INFRASTRUCTURE_BLOCKED`.
- `scripts/harness/validators/validate_closeout.py` is the machine decision for closeout classification.

### Rule 5: No Closeout Without closeout_decision.json Validated by validate_closeout.py
- Every execution MUST produce a `closeout_decision.json` validated by `scripts/harness/validators/validate_closeout.py`.
- No closeout = execution failure.
- Classifications: `INFRASTRUCTURE_BLOCKED`, `ROLLOUT_EXHAUSTED`, `ROUTE_SUCCESS`, `GATE_PASSED`, `INVALID_CLOSEOUT`.

### Rule 6: No Execution Without Campaign Task Spec
- Every task MUST have a `sovereign/experiment_specs/<task>.yaml` spec.
- Run `scripts/harness/agent_task_preflight.py` before any execution.
- Agent is an **executor**, not an authorizer. Task spec is the authorizer.

### Rule 7: No Untracked Evidence
- Untracked files never count as evidence.
- `git add .` is **forbidden**.
- Untracked remote-scope files must be committed before citation.

### Rule 8: No Force-Push Without Explicit Human Approval
- Always stop and get explicit human approval first.

---

## Forbidden Actions

| Action | Why |
|--------|-----|
| `git add .` | Scope creep |
| Force-push | Rewinds branch history |
| Direct write to `sovereign/current_truth.json` / `sovereign/next_actions.json` | Evidence laundering — must use `sovereign_cli.py` |
| Change acceptance criteria during execution | Authority drift |
| Classify crash as route failure | Misclassification |
| Execute without `sovereign/experiment_specs/<task>.yaml` | No contract |
| FSM mid-flight authority update | Authority drift |
| Phase 1H FSM execution before GOC-v3 | GOC-v2 authority drift |
| Modify `scripts/harness/validators/` during runtime task | Validators are immutable |
| Modify `autopilot/agent_execution_harness_lock.json` during runtime task | Lock is immutable |

---

## Harness Architecture (Campaign-Native)

```
agent_task_preflight.py (canonical entry point)
  V01: scripts/harness/validators/validate_task_authority.py --dry-run  (S0: authority initialized)
  V02: scripts/harness/validators/validate_diff_scope.py --dry-run       (S0: scope clean)
  [Agent executes within task spec scope]
  V03: scripts/harness/validators/validate_task_authority.py --runtime-check  (STOP on AUTHORITY_MISMATCH)
  V04: scripts/harness/validators/validate_closeout.py                    (machine closeout classification)
```

Also callable via:
```bash
python scripts/harness/sovereign_cli.py pre-flight-validator \
  --task-yaml sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml
```

---

## GOC-v3 Requirements

GOC-v2 (count-based) is insufficient. GOC-v3 must include:
- Explicit per-geom IDs (not just counts)
- `primary_owner`: gripper | robot_arm | drawer_body | visual_only | noncontact
- `contact_role`: legal_gripper_surface | forbidden_robot_surface | handle_surface
- `is_collision_contact_surface`: boolean

---

## Campaign-Specific Bootstrapping

### For A800 remote sessions (primary):
```bash
cd /mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1
python3 scripts/harness/sovereign_cli.py go
```

### For local workspace sessions:
```bash
# Read campaign truth from A800 sync
cat experiments/mint/mint_drawer_v1/sovereign/current_truth.json

# Run pre-flight against campaign spec
cd experiments/mint/mint_drawer_v1
MINT_TASK_ROOT=. python3 scripts/harness/agent_task_preflight.py \
  sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml --dry-run
```

---

## Immutable Files During Runtime Tasks

These files MUST NOT be modified during any runtime task (only `HARNESS_MAINTENANCE` task type may modify validators):

```
experiments/mint/mint_drawer_v1/scripts/harness/validators/validate_task_authority.py
experiments/mint/mint_drawer_v1/scripts/harness/validators/validate_diff_scope.py
experiments/mint/mint_drawer_v1/scripts/harness/validators/validate_closeout.py
experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py
experiments/mint/mint_drawer_v1/autopilot/agent_execution_harness_lock.json
experiments/mint/mint_drawer_v1/autopilot/v11_hard_goal_contract.json
experiments/mint/mint_drawer_v1/artifacts/phase1h_geometry_contract/geometry_ownership_contract.json
experiments/mint/mint_drawer_v1/sovereign/current_truth.json
experiments/mint/mint_drawer_v1/sovereign/next_actions.json
experiments/mint/mint_drawer_v1/sovereign/claims.yaml
experiments/mint/mint_drawer_v1/sovereign/evidence/*.yaml
```

---

## Local Workspace Context

Local macOS / Playground workspace is a **relay/audit layer**, NOT sovereign authority.

Local workspace MUST NOT:
- Run GOC-v3
- Run Phase 1H FSM
- Patch runtime code
- Rollout, render, or train
- Create new branches
- Force-push
- Modify `sovereign/current_truth.json` or `sovereign/next_actions.json` directly
- Use `tools/sprintctl prompt` for execution planning

Local workspace MAY:
- Invoke A800 campaign pre-flight
- Record local handoff / registry
- Save local replay/render artifacts
- Audit sovereign state from A800 sync

---

If you are unsure whether an action violates these guardrails, **stop and ask**.

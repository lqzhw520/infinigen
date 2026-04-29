# Agent Execution Guardrails

**Version**: v1
**Effective**: 2026-04-29
**Scope**: All agent sessions (Opus, Codex, Claude Code, and any future agent) on this workspace
**Persistence**: Long-lived. Remains valid after V11 closes.
**Read by**: Every agent session MUST read this file at session start.

---

## Why This Exists

Without explicit guardrails, agents default to maximizing their own task completion rather than respecting project-level claim boundaries, file scopes, and evidence standards. This document exists to prevent:

- Scope creep through incremental patches beyond approved files/functions
- Evidence laundering through untracked working-tree changes
- Loop behavior through unbounded micro-step refinement
- V11 success claims from non-qualifying proxy runtime artifacts

---

## The 6 Rules

### Rule 1: Think Before Coding

- State assumptions explicitly in comments or handoff notes.
- Do not hide uncertainty in implementation.
- If ambiguity affects patch scope or claim boundary, **stop** or write a blocker note.
- Do not proceed past ambiguity without a human-readable rationale.

### Rule 2: Simplicity First

- Do not add speculative flexibility, helper methods, or infrastructure "in case we need it later."
- Do not create new gates, plans, or branches unless the active goal contract explicitly allows it.
- Do not broaden runtime code changes beyond the approved file/function list.
- Every code edit must trace to a named approved scope item.

### Rule 3: Surgical Changes

- Every code edit MUST trace to an approved file, function, or sprint.
- **Never `git add .`** — always enumerate specific paths.
- Never modify `sovereign/current_truth.json` or `sovereign/next_actions.json` directly unless explicitly approved in a signed-off plan.
- Preserve all legacy evidence. Never delete or overwrite evidence files from earlier sprints.
- The working-tree (untracked/uncommitted) never counts as evidence.

### Rule 4: Goal-Driven Execution

- Every execution run MUST have a terminal closeout document.
- No closeout = execution failure.
- Every execution MUST cite `state/active_goal_contract.json` and name the target gate.
- Every success or failure claim MUST cite a specific artifact or commit hash.
- No claim without a traceable evidence pointer.

### Rule 5: Local/Remote Discipline

- Local Playground control-plane commits and remote A800 Infinigen commits are **separate** concerns.
- Remote runtime patches must be pushed to `my-origin` or explicitly marked `LOCAL-ONLY` in the commit message.
- Untracked files never count as evidence.
- `git diff` empty does not imply untracked files are committed.
- Always distinguish: tracked/staged/committed vs. untracked/working-tree.

### Rule 6: Anti-Loop Rule

- No interactive micro-step loops for execution.
- Use deterministic runners or bounded sprints with hard wall-clock and attempt limits.
- If context compaction or execution loop breaks agent state, **force closeout immediately**.
- If an agent session loops on re-grounding without making forward progress, write a blocker and stop.

---

## Required Pre-Session Readings

Before any task execution, every agent session MUST read:

1. `AGENT_EXECUTION_GUARDRAILS.md` — this file
2. `state/active_goal_contract.json` — active project goal
3. `state/v11_hard_goal_contract.json` or `my-origin/.../autopilot/v11_hard_goal_contract.json` — V11 hard goal and 8 gates
4. `sprints/current.md` — current sprint contract

---

## Forbidden Actions (Always)

| Action | Why |
|--------|-----|
| `git add .` | Scope creep risk |
| Direct write to `current_truth.json` | Evidence laundering |
| Direct write to `next_actions.json` | Evidence laundering |
| Unbounded loops | State corruption |
| Claim without artifact | False evidence |
| Rollout without sprint approval | Scope violation |
| Remote patch without approved plan | Scope violation |
| Training without gate 8 approval | Premature optimization |

---

## Enforcement

These guardrails are enforced by:

- `tools/sprintctl prompt` — refuses execution prompt output unless `sprints/current.md` has `status=active`
- Sprint contracts — each sprint has an explicit `next_gate` field
- `tools/a800ctl doctor` — validates control-plane file integrity before any execution
- Git hygiene — whitelisted files only; never `git add .`

If you are unsure whether an action violates these guardrails, **stop and ask**.

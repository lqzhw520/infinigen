# Agent Bootloader

Read these files at the start of every session — in order:

1. `docs/AGENT_EXECUTION_GUARDRAILS.md` — the 8 mandatory rules. Never modify validators, task specs, goal contracts, or GOC authority during a runtime task.
2. `state/active_goal_contract.json` — active project goal and GOC authority.
3. `sprints/current.md` — current sprint contract (check `status=active` before execution).
4. `tasks/*.yaml` — the task spec for the current task. **Do not execute without a task YAML.**

Rules enforced by this workspace:
- Never modify acceptance criteria during execution.
- Never change validators, task specs, goal contracts, or GOC authority during a runtime task.
- Crash is `INFRASTRUCTURE_BLOCKED`, not `ROLLOUT_EXHAUSTED`.
- No execution without `tasks/*.yaml`.
- No untracked evidence.
- No force-push without explicit human approval.

## Campaign Entry Point

For the MINT drawer V11 campaign, read:
- `experiments/mint/mint_drawer_v1/CLAUDE.md` — campaign agent bootloader
- `experiments/mint/mint_drawer_v1/AGENT_EXECUTION_GUARDRAILS.md` — campaign execution rules
- `experiments/mint/mint_drawer_v1/autopilot/agent_execution_harness_lock.json` — production lock
- `experiments/mint/mint_drawer_v1/autopilot/v11_hard_goal_contract.json` — active goal contract

Do not execute without the campaign task spec and pre-flight.

See `AGENTS.md` for recovery context and guardrails.
See `handoff/CURRENT.md` for current handoff state.

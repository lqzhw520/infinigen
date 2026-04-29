# Agent Bootloader — MINT Drawer V11 Campaign

**Campaign Root**: `experiments/mint/mint_drawer_v1/`
**Authority**: A800 remote sovereign, NOT local workspace
**Mode**: Phase 0 infrastructure / control-plane recovery only

---

Read these files at the start of every session — in order:

1. `AGENT_EXECUTION_GUARDRAILS.md` — the 8 mandatory rules. Never modify validators, task specs, goal contracts, or GOC authority during a runtime task.
2. `autopilot/v11_hard_goal_contract.json` — active goal contract.
3. `sovereign/current_truth.json` — canonical truth surface (canonical Tier-0, NOT generated doc).
4. `sovereign/next_actions.json` — canonical action queue.
5. `autopilot/agent_execution_harness_lock.json` — production lock. Check `harness_status` before any execution.
6. `sovereign/experiment_specs/<task>.yaml` — task spec for the current task. **Do not execute without pre-flight PASS.**

---

## Campaign Topology

```
Repo root → experiments/mint/mint_drawer_v1/  (V11 governance root)
  ↑ not sovereign authority itself
  └─ CLAUDE.md (this file) = minimal bootloader pointing here

Campaign root = experiments/mint/mint_drawer_v1/
  AGENT_EXECUTION_GUARDRAILS.md     ← execution rules (long-lived)
  CLAUDE.md (this file)             ← campaign bootloader
  sovereign/                        ← canonical truth + evidence (Tier-0)
  autopilot/                        ← production lock + goal contract
  scripts/harness/                  ← execution harness
  sovereign/experiment_specs/        ← campaign-native task specs
```

---

## Pre-Session Checklist

```bash
cd experiments/mint/mint_drawer_v1

# 1. Reconcile sovereign sources
python3 scripts/harness/sovereign_cli.py go

# 2. Verify production lock
cat autopilot/agent_execution_harness_lock.json
# Check: harness_status, campaign_preflight_callable, validators_installed_campaign
```

---

## Pre-Flight (Required Before Any Execution)

```bash
cd experiments/mint/mint_drawer_v1
python3 scripts/harness/agent_task_preflight.py \
  sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml --dry-run
```

**Do not proceed if pre-flight fails.**

---

## Rules

- **Never change acceptance criteria during execution** — GOC authority is immutable
- **Runtime disagreeing with GOC = STOP** — `AUTHORITY_MISMATCH`
- **Crash = INFRASTRUCTURE_BLOCKED** — not route failure
- **No spec = no execution** — run pre-flight first
- **Machine closeout classification** — `validate_closeout.py` decides, not human judgment
- **No force-push without explicit human approval**
- **Local = relay/audit only** — do not modify sovereign truth from local workspace

---

## Current Sprint

Check `sprints/current.md` in repo root for sprint contract (status, allowed/forbidden actions, commit policy).

---

## Forbidden Actions

| Action | Why |
|--------|-----|
| `git add .` | Scope creep |
| Force-push | Rewinds branch history |
| Direct write to `sovereign/current_truth.json` or `sovereign/next_actions.json` | Must use `sovereign_cli.py` |
| Modify validators/task specs/contracts during runtime task | Authority immutability |
| Classify crash as route failure | Misclassification |
| Phase 1H FSM execution before GOC-v3 | GOC-v2 authority drift |

---

## Next Gate

`GOC_V3_EXACT_ID_CONTRACT_REBUILD`

V11-G4 FSM execution is PROHIBITED until GOC-v3 with explicit per-geom IDs is complete and validated.

---

See `AGENTS.md` in repo root for recovery context and guardrails.
See `handoff/CURRENT.md` in repo root for current handoff state.

# Campaign Harness Integration — Inventory

**Version**: 1.0.0
**Generated**: 2026-04-29
**Campaign Root**: `experiments/mint/mint_drawer_v1`
**Source**: A800 mirror at `infinigen_harness_patch/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/`

---

## Classification Taxonomy

| Type | Count | Description |
|------|-------|-------------|
| `truth_generator` | 25 | Files that build/derive canonical truth surfaces |
| `generated_doc` | 5 | Convenience docs generated from truth — NOT CANONICAL |
| `status_surface` | 5 | Runtime status / lock / state records |
| `validator` | 5 | Execution gates that enforce harness rules |
| `execution_gate` | 4 | Pre-flight / entry-point blockers |
| `agent_instruction` | 8 | Human-readable guidance for agents |
| `task_spec` | 2 | Immutable task contracts |
| `log` | 7 | Historical runtime logs |

---

## Key Files (require special care)

### Execution Gates (non-negotiable)

| File | Type | Purpose |
|------|------|---------|
| `scripts/harness/agent_task_preflight.py` | execution_gate | V01 + V02 pre-flight. **Canonical entry point.** |
| `scripts/harness/validators/agent_task_preflight.py` | execution_gate | Duplicate at validators/ path — not canonical |
| `scripts/harness/validators/validate_task_authority.py` | validator | Blocks if GOC vs runtime mismatch |
| `scripts/harness/validators/validate_diff_scope.py` | validator | Blocks if forbidden files change |
| `scripts/harness/validators/validate_closeout.py` | validator | Machine-classifies closeout (crash ≠ route failure) |
| `scripts/harness/sovereign_cli.py` | truth_generator + execution_gate | Pre-flight-validator command wires agent_task_preflight.py |
| `autopilot/agent_execution_harness_lock.json` | status_surface | Production lock: harness status, regressions, enforcement rules |

### Canonical Truth Sources (Tier-0)

| File | Type | Authority |
|------|------|-----------|
| `sovereign/claims.yaml` | truth_generator | Tier-0 canonical |
| `sovereign/evidence/index.json` | truth_generator | Tier-0 canonical |
| `sovereign/evidence/*.yaml` | truth_generator | Tier-0 canonical |
| `sovereign/next_actions.json` | truth_generator | Tier-0 canonical |
| `sovereign/current_truth.json` | truth_generator | Derived from Tier-0 — IS canonical because it IS the truth surface |
| `artifacts/phase1h_geometry_contract/geometry_ownership_contract.json` | truth_generator | GOC v2 authority |
| `artifacts/current_dataset_manifest.json` | truth_generator | Tier-0 canonical |
| `sovereign/workspace_manifest.json` | truth_generator | Tier-0 canonical |
| `sovereign/model_load_fidelity.json` | truth_generator | Tier-0 canonical |

### Generated Docs (Convenience Only — NOT CANONICAL)

| File | Policy |
|------|--------|
| `HARNESS_USAGE_GUIDE.md` | Regenerate from `current_truth.json`. If conflicts, trust truth and regenerate. |
| `sovereign/CAMPAIGN_TRUTH.generated.md` | Generated from `current_truth.json` — NOT CANONICAL |
| `sovereign/SESSION_BOOTSTRAP.*.md` | Generated from `current_truth.json` — NOT CANONICAL |
| `sovereign/handoff.md` | Generated from `current_truth.json` — NOT CANONICAL |
| `sovereign/CAMPAIGN_TRUTH.v58_critical_analysis.generated.md` | Generated from `current_truth.json` — NOT CANONICAL |

---

## What Is Missing From Local Campaign Root

The local `experiments/mint/mint_drawer_v1/` is **sparse** — it only has:
- `artifacts/` (GOC artifact, dataset manifest)
- `autopilot/` (goal contract, harness lock, harness status)
- `sovereign/` (proposed deltas only — NOT the canonical sovereign)

**Missing from local** (only exist in A800 mirror):
- `scripts/harness/` — entire directory (the full execution harness)
- `sovereign/current_truth.json` — canonical truth surface
- `sovereign/next_actions.json` — canonical action queue
- `sovereign/evidence/` — all 27 evidence files + index
- `tasks/v11_g4_phase1h_contact_test.yaml` — root-relative task spec
- `CLAUDE.md` — campaign bootloader
- `AGENT_EXECUTION_GUARDRAILS.md` — campaign execution guardrails

---

## Governance Notes

- **HARNESS_USAGE_GUIDE.md** explicitly says: "generated from current_truth.json — NOT CANONICAL. If conflicts with current_truth.json, trust current_truth.json and regenerate." This means generated docs are convenience views; canonical is always the JSON/YAML source.
- **Root CLAUDE.md** was replaced with minimal bootloader but still references local workspace paths, not campaign root paths.
- **Campaign CLAUDE.md** does not exist in the local campaign root.
- **Campaign AGENT_EXECUTION_GUARDRAILS.md** does not exist in the local campaign root.
- **Task spec** exists at two locations: `tasks/v11_g4_phase1h_contact_test.yaml` (root-relative) and `scripts/harness/validators/v11_g4_phase1h_contact_test.yaml` (harness-native). Both need campaign-native canonicalization under `sovereign/experiment_specs/`.
- **Local workspace** is NOT the sovereign authority. A800 is. Local is relay/audit only.

---

## Validator Chain (V01 → V04)

```
agent_task_preflight.py (canonical entry)
    V01: validate_task_authority.py --dry-run   (S0: authority initialized)
    V02: validate_diff_scope.py --dry-run         (S0: scope clean)
    [Agent executes]
    V03: validate_task_authority.py --runtime-check (STOP on AUTHORITY_MISMATCH)
    V04: validate_closeout.py                    (machine closeout classification)
```

---

## Campaign Architecture (Correct Topology)

```
Repo Root (Layer A — minimal bootloader only)
  CLAUDE.md → "Read experiments/mint/mint_drawer_v1/CLAUDE.md"

Campaign Root (Layer B — V11 governance root)
  CLAUDE.md                          ← campaign agent bootloader
  AGENT_EXECUTION_GUARDRAILS.md     ← V11 execution rules
  sovereign/                        ← canonical truth + evidence
  autopilot/agent_execution_harness_lock.json  ← production lock
  scripts/harness/
    agent_task_preflight.py          ← canonical pre-flight entry
    sovereign_cli.py                 ← truth + pre-flight wiring
    truth_backend.py                 ← truth generation engine
    validators/
      validate_task_authority.py      ← authority gate
      validate_diff_scope.py          ← scope gate
      validate_closeout.py           ← closeout classifier
      v11_g4_phase1h_contact_test.yaml  ← task spec
  sovereign/experiment_specs/
    v11_g4_phase1h_contact_test.yaml  ← campaign-native task spec

Local Control Plane (Layer C — relay/audit only)
  handoff/                          ← local handoff records
  registry/                         ← local registry
  runs/                             ← local run records
  tools/sprintctl, tools/a800ctl    ← local adapters
```

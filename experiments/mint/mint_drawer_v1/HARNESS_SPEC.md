# Research Governance Harness Specification — mint_drawer_v1

**Version**: v1.4 Core
**Date**: 2026-04-01
**Status**: SPEC (normative — this is the source of truth for harness rules)

> This document is the normative specification. It describes what the harness IS and what it DOES.
> Execution status is in `ROLLOUT_STATUS.md`.
> Do NOT mix "what should be" with "what has been built" in this document.

---

## 0. What This Harness Is For

This is not an automation platform. It is a **research governance harness** for the `mint_drawer_v1` project.

Its purpose is to make every development session start from a consistent, auditable, claim-tracked ground truth — so that AI-assisted research on Infinigen/MINT does not drift into contradiction, stale context, or lost evidence.

The core question it answers at every session start:

> *"What do we believe, based on what evidence, and what changed since last time?"*

Night automation is **not** a goal. The system is designed for a human-in-the-loop research cycle where each session is supervised.

---

## 1. Three-Layer Architecture

```
┌──────────────────────────────────────────────────────┐
│  Layer A — Truth Governance                          │
│  sovereign/  (canonical source of truth)             │
│  evidence/  (canonical evidence registry)           │
│  proposals/ (structured claim revision proposals)   │
└──────────────────────────┬───────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────┐
│  Layer B — Session Control                          │
│  bootstrap  →  lint gates  →  handoff  →  CLI       │
│  (every session starts here)                        │
└──────────────────────────┬───────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────┐
│  Layer C — Maintenance                              │
│  GC reports  →  rollout status  →  spec/status docs │
└──────────────────────────────────────────────────────┘
```

---

## 2. Layer A — Truth Governance

### 2.1 Directory Structure

```
experiments/mint/mint_drawer_v1/
├── sovereign/                          # CANONICAL TRUTH (machine+human readable)
│   ├── manifest.yaml                   # Campaign metadata (read-only)
│   ├── state.json                      # Machine state snapshot (read+CLI-write)
│   ├── claims.yaml                     # Versioned claim registry (CLI-write only)
│   ├── next_actions.json              # Action queue (CLI-write only)
│   ├── handoff.md                     # Human-agent handoff (CLI-write or manual)
│   ├── CAMPAIGN_TRUTH.generated.md    # Generated truth (render-truth only)
│   ├── evidence/                       # CANONICAL evidence (record-evidence only)
│   │   ├── index.json               # Evidence index
│   │   └── E###.yaml               # Individual evidence files
│   └── proposals/                    # Structured claim revision proposals
│       └── {timestamp}_{claim_id}.yaml
│
├── runtime/                           # NON-CANONICAL runtime (read+script-write)
│   ├── evidence_inbox/              # Candidate evidence (staging)
│   └── events/                      # Runtime events (timestamped JSON)
│
├── archive/                            # Legacy archival (read-only)
│
├── scripts/harness/                    # Harness tooling
│   ├── sovereign_cli.py            # SINGLE CONTROLLED-WRITE ENTRY POINT
│   ├── 01_sync_pointers.sh         # Sync root symlinks
│   ├── 02_reconcile_sources.py     # Source-of-truth gate
│   ├── 03_claim_lint.py           # Claim + revision chain gate
│   ├── 04_action_lint.py         # Action queue gate
│   ├── 05_verify_env_contract.sh # Env contract scaffold
│   └── gc/                        # GC scripts (report-only)
│       ├── gc_01_stale_docs.py
│       ├── gc_02_orphan_artifacts.py
│       ├── gc_03_conflicting_claims.py
│       └── gc_04_outdated_next_actions.py
│
└── CAMPAIGN_TRUTH.md               # STUB FILE (points to sovereign/)
```

### 2.2 Five Non-Negotiable Non-Functional Principles

1. **Sovereignty**: canonical truth lives in one place only (`sovereign/`)
2. **No destructive migration**: legacy files are marked, not moved or deleted
3. **Write convergence**: all sovereign writes go through `sovereign_cli.py` only
4. **Generated truth**: `sovereign/CAMPAIGN_TRUTH.generated.md` is the single generated-truth file
5. **Evidence staging**: candidate evidence never enters canonical without explicit `record-evidence` call

### 2.3 Evidence Three-Tier System

| Tier | Location | Write Permission | Description |
|------|----------|-----------------|-------------|
| Canonical | `sovereign/evidence/` | `sovereign_cli.py record-evidence` only | Proven, indexed evidence |
| Staging | `runtime/evidence_inbox/` | Training scripts, agent | Candidate evidence |
| Proposals | `sovereign/proposals/` | `sovereign_cli.py propose-revision` | Claim revision proposals |

**Rule**: Candidate evidence does NOT enter `sovereign/evidence/` without `sovereign_cli.py record-evidence`.

### 2.4 `sovereign/claims.yaml` — Versioned Claim Registry

**schema_version**: 2

```yaml
schema_version: 2

claims:
  - claim_id: C_XXX
    current_revision: N
    lifecycle_status: active | superseded | split | archived
    superseded_by: ClaimID    # direct successor claim (v1.3.1)
    supersedes: ClaimID       # direct predecessor claim (v1.3.1)
    split_from: ClaimID      # source claim if split (v1.3.1)
    split_into: [ClaimID]    # target claims if split (v1.3.1)
    revisions:
      - revision: N
        statement: "..."
        scope: "..."
        status: hypothesis | candidate | supported | contradicted | archived
        evidence_ids: [E001, ...]
        change_type: create | status_update | evidence_added | scope_narrowed | scope_broadened | statement_revised | reframed | superseded
        change_reason: "..."
        created_at: "ISO8601"
        superseded_by: ClaimID@rev | null    # backward ref
        superseded_by_reason: "..." | null
        supersedes: ClaimID@rev | null        # forward ref (v1.3.1)
```

**Agent rule**: Do NOT edit `sovereign/claims.yaml` directly. Always use `sovereign_cli.py`.

### 2.5 Structured Proposal Schema (v1.3.1)

```yaml
proposal_id: 20260401_143022_C_SIGLIP_GENERALIZATION
schema_version: 1
claim_ref: C_SIGLIP_GENERALIZATION
revision_ref: C_SIGLIP_GENERALIZATION@3   # optional, defaults to current
proposal_type: revise_claim | supersede_claim | split_claim
proposed_change:
  status: candidate
  statement: "..."
  scope: "..."
  change_type: evidence_added
  change_reason: "..."
  evidence_ids_add: [E004]
  evidence_ids_remove: []
evidence_ids: [E004]
evidence_summary: "E4 shows 60% improvement..."
author: agent | human
created_at: "ISO8601"
review_status: pending | approved | rejected | deferred
approved_revision: C_XXX@N
rejected_reason: null
reviewed_by: null
reviewed_at: null
```

### 2.6 `next_actions.json` — Mandatory Revision Refs

All blocked actions MUST use revision-qualified claim IDs:

```json
"blocker_claim_ids": ["C_SIGLIP_GENERALIZATION@3"]
```

Bare claim IDs (`C_SIGLIP_GENERALIZATION` without `@N`) in blocked actions produce a **lint ERROR**.

---

## 3. Layer B — Session Control

### 3.1 sovereign_cli.py — Single Write Entry Point

```
bootstrap               Session startup: run lints + print session summary
reconcile               Run all lints (dev mode)
render-truth            Render sovereign/CAMPAIGN_TRUTH.generated.md
revise-claim           Add new revision to existing claim (maintains supersedes)
supersede-claim        Mark old claim superseded, create new claim
split-claim            Split one claim into multiple C_SUB_* claims
propose-revision        Write structured proposal to sovereign/proposals/
review-proposal         List/approve/reject/finalize proposals
record-evidence         Canonicalize evidence from inbox or direct path
close-experiment       Record evidence + update state.json
write-handoff          Write sovereign/handoff.md
```

### 3.2 Bootstrap — Every Session Starts Here

```
sovereign_cli.py bootstrap
```

Steps (automatic):
1. Run `01_sync_pointers.sh` — ensure root symlinks are valid
2. Run `02_reconcile_sources.py` — verify source-of-truth pointers
3. Run `03_claim_lint.py` — verify claim registry consistency
4. Run `04_action_lint.py` — verify next_actions consistency
5. If any lint fails → bootstrap fails, exit with error
6. Print session summary: active claims, recent revisions, superseded, blockers

Session summary output includes:
- Active claims with current status/scope
- Recent claim changes (last 3 revisions, most recent first)
- Superseded claims list
- Lint warnings (non-fatal)

### 3.3 Five Core Session Workflows

#### Workflow 1: Session Bootstrap
```bash
python scripts/harness/sovereign_cli.py bootstrap
```

#### Workflow 2: Root-Cause / Review
```bash
# After bootstrap:
# 1. Read current claims: sovereign/claims.yaml
# 2. Read generated truth: sovereign/CAMPAIGN_TRUTH.generated.md
# 3. Read recent changes: sovereign_cli.py bootstrap (recent claim changes)
# 4. Read handoff: sovereign/handoff.md
# 5. Identify blockers: sovereign/next_actions.json
```

#### Workflow 3: Experiment Closeout
```bash
# 1. Write candidate evidence
cp my_evidence.yaml runtime/evidence_inbox/E004.yaml

# 2. Canonicalize evidence
python scripts/harness/sovereign_cli.py record-evidence \
  --file runtime/evidence_inbox/E004.yaml

# 3. Close experiment
python scripts/harness/sovereign_cli.py close-experiment \
  --evidence E004 --experiment-id E4

# 4. Update claim if needed
python scripts/harness/sovereign_cli.py revise-claim \
  --claim C_SIGLIP_GENERALIZATION \
  --status supported \
  --evidence-add E004 \
  --change-type evidence_added \
  --change-reason "E4 shows significant improvement over baseline"

# 5. Render updated truth
python scripts/harness/sovereign_cli.py render-truth
```

#### Workflow 4: Claim Evolution
```bash
# Revise (change status/scope/statement, keeping same claim)
python scripts/harness/sovereign_cli.py revise-claim \
  --claim C_SIGLIP_GENERALIZATION \
  --status candidate \
  --scope current_mint_drawer_v1_pipeline \
  --change-type reframed \
  --change-reason "Advisor review: pipeline issues not fully diagnosed"

# Supersede (replace old claim with new one)
python scripts/harness/sovereign_cli.py supersede-claim \
  --old-claim C_SIGLIP_GENERALIZATION \
  --new-claim C_SYSTEM_ALIGNMENT \
  --new-statement "Failure is system-level misalignment..." \
  --supersede-reason "Reframed from vision-only to system-level" \
  --status candidate

# Split (split one claim into multiple sub-claims)
python scripts/harness/sovereign_cli.py split-claim \
  --source-claim C_SIGLIP_GENERALIZATION \
  --target-claims C_VISION_SUB C_ACTION_SUB \
  --split-reason "Split into vision and action quant concerns" \
  --split-scopes "vision_only" "action_quantization_only"
```

#### Workflow 5: Handoff / Takeover
```bash
# Before ending any session:
python scripts/harness/sovereign_cli.py write-handoff \
  --status candidate \
  --blockers "C_SIGLIP_GENERALIZATION@3,C_VQVAE_USABLE@1" \
  --reason "Need E4 eval before moving forward" \
  --decision "Run E4 then revisit C_SIGLIP_GENERALIZATION status"
```

---

## 4. Layer C — Maintenance

### 4.1 GC Scripts — Report-Only

All GC scripts are **report-only** (no automatic file modification):

| Script | Checks | Output |
|--------|--------|--------|
| `gc_01_stale_docs.py` | Working dir docs >7 days old | List of stale files |
| `gc_02_orphan_artifacts.py` | `artifacts/` not in `state.json` | List of orphan artifacts |
| `gc_03_conflicting_claims.py` | Claim revision chain + cross-ref consistency | Errors + warnings |
| `gc_04_outdated_next_actions.py` | next_actions references superseded/archived claims | Warnings |

Run regularly (before each session or weekly):
```bash
python scripts/harness/gc/gc_01_stale_docs.py
python scripts/harness/gc/gc_02_orphan_artifacts.py
python scripts/harness/gc/gc_03_conflicting_claims.py
python scripts/harness/gc/gc_04_outdated_next_actions.py
```

### 4.2 Night / Automation Scripts — Future Optional

The following scripts exist in the codebase and are functional, but are **not part of the current acceptance criteria**:

```
scripts/harness/night/
├── night_watcher.sh       # System monitoring (GPU, disk, tmux, logs)
├── night_planner.sh       # Sovereign → task plan
├── morning_handoff.sh     # Update handoff.md
└── night_runner.sh        # Master orchestrator
```

**Status**: Implemented and tested. **Deferred** from current scope.

These scripts are designed for future use when the research cycle is stable enough to benefit from automated monitoring. They are **never** allowed to directly modify `sovereign/claims.yaml` or any canonical sovereign state.

---

## 5. Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| v1.0 | 2026-03-? | Initial harness idea |
| v1.1 | 2026-03-? | Conservative correctness |
| v1.2b | 2026-03-? | Practical additions |
| v1.3 | 2026-03-31 | Phase 1+2+3+Night Runner P3a implemented |
| v1.3.1 | 2026-04-01 | Evidence staging, claim v2 schema, revision refs, proposal schema |
| **v1.4 Core** | 2026-04-01 | Night Runner removed from scope; 3-layer architecture; daytime research loop |

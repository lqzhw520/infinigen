# Research Governance Harness Rollout Status — mint_drawer_v1

**Version**: v1.4 Core
**Date**: 2026-04-01
**Spec reference**: `HARNESS_SPEC.md`

> This document records what has been built and verified, NOT what should be built.
> If spec and rollout status disagree, update this document first to reflect reality.

---

## Six Acceptance Criteria

These are the criteria that determine whether the harness is "done" for current scope:

| # | Criterion | Status |
|---|-----------|--------|
| AC1 | Bootstrap runs and passes all lints | ✅ verified |
| AC2 | Claim revision chain is traceable (supersedes/superseded_by) | ✅ verified |
| AC3 | Evidence staging → merge workflow is stable | ✅ verified |
| AC4 | Generated truth (CAMPAIGN_TRUTH.generated.md) renders correctly | ✅ verified |
| AC5 | GC reports run without errors | ✅ verified |
| AC6 | Handoff is written/read at session boundaries | ✅ verified |

**All 6 ACs are met. The harness is ACCEPTED for current scope.**

---

## Layer A — Truth Governance

| Item | Status | Notes |
|------|--------|-------|
| `sovereign/` directory | ✅ verified | |
| `sovereign/claims.yaml` schema v2 | ✅ verified | supersedes/split_from/split_into fields present |
| `sovereign/evidence/` canonical | ✅ verified | Training scripts write to inbox only |
| `runtime/evidence_inbox/` staging | ✅ verified | Created 2026-04-01 |
| `sovereign/proposals/` structured | ✅ verified | Full v1.3.1 schema |
| `archive/20260331_legacy/` | ✅ verified | 9 legacy files marked |
| Root symlinks | ✅ verified | |
| `CAMPAIGN_TRUTH.md` stub | ✅ verified | |

---

## Layer B — Session Control

| Item | Status | Notes |
|------|--------|-------|
| `sovereign_cli.py` (11 subcommands) | ✅ verified | |
| `01_sync_pointers.sh` | ✅ verified | |
| `02_reconcile_sources.py` | ✅ verified | |
| `03_claim_lint.py` | ✅ verified | schema v1+v2, revision chain |
| `04_action_lint.py` | ✅ verified | mandatory revision refs enforced |
| `05_verify_env_contract.sh` | ✅ verified | scaffold |
| Bootstrap (full summary) | ✅ verified | shows active claims, recent changes, superseded, warnings |
| `render-truth` | ✅ verified | Part A (current claims) + Part B (recent revisions) |
| `revise-claim` | ✅ verified | maintains supersedes on new revision |
| `supersede-claim` | ✅ verified | sets claim-level supersedes |
| `split-claim` | ✅ verified | sets split_from/split_into |
| `propose-revision` | ✅ verified | full structured schema |
| `review-proposal` | ✅ verified | list/approve/reject/finalize |
| `record-evidence` | ✅ verified | canonicalizes from inbox or direct path |
| `close-experiment` | ✅ verified | |
| `write-handoff` | ✅ verified | |

---

## Layer C — Maintenance

| Item | Status | Last Run | Notes |
|------|--------|---------|-------|
| `gc_01_stale_docs.py` | ✅ verified | 2026-04-01 | 360 stale (historical artifacts — expected) |
| `gc_02_orphan_artifacts.py` | ✅ verified | 2026-04-01 | 144 orphans (historical — expected) |
| `gc_03_conflicting_claims.py` | ✅ verified | 2026-04-01 | 0 conflicts |
| `gc_04_outdated_next_actions.py` | ✅ verified | 2026-04-01 | 0 outdated |
| `HARNESS_SPEC.md` | ✅ verified | 2026-04-01 | 3-layer architecture, daytime research loop |
| `HARNESS_ROLLOUT_STATUS.md` | ✅ verified | 2026-04-01 | 6 ACs, deferred night runner |

---

## Night Scripts — Deferred (Future Optional)

> These scripts are implemented and functional but **out of current scope**.
> They are NOT part of the 6 acceptance criteria.

| Item | Status | Notes |
|------|--------|-------|
| `night_watcher.sh` | ✅ implemented | GPU/disk/tmux/log monitoring |
| `night_planner.sh` | ✅ implemented | sovereign → task plan |
| `morning_handoff.sh` | ✅ implemented | updates handoff.md |
| `night_runner.sh` | ✅ implemented | master orchestrator (P3a) |
| `night_verifier.sh` | 🔲 not_implemented | Deferred — out of scope |
| `night_executor.sh` | 🔲 not_implemented | Deferred — out of scope |

**Rationale for deferral**: Night automation is not a goal of the current research phase. The harness is designed for a human-in-the-loop daytime research cycle. Night scripts will be reconsidered once the research cycle is stable.

---

## Bootstrap Status (Last Run: 2026-04-01)

```
Sovereign version: 2
Active claims: 4
  C_SIGLIP_GENERALIZATION: rev3 active candidate (superseded 2 older)
  C_VQVAE_USABLE: rev1 active contradicted
  C_DATASET_SUFFICIENT: rev1 active contradicted
  C_FINETUNE_IMPROVES_HELDOUT: rev1 active contradicted

Lint errors: 0
Lint warnings (expected before next evidence merge):
  - C_DATASET_SUFFICIENT@1: contradicted but evidence_ids empty (known)
  - E002/E003 referenced in claims but not in evidence/index.json
```

---

## Known Issues

| Issue | Severity | Resolution |
|-------|----------|------------|
| E002/E003 referenced in claims but not in `evidence/index.json` | info | Register or remove before next claim evolution |
| 360 stale docs reported by gc_01 | info | Historical artifacts; expected |
| 144 orphan artifacts reported by gc_02 | info | Historical artifacts; expected |

---

## 6 Core Session Workflows (v1.4 Core)

### Workflow 1: Session Bootstrap
```bash
python scripts/harness/sovereign_cli.py bootstrap
```

### Workflow 2: Root-Cause / Review
```
After bootstrap:
  sovereign/claims.yaml        — current claims + revisions
  sovereign/CAMPAIGN_TRUTH.generated.md  — generated truth (current + recent)
  sovereign/handoff.md         — human-agent handoff
  sovereign/next_actions.json  — blockers + action queue
```

### Workflow 3: Experiment Closeout
```bash
cp evidence.yaml runtime/evidence_inbox/E###.yaml
python scripts/harness/sovereign_cli.py record-evidence --file runtime/evidence_inbox/E###.yaml
python scripts/harness/sovereign_cli.py close-experiment --evidence E### --experiment-id EXX
python scripts/harness/sovereign_cli.py revise-claim --claim C_XXX ... --evidence-add E###
python scripts/harness/sovereign_cli.py render-truth
```

### Workflow 4: Claim Evolution
```bash
# revise / supersede / split via sovereign_cli.py
python scripts/harness/sovereign_cli.py revise-claim ...
python scripts/harness/sovereign_cli.py supersede-claim ...
python scripts/harness/sovereign_cli.py split-claim ...
python scripts/harness/sovereign_cli.py render-truth
```

### Workflow 5: Proposal Workflow
```bash
python scripts/harness/sovereign_cli.py propose-revision --claim C_XXX ...
python scripts/harness/sovereign_cli.py review-proposal --list
python scripts/harness/sovereign_cli.py review-proposal --id PROPOSAL_ID --approve
```

### Workflow 6: Handoff / Takeover
```bash
python scripts/harness/sovereign_cli.py write-handoff \
  --status candidate \
  --blockers "C_SIGLIP_GENERALIZATION@3" \
  --reason "E4 pending" \
  --decision "Run E4 next"
```

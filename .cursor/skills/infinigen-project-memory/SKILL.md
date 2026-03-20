---
name: infinigen-project-memory
description: Maintain a single source of truth for the Infinigen-AnyBox project state, architecture, iteration history, and memory. Use when the user says "update project status", "archive this iteration", "project briefing", "what is the current state", "show project history", or after completing a significant task or campaign milestone.
metadata:
  author: infinigen-team
  version: 2.2.0
  project: infinigen-anybox
---

# Infinigen Project Memory

Unified project memory system with a single source of truth (`.project-memory/`) that is synchronized at research milestones, keeps commit-time validation lightweight, and archives iteration history without dirtying the worktree after commit.

## Architecture: What Lives Where

```
.project-memory/                 <-- PRIMARY (single source of truth)
├── STATUS.md                    <-- Current project state (canonical, regenerated)
├── evolution.json               <-- Machine-readable iteration DB (append-only)
├── rules.json                   <-- Self-evolution config
├── history/                     <-- Immutable campaign and iteration snapshots
└── logs/                        <-- Untracked diagnostics such as post-commit validation logs

.session/checkpoint.md           <-- Deprecated synced view of STATUS.md
task_plan.md / findings.md       <-- Current task scratch
progress.md                      <-- Session progress log
experiments/physnap/*            <-- Campaign manifests, state, summary, review, memo
```

## Storage Hierarchy

| Location | Role | Lifecycle |
|----------|------|-----------|
| `.project-memory/STATUS.md` | Canonical project state | Regenerated at milestone finalization and explicit status sync |
| `.project-memory/evolution.json` | Structured iteration + commit history | Append-only, canonical writes go through `record_iteration.py` |
| `.project-memory/history/` | Immutable snapshots of iterations and campaign milestones | Write-once, never edit |
| `.project-memory/rules.json` | Self-evolution rule engine | Mutable config |
| `.project-memory/logs/` | Untracked validation logs | Append-only diagnostics |
| `.session/checkpoint.md` | Deprecated compatibility view | Auto-synced from STATUS.md |
| `task_plan.md` | Current task scratch | Updated during active work |
| `findings.md` | Current task discoveries | Updated during active work |
| `progress.md` | Session progress log | Updated during active work |
| `experiments/physnap/*/manifest.yaml` | Campaign contract | Versioned with code |
| `experiments/physnap/*/{state,summary,review,next_actions}.json` | Campaign working truth | Mutable campaign state |

## Canonical Sync Contract

Canonical tracked memory sync happens at milestone finalization, not after `git commit`.

```
phase or claim milestone completes
    │
    ▼
1. record_iteration.py
   - append canonical iteration payload to evolution.json
2. write_campaign_history.py
   - write immutable human-readable snapshot to history/
3. update_status.py
   - rebuild STATUS.md and sync .session/checkpoint.md
4. refresh root scratch files
   - task_plan.md / findings.md / progress.md must reflect the same truth
5. optional git commit
   - commit occurs only after tracked memory files already reflect the milestone
```

## Commit Hook Policy

The post-commit hook is now validate-only / log-only.

It may:
- record commit metadata into an untracked diagnostic log
- warn if the worktree is unexpectedly dirty after commit

It must not:
- rewrite `.project-memory/STATUS.md`
- rewrite `.project-memory/evolution.json`
- rewrite `.session/checkpoint.md`
- reset `task_plan.md`, `findings.md`, or `progress.md`
- mutate any other tracked file

A successful normal commit should leave `git status` clean unless a separate running experiment loop legitimately changes tracked files later.

## Explicit Commands

| User says | What the agent does |
|-----------|-------------------|
| "update project status" | Run `update_status.py` to regenerate STATUS.md |
| "record iteration" | Run `record_iteration.py` with a payload JSON |
| "write campaign history" | Run `write_campaign_history.py` for the active campaign |
| "project briefing" | Read STATUS.md + evolution.json and summarize canonical truth |
| "show project history" | Read evolution.json and `.project-memory/history/` |
| "evolve skill" | Run `evolve_skill.py --force` |

## After Completing Significant Work

Always do the following before considering the work synchronized:
1. Append a structured iteration via `record_iteration.py`
2. Write a human-readable history snapshot via `write_campaign_history.py`
3. Refresh `STATUS.md`, `.session/checkpoint.md`, and root scratch files
4. Verify campaign `review.json`, `decision_memo.md`, and project-memory agree
5. Only then make a git commit if desired

For manifest-driven campaigns:
1. Keep `manifest.yaml` as the decision contract
2. Keep `state.json`, `summary.json`, `review.json`, and `next_actions.json` coherent
3. When a phase diagnosis, handoff, repair milestone, or final verdict is reached, write both:
   - a structured iteration
   - a human-readable `.project-memory/history/` snapshot
4. Treat a missing terminal history snapshot as a sync failure, even if STATUS.md already looks correct

## Core Scripts

| Script | Purpose |
|--------|---------|
| `scripts/post_commit_hook.sh` | Validate-only post-commit logging |
| `scripts/update_status.py` | Regenerate STATUS.md from evolution + live campaign truth |
| `scripts/write_campaign_history.py` | Write human-readable campaign milestone snapshots |
| `scripts/record_iteration.py` | Append structured project-memory iterations |
| `scripts/evolve_skill.py` | Analyze trends and update rules |
| `scripts/load_checkpoint.py` | Show latest structured checkpoint |

## Rules

- STATUS.md is the **canonical** project state. If it conflicts with other docs, STATUS.md wins.
- `record_iteration.py` is the only canonical writer for `evolution.json`.
- `write_campaign_history.py` must be called for diagnosis locks, handoffs, repairs worth remembering, and final claim verdicts.
- history files are immutable once written.
- Final campaign verdicts must always create a dedicated terminal history snapshot.
- Manifest-driven loops must keep `state.json`, `summary.json`, `review.json`, `decision_memo.md`, `task_plan.md`, `findings.md`, and `progress.md` aligned.
- Commit hooks may validate and log, but must not rewrite tracked files after commit.
- `.session/checkpoint.md` is deprecated and exists only for backward compatibility.

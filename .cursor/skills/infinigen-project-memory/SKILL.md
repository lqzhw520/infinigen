---
name: infinigen-project-memory
description: Maintain a single source of truth for the Infinigen-AnyBox project state, architecture, iteration history, and memory. Auto-triggers on git commit. Use when the user says "update project status", "archive this iteration", "project briefing", "what is the current state", "show project history", "evolve skill", starts a new session, or after completing a significant task. Also use proactively at session start if .project-memory/STATUS.md exists.
metadata:
  author: infinigen-team
  version: 2.0.0
  project: infinigen-anybox
---

# Infinigen Project Memory

Unified project memory system with a single source of truth (`.project-memory/`) that auto-updates on every git commit, self-evolves its analysis dimensions, and archives iteration history.

## Architecture: What Lives Where

```
.project-memory/                 <-- PRIMARY (single source of truth)
├── STATUS.md                    <-- Current project state (auto-regenerated)
├── evolution.json               <-- Machine-readable iteration DB (append-only)
├── rules.json                   <-- Self-evolution config (auto-updated)
└── history/                     <-- Immutable iteration + evolution snapshots
    ├── YYYY-MM-DD_title.md
    └── YYYY-MM-DD_skill-evolution-vN-to-vM.md

.session/checkpoint.md           <-- DEPRECATED VIEW (auto-synced from STATUS.md)
task_plan.md / findings.md       <-- PER-TASK SCRATCH (auto-archived on completion)
```

### Storage Hierarchy

| Location | Role | Lifecycle |
|----------|------|-----------|
| `.project-memory/STATUS.md` | Canonical project state | Auto-regenerated on every commit |
| `.project-memory/evolution.json` | Structured iteration + commit history | Append-only, never delete |
| `.project-memory/history/` | Immutable snapshots of past iterations and evolution events | Write-once, never edit |
| `.project-memory/rules.json` | Self-evolution rule engine | Auto-updated by evolve_skill.py |
| `.session/checkpoint.md` | **Deprecated**. Legacy view, auto-synced from STATUS.md. Do not read directly; use STATUS.md. | Auto-synced |
| `task_plan.md` | Current task scratch (planning-with-files) | Created per task, auto-archived + reset when task_plan contains "COMPLETE" |
| `findings.md` | Current task discoveries | Same lifecycle as task_plan.md |

## Automatic Pipeline (git commit triggers everything)

The post-commit hook runs a 4-step pipeline on every `git commit`:

```
git commit
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Step 1: Log commit to evolution.json                     │
│   - hash, date, message, author, files_changed           │
│   - Cascade guard: skips if message starts with           │
│     "Auto-update" or "[project-memory]"                  │
├──────────────────────────────────────────────────────────┤
│ Step 2: Regenerate STATUS.md + sync checkpoint.md        │
│   - Runs update_status.py                                │
│   - STATUS.md rebuilt from evolution.json + git state     │
│   - .session/checkpoint.md synced (backward compat)      │
├──────────────────────────────────────────────────────────┤
│ Step 3: Conditional self-evolution                        │
│   - Only triggers if commit changes >= 3 files           │
│   - Runs evolve_skill.py which checks:                   │
│     * >= N new iterations since last evolve               │
│     * >= 2 new bugs OR >= 2 arch changes                  │
│   - If triggered: updates rules.json, writes snapshot    │
│     to history/YYYY-MM-DD_skill-evolution-vN-to-vM.md    │
├──────────────────────────────────────────────────────────┤
│ Step 4: Archive cleanup                                  │
│   - If task_plan.md contains "COMPLETE":                 │
│     * Snapshot task_plan + findings to history/           │
│     * Reset both files to blank state                    │
└──────────────────────────────────────────────────────────┘
```

### Install / Reinstall the Hook

```bash
cp .cursor/skills/infinigen-project-memory/scripts/post_commit_hook.sh .git/hooks/post-commit
chmod +x .git/hooks/post-commit
```

## Explicit Commands (user says)

| User says | What the agent does |
|-----------|-------------------|
| "update project status" | Runs `update_status.py` to regenerate STATUS.md |
| "archive this iteration" | Runs `archive_iteration.py --title "..." --clean` to snapshot + reset |
| "project briefing" | Reads STATUS.md + evolution.json, presents structured briefing |
| "show project history" | Reads evolution.json, lists all iterations with summaries |
| "evolve skill" | Runs `evolve_skill.py --force` to analyze trends and update rules |
| "load checkpoint" | Runs `load_checkpoint.py` to show latest iteration record |

## After Completing Significant Work

The agent should proactively:
1. Add a new iteration record to `evolution.json` (structured JSON with all fields)
2. Commit the code changes (this triggers the auto-pipeline above)
3. If a milestone is reached, also explicitly run `archive_iteration.py --title "..." --clean`
4. Verify STATUS.md was updated by reading it

## Self-Evolution: How the Skill Evolves Itself

The skill analyzes its own iteration data to detect emerging patterns:

```
evolution.json iterations
    │
    ▼
┌────────────────────────────────────────┐
│ evolve_skill.py                         │
│                                         │
│ 1. Classify bugs → detect uncovered     │
│    categories → add to rules.json       │
│ 2. Scan artifacts → detect new types    │
│    → add to rules.json                  │
│ 3. Count consecutive arch changes       │
│    → warn if documentation is stale     │
│ 4. Cluster lessons by theme             │
│    → promote recurring themes to        │
│      analysis_dimensions                │
│ 5. Log evolution event to rules.json    │
│    evolution_history array              │
│ 6. Write snapshot to history/           │
│    YYYY-MM-DD_skill-evolution-vN-to-vM  │
└────────────────────────────────────────┘
```

**When it auto-triggers**: Post-commit hook runs `evolve_skill.py` if the commit changed >= 3 files. The script itself checks if enough new data exists (configurable via `evolve_every_n_iterations` in rules.json, default: 3). It also auto-triggers if >= 2 new bugs or >= 2 new architecture changes accumulated since last evolution.

**When to force**: Run `python .cursor/skills/infinigen-project-memory/scripts/evolve_skill.py --force` to bypass thresholds.

**Evolution snapshots**: Every evolution event is recorded in:
- `rules.json` → `evolution_history` array (machine-readable)
- `history/YYYY-MM-DD_skill-evolution-vN-to-vM.md` (human-readable, immutable)

## Core Scripts

| Script | Purpose | Auto/Explicit |
|--------|---------|--------------|
| `scripts/post_commit_hook.sh` | Full pipeline: commit log + STATUS regen + evolve + archive | Auto (git hook) |
| `scripts/update_status.py` | Regenerate STATUS.md from evolution.json + git state | Auto (via hook) + Explicit |
| `scripts/archive_iteration.py` | Snapshot to history/, optionally clean task_plan/findings | Auto (via hook) + Explicit |
| `scripts/evolve_skill.py` | Analyze trends, update rules.json, write evolution snapshot | Auto (via hook, conditional) + Explicit |
| `scripts/load_checkpoint.py` | Programmatic access to latest iteration record | Explicit only |

## evolution.json Schema

Each iteration record:
```json
{
  "id": 1,
  "date": "YYYY-MM-DD",
  "branch": "branch-name",
  "commit": "short-hash",
  "summary": "one-line summary",
  "completed": ["list of completed items"],
  "bugs_fixed": [{"bug": "desc", "root_cause": "why", "fix": "how"}],
  "lessons": ["actionable lesson strings"],
  "artifacts": {"key": "path"},
  "architecture_changes": ["list of arch changes"],
  "next": ["prioritized next items"]
}
```

## Portability to Other Projects

This skill is designed to be portable. To use it in another project:

1. Copy `.cursor/skills/infinigen-project-memory/` to the new project
2. Create `.project-memory/` directory with empty `evolution.json`:
   ```json
   {"project": "my-project", "architecture_version": 1, "iterations": [], "commits": []}
   ```
3. Run `cp .cursor/skills/infinigen-project-memory/scripts/post_commit_hook.sh .git/hooks/post-commit && chmod +x .git/hooks/post-commit`
4. Make your first commit -- the pipeline bootstraps automatically

The only infinigen-specific content is in `evolution.json` data (iterations, bugs, lessons). The scripts, hook, and rules engine are project-agnostic.

## Rules

- STATUS.md is the **canonical** project state. If it conflicts with other docs, STATUS.md wins.
- evolution.json is append-only for iterations (never delete old records).
- history/ files are immutable once written (no edits after archiving).
- Keep STATUS.md under 300 lines. Link to docs/ for detailed architecture.
- Always include verification commands for artifacts.
- rules.json is the only mutable config -- updated by evolve_skill.py or manually.
- `.session/checkpoint.md` is deprecated -- kept only for backward compatibility with session-resumption skill. Do not rely on it for any new logic.

#!/usr/bin/env bash
# Post-commit hook: full project-memory pipeline
#   1. Append commit metadata to evolution.json
#   2. Regenerate STATUS.md + sync checkpoint.md
#   3. Conditionally run self-evolution (every N iterations)
#   4. Clean up completed task_plan.md / findings.md
#
# Install:
#   cp .cursor/skills/infinigen-project-memory/scripts/post_commit_hook.sh .git/hooks/post-commit
#   chmod +x .git/hooks/post-commit

set -euo pipefail

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
EVO_FILE="$PROJECT_ROOT/.project-memory/evolution.json"
SCRIPTS_DIR="$PROJECT_ROOT/.cursor/skills/infinigen-project-memory/scripts"

if [ ! -f "$EVO_FILE" ]; then
    exit 0
fi

# Guard against infinite loop from auto-commits
LAST_MSG=$(git log -1 --format='%s')
if echo "$LAST_MSG" | grep -q "^Auto-update evolution.json"; then
    exit 0
fi
if echo "$LAST_MSG" | grep -q "^\[project-memory\]"; then
    exit 0
fi

# --- Step 1: Append commit metadata ---
HASH=$(git rev-parse --short HEAD)
DATE=$(git log -1 --format='%aI')
MSG=$(git log -1 --format='%s' | sed 's/"/\\"/g')
AUTHOR=$(git log -1 --format='%an')
FILES_CHANGED=$(git diff-tree --no-commit-id --name-only -r HEAD | wc -l | tr -d ' ')

python3 -c "
import json
evo_path = '$EVO_FILE'
commit = {'hash':'$HASH','date':'$DATE','message':'$MSG','author':'$AUTHOR','files_changed':$FILES_CHANGED}
with open(evo_path) as f:
    evo = json.load(f)
evo.setdefault('commits', []).append(commit)
evo['last_updated'] = commit['date']
with open(evo_path, 'w') as f:
    json.dump(evo, f, indent=2, ensure_ascii=False)
" 2>/dev/null || true

# --- Step 2: Regenerate STATUS.md + checkpoint.md ---
if [ -f "$SCRIPTS_DIR/update_status.py" ]; then
    python3 "$SCRIPTS_DIR/update_status.py" --project-root "$PROJECT_ROOT" 2>/dev/null || true
fi

# --- Step 3: Conditional self-evolution ---
# Only run evolve if this is a "significant" commit (changed >= 3 files)
if [ "$FILES_CHANGED" -ge 3 ] && [ -f "$SCRIPTS_DIR/evolve_skill.py" ]; then
    python3 "$SCRIPTS_DIR/evolve_skill.py" --project-root "$PROJECT_ROOT" 2>/dev/null || true
fi

# --- Step 4: Archive cleanup ---
# If task_plan.md exists and contains "COMPLETE", archive and reset
TASK_PLAN="$PROJECT_ROOT/task_plan.md"
if [ -f "$TASK_PLAN" ]; then
    if grep -qi "COMPLETE\|DONE\|FINISHED" "$TASK_PLAN" 2>/dev/null; then
        HISTORY_DIR="$PROJECT_ROOT/.project-memory/history"
        mkdir -p "$HISTORY_DIR"
        DATE_SLUG=$(date +%Y-%m-%d)
        TITLE_SLUG=$(echo "$MSG" | tr '[:upper:]' '[:lower:]' | tr ' /' '-' | head -c 50)
        ARCHIVE="$HISTORY_DIR/${DATE_SLUG}_${TITLE_SLUG}_taskplan.md"
        {
            echo "# Archived Task Plan: $MSG"
            echo ""
            echo "**Archived**: $(date '+%Y-%m-%d %H:%M') (auto by post-commit hook)"
            echo ""
            cat "$TASK_PLAN"
            echo ""
            if [ -f "$PROJECT_ROOT/findings.md" ]; then
                echo "---"
                echo "## Findings"
                echo ""
                cat "$PROJECT_ROOT/findings.md"
            fi
        } > "$ARCHIVE"
        # Reset working files
        echo "# Task Plan" > "$TASK_PLAN"
        echo "" >> "$TASK_PLAN"
        echo "**Status**: AWAITING NEW TASK" >> "$TASK_PLAN"
        echo "**Last Updated**: $(date '+%Y-%m-%d')" >> "$TASK_PLAN"
        if [ -f "$PROJECT_ROOT/findings.md" ]; then
            echo "# Findings" > "$PROJECT_ROOT/findings.md"
            echo "" >> "$PROJECT_ROOT/findings.md"
            echo "(No active findings)" >> "$PROJECT_ROOT/findings.md"
        fi
    fi
fi

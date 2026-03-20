#!/usr/bin/env bash
# Post-commit hook: validate-only / log-only.
# Canonical tracked project-memory sync must happen before commit at milestone finalization,
# not after commit. This hook intentionally avoids mutating tracked files so a successful
# commit leaves the worktree clean.

set -euo pipefail

GIT_BIN="$(command -v git || true)"
if [ -z "$GIT_BIN" ]; then
  for candidate in \
    /root/anaconda3/envs/infinigen/bin/git \
    /root/miniconda3/bin/git \
    /usr/bin/git \
    /bin/git; do
    if [ -x "$candidate" ]; then
      GIT_BIN="$candidate"
      break
    fi
  done
fi

if [ -z "$GIT_BIN" ]; then
  exit 0
fi

PROJECT_ROOT="$($GIT_BIN rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$PROJECT_ROOT" ]; then
  exit 0
fi

LOG_DIR="$PROJECT_ROOT/.project-memory/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/post-commit.log"

HASH="$($GIT_BIN rev-parse --short HEAD 2>/dev/null || echo unknown)"
DATE="$($GIT_BIN log -1 --format=%aI 2>/dev/null || date -Is)"
MSG="$($GIT_BIN log -1 --format=%s 2>/dev/null || echo unknown)"
FILES_CHANGED="$($GIT_BIN diff-tree --no-commit-id --name-only -r HEAD 2>/dev/null | wc -l | tr -d " " || echo 0)"
STATUS_SHORT="$($GIT_BIN status --short 2>/dev/null || true)"

{
  echo "[$DATE] commit=$HASH files_changed=$FILES_CHANGED message=$MSG"
  if [ -n "$STATUS_SHORT" ]; then
    echo "[warn] worktree-not-clean-after-commit"
    printf %sn "$STATUS_SHORT"
  else
    echo "[ok] worktree-clean-after-commit"
  fi
  echo
} >> "$LOG_FILE"

exit 0

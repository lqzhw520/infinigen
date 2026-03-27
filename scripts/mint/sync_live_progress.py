#!/usr/bin/env python3
"""Keep campaign status/watch files aligned with live D1 progress."""

from __future__ import annotations

import json
import time
from pathlib import Path

from mint_common import (
    D1_PROGRESS_PATH,
    load_state,
    sync_running_step_progress,
)

POLL_SEC = 30


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _sync_once() -> bool:
    progress = _read_json(D1_PROGRESS_PATH)
    if not progress:
        return False
    state = load_state()
    runtime = state.get("active_runtime") or {}
    worker_pid = runtime.get("worker_pid")
    candidate_key = progress.get("candidate_key")
    variant_id = progress.get("variant_id")
    active_branch = (
        f"{candidate_key}::{variant_id}" if candidate_key and variant_id else None
    )
    sync_running_step_progress(
        "d1_single_rollout_overfit",
        worker_pid=worker_pid,
        worker_kind=str(runtime.get("worker_kind") or "manual_gate_subprocess"),
        active_lane="learnability",
        active_branch=active_branch,
        active_fingerprint=progress.get("source_fingerprint"),
        latest_artifact=str(D1_PROGRESS_PATH),
        stage=progress.get("stage"),
        last_error=None,
    )
    return progress.get("status") == "running"


def main() -> int:
    while True:
        keep_running = _sync_once()
        if not keep_running:
            return 0
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    raise SystemExit(main())

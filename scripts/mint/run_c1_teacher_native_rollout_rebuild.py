#!/usr/bin/env python3
"""C1: Rebuild native teacher rollouts across predefined repair branches."""

from __future__ import annotations

import json
import os
import time

from action_contract_repair import (
    branch_configs,
    oracle_branch_search_seeds,
    run_teacher_branch,
    write_branch_progress,
)
from mint_common import (
    ARTIFACT_DIR,
    C1_BRANCH_DIR,
    reconcile_step_truth,
    sync_running_step_progress,
)

ARTIFACT = ARTIFACT_DIR / "c1_teacher_native_rollout_rebuild.json"
STEP_ID = "c1_teacher_native_rollout_rebuild"


def run() -> bool:
    seeds = oracle_branch_search_seeds()
    leaderboard = []
    for branch in branch_configs():
        summary = run_teacher_branch(branch, seeds, "oracle_handle")
        leaderboard.append(summary)
    leaderboard = sorted(
        leaderboard,
        key=lambda item: (
            int(item.get("successful_seed_count", 0)),
            int(item.get("successful_rollouts", 0)),
        ),
        reverse=True,
    )
    best = leaderboard[0] if leaderboard else {}
    passed = bool(
        best
        and best.get("successful_seed_count", 0) >= 2
        and best.get("successful_rollouts", 0) >= 4
    )
    result = {
        "gate": "c1_teacher_native_rollout_rebuild",
        "passed": passed,
        "oracle_search_seeds": seeds,
        "leaderboard": leaderboard,
        "best_branch": best.get("branch_id"),
        "branch_root": str(C1_BRANCH_DIR),
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    write_branch_progress(
        {"active_branch": None, "stage": "c1_complete", "timestamp": time.time()}
    )
    print(json.dumps(result, indent=2))
    return passed


if __name__ == "__main__":
    sync_running_step_progress(
        STEP_ID,
        worker_pid=os.getpid(),
        worker_kind="python",
        active_lane="foundation",
        active_branch="teacher_contract_rebuild",
        active_fingerprint=None,
        latest_artifact=str(ARTIFACT),
        stage="teacher_contract_rebuild",
    )
    passed = False
    error = None
    try:
        passed = run()
    except Exception as exc:  # pragma: no cover - operational reconciliation
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        reconcile_step_truth(
            STEP_ID,
            passed=passed,
            latest_artifact=str(ARTIFACT) if ARTIFACT.exists() else None,
            last_error=None
            if passed
            else error or "c1_teacher_native_rollout_rebuild_failed",
            active_branch="teacher_contract_rebuild",
        )
    raise SystemExit(0 if passed else 1)

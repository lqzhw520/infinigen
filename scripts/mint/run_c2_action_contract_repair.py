#!/usr/bin/env python3
"""C2: Apply shortlisted teacher contracts to AnyGrasp-conditioned rollouts and validate replay."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from action_contract_repair import (
    C2_REPLAY_DIR,
    branch_configs,
    c2_episode_multiplier,
    c2_seed_set,
    evaluate_replay_dir,
    run_teacher_branch,
    shortlist_oracle_branches,
)
from contract_preflight import directory_preflight
from mint_common import (
    ACTIVE_ACTION_CONTRACT_PATH,
    ARTIFACT_DIR,
    reconcile_step_truth,
    sync_running_step_progress,
    write_json_atomic,
)

ARTIFACT = ARTIFACT_DIR / "c2_action_contract_repair.json"
C2_PREFLIGHT_ARTIFACT = ARTIFACT_DIR / "c2_contract_preflight.json"
C1_ARTIFACT = ARTIFACT_DIR / "c1_teacher_native_rollout_rebuild.json"
FALLBACK_MIN_SEEDS = 4
FALLBACK_MIN_ROLLOUTS = 10
STEP_ID = "c2_action_contract_repair"


def _clear_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.glob("*"):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def run() -> bool:
    c1 = json.loads(C1_ARTIFACT.read_text()) if C1_ARTIFACT.exists() else {}
    shortlisted = shortlist_oracle_branches(c1.get("leaderboard", []))
    config_map = {
        cfg["branch_id"]: cfg
        for cfg in branch_configs(expanded=c2_episode_multiplier() > 1)
    }
    seeds = c2_seed_set()
    results = []
    selected = None
    selected_target_dir = None
    fallback_selected = None
    fallback_target_dir = None
    strict_best = None

    for branch_id in shortlisted:
        branch = dict(config_map[branch_id])
        branch["episodes_per_seed"] = (
            int(branch.get("episodes_per_seed", 4)) * c2_episode_multiplier()
        )
        teacher_summary = run_teacher_branch(branch, seeds, "anygrasp")
        branch_target_dir = C2_REPLAY_DIR.parent / f"{C2_REPLAY_DIR.name}_{branch_id}"
        replay_summary = evaluate_replay_dir(branch_id, target_dir=branch_target_dir)
        result = {
            "branch_id": branch_id,
            "teacher_summary": teacher_summary,
            "replay_summary": replay_summary,
            "action_contract": branch["action_contract"],
        }
        results.append(result)
        if replay_summary.get("passed"):
            if strict_best is None or (
                replay_summary.get("successful_seed_count", 0),
                replay_summary.get("successful_replays", 0),
            ) > (
                strict_best["replay_summary"].get("successful_seed_count", 0),
                strict_best["replay_summary"].get("successful_replays", 0),
            ):
                strict_best = result
            if selected is None or (
                replay_summary.get("successful_seed_count", 0),
                replay_summary.get("successful_replays", 0),
            ) > (
                selected["replay_summary"].get("successful_seed_count", 0),
                selected["replay_summary"].get("successful_replays", 0),
            ):
                selected = result
                selected_target_dir = branch_target_dir
        teacher_summary = result["teacher_summary"]
        if (
            teacher_summary.get("successful_seed_count", 0) >= FALLBACK_MIN_SEEDS
            and teacher_summary.get("successful_rollouts", 0) >= FALLBACK_MIN_ROLLOUTS
        ):
            if fallback_selected is None or (
                teacher_summary.get("successful_seed_count", 0),
                teacher_summary.get("successful_rollouts", 0),
            ) > (
                fallback_selected["teacher_summary"].get("successful_seed_count", 0),
                fallback_selected["teacher_summary"].get("successful_rollouts", 0),
            ):
                fallback_selected = result
                fallback_target_dir = Path(teacher_summary["rollout_dir"])

    _clear_dir(C2_REPLAY_DIR)
    contract_mode = None
    chosen = selected
    chosen_dir = selected_target_dir
    if selected and selected_target_dir:
        contract_mode = "strict_replay"
    elif fallback_selected and fallback_target_dir:
        chosen = fallback_selected
        chosen_dir = fallback_target_dir
        contract_mode = "teacher_success_fallback"

    preflight = {
        "passed": False,
        "rollout_count": 0,
        "passed_count": 0,
        "best_rollout": None,
        "rollouts": [],
    }
    if chosen and chosen_dir:
        for item in sorted(chosen_dir.glob("*")):
            shutil.copy2(item, C2_REPLAY_DIR / item.name)
        rollout_paths = sorted(C2_REPLAY_DIR.glob("*.npz"))
        preflight = directory_preflight(rollout_paths)
        write_json_atomic(C2_PREFLIGHT_ARTIFACT, preflight)
        write_json_atomic(
            ACTIVE_ACTION_CONTRACT_PATH,
            {
                "branch_id": chosen["branch_id"],
                "contract_mode": contract_mode,
                **chosen["action_contract"],
                "teacher_summary": chosen["teacher_summary"],
                "replay_summary": chosen["replay_summary"],
                "contract_preflight": {
                    "passed": preflight.get("passed"),
                    "passed_count": preflight.get("passed_count"),
                    "best_rollout": preflight.get("best_rollout"),
                },
                "selected_at": time.time(),
            },
        )
    else:
        write_json_atomic(C2_PREFLIGHT_ARTIFACT, preflight)

    result = {
        "gate": "c2_action_contract_repair",
        "passed": bool(chosen and preflight.get("passed")),
        "oracle_shortlist": shortlisted,
        "c2_seed_set": seeds,
        "results": results,
        "selected_branch": None if not chosen else chosen["branch_id"],
        "strict_best_branch": None if not strict_best else strict_best["branch_id"],
        "contract_mode": contract_mode,
        "strict_replay_passed": bool(selected),
        "successful_replays": 0
        if not chosen
        else chosen["replay_summary"].get("successful_replays", 0),
        "successful_seed_count": 0
        if not chosen
        else chosen["replay_summary"].get("successful_seed_count", 0),
        "successful_seeds": []
        if not chosen
        else chosen["replay_summary"].get("successful_seeds", []),
        "teacher_successful_seed_count": 0
        if not chosen
        else chosen["teacher_summary"]["successful_seed_count"],
        "teacher_successful_rollouts": 0
        if not chosen
        else chosen["teacher_summary"]["successful_rollouts"],
        "contract_preflight": preflight,
        "natural_replay_dir": str(C2_REPLAY_DIR),
        "error": None
        if chosen and preflight.get("passed")
        else "No valid rollout source passed contract preflight",
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    sync_running_step_progress(
        STEP_ID,
        worker_pid=os.getpid(),
        worker_kind="python",
        active_lane="dual",
        active_branch="contract_repair",
        active_fingerprint=None,
        latest_artifact=str(ARTIFACT),
        stage="action_contract_repair",
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
            last_error=None if passed else error or "c2_action_contract_repair_failed",
            active_branch="contract_repair",
        )
    raise SystemExit(0 if passed else 1)

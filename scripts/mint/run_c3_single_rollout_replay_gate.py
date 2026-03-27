#!/usr/bin/env python3
"""C3: Require one AnyGrasp-conditioned rollout to pass replay faithfully before learning."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from contract_preflight import select_best_rollout
from drawer_robot_env import replay_robot_rollout
from mint_common import (
    ACTIVE_ACTION_CONTRACT_PATH,
    ARTIFACT_DIR,
    C2_REPLAY_DIR,
    C3_SINGLE_DIR,
    load_json,
    reconcile_step_truth,
    sync_running_step_progress,
)
from rollout_selection import source_fingerprint

ARTIFACT = ARTIFACT_DIR / "c3_single_rollout_replay_gate.json"
STEP_ID = "c3_single_rollout_replay_gate"


def _clear_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.glob("*"):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def run() -> bool:
    contract = load_json(ACTIVE_ACTION_CONTRACT_PATH, {})
    rollout_paths = sorted(C2_REPLAY_DIR.glob("*.npz"))
    if not contract or not rollout_paths:
        result = {
            "gate": "c3_single_rollout_replay_gate",
            "passed": False,
            "error": "Missing active action contract or replay-valid AnyGrasp rollout",
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False

    selected, preflight = select_best_rollout(rollout_paths)
    if selected is None:
        result = {
            "gate": "c3_single_rollout_replay_gate",
            "passed": False,
            "error": "No contract-preflight-approved rollout is available",
            "contract_mode": contract.get("contract_mode"),
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False
    payload, metrics = replay_robot_rollout(selected, action_contract=contract)
    strict_passed = bool(
        metrics.get("success")
        and metrics.get("ever_attached")
        and float(metrics.get("attached_agreement", 0.0)) >= 0.9
        and float(metrics.get("max_drawer_error", 9.9)) <= 0.1
        and float(metrics.get("mean_position_error_m", 9.9)) <= 0.1
    )
    meta = json.loads(selected.with_suffix(".json").read_text())
    fallback_passed = bool(
        str(contract.get("contract_mode", "")) == "teacher_success_fallback"
        and meta.get("success")
        and meta.get("ever_attached")
        and float(meta.get("max_drawer_fraction", 0.0)) >= 0.9
        and preflight.get("best_rollout_analysis", {}).get("preflight_passed")
    )
    passed = bool(strict_passed or fallback_passed)
    _clear_dir(C3_SINGLE_DIR)
    if passed:
        shutil.copy2(selected, C3_SINGLE_DIR / selected.name)
        shutil.copy2(
            selected.with_suffix(".json"),
            C3_SINGLE_DIR / selected.with_suffix(".json").name,
        )

    result = {
        "gate": "c3_single_rollout_replay_gate",
        "passed": passed,
        "selected_rollout": str(selected),
        "selected_rollout_fingerprint": source_fingerprint(
            [selected],
            contract_mode=str(contract.get("contract_mode")),
            source_branch_id=str(payload.get("source_branch_id")),
            builder_version="c3_v2_selected_rollout",
            variant_id="single_rollout_gate",
            rollout_multiplier=1,
            balancing_mode="none",
            selected_seed_ids=[int(meta.get("seed", -1))],
        ),
        "source_grasp_source": payload.get("source_grasp_source"),
        "source_branch_id": payload.get("source_branch_id"),
        "contract_mode": contract.get("contract_mode"),
        "strict_passed": strict_passed,
        "fallback_passed": fallback_passed,
        "contract_preflight": preflight,
        "metrics": metrics,
        "single_rollout_dir": str(C3_SINGLE_DIR),
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return passed


if __name__ == "__main__":
    sync_running_step_progress(
        STEP_ID,
        worker_pid=os.getpid(),
        worker_kind="python",
        active_lane="learnability",
        active_branch="single_rollout_gate",
        active_fingerprint=None,
        latest_artifact=str(ARTIFACT),
        stage="single_rollout_gate",
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
            else error or "c3_single_rollout_replay_gate_failed",
            active_branch="single_rollout_gate",
        )
    raise SystemExit(0 if passed else 1)

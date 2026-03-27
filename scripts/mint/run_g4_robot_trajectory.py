#!/usr/bin/env python3
"""G4: Compare oracle and AnyGrasp scripted robot rollouts, then select a learning source."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
from drawer_robot_env import (
    build_oracle_grasp_pose,
    build_robot_rollout,
    save_robot_rollout,
)
from mint_common import ARTIFACT_DIR, DEFAULT_TRAIN_SEEDS

ARTIFACT = ARTIFACT_DIR / "g4_robot_trajectory.json"
AUDIT_ARTIFACT = ARTIFACT_DIR / "g4_oracle_vs_anygrasp_rollouts.json"
GRASP_CACHE = ARTIFACT_DIR / "g3_anygrasp_grasps"
RENDER_CACHE = ARTIFACT_DIR / "g3_scene_renders"
ROLLOUT_DIR = ARTIFACT_DIR / "g4_rollouts"
ROLLOUT_DIR_ANYGRASP = ARTIFACT_DIR / "g4_rollouts_anygrasp"
ROLLOUT_DIR_ORACLE = ARTIFACT_DIR / "g4_rollouts_oracle"

EPISODES_PER_SEED = 6
MIN_ORACLE_SUCCESSFUL_SEEDS = 6
MIN_ORACLE_SUCCESSFUL_ROLLOUTS = 12
MIN_ANYGRASP_SUCCESSFUL_SEEDS = 4
MIN_ANYGRASP_SUCCESSFUL_ROLLOUTS = 8


def _clear_rollout_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.glob("*"):
        if item.is_file():
            item.unlink()


def _base_pose_from_cache(cache: dict) -> np.ndarray | None:
    if cache.get("selected_grasp"):
        return np.asarray(cache["selected_grasp"]["pose_world"], dtype=np.float32)
    top = cache.get("top_candidates") or []
    if top:
        return np.asarray(top[0]["pose_world"], dtype=np.float32)
    return None


def _run_mode(
    seed: int,
    pose_world: np.ndarray,
    rollout_dir: Path,
    grasp_source: str,
    grasp_score: float | None,
) -> dict:
    seed_successes = 0
    attempts = []
    for episode_index in range(EPISODES_PER_SEED):
        rollout = build_robot_rollout(
            seed=seed,
            grasp_pose_world=pose_world,
            episode_index=episode_index,
            max_steps=96,
            grasp_source=grasp_source,
            grasp_score=grasp_score,
        )
        success = bool(rollout["success"])
        rollout_path = None
        if success:
            out_path = rollout_dir / f"seed_{seed:03d}_episode_{episode_index:02d}.npz"
            save_robot_rollout(out_path, rollout)
            rollout_path = str(out_path)
            seed_successes += 1
        attempts.append(
            {
                "episode_index": episode_index,
                "success": success,
                "steps": int(rollout["steps"]),
                "drawer_fraction_final": float(rollout["final_drawer_fraction"]),
                "ever_attached": bool(rollout["ever_attached"]),
                "rollout_path": rollout_path,
            }
        )
    return {
        "passed": seed_successes > 0,
        "successful_attempts": seed_successes,
        "attempt_count": EPISODES_PER_SEED,
        "attempts": attempts,
    }


def _mode_summary(records: list[dict]) -> dict:
    successful_seed_count = sum(
        1 for item in records if item.get("successful_attempts", 0) > 0
    )
    successful_rollouts = sum(
        int(item.get("successful_attempts", 0)) for item in records
    )
    return {
        "successful_seed_count": successful_seed_count,
        "successful_rollouts": successful_rollouts,
        "record_count": len(records),
    }


def _copy_learning_source(source_dir: Path) -> int:
    _clear_rollout_dir(ROLLOUT_DIR)
    copied = 0
    for item in sorted(source_dir.glob("*")):
        if item.is_file():
            shutil.copy2(item, ROLLOUT_DIR / item.name)
            copied += 1
    return copied


def run() -> bool:
    for directory in [ROLLOUT_DIR, ROLLOUT_DIR_ANYGRASP, ROLLOUT_DIR_ORACLE]:
        _clear_rollout_dir(directory)

    anygrasp_records = []
    oracle_records = []

    for seed in DEFAULT_TRAIN_SEEDS:
        cache_path = GRASP_CACHE / f"seed_{seed:03d}.json"
        render_path = RENDER_CACHE / f"seed_{seed:03d}.npz"
        if not cache_path.exists() or not render_path.exists():
            missing = {
                "seed": seed,
                "passed": False,
                "error": "Missing grasp cache or render cache",
                "attempt_count": 0,
                "successful_attempts": 0,
                "attempts": [],
            }
            anygrasp_records.append({**missing, "mode": "anygrasp"})
            oracle_records.append({**missing, "mode": "oracle"})
            continue

        cache = json.loads(cache_path.read_text())
        base_pose = _base_pose_from_cache(cache)
        selected = cache.get("selected_grasp")
        render = np.load(render_path)
        if base_pose is None:
            missing = {
                "seed": seed,
                "passed": False,
                "error": "No AnyGrasp pose candidate available",
                "attempt_count": 0,
                "successful_attempts": 0,
                "attempts": [],
            }
            anygrasp_records.append({**missing, "mode": "anygrasp"})
            oracle_records.append({**missing, "mode": "oracle"})
            continue

        any_pose = np.asarray(base_pose, dtype=np.float32)
        oracle_pose = build_oracle_grasp_pose(
            any_pose, render["handle_center_world"].astype(np.float32)
        )
        selected_score = float(selected.get("score", 0.0)) if selected else None

        any_result = _run_mode(
            seed, any_pose, ROLLOUT_DIR_ANYGRASP, "anygrasp", selected_score
        )
        any_result.update({"seed": seed, "selected_score": selected_score})
        anygrasp_records.append(any_result)

        oracle_result = _run_mode(
            seed, oracle_pose, ROLLOUT_DIR_ORACLE, "oracle_handle", selected_score
        )
        oracle_result.update({"seed": seed, "selected_score": selected_score})
        oracle_records.append(oracle_result)

    any_summary = _mode_summary(anygrasp_records)
    oracle_summary = _mode_summary(oracle_records)

    anygrasp_pass = bool(
        any_summary["successful_seed_count"] >= MIN_ANYGRASP_SUCCESSFUL_SEEDS
        and any_summary["successful_rollouts"] >= MIN_ANYGRASP_SUCCESSFUL_ROLLOUTS
    )
    oracle_pass = bool(
        oracle_summary["successful_seed_count"] >= MIN_ORACLE_SUCCESSFUL_SEEDS
        and oracle_summary["successful_rollouts"] >= MIN_ORACLE_SUCCESSFUL_ROLLOUTS
    )

    learning_source = None
    learning_dir = None
    # During root-cause repair we isolate learnability from perception quality,
    # so oracle-backed rollouts are the default learning source whenever they pass.
    if oracle_pass:
        learning_source = "oracle_handle"
        learning_dir = ROLLOUT_DIR_ORACLE
    elif anygrasp_pass:
        learning_source = "anygrasp"
        learning_dir = ROLLOUT_DIR_ANYGRASP

    copied_files = _copy_learning_source(learning_dir) if learning_dir else 0

    audit = {
        "gate": "g4_robot_trajectory",
        "oracle": {
            **oracle_summary,
            "min_successful_seeds": MIN_ORACLE_SUCCESSFUL_SEEDS,
            "min_successful_rollouts": MIN_ORACLE_SUCCESSFUL_ROLLOUTS,
            "passed": oracle_pass,
            "rollout_dir": str(ROLLOUT_DIR_ORACLE),
            "records": oracle_records,
        },
        "anygrasp": {
            **any_summary,
            "min_successful_seeds": MIN_ANYGRASP_SUCCESSFUL_SEEDS,
            "min_successful_rollouts": MIN_ANYGRASP_SUCCESSFUL_ROLLOUTS,
            "passed": anygrasp_pass,
            "rollout_dir": str(ROLLOUT_DIR_ANYGRASP),
            "records": anygrasp_records,
        },
        "learning_source": learning_source,
        "learning_rollout_dir": str(ROLLOUT_DIR) if learning_source else None,
        "copied_learning_files": copied_files,
        "learning_source_policy": "prefer_oracle_during_root_cause_repair",
        "timestamp": time.time(),
    }
    AUDIT_ARTIFACT.write_text(json.dumps(audit, indent=2))

    result = {
        "gate": "g4_robot_trajectory",
        "passed": bool(
            oracle_pass and learning_source is not None and copied_files > 0
        ),
        "oracle_successful_seed_count": oracle_summary["successful_seed_count"],
        "oracle_successful_rollouts": oracle_summary["successful_rollouts"],
        "anygrasp_successful_seed_count": any_summary["successful_seed_count"],
        "anygrasp_successful_rollouts": any_summary["successful_rollouts"],
        "learning_source": learning_source,
        "rollout_dir": str(ROLLOUT_DIR),
        "copied_learning_files": copied_files,
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

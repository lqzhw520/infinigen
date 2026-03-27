#!/usr/bin/env python3
"""Shared helpers for C-layer action-contract repair."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from drawer_robot_env import (
    build_native_teacher_rollout,
    build_oracle_grasp_pose,
    replay_robot_rollout,
    save_robot_rollout,
)
from mint_common import (
    ARTIFACT_DIR,
    BRANCH_PROGRESS_PATH,
    C1_BRANCH_DIR,
    C2_REPLAY_DIR,
    DEFAULT_TRAIN_SEEDS,
    ROTATION_SCALE_RAD,
    TRANSLATION_SCALE_M,
    load_json,
    load_summary,
    write_json_atomic,
)

GRASP_CACHE = ARTIFACT_DIR / "g3_anygrasp_grasps"
RENDER_CACHE = ARTIFACT_DIR / "g3_scene_renders"

MIN_REPLAY_ATTACHED_AGREEMENT = 0.90
MAX_REPLAY_MEAN_POSITION_ERROR_M = 0.10
MAX_REPLAY_DRAWER_ERROR = 0.10
MAX_REPLAY_HANDLE_DISTANCE_M = 0.10
MIN_C2_SUCCESSFUL_SEEDS = 3
MIN_C2_SUCCESSFUL_REPLAYS = 6


def _clear_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.glob("*"):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def _base_pose_from_cache(cache: dict[str, Any]) -> np.ndarray | None:
    if cache.get("selected_grasp"):
        return np.asarray(cache["selected_grasp"]["pose_world"], dtype=np.float32)
    top = cache.get("top_candidates") or []
    if top:
        return np.asarray(top[0]["pose_world"], dtype=np.float32)
    return None


def load_pose_bundle(seed: int) -> dict[str, Any] | None:
    cache_path = GRASP_CACHE / f"seed_{seed:03d}.json"
    render_path = RENDER_CACHE / f"seed_{seed:03d}.npz"
    if not cache_path.exists() or not render_path.exists():
        return None
    cache = json.loads(cache_path.read_text())
    render = np.load(render_path)
    any_pose = _base_pose_from_cache(cache)
    if any_pose is None:
        return None
    oracle_pose = build_oracle_grasp_pose(
        any_pose, render["handle_center_world"].astype(np.float32)
    )
    selected = cache.get("selected_grasp") or {}
    return {
        "anygrasp_pose_world": any_pose,
        "oracle_pose_world": oracle_pose,
        "anygrasp_score": float(selected.get("score", 0.0)) if selected else 0.0,
    }


def _successful_seeds_from_baseline(mode: str) -> list[int]:
    audit = load_json(ARTIFACT_DIR / "g4_oracle_vs_anygrasp_rollouts.json", {})
    records = audit.get(mode, {}).get("records", [])
    ranked = []
    for record in records:
        ranked.append(
            (int(record.get("successful_attempts", 0)), int(record.get("seed", -1)))
        )
    ranked = [seed for count, seed in sorted(ranked, reverse=True) if count > 0]
    return ranked


def oracle_branch_search_seeds() -> list[int]:
    seeds = _successful_seeds_from_baseline("oracle")
    return seeds[:4] if seeds else DEFAULT_TRAIN_SEEDS[:4]


def anygrasp_candidate_seeds() -> list[int]:
    seeds = _successful_seeds_from_baseline("anygrasp")
    return seeds if seeds else DEFAULT_TRAIN_SEEDS


def branch_configs(*, expanded: bool = False) -> list[dict[str, Any]]:
    rollout_multiplier = 2 if expanded else 1
    configs = [
        {
            "branch_id": "branch1_position_dominant",
            "action_contract": {
                "action_frame": "world",
                "translation_scale_m": TRANSLATION_SCALE_M,
                "rotation_scale_rad": ROTATION_SCALE_RAD,
                "attach_threshold_m": 0.12,
            },
            "phase_caps": {
                "staging": {"translation_cap": 0.45, "rotation_cap": 0.10},
                "pregrasp": {"translation_cap": 0.35, "rotation_cap": 0.08},
                "grasp": {"translation_cap": 0.20, "rotation_cap": 0.12},
                "pull": {"translation_cap": 0.28, "rotation_cap": 0.08},
                "retreat": {"translation_cap": 0.40, "rotation_cap": 0.12},
            },
            "phase_limits": {
                "staging": 24,
                "pregrasp": 24,
                "grasp": 64,
                "pull": 16,
                "retreat": 20,
            },
            "use_closed_quat_in_pregrasp": False,
            "hold_close_steps": 8,
            "close_trigger_distance_m": 0.12,
            "pull_waypoints": 16,
            "episodes_per_seed": 3 * rollout_multiplier,
        },
        {
            "branch_id": "branch2_phase_scaled",
            "action_contract": {
                "action_frame": "world",
                "translation_scale_m": TRANSLATION_SCALE_M,
                "rotation_scale_rad": ROTATION_SCALE_RAD,
                "attach_threshold_m": 0.12,
            },
            "phase_caps": {
                "staging": {"translation_cap": 0.55, "rotation_cap": 0.12},
                "pregrasp": {"translation_cap": 0.35, "rotation_cap": 0.14},
                "grasp": {"translation_cap": 0.16, "rotation_cap": 0.12},
                "pull": {"translation_cap": 0.22, "rotation_cap": 0.10},
                "retreat": {"translation_cap": 0.45, "rotation_cap": 0.12},
            },
            "phase_limits": {
                "staging": 20,
                "pregrasp": 22,
                "grasp": 72,
                "pull": 18,
                "retreat": 16,
            },
            "use_closed_quat_in_pregrasp": True,
            "hold_close_steps": 10,
            "close_trigger_distance_m": 0.12,
            "pull_waypoints": 16,
            "episodes_per_seed": 4 * rollout_multiplier,
        },
        {
            "branch_id": "branch3_explicit_gripper_phases",
            "action_contract": {
                "action_frame": "world",
                "translation_scale_m": TRANSLATION_SCALE_M,
                "rotation_scale_rad": ROTATION_SCALE_RAD,
                "attach_threshold_m": 0.12,
            },
            "phase_caps": {
                "staging": {"translation_cap": 0.45, "rotation_cap": 0.12},
                "pregrasp": {"translation_cap": 0.30, "rotation_cap": 0.10},
                "grasp": {"translation_cap": 0.16, "rotation_cap": 0.10},
                "pull": {"translation_cap": 0.24, "rotation_cap": 0.10},
                "retreat": {"translation_cap": 0.40, "rotation_cap": 0.12},
            },
            "phase_limits": {
                "staging": 24,
                "pregrasp": 24,
                "grasp": 64,
                "pull": 16,
                "retreat": 18,
            },
            "hold_open_steps": 4,
            "hold_close_steps": 12,
            "use_closed_quat_in_pregrasp": True,
            "close_trigger_distance_m": 0.12,
            "pull_waypoints": 18,
            "episodes_per_seed": 4 * rollout_multiplier,
        },
        {
            "branch_id": "branch4_densified_attach_window",
            "action_contract": {
                "action_frame": "world",
                "translation_scale_m": TRANSLATION_SCALE_M,
                "rotation_scale_rad": ROTATION_SCALE_RAD,
                "attach_threshold_m": 0.15,
            },
            "phase_caps": {
                "staging": {"translation_cap": 0.40, "rotation_cap": 0.12},
                "pregrasp": {"translation_cap": 0.28, "rotation_cap": 0.10},
                "align": {"translation_cap": 0.20, "rotation_cap": 0.08},
                "grasp": {"translation_cap": 0.12, "rotation_cap": 0.08},
                "pull": {"translation_cap": 0.22, "rotation_cap": 0.10},
                "retreat": {"translation_cap": 0.35, "rotation_cap": 0.12},
            },
            "phase_limits": {
                "staging": 24,
                "pregrasp": 26,
                "align": 16,
                "grasp": 80,
                "pull": 20,
                "retreat": 20,
            },
            "densify_grasp": True,
            "hold_close_steps": 10,
            "use_closed_quat_in_pregrasp": True,
            "close_trigger_distance_m": 0.15,
            "pull_waypoints": 20,
            "episodes_per_seed": 5 * rollout_multiplier,
        },
        {
            "branch_id": "branch5_local_frame_translation",
            "action_contract": {
                "action_frame": "local",
                "translation_scale_m": TRANSLATION_SCALE_M,
                "rotation_scale_rad": ROTATION_SCALE_RAD,
                "attach_threshold_m": 0.12,
            },
            "phase_caps": {
                "staging": {"translation_cap": 0.45, "rotation_cap": 0.12},
                "pregrasp": {"translation_cap": 0.30, "rotation_cap": 0.10},
                "grasp": {"translation_cap": 0.18, "rotation_cap": 0.10},
                "pull": {"translation_cap": 0.26, "rotation_cap": 0.08},
                "retreat": {"translation_cap": 0.38, "rotation_cap": 0.12},
            },
            "phase_limits": {
                "staging": 22,
                "pregrasp": 22,
                "grasp": 72,
                "pull": 18,
                "retreat": 18,
            },
            "hold_close_steps": 8,
            "use_closed_quat_in_pregrasp": True,
            "close_trigger_distance_m": 0.12,
            "pull_waypoints": 16,
            "episodes_per_seed": 4 * rollout_multiplier,
        },
        {
            "branch_id": "branch6_rotation_lagged_pull",
            "action_contract": {
                "action_frame": "world",
                "translation_scale_m": TRANSLATION_SCALE_M,
                "rotation_scale_rad": ROTATION_SCALE_RAD,
                "attach_threshold_m": 0.12,
            },
            "phase_caps": {
                "staging": {"translation_cap": 0.45, "rotation_cap": 0.12},
                "pregrasp": {"translation_cap": 0.30, "rotation_cap": 0.08},
                "grasp": {"translation_cap": 0.16, "rotation_cap": 0.08},
                "pull": {"translation_cap": 0.25, "rotation_cap": 0.05},
                "retreat": {"translation_cap": 0.40, "rotation_cap": 0.10},
            },
            "phase_limits": {
                "staging": 22,
                "pregrasp": 24,
                "grasp": 72,
                "pull": 18,
                "retreat": 16,
            },
            "hold_close_steps": 10,
            "use_closed_quat_in_pregrasp": True,
            "lag_pull_rotation": True,
            "close_trigger_distance_m": 0.12,
            "pull_waypoints": 18,
            "episodes_per_seed": 4 * rollout_multiplier,
        },
    ]
    for config in configs:
        config.setdefault("teacher_mode", "closed_loop_native")
        config.setdefault("native_pull_step_m", 0.018)
    return configs


def _branch_dir(branch_id: str) -> Path:
    return C1_BRANCH_DIR / branch_id


def _seed_rollout_prefix(seed: int, episode_index: int) -> str:
    return f"seed_{seed:03d}_episode_{episode_index:02d}"


def write_branch_progress(payload: dict[str, Any]) -> None:
    write_json_atomic(BRANCH_PROGRESS_PATH, payload)


def run_teacher_branch(
    branch: dict[str, Any], seeds: list[int], grasp_mode: str
) -> dict[str, Any]:
    branch_dir = _branch_dir(branch["branch_id"])
    _clear_dir(branch_dir)
    records = []
    successful_replays = 0
    successful_seeds: set[int] = set()
    for seed in seeds:
        pose_bundle = load_pose_bundle(seed)
        if pose_bundle is None:
            records.append(
                {
                    "seed": seed,
                    "error": "missing_pose_bundle",
                    "successes": 0,
                    "attempts": [],
                }
            )
            continue
        pose = (
            pose_bundle["oracle_pose_world"]
            if grasp_mode == "oracle_handle"
            else pose_bundle["anygrasp_pose_world"]
        )
        score = None if grasp_mode == "oracle_handle" else pose_bundle["anygrasp_score"]
        successes = 0
        attempts = []
        for episode_index in range(int(branch.get("episodes_per_seed", 3))):
            write_branch_progress(
                {
                    "active_branch": branch["branch_id"],
                    "seed": seed,
                    "episode_index": episode_index,
                    "grasp_mode": grasp_mode,
                    "timestamp": time.time(),
                }
            )
            rollout = build_native_teacher_rollout(
                seed=seed,
                grasp_pose_world=pose,
                branch_config=branch,
                episode_index=episode_index,
                max_steps=96,
                grasp_source=grasp_mode,
                grasp_score=score,
            )
            attempt = {
                "episode_index": episode_index,
                "success": bool(rollout["success"]),
                "steps": int(rollout["steps"]),
                "final_drawer_fraction": float(rollout["final_drawer_fraction"]),
                "max_drawer_fraction": float(rollout["max_drawer_fraction"]),
                "ever_attached": bool(rollout["ever_attached"]),
            }
            if rollout["success"]:
                out_path = (
                    branch_dir / f"{_seed_rollout_prefix(seed, episode_index)}.npz"
                )
                save_robot_rollout(out_path, rollout)
                attempt["rollout_path"] = str(out_path)
                successes += 1
                successful_replays += 1
                successful_seeds.add(seed)
            attempts.append(attempt)
        records.append({"seed": seed, "successes": successes, "attempts": attempts})
    return {
        "branch_id": branch["branch_id"],
        "grasp_mode": grasp_mode,
        "successful_seed_count": len(successful_seeds),
        "successful_rollouts": successful_replays,
        "records": records,
        "rollout_dir": str(branch_dir),
        "action_contract": branch["action_contract"],
    }


def replay_passes(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics.get("success")
        and metrics.get("ever_attached")
        and float(metrics.get("attached_agreement", 0.0))
        >= MIN_REPLAY_ATTACHED_AGREEMENT
        and float(metrics.get("mean_position_error_m", 9.9))
        <= MAX_REPLAY_MEAN_POSITION_ERROR_M
        and float(metrics.get("max_drawer_error", 9.9)) <= MAX_REPLAY_DRAWER_ERROR
        and float(metrics.get("max_handle_distance_m", 9.9))
        <= MAX_REPLAY_HANDLE_DISTANCE_M
    )


def evaluate_replay_dir(
    branch_id: str, *, target_dir: Path | None = None
) -> dict[str, Any]:
    branch_dir = _branch_dir(branch_id)
    if target_dir is None:
        target_dir = C2_REPLAY_DIR
    _clear_dir(target_dir)
    records = []
    successful_replays = 0
    successful_seeds: set[int] = set()
    for npz_path in sorted(branch_dir.glob("*.npz")):
        replay_payload, metrics = replay_robot_rollout(npz_path)
        anchored_payload, anchored_metrics = replay_robot_rollout(
            npz_path, replay_mode="state_anchored"
        )
        passed = replay_passes(metrics)
        anchored_passed = replay_passes(anchored_metrics)
        record = {
            "source": str(npz_path),
            "seed": replay_payload["seed"],
            "source_grasp_source": replay_payload["source_grasp_source"],
            "source_branch_id": replay_payload["source_branch_id"],
            "passed": passed,
            "action_contract": replay_payload["action_contract"],
            "metrics": metrics,
            "diagnostics": {
                "state_anchored_replay": {
                    "passed": anchored_passed,
                    "metrics": anchored_metrics,
                }
            },
        }
        records.append(record)
        if passed:
            successful_replays += 1
            successful_seeds.add(int(replay_payload["seed"]))
            shutil.copy2(npz_path, target_dir / npz_path.name)
            shutil.copy2(
                npz_path.with_suffix(".json"),
                target_dir / npz_path.with_suffix(".json").name,
            )
    return {
        "branch_id": branch_id,
        "successful_replays": successful_replays,
        "successful_seed_count": len(successful_seeds),
        "successful_seeds": sorted(successful_seeds),
        "records": records,
        "target_dir": str(target_dir),
        "passed": bool(
            successful_replays >= MIN_C2_SUCCESSFUL_REPLAYS
            and len(successful_seeds) >= MIN_C2_SUCCESSFUL_SEEDS
        ),
    }


def shortlist_oracle_branches(branch_summaries: list[dict[str, Any]]) -> list[str]:
    ranked = sorted(
        branch_summaries,
        key=lambda item: (
            int(item.get("successful_seed_count", 0)),
            int(item.get("successful_rollouts", 0)),
        ),
        reverse=True,
    )
    return [
        item["branch_id"]
        for item in ranked[:3]
        if item.get("successful_rollouts", 0) > 0
    ]


def current_rollout_mode() -> str:
    summary = load_summary()
    return str(summary.get("coverage_expansion_mode", "base"))


def c2_seed_set() -> list[int]:
    summary = load_summary()
    mode = str(summary.get("coverage_expansion_mode", "base"))
    if mode == "expanded":
        return list(DEFAULT_TRAIN_SEEDS)
    return anygrasp_candidate_seeds() or list(DEFAULT_TRAIN_SEEDS)


def c2_episode_multiplier() -> int:
    return 2 if current_rollout_mode() == "expanded" else 1

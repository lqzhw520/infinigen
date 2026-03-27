#!/usr/bin/env python3
"""G5: Validate the natural robot-action contract by replaying exact G4 delta actions."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
from drawer_robot_env import DrawerRobotEnv
from mint_common import ARTIFACT_DIR
from scipy.spatial.transform import Rotation as R

ARTIFACT = ARTIFACT_DIR / "g5_delta_reconstruction.json"
AUDIT_ARTIFACT = ARTIFACT_DIR / "g5_contract_audit.json"
SOURCE_DIR = ARTIFACT_DIR / "g4_rollouts"
TARGET_DIR = ARTIFACT_DIR / "g5_delta_rollouts"

MIN_SUCCESSFUL_REPLAYS = 8
MIN_SUCCESSFUL_SEEDS = 4
MAX_POSITION_ERROR_M = 0.08
MEAN_POSITION_ERROR_M = 0.02
MAX_ROTATION_ERROR_RAD = 0.35
MAX_DRAWER_ERROR = 0.08
MAX_HANDLE_DISTANCE_M = 0.08


def _action_stats(actions: np.ndarray) -> dict:
    raw_actions = np.asarray(actions, dtype=np.float32)
    clipped_fraction = (
        float(np.mean(np.abs(raw_actions) > 1.0)) if raw_actions.size else 0.0
    )
    translation_norm = (
        np.linalg.norm(raw_actions[:, :3], axis=1)
        if raw_actions.size
        else np.zeros((0,), dtype=np.float32)
    )
    rotation_norm = (
        np.linalg.norm(raw_actions[:, 3:6], axis=1)
        if raw_actions.size
        else np.zeros((0,), dtype=np.float32)
    )
    return {
        "max_abs_action": float(np.max(np.abs(raw_actions)))
        if raw_actions.size
        else 0.0,
        "clipped_fraction": clipped_fraction,
        "translation_norm_p95": float(np.percentile(translation_norm, 95))
        if len(translation_norm)
        else 0.0,
        "rotation_norm_p95": float(np.percentile(rotation_norm, 95))
        if len(rotation_norm)
        else 0.0,
    }


def _rotation_errors(replay_quats: np.ndarray, target_quats: np.ndarray) -> np.ndarray:
    errors = []
    for replay_q, target_q in zip(replay_quats, target_quats):
        errors.append(
            float((R.from_quat(replay_q) * R.from_quat(target_q).inv()).magnitude())
        )
    return np.asarray(errors, dtype=np.float32)


def _replay_rollout(npz_path: Path) -> tuple[dict, dict]:
    source = np.load(npz_path, allow_pickle=True)
    meta = json.loads(npz_path.with_suffix(".json").read_text())
    env = DrawerRobotEnv(
        seed=int(meta["seed"]),
        image_size=224,
        max_steps=max(len(source["actions"]) + 8, 96),
    )
    replay_pos = []
    replay_quat = []
    replay_drawer = []
    replay_attached = []
    replay_handle_distance = []
    success = False
    ever_attached = False

    try:
        obs = env.reset()
        env.attachment_local = {
            "pos": source["grasp_pose_local_pos"].astype(np.float32),
            "quat": source["grasp_pose_local_quat"].astype(np.float32),
        }
        for action in source["actions"].astype(np.float32):
            obs, _reward, done, info = env.step(action)
            handle_pos, _ = env.local_to_world_pose(
                source["grasp_pose_local_pos"].astype(np.float32),
                source["grasp_pose_local_quat"].astype(np.float32),
                fraction=obs.drawer_fraction,
            )
            replay_pos.append(obs.eef_pos.copy())
            replay_quat.append(obs.eef_quat.copy())
            replay_drawer.append(float(obs.drawer_fraction))
            replay_attached.append(bool(info.get("attached")))
            replay_handle_distance.append(
                float(np.linalg.norm(obs.eef_pos - handle_pos))
            )
            ever_attached = ever_attached or bool(info.get("attached"))
            if done:
                success = bool(info["is_success"])
                break
        if replay_pos:
            replay_pos_arr = np.asarray(replay_pos, dtype=np.float32)
            replay_quat_arr = np.asarray(replay_quat, dtype=np.float32)
            replay_drawer_arr = np.asarray(replay_drawer, dtype=np.float32)
            replay_attached_arr = np.asarray(replay_attached, dtype=np.bool_)
            replay_handle_distance_arr = np.asarray(
                replay_handle_distance, dtype=np.float32
            )
        else:
            replay_pos_arr = np.zeros((0, 3), dtype=np.float32)
            replay_quat_arr = np.zeros((0, 4), dtype=np.float32)
            replay_drawer_arr = np.zeros((0,), dtype=np.float32)
            replay_attached_arr = np.zeros((0,), dtype=np.bool_)
            replay_handle_distance_arr = np.zeros((0,), dtype=np.float32)
    finally:
        env.close()

    target_pos = source["next_eef_positions"].astype(np.float32)
    target_quat = source["next_eef_quaternions"].astype(np.float32)
    target_drawer = source["next_drawer_fraction"].astype(np.float32)
    target_attached = source["attached_trace"].astype(np.bool_)
    pos_error = (
        np.linalg.norm(replay_pos_arr - target_pos[: len(replay_pos_arr)], axis=1)
        if len(replay_pos_arr)
        else np.zeros((0,), dtype=np.float32)
    )
    rot_error = (
        _rotation_errors(replay_quat_arr, target_quat[: len(replay_quat_arr)])
        if len(replay_quat_arr)
        else np.zeros((0,), dtype=np.float32)
    )
    drawer_error = (
        np.abs(replay_drawer_arr - target_drawer[: len(replay_drawer_arr)])
        if len(replay_drawer_arr)
        else np.zeros((0,), dtype=np.float32)
    )
    attached_agreement = (
        float(
            np.mean(replay_attached_arr == target_attached[: len(replay_attached_arr)])
        )
        if len(replay_attached_arr)
        else 0.0
    )

    metrics = {
        "replay_steps": int(len(replay_pos_arr)),
        "max_position_error_m": float(np.max(pos_error)) if len(pos_error) else 0.0,
        "mean_position_error_m": float(np.mean(pos_error)) if len(pos_error) else 0.0,
        "max_rotation_error_rad": float(np.max(rot_error)) if len(rot_error) else 0.0,
        "max_drawer_error": float(np.max(drawer_error)) if len(drawer_error) else 0.0,
        "max_handle_distance_m": float(np.max(replay_handle_distance_arr))
        if len(replay_handle_distance_arr)
        else 0.0,
        "attached_agreement": attached_agreement,
        "success": bool(success and ever_attached),
        "ever_attached": bool(ever_attached),
        "final_drawer_fraction": float(replay_drawer_arr[-1])
        if len(replay_drawer_arr)
        else 0.0,
    }
    replay_payload = {
        "seed": int(meta["seed"]),
        "task": str(meta.get("task", "open the drawer")),
        "source_success": bool(meta.get("success", False)),
        "source_grasp_source": meta.get("grasp_source", "unknown"),
        "images": source["images"].astype(np.uint8),
        "images2": source["images2"].astype(np.uint8),
        "states": source["states"].astype(np.float32),
        "actions": source["actions"].astype(np.float32),
        "grasp_pose_local_pos": source["grasp_pose_local_pos"].astype(np.float32),
        "grasp_pose_local_quat": source["grasp_pose_local_quat"].astype(np.float32),
        "position_error": pos_error.astype(np.float32),
        "rotation_error_rad": rot_error.astype(np.float32),
        "drawer_error": drawer_error.astype(np.float32),
        "replay_attached": replay_attached_arr.astype(np.bool_),
        "target_attached": target_attached[: len(replay_attached_arr)].astype(np.bool_),
    }
    return replay_payload, metrics


def run() -> bool:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for stale in TARGET_DIR.glob("*"):
        if stale.is_file():
            stale.unlink()

    records = []
    successful_replays = 0
    successful_seeds: set[int] = set()

    for npz_path in sorted(SOURCE_DIR.glob("*.npz")):
        replay_payload, metrics = _replay_rollout(npz_path)
        action_contract = _action_stats(replay_payload["actions"])
        passed = bool(
            metrics["success"]
            and action_contract["max_abs_action"] <= 1.0001
            and action_contract["clipped_fraction"] <= 0.0
            and metrics["max_position_error_m"] <= MAX_POSITION_ERROR_M
            and metrics["mean_position_error_m"] <= MEAN_POSITION_ERROR_M
            and metrics["max_rotation_error_rad"] <= MAX_ROTATION_ERROR_RAD
            and metrics["max_drawer_error"] <= MAX_DRAWER_ERROR
            and metrics["max_handle_distance_m"] <= MAX_HANDLE_DISTANCE_M
            and metrics["attached_agreement"] >= 0.95
        )

        record = {
            "source": str(npz_path),
            "seed": replay_payload["seed"],
            "passed": passed,
            "source_grasp_source": replay_payload["source_grasp_source"],
            "action_contract": action_contract,
            "metrics": metrics,
        }
        records.append(record)

        if not passed:
            continue
        successful_replays += 1
        successful_seeds.add(int(replay_payload["seed"]))
        shutil.copy2(npz_path, TARGET_DIR / npz_path.name)
        shutil.copy2(
            npz_path.with_suffix(".json"),
            TARGET_DIR / npz_path.with_suffix(".json").name,
        )

    audit = {
        "gate": "g5_delta_reconstruction",
        "record_count": len(records),
        "successful_replays": successful_replays,
        "successful_seed_count": len(successful_seeds),
        "successful_seeds": sorted(successful_seeds),
        "min_successful_replays": MIN_SUCCESSFUL_REPLAYS,
        "min_successful_seeds": MIN_SUCCESSFUL_SEEDS,
        "records": records,
        "timestamp": time.time(),
    }
    AUDIT_ARTIFACT.write_text(json.dumps(audit, indent=2))

    result = {
        "gate": "g5_delta_reconstruction",
        "passed": bool(
            successful_replays >= MIN_SUCCESSFUL_REPLAYS
            and len(successful_seeds) >= MIN_SUCCESSFUL_SEEDS
        ),
        "record_count": len(records),
        "successful_replays": successful_replays,
        "successful_seed_count": len(successful_seeds),
        "successful_seeds": sorted(successful_seeds),
        "min_successful_replays": MIN_SUCCESSFUL_REPLAYS,
        "min_successful_seeds": MIN_SUCCESSFUL_SEEDS,
        "natural_source_dir": str(SOURCE_DIR),
        "natural_replay_dir": str(TARGET_DIR),
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

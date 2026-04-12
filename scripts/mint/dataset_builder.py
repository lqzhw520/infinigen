#!/usr/bin/env python3
"""Utilities for packing robot rollout NPZ files into LeRobot datasets."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

STATE_NAMES = [
    "eef_pos_x",
    "eef_pos_y",
    "eef_pos_z",
    "motor_proxy_0",
    "motor_proxy_1",
    "motor_proxy_2",
    "motor_proxy_3",
    "gripper_joint",  # Historical compatibility label; provenance is emitted separately.
]

ACTION_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_rx",
    "delta_ry",
    "delta_rz",
    "gripper_command",
]

LIBERO_KEY_RENAME = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.image2",
    "state": "observation.state",
    "actions": "action",
}


def dataset_integrity(root: Path, repo_id: str) -> dict:
    data_files = sorted(root.glob("data/chunk-*/*.parquet"))
    payload = {
        "dataset_root": str(root),
        "repo_id": repo_id,
        "data_files": [str(p) for p in data_files[:10]],
        "total_chunks": len(list(root.glob("data/chunk-*"))),
        "meta_info_exists": (root / "meta" / "info.json").exists(),
        "meta_stats_exists": (root / "meta" / "stats.json").exists(),
        "meta_tasks_exists": (root / "meta" / "tasks.parquet").exists(),
        "meta_provenance_exists": (root / "meta" / "provenance.json").exists(),
    }
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id=repo_id, root=str(root))
        payload["dataset_loads"] = True
        payload["dataset_length"] = len(dataset)
        payload["feature_keys"] = sorted(dataset.features.keys())
        payload["image_shape"] = str(dataset.features.get("observation.images.image", {}).get("shape", "N/A"))
        payload["state_shape"] = str(dataset.features.get("observation.state", {}).get("shape", "N/A"))
        payload["action_shape"] = str(dataset.features.get("action", {}).get("shape", "N/A"))
    except Exception as exc:  # noqa: BLE001
        payload["dataset_loads"] = False
        payload["dataset_error"] = str(exc)
        payload["dataset_length"] = 0
    payload["passed"] = bool(
        data_files
        and payload["meta_info_exists"]
        and payload["meta_stats_exists"]
        and payload["meta_tasks_exists"]
        and payload["dataset_loads"]
        and payload["dataset_length"] > 0
    )
    return payload


def _infinigen_to_libero_action(action: np.ndarray, gripper_binarize: bool) -> np.ndarray:
    return action.astype(np.float32)


def _clip_gripper_to_libero_range(gripper_joint: float) -> float:
    return float(np.clip(gripper_joint, -0.042, +0.001))


def _build_libero_state(state_npz: np.ndarray, gripper_binarize: bool) -> np.ndarray:
    state = state_npz.astype(np.float32).copy()
    state[7] = _clip_gripper_to_libero_range(state[7])
    return state


def _rollout_provenance(meta: dict[str, Any], n_frames: int) -> dict[str, Any]:
    contract_config = meta.get("contract_config") or {}
    state_spec = meta.get("state_spec") or {}
    visual_mode_report = meta.get("visual_mode_report") or {}
    return {
        "seed": meta.get("seed"),
        "claim_policy": meta.get("claim_policy", "unspecified"),
        "raw_unique_frames": int(meta.get("raw_unique_frames", n_frames)),
        "effective_training_frames": int(meta.get("effective_training_frames", n_frames)),
        "success_repeat": int(meta.get("success_repeat", 1)),
        "contract_config": contract_config,
        "state_mode": state_spec.get("state_mode", contract_config.get("state_mode", "unknown")),
        "state_spec": state_spec,
        "calibration_mode": visual_mode_report.get("calibration_mode", contract_config.get("calibration_mode", "unknown")),
        "secondary_camera_mode": visual_mode_report.get("secondary_camera_mode", contract_config.get("secondary_camera_mode", "unknown")),
        "render_profile": visual_mode_report.get("render_profile", contract_config.get("render_profile", "legacy_surface")),
        "resource_budget_snapshot": meta.get("resource_budget_snapshot", {}),
        "visual_mode_report": visual_mode_report,
    }


def build_dataset_from_rollouts(
    rollout_paths: list[Path],
    dataset_root: Path,
    repo_id: str,
    *,
    robot_type: str = "infinigen_drawer_anygrasp_robot",
    remove_existing: bool = True,
    gripper_binarize: bool = False,
    image_size: int = 256,
) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    if remove_existing and dataset_root.exists():
        shutil.rmtree(dataset_root)

    img_shape = (image_size, image_size, 3)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        root=dataset_root,
        robot_type=robot_type,
        features={
            "observation.images.image": {
                "dtype": "image",
                "shape": img_shape,
                "names": ["height", "width", "channels"],
            },
            "observation.images.image2": {
                "dtype": "image",
                "shape": img_shape,
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": STATE_NAMES,
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ACTION_NAMES,
            },
        },
        use_videos=False,
    )

    episode_count = 0
    frame_count = 0
    seeds: list[int] = []
    tasks: list[str] = []
    source_files: list[str] = []
    provenance_records: list[dict[str, Any]] = []
    raw_unique_frame_total = 0
    effective_training_frame_total = 0
    claim_policies: set[str] = set()
    state_modes: set[str] = set()
    render_profiles: set[str] = set()
    success_repeats: set[int] = set()

    for npz_path in rollout_paths:
        data = np.load(npz_path, allow_pickle=True)
        meta_path = npz_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        task = str(meta.get("task", "open the middle drawer of the cabinet"))
        source_files.append(str(npz_path))
        tasks.append(task)
        n_frames = len(data["actions"])
        provenance = _rollout_provenance(meta, n_frames)
        provenance_records.append(provenance)
        raw_unique_frame_total += int(provenance["raw_unique_frames"])
        effective_training_frame_total += int(provenance["effective_training_frames"])
        claim_policies.add(str(provenance["claim_policy"]))
        state_modes.add(str(provenance["state_mode"]))
        render_profiles.add(str(provenance.get("render_profile", "legacy_surface")))
        success_repeats.add(int(provenance["success_repeat"]))

        for idx in range(n_frames):
            raw_action = data["actions"][idx].astype(np.float32)
            action = _infinigen_to_libero_action(raw_action, gripper_binarize)
            raw_state = data["states"][idx].astype(np.float32)
            state = _build_libero_state(raw_state, gripper_binarize)

            img = Image.fromarray(data["images"][idx].astype(np.uint8))
            img = img.resize((image_size, image_size), Image.BILINEAR)
            img_np = np.array(img, dtype=np.uint8)

            img2 = Image.fromarray(data["images2"][idx].astype(np.uint8))
            img2 = img2.resize((image_size, image_size), Image.BILINEAR)
            img2_np = np.array(img2, dtype=np.uint8)

            dataset.add_frame(
                {
                    "task": task,
                    "observation.images.image": img_np,
                    "observation.images.image2": img2_np,
                    "observation.state": state,
                    "action": action,
                }
            )
            frame_count += 1
        dataset.save_episode()
        episode_count += 1
        if "seed" in meta:
            seeds.append(int(meta["seed"]))

    dataset.finalize()

    provenance_payload = {
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "unique_successful_seeds": sorted(set(seeds)),
        "raw_unique_frames": int(raw_unique_frame_total),
        "effective_training_frames": int(effective_training_frame_total),
        "effective_episode_count": int(episode_count),
        "effective_frame_count": int(frame_count),
        "claim_policies": sorted(claim_policies),
        "state_modes": sorted(state_modes),
        "render_profiles": sorted(render_profiles),
        "success_repeat_values": sorted(success_repeats),
        "records": provenance_records,
    }
    provenance_path = dataset_root / "meta" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance_payload, indent=2, ensure_ascii=False) + "\n")

    integrity = dataset_integrity(dataset_root, repo_id)
    return {
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "episode_count": episode_count,
        "frame_count": frame_count,
        "seed_coverage": sorted(set(seeds)),
        "task_coverage": sorted(set(tasks)),
        "source_files": source_files,
        "integrity": integrity,
        "unique_successful_seeds": sorted(set(seeds)),
        "raw_unique_frames": int(raw_unique_frame_total),
        "effective_training_frames": int(effective_training_frame_total),
        "effective_episode_count": int(episode_count),
        "effective_frame_count": int(frame_count),
        "claim_policies": sorted(claim_policies),
        "state_modes": sorted(state_modes),
        "render_profiles": sorted(render_profiles),
        "success_repeat_values": sorted(success_repeats),
        "provenance_path": str(provenance_path),
    }

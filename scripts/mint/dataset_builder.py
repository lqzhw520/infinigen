#!/usr/bin/env python3
"""Utilities for packing robot rollout NPZ files into LeRobot datasets."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

STATE_NAMES = [
    "eef_pos_x",
    "eef_pos_y",
    "eef_pos_z",
    "eef_quat_x",
    "eef_quat_y",
    "eef_quat_z",
    "eef_quat_w",
    "gripper_open",
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


def dataset_integrity(root: Path, repo_id: str) -> dict:
    data_files = sorted(root.glob("data/chunk-*/*.parquet"))
    payload = {
        "dataset_root": str(root),
        "repo_id": repo_id,
        "data_files": [str(p) for p in data_files[:10]],
        "meta_info_exists": (root / "meta" / "info.json").exists(),
        "meta_stats_exists": (root / "meta" / "stats.json").exists(),
        "meta_tasks_exists": (root / "meta" / "tasks.parquet").exists(),
    }
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id=repo_id, root=root, revision="main")
        payload["dataset_loads"] = True
        payload["dataset_length"] = len(dataset)
        payload["feature_keys"] = sorted(dataset.features.keys())
    except Exception as exc:
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


def build_dataset_from_rollouts(
    rollout_paths: list[Path],
    dataset_root: Path,
    repo_id: str,
    *,
    robot_type: str = "infinigen_drawer_anygrasp_robot",
    remove_existing: bool = True,
    gripper_binarize: bool = True,
) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if remove_existing and dataset_root.exists():
        shutil.rmtree(dataset_root)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        root=dataset_root,
        robot_type=robot_type,
        features={
            "observation.images.image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.image2": {
                "dtype": "image",
                "shape": (224, 224, 3),
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
    seeds = []
    tasks = []
    source_files = []
    for npz_path in rollout_paths:
        data = np.load(npz_path, allow_pickle=True)
        meta_path = npz_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        task = str(meta.get("task", "open the drawer"))
        source_files.append(str(npz_path))
        tasks.append(task)
        for idx in range(len(data["actions"])):
            action = data["actions"][idx].astype(np.float32)
            # Dataset Wrapper (Option A): Binarize gripper to match MINT VQ-VAE prior
            # Infinigen data has continuous-gradual gripper (1.0 -> -1.0 over ~5 steps)
            # MINT VQ-VAE was trained on LIBERO human teleop with discrete binary gripper
            # Binarization: if gripper < 0 -> -1.0, else -> +1.0
            if gripper_binarize:
                gripper_val = action[6]
                action = action.copy()
                action[6] = -1.0 if gripper_val < 0 else 1.0
            frame = {
                "task": task,
                "observation.images.image": data["images"][idx].astype(np.uint8),
                "observation.images.image2": data["images2"][idx].astype(np.uint8),
                "observation.state": data["states"][idx].astype(np.float32),
                "action": action,
            }
            dataset.add_frame(frame)
            frame_count += 1
        dataset.save_episode()
        episode_count += 1
        if "seed" in meta:
            seeds.append(int(meta["seed"]))
    dataset.finalize()
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
    }

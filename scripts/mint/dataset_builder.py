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
    "motor_quat_x",   # LIBERO: motor joint positions (quat-like), not eef_quat!
    "motor_quat_y",
    "motor_quat_z",
    "motor_quat_w",
    "gripper_joint",  # LIBERO: continuous joint position, NOT binary gripper_open
]

ACTION_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_rx",
    "delta_ry",
    "delta_rz",
    "gripper_command",  # LIBERO: {-1.0=close, +1.0=open}, discrete binary
]

# LIBERO feature key mapping (flat → nested via LeRobotDataset normalize)
# LeRobotDataset with normalize=True: flat parquet cols → observation.images.* / observation.state
# NOTE: LIBERO uses "actions" (plural) for action key
LIBERO_KEY_RENAME = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.image2",
    "state": "observation.state",
    "actions": "action",    # LIBERO "actions" (plural) → MINT "action" (singular)
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
    }
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        # Load from local root; LeRobotDataset accepts root without needing a HuggingFace revision
        dataset = LeRobotDataset(repo_id=repo_id, root=str(root))
        payload["dataset_loads"] = True
        payload["dataset_length"] = len(dataset)
        payload["feature_keys"] = sorted(dataset.features.keys())
        # Verify key shapes match LIBERO alignment
        payload["image_shape"] = str(dataset.features.get("observation.images.image", {}).get("shape", "N/A"))
        payload["state_shape"] = str(dataset.features.get("observation.state", {}).get("shape", "N/A"))
        payload["action_shape"] = str(dataset.features.get("action", {}).get("shape", "N/A"))
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


def _infinigen_to_libero_gripper(gripper_binary: float) -> float:
    """Map Infinigen binary gripper {0.0=closed, 1.0=open} to LIBERO continuous joint range.

    LIBERO state[7] range: [-0.042, +0.001] (negative=closed, positive=open).
    LIBERO action[6]     : {-1.0=close, +1.0=open} (discrete binary).
    We store continuous state[7] so MINT's NormalizerProcessorStep(QUANTILES) can normalize it.

    NOTE: This function is kept for API compatibility. The actual gripper_joint in the
    LeRobot dataset now comes from continuous state[7] (PyBullet finger positions mapped
    to LIBERO range). Callers should use the gripper_binary → LIBERO mapping here
    only when the input is genuinely binary.
    """
    OPEN_JOINT = 0.001
    CLOSED_JOINT = -0.042
    if gripper_binary > 0.5:
        return OPEN_JOINT
    return CLOSED_JOINT


def _infinigen_to_libero_action(action: np.ndarray, gripper_binarize: bool) -> np.ndarray:
    """Map Infinigen action to LIBERO format.

    Infinigen: action[6] ∈ {-1.0, +1.0} (already binary from drawer_robot_env.py)
    LIBERO    : actions[6] ∈ {-1.0=close, +1.0=open} — same convention, no change needed.
    """
    return action.astype(np.float32)


def _clip_gripper_to_libero_range(gripper_joint: float) -> float:
    """Clip continuous gripper_joint to LIBERO-valid range.

    PyBullet physics can drive finger positions beyond the kinematic limits during contact.
    The formula finger_pos * 1.075 - 0.042 then produces values outside [-0.042, +0.001].
    Clamp to the physically valid range to ensure the LeRobot dataset is consistent.
    """
    LIBERO_MIN = -0.042
    LIBERO_MAX = +0.001
    return float(np.clip(gripper_joint, LIBERO_MIN, LIBERO_MAX))


def _build_libero_state(state_npz: np.ndarray, gripper_binarize: bool) -> np.ndarray:
    """Build LIBERO-style 8D state from Infinigen NPZ state.

    Infinigen NPZ state[8] (verified post-fix at commit bef89ca2):
        state[0:3] = eef_pos (world frame, m)
        state[3:7] = motor_joint_positions[0:4]  ← PyBullet arm joints → matches LIBERO
        state[7]   = gripper_joint (continuous, from PyBullet finger_pos → LIBERO range)
                    Raw range: ~[-0.05, +0.07] (PyBullet [0.0, 0.04] scaled, may exceed
                    due to PyBullet contact physics pushing fingers beyond kinematic limits)

    LIBERO ground truth state[8]:
        state[0:3] = eef_pos (world frame, m)
        state[3:7] = motor_joint_positions[0:4]
        state[7]   = gripper_joint (continuous ∈ [-0.042, +0.001])

    The PyBullet → LIBERO range mapping (finger_pos * 1.075 - 0.042) is already
    applied inside drawer_robot_env.py's _state_vector(). state[7] is therefore
    already in the correct continuous format and MUST NOT be passed through
    _infinigen_to_libero_gripper() (which would incorrectly binarize it, freezing
    all frames to -0.042).

    Args:
        state_npz: Infinigen state vector (motor_joints[3:7] + gripper_joint[7]).
        gripper_binarize: Unused. Kept for API compatibility.
    """
    state = state_npz.astype(np.float32).copy()
    # state[7] is already continuous gripper_joint from PyBullet. Do NOT convert.
    # Also clamp to LIBERO-valid range in case PyBullet physics produces out-of-range
    # values during contact (fingers pushed beyond kinematic limits).
    state[7] = _clip_gripper_to_libero_range(state[7])
    return state


def build_dataset_from_rollouts(
    rollout_paths: list[Path],
    dataset_root: Path,
    repo_id: str,
    *,
    robot_type: str = "infinigen_drawer_anygrasp_robot",
    remove_existing: bool = True,
    gripper_binarize: bool = False,  # LIBERO expects continuous gripper_joint in state[7]
    image_size: int = 256,           # LIBERO native: 128→resized 256; Infinigen renders 224→resized 256
) -> dict:
    """Pack robot rollout NPZ files into a LeRobot v2 dataset aligned with MINT preprocessing.

    Key LIBERO alignments:
    - Image resolution: 256×256 (was 224×224; LIBERO 128→resized 256)
    - State[7]         : continuous gripper_joint (was binary gripper_open; LIBERO [-0.042, +0.001])
    - Action key      : LeRobotDataset flat "actions" → renamed to MINT "action" via normalize
    - Camera keys     : image → observation.images.image, image2 → observation.images.image2
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

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
        n_frames = len(data["actions"])
        for idx in range(n_frames):
            raw_action = data["actions"][idx].astype(np.float32)
            action = _infinigen_to_libero_action(raw_action, gripper_binarize)
            raw_state = data["states"][idx].astype(np.float32)
            state = _build_libero_state(raw_state, gripper_binarize)

            # Image resize: Infinigen renders 224×224 → resize to 256×256 to match LIBERO
            from PIL import Image
            import io

            img = Image.fromarray(data["images"][idx].astype(np.uint8))
            img = img.resize((image_size, image_size), Image.BILINEAR)
            img_np = np.array(img, dtype=np.uint8)

            img2 = Image.fromarray(data["images2"][idx].astype(np.uint8))
            img2 = img2.resize((image_size, image_size), Image.BILINEAR)
            img2_np = np.array(img2, dtype=np.uint8)

            frame = {
                "task": task,
                "observation.images.image": img_np,
                "observation.images.image2": img2_np,
                "observation.state": state,
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

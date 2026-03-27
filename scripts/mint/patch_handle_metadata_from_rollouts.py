#!/usr/bin/env python3
"""Write minimal handle metadata for selected seeds from rollout-local grasp poses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from drawer_robot_env import DRAWER_ROOT
from mint_common import (
    ACTIVE_TEACHER_SOURCE_PATH,
    ARTIFACT_DIR,
    STRICT_TEACHER_AUDIT_PATH,
    STRONG_ROLLOUT_AUDIT_PATH,
    now_iso,
)

TARGET_SEEDS = [2, 10, 11]
DEFAULT_HALF_EXTENTS = np.array([0.015, 0.015, 0.015], dtype=np.float32)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _metadata_path(seed: int) -> Path:
    return DRAWER_ROOT / str(seed) / "metadata.json"


def _backup_path(seed: int) -> Path:
    return DRAWER_ROOT / str(seed) / "metadata.pre_handle_patch.json"


def _rollout_local_pose(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    pos = np.asarray(data["grasp_pose_local_pos"], dtype=np.float32)
    quat = np.asarray(data["grasp_pose_local_quat"], dtype=np.float32)
    return pos, quat


def _heuristic_local_center(meta: dict[str, Any]) -> np.ndarray:
    bbox = meta.get("bounding_box") or {}
    bb_min = np.asarray(bbox.get("min", [0.0, 0.0, 0.0]), dtype=np.float32)
    bb_max = np.asarray(bbox.get("max", [0.0, 0.0, 0.0]), dtype=np.float32)
    center = 0.5 * (bb_min + bb_max)
    extents = 0.5 * (bb_max - bb_min)
    dominant_axis = int(np.argmax(np.abs(extents)))
    center[dominant_axis] += extents[dominant_axis]
    return center.astype(np.float32)


def run() -> dict[str, Any]:
    strong_audit = _load_json(STRONG_ROLLOUT_AUDIT_PATH)
    strict_audit = _load_json(STRICT_TEACHER_AUDIT_PATH)
    active_source = _load_json(ACTIVE_TEACHER_SOURCE_PATH)
    seed_to_rollouts = {
        int(item["seed"]): item.get("strong_rollouts", [])
        for item in strong_audit.get("seed_summary", [])
    }
    fallback_by_seed = {}
    for row in (
        strict_audit.get("accepted_analyses_for_mainline", [])
        or strict_audit.get("accepted_analyses_by_learnability", [])
        or []
    ):
        seed = int(row.get("seed", -1))
        fallback_by_seed.setdefault(seed, []).append(row)
    patched = []
    for seed in TARGET_SEEDS:
        metadata_path = _metadata_path(seed)
        metadata = _load_json(metadata_path)
        if not metadata:
            continue
        backup = _backup_path(seed)
        if not backup.exists():
            backup.write_text(json.dumps(metadata, indent=2) + "\n")
        local_pos = None
        local_quat = None
        rollouts = seed_to_rollouts.get(seed) or []
        if rollouts:
            rollout_path = Path(rollouts[0]["rollout_path"])
            local_pos, local_quat = _rollout_local_pose(rollout_path)
            source = "strong_rollout_local_pose"
            source_rollout = str(rollout_path)
        elif fallback_by_seed.get(seed):
            rollout_path = Path(fallback_by_seed[seed][0]["rollout_path"])
            local_pos, local_quat = _rollout_local_pose(rollout_path)
            source = "fallback_accepted_rollout_local_pose"
            source_rollout = str(rollout_path)
        else:
            local_pos = _heuristic_local_center(metadata)
            local_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            source = "bbox_front_face_heuristic"
            source_rollout = None
        metadata["handle_center_local"] = local_pos.astype(float).tolist()
        metadata["handle_bbox_local"] = {
            "min": (local_pos - DEFAULT_HALF_EXTENTS).astype(float).tolist(),
            "max": (local_pos + DEFAULT_HALF_EXTENTS).astype(float).tolist(),
        }
        metadata["handle_axis_local"] = [0.0, 1.0, 0.0]
        metadata["handle_frame_local"] = {
            "pos": local_pos.astype(float).tolist(),
            "quat": local_quat.astype(float).tolist(),
        }
        metadata["handle_metadata_patch"] = {
            "patched_at": now_iso(),
            "source": source,
            "source_rollout": source_rollout,
            "active_teacher_source": active_source.get("source_dir"),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        patched.append(
            {
                "seed": seed,
                "metadata_path": str(metadata_path),
                "source": source,
                "source_rollout": source_rollout,
                "handle_center_local": metadata["handle_center_local"],
            }
        )
    payload = {
        "gate": "u3_handle_metadata_patch",
        "generated_at": now_iso(),
        "target_seeds": TARGET_SEEDS,
        "patched_seeds": patched,
        "active_teacher_source": active_source,
        "strong_rollout_audit_path": str(STRONG_ROLLOUT_AUDIT_PATH),
    }
    out_path = ARTIFACT_DIR / "u3_handle_metadata_patch.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run()

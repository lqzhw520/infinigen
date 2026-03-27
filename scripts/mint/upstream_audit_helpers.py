#!/usr/bin/env python3
"""Upstream Infinigen asset/export audit helpers for drawerbox seeds."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from drawer_robot_env import DRAWER_ROOT, DrawerRobotEnv
from mint_common import DEFAULT_HELD_OUT_SEEDS, DEFAULT_TRAIN_SEEDS

ALL_SEEDS = DEFAULT_TRAIN_SEEDS + DEFAULT_HELD_OUT_SEEDS


def seed_dir(seed: int) -> Path:
    return DRAWER_ROOT / str(seed)


def metadata_path(seed: int) -> Path:
    return seed_dir(seed) / "metadata.json"


def urdf_path(seed: int) -> Path:
    return seed_dir(seed) / "drawerbox.urdf"


def load_metadata(seed: int) -> dict[str, Any]:
    return json.loads(metadata_path(seed).read_text())


def parse_urdf(seed: int) -> ET.Element:
    return ET.parse(urdf_path(seed)).getroot()


def joint_record(seed: int) -> dict[str, Any]:
    root = parse_urdf(seed)
    prismatic = None
    for joint in root.findall("joint"):
        if joint.attrib.get("type") == "prismatic":
            prismatic = joint
            break
    if prismatic is None:
        raise RuntimeError(f"seed {seed} has no prismatic drawer joint")
    axis = np.fromstring(
        (
            prismatic.findtext("axis", default="")
            or prismatic.find("axis").attrib.get("xyz", "0 0 0")
        ),
        sep=" ",
        dtype=np.float32,
    )
    if axis.size != 3:
        axis = np.fromstring(
            prismatic.find("axis").attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float32
        )
    origin = np.fromstring(
        prismatic.find("origin").attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float32
    )
    limit = prismatic.find("limit")
    dynamics = prismatic.find("dynamics")
    return {
        "seed": seed,
        "joint_name": prismatic.attrib.get("name"),
        "joint_type": prismatic.attrib.get("type"),
        "axis_xyz": axis.tolist(),
        "axis_norm": float(np.linalg.norm(axis)),
        "axis_alignment_to_negative_y": float(
            np.dot(
                axis / max(np.linalg.norm(axis), 1e-8),
                np.array([0.0, -1.0, 0.0], dtype=np.float32),
            )
        ),
        "origin_xyz": origin.tolist(),
        "limit_lower": float(limit.attrib.get("lower", 0.0))
        if limit is not None
        else None,
        "limit_upper": float(limit.attrib.get("upper", 0.0))
        if limit is not None
        else None,
        "damping": float(dynamics.attrib.get("damping", 0.0))
        if dynamics is not None
        else None,
        "friction": float(dynamics.attrib.get("friction", 0.0))
        if dynamics is not None
        else None,
    }


def geometry_record(seed: int) -> dict[str, Any]:
    meta = load_metadata(seed)
    bbox = meta.get("bounding_box", {})
    bb_min = np.asarray(bbox.get("min", [0.0, 0.0, 0.0]), dtype=np.float32)
    bb_max = np.asarray(bbox.get("max", [0.0, 0.0, 0.0]), dtype=np.float32)
    dims = bb_max - bb_min
    return {
        "seed": seed,
        "bbox_min": bb_min.tolist(),
        "bbox_max": bb_max.tolist(),
        "bbox_dims": dims.tolist(),
        "bbox_volume": float(np.prod(np.maximum(dims, 1e-8))),
        "part_labels": meta.get("part_labels", []),
        "joint_labels": {
            key: value.get("joint label")
            for key, value in meta.items()
            if key.startswith("joint")
        },
    }


def env_record(seed: int) -> dict[str, Any]:
    env = DrawerRobotEnv(seed=seed, image_size=224, max_steps=32)
    obs = env.reset()
    anygrasp = env.anygrasp_payload()
    aabb_min, aabb_max = env.p.getAABB(
        env.drawer_id, env.drawer_joint, physicsClientId=env.client
    )
    record = {
        "seed": seed,
        "drawer_fraction": float(obs.drawer_fraction),
        "drawer_range": [float(env.drawer_low), float(env.drawer_high)],
        "drawer_travel_distance": float(env.drawer_travel_distance),
        "drawer_motion_axis": env.drawer_motion_axis.tolist(),
        "drawer_joint_index": int(env.drawer_joint),
        "drawer_aabb_world": [list(aabb_min), list(aabb_max)],
        "handle_center_world": anygrasp["handle_center_world"].tolist(),
        "image_minmax": [int(obs.image.min()), int(obs.image.max())],
        "depth_minmax": [float(obs.depth.min()), float(obs.depth.max())],
        "eef_pose": {
            "pos": obs.eef_pos.tolist(),
            "quat": obs.eef_quat.tolist(),
        },
    }
    env.close()
    return record


def robust_outliers(
    values: list[float], *, upper_scale: float = 2.5, lower_scale: float = 0.4
) -> list[int]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    median = float(np.median(arr))
    if median <= 1e-8:
        return []
    outliers = []
    for idx, value in enumerate(arr.tolist()):
        ratio = value / median
        if ratio > upper_scale or ratio < lower_scale:
            outliers.append(idx)
    return outliers


def classify_from_flags(flags: list[str]) -> str:
    if not flags:
        return "upstream_clean"
    if any(flag.endswith("_blocked") for flag in flags):
        return "upstream_blocked"
    return "upstream_suspect"


def load_rollout_metrics(rollout_json: Path) -> dict[str, Any]:
    return json.loads(rollout_json.read_text())


def finite_mean(values: list[float]) -> float | None:
    valid = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(valid)) if valid else None

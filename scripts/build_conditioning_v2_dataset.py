#!/usr/bin/env python3
"""Build PhysNAP-compatible conditioning datasets from Infinigen Phase-1 exports."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pybullet as p
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN_DIR = PROJECT_ROOT / "experiments" / "physnap" / "box_conditioning_v2"
BOX_SAMPLE_ROOTS = {
    "mailer": PROJECT_ROOT / "sim_exports" / "data_engine" / "phase1_1k_mailer" / "dataset" / "train",
    "drawer": PROJECT_ROOT / "sim_exports" / "data_engine" / "phase1_1k_drawer" / "dataset" / "train",
    "slip_lid": PROJECT_ROOT / "sim_exports" / "data_engine" / "phase1_1k_sliplid" / "dataset" / "train",
    "tuck_end": PROJECT_ROOT / "sim_exports" / "data_engine" / "phase1_1k_tuckend" / "dataset" / "train",
}


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def load_manifest(campaign_dir: Path) -> dict:
    return yaml.safe_load((campaign_dir / "manifest.yaml").read_text()) or {}


def conditioning_cfg(manifest: dict) -> dict:
    return manifest.get("conditioning", {}) or {}


def plan_path(campaign_dir: Path, manifest: dict) -> Path:
    rel = conditioning_cfg(manifest).get("plan_path", "experiments/physnap/box_conditioning_v2/runtime/conditioning_plan.json")
    path = Path(rel)
    return path if path.is_absolute() else PROJECT_ROOT / path


def output_root(campaign_dir: Path, manifest: dict) -> Path:
    rel = conditioning_cfg(manifest).get("output_root", str(campaign_dir / "conditioning"))
    path = Path(rel)
    return path if path.is_absolute() else PROJECT_ROOT / path


def choose_sample_plan(campaign_dir: Path, manifest: dict) -> dict:
    cfg = conditioning_cfg(manifest)
    samples_per_box = int(cfg.get("samples_per_box", 8))
    plan = {
        "samples_per_box": samples_per_box,
        "box_types": {},
    }
    for box_type, root in BOX_SAMPLE_ROOTS.items():
        sample_dirs = sorted(path for path in root.iterdir() if path.is_dir())[:samples_per_box]
        plan["box_types"][box_type] = [
            {
                "box_type": box_type,
                "sample_id": sample_dir.name,
                "sample_dir": str(sample_dir),
                "metadata_path": str(sample_dir / "metadata.json"),
                "urdf_path": str(sample_dir / "urdf_gt.urdf"),
            }
            for sample_dir in sample_dirs
        ]
    save_json(plan_path(campaign_dir, manifest), plan)
    return plan


def read_plan(campaign_dir: Path, manifest: dict) -> dict:
    path = plan_path(campaign_dir, manifest)
    if path.exists():
        return load_json(path, {})
    return choose_sample_plan(campaign_dir, manifest)


def load_joint_limits(urdf_path: Path) -> list[tuple[str, float, float]]:
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    joints = []
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type", "")
        if joint_type not in {"revolute", "prismatic"}:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lower = float(limit.attrib.get("lower", "0"))
        upper = float(limit.attrib.get("upper", "0"))
        joints.append((joint.attrib["name"], lower, upper))
    return joints


def joint_state_dict(urdf_path: Path, state_mode: str) -> list[dict[str, float]]:
    joints = load_joint_limits(urdf_path)
    if not joints:
        return [{}]
    if state_mode == "zero":
        return [{name: 0.0 for name, _, _ in joints}]
    ratios = [0.0, 0.5, 0.9]
    states = []
    for ratio in ratios:
        state = {}
        for name, lower, upper in joints:
            state[name] = lower + (upper - lower) * ratio
        states.append(state)
    return states


def camera_specs(view_mode: str) -> list[dict[str, float]]:
    if view_mode == "single":
        return [{"yaw": 180.0, "pitch": -50.0}]
    return [
        {"yaw": 150.0, "pitch": -50.0},
        {"yaw": 180.0, "pitch": -55.0},
        {"yaw": 210.0, "pitch": -50.0},
    ]


def camera_pose(target: np.ndarray, distance: float, yaw_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(-pitch_deg)
    eye = target + np.array(
        [
            distance * math.cos(pitch) * math.cos(yaw),
            distance * math.cos(pitch) * math.sin(yaw),
            distance * math.sin(pitch),
        ],
        dtype=np.float32,
    )
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up_guess = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, up_guess)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    return eye, target.astype(np.float32), up.astype(np.float32)


def backproject_depth(depth_buffer: np.ndarray, seg: np.ndarray, width: int, height: int, fov_deg: float, near: float, far: float, eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    valid = (seg >= 0) & (depth_buffer < 0.999)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    depth = far * near / (far - (far - near) * depth_buffer[valid])
    fx = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    fy = fx
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    u = us[valid]
    v = vs[valid]
    x_cam = (u - cx) / fx * depth
    y_cam = -(v - cy) / fy * depth
    z_cam = depth

    forward = (target - eye).astype(np.float32)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up).astype(np.float32)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward).astype(np.float32)
    world = eye[None, :] + x_cam[:, None] * right[None, :] + y_cam[:, None] * true_up[None, :] + z_cam[:, None] * forward[None, :]
    return world.astype(np.float32)


def sample_cloud(points: np.ndarray, target_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((target_points, 3), dtype=np.float32)
    if len(points) >= target_points:
        idx = rng.choice(len(points), size=target_points, replace=False)
    else:
        idx = rng.choice(len(points), size=target_points, replace=True)
    return points[idx].astype(np.float32)


def render_condition_observations(sample_dir: Path, state_mode: str, view_mode: str, width: int, height: int, points_per_cloud: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    metadata = load_json(sample_dir / "metadata.json", {})
    bbox = metadata.get("bbox_world", {})
    bbox_min = np.array(bbox.get("min", [-0.2, -0.2, -0.2]), dtype=np.float32)
    bbox_max = np.array(bbox.get("max", [0.2, 0.2, 0.2]), dtype=np.float32)
    target = ((bbox_min + bbox_max) / 2.0).astype(np.float32)
    extent = np.maximum(bbox_max - bbox_min, 1e-3)
    distance = float(np.linalg.norm(extent) * 2.5)
    distance = max(distance, 0.6)
    urdf_path = sample_dir / "urdf_gt.urdf"

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        p.setAdditionalSearchPath(str(sample_dir), physicsClientId=client)
        body_id = p.loadURDF(str(urdf_path), useFixedBase=True, physicsClientId=client)
        joint_name_to_index = {
            p.getJointInfo(body_id, idx, physicsClientId=client)[1].decode("utf-8"): idx
            for idx in range(p.getNumJoints(body_id, physicsClientId=client))
        }

        merged_points = []
        rgb_frames = []
        state_specs = joint_state_dict(urdf_path, state_mode)
        view_specs = camera_specs(view_mode)
        for state_index, joint_positions in enumerate(state_specs):
            for joint_name, qpos in joint_positions.items():
                if joint_name in joint_name_to_index:
                    p.resetJointState(body_id, joint_name_to_index[joint_name], targetValue=qpos, physicsClientId=client)
            for view_index, spec in enumerate(view_specs):
                eye, target_vec, up = camera_pose(target, distance, spec["yaw"], spec["pitch"])
                view_matrix = p.computeViewMatrix(cameraEyePosition=eye.tolist(), cameraTargetPosition=target_vec.tolist(), cameraUpVector=up.tolist())
                projection_matrix = p.computeProjectionMatrixFOV(fov=60.0, aspect=float(width) / float(height), nearVal=0.05, farVal=4.0)
                _, _, rgb, depth, seg = p.getCameraImage(
                    width=width,
                    height=height,
                    viewMatrix=view_matrix,
                    projectionMatrix=projection_matrix,
                    renderer=p.ER_TINY_RENDERER,
                    physicsClientId=client,
                )
                rgb = np.reshape(rgb, (height, width, 4))[..., :3].astype(np.uint8)
                depth = np.reshape(depth, (height, width)).astype(np.float32)
                seg = np.reshape(seg, (height, width)).astype(np.int32)
                points = backproject_depth(depth, seg, width, height, 60.0, 0.05, 4.0, eye, target_vec, up)
                merged_points.append(points)
                rgb_frames.append({"state_index": state_index, "view_index": view_index, "rgb": rgb})

        cloud = np.concatenate(merged_points, axis=0) if merged_points else np.zeros((0, 3), dtype=np.float32)
        cloud = sample_cloud(cloud, points_per_cloud, rng)
        preview = rgb_frames[0]["rgb"] if rgb_frames else np.zeros((height, width, 3), dtype=np.uint8)
        metadata_out = {
            "sample_dir": str(sample_dir),
            "box_type": metadata.get("box_type"),
            "state_mode": state_mode,
            "view_mode": view_mode,
            "n_states": len(state_specs),
            "n_views": len(view_specs),
            "bbox_center": target.tolist(),
            "distance": distance,
        }
        return cloud, preview, metadata_out
    finally:
        p.disconnect(physicsClientId=client)


def export_group(campaign_dir: Path, manifest: dict, group_id: str) -> dict:
    cfg = conditioning_cfg(manifest)
    group = cfg.get("groups", {}).get(group_id)
    if not group:
        raise ValueError(f"Unknown conditioning group: {group_id}")
    plan = read_plan(campaign_dir, manifest)
    target_dir = output_root(campaign_dir, manifest) / group_id
    rgb_dir = target_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    pcs = []
    categories = []
    object_ids = []
    metadata_rows = []
    width = int(cfg.get("render_width", 256))
    height = int(cfg.get("render_height", 256))
    points_per_cloud = int(cfg.get("points_per_cloud", 1000))

    global_index = 0
    for box_type, samples in plan.get("box_types", {}).items():
        for sample in samples:
            sample_dir = Path(sample["sample_dir"])
            cloud, preview, meta = render_condition_observations(
                sample_dir,
                group["state_mode"],
                group["view_mode"],
                width,
                height,
                points_per_cloud,
                rng,
            )
            pcs.append(cloud)
            categories.append(box_type)
            object_ids.append(f"{box_type}:{sample_dir.name}")
            meta["object_id"] = object_ids[-1]
            metadata_rows.append(meta)
            Image.fromarray(preview).save(rgb_dir / f"{global_index:04}.png")
            global_index += 1

    target_dir.mkdir(parents=True, exist_ok=True)
    np.save(target_dir / "pcs.npy", np.stack(pcs, axis=0))
    (target_dir / "categories.txt").write_text("\n".join(categories) + "\n")
    (target_dir / "object_ids.txt").write_text("\n".join(object_ids) + "\n")
    save_json(target_dir / "metadata.json", {"group_id": group_id, "objects": metadata_rows})
    summary = {
        "group_id": group_id,
        "cond_dir": str(target_dir),
        "n_objects": len(object_ids),
        "points_per_cloud": points_per_cloud,
        "state_mode": group["state_mode"],
        "view_mode": group["view_mode"],
    }
    save_json(campaign_dir / "runtime" / f"{group_id}_export_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build conditioning datasets for Phase 2")
    parser.add_argument("--campaign-dir", default=str(DEFAULT_CAMPAIGN_DIR))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare", help="Create deterministic sample selection plan")
    export_parser = subparsers.add_parser("export", help="Export one conditioning group")
    export_parser.add_argument("--group-id", required=True)
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir).resolve()
    manifest = load_manifest(campaign_dir)
    if args.command == "prepare":
        plan = choose_sample_plan(campaign_dir, manifest)
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    if args.command == "export":
        payload = export_group(campaign_dir, manifest, args.group_id)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

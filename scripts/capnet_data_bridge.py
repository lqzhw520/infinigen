#!/usr/bin/env python3
"""Convert Phase-1 dataset to CAPNet-compatible format.

For each sample, computes:
  - Per-link 6D pose (R, t) via forward kinematics from URDF + joint_state
  - Per-link 3D size from mesh AABB
  - Per-link AABB in world frame
  - NPCS-ready per-link normalized coordinates

CAPNet (CVPR'25 Highlight, arXiv:2504.11230) requires:
  - Part-level segmentation (already available in segmentation.npy)
  - Per-part 6D pose (rotation R, translation t)
  - Per-part 3D size s (bounding box extents)
  - NPCS maps (Normalized Part Coordinate Space)

Usage:
    python scripts/capnet_data_bridge.py sim_exports/data_engine/phase1_1k_mailer/
    python scripts/capnet_data_bridge.py sim_exports/data_engine/phase1_1k_mailer/ --max-samples 50
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def _parse_vec3(attr: str | None) -> np.ndarray:
    if not attr:
        return np.zeros(3)
    return np.array([float(x) for x in attr.strip().split()])


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    return Rotation.from_euler("xyz", rpy).as_matrix()


def _mat4_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    return T


def _revolute_transform(axis: np.ndarray, angle: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
    return T


def _prismatic_transform(axis: np.ndarray, displacement: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = axis * displacement
    return T


def _load_obj_vertices(obj_path: Path) -> np.ndarray:
    verts = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts) if verts else np.zeros((0, 3))


def compute_forward_kinematics(urdf_path: Path, joint_positions: dict[str, float]) -> dict[str, np.ndarray]:
    """Compute per-link 4x4 world poses via FK from URDF and joint states."""
    root = ET.parse(str(urdf_path)).getroot()

    link_names = [l.get("name") for l in root.iter("link")]
    joints = []
    for j in root.iter("joint"):
        origin = j.find("origin")
        xyz = _parse_vec3(origin.get("xyz") if origin is not None else None)
        rpy = _parse_vec3(origin.get("rpy") if origin is not None else None)
        axis_el = j.find("axis")
        axis = _parse_vec3(axis_el.get("xyz") if axis_el is not None else None)
        if np.linalg.norm(axis) > 0:
            axis = axis / np.linalg.norm(axis)
        joints.append({
            "name": j.get("name"),
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": xyz, "rpy": rpy, "axis": axis,
        })

    poses = {}
    # Find root link (link with no parent joint pointing to it as child)
    child_links = {j["child"] for j in joints}
    root_links = [n for n in link_names if n not in child_links]
    for rl in root_links:
        poses[rl] = np.eye(4)

    visited = set(poses.keys())
    changed = True
    while changed:
        changed = False
        for j in joints:
            if j["child"] in visited:
                continue
            if j["parent"] not in visited:
                continue

            parent_pose = poses[j["parent"]]
            joint_origin = _mat4_from_xyz_rpy(j["xyz"], j["rpy"])

            jtype = j["type"]
            q = joint_positions.get(j["name"], 0.0)

            if jtype == "revolute" or jtype == "continuous":
                joint_motion = _revolute_transform(j["axis"], q)
            elif jtype == "prismatic":
                joint_motion = _prismatic_transform(j["axis"], q)
            else:
                joint_motion = np.eye(4)

            poses[j["child"]] = parent_pose @ joint_origin @ joint_motion
            visited.add(j["child"])
            changed = True

    return poses


def compute_link_aabb_and_size(urdf_path: Path, link_poses: dict[str, np.ndarray]) -> dict[str, dict]:
    """Compute per-link AABB and size from mesh vertices transformed to world frame."""
    root = ET.parse(str(urdf_path)).getroot()
    sample_dir = urdf_path.parent
    result = {}

    for link in root.iter("link"):
        lname = link.get("name")
        if lname == "world":
            continue
        pose = link_poses.get(lname, np.eye(4))

        all_verts_world = []
        for vis in link.iter("visual"):
            origin = vis.find("origin")
            vis_xyz = _parse_vec3(origin.get("xyz") if origin is not None else None)
            vis_rpy = _parse_vec3(origin.get("rpy") if origin is not None else None)
            vis_T = _mat4_from_xyz_rpy(vis_xyz, vis_rpy)

            mesh = vis.find(".//mesh")
            if mesh is not None:
                obj_path = sample_dir / mesh.get("filename", "")
                if obj_path.exists():
                    verts = _load_obj_vertices(obj_path)
                    if verts.shape[0] > 0:
                        verts_h = np.hstack([verts, np.ones((verts.shape[0], 1))])
                        verts_world = (pose @ vis_T @ verts_h.T).T[:, :3]
                        all_verts_world.append(verts_world)

            box = vis.find(".//box")
            if box is not None:
                size_str = box.get("size", "0.1 0.1 0.1")
                half = np.array([float(x) / 2 for x in size_str.split()])
                corners = np.array([[sx * half[0], sy * half[1], sz * half[2]]
                                    for sx in [-1, 1] for sy in [-1, 1] for sz in [-1, 1]])
                corners_h = np.hstack([corners, np.ones((8, 1))])
                corners_world = (pose @ vis_T @ corners_h.T).T[:, :3]
                all_verts_world.append(corners_world)

        if all_verts_world:
            all_v = np.vstack(all_verts_world)
            aabb_min = all_v.min(axis=0)
            aabb_max = all_v.max(axis=0)
            size = aabb_max - aabb_min
            center = (aabb_min + aabb_max) / 2
        else:
            aabb_min = np.zeros(3)
            aabb_max = np.zeros(3)
            size = np.zeros(3)
            center = pose[:3, 3]

        result[lname] = {
            "aabb_min": aabb_min.tolist(),
            "aabb_max": aabb_max.tolist(),
            "size": size.tolist(),
            "center": center.tolist(),
        }

    return result


def compute_npcs(urdf_path: Path, link_poses: dict[str, np.ndarray]) -> dict[str, dict]:
    """Compute NPCS (Normalized Part Coordinate Space) per link.

    NPCS normalizes each part's vertices to [-0.5, 0.5]^3 by centering
    at the part's local AABB center and dividing by max extent.
    Returns the normalization parameters needed to reconstruct world coords.
    """
    root = ET.parse(str(urdf_path)).getroot()
    sample_dir = urdf_path.parent
    result = {}

    for link in root.iter("link"):
        lname = link.get("name")
        if lname == "world":
            continue

        all_verts_local = []
        for vis in link.iter("visual"):
            origin = vis.find("origin")
            vis_xyz = _parse_vec3(origin.get("xyz") if origin is not None else None)
            vis_rpy = _parse_vec3(origin.get("rpy") if origin is not None else None)
            vis_T = _mat4_from_xyz_rpy(vis_xyz, vis_rpy)

            mesh = vis.find(".//mesh")
            if mesh is not None:
                obj_path = sample_dir / mesh.get("filename", "")
                if obj_path.exists():
                    verts = _load_obj_vertices(obj_path)
                    if verts.shape[0] > 0:
                        verts_h = np.hstack([verts, np.ones((verts.shape[0], 1))])
                        verts_local = (vis_T @ verts_h.T).T[:, :3]
                        all_verts_local.append(verts_local)

        if not all_verts_local:
            result[lname] = {"scale": 1.0, "offset": [0, 0, 0]}
            continue

        all_v = np.vstack(all_verts_local)
        aabb_min = all_v.min(axis=0)
        aabb_max = all_v.max(axis=0)
        center = (aabb_min + aabb_max) / 2
        extent = aabb_max - aabb_min
        scale = float(max(extent.max(), 1e-6))

        result[lname] = {
            "scale": scale,
            "offset": center.tolist(),
            "local_aabb_min": aabb_min.tolist(),
            "local_aabb_max": aabb_max.tolist(),
        }

    return result


def process_sample(sample_dir: Path) -> dict[str, Any] | None:
    """Convert a single Phase-1 sample to CAPNet format."""
    urdf_path = sample_dir / "urdf_gt.urdf"
    js_path = sample_dir / "joint_state.json"
    meta_path = sample_dir / "metadata.json"

    if not urdf_path.exists() or not js_path.exists():
        return None

    js_data = json.loads(js_path.read_text())
    raw_pos = js_data.get("joint_positions", {})
    if isinstance(raw_pos, dict):
        pos_map = {k: float(v) for k, v in raw_pos.items()}
    elif isinstance(raw_pos, list):
        names = js_data.get("joint_names_order", [])
        pos_map = {n: float(raw_pos[i]) for i, n in enumerate(names) if i < len(raw_pos)}
    else:
        pos_map = {}

    link_poses = compute_forward_kinematics(urdf_path, pos_map)
    aabb_info = compute_link_aabb_and_size(urdf_path, link_poses)
    npcs_info = compute_npcs(urdf_path, link_poses)

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    link_annotations = {}
    for lname, pose_4x4 in link_poses.items():
        if lname == "world":
            continue
        R = pose_4x4[:3, :3]
        t = pose_4x4[:3, 3]
        quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]

        aabb = aabb_info.get(lname, {})
        npcs = npcs_info.get(lname, {})

        link_annotations[lname] = {
            "rotation_matrix": R.tolist(),
            "quaternion_xyzw": quat.tolist(),
            "translation": t.tolist(),
            "size": aabb.get("size", [0, 0, 0]),
            "aabb_min": aabb.get("aabb_min", [0, 0, 0]),
            "aabb_max": aabb.get("aabb_max", [0, 0, 0]),
            "aabb_center": aabb.get("center", [0, 0, 0]),
            "npcs_scale": npcs.get("scale", 1.0),
            "npcs_offset": npcs.get("offset", [0, 0, 0]),
        }

    # Object-level pose (from base link or link_0)
    base_link = None
    for name in ["link_0", "base_link"]:
        if name in link_poses:
            base_link = name
            break
    if base_link is None and link_poses:
        base_link = next(n for n in link_poses if n != "world")

    obj_R = link_poses[base_link][:3, :3] if base_link else np.eye(3)
    obj_t = link_poses[base_link][:3, 3] if base_link else np.zeros(3)
    obj_quat = Rotation.from_matrix(obj_R).as_quat()

    all_verts_world = []
    for aabb in aabb_info.values():
        mn = np.array(aabb["aabb_min"])
        mx = np.array(aabb["aabb_max"])
        all_verts_world.extend([mn, mx])
    if all_verts_world:
        all_pts = np.array(all_verts_world)
        obj_aabb_min = all_pts.min(0)
        obj_aabb_max = all_pts.max(0)
    else:
        obj_aabb_min = np.zeros(3)
        obj_aabb_max = np.zeros(3)

    cam_data = {}
    cam_path = sample_dir / "camera_intrinsics.json"
    if cam_path.exists():
        cam_raw = json.loads(cam_path.read_text())
        cam_data = {
            "K": cam_raw.get("K"),
            "T_cam_to_world": cam_raw.get("T_cam_to_world"),
            "HW": cam_raw.get("HW"),
        }

    return {
        "sample_id": sample_dir.name,
        "box_type": meta.get("box_type", js_data.get("box_type", "unknown")),
        "seed": meta.get("seed", js_data.get("seed")),
        "joint_state": pos_map,
        "camera": cam_data,
        "object_pose": {
            "rotation_matrix": obj_R.tolist(),
            "quaternion_xyzw": obj_quat.tolist(),
            "translation": obj_t.tolist(),
            "aabb_min": obj_aabb_min.tolist(),
            "aabb_max": obj_aabb_max.tolist(),
            "size": (obj_aabb_max - obj_aabb_min).tolist(),
        },
        "link_annotations": link_annotations,
        "n_parts": len(link_annotations),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert Phase-1 dataset to CAPNet format")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output dir for CAPNet annotations (default: <dataset>/capnet_annotations/)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    train_dir = dataset_dir / "dataset" / "train"
    if not train_dir.exists():
        print(f"ERROR: {train_dir} not found")
        return 1

    output_dir = args.output_dir or (dataset_dir / "capnet_annotations")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(d for d in train_dir.iterdir() if d.is_dir() and d.name.isdigit())
    if args.max_samples > 0:
        sample_dirs = sample_dirs[:args.max_samples]

    print(f"Converting {len(sample_dirs)} samples from {dataset_dir}")

    all_link_annots = []
    all_obj_annots = []

    for i, sd in enumerate(sample_dirs):
        result = process_sample(sd)
        if result is None:
            print(f"  [{i+1}] {sd.name}: SKIP (missing files)")
            continue

        per_sample_path = output_dir / f"{sd.name}.json"
        per_sample_path.write_text(json.dumps(result, indent=2))

        all_link_annots.append({
            "sample_id": result["sample_id"],
            "link_annotations": result["link_annotations"],
        })
        all_obj_annots.append({
            "sample_id": result["sample_id"],
            "box_type": result["box_type"],
            "object_pose": result["object_pose"],
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(sample_dirs)}] ...")

    (output_dir / "link_pos_quat_aabb.json").write_text(
        json.dumps(all_link_annots, indent=2))
    (output_dir / "obj_pos_quat_aabb.json").write_text(
        json.dumps(all_obj_annots, indent=2))

    summary = {
        "dataset": str(dataset_dir),
        "n_samples": len(all_obj_annots),
        "n_link_annotations": sum(len(x["link_annotations"]) for x in all_link_annots),
        "box_types": list({x["box_type"] for x in all_obj_annots}),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nDone. {len(all_obj_annots)} samples -> {output_dir}")
    print(f"  link_pos_quat_aabb.json: {len(all_link_annots)} entries")
    print(f"  obj_pos_quat_aabb.json: {len(all_obj_annots)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

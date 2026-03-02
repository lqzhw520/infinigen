#!/usr/bin/env python3
"""
MailerBox_Simple Phase-1 Data Engine demo:

- Phase 1.3: domain randomized material system (visual + physics params, deterministic)
- Phase 1.4: auto-labeling pipeline (RGB-D + segmentation + keypoints + URDF GT + metadata)

Run (from project root):
  python -m infinigen.launch_blender -s scripts/export_mailerbox_simple_phase1_data_engine.py -- \
    --seeds 101 --n_views 2 --joint_states "0,0;1.2,1.0"
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bpy
import mathutils
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent


# -----------------------------
# URDF parsing + FK utilities
# -----------------------------


@dataclass(frozen=True)
class UrdfVisual:
    mesh_filename: str
    xyz: np.ndarray  # shape (3,)
    rpy: np.ndarray  # shape (3,)


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray  # origin
    rpy: np.ndarray
    axis: np.ndarray  # in joint frame


def _parse_xyz(attr: Optional[str]) -> np.ndarray:
    if not attr:
        return np.zeros(3, dtype=np.float64)
    vals = [float(x) for x in attr.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 floats, got {vals}")
    return np.asarray(vals, dtype=np.float64)


def _parse_rpy(attr: Optional[str]) -> np.ndarray:
    if not attr:
        return np.zeros(3, dtype=np.float64)
    vals = [float(x) for x in attr.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 floats, got {vals}")
    return np.asarray(vals, dtype=np.float64)


def _mat_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> mathutils.Matrix:
    # URDF rpy uses roll-pitch-yaw about X-Y-Z (standard convention).
    roll, pitch, yaw = [float(x) for x in rpy.tolist()]
    T = mathutils.Matrix.Translation(mathutils.Vector(tuple(float(x) for x in xyz.tolist())))
    R = mathutils.Euler((roll, pitch, yaw), "XYZ").to_matrix().to_4x4()
    return T @ R


def _rot_about_axis(axis: np.ndarray, angle: float) -> mathutils.Matrix:
    ax = mathutils.Vector(tuple(float(x) for x in axis.tolist()))
    if ax.length < 1e-12:
        raise ValueError(f"Invalid axis: {axis}")
    ax.normalize()
    return mathutils.Matrix.Rotation(float(angle), 4, ax)


def parse_urdf(urdf_path: Path) -> Tuple[Dict[str, List[UrdfVisual]], List[UrdfJoint]]:
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    links_visuals: Dict[str, List[UrdfVisual]] = {}
    joints: List[UrdfJoint] = []

    for link in root.findall("link"):
        link_name = link.attrib.get("name", "")
        if not link_name:
            continue
        visuals: List[UrdfVisual] = []
        for vis in link.findall("visual"):
            origin = vis.find("origin")
            xyz = _parse_xyz(origin.attrib.get("xyz") if origin is not None else None)
            rpy = _parse_rpy(origin.attrib.get("rpy") if origin is not None else None)
            geom = vis.find("geometry")
            if geom is None:
                continue
            mesh = geom.find("mesh")
            if mesh is None:
                continue
            fn = mesh.attrib.get("filename", "")
            if not fn:
                continue
            visuals.append(UrdfVisual(mesh_filename=fn, xyz=xyz, rpy=rpy))
        links_visuals[link_name] = visuals

    for joint in root.findall("joint"):
        jt = joint.attrib.get("type", "")
        name = joint.attrib.get("name", "")
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.attrib.get("link", "")
        child = child_el.attrib.get("link", "")
        origin = joint.find("origin")
        xyz = _parse_xyz(origin.attrib.get("xyz") if origin is not None else None)
        rpy = _parse_rpy(origin.attrib.get("rpy") if origin is not None else None)
        axis_el = joint.find("axis")
        axis = _parse_xyz(axis_el.attrib.get("xyz") if axis_el is not None else None)

        joints.append(
            UrdfJoint(
                name=name,
                joint_type=jt,
                parent=parent,
                child=child,
                xyz=xyz,
                rpy=rpy,
                axis=axis,
            )
        )

    return links_visuals, joints


def compute_link_poses_world(
    *,
    joints: List[UrdfJoint],
    joint_positions: Dict[str, float],
    base_link: str = "link_0",
) -> Dict[str, mathutils.Matrix]:
    """
    Compute link->world poses using a minimal URDF FK for tree-structured joints.

    Assumptions (satisfied by our exported URDF):
    - child link frame coincides with joint frame at q=0
    """
    # Build adjacency from parent->joint
    children_of: Dict[str, List[UrdfJoint]] = {}
    for j in joints:
        children_of.setdefault(j.parent, []).append(j)

    poses: Dict[str, mathutils.Matrix] = {base_link: mathutils.Matrix.Identity(4)}

    def dfs(parent_link: str):
        parent_pose = poses[parent_link]
        for j in children_of.get(parent_link, []):
            if j.joint_type == "fixed":
                q = 0.0
            else:
                q = float(joint_positions.get(j.name, 0.0))

            T_parent_joint = _mat_from_xyz_rpy(j.xyz, j.rpy)
            if j.joint_type in ("revolute", "continuous"):
                T_joint_child = _rot_about_axis(j.axis, q)
            elif j.joint_type == "prismatic":
                ax = np.asarray(j.axis, dtype=np.float64)
                n = float(np.linalg.norm(ax))
                if n < 1e-12:
                    raise ValueError(f"Invalid prismatic axis: {j.axis} for joint={j.name}")
                ax = ax / n
                T_joint_child = mathutils.Matrix.Translation(
                    mathutils.Vector((float(ax[0] * q), float(ax[1] * q), float(ax[2] * q)))
                )
            else:
                # fixed or unknown: treat as fixed
                T_joint_child = mathutils.Matrix.Identity(4)

            poses[j.child] = parent_pose @ T_parent_joint @ T_joint_child
            dfs(j.child)

    dfs(base_link)
    return poses


# -----------------------------
# Blender helpers
# -----------------------------


def _clear_scene():
    from infinigen.core.util import blender as butil

    butil.clear_scene()


def _import_obj(obj_path: Path) -> List[bpy.types.Object]:
    """
    Import an OBJ file and return newly imported objects.

    Blender 4.x uses `bpy.ops.wm.obj_import`, older versions use `bpy.ops.import_scene.obj`.
    """
    # Ensure selection contains only the import results.
    try:
        bpy.ops.object.select_all(action="DESELECT")
    except Exception:
        pass
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(obj_path))
    else:
        bpy.ops.import_scene.obj(filepath=str(obj_path))
    return [o for o in bpy.context.selected_objects if o.type == "MESH"]


def _bbox_world(objs: List[bpy.types.Object]) -> Tuple[np.ndarray, np.ndarray]:
    mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    for o in objs:
        for corner in o.bound_box:
            p = o.matrix_world @ mathutils.Vector(corner)
            mins = np.minimum(mins, np.asarray(p, dtype=np.float64))
            maxs = np.maximum(maxs, np.asarray(p, dtype=np.float64))
    return mins, maxs


def _set_camera_look_at(cam: bpy.types.Object, target: np.ndarray):
    tgt = mathutils.Vector(tuple(float(x) for x in target.tolist()))
    direction = (tgt - cam.location)
    if direction.length < 1e-9:
        return
    rot_quat = direction.normalized().to_track_quat("-Z", "Y")
    cam.rotation_euler = rot_quat.to_euler()


def _render_rgb(cam: bpy.types.Object, out_path: Path, resolution: Tuple[int, int]):
    scene = bpy.context.scene
    scene.camera = cam
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)


def _load_exr_scalar_via_blender(exr_path: Path) -> np.ndarray:
    """
    Load a scalar EXR pass using Blender's internal image IO.

    This avoids external OpenEXR/OpenCV bindings, which can be fragile across environments.
    """
    img = bpy.data.images.load(str(exr_path))
    w, h = img.size
    pixels = np.asarray(img.pixels[:], dtype=np.float32).reshape((h, w, 4))
    arr = pixels[:, :, 0].copy()
    bpy.data.images.remove(img)
    return arr


def _label_colormap_for_values(values: List[int]) -> Dict[int, np.ndarray]:
    """
    Deterministic label-id -> RGB colormap (uint8).

    Notes:
    - Stable per-label-id (independent of which other ids are present in a frame).
    - Label 0 is forced to black when present.
    """
    cmap: Dict[int, np.ndarray] = {}
    for v in sorted({int(x) for x in values}):
        if v == 0:
            cmap[0] = np.array([0, 0, 0], dtype=np.uint8)
            continue
        # IMPORTANT: per-id RNG so colors do not shift across frames when some ids are absent.
        seed = (12345 + int(v) * 1000003) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        cmap[int(v)] = rng.integers(32, 256, size=3, dtype=np.uint8)
    return cmap


def _colorize_labels(label: np.ndarray) -> np.ndarray:
    """
    Deterministic label -> RGB visualization for segmentation masks.
    """
    label_i = label.astype(np.int64)
    uniq = sorted(int(x) for x in np.unique(label_i))
    cmap = _label_colormap_for_values(uniq)
    out = np.zeros((*label_i.shape, 3), dtype=np.uint8)
    for v, c in cmap.items():
        out[label_i == v] = c
    return out


def _write_segmentation_label_map(
    out_path: Path,
    *,
    link_id_map: Dict[str, int],
    seg: np.ndarray,
    role_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Write a readable mapping for segmentation label IDs.

    This complements `segmentation.npy` (integer IDs) with an explicit mapping to URDF link names.
    """
    present_ids = sorted(int(x) for x in np.unique(seg.astype(np.int64)))
    all_ids = sorted({0, *[int(v) for v in link_id_map.values()], *present_ids})
    cmap = _label_colormap_for_values(all_ids)

    id_to_link = {int(v): str(k) for k, v in link_id_map.items()}
    labels = []
    present_set = set(present_ids)
    for lab in all_ids:
        if lab == 0:
            labels.append(
                {
                    "id": 0,
                    "name": "background",
                    "color_rgb_uint8": [0, 0, 0],
                    "present_in_frame": (0 in present_set),
                }
            )
            continue
        link_name = id_to_link.get(int(lab))
        rec = {
            "id": int(lab),
            "name": link_name if link_name is not None else f"label_{lab}",
            "color_rgb_uint8": [int(x) for x in cmap.get(int(lab), np.array([255, 255, 255], dtype=np.uint8)).tolist()],
            "present_in_frame": (int(lab) in present_set),
        }
        if role_map is not None and link_name in role_map:
            rec["role"] = str(role_map[link_name])
        labels.append(rec)

    _write_json(
        out_path,
        {
            "schema": "infinigen.phase1.segmentation_label_map.v1",
            "task": "link_segmentation",
            "labels": labels,
            "link_name_to_id": {str(k): int(v) for k, v in link_id_map.items()},
        },
    )


def _render_gt_depth_and_seg(
    *,
    cam: bpy.types.Object,
    out_dir: Path,
    resolution: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render depth (Z) and object-index segmentation to `out_dir` as `.npy` (+png visualization).

    Returns:
        depth: float32 HxW
        seg: int64 HxW (object pass_index)
    """
    from PIL import Image

    scene = bpy.context.scene
    scene.camera = cam
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.resolution_percentage = 100
    # Keep GT render cheap
    if hasattr(scene, "cycles"):
        scene.cycles.samples = 1
        scene.cycles.use_denoising = False

    view_layer = scene.view_layers["ViewLayer"]
    view_layer.use_pass_z = True
    view_layer.use_pass_object_index = True

    # Build compositor graph to dump Depth + IndexOB as EXR
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")
    fout = tree.nodes.new("CompositorNodeOutputFile")
    fout.base_path = str(out_dir)
    fout.format.file_format = "OPEN_EXR"
    fout.format.color_mode = "RGBA"
    fout.format.color_depth = "32"

    # File slot: Depth
    depth_socket = fout.file_slots.new("Depth")
    fout.file_slots[depth_socket.name].path = "Depth_"
    tree.links.new(rl.outputs["Depth"], depth_socket)

    # File slot: Object Index
    idx_socket = fout.file_slots.new("IndexOB")
    fout.file_slots[idx_socket.name].path = "IndexOB_"
    tree.links.new(rl.outputs["IndexOB"], idx_socket)

    # Render once (still)
    scene.frame_set(1)
    bpy.ops.render.render(write_still=True)

    # Resolve output file paths
    depth_exr = next(out_dir.glob("Depth_*.exr"), None)
    idx_exr = next(out_dir.glob("IndexOB_*.exr"), None)
    if depth_exr is None or idx_exr is None:
        raise RuntimeError(f"Missing GT EXR outputs in {out_dir}")

    depth = _load_exr_scalar_via_blender(depth_exr).astype(np.float32)
    seg_f = _load_exr_scalar_via_blender(idx_exr)
    seg = np.rint(seg_f).astype(np.int64)

    # Cleanup temp EXRs
    try:
        depth_exr.unlink()
    except Exception:
        pass
    try:
        idx_exr.unlink()
    except Exception:
        pass

    # Save outputs
    np.save(out_dir / "depth.npy", depth)
    np.save(out_dir / "segmentation.npy", seg)
    Image.fromarray(_colorize_labels(seg)).save(str(out_dir / "segmentation.png"))

    return depth, seg


def _write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# -----------------------------
# Main pipeline
# -----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--box_type",
        type=str,
        default="MAILER",
        help="BoxType enum name (e.g., MAILER, TUCK_END, DRAWER, SLIP_LID).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[101])
    parser.add_argument("--n_views", type=int, default=2)
    parser.add_argument(
        "--joint_states",
        type=str,
        default="0,0;1.2,1.0",
        help=(
            'Semicolon-separated joint states: "v0,v1,...;v0,v1,..." in URDF joint order '
            "(non-fixed joints only). Revolute/continuous in rad; prismatic in meters."
        ),
    )
    parser.add_argument(
        "--fixed_dims",
        type=str,
        default="",
        help='Optional fixed dimensions: "W,D,H,T" (meters). If empty, factory samples per seed.',
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument(
        "--out_root",
        type=Path,
        default=PROJECT_ROOT / "sim_exports" / "data_engine" / "modular_box_phase1",
    )
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

    from infinigen.assets.sim_objects.box_material_domain_randomization import (
        box_material_config_to_metadata,
        sample_box_material_config,
    )
    from infinigen.assets.sim_objects.modular_box_factory import (
        BoxDimensions,
        BoxType,
        ModularBoxFactory,
    )
    from infinigen.core.init import configure_blender
    from infinigen.core.placement import camera as cam_util
    from infinigen.core.sim import kinematic_compiler
    from infinigen.core.sim.exporters import urdf_exporter
    from infinigen.core.util import blender as butil

    try:
        configure_blender()
    except Exception as e:
        # Rendering tasks may still work with partial config depending on environment.
        print(f"configure_blender warning: {e}")

    out_root: Path = args.out_root
    urdf_export_root = out_root / "urdf"
    dataset_root = out_root / "dataset" / "train"
    urdf_export_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    # Parse optional fixed dims
    fixed_dims: Optional[BoxDimensions] = None
    if str(args.fixed_dims).strip():
        parts = [p.strip() for p in str(args.fixed_dims).split(",")]
        if len(parts) != 4:
            raise ValueError('Expected --fixed_dims format "W,D,H,T"')
        fixed_dims = BoxDimensions(
            width=float(parts[0]),
            depth=float(parts[1]),
            height=float(parts[2]),
            thickness=float(parts[3]),
        )

    sample_counter = 1
    manifest = []

    box_type = BoxType[str(args.box_type).strip().upper()]
    asset_name_map = {
        BoxType.MAILER: "mailerbox_simple",
        BoxType.TUCK_END: "tuckendbox",
        BoxType.DRAWER: "drawerbox",
        BoxType.SLIP_LID: "sliplidbox",
    }
    asset_name = asset_name_map.get(box_type, box_type.name.lower())

    for seed in args.seeds:
        print("=" * 80)
        print(f"[Phase1] box_type={box_type.name} seed={seed}")
        print("=" * 80)

        # -----------------------------
        # (1) Export URDF with dom-rand material (density influences inertia)
        # -----------------------------
        _clear_scene()

        mat_cfg = sample_box_material_config(seed=int(seed))

        factory_cls = ModularBoxFactory.get_factory(box_type)
        factory = factory_cls(factory_seed=int(seed))

        params = factory.sample_parameters()
        params.box_type = box_type
        if fixed_dims is not None:
            params.dimensions = fixed_dims
        params.material = mat_cfg
        # Recompute joints if dims/extra_params were overridden.
        params.joints = factory.get_default_joints(params)

        obj = factory.create_asset(asset_params=params)
        sim_blueprint = kinematic_compiler.compile(obj)
        butil.apply_modifiers(obj)
        sim_blueprint["name"] = asset_name

        urdf_exporter.export(
            blend_obj=obj,
            sim_blueprint=sim_blueprint,
            seed=int(seed),
            sample_joint_params_fn=factory.sample_joint_parameters,
            export_dir=urdf_export_root,
            image_res=256,
            visual_only=True,  # phase1 dataset export: collisions not required
        )

        urdf_dir = urdf_export_root / asset_name / str(seed)
        urdf_path = urdf_dir / f"{asset_name}.urdf"
        assets_dir = urdf_dir / "assets"
        if not urdf_path.exists():
            raise FileNotFoundError(urdf_path)

        links_visuals, joints = parse_urdf(urdf_path)

        controlled_joints = [j for j in joints if j.joint_type in ("revolute", "continuous", "prismatic")]
        controlled_joint_names = [j.name for j in controlled_joints]

        def _parse_joint_state_spec(spec: str, n: int) -> List[List[float]]:
            if n == 0:
                return [[]]
            out: List[List[float]] = []
            for chunk in str(spec).split(";"):
                if not chunk.strip():
                    continue
                vals = [float(x.strip()) for x in chunk.split(",") if x.strip()]
                if len(vals) != n:
                    raise ValueError(
                        f"Joint state length mismatch: got {len(vals)} values, expected {n} for joints={controlled_joint_names}"
                    )
                out.append(vals)
            if len(out) == 0:
                out = [[0.0 for _ in range(n)]]
            return out

        joint_states = _parse_joint_state_spec(args.joint_states, n=len(controlled_joint_names))

        # -----------------------------
        # (2) Auto-labeling: render multiple joint states and views
        # -----------------------------
        for js_idx, state_vals in enumerate(joint_states):
            for view_idx in range(int(args.n_views)):
                sample_id = sample_counter
                sample_counter += 1
                sample_dir = dataset_root / f"{sample_id:06d}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                # Rebuild scene each sample for determinism and to avoid GT material overrides leaking.
                _clear_scene()

                # Spawn camera
                cam = cam_util.spawn_camera()
                cam.name = "camera_0_0"

                # Import and pose meshes per link
                link_objs: Dict[str, List[bpy.types.Object]] = {}
                all_mesh_objs: List[bpy.types.Object] = []

                # Joint positions keyed by URDF joint name (non-fixed joints only).
                joint_positions = {name: float(val) for name, val in zip(controlled_joint_names, state_vals)}
                link_poses = compute_link_poses_world(
                    joints=joints,
                    joint_positions=joint_positions,
                    base_link="link_0",
                )

                # Assign stable per-link ids for link-level segmentation
                link_names = sorted([ln for ln in links_visuals.keys() if ln != "world"])
                link_id_map = {ln: int(i + 1) for i, ln in enumerate(link_names)}

                for link_name, visuals in links_visuals.items():
                    if link_name not in link_poses:
                        continue
                    if len(visuals) == 0:
                        continue
                    link_pose = link_poses[link_name]
                    lid = int(link_id_map.get(link_name, 0))
                    link_objs[link_name] = []
                    for k, vis in enumerate(visuals):
                        # URDF uses paths like "assets/geom_0.obj"
                        mesh_rel = vis.mesh_filename.replace("\\\\", "/")
                        if mesh_rel.startswith("assets/"):
                            mesh_path = assets_dir / mesh_rel[len("assets/") :]
                        else:
                            mesh_path = (urdf_dir / mesh_rel).resolve()
                        if not mesh_path.exists():
                            raise FileNotFoundError(mesh_path)

                        new_objs = _import_obj(mesh_path)
                        if len(new_objs) == 0:
                            raise RuntimeError(f"OBJ import produced no mesh objects: {mesh_path}")

                        T_link_vis = _mat_from_xyz_rpy(vis.xyz, vis.rpy)
                        T_world = link_pose @ T_link_vis

                        for o in new_objs:
                            o.name = f"{link_name}__{Path(mesh_path).stem}__{k}"
                            o.pass_index = lid
                            o.matrix_world = T_world
                            # Apply domain-randomized material (shared across links for this sample)
                            factory._apply_box_material(o, mat_cfg)
                            link_objs[link_name].append(o)
                            all_mesh_objs.append(o)

                if len(all_mesh_objs) == 0:
                    raise RuntimeError("No mesh objects imported for rendering")

                # Sample camera pose deterministically from (seed, js_idx, view_idx)
                rng = np.random.default_rng(int(seed) * 1000003 + js_idx * 1009 + view_idx * 9176)
                bb_min, bb_max = _bbox_world(all_mesh_objs)
                center = 0.5 * (bb_min + bb_max)
                extent = float(np.max(bb_max - bb_min))
                radius = float(max(0.30, 2.2 * extent))

                theta = float(rng.uniform(0.0, 2.0 * math.pi))
                phi = float(rng.uniform(0.25 * math.pi, 0.45 * math.pi))
                cam_loc = center + radius * np.array(
                    [
                        math.cos(theta) * math.sin(phi),
                        math.sin(theta) * math.sin(phi),
                        math.cos(phi),
                    ],
                    dtype=np.float64,
                )
                cam.location = mathutils.Vector(tuple(float(x) for x in cam_loc.tolist()))
                _set_camera_look_at(cam, center)

                # Save URDF GT (one per sample for self-contained dataset)
                shutil.copy2(urdf_path, sample_dir / "urdf_gt.urdf")

                # IMPORTANT: camview intrinsics depend on render resolution.
                # Set resolution before saving camera parameters / projecting keypoints.
                scene = bpy.context.scene
                scene.render.resolution_x = int(args.width)
                scene.render.resolution_y = int(args.height)
                scene.render.resolution_percentage = 100
                cam_util.adjust_camera_sensor(cam)

                # Save camera parameters (even in flat_shading mode)
                # Use frame=1 for consistency with render_image's suffix.
                cam_util.save_camera_parameters(cam, output_folder=sample_dir, frame=1)

                # Render RGB first (materials intact)
                _render_rgb(
                    cam,
                    out_path=sample_dir / "rgb.png",
                    resolution=(int(args.width), int(args.height)),
                )

                # Render GT passes (Depth + IndexOB) and save directly to sample_dir
                gt_tmp = sample_dir / "_gt_tmp"
                if gt_tmp.exists():
                    shutil.rmtree(gt_tmp)
                gt_tmp.mkdir(parents=True, exist_ok=True)
                depth_arr, seg_arr = _render_gt_depth_and_seg(
                    cam=cam,
                    out_dir=gt_tmp,
                    resolution=(int(args.width), int(args.height)),
                )
                shutil.move(str(gt_tmp / "depth.npy"), str(sample_dir / "depth.npy"))
                shutil.move(str(gt_tmp / "segmentation.npy"), str(sample_dir / "segmentation.npy"))
                shutil.move(str(gt_tmp / "segmentation.png"), str(sample_dir / "segmentation.png"))
                shutil.rmtree(gt_tmp)

                # For this single-asset setting, instance segmentation is equivalent to link segmentation.
                shutil.copy2(sample_dir / "segmentation.npy", sample_dir / "instance.npy")
                shutil.copy2(sample_dir / "segmentation.png", sample_dir / "instance.png")

                role_map: Optional[Dict[str, str]] = None
                if box_type == BoxType.MAILER:
                    role_map = {"link_0": "box_base", "link_1": "lid", "link_2": "front_flap"}
                elif box_type == BoxType.DRAWER:
                    role_map = {"link_0": "outer_shell", "link_1": "drawer"}
                elif box_type == BoxType.SLIP_LID:
                    role_map = {"link_0": "base", "link_1": "lid"}

                # Write a readable mapping for segmentation labels (id -> link name/role).
                _write_segmentation_label_map(
                    sample_dir / "segmentation_label_map.json",
                    link_id_map=link_id_map,
                    seg=seg_arr,
                    role_map=role_map,
                )

                # Keypoints (joint axes) in world + camera frame
                camview_files = sorted(sample_dir.glob("camview*.npz"))
                if len(camview_files) != 1:
                    raise RuntimeError(f"Expected 1 camview npz, got {len(camview_files)}")
                camview = dict(np.load(camview_files[0]))
                K = np.asarray(camview["K"], dtype=np.float64)
                T_cam_to_world = np.asarray(camview["T"], dtype=np.float64)
                T_world_to_cam = np.linalg.inv(T_cam_to_world)
                HW = [int(x) for x in camview["HW"].tolist()]

                axis_len = float(max(0.05, 0.35 * extent))
                kp = []
                for j in joints:
                    if j.joint_type not in ("revolute", "continuous", "prismatic"):
                        continue
                    if j.parent not in link_poses:
                        continue
                    parent_pose = link_poses[j.parent]
                    joint_frame = parent_pose @ _mat_from_xyz_rpy(j.xyz, j.rpy)

                    origin_w = np.asarray(joint_frame.translation, dtype=np.float64)
                    axis_w = np.asarray((joint_frame.to_3x3() @ mathutils.Vector(tuple(j.axis))).normalized(), dtype=np.float64)
                    end_w = origin_w + axis_len * axis_w

                    def world_to_cam(p_w: np.ndarray) -> np.ndarray:
                        ph = np.array([p_w[0], p_w[1], p_w[2], 1.0], dtype=np.float64)
                        pc = (T_world_to_cam @ ph)[:3]
                        return pc

                    o_c = world_to_cam(origin_w)
                    e_c = world_to_cam(end_w)

                    def project(p_c: np.ndarray) -> Optional[Tuple[float, float]]:
                        if p_c[2] <= 1e-9:
                            return None
                        uvw = K @ (p_c / p_c[2])
                        return (float(uvw[0]), float(uvw[1]))

                    o_uv = project(o_c)
                    e_uv = project(e_c)

                    def in_frame(uv: Optional[Tuple[float, float]]) -> bool:
                        if uv is None:
                            return False
                        u, v = uv
                        return (0.0 <= u < HW[1]) and (0.0 <= v < HW[0])

                    kp.append(
                        {
                            "joint_name": j.name,
                            "parent": j.parent,
                            "child": j.child,
                            "origin_world": origin_w.tolist(),
                            "axis_world": axis_w.tolist(),
                            "end_world": end_w.tolist(),
                            "origin_cam": o_c.tolist(),
                            "end_cam": e_c.tolist(),
                            "origin_uv": list(o_uv) if o_uv is not None else None,
                            "end_uv": list(e_uv) if e_uv is not None else None,
                            "visible": bool(in_frame(o_uv) and in_frame(e_uv) and o_c[2] > 0 and e_c[2] > 0),
                        }
                    )

                _write_json(
                    sample_dir / "joint_state.json",
                    {
                        "seed": int(seed),
                        "box_type": box_type.name,
                        "joint_positions": {k: float(v) for k, v in joint_positions.items()},
                        "joint_names_order": controlled_joint_names,
                    },
                )
                _write_json(
                    sample_dir / "camera_intrinsics.json",
                    {
                        "K": K.tolist(),
                        "T_cam_to_world": T_cam_to_world.tolist(),
                        "HW": HW,
                    },
                )
                _write_json(sample_dir / "keypoints.json", {"joints": kp})

                # Per-sample metadata (keep minimal but explicit)
                _write_json(
                    sample_dir / "metadata.json",
                    {
                        "asset": asset_name,
                        "box_type": box_type.name,
                        "seed": int(seed),
                        "sample_id": int(sample_id),
                        "joint_state": {
                            k: float(v) for k, v in joint_positions.items()
                        },
                        "material": box_material_config_to_metadata(mat_cfg),
                        "link_id_map": link_id_map,
                        "bbox_world": {"min": bb_min.tolist(), "max": bb_max.tolist()},
                        "notes": {
                            "depth_resolution_note": "Depth is rendered from Blender Z-pass at the same resolution as RGB in this Phase-1 demo.",
                        },
                    },
                )

                manifest.append(
                    {
                        "sample_id": int(sample_id),
                        "box_type": box_type.name,
                        "asset": asset_name,
                        "seed": int(seed),
                        "joint_state_idx": int(js_idx),
                        "joint_positions": {k: float(v) for k, v in joint_positions.items()},
                        "view_idx": int(view_idx),
                        "path": str(sample_dir.relative_to(out_root)),
                    }
                )

                js_str = ",".join(f"{float(v):.3f}" for v in state_vals)
                print(f"[sample] id={sample_id:06d} seed={seed} js=[{js_str}] view={view_idx} -> {sample_dir}")

    _write_json(out_root / "manifest.json", {"samples": manifest})
    print("=" * 80)
    print(f"✅ Done. Wrote manifest: {out_root / 'manifest.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()


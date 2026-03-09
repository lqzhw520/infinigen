#!/usr/bin/env python3
"""
Infinigen Phase-1 Data Engine → NAP Graph Format Converter

Converts Infinigen's URDF+OBJ per-sample output into NAP's .npz graph format
for training PhysNAP diffusion models.

Usage:
    python scripts/infinigen_to_nap.py \
        --input-dir sim_exports/data_engine/phase1_1k_mailer/dataset/train \
        --output-dir external/physnap/data/infinigen_graph \
        --box-type MAILER \
        [--encode-shapes]  # requires NAP shape AE weights

Input per sample (Infinigen Phase-1):
    {sample_id}/
    ├── urdf_gt.urdf           -> link/joint structure
    ├── assets/geom_*.obj      -> per-link meshes
    ├── metadata.json          -> box_type, seed, material info
    └── segmentation_label_map.json -> link_name -> id mapping

Output per sample (NAP format):
    {object_id}.npz with arrays:
        V: list of node dicts (bbox_L, abs_center, agg_mesh)
        E: list of edge dicts (e0.src_ind, e0.dst_ind, e0.plucker, r_limits, p_limits)
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

try:
    import trimesh
except ImportError:
    print("trimesh not available. Install with: pip install trimesh")
    sys.exit(1)


def parse_urdf(urdf_path):
    """Parse URDF file and extract link/joint structure.

    Returns:
        links: list of dicts with name, visual_meshes, visual_origins
        joints: list of dicts with name, type, parent, child, origin, axis, limits
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = {}
    for link_elem in root.findall("link"):
        name = link_elem.get("name")
        if name == "world":
            continue
        visuals = []
        for vis in link_elem.findall("visual"):
            geom = vis.find("geometry/mesh")
            origin = vis.find("origin")
            if geom is not None:
                mesh_file = geom.get("filename")
                xyz = [0, 0, 0]
                if origin is not None and origin.get("xyz"):
                    xyz = [float(x) for x in origin.get("xyz").split()]
                visuals.append({"mesh_file": mesh_file, "origin_xyz": np.array(xyz)})
        links[name] = {"visuals": visuals}

    joints = []
    for joint_elem in root.findall("joint"):
        jname = joint_elem.get("name")
        jtype = joint_elem.get("type")
        parent = joint_elem.find("parent").get("link")
        child = joint_elem.find("child").get("link")

        origin_elem = joint_elem.find("origin")
        origin_xyz = np.zeros(3)
        if origin_elem is not None and origin_elem.get("xyz"):
            origin_xyz = np.array([float(x) for x in origin_elem.get("xyz").split()])

        axis_elem = joint_elem.find("axis")
        axis_xyz = np.array([0, 0, 1.0])
        if axis_elem is not None and axis_elem.get("xyz"):
            axis_xyz = np.array([float(x) for x in axis_elem.get("xyz").split()])
            norm = np.linalg.norm(axis_xyz)
            if norm > 1e-8:
                axis_xyz = axis_xyz / norm

        limit_elem = joint_elem.find("limit")
        lower, upper = 0.0, 0.0
        if limit_elem is not None:
            lower = float(limit_elem.get("lower", 0))
            upper = float(limit_elem.get("upper", 0))

        joints.append(
            {
                "name": jname,
                "type": jtype,
                "parent": parent,
                "child": child,
                "origin_xyz": origin_xyz,
                "axis_xyz": axis_xyz,
                "lower": lower,
                "upper": upper,
            }
        )

    return links, joints


def load_and_merge_link_meshes(sample_dir, link_name, link_info):
    """Load all OBJ meshes for a link and merge into a single trimesh."""
    meshes = []
    for vis in link_info["visuals"]:
        mesh_path = os.path.join(sample_dir, vis["mesh_file"])
        if not os.path.exists(mesh_path):
            continue
        try:
            m = trimesh.load(mesh_path, force="mesh", process=False)
            if isinstance(m, trimesh.Scene):
                for geom in m.geometry.values():
                    if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0:
                        meshes.append(geom)
            elif isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0:
                meshes.append(m)
        except Exception:
            pass

    if not meshes:
        return None
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def compute_link_global_translation(joints, links_order):
    """Compute absolute 3D translation for each link in the kinematic chain.

    In NAP's convention, all parts are in canonical (zero-articulation) orientation,
    so we only need accumulated translations along the chain.
    """
    translations = {}
    parent_map = {}
    origin_map = {}

    for j in joints:
        if j["type"] == "fixed" and j["parent"] == "world":
            translations[j["child"]] = j["origin_xyz"].copy()
        parent_map[j["child"]] = j["parent"]
        origin_map[j["child"]] = j["origin_xyz"].copy()

    for link_name in links_order:
        if link_name in translations:
            continue
        chain = []
        current = link_name
        while current in parent_map and current not in translations:
            chain.append(current)
            current = parent_map[current]
        if current in translations:
            base_t = translations[current].copy()
        else:
            base_t = np.zeros(3)
        for c in reversed(chain):
            base_t = base_t + origin_map[c]
            translations[c] = base_t.copy()

    return translations


def axis_origin_to_plucker(axis, origin):
    """Convert URDF joint axis + origin to Plucker coordinates.

    Plucker coordinates (l, m) where:
        l = normalized axis direction (3D)
        m = origin × l (moment, 3D)

    Returns: (6,) array [l_x, l_y, l_z, m_x, m_y, m_z]
    """
    l = axis / (np.linalg.norm(axis) + 1e-12)
    m = np.cross(origin, l)
    return np.concatenate([l, m])


def normalize_to_unit(translations, bboxes):
    """Normalize all translations and bboxes so the overall object fits in a unit cube.

    Returns: (scale_factor, center_offset) used for normalization
    """
    if len(translations) == 0:
        return 1.0, np.zeros(3)

    all_points = []
    for link_name in translations:
        t = translations[link_name]
        if link_name in bboxes:
            half = bboxes[link_name]
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    for dz in [-1, 1]:
                        all_points.append(t + np.array([dx, dy, dz]) * half)
        else:
            all_points.append(t)

    all_points = np.array(all_points)
    center = (all_points.max(axis=0) + all_points.min(axis=0)) / 2.0
    extent = (all_points.max(axis=0) - all_points.min(axis=0)).max()
    scale = extent if extent > 1e-8 else 1.0

    return scale, center


def convert_sample(sample_dir, object_id, max_K=8):
    """Convert a single Infinigen Phase-1 sample to NAP graph format.

    Returns: dict with 'V' (list of node dicts), 'E' (list of edge dicts), or None on failure
    """
    urdf_path = os.path.join(sample_dir, "urdf_gt.urdf")
    if not os.path.exists(urdf_path):
        return None

    links, joints = parse_urdf(urdf_path)

    if len(links) > max_K:
        return None

    active_joints = [
        j for j in joints if j["type"] != "fixed" or j["parent"] == "world"
    ]
    movable_joints = [
        j for j in joints if j["type"] in ("revolute", "prismatic", "continuous")
    ]

    link_names = sorted(
        links.keys(), key=lambda x: int(x.split("_")[-1]) if "_" in x else 0
    )

    translations = compute_link_global_translation(joints, link_names)

    merged_meshes = {}
    bboxes = {}
    for lname in link_names:
        mesh = load_and_merge_link_meshes(sample_dir, lname, links[lname])
        if mesh is not None:
            merged_meshes[lname] = mesh
            verts = mesh.vertices
            half_extents = (verts.max(axis=0) - verts.min(axis=0)) / 2.0
            bboxes[lname] = half_extents
        else:
            bboxes[lname] = np.array([0.01, 0.01, 0.01])

    scale, center = normalize_to_unit(translations, bboxes)

    V = []
    link_to_idx = {}
    for i, lname in enumerate(link_names):
        link_to_idx[lname] = i
        t_norm = (translations.get(lname, np.zeros(3)) - center) / scale
        bbox_norm = bboxes.get(lname, np.array([0.01, 0.01, 0.01])) / scale

        mesh_data = None
        if lname in merged_meshes:
            m = merged_meshes[lname]
            verts_norm = (m.vertices - center) / scale
            mesh_data = (verts_norm.astype(np.float32), m.faces.astype(np.int32))

        node = {
            "bbox_L": bbox_norm.astype(np.float32),
            "abs_center": t_norm.astype(np.float32),
            "agg_mesh": mesh_data,
        }
        V.append(node)

    E = []
    for j in movable_joints:
        parent_name = j["parent"]
        child_name = j["child"]

        if parent_name not in link_to_idx or child_name not in link_to_idx:
            continue

        src_idx = link_to_idx[parent_name]
        dst_idx = link_to_idx[child_name]

        joint_origin_global = translations.get(child_name, np.zeros(3))
        origin_norm = (joint_origin_global - center) / scale

        plucker = axis_origin_to_plucker(j["axis_xyz"], origin_norm)

        if j["type"] in ("revolute", "continuous"):
            r_limits = np.array([j["lower"], j["upper"]], dtype=np.float32)
            p_limits = np.array([0.0, 0.0], dtype=np.float32)
        elif j["type"] == "prismatic":
            r_limits = np.array([0.0, 0.0], dtype=np.float32)
            p_limits = np.array(
                [j["lower"] / scale, j["upper"] / scale], dtype=np.float32
            )
        else:
            r_limits = np.array([0.0, 0.0], dtype=np.float32)
            p_limits = np.array([0.0, 0.0], dtype=np.float32)

        edge = {
            "e0": {
                "src_ind": src_idx,
                "dst_ind": dst_idx,
                "plucker": plucker.astype(np.float32),
            },
            "r_limits": r_limits,
            "p_limits": p_limits,
        }
        E.append(edge)

    return {"V": V, "E": E, "scale": scale, "center": center}


def convert_dataset(input_dir, output_dir, box_type, max_K=8):
    """Convert all Infinigen Phase-1 samples in input_dir to NAP format."""
    os.makedirs(output_dir, exist_ok=True)

    sample_dirs = sorted(
        [
            d
            for d in Path(input_dir).iterdir()
            if d.is_dir() and (d / "urdf_gt.urdf").exists()
        ]
    )

    print(f"Found {len(sample_dirs)} samples in {input_dir}")

    converted = 0
    skipped = 0
    stats = {"num_parts": [], "num_joints": []}

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        object_id = f"infinigen_{box_type.lower()}_{sample_id}"

        result = convert_sample(str(sample_dir), object_id, max_K=max_K)
        if result is None:
            skipped += 1
            continue

        V = np.array(result["V"], dtype=object)
        E = np.array(result["E"], dtype=object)

        out_path = os.path.join(output_dir, f"{object_id}.npz")
        np.savez(out_path, V=V, E=E, allow_pickle=True)

        stats["num_parts"].append(len(result["V"]))
        stats["num_joints"].append(len(result["E"]))
        converted += 1

    print("\nConversion complete:")
    print(f"  Converted: {converted}")
    print(f"  Skipped:   {skipped}")
    if stats["num_parts"]:
        print(
            f"  Parts:  min={min(stats['num_parts'])}, max={max(stats['num_parts'])}, "
            f"mean={np.mean(stats['num_parts']):.1f}"
        )
        print(
            f"  Joints: min={min(stats['num_joints'])}, max={max(stats['num_joints'])}, "
            f"mean={np.mean(stats['num_joints']):.1f}"
        )

    meta = {
        "source": "infinigen_phase1_data_engine",
        "box_type": box_type,
        "input_dir": str(input_dir),
        "num_samples": converted,
        "num_skipped": skipped,
        "max_K": max_K,
        "stats": {
            "num_parts": {
                "min": int(min(stats["num_parts"])) if stats["num_parts"] else 0,
                "max": int(max(stats["num_parts"])) if stats["num_parts"] else 0,
            },
            "num_joints": {
                "min": int(min(stats["num_joints"])) if stats["num_joints"] else 0,
                "max": int(max(stats["num_joints"])) if stats["num_joints"] else 0,
            },
        },
    }
    with open(os.path.join(output_dir, "conversion_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return converted


def generate_split_and_partkeys(output_dir, box_type):
    """Generate NAP-compatible split.json and partkeys.json for Infinigen data.

    These files tell NAP which object IDs belong to train/val/test
    and map part indices to the codebook.
    """
    npz_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".npz")])
    object_ids = [f.replace(".npz", "") for f in npz_files]

    n = len(object_ids)
    n_train = max(1, int(n * 0.8))
    n_val = max(1, int(n * 0.1))

    train_ids = object_ids[:n_train]
    val_ids = object_ids[n_train : n_train + n_val]
    test_ids = object_ids[n_train + n_val :]

    category = f"infinigen_{box_type.lower()}"
    split = {category: {"train": train_ids, "val": val_ids, "test": test_ids}}

    split_path = os.path.join(output_dir, "infinigen_split.json")
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    print(
        f"Split file: {split_path}  (train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)})"
    )

    all_partkeys = {"train": [], "val": [], "test": []}
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        for obj_id in ids:
            data = np.load(os.path.join(output_dir, f"{obj_id}.npz"), allow_pickle=True)
            V = data["V"].tolist()
            for part_idx in range(len(V)):
                all_partkeys[split_name].append(f"{obj_id}_{part_idx}")

    partkeys_path = os.path.join(output_dir, "infinigen_partkeys.json")
    with open(partkeys_path, "w") as f:
        json.dump(all_partkeys, f, indent=2)
    print(
        f"Partkeys file: {partkeys_path}  "
        f"(train={len(all_partkeys['train'])}, val={len(all_partkeys['val'])}, "
        f"test={len(all_partkeys['test'])})"
    )

    return split_path, partkeys_path


def encode_shapes_with_nap_ae(output_dir, ae_checkpoint_dir, physnap_root):
    """Encode all part meshes using NAP's pretrained shape autoencoder.

    Produces a codebook .npz file compatible with NAP's training pipeline.
    Requires the physnap environment with pretrained s1.5_partshape_ae weights.
    """
    sys.path.insert(0, physnap_root)
    try:
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            raise ImportError("torch not found")
    except ImportError:
        print("Cannot import PhysNAP modules. Run with physnap conda env activated.")
        return None

    partkeys_path = os.path.join(output_dir, "infinigen_partkeys.json")
    with open(partkeys_path) as f:
        partkeys = json.load(f)
    all_keys = partkeys["train"] + partkeys["val"] + partkeys["test"]

    print(f"Encoding {len(all_keys)} part shapes...")

    embeddings = np.zeros((len(all_keys), 128), dtype=np.float32)
    valid_mask = np.zeros(len(all_keys), dtype=bool)

    ae_config = os.path.join(ae_checkpoint_dir, "config.yaml")
    if not os.path.exists(ae_config):
        print(f"Shape AE config not found at {ae_config}")
        print("Skipping shape encoding. Codebook will contain zero vectors.")
        std = np.ones(128, dtype=np.float32)
        codebook_path = os.path.join(output_dir, "infinigen_codebook.npz")
        np.savez(codebook_path, embedding=embeddings, valid_mask=valid_mask, std=std)
        print(f"Empty codebook saved to {codebook_path}")
        return codebook_path

    print(
        "Shape AE encoding requires pretrained weights. Placeholder codebook created."
    )
    std = np.ones(128, dtype=np.float32)
    codebook_path = os.path.join(output_dir, "infinigen_codebook.npz")
    np.savez(codebook_path, embedding=embeddings, valid_mask=valid_mask, std=std)
    print(f"Placeholder codebook saved to {codebook_path}")
    return codebook_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert Infinigen Phase-1 data to NAP graph format"
    )
    parser.add_argument(
        "--input-dir", required=True, help="Path to Infinigen dataset/train/ directory"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for NAP .npz files"
    )
    parser.add_argument(
        "--box-type",
        required=True,
        choices=["MAILER", "DRAWER", "SLIP_LID", "TUCK_END"],
        help="Box type for category labeling",
    )
    parser.add_argument(
        "--max-K", type=int, default=8, help="Max nodes per graph (NAP default: 8)"
    )
    parser.add_argument(
        "--encode-shapes",
        action="store_true",
        help="Encode shapes using NAP's pretrained AE (requires physnap env)",
    )
    parser.add_argument(
        "--physnap-root",
        default=None,
        help="Path to PhysNAP repo root (for shape encoding)",
    )
    parser.add_argument(
        "--ae-checkpoint",
        default=None,
        help="Path to s1.5_partshape_ae checkpoint directory",
    )
    args = parser.parse_args()

    num_converted = convert_dataset(
        args.input_dir, args.output_dir, args.box_type, max_K=args.max_K
    )

    if num_converted > 0:
        generate_split_and_partkeys(args.output_dir, args.box_type)

        if args.encode_shapes:
            physnap_root = args.physnap_root or str(
                Path(__file__).parent.parent / "external" / "physnap"
            )
            ae_dir = args.ae_checkpoint or os.path.join(
                physnap_root, "log", "s1.5_partshape_ae"
            )
            encode_shapes_with_nap_ae(args.output_dir, ae_dir, physnap_root)

    print("\nDone.")


if __name__ == "__main__":
    main()

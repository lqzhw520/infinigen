#!/usr/bin/env python3
"""Audit Infinigen -> NAP conversion fidelity and GT physics health."""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHYSNAP_ROOT = PROJECT_ROOT / "external" / "physnap"


def ensure_binary_compat_preload() -> None:
    conda_prefix = Path(os.environ.get("CONDA_PREFIX", "/root/anaconda3/envs/physnap"))
    libstdcpp = conda_prefix / "lib" / "libstdc++.so.6"
    marker = "CODEX_LIBSTDCPP_PRELOADED"
    if not libstdcpp.exists() or os.environ.get(marker) == "1":
        return
    current = os.environ.get("LD_PRELOAD", "")
    parts = [str(libstdcpp)] + ([current] if current else [])
    os.environ["LD_PRELOAD"] = ":".join(dict.fromkeys(parts))
    os.environ[marker] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], os.environ.copy())


ensure_binary_compat_preload()



def ensure_runtime_ld_library_path() -> None:
    conda_prefix = Path(os.environ.get("CONDA_PREFIX", "/root/anaconda3/envs/physnap"))
    torch_lib = (
        conda_prefix
        / "lib"
        / "python3.9"
        / "site-packages"
        / "torch"
        / "lib"
    )
    candidates = [conda_prefix / "lib", torch_lib]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    prefixes = [str(path) for path in candidates if path.exists()]
    merged = ":".join(prefixes + ([existing] if existing else []))
    if merged:
        os.environ["LD_LIBRARY_PATH"] = merged


ensure_runtime_ld_library_path()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_yaml(path: Path):
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def add_physnap_imports() -> None:
    for candidate in [str(PHYSNAP_ROOT), str(PHYSNAP_ROOT / "eval"), str(PROJECT_ROOT)]:
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


def parse_urdf(urdf_path: Path):
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
            if geom is None:
                continue
            xyz = np.zeros(3, dtype=np.float32)
            if origin is not None and origin.get("xyz"):
                xyz = np.array([float(x) for x in origin.get("xyz").split()], dtype=np.float32)
            visuals.append({"mesh_file": geom.get("filename"), "origin_xyz": xyz})
        links[name] = {"visuals": visuals}

    joints = []
    for joint_elem in root.findall("joint"):
        jtype = joint_elem.get("type")
        parent = joint_elem.find("parent").get("link")
        child = joint_elem.find("child").get("link")
        origin_elem = joint_elem.find("origin")
        axis_elem = joint_elem.find("axis")
        limit_elem = joint_elem.find("limit")
        origin_xyz = np.zeros(3, dtype=np.float32)
        axis_xyz = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if origin_elem is not None and origin_elem.get("xyz"):
            origin_xyz = np.array([float(x) for x in origin_elem.get("xyz").split()], dtype=np.float32)
        if axis_elem is not None and axis_elem.get("xyz"):
            axis_xyz = np.array([float(x) for x in axis_elem.get("xyz").split()], dtype=np.float32)
            norm = np.linalg.norm(axis_xyz)
            if norm > 1e-8:
                axis_xyz = axis_xyz / norm
        lower = float(limit_elem.get("lower", 0.0)) if limit_elem is not None else 0.0
        upper = float(limit_elem.get("upper", 0.0)) if limit_elem is not None else 0.0
        joints.append(
            {
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


def compute_link_global_translation(joints, link_names):
    translations = {}
    parent_map = {}
    origin_map = {}
    for joint in joints:
        if joint["type"] == "fixed" and joint["parent"] == "world":
            translations[joint["child"]] = joint["origin_xyz"].copy()
        parent_map[joint["child"]] = joint["parent"]
        origin_map[joint["child"]] = joint["origin_xyz"].copy()
    for link_name in link_names:
        if link_name in translations:
            continue
        chain = []
        current = link_name
        while current in parent_map and current not in translations:
            chain.append(current)
            current = parent_map[current]
        base = translations[current].copy() if current in translations else np.zeros(3, dtype=np.float32)
        for child in reversed(chain):
            base = base + origin_map[child]
            translations[child] = base.copy()
    return translations


def normalize_to_unit(translations, bboxes):
    all_points = []
    for link_name, translation in translations.items():
        half = bboxes[link_name]
        for dx in (-1, 1):
            for dy in (-1, 1):
                for dz in (-1, 1):
                    all_points.append(translation + np.array([dx, dy, dz], dtype=np.float32) * half)
    all_points = np.asarray(all_points, dtype=np.float32)
    center = (all_points.max(axis=0) + all_points.min(axis=0)) / 2.0
    extent = (all_points.max(axis=0) - all_points.min(axis=0)).max()
    scale = float(extent if extent > 1e-8 else 1.0)
    return scale, center


def axis_origin_to_plucker(axis, origin):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return np.concatenate([axis, np.cross(origin, axis)], axis=0)


def load_link_meshes(sample_dir: Path, visuals: list[dict], *, apply_visual_origins: bool):
    meshes = []
    for vis in visuals:
        mesh_path = sample_dir / vis["mesh_file"]
        if not mesh_path.exists():
            continue
        try:
            loaded = trimesh.load(mesh_path, force="mesh", process=False)
        except Exception:
            continue
        candidates = []
        if isinstance(loaded, trimesh.Scene):
            candidates.extend(
                geom.copy()
                for geom in loaded.geometry.values()
                if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0
            )
        elif isinstance(loaded, trimesh.Trimesh) and len(loaded.vertices) > 0:
            candidates.append(loaded.copy())
        for mesh in candidates:
            if apply_visual_origins:
                mesh.apply_translation(vis["origin_xyz"])
            meshes.append(mesh)
    if not meshes:
        return None
    return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)


def convert_sample_variant(sample_dir: Path, *, apply_visual_origins: bool):
    links, joints = parse_urdf(sample_dir / "urdf_gt.urdf")
    link_names = sorted(links.keys(), key=lambda x: int(x.split("_")[-1]) if "_" in x else 0)
    translations = compute_link_global_translation(joints, link_names)
    merged_meshes = {}
    bboxes = {}
    for link_name in link_names:
        mesh = load_link_meshes(sample_dir, links[link_name]["visuals"], apply_visual_origins=apply_visual_origins)
        if mesh is not None:
            merged_meshes[link_name] = mesh
            verts = mesh.vertices
            bboxes[link_name] = ((verts.max(axis=0) - verts.min(axis=0)) / 2.0).astype(np.float32)
        else:
            bboxes[link_name] = np.array([0.01, 0.01, 0.01], dtype=np.float32)
    scale, center = normalize_to_unit(translations, bboxes)
    active_joints = [joint for joint in joints if joint["type"] in ("revolute", "prismatic", "continuous")]
    nodes = []
    link_to_idx = {}
    for idx, link_name in enumerate(link_names):
        link_to_idx[link_name] = idx
        nodes.append(
            {
                "bbox_L": (bboxes[link_name] / scale).astype(np.float32),
                "abs_center": ((translations.get(link_name, np.zeros(3, dtype=np.float32)) - center) / scale).astype(np.float32),
            }
        )
    edges = []
    for joint in active_joints:
        src = link_to_idx[joint["parent"]]
        dst = link_to_idx[joint["child"]]
        origin_norm = ((translations[joint["child"]] - center) / scale).astype(np.float32)
        if joint["type"] == "prismatic":
            r_limits = np.array([0.0, 0.0], dtype=np.float32)
            p_limits = np.array([joint["lower"] / scale, joint["upper"] / scale], dtype=np.float32)
        else:
            r_limits = np.array([joint["lower"], joint["upper"]], dtype=np.float32)
            p_limits = np.array([0.0, 0.0], dtype=np.float32)
        edges.append(
            {
                "src_ind": src,
                "dst_ind": dst,
                "plucker": axis_origin_to_plucker(joint["axis_xyz"], origin_norm).astype(np.float32),
                "r_limits": r_limits,
                "p_limits": p_limits,
            }
        )
    return {
        "links": links,
        "joints": joints,
        "nodes": nodes,
        "edges": edges,
        "scale": scale,
        "center": center,
        "merged_meshes": merged_meshes,
    }


def prepare_dataset_cfg(config_path: Path) -> dict:
    add_physnap_imports()
    from init.config_utils import load_config

    cfg = load_config(str(config_path), default_path=str(PHYSNAP_ROOT / "init" / "default.yaml"))
    cfg["root"] = str(PHYSNAP_ROOT)
    cfg["modes"] = ["train", "val", "test"]
    cfg["dataset"]["dataset_proportion"] = [1.0, 1.0, 1.0]
    for key in ["split_path", "embedding_index_file", "embedding_precompute_path"]:
        value = cfg["dataset"].get(key)
        if value and not os.path.isabs(value):
            cfg["dataset"][key] = str(PHYSNAP_ROOT / value)
    shapeprior = cfg["model"]["part_shape_prior"].get("pretrained_shapeprior_path")
    if shapeprior and not os.path.isabs(shapeprior):
        cfg["model"]["part_shape_prior"]["pretrained_shapeprior_path"] = str(PHYSNAP_ROOT / shapeprior)
    return cfg


def build_gt_eval_model(config_path: Path):
    add_physnap_imports()
    from core.models import arti_ddpm_v2
    from core.models.arti_ddpm_v2 import EvaluatePairwiseSDFs, WrapDec

    cfg = prepare_dataset_cfg(config_path)
    model = arti_ddpm_v2.Model(cfg)
    model.to_gpus()
    model.set_eval()
    net = model.network.module if model.__dataparallel_flag__ else model.network
    net.sdf_decoder = WrapDec(net.network_dict["sdf_decoder"], net.freq, net.N_pe > 0)
    net.eval_pairwise_sdfs = EvaluatePairwiseSDFs(net.sdf_decoder)
    return model, cfg


def build_guide_cfg():
    return {
        "cond_fac": 0.0,
        "pen_fac": 1.0,
        "mob_fac": 1.0,
        "shapelatent_fac": 0.0,
        "mob_nstates_final": 10,
        "mob_zero": False,
    }


def gt_self_eval(config_path: Path, split: str, limit: int) -> dict:
    add_physnap_imports()
    from dataset import get_dataset

    model, cfg = build_gt_eval_model(config_path)
    net = model.network.module if model.__dataparallel_flag__ else model.network
    dataset_cls = get_dataset(cfg)
    dataset = dataset_cls(cfg, split)
    guide_cfg = build_guide_cfg()

    per_pen = []
    per_mob = []
    num_intersect = []
    query_points = []
    batch_size = 8
    total = min(len(dataset), limit if limit > 0 else len(dataset))

    for start in range(0, total, batch_size):
        batch_items = [dataset[idx][0] for idx in range(start, min(start + batch_size, total))]
        V = torch.stack([item["V"] for item in batch_items], dim=0).cuda().float()
        E = torch.stack([item["E"] for item in batch_items], dim=0).cuda().float()
        V_scale = torch.stack([item["V_scale"] for item in batch_items], dim=0).cuda().float()[0]
        E_scale = torch.stack([item["E_scale"] for item in batch_items], dim=0).cuda().float()[0]
        V_reduced = torch.cat([V[..., :4], V[..., 7:]], dim=-1)
        existence, bbox, transform, sdf_latent = net.extract_node_attributes(V_reduced, V_scale)
        E_real = E * E_scale[None, None, :]
        plucker = E_real[..., 3:9]
        empty_edges = torch.argmax(E_real[..., :3], dim=-1) == 0
        invalid_plucker = empty_edges | (torch.linalg.norm(plucker[..., :3], dim=-1) < 1e-6)
        if invalid_plucker.any():
            plucker = plucker.clone()
            plucker[invalid_plucker] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=plucker.device, dtype=plucker.dtype)
            E_real = E_real.clone()
            E_real[..., 3:9] = plucker
        with torch.no_grad():
            _, stats = net.guidance_losses(
                existence,
                bbox,
                transform,
                sdf_latent,
                None,
                E_real,
                guide_cfg=guide_cfg,
                final=True,
            )
        per_pen.extend(np.asarray(stats["pen_error_per"]).tolist())
        per_mob.extend(np.asarray(stats["mob_error_per"]).tolist())
        num_intersect.append(float(stats["num_intersect_mean"]))
        query_points.append(float(stats["query_points_mean"]))

    return {
        "split": split,
        "num_samples": total,
        "pen_error_mean": float(np.mean(per_pen)) if per_pen else None,
        "mob_error_mean": float(np.mean(per_mob)) if per_mob else None,
        "pen_error_p95": float(np.percentile(per_pen, 95)) if per_pen else None,
        "mob_error_p95": float(np.percentile(per_mob, 95)) if per_mob else None,
        "num_intersect_mean": float(np.mean(num_intersect)) if num_intersect else None,
        "query_points_mean": float(np.mean(query_points)) if query_points else None,
    }


@dataclass
class CategorySource:
    box_type: str
    input_dir: Path


def category_sources() -> dict[str, CategorySource]:
    sources = {}
    for box_type in ["mailer", "drawer", "slip_lid", "tuck_end"]:
        meta_path = PHYSNAP_ROOT / "data" / f"infinigen_graph_{box_type.replace('_', '') if box_type == 'slip_lid' else box_type}" / "conversion_meta.json"
        if not meta_path.exists():
            # fall back to real directory names
            candidates = sorted((PHYSNAP_ROOT / "data").glob(f"infinigen_graph*{box_type.replace('_', '')}*"))
            if candidates:
                meta_path = candidates[0] / "conversion_meta.json"
        meta = load_json(meta_path)
        sources[box_type] = CategorySource(
            box_type=box_type,
            input_dir=resolve_path(meta["input_dir"]),
        )
    return sources


def audit_saved_sample(category: str, sample_id: str, combined_root: Path):
    object_id = f"infinigen_{category}_{sample_id}"
    saved = np.load(combined_root / f"{object_id}.npz", allow_pickle=True)
    saved_nodes = saved["V"].tolist()
    saved_edges = saved["E"].tolist()

    source_dir = category_sources()[category].input_dir / sample_id
    variant_buggy = convert_sample_variant(source_dir, apply_visual_origins=False)
    variant_fixed = convert_sample_variant(source_dir, apply_visual_origins=True)

    axis_cos_errors = []
    limit_errors = []
    saved_center_vs_buggy = []
    saved_center_vs_fixed = []
    saved_bbox_vs_buggy = []
    saved_bbox_vs_fixed = []
    bbox_origin_deltas = []
    links_with_origin = 0
    for link_name, link_info in variant_fixed["links"].items():
        has_origin = any(np.linalg.norm(vis["origin_xyz"]) > 1e-8 for vis in link_info["visuals"])
        if has_origin:
            links_with_origin += 1
        idx = int(link_name.split("_")[-1]) if "_" in link_name else 0
        if idx < len(saved_nodes):
            bbox_buggy = np.asarray(variant_buggy["nodes"][idx]["bbox_L"])
            bbox_fixed = np.asarray(variant_fixed["nodes"][idx]["bbox_L"])
            bbox_saved = np.asarray(saved_nodes[idx]["bbox_L"])
            center_buggy = np.asarray(variant_buggy["nodes"][idx]["abs_center"])
            center_fixed = np.asarray(variant_fixed["nodes"][idx]["abs_center"])
            center_saved = np.asarray(saved_nodes[idx]["abs_center"])
            bbox_origin_deltas.append(float(np.max(np.abs(bbox_fixed - bbox_buggy))))
            saved_bbox_vs_buggy.append(float(np.max(np.abs(bbox_saved - bbox_buggy))))
            saved_bbox_vs_fixed.append(float(np.max(np.abs(bbox_saved - bbox_fixed))))
            saved_center_vs_buggy.append(float(np.max(np.abs(center_saved - center_buggy))))
            saved_center_vs_fixed.append(float(np.max(np.abs(center_saved - center_fixed))))

    for edge_idx, edge in enumerate(saved_edges):
        if edge_idx >= len(variant_buggy["edges"]):
            continue
        expected = variant_buggy["edges"][edge_idx]
        saved_plucker = np.asarray(edge["e0"]["plucker"])
        exp_plucker = np.asarray(expected["plucker"])
        axis_cos = float(np.clip(np.abs(np.dot(saved_plucker[:3], exp_plucker[:3])), 0.0, 1.0))
        axis_cos_errors.append(1.0 - axis_cos)
        limit_errors.append(
            float(
                max(
                    np.max(np.abs(np.asarray(edge["r_limits"]) - expected["r_limits"])),
                    np.max(np.abs(np.asarray(edge["p_limits"]) - expected["p_limits"])),
                )
            )
        )

    alignment_fixed = max(saved_bbox_vs_fixed or [0.0]) + max(saved_center_vs_fixed or [0.0])
    alignment_buggy = max(saved_bbox_vs_buggy or [0.0]) + max(saved_center_vs_buggy or [0.0])
    saved_alignment = "fixed" if alignment_fixed <= alignment_buggy else "buggy_or_stale"

    return {
        "category": category,
        "sample_id": sample_id,
        "saved_object_id": object_id,
        "part_count_saved": len(saved_nodes),
        "part_count_urdf": len(variant_buggy["nodes"]),
        "joint_count_saved": len(saved_edges),
        "joint_count_urdf": len(variant_buggy["edges"]),
        "links_with_nonzero_visual_origin": links_with_origin,
        "max_axis_cos_error": float(max(axis_cos_errors) if axis_cos_errors else 0.0),
        "max_limit_error": float(max(limit_errors) if limit_errors else 0.0),
        "max_saved_center_vs_buggy_converter_error": float(max(saved_center_vs_buggy) if saved_center_vs_buggy else 0.0),
        "max_saved_center_vs_fixed_converter_error": float(max(saved_center_vs_fixed) if saved_center_vs_fixed else 0.0),
        "max_saved_bbox_vs_buggy_converter_error": float(max(saved_bbox_vs_buggy) if saved_bbox_vs_buggy else 0.0),
        "max_saved_bbox_vs_fixed_converter_error": float(max(saved_bbox_vs_fixed) if saved_bbox_vs_fixed else 0.0),
        "max_bbox_delta_due_to_visual_origin": float(max(bbox_origin_deltas) if bbox_origin_deltas else 0.0),
        "saved_alignment": saved_alignment,
    }

def audit_conversion(manifest: dict, acceptance: dict, campaign_dir: Path, *, gt_limit: int) -> dict:
    refs = manifest["references"]
    box_config = resolve_path(refs["box_val"]["config_path"])
    diagnostics_dir = campaign_dir / "diagnostics"
    partnet_limit = min(gt_limit, 64)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    gt_eval = gt_self_eval(box_config, refs["box_val"]["split"], gt_limit)
    partnet_gt_eval = gt_self_eval(
        resolve_path(refs["partnet_val"]["config_path"]),
        refs["partnet_val"]["split"],
        partnet_limit,
    )

    combined_root = PHYSNAP_ROOT / "data" / "infinigen_graph_combined_k10"
    sample_count = acceptance.get("diagnostics", {}).get("samples_per_category", 2)
    sample_results = []
    for category, source in category_sources().items():
        if not source.input_dir.exists():
            continue
        sample_ids = sorted(path.name for path in source.input_dir.iterdir() if (path / "urdf_gt.urdf").exists())[:sample_count]
        for sample_id in sample_ids:
            sample_results.append(audit_saved_sample(category, sample_id, combined_root))

    codebook = np.load(combined_root / "infinigen_codebook.npz")
    shape_code = {
        "codebook_path": str(combined_root / "infinigen_codebook.npz"),
        "valid_true": int(codebook["valid_mask"].sum()),
        "valid_total": int(codebook["valid_mask"].shape[0]),
        "embedding_abs_sum": float(np.abs(codebook["embedding"]).sum()),
        "generator_script": str(PROJECT_ROOT / "scripts" / "encode_infinigen_shapes.py"),
        "deprecated_placeholder_path": str(PROJECT_ROOT / "scripts" / "infinigen_to_nap.py"),
    }

    diag_cfg = acceptance.get("diagnostics", {})
    blocker_reasons = []
    pen_threshold = diag_cfg.get("gt_self_eval_pen_mean_max", 1.0)
    mob_threshold = diag_cfg.get("gt_self_eval_mob_mean_max", 1.0)

    if gt_eval["pen_error_mean"] is not None and gt_eval["pen_error_mean"] > pen_threshold:
        if partnet_gt_eval["pen_error_mean"] is not None and partnet_gt_eval["pen_error_mean"] <= pen_threshold:
            blocker_reasons.append(
                f"Box GT self-eval penetration mean {gt_eval['pen_error_mean']:.4f} exceeds threshold {pen_threshold:.4f} while PartNet GT remains {partnet_gt_eval['pen_error_mean']:.4f}; conversion or representation fidelity blocker"
            )
        else:
            blocker_reasons.append(
                f"Both box GT ({gt_eval['pen_error_mean']:.4f}) and PartNet GT ({partnet_gt_eval['pen_error_mean']:.4f}) penetration exceed threshold {pen_threshold:.4f}; evaluator or metric validity blocker"
            )
    if gt_eval["mob_error_mean"] is not None and gt_eval["mob_error_mean"] > mob_threshold:
        if partnet_gt_eval["mob_error_mean"] is not None and partnet_gt_eval["mob_error_mean"] <= mob_threshold:
            blocker_reasons.append(
                f"Box GT self-eval mobility mean {gt_eval['mob_error_mean']:.4f} exceeds threshold {mob_threshold:.4f} while PartNet GT remains {partnet_gt_eval['mob_error_mean']:.4f}; conversion or representation fidelity blocker"
            )
        else:
            blocker_reasons.append(
                f"Both box GT ({gt_eval['mob_error_mean']:.4f}) and PartNet GT ({partnet_gt_eval['mob_error_mean']:.4f}) mobility exceed threshold {mob_threshold:.4f}; evaluator or metric validity blocker"
            )

    max_bbox_delta = max((sample["max_bbox_delta_due_to_visual_origin"] for sample in sample_results), default=0.0)
    max_saved_bbox_vs_fixed = max((sample["max_saved_bbox_vs_fixed_converter_error"] for sample in sample_results), default=0.0)
    alignment_counts = {
        "fixed": sum(1 for sample in sample_results if sample.get("saved_alignment") == "fixed"),
        "buggy_or_stale": sum(1 for sample in sample_results if sample.get("saved_alignment") == "buggy_or_stale"),
    }
    saved_bbox_threshold = diag_cfg.get("saved_bbox_vs_fixed_max", diag_cfg.get("visual_origin_bbox_delta_max", 1e-4))
    if max_saved_bbox_vs_fixed > saved_bbox_threshold:
        blocker_reasons.append(
            f"Saved box graphs disagree with the fixed converter by up to {max_saved_bbox_vs_fixed:.6f} normalized bbox units, exceeding threshold {saved_bbox_threshold:.6f}"
        )

    decision = "implementation_blocked" if blocker_reasons else "healthy"
    payload = {
        "generated_at": now_iso(),
        "decision": decision,
        "gt_self_eval": gt_eval,
        "partnet_gt_self_eval": partnet_gt_eval,
        "urdf_graph_consistency": {
            "samples_audited": len(sample_results),
            "sample_results": sample_results,
        },
        "saved_dataset_fidelity": {
            "max_saved_bbox_vs_fixed_converter_error": max_saved_bbox_vs_fixed,
            "max_bbox_delta_due_to_visual_origin": max_bbox_delta,
            "alignment_counts": alignment_counts,
        },
        "shape_code_reproducibility": shape_code,
        "blocker_reasons": blocker_reasons,
    }
    save_json(diagnostics_dir / "conversion_audit.json", payload)
    return payload

def render_markdown(payload: dict) -> str:
    lines = [
        "# Conversion Audit",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Decision: `{payload['decision']}`",
        "",
        "## GT Self-Eval (Box)",
        "",
        f"- Samples: `{payload['gt_self_eval']['num_samples']}`",
        f"- Penetration mean: `{payload['gt_self_eval']['pen_error_mean']}`",
        f"- Mobility mean: `{payload['gt_self_eval']['mob_error_mean']}`",
        f"- Penetration p95: `{payload['gt_self_eval']['pen_error_p95']}`",
        f"- Mobility p95: `{payload['gt_self_eval']['mob_error_p95']}`",
        "",
        "## GT Self-Eval (PartNet Reference)",
        "",
        f"- Samples: `{payload['partnet_gt_self_eval']['num_samples']}`",
        f"- Penetration mean: `{payload['partnet_gt_self_eval']['pen_error_mean']}`",
        f"- Mobility mean: `{payload['partnet_gt_self_eval']['mob_error_mean']}`",
        "",
        "## Conversion Consistency",
        "",
    ]
    for sample in payload["urdf_graph_consistency"]["sample_results"]:
        lines.append(
            "- `{category}/{sample_id}` parts saved/urdf=`{part_count_saved}/{part_count_urdf}` joints saved/urdf=`{joint_count_saved}/{joint_count_urdf}` "
            "axis_err=`{max_axis_cos_error:.6f}` limit_err=`{max_limit_error:.6f}` "
            "saved_bbox_vs_fixed=`{max_saved_bbox_vs_fixed_converter_error:.6f}` visual_origin_effect=`{max_bbox_delta_due_to_visual_origin:.6f}` alignment=`{saved_alignment}`".format(
                **sample
            )
        )
    lines.extend(
        [
            "",
            "## Saved Dataset Fidelity",
            "",
            f"- Alignment counts: `{payload['saved_dataset_fidelity']['alignment_counts']}`",
            f"- Max saved bbox vs fixed converter error: `{payload['saved_dataset_fidelity']['max_saved_bbox_vs_fixed_converter_error']}`",
            f"- Max bbox effect size from visual origin: `{payload['saved_dataset_fidelity']['max_bbox_delta_due_to_visual_origin']}`",
            "",
            "## Shape Code",
            "",
            f"- Codebook: `{payload['shape_code_reproducibility']['codebook_path']}`",
            f"- Valid embeddings: `{payload['shape_code_reproducibility']['valid_true']}/{payload['shape_code_reproducibility']['valid_total']}`",
            f"- Embedding abs sum: `{payload['shape_code_reproducibility']['embedding_abs_sum']}`",
            "",
            "## Blockers",
            "",
        ]
    )
    if payload["blocker_reasons"]:
        for reason in payload["blocker_reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("- No conversion blockers detected.")
    return "\n".join(lines) + "\n"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Infinigen conversion fidelity")
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--gt-limit", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign_dir = Path(args.campaign_dir).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else campaign_dir / "manifest.yaml"
    manifest = load_yaml(manifest_path)
    acceptance = load_json(campaign_dir / "acceptance_criteria.json")
    payload = audit_conversion(manifest, acceptance, campaign_dir, gt_limit=args.gt_limit)
    (campaign_dir / "diagnostics" / "conversion_audit.md").write_text(render_markdown(payload))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

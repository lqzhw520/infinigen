#!/usr/bin/env python3
"""Comprehensive Phase-1 dataset validator with 6-level checks.

Runs ALL validation checks on a Phase-1 dataset directory and produces
a verification_report.json with per-sample pass/fail and batch statistics.

Levels (fail-fast per sample):
  1. File Presence
  2. Array Integrity (depth, seg, instance shapes/ranges)
  3. URDF Structure (inertial, collision, material, limits, DOF)
  4. Physics Validation (PyBullet load, joint limits, inertia)
  5. Cross-Consistency (keypoints vs URDF, label map vs URDF)
  6. Batch Statistics (seed/joint/material distribution)

Usage:
    python scripts/validate_dataset.py sim_exports/data_engine/phase1_1k_mailer/
    python scripts/validate_dataset.py sim_exports/data_engine/phase1_1k_mailer/ --level 3
    python scripts/validate_dataset.py sim_exports/data_engine/phase1_1k_mailer/ --max-samples 50
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_FILES = [
    "rgb.png", "depth.npy", "segmentation.npy", "segmentation.png",
    "segmentation_label_map.json", "instance.npy", "instance.png",
    "keypoints.json", "joint_state.json", "camera_intrinsics.json",
    "urdf_gt.urdf", "metadata.json",
]

EXPECTED_DOF = {
    "MAILER": 2,
    "DRAWER": 1,
    "SLIP_LID": 1,
    "TUCK_END": 8,
}


def _parse_urdf(urdf_path: Path) -> ET.Element:
    return ET.parse(str(urdf_path)).getroot()


def _movable_joints(root: ET.Element) -> list[ET.Element]:
    return [j for j in root.iter("joint") if j.get("type") in ("revolute", "prismatic", "continuous")]


# ---------------------------------------------------------------------------
# Level 1: File Presence
# ---------------------------------------------------------------------------
def check_level1(sample_dir: Path) -> dict[str, Any]:
    missing = [f for f in REQUIRED_FILES if not (sample_dir / f).exists()]
    return {
        "level": 1,
        "name": "file_presence",
        "passed": len(missing) == 0,
        "missing_files": missing,
    }


# ---------------------------------------------------------------------------
# Level 2: Array Integrity
# ---------------------------------------------------------------------------
def check_level2(sample_dir: Path) -> dict[str, Any]:
    errors = []
    depth = np.load(sample_dir / "depth.npy")
    seg = np.load(sample_dir / "segmentation.npy")
    ins = np.load(sample_dir / "instance.npy")

    if depth.shape != seg.shape:
        errors.append(f"depth shape {depth.shape} != seg shape {seg.shape}")
    if depth.shape != ins.shape:
        errors.append(f"depth shape {depth.shape} != instance shape {ins.shape}")

    finite_mask = np.isfinite(depth)
    if not finite_mask.any():
        errors.append("depth has no finite values at all")
    elif np.isnan(depth).any():
        errors.append("depth contains NaN")

    finite_depth = depth[finite_mask] if finite_mask.any() else np.array([0.0])
    reasonable_mask = finite_depth < 100.0
    if reasonable_mask.any():
        d_reasonable = finite_depth[reasonable_mask]
        if d_reasonable.max() > 50.0:
            errors.append(f"depth foreground max {d_reasonable.max():.2f} > 50m")
    bg_ratio = 1.0 - float(reasonable_mask.sum()) / max(finite_depth.size, 1)
    if bg_ratio > 0.99:
        errors.append("depth >99% background -- object may not be visible")

    uniq = np.unique(seg.astype(np.int64))
    if uniq.size <= 1:
        errors.append(f"segmentation has only {uniq.size} unique label(s)")

    from PIL import Image
    rgb = np.array(Image.open(sample_dir / "rgb.png"))
    if rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]:
        errors.append(f"rgb shape {rgb.shape[:2]} != depth shape {depth.shape}")

    return {
        "level": 2,
        "name": "array_integrity",
        "passed": len(errors) == 0,
        "errors": errors,
        "depth_fg_range": [float(np.nanmin(finite_depth[reasonable_mask])) if reasonable_mask.any() else 0.0,
                          float(np.nanmax(finite_depth[reasonable_mask])) if reasonable_mask.any() else 0.0],
        "seg_labels": int(uniq.size),
    }


# ---------------------------------------------------------------------------
# Level 3: URDF Structure
# ---------------------------------------------------------------------------
def check_level3(sample_dir: Path, box_type: str | None = None) -> dict[str, Any]:
    errors = []
    warnings = []
    urdf_path = sample_dir / "urdf_gt.urdf"
    root = _parse_urdf(urdf_path)

    for link in root.iter("link"):
        if link.get("name") == "world":
            continue
        inertials = list(link.iter("inertial"))
        if len(inertials) != 1:
            errors.append(f"link '{link.get('name')}' has {len(inertials)} <inertial> (expected 1)")
        collisions = list(link.iter("collision"))
        if len(collisions) == 0:
            errors.append(f"link '{link.get('name')}' has no <collision> tag")
        materials = list(link.iter("material"))
        if len(materials) == 0:
            warnings.append(f"link '{link.get('name')}' has no <material> tag")

    movable = _movable_joints(root)
    for j in movable:
        lim = j.find("limit")
        if lim is None:
            errors.append(f"joint '{j.get('name')}' type={j.get('type')} has no <limit>")
        else:
            lo = lim.get("lower")
            hi = lim.get("upper")
            if lo is None or hi is None:
                errors.append(f"joint '{j.get('name')}' limit missing lower/upper")
            else:
                if float(lo) >= float(hi):
                    errors.append(f"joint '{j.get('name')}' limit lower={lo} >= upper={hi}")

    if box_type and box_type in EXPECTED_DOF:
        expected = EXPECTED_DOF[box_type]
        actual = len(movable)
        if actual != expected:
            errors.append(f"DOF mismatch: {box_type} expects {expected} movable joints, got {actual}")

    return {
        "level": 3,
        "name": "urdf_structure",
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "n_movable_joints": len(movable),
        "joint_names": [j.get("name") for j in movable],
    }


# ---------------------------------------------------------------------------
# Level 4: Physics Validation
# ---------------------------------------------------------------------------
_pybullet_available = None
_pybullet_module = None


def _get_pybullet():
    global _pybullet_available, _pybullet_module
    if _pybullet_available is None:
        try:
            import pybullet
            _pybullet_module = pybullet
            _pybullet_available = True
        except ImportError:
            _pybullet_available = False
    return _pybullet_module if _pybullet_available else None


def check_level4(sample_dir: Path) -> dict[str, Any]:
    errors = []
    urdf_path = sample_dir / "urdf_gt.urdf"
    root = _parse_urdf(urdf_path)

    for link in root.iter("link"):
        if link.get("name") == "world":
            continue
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_el = inertial.find("mass")
        inertia_el = inertial.find("inertia")
        if mass_el is not None:
            m = float(mass_el.get("value", 0))
            if m <= 0:
                errors.append(f"link '{link.get('name')}' mass={m} <= 0")
        if inertia_el is not None:
            ixx = float(inertia_el.get("ixx", 0))
            iyy = float(inertia_el.get("iyy", 0))
            izz = float(inertia_el.get("izz", 0))
            if ixx <= 0 or iyy <= 0 or izz <= 0:
                errors.append(f"link '{link.get('name')}' inertia diagonal not positive: ({ixx},{iyy},{izz})")
            if not (ixx + iyy >= izz and ixx + izz >= iyy and iyy + izz >= ixx):
                errors.append(f"link '{link.get('name')}' inertia fails triangle inequality")

    p = _get_pybullet()
    if p is None:
        return {"level": 4, "name": "physics_validation", "passed": len(errors) == 0,
                "errors": errors, "pybullet": "unavailable (inertia checks still ran)"}

    cid = None
    try:
        cid = p.connect(p.DIRECT)
        flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
        bid = p.loadURDF(str(urdf_path), useFixedBase=True, flags=flags)

        js_data = json.loads((sample_dir / "joint_state.json").read_text())
        raw_positions = js_data.get("joint_positions", {})
        names_order = js_data.get("joint_names_order", [])

        if isinstance(raw_positions, dict):
            pos_map = {k: float(v) for k, v in raw_positions.items()}
        elif isinstance(raw_positions, list):
            pos_map = {n: float(raw_positions[i]) for i, n in enumerate(names_order) if i < len(raw_positions)}
        else:
            pos_map = {}

        for i in range(p.getNumJoints(bid)):
            info = p.getJointInfo(bid, i)
            jtype = info[2]
            if jtype == 4:  # JOINT_FIXED
                continue
            jname = info[1].decode()
            lower, upper = float(info[8]), float(info[9])
            if lower < upper and jname in pos_map:
                pos = pos_map[jname]
                if pos < lower - 1e-3 or pos > upper + 1e-3:
                    errors.append(
                        f"joint '{jname}' state {pos:.4f} outside limits [{lower:.4f}, {upper:.4f}]"
                    )
    except Exception as e:
        errors.append(f"PyBullet error: {type(e).__name__}: {e}")
    finally:
        if cid is not None:
            try:
                p.disconnect(cid)
            except Exception:
                pass

    return {
        "level": 4,
        "name": "physics_validation",
        "passed": len(errors) == 0,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Level 5: Cross-Consistency
# ---------------------------------------------------------------------------
def check_level5(sample_dir: Path) -> dict[str, Any]:
    errors = []
    warnings = []
    root = _parse_urdf(sample_dir / "urdf_gt.urdf")

    urdf_joint_names = {j.get("name") for j in _movable_joints(root)}
    urdf_link_names = {l.get("name") for l in root.iter("link") if l.get("name") != "world"}

    kp = json.loads((sample_dir / "keypoints.json").read_text())
    kp_joint_names = {j.get("joint_name") for j in kp.get("joints", []) if "joint_name" in j}
    if kp_joint_names and urdf_joint_names:
        missing_in_kp = urdf_joint_names - kp_joint_names
        if missing_in_kp:
            warnings.append(f"URDF joints not in keypoints: {missing_in_kp}")

    for j in kp.get("joints", []):
        axis = j.get("axis_world")
        if axis is not None:
            norm = np.linalg.norm(axis)
            if abs(norm - 1.0) > 0.05:
                warnings.append(f"keypoint '{j.get('joint_name')}' axis not unit: norm={norm:.4f}")

    lm = json.loads((sample_dir / "segmentation_label_map.json").read_text())
    lm_link_names = set()
    for rec in lm.get("labels", []):
        if isinstance(rec, dict) and rec.get("name") and rec["name"] != "background":
            lm_link_names.add(rec["name"])
    link_name_to_id = lm.get("link_name_to_id", {})
    if link_name_to_id:
        lm_link_names.update(link_name_to_id.keys())

    if lm_link_names and urdf_link_names:
        missing_in_lm = urdf_link_names - lm_link_names
        extra_in_lm = lm_link_names - urdf_link_names
        if missing_in_lm:
            warnings.append(f"URDF links not in label_map: {missing_in_lm}")
        if extra_in_lm:
            warnings.append(f"label_map links not in URDF: {extra_in_lm}")

    js = json.loads((sample_dir / "joint_state.json").read_text())
    js_names = set(js.get("joint_names_order", []))
    if js_names and urdf_joint_names:
        if js_names != urdf_joint_names:
            mismatch_extra = js_names - urdf_joint_names
            mismatch_missing = urdf_joint_names - js_names
            if mismatch_extra or mismatch_missing:
                errors.append(f"joint_state names mismatch URDF: extra={mismatch_extra} missing={mismatch_missing}")

    return {
        "level": 5,
        "name": "cross_consistency",
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Level 6: Batch Statistics (runs on all sample results)
# ---------------------------------------------------------------------------
def compute_batch_stats(sample_results: list[dict], dataset_dir: Path) -> dict[str, Any]:
    total = len(sample_results)
    passed = sum(1 for r in sample_results if r["overall_passed"])
    failed = total - passed

    seeds = Counter()
    box_types = Counter()
    materials = Counter()
    joint_states_flat = []

    for r in sample_results:
        meta = r.get("metadata", {})
        seeds[meta.get("seed")] += 1
        box_types[meta.get("box_type", "unknown")] += 1
        materials[meta.get("material_type", "unknown")] += 1
        js = r.get("joint_positions", [])
        joint_states_flat.extend(js)

    js_arr = np.array(joint_states_flat) if joint_states_flat else np.array([])
    js_diversity = float(np.std(js_arr)) if js_arr.size > 1 else 0.0

    anomalies = []
    if len(seeds) == 1:
        anomalies.append("All samples use the same seed -- no diversity")
    if js_diversity < 0.01 and js_arr.size > 0:
        anomalies.append(f"Joint state std={js_diversity:.4f} -- very low diversity")
    if len(materials) == 1:
        anomalies.append("All samples use the same material type")

    return {
        "level": 6,
        "name": "batch_statistics",
        "total_samples": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{100*passed/total:.1f}%" if total > 0 else "N/A",
        "unique_seeds": len(seeds),
        "seed_distribution": dict(seeds.most_common(10)),
        "box_type_distribution": dict(box_types),
        "material_distribution": dict(materials),
        "joint_state_diversity_std": round(js_diversity, 4),
        "anomalies": anomalies,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def validate_sample(sample_dir: Path, max_level: int = 6) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample": sample_dir.name,
        "overall_passed": True,
        "levels": [],
        "metadata": {},
        "joint_positions": [],
    }

    meta_path = sample_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        result["metadata"] = {
            "box_type": meta.get("box_type"),
            "seed": meta.get("seed"),
            "material_type": meta.get("material", {}).get("material_type"),
        }

    js_path = sample_dir / "joint_state.json"
    if js_path.exists():
        js = json.loads(js_path.read_text())
        raw_pos = js.get("joint_positions", {})
        if isinstance(raw_pos, dict):
            result["joint_positions"] = [float(v) for v in raw_pos.values()]
        elif isinstance(raw_pos, list):
            result["joint_positions"] = [float(x) for x in raw_pos if isinstance(x, (int, float))]

    box_type = result["metadata"].get("box_type")

    checks = [
        (1, lambda: check_level1(sample_dir)),
        (2, lambda: check_level2(sample_dir)),
        (3, lambda: check_level3(sample_dir, box_type)),
        (4, lambda: check_level4(sample_dir)),
        (5, lambda: check_level5(sample_dir)),
    ]

    for lvl, check_fn in checks:
        if lvl > max_level:
            break
        try:
            out = check_fn()
        except Exception as e:
            out = {"level": lvl, "passed": False, "errors": [f"Exception: {e}"]}
        result["levels"].append(out)
        if not out.get("passed", False):
            result["overall_passed"] = False
            break

    return result


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Phase-1 dataset validator")
    parser.add_argument("dataset_dir", type=Path, help="Path to phase1_1k_* directory")
    parser.add_argument("--level", type=int, default=6, help="Max validation level (1-6)")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to check (0=all)")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    train_dir = dataset_dir / "dataset" / "train"
    if not train_dir.exists():
        print(f"ERROR: {train_dir} not found")
        return 1

    sample_dirs = sorted(train_dir.iterdir())
    sample_dirs = [d for d in sample_dirs if d.is_dir() and d.name.isdigit()]
    if args.max_samples > 0:
        sample_dirs = sample_dirs[:args.max_samples]

    if not args.quiet:
        print(f"Validating {len(sample_dirs)} samples in {dataset_dir}")
        print(f"Max level: {args.level}")

    all_results = []
    for i, sd in enumerate(sample_dirs):
        r = validate_sample(sd, max_level=args.level)
        all_results.append(r)
        status = "PASS" if r["overall_passed"] else "FAIL"
        if not args.quiet:
            if not r["overall_passed"]:
                failed_lvl = r["levels"][-1] if r["levels"] else {}
                errs = failed_lvl.get("errors", [])
                print(f"  [{i+1:4d}/{len(sample_dirs)}] {sd.name}: {status} -- {errs[:2]}")
            elif (i + 1) % 50 == 0:
                print(f"  [{i+1:4d}/{len(sample_dirs)}] ... running ...")

    batch_stats = compute_batch_stats(all_results, dataset_dir)

    report = {
        "dataset": str(dataset_dir),
        "total_samples": len(all_results),
        "max_level": args.level,
        "batch_statistics": batch_stats,
        "failed_samples": [
            {
                "sample": r["sample"],
                "failed_at_level": r["levels"][-1]["level"] if r["levels"] else 0,
                "errors": r["levels"][-1].get("errors", []) if r["levels"] else [],
            }
            for r in all_results if not r["overall_passed"]
        ],
    }

    report_path = dataset_dir / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    n_pass = batch_stats["passed"]
    n_fail = batch_stats["failed"]
    total = batch_stats["total_samples"]
    print(f"\n{'='*60}")
    print(f"RESULT: {n_pass}/{total} PASS, {n_fail} FAIL")
    print(f"Report: {report_path}")
    if batch_stats["anomalies"]:
        print("Anomalies:")
        for a in batch_stats["anomalies"]:
            print(f"  - {a}")
    print(f"{'='*60}")

    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Phase-1 Data Engine sample folder verifier.

This script validates that a single exported sample folder contains the expected
artifacts and that they are internally consistent (shapes, basic ranges, schema).

Usage:
  cd /mnt/afs2/zhuhaowu/infinigen
  python scripts/verify_phase1_sample_folder.py sim_exports/data_engine/_smoke_test15_labelmap_complete_mailer/dataset/train/000001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REQUIRED_FILES = [
    "rgb.png",
    "depth.npy",
    "segmentation.npy",
    "segmentation.png",
    "segmentation_label_map.json",
    "instance.npy",
    "instance.png",
    "keypoints.json",
    "joint_state.json",
    "camera_intrinsics.json",
    "urdf_gt.urdf",
    "metadata.json",
]


def _as_int_set(arr: np.ndarray) -> set[int]:
    return {int(x) for x in np.unique(arr.astype(np.int64)).tolist()}


def verify_phase1_sample_folder(sample_dir: Path) -> Dict[str, Any]:
    results: Dict[str, Any] = {"passed": True, "errors": [], "warnings": [], "checks": {}}

    # 1) file presence
    missing = [f for f in REQUIRED_FILES if not (sample_dir / f).exists()]
    results["checks"]["missing_files"] = missing
    if missing:
        results["passed"] = False
        results["errors"].append(f"Missing required files: {missing}")
        return results

    # 2) load arrays
    depth = np.load(sample_dir / "depth.npy")
    seg = np.load(sample_dir / "segmentation.npy")
    ins = np.load(sample_dir / "instance.npy")

    results["checks"]["shapes"] = {
        "depth": list(depth.shape),
        "seg": list(seg.shape),
        "instance": list(ins.shape),
    }
    if depth.shape != seg.shape or depth.shape != ins.shape:
        results["passed"] = False
        results["errors"].append("depth/seg/instance shape mismatch")

    results["checks"]["depth_finite"] = bool(np.isfinite(depth).all())
    if not results["checks"]["depth_finite"]:
        results["passed"] = False
        results["errors"].append("depth contains NaN/Inf")

    uniq = np.unique(seg.astype(np.int64))
    results["checks"]["seg_unique_labels"] = {
        "count": int(uniq.size),
        "min": int(uniq.min()) if uniq.size else None,
        "max": int(uniq.max()) if uniq.size else None,
    }
    if uniq.size <= 1:
        results["passed"] = False
        results["errors"].append("segmentation has <=1 unique label")

    # 3) camera intrinsics
    cam = json.loads((sample_dir / "camera_intrinsics.json").read_text())
    HW = cam.get("HW")
    K = np.asarray(cam.get("K"), dtype=float) if cam.get("K") is not None else None
    results["checks"]["camera_HW"] = HW
    results["checks"]["camera_K_shape"] = list(K.shape) if isinstance(K, np.ndarray) else None

    if not (isinstance(HW, list) and len(HW) == 2 and isinstance(K, np.ndarray) and K.shape == (3, 3)):
        results["passed"] = False
        results["errors"].append("camera_intrinsics.json missing HW/K or invalid shapes")
    else:
        H, W = int(HW[0]), int(HW[1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        if not (0.0 <= cx <= W and 0.0 <= cy <= H):
            results["warnings"].append(f"Principal point out of bounds: {(cx, cy)} for {(W, H)}")

    # 4) keypoints visibility sanity
    kp = json.loads((sample_dir / "keypoints.json").read_text())
    joints = kp.get("joints", [])
    in_frame_violations = 0
    if isinstance(HW, list) and len(HW) == 2:
        H, W = int(HW[0]), int(HW[1])
        for j in joints:
            if not j.get("visible", False):
                continue
            for k in ("origin_uv", "end_uv"):
                uv = j.get(k)
                if uv is None:
                    in_frame_violations += 1
                    continue
                u, v = float(uv[0]), float(uv[1])
                if not (0.0 <= u < W and 0.0 <= v < H):
                    in_frame_violations += 1
    results["checks"]["keypoints_in_frame_violations"] = int(in_frame_violations)
    if in_frame_violations > 0:
        results["warnings"].append(f"{in_frame_violations} visible keypoints out of frame")

    # 5) metadata contains dom-rand material physics
    meta = json.loads((sample_dir / "metadata.json").read_text())
    mat = meta.get("material", {})
    for k in ("density", "friction", "restitution", "material_type"):
        if k not in mat:
            results["passed"] = False
            results["errors"].append(f"metadata.material missing field: {k}")
    results["checks"]["material_type"] = mat.get("material_type")

    # 6) segmentation label map sanity
    try:
        lm = json.loads((sample_dir / "segmentation_label_map.json").read_text())
        labels = lm.get("labels", [])
        id_set = {int(x.get("id")) for x in labels if isinstance(x, dict) and "id" in x}
        results["checks"]["label_map_ids"] = sorted(id_set)
        if 0 not in id_set:
            results["warnings"].append("label map missing background id=0")

        link_name_to_id = lm.get("link_name_to_id", {})
        expected_ids = {0}
        if isinstance(link_name_to_id, dict):
            for v in link_name_to_id.values():
                try:
                    expected_ids.add(int(v))
                except Exception:
                    pass

        missing_expected = sorted(expected_ids - id_set)
        results["checks"]["label_map_missing_expected_ids"] = missing_expected
        if missing_expected:
            results["warnings"].append(f"label map missing expected ids: {missing_expected}")

        # Basic record schema checks
        bad_records: List[str] = []
        for rec in labels:
            if not isinstance(rec, dict):
                bad_records.append("<non-dict>")
                continue
            for k in ("id", "name", "present_in_frame", "color_rgb_uint8"):
                if k not in rec:
                    bad_records.append(str(rec.get("id", "<missing-id>")))
                    break
            col = rec.get("color_rgb_uint8")
            if not (
                isinstance(col, list)
                and len(col) == 3
                and all(isinstance(c, int) for c in col)
                and all(0 <= c <= 255 for c in col)
            ):
                bad_records.append(str(rec.get("id", "<missing-id>")))
        results["checks"]["label_map_bad_records"] = bad_records
        if bad_records:
            results["warnings"].append(f"label map has malformed records: {bad_records[:10]}")

        # Consistency with segmentation.npy
        present_ids = _as_int_set(seg)
        present_mismatches = []
        for rec in labels:
            if not isinstance(rec, dict) or "id" not in rec or "present_in_frame" not in rec:
                continue
            lid = int(rec["id"])
            expected_present = lid in present_ids
            if bool(rec["present_in_frame"]) != bool(expected_present):
                present_mismatches.append(lid)
        results["checks"]["label_map_present_mismatches"] = sorted(set(present_mismatches))
        if present_mismatches:
            results["warnings"].append(f"label_map.present_in_frame mismatches for ids: {sorted(set(present_mismatches))}")
    except Exception as e:
        results["passed"] = False
        results["errors"].append(f"failed to parse segmentation_label_map.json: {e}")

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir", type=Path)
    args = ap.parse_args()
    out = verify_phase1_sample_folder(args.sample_dir)
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if out["passed"] else 2)


if __name__ == "__main__":
    main()


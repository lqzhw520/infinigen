#!/usr/bin/env python3
"""Evaluate CAPNet-style metrics on articulated part predictions.

Implements the evaluation protocol from CAPNet (CVPR'25, arXiv:2504.11230):
  - Re: rotation error (degrees) via geodesic distance
  - Te: translation error (cm)
  - Se: scale/size error (ratio)
  - A5:  accuracy at 5deg, 5cm threshold
  - A10: accuracy at 10deg, 10cm threshold
  - mIoU: mean 3D IoU of axis-aligned bounding boxes

Also supports Umeyama alignment for comparing predicted vs GT point sets.

Usage:
    python scripts/eval_capnet_metrics.py pred.json gt.json
    python scripts/eval_capnet_metrics.py pred.json gt.json --per-part --output metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic rotation error in degrees: arccos((tr(R_pred^T R_gt) - 1) / 2)."""
    R_diff = R_pred.T @ R_gt
    trace_val = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace_val)))


def translation_error_cm(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """L2 translation error in centimeters."""
    return float(np.linalg.norm(t_pred - t_gt) * 100.0)


def scale_error(s_pred: np.ndarray, s_gt: np.ndarray) -> float:
    """Mean absolute relative scale error: mean(|s_pred - s_gt| / max(s_gt, eps))."""
    eps = 1e-6
    return float(np.mean(np.abs(s_pred - s_gt) / np.maximum(s_gt, eps)))


def iou_3d_aabb(box1_min: np.ndarray, box1_max: np.ndarray,
                box2_min: np.ndarray, box2_max: np.ndarray) -> float:
    """Axis-aligned 3D bounding box IoU."""
    inter_min = np.maximum(box1_min, box2_min)
    inter_max = np.minimum(box1_max, box2_max)
    inter_dims = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(np.prod(inter_dims))

    vol1 = float(np.prod(np.maximum(box1_max - box1_min, 0.0)))
    vol2 = float(np.prod(np.maximum(box2_max - box2_min, 0.0)))
    union_vol = vol1 + vol2 - inter_vol

    return inter_vol / max(union_vol, 1e-12)


def umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Umeyama alignment: find R, t, s such that dst ≈ s * R @ src + t.

    Args:
        src: (N, 3) source points
        dst: (N, 3) target points
    Returns:
        R (3,3), t (3,), s (scalar)
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    sigma_src = np.sum(src_c ** 2) / n
    cov = (dst_c.T @ src_c) / n

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s_val = float(np.trace(np.diag(D) @ S) / sigma_src) if sigma_src > 1e-12 else 1.0
    t = mu_dst - s_val * R @ mu_src

    return R, t, s_val


def evaluate_per_part(pred_parts: dict[str, dict], gt_parts: dict[str, dict]) -> list[dict]:
    """Evaluate CAPNet metrics per part (link)."""
    results = []
    common_parts = set(pred_parts.keys()) & set(gt_parts.keys())

    for pname in sorted(common_parts):
        pp = pred_parts[pname]
        gp = gt_parts[pname]

        R_pred = np.array(pp["rotation_matrix"])
        R_gt = np.array(gp["rotation_matrix"])
        t_pred = np.array(pp["translation"])
        t_gt = np.array(gp["translation"])
        s_pred = np.array(pp["size"])
        s_gt = np.array(gp["size"])

        re = rotation_error_deg(R_pred, R_gt)
        te = translation_error_cm(t_pred, t_gt)
        se = scale_error(s_pred, s_gt)

        aabb_min_pred = np.array(pp.get("aabb_min", t_pred - s_pred / 2))
        aabb_max_pred = np.array(pp.get("aabb_max", t_pred + s_pred / 2))
        aabb_min_gt = np.array(gp.get("aabb_min", t_gt - s_gt / 2))
        aabb_max_gt = np.array(gp.get("aabb_max", t_gt + s_gt / 2))
        iou = iou_3d_aabb(aabb_min_pred, aabb_max_pred, aabb_min_gt, aabb_max_gt)

        results.append({
            "part": pname,
            "Re_deg": round(re, 4),
            "Te_cm": round(te, 4),
            "Se": round(se, 4),
            "IoU_3d": round(iou, 4),
            "A5": int(re < 5.0 and te < 5.0),
            "A10": int(re < 10.0 and te < 10.0),
        })

    return results


def evaluate_dataset(pred_annotations: list[dict], gt_annotations: list[dict]) -> dict[str, Any]:
    """Evaluate metrics across a full dataset."""
    gt_by_id = {str(g["sample_id"]): g for g in gt_annotations}

    all_Re, all_Te, all_Se, all_IoU = [], [], [], []
    all_A5, all_A10 = [], []
    per_sample = []

    for pred in pred_annotations:
        sid = str(pred["sample_id"])
        gt = gt_by_id.get(sid)
        if gt is None:
            continue

        pred_parts = pred.get("link_annotations", {})
        gt_parts = gt.get("link_annotations", {})
        part_results = evaluate_per_part(pred_parts, gt_parts)

        for pr in part_results:
            all_Re.append(pr["Re_deg"])
            all_Te.append(pr["Te_cm"])
            all_Se.append(pr["Se"])
            all_IoU.append(pr["IoU_3d"])
            all_A5.append(pr["A5"])
            all_A10.append(pr["A10"])

        per_sample.append({"sample_id": sid, "parts": part_results})

    n = len(all_Re) if all_Re else 1
    summary = {
        "n_samples": len(per_sample),
        "n_parts_evaluated": len(all_Re),
        "mean_Re_deg": round(float(np.mean(all_Re)), 4) if all_Re else None,
        "mean_Te_cm": round(float(np.mean(all_Te)), 4) if all_Te else None,
        "mean_Se": round(float(np.mean(all_Se)), 4) if all_Se else None,
        "mean_IoU_3d": round(float(np.mean(all_IoU)), 4) if all_IoU else None,
        "A5": round(float(np.sum(all_A5)) / n, 4) if all_A5 else None,
        "A10": round(float(np.sum(all_A10)) / n, 4) if all_A10 else None,
        "median_Re_deg": round(float(np.median(all_Re)), 4) if all_Re else None,
        "median_Te_cm": round(float(np.median(all_Te)), 4) if all_Te else None,
    }

    return {"summary": summary, "per_sample": per_sample}


def self_consistency_check(dataset_dir: Path, max_samples: int = 0) -> dict:
    """Run GT-vs-GT evaluation to verify metrics pipeline correctness.

    When pred == gt, all errors should be zero and accuracy should be 1.0.
    """
    annot_dir = dataset_dir / "capnet_annotations"
    link_path = annot_dir / "link_pos_quat_aabb.json"
    if not link_path.exists():
        return {"error": f"{link_path} not found. Run capnet_data_bridge.py first."}

    gt = json.loads(link_path.read_text())
    if max_samples > 0:
        gt = gt[:max_samples]

    result = evaluate_dataset(gt, gt)
    return result


def main():
    parser = argparse.ArgumentParser(description="CAPNet-style metrics evaluator")
    parser.add_argument("pred", type=Path, help="Predicted annotations JSON (link_pos_quat_aabb.json format)")
    parser.add_argument("gt", type=Path, nargs="?", default=None,
                        help="Ground-truth annotations JSON (if omitted, runs self-consistency check)")
    parser.add_argument("--per-part", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    if args.gt is None:
        dataset_dir = args.pred.resolve()
        print(f"Self-consistency check on {dataset_dir}")
        result = self_consistency_check(dataset_dir, args.max_samples)
    else:
        pred_data = json.loads(args.pred.read_text())
        gt_data = json.loads(args.gt.read_text())
        if args.max_samples > 0:
            pred_data = pred_data[:args.max_samples]
            gt_data = gt_data[:args.max_samples]
        result = evaluate_dataset(pred_data, gt_data)

    summary = result.get("summary", result)
    print("\nCAPNet Metrics Summary:")
    print(f"  Samples: {summary.get('n_samples', 'N/A')}")
    print(f"  Parts:   {summary.get('n_parts_evaluated', 'N/A')}")
    print(f"  Re (deg): {summary.get('mean_Re_deg', 'N/A')} (median: {summary.get('median_Re_deg', 'N/A')})")
    print(f"  Te (cm):  {summary.get('mean_Te_cm', 'N/A')} (median: {summary.get('median_Te_cm', 'N/A')})")
    print(f"  Se:       {summary.get('mean_Se', 'N/A')}")
    print(f"  mIoU 3D:  {summary.get('mean_IoU_3d', 'N/A')}")
    print(f"  A5:       {summary.get('A5', 'N/A')}")
    print(f"  A10:      {summary.get('A10', 'N/A')}")

    if args.per_part and "per_sample" in result:
        for ps in result["per_sample"][:5]:
            print(f"\n  Sample {ps['sample_id']}:")
            for pr in ps["parts"]:
                print(f"    {pr['part']}: Re={pr['Re_deg']}° Te={pr['Te_cm']}cm Se={pr['Se']} IoU={pr['IoU_3d']}")

    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\nFull results saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

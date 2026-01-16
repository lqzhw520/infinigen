#!/usr/bin/env python3
"""
Generate "viewer-safe" URDFs for MailerBox-simple as a debugging aid for online
URDF viewers.

Important:
- This is ONLY for quick visual verification in viewers that do not enforce
  collision constraints. Joint limits can approximate a "stop at boundary",
  but URDF limits are STATIC and cannot depend on other joint angles.
- The original exported URDFs (mailerbox_simple.urdf) are NOT modified.
- Downstream motion planners should still use the original URDF with full
  [-pi, +pi] joint range and rely on <collision> for legality checks.

Usage:
  conda run -n infinigen python scripts/postprocess_mailerbox_simple_viewer_safe_limits.py \
    --root /mnt/afs2/zhuhaowu/infinigen/sim_exports/urdf/mailerbox_simple
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_manifest(root: Path) -> Dict:
    manifest_path = root / "variants_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def _find_joint_indices(pb, body_id: int) -> Dict[str, int]:
    out = {}
    for j in range(pb.getNumJoints(body_id)):
        name = pb.getJointInfo(body_id, j)[1].decode("utf-8")
        out[name] = j
    return out


def _has_penetration(pb, body_id: int, pairs: List[Tuple[int, int]], eps: float) -> bool:
    """Return True if any contact pair penetrates deeper than eps (negative distance)."""
    cps = pb.getContactPoints(body_id, body_id)
    for c in cps:
        la, lb, dist = c[3], c[4], c[8]
        if dist >= -eps:
            continue
        a, b = (la, lb) if la <= lb else (lb, la)
        if (a, b) in pairs:
            return True
    return False


def _collision_free_intervals(mask: List[bool], xs: np.ndarray) -> List[Tuple[float, float]]:
    """mask[i]=True means collision-free at xs[i]. Return contiguous [start,end] intervals."""
    intervals = []
    start: Optional[int] = None
    for i, ok in enumerate(mask + [False]):  # sentinel
        if ok and start is None:
            start = i
        if (not ok) and start is not None:
            end = i - 1
            intervals.append((float(xs[start]), float(xs[end])))
            start = None
    return intervals


def _pick_interval_containing(intervals: List[Tuple[float, float]], target: float) -> Optional[Tuple[float, float]]:
    for a, b in intervals:
        if a <= target <= b:
            return (a, b)
    return None


def _compute_safe_limits_for_seed(root: Path, seed: int, eps: float = 1e-4) -> Dict:
    import pybullet as pb

    urdf_path = root / str(seed) / "mailerbox_simple.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"Missing URDF: {urdf_path}")

    pb.connect(pb.DIRECT)
    try:
        body = pb.loadURDF(str(urdf_path), useFixedBase=True, flags=pb.URDF_USE_SELF_COLLISION)
        jmap = _find_joint_indices(pb, body)
        lid_j = jmap["mailer_lid_0"]
        flap_j = jmap["mailer_front_flap_0"]

        # --- Safe lid upper limit in the closing direction (+) ---
        # Heuristic: keep flap at 0, and find the last angle before penetration between
        # body (link0) and lid/flap (link1/link2).
        angles = np.linspace(0.0, math.pi, 721)  # 0.25deg steps
        free = []
        for a in angles:
            pb.resetJointState(body, lid_j, float(a))
            pb.resetJointState(body, flap_j, 0.0)
            pb.performCollisionDetection()
            free.append(not _has_penetration(pb, body, pairs=[(0, 1), (0, 2)], eps=eps))

        # find first collision, safe=previous sample
        safe_lid = float(angles[-1])
        for i in range(len(angles)):
            if not free[i]:
                safe_lid = float(angles[max(i - 1, 0)])
                break

        # --- Safe front-flap interval when lid is near closed ---
        # Fix lid to safe_lid and scan flap angles; choose interval containing -pi/2 if possible.
        pb.resetJointState(body, lid_j, safe_lid)
        flap_angles = np.linspace(-math.pi, math.pi, 721)  # 0.5deg-ish
        flap_free = []
        for a in flap_angles:
            pb.resetJointState(body, flap_j, float(a))
            pb.performCollisionDetection()
            flap_free.append(not _has_penetration(pb, body, pairs=[(0, 2)], eps=eps))

        intervals = _collision_free_intervals(flap_free, flap_angles)
        picked = _pick_interval_containing(intervals, target=-math.pi / 2)
        if picked is None and intervals:
            # fallback to the longest interval
            picked = max(intervals, key=lambda ab: ab[1] - ab[0])

        safe_flap = picked if picked is not None else (-math.pi, math.pi)

        return {
            "seed": seed,
            "urdf": str(urdf_path),
            "eps": eps,
            "safe_limits": {
                "mailer_lid_0": {"lower": -math.pi, "upper": safe_lid},
                "mailer_front_flap_0": {"lower": float(safe_flap[0]), "upper": float(safe_flap[1])},
            },
        }
    finally:
        pb.disconnect()


def _write_viewer_safe_urdf(original_urdf: Path, out_urdf: Path, limits: Dict[str, Dict[str, float]]):
    tree = ET.parse(original_urdf)
    root = tree.getroot()

    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in limits:
            continue
        lim = joint.find("limit")
        if lim is None:
            lim = ET.SubElement(joint, "limit")
        lim.set("lower", str(limits[name]["lower"]))
        lim.set("upper", str(limits[name]["upper"]))

    out_urdf.write_text(ET.tostring(root, encoding="unicode"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="mailerbox_simple export root directory")
    ap.add_argument("--eps", type=float, default=1e-4, help="penetration threshold (m) treated as collision")
    args = ap.parse_args()

    root = args.root
    manifest = _load_manifest(root)
    seeds = [int(v["seed"]) for v in manifest["variants"]]

    report = {
        "asset": manifest.get("asset_name", "mailerbox_simple"),
        "root": str(root),
        "eps": args.eps,
        "outputs": {
            # A) Clamp lid only (flap keeps full [-pi, pi]) so you can still test flap
            # freely when the lid is open in the viewer.
            "viewer_safe_lid_only": "mailerbox_simple_viewer_safe.urdf",
            # B) Lock lid at its near-closed safe angle, then clamp flap to collision-free
            # interval w.r.t. the body. This isolates the flap-boundary test.
            "viewer_safe_flap_with_lid_locked": "mailerbox_simple_viewer_safe_flap_closed_lid.urdf",
        },
        "per_seed": [],
    }

    for seed in seeds:
        info = _compute_safe_limits_for_seed(root, seed, eps=args.eps)
        report["per_seed"].append(info)

        src = Path(info["urdf"])
        safe_lid_upper = float(info["safe_limits"]["mailer_lid_0"]["upper"])
        safe_flap = info["safe_limits"]["mailer_front_flap_0"]

        # A) viewer_safe: clamp lid only, keep flap full range for intuitive interaction
        dst_a = src.parent / report["outputs"]["viewer_safe_lid_only"]
        _write_viewer_safe_urdf(
            src,
            dst_a,
            limits={
                "mailer_lid_0": {"lower": -math.pi, "upper": safe_lid_upper},
                "mailer_front_flap_0": {"lower": -math.pi, "upper": math.pi},
            },
        )

        # B) lock lid at safe_lid_upper, clamp flap to collision-free interval at that pose
        dst_b = src.parent / report["outputs"]["viewer_safe_flap_with_lid_locked"]
        _write_viewer_safe_urdf(
            src,
            dst_b,
            limits={
                "mailer_lid_0": {"lower": safe_lid_upper, "upper": safe_lid_upper},
                "mailer_front_flap_0": {"lower": float(safe_flap["lower"]), "upper": float(safe_flap["upper"])},
            },
        )

    out_report = root / "viewer_safe_limits_report.json"
    out_report.write_text(json.dumps(report, indent=2))
    print(f"✅ Wrote: {out_report}")
    print("✅ Wrote per-seed viewer-safe URDFs:")
    for k, v in report["outputs"].items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()


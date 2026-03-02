#!/usr/bin/env python3
"""
Minimal motion-planning demo runner (Phase 3.2 integration scaffold).

This uses the baseline planner:
  - Joint-space linear interpolation
  - Optional PyBullet self-collision checking

Usage (Phase-1 sample folder):
  cd /mnt/afs2/zhuhaowu/infinigen
  python scripts/planning/run_motion_planner_demo.py \
    --sample-dir sim_exports/data_engine/_smoke_test15_labelmap_complete_mailer/dataset/train/000001 \
    --out sim_exports/planning_demo_mailer.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from infinigen.perception.topo_box_net.urdf import parse_urdf
from infinigen.planning.motion_planner_api import JointLimit, PlanRequest
from infinigen.planning.simple_jointspace_planner import JointSpaceLinearPlanner


def _load_joint_positions(sample_dir: Path) -> Dict[str, float]:
    js = json.loads((sample_dir / "joint_state.json").read_text())
    return {str(k): float(v) for k, v in js.get("joint_positions", {}).items()}

def _infer_urdf_with_assets(sample_dir: Path) -> Path:
    """
    Prefer a URDF that has a sibling `assets/` folder so PyBullet can load meshes.

    Priority:
      1) repo_root/sim_exports/urdf/<asset>/<seed>/<asset>.urdf  (visual+collision exports)
      2) out_root/urdf/<asset>/<seed>/<asset>.urdf              (Phase-1 exporter URDF cache)
      3) sample_dir/urdf_gt.urdf                                (may be missing meshes)
    """
    fallback = sample_dir / "urdf_gt.urdf"
    try:
        meta = json.loads((sample_dir / "metadata.json").read_text())
        asset = str(meta.get("asset", "")).strip()
        seed = int(meta.get("seed"))
        if asset:
            cand1 = PROJECT_ROOT / "sim_exports" / "urdf" / asset / str(seed) / f"{asset}.urdf"
            if cand1.exists():
                return cand1
            out_root = sample_dir.parents[2]
            cand2 = out_root / "urdf" / asset / str(seed) / f"{asset}.urdf"
            if cand2.exists():
                return cand2
    except Exception:
        pass
    return fallback


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, help="Phase-1 sample folder containing urdf_gt.urdf + joint_state.json")
    ap.add_argument("--urdf", type=Path, help="URDF path (if not using --sample-dir)")
    ap.add_argument("--out", type=Path, help="Write PlanResult JSON here")
    ap.add_argument("--n-steps", type=int, default=60)
    ap.add_argument("--no-collision-check", action="store_true")
    args = ap.parse_args()

    if args.sample_dir is None and args.urdf is None:
        raise SystemExit("Need either --sample-dir or --urdf")

    if args.sample_dir is not None:
        urdf_path = _infer_urdf_with_assets(args.sample_dir)
        start = _load_joint_positions(args.sample_dir)
    else:
        urdf_path = args.urdf
        start = {}

    urdf = parse_urdf(urdf_path)

    # Build joint limits + a simple "open/pull" goal.
    joint_limits: Dict[str, JointLimit] = {}
    goal: Dict[str, float] = dict(start)

    for j in urdf.controlled_joints():
        lo: Optional[float] = j.limit_lower
        hi: Optional[float] = j.limit_upper
        if lo is not None and hi is not None:
            joint_limits[j.name] = JointLimit(lower=float(lo), upper=float(hi))

        q0 = float(start.get(j.name, 0.0))

        if j.joint_type == "prismatic":
            if hi is not None and lo is not None:
                goal[j.name] = float(min(hi, lo + 0.9 * (hi - lo)))
            elif hi is not None:
                goal[j.name] = float(0.9 * hi)
            else:
                goal[j.name] = q0 + 0.05  # 5cm fallback
        else:  # revolute / continuous
            if hi is not None and lo is not None and hi > lo:
                # Prefer a "safe-open" pose inside limits (avoid slamming to the hard upper bound).
                goal[j.name] = float(lo + 0.75 * (hi - lo))  # e.g. [-pi,pi] -> +pi/2
            elif hi is not None:
                goal[j.name] = float(0.75 * hi)
            else:
                goal[j.name] = q0 + 1.57079632679  # pi/2 fallback

    req = PlanRequest(
        urdf_path=str(urdf_path),
        start_positions=start,
        goal_positions=goal,
        joint_limits=joint_limits if joint_limits else None,
        n_steps=int(args.n_steps),
        collision_check=(not args.no_collision_check),
    )

    planner = JointSpaceLinearPlanner()
    res = planner.plan(req)

    payload = {
        "request": {
            "urdf_path": req.urdf_path,
            "n_steps": req.n_steps,
            "collision_check": req.collision_check,
            "self_collision": req.self_collision,
            "use_inertia_from_file": req.use_inertia_from_file,
        },
        "result": {
            "success": bool(res.success),
            "message": str(res.message),
            "metrics": res.metrics,
            "trajectory_len": len(res.trajectory),
        },
    }
    txt = json.dumps(payload, indent=2)
    print(txt)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(txt)
        print(f"✅ wrote: {args.out}")


if __name__ == "__main__":
    main()


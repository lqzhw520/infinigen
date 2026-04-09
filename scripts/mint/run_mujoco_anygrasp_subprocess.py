#!/usr/bin/env python3
"""Run AnyGrasp detection in the Python 3.10 graspnet environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anygrasp_helper import run_anygrasp, stage_detection_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, help="Path to the staged payload .npz")
    parser.add_argument("--output", required=True, help="Path to the output .json")
    parser.add_argument("--max-candidates", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload_path = Path(args.payload)
    output_path = Path(args.output)

    setup = stage_detection_assets()
    if not setup.get("passed"):
        output_path.write_text(
            json.dumps(
                {
                    "passed": False,
                    "error": setup.get("error", "AnyGrasp setup failed"),
                    "setup": setup,
                },
                indent=2,
            )
        )
        return 1

    payload = np.load(payload_path, allow_pickle=False)
    points = payload["pc"].astype(np.float32)
    colors = payload["colors"].astype(np.float32)
    lims = payload["limits"].astype(np.float32)
    handle = payload["handle_center_world"].astype(np.float32)

    candidates = run_anygrasp(points, colors, lims, max_candidates=args.max_candidates)
    top_candidates: list[dict[str, object]] = []
    selected = None
    for idx, candidate in enumerate(candidates, start=1):
        pose = np.asarray(candidate["pose"], dtype=np.float32)
        dist = float(np.linalg.norm(pose[:3, 3] - handle))
        item = {
            "rank": idx,
            "score": float(candidate["score"]),
            "distance_to_handle": dist,
            "pose_world": pose.tolist(),
        }
        top_candidates.append(item)
        if selected is None and dist <= 0.10:
            selected = item
    if selected is None and top_candidates:
        selected = top_candidates[0]

    report = {
        "passed": selected is not None,
        "selected_grasp": selected,
        "top_candidates": top_candidates,
        "handle_center_world": handle.tolist(),
        "drawer_motion_axis": payload["drawer_motion_axis"].astype(np.float32).tolist(),
        "drawer_aabb_world": payload["drawer_aabb_world"].astype(np.float32).tolist(),
        "setup": setup,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""G3b: Stage AnyGrasp detection assets and verify the SDK can load."""

from __future__ import annotations

import json
import time

import numpy as np
from anygrasp_helper import build_anygrasp, stage_detection_assets
from mint_common import ARTIFACT_DIR

ARTIFACT = ARTIFACT_DIR / "g3_anygrasp_ready.json"
RENDER_CACHE = ARTIFACT_DIR / "g3_scene_renders" / "seed_001.npz"


def run() -> bool:
    result = stage_detection_assets()
    if result.get("passed"):
        try:
            detector = build_anygrasp()
            result["load_net"] = "ok"
            if RENDER_CACHE.exists():
                payload = np.load(RENDER_CACHE)
                grasps, _cloud = detector.get_grasp(
                    payload["pc"].astype(np.float32),
                    payload["colors"].astype(np.float32),
                    lims=payload["limits"].astype(np.float32).tolist(),
                    apply_object_mask=True,
                    dense_grasp=False,
                    collision_detection=True,
                )
                result["smoke_inference"] = {
                    "render_cache": str(RENDER_CACHE),
                    "candidate_count": int(len(grasps)),
                }
        except Exception as exc:
            result["passed"] = False
            result["load_net"] = f"error: {exc}"
            result["error"] = str(exc)
    result["gate"] = "g3_anygrasp_ready"
    result["timestamp"] = time.time()
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

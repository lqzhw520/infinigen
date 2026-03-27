#!/usr/bin/env python3
"""A2: Lock frame, observation, and handle-region alignment truth."""

from __future__ import annotations

import json
import time

from mint_common import ARTIFACT_DIR, validate_json_file

ARTIFACT = ARTIFACT_DIR / "a2_frame_transform_audit.json"


def run() -> bool:
    g2 = validate_json_file(ARTIFACT_DIR / "g2_obs_contract.json") or {}
    scene = validate_json_file(ARTIFACT_DIR / "scene_frame_audit.json") or {}
    handle = validate_json_file(ARTIFACT_DIR / "handle_region_audit.json") or {}
    passed = bool(g2.get("passed") and scene.get("passed") and handle.get("passed"))
    result = {
        "gate": "a2_frame_transform_audit",
        "passed": passed,
        "g2_obs_contract": g2,
        "scene_frame_audit": scene,
        "handle_region_audit": handle,
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

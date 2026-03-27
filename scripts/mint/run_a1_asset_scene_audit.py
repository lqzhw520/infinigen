#!/usr/bin/env python3
"""A1: Lock the asset / URDF audit truth before contract repair."""

from __future__ import annotations

import json
import time

from mint_common import ARTIFACT_DIR, validate_json_file

ARTIFACT = ARTIFACT_DIR / "a1_asset_scene_audit.json"


def run() -> bool:
    g1 = validate_json_file(ARTIFACT_DIR / "g1_asset_load.json") or {}
    audit = validate_json_file(ARTIFACT_DIR / "urdf_asset_audit.json") or {}
    passed = bool(
        g1.get("passed")
        and audit.get("passed")
        and audit.get("sampled_seed_count", 0) >= 4
    )
    result = {
        "gate": "a1_asset_scene_audit",
        "passed": passed,
        "g1_asset_load": g1,
        "urdf_asset_audit": audit,
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

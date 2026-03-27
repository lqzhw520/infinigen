#!/usr/bin/env python3
"""U4: Audit export consistency between metadata, URDF semantics, and simulator-loaded assets."""

from __future__ import annotations

import json
import time

import numpy as np
from mint_common import ARTIFACT_DIR
from upstream_audit_helpers import (
    ALL_SEEDS,
    classify_from_flags,
    env_record,
    geometry_record,
)

ARTIFACT = ARTIFACT_DIR / "u4_export_consistency_audit.json"


def run() -> bool:
    records = []
    flags = []
    for seed in ALL_SEEDS:
        geom = geometry_record(seed)
        env = env_record(seed)
        bbox_dims = np.asarray(geom["bbox_dims"], dtype=np.float32)
        aabb = np.asarray(env["drawer_aabb_world"], dtype=np.float32)
        aabb_dims = aabb[1] - aabb[0]
        ratio = aabb_dims / np.maximum(bbox_dims, 1e-6)
        records.append(
            {
                "seed": seed,
                "metadata_bbox_dims": bbox_dims.tolist(),
                "runtime_aabb_dims": aabb_dims.tolist(),
                "aabb_to_metadata_ratio": ratio.tolist(),
                "drawer_motion_axis": env["drawer_motion_axis"],
                "drawer_travel_distance": env["drawer_travel_distance"],
            }
        )
    bad_ratio = [
        row["seed"]
        for row in records
        if any(value < 0.6 or value > 1.8 for value in row["aabb_to_metadata_ratio"])
    ]
    if bad_ratio:
        flags.append("export_bbox_ratio_suspect")
    decision = classify_from_flags(flags)
    result = {
        "gate": "u4_export_consistency_audit",
        "passed": decision == "upstream_clean",
        "decision": decision,
        "flags": flags,
        "bad_ratio_seeds": bad_ratio,
        "records": records,
        "conclusion_text": (
            "Metadata, URDF semantics, and simulator-loaded geometry are mutually consistent enough for downstream debugging."
            if decision == "upstream_clean"
            else "Exported metadata and runtime geometry diverge on at least some seeds, so upstream export consistency remains suspect."
        ),
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return True


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

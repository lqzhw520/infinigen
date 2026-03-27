#!/usr/bin/env python3
"""U3: Audit whether exported drawer seeds provide a stable graspable handle signal."""

from __future__ import annotations

import json
import time
from collections import defaultdict

import numpy as np
from mint_common import ARTIFACT_DIR
from upstream_audit_helpers import (
    ALL_SEEDS,
    classify_from_flags,
    env_record,
    load_metadata,
)

ARTIFACT = ARTIFACT_DIR / "u3_handle_region_audit.json"
C2_DIR = ARTIFACT_DIR / "c2_replay_valid_rollouts"


def run() -> bool:
    env_records = {row["seed"]: row for row in [env_record(seed) for seed in ALL_SEEDS]}
    distances_by_seed = defaultdict(list)
    records = []
    for meta_path in sorted(C2_DIR.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        seed = int(meta["seed"])
        handle_center = np.asarray(
            env_records[seed]["handle_center_world"], dtype=np.float32
        )
        grasp_pos = np.asarray(
            meta.get("grasp_pose", {}).get("pos", [0.0, 0.0, 0.0]), dtype=np.float32
        )
        distance = float(np.linalg.norm(grasp_pos - handle_center))
        distances_by_seed[seed].append(distance)
        records.append(
            {
                "seed": seed,
                "episode_index": int(meta.get("episode_index", -1)),
                "grasp_score": float(meta.get("grasp_score", 0.0)),
                "max_drawer_fraction": float(meta.get("max_drawer_fraction", 0.0)),
                "handle_center_distance_m": distance,
                "part_labels": load_metadata(seed).get("part_labels", []),
            }
        )
    seed_summary = []
    flags = []
    for seed in ALL_SEEDS:
        distances = distances_by_seed.get(seed, [])
        seed_summary.append(
            {
                "seed": seed,
                "n_rollouts": len(distances),
                "mean_handle_distance_m": float(np.mean(distances))
                if distances
                else None,
                "max_handle_distance_m": float(np.max(distances))
                if distances
                else None,
                "part_labels_present": bool(load_metadata(seed).get("part_labels")),
            }
        )
    if all(not item["part_labels_present"] for item in seed_summary):
        flags.append("missing_handle_part_labels_suspect")
    mean_distances = [
        item["mean_handle_distance_m"]
        for item in seed_summary
        if item["mean_handle_distance_m"] is not None
    ]
    if mean_distances and float(np.mean(mean_distances)) > 0.12:
        flags.append("handle_heuristic_offset_suspect")
    decision = classify_from_flags(flags)
    result = {
        "gate": "u3_handle_region_audit",
        "passed": decision == "upstream_clean",
        "decision": decision,
        "flags": flags,
        "seed_summary": seed_summary,
        "records": records,
        "conclusion_text": (
            "Handle-region signal is sufficiently explicit and geometrically consistent."
            if decision == "upstream_clean"
            else "Handle semantics are weakly exported or heuristically inferred, so upstream asset/export metadata remains a real suspect."
        ),
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return True


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

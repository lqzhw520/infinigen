#!/usr/bin/env python3
"""U1: Audit drawerbox geometry ranges across train and held-out seeds."""

from __future__ import annotations

import json
import time

import numpy as np
from mint_common import ARTIFACT_DIR, DEFAULT_HELD_OUT_SEEDS, DEFAULT_TRAIN_SEEDS
from upstream_audit_helpers import (
    ALL_SEEDS,
    classify_from_flags,
    geometry_record,
    robust_outliers,
)

ARTIFACT = ARTIFACT_DIR / "u1_asset_geometry_audit.json"


def run() -> bool:
    records = [geometry_record(seed) for seed in ALL_SEEDS]
    dims = np.asarray([row["bbox_dims"] for row in records], dtype=np.float32)
    volumes = [row["bbox_volume"] for row in records]
    flags = []
    outlier_seed_ids = sorted(
        {records[idx]["seed"] for idx in robust_outliers(volumes)}
    )
    if outlier_seed_ids:
        flags.append("bbox_volume_suspect")
    if any(any(dim <= 0.0 for dim in row["bbox_dims"]) for row in records):
        flags.append("non_positive_bbox_blocked")
    if any(not row["joint_labels"] for row in records):
        flags.append("missing_joint_label_suspect")
    decision = classify_from_flags(flags)
    result = {
        "gate": "u1_asset_geometry_audit",
        "passed": decision == "upstream_clean",
        "decision": decision,
        "sampled_seeds": ALL_SEEDS,
        "train_seeds": DEFAULT_TRAIN_SEEDS,
        "held_out_seeds": DEFAULT_HELD_OUT_SEEDS,
        "bbox_dim_stats": {
            "mean": dims.mean(axis=0).tolist(),
            "min": dims.min(axis=0).tolist(),
            "max": dims.max(axis=0).tolist(),
        },
        "bbox_volume_stats": {
            "mean": float(np.mean(volumes)),
            "min": float(np.min(volumes)),
            "max": float(np.max(volumes)),
        },
        "outlier_seed_ids": outlier_seed_ids,
        "flags": flags,
        "records": records,
        "conclusion_text": (
            "Geometry ranges are consistent enough across drawerbox seeds for downstream work."
            if decision == "upstream_clean"
            else "Geometry audit found seed-level or semantic irregularities that must be considered before blaming only downstream learning."
        ),
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return True


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

#!/usr/bin/env python3
"""G6: Materialize the selected canonical train cell and pack it into LeRobot."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dataset_builder import build_dataset_from_rollouts
from mint_common import ARTIFACT_DIR, DATASET_DIR, DATASET_REPO_ID
from tiny_retrain_mainline import (
    DEFAULT_ROLLOUT_SOURCE_DIR,
    MIN_TRAIN_EPISODES,
    dataset_guard,
    expected_training_targets,
    materialize_canonical_train_rollouts,
)

ARTIFACT = ARTIFACT_DIR / "g6_lerobot_pack.json"
INTEGRITY = ARTIFACT_DIR / "g6_dataset_integrity.json"


def run() -> bool:
    expected = expected_training_targets()
    override_source = os.environ.get("MINT_G6_SOURCE_DIR")
    source_dir = Path(override_source) if override_source else DEFAULT_ROLLOUT_SOURCE_DIR

    materialization_report = None
    rollout_paths = sorted(source_dir.glob("*.npz")) if source_dir.exists() else []
    if not rollout_paths or not override_source:
        materialization_report = materialize_canonical_train_rollouts(
            source_dir,
            expected=expected,
            force_rebuild=True,
        )
        if not materialization_report.get("passed"):
            result = {
                "gate": "g6_lerobot_pack",
                "passed": False,
                "error": materialization_report.get("error", "Canonical rollout materialization failed"),
                "source_dir": str(source_dir),
                "dataset_root": str(DATASET_DIR),
                "repo_id": DATASET_REPO_ID,
                "materialization": materialization_report,
                **expected,
                "timestamp": time.time(),
            }
            ARTIFACT.write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
            return False
        rollout_paths = sorted(source_dir.glob("*.npz")) if source_dir.exists() else []

    if not rollout_paths:
        result = {
            "gate": "g6_lerobot_pack",
            "passed": False,
            "error": "No rollout NPZ files found for packaging",
            "source_dir": str(source_dir),
            "dataset_root": str(DATASET_DIR),
            "repo_id": DATASET_REPO_ID,
            "materialization": materialization_report,
            **expected,
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False

    payload = build_dataset_from_rollouts(rollout_paths, DATASET_DIR, DATASET_REPO_ID)
    integrity = payload["integrity"]
    guard_ok, guard_report = dataset_guard(expected)
    INTEGRITY.write_text(json.dumps({**integrity, "dataset_guard": guard_report, "timestamp": time.time()}, indent=2))

    result = {
        "gate": "g6_lerobot_pack",
        "passed": bool(payload["episode_count"] >= MIN_TRAIN_EPISODES and integrity["passed"] and guard_ok),
        "source_dir": str(source_dir),
        "dataset_root": str(DATASET_DIR),
        "repo_id": DATASET_REPO_ID,
        "episode_count": payload["episode_count"],
        "min_episodes": MIN_TRAIN_EPISODES,
        "frame_count": payload["frame_count"],
        "seed_coverage": payload["seed_coverage"],
        "task_coverage": payload["task_coverage"],
        "source_files": payload["source_files"],
        "provenance_path": payload.get("provenance_path"),
        "materialization": materialization_report,
        "dataset_guard": guard_report,
        **expected,
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

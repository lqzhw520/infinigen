#!/usr/bin/env python3
"""G6: Package natural robot-scene rollouts into a finalized LeRobot dataset."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dataset_builder import build_dataset_from_rollouts
from mint_common import ARTIFACT_DIR, DATASET_DIR, DATASET_REPO_ID, load_json

ARTIFACT = ARTIFACT_DIR / "g6_lerobot_pack.json"
INTEGRITY = ARTIFACT_DIR / "g6_dataset_integrity.json"
DEFAULT_SOURCE_DIR = ARTIFACT_DIR / "g5_delta_rollouts"
MIN_EPISODES = 12
RCA5_ARTIFACT = ARTIFACT_DIR / "p2rca5_frozen_matrix_screen.json"
RCA7_ARTIFACT = ARTIFACT_DIR / "p2rca7_tiny_retrain_if_eligible.json"


def _expected_training_targets() -> dict[str, object]:
    rca5 = load_json(RCA5_ARTIFACT, {})
    rca7 = load_json(RCA7_ARTIFACT, {})
    return {
        "best_transition_cell": rca5.get("best_transition_cell"),
        "canonical_train_cell": rca5.get("canonical_train_cell") or rca7.get("canonical_train_cell"),
        "best_train_state_mode": rca5.get("best_train_state_mode") or rca7.get("best_train_state_mode"),
        "tiny_retrain_permitted": bool(rca7.get("tiny_retrain_permitted", False)),
    }


def run() -> bool:
    source_dir = Path(os.environ.get("MINT_G6_SOURCE_DIR", str(DEFAULT_SOURCE_DIR)))
    expected = _expected_training_targets()
    rollout_paths = sorted(source_dir.glob("*.npz")) if source_dir.exists() else []
    if not rollout_paths:
        result = {
            "gate": "g6_lerobot_pack",
            "passed": False,
            "error": "No rollout NPZ files found for packaging",
            "source_dir": str(source_dir),
            "dataset_root": str(DATASET_DIR),
            "repo_id": DATASET_REPO_ID,
            **expected,
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False

    payload = build_dataset_from_rollouts(rollout_paths, DATASET_DIR, DATASET_REPO_ID)
    integrity = payload["integrity"]
    INTEGRITY.write_text(json.dumps({**integrity, "timestamp": time.time()}, indent=2))

    result = {
        "gate": "g6_lerobot_pack",
        "passed": bool(payload["episode_count"] >= MIN_EPISODES and integrity["passed"]),
        "source_dir": str(source_dir),
        "dataset_root": str(DATASET_DIR),
        "repo_id": DATASET_REPO_ID,
        "episode_count": payload["episode_count"],
        "min_episodes": MIN_EPISODES,
        "frame_count": payload["frame_count"],
        "seed_coverage": payload["seed_coverage"],
        "task_coverage": payload["task_coverage"],
        "source_files": payload["source_files"],
        "provenance_path": payload.get("provenance_path"),
        **expected,
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

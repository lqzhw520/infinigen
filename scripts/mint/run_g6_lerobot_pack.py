#!/usr/bin/env python3
"""G6: Package natural robot-scene rollouts into a finalized LeRobot dataset."""

from __future__ import annotations

import json
import time

from dataset_builder import build_dataset_from_rollouts
from mint_common import ARTIFACT_DIR, DATASET_DIR, DATASET_REPO_ID

ARTIFACT = ARTIFACT_DIR / "g6_lerobot_pack.json"
INTEGRITY = ARTIFACT_DIR / "g6_dataset_integrity.json"
SOURCE_DIR = ARTIFACT_DIR / "g5_delta_rollouts"
MIN_EPISODES = 12


def run() -> bool:
    rollout_paths = sorted(SOURCE_DIR.glob("*.npz"))
    payload = build_dataset_from_rollouts(rollout_paths, DATASET_DIR, DATASET_REPO_ID)
    integrity = payload["integrity"]
    INTEGRITY.write_text(json.dumps({**integrity, "timestamp": time.time()}, indent=2))

    result = {
        "gate": "g6_lerobot_pack",
        "passed": bool(
            payload["episode_count"] >= MIN_EPISODES and integrity["passed"]
        ),
        "dataset_root": str(DATASET_DIR),
        "repo_id": DATASET_REPO_ID,
        "episode_count": payload["episode_count"],
        "min_episodes": MIN_EPISODES,
        "frame_count": payload["frame_count"],
        "seed_coverage": payload["seed_coverage"],
        "task_coverage": payload["task_coverage"],
        "source_files": payload["source_files"],
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

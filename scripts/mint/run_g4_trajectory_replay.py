#!/usr/bin/env python3
"""G4: Generate and verify proxy drawer-opening rollouts on training seeds."""

from __future__ import annotations

import json
import time

from drawer_proxy import rollout_episode, save_rollout
from mint_common import ARTIFACT_DIR, DEFAULT_TRAIN_SEEDS

ARTIFACT = ARTIFACT_DIR / "g4_trajectory_replay.json"
ROLLOUT_DIR = ARTIFACT_DIR / "g4_rollouts"


def run() -> bool:
    ROLLOUT_DIR.mkdir(parents=True, exist_ok=True)
    episodes = []
    success_count = 0
    for seed in DEFAULT_TRAIN_SEEDS:
        for episode_idx in range(2):
            rollout = rollout_episode(
                seed=seed,
                policy="expert",
                max_steps=24,
                rng_seed=seed * 100 + episode_idx,
            )
            path = ROLLOUT_DIR / f"seed_{seed:03d}_episode_{episode_idx:02d}.npz"
            save_rollout(path, rollout)
            success_count += int(rollout["success"])
            episodes.append(
                {
                    "seed": seed,
                    "episode_index": episode_idx,
                    "path": str(path),
                    "success": bool(rollout["success"]),
                    "steps": int(rollout["steps"]),
                }
            )

    result = {
        "gate": "g4_trajectory_replay",
        "passed": success_count == len(episodes),
        "episodes": episodes,
        "success_count": success_count,
        "episode_count": len(episodes),
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

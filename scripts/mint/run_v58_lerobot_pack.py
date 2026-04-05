#!/usr/bin/env python3
"""V58 G6: Package V58 physics-legal rollouts into a LeRobot dataset.

Reads rollouts from:
  1. v58_physics_legal_rollouts/ (new rollouts from run_v58_data_generation.py)
  2. p4_physics_legal_rollouts/ (existing P4 rollouts for seeds 1-6)

Outputs to: experiments/mint/mint_drawer_v1/dataset/

Usage:
  python scripts/mint/run_v58_lerobot_pack.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
SCRIPTS_MINT = PROJECT_ROOT / "scripts" / "mint"
sys.path.insert(0, str(SCRIPTS_MINT))

from dataset_builder import (
    STATE_NAMES,
    ACTION_NAMES,
    build_dataset_from_rollouts,
    _infinigen_to_libero_gripper,
    _infinigen_to_libero_action,
)
from mint_common import ARTIFACT_DIR, DATASET_DIR, DATASET_REPO_ID

V58_ROLLOUTS = ARTIFACT_DIR / "v58_physics_legal_rollouts"
P4_ROLLOUTS = ARTIFACT_DIR / "p4_physics_legal_rollouts"
ARTIFACT = ARTIFACT_DIR / "v58_lerobot_pack.json"
INTEGRITY = ARTIFACT_DIR / "v58_dataset_integrity.json"

# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_list(val):
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return val
    return list(val)


def find_legal_npz(source_dirs: list[Path]) -> list[Path]:
    """Collect all successful NPZ files from source dirs."""
    paths = []
    for d in source_dirs:
        if not d.exists():
            continue
        for pf in sorted(d.glob("*.npz")):
            try:
                data = dict(np.load(pf, allow_pickle=True))
                if data.get("success"):
                    paths.append(pf)
            except Exception:
                pass
    return paths


def pack_v58_dataset() -> dict:
    """Pack V58 rollouts into LeRobot dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rollout_paths = find_legal_npz([V58_ROLLOUTS, P4_ROLLOUTS])

    if not rollout_paths:
        return {
            "passed": False,
            "error": f"No legal rollouts found in {[str(d) for d in [V58_ROLLOUTS, P4_ROLLOUTS]]}",
            "rollout_count": 0,
            "frame_count": 0,
        }

    print(f"[V58 pack] Found {len(rollout_paths)} legal rollouts")
    for pf in rollout_paths:
        print(f"  {pf.name}")

    # Build the LeRobot dataset
    result = build_dataset_from_rollouts(
        rollout_paths=rollout_paths,
        dataset_root=DATASET_DIR,
        repo_id=DATASET_REPO_ID,
        robot_type="infinigen_drawer_anygrasp_robot",
        remove_existing=True,
        gripper_binarize=False,
        image_size=256,
    )

    return result


def run() -> bool:
    t0 = time.time()

    print(f"[V58 pack] Packing rollouts into LeRobot dataset")
    print(f"[V58 pack] Output: {DATASET_DIR}")
    print(f"[V58 pack] Repo ID: {DATASET_REPO_ID}")

    # Pack
    payload = pack_v58_dataset()
    integrity = payload.get("integrity", {})

    elapsed = time.time() - t0

    passed = (
        payload.get("integrity", {}).get("passed", False)
        and payload.get("episode_count", 0) > 0
        and payload.get("frame_count", 0) >= 100  # at least 100 frames
    )

    result = {
        "gate": "v58_lerobot_pack",
        "passed": passed,
        "dataset_root": str(DATASET_DIR),
        "repo_id": DATASET_REPO_ID,
        "episode_count": payload.get("episode_count", 0),
        "frame_count": payload.get("frame_count", 0),
        "seed_coverage": payload.get("seed_coverage", []),
        "task_coverage": payload.get("task_coverage", []),
        "integrity": integrity,
        "elapsed_sec": round(elapsed, 1),
        "timestamp": time.time(),
    }

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(result, indent=2))
    if INTEGRITY.parent.exists():
        INTEGRITY.write_text(json.dumps({**integrity, "timestamp": time.time()}, indent=2))

    print(json.dumps(result, indent=2))
    return passed


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(SCRIPTS_MINT))
    raise SystemExit(0 if run() else 1)

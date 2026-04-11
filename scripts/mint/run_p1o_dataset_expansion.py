#!/usr/bin/env python3
"""P1B: expand dataset with more unique successful seeds and limited repetition."""

from __future__ import annotations

import json
from pathlib import Path

from mint_common import now_iso, write_json_atomic, write_text_atomic
from p1_execution_common import ARTIFACT_DIR, CAMPAIGN_DIR, OUTPUT_DIR, upsert_evidence
from drawer_robot_env_mujoco import drawer_manifest
from mujoco_mainline_common import default_heldout_seeds
import run_mujoco_infinigen_mainline_night as mainline

ARTIFACT_PATH = ARTIFACT_DIR / "p1o_dataset_expansion_manifest.json"
REPORT_PATH = OUTPUT_DIR / "p1o_dataset_expansion_manifest.md"
EVIDENCE_ID = "E057"
EXPERIMENT_ID = "p1o_dataset_expansion_manifest"
TARGET_UNIQUE_SEEDS = 15
SUCCESS_REPEAT = 2
MAX_STEPS = 200
DATASET_ROOT = CAMPAIGN_DIR / "dataset_p1o_v2"
DATASET_REPO_ID = "infinigen_drawer_robot_v2"


def main() -> int:
    manifest = drawer_manifest()
    heldout = default_heldout_seeds()
    train_pool = [seed for seed in manifest["available_seeds"] if seed not in heldout]

    mainline.TRAIN_SEEDS = train_pool
    mainline.HELDOUT_SEEDS = heldout
    mainline.MIN_ANYGRASP_SEEDS = TARGET_UNIQUE_SEEDS
    mainline.MIN_ORACLE_SEEDS = TARGET_UNIQUE_SEEDS
    mainline.SUCCESS_REPEAT = SUCCESS_REPEAT
    mainline.MAX_STEPS = MAX_STEPS
    mainline.M3_CACHE_DIR = ARTIFACT_DIR / "p1o_m3_mujoco_anygrasp"
    mainline.M4_ROLLOUT_ANY = ARTIFACT_DIR / "p1o_rollouts_anygrasp_mujoco"
    mainline.M4_ROLLOUT_ORACLE = ARTIFACT_DIR / "p1o_rollouts_oracle_mujoco"
    mainline.M4_ROLLOUT_LEARNING = ARTIFACT_DIR / "p1o_rollouts_learning_mujoco"
    mainline.DATASET_DIR = DATASET_ROOT
    mainline.DATASET_REPO_ID = DATASET_REPO_ID

    m3 = mainline.run_m3_anygrasp_gate()
    m4 = mainline.run_m4_robot_rollout_gate() if m3.get("passed") else {"passed": False, "error": "m3 failed"}
    m5 = mainline.run_m5_dataset_pack_gate() if m4.get("passed") else {"passed": False, "error": "m4 failed"}

    unique_successful = [int(x) for x in m4.get("successful_learning_seeds", [])]
    total_frames = int(m5.get("frame_count", 0) or 0)
    gate_passed = len(unique_successful) >= TARGET_UNIQUE_SEEDS and total_frames >= 5000 and bool(m5.get("passed"))

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at": now_iso(),
        "passed": gate_passed,
        "target_unique_train_seeds": TARGET_UNIQUE_SEEDS,
        "heldout_seeds": heldout,
        "train_seed_pool": train_pool,
        "success_repeat": SUCCESS_REPEAT,
        "max_steps": MAX_STEPS,
        "dataset_root": str(DATASET_ROOT),
        "dataset_repo_id": DATASET_REPO_ID,
        "unique_successful_train_seeds": unique_successful,
        "unique_successful_train_seed_count": len(unique_successful),
        "total_frames": total_frames,
        "total_episodes": int(m5.get("episode_count", 0) or 0),
        "per_seed_success_rate": {
            str(item["seed"]): float(item.get("passed", False))
            for item in m4.get("anygrasp_records", [])
        },
        "rollout_source": m4.get("learning_source"),
        "m3": m3,
        "m4": m4,
        "m5": m5,
        "gate_reason": (
            "Dataset expansion met unique-seed and frame-count thresholds."
            if gate_passed
            else "Dataset expansion did not meet >=15 successful unique train seeds and ~5000+ frames."
        ),
    }
    write_json_atomic(ARTIFACT_PATH, payload)

    lines = [
        "# p1o Dataset Expansion Manifest",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Summary",
        f"- passed: {gate_passed}",
        f"- unique_successful_train_seed_count: {len(unique_successful)}",
        f"- total_frames: {total_frames}",
        f"- total_episodes: {payload['total_episodes']}",
        f"- rollout_source: {payload['rollout_source']}",
        f"- dataset_root: `{DATASET_ROOT}`",
        f"- heldout_seeds: {heldout}",
    ]
    write_text_atomic(REPORT_PATH, "\n".join(lines).rstrip() + "\n")

    upsert_evidence(
        EVIDENCE_ID,
        EXPERIMENT_ID,
        "dataset_expansion_manifest",
        ARTIFACT_PATH,
        "Expanded MuJoCo dataset manifest using a wider train-seed pool, capped repetition, and a separate v2 dataset root to avoid overwriting the canonical baseline dataset.",
        verified=gate_passed,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

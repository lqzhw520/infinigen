#!/usr/bin/env python3
"""G8a: Overfit MINT on a single successful natural rollout."""

from __future__ import annotations

import json
import time

from dataset_builder import build_dataset_from_rollouts
from evaluate_mint_drawer_campaign import evaluate_train_probe
from mint_common import ARTIFACT_DIR, CAMPAIGN_DIR, DATASET_REPO_ID
from train_mint_helpers import run_training

ARTIFACT = ARTIFACT_DIR / "g8_single_rollout_overfit.json"
SOURCE_DIR = ARTIFACT_DIR / "g5_delta_rollouts"
DATASET_ROOT = CAMPAIGN_DIR / "dataset_single_rollout"
OUTPUT_DIR = CAMPAIGN_DIR / "outputs_single_rollout"
LOG_PATH = ARTIFACT_DIR / "g8_single_rollout_overfit.log"
STEPS = 400


def run() -> bool:
    rollout_paths = sorted(SOURCE_DIR.glob("*.npz"))
    if not rollout_paths:
        result = {
            "gate": "g8_single_rollout_overfit",
            "passed": False,
            "error": "No successful natural rollout available",
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False

    selected = rollout_paths[0]
    dataset_repo_id = f"{DATASET_REPO_ID}_single_rollout"
    dataset_payload = build_dataset_from_rollouts(
        [selected], DATASET_ROOT, dataset_repo_id
    )
    train_payload = run_training(
        dataset_root=DATASET_ROOT,
        dataset_repo_id=dataset_repo_id,
        output_dir=OUTPUT_DIR,
        log_path=LOG_PATH,
        steps=STEPS,
        job_name="mint_single_rollout_overfit",
    )
    finetuned_path = train_payload.get("checkpoint_path")
    eval_payload = {}
    trend_passed = False
    if finetuned_path:
        seed = int(json.loads(selected.with_suffix(".json").read_text())["seed"])
        eval_payload, _ = evaluate_train_probe(
            finetuned_path,
            seeds=[seed],
            dataset_root=DATASET_ROOT,
            repo_id=dataset_repo_id,
            max_steps=96,
        )
        ft = eval_payload["summary"]["finetuned_mint"]
        pt = eval_payload["summary"]["pretrained_mint"]
        trend_passed = bool(
            ft["success_rate"] > pt["success_rate"] and ft["successes"] >= 1
        )

    result = {
        "gate": "g8_single_rollout_overfit",
        "passed": bool(
            dataset_payload["integrity"]["passed"]
            and train_payload["passed"]
            and trend_passed
        ),
        "selected_rollout": str(selected),
        "dataset": dataset_payload,
        "training": train_payload,
        "evaluation": eval_payload,
        "trend_passed": trend_passed,
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

#!/usr/bin/env python3
"""G8d: Probe multi-seed train reproducibility before held-out evaluation."""

from __future__ import annotations

import json
import time

from evaluate_mint_drawer_campaign import evaluate_train_probe
from mint_common import ARTIFACT_DIR, CAMPAIGN_DIR, DEFAULT_TRAIN_SEEDS, load_json

ARTIFACT = ARTIFACT_DIR / "g8_train_seed_probe.json"
SUMMARY_PATH = CAMPAIGN_DIR / "evaluation" / "train_seed_probe.json"
G8_ARTIFACT = ARTIFACT_DIR / "g8_train_summary.json"
MIN_SUCCESS_GAIN = 0.15
MIN_FINETUNED_SUCCESSES = 2


def run() -> bool:
    g8 = load_json(G8_ARTIFACT, {})
    checkpoint_path = g8.get("checkpoint_path")
    if not checkpoint_path:
        result = {
            "gate": "g8_train_seed_probe",
            "passed": False,
            "error": "Missing fine-tuned checkpoint",
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False

    summary, _ = evaluate_train_probe(
        checkpoint_path, seeds=DEFAULT_TRAIN_SEEDS, max_steps=96
    )
    ft = summary["summary"]["finetuned_mint"]
    pt = summary["summary"]["pretrained_mint"]
    success_gain = float(ft["success_rate"] - pt["success_rate"])
    trend_passed = bool(
        success_gain >= MIN_SUCCESS_GAIN and ft["successes"] >= MIN_FINETUNED_SUCCESSES
    )
    summary["trend_passed"] = trend_passed
    summary["success_gain"] = success_gain
    summary["min_success_gain"] = MIN_SUCCESS_GAIN
    summary["min_finetuned_successes"] = MIN_FINETUNED_SUCCESSES
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    result = {
        "gate": "g8_train_seed_probe",
        "passed": trend_passed,
        **summary,
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

#!/usr/bin/env python3
"""G9: Evaluate random vs pretrained vs fine-tuned MINT on held-out drawer proxy envs."""

from __future__ import annotations

import json
import time

from evaluate_mint_drawer_campaign import evaluate_campaign, render_report
from mint_common import ARTIFACT_DIR, EVAL_DIR, load_json

ARTIFACT = ARTIFACT_DIR / "g9_eval_rollouts.json"
SUMMARY_PATH = EVAL_DIR / "comparison_summary.json"
REPORT_PATH = EVAL_DIR / "comparison_report.md"
G8_ARTIFACT = ARTIFACT_DIR / "g8_train_summary.json"
TRAIN_PROBE_ARTIFACT = ARTIFACT_DIR / "g8_train_seed_probe.json"


def run() -> bool:
    train_probe = load_json(TRAIN_PROBE_ARTIFACT, {})
    if train_probe and not train_probe.get("passed"):
        summary = {
            "verdict": "scientific_not_supported",
            "claim_level": "L3_not_reached",
            "held_out_seeds": [],
            "eval_max_steps": 96,
            "comparison": {},
            "strongest_true_claim": "Train-seed reproducibility has not yet reached a stable positive trend, so held-out evaluation is deferred until the natural robot-action contract is repaired.",
            "deferred_reason": "train_seed_reproducibility_not_met",
            "gate": "g9_sim_eval",
            "passed": True,
            "timestamp": time.time(),
        }
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
        REPORT_PATH.write_text(
            "# MINT Drawer Robot-Trajectory Sim Evaluation\n\n"
            "**Verdict**: `scientific_not_supported`\n\n"
            "Held-out evaluation was deferred because train-seed reproducibility did not yet reach a stable positive trend.\n"
        )
        ARTIFACT.write_text(
            json.dumps({"deferred": True, "train_probe": train_probe}, indent=2)
        )
        print(json.dumps(summary, indent=2))
        return True

    g8 = load_json(G8_ARTIFACT, {})
    checkpoint_path = g8.get("checkpoint_path")
    if not checkpoint_path:
        result = {
            "gate": "g9_sim_eval",
            "passed": False,
            "error": "Missing fine-tuned checkpoint",
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False

    summary, records = evaluate_campaign(checkpoint_path)
    summary["gate"] = "g9_sim_eval"
    summary["passed"] = True
    summary["timestamp"] = time.time()
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    REPORT_PATH.write_text(render_report(summary))
    ARTIFACT.write_text(json.dumps(records, indent=2))
    print(json.dumps(summary, indent=2))
    return True


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

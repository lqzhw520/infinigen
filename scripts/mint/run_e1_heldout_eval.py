#!/usr/bin/env python3
"""E1: Run the held-out claim evaluation after train-side reproducibility passes."""

from __future__ import annotations

import json
import time

from evaluate_mint_drawer_campaign import evaluate_campaign, render_report
from mint_common import CAMPAIGN_DIR, EVAL_DIR, load_json
from video_reporting import choose_representative_seed, safe_render_policy_pair

ARTIFACT = CAMPAIGN_DIR / "artifacts" / "e1_eval_rollouts.json"
D3_ARTIFACT = CAMPAIGN_DIR / "artifacts" / "d3_train_seed_probe.json"


def run() -> bool:
    d3 = load_json(D3_ARTIFACT, {})
    if not d3.get("passed"):
        result = {
            "gate": "e1_heldout_eval",
            "passed": False,
            "error": "Train-seed reproducibility gate has not passed",
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False
    checkpoint_path = d3.get("training", {}).get("checkpoint_path")
    dataset_root = d3.get("dataset", {}).get("dataset_root")
    repo_id = d3.get("dataset", {}).get("repo_id")
    if not checkpoint_path or not dataset_root or not repo_id:
        result = {
            "gate": "e1_heldout_eval",
            "passed": False,
            "error": "Missing D3 checkpoint or dataset metadata",
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False

    summary, records = evaluate_campaign(
        checkpoint_path, dataset_root=dataset_root, repo_id=repo_id
    )
    summary["gate"] = "e1_heldout_eval"
    summary["passed"] = True
    summary["timestamp"] = time.time()
    representative_seed = choose_representative_seed(
        summary.get("comparison", {}),
        preferred_policy="finetuned_mint",
        fallback_policy="pretrained_mint",
    )
    video_outputs = {}
    if representative_seed is not None:
        try:
            video_outputs = safe_render_policy_pair(
                campaign_dir=CAMPAIGN_DIR,
                stage="e1",
                seed=int(representative_seed),
                finetuned_path=str(checkpoint_path),
                dataset_root=str(dataset_root),
                repo_id=str(repo_id),
                variant="heldout_eval",
                max_steps=96,
                attempt_idx=0,
            )
        except (
            Exception
        ) as exc:  # pragma: no cover - reporting should not fail the gate
            video_outputs = {"error": f"{type(exc).__name__}: {exc}"}
    summary["representative_seed"] = representative_seed
    summary["video_outputs"] = video_outputs
    (EVAL_DIR / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    (EVAL_DIR / "comparison_report.md").write_text(render_report(summary))
    ARTIFACT.write_text(json.dumps(records, indent=2))
    print(json.dumps(summary, indent=2))
    return True


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

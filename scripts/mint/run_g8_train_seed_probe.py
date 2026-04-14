#!/usr/bin/env python3
"""G8d: Plan-driven train-seed probe for tiny retrain confirmation."""

from __future__ import annotations

import json
import time
from pathlib import Path

from evaluate_mint_drawer_campaign_mujoco import evaluate_train_probe
from mint_common import (
    ARTIFACT_DIR,
    PROJECT_ROOT,
    TINY_RETRAIN_EVAL_DIR,
    TINY_RETRAIN_PLAN_PATH,
    load_json,
)

ARTIFACT = ARTIFACT_DIR / "g8_train_seed_probe.json"
G8_ARTIFACT = ARTIFACT_DIR / "g8_train_summary.json"


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def run() -> bool:
    plan = load_json(TINY_RETRAIN_PLAN_PATH, {})
    if not plan:
        raise SystemExit(f"Missing active tiny retrain plan: {TINY_RETRAIN_PLAN_PATH}")
    g8 = load_json(G8_ARTIFACT, {})
    checkpoint_path = g8.get("checkpoint_path")
    if not checkpoint_path:
        result = {
            "gate": "g8_train_seed_probe",
            "training_mode": "tiny_retrain_confirmation",
            "canonical_train_cell": plan.get("canonical_train_cell"),
            "probe_seeds": plan.get("train_seeds", []),
            "min_success_gain": float(plan.get("train_probe_min_success_gain", 0.15)),
            "min_finetuned_successes": int(plan.get("train_probe_min_successes", 2)),
            "trend_passed": False,
            "passed": False,
            "error": "Missing fine-tuned checkpoint",
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False

    probe_seeds = [int(seed) for seed in plan.get("train_seeds", [])]
    dataset_root = _resolve_repo_path(plan["dataset_root"])
    summary, _ = evaluate_train_probe(
        checkpoint_path,
        seeds=probe_seeds,
        dataset_root=dataset_root,
        repo_id=str(plan["dataset_repo_id"]),
        canonical_train_cell=str(plan.get("evaluation_cell_id") or plan.get("canonical_train_cell")),
        best_train_state_mode=str(plan.get("best_train_state_mode")),
        episodes_per_seed=int(plan.get("evaluation_probe_episodes_per_seed", 3)),
        max_steps=int(plan.get("evaluation_max_steps", 96)),
        image_size=int(plan.get("evaluation_image_size", 256)),
        evaluation_backend=str(plan.get("evaluation_backend", "mujoco")),
    )
    ft = summary["summary"]["finetuned_mint"]
    pt = summary["summary"]["pretrained_mint"]
    min_success_gain = float(plan.get("train_probe_min_success_gain", 0.15))
    min_finetuned_successes = int(plan.get("train_probe_min_successes", 2))
    success_gain = float(ft["success_rate"] - pt["success_rate"])
    trend_passed = bool(success_gain >= min_success_gain and ft["successes"] >= min_finetuned_successes)
    summary_path = _resolve_repo_path(plan["evaluation_dir"]) / "train_seed_probe.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        **summary,
        "training_mode": "tiny_retrain_confirmation",
        "canonical_train_cell": plan.get("canonical_train_cell"),
        "best_train_state_mode": plan.get("best_train_state_mode"),
        "evaluation_backend": plan.get("evaluation_backend"),
        "evaluation_env_family": plan.get("evaluation_env_family"),
        "evaluation_cell_id": plan.get("evaluation_cell_id"),
        "evaluation_state_mode_name": plan.get("evaluation_state_mode_name"),
        "evaluation_interaction_mode": plan.get("evaluation_interaction_mode"),
        "probe_seeds": probe_seeds,
        "min_success_gain": min_success_gain,
        "min_finetuned_successes": min_finetuned_successes,
        "success_gain": success_gain,
        "trend_passed": trend_passed,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    result = {
        "gate": "g8_train_seed_probe",
        "training_mode": "tiny_retrain_confirmation",
        "canonical_train_cell": plan.get("canonical_train_cell"),
        "best_train_state_mode": plan.get("best_train_state_mode"),
        "evaluation_backend": plan.get("evaluation_backend"),
        "evaluation_env_family": plan.get("evaluation_env_family"),
        "evaluation_cell_id": plan.get("evaluation_cell_id"),
        "evaluation_state_mode_name": plan.get("evaluation_state_mode_name"),
        "evaluation_interaction_mode": plan.get("evaluation_interaction_mode"),
        "probe_seeds": probe_seeds,
        "min_success_gain": min_success_gain,
        "min_finetuned_successes": min_finetuned_successes,
        "success_gain": success_gain,
        "trend_passed": trend_passed,
        "passed": trend_passed,
        **summary,
        "summary_path": str(summary_path),
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

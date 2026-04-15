#!/usr/bin/env python3
"""G9: Plan-driven held-out sim eval for tiny retrain confirmation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from evaluate_mint_drawer_campaign_mujoco import evaluate_campaign, render_report
from mint_common import EVAL_DIR, PROJECT_ROOT, TINY_RETRAIN_PLAN_PATH, load_json

ARTIFACT = EVAL_DIR / "g9_eval_rollouts.json"
SUMMARY_PATH = EVAL_DIR / "comparison_summary.json"
REPORT_PATH = EVAL_DIR / "comparison_report.md"
G8_ARTIFACT = Path("/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/g8_train_summary.json")
TRAIN_PROBE_ARTIFACT = Path("/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/g8_train_seed_probe.json")


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _source_canonical_train_cell(plan: dict[str, Any]) -> str:
    return str(plan.get("source_canonical_train_cell") or plan.get("canonical_train_cell") or "")


def _source_best_train_state_mode(plan: dict[str, Any]) -> str:
    return str(plan.get("source_best_train_state_mode") or plan.get("best_train_state_mode") or "")


def _active_train_state_mode(plan: dict[str, Any]) -> str:
    return str(plan.get("active_train_state_mode") or _source_best_train_state_mode(plan))


def _active_state_mode_name(plan: dict[str, Any]) -> str:
    return str(plan.get("active_state_mode_name") or _active_train_state_mode(plan))


def run() -> bool:
    plan = load_json(TINY_RETRAIN_PLAN_PATH, {})
    if not plan:
        raise SystemExit(f"Missing active tiny retrain plan: {TINY_RETRAIN_PLAN_PATH}")
    g8 = load_json(G8_ARTIFACT, {})
    probe = load_json(TRAIN_PROBE_ARTIFACT, {})
    if str(g8.get("run_instance_id") or "") != str(plan.get("run_instance_id") or ""):
        raise SystemExit("G8 train summary run_instance_id does not match active plan")
    if str(probe.get("run_instance_id") or "") != str(plan.get("run_instance_id") or ""):
        raise SystemExit("Train probe run_instance_id does not match active plan")
    checkpoint_path = probe.get("selected_bridge_checkpoint") or g8.get("checkpoint_path")
    if _source_canonical_train_cell(plan) != "V1cT2S0":
        raise SystemExit("Active tiny retrain plan source_canonical_train_cell is not V1cT2S0")
    if not checkpoint_path:
        raise SystemExit("Missing fine-tuned checkpoint for held-out eval")
    if not probe or not bool(probe.get("train_probe_claim_pass", probe.get("trend_passed", False))):
        raise SystemExit("Train probe did not pass; held-out eval must not run")

    heldout_seeds = [int(seed) for seed in plan.get("heldout_seeds", [])]
    summary, records = evaluate_campaign(
        checkpoint_path,
        dataset_root=_resolve_repo_path(plan["dataset_root"]),
        repo_id=str(plan["dataset_repo_id"]),
        heldout_seeds=heldout_seeds,
        canonical_train_cell=str(plan.get("evaluation_cell_id") or _source_canonical_train_cell(plan)),
        source_best_train_state_mode=_source_best_train_state_mode(plan),
        active_train_state_mode=_active_train_state_mode(plan),
        active_state_mode_name=_active_state_mode_name(plan),
        episodes_per_seed=int(plan.get("evaluation_heldout_episodes_per_seed", 3)),
        max_steps=int(plan.get("evaluation_max_steps", 96)),
        image_size=int(plan.get("evaluation_image_size", 256)),
        evaluation_backend=str(plan.get("evaluation_backend", "mujoco")),
    )
    summary.update(
        {
            "gate": "g9_sim_eval",
            "training_mode": "tiny_retrain_confirmation",
            "run_instance_id": plan.get("run_instance_id"),
            "plan_version": plan.get("plan_version"),
            "source_base_commit": plan.get("source_base_commit"),
            "working_head_commit": plan.get("working_head_commit"),
            "source_canonical_train_cell": _source_canonical_train_cell(plan),
            "source_best_train_state_mode": _source_best_train_state_mode(plan),
            "canonical_train_cell": _source_canonical_train_cell(plan),
            "best_train_state_mode": _source_best_train_state_mode(plan),
            "active_train_state_mode": _active_train_state_mode(plan),
            "active_state_mode_name": _active_state_mode_name(plan),
            "bridge_stage": plan.get("bridge_stage"),
            "bridge_attempt": plan.get("bridge_attempt"),
            "evaluation_backend": plan.get("evaluation_backend"),
            "evaluation_env_family": plan.get("evaluation_env_family"),
            "evaluation_cell_id": plan.get("evaluation_cell_id"),
            "evaluation_interaction_mode": plan.get("evaluation_interaction_mode"),
            "heldout_eval_run": True,
            "claim_supported": summary.get("verdict") == "claim_supported",
            "selected_bridge_checkpoint": probe.get("selected_bridge_checkpoint"),
            "selected_checkpoint_step": probe.get("selected_checkpoint_step"),
            "timestamp": time.time(),
        }
    )
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    REPORT_PATH.write_text(render_report(summary))
    ARTIFACT.write_text(json.dumps(records, indent=2))
    print(json.dumps(summary, indent=2))
    return True


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

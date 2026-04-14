#!/usr/bin/env python3
"""G9: Plan-driven held-out sim eval for tiny retrain confirmation."""

from __future__ import annotations

import json
import time
from pathlib import Path

from evaluate_mint_drawer_campaign import evaluate_campaign, render_report
from mint_common import EVAL_DIR, PROJECT_ROOT, TINY_RETRAIN_PLAN_PATH, load_json

ARTIFACT = EVAL_DIR / "g9_eval_rollouts.json"
SUMMARY_PATH = EVAL_DIR / "comparison_summary.json"
REPORT_PATH = EVAL_DIR / "comparison_report.md"
G8_ARTIFACT = Path("/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/g8_train_summary.json")
TRAIN_PROBE_ARTIFACT = Path("/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/g8_train_seed_probe.json")


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def run() -> bool:
    plan = load_json(TINY_RETRAIN_PLAN_PATH, {})
    if not plan:
        raise SystemExit(f"Missing active tiny retrain plan: {TINY_RETRAIN_PLAN_PATH}")
    g8 = load_json(G8_ARTIFACT, {})
    probe = load_json(TRAIN_PROBE_ARTIFACT, {})
    checkpoint_path = g8.get("checkpoint_path")
    if str(plan.get("canonical_train_cell")) != "V1cT2S0":
        raise SystemExit("Active tiny retrain plan canonical_train_cell is not V1cT2S0")
    if str(plan.get("best_train_state_mode")) != "S0":
        raise SystemExit("Active tiny retrain plan best_train_state_mode is not S0")
    if not checkpoint_path:
        raise SystemExit("Missing fine-tuned checkpoint for held-out eval")
    if not probe or not bool(probe.get("trend_passed", False)):
        raise SystemExit("Train probe did not pass; held-out eval must not run")

    heldout_seeds = [int(seed) for seed in plan.get("heldout_seeds", [])]
    summary, records = evaluate_campaign(
        checkpoint_path,
        dataset_root=_resolve_repo_path(plan["dataset_root"]),
        repo_id=str(plan["dataset_repo_id"]),
        held_out_seeds=heldout_seeds,
        episodes_per_seed=3,
    )
    summary.update({
        "gate": "g9_sim_eval",
        "training_mode": "tiny_retrain_confirmation",
        "canonical_train_cell": plan.get("canonical_train_cell"),
        "best_train_state_mode": plan.get("best_train_state_mode"),
        "heldout_eval_run": True,
        "claim_supported": summary.get("verdict") == "claim_supported",
        "timestamp": time.time(),
    })
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    REPORT_PATH.write_text(render_report(summary))
    ARTIFACT.write_text(json.dumps(records, indent=2))
    print(json.dumps(summary, indent=2))
    return True


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

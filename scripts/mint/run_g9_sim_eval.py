#!/usr/bin/env python3
"""G9: Plan-driven held-out sim eval for tiny retrain confirmation."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from evaluate_mint_drawer_campaign_mujoco import render_report
from mint_common import EVAL_DIR, PROJECT_ROOT, TINY_RETRAIN_PLAN_PATH, load_json
from run_p1c10_release_runtime_matched_ab import AUTHORITATIVE_MINT_PYTHON, PATCHED_MINT_SRC, build_child_env

EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "mint" / "evaluate_mint_drawer_campaign_mujoco.py"

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


def _run_authoritative_campaign(plan: dict[str, Any], checkpoint_path: str, heldout_seeds: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not AUTHORITATIVE_MINT_PYTHON.exists():
        raise RuntimeError(f"Authoritative MINT python missing: {AUTHORITATIVE_MINT_PYTHON}")
    eval_dir = _resolve_repo_path(plan["evaluation_dir"])
    summary_path = eval_dir / "authoritative_campaign_summary.json"
    records_path = eval_dir / "authoritative_campaign_records.json"
    seeds_arg = ",".join(str(int(seed)) for seed in heldout_seeds)
    cmd = [
        str(AUTHORITATIVE_MINT_PYTHON),
        str(EVAL_SCRIPT),
        "--mode",
        "campaign",
        "--checkpoint-path",
        str(checkpoint_path),
        "--dataset-root",
        str(_resolve_repo_path(plan["dataset_root"])),
        "--repo-id",
        str(plan["dataset_repo_id"]),
        "--summary-path",
        str(summary_path),
        "--records-path",
        str(records_path),
        "--seeds",
        seeds_arg,
        "--max-steps",
        str(int(plan.get("evaluation_max_steps", 96))),
        "--episodes-per-seed",
        str(int(plan.get("evaluation_heldout_episodes_per_seed", 3))),
        "--image-size",
        str(int(plan.get("evaluation_image_size", 256))),
        "--canonical-train-cell",
        str(plan.get("evaluation_cell_id") or _source_canonical_train_cell(plan)),
        "--source-best-train-state-mode",
        _source_best_train_state_mode(plan),
        "--active-train-state-mode",
        _active_train_state_mode(plan),
        "--active-state-mode-name",
        _active_state_mode_name(plan),
        "--evaluation-backend",
        str(plan.get("evaluation_backend", "mujoco")),
    ]
    env = build_child_env(PATCHED_MINT_SRC)
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "authoritative heldout campaign failed: "
            f"stdout={proc.stdout[-2000:]} stderr={proc.stderr[-2000:]}"
        )
    return json.loads(summary_path.read_text()), json.loads(records_path.read_text())




def _scope_fields(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution_scope": str(plan.get("execution_scope") or "unspecified"),
        "diagnostic_only": bool(plan.get("diagnostic_only", False)),
        "claim_bearing": bool(plan.get("claim_bearing", False)),
        "publication_scope": str(plan.get("publication_scope") or "unspecified"),
        "result_scope": str(plan.get("result_scope") or "unspecified"),
    }


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
    summary, records = _run_authoritative_campaign(plan, str(checkpoint_path), heldout_seeds)
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
            **_scope_fields(plan),
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

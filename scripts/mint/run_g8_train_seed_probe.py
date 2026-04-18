#!/usr/bin/env python3
"""G8d: Plan-driven train-seed probe for tiny retrain confirmation."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from mint_common import ARTIFACT_DIR, PROJECT_ROOT, TINY_RETRAIN_PLAN_PATH, load_json, write_json_atomic
from run_p1c10_release_runtime_matched_ab import AUTHORITATIVE_MINT_PYTHON, PATCHED_MINT_SRC, build_child_env

EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "mint" / "evaluate_mint_drawer_campaign_mujoco.py"

ARTIFACT = ARTIFACT_DIR / "g8_train_seed_probe.json"
G8_ARTIFACT = ARTIFACT_DIR / "g8_train_summary.json"


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


def _scope_fields(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution_scope": str(plan.get("execution_scope") or "unspecified"),
        "diagnostic_only": bool(plan.get("diagnostic_only", False)),
        "claim_bearing": bool(plan.get("claim_bearing", False)),
        "publication_scope": str(plan.get("publication_scope") or "unspecified"),
        "result_scope": str(plan.get("result_scope") or "unspecified"),
    }


def _run_authoritative_train_probe(plan: dict[str, Any], checkpoint_path: str, checkpoint_step: int | None) -> dict[str, Any]:
    if not AUTHORITATIVE_MINT_PYTHON.exists():
        raise RuntimeError(f"Authoritative MINT python missing: {AUTHORITATIVE_MINT_PYTHON}")
    eval_dir = _resolve_repo_path(plan["evaluation_dir"])
    step_label = f"{int(checkpoint_step):06d}" if checkpoint_step is not None else "latest"
    summary_path = eval_dir / f"authoritative_train_probe_{step_label}_summary.json"
    records_path = eval_dir / f"authoritative_train_probe_{step_label}_records.json"
    seeds_arg = ",".join(str(int(seed)) for seed in plan.get("train_seeds", []))
    cmd = [
        str(AUTHORITATIVE_MINT_PYTHON),
        str(EVAL_SCRIPT),
        "--mode",
        "train_probe",
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
        str(int(plan.get("evaluation_probe_episodes_per_seed", 3))),
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
            "authoritative train probe failed: "
            f"stdout={proc.stdout[-2000:]} stderr={proc.stderr[-2000:]}"
        )
    return json.loads(summary_path.read_text())




def _bridge_delta(ft: dict[str, Any], pt: dict[str, Any]) -> dict[str, float]:
    keys = [
        "ever_attached_rate",
        "close_cmd_rate_mean",
        "distance_pass_rate_mean",
        "orientation_gate_pass_rate_mean",
        "approach_gate_pass_rate_mean",
        "attach_eligible_rate_mean",
        "stable_attach_rate_mean",
        "phase_locked_rate_mean",
        "max_drawer_fraction_mean",
        "grasp_success_rate",
        "success_rate",
    ]
    return {key: float(ft.get(key, 0.0)) - float(pt.get(key, 0.0)) for key in keys}


def _build_outputs(plan: dict[str, Any], checkpoint_path: str, checkpoint_step: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    probe_seeds = [int(seed) for seed in plan.get("train_seeds", [])]
    dataset_root = _resolve_repo_path(plan["dataset_root"])
    summary = _run_authoritative_train_probe(plan, str(checkpoint_path), checkpoint_step)
    ft = summary["summary"]["finetuned_mint"]
    pt = summary["summary"]["pretrained_mint"]
    min_success_gain = float(plan.get("train_probe_min_success_gain", 0.15))
    min_finetuned_successes = int(plan.get("train_probe_min_successes", 2))
    success_gain = float(ft["success_rate"] - pt["success_rate"])
    ever_attached_rate_gain = float(
        ft.get("ever_attached_rate", ft.get("grasp_success_rate", 0.0))
        - pt.get("ever_attached_rate", pt.get("grasp_success_rate", 0.0))
    )
    ever_attach_eligible_fraction_gain = float(
        ft.get("ever_attach_eligible_fraction", 0.0)
        - pt.get("ever_attach_eligible_fraction", 0.0)
    )
    stable_attach_gain = float(ft.get("ever_stable_attach_fraction", 0.0) - pt.get("ever_stable_attach_fraction", 0.0))
    phase_locked_gain = float(ft.get("phase_locked_rate_mean", 0.0) - pt.get("phase_locked_rate_mean", 0.0))
    max_drawer_fraction_gain = float(
        ft.get("max_drawer_fraction_mean", ft.get("pull_distance_mean", 0.0))
        - pt.get("max_drawer_fraction_mean", pt.get("pull_distance_mean", 0.0))
    )
    attached_seed_count = int(ft.get("attached_seed_count", 0))
    trend_passed = bool(
        success_gain >= min_success_gain
        and int(ft["successes"]) >= int(pt.get("successes", 0)) + min_finetuned_successes
        and float(ft["success_rate"]) >= 0.20
    )
    attach_bridge_pass = bool(
        ever_attach_eligible_fraction_gain >= 0.10
        and ever_attached_rate_gain >= 0.25
        and stable_attach_gain >= 0.20
        and phase_locked_gain >= 0.10
        and max_drawer_fraction_gain >= 0.10
        and attached_seed_count >= 4
    )
    attach_first_rank_vector = {
        "attach_bridge_pass": 1 if attach_bridge_pass else 0,
        "ever_attach_eligible_fraction": float(
            ft.get("ever_attach_eligible_fraction", 0.0)
        ),
        "ever_attached_rate_gain": ever_attached_rate_gain,
        "stable_attach_gain": stable_attach_gain,
        "phase_locked_gain": phase_locked_gain,
        "strict_success_rate_gain": float(ft.get("success_rate", 0.0) - pt.get("success_rate", 0.0)),
        "checkpoint_step": int(checkpoint_step or 0),
    }
    summary_payload = {
        **summary,
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
        "probe_seeds": probe_seeds,
        "checkpoint_path": checkpoint_path,
        "checkpoint_step": checkpoint_step,
        "min_success_gain": min_success_gain,
        "min_finetuned_successes": min_finetuned_successes,
        "success_gain": success_gain,
        "ever_attach_eligible_fraction_gain": ever_attach_eligible_fraction_gain,
        "ever_attached_rate_gain": ever_attached_rate_gain,
        "stable_attach_gain": stable_attach_gain,
        "phase_locked_gain": phase_locked_gain,
        "max_drawer_fraction_gain": max_drawer_fraction_gain,
        "attached_seed_count": attached_seed_count,
        "trend_passed": trend_passed,
        "train_probe_claim_pass": trend_passed,
        "attach_bridge_pass": attach_bridge_pass,
        "attach_first_rank_vector": attach_first_rank_vector,
        "pretrained_dominant_failure_mode": pt.get("dominant_failure_mode"),
        "finetuned_dominant_failure_mode": ft.get("dominant_failure_mode"),
        "bridge_delta": _bridge_delta(ft, pt),
    }
    result = {
        "gate": "g8_train_seed_probe",
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
        "probe_seeds": probe_seeds,
        "checkpoint_path": checkpoint_path,
        "checkpoint_step": checkpoint_step,
        "min_success_gain": min_success_gain,
        "min_finetuned_successes": min_finetuned_successes,
        "success_gain": success_gain,
        "ever_attach_eligible_fraction_gain": ever_attach_eligible_fraction_gain,
        "ever_attached_rate_gain": ever_attached_rate_gain,
        "stable_attach_gain": stable_attach_gain,
        "phase_locked_gain": phase_locked_gain,
        "max_drawer_fraction_gain": max_drawer_fraction_gain,
        "attached_seed_count": attached_seed_count,
        "trend_passed": trend_passed,
        "train_probe_claim_pass": trend_passed,
        "attach_bridge_pass": attach_bridge_pass,
        "attach_first_rank_vector": attach_first_rank_vector,
        "pretrained_dominant_failure_mode": pt.get("dominant_failure_mode"),
        "finetuned_dominant_failure_mode": ft.get("dominant_failure_mode"),
        "bridge_delta": _bridge_delta(ft, pt),
        "passed": bool(trend_passed or attach_bridge_pass),
        **summary,
        "timestamp": time.time(),
    }
    return summary_payload, result


def run(
    *,
    checkpoint_path_override: str | None = None,
    checkpoint_step: int | None = None,
    artifact_path_override: str | Path | None = None,
    summary_path_override: str | Path | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    plan = load_json(TINY_RETRAIN_PLAN_PATH, {})
    if not plan:
        raise SystemExit(f"Missing active tiny retrain plan: {TINY_RETRAIN_PLAN_PATH}")
    g8 = load_json(G8_ARTIFACT, {})
    if g8 and str(g8.get("run_instance_id") or "") != str(plan.get("run_instance_id") or ""):
        raise SystemExit("G8 train summary run_instance_id does not match active plan")
    checkpoint_path = checkpoint_path_override or g8.get("checkpoint_path")
    if not checkpoint_path:
        result = {
            "gate": "g8_train_seed_probe",
            "training_mode": "tiny_retrain_confirmation",
            "run_instance_id": plan.get("run_instance_id"),
            "plan_version": plan.get("plan_version"),
            "source_base_commit": plan.get("source_base_commit"),
            "working_head_commit": plan.get("working_head_commit"),
            "source_canonical_train_cell": _source_canonical_train_cell(plan),
            "active_train_state_mode": _active_train_state_mode(plan),
            "probe_seeds": plan.get("train_seeds", []),
            "min_success_gain": float(plan.get("train_probe_min_success_gain", 0.15)),
            "min_finetuned_successes": int(plan.get("train_probe_min_successes", 2)),
            "trend_passed": False,
            "attach_bridge_pass": False,
            "ever_attached_rate_gain": 0.0,
            "stable_attach_gain": 0.0,
            "phase_locked_gain": 0.0,
            "max_drawer_fraction_gain": 0.0,
            "attached_seed_count": 0,
            "passed": False,
            "error": "Missing fine-tuned checkpoint",
            "checkpoint_path": None,
            "checkpoint_step": checkpoint_step,
            "timestamp": time.time(),
        }
        if write_outputs:
            artifact_path = Path(artifact_path_override) if artifact_path_override else ARTIFACT
            write_json_atomic(artifact_path, result)
            print(json.dumps(result, indent=2))
        return result

    summary_payload, result = _build_outputs(plan, str(checkpoint_path), checkpoint_step)
    if write_outputs:
        summary_path = Path(summary_path_override) if summary_path_override else (_resolve_repo_path(plan["evaluation_dir"]) / "train_seed_probe.json")
        artifact_path = Path(artifact_path_override) if artifact_path_override else ARTIFACT
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(summary_path, summary_payload)
        result["summary_path"] = str(summary_path)
        write_json_atomic(artifact_path, result)
        print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    payload = run()
    raise SystemExit(0 if payload.get("passed") else 1)

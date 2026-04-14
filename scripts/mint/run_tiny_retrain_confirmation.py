#!/usr/bin/env python3
"""Bounded v8.2 tiny retrain bridge completion loop."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dataset_builder import (
    build_dataset_from_rollouts,
    validate_built_dataset_provenance,
    validate_rollout_for_canonical_training,
)
from mint_common import (
    ACTIVE_ACTION_CONTRACT_PATH,
    ARTIFACT_DIR,
    CAMPAIGN_DIR,
    DATASET_DIR,
    DATASET_REPO_ID,
    EVAL_DIR,
    PROJECT_ROOT,
    TINY_RETRAIN_DATASET_BUILD_PATH,
    TINY_RETRAIN_EVAL_DIR,
    TINY_RETRAIN_OUTPUT_DIR,
    TINY_RETRAIN_PLAN_PATH,
    TINY_RETRAIN_SUMMARY_PATH,
    load_json,
    tiny_retrain_backend_defaults,
    write_json_atomic,
)
from tiny_retrain_mainline import DEFAULT_ROLLOUT_SOURCE_DIR, MATERIALIZATION_ARTIFACT, TERMINAL_COMMIT, materialize_canonical_train_rollouts

ROUTE_DECISION_PATH = CAMPAIGN_DIR / "autopilot" / "route_decision.json"
FINAL_RUN_SUMMARY_PATH = CAMPAIGN_DIR / "autopilot" / "final_run_summary.json"
CYCLE_STATE_PATH = CAMPAIGN_DIR / "autopilot" / "cycle_state.json"
RCA1_PATH = ARTIFACT_DIR / "p2rca1_true_handle_measurement_audit.json"
RCA2_PATH = ARTIFACT_DIR / "p2rca2_local_affordance_visual_attack.json"
RCA3_PATH = ARTIFACT_DIR / "p2rca3_affordance_plus_transition_contract.json"
RCA4_PATH = ARTIFACT_DIR / "p2rca4_affordance_plus_transition_state.json"
RCA5_PATH = ARTIFACT_DIR / "p2rca5_frozen_matrix_screen.json"
RCA6_PATH = ARTIFACT_DIR / "p2rca6_best_cell_replicate.json"
RCA7_PATH = ARTIFACT_DIR / "p2rca7_tiny_retrain_if_eligible.json"
REJECTED_ROLLOUTS_PATH = ARTIFACT_DIR / "g8_canonical_dataset_rejected_rollouts.json"
G8_SUMMARY_PATH = ARTIFACT_DIR / "g8_train_summary.json"
G8_PROBE_PATH = ARTIFACT_DIR / "g8_train_seed_probe.json"
G9_SUMMARY_PATH = EVAL_DIR / "comparison_summary.json"
G9_REPORT_PATH = EVAL_DIR / "comparison_report.md"
EXECUTION_MEMO_PATH = EVAL_DIR / "tiny_retrain_execution_memo.md"

HONEST_EPISODES_PER_SEED = 12
HONEST_MIN_TRAIN_EPISODES = 48
HONEST_TRAIN_STEPS = 10000
HONEST_SAVE_FREQ = 2000
HONEST_PROBE_STEPS = [2000, 4000, 6000, 8000, 10000]
AMPLIFIED_EPISODES_PER_SEED = 16
AMPLIFIED_MIN_TRAIN_EPISODES = 64
AMPLIFIED_TRAIN_STEPS = 20000
AMPLIFIED_SAVE_FREQ = 4000
AMPLIFIED_PROBE_STEPS = [4000, 8000, 12000, 16000, 20000]


def _repo_rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _git(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=PROJECT_ROOT, text=True).strip()


def _active_state_mode_name(active_train_state_mode: str) -> str:
    return {
        "S0": "m0_proxy",
        "S1": "telemetry_candidate_v3_transition",
        "S2": "telemetry_candidate_v4_task_identity",
    }[str(active_train_state_mode)]


def _stage_dataset_root(source_cell: str, active_state: str, bridge_attempt: str) -> Path:
    return DATASET_DIR.parent / f"dataset_{source_cell}_{active_state.lower()}_{bridge_attempt.lower()}"


def _stage_train_output_dir(source_cell: str, active_state: str, bridge_attempt: str) -> Path:
    return TINY_RETRAIN_OUTPUT_DIR / source_cell / active_state.lower() / bridge_attempt.lower()


def _stage_evaluation_dir(source_cell: str, active_state: str, bridge_attempt: str) -> Path:
    return TINY_RETRAIN_EVAL_DIR / source_cell / active_state.lower() / bridge_attempt.lower()


def load_terminal_artifacts() -> dict[str, Any]:
    return {
        "route_decision": load_json(ROUTE_DECISION_PATH, {}),
        "final_run_summary": load_json(FINAL_RUN_SUMMARY_PATH, {}),
        "cycle_state": load_json(CYCLE_STATE_PATH, {}),
        "rca1": load_json(RCA1_PATH, {}),
        "rca2": load_json(RCA2_PATH, {}),
        "rca3": load_json(RCA3_PATH, {}),
        "rca4": load_json(RCA4_PATH, {}),
        "rca5": load_json(RCA5_PATH, {}),
        "rca6": load_json(RCA6_PATH, {}),
        "rca7": load_json(RCA7_PATH, {}),
    }


def assert_terminal_ready(artifacts: dict[str, Any]) -> dict[str, Any]:
    route = artifacts["route_decision"]
    rca1 = artifacts["rca1"]
    rca2 = artifacts["rca2"]
    rca5 = artifacts["rca5"]
    rca6 = artifacts["rca6"]
    rca7 = artifacts["rca7"]
    checks = [
        (route.get("scientific_terminal_state") == "TS_CANONICAL_POSITIVE_ESTABLISHED", "route_terminal_state_mismatch"),
        (route.get("route_next_branch") == "tiny_retrain_confirmation", "route_next_branch_mismatch"),
        (bool(rca7.get("tiny_retrain_permitted", False)) is True, "rca7_tiny_retrain_not_permitted"),
        (str(rca7.get("canonical_train_cell")) == "V1cT2S0", "rca7_canonical_train_cell_mismatch"),
        (str(rca7.get("best_train_state_mode")) == "S0", "rca7_best_train_state_mode_mismatch"),
        (bool(rca6.get("replicate_positive", False)) is True, "rca6_replicate_positive_false"),
        (str(rca5.get("best_transition_cell")) == "V1cT2S0", "rca5_best_transition_cell_mismatch"),
        (bool(rca2.get("v1c_carrier_canonical_ready", False)) is True, "rca2_v1c_not_canonical_ready"),
        (bool(rca1.get("measurement_truthful_available", False)) is True, "rca1_measurement_not_truthful"),
    ]
    failures = [reason for passed, reason in checks if not passed]
    if failures:
        raise SystemExit(f"Terminal tiny retrain prerequisites not satisfied: {failures}")
    return {
        "source_canonical_train_cell": "V1cT2S0",
        "source_best_train_state_mode": "S0",
        "best_transition_cell": "V1cT2S0",
        "selector_mode": str(rca5.get("selector_mode") or rca7.get("selector_mode") or "frozen_v5_pro"),
        "frozen_matrix_hash": str(rca5.get("frozen_matrix_hash") or rca7.get("frozen_matrix_hash") or ""),
        "ts_baseline_cell_id": str(rca5.get("ts_baseline_cell_id") or rca7.get("ts_baseline_cell_id") or "V1cT0S0"),
        "measurement_truth_tier": str(artifacts["rca1"].get("measurement_truth_tier") or "manifest_entity_verified"),
        "measurement_backend": str(artifacts["rca1"].get("measurement_backend") or "segmentation_render"),
        "measurement_verifier": str((artifacts["rca1"].get("measurement_report") or {}).get("measurement_verifier") or artifacts["rca1"].get("measurement_verifier") or "isolated_rgb_threshold"),
        "runtime_visible_handle_mapping_source": str(artifacts["rca1"].get("runtime_visible_handle_mapping_source") or "collision_geom"),
    }


def materialize_stage_plan(
    artifacts: dict[str, Any],
    *,
    active_train_state_mode: str,
    bridge_stage: str,
    bridge_attempt: str,
    episodes_per_seed: int,
    min_train_episodes: int,
    train_steps: int,
    save_freq: int,
    checkpoint_probe_steps: list[int],
) -> dict[str, Any]:
    ready = assert_terminal_ready(artifacts)
    branch = _git(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    head = _git(["git", "rev-parse", "HEAD"])
    backend_defaults = tiny_retrain_backend_defaults()
    source_cell = ready["source_canonical_train_cell"]
    active_state_mode_name = _active_state_mode_name(active_train_state_mode)
    dataset_root = _stage_dataset_root(source_cell, active_train_state_mode, bridge_attempt)
    train_output_dir = _stage_train_output_dir(source_cell, active_train_state_mode, bridge_attempt)
    evaluation_dir = _stage_evaluation_dir(source_cell, active_train_state_mode, bridge_attempt)
    plan = {
        "plan_version": "tiny_retrain_confirmation_v8_2",
        "source_branch": branch,
        "source_commit": head,
        "terminal_commit": TERMINAL_COMMIT,
        "scientific_terminal_state": "TS_CANONICAL_POSITIVE_ESTABLISHED",
        "route_next_branch": "tiny_retrain_confirmation",
        "source_canonical_train_cell": source_cell,
        "source_best_train_state_mode": ready["source_best_train_state_mode"],
        "canonical_train_cell": source_cell,
        "best_train_state_mode": ready["source_best_train_state_mode"],
        "best_transition_cell": ready["best_transition_cell"],
        "active_train_state_mode": active_train_state_mode,
        "active_state_mode_name": active_state_mode_name,
        "bridge_stage": bridge_stage,
        "bridge_attempt": bridge_attempt,
        "preferred_canonical_train_cell": "V1cT2S2",
        "ts_baseline_cell_id": ready["ts_baseline_cell_id"],
        "selector_mode": ready["selector_mode"],
        "frozen_matrix_hash": ready["frozen_matrix_hash"],
        "train_seeds": [1, 2, 3, 4, 5, 6, 7, 8],
        "heldout_seeds": [11, 12, 13, 14, 15],
        "active_action_contract_path": _repo_rel(ACTIVE_ACTION_CONTRACT_PATH),
        "dataset_root": _repo_rel(dataset_root),
        "dataset_repo_id": DATASET_REPO_ID,
        "train_output_dir": _repo_rel(train_output_dir),
        "evaluation_dir": _repo_rel(evaluation_dir),
        "measurement_truth_tier": ready["measurement_truth_tier"],
        "measurement_backend": ready["measurement_backend"],
        "measurement_verifier": ready["measurement_verifier"],
        "runtime_visible_handle_mapping_source": ready["runtime_visible_handle_mapping_source"],
        **backend_defaults,
        "evaluation_cell_id": source_cell,
        "evaluation_state_mode_name": active_state_mode_name,
        "train_steps": train_steps,
        "batch_size": 8,
        "save_freq": save_freq,
        "checkpoint_probe_steps": checkpoint_probe_steps,
        "train_probe_min_success_gain": 0.15,
        "train_probe_min_successes": 2,
        "episodes_per_seed": episodes_per_seed,
        "min_train_episodes": min_train_episodes,
        "teacher_truth_gate": "trace_window_v1",
        "authoritative_truth_field": "measurement_truthful_for_training",
    }
    write_json_atomic(TINY_RETRAIN_PLAN_PATH, plan)
    return plan


def _find_rollout_refs(payload: Any) -> set[Path]:
    refs: set[Path] = set()
    if isinstance(payload, dict):
        for value in payload.values():
            refs.update(_find_rollout_refs(value))
    elif isinstance(payload, list):
        for value in payload:
            refs.update(_find_rollout_refs(value))
    elif isinstance(payload, str) and payload.endswith(".npz"):
        path = Path(payload)
        refs.add(path if path.is_absolute() else PROJECT_ROOT / path)
    return refs


def _load_validated_rollout_paths(paths: list[Path], plan: dict[str, Any]) -> tuple[list[Path], list[dict[str, Any]]]:
    valid: list[Path] = []
    rejected: list[dict[str, Any]] = []
    for npz_path in sorted(paths):
        meta_path = npz_path.with_suffix(".json")
        if not meta_path.exists():
            rejected.append({"npz_path": str(npz_path), "reason": "missing_meta_json"})
            continue
        meta = json.loads(meta_path.read_text())
        ok, reason = validate_rollout_for_canonical_training(meta, plan)
        if ok:
            valid.append(npz_path)
        else:
            rejected.append(
                {
                    "npz_path": str(npz_path),
                    "reason": reason,
                    "seed": meta.get("seed"),
                    "measurement_truthful_for_training": meta.get("measurement_truthful_for_training"),
                    "teacher_truth_adjudication": meta.get("teacher_truth_adjudication"),
                    "teacher_truthful_window_frame_count": meta.get("teacher_truthful_window_frame_count"),
                    "bridge_in_truthful_window": meta.get("bridge_in_truthful_window"),
                    "final_snapshot_measurement_truthful": meta.get("final_snapshot_measurement_truthful"),
                    "final_snapshot_measurement_truth_tier": meta.get("final_snapshot_measurement_truth_tier"),
                }
            )
    return valid, rejected


def _rejected_rollout_summary(rejected: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, list[int]]]:
    counts = Counter(item.get("reason") for item in rejected)
    seeds_by_reason: dict[str, set[int]] = defaultdict(set)
    for item in rejected:
        seed = item.get("seed")
        if seed is not None:
            seeds_by_reason[str(item.get("reason"))].add(int(seed))
    return dict(counts), {reason: sorted(values) for reason, values in seeds_by_reason.items()}


def build_or_refresh_canonical_dataset(plan: dict[str, Any]) -> dict[str, Any]:
    sources_checked: list[str] = []
    candidate_paths: set[Path] = set()
    for path in [RCA5_PATH, RCA6_PATH, CAMPAIGN_DIR / "runtime" / "controller_cycles" / "cycle_20260414_134515" / "lane_results.json"]:
        payload = load_json(path, {})
        refs = _find_rollout_refs(payload)
        sources_checked.append(f"{path}:{len(refs)}refs")
        candidate_paths.update(refs)
    if DEFAULT_ROLLOUT_SOURCE_DIR.exists():
        sources_checked.append(f"{DEFAULT_ROLLOUT_SOURCE_DIR}:existing")
        candidate_paths.update(DEFAULT_ROLLOUT_SOURCE_DIR.glob("*.npz"))

    materialization = load_json(MATERIALIZATION_ARTIFACT, {})
    expected_attempted_rollouts = len(plan.get("train_seeds", [])) * int(plan.get("episodes_per_seed", HONEST_EPISODES_PER_SEED))
    min_train_episodes = int(plan.get("min_train_episodes", HONEST_MIN_TRAIN_EPISODES))
    valid_rollouts, rejected = _load_validated_rollout_paths(sorted(candidate_paths), plan)
    unique_valid_seeds = {
        int(json.loads(path.with_suffix(".json").read_text()).get("seed"))
        for path in valid_rollouts
        if path.with_suffix(".json").exists()
    }
    materialization_ready = bool(
        materialization.get("passed")
        and int(materialization.get("attempted_rollouts", 0)) == expected_attempted_rollouts
        and int(materialization.get("saved_rollouts", 0)) >= min_train_episodes
        and int(materialization.get("successful_seed_count", 0)) >= 6
        and str(materialization.get("active_train_state_mode")) == str(plan.get("active_train_state_mode"))
    )
    need_refresh = (
        len(valid_rollouts) < min_train_episodes
        or len(unique_valid_seeds) < 6
        or not materialization_ready
    )
    if need_refresh:
        materialization = materialize_canonical_train_rollouts(
            DEFAULT_ROLLOUT_SOURCE_DIR,
            expected=plan,
            force_rebuild=True,
        )
        valid_rollouts, rejected = _load_validated_rollout_paths(sorted(DEFAULT_ROLLOUT_SOURCE_DIR.glob("*.npz")), plan)
        unique_valid_seeds = {
            int(json.loads(path.with_suffix(".json").read_text()).get("seed"))
            for path in valid_rollouts
            if path.with_suffix(".json").exists()
        }
    rejected_by_reason, rejected_seeds_by_reason = _rejected_rollout_summary(rejected)
    raw_saved_rollout_count = int(materialization.get("saved_rollouts", 0) or 0)
    raw_successful_seed_count = int(materialization.get("successful_seed_count", 0) or 0)

    dataset_root = _resolve_repo_path(plan["dataset_root"])
    failure_common = {
        "dataset_root": str(dataset_root),
        "dataset_repo_id": str(plan["dataset_repo_id"]),
        "source_canonical_train_cell": plan.get("source_canonical_train_cell"),
        "source_best_train_state_mode": plan.get("source_best_train_state_mode"),
        "active_train_state_mode": plan.get("active_train_state_mode"),
        "active_state_mode_name": plan.get("active_state_mode_name"),
        "teacher_truth_gate": "trace_window_v1",
        "authoritative_truth_field": "measurement_truthful_for_training",
        "raw_saved_rollout_count": raw_saved_rollout_count,
        "raw_successful_seed_count": raw_successful_seed_count,
        "valid_rollout_count": len(valid_rollouts),
        "valid_seed_count": len(unique_valid_seeds),
        "valid_seeds": sorted(unique_valid_seeds),
        "rejected_rollout_count": len(rejected),
        "rejected_by_reason": rejected_by_reason,
        "rejected_seeds_by_reason": rejected_seeds_by_reason,
        "sources_checked": sources_checked,
        "materialization": materialization,
        "used_rollout_paths": [str(path) for path in valid_rollouts],
    }

    if not valid_rollouts:
        failure_report = {
            **failure_common,
            "dataset_valid": False,
            "error": "no_valid_canonical_rollouts_after_validation",
        }
        write_json_atomic(REJECTED_ROLLOUTS_PATH, {"rejected_rollouts": rejected})
        write_json_atomic(TINY_RETRAIN_DATASET_BUILD_PATH, failure_report)
        raise SystemExit("No valid canonical rollouts remain after validation")

    if raw_saved_rollout_count < min_train_episodes or raw_successful_seed_count < 6:
        failure_report = {
            **failure_common,
            "dataset_valid": False,
            "error": "insufficient_canonical_bridge_data",
        }
        write_json_atomic(REJECTED_ROLLOUTS_PATH, {"rejected_rollouts": rejected})
        write_json_atomic(TINY_RETRAIN_DATASET_BUILD_PATH, failure_report)
        raise SystemExit("Canonical rollout materialization failed readiness thresholds")

    if len(valid_rollouts) < min_train_episodes or len(unique_valid_seeds) < 6:
        failure_report = {
            **failure_common,
            "dataset_valid": False,
            "error": "insufficient_trace_truth_adjudicated_bridge_data_after_validation",
        }
        write_json_atomic(REJECTED_ROLLOUTS_PATH, {"rejected_rollouts": rejected})
        write_json_atomic(TINY_RETRAIN_DATASET_BUILD_PATH, failure_report)
        raise SystemExit("Canonical dataset build failed post-validation truth-window thresholds")

    payload = build_dataset_from_rollouts(valid_rollouts, dataset_root, str(plan["dataset_repo_id"]))
    provenance_report = validate_built_dataset_provenance(dataset_root, plan)
    build_report = {
        **provenance_report,
        **failure_common,
        "dataset_root": str(dataset_root),
        "dataset_repo_id": str(plan["dataset_repo_id"]),
        "dataset_valid": bool(provenance_report.get("dataset_valid", False)),
        "integrity": payload.get("integrity", {}),
    }
    write_json_atomic(REJECTED_ROLLOUTS_PATH, {"rejected_rollouts": rejected})
    write_json_atomic(TINY_RETRAIN_DATASET_BUILD_PATH, build_report)
    if not build_report.get("dataset_valid"):
        raise SystemExit(f"Canonical dataset build failed validation: {build_report.get('errors', [])}")
    return build_report


def launch_tiny_retrain(plan: dict[str, Any]) -> dict[str, Any]:
    from run_g8_mint_train import run as run_train

    run_train()
    return load_json(G8_SUMMARY_PATH, {})


def _checkpoint_probe_sweep_path(plan: dict[str, Any]) -> Path:
    return _resolve_repo_path(plan["evaluation_dir"]) / "checkpoint_probe_sweep.json"


def _checkpoint_path_for_step(train_output_dir: Path, step: int) -> Path | None:
    candidate = train_output_dir / "checkpoints" / f"{int(step):06d}" / "pretrained_model"
    return candidate if candidate.exists() else None


def _checkpoint_rank_key(result: dict[str, Any]) -> tuple[float, float, float, float, float, int]:
    ft = (result.get("summary") or {}).get("finetuned_mint", {})
    return (
        float(ft.get("success_rate", 0.0)),
        float(ft.get("grasp_success_rate", 0.0)),
        float(ft.get("ever_stable_attach_fraction", 0.0)),
        float(ft.get("attach_eligible_rate_mean", 0.0)),
        float(ft.get("phase_locked_rate_mean", 0.0)),
        int(result.get("checkpoint_step") or 0),
    )


def _selected_probe_summary_payload(selected: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval_max_steps": int(plan.get("evaluation_max_steps", 96)),
        "train_seeds": [int(seed) for seed in plan.get("train_seeds", [])],
        "episodes_per_seed": int(plan.get("evaluation_probe_episodes_per_seed", 3)),
        "evaluation_backend": plan.get("evaluation_backend"),
        "source_canonical_train_cell": plan.get("source_canonical_train_cell"),
        "source_best_train_state_mode": plan.get("source_best_train_state_mode"),
        "canonical_train_cell": plan.get("source_canonical_train_cell"),
        "best_train_state_mode": plan.get("source_best_train_state_mode"),
        "active_train_state_mode": plan.get("active_train_state_mode"),
        "active_state_mode_name": plan.get("active_state_mode_name"),
        "training_mode": "tiny_retrain_confirmation",
        "bridge_stage": plan.get("bridge_stage"),
        "bridge_attempt": plan.get("bridge_attempt"),
        "evaluation_env_family": plan.get("evaluation_env_family"),
        "evaluation_cell_id": plan.get("evaluation_cell_id"),
        "evaluation_interaction_mode": plan.get("evaluation_interaction_mode"),
        "probe_seeds": [int(seed) for seed in plan.get("train_seeds", [])],
        "checkpoint_path": selected.get("checkpoint_path"),
        "checkpoint_step": selected.get("checkpoint_step"),
        "min_success_gain": float(plan.get("train_probe_min_success_gain", 0.15)),
        "min_finetuned_successes": int(plan.get("train_probe_min_successes", 2)),
        "success_gain": float(selected.get("success_gain", 0.0)),
        "trend_passed": bool(selected.get("trend_passed", False)),
        "attach_bridge_pass": bool(selected.get("attach_bridge_pass", False)),
        "pretrained_dominant_failure_mode": selected.get("pretrained_dominant_failure_mode"),
        "finetuned_dominant_failure_mode": selected.get("finetuned_dominant_failure_mode"),
        "bridge_delta": dict(selected.get("bridge_delta") or {}),
        "summary": dict(selected.get("summary") or {}),
        "records": dict(selected.get("records") or {}),
        "selected_bridge_checkpoint": selected.get("checkpoint_path"),
        "selected_checkpoint_step": selected.get("checkpoint_step"),
    }


def run_train_probe(plan: dict[str, Any], train_summary: dict[str, Any]) -> dict[str, Any]:
    from run_g8_train_seed_probe import run as run_probe

    if not bool(train_summary.get("passed", False)):
        raise SystemExit("Cannot run train probe without a passed train summary")
    train_output_dir = _resolve_repo_path(plan["train_output_dir"])
    requested_steps = [int(step) for step in plan.get("checkpoint_probe_steps", [])]
    checkpoint_results: list[dict[str, Any]] = []
    missing_steps: list[int] = []
    for step in requested_steps:
        checkpoint_path = _checkpoint_path_for_step(train_output_dir, step)
        if checkpoint_path is None:
            missing_steps.append(int(step))
            continue
        checkpoint_results.append(
            run_probe(
                checkpoint_path_override=str(checkpoint_path),
                checkpoint_step=int(step),
                write_outputs=False,
            )
        )
    if missing_steps:
        raise SystemExit(f"Missing checkpoint after G8 train: {missing_steps}")
    if not checkpoint_results:
        raise SystemExit("No checkpoint probe results were produced")
    selected = max(checkpoint_results, key=_checkpoint_rank_key)
    selected_probe = dict(selected)
    selected_probe["selected_bridge_checkpoint"] = selected.get("checkpoint_path")
    selected_probe["selected_checkpoint_step"] = selected.get("checkpoint_step")
    selected_probe["summary_path"] = str(_resolve_repo_path(plan["evaluation_dir"]) / "train_seed_probe.json")
    sweep_payload = {
        "gate": "g8_checkpoint_probe_sweep",
        "training_mode": "tiny_retrain_confirmation",
        "source_canonical_train_cell": plan.get("source_canonical_train_cell"),
        "source_best_train_state_mode": plan.get("source_best_train_state_mode"),
        "active_train_state_mode": plan.get("active_train_state_mode"),
        "active_state_mode_name": plan.get("active_state_mode_name"),
        "bridge_stage": plan.get("bridge_stage"),
        "bridge_attempt": plan.get("bridge_attempt"),
        "evaluation_backend": plan.get("evaluation_backend"),
        "evaluation_env_family": plan.get("evaluation_env_family"),
        "evaluation_cell_id": plan.get("evaluation_cell_id"),
        "evaluation_interaction_mode": plan.get("evaluation_interaction_mode"),
        "requested_checkpoint_probe_steps": requested_steps,
        "checkpoint_results": checkpoint_results,
        "selected_bridge_checkpoint": selected.get("checkpoint_path"),
        "selected_checkpoint_step": selected.get("checkpoint_step"),
        "trend_passed": bool(selected.get("trend_passed", False)),
        "attach_bridge_pass": bool(selected.get("attach_bridge_pass", False)),
        "pretrained_dominant_failure_mode": selected.get("pretrained_dominant_failure_mode"),
        "finetuned_dominant_failure_mode": selected.get("finetuned_dominant_failure_mode"),
    }
    write_json_atomic(_checkpoint_probe_sweep_path(plan), sweep_payload)
    write_json_atomic(_resolve_repo_path(plan["evaluation_dir"]) / "train_seed_probe.json", _selected_probe_summary_payload(selected_probe, plan))
    write_json_atomic(G8_PROBE_PATH, selected_probe)
    return selected_probe


def run_heldout_eval(plan: dict[str, Any], train_summary: dict[str, Any], probe_summary: dict[str, Any]) -> dict[str, Any] | None:
    from run_g9_sim_eval import run as run_eval

    if not bool(probe_summary.get("trend_passed", False)):
        return None
    if not bool(train_summary.get("passed", False)):
        raise SystemExit("Cannot run held-out eval without a passed train summary")
    run_eval()
    return load_json(G9_SUMMARY_PATH, {})


def write_skipped_heldout_eval_summary(plan: dict[str, Any], probe_summary: dict[str, Any]) -> dict[str, Any]:
    attach_bridge_pass = bool(probe_summary.get("attach_bridge_pass", False))
    verdict = "attach_bridge_established_not_claim_supported" if attach_bridge_pass else "trainability_bridge_not_established"
    reason = (
        "Attach bridge improved under MuJoCo parity, but strict train-probe success did not pass; held-out eval skipped."
        if attach_bridge_pass
        else "Trainability bridge was not established under MuJoCo parity; held-out eval skipped."
    )
    summary = {
        "gate": "g9_sim_eval",
        "training_mode": "tiny_retrain_confirmation",
        "source_canonical_train_cell": plan.get("source_canonical_train_cell"),
        "source_best_train_state_mode": plan.get("source_best_train_state_mode"),
        "canonical_train_cell": plan.get("source_canonical_train_cell"),
        "best_train_state_mode": plan.get("source_best_train_state_mode"),
        "active_train_state_mode": plan.get("active_train_state_mode"),
        "active_state_mode_name": plan.get("active_state_mode_name"),
        "bridge_stage": plan.get("bridge_stage"),
        "bridge_attempt": plan.get("bridge_attempt"),
        "evaluation_backend": plan.get("evaluation_backend"),
        "evaluation_env_family": plan.get("evaluation_env_family"),
        "evaluation_cell_id": plan.get("evaluation_cell_id"),
        "evaluation_interaction_mode": plan.get("evaluation_interaction_mode"),
        "heldout_eval_run": False,
        "claim_supported": False,
        "verdict": verdict,
        "reason": reason,
        "trend_passed": bool(probe_summary.get("trend_passed", False)),
        "attach_bridge_pass": attach_bridge_pass,
        "selected_bridge_checkpoint": probe_summary.get("selected_bridge_checkpoint"),
        "selected_checkpoint_step": probe_summary.get("selected_checkpoint_step"),
        "held_out_seeds": plan.get("heldout_seeds", []),
        "episodes_per_seed": int(plan.get("evaluation_heldout_episodes_per_seed", 3)),
        "eval_max_steps": int(plan.get("evaluation_max_steps", 96)),
    }
    write_json_atomic(G9_SUMMARY_PATH, summary)
    G9_REPORT_PATH.write_text("# Held-out Eval Skipped\n\n" + reason + "\n")
    return summary


def write_execution_memo(summary: dict[str, Any]) -> None:
    history = summary.get("loop_history", [])
    lines = [
        "# Tiny Retrain Execution Memo",
        "",
        f"- source canonical cell: `{summary.get('source_canonical_train_cell')}`",
        f"- source best train state: `{summary.get('source_best_train_state_mode')}`",
        f"- final verdict: `{summary.get('final_verdict')}`",
        f"- scientific terminal state: `{summary.get('scientific_terminal_state')}`",
        f"- stages executed: `{summary.get('stages_executed')}`",
        "",
        "## Loop History",
    ]
    for item in history:
        lines.extend(
            [
                f"- `{item.get('bridge_stage')}` / `{item.get('bridge_attempt')}` / active `{item.get('active_train_state_mode')}`",
                f"  dataset_valid=`{item.get('dataset_valid')}` train_passed=`{item.get('train_passed')}` trend_passed=`{item.get('train_probe_passed')}` attach_bridge_pass=`{item.get('attach_bridge_pass')}` verdict=`{item.get('stage_verdict')}`",
            ]
        )
    EXECUTION_MEMO_PATH.write_text("\n".join(lines) + "\n")


def _build_stage_summary(
    plan: dict[str, Any],
    dataset_summary: dict[str, Any],
    train_summary: dict[str, Any],
    probe_summary: dict[str, Any],
    eval_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    train_passed = bool(train_summary.get("passed", False))
    train_probe_passed = bool(probe_summary.get("trend_passed", False))
    attach_bridge_pass = bool(probe_summary.get("attach_bridge_pass", False))
    heldout_eval_run = bool(eval_summary and eval_summary.get("heldout_eval_run", False))
    claim_supported = bool(heldout_eval_run and eval_summary and eval_summary.get("verdict") == "claim_supported")
    if claim_supported:
        stage_verdict = "claim_supported"
    elif attach_bridge_pass or train_probe_passed:
        stage_verdict = "attach_bridge_established_not_claim_supported"
    else:
        stage_verdict = "trainability_bridge_not_established"
    return {
        "bridge_stage": plan.get("bridge_stage"),
        "bridge_attempt": plan.get("bridge_attempt"),
        "source_branch": plan.get("source_branch"),
        "source_commit": plan.get("source_commit"),
        "source_canonical_train_cell": plan.get("source_canonical_train_cell"),
        "source_best_train_state_mode": plan.get("source_best_train_state_mode"),
        "active_train_state_mode": plan.get("active_train_state_mode"),
        "active_state_mode_name": plan.get("active_state_mode_name"),
        "dataset_root": plan.get("dataset_root"),
        "train_output_dir": plan.get("train_output_dir"),
        "evaluation_dir": plan.get("evaluation_dir"),
        "dataset_valid": bool(dataset_summary.get("dataset_valid", False)),
        "train_passed": train_passed,
        "train_probe_passed": train_probe_passed,
        "attach_bridge_pass": attach_bridge_pass,
        "heldout_eval_run": heldout_eval_run,
        "claim_supported": claim_supported,
        "stage_verdict": stage_verdict,
        "selected_bridge_checkpoint": probe_summary.get("selected_bridge_checkpoint") or train_summary.get("checkpoint_path"),
        "selected_checkpoint_step": probe_summary.get("selected_checkpoint_step"),
        "pretrained_dominant_failure_mode": probe_summary.get("pretrained_dominant_failure_mode"),
        "finetuned_dominant_failure_mode": probe_summary.get("finetuned_dominant_failure_mode"),
        "dataset_summary": dataset_summary,
        "train_summary": train_summary,
        "probe_summary": probe_summary,
        "eval_summary": eval_summary or {},
    }


def _run_stage(
    artifacts: dict[str, Any],
    *,
    active_train_state_mode: str,
    bridge_stage: str,
    bridge_attempt: str,
    episodes_per_seed: int,
    min_train_episodes: int,
    train_steps: int,
    save_freq: int,
    checkpoint_probe_steps: list[int],
) -> dict[str, Any]:
    plan = materialize_stage_plan(
        artifacts,
        active_train_state_mode=active_train_state_mode,
        bridge_stage=bridge_stage,
        bridge_attempt=bridge_attempt,
        episodes_per_seed=episodes_per_seed,
        min_train_episodes=min_train_episodes,
        train_steps=train_steps,
        save_freq=save_freq,
        checkpoint_probe_steps=checkpoint_probe_steps,
    )
    try:
        dataset_summary = build_or_refresh_canonical_dataset(plan)
    except SystemExit as exc:
        dataset_summary = load_json(TINY_RETRAIN_DATASET_BUILD_PATH, {})
        train_summary = {
            "gate": "g8_mint_train",
            "passed": False,
            "error": "dataset_prepare_failed",
            "reason": str(exc),
            "source_canonical_train_cell": plan.get("source_canonical_train_cell"),
            "source_best_train_state_mode": plan.get("source_best_train_state_mode"),
            "active_train_state_mode": plan.get("active_train_state_mode"),
            "active_state_mode_name": plan.get("active_state_mode_name"),
            "train_output_dir": plan.get("train_output_dir"),
        }
        probe_summary = {
            "trend_passed": False,
            "attach_bridge_pass": False,
            "selected_bridge_checkpoint": None,
            "selected_checkpoint_step": None,
            "pretrained_dominant_failure_mode": None,
            "finetuned_dominant_failure_mode": "dataset_prepare_failed",
        }
        eval_summary = write_skipped_heldout_eval_summary(plan, probe_summary)
        return _build_stage_summary(plan, dataset_summary, train_summary, probe_summary, eval_summary)

    train_summary = launch_tiny_retrain(plan)
    if not bool(train_summary.get("passed", False)):
        probe_summary: dict[str, Any] = {}
        eval_summary = write_skipped_heldout_eval_summary(
            plan,
            {
                "trend_passed": False,
                "attach_bridge_pass": False,
                "selected_bridge_checkpoint": train_summary.get("checkpoint_path"),
                "selected_checkpoint_step": None,
                "pretrained_dominant_failure_mode": None,
                "finetuned_dominant_failure_mode": "train_failed",
            },
        )
        return _build_stage_summary(plan, dataset_summary, train_summary, probe_summary, eval_summary)
    probe_summary = run_train_probe(plan, train_summary)
    if bool(probe_summary.get("trend_passed", False)):
        eval_summary = run_heldout_eval(plan, train_summary, probe_summary)
    else:
        eval_summary = write_skipped_heldout_eval_summary(plan, probe_summary)
    return _build_stage_summary(plan, dataset_summary, train_summary, probe_summary, eval_summary)


def _write_loop_summary(history: list[dict[str, Any]], final_verdict: str, scientific_terminal_state: str | None) -> dict[str, Any]:
    last = history[-1] if history else {}
    summary = {
        "confirmation_mode": "tiny_retrain_completion_loop_v8_2",
        "source_branch": last.get("source_branch"),
        "source_commit": last.get("source_commit"),
        "terminal_commit": TERMINAL_COMMIT,
        "source_canonical_train_cell": last.get("source_canonical_train_cell", "V1cT2S0"),
        "source_best_train_state_mode": last.get("source_best_train_state_mode", "S0"),
        "stages_executed": [item.get("bridge_stage") for item in history],
        "loop_history": history,
        "claim_supported": final_verdict == "claim_supported",
        "final_verdict": final_verdict,
        "scientific_terminal_state": scientific_terminal_state,
        "final_active_train_state_mode": last.get("active_train_state_mode"),
        "final_active_state_mode_name": last.get("active_state_mode_name"),
        "selected_bridge_checkpoint": last.get("selected_bridge_checkpoint"),
        "selected_checkpoint_step": last.get("selected_checkpoint_step"),
        "heldout_eval_run": bool(last.get("heldout_eval_run", False)),
        "attach_bridge_pass": bool(last.get("attach_bridge_pass", False)),
        "train_probe_passed": bool(last.get("train_probe_passed", False)),
        "evaluation_backend": "mujoco",
        "evaluation_env_family": "canonical_cell_runtime",
        "evaluation_cell_id": "V1cT2S0",
        "failure_stage": None if final_verdict == "claim_supported" else last.get("bridge_stage"),
        "completion_status": "loop_completed",
    }
    write_json_atomic(TINY_RETRAIN_SUMMARY_PATH, summary)
    write_execution_memo(summary)
    return summary


def run_completion_loop() -> dict[str, Any]:
    artifacts = load_terminal_artifacts()
    history: list[dict[str, Any]] = []

    s0_honest = _run_stage(
        artifacts,
        active_train_state_mode="S0",
        bridge_stage="s0_honest_bridge_run",
        bridge_attempt="honest",
        episodes_per_seed=HONEST_EPISODES_PER_SEED,
        min_train_episodes=HONEST_MIN_TRAIN_EPISODES,
        train_steps=HONEST_TRAIN_STEPS,
        save_freq=HONEST_SAVE_FREQ,
        checkpoint_probe_steps=HONEST_PROBE_STEPS,
    )
    history.append(s0_honest)
    if s0_honest["stage_verdict"] == "claim_supported":
        return _write_loop_summary(history, "claim_supported", None)

    if s0_honest["stage_verdict"] == "attach_bridge_established_not_claim_supported":
        s0_amp = _run_stage(
            artifacts,
            active_train_state_mode="S0",
            bridge_stage="s0_amplification_run",
            bridge_attempt="amplification",
            episodes_per_seed=AMPLIFIED_EPISODES_PER_SEED,
            min_train_episodes=AMPLIFIED_MIN_TRAIN_EPISODES,
            train_steps=AMPLIFIED_TRAIN_STEPS,
            save_freq=AMPLIFIED_SAVE_FREQ,
            checkpoint_probe_steps=AMPLIFIED_PROBE_STEPS,
        )
        history.append(s0_amp)
        if s0_amp["stage_verdict"] == "claim_supported":
            return _write_loop_summary(history, "claim_supported", None)

    s2_honest = _run_stage(
        artifacts,
        active_train_state_mode="S2",
        bridge_stage="s2_honest_bridge_run",
        bridge_attempt="honest",
        episodes_per_seed=HONEST_EPISODES_PER_SEED,
        min_train_episodes=HONEST_MIN_TRAIN_EPISODES,
        train_steps=HONEST_TRAIN_STEPS,
        save_freq=HONEST_SAVE_FREQ,
        checkpoint_probe_steps=HONEST_PROBE_STEPS,
    )
    history.append(s2_honest)
    if s2_honest["stage_verdict"] == "claim_supported":
        return _write_loop_summary(history, "claim_supported", None)

    if s2_honest["stage_verdict"] == "attach_bridge_established_not_claim_supported":
        s2_amp = _run_stage(
            artifacts,
            active_train_state_mode="S2",
            bridge_stage="s2_amplification_run",
            bridge_attempt="amplification",
            episodes_per_seed=AMPLIFIED_EPISODES_PER_SEED,
            min_train_episodes=AMPLIFIED_MIN_TRAIN_EPISODES,
            train_steps=AMPLIFIED_TRAIN_STEPS,
            save_freq=AMPLIFIED_SAVE_FREQ,
            checkpoint_probe_steps=AMPLIFIED_PROBE_STEPS,
        )
        history.append(s2_amp)
        if s2_amp["stage_verdict"] == "claim_supported":
            return _write_loop_summary(history, "claim_supported", None)

    return _write_loop_summary(history, "TINY_RETRAIN_NOT_ESTABLISHED_AFTER_S0_S2", "TINY_RETRAIN_NOT_ESTABLISHED_AFTER_S0_S2")


def _phase_prepare() -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts = load_terminal_artifacts()
    plan = materialize_stage_plan(
        artifacts,
        active_train_state_mode="S0",
        bridge_stage="truth_gate_alignment_s0_prepare",
        bridge_attempt="honest",
        episodes_per_seed=HONEST_EPISODES_PER_SEED,
        min_train_episodes=HONEST_MIN_TRAIN_EPISODES,
        train_steps=HONEST_TRAIN_STEPS,
        save_freq=HONEST_SAVE_FREQ,
        checkpoint_probe_steps=HONEST_PROBE_STEPS,
    )
    dataset_summary = build_or_refresh_canonical_dataset(plan)
    return plan, dataset_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "train", "probe", "eval", "finalize", "all"], default="all")
    args = parser.parse_args()

    if args.phase == "prepare":
        plan, dataset_summary = _phase_prepare()
        print(json.dumps({"phase": "prepare", "plan": plan, "dataset_summary": dataset_summary}, indent=2))
        return 0

    plan = load_json(TINY_RETRAIN_PLAN_PATH, {})
    dataset_summary = load_json(TINY_RETRAIN_DATASET_BUILD_PATH, {})
    train_summary = load_json(G8_SUMMARY_PATH, {})
    probe_summary = load_json(G8_PROBE_PATH, {})
    eval_summary = load_json(G9_SUMMARY_PATH, {})

    if args.phase == "train":
        if not plan or not dataset_summary:
            plan, dataset_summary = _phase_prepare()
        train_summary = launch_tiny_retrain(plan)
        print(json.dumps(train_summary, indent=2))
        return 0 if train_summary.get("passed") else 1

    if args.phase == "probe":
        if not train_summary:
            raise SystemExit("Missing g8_train_summary.json before probe")
        probe_summary = run_train_probe(plan, train_summary)
        print(json.dumps(probe_summary, indent=2))
        return 0 if (probe_summary.get("trend_passed") or probe_summary.get("attach_bridge_pass")) else 1

    if args.phase == "eval":
        if not probe_summary or not bool(probe_summary.get("trend_passed", False)):
            raise SystemExit("Train probe must pass before held-out eval")
        eval_summary = run_heldout_eval(plan, train_summary, probe_summary)
        print(json.dumps(eval_summary or {}, indent=2))
        return 0 if eval_summary else 1

    if args.phase == "finalize":
        stage_summary = _build_stage_summary(plan, dataset_summary, train_summary, probe_summary, eval_summary if eval_summary else None)
        summary = _write_loop_summary([stage_summary], stage_summary["stage_verdict"] if stage_summary["stage_verdict"] == "claim_supported" else "TINY_RETRAIN_NOT_ESTABLISHED_AFTER_S0_S2", None if stage_summary["stage_verdict"] == "claim_supported" else "TINY_RETRAIN_NOT_ESTABLISHED_AFTER_S0_S2")
        print(json.dumps(summary, indent=2))
        return 0 if summary.get("final_verdict") == "claim_supported" else 1

    summary = run_completion_loop()
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("final_verdict") == "claim_supported" else 1


if __name__ == "__main__":
    raise SystemExit(main())

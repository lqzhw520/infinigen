#!/usr/bin/env python3
"""v11 support-family diversification gate runner."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dataset_builder import (
    build_dataset_from_rollouts,
    validate_built_dataset_provenance,
    validate_rollout_for_canonical_training,
    validate_rollout_for_learning_support_training,
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
from strict_success import evaluate_strict_success
from tiny_retrain_mainline import (
    DEFAULT_ROLLOUT_SOURCE_DIR,
    EFFECTIVE_SUPPORT_SIGNATURE_VERSION,
    LEARNING_SUPPORT_FINGERPRINT_VERSION,
    LEARNING_SUPPORT_MATERIALIZATION_ARTIFACT,
    LEARNING_SUPPORT_ROLLOUT_SOURCE_DIR,
    MATERIALIZATION_ARTIFACT,
    STRICT_UTILITY_VERSION,
    TEACHER_FAMILY_GRID_VERSION,
    TEACHER_FINGERPRINT_VERSION,
    TERMINAL_COMMIT,
    TRUTH_CONTRACT_PATH,
    TRUTH_UTILITY_VERSION,
    _truth_contract_hash,
    _truth_contract_payload,
    _truth_contract_required_str,
    materialize_canonical_train_rollouts,
)
from tiny_retrain_gate_utils import (
    DOCS_UPDATE_INTENT_PATH,
    GATES_DIR,
    HARNESS_STATE_PATH,
    PUBLICATION_STATE_PATH,
    SPEC_DOC_PATH,
    SOVEREIGN_SNAPSHOT_PATH,
    ACCEPTANCE_CONTRACT_PATH,
    docs_lock_consistent,
    gate_scope_from_g4,
    load_gate,
    readiness_failed_clauses,
    sha256_file,
    write_docs_lock_manifest,
    write_docs_update_intent,
    write_gate,
    write_harness_state,
    write_publication_state,
    write_sovereign_snapshot,
)
from run_v84_learning_support_family_audit import run as run_family_support_audit

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
G1_SUPPORT_FAMILY_AUDIT_PATH = (
    ARTIFACT_DIR / "v11_learning_support_family_forensic_audit.json"
)
V10_GATES_DIR = CAMPAIGN_DIR / "autopilot" / "gates"
V10_G3_GATE_PATH = V10_GATES_DIR / "G3_dataset_integrity.json"
V10_G4_GATE_PATH = V10_GATES_DIR / "G4_readiness_interpretation.json"
V10_G8_GATE_PATH = V10_GATES_DIR / "G8_publication_gate.json"
G6_TEACHER_READINESS_CONTRACT_PATH = ARTIFACT_DIR / "g6_teacher_readiness_contract.json"
G6_TRAINABILITY_SUPPORT_CONTRACT_PATH = (
    ARTIFACT_DIR / "g6_trainability_support_contract.json"
)
G6_STRICT_SEMANTICS_ALIGNMENT_PATH = ARTIFACT_DIR / "g6_strict_semantics_alignment.json"
EXECUTION_ANCHOR_PATH = ARTIFACT_DIR / "v8_3_execution_anchor.json"

PLAN_VERSION = "tiny_retrain_confirmation_v11"
SOURCE_BASE_COMMIT = "dfa0ef0f0acd9443dc010d019d25ad7eb78b40d6"

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


def _generate_run_instance_id(head: str) -> str:
    env_override = str(os.environ.get("MINT_RUN_INSTANCE_ID") or "").strip()
    if env_override:
        return env_override
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"v11_{stamp}_{str(head)[:8]}_{uuid.uuid4().hex[:8]}"


def _load_execution_anchor() -> dict[str, Any]:
    return load_json(EXECUTION_ANCHOR_PATH, {})


def _resolve_run_instance_id(branch: str, head: str) -> str:
    anchor = _load_execution_anchor()
    if (
        anchor.get("plan_version") == PLAN_VERSION
        and anchor.get("source_base_commit") == SOURCE_BASE_COMMIT
        and anchor.get("working_head_commit") == head
        and anchor.get("source_branch") == branch
        and str(anchor.get("run_instance_id") or "").strip()
    ):
        return str(anchor["run_instance_id"])
    return _generate_run_instance_id(head)


def _write_execution_anchor(plan: dict[str, Any]) -> dict[str, Any]:
    anchor = {
        "anchor_version": "v8_3_execution_anchor_v1",
        "remote_host": "ssh infinigen-cursor",
        "repo_root": str(PROJECT_ROOT),
        "plan_version": plan.get("plan_version"),
        "run_instance_id": plan.get("run_instance_id"),
        "source_branch": plan.get("source_branch"),
        "working_head_commit": plan.get("working_head_commit"),
        "source_base_commit": plan.get("source_base_commit"),
        "source_canonical_train_cell": plan.get("source_canonical_train_cell"),
        "source_best_train_state_mode": plan.get("source_best_train_state_mode"),
        "backend": plan.get("evaluation_backend"),
        "interaction_mode": plan.get("evaluation_interaction_mode"),
        "teacher_truth_gate": plan.get("teacher_truth_gate"),
        "strict_utility_version": plan.get("strict_utility_version"),
        "truth_utility_version": plan.get("truth_utility_version"),
        "teacher_fingerprint_version": plan.get("teacher_fingerprint_version"),
        "execution_model": "remote_a800_authoritative",
        "codebase_analysis_mode": "direct_remote_repo_audit_fallback",
    }
    write_json_atomic(EXECUTION_ANCHOR_PATH, anchor)
    return anchor


def _artifact_matches_plan(payload: dict[str, Any], plan: dict[str, Any]) -> bool:
    if not payload:
        return False
    checks = [
        str(payload.get("run_instance_id") or "")
        == str(plan.get("run_instance_id") or ""),
        str(payload.get("plan_version") or "") == str(plan.get("plan_version") or ""),
        str(payload.get("source_base_commit") or "")
        == str(plan.get("source_base_commit") or ""),
        str(payload.get("teacher_truth_gate") or "")
        == str(plan.get("teacher_truth_gate") or ""),
        str(payload.get("active_train_state_mode") or "")
        == str(plan.get("active_train_state_mode") or ""),
        str(payload.get("source_canonical_train_cell") or "")
        == str(plan.get("source_canonical_train_cell") or ""),
        str(payload.get("support_family_repair_mode") or "")
        == str(plan.get("support_family_repair_mode") or ""),
        str(payload.get("teacher_family_grid_version") or "")
        == str(plan.get("teacher_family_grid_version") or ""),
    ]
    return all(checks)


def _strict_alignment_entry(npz_path: Path) -> dict[str, Any]:
    meta_path = npz_path.with_suffix(".json")
    meta = load_json(meta_path, {})
    if not meta_path.exists():
        return {
            "npz_path": str(npz_path),
            "aligned": False,
            "reason": "missing_meta_json",
        }
    with np.load(npz_path, allow_pickle=True) as data:
        drawer_key = (
            "next_drawer_fractions"
            if "next_drawer_fractions" in data
            else "absolute_drawer_fraction"
        )
        drawer_trace = np.asarray(data[drawer_key], dtype=np.float32).reshape(-1)
        attached_trace = (
            np.asarray(data["attached_trace"], dtype=bool).reshape(-1)
            if "attached_trace" in data
            else np.zeros_like(drawer_trace, dtype=bool)
        )
    eval_metrics = evaluate_strict_success(drawer_trace, attached_trace)
    teacher_metrics = dict(meta.get("strict_metrics") or {})
    aligned = bool(
        teacher_metrics
        and bool(teacher_metrics.get("strict_success", False))
        == bool(eval_metrics.get("strict_success", False))
        and int(teacher_metrics.get("attach_persistence", 0) or 0)
        == int(eval_metrics.get("attach_persistence", 0) or 0)
        and abs(
            float(teacher_metrics.get("post_attach_drawer_delta", 0.0) or 0.0)
            - float(eval_metrics.get("post_attach_drawer_delta", 0.0) or 0.0)
        )
        <= 1e-6
        and abs(
            float(teacher_metrics.get("max_drawer_fraction", 0.0) or 0.0)
            - float(eval_metrics.get("max_drawer_fraction", 0.0) or 0.0)
        )
        <= 1e-6
    )
    return {
        "npz_path": str(npz_path),
        "meta_path": str(meta_path),
        "drawer_trace_source": drawer_key,
        "teacher_replay": {
            "max_post_step_drawer_fraction": float(
                teacher_metrics.get("max_drawer_fraction", 0.0) or 0.0
            ),
            "attach_persistence": int(
                teacher_metrics.get("attach_persistence", 0) or 0
            ),
            "post_attach_drawer_delta": float(
                teacher_metrics.get("post_attach_drawer_delta", 0.0) or 0.0
            ),
            "strict_success": bool(teacher_metrics.get("strict_success", False)),
        },
        "eval_replay": {
            "max_post_step_drawer_fraction": float(
                eval_metrics.get("max_drawer_fraction", 0.0) or 0.0
            ),
            "attach_persistence": int(eval_metrics.get("attach_persistence", 0) or 0),
            "post_attach_drawer_delta": float(
                eval_metrics.get("post_attach_drawer_delta", 0.0) or 0.0
            ),
            "strict_success": bool(eval_metrics.get("strict_success", False)),
        },
        "aligned": aligned,
        "strict_success_version": meta.get("strict_success_version"),
    }


def _write_strict_semantics_alignment(
    paths: list[Path], plan: dict[str, Any]
) -> dict[str, Any]:
    entries = [_strict_alignment_entry(path) for path in sorted(paths)]
    report = {
        "gate": "g6_strict_semantics_alignment",
        "run_instance_id": plan.get("run_instance_id"),
        "plan_version": plan.get("plan_version"),
        "source_base_commit": plan.get("source_base_commit"),
        "source_canonical_train_cell": plan.get("source_canonical_train_cell"),
        "source_best_train_state_mode": plan.get("source_best_train_state_mode"),
        "active_train_state_mode": plan.get("active_train_state_mode"),
        "strict_utility_version": plan.get("strict_utility_version"),
        "paired_rollout_count": len(entries),
        "aligned_rollout_count": int(
            sum(1 for item in entries if bool(item.get("aligned", False)))
        ),
        "passed": bool(entries)
        and all(bool(item.get("aligned", False)) for item in entries),
        "entries": entries,
    }
    write_json_atomic(G6_STRICT_SEMANTICS_ALIGNMENT_PATH, report)
    return report


def _active_state_mode_name(active_train_state_mode: str) -> str:
    return {
        "S0": "m0_proxy",
        "S1": "telemetry_candidate_v3_transition",
        "S2": "telemetry_candidate_v4_task_identity",
    }[str(active_train_state_mode)]


def _stage_dataset_root(
    source_cell: str, active_state: str, bridge_attempt: str
) -> Path:
    return (
        DATASET_DIR.parent
        / f"dataset_{source_cell}_{active_state.lower()}_{bridge_attempt.lower()}"
    )


def _stage_train_output_dir(
    source_cell: str, active_state: str, bridge_attempt: str
) -> Path:
    return (
        TINY_RETRAIN_OUTPUT_DIR
        / source_cell
        / active_state.lower()
        / bridge_attempt.lower()
    )


def _stage_evaluation_dir(
    source_cell: str, active_state: str, bridge_attempt: str
) -> Path:
    return (
        TINY_RETRAIN_EVAL_DIR
        / source_cell
        / active_state.lower()
        / bridge_attempt.lower()
    )


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
        (
            route.get("scientific_terminal_state")
            == "TS_CANONICAL_POSITIVE_ESTABLISHED",
            "route_terminal_state_mismatch",
        ),
        (
            route.get("route_next_branch") == "tiny_retrain_confirmation",
            "route_next_branch_mismatch",
        ),
        (
            bool(rca7.get("tiny_retrain_permitted", False)) is True,
            "rca7_tiny_retrain_not_permitted",
        ),
        (
            str(rca7.get("canonical_train_cell")) == "V1cT2S0",
            "rca7_canonical_train_cell_mismatch",
        ),
        (
            str(rca7.get("best_train_state_mode")) == "S0",
            "rca7_best_train_state_mode_mismatch",
        ),
        (
            bool(rca6.get("replicate_positive", False)) is True,
            "rca6_replicate_positive_false",
        ),
        (
            str(rca5.get("best_transition_cell")) == "V1cT2S0",
            "rca5_best_transition_cell_mismatch",
        ),
        (
            bool(rca2.get("v1c_carrier_canonical_ready", False)) is True,
            "rca2_v1c_not_canonical_ready",
        ),
        (
            bool(rca1.get("measurement_truthful_available", False)) is True,
            "rca1_measurement_not_truthful",
        ),
    ]
    failures = [reason for passed, reason in checks if not passed]
    if failures:
        raise SystemExit(
            f"Terminal tiny retrain prerequisites not satisfied: {failures}"
        )
    return {
        "source_canonical_train_cell": "V1cT2S0",
        "source_best_train_state_mode": "S0",
        "best_transition_cell": "V1cT2S0",
        "selector_mode": str(
            rca5.get("selector_mode") or rca7.get("selector_mode") or "frozen_v5_pro"
        ),
        "frozen_matrix_hash": str(
            rca5.get("frozen_matrix_hash") or rca7.get("frozen_matrix_hash") or ""
        ),
        "ts_baseline_cell_id": str(
            rca5.get("ts_baseline_cell_id")
            or rca7.get("ts_baseline_cell_id")
            or "V1cT0S0"
        ),
        "measurement_truth_tier": str(
            artifacts["rca1"].get("measurement_truth_tier")
            or "manifest_entity_verified"
        ),
        "measurement_backend": str(
            artifacts["rca1"].get("measurement_backend") or "segmentation_render"
        ),
        "measurement_verifier": str(
            (artifacts["rca1"].get("measurement_report") or {}).get(
                "measurement_verifier"
            )
            or artifacts["rca1"].get("measurement_verifier")
            or "isolated_rgb_threshold"
        ),
        "runtime_visible_handle_mapping_source": str(
            artifacts["rca1"].get("runtime_visible_handle_mapping_source")
            or "collision_geom"
        ),
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
    run_instance_id = _resolve_run_instance_id(branch, head)
    backend_defaults = tiny_retrain_backend_defaults()
    source_cell = ready["source_canonical_train_cell"]
    active_state_mode_name = _active_state_mode_name(active_train_state_mode)
    dataset_root = _stage_dataset_root(
        source_cell, active_train_state_mode, bridge_attempt
    )
    train_output_dir = _stage_train_output_dir(
        source_cell, active_train_state_mode, bridge_attempt
    )
    evaluation_dir = _stage_evaluation_dir(
        source_cell, active_train_state_mode, bridge_attempt
    )
    plan = {
        "plan_version": PLAN_VERSION,
        "run_instance_id": run_instance_id,
        "source_branch": branch,
        "source_commit": head,
        "working_head_commit": head,
        "source_base_commit": SOURCE_BASE_COMMIT,
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
        "runtime_visible_handle_mapping_source": ready[
            "runtime_visible_handle_mapping_source"
        ],
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
        "teacher_controller_mode": "interaction_frame_hybrid",
        "teacher_controller_max_steps": 144,
        "teacher_pull_open_fraction": 0.92,
        "teacher_truth_gate": _truth_contract_required_str(
            _truth_contract_payload(), "teacher_truth_predicate"
        ),
        "dataset_selection_mode": "diagnostic_learning_support",
        "truth_contract_path": str(TRUTH_CONTRACT_PATH),
        "truth_contract_hash": _truth_contract_hash(),
        "acceptance_contract_path": str(ACCEPTANCE_CONTRACT_PATH),
        "acceptance_contract_hash": sha256_file(ACCEPTANCE_CONTRACT_PATH),
        "authoritative_truth_field": "measurement_truthful_for_learning_support",
        "strict_utility_version": STRICT_UTILITY_VERSION,
        "truth_utility_version": TRUTH_UTILITY_VERSION,
        "teacher_fingerprint_version": TEACHER_FINGERPRINT_VERSION,
        "learning_support_fingerprint_version": "v10_learning_support_fingerprint_v1",
        "effective_support_signature_version": EFFECTIVE_SUPPORT_SIGNATURE_VERSION,
        "support_family_repair_mode": "pending_audit",
        "teacher_family_grid_version": TEACHER_FAMILY_GRID_VERSION,
        "execution_scope": "v11_support_family_pending_prepare",
        "diagnostic_only": True,
        "claim_bearing": False,
        "publication_scope": "pending",
        "result_scope": "pending",
    }
    _write_execution_anchor(plan)
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


def _load_validated_rollout_paths(
    paths: list[Path], plan: dict[str, Any]
) -> tuple[list[Path], list[dict[str, Any]]]:
    dataset_selection_mode = str(plan.get("dataset_selection_mode") or "claim_canonical")
    validator = (
        validate_rollout_for_learning_support_training
        if dataset_selection_mode == "diagnostic_learning_support"
        else validate_rollout_for_canonical_training
    )
    valid: list[Path] = []
    rejected: list[dict[str, Any]] = []
    for npz_path in sorted(paths):
        meta_path = npz_path.with_suffix(".json")
        if not meta_path.exists():
            rejected.append({"npz_path": str(npz_path), "reason": "missing_meta_json"})
            continue
        meta = json.loads(meta_path.read_text())
        ok, reason = validator(meta, plan)
        if ok:
            valid.append(npz_path)
        else:
            rejected.append(
                {
                    "npz_path": str(npz_path),
                    "reason": reason,
                    "seed": meta.get("seed"),
                    "run_instance_id": meta.get("run_instance_id"),
                    "plan_version": meta.get("plan_version"),
                    "teacher_episode_class": meta.get("teacher_episode_class"),
                    "teacher_fingerprint": meta.get("teacher_fingerprint"),
                    "learning_support_teacher_class": meta.get(
                        "learning_support_teacher_class"
                    ),
                    "learning_support_fingerprint": meta.get(
                        "learning_support_fingerprint"
                    ),
                    "measurement_truthful_for_training": meta.get(
                        "measurement_truthful_for_training"
                    ),
                    "measurement_truthful_for_learning_support": meta.get(
                        "measurement_truthful_for_learning_support"
                    ),
                    "teacher_truth_adjudication": meta.get(
                        "teacher_truth_adjudication"
                    ),
                    "learning_support_truth_adjudication": meta.get(
                        "learning_support_truth_adjudication"
                    ),
                    "teacher_truthful_window_frame_count": meta.get(
                        "teacher_truthful_window_frame_count"
                    ),
                    "truthful_window_ratio": meta.get("truthful_window_ratio"),
                    "truthful_window_longest_interior_gap": meta.get(
                        "truthful_window_longest_interior_gap"
                    ),
                    "truthful_window_tail_truthful_count_last6": meta.get(
                        "truthful_window_tail_truthful_count_last6"
                    ),
                    "bridge_in_truthful_window": meta.get("bridge_in_truthful_window"),
                    "learning_support_window_len": meta.get(
                        "learning_support_window_len"
                    ),
                    "learning_support_prebridge_frame_count": meta.get(
                        "learning_support_prebridge_frame_count"
                    ),
                    "learning_support_contains_prebridge": meta.get(
                        "learning_support_contains_prebridge"
                    ),
                    "learning_support_contains_bridge": meta.get(
                        "learning_support_contains_bridge"
                    ),
                    "final_snapshot_measurement_truthful": meta.get(
                        "final_snapshot_measurement_truthful"
                    ),
                    "final_snapshot_measurement_truth_tier": meta.get(
                        "final_snapshot_measurement_truth_tier"
                    ),
                }
            )
    return valid, rejected


def _rejected_rollout_summary(
    rejected: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, list[int]]]:
    counts = Counter(item.get("reason") for item in rejected)
    seeds_by_reason: dict[str, set[int]] = defaultdict(set)
    for item in rejected:
        seed = item.get("seed")
        if seed is not None:
            seeds_by_reason[str(item.get("reason"))].add(int(seed))
    return dict(counts), {
        reason: sorted(values) for reason, values in seeds_by_reason.items()
    }


def _teacher_readiness_contract(
    dataset_report: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    accepted_seed_coverage = {
        int(x) for x in dataset_report.get("accepted_seed_coverage", [])
    }
    near_strict_seed_coverage = {
        int(x) for x in dataset_report.get("near_strict_seed_coverage", [])
    }
    strict_alignment = load_json(G6_STRICT_SEMANTICS_ALIGNMENT_PATH, {})
    clauses = {
        "source_canonical_train_cell_frozen": str(
            plan.get("source_canonical_train_cell")
        )
        == "V1cT2S0",
        "truthful_accepted_train_seeds_ge_6": len(accepted_seed_coverage) >= 6,
        "accepted_unique_teacher_families_ge_18": int(
            dataset_report.get("accepted_unique_teacher_family_count", 0) or 0
        )
        >= 18,
        "strict_unique_teacher_families_ge_8": int(
            dataset_report.get("strict_unique_teacher_family_count", 0) or 0
        )
        >= 8,
        "near_strict_unique_teacher_families_ge_6": int(
            dataset_report.get("near_strict_unique_teacher_family_count", 0) or 0
        )
        >= 6,
        "seed2_has_deterministic_near_strict": 2 in near_strict_seed_coverage
        or 2 in accepted_seed_coverage,
        "seed4_has_deterministic_near_strict": 4 in near_strict_seed_coverage
        or 4 in accepted_seed_coverage,
        "seed3_has_accepted_truthful_bridge": 3 in accepted_seed_coverage,
        "strict_semantics_aligned": bool(strict_alignment.get("passed", False)),
        "duplicate_inflation_not_authoritative": int(
            dataset_report.get("accepted_unique_teacher_family_count", 0) or 0
        )
        <= int(dataset_report.get("used_rollout_count", 0) or 0),
    }
    failed_clauses = [name for name, passed in clauses.items() if not bool(passed)]
    report = {
        "gate": "g6_teacher_readiness_contract",
        "readiness_scope": "claim_readiness",
        "run_instance_id": plan.get("run_instance_id"),
        "plan_version": plan.get("plan_version"),
        "source_base_commit": plan.get("source_base_commit"),
        "source_canonical_train_cell": plan.get("source_canonical_train_cell"),
        "source_best_train_state_mode": plan.get("source_best_train_state_mode"),
        "active_train_state_mode": plan.get("active_train_state_mode"),
        "accepted_seed_coverage": sorted(accepted_seed_coverage),
        "near_strict_seed_coverage": sorted(near_strict_seed_coverage),
        "accepted_unique_teacher_family_count": int(
            dataset_report.get("accepted_unique_teacher_family_count", 0) or 0
        ),
        "strict_unique_teacher_family_count": int(
            dataset_report.get("strict_unique_teacher_family_count", 0) or 0
        ),
        "near_strict_unique_teacher_family_count": int(
            dataset_report.get("near_strict_unique_teacher_family_count", 0) or 0
        ),
        "teacher_episode_class_counts": dict(
            dataset_report.get("teacher_episode_class_counts") or {}
        ),
        "strict_semantics_alignment": strict_alignment,
        "truth_contract_path": str(TRUTH_CONTRACT_PATH),
        "truth_contract_hash": _truth_contract_hash(),
        "clauses": clauses,
        "failed_clauses": failed_clauses,
        "passed": not failed_clauses,
    }
    write_json_atomic(G6_TEACHER_READINESS_CONTRACT_PATH, report)
    return report


def _trainability_support_contract(
    dataset_report: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    learning_support_seed_coverage = {
        int(x) for x in dataset_report.get("learning_support_seed_coverage", [])
    }
    attach_eligible_seed_coverage = {
        int(x) for x in dataset_report.get("attach_eligible_seed_coverage", [])
    }
    learning_support_family_count_by_seed = dict(
        dataset_report.get("learning_support_family_count_by_seed") or {}
    )
    clauses = {
        "learning_support_seed_coverage_all_train_seeds": learning_support_seed_coverage
        == {int(x) for x in plan.get("train_seeds", [])},
        "seed2_learning_support_families_ge_2": int(
            learning_support_family_count_by_seed.get("2", 0) or 0
        )
        >= 2,
        "seed4_learning_support_families_ge_2": int(
            learning_support_family_count_by_seed.get("4", 0) or 0
        )
        >= 2,
        "learning_support_unique_teacher_families_ge_12": int(
            dataset_report.get("learning_support_unique_teacher_family_count", 0) or 0
        )
        >= 12,
        "learning_support_prebridge_frame_p50_ge_16": int(
            dataset_report.get("learning_support_prebridge_frame_p50", 0) or 0
        )
        >= 16,
        "attach_eligible_seed_coverage_ge_6": len(attach_eligible_seed_coverage) >= 6,
    }
    failed_clauses = [name for name, passed in clauses.items() if not bool(passed)]
    rca_classes: list[str] = []
    if "learning_support_seed_coverage_all_train_seeds" in failed_clauses:
        rca_classes.append("learning_support_seed_gap")
    if any(
        name in failed_clauses
        for name in ("seed2_learning_support_families_ge_2", "seed4_learning_support_families_ge_2")
    ):
        rca_classes.append("hard_seed_support_gap")
    if "learning_support_unique_teacher_families_ge_12" in failed_clauses:
        rca_classes.append("support_family_collapse")
    if "learning_support_prebridge_frame_p50_ge_16" in failed_clauses:
        rca_classes.append("prebridge_window_too_short")
    if "attach_eligible_seed_coverage_ge_6" in failed_clauses:
        rca_classes.append("attach_eligible_support_gap")
    report = {
        "gate": "g6_trainability_support_contract",
        "readiness_scope": "trainability_support",
        "run_instance_id": plan.get("run_instance_id"),
        "plan_version": plan.get("plan_version"),
        "source_base_commit": plan.get("source_base_commit"),
        "source_canonical_train_cell": plan.get("source_canonical_train_cell"),
        "source_best_train_state_mode": plan.get("source_best_train_state_mode"),
        "active_train_state_mode": plan.get("active_train_state_mode"),
        "dataset_selection_mode": plan.get("dataset_selection_mode"),
        "learning_support_seed_coverage": sorted(learning_support_seed_coverage),
        "attach_eligible_seed_coverage": sorted(attach_eligible_seed_coverage),
        "learning_support_unique_teacher_family_count": int(
            dataset_report.get("learning_support_unique_teacher_family_count", 0) or 0
        ),
        "learning_support_family_count_by_seed": learning_support_family_count_by_seed,
        "learning_support_teacher_class_counts": dict(
            dataset_report.get("learning_support_teacher_class_counts") or {}
        ),
        "learning_support_prebridge_frame_p50": int(
            dataset_report.get("learning_support_prebridge_frame_p50", 0) or 0
        ),
        "learning_support_prebridge_frame_p90": int(
            dataset_report.get("learning_support_prebridge_frame_p90", 0) or 0
        ),
        "learning_support_window_len_p50": int(
            dataset_report.get("learning_support_window_len_p50", 0) or 0
        ),
        "learning_support_window_len_p90": int(
            dataset_report.get("learning_support_window_len_p90", 0) or 0
        ),
        "truth_contract_path": str(TRUTH_CONTRACT_PATH),
        "truth_contract_hash": _truth_contract_hash(),
        "clauses": clauses,
        "failed_clauses": failed_clauses,
        "rca_classes": rca_classes,
        "passed": not failed_clauses,
    }
    write_json_atomic(G6_TRAINABILITY_SUPPORT_CONTRACT_PATH, report)
    return report


def _write_failed_teacher_readiness_contract(
    plan: dict[str, Any], reason: str
) -> dict[str, Any]:
    report = {
        "gate": "g6_teacher_readiness_contract",
        "readiness_scope": "claim_readiness",
        "run_instance_id": plan.get("run_instance_id"),
        "plan_version": plan.get("plan_version"),
        "source_base_commit": plan.get("source_base_commit"),
        "source_canonical_train_cell": plan.get("source_canonical_train_cell"),
        "source_best_train_state_mode": plan.get("source_best_train_state_mode"),
        "active_train_state_mode": plan.get("active_train_state_mode"),
        "truth_contract_path": str(TRUTH_CONTRACT_PATH),
        "truth_contract_hash": _truth_contract_hash(),
        "clauses": {},
        "failed_clauses": [],
        "passed": False,
        "failure_reason": reason,
    }
    write_json_atomic(G6_TEACHER_READINESS_CONTRACT_PATH, report)
    return report


def _write_failed_trainability_support_contract(
    plan: dict[str, Any], reason: str
) -> dict[str, Any]:
    report = {
        "gate": "g6_trainability_support_contract",
        "readiness_scope": "trainability_support",
        "run_instance_id": plan.get("run_instance_id"),
        "plan_version": plan.get("plan_version"),
        "source_base_commit": plan.get("source_base_commit"),
        "source_canonical_train_cell": plan.get("source_canonical_train_cell"),
        "source_best_train_state_mode": plan.get("source_best_train_state_mode"),
        "active_train_state_mode": plan.get("active_train_state_mode"),
        "dataset_selection_mode": plan.get("dataset_selection_mode"),
        "truth_contract_path": str(TRUTH_CONTRACT_PATH),
        "truth_contract_hash": _truth_contract_hash(),
        "clauses": {},
        "failed_clauses": [],
        "rca_classes": [],
        "passed": False,
        "failure_reason": reason,
    }
    write_json_atomic(G6_TRAINABILITY_SUPPORT_CONTRACT_PATH, report)
    return report


def build_or_refresh_canonical_dataset(plan: dict[str, Any]) -> dict[str, Any]:
    dataset_selection_mode = str(plan.get("dataset_selection_mode") or "claim_canonical")
    rollout_source_dir = (
        LEARNING_SUPPORT_ROLLOUT_SOURCE_DIR
        if dataset_selection_mode == "diagnostic_learning_support"
        else DEFAULT_ROLLOUT_SOURCE_DIR
    )
    materialization_artifact_path = (
        LEARNING_SUPPORT_MATERIALIZATION_ARTIFACT
        if dataset_selection_mode == "diagnostic_learning_support"
        else MATERIALIZATION_ARTIFACT
    )
    authoritative_truth_field = (
        "measurement_truthful_for_learning_support"
        if dataset_selection_mode == "diagnostic_learning_support"
        else "measurement_truthful_for_training"
    )
    sources_checked: list[str] = []
    candidate_paths: set[Path] = set()
    for path in [
        RCA5_PATH,
        RCA6_PATH,
        CAMPAIGN_DIR
        / "runtime"
        / "controller_cycles"
        / "cycle_20260414_134515"
        / "lane_results.json",
    ]:
        payload = load_json(path, {})
        refs = _find_rollout_refs(payload)
        sources_checked.append(f"{path}:{len(refs)}refs")
        candidate_paths.update(refs)
    if rollout_source_dir.exists():
        sources_checked.append(f"{rollout_source_dir}:existing")
        candidate_paths.update(rollout_source_dir.glob("*.npz"))

    materialization = load_json(materialization_artifact_path, {})
    existing_dataset_report = load_json(TINY_RETRAIN_DATASET_BUILD_PATH, {})
    expected_attempted_rollouts = len(plan.get("train_seeds", [])) * int(
        plan.get("episodes_per_seed", HONEST_EPISODES_PER_SEED)
    )
    min_train_episodes = int(plan.get("min_train_episodes", HONEST_MIN_TRAIN_EPISODES))

    valid_rollouts, rejected = _load_validated_rollout_paths(
        sorted(candidate_paths), plan
    )
    unique_valid_seeds = {
        int(json.loads(path.with_suffix(".json").read_text()).get("seed"))
        for path in valid_rollouts
        if path.with_suffix(".json").exists()
    }
    materialization_ready = bool(
        _artifact_matches_plan(materialization, plan)
        and materialization.get("passed")
        and int(materialization.get("attempted_rollouts", 0))
        == expected_attempted_rollouts
        and int(materialization.get("saved_rollouts", 0)) >= min_train_episodes
        and int(materialization.get("successful_seed_count", 0)) >= 6
        and str(materialization.get("active_train_state_mode"))
        == str(plan.get("active_train_state_mode"))
    )
    existing_dataset_reusable = bool(
        _artifact_matches_plan(existing_dataset_report, plan)
        and bool(existing_dataset_report.get("dataset_valid", False))
        and int(existing_dataset_report.get("valid_rollout_count", 0) or 0)
        >= min_train_episodes
        and int(existing_dataset_report.get("valid_seed_count", 0) or 0) >= 6
        and materialization_ready
    )
    need_refresh = not existing_dataset_reusable
    refresh_failure_error: str | None = None

    if need_refresh:
        try:
            materialize_canonical_train_rollouts(
                DEFAULT_ROLLOUT_SOURCE_DIR,
                expected=plan,
                force_rebuild=True,
            )
        except Exception as exc:
            refresh_failure_error = (
                f"materialization_refresh_failed:{type(exc).__name__}:{exc}"
            )
        else:
            materialization = load_json(materialization_artifact_path, {})
            valid_rollouts, rejected = _load_validated_rollout_paths(
                sorted(rollout_source_dir.glob("*.npz")), plan
            )
            unique_valid_seeds = {
                int(json.loads(path.with_suffix(".json").read_text()).get("seed"))
                for path in valid_rollouts
                if path.with_suffix(".json").exists()
            }
            materialization_ready = bool(
                _artifact_matches_plan(materialization, plan)
                and materialization.get("passed")
                and int(materialization.get("attempted_rollouts", 0))
                == expected_attempted_rollouts
                and int(materialization.get("saved_rollouts", 0)) >= min_train_episodes
                and int(materialization.get("successful_seed_count", 0)) >= 6
                and str(materialization.get("active_train_state_mode"))
                == str(plan.get("active_train_state_mode"))
            )

    alignment_source_paths = (
        sorted(rollout_source_dir.glob("*.npz"))
        if rollout_source_dir.exists()
        else sorted(candidate_paths)
    )
    strict_alignment_report = _write_strict_semantics_alignment(
        alignment_source_paths, plan
    )
    rejected_by_reason, rejected_seeds_by_reason = _rejected_rollout_summary(rejected)
    raw_saved_rollout_count = int(materialization.get("saved_rollouts", 0) or 0)
    raw_successful_seed_count = int(
        materialization.get("successful_seed_count", 0) or 0
    )

    dataset_root = _resolve_repo_path(plan["dataset_root"])
    failure_common = {
        "run_instance_id": plan.get("run_instance_id"),
        "plan_version": plan.get("plan_version"),
        "source_base_commit": plan.get("source_base_commit"),
        "working_head_commit": plan.get("working_head_commit"),
        "dataset_root": str(dataset_root),
        "dataset_repo_id": str(plan["dataset_repo_id"]),
        "source_canonical_train_cell": plan.get("source_canonical_train_cell"),
        "source_best_train_state_mode": plan.get("source_best_train_state_mode"),
        "active_train_state_mode": plan.get("active_train_state_mode"),
        "active_state_mode_name": plan.get("active_state_mode_name"),
        "teacher_truth_gate": str(plan["teacher_truth_gate"]),
        "truth_contract_path": str(
            plan.get("truth_contract_path") or TRUTH_CONTRACT_PATH
        ),
        "truth_contract_hash": str(
            plan.get("truth_contract_hash") or _truth_contract_hash()
        ),
        "dataset_selection_mode": dataset_selection_mode,
        "authoritative_truth_field": authoritative_truth_field,
        "strict_utility_version": plan.get("strict_utility_version"),
        "truth_utility_version": plan.get("truth_utility_version"),
        "teacher_fingerprint_version": plan.get("teacher_fingerprint_version"),
        "learning_support_fingerprint_version": plan.get(
            "learning_support_fingerprint_version"
        ),
        "raw_saved_rollout_count": raw_saved_rollout_count,
        "raw_successful_seed_count": raw_successful_seed_count,
        "valid_rollout_count": len(valid_rollouts),
        "valid_seed_count": len(unique_valid_seeds),
        "valid_seeds": sorted(unique_valid_seeds),
        "rejected_rollout_count": len(rejected),
        "rejected_by_reason": rejected_by_reason,
        "rejected_seeds_by_reason": rejected_seeds_by_reason,
        "sources_checked": sources_checked,
        "materialization_ready": materialization_ready,
        "need_refresh": need_refresh,
        "materialization": materialization,
        "used_rollout_paths": [str(path) for path in valid_rollouts],
        "strict_semantics_alignment": strict_alignment_report,
    }

    def _finalize_failure(error: str) -> dict[str, Any]:
        claim_report = _write_failed_teacher_readiness_contract(plan, error)
        trainability_report = _write_failed_trainability_support_contract(plan, error)
        failure_report = {
            **failure_common,
            "dataset_valid": False,
            "teacher_readiness_passed": False,
            "teacher_readiness_contract": claim_report,
            "teacher_readiness_failed_clauses": list(
                claim_report.get("failed_clauses") or []
            ),
            "claim_readiness_contract": claim_report,
            "claim_readiness_passed": False,
            "claim_readiness_failed_clauses": list(
                claim_report.get("failed_clauses") or []
            ),
            "trainability_support_contract": trainability_report,
            "trainability_support_passed": False,
            "trainability_support_failed_clauses": list(
                trainability_report.get("failed_clauses") or []
            ),
            "error": error,
        }
        write_json_atomic(REJECTED_ROLLOUTS_PATH, {"rejected_rollouts": rejected})
        write_json_atomic(TINY_RETRAIN_DATASET_BUILD_PATH, failure_report)
        return failure_report

    if refresh_failure_error:
        return _finalize_failure(refresh_failure_error)

    if not valid_rollouts:
        return _finalize_failure("no_valid_canonical_rollouts_after_validation")

    if raw_saved_rollout_count < min_train_episodes or raw_successful_seed_count < 6:
        return _finalize_failure("insufficient_canonical_bridge_data")

    if len(valid_rollouts) < min_train_episodes or len(unique_valid_seeds) < 6:
        return _finalize_failure(
            "insufficient_trace_truth_adjudicated_bridge_data_after_validation"
        )

    payload = build_dataset_from_rollouts(
        valid_rollouts, dataset_root, str(plan["dataset_repo_id"])
    )
    provenance_report = validate_built_dataset_provenance(dataset_root, plan)
    build_report = {
        **provenance_report,
        **failure_common,
        "dataset_root": str(dataset_root),
        "dataset_repo_id": str(plan["dataset_repo_id"]),
        "dataset_valid": bool(provenance_report.get("dataset_valid", False)),
        "integrity": payload.get("integrity", {}),
    }
    if not build_report.get("dataset_valid", False):
        claim_report = _write_failed_teacher_readiness_contract(
            plan, "dataset_validation_failed"
        )
        trainability_report = _write_failed_trainability_support_contract(
            plan, "dataset_validation_failed"
        )
        build_report["teacher_readiness_contract"] = claim_report
        build_report["teacher_readiness_passed"] = False
        build_report["teacher_readiness_failed_clauses"] = list(
            claim_report.get("failed_clauses") or []
        )
        build_report["claim_readiness_contract"] = claim_report
        build_report["claim_readiness_passed"] = False
        build_report["claim_readiness_failed_clauses"] = list(
            claim_report.get("failed_clauses") or []
        )
        build_report["trainability_support_contract"] = trainability_report
        build_report["trainability_support_passed"] = False
        build_report["trainability_support_failed_clauses"] = list(
            trainability_report.get("failed_clauses") or []
        )
        write_json_atomic(REJECTED_ROLLOUTS_PATH, {"rejected_rollouts": rejected})
        write_json_atomic(TINY_RETRAIN_DATASET_BUILD_PATH, build_report)
        return build_report

    claim_report = _teacher_readiness_contract(build_report, plan)
    trainability_report = _trainability_support_contract(build_report, plan)
    build_report["teacher_readiness_contract"] = claim_report
    build_report["teacher_readiness_failed_clauses"] = list(
        claim_report.get("failed_clauses") or []
    )
    build_report["teacher_readiness_passed"] = bool(
        claim_report.get("passed", False)
    )
    build_report["claim_readiness_contract"] = claim_report
    build_report["claim_readiness_failed_clauses"] = list(
        claim_report.get("failed_clauses") or []
    )
    build_report["claim_readiness_passed"] = bool(
        claim_report.get("passed", False)
    )
    build_report["trainability_support_contract"] = trainability_report
    build_report["trainability_support_failed_clauses"] = list(
        trainability_report.get("failed_clauses") or []
    )
    build_report["trainability_support_passed"] = bool(
        trainability_report.get("passed", False)
    )
    write_json_atomic(REJECTED_ROLLOUTS_PATH, {"rejected_rollouts": rejected})
    write_json_atomic(TINY_RETRAIN_DATASET_BUILD_PATH, build_report)
    return build_report


def build_dataset_from_explicit_rollouts(
    explicit_rollout_paths: list[Path],
    plan: dict[str, Any],
    *,
    dataset_label: str = "explicit_rollout_dataset",
) -> dict[str, Any]:
    dataset_selection_mode = str(plan.get("dataset_selection_mode") or "claim_canonical")
    authoritative_truth_field = (
        "measurement_truthful_for_learning_support"
        if dataset_selection_mode == "diagnostic_learning_support"
        else "measurement_truthful_for_training"
    )
    explicit_paths = sorted({Path(path) for path in explicit_rollout_paths})
    valid_rollouts, rejected = _load_validated_rollout_paths(explicit_paths, plan)
    unique_valid_seeds = {
        int(json.loads(path.with_suffix(".json").read_text()).get("seed"))
        for path in valid_rollouts
        if path.with_suffix(".json").exists()
    }
    rejected_by_reason, rejected_seeds_by_reason = _rejected_rollout_summary(rejected)
    dataset_root = _resolve_repo_path(plan["dataset_root"])
    failure_common = {
        "run_instance_id": plan.get("run_instance_id"),
        "plan_version": plan.get("plan_version"),
        "source_base_commit": plan.get("source_base_commit"),
        "working_head_commit": plan.get("working_head_commit"),
        "dataset_root": str(dataset_root),
        "dataset_repo_id": str(plan["dataset_repo_id"]),
        "source_canonical_train_cell": plan.get("source_canonical_train_cell"),
        "source_best_train_state_mode": plan.get("source_best_train_state_mode"),
        "active_train_state_mode": plan.get("active_train_state_mode"),
        "active_state_mode_name": plan.get("active_state_mode_name"),
        "teacher_truth_gate": str(plan["teacher_truth_gate"]),
        "truth_contract_path": str(
            plan.get("truth_contract_path") or TRUTH_CONTRACT_PATH
        ),
        "truth_contract_hash": str(
            plan.get("truth_contract_hash") or _truth_contract_hash()
        ),
        "dataset_selection_mode": dataset_selection_mode,
        "authoritative_truth_field": authoritative_truth_field,
        "strict_utility_version": plan.get("strict_utility_version"),
        "truth_utility_version": plan.get("truth_utility_version"),
        "teacher_fingerprint_version": plan.get("teacher_fingerprint_version"),
        "learning_support_fingerprint_version": plan.get(
            "learning_support_fingerprint_version"
        ),
        "raw_saved_rollout_count": len(explicit_paths),
        "raw_successful_seed_count": len(
            {
                int(json.loads(path.with_suffix(".json").read_text()).get("seed"))
                for path in explicit_paths
                if path.with_suffix(".json").exists()
                and json.loads(path.with_suffix(".json").read_text()).get("seed")
                is not None
            }
        ),
        "valid_rollout_count": len(valid_rollouts),
        "valid_seed_count": len(unique_valid_seeds),
        "valid_seeds": sorted(unique_valid_seeds),
        "rejected_rollout_count": len(rejected),
        "rejected_by_reason": rejected_by_reason,
        "rejected_seeds_by_reason": rejected_seeds_by_reason,
        "sources_checked": [f"explicit:{len(explicit_paths)}"],
        "materialization_ready": True,
        "need_refresh": False,
        "materialization": {
            "source": dataset_label,
            "explicit_rollout_count": len(explicit_paths),
        },
        "used_rollout_paths": [str(path) for path in valid_rollouts],
        "strict_semantics_alignment": load_json(G6_STRICT_SEMANTICS_ALIGNMENT_PATH, {}),
        "dataset_source": dataset_label,
    }

    def _finalize_failure(error: str) -> dict[str, Any]:
        claim_report = _write_failed_teacher_readiness_contract(plan, error)
        trainability_report = _write_failed_trainability_support_contract(plan, error)
        failure_report = {
            **failure_common,
            "dataset_valid": False,
            "teacher_readiness_passed": False,
            "teacher_readiness_contract": claim_report,
            "teacher_readiness_failed_clauses": list(
                claim_report.get("failed_clauses") or []
            ),
            "claim_readiness_contract": claim_report,
            "claim_readiness_passed": False,
            "claim_readiness_failed_clauses": list(
                claim_report.get("failed_clauses") or []
            ),
            "trainability_support_contract": trainability_report,
            "trainability_support_passed": False,
            "trainability_support_failed_clauses": list(
                trainability_report.get("failed_clauses") or []
            ),
            "error": error,
        }
        write_json_atomic(REJECTED_ROLLOUTS_PATH, {"rejected_rollouts": rejected})
        write_json_atomic(TINY_RETRAIN_DATASET_BUILD_PATH, failure_report)
        return failure_report

    if not valid_rollouts:
        return _finalize_failure("no_valid_explicit_rollouts_after_validation")

    payload = build_dataset_from_rollouts(
        valid_rollouts, dataset_root, str(plan["dataset_repo_id"])
    )
    provenance_report = validate_built_dataset_provenance(dataset_root, plan)
    build_report = {
        **provenance_report,
        **failure_common,
        "dataset_root": str(dataset_root),
        "dataset_repo_id": str(plan["dataset_repo_id"]),
        "dataset_valid": bool(provenance_report.get("dataset_valid", False)),
        "integrity": payload.get("integrity", {}),
    }
    if not build_report.get("dataset_valid", False):
        claim_report = _write_failed_teacher_readiness_contract(
            plan, "dataset_validation_failed"
        )
        trainability_report = _write_failed_trainability_support_contract(
            plan, "dataset_validation_failed"
        )
        build_report["teacher_readiness_contract"] = claim_report
        build_report["teacher_readiness_passed"] = False
        build_report["teacher_readiness_failed_clauses"] = list(
            claim_report.get("failed_clauses") or []
        )
        build_report["claim_readiness_contract"] = claim_report
        build_report["claim_readiness_passed"] = False
        build_report["claim_readiness_failed_clauses"] = list(
            claim_report.get("failed_clauses") or []
        )
        build_report["trainability_support_contract"] = trainability_report
        build_report["trainability_support_passed"] = False
        build_report["trainability_support_failed_clauses"] = list(
            trainability_report.get("failed_clauses") or []
        )
        write_json_atomic(REJECTED_ROLLOUTS_PATH, {"rejected_rollouts": rejected})
        write_json_atomic(TINY_RETRAIN_DATASET_BUILD_PATH, build_report)
        return build_report

    claim_report = _teacher_readiness_contract(build_report, plan)
    trainability_report = _trainability_support_contract(build_report, plan)
    build_report["teacher_readiness_contract"] = claim_report
    build_report["teacher_readiness_failed_clauses"] = list(
        claim_report.get("failed_clauses") or []
    )
    build_report["teacher_readiness_passed"] = bool(
        claim_report.get("passed", False)
    )
    build_report["claim_readiness_contract"] = claim_report
    build_report["claim_readiness_failed_clauses"] = list(
        claim_report.get("failed_clauses") or []
    )
    build_report["claim_readiness_passed"] = bool(
        claim_report.get("passed", False)
    )
    build_report["trainability_support_contract"] = trainability_report
    build_report["trainability_support_failed_clauses"] = list(
        trainability_report.get("failed_clauses") or []
    )
    build_report["trainability_support_passed"] = bool(
        trainability_report.get("passed", False)
    )
    write_json_atomic(REJECTED_ROLLOUTS_PATH, {"rejected_rollouts": rejected})
    write_json_atomic(TINY_RETRAIN_DATASET_BUILD_PATH, build_report)
    return build_report


def launch_tiny_retrain(plan: dict[str, Any]) -> dict[str, Any]:
    from run_g8_mint_train import run as run_train

    run_train()
    summary = load_json(G8_SUMMARY_PATH, {})
    if str(summary.get("run_instance_id") or "") != str(
        plan.get("run_instance_id") or ""
    ):
        raise SystemExit("G8 train summary run_instance_id does not match active plan")
    return summary


def _checkpoint_probe_sweep_path(plan: dict[str, Any]) -> Path:
    return _resolve_repo_path(plan["evaluation_dir"]) / "checkpoint_probe_sweep.json"


def _checkpoint_path_for_step(train_output_dir: Path, step: int) -> Path | None:
    candidate = (
        train_output_dir / "checkpoints" / f"{int(step):06d}" / "pretrained_model"
    )
    return candidate if candidate.exists() else None


def _checkpoint_rank_key(
    result: dict[str, Any],
) -> tuple[int, float, float, float, float, float, int]:
    rank_vector = dict(result.get("attach_first_rank_vector") or {})
    return (
        1 if bool(result.get("attach_bridge_pass", False)) else 0,
        float(rank_vector.get("ever_attach_eligible_fraction", 0.0)),
        float(result.get("ever_attached_rate_gain", 0.0)),
        float(result.get("stable_attach_gain", 0.0)),
        float(result.get("phase_locked_gain", 0.0)),
        float(result.get("success_gain", 0.0)),
        int(result.get("checkpoint_step") or 0),
    )


def _selected_probe_summary_payload(
    selected: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    return {
        "run_instance_id": plan.get("run_instance_id"),
        "plan_version": plan.get("plan_version"),
        "source_base_commit": plan.get("source_base_commit"),
        "working_head_commit": plan.get("working_head_commit"),
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
        "ever_attach_eligible_fraction_gain": float(
            selected.get("ever_attach_eligible_fraction_gain", 0.0)
        ),
        "ever_attached_rate_gain": float(selected.get("ever_attached_rate_gain", 0.0)),
        "stable_attach_gain": float(selected.get("stable_attach_gain", 0.0)),
        "phase_locked_gain": float(selected.get("phase_locked_gain", 0.0)),
        "max_drawer_fraction_gain": float(
            selected.get("max_drawer_fraction_gain", 0.0)
        ),
        "attached_seed_count": int(selected.get("attached_seed_count", 0)),
        "trend_passed": bool(selected.get("trend_passed", False)),
        "train_probe_claim_pass": bool(
            selected.get("train_probe_claim_pass", selected.get("trend_passed", False))
        ),
        "attach_bridge_pass": bool(selected.get("attach_bridge_pass", False)),
        "selected_checkpoint_reason": selected.get("selected_checkpoint_reason"),
        "attach_first_rank_vector": dict(
            selected.get("attach_first_rank_vector") or {}
        ),
        "pretrained_dominant_failure_mode": selected.get(
            "pretrained_dominant_failure_mode"
        ),
        "finetuned_dominant_failure_mode": selected.get(
            "finetuned_dominant_failure_mode"
        ),
        "bridge_delta": dict(selected.get("bridge_delta") or {}),
        "summary": dict(selected.get("summary") or {}),
        "records": dict(selected.get("records") or {}),
        "selected_bridge_checkpoint": selected.get("checkpoint_path"),
        "selected_checkpoint_step": selected.get("checkpoint_step"),
    }


def run_train_probe(
    plan: dict[str, Any], train_summary: dict[str, Any]
) -> dict[str, Any]:
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
    selected_probe["selected_checkpoint_reason"] = (
        "attach_bridge_pass"
        if bool(selected.get("attach_bridge_pass", False))
        else "highest_attach_first_rank"
    )
    selected_probe["run_instance_id"] = plan.get("run_instance_id")
    selected_probe["plan_version"] = plan.get("plan_version")
    selected_probe["source_base_commit"] = plan.get("source_base_commit")
    selected_probe["working_head_commit"] = plan.get("working_head_commit")
    selected_probe["selected_bridge_checkpoint"] = selected.get("checkpoint_path")
    selected_probe["selected_checkpoint_step"] = selected.get("checkpoint_step")
    selected_probe["summary_path"] = str(
        _resolve_repo_path(plan["evaluation_dir"]) / "train_seed_probe.json"
    )
    sweep_payload = {
        "gate": "g8_checkpoint_probe_sweep",
        "training_mode": "tiny_retrain_confirmation",
        "run_instance_id": plan.get("run_instance_id"),
        "plan_version": plan.get("plan_version"),
        "source_base_commit": plan.get("source_base_commit"),
        "working_head_commit": plan.get("working_head_commit"),
        "source_canonical_train_cell": plan.get("source_canonical_train_cell"),
        "source_best_train_state_mode": plan.get("source_best_train_state_mode"),
        "active_train_state_mode": plan.get("active_train_state_mode"),
        "active_state_mode_name": plan.get("active_state_mode_name"),
        "bridge_stage": plan.get("bridge_stage"),
        "bridge_attempt": plan.get("bridge_attempt"),
        "execution_scope": str(plan.get("execution_scope") or "unspecified"),
        "diagnostic_only": bool(plan.get("diagnostic_only", False)),
        "claim_bearing": bool(plan.get("claim_bearing", False)),
        "publication_scope": str(plan.get("publication_scope") or "unspecified"),
        "result_scope": str(plan.get("result_scope") or "unspecified"),
        "evaluation_backend": plan.get("evaluation_backend"),
        "evaluation_env_family": plan.get("evaluation_env_family"),
        "evaluation_cell_id": plan.get("evaluation_cell_id"),
        "evaluation_interaction_mode": plan.get("evaluation_interaction_mode"),
        "requested_checkpoint_probe_steps": requested_steps,
        "checkpoint_results": checkpoint_results,
        "selected_bridge_checkpoint": selected.get("checkpoint_path"),
        "selected_checkpoint_step": selected.get("checkpoint_step"),
        "trend_passed": bool(selected.get("trend_passed", False)),
        "train_probe_claim_pass": bool(
            selected.get("train_probe_claim_pass", selected.get("trend_passed", False))
        ),
        "attach_bridge_pass": bool(selected.get("attach_bridge_pass", False)),
        "ever_attach_eligible_fraction_gain": float(
            selected.get("ever_attach_eligible_fraction_gain", 0.0)
        ),
        "ever_attached_rate_gain": float(selected.get("ever_attached_rate_gain", 0.0)),
        "stable_attach_gain": float(selected.get("stable_attach_gain", 0.0)),
        "phase_locked_gain": float(selected.get("phase_locked_gain", 0.0)),
        "max_drawer_fraction_gain": float(
            selected.get("max_drawer_fraction_gain", 0.0)
        ),
        "attached_seed_count": int(selected.get("attached_seed_count", 0)),
        "selected_checkpoint_reason": selected_probe.get("selected_checkpoint_reason"),
        "attach_first_rank_vector": dict(
            selected.get("attach_first_rank_vector") or {}
        ),
        "pretrained_dominant_failure_mode": selected.get(
            "pretrained_dominant_failure_mode"
        ),
        "finetuned_dominant_failure_mode": selected.get(
            "finetuned_dominant_failure_mode"
        ),
    }
    write_json_atomic(_checkpoint_probe_sweep_path(plan), sweep_payload)
    write_json_atomic(
        _resolve_repo_path(plan["evaluation_dir"]) / "train_seed_probe.json",
        _selected_probe_summary_payload(selected_probe, plan),
    )
    write_json_atomic(G8_PROBE_PATH, selected_probe)
    return selected_probe


def run_heldout_eval(
    plan: dict[str, Any], train_summary: dict[str, Any], probe_summary: dict[str, Any]
) -> dict[str, Any] | None:
    from run_g9_sim_eval import run as run_eval

    if not bool(
        probe_summary.get(
            "train_probe_claim_pass", probe_summary.get("trend_passed", False)
        )
    ):
        return None
    if not bool(train_summary.get("passed", False)):
        raise SystemExit("Cannot run held-out eval without a passed train summary")
    run_eval()
    return load_json(G9_SUMMARY_PATH, {})


def write_skipped_heldout_eval_summary(
    plan: dict[str, Any], probe_summary: dict[str, Any]
) -> dict[str, Any]:
    attach_bridge_pass = bool(probe_summary.get("attach_bridge_pass", False))
    verdict = (
        "attach_bridge_established_not_claim_supported"
        if attach_bridge_pass
        else "trainability_bridge_not_established"
    )
    reason = (
        "Attach bridge improved under MuJoCo parity, but strict train-probe success did not pass; held-out eval skipped."
        if attach_bridge_pass
        else "Trainability bridge was not established under MuJoCo parity; held-out eval skipped."
    )
    summary = {
        "gate": "g9_sim_eval",
        "training_mode": "tiny_retrain_confirmation",
        "run_instance_id": plan.get("run_instance_id"),
        "plan_version": plan.get("plan_version"),
        "source_base_commit": plan.get("source_base_commit"),
        "working_head_commit": plan.get("working_head_commit"),
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
        "heldout_skip_reason": "diagnostic_only_scope"
        if bool(plan.get("diagnostic_only", False))
        else "train_probe_not_claim_pass",
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
        f"- run instance id: `{summary.get('run_instance_id')}`",
        f"- source base commit: `{summary.get('source_base_commit')}`",
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
    *,
    diagnostic_only: bool = False,
    require_teacher_readiness: bool = True,
    unsupported_by_readiness_contract: bool = False,
) -> dict[str, Any]:
    train_passed = bool(train_summary.get("passed", False))
    train_probe_passed = bool(
        probe_summary.get(
            "train_probe_claim_pass", probe_summary.get("trend_passed", False)
        )
    )
    attach_bridge_pass = bool(probe_summary.get("attach_bridge_pass", False))
    heldout_eval_run = bool(
        eval_summary and eval_summary.get("heldout_eval_run", False)
    )
    claim_supported = bool(
        heldout_eval_run
        and eval_summary
        and eval_summary.get("verdict") == "claim_supported"
    )
    if claim_supported:
        stage_verdict = "claim_supported"
    elif attach_bridge_pass or train_probe_passed:
        stage_verdict = "attach_bridge_established_not_claim_supported"
    else:
        stage_verdict = "trainability_bridge_not_established"
    return {
        "bridge_stage": plan.get("bridge_stage"),
        "bridge_attempt": plan.get("bridge_attempt"),
        "run_instance_id": plan.get("run_instance_id"),
        "plan_version": plan.get("plan_version"),
        "source_base_commit": plan.get("source_base_commit"),
        "working_head_commit": plan.get("working_head_commit"),
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
        "execution_scope": str(plan.get("execution_scope") or "unspecified"),
        "diagnostic_only": bool(diagnostic_only),
        "claim_bearing": bool(plan.get("claim_bearing", False)),
        "publication_scope": str(plan.get("publication_scope") or "unspecified"),
        "result_scope": str(plan.get("result_scope") or "unspecified"),
        "require_teacher_readiness": bool(require_teacher_readiness),
        "unsupported_by_readiness_contract": bool(unsupported_by_readiness_contract),
        "authoritative_use": "diagnostic_only" if diagnostic_only else "authoritative",
        "selected_bridge_checkpoint": probe_summary.get("selected_bridge_checkpoint")
        or train_summary.get("checkpoint_path"),
        "selected_checkpoint_step": probe_summary.get("selected_checkpoint_step"),
        "pretrained_dominant_failure_mode": probe_summary.get(
            "pretrained_dominant_failure_mode"
        ),
        "finetuned_dominant_failure_mode": probe_summary.get(
            "finetuned_dominant_failure_mode"
        ),
        "dataset_summary": dataset_summary,
        "train_summary": train_summary,
        "probe_summary": probe_summary_payload,
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
    require_teacher_readiness: bool = True,
    diagnostic_only: bool = False,
    explicit_rollout_paths: list[Path] | None = None,
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
    dataset_summary = (
        build_dataset_from_explicit_rollouts(
            explicit_rollout_paths or [],
            plan,
            dataset_label=f"{bridge_stage}_{bridge_attempt}",
        )
        if explicit_rollout_paths is not None
        else build_or_refresh_canonical_dataset(plan)
    )
    teacher_readiness_passed = bool(
        dataset_summary.get("teacher_readiness_passed", False)
    )
    dataset_valid = bool(dataset_summary.get("dataset_valid", False))
    if (not dataset_valid) or (
        require_teacher_readiness and not teacher_readiness_passed
    ):
        failure_reason = str(
            dataset_summary.get("error")
            or (
                "teacher_readiness_not_established"
                if (dataset_valid and not teacher_readiness_passed)
                else "dataset_prepare_failed"
            )
        )
        train_summary = {
            "gate": "g8_mint_train",
            "passed": False,
            "error": failure_reason,
            "reason": failure_reason,
            "run_instance_id": plan.get("run_instance_id"),
            "plan_version": plan.get("plan_version"),
            "source_base_commit": plan.get("source_base_commit"),
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
            "finetuned_dominant_failure_mode": failure_reason,
        }
        eval_summary = write_skipped_heldout_eval_summary(plan, probe_summary)
        return _build_stage_summary(
            plan,
            dataset_summary,
            train_summary,
            probe_summary,
            eval_summary,
            diagnostic_only=diagnostic_only,
            require_teacher_readiness=require_teacher_readiness,
            unsupported_by_readiness_contract=bool(
                (not require_teacher_readiness) and (not teacher_readiness_passed)
            ),
        )

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
        return _build_stage_summary(
            plan,
            dataset_summary,
            train_summary,
            probe_summary,
            eval_summary,
            diagnostic_only=diagnostic_only,
            require_teacher_readiness=require_teacher_readiness,
            unsupported_by_readiness_contract=bool(
                (not require_teacher_readiness)
                and (not bool(dataset_summary.get("teacher_readiness_passed", False)))
            ),
        )
    probe_summary = run_train_probe(plan, train_summary)
    if bool(
        probe_summary.get(
            "train_probe_claim_pass", probe_summary.get("trend_passed", False)
        )
    ):
        eval_summary = run_heldout_eval(plan, train_summary, probe_summary)
    else:
        eval_summary = write_skipped_heldout_eval_summary(plan, probe_summary)
    return _build_stage_summary(
        plan,
        dataset_summary,
        train_summary,
        probe_summary,
        eval_summary,
        diagnostic_only=diagnostic_only,
        require_teacher_readiness=require_teacher_readiness,
        unsupported_by_readiness_contract=bool(
            (not require_teacher_readiness)
            and (not bool(dataset_summary.get("teacher_readiness_passed", False)))
        ),
    )


def _write_loop_summary(
    history: list[dict[str, Any]],
    final_verdict: str,
    scientific_terminal_state: str | None,
) -> dict[str, Any]:
    last = history[-1] if history else {}
    summary = {
        "confirmation_mode": "tiny_retrain_completion_loop_v8_3",
        "run_instance_id": last.get("run_instance_id"),
        "plan_version": last.get("plan_version"),
        "source_base_commit": last.get("source_base_commit"),
        "working_head_commit": last.get("working_head_commit"),
        "source_branch": last.get("source_branch"),
        "source_commit": last.get("source_commit"),
        "terminal_commit": TERMINAL_COMMIT,
        "source_canonical_train_cell": last.get(
            "source_canonical_train_cell", "V1cT2S0"
        ),
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
        "failure_stage": None
        if final_verdict == "claim_supported"
        else last.get("bridge_stage"),
        "completion_status": "loop_completed",
    }
    write_json_atomic(TINY_RETRAIN_SUMMARY_PATH, summary)
    write_execution_memo(summary)
    return summary


def run_completion_loop() -> dict[str, Any]:
    artifacts = load_terminal_artifacts()
    history: list[dict[str, Any]] = []
    prepare_plan, prepare_dataset = _phase_prepare()
    if not bool(prepare_dataset.get("dataset_valid", False)) or not bool(
        prepare_dataset.get("teacher_readiness_passed", False)
    ):
        history.append(
            {
                "bridge_stage": "truth_gate_alignment_s0_prepare",
                "bridge_attempt": "honest",
                "run_instance_id": prepare_plan.get("run_instance_id"),
                "plan_version": prepare_plan.get("plan_version"),
                "source_base_commit": prepare_plan.get("source_base_commit"),
                "working_head_commit": prepare_plan.get("working_head_commit"),
                "source_branch": prepare_plan.get("source_branch"),
                "source_commit": prepare_plan.get("source_commit"),
                "source_canonical_train_cell": prepare_plan.get(
                    "source_canonical_train_cell", "V1cT2S0"
                ),
                "source_best_train_state_mode": prepare_plan.get(
                    "source_best_train_state_mode", "S0"
                ),
                "active_train_state_mode": prepare_plan.get(
                    "active_train_state_mode", "S0"
                ),
                "active_state_mode_name": prepare_plan.get("active_state_mode_name"),
                "dataset_valid": bool(prepare_dataset.get("dataset_valid", False)),
                "train_passed": False,
                "train_probe_passed": False,
                "attach_bridge_pass": False,
                "heldout_eval_run": False,
                "claim_supported": False,
                "stage_verdict": "TEACHER_READINESS_NOT_ESTABLISHED",
                "dataset_summary": prepare_dataset,
                "train_summary": {},
                "probe_summary": {},
                "eval_summary": {},
            }
        )
        return _write_loop_summary(
            history,
            "TEACHER_READINESS_NOT_ESTABLISHED",
            "TEACHER_READINESS_NOT_ESTABLISHED",
        )

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

    return _write_loop_summary(
        history,
        "TINY_RETRAIN_NOT_ESTABLISHED_AFTER_S0_S2",
        "TINY_RETRAIN_NOT_ESTABLISHED_AFTER_S0_S2",
    )


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


G6_FAMILY_CONDITIONED_PROBE_PATH = ARTIFACT_DIR / "g8_family_conditioned_probe.json"
G6_ATTACH_BRIDGE_SUMMARY_PATH = ARTIFACT_DIR / "g8_attach_bridge_summary.json"


def _scope_fields(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution_scope": str(plan.get("execution_scope") or "unspecified"),
        "diagnostic_only": bool(plan.get("diagnostic_only", False)),
        "claim_bearing": bool(plan.get("claim_bearing", False)),
        "publication_scope": str(plan.get("publication_scope") or "unspecified"),
        "result_scope": str(plan.get("result_scope") or "unspecified"),
    }


def _persist_plan_with_scope(plan: dict[str, Any], g4_status: str) -> dict[str, Any]:
    plan.update(gate_scope_from_g4(g4_status))
    write_json_atomic(TINY_RETRAIN_PLAN_PATH, plan)
    return plan


def _prepare_non_readiness_blockers(
    plan: dict[str, Any], dataset_summary: dict[str, Any]
) -> tuple[list[str], dict[str, Any], dict[str, Any], dict[str, Any]]:
    materialization_path = (
        LEARNING_SUPPORT_MATERIALIZATION_ARTIFACT
        if str(plan.get("dataset_selection_mode") or "") == "diagnostic_learning_support"
        else MATERIALIZATION_ARTIFACT
    )
    materialization = load_json(materialization_path, {})
    claim_report = dict(
        dataset_summary.get("claim_readiness_contract")
        or dataset_summary.get("teacher_readiness_contract")
        or load_json(G6_TEACHER_READINESS_CONTRACT_PATH, {})
    )
    trainability_report = dict(
        dataset_summary.get("trainability_support_contract")
        or load_json(G6_TRAINABILITY_SUPPORT_CONTRACT_PATH, {})
    )
    blockers: list[str] = []
    if not _artifact_matches_plan(materialization, plan):
        blockers.append("materialization_plan_mismatch")
    if str(materialization.get("teacher_controller_mode") or "") != str(plan.get("teacher_controller_mode") or ""):
        blockers.append("teacher_controller_mode_mismatch")
    if bool(materialization.get("stale_source_mismatch", False)):
        blockers.append("stale_source_mismatch")
    if not bool(dataset_summary.get("dataset_valid", False)):
        blockers.append("dataset_invalid")
    if dataset_summary.get("errors"):
        blockers.extend(str(x) for x in dataset_summary.get("errors") or [])
    if str(dataset_summary.get("truth_contract_hash") or plan.get("truth_contract_hash") or "") != str(plan.get("truth_contract_hash") or ""):
        blockers.append("truth_contract_hash_mismatch")
    if str(plan.get("acceptance_contract_hash") or "") != sha256_file(ACCEPTANCE_CONTRACT_PATH):
        blockers.append("acceptance_contract_hash_mismatch")
    return blockers, materialization, claim_report, trainability_report


def _emit_prepare_gates(plan: dict[str, Any], dataset_summary: dict[str, Any]) -> dict[str, Any]:
    docs_manifest = write_docs_lock_manifest()
    sovereign = write_sovereign_snapshot(plan)

    g0_reasons: list[str] = []
    if str(plan.get("working_head_commit") or "") != str(sovereign.get("working_head_commit") or ""):
        g0_reasons.append("working_head_commit_mismatch")
    if str(plan.get("source_branch") or "") != str(sovereign.get("branch") or ""):
        g0_reasons.append("branch_mismatch")
    g0 = write_gate(
        "G0",
        "sovereign_sync",
        plan,
        status="PASS" if not g0_reasons else "STOP",
        blocking_reasons=g0_reasons,
        allowed_next_phases=["prepare"] if not g0_reasons else [],
        extra={"sovereign_snapshot_path": str(SOVEREIGN_SNAPSHOT_PATH)},
    )

    g1_reasons: list[str] = []
    if not docs_manifest.get("docs"):
        g1_reasons.append("docs_lock_manifest_empty")
    g1 = write_gate(
        "G1",
        "plan_freeze",
        plan,
        status="PASS" if not g1_reasons else "STOP",
        blocking_reasons=g1_reasons,
        allowed_next_phases=["prepare"] if not g1_reasons else [],
        extra={"docs_lock_manifest_path": str((GATES_DIR.parent / 'docs_lock_manifest.json'))},
    )

    (
        non_readiness_blockers,
        materialization,
        claim_report,
        trainability_report,
    ) = _prepare_non_readiness_blockers(plan, dataset_summary)
    g2 = write_gate(
        "G2",
        "materialization_sync",
        plan,
        status="PASS" if not non_readiness_blockers else "STOP",
        blocking_reasons=non_readiness_blockers,
        allowed_next_phases=["prepare"] if not non_readiness_blockers else [],
        extra={
            "materialization_artifact": str(
                LEARNING_SUPPORT_MATERIALIZATION_ARTIFACT
                if str(plan.get("dataset_selection_mode") or "")
                == "diagnostic_learning_support"
                else MATERIALIZATION_ARTIFACT
            ),
            "dataset_selection_mode": plan.get("dataset_selection_mode"),
            "teacher_controller_mode": materialization.get("teacher_controller_mode"),
            "teacher_controller_max_steps": materialization.get("teacher_controller_max_steps"),
            "teacher_pull_open_fraction": materialization.get("teacher_pull_open_fraction"),
        },
    )

    g3_reasons = [] if bool(dataset_summary.get("dataset_valid", False)) else [str(dataset_summary.get("error") or "dataset_invalid")]
    g3 = write_gate(
        "G3",
        "dataset_integrity",
        plan,
        status="PASS" if not g3_reasons else "STOP",
        blocking_reasons=g3_reasons,
        allowed_next_phases=["prepare"] if not g3_reasons else [],
        extra={
            "dataset_root": dataset_summary.get("dataset_root"),
            "used_rollout_count": dataset_summary.get("used_rollout_count"),
            "effective_frame_count": dataset_summary.get("effective_frame_count"),
            "accepted_unique_teacher_family_count": dataset_summary.get("accepted_unique_teacher_family_count"),
            "strict_unique_teacher_family_count": dataset_summary.get("strict_unique_teacher_family_count"),
            "near_strict_unique_teacher_family_count": dataset_summary.get("near_strict_unique_teacher_family_count"),
        },
    )

    claim_failed_clauses = readiness_failed_clauses(claim_report)
    trainability_failed_clauses = readiness_failed_clauses(trainability_report)
    g4_reasons: list[str] = []
    g4_status = "STOP"
    if g0.get("status") == "PASS" and g1.get("status") == "PASS" and g2.get("status") == "PASS" and g3.get("status") == "PASS":
        if bool(claim_report.get("passed", False)):
            g4_status = "AUTHORITATIVE_PASS"
        elif bool(trainability_report.get("passed", False)):
            g4_status = "DIAGNOSTIC_PASS"
        else:
            g4_reasons.extend(
                trainability_failed_clauses or ["trainability_support_not_established"]
            )
    else:
        g4_reasons.extend(non_readiness_blockers)
        if g3_reasons:
            g4_reasons.extend(g3_reasons)

    plan = _persist_plan_with_scope(plan, g4_status)
    docs_intent = write_docs_update_intent({
        "run_instance_id": plan.get("run_instance_id"),
        "spec_doc_path": str(SPEC_DOC_PATH),
        "intent": "defer_canonical_doc_updates_until_g8",
        "allowed_after_gate": "G8",
        "publication_scope": plan.get("publication_scope"),
        "result_scope": plan.get("result_scope"),
        "claim_bearing": bool(plan.get("claim_bearing", False)),
        "canonical_doc_updates_allowed_before_g8": False,
        "requested_addendum_behavior": "none",
    })
    publication_state = write_publication_state({
        "run_instance_id": plan.get("run_instance_id"),
        **_scope_fields(plan),
        "g4_status": g4_status,
        "final_verdict": None,
    })
    harness_state = write_harness_state({
        "run_instance_id": plan.get("run_instance_id"),
        "line_a_state": "operational_freeze",
        "line_b_state": "frozen",
        "line_c_state": "frozen_vendor_baseline",
        "current_active_blocker": "diagnostic_tiny_retrain_gate" if g4_status != "STOP" else "prepare_stop",
        "g4_status": g4_status,
        "teacher_readiness_failed_clauses": claim_failed_clauses,
        "trainability_support_failed_clauses": trainability_failed_clauses,
        "authoritative_baseline_launcher": "run_p1c10_release_runtime_matched_ab.py --variant-run via /root/anaconda3/envs/mint/bin/python",
        "diagnostic_wrapper": "run_p1c7_official_libero_goal_drawer_baseline.py",
        "runtime_preflight_gate": "run_g8_runtime_compat_smoke.py",
        "probe_execution_backend": "authoritative_mint_subprocess",
        "eval_execution_backend": "authoritative_mint_subprocess",
    })
    g4 = write_gate(
        "G4",
        "readiness_interpretation",
        plan,
        status=g4_status,
        blocking_reasons=g4_reasons,
        allowed_next_phases=["train"] if g4_status in {"DIAGNOSTIC_PASS", "AUTHORITATIVE_PASS"} else [],
        extra={
            "teacher_readiness_passed": bool(claim_report.get("passed", False)),
            "failed_readiness_clauses": claim_failed_clauses,
            "claim_readiness_passed": bool(claim_report.get("passed", False)),
            "claim_readiness_failed_clauses": claim_failed_clauses,
            "trainability_support_passed": bool(
                trainability_report.get("passed", False)
            ),
            "trainability_support_failed_clauses": trainability_failed_clauses,
            **_scope_fields(plan),
            "docs_update_intent_path": str(DOCS_UPDATE_INTENT_PATH),
            "publication_state_path": str(PUBLICATION_STATE_PATH),
            "harness_state_path": str(HARNESS_STATE_PATH),
        },
    )
    return {
        "plan": plan,
        "dataset_summary": dataset_summary,
        "docs_manifest": docs_manifest,
        "sovereign_snapshot": sovereign,
        "publication_state": publication_state,
        "harness_state": harness_state,
        "docs_update_intent": docs_intent,
        "gates": {"g0": g0, "g1": g1, "g2": g2, "g3": g3, "g4": g4},
    }


def _load_required_g4_gate() -> dict[str, Any]:
    gate = load_gate("G4", "readiness_interpretation")
    if not gate:
        raise SystemExit("Missing G4 readiness interpretation gate")
    return gate


def _artifact_for_plan(payload: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    if str(payload.get("run_instance_id") or "") != str(plan.get("run_instance_id") or ""):
        return {}
    return dict(payload)


def _probe_metric_mean(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    values = [float(item.get(key, 0.0) or 0.0) for item in records]
    return float(np.mean(np.asarray(values, dtype=np.float32))) if values else 0.0


def _strict_success_rate(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    values = [1.0 if bool((item.get("strict_metrics") or {}).get("strict_success", False)) else 0.0 for item in records]
    return float(np.mean(np.asarray(values, dtype=np.float32))) if values else 0.0


def _ever_attached_rate(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    flags = []
    for item in records:
        attached_trace = item.get("attached_trace") or []
        flags.append(1.0 if bool(item.get("grasp_success", False)) or any(bool(x) for x in attached_trace) else 0.0)
    return float(np.mean(np.asarray(flags, dtype=np.float32))) if flags else 0.0


def _attach_bridge_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "episode_count": int(len(records)),
        "ever_attach_eligible_fraction": _probe_metric_mean(
            records, "ever_attach_eligible_fraction"
        ),
        "ever_attached_rate": _ever_attached_rate(records),
        "stable_attach_rate": _probe_metric_mean(records, "stable_attach_rate"),
        "phase_locked_rate": _probe_metric_mean(records, "phase_locked_rate"),
        "effective_pull_progress_peak_mean": _probe_metric_mean(records, "effective_pull_progress_peak"),
        "max_drawer_fraction_mean": float(np.mean(np.asarray([
            float((item.get("strict_metrics") or {}).get("max_drawer_fraction", item.get("pull_distance", 0.0)) or 0.0)
            for item in records
        ], dtype=np.float32))) if records else 0.0,
        "strict_success_rate": _strict_success_rate(records),
    }


def _failure_mode_moves_later_than_attach_distance(
    pretrained_mode: str | None, finetuned_mode: str | None
) -> bool:
    baseline = "never_reach_attach_distance"
    if str(pretrained_mode or "") != baseline:
        return False
    later_modes = {
        "never_reach_attach_orientation",
        "never_reach_attach_approach",
        "never_stabilize_attach",
        "never_phase_lock",
        "no_open_after_phase_lock",
        "partial_open_only",
        "success",
    }
    return str(finetuned_mode or "") in later_modes


def _classify_probe_signal(probe_summary: dict[str, Any], plan: dict[str, Any]) -> str:
    if bool(plan.get("claim_bearing", False)) and bool(
        probe_summary.get(
            "train_probe_claim_pass", probe_summary.get("trend_passed", False)
        )
    ):
        return "claim_support_candidate"
    if bool(probe_summary.get("attach_bridge_pass", False)):
        return "diagnostic_learning_support_signal_detected"
    gains = {
        "ever_attach_eligible_fraction_gain": float(
            probe_summary.get("ever_attach_eligible_fraction_gain", 0.0)
        ),
        "ever_attached_rate_gain": float(
            probe_summary.get("ever_attached_rate_gain", 0.0)
        ),
        "stable_attach_gain": float(probe_summary.get("stable_attach_gain", 0.0)),
        "phase_locked_gain": float(probe_summary.get("phase_locked_gain", 0.0)),
    }
    later_failure = _failure_mode_moves_later_than_attach_distance(
        str(probe_summary.get("pretrained_dominant_failure_mode") or ""),
        str(probe_summary.get("finetuned_dominant_failure_mode") or ""),
    )
    return (
        "diagnostic_learning_support_signal_detected"
        if (
            gains["ever_attach_eligible_fraction_gain"] > 0.05
            or gains["ever_attached_rate_gain"] > 0.05
            or gains["stable_attach_gain"] > 0.03
            or gains["phase_locked_gain"] > 0.03
            or later_failure
        )
        else "diagnostic_learning_support_no_signal_after_support_family_fix"
    )


def _build_attach_bridge_summary(plan: dict[str, Any], probe_summary: dict[str, Any]) -> dict[str, Any]:
    records = dict(probe_summary.get("records") or {})
    pretrained = _attach_bridge_metrics(list(records.get("pretrained_mint") or []))
    finetuned = _attach_bridge_metrics(list(records.get("finetuned_mint") or []))
    delta = {key: float(finetuned.get(key, 0.0)) - float(pretrained.get(key, 0.0)) for key in pretrained.keys() if key != "episode_count"}
    payload = {
        "gate": "g6_attach_bridge_summary",
        "run_instance_id": plan.get("run_instance_id"),
        "working_head_commit": plan.get("working_head_commit"),
        **_scope_fields(plan),
        "classification": None,
        "pretrained": pretrained,
        "finetuned": finetuned,
        "delta": delta,
    }
    write_json_atomic(G6_ATTACH_BRIDGE_SUMMARY_PATH, payload)
    return payload


def _build_family_conditioned_probe(plan: dict[str, Any], dataset_summary: dict[str, Any], probe_summary: dict[str, Any]) -> dict[str, Any]:
    records = dict(probe_summary.get("records") or {})
    provenance_records: list[dict[str, Any]] = []
    provenance_path = str(dataset_summary.get("provenance_path") or "")
    if provenance_path:
        provenance = load_json(provenance_path, {})
        provenance_records = list(provenance.get("records") or [])
    if not provenance_records:
        for path_str in list(dataset_summary.get("used_rollout_paths") or []):
            meta_path = Path(path_str).with_suffix(".json")
            if meta_path.exists():
                provenance_records.append(load_json(meta_path, {}))
    pretrained_records = list(records.get("pretrained_mint") or [])
    finetuned_records = list(records.get("finetuned_mint") or [])
    by_seed = {}
    all_seeds = sorted({int(item.get("seed")) for item in pretrained_records + finetuned_records if item.get("seed") is not None})
    for seed in all_seeds:
        pt = [item for item in pretrained_records if int(item.get("seed")) == seed]
        ft = [item for item in finetuned_records if int(item.get("seed")) == seed]
        by_seed[str(seed)] = {
            "pretrained": _attach_bridge_metrics(pt),
            "finetuned": _attach_bridge_metrics(ft),
        }
    def _seed_group_metrics(seed_set: set[int]) -> dict[str, Any]:
        pt = [item for item in pretrained_records if int(item.get("seed")) in seed_set]
        ft = [item for item in finetuned_records if int(item.get("seed")) in seed_set]
        return {"pretrained": _attach_bridge_metrics(pt), "finetuned": _attach_bridge_metrics(ft), "seed_count": len(seed_set)}
    by_episode_class = {}
    episode_classes = sorted({str(rec.get("teacher_episode_class")) for rec in provenance_records if rec.get("teacher_episode_class")})
    for klass in episode_classes:
        seed_set = {int(rec.get("seed")) for rec in provenance_records if rec.get("seed") is not None and str(rec.get("teacher_episode_class")) == klass}
        by_episode_class[klass] = _seed_group_metrics(seed_set)
    by_teacher_fingerprint = {}
    fingerprints = sorted({str(rec.get("teacher_fingerprint")) for rec in provenance_records if rec.get("teacher_fingerprint")})
    for fingerprint in fingerprints:
        seed_set = {int(rec.get("seed")) for rec in provenance_records if rec.get("seed") is not None and str(rec.get("teacher_fingerprint")) == fingerprint}
        by_teacher_fingerprint[fingerprint] = _seed_group_metrics(seed_set)
    by_teacher_family_variant = {}
    variants = sorted(
        {
            str(rec.get("teacher_family_variant"))
            for rec in provenance_records
            if rec.get("teacher_family_variant")
        }
    )
    for variant in variants:
        seed_set = {
            int(rec.get("seed"))
            for rec in provenance_records
            if rec.get("seed") is not None
            and str(rec.get("teacher_family_variant")) == variant
        }
        by_teacher_family_variant[variant] = _seed_group_metrics(seed_set)
    payload = {
        "gate": "g6_family_conditioned_probe",
        "run_instance_id": plan.get("run_instance_id"),
        "working_head_commit": plan.get("working_head_commit"),
        **_scope_fields(plan),
        "classification": None,
        "overall_best_checkpoint_step": int(probe_summary.get("checkpoint_step") or 0),
        "by_seed": by_seed,
        "hard_vs_rest": {
            "hard_2_4": _seed_group_metrics({2, 4}),
            "rest": _seed_group_metrics(set(all_seeds) - {2, 4}),
        },
        "hard_vs_rest_summary": {
            "hard_2_4": _seed_group_metrics({2, 4}),
            "rest": _seed_group_metrics(set(all_seeds) - {2, 4}),
        },
        "by_teacher_episode_class": by_episode_class,
        "by_teacher_fingerprint": by_teacher_fingerprint,
        "by_teacher_family_variant": by_teacher_family_variant,
    }
    write_json_atomic(G6_FAMILY_CONDITIONED_PROBE_PATH, payload)
    return payload


def _write_g5_gate(plan: dict[str, Any], train_summary: dict[str, Any]) -> dict[str, Any]:
    status = "PASS" if bool(train_summary.get("passed", False)) else "STOP"
    return write_gate(
        "G5",
        "train_launch",
        plan,
        status=status,
        blocking_reasons=[] if status == "PASS" else [str(train_summary.get("stderr_tail") or train_summary.get("error") or "train_failed")],
        allowed_next_phases=["probe"] if status == "PASS" else [],
        extra={**_scope_fields(plan), "train_summary_path": str(G8_SUMMARY_PATH)},
    )


def _write_g6_gate(plan: dict[str, Any], dataset_summary: dict[str, Any], probe_summary: dict[str, Any]) -> dict[str, Any]:
    classification = _classify_probe_signal(probe_summary, plan)
    attach_bridge_summary = _build_attach_bridge_summary(plan, probe_summary)
    family_conditioned_probe = _build_family_conditioned_probe(plan, dataset_summary, probe_summary)
    attach_bridge_summary["classification"] = classification
    write_json_atomic(G6_ATTACH_BRIDGE_SUMMARY_PATH, attach_bridge_summary)
    family_conditioned_probe["classification"] = classification
    family_conditioned_probe["overall_best_checkpoint_step"] = int(probe_summary.get("checkpoint_step") or 0)
    family_conditioned_probe["hard_vs_rest_summary"] = family_conditioned_probe.get("hard_vs_rest")
    write_json_atomic(G6_FAMILY_CONDITIONED_PROBE_PATH, family_conditioned_probe)
    status = {
        "claim_support_candidate": "AUTHORITATIVE_PASS" if bool(plan.get("claim_bearing", False)) else "DIAGNOSTIC_PASS",
        "diagnostic_learning_support_signal_detected": "DIAGNOSTIC_PASS",
        "diagnostic_learning_support_no_signal_after_support_family_fix": "STOP",
    }[classification]
    return write_gate(
        "G6",
        "probe_analysis",
        plan,
        status=status,
        blocking_reasons=[]
        if classification != "diagnostic_learning_support_no_signal_after_support_family_fix"
        else ["no_learning_signal"],
        allowed_next_phases=["eval", "finalize"]
        if classification != "diagnostic_learning_support_no_signal_after_support_family_fix"
        else ["finalize"],
        extra={
            **_scope_fields(plan),
            "classification": classification,
            "family_conditioned_probe_path": str(G6_FAMILY_CONDITIONED_PROBE_PATH),
            "attach_bridge_summary_path": str(G6_ATTACH_BRIDGE_SUMMARY_PATH),
            "train_seed_probe_path": str(G8_PROBE_PATH),
            "attach_bridge_pass": bool(probe_summary.get("attach_bridge_pass", False)),
            "train_probe_claim_pass": bool(probe_summary.get("train_probe_claim_pass", probe_summary.get("trend_passed", False))),
        },
    )


def _write_g7_gate(plan: dict[str, Any], eval_summary: dict[str, Any]) -> dict[str, Any]:
    if bool(plan.get("diagnostic_only", False)):
        status = "DIAGNOSTIC_PASS"
        blockers: list[str] = []
        allowed = ["finalize"]
    else:
        status = "AUTHORITATIVE_PASS" if bool(eval_summary.get("heldout_eval_run", False)) else "STOP"
        blockers = [] if status != "STOP" else [str(eval_summary.get("heldout_skip_reason") or "heldout_not_run")]
        allowed = ["finalize"] if status != "STOP" else []
    return write_gate(
        "G7",
        "heldout_eligibility",
        plan,
        status=status,
        blocking_reasons=blockers,
        allowed_next_phases=allowed,
        extra={**_scope_fields(plan), **dict(eval_summary or {})},
    )


def _write_v10_finalize_summary(plan: dict[str, Any], dataset_summary: dict[str, Any], train_summary: dict[str, Any], probe_summary: dict[str, Any], eval_summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    train_summary = _artifact_for_plan(train_summary, plan)
    probe_summary = _artifact_for_plan(probe_summary, plan)
    eval_summary = _artifact_for_plan(eval_summary, plan)
    g4 = _load_required_g4_gate()
    g5 = _artifact_for_plan(load_gate("G5", "train_launch"), plan)
    g6 = _artifact_for_plan(load_gate("G6", "probe_analysis"), plan)
    g7 = _artifact_for_plan(load_gate("G7", "heldout_eligibility"), plan)
    docs_ok, docs_drift, docs_manifest = docs_lock_consistent()
    sovereign_snapshot = load_json(SOVEREIGN_SNAPSHOT_PATH, {})
    ident = _git(["git", "rev-parse", "HEAD"]), _git(["git", "rev-parse", "HEAD:external/MINT"])
    publication_issues: list[str] = []
    if not docs_ok:
        publication_issues.append("docs_lock_drift")
    if str(sovereign_snapshot.get("working_head_commit") or "") != ident[0]:
        publication_issues.append("working_head_commit_drift")
    if str(sovereign_snapshot.get("vendor_head_commit") or "") != ident[1]:
        publication_issues.append("vendor_head_commit_drift")
    if not DOCS_UPDATE_INTENT_PATH.exists():
        publication_issues.append("docs_update_intent_missing")
    if bool(plan.get("diagnostic_only", False)) and bool(plan.get("claim_bearing", False)):
        publication_issues.append("diagnostic_scope_claim_bearing_conflict")

    classification = str(g6.get("classification") or "")
    if publication_issues:
        final_verdict = "invalid_publication_state"
        g8_status = "STOP"
    elif g4.get("status") == "STOP":
        final_verdict = "prepare_stop"
        g8_status = "STOP"
    elif g4.get("status") != "AUTHORITATIVE_PASS":
        if classification == "diagnostic_learning_support_signal_detected":
            final_verdict = "diagnostic_learning_support_signal_detected"
        elif classification == "diagnostic_learning_support_no_signal":
            final_verdict = "diagnostic_learning_support_no_signal"
        else:
            final_verdict = "diagnostic_learning_support_signal_detected"
        g8_status = "DIAGNOSTIC_PASS"
    else:
        final_verdict = "claim_supported" if eval_summary.get("verdict") == "claim_supported" else "invalid_publication_state"
        g8_status = "AUTHORITATIVE_PASS" if final_verdict == "claim_supported" else "STOP"

    summary = {
        "confirmation_mode": "tiny_retrain_completion_loop_v10",
        "run_instance_id": plan.get("run_instance_id"),
        "plan_version": plan.get("plan_version"),
        "source_base_commit": plan.get("source_base_commit"),
        "working_head_commit": plan.get("working_head_commit"),
        "source_branch": plan.get("source_branch"),
        "source_commit": plan.get("source_commit"),
        **_scope_fields(plan),
        "source_canonical_train_cell": plan.get("source_canonical_train_cell"),
        "source_best_train_state_mode": plan.get("source_best_train_state_mode"),
        "active_train_state_mode": plan.get("active_train_state_mode"),
        "active_state_mode_name": plan.get("active_state_mode_name"),
        "final_verdict": final_verdict,
        "dataset_valid": bool(dataset_summary.get("dataset_valid", False)),
        "teacher_readiness_passed": bool(dataset_summary.get("teacher_readiness_passed", False)),
        "teacher_readiness_failed_clauses": list(dataset_summary.get("teacher_readiness_failed_clauses") or []),
        "claim_readiness_passed": bool(
            dataset_summary.get("claim_readiness_passed", False)
        ),
        "claim_readiness_failed_clauses": list(
            dataset_summary.get("claim_readiness_failed_clauses") or []
        ),
        "trainability_support_passed": bool(
            dataset_summary.get("trainability_support_passed", False)
        ),
        "trainability_support_failed_clauses": list(
            dataset_summary.get("trainability_support_failed_clauses") or []
        ),
        "train_passed": bool(train_summary.get("passed", False)),
        "probe_classification": classification,
        "heldout_eval_run": bool(eval_summary.get("heldout_eval_run", False)),
        "heldout_skip_reason": eval_summary.get("heldout_skip_reason"),
        "publication_issues": publication_issues,
        "gate_statuses": {
            "G4": g4.get("status"),
            "G5": g5.get("status"),
            "G6": g6.get("status"),
            "G7": g7.get("status"),
        },
        "dataset_summary": dataset_summary,
        "train_summary": train_summary,
        "probe_summary": probe_summary,
        "eval_summary": eval_summary,
        "docs_lock_manifest": docs_manifest,
    }
    write_json_atomic(TINY_RETRAIN_SUMMARY_PATH, summary)
    docs_intent = write_docs_update_intent({
        "run_instance_id": plan.get("run_instance_id"),
        "intent": "post_g8_diagnostic_report_sync",
        "allowed_after_gate": "G8",
        "publication_scope": plan.get("publication_scope"),
        "result_scope": plan.get("result_scope"),
        "claim_bearing": bool(plan.get("claim_bearing", False)),
        "canonical_doc_updates_allowed_before_g8": False,
        "requested_addendum_behavior": "diagnostic_report_and_audit_sync_only",
    })
    harness_state = write_harness_state({
        "run_instance_id": plan.get("run_instance_id"),
        "line_a_state": "operational_freeze",
        "line_b_state": "frozen",
        "line_c_state": "frozen_vendor_baseline",
        "current_active_blocker": "learning_support_root_cause_analysis"
        if final_verdict == "diagnostic_learning_support_no_signal"
        else "diagnostic_learning_support_signal_followup",
        "g4_status": g4.get("status"),
        "g6_status": g6.get("status"),
        "final_verdict": final_verdict,
        "teacher_readiness_failed_clauses": list(dataset_summary.get("teacher_readiness_failed_clauses") or []),
        "trainability_support_failed_clauses": list(
            dataset_summary.get("trainability_support_failed_clauses") or []
        ),
        "authoritative_baseline_launcher": "run_p1c10_release_runtime_matched_ab.py --variant-run via /root/anaconda3/envs/mint/bin/python",
        "diagnostic_wrapper": "run_p1c7_official_libero_goal_drawer_baseline.py",
        "runtime_preflight_gate": "run_g8_runtime_compat_smoke.py",
        "probe_execution_backend": "authoritative_mint_subprocess",
        "eval_execution_backend": "authoritative_mint_subprocess",
    })
    publication_state = write_publication_state({
        "run_instance_id": plan.get("run_instance_id"),
        **_scope_fields(plan),
        "final_verdict": final_verdict,
        "claim_supported": final_verdict == "claim_supported",
    })
    g8 = write_gate(
        "G8",
        "publication_gate",
        plan,
        status=g8_status,
        blocking_reasons=publication_issues,
        allowed_next_phases=[],
        extra={
            **_scope_fields(plan),
            "final_verdict": final_verdict,
            "publication_state_path": str(PUBLICATION_STATE_PATH),
            "docs_update_intent_path": str(DOCS_UPDATE_INTENT_PATH),
        },
    )
    return summary, g8


def _load_state() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        load_json(TINY_RETRAIN_PLAN_PATH, {}),
        load_json(TINY_RETRAIN_DATASET_BUILD_PATH, {}),
        load_json(G8_SUMMARY_PATH, {}),
        load_json(G8_PROBE_PATH, {}),
        load_json(G9_SUMMARY_PATH, {}),
    )


def _v10_gate_sync_reasons() -> list[str]:
    reasons: list[str] = []
    if _git(["git", "rev-parse", "--abbrev-ref", "HEAD"]) != "feature/mint-env-reformulation-v1-visual-fidelity":
        reasons.append("branch_mismatch")
    if _git(["git", "rev-parse", "HEAD:external/MINT"]) != "4eab5795345721001c412ff1ca2c886a11eab606":
        reasons.append("vendor_head_mismatch")
    g3 = load_json(V10_G3_GATE_PATH, {})
    g4 = load_json(V10_G4_GATE_PATH, {})
    if str(g3.get("status") or "") != "PASS":
        reasons.append("v10_g3_not_pass")
    if str(g4.get("status") or "") != "STOP":
        reasons.append("v10_g4_not_stop")
    expected_failed = {
        "seed2_learning_support_families_ge_2",
        "seed4_learning_support_families_ge_2",
        "learning_support_unique_teacher_families_ge_12",
    }
    got_failed = {
        str(x) for x in (g4.get("trainability_support_failed_clauses") or [])
    }
    if got_failed != expected_failed:
        reasons.append("v10_trainability_failed_clause_mismatch")
    return reasons


def _support_family_branch_from_audit(audit: dict[str, Any]) -> str:
    recommended = str(audit.get("recommended_next_action") or "")
    if recommended == "fix_fingerprint_only":
        return "G2A"
    if recommended == "run_bounded_teacher_family_grid":
        return "G2B"
    return "STOP"


def _prepare_stop_dataset_summary(
    plan: dict[str, Any], audit: dict[str, Any], error: str
) -> dict[str, Any]:
    summary = {
        "run_instance_id": plan.get("run_instance_id"),
        "plan_version": plan.get("plan_version"),
        "source_base_commit": plan.get("source_base_commit"),
        "working_head_commit": plan.get("working_head_commit"),
        "dataset_root": str(_resolve_repo_path(plan["dataset_root"])),
        "dataset_repo_id": str(plan["dataset_repo_id"]),
        "dataset_selection_mode": plan.get("dataset_selection_mode"),
        "dataset_valid": False,
        "claim_readiness_passed": False,
        "trainability_support_passed": False,
        "teacher_readiness_passed": False,
        "teacher_readiness_failed_clauses": [],
        "claim_readiness_failed_clauses": [],
        "trainability_support_failed_clauses": [],
        "error": error,
        "family_forensic_audit": audit,
    }
    write_json_atomic(TINY_RETRAIN_DATASET_BUILD_PATH, summary)
    return summary


def _emit_prepare_gates_v11(
    plan: dict[str, Any], audit: dict[str, Any], dataset_summary: dict[str, Any]
) -> dict[str, Any]:
    docs_manifest = write_docs_lock_manifest()
    sovereign = write_sovereign_snapshot(plan)
    g0_reasons = _v10_gate_sync_reasons()
    g0 = write_gate(
        "G0",
        "sovereign_sync",
        plan,
        status="PASS" if not g0_reasons else "STOP",
        blocking_reasons=g0_reasons,
        allowed_next_phases=["prepare"] if not g0_reasons else [],
        extra={
            "sovereign_snapshot_path": str(SOVEREIGN_SNAPSHOT_PATH),
            "v10_g3_gate_path": str(V10_G3_GATE_PATH),
            "v10_g4_gate_path": str(V10_G4_GATE_PATH),
            "v10_g8_gate_path": str(V10_G8_GATE_PATH),
        },
    )
    g1_reasons = list(audit.get("parse_errors") or [])
    recommended = str(audit.get("recommended_next_action") or "")
    if recommended not in {
        "fix_fingerprint_only",
        "run_bounded_teacher_family_grid",
        "cannot_decide_trace_missing",
    }:
        g1_reasons.append("recommended_next_action_invalid")
    if recommended == "cannot_decide_trace_missing":
        g1_reasons.append("cannot_decide_trace_missing")
    g1 = write_gate(
        "G1",
        "family_collapse_forensic_audit",
        plan,
        status="PASS" if not g1_reasons else "STOP",
        blocking_reasons=g1_reasons,
        allowed_next_phases=["prepare"] if not g1_reasons else [],
        extra={
            "audit_artifact_path": str(G1_SUPPORT_FAMILY_AUDIT_PATH),
            "recommended_next_action": recommended,
            "recorded_learning_support_unique_family_count": audit.get(
                "recorded_learning_support_unique_family_count"
            ),
            "effective_support_unique_family_count": audit.get(
                "effective_support_unique_family_count"
            ),
            "seed2_effective_support_families": audit.get(
                "seed2_effective_support_families"
            ),
            "seed4_effective_support_families": audit.get(
                "seed4_effective_support_families"
            ),
        },
    )
    selected_branch = str(plan.get("support_family_repair_mode") or _support_family_branch_from_audit(audit))
    g2 = write_gate(
        selected_branch if selected_branch in {"G2A", "G2B"} else "G2",
        "support_family_repair_branch",
        plan,
        status="PASS" if selected_branch in {"G2A", "G2B"} else "STOP",
        blocking_reasons=[] if selected_branch in {"G2A", "G2B"} else ["trace_logging_insufficient"],
        allowed_next_phases=["prepare"] if selected_branch in {"G2A", "G2B"} else [],
        extra={
            "selected_family_diversification_mode": selected_branch,
            "audit_recommended_next_action": str(audit.get("recommended_next_action") or ""),
            "support_family_repair_mode": plan.get("support_family_repair_mode"),
            "teacher_family_grid_version": plan.get("teacher_family_grid_version"),
        },
    )
    g3_reasons: list[str] = []
    if not bool(dataset_summary.get("dataset_valid", False)):
        g3_reasons.append(str(dataset_summary.get("error") or "dataset_invalid"))
    if not bool(dataset_summary.get("trainability_support_passed", False)):
        g3_reasons.extend(
            str(x) for x in (dataset_summary.get("trainability_support_failed_clauses") or [])
        )
    g3 = write_gate(
        "G3",
        "dataset_rebuild",
        plan,
        status="PASS" if not g3_reasons else "STOP",
        blocking_reasons=g3_reasons,
        allowed_next_phases=["train"] if not g3_reasons else [],
        extra={
            "dataset_root": dataset_summary.get("dataset_root"),
            "dataset_valid": bool(dataset_summary.get("dataset_valid", False)),
            "trainability_support_passed": bool(
                dataset_summary.get("trainability_support_passed", False)
            ),
            "claim_readiness_passed": bool(
                dataset_summary.get("claim_readiness_passed", False)
            ),
            "learning_support_unique_teacher_family_count": dataset_summary.get(
                "learning_support_unique_teacher_family_count"
            ),
            "learning_support_family_count_by_seed": dataset_summary.get(
                "learning_support_family_count_by_seed"
            ),
            "learning_support_prebridge_frame_p50": dataset_summary.get(
                "learning_support_prebridge_frame_p50"
            ),
            "attach_eligible_seed_coverage": dataset_summary.get(
                "attach_eligible_seed_coverage"
            ),
        },
    )
    g4_status = "DIAGNOSTIC_PASS" if g3.get("status") == "PASS" else "STOP"
    plan = _persist_plan_with_scope(plan, g4_status)
    docs_intent = write_docs_update_intent(
        {
            "run_instance_id": plan.get("run_instance_id"),
            "spec_doc_path": str(SPEC_DOC_PATH),
            "intent": "defer_v11_publication_until_g6",
            "allowed_after_gate": "G6",
            "publication_scope": plan.get("publication_scope"),
            "result_scope": plan.get("result_scope"),
            "claim_bearing": False,
            "canonical_doc_updates_allowed_before_g6": False,
            "requested_addendum_behavior": "v11_execution_report_only",
        }
    )
    publication_state = write_publication_state(
        {
            "run_instance_id": plan.get("run_instance_id"),
            **_scope_fields(plan),
            "selected_family_diversification_mode": selected_branch,
            "final_verdict": None,
        }
    )
    harness_state = write_harness_state(
        {
            "run_instance_id": plan.get("run_instance_id"),
            "line_a_state": "operational_freeze",
            "line_b_state": "frozen",
            "line_c_state": "frozen_vendor_baseline",
            "current_active_blocker": "support_family_collapse",
            "selected_family_diversification_mode": selected_branch,
            "trainability_support_failed_clauses": list(
                dataset_summary.get("trainability_support_failed_clauses") or []
            ),
            "claim_readiness_failed_clauses": list(
                dataset_summary.get("claim_readiness_failed_clauses") or []
            ),
        }
    )
    return {
        "plan": plan,
        "dataset_summary": dataset_summary,
        "audit": audit,
        "docs_manifest": docs_manifest,
        "sovereign_snapshot": sovereign,
        "publication_state": publication_state,
        "harness_state": harness_state,
        "docs_update_intent": docs_intent,
        "gates": {"g0": g0, "g1": g1, "g2": g2, "g3": g3},
    }


def _phase_prepare_v11(dataset_selection_mode: str = "diagnostic_learning_support") -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    artifacts = load_terminal_artifacts()
    plan = materialize_stage_plan(
        artifacts,
        active_train_state_mode="S0",
        bridge_stage="v11_support_family_diversification_prepare",
        bridge_attempt="diagnostic",
        episodes_per_seed=HONEST_EPISODES_PER_SEED,
        min_train_episodes=HONEST_MIN_TRAIN_EPISODES,
        train_steps=HONEST_TRAIN_STEPS,
        save_freq=HONEST_SAVE_FREQ,
        checkpoint_probe_steps=HONEST_PROBE_STEPS,
    )
    plan["dataset_selection_mode"] = dataset_selection_mode
    plan["authoritative_truth_field"] = "measurement_truthful_for_learning_support"
    audit = run_family_support_audit(
        LEARNING_SUPPORT_ROLLOUT_SOURCE_DIR, G1_SUPPORT_FAMILY_AUDIT_PATH
    )
    selected_branch = _support_family_branch_from_audit(audit)
    if selected_branch == "G2A":
        plan["support_family_repair_mode"] = "G2A"
        plan["learning_support_fingerprint_version"] = (
            EFFECTIVE_SUPPORT_SIGNATURE_VERSION
        )
    elif selected_branch == "G2B":
        plan["support_family_repair_mode"] = "G2B"
        plan["teacher_family_grid_version"] = TEACHER_FAMILY_GRID_VERSION
    else:
        plan["support_family_repair_mode"] = "STOP"
    write_json_atomic(TINY_RETRAIN_PLAN_PATH, plan)
    if selected_branch == "STOP":
        dataset_summary = _prepare_stop_dataset_summary(
            plan, audit, "trace_logging_insufficient"
        )
    else:
        dataset_summary = build_or_refresh_canonical_dataset(plan)
        if (
            selected_branch == "G2A"
            and not bool(dataset_summary.get("trainability_support_passed", False))
        ):
            plan["support_family_repair_mode"] = "G2B"
            plan["learning_support_fingerprint_version"] = (
                LEARNING_SUPPORT_FINGERPRINT_VERSION
            )
            plan["teacher_family_grid_version"] = TEACHER_FAMILY_GRID_VERSION
            write_json_atomic(TINY_RETRAIN_PLAN_PATH, plan)
            dataset_summary = build_or_refresh_canonical_dataset(plan)
    gate_bundle = _emit_prepare_gates_v11(plan, audit, dataset_summary)
    return gate_bundle["plan"], gate_bundle["dataset_summary"], audit, gate_bundle


def _require_v11_gate_pass(gate_id: str, gate_name: str) -> dict[str, Any]:
    gate = load_gate(gate_id, gate_name)
    if not gate:
        raise SystemExit(f"Missing {gate_id} {gate_name} gate")
    if str(gate.get("status") or "") == "STOP":
        raise SystemExit(json.dumps(gate, indent=2))
    return gate


def _write_v11_train_gate(plan: dict[str, Any], train_summary: dict[str, Any]) -> dict[str, Any]:
    status = "PASS" if bool(train_summary.get("passed", False)) else "STOP"
    return write_gate(
        "G4",
        "diagnostic_training",
        plan,
        status=status,
        blocking_reasons=[]
        if status == "PASS"
        else [str(train_summary.get("stderr_tail") or train_summary.get("error") or "train_failed")],
        allowed_next_phases=["probe"] if status == "PASS" else [],
        extra={
            **_scope_fields(plan),
            "selected_family_diversification_mode": plan.get(
                "support_family_repair_mode"
            ),
            "train_summary_path": str(G8_SUMMARY_PATH),
            "trainability_support_passed": True,
        },
    )


def _write_v11_probe_sidecars(
    plan: dict[str, Any],
    dataset_summary: dict[str, Any],
    probe_summary: dict[str, Any],
    classification: str,
) -> None:
    if not probe_summary:
        return
    attach_bridge_summary = _build_attach_bridge_summary(plan, probe_summary)
    family_conditioned_probe = _build_family_conditioned_probe(
        plan, dataset_summary, probe_summary
    )
    attach_bridge_summary["classification"] = classification
    write_json_atomic(G6_ATTACH_BRIDGE_SUMMARY_PATH, attach_bridge_summary)
    family_conditioned_probe["classification"] = classification
    family_conditioned_probe["overall_best_checkpoint_step"] = int(
        probe_summary.get("checkpoint_step") or 0
    )
    family_conditioned_probe["hard_vs_rest_summary"] = family_conditioned_probe.get(
        "hard_vs_rest"
    )
    write_json_atomic(G6_FAMILY_CONDITIONED_PROBE_PATH, family_conditioned_probe)


def _write_v11_probe_gate(
    plan: dict[str, Any],
    dataset_summary: dict[str, Any],
    probe_summary: dict[str, Any],
) -> dict[str, Any]:
    classification = _classify_probe_signal(probe_summary, plan)
    _write_v11_probe_sidecars(plan, dataset_summary, probe_summary, classification)
    probe_payload = dict(probe_summary)
    probe_payload["classification"] = classification
    write_json_atomic(G8_PROBE_PATH, probe_payload)
    return write_gate(
        "G5",
        "attach_first_probe",
        plan,
        status="PASS",
        blocking_reasons=[],
        allowed_next_phases=["finalize"],
        extra={
            **_scope_fields(plan),
            "classification": classification,
            "attach_bridge_pass": bool(probe_summary.get("attach_bridge_pass", False)),
            "ever_attach_eligible_fraction_gain": float(
                probe_summary.get("ever_attach_eligible_fraction_gain", 0.0)
            ),
            "ever_attached_rate_gain": float(
                probe_summary.get("ever_attached_rate_gain", 0.0)
            ),
            "stable_attach_gain": float(probe_summary.get("stable_attach_gain", 0.0)),
            "phase_locked_gain": float(probe_summary.get("phase_locked_gain", 0.0)),
            "strict_success_rate_gain": float(
                probe_summary.get("success_gain", 0.0)
            ),
            "selected_checkpoint_reason": probe_summary.get(
                "selected_checkpoint_reason"
            ),
        },
    )
def _write_v11_finalize_summary(
    plan: dict[str, Any],
    dataset_summary: dict[str, Any],
    train_summary: dict[str, Any],
    probe_summary: dict[str, Any],
    eval_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    g0 = load_gate("G0", "sovereign_sync")
    g1 = load_gate("G1", "family_collapse_forensic_audit")
    g2 = load_gate("G2A", "support_family_repair_branch") or load_gate(
        "G2B", "support_family_repair_branch"
    ) or load_gate("G2", "support_family_repair_branch")
    g3 = load_gate("G3", "dataset_rebuild")
    g4 = load_gate("G4", "diagnostic_training")
    g5 = load_gate("G5", "attach_first_probe")
    docs_ok, docs_drift, _ = docs_lock_consistent()
    publication_issues: list[str] = []
    if not docs_ok:
        publication_issues.append("docs_lock_drift")
    if not DOCS_UPDATE_INTENT_PATH.exists():
        publication_issues.append("docs_update_intent_missing")
    classification = _classify_probe_signal(probe_summary, plan) if probe_summary else ""
    _write_v11_probe_sidecars(plan, dataset_summary, probe_summary, classification)
    probe_summary_payload = dict(probe_summary or {})
    if classification and probe_summary_payload:
        probe_summary_payload["classification"] = classification
    selected_mode = str(plan.get("support_family_repair_mode") or "")
    if selected_mode == "STOP":
        final_verdict = "trace_logging_insufficient"
    elif str(g3.get("status") or "") == "STOP":
        final_verdict = (
            "teacher_family_grid_fake_variants"
            if selected_mode == "G2B"
            else "trace_logging_insufficient"
        )
    elif not train_summary and not probe_summary:
        final_verdict = (
            "fingerprint_underexpression_fixed_prepare_pass"
            if selected_mode == "G2A"
            else "teacher_family_grid_prepare_pass"
        )
    elif classification == "diagnostic_learning_support_signal_detected":
        final_verdict = "diagnostic_learning_support_signal_detected"
    else:
        final_verdict = "diagnostic_learning_support_no_signal_after_support_family_fix"
    summary = {
        "confirmation_mode": "tiny_retrain_completion_loop_v11",
        "run_instance_id": plan.get("run_instance_id"),
        "plan_version": plan.get("plan_version"),
        "source_base_commit": plan.get("source_base_commit"),
        "working_head_commit": plan.get("working_head_commit"),
        "source_branch": plan.get("source_branch"),
        "source_commit": plan.get("source_commit"),
        "execution_scope": plan.get("execution_scope"),
        "diagnostic_only": bool(plan.get("diagnostic_only", True)),
        "claim_bearing": False,
        "publication_scope": plan.get("publication_scope"),
        "result_scope": plan.get("result_scope"),
        "source_canonical_train_cell": plan.get("source_canonical_train_cell"),
        "source_best_train_state_mode": plan.get("source_best_train_state_mode"),
        "active_train_state_mode": plan.get("active_train_state_mode"),
        "active_state_mode_name": plan.get("active_state_mode_name"),
        "selected_family_diversification_mode": plan.get(
            "support_family_repair_mode"
        ),
        "family_forensic_audit_path": str(G1_SUPPORT_FAMILY_AUDIT_PATH),
        "final_verdict": final_verdict,
        "dataset_valid": bool(dataset_summary.get("dataset_valid", False)),
        "claim_readiness_passed": bool(dataset_summary.get("claim_readiness_passed", False)),
        "trainability_support_passed": bool(dataset_summary.get("trainability_support_passed", False)),
        "train_passed": bool(train_summary.get("passed", False)),
        "probe_classification": classification,
        "heldout_eval_run": False,
        "heldout_skip_reason": "diagnostic_only_scope",
        "publication_issues": publication_issues,
        "gate_statuses": {
            "G0": g0.get("status"),
            "G1": g1.get("status"),
            "G2": g2.get("status"),
            "G3": g3.get("status"),
            "G4": g4.get("status") if g4 else None,
            "G5": g5.get("status") if g5 else None,
        },
        "dataset_summary": dataset_summary,
        "train_summary": train_summary,
        "probe_summary": probe_summary,
        "eval_summary": eval_summary,
    }
    write_json_atomic(TINY_RETRAIN_SUMMARY_PATH, summary)
    publication_state = write_publication_state(
        {
            "run_instance_id": plan.get("run_instance_id"),
            **_scope_fields(plan),
            "selected_family_diversification_mode": selected_mode,
            "final_verdict": final_verdict,
        }
    )
    harness_state = write_harness_state(
        {
            "run_instance_id": plan.get("run_instance_id"),
            "current_active_blocker": (
                "state_conditioning_or_architecture"
                if final_verdict
                == "diagnostic_learning_support_no_signal_after_support_family_fix"
                else "support_family_collapse"
            ),
            "selected_family_diversification_mode": selected_mode,
            "trainability_support_failed_clauses": list(
                dataset_summary.get("trainability_support_failed_clauses") or []
            ),
            "final_verdict": final_verdict,
        }
    )
    g6_status = "DIAGNOSTIC_PASS" if final_verdict in {
        "fingerprint_underexpression_fixed_prepare_pass",
        "teacher_family_grid_prepare_pass",
        "diagnostic_learning_support_signal_detected",
    } else "STOP"
    g6 = write_gate(
        "G6",
        "interpretation",
        plan,
        status=g6_status,
        blocking_reasons=publication_issues
        + (
            []
            if final_verdict != "diagnostic_learning_support_no_signal_after_support_family_fix"
            else ["no_learning_signal_after_support_family_fix"]
        ),
        allowed_next_phases=[],
        extra={
            **_scope_fields(plan),
            "final_verdict": final_verdict,
            "publication_state_path": str(PUBLICATION_STATE_PATH),
            "harness_state_path": str(HARNESS_STATE_PATH),
            "docs_update_intent_path": str(DOCS_UPDATE_INTENT_PATH),
        },
    )
    return summary, g6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["prepare", "train", "probe", "eval", "finalize", "all"],
        default="all",
    )
    parser.add_argument(
        "--dataset-selection-mode",
        default="diagnostic_learning_support",
    )
    args = parser.parse_args()

    if args.phase == "prepare":
        plan, dataset_summary, audit, gate_bundle = _phase_prepare_v11(
            dataset_selection_mode=args.dataset_selection_mode
        )
        print(
            json.dumps(
                {
                    "phase": "prepare",
                    "plan": plan,
                    "dataset_summary": dataset_summary,
                    "audit": audit,
                    "gates": gate_bundle["gates"],
                },
                indent=2,
            )
        )
        return 0 if gate_bundle["gates"]["g3"].get("status") == "PASS" else 1

    plan, dataset_summary, train_summary, probe_summary, eval_summary = _load_state()

    if args.phase == "train":
        if not plan or not dataset_summary:
            plan, dataset_summary, _, gate_bundle = _phase_prepare_v11(
                dataset_selection_mode=args.dataset_selection_mode
            )
            if gate_bundle["gates"]["g3"].get("status") != "PASS":
                print(json.dumps(gate_bundle, indent=2))
                return 1
        _require_v11_gate_pass("G3", "dataset_rebuild")
        train_summary = launch_tiny_retrain(plan)
        g4 = _write_v11_train_gate(plan, train_summary)
        print(json.dumps({"phase": "train", "train_summary": train_summary, "g4": g4}, indent=2))
        return 0 if g4.get("status") == "PASS" else 1

    if args.phase == "probe":
        if not plan or not dataset_summary or not train_summary:
            raise SystemExit("Missing prepare/train artifacts before probe")
        _require_v11_gate_pass("G4", "diagnostic_training")
        probe_summary = run_train_probe(plan, train_summary)
        g5 = _write_v11_probe_gate(plan, dataset_summary, probe_summary)
        print(json.dumps({"phase": "probe", "probe_summary": probe_summary, "g5": g5}, indent=2))
        return 0

    if args.phase == "eval":
        if not plan:
            raise SystemExit("Missing prepare artifacts before eval")
        eval_summary = write_skipped_heldout_eval_summary(plan, probe_summary or {})
        print(json.dumps({"phase": "eval", "eval_summary": eval_summary}, indent=2))
        return 0

    if args.phase == "finalize":
        if not plan or not dataset_summary:
            raise SystemExit("Missing prepare artifacts before finalize")
        if not eval_summary:
            eval_summary = write_skipped_heldout_eval_summary(plan, probe_summary or {})
        summary, g6 = _write_v11_finalize_summary(
            plan,
            dataset_summary,
            train_summary,
            probe_summary,
            eval_summary,
        )
        print(json.dumps({"phase": "finalize", "summary": summary, "g6": g6}, indent=2))
        return 0 if g6.get("status") != "STOP" else 1

    plan, dataset_summary, _, gate_bundle = _phase_prepare_v11(
        dataset_selection_mode=args.dataset_selection_mode
    )
    if gate_bundle["gates"]["g3"].get("status") != "PASS":
        eval_summary = write_skipped_heldout_eval_summary(plan, {})
        summary, g6 = _write_v11_finalize_summary(plan, dataset_summary, {}, {}, eval_summary)
        print(json.dumps({"phase": "all", "summary": summary, "g6": g6}, indent=2))
        return 1

    train_summary = launch_tiny_retrain(plan)
    g4 = _write_v11_train_gate(plan, train_summary)
    if g4.get("status") != "PASS":
        eval_summary = write_skipped_heldout_eval_summary(plan, {})
        summary, g6 = _write_v11_finalize_summary(plan, dataset_summary, train_summary, {}, eval_summary)
        print(json.dumps({"phase": "all", "summary": summary, "g6": g6}, indent=2))
        return 1
    probe_summary = run_train_probe(plan, train_summary)
    _write_v11_probe_gate(plan, dataset_summary, probe_summary)
    eval_summary = write_skipped_heldout_eval_summary(plan, probe_summary)
    summary, g6 = _write_v11_finalize_summary(
        plan, dataset_summary, train_summary, probe_summary, eval_summary
    )
    print(json.dumps({"phase": "all", "summary": summary, "g6": g6}, indent=2))
    return 0 if g6.get("status") != "STOP" else 1


if __name__ == "__main__":
    raise SystemExit(main())

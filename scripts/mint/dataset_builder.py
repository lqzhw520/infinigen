#!/usr/bin/env python3
"""Utilities for packing robot rollout NPZ files into LeRobot datasets."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRUTH_CONTRACT_PATH = PROJECT_ROOT / "docs" / "contracts" / "truth_contract_v84.json"

ACTION_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_rx",
    "delta_ry",
    "delta_rz",
    "gripper_command",
]

LIBERO_KEY_RENAME = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.image2",
    "state": "observation.state",
    "actions": "action",
}

STATE_MODE_ALIAS = {
    "S0": "m0_proxy",
    "S1": "telemetry_candidate_v3_transition",
    "S2": "telemetry_candidate_v4_task_identity",
}

FALLBACK_STATE_DIM_NAMES = {
    "m0_proxy": [
        "eef_pos_x",
        "eef_pos_y",
        "eef_pos_z",
        "handle_rel_x_norm",
        "handle_rel_y_norm",
        "handle_rel_z_norm",
        "drawer_fraction_signed",
        "gripper_joint",
    ],
    "telemetry_candidate_v4_task_identity": [
        "handle_rel_x_norm",
        "handle_rel_y_norm",
        "handle_rel_z_norm",
        "drawer_fraction_signed",
        "pull_alignment_cos",
        "grasp_slip_norm",
        "effective_pull_progress_norm",
        "phase_locked",
    ],
}


def _plan_source_canonical_train_cell(plan: dict[str, Any]) -> str:
    return str(
        plan.get("source_canonical_train_cell")
        or plan.get("canonical_train_cell")
        or ""
    )


def _plan_source_best_train_state_mode(plan: dict[str, Any]) -> str:
    return str(
        plan.get("source_best_train_state_mode")
        or plan.get("best_train_state_mode")
        or ""
    )


def _plan_active_train_state_mode(plan: dict[str, Any]) -> str:
    return str(
        plan.get("active_train_state_mode") or _plan_source_best_train_state_mode(plan)
    )


def _plan_active_state_mode_name(plan: dict[str, Any]) -> str:
    active = _plan_active_train_state_mode(plan)
    return str(
        plan.get("active_state_mode_name") or STATE_MODE_ALIAS.get(active, active)
    )


def _plan_dataset_selection_mode(plan: dict[str, Any]) -> str:
    return str(plan.get("dataset_selection_mode") or "claim_canonical")


def _learning_support_family_key(rec: dict[str, Any]) -> str:
    return str(
        rec.get("effective_support_signature_v1")
        or rec.get("learning_support_fingerprint")
        or ""
    )


def _orientation_support_family_key(rec: dict[str, Any]) -> str:
    return str(rec.get("orientation_support_fingerprint") or "")


def _state_dim_names(meta: dict[str, Any]) -> list[str]:
    state_spec = meta.get("state_spec") or {}
    dim_names = [str(item) for item in (state_spec.get("dim_names") or []) if str(item)]
    if dim_names:
        return dim_names
    state_mode = str(
        state_spec.get("state_mode")
        or meta.get("state_mode")
        or meta.get("active_state_mode_name")
        or ""
    )
    if state_mode in FALLBACK_STATE_DIM_NAMES:
        return list(FALLBACK_STATE_DIM_NAMES[state_mode])
    raise ValueError(f"Missing state_spec.dim_names for state_mode={state_mode!r}")


def _nested_rollout_value(meta: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in meta and meta.get(key) is not None:
        return meta.get(key)
    state_spec = meta.get("state_spec") or {}
    contract_config = meta.get("contract_config") or {}
    handle_probe = meta.get("handle_probe_metadata") or {}
    orientation = meta.get("orientation_telemetry") or {}
    training_truth = meta.get("canonical_training_truth") or {}
    if key == "state_mode":
        return state_spec.get("state_mode", contract_config.get("state_mode", default))
    if key == "interaction_mode":
        return contract_config.get("interaction_mode", default)
    if key == "measurement_truthful":
        if meta.get("measurement_truthful_for_training") is not None:
            return meta.get("measurement_truthful_for_training")
        if training_truth.get("measurement_truthful_for_training") is not None:
            return training_truth.get("measurement_truthful_for_training")
        return handle_probe.get("measurement_truthful", default)
    if key == "measurement_truthful_for_training":
        if meta.get("measurement_truthful_for_training") is not None:
            return meta.get("measurement_truthful_for_training")
        return training_truth.get("measurement_truthful_for_training", default)
    if key == "teacher_truth_adjudication":
        if meta.get("teacher_truth_adjudication") is not None:
            return meta.get("teacher_truth_adjudication")
        return training_truth.get("teacher_truth_adjudication", default)
    if key == "teacher_truthful_window_frame_count":
        if meta.get("teacher_truthful_window_frame_count") is not None:
            return meta.get("teacher_truthful_window_frame_count")
        return training_truth.get("truthful_window_frame_count", default)
    if key == "truthful_window_ratio":
        if meta.get("truthful_window_ratio") is not None:
            return meta.get("truthful_window_ratio")
        return training_truth.get("truthful_window_ratio", default)
    if key == "truthful_window_longest_interior_gap":
        if meta.get("truthful_window_longest_interior_gap") is not None:
            return meta.get("truthful_window_longest_interior_gap")
        return training_truth.get("truthful_window_longest_interior_gap", default)
    if key == "truthful_window_tail_truthful_count_last6":
        if meta.get("truthful_window_tail_truthful_count_last6") is not None:
            return meta.get("truthful_window_tail_truthful_count_last6")
        return training_truth.get("truthful_window_tail_truthful_count_last6", default)
    if key == "bridge_in_truthful_window":
        if meta.get("bridge_in_truthful_window") is not None:
            return meta.get("bridge_in_truthful_window")
        return training_truth.get("bridge_in_truthful_window", default)
    if key == "measurement_truth_tier":
        return handle_probe.get("measurement_truth_tier", default)
    if key == "measurement_backend":
        return handle_probe.get("measurement_backend", default)
    if key == "measurement_verifier":
        return handle_probe.get("measurement_verifier", default)
    if key == "runtime_visible_handle_mapping_source":
        return handle_probe.get("runtime_visible_handle_mapping_source", default)
    if key == "runtime_handle_anchor_valid":
        if meta.get("runtime_handle_anchor_valid") is not None:
            return meta.get("runtime_handle_anchor_valid")
        return orientation.get("runtime_handle_anchor_valid", default)
    return default


def validate_rollout_for_canonical_training(
    meta: dict[str, Any], plan: dict[str, Any]
) -> tuple[bool, str]:
    expected_source_cell = _plan_source_canonical_train_cell(plan)
    expected_transition = str(plan.get("best_transition_cell"))
    expected_source_state = _plan_source_best_train_state_mode(plan)
    expected_active_state = _plan_active_train_state_mode(plan)
    expected_active_state_name = _plan_active_state_mode_name(plan)
    expected_run_instance_id = str(plan.get("run_instance_id") or "")
    expected_plan_version = str(plan.get("plan_version") or "")
    expected_source_base_commit = str(plan.get("source_base_commit") or "")
    expected_teacher_truth_gate = str(plan.get("teacher_truth_gate") or "")
    seed = meta.get("seed")
    strict_metrics = meta.get("strict_metrics") or {}
    teacher_episode_class_raw = meta.get("teacher_episode_class")
    if teacher_episode_class_raw is None or not str(teacher_episode_class_raw).strip():
        return False, "teacher_episode_class_missing"
    teacher_episode_class = str(teacher_episode_class_raw)
    teacher_fingerprint_raw = meta.get("teacher_fingerprint")
    if teacher_fingerprint_raw is None or not str(teacher_fingerprint_raw).strip():
        return False, "teacher_fingerprint_missing"
    run_instance_id = meta.get("run_instance_id")
    if expected_run_instance_id and (
        run_instance_id is None or not str(run_instance_id).strip()
    ):
        return False, "run_instance_id_missing"
    checks = [
        (
            (not expected_run_instance_id)
            or str(run_instance_id) == expected_run_instance_id,
            "run_instance_id_mismatch",
        ),
        (
            (not expected_plan_version)
            or str(meta.get("plan_version") or "") == expected_plan_version,
            "plan_version_mismatch",
        ),
        (
            (not expected_source_base_commit)
            or str(meta.get("source_base_commit") or "") == expected_source_base_commit,
            "source_base_commit_mismatch",
        ),
        (
            str(
                meta.get("source_canonical_train_cell")
                or meta.get("canonical_train_cell")
            )
            == expected_source_cell,
            "source_canonical_train_cell_mismatch",
        ),
        (
            str(meta.get("best_transition_cell")) == expected_transition,
            "best_transition_cell_mismatch",
        ),
        (
            str(
                meta.get("source_best_train_state_mode")
                or meta.get("best_train_state_mode")
            )
            == expected_source_state,
            "source_best_train_state_mode_mismatch",
        ),
        (
            str(
                meta.get("active_train_state_mode") or meta.get("best_train_state_mode")
            )
            == expected_active_state,
            "active_train_state_mode_mismatch",
        ),
        (
            str(meta.get("selector_mode")) == str(plan.get("selector_mode")),
            "selector_mode_mismatch",
        ),
        (
            bool(
                _nested_rollout_value(meta, "measurement_truthful_for_training", False)
            )
            is True,
            "measurement_not_truthful",
        ),
        (
            str(_nested_rollout_value(meta, "teacher_truth_adjudication", ""))
            == "teacher_window_truth_v84",
            "teacher_truth_adjudication_mismatch",
        ),
        (
            (not expected_teacher_truth_gate)
            or str(
                meta.get("teacher_truth_gate")
                or _nested_rollout_value(meta, "teacher_truth_adjudication", "")
            )
            == expected_teacher_truth_gate,
            "teacher_truth_gate_mismatch",
        ),
        (
            int(
                _nested_rollout_value(meta, "teacher_truthful_window_frame_count", 0)
                or 0
            )
            >= 16,
            "teacher_truthful_window_too_short",
        ),
        (
            float(_nested_rollout_value(meta, "truthful_window_ratio", 0.0) or 0.0)
            >= 0.80,
            "truthful_window_ratio_too_low",
        ),
        (
            int(
                _nested_rollout_value(meta, "truthful_window_longest_interior_gap", 0)
                or 0
            )
            <= 2,
            "truthful_window_gap_too_long",
        ),
        (
            int(
                _nested_rollout_value(
                    meta, "truthful_window_tail_truthful_count_last6", 0
                )
                or 0
            )
            >= 5,
            "truthful_window_tail_not_stable",
        ),
        (
            bool(_nested_rollout_value(meta, "bridge_in_truthful_window", False))
            is True,
            "bridge_not_in_truthful_window",
        ),
        (
            str(_nested_rollout_value(meta, "measurement_backend", ""))
            == str(plan.get("measurement_backend")),
            "measurement_backend_mismatch",
        ),
        (
            str(_nested_rollout_value(meta, "measurement_verifier", ""))
            == str(plan.get("measurement_verifier")),
            "measurement_verifier_mismatch",
        ),
        (
            str(
                _nested_rollout_value(meta, "runtime_visible_handle_mapping_source", "")
            )
            == str(plan.get("runtime_visible_handle_mapping_source")),
            "runtime_visible_handle_mapping_source_mismatch",
        ),
        (
            bool(_nested_rollout_value(meta, "runtime_handle_anchor_valid", False))
            is True,
            "runtime_handle_anchor_invalid",
        ),
        (
            str(_nested_rollout_value(meta, "interaction_mode", ""))
            == "orientation_sensitive_v3_task_identity_locked",
            "interaction_mode_mismatch",
        ),
        (
            str(_nested_rollout_value(meta, "state_mode", ""))
            == expected_active_state_name,
            "state_mode_mismatch",
        ),
        (
            int(seed) in [int(x) for x in plan.get("train_seeds", [])]
            if seed is not None
            else False,
            "seed_not_in_train_split",
        ),
        (
            int(seed) not in [int(x) for x in plan.get("heldout_seeds", [])]
            if seed is not None
            else False,
            "seed_in_heldout_split",
        ),
        (
            teacher_episode_class in {"strict_teacher", "near_strict_teacher"},
            "teacher_episode_class_not_accepted",
        ),
        (
            bool(strict_metrics.get("strict_success"))
            or teacher_episode_class == "near_strict_teacher",
            "not_strict_or_near_strict_teacher",
        ),
    ]
    for passed, reason in checks:
        if not passed:
            return False, reason
    return True, "ok"


def validate_rollout_for_learning_support_training(
    meta: dict[str, Any], plan: dict[str, Any]
) -> tuple[bool, str]:
    expected_source_cell = _plan_source_canonical_train_cell(plan)
    expected_transition = str(plan.get("best_transition_cell"))
    expected_source_state = _plan_source_best_train_state_mode(plan)
    expected_active_state = _plan_active_train_state_mode(plan)
    expected_run_instance_id = str(plan.get("run_instance_id") or "")
    expected_plan_version = str(plan.get("plan_version") or "")
    expected_source_base_commit = str(plan.get("source_base_commit") or "")
    expected_teacher_truth_gate = str(plan.get("teacher_truth_gate") or "")
    learning_support_teacher_class_raw = meta.get("learning_support_teacher_class")
    if learning_support_teacher_class_raw is None or not str(
        learning_support_teacher_class_raw
    ).strip():
        return False, "learning_support_teacher_class_missing"
    learning_support_teacher_class = str(learning_support_teacher_class_raw)
    learning_support_fingerprint_raw = meta.get("learning_support_fingerprint")
    if learning_support_fingerprint_raw is None or not str(
        learning_support_fingerprint_raw
    ).strip():
        return False, "learning_support_fingerprint_missing"
    effective_support_signature_raw = meta.get("effective_support_signature_v1")
    if effective_support_signature_raw is None or not str(
        effective_support_signature_raw
    ).strip():
        return False, "effective_support_signature_missing"
    run_instance_id = meta.get("run_instance_id")
    if expected_run_instance_id and (
        run_instance_id is None or not str(run_instance_id).strip()
    ):
        return False, "run_instance_id_missing"
    checks = [
        (
            (not expected_run_instance_id)
            or str(run_instance_id) == expected_run_instance_id,
            "run_instance_id_mismatch",
        ),
        (
            (not expected_plan_version)
            or str(meta.get("plan_version") or "") == expected_plan_version,
            "plan_version_mismatch",
        ),
        (
            (not expected_source_base_commit)
            or str(meta.get("source_base_commit") or "") == expected_source_base_commit,
            "source_base_commit_mismatch",
        ),
        (
            str(
                meta.get("source_canonical_train_cell")
                or meta.get("canonical_train_cell")
            )
            == expected_source_cell,
            "source_canonical_train_cell_mismatch",
        ),
        (
            str(meta.get("best_transition_cell")) == expected_transition,
            "best_transition_cell_mismatch",
        ),
        (
            str(
                meta.get("source_best_train_state_mode")
                or meta.get("best_train_state_mode")
            )
            == expected_source_state,
            "source_best_train_state_mode_mismatch",
        ),
        (
            str(
                meta.get("active_train_state_mode") or meta.get("best_train_state_mode")
            )
            == expected_active_state,
            "active_train_state_mode_mismatch",
        ),
        (
            str(meta.get("selector_mode")) == str(plan.get("selector_mode")),
            "selector_mode_mismatch",
        ),
        (
            bool(meta.get("measurement_truthful_for_learning_support", False)) is True,
            "learning_support_not_truthful",
        ),
        (
            str(meta.get("learning_support_truth_adjudication") or "")
            == "learning_support_window_v1",
            "learning_support_truth_adjudication_mismatch",
        ),
        (
            (not expected_teacher_truth_gate)
            or str(meta.get("teacher_truth_gate") or "") == expected_teacher_truth_gate,
            "teacher_truth_gate_mismatch",
        ),
        (
            int(meta.get("learning_support_window_len", 0) or 0) >= 24,
            "learning_support_window_too_short",
        ),
        (
            int(meta.get("learning_support_prebridge_frame_count", 0) or 0) >= 8,
            "learning_support_prebridge_too_short",
        ),
        (
            bool(meta.get("learning_support_contains_prebridge", False)) is True,
            "learning_support_missing_prebridge",
        ),
        (
            bool(meta.get("learning_support_contains_bridge", False)) is True,
            "learning_support_missing_bridge",
        ),
        (
            float(meta.get("learning_support_whole_window_truthful_ratio", 0.0) or 0.0)
            >= 0.60,
            "learning_support_whole_window_truth_ratio_too_low",
        ),
        (
            float(meta.get("learning_support_bridge_truthful_ratio", 0.0) or 0.0)
            >= 0.80,
            "learning_support_bridge_truth_ratio_too_low",
        ),
        (
            int(meta.get("learning_support_bridge_longest_interior_gap", 0) or 0) <= 2,
            "learning_support_bridge_gap_too_long",
        ),
        (
            int(meta.get("learning_support_bridge_tail_truthful_count_last6", 0) or 0)
            >= 5,
            "learning_support_bridge_tail_not_stable",
        ),
        (
            bool(meta.get("learning_support_interval_mask_empty", False)) is False,
            "learning_support_interval_mask_empty",
        ),
        (
            bool(meta.get("learning_support_interval_anchor_invalid", False)) is False,
            "learning_support_interval_anchor_invalid",
        ),
        (
            str(meta.get("measurement_backend") or "")
            == str(plan.get("measurement_backend")),
            "measurement_backend_mismatch",
        ),
        (
            str(meta.get("measurement_verifier") or "")
            == str(plan.get("measurement_verifier")),
            "measurement_verifier_mismatch",
        ),
        (
            str(meta.get("runtime_visible_handle_mapping_source") or "")
            == str(plan.get("runtime_visible_handle_mapping_source")),
            "runtime_visible_handle_mapping_source_mismatch",
        ),
        (
            bool(meta.get("runtime_handle_anchor_valid", False)) is True,
            "runtime_handle_anchor_invalid",
        ),
        (
            meta.get("seed") is not None
            and int(meta.get("seed")) in [int(x) for x in plan.get("train_seeds", [])],
            "seed_not_in_train_split",
        ),
        (
            int(meta.get("seed")) not in [int(x) for x in plan.get("heldout_seeds", [])]
            if meta.get("seed") is not None
            else False,
            "seed_in_heldout_split",
        ),
        (
            learning_support_teacher_class
            in {"strict_teacher", "near_strict_teacher", "learning_support_teacher"},
            "learning_support_teacher_class_not_accepted",
        ),
    ]
    for passed, reason in checks:
        if not passed:
            return False, reason
    return True, "ok"




def validate_rollout_for_orientation_support_training(
    meta: dict[str, Any], plan: dict[str, Any]
) -> tuple[bool, str]:
    expected_source_cell = _plan_source_canonical_train_cell(plan)
    expected_transition = str(plan.get("best_transition_cell"))
    expected_source_state = _plan_source_best_train_state_mode(plan)
    expected_active_state = _plan_active_train_state_mode(plan)
    expected_run_instance_id = str(plan.get("run_instance_id") or "")
    expected_plan_version = str(plan.get("plan_version") or "")
    expected_source_base_commit = str(plan.get("source_base_commit") or "")
    expected_teacher_truth_gate = str(plan.get("teacher_truth_gate") or "")
    orientation_teacher_class_raw = meta.get("orientation_support_teacher_class")
    if orientation_teacher_class_raw is None or not str(orientation_teacher_class_raw).strip():
        return False, "orientation_support_teacher_class_missing"
    orientation_teacher_class = str(orientation_teacher_class_raw)
    orientation_fingerprint_raw = meta.get("orientation_support_fingerprint")
    if orientation_fingerprint_raw is None or not str(orientation_fingerprint_raw).strip():
        return False, "orientation_support_fingerprint_missing"
    run_instance_id = meta.get("run_instance_id")
    if expected_run_instance_id and (run_instance_id is None or not str(run_instance_id).strip()):
        return False, "run_instance_id_missing"
    checks = [
        (
            (not expected_run_instance_id)
            or str(run_instance_id) == expected_run_instance_id,
            "run_instance_id_mismatch",
        ),
        (
            (not expected_plan_version)
            or str(meta.get("plan_version") or "") == expected_plan_version,
            "plan_version_mismatch",
        ),
        (
            (not expected_source_base_commit)
            or str(meta.get("source_base_commit") or "") == expected_source_base_commit,
            "source_base_commit_mismatch",
        ),
        (
            str(
                meta.get("source_canonical_train_cell")
                or meta.get("canonical_train_cell")
            )
            == expected_source_cell,
            "source_canonical_train_cell_mismatch",
        ),
        (
            str(meta.get("best_transition_cell")) == expected_transition,
            "best_transition_cell_mismatch",
        ),
        (
            str(
                meta.get("source_best_train_state_mode")
                or meta.get("best_train_state_mode")
            )
            == expected_source_state,
            "source_best_train_state_mode_mismatch",
        ),
        (
            str(
                meta.get("active_train_state_mode") or meta.get("best_train_state_mode")
            )
            == expected_active_state,
            "active_train_state_mode_mismatch",
        ),
        (
            str(meta.get("selector_mode")) == str(plan.get("selector_mode")),
            "selector_mode_mismatch",
        ),
        (
            bool(meta.get("measurement_truthful_for_orientation_support", False)) is True,
            "orientation_support_not_truthful",
        ),
        (
            str(meta.get("orientation_support_truth_adjudication") or "")
            == "orientation_support_window_v1",
            "orientation_support_truth_adjudication_mismatch",
        ),
        (
            (not expected_teacher_truth_gate)
            or str(meta.get("teacher_truth_gate") or "") == expected_teacher_truth_gate,
            "teacher_truth_gate_mismatch",
        ),
        (
            int(meta.get("orientation_support_window_len", 0) or 0) >= 16,
            "orientation_support_window_too_short",
        ),
        (
            bool(meta.get("orientation_support_contains_distance_pass", False)) is True,
            "orientation_support_missing_distance_pass",
        ),
        (
            bool(meta.get("orientation_support_contains_approach_pass", False)) is True,
            "orientation_support_missing_approach_pass",
        ),
        (
            float(meta.get("orientation_support_truthful_ratio", 0.0) or 0.0) >= 0.60,
            "orientation_support_truth_ratio_too_low",
        ),
        (
            bool(meta.get("orientation_support_anchor_valid", False)) is True,
            "orientation_support_anchor_invalid",
        ),
        (
            str(meta.get("measurement_backend") or "")
            == str(plan.get("measurement_backend")),
            "measurement_backend_mismatch",
        ),
        (
            str(meta.get("measurement_verifier") or "")
            == str(plan.get("measurement_verifier")),
            "measurement_verifier_mismatch",
        ),
        (
            str(meta.get("runtime_visible_handle_mapping_source") or "")
            == str(plan.get("runtime_visible_handle_mapping_source")),
            "runtime_visible_handle_mapping_source_mismatch",
        ),
        (
            bool(meta.get("runtime_handle_anchor_valid", False)) is True,
            "runtime_handle_anchor_invalid",
        ),
        (
            meta.get("seed") is not None
            and int(meta.get("seed")) in [int(x) for x in plan.get("train_seeds", [])],
            "seed_not_in_train_split",
        ),
        (
            int(meta.get("seed")) not in [int(x) for x in plan.get("heldout_seeds", [])]
            if meta.get("seed") is not None
            else False,
            "seed_in_heldout_split",
        ),
        (
            orientation_teacher_class
            in {
                "orientation_transition_teacher",
                "attach_eligible_transition_teacher",
                "orientation_context_teacher",
            },
            "orientation_support_teacher_class_not_accepted",
        ),
    ]
    for passed, reason in checks:
        if not passed:
            return False, reason
    return True, "ok"


def validate_built_dataset_provenance(
    dataset_root: Path, plan: dict[str, Any]
) -> dict[str, Any]:
    dataset_selection_mode = _plan_dataset_selection_mode(plan)
    provenance_path = dataset_root / "meta" / "provenance.json"
    report = {
        "dataset_root": str(dataset_root),
        "dataset_selection_mode": dataset_selection_mode,
        "truth_contract_path": str(TRUTH_CONTRACT_PATH),
        "dataset_repo_id": plan.get("dataset_repo_id"),
        "source_canonical_train_cell": _plan_source_canonical_train_cell(plan),
        "source_best_train_state_mode": _plan_source_best_train_state_mode(plan),
        "active_train_state_mode": _plan_active_train_state_mode(plan),
        "active_state_mode_name": _plan_active_state_mode_name(plan),
        "run_instance_id": plan.get("run_instance_id"),
        "plan_version": plan.get("plan_version"),
        "source_base_commit": plan.get("source_base_commit"),
        "train_seed_count": len(plan.get("train_seeds", [])),
        "used_successful_seed_count": 0,
        "used_rollout_count": 0,
        "effective_frame_count": 0,
        "rejected_rollout_count": 0,
        "provenance_hash": None,
        "dataset_valid": False,
        "errors": [],
    }
    if not provenance_path.exists():
        report["errors"].append("provenance_missing")
        return report
    raw = provenance_path.read_bytes()
    report["provenance_hash"] = hashlib.sha256(raw).hexdigest()
    provenance = json.loads(raw.decode("utf-8"))
    records = provenance.get("records", [])
    train_seeds = {int(x) for x in plan.get("train_seeds", [])}
    heldout_seeds = {int(x) for x in plan.get("heldout_seeds", [])}
    unique_successful_seeds = {
        int(x) for x in provenance.get("unique_successful_seeds", [])
    }
    source_cells = {
        rec.get("source_canonical_train_cell") or rec.get("canonical_train_cell")
        for rec in records
        if rec.get("source_canonical_train_cell") or rec.get("canonical_train_cell")
    }
    transition_values = {
        rec.get("best_transition_cell")
        for rec in records
        if rec.get("best_transition_cell") is not None
    }
    source_state_values = {
        rec.get("source_best_train_state_mode") or rec.get("best_train_state_mode")
        for rec in records
        if rec.get("source_best_train_state_mode") or rec.get("best_train_state_mode")
    }
    active_state_values = {
        rec.get("active_train_state_mode")
        for rec in records
        if rec.get("active_train_state_mode") is not None
    }
    state_mode_values = {
        rec.get("state_mode") for rec in records if rec.get("state_mode") is not None
    }
    run_instance_values = {
        rec.get("run_instance_id") for rec in records if rec.get("run_instance_id")
    }
    plan_version_values = {
        rec.get("plan_version") for rec in records if rec.get("plan_version")
    }
    source_base_commit_values = {
        rec.get("source_base_commit")
        for rec in records
        if rec.get("source_base_commit")
    }
    missing_run_instance_count = int(
        sum(1 for rec in records if not rec.get("run_instance_id"))
    )
    missing_plan_version_count = int(
        sum(1 for rec in records if not rec.get("plan_version"))
    )
    missing_source_base_commit_count = int(
        sum(1 for rec in records if not rec.get("source_base_commit"))
    )
    state_dim_signatures = {
        tuple(rec.get("state_dim_names") or [])
        for rec in records
        if rec.get("state_dim_names")
    }
    measurement_backends = {
        rec.get("measurement_backend")
        for rec in records
        if rec.get("measurement_backend") is not None
    }
    measurement_verifiers = {
        rec.get("measurement_verifier")
        for rec in records
        if rec.get("measurement_verifier") is not None
    }
    runtime_anchor_values = [
        bool(rec.get("runtime_handle_anchor_valid", False)) for rec in records
    ]
    training_truth_flags = [
        bool(rec.get("measurement_truthful_for_training", False)) for rec in records
    ]
    bridge_truth_flags = [
        bool(rec.get("bridge_in_truthful_window", False)) for rec in records
    ]
    teacher_truth_adjudications = {
        rec.get("teacher_truth_adjudication")
        for rec in records
        if rec.get("teacher_truth_adjudication")
    }
    truth_contract_hashes = {
        rec.get("truth_contract_hash")
        for rec in records
        if rec.get("truth_contract_hash")
    }
    truth_contract_roots = {
        rec.get("truth_contract_root_object")
        for rec in records
        if rec.get("truth_contract_root_object")
    }
    teacher_episode_classes = [
        str(rec.get("teacher_episode_class", "rejected_teacher")) for rec in records
    ]
    learning_support_teacher_classes = [
        str(rec.get("learning_support_teacher_class", "rejected_teacher"))
        for rec in records
    ]
    orientation_support_teacher_classes = [
        str(rec.get("orientation_support_teacher_class", "rejected_orientation_teacher"))
        for rec in records
    ]
    teacher_fingerprints = [
        str(rec.get("teacher_fingerprint", ""))
        for rec in records
        if rec.get("teacher_fingerprint")
    ]
    learning_support_fingerprints = [
        _learning_support_family_key(rec)
        for rec in records
        if _learning_support_family_key(rec)
    ]
    orientation_support_fingerprints = [
        _orientation_support_family_key(rec)
        for rec in records
        if _orientation_support_family_key(rec)
    ]
    unique_teacher_families = {fp for fp in teacher_fingerprints if fp}
    unique_learning_support_families = {
        fp for fp in learning_support_fingerprints if fp
    }
    unique_orientation_support_families = {
        fp for fp in orientation_support_fingerprints if fp
    }
    accepted_records_missing_fingerprint = int(
        sum(
            1
            for rec in records
            if rec.get("teacher_episode_class")
            in {"strict_teacher", "near_strict_teacher"}
            and not rec.get("teacher_fingerprint")
        )
    )
    accepted_family_count_by_seed = dict(
        provenance.get("accepted_family_count_by_seed") or {}
    )
    strict_family_count_by_seed = dict(
        provenance.get("strict_family_count_by_seed") or {}
    )
    near_strict_family_count_by_seed = dict(
        provenance.get("near_strict_family_count_by_seed") or {}
    )
    teacher_fingerprint_examples_by_seed = dict(
        provenance.get("teacher_fingerprint_examples_by_seed") or {}
    )
    learning_support_family_count_by_seed = dict(
        provenance.get("learning_support_family_count_by_seed") or {}
    )
    orientation_support_family_count_by_seed = dict(
        provenance.get("orientation_support_family_count_by_seed") or {}
    )
    family_collapse_suspected = bool(
        provenance.get("family_collapse_suspected", False)
    )
    family_diversity_notes = list(provenance.get("family_diversity_notes") or [])
    learning_support_teacher_class_counts = dict(
        Counter(learning_support_teacher_classes)
    )
    orientation_support_teacher_class_counts = dict(
        Counter(orientation_support_teacher_classes)
    )
    learning_support_unique_teacher_family_count = int(
        provenance.get("learning_support_unique_teacher_family_count", 0)
        or len(unique_learning_support_families)
    )
    strict_family_count = len(
        {
            rec.get("teacher_fingerprint")
            for rec in records
            if rec.get("teacher_episode_class") == "strict_teacher"
            and rec.get("teacher_fingerprint")
        }
    )
    near_strict_family_count = len(
        {
            rec.get("teacher_fingerprint")
            for rec in records
            if rec.get("teacher_episode_class") == "near_strict_teacher"
            and rec.get("teacher_fingerprint")
        }
    )
    accepted_seed_coverage = sorted(
        {
            int(rec.get("seed"))
            for rec in records
            if rec.get("seed") is not None
            and rec.get("teacher_episode_class")
            in {"strict_teacher", "near_strict_teacher"}
        }
    )
    strict_seed_coverage = sorted(
        {
            int(rec.get("seed"))
            for rec in records
            if rec.get("seed") is not None
            and rec.get("teacher_episode_class") == "strict_teacher"
        }
    )
    near_strict_seed_coverage = sorted(
        {
            int(rec.get("seed"))
            for rec in records
            if rec.get("seed") is not None
            and rec.get("teacher_episode_class") == "near_strict_teacher"
        }
    )
    learning_support_seed_coverage = sorted(
        {
            int(rec.get("seed"))
            for rec in records
            if rec.get("seed") is not None
            and rec.get("learning_support_teacher_class")
            in {"strict_teacher", "near_strict_teacher", "learning_support_teacher"}
        }
    )
    orientation_support_seed_coverage = sorted(
        {
            int(rec.get("seed"))
            for rec in records
            if rec.get("seed") is not None
            and rec.get("orientation_support_teacher_class")
            in {
                "orientation_transition_teacher",
                "attach_eligible_transition_teacher",
                "orientation_context_teacher",
            }
        }
    )
    attach_eligible_seed_coverage = sorted(
        {
            int(rec.get("seed"))
            for rec in records
            if rec.get("seed") is not None
            and rec.get("first_attach_eligible_step") is not None
            and rec.get("learning_support_teacher_class")
            in {"strict_teacher", "near_strict_teacher", "learning_support_teacher"}
        }
    )
    truth_window_ratios = [
        float(rec.get("truthful_window_ratio", 0.0) or 0.0) for rec in records
    ]
    learning_support_prebridge_counts = [
        int(rec.get("learning_support_prebridge_frame_count", 0) or 0)
        for rec in records
        if rec.get("learning_support_teacher_class")
        in {"strict_teacher", "near_strict_teacher", "learning_support_teacher"}
    ]
    learning_support_window_lengths = [
        int(rec.get("learning_support_window_len", 0) or 0)
        for rec in records
        if rec.get("learning_support_teacher_class")
        in {"strict_teacher", "near_strict_teacher", "learning_support_teacher"}
    ]
    orientation_support_truthful_ratios = [
        float(rec.get("orientation_support_truthful_ratio", 0.0) or 0.0)
        for rec in records
        if rec.get("orientation_support_teacher_class")
        in {
            "orientation_transition_teacher",
            "attach_eligible_transition_teacher",
            "orientation_context_teacher",
        }
    ]
    orientation_context_frame_counts = [
        int(rec.get("orientation_context_frame_count", 0) or 0)
        for rec in records
        if rec.get("orientation_support_teacher_class")
        in {
            "orientation_transition_teacher",
            "attach_eligible_transition_teacher",
            "orientation_context_teacher",
        }
    ]
    runtime_anchor_valid_rate = (
        float(sum(1 for x in runtime_anchor_values if x) / len(runtime_anchor_values))
        if runtime_anchor_values
        else 0.0
    )
    if len(truth_contract_hashes) != 1:
        report["errors"].append("truth_contract_hash_mismatch")
    report.update(
        {
            "used_successful_seed_count": len(unique_successful_seeds),
            "used_rollout_count": len(records),
            "effective_frame_count": int(
                provenance.get("effective_frame_count", 0) or 0
            ),
            "unique_successful_seeds": sorted(unique_successful_seeds),
            "run_instance_values": sorted(
                str(x) for x in run_instance_values if x is not None
            ),
            "plan_version_values": sorted(
                str(x) for x in plan_version_values if x is not None
            ),
            "source_base_commit_values": sorted(
                str(x) for x in source_base_commit_values if x is not None
            ),
            "missing_run_instance_count": missing_run_instance_count,
            "missing_plan_version_count": missing_plan_version_count,
            "missing_source_base_commit_count": missing_source_base_commit_count,
            "claim_policies": provenance.get("claim_policies", []),
            "state_modes": provenance.get("state_modes", []),
            "state_dim_names": provenance.get("state_dim_names", []),
            "state_dim_signatures": [list(sig) for sig in sorted(state_dim_signatures)],
            "teacher_truth_adjudications": sorted(
                str(x) for x in teacher_truth_adjudications if x is not None
            ),
            "teacher_episode_class_counts": dict(Counter(teacher_episode_classes)),
            "learning_support_teacher_class_counts": learning_support_teacher_class_counts,
            "orientation_support_teacher_class_counts": orientation_support_teacher_class_counts,
            "accepted_unique_teacher_family_count": len(unique_teacher_families),
            "learning_support_unique_teacher_family_count": (
                learning_support_unique_teacher_family_count
            ),
            "orientation_support_unique_teacher_family_count": int(
                provenance.get("orientation_support_unique_teacher_family_count", 0)
                or len(unique_orientation_support_families)
            ),
            "strict_unique_teacher_family_count": strict_family_count,
            "near_strict_unique_teacher_family_count": near_strict_family_count,
            "accepted_family_count_by_seed": accepted_family_count_by_seed,
            "strict_family_count_by_seed": strict_family_count_by_seed,
            "near_strict_family_count_by_seed": near_strict_family_count_by_seed,
            "learning_support_family_count_by_seed": (
                learning_support_family_count_by_seed
            ),
            "orientation_support_family_count_by_seed": (
                orientation_support_family_count_by_seed
            ),
            "teacher_fingerprint_examples_by_seed": teacher_fingerprint_examples_by_seed,
            "family_collapse_suspected": family_collapse_suspected,
            "family_diversity_notes": family_diversity_notes,
            "accepted_seed_coverage": accepted_seed_coverage,
            "strict_seed_coverage": strict_seed_coverage,
            "near_strict_seed_coverage": near_strict_seed_coverage,
            "learning_support_seed_coverage": learning_support_seed_coverage,
            "orientation_support_seed_coverage": orientation_support_seed_coverage,
            "attach_eligible_seed_coverage": attach_eligible_seed_coverage,
            "accepted_records_missing_fingerprint": accepted_records_missing_fingerprint,
            "learning_support_prebridge_frame_p50": int(
                round(
                    float(np.percentile(learning_support_prebridge_counts, 50))
                    if learning_support_prebridge_counts
                    else 0.0
                )
            ),
            "learning_support_prebridge_frame_p90": int(
                round(
                    float(np.percentile(learning_support_prebridge_counts, 90))
                    if learning_support_prebridge_counts
                    else 0.0
                )
            ),
            "learning_support_window_len_p50": int(
                round(
                    float(np.percentile(learning_support_window_lengths, 50))
                    if learning_support_window_lengths
                    else 0.0
                )
            ),
            "learning_support_window_len_p90": int(
                round(
                    float(np.percentile(learning_support_window_lengths, 90))
                    if learning_support_window_lengths
                    else 0.0
                )
            ),
            "orientation_support_truthful_ratio_p50": float(
                np.percentile(np.asarray(orientation_support_truthful_ratios, dtype=np.float32), 50)
            )
            if orientation_support_truthful_ratios
            else 0.0,
            "orientation_support_truthful_ratio_p90": float(
                np.percentile(np.asarray(orientation_support_truthful_ratios, dtype=np.float32), 90)
            )
            if orientation_support_truthful_ratios
            else 0.0,
            "orientation_context_frame_p50": int(
                round(
                    float(np.percentile(orientation_context_frame_counts, 50))
                    if orientation_context_frame_counts
                    else 0.0
                )
            ),
            "orientation_context_frame_p90": int(
                round(
                    float(np.percentile(orientation_context_frame_counts, 90))
                    if orientation_context_frame_counts
                    else 0.0
                )
            ),
            "truthful_window_ratio_min": min(truth_window_ratios)
            if truth_window_ratios
            else 0.0,
            "measurement_backends": sorted(
                x for x in measurement_backends if x is not None
            ),
            "measurement_verifiers": sorted(
                x for x in measurement_verifiers if x is not None
            ),
            "runtime_handle_anchor_valid_rate": runtime_anchor_valid_rate,
        }
    )
    checks = [
        (missing_run_instance_count == 0, "run_instance_id_missing"),
        (missing_plan_version_count == 0, "plan_version_missing"),
        (missing_source_base_commit_count == 0, "source_base_commit_missing"),
        (
            run_instance_values == {plan.get("run_instance_id")},
            "run_instance_id_not_unique",
        ),
        (plan_version_values == {plan.get("plan_version")}, "plan_version_not_unique"),
        (
            source_base_commit_values == {plan.get("source_base_commit")},
            "source_base_commit_not_unique",
        ),
        (
            source_cells == {_plan_source_canonical_train_cell(plan)},
            "source_canonical_train_cell_not_unique",
        ),
        (
            transition_values == {plan.get("best_transition_cell")},
            "best_transition_cell_not_unique",
        ),
        (
            source_state_values == {_plan_source_best_train_state_mode(plan)},
            "source_best_train_state_mode_not_unique",
        ),
        (
            active_state_values == {_plan_active_train_state_mode(plan)},
            "active_train_state_mode_not_unique",
        ),
        (
            state_mode_values == {_plan_active_state_mode_name(plan)},
            "state_mode_not_unique",
        ),
        (
            set(provenance.get("state_modes", []))
            == {_plan_active_state_mode_name(plan)},
            "state_modes_not_unique",
        ),
        (
            set(provenance.get("claim_policies", [])) == {"canonical"},
            "claim_policy_not_canonical",
        ),
        (len(state_dim_signatures) == 1, "state_dim_signature_not_unique"),
        (bool(provenance.get("state_dim_names")), "state_dim_names_missing"),
        (bool(unique_successful_seeds), "no_successful_seeds"),
        (
            unique_successful_seeds.issubset(train_seeds),
            "successful_seeds_not_subset_of_train",
        ),
        (
            unique_successful_seeds.isdisjoint(heldout_seeds),
            "successful_seeds_overlap_heldout",
        ),
        (report["effective_frame_count"] > 0, "effective_frame_count_nonpositive"),
        (
            measurement_backends == {plan.get("measurement_backend")},
            "measurement_backend_inconsistent",
        ),
        (
            measurement_verifiers == {plan.get("measurement_verifier")},
            "measurement_verifier_inconsistent",
        ),
        (
            runtime_anchor_valid_rate == 1.0 or all(runtime_anchor_values),
            "runtime_handle_anchor_not_fully_valid",
        ),
    ]
    if dataset_selection_mode == "claim_canonical":
        checks.extend(
            [
                (
                    all(training_truth_flags),
                    "measurement_truthful_for_training_inconsistent",
                ),
                (
                    teacher_truth_adjudications == {"teacher_window_truth_v84"},
                    "teacher_truth_adjudication_inconsistent",
                ),
                (
                    all(bridge_truth_flags),
                    "bridge_not_in_truthful_window_inconsistent",
                ),
                (
                    all(
                        cls in {"strict_teacher", "near_strict_teacher"}
                        for cls in teacher_episode_classes
                    ),
                    "teacher_episode_class_inconsistent",
                ),
                (
                    accepted_records_missing_fingerprint == 0,
                    "teacher_fingerprint_missing_for_accepted_records",
                ),
            ]
        )
    elif dataset_selection_mode == "diagnostic_orientation_support":
        orientation_truth_flags = [
            bool(rec.get("measurement_truthful_for_orientation_support", False))
            for rec in records
        ]
        orientation_anchor_flags = [
            bool(rec.get("orientation_support_anchor_valid", False)) for rec in records
        ]
        checks.extend(
            [
                (
                    all(orientation_truth_flags),
                    "measurement_truthful_for_orientation_support_inconsistent",
                ),
                (
                    all(orientation_anchor_flags),
                    "orientation_support_anchor_invalid_inconsistent",
                ),
                (
                    all(
                        cls
                        in {
                            "orientation_transition_teacher",
                            "attach_eligible_transition_teacher",
                            "orientation_context_teacher",
                        }
                        for cls in orientation_support_teacher_classes
                    ),
                    "orientation_support_teacher_class_inconsistent",
                ),
                (
                    len(unique_orientation_support_families)
                    == int(report.get("orientation_support_unique_teacher_family_count", 0) or 0),
                    "orientation_support_fingerprint_count_inconsistent",
                ),
            ]
        )
    else:
        learning_support_truth_flags = [
            bool(rec.get("measurement_truthful_for_learning_support", False))
            for rec in records
        ]
        learning_support_prebridge_flags = [
            bool(rec.get("learning_support_contains_prebridge", False))
            for rec in records
        ]
        learning_support_bridge_flags = [
            bool(rec.get("learning_support_contains_bridge", False))
            for rec in records
        ]
        learning_support_anchor_invalid_flags = [
            bool(rec.get("learning_support_interval_anchor_invalid", False))
            for rec in records
        ]
        learning_support_mask_empty_flags = [
            bool(rec.get("learning_support_interval_mask_empty", False))
            for rec in records
        ]
        checks.extend(
            [
                (
                    all(learning_support_truth_flags),
                    "measurement_truthful_for_learning_support_inconsistent",
                ),
                (
                    all(learning_support_prebridge_flags),
                    "learning_support_missing_prebridge_inconsistent",
                ),
                (
                    all(learning_support_bridge_flags),
                    "learning_support_missing_bridge_inconsistent",
                ),
                (
                    not any(learning_support_anchor_invalid_flags),
                    "learning_support_interval_anchor_invalid_inconsistent",
                ),
                (
                    not any(learning_support_mask_empty_flags),
                    "learning_support_interval_mask_empty_inconsistent",
                ),
                (
                    all(
                        cls
                        in {
                            "strict_teacher",
                            "near_strict_teacher",
                            "learning_support_teacher",
                        }
                        for cls in learning_support_teacher_classes
                    ),
                    "learning_support_teacher_class_inconsistent",
                ),
                (
                    len(unique_learning_support_families)
                    == learning_support_unique_teacher_family_count,
                    "learning_support_fingerprint_count_inconsistent",
                ),
            ]
        )
    for passed, reason in checks:
        if not passed:
            report["errors"].append(reason)
    report["dataset_valid"] = not report["errors"]
    return report


def dataset_integrity(root: Path, repo_id: str) -> dict[str, Any]:
    data_files = sorted(root.glob("data/chunk-*/*.parquet"))
    payload = {
        "dataset_root": str(root),
        "repo_id": repo_id,
        "data_files": [str(p) for p in data_files[:10]],
        "total_chunks": len(list(root.glob("data/chunk-*"))),
        "meta_info_exists": (root / "meta" / "info.json").exists(),
        "meta_stats_exists": (root / "meta" / "stats.json").exists(),
        "meta_tasks_exists": (root / "meta" / "tasks.parquet").exists(),
        "meta_provenance_exists": (root / "meta" / "provenance.json").exists(),
    }
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id=repo_id, root=str(root))
        payload["dataset_loads"] = True
        payload["dataset_length"] = len(dataset)
        payload["feature_keys"] = sorted(dataset.features.keys())
        payload["image_shape"] = str(
            dataset.features.get("observation.images.image", {}).get("shape", "N/A")
        )
        payload["state_shape"] = str(
            dataset.features.get("observation.state", {}).get("shape", "N/A")
        )
        payload["action_shape"] = str(
            dataset.features.get("action", {}).get("shape", "N/A")
        )
    except Exception as exc:  # noqa: BLE001
        payload["dataset_loads"] = False
        payload["dataset_error"] = str(exc)
        payload["dataset_length"] = 0
    payload["passed"] = bool(
        data_files
        and payload["meta_info_exists"]
        and payload["meta_stats_exists"]
        and payload["meta_tasks_exists"]
        and payload["dataset_loads"]
        and payload["dataset_length"] > 0
    )
    return payload


def _infinigen_to_libero_action(
    action: np.ndarray, gripper_binarize: bool
) -> np.ndarray:
    return action.astype(np.float32)


def _clip_gripper_to_libero_range(gripper_joint: float) -> float:
    return float(np.clip(gripper_joint, -0.042, +0.001))


def _build_libero_state(
    state_npz: np.ndarray, gripper_binarize: bool, state_names: list[str]
) -> np.ndarray:
    state = state_npz.astype(np.float32).copy()
    for idx, name in enumerate(state_names):
        if name == "gripper_joint":
            state[idx] = _clip_gripper_to_libero_range(float(state[idx]))
    return state


def _rollout_provenance(meta: dict[str, Any], n_frames: int) -> dict[str, Any]:
    contract_config = meta.get("contract_config") or {}
    state_spec = meta.get("state_spec") or {}
    visual_mode_report = meta.get("visual_mode_report") or {}
    handle_probe = meta.get("handle_probe_metadata") or {}
    canonical_training_truth = meta.get("canonical_training_truth") or {}
    learning_support_truth = meta.get("learning_support_truth") or {}
    state_dim_names = _state_dim_names(meta)
    return {
        "seed": meta.get("seed"),
        "claim_policy": meta.get("claim_policy", "unspecified"),
        "raw_unique_frames": int(meta.get("raw_unique_frames", n_frames)),
        "effective_training_frames": int(
            meta.get("effective_training_frames", n_frames)
        ),
        "success_repeat": int(meta.get("success_repeat", 1)),
        "contract_config": contract_config,
        "state_mode": state_spec.get(
            "state_mode", contract_config.get("state_mode", "unknown")
        ),
        "state_spec": state_spec,
        "state_dim_names": state_dim_names,
        "diagnostic_only": bool(visual_mode_report.get("diagnostic_only", False)),
        "calibration_mode": visual_mode_report.get(
            "calibration_mode", contract_config.get("calibration_mode", "unknown")
        ),
        "secondary_camera_mode": visual_mode_report.get(
            "secondary_camera_mode",
            contract_config.get("secondary_camera_mode", "unknown"),
        ),
        "render_profile": visual_mode_report.get(
            "render_profile", contract_config.get("render_profile", "legacy_surface")
        ),
        "background_mode": visual_mode_report.get(
            "background_mode", contract_config.get("background_mode", "unknown")
        ),
        "lighting_profile": visual_mode_report.get(
            "lighting_profile", contract_config.get("lighting_profile", "unknown")
        ),
        "material_policy": visual_mode_report.get(
            "material_policy", contract_config.get("material_policy", "unknown")
        ),
        "camera_framing_profile": visual_mode_report.get(
            "camera_framing_profile",
            contract_config.get("camera_framing_profile", "unknown"),
        ),
        "measurement_mode": handle_probe.get(
            "probe_measurement_mode", contract_config.get("measurement_mode", "unknown")
        ),
        "truth_root_object": handle_probe.get("truth_root_object", "unknown"),
        "identity_resolution_tier": handle_probe.get(
            "identity_resolution_tier", "unknown"
        ),
        "measurement_backend": handle_probe.get("measurement_backend", "unknown"),
        "measurement_verifier": handle_probe.get("measurement_verifier", "unknown"),
        "measurement_truth_tier": handle_probe.get("measurement_truth_tier", "unknown"),
        "measurement_truthful": bool(handle_probe.get("measurement_truthful", False)),
        "measurement_truthful_for_training": bool(
            meta.get(
                "measurement_truthful_for_training",
                canonical_training_truth.get(
                    "measurement_truthful_for_training", False
                ),
            )
        ),
        "measurement_truthful_for_learning_support": bool(
            meta.get(
                "measurement_truthful_for_learning_support",
                learning_support_truth.get(
                    "measurement_truthful_for_learning_support", False
                ),
            )
        ),
        "teacher_truth_adjudication": meta.get(
            "teacher_truth_adjudication",
            canonical_training_truth.get("teacher_truth_adjudication"),
        ),
        "learning_support_truth_adjudication": meta.get(
            "learning_support_truth_adjudication",
            learning_support_truth.get("learning_support_truth_adjudication"),
        ),
        "teacher_truthful_window_frame_count": int(
            meta.get(
                "teacher_truthful_window_frame_count",
                canonical_training_truth.get("truthful_window_frame_count", 0),
            )
            or 0
        ),
        "truthful_window_ratio": float(
            meta.get(
                "truthful_window_ratio",
                canonical_training_truth.get("truthful_window_ratio", 0.0),
            )
            or 0.0
        ),
        "truthful_window_longest_interior_gap": int(
            meta.get(
                "truthful_window_longest_interior_gap",
                canonical_training_truth.get("truthful_window_longest_interior_gap", 0),
            )
            or 0
        ),
        "truthful_window_tail_truthful_count_last6": int(
            meta.get(
                "truthful_window_tail_truthful_count_last6",
                canonical_training_truth.get(
                    "truthful_window_tail_truthful_count_last6", 0
                ),
            )
            or 0
        ),
        "bridge_in_truthful_window": bool(
            meta.get(
                "bridge_in_truthful_window",
                canonical_training_truth.get("bridge_in_truthful_window", False),
            )
        ),
        "learning_support_window_start": meta.get(
            "learning_support_window_start",
            learning_support_truth.get("learning_support_window_start"),
        ),
        "learning_support_window_end": meta.get(
            "learning_support_window_end",
            learning_support_truth.get("learning_support_window_end"),
        ),
        "learning_support_window_len": int(
            meta.get(
                "learning_support_window_len",
                learning_support_truth.get("learning_support_window_len", 0),
            )
            or 0
        ),
        "learning_support_prebridge_frame_count": int(
            meta.get(
                "learning_support_prebridge_frame_count",
                learning_support_truth.get("learning_support_prebridge_frame_count", 0),
            )
            or 0
        ),
        "learning_support_prebridge_truthful_ratio": float(
            meta.get(
                "learning_support_prebridge_truthful_ratio",
                learning_support_truth.get(
                    "learning_support_prebridge_truthful_ratio", 0.0
                ),
            )
            or 0.0
        ),
        "learning_support_whole_window_truthful_ratio": float(
            meta.get(
                "learning_support_whole_window_truthful_ratio",
                learning_support_truth.get(
                    "learning_support_whole_window_truthful_ratio", 0.0
                ),
            )
            or 0.0
        ),
        "learning_support_bridge_truthful_ratio": float(
            meta.get(
                "learning_support_bridge_truthful_ratio",
                learning_support_truth.get(
                    "learning_support_bridge_truthful_ratio", 0.0
                ),
            )
            or 0.0
        ),
        "learning_support_bridge_longest_interior_gap": int(
            meta.get(
                "learning_support_bridge_longest_interior_gap",
                learning_support_truth.get(
                    "learning_support_bridge_longest_interior_gap", 0
                ),
            )
            or 0
        ),
        "learning_support_bridge_tail_truthful_count_last6": int(
            meta.get(
                "learning_support_bridge_tail_truthful_count_last6",
                learning_support_truth.get(
                    "learning_support_bridge_tail_truthful_count_last6", 0
                ),
            )
            or 0
        ),
        "learning_support_contains_prebridge": bool(
            meta.get(
                "learning_support_contains_prebridge",
                learning_support_truth.get("learning_support_contains_prebridge", False),
            )
        ),
        "learning_support_contains_bridge": bool(
            meta.get(
                "learning_support_contains_bridge",
                learning_support_truth.get("learning_support_contains_bridge", False),
            )
        ),
        "learning_support_interval_mask_empty": bool(
            meta.get(
                "learning_support_interval_mask_empty",
                learning_support_truth.get("learning_support_interval_mask_empty", False),
            )
        ),
        "learning_support_interval_anchor_invalid": bool(
            meta.get(
                "learning_support_interval_anchor_invalid",
                learning_support_truth.get(
                    "learning_support_interval_anchor_invalid", False
                ),
            )
        ),
        "learning_support_anchor_step": meta.get(
            "learning_support_anchor_step",
            learning_support_truth.get("learning_support_anchor_step"),
        ),
        "first_attach_eligible_step": meta.get(
            "first_attach_eligible_step",
            learning_support_truth.get("first_attach_eligible_step"),
        ),
        "orientation_support_window_start": meta.get(
            "orientation_support_window_start",
            (meta.get("orientation_support_truth") or {}).get("orientation_support_window_start"),
        ),
        "orientation_support_window_end": meta.get(
            "orientation_support_window_end",
            (meta.get("orientation_support_truth") or {}).get("orientation_support_window_end"),
        ),
        "orientation_support_window_len": int(
            meta.get(
                "orientation_support_window_len",
                (meta.get("orientation_support_truth") or {}).get("orientation_support_window_len", 0),
            )
            or 0
        ),
        "orientation_context_frame_count": int(
            meta.get("orientation_context_frame_count", 0) or 0
        ),
        "orientation_support_contains_distance_pass": bool(
            meta.get("orientation_support_contains_distance_pass", False)
        ),
        "orientation_support_contains_approach_pass": bool(
            meta.get("orientation_support_contains_approach_pass", False)
        ),
        "orientation_support_contains_orientation_correction": bool(
            meta.get("orientation_support_contains_orientation_correction", False)
        ),
        "orientation_support_contains_attach_eligible": bool(
            meta.get("orientation_support_contains_attach_eligible", False)
        ),
        "orientation_support_truthful_ratio": float(
            meta.get("orientation_support_truthful_ratio", 0.0) or 0.0
        ),
        "orientation_support_anchor_valid": bool(
            meta.get("orientation_support_anchor_valid", False)
        ),
        "measurement_truthful_for_orientation_support": bool(
            meta.get("measurement_truthful_for_orientation_support", False)
        ),
        "orientation_support_truth_adjudication": meta.get(
            "orientation_support_truth_adjudication"
        ),
        "orientation_support_teacher_class": meta.get(
            "orientation_support_teacher_class", "rejected_orientation_teacher"
        ),
        "orientation_support_fingerprint": meta.get("orientation_support_fingerprint", ""),
        "final_snapshot_measurement_truthful": bool(
            meta.get(
                "final_snapshot_measurement_truthful",
                canonical_training_truth.get(
                    "final_snapshot_measurement_truthful", False
                ),
            )
        ),
        "final_snapshot_measurement_truth_tier": meta.get(
            "final_snapshot_measurement_truth_tier",
            canonical_training_truth.get("final_snapshot_measurement_truth_tier"),
        ),
        "measurement_warning_flags": handle_probe.get("measurement_warning_flags", []),
        "manifest_handle_entity_unique": bool(
            handle_probe.get("manifest_handle_entity_unique", False)
        ),
        "runtime_handle_visual_geom_mapping_unique": bool(
            handle_probe.get("runtime_handle_visual_geom_mapping_unique", False)
        ),
        "runtime_handle_collision_geom_mapping_unique": bool(
            handle_probe.get("runtime_handle_collision_geom_mapping_unique", False)
        ),
        "duplicate_runtime_geom_name_flag": bool(
            handle_probe.get("duplicate_runtime_geom_name_flag", False)
        ),
        "unnamed_runtime_geom_flag": bool(
            handle_probe.get("unnamed_runtime_geom_flag", False)
        ),
        "resolved_handle_visual_geom_ids": handle_probe.get(
            "resolved_handle_visual_geom_ids", []
        ),
        "resolved_handle_visual_geom_names": handle_probe.get(
            "resolved_handle_visual_geom_names", []
        ),
        "resolved_handle_collision_geom_ids": handle_probe.get(
            "resolved_handle_collision_geom_ids", []
        ),
        "resolved_handle_collision_geom_names": handle_probe.get(
            "resolved_handle_collision_geom_names", []
        ),
        "handle_geom_ids": handle_probe.get("handle_geom_ids", []),
        "handle_geom_names": handle_probe.get("handle_geom_names", []),
        "semantic_mapping_hash": handle_probe.get("semantic_mapping_hash", ""),
        "segmentation_mask_support_rate_secondary": handle_probe.get(
            "segmentation_mask_support_rate_secondary", 0.0
        ),
        "isolated_mask_support_rate_secondary": handle_probe.get(
            "isolated_mask_support_rate_secondary", 0.0
        ),
        "segmentation_isolated_iou_secondary": handle_probe.get(
            "segmentation_isolated_iou_secondary", 0.0
        ),
        "segmentation_isolated_centroid_delta_px_secondary": handle_probe.get(
            "segmentation_isolated_centroid_delta_px_secondary", 0.0
        ),
        "handle_mask_area_ratio_primary": handle_probe.get(
            "handle_mask_area_ratio_primary", 0.0
        ),
        "handle_mask_area_ratio_secondary": handle_probe.get(
            "handle_mask_area_ratio_secondary", 0.0
        ),
        "bbox_over_mask_ratio_primary": handle_probe.get(
            "bbox_over_mask_ratio_primary", 0.0
        ),
        "bbox_over_mask_ratio_secondary": handle_probe.get(
            "bbox_over_mask_ratio_secondary", 0.0
        ),
        "selector_mode": meta.get("selector_mode", "adaptive"),
        "baseline_cell_id": meta.get("baseline_cell_id"),
        "ts_baseline_cell_id": meta.get("ts_baseline_cell_id"),
        "best_transition_cell": meta.get("best_transition_cell"),
        "canonical_train_cell": meta.get("canonical_train_cell"),
        "best_train_state_mode": meta.get("best_train_state_mode"),
        "source_canonical_train_cell": meta.get(
            "source_canonical_train_cell", meta.get("canonical_train_cell")
        ),
        "source_best_train_state_mode": meta.get(
            "source_best_train_state_mode", meta.get("best_train_state_mode")
        ),
        "active_train_state_mode": meta.get(
            "active_train_state_mode", meta.get("best_train_state_mode")
        ),
        "active_state_mode_name": meta.get(
            "active_state_mode_name",
            state_spec.get("state_mode", contract_config.get("state_mode", "unknown")),
        ),
        "run_instance_id": meta.get("run_instance_id"),
        "plan_version": meta.get("plan_version"),
        "source_base_commit": meta.get("source_base_commit"),
        "working_head_commit": meta.get("working_head_commit"),
        "strict_utility_version": meta.get("strict_utility_version"),
        "truth_utility_version": meta.get("truth_utility_version"),
        "teacher_fingerprint_version": meta.get("teacher_fingerprint_version"),
        "teacher_truth_gate": meta.get("teacher_truth_gate"),
        "truth_contract_hash": meta.get("truth_contract_hash"),
        "truth_contract_path": meta.get("truth_contract_path"),
        "truth_contract_root_object": meta.get("truth_contract_root_object"),
        "bridge_stage": meta.get("bridge_stage"),
        "bridge_attempt": meta.get("bridge_attempt"),
        "runtime_handle_anchor_valid": bool(
            meta.get("runtime_handle_anchor_valid", False)
        ),
        "phase_locked_rate": float(meta.get("phase_locked_rate", 0.0) or 0.0),
        "mean_effective_pull_progress": float(
            meta.get("mean_effective_pull_progress", 0.0) or 0.0
        ),
        "teacher_episode_class": str(meta.get("teacher_episode_class") or ""),
        "teacher_fingerprint": str(meta.get("teacher_fingerprint", "")),
        "learning_support_teacher_class": str(
            meta.get("learning_support_teacher_class") or ""
        ),
        "learning_support_fingerprint": str(
            meta.get("learning_support_fingerprint", "")
        ),
        "learning_support_fingerprint_version": meta.get(
            "learning_support_fingerprint_version"
        ),
        "effective_support_signature_v1": str(
            meta.get("effective_support_signature_v1", "")
        ),
        "effective_support_signature_version": meta.get(
            "effective_support_signature_version"
        ),
        "first_attach_step": strict_metrics.get("first_attach_step")
        if (strict_metrics := meta.get("strict_metrics") or {})
        else None,
        "attach_persistence": int(strict_metrics.get("attach_persistence", 0) or 0),
        "post_attach_drawer_delta": float(
            strict_metrics.get("post_attach_drawer_delta", 0.0) or 0.0
        ),
        "max_drawer_fraction": float(
            strict_metrics.get(
                "max_drawer_fraction", meta.get("max_drawer_fraction", 0.0)
            )
            or 0.0
        ),
        "frozen_matrix_hash": meta.get("frozen_matrix_hash"),
        "resource_budget_snapshot": meta.get("resource_budget_snapshot", {}),
        "visual_mode_report": visual_mode_report,
    }


def build_dataset_from_rollouts(
    rollout_paths: list[Path],
    dataset_root: Path,
    repo_id: str,
    *,
    robot_type: str = "infinigen_drawer_anygrasp_robot",
    remove_existing: bool = True,
    gripper_binarize: bool = False,
    image_size: int = 256,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    if remove_existing and dataset_root.exists():
        shutil.rmtree(dataset_root)

    metas: list[dict[str, Any]] = []
    state_signatures: set[tuple[str, ...]] = set()
    for npz_path in rollout_paths:
        meta_path = npz_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        metas.append(meta)
        state_signatures.add(tuple(_state_dim_names(meta)))
    if not metas:
        raise ValueError("No rollout metadata available for dataset build")
    if len(state_signatures) != 1:
        raise ValueError(
            f"Mixed state_spec.dim_names detected in one dataset build: {sorted(state_signatures)}"
        )
    state_names = list(next(iter(state_signatures)))

    img_shape = (image_size, image_size, 3)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        root=dataset_root,
        robot_type=robot_type,
        features={
            "observation.images.image": {
                "dtype": "image",
                "shape": img_shape,
                "names": ["height", "width", "channels"],
            },
            "observation.images.image2": {
                "dtype": "image",
                "shape": img_shape,
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ACTION_NAMES,
            },
        },
        use_videos=False,
    )

    episode_count = 0
    frame_count = 0
    seeds: list[int] = []
    tasks: list[str] = []
    source_files: list[str] = []
    provenance_records: list[dict[str, Any]] = []
    raw_unique_frame_total = 0
    effective_training_frame_total = 0
    claim_policies: set[str] = set()
    state_modes: set[str] = set()
    render_profiles: set[str] = set()
    background_modes: set[str] = set()
    lighting_profiles: set[str] = set()
    material_policies: set[str] = set()
    camera_framing_profiles: set[str] = set()
    success_repeats: set[int] = set()
    source_canonical_cells: set[str] = set()
    source_best_states: set[str] = set()
    active_train_states: set[str] = set()
    teacher_truth_adjudications: set[str] = set()
    truth_contract_hashes: set[str] = set()
    truth_contract_roots: set[str] = set()
    run_instance_ids: set[str] = set()
    plan_versions: set[str] = set()
    source_base_commits: set[str] = set()

    for npz_path, meta in zip(rollout_paths, metas, strict=False):
        data = np.load(npz_path, allow_pickle=True)
        task = str(meta.get("task", "open the middle drawer of the cabinet"))
        source_files.append(str(npz_path))
        tasks.append(task)
        n_frames = len(data["actions"])
        provenance = _rollout_provenance(meta, n_frames)
        provenance_records.append(provenance)
        raw_unique_frame_total += int(provenance["raw_unique_frames"])
        effective_training_frame_total += int(provenance["effective_training_frames"])
        claim_policies.add(str(provenance["claim_policy"]))
        state_modes.add(str(provenance["state_mode"]))
        render_profiles.add(str(provenance.get("render_profile", "legacy_surface")))
        background_modes.add(str(provenance.get("background_mode", "unknown")))
        lighting_profiles.add(str(provenance.get("lighting_profile", "unknown")))
        material_policies.add(str(provenance.get("material_policy", "unknown")))
        camera_framing_profiles.add(
            str(provenance.get("camera_framing_profile", "unknown"))
        )
        success_repeats.add(int(provenance["success_repeat"]))
        source_canonical_cells.add(
            str(provenance.get("source_canonical_train_cell") or "")
        )
        source_best_states.add(
            str(provenance.get("source_best_train_state_mode") or "")
        )
        active_train_states.add(str(provenance.get("active_train_state_mode") or ""))
        teacher_truth_adjudications.add(
            str(provenance.get("teacher_truth_adjudication") or "")
        )
        run_instance_ids.add(str(provenance.get("run_instance_id") or ""))
        plan_versions.add(str(provenance.get("plan_version") or ""))
        source_base_commits.add(str(provenance.get("source_base_commit") or ""))

        for idx in range(n_frames):
            raw_action = data["actions"][idx].astype(np.float32)
            action = _infinigen_to_libero_action(raw_action, gripper_binarize)
            raw_state = data["states"][idx].astype(np.float32)
            state = _build_libero_state(raw_state, gripper_binarize, state_names)

            img = Image.fromarray(data["images"][idx].astype(np.uint8))
            img = img.resize((image_size, image_size), Image.BILINEAR)
            img_np = np.array(img, dtype=np.uint8)

            img2 = Image.fromarray(data["images2"][idx].astype(np.uint8))
            img2 = img2.resize((image_size, image_size), Image.BILINEAR)
            img2_np = np.array(img2, dtype=np.uint8)

            dataset.add_frame(
                {
                    "task": task,
                    "observation.images.image": img_np,
                    "observation.images.image2": img2_np,
                    "observation.state": state,
                    "action": action,
                }
            )
            frame_count += 1
        dataset.save_episode()
        episode_count += 1
        if "seed" in meta:
            seeds.append(int(meta["seed"]))

    dataset.finalize()

    accepted_fingerprints = {
        rec.get("teacher_fingerprint")
        for rec in provenance_records
        if rec.get("teacher_episode_class") in {"strict_teacher", "near_strict_teacher"}
        and rec.get("teacher_fingerprint")
    }
    strict_fingerprints = {
        rec.get("teacher_fingerprint")
        for rec in provenance_records
        if rec.get("teacher_episode_class") == "strict_teacher"
        and rec.get("teacher_fingerprint")
    }
    near_strict_fingerprints = {
        rec.get("teacher_fingerprint")
        for rec in provenance_records
        if rec.get("teacher_episode_class") == "near_strict_teacher"
        and rec.get("teacher_fingerprint")
    }
    learning_support_fingerprints = {
        _learning_support_family_key(rec)
        for rec in provenance_records
        if rec.get("learning_support_teacher_class")
        in {"strict_teacher", "near_strict_teacher", "learning_support_teacher"}
        and _learning_support_family_key(rec)
    }
    orientation_support_fingerprints = {
        _orientation_support_family_key(rec)
        for rec in provenance_records
        if rec.get("orientation_support_teacher_class")
        in {
            "orientation_transition_teacher",
            "attach_eligible_transition_teacher",
            "orientation_context_teacher",
        }
        and _orientation_support_family_key(rec)
    }
    seeds_with_records = sorted(
        {int(rec.get("seed")) for rec in provenance_records if rec.get("seed") is not None}
    )
    accepted_family_count_by_seed: dict[str, int] = {}
    strict_family_count_by_seed: dict[str, int] = {}
    near_strict_family_count_by_seed: dict[str, int] = {}
    learning_support_family_count_by_seed: dict[str, int] = {}
    orientation_support_family_count_by_seed: dict[str, int] = {}
    teacher_fingerprint_examples_by_seed: dict[str, list[str]] = {}
    learning_support_prebridge_counts: list[int] = []
    learning_support_window_lengths: list[int] = []
    attach_eligible_seed_coverage: set[int] = set()
    learning_support_seed_coverage: set[int] = set()
    orientation_support_seed_coverage: set[int] = set()
    orientation_support_truthful_ratios: list[float] = []
    orientation_context_frame_counts: list[int] = []
    for seed in seeds_with_records:
        seed_records = [rec for rec in provenance_records if rec.get("seed") is not None and int(rec.get("seed")) == seed]
        accepted_seed_fps = sorted({
            rec.get("teacher_fingerprint")
            for rec in seed_records
            if rec.get("teacher_episode_class") in {"strict_teacher", "near_strict_teacher"}
            and rec.get("teacher_fingerprint")
        })
        strict_seed_fps = sorted({
            rec.get("teacher_fingerprint")
            for rec in seed_records
            if rec.get("teacher_episode_class") == "strict_teacher"
            and rec.get("teacher_fingerprint")
        })
        near_seed_fps = sorted({
            rec.get("teacher_fingerprint")
            for rec in seed_records
            if rec.get("teacher_episode_class") == "near_strict_teacher"
            and rec.get("teacher_fingerprint")
        })
        learning_support_seed_fps = sorted({
            _learning_support_family_key(rec)
            for rec in seed_records
            if rec.get("learning_support_teacher_class") in {"strict_teacher", "near_strict_teacher", "learning_support_teacher"}
            and _learning_support_family_key(rec)
        })
        orientation_support_seed_fps = sorted({
            _orientation_support_family_key(rec)
            for rec in seed_records
            if rec.get("orientation_support_teacher_class") in {
                "orientation_transition_teacher",
                "attach_eligible_transition_teacher",
                "orientation_context_teacher",
            }
            and _orientation_support_family_key(rec)
        })
        accepted_family_count_by_seed[str(seed)] = len(accepted_seed_fps)
        strict_family_count_by_seed[str(seed)] = len(strict_seed_fps)
        near_strict_family_count_by_seed[str(seed)] = len(near_seed_fps)
        learning_support_family_count_by_seed[str(seed)] = len(learning_support_seed_fps)
        orientation_support_family_count_by_seed[str(seed)] = len(orientation_support_seed_fps)
        teacher_fingerprint_examples_by_seed[str(seed)] = accepted_seed_fps[:5]
        if learning_support_seed_fps:
            learning_support_seed_coverage.add(int(seed))
        if orientation_support_seed_fps:
            orientation_support_seed_coverage.add(int(seed))
        if any(rec.get("first_attach_eligible_step") is not None for rec in seed_records):
            attach_eligible_seed_coverage.add(int(seed))
        learning_support_prebridge_counts.extend(
            int(rec.get("learning_support_prebridge_frame_count", 0) or 0)
            for rec in seed_records
            if rec.get("learning_support_teacher_class")
            in {"strict_teacher", "near_strict_teacher", "learning_support_teacher"}
        )
        learning_support_window_lengths.extend(
            int(rec.get("learning_support_window_len", 0) or 0)
            for rec in seed_records
            if rec.get("learning_support_teacher_class")
            in {"strict_teacher", "near_strict_teacher", "learning_support_teacher"}
        )
        orientation_support_truthful_ratios.extend(
            float(rec.get("orientation_support_truthful_ratio", 0.0) or 0.0)
            for rec in seed_records
            if rec.get("orientation_support_teacher_class")
            in {
                "orientation_transition_teacher",
                "attach_eligible_transition_teacher",
                "orientation_context_teacher",
            }
        )
        orientation_context_frame_counts.extend(
            int(rec.get("orientation_context_frame_count", 0) or 0)
            for rec in seed_records
            if rec.get("orientation_support_teacher_class")
            in {
                "orientation_transition_teacher",
                "attach_eligible_transition_teacher",
                "orientation_context_teacher",
            }
        )

    family_diversity_notes: list[str] = []
    if len(accepted_fingerprints) <= max(len(seeds_with_records), 1):
        family_diversity_notes.append(
            "accepted_family_count_is_close_to_seed_count"
        )
    if len(near_strict_fingerprints) == 0:
        family_diversity_notes.append("near_strict_family_supply_absent")
    if any(count <= 1 for count in accepted_family_count_by_seed.values()):
        family_diversity_notes.append("at_least_one_seed_has_single_family_support")

    family_collapse_suspected = bool(family_diversity_notes)

    provenance_payload = {
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "unique_successful_seeds": sorted(set(seeds)),
        "raw_unique_frames": int(raw_unique_frame_total),
        "effective_training_frames": int(effective_training_frame_total),
        "effective_episode_count": int(episode_count),
        "effective_frame_count": int(frame_count),
        "claim_policies": sorted(claim_policies),
        "state_modes": sorted(state_modes),
        "state_dim_names": state_names,
        "render_profiles": sorted(render_profiles),
        "background_modes": sorted(background_modes),
        "lighting_profiles": sorted(lighting_profiles),
        "material_policies": sorted(material_policies),
        "camera_framing_profiles": sorted(camera_framing_profiles),
        "success_repeat_values": sorted(success_repeats),
        "source_canonical_train_cells": sorted(x for x in source_canonical_cells if x),
        "source_best_train_state_modes": sorted(x for x in source_best_states if x),
        "active_train_state_modes": sorted(x for x in active_train_states if x),
        "run_instance_ids": sorted(x for x in run_instance_ids if x),
        "plan_versions": sorted(x for x in plan_versions if x),
        "source_base_commits": sorted(x for x in source_base_commits if x),
        "teacher_truth_adjudications": sorted(
            x for x in teacher_truth_adjudications if x
        ),
        "truth_contract_hashes": sorted(x for x in truth_contract_hashes if x),
        "truth_contract_roots": sorted(x for x in truth_contract_roots if x),
        "accepted_unique_teacher_families": sorted(accepted_fingerprints),
        "accepted_unique_teacher_family_count": int(len(accepted_fingerprints)),
        "strict_unique_teacher_family_count": int(len(strict_fingerprints)),
        "near_strict_unique_teacher_family_count": int(len(near_strict_fingerprints)),
        "learning_support_unique_teacher_family_count": int(
            len(learning_support_fingerprints)
        ),
        "orientation_support_unique_teacher_family_count": int(
            len(orientation_support_fingerprints)
        ),
        "accepted_family_count_by_seed": accepted_family_count_by_seed,
        "strict_family_count_by_seed": strict_family_count_by_seed,
        "near_strict_family_count_by_seed": near_strict_family_count_by_seed,
        "learning_support_family_count_by_seed": learning_support_family_count_by_seed,
        "orientation_support_family_count_by_seed": orientation_support_family_count_by_seed,
        "teacher_fingerprint_examples_by_seed": teacher_fingerprint_examples_by_seed,
        "family_collapse_suspected": family_collapse_suspected,
        "family_diversity_notes": family_diversity_notes,
        "teacher_episode_class_counts": dict(
            Counter(
                str(rec.get("teacher_episode_class", "rejected_teacher"))
                for rec in provenance_records
            )
        ),
        "learning_support_teacher_class_counts": dict(
            Counter(
                str(rec.get("learning_support_teacher_class", "rejected_teacher"))
                for rec in provenance_records
            )
        ),
        "orientation_support_teacher_class_counts": dict(
            Counter(
                str(rec.get("orientation_support_teacher_class", "rejected_orientation_teacher"))
                for rec in provenance_records
            )
        ),
        "learning_support_seed_coverage": sorted(learning_support_seed_coverage),
        "orientation_support_seed_coverage": sorted(orientation_support_seed_coverage),
        "attach_eligible_seed_coverage": sorted(attach_eligible_seed_coverage),
        "learning_support_prebridge_frame_p50": int(
            round(
                float(np.percentile(learning_support_prebridge_counts, 50))
                if learning_support_prebridge_counts
                else 0.0
            )
        ),
        "learning_support_prebridge_frame_p90": int(
            round(
                float(np.percentile(learning_support_prebridge_counts, 90))
                if learning_support_prebridge_counts
                else 0.0
            )
        ),
        "learning_support_window_len_p50": int(
            round(
                float(np.percentile(learning_support_window_lengths, 50))
                if learning_support_window_lengths
                else 0.0
            )
        ),
        "learning_support_window_len_p90": int(
            round(
                float(np.percentile(learning_support_window_lengths, 90))
                if learning_support_window_lengths
                else 0.0
            )
        ),
        "orientation_support_truthful_ratio_p50": float(
            np.percentile(np.asarray(orientation_support_truthful_ratios, dtype=np.float32), 50)
        )
        if orientation_support_truthful_ratios
        else 0.0,
        "orientation_support_truthful_ratio_p90": float(
            np.percentile(np.asarray(orientation_support_truthful_ratios, dtype=np.float32), 90)
        )
        if orientation_support_truthful_ratios
        else 0.0,
        "orientation_context_frame_p50": int(
            round(
                float(np.percentile(orientation_context_frame_counts, 50))
                if orientation_context_frame_counts
                else 0.0
            )
        ),
        "orientation_context_frame_p90": int(
            round(
                float(np.percentile(orientation_context_frame_counts, 90))
                if orientation_context_frame_counts
                else 0.0
            )
        ),
        "records": provenance_records,
    }
    provenance_path = dataset_root / "meta" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance_payload, indent=2, ensure_ascii=False) + "\n"
    )

    integrity = dataset_integrity(dataset_root, repo_id)
    return {
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "episode_count": episode_count,
        "frame_count": frame_count,
        "seed_coverage": sorted(set(seeds)),
        "task_coverage": sorted(set(tasks)),
        "source_files": source_files,
        "integrity": integrity,
        "unique_successful_seeds": sorted(set(seeds)),
        "raw_unique_frames": int(raw_unique_frame_total),
        "effective_training_frames": int(effective_training_frame_total),
        "effective_episode_count": int(episode_count),
        "effective_frame_count": int(frame_count),
        "claim_policies": sorted(claim_policies),
        "state_modes": sorted(state_modes),
        "state_dim_names": state_names,
        "render_profiles": sorted(render_profiles),
        "background_modes": sorted(background_modes),
        "lighting_profiles": sorted(lighting_profiles),
        "material_policies": sorted(material_policies),
        "camera_framing_profiles": sorted(camera_framing_profiles),
        "success_repeat_values": sorted(success_repeats),
        "source_canonical_train_cells": sorted(x for x in source_canonical_cells if x),
        "source_best_train_state_modes": sorted(x for x in source_best_states if x),
        "active_train_state_modes": sorted(x for x in active_train_states if x),
        "run_instance_ids": sorted(x for x in run_instance_ids if x),
        "plan_versions": sorted(x for x in plan_versions if x),
        "source_base_commits": sorted(x for x in source_base_commits if x),
        "teacher_truth_adjudications": sorted(
            x for x in teacher_truth_adjudications if x
        ),
        "accepted_unique_teacher_family_count": provenance_payload[
            "accepted_unique_teacher_family_count"
        ],
        "strict_unique_teacher_family_count": provenance_payload[
            "strict_unique_teacher_family_count"
        ],
        "near_strict_unique_teacher_family_count": provenance_payload[
            "near_strict_unique_teacher_family_count"
        ],
        "learning_support_unique_teacher_family_count": provenance_payload[
            "learning_support_unique_teacher_family_count"
        ],
        "orientation_support_unique_teacher_family_count": provenance_payload[
            "orientation_support_unique_teacher_family_count"
        ],
        "teacher_episode_class_counts": provenance_payload[
            "teacher_episode_class_counts"
        ],
        "learning_support_teacher_class_counts": provenance_payload[
            "learning_support_teacher_class_counts"
        ],
        "orientation_support_teacher_class_counts": provenance_payload[
            "orientation_support_teacher_class_counts"
        ],
        "accepted_family_count_by_seed": provenance_payload[
            "accepted_family_count_by_seed"
        ],
        "strict_family_count_by_seed": provenance_payload[
            "strict_family_count_by_seed"
        ],
        "near_strict_family_count_by_seed": provenance_payload[
            "near_strict_family_count_by_seed"
        ],
        "learning_support_family_count_by_seed": provenance_payload[
            "learning_support_family_count_by_seed"
        ],
        "orientation_support_family_count_by_seed": provenance_payload[
            "orientation_support_family_count_by_seed"
        ],
        "teacher_fingerprint_examples_by_seed": provenance_payload[
            "teacher_fingerprint_examples_by_seed"
        ],
        "family_collapse_suspected": provenance_payload[
            "family_collapse_suspected"
        ],
        "family_diversity_notes": provenance_payload[
            "family_diversity_notes"
        ],
        "learning_support_seed_coverage": provenance_payload[
            "learning_support_seed_coverage"
        ],
        "orientation_support_seed_coverage": provenance_payload[
            "orientation_support_seed_coverage"
        ],
        "attach_eligible_seed_coverage": provenance_payload[
            "attach_eligible_seed_coverage"
        ],
        "learning_support_prebridge_frame_p50": provenance_payload[
            "learning_support_prebridge_frame_p50"
        ],
        "learning_support_prebridge_frame_p90": provenance_payload[
            "learning_support_prebridge_frame_p90"
        ],
        "orientation_support_truthful_ratio_p50": provenance_payload[
            "orientation_support_truthful_ratio_p50"
        ],
        "orientation_support_truthful_ratio_p90": provenance_payload[
            "orientation_support_truthful_ratio_p90"
        ],
        "orientation_context_frame_p50": provenance_payload[
            "orientation_context_frame_p50"
        ],
        "orientation_context_frame_p90": provenance_payload[
            "orientation_context_frame_p90"
        ],
        "learning_support_window_len_p50": provenance_payload[
            "learning_support_window_len_p50"
        ],
        "learning_support_window_len_p90": provenance_payload[
            "learning_support_window_len_p90"
        ],
        "provenance_path": str(provenance_path),
    }

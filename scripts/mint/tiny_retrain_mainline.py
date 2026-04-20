#!/usr/bin/env python3
"""Shared helpers for the V1cT2 tiny-retrain mainline."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from drawer_robot_env_mujoco import build_robot_rollout, save_robot_rollout
from mint_common import (
    ARTIFACT_DIR,
    DATASET_DIR,
    DATASET_REPO_ID,
    DEFAULT_HELD_OUT_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    PROJECT_ROOT,
    load_json,
)
from root_cause_controller import RootCauseController
from strict_success import STRICT_SUCCESS_VERSION, evaluate_strict_success

RCA5_ARTIFACT = ARTIFACT_DIR / "p2rca5_frozen_matrix_screen.json"
RCA7_ARTIFACT = ARTIFACT_DIR / "p2rca7_tiny_retrain_if_eligible.json"
DEFAULT_ROLLOUT_SOURCE_DIR = ARTIFACT_DIR / "g6_canonical_train_rollouts"
MATERIALIZATION_ARTIFACT = ARTIFACT_DIR / "g6_canonical_rollout_materialization.json"
LEARNING_SUPPORT_ROLLOUT_SOURCE_DIR = ARTIFACT_DIR / "g6_learning_support_train_rollouts"
LEARNING_SUPPORT_MATERIALIZATION_ARTIFACT = (
    ARTIFACT_DIR / "g6_learning_support_rollout_materialization.json"
)
ORIENTATION_SUPPORT_ROLLOUT_SOURCE_DIR = (
    ARTIFACT_DIR / "g6_orientation_support_train_rollouts"
)
ORIENTATION_SUPPORT_MATERIALIZATION_ARTIFACT = (
    ARTIFACT_DIR / "g6_orientation_support_rollout_materialization.json"
)
MIN_TRAIN_EPISODES = 48
DEFAULT_EPISODES_PER_SEED = 12
MIN_SUCCESSFUL_SEEDS = 6
TRUTHFUL_WINDOW_MIN_FRAMES = 16
LEARNING_SUPPORT_WINDOW_MIN_FRAMES = 24
LEARNING_SUPPORT_PREBRIDGE_FRAMES = 24
LEARNING_SUPPORT_MIN_PREBRIDGE_FRAMES = 8
ORIENTATION_SUPPORT_WINDOW_MIN_FRAMES = 16
ORIENTATION_CONTEXT_PRE_FRAMES = 9
ORIENTATION_CONTEXT_POST_FRAMES = 8
ORIENTATION_ATTACH_THRESHOLD_M = 0.06
ORIENTATION_NEAR_DISTANCE_THRESHOLD_M = ORIENTATION_ATTACH_THRESHOLD_M + 0.05
ORIENTATION_APPROACH_THRESHOLD = 0.25
ORIENTATION_NEAR_APPROACH_THRESHOLD = 0.15
ORIENTATION_THRESHOLD = 0.60
ORIENTATION_ALIGNMENT_MIN_DELTA = 0.05
ORIENTATION_TRANSITION_DELTA = 0.10
ORIENTATION_TRANSITION_CLASS_DELTA = 0.20
ORIENTATION_TRANSITION_CLASS_MIN_CONTEXT_FRAMES = 16
ORIENTATION_WINDOW_LOOKAHEAD = 12
TERMINAL_COMMIT = "5aaf117b66902219ac997082763fb4e2ea8891b3"
STRICT_UTILITY_VERSION = STRICT_SUCCESS_VERSION
TRUTH_UTILITY_VERSION = "trace_window_v2"
TEACHER_FINGERPRINT_VERSION = "v8_3_fingerprint_v1"
LEARNING_SUPPORT_TRUTH_VERSION = "learning_support_window_v1"
ORIENTATION_SUPPORT_TRUTH_VERSION = "orientation_support_window_v1"
LEARNING_SUPPORT_FINGERPRINT_VERSION = "v10_learning_support_fingerprint_v1"
ORIENTATION_SUPPORT_FINGERPRINT_VERSION = "v12_orientation_support_fingerprint_v1"
EFFECTIVE_SUPPORT_SIGNATURE_VERSION = "effective_support_signature_v1"
TEACHER_FAMILY_GRID_VERSION = "v11_support_family_grid_v1"
TEACHER_FAMILY_GRID_V11 = [
    {
        "teacher_family_variant": "base",
        "handle_tangent_offset_m": 0.0,
        "handle_vertical_offset_m": 0.0,
        "approach_speed_scale": 1.0,
        "close_distance_offset_m": 0.0,
        "pregrasp_hold_steps": 0,
    },
    {
        "teacher_family_variant": "early_close_slow",
        "handle_tangent_offset_m": 0.0,
        "handle_vertical_offset_m": 0.0,
        "approach_speed_scale": 0.75,
        "close_distance_offset_m": 0.015,
        "pregrasp_hold_steps": 2,
    },
    {
        "teacher_family_variant": "tangent_plus",
        "handle_tangent_offset_m": 0.012,
        "handle_vertical_offset_m": 0.0,
        "approach_speed_scale": 0.9,
        "close_distance_offset_m": 0.005,
        "pregrasp_hold_steps": 1,
    },
    {
        "teacher_family_variant": "vertical_plus",
        "handle_tangent_offset_m": 0.0,
        "handle_vertical_offset_m": 0.010,
        "approach_speed_scale": 0.9,
        "close_distance_offset_m": 0.005,
        "pregrasp_hold_steps": 1,
    },
]
TRUTH_CONTRACT_PATH = PROJECT_ROOT / "docs" / "contracts" / "truth_contract_v84.json"
STATE_MODE_MAP = {
    "S0": "m0_proxy",
    "S1": "telemetry_candidate_v3_transition",
    "S2": "telemetry_candidate_v4_task_identity",
}
TRUTH_CONTRACT_REQUIRED_KEYS = (
    "truth_root_object",
    "teacher_truth_predicate",
    "dataset_truth_predicate",
    "probe_truth_predicate",
    "claim_truth_predicate",
    "allowed_fallbacks",
    "forbidden_fallbacks",
)
TRUTH_CONTRACT_FORBIDDEN_KEYS = (
    "teacher_predicate",
    "dataset_predicate",
    "probe_predicate",
    "claim_predicate",
)


def _truth_contract_required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            f"Truth contract missing required string '{key}' in {TRUTH_CONTRACT_PATH}"
        )
    return value.strip()


def _truth_contract_payload() -> dict[str, Any]:
    payload = load_json(TRUTH_CONTRACT_PATH, {})
    if not payload:
        raise RuntimeError(f"Missing truth contract payload: {TRUTH_CONTRACT_PATH}")
    missing = [key for key in TRUTH_CONTRACT_REQUIRED_KEYS if key not in payload]
    if missing:
        raise RuntimeError(
            f"Truth contract missing required keys {missing}: {TRUTH_CONTRACT_PATH}"
        )
    present_forbidden = [key for key in TRUTH_CONTRACT_FORBIDDEN_KEYS if key in payload]
    if present_forbidden:
        raise RuntimeError(
            f"Truth contract contains forbidden legacy keys {present_forbidden}: {TRUTH_CONTRACT_PATH}"
        )
    forbidden_fallbacks = {
        str(item).strip()
        for item in (payload.get("forbidden_fallbacks") or [])
        if str(item).strip()
    }
    missing_forbidden = [
        key for key in TRUTH_CONTRACT_FORBIDDEN_KEYS if key not in forbidden_fallbacks
    ]
    if missing_forbidden:
        raise RuntimeError(
            f"Truth contract forbidden_fallbacks missing legacy keys {missing_forbidden}: {TRUTH_CONTRACT_PATH}"
        )
    return payload


def _truth_contract_hash(payload: dict[str, Any] | None = None) -> str:
    _ = payload  # compatibility with older call sites that passed a payload explicitly
    return hashlib.sha256(TRUTH_CONTRACT_PATH.read_bytes()).hexdigest()


def _seed_list(payload: dict[str, Any], primary: str, fallback: list[int]) -> list[int]:
    raw = payload.get(primary)
    if not raw and primary == "train_seeds":
        raw = payload.get("training_seeds")
    if not raw and primary == "heldout_seeds":
        raw = payload.get("held_out_seeds")
    if raw:
        return [int(seed) for seed in raw]
    return list(fallback)


def _plan_source_canonical_train_cell(expected: dict[str, Any]) -> str:
    return str(
        expected.get("source_canonical_train_cell")
        or expected.get("canonical_train_cell")
        or ""
    )


def _plan_source_best_train_state_mode(expected: dict[str, Any]) -> str:
    return str(
        expected.get("source_best_train_state_mode")
        or expected.get("best_train_state_mode")
        or ""
    )


def _plan_active_train_state_mode(expected: dict[str, Any]) -> str:
    active = expected.get("active_train_state_mode")
    return str(active or _plan_source_best_train_state_mode(expected))


def _plan_active_state_mode_name(expected: dict[str, Any]) -> str:
    active = _plan_active_train_state_mode(expected)
    return str(
        expected.get("active_state_mode_name") or STATE_MODE_MAP.get(active, active)
    )


def _plan_dataset_root(expected: dict[str, Any]) -> Path:
    raw = expected.get("dataset_root")
    if raw:
        path = Path(str(raw))
        return path if path.is_absolute() else path
    return DATASET_DIR


def _teacher_family_variant_payload(
    expected: dict[str, Any], episode_index: int
) -> dict[str, Any]:
    mode = str(expected.get("support_family_repair_mode") or "").strip().upper()
    if mode != "G2B":
        return dict(TEACHER_FAMILY_GRID_V11[0])
    return dict(TEACHER_FAMILY_GRID_V11[int(episode_index) % len(TEACHER_FAMILY_GRID_V11)])


def expected_training_targets() -> dict[str, Any]:
    rca5 = load_json(RCA5_ARTIFACT, {})
    rca7 = load_json(RCA7_ARTIFACT, {})
    split_a = _seed_list(rca5, "split_a_seeds", DEFAULT_TRAIN_SEEDS)
    split_b = _seed_list(rca5, "split_b_seeds", DEFAULT_HELD_OUT_SEEDS)
    seed_source = "rca5_split_a" if rca5.get("split_a_seeds") else "default_train_seeds"
    source_best_state = str(
        rca5.get("best_train_state_mode") or rca7.get("best_train_state_mode") or "S0"
    )
    source_cell = str(
        rca5.get("canonical_train_cell") or rca7.get("canonical_train_cell") or ""
    )
    return {
        "best_transition_cell": rca5.get("best_transition_cell") or source_cell,
        "canonical_train_cell": source_cell,
        "best_train_state_mode": source_best_state,
        "source_canonical_train_cell": source_cell,
        "source_best_train_state_mode": source_best_state,
        "active_train_state_mode": source_best_state,
        "active_state_mode_name": STATE_MODE_MAP.get(
            source_best_state, source_best_state
        ),
        "tiny_retrain_permitted": bool(rca7.get("tiny_retrain_permitted", False)),
        "training_seeds": split_a,
        "train_seeds": split_a,
        "heldout_seeds": split_b,
        "training_seed_source": seed_source,
        "preferred_canonical_train_cell": rca5.get("preferred_canonical_train_cell")
        or rca7.get("preferred_canonical_train_cell"),
        "terminal_commit": TERMINAL_COMMIT,
    }


def dataset_guard(expected: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    dataset_root = _plan_dataset_root(expected)
    provenance_path = dataset_root / "meta" / "provenance.json"
    report = {
        **expected,
        "dataset_root": str(dataset_root),
        "repo_id": expected.get("dataset_repo_id", DATASET_REPO_ID),
        "provenance_exists": provenance_path.exists(),
    }
    if not bool(expected.get("tiny_retrain_permitted", True)):
        report["error"] = "RCA7 did not permit tiny retrain"
        return False, report
    if not provenance_path.exists():
        report["error"] = "Dataset provenance.json is missing"
        return False, report
    provenance = json.loads(provenance_path.read_text())
    report["provenance_path"] = str(provenance_path)
    report["state_modes"] = provenance.get("state_modes", [])
    records = provenance.get("records", [])
    report["record_count"] = len(records)
    source_cell = _plan_source_canonical_train_cell(expected)
    source_state = _plan_source_best_train_state_mode(expected)
    active_state = _plan_active_train_state_mode(expected)
    active_state_name = _plan_active_state_mode_name(expected)
    source_cells = sorted(
        {
            rec.get("source_canonical_train_cell") or rec.get("canonical_train_cell")
            for rec in records
            if rec.get("source_canonical_train_cell") or rec.get("canonical_train_cell")
        }
    )
    transition_values = sorted(
        {
            rec.get("best_transition_cell")
            for rec in records
            if rec.get("best_transition_cell") is not None
        }
    )
    source_state_values = sorted(
        {
            rec.get("source_best_train_state_mode") or rec.get("best_train_state_mode")
            for rec in records
            if rec.get("source_best_train_state_mode")
            or rec.get("best_train_state_mode")
        }
    )
    active_state_values = sorted(
        {
            rec.get("active_train_state_mode")
            for rec in records
            if rec.get("active_train_state_mode") is not None
        }
    )
    report["record_source_canonical_train_cells"] = source_cells
    report["record_best_transition_cells"] = transition_values
    report["record_source_best_train_state_modes"] = source_state_values
    report["record_active_train_state_modes"] = active_state_values
    report["expected_active_state_mode_name"] = active_state_name
    if not source_cells or source_cell not in source_cells:
        report["error"] = (
            "Dataset provenance is not bound to the selected source_canonical_train_cell"
        )
        return False, report
    if (
        not transition_values
        or str(expected.get("best_transition_cell")) not in transition_values
    ):
        report["error"] = (
            "Dataset provenance is not bound to the selected best_transition_cell"
        )
        return False, report
    if not source_state_values or source_state not in source_state_values:
        report["error"] = (
            "Dataset provenance is not bound to the selected source_best_train_state_mode"
        )
        return False, report
    if not active_state_values or active_state not in active_state_values:
        report["error"] = (
            "Dataset provenance is not bound to the selected active_train_state_mode"
        )
        return False, report
    if active_state_name not in set(provenance.get("state_modes", [])):
        report["error"] = (
            "Dataset state_modes do not include the expected active state mode"
        )
        return False, report
    report["passed_preflight"] = True
    return True, report


def _strict_rollout_summary(rollout: dict[str, Any]) -> dict[str, Any]:
    drawer_trace = np.asarray(
        rollout.get(
            "next_drawer_fractions", rollout.get("absolute_drawer_fraction", [])
        ),
        dtype=np.float32,
    ).reshape(-1)
    attached_trace = np.asarray(rollout.get("attached_trace", []), dtype=bool).reshape(
        -1
    )
    if drawer_trace.size:
        strict = evaluate_strict_success(drawer_trace, attached_trace)
    else:
        strict = {
            "strict_success": bool(rollout.get("success", False)),
            "strict_success_version": STRICT_SUCCESS_VERSION,
        }
    strict["strict_success_version"] = STRICT_SUCCESS_VERSION
    return strict


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, int(mask.size)))
    return runs


def _mask_empty_probe(item: dict[str, Any]) -> bool:
    seg_present = "segmentation_mask_support_rate_secondary" in item
    iso_present = "isolated_mask_support_rate_secondary" in item
    seg_empty = (
        seg_present
        and float(item.get("segmentation_mask_support_rate_secondary", 0.0) or 0.0)
        <= 0.0
    )
    iso_empty = (
        iso_present
        and float(item.get("isolated_mask_support_rate_secondary", 0.0) or 0.0) <= 0.0
    )
    return bool(seg_empty or iso_empty)


def _truth_gap_metrics(mask: np.ndarray, start: int, end: int) -> tuple[int, int, int]:
    if end <= start:
        return 0, 0, 0
    window = mask[start:end]
    truthful_ratio = float(window.mean()) if window.size else 0.0
    longest_gap = 0
    gap = 0
    for idx, flag in enumerate(window.tolist()):
        if flag:
            gap = 0
            continue
        is_interior = idx > 0 and idx < (len(window) - 1)
        if is_interior:
            gap += 1
            longest_gap = max(longest_gap, gap)
        else:
            gap = 0
    tail = window[-6:] if window.size >= 6 else window
    tail_truthful = int(np.sum(tail)) if tail.size else 0
    return int(longest_gap), int(tail_truthful), int(round(truthful_ratio * 1000))


def _first_true_index(mask: np.ndarray) -> int | None:
    if mask.size == 0 or not bool(np.any(mask)):
        return None
    return int(np.flatnonzero(mask)[0])


def _bucket_floor_int(value: Any, bucket: int) -> int | str:
    if value is None:
        return "none"
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return "none"
    return int((numeric // bucket) * bucket)


def _bucket_floor_float(value: Any, bucket: float) -> float | str:
    if value is None:
        return "none"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "none"
    return round(float(np.floor(numeric / bucket) * bucket), 4)


def _hash_json_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _sample_signature_points(array: np.ndarray, sample_count: int = 5) -> list[list[float]]:
    if array.ndim != 2 or array.shape[0] <= 0:
        return []
    if array.shape[0] == 1:
        sampled = array[[0]]
    else:
        indices = np.linspace(0, array.shape[0] - 1, num=sample_count, dtype=int)
        sampled = array[indices]
    base = sampled[0].copy()
    rel = sampled - base
    return np.round(rel.astype(np.float32), 2).tolist()


def _stats_signature(values: np.ndarray) -> list[float] | str:
    if values.size == 0:
        return "missing"
    return np.round(
        np.asarray([np.min(values), np.mean(values), values[-1]], dtype=np.float32), 2
    ).tolist()


def _actions_signature(values: np.ndarray) -> dict[str, list[float]] | str:
    if values.size == 0:
        return "missing"
    return {
        "mean": np.round(np.mean(values, axis=0).astype(np.float32), 2).tolist(),
        "std": np.round(np.std(values, axis=0).astype(np.float32), 2).tolist(),
    }


def _close_onset_step(rollout: dict[str, Any], start: int, end: int) -> int | None:
    actions = np.asarray(rollout.get("actions", []), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] <= start or actions.shape[1] < 7:
        return None
    stop = min(int(end), int(actions.shape[0]))
    if stop <= start:
        return None
    close_mask = actions[start:stop, 6] < 0.0
    if not bool(np.any(close_mask)):
        return None
    return int(start + np.flatnonzero(close_mask)[0])


def _bridge_interval(
    rollout: dict[str, Any], n_frames: int
) -> tuple[int | None, int | None, np.ndarray]:
    phase_locked = np.zeros(n_frames, dtype=bool)
    raw_phase_locked = np.asarray(
        rollout.get("phase_locked_trace", []), dtype=np.float32
    ).reshape(-1)
    if raw_phase_locked.size:
        phase_locked[: min(n_frames, raw_phase_locked.size)] = (
            raw_phase_locked[: min(n_frames, raw_phase_locked.size)] > 0
        )
    effective_pull = np.zeros(n_frames, dtype=bool)
    raw_pull = np.asarray(
        rollout.get("effective_pull_progress_trace", []), dtype=np.float32
    ).reshape(-1)
    if raw_pull.size:
        effective_pull[: min(n_frames, raw_pull.size)] = (
            raw_pull[: min(n_frames, raw_pull.size)] > 0
        )
    bridge_mask = np.logical_or(phase_locked, effective_pull)
    if not bool(np.any(bridge_mask)):
        return None, None, bridge_mask
    indices = np.flatnonzero(bridge_mask)
    return int(indices[0]), int(indices[-1] + 1), bridge_mask


def _canonical_training_truth_summary(rollout: dict[str, Any]) -> dict[str, Any]:
    trace = list(rollout.get("handle_probe_metadata_trace") or [])
    n_frames = len(trace)
    truthful_mask = np.asarray(
        [bool((item or {}).get("measurement_truthful", False)) for item in trace],
        dtype=bool,
    )
    anchor_trace = np.asarray(
        rollout.get("runtime_handle_anchor_valid_trace", []), dtype=bool
    ).reshape(-1)
    bridge_start, bridge_end, bridge_mask = _bridge_interval(rollout, n_frames)
    runtime_handle_anchor_valid = (
        bool(anchor_trace.any())
        if anchor_trace.size
        else bool(rollout.get("runtime_handle_anchor_valid", False))
    )
    final_probe = dict(rollout.get("handle_probe_metadata") or {})
    mask_empty_count = int(sum(1 for item in trace if _mask_empty_probe(item or {})))
    anchor_invalid_count = (
        int(np.sum(~anchor_trace[:n_frames]))
        if anchor_trace.size
        else int(
            sum(
                1
                for item in trace
                if not bool(
                    (item or {}).get(
                        "runtime_handle_anchor_valid", runtime_handle_anchor_valid
                    )
                )
            )
        )
    )
    truth_contract = _truth_contract_payload()
    bridge_score = 0.0
    summary: dict[str, Any] = {
        "teacher_truth_adjudication": _truth_contract_required_str(
            truth_contract, "teacher_truth_predicate"
        ),
        "truth_contract_path": str(TRUTH_CONTRACT_PATH),
        "truth_contract_hash": _truth_contract_hash(truth_contract),
        "truth_contract_root_object": truth_contract.get("truth_root_object"),
        "truthful_step_count": int(truthful_mask.sum()),
        "truthful_step_ratio": float(
            float(truthful_mask.mean()) if truthful_mask.size else 0.0
        ),
        "truthful_window_start": None,
        "truthful_window_end": None,
        "truthful_window_frame_count": 0,
        "truthful_window_ratio": 0.0,
        "truthful_window_longest_interior_gap": 0,
        "truthful_window_tail_truthful_count_last6": 0,
        "bridge_in_truthful_window": False,
        "measurement_truthful_for_training": False,
        "runtime_handle_anchor_valid": runtime_handle_anchor_valid,
        "truthful_window_interval_mask_empty": False,
        "truthful_window_interval_anchor_invalid": False,
        "bridge_start": bridge_start,
        "bridge_end": None
        if bridge_end is None
        else int(max(bridge_end - 1, bridge_start or 0)),
        "truthful_training_window_start": None,
        "truthful_training_window_end": None,
        "truthful_training_window_len": 0,
        "truthful_training_window_contains_bridge": False,
        "truthful_training_window_bridge_score": 0.0,
        "mask_empty_count": mask_empty_count,
        "anchor_invalid_count": anchor_invalid_count,
        "final_snapshot_measurement_truthful": bool(
            final_probe.get("measurement_truthful", False)
        ),
        "final_snapshot_measurement_truth_tier": final_probe.get(
            "measurement_truth_tier"
        ),
    }
    if bridge_start is None or bridge_end is None or bridge_end <= bridge_start:
        return summary
    window_mask = truthful_mask[bridge_start:bridge_end]
    window_count = int(bridge_end - bridge_start)
    truthful_ratio = float(window_mask.mean()) if window_mask.size else 0.0
    longest_gap, tail_truthful, _ = _truth_gap_metrics(
        truthful_mask, bridge_start, bridge_end
    )
    interval_trace = trace[bridge_start:bridge_end]
    interval_mask_empty = any(_mask_empty_probe(item or {}) for item in interval_trace)
    interval_anchor_invalid = (
        bool(np.any(~anchor_trace[bridge_start:bridge_end]))
        if anchor_trace.size
        else any(
            not bool(
                (item or {}).get(
                    "runtime_handle_anchor_valid", runtime_handle_anchor_valid
                )
            )
            for item in interval_trace
        )
    )
    bridge_in_window = bool(np.any(bridge_mask[bridge_start:bridge_end]))
    bridge_score = (
        float(np.mean(bridge_mask[bridge_start:bridge_end]))
        if window_count > 0
        else 0.0
    )
    measurement_truthful_for_training = bool(
        window_count >= TRUTHFUL_WINDOW_MIN_FRAMES
        and truthful_ratio >= 0.80
        and longest_gap <= 2
        and tail_truthful >= min(5, window_count)
        and bridge_in_window
        and (not anchor_trace.size or runtime_handle_anchor_valid)
        and not interval_mask_empty
        and not interval_anchor_invalid
    )
    summary.update(
        {
            "truthful_window_start": int(bridge_start),
            "truthful_window_end": int(bridge_end),
            "truthful_window_frame_count": window_count,
            "truthful_window_ratio": truthful_ratio,
            "truthful_window_longest_interior_gap": int(longest_gap),
            "truthful_window_tail_truthful_count_last6": int(tail_truthful),
            "bridge_in_truthful_window": bridge_in_window,
            "truthful_window_interval_mask_empty": bool(interval_mask_empty),
            "truthful_window_interval_anchor_invalid": bool(interval_anchor_invalid),
            "measurement_truthful_for_training": measurement_truthful_for_training,
            "truthful_training_window_start": int(bridge_start),
            "truthful_training_window_end": int(bridge_end),
            "truthful_training_window_len": window_count,
            "truthful_training_window_contains_bridge": bridge_in_window,
            "truthful_training_window_bridge_score": bridge_score,
        }
    )
    return summary


def _learning_support_truth_summary(rollout: dict[str, Any]) -> dict[str, Any]:
    trace = list(rollout.get("handle_probe_metadata_trace") or [])
    n_frames = len(trace)
    truthful_mask = np.asarray(
        [bool((item or {}).get("measurement_truthful", False)) for item in trace],
        dtype=bool,
    )
    anchor_trace = np.asarray(
        rollout.get("runtime_handle_anchor_valid_trace", []), dtype=bool
    ).reshape(-1)
    attach_eligible_trace = np.asarray(
        rollout.get("attach_eligible_trace", []), dtype=bool
    ).reshape(-1)
    bridge_start, bridge_end, bridge_mask = _bridge_interval(rollout, n_frames)
    strict = dict(rollout.get("strict_metrics") or {})
    attach_step_raw = rollout.get("attach_step")
    attach_step = (
        int(attach_step_raw)
        if attach_step_raw is not None
        else (
            int(strict.get("first_attach_step"))
            if strict.get("first_attach_step") is not None
            else None
        )
    )
    first_attach_eligible_step = (
        _first_true_index(attach_eligible_trace[:n_frames]) if attach_eligible_trace.size else None
    )
    support_anchor_candidates = [
        int(value)
        for value in (bridge_start, attach_step, first_attach_eligible_step)
        if value is not None
    ]
    support_anchor_step = (
        min(support_anchor_candidates) if support_anchor_candidates else None
    )
    runtime_handle_anchor_valid = bool(rollout.get("runtime_handle_anchor_valid", False))
    summary: dict[str, Any] = {
        "learning_support_truth_adjudication": LEARNING_SUPPORT_TRUTH_VERSION,
        "bridge_start": bridge_start,
        "bridge_end": bridge_end,
        "learning_support_window_start": None,
        "learning_support_window_end": None,
        "learning_support_window_len": 0,
        "learning_support_contains_prebridge": False,
        "learning_support_contains_bridge": False,
        "learning_support_prebridge_frame_count": 0,
        "learning_support_prebridge_truthful_ratio": 0.0,
        "learning_support_whole_window_truthful_ratio": 0.0,
        "learning_support_bridge_truthful_ratio": 0.0,
        "learning_support_bridge_longest_interior_gap": 0,
        "learning_support_bridge_tail_truthful_count_last6": 0,
        "learning_support_interval_mask_empty": False,
        "learning_support_interval_anchor_invalid": False,
        "first_attach_eligible_step": first_attach_eligible_step,
        "learning_support_anchor_step": support_anchor_step,
        "measurement_truthful_for_learning_support": False,
    }
    if (
        bridge_start is None
        or bridge_end is None
        or bridge_end <= bridge_start
        or support_anchor_step is None
    ):
        return summary
    support_start = max(0, int(support_anchor_step) - LEARNING_SUPPORT_PREBRIDGE_FRAMES)
    support_end = int(bridge_end)
    support_window_len = int(max(0, support_end - support_start))
    prebridge_frame_count = int(max(0, bridge_start - support_start))
    support_interval_trace = trace[support_start:support_end]
    bridge_interval_trace = trace[bridge_start:bridge_end]
    whole_window_truthful_ratio = (
        float(np.mean(truthful_mask[support_start:support_end]))
        if support_window_len > 0
        else 0.0
    )
    prebridge_truthful_ratio = (
        float(np.mean(truthful_mask[support_start:bridge_start]))
        if prebridge_frame_count > 0
        else 0.0
    )
    bridge_window_truthful_ratio = (
        float(np.mean(truthful_mask[bridge_start:bridge_end]))
        if bridge_end > bridge_start
        else 0.0
    )
    bridge_longest_gap, bridge_tail_truthful, _ = _truth_gap_metrics(
        truthful_mask, bridge_start, bridge_end
    )
    interval_mask_empty = any(
        _mask_empty_probe(item or {}) for item in bridge_interval_trace
    )
    interval_anchor_invalid = (
        bool(np.any(~anchor_trace[bridge_start:bridge_end]))
        if anchor_trace.size
        else any(
            not bool(
                (item or {}).get(
                    "runtime_handle_anchor_valid", runtime_handle_anchor_valid
                )
            )
            for item in bridge_interval_trace
        )
    )
    contains_bridge = bool(np.any(bridge_mask[bridge_start:bridge_end]))
    contains_prebridge = bool(prebridge_frame_count > 0)
    measurement_truthful_for_learning_support = bool(
        support_window_len >= LEARNING_SUPPORT_WINDOW_MIN_FRAMES
        and prebridge_frame_count >= LEARNING_SUPPORT_MIN_PREBRIDGE_FRAMES
        and contains_bridge
        and contains_prebridge
        and whole_window_truthful_ratio >= 0.60
        and bridge_window_truthful_ratio >= 0.80
        and bridge_longest_gap <= 2
        and bridge_tail_truthful >= 5
        and not interval_mask_empty
        and not interval_anchor_invalid
    )
    summary.update(
        {
            "learning_support_window_start": support_start,
            "learning_support_window_end": support_end,
            "learning_support_window_len": support_window_len,
            "learning_support_contains_prebridge": contains_prebridge,
            "learning_support_contains_bridge": contains_bridge,
            "learning_support_prebridge_frame_count": prebridge_frame_count,
            "learning_support_prebridge_truthful_ratio": prebridge_truthful_ratio,
            "learning_support_whole_window_truthful_ratio": whole_window_truthful_ratio,
            "learning_support_bridge_truthful_ratio": bridge_window_truthful_ratio,
            "learning_support_bridge_longest_interior_gap": int(bridge_longest_gap),
            "learning_support_bridge_tail_truthful_count_last6": int(
                bridge_tail_truthful
            ),
            "learning_support_interval_mask_empty": bool(interval_mask_empty),
            "learning_support_interval_anchor_invalid": bool(interval_anchor_invalid),
            "measurement_truthful_for_learning_support": (
                measurement_truthful_for_learning_support
            ),
        }
    )
    return summary




def _normalize_vector(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return fallback.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _approach_alignment_trace_from_rollout(
    rollout: dict[str, Any], n_frames: int
) -> np.ndarray:
    states = np.asarray(rollout.get("states", []), dtype=np.float32)
    if states.ndim != 2 or states.shape[0] <= 0 or states.shape[1] < 3:
        return np.zeros((n_frames,), dtype=np.float32)
    usable = min(int(n_frames), int(states.shape[0]))
    out = np.zeros((n_frames,), dtype=np.float32)
    eef_pos = states[:usable, :3]
    handle_anchor = np.asarray(
        (rollout.get("orientation_telemetry") or {}).get(
            "runtime_handle_anchor_world"
        )
        or [0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    world_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for idx in range(1, usable):
        delta = _normalize_vector(eef_pos[idx] - eef_pos[idx - 1], world_x)
        desired = _normalize_vector(handle_anchor - eef_pos[idx - 1], world_x)
        out[idx] = float(np.clip(np.dot(delta, desired), -1.0, 1.0))
    return out


def _orientation_support_truth_summary(rollout: dict[str, Any]) -> dict[str, Any]:
    trace = list(rollout.get("handle_probe_metadata_trace") or [])
    n_frames = len(trace)
    summary = {
        "orientation_support_truth_adjudication": ORIENTATION_SUPPORT_TRUTH_VERSION,
        "orientation_support_window_start": None,
        "orientation_support_window_end": None,
        "orientation_support_window_len": 0,
        "orientation_context_frame_count": 0,
        "orientation_support_contains_distance_pass": False,
        "orientation_support_contains_approach_pass": False,
        "orientation_support_contains_orientation_correction": False,
        "orientation_support_contains_attach_eligible": False,
        "orientation_error_start": 0.0,
        "orientation_error_end": 0.0,
        "orientation_error_delta": 0.0,
        "orientation_alignment_start": 0.0,
        "orientation_alignment_end": 0.0,
        "orientation_alignment_delta": 0.0,
        "orientation_gate_crossed": False,
        "orientation_support_truthful_ratio": 0.0,
        "orientation_support_anchor_valid": False,
        "orientation_support_anchor_valid_ratio": 0.0,
        "measurement_truthful_for_orientation_support": False,
        "orientation_support_core_start": None,
        "orientation_support_core_end": None,
        "orientation_support_approach_alignment_start": 0.0,
        "orientation_support_distance_to_handle_start": 0.0,
        "orientation_support_phase_orientation_schedule_hash": "missing",
    }
    if n_frames <= 0:
        return summary
    truthful_mask = np.asarray(
        [bool((item or {}).get("measurement_truthful", False)) for item in trace],
        dtype=bool,
    )
    anchor_trace = np.asarray(
        rollout.get("runtime_handle_anchor_valid_trace", []), dtype=bool
    ).reshape(-1)
    distance_trace = np.asarray(
        rollout.get("handle_distance_trace", []), dtype=np.float32
    ).reshape(-1)
    orientation_alignment = np.asarray(
        rollout.get("orientation_alignment_trace", []), dtype=np.float32
    ).reshape(-1)
    orientation_error = np.asarray(
        rollout.get("orientation_error_trace", []), dtype=np.float32
    ).reshape(-1)
    orientation_gate = np.asarray(
        rollout.get("orientation_gate_trace", []), dtype=bool
    ).reshape(-1)
    attach_eligible = np.asarray(
        rollout.get("attach_eligible_trace", []), dtype=bool
    ).reshape(-1)
    attached = np.asarray(rollout.get("attached_trace", []), dtype=bool).reshape(-1)
    stable_attach = np.asarray(
        rollout.get("stable_attach_trace", []), dtype=bool
    ).reshape(-1)
    approach_alignment = _approach_alignment_trace_from_rollout(rollout, n_frames)
    phase_labels = [str(x) for x in rollout.get("phase_labels", [])]
    _, _, bridge_mask = _bridge_interval(rollout, n_frames)

    usable = min(
        n_frames,
        int(distance_trace.size) if distance_trace.size else n_frames,
        int(orientation_alignment.size) if orientation_alignment.size else n_frames,
    )
    if usable <= 0:
        return summary

    def _pad_bool(values: np.ndarray, fill: bool = False) -> np.ndarray:
        padded = np.full((usable,), fill, dtype=bool)
        if values.size:
            padded[: min(values.size, usable)] = values[: min(values.size, usable)]
        return padded

    def _pad_float(values: np.ndarray, fill: float) -> np.ndarray:
        padded = np.full((usable,), fill, dtype=np.float32)
        if values.size:
            padded[: min(values.size, usable)] = values[: min(values.size, usable)]
        return padded

    distance_trace = _pad_float(distance_trace, np.inf)
    orientation_alignment = _pad_float(orientation_alignment, 0.0)
    orientation_error = _pad_float(orientation_error, 1.0)
    orientation_gate = _pad_bool(orientation_gate)
    attach_eligible = _pad_bool(attach_eligible)
    attached = _pad_bool(attached)
    stable_attach = _pad_bool(stable_attach)
    anchor_mask = (
        _pad_bool(anchor_trace)
        if anchor_trace.size
        else np.full((usable,), bool(rollout.get("runtime_handle_anchor_valid", False)), dtype=bool)
    )
    truthful_window = np.zeros((usable,), dtype=bool)
    truthful_window[: min(usable, truthful_mask.size)] = truthful_mask[: min(usable, truthful_mask.size)]
    approach_window = approach_alignment[:usable]
    bridge_window = bridge_mask[:usable] if bridge_mask.size else np.zeros((usable,), dtype=bool)

    distance_pass = distance_trace <= ORIENTATION_ATTACH_THRESHOLD_M
    near_distance = np.logical_or(distance_pass, distance_trace <= ORIENTATION_NEAR_DISTANCE_THRESHOLD_M)
    approach_ok = approach_window >= ORIENTATION_APPROACH_THRESHOLD
    near_approach = approach_window >= ORIENTATION_NEAR_APPROACH_THRESHOLD
    orientation_not_ok = orientation_alignment < ORIENTATION_THRESHOLD
    before_attach = np.logical_not(np.logical_or(attach_eligible, stable_attach))
    start_mask = near_distance & np.logical_or(approach_ok, near_approach) & orientation_not_ok & before_attach
    orientation_start = _first_true_index(start_mask)
    summary["orientation_support_core_start"] = orientation_start
    if orientation_start is None:
        return summary

    stop = min(usable, orientation_start + ORIENTATION_WINDOW_LOOKAHEAD + 1)
    end_event = None
    for idx in range(orientation_start + 1, stop):
        if (
            bool(orientation_gate[idx])
            or bool(attach_eligible[idx])
            or bool(attached[idx])
            or bool(stable_attach[idx])
            or bool(bridge_window[idx])
        ):
            end_event = idx
            break
    if end_event is None:
        end_event = max(orientation_start + 1, stop - 1)
    start = max(0, orientation_start - ORIENTATION_CONTEXT_PRE_FRAMES)
    end = min(usable, end_event + 1 + ORIENTATION_CONTEXT_POST_FRAMES)
    if end <= start:
        return summary

    window_len = int(end - start)
    core_start = int(orientation_start)
    core_end = int(end_event + 1)
    truthful_ratio = float(np.mean(truthful_window[start:end])) if window_len > 0 else 0.0
    anchor_valid_ratio = float(np.mean(anchor_mask[start:end])) if window_len > 0 else 0.0
    alignment_start = float(orientation_alignment[orientation_start])
    alignment_end = float(orientation_alignment[end_event])
    alignment_delta = float(alignment_end - alignment_start)
    error_start = float(orientation_error[orientation_start])
    error_end = float(orientation_error[end_event])
    error_delta = float(error_end - error_start)
    gate_crossed = bool(np.any(orientation_gate[core_start:core_end]))
    contains_attach_eligible = bool(np.any(attach_eligible[start:end]))
    contains_distance_pass = bool(np.any(distance_pass[start:end]))
    contains_approach_pass = bool(np.any(np.logical_or(approach_ok[start:end], near_approach[start:end])))
    contains_orientation_correction = bool(
        alignment_delta >= ORIENTATION_ALIGNMENT_MIN_DELTA or gate_crossed
    )
    # Count the transition event frame itself as usable orientation support context.
    context_frames = int((orientation_start - start) + (end - core_end) + 1)
    summary.update(
        {
            "orientation_support_window_start": int(start),
            "orientation_support_window_end": int(end),
            "orientation_support_window_len": window_len,
            "orientation_context_frame_count": context_frames,
            "orientation_support_contains_distance_pass": contains_distance_pass,
            "orientation_support_contains_approach_pass": contains_approach_pass,
            "orientation_support_contains_orientation_correction": contains_orientation_correction,
            "orientation_support_contains_attach_eligible": contains_attach_eligible,
            "orientation_error_start": error_start,
            "orientation_error_end": error_end,
            "orientation_error_delta": error_delta,
            "orientation_alignment_start": alignment_start,
            "orientation_alignment_end": alignment_end,
            "orientation_alignment_delta": alignment_delta,
            "orientation_gate_crossed": gate_crossed,
            "orientation_support_truthful_ratio": truthful_ratio,
            "orientation_support_anchor_valid": bool(anchor_valid_ratio >= 0.80),
            "orientation_support_anchor_valid_ratio": anchor_valid_ratio,
            "measurement_truthful_for_orientation_support": bool(
                window_len >= ORIENTATION_SUPPORT_WINDOW_MIN_FRAMES
                and truthful_ratio >= 0.60
                and anchor_valid_ratio >= 0.80
                and (
                    alignment_delta >= ORIENTATION_ALIGNMENT_MIN_DELTA
                    or gate_crossed
                    or contains_attach_eligible
                )
            ),
            "orientation_support_core_start": core_start,
            "orientation_support_core_end": core_end,
            "orientation_support_approach_alignment_start": float(approach_window[orientation_start]),
            "orientation_support_distance_to_handle_start": float(distance_trace[orientation_start]),
            "orientation_support_phase_orientation_schedule_hash": _hash_json_payload(
                phase_labels[start:end] if phase_labels else []
            ),
        }
    )
    return summary


def _orientation_support_teacher_class(rollout: dict[str, Any]) -> str:
    support = dict(rollout.get("orientation_support_truth") or {})
    if not bool(support.get("measurement_truthful_for_orientation_support", False)):
        return "rejected_orientation_teacher"
    attach_eligible_trace = np.asarray(
        rollout.get("attach_eligible_trace", []), dtype=bool
    ).reshape(-1)
    window_end = support.get("orientation_support_window_end")
    immediate_post_attach_eligible = False
    if window_end is not None and attach_eligible_trace.size:
        start = min(int(window_end), int(attach_eligible_trace.size))
        stop = min(start + 2, int(attach_eligible_trace.size))
        immediate_post_attach_eligible = bool(np.any(attach_eligible_trace[start:stop]))
    strong_orientation_transition = float(
        support.get("orientation_alignment_delta", 0.0) or 0.0
    ) >= ORIENTATION_TRANSITION_CLASS_DELTA
    sufficient_transition_context = int(
        support.get("orientation_context_frame_count", 0) or 0
    ) >= ORIENTATION_TRANSITION_CLASS_MIN_CONTEXT_FRAMES
    if strong_orientation_transition and sufficient_transition_context:
        return "orientation_transition_teacher"
    if bool(support.get("orientation_support_contains_attach_eligible", False)) or immediate_post_attach_eligible:
        return "attach_eligible_transition_teacher"
    if bool(support.get("orientation_gate_crossed", False)) or float(
        support.get("orientation_alignment_delta", 0.0) or 0.0
    ) >= ORIENTATION_TRANSITION_DELTA:
        return "orientation_transition_teacher"
    if bool(support.get("orientation_support_contains_distance_pass", False)) and bool(
        support.get("orientation_support_contains_approach_pass", False)
    ):
        return "orientation_context_teacher"
    return "rejected_orientation_teacher"
def _orientation_support_fingerprint(rollout: dict[str, Any]) -> str:
    support = dict(rollout.get("orientation_support_truth") or {})
    quats = np.asarray(rollout.get("eef_quat_trace", []), dtype=np.float32)
    if quats.ndim != 2:
        quats = np.asarray([], dtype=np.float32).reshape(0, 4)
    actions = np.asarray(rollout.get("actions", []), dtype=np.float32)
    if actions.ndim != 2:
        actions = np.asarray([], dtype=np.float32).reshape(0, 7)
    start = int(support.get("orientation_support_window_start") or 0)
    end = int(support.get("orientation_support_window_end") or 0)
    phase_labels = [str(x) for x in rollout.get("phase_labels", [])]
    proposal_source = str(
        rollout.get("orientation_prior_source")
        or rollout.get("proposal_source")
        or "native_teacher"
    )
    payload = {
        "seed": int(rollout.get("seed", -1) or -1),
        "teacher_family_variant": str(rollout.get("teacher_family_variant") or "base"),
        "orientation_start_bucket": _bucket_floor_int(
            support.get("orientation_support_core_start"), 8
        ),
        "orientation_window_len_bucket": _bucket_floor_int(
            support.get("orientation_support_window_len", 0), 8
        ),
        "orientation_alignment_start_bucket": _bucket_floor_float(
            support.get("orientation_alignment_start"), 0.1
        ),
        "orientation_alignment_delta_bucket": _bucket_floor_float(
            support.get("orientation_alignment_delta"), 0.1
        ),
        "approach_alignment_bucket": _bucket_floor_float(
            support.get("orientation_support_approach_alignment_start"), 0.1
        ),
        "distance_to_handle_bucket": _bucket_floor_float(
            support.get("orientation_support_distance_to_handle_start"), 0.02
        ),
        "eef_orientation_path_signature": _hash_json_payload(
            _sample_signature_points(quats[start:end])
        )
        if quats.size and end > start
        else "missing",
        "action_rotation_signature": _hash_json_payload(
            _actions_signature(actions[start:end, 3:6])
        )
        if actions.size and end > start and actions.shape[1] >= 6
        else "missing",
        "phase_orientation_schedule_hash": str(
            support.get("orientation_support_phase_orientation_schedule_hash")
            or _hash_json_payload(phase_labels[start:end] if end > start else [])
        ),
        "proposal_source": proposal_source,
    }
    return json.dumps(payload, sort_keys=True)
def _slice_rollout_to_training_window(
    rollout: dict[str, Any], start: int, end: int
) -> dict[str, Any]:
    n_frames = int(len(rollout.get("actions", [])))
    sliced = dict(rollout)
    for key, value in list(rollout.items()):
        if (
            isinstance(value, np.ndarray)
            and value.ndim >= 1
            and value.shape[0] == n_frames
        ):
            sliced[key] = value[start:end]
        elif isinstance(value, list) and len(value) == n_frames:
            sliced[key] = value[start:end]
    sliced["raw_unique_frames"] = int(rollout.get("raw_unique_frames", n_frames))
    sliced["effective_training_frames"] = int(max(0, end - start))
    sliced["training_window"] = {
        "start": int(start),
        "end": int(end),
        "frame_count": int(max(0, end - start)),
    }
    return sliced


def _teacher_episode_class(rollout: dict[str, Any]) -> str:
    strict = dict(rollout.get("strict_metrics") or {})
    truth = dict(rollout.get("canonical_training_truth") or {})
    truth_ok = bool(truth.get("measurement_truthful_for_training", False))
    if bool(strict.get("strict_success", False)) and truth_ok:
        return "strict_teacher"
    near_ok = bool(
        truth_ok
        and bool(truth.get("runtime_handle_anchor_valid", False))
        and bool(strict.get("ever_attached", False))
        and int(strict.get("attach_persistence", 0) or 0) >= 3
        and float(strict.get("post_attach_drawer_delta", 0.0) or 0.0) >= 0.35
        and float(strict.get("max_drawer_fraction", 0.0) or 0.0) >= 0.85
    )
    return "near_strict_teacher" if near_ok else "rejected_teacher"


def _learning_support_teacher_class(rollout: dict[str, Any]) -> str:
    canonical_class = str(rollout.get("teacher_episode_class", "rejected_teacher"))
    if canonical_class in {"strict_teacher", "near_strict_teacher"}:
        return canonical_class
    support = dict(rollout.get("learning_support_truth") or {})
    if bool(
        support.get("measurement_truthful_for_learning_support", False)
        and support.get("learning_support_contains_prebridge", False)
        and support.get("learning_support_contains_bridge", False)
        and (
            bool(rollout.get("ever_attached", False))
            or support.get("first_attach_eligible_step") is not None
            or rollout.get("attach_step") is not None
        )
    ):
        return "learning_support_teacher"
    return "rejected_teacher"


def _teacher_fingerprint(rollout: dict[str, Any]) -> str:
    strict = dict(rollout.get("strict_metrics") or {})
    truth = dict(rollout.get("canonical_training_truth") or {})
    phase_labels = [str(x) for x in rollout.get("phase_labels", [])]
    phase_hash = hashlib.sha256("|".join(phase_labels).encode("utf-8")).hexdigest()[:16]
    payload = {
        "seed": int(rollout.get("seed", -1) or -1),
        "first_attach_step": strict.get("first_attach_step"),
        "attach_persistence": int(strict.get("attach_persistence", 0) or 0),
        "max_drawer_fraction": round(
            float(
                strict.get(
                    "max_drawer_fraction", rollout.get("max_drawer_fraction", 0.0)
                )
                or 0.0
            ),
            4,
        ),
        "truthful_window_frame_count": int(
            truth.get("truthful_window_frame_count", 0) or 0
        ),
        "phase_locked_rate": round(
            float(
                np.mean(
                    np.asarray(rollout.get("phase_locked_trace", []), dtype=np.float32)
                )
            )
            if len(rollout.get("phase_locked_trace", []))
            else 0.0,
            4,
        ),
        "effective_pull_progress_peak": round(
            float(
                np.max(
                    np.asarray(
                        rollout.get("effective_pull_progress_trace", []),
                        dtype=np.float32,
                    )
                )
            )
            if len(rollout.get("effective_pull_progress_trace", []))
            else 0.0,
            4,
        ),
        "phase_schedule_hash": phase_hash,
    }
    return json.dumps(payload, sort_keys=True)


def _learning_support_fingerprint(rollout: dict[str, Any]) -> str:
    strict = dict(rollout.get("strict_metrics") or {})
    support = dict(rollout.get("learning_support_truth") or {})
    phase_labels = [str(x) for x in rollout.get("phase_labels", [])]
    bridge_start = support.get("bridge_start")
    approach_labels = (
        phase_labels[: int(bridge_start)]
        if bridge_start is not None
        else list(phase_labels)
    )
    approach_phase_schedule_hash = hashlib.sha256(
        "|".join(approach_labels).encode("utf-8")
    ).hexdigest()[:16]
    payload = {
        "seed": int(rollout.get("seed", -1) or -1),
        "first_attach_step_bucket": _bucket_floor_int(
            strict.get("first_attach_step"), 8
        ),
        "first_attach_eligible_step_bucket": _bucket_floor_int(
            support.get("first_attach_eligible_step"), 8
        ),
        "learning_support_prebridge_frame_count_bucket": _bucket_floor_int(
            support.get("learning_support_prebridge_frame_count", 0), 8
        ),
        "learning_support_window_len_bucket": _bucket_floor_int(
            support.get("learning_support_window_len", 0), 8
        ),
        "learning_support_prebridge_truthful_ratio_bucket": _bucket_floor_float(
            support.get("learning_support_prebridge_truthful_ratio"), 0.1
        ),
        "attach_persistence_bucket": _bucket_floor_int(
            strict.get("attach_persistence", 0), 4
        ),
        "max_drawer_fraction_bucket": _bucket_floor_float(
            strict.get(
                "max_drawer_fraction", rollout.get("max_drawer_fraction", 0.0)
            ),
            0.05,
        ),
        "approach_phase_schedule_hash": approach_phase_schedule_hash,
    }
    return json.dumps(payload, sort_keys=True)


def _effective_support_signature_payload(rollout: dict[str, Any]) -> dict[str, Any]:
    support = dict(rollout.get("learning_support_truth") or {})
    strict = dict(rollout.get("strict_metrics") or {})
    phase_labels = [str(x) for x in rollout.get("phase_labels", [])]
    support_start = support.get("learning_support_window_start")
    bridge_start = support.get("bridge_start")
    bridge_end = support.get("bridge_end")
    if support_start is None or bridge_start is None or bridge_end is None:
        return {
            "seed": int(rollout.get("seed", -1) or -1),
            "first_attach_eligible_step_bucket": "none",
            "first_attach_step_bucket": "none",
            "close_onset_step_bucket": "none",
            "prebridge_len_bucket": "none",
            "phase_prebridge_hash": "missing",
            "eef_path_signature_hash": "missing",
            "action_signature_hash": "missing",
            "distance_curve_signature_hash": "missing",
            "orientation_curve_signature_hash": "missing",
        }
    support_start = int(support_start)
    bridge_start = int(bridge_start)
    bridge_end = int(bridge_end)
    if bridge_start < support_start:
        bridge_start = support_start
    prebridge_end = max(support_start, bridge_start)
    phase_prebridge = phase_labels[support_start:prebridge_end]
    states = np.asarray(rollout.get("states", []), dtype=np.float32)
    if states.ndim == 2 and states.shape[1] >= 3:
        eef_prebridge = states[support_start:prebridge_end, :3]
    else:
        eef_prebridge = np.asarray([], dtype=np.float32).reshape(0, 3)
    actions = np.asarray(rollout.get("actions", []), dtype=np.float32)
    if actions.ndim == 2:
        actions_prebridge = actions[support_start:prebridge_end]
    else:
        actions_prebridge = np.asarray([], dtype=np.float32).reshape(0, 7)
    handle_distance = np.asarray(
        rollout.get("handle_distance_trace", []), dtype=np.float32
    ).reshape(-1)
    distance_prebridge = handle_distance[support_start:prebridge_end]
    orientation_alignment = np.asarray(
        rollout.get("orientation_alignment_trace", []), dtype=np.float32
    ).reshape(-1)
    orientation_error = np.asarray(
        rollout.get("orientation_error_trace", []), dtype=np.float32
    ).reshape(-1)
    if orientation_alignment.size >= prebridge_end and prebridge_end > support_start:
        orientation_values = orientation_alignment[support_start:prebridge_end]
    elif orientation_error.size >= prebridge_end and prebridge_end > support_start:
        orientation_values = orientation_error[support_start:prebridge_end]
    else:
        orientation_values = np.asarray([], dtype=np.float32)
    close_onset_step = _close_onset_step(rollout, support_start, bridge_end)
    payload = {
        "seed": int(rollout.get("seed", -1) or -1),
        "first_attach_eligible_step_bucket": _bucket_floor_int(
            support.get("first_attach_eligible_step"), 8
        ),
        "first_attach_step_bucket": _bucket_floor_int(
            strict.get("first_attach_step"), 8
        ),
        "close_onset_step_bucket": _bucket_floor_int(close_onset_step, 8),
        "prebridge_len_bucket": _bucket_floor_int(
            support.get("learning_support_prebridge_frame_count", 0), 8
        ),
        "phase_prebridge_hash": _hash_json_payload(phase_prebridge)
        if phase_prebridge
        else "missing",
        "eef_path_signature_hash": _hash_json_payload(
            _sample_signature_points(eef_prebridge)
        )
        if eef_prebridge.size
        else "missing",
        "action_signature_hash": _hash_json_payload(_actions_signature(actions_prebridge))
        if actions_prebridge.size
        else "missing",
        "distance_curve_signature_hash": _hash_json_payload(
            _stats_signature(distance_prebridge)
        )
        if distance_prebridge.size
        else "missing",
        "orientation_curve_signature_hash": _hash_json_payload(
            _stats_signature(orientation_values)
        )
        if orientation_values.size
        else "missing",
    }
    return payload


def _effective_support_signature_v1(rollout: dict[str, Any]) -> str:
    return json.dumps(_effective_support_signature_payload(rollout), sort_keys=True)


def _augment_rollout_metadata(
    controller: RootCauseController,
    rollout: dict[str, Any],
    spec,
    expected: dict[str, Any],
) -> dict[str, Any]:
    matrix_hash = (
        controller._frozen_matrix_hash_v5_pro()
        if controller.selector_mode == "frozen_v5_pro"
        else None
    )
    baseline_cell_id = (
        controller._baseline_cell_id_v5_pro()
        if controller.selector_mode == "frozen_v5_pro"
        else None
    )
    ts_baseline_cell_id = (
        controller._ts_baseline_cell_id_v6_1()
        if controller.selector_mode == "frozen_v5_pro"
        else None
    )
    rollout["resource_budget_snapshot"] = controller._resource_snapshot()
    rollout["selector_mode"] = controller.selector_mode
    rollout["baseline_cell_id"] = baseline_cell_id
    rollout["ts_baseline_cell_id"] = ts_baseline_cell_id
    rollout["frozen_matrix_hash"] = matrix_hash
    rollout["lane_hash"] = spec.lane_hash
    rollout["best_transition_cell"] = expected.get("best_transition_cell")
    rollout["canonical_train_cell"] = _plan_source_canonical_train_cell(expected)
    rollout["best_train_state_mode"] = _plan_source_best_train_state_mode(expected)
    rollout["source_canonical_train_cell"] = _plan_source_canonical_train_cell(expected)
    rollout["source_best_train_state_mode"] = _plan_source_best_train_state_mode(
        expected
    )
    rollout["active_train_state_mode"] = _plan_active_train_state_mode(expected)
    rollout["active_state_mode_name"] = _plan_active_state_mode_name(expected)
    rollout["run_instance_id"] = expected.get("run_instance_id")
    rollout["plan_version"] = expected.get("plan_version")
    rollout["source_base_commit"] = expected.get("source_base_commit")
    rollout["working_head_commit"] = expected.get("working_head_commit")
    rollout["bridge_stage"] = expected.get("bridge_stage")
    rollout["bridge_attempt"] = expected.get("bridge_attempt")
    rollout["strict_utility_version"] = expected.get(
        "strict_utility_version", STRICT_UTILITY_VERSION
    )
    rollout["truth_utility_version"] = expected.get(
        "truth_utility_version", TRUTH_UTILITY_VERSION
    )
    rollout["teacher_fingerprint_version"] = expected.get(
        "teacher_fingerprint_version", TEACHER_FINGERPRINT_VERSION
    )
    rollout["teacher_truth_gate"] = expected.get(
        "teacher_truth_gate", TRUTH_UTILITY_VERSION
    )
    rollout["train_seeds"] = list(
        expected.get("train_seeds") or expected.get("training_seeds") or []
    )
    rollout["heldout_seeds"] = list(expected.get("heldout_seeds") or [])

    strict = _strict_rollout_summary(rollout)
    rollout["strict_success_version"] = STRICT_SUCCESS_VERSION
    rollout["strict_metrics"] = strict

    probe = dict(rollout.get("handle_probe_metadata") or {})
    orientation = dict(rollout.get("orientation_telemetry") or {})
    contract_config = dict(rollout.get("contract_config") or {})
    state_spec = dict(rollout.get("state_spec") or {})

    rollout["measurement_truthful"] = bool(probe.get("measurement_truthful", False))
    rollout["measurement_truth_tier"] = probe.get("measurement_truth_tier")
    rollout["measurement_backend"] = probe.get("measurement_backend")
    rollout["measurement_verifier"] = probe.get("measurement_verifier")
    rollout["runtime_visible_handle_mapping_source"] = probe.get(
        "runtime_visible_handle_mapping_source"
    )
    runtime_anchor_trace = np.asarray(
        rollout.get("runtime_handle_anchor_valid_trace", []), dtype=bool
    ).reshape(-1)
    rollout["runtime_handle_anchor_valid"] = (
        bool(runtime_anchor_trace.any())
        if runtime_anchor_trace.size
        else bool(orientation.get("runtime_handle_anchor_valid", False))
    )
    rollout["interaction_mode"] = contract_config.get("interaction_mode")
    rollout["state_mode"] = state_spec.get(
        "state_mode", contract_config.get("state_mode")
    )
    rollout["final_snapshot_measurement_truthful"] = bool(
        probe.get("measurement_truthful", False)
    )
    rollout["final_snapshot_measurement_truth_tier"] = probe.get(
        "measurement_truth_tier"
    )

    truth_summary = _canonical_training_truth_summary(rollout)
    learning_support_summary = _learning_support_truth_summary(rollout)
    orientation_support_summary = _orientation_support_truth_summary(rollout)
    rollout["canonical_training_truth"] = truth_summary
    rollout["learning_support_truth"] = learning_support_summary
    rollout["orientation_support_truth"] = orientation_support_summary
    rollout["measurement_truthful_for_training"] = bool(
        truth_summary.get("measurement_truthful_for_training", False)
    )
    rollout["measurement_truthful_for_learning_support"] = bool(
        learning_support_summary.get("measurement_truthful_for_learning_support", False)
    )
    rollout["measurement_truthful_for_orientation_support"] = bool(
        orientation_support_summary.get(
            "measurement_truthful_for_orientation_support", False
        )
    )
    rollout["teacher_truth_adjudication"] = truth_summary.get(
        "teacher_truth_adjudication"
    )
    rollout["learning_support_truth_adjudication"] = learning_support_summary.get(
        "learning_support_truth_adjudication"
    )
    rollout["orientation_support_truth_adjudication"] = orientation_support_summary.get(
        "orientation_support_truth_adjudication"
    )
    rollout["truth_contract_hash"] = truth_summary.get("truth_contract_hash")
    rollout["truth_contract_path"] = truth_summary.get("truth_contract_path")
    rollout["truth_contract_root_object"] = truth_summary.get(
        "truth_contract_root_object"
    )
    rollout["teacher_truthful_window_frame_count"] = int(
        truth_summary.get("truthful_window_frame_count", 0) or 0
    )
    rollout["bridge_in_truthful_window"] = bool(
        truth_summary.get("bridge_in_truthful_window", False)
    )
    rollout["truthful_window_ratio"] = float(
        truth_summary.get("truthful_window_ratio", 0.0) or 0.0
    )
    rollout["truthful_window_longest_interior_gap"] = int(
        truth_summary.get("truthful_window_longest_interior_gap", 0) or 0
    )
    rollout["truthful_window_tail_truthful_count_last6"] = int(
        truth_summary.get("truthful_window_tail_truthful_count_last6", 0) or 0
    )
    rollout["first_attach_eligible_step"] = learning_support_summary.get(
        "first_attach_eligible_step"
    )
    rollout["learning_support_window_start"] = learning_support_summary.get(
        "learning_support_window_start"
    )
    rollout["learning_support_window_end"] = learning_support_summary.get(
        "learning_support_window_end"
    )
    rollout["learning_support_window_len"] = int(
        learning_support_summary.get("learning_support_window_len", 0) or 0
    )
    rollout["learning_support_prebridge_frame_count"] = int(
        learning_support_summary.get("learning_support_prebridge_frame_count", 0) or 0
    )
    rollout["learning_support_prebridge_truthful_ratio"] = float(
        learning_support_summary.get("learning_support_prebridge_truthful_ratio", 0.0)
        or 0.0
    )
    rollout["learning_support_whole_window_truthful_ratio"] = float(
        learning_support_summary.get("learning_support_whole_window_truthful_ratio", 0.0)
        or 0.0
    )
    rollout["learning_support_bridge_truthful_ratio"] = float(
        learning_support_summary.get("learning_support_bridge_truthful_ratio", 0.0)
        or 0.0
    )
    rollout["learning_support_bridge_longest_interior_gap"] = int(
        learning_support_summary.get(
            "learning_support_bridge_longest_interior_gap", 0
        )
        or 0
    )
    rollout["learning_support_bridge_tail_truthful_count_last6"] = int(
        learning_support_summary.get(
            "learning_support_bridge_tail_truthful_count_last6", 0
        )
        or 0
    )
    rollout["learning_support_interval_mask_empty"] = bool(
        learning_support_summary.get("learning_support_interval_mask_empty", False)
    )
    rollout["learning_support_interval_anchor_invalid"] = bool(
        learning_support_summary.get(
            "learning_support_interval_anchor_invalid", False
        )
    )
    rollout["learning_support_contains_prebridge"] = bool(
        learning_support_summary.get("learning_support_contains_prebridge", False)
    )
    rollout["learning_support_contains_bridge"] = bool(
        learning_support_summary.get("learning_support_contains_bridge", False)
    )
    rollout["learning_support_anchor_step"] = learning_support_summary.get(
        "learning_support_anchor_step"
    )
    rollout["orientation_support_window_start"] = orientation_support_summary.get(
        "orientation_support_window_start"
    )
    rollout["orientation_support_window_end"] = orientation_support_summary.get(
        "orientation_support_window_end"
    )
    rollout["orientation_support_window_len"] = int(
        orientation_support_summary.get("orientation_support_window_len", 0) or 0
    )
    rollout["orientation_context_frame_count"] = int(
        orientation_support_summary.get("orientation_context_frame_count", 0) or 0
    )
    rollout["orientation_support_contains_distance_pass"] = bool(
        orientation_support_summary.get("orientation_support_contains_distance_pass", False)
    )
    rollout["orientation_support_contains_approach_pass"] = bool(
        orientation_support_summary.get("orientation_support_contains_approach_pass", False)
    )
    rollout["orientation_support_contains_orientation_correction"] = bool(
        orientation_support_summary.get(
            "orientation_support_contains_orientation_correction", False
        )
    )
    rollout["orientation_support_contains_attach_eligible"] = bool(
        orientation_support_summary.get("orientation_support_contains_attach_eligible", False)
    )
    rollout["orientation_error_start"] = float(
        orientation_support_summary.get("orientation_error_start", 0.0) or 0.0
    )
    rollout["orientation_error_end"] = float(
        orientation_support_summary.get("orientation_error_end", 0.0) or 0.0
    )
    rollout["orientation_error_delta"] = float(
        orientation_support_summary.get("orientation_error_delta", 0.0) or 0.0
    )
    rollout["orientation_alignment_start"] = float(
        orientation_support_summary.get("orientation_alignment_start", 0.0) or 0.0
    )
    rollout["orientation_alignment_end"] = float(
        orientation_support_summary.get("orientation_alignment_end", 0.0) or 0.0
    )
    rollout["orientation_alignment_delta"] = float(
        orientation_support_summary.get("orientation_alignment_delta", 0.0) or 0.0
    )
    rollout["orientation_gate_crossed"] = bool(
        orientation_support_summary.get("orientation_gate_crossed", False)
    )
    rollout["orientation_support_truthful_ratio"] = float(
        orientation_support_summary.get("orientation_support_truthful_ratio", 0.0)
        or 0.0
    )
    rollout["orientation_support_anchor_valid"] = bool(
        orientation_support_summary.get("orientation_support_anchor_valid", False)
    )
    rollout["orientation_support_anchor_valid_ratio"] = float(
        orientation_support_summary.get("orientation_support_anchor_valid_ratio", 0.0)
        or 0.0
    )
    rollout["orientation_support_core_start"] = orientation_support_summary.get(
        "orientation_support_core_start"
    )
    rollout["orientation_support_core_end"] = orientation_support_summary.get(
        "orientation_support_core_end"
    )
    support_family_repair_mode = str(
        expected.get("support_family_repair_mode") or "G2B"
    )
    rollout["teacher_episode_class"] = _teacher_episode_class(rollout)
    rollout["learning_support_teacher_class"] = _learning_support_teacher_class(rollout)
    rollout["orientation_support_teacher_class"] = _orientation_support_teacher_class(
        rollout
    )
    rollout["teacher_fingerprint"] = _teacher_fingerprint(rollout)
    rollout["support_family_repair_mode"] = support_family_repair_mode
    rollout["effective_support_signature_version"] = EFFECTIVE_SUPPORT_SIGNATURE_VERSION
    rollout["effective_support_signature_v1"] = _effective_support_signature_v1(
        rollout
    )
    if support_family_repair_mode == "G2A":
        rollout["learning_support_fingerprint_version"] = (
            EFFECTIVE_SUPPORT_SIGNATURE_VERSION
        )
        rollout["learning_support_fingerprint"] = rollout[
            "effective_support_signature_v1"
        ]
    else:
        rollout["learning_support_fingerprint_version"] = (
            LEARNING_SUPPORT_FINGERPRINT_VERSION
        )
        rollout["learning_support_fingerprint"] = _learning_support_fingerprint(
            rollout
        )
    rollout["orientation_support_fingerprint_version"] = (
        ORIENTATION_SUPPORT_FINGERPRINT_VERSION
    )
    rollout["orientation_support_fingerprint"] = _orientation_support_fingerprint(
        rollout
    )

    if probe:
        probe["selector_mode"] = controller.selector_mode
        probe["baseline_cell_id"] = baseline_cell_id
        probe["ts_baseline_cell_id"] = ts_baseline_cell_id
        probe["frozen_matrix_hash"] = matrix_hash
        rollout["handle_probe_metadata"] = probe
    trace = []
    for item in rollout.get("handle_probe_metadata_trace", []):
        probe_item = dict(item or {})
        probe_item["selector_mode"] = controller.selector_mode
        probe_item["baseline_cell_id"] = baseline_cell_id
        probe_item["ts_baseline_cell_id"] = ts_baseline_cell_id
        probe_item["frozen_matrix_hash"] = matrix_hash
        trace.append(probe_item)
    if trace:
        rollout["handle_probe_metadata_trace"] = trace
    return rollout


def _percentile_int(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    arr = np.asarray(values, dtype=np.float32)
    return int(round(float(np.percentile(arr, percentile))))


def _materialization_summary(
    *,
    gate_name: str,
    source_dir: Path,
    expected: dict[str, Any],
    train_seeds: list[int],
    episodes_per_seed: int,
    min_train_episodes: int,
    teacher_controller_mode: str,
    teacher_controller_max_steps: int,
    teacher_pull_open_fraction: float,
    stale_source_mismatch: bool,
    attempted_rollouts: int,
    saved_rollouts: int,
    successful_seeds: set[int],
    saved_files: list[str],
    records: list[dict[str, Any]],
    dataset_selection_mode: str,
) -> dict[str, Any]:
    accepted_fingerprints = sorted(
        {
            rec.get("teacher_fingerprint")
            for rec in records
            if rec.get("teacher_episode_class") in {"strict_teacher", "near_strict_teacher"}
            and rec.get("teacher_fingerprint")
        }
    )
    strict_fingerprints = sorted(
        {
            rec.get("teacher_fingerprint")
            for rec in records
            if rec.get("teacher_episode_class") == "strict_teacher"
            and rec.get("teacher_fingerprint")
        }
    )
    near_strict_fingerprints = sorted(
        {
            rec.get("teacher_fingerprint")
            for rec in records
            if rec.get("teacher_episode_class") == "near_strict_teacher"
            and rec.get("teacher_fingerprint")
        }
    )
    recorded_learning_support_fingerprints = sorted(
        {
            rec.get("learning_support_fingerprint")
            for rec in records
            if rec.get("learning_support_teacher_class")
            in {"strict_teacher", "near_strict_teacher", "learning_support_teacher"}
            and rec.get("learning_support_fingerprint")
        }
    )
    effective_support_fingerprints = sorted(
        {
            rec.get("effective_support_signature_v1")
            for rec in records
            if rec.get("learning_support_teacher_class")
            in {"strict_teacher", "near_strict_teacher", "learning_support_teacher"}
            and rec.get("effective_support_signature_v1")
        }
    )
    accepted_family_count_by_seed: dict[str, int] = {}
    strict_family_count_by_seed: dict[str, int] = {}
    near_strict_family_count_by_seed: dict[str, int] = {}
    learning_support_family_count_by_seed: dict[str, int] = {}
    recorded_learning_support_family_count_by_seed: dict[str, int] = {}
    learning_support_seed_coverage: list[int] = []
    attach_eligible_seed_coverage: list[int] = []
    prebridge_counts: list[int] = []
    for seed in train_seeds:
        seed_records = [rec for rec in records if int(rec.get("seed", -1)) == int(seed)]
        accepted_seed = {
            rec.get("teacher_fingerprint")
            for rec in seed_records
            if rec.get("teacher_episode_class") in {"strict_teacher", "near_strict_teacher"}
            and rec.get("teacher_fingerprint")
        }
        strict_seed = {
            rec.get("teacher_fingerprint")
            for rec in seed_records
            if rec.get("teacher_episode_class") == "strict_teacher"
            and rec.get("teacher_fingerprint")
        }
        near_seed = {
            rec.get("teacher_fingerprint")
            for rec in seed_records
            if rec.get("teacher_episode_class") == "near_strict_teacher"
            and rec.get("teacher_fingerprint")
        }
        recorded_support_seed = {
            rec.get("learning_support_fingerprint")
            for rec in seed_records
            if rec.get("learning_support_teacher_class")
            in {"strict_teacher", "near_strict_teacher", "learning_support_teacher"}
            and rec.get("learning_support_fingerprint")
        }
        effective_support_seed = {
            rec.get("effective_support_signature_v1")
            for rec in seed_records
            if rec.get("learning_support_teacher_class")
            in {"strict_teacher", "near_strict_teacher", "learning_support_teacher"}
            and rec.get("effective_support_signature_v1")
        }
        accepted_family_count_by_seed[str(seed)] = len(accepted_seed)
        strict_family_count_by_seed[str(seed)] = len(strict_seed)
        near_strict_family_count_by_seed[str(seed)] = len(near_seed)
        recorded_learning_support_family_count_by_seed[str(seed)] = len(
            recorded_support_seed
        )
        learning_support_family_count_by_seed[str(seed)] = len(effective_support_seed)
        if effective_support_seed:
            learning_support_seed_coverage.append(int(seed))
        if any(
            rec.get("first_attach_eligible_step") is not None
            and rec.get("learning_support_teacher_class")
            in {"strict_teacher", "near_strict_teacher", "learning_support_teacher"}
            for rec in seed_records
        ):
            attach_eligible_seed_coverage.append(int(seed))
        prebridge_counts.extend(
            int(rec.get("learning_support_prebridge_frame_count", 0) or 0)
            for rec in seed_records
            if rec.get("learning_support_teacher_class")
            in {"strict_teacher", "near_strict_teacher", "learning_support_teacher"}
        )
    report: dict[str, Any] = {
        "gate": gate_name,
        "source_dir": str(source_dir),
        **expected,
        "run_instance_id": expected.get("run_instance_id"),
        "working_head_commit": expected.get("working_head_commit"),
        "source_canonical_train_cell": _plan_source_canonical_train_cell(expected),
        "source_best_train_state_mode": _plan_source_best_train_state_mode(expected),
        "active_train_state_mode": _plan_active_train_state_mode(expected),
        "active_state_mode_name": _plan_active_state_mode_name(expected),
        "train_seeds": train_seeds,
        "heldout_seeds": [
            int(seed)
            for seed in (expected.get("heldout_seeds") or DEFAULT_HELD_OUT_SEEDS)
        ],
        "episodes_per_seed": episodes_per_seed,
        "min_train_episodes": min_train_episodes,
        "min_successful_seeds": MIN_SUCCESSFUL_SEEDS,
        "teacher_controller_mode": teacher_controller_mode,
        "teacher_controller_max_steps": teacher_controller_max_steps,
        "teacher_pull_open_fraction": teacher_pull_open_fraction,
        "teacher_truth_gate": _truth_contract_required_str(
            _truth_contract_payload(), "teacher_truth_predicate"
        ),
        "truth_contract_path": str(TRUTH_CONTRACT_PATH),
        "truth_contract_hash": str(
            expected.get("truth_contract_hash") or _truth_contract_hash()
        ),
        "acceptance_contract_hash": str(expected.get("acceptance_contract_hash") or ""),
        "stale_source_mismatch": stale_source_mismatch,
        "strict_utility_version": expected.get(
            "strict_utility_version", STRICT_UTILITY_VERSION
        ),
        "truth_utility_version": expected.get(
            "truth_utility_version", TRUTH_UTILITY_VERSION
        ),
        "teacher_fingerprint_version": expected.get(
            "teacher_fingerprint_version", TEACHER_FINGERPRINT_VERSION
        ),
        "learning_support_fingerprint_version": str(
            expected.get("learning_support_fingerprint_version")
            or LEARNING_SUPPORT_FINGERPRINT_VERSION
        ),
        "effective_support_signature_version": EFFECTIVE_SUPPORT_SIGNATURE_VERSION,
        "support_family_repair_mode": expected.get("support_family_repair_mode"),
        "teacher_family_grid_version": expected.get("teacher_family_grid_version"),
        "dataset_selection_mode": dataset_selection_mode,
        "attempted_rollouts": attempted_rollouts,
        "expected_attempted_rollouts": len(train_seeds) * episodes_per_seed,
        "saved_rollouts": saved_rollouts,
        "raw_saved_rollout_count": saved_rollouts,
        "successful_seed_count": len(successful_seeds),
        "raw_successful_seed_count": len(successful_seeds),
        "successful_seeds": sorted(successful_seeds),
        "saved_files": saved_files,
        "records": records,
        "accepted_unique_teacher_family_count": int(len(accepted_fingerprints)),
        "strict_unique_teacher_family_count": int(len(strict_fingerprints)),
        "near_strict_unique_teacher_family_count": int(len(near_strict_fingerprints)),
        "recorded_learning_support_unique_teacher_family_count": int(
            len(recorded_learning_support_fingerprints)
        ),
        "learning_support_unique_teacher_family_count": int(
            len(effective_support_fingerprints)
        ),
        "accepted_family_count_by_seed": accepted_family_count_by_seed,
        "strict_family_count_by_seed": strict_family_count_by_seed,
        "near_strict_family_count_by_seed": near_strict_family_count_by_seed,
        "recorded_learning_support_family_count_by_seed": (
            recorded_learning_support_family_count_by_seed
        ),
        "learning_support_family_count_by_seed": learning_support_family_count_by_seed,
        "learning_support_seed_coverage": sorted(learning_support_seed_coverage),
        "attach_eligible_seed_coverage": sorted(attach_eligible_seed_coverage),
        "learning_support_prebridge_frame_p50": _percentile_int(prebridge_counts, 50.0),
        "learning_support_prebridge_frame_p90": _percentile_int(prebridge_counts, 90.0),
        "passed": bool(
            attempted_rollouts == len(train_seeds) * episodes_per_seed
            and saved_rollouts >= min_train_episodes
            and len(successful_seeds) >= MIN_SUCCESSFUL_SEEDS
        ),
        "timestamp": time.time(),
    }
    return report




def _enrich_orientation_materialization_report(
    report: dict[str, Any], records: list[dict[str, Any]], train_seeds: list[int]
) -> dict[str, Any]:
    accepted_classes = {
        "orientation_transition_teacher",
        "attach_eligible_transition_teacher",
        "orientation_context_teacher",
    }
    orientation_fingerprints = sorted(
        {
            rec.get("orientation_support_fingerprint")
            for rec in records
            if rec.get("orientation_support_teacher_class") in accepted_classes
            and rec.get("orientation_support_fingerprint")
        }
    )
    family_count_by_seed: dict[str, int] = {}
    seed_coverage: list[int] = []
    truthful_ratios: list[float] = []
    context_counts: list[int] = []
    class_counts = Counter(
        str(rec.get("orientation_support_teacher_class", "rejected_orientation_teacher"))
        for rec in records
    )
    for seed in train_seeds:
        seed_records = [rec for rec in records if int(rec.get("seed", -1)) == int(seed)]
        seed_families = {
            rec.get("orientation_support_fingerprint")
            for rec in seed_records
            if rec.get("orientation_support_teacher_class") in accepted_classes
            and rec.get("orientation_support_fingerprint")
        }
        family_count_by_seed[str(seed)] = len(seed_families)
        if seed_families:
            seed_coverage.append(int(seed))
        truthful_ratios.extend(
            float(rec.get("orientation_support_truthful_ratio", 0.0) or 0.0)
            for rec in seed_records
            if rec.get("orientation_support_teacher_class") in accepted_classes
        )
        context_counts.extend(
            int(rec.get("orientation_context_frame_count", 0) or 0)
            for rec in seed_records
            if rec.get("orientation_support_teacher_class") in accepted_classes
        )
    report.update(
        {
            "orientation_support_unique_teacher_family_count": int(
                len(orientation_fingerprints)
            ),
            "orientation_support_family_count_by_seed": family_count_by_seed,
            "orientation_support_seed_coverage": sorted(seed_coverage),
            "orientation_support_teacher_class_counts": dict(class_counts),
            "orientation_transition_teacher_count": int(
                class_counts.get("orientation_transition_teacher", 0)
            ),
            "attach_eligible_transition_teacher_count": int(
                class_counts.get("attach_eligible_transition_teacher", 0)
            ),
            "orientation_support_truthful_ratio_p50": float(
                np.percentile(np.asarray(truthful_ratios, dtype=np.float32), 50)
            )
            if truthful_ratios
            else 0.0,
            "orientation_support_truthful_ratio_p90": float(
                np.percentile(np.asarray(truthful_ratios, dtype=np.float32), 90)
            )
            if truthful_ratios
            else 0.0,
            "orientation_context_frame_p50": _percentile_int(context_counts, 50.0),
            "orientation_context_frame_p90": _percentile_int(context_counts, 90.0),
        }
    )
    return report
def materialize_canonical_train_rollouts(
    source_dir: Path | None = None,
    *,
    expected: dict[str, Any] | None = None,
    force_rebuild: bool = True,
) -> dict[str, Any]:
    expected = dict(expected or expected_training_targets())
    source_dir = Path(source_dir or DEFAULT_ROLLOUT_SOURCE_DIR)
    train_seeds = [
        int(seed)
        for seed in (
            expected.get("train_seeds")
            or expected.get("training_seeds")
            or DEFAULT_TRAIN_SEEDS
        )
    ]
    episodes_per_seed = int(
        os.environ.get(
            "MINT_G6_EPISODES_PER_SEED",
            str(expected.get("episodes_per_seed", DEFAULT_EPISODES_PER_SEED)),
        )
    )
    min_train_episodes = int(expected.get("min_train_episodes", MIN_TRAIN_EPISODES))
    active_train_state_mode = _plan_active_train_state_mode(expected)
    active_state_mode_name = _plan_active_state_mode_name(expected)
    source_canonical_train_cell = _plan_source_canonical_train_cell(expected)
    teacher_controller_mode = str(
        expected.get("teacher_controller_mode") or "interaction_frame_hybrid"
    )
    teacher_controller_max_steps = int(
        expected.get("teacher_controller_max_steps") or 144
    )
    teacher_pull_open_fraction = float(
        expected.get("teacher_pull_open_fraction") or 0.92
    )
    stale_source_mismatch = bool(
        str(expected.get("teacher_controller_mode") or teacher_controller_mode)
        != teacher_controller_mode
        or str(expected.get("source_canonical_train_cell") or source_canonical_train_cell)
        != source_canonical_train_cell
    )
    learning_support_dir = LEARNING_SUPPORT_ROLLOUT_SOURCE_DIR
    orientation_support_dir = ORIENTATION_SUPPORT_ROLLOUT_SOURCE_DIR
    report_base: dict[str, Any] = {
        **expected,
        "run_instance_id": expected.get("run_instance_id"),
        "working_head_commit": expected.get("working_head_commit"),
        "source_canonical_train_cell": source_canonical_train_cell,
        "source_best_train_state_mode": _plan_source_best_train_state_mode(expected),
        "active_train_state_mode": active_train_state_mode,
        "active_state_mode_name": active_state_mode_name,
        "train_seeds": train_seeds,
        "heldout_seeds": [
            int(seed)
            for seed in (expected.get("heldout_seeds") or DEFAULT_HELD_OUT_SEEDS)
        ],
        "episodes_per_seed": episodes_per_seed,
        "min_train_episodes": min_train_episodes,
        "min_successful_seeds": MIN_SUCCESSFUL_SEEDS,
        "teacher_controller_mode": teacher_controller_mode,
        "teacher_controller_max_steps": teacher_controller_max_steps,
        "teacher_pull_open_fraction": teacher_pull_open_fraction,
        "teacher_truth_gate": _truth_contract_required_str(
            _truth_contract_payload(), "teacher_truth_predicate"
        ),
        "truth_contract_path": str(TRUTH_CONTRACT_PATH),
        "truth_contract_hash": str(
            expected.get("truth_contract_hash") or _truth_contract_hash()
        ),
        "acceptance_contract_hash": str(expected.get("acceptance_contract_hash") or ""),
        "stale_source_mismatch": stale_source_mismatch,
        "strict_utility_version": expected.get(
            "strict_utility_version", STRICT_UTILITY_VERSION
        ),
        "truth_utility_version": expected.get(
            "truth_utility_version", TRUTH_UTILITY_VERSION
        ),
        "teacher_fingerprint_version": expected.get(
            "teacher_fingerprint_version", TEACHER_FINGERPRINT_VERSION
        ),
        "learning_support_fingerprint_version": str(
            expected.get("learning_support_fingerprint_version")
            or LEARNING_SUPPORT_FINGERPRINT_VERSION
        ),
        "effective_support_signature_version": EFFECTIVE_SUPPORT_SIGNATURE_VERSION,
        "support_family_repair_mode": str(
            expected.get("support_family_repair_mode") or "G2B"
        ),
        "teacher_family_grid_version": str(
            expected.get("teacher_family_grid_version") or TEACHER_FAMILY_GRID_VERSION
        ),
    }
    report: dict[str, Any] = {
        "gate": "g6_canonical_rollout_materialization",
        "source_dir": str(source_dir),
        **report_base,
        "dataset_selection_mode": "claim_canonical",
        "timestamp": time.time(),
    }
    learning_support_report: dict[str, Any] = {
        "gate": "g6_learning_support_rollout_materialization",
        "source_dir": str(learning_support_dir),
        **report_base,
        "dataset_selection_mode": "diagnostic_learning_support",
        "timestamp": time.time(),
    }
    orientation_support_report: dict[str, Any] = {
        "gate": "g6_orientation_support_rollout_materialization",
        "source_dir": str(orientation_support_dir),
        **report_base,
        "dataset_selection_mode": "diagnostic_orientation_support",
        "timestamp": time.time(),
    }
    if not bool(expected.get("tiny_retrain_permitted", True)):
        report["passed"] = False
        report["error"] = "RCA7 did not permit tiny retrain"
        learning_support_report.update({"passed": False, "error": report["error"]})
        orientation_support_report.update({"passed": False, "error": report["error"]})
        MATERIALIZATION_ARTIFACT.write_text(json.dumps(report, indent=2))
        LEARNING_SUPPORT_MATERIALIZATION_ARTIFACT.write_text(
            json.dumps(learning_support_report, indent=2)
        )
        ORIENTATION_SUPPORT_MATERIALIZATION_ARTIFACT.write_text(
            json.dumps(orientation_support_report, indent=2)
        )
        return report
    if not source_canonical_train_cell:
        report["passed"] = False
        report["error"] = "Missing source_canonical_train_cell from RCA5/RCA7"
        learning_support_report.update({"passed": False, "error": report["error"]})
        orientation_support_report.update({"passed": False, "error": report["error"]})
        MATERIALIZATION_ARTIFACT.write_text(json.dumps(report, indent=2))
        LEARNING_SUPPORT_MATERIALIZATION_ARTIFACT.write_text(
            json.dumps(learning_support_report, indent=2)
        )
        ORIENTATION_SUPPORT_MATERIALIZATION_ARTIFACT.write_text(
            json.dumps(orientation_support_report, indent=2)
        )
        return report
    if force_rebuild and source_dir.exists():
        shutil.rmtree(source_dir)
    if force_rebuild and learning_support_dir.exists():
        shutil.rmtree(learning_support_dir)
    if force_rebuild and orientation_support_dir.exists():
        shutil.rmtree(orientation_support_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    learning_support_dir.mkdir(parents=True, exist_ok=True)
    orientation_support_dir.mkdir(parents=True, exist_ok=True)

    controller = RootCauseController(
        max_experiments_per_cycle=1,
        dry_run=False,
        experiment_family="RCA",
        selector_mode="frozen_v5_pro",
        seed_split_a=train_seeds,
        truthful_measurement_required=True,
        max_rollouts_per_experiment=max(3, episodes_per_seed),
    )
    spec = controller._matrix_cell_lane_spec(
        source_canonical_train_cell,
        "G6",
        note="Tiny retrain canonical rollout materialization.",
    )
    contract = controller._materialize_contract(spec.env_contract_config)
    if str(contract.state_mode) != active_state_mode_name:
        contract = replace(contract, state_mode=active_state_mode_name)
    interventions = dict(spec.interventions or {})
    rotation_source = str(interventions.get("rotation_source", "zero"))

    attempted_rollouts = 0
    saved_rollouts = 0
    learning_support_saved_rollouts = 0
    orientation_support_saved_rollouts = 0
    successful_seeds: set[int] = set()
    learning_support_successful_seeds: set[int] = set()
    orientation_support_successful_seeds: set[int] = set()
    saved_files: list[str] = []
    learning_support_saved_files: list[str] = []
    orientation_support_saved_files: list[str] = []
    records: list[dict[str, Any]] = []
    for seed in train_seeds:
        for episode_index in range(episodes_per_seed):
            attempted_rollouts += 1
            episode_interventions = dict(interventions)
            episode_interventions.update(
                _teacher_family_variant_payload(expected, int(episode_index))
            )
            rollout = build_robot_rollout(
                seed=int(seed),
                grasp_pose_world=np.eye(4, dtype=np.float32),
                episode_index=int(episode_index),
                max_steps=teacher_controller_max_steps,
                image_size=256,
                contract=contract,
                rotation_source=rotation_source,
                claim_policy=spec.claim_policy,
                interventions=episode_interventions,
                teacher_controller_mode=teacher_controller_mode,
                pull_open_fraction=teacher_pull_open_fraction,
            )
            rollout = _augment_rollout_metadata(controller, rollout, spec, expected)
            truth_summary = dict(rollout.get("canonical_training_truth") or {})
            success = bool(rollout.get("success"))
            phase_locked_trace = np.asarray(
                rollout.get("phase_locked_trace", []), dtype=np.float32
            ).reshape(-1)
            effective_pull_trace = np.asarray(
                rollout.get("effective_pull_progress_trace", []), dtype=np.float32
            ).reshape(-1)
            record = {
                "seed": int(seed),
                "episode_index": int(episode_index),
                "success": success,
                "strict_success": bool(
                    (rollout.get("strict_metrics") or {}).get("strict_success", False)
                ),
                "ever_attached": bool(rollout.get("ever_attached", False)),
                "max_drawer_fraction": float(
                    rollout.get("max_drawer_fraction", 0.0) or 0.0
                ),
                "phase_locked_rate": float(np.mean(phase_locked_trace))
                if phase_locked_trace.size
                else 0.0,
                "effective_pull_progress_peak": float(np.max(effective_pull_trace))
                if effective_pull_trace.size
                else 0.0,
                "measurement_truthful_for_training": bool(
                    truth_summary.get("measurement_truthful_for_training", False)
                ),
                "teacher_truth_adjudication": truth_summary.get(
                    "teacher_truth_adjudication"
                ),
                "teacher_truthful_window_frame_count": int(
                    truth_summary.get("truthful_window_frame_count", 0) or 0
                ),
                "truthful_window_ratio": float(
                    truth_summary.get("truthful_window_ratio", 0.0) or 0.0
                ),
                "truthful_window_longest_interior_gap": int(
                    truth_summary.get("truthful_window_longest_interior_gap", 0) or 0
                ),
                "truthful_window_tail_truthful_count_last6": int(
                    truth_summary.get("truthful_window_tail_truthful_count_last6", 0)
                    or 0
                ),
                "bridge_in_truthful_window": bool(
                    truth_summary.get("bridge_in_truthful_window", False)
                ),
                "truthful_window_interval_mask_empty": bool(
                    truth_summary.get("truthful_window_interval_mask_empty", False)
                ),
                "truthful_window_interval_anchor_invalid": bool(
                    truth_summary.get("truthful_window_interval_anchor_invalid", False)
                ),
                "final_snapshot_measurement_truthful": bool(
                    truth_summary.get("final_snapshot_measurement_truthful", False)
                ),
                "final_snapshot_measurement_truth_tier": truth_summary.get(
                    "final_snapshot_measurement_truth_tier"
                ),
                "teacher_episode_class": str(
                    rollout.get("teacher_episode_class", "rejected_teacher")
                ),
                "teacher_fingerprint": str(rollout.get("teacher_fingerprint", "")),
                "learning_support_teacher_class": str(
                    rollout.get("learning_support_teacher_class", "rejected_teacher")
                ),
                "learning_support_fingerprint": str(
                    rollout.get("learning_support_fingerprint", "")
                ),
                "effective_support_signature_v1": str(
                    rollout.get("effective_support_signature_v1", "")
                ),
                "measurement_truthful_for_learning_support": bool(
                    rollout.get("measurement_truthful_for_learning_support", False)
                ),
                "learning_support_window_len": int(
                    rollout.get("learning_support_window_len", 0) or 0
                ),
                "learning_support_prebridge_frame_count": int(
                    rollout.get("learning_support_prebridge_frame_count", 0) or 0
                ),
                "learning_support_contains_prebridge": bool(
                    rollout.get("learning_support_contains_prebridge", False)
                ),
                "learning_support_contains_bridge": bool(
                    rollout.get("learning_support_contains_bridge", False)
                ),
                "first_attach_eligible_step": rollout.get("first_attach_eligible_step"),
                "teacher_family_variant": str(
                    rollout.get("teacher_family_variant") or ""
                ),
                "orientation_support_teacher_class": str(
                    rollout.get(
                        "orientation_support_teacher_class",
                        "rejected_orientation_teacher",
                    )
                ),
                "orientation_support_fingerprint": str(
                    rollout.get("orientation_support_fingerprint", "")
                ),
                "measurement_truthful_for_orientation_support": bool(
                    rollout.get("measurement_truthful_for_orientation_support", False)
                ),
                "orientation_support_window_len": int(
                    rollout.get("orientation_support_window_len", 0) or 0
                ),
                "orientation_context_frame_count": int(
                    rollout.get("orientation_context_frame_count", 0) or 0
                ),
                "orientation_support_truthful_ratio": float(
                    rollout.get("orientation_support_truthful_ratio", 0.0) or 0.0
                ),
                "orientation_support_contains_distance_pass": bool(
                    rollout.get("orientation_support_contains_distance_pass", False)
                ),
                "orientation_support_contains_approach_pass": bool(
                    rollout.get("orientation_support_contains_approach_pass", False)
                ),
                "orientation_support_contains_orientation_correction": bool(
                    rollout.get(
                        "orientation_support_contains_orientation_correction", False
                    )
                ),
                "orientation_support_contains_attach_eligible": bool(
                    rollout.get("orientation_support_contains_attach_eligible", False)
                ),
                "orientation_support_anchor_valid": bool(
                    rollout.get("orientation_support_anchor_valid", False)
                ),
                "orientation_gate_crossed": bool(
                    rollout.get("orientation_gate_crossed", False)
                ),
                "orientation_alignment_delta": float(
                    rollout.get("orientation_alignment_delta", 0.0) or 0.0
                ),
            }
            records.append(record)
            if str(
                rollout.get("orientation_support_teacher_class", "rejected_orientation_teacher")
            ) in {
                "orientation_transition_teacher",
                "attach_eligible_transition_teacher",
                "orientation_context_teacher",
            }:
                orientation_start = rollout.get("orientation_support_window_start")
                orientation_end = rollout.get("orientation_support_window_end")
                if orientation_start is not None and orientation_end is not None:
                    orientation_rollout = _slice_rollout_to_training_window(
                        rollout, int(orientation_start), int(orientation_end)
                    )
                    orientation_out_path = (
                        orientation_support_dir
                        / f"seed_{int(seed):03d}_episode_{int(episode_index):02d}.npz"
                    )
                    save_robot_rollout(orientation_out_path, orientation_rollout)
                    orientation_support_saved_rollouts += 1
                    orientation_support_successful_seeds.add(int(seed))
                    orientation_support_saved_files.append(str(orientation_out_path))
            if str(
                rollout.get("learning_support_teacher_class", "rejected_teacher")
            ) in {"strict_teacher", "near_strict_teacher", "learning_support_teacher"}:
                support_start = rollout.get("learning_support_window_start")
                support_end = rollout.get("learning_support_window_end")
                if support_start is not None and support_end is not None:
                    support_rollout = _slice_rollout_to_training_window(
                        rollout, int(support_start), int(support_end)
                    )
                    support_out_path = (
                        learning_support_dir
                        / f"seed_{int(seed):03d}_episode_{int(episode_index):02d}.npz"
                    )
                    save_robot_rollout(support_out_path, support_rollout)
                    learning_support_saved_rollouts += 1
                    learning_support_successful_seeds.add(int(seed))
                    learning_support_saved_files.append(str(support_out_path))
            if str(rollout.get("teacher_episode_class", "rejected_teacher")) not in {
                "strict_teacher",
                "near_strict_teacher",
            }:
                continue
            start = truth_summary.get("truthful_window_start")
            end = truth_summary.get("truthful_window_end")
            if start is None or end is None:
                continue
            rollout = _slice_rollout_to_training_window(rollout, int(start), int(end))
            out_path = (
                source_dir
                / f"seed_{int(seed):03d}_episode_{int(episode_index):02d}.npz"
            )
            save_robot_rollout(out_path, rollout)
            saved_rollouts += 1
            successful_seeds.add(int(seed))
            saved_files.append(str(out_path))

    report = _materialization_summary(
        gate_name="g6_canonical_rollout_materialization",
        source_dir=source_dir,
        expected=expected,
        train_seeds=train_seeds,
        episodes_per_seed=episodes_per_seed,
        min_train_episodes=min_train_episodes,
        teacher_controller_mode=teacher_controller_mode,
        teacher_controller_max_steps=teacher_controller_max_steps,
        teacher_pull_open_fraction=teacher_pull_open_fraction,
        stale_source_mismatch=stale_source_mismatch,
        attempted_rollouts=attempted_rollouts,
        saved_rollouts=saved_rollouts,
        successful_seeds=successful_seeds,
        saved_files=saved_files,
        records=records,
        dataset_selection_mode="claim_canonical",
    )
    learning_support_report = _materialization_summary(
        gate_name="g6_learning_support_rollout_materialization",
        source_dir=learning_support_dir,
        expected=expected,
        train_seeds=train_seeds,
        episodes_per_seed=episodes_per_seed,
        min_train_episodes=min_train_episodes,
        teacher_controller_mode=teacher_controller_mode,
        teacher_controller_max_steps=teacher_controller_max_steps,
        teacher_pull_open_fraction=teacher_pull_open_fraction,
        stale_source_mismatch=stale_source_mismatch,
        attempted_rollouts=attempted_rollouts,
        saved_rollouts=learning_support_saved_rollouts,
        successful_seeds=learning_support_successful_seeds,
        saved_files=learning_support_saved_files,
        records=records,
        dataset_selection_mode="diagnostic_learning_support",
    )
    orientation_support_report = _materialization_summary(
        gate_name="g6_orientation_support_rollout_materialization",
        source_dir=orientation_support_dir,
        expected=expected,
        train_seeds=train_seeds,
        episodes_per_seed=episodes_per_seed,
        min_train_episodes=min_train_episodes,
        teacher_controller_mode=teacher_controller_mode,
        teacher_controller_max_steps=teacher_controller_max_steps,
        teacher_pull_open_fraction=teacher_pull_open_fraction,
        stale_source_mismatch=stale_source_mismatch,
        attempted_rollouts=attempted_rollouts,
        saved_rollouts=orientation_support_saved_rollouts,
        successful_seeds=orientation_support_successful_seeds,
        saved_files=orientation_support_saved_files,
        records=records,
        dataset_selection_mode="diagnostic_orientation_support",
    )
    orientation_support_report = _enrich_orientation_materialization_report(
        orientation_support_report, records, train_seeds
    )
    if not report["passed"]:
        report["error"] = "insufficient_canonical_bridge_data"
    if not learning_support_report["passed"]:
        learning_support_report["error"] = "insufficient_learning_support_data"
    if not orientation_support_report["passed"]:
        orientation_support_report["error"] = "insufficient_orientation_support_data"
    MATERIALIZATION_ARTIFACT.write_text(json.dumps(report, indent=2))
    LEARNING_SUPPORT_MATERIALIZATION_ARTIFACT.write_text(
        json.dumps(learning_support_report, indent=2)
    )
    ORIENTATION_SUPPORT_MATERIALIZATION_ARTIFACT.write_text(
        json.dumps(orientation_support_report, indent=2)
    )
    return report

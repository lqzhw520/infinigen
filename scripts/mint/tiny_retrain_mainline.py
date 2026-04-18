#!/usr/bin/env python3
"""Shared helpers for the V1cT2 tiny-retrain mainline."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
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
MIN_TRAIN_EPISODES = 48
DEFAULT_EPISODES_PER_SEED = 12
MIN_SUCCESSFUL_SEEDS = 6
TRUTHFUL_WINDOW_MIN_FRAMES = 16
LEARNING_SUPPORT_WINDOW_MIN_FRAMES = 24
LEARNING_SUPPORT_PREBRIDGE_FRAMES = 24
LEARNING_SUPPORT_MIN_PREBRIDGE_FRAMES = 8
TERMINAL_COMMIT = "5aaf117b66902219ac997082763fb4e2ea8891b3"
STRICT_UTILITY_VERSION = STRICT_SUCCESS_VERSION
TRUTH_UTILITY_VERSION = "trace_window_v2"
TEACHER_FINGERPRINT_VERSION = "v8_3_fingerprint_v1"
LEARNING_SUPPORT_TRUTH_VERSION = "learning_support_window_v1"
LEARNING_SUPPORT_FINGERPRINT_VERSION = "v10_learning_support_fingerprint_v1"
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
    rollout["canonical_training_truth"] = truth_summary
    rollout["learning_support_truth"] = learning_support_summary
    rollout["measurement_truthful_for_training"] = bool(
        truth_summary.get("measurement_truthful_for_training", False)
    )
    rollout["measurement_truthful_for_learning_support"] = bool(
        learning_support_summary.get("measurement_truthful_for_learning_support", False)
    )
    rollout["teacher_truth_adjudication"] = truth_summary.get(
        "teacher_truth_adjudication"
    )
    rollout["learning_support_truth_adjudication"] = learning_support_summary.get(
        "learning_support_truth_adjudication"
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
    rollout["teacher_episode_class"] = _teacher_episode_class(rollout)
    rollout["learning_support_teacher_class"] = _learning_support_teacher_class(rollout)
    rollout["teacher_fingerprint"] = _teacher_fingerprint(rollout)
    rollout["learning_support_fingerprint_version"] = LEARNING_SUPPORT_FINGERPRINT_VERSION
    rollout["learning_support_fingerprint"] = _learning_support_fingerprint(rollout)

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
    learning_support_fingerprints = sorted(
        {
            rec.get("learning_support_fingerprint")
            for rec in records
            if rec.get("learning_support_teacher_class")
            in {"strict_teacher", "near_strict_teacher", "learning_support_teacher"}
            and rec.get("learning_support_fingerprint")
        }
    )
    accepted_family_count_by_seed: dict[str, int] = {}
    strict_family_count_by_seed: dict[str, int] = {}
    near_strict_family_count_by_seed: dict[str, int] = {}
    learning_support_family_count_by_seed: dict[str, int] = {}
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
        support_seed = {
            rec.get("learning_support_fingerprint")
            for rec in seed_records
            if rec.get("learning_support_teacher_class")
            in {"strict_teacher", "near_strict_teacher", "learning_support_teacher"}
            and rec.get("learning_support_fingerprint")
        }
        accepted_family_count_by_seed[str(seed)] = len(accepted_seed)
        strict_family_count_by_seed[str(seed)] = len(strict_seed)
        near_strict_family_count_by_seed[str(seed)] = len(near_seed)
        learning_support_family_count_by_seed[str(seed)] = len(support_seed)
        if support_seed:
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
        "learning_support_fingerprint_version": LEARNING_SUPPORT_FINGERPRINT_VERSION,
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
        "learning_support_unique_teacher_family_count": int(
            len(learning_support_fingerprints)
        ),
        "accepted_family_count_by_seed": accepted_family_count_by_seed,
        "strict_family_count_by_seed": strict_family_count_by_seed,
        "near_strict_family_count_by_seed": near_strict_family_count_by_seed,
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
        "learning_support_fingerprint_version": LEARNING_SUPPORT_FINGERPRINT_VERSION,
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
    if not bool(expected.get("tiny_retrain_permitted", True)):
        report["passed"] = False
        report["error"] = "RCA7 did not permit tiny retrain"
        learning_support_report.update({"passed": False, "error": report["error"]})
        MATERIALIZATION_ARTIFACT.write_text(json.dumps(report, indent=2))
        LEARNING_SUPPORT_MATERIALIZATION_ARTIFACT.write_text(
            json.dumps(learning_support_report, indent=2)
        )
        return report
    if not source_canonical_train_cell:
        report["passed"] = False
        report["error"] = "Missing source_canonical_train_cell from RCA5/RCA7"
        learning_support_report.update({"passed": False, "error": report["error"]})
        MATERIALIZATION_ARTIFACT.write_text(json.dumps(report, indent=2))
        LEARNING_SUPPORT_MATERIALIZATION_ARTIFACT.write_text(
            json.dumps(learning_support_report, indent=2)
        )
        return report
    if force_rebuild and source_dir.exists():
        shutil.rmtree(source_dir)
    if force_rebuild and learning_support_dir.exists():
        shutil.rmtree(learning_support_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    learning_support_dir.mkdir(parents=True, exist_ok=True)

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
    successful_seeds: set[int] = set()
    learning_support_successful_seeds: set[int] = set()
    saved_files: list[str] = []
    learning_support_saved_files: list[str] = []
    records: list[dict[str, Any]] = []
    for seed in train_seeds:
        for episode_index in range(episodes_per_seed):
            attempted_rollouts += 1
            rollout = build_robot_rollout(
                seed=int(seed),
                grasp_pose_world=np.eye(4, dtype=np.float32),
                episode_index=int(episode_index),
                max_steps=teacher_controller_max_steps,
                image_size=256,
                contract=contract,
                rotation_source=rotation_source,
                claim_policy=spec.claim_policy,
                interventions=interventions,
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
            }
            records.append(record)
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
    if not report["passed"]:
        report["error"] = "insufficient_canonical_bridge_data"
    if not learning_support_report["passed"]:
        learning_support_report["error"] = "insufficient_learning_support_data"
    MATERIALIZATION_ARTIFACT.write_text(json.dumps(report, indent=2))
    LEARNING_SUPPORT_MATERIALIZATION_ARTIFACT.write_text(
        json.dumps(learning_support_report, indent=2)
    )
    return report

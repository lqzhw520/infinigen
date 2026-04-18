#!/usr/bin/env python3
"""G1 forensic audit for learning-support family collapse."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _hash_json_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _bucket_floor_int(value: Any, bucket: int) -> int | str:
    if value is None:
        return "none"
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return "none"
    return int((numeric // bucket) * bucket)


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


def _close_onset_step(actions: np.ndarray, start: int, end: int) -> int | None:
    if actions.ndim != 2 or actions.shape[0] <= start or actions.shape[1] < 7:
        return None
    stop = min(int(end), int(actions.shape[0]))
    if stop <= start:
        return None
    close_mask = actions[start:stop, 6] < 0.0
    if not bool(np.any(close_mask)):
        return None
    return int(start + np.flatnonzero(close_mask)[0])


def _load_meta(meta_path: Path) -> dict[str, Any]:
    if not meta_path.exists():
        raise FileNotFoundError(f"missing meta json: {meta_path}")
    return json.loads(meta_path.read_text())


def _effective_signature_from_rollout(npz_path: Path, meta: dict[str, Any]) -> tuple[str, dict[str, int]]:
    with np.load(npz_path, allow_pickle=True) as data:
        phase_labels = [str(x) for x in data["phase_labels"]] if "phase_labels" in data else []
        states = (
            np.asarray(data["states"], dtype=np.float32)
            if "states" in data
            else np.asarray([], dtype=np.float32).reshape(0, 0)
        )
        actions = (
            np.asarray(data["actions"], dtype=np.float32)
            if "actions" in data
            else np.asarray([], dtype=np.float32).reshape(0, 0)
        )
        handle_distance = (
            np.asarray(data["handle_distance_trace"], dtype=np.float32).reshape(-1)
            if "handle_distance_trace" in data
            else np.asarray([], dtype=np.float32)
        )
        orientation_alignment = (
            np.asarray(data["orientation_alignment_trace"], dtype=np.float32).reshape(-1)
            if "orientation_alignment_trace" in data
            else np.asarray([], dtype=np.float32)
        )
        orientation_error = (
            np.asarray(data["orientation_error_trace"], dtype=np.float32).reshape(-1)
            if "orientation_error_trace" in data
            else np.asarray([], dtype=np.float32)
        )

    start = int(meta.get("learning_support_window_start", 0) or 0)
    end = int(meta.get("learning_support_window_end", 0) or 0)
    prebridge_len = int(meta.get("learning_support_prebridge_frame_count", 0) or 0)
    prebridge_end = max(start, start + prebridge_len)
    first_attach_eligible_step = meta.get("first_attach_eligible_step")
    first_attach_step = meta.get("attach_step")
    if first_attach_step is None:
        first_attach_step = (meta.get("strict_metrics") or {}).get("first_attach_step")
    close_onset_step = _close_onset_step(actions, start, end)

    phase_prebridge = phase_labels[start:prebridge_end]
    if states.ndim == 2 and states.shape[1] >= 3:
        eef_prebridge = states[start:prebridge_end, :3]
    else:
        eef_prebridge = np.asarray([], dtype=np.float32).reshape(0, 3)
    if actions.ndim == 2:
        actions_prebridge = actions[start:prebridge_end]
    else:
        actions_prebridge = np.asarray([], dtype=np.float32).reshape(0, 7)
    distance_prebridge = handle_distance[start:prebridge_end]
    if orientation_alignment.size >= prebridge_end and prebridge_end > start:
        orientation_values = orientation_alignment[start:prebridge_end]
    elif orientation_error.size >= prebridge_end and prebridge_end > start:
        orientation_values = orientation_error[start:prebridge_end]
    else:
        orientation_values = np.asarray([], dtype=np.float32)

    payload = {
        "seed": int(meta.get("seed", -1) or -1),
        "first_attach_eligible_step_bucket": _bucket_floor_int(
            first_attach_eligible_step, 8
        ),
        "first_attach_step_bucket": _bucket_floor_int(first_attach_step, 8),
        "close_onset_step_bucket": _bucket_floor_int(close_onset_step, 8),
        "prebridge_len_bucket": _bucket_floor_int(prebridge_len, 8),
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
    missing = Counter()
    if not phase_prebridge:
        missing["phase_labels_missing"] += 1
    if not eef_prebridge.size:
        missing["eef_path_missing"] += 1
    if not actions_prebridge.size:
        missing["actions_missing"] += 1
    if not distance_prebridge.size:
        missing["distance_curve_missing"] += 1
    if not orientation_values.size:
        missing["orientation_curve_missing"] += 1
    return json.dumps(payload, sort_keys=True), dict(missing)


def run(rollout_dir: Path, output_path: Path) -> dict[str, Any]:
    npz_paths = sorted(rollout_dir.glob("*.npz"))
    recorded_families: set[str] = set()
    effective_families: set[str] = set()
    recorded_by_seed: dict[str, set[str]] = defaultdict(set)
    effective_by_seed: dict[str, set[str]] = defaultdict(set)
    trace_missingness: Counter[str] = Counter()
    analyzed = 0
    parse_errors: list[str] = []

    for npz_path in npz_paths:
        meta_path = npz_path.with_suffix(".json")
        try:
            meta = _load_meta(meta_path)
            seed = str(int(meta.get("seed", -1) or -1))
            recorded = str(meta.get("learning_support_fingerprint", "") or "")
            if recorded:
                recorded_families.add(recorded)
                recorded_by_seed[seed].add(recorded)
            effective, missing = _effective_signature_from_rollout(npz_path, meta)
            effective_families.add(effective)
            effective_by_seed[seed].add(effective)
            trace_missingness.update(missing)
            analyzed += 1
        except Exception as exc:  # noqa: BLE001
            parse_errors.append(f"{npz_path.name}:{type(exc).__name__}:{exc}")

    recorded_family_count_by_seed = {
        seed: len(values) for seed, values in sorted(recorded_by_seed.items())
    }
    effective_family_count_by_seed = {
        seed: len(values) for seed, values in sorted(effective_by_seed.items())
    }
    seed2_effective = int(effective_family_count_by_seed.get("2", 0) or 0)
    seed4_effective = int(effective_family_count_by_seed.get("4", 0) or 0)
    effective_unique = len(effective_families)
    recorded_unique = len(recorded_families)
    fingerprint_underexpression_flag = bool(
        effective_unique >= 12
        and seed2_effective >= 2
        and seed4_effective >= 2
        and (
            recorded_unique < effective_unique
            or int(recorded_family_count_by_seed.get("2", 0) or 0) < seed2_effective
            or int(recorded_family_count_by_seed.get("4", 0) or 0) < seed4_effective
        )
    )
    true_generator_collapse_flag = bool(
        not fingerprint_underexpression_flag
        and (effective_unique < 12 or seed2_effective < 2 or seed4_effective < 2)
    )
    cannot_decide_trace_missing = bool(parse_errors)
    if cannot_decide_trace_missing:
        recommended = "cannot_decide_trace_missing"
        adjudication = "trace_logging_insufficient"
    elif fingerprint_underexpression_flag:
        recommended = "fix_fingerprint_only"
        adjudication = "fingerprint_underexpression"
    else:
        recommended = "run_bounded_teacher_family_grid"
        adjudication = "true_generator_collapse"
    report = {
        "gate": "G1_family_collapse_forensic_audit",
        "audit_version": "v11_learning_support_family_audit_v1",
        "rollout_dir": str(rollout_dir),
        "analyzed_rollout_count": analyzed,
        "expected_rollout_count": len(npz_paths),
        "parse_errors": parse_errors,
        "recorded_learning_support_unique_family_count": recorded_unique,
        "effective_support_unique_family_count": effective_unique,
        "recorded_family_count_by_seed": recorded_family_count_by_seed,
        "effective_family_count_by_seed": effective_family_count_by_seed,
        "seed2_effective_support_families": seed2_effective,
        "seed4_effective_support_families": seed4_effective,
        "fingerprint_underexpression_flag": fingerprint_underexpression_flag,
        "true_generator_collapse_flag": true_generator_collapse_flag,
        "trace_missingness_summary": dict(trace_missingness),
        "recommended_next_action": recommended,
        "adjudication": adjudication,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    report = run(args.rollout_dir, args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Shared rollout-contract diagnostics for the MINT robot-trajectory loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_meta(npz_path: Path) -> dict[str, Any]:
    meta_path = npz_path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def _decode_phases(values: np.ndarray | None) -> list[str]:
    if values is None:
        return []
    decoded: list[str] = []
    for value in values.tolist():
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8", errors="ignore"))
        else:
            decoded.append(str(value))
    return decoded


def analyze_rollout(npz_path: Path) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    meta = _load_meta(npz_path)

    actions = data["actions"].astype(np.float32)
    states = data["states"].astype(np.float32)
    phase_labels = _decode_phases(
        data["phase_labels"] if "phase_labels" in data.files else None
    )
    attached_trace = (
        data["attached_trace"].astype(bool) if "attached_trace" in data.files else None
    )
    handle_distance_trace = (
        data["handle_distance_trace"].astype(np.float32)
        if "handle_distance_trace" in data.files
        else None
    )
    drawer_trace = (
        data["absolute_drawer_fraction"].astype(np.float32)
        if "absolute_drawer_fraction" in data.files
        else data["next_drawer_fraction"].astype(np.float32)
        if "next_drawer_fraction" in data.files
        else None
    )

    unique_phases = sorted(set(phase_labels))
    gripper = actions[:, 6]
    abs_actions = np.abs(actions)
    clipped_mask = abs_actions >= 0.999
    clipped_fraction = float(np.mean(clipped_mask))
    gripper_transition_count = int(np.count_nonzero(np.abs(np.diff(gripper)) > 0.25))
    handle_distance_min = (
        float(np.min(handle_distance_trace))
        if handle_distance_trace is not None
        else None
    )
    attach_step = None
    if attached_trace is not None and np.any(attached_trace):
        attach_step = int(np.argmax(attached_trace))

    success = bool(meta.get("success"))
    ever_attached = bool(meta.get("ever_attached")) or bool(
        attached_trace is not None and np.any(attached_trace)
    )
    max_drawer_fraction = float(
        meta.get(
            "max_drawer_fraction",
            np.max(drawer_trace)
            if drawer_trace is not None and drawer_trace.size
            else 0.0,
        )
    )
    gripper_range = float(np.max(gripper) - np.min(gripper))
    phase_counts = {phase: int(phase_labels.count(phase)) for phase in unique_phases}

    return {
        "rollout_path": str(npz_path),
        "seed": meta.get("seed"),
        "episode_index": meta.get("episode_index"),
        "source_branch_id": meta.get("branch_id") or meta.get("source_branch_id"),
        "contract_mode": meta.get("contract_mode"),
        "task": meta.get("task"),
        "success": success,
        "ever_attached": ever_attached,
        "attach_step": attach_step,
        "max_drawer_fraction": max_drawer_fraction,
        "handle_distance_min": handle_distance_min,
        "actions_shape": list(actions.shape),
        "states_shape": list(states.shape),
        "phase_values": unique_phases,
        "phase_step_counts": phase_counts,
        "gripper_unique_values": sorted(set(np.round(gripper, 3).tolist())),
        "gripper_range": gripper_range,
        "gripper_transition_count": gripper_transition_count,
        "action_clipped_fraction": clipped_fraction,
        "action_norm_p95": float(
            np.percentile(np.linalg.norm(actions[:, :6], axis=1), 95)
        ),
        "action_min": float(np.min(actions)),
        "action_max": float(np.max(actions)),
        "has_nan": bool(np.isnan(actions).any() or np.isnan(states).any()),
        "metadata_present": {
            "branch_id": bool(meta.get("branch_id") or meta.get("source_branch_id")),
            "seed": "seed" in meta,
            "task": bool(meta.get("task")),
            "success": "success" in meta,
        },
    }


def preflight_rollout(npz_path: Path) -> dict[str, Any]:
    analysis = analyze_rollout(npz_path)
    phases = set(analysis["phase_values"])
    reasons: list[str] = []

    if analysis["actions_shape"][1:] != [7]:
        reasons.append("action_dim_mismatch")
    if analysis["states_shape"][1:] != [8]:
        reasons.append("state_dim_mismatch")
    if analysis["has_nan"]:
        reasons.append("nan_in_rollout")
    if not analysis["metadata_present"]["branch_id"]:
        reasons.append("missing_branch_id")
    if not analysis["metadata_present"]["seed"]:
        reasons.append("missing_seed")
    if not analysis["success"] or not analysis["ever_attached"]:
        reasons.append("rollout_not_successful")
    if analysis["max_drawer_fraction"] < 0.9:
        reasons.append("drawer_open_fraction_too_low")
    if analysis["gripper_range"] < 0.5 or analysis["gripper_transition_count"] < 1:
        reasons.append("gripper_phase_not_expressed")
    if analysis["action_clipped_fraction"] > 0.5:
        reasons.append("action_clipping_too_high")
    if not {"grasp", "pull"}.issubset(phases):
        reasons.append("missing_grasp_or_pull_phase")
    if len(phases) < 4:
        reasons.append("phase_coverage_too_small")

    analysis["preflight_passed"] = not reasons
    analysis["preflight_reasons"] = reasons
    return analysis


def directory_preflight(rollout_paths: list[Path]) -> dict[str, Any]:
    analyses = [preflight_rollout(path) for path in rollout_paths]
    passed = [item for item in analyses if item["preflight_passed"]]
    best = sorted(
        analyses,
        key=lambda item: (
            bool(item["preflight_passed"]),
            bool(item["success"]),
            bool(item["ever_attached"]),
            float(item["max_drawer_fraction"]),
            -float(item["action_clipped_fraction"]),
            float(item["handle_distance_min"])
            if item["handle_distance_min"] is not None
            else -999.0,
        ),
        reverse=True,
    )
    return {
        "passed": bool(passed),
        "rollout_count": len(analyses),
        "passed_count": len(passed),
        "best_rollout": None if not best else best[0]["rollout_path"],
        "best_rollout_analysis": None if not best else best[0],
        "rollouts": analyses,
    }


def select_best_rollout(
    rollout_paths: list[Path],
) -> tuple[Path | None, dict[str, Any]]:
    summary = directory_preflight(rollout_paths)
    best_path = Path(summary["best_rollout"]) if summary.get("best_rollout") else None
    return best_path, summary

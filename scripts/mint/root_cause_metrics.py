#!/usr/bin/env python3
"""Pure metrics for the v2 root-cause controller."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def image_metrics(rgb: np.ndarray) -> dict[str, Any]:
    arr = rgb.astype(np.float32) / 255.0
    flat = arr.reshape(-1, 3)
    gray = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    hist = np.bincount(gray.flatten(), minlength=256).astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    nonzero = hist[hist > 0]
    return {
        "mean_rgb": flat.mean(axis=0).round(6).tolist(),
        "std_rgb": flat.std(axis=0).round(6).tolist(),
        "brightness_mean": float(gray.mean() / 255.0),
        "edge_density_mean": float((edges > 0).mean()),
        "entropy_mean": float(-(nonzero * np.log2(nonzero)).sum()),
    }


def summarize_visual(samples: list[np.ndarray]) -> dict[str, Any]:
    metrics = [image_metrics(sample) for sample in samples]
    if not metrics:
        return {
            "sample_count": 0,
            "mean_rgb": [0.0, 0.0, 0.0],
            "std_rgb": [0.0, 0.0, 0.0],
            "brightness_mean": 0.0,
            "edge_density_mean": 0.0,
            "entropy_mean": 0.0,
        }
    return {
        "sample_count": len(samples),
        "mean_rgb": np.mean(np.asarray([m["mean_rgb"] for m in metrics], dtype=np.float32), axis=0).round(6).tolist(),
        "std_rgb": np.mean(np.asarray([m["std_rgb"] for m in metrics], dtype=np.float32), axis=0).round(6).tolist(),
        "brightness_mean": float(np.mean([m["brightness_mean"] for m in metrics])),
        "edge_density_mean": float(np.mean([m["edge_density_mean"] for m in metrics])),
        "entropy_mean": float(np.mean([m["entropy_mean"] for m in metrics])),
    }


def visual_gap(control: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    return {
        "brightness_gap": round(abs(control["brightness_mean"] - target["brightness_mean"]), 6),
        "std_rgb_abs_gap": [round(abs(a - b), 6) for a, b in zip(control["std_rgb"], target["std_rgb"])],
        "edge_density_gap": round(abs(control["edge_density_mean"] - target["edge_density_mean"]), 6),
        "entropy_gap": round(abs(control["entropy_mean"] - target["entropy_mean"]), 6),
    }


def mean_std_rgb_gap(gap_payload: dict[str, Any]) -> float:
    return float(np.mean(gap_payload.get("std_rgb_abs_gap", [0.0, 0.0, 0.0])))


def weighted_visual_gap(gap_payload: dict[str, Any]) -> float:
    return float(
        0.40 * float(gap_payload.get("edge_density_gap", 0.0))
        + 0.40 * float(gap_payload.get("entropy_gap", 0.0))
        + 0.20 * mean_std_rgb_gap(gap_payload)
    )


def visual_alignment_gain(baseline_gap_value: float, lane_gap_value: float) -> float:
    denom = max(float(baseline_gap_value), 1e-6)
    return float(1.0 - float(lane_gap_value) / denom)


def rollout_ceiling_lift(baseline: dict[str, Any], lane: dict[str, Any]) -> dict[str, Any]:
    delta_unique_success_rate = float(lane.get("unique_success_rate", 0.0) - baseline.get("unique_success_rate", 0.0))
    delta_attach_rate = float(lane.get("attach_rate", 0.0) - baseline.get("attach_rate", 0.0))
    delta_mean_max_drawer_fraction = float(lane.get("mean_max_drawer_fraction", 0.0) - baseline.get("mean_max_drawer_fraction", 0.0))
    baseline_steps = max(float(baseline.get("avg_episode_length", 1.0)), 1e-6)
    delta_avg_episode_length_ratio = float(lane.get("avg_episode_length", 0.0) / baseline_steps - 1.0)
    delta_orientation_causal_sensitivity = max(
        0.0,
        float(lane.get("orientation_causal_sensitivity", 0.0) - baseline.get("orientation_causal_sensitivity", 0.0)),
    )
    score = (
        0.30 * delta_unique_success_rate
        + 0.25 * delta_attach_rate
        + 0.20 * delta_mean_max_drawer_fraction
        + 0.15 * delta_avg_episode_length_ratio
        + 0.10 * delta_orientation_causal_sensitivity
    )
    return {
        "score": float(score),
        "delta_unique_success_rate": delta_unique_success_rate,
        "delta_attach_rate": delta_attach_rate,
        "delta_mean_max_drawer_fraction": delta_mean_max_drawer_fraction,
        "delta_avg_episode_length_ratio": delta_avg_episode_length_ratio,
        "delta_orientation_causal_sensitivity": delta_orientation_causal_sensitivity,
    }


def state_alignment_gain(baseline_score: float, lane_score: float) -> float:
    denom = max(float(baseline_score), 1e-6)
    return float(1.0 - float(lane_score) / denom)


def sensor_contract_gain(
    *,
    wrist_relativeness_gain: float,
    no_overlay_gain: float,
    visual_alignment_gain_value: float,
    framing_gain: float,
) -> dict[str, Any]:
    score = (
        0.40 * float(wrist_relativeness_gain)
        + 0.25 * float(no_overlay_gain)
        + 0.20 * float(visual_alignment_gain_value)
        + 0.15 * float(framing_gain)
    )
    return {
        "score": float(score),
        "wrist_relativeness_gain": float(wrist_relativeness_gain),
        "no_overlay_gain": float(no_overlay_gain),
        "visual_alignment_gain": float(visual_alignment_gain_value),
        "framing_gain": float(framing_gain),
    }


def contract_validity_score(*passes: bool) -> float:
    if not passes:
        return 0.0
    return float(np.mean(np.asarray([1.0 if x else 0.0 for x in passes], dtype=np.float32)))


def observation_contract_pass(lane_report: dict[str, Any]) -> bool:
    return bool(
        lane_report.get("secondary_camera_wrist_like", False)
        and not lane_report.get("marker_overlay_enabled", True)
        and not lane_report.get("diagnostic_only", True)
        and lane_report.get("provenance_complete", False)
    )

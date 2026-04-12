#!/usr/bin/env python3
"""Pure metrics for the integrated root-case controller."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


CONTRAST_THRESHOLD = 0.08
HANDLE_ENTROPY_REFERENCE = 4.0
PERCEPTUAL_READABILITY_THRESHOLD = 0.55
HANDLE_BOUNDARY_CONTRAST_THRESHOLD = 0.08
HANDLE_EDGE_DENSITY_THRESHOLD = 0.03
HANDLE_CROP_ENTROPY_THRESHOLD = 1.00
HANDLE_AREA_RATIO_THRESHOLD = 0.01
SECONDARY_FRAMING_THRESHOLD = 0.70
LOCAL_AFFORDANCE_READABILITY_THRESHOLD = 0.60
GLOBAL_VISUAL_ALIGNMENT_THRESHOLD = 0.10


def _to_uint8(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr, 0, 255).astype(np.uint8)


def _gray_uint8(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(_to_uint8(rgb), cv2.COLOR_RGB2GRAY)


def _entropy_from_gray(gray: np.ndarray) -> float:
    hist = np.bincount(gray.flatten(), minlength=256).astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    nonzero = hist[hist > 0]
    return float(-(nonzero * np.log2(nonzero)).sum()) if len(nonzero) else 0.0


def image_metrics(rgb: np.ndarray) -> dict[str, Any]:
    arr = _to_uint8(rgb).astype(np.float32) / 255.0
    flat = arr.reshape(-1, 3)
    gray = _gray_uint8(rgb)
    edges = cv2.Canny(gray, 80, 160)
    return {
        "mean_rgb": flat.mean(axis=0).round(6).tolist(),
        "std_rgb": flat.std(axis=0).round(6).tolist(),
        "brightness_mean": float(gray.mean() / 255.0),
        "edge_density_mean": float((edges > 0).mean()),
        "entropy_mean": _entropy_from_gray(gray),
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


def _bbox_area(bbox: list[int] | tuple[int, int, int, int] | None) -> int:
    if not bbox or len(bbox) != 4:
        return 0
    x0, y0, x1, y1 = [int(v) for v in bbox]
    return max(0, x1 - x0) * max(0, y1 - y0)


def _trace_items(metadata_trace: list[dict[str, Any]] | dict[str, Any] | None) -> list[dict[str, Any]]:
    if metadata_trace is None:
        return []
    if isinstance(metadata_trace, dict):
        return [metadata_trace]
    return list(metadata_trace)


def _mean_trace_value(metadata_trace: list[dict[str, Any]] | dict[str, Any], key: str) -> float:
    values = []
    for item in _trace_items(metadata_trace):
        probe = item.get("handle_probe_metadata") if isinstance(item, dict) and "handle_probe_metadata" in item else item
        if not isinstance(probe, dict):
            continue
        value = probe.get(key)
        if value is not None:
            values.append(float(value))
    return float(np.mean(values)) if values else 0.0


def handle_bbox_from_rollout_metadata(rollout_metadata: dict[str, Any], key: str = "primary") -> list[int] | None:
    probe = rollout_metadata.get("handle_probe_metadata") or rollout_metadata or {}
    bbox = probe.get(f"handle_bbox_{key}")
    if bbox is not None:
        return [int(v) for v in bbox]
    camera_meta = rollout_metadata.get("camera_metadata") or {}
    nested = (camera_meta.get(key) or {}).get("handle_bbox")
    if nested is None:
        return None
    return [int(v) for v in nested]


def handle_visibility_fraction(metadata_trace: list[dict[str, Any]] | dict[str, Any], key: str = "primary") -> float:
    return _mean_trace_value(metadata_trace, f"handle_visibility_fraction_{key}")


def handle_local_contrast(metadata_trace: list[dict[str, Any]] | dict[str, Any], key: str = "primary") -> float:
    value = _mean_trace_value(metadata_trace, f"handle_local_contrast_{key}")
    if value > 0.0:
        return value
    return _mean_trace_value(metadata_trace, f"handle_boundary_contrast_{key}")


def handle_crop_entropy(metadata_trace: list[dict[str, Any]] | dict[str, Any], key: str = "primary") -> float:
    return _mean_trace_value(metadata_trace, f"handle_crop_entropy_{key}")


def handle_readability_score(
    *,
    visibility_fraction: float,
    local_contrast: float,
    crop_entropy: float,
    framing_score: float,
) -> float:
    visibility_term = float(np.clip(visibility_fraction, 0.0, 1.0))
    contrast_term = float(np.clip(local_contrast / max(CONTRAST_THRESHOLD, 1e-6), 0.0, 1.0))
    entropy_term = float(np.clip(crop_entropy / max(HANDLE_ENTROPY_REFERENCE, 1e-6), 0.0, 1.0))
    framing_term = float(np.clip(framing_score, 0.0, 1.0))
    return float(0.30 * visibility_term + 0.30 * contrast_term + 0.20 * entropy_term + 0.20 * framing_term)


def perceptual_readability_score_v2(
    *,
    visual_alignment_gain_value: float,
    handle_readability_score_value: float,
    encoder_readability_pass: bool | None = None,
) -> float:
    score = 0.45 * float(max(visual_alignment_gain_value, 0.0)) + 0.55 * float(np.clip(handle_readability_score_value, 0.0, 1.0))
    if encoder_readability_pass is False:
        score *= 0.9
    elif encoder_readability_pass is True:
        score = min(1.0, score + 0.05)
    return float(score)


def handle_area_ratio(mask_bbox: list[int] | tuple[int, int, int, int] | None, image_shape: tuple[int, ...]) -> float:
    if mask_bbox is None or len(image_shape) < 2:
        return 0.0
    h = max(int(image_shape[0]), 1)
    w = max(int(image_shape[1]), 1)
    return float(_bbox_area(mask_bbox) / float(h * w))


def handle_boundary_contrast(mask_ring: np.ndarray, image: np.ndarray) -> float:
    gray = _gray_uint8(image).astype(np.float32)
    ring = np.asarray(mask_ring, dtype=bool)
    if gray.shape[:2] != ring.shape or not ring.any():
        return 0.0
    dilated = cv2.dilate(ring.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    interior = ring
    exterior = np.logical_and(dilated, np.logical_not(ring))
    if not exterior.any():
        exterior = np.logical_not(ring)
    inside_mean = float(gray[interior].mean()) if interior.any() else 0.0
    outside_mean = float(gray[exterior].mean()) if exterior.any() else 0.0
    return float(abs(inside_mean - outside_mean) / 255.0)


def handle_edge_density(mask_bbox_crop: np.ndarray) -> float:
    crop = _to_uint8(mask_bbox_crop)
    if crop.size == 0:
        return 0.0
    gray = _gray_uint8(crop)
    edges = cv2.Canny(gray, 80, 160)
    return float((edges > 0).mean())


def handle_crop_entropy_from_crop(mask_bbox_crop: np.ndarray) -> float:
    crop = _to_uint8(mask_bbox_crop)
    if crop.size == 0:
        return 0.0
    return _entropy_from_gray(_gray_uint8(crop))


def local_affordance_readability_score_v3(
    *,
    handle_area_ratio_value: float,
    handle_boundary_contrast_value: float,
    handle_edge_density_value: float,
    handle_crop_entropy_value: float,
    secondary_framing_score: float,
) -> float:
    area_term = float(np.clip(handle_area_ratio_value / max(HANDLE_AREA_RATIO_THRESHOLD, 1e-6), 0.0, 1.0))
    boundary_term = float(np.clip(handle_boundary_contrast_value / max(HANDLE_BOUNDARY_CONTRAST_THRESHOLD, 1e-6), 0.0, 1.0))
    edge_term = float(np.clip(handle_edge_density_value / max(HANDLE_EDGE_DENSITY_THRESHOLD, 1e-6), 0.0, 1.0))
    entropy_term = float(np.clip(handle_crop_entropy_value / max(HANDLE_CROP_ENTROPY_THRESHOLD, 1e-6), 0.0, 1.0))
    framing_term = float(np.clip(secondary_framing_score / max(SECONDARY_FRAMING_THRESHOLD, 1e-6), 0.0, 1.0))
    return float(0.20 * area_term + 0.25 * boundary_term + 0.20 * edge_term + 0.20 * entropy_term + 0.15 * framing_term)


def local_affordance_gate_breakdown(
    *,
    handle_bbox: list[int] | tuple[int, int, int, int] | None,
    image_shape: tuple[int, ...],
    handle_boundary_contrast_value: float,
    handle_edge_density_value: float,
    handle_crop_entropy_value: float,
    secondary_framing_score: float,
) -> dict[str, Any]:
    area_ratio = handle_area_ratio(handle_bbox, image_shape)
    score = local_affordance_readability_score_v3(
        handle_area_ratio_value=area_ratio,
        handle_boundary_contrast_value=handle_boundary_contrast_value,
        handle_edge_density_value=handle_edge_density_value,
        handle_crop_entropy_value=handle_crop_entropy_value,
        secondary_framing_score=secondary_framing_score,
    )
    return {
        "handle_area_ratio": float(area_ratio),
        "handle_boundary_contrast": float(handle_boundary_contrast_value),
        "handle_edge_density": float(handle_edge_density_value),
        "handle_crop_entropy": float(handle_crop_entropy_value),
        "secondary_framing_score": float(secondary_framing_score),
        "local_affordance_readability_score_v3": float(score),
    }


def visual_gate_breakdown(
    *,
    baseline_weighted_visual_gap: float,
    lane_weighted_visual_gap: float,
    visual_alignment_gain_value: float,
    framing_score: float,
    visibility_fraction: float,
    local_contrast: float,
    crop_entropy: float,
    camera_relativeness_residual: float | None,
    encoder_readability_pass: bool | None = None,
) -> dict[str, Any]:
    handle_score = handle_readability_score(
        visibility_fraction=visibility_fraction,
        local_contrast=local_contrast,
        crop_entropy=crop_entropy,
        framing_score=framing_score,
    )
    readability_score = perceptual_readability_score_v2(
        visual_alignment_gain_value=visual_alignment_gain_value,
        handle_readability_score_value=handle_score,
        encoder_readability_pass=encoder_readability_pass,
    )
    return {
        "baseline_weighted_visual_gap": float(baseline_weighted_visual_gap),
        "lane_weighted_visual_gap": float(lane_weighted_visual_gap),
        "visual_alignment_gain": float(visual_alignment_gain_value),
        "framing_score": float(framing_score),
        "handle_visibility_fraction": float(visibility_fraction),
        "handle_local_contrast": float(local_contrast),
        "handle_crop_entropy": float(crop_entropy),
        "handle_readability_score": float(handle_score),
        "perceptual_readability_score_v2": float(readability_score),
        "camera_relativeness_residual": None if camera_relativeness_residual is None else float(camera_relativeness_residual),
        "encoder_readability_pass": encoder_readability_pass,
    }


def rollout_ceiling_lift(baseline: dict[str, Any], lane: dict[str, Any]) -> dict[str, Any]:
    delta_unique_success_rate = float(lane.get("unique_success_rate", 0.0) - baseline.get("unique_success_rate", 0.0))
    delta_attach_rate = float(lane.get("attach_rate", 0.0) - baseline.get("attach_rate", 0.0))
    delta_mean_max_drawer_fraction = float(lane.get("mean_max_drawer_fraction", 0.0) - baseline.get("mean_max_drawer_fraction", 0.0))
    baseline_steps = max(float(baseline.get("avg_episode_length", 1.0)), 1e-6)
    delta_avg_episode_length_ratio = float(lane.get("avg_episode_length", 0.0) / baseline_steps - 1.0)
    delta_orientation_causal_sensitivity = max(0.0, float(lane.get("orientation_causal_sensitivity", 0.0) - baseline.get("orientation_causal_sensitivity", 0.0)))
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


def sensor_contract_gain(*, wrist_relativeness_gain: float, no_overlay_gain: float, visual_alignment_gain_value: float, framing_gain: float) -> dict[str, Any]:
    score = 0.40 * float(wrist_relativeness_gain) + 0.25 * float(no_overlay_gain) + 0.20 * float(visual_alignment_gain_value) + 0.15 * float(framing_gain)
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


def contract_validity_score_v2(*, g0: bool, g1: bool, g2: bool, g3: bool, g4a: bool, g4b_local: bool, g4c_global: bool) -> float:
    return contract_validity_score(g0, g1, g2, g3, g4a, g4b_local, g4c_global)


def observation_contract_pass(lane_report: dict[str, Any]) -> bool:
    return bool(
        lane_report.get("secondary_camera_wrist_like", False)
        and not lane_report.get("marker_overlay_enabled", True)
        and not lane_report.get("diagnostic_only", True)
        and lane_report.get("provenance_complete", False)
    )

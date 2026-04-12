#!/usr/bin/env python3
"""Actual-handle-crop perception probe for the integrated RCA campaign."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _to_uint8(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr, 0, 255).astype(np.uint8)


def _entropy(gray: np.ndarray) -> float:
    hist = np.bincount(gray.flatten(), minlength=256).astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    nonzero = hist[hist > 0]
    return float(-(nonzero * np.log2(nonzero)).sum()) if len(nonzero) else 0.0


def _crop(image: np.ndarray, bbox: list[int] | None) -> np.ndarray | None:
    if bbox is None or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = [int(v) for v in bbox]
    if x1 <= x0 or y1 <= y0:
        return None
    crop = _to_uint8(image)[y0:y1, x0:x1]
    return crop if crop.size else None


def _negative_crop(image: np.ndarray, bbox: list[int] | None) -> np.ndarray | None:
    if bbox is None or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = [int(v) for v in bbox]
    h, w = image.shape[:2]
    box_w = max(x1 - x0, 1)
    box_h = max(y1 - y0, 1)
    shift = max(box_w, box_h)
    nx0 = min(max(0, x0 + shift), max(0, w - box_w))
    ny0 = y0
    nx1 = min(w, nx0 + box_w)
    ny1 = min(h, ny0 + box_h)
    crop = _to_uint8(image)[ny0:ny1, nx0:nx1]
    return crop if crop.size else None


def _feature(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(_to_uint8(crop), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    feat = np.array([
        gray.mean() / 255.0,
        gray.std() / 255.0,
        float((edges > 0).mean()),
        _entropy(gray) / 8.0,
    ], dtype=np.float32)
    return feat


def run_perception_probe(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    positives = []
    negatives = []
    for rollout in rollouts:
        images = rollout.get("images2")
        if images is None or len(images) == 0:
            images = rollout.get("images")
        if images is None:
            images = []
        trace = rollout.get("handle_probe_metadata_trace") or []
        for image, item in zip(images, trace):
            probe = item if isinstance(item, dict) else {}
            bbox = probe.get("handle_bbox_secondary") or probe.get("handle_bbox_primary")
            pos = _crop(np.asarray(image, dtype=np.uint8), bbox)
            neg = _negative_crop(np.asarray(image, dtype=np.uint8), bbox)
            if pos is not None:
                positives.append(_feature(pos))
            if neg is not None:
                negatives.append(_feature(neg))
    if not positives or not negatives:
        return {
            "perception_probe_mode": "crop_feature_proxy_v1",
            "perception_probe_available": False,
            "handle_patch_separability": 0.0,
            "encoder_readability_pass": None,
            "positive_count": len(positives),
            "negative_count": len(negatives),
        }
    pos = np.stack(positives, axis=0)
    neg = np.stack(negatives, axis=0)
    pos_center = pos.mean(axis=0)
    neg_center = neg.mean(axis=0)
    between = float(np.linalg.norm(pos_center - neg_center))
    within = float(np.mean(np.linalg.norm(pos - pos_center[None, :], axis=1))) + float(np.mean(np.linalg.norm(neg - neg_center[None, :], axis=1)))
    separability = between / max(within, 1e-6)
    return {
        "perception_probe_mode": "crop_feature_proxy_v1",
        "perception_probe_available": True,
        "handle_patch_separability": float(separability),
        "encoder_readability_pass": bool(separability >= 1.10),
        "positive_count": int(len(positives)),
        "negative_count": int(len(negatives)),
    }

#!/usr/bin/env python3
"""Perception-side probes for the integrated RCA campaign."""

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


def _proxy_feature(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(_to_uint8(crop), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    feat = np.array([
        gray.mean() / 255.0,
        gray.std() / 255.0,
        float((edges > 0).mean()),
        _entropy(gray) / 8.0,
    ], dtype=np.float32)
    return feat


def _separability(positives: list[np.ndarray], negatives: list[np.ndarray]) -> float:
    if not positives or not negatives:
        return 0.0
    pos = np.stack(positives, axis=0)
    neg = np.stack(negatives, axis=0)
    pos_center = pos.mean(axis=0)
    neg_center = neg.mean(axis=0)
    between = float(np.linalg.norm(pos_center - neg_center))
    within = float(np.mean(np.linalg.norm(pos - pos_center[None, :], axis=1))) + float(np.mean(np.linalg.norm(neg - neg_center[None, :], axis=1)))
    return float(between / max(within, 1e-6))


def _extract_feature(feature_extractor: Any, crop: np.ndarray) -> np.ndarray:
    if callable(feature_extractor):
        feat = feature_extractor(crop)
    elif hasattr(feature_extractor, 'encode'):
        feat = feature_extractor.encode(crop)
    else:
        raise TypeError('Unsupported feature_extractor interface')
    arr = np.asarray(feat, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError('Empty feature vector from policy encoder probe')
    return arr


def _iter_truthful_crops(rollouts: list[dict[str, Any]], require_truthful_mask: bool) -> tuple[list[np.ndarray], list[np.ndarray], int, int, list[str]]:
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    warning_flags: list[str] = []
    positive_count = 0
    negative_count = 0
    for rollout in rollouts:
        images = rollout.get('images2')
        if images is None or len(images) == 0:
            images = rollout.get('images')
        if images is None:
            images = []
        trace = rollout.get('handle_probe_metadata_trace') or []
        for image, item in zip(images, trace):
            probe = item if isinstance(item, dict) else {}
            truthful = bool(probe.get('measurement_truthful'))
            if require_truthful_mask and not truthful:
                continue
            bbox = probe.get('handle_bbox_secondary') or probe.get('handle_bbox_primary')
            pos = _crop(np.asarray(image, dtype=np.uint8), bbox)
            neg = _negative_crop(np.asarray(image, dtype=np.uint8), bbox)
            if pos is not None:
                positives.append(pos)
                positive_count += 1
            if neg is not None:
                negatives.append(neg)
                negative_count += 1
            if not truthful:
                warning_flags.append('non_truthful_mask_in_probe_trace')
    return positives, negatives, positive_count, negative_count, sorted(set(warning_flags))


def _run_proxy_probe(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    positives, negatives, positive_count, negative_count, warning_flags = _iter_truthful_crops(rollouts, require_truthful_mask=False)
    pos_feats = [_proxy_feature(crop) for crop in positives]
    neg_feats = [_proxy_feature(crop) for crop in negatives]
    if not pos_feats or not neg_feats:
        return {
            'perception_probe_mode': 'proxy_only',
            'perception_probe_available': False,
            'handle_patch_separability': 0.0,
            'encoder_readability_pass': None,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'warning_flags': warning_flags,
        }
    separability = _separability(pos_feats, neg_feats)
    return {
        'perception_probe_mode': 'proxy_only',
        'perception_probe_available': True,
        'handle_patch_separability': float(separability),
        'encoder_readability_pass': None,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'warning_flags': warning_flags,
    }


def run_policy_encoder_probe(rollouts: list[dict[str, Any]], feature_extractor: Any, require_truthful_mask: bool = True) -> dict[str, Any]:
    positives, negatives, positive_count, negative_count, warning_flags = _iter_truthful_crops(rollouts, require_truthful_mask=require_truthful_mask)
    if feature_extractor is None:
        return {
            'perception_probe_mode': 'policy_encoder_v1',
            'perception_probe_available': False,
            'handle_patch_separability': 0.0,
            'encoder_readability_pass': None,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'warning_flags': sorted(set(warning_flags + ['feature_extractor_unavailable'])),
        }
    pos_feats = []
    neg_feats = []
    try:
        for crop in positives:
            pos_feats.append(_extract_feature(feature_extractor, crop))
        for crop in negatives:
            neg_feats.append(_extract_feature(feature_extractor, crop))
    except Exception as exc:  # noqa: BLE001
        return {
            'perception_probe_mode': 'policy_encoder_v1',
            'perception_probe_available': False,
            'handle_patch_separability': 0.0,
            'encoder_readability_pass': None,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'warning_flags': sorted(set(warning_flags + [f'feature_extractor_error:{exc}'])),
        }
    if not pos_feats or not neg_feats:
        return {
            'perception_probe_mode': 'policy_encoder_v1',
            'perception_probe_available': False,
            'handle_patch_separability': 0.0,
            'encoder_readability_pass': None,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'warning_flags': sorted(set(warning_flags + ['insufficient_truthful_crops'])),
        }
    separability = _separability(pos_feats, neg_feats)
    return {
        'perception_probe_mode': 'policy_encoder_v1',
        'perception_probe_available': True,
        'handle_patch_separability': float(separability),
        'encoder_readability_pass': bool(separability >= 1.10),
        'positive_count': int(positive_count),
        'negative_count': int(negative_count),
        'warning_flags': warning_flags,
    }


def run_perception_probe(rollouts: list[dict[str, Any]], feature_extractor: Any = None) -> dict[str, Any]:
    if feature_extractor is not None:
        encoder_probe = run_policy_encoder_probe(rollouts, feature_extractor, require_truthful_mask=True)
        if encoder_probe.get('perception_probe_available'):
            return encoder_probe
        proxy = _run_proxy_probe(rollouts)
        encoder_probe['proxy_probe'] = proxy
        return encoder_probe
    proxy = _run_proxy_probe(rollouts)
    return {
        'perception_probe_mode': 'policy_encoder_v1',
        'perception_probe_available': False,
        'handle_patch_separability': 0.0,
        'encoder_readability_pass': None,
        'positive_count': int(proxy.get('positive_count', 0)),
        'negative_count': int(proxy.get('negative_count', 0)),
        'warning_flags': sorted(set(list(proxy.get('warning_flags', [])) + ['policy_encoder_unavailable'])),
        'proxy_probe': proxy,
    }

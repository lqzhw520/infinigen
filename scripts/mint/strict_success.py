#!/usr/bin/env python3
"""Shared strict success semantics for teacher and policy evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

STRICT_SUCCESS_VERSION = "v3_attached_open_contract"

STRICT_SUCCESS_CONFIG = {
    "drawer_open_threshold": 0.90,
    "min_attach_persistence_steps": 3,
    "max_pre_attach_drawer_motion": 0.30,
    "min_post_attach_drawer_delta": 0.35,
    "major_open_threshold": 0.35,
}


def _max_consecutive_true(values: np.ndarray) -> int:
    run = 0
    best = 0
    for value in values.astype(bool).tolist():
        if value:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return int(best)


def evaluate_strict_success(
    drawer_trace: np.ndarray,
    attached_trace: np.ndarray,
    *,
    config: dict[str, float] | None = None,
) -> dict[str, Any]:
    cfg = {**STRICT_SUCCESS_CONFIG, **(config or {})}
    drawer = np.asarray(drawer_trace, dtype=np.float32).reshape(-1)
    attached = np.asarray(attached_trace, dtype=bool).reshape(-1)
    if len(drawer) != len(attached):
        raise ValueError(
            f"drawer/attached trace length mismatch: {len(drawer)} vs {len(attached)}"
        )

    ever_attached = bool(attached.any())
    first_attach_step = int(np.argmax(attached)) if ever_attached else None
    attach_persistence = _max_consecutive_true(attached)
    pre_attach_drawer_motion = (
        float(np.max(drawer[:first_attach_step]))
        if first_attach_step not in (None, 0)
        else 0.0
    )
    attach_drawer_fraction = (
        float(drawer[first_attach_step]) if first_attach_step is not None else 0.0
    )
    post_attach_max = (
        float(np.max(drawer[first_attach_step:]))
        if first_attach_step is not None
        else float(np.max(drawer))
    )
    post_attach_drawer_delta = (
        float(post_attach_max - attach_drawer_fraction)
        if first_attach_step is not None
        else 0.0
    )
    drawer_open = bool(float(np.max(drawer)) >= float(cfg["drawer_open_threshold"]))
    unattached_drawer_delta_max = (
        pre_attach_drawer_motion
        if ever_attached
        else (float(np.max(drawer)) if len(drawer) else 0.0)
    )
    major_open_indices = np.flatnonzero(drawer >= float(cfg["major_open_threshold"]))
    first_major_open_step = (
        int(major_open_indices[0]) if len(major_open_indices) else None
    )
    attach_before_major_open = bool(
        first_major_open_step is None
        or (
            first_attach_step is not None and first_attach_step <= first_major_open_step
        )
    )
    strict_success = bool(
        drawer_open
        and ever_attached
        and attach_persistence >= int(cfg["min_attach_persistence_steps"])
        and pre_attach_drawer_motion <= float(cfg["max_pre_attach_drawer_motion"])
        and post_attach_drawer_delta >= float(cfg["min_post_attach_drawer_delta"])
        and attach_before_major_open
    )
    return {
        "strict_success_version": STRICT_SUCCESS_VERSION,
        "strict_success_config": cfg,
        "strict_success": strict_success,
        "drawer_open": drawer_open,
        "ever_attached": ever_attached,
        "first_attach_step": first_attach_step,
        "attach_persistence": attach_persistence,
        "pre_attach_drawer_motion": pre_attach_drawer_motion,
        "unattached_drawer_delta_max": unattached_drawer_delta_max,
        "attach_drawer_fraction": attach_drawer_fraction,
        "post_attach_drawer_delta": post_attach_drawer_delta,
        "first_major_open_step": first_major_open_step,
        "attach_before_major_open": attach_before_major_open,
        "max_drawer_fraction": float(np.max(drawer)) if len(drawer) else 0.0,
    }

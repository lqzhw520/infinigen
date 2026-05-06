#!/usr/bin/env python3
"""Handle-affordance classification and strategy dispatch for V11 drawer tasks.

This module is intentionally lightweight: it records the handle affordance branch
that a candidate belongs to and prevents non-knob handles from silently using the
round-knob controller.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROUND_KNOB = "ROUND_KNOB"
LINE_BAR_HANDLE = "LINE_BAR_HANDLE"
ARC_OR_C_HANDLE = "ARC_OR_C_HANDLE"
RECESSED_GROOVE_OR_LIP = "RECESSED_GROOVE_OR_LIP"
FLAT_FRONT_NO_GRASPABLE_HANDLE = "FLAT_FRONT_NO_GRASPABLE_HANDLE"
UNKNOWN_OR_UNSUPPORTED = "UNKNOWN_OR_UNSUPPORTED"

IMPLEMENTED_STRATEGY = "RoundKnobBilateralPinchLatchStrategy"
UNSUPPORTED_CLOSEOUT = "HANDLE_AFFORDANCE_BRANCH_NOT_IMPLEMENTED"


@dataclass(frozen=True)
class HandleClassification:
    candidate_id: str
    handle_class: str
    confidence: float
    selected_strategy: str
    branch_implemented: bool
    unsupported_closeout: str | None
    reasons: tuple[str, ...]
    geometry: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "handle_class": self.handle_class,
            "confidence": self.confidence,
            "selected_strategy": self.selected_strategy,
            "branch_implemented": self.branch_implemented,
            "unsupported_closeout": self.unsupported_closeout,
            "reasons": list(self.reasons),
            "geometry": self.geometry,
        }


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _params(candidate: dict[str, Any]) -> dict[str, Any]:
    params = candidate.get("model_builder_parameters") or {}
    return params if isinstance(params, dict) else {}


def handle_geometry_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    params = _params(candidate)
    radius = _as_float(params.get("handle_radius"), 0.024)
    diameter = 2.0 * radius if radius > 0.0 else 0.0
    half_length = _as_float(params.get("handle_half_length"), radius)
    stub_length = _as_float(
        params.get("stub_length", params.get("connector_length")),
        0.18 * diameter if diameter > 0.0 else 0.0,
    )
    connector_ratio = stub_length / diameter if diameter > 1e-9 else None
    axis_lengths = sorted(
        [2.0 * radius, 2.0 * max(radius, half_length), 2.0 * radius], reverse=True
    )
    elongation_ratio = axis_lengths[0] / max(axis_lengths[-1], 1e-9)
    compactness = axis_lengths[-1] / max(axis_lengths[0], 1e-9)
    center = [
        _as_float(params.get("handle_x"), -0.18),
        _as_float(params.get("handle_y"), -0.07),
        _as_float(params.get("handle_z"), 0.402),
    ]
    return {
        "handle_kind": str(
            params.get("handle_kind", params.get("handle_type", "sphere"))
        ).lower(),
        "handle_radius_m": radius,
        "knob_diameter_m": diameter,
        "handle_half_length_m": half_length,
        "stub_length_m": stub_length,
        "stub_length_to_knob_diameter": connector_ratio,
        "pca_axis_lengths_proxy_m": axis_lengths,
        "elongation_ratio_proxy": elongation_ratio,
        "compactness_proxy": compactness,
        "handle_center_m": center,
        "short_stub_invariant_passed": bool(
            connector_ratio is not None and connector_ratio <= 0.35
        ),
        "preferred_stub_ratio_range_passed": bool(
            connector_ratio is not None and 0.10 <= connector_ratio <= 0.25
        ),
        "topology_repair_provenance": params.get("visual_topology_repair")
        or params.get("fast_guarded_codesign_operator")
        or params.get("repair_cycle_note"),
    }


def classify_handle_affordance(candidate: dict[str, Any]) -> HandleClassification:
    cid = str(candidate.get("candidate_id", "unknown_candidate"))
    geom = handle_geometry_summary(candidate)
    kind = str(geom["handle_kind"]).lower()
    elongation = _as_float(geom.get("elongation_ratio_proxy"), 1.0)
    radius = _as_float(geom.get("handle_radius_m"), 0.0)
    diameter = _as_float(geom.get("knob_diameter_m"), 0.0)
    connector_ratio = geom.get("stub_length_to_knob_diameter")
    short_stub = bool(geom.get("short_stub_invariant_passed"))
    reasons: list[str] = []

    if any(token in kind for token in ["recess", "groove", "lip", "milled", "slot"]):
        return HandleClassification(
            cid,
            RECESSED_GROOVE_OR_LIP,
            0.75,
            "RecessedLipInsertionStrategy",
            False,
            UNSUPPORTED_CLOSEOUT,
            ("negative_space_or_lip_keyword",),
            geom,
        )
    if any(token in kind for token in ["arc", "c_handle", "loop", "curved"]):
        return HandleClassification(
            cid,
            ARC_OR_C_HANDLE,
            0.70,
            "ArcHandleHookStrategy",
            False,
            UNSUPPORTED_CLOSEOUT,
            ("arc_or_loop_keyword",),
            geom,
        )
    if any(token in kind for token in ["bar", "line", "rod", "cylinder"]):
        return HandleClassification(
            cid,
            LINE_BAR_HANDLE,
            0.80,
            "LineBarHookOrPinchStrategy",
            False,
            UNSUPPORTED_CLOSEOUT,
            ("elongated_bar_keyword",),
            geom,
        )
    if any(token in kind for token in ["none", "flat", "front"]):
        return HandleClassification(
            cid,
            FLAT_FRONT_NO_GRASPABLE_HANDLE,
            0.65,
            "UnsupportedHandleStrategy",
            False,
            UNSUPPORTED_CLOSEOUT,
            ("no_graspable_handle_keyword",),
            geom,
        )

    if any(token in kind for token in ["sphere", "spherical", "knob", "round"]):
        if radius > 0.0 and diameter > 0.0 and short_stub:
            reasons.extend(
                [
                    "semantic_round_knob_handle_kind",
                    "short_stub_connector_present",
                    "round_knob_semantics_override_bar_elongation_proxy",
                ]
            )
            return HandleClassification(
                cid,
                ROUND_KNOB,
                0.95,
                IMPLEMENTED_STRATEGY,
                True,
                None,
                tuple(reasons),
                geom,
            )
        return HandleClassification(
            cid,
            UNKNOWN_OR_UNSUPPORTED,
            0.45,
            "UnsupportedHandleStrategy",
            False,
            UNSUPPORTED_CLOSEOUT,
            ("round_knob_semantic_but_short_stub_or_radius_missing",),
            geom,
        )

    if radius > 0.0 and diameter > 0.0 and elongation <= 1.6 and short_stub:
        reasons.extend(
            [
                "compact_near_spherical_geometry",
                "short_stub_connector_present",
                "not_elongated_bar",
            ]
        )
        return HandleClassification(
            cid,
            ROUND_KNOB,
            0.90,
            IMPLEMENTED_STRATEGY,
            True,
            None,
            tuple(reasons),
            geom,
        )
    if elongation > 2.3:
        return HandleClassification(
            cid,
            LINE_BAR_HANDLE,
            0.55,
            "LineBarHookOrPinchStrategy",
            False,
            UNSUPPORTED_CLOSEOUT,
            ("elongation_ratio_proxy_gt_2p3",),
            geom,
        )
    if radius > 0.0 and connector_ratio is not None and connector_ratio > 0.35:
        return HandleClassification(
            cid,
            UNKNOWN_OR_UNSUPPORTED,
            0.45,
            "UnsupportedHandleStrategy",
            False,
            UNSUPPORTED_CLOSEOUT,
            ("compact_handle_but_connector_too_long_for_round_knob_claim",),
            geom,
        )
    return HandleClassification(
        cid,
        UNKNOWN_OR_UNSUPPORTED,
        0.30,
        "UnsupportedHandleStrategy",
        False,
        UNSUPPORTED_CLOSEOUT,
        ("insufficient_semantic_or_geometry_evidence",),
        geom,
    )


def strategy_registry() -> dict[str, Any]:
    return {
        "generated_at_note": "static registry definition; run artifact records timestamp",
        "strategies": [
            {
                "handle_class": ROUND_KNOB,
                "strategy": IMPLEMENTED_STRATEGY,
                "implemented": True,
                "claim_boundary": "Certified only for short-stub spherical/round knob branches in this phase.",
                "contact_mode": [
                    "ALIGN",
                    "CAPTURE",
                    "BILATERAL_LATCH",
                    "LOCKED_PULL",
                    "RESEAT",
                ],
            },
            {
                "handle_class": LINE_BAR_HANDLE,
                "strategy": "LineBarHookOrPinchStrategy",
                "implemented": False,
                "unsupported_closeout": UNSUPPORTED_CLOSEOUT,
                "required_future_mode": "bar-axis hook-or-pinch pull",
            },
            {
                "handle_class": ARC_OR_C_HANDLE,
                "strategy": "ArcHandleHookStrategy",
                "implemented": False,
                "unsupported_closeout": UNSUPPORTED_CLOSEOUT,
                "required_future_mode": "inner-gap hook insertion and pull",
            },
            {
                "handle_class": RECESSED_GROOVE_OR_LIP,
                "strategy": "RecessedLipInsertionStrategy",
                "implemented": False,
                "unsupported_closeout": UNSUPPORTED_CLOSEOUT,
                "required_future_mode": "finger insertion or lip hook pull",
            },
            {
                "handle_class": FLAT_FRONT_NO_GRASPABLE_HANDLE,
                "strategy": "UnsupportedHandleStrategy",
                "implemented": False,
                "unsupported_closeout": UNSUPPORTED_CLOSEOUT,
            },
            {
                "handle_class": UNKNOWN_OR_UNSUPPORTED,
                "strategy": "UnsupportedHandleStrategy",
                "implemented": False,
                "unsupported_closeout": UNSUPPORTED_CLOSEOUT,
            },
        ],
        "hard_dispatch_rule": "Non-ROUND_KNOB handles must not use the round-knob strategy unless separately implemented and certified.",
    }


def classify_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [classify_handle_affordance(candidate).as_dict() for candidate in candidates]


def write_registry_artifacts(
    run_dir: Path, candidates: list[dict[str, Any]], generated_at_utc: str
) -> dict[str, Any]:
    classifications = classify_candidates(candidates)
    non_knob = [c for c in classifications if c.get("handle_class") != ROUND_KNOB]
    payload = {
        "generated_at_utc": generated_at_utc,
        "candidate_count": len(candidates),
        "round_knob_candidate_count": sum(
            1 for c in classifications if c.get("handle_class") == ROUND_KNOB
        ),
        "non_knob_candidate_count": len(non_knob),
        "non_knob_candidate_ids": [c.get("candidate_id") for c in non_knob],
        "hard_rule_round_knob_only_for_this_phase": True,
        "classifications": classifications,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "handle_affordance_classification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    registry = {"generated_at_utc": generated_at_utc, **strategy_registry()}
    (run_dir / "handle_strategy_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n"
    )
    md_lines = [
        "# Handle Strategy Registry",
        "",
        f"- candidate_count: `{len(candidates)}`",
        "- round_knob_candidate_count: `{}`".format(
            payload["round_knob_candidate_count"]
        ),
        f"- non_knob_candidate_count: `{len(non_knob)}`",
        "- current_phase_claim: `ROUND_KNOB` only",
        "- non_knob_dispatch: `HANDLE_AFFORDANCE_BRANCH_NOT_IMPLEMENTED`",
        "",
        "| handle_class | strategy | implemented |",
        "|---|---:|---:|",
    ]
    for item in registry["strategies"]:
        md_lines.append(
            "| `{}` | `{}` | `{}` |".format(
                item["handle_class"], item["strategy"], bool(item["implemented"])
            )
        )
    (run_dir / "handle_strategy_registry.md").write_text(
        "\n".join(md_lines).rstrip() + "\n"
    )
    return {"classification": payload, "registry": registry}


if __name__ == "__main__":
    print(json.dumps(strategy_registry(), indent=2, sort_keys=True))

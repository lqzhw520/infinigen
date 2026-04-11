#!/usr/bin/env python3
"""Hypothesis board management for the v2 root-cause controller."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from root_cause_contracts import HypothesisState


DEFAULT_HYPOTHESES = [
    HypothesisState(
        id="H0_same_problem_identity",
        description="Current environment is still close enough to the upstream benchmark-alignment claim.",
        prior=0.35,
        posterior=0.35,
        importance=1.0,
        next_best_test="E0",
    ),
    HypothesisState(
        id="H1_embodiment_causal_contract",
        description="Primary blocker is missing embodiment and causal action semantics, especially orientation-insensitive interaction.",
        prior=0.60,
        posterior=0.60,
        importance=1.0,
        next_best_test="E1",
    ),
    HypothesisState(
        id="H2_state_representability",
        description="Primary blocker is unresolved or impossible control-compatible state representability under the current embodiment.",
        prior=0.65,
        posterior=0.65,
        importance=0.95,
        next_best_test="E2",
    ),
    HypothesisState(
        id="H3_observation_visual_contract",
        description="Observation contract and visual semantics are a strong co-blocker.",
        prior=0.70,
        posterior=0.70,
        importance=0.95,
        next_best_test="E0",
    ),
    HypothesisState(
        id="H4_unique_data_scale_only",
        description="More unique successful data alone can solve the current failure without contract repair.",
        prior=0.25,
        posterior=0.25,
        importance=0.70,
        next_best_test="E5",
    ),
    HypothesisState(
        id="H5_optimization_only",
        description="More repeat count or more train steps on the same weak data can solve the current failure.",
        prior=0.15,
        posterior=0.15,
        importance=0.60,
        next_best_test="E6",
    ),
    HypothesisState(
        id="H6_environment_invalid_for_claim",
        description="Current environment formulation is insufficient for the benchmark-alignment claim.",
        prior=0.55,
        posterior=0.55,
        importance=1.0,
        next_best_test="E4",
    ),
]


def default_board() -> dict[str, dict[str, Any]]:
    return {item.id: deepcopy(item.__dict__) for item in DEFAULT_HYPOTHESES}


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def update_hypothesis(
    state: dict[str, Any],
    *,
    support_score: float,
    alpha: float,
    evidence_ref: str,
) -> dict[str, Any]:
    updated = deepcopy(state)
    prior = float(updated.get("posterior", updated.get("prior", 0.5)))
    logit_post = max(-4.0, min(4.0, _logit(prior) + float(alpha) * float(support_score)))
    posterior = _sigmoid(logit_post)
    updated["prior"] = prior
    updated["posterior"] = posterior
    if support_score >= 0:
        updated.setdefault("supporting_experiments", []).append(evidence_ref)
    else:
        updated.setdefault("contradicting_experiments", []).append(evidence_ref)
    return updated


def apply_updates(board: dict[str, dict[str, Any]], updates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = deepcopy(board)
    for item in updates:
        hypothesis_id = item["hypothesis_id"]
        if hypothesis_id not in out:
            continue
        out[hypothesis_id] = update_hypothesis(
            out[hypothesis_id],
            support_score=float(item["support_score"]),
            alpha=float(item.get("alpha", 0.3)),
            evidence_ref=str(item["evidence_ref"]),
        )
    return out


def unresolvedness(state: dict[str, Any]) -> float:
    posterior = float(state.get("posterior", 0.5))
    return float(1.0 - abs(2.0 * posterior - 1.0))

#!/usr/bin/env python3
"""Typed contracts and claim-boundary helpers for the visual-fidelity root-cause controller."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from mint_common import CAMPAIGN_DIR, RUNTIME_DIR, now_iso
from p1_execution_common import CURRENT_TRUTH_PATH, NEXT_ACTIONS_PATH, SOVEREIGN_DIR

ClaimPolicy = Literal["canonical", "diagnostic", "new_claim_required"]
LaneStage = Literal["probe", "control", "train"]
ScientificTerminalState = Literal[
    "ALIGNED_CANONICAL_LANE_FOUND",
    "MINIMAL_REPAIR_INSUFFICIENT",
    "BENCHMARK_ALIGNMENT_UNSUPPORTED_UNDER_CURRENT_FORMULATION",
    "TS_CANONICAL_POSITIVE_ESTABLISHED",
    "TS_MAINLINE_REJECTED",
]
RouteNextBranch = Literal[
    "stay_current_branch",
    "environment_reformulation",
    "tiny_retrain_confirmation",
    "claim_scope_revision",
]

CLAIM_BOUNDARY_PATH = SOVEREIGN_DIR / "claim_boundary.yaml"
AUTOPILOT_DIR = CAMPAIGN_DIR / "autopilot"
CYCLE_STATE_PATH = AUTOPILOT_DIR / "cycle_state.json"
HYPOTHESIS_BOARD_PATH = AUTOPILOT_DIR / "hypothesis_board.json"
CONTROLLER_RUNTIME_DIR = RUNTIME_DIR / "controller_cycles"
EXPERIMENT_REGISTRY_PATH = SOVEREIGN_DIR / "experiment_registry.jsonl"
EVIDENCE_LEDGER_PATH = SOVEREIGN_DIR / "evidence_ledger.jsonl"
DEVIATION_LOG_PATH = AUTOPILOT_DIR / "deviation_log.jsonl"


@dataclass(frozen=True)
class DatasetConfig:
    unique_data_budget: int | None = None
    success_repeat: int = 1
    train_seed_pool: list[int] = field(default_factory=list)
    heldout_seed_pool: list[int] = field(default_factory=list)
    max_steps: int = 96


@dataclass(frozen=True)
class TrainConfig:
    train_steps: int = 0
    batch_size: int = 0
    top_k: int = 0
    tiny_retrain: bool = False
    full_retrain: bool = False


@dataclass(frozen=True)
class LaneSpec:
    lane_id: str
    lane_family: str
    stage: LaneStage
    env_contract_config: dict[str, Any]
    interventions: dict[str, Any] = field(default_factory=dict)
    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    claim_policy: ClaimPolicy = "diagnostic"
    diagnostic_only: bool = False
    strongest_negative_capped: bool = True
    resource_budget_snapshot: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    experiment_id: str = ""
    coverage: dict[str, float] = field(default_factory=dict)
    cost: float = 1.0

    @property
    def lane_hash(self) -> str:
        return stable_config_hash(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lane_hash"] = stable_config_hash(payload)
        return payload


@dataclass
class LaneResult:
    lane_id: str
    lane_family: str
    experiment_id: str
    status: Literal["passed", "failed", "skipped", "blocked"]
    claim_policy: ClaimPolicy
    diagnostic_only: bool
    strongest_negative_capped: bool
    metrics: dict[str, Any]
    artifact_paths: list[str] = field(default_factory=list)
    summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    scientific_terminal_state: ScientificTerminalState | None = None
    route_next_branch: RouteNextBranch | None = None


@dataclass
class GateDecision:
    gate_id: str
    passed: bool
    summary: str
    blocking: bool = True
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HypothesisState:
    id: str
    description: str
    prior: float
    posterior: float
    importance: float
    supporting_experiments: list[str] = field(default_factory=list)
    contradicting_experiments: list[str] = field(default_factory=list)
    next_best_test: str = ""
    status: str = "active"


@dataclass
class CycleSummary:
    cycle_id: str
    started_at: str
    ended_at: str | None
    selected_experiments: list[str]
    lane_ids: list[str]
    lane_family: str = "VR"
    scientific_terminal_state: ScientificTerminalState | None = None
    route_next_branch: RouteNextBranch | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ControllerRoute:
    scientific_terminal_state: ScientificTerminalState | None
    route_next_branch: RouteNextBranch
    why: str
    strongest_negative_capped: bool = True
    cap_active: bool = True
    h0_posterior: float | None = None
    h6_posterior: float | None = None


def stable_config_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def load_claim_boundary(path: Path = CLAIM_BOUNDARY_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def load_authority_snapshot() -> dict[str, Any]:
    current_truth = json.loads(CURRENT_TRUTH_PATH.read_text()) if CURRENT_TRUTH_PATH.exists() else {}
    next_actions = json.loads(NEXT_ACTIONS_PATH.read_text()) if NEXT_ACTIONS_PATH.exists() else {}
    current = current_truth.get("current", {})
    return {
        "phase": current.get("phase"),
        "verdict": current.get("verdict"),
        "next_action": current.get("next_action"),
        "open_gates": current.get("open_gates", []),
        "blockers": current.get("blockers", []),
        "next_actions": next_actions.get("actions", []),
        "read_at": now_iso(),
    }


def classify_claim_policy(boundary: dict[str, Any], lane: LaneSpec) -> tuple[ClaimPolicy, list[str]]:
    violations: list[str] = []
    env = lane.env_contract_config
    if env.get("enable_marker_overlay"):
        violations.append("marker_overlay_enabled")
    if env.get("calibration_mode") in {"diagnostic_texture", "legacy_bg_gain_bias"}:
        violations.append("diagnostic_texture_as_canonical")
    if lane.claim_policy == "canonical" and env.get("secondary_camera_mode") != "wrist_dynamic":
        violations.append("non_wrist_secondary_camera_for_benchmark_claim")
    if env.get("image_size") not in (None, 256):
        violations.append("changed_resolution")
    state_spec = env.get("state_spec") or {}
    if state_spec.get("fraud_padding") or env.get("state_mode") in {"control_like", "duplicated_padding"}:
        violations.append("duplicated_state_dims_for_shape_only")
    if lane.diagnostic_only:
        violations.append("diagnostic_only_lane")
    if lane.claim_policy != "canonical":
        return lane.claim_policy, violations
    if violations:
        return "new_claim_required", violations
    return "canonical", violations

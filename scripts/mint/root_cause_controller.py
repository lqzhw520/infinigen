#!/usr/bin/env python3
"""Reformulation-aware gate-first controller for MINT/Infinigen root-cause investigation."""

from __future__ import annotations

import os
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance

from drawer_robot_env_mujoco import DrawerEnvContractConfig, build_robot_rollout, drawer_manifest
from mint_common import (
    acquire_controller_lease,
    controller_identity,
    load_json,
    now_iso,
    release_controller_lease,
    refresh_controller_lease,
    write_json_atomic,
    write_text_atomic,
)
from mujoco_mainline_common import default_heldout_seeds
from p1_execution_common import ARTIFACT_DIR, OUTPUT_DIR
from root_cause_contracts import (
    AUTOPILOT_DIR,
    CLAIM_BOUNDARY_PATH,
    CONTROLLER_RUNTIME_DIR,
    CYCLE_STATE_PATH,
    HYPOTHESIS_BOARD_PATH,
    ClaimPolicy,
    DatasetConfig,
    GateDecision,
    LaneSpec,
    TrainConfig,
    classify_claim_policy,
    load_authority_snapshot,
    load_claim_boundary,
)
from root_cause_hypotheses import apply_updates, default_board, unresolvedness
from root_cause_metrics import (
    contract_validity_score,
    observation_contract_pass,
    rollout_ceiling_lift,
    sensor_contract_gain,
    state_alignment_gain,
    summarize_visual,
    visual_alignment_gain,
    visual_gap,
    weighted_visual_gap,
)
from root_cause_registry import (
    append_cycle_record,
    append_evidence_record,
    create_cycle_dir,
    ensure_registry_layout,
    write_cycle_bundle,
    write_final_run_outputs,
    write_hypothesis_board,
)


CONTROL_TRACE_ROOT = ARTIFACT_DIR / "p1k_control_traces"
P1J_ARTIFACT = ARTIFACT_DIR / "p1j_control_rerun_v2.json"
P1L_ARTIFACT = ARTIFACT_DIR / "p1l_eval_parity_matrix.json"

RF0_ARTIFACT = ARTIFACT_DIR / "p2rf0_terminal_negative_replication.json"
RF1_ARTIFACT = ARTIFACT_DIR / "p2rf1_reformulated_observation_contract_probe.json"
RF2_ARTIFACT = ARTIFACT_DIR / "p2rf2_reformulated_action_causality_probe.json"
RF3_ARTIFACT = ARTIFACT_DIR / "p2rf3_reformulated_state_semantics_probe.json"
RF4_ARTIFACT = ARTIFACT_DIR / "p2rf4_reformulated_visual_readability_probe.json"
RF5_ARTIFACT = ARTIFACT_DIR / "p2rf5_reformulated_bundle_ab.json"
RF6_ARTIFACT = ARTIFACT_DIR / "p2rf6_replicate_strongest_signal.json"

RF0_REPORT = OUTPUT_DIR / "p2rf0_terminal_negative_replication.md"
RF1_REPORT = OUTPUT_DIR / "p2rf1_reformulated_observation_contract_probe.md"
RF2_REPORT = OUTPUT_DIR / "p2rf2_reformulated_action_causality_probe.md"
RF3_REPORT = OUTPUT_DIR / "p2rf3_reformulated_state_semantics_probe.md"
RF4_REPORT = OUTPUT_DIR / "p2rf4_reformulated_visual_readability_probe.md"
RF5_REPORT = OUTPUT_DIR / "p2rf5_reformulated_bundle_ab.md"
RF6_REPORT = OUTPUT_DIR / "p2rf6_replicate_strongest_signal.md"

SCIENTIFIC_TERMINALS = {
    "aligned": "ALIGNED_CANONICAL_LANE_FOUND",
    "minimal_negative": "MINIMAL_REPAIR_INSUFFICIENT",
    "unsupported": "BENCHMARK_ALIGNMENT_UNSUPPORTED_UNDER_CURRENT_FORMULATION",
}

ROUTE_OPTIONS = {
    "reformulation": "environment_reformulation",
    "stay": "stay_current_branch",
    "tiny_retrain": "tiny_retrain_confirmation",
}


class RootCauseController:
    def __init__(
        self,
        *,
        max_experiments_per_cycle: int = 3,
        dry_run: bool = False,
        auto_promote_sovereign: bool = False,
        allow_full_retrain: bool = False,
        allow_new_claim: bool = False,
        run_mode: str = "autonomous_cycle_phase",
        max_rollouts_per_experiment: int = 3,
        max_disk_growth_mb: int = 4096,
        retry_backoff_seconds: int = 5,
        max_retries: int = 2,
        max_cycles_per_run: int = 4,
        max_tiny_retrain_budget: int = 1,
    ):
        ensure_registry_layout()
        self.max_experiments_per_cycle = int(max_experiments_per_cycle)
        self.dry_run = bool(dry_run)
        self.policy = {
            "AUTO_PROMOTE_SOVEREIGN": bool(auto_promote_sovereign),
            "ALLOW_FULL_RETRAIN": bool(allow_full_retrain),
            "ALLOW_NEW_CLAIM": bool(allow_new_claim),
            "RUN_MODE": run_mode,
        }
        self.resource_limits = {
            "max_rollouts_per_experiment": int(max_rollouts_per_experiment),
            "max_disk_growth_mb": int(max_disk_growth_mb),
            "retry_backoff_seconds": int(retry_backoff_seconds),
            "max_retries": int(max_retries),
            "max_cycles_per_run": int(max_cycles_per_run),
            "max_tiny_retrain_budget": int(max_tiny_retrain_budget),
        }
        self.claim_boundary = load_claim_boundary(CLAIM_BOUNDARY_PATH)
        self.authority = load_authority_snapshot()
        self.board = load_json(HYPOTHESIS_BOARD_PATH, default_board()) or default_board()
        self.completed_experiments = list(load_json(CYCLE_STATE_PATH, {}).get("completed_experiments", []))
        self.heldout_seeds = default_heldout_seeds()
        available = [int(seed) for seed in drawer_manifest().get("available_seeds", [])]
        self.train_seeds = [seed for seed in available if seed not in self.heldout_seeds][:6]
        self.control_images = self._load_control_images()
        self.control_state, self.control_action = self._load_control_arrays()
        self.disk_start_bytes = self._tree_size(AUTOPILOT_DIR) + self._tree_size(CONTROLLER_RUNTIME_DIR)

    def _tree_size(self, root: Path) -> int:
        if not root.exists():
            return 0
        total = 0
        for path in root.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        return total

    def _disk_growth_mb(self) -> float:
        current = self._tree_size(AUTOPILOT_DIR) + self._tree_size(CONTROLLER_RUNTIME_DIR)
        return float(current - self.disk_start_bytes) / (1024.0 * 1024.0)

    def _resource_snapshot(self) -> dict[str, Any]:
        return {
            "policy": dict(self.policy),
            "limits": dict(self.resource_limits),
            "disk_growth_mb": round(self._disk_growth_mb(), 3),
            "completed_experiments": list(self.completed_experiments),
        }

    def _baseline_integrity_details(self) -> dict[str, Any]:
        p1j = load_json(P1J_ARTIFACT, {}) if P1J_ARTIFACT.exists() else {}
        p1l = load_json(P1L_ARTIFACT, {}) if P1L_ARTIFACT.exists() else {}
        control_pass = bool(p1j.get("passed")) and float(((p1j.get("aggregate_metrics") or {}).get("pc_success") or 0.0)) >= 99.0
        parity_payload = p1l.get("decision") or {}
        parity_pass = bool(p1l.get("passed")) and not bool(parity_payload.get("parity_blocker", True))
        authority_readable = bool(self.authority.get("phase") and self.authority.get("next_action") is not None)
        return {
            "p1j_exists": P1J_ARTIFACT.exists(),
            "p1l_exists": P1L_ARTIFACT.exists(),
            "control_baseline_pass": control_pass,
            "control_pc_success": float(((p1j.get("aggregate_metrics") or {}).get("pc_success") or 0.0)) if p1j else None,
            "parity_non_blocker": parity_pass,
            "parity_decision": parity_payload,
            "authority_readable": authority_readable,
        }

    def _load_control_images(self, limit: int = 60) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for trace_path in sorted(CONTROL_TRACE_ROOT.glob("episode_*/trace.npz")):
            data = np.load(trace_path)
            for frame in data["image"]:
                img = np.asarray(frame, dtype=np.float32)
                if img.max() <= 1.0:
                    img = img * 255.0
                frames.append(np.clip(img, 0, 255).astype(np.uint8))
                if len(frames) >= limit:
                    return frames
        return frames

    def _load_control_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        states = []
        actions = []
        for trace_path in sorted(CONTROL_TRACE_ROOT.glob("episode_*/trace.npz")):
            data = np.load(trace_path)
            states.append(np.asarray(data["state_pre"], dtype=np.float32))
            actions.append(np.asarray(data["action_sent"], dtype=np.float32))
        if not states or not actions:
            return np.zeros((1, 8), dtype=np.float32), np.zeros((1, 7), dtype=np.float32)
        return np.concatenate(states, axis=0), np.concatenate(actions, axis=0)

    def _make_cycle_id(self) -> str:
        return f"cycle_{time.strftime('%Y%m%d_%H%M%S')}"

    def _legacy_contract(self) -> dict[str, Any]:
        return asdict(DrawerEnvContractConfig.legacy_defaults())

    def _minimal_contract_repair_v1(self, *, state_mode: str = "telemetry_candidate_v1") -> dict[str, Any]:
        return {
            "secondary_camera_mode": "wrist_dynamic",
            "enable_marker_overlay": False,
            "calibration_mode": "none",
            "interaction_mode": "orientation_sensitive_v1",
            "state_mode": state_mode,
            "emit_orientation_telemetry": True,
            "emit_camera_metadata": True,
            "canonical_lane": True,
        }

    def _reformulation_v0_contract(self) -> dict[str, Any]:
        return {
            "secondary_camera_mode": "wrist_dynamic",
            "enable_marker_overlay": False,
            "calibration_mode": "none",
            "interaction_mode": "orientation_sensitive_v1",
            "state_mode": "telemetry_candidate_v2",
            "emit_orientation_telemetry": True,
            "emit_camera_metadata": True,
            "canonical_lane": True,
        }

    def _make_lane_spec(
        self,
        *,
        lane_id: str,
        stage: str,
        env_contract: dict[str, Any],
        interventions: dict[str, Any] | None = None,
        claim_policy: ClaimPolicy = "diagnostic",
        note: str = "",
        experiment_id: str,
        coverage: dict[str, float] | None = None,
        cost: float = 1.0,
        dataset_config: DatasetConfig | None = None,
        train_config: TrainConfig | None = None,
    ) -> LaneSpec:
        spec = LaneSpec(
            lane_id=lane_id,
            stage=stage,
            env_contract_config=env_contract,
            interventions=interventions or {},
            dataset_config=dataset_config or DatasetConfig(train_seed_pool=self.train_seeds, heldout_seed_pool=self.heldout_seeds),
            train_config=train_config or TrainConfig(),
            claim_policy=claim_policy,
            note=note,
            experiment_id=experiment_id,
            coverage=coverage or {},
            cost=cost,
        )
        resolved_policy, violations = classify_claim_policy(self.claim_boundary, spec)
        if resolved_policy != spec.claim_policy and not self.policy["ALLOW_NEW_CLAIM"]:
            spec = LaneSpec(
                lane_id=spec.lane_id,
                stage=spec.stage,
                env_contract_config=spec.env_contract_config,
                interventions=spec.interventions,
                dataset_config=spec.dataset_config,
                train_config=spec.train_config,
                claim_policy=resolved_policy,
                note=(spec.note + f" Violations: {violations}").strip(),
                experiment_id=spec.experiment_id,
                coverage=spec.coverage,
                cost=spec.cost,
            )
        return spec

    def _materialize_contract(self, payload: dict[str, Any]) -> DrawerEnvContractConfig:
        keys = {
            "secondary_camera_mode",
            "enable_marker_overlay",
            "calibration_mode",
            "interaction_mode",
            "state_mode",
            "emit_orientation_telemetry",
            "emit_camera_metadata",
            "canonical_lane",
        }
        kwargs = {key: payload[key] for key in keys if key in payload}
        return DrawerEnvContractConfig(**kwargs)

    def _limited_seeds(self, seeds: list[int]) -> list[int]:
        limit = max(1, int(self.resource_limits["max_rollouts_per_experiment"]))
        return list(seeds)[:limit]

    def _run_rollout_set(self, spec: LaneSpec, seeds: list[int]) -> list[dict[str, Any]]:
        rollouts = []
        contract = self._materialize_contract(spec.env_contract_config)
        rotation_source = str(spec.interventions.get("rotation_source", "zero"))
        grasp_pose = np.eye(4, dtype=np.float32)
        for episode_index, seed in enumerate(self._limited_seeds(seeds)):
            last_error = None
            for attempt in range(int(self.resource_limits["max_retries"]) + 1):
                try:
                    rollout = build_robot_rollout(
                        seed=int(seed),
                        grasp_pose_world=grasp_pose,
                        episode_index=episode_index,
                        max_steps=int(spec.dataset_config.max_steps),
                        image_size=256,
                        contract=contract,
                        rotation_source=rotation_source,
                        claim_policy=spec.claim_policy,
                    )
                    rollouts.append(rollout)
                    last_error = None
                    break
                except Exception as exc:  # pragma: no cover - safety path for unattended execution
                    last_error = exc
                    if attempt >= int(self.resource_limits["max_retries"]):
                        raise
                    time.sleep(float(self.resource_limits["retry_backoff_seconds"]))
            if last_error is not None:
                raise last_error
        return rollouts

    def _rollout_summary(self, rollouts: list[dict[str, Any]]) -> dict[str, Any]:
        if not rollouts:
            return {
                "rollout_count": 0,
                "unique_success_rate": 0.0,
                "attach_rate": 0.0,
                "mean_max_drawer_fraction": 0.0,
                "avg_episode_length": 0.0,
                "success_count": 0,
            }
        return {
            "rollout_count": len(rollouts),
            "unique_success_rate": float(np.mean([1.0 if r.get("success") else 0.0 for r in rollouts])),
            "attach_rate": float(np.mean([1.0 if r.get("ever_attached") else 0.0 for r in rollouts])),
            "mean_max_drawer_fraction": float(np.mean([float(r.get("max_drawer_fraction", 0.0)) for r in rollouts])),
            "avg_episode_length": float(np.mean([float(r.get("steps", 0.0)) for r in rollouts])),
            "success_count": int(sum(1 for r in rollouts if r.get("success"))),
        }

    def _sample_images(self, rollouts: list[dict[str, Any]], key: str = "images", limit: int = 30) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for rollout in rollouts:
            for frame in rollout.get(key, []):
                frames.append(np.asarray(frame, dtype=np.uint8))
                if len(frames) >= limit:
                    return frames
        return frames

    def _secondary_residuals(self, rollouts: list[dict[str, Any]]) -> list[float]:
        residuals: list[float] = []
        for rollout in rollouts:
            for item in rollout.get("camera_metadata_trace", []):
                secondary = (item or {}).get("secondary", {})
                value = secondary.get("eef_cam_relativeness_residual")
                if value is not None:
                    residuals.append(float(value))
        return residuals

    def _observation_report(self, spec: LaneSpec, seeds: list[int], control_stats: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rollouts = self._run_rollout_set(spec, seeds)
        residuals = self._secondary_residuals(rollouts)
        target_stats = summarize_visual(self._sample_images(rollouts, limit=24))
        gap_payload = visual_gap(control_stats, target_stats)
        report = {
            "lane": spec.to_payload(),
            "claim_policy": spec.claim_policy,
            "secondary_camera_wrist_like": bool(residuals and float(np.mean(residuals)) < 0.02 and float(np.max(residuals)) < 0.05),
            "secondary_camera_residual_mean": float(np.mean(residuals)) if residuals else None,
            "secondary_camera_residual_max": float(np.max(residuals)) if residuals else None,
            "marker_overlay_enabled": bool(spec.env_contract_config.get("enable_marker_overlay", False)),
            "diagnostic_only": spec.claim_policy != "canonical" or spec.env_contract_config.get("calibration_mode") == "diagnostic_texture",
            "provenance_complete": True,
            "target_stats": target_stats,
            "gap": gap_payload,
            "weighted_visual_gap": weighted_visual_gap(gap_payload),
            "framing_score": 1.0 if residuals and float(np.mean(residuals)) < 0.02 else 0.0,
        }
        report["observation_contract_pass"] = observation_contract_pass(report)
        return rollouts, report

    def _per_dim_wasserstein(self, a: np.ndarray, b: np.ndarray) -> list[float]:
        dims = min(a.shape[1], b.shape[1])
        return [float(wasserstein_distance(a[:, i], b[:, i])) for i in range(dims)]

    def _per_dim_range_overlap(self, a: np.ndarray, b: np.ndarray) -> list[float]:
        dims = min(a.shape[1], b.shape[1])
        overlaps = []
        for i in range(dims):
            a_min, a_max = float(np.min(a[:, i])), float(np.max(a[:, i]))
            b_min, b_max = float(np.min(b[:, i])), float(np.max(b[:, i]))
            inter = max(0.0, min(a_max, b_max) - max(a_min, b_min))
            union = max(a_max, b_max) - min(a_min, b_min)
            overlaps.append(float(inter / union) if union > 1e-8 else 1.0)
        return overlaps

    def _temporal_smoothness_gap(self, a: np.ndarray, b: np.ndarray) -> list[float]:
        dims = min(a.shape[1], b.shape[1])
        a_diff = np.diff(a[:, :dims], axis=0)
        b_diff = np.diff(b[:, :dims], axis=0)
        return [float(abs(np.std(a_diff[:, i]) - np.std(b_diff[:, i]))) for i in range(dims)]

    def _normalized_state_score(self, wd: list[float], overlap: list[float], smooth_gap: list[float], control: np.ndarray, target: np.ndarray) -> float:
        normed = []
        dims = min(len(wd), control.shape[1], target.shape[1])
        for i, value in enumerate(wd[:dims]):
            a_min, a_max = float(np.min(control[:, i])), float(np.max(control[:, i]))
            b_min, b_max = float(np.min(target[:, i])), float(np.max(target[:, i]))
            span = max(a_max, b_max) - min(a_min, b_min)
            normed.append(float(value / max(span, 1e-6)))
        return float(np.mean(normed) + 0.25 * np.mean(smooth_gap[:dims]) + 0.1 * np.mean([1.0 - x for x in overlap[:dims]]))

    def _artifact_paths(self, experiment_id: str) -> tuple[Path, Path]:
        mapping = {
            "RF0": (RF0_ARTIFACT, RF0_REPORT),
            "RF1": (RF1_ARTIFACT, RF1_REPORT),
            "RF2": (RF2_ARTIFACT, RF2_REPORT),
            "RF3": (RF3_ARTIFACT, RF3_REPORT),
            "RF4": (RF4_ARTIFACT, RF4_REPORT),
            "RF5": (RF5_ARTIFACT, RF5_REPORT),
            "RF6": (RF6_ARTIFACT, RF6_REPORT),
        }
        return mapping[experiment_id]

    def experiment_catalog(self) -> list[dict[str, Any]]:
        return [
            {"id": "RF0", "deps": [], "coverage": {"H6_environment_invalid_for_claim": 0.7, "H0_same_problem_identity": 0.4}, "cost": 0.8, "runner": self.run_rf0},
            {"id": "RF1", "deps": ["RF0"], "coverage": {"H3_observation_visual_contract": 1.0, "H0_same_problem_identity": 0.7}, "cost": 1.0, "runner": self.run_rf1},
            {"id": "RF2", "deps": ["RF0"], "coverage": {"H1_embodiment_causal_contract": 1.0, "H6_environment_invalid_for_claim": 0.4}, "cost": 1.1, "runner": self.run_rf2},
            {"id": "RF3", "deps": ["RF0"], "coverage": {"H2_state_representability": 1.0, "H6_environment_invalid_for_claim": 0.7}, "cost": 1.1, "runner": self.run_rf3},
            {"id": "RF4", "deps": ["RF1"], "coverage": {"H3_observation_visual_contract": 1.0, "H0_same_problem_identity": 0.5}, "cost": 1.0, "runner": self.run_rf4},
            {"id": "RF5", "deps": ["RF1", "RF2", "RF3", "RF4"], "coverage": {"H6_environment_invalid_for_claim": 1.0, "H0_same_problem_identity": 0.8, "H1_embodiment_causal_contract": 0.4}, "cost": 1.4, "runner": self.run_rf5},
            {"id": "RF6", "deps": ["RF5"], "coverage": {"H6_environment_invalid_for_claim": 0.9, "H0_same_problem_identity": 0.7}, "cost": 1.0, "runner": self.run_rf6},
        ]

    def select_experiments(self) -> list[dict[str, Any]]:
        catalog = self.experiment_catalog()
        if "RF0" not in self.completed_experiments:
            return [next(item for item in catalog if item["id"] == "RF0")]
        completed = set(self.completed_experiments)
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        while len(selected) < self.max_experiments_per_cycle:
            candidates = []
            available = completed | selected_ids
            for item in catalog:
                if item["id"] in available:
                    continue
                if not all(dep in available for dep in item.get("deps", [])):
                    continue
                score = 0.0
                for hypothesis_id, coverage in item["coverage"].items():
                    state = self.board.get(hypothesis_id)
                    if state is None:
                        continue
                    score += float(state.get("importance", 1.0)) * unresolvedness(state) * float(coverage)
                candidates.append((score / max(float(item["cost"]), 1e-6), item))
            if not candidates:
                break
            candidates.sort(key=lambda pair: pair[0], reverse=True)
            chosen = candidates[0][1]
            selected.append(chosen)
            selected_ids.add(chosen["id"])
        return selected

    def run_rf0(self) -> dict[str, Any]:
        seeds = self.train_seeds[:3]
        baseline = self._make_lane_spec(
            lane_id="RF0_legacy_current",
            stage="probe",
            env_contract=self._legacy_contract(),
            interventions={"rotation_source": "zero", "reformulation_profile": "legacy_current"},
            claim_policy="diagnostic",
            note="Replicate current legacy baseline negative decision.",
            experiment_id="RF0",
        )
        minimal = self._make_lane_spec(
            lane_id="RF0_minimal_contract_repair_v1",
            stage="probe",
            env_contract=self._minimal_contract_repair_v1(state_mode="telemetry_candidate_v1"),
            interventions={"rotation_source": "aligned", "reformulation_profile": "minimal_contract_repair_v1"},
            claim_policy="canonical",
            note="Replicate prior minimal contract bundle negative decision.",
            experiment_id="RF0",
        )
        baseline_rollouts = self._run_rollout_set(baseline, seeds)
        minimal_rollouts = self._run_rollout_set(minimal, seeds)
        baseline_summary = self._rollout_summary(baseline_rollouts)
        minimal_summary = self._rollout_summary(minimal_rollouts)
        minimal_summary["orientation_causal_sensitivity"] = float(minimal_summary["mean_max_drawer_fraction"] - baseline_summary["mean_max_drawer_fraction"])
        baseline_summary["orientation_causal_sensitivity"] = 0.0
        ceiling = rollout_ceiling_lift(baseline_summary, minimal_summary)
        replicated_negative = bool(ceiling["score"] <= 0.0 and minimal_summary["mean_max_drawer_fraction"] <= baseline_summary["mean_max_drawer_fraction"] + 0.05)
        payload = {
            "experiment_id": "RF0",
            "generated_at": now_iso(),
            "passed": replicated_negative,
            "baseline_summary": baseline_summary,
            "minimal_summary": minimal_summary,
            "rollout_ceiling_lift": ceiling,
            "replicated_negative": replicated_negative,
        }
        updates = [
            {"hypothesis_id": "H6_environment_invalid_for_claim", "support_score": 0.7 if replicated_negative else -0.3, "alpha": 0.8, "evidence_ref": "RF0"},
            {"hypothesis_id": "H0_same_problem_identity", "support_score": -0.5 if replicated_negative else 0.3, "alpha": 0.6, "evidence_ref": "RF0"},
        ]
        artifact_path, report_path = self._artifact_paths("RF0")
        write_json_atomic(artifact_path, payload)
        write_text_atomic(report_path, "\n".join([
            "# RF0 Terminal Negative Replication",
            "",
            f"Generated: {payload['generated_at']}",
            f"- replicated_negative: {replicated_negative}",
            f"- rollout_ceiling_lift: {ceiling['score']:.6f}",
        ]).rstrip() + "\n")
        return {"payload": payload, "updates": updates, "lane_specs": [baseline.to_payload(), minimal.to_payload()]}

    def run_rf1(self) -> dict[str, Any]:
        seeds = self.train_seeds[:2]
        control_stats = summarize_visual(self.control_images)
        lanes = [
            self._make_lane_spec(
                lane_id="RF1_current_surface_clean",
                stage="probe",
                env_contract={**self._legacy_contract(), "enable_marker_overlay": False, "calibration_mode": "none", "canonical_lane": True},
                interventions={"rotation_source": "zero", "reformulation_profile": "legacy_current"},
                claim_policy="canonical",
                note="Fixed-scene surface cleaned of overlay/calibration for observation comparison.",
                experiment_id="RF1",
            ),
            self._make_lane_spec(
                lane_id="RF1_wrist_raw_no_overlay",
                stage="probe",
                env_contract={
                    "secondary_camera_mode": "wrist_dynamic",
                    "enable_marker_overlay": False,
                    "calibration_mode": "none",
                    "interaction_mode": "legacy_translation_only",
                    "state_mode": "m0_proxy",
                    "emit_orientation_telemetry": True,
                    "emit_camera_metadata": True,
                    "canonical_lane": True,
                },
                interventions={"rotation_source": "zero", "reformulation_profile": "minimal_contract_repair_v1"},
                claim_policy="canonical",
                note="Raw wrist-like observation contract without reformulated dynamics.",
                experiment_id="RF1",
            ),
            self._make_lane_spec(
                lane_id="RF1_reformulation_v0_observation",
                stage="probe",
                env_contract=self._reformulation_v0_contract(),
                interventions={"rotation_source": "aligned", "reformulation_profile": "reformulation_v0"},
                claim_policy="canonical",
                note="Observation contract under reformulation_v0 profile.",
                experiment_id="RF1",
            ),
        ]
        lane_reports = []
        for spec in lanes:
            _, report = self._observation_report(spec, seeds, control_stats)
            lane_reports.append(report)
        canonical_reports = [report for report in lane_reports if report["claim_policy"] == "canonical"]
        best_canonical = min(canonical_reports, key=lambda item: item["weighted_visual_gap"], default=None)
        baseline_gap = lane_reports[0]["weighted_visual_gap"]
        best_gain = visual_alignment_gain(baseline_gap, best_canonical["weighted_visual_gap"]) if best_canonical else -1.0
        observation_contract_passed = bool(best_canonical and best_canonical["observation_contract_pass"])
        payload = {
            "experiment_id": "RF1",
            "generated_at": now_iso(),
            "passed": observation_contract_passed,
            "control_stats": control_stats,
            "lanes": lane_reports,
            "best_canonical_lane": best_canonical,
            "best_visual_alignment_gain": best_gain,
            "observation_contract_pass": observation_contract_passed,
        }
        updates = [
            {"hypothesis_id": "H3_observation_visual_contract", "support_score": 0.8 if observation_contract_passed else 0.1, "alpha": 0.7, "evidence_ref": "RF1"},
            {"hypothesis_id": "H0_same_problem_identity", "support_score": 0.3 if observation_contract_passed and best_gain > 0.1 else -0.5, "alpha": 0.6, "evidence_ref": "RF1"},
        ]
        artifact_path, report_path = self._artifact_paths("RF1")
        write_json_atomic(artifact_path, payload)
        lines = [
            "# RF1 Reformulated Observation Contract Probe",
            "",
            f"Generated: {payload['generated_at']}",
            f"- observation_contract_pass: {observation_contract_passed}",
            f"- best_visual_alignment_gain: {best_gain:.6f}",
        ]
        for report in lane_reports:
            lines.append(
                f"- {report['lane']['lane_id']}: claim_policy={report['claim_policy']} wrist_like={report['secondary_camera_wrist_like']} weighted_visual_gap={report['weighted_visual_gap']:.6f}"
            )
        write_text_atomic(report_path, "\n".join(lines).rstrip() + "\n")
        return {"payload": payload, "updates": updates, "lane_specs": [spec.to_payload() for spec in lanes]}

    def run_rf2(self) -> dict[str, Any]:
        seeds = self.train_seeds[:3]
        lane_specs: list[LaneSpec] = []
        for interaction_mode in ["legacy_translation_only", "orientation_sensitive_v1"]:
            for rotation_source in ["zero", "aligned", "random"]:
                lane_specs.append(
                    self._make_lane_spec(
                        lane_id=f"RF2_{interaction_mode}_{rotation_source}",
                        stage="probe",
                        env_contract={
                            "secondary_camera_mode": "wrist_dynamic",
                            "enable_marker_overlay": False,
                            "calibration_mode": "none",
                            "interaction_mode": interaction_mode,
                            "state_mode": "m0_proxy",
                            "emit_orientation_telemetry": True,
                            "emit_camera_metadata": True,
                            "canonical_lane": interaction_mode == "orientation_sensitive_v1",
                        },
                        interventions={"rotation_source": rotation_source, "reformulation_profile": "minimal_contract_repair_v1"},
                        claim_policy="diagnostic" if interaction_mode == "legacy_translation_only" else "canonical",
                        note="Reformulated action-causality matrix lane.",
                        experiment_id="RF2",
                    )
                )
        matrix: dict[str, Any] = {}
        for spec in lane_specs:
            rollouts = self._run_rollout_set(spec, seeds)
            matrix[spec.lane_id] = {"lane": spec.to_payload(), "rollout_summary": self._rollout_summary(rollouts)}
        aligned_sensitive = matrix["RF2_orientation_sensitive_v1_aligned"]["rollout_summary"]["mean_max_drawer_fraction"]
        zero_sensitive = matrix["RF2_orientation_sensitive_v1_zero"]["rollout_summary"]["mean_max_drawer_fraction"]
        random_sensitive = matrix["RF2_orientation_sensitive_v1_random"]["rollout_summary"]["mean_max_drawer_fraction"]
        legacy_aligned = matrix["RF2_legacy_translation_only_aligned"]["rollout_summary"]["mean_max_drawer_fraction"]
        legacy_zero = matrix["RF2_legacy_translation_only_zero"]["rollout_summary"]["mean_max_drawer_fraction"]
        legacy_random = matrix["RF2_legacy_translation_only_random"]["rollout_summary"]["mean_max_drawer_fraction"]
        legacy_gap = float(legacy_aligned - max(legacy_zero, legacy_random))
        orientation_causal_sensitivity = float(aligned_sensitive - max(zero_sensitive, random_sensitive))
        effect_size = float(aligned_sensitive - random_sensitive)
        payload = {
            "experiment_id": "RF2",
            "generated_at": now_iso(),
            "passed": orientation_causal_sensitivity > 0.01 and effect_size > 0.01,
            "matrix": matrix,
            "orientation_causal_sensitivity": orientation_causal_sensitivity,
            "legacy_alignment_gap": legacy_gap,
            "aligned_beats_random_under_sensitive": bool(aligned_sensitive > random_sensitive),
            "effect_size": effect_size,
        }
        updates = [
            {"hypothesis_id": "H1_embodiment_causal_contract", "support_score": 0.9 if payload["passed"] and abs(legacy_gap) < 0.02 else -0.3, "alpha": 1.0, "evidence_ref": "RF2"},
            {"hypothesis_id": "H6_environment_invalid_for_claim", "support_score": 0.5 if orientation_causal_sensitivity <= 0.0 else -0.2, "alpha": 0.5, "evidence_ref": "RF2"},
        ]
        artifact_path, report_path = self._artifact_paths("RF2")
        write_json_atomic(artifact_path, payload)
        lines = [
            "# RF2 Reformulated Action Causality Probe",
            "",
            f"Generated: {payload['generated_at']}",
            f"- orientation_causal_sensitivity: {orientation_causal_sensitivity:.6f}",
            f"- effect_size: {effect_size:.6f}",
            f"- legacy_alignment_gap: {legacy_gap:.6f}",
        ]
        write_text_atomic(report_path, "\n".join(lines).rstrip() + "\n")
        return {"payload": payload, "updates": updates, "lane_specs": [spec.to_payload() for spec in lane_specs]}

    def run_rf3(self) -> dict[str, Any]:
        seeds = self.train_seeds[:3]
        candidates = {
            "M0_proxy": self._make_lane_spec(
                lane_id="RF3_m0_proxy",
                stage="probe",
                env_contract={
                    "secondary_camera_mode": "wrist_dynamic",
                    "enable_marker_overlay": False,
                    "calibration_mode": "none",
                    "interaction_mode": "legacy_translation_only",
                    "state_mode": "m0_proxy",
                    "emit_orientation_telemetry": True,
                    "emit_camera_metadata": True,
                    "canonical_lane": False,
                },
                interventions={"rotation_source": "zero", "reformulation_profile": "legacy_current"},
                claim_policy="diagnostic",
                note="Current proxy baseline.",
                experiment_id="RF3",
            ),
            "M_pose": self._make_lane_spec(
                lane_id="RF3_eef_pose_gripper",
                stage="probe",
                env_contract={
                    "secondary_camera_mode": "wrist_dynamic",
                    "enable_marker_overlay": False,
                    "calibration_mode": "none",
                    "interaction_mode": "legacy_translation_only",
                    "state_mode": "eef_pose_gripper",
                    "emit_orientation_telemetry": True,
                    "emit_camera_metadata": True,
                    "canonical_lane": False,
                },
                interventions={"rotation_source": "zero", "reformulation_profile": "legacy_current"},
                claim_policy="diagnostic",
                note="Pose/gripper diagnostic candidate.",
                experiment_id="RF3",
            ),
            "M_telemetry_v1": self._make_lane_spec(
                lane_id="RF3_telemetry_candidate_v1",
                stage="probe",
                env_contract=self._minimal_contract_repair_v1(state_mode="telemetry_candidate_v1"),
                interventions={"rotation_source": "aligned", "reformulation_profile": "minimal_contract_repair_v1"},
                claim_policy="canonical",
                note="Telemetry candidate v1.",
                experiment_id="RF3",
            ),
            "M_telemetry_v2": self._make_lane_spec(
                lane_id="RF3_telemetry_candidate_v2",
                stage="probe",
                env_contract=self._reformulation_v0_contract(),
                interventions={"rotation_source": "aligned", "reformulation_profile": "reformulation_v0"},
                claim_policy="canonical",
                note="Telemetry candidate v2 under reformulation_v0.",
                experiment_id="RF3",
            ),
        }
        candidate_scores: dict[str, Any] = {}
        for candidate_id, spec in candidates.items():
            rollouts = self._run_rollout_set(spec, seeds)
            states = np.concatenate([np.asarray(r["states"], dtype=np.float32) for r in rollouts], axis=0)
            state_spec = rollouts[0].get("state_spec", {}) if rollouts else {}
            valid = not bool(state_spec.get("fraud_padding", False))
            if valid:
                wd = self._per_dim_wasserstein(self.control_state, states)
                overlap = self._per_dim_range_overlap(self.control_state, states)
                smooth_gap = self._temporal_smoothness_gap(self.control_state, states)
                overall = self._normalized_state_score(wd, overlap, smooth_gap, self.control_state, states)
                candidate_scores[candidate_id] = {
                    "valid": True,
                    "overall_score": overall,
                    "wasserstein": wd,
                    "range_overlap": overlap,
                    "temporal_smoothness_gap": smooth_gap,
                    "state_spec": state_spec,
                    "claim_policy": spec.claim_policy,
                }
            else:
                candidate_scores[candidate_id] = {
                    "valid": False,
                    "overall_score": None,
                    "state_spec": state_spec,
                    "invalid_reason": "fraud_padding_or_duplicate_dims",
                    "claim_policy": spec.claim_policy,
                }
        valid_candidates = {k: v for k, v in candidate_scores.items() if v.get("valid")}
        selected_mapping = min(valid_candidates, key=lambda key: valid_candidates[key]["overall_score"], default="IMPOSSIBLE")
        baseline_score = float(candidate_scores.get("M0_proxy", {}).get("overall_score") or 1.0)
        telemetry_candidates = {k: v for k, v in candidate_scores.items() if k.startswith("M_telemetry") and v.get("overall_score") is not None}
        best_telemetry_key = min(telemetry_candidates, key=lambda key: telemetry_candidates[key]["overall_score"], default=None)
        best_telemetry_score = telemetry_candidates[best_telemetry_key]["overall_score"] if best_telemetry_key else None
        telemetry_gain = state_alignment_gain(baseline_score, best_telemetry_score) if best_telemetry_score is not None else -1.0
        impossible = best_telemetry_key is None
        telemetry_spec = telemetry_candidates.get(best_telemetry_key, {}).get("state_spec", {}) if best_telemetry_key else {}
        telemetry_semantic_pass = bool(
            best_telemetry_key
            and telemetry_candidates[best_telemetry_key].get("valid")
            and not telemetry_spec.get("fraud_padding", False)
            and telemetry_spec.get("all_dims_explained", False)
        )
        strict_state_gate_ready = bool(
            not impossible
            and selected_mapping == best_telemetry_key
            and telemetry_semantic_pass
            and telemetry_gain >= 0.05
        )
        payload = {
            "experiment_id": "RF3",
            "generated_at": now_iso(),
            "passed": strict_state_gate_ready,
            "selected_state_mapping": selected_mapping,
            "best_telemetry_candidate": best_telemetry_key,
            "candidate_state_scores": candidate_scores,
            "state_alignment_gain_vs_m0": telemetry_gain,
            "telemetry_semantic_pass": telemetry_semantic_pass,
            "strict_state_gate_ready": strict_state_gate_ready,
            "impossible": impossible,
        }
        updates = [
            {"hypothesis_id": "H2_state_representability", "support_score": 0.8 if impossible or telemetry_gain < 0.05 else -0.4, "alpha": 1.0, "evidence_ref": "RF3"},
            {"hypothesis_id": "H6_environment_invalid_for_claim", "support_score": 0.7 if impossible else 0.2, "alpha": 0.8, "evidence_ref": "RF3"},
        ]
        artifact_path, report_path = self._artifact_paths("RF3")
        write_json_atomic(artifact_path, payload)
        lines = [
            "# RF3 Reformulated State Semantics Probe",
            "",
            f"Generated: {payload['generated_at']}",
            f"- selected_state_mapping: {selected_mapping}",
            f"- best_telemetry_candidate: {best_telemetry_key}",
            f"- impossible: {impossible}",
            f"- telemetry_semantic_pass: {telemetry_semantic_pass}",
            f"- strict_state_gate_ready: {strict_state_gate_ready}",
            f"- state_alignment_gain_vs_m0: {telemetry_gain:.6f}",
        ]
        write_text_atomic(report_path, "\n".join(lines).rstrip() + "\n")
        return {"payload": payload, "updates": updates, "lane_specs": [spec.to_payload() for spec in candidates.values()]}

    def run_rf4(self) -> dict[str, Any]:
        seeds = self.train_seeds[:2]
        control_stats = summarize_visual(self.control_images)
        lanes = [
            self._make_lane_spec(
                lane_id="RF4_current_surface_clean",
                stage="probe",
                env_contract={**self._legacy_contract(), "enable_marker_overlay": False, "calibration_mode": "none", "canonical_lane": True},
                interventions={"rotation_source": "zero", "reformulation_profile": "legacy_current"},
                claim_policy="canonical",
                note="Current visual surface cleaned for readability baseline.",
                experiment_id="RF4",
            ),
            self._make_lane_spec(
                lane_id="RF4_wrist_raw_surface",
                stage="probe",
                env_contract={
                    "secondary_camera_mode": "wrist_dynamic",
                    "enable_marker_overlay": False,
                    "calibration_mode": "none",
                    "interaction_mode": "legacy_translation_only",
                    "state_mode": "m0_proxy",
                    "emit_orientation_telemetry": True,
                    "emit_camera_metadata": True,
                    "canonical_lane": True,
                },
                interventions={"rotation_source": "zero", "reformulation_profile": "minimal_contract_repair_v1"},
                claim_policy="canonical",
                note="Wrist/raw/no-overlay visual surface.",
                experiment_id="RF4",
            ),
            self._make_lane_spec(
                lane_id="RF4_reformulation_v0_surface",
                stage="probe",
                env_contract=self._reformulation_v0_contract(),
                interventions={"rotation_source": "aligned", "reformulation_profile": "reformulation_v0"},
                claim_policy="canonical",
                note="Reformulated visual surface under reformulation_v0.",
                experiment_id="RF4",
            ),
        ]
        lane_reports = []
        for spec in lanes:
            _, report = self._observation_report(spec, seeds, control_stats)
            lane_reports.append(report)
        baseline_gap = lane_reports[0]["weighted_visual_gap"]
        canonical_reports = [report for report in lane_reports if report["claim_policy"] == "canonical"]
        best_lane = min(canonical_reports, key=lambda item: item["weighted_visual_gap"], default=None)
        best_gain = visual_alignment_gain(baseline_gap, best_lane["weighted_visual_gap"]) if best_lane else -1.0
        observation_semantics_pass = bool(best_lane and best_lane["observation_contract_pass"])
        perceptual_readability_pass = bool(best_lane and best_gain >= 0.10 and best_lane["weighted_visual_gap"] < baseline_gap)
        payload = {
            "experiment_id": "RF4",
            "generated_at": now_iso(),
            "passed": observation_semantics_pass and perceptual_readability_pass,
            "control_stats": control_stats,
            "lanes": lane_reports,
            "best_lane": best_lane,
            "best_visual_alignment_gain": best_gain,
            "observation_semantics_pass": observation_semantics_pass,
            "perceptual_readability_pass": perceptual_readability_pass,
        }
        updates = [
            {"hypothesis_id": "H3_observation_visual_contract", "support_score": 0.8 if payload["passed"] else 0.2, "alpha": 0.7, "evidence_ref": "RF4"},
            {"hypothesis_id": "H0_same_problem_identity", "support_score": 0.2 if payload["passed"] else -0.4, "alpha": 0.5, "evidence_ref": "RF4"},
        ]
        artifact_path, report_path = self._artifact_paths("RF4")
        write_json_atomic(artifact_path, payload)
        write_text_atomic(report_path, "\n".join([
            "# RF4 Reformulated Visual Readability Probe",
            "",
            f"Generated: {payload['generated_at']}",
            f"- best_lane: {best_lane['lane']['lane_id'] if best_lane else 'none'}",
            f"- observation_semantics_pass: {observation_semantics_pass}",
            f"- perceptual_readability_pass: {perceptual_readability_pass}",
            f"- best_visual_alignment_gain: {best_gain:.6f}",
        ]).rstrip() + "\n")
        return {"payload": payload, "updates": updates, "lane_specs": [spec.to_payload() for spec in lanes]}

    def run_rf5(self) -> dict[str, Any]:
        seeds = self.train_seeds[:3]
        control_stats = summarize_visual(self.control_images)
        profiles = {
            "legacy_current": self._make_lane_spec(
                lane_id="RF5_legacy_current",
                stage="probe",
                env_contract=self._legacy_contract(),
                interventions={"rotation_source": "zero", "reformulation_profile": "legacy_current"},
                claim_policy="diagnostic",
                note="Legacy current formulation.",
                experiment_id="RF5",
            ),
            "minimal_contract_repair_v1": self._make_lane_spec(
                lane_id="RF5_minimal_contract_repair_v1",
                stage="probe",
                env_contract=self._minimal_contract_repair_v1(state_mode="telemetry_candidate_v1"),
                interventions={"rotation_source": "aligned", "reformulation_profile": "minimal_contract_repair_v1"},
                claim_policy="canonical",
                note="Minimal repair bundle under telemetry_candidate_v1.",
                experiment_id="RF5",
            ),
            "reformulation_v0": self._make_lane_spec(
                lane_id="RF5_reformulation_v0",
                stage="probe",
                env_contract=self._reformulation_v0_contract(),
                interventions={"rotation_source": "aligned", "reformulation_profile": "reformulation_v0"},
                claim_policy="canonical",
                note="Reformulation bootstrap bundle under telemetry_candidate_v2.",
                experiment_id="RF5",
            ),
        }
        observations: dict[str, Any] = {}
        summaries: dict[str, Any] = {}
        for profile_id, spec in profiles.items():
            rollouts, report = self._observation_report(spec, seeds, control_stats)
            summary = self._rollout_summary(rollouts)
            summary["orientation_causal_sensitivity"] = float(np.mean([float(r.get("max_drawer_fraction", 0.0)) for r in rollouts])) if spec.env_contract_config.get("interaction_mode") == "orientation_sensitive_v1" else 0.0
            observations[profile_id] = report
            summaries[profile_id] = summary
        baseline_summary = summaries["legacy_current"]
        rf3 = load_json(RF3_ARTIFACT, {})
        candidate_scores = rf3.get("candidate_state_scores", {})
        baseline_state_score = float(candidate_scores.get("M0_proxy", {}).get("overall_score") or 1.0)
        v1_state_score = float(candidate_scores.get("M_telemetry_v1", {}).get("overall_score") or baseline_state_score)
        v2_state_score = float(candidate_scores.get("M_telemetry_v2", {}).get("overall_score") or baseline_state_score)
        profile_reports = {}
        for profile_id in ["minimal_contract_repair_v1", "reformulation_v0"]:
            state_score = v1_state_score if profile_id == "minimal_contract_repair_v1" else v2_state_score
            ceiling = rollout_ceiling_lift(baseline_summary, summaries[profile_id])
            sensor_gain = sensor_contract_gain(
                wrist_relativeness_gain=1.0 if observations[profile_id].get("secondary_camera_wrist_like") else 0.0,
                no_overlay_gain=1.0 if not observations[profile_id].get("marker_overlay_enabled") else 0.0,
                visual_alignment_gain_value=visual_alignment_gain(observations["legacy_current"]["weighted_visual_gap"], observations[profile_id]["weighted_visual_gap"]),
                framing_gain=observations[profile_id].get("framing_score", 0.0),
            )
            profile_reports[profile_id] = {
                "lane": profiles[profile_id].to_payload(),
                "rollout_summary": summaries[profile_id],
                "observation_report": observations[profile_id],
                "rollout_ceiling_lift": ceiling,
                "state_alignment_gain": state_alignment_gain(baseline_state_score, state_score),
                "sensor_contract_gain": sensor_gain,
            }
        best_profile_id = max(
            profile_reports,
            key=lambda key: 0.5 * profile_reports[key]["rollout_ceiling_lift"]["score"]
            + 0.25 * profile_reports[key]["state_alignment_gain"]
            + 0.25 * profile_reports[key]["sensor_contract_gain"]["score"],
        )
        best_profile = profile_reports[best_profile_id]
        passed = bool(best_profile["rollout_ceiling_lift"]["score"] > 0.0 and best_profile["sensor_contract_gain"]["score"] > 0.25)
        payload = {
            "experiment_id": "RF5",
            "generated_at": now_iso(),
            "passed": passed,
            "baseline_summary": baseline_summary,
            "profiles": profile_reports,
            "best_profile_id": best_profile_id,
            "best_profile": best_profile,
            "rollout_ceiling_lift": best_profile["rollout_ceiling_lift"],
            "state_alignment_gain": best_profile["state_alignment_gain"],
            "sensor_contract_gain": best_profile["sensor_contract_gain"],
        }
        updates = [
            {"hypothesis_id": "H6_environment_invalid_for_claim", "support_score": 0.9 if best_profile["rollout_ceiling_lift"]["score"] <= 0.0 else -0.4, "alpha": 1.0, "evidence_ref": "RF5"},
            {"hypothesis_id": "H0_same_problem_identity", "support_score": 0.6 if passed else -0.6, "alpha": 0.7, "evidence_ref": "RF5"},
        ]
        artifact_path, report_path = self._artifact_paths("RF5")
        write_json_atomic(artifact_path, payload)
        write_text_atomic(report_path, "\n".join([
            "# RF5 Reformulated Bundle A/B",
            "",
            f"Generated: {payload['generated_at']}",
            f"- best_profile_id: {best_profile_id}",
            f"- rollout_ceiling_lift: {best_profile['rollout_ceiling_lift']['score']:.6f}",
            f"- state_alignment_gain: {best_profile['state_alignment_gain']:.6f}",
            f"- sensor_contract_gain: {best_profile['sensor_contract_gain']['score']:.6f}",
            f"- passed: {passed}",
        ]).rstrip() + "\n")
        return {"payload": payload, "updates": updates, "lane_specs": [spec.to_payload() for spec in profiles.values()]}

    def run_rf6(self) -> dict[str, Any]:
        rf5 = load_json(RF5_ARTIFACT, {})
        best_profile_id = rf5.get("best_profile_id", "reformulation_v0")
        seeds = self.train_seeds[2:5] if len(self.train_seeds) >= 5 else self.train_seeds[:3]
        baseline = self._make_lane_spec(
            lane_id="RF6_legacy_current_replication",
            stage="probe",
            env_contract=self._legacy_contract(),
            interventions={"rotation_source": "zero", "reformulation_profile": "legacy_current"},
            claim_policy="diagnostic",
            note="Replication baseline on alternate seeds.",
            experiment_id="RF6",
        )
        target_contract = self._reformulation_v0_contract() if best_profile_id == "reformulation_v0" else self._minimal_contract_repair_v1(state_mode="telemetry_candidate_v1")
        target_state_mode = target_contract["state_mode"]
        candidate = self._make_lane_spec(
            lane_id=f"RF6_{best_profile_id}_replication",
            stage="probe",
            env_contract=target_contract,
            interventions={"rotation_source": "aligned", "reformulation_profile": best_profile_id},
            claim_policy="canonical",
            note=f"Replication run for {best_profile_id} on alternate seeds.",
            experiment_id="RF6",
        )
        baseline_rollouts = self._run_rollout_set(baseline, seeds)
        candidate_rollouts = self._run_rollout_set(candidate, seeds)
        baseline_summary = self._rollout_summary(baseline_rollouts)
        candidate_summary = self._rollout_summary(candidate_rollouts)
        ceiling = rollout_ceiling_lift(baseline_summary, candidate_summary)
        prior_score = float((rf5.get("best_profile") or {}).get("rollout_ceiling_lift", {}).get("score", 0.0))
        replication_consistent = bool((prior_score <= 0.0 and ceiling["score"] <= 0.0) or (prior_score > 0.0 and ceiling["score"] > 0.0))
        payload = {
            "experiment_id": "RF6",
            "generated_at": now_iso(),
            "passed": replication_consistent,
            "best_profile_id": best_profile_id,
            "replicated_state_mode": target_state_mode,
            "baseline_summary": baseline_summary,
            "candidate_summary": candidate_summary,
            "rollout_ceiling_lift": ceiling,
            "prior_rollout_ceiling_lift": prior_score,
            "replication_consistent": replication_consistent,
        }
        updates = [
            {"hypothesis_id": "H6_environment_invalid_for_claim", "support_score": 0.6 if replication_consistent and ceiling["score"] <= 0.0 else -0.2, "alpha": 0.8, "evidence_ref": "RF6"},
            {"hypothesis_id": "H0_same_problem_identity", "support_score": -0.4 if replication_consistent and ceiling["score"] <= 0.0 else 0.2, "alpha": 0.5, "evidence_ref": "RF6"},
        ]
        artifact_path, report_path = self._artifact_paths("RF6")
        write_json_atomic(artifact_path, payload)
        write_text_atomic(report_path, "\n".join([
            "# RF6 Replicate Strongest Signal",
            "",
            f"Generated: {payload['generated_at']}",
            f"- best_profile_id: {best_profile_id}",
            f"- replication_consistent: {replication_consistent}",
            f"- rollout_ceiling_lift: {ceiling['score']:.6f}",
            f"- prior_rollout_ceiling_lift: {prior_score:.6f}",
        ]).rstrip() + "\n")
        return {"payload": payload, "updates": updates, "lane_specs": [baseline.to_payload(), candidate.to_payload()]}

    def evaluate_gates(self) -> dict[str, Any]:
        completed = set(self.completed_experiments)
        rf0 = load_json(RF0_ARTIFACT, {}) if "RF0" in completed else {}
        rf1 = load_json(RF1_ARTIFACT, {}) if "RF1" in completed else {}
        rf2 = load_json(RF2_ARTIFACT, {}) if "RF2" in completed else {}
        rf3 = load_json(RF3_ARTIFACT, {}) if "RF3" in completed else {}
        rf4 = load_json(RF4_ARTIFACT, {}) if "RF4" in completed else {}
        rf5 = load_json(RF5_ARTIFACT, {}) if "RF5" in completed else {}
        rf6 = load_json(RF6_ARTIFACT, {}) if "RF6" in completed else {}

        decisions: list[GateDecision] = []
        baseline_details = self._baseline_integrity_details()
        baseline_ready = bool(
            baseline_details["p1j_exists"]
            and baseline_details["p1l_exists"]
            and baseline_details["control_baseline_pass"]
            and baseline_details["parity_non_blocker"]
            and baseline_details["authority_readable"]
        )
        decisions.append(GateDecision(
            gate_id="G0_baseline_integrity",
            passed=baseline_ready,
            summary="Authority readable and verified p1j/p1l baseline anchors confirm a healthy control baseline and non-blocking parity.",
            blocking=True,
            details=baseline_details,
        ))

        rf1_best = rf1.get("best_canonical_lane") or {}
        decisions.append(GateDecision(
            gate_id="G1_observation_contract",
            passed=bool(rf1 and rf1.get("observation_contract_pass", False)),
            summary="Observation contract requires wrist-like second camera, no overlay, no diagnostic trick, and provenance-complete canonical lane.",
            blocking=True,
            details=rf1_best,
        ))

        decisions.append(GateDecision(
            gate_id="G2_action_causality",
            passed=bool(rf2 and rf2.get("orientation_causal_sensitivity", 0.0) > 0.01 and rf2.get("effect_size", 0.0) > 0.01 and rf2.get("aligned_beats_random_under_sensitive", False)),
            summary="Rotation only counts if aligned rotation measurably beats zero/random under orientation-sensitive physics.",
            blocking=True,
            details={
                "orientation_causal_sensitivity": rf2.get("orientation_causal_sensitivity"),
                "effect_size": rf2.get("effect_size"),
                "aligned_beats_random_under_sensitive": rf2.get("aligned_beats_random_under_sensitive"),
            },
        ))

        impossible = bool(rf3.get("impossible", False))
        state_gain = float(rf3.get("state_alignment_gain_vs_m0", -1.0))
        decisions.append(GateDecision(
            gate_id="G3_state_semantics",
            passed=bool(rf3 and rf3.get("strict_state_gate_ready", False) and not impossible and state_gain >= 0.05),
            summary="State gate requires a telemetry-compatible, provenance-complete, non-duplicated state candidate with non-trivial measured adequacy over M0.",
            blocking=True,
            details={
                "selected_state_mapping": rf3.get("selected_state_mapping"),
                "best_telemetry_candidate": rf3.get("best_telemetry_candidate"),
                "telemetry_semantic_pass": rf3.get("telemetry_semantic_pass"),
                "strict_state_gate_ready": rf3.get("strict_state_gate_ready"),
                "state_alignment_gain_vs_m0": state_gain,
                "impossible": impossible,
            },
        ))

        best_lane = rf4.get("best_lane") or {}
        observation_semantics_pass = bool(rf4.get("observation_semantics_pass", False))
        perceptual_readability_pass = bool(rf4.get("perceptual_readability_pass", False))
        decisions.append(GateDecision(
            gate_id="G4_visual_canonicality",
            passed=bool(rf4 and observation_semantics_pass and perceptual_readability_pass and best_lane.get("claim_policy") == "canonical"),
            summary="Visual gate requires clean canonical observation semantics plus material perceptual readability improvement.",
            blocking=True,
            details={
                "best_lane": best_lane,
                "observation_semantics_pass": observation_semantics_pass,
                "perceptual_readability_pass": perceptual_readability_pass,
                "best_visual_alignment_gain": rf4.get("best_visual_alignment_gain"),
            },
        ))

        best_profile = rf5.get("best_profile") or {}
        best_profile_lane = best_profile.get("lane") or {}
        train_eligible = bool(
            rf5
            and all(decision.passed for decision in decisions)
            and best_profile_lane.get("claim_policy") == "canonical"
            and (
                float((rf5.get("rollout_ceiling_lift") or {}).get("score", 0.0)) >= 0.15
                or float(rf5.get("state_alignment_gain", 0.0)) >= 0.20
                or float((rf5.get("sensor_contract_gain") or {}).get("score", 0.0)) >= 0.25
            )
        )
        decisions.append(GateDecision(
            gate_id="G5_training_eligibility",
            passed=train_eligible,
            summary="Training is eligible only after G1-G4 pass and at least one causal gain threshold is met.",
            blocking=False,
            details={
                "best_profile_id": rf5.get("best_profile_id"),
                "rollout_ceiling_lift": (rf5.get("rollout_ceiling_lift") or {}).get("score"),
                "state_alignment_gain": rf5.get("state_alignment_gain"),
                "sensor_contract_gain": (rf5.get("sensor_contract_gain") or {}).get("score"),
            },
        ))

        route_decision = self._classify_route(decisions, rf0=rf0, rf5=rf5, rf6=rf6)
        gate_report = {
            "generated_at": now_iso(),
            "gates": [decision.__dict__ for decision in decisions],
            "contract_validity_score": contract_validity_score(*(decision.passed for decision in decisions[:4])),
            "scientific_terminal_state": route_decision["scientific_terminal_state"],
            "route_next_branch": route_decision["route_next_branch"],
            "route_why": route_decision.get("why"),
            "route_decision": route_decision,
        }
        return gate_report

    def _classify_route(self, decisions: list[GateDecision], *, rf0: dict[str, Any], rf5: dict[str, Any], rf6: dict[str, Any]) -> dict[str, Any]:
        gate_map = {decision.gate_id: decision for decision in decisions}
        h0 = float(self.board.get("H0_same_problem_identity", {}).get("posterior", 0.5))
        h6 = float(self.board.get("H6_environment_invalid_for_claim", {}).get("posterior", 0.5))
        g1 = gate_map.get("G1_observation_contract")
        g3 = gate_map.get("G3_state_semantics")
        g4 = gate_map.get("G4_visual_canonicality")
        g5 = gate_map.get("G5_training_eligibility")
        if g5 and g5.passed:
            return {
                "scientific_terminal_state": SCIENTIFIC_TERMINALS["aligned"],
                "route_next_branch": ROUTE_OPTIONS["tiny_retrain"],
                "why": "A reformulated canonical lane passed G1-G5 and is eligible for tiny retrain confirmation.",
                "h0_posterior": h0,
                "h6_posterior": h6,
            }
        enough_negative_evidence = bool(rf0 and rf5 and rf0.get("replicated_negative"))
        replicated = bool(rf6.get("replication_consistent", False)) if rf6 else False
        if enough_negative_evidence:
            if replicated and h6 >= 0.85 and h0 <= 0.25 and ((g1 and not g1.passed) or (g3 and not g3.passed) or (g4 and not g4.passed)):
                return {
                    "scientific_terminal_state": SCIENTIFIC_TERMINALS["unsupported"],
                    "route_next_branch": ROUTE_OPTIONS["reformulation"],
                    "why": "Repeated evidence supports same-problem identity failure under the current formulation.",
                    "h0_posterior": h0,
                    "h6_posterior": h6,
                }
            if h6 >= 0.70:
                return {
                    "scientific_terminal_state": SCIENTIFIC_TERMINALS["minimal_negative"],
                    "route_next_branch": ROUTE_OPTIONS["reformulation"],
                    "why": "Minimal repair remains insufficient and the next scientific branch should be environment reformulation.",
                    "h0_posterior": h0,
                    "h6_posterior": h6,
                }
        return {
            "scientific_terminal_state": None,
            "route_next_branch": ROUTE_OPTIONS["stay"],
            "why": "More reformulation evidence is still required before terminal classification.",
            "h0_posterior": h0,
            "h6_posterior": h6,
        }

    def _proposed_truth_delta(self, gate_report: dict[str, Any], route_decision: dict[str, Any], selected: list[dict[str, Any]]) -> dict[str, Any]:
        scientific_terminal_state = route_decision.get("scientific_terminal_state")
        route_next_branch = route_decision.get("route_next_branch")
        if scientific_terminal_state:
            return {
                "phase": "v65_REFORMULATION_BOOTSTRAP_AFTER_STAGE_A_TERMINAL_NEGATIVE",
                "verdict": route_decision.get("why"),
                "next_action": {
                    "type": route_next_branch,
                    "id": route_next_branch,
                    "status": "pending",
                    "priority": "P0",
                },
                "scientific_terminal_state": scientific_terminal_state,
            }
        return {
            "phase": self.authority.get("phase"),
            "verdict": self.authority.get("verdict"),
            "next_action": self.authority.get("next_action"),
            "selected_experiments": [item["id"] for item in selected],
        }

    def _proposed_next_actions(self, route_decision: dict[str, Any], selected: list[dict[str, Any]]) -> dict[str, Any]:
        scientific_terminal_state = route_decision.get("scientific_terminal_state")
        if scientific_terminal_state:
            return {
                "actions": [
                    {
                        "type": route_decision.get("route_next_branch"),
                        "id": route_decision.get("route_next_branch"),
                        "status": "pending",
                        "priority": "P0",
                        "scientific_terminal_state": scientific_terminal_state,
                    }
                ]
            }
        remaining = [item["id"] for item in self.experiment_catalog() if item["id"] not in self.completed_experiments and item["id"] not in [x["id"] for x in selected]]
        return {
            "actions": [
                {
                    "type": "ROOT_CAUSE_CONTROLLER_NEXT_EXPERIMENTS",
                    "id": "root_cause_controller_pending_queue",
                    "status": "pending",
                    "priority": "P0",
                    "remaining": remaining,
                }
            ]
        }

    def _cycle_memo(self, cycle_id: str, selected: list[dict[str, Any]], gate_report: dict[str, Any], route_decision: dict[str, Any]) -> str:
        lines = [
            "# Reformulation Root-Cause Controller Cycle Memo",
            "",
            f"- cycle_id: {cycle_id}",
            f"- authority_phase: {self.authority.get('phase')}",
            f"- selected_experiments: {[item['id'] for item in selected]}",
            f"- scientific_terminal_state: {route_decision.get('scientific_terminal_state')}",
            f"- route_next_branch: {route_decision.get('route_next_branch')}",
            "",
            "## Gates",
        ]
        for gate in gate_report.get("gates", []):
            lines.append(f"- {gate['gate_id']}: passed={gate['passed']} summary={gate['summary']}")
        lines.extend(["", "## Hypotheses"])
        for hypothesis_id, state in self.board.items():
            lines.append(f"- {hypothesis_id}: posterior={float(state.get('posterior', 0.5)):.4f}")
        lines.extend(["", "## Route", f"- why: {route_decision.get('why')}"])
        return "\n".join(lines).rstrip() + "\n"

    def _cycle_summary(self, cycle_id: str, selected: list[dict[str, Any]], route_decision: dict[str, Any]) -> dict[str, Any]:
        return {
            "cycle_id": cycle_id,
            "generated_at": now_iso(),
            "selected_experiments": [item["id"] for item in selected],
            "completed_experiments": list(self.completed_experiments),
            "scientific_terminal_state": route_decision.get("scientific_terminal_state"),
            "route_next_branch": route_decision.get("route_next_branch"),
            "resource_snapshot": self._resource_snapshot(),
        }

    def _morning_memo(self, history: list[dict[str, Any]], route_decision: dict[str, Any]) -> str:
        rf5 = load_json(RF5_ARTIFACT, {})
        best_profile_id = rf5.get("best_profile_id")
        best_profile = rf5.get("best_profile") or {}
        g5_details = None
        if history:
            for gate in history[-1].get("gate_report", {}).get("gates", []):
                if gate.get("gate_id") == "G5_training_eligibility":
                    g5_details = gate
                    break
        lines = [
            "# Morning Memo",
            "",
            f"Generated: {now_iso()}",
            f"- scientific_terminal_state: {route_decision.get('scientific_terminal_state')}",
            f"- route_next_branch: {route_decision.get('route_next_branch')}",
            "",
            "## Answers",
            f"1. reformulated profile materially reduced same-problem mismatch? {'yes' if best_profile and (best_profile.get('sensor_contract_gain') or {}).get('score', 0.0) > 0.25 else 'no'}",
            f"2. any reformulated lane train-eligible? {'yes' if g5_details and g5_details.get('passed') else 'no'}",
            f"3. correct next branch: {route_decision.get('route_next_branch')}",
            f"4. why Stage B justified now? {'only for tiny retrain confirmation' if route_decision.get('route_next_branch') == ROUTE_OPTIONS['tiny_retrain'] else 'not justified on the legacy formulation'}",
            "",
            "## Best Reformulation Signal",
            f"- best_profile_id: {best_profile_id}",
            f"- rollout_ceiling_lift: {(best_profile.get('rollout_ceiling_lift') or {}).get('score')}",
            f"- state_alignment_gain: {best_profile.get('state_alignment_gain')}",
            f"- sensor_contract_gain: {(best_profile.get('sensor_contract_gain') or {}).get('score')}",
            "",
            "## Hypotheses",
        ]
        for hypothesis_id, state in self.board.items():
            lines.append(f"- {hypothesis_id}: posterior={float(state.get('posterior', 0.5)):.4f}")
        return "\n".join(lines).rstrip() + "\n"

    def run_cycle(self) -> dict[str, Any]:
        cycle_id = self._make_cycle_id()
        cycle_dir = create_cycle_dir(cycle_id)
        selected = self.select_experiments()
        results: dict[str, Any] = {}
        updates: list[dict[str, Any]] = []
        lane_specs_payload: dict[str, Any] = {}
        if not selected:
            gate_report = self.evaluate_gates()
            route_decision = dict(gate_report.get("route_decision") or {
                "scientific_terminal_state": gate_report.get("scientific_terminal_state"),
                "route_next_branch": gate_report.get("route_next_branch"),
                "why": "No additional experiments were selected; using current route decision.",
            })
            cycle_summary = self._cycle_summary(cycle_id, selected, route_decision)
            proposed_current_truth_delta = self._proposed_truth_delta(gate_report, route_decision, selected)
            proposed_next_actions = self._proposed_next_actions(route_decision, selected)
            memo = self._cycle_memo(cycle_id, selected, gate_report, route_decision)
            write_cycle_bundle(
                cycle_dir,
                lane_specs=lane_specs_payload,
                lane_results=results,
                gate_report=gate_report,
                hypothesis_board=self.board,
                cycle_memo=memo,
                proposed_current_truth_delta=proposed_current_truth_delta,
                proposed_next_actions=proposed_next_actions,
                cycle_summary=cycle_summary,
                route_decision=route_decision,
            )
            append_cycle_record({
                "timestamp": now_iso(),
                "cycle_id": cycle_id,
                "selected_experiments": [],
                "scientific_terminal_state": route_decision.get("scientific_terminal_state"),
                "route_next_branch": route_decision.get("route_next_branch"),
            })
            return {
                "cycle_id": cycle_id,
                "selected_experiments": [],
                "gate_report": gate_report,
                "route_decision": route_decision,
                "scientific_terminal_state": route_decision.get("scientific_terminal_state"),
            }
        for item in selected:
            result = item["runner"]()
            results[item["id"]] = result["payload"]
            for spec in result.get("lane_specs", []):
                payload = dict(spec)
                payload["resource_budget_snapshot"] = self._resource_snapshot()
                lane_specs_payload[payload["lane_id"]] = payload
            updates.extend(result.get("updates", []))
            self.completed_experiments.append(item["id"])
            append_evidence_record(
                {
                    "timestamp": now_iso(),
                    "cycle_id": cycle_id,
                    "experiment_id": item["id"],
                    "artifact_path": str(self._artifact_paths(item["id"])[0]),
                    "summary": result["payload"].get("passed"),
                }
            )
        self.board = apply_updates(self.board, updates)
        write_hypothesis_board(self.board)
        gate_report = self.evaluate_gates()
        route_decision = dict(gate_report.get("route_decision") or {
            "scientific_terminal_state": gate_report.get("scientific_terminal_state"),
            "route_next_branch": gate_report.get("route_next_branch"),
            "why": gate_report.get("route_why"),
        })
        cycle_summary = self._cycle_summary(cycle_id, selected, route_decision)
        proposed_current_truth_delta = self._proposed_truth_delta(gate_report, route_decision, selected)
        proposed_next_actions = self._proposed_next_actions(route_decision, selected)
        memo = self._cycle_memo(cycle_id, selected, gate_report, route_decision)
        write_cycle_bundle(
            cycle_dir,
            lane_specs=lane_specs_payload,
            lane_results=results,
            gate_report=gate_report,
            hypothesis_board=self.board,
            cycle_memo=memo,
            proposed_current_truth_delta=proposed_current_truth_delta,
            proposed_next_actions=proposed_next_actions,
            cycle_summary=cycle_summary,
            route_decision=route_decision,
        )
        append_cycle_record(
            {
                "timestamp": now_iso(),
                "cycle_id": cycle_id,
                "selected_experiments": [item["id"] for item in selected],
                "scientific_terminal_state": route_decision.get("scientific_terminal_state"),
                "route_next_branch": route_decision.get("route_next_branch"),
            }
        )
        return {
            "cycle_id": cycle_id,
            "selected_experiments": [item["id"] for item in selected],
            "gate_report": gate_report,
            "route_decision": route_decision,
            "scientific_terminal_state": route_decision.get("scientific_terminal_state"),
        }


def run_controller(
    *,
    max_cycles: int = 1,
    max_experiments_per_cycle: int = 3,
    sleep_seconds: int = 0,
    dry_run: bool = False,
    auto_promote_sovereign: bool = False,
    allow_full_retrain: bool = False,
    allow_new_claim: bool = False,
    run_mode: str = "autonomous_cycle_phase",
    max_rollouts_per_experiment: int = 3,
    max_disk_growth_mb: int = 4096,
    retry_backoff_seconds: int = 5,
    max_retries: int = 2,
    max_cycles_per_run: int = 4,
    max_tiny_retrain_budget: int = 1,
) -> dict[str, Any]:
    identity = controller_identity(owner="mint_v2_root_cause_controller")
    acquired, _lease = acquire_controller_lease(identity["controller_id"], identity["run_id"], owner=identity["owner"], pid=identity["pid"])
    if not acquired:
        raise RuntimeError("Another controller lease is currently active.")
    controller = RootCauseController(
        max_experiments_per_cycle=max_experiments_per_cycle,
        dry_run=dry_run,
        auto_promote_sovereign=auto_promote_sovereign,
        allow_full_retrain=allow_full_retrain,
        allow_new_claim=allow_new_claim,
        run_mode=run_mode,
        max_rollouts_per_experiment=max_rollouts_per_experiment,
        max_disk_growth_mb=max_disk_growth_mb,
        retry_backoff_seconds=retry_backoff_seconds,
        max_retries=max_retries,
        max_cycles_per_run=max_cycles_per_run,
        max_tiny_retrain_budget=max_tiny_retrain_budget,
    )
    history: list[dict[str, Any]] = []
    final_route_decision = {"scientific_terminal_state": None, "route_next_branch": ROUTE_OPTIONS["stay"], "why": "Run not started."}
    try:
        effective_cycles = min(int(max_cycles), int(max_cycles_per_run))
        for _ in range(effective_cycles):
            refresh_controller_lease(identity["controller_id"], identity["run_id"], pid=identity["pid"])
            result = controller.run_cycle()
            history.append(result)
            final_route_decision = result.get("route_decision", final_route_decision)
            if controller._disk_growth_mb() > float(max_disk_growth_mb):
                final_route_decision = {
                    "scientific_terminal_state": final_route_decision.get("scientific_terminal_state"),
                    "route_next_branch": ROUTE_OPTIONS["reformulation"],
                    "why": f"Stopped because disk growth exceeded budget ({controller._disk_growth_mb():.2f} MB > {max_disk_growth_mb} MB).",
                }
                break
            if result.get("scientific_terminal_state"):
                break
            if sleep_seconds > 0:
                time.sleep(float(sleep_seconds))
                refresh_controller_lease(identity["controller_id"], identity["run_id"], pid=identity["pid"])
    finally:
        release_controller_lease(identity["controller_id"], identity["run_id"])
    morning_memo = controller._morning_memo(history, final_route_decision)
    final_summary = {
        "controller_id": identity["controller_id"],
        "run_id": identity["run_id"],
        "history": history,
        "finished_at": now_iso(),
        "resource_snapshot": controller._resource_snapshot(),
    }
    write_final_run_outputs(route_decision=final_route_decision, morning_memo=morning_memo, final_summary=final_summary)
    return {
        **final_summary,
        "scientific_terminal_state": final_route_decision.get("scientific_terminal_state"),
        "route_next_branch": final_route_decision.get("route_next_branch"),
    }

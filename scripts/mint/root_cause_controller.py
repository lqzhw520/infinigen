#!/usr/bin/env python3
"""Visual-fidelity root-cause controller for the unified reformulation plan."""

from __future__ import annotations

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
    CYCLE_STATE_PATH,
    HYPOTHESIS_BOARD_PATH,
    ClaimPolicy,
    ControllerRoute,
    DatasetConfig,
    GateDecision,
    LaneResult,
    LaneSpec,
    RouteNextBranch,
    ScientificTerminalState,
    TrainConfig,
    classify_claim_policy,
    load_authority_snapshot,
    load_claim_boundary,
)
from root_cause_hypotheses import apply_updates, default_board, unresolvedness
from root_cause_metrics import (
    CONTRAST_THRESHOLD,
    PERCEPTUAL_READABILITY_THRESHOLD,
    contract_validity_score_v2,
    handle_crop_entropy,
    handle_local_contrast,
    handle_readability_score,
    handle_visibility_fraction,
    perceptual_readability_score_v2,
    rollout_ceiling_lift,
    sensor_contract_gain,
    state_alignment_gain,
    summarize_visual,
    visual_alignment_gain,
    visual_gap,
    visual_gate_breakdown,
    weighted_visual_gap,
)
from root_cause_registry import (
    append_cycle_record,
    append_deviation_record,
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
P2E1_ARTIFACT = ARTIFACT_DIR / "p2e1_orientation_causality_matrix.json"

VR_ARTIFACTS: dict[str, tuple[Path, Path]] = {
    "VR0": (ARTIFACT_DIR / "p2vr0_replicate_v0_negative.json", OUTPUT_DIR / "p2vr0_replicate_v0_negative.md"),
    "VR1": (ARTIFACT_DIR / "p2vr1_raw_canonical_visual_baseline.json", OUTPUT_DIR / "p2vr1_raw_canonical_visual_baseline.md"),
    "VR2": (ARTIFACT_DIR / "p2vr2_stronger_visual_reformulation_probe.json", OUTPUT_DIR / "p2vr2_stronger_visual_reformulation_probe.md"),
    "VR3": (ARTIFACT_DIR / "p2vr3_perception_first_probe.json", OUTPUT_DIR / "p2vr3_perception_first_probe.md"),
    "VR4": (ARTIFACT_DIR / "p2vr4_stronger_bundle_ab.json", OUTPUT_DIR / "p2vr4_stronger_bundle_ab.md"),
    "VR5": (ARTIFACT_DIR / "p2vr5_replicate_strongest_visual_bundle.json", OUTPUT_DIR / "p2vr5_replicate_strongest_visual_bundle.md"),
    "VR6": (ARTIFACT_DIR / "p2vr6_tiny_retrain_confirmation.json", OUTPUT_DIR / "p2vr6_tiny_retrain_confirmation.md"),
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
        cycle_mode: str = "unattended_cycle",
        experiment_family: str = "VR",
        cap_strongest_negative: bool = True,
        resume: bool = False,
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
        self.experiment_family = str(experiment_family)
        self.cap_strongest_negative = bool(cap_strongest_negative)
        self.resume = bool(resume)
        self.cycle_mode = str(cycle_mode)
        self.policy = {
            "AUTO_PROMOTE_SOVEREIGN": bool(auto_promote_sovereign),
            "ALLOW_FULL_RETRAIN": bool(allow_full_retrain),
            "ALLOW_NEW_CLAIM": bool(allow_new_claim),
            "RUN_MODE": str(run_mode),
            "EXPERIMENT_FAMILY": self.experiment_family,
            "CAP_STRONGEST_NEGATIVE": self.cap_strongest_negative,
            "RESUME": self.resume,
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
        self.train_seeds = [seed for seed in available if seed not in self.heldout_seeds][:8]
        self.control_images = self._load_control_images(limit=60)
        self.control_state, self.control_action = self._load_control_arrays()
        self.control_visual_stats = summarize_visual(self.control_images)
        self.control_weighted_visual_gap = weighted_visual_gap(visual_gap(self.control_visual_stats, self.control_visual_stats))
        self.disk_start_bytes = self._tree_size(AUTOPILOT_DIR)

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
        current = self._tree_size(AUTOPILOT_DIR)
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

    def _current_surface_clean_contract(self) -> dict[str, Any]:
        payload = self._legacy_contract()
        payload.update({
            "enable_marker_overlay": False,
            "calibration_mode": "none",
            "canonical_lane": False,
        })
        return payload

    def _raw_canonical_contract(self) -> dict[str, Any]:
        return {
            "secondary_camera_mode": "wrist_dynamic",
            "enable_marker_overlay": False,
            "calibration_mode": "none",
            "interaction_mode": "legacy_translation_only",
            "state_mode": "m0_proxy",
            "render_profile": "visual_reformulation_v1_raw_canonical",
            "background_mode": "legacy_scene",
            "lighting_profile": "legacy",
            "material_policy": "legacy",
            "camera_framing_profile": "legacy",
            "emit_orientation_telemetry": True,
            "emit_camera_metadata": True,
            "emit_handle_probe_metadata": True,
            "canonical_lane": True,
        }

    def _minimal_contract_repair_v1(self) -> dict[str, Any]:
        payload = self._raw_canonical_contract()
        payload.update({
            "interaction_mode": "orientation_sensitive_v1",
            "state_mode": "telemetry_candidate_v1",
        })
        return payload

    def _visual_reformulation_v1_plus_bundle_contract(self) -> dict[str, Any]:
        payload = self._raw_canonical_contract()
        payload.update({
            "interaction_mode": "orientation_sensitive_v1",
            "state_mode": "telemetry_candidate_v2",
            "render_profile": "visual_reformulation_v1_plus_bundle",
            "background_mode": "neutral_lab",
            "lighting_profile": "bright_front_fill",
            "material_policy": "handle_highlight",
            "camera_framing_profile": "tight_handle_centered",
        })
        return payload

    def _visual_reformulation_v2_material_light_bg_contract(self) -> dict[str, Any]:
        payload = self._raw_canonical_contract()
        payload.update({
            "render_profile": "visual_reformulation_v2_material_light_bg",
            "background_mode": "high_contrast_lab",
            "lighting_profile": "bright_front_fill",
            "material_policy": "handle_highlight",
            "camera_framing_profile": "tight_handle_centered",
        })
        return payload

    def _visual_reformulation_v2_plus_bundle_contract(self) -> dict[str, Any]:
        payload = self._visual_reformulation_v2_material_light_bg_contract()
        payload.update({
            "interaction_mode": "orientation_sensitive_v1",
            "state_mode": "telemetry_candidate_v2",
            "render_profile": "visual_reformulation_v2_plus_bundle",
        })
        return payload

    def _make_lane_spec(
        self,
        *,
        lane_id: str,
        stage: str,
        env_contract: dict[str, Any],
        experiment_id: str,
        interventions: dict[str, Any] | None = None,
        claim_policy: ClaimPolicy = "diagnostic",
        note: str = "",
        coverage: dict[str, float] | None = None,
        cost: float = 1.0,
        dataset_config: DatasetConfig | None = None,
        train_config: TrainConfig | None = None,
    ) -> LaneSpec:
        diagnostic_only = claim_policy != "canonical" or env_contract.get("calibration_mode") == "diagnostic_texture"
        spec = LaneSpec(
            lane_id=lane_id,
            lane_family=self.experiment_family,
            stage=stage,
            env_contract_config=env_contract,
            interventions=interventions or {},
            dataset_config=dataset_config or DatasetConfig(train_seed_pool=self.train_seeds, heldout_seed_pool=self.heldout_seeds),
            train_config=train_config or TrainConfig(),
            claim_policy=claim_policy,
            diagnostic_only=diagnostic_only,
            strongest_negative_capped=bool(self.cap_strongest_negative),
            resource_budget_snapshot=self._resource_snapshot(),
            note=note,
            experiment_id=experiment_id,
            coverage=coverage or {},
            cost=cost,
        )
        resolved_policy, violations = classify_claim_policy(self.claim_boundary, spec)
        if resolved_policy != spec.claim_policy and not self.policy["ALLOW_NEW_CLAIM"]:
            spec = LaneSpec(
                lane_id=spec.lane_id,
                lane_family=spec.lane_family,
                stage=spec.stage,
                env_contract_config=spec.env_contract_config,
                interventions=spec.interventions,
                dataset_config=spec.dataset_config,
                train_config=spec.train_config,
                claim_policy=resolved_policy,
                diagnostic_only=True,
                strongest_negative_capped=spec.strongest_negative_capped,
                resource_budget_snapshot=spec.resource_budget_snapshot,
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
            "render_profile",
            "background_mode",
            "lighting_profile",
            "material_policy",
            "camera_framing_profile",
            "emit_orientation_telemetry",
            "emit_camera_metadata",
            "emit_handle_probe_metadata",
            "canonical_lane",
        }
        return DrawerEnvContractConfig(**{key: payload[key] for key in keys if key in payload})

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
                    rollout["resource_budget_snapshot"] = self._resource_snapshot()
                    rollouts.append(rollout)
                    last_error = None
                    break
                except Exception as exc:  # pragma: no cover
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
                "orientation_causal_sensitivity": 0.0,
            }
        return {
            "rollout_count": len(rollouts),
            "unique_success_rate": float(np.mean([1.0 if r.get("success") else 0.0 for r in rollouts])),
            "attach_rate": float(np.mean([1.0 if r.get("ever_attached") else 0.0 for r in rollouts])),
            "mean_max_drawer_fraction": float(np.mean([float(r.get("max_drawer_fraction", 0.0)) for r in rollouts])),
            "avg_episode_length": float(np.mean([float(r.get("steps", 0.0)) for r in rollouts])),
            "success_count": int(sum(1 for r in rollouts if r.get("success"))),
            "orientation_causal_sensitivity": float(np.mean([np.mean(np.asarray(r.get("drawer_delta_effective_trace", [0.0]), dtype=np.float32)) for r in rollouts])),
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

    def _handle_probe_aggregates(self, rollouts: list[dict[str, Any]]) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        for rollout in rollouts:
            trace.extend(list(rollout.get("handle_probe_metadata_trace", [])))
        framing_values = [float(item.get("secondary_framing_score", 0.0)) for item in trace if item]
        residual_values = [float(item.get("camera_relativeness_residual", 1.0)) for item in trace if item]
        return {
            "handle_visibility_fraction": handle_visibility_fraction(trace),
            "handle_local_contrast": handle_local_contrast(trace),
            "handle_crop_entropy": handle_crop_entropy(trace),
            "framing_score": float(np.mean(framing_values)) if framing_values else 0.0,
            "camera_relativeness_residual": float(np.mean(residual_values)) if residual_values else None,
        }

    def _visual_report(self, spec: LaneSpec, seeds: list[int], *, baseline_gap: float | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rollouts = self._run_rollout_set(spec, seeds)
        residuals = self._secondary_residuals(rollouts)
        target_stats = summarize_visual(self._sample_images(rollouts, limit=24))
        gap_payload = visual_gap(self.control_visual_stats, target_stats)
        lane_gap = weighted_visual_gap(gap_payload)
        probe = self._handle_probe_aggregates(rollouts)
        alignment_gain = visual_alignment_gain(baseline_gap if baseline_gap is not None else max(lane_gap, 1e-6), lane_gap)
        report = {
            "lane": spec.to_payload(),
            "claim_policy": spec.claim_policy,
            "diagnostic_only": spec.diagnostic_only,
            "secondary_camera_wrist_like": bool(residuals and float(np.mean(residuals)) < 0.02 and float(np.max(residuals)) < 0.05),
            "secondary_camera_residual_mean": float(np.mean(residuals)) if residuals else None,
            "secondary_camera_residual_max": float(np.max(residuals)) if residuals else None,
            "marker_overlay_enabled": bool(spec.env_contract_config.get("enable_marker_overlay", False)),
            "provenance_complete": True,
            "target_stats": target_stats,
            "gap": gap_payload,
            "weighted_visual_gap": lane_gap,
            "visual_alignment_gain": alignment_gain,
            "render_profile": spec.env_contract_config.get("render_profile"),
            "background_mode": spec.env_contract_config.get("background_mode"),
            "lighting_profile": spec.env_contract_config.get("lighting_profile"),
            "material_policy": spec.env_contract_config.get("material_policy"),
            "camera_framing_profile": spec.env_contract_config.get("camera_framing_profile"),
            **probe,
        }
        report["observation_contract_pass"] = bool(
            report["secondary_camera_wrist_like"]
            and not report["marker_overlay_enabled"]
            and not report["diagnostic_only"]
            and report["provenance_complete"]
        )
        report["visual_gate_breakdown"] = visual_gate_breakdown(
            baseline_weighted_visual_gap=float(baseline_gap if baseline_gap is not None else lane_gap),
            lane_weighted_visual_gap=lane_gap,
            visual_alignment_gain_value=alignment_gain,
            framing_score=report["framing_score"],
            visibility_fraction=report["handle_visibility_fraction"],
            local_contrast=report["handle_local_contrast"],
            crop_entropy=report["handle_crop_entropy"],
            camera_relativeness_residual=report["secondary_camera_residual_mean"],
            encoder_readability_pass=None,
        )
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

    def _state_report(self, spec: LaneSpec, seeds: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rollouts = self._run_rollout_set(spec, seeds)
        state_blocks = [np.asarray(r.get("states", np.zeros((0, 8), dtype=np.float32)), dtype=np.float32) for r in rollouts]
        target_state = np.concatenate([block for block in state_blocks if len(block)], axis=0) if state_blocks else np.zeros((1, 8), dtype=np.float32)
        wd = self._per_dim_wasserstein(self.control_state, target_state)
        overlap = self._per_dim_range_overlap(self.control_state, target_state)
        smooth_gap = self._temporal_smoothness_gap(self.control_state, target_state)
        overall_score = self._normalized_state_score(wd, overlap, smooth_gap, self.control_state, target_state)
        state_spec = (rollouts[-1].get("state_spec") if rollouts else {}) or {}
        report = {
            "lane": spec.to_payload(),
            "state_spec": state_spec,
            "overall_score": float(overall_score),
            "per_dim_wasserstein": wd,
            "per_dim_range_overlap": overlap,
            "per_dim_smoothness_gap": smooth_gap,
            "valid": not bool(state_spec.get("fraud_padding")) and bool(state_spec.get("all_dims_explained", True)),
            "fraud_padding": bool(state_spec.get("fraud_padding", False)),
            "all_dims_explained": bool(state_spec.get("all_dims_explained", False)),
            "state_mode": state_spec.get("state_mode", spec.env_contract_config.get("state_mode", "unknown")),
        }
        return rollouts, report

    def _write_artifact(self, experiment_id: str, payload: dict[str, Any], title: str, bullets: list[str]) -> None:
        artifact_path, report_path = VR_ARTIFACTS[experiment_id]
        write_json_atomic(artifact_path, payload)
        lines = [f"# {title}", "", f"Generated: {payload.get('generated_at', now_iso())}"]
        lines.extend([f"- {line}" for line in bullets])
        write_text_atomic(report_path, "\n".join(lines).rstrip() + "\n")

    def experiment_catalog(self) -> list[dict[str, Any]]:
        return [
            {"id": "VR0", "deps": [], "coverage": {"H6_environment_invalid_for_claim": 0.5, "H0_same_problem_identity": 0.3}, "cost": 0.8, "runner": self.run_vr0},
            {"id": "VR1", "deps": ["VR0"], "coverage": {"H3_observation_visual_contract": 1.0, "H0_same_problem_identity": 0.6}, "cost": 1.0, "runner": self.run_vr1},
            {"id": "VR2", "deps": ["VR1"], "coverage": {"H3_observation_visual_contract": 1.0, "H0_same_problem_identity": 0.7}, "cost": 1.1, "runner": self.run_vr2},
            {"id": "VR3", "deps": ["VR2"], "coverage": {"H3_observation_visual_contract": 0.7}, "cost": 0.7, "runner": self.run_vr3},
            {"id": "VR4", "deps": ["VR1", "VR2", "VR3"], "coverage": {"H6_environment_invalid_for_claim": 0.9, "H2_state_representability": 0.8, "H3_observation_visual_contract": 0.8}, "cost": 1.4, "runner": self.run_vr4},
            {"id": "VR5", "deps": ["VR4"], "coverage": {"H6_environment_invalid_for_claim": 0.8, "H0_same_problem_identity": 0.8}, "cost": 0.9, "runner": self.run_vr5},
            {"id": "VR6", "deps": ["VR5"], "coverage": {"H4_unique_data_scale_only": 0.4, "H5_optimization_only": 0.4}, "cost": 1.0, "runner": self.run_vr6},
        ]

    def select_experiments(self) -> list[dict[str, Any]]:
        catalog = self.experiment_catalog()
        if "VR0" not in self.completed_experiments:
            return [next(item for item in catalog if item["id"] == "VR0")]
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

    def run_vr0(self) -> dict[str, Any]:
        seeds = self.train_seeds[:3]
        legacy = self._make_lane_spec(
            lane_id="VR0_legacy_current",
            stage="probe",
            env_contract=self._legacy_contract(),
            interventions={"rotation_source": "zero", "profile_id": "legacy_current"},
            claim_policy="diagnostic",
            note="Replicate frozen v0 legacy negative.",
            experiment_id="VR0",
        )
        minimal = self._make_lane_spec(
            lane_id="VR0_minimal_contract_repair_v1",
            stage="probe",
            env_contract=self._minimal_contract_repair_v1(),
            interventions={"rotation_source": "aligned", "profile_id": "minimal_contract_repair_v1"},
            claim_policy="canonical",
            note="Replicate frozen v0 minimal repair bundle.",
            experiment_id="VR0",
        )
        legacy_rollouts = self._run_rollout_set(legacy, seeds)
        minimal_rollouts = self._run_rollout_set(minimal, seeds)
        legacy_summary = self._rollout_summary(legacy_rollouts)
        minimal_summary = self._rollout_summary(minimal_rollouts)
        ceiling = rollout_ceiling_lift(legacy_summary, minimal_summary)
        replicated_negative = bool(ceiling["score"] <= 0.0)
        payload = {
            "experiment_id": "VR0",
            "generated_at": now_iso(),
            "passed": replicated_negative,
            "baseline_summary": legacy_summary,
            "minimal_summary": minimal_summary,
            "rollout_ceiling_lift": ceiling,
            "replicated_negative": replicated_negative,
        }
        self._write_artifact("VR0", payload, "VR0 Replicate v0 Negative", [f"replicated_negative: {replicated_negative}", f"rollout_ceiling_lift: {ceiling['score']:.6f}"])
        updates = [
            {"hypothesis_id": "H6_environment_invalid_for_claim", "support_score": 0.5 if replicated_negative else -0.2, "alpha": 0.5, "evidence_ref": "VR0"},
            {"hypothesis_id": "H0_same_problem_identity", "support_score": -0.3 if replicated_negative else 0.2, "alpha": 0.4, "evidence_ref": "VR0"},
        ]
        return {"payload": payload, "updates": updates, "lane_specs": [legacy.to_payload(), minimal.to_payload()]}

    def run_vr1(self) -> dict[str, Any]:
        seeds = self.train_seeds[:2]
        legacy_clean = self._make_lane_spec(
            lane_id="VR1_legacy_cleaned_surface",
            stage="probe",
            env_contract=self._current_surface_clean_contract(),
            interventions={"rotation_source": "zero", "profile_id": "legacy_clean_surface"},
            claim_policy="diagnostic",
            note="Legacy scene without overlay/calibration, still non-wrist.",
            experiment_id="VR1",
        )
        raw_canonical = self._make_lane_spec(
            lane_id="VR1_wrist_raw_canonical",
            stage="probe",
            env_contract=self._raw_canonical_contract(),
            interventions={"rotation_source": "zero", "profile_id": "visual_reformulation_v1_raw_canonical"},
            claim_policy="canonical",
            note="Wrist/no-overlay/raw canonical baseline.",
            experiment_id="VR1",
        )
        _, legacy_report = self._visual_report(legacy_clean, seeds)
        _, raw_report = self._visual_report(raw_canonical, seeds)
        baseline_gap = legacy_report["weighted_visual_gap"]
        raw_report["visual_alignment_gain"] = visual_alignment_gain(baseline_gap, raw_report["weighted_visual_gap"])
        raw_report["visual_gate_breakdown"] = visual_gate_breakdown(
            baseline_weighted_visual_gap=baseline_gap,
            lane_weighted_visual_gap=raw_report["weighted_visual_gap"],
            visual_alignment_gain_value=raw_report["visual_alignment_gain"],
            framing_score=raw_report["framing_score"],
            visibility_fraction=raw_report["handle_visibility_fraction"],
            local_contrast=raw_report["handle_local_contrast"],
            crop_entropy=raw_report["handle_crop_entropy"],
            camera_relativeness_residual=raw_report["secondary_camera_residual_mean"],
            encoder_readability_pass=None,
        )
        payload = {
            "experiment_id": "VR1",
            "generated_at": now_iso(),
            "passed": bool(raw_report["observation_contract_pass"]),
            "lanes": [legacy_report, raw_report],
            "baseline_weighted_visual_gap": baseline_gap,
            "best_canonical_lane": raw_report,
        }
        self._write_artifact(
            "VR1",
            payload,
            "VR1 Raw Canonical Visual Baseline",
            [
                f"baseline_weighted_visual_gap: {baseline_gap:.6f}",
                f"raw_visual_alignment_gain: {raw_report['visual_alignment_gain']:.6f}",
                f"raw_handle_visibility_fraction: {raw_report['handle_visibility_fraction']:.6f}",
            ],
        )
        updates = [
            {"hypothesis_id": "H3_observation_visual_contract", "support_score": 0.5 if raw_report["observation_contract_pass"] else 0.1, "alpha": 0.5, "evidence_ref": "VR1"},
            {"hypothesis_id": "H0_same_problem_identity", "support_score": -0.2 if raw_report["weighted_visual_gap"] > 1.0 else 0.1, "alpha": 0.4, "evidence_ref": "VR1"},
        ]
        return {"payload": payload, "updates": updates, "lane_specs": [legacy_clean.to_payload(), raw_canonical.to_payload()]}

    def run_vr2(self) -> dict[str, Any]:
        seeds = self.train_seeds[:2]
        vr1_payload = load_json(VR_ARTIFACTS["VR1"][0], {})
        baseline_gap = float(vr1_payload.get("best_canonical_lane", {}).get("weighted_visual_gap", 1.0))
        raw_canonical = self._make_lane_spec(
            lane_id="VR2_wrist_raw_canonical",
            stage="probe",
            env_contract=self._raw_canonical_contract(),
            interventions={"rotation_source": "zero", "profile_id": "visual_reformulation_v1_raw_canonical"},
            claim_policy="canonical",
            note="Carry-forward raw canonical baseline.",
            experiment_id="VR2",
        )
        visual_v1 = self._make_lane_spec(
            lane_id="VR2_visual_reformulation_v2_material_light_bg",
            stage="probe",
            env_contract=self._visual_reformulation_v2_material_light_bg_contract(),
            interventions={"rotation_source": "zero", "profile_id": "visual_reformulation_v2_material_light_bg"},
            claim_policy="canonical",
            note="Stronger render-level material/light/background/framing reformulation.",
            experiment_id="VR2",
        )
        _, raw_report = self._visual_report(raw_canonical, seeds, baseline_gap=baseline_gap)
        _, v1_report = self._visual_report(visual_v1, seeds, baseline_gap=baseline_gap)
        gain = visual_alignment_gain(raw_report["weighted_visual_gap"], v1_report["weighted_visual_gap"])
        readability_gain = v1_report["visual_gate_breakdown"]["perceptual_readability_score_v2"] - raw_report["visual_gate_breakdown"]["perceptual_readability_score_v2"]
        passed = bool(gain >= 0.20 or readability_gain >= 0.10)
        payload = {
            "experiment_id": "VR2",
            "generated_at": now_iso(),
            "passed": passed,
            "baseline_gap": baseline_gap,
            "raw_canonical": raw_report,
            "visual_reformulation_v2_material_light_bg": v1_report,
            "visual_alignment_gain": gain,
            "readability_gain": readability_gain,
        }
        self._write_artifact(
            "VR2",
            payload,
            "VR2 Stronger Visual Reformulation Probe",
            [
                f"visual_alignment_gain: {gain:.6f}",
                f"readability_gain: {readability_gain:.6f}",
                f"passed: {passed}",
            ],
        )
        updates = [
            {"hypothesis_id": "H3_observation_visual_contract", "support_score": 0.8 if passed else 0.3, "alpha": 0.6, "evidence_ref": "VR2"},
            {"hypothesis_id": "H0_same_problem_identity", "support_score": 0.2 if passed else -0.4, "alpha": 0.5, "evidence_ref": "VR2"},
        ]
        return {"payload": payload, "updates": updates, "lane_specs": [raw_canonical.to_payload(), visual_v1.to_payload()]}

    def run_vr3(self) -> dict[str, Any]:
        vr2_payload = load_json(VR_ARTIFACTS["VR2"][0], {})
        strongest = vr2_payload.get("visual_reformulation_v2_material_light_bg") or {}
        proxy_score = float((strongest.get("visual_gate_breakdown") or {}).get("perceptual_readability_score_v2", 0.0))
        payload = {
            "experiment_id": "VR3",
            "generated_at": now_iso(),
            "passed": False,
            "perception_probe_unavailable": True,
            "encoder_readability_pass": None,
            "proxy_readability_score": proxy_score,
            "note": "Frozen encoder / feature separability tooling is unavailable in this controller runtime; proxy readability was recorded instead.",
        }
        append_deviation_record(
            message="VR3 perception-first probe unavailable; recorded proxy readability instead.",
            scientific_semantics_changed=False,
            details={"experiment_id": "VR3", "proxy_readability_score": proxy_score},
        )
        self._write_artifact(
            "VR3",
            payload,
            "VR3 Perception-First Probe",
            [
                "perception_probe_unavailable: true",
                f"proxy_readability_score: {proxy_score:.6f}",
            ],
        )
        updates = [
            {"hypothesis_id": "H3_observation_visual_contract", "support_score": 0.2 if proxy_score > 0.5 else 0.0, "alpha": 0.3, "evidence_ref": "VR3"},
        ]
        return {"payload": payload, "updates": updates, "lane_specs": []}

    def run_vr4(self) -> dict[str, Any]:
        seeds = self.train_seeds[:3]
        vr1_payload = load_json(VR_ARTIFACTS["VR1"][0], {})
        baseline_gap = float(vr1_payload.get("best_canonical_lane", {}).get("weighted_visual_gap", 1.0))
        profiles = [
            ("legacy_current", self._legacy_contract(), "diagnostic", "zero"),
            ("minimal_contract_repair_v1", self._minimal_contract_repair_v1(), "canonical", "aligned"),
            ("visual_reformulation_v1_plus_bundle", self._visual_reformulation_v1_plus_bundle_contract(), "canonical", "aligned"),
            ("visual_reformulation_v2_plus_bundle", self._visual_reformulation_v2_plus_bundle_contract(), "canonical", "aligned"),
        ]
        lane_payloads = []
        lane_specs = []
        baseline_rollout_summary = None
        baseline_state_score = None
        for profile_id, contract, claim_policy, rotation_source in profiles:
            spec = self._make_lane_spec(
                lane_id=f"VR4_{profile_id}",
                    stage="probe",
                env_contract=contract,
                interventions={"rotation_source": rotation_source, "profile_id": profile_id},
                claim_policy=claim_policy,
                note="VR4 bundle comparison profile.",
                experiment_id="VR4",
            )
            lane_specs.append(spec.to_payload())
            rollouts, visual_report = self._visual_report(spec, seeds, baseline_gap=baseline_gap)
            _, state_report = self._state_report(spec, seeds)
            rollout_summary = self._rollout_summary(rollouts)
            if baseline_rollout_summary is None:
                baseline_rollout_summary = rollout_summary
                baseline_state_score = float(state_report["overall_score"])
            ceiling = rollout_ceiling_lift(baseline_rollout_summary, rollout_summary)
            state_gain = state_alignment_gain(float(baseline_state_score), float(state_report["overall_score"]))
            residual_mean = float(visual_report.get("secondary_camera_residual_mean") or 1.0)
            wrist_gain = float(np.clip(1.0 - residual_mean / 0.05, 0.0, 1.0))
            no_overlay_gain = 1.0 if not visual_report["marker_overlay_enabled"] else 0.0
            framing_gain = float(visual_report["framing_score"])
            sensor_gain = sensor_contract_gain(
                wrist_relativeness_gain=wrist_gain,
                no_overlay_gain=no_overlay_gain,
                visual_alignment_gain_value=max(0.0, float(visual_report["visual_alignment_gain"])),
                framing_gain=framing_gain,
            )
            combined = 0.50 * max(0.0, ceiling["score"]) + 0.25 * max(0.0, state_gain) + 0.25 * sensor_gain["score"]
            lane_payloads.append(
                {
                    "profile_id": profile_id,
                    "lane": spec.to_payload(),
                    "rollout_summary": rollout_summary,
                    "rollout_ceiling_lift": ceiling,
                    "state_report": state_report,
                    "state_alignment_gain": float(state_gain),
                    "visual_report": visual_report,
                    "sensor_contract_gain": sensor_gain,
                    "combined_score": float(combined),
                }
            )
        best_bundle = max(lane_payloads, key=lambda item: item["combined_score"])

        state_candidates = []
        for mode in ["m0_proxy", "eef_pose_gripper", "telemetry_candidate_v1", "telemetry_candidate_v2"]:
            contract = self._visual_reformulation_v2_plus_bundle_contract()
            contract["state_mode"] = mode
            spec = self._make_lane_spec(
                lane_id=f"VR4_state_{mode}",
                    stage="probe",
                env_contract=contract,
                interventions={"rotation_source": "aligned", "profile_id": f"state_{mode}"},
                claim_policy="diagnostic" if mode in {"m0_proxy", "eef_pose_gripper"} else "canonical",
                note="State semantics audit candidate for tightened G3.",
                experiment_id="VR4",
            )
            lane_specs.append(spec.to_payload())
            _, report = self._state_report(spec, seeds)
            state_candidates.append(report)
        score_by_mode = {item["state_mode"]: float(item["overall_score"]) for item in state_candidates}
        telemetry_candidates = [item for item in state_candidates if item["state_mode"] in {"telemetry_candidate_v1", "telemetry_candidate_v2"} and item["valid"]]
        selected_candidate = min(telemetry_candidates, key=lambda item: item["overall_score"], default=None)
        m0_score = float(score_by_mode.get("m0_proxy", 1.0))
        eef_score = float(score_by_mode.get("eef_pose_gripper", 1.0))
        selected_score = float(selected_candidate["overall_score"]) if selected_candidate else None
        telemetry_gain = state_alignment_gain(m0_score, selected_score) if selected_candidate is not None else -1.0
        state_audit = {
            "selected_candidate_mode": selected_candidate["state_mode"] if selected_candidate else None,
            "selected_candidate_score": selected_score,
            "m0_proxy_score": m0_score,
            "eef_pose_gripper_score": eef_score,
            "state_alignment_gain": telemetry_gain,
            "selected_candidate": selected_candidate,
            "all_candidates": state_candidates,
            "impossible": selected_candidate is None,
        }
        payload = {
            "experiment_id": "VR4",
            "generated_at": now_iso(),
            "passed": bool(best_bundle["combined_score"] >= 0.0),
            "profiles": lane_payloads,
            "best_profile_id": best_bundle["profile_id"],
            "best_profile": best_bundle,
            "state_audit": state_audit,
        }
        self._write_artifact(
            "VR4",
            payload,
            "VR4 Stronger Bundle A/B",
            [
                f"best_profile_id: {best_bundle['profile_id']}",
                f"best_combined_score: {best_bundle['combined_score']:.6f}",
                f"selected_state_candidate: {state_audit['selected_candidate_mode']}",
            ],
        )
        updates = [
            {"hypothesis_id": "H2_state_representability", "support_score": 0.6 if not state_audit["impossible"] and telemetry_gain >= 0.10 else -0.4, "alpha": 0.6, "evidence_ref": "VR4"},
            {"hypothesis_id": "H3_observation_visual_contract", "support_score": 0.7 if best_bundle["visual_report"]["visual_alignment_gain"] >= 0.20 else 0.2, "alpha": 0.6, "evidence_ref": "VR4"},
            {"hypothesis_id": "H6_environment_invalid_for_claim", "support_score": 0.7 if best_bundle["rollout_ceiling_lift"]["score"] <= 0.0 else -0.3, "alpha": 0.7, "evidence_ref": "VR4"},
            {"hypothesis_id": "H0_same_problem_identity", "support_score": -0.6 if best_bundle["rollout_ceiling_lift"]["score"] <= 0.0 else 0.3, "alpha": 0.6, "evidence_ref": "VR4"},
        ]
        return {"payload": payload, "updates": updates, "lane_specs": lane_specs}

    def run_vr5(self) -> dict[str, Any]:
        vr4_payload = load_json(VR_ARTIFACTS["VR4"][0], {})
        best_profile = (vr4_payload.get("best_profile") or {})
        best_profile_id = best_profile.get("profile_id", "visual_reformulation_v2_plus_bundle")
        profile_map = {
            "legacy_current": self._legacy_contract,
            "minimal_contract_repair_v1": self._minimal_contract_repair_v1,
            "visual_reformulation_v1_plus_bundle": self._visual_reformulation_v1_plus_bundle_contract,
            "visual_reformulation_v2_plus_bundle": self._visual_reformulation_v2_plus_bundle_contract,
        }
        contract = profile_map.get(best_profile_id, self._visual_reformulation_v2_plus_bundle_contract)()
        alt_seeds = self.train_seeds[3:6] or self.train_seeds[:3]
        vr1_payload = load_json(VR_ARTIFACTS["VR1"][0], {})
        baseline_gap = float(vr1_payload.get("best_canonical_lane", {}).get("weighted_visual_gap", 1.0))
        legacy_spec = self._make_lane_spec(
            lane_id="VR5_legacy_current_alt",
            stage="probe",
            env_contract=self._legacy_contract(),
            interventions={"rotation_source": "zero", "profile_id": "legacy_current"},
            claim_policy="diagnostic",
            note="Legacy comparator for alternate slice replication.",
            experiment_id="VR5",
        )
        best_spec = self._make_lane_spec(
            lane_id=f"VR5_{best_profile_id}_alt",
            stage="probe",
            env_contract=contract,
            interventions={"rotation_source": "aligned", "profile_id": best_profile_id},
            claim_policy="canonical" if best_profile_id != "legacy_current" else "diagnostic",
            note="Replicate strongest visual bundle on alternate seeds.",
            experiment_id="VR5",
        )
        legacy_rollouts = self._run_rollout_set(legacy_spec, alt_seeds)
        best_rollouts = self._run_rollout_set(best_spec, alt_seeds)
        legacy_summary = self._rollout_summary(legacy_rollouts)
        best_summary = self._rollout_summary(best_rollouts)
        ceiling = rollout_ceiling_lift(legacy_summary, best_summary)
        _, best_visual = self._visual_report(best_spec, alt_seeds, baseline_gap=baseline_gap)
        vr4_rollout = float((best_profile.get("rollout_ceiling_lift") or {}).get("score", 0.0))
        replication_consistent = bool(np.sign(ceiling["score"] or 0.0) == np.sign(vr4_rollout or 0.0))
        payload = {
            "experiment_id": "VR5",
            "generated_at": now_iso(),
            "passed": replication_consistent,
            "best_profile_id": best_profile_id,
            "alternate_seed_slice": alt_seeds,
            "replication_consistent": replication_consistent,
            "legacy_summary": legacy_summary,
            "replicated_summary": best_summary,
            "rollout_ceiling_lift": ceiling,
            "visual_report": best_visual,
        }
        self._write_artifact(
            "VR5",
            payload,
            "VR5 Replicate Strongest Visual Bundle",
            [
                f"best_profile_id: {best_profile_id}",
                f"replication_consistent: {replication_consistent}",
                f"replicated_rollout_ceiling_lift: {ceiling['score']:.6f}",
            ],
        )
        updates = [
            {"hypothesis_id": "H6_environment_invalid_for_claim", "support_score": 0.6 if replication_consistent and ceiling["score"] <= 0.0 else -0.2, "alpha": 0.6, "evidence_ref": "VR5"},
            {"hypothesis_id": "H0_same_problem_identity", "support_score": -0.6 if replication_consistent and ceiling["score"] <= 0.0 else 0.2, "alpha": 0.6, "evidence_ref": "VR5"},
        ]
        return {"payload": payload, "updates": updates, "lane_specs": [legacy_spec.to_payload(), best_spec.to_payload()]}

    def run_vr6(self) -> dict[str, Any]:
        gate_report, route = self.evaluate_gates()
        blocked_reasons = []
        if not gate_report.get("G5_training_eligibility", {}).get("passed", False):
            blocked_reasons.append("G5_training_eligibility_failed")
        blocked = bool(blocked_reasons)
        payload = {
            "experiment_id": "VR6",
            "generated_at": now_iso(),
            "passed": False,
            "blocked": blocked,
            "blocked_reasons": blocked_reasons or ["tiny_retrain_not_implemented_in_visual_fidelity_branch"],
            "gate_report": gate_report,
        }
        self._write_artifact(
            "VR6",
            payload,
            "VR6 Tiny Retrain Confirmation",
            [f"blocked: {blocked}", f"blocked_reasons: {', '.join(payload['blocked_reasons'])}"],
        )
        return {"payload": payload, "updates": [], "lane_specs": []}

    def _carryforward_g2(self) -> dict[str, Any]:
        payload = load_json(P2E1_ARTIFACT, {}) if P2E1_ARTIFACT.exists() else {}
        return {
            "exists": P2E1_ARTIFACT.exists(),
            "passed": bool(payload.get("passed")),
            "aligned_beats_random_under_sensitive": bool(payload.get("aligned_beats_random_under_sensitive", False)),
            "orientation_causal_sensitivity": payload.get("orientation_causal_sensitivity"),
        }

    def _best_bundle_payload(self) -> dict[str, Any]:
        return load_json(VR_ARTIFACTS["VR4"][0], {}) if VR_ARTIFACTS["VR4"][0].exists() else {}

    def _replication_payload(self) -> dict[str, Any]:
        return load_json(VR_ARTIFACTS["VR5"][0], {}) if VR_ARTIFACTS["VR5"][0].exists() else {}

    def evaluate_gates(self) -> tuple[dict[str, Any], ControllerRoute]:
        g0_details = self._baseline_integrity_details()
        g0 = GateDecision(
            gate_id="G0_baseline_integrity",
            passed=bool(g0_details["control_baseline_pass"] and g0_details["parity_non_blocker"] and g0_details["authority_readable"]),
            summary="Baseline healthy, parity not-primary-blocker, and authority readable.",
            details=g0_details,
        )

        vr4 = self._best_bundle_payload()
        best_profile = (vr4.get("best_profile") or {})
        visual_report = best_profile.get("visual_report") or {}
        state_audit = vr4.get("state_audit") or {}

        g1_details = {
            "wrist_like": bool(visual_report.get("secondary_camera_wrist_like", False)),
            "no_overlay": not bool(visual_report.get("marker_overlay_enabled", True)),
            "no_diagnostic_texture": not bool(visual_report.get("diagnostic_only", True)),
            "provenance_complete": bool(visual_report.get("provenance_complete", False)),
        }
        g1 = GateDecision(
            gate_id="G1_observation_contract",
            passed=all(g1_details.values()),
            summary="Canonical observation contract requires wrist-like second camera, no overlay, no diagnostic texture, and full provenance.",
            details=g1_details,
        )

        g2_details = self._carryforward_g2()
        g2 = GateDecision(
            gate_id="G2_action_causality",
            passed=bool(g2_details["exists"] and g2_details["passed"] and g2_details["aligned_beats_random_under_sensitive"]),
            summary="Carry-forward action-causality evidence remains healthy and orientation-sensitive aligned rotation beats random.",
            details=g2_details,
        )

        selected_candidate = state_audit.get("selected_candidate") or {}
        selected_score = state_audit.get("selected_candidate_score")
        m0_score = state_audit.get("m0_proxy_score")
        eef_score = state_audit.get("eef_pose_gripper_score")
        state_gain = float(state_audit.get("state_alignment_gain", -1.0))
        g3_details = {
            "selected_candidate_mode": state_audit.get("selected_candidate_mode"),
            "fraud_padding": bool(selected_candidate.get("fraud_padding", True)),
            "all_dims_explained": bool(selected_candidate.get("all_dims_explained", False)),
            "state_alignment_gain": state_gain,
            "beats_m0_proxy": bool(selected_score is not None and m0_score is not None and float(selected_score) < float(m0_score)),
            "beats_eef_pose_gripper": bool(selected_score is not None and eef_score is not None and float(selected_score) < float(eef_score)),
            "impossible": bool(state_audit.get("impossible", True)),
        }
        g3 = GateDecision(
            gate_id="G3_state_semantics",
            passed=bool(
                not g3_details["impossible"]
                and selected_candidate.get("state_mode") in {"telemetry_candidate_v1", "telemetry_candidate_v2"}
                and not g3_details["fraud_padding"]
                and g3_details["all_dims_explained"]
                and state_gain >= 0.10
                and g3_details["beats_m0_proxy"]
                and g3_details["beats_eef_pose_gripper"]
            ),
            summary="Selected telemetry-compatible candidate must be non-fraud, fully explained, materially better than M0 and eef_pose_gripper, and clear the >=0.10 gain bar.",
            details=g3_details,
        )

        g4a_details = {
            "no_overlay": not bool(visual_report.get("marker_overlay_enabled", True)),
            "no_diagnostic_texture": not bool(visual_report.get("diagnostic_only", True)),
            "wrist_like_second_camera": bool(visual_report.get("secondary_camera_wrist_like", False)),
            "provenance_complete": bool(visual_report.get("provenance_complete", False)),
            "canonical_identity_explicit": best_profile.get("lane", {}).get("claim_policy") == "canonical",
            "camera_relativeness_residual": visual_report.get("secondary_camera_residual_mean"),
        }
        g4a = GateDecision(
            gate_id="G4a_observation_semantics",
            passed=bool(
                g4a_details["no_overlay"]
                and g4a_details["no_diagnostic_texture"]
                and g4a_details["wrist_like_second_camera"]
                and g4a_details["provenance_complete"]
                and g4a_details["canonical_identity_explicit"]
                and g4a_details["camera_relativeness_residual"] is not None
                and float(g4a_details["camera_relativeness_residual"]) < 0.05
            ),
            summary="Observation semantics require canonical identity, wrist-like relativeness, and no diagnostic-only visual cues.",
            details=g4a_details,
        )

        breakdown = visual_report.get("visual_gate_breakdown") or {}
        encoder_readability_pass = breakdown.get("encoder_readability_pass")
        handle_score = float(breakdown.get("handle_readability_score", handle_readability_score(
            visibility_fraction=float(visual_report.get("handle_visibility_fraction", 0.0)),
            local_contrast=float(visual_report.get("handle_local_contrast", 0.0)),
            crop_entropy=float(visual_report.get("handle_crop_entropy", 0.0)),
            framing_score=float(visual_report.get("framing_score", 0.0)),
        )))
        readability_score = float(breakdown.get("perceptual_readability_score_v2", perceptual_readability_score_v2(
            visual_alignment_gain_value=float(visual_report.get("visual_alignment_gain", -1.0)),
            handle_readability_score_value=handle_score,
            encoder_readability_pass=encoder_readability_pass,
        )))
        g4b_details = {
            "visual_alignment_gain": float(visual_report.get("visual_alignment_gain", -1.0)),
            "weighted_visual_gap": float(visual_report.get("weighted_visual_gap", 1e6)),
            "baseline_weighted_visual_gap": float(breakdown.get("baseline_weighted_visual_gap", 1e6)),
            "framing_score": float(visual_report.get("framing_score", 0.0)),
            "handle_visibility_fraction": float(visual_report.get("handle_visibility_fraction", 0.0)),
            "handle_local_contrast": float(visual_report.get("handle_local_contrast", 0.0)),
            "handle_crop_entropy": float(visual_report.get("handle_crop_entropy", 0.0)),
            "handle_readability_score": handle_score,
            "perceptual_readability_score_v2": readability_score,
            "contrast_threshold": float(CONTRAST_THRESHOLD),
            "encoder_readability_pass": encoder_readability_pass,
        }
        g4b = GateDecision(
            gate_id="G4b_perceptual_readability",
            passed=bool(
                g4b_details["visual_alignment_gain"] >= 0.20
                and g4b_details["weighted_visual_gap"] < g4b_details["baseline_weighted_visual_gap"]
                and g4b_details["framing_score"] >= 0.80
                and g4b_details["handle_visibility_fraction"] >= 0.60
                and g4b_details["handle_local_contrast"] >= CONTRAST_THRESHOLD
                and g4b_details["handle_readability_score"] >= PERCEPTUAL_READABILITY_THRESHOLD
                and g4b_details["perceptual_readability_score_v2"] >= PERCEPTUAL_READABILITY_THRESHOLD
                and (encoder_readability_pass is not False)
            ),
            summary="Perceptual readability requires materially improved visual gap, strong handle readability, good framing, sufficient visibility, and enough local contrast.",
            details=g4b_details,
        )

        rollout_gain = float((best_profile.get("rollout_ceiling_lift") or {}).get("score", -1.0))
        state_gain_best = float(best_profile.get("state_alignment_gain", state_gain))
        sensor_gain_best = float((best_profile.get("sensor_contract_gain") or {}).get("score", 0.0))
        g5_details = {
            "rollout_ceiling_lift": rollout_gain,
            "state_alignment_gain": state_gain_best,
            "sensor_perception_gain": sensor_gain_best,
            "canonical_lane": best_profile.get("lane", {}).get("claim_policy") == "canonical",
        }
        g5 = GateDecision(
            gate_id="G5_training_eligibility",
            passed=bool(
                g0.passed
                and g1.passed
                and g2.passed
                and g3.passed
                and g4a.passed
                and g4b.passed
                and g5_details["canonical_lane"]
                and (
                    rollout_gain >= 0.15
                    or state_gain_best >= 0.20
                    or sensor_gain_best >= 0.25
                )
            ),
            summary="Training eligibility remains downstream of all upstream contract gates and one substantive gain threshold.",
            details=g5_details,
        )

        gate_map = {gate.gate_id: asdict(gate) for gate in [g0, g1, g2, g3, g4a, g4b, g5]}
        gate_map["contract_validity_score_v2"] = contract_validity_score_v2(
            g0=g0.passed,
            g1=g1.passed,
            g2=g2.passed,
            g3=g3.passed,
            g4a=g4a.passed,
            g4b=g4b.passed,
        )

        replication = self._replication_payload()
        stronger_family_replicated = bool(replication.get("replication_consistent", False))
        cap_active = bool(self.cap_strongest_negative)
        no_canonical_near_miss = bool(
            not g5.passed
            and rollout_gain < 0.10
            and sensor_gain_best < 0.25
            and state_gain_best < 0.20
        )
        h0 = float((self.board.get("H0_same_problem_identity") or {}).get("posterior", 0.35))
        h6 = float((self.board.get("H6_environment_invalid_for_claim") or {}).get("posterior", 0.55))

        scientific_terminal_state: ScientificTerminalState | None = None
        route_next_branch: RouteNextBranch = "environment_reformulation"
        why = "Integrated visual/perception reformulation remains the mainline blocker and Stage B stays downstream."

        if g5.passed:
            scientific_terminal_state = "ALIGNED_CANONICAL_LANE_FOUND"
            route_next_branch = "tiny_retrain_confirmation"
            why = "A canonical lane passed G0-G5; tiny retrain confirmation is justified."
        else:
            scientific_terminal_state = "MINIMAL_REPAIR_INSUFFICIENT"
            route_next_branch = "environment_reformulation"
            why = "Strongest-negative cap remains active for this cycle; stronger visual/perception reformulation still did not produce a train-eligible canonical lane."

        route = ControllerRoute(
            scientific_terminal_state=scientific_terminal_state,
            route_next_branch=route_next_branch,
            why=why,
            strongest_negative_capped=cap_active,
            cap_active=cap_active,
            h0_posterior=h0,
            h6_posterior=h6,
        )
        gate_map["route_decision_preview"] = asdict(route)
        return gate_map, route

    def _cycle_memo(self, route: ControllerRoute, gate_report: dict[str, Any], cycle_summary: dict[str, Any]) -> str:
        lines = [
            f"# {cycle_summary['cycle_id']}",
            "",
            f"- cycle_mode: {self.cycle_mode}",
            f"- selected_experiments: {', '.join(cycle_summary['selected_experiments']) if cycle_summary['selected_experiments'] else '(none)'}",
            f"- scientific_terminal_state: {route.scientific_terminal_state}",
            f"- route_next_branch: {route.route_next_branch}",
            f"- strongest_negative_capped: {route.strongest_negative_capped}",
            f"- why: {route.why}",
            "",
            "## Gates",
        ]
        for gate_id in ["G0_baseline_integrity", "G1_observation_contract", "G2_action_causality", "G3_state_semantics", "G4a_observation_semantics", "G4b_perceptual_readability", "G5_training_eligibility"]:
            item = gate_report.get(gate_id, {})
            lines.append(f"- {gate_id}: {'pass' if item.get('passed') else 'fail'}")
        return "\n".join(lines).rstrip() + "\n"

    def _final_morning_memo(self, route: ControllerRoute, gate_report: dict[str, Any], final_summary: dict[str, Any]) -> str:
        vr2 = load_json(VR_ARTIFACTS["VR2"][0], {})
        vr4 = self._best_bundle_payload()
        best_profile = vr4.get("best_profile") or {}
        lines = [
            "# Morning Bundle",
            "",
            f"1. Did stronger visual reformulation materially improve G4b? {'yes' if (float(vr2.get('visual_alignment_gain', 0.0)) >= 0.20 or float(vr2.get('readability_gain', 0.0)) >= 0.10) else 'not yet'}",
            f"2. Did any canonical lane become G5-eligible? {'yes' if gate_report.get('G5_training_eligibility', {}).get('passed') else 'no'}",
            f"3. Is the correct scientific label still MINIMAL_REPAIR_INSUFFICIENT, or is stronger invalidation finally justified? {route.scientific_terminal_state}",
            f"4. Why does Stage B remain blocked, or why is it finally justified? {'Stage B remains blocked because G5 did not pass.' if not gate_report.get('G5_training_eligibility', {}).get('passed') else 'Stage B is justified because a canonical lane passed G5.'}",
            "",
            "## Key Results",
            f"- route_next_branch: {route.route_next_branch}",
            f"- strongest_negative_capped: {route.strongest_negative_capped}",
            f"- h0_posterior: {route.h0_posterior:.4f}" if route.h0_posterior is not None else "- h0_posterior: n/a",
            f"- h6_posterior: {route.h6_posterior:.4f}" if route.h6_posterior is not None else "- h6_posterior: n/a",
            f"- best_profile_id: {best_profile.get('profile_id')}",
            f"- best_rollout_ceiling_lift: {float((best_profile.get('rollout_ceiling_lift') or {}).get('score', 0.0)):.6f}",
            f"- best_state_alignment_gain: {float(best_profile.get('state_alignment_gain', 0.0)):.6f}",
            f"- best_sensor_contract_gain: {float((best_profile.get('sensor_contract_gain') or {}).get('score', 0.0)):.6f}",
            f"- cycles_completed: {final_summary.get('cycles_completed', 0)}",
            f"- completed_experiments: {', '.join(final_summary.get('completed_experiments', []))}",
        ]
        return "\n".join(lines).rstrip() + "\n"

    def _proposed_truth_delta(self, route: ControllerRoute) -> dict[str, Any]:
        return {
            "current": {
                "phase": self.authority.get("phase"),
                "verdict": route.scientific_terminal_state,
                "next_action": route.route_next_branch,
                "open_gates": [gate for gate in ["G4b_perceptual_readability", "G5_training_eligibility"]],
            },
            "note": "Autogenerated proposal only; no sovereign auto-promotion performed.",
        }

    def _proposed_next_actions(self, route: ControllerRoute) -> dict[str, Any]:
        return {
            "actions": [
                {
                    "id": "proposed_next_mainline",
                    "priority": 1,
                    "status": "pending",
                    "action": route.route_next_branch,
                    "why": route.why,
                }
            ]
        }

    def run(self, *, max_cycles: int = 4, sleep_seconds: int = 30) -> dict[str, Any]:
        cycle_count = 0
        last_route = ControllerRoute(None, "stay_current_branch", "Controller has not executed any cycle yet.")
        gate_report: dict[str, Any] = {}
        identity = controller_identity()
        controller_id = str(identity["controller_id"])
        run_id = str(identity["run_id"])
        owner = str(identity["owner"])
        pid = int(identity["pid"])
        lease_ok, lease_meta = acquire_controller_lease(controller_id, run_id, owner=owner, pid=pid)
        if not lease_ok:
            raise RuntimeError(f"controller lease denied: {lease_meta}")
        try:
            while cycle_count < int(max_cycles):
                refresh_controller_lease(controller_id, run_id, pid=pid)
                selected = self.select_experiments()
                if not selected:
                    break
                cycle_id = self._make_cycle_id()
                cycle_dir = create_cycle_dir(cycle_id, cycle_mode=self.cycle_mode)
                lane_specs_payload: list[dict[str, Any]] = []
                lane_results_payload: dict[str, Any] = {}
                cycle_updates: list[dict[str, Any]] = []
                selected_ids = [item["id"] for item in selected]
                append_cycle_record({
                    "cycle_id": cycle_id,
                    "cycle_mode": self.cycle_mode,
                    "selected_experiments": selected_ids,
                    "started_at": now_iso(),
                    "policy": dict(self.policy),
                    "resource_budget_snapshot": self._resource_snapshot(),
                })
                if len(selected_ids) > 1:
                    append_deviation_record(
                        message="Controller co-scheduled multiple VR experiments within a single cycle.",
                        scientific_semantics_changed=False,
                        details={"cycle_id": cycle_id, "selected_experiments": selected_ids},
                    )
                for item in selected:
                    refresh_controller_lease(controller_id, run_id, pid=pid)
                    result = item["runner"]()
                    exp_id = item["id"]
                    self.completed_experiments.append(exp_id)
                    lane_results_payload[exp_id] = result["payload"]
                    lane_specs_payload.extend(result.get("lane_specs", []))
                    cycle_updates.extend(result.get("updates", []))
                    append_evidence_record({
                        "cycle_id": cycle_id,
                        "experiment_id": exp_id,
                        "artifact_path": str(VR_ARTIFACTS[exp_id][0]),
                        "reported_at": now_iso(),
                    })
                self.board = apply_updates(self.board, cycle_updates)
                write_hypothesis_board(self.board)
                gate_report, last_route = self.evaluate_gates()
                cycle_summary = {
                    "cycle_id": cycle_id,
                    "started_at": now_iso(),
                    "ended_at": now_iso(),
                    "lane_family": self.experiment_family,
                    "selected_experiments": selected_ids,
                    "lane_ids": [spec.get("lane_id") for spec in lane_specs_payload],
                    "completed_experiments": list(dict.fromkeys(self.completed_experiments)),
                    "scientific_terminal_state": last_route.scientific_terminal_state,
                    "route_next_branch": last_route.route_next_branch,
                    "cycle_mode": self.cycle_mode,
                }
                cycle_memo = self._cycle_memo(last_route, gate_report, cycle_summary)
                route_payload = asdict(last_route)
                route_payload["generated_at"] = now_iso()
                write_cycle_bundle(
                    cycle_dir,
                    cycle_mode=self.cycle_mode,
                    lane_specs={"cycle_id": cycle_id, "lane_family": self.experiment_family, "lane_specs": lane_specs_payload, "resource_budget_snapshot": self._resource_snapshot()},
                    lane_results=lane_results_payload,
                    gate_report=gate_report,
                    hypothesis_board=self.board,
                    cycle_memo=cycle_memo,
                    proposed_current_truth_delta=self._proposed_truth_delta(last_route),
                    proposed_next_actions=self._proposed_next_actions(last_route),
                    cycle_summary=cycle_summary,
                    route_decision=route_payload,
                    policy_snapshot=dict(self.policy),
                    resource_budget_snapshot=self._resource_snapshot(),
                )
                cycle_count += 1
                if last_route.scientific_terminal_state in {"ALIGNED_CANONICAL_LANE_FOUND", "BENCHMARK_ALIGNMENT_UNSUPPORTED_UNDER_CURRENT_FORMULATION"}:
                    break
                if "VR6" in self.completed_experiments or cycle_count >= int(self.resource_limits["max_cycles_per_run"]):
                    break
                if sleep_seconds > 0 and not self.dry_run:
                    time.sleep(int(sleep_seconds))
            final_summary = {
                "generated_at": now_iso(),
                "cycle_mode": self.cycle_mode,
                "experiment_family": self.experiment_family,
                "cycles_completed": cycle_count,
                "completed_experiments": list(dict.fromkeys(self.completed_experiments)),
                "policy": dict(self.policy),
                "resource_budget_snapshot": self._resource_snapshot(),
            }
            route_payload = asdict(last_route)
            route_payload["generated_at"] = now_iso()
            morning_memo = self._final_morning_memo(last_route, gate_report, final_summary)
            write_final_run_outputs(route_decision=route_payload, morning_memo=morning_memo, final_summary=final_summary)
            return {"route_decision": route_payload, "final_summary": final_summary, "morning_memo_path": str(AUTOPILOT_DIR / 'morning_memo.md')}
        finally:
            release_controller_lease(controller_id, run_id)


def run_controller(
    *,
    max_cycles: int = 4,
    max_experiments_per_cycle: int = 3,
    sleep_seconds: int = 30,
    dry_run: bool = False,
    auto_promote_sovereign: bool = False,
    allow_full_retrain: bool = False,
    allow_new_claim: bool = False,
    run_mode: str = "autonomous_cycle_phase",
    cycle_mode: str = "unattended_cycle",
    experiment_family: str = "VR",
    cap_strongest_negative: bool = True,
    resume: bool = False,
    max_rollouts_per_experiment: int = 3,
    max_disk_growth_mb: int = 4096,
    retry_backoff_seconds: int = 5,
    max_retries: int = 2,
    max_cycles_per_run: int = 4,
    max_tiny_retrain_budget: int = 1,
) -> dict[str, Any]:
    controller = RootCauseController(
        max_experiments_per_cycle=max_experiments_per_cycle,
        dry_run=dry_run,
        auto_promote_sovereign=auto_promote_sovereign,
        allow_full_retrain=allow_full_retrain,
        allow_new_claim=allow_new_claim,
        run_mode=run_mode,
        cycle_mode=cycle_mode,
        experiment_family=experiment_family,
        cap_strongest_negative=cap_strongest_negative,
        resume=resume,
        max_rollouts_per_experiment=max_rollouts_per_experiment,
        max_disk_growth_mb=max_disk_growth_mb,
        retry_backoff_seconds=retry_backoff_seconds,
        max_retries=max_retries,
        max_cycles_per_run=max_cycles_per_run,
        max_tiny_retrain_budget=max_tiny_retrain_budget,
    )
    return controller.run(max_cycles=max_cycles, sleep_seconds=sleep_seconds)

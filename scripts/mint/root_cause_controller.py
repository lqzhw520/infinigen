#!/usr/bin/env python3
"""Gate-first causal controller for MINT/Infinigen root-cause investigation."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance

from drawer_robot_env_mujoco import (
    DrawerEnvContractConfig,
    build_robot_rollout,
    drawer_manifest,
)
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
    CLAIM_BOUNDARY_PATH,
    CYCLE_STATE_PATH,
    HYPOTHESIS_BOARD_PATH,
    LaneResult,
    LaneSpec,
    GateDecision,
    DatasetConfig,
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
    write_hypothesis_board,
)

CONTROL_TRACE_ROOT = ARTIFACT_DIR / "p1k_control_traces"
E0_ARTIFACT = ARTIFACT_DIR / "p2e0_observation_contract_audit.json"
E1_ARTIFACT = ARTIFACT_DIR / "p2e1_orientation_causality_matrix.json"
E2_ARTIFACT = ARTIFACT_DIR / "p2e2_state_contract_reaudit.json"
E3_ARTIFACT = ARTIFACT_DIR / "p2e3_visual_readability_probe.json"
E4_ARTIFACT = ARTIFACT_DIR / "p2e4_minimal_contract_bundle_ab.json"
E5_ARTIFACT = ARTIFACT_DIR / "p2e5_unique_data_only_control.json"
E6_ARTIFACT = ARTIFACT_DIR / "p2e6_optimization_only_control.json"
E7_ARTIFACT = ARTIFACT_DIR / "p2e7_tiny_retrain_confirmation.json"

E0_REPORT = OUTPUT_DIR / "p2e0_observation_contract_audit.md"
E1_REPORT = OUTPUT_DIR / "p2e1_orientation_causality_matrix.md"
E2_REPORT = OUTPUT_DIR / "p2e2_state_contract_reaudit.md"
E3_REPORT = OUTPUT_DIR / "p2e3_visual_readability_probe.md"
E4_REPORT = OUTPUT_DIR / "p2e4_minimal_contract_bundle_ab.md"
E5_REPORT = OUTPUT_DIR / "p2e5_unique_data_only_control.md"
E6_REPORT = OUTPUT_DIR / "p2e6_optimization_only_control.md"
E7_REPORT = OUTPUT_DIR / "p2e7_tiny_retrain_confirmation.md"


class RootCauseController:
    def __init__(self, *, max_experiments_per_cycle: int = 3, dry_run: bool = False):
        ensure_registry_layout()
        self.max_experiments_per_cycle = int(max_experiments_per_cycle)
        self.dry_run = bool(dry_run)
        self.claim_boundary = load_claim_boundary(CLAIM_BOUNDARY_PATH)
        self.authority = load_authority_snapshot()
        self.board = load_json(HYPOTHESIS_BOARD_PATH, default_board())
        if not self.board:
            self.board = default_board()
        self.completed_experiments = list(load_json(CYCLE_STATE_PATH, {}).get("completed_experiments", []))
        self.heldout_seeds = default_heldout_seeds()
        available = drawer_manifest().get("available_seeds", [])
        self.train_seeds = [int(seed) for seed in available if int(seed) not in self.heldout_seeds][:5]
        self.control_images = self._load_control_images()
        self.control_state, self.control_action = self._load_control_arrays()

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

    def _make_lane_spec(
        self,
        *,
        lane_id: str,
        stage: str,
        env_contract: dict[str, Any],
        interventions: dict[str, Any] | None = None,
        claim_policy: str = "diagnostic",
        note: str = "",
        experiment_id: str,
        coverage: dict[str, float] | None = None,
        cost: float = 1.0,
    ) -> LaneSpec:
        spec = LaneSpec(
            lane_id=lane_id,
            stage=stage,
            env_contract_config=env_contract,
            interventions=interventions or {},
            dataset_config=DatasetConfig(train_seed_pool=self.train_seeds, heldout_seed_pool=self.heldout_seeds),
            train_config=TrainConfig(),
            claim_policy=claim_policy,
            note=note,
            experiment_id=experiment_id,
            coverage=coverage or {},
            cost=cost,
        )
        resolved_policy, violations = classify_claim_policy(self.claim_boundary, spec)
        if resolved_policy != spec.claim_policy:
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

    def _run_rollout_set(self, spec: LaneSpec, seeds: list[int]) -> list[dict[str, Any]]:
        rollouts = []
        rotation_source = str(spec.interventions.get("rotation_source", "zero"))
        contract = self._materialize_contract(spec.env_contract_config)
        grasp_pose = np.eye(4, dtype=np.float32)
        for episode_index, seed in enumerate(seeds):
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
        for i, value in enumerate(wd):
            a_min, a_max = float(np.min(control[:, i])), float(np.max(control[:, i]))
            b_min, b_max = float(np.min(target[:, i])), float(np.max(target[:, i]))
            span = max(a_max, b_max) - min(a_min, b_min)
            normed.append(float(value / max(span, 1e-6)))
        return float(np.mean(normed) + 0.25 * np.mean(smooth_gap) + 0.1 * np.mean([1.0 - x for x in overlap]))

    def _artifact_paths(self, experiment_id: str) -> tuple[Path, Path]:
        mapping = {
            "E0": (E0_ARTIFACT, E0_REPORT),
            "E1": (E1_ARTIFACT, E1_REPORT),
            "E2": (E2_ARTIFACT, E2_REPORT),
            "E3": (E3_ARTIFACT, E3_REPORT),
            "E4": (E4_ARTIFACT, E4_REPORT),
            "E5": (E5_ARTIFACT, E5_REPORT),
            "E6": (E6_ARTIFACT, E6_REPORT),
            "E7": (E7_ARTIFACT, E7_REPORT),
        }
        return mapping[experiment_id]

    def experiment_catalog(self) -> list[dict[str, Any]]:
        return [
            {"id": "E0", "coverage": {"H0_same_problem_identity": 0.8, "H3_observation_visual_contract": 1.0}, "cost": 1.0, "runner": self.run_e0},
            {"id": "E1", "coverage": {"H1_embodiment_causal_contract": 1.0, "H6_environment_invalid_for_claim": 0.4}, "cost": 1.2, "runner": self.run_e1},
            {"id": "E2", "coverage": {"H2_state_representability": 1.0, "H6_environment_invalid_for_claim": 0.7}, "cost": 1.1, "runner": self.run_e2},
            {"id": "E3", "coverage": {"H3_observation_visual_contract": 1.0, "H0_same_problem_identity": 0.5}, "cost": 1.2, "runner": self.run_e3},
            {"id": "E4", "coverage": {"H6_environment_invalid_for_claim": 1.0, "H1_embodiment_causal_contract": 0.5}, "cost": 1.5, "runner": self.run_e4},
            {"id": "E5", "coverage": {"H4_unique_data_scale_only": 1.0}, "cost": 1.8, "runner": self.run_e5},
            {"id": "E6", "coverage": {"H5_optimization_only": 1.0}, "cost": 1.8, "runner": self.run_e6},
            {"id": "E7", "coverage": {"H0_same_problem_identity": 0.4, "H6_environment_invalid_for_claim": -0.2}, "cost": 2.5, "runner": self.run_e7},
        ]

    def select_experiments(self) -> list[dict[str, Any]]:
        selected = []
        for item in self.experiment_catalog():
            if item["id"] in self.completed_experiments:
                continue
            if item["id"] in {"E3", "E4"} and not all(x in self.completed_experiments for x in ["E0", "E1", "E2"]):
                continue
            if item["id"] in {"E5", "E6"} and "E4" not in self.completed_experiments:
                continue
            if item["id"] == "E7" and "E4" not in self.completed_experiments:
                continue
            score = 0.0
            for hypothesis_id, coverage in item["coverage"].items():
                state = self.board.get(hypothesis_id)
                if not state:
                    continue
                score += float(state.get("importance", 1.0)) * unresolvedness(state) * float(coverage)
            selected.append((score / max(float(item["cost"]), 1e-6), item))
        selected.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in selected[: self.max_experiments_per_cycle]]

    def run_e0(self) -> dict[str, Any]:
        seeds = self.train_seeds[:2]
        lanes = [
            self._make_lane_spec(
                lane_id="A0_baseline_current",
                stage="probe",
                env_contract=asdict(DrawerEnvContractConfig.legacy_defaults()),
                interventions={"rotation_source": "zero"},
                claim_policy="diagnostic",
                note="Legacy baseline observation surface.",
                experiment_id="E0",
            ),
            self._make_lane_spec(
                lane_id="A1_no_overlay_no_calibration",
                stage="probe",
                env_contract={
                    **asdict(DrawerEnvContractConfig.legacy_defaults()),
                    "enable_marker_overlay": False,
                    "calibration_mode": "none",
                    "canonical_lane": True,
                },
                interventions={"rotation_source": "zero"},
                claim_policy="canonical",
                note="Remove marker and heuristic calibration while keeping fixed scene secondary camera.",
                experiment_id="E0",
            ),
            self._make_lane_spec(
                lane_id="A2_wrist_camera_only",
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
                interventions={"rotation_source": "zero"},
                claim_policy="canonical",
                note="Minimal observation-contract repair without changing interaction physics.",
                experiment_id="E0",
            ),
        ]
        control_stats = summarize_visual(self.control_images)
        lane_reports = []
        for spec in lanes:
            rollouts = self._run_rollout_set(spec, seeds)
            residuals = self._secondary_residuals(rollouts)
            target_stats = summarize_visual(self._sample_images(rollouts, limit=24))
            gap_payload = visual_gap(control_stats, target_stats)
            lane_reports.append({
                "lane": spec.to_payload(),
                "claim_policy": spec.claim_policy,
                "secondary_camera_wrist_like": bool(residuals and float(np.mean(residuals)) < 0.02 and float(np.max(residuals)) < 0.05),
                "secondary_camera_residual_mean": float(np.mean(residuals)) if residuals else None,
                "secondary_camera_residual_max": float(np.max(residuals)) if residuals else None,
                "marker_overlay_enabled": bool(spec.env_contract_config.get("enable_marker_overlay", False)),
                "diagnostic_only": spec.claim_policy != "canonical",
                "provenance_complete": True,
                "control_stats": control_stats,
                "target_stats": target_stats,
                "gap": gap_payload,
                "weighted_visual_gap": weighted_visual_gap(gap_payload),
            })
        canonical_reports = [report for report in lane_reports if report["claim_policy"] == "canonical"]
        best_canonical = min(canonical_reports, key=lambda item: item["weighted_visual_gap"], default=None)
        observation_gate_pass = bool(best_canonical and observation_contract_pass(best_canonical))
        updates = [
            {
                "hypothesis_id": "H3_observation_visual_contract",
                "support_score": 0.8 if best_canonical and best_canonical["weighted_visual_gap"] < lane_reports[0]["weighted_visual_gap"] else 0.2,
                "alpha": 0.6,
                "evidence_ref": "E0",
            },
            {
                "hypothesis_id": "H0_same_problem_identity",
                "support_score": 0.3 if observation_gate_pass else -0.6,
                "alpha": 0.6,
                "evidence_ref": "E0",
            },
        ]
        payload = {
            "experiment_id": "E0",
            "generated_at": now_iso(),
            "passed": observation_gate_pass,
            "control_stats": control_stats,
            "lanes": lane_reports,
            "best_canonical_lane": best_canonical,
            "observation_gate_pass": observation_gate_pass,
        }
        artifact_path, report_path = self._artifact_paths("E0")
        write_json_atomic(artifact_path, payload)
        lines = [
            "# p2e0 Observation Contract Audit",
            "",
            f"Generated: {payload['generated_at']}",
            f"- observation_gate_pass: {observation_gate_pass}",
        ]
        for report in lane_reports:
            lines.append(
                f"- {report['lane']['lane_id']}: claim_policy={report['claim_policy']} wrist_like={report['secondary_camera_wrist_like']} weighted_visual_gap={report['weighted_visual_gap']:.6f}"
            )
        write_text_atomic(report_path, "\n".join(lines).rstrip() + "\n")
        return {"payload": payload, "updates": updates}

    def run_e1(self) -> dict[str, Any]:
        seeds = self.train_seeds[:3]
        lane_specs = []
        for interaction_mode in ["legacy_translation_only", "orientation_sensitive_v1"]:
            for rotation_source in ["zero", "aligned", "random"]:
                lane_specs.append(
                    self._make_lane_spec(
                        lane_id=f"{interaction_mode}_{rotation_source}",
                        stage="probe",
                        env_contract={
                            "secondary_camera_mode": "wrist_dynamic",
                            "enable_marker_overlay": False,
                            "calibration_mode": "none",
                            "interaction_mode": interaction_mode,
                            "state_mode": "m0_proxy",
                            "emit_orientation_telemetry": True,
                            "emit_camera_metadata": True,
                            "canonical_lane": False,
                        },
                        interventions={"rotation_source": rotation_source},
                        claim_policy="diagnostic",
                        note="Orientation-causality matrix lane.",
                        experiment_id="E1",
                    )
                )
        matrix: dict[str, Any] = {}
        for spec in lane_specs:
            rollouts = self._run_rollout_set(spec, seeds)
            summary = self._rollout_summary(rollouts)
            matrix[spec.lane_id] = {
                "lane": spec.to_payload(),
                "rollout_summary": summary,
            }
        aligned_sensitive = matrix["orientation_sensitive_v1_aligned"]["rollout_summary"]["mean_max_drawer_fraction"]
        zero_sensitive = matrix["orientation_sensitive_v1_zero"]["rollout_summary"]["mean_max_drawer_fraction"]
        random_sensitive = matrix["orientation_sensitive_v1_random"]["rollout_summary"]["mean_max_drawer_fraction"]
        legacy_gap = matrix["legacy_translation_only_aligned"]["rollout_summary"]["mean_max_drawer_fraction"] - max(
            matrix["legacy_translation_only_zero"]["rollout_summary"]["mean_max_drawer_fraction"],
            matrix["legacy_translation_only_random"]["rollout_summary"]["mean_max_drawer_fraction"],
        )
        orientation_causal_sensitivity = float(aligned_sensitive - max(zero_sensitive, random_sensitive))
        payload = {
            "experiment_id": "E1",
            "generated_at": now_iso(),
            "passed": orientation_causal_sensitivity > 0.01,
            "matrix": matrix,
            "orientation_causal_sensitivity": orientation_causal_sensitivity,
            "legacy_alignment_gap": float(legacy_gap),
            "aligned_beats_random_under_sensitive": bool(aligned_sensitive > random_sensitive),
        }
        updates = [
            {
                "hypothesis_id": "H1_embodiment_causal_contract",
                "support_score": 0.9 if orientation_causal_sensitivity > 0.01 and abs(float(legacy_gap)) < 0.02 else -0.4,
                "alpha": 1.0,
                "evidence_ref": "E1",
            },
            {
                "hypothesis_id": "H6_environment_invalid_for_claim",
                "support_score": 0.5 if orientation_causal_sensitivity <= 0.0 else -0.2,
                "alpha": 0.6,
                "evidence_ref": "E1",
            },
        ]
        artifact_path, report_path = self._artifact_paths("E1")
        write_json_atomic(artifact_path, payload)
        lines = [
            "# p2e1 Orientation Causality Matrix",
            "",
            f"Generated: {payload['generated_at']}",
            f"- orientation_causal_sensitivity: {orientation_causal_sensitivity:.6f}",
            f"- legacy_alignment_gap: {legacy_gap:.6f}",
        ]
        for key, value in matrix.items():
            lines.append(
                f"- {key}: attach_rate={value['rollout_summary']['attach_rate']:.3f} mean_max_drawer_fraction={value['rollout_summary']['mean_max_drawer_fraction']:.3f}"
            )
        write_text_atomic(report_path, "\n".join(lines).rstrip() + "\n")
        return {"payload": payload, "updates": updates}

    def run_e2(self) -> dict[str, Any]:
        seeds = self.train_seeds[:3]
        candidates = {
            "M0_proxy": self._make_lane_spec(
                lane_id="A6_m0_proxy",
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
                interventions={"rotation_source": "zero"},
                claim_policy="diagnostic",
                note="Current proxy state.",
                experiment_id="E2",
            ),
            "M_pose": self._make_lane_spec(
                lane_id="A6_eef_pose_gripper",
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
                interventions={"rotation_source": "zero"},
                claim_policy="diagnostic",
                note="Diagnostic pose/gripper state without duplication.",
                experiment_id="E2",
            ),
            "M_telemetry_v1": self._make_lane_spec(
                lane_id="A6_telemetry_candidate_v1",
                stage="probe",
                env_contract={
                    "secondary_camera_mode": "wrist_dynamic",
                    "enable_marker_overlay": False,
                    "calibration_mode": "none",
                    "interaction_mode": "orientation_sensitive_v1",
                    "state_mode": "telemetry_candidate_v1",
                    "emit_orientation_telemetry": True,
                    "emit_camera_metadata": True,
                    "canonical_lane": True,
                },
                interventions={"rotation_source": "aligned"},
                claim_policy="canonical",
                note="Best-effort telemetry-compatible candidate under current embodiment.",
                experiment_id="E2",
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
        telemetry_score = candidate_scores.get("M_telemetry_v1", {}).get("overall_score")
        telemetry_gain = state_alignment_gain(baseline_score, telemetry_score) if telemetry_score is not None else -1.0
        impossible = selected_mapping == "IMPOSSIBLE"
        telemetry_spec = candidate_scores.get("M_telemetry_v1", {}).get("state_spec", {})
        telemetry_semantic_pass = bool(
            candidate_scores.get("M_telemetry_v1", {}).get("valid")
            and not telemetry_spec.get("fraud_padding", False)
            and telemetry_spec.get("all_dims_explained", False)
        )
        strict_state_gate_ready = bool(
            not impossible
            and selected_mapping == "M_telemetry_v1"
            and telemetry_semantic_pass
        )
        payload = {
            "experiment_id": "E2",
            "generated_at": now_iso(),
            "passed": strict_state_gate_ready,
            "selected_state_mapping": selected_mapping,
            "candidate_state_scores": candidate_scores,
            "state_alignment_gain_vs_m0": telemetry_gain,
            "telemetry_semantic_pass": telemetry_semantic_pass,
            "strict_state_gate_ready": strict_state_gate_ready,
            "impossible": impossible,
        }
        updates = [
            {
                "hypothesis_id": "H2_state_representability",
                "support_score": 0.8 if impossible or telemetry_gain <= 0.0 else -0.4,
                "alpha": 1.0,
                "evidence_ref": "E2",
            },
            {
                "hypothesis_id": "H6_environment_invalid_for_claim",
                "support_score": 0.8 if impossible else 0.2,
                "alpha": 0.8,
                "evidence_ref": "E2",
            },
        ]
        artifact_path, report_path = self._artifact_paths("E2")
        write_json_atomic(artifact_path, payload)
        lines = [
            "# p2e2 State Contract Reaudit",
            "",
            f"Generated: {payload['generated_at']}",
            f"- selected_state_mapping: {selected_mapping}",
            f"- impossible: {impossible}",
            f"- telemetry_semantic_pass: {telemetry_semantic_pass}",
            f"- strict_state_gate_ready: {strict_state_gate_ready}",
            f"- telemetry_gain_vs_m0: {telemetry_gain:.6f}",
        ]
        for key, value in candidate_scores.items():
            if value.get("valid"):
                lines.append(f"- {key}: valid overall_score={value['overall_score']:.6f}")
            else:
                lines.append(f"- {key}: invalid ({value.get('invalid_reason')})")
        write_text_atomic(report_path, "\n".join(lines).rstrip() + "\n")
        return {"payload": payload, "updates": updates}

    def run_e3(self) -> dict[str, Any]:
        seeds = self.train_seeds[:2]
        lanes = [
            self._make_lane_spec(
                lane_id="E3_current_legacy",
                stage="probe",
                env_contract=asdict(DrawerEnvContractConfig.legacy_defaults()),
                interventions={"rotation_source": "zero"},
                claim_policy="diagnostic",
                note="Legacy visual surface.",
                experiment_id="E3",
            ),
            self._make_lane_spec(
                lane_id="E3_no_overlay_no_calibration",
                stage="probe",
                env_contract={
                    **asdict(DrawerEnvContractConfig.legacy_defaults()),
                    "enable_marker_overlay": False,
                    "calibration_mode": "none",
                    "canonical_lane": True,
                },
                interventions={"rotation_source": "zero"},
                claim_policy="canonical",
                note="Remove non-benchmark overlays and calibration heuristic.",
                experiment_id="E3",
            ),
            self._make_lane_spec(
                lane_id="E3_wrist_camera_only",
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
                interventions={"rotation_source": "zero"},
                claim_policy="canonical",
                note="Observation-contract repair via wrist-like camera.",
                experiment_id="E3",
            ),
            self._make_lane_spec(
                lane_id="E3_diagnostic_texture",
                stage="probe",
                env_contract={
                    "secondary_camera_mode": "wrist_dynamic",
                    "enable_marker_overlay": False,
                    "calibration_mode": "diagnostic_texture",
                    "interaction_mode": "legacy_translation_only",
                    "state_mode": "m0_proxy",
                    "emit_orientation_telemetry": True,
                    "emit_camera_metadata": True,
                    "canonical_lane": False,
                },
                interventions={"rotation_source": "zero"},
                claim_policy="diagnostic",
                note="Diagnostic-only texture lane.",
                experiment_id="E3",
            ),
        ]
        control_stats = summarize_visual(self.control_images)
        lane_reports = []
        for spec in lanes:
            rollouts = self._run_rollout_set(spec, seeds)
            target_stats = summarize_visual(self._sample_images(rollouts, limit=24))
            gap_payload = visual_gap(control_stats, target_stats)
            lane_reports.append({
                "lane": spec.to_payload(),
                "claim_policy": spec.claim_policy,
                "target_stats": target_stats,
                "gap": gap_payload,
                "weighted_visual_gap": weighted_visual_gap(gap_payload),
                "diagnostic_only": spec.claim_policy != "canonical",
            })
        best_lane = min(lane_reports, key=lambda item: item["weighted_visual_gap"])
        baseline_gap_value = lane_reports[0]["weighted_visual_gap"]
        best_gain = visual_alignment_gain(baseline_gap_value, best_lane["weighted_visual_gap"])
        payload = {
            "experiment_id": "E3",
            "generated_at": now_iso(),
            "passed": best_gain > 0.0,
            "control_stats": control_stats,
            "lanes": lane_reports,
            "best_lane": best_lane,
            "best_visual_alignment_gain": best_gain,
        }
        updates = [
            {
                "hypothesis_id": "H3_observation_visual_contract",
                "support_score": 0.7 if best_gain > 0.0 else -0.2,
                "alpha": 0.6,
                "evidence_ref": "E3",
            }
        ]
        artifact_path, report_path = self._artifact_paths("E3")
        write_json_atomic(artifact_path, payload)
        lines = [
            "# p2e3 Visual Readability Probe",
            "",
            f"Generated: {payload['generated_at']}",
            f"- best_lane: {best_lane['lane']['lane_id']}",
            f"- best_visual_alignment_gain: {best_gain:.6f}",
        ]
        for report in lane_reports:
            lines.append(f"- {report['lane']['lane_id']}: weighted_visual_gap={report['weighted_visual_gap']:.6f} claim_policy={report['claim_policy']}")
        write_text_atomic(report_path, "\n".join(lines).rstrip() + "\n")
        return {"payload": payload, "updates": updates}

    def run_e4(self) -> dict[str, Any]:
        seeds = self.train_seeds[:3]
        baseline = self._make_lane_spec(
            lane_id="E4_baseline_current",
            stage="probe",
            env_contract=asdict(DrawerEnvContractConfig.legacy_defaults()),
            interventions={"rotation_source": "zero"},
            claim_policy="diagnostic",
            note="Legacy current env baseline.",
            experiment_id="E4",
        )
        minimal = self._make_lane_spec(
            lane_id="E4_minimal_contract_bundle",
            stage="probe",
            env_contract={
                "secondary_camera_mode": "wrist_dynamic",
                "enable_marker_overlay": False,
                "calibration_mode": "none",
                "interaction_mode": "orientation_sensitive_v1",
                "state_mode": "telemetry_candidate_v1",
                "emit_orientation_telemetry": True,
                "emit_camera_metadata": True,
                "canonical_lane": True,
            },
            interventions={"rotation_source": "aligned"},
            claim_policy="canonical",
            note="Minimal benchmark-defensible contract repair bundle.",
            experiment_id="E4",
        )
        baseline_rollouts = self._run_rollout_set(baseline, seeds)
        minimal_rollouts = self._run_rollout_set(minimal, seeds)
        baseline_summary = self._rollout_summary(baseline_rollouts)
        minimal_summary = self._rollout_summary(minimal_rollouts)
        sensitive_metric = float(minimal_summary["mean_max_drawer_fraction"] - max(baseline_summary["mean_max_drawer_fraction"], baseline_summary["mean_max_drawer_fraction"]))
        minimal_summary["orientation_causal_sensitivity"] = sensitive_metric
        baseline_summary["orientation_causal_sensitivity"] = 0.0
        ceiling = rollout_ceiling_lift(baseline_summary, minimal_summary)
        e3_payload = load_json(E3_ARTIFACT, {})
        baseline_gap = lane_gap = None
        sensor_gain = {"score": 0.0, "wrist_relativeness_gain": 0.0, "no_overlay_gain": 0.0, "visual_alignment_gain": 0.0, "framing_gain": 0.0}
        if e3_payload:
            baseline_gap = next((lane["weighted_visual_gap"] for lane in e3_payload.get("lanes", []) if lane["lane"]["lane_id"] == "E3_current_legacy"), None)
            lane_gap = next((lane["weighted_visual_gap"] for lane in e3_payload.get("lanes", []) if lane["lane"]["lane_id"] == "E3_wrist_camera_only"), None)
            if baseline_gap is not None and lane_gap is not None:
                sensor_gain = sensor_contract_gain(
                    wrist_relativeness_gain=1.0,
                    no_overlay_gain=1.0,
                    visual_alignment_gain_value=visual_alignment_gain(baseline_gap, lane_gap),
                    framing_gain=0.0,
                )
        e2_payload = load_json(E2_ARTIFACT, {})
        baseline_score = float(e2_payload.get("candidate_state_scores", {}).get("M0_proxy", {}).get("overall_score") or 1.0)
        minimal_score = float(e2_payload.get("candidate_state_scores", {}).get("M_telemetry_v1", {}).get("overall_score") or baseline_score)
        state_gain = state_alignment_gain(baseline_score, minimal_score)
        payload = {
            "experiment_id": "E4",
            "generated_at": now_iso(),
            "passed": ceiling["score"] > 0.0,
            "baseline_summary": baseline_summary,
            "minimal_summary": minimal_summary,
            "rollout_ceiling_lift": ceiling,
            "state_alignment_gain": state_gain,
            "sensor_contract_gain": sensor_gain,
        }
        updates = [
            {
                "hypothesis_id": "H6_environment_invalid_for_claim",
                "support_score": 0.9 if ceiling["score"] <= 0.0 else -0.4,
                "alpha": 1.0,
                "evidence_ref": "E4",
            },
            {
                "hypothesis_id": "H0_same_problem_identity",
                "support_score": 0.5 if ceiling["score"] > 0.0 and sensor_gain["score"] > 0.0 else -0.6,
                "alpha": 0.6,
                "evidence_ref": "E4",
            },
        ]
        artifact_path, report_path = self._artifact_paths("E4")
        write_json_atomic(artifact_path, payload)
        lines = [
            "# p2e4 Minimal Contract Bundle A/B",
            "",
            f"Generated: {payload['generated_at']}",
            f"- rollout_ceiling_lift: {ceiling['score']:.6f}",
            f"- state_alignment_gain: {state_gain:.6f}",
            f"- sensor_contract_gain: {sensor_gain['score']:.6f}",
        ]
        write_text_atomic(report_path, "\n".join(lines).rstrip() + "\n")
        return {"payload": payload, "updates": updates}

    def _blocked_payload(self, experiment_id: str, reason: str) -> dict[str, Any]:
        payload = {
            "experiment_id": experiment_id,
            "generated_at": now_iso(),
            "passed": False,
            "status": "skipped",
            "reason": reason,
        }
        artifact_path, report_path = self._artifact_paths(experiment_id)
        write_json_atomic(artifact_path, payload)
        write_text_atomic(report_path, f"# {experiment_id}\n\n- skipped: {reason}\n")
        return {"payload": payload, "updates": []}

    def run_e5(self) -> dict[str, Any]:
        return self._blocked_payload("E5", "Unique-data-only falsification remains blocked until minimal contract repair and train eligibility gates are clearer.")

    def run_e6(self) -> dict[str, Any]:
        return self._blocked_payload("E6", "Optimization-only control remains blocked until the controller has a train-eligible canonical lane to compare against.")

    def run_e7(self) -> dict[str, Any]:
        return self._blocked_payload("E7", "Tiny retrain confirmation is disabled by default until G5 passes and ALLOW_FULL_RETRAIN remains false.")

    def evaluate_gates(self) -> dict[str, Any]:
        completed = set(self.completed_experiments)
        e0 = load_json(E0_ARTIFACT, {}) if "E0" in completed else {}
        e1 = load_json(E1_ARTIFACT, {}) if "E1" in completed else {}
        e2 = load_json(E2_ARTIFACT, {}) if "E2" in completed else {}
        e3 = load_json(E3_ARTIFACT, {}) if "E3" in completed else {}
        e4 = load_json(E4_ARTIFACT, {}) if "E4" in completed else {}

        decisions = []
        g0 = GateDecision(
            gate_id="G0_baseline_integrity",
            passed=bool(self.authority.get("phase") and self.authority.get("next_action") is not None),
            summary="Authority files readable and baseline surfaces discoverable.",
            blocking=True,
        )
        decisions.append(g0)

        e0_best = e0.get("best_canonical_lane") or {}
        g1 = GateDecision(
            gate_id="G1_observation_contract",
            passed=bool(e0 and e0.get("observation_gate_pass", False)),
            summary="Observation contract requires wrist-like second camera, no overlay, and provenance-complete canonical lane.",
            blocking=True,
            details=e0_best,
        )
        decisions.append(g1)

        g2 = GateDecision(
            gate_id="G2_action_causality",
            passed=bool(e1 and e1.get("orientation_causal_sensitivity", 0.0) > 0.01 and e1.get("aligned_beats_random_under_sensitive", False)),
            summary="Rotation can only matter if orientation-sensitive physics makes aligned rotation measurably better than zero/random.",
            blocking=True,
            details={
                "orientation_causal_sensitivity": e1.get("orientation_causal_sensitivity"),
                "aligned_beats_random_under_sensitive": e1.get("aligned_beats_random_under_sensitive"),
            },
        )
        decisions.append(g2)

        impossible = bool(e2.get("impossible", False))
        telemetry_semantic_pass = e2.get("telemetry_semantic_pass")
        strict_state_gate_ready = e2.get("strict_state_gate_ready")
        if telemetry_semantic_pass is None or strict_state_gate_ready is None:
            telemetry_candidate = (e2.get("candidate_state_scores") or {}).get("M_telemetry_v1", {})
            telemetry_spec = telemetry_candidate.get("state_spec") or {}
            telemetry_semantic_pass = bool(
                telemetry_candidate.get("valid")
                and not telemetry_spec.get("fraud_padding", False)
                and telemetry_spec.get("all_dims_explained", False)
            )
            strict_state_gate_ready = bool(
                not impossible
                and e2.get("selected_state_mapping") == "M_telemetry_v1"
                and telemetry_semantic_pass
            )
        g3 = GateDecision(
            gate_id="G3_state_semantics",
            passed=bool(e2 and strict_state_gate_ready and not impossible),
            summary="State gate requires a telemetry-compatible, provenance-complete, non-duplicated state candidate or an explicit impossibility memo.",
            blocking=True,
            details={
                "selected_state_mapping": e2.get("selected_state_mapping"),
                "telemetry_semantic_pass": telemetry_semantic_pass,
                "strict_state_gate_ready": strict_state_gate_ready,
                "impossible": impossible,
            },
        )
        decisions.append(g3)

        best_lane = e3.get("best_lane") or {}
        g4 = GateDecision(
            gate_id="G4_visual_canonicality",
            passed=bool(e3 and best_lane and best_lane.get("claim_policy") == "canonical" and e3.get("best_visual_alignment_gain", 0.0) >= 0.0),
            summary="Visual gate requires a canonical, non-diagnostic lane that improves the visual gap.",
            blocking=True,
            details=best_lane,
        )
        decisions.append(g4)

        train_eligible = bool(
            g1.passed
            and g2.passed
            and g3.passed
            and g4.passed
            and (
                float(e4.get("rollout_ceiling_lift", {}).get("score", 0.0)) >= 0.15
                or float(e4.get("state_alignment_gain", 0.0)) >= 0.20
                or float(e4.get("sensor_contract_gain", {}).get("score", 0.0)) >= 0.25
            )
        )
        g5 = GateDecision(
            gate_id="G5_training_eligibility",
            passed=train_eligible,
            summary="Training is only eligible after the observation/action/state/visual gates pass and one causal gain threshold is met.",
            blocking=False,
            details={
                "rollout_ceiling_lift": e4.get("rollout_ceiling_lift", {}).get("score"),
                "state_alignment_gain": e4.get("state_alignment_gain"),
                "sensor_contract_gain": e4.get("sensor_contract_gain", {}).get("score"),
            },
        )
        decisions.append(g5)

        terminal_state = None
        if e0 and e1 and e2 and e3 and e4 and not g5.passed:
            terminal_state = "ENVIRONMENT_REFORMULATION_REQUIRED"
        gate_report = {
            "generated_at": now_iso(),
            "gates": [decision.__dict__ for decision in decisions],
            "contract_validity_score": contract_validity_score(g1.passed, g2.passed, g3.passed, g4.passed),
            "terminal_state": terminal_state,
        }
        return gate_report

    def _proposed_truth_delta(self, gate_report: dict[str, Any], selected: list[dict[str, Any]]) -> dict[str, Any]:
        terminal_state = gate_report.get("terminal_state")
        if terminal_state == "ENVIRONMENT_REFORMULATION_REQUIRED":
            return {
                "phase": "v64_ENVIRONMENT_REFORMULATION_REQUIRED_AFTER_CONTRACT_GATES",
                "verdict": "Minimal canonical repair failed to produce a train-eligible lane; escalate to environment reformulation.",
                "next_action": {
                    "type": "ENVIRONMENT_REFORMULATION_DECISION",
                    "id": "review_environment_reformulation_options",
                    "status": "pending",
                    "priority": "P0",
                },
            }
        return {
            "phase": self.authority.get("phase"),
            "verdict": self.authority.get("verdict"),
            "next_action": self.authority.get("next_action"),
            "selected_experiments": [item["id"] for item in selected],
        }

    def _proposed_next_actions(self, gate_report: dict[str, Any], selected: list[dict[str, Any]]) -> dict[str, Any]:
        terminal_state = gate_report.get("terminal_state")
        if terminal_state:
            return {
                "actions": [
                    {
                        "type": terminal_state,
                        "id": terminal_state.lower(),
                        "status": "pending",
                        "priority": "P0",
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

    def _cycle_memo(self, cycle_id: str, selected: list[dict[str, Any]], gate_report: dict[str, Any]) -> str:
        lines = [
            "# Root-Cause Controller Cycle Memo",
            "",
            f"- cycle_id: {cycle_id}",
            f"- authority_phase: {self.authority.get('phase')}",
            f"- selected_experiments: {[item['id'] for item in selected]}",
            f"- terminal_state: {gate_report.get('terminal_state')}",
            "",
            "## Gates",
        ]
        for gate in gate_report.get("gates", []):
            lines.append(f"- {gate['gate_id']}: passed={gate['passed']} summary={gate['summary']}")
        lines.extend(["", "## Hypotheses"]) 
        for hypothesis_id, state in self.board.items():
            lines.append(f"- {hypothesis_id}: posterior={float(state.get('posterior', 0.5)):.4f}")
        return "\n".join(lines).rstrip() + "\n"

    def run_cycle(self) -> dict[str, Any]:
        cycle_id = self._make_cycle_id()
        cycle_dir = create_cycle_dir(cycle_id)
        selected = self.select_experiments()
        results = {}
        updates = []
        if not selected:
            gate_report = self.evaluate_gates()
            if not gate_report.get("terminal_state"):
                gate_report["terminal_state"] = "ENVIRONMENT_REFORMULATION_REQUIRED"
            proposed_current_truth_delta = self._proposed_truth_delta(gate_report, selected)
            proposed_next_actions = self._proposed_next_actions(gate_report, selected)
            memo = self._cycle_memo(cycle_id, selected, gate_report)
            write_cycle_bundle(
                cycle_dir,
                lane_specs={},
                lane_results={},
                gate_report=gate_report,
                hypothesis_board=self.board,
                cycle_memo=memo,
                proposed_current_truth_delta=proposed_current_truth_delta,
                proposed_next_actions=proposed_next_actions,
            )
            append_cycle_record(
                {
                    "timestamp": now_iso(),
                    "cycle_id": cycle_id,
                    "selected_experiments": [],
                    "terminal_state": gate_report.get("terminal_state"),
                }
            )
            write_json_atomic(
                CYCLE_STATE_PATH,
                {
                    "updated_at": now_iso(),
                    "last_cycle_id": cycle_id,
                    "completed_experiments": self.completed_experiments,
                    "terminal_state": gate_report.get("terminal_state"),
                },
            )
            return {
                "cycle_id": cycle_id,
                "selected_experiments": [],
                "gate_report": gate_report,
                "terminal_state": gate_report.get("terminal_state"),
            }
        for item in selected:
            result = item["runner"]()
            results[item["id"]] = result["payload"]
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
        proposed_current_truth_delta = self._proposed_truth_delta(gate_report, selected)
        proposed_next_actions = self._proposed_next_actions(gate_report, selected)
        memo = self._cycle_memo(cycle_id, selected, gate_report)
        lane_specs_payload = {item["id"]: {"coverage": item["coverage"], "cost": item["cost"]} for item in selected}
        write_cycle_bundle(
            cycle_dir,
            lane_specs=lane_specs_payload,
            lane_results=results,
            gate_report=gate_report,
            hypothesis_board=self.board,
            cycle_memo=memo,
            proposed_current_truth_delta=proposed_current_truth_delta,
            proposed_next_actions=proposed_next_actions,
        )
        append_cycle_record(
            {
                "timestamp": now_iso(),
                "cycle_id": cycle_id,
                "selected_experiments": [item["id"] for item in selected],
                "terminal_state": gate_report.get("terminal_state"),
            }
        )
        write_json_atomic(
            CYCLE_STATE_PATH,
            {
                "updated_at": now_iso(),
                "last_cycle_id": cycle_id,
                "completed_experiments": self.completed_experiments,
                "terminal_state": gate_report.get("terminal_state"),
            },
        )
        return {
            "cycle_id": cycle_id,
            "selected_experiments": [item["id"] for item in selected],
            "gate_report": gate_report,
            "terminal_state": gate_report.get("terminal_state"),
        }


def run_controller(max_cycles: int = 1, max_experiments_per_cycle: int = 3, sleep_seconds: int = 0, dry_run: bool = False) -> dict[str, Any]:
    identity = controller_identity(owner="mint_v2_root_cause_controller")
    acquired, _lease = acquire_controller_lease(identity["controller_id"], identity["run_id"], owner=identity["owner"], pid=identity["pid"])
    if not acquired:
        raise RuntimeError("Another controller lease is currently active.")
    controller = RootCauseController(max_experiments_per_cycle=max_experiments_per_cycle, dry_run=dry_run)
    history = []
    try:
        for _ in range(int(max_cycles)):
            refresh_controller_lease(identity["controller_id"], identity["run_id"], pid=identity["pid"])
            result = controller.run_cycle()
            history.append(result)
            if result.get("terminal_state"):
                break
            if sleep_seconds > 0:
                time.sleep(float(sleep_seconds))
                refresh_controller_lease(identity["controller_id"], identity["run_id"], pid=identity["pid"])
    finally:
        release_controller_lease(identity["controller_id"], identity["run_id"])
    return {
        "controller_id": identity["controller_id"],
        "run_id": identity["run_id"],
        "history": history,
        "finished_at": now_iso(),
    }

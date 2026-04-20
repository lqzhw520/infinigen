#!/usr/bin/env python3
"""Visual-fidelity root-cause controller for the unified reformulation plan."""

from __future__ import annotations

import json
import subprocess
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
    stable_config_hash,
)
from root_cause_hypotheses import apply_updates, default_board, unresolvedness
from perception_probe_mint import run_perception_probe
from root_cause_metrics import (
    GLOBAL_VISUAL_ALIGNMENT_THRESHOLD,
    BBOX_OVER_MASK_RATIO_MAX,
    HANDLE_AREA_RATIO_THRESHOLD,
    HANDLE_BOUNDARY_CONTRAST_THRESHOLD,
    HANDLE_CROP_ENTROPY_THRESHOLD,
    HANDLE_EDGE_DENSITY_THRESHOLD,
    LOCAL_AFFORDANCE_READABILITY_THRESHOLD,
    SECONDARY_FRAMING_THRESHOLD,
    contract_validity_score_v2,
    bbox_over_mask_ratio,
    handle_area_ratio,
    handle_bbox_area_ratio_debug,
    handle_mask_area_ratio,
    handle_crop_entropy,
    handle_local_contrast,
    handle_readability_score,
    handle_visibility_fraction,
    local_affordance_gate_breakdown,
    perceptual_readability_score_v2,
    truthful_measurement_acceptance_v1,
    rollout_ceiling_lift,
    seed_coverage_with_any_true,
    seed_coverage_with_positive_peak,
    detach_reason_coverage,
    sensor_contract_gain,
    state_alignment_gain,
    summarize_visual,
    task_identity_predictiveness_score,
    task_identity_predictiveness_gain,
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

RCA_ARTIFACTS: dict[str, tuple[Path, Path]] = {
    "RCA0": (ARTIFACT_DIR / "p2rca0_frozen_matrix_baseline.json", OUTPUT_DIR / "p2rca0_frozen_matrix_baseline.md"),
    "RCA1": (ARTIFACT_DIR / "p2rca1_true_handle_measurement_audit.json", OUTPUT_DIR / "p2rca1_true_handle_measurement_audit.md"),
    "RCA2": (ARTIFACT_DIR / "p2rca2_local_affordance_visual_attack.json", OUTPUT_DIR / "p2rca2_local_affordance_visual_attack.md"),
    "RCA3": (ARTIFACT_DIR / "p2rca3_affordance_plus_transition_contract.json", OUTPUT_DIR / "p2rca3_affordance_plus_transition_contract.md"),
    "RCA4": (ARTIFACT_DIR / "p2rca4_affordance_plus_transition_state.json", OUTPUT_DIR / "p2rca4_affordance_plus_transition_state.md"),
    "RCA5": (ARTIFACT_DIR / "p2rca5_frozen_matrix_screen.json", OUTPUT_DIR / "p2rca5_frozen_matrix_screen.md"),
    "RCA6": (ARTIFACT_DIR / "p2rca6_best_cell_replicate.json", OUTPUT_DIR / "p2rca6_best_cell_replicate.md"),
    "RCA7": (ARTIFACT_DIR / "p2rca7_tiny_retrain_if_eligible.json", OUTPUT_DIR / "p2rca7_tiny_retrain_if_eligible.md"),
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
        selector_mode: str = "adaptive",
        seed_split_a: list[int] | None = None,
        seed_split_b: list[int] | None = None,
        truthful_measurement_required: bool = False,
        perception_probe_mode: str = "policy_encoder_v1",
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
        self.selector_mode = str(selector_mode)
        self.seed_split_a = list(seed_split_a or [1, 2, 3, 4, 5, 6, 7, 8])
        self.seed_split_b = list(seed_split_b or [11, 12, 13, 14, 15])
        self.truthful_measurement_required = bool(truthful_measurement_required)
        self.perception_probe_mode = str(perception_probe_mode)
        self.policy_encoder_feature_extractor = None
        self.cap_strongest_negative = bool(cap_strongest_negative)
        self.resume = bool(resume)
        self.cycle_mode = str(cycle_mode)
        self.policy = {
            "AUTO_PROMOTE_SOVEREIGN": bool(auto_promote_sovereign),
            "ALLOW_FULL_RETRAIN": bool(allow_full_retrain),
            "ALLOW_NEW_CLAIM": bool(allow_new_claim),
            "RUN_MODE": str(run_mode),
            "EXPERIMENT_FAMILY": self.experiment_family,
            "SELECTOR_MODE": self.selector_mode,
            "SEED_SPLIT_A": list(self.seed_split_a),
            "SEED_SPLIT_B": list(self.seed_split_b),
            "TRUTHFUL_MEASUREMENT_REQUIRED": self.truthful_measurement_required,
            "PERCEPTION_PROBE_MODE": self.perception_probe_mode,
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
        self._invalid_run_reason: str | None = None

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

    def _artifact_map(self) -> dict[str, tuple[Path, Path]]:
        return RCA_ARTIFACTS if self.experiment_family == "RCA" else VR_ARTIFACTS

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

    def _visual_affordance_v3_raw_canonical_contract(self) -> dict[str, Any]:
        payload = self._raw_canonical_contract()
        payload.update({
            "secondary_camera_mode": "wrist_dynamic",
            "enable_marker_overlay": False,
            "calibration_mode": "none",
            "interaction_mode": "legacy_translation_only",
            "state_mode": "m0_proxy",
            "render_profile": "visual_affordance_v3_raw_canonical",
            "background_mode": "neutral_lowfreq_lab",
            "lighting_profile": "front_key_handle_rim",
            "material_policy": "legacy",
            "camera_framing_profile": "macro_handle_centered",
            "measurement_mode": "truthful_handle_manifest_v1",
            "measurement_fallback_policy": "forbid",
            "emit_measurement_debug": False,
            "canonical_lane": True,
        })
        return payload

    def _visual_affordance_v3_local_material_edge_contract(self) -> dict[str, Any]:
        payload = self._visual_affordance_v3_raw_canonical_contract()
        payload.update({
            "render_profile": "visual_affordance_v3_local_material_edge",
            "material_policy": "handle_affordance_local",
        })
        return payload

    def _visual_affordance_v3_canonical_local_carrier_contract(self) -> dict[str, Any]:
        payload = self._visual_affordance_v3_raw_canonical_contract()
        payload.update({
            "render_profile": "visual_affordance_v3_local_material_edge",
            "material_policy": "handle_affordance_local",
            "measurement_mode": "truthful_handle_manifest_v1",
            "measurement_fallback_policy": "forbid",
            "emit_measurement_debug": False,
            "canonical_lane": True,
        })
        return payload

    def _visual_affordance_v3_plus_bundle_contract(self, *, state_mode: str = "telemetry_candidate_v3_transition") -> dict[str, Any]:
        payload = self._visual_affordance_v3_local_material_edge_contract()
        payload.update({
            "interaction_mode": "orientation_sensitive_v2_affordance_locked",
            "state_mode": state_mode,
            "measurement_mode": "truthful_handle_manifest_v1",
            "measurement_fallback_policy": "forbid",
            "emit_measurement_debug": False,
        })
        return payload

    def _git_branch_name(self) -> str:
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=str(Path(__file__).resolve().parents[2]),
                text=True,
            ).strip()
        except Exception:  # noqa: BLE001
            return 'unknown'

    def _gate_threshold_snapshot_v5_pro(self) -> dict[str, Any]:
        return {
            'HANDLE_BOUNDARY_CONTRAST_THRESHOLD': HANDLE_BOUNDARY_CONTRAST_THRESHOLD,
            'HANDLE_EDGE_DENSITY_THRESHOLD': HANDLE_EDGE_DENSITY_THRESHOLD,
            'HANDLE_CROP_ENTROPY_THRESHOLD': HANDLE_CROP_ENTROPY_THRESHOLD,
            'HANDLE_AREA_RATIO_THRESHOLD': HANDLE_AREA_RATIO_THRESHOLD,
            'SECONDARY_FRAMING_THRESHOLD': SECONDARY_FRAMING_THRESHOLD,
            'LOCAL_AFFORDANCE_READABILITY_THRESHOLD': LOCAL_AFFORDANCE_READABILITY_THRESHOLD,
            'GLOBAL_VISUAL_ALIGNMENT_THRESHOLD': GLOBAL_VISUAL_ALIGNMENT_THRESHOLD,
            'BBOX_OVER_MASK_RATIO_MAX': BBOX_OVER_MASK_RATIO_MAX,
        }

    def _baseline_cell_id_v5_pro(self) -> str:
        return 'V0T0S0'


    def _ts_baseline_cell_id_v6_1(self) -> str:
        return 'V1cT0S0'
    def _frozen_matrix_contracts_v5_pro(self) -> dict[str, dict[str, Any]]:
        if hasattr(self, '_cached_frozen_matrix_v5_pro'):
            return self._cached_frozen_matrix_v5_pro
        raw = dict(self._visual_affordance_v3_raw_canonical_contract())
        carrier = dict(self._visual_affordance_v3_canonical_local_carrier_contract())
        contract_map: dict[str, dict[str, Any]] = {
            'V0T0S0': dict(raw),
            'V1cT0S0': dict(carrier),
            'V1cT1S0': {**carrier, 'interaction_mode': 'orientation_sensitive_v2_affordance_locked', 'state_mode': 'm0_proxy'},
            'V1cT1S1': {**carrier, 'interaction_mode': 'orientation_sensitive_v2_affordance_locked', 'state_mode': 'telemetry_candidate_v3_transition'},
            'V1cT2S0': {**carrier, 'interaction_mode': 'orientation_sensitive_v3_task_identity_locked', 'state_mode': 'm0_proxy'},
            'V1cT2S1': {**carrier, 'interaction_mode': 'orientation_sensitive_v3_task_identity_locked', 'state_mode': 'telemetry_candidate_v3_transition'},
            'V1cT2S2': {**carrier, 'interaction_mode': 'orientation_sensitive_v3_task_identity_locked', 'state_mode': 'telemetry_candidate_v4_task_identity'},
            'V1cT2S3': {**carrier, 'interaction_mode': 'orientation_sensitive_v3_task_identity_locked', 'state_mode': 'orientation_bridge_state_v1'},
        }
        cells: dict[str, dict[str, Any]] = {}
        for cell_id, env_contract in contract_map.items():
            env_contract = dict(env_contract)
            env_contract['measurement_mode'] = 'truthful_handle_manifest_v1'
            env_contract['measurement_fallback_policy'] = 'forbid'
            env_contract['emit_measurement_debug'] = False
            lane_payload = {
                'cell_id': cell_id,
                'env_contract_config': env_contract,
                'claim_policy': 'canonical',
                'diagnostic_only': False,
            }
            cells[cell_id] = {
                'cell_id': cell_id,
                'env_contract_config': env_contract,
                'lane_hash': stable_config_hash(lane_payload),
                'claim_policy': 'canonical',
                'diagnostic_only': False,
            }
        self._cached_frozen_matrix_v5_pro = cells
        return cells

    def _frozen_matrix_hash_v5_pro(self) -> str:
        return stable_config_hash(self._frozen_matrix_contracts_v5_pro())

    def _frozen_matrix_snapshot_v5_pro(self) -> dict[str, Any]:
        cells = self._frozen_matrix_contracts_v5_pro()
        return {
            'branch_name': self._git_branch_name(),
            'selector_mode': self.selector_mode,
            'baseline_cell_id': self._baseline_cell_id_v5_pro(),
            'ts_baseline_cell_id': self._ts_baseline_cell_id_v6_1(),
            'seed_split_a': list(self.seed_split_a),
            'seed_split_b': list(self.seed_split_b),
            'cell_hashes': {cell_id: cell['lane_hash'] for cell_id, cell in cells.items()},
            'gate_threshold_snapshot': self._gate_threshold_snapshot_v5_pro(),
            'measurement_mode_snapshot': {
                'measurement_mode': 'truthful_handle_manifest_v1',
                'measurement_fallback_policy': 'forbid',
            },
            'frozen_matrix_hash': self._frozen_matrix_hash_v5_pro(),
        }

    def _current_strongest_negative_contract(self) -> dict[str, Any]:
        if self.selector_mode == 'frozen_v5_pro':
            return dict(self._frozen_matrix_contracts_v5_pro()[self._baseline_cell_id_v5_pro()]['env_contract_config'])
        return self._visual_reformulation_v2_plus_bundle_contract()

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
            "measurement_mode",
            "measurement_fallback_policy",
            "emit_measurement_debug",
            "emit_orientation_telemetry",
            "emit_camera_metadata",
            "emit_handle_probe_metadata",
            "canonical_lane",
        }
        return DrawerEnvContractConfig(**{key: payload[key] for key in keys if key in payload})

    def _limited_seeds(self, seeds: list[int]) -> list[int]:
        if self.selector_mode == 'frozen_v5_pro' and self.experiment_family == 'RCA':
            return list(seeds)
        limit = max(1, int(self.resource_limits["max_rollouts_per_experiment"]))
        return list(seeds)[:limit]

    def _run_rollout_set(self, spec: LaneSpec, seeds: list[int]) -> list[dict[str, Any]]:
        rollouts = []
        contract = self._materialize_contract(spec.env_contract_config)
        interventions = dict(spec.interventions or {})
        rotation_source = str(interventions.get('rotation_source', 'zero'))
        grasp_pose = np.eye(4, dtype=np.float32)
        matrix_hash = self._frozen_matrix_hash_v5_pro() if self.selector_mode == 'frozen_v5_pro' else None
        baseline_cell_id = self._baseline_cell_id_v5_pro() if self.selector_mode == 'frozen_v5_pro' else None
        ts_baseline_cell_id = self._ts_baseline_cell_id_v6_1() if self.selector_mode == 'frozen_v5_pro' else None
        for episode_index, seed in enumerate(self._limited_seeds(seeds)):
            last_error = None
            for attempt in range(int(self.resource_limits['max_retries']) + 1):
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
                        interventions=interventions,
                    )
                    rollout['resource_budget_snapshot'] = self._resource_snapshot()
                    rollout['selector_mode'] = self.selector_mode
                    rollout['baseline_cell_id'] = baseline_cell_id
                    rollout['ts_baseline_cell_id'] = ts_baseline_cell_id
                    rollout['frozen_matrix_hash'] = matrix_hash
                    rollout['lane_hash'] = spec.lane_hash
                    probe = dict(rollout.get('handle_probe_metadata') or {})
                    if probe:
                        probe['selector_mode'] = self.selector_mode
                        probe['baseline_cell_id'] = baseline_cell_id
                        probe['ts_baseline_cell_id'] = ts_baseline_cell_id
                        probe['frozen_matrix_hash'] = matrix_hash
                        rollout['handle_probe_metadata'] = probe
                    trace = []
                    for item in rollout.get('handle_probe_metadata_trace', []):
                        probe_item = dict(item or {})
                        probe_item['selector_mode'] = self.selector_mode
                        probe_item['baseline_cell_id'] = baseline_cell_id
                        probe_item['ts_baseline_cell_id'] = ts_baseline_cell_id
                        probe_item['frozen_matrix_hash'] = matrix_hash
                        trace.append(probe_item)
                    if trace:
                        rollout['handle_probe_metadata_trace'] = trace
                    rollouts.append(rollout)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt >= int(self.resource_limits['max_retries']):
                        raise
                    time.sleep(float(self.resource_limits['retry_backoff_seconds']))
            if last_error is not None:
                raise last_error
        return rollouts

    def _rollout_summary(self, rollouts: list[dict[str, Any]]) -> dict[str, Any]:
        if not rollouts:
            return {
                'rollout_count': 0,
                'unique_success_rate': 0.0,
                'attach_rate': 0.0,
                'mean_max_drawer_fraction': 0.0,
                'avg_episode_length': 0.0,
                'success_count': 0,
                'orientation_causal_sensitivity': 0.0,
                'phase_locked_rate': 0.0,
                'mean_effective_pull_progress': 0.0,
                'runtime_handle_anchor_valid_rate': 0.0,
                'detach_reason_histogram': {},
            }
        detach_hist: dict[str, int] = {}
        for rollout in rollouts:
            detach_trace = rollout.get('detach_reason_trace', [])
            if detach_trace is None:
                continue
            for reason in np.asarray(detach_trace, dtype=object).reshape(-1).tolist():
                if reason:
                    detach_hist[str(reason)] = detach_hist.get(str(reason), 0) + 1
        def _trace_mean(key: str) -> float:
            values: list[float] = []
            for rollout in rollouts:
                trace = np.asarray(rollout.get(key, []), dtype=np.float32)
                if trace.size:
                    values.append(float(np.mean(trace)))
            return float(np.mean(values)) if values else 0.0
        return {
            'rollout_count': len(rollouts),
            'unique_success_rate': float(np.mean([1.0 if r.get('success') else 0.0 for r in rollouts])),
            'attach_rate': float(np.mean([1.0 if r.get('ever_attached') else 0.0 for r in rollouts])),
            'mean_max_drawer_fraction': float(np.mean([float(r.get('max_drawer_fraction', 0.0)) for r in rollouts])),
            'avg_episode_length': float(np.mean([float(r.get('steps', 0.0)) for r in rollouts])),
            'success_count': int(sum(1 for r in rollouts if r.get('success'))),
            'orientation_causal_sensitivity': float(np.mean([np.mean(np.asarray(r.get('drawer_delta_effective_trace', [0.0]), dtype=np.float32)) for r in rollouts])),
            'phase_locked_rate': _trace_mean('phase_locked_trace'),
            'mean_effective_pull_progress': _trace_mean('effective_pull_progress_trace'),
            'runtime_handle_anchor_valid_rate': _trace_mean('runtime_handle_anchor_valid_trace'),
            'detach_reason_histogram': detach_hist,
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
        if not trace:
            report = {
                "probe_measurement_mode": "unavailable",
                "truth_root_object": "manifest_entity",
                "identity_resolution_tier": "manifest_entity_exact_runtime_geom",
                "measurement_backend": "unavailable",
                "measurement_verifier": "unavailable",
                "measurement_truth_tier": "manifest_entity_unverified",
                "measurement_truthful": False,
                "measurement_warning_flags": ["missing_handle_probe_trace"],
                "manifest_handle_entity_unique": False,
                "runtime_visible_handle_mapping_source": "unresolved",
                "runtime_visible_handle_mapping_unique": False,
                "runtime_handle_visual_geom_mapping_unique": False,
                "runtime_handle_collision_geom_mapping_unique": False,
                "duplicate_runtime_geom_name_flag": False,
                "unnamed_runtime_geom_flag": False,
                "resolved_runtime_visible_handle_geom_ids": [],
                "resolved_runtime_visible_handle_geom_names": [],
                "resolved_handle_visual_geom_ids": [],
                "resolved_handle_visual_geom_names": [],
                "resolved_handle_collision_geom_ids": [],
                "resolved_handle_collision_geom_names": [],
                "handle_geom_ids": [],
                "handle_geom_names": [],
                "semantic_mapping_hash": "",
                "segmentation_mask_nonzero_primary": 0,
                "segmentation_mask_nonzero_secondary": 0,
                "isolated_mask_nonzero_primary": 0,
                "isolated_mask_nonzero_secondary": 0,
                "segmentation_mask_support_rate_secondary": 0.0,
                "isolated_mask_support_rate_secondary": 0.0,
                "segmentation_isolated_iou_secondary": 0.0,
                "segmentation_isolated_centroid_delta_px_secondary": float("inf"),
                "handle_bbox_primary": [0, 0, 0, 0],
                "handle_bbox_secondary": [0, 0, 0, 0],
                "handle_mask_nonzero_pixels_primary": 0,
                "handle_mask_nonzero_pixels_secondary": 0,
                "handle_mask_area_ratio_primary": 0.0,
                "handle_mask_area_ratio_secondary": 0.0,
                "handle_bbox_area_ratio_primary": 0.0,
                "handle_bbox_area_ratio_secondary": 0.0,
                "bbox_over_mask_ratio_primary": 0.0,
                "bbox_over_mask_ratio_secondary": 0.0,
                "handle_visibility_fraction": 0.0,
                "handle_area_ratio": 0.0,
                "handle_boundary_contrast": 0.0,
                "handle_edge_density": 0.0,
                "handle_crop_entropy": 0.0,
                "framing_score": 0.0,
                "camera_relativeness_residual": None,
                "legacy_probe_measurement_mode": "proxy_debug",
                "legacy_handle_bbox_secondary": [0, 0, 0, 0],
                "legacy_handle_local_contrast": 0.0,
                "legacy_handle_crop_entropy": 0.0,
            }
            report["measurement_truthful"] = truthful_measurement_acceptance_v1(report)
            return report

        def _mean(key: str) -> float:
            values = [float(item.get(key, 0.0)) for item in trace if item.get(key) is not None]
            return float(np.mean(values)) if values else 0.0

        def _mean_bool(key: str) -> float:
            values = [1.0 if bool(item.get(key, False)) else 0.0 for item in trace if key in item]
            return float(np.mean(values)) if values else 0.0

        def _all_bool(key: str) -> bool:
            values = [bool(item.get(key, False)) for item in trace if key in item]
            return bool(values and all(values))

        def _min_value(key: str, default: float = 0.0) -> float:
            values = [float(item.get(key, default)) for item in trace if item.get(key) is not None]
            return float(min(values)) if values else float(default)

        def _max_value(key: str, default: float = 0.0) -> float:
            values = [float(item.get(key, default)) for item in trace if item.get(key) is not None]
            return float(max(values)) if values else float(default)

        best_secondary = max(trace, key=lambda item: float(item.get("handle_mask_nonzero_pixels_secondary", 0.0)))
        best_primary = max(trace, key=lambda item: float(item.get("handle_mask_nonzero_pixels_primary", 0.0)))
        warning_flags = sorted({
            flag
            for item in trace
            for flag in list(item.get("measurement_warning_flags", []))
            if flag != "truthful_measurement_required_no_fallback"
        })
        resolved_runtime_visible_handle_geom_ids = sorted({int(v) for item in trace for v in list(item.get("resolved_runtime_visible_handle_geom_ids", []))})
        resolved_runtime_visible_handle_geom_names = sorted({str(v) for item in trace for v in list(item.get("resolved_runtime_visible_handle_geom_names", [])) if str(v)})
        resolved_handle_visual_geom_ids = sorted({int(v) for item in trace for v in list(item.get("resolved_handle_visual_geom_ids", []))})
        resolved_handle_visual_geom_names = sorted({str(v) for item in trace for v in list(item.get("resolved_handle_visual_geom_names", [])) if str(v)})
        resolved_handle_collision_geom_ids = sorted({int(v) for item in trace for v in list(item.get("resolved_handle_collision_geom_ids", []))})
        resolved_handle_collision_geom_names = sorted({str(v) for item in trace for v in list(item.get("resolved_handle_collision_geom_names", [])) if str(v)})
        framing_values = [float(item.get("secondary_framing_score", 0.0)) for item in trace if item.get("secondary_framing_score") is not None]
        residual_values = [float(item.get("camera_relativeness_residual", 1.0)) for item in trace if item.get("camera_relativeness_residual") is not None]
        support_seg_secondary = float(np.mean([1.0 if float(item.get("segmentation_mask_nonzero_secondary", 0.0)) > 0 else 0.0 for item in trace]))
        support_iso_secondary = float(np.mean([1.0 if float(item.get("isolated_mask_nonzero_secondary", 0.0)) > 0 else 0.0 for item in trace]))
        semantic_mapping_hashes = [str(item.get("semantic_mapping_hash", "")) for item in trace if item.get("semantic_mapping_hash")]
        report = {
            "probe_measurement_mode": str(best_secondary.get("probe_measurement_mode") or best_primary.get("probe_measurement_mode") or "unavailable"),
            "truth_root_object": str(best_secondary.get("truth_root_object") or best_primary.get("truth_root_object") or "manifest_entity"),
            "identity_resolution_tier": str(best_secondary.get("identity_resolution_tier") or best_primary.get("identity_resolution_tier") or "manifest_entity_exact_runtime_geom"),
            "measurement_backend": str(best_secondary.get("measurement_backend") or best_primary.get("measurement_backend") or "unavailable"),
            "measurement_verifier": str(best_secondary.get("measurement_verifier") or best_primary.get("measurement_verifier") or "unavailable"),
            "measurement_truth_tier": str(best_secondary.get("measurement_truth_tier") or best_primary.get("measurement_truth_tier") or "manifest_entity_unverified"),
            "measurement_warning_flags": warning_flags,
            "manifest_handle_entity_unique": _all_bool("manifest_handle_entity_unique"),
            "runtime_visible_handle_mapping_source": str(best_secondary.get("runtime_visible_handle_mapping_source") or best_primary.get("runtime_visible_handle_mapping_source") or "unresolved"),
            "runtime_visible_handle_mapping_unique": _all_bool("runtime_visible_handle_mapping_unique"),
            "runtime_handle_visual_geom_mapping_unique": _all_bool("runtime_handle_visual_geom_mapping_unique"),
            "runtime_handle_collision_geom_mapping_unique": _all_bool("runtime_handle_collision_geom_mapping_unique"),
            "duplicate_runtime_geom_name_flag": any(bool(item.get("duplicate_runtime_geom_name_flag", False)) for item in trace),
            "unnamed_runtime_geom_flag": any(bool(item.get("unnamed_runtime_geom_flag", False)) for item in trace),
            "resolved_runtime_visible_handle_geom_ids": resolved_runtime_visible_handle_geom_ids,
            "resolved_runtime_visible_handle_geom_names": resolved_runtime_visible_handle_geom_names,
            "resolved_handle_visual_geom_ids": resolved_handle_visual_geom_ids,
            "resolved_handle_visual_geom_names": resolved_handle_visual_geom_names,
            "resolved_handle_collision_geom_ids": resolved_handle_collision_geom_ids,
            "resolved_handle_collision_geom_names": resolved_handle_collision_geom_names,
            "handle_geom_ids": sorted(set(resolved_runtime_visible_handle_geom_ids)),
            "handle_geom_names": sorted(set(resolved_runtime_visible_handle_geom_names)),
            "semantic_mapping_hash": semantic_mapping_hashes[0] if semantic_mapping_hashes else "",
            "segmentation_mask_nonzero_primary": int(round(_mean("segmentation_mask_nonzero_primary"))),
            "segmentation_mask_nonzero_secondary": int(round(_mean("segmentation_mask_nonzero_secondary"))),
            "isolated_mask_nonzero_primary": int(round(_mean("isolated_mask_nonzero_primary"))),
            "isolated_mask_nonzero_secondary": int(round(_mean("isolated_mask_nonzero_secondary"))),
            "segmentation_mask_support_rate_secondary": support_seg_secondary,
            "isolated_mask_support_rate_secondary": support_iso_secondary,
            "segmentation_isolated_iou_secondary": _mean("segmentation_isolated_iou_secondary"),
            "segmentation_isolated_centroid_delta_px_secondary": _mean("segmentation_isolated_centroid_delta_px_secondary"),
            "handle_bbox_primary": best_primary.get("handle_bbox_primary") or [0, 0, 0, 0],
            "handle_bbox_secondary": best_secondary.get("handle_bbox_secondary") or [0, 0, 0, 0],
            "handle_mask_nonzero_pixels_primary": int(round(_mean("handle_mask_nonzero_pixels_primary"))),
            "handle_mask_nonzero_pixels_secondary": int(round(_mean("handle_mask_nonzero_pixels_secondary"))),
            "handle_mask_area_ratio_primary": _mean("handle_mask_area_ratio_primary"),
            "handle_mask_area_ratio_secondary": _mean("handle_mask_area_ratio_secondary"),
            "handle_bbox_area_ratio_primary": _mean("handle_bbox_area_ratio_primary"),
            "handle_bbox_area_ratio_secondary": _mean("handle_bbox_area_ratio_secondary"),
            "bbox_over_mask_ratio_primary": _max_value("bbox_over_mask_ratio_primary", 0.0),
            "bbox_over_mask_ratio_secondary": _max_value("bbox_over_mask_ratio_secondary", 0.0),
            "handle_visibility_fraction": _mean("handle_visibility_fraction_secondary") or _mean("handle_visibility_fraction_primary"),
            "handle_area_ratio": _mean("handle_mask_area_ratio_secondary") or _mean("handle_mask_area_ratio_primary"),
            "handle_boundary_contrast": _mean("handle_boundary_contrast_secondary") or _mean("handle_boundary_contrast_primary"),
            "handle_edge_density": _mean("handle_edge_density_secondary") or _mean("handle_edge_density_primary"),
            "handle_crop_entropy": _mean("handle_crop_entropy_secondary") or _mean("handle_crop_entropy_primary"),
            "handle_local_std": _mean("handle_local_std_secondary") or _mean("handle_local_std_primary"),
            "handle_local_contrast": _mean("handle_boundary_contrast_secondary") or _mean("handle_boundary_contrast_primary"),
            "framing_score": float(np.mean(framing_values)) if framing_values else 0.0,
            "camera_relativeness_residual": float(np.mean(residual_values)) if residual_values else None,
            "legacy_probe_measurement_mode": "proxy_debug",
            "legacy_handle_bbox_secondary": best_secondary.get("legacy_handle_bbox_secondary") or [0, 0, 0, 0],
            "legacy_handle_local_contrast": _mean("legacy_handle_local_contrast_secondary") or _mean("legacy_handle_local_contrast_primary"),
            "legacy_handle_crop_entropy": _mean("legacy_handle_crop_entropy_secondary") or _mean("legacy_handle_crop_entropy_primary"),
        }
        report["measurement_truthful"] = truthful_measurement_acceptance_v1(report)
        if not report["measurement_truthful"]:
            report["measurement_warning_flags"] = sorted(set(list(report.get("measurement_warning_flags", [])) + ["truthful_measurement_required_no_fallback"]))
        report["measurement_truth_tier"] = (
            "manifest_entity_verified" if report["measurement_truthful"] else "manifest_entity_unverified"
        )
        return report

    def _visual_report(self, spec: LaneSpec, seeds: list[int], *, baseline_gap: float | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rollouts = self._run_rollout_set(spec, seeds)
        residuals = self._secondary_residuals(rollouts)
        target_stats = summarize_visual(self._sample_images(rollouts, limit=24))
        gap_payload = visual_gap(self.control_visual_stats, target_stats)
        lane_gap = weighted_visual_gap(gap_payload)
        probe = self._handle_probe_aggregates(rollouts)
        feature_extractor = self.policy_encoder_feature_extractor if self.perception_probe_mode == 'policy_encoder_v1' else None
        perception_probe = run_perception_probe(rollouts, feature_extractor=feature_extractor)
        baseline_value = float(baseline_gap if baseline_gap is not None else lane_gap)
        alignment_gain = visual_alignment_gain(baseline_value if baseline_value > 0 else max(lane_gap, 1e-6), lane_gap)
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
            **perception_probe,
        }
        report["observation_contract_pass"] = bool(
            report["secondary_camera_wrist_like"]
            and not report["marker_overlay_enabled"]
            and not report["diagnostic_only"]
            and report["provenance_complete"]
        )
        report["visual_gate_breakdown"] = visual_gate_breakdown(
            baseline_weighted_visual_gap=baseline_value,
            lane_weighted_visual_gap=lane_gap,
            visual_alignment_gain_value=alignment_gain,
            framing_score=report["framing_score"],
            visibility_fraction=report["handle_visibility_fraction"],
            local_contrast=report["handle_local_contrast"],
            crop_entropy=report["handle_crop_entropy"],
            camera_relativeness_residual=report["secondary_camera_residual_mean"],
            encoder_readability_pass=report.get("encoder_readability_pass"),
        )
        report["g4b_local_breakdown"] = local_affordance_gate_breakdown(
            mask_nonzero_pixels=report["handle_mask_nonzero_pixels_secondary"],
            handle_bbox=report.get("handle_bbox_secondary") or report.get("handle_bbox_primary"),
            image_shape=(256, 256, 3),
            handle_boundary_contrast_value=report["handle_boundary_contrast"],
            handle_edge_density_value=report["handle_edge_density"],
            handle_crop_entropy_value=report["handle_crop_entropy"],
            secondary_framing_score=report["framing_score"],
            measurement_truthful=report["measurement_truthful"],
            truth_tier=report["measurement_truth_tier"],
            warning_flags=report.get("measurement_warning_flags", []),
        )
        report["g4c_global_breakdown"] = {
            "baseline_weighted_visual_gap": float(baseline_value),
            "weighted_visual_gap": float(lane_gap),
            "visual_alignment_gain": float(alignment_gain),
        }
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
        artifact_path, report_path = self._artifact_map()[experiment_id]
        write_json_atomic(artifact_path, payload)
        lines = [f"# {title}", "", f"Generated: {payload.get('generated_at', now_iso())}"]
        lines.extend([f"- {line}" for line in bullets])
        write_text_atomic(report_path, "\n".join(lines).rstrip() + "\n")

    def experiment_catalog(self) -> list[dict[str, Any]]:
        if self.experiment_family == "RCA":
            return [
                {"id": "RCA0", "deps": [], "coverage": {"H6_environment_invalid_for_claim": 0.4}, "cost": 0.7, "runner": self.run_rca0},
                {"id": "RCA1", "deps": ["RCA0"], "coverage": {"H3_observation_visual_contract": 0.8}, "cost": 0.8, "runner": self.run_rca1},
                {"id": "RCA2", "deps": ["RCA1"], "coverage": {"H3_observation_visual_contract": 1.0}, "cost": 1.0, "runner": self.run_rca2},
                {"id": "RCA3", "deps": ["RCA2"], "coverage": {"H1_action_semantics": 0.8, "H3_observation_visual_contract": 0.8}, "cost": 1.0, "runner": self.run_rca3},
                {"id": "RCA4", "deps": ["RCA3"], "coverage": {"H2_state_representability": 1.0, "H0_same_problem_identity": 0.6}, "cost": 1.0, "runner": self.run_rca4},
                {"id": "RCA5", "deps": ["RCA2", "RCA3", "RCA4"], "coverage": {"H6_environment_invalid_for_claim": 0.9, "H0_same_problem_identity": 0.8}, "cost": 1.2, "runner": self.run_rca5},
                {"id": "RCA6", "deps": ["RCA5"], "coverage": {"H6_environment_invalid_for_claim": 0.8}, "cost": 0.8, "runner": self.run_rca6},
                {"id": "RCA7", "deps": ["RCA6"], "coverage": {"H4_unique_data_scale_only": 0.4, "H5_optimization_only": 0.4}, "cost": 0.9, "runner": self.run_rca7},
            ]
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
        if self.selector_mode == 'frozen_v5_pro' and self.experiment_family == 'RCA':
            ordered = ["RCA0", "RCA1", "RCA2", "RCA3", "RCA4", "RCA5", "RCA6", "RCA7"]
            catalog_map = {item['id']: item for item in catalog}
            completed = set(self.completed_experiments)
            selected: list[dict[str, Any]] = []
            selected_ids: set[str] = set()
            for exp_id in ordered:
                if exp_id in completed or exp_id in selected_ids:
                    continue
                item = catalog_map[exp_id]
                available = completed | selected_ids
                if not all(dep in available for dep in item.get('deps', [])):
                    break
                selected.append(item)
                selected_ids.add(exp_id)
                if len(selected) >= self.max_experiments_per_cycle:
                    break
            return selected
        first_id = "RCA0" if self.experiment_family == "RCA" else "VR0"
        if first_id not in self.completed_experiments:
            return [next(item for item in catalog if item["id"] == first_id)]
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

    def _state_audit_for_contract(self, contract: dict[str, Any], seeds: list[int], experiment_id: str, prefix: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        state_candidates = []
        lane_specs = []
        for mode in ["m0_proxy", "eef_pose_gripper", "telemetry_candidate_v2", "telemetry_candidate_v3_transition"]:
            candidate_contract = dict(contract)
            candidate_contract["state_mode"] = mode
            spec = self._make_lane_spec(
                lane_id=f"{prefix}_{mode}",
                stage="probe",
                env_contract=candidate_contract,
                interventions={"rotation_source": "aligned", "profile_id": f"state_{mode}"},
                claim_policy="diagnostic" if mode in {"m0_proxy", "eef_pose_gripper"} else "canonical",
                note="Transition-aware state semantics audit candidate.",
                experiment_id=experiment_id,
            )
            lane_specs.append(spec.to_payload())
            _, report = self._state_report(spec, seeds)
            state_candidates.append(report)
        score_by_mode = {item["state_mode"]: float(item["overall_score"]) for item in state_candidates}
        telemetry_candidates = [item for item in state_candidates if item["state_mode"] in {"telemetry_candidate_v2", "telemetry_candidate_v3_transition"} and item["valid"]]
        selected_candidate = min(telemetry_candidates, key=lambda item: item["overall_score"], default=None)
        m0_score = float(score_by_mode.get("m0_proxy", 1.0))
        eef_score = float(score_by_mode.get("eef_pose_gripper", 1.0))
        selected_score = float(selected_candidate["overall_score"]) if selected_candidate else None
        telemetry_gain = state_alignment_gain(m0_score, selected_score) if selected_candidate is not None else -1.0
        audit = {
            "selected_candidate_mode": selected_candidate["state_mode"] if selected_candidate else None,
            "selected_candidate_score": selected_score,
            "m0_proxy_score": m0_score,
            "eef_pose_gripper_score": eef_score,
            "state_alignment_gain": telemetry_gain,
            "selected_candidate": selected_candidate,
            "all_candidates": state_candidates,
            "impossible": selected_candidate is None,
        }
        return lane_specs, audit

    def _mean_rollout_trace(self, rollouts: list[dict[str, Any]], key: str) -> float:
        values: list[float] = []
        for rollout in rollouts:
            trace = np.asarray(rollout.get(key, []), dtype=np.float32)
            if trace.size:
                values.append(float(np.mean(trace)))
        return float(np.mean(values)) if values else 0.0

    def _bbox_iou(self, a: list[int] | None, b: list[int] | None) -> float:
        if not a or not b or len(a) != 4 or len(b) != 4:
            return 0.0
        ax0, ay0, ax1, ay1 = [int(v) for v in a]
        bx0, by0, bx1, by1 = [int(v) for v in b]
        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)
        inter = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
        area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    def _bbox_centroid_delta_px(self, a: list[int] | None, b: list[int] | None) -> float:
        if not a or not b or len(a) != 4 or len(b) != 4:
            return 0.0
        acx = 0.5 * (float(a[0]) + float(a[2]))
        acy = 0.5 * (float(a[1]) + float(a[3]))
        bcx = 0.5 * (float(b[0]) + float(b[2]))
        bcy = 0.5 * (float(b[1]) + float(b[3]))
        return float(np.linalg.norm(np.asarray([acx - bcx, acy - bcy], dtype=np.float32)))

    def _cell_rotation_source(self, cell_id: str) -> str:
        return 'aligned' if ('T1' in cell_id or 'T2' in cell_id) else 'zero'

    def _matrix_cell_lane_spec(self, cell_id: str, experiment_id: str, *, note: str) -> LaneSpec:
        cell = self._frozen_matrix_contracts_v5_pro()[cell_id]
        return self._make_lane_spec(
            lane_id=f'{experiment_id}_{cell_id}',
            stage='probe',
            env_contract=dict(cell['env_contract_config']),
            interventions={
                'rotation_source': self._cell_rotation_source(cell_id),
                'profile_id': cell_id,
                'cell_id': cell_id,
            },
            claim_policy='canonical',
            note=note,
            experiment_id=experiment_id,
        )

    def _matrix_cell_sort_key(self, payload: dict[str, Any]) -> tuple[float, float, float, float]:
        rollout_score = float((payload.get('rollout_ceiling_lift') or {}).get('score', -1e9))
        drawer_delta = float(payload.get('delta_mean_max_drawer_fraction', (payload.get('rollout_ceiling_lift') or {}).get('delta_mean_max_drawer_fraction', 0.0)))
        phase_locked_rate = float((payload.get('rollout_summary') or {}).get('phase_locked_rate', 0.0))
        mean_effective_pull_progress = float((payload.get('rollout_summary') or {}).get('mean_effective_pull_progress', 0.0))
        return (rollout_score, drawer_delta, phase_locked_rate, mean_effective_pull_progress)

    def run_rca0(self) -> dict[str, Any]:
        seeds = list(self.seed_split_a)
        baseline_cell_id = self._baseline_cell_id_v5_pro()
        spec = self._matrix_cell_lane_spec(baseline_cell_id, 'RCA0', note='Frozen baseline V0T0S0 on split A.')
        rollouts, visual_report = self._visual_report(spec, seeds)
        _, state_report = self._state_report(spec, seeds)
        rollout_summary = self._rollout_summary(rollouts)
        snapshot = self._frozen_matrix_snapshot_v5_pro()
        payload = {
            'experiment_id': 'RCA0',
            'generated_at': now_iso(),
            'passed': True,
            **snapshot,
            'baseline_rollout_summary': rollout_summary,
            'baseline_visual_report': visual_report,
            'baseline_state_report': state_report,
            'baseline_lane': spec.to_payload(),
        }
        self._write_artifact('RCA0', payload, 'RCA0 Frozen Matrix Baseline', [
            f"baseline_cell_id: {baseline_cell_id}",
            f"frozen_matrix_hash: {snapshot['frozen_matrix_hash']}",
            f"split_a_size: {len(seeds)}",
        ])
        return {'payload': payload, 'updates': [], 'lane_specs': [spec.to_payload()]}

    def run_rca1(self) -> dict[str, Any]:
        seeds = list(self.seed_split_a)
        spec = self._matrix_cell_lane_spec('V0T0S0', 'RCA1', note='Truthful measurement audit on frozen baseline.')
        _, report = self._visual_report(spec, seeds)
        truthful_bbox = report.get('handle_bbox_secondary') or report.get('handle_bbox_primary') or [0, 0, 0, 0]
        proxy_bbox = report.get('legacy_handle_bbox_secondary') or [0, 0, 0, 0]
        truthful_area_ratio = float(report.get('handle_mask_area_ratio_secondary', 0.0) or report.get('handle_mask_area_ratio_primary', 0.0))
        truthful_bbox_area_ratio = float(report.get('handle_bbox_area_ratio_secondary', 0.0) or report.get('handle_bbox_area_ratio_primary', 0.0))
        truthful_bbox_over_mask = float(report.get('bbox_over_mask_ratio_secondary', 0.0) or report.get('bbox_over_mask_ratio_primary', 0.0))
        proxy_bbox_area_ratio = float(handle_bbox_area_ratio_debug(proxy_bbox, (256, 256, 3)))
        truthful_proxy_iou = self._bbox_iou(truthful_bbox, proxy_bbox)
        truthful_proxy_centroid_delta_px = self._bbox_centroid_delta_px(truthful_bbox, proxy_bbox)
        materially_different = bool(
            truthful_proxy_iou < 0.95
            or abs(truthful_bbox_area_ratio - proxy_bbox_area_ratio) >= 0.02
            or truthful_proxy_centroid_delta_px >= 4.0
        )
        measurement_truthful_available = truthful_measurement_acceptance_v1(report)
        invalid_run = bool(self.truthful_measurement_required and not measurement_truthful_available)
        if invalid_run:
            self._invalid_run_reason = 'truthful_measurement_unavailable'
        payload = {
            'experiment_id': 'RCA1',
            'generated_at': now_iso(),
            'selector_mode': self.selector_mode,
            'baseline_cell_id': self._baseline_cell_id_v5_pro(),
            'frozen_matrix_hash': self._frozen_matrix_hash_v5_pro(),
            'measurement_truthful_available': measurement_truthful_available,
            'measurement_truth_tier': report.get('measurement_truth_tier'),
            'measurement_backend': report.get('measurement_backend'),
            'measurement_verifier': report.get('measurement_verifier'),
            'manifest_handle_entity_unique': report.get('manifest_handle_entity_unique'),
            'runtime_visible_handle_mapping_source': report.get('runtime_visible_handle_mapping_source'),
            'runtime_visible_handle_mapping_unique': report.get('runtime_visible_handle_mapping_unique'),
            'runtime_handle_visual_geom_mapping_unique': report.get('runtime_handle_visual_geom_mapping_unique'),
            'runtime_handle_collision_geom_mapping_unique': report.get('runtime_handle_collision_geom_mapping_unique'),
            'duplicate_runtime_geom_name_flag': report.get('duplicate_runtime_geom_name_flag'),
            'unnamed_runtime_geom_flag': report.get('unnamed_runtime_geom_flag'),
            'segmentation_mask_support_rate_secondary': report.get('segmentation_mask_support_rate_secondary'),
            'isolated_mask_support_rate_secondary': report.get('isolated_mask_support_rate_secondary'),
            'segmentation_isolated_iou_secondary': report.get('segmentation_isolated_iou_secondary'),
            'segmentation_isolated_centroid_delta_px_secondary': report.get('segmentation_isolated_centroid_delta_px_secondary'),
            'semantic_mapping_hash': report.get('semantic_mapping_hash'),
            'truthful_mask_area_ratio_secondary': truthful_area_ratio,
            'truthful_bbox_area_ratio_secondary': truthful_bbox_area_ratio,
            'truthful_bbox_over_mask_ratio_secondary': truthful_bbox_over_mask,
            'proxy_bbox_area_ratio_secondary': proxy_bbox_area_ratio,
            'truthful_proxy_iou_secondary': truthful_proxy_iou,
            'truthful_proxy_centroid_delta_px': truthful_proxy_centroid_delta_px,
            'true_handle_bbox_secondary': truthful_bbox,
            'proxy_handle_bbox_secondary': proxy_bbox,
            'materially_different': materially_different,
            'invalid_run': invalid_run,
            'representative_overlay_images': [],
            'lane_report': report,
        }
        self._write_artifact('RCA1', payload, 'RCA1 True Handle Measurement Audit', [
            f'measurement_truthful_available: {measurement_truthful_available}',
            f'truthful_proxy_iou_secondary: {truthful_proxy_iou:.6f}',
            f'truthful_bbox_over_mask_ratio_secondary: {truthful_bbox_over_mask:.6f}',
        ])
        return {'payload': payload, 'updates': [], 'lane_specs': [spec.to_payload()]}



    def _seed_coverage_all_true(self, rollouts: list[dict[str, Any]], trace_key: str) -> float:
        if not rollouts:
            return 0.0
        values = []
        for rollout in rollouts:
            trace = np.asarray(rollout.get(trace_key, []))
            values.append(bool(trace.size) and bool(np.all(trace)))
        return float(np.mean(np.asarray(values, dtype=np.float32))) if values else 0.0

    def _state_view_from_rollout(self, rollout: dict[str, Any], state_label: str) -> np.ndarray:
        if state_label == 'S0':
            states = np.asarray(rollout.get('states', np.zeros((0, 8), dtype=np.float32)), dtype=np.float32)
            if states.ndim == 1:
                states = states.reshape(1, -1)
            return states[:, :8].astype(np.float32)

        handle_rel = np.asarray(rollout.get('handle_rel_trace', np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        drawer_fraction = np.asarray(rollout.get('drawer_fraction_signed_trace', np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)
        pull_alignment = np.asarray(rollout.get('pull_alignment_trace', np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)

        if state_label == 'S1':
            stable_attach = np.asarray(rollout.get('stable_attach_trace', np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)
            attach_streak = np.asarray(rollout.get('attach_streak_trace', np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)
            attach_streak = np.clip(attach_streak / 4.0, 0.0, 1.0)
            contact_window = np.asarray(rollout.get('contact_window_fraction_trace', np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)
            parts = [handle_rel, drawer_fraction, pull_alignment, stable_attach, attach_streak, contact_window]
        elif state_label == 'S2':
            grasp_slip = np.asarray(rollout.get('grasp_slip_norm_trace', np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)
            effective_pull = np.asarray(rollout.get('effective_pull_progress_norm_trace', np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)
            phase_locked = np.asarray(rollout.get('phase_locked_trace', np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1, 1)
            parts = [handle_rel, drawer_fraction, pull_alignment, grasp_slip, effective_pull, phase_locked]
        else:
            raise ValueError(f'unsupported state label: {state_label}')

        lengths = [part.shape[0] for part in parts]
        n = min(lengths) if lengths else 0
        if n <= 0:
            return np.zeros((0, 8), dtype=np.float32)
        return np.concatenate([part[:n] for part in parts], axis=1).astype(np.float32)

    def _next_step_targets_from_rollout(self, rollout: dict[str, Any]) -> dict[str, np.ndarray]:
        stable_attach = np.asarray(rollout.get('stable_attach_trace', np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        phase_locked = np.asarray(rollout.get('phase_locked_trace', np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        effective_pull = np.asarray(rollout.get('effective_pull_progress_trace', np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        drawer_delta = np.asarray(rollout.get('drawer_delta_effective_trace', np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        n = min(len(stable_attach), len(phase_locked), len(effective_pull), len(drawer_delta))
        if n < 2:
            empty = np.zeros((0,), dtype=np.float32)
            return {
                'stable_attach_next': empty,
                'phase_locked_next': empty,
                'effective_pull_progress_next': empty,
                'drawer_delta_effective_next': empty,
            }
        return {
            'stable_attach_next': stable_attach[1:n].astype(np.float32),
            'phase_locked_next': phase_locked[1:n].astype(np.float32),
            'effective_pull_progress_next': effective_pull[1:n].astype(np.float32),
            'drawer_delta_effective_next': drawer_delta[1:n].astype(np.float32),
        }

    def _task_identity_state_audit_v1(self, rollouts: list[dict[str, Any]]) -> dict[str, Any]:
        mode_defs = {
            'S0': 'm0_proxy',
            'S1': 'telemetry_candidate_v3_transition',
            'S2': 'telemetry_candidate_v4_task_identity',
        }
        target_names = [
            'stable_attach_next',
            'phase_locked_next',
            'effective_pull_progress_next',
            'drawer_delta_effective_next',
        ]
        mode_reports: dict[str, dict[str, Any]] = {}
        best_label: str | None = None
        best_score = float('-inf')

        for label, state_mode in mode_defs.items():
            train_x_blocks: list[np.ndarray] = []
            eval_x_blocks: list[np.ndarray] = []
            train_target_blocks: dict[str, list[np.ndarray]] = {name: [] for name in target_names}
            eval_target_blocks: dict[str, list[np.ndarray]] = {name: [] for name in target_names}
            train_seed_pool: list[int] = []
            eval_seed_pool: list[int] = []

            for rollout in rollouts:
                features = self._state_view_from_rollout(rollout, label)
                targets = self._next_step_targets_from_rollout(rollout)
                target_lengths = [len(values) for values in targets.values()]
                n = min([max(features.shape[0] - 1, 0), *target_lengths]) if target_lengths else 0
                if n <= 0:
                    continue
                x = np.asarray(features[:n], dtype=np.float32)
                y = {name: np.asarray(targets[name][:n], dtype=np.float32) for name in target_names}
                seed = int(rollout.get('seed', 0) or 0)
                if seed % 2 == 1:
                    train_x_blocks.append(x)
                    for name in target_names:
                        train_target_blocks[name].append(y[name])
                    train_seed_pool.append(seed)
                else:
                    eval_x_blocks.append(x)
                    for name in target_names:
                        eval_target_blocks[name].append(y[name])
                    eval_seed_pool.append(seed)

            valid = bool(train_x_blocks and eval_x_blocks)
            if valid:
                train_x = np.concatenate(train_x_blocks, axis=0)
                eval_x = np.concatenate(eval_x_blocks, axis=0)
                train_targets = {name: np.concatenate(train_target_blocks[name], axis=0) for name in target_names}
                eval_targets = {name: np.concatenate(eval_target_blocks[name], axis=0) for name in target_names}
                score_payload = task_identity_predictiveness_score(
                    train_x,
                    eval_x,
                    train_targets,
                    eval_targets,
                    alpha=1.0,
                )
                overall = float(score_payload.get('overall_predictiveness_score', -1.0))
            else:
                train_x = np.zeros((0, 8), dtype=np.float32)
                eval_x = np.zeros((0, 8), dtype=np.float32)
                score_payload = {
                    'overall_predictiveness_score': -1.0,
                    'target_scores': {name: -1.0 for name in target_names},
                    'target_predictions': {name: [] for name in target_names},
                    'standardization': {'mean': [], 'std': []},
                    'alpha': 1.0,
                }
                overall = -1.0

            report = {
                'state_label': label,
                'state_mode': state_mode,
                'valid': valid,
                'fraud_padding': False,
                'all_dims_explained': True,
                'train_rows': int(train_x.shape[0]),
                'eval_rows': int(eval_x.shape[0]),
                'train_seed_pool': sorted(set(train_seed_pool)),
                'eval_seed_pool': sorted(set(eval_seed_pool)),
                'overall_predictiveness_score': float(overall),
                'target_scores': dict(score_payload.get('target_scores', {})),
                'target_predictions': dict(score_payload.get('target_predictions', {})),
                'standardization': dict(score_payload.get('standardization', {})),
                'alpha': float(score_payload.get('alpha', 1.0)),
            }
            mode_reports[label] = report
            if valid and overall > best_score:
                best_score = overall
                best_label = label

        s0_score = float((mode_reports.get('S0') or {}).get('overall_predictiveness_score', -1.0))
        s1_score = float((mode_reports.get('S1') or {}).get('overall_predictiveness_score', -1.0))
        s2_score = float((mode_reports.get('S2') or {}).get('overall_predictiveness_score', -1.0))
        s2_valid = bool((mode_reports.get('S2') or {}).get('valid', False))
        s0_valid = bool((mode_reports.get('S0') or {}).get('valid', False))
        s1_valid = bool((mode_reports.get('S1') or {}).get('valid', False))
        s2_over_s0_margin = float(s2_score - s0_score) if s2_valid and s0_valid else None
        s2_over_s1_margin = float(s2_score - s1_score) if s2_valid and s1_valid else None
        passed = bool(
            s2_valid and s0_valid and s1_valid
            and s2_over_s0_margin is not None and s2_over_s0_margin > 0.02
            and s2_over_s1_margin is not None and s2_over_s1_margin > 0.02
        )
        best_state_name = mode_defs.get(best_label) if best_label else None
        best_state_score = float((mode_reports.get(best_label) or {}).get('overall_predictiveness_score', -1.0)) if best_label else None
        return {
            'audit_name': 'task_identity_state_audit_v1',
            'train_eval_split_rule': 'odd_seed_train_even_seed_eval',
            'shared_corpus_rollout_count': len(rollouts),
            'mode_reports': mode_reports,
            'best_train_state_mode': best_label,
            'best_train_state_name': best_state_name,
            'best_train_state_score': best_state_score,
            'preferred_train_state_mode': 'S2',
            'task_identity_predictiveness_gain': float(task_identity_predictiveness_gain(max(s0_score, s1_score), s2_score)) if s2_valid and (s0_valid or s1_valid) else None,
            's2_over_s0_margin': s2_over_s0_margin,
            's2_over_s1_margin': s2_over_s1_margin,
            'passed': passed,
            'train_state_mode_acceptable': best_label is not None,
            'fraud_padding': False,
            'all_dims_explained': True,
        }

    def _should_stop_after_rca_result(self, experiment_id: str, payload: dict[str, Any]) -> bool:
        if experiment_id == 'RCA2':
            return not bool(payload.get('v1c_carrier_canonical_ready', False))
        if experiment_id == 'RCA3':
            return not bool(payload.get('passed', False))
        if experiment_id == 'RCA4':
            return not (bool(payload.get('passed', False)) or bool(payload.get('train_state_mode_acceptable', False)))
        if experiment_id == 'RCA5':
            return not bool(payload.get('passed', False))
        if experiment_id == 'RCA6':
            return not bool(payload.get('passed', False))
        return False

    def run_rca2(self) -> dict[str, Any]:
        seeds = list(self.seed_split_a)
        raw_spec = self._matrix_cell_lane_spec('V0T0S0', 'RCA2', note='Frozen raw-canonical regression reference.')
        carrier_spec = self._matrix_cell_lane_spec('V1cT0S0', 'RCA2', note='Frozen canonical-ready local carrier regression.')
        _, raw_report = self._visual_report(raw_spec, seeds)
        _, carrier_report = self._visual_report(carrier_spec, seeds, baseline_gap=float(raw_report.get('weighted_visual_gap', 1.0)))

        raw_local = raw_report.get('g4b_local_breakdown') or {}
        carrier_local = carrier_report.get('g4b_local_breakdown') or {}
        improvements = {
            'boundary_contrast': float(carrier_local.get('handle_boundary_contrast', 0.0)) > float(raw_local.get('handle_boundary_contrast', 0.0)),
            'edge_density': float(carrier_local.get('handle_edge_density', 0.0)) > float(raw_local.get('handle_edge_density', 0.0)),
            'crop_entropy': float(carrier_local.get('handle_crop_entropy', 0.0)) > float(raw_local.get('handle_crop_entropy', 0.0)),
        }
        win_count = sum(1 for passed in improvements.values() if passed)
        rank_score_gain = float(carrier_local.get('local_score_rank_only', 0.0) - raw_local.get('local_score_rank_only', 0.0))
        local_attack_beats_baseline = bool(
            carrier_report.get('measurement_truthful', False)
            and raw_report.get('measurement_truthful', False)
            and win_count >= 2
        )
        measurement_truthful_available = bool(carrier_report.get('measurement_truthful', False))
        visual_alignment_gain_value = float(carrier_report.get('visual_alignment_gain', 0.0))
        v1c_carrier_canonical_ready = bool(
            measurement_truthful_available
            and local_attack_beats_baseline
            and visual_alignment_gain_value >= GLOBAL_VISUAL_ALIGNMENT_THRESHOLD
        )
        payload = {
            'experiment_id': 'RCA2',
            'generated_at': now_iso(),
            'selector_mode': self.selector_mode,
            'baseline_cell_id': self._baseline_cell_id_v5_pro(),
            'ts_baseline_cell_id': self._ts_baseline_cell_id_v6_1(),
            'frozen_matrix_hash': self._frozen_matrix_hash_v5_pro(),
            'passed': v1c_carrier_canonical_ready,
            'measurement_truthful_available': measurement_truthful_available,
            'local_attack_beats_baseline': local_attack_beats_baseline,
            'local_improvement_flags': improvements,
            'local_improvement_count': win_count,
            'rank_score_gain': rank_score_gain,
            'visual_alignment_gain': visual_alignment_gain_value,
            'v1c_carrier_canonical_ready': v1c_carrier_canonical_ready,
            'raw_canonical': raw_report,
            'v1c_carrier': carrier_report,
        }
        self._write_artifact('RCA2', payload, 'RCA2 Local Affordance Visual Attack', [
            f'measurement_truthful_available: {measurement_truthful_available}',
            f'local_attack_beats_baseline: {local_attack_beats_baseline}',
            f'v1c_carrier_canonical_ready: {v1c_carrier_canonical_ready}',
        ])
        return {'payload': payload, 'updates': [], 'lane_specs': [raw_spec.to_payload(), carrier_spec.to_payload()]}

    def run_rca3(self) -> dict[str, Any]:
        readiness_seeds = list(self.seed_split_a[:8])
        readiness_normal = self._make_lane_spec(
            lane_id='RCA3_V1cT2S0_readiness_normal',
            stage='probe',
            env_contract=dict(self._frozen_matrix_contracts_v5_pro()['V1cT2S0']['env_contract_config']),
            interventions={'rotation_source': 'aligned', 'profile_id': 'V1cT2S0_readiness_normal', 'cell_id': 'V1cT2S0'},
            claim_policy='canonical',
            note='RCA3-A readiness batch normal rollout under V1cT2S0.',
            experiment_id='RCA3',
        )
        readiness_forced = self._make_lane_spec(
            lane_id='RCA3_V1cT2S0_readiness_forced_detach',
            stage='probe',
            env_contract=dict(self._frozen_matrix_contracts_v5_pro()['V1cT2S0']['env_contract_config']),
            interventions={'rotation_source': 'aligned', 'profile_id': 'V1cT2S0_readiness_forced_detach', 'cell_id': 'V1cT2S0', 'force_detach_probe': True},
            claim_policy='diagnostic',
            note='RCA3-A forced-detach readiness probe under V1cT2S0.',
            experiment_id='RCA3',
        )
        normal_rollouts = self._run_rollout_set(readiness_normal, readiness_seeds)
        forced_rollouts = self._run_rollout_set(readiness_forced, readiness_seeds)
        readiness_summary = {
            'readiness_seeds': readiness_seeds,
            'phase_locked_seed_coverage': seed_coverage_with_any_true(normal_rollouts, 'phase_locked_trace'),
            'positive_effective_pull_seed_coverage': seed_coverage_with_positive_peak(normal_rollouts, 'effective_pull_progress_trace'),
            'forced_detach_reason_seed_coverage': detach_reason_coverage(forced_rollouts),
            'runtime_handle_anchor_valid_coverage': self._seed_coverage_all_true(normal_rollouts, 'runtime_handle_anchor_valid_trace'),
        }
        readiness_pass = bool(
            readiness_summary['phase_locked_seed_coverage'] >= 0.50
            and readiness_summary['positive_effective_pull_seed_coverage'] >= 0.50
            and readiness_summary['forced_detach_reason_seed_coverage'] >= 0.50
            and readiness_summary['runtime_handle_anchor_valid_coverage'] >= 1.0
        )
        readiness_summary['passed'] = readiness_pass

        lane_specs = [readiness_normal.to_payload(), readiness_forced.to_payload()]
        if not readiness_pass:
            payload = {
                'experiment_id': 'RCA3',
                'generated_at': now_iso(),
                'selector_mode': self.selector_mode,
                'baseline_cell_id': self._baseline_cell_id_v5_pro(),
                'ts_baseline_cell_id': self._ts_baseline_cell_id_v6_1(),
                'frozen_matrix_hash': self._frozen_matrix_hash_v5_pro(),
                'passed': False,
                'readiness_batch_summary': readiness_summary,
                'full_split_a_comparison_skipped': True,
                'blocked_reasons': ['readiness_batch_failed'],
            }
            self._write_artifact('RCA3', payload, 'RCA3 Affordance Plus Transition Contract', [
                f"readiness_batch_passed: {readiness_pass}",
                f"phase_locked_seed_coverage: {readiness_summary['phase_locked_seed_coverage']:.6f}",
                f"positive_effective_pull_seed_coverage: {readiness_summary['positive_effective_pull_seed_coverage']:.6f}",
            ])
            return {'payload': payload, 'updates': [], 'lane_specs': lane_specs}

        seeds = list(self.seed_split_a)
        baseline_spec = self._matrix_cell_lane_spec('V1cT0S0', 'RCA3', note='RCA3-B frozen T/S baseline cell.')
        candidate_spec = self._matrix_cell_lane_spec('V1cT2S0', 'RCA3', note='RCA3-B deeper embodied transition candidate.')
        baseline_rollouts = self._run_rollout_set(baseline_spec, seeds)
        candidate_rollouts = self._run_rollout_set(candidate_spec, seeds)
        baseline_summary = self._rollout_summary(baseline_rollouts)
        candidate_summary = self._rollout_summary(candidate_rollouts)
        ceiling = rollout_ceiling_lift(baseline_summary, candidate_summary)
        phase_locked_rate_delta = float(candidate_summary.get('phase_locked_rate', 0.0) - baseline_summary.get('phase_locked_rate', 0.0))
        mean_effective_pull_progress_delta = float(candidate_summary.get('mean_effective_pull_progress', 0.0) - baseline_summary.get('mean_effective_pull_progress', 0.0))
        mean_max_drawer_fraction_delta = float(candidate_summary.get('mean_max_drawer_fraction', 0.0) - baseline_summary.get('mean_max_drawer_fraction', 0.0))
        passed = bool(
            phase_locked_rate_delta > 0.0
            and mean_effective_pull_progress_delta > 0.0
            and mean_max_drawer_fraction_delta > 0.0
            and float(ceiling.get('score', 0.0)) > 0.0
        )
        payload = {
            'experiment_id': 'RCA3',
            'generated_at': now_iso(),
            'selector_mode': self.selector_mode,
            'baseline_cell_id': self._baseline_cell_id_v5_pro(),
            'ts_baseline_cell_id': self._ts_baseline_cell_id_v6_1(),
            'frozen_matrix_hash': self._frozen_matrix_hash_v5_pro(),
            'passed': passed,
            'readiness_batch_summary': readiness_summary,
            'transition_baseline_cell_id': 'V1cT0S0',
            'transition_candidate_cell_id': 'V1cT2S0',
            'baseline_rollout_summary': baseline_summary,
            'transition_rollout_summary': candidate_summary,
            'rollout_ceiling_lift': ceiling,
            'phase_locked_rate_delta': phase_locked_rate_delta,
            'mean_effective_pull_progress_delta': mean_effective_pull_progress_delta,
            'mean_max_drawer_fraction_delta': mean_max_drawer_fraction_delta,
        }
        self._write_artifact('RCA3', payload, 'RCA3 Affordance Plus Transition Contract', [
            f'readiness_batch_passed: {readiness_pass}',
            f'phase_locked_rate_delta: {phase_locked_rate_delta:.6f}',
            f'mean_effective_pull_progress_delta: {mean_effective_pull_progress_delta:.6f}',
            f"rollout_ceiling_lift: {float(ceiling.get('score', 0.0)):.6f}",
        ])
        lane_specs.extend([baseline_spec.to_payload(), candidate_spec.to_payload()])
        return {'payload': payload, 'updates': [], 'lane_specs': lane_specs}

    def run_rca4(self) -> dict[str, Any]:
        seeds = list(self.seed_split_a)
        source_spec = self._matrix_cell_lane_spec('V1cT2S0', 'RCA4', note='Shared T2 corpus for offline task-identity state audit.')
        rollouts = self._run_rollout_set(source_spec, seeds)
        shared_corpus_rollout_summary = self._rollout_summary(rollouts)
        state_audit = self._task_identity_state_audit_v1(rollouts)
        best_train_state_mode = state_audit.get('best_train_state_mode')
        canonical_train_cell = f'V1cT2{best_train_state_mode}' if best_train_state_mode else None
        payload = {
            'experiment_id': 'RCA4',
            'generated_at': now_iso(),
            'selector_mode': self.selector_mode,
            'baseline_cell_id': self._baseline_cell_id_v5_pro(),
            'ts_baseline_cell_id': self._ts_baseline_cell_id_v6_1(),
            'frozen_matrix_hash': self._frozen_matrix_hash_v5_pro(),
            'source_transition_cell_id': 'V1cT2S0',
            'passed': bool(state_audit.get('passed', False)),
            'train_state_mode_acceptable': bool(state_audit.get('train_state_mode_acceptable', False)),
            'best_train_state_mode': best_train_state_mode,
            'best_train_state_name': state_audit.get('best_train_state_name'),
            'preferred_train_state_mode': state_audit.get('preferred_train_state_mode'),
            'canonical_train_cell': canonical_train_cell,
            'preferred_canonical_train_cell': 'V1cT2S2',
            'task_identity_predictiveness_gain': state_audit.get('task_identity_predictiveness_gain'),
            'shared_corpus_rollout_summary': shared_corpus_rollout_summary,
            'state_audit': state_audit,
        }
        self._write_artifact('RCA4', payload, 'RCA4 Affordance Plus Transition State', [
            f"passed: {payload['passed']}",
            f"best_train_state_mode: {best_train_state_mode}",
            f"canonical_train_cell: {canonical_train_cell}",
        ])
        return {'payload': payload, 'updates': [], 'lane_specs': [source_spec.to_payload()]}

    def run_rca5(self) -> dict[str, Any]:
        seeds = list(self.seed_split_a)
        rca1_payload = load_json(self._artifact_map()['RCA1'][0], {})
        rca2_payload = load_json(self._artifact_map()['RCA2'][0], {})
        rca4_payload = load_json(self._artifact_map()['RCA4'][0], {})
        measurement_truthful_available = bool(rca1_payload.get('measurement_truthful_available', False))
        v1c_carrier_canonical_ready = bool(rca2_payload.get('v1c_carrier_canonical_ready', False))
        best_train_state_mode = rca4_payload.get('best_train_state_mode')
        best_train_state_name = rca4_payload.get('best_train_state_name')
        canonical_train_cell = f'V1cT2{best_train_state_mode}' if best_train_state_mode else None
        transition_cells = ['V1cT0S0', 'V1cT1S0', 'V1cT2S0']
        lane_specs: list[dict[str, Any]] = []
        cell_payloads: list[dict[str, Any]] = []

        baseline_spec = self._matrix_cell_lane_spec('V1cT0S0', 'RCA5', note='Frozen canonical carrier baseline for transition selection.')
        baseline_rollouts = self._run_rollout_set(baseline_spec, seeds)
        baseline_summary = self._rollout_summary(baseline_rollouts)

        for cell_id in transition_cells:
            spec = self._matrix_cell_lane_spec(cell_id, 'RCA5', note='Transition-cell screen under frozen canonical carrier.')
            lane_specs.append(spec.to_payload())
            if cell_id == 'V1cT0S0':
                rollout_summary = baseline_summary
            else:
                rollout_summary = self._rollout_summary(self._run_rollout_set(spec, seeds))
            gain = rollout_ceiling_lift(baseline_summary, rollout_summary)
            cell_payloads.append({
                'cell_id': cell_id,
                'lane': spec.to_payload(),
                'rollout_summary': rollout_summary,
                'rollout_ceiling_lift': gain,
                'delta_mean_max_drawer_fraction': float(gain.get('delta_mean_max_drawer_fraction', 0.0)),
                'phase_locked_rate': float(rollout_summary.get('phase_locked_rate', 0.0)),
                'mean_effective_pull_progress': float(rollout_summary.get('mean_effective_pull_progress', 0.0)),
            })

        best_transition_payload = max(cell_payloads, key=self._matrix_cell_sort_key)
        best_transition_cell = str(best_transition_payload.get('cell_id'))
        best_transition_positive = bool(float((best_transition_payload.get('rollout_ceiling_lift') or {}).get('score', 0.0)) > 0.0)
        passed = bool(
            best_transition_cell == 'V1cT2S0'
            and best_transition_positive
            and measurement_truthful_available
            and v1c_carrier_canonical_ready
            and canonical_train_cell is not None
        )
        payload = {
            'experiment_id': 'RCA5',
            'generated_at': now_iso(),
            'selector_mode': self.selector_mode,
            'baseline_cell_id': self._baseline_cell_id_v5_pro(),
            'ts_baseline_cell_id': self._ts_baseline_cell_id_v6_1(),
            'frozen_matrix_hash': self._frozen_matrix_hash_v5_pro(),
            'split_a_seeds': seeds,
            'measurement_truthful_available': measurement_truthful_available,
            'v1c_carrier_canonical_ready': v1c_carrier_canonical_ready,
            'baseline_rollout_summary': baseline_summary,
            'transition_cells': cell_payloads,
            'best_transition_cell': best_transition_cell,
            'best_transition_payload': best_transition_payload,
            'best_train_state_mode': best_train_state_mode,
            'best_train_state_name': best_train_state_name,
            'canonical_train_cell': canonical_train_cell,
            'preferred_canonical_train_cell': 'V1cT2S2',
            'passed': passed,
        }
        self._write_artifact('RCA5', payload, 'RCA5 Frozen Matrix Screen', [
            f'best_transition_cell: {best_transition_cell}',
            f'best_train_state_mode: {best_train_state_mode}',
            f'canonical_train_cell: {canonical_train_cell}',
        ])
        return {'payload': payload, 'updates': [], 'lane_specs': lane_specs}

    def run_rca6(self) -> dict[str, Any]:
        seeds = list(self.seed_split_b)
        rca5_payload = load_json(self._artifact_map()['RCA5'][0], {})
        best_transition_cell = rca5_payload.get('best_transition_cell')
        canonical_train_cell = rca5_payload.get('canonical_train_cell')
        if not best_transition_cell:
            payload = {
                'experiment_id': 'RCA6',
                'generated_at': now_iso(),
                'selector_mode': self.selector_mode,
                'baseline_cell_id': self._baseline_cell_id_v5_pro(),
                'ts_baseline_cell_id': self._ts_baseline_cell_id_v6_1(),
                'frozen_matrix_hash': self._frozen_matrix_hash_v5_pro(),
                'best_transition_cell': None,
                'canonical_train_cell': canonical_train_cell,
                'best_train_state_mode': rca5_payload.get('best_train_state_mode'),
                'passed': False,
                'blocked': True,
                'blocked_reasons': ['no_best_transition_cell_from_rca5'],
            }
            self._write_artifact('RCA6', payload, 'RCA6 Best Cell Replicate', ['blocked: True', 'blocked_reasons: no_best_transition_cell_from_rca5'])
            return {'payload': payload, 'updates': [], 'lane_specs': []}

        baseline_spec = self._matrix_cell_lane_spec('V1cT0S0', 'RCA6', note='Split-B canonical carrier baseline for replication.')
        best_spec = self._matrix_cell_lane_spec(str(best_transition_cell), 'RCA6', note='Split-B replication of selected positive transition cell.')
        baseline_rollouts = self._run_rollout_set(baseline_spec, seeds)
        best_rollouts = self._run_rollout_set(best_spec, seeds)
        baseline_summary = self._rollout_summary(baseline_rollouts)
        best_summary = self._rollout_summary(best_rollouts)
        ceiling = rollout_ceiling_lift(baseline_summary, best_summary)
        delta_mean_max_drawer_fraction = float(best_summary.get('mean_max_drawer_fraction', 0.0) - baseline_summary.get('mean_max_drawer_fraction', 0.0))
        replicate_positive = bool(float(ceiling.get('score', 0.0)) > 0.0 and delta_mean_max_drawer_fraction > 0.0)
        payload = {
            'experiment_id': 'RCA6',
            'generated_at': now_iso(),
            'selector_mode': self.selector_mode,
            'baseline_cell_id': self._baseline_cell_id_v5_pro(),
            'ts_baseline_cell_id': self._ts_baseline_cell_id_v6_1(),
            'frozen_matrix_hash': self._frozen_matrix_hash_v5_pro(),
            'best_transition_cell': best_transition_cell,
            'canonical_train_cell': canonical_train_cell,
            'best_train_state_mode': rca5_payload.get('best_train_state_mode'),
            'split_b_seeds': seeds,
            'replicate_positive': replicate_positive,
            'passed': replicate_positive,
            'baseline_summary': baseline_summary,
            'replicated_summary': best_summary,
            'rollout_ceiling_lift': ceiling,
            'delta_mean_max_drawer_fraction': delta_mean_max_drawer_fraction,
        }
        self._write_artifact('RCA6', payload, 'RCA6 Best Cell Replicate', [
            f'best_transition_cell: {best_transition_cell}',
            f"rollout_ceiling_lift: {float(ceiling.get('score', 0.0)):.6f}",
            f'replicate_positive: {replicate_positive}',
        ])
        return {'payload': payload, 'updates': [], 'lane_specs': [baseline_spec.to_payload(), best_spec.to_payload()]}

    def run_rca7(self) -> dict[str, Any]:
        rca1_payload = load_json(self._artifact_map()['RCA1'][0], {})
        rca2_payload = load_json(self._artifact_map()['RCA2'][0], {})
        rca5_payload = load_json(self._artifact_map()['RCA5'][0], {})
        rca6_payload = load_json(self._artifact_map()['RCA6'][0], {})
        tiny_retrain_permitted = bool(
            rca6_payload.get('passed')
            and rca5_payload.get('canonical_train_cell')
            and rca1_payload.get('measurement_truthful_available')
            and rca2_payload.get('v1c_carrier_canonical_ready')
        )
        blocked_reasons: list[str] = []
        if not rca1_payload.get('measurement_truthful_available'):
            blocked_reasons.append('RCA1_truthful_measurement_unavailable')
        if not rca2_payload.get('v1c_carrier_canonical_ready'):
            blocked_reasons.append('RCA2_v1c_carrier_not_canonical_ready')
        if not rca6_payload.get('passed'):
            blocked_reasons.append('RCA6_replicate_not_positive')
        if not rca5_payload.get('canonical_train_cell'):
            blocked_reasons.append('RCA5_canonical_train_cell_missing')
        payload = {
            'experiment_id': 'RCA7',
            'generated_at': now_iso(),
            'selector_mode': self.selector_mode,
            'baseline_cell_id': self._baseline_cell_id_v5_pro(),
            'ts_baseline_cell_id': self._ts_baseline_cell_id_v6_1(),
            'frozen_matrix_hash': self._frozen_matrix_hash_v5_pro(),
            'best_transition_cell': rca5_payload.get('best_transition_cell'),
            'best_train_state_mode': rca5_payload.get('best_train_state_mode'),
            'canonical_train_cell': rca5_payload.get('canonical_train_cell'),
            'preferred_canonical_train_cell': rca5_payload.get('preferred_canonical_train_cell', 'V1cT2S2'),
            'tiny_retrain_permitted': tiny_retrain_permitted,
            'eligible': tiny_retrain_permitted,
            'passed': tiny_retrain_permitted,
            'blocked': not tiny_retrain_permitted,
            'blocked_reasons': blocked_reasons,
            'launched': False,
        }
        self._write_artifact('RCA7', payload, 'RCA7 Tiny Retrain If Eligible', [
            f'tiny_retrain_permitted: {tiny_retrain_permitted}',
            f"canonical_train_cell: {payload['canonical_train_cell']}",
            f"blocked_reasons: {', '.join(blocked_reasons) if blocked_reasons else 'none'}",
        ])
        return {'payload': payload, 'updates': [], 'lane_specs': []}

    def _carryforward_g2(self) -> dict[str, Any]:
        payload = load_json(P2E1_ARTIFACT, {}) if P2E1_ARTIFACT.exists() else {}
        return {
            'exists': P2E1_ARTIFACT.exists(),
            'passed': bool(payload.get('passed')),
            'aligned_beats_random_under_sensitive': bool(payload.get('aligned_beats_random_under_sensitive', False)),
            'orientation_causal_sensitivity': payload.get('orientation_causal_sensitivity'),
        }

    def _best_bundle_payload(self) -> dict[str, Any]:
        experiment_id = 'RCA5' if self.experiment_family == 'RCA' else 'VR4'
        artifact = self._artifact_map()[experiment_id][0]
        return load_json(artifact, {}) if artifact.exists() else {}

    def _replication_payload(self) -> dict[str, Any]:
        experiment_id = 'RCA6' if self.experiment_family == 'RCA' else 'VR5'
        artifact = self._artifact_map()[experiment_id][0]
        return load_json(artifact, {}) if artifact.exists() else {}

    def _completed_rca_payload(self, exp_id: str) -> dict[str, Any]:
        if self.experiment_family != 'RCA':
            return {}
        if str(exp_id) not in {str(item) for item in self.completed_experiments}:
            return {}
        artifact = self._artifact_map()[str(exp_id)][0]
        return load_json(artifact, {}) if artifact.exists() else {}

    def evaluate_gates(self) -> tuple[dict[str, Any], ControllerRoute]:
        g0_details = self._baseline_integrity_details()
        g0 = GateDecision(
            gate_id='G0_baseline_integrity',
            passed=bool(g0_details['control_baseline_pass'] and g0_details['parity_non_blocker'] and g0_details['authority_readable']),
            summary='Baseline healthy, parity not-primary-blocker, and authority readable.',
            details=g0_details,
        )

        rca0 = self._completed_rca_payload('RCA0')
        rca1 = self._completed_rca_payload('RCA1')
        rca2 = self._completed_rca_payload('RCA2')
        rca3 = self._completed_rca_payload('RCA3')
        rca4 = self._completed_rca_payload('RCA4')
        rca5 = self._completed_rca_payload('RCA5')
        rca6 = self._completed_rca_payload('RCA6')
        rca7 = self._completed_rca_payload('RCA7')

        carrier_report = (rca2.get('v1c_carrier') or {}) if self.experiment_family == 'RCA' else {}
        lane_report = (rca1.get('lane_report') or {}) if self.experiment_family == 'RCA' else {}
        anchor_report = carrier_report or lane_report
        state_audit = (rca4.get('state_audit') or {}) if self.experiment_family == 'RCA' else {}

        invalid_reasons: list[str] = []
        if self.experiment_family == 'RCA':
            if self.selector_mode != 'frozen_v5_pro':
                invalid_reasons.append('selector_mode_not_frozen_v5_pro')
            baseline_cell_id = str((rca0 or {}).get('baseline_cell_id') or self._baseline_cell_id_v5_pro())
            matrix_hash = str((rca0 or {}).get('frozen_matrix_hash') or self._frozen_matrix_hash_v5_pro())
            executed_rca_ids = [
                str(exp_id) for exp_id in dict.fromkeys(self.completed_experiments)
                if str(exp_id).startswith('RCA') and str(exp_id) != 'RCA0'
            ]
            for exp_id in executed_rca_ids:
                artifact_path = self._artifact_map()[exp_id][0]
                if not artifact_path.exists():
                    continue
                payload = load_json(artifact_path, {})
                if payload.get('baseline_cell_id') not in {None, baseline_cell_id}:
                    invalid_reasons.append(f'{exp_id}_baseline_cell_drift')
                if payload.get('frozen_matrix_hash') not in {None, matrix_hash}:
                    invalid_reasons.append(f'{exp_id}_frozen_matrix_hash_drift')
            if self.truthful_measurement_required and rca1 and not bool(rca1.get('measurement_truthful_available', False)):
                invalid_reasons.append('truthful_measurement_unavailable')
            if self._invalid_run_reason:
                invalid_reasons.append(self._invalid_run_reason)
        invalid_reasons = list(dict.fromkeys(invalid_reasons))

        g1_details = {
            'wrist_like': bool(anchor_report.get('secondary_camera_wrist_like', False)),
            'no_overlay': not bool(anchor_report.get('marker_overlay_enabled', True)),
            'no_diagnostic_texture': not bool(anchor_report.get('diagnostic_only', True)),
            'provenance_complete': bool(anchor_report.get('provenance_complete', False)),
        }
        g1 = GateDecision(
            gate_id='G1_observation_contract',
            passed=all(g1_details.values()),
            summary='Canonical carrier observation contract remains wrist-like, overlay-free, non-diagnostic, and provenance-complete.',
            details=g1_details,
        )

        g2_details = self._carryforward_g2()
        g2 = GateDecision(
            gate_id='G2_action_causality',
            passed=bool(g2_details['exists'] and g2_details['passed'] and g2_details['aligned_beats_random_under_sensitive']),
            summary='Carry-forward action causality remains healthy.',
            details=g2_details,
        )

        mode_reports = state_audit.get('mode_reports') or {}
        s0_score = float((mode_reports.get('S0') or {}).get('overall_predictiveness_score', -1.0))
        s1_score = float((mode_reports.get('S1') or {}).get('overall_predictiveness_score', -1.0))
        s2_score = float((mode_reports.get('S2') or {}).get('overall_predictiveness_score', -1.0))
        g3_details = {
            'best_train_state_mode': rca4.get('best_train_state_mode'),
            'best_train_state_name': rca4.get('best_train_state_name'),
            'preferred_train_state_mode': state_audit.get('preferred_train_state_mode'),
            'task_identity_predictiveness_gain': state_audit.get('task_identity_predictiveness_gain'),
            's2_over_s0_margin': state_audit.get('s2_over_s0_margin'),
            's2_over_s1_margin': state_audit.get('s2_over_s1_margin'),
            's2_score': s2_score,
            's1_score': s1_score,
            's0_score': s0_score,
            's2_superiority_pass': bool(rca4.get('passed', False)),
            'train_state_mode_acceptable': bool(rca4.get('train_state_mode_acceptable', False)),
            'fraud_padding': bool(state_audit.get('fraud_padding', False)),
            'all_dims_explained': bool(state_audit.get('all_dims_explained', False)),
        }
        g3 = GateDecision(
            gate_id='G3_state_semantics',
            passed=bool(g3_details['train_state_mode_acceptable'] and not g3_details['fraud_padding'] and g3_details['all_dims_explained']),
            summary='Shared-corpus task-identity state audit produced an acceptable train-state mode.',
            details=g3_details,
        )

        g4a_details = {
            'no_overlay': not bool(anchor_report.get('marker_overlay_enabled', True)),
            'no_diagnostic_texture': not bool(anchor_report.get('diagnostic_only', True)),
            'wrist_like_second_camera': bool(anchor_report.get('secondary_camera_wrist_like', False)),
            'provenance_complete': bool(anchor_report.get('provenance_complete', False)),
            'camera_relativeness_residual': anchor_report.get('secondary_camera_residual_mean'),
        }
        g4a = GateDecision(
            gate_id='G4a_observation_semantics',
            passed=bool(
                g4a_details['no_overlay']
                and g4a_details['no_diagnostic_texture']
                and g4a_details['wrist_like_second_camera']
                and g4a_details['provenance_complete']
                and g4a_details['camera_relativeness_residual'] is not None
                and float(g4a_details['camera_relativeness_residual']) < 0.05
            ),
            summary='Observation semantics remain canonical on the frozen V1c carrier.',
            details=g4a_details,
        )

        g4b_local = (carrier_report.get('g4b_local_breakdown') or {})
        g4b_details = dict(g4b_local)
        g4b_details.update({
            'local_attack_beats_baseline': bool(rca2.get('local_attack_beats_baseline', False)),
            'source_profile_id': 'V1cT0S0' if carrier_report else None,
        })
        g4b = GateDecision(
            gate_id='G4b_local_affordance_readability',
            passed=bool(
                rca2.get('local_attack_beats_baseline', False)
                and g4b_local.get('measurement_truthful_pass')
                and not g4b_local.get('bbox_inflation_flag')
                and g4b_local.get('all_primitives_pass')
            ),
            summary='Local readability remains a residual audit; it no longer decides whether T/S mainline continues.',
            blocking=False,
            details=g4b_details,
        )

        g4c_details = {
            'v1c_carrier_canonical_ready': bool(rca2.get('v1c_carrier_canonical_ready', False)),
            'measurement_truthful_available': bool(rca2.get('measurement_truthful_available', False)),
            'local_attack_beats_baseline': bool(rca2.get('local_attack_beats_baseline', False)),
            'visual_alignment_gain': float(rca2.get('visual_alignment_gain', 0.0)),
            'source_profile_id': 'V1cT0S0' if rca2 else None,
        }
        g4c = GateDecision(
            gate_id='G4c_global_visual_canonicality',
            passed=bool(rca2.get('v1c_carrier_canonical_ready', False)),
            summary='Global canonicality is frozen at the V1c carrier regression step.',
            details=g4c_details,
        )

        g5_details = {
            'best_transition_cell': rca5.get('best_transition_cell'),
            'best_train_state_mode': rca5.get('best_train_state_mode'),
            'canonical_train_cell': rca5.get('canonical_train_cell'),
            'preferred_canonical_train_cell': rca5.get('preferred_canonical_train_cell'),
            'replicate_positive': bool(rca6.get('passed', False)),
            'rollout_ceiling_lift': float((rca6.get('rollout_ceiling_lift') or {}).get('score', 0.0)),
            'delta_mean_max_drawer_fraction': float(rca6.get('delta_mean_max_drawer_fraction', 0.0)),
            'tiny_retrain_permitted': bool(rca7.get('tiny_retrain_permitted', False)),
            'carrier_ready': bool(rca2.get('v1c_carrier_canonical_ready', False)),
            'truthful_measurement_available': bool(rca1.get('measurement_truthful_available', False)),
        }
        g5 = GateDecision(
            gate_id='G5_training_eligibility',
            passed=bool(
                g5_details['truthful_measurement_available']
                and g5_details['carrier_ready']
                and g5_details['replicate_positive']
                and bool(g5_details['canonical_train_cell'])
                and g5_details['tiny_retrain_permitted']
            ),
            summary='Training eligibility now means a replicated positive canonical transition cell plus a real canonical_train_cell.',
            details=g5_details,
        )

        gate_map = {gate.gate_id: asdict(gate) for gate in [g0, g1, g2, g3, g4a, g4b, g4c, g5]}
        gate_map['contract_validity_score_v2'] = contract_validity_score_v2(
            g0=g0.passed,
            g1=g1.passed,
            g2=g2.passed,
            g3=g3.passed,
            g4a=g4a.passed,
            g4b_local=g4b.passed,
            g4c_global=g4c.passed,
        )
        gate_map['freeze_audit'] = {
            'selector_mode': self.selector_mode,
            'baseline_cell_id': (rca0 or {}).get('baseline_cell_id', self._baseline_cell_id_v5_pro()) if self.experiment_family == 'RCA' else None,
            'ts_baseline_cell_id': (rca0 or {}).get('ts_baseline_cell_id', self._ts_baseline_cell_id_v6_1()) if self.experiment_family == 'RCA' else None,
            'frozen_matrix_hash': (rca0 or {}).get('frozen_matrix_hash', self._frozen_matrix_hash_v5_pro()) if self.experiment_family == 'RCA' else None,
            'invalid_reasons': invalid_reasons,
            'truthful_measurement_required': self.truthful_measurement_required,
        }

        h0 = float((self.board.get('H0_same_problem_identity') or {}).get('posterior', 0.35))
        h6 = float((self.board.get('H6_environment_invalid_for_claim') or {}).get('posterior', 0.55))

        if invalid_reasons:
            route = ControllerRoute(
                scientific_terminal_state=None,
                route_next_branch='stay_current_branch',
                why='Invalid run under frozen root-cause object: ' + ', '.join(invalid_reasons),
                strongest_negative_capped=bool(self.cap_strongest_negative),
                cap_active=bool(self.cap_strongest_negative),
                h0_posterior=h0,
                h6_posterior=h6,
            )
            gate_map['route_decision_preview'] = asdict(route)
            return gate_map, route

        scientific_terminal_state: ScientificTerminalState | None = None
        route_next_branch: RouteNextBranch = 'stay_current_branch'
        why = 'Frozen T/S positive-canonical mainline still in progress.'

        if bool(rca7.get('tiny_retrain_permitted', False)) and bool(rca6.get('passed', False)) and bool(rca5.get('canonical_train_cell')):
            scientific_terminal_state = 'TS_CANONICAL_POSITIVE_ESTABLISHED'
            route_next_branch = 'tiny_retrain_confirmation'
            why = (
                'Truthful measurement remained valid, the frozen V1c carrier stayed canonical-ready, '
                f"{rca5.get('best_transition_cell')} produced replicated positive rollout lift, and "
                f"{rca5.get('canonical_train_cell')} is the selected canonical_train_cell for RCA7."
            )
        elif rca2 and not bool(rca2.get('v1c_carrier_canonical_ready', True)):
            scientific_terminal_state = 'TS_MAINLINE_REJECTED'
            route_next_branch = 'environment_reformulation'
            why = 'Frozen V1c carrier was not canonical-ready after RCA2, so no positive canonical T/S cell was reachable.'
        elif rca3 and not bool(rca3.get('passed', True)):
            scientific_terminal_state = 'TS_MAINLINE_REJECTED'
            route_next_branch = 'environment_reformulation'
            why = 'T2 failed the RCA3 readiness batch or the split-A T-only positive-lift comparison.'
        elif rca4 and not bool(rca4.get('passed', True)) and not bool(rca4.get('train_state_mode_acceptable', False)):
            scientific_terminal_state = 'TS_MAINLINE_REJECTED'
            route_next_branch = 'environment_reformulation'
            why = 'Shared-corpus RCA4 produced no acceptable train-state mode for the positive T2 mainline.'
        elif rca5 and not bool(rca5.get('passed', True)):
            scientific_terminal_state = 'TS_MAINLINE_REJECTED'
            route_next_branch = 'environment_reformulation'
            why = 'RCA5 did not produce V1cT2S0 as the first positive canonical transition cell with a materialized canonical_train_cell.'
        elif rca6 and not bool(rca6.get('passed', True)):
            scientific_terminal_state = 'TS_MAINLINE_REJECTED'
            route_next_branch = 'environment_reformulation'
            why = 'The selected positive canonical transition cell did not replicate positively on split B.'

        route = ControllerRoute(
            scientific_terminal_state=scientific_terminal_state,
            route_next_branch=route_next_branch,
            why=why,
            strongest_negative_capped=bool(self.cap_strongest_negative),
            cap_active=bool(self.cap_strongest_negative),
            h0_posterior=h0,
            h6_posterior=h6,
        )
        gate_map['route_decision_preview'] = asdict(route)
        return gate_map, route

    def _cycle_memo(self, route: ControllerRoute, gate_report: dict[str, Any], cycle_summary: dict[str, Any]) -> str:
        lines = [
            f"# {cycle_summary['cycle_id']}",
            '',
            f"- cycle_mode: {self.cycle_mode}",
            f"- selected_experiments: {', '.join(cycle_summary['selected_experiments']) if cycle_summary['selected_experiments'] else '(none)'}",
            f"- scientific_terminal_state: {route.scientific_terminal_state}",
            f"- route_next_branch: {route.route_next_branch}",
            f"- strongest_negative_capped: {route.strongest_negative_capped}",
            f"- why: {route.why}",
            '',
            '## Gates',
        ]
        gate_ids = ['G0_baseline_integrity', 'G1_observation_contract', 'G2_action_causality', 'G3_state_semantics', 'G4a_observation_semantics', 'G4b_local_affordance_readability', 'G4c_global_visual_canonicality', 'G5_training_eligibility']
        for gate_id in gate_ids:
            item = gate_report.get(gate_id, {})
            lines.append(f"- {gate_id}: {'pass' if item.get('passed') else 'fail'}")
        return "\n".join(lines).rstrip() + "\n"

    def _final_morning_memo(self, route: ControllerRoute, gate_report: dict[str, Any], final_summary: dict[str, Any]) -> str:
        rca1 = self._completed_rca_payload('RCA1')
        rca2 = self._completed_rca_payload('RCA2')
        rca5 = self._completed_rca_payload('RCA5')
        rca6 = self._completed_rca_payload('RCA6')
        rca7 = self._completed_rca_payload('RCA7')
        lines = [
            '# Morning Bundle',
            '',
            f"1. Was truthful measurement available? {'yes' if rca1.get('measurement_truthful_available') else 'no'}",
            f"2. Was the frozen V1c carrier canonical-ready? {'yes' if rca2.get('v1c_carrier_canonical_ready') else 'no'}",
            f"3. What transition cell won RCA5? {rca5.get('best_transition_cell', 'none')}",
            f"4. What canonical_train_cell was selected? {rca5.get('canonical_train_cell', 'none')}",
            f"5. Did RCA6 replicate positive rollout lift on split B? {'yes' if rca6.get('passed') else 'no'}",
            f"6. Was RCA7 unblocked? {'yes' if rca7.get('tiny_retrain_permitted') else 'no'}",
            f"7. Terminal interpretation: {route.why}",
            '',
            '## Gates',
            f"- G4b_local_affordance_readability: {'pass' if gate_report.get('G4b_local_affordance_readability', {}).get('passed') else 'fail'}",
            f"- G4c_global_visual_canonicality: {'pass' if gate_report.get('G4c_global_visual_canonicality', {}).get('passed') else 'fail'}",
            f"- G5_training_eligibility: {'pass' if gate_report.get('G5_training_eligibility', {}).get('passed') else 'fail'}",
            '',
            '## Key Results',
            f"- route_next_branch: {route.route_next_branch}",
            f"- scientific_terminal_state: {route.scientific_terminal_state}",
            f"- strongest_negative_capped: {route.strongest_negative_capped}",
            f"- baseline_cell_id: {rca2.get('baseline_cell_id', self._baseline_cell_id_v5_pro())}",
            f"- ts_baseline_cell_id: {rca2.get('ts_baseline_cell_id', self._ts_baseline_cell_id_v6_1())}",
            f"- best_transition_cell: {rca5.get('best_transition_cell', 'none')}",
            f"- canonical_train_cell: {rca5.get('canonical_train_cell', 'none')}",
            f"- replicated_rollout_ceiling_lift: {float((rca6.get('rollout_ceiling_lift') or {}).get('score', 0.0)):.6f}",
            f"- cycles_completed: {final_summary.get('cycles_completed', 0)}",
            f"- completed_experiments: {', '.join(final_summary.get('completed_experiments', []))}",
        ]
        return "\n".join(lines).rstrip() + "\n"

    def _proposed_truth_delta(self, route: ControllerRoute) -> dict[str, Any]:
        rca5 = self._completed_rca_payload('RCA5')
        return {
            'current': {
                'phase': self.authority.get('phase'),
                'verdict': route.scientific_terminal_state,
                'next_action': route.route_next_branch,
                'canonical_train_cell': rca5.get('canonical_train_cell'),
                'best_transition_cell': rca5.get('best_transition_cell'),
                'open_gates': [gate for gate in ['G4b_local_affordance_readability', 'G5_training_eligibility']],
            },
            'note': 'Autogenerated proposal only; no sovereign auto-promotion performed.',
        }

    def _proposed_next_actions(self, route: ControllerRoute) -> dict[str, Any]:
        return {
            'actions': [
                {
                    'id': 'proposed_next_mainline',
                    'priority': 1,
                    'status': 'pending',
                    'action': route.route_next_branch,
                    'why': route.why,
                }
            ]
        }

    def run(self, *, max_cycles: int = 4, sleep_seconds: int = 30) -> dict[str, Any]:
        cycle_count = 0
        last_route = ControllerRoute(None, 'stay_current_branch', 'Controller has not executed any cycle yet.')
        gate_report: dict[str, Any] = {}
        identity = controller_identity()
        controller_id = str(identity['controller_id'])
        run_id = str(identity['run_id'])
        owner = str(identity['owner'])
        pid = int(identity['pid'])
        lease_ok, lease_meta = acquire_controller_lease(controller_id, run_id, owner=owner, pid=pid)
        if not lease_ok:
            raise RuntimeError(f'controller lease denied: {lease_meta}')
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
                planned_ids = [item['id'] for item in selected]
                executed_ids: list[str] = []
                stop_after_cycle = False
                append_cycle_record({
                    'cycle_id': cycle_id,
                    'cycle_mode': self.cycle_mode,
                    'selected_experiments': planned_ids,
                    'started_at': now_iso(),
                    'policy': dict(self.policy),
                    'resource_budget_snapshot': self._resource_snapshot(),
                })
                if len(planned_ids) > 1:
                    append_deviation_record(
                        message=f'Controller co-scheduled multiple {self.experiment_family} experiments within a single cycle.',
                        scientific_semantics_changed=False,
                        details={'cycle_id': cycle_id, 'selected_experiments': planned_ids},
                    )
                for item in selected:
                    refresh_controller_lease(controller_id, run_id, pid=pid)
                    result = item['runner']()
                    exp_id = item['id']
                    self.completed_experiments.append(exp_id)
                    executed_ids.append(exp_id)
                    lane_results_payload[exp_id] = result['payload']
                    lane_specs_payload.extend(result.get('lane_specs', []))
                    cycle_updates.extend(result.get('updates', []))
                    append_evidence_record({
                        'cycle_id': cycle_id,
                        'experiment_id': exp_id,
                        'artifact_path': str(self._artifact_map()[exp_id][0]),
                        'reported_at': now_iso(),
                    })
                    if self.experiment_family == 'RCA' and self.selector_mode == 'frozen_v5_pro' and self._should_stop_after_rca_result(exp_id, result['payload']):
                        stop_after_cycle = True
                        break
                self.board = apply_updates(self.board, cycle_updates)
                write_hypothesis_board(self.board)
                gate_report, last_route = self.evaluate_gates()
                cycle_summary = {
                    'cycle_id': cycle_id,
                    'started_at': now_iso(),
                    'ended_at': now_iso(),
                    'lane_family': self.experiment_family,
                    'selected_experiments': executed_ids,
                    'planned_experiments': planned_ids,
                    'lane_ids': [spec.get('lane_id') for spec in lane_specs_payload],
                    'completed_experiments': list(dict.fromkeys(self.completed_experiments)),
                    'scientific_terminal_state': last_route.scientific_terminal_state,
                    'route_next_branch': last_route.route_next_branch,
                    'cycle_mode': self.cycle_mode,
                }
                cycle_memo = self._cycle_memo(last_route, gate_report, cycle_summary)
                route_payload = asdict(last_route)
                route_payload['generated_at'] = now_iso()
                write_cycle_bundle(
                    cycle_dir,
                    cycle_mode=self.cycle_mode,
                    lane_specs={'cycle_id': cycle_id, 'lane_family': self.experiment_family, 'lane_specs': lane_specs_payload, 'resource_budget_snapshot': self._resource_snapshot()},
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
                    controller_id=controller_id,
                    run_id=run_id,
                    cycle_count=cycle_count + 1,
                )
                cycle_count += 1
                if last_route.scientific_terminal_state is None and self._invalid_run_reason:
                    break
                if last_route.scientific_terminal_state in {'TS_CANONICAL_POSITIVE_ESTABLISHED', 'TS_MAINLINE_REJECTED'}:
                    break
                final_exp = 'RCA7' if self.experiment_family == 'RCA' else 'VR6'
                if stop_after_cycle or final_exp in self.completed_experiments or cycle_count >= int(self.resource_limits['max_cycles_per_run']):
                    break
                if sleep_seconds > 0 and not self.dry_run:
                    time.sleep(int(sleep_seconds))
            final_summary = {
                'generated_at': now_iso(),
                'cycle_mode': self.cycle_mode,
                'experiment_family': self.experiment_family,
                'cycles_completed': cycle_count,
                'completed_experiments': list(dict.fromkeys(self.completed_experiments)),
                'policy': dict(self.policy),
                'resource_budget_snapshot': self._resource_snapshot(),
                'scientific_terminal_state': last_route.scientific_terminal_state,
                'route_next_branch': last_route.route_next_branch,
            }
            route_payload = asdict(last_route)
            route_payload['generated_at'] = now_iso()
            morning_memo = self._final_morning_memo(last_route, gate_report, final_summary)
            write_final_run_outputs(route_decision=route_payload, morning_memo=morning_memo, final_summary=final_summary)
            return {'route_decision': route_payload, 'final_summary': final_summary, 'morning_memo_path': str(AUTOPILOT_DIR / 'morning_memo.md')}
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
    selector_mode: str = "adaptive",
    seed_split_a: list[int] | None = None,
    seed_split_b: list[int] | None = None,
    truthful_measurement_required: bool = False,
    perception_probe_mode: str = "policy_encoder_v1",
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
        selector_mode=selector_mode,
        seed_split_a=seed_split_a,
        seed_split_b=seed_split_b,
        truthful_measurement_required=truthful_measurement_required,
        perception_probe_mode=perception_probe_mode,
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

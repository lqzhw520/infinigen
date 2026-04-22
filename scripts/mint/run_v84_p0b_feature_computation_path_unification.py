#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
MINT_SCRIPTS = PROJECT_ROOT / "scripts" / "mint"
sys.path.insert(0, str(MINT_SCRIPTS))

from drawer_robot_env_mujoco import (  # noqa: E402
    DrawerEnvContractConfig,
    ORIENTATION_BRIDGE_STATE_DIM_NAMES,
    build_robot_rollout,
    reconstruct_orientation_bridge_state_trace,
    save_robot_rollout,
)
from root_cause_controller import RootCauseController  # noqa: E402
from v13_audit_common import (  # noqa: E402
    ACCEPTANCE_CONTRACT_PATH,
    ARTIFACT_DIR,
    AUTOPILOT_DIR,
    PLAN_PATH,
    TRUTH_CONTRACT_PATH,
    current_repo_identity,
    load_json,
    make_run_instance,
    save_active_plan,
    sha256_file,
    teacher_family_dispatch_payload,
    write_gate,
    write_json_atomic,
)

SPEC_REFERENCE = str(PROJECT_ROOT / "docs" / "MINT_V84_P0_INFRASTRUCTURE_REPAIR_SPEC.md")
PLAN_OUTPUT_PATH = ARTIFACT_DIR / "p0_infrastructure_repair_plan.json"
OUTPUT_PATH = ARTIFACT_DIR / "p0b_feature_computation_path_unification.json"
ROLL_OUT_DIR = ARTIFACT_DIR / "p0b_feature_unification_rollouts"
GATE_PATH = AUTOPILOT_DIR / "gates_p0_infrastructure" / "P0b_feature_computation_path_unification.json"
SOURCE_P0A_GATE = AUTOPILOT_DIR / "gates_p0_infrastructure" / "P0a_live_rollout_family_dispatch_audit.json"
SOURCE_P0A_ARTIFACT = ARTIFACT_DIR / "p0a_live_rollout_family_dispatch_audit.json"
TARGET_SEEDS = [1, 3, 5, 6, 7, 8]
TARGET_EPISODE_INDEX = 0
PHASE1_FEATURES = [
    "approach_alignment_cos",
    "orientation_alignment_cos",
    "orientation_error_cos",
]
CONTINUOUS_FEATURES = [
    "distance_to_handle_norm",
    "approach_alignment_cos",
    "orientation_alignment_cos",
    "orientation_error_sin",
    "orientation_error_cos",
]


def _make_plan() -> dict[str, Any]:
    ident = current_repo_identity()
    run_instance_id = make_run_instance("p0b", ident["working_head_commit"])
    p0a_gate = load_json(SOURCE_P0A_GATE, {})
    p0a_artifact = load_json(SOURCE_P0A_ARTIFACT, {})
    plan = {
        "run_instance_id": run_instance_id,
        "plan_version": "tiny_retrain_confirmation_p0_infrastructure_repair",
        "slice_name": "p0b-feature-computation-path-unification",
        "working_head_commit": ident["working_head_commit"],
        "vendor_head_commit": ident["vendor_head_commit"],
        "branch": ident["branch"],
        "execution_scope": "p0b_feature_computation_path_unification",
        "bridge_stage": "p0_infrastructure_repair",
        "diagnostic_only": True,
        "claim_bearing": False,
        "training_allowed": False,
        "probe_allowed": False,
        "publication_scope": "blocked_before_canary",
        "result_scope": "p0_infrastructure_only",
        "spec_reference": SPEC_REFERENCE,
        "source_p0a_run_instance_id": p0a_artifact.get("run_instance_id"),
        "source_p0a_gate_status": p0a_gate.get("status"),
        "truth_contract_path": str(TRUTH_CONTRACT_PATH),
        "truth_contract_hash": sha256_file(TRUTH_CONTRACT_PATH),
        "acceptance_contract_path": str(ACCEPTANCE_CONTRACT_PATH),
        "acceptance_contract_hash": sha256_file(ACCEPTANCE_CONTRACT_PATH),
        "allowed_next_phases": ["P0b", "Re-C1"],
    }
    write_json_atomic(PLAN_OUTPUT_PATH, plan)
    save_active_plan(plan)
    return plan


def _live_contract() -> DrawerEnvContractConfig:
    controller = RootCauseController()
    payload = dict(
        controller._frozen_matrix_contracts_v5_pro()["V1cT2S3"]["env_contract_config"]
    )
    return DrawerEnvContractConfig(**payload)


def _load_rollout_arrays(meta_path: Path) -> dict[str, np.ndarray]:
    data = np.load(meta_path.with_suffix(".npz"), allow_pickle=True)
    return {key: np.asarray(data[key]) for key in data.files}


def _parity_summary(live: np.ndarray, diag: np.ndarray) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for idx, name in enumerate(ORIENTATION_BRIDGE_STATE_DIM_NAMES):
        diff = np.abs(live[:, idx] - diag[:, idx])
        if name in CONTINUOUS_FEATURES:
            summary[name] = {
                "p95_abs_diff": float(np.quantile(diff, 0.95)),
                "mean_abs_diff": float(np.mean(diff)),
                "max_abs_diff": float(np.max(diff)),
            }
        else:
            summary[name] = {
                "exact_agreement_rate": float(np.mean(live[:, idx] == diag[:, idx])),
                "mismatch_count": int(np.sum(live[:, idx] != diag[:, idx])),
            }
    return summary


def run() -> int:
    plan = _make_plan()
    p0a_gate = load_json(SOURCE_P0A_GATE, {})
    blocking: list[str] = []
    if str(p0a_gate.get("status") or "") != "PASS":
        blocking.append("p0a_not_passed")

    if ROLL_OUT_DIR.exists():
        for path in sorted(ROLL_OUT_DIR.glob("*")):
            if path.is_file():
                path.unlink()
            else:
                import shutil

                shutil.rmtree(path)
    ROLL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    contract = _live_contract()
    episode_rows: list[dict[str, Any]] = []
    aggregate_continuous: dict[str, list[float]] = {name: [] for name in CONTINUOUS_FEATURES}
    aggregate_attach: list[float] = []

    if not blocking:
        for seed in TARGET_SEEDS:
            rollout = build_robot_rollout(
                seed=seed,
                grasp_pose_world=np.eye(4, dtype=np.float32),
                episode_index=TARGET_EPISODE_INDEX,
                image_size=64,
                max_steps=96,
                contract=contract,
                rotation_source="aligned",
                claim_policy="diagnostic",
                teacher_controller_mode="interaction_frame_hybrid",
                interventions=teacher_family_dispatch_payload(TARGET_EPISODE_INDEX),
            )
            rollout["run_instance_id"] = plan["run_instance_id"]
            rollout["plan_version"] = plan["plan_version"]
            rollout["working_head_commit"] = plan["working_head_commit"]
            rollout["bridge_stage"] = plan["execution_scope"]
            rollout["bridge_attempt"] = "p0b_feature_unification"
            base = ROLL_OUT_DIR / f"seed_{seed:03d}_episode_{TARGET_EPISODE_INDEX:02d}"
            save_robot_rollout(base, rollout)

            arrays = _load_rollout_arrays(base.with_suffix(".json"))
            live_states = np.asarray(arrays["states"], dtype=np.float32)
            diag_states = reconstruct_orientation_bridge_state_trace(
                handle_distance_trace=np.asarray(arrays["state_handle_distance_trace"], dtype=np.float32),
                approach_alignment_trace=np.asarray(arrays["approach_alignment_trace"], dtype=np.float32),
                orientation_alignment_trace=np.asarray(arrays["orientation_alignment_trace"], dtype=np.float32),
                orientation_error_trace=np.asarray(arrays["orientation_error_trace"], dtype=np.float32),
                actions=np.asarray(arrays["actions"], dtype=np.float32),
                attach_eligible_trace=np.asarray(arrays["attach_eligible_trace"], dtype=bool),
                initial_state_frame=live_states[0],
            )
            parity = _parity_summary(live_states, diag_states)
            for name in CONTINUOUS_FEATURES:
                aggregate_continuous[name].append(parity[name]["p95_abs_diff"])
            aggregate_attach.append(
                parity["attach_eligible_proxy"]["exact_agreement_rate"]
            )
            episode_rows.append(
                {
                    "seed": seed,
                    "episode_index": TARGET_EPISODE_INDEX,
                    "frame_count": int(live_states.shape[0]),
                    "rollout_path": str(base.with_suffix(".json")),
                    "parity": parity,
                }
            )

    phase1_pass = (
        not blocking
        and all(
            max(aggregate_continuous[name] or [1.0]) < 0.5
            for name in PHASE1_FEATURES
        )
    )
    phase2_pass = (
        not blocking
        and all(
            max(aggregate_continuous[name] or [1.0]) < 1e-3
            for name in CONTINUOUS_FEATURES
        )
        and min(aggregate_attach or [0.0]) > 0.99
    )
    if not phase1_pass:
        blocking.append("p0b_phase1_not_passed")
    if phase1_pass and not phase2_pass:
        blocking.append("p0b_phase2_not_passed")

    artifact = {
        "audit": "p0b_feature_computation_path_unification",
        "run_instance_id": plan["run_instance_id"],
        "plan_version": plan["plan_version"],
        "working_head_commit": plan["working_head_commit"],
        "execution_scope": plan["execution_scope"],
        "diagnostic_only": True,
        "claim_bearing": False,
        "source_p0a_gate_status": p0a_gate.get("status"),
        "canonical_source_functions": {
            "live_state_vector": "drawer_robot_env_mujoco.compute_orientation_bridge_state_vector",
            "offline_reconstruction": "drawer_robot_env_mujoco.reconstruct_orientation_bridge_state_trace",
        },
        "single_source_target_features": ORIENTATION_BRIDGE_STATE_DIM_NAMES,
        "episode_set": [{"seed": seed, "episode_index": TARGET_EPISODE_INDEX} for seed in TARGET_SEEDS],
        "episode_parity_summary": episode_rows,
        "phase_1_metrics": {
            name: max(aggregate_continuous[name] or [1.0]) for name in PHASE1_FEATURES
        },
        "phase_1_pass": phase1_pass,
        "phase_2_metrics": {
            **{
                name: max(aggregate_continuous[name] or [1.0])
                for name in CONTINUOUS_FEATURES
            },
            "attach_eligible_proxy_min_exact_agreement_rate": min(aggregate_attach or [0.0]),
        },
        "phase_2_pass": phase2_pass,
        "p0b_status": "PASS" if phase1_pass and phase2_pass else "FAIL",
    }
    write_json_atomic(OUTPUT_PATH, artifact)

    gate = write_gate(
        gate_path=GATE_PATH,
        gate_id="P0b",
        gate_name="feature_computation_path_unification",
        run_instance_id=plan["run_instance_id"],
        status="PASS" if phase1_pass and phase2_pass else "STOP",
        blocking_reasons=blocking,
        allowed_next_phases=["Re-C1"] if phase1_pass and phase2_pass else [],
        extra={
            "audit_artifact_path": str(OUTPUT_PATH),
            "phase_1_pass": phase1_pass,
            "phase_2_pass": phase2_pass,
            "phase_1_metrics": artifact["phase_1_metrics"],
            "phase_2_metrics": artifact["phase_2_metrics"],
            "canonical_source_functions": artifact["canonical_source_functions"],
            "episode_set": artifact["episode_set"],
        },
        spec_reference=SPEC_REFERENCE,
    )
    print(json.dumps({"artifact": artifact, "gate": gate}, indent=2))
    return 0 if phase1_pass and phase2_pass else 1


if __name__ == "__main__":
    raise SystemExit(run())

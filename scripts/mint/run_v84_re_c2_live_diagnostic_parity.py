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
OUTPUT_PATH = ARTIFACT_DIR / "re_c2_live_diagnostic_parity.json"
ROLL_OUT_DIR = ARTIFACT_DIR / "re_c2_parity_rollouts"
GATE_PATH = AUTOPILOT_DIR / "gates_p0_infrastructure" / "Re_C2_live_diagnostic_parity.json"
SOURCE_RE_C1_GATE = AUTOPILOT_DIR / "gates_p0_infrastructure" / "Re_C1_live_support_expansion.json"
TARGET_EPISODES = [(1, 0), (3, 0), (5, 0), (6, 0), (7, 0), (8, 0)]
CONTINUOUS_FEATURES = [
    "distance_to_handle_norm",
    "approach_alignment_cos",
    "orientation_alignment_cos",
    "orientation_error_sin",
    "orientation_error_cos",
]


def _make_plan() -> dict[str, Any]:
    ident = current_repo_identity()
    run_instance_id = make_run_instance("rec2", ident["working_head_commit"])
    plan = {
        "run_instance_id": run_instance_id,
        "plan_version": "tiny_retrain_confirmation_p0_infrastructure_repair",
        "slice_name": "re-c2-live-diagnostic-parity",
        "working_head_commit": ident["working_head_commit"],
        "vendor_head_commit": ident["vendor_head_commit"],
        "branch": ident["branch"],
        "execution_scope": "re_c2_live_diagnostic_parity",
        "bridge_stage": "p0_infrastructure_repair",
        "diagnostic_only": True,
        "claim_bearing": False,
        "training_allowed": False,
        "probe_allowed": False,
        "spec_reference": SPEC_REFERENCE,
        "truth_contract_hash": sha256_file(TRUTH_CONTRACT_PATH),
        "acceptance_contract_hash": sha256_file(ACCEPTANCE_CONTRACT_PATH),
        "allowed_next_phases": ["Re-C2", "Re-C3"],
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
    rec1_gate = load_json(SOURCE_RE_C1_GATE, {})
    blocking: list[str] = []
    if str(rec1_gate.get("status") or "") != "PASS":
        blocking.append("re_c1_not_passed")

    if ROLL_OUT_DIR.exists():
        import shutil

        shutil.rmtree(ROLL_OUT_DIR)
    ROLL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    episode_rows: list[dict[str, Any]] = []
    aggregate_continuous: dict[str, list[float]] = {name: [] for name in CONTINUOUS_FEATURES}
    aggregate_attach: list[float] = []

    if not blocking:
        contract = _live_contract()
        for seed, episode_index in TARGET_EPISODES:
            rollout = build_robot_rollout(
                seed=seed,
                grasp_pose_world=np.eye(4, dtype=np.float32),
                episode_index=episode_index,
                image_size=64,
                max_steps=96,
                contract=contract,
                rotation_source="aligned",
                claim_policy="diagnostic",
                teacher_controller_mode="interaction_frame_hybrid",
                interventions=teacher_family_dispatch_payload(episode_index),
            )
            rollout["run_instance_id"] = plan["run_instance_id"]
            rollout["plan_version"] = plan["plan_version"]
            rollout["working_head_commit"] = plan["working_head_commit"]
            rollout["bridge_stage"] = plan["execution_scope"]
            rollout["bridge_attempt"] = "re_c2_live_diagnostic_parity"
            base = ROLL_OUT_DIR / f"seed_{seed:03d}_episode_{episode_index:02d}"
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
                    "episode_index": episode_index,
                    "frame_count": int(live_states.shape[0]),
                    "rollout_path": str(base.with_suffix(".json")),
                    "parity": parity,
                }
            )
        for name in CONTINUOUS_FEATURES:
            if max(aggregate_continuous[name] or [1.0]) >= 1e-3:
                blocking.append(f"{name}_p95_abs_diff_ge_1e-3")
        if min(aggregate_attach or [0.0]) <= 0.99:
            blocking.append("attach_eligible_proxy_exact_agreement_le_99pct")

    artifact = {
        "audit": "re_c2_live_diagnostic_parity",
        "run_instance_id": plan["run_instance_id"],
        "plan_version": plan["plan_version"],
        "working_head_commit": plan["working_head_commit"],
        "execution_scope": plan["execution_scope"],
        "diagnostic_only": True,
        "claim_bearing": False,
        "episode_set": [{"seed": s, "episode_index": e} for s, e in TARGET_EPISODES],
        "episode_parity_summary": episode_rows,
        "phase_2_metrics": {
            **{
                name: max(aggregate_continuous[name] or [1.0])
                for name in CONTINUOUS_FEATURES
            },
            "attach_eligible_proxy_min_exact_agreement_rate": min(aggregate_attach or [0.0]),
        },
        "re_c2_status": "PASS" if not blocking else "FAIL",
    }
    write_json_atomic(OUTPUT_PATH, artifact)

    gate = write_gate(
        gate_path=GATE_PATH,
        gate_id="Re-C2",
        gate_name="live_diagnostic_parity",
        run_instance_id=plan["run_instance_id"],
        status="PASS" if not blocking else "STOP",
        blocking_reasons=blocking,
        allowed_next_phases=["Re-C3"] if not blocking else [],
        extra={
            "audit_artifact_path": str(OUTPUT_PATH),
            "phase_2_metrics": artifact["phase_2_metrics"],
            "episode_set": artifact["episode_set"],
            "rollout_dir": str(ROLL_OUT_DIR),
        },
        spec_reference=SPEC_REFERENCE,
    )
    print(json.dumps({"artifact": artifact, "gate": gate}, indent=2))
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(run())

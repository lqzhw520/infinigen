#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v13_audit_common import (
    ARTIFACT_DIR,
    GATES_V13_DIR,
    GATES_V13_SLICE2_DIR,
    V13_SPEC_REFERENCE,
    current_repo_identity,
    load_active_plan,
    load_json,
    make_run_instance,
    save_active_plan,
    write_gate,
    write_json_atomic,
)

OUTPUT_PATH = ARTIFACT_DIR / "v13_slice2_state_action_interface_wiring_plan.json"
GATE_PATH = GATES_V13_SLICE2_DIR / "G0_sovereign_sync.json"


def run() -> int:
    ident = current_repo_identity()
    prior_plan = load_active_plan()
    prior_summary = load_json(ARTIFACT_DIR / "v13_g1_g4_tranche_summary.json", {})
    prior_g0 = load_json(GATES_V13_DIR / "G0_sovereign_sync.json", {})
    prior_g1 = load_json(GATES_V13_DIR / "G1_classification_sanity.json", {})
    prior_g2 = load_json(GATES_V13_DIR / "G2_action_interface_audit.json", {})
    prior_g3 = load_json(GATES_V13_DIR / "G3_orientation_state_consumption.json", {})
    prior_g4 = load_json(GATES_V13_DIR / "G4_one_step_supervised_fit.json", {})

    blocking: list[str] = []
    for gate_name, payload in {
        "G0": prior_g0,
        "G1": prior_g1,
        "G2": prior_g2,
        "G3": prior_g3,
        "G4": prior_g4,
    }.items():
        if payload.get("status") != "PASS":
            blocking.append(f"prior_{gate_name}_not_passed")
    if prior_summary.get("stop_point") != "G4":
        blocking.append("prior_v13_stop_point_unexpected")
    if prior_summary.get("one_step_fit_status") != "one_step_action_fit_ok":
        blocking.append("prior_v13_one_step_fit_not_ok")

    run_instance_id = make_run_instance("v13s2", ident["working_head_commit"])
    plan = {
        "run_instance_id": run_instance_id,
        "plan_version": "tiny_retrain_confirmation_v13_slice_2_state_action_wiring",
        "slice_name": "v13-slice-2-state-action-interface-wiring-only",
        "working_head_commit": ident["working_head_commit"],
        "current_publication_head_commit": ident["working_head_commit"],
        "vendor_head_commit": ident["vendor_head_commit"],
        "branch": ident["branch"],
        "source_v13_run_instance_id": prior_summary.get("run_instance_id"),
        "source_v13_working_head_commit": prior_summary.get("working_head_commit"),
        "source_v13_execution_scope": prior_summary.get("execution_scope"),
        "source_v13_action_interface_status": prior_g2.get("action_interface_status"),
        "source_v13_orientation_state_status": prior_g3.get("orientation_state_consumption_status"),
        "source_v13_one_step_fit_status": prior_g4.get("one_step_fit_status"),
        "source_canonical_train_cell": prior_plan.get("source_canonical_train_cell") or prior_plan.get("canonical_train_cell") or "V1cT2S0",
        "canonical_train_cell": "V1cT2S3",
        "source_best_train_state_mode": prior_plan.get("source_best_train_state_mode") or prior_plan.get("best_train_state_mode") or "S0",
        "best_train_state_mode": "S3",
        "active_train_state_mode": "S3",
        "active_state_mode_name": "orientation_bridge_state_v1",
        "execution_scope": "v13_slice_2_sovereign_sync",
        "bridge_stage": "v13_slice_2_state_action_interface_wiring_only",
        "bridge_attempt": "diagnostic_wiring_only",
        "dataset_selection_mode": "diagnostic_state_action_interface_wiring",
        "diagnostic_only": True,
        "claim_bearing": False,
        "publication_scope": "blocked_before_g5",
        "result_scope": "pre_train_wiring_only",
        "allowed_next_phases": ["G2", "G3", "G4"],
        "training_allowed": False,
        "spec_reference": V13_SPEC_REFERENCE,
        "goal": "Wire orientation_bridge_state_v1 into the live consumed path and rerun only bounded G2/G3/G4 audits.",
        "minimum_change_set": [
            "scripts/mint/drawer_robot_env_mujoco.py",
            "scripts/mint/root_cause_controller.py",
            "scripts/mint/v13_audit_common.py",
            "scripts/mint/run_v84_v13_slice2_sovereign_sync.py",
            "scripts/mint/run_v84_v13_slice2_action_interface_audit.py",
            "scripts/mint/run_v84_v13_slice2_state_consumption_audit.py",
            "scripts/mint/run_v84_v13_slice2_one_step_fit_audit.py",
        ],
        "constraints": [
            "Do not start training.",
            "Do not start probe.",
            "Do not modify external/MINT.",
            "Do not change claim-bearing truth predicates or thresholds.",
            "Do not broaden into AnyGrasp or unrelated legacy artifacts.",
        ],
        "gate_design": {
            "G2": "Recheck live close/action interface semantics after wiring.",
            "G3": "Materialize a small live rollout set with orientation_bridge_state_v1 and verify those features are truly in observation.state/model input.",
            "G4": "Run one-step supervised fit on the live-wired state only; no training/probe beyond that.",
        },
        "decision_boundary": {
            "allow_later_train_probe_only_if": ["G2 PASS", "G3 PASS", "G4 PASS"],
            "otherwise": "STOP and refine wiring rather than training.",
        },
    }
    save_active_plan(plan)
    artifact = {**plan, "hard_freeze_reconciled": True, "prior_v13_gates": prior_summary.get("gates", {})}
    write_json_atomic(OUTPUT_PATH, artifact)
    gate = write_gate(
        gate_path=GATE_PATH,
        gate_id="G0",
        gate_name="sovereign_sync",
        run_instance_id=run_instance_id,
        status="PASS" if not blocking else "STOP",
        blocking_reasons=blocking,
        allowed_next_phases=["G2", "G3", "G4"] if not blocking else [],
        extra={
            "audit_artifact_path": str(OUTPUT_PATH),
            "source_v13_run_instance_id": plan["source_v13_run_instance_id"],
            "source_v13_one_step_fit_status": plan["source_v13_one_step_fit_status"],
            "target_live_state_mode": plan["active_state_mode_name"],
            "target_live_train_cell": plan["canonical_train_cell"],
            "hard_freeze_reconciled": True,
        },
    )
    print(json.dumps({"artifact": artifact, "gate": gate}, indent=2))
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(run())


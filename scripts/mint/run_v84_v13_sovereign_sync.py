#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v13_audit_common import (
    ARTIFACT_DIR,
    GATES_V13_DIR,
    V13_SPEC_REFERENCE,
    current_repo_identity,
    load_json,
    make_run_instance,
    save_active_plan,
    source_v12_artifact,
    source_v12_gate,
    write_gate,
    write_json_atomic,
)

OUTPUT_PATH = ARTIFACT_DIR / "v13_g1_g4_tranche_plan.json"
GATE_PATH = GATES_V13_DIR / "G0_sovereign_sync.json"


def _v12_required_state() -> tuple[list[str], dict[str, Any]]:
    required = {
        "G1": source_v12_gate("G1_orientation_stage_audit.json"),
        "G1b": source_v12_gate("G1b_anygrasp_orientation_prior_audit.json"),
        "G2": source_v12_gate("G2_state_conditioning_audit.json"),
        "G3": source_v12_gate("G3_orientation_support_corpus.json"),
        "G4": source_v12_gate("G4_orientation_trainability_contract.json"),
        "G5": source_v12_gate("G5_two_stage_orientation_bridge_training.json"),
        "G6": source_v12_gate("G6_orientation_first_probe.json"),
    }
    probe = source_v12_artifact("g8_train_seed_probe.json")
    blocking: list[str] = []
    if required["G1"].get("status") != "PASS":
        blocking.append("v12_G1_not_passed")
    if required["G1b"].get("status") != "STOP":
        blocking.append("v12_G1b_status_unexpected")
    elif "anygrasp_sidecar_unavailable" not in (required["G1b"].get("blocking_reasons") or []):
        blocking.append("v12_G1b_blocker_unexpected")
    for gate_id in ["G2", "G3", "G4", "G5"]:
        if required[gate_id].get("status") != "PASS":
            blocking.append(f"v12_{gate_id}_not_passed")
    if required["G6"].get("status") not in {"PASS", "DIAGNOSTIC_PASS"}:
        blocking.append("v12_G6_not_complete")
    if not bool(required["G4"].get("orientation_trainability_passed", False)):
        blocking.append("v12_orientation_trainability_not_passed")
    if not probe:
        blocking.append("v12_probe_summary_missing")
    else:
        zero_gains = [
            float(probe.get("ever_attach_eligible_fraction_gain", 0.0) or 0.0) == 0.0,
            float(probe.get("ever_attached_rate_gain", 0.0) or 0.0) == 0.0,
            float(probe.get("stable_attach_gain", 0.0) or 0.0) == 0.0,
            float(probe.get("phase_locked_gain", 0.0) or 0.0) == 0.0,
            bool(probe.get("attach_bridge_pass", False)) is False,
        ]
        if not all(zero_gains):
            blocking.append("v12_attach_bridge_not_zero")
    return blocking, {**required, "probe": probe}


def run() -> int:
    ident = current_repo_identity()
    blocking, v12 = _v12_required_state()
    source_plan = load_json(ARTIFACT_DIR / "active_tiny_retrain_plan.json", {})
    run_instance_id = make_run_instance("v13", ident["working_head_commit"])
    plan = {
        "run_instance_id": run_instance_id,
        "plan_version": "tiny_retrain_confirmation_v13_state_action_interface",
        "working_head_commit": ident["working_head_commit"],
        "current_publication_head_commit": ident["working_head_commit"],
        "vendor_head_commit": ident["vendor_head_commit"],
        "branch": ident["branch"],
        "source_v12_run_instance_id": v12["G6"].get("run_instance_id"),
        "source_v12_working_head_commit": v12["G6"].get("working_head_commit"),
        "source_v12_emitted_g6_classification": v12["G6"].get("classification"),
        "source_base_commit": v12["probe"].get("source_base_commit"),
        "source_canonical_train_cell": source_plan.get("source_canonical_train_cell") or source_plan.get("canonical_train_cell") or "V1cT2S0",
        "canonical_train_cell": source_plan.get("source_canonical_train_cell") or source_plan.get("canonical_train_cell") or "V1cT2S0",
        "source_best_train_state_mode": source_plan.get("source_best_train_state_mode") or source_plan.get("best_train_state_mode") or "S0",
        "best_train_state_mode": source_plan.get("source_best_train_state_mode") or source_plan.get("best_train_state_mode") or "S0",
        "active_train_state_mode": source_plan.get("active_train_state_mode") or source_plan.get("source_best_train_state_mode") or "S0",
        "active_state_mode_name": source_plan.get("active_state_mode_name") or "m0_proxy",
        "execution_scope": "v13_g1_g4_audit",
        "bridge_stage": "v13_g1_g4_audit",
        "bridge_attempt": "diagnostic",
        "dataset_selection_mode": "diagnostic_state_action_interface",
        "diagnostic_only": True,
        "claim_bearing": False,
        "publication_scope": "pending_v13_g9",
        "result_scope": "pending_v13_g8",
        "allowed_next_phases": ["G1", "G2", "G3", "G4"],
        "training_allowed": False,
        "spec_reference": V13_SPEC_REFERENCE,
    }
    save_active_plan(plan)
    artifact = {
        **plan,
        "hard_freeze_reconciled": True,
        "v12_import_summary": {
            "g1_status": v12["G1"].get("status"),
            "g1b_status": v12["G1b"].get("status"),
            "g2_status": v12["G2"].get("status"),
            "g3_status": v12["G3"].get("status"),
            "g4_status": v12["G4"].get("status"),
            "g5_status": v12["G5"].get("status"),
            "g6_status": v12["G6"].get("status"),
            "attach_bridge_pass": v12["probe"].get("attach_bridge_pass"),
            "ever_attach_eligible_fraction_gain": v12["probe"].get("ever_attach_eligible_fraction_gain"),
            "ever_attached_rate_gain": v12["probe"].get("ever_attached_rate_gain"),
            "stable_attach_gain": v12["probe"].get("stable_attach_gain"),
            "phase_locked_gain": v12["probe"].get("phase_locked_gain"),
        },
    }
    write_json_atomic(OUTPUT_PATH, artifact)
    gate = write_gate(
        gate_path=GATE_PATH,
        gate_id="G0",
        gate_name="sovereign_sync",
        run_instance_id=run_instance_id,
        status="PASS" if not blocking else "STOP",
        blocking_reasons=blocking,
        allowed_next_phases=["G1", "G2", "G3", "G4"] if not blocking else [],
        extra={
            "audit_artifact_path": str(OUTPUT_PATH),
            "source_v12_run_instance_id": plan["source_v12_run_instance_id"],
            "source_v12_working_head_commit": plan["source_v12_working_head_commit"],
            "source_v12_emitted_g6_classification": plan["source_v12_emitted_g6_classification"],
            "hard_freeze_reconciled": True,
        },
    )
    print(json.dumps({"artifact": artifact, "gate": gate}, indent=2))
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(run())

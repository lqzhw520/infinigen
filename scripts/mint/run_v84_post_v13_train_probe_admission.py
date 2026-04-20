#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from v13_audit_common import (
    ARTIFACT_DIR,
    AUTOPILOT_DIR,
    V12_SPEC_REFERENCE,
    V13_SPEC_REFERENCE,
    current_repo_identity,
    ensure_v13_slice2_plan,
    load_json,
    make_run_instance,
    save_active_plan,
    utc_now,
    write_gate,
    write_json_atomic,
)

POST_V13_SPEC_REFERENCE = "/Users/zhuhaowu/ws/phd-anyboxs/infinigen_from_servers/docs/gpt5.4Pro/codex_mint_v84_post_v13_train_probe_admission_execution_spec.md"
PLAN_VERSION = "tiny_retrain_confirmation_post_v13_admission"
GATES_POST_V13_DIR = AUTOPILOT_DIR / "gates_post_v13"
PLAN_ARTIFACT_PATH = ARTIFACT_DIR / "post_v13_admission_plan.json"
SUMMARY_PATH = ARTIFACT_DIR / "post_v13_admission_summary.json"

GATES_V12_DIR = AUTOPILOT_DIR / "gates_v12"
GATES_V13_DIR = AUTOPILOT_DIR / "gates_v13"
GATES_V13_SLICE2_DIR = AUTOPILOT_DIR / "gates_v13_slice2"

EXPECTED_HEAD = "bc802069816e6c81beab48ba1f2141c9f3a264d4"
EXPECTED_VENDOR = "4eab5795345721001c412ff1ca2c886a11eab606"
EXPECTED_BRANCH = "feature/mint-env-reformulation-v1-visual-fidelity"
EXPECTED_SLICE2_RUN = "v13s2_20260420T121619Z_bc802069_40636282"
EXPECTED_SLICE2_PLAN = "tiny_retrain_confirmation_v13_slice_2_state_action_wiring"
EXPECTED_SLICE2_SCOPE = "v13_slice_2_one_step_fit"


def _load(path: Path) -> dict[str, Any]:
    return load_json(path, {})


def _all_zero_attach(v12_gate: dict[str, Any]) -> bool:
    keys = [
        "ever_attach_eligible_fraction_gain",
        "ever_attached_rate_gain",
        "stable_attach_gain",
        "phase_locked_gain",
    ]
    return all(float(v12_gate.get(key, 0.0) or 0.0) == 0.0 for key in keys)


def _gate_pass(payload: dict[str, Any]) -> bool:
    return str(payload.get("status") or "") == "PASS"


def _gate_completed(payload: dict[str, Any]) -> bool:
    return str(payload.get("status") or "") in {"PASS", "DIAGNOSTIC_PASS"}


def _build_plan(run_instance_id: str, source_slice2_plan: dict[str, Any], ident: dict[str, str]) -> dict[str, Any]:
    return {
        "run_instance_id": run_instance_id,
        "plan_version": PLAN_VERSION,
        "slice_name": "post-v13-canary-train-probe-admission",
        "working_head_commit": ident["working_head_commit"],
        "current_publication_head_commit": ident["working_head_commit"],
        "vendor_head_commit": ident["vendor_head_commit"],
        "branch": ident["branch"],
        "source_slice2_run_instance_id": source_slice2_plan.get("run_instance_id"),
        "source_slice2_plan_version": source_slice2_plan.get("plan_version"),
        "source_slice2_execution_scope": source_slice2_plan.get("execution_scope"),
        "execution_scope": "post_v13_admission_verdict",
        "bridge_stage": "post_v13_admission",
        "dataset_selection_mode": "none_decision_only",
        "diagnostic_only": True,
        "claim_bearing": False,
        "training_allowed": False,
        "probe_allowed": False,
        "publication_scope": "blocked_before_canary",
        "result_scope": "admission_decision_only",
        "allowed_next_phases": ["D0", "D1", "D2", "D3", "D4"],
        "spec_reference": POST_V13_SPEC_REFERENCE,
        "carryforward_spec_references": [
            V12_SPEC_REFERENCE,
            V13_SPEC_REFERENCE,
        ],
        "goal": "Decide whether v13 and v13-slice-2 justify a tightly bounded canary train/probe tranche, without starting train/probe.",
        "minimum_change_set": [
            "scripts/mint/run_v84_post_v13_train_probe_admission.py",
        ],
        "constraints": [
            "Do not start training.",
            "Do not start probe.",
            "Do not modify external/MINT.",
            "Do not change claim-bearing truth predicates or thresholds.",
            "Do not reopen earlier corpus blockers without hard contradiction.",
        ],
        "expected_outputs": [
            "experiments/mint/mint_drawer_v1/autopilot/gates_post_v13/D0_authority_refreeze.json",
            "experiments/mint/mint_drawer_v1/autopilot/gates_post_v13/D1_carryforward_evidence_pack.json",
            "experiments/mint/mint_drawer_v1/autopilot/gates_post_v13/D2_interface_closure_contract.json",
            "experiments/mint/mint_drawer_v1/autopilot/gates_post_v13/D3_canary_design_gate.json",
            "experiments/mint/mint_drawer_v1/autopilot/gates_post_v13/D4_admission_verdict.json",
            "experiments/mint/mint_drawer_v1/artifacts/post_v13_admission_summary.json",
        ],
    }


def run() -> int:
    ident = current_repo_identity()
    active_slice2_plan = ensure_v13_slice2_plan()
    run_instance_id = make_run_instance("postv13", ident["working_head_commit"])

    d0_path = GATES_POST_V13_DIR / "D0_authority_refreeze.json"
    d1_path = GATES_POST_V13_DIR / "D1_carryforward_evidence_pack.json"
    d2_path = GATES_POST_V13_DIR / "D2_interface_closure_contract.json"
    d3_path = GATES_POST_V13_DIR / "D3_canary_design_gate.json"
    d4_path = GATES_POST_V13_DIR / "D4_admission_verdict.json"

    v12_g6_gate = _load(GATES_V12_DIR / "G6_orientation_first_probe.json")
    v12_probe = _load(ARTIFACT_DIR / "g8_train_seed_probe.json")
    v12_attach = _load(ARTIFACT_DIR / "g8_attach_bridge_summary.json")

    v13_summary = _load(ARTIFACT_DIR / "v13_g1_g4_tranche_summary.json")
    v13_class = _load(ARTIFACT_DIR / "v13_v12_classification_sanity_audit.json")
    v13_action = _load(ARTIFACT_DIR / "v13_action_interface_audit.json")
    v13_state = _load(ARTIFACT_DIR / "v13_orientation_state_consumption_audit.json")
    v13_fit = _load(ARTIFACT_DIR / "v13_one_step_supervised_fit_audit.json")
    v13_g0 = _load(GATES_V13_DIR / "G0_sovereign_sync.json")
    v13_g1 = _load(GATES_V13_DIR / "G1_classification_sanity.json")
    v13_g2 = _load(GATES_V13_DIR / "G2_action_interface_audit.json")
    v13_g3 = _load(GATES_V13_DIR / "G3_orientation_state_consumption.json")
    v13_g4 = _load(GATES_V13_DIR / "G4_one_step_supervised_fit.json")

    v13s2_summary = _load(ARTIFACT_DIR / "v13_slice2_tranche_summary.json")
    v13s2_action = _load(ARTIFACT_DIR / "v13_slice2_action_interface_audit.json")
    v13s2_state = _load(ARTIFACT_DIR / "v13_slice2_state_consumption_audit.json")
    v13s2_fit = _load(ARTIFACT_DIR / "v13_slice2_one_step_fit_audit.json")
    v13s2_g0 = _load(GATES_V13_SLICE2_DIR / "G0_sovereign_sync.json")
    v13s2_g2 = _load(GATES_V13_SLICE2_DIR / "G2_action_interface_audit.json")
    v13s2_g3 = _load(GATES_V13_SLICE2_DIR / "G3_state_consumption.json")
    v13s2_g4 = _load(GATES_V13_SLICE2_DIR / "G4_one_step_fit.json")

    d0_blocking: list[str] = []
    if ident.get("branch") != EXPECTED_BRANCH:
        d0_blocking.append("branch_drift")
    if ident.get("working_head_commit") != EXPECTED_HEAD:
        d0_blocking.append("head_drift")
    if ident.get("vendor_head_commit") != EXPECTED_VENDOR:
        d0_blocking.append("vendor_drift")
    if active_slice2_plan.get("run_instance_id") != EXPECTED_SLICE2_RUN:
        d0_blocking.append("active_slice2_run_instance_mismatch")
    if active_slice2_plan.get("plan_version") != EXPECTED_SLICE2_PLAN:
        d0_blocking.append("active_slice2_plan_version_mismatch")
    if active_slice2_plan.get("execution_scope") != EXPECTED_SLICE2_SCOPE:
        d0_blocking.append("active_slice2_execution_scope_mismatch")
    d0_status = "PASS" if not d0_blocking else "STOP_authority_drift"
    d0_gate = write_gate(
        gate_path=d0_path,
        gate_id="D0",
        gate_name="authority_refreeze",
        run_instance_id=run_instance_id,
        status=d0_status,
        blocking_reasons=d0_blocking,
        allowed_next_phases=["D1"] if d0_status == "PASS" else [],
        extra={
            "expected_head_commit": EXPECTED_HEAD,
            "expected_vendor_head_commit": EXPECTED_VENDOR,
            "expected_slice2_run_instance_id": EXPECTED_SLICE2_RUN,
            "expected_slice2_execution_scope": EXPECTED_SLICE2_SCOPE,
            "source_slice2_plan_snapshot": {
                "run_instance_id": active_slice2_plan.get("run_instance_id"),
                "plan_version": active_slice2_plan.get("plan_version"),
                "execution_scope": active_slice2_plan.get("execution_scope"),
            },
        },
        spec_reference=POST_V13_SPEC_REFERENCE,
    )

    plan = _build_plan(run_instance_id, active_slice2_plan, ident)
    write_json_atomic(PLAN_ARTIFACT_PATH, plan)
    save_active_plan(plan)

    d1_blocking: list[str] = []
    if not _gate_completed(v12_g6_gate):
        d1_blocking.append("v12_g6_not_completed")
    if not _all_zero_attach(v12_g6_gate):
        d1_blocking.append("v12_attach_gains_not_all_zero")
    if str(v12_g6_gate.get("pretrained_dominant_failure_mode") or "") != "never_reach_attach_distance":
        d1_blocking.append("v12_pretrained_failure_mode_unexpected")
    if str(v12_g6_gate.get("finetuned_dominant_failure_mode") or "") != "never_reach_attach_distance":
        d1_blocking.append("v12_finetuned_failure_mode_unexpected")

    for gate_name, payload in {
        "v13_G0": v13_g0,
        "v13_G1": v13_g1,
        "v13_G2": v13_g2,
        "v13_G3": v13_g3,
        "v13_G4": v13_g4,
        "v13s2_G0": v13s2_g0,
        "v13s2_G2": v13s2_g2,
        "v13s2_G3": v13s2_g3,
        "v13s2_G4": v13s2_g4,
    }.items():
        if not _gate_pass(payload):
            d1_blocking.append(f"{gate_name.lower()}_not_passed")

    if v13_summary.get("stop_point") != "G4":
        d1_blocking.append("v13_stop_point_unexpected")
    if v13s2_summary.get("stop_point") != "G4":
        d1_blocking.append("v13s2_stop_point_unexpected")
    if v13_fit.get("one_step_fit_status") != "one_step_action_fit_ok":
        d1_blocking.append("v13_one_step_fit_not_ok")
    if v13s2_fit.get("one_step_fit_status") != "one_step_action_fit_ok":
        d1_blocking.append("v13s2_one_step_fit_not_ok")

    d1_status = "PASS" if not d1_blocking and d0_status == "PASS" else "STOP_evidence_pack_incomplete"
    d1_gate = write_gate(
        gate_path=d1_path,
        gate_id="D1",
        gate_name="carryforward_evidence_pack",
        run_instance_id=run_instance_id,
        status=d1_status,
        blocking_reasons=d1_blocking,
        allowed_next_phases=["D2"] if d1_status == "PASS" else [],
        extra={
            "v12_carryforward": {
                "run_instance_id": v12_g6_gate.get("run_instance_id"),
                "working_head_commit": v12_g6_gate.get("working_head_commit"),
                "execution_scope": v12_g6_gate.get("execution_scope"),
                "classification": v12_g6_gate.get("classification"),
                "attach_bridge_pass": v12_g6_gate.get("attach_bridge_pass"),
                "ever_attach_eligible_fraction_gain": v12_g6_gate.get("ever_attach_eligible_fraction_gain"),
                "ever_attached_rate_gain": v12_g6_gate.get("ever_attached_rate_gain"),
                "stable_attach_gain": v12_g6_gate.get("stable_attach_gain"),
                "phase_locked_gain": v12_g6_gate.get("phase_locked_gain"),
                "pretrained_dominant_failure_mode": v12_g6_gate.get("pretrained_dominant_failure_mode"),
                "finetuned_dominant_failure_mode": v12_g6_gate.get("finetuned_dominant_failure_mode"),
            },
            "v13_carryforward": {
                "run_instance_id": v13_summary.get("run_instance_id"),
                "stop_point": v13_summary.get("stop_point"),
                "gates": v13_summary.get("gates"),
                "canonical_classification": v13_class.get("canonical_classification"),
                "action_interface_status": v13_action.get("action_interface_status"),
                "orientation_state_consumption_status": v13_state.get("orientation_state_consumption_status"),
                "one_step_fit_status": v13_fit.get("one_step_fit_status"),
            },
            "v13s2_carryforward": {
                "run_instance_id": v13s2_summary.get("run_instance_id"),
                "stop_point": v13s2_summary.get("stop_point"),
                "gates": v13s2_summary.get("gates"),
                "action_interface_status": v13s2_action.get("action_interface_status"),
                "orientation_state_consumption_status": v13s2_state.get("orientation_state_consumption_status"),
                "one_step_fit_status": v13s2_fit.get("one_step_fit_status"),
                "decision_boundary": v13s2_summary.get("decision_boundary"),
            },
            "persistence_hole_open": False,
            "source_artifacts": {
                "v12_probe_summary": str(ARTIFACT_DIR / "g8_train_seed_probe.json"),
                "v12_attach_summary": str(ARTIFACT_DIR / "g8_attach_bridge_summary.json"),
                "v13_summary": str(ARTIFACT_DIR / "v13_g1_g4_tranche_summary.json"),
                "v13s2_summary": str(ARTIFACT_DIR / "v13_slice2_tranche_summary.json"),
            },
        },
        spec_reference=POST_V13_SPEC_REFERENCE,
    )

    d2_blocking: list[str] = []
    hard_checks = {
        "action_dim_mapping_consistent": v13s2_action.get("action_dim_mapping_consistent") is True,
        "live_prev_close_cmd_toggles": v13s2_action.get("live_prev_close_cmd_toggles") is True,
        "action_interface_status_live_close_state_wired": v13s2_action.get("action_interface_status") == "live_close_state_wired",
        "orientation_state_consumption_status_live": v13s2_state.get("orientation_state_consumption_status") == "orientation_bridge_state_live",
        "state_spec_match_expected": v13s2_state.get("state_spec_match_expected") is True,
        "live_state_mode_orientation_bridge_state_v1": "orientation_bridge_state_v1" in (v13s2_state.get("live_state_mode") or []),
        "one_step_fit_status_ok": v13s2_fit.get("one_step_fit_status") == "one_step_action_fit_ok",
        "loss_drop_fraction_gt_0_50": float(v13s2_fit.get("loss_drop_fraction") or 0.0) > 0.50,
        "close_accuracy_gain_gt_0": float(v13s2_fit.get("close_accuracy_gain_over_baseline") or 0.0) > 0.0,
        "orientation_mse_improvement_gt_0": float(v13s2_fit.get("orientation_mse_improvement_fraction") or 0.0) > 0.0,
    }
    for name, passed in hard_checks.items():
        if not passed:
            d2_blocking.append(name)
    signal_band_ambiguity = v13s2_fit.get("fit_retains_prior_signal_band") is False
    if d2_blocking:
        d2_status = "STOP_interface_not_closed"
        d2_allowed = []
    elif signal_band_ambiguity:
        d2_status = "HOLD_signal_band_ambiguity"
        d2_allowed = ["D3"]
    else:
        d2_status = "PASS"
        d2_allowed = ["D3"]
    d2_gate = write_gate(
        gate_path=d2_path,
        gate_id="D2",
        gate_name="interface_closure_contract",
        run_instance_id=run_instance_id,
        status=d2_status,
        blocking_reasons=d2_blocking if d2_blocking else (["signal_band_ambiguity_present"] if signal_band_ambiguity else []),
        allowed_next_phases=d2_allowed,
        extra={
            "hard_checks": hard_checks,
            "signal_band_ambiguity_present": signal_band_ambiguity,
            "slice2_fit": {
                "loss_drop_fraction": v13s2_fit.get("loss_drop_fraction"),
                "close_accuracy_gain_over_baseline": v13s2_fit.get("close_accuracy_gain_over_baseline"),
                "orientation_mse_improvement_fraction": v13s2_fit.get("orientation_mse_improvement_fraction"),
                "fit_retains_prior_signal_band": v13s2_fit.get("fit_retains_prior_signal_band"),
            },
            "prior_v13_fit": {
                "loss_drop_fraction": v13_fit.get("loss_drop_fraction"),
                "close_accuracy_gain_over_baseline": v13_fit.get("close_accuracy_gain_over_baseline"),
                "orientation_mse_improvement_fraction": v13_fit.get("orientation_mse_improvement_fraction"),
            },
        },
        spec_reference=POST_V13_SPEC_REFERENCE,
    )

    d3_blocking: list[str] = []
    canary_design = {
        "max_train_steps": 2000,
        "checkpoint_ladder": [500, 1000, 1500, 2000],
        "early_stop_rule": "Stop if checkpoint 1000 shows zero gain on pre-attach close/attach-eligible indicators relative to pretrained baseline.",
        "probe_scope": "pre-attach, close, attach-eligible only",
        "native_route_only": True,
        "anygrasp_sidecar_allowed": False,
        "publication_scope": "diagnostic_only",
        "claim_bearing": False,
        "future_interpretation_map_prebound": True,
    }
    if d2_status == "STOP_interface_not_closed":
        d3_status = "STOP_canary_design_underbound"
        d3_blocking.append("interface_closure_not_passed")
        ambiguity_resolution = "not_applicable_due_to_interface_stop"
    elif signal_band_ambiguity:
        d3_status = "STOP_signal_band_ambiguity_unresolved"
        d3_blocking.extend(
            [
                "slice2_live_fit_does_not_retain_prior_signal_band",
                "slice2_live_rollout_support_is_only_4_episodes",
                "current_evidence_cannot_separate_benign_live_distribution_shift_from_unresolved_signal_instability",
            ]
        )
        ambiguity_resolution = "unresolved"
    else:
        d3_status = "PASS_canary_design_ready"
        ambiguity_resolution = "none"
    d3_gate = write_gate(
        gate_path=d3_path,
        gate_id="D3",
        gate_name="canary_design_gate",
        run_instance_id=run_instance_id,
        status=d3_status,
        blocking_reasons=d3_blocking,
        allowed_next_phases=["D4"] if d3_status == "PASS_canary_design_ready" else ["D4"],
        extra={
            "proposed_canary_design": canary_design,
            "signal_band_ambiguity_resolution": ambiguity_resolution,
            "design_is_bounded": True,
            "training_not_started": True,
        },
        spec_reference=POST_V13_SPEC_REFERENCE,
    )

    if d0_status != "PASS" or d1_status != "PASS":
        final_verdict = "STOP_PRETRAIN_EVIDENCE_INCOMPLETE"
        final_blocking = d0_blocking + d1_blocking
    elif d2_status == "STOP_interface_not_closed":
        final_verdict = "STOP_PRETRAIN_INTERFACE_UNCLOSED"
        final_blocking = d2_blocking
    elif d3_status != "PASS_canary_design_ready":
        final_verdict = "STOP_PRETRAIN_ADMISSION_AMBIGUOUS"
        final_blocking = d3_blocking
    else:
        final_verdict = "ALLOW_CANARY_TRAIN_PROBE"
        final_blocking = []

    summary = {
        "run_instance_id": run_instance_id,
        "plan_version": PLAN_VERSION,
        "working_head_commit": ident["working_head_commit"],
        "execution_scope": "post_v13_admission_verdict",
        "diagnostic_only": True,
        "claim_bearing": False,
        "gates": {
            "D0": d0_status,
            "D1": d1_status,
            "D2": d2_status,
            "D3": d3_status,
            "D4": "PASS",
        },
        "final_verdict": final_verdict,
        "blocking_reasons": final_blocking,
        "carryforward": {
            "v12_classification": v12_g6_gate.get("classification"),
            "v13_canonical_classification": v13_class.get("canonical_classification"),
            "v13_action_interface_status": v13_action.get("action_interface_status"),
            "v13s2_action_interface_status": v13s2_action.get("action_interface_status"),
            "v13s2_orientation_state_status": v13s2_state.get("orientation_state_consumption_status"),
        },
        "admission_boundary": {
            "training_allowed_now": final_verdict == "ALLOW_CANARY_TRAIN_PROBE",
            "probe_allowed_now": final_verdict == "ALLOW_CANARY_TRAIN_PROBE",
            "future_canary_must_still_be_separate_tranche": True,
            "future_canary_budget": canary_design if final_verdict == "ALLOW_CANARY_TRAIN_PROBE" else None,
        },
        "failure_interpretation_prebound": {
            "zero_signal": "state_action_interface_repaired_no_signal",
            "pre_attach_shift_only": "pre_attach_metric_shift_no_attach_conversion",
            "attach_eligible_gain": "state_action_interface_repaired_attach_signal_detected",
            "close_regression": "action_close_interface_mismatch_confirmed",
            "post_interface_optimization_failure": "objective_weighting_needed_after_interface_repair",
        },
        "timestamp_utc": utc_now(),
    }
    write_json_atomic(SUMMARY_PATH, summary)
    d4_gate = write_gate(
        gate_path=d4_path,
        gate_id="D4",
        gate_name="admission_verdict",
        run_instance_id=run_instance_id,
        status="PASS",
        blocking_reasons=final_blocking,
        allowed_next_phases=["CANARY_TRAIN_PROBE"] if final_verdict == "ALLOW_CANARY_TRAIN_PROBE" else [],
        extra={
            "final_verdict": final_verdict,
            "summary_artifact_path": str(SUMMARY_PATH),
            "training_allowed_now": final_verdict == "ALLOW_CANARY_TRAIN_PROBE",
            "probe_allowed_now": final_verdict == "ALLOW_CANARY_TRAIN_PROBE",
            "signal_band_ambiguity_present": signal_band_ambiguity,
            "source_slice2_run_instance_id": active_slice2_plan.get("run_instance_id"),
        },
        spec_reference=POST_V13_SPEC_REFERENCE,
    )

    print(
        json.dumps(
            {
                "plan": plan,
                "D0": d0_gate,
                "D1": d1_gate,
                "D2": d2_gate,
                "D3": d3_gate,
                "D4": d4_gate,
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

#!/usr/bin/env python3
from __future__ import annotations

import json

from v13_audit_common import (
    ARTIFACT_DIR,
    GATES_V12_DIR,
    GATES_V13_DIR,
    ensure_v13_plan,
    load_json,
    persist_plan_scope,
    write_gate,
    write_json_atomic,
)

OUTPUT_PATH = ARTIFACT_DIR / "v13_v12_classification_sanity_audit.json"
GATE_PATH = GATES_V13_DIR / "G1_classification_sanity.json"


def run() -> int:
    plan = persist_plan_scope(ensure_v13_plan(), "v13_classification_sanity")
    run_instance_id = str(plan["run_instance_id"])
    source_run_id = str(plan.get("source_v12_run_instance_id") or "")
    source_head = str(plan.get("source_v12_working_head_commit") or "")

    probe = load_json(ARTIFACT_DIR / "g8_train_seed_probe.json", {})
    attach = load_json(ARTIFACT_DIR / "g8_attach_bridge_summary.json", {})
    family = load_json(ARTIFACT_DIR / "g8_family_conditioned_probe.json", {})
    g6 = load_json(GATES_V12_DIR / "G6_orientation_first_probe.json", {})
    blocking: list[str] = []
    for name, obj in [("probe", probe), ("attach", attach), ("family", family), ("g6", g6)]:
        if not obj:
            blocking.append(f"missing_{name}_artifact")
            continue
        if obj.get("run_instance_id") != source_run_id:
            blocking.append(f"{name}_run_instance_mismatch")
        if obj.get("working_head_commit") != source_head:
            blocking.append(f"{name}_working_head_mismatch")

    bridge_delta = probe.get("bridge_delta") or {}
    attach_bridge_pass = bool(probe.get("attach_bridge_pass", False) or g6.get("attach_bridge_pass", False))
    dominant_pre = str(probe.get("pretrained_dominant_failure_mode") or g6.get("pretrained_dominant_failure_mode") or "")
    dominant_ft = str(probe.get("finetuned_dominant_failure_mode") or g6.get("finetuned_dominant_failure_mode") or "")
    attach_eligible_delta = float(probe.get("ever_attach_eligible_fraction_gain", 0.0) or 0.0)
    attach_delta = float(probe.get("ever_attached_rate_gain", 0.0) or 0.0)
    phase_locked_delta = float(probe.get("phase_locked_gain", 0.0) or 0.0)
    close_delta = float(bridge_delta.get("close_cmd_rate_mean", 0.0) or 0.0)
    distance_delta = float(bridge_delta.get("distance_pass_rate_mean", 0.0) or 0.0)
    approach_delta = float(bridge_delta.get("approach_gate_pass_rate_mean", 0.0) or 0.0)
    orientation_delta = float(bridge_delta.get("orientation_gate_pass_rate_mean", 0.0) or 0.0)

    if attach_bridge_pass or attach_eligible_delta > 0.0 or attach_delta > 0.0:
        canonical = "attach_transition_signal_detected"
    elif dominant_pre == "never_reach_attach_distance" and dominant_ft == "never_reach_attach_distance" and attach_eligible_delta == 0.0:
        canonical = "pre_attach_metric_shift_no_attach_conversion"
    elif dominant_ft == "orientation_gate_miss" and orientation_delta > 0.0 and attach_eligible_delta == 0.0:
        canonical = "orientation_stage_signal_detected_no_attach"
    else:
        canonical = "evaluator_decomposition_ambiguous"

    artifact = {
        "audit": "v13_v12_classification_sanity_audit",
        "run_instance_id": run_instance_id,
        "plan_version": plan["plan_version"],
        "working_head_commit": plan["working_head_commit"],
        "execution_scope": plan["execution_scope"],
        "source_v12_run_instance_id": source_run_id,
        "source_v12_working_head_commit": source_head,
        "emitted_g6_classification": g6.get("classification"),
        "canonical_classification": canonical,
        "classification_demotion_required": canonical != g6.get("classification"),
        "attach_bridge_pass": attach_bridge_pass,
        "close_cmd_rate_delta": close_delta,
        "distance_pass_rate_delta": distance_delta,
        "approach_gate_rate_delta": approach_delta,
        "orientation_gate_rate_delta": orientation_delta,
        "attach_eligible_delta": attach_eligible_delta,
        "attach_delta": attach_delta,
        "phase_locked_delta": phase_locked_delta,
        "pretrained_dominant_failure_mode": dominant_pre,
        "finetuned_dominant_failure_mode": dominant_ft,
        "source_artifacts": {
            "g6_gate": str(GATES_V12_DIR / "G6_orientation_first_probe.json"),
            "probe_summary": str(ARTIFACT_DIR / "g8_train_seed_probe.json"),
            "attach_summary": str(ARTIFACT_DIR / "g8_attach_bridge_summary.json"),
            "family_summary": str(ARTIFACT_DIR / "g8_family_conditioned_probe.json"),
        },
    }
    write_json_atomic(OUTPUT_PATH, artifact)
    gate = write_gate(
        gate_path=GATE_PATH,
        gate_id="G1",
        gate_name="classification_sanity",
        run_instance_id=run_instance_id,
        status="PASS" if not blocking else "STOP",
        blocking_reasons=blocking,
        allowed_next_phases=["G2", "G3", "G4"] if not blocking else [],
        extra={
            "audit_artifact_path": str(OUTPUT_PATH),
            "emitted_g6_classification": g6.get("classification"),
            "canonical_classification": canonical,
            "attach_bridge_pass": attach_bridge_pass,
            "pretrained_dominant_failure_mode": dominant_pre,
            "finetuned_dominant_failure_mode": dominant_ft,
        },
    )
    print(json.dumps({"artifact": artifact, "gate": gate}, indent=2))
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(run())

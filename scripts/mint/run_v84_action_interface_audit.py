#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

from v13_audit_common import (
    ARTIFACT_DIR,
    GATES_V13_DIR,
    PROJECT_ROOT,
    ensure_v13_plan,
    load_json,
    persist_plan_scope,
    write_gate,
    write_json_atomic,
)

OUTPUT_PATH = ARTIFACT_DIR / "v13_action_interface_audit.json"
GATE_PATH = GATES_V13_DIR / "G2_action_interface_audit.json"
ROLLOUT_DIR = ARTIFACT_DIR / "g6_orientation_support_train_rollouts"
CHECKPOINT_SWEEP_PATH = ARTIFACT_DIR / "checkpoint_probe_sweep.json"
RECORD_TEMPLATE = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1" / "evaluation" / "tiny_retrain" / "V1cT2S0" / "s0" / "diagnostic"
ENV_SOURCE = PROJECT_ROOT / "scripts" / "mint" / "drawer_robot_env_mujoco.py"


def _teacher_window_stats() -> dict[str, float]:
    close_rates = []
    orient_norms = []
    action_dims = set()
    labels_available = False
    for meta_path in sorted(ROLLOUT_DIR.glob("*.json")):
        meta = load_json(meta_path, {})
        teacher_class = str(meta.get("orientation_support_teacher_class") or "")
        if teacher_class not in {"orientation_transition_teacher", "attach_eligible_transition_teacher", "orientation_context_teacher"}:
            continue
        npz_path = meta_path.with_suffix(".npz")
        data = np.load(npz_path, allow_pickle=True)
        actions = np.asarray(data["actions"], dtype=np.float32)
        action_dims.add(actions.shape[1])
        close_rates.append(float(np.mean(actions[:, 6] < 0.0)))
        orient_norms.append(float(np.mean(np.linalg.norm(actions[:, 3:6], axis=1))))
        if np.any(actions[:, 6] < 0.0):
            labels_available = True
    return {
        "teacher_close_rate_orientation_windows": float(statistics.mean(close_rates)) if close_rates else 0.0,
        "teacher_orientation_action_norm_mean": float(statistics.mean(orient_norms)) if orient_norms else 0.0,
        "teacher_action_dim_unique": sorted(action_dims),
        "close_command_label_available": labels_available,
    }


def _selected_records_path() -> Path:
    sweep = load_json(CHECKPOINT_SWEEP_PATH, {})
    step = int(sweep.get("selected_checkpoint_step") or 10000)
    return RECORD_TEMPLATE / f"authoritative_train_probe_{step:06d}_records.json"


def _probe_stats(records_path: Path) -> dict[str, float]:
    payload = load_json(records_path, {})
    out: dict[str, float] = {}
    for name in ["pretrained_mint", "finetuned_mint"]:
        recs = payload.get(name) or []
        if not recs:
            continue
        prefix = "pretrained" if name.startswith("pretrained") else "finetuned"
        out[f"{prefix}_close_rate"] = float(statistics.mean(float(r.get("close_cmd_rate", 0.0)) for r in recs))
        out[f"{prefix}_distance_pass_rate"] = float(statistics.mean(float(r.get("distance_pass_rate", 0.0)) for r in recs))
        out[f"{prefix}_approach_gate_pass_rate"] = float(statistics.mean(float(r.get("approach_gate_pass_rate", 0.0)) for r in recs))
        out[f"{prefix}_orientation_gate_pass_rate"] = float(statistics.mean(float(r.get("orientation_gate_pass_rate", 0.0)) for r in recs))
        out[f"{prefix}_attach_eligible_rate"] = float(statistics.mean(float(r.get("attach_eligible_rate", 0.0)) for r in recs))
    return out


def _action_mapping() -> dict[str, object]:
    text = ENV_SOURCE.read_text()
    mapping_consistent = "action[6]" in text and "float(action[6]) < 0.0" in text and "delta_rot = np.clip(action[3:6]" in text
    return {
        "action_dim": 7,
        "gripper_close_dim_index": 6,
        "orientation_action_dim_indices": [3, 4, 5],
        "gripper_close_sign": "negative_means_close",
        "gripper_close_threshold": 0.0,
        "action_dim_mapping_consistent": mapping_consistent,
    }


def run() -> int:
    plan = persist_plan_scope(ensure_v13_plan(), "v13_action_interface_audit")
    run_instance_id = str(plan["run_instance_id"])
    teacher = _teacher_window_stats()
    mapping = _action_mapping()
    records_path = _selected_records_path()
    probe = _probe_stats(records_path)

    blocking: list[str] = []
    if not mapping["action_dim_mapping_consistent"]:
        blocking.append("action_dim_mapping_mismatch")
    if not records_path.exists():
        blocking.append("cannot_determine_action_interface")

    pretrained_close = float(probe.get("pretrained_close_rate", 0.0))
    finetuned_close = float(probe.get("finetuned_close_rate", 0.0))
    close_delta = finetuned_close - pretrained_close
    orientation_delta = float(probe.get("finetuned_orientation_gate_pass_rate", 0.0)) - float(probe.get("pretrained_orientation_gate_pass_rate", 0.0))
    close_command_learned = finetuned_close >= max(0.80 * teacher["teacher_close_rate_orientation_windows"], pretrained_close - 0.05)
    orientation_action_not_learned = orientation_delta <= 0.0

    if blocking:
        status_label = blocking[0]
    elif not teacher["close_command_label_available"]:
        status_label = "close_command_label_sparse"
    elif not close_command_learned:
        status_label = "close_command_not_learned"
    elif orientation_action_not_learned:
        status_label = "orientation_action_not_learned"
    else:
        status_label = "action_interface_consistent"

    artifact = {
        "audit": "v13_action_interface_audit",
        "run_instance_id": run_instance_id,
        "plan_version": plan["plan_version"],
        "working_head_commit": plan["working_head_commit"],
        "execution_scope": plan["execution_scope"],
        **mapping,
        **teacher,
        **probe,
        "records_path": str(records_path),
        "finetuned_close_rate_orientation_probe": finetuned_close,
        "close_rate_delta": close_delta,
        "orientation_gate_rate_delta": orientation_delta,
        "teacher_orientation_action_norm_mean": teacher["teacher_orientation_action_norm_mean"],
        "finetuned_orientation_action_norm_mean": None,
        "close_command_learned": close_command_learned,
        "orientation_action_not_learned": orientation_action_not_learned,
        "action_interface_status": status_label,
    }
    write_json_atomic(OUTPUT_PATH, artifact)
    gate = write_gate(
        gate_path=GATE_PATH,
        gate_id="G2",
        gate_name="action_interface_audit",
        run_instance_id=run_instance_id,
        status="PASS" if not blocking else "STOP",
        blocking_reasons=blocking,
        allowed_next_phases=["G3", "G4"] if not blocking else [],
        extra={
            "audit_artifact_path": str(OUTPUT_PATH),
            "action_interface_status": status_label,
            "action_dim_mapping_consistent": mapping["action_dim_mapping_consistent"],
            "close_command_label_available": teacher["close_command_label_available"],
            "close_command_learned": close_command_learned,
            "close_rate_delta": close_delta,
            "orientation_action_not_learned": orientation_action_not_learned,
        },
    )
    print(json.dumps({"artifact": artifact, "gate": gate}, indent=2))
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(run())

#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

import numpy as np
import torch

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

OUTPUT_PATH = ARTIFACT_DIR / "v13_orientation_state_consumption_audit.json"
GATE_PATH = GATES_V13_DIR / "G3_orientation_state_consumption.json"
ROLLOUT_DIR = ARTIFACT_DIR / "g6_orientation_support_train_rollouts"
PROCESSOR_PATH = PROJECT_ROOT / "external" / "MINT" / "lerobot_policy_mint" / "src"

sys.path.insert(0, str(PROCESSOR_PATH))

from lerobot.utils.constants import OBS_STATE  # noqa: E402
from lerobot.processor.core import TransitionKey  # noqa: E402
from lerobot_policy_mint.processor_mint import MINTPrepareStateTokenizerProcessorStep  # noqa: E402


def _sample_rollout() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    meta_path = sorted(ROLLOUT_DIR.glob("*.json"))[0]
    meta = load_json(meta_path, {})
    data = np.load(meta_path.with_suffix(".npz"), allow_pickle=True)
    return (
        meta,
        np.asarray(data["states"], dtype=np.float32),
        np.asarray(data["actions"], dtype=np.float32),
        np.asarray(data["handle_distance_trace"], dtype=np.float32),
        np.asarray(data["orientation_alignment_trace"], dtype=np.float32),
        np.asarray(data["orientation_error_trace"], dtype=np.float32),
    )


def _approach_alignment(states: np.ndarray) -> np.ndarray:
    eef = states[:, :3]
    handle_rel = states[:, 3:6] * np.array([0.22, 0.18, 0.14], dtype=np.float32)
    handle_world = eef + handle_rel
    out = np.zeros((len(states),), dtype=np.float32)
    for i in range(1, len(states)):
        motion = eef[i] - eef[i - 1]
        desired = handle_world[i - 1] - eef[i - 1]
        m = float(np.linalg.norm(motion))
        d = float(np.linalg.norm(desired))
        if m > 1e-8 and d > 1e-8:
            out[i] = float(np.clip(np.dot(motion / m, desired / d), -1.0, 1.0))
    return out


def _diagnostic_state(states: np.ndarray, actions: np.ndarray, handle_distance: np.ndarray, orientation_alignment: np.ndarray, orientation_error: np.ndarray) -> np.ndarray:
    approach = _approach_alignment(states)
    prev_close = np.concatenate([[0.0], (actions[:-1, 6] < 0.0).astype(np.float32)])
    gripper = states[:, 7]
    distance_norm = np.clip(handle_distance / 0.35, 0.0, 1.0)
    attach_proxy = ((handle_distance <= 0.08) & (orientation_alignment >= 0.60) & (prev_close > 0.5)).astype(np.float32)
    diag = np.stack(
        [
            distance_norm,
            approach,
            orientation_alignment,
            np.sin(orientation_error).astype(np.float32),
            np.cos(orientation_error).astype(np.float32),
            prev_close,
            gripper,
            attach_proxy,
        ],
        axis=1,
    ).astype(np.float32)
    return diag


def _prompt_delta(task: str, vector_a: np.ndarray, vector_b: np.ndarray) -> dict[str, object]:
    step = MINTPrepareStateTokenizerProcessorStep(max_state_dim=32)
    base = torch.tensor(vector_a[None, :], dtype=torch.float32)
    alt = torch.tensor(vector_b[None, :], dtype=torch.float32)
    out_a = step({TransitionKey.OBSERVATION: {OBS_STATE: base}, TransitionKey.COMPLEMENTARY_DATA: {"task": [task]}})
    out_b = step({TransitionKey.OBSERVATION: {OBS_STATE: alt}, TransitionKey.COMPLEMENTARY_DATA: {"task": [task]}})
    prompt_a = out_a[TransitionKey.COMPLEMENTARY_DATA]["task"][0]
    prompt_b = out_b[TransitionKey.COMPLEMENTARY_DATA]["task"][0]
    return {
        "prompt_changed": prompt_a != prompt_b,
        "prompt_a_prefix": prompt_a[:160],
        "prompt_b_prefix": prompt_b[:160],
    }


def run() -> int:
    plan = persist_plan_scope(ensure_v13_plan(), "v13_orientation_state_consumption")
    run_instance_id = str(plan["run_instance_id"])
    prior = load_json(ARTIFACT_DIR / "v12_state_conditioning_audit.json", {})
    meta, states, actions, handle_distance, orientation_alignment, orientation_error = _sample_rollout()
    state_spec = meta.get("state_spec") or {}
    current_dim_names = list(state_spec.get("dim_names") or [])
    diagnostic_dim_names = [
        "distance_to_handle_norm",
        "approach_alignment_cos",
        "orientation_alignment_cos",
        "orientation_error_sin",
        "orientation_error_cos",
        "prev_close_cmd",
        "gripper_joint",
        "attach_eligible_proxy",
    ]
    diag_state = _diagnostic_state(states, actions, handle_distance, orientation_alignment, orientation_error)
    task = str(meta.get("task") or meta.get("language_instruction") or "open drawer")
    prompt_delta = _prompt_delta(task, diag_state[0], diag_state[min(len(diag_state) - 1, 1)])

    feature_audit = {
        "orientation_alignment_cos": {
            "in_rollout_metadata": False,
            "in_dataset_item": True,
            "in_model_input": "orientation_alignment_cos" in current_dim_names,
        },
        "approach_alignment_cos": {
            "in_rollout_metadata": False,
            "in_dataset_item": True,
            "in_model_input": "approach_alignment_cos" in current_dim_names,
        },
        "eef_orientation": {
            "in_rollout_metadata": False,
            "in_dataset_item": False,
            "in_model_input": any(name.startswith("eef_quat") or name.startswith("eef_rotvec") for name in current_dim_names),
        },
        "handle_frame_orientation_error": {
            "in_rollout_metadata": False,
            "in_dataset_item": True,
            "in_model_input": any("orientation_error" in name for name in current_dim_names),
        },
        "relative_handle_pose": {
            "in_rollout_metadata": False,
            "in_dataset_item": all(x in current_dim_names for x in ["handle_rel_x_norm", "handle_rel_y_norm", "handle_rel_z_norm"]),
            "in_model_input": all(x in current_dim_names for x in ["handle_rel_x_norm", "handle_rel_y_norm", "handle_rel_z_norm"]),
        },
        "attach_eligible_proxy": {
            "in_rollout_metadata": False,
            "in_dataset_item": True,
            "in_model_input": "attach_eligible_proxy" in current_dim_names,
        },
        "close_command_or_gripper_state": {
            "in_rollout_metadata": False,
            "in_dataset_item": True,
            "in_model_input": "gripper_joint" in current_dim_names,
        },
    }
    state_input_consumed = bool(prior.get("state_input_consumed_by_policy", False))
    state_input_present = bool(prior.get("state_input_present", True))
    orientation_features_consumed_now = any(v["in_model_input"] for k, v in feature_audit.items() if k in {"orientation_alignment_cos", "approach_alignment_cos", "handle_frame_orientation_error", "attach_eligible_proxy"})
    repairable = state_input_present and state_input_consumed and not orientation_features_consumed_now and bool(prompt_delta["prompt_changed"])

    if not state_input_present or not state_input_consumed:
        status_label = "state_input_not_consumed"
    elif orientation_features_consumed_now:
        status_label = "orientation_state_consumed"
    elif repairable:
        status_label = "orientation_state_missing_but_repairable"
    else:
        status_label = "state_repair_failed"

    artifact = {
        "audit": "v13_orientation_state_consumption_repair_audit",
        "run_instance_id": run_instance_id,
        "plan_version": plan["plan_version"],
        "working_head_commit": plan["working_head_commit"],
        "execution_scope": plan["execution_scope"],
        "state_input_present": state_input_present,
        "state_input_consumed_by_policy": state_input_consumed,
        "current_state_mode": state_spec.get("state_mode"),
        "current_state_dim_names": current_dim_names,
        "diagnostic_state_mode": "orientation_bridge_state_v1",
        "diagnostic_state_dim_names": diagnostic_dim_names,
        "feature_audit": feature_audit,
        "diagnostic_state_vector_preview": diag_state[0].round(6).tolist(),
        "prompt_perturbation_preview": prompt_delta,
        "orientation_state_consumption_status": status_label,
    }
    write_json_atomic(OUTPUT_PATH, artifact)
    blocking = ["state_repair_failed"] if status_label == "state_repair_failed" else []
    gate = write_gate(
        gate_path=GATE_PATH,
        gate_id="G3",
        gate_name="orientation_state_consumption",
        run_instance_id=run_instance_id,
        status="PASS" if not blocking else "STOP",
        blocking_reasons=blocking,
        allowed_next_phases=["G4"] if not blocking else [],
        extra={
            "audit_artifact_path": str(OUTPUT_PATH),
            "orientation_state_consumption_status": status_label,
            "diagnostic_state_mode": "orientation_bridge_state_v1",
            "prompt_perturbation_changes_model_input": bool(prompt_delta["prompt_changed"]),
        },
    )
    print(json.dumps({"artifact": artifact, "gate": gate}, indent=2))
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(run())

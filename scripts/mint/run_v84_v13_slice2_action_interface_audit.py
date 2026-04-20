#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from drawer_robot_env_mujoco import DrawerEnvContractConfig, DrawerRobotEnvMuJoCo
from root_cause_controller import RootCauseController
from v13_audit_common import (
    ARTIFACT_DIR,
    GATES_V13_SLICE2_DIR,
    ensure_v13_slice2_plan,
    persist_plan_scope,
    write_gate,
    write_json_atomic,
)

OUTPUT_PATH = ARTIFACT_DIR / "v13_slice2_action_interface_audit.json"
GATE_PATH = GATES_V13_SLICE2_DIR / "G2_action_interface_audit.json"
ENV_SOURCE = Path(__file__).resolve().parent / "drawer_robot_env_mujoco.py"


def _live_contract() -> DrawerEnvContractConfig:
    controller = RootCauseController()
    payload = dict(
        controller._frozen_matrix_contracts_v5_pro()["V1cT2S3"]["env_contract_config"]
    )
    return DrawerEnvContractConfig(**payload)


def _action_mapping() -> dict[str, object]:
    text = ENV_SOURCE.read_text()
    mapping_consistent = (
        "float(action[6]) < 0.0" in text and "delta_rot = np.clip(action[3:6]" in text
    )
    return {
        "action_dim": 7,
        "gripper_close_dim_index": 6,
        "orientation_action_dim_indices": [3, 4, 5],
        "gripper_close_sign": "negative_means_close",
        "gripper_close_threshold": 0.0,
        "action_dim_mapping_consistent": mapping_consistent,
    }


def _prev_close_cmd_smoke(contract: DrawerEnvContractConfig) -> dict[str, object]:
    env = DrawerRobotEnvMuJoCo(seed=2, image_size=64, max_steps=4, contract=contract)
    try:
        obs0 = env.reset()
        names = list((obs0.state_spec or {}).get("dim_names") or [])
        idx = names.index("prev_close_cmd")
        a_close = np.zeros(7, dtype=np.float32)
        a_close[6] = -1.0
        obs1, _, _, _ = env.step(a_close)
        a_open = np.zeros(7, dtype=np.float32)
        a_open[6] = 1.0
        obs2, _, _, _ = env.step(a_open)
        return {
            "state_mode": (obs0.state_spec or {}).get("state_mode"),
            "dim_names": names,
            "prev_close_cmd_index": idx,
            "reset_prev_close_cmd": float(obs0.state[idx]),
            "after_close_prev_close_cmd": float(obs1.state[idx]),
            "after_open_prev_close_cmd": float(obs2.state[idx]),
            "live_prev_close_cmd_toggles": float(obs0.state[idx]) == 0.0 and float(obs1.state[idx]) == 1.0 and float(obs2.state[idx]) == 0.0,
        }
    finally:
        env.close()


def run() -> int:
    plan = persist_plan_scope(ensure_v13_slice2_plan(), "v13_slice_2_action_interface_audit")
    mapping = _action_mapping()
    smoke = _prev_close_cmd_smoke(_live_contract())

    blocking: list[str] = []
    if not mapping["action_dim_mapping_consistent"]:
        blocking.append("action_dim_mapping_mismatch")
    if smoke.get("state_mode") != "orientation_bridge_state_v1":
        blocking.append("live_state_mode_not_rewired")
    if not smoke.get("live_prev_close_cmd_toggles", False):
        blocking.append("prev_close_cmd_not_live")

    artifact = {
        "audit": "v13_slice2_action_interface_audit",
        "run_instance_id": plan["run_instance_id"],
        "plan_version": plan["plan_version"],
        "working_head_commit": plan["working_head_commit"],
        "execution_scope": plan["execution_scope"],
        **mapping,
        **smoke,
        "action_interface_status": "live_close_state_wired" if not blocking else blocking[0],
        "training_allowed_after_g2": False,
    }
    write_json_atomic(OUTPUT_PATH, artifact)
    gate = write_gate(
        gate_path=GATE_PATH,
        gate_id="G2",
        gate_name="action_interface_audit",
        run_instance_id=plan["run_instance_id"],
        status="PASS" if not blocking else "STOP",
        blocking_reasons=blocking,
        allowed_next_phases=["G3", "G4"] if not blocking else [],
        extra={
            "audit_artifact_path": str(OUTPUT_PATH),
            "action_interface_status": artifact["action_interface_status"],
            "action_dim_mapping_consistent": mapping["action_dim_mapping_consistent"],
            "live_prev_close_cmd_toggles": bool(smoke["live_prev_close_cmd_toggles"]),
            "training_allowed_after_g2": False,
        },
    )
    print(json.dumps({"artifact": artifact, "gate": gate}, indent=2))
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(run())


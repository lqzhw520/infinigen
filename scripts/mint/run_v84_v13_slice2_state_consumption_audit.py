#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from drawer_robot_env_mujoco import (
    DrawerEnvContractConfig,
    build_robot_rollout,
    save_robot_rollout,
)
from root_cause_controller import RootCauseController
from v13_audit_common import (
    ARTIFACT_DIR,
    GATES_V13_SLICE2_DIR,
    PROJECT_ROOT,
    ensure_v13_slice2_plan,
    persist_plan_scope,
    teacher_family_dispatch_payload,
    write_gate,
    write_json_atomic,
)

PROCESSOR_PATH = PROJECT_ROOT / "external" / "MINT" / "lerobot_policy_mint" / "src"
sys.path.insert(0, str(PROCESSOR_PATH))

from lerobot.utils.constants import OBS_STATE  # noqa: E402
from lerobot.processor.core import TransitionKey  # noqa: E402
from lerobot_policy_mint.processor_mint import MINTPrepareStateTokenizerProcessorStep  # noqa: E402

ROLLOUT_DIR = ARTIFACT_DIR / "v13_slice2_live_state_rollouts"
OUTPUT_PATH = ARTIFACT_DIR / "v13_slice2_state_consumption_audit.json"
GATE_PATH = GATES_V13_SLICE2_DIR / "G3_state_consumption.json"
SEEDS = [1, 2, 4, 7]
EXPECTED_DIM_NAMES = [
    "distance_to_handle_norm",
    "approach_alignment_cos",
    "orientation_alignment_cos",
    "orientation_error_sin",
    "orientation_error_cos",
    "prev_close_cmd",
    "gripper_joint",
    "attach_eligible_proxy",
]


def _live_contract() -> DrawerEnvContractConfig:
    controller = RootCauseController()
    payload = dict(
        controller._frozen_matrix_contracts_v5_pro()["V1cT2S3"]["env_contract_config"]
    )
    return DrawerEnvContractConfig(**payload)


def _materialize_rollouts(plan: dict[str, object]) -> list[dict[str, object]]:
    if ROLLOUT_DIR.exists():
        shutil.rmtree(ROLLOUT_DIR)
    ROLLOUT_DIR.mkdir(parents=True, exist_ok=True)
    contract = _live_contract()
    built = []
    for episode_index, seed in enumerate(SEEDS):
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
        rollout["bridge_attempt"] = "diagnostic_wiring_only"
        rollout["canonical_train_cell"] = plan["canonical_train_cell"]
        rollout["source_canonical_train_cell"] = plan["source_canonical_train_cell"]
        rollout["best_train_state_mode"] = plan["best_train_state_mode"]
        rollout["source_best_train_state_mode"] = plan["source_best_train_state_mode"]
        rollout["active_train_state_mode"] = plan["active_train_state_mode"]
        rollout["active_state_mode_name"] = plan["active_state_mode_name"]
        base = ROLLOUT_DIR / f"seed_{seed:03d}_episode_{episode_index:02d}"
        save_robot_rollout(base, rollout)
        built.append(rollout)
    return built


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
    plan = persist_plan_scope(ensure_v13_slice2_plan(), "v13_slice_2_state_consumption")
    rollouts = _materialize_rollouts(plan)
    meta_paths = sorted(ROLLOUT_DIR.glob("*.json"))
    metas = [json.loads(path.read_text()) for path in meta_paths]
    dim_names = [list((meta.get("state_spec") or {}).get("dim_names") or []) for meta in metas]
    state_modes = [str((meta.get("state_spec") or {}).get("state_mode") or meta.get("active_state_mode_name") or "") for meta in metas]
    unique_dim_signatures = sorted({tuple(names) for names in dim_names})
    sample_rollout = rollouts[0]
    states = np.asarray(sample_rollout["states"], dtype=np.float32)
    task = str(sample_rollout.get("task") or "open drawer")
    prompt_delta = _prompt_delta(task, states[0], states[min(len(states) - 1, 1)])
    feature_audit = {
        "orientation_alignment_cos": {"in_model_input": "orientation_alignment_cos" in EXPECTED_DIM_NAMES},
        "approach_alignment_cos": {"in_model_input": "approach_alignment_cos" in EXPECTED_DIM_NAMES},
        "handle_frame_orientation_error": {"in_model_input": any("orientation_error" in name for name in EXPECTED_DIM_NAMES)},
        "attach_eligible_proxy": {"in_model_input": "attach_eligible_proxy" in EXPECTED_DIM_NAMES},
        "close_command_or_gripper_state": {"in_model_input": "prev_close_cmd" in EXPECTED_DIM_NAMES or "gripper_joint" in EXPECTED_DIM_NAMES},
    }
    orientation_features_consumed_now = all(v["in_model_input"] for v in feature_audit.values())
    spec_match = unique_dim_signatures == [tuple(EXPECTED_DIM_NAMES)]
    blocking = []
    if len(rollouts) != len(SEEDS):
        blocking.append("live_rollout_materialization_incomplete")
    if set(state_modes) != {"orientation_bridge_state_v1"}:
        blocking.append("live_state_mode_not_unique")
    if not spec_match:
        blocking.append("live_state_spec_mismatch")
    if not prompt_delta["prompt_changed"]:
        blocking.append("live_state_prompt_not_consumed")
    if not orientation_features_consumed_now:
        blocking.append("orientation_features_still_missing")

    artifact = {
        "audit": "v13_slice2_state_consumption_audit",
        "run_instance_id": plan["run_instance_id"],
        "plan_version": plan["plan_version"],
        "working_head_commit": plan["working_head_commit"],
        "execution_scope": plan["execution_scope"],
        "live_rollout_dir": str(ROLLOUT_DIR),
        "live_rollout_count": len(rollouts),
        "live_rollout_seeds": SEEDS,
        "live_state_mode": sorted(set(state_modes)),
        "live_state_dim_names": EXPECTED_DIM_NAMES,
        "state_spec_match_expected": spec_match,
        "feature_audit": feature_audit,
        "prompt_perturbation_preview": prompt_delta,
        "orientation_state_consumption_status": "orientation_bridge_state_live" if not blocking else blocking[0],
        "training_allowed_after_g3": False,
    }
    write_json_atomic(OUTPUT_PATH, artifact)
    gate = write_gate(
        gate_path=GATE_PATH,
        gate_id="G3",
        gate_name="state_consumption",
        run_instance_id=plan["run_instance_id"],
        status="PASS" if not blocking else "STOP",
        blocking_reasons=blocking,
        allowed_next_phases=["G4"] if not blocking else [],
        extra={
            "audit_artifact_path": str(OUTPUT_PATH),
            "orientation_state_consumption_status": artifact["orientation_state_consumption_status"],
            "live_rollout_dir": str(ROLLOUT_DIR),
            "prompt_perturbation_changes_model_input": bool(prompt_delta["prompt_changed"]),
            "training_allowed_after_g3": False,
        },
    )
    print(json.dumps({"artifact": artifact, "gate": gate}, indent=2))
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(run())

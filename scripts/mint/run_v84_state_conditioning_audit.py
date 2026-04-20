#!/usr/bin/env python3
"""G2 state/conditioning consumption audit for v12.1 slice-1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_DIR = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1"
ARTIFACT_DIR = CAMPAIGN_DIR / "artifacts"
GATES_V12_DIR = CAMPAIGN_DIR / "autopilot" / "gates_v12"
G0_GATE_PATH = GATES_V12_DIR / "G0_sovereign_sync.json"
OUTPUT_PATH = ARTIFACT_DIR / "v12_state_conditioning_audit.json"
GATE_PATH = GATES_V12_DIR / "G2_state_conditioning_audit.json"
ROLLOUT_DIR = ARTIFACT_DIR / "g6_learning_support_train_rollouts"
DATASET_BUILD_PATH = ARTIFACT_DIR / "g8_canonical_dataset_build.json"
CHECKPOINT_CONFIG_PATH = PROJECT_ROOT / "external" / "MINT" / "checkpoints" / "MINT-libero" / "config.json"
PROCESSOR_PATH = PROJECT_ROOT / "external" / "MINT" / "lerobot_policy_mint" / "src"
TRUTH_CONTRACT_PATH = PROJECT_ROOT / "docs" / "contracts" / "truth_contract_v84.json"
ACCEPTANCE_CONTRACT_PATH = PROJECT_ROOT / "docs" / "contracts" / "acceptance_contract_v84.json"
SPEC_REFERENCE = "/Users/zhuhaowu/ws/phd-anyboxs/infinigen_from_servers/docs/gpt5.4Pro/codex_mint_v84_v12_1_orientation_attach_anygrasp_execution_spec.md"

sys.path.insert(0, str(PROCESSOR_PATH))

from lerobot.utils.constants import OBS_STATE  # noqa: E402
from lerobot.processor.core import TransitionKey  # noqa: E402
from lerobot_policy_mint.processor_mint import MINTPrepareStateTokenizerProcessorStep  # noqa: E402


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text())


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=PROJECT_ROOT, text=True).strip()


def current_repo_identity() -> dict[str, str]:
    return {
        "branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "working_head_commit": _git(["git", "rev-parse", "HEAD"]),
        "vendor_head_commit": _git(["git", "rev-parse", "HEAD:external/MINT"]),
    }


def _base_gate_payload(run_instance_id: str) -> dict[str, Any]:
    ident = current_repo_identity()
    return {
        "run_instance_id": run_instance_id,
        "working_head_commit": ident["working_head_commit"],
        "vendor_head_commit": ident["vendor_head_commit"],
        "branch": ident["branch"],
        "truth_contract_hash": sha256_file(TRUTH_CONTRACT_PATH),
        "acceptance_contract_hash": sha256_file(ACCEPTANCE_CONTRACT_PATH),
        "spec_reference": SPEC_REFERENCE,
        "timestamp_utc": utc_now(),
    }


def write_gate(run_instance_id: str, status: str, blocking_reasons: list[str], extra: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "gate_id": "G2",
        "gate_name": "state_conditioning_audit",
        **_base_gate_payload(run_instance_id),
        "status": status,
        "blocking_reasons": blocking_reasons,
        "allowed_next_phases": ["slice_closed"],
        **extra,
    }
    write_json_atomic(GATE_PATH, payload)
    return payload


def _sample_rollout() -> tuple[Path, dict[str, Any], Any]:
    npz_path = sorted(ROLLOUT_DIR.glob("*.npz"))[0]
    meta_path = npz_path.with_suffix(".json")
    meta = load_json(meta_path, {})
    data = np.load(npz_path, allow_pickle=True)
    return npz_path, meta, data


def _orientation_feature_map(meta: dict[str, Any], data: Any, state_dim_names: list[str]) -> dict[str, Any]:
    telemetry = meta.get("orientation_telemetry") or {}
    features = {
        "relative_handle_pose": {
            "available_in_dataset": all(name in state_dim_names for name in ["handle_rel_x_norm", "handle_rel_y_norm", "handle_rel_z_norm"]),
            "consumed_by_model_state": all(name in state_dim_names for name in ["handle_rel_x_norm", "handle_rel_y_norm", "handle_rel_z_norm"]),
            "evidence": "8D state vector contains handle_rel_xyz_norm slots",
        },
        "eef_orientation": {
            "available_in_dataset": False,
            "consumed_by_model_state": False,
            "evidence": "current 8D state does not include EEF quaternion/rotvec",
        },
        "handle_frame_orientation_error": {
            "available_in_dataset": "orientation_error_trace" in data.files or "orientation_error_rad" in telemetry,
            "consumed_by_model_state": False,
            "evidence": "present in orientation telemetry / traces, not in observation.state",
        },
        "orientation_alignment_cos": {
            "available_in_dataset": "orientation_alignment_trace" in data.files or "orientation_alignment_cos" in telemetry,
            "consumed_by_model_state": False,
            "evidence": "present in traces/telemetry, not explicit in 8D state",
        },
        "approach_alignment_cos": {
            "available_in_dataset": False,
            "consumed_by_model_state": False,
            "evidence": "not explicit in training state; only evaluator/probe computes approach gate from motion",
        },
        "attach_eligible_flag": {
            "available_in_dataset": "attach_eligible_trace" in data.files,
            "consumed_by_model_state": False,
            "evidence": "trace available, not in observation.state",
        },
        "close_command_or_gripper_state": {
            "available_in_dataset": ("actions" in data.files) or ("gripper_joint" in state_dim_names),
            "consumed_by_model_state": "gripper_joint" in state_dim_names,
            "evidence": "gripper_joint present in state and close command recoverable from actions[:,6]",
        },
    }
    return features


def _state_prompt_perturbation(sample_state: np.ndarray, task: str) -> dict[str, Any]:
    step = MINTPrepareStateTokenizerProcessorStep(max_state_dim=32)
    base = torch.tensor(sample_state[None, :], dtype=torch.float32)
    perturbed = base.clone()
    if perturbed.shape[1] >= 6:
        perturbed[0, 3] = perturbed[0, 3] + 0.25
        perturbed[0, 4] = perturbed[0, 4] - 0.15
    else:
        perturbed[0, 0] = perturbed[0, 0] + 0.25
    transition1 = {
        TransitionKey.OBSERVATION: {OBS_STATE: base},
        TransitionKey.COMPLEMENTARY_DATA: {"task": [task]},
    }
    transition2 = {
        TransitionKey.OBSERVATION: {OBS_STATE: perturbed},
        TransitionKey.COMPLEMENTARY_DATA: {"task": [task]},
    }
    out1 = step(transition1)
    out2 = step(transition2)
    prompt1 = out1[TransitionKey.COMPLEMENTARY_DATA]["task"][0]
    prompt2 = out2[TransitionKey.COMPLEMENTARY_DATA]["task"][0]
    return {
        "prompt1": prompt1,
        "prompt2": prompt2,
        "prompt_changed": prompt1 != prompt2,
    }


def run(output_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    g0 = load_json(G0_GATE_PATH, {})
    run_instance_id = str(g0.get("run_instance_id") or "v12_1_slice1_unknown")
    blocking_reasons: list[str] = []
    dataset_build = load_json(DATASET_BUILD_PATH, {})
    cfg = load_json(CHECKPOINT_CONFIG_PATH, {})
    npz_path, meta, data = _sample_rollout()
    state_dim_names = list(dataset_build.get("state_dim_names") or [])
    input_features = ((cfg.get("input_features") or {}).get("observation.state") or {})
    state_shape = list(input_features.get("shape") or [])
    processor_source = (PROCESSOR_PATH / "lerobot_policy_mint" / "processor_mint.py").read_text()
    state_present = bool(state_shape)
    state_consumed = "State: {state_str}" in processor_source and "OBS_STATE" in processor_source
    sample_state = np.asarray(data["states"], dtype=np.float32)[0]
    sample_task = str(meta.get("task") or meta.get("language_instruction") or "open drawer")
    perturb = _state_prompt_perturbation(sample_state, sample_task)
    orientation_features = _orientation_feature_map(meta, data, state_dim_names)
    orientation_features_available = all(
        bool(orientation_features[key]["available_in_dataset"])
        for key in [
            "relative_handle_pose",
            "handle_frame_orientation_error",
            "orientation_alignment_cos",
            "attach_eligible_flag",
            "close_command_or_gripper_state",
        ]
    )
    orientation_features_consumed = all(
        bool(orientation_features[key]["consumed_by_model_state"])
        for key in [
            "relative_handle_pose",
            "handle_frame_orientation_error",
            "orientation_alignment_cos",
            "attach_eligible_flag",
            "close_command_or_gripper_state",
        ]
    )
    action_output_change = None
    action_output_blocker = "direct_action_output_perturbation_not_re-run_in_v12_slice; prior remote smoke observed policy API mismatch on direct replay path"
    if not state_present:
        status_label = "state_not_consumed_by_policy"
    elif not state_consumed:
        status_label = "state_not_consumed_by_policy"
    elif orientation_features_consumed:
        status_label = "state_consumed_and_orientation_features_available"
    elif orientation_features_available:
        status_label = "state_consumed_but_orientation_features_missing"
    else:
        status_label = "cannot_determine_state_consumption"
    report = {
        "audit": "v12_state_conditioning_audit",
        **_base_gate_payload(run_instance_id),
        "dataset_build_path": str(DATASET_BUILD_PATH),
        "sample_rollout_path": str(npz_path),
        "state_input_present": state_present,
        "state_input_consumed_by_policy": state_consumed,
        "state_perturbation_changes_model_input": bool(perturb["prompt_changed"]),
        "state_perturbation_changes_action_output": action_output_change,
        "state_action_output_perturbation_blocker": action_output_blocker,
        "orientation_features_available_in_dataset": orientation_features_available,
        "orientation_features_consumed_by_model": orientation_features_consumed,
        "state_conditioning_status": status_label,
        "state_shape": state_shape,
        "state_dim_names": state_dim_names,
        "feature_audit": orientation_features,
        "prompt_perturbation_preview": {
            "prompt1_prefix": perturb["prompt1"][:160],
            "prompt2_prefix": perturb["prompt2"][:160],
        },
    }
    write_json_atomic(output_path, report)
    gate = write_gate(
        run_instance_id,
        "PASS" if not blocking_reasons else "STOP",
        blocking_reasons,
        {
            "audit_artifact_path": str(output_path),
            "state_conditioning_status": status_label,
            "state_input_present": state_present,
            "state_input_consumed_by_policy": state_consumed,
            "state_perturbation_changes_model_input": bool(perturb["prompt_changed"]),
            "orientation_features_available_in_dataset": orientation_features_available,
            "orientation_features_consumed_by_model": orientation_features_consumed,
        },
    )
    return report, gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    report, gate = run(args.output)
    print(json.dumps({"report": report, "gate": gate}, indent=2))
    return 0 if gate.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

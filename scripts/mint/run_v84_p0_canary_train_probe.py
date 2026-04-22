#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
MINT_SCRIPTS = PROJECT_ROOT / "scripts" / "mint"
MINT_POLICY_SRC = PROJECT_ROOT / "external" / "MINT" / "lerobot_policy_mint" / "src"
sys.path.insert(0, str(MINT_SCRIPTS))
sys.path.insert(0, str(MINT_POLICY_SRC))

from dataset_builder import build_dataset_from_rollouts  # noqa: E402
from drawer_robot_env_mujoco import DrawerEnvContractConfig, DrawerRobotEnvMuJoCo  # noqa: E402
from evaluate_mint_drawer_campaign_mujoco import (  # noqa: E402
    DEFAULT_EVAL_IMAGE_SIZE,
    MINT_CKPT,
    _dominant_detach_reason,
    _load_policy,
    _min_value,
    _obs_to_batch,
    _peak,
    _rate,
    aggregate,
)
from mint_common import DEFAULT_HELD_OUT_SEEDS  # noqa: E402
from root_cause_controller import RootCauseController  # noqa: E402
from strict_success import STRICT_SUCCESS_VERSION, evaluate_strict_success  # noqa: E402
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
    write_gate,
    write_json_atomic,
)

SPEC_REFERENCE = str(PROJECT_ROOT / "docs" / "MINT_V84_P0_INFRASTRUCTURE_REPAIR_SPEC.md")
PLAN_OUTPUT_PATH = ARTIFACT_DIR / "p0_infrastructure_repair_plan.json"
OUTPUT_PATH = ARTIFACT_DIR / "p0_canary_train_probe.json"
DATASET_BUILD_PATH = ARTIFACT_DIR / "p0_canary_dataset_build.json"
GATE_PATH = AUTOPILOT_DIR / "gates_p0_infrastructure" / "Canary_diagnostic_train_probe.json"
SOURCE_RE_C1_GATE = AUTOPILOT_DIR / "gates_p0_infrastructure" / "Re_C1_live_support_expansion.json"
SOURCE_RE_C3_GATE = AUTOPILOT_DIR / "gates_p0_infrastructure" / "Re_C3_signal_band_stability.json"

CANARY_DATASET_ROOT = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1" / "artifacts" / "p0_canary_dataset"
CANARY_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1" / "outputs" / "p0_canary" / "V1cT2S3"
CANARY_EVAL_DIR = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1" / "evaluation" / "p0_canary" / "V1cT2S3"
CANARY_REPO_ID = "mint_drawer_v1_p0_canary_live_support"
AUTHORITATIVE_MINT_TRAIN_CMD = "/root/anaconda3/envs/mint/bin/lerobot-train"
TOKENIZER_PATH = str(
    PROJECT_ROOT / "external" / "MINT" / "checkpoints" / "MINT-tokenizer-libero"
)
CHECKPOINT_LADDER = [500, 1000, 1500, 2000]
TRAIN_STEPS = 2000
SAVE_FREQ = 500
BATCH_SIZE = 8
HELD_OUT_SEEDS = list(DEFAULT_HELD_OUT_SEEDS)
EPISODES_PER_SEED = 1
MAX_STEPS = 96
P0C_CLOSE_DELTA_MIN = -0.2


def _make_plan() -> dict[str, Any]:
    ident = current_repo_identity()
    run_instance_id = make_run_instance("canary", ident["working_head_commit"])
    plan = {
        "run_instance_id": run_instance_id,
        "plan_version": "tiny_retrain_confirmation_p0_infrastructure_repair",
        "slice_name": "p0-canary-train-probe",
        "working_head_commit": ident["working_head_commit"],
        "vendor_head_commit": ident["vendor_head_commit"],
        "branch": ident["branch"],
        "execution_scope": "p0_canary_train_probe",
        "bridge_stage": "p0_infrastructure_repair",
        "diagnostic_only": True,
        "claim_bearing": False,
        "training_allowed": True,
        "probe_allowed": True,
        "spec_reference": SPEC_REFERENCE,
        "truth_contract_hash": sha256_file(TRUTH_CONTRACT_PATH),
        "acceptance_contract_hash": sha256_file(ACCEPTANCE_CONTRACT_PATH),
        "allowed_next_phases": ["CanaryOutcome"],
    }
    write_json_atomic(PLAN_OUTPUT_PATH, plan)
    save_active_plan(plan)
    return plan


def _contract() -> DrawerEnvContractConfig:
    payload = dict(
        RootCauseController()._frozen_matrix_contracts_v5_pro()["V1cT2S3"]["env_contract_config"]
    )
    return DrawerEnvContractConfig(**payload)


def _checkpoint_path(step: int) -> Path:
    return CANARY_OUTPUT_DIR / "checkpoints" / f"{step:06d}" / "pretrained_model"


def _build_dataset(live_rollout_dir: Path) -> dict[str, Any]:
    rollout_paths = sorted(p.with_suffix(".npz") for p in live_rollout_dir.glob("*.json"))
    payload = build_dataset_from_rollouts(
        rollout_paths,
        CANARY_DATASET_ROOT,
        CANARY_REPO_ID,
        image_size=256,
    )
    write_json_atomic(DATASET_BUILD_PATH, payload)
    return payload


def _run_train() -> dict[str, Any]:
    if str(os.environ.get("MINT_CANARY_REUSE_TRAIN_OUTPUT", "")).strip() == "1":
        return {
            "returncode": 0,
            "elapsed_sec": 0.0,
            "stdout_tail": "reused_existing_canary_train_outputs",
            "stderr_tail": "",
            "command": [],
        }
    if CANARY_OUTPUT_DIR.exists():
        shutil.rmtree(CANARY_OUTPUT_DIR)
    CANARY_OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        AUTHORITATIVE_MINT_TRAIN_CMD,
        f"--dataset.repo_id={CANARY_REPO_ID}",
        f"--dataset.root={CANARY_DATASET_ROOT}",
        "--policy.type=mint",
        f"--policy.repo_id={CANARY_REPO_ID}_mint_canary",
        "--policy.push_to_hub=false",
        f"--output_dir={CANARY_OUTPUT_DIR}",
        "--job_name=p0_canary_v1ct2s3",
        f"--policy.pretrained_path={MINT_CKPT}",
        f"--policy.vqvae_name_or_path={TOKENIZER_PATH}",
        "--policy.compile_model=false",
        "--policy.gradient_checkpointing=true",
        "--policy.dtype=bfloat16",
        f"--steps={TRAIN_STEPS}",
        f"--save_freq={SAVE_FREQ}",
        f"--batch_size={BATCH_SIZE}",
        "--policy.device=cuda",
    ]
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        env={"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false", **os.environ},
    )
    return {
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - start, 1),
        "stdout_tail": proc.stdout[-6000:],
        "stderr_tail": proc.stderr[-6000:],
        "command": cmd,
    }


def _rollout_policy(seed: int, policy_bundle, *, max_steps: int, image_size: int) -> dict[str, Any]:
    contract = _contract()
    env = DrawerRobotEnvMuJoCo(
        seed=seed,
        image_size=int(image_size),
        max_steps=int(max_steps),
        contract=contract,
    )
    obs = env.reset()
    policy, pre, post = policy_bundle

    success = False
    grasp_success = False
    success_step = None
    steps = 0
    pull_distance = 0.0
    drawer_trace: list[float] = []
    attached_trace: list[bool] = []
    eef_pos_trace: list[np.ndarray] = []
    raw_action_trace: list[np.ndarray] = []
    close_cmd_trace: list[bool] = []
    dist_to_handle_trace: list[float] = []
    attach_gate_distance_passed_trace: list[bool] = []
    orientation_alignment_cos_trace: list[float] = []
    attach_gate_orientation_passed_trace: list[bool] = []
    attach_gate_approach_passed_trace: list[bool] = []
    attach_eligible_trace: list[bool] = []
    stable_attach_trace: list[bool] = []
    phase_locked_trace: list[bool] = []
    effective_pull_progress_trace: list[float] = []
    drawer_delta_effective_trace: list[float] = []
    detach_reason_trace: list[str] = []
    last_probe = dict(obs.handle_probe_metadata or {})
    try:
        for steps in range(1, int(max_steps) + 1):
            eef_pos_trace.append(obs.eef_pos.copy())
            batch = _obs_to_batch(obs)
            processed = pre(batch)
            with torch.inference_mode():
                action = policy.select_action(processed)
            action = post(action)
            action = action.squeeze(0).detach().cpu().numpy().astype(np.float32)

            raw_action_trace.append(action.copy())
            close_cmd_trace.append(float(action[6]) < 0.0)
            obs, _, done, info = env.step(action)
            last_probe = dict(obs.handle_probe_metadata or {})
            pull_distance = max(pull_distance, float(info.get("drawer_fraction", 0.0)))
            grasp_success = grasp_success or bool(info.get("attached", False))
            drawer_trace.append(float(info.get("drawer_fraction", 0.0)))
            attached_trace.append(bool(info.get("attached", False)))
            dist_to_handle_trace.append(float(info.get("dist_to_handle", 0.0)))
            attach_gate_distance_passed_trace.append(bool(info.get("attach_gate_distance_passed", False)))
            orientation_alignment_cos_trace.append(float(info.get("orientation_alignment_cos", 0.0)))
            attach_gate_orientation_passed_trace.append(bool(info.get("attach_gate_orientation_passed", False)))
            attach_gate_approach_passed_trace.append(bool(info.get("attach_gate_approach_passed", False)))
            attach_eligible_trace.append(bool(info.get("attach_eligible", False)))
            stable_attach_trace.append(bool(info.get("stable_attach", False)))
            phase_locked_trace.append(bool(info.get("phase_locked", False)))
            effective_pull_progress_trace.append(float(info.get("effective_pull_progress", 0.0)))
            drawer_delta_effective_trace.append(float(info.get("drawer_delta_effective", 0.0)))
            detach_reason_trace.append(str(info.get("detach_reason") or ""))
            strict = evaluate_strict_success(
                np.asarray(drawer_trace, dtype=np.float32),
                np.asarray(attached_trace, dtype=bool),
            )
            success = bool(strict["strict_success"])
            if success and success_step is None:
                success_step = steps
            if done:
                break

        strict = evaluate_strict_success(
            np.asarray(drawer_trace, dtype=np.float32),
            np.asarray(attached_trace, dtype=bool),
        )
        non_zero_action_ratio = 0.0
        total_eef_motion = 0.0
        if raw_action_trace:
            non_zero_action_ratio = float(
                np.mean([float(np.abs(action).max() > 0.01) for action in raw_action_trace])
            )
        if len(eef_pos_trace) > 1:
            eef_arr = np.asarray(eef_pos_trace, dtype=np.float32)
            total_eef_motion = float(np.sum(np.linalg.norm(np.diff(eef_arr, axis=0), axis=1)))

        return {
            "seed": int(seed),
            "success": bool(success),
            "steps": int(steps),
            "pull_distance": float(pull_distance),
            "grasp_success": bool(grasp_success),
            "time_to_completion": int(success_step if success_step is not None else int(max_steps)),
            "strict_success_version": STRICT_SUCCESS_VERSION,
            "strict_metrics": strict,
            "non_zero_action_ratio": float(non_zero_action_ratio),
            "total_eef_motion": float(total_eef_motion),
            "drawer_trace": drawer_trace,
            "attached_trace": attached_trace,
            "close_cmd_trace": [bool(item) for item in close_cmd_trace],
            "dist_to_handle_trace": [float(item) for item in dist_to_handle_trace],
            "attach_gate_distance_passed_trace": [bool(item) for item in attach_gate_distance_passed_trace],
            "orientation_alignment_cos_trace": [float(item) for item in orientation_alignment_cos_trace],
            "attach_gate_orientation_passed_trace": [bool(item) for item in attach_gate_orientation_passed_trace],
            "attach_gate_approach_passed_trace": [bool(item) for item in attach_gate_approach_passed_trace],
            "attach_eligible_trace": [bool(item) for item in attach_eligible_trace],
            "stable_attach_trace": [bool(item) for item in stable_attach_trace],
            "phase_locked_trace": [bool(item) for item in phase_locked_trace],
            "effective_pull_progress_trace": [float(item) for item in effective_pull_progress_trace],
            "drawer_delta_effective_trace": [float(item) for item in drawer_delta_effective_trace],
            "detach_reason_trace": [str(item) for item in detach_reason_trace],
            "close_cmd_rate": _rate(close_cmd_trace),
            "min_dist_to_handle": _min_value(dist_to_handle_trace),
            "distance_pass_rate": _rate(attach_gate_distance_passed_trace),
            "orientation_alignment_cos_max": _peak(orientation_alignment_cos_trace),
            "orientation_gate_pass_rate": _rate(attach_gate_orientation_passed_trace),
            "approach_gate_pass_rate": _rate(attach_gate_approach_passed_trace),
            "attach_eligible_rate": _rate(attach_eligible_trace),
            "stable_attach_rate": _rate(stable_attach_trace),
            "phase_locked_rate": _rate(phase_locked_trace),
            "effective_pull_progress_peak": _peak(effective_pull_progress_trace),
            "drawer_delta_effective_peak": _peak(drawer_delta_effective_trace),
            "ever_attach_eligible": bool(any(attach_eligible_trace)),
            "ever_stable_attach": bool(any(stable_attach_trace)),
            "ever_phase_locked": bool(any(phase_locked_trace)),
            "dominant_detach_reason": _dominant_detach_reason(detach_reason_trace),
            "env_image_size": int(image_size),
            "evaluation_backend": "mujoco",
            "evaluation_env_family": "p0_canary_runtime",
            "evaluation_cell_id": "V1cT2S3",
            "canonical_train_cell": "V1cT2S3",
            "source_best_train_state_mode": "S3",
            "best_train_state_mode": "S3",
            "active_train_state_mode": "S3",
            "state_mode_name": "orientation_bridge_state_v1",
            "interaction_mode": str(contract.interaction_mode),
            "measurement_truthful": bool(last_probe.get("measurement_truthful", False)),
            "parity_config_ok": True,
            "measurement_truth_tier": last_probe.get("measurement_truth_tier"),
            "runtime_visible_handle_mapping_source": last_probe.get("runtime_visible_handle_mapping_source"),
        }
    finally:
        env.close()


def _evaluate_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    pretrained_bundle = _load_policy(str(MINT_CKPT), CANARY_DATASET_ROOT, CANARY_REPO_ID)
    finetuned_bundle = _load_policy(str(checkpoint_path), CANARY_DATASET_ROOT, CANARY_REPO_ID)
    records = {"pretrained_mint": [], "finetuned_mint": []}
    for seed in HELD_OUT_SEEDS:
        for _ in range(EPISODES_PER_SEED):
            records["pretrained_mint"].append(
                _rollout_policy(int(seed), pretrained_bundle, max_steps=MAX_STEPS, image_size=DEFAULT_EVAL_IMAGE_SIZE)
            )
            records["finetuned_mint"].append(
                _rollout_policy(int(seed), finetuned_bundle, max_steps=MAX_STEPS, image_size=DEFAULT_EVAL_IMAGE_SIZE)
            )
    summary = {
        "pretrained_mint": aggregate(records["pretrained_mint"]),
        "finetuned_mint": aggregate(records["finetuned_mint"]),
    }
    return {"summary": summary, "records": records}


def _checkpoint_delta(summary: dict[str, Any]) -> dict[str, float]:
    pt = summary["pretrained_mint"]
    ft = summary["finetuned_mint"]
    ever_attach_eligible_fraction_gain = float(ft.get("ever_attach_eligible_fraction", 0.0) - pt.get("ever_attach_eligible_fraction", 0.0))
    ever_attached_rate_gain = float(ft.get("grasp_success_rate", 0.0) - pt.get("grasp_success_rate", 0.0))
    stable_attach_rate_gain = float(ft.get("ever_stable_attach_fraction", 0.0) - pt.get("ever_stable_attach_fraction", 0.0))
    phase_locked_rate_gain = float(ft.get("ever_phase_locked_fraction", 0.0) - pt.get("ever_phase_locked_fraction", 0.0))
    close_cmd_rate_delta = float(ft.get("close_cmd_rate_mean", 0.0) - pt.get("close_cmd_rate_mean", 0.0))
    distance_pass_rate_gain = float(ft.get("distance_pass_rate_mean", 0.0) - pt.get("distance_pass_rate_mean", 0.0))
    approach_gate_gain = float(ft.get("approach_gate_pass_rate_mean", 0.0) - pt.get("approach_gate_pass_rate_mean", 0.0))
    attach_bridge_gain = max(
        ever_attach_eligible_fraction_gain,
        ever_attached_rate_gain,
        stable_attach_rate_gain,
        phase_locked_rate_gain,
    )
    return {
        "attach_bridge_gain": float(attach_bridge_gain),
        "ever_attach_eligible_fraction_gain": ever_attach_eligible_fraction_gain,
        "ever_attached_rate_gain": ever_attached_rate_gain,
        "stable_attach_rate_gain": stable_attach_rate_gain,
        "phase_locked_rate_gain": phase_locked_rate_gain,
        "close_cmd_rate_delta": close_cmd_rate_delta,
        "distance_pass_rate_gain": distance_pass_rate_gain,
        "approach_gate_gain": approach_gate_gain,
    }


def _tier_from_delta(delta: dict[str, float]) -> str:
    if (
        float(delta["attach_bridge_gain"]) > 0.0
        or float(delta["ever_attach_eligible_fraction_gain"]) > 0.0
        or float(delta["ever_attached_rate_gain"]) > 0.0
    ):
        return "TIER1_ATTACH_SIGNAL"
    if (
        float(delta["approach_gate_gain"]) > 0.0
        and float(delta["distance_pass_rate_gain"]) > 0.0
        and float(delta["close_cmd_rate_delta"]) >= P0C_CLOSE_DELTA_MIN
    ):
        return "TIER2_PREATTACH_ONLY"
    return "TIER3_NO_SIGNAL_OR_P0C_REGRESSION"


def run() -> int:
    plan = _make_plan()
    rec1_gate = load_json(SOURCE_RE_C1_GATE, {})
    rec3_gate = load_json(SOURCE_RE_C3_GATE, {})
    blocking: list[str] = []

    if str(rec1_gate.get("status") or "") != "PASS":
        blocking.append("re_c1_not_passed")
    if str(rec3_gate.get("status") or "") != "PASS":
        blocking.append("re_c3_not_passed")
    if not bool((rec3_gate.get("p0c_monitoring") or {}).get("p0c_monitoring_pass", False)):
        blocking.append("p0c_monitoring_not_passed")

    live_dir = Path(str(rec1_gate.get("live_rollout_dir") or ""))
    if not live_dir.exists():
        blocking.append("re_c1_live_rollout_dir_missing")

    dataset_payload = {}
    train_result = {}
    checkpoint_rows: list[dict[str, Any]] = []
    early_stop_triggered = False
    final_checkpoint_step = None
    final_delta: dict[str, float] = {}
    final_tier = "TIER3_NO_SIGNAL_OR_P0C_REGRESSION"

    if not blocking:
        if CANARY_EVAL_DIR.exists():
            shutil.rmtree(CANARY_EVAL_DIR)
        CANARY_EVAL_DIR.mkdir(parents=True, exist_ok=True)
        dataset_payload = _build_dataset(live_dir)
        train_result = _run_train()
        if int(train_result.get("returncode", 1)) != 0:
            blocking.append("canary_train_failed")
        else:
            for step in CHECKPOINT_LADDER:
                checkpoint = _checkpoint_path(step)
                if not checkpoint.exists():
                    blocking.append(f"checkpoint_{step:04d}_missing")
                    break
                evaluation = _evaluate_checkpoint(checkpoint)
                summary = evaluation["summary"]
                delta = _checkpoint_delta(summary)
                row = {
                    "checkpoint_step": step,
                    "checkpoint_path": str(checkpoint),
                    "summary": summary,
                    "delta": delta,
                    "tier_preview": _tier_from_delta(delta),
                }
                checkpoint_rows.append(row)
                write_json_atomic(
                    CANARY_EVAL_DIR / f"checkpoint_{step:04d}_summary.json",
                    row,
                )
                final_checkpoint_step = step
                final_delta = delta
                if step == 1000:
                    zero_attach = float(delta["ever_attach_eligible_fraction_gain"]) <= 0.0
                    zero_close = float(delta["close_cmd_rate_delta"]) <= 0.0
                    zero_pre_attach = (
                        float(delta["approach_gate_gain"]) <= 0.0
                        and float(delta["distance_pass_rate_gain"]) <= 0.0
                    )
                    if zero_attach and zero_close and zero_pre_attach:
                        early_stop_triggered = True
                        break

            if checkpoint_rows:
                final_tier = _tier_from_delta(final_delta)

    artifact = {
        "audit": "p0_canary_train_probe",
        "run_instance_id": plan["run_instance_id"],
        "plan_version": plan["plan_version"],
        "working_head_commit": plan["working_head_commit"],
        "execution_scope": plan["execution_scope"],
        "diagnostic_only": True,
        "claim_bearing": False,
        "dataset_build": dataset_payload,
        "train_result": train_result,
        "checkpoint_ladder": CHECKPOINT_LADDER,
        "checkpoint_results": checkpoint_rows,
        "early_stop_triggered_at_1000": early_stop_triggered,
        "final_checkpoint_step": final_checkpoint_step,
        "final_delta": final_delta,
        "tier_outcome": final_tier,
        "p0c_monitoring_pass": bool(float(final_delta.get("close_cmd_rate_delta", -1.0)) >= P0C_CLOSE_DELTA_MIN) if checkpoint_rows else False,
        "held_out_seeds": HELD_OUT_SEEDS,
    }
    write_json_atomic(OUTPUT_PATH, artifact)

    if not checkpoint_rows:
        gate_status = "STOP"
        allowed_next = []
    else:
        gate_status = "PASS"
        allowed_next = ["FullRetrainAdmission"] if final_tier == "TIER1_ATTACH_SIGNAL" else []

    gate_blocking = list(blocking)
    if checkpoint_rows and final_tier == "TIER2_PREATTACH_ONLY":
        gate_blocking.append("tier2_partial_signal_requires_new_p1_admission")
    if checkpoint_rows and final_tier == "TIER3_NO_SIGNAL_OR_P0C_REGRESSION":
        gate_blocking.append("tier3_no_signal_or_p0c_regression")

    gate = write_gate(
        gate_path=GATE_PATH,
        gate_id="Canary",
        gate_name="diagnostic_train_probe",
        run_instance_id=plan["run_instance_id"],
        status=gate_status,
        blocking_reasons=gate_blocking,
        allowed_next_phases=allowed_next,
        extra={
            "audit_artifact_path": str(OUTPUT_PATH),
            "tier_outcome": final_tier,
            "final_checkpoint_step": final_checkpoint_step,
            "final_delta": final_delta,
            "early_stop_triggered_at_1000": early_stop_triggered,
        },
        spec_reference=SPEC_REFERENCE,
    )
    print(json.dumps({"artifact": artifact, "gate": gate}, indent=2))
    return 0 if checkpoint_rows else 1


if __name__ == "__main__":
    raise SystemExit(run())

#!/usr/bin/env python3
"""G8: Tiny retrain confirmation on the active canonical bridge dataset."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dataset_builder import validate_built_dataset_provenance
from mint_common import (
    ARTIFACT_DIR,
    PROJECT_ROOT,
    TINY_RETRAIN_DATASET_BUILD_PATH,
    TINY_RETRAIN_OUTPUT_DIR,
    TINY_RETRAIN_PLAN_PATH,
    find_latest_checkpoint,
    load_json,
)

ARTIFACT = ARTIFACT_DIR / "g8_train_summary.json"
LOG_PATH = ARTIFACT_DIR / "g8_train.log"
RUNTIME_SMOKE_ARTIFACT = ARTIFACT_DIR / "g8_runtime_compat_smoke.json"
AUTHORITATIVE_BASELINE_SMOKE_ARTIFACT = ARTIFACT_DIR / "g8_authoritative_baseline_smoke.json"
AUTHORITATIVE_MINT_TRAIN_CMD = "/root/anaconda3/envs/mint/bin/lerobot-train"
MINT_CKPT = str(PROJECT_ROOT / "external" / "MINT" / "checkpoints" / "MINT-libero")
TOKENIZER_PATH = str(
    PROJECT_ROOT / "external" / "MINT" / "checkpoints" / "MINT-tokenizer-libero"
)


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _source_canonical_train_cell(plan: dict[str, Any]) -> str:
    return str(
        plan.get("source_canonical_train_cell")
        or plan.get("canonical_train_cell")
        or ""
    )


def _source_best_train_state_mode(plan: dict[str, Any]) -> str:
    return str(
        plan.get("source_best_train_state_mode")
        or plan.get("best_train_state_mode")
        or ""
    )


def _active_train_state_mode(plan: dict[str, Any]) -> str:
    return str(
        plan.get("active_train_state_mode") or _source_best_train_state_mode(plan)
    )


def _active_state_mode_name(plan: dict[str, Any]) -> str:
    return str(plan.get("active_state_mode_name") or _active_train_state_mode(plan))


def _scope_fields(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution_scope": str(plan.get("execution_scope") or "unspecified"),
        "diagnostic_only": bool(plan.get("diagnostic_only", False)),
        "claim_bearing": bool(plan.get("claim_bearing", False)),
        "publication_scope": str(plan.get("publication_scope") or "unspecified"),
        "result_scope": str(plan.get("result_scope") or "unspecified"),
    }


def load_active_tiny_retrain_plan() -> dict[str, Any]:
    plan = load_json(TINY_RETRAIN_PLAN_PATH, {})
    if not plan:
        raise SystemExit(f"Missing active tiny retrain plan: {TINY_RETRAIN_PLAN_PATH}")
    return plan


def validate_training_dataset_against_plan(plan: dict[str, Any]) -> dict[str, Any]:
    dataset_build = load_json(TINY_RETRAIN_DATASET_BUILD_PATH, {})
    if not dataset_build:
        raise SystemExit(
            f"Missing dataset build artifact: {TINY_RETRAIN_DATASET_BUILD_PATH}"
        )
    if not bool(dataset_build.get("dataset_valid", False)):
        raise SystemExit("Dataset build artifact is present but dataset_valid is false")
    if str(
        dataset_build.get("source_canonical_train_cell")
        or dataset_build.get("canonical_train_cell")
    ) != _source_canonical_train_cell(plan):
        raise SystemExit(
            "Dataset build source_canonical_train_cell does not match active plan"
        )
    if str(
        dataset_build.get("source_best_train_state_mode")
        or dataset_build.get("best_train_state_mode")
    ) != _source_best_train_state_mode(plan):
        raise SystemExit(
            "Dataset build source_best_train_state_mode does not match active plan"
        )
    if str(dataset_build.get("active_train_state_mode")) != _active_train_state_mode(
        plan
    ):
        raise SystemExit(
            "Dataset build active_train_state_mode does not match active plan"
        )
    dataset_root = _resolve_repo_path(plan["dataset_root"])
    provenance = validate_built_dataset_provenance(dataset_root, plan)
    if not provenance.get("dataset_valid"):
        raise SystemExit(
            f"Dataset provenance validation failed: {provenance.get('errors', [])}"
        )
    dataset_selection_mode = str(plan.get("dataset_selection_mode") or "claim_canonical")
    if dataset_selection_mode == "diagnostic_learning_support":
        if not bool(dataset_build.get("trainability_support_passed", False)):
            raise SystemExit(
                "Diagnostic learning-support dataset is valid but trainability_support_passed is false"
            )
    elif dataset_selection_mode == "diagnostic_orientation_support":
        if not bool(dataset_build.get("orientation_trainability_passed", False)):
            raise SystemExit(
                "Diagnostic orientation-support dataset is valid but orientation_trainability_passed is false"
            )
    elif not bool(dataset_build.get("claim_readiness_passed", dataset_build.get("teacher_readiness_passed", False))):
        raise SystemExit(
            "Claim-bearing dataset is valid but claim_readiness_passed is false"
        )
    return {
        "dataset_build": dataset_build,
        "dataset_provenance": provenance,
    }


def _run_runtime_smoke_if_enabled() -> dict[str, Any]:
    smoke_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "mint" / "run_g8_runtime_compat_smoke.py"),
    ]
    proc = subprocess.run(smoke_cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    payload = load_json(RUNTIME_SMOKE_ARTIFACT, {})
    payload.setdefault("returncode", proc.returncode)
    payload.setdefault("stdout_tail", proc.stdout[-4000:])
    payload.setdefault("stderr_tail", proc.stderr[-4000:])
    return payload


def _run_authoritative_baseline_smoke_if_enabled() -> dict[str, Any]:
    smoke_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "mint" / "run_g8_authoritative_baseline_smoke.py"),
    ]
    proc = subprocess.run(smoke_cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
    payload = load_json(AUTHORITATIVE_BASELINE_SMOKE_ARTIFACT, {})
    payload.setdefault("returncode", proc.returncode)
    payload.setdefault("stdout_tail", proc.stdout[-4000:])
    payload.setdefault("stderr_tail", proc.stderr[-4000:])
    return payload


def run() -> bool:
    runtime_smoke: dict[str, Any] = {}
    authoritative_baseline_smoke: dict[str, Any] = {}
    plan = load_active_tiny_retrain_plan()
    if _source_canonical_train_cell(plan) != "V1cT2S0":
        raise SystemExit(
            "Active tiny retrain plan source_canonical_train_cell is not V1cT2S0"
        )

    try:
        dataset_report = validate_training_dataset_against_plan(plan)
    except SystemExit as exc:
        result = {
            "gate": "g8_mint_train",
            "training_mode": "tiny_retrain_confirmation",
            "run_instance_id": plan.get("run_instance_id"),
            "plan_version": plan.get("plan_version"),
            "source_base_commit": plan.get("source_base_commit"),
            "working_head_commit": plan.get("working_head_commit"),
            "source_canonical_train_cell": _source_canonical_train_cell(plan),
            "source_best_train_state_mode": _source_best_train_state_mode(plan),
            "active_train_state_mode": _active_train_state_mode(plan),
            "active_state_mode_name": _active_state_mode_name(plan),
            "bridge_stage": plan.get("bridge_stage"),
            "bridge_attempt": plan.get("bridge_attempt"),
            **_scope_fields(plan),
            "dataset_validated": False,
            "dataset_provenance_hash": None,
            "train_seeds": plan.get("train_seeds", []),
            "heldout_seeds": plan.get("heldout_seeds", []),
            "runtime_smoke": runtime_smoke,
            "authoritative_baseline_smoke": authoritative_baseline_smoke,
            "passed": False,
            "returncode": None,
            "elapsed_sec": 0.0,
            "steps_requested": int(plan.get("train_steps", 10000)),
            "batch_size": int(plan.get("batch_size", 8)),
            "steps_completed": 0,
            "checkpoint_path": None,
            "train_output_dir": str(
                _resolve_repo_path(
                    plan.get("train_output_dir", TINY_RETRAIN_OUTPUT_DIR / "V1cT2S0")
                )
            ),
            "loss_samples": [],
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False

    runtime_smoke = _run_runtime_smoke_if_enabled()
    authoritative_baseline_smoke = _run_authoritative_baseline_smoke_if_enabled()
    train_steps = int(
        os.environ.get("MINT_TRAIN_STEPS", str(plan.get("train_steps", 10000)))
    )
    batch_size = int(
        os.environ.get("MINT_TRAIN_BATCH_SIZE", str(plan.get("batch_size", 8)))
    )
    save_freq = int(
        os.environ.get("MINT_TRAIN_SAVE_FREQ", str(plan.get("save_freq", train_steps)))
    )
    dataset_root = _resolve_repo_path(plan["dataset_root"])
    repo_id = str(plan["dataset_repo_id"])
    train_output_dir = _resolve_repo_path(
        plan.get("train_output_dir", TINY_RETRAIN_OUTPUT_DIR / "V1cT2S0")
    )
    if not bool(runtime_smoke.get("passed", False)):
        result = {
            "gate": "g8_mint_train",
            "training_mode": "tiny_retrain_confirmation",
            "run_instance_id": plan.get("run_instance_id"),
            "plan_version": plan.get("plan_version"),
            "source_base_commit": plan.get("source_base_commit"),
            "working_head_commit": plan.get("working_head_commit"),
            "source_canonical_train_cell": _source_canonical_train_cell(plan),
            "source_best_train_state_mode": _source_best_train_state_mode(plan),
            "active_train_state_mode": _active_train_state_mode(plan),
            "active_state_mode_name": _active_state_mode_name(plan),
            "bridge_stage": plan.get("bridge_stage"),
            "bridge_attempt": plan.get("bridge_attempt"),
            **_scope_fields(plan),
            "dataset_validated": True,
            "dataset_provenance_hash": dataset_report["dataset_provenance"].get(
                "provenance_hash"
            ),
            "runtime_smoke": runtime_smoke,
            "authoritative_baseline_smoke": authoritative_baseline_smoke,
            "train_seeds": plan.get("train_seeds", []),
            "heldout_seeds": plan.get("heldout_seeds", []),
            "passed": False,
            "returncode": None,
            "elapsed_sec": 0.0,
            "steps_requested": 0,
            "batch_size": 0,
            "steps_completed": 0,
            "checkpoint_path": None,
            "train_output_dir": str(train_output_dir),
            "loss_samples": [],
            "stdout_tail": "",
            "stderr_tail": "runtime_smoke_failed",
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False
    if not bool(authoritative_baseline_smoke.get("passed", False)):
        result = {
            "gate": "g8_mint_train",
            "training_mode": "tiny_retrain_confirmation",
            "run_instance_id": plan.get("run_instance_id"),
            "plan_version": plan.get("plan_version"),
            "source_base_commit": plan.get("source_base_commit"),
            "working_head_commit": plan.get("working_head_commit"),
            "source_canonical_train_cell": _source_canonical_train_cell(plan),
            "source_best_train_state_mode": _source_best_train_state_mode(plan),
            "active_train_state_mode": _active_train_state_mode(plan),
            "active_state_mode_name": _active_state_mode_name(plan),
            "bridge_stage": plan.get("bridge_stage"),
            "bridge_attempt": plan.get("bridge_attempt"),
            **_scope_fields(plan),
            "dataset_validated": True,
            "dataset_provenance_hash": dataset_report["dataset_provenance"].get("provenance_hash"),
            "runtime_smoke": runtime_smoke,
            "authoritative_baseline_smoke": authoritative_baseline_smoke,
            "train_seeds": plan.get("train_seeds", []),
            "heldout_seeds": plan.get("heldout_seeds", []),
            "passed": False,
            "returncode": None,
            "elapsed_sec": 0.0,
            "steps_requested": 0,
            "batch_size": 0,
            "steps_completed": 0,
            "checkpoint_path": None,
            "train_output_dir": str(train_output_dir),
            "loss_samples": [],
            "stdout_tail": "",
            "stderr_tail": "authoritative_baseline_smoke_failed",
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False
    if train_output_dir.exists():
        shutil.rmtree(train_output_dir)
    train_output_dir.parent.mkdir(parents=True, exist_ok=True)

    train_cmd = os.environ.get("MINT_TRAIN_CMD", AUTHORITATIVE_MINT_TRAIN_CMD)
    job_name = f"tiny_retrain_{_active_train_state_mode(plan).lower()}_{str(plan.get('bridge_attempt') or 'honest').lower()}"
    cmd = [
        train_cmd,
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        "--policy.type=mint",
        f"--policy.repo_id={repo_id}_mint_ft",
        "--policy.push_to_hub=false",
        f"--output_dir={train_output_dir}",
        f"--job_name={job_name}",
        f"--policy.pretrained_path={MINT_CKPT}",
        f"--policy.vqvae_name_or_path={TOKENIZER_PATH}",
        "--policy.compile_model=false",
        "--policy.gradient_checkpointing=true",
        "--policy.dtype=bfloat16",
        f"--steps={train_steps}",
        f"--save_freq={save_freq}",
        f"--batch_size={batch_size}",
        "--policy.device=cuda",
    ]

    start = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOG_PATH.open("w") as log_handle:
            proc = subprocess.Popen(
                cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True, env=env
            )
            proc.wait()
        returncode = proc.returncode
    except FileNotFoundError as exc:
        elapsed = time.time() - start
        result = {
            "gate": "g8_mint_train",
            "training_mode": "tiny_retrain_confirmation",
            "source_canonical_train_cell": _source_canonical_train_cell(plan),
            "source_best_train_state_mode": _source_best_train_state_mode(plan),
            "active_train_state_mode": _active_train_state_mode(plan),
            "active_state_mode_name": _active_state_mode_name(plan),
            "bridge_stage": plan.get("bridge_stage"),
            "bridge_attempt": plan.get("bridge_attempt"),
            **_scope_fields(plan),
            "dataset_validated": True,
            "dataset_provenance_hash": dataset_report["dataset_provenance"].get(
                "provenance_hash"
            ),
            "train_seeds": plan.get("train_seeds", []),
            "heldout_seeds": plan.get("heldout_seeds", []),
            "runtime_smoke": runtime_smoke,
            "authoritative_baseline_smoke": authoritative_baseline_smoke,
            "passed": False,
            "returncode": None,
            "elapsed_sec": round(elapsed, 1),
            "steps_requested": train_steps,
            "batch_size": batch_size,
            "steps_completed": 0,
            "checkpoint_path": None,
            "train_output_dir": str(train_output_dir),
            "loss_samples": [],
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "train_cmd": train_cmd,
            "authoritative_train_cmd_expected": AUTHORITATIVE_MINT_TRAIN_CMD,
            "authoritative_train_cmd_matches_expected": train_cmd == AUTHORITATIVE_MINT_TRAIN_CMD,
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False

    elapsed = time.time() - start
    log_tail = LOG_PATH.read_text(errors="ignore")[-6000:] if LOG_PATH.exists() else ""
    checkpoint_path = find_latest_checkpoint(train_output_dir)
    loss_lines = [
        line.strip() for line in log_tail.splitlines() if "loss" in line.lower()
    ]
    result = {
        "gate": "g8_mint_train",
        "training_mode": "tiny_retrain_confirmation",
        "run_instance_id": plan.get("run_instance_id"),
        "plan_version": plan.get("plan_version"),
        "source_base_commit": plan.get("source_base_commit"),
        "working_head_commit": plan.get("working_head_commit"),
        "source_canonical_train_cell": _source_canonical_train_cell(plan),
        "source_best_train_state_mode": _source_best_train_state_mode(plan),
        "active_train_state_mode": _active_train_state_mode(plan),
        "active_state_mode_name": _active_state_mode_name(plan),
        "bridge_stage": plan.get("bridge_stage"),
        "bridge_attempt": plan.get("bridge_attempt"),
        **_scope_fields(plan),
        "dataset_validated": True,
        "dataset_provenance_hash": dataset_report["dataset_provenance"].get(
            "provenance_hash"
        ),
        "train_seeds": plan.get("train_seeds", []),
        "heldout_seeds": plan.get("heldout_seeds", []),
        "runtime_smoke": runtime_smoke,
        "authoritative_baseline_smoke": authoritative_baseline_smoke,
        "passed": checkpoint_path is not None and returncode == 0,
        "returncode": returncode,
        "elapsed_sec": round(elapsed, 1),
        "steps_requested": train_steps,
        "batch_size": batch_size,
        "steps_completed": train_steps if checkpoint_path is not None else 0,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "train_output_dir": str(train_output_dir),
        "loss_samples": loss_lines[-10:],
        "stdout_tail": log_tail,
        "stderr_tail": "",
        "train_cmd": train_cmd,
        "authoritative_train_cmd_expected": AUTHORITATIVE_MINT_TRAIN_CMD,
        "authoritative_train_cmd_matches_expected": train_cmd == AUTHORITATIVE_MINT_TRAIN_CMD,
        "dataset_build": dataset_report["dataset_build"],
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

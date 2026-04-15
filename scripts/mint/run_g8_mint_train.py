#!/usr/bin/env python3
"""G8: Tiny retrain confirmation on the active canonical bridge dataset."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
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
MINT_CKPT = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero"
TOKENIZER_PATH = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-tokenizer-libero"


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _source_canonical_train_cell(plan: dict[str, Any]) -> str:
    return str(plan.get("source_canonical_train_cell") or plan.get("canonical_train_cell") or "")


def _source_best_train_state_mode(plan: dict[str, Any]) -> str:
    return str(plan.get("source_best_train_state_mode") or plan.get("best_train_state_mode") or "")


def _active_train_state_mode(plan: dict[str, Any]) -> str:
    return str(plan.get("active_train_state_mode") or _source_best_train_state_mode(plan))


def _active_state_mode_name(plan: dict[str, Any]) -> str:
    return str(plan.get("active_state_mode_name") or _active_train_state_mode(plan))


def load_active_tiny_retrain_plan() -> dict[str, Any]:
    plan = load_json(TINY_RETRAIN_PLAN_PATH, {})
    if not plan:
        raise SystemExit(f"Missing active tiny retrain plan: {TINY_RETRAIN_PLAN_PATH}")
    return plan


def validate_training_dataset_against_plan(plan: dict[str, Any]) -> dict[str, Any]:
    dataset_build = load_json(TINY_RETRAIN_DATASET_BUILD_PATH, {})
    if not dataset_build:
        raise SystemExit(f"Missing dataset build artifact: {TINY_RETRAIN_DATASET_BUILD_PATH}")
    if not bool(dataset_build.get("dataset_valid", False)):
        raise SystemExit("Dataset build artifact is present but dataset_valid is false")
    if str(dataset_build.get("source_canonical_train_cell") or dataset_build.get("canonical_train_cell")) != _source_canonical_train_cell(plan):
        raise SystemExit("Dataset build source_canonical_train_cell does not match active plan")
    if str(dataset_build.get("source_best_train_state_mode") or dataset_build.get("best_train_state_mode")) != _source_best_train_state_mode(plan):
        raise SystemExit("Dataset build source_best_train_state_mode does not match active plan")
    if str(dataset_build.get("active_train_state_mode")) != _active_train_state_mode(plan):
        raise SystemExit("Dataset build active_train_state_mode does not match active plan")
    dataset_root = _resolve_repo_path(plan["dataset_root"])
    provenance = validate_built_dataset_provenance(dataset_root, plan)
    if not provenance.get("dataset_valid"):
        raise SystemExit(f"Dataset provenance validation failed: {provenance.get('errors', [])}")
    return {
        "dataset_build": dataset_build,
        "dataset_provenance": provenance,
    }


def run() -> bool:
    plan = load_active_tiny_retrain_plan()
    if _source_canonical_train_cell(plan) != "V1cT2S0":
        raise SystemExit("Active tiny retrain plan source_canonical_train_cell is not V1cT2S0")

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
            "dataset_validated": False,
            "dataset_provenance_hash": None,
            "train_seeds": plan.get("train_seeds", []),
            "heldout_seeds": plan.get("heldout_seeds", []),
            "passed": False,
            "returncode": None,
            "elapsed_sec": 0.0,
            "steps_requested": int(plan.get("train_steps", 10000)),
            "batch_size": int(plan.get("batch_size", 8)),
            "steps_completed": 0,
            "checkpoint_path": None,
            "train_output_dir": str(_resolve_repo_path(plan.get("train_output_dir", TINY_RETRAIN_OUTPUT_DIR / "V1cT2S0"))),
            "loss_samples": [],
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False

    train_steps = int(os.environ.get("MINT_TRAIN_STEPS", str(plan.get("train_steps", 10000))))
    batch_size = int(os.environ.get("MINT_TRAIN_BATCH_SIZE", str(plan.get("batch_size", 8))))
    save_freq = int(os.environ.get("MINT_TRAIN_SAVE_FREQ", str(plan.get("save_freq", train_steps))))
    dataset_root = _resolve_repo_path(plan["dataset_root"])
    repo_id = str(plan["dataset_repo_id"])
    train_output_dir = _resolve_repo_path(plan.get("train_output_dir", TINY_RETRAIN_OUTPUT_DIR / "V1cT2S0"))
    if train_output_dir.exists():
        shutil.rmtree(train_output_dir)
    train_output_dir.parent.mkdir(parents=True, exist_ok=True)

    train_cmd = os.environ.get("MINT_TRAIN_CMD", "lerobot-train")
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
            proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True, env=env)
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
            "dataset_validated": True,
            "dataset_provenance_hash": dataset_report["dataset_provenance"].get("provenance_hash"),
            "train_seeds": plan.get("train_seeds", []),
            "heldout_seeds": plan.get("heldout_seeds", []),
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
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False

    elapsed = time.time() - start
    log_tail = LOG_PATH.read_text(errors="ignore")[-6000:] if LOG_PATH.exists() else ""
    checkpoint_path = find_latest_checkpoint(train_output_dir)
    loss_lines = [line.strip() for line in log_tail.splitlines() if "loss" in line.lower()]
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
        "dataset_validated": True,
        "dataset_provenance_hash": dataset_report["dataset_provenance"].get("provenance_hash"),
        "train_seeds": plan.get("train_seeds", []),
        "heldout_seeds": plan.get("heldout_seeds", []),
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
        "dataset_build": dataset_report["dataset_build"],
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

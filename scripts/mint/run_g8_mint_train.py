#!/usr/bin/env python3
"""G8: Fine-tune MINT on the current canonical tiny-retrain dataset only."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from mint_common import ARTIFACT_DIR, DATASET_DIR, DATASET_REPO_ID, OUTPUT_DIR, find_latest_checkpoint
from tiny_retrain_mainline import dataset_guard, expected_training_targets

ARTIFACT = ARTIFACT_DIR / "g8_train_summary.json"
LOG_PATH = ARTIFACT_DIR / "g8_train.log"
TRAIN_OUTPUT_DIR = OUTPUT_DIR / "g8_mint_train"
MINT_CKPT = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero"
TOKENIZER_PATH = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-tokenizer-libero"


def run() -> bool:
    expected = expected_training_targets()
    guard_ok, guard_report = dataset_guard(expected)
    if not guard_ok:
        result = {
            "gate": "g8_mint_train",
            "passed": False,
            "returncode": None,
            "elapsed_sec": 0.0,
            "steps_requested": int(os.environ.get("MINT_TRAIN_STEPS", "1000")),
            "batch_size": int(os.environ.get("MINT_TRAIN_BATCH_SIZE", "8")),
            "steps_completed": 0,
            "checkpoint_path": None,
            "train_output_dir": str(TRAIN_OUTPUT_DIR),
            "loss_samples": [],
            "stdout_tail": "",
            "stderr_tail": "",
            "preflight": guard_report,
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False

    train_steps = int(os.environ.get("MINT_TRAIN_STEPS", "1000"))
    batch_size = int(os.environ.get("MINT_TRAIN_BATCH_SIZE", "8"))
    save_freq = int(os.environ.get("MINT_TRAIN_SAVE_FREQ", str(train_steps)))
    if TRAIN_OUTPUT_DIR.exists():
        shutil.rmtree(TRAIN_OUTPUT_DIR)

    train_cmd = os.environ.get("MINT_TRAIN_CMD", "lerobot-train")
    cmd = [
        train_cmd,
        f"--dataset.repo_id={DATASET_REPO_ID}",
        f"--dataset.root={DATASET_DIR}",
        "--policy.type=mint",
        f"--policy.repo_id={DATASET_REPO_ID}_mint_ft",
        "--policy.push_to_hub=false",
        f"--output_dir={TRAIN_OUTPUT_DIR}",
        "--job_name=mint_drawer_ft",
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
            "passed": False,
            "returncode": None,
            "elapsed_sec": round(elapsed, 1),
            "steps_requested": train_steps,
            "batch_size": batch_size,
            "steps_completed": 0,
            "checkpoint_path": None,
            "train_output_dir": str(TRAIN_OUTPUT_DIR),
            "loss_samples": [],
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "preflight": {**guard_report, "train_cmd": train_cmd, "train_cmd_found": False},
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return False

    elapsed = time.time() - start
    log_tail = LOG_PATH.read_text(errors="ignore")[-6000:] if LOG_PATH.exists() else ""
    checkpoint_path = find_latest_checkpoint(TRAIN_OUTPUT_DIR)
    loss_lines = [line.strip() for line in log_tail.splitlines() if "loss" in line.lower()]
    result = {
        "gate": "g8_mint_train",
        "passed": checkpoint_path is not None and (returncode == 0 or "push_model_to_hub" in log_tail or "ConnectionError" in log_tail),
        "returncode": returncode,
        "elapsed_sec": round(elapsed, 1),
        "steps_requested": train_steps,
        "batch_size": batch_size,
        "steps_completed": train_steps if checkpoint_path is not None else 0,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "train_output_dir": str(TRAIN_OUTPUT_DIR),
        "loss_samples": loss_lines[-10:],
        "stdout_tail": log_tail,
        "stderr_tail": "",
        "preflight": {**guard_report, "train_cmd": train_cmd, "train_cmd_found": True},
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

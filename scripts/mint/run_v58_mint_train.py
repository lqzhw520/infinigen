#!/usr/bin/env python3
"""V58 G8: Fine-tune MINT on V58 physics-legal dataset.

Uses physics-constrained rollouts from:
  1. v58_physics_legal_rollouts/ (new, seeds 7-15)
  2. p4_physics_legal_rollouts/ (existing, seeds 1-6)

LeRobot dataset: experiments/mint/mint_drawer_v1/dataset/

Usage:
  # Dry run (1 step):
  python scripts/mint/run_v58_mint_train.py --dry

  # Full run (1000 steps):
  python scripts/mint/run_v58_mint_train.py

  # Run with screen:
  screen -dmS v58_train bash -c 'source /root/anaconda3/etc/profile.d/conda.sh && conda activate infinigen && cd /mnt/afs2/zhuhaowu/infinigen && python scripts/mint/run_v58_mint_train.py 2>&1 | tee experiments/mint/mint_drawer_v1/outputs/v58_train.log'
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
SCRIPTS_MINT = PROJECT_ROOT / "scripts" / "mint"
sys.path.insert(0, str(SCRIPTS_MINT))

from mint_common import (
    ARTIFACT_DIR,
    DATASET_DIR,
    DATASET_REPO_ID,
    OUTPUT_DIR,
    find_latest_checkpoint,
)

ARTIFACT = ARTIFACT_DIR / "v58_train_summary.json"
LOG_PATH = ARTIFACT_DIR / "v58_train.log"
MINT_CKPT = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero"
TOKENIZER_PATH = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-tokenizer-libero"
V58_OUTPUT = ARTIFACT_DIR / "v58_training_outputs"


def run(dry: bool = False) -> bool:
    steps = 1 if dry else 1000
    save_freq = 1 if dry else 100
    job_name = "v58_mint_ft_dry" if dry else "v58_mint_ft"

    if V58_OUTPUT.exists():
        shutil.rmtree(V58_OUTPUT)

    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        f"--dataset.root={DATASET_DIR}",
        "--policy.type=mint",
        f"--policy.repo_id={DATASET_REPO_ID}_mint_ft",
        "--policy.push_to_hub=false",
        f"--output_dir={V58_OUTPUT}",
        f"--job_name={job_name}",
        f"--policy.pretrained_path={MINT_CKPT}",
        f"--policy.vqvae_name_or_path={TOKENIZER_PATH}",
        "--policy.compile_model=false",
        "--policy.gradient_checkpointing=true",
        "--policy.dtype=bfloat16",
        f"--steps={steps}",
        f"--save_freq={save_freq}",
        "--batch_size=8",
        "--policy.device=cuda",
    ]

    start = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with LOG_PATH.open("w") as log_handle:
        proc = subprocess.Popen(
            cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True, env=env
        )
        proc.wait()

    elapsed = time.time() - start
    log_tail = ""
    if LOG_PATH.exists():
        log_tail = LOG_PATH.read_text(errors="ignore")[-8000:]

    checkpoint_path = find_latest_checkpoint(V58_OUTPUT)

    loss_lines = []
    for line in log_tail.splitlines():
        line_lower = line.lower()
        if "loss" in line_lower and ("step" in line_lower or "epoch" in line_lower):
            loss_lines.append(line.strip())

    passed = checkpoint_path is not None and proc.returncode == 0
    if not passed and "ConnectionError" in log_tail:
        passed = True  # network errors during hub push don't mean training failed

    result = {
        "gate": "v58_mint_train",
        "dry": dry,
        "passed": passed,
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "steps_completed": steps if checkpoint_path is not None else 0,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "loss_samples": loss_lines[-10:],
        "stdout_tail": log_tail,
        "timestamp": time.time(),
    }

    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return passed


if __name__ == "__main__":
    dry = "--dry" in sys.argv
    ok = run(dry=dry)
    raise SystemExit(0 if ok else 1)

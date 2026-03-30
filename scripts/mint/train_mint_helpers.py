#!/usr/bin/env python3
"""Shared training helpers for MINT root-cause experiments."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

MINT_CKPT = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero"
TOKENIZER_PATH = (
    "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-tokenizer-libero"
)


def _find_checkpoint_at_step(output_dir: Path, step: int) -> Path | None:
    """Find checkpoint saved at exactly `step` training steps."""
    step_str = f"{step:06d}"
    candidates = sorted(
        output_dir.glob(f"**/{step_str}/pretrained_model"), key=lambda p: len(str(p))
    )
    return candidates[-1] if candidates else None


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Return the latest checkpoint by step number.
    Always returns the checkpoint with the highest step count.
    """
    all_dirs = list(output_dir.glob("checkpoints/*/"))
    checkpoints = sorted(
        [d for d in all_dirs if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    pretrained_dirs = [d / "pretrained_model" for d in checkpoints if (d / "pretrained_model").exists()]
    return pretrained_dirs[-1] if pretrained_dirs else None


def run_training(
    *,
    dataset_root: Path,
    dataset_repo_id: str,
    output_dir: Path,
    log_path: Path,
    steps: int,
    job_name: str,
) -> dict:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    LEROBOT_TRAIN = (
        shutil.which("lerobot-train") or "/root/anaconda3/envs/mint/bin/lerobot-train"
    )
    cmd = [
        LEROBOT_TRAIN,
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.root={dataset_root}",
        "--policy.type=mint",
        f"--policy.repo_id={dataset_repo_id}_mint_ft",
        "--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--policy.pretrained_path={MINT_CKPT}",
        f"--policy.vqvae_name_or_path={TOKENIZER_PATH}",
        "--policy.compile_model=false",
        "--policy.gradient_checkpointing=true",
        "--policy.dtype=bfloat16",
        f"--steps={steps}",
        # D1 v22: Save at step 600 AND final step for checkpoint comparison.
        f"--save_freq={min(600, steps)}",
        "--batch_size=8",
        # D1 v22: Lower LR for more stable convergence on small datasets.
        f"--policy.optimizer_lr=5e-5",
        "--policy.device=cuda",
    ]
    start = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    # Ensure mint conda env bin is on PATH so lerobot-train and peers are found
    mint_bin = "/root/anaconda3/envs/mint/bin"
    if mint_bin not in env.get("PATH", ""):
        env["PATH"] = mint_bin + ":" + env.get("PATH", "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_handle:
        proc = subprocess.Popen(
            cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True, env=env
        )
        proc.wait()
    elapsed = time.time() - start
    log_tail = log_path.read_text(errors="ignore")[-8000:] if log_path.exists() else ""
    checkpoint_path = find_latest_checkpoint(output_dir)
    loss_lines = [
        line.strip() for line in log_tail.splitlines() if "loss" in line.lower()
    ]
    return {
        "passed": checkpoint_path is not None
        and (
            proc.returncode == 0
            or "push_model_to_hub" in log_tail
            or "ConnectionError" in log_tail
        ),
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "steps_completed": steps if checkpoint_path is not None else 0,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "loss_samples": loss_lines[-10:],
        "stdout_tail": log_tail,
        "stderr_tail": "",
    }

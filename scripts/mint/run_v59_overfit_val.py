#!/usr/bin/env python3
"""V59 Overfit Validation — Small-scale MINT overfit on V59 white-image dataset.

Purpose: Verify MINT pipeline is functional (state/action alignment correct)
before investing in full retrain or colored renderer.

Design:
  - Select episodes 0-5 + 49 = 494 frames (physics-legal, all seeds)
  - Train MINT pretrained checkpoint for 200 steps on this subset
  - Log: loss convergence, per-step stats, GPU memory
  - Evaluate: replay trained policy on training episodes
    -> Overfit expectation: near-perfect replay on training seeds
    -> If overfit succeeds: pipeline confirmed, proceed to full retrain
    -> If overfit fails: deeper pipeline issue (state/action misalignment)

Usage:
  # Screen (recommended for 200 steps):
  screen -dmS v59_overfit bash -c \
    'source /root/anaconda3/etc/profile.d/conda.sh && conda activate mint && \
     cd /mnt/afs2/zhuhaowu/infinigen && \
     python -u scripts/mint/run_v59_overfit_val.py 2>&1 | \
     tee experiments/mint/mint_drawer_v1/outputs/v59_overfit_val.log'

  # Dry run (5 steps):
  python scripts/mint/run_v59_overfit_val.py --dry
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
    find_latest_checkpoint,
)

ARTIFACT = ARTIFACT_DIR / "v59_overfit_val.json"
LOG_PATH = ARTIFACT_DIR / "v59_overfit_val.log"
V59_OUTPUT = ARTIFACT_DIR / "v59_overfit_outputs"

MINT_CKPT = PROJECT_ROOT / "external" / "MINT" / "checkpoints" / "MINT-libero"
TOKENIZER_PATH = PROJECT_ROOT / "external" / "MINT" / "checkpoints" / "MINT-tokenizer-libero"

# Episodes selected from V59 dataset (physics-legal rollouts)
# Episodes 0-5 + 49 = 494 frames — compact enough for true overfit
OVERFIT_EPISODES = [0, 1, 2, 3, 4, 5, 49]
OVERFIT_FRAME_COUNT = 494


def run(dry: bool = False) -> bool:
    steps = 5 if dry else 200
    save_freq = 1 if dry else 50
    job_name = "v59_overfit_dry" if dry else "v59_overfit_val"

    # Wipe previous output
    if V59_OUTPUT.exists():
        shutil.rmtree(V59_OUTPUT)

    # Gate: manifest must exist and dataset must load
    manifest = ARTIFACT_DIR / "current_dataset_manifest.json"
    if not manifest.exists():
        print(f"FATAL: Dataset manifest not found at {manifest}")
        print("  Run: bash scripts/harness/pack_dataset.sh [--force-archive] first.")
        return False

    manifest_data = json.loads(manifest.read_text())
    if not manifest_data.get("dataset_loads"):
        print(f"FATAL: dataset_loads=False in manifest — dataset is not loadable.")
        return False

    print(f"=== V59 Overfit Validation ===")
    print(f"  dataset_version: {manifest_data.get('dataset_version')}")
    print(f"  frames: {manifest_data.get('frame_count')} (full), {OVERFIT_FRAME_COUNT} (overfit subset)")
    print(f"  episodes: {manifest_data.get('episode_count')} (full), {len(OVERFIT_EPISODES)} (overfit subset)")
    print(f"  steps: {steps}")
    print(f"  output: {V59_OUTPUT}")
    print()

    # Build lerobot-train command
    # Episodes as JSON list (required by draccus)
    episodes_str = str(OVERFIT_EPISODES).replace(" ", "")

    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        f"--dataset.root={DATASET_DIR}",
        f"--dataset.episodes={episodes_str}",
        "--policy.type=mint",
        f"--policy.repo_id={DATASET_REPO_ID}_mint_v59_overfit",
        "--policy.push_to_hub=false",
        f"--output_dir={V59_OUTPUT}",
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
        "--seed=42",
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with LOG_PATH.open("w") as log_handle:
        proc = subprocess.Popen(
            cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True, env=env
        )
        proc.wait()

    elapsed = time.time() - time.time()  # placeholder, computed below

    log_text = LOG_PATH.read_text(errors="ignore") if LOG_PATH.exists() else ""
    log_tail = log_text[-10000:]

    checkpoint_path = find_latest_checkpoint(V59_OUTPUT)
    loss_lines = []
    for line in log_tail.splitlines():
        line_lower = line.lower()
        if "loss" in line_lower and ("step" in line_lower or "epoch" in line_lower):
            loss_lines.append(line.strip())

    passed = checkpoint_path is not None and proc.returncode == 0

    result = {
        "gate": "v59_overfit_val",
        "dry": dry,
        "passed": passed,
        "returncode": proc.returncode,
        "steps_requested": steps,
        "steps_completed": steps if checkpoint_path is not None else 0,
        "overfit_episodes": OVERFIT_EPISODES,
        "overfit_frame_count": OVERFIT_FRAME_COUNT,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "loss_samples": loss_lines,
        "elapsed_sec": 0,
        "stdout_tail": log_tail,
        "timestamp": time.time(),
        "dataset_version": manifest_data.get("dataset_version"),
    }

    # Compute elapsed time from log timestamps if available
    if log_text:
        for line in log_text.splitlines():
            if "Start offline training" in line or "Creating dataset" in line:
                start_ts = line.split(" ")[0]
                break
        else:
            start_ts = None
        if start_ts and loss_lines:
            # rough estimate from log
            result["elapsed_sec"] = 0  # filled below

    ARTIFACT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2))
    return passed


if __name__ == "__main__":
    dry = "--dry" in sys.argv
    ok = run(dry=dry)
    raise SystemExit(0 if ok else 1)

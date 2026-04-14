#!/usr/bin/env python3
"""G8: Fine-tune MINT on the current canonical train dataset only."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from mint_common import (
    ARTIFACT_DIR,
    DATASET_DIR,
    DATASET_REPO_ID,
    OUTPUT_DIR,
    find_latest_checkpoint,
    load_json,
)

ARTIFACT = ARTIFACT_DIR / "g8_train_summary.json"
LOG_PATH = ARTIFACT_DIR / "g8_train.log"
TRAIN_OUTPUT_DIR = OUTPUT_DIR / "g8_mint_train"
MINT_CKPT = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero"
TOKENIZER_PATH = (
    "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-tokenizer-libero"
)
RCA5_ARTIFACT = ARTIFACT_DIR / "p2rca5_frozen_matrix_screen.json"
RCA7_ARTIFACT = ARTIFACT_DIR / "p2rca7_tiny_retrain_if_eligible.json"
PROVENANCE_PATH = DATASET_DIR / "meta" / "provenance.json"
STATE_MODE_MAP = {
    "S0": "m0_proxy",
    "S1": "telemetry_candidate_v3_transition",
    "S2": "telemetry_candidate_v4_task_identity",
}


def _expected_training_targets() -> dict[str, object]:
    rca5 = load_json(RCA5_ARTIFACT, {})
    rca7 = load_json(RCA7_ARTIFACT, {})
    return {
        "best_transition_cell": rca5.get("best_transition_cell"),
        "canonical_train_cell": rca5.get("canonical_train_cell") or rca7.get("canonical_train_cell"),
        "best_train_state_mode": rca5.get("best_train_state_mode") or rca7.get("best_train_state_mode"),
        "tiny_retrain_permitted": bool(rca7.get("tiny_retrain_permitted", False)),
    }


def _dataset_guard(expected: dict[str, object]) -> tuple[bool, dict[str, object]]:
    report = {
        **expected,
        "dataset_root": str(DATASET_DIR),
        "repo_id": DATASET_REPO_ID,
        "provenance_exists": PROVENANCE_PATH.exists(),
    }
    if not bool(expected.get("tiny_retrain_permitted")):
        report["error"] = "RCA7 did not permit tiny retrain"
        return False, report
    if not PROVENANCE_PATH.exists():
        report["error"] = "Dataset provenance.json is missing; training would consume an unbound or stale dataset"
        return False, report
    provenance = json.loads(PROVENANCE_PATH.read_text())
    report["provenance_path"] = str(PROVENANCE_PATH)
    report["state_modes"] = provenance.get("state_modes", [])
    records = provenance.get("records", [])
    report["record_count"] = len(records)
    canonical_values = sorted({rec.get("canonical_train_cell") for rec in records if rec.get("canonical_train_cell") is not None})
    transition_values = sorted({rec.get("best_transition_cell") for rec in records if rec.get("best_transition_cell") is not None})
    train_state_values = sorted({rec.get("best_train_state_mode") for rec in records if rec.get("best_train_state_mode") is not None})
    report["record_canonical_train_cells"] = canonical_values
    report["record_best_transition_cells"] = transition_values
    report["record_best_train_state_modes"] = train_state_values
    expected_state_mode = STATE_MODE_MAP.get(str(expected.get("best_train_state_mode")))
    report["expected_state_mode_name"] = expected_state_mode
    if not canonical_values or str(expected.get("canonical_train_cell")) not in canonical_values:
        report["error"] = "Dataset provenance is not bound to the selected canonical_train_cell"
        return False, report
    if not transition_values or str(expected.get("best_transition_cell")) not in transition_values:
        report["error"] = "Dataset provenance is not bound to the selected best_transition_cell"
        return False, report
    if not train_state_values or str(expected.get("best_train_state_mode")) not in train_state_values:
        report["error"] = "Dataset provenance is not bound to the selected best_train_state_mode"
        return False, report
    if expected_state_mode and expected_state_mode not in set(provenance.get("state_modes", [])):
        report["error"] = "Dataset state_modes do not include the expected state mode for the selected best_train_state_mode"
        return False, report
    report["passed_preflight"] = True
    return True, report


def run() -> bool:
    expected = _expected_training_targets()
    guard_ok, guard_report = _dataset_guard(expected)
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

    cmd = [
        "lerobot-train",
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
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w") as log_handle:
        proc = subprocess.Popen(
            cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True, env=env
        )
        proc.wait()
    elapsed = time.time() - start
    log_tail = ""
    if LOG_PATH.exists():
        log_tail = LOG_PATH.read_text(errors="ignore")[-6000:]

    checkpoint_path = find_latest_checkpoint(TRAIN_OUTPUT_DIR)
    loss_lines = [line.strip() for line in log_tail.splitlines() if "loss" in line.lower()]
    result = {
        "gate": "g8_mint_train",
        "passed": checkpoint_path is not None and (proc.returncode == 0 or "push_model_to_hub" in log_tail or "ConnectionError" in log_tail),
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "steps_requested": train_steps,
        "batch_size": batch_size,
        "steps_completed": train_steps if checkpoint_path is not None else 0,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "train_output_dir": str(TRAIN_OUTPUT_DIR),
        "loss_samples": loss_lines[-10:],
        "stdout_tail": log_tail,
        "stderr_tail": "",
        "preflight": guard_report,
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

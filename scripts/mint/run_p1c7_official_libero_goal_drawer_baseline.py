#!/usr/bin/env python3
"""Run the official LIBERO drawer baseline through lerobot-eval."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mujoco_pilot_common import MINT_REPO_ROOT, OUTPUT_ROOT, PROJECT_ROOT, ensure_dir, now_iso, write_json

ARTIFACT_ROOT = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1" / "artifacts"
ARTIFACT_PATH = ARTIFACT_ROOT / "p1c7_official_libero_goal_drawer_baseline.json"
OUTPUT_DIR = OUTPUT_ROOT / "p1c7_official_libero_goal_drawer_baseline"
EVAL_INFO_PATH = OUTPUT_DIR / "eval_info.json"
MINT_CKPT = MINT_REPO_ROOT / "checkpoints" / "MINT-libero"
MINT_TOKENIZER = MINT_REPO_ROOT / "checkpoints" / "MINT-tokenizer-libero"
LIBERO_ROOT = PROJECT_ROOT / "external" / "LIBERO"
MINT_POLICY_SRC = MINT_REPO_ROOT / "lerobot_policy_mint" / "src"
AUTHORITATIVE_MINT_PYTHON = Path("/root/anaconda3/envs/mint/bin/python")
AUTHORITATIVE_ENTRYPOINT = PROJECT_ROOT / "scripts" / "mint" / "run_p1c10_release_runtime_matched_ab.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--n-action-steps", type=int, default=4)
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        shutil.which("lerobot-eval") or "lerobot-eval",
        f"--policy.path={MINT_CKPT}",
        f"--policy.vqvae_name_or_path={MINT_TOKENIZER}",
        "--env.type=libero",
        f"--env.task={args.task_suite}",
        f"--env.task_ids=[{args.task_id}]",
        f"--env.observation_height={args.height}",
        f"--env.observation_width={args.width}",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--policy.device={args.device}",
        "--eval.batch_size=1",
        f"--eval.n_episodes={args.n_episodes}",
        f"--seed={args.seed}",
        f"--output_dir={OUTPUT_DIR}",
    ]


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["MUJOCO_GL"] = "osmesa"
    env["PYOPENGL_PLATFORM"] = "osmesa"
    extra_paths = [str(LIBERO_ROOT), str(MINT_POLICY_SRC)]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(extra_paths + ([existing] if existing else []))
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def load_eval_info() -> dict[str, Any] | None:
    if not EVAL_INFO_PATH.exists():
        return None
    return json.loads(EVAL_INFO_PATH.read_text())


def _failure_class(*, returncode: int, stderr_tail: str, authoritative_python_matches_expected: bool) -> str:
    if returncode == 0:
        return "wrapper_path_green"
    if "unrecognized arguments: --env.task_ids" in stderr_tail:
        return "cli_contract_drift"
    if not authoritative_python_matches_expected:
        return "environment_selection_drift"
    return "wrapper_path_red"


def main() -> int:
    args = parse_args()
    ensure_dir(ARTIFACT_PATH.parent)
    ensure_dir(OUTPUT_DIR.parent)

    command = build_command(args)
    env = build_env()
    if EVAL_INFO_PATH.exists():
        EVAL_INFO_PATH.unlink()
    started = time.time()
    proc = None
    launch_error = None
    try:
        proc = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
    except OSError as exc:  # noqa: BLE001
        launch_error = repr(exc)
    duration_s = time.time() - started
    eval_info = load_eval_info() if proc is not None and proc.returncode == 0 else None
    authoritative_python_expected = str(AUTHORITATIVE_MINT_PYTHON)
    authoritative_python_used = sys.executable
    authoritative_python_matches_expected = authoritative_python_used == authoritative_python_expected
    stdout_tail = proc.stdout[-4000:] if proc is not None else ""
    stderr_tail = proc.stderr[-4000:] if proc is not None else (launch_error or "")
    payload = {
        "stage": "p1c7",
        "generated_at": now_iso(),
        "verdict_scope": "diagnostic_only",
        "authoritative_override_allowed": False,
        "authoritative_baseline_entrypoint": str(AUTHORITATIVE_ENTRYPOINT),
        "authoritative_python_expected": authoritative_python_expected,
        "authoritative_python_used": authoritative_python_used,
        "authoritative_python_matches_expected": authoritative_python_matches_expected,
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "n_episodes": args.n_episodes,
        "seed": args.seed,
        "command": command,
        "cwd": str(PROJECT_ROOT),
        "runtime": {
            "python_executable": sys.executable,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "MUJOCO_GL": env["MUJOCO_GL"],
            "PYOPENGL_PLATFORM": env["PYOPENGL_PLATFORM"],
            "TOKENIZERS_PARALLELISM": env["TOKENIZERS_PARALLELISM"],
        },
        "returncode": proc.returncode if proc is not None else None,
        "duration_s": duration_s,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "eval_info_path": str(EVAL_INFO_PATH) if EVAL_INFO_PATH.exists() else None,
        "eval_info": eval_info,
        "launch_error": launch_error,
        "passed": proc is not None and proc.returncode == 0,
        "failure_class": _failure_class(
            returncode=proc.returncode if proc is not None else 1,
            stderr_tail=stderr_tail,
            authoritative_python_matches_expected=authoritative_python_matches_expected,
        ),
    }
    write_json(ARTIFACT_PATH, payload)
    print(f"[p1c7] wrote {ARTIFACT_PATH}")
    if eval_info is not None:
        print(f"[p1c7] overall={eval_info.get('overall')}")
    else:
        print("[p1c7] eval_info.json missing")
    return proc.returncode if proc is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())

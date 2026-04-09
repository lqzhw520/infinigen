#!/usr/bin/env python3
"""A/B compare upstream-release MINT code vs current patched MINT code on official LIBERO drawer eval."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from mujoco_pilot_common import MINT_REPO_ROOT, OUTPUT_ROOT, PROJECT_ROOT, ensure_dir, now_iso, write_json

ARTIFACT_ROOT = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1" / "artifacts"
ARTIFACT_PATH = ARTIFACT_ROOT / "p1c8_mint_release_vs_patched_ab.json"
OUTPUT_BASE = OUTPUT_ROOT / "p1c8_mint_release_vs_patched_ab"
MINT_CKPT = MINT_REPO_ROOT / "checkpoints" / "MINT-libero"
MINT_TOKENIZER = MINT_REPO_ROOT / "checkpoints" / "MINT-tokenizer-libero"
LIBERO_ROOT = PROJECT_ROOT / "external" / "LIBERO"
PATCHED_MINT_SRC = MINT_REPO_ROOT / "lerobot_policy_mint" / "src"
UPSTREAM_COMMIT = "4eab5795345721001c412ff1ca2c886a11eab606"
UPSTREAM_WORKTREE = OUTPUT_BASE / "mint_upstream_4eab579"


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)


def ensure_upstream_worktree() -> dict[str, Any]:
    ensure_dir(OUTPUT_BASE)
    if not UPSTREAM_WORKTREE.exists():
        proc = run(
            ["git", "worktree", "add", "--detach", str(UPSTREAM_WORKTREE), UPSTREAM_COMMIT],
            cwd=MINT_REPO_ROOT,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git worktree add failed: {proc.stderr}")
    head = run(["git", "rev-parse", "HEAD"], cwd=UPSTREAM_WORKTREE)
    return {
        "path": str(UPSTREAM_WORKTREE),
        "head": head.stdout.strip(),
    }


def build_env(mint_src: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["MUJOCO_GL"] = "osmesa"
    env["PYOPENGL_PLATFORM"] = "osmesa"
    env["TOKENIZERS_PARALLELISM"] = "false"
    extra_paths = [str(LIBERO_ROOT), str(mint_src)]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(extra_paths + ([existing] if existing else []))
    return env


def build_command(output_dir: Path) -> list[str]:
    return [
        shutil.which("lerobot-eval") or "lerobot-eval",
        f"--policy.path={MINT_CKPT}",
        f"--policy.vqvae_name_or_path={MINT_TOKENIZER}",
        "--env.type=libero",
        "--env.task=libero_goal",
        "--env.task_ids=[0]",
        "--env.observation_height=256",
        "--env.observation_width=256",
        "--policy.n_action_steps=4",
        "--policy.device=cuda",
        "--eval.batch_size=1",
        "--eval.n_episodes=3",
        "--seed=42",
        f"--output_dir={output_dir}",
    ]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def run_variant(*, variant: str, mint_src: Path) -> dict[str, Any]:
    output_dir = OUTPUT_BASE / variant
    ensure_dir(output_dir.parent)
    env = build_env(mint_src)
    command = build_command(output_dir)
    started = time.time()
    proc = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True)
    duration_s = time.time() - started
    eval_info = read_json(output_dir / "eval_info.json")
    return {
        "variant": variant,
        "mint_src": str(mint_src),
        "command": command,
        "runtime": {
            "MUJOCO_GL": env["MUJOCO_GL"],
            "PYOPENGL_PLATFORM": env["PYOPENGL_PLATFORM"],
            "TOKENIZERS_PARALLELISM": env["TOKENIZERS_PARALLELISM"],
            "PYTHONPATH_prefix": env["PYTHONPATH"].split(":")[:2],
        },
        "returncode": proc.returncode,
        "duration_s": duration_s,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "eval_info_path": str(output_dir / "eval_info.json") if (output_dir / "eval_info.json").exists() else None,
        "eval_info": eval_info,
    }


def summarize_success(result: dict[str, Any]) -> float | None:
    eval_info = result.get("eval_info")
    if not eval_info:
        return None
    return eval_info.get("overall", {}).get("pc_success")


def main() -> int:
    ensure_dir(ARTIFACT_PATH.parent)
    upstream = ensure_upstream_worktree()
    current_head = run(["git", "rev-parse", "HEAD"], cwd=MINT_REPO_ROOT)
    variants = [
        {
            "name": "upstream_release_4eab579",
            "src": UPSTREAM_WORKTREE / "lerobot_policy_mint" / "src",
            "head": upstream["head"],
        },
        {
            "name": "current_patched_b5eabd4",
            "src": PATCHED_MINT_SRC,
            "head": current_head.stdout.strip(),
        },
    ]
    results = []
    for spec in variants:
        results.append(run_variant(variant=spec["name"], mint_src=spec["src"]))
    payload = {
        "stage": "p1c8",
        "generated_at": now_iso(),
        "checkpoint": str(MINT_CKPT),
        "tokenizer": str(MINT_TOKENIZER),
        "task_suite": "libero_goal",
        "task_id": 0,
        "n_episodes": 3,
        "seed": 42,
        "variants": [
            {"name": spec["name"], "head": spec["head"], "src": str(spec["src"])} for spec in variants
        ],
        "results": results,
        "comparison": {
            "upstream_pc_success": summarize_success(results[0]),
            "patched_pc_success": summarize_success(results[1]),
            "same_checkpoint": True,
            "same_task": True,
            "same_cli": True,
            "differs_only_in_mint_code_path": True,
        },
        "passed": all(result["returncode"] == 0 for result in results),
    }
    write_json(ARTIFACT_PATH, payload)
    print(f"[p1c8] wrote {ARTIFACT_PATH}")
    print(json.dumps(payload["comparison"], indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

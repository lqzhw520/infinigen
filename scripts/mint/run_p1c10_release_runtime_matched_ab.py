#!/usr/bin/env python3
"""Run a clean A/B between upstream and patched MINT code on the release-era runtime."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from mujoco_pilot_common import MINT_REPO_ROOT, OUTPUT_ROOT, PROJECT_ROOT, ensure_dir, now_iso, write_json

ARTIFACT_ROOT = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1" / "artifacts"
ARTIFACT_PATH = ARTIFACT_ROOT / "p1c10_release_runtime_matched_ab.json"
OUTPUT_BASE = OUTPUT_ROOT / "p1c10_release_runtime_matched_ab"
MINT_CKPT = MINT_REPO_ROOT / "checkpoints" / "MINT-libero"
MINT_TOKENIZER = MINT_REPO_ROOT / "checkpoints" / "MINT-tokenizer-libero"
MINT_TOKENIZER_CKPT = MINT_TOKENIZER / "ms_vqvae.pth"
LIBERO_ROOT = PROJECT_ROOT / "external" / "LIBERO"
PATCHED_MINT_SRC = MINT_REPO_ROOT / "lerobot_policy_mint" / "src"
UPSTREAM_COMMIT = "4eab5795345721001c412ff1ca2c886a11eab606"
UPSTREAM_WORKTREE = OUTPUT_BASE / "mint_upstream_4eab579"
TASK_SUITE = "libero_goal"
TASK_ID = 0
TASK_NAME = "open_the_middle_drawer_of_the_cabinet"
N_EPISODES = 3
SEED = 42
HEIGHT = 256
WIDTH = 256
N_ACTION_STEPS = 4
AUTHORITATIVE_MINT_PYTHON = Path("/root/anaconda3/envs/mint/bin/python")


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
    status = run(["git", "status", "--short"], cwd=UPSTREAM_WORKTREE)
    return {
        "path": str(UPSTREAM_WORKTREE),
        "head": head.stdout.strip(),
        "status": status.stdout.strip(),
    }


def git_head(repo_root: Path) -> str:
    proc = run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git rev-parse failed under {repo_root}")
    return proc.stdout.strip()


def git_status(repo_root: Path) -> str:
    proc = run(["git", "status", "--short"], cwd=repo_root)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git status failed under {repo_root}")
    return proc.stdout.strip()


def build_child_env(mint_src: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["MUJOCO_GL"] = "osmesa"
    env["PYOPENGL_PLATFORM"] = "osmesa"
    env["TOKENIZERS_PARALLELISM"] = "false"
    extra_paths = [str(mint_src), str(LIBERO_ROOT)]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(extra_paths + ([existing] if existing else []))
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-run", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--mint-src")
    parser.add_argument("--output-dir")
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def variant_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "eval_info": output_dir / "eval_info.json",
        "summary": output_dir / "variant_summary.json",
        "failure": output_dir / "variant_failure.json",
    }


def child_main(args: argparse.Namespace) -> int:
    if not args.label or not args.mint_src or not args.output_dir:
        raise SystemExit("--variant-run requires --label, --mint-src, and --output-dir")

    mint_src = Path(args.mint_src).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    paths = variant_output_paths(output_dir)

    started = time.time()
    payload: dict[str, Any] = {
        "label": args.label,
        "generated_at": now_iso(),
        "mint_src": str(mint_src),
        "runtime_env": {
            "python_executable": sys.executable,
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
            "PYTHONPATH_prefix": os.environ.get("PYTHONPATH", "").split(":")[:4],
        },
        "experiment": {
            "checkpoint": str(MINT_CKPT),
            "tokenizer": str(MINT_TOKENIZER),
            "tokenizer_checkpoint": str(MINT_TOKENIZER_CKPT),
            "task_suite": TASK_SUITE,
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "n_episodes": int(args.n_episodes),
            "seed": int(args.seed),
            "observation_height": HEIGHT,
            "observation_width": WIDTH,
            "n_action_steps": N_ACTION_STEPS,
        },
    }

    try:
        import gymnasium as gym
        import lerobot
        import torch
        import transformers
        import lerobot_policy_mint
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.envs.libero import create_libero_envs
        from lerobot.envs.utils import close_envs
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.scripts.lerobot_eval import eval_policy_all

        payload["runtime_versions"] = {
            "python": platform.python_version(),
            "lerobot": getattr(lerobot, "__version__", "unknown"),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
        payload["code"] = {
            "lerobot_policy_mint_init": inspect.getsourcefile(lerobot_policy_mint),
            "lerobot_policy_mint_package": str(Path(inspect.getsourcefile(lerobot_policy_mint) or "").resolve()),
        }

        cfg = PreTrainedConfig.from_pretrained(str(MINT_CKPT))
        payload["config_pretrained_path_before_override"] = str(getattr(cfg, "pretrained_path", ""))
        cfg.pretrained_path = str(MINT_CKPT)
        cfg.vqvae_name_or_path = str(MINT_TOKENIZER_CKPT)
        cfg.n_action_steps = N_ACTION_STEPS
        cfg.device = "cuda"
        payload["config_pretrained_path_after_override"] = str(getattr(cfg, "pretrained_path", ""))

        env_cfg = LiberoEnvConfig(
            task=TASK_SUITE,
            observation_height=HEIGHT,
            observation_width=WIDTH,
            control_mode="relative",
            camera_name="agentview_image,robot0_eye_in_hand_image",
        )
        envs = create_libero_envs(
            task=TASK_SUITE,
            n_envs=1,
            gym_kwargs={
                "task_ids": [TASK_ID],
                "obs_type": env_cfg.obs_type,
                "render_mode": env_cfg.render_mode,
                "camera_name_mapping": env_cfg.camera_name_mapping,
                "observation_height": env_cfg.observation_height,
                "observation_width": env_cfg.observation_width,
            },
            camera_name=env_cfg.camera_name,
            init_states=env_cfg.init_states,
            env_cls=gym.vector.SyncVectorEnv,
            control_mode=env_cfg.control_mode,
            episode_length=env_cfg.episode_length,
        )

        policy = make_policy(cfg, env_cfg=env_cfg)
        preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=str(MINT_CKPT))
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, cfg)

        first_env = envs[TASK_SUITE][TASK_ID].envs[0]
        payload["task_binding"] = {
            "task_description": getattr(first_env, "task_description", None),
            "task_attr": getattr(first_env, "task", None),
            "max_episode_steps": getattr(first_env, "_max_episode_steps", None),
        }
        payload["code"].update(
            {
                "policy_class": type(policy).__name__,
                "policy_source": inspect.getsourcefile(type(policy)),
                "config_class": type(cfg).__name__,
                "config_source": inspect.getsourcefile(type(cfg)),
            }
        )

        eval_info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=int(args.n_episodes),
            max_episodes_rendered=int(args.n_episodes),
            videos_dir=output_dir / "videos" / f"{TASK_SUITE}_{TASK_ID}",
            return_episode_data=False,
            start_seed=int(args.seed),
            max_parallel_tasks=1,
        )
        close_envs(envs)

        payload["eval_info"] = eval_info
        payload["returncode"] = 0
        payload["duration_s"] = time.time() - started
        write_json(paths["eval_info"], eval_info)
        write_json(paths["summary"], payload)
        print(json.dumps({"label": args.label, "returncode": 0, "eval_info_path": str(paths["eval_info"])}))
        return 0
    except Exception as exc:  # pragma: no cover - debugging path
        payload["returncode"] = 1
        payload["duration_s"] = time.time() - started
        payload["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(paths["failure"], payload)
        print(json.dumps({"label": args.label, "returncode": 1, "failure_path": str(paths["failure"])}))
        return 1


def run_variant(*, label: str, mint_src: Path, n_episodes: int = N_EPISODES, seed: int = SEED) -> dict[str, Any]:
    output_dir = OUTPUT_BASE / label
    ensure_dir(output_dir)
    env = build_child_env(mint_src)
    if not AUTHORITATIVE_MINT_PYTHON.exists():
        raise RuntimeError(f"Authoritative MINT python missing: {AUTHORITATIVE_MINT_PYTHON}")
    command = [
        str(AUTHORITATIVE_MINT_PYTHON),
        str(Path(__file__).resolve()),
        "--variant-run",
        "--label",
        label,
        "--mint-src",
        str(mint_src),
        "--output-dir",
        str(output_dir),
        "--n-episodes",
        str(int(n_episodes)),
        "--seed",
        str(int(seed)),
    ]
    started = time.time()
    proc = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True)
    duration_s = time.time() - started
    paths = variant_output_paths(output_dir)
    return {
        "label": label,
        "mint_src": str(mint_src),
        "command": command,
        "authoritative_python_expected": str(AUTHORITATIVE_MINT_PYTHON),
        "authoritative_python_used": command[0],
        "authoritative_python_matches_expected": command[0] == str(AUTHORITATIVE_MINT_PYTHON),
        "returncode": proc.returncode,
        "duration_s": duration_s,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "summary_path": str(paths["summary"]) if paths["summary"].exists() else None,
        "failure_path": str(paths["failure"]) if paths["failure"].exists() else None,
        "eval_info_path": str(paths["eval_info"]) if paths["eval_info"].exists() else None,
        "summary": read_json(paths["summary"]),
        "failure": read_json(paths["failure"]),
        "eval_info": read_json(paths["eval_info"]),
    }


def summarize_pc_success(result: dict[str, Any]) -> float | None:
    eval_info = result.get("eval_info")
    if not eval_info:
        return None
    overall = eval_info.get("overall", {})
    return overall.get("pc_success")


def parent_main() -> int:
    ensure_dir(ARTIFACT_PATH.parent)
    upstream = ensure_upstream_worktree()
    patched_head = git_head(MINT_REPO_ROOT)
    patched_status = git_status(MINT_REPO_ROOT)
    variants = [
        {
            "label": "upstream_release_4eab579_release_runtime",
            "mint_src": UPSTREAM_WORKTREE / "lerobot_policy_mint" / "src",
            "head": upstream["head"],
            "status": upstream["status"],
        },
        {
            "label": f"current_vendor_{patched_head[:7]}_release_runtime",
            "mint_src": PATCHED_MINT_SRC,
            "head": patched_head,
            "status": patched_status,
        },
    ]

    results = [run_variant(label=spec["label"], mint_src=spec["mint_src"]) for spec in variants]

    payload = {
        "stage": "p1c10",
        "generated_at": now_iso(),
        "goal": "match release-era lerobot runtime and compare upstream vs patched MINT code on the same official LIBERO drawer task",
        "runtime_target": {
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "python_executable": sys.executable,
            "authoritative_python_expected": str(AUTHORITATIVE_MINT_PYTHON),
        },
        "fixed_conditions": {
            "checkpoint": str(MINT_CKPT),
            "tokenizer": str(MINT_TOKENIZER),
            "tokenizer_checkpoint": str(MINT_TOKENIZER_CKPT),
            "task_suite": TASK_SUITE,
            "task_id": TASK_ID,
            "task_name": TASK_NAME,
            "n_episodes": N_EPISODES,
            "seed": SEED,
            "observation_height": HEIGHT,
            "observation_width": WIDTH,
            "n_action_steps": N_ACTION_STEPS,
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "TOKENIZERS_PARALLELISM": "false",
            "evaluation_path": "official lerobot Python eval pipeline (create_libero_envs + eval_policy_all)",
        },
        "variants": [
            {
                "label": spec["label"],
                "mint_src": str(spec["mint_src"]),
                "mint_git_head": spec["head"],
                "mint_git_status": spec["status"],
            }
            for spec in variants
        ],
        "results": results,
        "comparison": {
            "upstream_returncode": results[0]["returncode"],
            "patched_returncode": results[1]["returncode"],
            "upstream_pc_success": summarize_pc_success(results[0]),
            "patched_pc_success": summarize_pc_success(results[1]),
            "same_checkpoint": True,
            "same_tokenizer": True,
            "same_task": True,
            "same_release_runtime": True,
            "same_eval_pipeline": True,
            "differs_only_in_mint_code_path": True,
        },
        "passed": all(result["returncode"] == 0 for result in results),
    }
    write_json(ARTIFACT_PATH, payload)
    print(f"[p1c10] wrote {ARTIFACT_PATH}")
    print(json.dumps(payload["comparison"], indent=2))
    return 0 if payload["passed"] else 1


def main() -> int:
    args = parse_args()
    if args.variant_run:
        return child_main(args)
    return parent_main()


if __name__ == "__main__":
    raise SystemExit(main())

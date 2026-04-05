#!/usr/bin/env python3
"""
Render D1/V57 comparison videos for advisor report.

This script runs pretrained vs finetuned MINT policy rollouts on seed 2 (train)
and seeds 11-15 (held-out), generates side-by-side annotated MP4 videos,
and outputs a summary CSV.

Usage (must run from scripts/mint/ directory with `mint` conda env):
    # Preview mode (render pretrained + finetuned + random on seed 2):
    cd scripts/mint && conda activate mint && python render_v57_comparison.py --preview

    # Full E1 eval mode (seeds 11-15, all 3 policies):
    cd scripts/mint && conda activate mint && python render_v57_comparison.py --full_eval --finetuned_path PATH

    # Quick single-seed demo:
    cd scripts/mint && conda activate mint && python render_v57_comparison.py --seed 2 --policy pretrained
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Apply compatibility patches BEFORE importing lerobot/MINT.
from mint_eval_patches import apply as _apply_patches

_apply_patches()

# ---- Local imports (scripts/mint/ must be cwd) ----
from drawer_robot_env import DrawerRobotEnv
from mint_common import (
    DATASET_DIR,
    DATASET_REPO_ID,
    DEFAULT_HELD_OUT_SEEDS,
)
from render_rollout_video import render_trace_video

# ---- MINT policy loading ----
MINT_CKPT = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero"


def _load_policy(path: str, dataset_root: Path, repo_id: str):
    """Load a MINT policy with its pre/post processors."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot_policy_mint.modeling_mint import MINTPolicy

    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, revision="main")
    stats = dataset.meta.stats
    if stats is None:
        stats = dataset.meta if isinstance(dataset.meta, dict) else {}

    policy = MINTPolicy.from_pretrained(
        path, local_files_only=True, dataset_stats=stats
    )
    policy.eval()

    if hasattr(policy.model, "direct_grip_head"):
        model_dtype = next(
            iter(policy.model.paligemma_with_expert.gemma_expert.model.parameters())
        ).dtype
        policy.model.direct_grip_head = policy.model.direct_grip_head.to(dtype=model_dtype)

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=path, dataset_stats=stats
    )
    return policy, preprocessor, postprocessor


def _obs_to_batch(obs):
    """Convert a DrawerRobotEnv observation to LeRobot batch format."""
    import torch
    return {
        "observation.images.image": (
            torch.from_numpy(obs.image).permute(2, 0, 1).to(torch.float32) / 255.0
        ),
        "observation.images.image2": (
            torch.from_numpy(obs.image2).permute(2, 0, 1).to(torch.float32) / 255.0
        ),
        "observation.state": torch.from_numpy(obs.state.astype(np.float32)),
        "task": obs.task,
    }


def random_policy(rng: np.random.Generator) -> np.ndarray:
    """Random policy: uniform action + discrete gripper."""
    action = rng.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)
    action[6] = float(rng.choice([-1.0, 1.0]))
    return action


def rollout_for_video(
    seed: int,
    policy_kind: str,
    policy_bundle,
    *,
    rng_seed: int = 0,
    max_steps: int = 96,
    label: str = "eval",
) -> dict:
    """Run a policy rollout, capture frames and metrics, return record dict."""
    import torch

    env = DrawerRobotEnv(seed=seed, image_size=224, max_steps=max_steps)
    obs = env.reset()
    rng = np.random.default_rng(rng_seed + seed)

    frames, frames2 = [], []
    drawer_trace, attached_trace = [], []
    eef_pos_trace = []
    raw_action_trace = []
    grasp_success = False

    for _ in range(1, max_steps + 1):
        frames.append(obs.image.copy())
        frames2.append(obs.image2.copy())
        eef_pos_trace.append(
            obs.eef_pos.copy() if hasattr(obs, "eef_pos") else obs.state[:3].copy()
        )

        if policy_kind == "random":
            action = random_policy(rng)
        else:
            policy, pre, post = policy_bundle
            batch = _obs_to_batch(obs)
            processed = pre(batch)
            with torch.inference_mode():
                action = policy.select_action(processed)
            action = post(action).squeeze(0).detach().cpu().numpy().astype(np.float32)

        raw_action_trace.append(action.copy() if action is not None else None)
        obs, _, done, info = env.step(action)
        drawer_trace.append(float(info["drawer_fraction"]))
        attached_trace.append(bool(info.get("attached", False)))
        grasp_success = grasp_success or bool(info.get("ever_attached", False))
        if done:
            break

    # Compute metrics
    pull_distance = max(drawer_trace) if drawer_trace else 0.0
    valid_actions = [a for a in raw_action_trace if a is not None]
    non_zero_ratio = (
        float(np.mean([float(np.abs(a).max() > 0.01) for a in valid_actions]))
        if valid_actions
        else 0.0
    )
    eef_arr = np.array(eef_pos_trace)
    step_deltas = (
        np.linalg.norm(np.diff(eef_arr, axis=0), axis=1)
        if len(eef_arr) > 1
        else np.array([0.0])
    )
    total_eef_motion = float(np.sum(step_deltas))

    # Strict success
    drawer_arr = np.array(drawer_trace, dtype=np.float32)
    attached_arr = np.array(attached_trace, dtype=bool)
    strict = _evaluate_strict_success(drawer_arr, attached_arr)

    return {
        "seed": seed,
        "policy": policy_kind,
        "label": label,
        "grasp_success": grasp_success,
        "success": bool(strict["strict_success"]),
        "steps": len(frames),
        "pull_distance": float(pull_distance),
        "total_eef_motion": total_eef_motion,
        "non_zero_action_ratio": non_zero_ratio,
        "strict": strict,
        "frames": frames,
        "frames2": frames2,
        "drawer_trace": drawer_trace,
        "attached_trace": attached_trace,
    }


def _evaluate_strict_success(drawer_arr, attached_arr):
    """Evaluate strict success criteria (same as strict_success.py)."""
    drawer_open_threshold = 0.9
    min_attach_persistence = 3
    min_post_attach_delta = 0.35

    if len(drawer_arr) == 0 or len(attached_arr) == 0:
        return {"strict_success": False}

    ever_attached = bool(np.any(attached_arr))
    first_attach = int(np.argmax(attached_arr)) if ever_attached else -1
    attach_persistence = 0
    if ever_attached and first_attach >= 0:
        for i in range(first_attach, len(attached_arr)):
            if attached_arr[i]:
                attach_persistence += 1
            else:
                break

    post_attach_drawer_delta = 0.0
    if ever_attached and first_attach >= 0:
        post_vals = drawer_arr[first_attach:]
        if len(post_vals) > 1:
            post_attach_drawer_delta = float(np.max(post_vals) - np.min(post_vals))

    max_drawer_fraction = float(np.max(drawer_arr))
    drawer_open = max_drawer_fraction >= drawer_open_threshold

    return {
        "strict_success": bool(
            ever_attached
            and attach_persistence >= min_attach_persistence
            and post_attach_drawer_delta >= min_post_attach_delta
            and drawer_open
        ),
        "drawer_open": drawer_open,
        "ever_attached": ever_attached,
        "attach_persistence": attach_persistence,
        "max_drawer_fraction": max_drawer_fraction,
        "post_attach_drawer_delta": post_attach_drawer_delta,
    }


def render_record_to_video(record: dict, output_path: Path, fps: int = 10):
    """Render a rollout record to MP4."""
    render_trace_video(
        images=record["frames"],
        images2=record["frames2"],
        output_path=output_path,
        stage=record["label"],
        policy=record["policy"],
        seed=record["seed"],
        attempt="eval",
        variant="d1_v57",
        drawer_trace=record["drawer_trace"],
        attached_trace=record["attached_trace"],
        strict_success=record["success"],
        phase_labels=[record["label"]] * len(record["frames"]),
        extra_line=f"eef_motion={record['total_eef_motion']:.2f}m | pull={record['pull_distance']:.3f}",
        fps=fps,
    )


def main():
    parser = argparse.ArgumentParser(description="Render D1/V57 comparison videos")
    parser.add_argument(
        "--finetuned_path",
        type=str,
        default=(
            "/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/"
            "outputs_d1_seed_002_episode_01_multi_episode_overfit_1774878439_c895350f/"
            "checkpoints/003000/pretrained_model"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/videos",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Run on specific seed only"
    )
    parser.add_argument(
        "--policy", type=str, default=None,
        choices=["pretrained", "finetuned", "random"],
        help="Run specific policy only"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Preview mode: pretrained+finetuned+random on seed 2 (~15min)"
    )
    parser.add_argument(
        "--full_eval", action="store_true",
        help="Full E1 eval: seeds 11-15, all 3 policies (~1h)"
    )
    parser.add_argument("--max_steps", type=int, default=96)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"[{ts}] D1/V57 Video Render")
    print(f"{'='*60}")
    print(f"  Finetuned path: {args.finetuned_path}")
    print(f"  Dataset repo:   {DATASET_REPO_ID}")
    print(f"  Dataset dir:    {DATASET_DIR}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Max steps:     {args.max_steps}")
    print(f"  FPS:           {args.fps}")

    # Load policies
    print("\n  Loading pretrained MINT...")
    pretrained_bundle = _load_policy(MINT_CKPT, DATASET_DIR, DATASET_REPO_ID)
    print("  Loading finetuned MINT (V57)...")
    finetuned_bundle = _load_policy(args.finetuned_path, DATASET_DIR, DATASET_REPO_ID)
    print("  Both policies loaded OK\n")

    records = []

    if args.preview:
        seeds = [2]
        policies = ["pretrained", "finetuned", "random"]
        print(f"  Mode: preview (seed=2, policies={'/'.join(policies)})\n")
    elif args.full_eval:
        seeds = list(range(11, 16))
        policies = ["pretrained", "finetuned", "random"]
        print(f"  Mode: full_eval (seeds=11-15, policies={'/'.join(policies)})\n")
    elif args.seed is not None:
        seeds = [args.seed]
        policies = [args.policy] if args.policy else ["pretrained", "finetuned"]
        print(f"  Mode: single (seed={seeds[0]}, policy={policies[0]})\n")
    else:
        seeds = [2] + list(range(11, 16))
        policies = ["pretrained", "finetuned"]
        print(f"  Mode: default (train seed=2 + held-out seeds 11-15, policies={'/'.join(policies)})\n")

    total_runs = len(seeds) * len(policies)
    run_idx = 0

    for seed in seeds:
        for pol in policies:
            run_idx += 1
            is_train = seed == 2
            label_base = "D1_train" if is_train else f"E1_heldout_s{seed}"
            rng_seed = 42
            bundle = None

            if pol == "pretrained":
                bundle = pretrained_bundle
                label = f"{label_base}_pretrained"
            elif pol == "finetuned":
                bundle = finetuned_bundle
                label = f"{label_base}_finetuned"
            else:
                label = f"{label_base}_random"

            print(
                f"  [{run_idx}/{total_runs}] Rendering: {label:<45s} "
                f"(policy={pol}, max_steps={args.max_steps})"
            )
            start = time.time()
            try:
                record = rollout_for_video(
                    seed=seed,
                    policy_kind=pol,
                    policy_bundle=bundle,
                    rng_seed=rng_seed,
                    max_steps=args.max_steps,
                    label=label,
                )
                elapsed = time.time() - start

                # Summary line
                status = "SUCCESS" if record["success"] else "FAIL"
                grasp = "GRASP" if record["grasp_success"] else "nograsp"
                print(
                    f"             => [{status}/{grasp}] "
                    f"eef={record['total_eef_motion']:.2f}m "
                    f"pull={record['pull_distance']:.3f} "
                    f"frames={record['steps']} "
                    f"time={elapsed:.0f}s"
                )

                # Save video
                video_name = f"{ts}_{label}_seed{seed}.mp4"
                video_path = output_dir / video_name
                render_record_to_video(record, video_path, fps=args.fps)
                print(f"             => Video: {video_path}")

                record["video_path"] = str(video_path)
                record_clean = {
                    k: v for k, v in record.items()
                    if k not in ("frames", "frames2")
                }
                records.append(record_clean)

            except Exception as ex:
                import traceback
                elapsed = time.time() - start
                print(f"             => ERROR: {ex} (after {elapsed:.0f}s)")
                traceback.print_exc()
                records.append({
                    "seed": seed,
                    "policy": pol,
                    "label": label,
                    "error": str(ex),
                    "video_path": None,
                    "success": False,
                    "grasp_success": False,
                    "total_eef_motion": 0.0,
                    "pull_distance": 0.0,
                    "steps": 0,
                })

    # Save summaries
    print(f"\n{'='*60}")
    print(f"Summary ({len(records)} runs)")

    csv_path = output_dir / f"{ts}_d1_v57_summary.csv"
    if records:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "seed", "policy", "label", "success", "grasp_success",
                    "total_eef_motion", "pull_distance", "steps", "video_path"
                ],
            )
            writer.writeheader()
            for r in records:
                writer.writerow({k: r.get(k) for k in writer.fieldnames})
        print(f"  CSV: {csv_path}")

    json_path = output_dir / f"{ts}_d1_v57_summary.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"  JSON: {json_path}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("Policy Comparison:")
    for pol in sorted(set(r["policy"] for r in records)):
        pol_records = [r for r in records if r["policy"] == pol]
        n = len(pol_records)
        successes = sum(1 for r in pol_records if r.get("success"))
        grasps = sum(1 for r in pol_records if r.get("grasp_success"))
        eef_mean = (
            sum(r["total_eef_motion"] for r in pol_records) / n
            if n > 0 else 0
        )
        print(
            f"  {pol:<20s}: success={successes}/{n} ({100*successes/max(n,1):.0f}%) "
            f"grasp={grasps}/{n} ({100*grasps/max(n,1):.0f}%) "
            f"eef_mean={eef_mean:.2f}m"
        )

    print(f"\n  Videos: {output_dir}")
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done.")


if __name__ == "__main__":
    main()

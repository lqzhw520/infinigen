#!/usr/bin/env python3
"""
P0b Drawer-Only Night Runner — Bounded Artifact Collection

BOUNDED TO RAW ARTIFACT COLLECTION ONLY.
DO NOT: upgrade sovereign verdict, change claims, mark RCA as resolved.

This script runs the full matched A/B experiment for P0b drawer-only:
  6 rollouts: 3 seeds × 2 policies × 96 steps × drawer-colored env
  E022 baseline (seeds 1-3, white images): 0/6 success

After each rollout:
  - Frames saved as PNG
  - MP4 generated (8fps, 12s)
  - Partial rollout_meta.json written
  - Progress logged

After all rollouts:
  - Aggregate results written to p0b_night_results.json
  - MP4s for all 6 rollouts available for review

Verdict upgrade requires human review of mp4s. This script only collects artifacts.

Rollout matrix:
  seed=1, pretrained → seed=1, finetuned
  seed=2, pretrained → seed=2, finetuned
  seed=3, pretrained → seed=3, finetuned

SKIP logic: if rollout directory already exists with rollout_meta.json, skip (idempotent).

Usage:
  python run_p0b_night_runner.py 2>&1 | tee logs/p0b_night_runner.log
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts/mint"))

CAMPAIGN_DIR = PROJECT_ROOT / "experiments/mint/mint_drawer_v1"
ARTIFACT_DIR = CAMPAIGN_DIR / "artifacts"
COLORED_DIR = ARTIFACT_DIR / "p0b_colored_rollouts"
VIDEO_DIR = COLORED_DIR / "videos"
LOG_DIR = COLORED_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
VIDEO_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"p0b_night_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("p0b_night")

SEEDS = [1, 2, 3]
POLICIES = ["pretrained", "finetuned"]
MAX_STEPS = 96
E022_SEEDS = [1, 2, 3]

E022_WHITE_RESULTS = {
    "pretrained": {"success": 0, "attached": 0, "total": 6},
    "finetuned": {"success": 0, "attached": 0, "total": 6},
}


def get_imageio():
    """Lazy import to avoid startup overhead."""
    import imageio
    return imageio


def generate_mp4(frame_dir: Path, video_path: Path, fps: int = 8) -> bool:
    """Generate mp4 from PNG frames. Returns True if successful."""
    try:
        frames = sorted(frame_dir.glob("frame_*.png"))
        if not frames:
            log.warning(f"No frames found in {frame_dir}")
            return False
        imageio = get_imageio()
        writer = imageio.get_writer(
            video_path, fps=fps, codec="libx264",
            quality=7, pixelformat="yuv420p",
        )
        for f in frames:
            img = imageio.imread(f)
            writer.append_data(img)
        writer.close()
        size_kb = video_path.stat().st_size / 1024
        log.info(f"  mp4: {video_path.name} ({len(frames)} frames, {size_kb:.0f} KB)")
        return True
    except Exception as e:
        log.error(f"  mp4 FAILED: {e}")
        return False


def rollout_exists(seed: int, policy: str) -> bool:
    """Check if rollout already completed."""
    rollout_id = f"seed_{seed:03d}_{policy}_drawer_colored"
    rollout_dir = COLORED_DIR / rollout_id
    meta_path = rollout_dir / "rollout_meta.json"
    return meta_path.exists()


def run_single_rollout(seed: int, policy: str, max_steps: int) -> dict:
    """Run one rollout. Returns result dict or error dict."""
    rollout_id = f"seed_{seed:03d}_{policy}_drawer_colored"
    rollout_dir = COLORED_DIR / rollout_id
    meta_path = rollout_dir / "rollout_meta.json"
    video_path = VIDEO_DIR / f"{rollout_id}.mp4"

    # Idempotent skip
    if meta_path.exists():
        log.info(f"SKIP {rollout_id} (already exists)")
        result = json.load(open(meta_path))
        result["skipped"] = True
        return result

    log.info(f"START {rollout_id} (seed={seed}, policy={policy}, steps={max_steps})")
    t0 = time.time()

    try:
        # Run rollout inline (same as run_p0b_colored_reroll.py)
        import torch
        from PIL import Image

        from drawer_robot_env import DrawerRobotEnv
        from mint_common import DATASET_DIR, DATASET_REPO_ID, load_json
        from evaluate_mint_drawer_campaign import _load_policy, _obs_to_batch
        from mint_eval_patches import apply as _apply_patches

        _apply_patches()

        # Find checkpoint
        from run_p0b_colored_reroll import find_checkpoint
        ckpt = find_checkpoint(policy)
        if ckpt is None:
            return {"seed": seed, "policy": policy, "error": f"Checkpoint not found: {policy}"}

        # Load policy
        policy_bundle = _load_policy(ckpt, DATASET_DIR, DATASET_REPO_ID)
        policy_obj, preprocessor, postprocessor = policy_bundle

        ACTION_CONTRACT = load_json(CAMPAIGN_DIR / "artifacts/active_action_contract.json", {})

        # Create colored env
        from run_p0b_colored_reroll import apply_colored_drawer
        env = DrawerRobotEnv(
            seed=seed,
            image_size=224,
            max_steps=max_steps,
            action_contract=ACTION_CONTRACT,
        )
        obs = env.reset()
        apply_colored_drawer(env)

        # Rollout loop
        drawer_trace = []
        attached_trace = []
        frames = []
        eef_pos_trace = []
        raw_action_trace = []

        for step in range(1, max_steps + 1):
            frames.append(obs.image.copy())
            eef_pos_trace.append(obs.state[:3].copy())

            if policy == "random":
                action = np.random.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)
                action[6] = float(np.random.choice([-1.0, 1.0]))
            else:
                batch = _obs_to_batch(obs)
                processed = preprocessor(batch)
                with torch.inference_mode():
                    action = policy_obj.select_action(processed)
                action = postprocessor(action)
                action = action.squeeze(0).detach().cpu().numpy().astype(np.float32)

            raw_action_trace.append(action.copy() if action is not None else None)
            obs, _, done, info = env.step(action)
            drawer_trace.append(float(info["drawer_fraction"]))
            attached_trace.append(bool(info.get("attached")))
            if done:
                break

        env.close()
        elapsed = time.time() - t0

        # Save frames
        rollout_dir.mkdir(exist_ok=True)
        image_paths = []
        frame_rgb_stats = []
        for i, img in enumerate(frames):
            img_path = rollout_dir / f"frame_{i:04d}.png"
            Image.fromarray(img).save(img_path)
            image_paths.append(str(img_path))
            if i < 5 or i % 20 == 0:
                rgb = img[:, :, :3].astype(float)
                white_mask = np.all(rgb > 245, axis=2)
                non_white_pct = (~white_mask).sum() / (224 * 224) * 100
                rgb_std = float(rgb.std())
                rgb_mean = float(rgb.mean())
                frame_rgb_stats.append({
                    "rgb_mean": rgb_mean, "rgb_std": rgb_std,
                    "non_white_pct": non_white_pct,
                    "image_is_colored": bool(non_white_pct > 0.5 and rgb_std > 10.0),
                })

        # Metrics
        max_drawer = max(drawer_trace) if drawer_trace else 0.0
        final_drawer = drawer_trace[-1] if drawer_trace else 0.0
        ever_attached = any(attached_trace)
        non_zero_actions = sum(
            1 for a in raw_action_trace
            if a is not None and np.abs(a).max() > 1e-6
        )
        total_eef = 0.0
        for i in range(1, len(eef_pos_trace)):
            total_eef += float(np.linalg.norm(
                np.array(eef_pos_trace[i]) - np.array(eef_pos_trace[i-1])
            ))

        from strict_success import evaluate_strict_success
        strict = evaluate_strict_success(
            np.asarray(drawer_trace, dtype=np.float32),
            np.asarray(attached_trace, dtype=bool),
        )

        colored_count = sum(1 for s in frame_rgb_stats if s["image_is_colored"])
        mean_std = float(np.mean([s["rgb_std"] for s in frame_rgb_stats])) if frame_rgb_stats else 0.0
        mean_non_white = float(np.mean([s["non_white_pct"] for s in frame_rgb_stats])) if frame_rgb_stats else 0.0

        color_check = {
            "rgb_std": mean_std,
            "non_white_pct": mean_non_white,
            "image_is_colored": bool(colored_count >= max(1, len(frame_rgb_stats) // 2)),
            "colored_frames_analyzed": len(frame_rgb_stats),
            "colored_frames_among_analyzed": colored_count,
            "note": "Verified from actual rollout frames",
        }

        result = {
            "rollout_id": rollout_id,
            "seed": seed,
            "policy": policy,
            "checkpoint": ckpt,
            "max_steps": len(drawer_trace),
            "success": bool(strict["strict_success"]),
            "final_drawer_fraction": float(final_drawer),
            "max_drawer_fraction": float(max_drawer),
            "ever_attached": ever_attached,
            "n_attached_frames": int(sum(1 for a in attached_trace if a)),
            "drawer_fraction_timeline": [float(d) for d in drawer_trace],
            "strict_success": strict,
            "images_dir": str(rollout_dir),
            "n_images": len(image_paths),
            "first_image_path": image_paths[0] if image_paths else None,
            "color_check": color_check,
            "non_zero_action_ratio": float(non_zero_actions / max(len(raw_action_trace), 1)),
            "total_eef_motion": float(total_eef),
            "elapsed_sec": round(elapsed, 1),
            "metadata": {
                "colored_rendering": True,
                "renderer": "ER_TINY_RENDERER + changeVisualShape",
                "cardboard_color": [0.60, 0.50, 0.40, 1.0],
                "drawer_only_intervention": True,
                "robot_recolor": False,
                "matched_to_E022": True,
                "E022_seeds": E022_SEEDS,
            },
        }

        # Write partial meta immediately
        with open(meta_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        # Generate mp4
        generate_mp4(rollout_dir, video_path)

        log.info(
            f"  DONE {rollout_id}: success={strict['strict_success']}, "
            f"max_drawer={max_drawer:.3f}, attached={ever_attached}, "
            f"colored={color_check['image_is_colored']}, "
            f"rgb_std={mean_std:.1f}, elapsed={elapsed:.0f}s"
        )
        return result

    except Exception as e:
        log.error(f"  ERROR {rollout_id}: {e}")
        import traceback
        log.error(traceback.format_exc())
        error_result = {
            "rollout_id": rollout_id,
            "seed": seed, "policy": policy,
            "error": str(e),
            "success": False,
            "max_drawer_fraction": 0.0,
            "ever_attached": False,
        }
        with open(meta_path, "w") as f:
            json.dump(error_result, f, indent=2, default=str)
        return error_result


def aggregate(all_results: list[dict]) -> dict:
    """Aggregate results across all rollouts."""
    valid = [r for r in all_results if "error" not in r]
    errors = [r for r in all_results if "error" in r]

    by_policy = {}
    for r in valid:
        p = r.get("policy", "unknown")
        by_policy.setdefault(p, []).append(r)

    policy_agg = {}
    for p, rs in by_policy.items():
        policy_agg[p] = {
            "n": len(rs),
            "n_success": sum(1 for r in rs if r.get("success")),
            "n_attached": sum(1 for r in rs if r.get("ever_attached")),
            "mean_max_drawer": float(np.mean([r.get("max_drawer_fraction", 0) for r in rs])),
            "mean_rgb_std": float(np.mean([r.get("color_check", {}).get("rgb_std", 0) for r in rs])),
        }

    return {
        "n_total": len(all_results),
        "n_success": sum(1 for r in valid if r.get("success")),
        "n_attached": sum(1 for r in valid if r.get("ever_attached")),
        "n_errors": len(errors),
        "n_skipped": sum(1 for r in all_results if r.get("skipped")),
        "mean_max_drawer_fraction": float(np.mean(
            [r.get("max_drawer_fraction", 0) for r in valid]
        )) if valid else 0.0,
        "mean_rgb_std": float(np.mean(
            [r.get("color_check", {}).get("rgb_std", 0) for r in valid]
        )) if valid else 0.0,
        "success_rate": float(sum(1 for r in valid if r.get("success")) / max(len(valid), 1)),
        "attach_rate": float(sum(1 for r in valid if r.get("ever_attached")) / max(len(valid), 1)),
        "by_policy": policy_agg,
    }


def run_night_runner(max_runtime_hours: float = 8.0) -> dict:
    """Run the full P0b night runner. Returns session summary."""
    t0_total = time.time()
    max_runtime_sec = max_runtime_hours * 3600
    log.info("=" * 60)
    log.info("P0b Drawer-Only Night Runner — START")
    log.info(f"Rollout matrix: seeds={SEEDS} × policies={POLICIES} × steps={MAX_STEPS}")
    log.info(f"Max runtime: {max_runtime_hours:.1f}h ({max_runtime_sec:.0f}s)")
    log.info("=" * 60)

    all_results = []
    completed = 0
    errors = 0

    for seed in SEEDS:
        for policy in POLICIES:
            elapsed_total = time.time() - t0_total
            if elapsed_total > max_runtime_sec:
                log.warning(
                    f"MAX RUNTIME ({max_runtime_hours:.1f}h) approaching. "
                    f"Stopping after {elapsed_total/3600:.1f}h. "
                    f"Completed {completed}/{len(SEEDS)*len(POLICIES)}."
                )
                break

            result = run_single_rollout(seed, policy, MAX_STEPS)
            all_results.append(result)
            if "error" not in result:
                completed += 1
            else:
                errors += 1
                log.error(f"  {result['rollout_id']} FAILED: {result['error']}")

        if elapsed_total > max_runtime_sec:
            break

    elapsed_total = time.time() - t0_total
    agg = aggregate(all_results)
    valid = [r for r in all_results if "error" not in r]

    # Write aggregate
    results_file = COLORED_DIR / "p0b_night_results.json"
    summary = {
        "session": "P0b_drawer_only_night_runner",
        "started_at": datetime.fromtimestamp(t0_total, tz=timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_hours": round(elapsed_total / 3600, 2),
        "max_runtime_hours": max_runtime_hours,
        "rollout_matrix": {
            "seeds": SEEDS,
            "policies": POLICIES,
            "max_steps": MAX_STEPS,
            "total_rollouts": len(SEEDS) * len(POLICIES),
            "completed": completed,
            "errors": errors,
        },
        "aggregate": agg,
        "rollouts": all_results,
        "a_b_comparison": {
            "A_white_baseline (E022)": {
                "seeds": E022_WHITE_RESULTS["pretrained"]["total"],
                "description": "E022: seeds [1,2,3], 96 steps, white images",
                "pretrained_success": f"{E022_WHITE_RESULTS['pretrained']['success']}/{E022_WHITE_RESULTS['pretrained']['total']}",
                "finetuned_success": f"{E022_WHITE_RESULTS['finetuned']['success']}/{E022_WHITE_RESULTS['finetuned']['total']}",
            },
            "B_colored_drawer (P0b night)": {
                "seeds": SEEDS,
                "description": "Drawer-only colored, same seeds/policies/steps as E022",
                "pretrained_success": f"{agg.get('by_policy', {}).get('pretrained', {}).get('n_success', 0)}/{agg.get('by_policy', {}).get('pretrained', {}).get('n', 0)}",
                "finetuned_success": f"{agg.get('by_policy', {}).get('finetuned', {}).get('n_success', 0)}/{agg.get('by_policy', {}).get('finetuned', {}).get('n', 0)}",
            },
            "matched_A_B": True,
            "same_seeds": True,
            "same_policies": True,
            "same_steps": True,
            "intervention": "drawer-only recolor (cardboard RGBA)",
        },
        "sovereign_update_required": "NO — bounded to raw artifact collection",
        "human_review_gate": "REQUIRED before sovereign verdict upgrade",
    }

    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("=" * 60)
    log.info("P0b Night Runner — COMPLETE")
    log.info(f"Elapsed: {elapsed_total/3600:.2f}h")
    log.info(f"Completed: {completed}/{len(SEEDS)*len(POLICIES)} rollouts")
    log.info(f"Errors: {errors}")
    log.info(f"Results: {results_file}")
    log.info("=" * 60)
    log.info("VERDICT UPGRADE: BLOCKED — human review gate required")
    log.info("Files ready for review:")
    for seed in SEEDS:
        for policy in POLICIES:
            vid = VIDEO_DIR / f"seed_{seed:03d}_{policy}_drawer_colored.mp4"
            if vid.exists():
                log.info(f"  {vid.name}")
    log.info("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P0b Drawer-Only Night Runner")
    parser.add_argument("--max-hours", type=float, default=8.0)
    args = parser.parse_args()
    result = run_night_runner(max_runtime_hours=args.max_hours)
    sys.exit(0)

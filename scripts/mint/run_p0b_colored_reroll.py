#!/usr/bin/env python3
"""
run_p0b_colored_reroll.py — P0b: Colored Drawer-Only Rollout Recording

MATCHED A/B EXPERIMENT — drawer-only intervention, no robot recolor.

Intervention: only DrawerRobotEnv drawer links are colored (cardboard RGBA).
Robot/gripper/background/camera/policy/checkpoint all UNCHANGED from E022.

A/B Design (MATCHED seeds):
  A_white = E022 baseline: seeds [1,2,3], pretrained+finetuned, white images, 96 steps
  B_colored = P0b drawer-only: seeds [1,2,3], pretrained+finetuned, colored drawer, 96 steps
  Same seeds, same policies, same checkpoint, same steps — only drawer color differs.

NOT included in this experiment:
  - Robot recolor (would confound drawer-only conclusion)
  - Full-scene recolor
  - Different seeds than E022

Usage:
  python run_p0b_colored_reroll.py --policy all --seeds 1 2 3 --max-steps 96
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts/mint"))

CAMPAIGN_DIR = PROJECT_ROOT / "experiments/mint/mint_drawer_v1"
ARTIFACT_DIR = CAMPAIGN_DIR / "artifacts"
COLORED_DIR = ARTIFACT_DIR / "p0b_colored_rollouts"
COLORED_DIR.mkdir(exist_ok=True)

PRETRAINED_CKPT = str(PROJECT_ROOT / "external/MINT/checkpoints/MINT-libero")
FINETUNED_CKPT = str(
    CAMPAIGN_DIR / "artifacts/v59_overfit_outputs/checkpoints/last/pretrained_model"
)

E022_SEEDS = [1, 2, 3]  # MUST match E022 baseline exactly for matched A/B


def find_checkpoint(policy: str) -> str | None:
    """Find the correct checkpoint path for the given policy."""
    if policy == "pretrained":
        p = Path(PRETRAINED_CKPT)
        if p.exists():
            return PRETRAINED_CKPT
    elif policy == "finetuned":
        p = Path(FINETUNED_CKPT)
        if p.exists():
            return FINETUNED_CKPT
        for ckpt_dir in sorted(
            (CAMPAIGN_DIR / "artifacts/v59_overfit_outputs/checkpoints").glob("*"),
            reverse=True,
        ):
            if ckpt_dir.is_dir() and not ckpt_dir.name.startswith("."):
                model_path = ckpt_dir / "pretrained_model"
                if (model_path / "model.safetensors").exists():
                    return str(model_path)
    return None


def apply_colored_drawer(env) -> None:
    """Apply cardboard color to DRAWER LINKS ONLY. Robot/gripper/background unchanged.
    
    This is a DRAWER-ONLY intervention for the P0b matched A/B experiment.
    Robot recolor would confound the drawer-only conclusion and is prohibited.
    """
    cardboard_rgba = [0.60, 0.50, 0.40, 1.0]

    drawer_id = env.drawer_id

    for link in range(-1, env.p.getNumJoints(drawer_id)):
        env.p.changeVisualShape(
            drawer_id, linkIndex=link, rgbaColor=cardboard_rgba, physicsClientId=env.client,
        )
    # DO NOT recolor robot — robot recolor is NOT part of drawer-only experiment


def analyze_image_rgb(img_array: np.ndarray, image_size: int = 224) -> dict:
    """Compute RGB stats on a single frame. Used for both verify and actual frames."""
    rgb = img_array[:, :, :3].astype(float)
    white_mask = np.all(rgb > 245, axis=2)
    non_white_count = (~white_mask).sum()
    non_white_pct = non_white_count / (image_size * image_size) * 100
    rgb_mean = float(rgb.mean())
    rgb_std = float(rgb.std())
    return {
        "rgb_mean": rgb_mean,
        "rgb_std": rgb_std,
        "non_white_pixels": int(non_white_count),
        "non_white_pct": float(non_white_pct),
        "image_is_colored": bool(non_white_pct > 0.5 and rgb_std > 10.0),
        "rgb_sample_center": [int(rgb[image_size//2, image_size//2, c]) for c in range(3)],
        "rgb_sample_corner": [int(rgb[4, 4, c]) for c in range(3)],
    }


def run_colored_rollout(
    seed: int,
    policy: str,
    max_steps: int = 96,
    image_size: int = 224,
) -> dict:
    """
    Run a single MINT policy rollout with colored drawer rendering.
    Uses the COLORED env directly (not rollout_policy() which creates a plain env).

    Returns rollout metrics with colored frames saved to disk.
    """
    import torch
    from PIL import Image

    from drawer_robot_env import DrawerRobotEnv
    from mint_common import DATASET_DIR, DATASET_REPO_ID, load_json
    from evaluate_mint_drawer_campaign import _load_policy, _obs_to_batch
    from mint_eval_patches import apply as _apply_patches

    _apply_patches()

    ckpt = find_checkpoint(policy)
    if ckpt is None:
        return {
            "seed": seed, "policy": policy,
            "error": f"Checkpoint not found for policy={policy}",
        }

    print(f"  Seed {seed}, policy={policy}: ckpt={Path(ckpt).name}")

    # ── Load policy ────────────────────────────────────────────────────────
    policy_bundle = _load_policy(ckpt, DATASET_DIR, DATASET_REPO_ID)
    policy_obj, preprocessor, postprocessor = policy_bundle

    ACTION_CONTRACT = load_json(
        CAMPAIGN_DIR / "artifacts/active_action_contract.json", {}
    )

    # ── Create colored env (THIS is the env used for the entire rollout) ──
    env = DrawerRobotEnv(
        seed=seed,
        image_size=image_size,
        max_steps=max_steps,
        action_contract=ACTION_CONTRACT,
    )
    obs = env.reset()

    # Apply cardboard color BEFORE any frames are captured
    apply_colored_drawer(env)

    # ── Run rollout loop directly on the colored env ───────────────────────
    drawer_trace = []
    attached_trace = []
    frames = []
    eef_pos_trace = []
    raw_action_trace = []

    for step in range(1, max_steps + 1):
        # Capture frame from colored env (this is the key fix)
        frames.append(obs.image.copy())

        # Track EEF
        eef_pos_trace.append(
            obs.eef_pos.copy() if hasattr(obs, "eef_pos") else obs.state[:3].copy()
        )

        # Policy inference (same as evaluate_mint_drawer_campaign.py)
        if policy == "random":
            action = np.random.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)
            action[6] = float(np.random.choice([-1.0, 1.0]))
        else:
            batch = _obs_to_batch(obs)
            processed = preprocessor(batch)  # CRITICAL FIX: preprocess adds language.tokens
            with torch.inference_mode():
                action = policy_obj.select_action(processed)
            action = postprocessor(action)
            action = action.squeeze(0).detach().cpu().numpy().astype(np.float32)

        raw_action_trace.append(action.copy() if action is not None else None)

        # Step
        obs, _, done, info = env.step(action)

        drawer_trace.append(float(info["drawer_fraction"]))
        attached_trace.append(bool(info.get("attached")))

        if done:
            break

    env.close()

    # ── Save colored frames ───────────────────────────────────────────────
    rollout_id = f"seed_{seed:03d}_{policy}_drawer_colored"
    rollout_dir = COLORED_DIR / rollout_id
    rollout_dir.mkdir(exist_ok=True)

    image_paths = []
    frame_rgb_stats = []
    for i, img in enumerate(frames):
        img_path = rollout_dir / f"frame_{i:04d}.png"
        Image.fromarray(img).save(img_path)
        image_paths.append(str(img_path))
        if i < 5 or i % 20 == 0:
            frame_rgb_stats.append(analyze_image_rgb(img, image_size))

    # ── Compute metrics ───────────────────────────────────────────────────
    max_drawer = max(drawer_trace) if drawer_trace else 0.0
    final_drawer = drawer_trace[-1] if drawer_trace else 0.0
    ever_attached = any(attached_trace)
    non_zero_actions = sum(
        1 for a in raw_action_trace if a is not None and np.abs(a).max() > 1e-6
    )
    total_eef = 0.0
    for i in range(1, len(eef_pos_trace)):
        total_eef += float(np.linalg.norm(np.array(eef_pos_trace[i]) - np.array(eef_pos_trace[i-1])))

    # Strict success (same as evaluate_mint_drawer_campaign.py)
    from strict_success import evaluate_strict_success
    strict = evaluate_strict_success(
        np.asarray(drawer_trace, dtype=np.float32),
        np.asarray(attached_trace, dtype=bool),
    )

    # ── Verify actual frames are colored ─────────────────────────────────
    actual_colored_count = sum(
        1 for s in frame_rgb_stats if s["image_is_colored"]
    ) if frame_rgb_stats else 0
    mean_actual_std = float(np.mean([s["rgb_std"] for s in frame_rgb_stats])) if frame_rgb_stats else 0.0
    mean_actual_non_white = float(np.mean([s["non_white_pct"] for s in frame_rgb_stats])) if frame_rgb_stats else 0.0

    color_check = {
        "rgb_mean": float(np.mean([s["rgb_mean"] for s in frame_rgb_stats])) if frame_rgb_stats else 0.0,
        "rgb_std": mean_actual_std,
        "non_white_pct": mean_actual_non_white,
        "image_is_colored": bool(actual_colored_count >= max(1, len(frame_rgb_stats) // 2)),
        "colored_frames_analyzed": len(frame_rgb_stats),
        "colored_frames_among_analyzed": actual_colored_count,
        "note": "Verified from ACTUAL rollout frames, not test frame",
    }

    print(
        f"  Color check (actual frames): std={mean_actual_std:.2f}, "
        f"non_white={mean_actual_non_white:.1f}%, colored={color_check['image_is_colored']}"
    )

    output = {
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
        "metadata": {
            "colored_rendering": True,
            "renderer": "ER_TINY_RENDERER + changeVisualShape",
            "cardboard_color": [0.60, 0.50, 0.40, 1.0],
            "p0b_scope": "Matched A/B: drawer-only recolor, same seeds/policies/steps as E022",
            "drawer_only_intervention": True,
            "robot_recolor": False,
            "matched_to_E022": True,
            "E022_seeds": E022_SEEDS,
            "critical_fix": "rollout_policy() replaced with inline loop using drawer-only colored env",
        },
    }

    meta_path = rollout_dir / "rollout_meta.json"
    with open(meta_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(
        f"  → success={strict['strict_success']}, max_drawer={max_drawer:.3f}, "
        f"attached={ever_attached}, colored={color_check['image_is_colored']}, "
        f"frames={len(image_paths)}"
    )

    return output


def aggregate_results(results: list[dict]) -> dict:
    """Aggregate results across rollouts."""
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    n_colored = sum(
        1 for r in valid
        if r.get("color_check", {}).get("image_is_colored", False)
    )
    mean_max_drawer = float(np.mean([r.get("max_drawer_fraction", 0.0) for r in valid])) if valid else 0.0
    mean_rgb_std = float(np.mean(
        [r.get("color_check", {}).get("rgb_std", 0.0) for r in valid]
    )) if valid else 0.0

    return {
        "n_total": len(results),
        "n_success": sum(1 for r in valid if r.get("success")),
        "n_attached": sum(1 for r in valid if r.get("ever_attached")),
        "n_errors": len(errors),
        "n_colored": n_colored,
        "colored_fraction": float(n_colored / max(len(valid), 1)),
        "mean_max_drawer_fraction": mean_max_drawer,
        "mean_rgb_std": mean_rgb_std,
        "success_rate": float(sum(1 for r in valid if r.get("success")) / max(len(valid), 1)),
        "attach_rate": float(sum(1 for r in valid if r.get("ever_attached")) / max(len(valid), 1)),
    }


def main():
    parser = argparse.ArgumentParser(description="P0b: Colored Rollout Recording")
    parser.add_argument("--policy", choices=["pretrained", "finetuned", "all"], default="all")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir:
        global COLORED_DIR
        COLORED_DIR = Path(args.output_dir)
        COLORED_DIR.mkdir(exist_ok=True)

    seeds = [args.seed] if args.seed is not None else (args.seeds or E022_SEEDS)
    policies = ["pretrained", "finetuned"] if args.policy == "all" else [args.policy]

    print(f"=== P0b Colored Reroll ===")
    print(f"Seeds: {seeds}")
    print(f"Policies: {policies}")
    print(f"Max steps: {args.max_steps}")
    print(f"Output: {COLORED_DIR}")
    print()

    all_results = []
    for policy in policies:
        print(f"[{policy.upper()}]")
        for seed in seeds:
            result = run_colored_rollout(
                seed=seed,
                policy=policy,
                max_steps=args.max_steps,
                image_size=args.image_size,
            )
            all_results.append(result)
        print()

    agg = aggregate_results(all_results)
    valid = [r for r in all_results if "error" not in r]

    print(f"=== Aggregate ===")
    print(f"  Colored: {agg['n_colored']}/{len(valid)}")
    print(f"  Success: {agg['n_success']}/{len(valid)}")
    print(f"  Attached: {agg['n_attached']}/{len(valid)}")
    print(f"  Mean max drawer: {agg['mean_max_drawer_fraction']:.3f}")
    print(f"  Mean rgb std: {agg['mean_rgb_std']:.2f}")
    print()

    print(f"=== E022 Baseline (white images, seeds [1,2,3]) ===")
    print(f"  Pretrained: 0/6 success, 0/6 attached (E022: seeds [1,2,3], 96 steps)")
    print(f"  Finetuned: 0/6 success, 0/6 attached")
    print()
    print(f"=== P0b Drawer-Only Results (colored drawer, seeds {seeds}, 96 steps) ===")
    for policy in policies:
        policy_results = [r for r in valid if r.get("policy") == policy]
        p_agg = aggregate_results(policy_results)
        print(
            f"  {policy}: {p_agg['n_success']}/{len(policy_results)} success, "
            f"{p_agg['n_attached']}/{len(policy_results)} attached, "
            f"mean_max_drawer={p_agg['mean_max_drawer_fraction']:.3f}"
        )
    print()

    if agg["n_colored"] == 0:
        print("WARNING: No colored images detected. changeVisualShape may not work with ER_TINY_RENDERER.")

    output_file = COLORED_DIR / "p0b_colored_reroll_results.json"
    overall = {
        "experiment": "P0b_colored_reroll",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        "seeds": seeds,
        "policies": policies,
        "rollouts": all_results,
        "aggregate": agg,
        "a_b_context": {
            "A_white_baseline": "E022 env gate: 0/6 success, 0/6 attached for pretrained+finetuned (seeds [1,2,3], 96 steps, white images)",
            "B_colored_target": f"P0b drawer-only: {agg['n_success']}/{len(valid)} success, {agg['n_attached']}/{len(valid)} attached (seeds [1,2,3], 96 steps, colored drawer)",
            "matched_A_B": True,
            "same_seeds": E022_SEEDS,
            "same_policies": ["pretrained", "finetuned"],
            "same_steps": 96,
            "intervention": "drawer-only recolor (cardboard RGBA via changeVisualShape)",
            "robot_recolor": False,
            "if_B_gt_A": "drawer color IS likely a bottleneck (requires matched A/B confirmation)",
            "if_B_eq_A": "drawer color NOT the sole bottleneck (action/physics mismatch remains)",
        },
        "scope_guardrail": (
            "This is a MATCHED A/B: same seeds [1,2,3], same policies, same 96 steps. "
            "Only intervention is drawer-only color change. "
            "Robot/background/camera/policy/checkpoint are UNCHANGED from E022 baseline. "
            "DO NOT mix with seed 7/9/10 exploratory results."
        ),
    }
    with open(output_file, "w") as f:
        json.dump(overall, f, indent=2, default=str)
    print(f"Results: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

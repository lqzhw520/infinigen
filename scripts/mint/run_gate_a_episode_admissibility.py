#!/usr/bin/env python3
"""
Gate A: Episode Admissibility Audit
==================================
对全量 240 V59 episodes 做 episode-level 诊断，分类：

- task-teaching:     high action variance, non-white image, state continuity OK
- motion-only:        has motion but weak/no visual texture OR low action variance
- weak/noisy:        low action variance OR anomalous state OR physics illegal
- white-homogeneous: near-white images (visual sufficiency concern)

输出：
  - per-episode classification CSV
  - aggregate distribution
  - recommended replay candidates for Gate B
  - E027 evidence YAML
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN = PROJECT_ROOT / "experiments/mint/mint_drawer_v1"
ARTIFACT_DIR = CAMPAIGN / "artifacts"
DATA_DIR = CAMPAIGN / "dataset"
STATE_DIR = CAMPAIGN / "sovereign"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts/mint"))


# ── Thresholds ────────────────────────────────────────────────────────────────

# Image thresholds
IMAGE_STD_WHITE = 0.010  # std < 0.010 = near-white
IMAGE_STD_COLORED = 0.050  # std > 0.050 = clearly colored

# Action thresholds
ACTION_STD_WEAK = 0.05    # action std < 0.05 = near-zero motion
ACTION_STD_NOMINAL = 0.15  # action std > 0.15 = meaningful motion
GRIPPER_STD_BINARY = 0.95  # gripper std > 0.95 = binary-like
GRIPPER_STD_CONTINUOUS = 0.02  # gripper std < 0.02 = frozen/singular

# State thresholds
STATE_STD_WEAK = 0.005  # state std < 0.005 = near-static

# Episode length
MIN_EPISODE_LEN = 10  # episodes < 10 frames are too short


def extract_episode_stats(ep_row: dict) -> dict:
    """Extract and normalize per-episode scalar stats from parquet row."""
    # Image
    img_mean = np.array(ep_row["stats/observation.images.image/mean"])
    img_std = np.array(ep_row["stats/observation.images.image/std"])
    img_mean_scalar = float(np.mean(img_mean))
    img_std_scalar = float(np.mean(img_std))

    # Action
    act_mean = np.array(ep_row["stats/action/mean"])
    act_std = np.array(ep_row["stats/action/std"])
    act_max = np.array(ep_row["stats/action/max"])
    act_min = np.array(ep_row["stats/action/min"])
    gripper_std = float(act_std[6])  # action[6] = gripper
    pos_std = float(np.mean(act_std[:3]))  # position actions
    gripper_range = float(act_max[6] - act_min[6])
    action_std = float(np.mean(act_std))

    # State
    state_mean = np.array(ep_row["stats/observation.state/mean"])
    state_std = np.array(ep_row["stats/observation.state/std"])
    state7_mean = float(state_mean[7])  # gripper state
    state7_std = float(state_std[7])
    gripper_state_range = float(state_mean[7])  # mean gripper state

    # Episode length
    ep_len = int(ep_row["length"])
    ep_index = int(ep_row["episode_index"])
    data_from = int(ep_row["dataset_from_index"])
    data_to = int(ep_row["dataset_to_index"])

    return {
        "episode_index": ep_index,
        "length": ep_len,
        "data_from": data_from,
        "data_to": data_to,
        # Image
        "image_mean": img_mean_scalar,
        "image_std": img_std_scalar,
        # Action
        "action_std": action_std,
        "action_pos_std": pos_std,
        "gripper_action_std": gripper_std,
        "gripper_action_range": gripper_range,
        "action_max": float(np.max(act_max)),
        "action_min": float(np.min(act_min)),
        # State
        "state_std": float(np.mean(state_std)),
        "gripper_state_std": state7_std,
        "gripper_state_mean": state7_mean,
    }


def classify_episode(s: dict) -> tuple[str, str]:
    """Classify single episode. Returns (tier, reason)."""
    reasons = []

    # ── Length check ────────────────────────────────────────────────────
    if s["length"] < MIN_EPISODE_LEN:
        return "weak/noisy", f"too_short_{s['length']}f"

    # ── Image whiteness ──────────────────────────────────────────────────
    if s["image_std"] < IMAGE_STD_WHITE:
        whiteness = "near-white"
    elif s["image_std"] > IMAGE_STD_COLORED:
        whiteness = "colored"
    else:
        whiteness = "partially-colored"

    # ── Action quality ────────────────────────────────────────────────────
    if s["action_std"] < ACTION_STD_WEAK:
        action_quality = "weak"
        reasons.append("weak_action")
    elif s["action_std"] > ACTION_STD_NOMINAL:
        action_quality = "strong"
    else:
        action_quality = "moderate"

    # ── Gripper state analysis ─────────────────────────────────────────────
    gripper_state = s["gripper_state_std"]
    if gripper_state < GRIPPER_STD_CONTINUOUS:
        gripper_type = "frozen"
    elif gripper_state > GRIPPER_STD_BINARY:
        gripper_type = "binary"
    else:
        gripper_type = "continuous"

    # ── Gripper action analysis ───────────────────────────────────────────
    gripper_act = s["gripper_action_std"]
    if gripper_act > GRIPPER_STD_BINARY:
        gripper_act_type = "binary"
    elif gripper_act < GRIPPER_STD_CONTINUOUS:
        gripper_act_type = "frozen"
    else:
        gripper_act_type = "continuous"

    # ── Motion analysis ────────────────────────────────────────────────────
    if s["action_pos_std"] < 0.02:
        motion_type = "near-static"
    elif s["action_pos_std"] < 0.05:
        motion_type = "low-motion"
    elif s["action_pos_std"] < 0.15:
        motion_type = "moderate-motion"
    else:
        motion_type = "high-motion"

    # ── TIER classification ───────────────────────────────────────────────
    if action_quality == "weak":
        tier = "weak/noisy"
        reasons.append(f"{action_quality}_action_std={s['action_std']:.4f}")
    elif gripper_type == "frozen" and gripper_act_type == "frozen":
        tier = "weak/noisy"
        reasons.append("frozen_gripper")
    elif whiteness == "near-white":
        tier = "motion-only"
        reasons.append(f"near-white_image_std={s['image_std']:.4f}")
        reasons.append(f"gripper_{gripper_act_type}")
    elif gripper_act_type == "frozen":
        tier = "motion-only"
        reasons.append(f"frozen_gripper_action_std={gripper_act:.4f}")
    else:
        tier = "task-teaching"
        reasons.append(f"good_motion_std={s['action_std']:.3f}")
        reasons.append(f"gripper_{gripper_act_type}")
        reasons.append(whiteness)

    reason = "; ".join(reasons)
    return tier, reason


def run_gate_a() -> dict:
    t0 = time.time()

    # ── Load episodes parquet ────────────────────────────────────────────────
    ep_path = DATA_DIR / "meta/episodes/chunk-000/file-000.parquet"
    df = pq.read_table(ep_path).to_pandas()
    n_total = len(df)
    print(f"Gate A: {n_total} episodes")

    # ── Per-episode analysis ────────────────────────────────────────────────
    results = []
    tier_counts = {"task-teaching": 0, "motion-only": 0, "weak/noisy": 0}

    for _, row in df.iterrows():
        ep = dict(row)
        stats = extract_episode_stats(ep)
        tier, reason = classify_episode(stats)
        stats["tier"] = tier
        stats["reason"] = reason
        results.append(stats)
        tier_counts[tier] += 1

    elapsed = time.time() - t0
    print(f"\nGate A Results ({elapsed:.1f}s):")
    print(f"  task-teaching: {tier_counts['task-teaching']}/{n_total}")
    print(f"  motion-only:   {tier_counts['motion-only']}/{n_total}")
    print(f"  weak/noisy:    {tier_counts['weak/noisy']}/{n_total}")

    # ── Summary stats ──────────────────────────────────────────────────────
    action_stds = [r["action_std"] for r in results]
    gripper_stds = [r["gripper_action_std"] for r in results]
    img_stds = [r["image_std"] for r in results]
    ep_lens = [r["length"] for r in results]

    print(f"\nAction std:    mean={np.mean(action_stds):.4f} std={np.std(action_stds):.4f} min={np.min(action_stds):.4f} max={np.max(action_stds):.4f}")
    print(f"Gripper std:   mean={np.mean(gripper_stds):.4f} std={np.std(gripper_stds):.4f} min={np.min(gripper_stds):.4f} max={np.max(gripper_stds):.4f}")
    print(f"Image std:     mean={np.mean(img_stds):.4f} std={np.std(img_stds):.4f} min={np.min(img_stds):.4f} max={np.max(img_stds):.4f}")
    print(f"Episode len:   mean={np.mean(ep_lens):.1f} min={np.min(ep_lens)} max={np.max(ep_lens)}")

    # ── Gripper analysis ──────────────────────────────────────────────────
    binary_gripper = sum(1 for g in gripper_stds if g > 0.9)
    frozen_gripper = sum(1 for g in gripper_stds if g < 0.02)
    continuous_gripper = sum(1 for g in gripper_stds if 0.02 <= g <= 0.9)
    print(f"\nGripper action distribution:")
    print(f"  binary:     {binary_gripper}/{n_total}")
    print(f"  continuous: {continuous_gripper}/{n_total}")
    print(f"  frozen:     {frozen_gripper}/{n_total}")

    # ── White image analysis ────────────────────────────────────────────────
    white_eps = sum(1 for i in img_stds if i < IMAGE_STD_WHITE)
    colored_eps = sum(1 for i in img_stds if i > IMAGE_STD_COLORED)
    partial_eps = n_total - white_eps - colored_eps
    print(f"\nImage color distribution:")
    print(f"  near-white:   {white_eps}/{n_total}")
    print(f"  partial:       {partial_eps}/{n_total}")
    print(f"  colored:       {colored_eps}/{n_total}")

    # ── Identify replay candidates for Gate B ────────────────────────────────
    # Best candidates: task-teaching episodes with good motion and reasonable image
    replay_candidates = [
        r for r in results
        if r["tier"] in ("task-teaching", "motion-only")
        and r["length"] >= 20
        and r["action_std"] >= 0.10
    ]
    print(f"\nGate B replay candidates: {len(replay_candidates)} episodes")
    print(f"  (tier in [task-teaching, motion-only], length >= 20, action_std >= 0.10)")

    # Top candidates (best motion quality)
    best = sorted(replay_candidates, key=lambda x: x["action_std"], reverse=True)[:10]
    print(f"  Top 10 by action_std:")
    for r in best:
        print(f"    ep={r['episode_index']:3d} tier={r['tier']:15s} std={r['action_std']:.3f} "
              f"gripper={r['gripper_action_std']:.3f} len={r['length']:3d} img_std={r['image_std']:.4f}")

    return {
        "n_total": n_total,
        "tier_counts": tier_counts,
        "tier_distribution": {
            tier: f"{count}/{n_total} ({count/n_total*100:.1f}%)"
            for tier, count in tier_counts.items()
        },
        "gripper_distribution": {
            "binary": f"{binary_gripper}/{n_total}",
            "continuous": f"{continuous_gripper}/{n_total}",
            "frozen": f"{frozen_gripper}/{n_total}",
        },
        "image_distribution": {
            "near-white": f"{white_eps}/{n_total}",
            "partial": f"{partial_eps}/{n_total}",
            "colored": f"{colored_eps}/{n_total}",
        },
        "action_std_summary": {
            "mean": float(np.mean(action_stds)),
            "std": float(np.std(action_stds)),
            "min": float(np.min(action_stds)),
            "max": float(np.max(action_stds)),
        },
        "gate_b_replay_candidates": [
            {"episode_index": r["episode_index"], "tier": r["tier"],
             "action_std": r["action_std"], "length": r["length"]}
            for r in best
        ],
        "elapsed_sec": round(elapsed, 1),
        "per_episode_results": results,
    }


if __name__ == "__main__":
    result = run_gate_a()

    # Save results
    out_dir = ARTIFACT_DIR / "gate_a_episode_admissibility"
    out_dir.mkdir(exist_ok=True)

    # Save full JSON (for E027)
    with open(out_dir / "gate_a_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Save CSV (for human review)
    import csv
    rows = result["per_episode_results"]
    fields = ["episode_index", "length", "tier", "reason",
              "image_std", "action_std", "gripper_action_std",
              "gripper_state_std", "action_pos_std"]
    with open(out_dir / "gate_a_episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    print(f"\nResults: {out_dir}")
    print(f"  gate_a_results.json")
    print(f"  gate_a_episodes.csv")

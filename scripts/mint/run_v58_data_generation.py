#!/usr/bin/env python3
"""V58: Generate physics-legal rollouts for V58 training (Option C pilot).

Goal: Generate 5000+ physics-legal frames for MINT fine-tuning.
Strategy:
  1. Reuse existing P4 rollouts (13 rollouts, ~795 frames)
  2. Extend to seeds 7-15 (currently only seeds 1-6 have legal rollouts)
  3. Use physics_constrained_teacher_rollout() with branch sweep

This is the Option C "先 pilot 数据生产，确认 V58 训练可行，再决定 render/tokenizer 路线"
execution script.

Usage:
  # Run with screen:
  screen -dmS v58_data bash -c 'source /root/anaconda3/etc/profile.d/conda.sh && conda activate infinigen && cd /mnt/afs2/zhuhaowu/infinigen && python scripts/mint/run_v58_data_generation.py 2>&1 | tee experiments/mint/mint_drawer_v1/outputs/v58_data.log'
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
SCRIPTS_MINT = PROJECT_ROOT / "scripts" / "mint"
sys.path.insert(0, str(SCRIPTS_MINT))

from action_contract_repair import branch_configs
from drawer_robot_env import build_native_teacher_rollout
from mint_common import ARTIFACT_DIR
from physics_legality import (
    DEFAULT_PHYSICS_THRESHOLDS,
    check_physics_legality,
    physics_constrained_teacher_rollout,
)

# ── Paths ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = ARTIFACT_DIR / "v58_physics_legal_rollouts"
GRASP_CACHE = ARTIFACT_DIR / "g3_anygrasp_grasps"

# ── Targets ────────────────────────────────────────────────────────────────────
MIN_LEGAL_SEEDS = 10       # at least 10 unique seeds
TARGET_FRAMES = 5000       # target total frames
EPISODES_PER_SEED = 3      # try 3 episodes per (seed, branch) combo
MAX_SEEDS_TO_TRY = 15      # [1..15]


def _seed_list() -> list[int]:
    # Already have seeds 1-6 from P4, extend to 7-15
    return list(range(7, MAX_SEEDS_TO_TRY + 1))


def _load_grasp_pose(seed: int) -> dict | None:
    gf = GRASP_CACHE / f"seed_{seed:03d}.json"
    if not gf.exists():
        return None
    try:
        data = json.loads(gf.read_text())
        sel = data.get("selected_grasp")
        if not sel:
            top = data.get("top_candidates", [])
            sel = top[0] if top else None
        return sel
    except (json.JSONDecodeError, OSError):
        return None


def _to_list(val):
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return val
    return list(val)


def _save_rollout(
    seed: int,
    branch_id: str,
    episode_index: int,
    rollout: dict,
    rollouts_dir: Path,
) -> Path:
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    path = rollouts_dir / f"seed_{seed:03d}_{branch_id}_ep{episode_index}.npz"
    np.savez_compressed(
        path,
        states=np.asarray(rollout.get("states", []), dtype=np.float32),
        actions=np.asarray(rollout.get("actions", []), dtype=np.float32),
        images=np.asarray(_to_list(rollout.get("images")), dtype=np.uint8),
        images2=np.asarray(_to_list(rollout.get("images2")), dtype=np.uint8),
        gripper_values=np.asarray(_to_list(rollout.get("gripper_values")), dtype=np.float32),
        eef_positions=np.asarray(_to_list(rollout.get("eef_positions")), dtype=np.float32),
        absolute_drawer_fraction=np.asarray(
            _to_list(rollout.get("absolute_drawer_fraction")), dtype=np.float32
        ),
        seed=np.int32(seed),
        branch_id=branch_id,
        episode_index=np.int32(episode_index),
        success=np.bool_(rollout.get("success", False)),
        final_drawer_fraction=np.float32(rollout.get("final_drawer_fraction", 0.0)),
        max_drawer_fraction=np.float32(rollout.get("max_drawer_fraction", 0.0)),
        ever_attached=np.bool_(rollout.get("ever_attached", False)),
        joint_positions=np.asarray(
            _to_list(rollout.get("joint_positions")), dtype=np.float32
        ),
        attached_trace=np.asarray(
            _to_list(rollout.get("attached_trace")), dtype=np.bool_
        ),
        handle_distance_trace=np.asarray(
            _to_list(rollout.get("handle_distance_trace")), dtype=np.float32
        ),
    )
    return path


def _frames_to_rollout_dict(frames: list) -> dict:
    T = len(frames)
    if T == 0:
        return {}
    states = np.array([f["state"] for f in frames], dtype=np.float32)
    actions = np.array([f["action"] for f in frames], dtype=np.float32)
    drawer_fracs = np.array([f.get("drawer_fraction", 0.0) for f in frames], dtype=np.float32)
    return {
        "states": states,
        "actions": actions,
        "images": [],
        "images2": [],
        "eef_positions": states[:, :3].copy() if T > 0 else np.zeros((0, 3), dtype=np.float32),
        "gripper_values": states[:, 7].copy() if T > 0 else np.zeros(0, dtype=np.float32),
        "absolute_drawer_fraction": drawer_fracs,
        "final_drawer_fraction": float(drawer_fracs[-1]) if T > 0 else 0.0,
        "max_drawer_fraction": float(drawer_fracs.max()) if T > 0 else 0.0,
        "joint_positions": states[:, 3:7].copy() if T > 0 else np.zeros((0, 4), dtype=np.float32),
    }


def run() -> bool:
    t0 = time.monotonic()
    seeds = _seed_list()
    branches = branch_configs()

    print(f"[V58] Generating physics-legal rollouts for seeds {seeds}")
    print(f"[V58] Target: >= {MIN_LEGAL_SEEDS} seeds, >= {TARGET_FRAMES} frames")
    print(f"[V58] Output: {OUTPUT_DIR}")

    records = []
    legal_seeds = set()
    total_frames = 0
    legal_rollouts = []

    for seed in seeds:
        grasp = _load_grasp_pose(seed)
        if grasp is None:
            print(f"  [skip] seed={seed:03d}: no grasp plan")
            records.append({"seed": seed, "passed": False, "reason": "no_grasp"})
            continue

        pose_world = np.array(grasp["pose_world"], dtype=np.float32)
        anygrasp_candidates = [pose_world]
        for top in grasp.get("top_candidates", [])[:4]:
            anygrasp_candidates.append(np.array(top["pose_world"], dtype=np.float32))

        seed_legal = False
        for branch in branches:
            bid = branch["branch_id"]
            for ep_idx in range(EPISODES_PER_SEED):
                rollout = None
                legality_result = {"legal": False}
                method = "unknown"

                # ── Phase 1: Physics-constrained ─────────────────────────────────
                try:
                    result = physics_constrained_teacher_rollout(
                        seed=seed,
                        anygrasp_candidates=anygrasp_candidates,
                        max_retry=3,
                        verbose=False,
                    )
                    method = "physics_constrained"
                    legality_result = result.get("legality", {})
                    if legality_result.get("legal"):
                        frames = result.get("rollout_frames", [])
                        if frames:
                            rollout = _frames_to_rollout_dict(frames)
                            rollout["success"] = legality_result.get("robot_success", True)
                            rollout["final_drawer_fraction"] = legality_result.get("final_drawer_fraction", 0.0)
                            rollout["max_drawer_fraction"] = legality_result.get("max_drawer_fraction", 0.0)
                            rollout["ever_attached"] = bool(np.any([
                                f.get("collision", {}).get("attached", False) for f in frames
                            ]))
                except Exception as exc:
                    legality_result = {"legal": False, "issues": [f"physics_constrained: {exc}"]}

                # ── Phase 2: Fallback to build_native ─────────────────────────
                if rollout is None:
                    try:
                        rollout = build_native_teacher_rollout(
                            seed=seed,
                            grasp_pose_world=pose_world,
                            branch_config=branch,
                            episode_index=ep_idx,
                            max_steps=96,
                        )
                        method = "build_native_fallback"
                        legality_result = check_physics_legality(
                            rollout, DEFAULT_PHYSICS_THRESHOLDS
                        )
                    except Exception as exc:
                        print(f"  [error] seed={seed:03d} branch={bid} ep={ep_idx}: {exc}")
                        records.append({
                            "seed": seed, "branch": bid, "episode": ep_idx,
                            "passed": False, "reason": str(exc),
                        })
                        continue

                robot_success = bool(rollout.get("success", False))
                physics_ok = bool(legality_result.get("legal", False))
                passed = robot_success and physics_ok

                n_frames = len(rollout.get("actions", []))

                record = {
                    "seed": seed,
                    "branch": bid,
                    "episode": ep_idx,
                    "passed": passed,
                    "method": method,
                    "n_frames": n_frames,
                    "max_drawer_fraction": float(rollout.get("max_drawer_fraction", 0.0)),
                    "issues": legality_result.get("issues", []),
                }

                if passed:
                    path = _save_rollout(seed, bid, ep_idx, rollout, OUTPUT_DIR)
                    record["path"] = str(path)
                    legal_seeds.add(seed)
                    total_frames += n_frames
                    legal_rollouts.append(record)
                    print(
                        f"  [legal:{method}] seed={seed:03d} branch={bid} ep={ep_idx}"
                        f"  frames={n_frames} drawer={record['max_drawer_fraction']:.2f}"
                    )
                else:
                    issues_str = ", ".join(record["issues"][:1]) if record["issues"] else "none"
                    print(
                        f"  [reject:{method}] seed={seed:03d} branch={bid} ep={ep_idx}"
                        f"  robot_ok={robot_success} physics_ok={physics_ok}"
                        f"  issues=[{issues_str[:60]}]"
                    )

                records.append(record)

                # Early stop if we have enough frames
                if total_frames >= TARGET_FRAMES and len(legal_seeds) >= MIN_LEGAL_SEEDS:
                    print(f"\n[V58] Reached target: {total_frames} frames, {len(legal_seeds)} seeds")
                    break
            if total_frames >= TARGET_FRAMES and len(legal_seeds) >= MIN_LEGAL_SEEDS:
                break
        if total_frames >= TARGET_FRAMES and len(legal_seeds) >= MIN_LEGAL_SEEDS:
            break

    elapsed = time.monotonic() - t0

    # Check existing P4 rollouts (seeds 1-6) and add their frame counts
    p4_dir = ARTIFACT_DIR / "p4_physics_legal_rollouts"
    p4_frames = 0
    p4_seeds = set()
    if p4_dir.exists():
        for pf in p4_dir.glob("*.npz"):
            try:
                d = dict(np.load(pf, allow_pickle=True))
                if d.get("success"):
                    p4_frames += len(d.get("actions", []))
                    p4_seeds.add(int(d.get("seed", 0)))
            except Exception:
                pass

    total_all_frames = total_frames + p4_frames
    total_all_seeds = legal_seeds | p4_seeds

    result = {
        "gate": "v58_data_generation",
        "passed": len(total_all_seeds) >= MIN_LEGAL_SEEDS and total_all_frames >= TARGET_FRAMES,
        "new_rollouts": len(legal_rollouts),
        "new_frames": total_frames,
        "new_seeds": sorted(legal_seeds),
        "p4_frames": p4_frames,
        "p4_seeds": sorted(p4_seeds),
        "total_frames": total_all_frames,
        "total_seeds": sorted(total_all_seeds),
        "target_frames": TARGET_FRAMES,
        "target_seeds": MIN_LEGAL_SEEDS,
        "records": records,
        "legal_rollouts": legal_rollouts,
        "elapsed_s": round(elapsed, 1),
        "timestamp": time.time(),
    }

    ARTIFACT_OUT = ARTIFACT_DIR / "v58_data_generation.json"
    ARTIFACT_OUT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_OUT.write_text(json.dumps(result, indent=2))

    print(f"\n{'='*60}")
    print(f"[V58] {'PASS' if result['passed'] else 'INCOMPLETE'}")
    print(f"  New rollouts: {len(legal_rollouts)}, New frames: {total_frames}")
    print(f"  P4 frames: {p4_frames} (seeds 1-6)")
    print(f"  Total frames: {total_all_frames} / {TARGET_FRAMES}")
    print(f"  Total seeds: {len(total_all_seeds)} / {MIN_LEGAL_SEEDS}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print(f"{'='*60}")
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

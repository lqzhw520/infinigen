#!/usr/bin/env python3
"""P4: Physics-legal teacher rollout gate.

Gate condition: ≥ MIN_LEGAL unique seeds produce physics-legal rollouts.

ROOT FIX (vs. previous version):
  - OLD: checked ||Δaction||_2 against meter thresholds → false failures
        (normalized gripper ±1 jumps already give Δ≈2, not 0.05m)
  - NEW: judges real physics outcomes — Δeef_position_world, attach_persistence,
        drawer_causal_follow, replay determinism

Generation path (in priority order):
  1. physics_constrained_teacher_rollout(): impedance controller + grasp feasibility
  2. build_native_teacher_rollout(): fallback with branch sweep

Exit codes:
  0 = gate passed (≥ MIN_LEGAL physics-legal rollouts)
  1 = gate failed (fewer than MIN_LEGAL physics-legal rollouts)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
SCRIPTS_MINT = PROJECT_ROOT / "scripts" / "mint"

import sys

sys.path.insert(0, str(SCRIPTS_MINT))

from action_contract_repair import branch_configs
from drawer_robot_env import build_native_teacher_rollout
from mint_common import ARTIFACT_DIR
from physics_legality import (
    check_physics_legality,
    physics_constrained_teacher_rollout,
    stage0_check_grasp_feasibility,
    DEFAULT_PHYSICS_THRESHOLDS,
)

# Paths
ARTIFACT = ARTIFACT_DIR / "p4_physics_legal_gate.json"
OUTPUT_DIR = ARTIFACT_DIR / "p4_physics_legal_rollouts"
GRASP_CACHE = ARTIFACT_DIR / "g3_anygrasp_grasps"

# Gate parameters
MIN_LEGAL = 6  # gate threshold: must have ≥ this many physics-legal rollouts
MAX_SEEDS_TO_TRY = 15  # [1..10] train + [11..15] held-out = 15 seeds total

# Branch multiplier: how many episodes per (seed, branch) to try
# Branch configs define episodes_per_seed internally; we use 1 per combo here
# to keep runtime bounded (full multi-episode sweep can be done in a separate run)
EPISODES_PER_COMBO = 1


def _seed_list() -> list[int]:
    seeds = list(range(1, 11))  # train seeds 1-10
    if MAX_SEEDS_TO_TRY > 10:
        seeds += list(range(11, MAX_SEEDS_TO_TRY + 1))  # held-out 11-15
    return seeds


def _load_grasp_pose(seed: int) -> dict[str, Any] | None:
    gf = GRASP_CACHE / f"seed_{seed:03d}.json"
    if not gf.exists():
        return None
    try:
        data = json.loads(gf.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    sel = data.get("selected_grasp")
    if not sel:
        top = data.get("top_candidates", [])
        sel = top[0] if top else None
    return sel


def _save_rollout(
    seed: int,
    branch_id: str,
    episode_index: int,
    rollout: dict[str, Any],
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"seed_{seed:03d}_{branch_id}_ep{episode_index}.npz"
    np.savez_compressed(
        path,
        # Core trajectory arrays
        states=np.asarray(rollout["states"], dtype=np.float32),
        actions=np.asarray(rollout["actions"], dtype=np.float32),
        images=np.asarray(rollout["images"], dtype=np.uint8),
        images2=np.asarray(_to_list(rollout.get("images2")), dtype=np.uint8),
        gripper_values=np.asarray(_to_list(rollout.get("gripper_values")), dtype=np.float32),
        eef_positions=np.asarray(_to_list(rollout.get("eef_positions")), dtype=np.float32),
        absolute_drawer_fraction=np.asarray(
            _to_list(rollout.get("absolute_drawer_fraction")), dtype=np.float32
        ),
        # Metadata
        seed=np.int32(seed),
        branch_id=branch_id,
        episode_index=np.int32(episode_index),
        success=np.bool_(rollout.get("success", False)),
        final_drawer_fraction=np.float32(rollout.get("final_drawer_fraction", 0.0)),
        max_drawer_fraction=np.float32(rollout.get("max_drawer_fraction", 0.0)),
        ever_attached=np.bool_(rollout.get("ever_attached", False)),
        # Optional traces
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


def _to_list(val: Any) -> list:
    """Safe conversion: handles None, list, np.ndarray without ambiguous bool checks."""
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return val
    return list(val)


def _legal_key(seed: int, branch_id: str, episode_index: int) -> str:
    return f"seed_{seed:03d}_{branch_id}_ep{episode_index}"


def _frames_to_rollout_dict(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Convert frames from physics_constrained_teacher_rollout to the same dict format
    as build_native_teacher_rollout, so the rest of the gate code is uniform.
    """
    T = len(frames)
    states = np.array([f["state"] for f in frames], dtype=np.float32) if T > 0 else np.zeros((0, 8), dtype=np.float32)
    actions = np.array([f["action"] for f in frames], dtype=np.float32) if T > 0 else np.zeros((0, 7), dtype=np.float32)
    drawer_fracs = np.array([f.get("drawer_fraction", 0.0) for f in frames], dtype=np.float32)

    return {
        "states": states,
        "actions": actions,
        "images": [],   # no images captured during physics_constrained generation
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
    records: list[dict[str, Any]] = []
    legal_keys: set[str] = set()
    legal_seed_set: set[int] = set()  # track unique seeds with ≥1 legal rollout
    legal_records: list[dict[str, Any]] = []

    print(f"[P4] seeds={seeds}  branches={[b['branch_id'] for b in branches]}")
    print(f"[P4] MIN_LEGAL={MIN_LEGAL}  EPISODES_PER_COMBO={EPISODES_PER_COMBO}")

    for seed in seeds:
        grasp = _load_grasp_pose(seed)
        if grasp is None:
            msg = f"no grasp plan for seed {seed}"
            print(f"  [skip] seed={seed:03d}  reason={msg}")
            records.append({"seed": seed, "branch": None, "passed": False, "error": msg})
            continue

        pose_world = np.array(grasp["pose_world"], dtype=np.float32)

        # Build anygrasp_candidates: primary grasp + top_candidates (up to 5)
        anygrasp_candidates = [pose_world]
        for top in grasp.get("top_candidates", [])[:4]:
            anygrasp_candidates.append(np.array(top["pose_world"], dtype=np.float32))

        for branch in branches:
            bid = branch["branch_id"]
            for ep_idx in range(EPISODES_PER_COMBO):
                rollout = None
                method = "unknown"

                # ── Phase 1: Physics-constrained compliant controller ─────────
                # Uses impedance control + grasp feasibility filter. This is the
                # correct generation path — not the old world-frame delta controller.
                try:
                    result = physics_constrained_teacher_rollout(
                        seed=seed,
                        anygrasp_candidates=anygrasp_candidates,
                        max_retry=3,   # retry different grasp candidates
                        verbose=False,
                    )
                    method = "physics_constrained"
                    legality = result.get("legality", {})
                    if legality.get("legal"):
                        # Extract rollout data from physics_constrained output
                        frames = result.get("rollout_frames", [])
                        if frames:
                            rollout = _frames_to_rollout_dict(frames)
                            rollout["success"] = legality.get("robot_success", True)
                            rollout["final_drawer_fraction"] = legality.get("final_drawer_fraction", 0.0)
                            rollout["max_drawer_fraction"] = legality.get("max_drawer_fraction", 0.0)
                            rollout["ever_attached"] = bool(np.any([
                                f.get("collision", {}).get("attached", False) for f in frames
                            ]))
                        else:
                            rollout = None
                except Exception as exc:
                    result = None
                    legality = {"legal": False, "issues": [f"physics_constrained exception: {exc}"]}

                # ── Phase 2: Fallback to build_native_teacher_rollout() ───────
                # Search across branches for any physics-legal trajectory.
                # Even if Phase 1 fails, a branch variant might succeed.
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
                        legality = check_physics_legality(rollout, DEFAULT_PHYSICS_THRESHOLDS)
                    except Exception as exc:
                        msg = f"{type(exc).__name__}: {exc}"
                        print(f"  [error] seed={seed:03d}  branch={bid}  ep={ep_idx}  {msg}")
                        records.append({
                            "seed": seed, "branch": bid, "episode": ep_idx,
                            "passed": False, "error": msg,
                        })
                        continue

                robot_success = bool(rollout.get("success", False))
                physics_ok = bool(legality.get("legal", False))
                passed = robot_success and physics_ok

                record = {
                    "seed": seed,
                    "branch": bid,
                    "episode": ep_idx,
                    "passed": passed,
                    "method": method,
                    "robot_success": robot_success,
                    "physics_legal": physics_ok,
                    "final_drawer_fraction": float(rollout.get("final_drawer_fraction", 0.0)),
                    "max_drawer_fraction": float(rollout.get("max_drawer_fraction", 0.0)),
                    "ever_attached": bool(rollout.get("ever_attached", False)),
                    "issues": legality.get("issues", []),
                    "metrics": legality.get("metrics", {}),
                }

                if passed:
                    lkey = _legal_key(seed, bid, ep_idx)
                    legal_keys.add(lkey)
                    legal_seed_set.add(seed)   # track unique seeds
                    path = _save_rollout(seed, bid, ep_idx, rollout)
                    record["rollout_path"] = str(path)
                    legal_records.append(record)
                    print(
                        f"  [legal:{method}] seed={seed:03d}  branch={bid}  ep={ep_idx}"
                        f"  drawer={record['max_drawer_fraction']:.2f}"
                    )
                    if len(legal_seed_set) >= MIN_LEGAL:
                        print(f"[P4] Reached MIN_LEGAL={MIN_LEGAL} — stopping search")
                        break
                else:
                    issues_str = ", ".join(record["issues"][:2]) if record["issues"] else "none"
                    print(
                        f"  [reject:{method}] seed={seed:03d}  branch={bid}  ep={ep_idx}"
                        f"  robot_ok={robot_success}  physics_ok={physics_ok}"
                        f"  issues=[{issues_str[:80]}]"
                    )

                records.append(record)

            if len(legal_seed_set) >= MIN_LEGAL:
                break
        if len(legal_seed_set) >= MIN_LEGAL:
            break

    elapsed_s = time.monotonic() - t0
    unique_seeds = sorted({
        r["seed"] for r in legal_records
    })
    passed = len(unique_seeds) >= MIN_LEGAL

    result = {
        "gate": "p4_physics_legal_gate",
        "passed": passed,
        "min_required": MIN_LEGAL,
        "legal_rollout_count": len(legal_keys),
        "legal_seed_count": len(unique_seeds),
        "legal_seeds": unique_seeds,
        "legal_records": legal_records,
        "all_records": records,
        "seeds_tried": seeds,
        "branches_tried": [b["branch_id"] for b in branches],
        "elapsed_s": round(elapsed_s, 1),
        "thresholds_used": {
            k: float(v) for k, v in DEFAULT_PHYSICS_THRESHOLDS.items()
        },
        "timestamp": time.time(),
    }

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    verdict = "PASS" if passed else "FAIL"
    print(
        f"\n[P4] {verdict}: {len(unique_seeds)}/{MIN_LEGAL} physics-legal seeds"
        f"  ({len(legal_keys)} rollouts total)  elapsed={elapsed_s:.0f}s"
    )
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

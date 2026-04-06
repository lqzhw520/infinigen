#!/usr/bin/env python3
"""
Gate B: Teacher Replayability Audit
====================================
Verifies: Can teacher actions (successful in source sim) replay successfully in DrawerRobotEnv?

Protocol:
  1. Load top-10 episodes by action_std from v58 NPZ source files
  2. Run DrawerRobotEnv replay in TWO modes:
     - open_loop:    reset to HOME_JOINTS, feed actions directly
     - state_anchored: restore source sim state (joints, drawer, gripper) each step
  3. Report: attach_rate, success_rate, tracking errors vs source sim

Decision tree:
  - attach_rate > 0 in BOTH modes → teacher replay WORKS; problem is policy learning
  - attach_rate = 0 in BOTH modes → simulator physics is the primary bottleneck
  - attach_rate > 0 in state_anchored but 0 in open_loop → initial state mismatch is key
  - attach_rate > 0 in open_loop but 0 in state_anchored → state restoration is broken

Top-10 candidates (from E027, ranked by action_std):
  157 (0.324), 205 (0.319), 25 (0.316), 180 (0.315), 3 (0.314),
  181 (0.313), 158 (0.312), 182 (0.312), 138 (0.312), 209 (0.311)

NPZ sources:
  - v58_physics_legal_rollouts/ (episode 0-226)
  - p4_physics_legal_rollouts/  (episode 227-239)
  Mapping confirmed: sorted by success=True NPZ order = episode_index 0-239
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN = PROJECT_ROOT / "experiments/mint/mint_drawer_v1"
ARTIFACT_DIR = CAMPAIGN / "artifacts"
SOVEREIGN_DIR = CAMPAIGN / "sovereign"
SCRIPTS_MINT = PROJECT_ROOT / "scripts" / "mint"

sys.path.insert(0, str(SCRIPTS_MINT))
sys.path.insert(0, str(PROJECT_ROOT))

from drawer_robot_env import (
    DrawerRobotEnv,
    ARM_DOF,
    GRIPPER_JOINTS,
    HOME_JOINTS,
    HANDLE_ATTACH_THRESHOLD,
)

# ── Constants ────────────────────────────────────────────────────────────────

V58_ROLLOUTS = ARTIFACT_DIR / "v58_physics_legal_rollouts"
P4_ROLLOUTS = ARTIFACT_DIR / "p4_physics_legal_rollouts"
OUTPUT_DIR = ARTIFACT_DIR / "gate_b_teacher_replayability"
EVIDENCE_OUT = SOVEREIGN_DIR / "evidence" / "E028.yaml"

# Top-10 episode candidates from E027 (ranked by action_std descending)
TOP10 = [
    (157, "seed_013_branch2_phase_scaled_ep0.npz"),
    (205, "seed_015_branch2_phase_scaled_ep0.npz"),
    (25,  "seed_008_branch2_phase_scaled_ep0.npz"),
    (180, "seed_014_branch2_phase_scaled_ep1.npz"),
    (3,   "seed_007_branch2_phase_scaled_ep3.npz"),
    (181, "seed_014_branch2_phase_scaled_ep3.npz"),
    (158, "seed_013_branch2_phase_scaled_ep2.npz"),
    (182, "seed_014_branch2_phase_scaled_ep4.npz"),
    (138, "seed_012_branch2_phase_scaled_ep4.npz"),
    (209, "seed_015_branch2_phase_scaled_ep4.npz"),
]

DEFAULT_ACTION_CONTRACT = {
    "action_frame": "world",
    "translation_scale_m": 0.03,
    "rotation_scale_rad": 0.25,
    "attach_threshold_m": 0.050,
}

MAX_STEPS_OVERRIDE = 200  # enough for full episode + buffer


# ── Helper: resolve NPZ path ─────────────────────────────────────────────────

def resolve_npz(npz_name: str) -> Path:
    p = V58_ROLLOUTS / npz_name
    if p.exists():
        return p
    p = P4_ROLLOUTS / npz_name
    if p.exists():
        return p
    raise FileNotFoundError(f"NPZ not found: {npz_name}")


# ── Core: single-episode replay in one mode ──────────────────────────────────

def replay_episode(
    npz_data: dict,
    seed: int,
    replay_mode: str,
    action_contract: dict | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """
    Replay one episode in DrawerRobotEnv.
    Handles missing grasp_pose_local_{pos,quat} gracefully.

    Returns per-step records and summary metrics.
    """
    contract = dict(DEFAULT_ACTION_CONTRACT)
    if action_contract:
        contract.update(action_contract)

    actions = npz_data["actions"].astype(np.float32)
    n_frames = len(actions)
    max_s = max_steps or min(n_frames + 20, MAX_STEPS_OVERRIDE)

    env = DrawerRobotEnv(
        seed=seed,
        image_size=224,
        max_steps=max_s,
        action_contract=contract,
    )

    records = {
        "replay_eef_pos": [],
        "replay_eef_quat": [],
        "replay_drawer": [],
        "replay_attached": [],
        "replay_handle_distance": [],
        "replay_action": [],
    }
    success = False
    done = False

    try:
        obs = env.reset()

        if replay_mode == "state_anchored":
            _restore_state(env, npz_data, step_index=0)

        for step_i in range(max_s):
            action = actions[step_i] if step_i < n_frames else np.zeros(7, np.float32)
            obs, _reward, done, info = env.step(action)

            eef_pos = np.array(env.p.getLinkState(
                env.robot_id, env.ee_link,
                computeForwardKinematics=True,
                physicsClientId=env.client,
            )[4], dtype=np.float32)
            _, eef_quat_raw = env.eef_pose()
            eef_quat = np.array(eef_quat_raw, dtype=np.float32)

            records["replay_eef_pos"].append(eef_pos.copy())
            records["replay_eef_quat"].append(eef_quat.copy())
            records["replay_drawer"].append(float(obs.drawer_fraction))
            records["replay_attached"].append(bool(info.get("attached", False)))
            records["replay_action"].append(action.copy())

            # Handle distance: compare EEF to drawer handle
            # (attachment_local is computed by DrawerRobotEnv.reset() from URDF)
            if env.attachment_local is not None:
                h_pos, _ = env.local_to_world_pose(
                    env.attachment_local["pos"],
                    env.attachment_local["quat"],
                )
                h_dist = float(np.linalg.norm(eef_pos - h_pos))
            else:
                h_dist = float("nan")
            records["replay_handle_distance"].append(h_dist)

            if info.get("attached") and env.attach_step is None:
                env.attach_step = step_i
            if done:
                success = bool(info.get("is_success", False))
                break
            if step_i >= n_frames - 1:
                break
            if replay_mode == "state_anchored":
                _restore_state(env, npz_data, step_index=step_i + 1)

    finally:
        env.close()

    # Convert to arrays
    for k, v in records.items():
        records[k] = np.asarray(v, dtype=np.float32) if v else np.zeros((0,))

    # Source truth
    src_eef = npz_data["eef_positions"].astype(np.float32)
    src_drawer = npz_data["absolute_drawer_fraction"].astype(np.float32)
    src_attached = npz_data["attached_trace"].astype(np.bool_)
    src_dist = npz_data["handle_distance_trace"].astype(np.float32)

    n_replay = len(records["replay_eef_pos"])
    n_compare = min(n_replay, n_frames)  # compare up to min(replay_steps, source_actions)

    # Position error vs source EEF (up to n_compare steps)
    if n_compare > 0:
        pos_error = np.linalg.norm(
            records["replay_eef_pos"][:n_compare] - src_eef[:n_compare], axis=1
        )
        drawer_error = np.abs(records["replay_drawer"][:n_compare] - src_drawer[:n_compare])
        dist_error = np.abs(
            records["replay_handle_distance"][:n_compare] -
            src_dist[:n_compare]
        )
        dist_agreement = np.mean(
            (records["replay_attached"][:n_compare] == src_attached[:n_compare]).astype(np.float32)
        )
    else:
        pos_error = np.zeros((0,))
        drawer_error = np.zeros((0,))
        dist_error = np.zeros((0,))
        dist_agreement = 0.0

    metrics = {
        "replay_steps": n_replay,
        "source_steps": n_frames,
        "max_position_error_m": float(np.max(pos_error)) if len(pos_error) else 0.0,
        "mean_position_error_m": float(np.mean(pos_error)) if len(pos_error) else 0.0,
        "p95_position_error_m": float(np.percentile(pos_error, 95)) if len(pos_error) else 0.0,
        "max_drawer_error": float(np.max(drawer_error)) if len(drawer_error) else 0.0,
        "mean_drawer_error": float(np.mean(drawer_error)) if len(drawer_error) else 0.0,
        "max_handle_distance_m": float(np.nanmax(records["replay_handle_distance"])) if len(records["replay_handle_distance"]) else float("nan"),
        "mean_handle_distance_m": float(np.nanmean(records["replay_handle_distance"])) if len(records["replay_handle_distance"]) else float("nan"),
        "handle_distance_vs_source_error": float(np.nanmean(dist_error)) if len(dist_error) and not np.all(np.isnan(dist_error)) else float("nan"),
        "attached_agreement": float(dist_agreement),
        "success": bool(success),
        "ever_attached_replay": bool(np.any(records["replay_attached"])),
        "ever_attached_source": bool(np.any(src_attached)),
        "attach_step_replay": int(env.attach_step) if env.attach_step is not None else None,
        "final_drawer_fraction": float(records["replay_drawer"][-1]) if len(records["replay_drawer"]) else 0.0,
        "max_drawer_fraction": float(np.max(records["replay_drawer"])) if len(records["replay_drawer"]) else 0.0,
        "replay_mode": replay_mode,
    }

    return records, metrics


# ── Helper: restore source sim state (state_anchored mode) ───────────────────

def _restore_state(env: DrawerRobotEnv, source: dict, step_index: int) -> None:
    """Restore source simulation state at a given step_index."""
    joint_pos = source["joint_positions"].astype(np.float32)
    curr_drawer = source["absolute_drawer_fraction"].astype(np.float32)
    curr_gripper = source["gripper_values"].astype(np.float32)
    attached_trace = source["attached_trace"].astype(np.bool_)

    # Arm joints: step 0 starts from HOME_JOINTS; step t uses joint_pos[t-1]
    if step_index == 0:
        j_state = np.asarray(HOME_JOINTS, dtype=np.float32)
    else:
        j_state = joint_pos[min(step_index - 1, len(joint_pos) - 1)]

    for idx, val in zip(env.arm_joint_indices, j_state.tolist()):
        env.p.resetJointState(
            env.robot_id, idx, float(val),
            targetVelocity=0.0, physicsClientId=env.client,
        )

    # Drawer
    if step_index < len(curr_drawer):
        df = float(curr_drawer[step_index])
        dq = env.drawer_low + float(np.clip(df, 0.0, 1.0)) * (
            env.drawer_high - env.drawer_low
        )
        env.p.resetJointState(
            env.drawer_id, env.drawer_joint, dq,
            targetVelocity=0.0, physicsClientId=env.client,
        )

    # Gripper
    if step_index < len(curr_gripper):
        env._set_gripper_open(float(curr_gripper[step_index]))

    env._step_world(2)

    # Restore attach state
    prev_attached = (
        bool(attached_trace[step_index - 1])
        if step_index > 0 and step_index - 1 < len(attached_trace)
        else False
    )
    env.attached = prev_attached
    env.ever_attached = (
        bool(np.any(attached_trace[:step_index])) if step_index > 0 else False
    )
    env.step_count = int(step_index)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    summary_by_mode = {"open_loop": [], "state_anchored": []}

    for ep_idx, npz_name in TOP10:
        npz_path = resolve_npz(npz_name)
        npz_data = dict(np.load(npz_path, allow_pickle=True))

        ep_result = {
            "episode_index": int(ep_idx),
            "npz_name": npz_name,
            "seed": int(npz_data["seed"]),
            "n_frames": len(npz_data["actions"]),
            "source_success": bool(npz_data["success"]),
            "source_ever_attached": bool(npz_data["ever_attached"]),
            "source_final_drawer": float(npz_data["final_drawer_fraction"]),
            "source_max_drawer": float(npz_data["max_drawer_fraction"]),
            "source_handle_dist_mean": float(np.nanmean(npz_data["handle_distance_trace"])),
            "source_handle_dist_min": float(np.nanmin(npz_data["handle_distance_trace"])),
        }

        # Run both modes
        for mode in ["open_loop", "state_anchored"]:
            records, metrics = replay_episode(
                npz_data=npz_data,
                seed=int(npz_data["seed"]),
                replay_mode=mode,
                max_steps=MAX_STEPS_OVERRIDE,
            )
            ep_result[f"replay_{mode}"] = metrics
            summary_by_mode[mode].append(metrics)

            # Save per-episode JSON
            ep_json_path = OUTPUT_DIR / f"ep{ep_idx:03d}_{mode}.json"
            with open(ep_json_path, "w") as f:
                json.dump({**ep_result, f"metrics_{mode}": metrics}, f, indent=2, default=str)

        # Summary line
        ol = ep_result["replay_open_loop"]
        sa = ep_result["replay_state_anchored"]
        print(
            f"  ep{ep_idx:3d} | "
            f"src_att={int(ep_result['source_ever_attached'])} "
            f"src_drw={ep_result['source_final_drawer']:.2f} | "
            f"OL_att={int(ol['ever_attached_replay'])} "
            f"OL_drw={ol['final_drawer_fraction']:.2f} "
            f"OL_err={ol['mean_position_error_m']:.3f}m | "
            f"SA_att={int(sa['ever_attached_replay'])} "
            f"SA_drw={sa['final_drawer_fraction']:.2f} "
            f"SA_err={sa['mean_position_error_m']:.3f}m"
        )

        all_results.append(ep_result)

    elapsed = time.time() - t0

    # ── Aggregate summary ────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"Gate B: Teacher Replayability Audit — COMPLETED ({elapsed:.1f}s)")

    for mode, metrics_list in summary_by_mode.items():
        n = len(metrics_list)
        attach_rate = sum(1 for m in metrics_list if m["ever_attached_replay"]) / n
        success_rate = sum(1 for m in metrics_list if m["success"]) / n
        mean_pos_err = np.mean([m["mean_position_error_m"] for m in metrics_list])
        mean_drawer_err = np.mean([m["mean_drawer_error"] for m in metrics_list])
        mean_h_dist = np.nanmean([m["mean_handle_distance_m"] for m in metrics_list
                                  if not np.isnan(m["mean_handle_distance_m"])])
        src_attach_rate = sum(1 for m in metrics_list if m["ever_attached_source"]) / n

        print(f"\n  {mode.upper()} ({n} episodes):")
        print(f"    Source attach rate:     {src_attach_rate:.1%}")
        print(f"    Replay attach rate:     {attach_rate:.1%}")
        print(f"    Replay success rate:    {success_rate:.1%}")
        print(f"    Mean EEF pos error:     {mean_pos_err:.3f} m")
        print(f"    Mean drawer error:      {mean_drawer_err:.3f}")
        print(f"    Mean handle dist:       {mean_h_dist:.3f} m" if not np.isnan(mean_h_dist) else "    Mean handle dist:       N/A")

    # ── Write aggregate results ──────────────────────────────────────────────
    results_path = OUTPUT_DIR / "gate_b_results.json"
    aggregate = {
        "gate": "B",
        "n_episodes": len(all_results),
        "elapsed_sec": elapsed,
        "top10_episodes": [ep_idx for ep_idx, _ in TOP10],
        "by_mode": {},
    }
    for mode, metrics_list in summary_by_mode.items():
        n = len(metrics_list)
        aggregate["by_mode"][mode] = {
            "attach_rate": sum(1 for m in metrics_list if m["ever_attached_replay"]) / n,
            "success_rate": sum(1 for m in metrics_list if m["success"]) / n,
            "source_attach_rate": sum(1 for m in metrics_list if m["ever_attached_source"]) / n,
            "mean_position_error_m": float(np.mean([m["mean_position_error_m"] for m in metrics_list])),
            "p95_position_error_m": float(np.percentile([m["p95_position_error_m"] for m in metrics_list], 90)),
            "max_position_error_m": float(np.max([m["max_position_error_m"] for m in metrics_list])),
            "mean_drawer_error": float(np.mean([m["mean_drawer_error"] for m in metrics_list])),
            "mean_handle_distance_m": float(np.nanmean([m["mean_handle_distance_m"] for m in metrics_list
                                                          if not np.isnan(m["mean_handle_distance_m"])])),
            "handle_distance_vs_source_error": float(np.nanmean([m["handle_distance_vs_source_error"]
                                                                  for m in metrics_list
                                                                  if not np.isnan(m["handle_distance_vs_source_error"])])),
        }

    with open(results_path, "w") as f:
        json.dump(aggregate, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")

    return aggregate


if __name__ == "__main__":
    main()

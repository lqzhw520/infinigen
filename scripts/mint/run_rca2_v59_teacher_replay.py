#!/usr/bin/env python3
"""
run_rca2_v59_teacher_replay.py

RCA2: Verify that the exact actions from V59 overfit episodes (0-5, 49)
replay successfully in DrawerRobotEnv.

This directly tests whether the teacher actions that achieved 100% success
in the source env also work in DrawerRobotEnv.

SCOPE: Only 7 overfit episodes. Does NOT generalize to full V59 dataset.

Usage:
  python run_rca2_v59_teacher_replay.py [--seed 42]
"""
import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts/mint"))

CAMPAIGN_DIR = PROJECT_ROOT / "experiments/mint/mint_drawer_v1"
ARTIFACT_DIR = CAMPAIGN_DIR / "artifacts"
OUTPUT_FILE = ARTIFACT_DIR / "rca2_v59_teacher_replay.json"

# V59 overfit episodes (from E023 — LeRobot episode indices)
OVERFIT_EPISODES = [0, 1, 2, 3, 4, 5, 49]
# Their NPZ source files (from E023 mapping)
EPISODE_NPZ_MAP = {
    # LeRobot episode index -> NPZ filename
    0: "seed_007_branch1_position_dominant_ep0.npz",
    1: "seed_007_branch2_phase_scaled_ep0.npz",
    2: "seed_007_branch2_phase_scaled_ep2.npz",
    3: "seed_007_branch2_phase_scaled_ep3.npz",
    4: "seed_007_branch2_phase_scaled_ep4.npz",
    5: "seed_007_branch3_explicit_gripper_phases_ep0.npz",
    49: "seed_009_branch2_phase_scaled_ep2.npz",
}
# Map from NPZ filename -> drawer robot seed (from NPZ 'seed' field)
NPZ_ROBOT_SEED = {
    "seed_007_branch1_position_dominant_ep0.npz": 7,
    "seed_007_branch2_phase_scaled_ep0.npz": 7,
    "seed_007_branch2_phase_scaled_ep2.npz": 7,
    "seed_007_branch2_phase_scaled_ep3.npz": 7,
    "seed_007_branch2_phase_scaled_ep4.npz": 7,
    "seed_007_branch3_explicit_gripper_phases_ep0.npz": 7,
    "seed_009_branch2_phase_scaled_ep2.npz": 9,
}

NPZ_DIRS = [
    CAMPAIGN_DIR / "artifacts/v58_physics_legal_rollouts",
    CAMPAIGN_DIR / "artifacts/p4_physics_legal_rollouts",
]


def find_npz(episode_index: int) -> Path | None:
    """Find the NPZ file for a given LeRobot episode index."""
    filename = EPISODE_NPZ_MAP.get(episode_index)
    if filename is None:
        return None
    for d in NPZ_DIRS:
        p = d / filename
        if p.exists():
            return p
    return None


def replay_episode(npz_path: Path, drawer_seed: int, max_steps: int = 200):
    """
    Replay the exact actions from an NPZ in DrawerRobotEnv.
    Uses drawer_seed for env init (determines drawer URDF config).
    Robot home position is FIXED (HOME_JOINTS) regardless of drawer seed.
    """
    npz = np.load(npz_path)

    # Load actions: in [-1, 1] range, IDENTITY normalized
    actions = npz["actions"]  # shape: (T, 7)
    states = npz["states"]    # shape: (T, 8) — [x,y,z,j0,j1,j2,j3,gripper_joint]
    teacher_success = bool(npz.get("success", False))
    teacher_max_drawer = float(npz.get("max_drawer_fraction", 0.0))

    from drawer_robot_env import DrawerRobotEnv

    env = DrawerRobotEnv(
        seed=drawer_seed,
        image_size=224,
        max_steps=max_steps,
    )

    # Restore initial state from NPZ: states[0] = [x,y,z,j0,j1,j2,j3,gripper_joint]
    npz_initial_joints = states[0][3:7]  # 4 arm joints
    for idx, val in zip(env.arm_joint_indices, npz_initial_joints.tolist()):
        env.p.resetJointState(env.robot_id, idx, float(val), physicsClientId=env.client)
    env._set_gripper_open(float(states[0][7]))
    env._step_world(8)

    # ── Compute attachment_local from NPZ data ──────────────────────────────────
    # Strategy: find where gripper CLOSES (action[s][6] goes negative)
    # Use that as the attachment step.
    GRIPPER_LENGTH = 0.10  # Panda fingertip offset from EEF center

    hdt = npz.get("handle_distance_trace", [])
    gripper_actions = actions[:, 6]
    close_steps = [s for s in range(1, len(gripper_actions))
                 if gripper_actions[s-1] >= 0 and gripper_actions[s] < 0]
    attach_step = close_steps[0] if close_steps else (len(hdt) - 1)

    # At attach_step, handle distance should be minimal
    attach_handle_dist = float(hdt[attach_step])

    # Get EEF pose at attach_step from env replay
    for s in range(attach_step):
        env.step(actions[s])

    eef_pos, eef_quat = env.eef_pose()
    R_eef = R.from_quat(np.asarray(eef_quat, dtype=np.float32))
    gripper_z_world = R_eef.apply([0.0, 0.0, -1.0])

    # Handle world position (gripper tip → handle center)
    handle_world = np.asarray(eef_pos, dtype=np.float32) + (
        GRIPPER_LENGTH + attach_handle_dist
    ) * np.asarray(gripper_z_world, dtype=np.float32)

    # Convert to drawer link frame for attachment_local
    drawer_link_pos, drawer_link_quat = env._drawer_link_pose()
    R_drawer = R.from_quat(np.asarray(drawer_link_quat, dtype=np.float32))
    att_local_pos = R_drawer.inv().apply(handle_world - np.asarray(drawer_link_pos, dtype=np.float32)).astype(np.float32)
    att_local_quat = (R_drawer.inv() * R_eef).as_quat().astype(np.float32)

    env.attachment_local = {
        "pos": att_local_pos,
        "quat": att_local_quat,
    }
    # ── End attachment_local computation ─────────────────────────────────────────

    results = {
        "npz_path": str(npz_path),
        "drawer_seed": drawer_seed,
        "n_actions": len(actions),
        "teacher_success": teacher_success,
        "teacher_max_drawer": teacher_max_drawer,
        "replay_steps": 0,
        "success": False,
        "ever_attached": False,
        "attach_step": None,
        "max_drawer_fraction": 0.0,
        "final_drawer_fraction": 0.0,
        "final_handle_distance": None,
        "phase_sequence": [],
        "error": None,
    }

    try:
        # Step through replaying actions
        n_steps = min(len(actions), max_steps)
        last_info = None
        for step in range(n_steps):
            action = actions[step]
            obs, reward, terminated, last_info = env.step(action)
            drawer_frac = float(last_info.get("drawer_fraction", 0.0))

            if not results["ever_attached"] and last_info.get("attached"):
                results["ever_attached"] = True
                results["attach_step"] = step

            results["replay_steps"] = step + 1
            results["max_drawer_fraction"] = max(results["max_drawer_fraction"], drawer_frac)

            if terminated:
                break

        if last_info is not None:
            results["success"] = bool(last_info.get("is_success", False))
            results["final_drawer_fraction"] = float(last_info.get("drawer_fraction", 0.0))

    except Exception as e:
        results["error"] = str(e)

    finally:
        try:
            env.close()
        except Exception:
            pass

    return results


def main():
    parser = argparse.ArgumentParser(description="RCA2 V59 teacher action replay")
    parser.add_argument("--seed", type=int, default=42, help="Base env seed")
    parser.add_argument("--max-steps", type=int, default=300, help="Max env steps")
    parser.add_argument("--dry-run", action="store_true", help="Just list files")
    args = parser.parse_args()

    print(f"=== RCA2 V59 Teacher Replay ===")
    print(f"Episodes: {OVERFIT_EPISODES}")
    print(f"Base seed: {args.seed}")
    print()

    all_results = []

    for i, ep_idx in enumerate(OVERFIT_EPISODES):
        npz_path = find_npz(ep_idx)
        drawer_seed = NPZ_ROBOT_SEED.get(npz_path.name if npz_path else "", 7)

        print(f"[{i+1}/{len(OVERFIT_EPISODES)}] Episode {ep_idx}: ", end="")

        if npz_path is None or not npz_path.exists():
            print(f"NOT FOUND (expected {EPISODE_NPZ_MAP.get(ep_idx, 'unknown')})")
            all_results.append({"episode_index": ep_idx, "error": "NPZ not found"})
            continue

        print(f"NPZ: {npz_path.name} | drawer_seed={drawer_seed}")

        result = replay_episode(npz_path, drawer_seed, args.max_steps)

        # Compute if this matches teacher success
        npz = np.load(npz_path)
        teacher_success = bool(npz.get("success", False))
        teacher_max_drawer = float(npz.get("max_drawer_fraction", 0.0))

        result["episode_index"] = ep_idx
        result["teacher_success"] = teacher_success
        result["teacher_max_drawer"] = teacher_max_drawer
        result["replay_matches_teacher"] = (
            result["success"] == teacher_success
            and abs(result["max_drawer_fraction"] - teacher_max_drawer) < 0.01
        )

        print(f"  Teacher: success={teacher_success}, max_drawer={teacher_max_drawer:.3f}")
        print(f"  Replay:  success={result['success']}, max_drawer={result['max_drawer_fraction']:.3f}, "
              f"attached={result['ever_attached']}")
        if result.get("error"):
            print(f"  ERROR: {result['error']}")

        all_results.append(result)

    # Summary
    n_success = sum(1 for r in all_results if r.get("success") and not r.get("error"))
    n_ever_attached = sum(1 for r in all_results if r.get("ever_attached") and not r.get("error"))
    n_errors = sum(1 for r in all_results if r.get("error"))

    print()
    print(f"=== Summary ===")
    print(f"Episodes: {len(all_results)}")
    print(f"Replayed successfully: {n_success}/{len(all_results)}")
    print(f"Ever attached: {n_ever_attached}/{len(all_results)}")
    print(f"Errors: {n_errors}")

    if n_errors > 0:
        print(f"Errors: {[r['episode_index'] for r in all_results if r.get('error')]}")

    # Compute RCA2 verdict
    if n_errors == len(all_results):
        verdict = "ERROR_ALL"
        root_cause = "REPLAY_ERROR"
    elif n_ever_attached == 0:
        verdict = "ACTION_MISMATCH"
        root_cause = "ACTION_CONTRACT_MISMATCH"
    elif n_success == 0:
        verdict = "PARTIAL_ATTACH_NO_DRAWER"
        root_cause = "ACTION_MISMATCH_OR_STATE_MISMATCH"
    elif n_success < len(all_results):
        verdict = "MIXED"
        root_cause = "PARTIAL_MATCH"
    else:
        verdict = "PERFECT_MATCH"
        root_cause = "ELIMINATED_ACTION_NORMALIZATION"

    output = {
        "rca_id": "RCA2",
        "experiment": "v59_teacher_replay",
        "scope": "overfit_subset_7_episodes",
        "scope_guardrail": "Conclusions apply ONLY to 7 overfit episodes. Do NOT extrapolate to full V59.",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        "parameters": {
            "overfit_episodes": OVERFIT_EPISODES,
            "base_seed": args.seed,
            "max_steps": args.max_steps,
        },
        "results": all_results,
        "summary": {
            "n_total": len(all_results),
            "n_replay_success": n_success,
            "n_ever_attached": n_ever_attached,
            "n_errors": n_errors,
        },
        "verdict": verdict,
        "root_cause": root_cause,
        "rca2_eliminates_action_normalization": root_cause == "ELIMINATED_ACTION_NORMALIZATION",
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved: {OUTPUT_FILE}")
    print(f"Verdict: {verdict}")
    print(f"Root cause: {root_cause}")

    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

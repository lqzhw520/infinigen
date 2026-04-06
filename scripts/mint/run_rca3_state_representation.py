#!/usr/bin/env python3
"""
run_rca3_state_representation.py

RCA3: Verify state representation compatibility between V59 dataset
(LeRobot parquet) and DrawerRobotEnv._state_vector().

Core questions:
  Q1: Does DrawerRobotEnv._state_vector() produce the same values as
      the dataset states for identical robot configurations?
  Q2: Is state[3:7] (motor joints) in the same range as the MINT tokenizer expects?
  Q3: Is gripper state[7] truly continuous in the dataset (vs binary)?

SCOPE: 7 overfit episodes (LeRobot indices 0-5, 49).
      Conclusions do NOT generalize to full V59.

Key finding from initial analysis:
  - NPZ "gripper_values" array is binary [0, 1] — raw PyBullet finger positions
  - But parquet observation.state[7] is continuous ∈ [-0.042, +0.001] (LIBERO-normalized)
  - This means dataset_builder does the clipping correctly; parquet is ground truth

Usage:
  python run_rca3_state_representation.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts/mint"))

CAMPAIGN_DIR = PROJECT_ROOT / "experiments/mint/mint_drawer_v1"
ARTIFACT_DIR = CAMPAIGN_DIR / "artifacts"
PARQUET_PATH = CAMPAIGN_DIR / "dataset/data/chunk-000/file-000.parquet"
OUTPUT_FILE = ARTIFACT_DIR / "rca3_state_representation.json"

# V59 overfit episodes (LeRobot indices)
OVERFIT_EPISODES = [0, 1, 2, 3, 4, 5, 49]

# Corresponding drawer seeds
EPISODE_SEED = {
    0: 7, 1: 7, 2: 7, 3: 7, 4: 7, 5: 7, 49: 9,
}


def load_parquet_states(episode_index: int) -> dict | None:
    """Load state vectors for a given episode from LeRobot parquet."""
    if not PARQUET_PATH.exists():
        return None
    df = pd.read_parquet(PARQUET_PATH)
    ep_df = df[df["episode_index"] == episode_index].sort_index()
    if len(ep_df) == 0:
        return None
    states = np.stack(ep_df["observation.state"].values)
    actions = np.stack(ep_df["action"].values)
    return {
        "states": states,          # shape (T, 8): [eef_x, eef_y, eef_z, j0, j1, j2, j3, gripper_joint]
        "actions": actions,        # shape (T, 7): delta actions
        "n_frames": len(states),
    }


def verify_dataset_state_continuity(data: dict) -> dict:
    """Analyze if state[7] (gripper) is continuous vs binary in the dataset."""
    states = data["states"]
    state7 = states[:, 7]
    unique_vals = np.unique(state7)
    return {
        "n_frames": len(states),
        "n_unique_state7": int(len(unique_vals)),
        "state7_unique_sample": sorted([round(float(v), 6) for v in unique_vals[:20]]),
        "state7_min": float(state7.min()),
        "state7_max": float(state7.max()),
        "state7_mean": float(state7.mean()),
        "state7_std": float(state7.std()),
        "state7_is_binary": bool(len(unique_vals) <= 2),
        "state7_libero_bounds": [-0.042, 0.001],
        "state7_in_libero_bounds": bool(
            state7.min() >= -0.042 and state7.max() <= 0.001
        ),
    }


def verify_env_state_match(data: dict, drawer_seed: int) -> dict:
    """
    Compare dataset state[0:3] (EEF pos) with DrawerRobotEnv state at HOME_JOINTS.
    Dataset states are from LeRobot parquet — verified continuous.
    We check: does _state_vector() match parquet states at equivalent configs?
    """
    from drawer_robot_env import DrawerRobotEnv

    states = data["states"]
    env = DrawerRobotEnv(seed=drawer_seed, max_steps=300)
    env_obs = env.reset()
    env_state = env._state_vector()

    # Dataset initial state vs env initial state
    ds_init = states[0]
    env_init = env_state

    result = {
        "drawer_seed": drawer_seed,
        # EEF position comparison
        "ds_init_eef": [float(x) for x in ds_init[:3]],
        "env_init_eef": [float(x) for x in env_init[:3]],
        "eef_diff_m": float(np.linalg.norm(env_init[:3] - ds_init[:3])),
        "eef_match": bool(np.linalg.norm(env_init[:3] - ds_init[:3]) < 0.01),
        # Motor joints comparison
        "ds_init_joints": [float(x) for x in ds_init[3:7]],
        "env_init_joints": [float(x) for x in env_init[3:7]],
        "joints_diff": float(np.linalg.norm(env_init[3:7] - ds_init[3:7])),
        "joints_match": bool(np.linalg.norm(env_init[3:7] - ds_init[3:7]) < 0.05),
        # Gripper comparison
        "ds_init_gripper": float(ds_init[7]),
        "env_init_gripper": float(env_init[7]),
        "gripper_diff": float(abs(env_init[7] - ds_init[7])),
        "gripper_match": bool(abs(env_init[7] - ds_init[7]) < 0.01),
    }

    # Check full state[3:7] range in dataset
    motor_joints = states[:, 3:7]
    result["motor_joints_range"] = {
        dim: {
            "min": float(motor_joints[:, dim].min()),
            "max": float(motor_joints[:, dim].max()),
            "mean": float(motor_joints[:, dim].mean()),
            "std": float(motor_joints[:, dim].std()),
        }
        for dim in range(4)
    }

    env.close()
    return result


def verify_gripper_action_state_gap(data: dict) -> dict:
    """
    Key analysis: gripper action (binary ±1.0) vs gripper state (continuous).

    Risk: If gripper actions are binary but gripper states are continuous,
    the model must learn to predict smooth state transitions from binary action signals.
    This is a representational gap IF the model uses action-conditioned state prediction.
    """
    states = data["states"]
    actions = data["actions"]
    state7 = states[:, 7]
    action6 = actions[:, 6]

    # When does gripper action change?
    action_changes = np.diff(np.sign(action6))
    n_transitions = int(np.sum(np.abs(action_changes) > 0))

    # When does gripper state change?
    state_changes = np.abs(np.diff(state7))
    n_state_transitions = int(np.sum(state_changes > 1e-4))

    # Check if state transitions align with action transitions
    action_transition_steps = np.where(np.abs(action_changes) > 0)[0]
    state_transition_steps = np.where(state_changes > 1e-4)[0]

    # Alignment: do action transitions precede state transitions?
    # (This is expected: action causes state change in next step)
    state7_after_close = state7[1:][action6[:-1] < 0] if (action6 < 0).sum() > 0 else np.array([])
    state7_after_open = state7[1:][action6[:-1] >= 0] if (action6 >= 0).sum() > 0 else np.array([])

    return {
        "action6_unique": sorted([round(float(v), 1) for v in np.unique(action6)]),
        "action6_is_binary": bool(len(np.unique(action6)) <= 2),
        "state7_is_continuous": bool(len(np.unique(state7)) > 10),
        "n_action_transitions": n_transitions,
        "n_state_transitions": n_state_transitions,
        "state_transition_leads_action": bool(n_state_transitions >= n_transitions),
        "gripper_action_state_gap_risk": (
            "LOW" if n_state_transitions >= n_transitions * 0.5
            else "MEDIUM" if n_state_transitions > 0
            else "HIGH"
        ),
        "gap_description": (
            "Gripper actions are binary (±1.0) but gripper state[7] is continuous. "
            "The model predicts state7 from image+state context. The action[6] is used "
            "for action-conditioned prediction during training. "
            "If action[6] is binary but the target state7 is continuous, "
            "the model must learn that binary gripper commands → smooth gripper state changes. "
            "This is learnable IF the dataset contains sufficient gripper-closing frames "
            "with continuous intermediate states (which V59 has)."
        ),
        "close_frames_mean_state7": (
            float(state7_after_close.mean())
            if len(state7_after_close) > 0 else None
        ),
        "open_frames_mean_state7": (
            float(state7_after_open.mean())
            if len(state7_after_open) > 0 else None
        ),
        "open_minus_close": (
            float(state7_after_open.mean() - state7_after_close.mean())
            if (len(state7_after_open) > 0 and len(state7_after_close) > 0) else None
        ),
    }


def main():
    print(f"=== RCA3 State Representation Analysis ===")
    print(f"Episodes: {OVERFIT_EPISODES}")
    print(f"Parquet: {PARQUET_PATH}")
    print()

    all_results = []

    for i, ep_idx in enumerate(OVERFIT_EPISODES):
        print(f"[{i+1}/{len(OVERFIT_EPISODES)}] Episode {ep_idx}: ", end="")

        data = load_parquet_states(ep_idx)
        if data is None:
            print(f"PARQUET NOT FOUND")
            all_results.append({"episode_index": ep_idx, "error": "Parquet not found"})
            continue

        print(f"frames={data['n_frames']}")

        try:
            continuity = verify_dataset_state_continuity(data)
            env_check = verify_env_state_match(data, EPISODE_SEED.get(ep_idx, 7))
            gap_check = verify_gripper_action_state_gap(data)

            result = {
                "episode_index": ep_idx,
                "drawer_seed": EPISODE_SEED.get(ep_idx, 7),
                "n_frames": data["n_frames"],
                "dataset_state_continuity": continuity,
                "env_state_match": env_check,
                "gripper_action_state_gap": gap_check,
            }
            all_results.append(result)

            print(
                f"  state7: {continuity['n_unique_state7']} unique "
                f"(binary={continuity['state7_is_binary']})"
            )
            print(
                f"  Env match: eef={env_check['eef_match']} "
                f"joints={env_check['joints_match']} "
                f"gripper={env_check['gripper_match']}"
            )
            print(
                f"  Action-gap risk: {gap_check['gripper_action_state_gap_risk']}"
            )

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"episode_index": ep_idx, "error": str(e)})

    # Aggregate
    n_errors = sum(1 for r in all_results if r.get("error"))
    n_eef_match = sum(
        1 for r in all_results
        if not r.get("error") and r.get("env_state_match", {}).get("eef_match")
    )
    n_joints_match = sum(
        1 for r in all_results
        if not r.get("error") and r.get("env_state_match", {}).get("joints_match")
    )
    n_gripper_match = sum(
        1 for r in all_results
        if not r.get("error") and r.get("env_state_match", {}).get("gripper_match")
    )
    n_continuous_state7 = sum(
        1 for r in all_results
        if not r.get("error") and not r.get("dataset_state_continuity", {}).get("state7_is_binary")
    )
    gap_risks = [
        r.get("gripper_action_state_gap", {}).get("gripper_action_state_gap_risk", "UNKNOWN")
        for r in all_results if not r.get("error")
    ]
    high_risk_count = sum(1 for r in gap_risks if r == "HIGH")
    low_risk_count = sum(1 for r in gap_risks if r == "LOW")

    n_total_valid = len(all_results) - n_errors

    # Verdict
    if n_errors == len(all_results):
        verdict = "ERROR_ALL"
    elif n_eef_match == n_total_valid and n_joints_match == n_total_valid:
        if high_risk_count == n_total_valid:
            verdict = "STATE_FORMAT_MATCH_GA_INSIGHT"
        else:
            verdict = "STATE_FORMAT_MATCH_ELIMINATED"
    elif n_eef_match >= n_total_valid - 1:
        verdict = "PARTIAL_MATCH_INVESTIGATE"
    else:
        verdict = "STATE_MISMATCH_FOUND"

    output = {
        "rca_id": "RCA3",
        "scope": "overfit_subset_7_episodes",
        "scope_guardrail": (
            "Conclusions apply ONLY to the 7 overfit episodes "
            "(LeRobot indices 0,1,2,3,4,5,49). "
            "Do NOT extrapolate to the full V59 dataset."
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        "results": all_results,
        "summary": {
            "n_total": len(all_results),
            "n_errors": n_errors,
            "n_eef_match": n_eef_match,
            "n_joints_match": n_joints_match,
            "n_gripper_match": n_gripper_match,
            "n_continuous_state7": n_continuous_state7,
            "n_high_gap_risk": high_risk_count,
            "n_low_gap_risk": low_risk_count,
        },
        "verdict": verdict,
        "rca3_eliminates_state_representation": verdict in (
            "STATE_FORMAT_MATCH_ELIMINATED", "STATE_FORMAT_MATCH_GA_INSIGHT"
        ),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n=== RCA3 Summary ===")
    print(f"Episodes: {len(all_results)}, Errors: {n_errors}")
    print(f"EEF match: {n_eef_match}/{n_total_valid}")
    print(f"Joints match: {n_joints_match}/{n_total_valid}")
    print(f"Gripper match: {n_gripper_match}/{n_total_valid}")
    print(f"Continuous state7: {n_continuous_state7}/{n_total_valid}")
    print(f"Gap risk HIGH: {high_risk_count}/{n_total_valid}")
    print(f"Gap risk LOW: {low_risk_count}/{n_total_valid}")
    print(f"\nVerdict: {verdict}")
    print(f"Results: {OUTPUT_FILE}")

    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Diagnostic script to understand why finetuned model produces identical EEF motion
across all 5 episodes while pretrained model produces varied motion.

Tests three hypotheses:
a) The model produces identical actions each episode (deterministic + identical observations)
b) The model produces different actions but they result in same EEF motion
c) Something else is wrong
"""

import numpy as np
import torch
from pathlib import Path

# Add scripts/mint to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Apply compatibility patches BEFORE importing lerobot/MINT.
from mint_eval_patches import apply as _apply_patches

_apply_patches()

from drawer_robot_env import DrawerRobotEnv
from mint_common import (
    ACTIVE_ACTION_CONTRACT_PATH,
    DATASET_DIR,
    DATASET_REPO_ID,
    load_json,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot_policy_mint.modeling_mint import MINTPolicy

MINT_CKPT = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero"

# Paths from user's request
FINETUNED_PATH = "/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/outputs_d1_seed_002_episode_01_multi_episode_overfit/checkpoints/000600/pretrained_model/"
DATASET_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/dataset_d1_seed_002_episode_01_multi_episode_overfit/")


def _load_policy(path: str, dataset_root: Path, repo_id: str):
    """Load policy with pre/post processors."""
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, revision="main")
    policy = MINTPolicy.from_pretrained(
        path, local_files_only=True, dataset_stats=dataset.meta.stats
    )
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=path, dataset_stats=dataset.meta.stats
    )
    return policy, preprocessor, postprocessor


def _obs_to_batch(obs) -> dict:
    """Convert observation to model batch format."""
    return {
        "observation.images.image": torch.from_numpy(obs.image)
        .permute(2, 0, 1)
        .to(torch.float32)
        / 255.0,
        "observation.images.image2": torch.from_numpy(obs.image2)
        .permute(2, 0, 1)
        .to(torch.float32)
        / 255.0,
        "observation.state": torch.from_numpy(obs.state.astype(np.float32)),
        "task": obs.task,
    }


def run_single_episode(
    env: DrawerRobotEnv,
    policy, pre, post,
    seed: int,
    rng_seed: int,
    max_steps: int = 96,
    reset_policy: bool = True,
) -> dict:
    """Run a single evaluation episode and return detailed diagnostics."""

    # Reset environment
    obs = env.reset()

    # Reset policy internal state (action queue, ensembler)
    if reset_policy:
        policy.reset()

    # Create RNG for potential random components
    rng = np.random.default_rng(rng_seed + seed)

    raw_action_trace = []
    eef_pos_trace = []
    drawer_fraction_trace = []
    attached_trace = []

    # Track initial EEF position
    eef_pos_trace.append(obs.eef_pos.copy())

    for step in range(1, max_steps + 1):
        # Get observation
        batch = _obs_to_batch(obs)
        processed = pre(batch)

        with torch.inference_mode():
            action = policy.select_action(processed)

        action = post(action)
        action = action.squeeze(0).detach().cpu().numpy().astype(np.float32)

        # Track raw actions
        raw_action_trace.append(action.copy())

        # Track EEF position before step
        eef_pos_trace.append(obs.eef_pos.copy())

        # Step environment
        obs, _, done, info = env.step(action)

        # Track traces
        drawer_fraction_trace.append(float(info["drawer_fraction"]))
        attached_trace.append(bool(info.get("attached", False)))

        if done:
            break

    # Compute total EEF motion
    eef_arr = np.array(eef_pos_trace)
    step_deltas = np.linalg.norm(np.diff(eef_arr, axis=0), axis=1)
    total_eef_motion = float(np.sum(step_deltas))

    return {
        "seed": seed,
        "rng_seed": rng_seed,
        "raw_action_trace": raw_action_trace,
        "eef_pos_trace": eef_pos_trace,
        "drawer_fraction_trace": drawer_fraction_trace,
        "attached_trace": attached_trace,
        "total_eef_motion": total_eef_motion,
        "n_steps": len(raw_action_trace),
        "final_drawer_fraction": drawer_fraction_trace[-1] if drawer_fraction_trace else 0.0,
        "ever_attached": any(attached_trace),
    }


def test_env_determinism():
    """Test if DrawerRobotEnv.reset is deterministic with seed=2."""
    print("=" * 80)
    print("TEST 0: Environment Determinism Check")
    print("=" * 80)

    results = []
    for i in range(3):
        env = DrawerRobotEnv(seed=2, image_size=224, max_steps=96)
        obs = env.reset()

        initial_eef = obs.eef_pos.copy()
        initial_drawer = obs.drawer_fraction

        results.append({
            "i": i,
            "initial_eef": initial_eef,
            "initial_drawer": initial_drawer,
            "eef_image_hash": hash(obs.image.tobytes()),
            "state_hash": hash(obs.state.tobytes()),
        })

        env.close()

    print("\nEnvironment reset results (seed=2):")
    for r in results:
        print(f"  Run {r['i']}: EEF={r['initial_eef']}, drawer={r['initial_drawer']:.4f}")

    # Check if all are identical
    all_same = all(
        np.allclose(r1["initial_eef"], r2["initial_eef"])
        for r1, r2 in zip(results[:-1], results[1:])
    )
    print(f"\nDeterministic: {all_same}")

    return results


def test_policy_diagnostics(policy_name: str, policy_bundle, n_episodes: int = 5):
    """Run 5 episodes and analyze policy behavior."""
    print("=" * 80)
    print(f"TEST: {policy_name} Policy Diagnostics ({n_episodes} episodes)")
    print("=" * 80)

    policy, pre, post = policy_bundle
    action_contract = load_json(ACTIVE_ACTION_CONTRACT_PATH, {})

    # Test 1: Create ONE env, run all episodes with env.reset() but WITHOUT policy.reset()
    print("\n--- Test 1a: Episodes with env.reset() only (NO policy.reset()) ---")
    env = DrawerRobotEnv(
        seed=2, image_size=224, max_steps=96, action_contract=action_contract
    )

    episodes_no_reset = []
    for attempt_idx in range(n_episodes):
        result = run_single_episode(
            env, policy, pre, post,
            seed=2,
            rng_seed=19 + attempt_idx * 101,
            max_steps=96,
            reset_policy=False,  # DO NOT reset policy
        )
        episodes_no_reset.append(result)
        print(f"  Episode {attempt_idx}: EEF_motion={result['total_eef_motion']:.6f}, "
              f"drawer={result['final_drawer_fraction']:.4f}, "
              f"attached={result['ever_attached']}, "
              f"steps={result['n_steps']}")

    env.close()

    # Test 2: Create ONE env, run all episodes WITH policy.reset() between episodes
    print("\n--- Test 1b: Episodes with env.reset() AND policy.reset() ---")
    env = DrawerRobotEnv(
        seed=2, image_size=224, max_steps=96, action_contract=action_contract
    )

    episodes_with_reset = []
    for attempt_idx in range(n_episodes):
        result = run_single_episode(
            env, policy, pre, post,
            seed=2,
            rng_seed=19 + attempt_idx * 101,
            max_steps=96,
            reset_policy=True,  # RESET policy
        )
        episodes_with_reset.append(result)
        print(f"  Episode {attempt_idx}: EEF_motion={result['total_eef_motion']:.6f}, "
              f"drawer={result['final_drawer_fraction']:.4f}, "
              f"attached={result['ever_attached']}, "
              f"steps={result['n_steps']}")

    env.close()

    # Test 3: NEW env for each episode (both reset)
    print("\n--- Test 2: Fresh env for each episode (both env.reset() and policy.reset()) ---")
    episodes_fresh = []
    for attempt_idx in range(n_episodes):
        env = DrawerRobotEnv(
            seed=2, image_size=224, max_steps=96, action_contract=action_contract
        )
        result = run_single_episode(
            env, policy, pre, post,
            seed=2,
            rng_seed=19 + attempt_idx * 101,
            max_steps=96,
            reset_policy=True,
        )
        episodes_fresh.append(result)
        print(f"  Episode {attempt_idx}: EEF_motion={result['total_eef_motion']:.6f}, "
              f"drawer={result['final_drawer_fraction']:.4f}, "
              f"attached={result['ever_attached']}, "
              f"steps={result['n_steps']}")
        env.close()

    # Analyze action sequences
    print("\n--- Action Sequence Analysis ---")
    for test_name, episodes in [
        ("no_policy_reset", episodes_no_reset),
        ("with_policy_reset", episodes_with_reset),
        ("fresh_env", episodes_fresh),
    ]:
        motion_values = [ep["total_eef_motion"] for ep in episodes]
        motion_std = np.std(motion_values)
        motion_range = max(motion_values) - min(motion_values)

        # Check if actions are identical across episodes
        if len(episodes) > 1:
            first_actions = episodes[0]["raw_action_trace"]
            actions_identical = all(
                np.allclose(ep["raw_action_trace"][:len(first_actions)], first_actions)
                for ep in episodes[1:]
            )
        else:
            actions_identical = True

        print(f"\n  {test_name}:")
        print(f"    Motion values: {[f'{m:.6f}' for m in motion_values]}")
        print(f"    Motion std: {motion_std:.6f}")
        print(f"    Motion range: {motion_range:.6f}")
        print(f"    Actions identical: {actions_identical}")

    # Print first 10 actions from first episode
    print("\n--- First 10 Actions (first episode, fresh env) ---")
    first_ep = episodes_fresh[0]
    for i, action in enumerate(first_ep["raw_action_trace"][:10]):
        print(f"  Step {i+1}: {action}")

    # Check action queue state after first episode
    print(f"\n--- Policy Internal State After First Episode ---")
    print(f"  Action queue length: {len(policy._action_queue)}")
    if hasattr(policy.ensembler, '_action_buffer'):
        print(f"  Ensembler action buffer: {len(policy.ensembler._action_buffer)}")

    return {
        "no_reset": episodes_no_reset,
        "with_reset": episodes_with_reset,
        "fresh_env": episodes_fresh,
    }


def compare_action_sequences(ep1: dict, ep2: dict) -> dict:
    """Compare two episodes' action sequences."""
    n_compare = min(len(ep1["raw_action_trace"]), len(ep2["raw_action_trace"]))

    action_diffs = []
    for i in range(n_compare):
        diff = np.abs(ep1["raw_action_trace"][i] - ep2["raw_action_trace"][i])
        action_diffs.append(diff)

    mean_diff = np.mean(action_diffs, axis=0) if action_diffs else np.zeros(7)
    max_diff = np.max(action_diffs, axis=0) if action_diffs else np.zeros(7)

    return {
        "n_steps_compared": n_compare,
        "mean_action_diff": mean_diff,
        "max_action_diff": max_diff,
        "actions_identical": np.allclose(ep1["raw_action_trace"][:n_compare],
                                         ep2["raw_action_trace"][:n_compare]),
    }


def main():
    print("Loading policies...")

    print("\nLoading FINETUNED policy...")
    finetuned_bundle = _load_policy(FINETUNED_PATH, DATASET_ROOT, DATASET_REPO_ID)

    print("\nLoading PRETRAINED policy...")
    pretrained_bundle = _load_policy(MINT_CKPT, DATASET_ROOT, DATASET_REPO_ID)

    # Test 0: Environment determinism
    test_env_determinism()

    # Test 1: Finetuned model diagnostics
    finetuned_results = test_policy_diagnostics("FINETUNED", finetuned_bundle)

    # Test 2: Pretrained model diagnostics (for comparison)
    pretrained_results = test_policy_diagnostics("PRETRAINED", pretrained_bundle)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nExpected values:")
    print("  Finetuned: all 5 episodes = 7.557735919952393")
    print("  Pretrained: [1.586, 1.644, 10.400, 1.483, 4.901]")

    print("\nObserved values (fresh env for each episode):")
    print("  Finetuned:")
    for i, ep in enumerate(finetuned_results["fresh_env"]):
        print(f"    Episode {i}: {ep['total_eef_motion']:.6f}")

    print("  Pretrained:")
    for i, ep in enumerate(pretrained_results["fresh_env"]):
        print(f"    Episode {i}: {ep['total_eef_motion']:.6f}")

    # Check hypothesis
    print("\n" + "=" * 80)
    print("HYPOTHESIS TESTING")
    print("=" * 80)

    finetuned_motions = [ep["total_eef_motion"] for ep in finetuned_results["fresh_env"]]
    pretrained_motions = [ep["total_eef_motion"] for ep in pretrained_results["fresh_env"]]

    finetuned_varied = len(set([f"{m:.6f}" for m in finetuned_motions])) > 1
    pretrained_varied = len(set([f"{m:.6f}" for m in pretrained_motions])) > 1

    print(f"\nFinetuned motion varied: {finetuned_varied}")
    print(f"Pretrained motion varied: {pretrained_varied}")

    if not finetuned_varied and pretrained_varied:
        print("\nHypothesis (a) LIKELY: Model produces identical actions")
        print("  -> Actions should be same across episodes for finetuned model")
    elif finetuned_varied and pretrained_varied:
        print("\nHypothesis (b) LIKELY: Different actions but same motion by chance")
    else:
        print("\nHypothesis (c) LIKELY: Something else is wrong")

    # Compare first episode actions between finetuned and pretrained
    print("\n--- Comparing first episode actions between models ---")
    ft_first = finetuned_results["fresh_env"][0]["raw_action_trace"][:5]
    pt_first = pretrained_results["fresh_env"][0]["raw_action_trace"][:5]

    print("Finetuned first 5 actions:")
    for i, a in enumerate(ft_first):
        print(f"  {a}")

    print("\nPretrained first 5 actions:")
    for i, a in enumerate(pt_first):
        print(f"  {a}")


if __name__ == "__main__":
    main()

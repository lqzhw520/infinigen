#!/usr/bin/env python3
"""
Simplified diagnostic to understand identical EEF motion issue.
Outputs directly to file to avoid buffering issues.
"""

import numpy as np
import torch
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from drawer_robot_env import DrawerRobotEnv
from mint_common import (
    ACTIVE_ACTION_CONTRACT_PATH,
    load_json,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot_policy_mint.modeling_mint import MINTPolicy

MINT_CKPT = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero"
FINETUNED_PATH = "/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/outputs_d1_seed_002_episode_01_multi_episode_overfit/checkpoints/000600/pretrained_model/"
DATASET_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/dataset_d1_seed_002_episode_01_multi_episode_overfit/")

OUTPUT_FILE = "/tmp/d1_diagnostic_output.txt"

def log(msg):
    """Log to both stdout and file."""
    print(msg)
    with open(OUTPUT_FILE, "a") as f:
        f.write(msg + "\n")

def _load_policy(path: str, dataset_root: Path, repo_id: str):
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, revision="main")
    policy = MINTPolicy.from_pretrained(
        path, local_files_only=True, dataset_stats=dataset.meta.stats
    )
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=path, dataset_stats=dataset.meta.stats
    )
    return policy, preprocessor, postprocessor

def _obs_to_batch(obs):
    return {
        "observation.images.image": torch.from_numpy(obs.image).permute(2, 0, 1).to(torch.float32) / 255.0,
        "observation.images.image2": torch.from_numpy(obs.image2).permute(2, 0, 1).to(torch.float32) / 255.0,
        "observation.state": torch.from_numpy(obs.state.astype(np.float32)),
        "task": obs.task,
    }

def run_episode(env, policy, pre, post, seed, rng_seed, reset_policy=True, max_steps=96):
    obs = env.reset()
    if reset_policy:
        policy.reset()
    
    raw_actions = []
    eef_positions = [obs.eef_pos.copy()]
    drawer_fractions = []
    attached_flags = []
    
    rng = np.random.default_rng(rng_seed + seed)
    
    for step in range(max_steps):
        batch = _obs_to_batch(obs)
        processed = pre(batch)
        
        with torch.inference_mode():
            action = policy.select_action(processed)
        
        action = post(action).squeeze(0).detach().cpu().numpy().astype(np.float32)
        raw_actions.append(action.copy())
        eef_positions.append(obs.eef_pos.copy())
        
        obs, _, done, info = env.step(action)
        drawer_fractions.append(float(info["drawer_fraction"]))
        attached_flags.append(bool(info.get("attached", False)))
        
        if done:
            break
    
    # Compute EEF motion
    eef_arr = np.array(eef_positions)
    step_deltas = np.linalg.norm(np.diff(eef_arr, axis=0), axis=1)
    total_motion = float(np.sum(step_deltas))
    
    return {
        "raw_actions": raw_actions,
        "eef_positions": eef_positions,
        "total_eef_motion": total_motion,
        "final_drawer": drawer_fractions[-1] if drawer_fractions else 0.0,
        "ever_attached": any(attached_flags),
        "n_steps": len(raw_actions),
    }

def test_env_determinism():
    log("=" * 60)
    log("TEST 0: Environment Determinism")
    log("=" * 60)
    
    results = []
    for i in range(3):
        env = DrawerRobotEnv(seed=2, image_size=224, max_steps=96)
        obs = env.reset()
        results.append({
            "eef": obs.eef_pos.copy(),
            "drawer": obs.drawer_fraction,
        })
        env.close()
    
    for i, r in enumerate(results):
        log(f"  Run {i}: EEF={r['eef']}, drawer={r['drawer']:.4f}")
    
    all_same = all(np.allclose(r1["eef"], r2["eef"]) for r1, r2 in zip(results[:-1], results[1:]))
    log(f"  Deterministic: {all_same}")

def run_policy_test(policy_name, policy_bundle, n_episodes=5):
    log(f"\n{'=' * 60}")
    log(f"TEST: {policy_name} ({n_episodes} episodes)")
    log("=" * 60)
    
    policy, pre, post = policy_bundle
    action_contract = load_json(ACTIVE_ACTION_CONTRACT_PATH, {})
    
    # Test 1: Fresh env per episode WITH policy.reset()
    log(f"\n--- Test 1: Fresh env + policy.reset() ---")
    episodes = []
    for i in range(n_episodes):
        log(f"  Starting episode {i}...")
        env = DrawerRobotEnv(seed=2, image_size=224, max_steps=96, action_contract=action_contract)
        result = run_episode(env, policy, pre, post, seed=2, rng_seed=19 + i * 101, reset_policy=True)
        episodes.append(result)
        log(f"  Episode {i}: motion={result['total_eef_motion']:.6f}, drawer={result['final_drawer']:.4f}, attached={result['ever_attached']}")
        env.close()
    
    motions = [ep["total_eef_motion"] for ep in episodes]
    log(f"\n  Motion values: {[f'{m:.6f}' for m in motions]}")
    log(f"  Motion std: {np.std(motions):.6f}")
    log(f"  Motion range: {max(motions) - min(motions):.6f}")
    
    # Check if actions identical
    first_actions = episodes[0]["raw_actions"]
    identical = all(
        np.allclose(ep["raw_actions"][:len(first_actions)], first_actions)
        for ep in episodes[1:]
    )
    log(f"  Actions identical: {identical}")
    
    # Print first 10 actions from first episode
    log(f"\n--- First 10 actions (episode 0) ---")
    for i, a in enumerate(episodes[0]["raw_actions"][:10]):
        log(f"  Step {i}: {a}")
    
    # Test 2: No policy.reset() between episodes
    log(f"\n--- Test 2: Same env, NO policy.reset() ---")
    env = DrawerRobotEnv(seed=2, image_size=224, max_steps=96, action_contract=action_contract)
    episodes_no_reset = []
    for i in range(n_episodes):
        result = run_episode(env, policy, pre, post, seed=2, rng_seed=19 + i * 101, reset_policy=False)
        episodes_no_reset.append(result)
        log(f"  Episode {i}: motion={result['total_eef_motion']:.6f}")
    env.close()
    
    motions_no_reset = [ep["total_eef_motion"] for ep in episodes_no_reset]
    log(f"  Motion values: {[f'{m:.6f}' for m in motions_no_reset]}")
    
    return episodes, episodes_no_reset

def main():
    import os
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    
    log("Loading policies...")
    log(f"  Finetuned: {FINETUNED_PATH}")
    log(f"  Dataset: {DATASET_ROOT}")
    
    log("\nLoading FINETUNED policy...")
    finetuned_bundle = _load_policy(FINETUNED_PATH, DATASET_ROOT, DATASET_REPO_ID := "MINT-libero")
    
    log("Loading PRETRAINED policy...")
    pretrained_bundle = _load_policy(MINT_CKPT, DATASET_ROOT, DATASET_REPO_ID)
    
    test_env_determinism()
    
    ft_results = run_policy_test("FINETUNED", finetuned_bundle)
    pt_results = run_policy_test("PRETRAINED", pretrained_bundle)
    
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log("\nExpected:")
    log("  Finetuned: all 5 = 7.557735919952393")
    log("  Pretrained: [1.586, 1.644, 10.400, 1.483, 4.901]")
    
    log("\nObserved (fresh env + policy.reset()):")
    ft_motions = [ep["total_eef_motion"] for ep in ft_results[0]]
    pt_motions = [ep["total_eef_motion"] for ep in pt_results[0]]
    log(f"  Finetuned: {[f'{m:.3f}' for m in ft_motions]}")
    log(f"  Pretrained: {[f'{m:.3f}' for m in pt_motions]}")
    
    log("\nObserved (no policy.reset()):")
    ft_motions_no_reset = [ep["total_eef_motion"] for ep in ft_results[1]]
    pt_motions_no_reset = [ep["total_eef_motion"] for ep in pt_results[1]]
    log(f"  Finetuned: {[f'{m:.3f}' for m in ft_motions_no_reset]}")
    log(f"  Pretrained: {[f'{m:.3f}' for m in pt_motions_no_reset]}")
    
    log("\nDone! Output saved to: " + OUTPUT_FILE)

if __name__ == "__main__":
    main()

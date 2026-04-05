#!/usr/bin/env python3
"""Diagnostic eval: verbose single-episode trace comparing policy vs teacher ground truth.

Answers:
  1. Is the policy outputting non-zero actions?
  2. Is the arm moving?
  3. Where does policy divergence start vs teacher ground truth?
  4. Does the checkpoint load cleanly (missing keys)?
"""

from __future__ import annotations

# Resolve relative imports from scripts/mint/ working directory
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

# Apply compatibility patches BEFORE importing lerobot/MINT.
from mint_eval_patches import apply as _apply_patches

_apply_patches()

from drawer_robot_env import DrawerRobotEnv
from evaluate_mint_drawer_campaign import _obs_to_batch
from mint_common import ACTIVE_ACTION_CONTRACT_PATH, load_json

MINT_CKPT = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero"

# --- Configurable diagnostic parameters ---
DIAGNOSTIC_SEED = 2
MAX_STEPS = 96
TEACHER_NPZ = Path(
    "/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/"
    "artifacts/c2_replay_valid_rollouts/seed_002_episode_01.npz"
)
# Use the NEW D1 checkpoint (multi_episode_overfit, VQ-VAE unfrozen)
CHECKPOINT_DIR = Path(
    "/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/"
    "outputs_d1_seed_002_episode_01_multi_episode_overfit/checkpoints/000600"
)
FINETUNED_CKPT = str(CHECKPOINT_DIR / "pretrained_model")
DATASET_ROOT = Path(
    "/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/"
    "dataset_d1_seed_002_episode_01_multi_episode_overfit"
)
DATASET_REPO_ID = "infinigen_drawer_robot_v1_d1_seed_002_episode_01_multi_episode_overfit"


def load_checkpoint_with_warnings(path: str, dataset_root: Path, repo_id: str) -> tuple:
    """Load policy and return (policy, pre, post) with any weight warnings."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot_policy_mint.modeling_mint import MINTPolicy

    print(f"\n{'='*70}")
    print(f"Loading finetuned checkpoint: {path}")
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, revision="main")
    policy = MINTPolicy.from_pretrained(
        path, local_files_only=True, dataset_stats=dataset.meta.stats
    )
    policy.eval()

    # Check for missing/unexpected keys at load time
    safetensors_path = Path(path) / "model.safetensors.index.json"
    if safetensors_path.exists():
        # Load safetensors index JSON to check key counts
        import json as json_mod
        with open(safetensors_path) as f:
            index_data = json_mod.load(f)
        sd = {"index": index_data, "count": len(index_data.get("weight_map", {}))}
    else:
        # No safetensors index, skip key check (model.safetensors can't be loaded by torch.load)
        sd = {}

    print(
        f"Loaded policy with {sd.get('count', 'unknown')} safetensors entries"
    )

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=path, dataset_stats=dataset.meta.stats
    )
    print(f"Policy loaded successfully")
    return policy, preprocessor, postprocessor, dataset


def load_teacher_npz(path: Path) -> dict:
    """Load teacher rollout NPZ and extract key fields."""
    print(f"\n{'='*70}")
    print(f"Loading teacher NPZ: {path}")
    npz = np.load(path)
    data = {k: npz[k] for k in npz.keys()}
    print(f"  frames: {len(data['actions'])}")
    print(f"  action shape: {data['actions'].shape}")
    print(f"  state shape: {data['states'].shape}")
    print(f"  image shape: {data['images'].shape}")
    print(f"  attach_trace sum: {np.sum(data['attached_trace'])}")
    print(f"  max drawer fraction: {np.max(data['absolute_drawer_fraction']):.4f}")
    return data


def run_verbose_eval(
    policy, pre, post, teacher_npz: dict, seed: int, max_steps: int
) -> dict:
    """Run policy on seed, compare to teacher ground truth at each step."""
    action_contract = load_json(ACTIVE_ACTION_CONTRACT_PATH, {})

    print(f"\n{'='*70}")
    print(f"Running verbose eval: seed={seed}, max_steps={max_steps}")

    env = DrawerRobotEnv(
        seed=seed, image_size=224, max_steps=max_steps, action_contract=action_contract
    )
    obs = env.reset()

    policy_actions = []
    eef_pos_trace = []
    drawer_trace = []
    attached_trace = []
    gripper_trace = []

    # Teacher ground truth
    t_actions = teacher_npz["actions"]
    t_states = teacher_npz["states"]
    t_drawer = teacher_npz["absolute_drawer_fraction"]
    t_attached = teacher_npz["attached_trace"]

    # Translation scale from contract (for interpreting teacher actions)
    t_scale = action_contract.get("translation_scale_m", 0.03)

    try:
        for step in range(1, max_steps + 1):
            # --- Record state BEFORE action ---
            eef_pos_trace.append(obs.eef_pos.copy())
            gripper_trace.append(float(obs.gripper_open))

            # Policy inference
            batch = _obs_to_batch(obs)
            processed = pre(batch)
            with torch.inference_mode():
                action = policy.select_action(processed)
            action = post(action)
            policy_action = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
            policy_actions.append(policy_action)

            # Step env
            obs, _, done, info = env.step(policy_action)

            drawer_trace.append(float(info["drawer_fraction"]))
            attached_trace.append(bool(info.get("attached")))

            if done:
                break
    finally:
        env.close()

    # --- Compute diagnostic metrics ---
    valid_actions = np.array([a for a in policy_actions if a is not None])
    non_zero_ratio = (
        float(np.mean([float(np.abs(a).max() > 0.01) for a in valid_actions]))
        if len(valid_actions)
        else 0.0
    )
    total_eef_motion = (
        float(np.sum(np.linalg.norm(np.diff(np.array(eef_pos_trace), axis=0), axis=1)))
        if len(eef_pos_trace) > 1
        else 0.0
    )

    print(f"\n{'='*70}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")
    print(f"Steps run:          {len(policy_actions)}")
    print(f"Non-zero action %: {non_zero_ratio*100:.1f}%")
    print(f"Total EEF motion:  {total_eef_motion:.4f} m")
    print(f"Final drawer frac: {drawer_trace[-1]:.4f}" if drawer_trace else "N/A")
    print(f"Max drawer frac:   {max(drawer_trace):.4f}" if drawer_trace else "N/A")
    print(f"Ever attached:     {any(attached_trace)}")
    print(f"Teacher max frac:  {np.max(t_drawer):.4f}")
    print(f"Teacher attach sum:{np.sum(t_attached)}")
    print(f"Teacher steps:     {len(t_actions)}")

    # --- Per-step trace: policy vs teacher ---
    print(f"\n{'='*70}")
    print(
        f"{'STEP':>5} | {'POLICY ACTION (xyz/grip)':^35} | {'POLICY EEF':^20} | "
        f"{'DRW':^6} | {'ATT':^4} | {'TEACHER ACT (xyz/grip)':^35}"
    )
    print(f"{'-'*5}-+-{'-'*35}-+-{'-'*20}-+-{'-'*6}-+-{'-'*4}-+-{'-'*35}")

    n_teacher_frames = len(t_actions)
    for i, (pa, dw, att, ef) in enumerate(
        zip(policy_actions, drawer_trace, attached_trace, eef_pos_trace)
    ):
        # Pad teacher actions if eval runs longer than teacher rollout
        if i < n_teacher_frames:
            ta = t_actions[i]
            teacher_str = f"{ta[0]:+.3f},{ta[1]:+.3f},{ta[2]:+.3f} | g={ta[6]:+.1f}"
        else:
            ta = None
            teacher_str = "(beyond teacher)"

        pa_str = f"{pa[0]:+.3f},{pa[1]:+.3f},{pa[2]:+.3f} | g={pa[6]:+.1f}"
        eef_str = f"{ef[0]:+.3f},{ef[1]:+.3f},{ef[2]:+.3f}"
        print(
            f"{i+1:>5} | {pa_str:^35} | {eef_str:^20} | "
            f"{dw:>6.4f} | {'Y' if att else 'N':^4} | {teacher_str}"
        )

    # --- Action magnitude analysis ---
    print(f"\n{'='*70}")
    print("ACTION MAGNITUDE ANALYSIS")
    print(f"{'='*70}")
    if len(valid_actions) > 0:
        act_arr = np.array(valid_actions)
        for dim, name in enumerate(["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]):
            vals = act_arr[:, dim]
            print(
                f"  {name:>8}: mean={vals.mean():+.4f}, std={vals.std():.4f}, "
                f"min={vals.min():+.4f}, max={vals.max():+.4f}"
            )

    # --- Comparison: policy vs teacher on common frames ---
    print(f"\n{'='*70}")
    print("POLICY vs TEACHER DIVERGENCE (first 20 steps)")
    print(f"{'='*70}")
    common = min(len(policy_actions), n_teacher_frames, 20)
    for i in range(common):
        pa = valid_actions[i] if i < len(valid_actions) else np.zeros(7)
        ta = t_actions[i]
        delta = np.abs(pa - ta)
        print(
            f"  [{i:02d}] policy={pa[:3].round(3)} | "
            f"teacher={ta[:3].round(3)} | "
            f"Δ={delta[:3].round(3)} | "
            f"maxΔ={delta[:3].max():.3f}"
        )

    return {
        "non_zero_action_ratio": non_zero_ratio,
        "total_eef_motion": total_eef_motion,
        "drawer_trace": drawer_trace,
        "attached_trace": attached_trace,
        "policy_actions": [a.tolist() for a in valid_actions],
        "eef_pos_trace": [p.tolist() for p in eef_pos_trace],
    }


def main():
    print("=" * 70)
    print("MINT D1 DIAGNOSTIC EVAL — Verbose Single-Episode Trace")
    print("=" * 70)
    print(f"Finetuned checkpoint: {FINETUNED_CKPT}")
    print(f"Teacher NPZ:         {TEACHER_NPZ}")
    print(f"Dataset:             {DATASET_ROOT}")
    print(f"Diagnostic seed:     {DIAGNOSTIC_SEED}")

    # Check checkpoint exists
    if not Path(FINETUNED_CKPT).exists():
        print(f"\nERROR: Checkpoint not found at {FINETUNED_CKPT}")
        print("Cannot run diagnostic. Check that the D1 training completed.")
        return 1

    # Load teacher ground truth
    teacher_npz = load_teacher_npz(TEACHER_NPZ)

    # Load finetuned policy with full diagnostic
    policy, pre, post, dataset = load_checkpoint_with_warnings(
        FINETUNED_CKPT, DATASET_ROOT, DATASET_REPO_ID
    )

    # Run verbose eval
    result = run_verbose_eval(
        policy, pre, post, teacher_npz, DIAGNOSTIC_SEED, MAX_STEPS
    )

    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print("Key findings:")
    print(
        f"  - Policy non-zero action ratio: {result['non_zero_action_ratio']*100:.1f}%"
    )
    print(f"  - Total EEF motion: {result['total_eef_motion']:.4f} m")
    print(
        f"  - Max drawer fraction: {max(result['drawer_trace']):.4f}"
        if result["drawer_trace"]
        else ""
    )
    print(f"  - Ever attached: {any(result['attached_trace'])}")
    print(f"{'='*70}")

    # Exit code: 0 if policy produced non-trivial actions, 1 if all-zero
    if result["non_zero_action_ratio"] < 0.05:
        print("\nWARNING: Policy is producing nearly all-zero actions.")
        print("This indicates the policy has catastrophically overfit to action=0")
        print("OR the model output is being zeroed by post-processing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

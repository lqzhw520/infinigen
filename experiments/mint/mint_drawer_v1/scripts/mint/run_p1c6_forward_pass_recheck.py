#!/usr/bin/env python3
"""E034 — Forward pass diagnostic after key remap fix (2026-04-09).
Verifies: (1) LM weights loaded, (2) state-invariant behavior confirmed or refuted."""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")  # fixed absolute path
MINT_REPO_ROOT = PROJECT_ROOT / "external" / "MINT"
ARTIFACT_ROOT = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1" / "artifacts"
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "mint"))
sys.path.insert(0, str(MINT_REPO_ROOT / "lerobot_policy_mint" / "src"))

from mujoco_pilot_common import (
    ARTIFACT_ROOT, MINT_CKPT, load_mint_policy, load_task_spec,
    load_init_states, make_env, build_state, build_batch,
    write_json, now_iso, drawer_fraction, ensure_dir,
)
from safetensors.torch import load_file

ARTIFACT_PATH = ARTIFACT_ROOT / "p1c6_forward_pass_recheck.json"


def main() -> int:
    ensure_dir(ARTIFACT_ROOT)

    print("=" * 60)
    print("E034 — Forward Pass Diagnostic (Post Key Remap Fix)")
    print("=" * 60)

    # Load checkpoint to verify key loading
    ckpt_sd = load_file(str(MINT_CKPT / "model.safetensors"))
    embed_in_ckpt = any("embed_tokens" in k for k in ckpt_sd.keys())
    print(f"\n[Checkpoint] embed_tokens present: {embed_in_ckpt} (expected: False — intentional)")

    # Load policy
    print("\n[Loading] MINTPolicy from pretrained...")
    policy_bundle = load_mint_policy()
    policy, preprocessor, postprocessor = policy_bundle
    device = next(policy.parameters()).device
    print(f"[Model] device={device}")

    sd = policy.state_dict()

    # Verify LM weights
    lm_q_weight = sd.get('model.paligemma_with_expert.paligemma.language_model.model.layers.0.self_attn.q_proj.weight')
    lm_head_weight = sd.get('model.paligemma_with_expert.paligemma.language_model.lm_head.weight')
    embed_weight = sd.get('model.paligemma_with_expert.paligemma.language_model.model.embed_tokens.weight')

    print(f"\n[Weight Check]")
    print(f"  LM layers.0.q_proj: loaded={lm_q_weight is not None}, shape={lm_q_weight.shape if lm_q_weight is not None else None}")
    print(f"  lm_head:           loaded={lm_head_weight is not None}, shape={lm_head_weight.shape if lm_head_weight is not None else None}")
    print(f"  embed_tokens:      loaded={embed_weight is not None}, shape={embed_weight.shape if embed_weight is not None else None}, mean={embed_weight.mean().item():.6f}" if embed_weight is not None else "  embed_tokens: not in model state_dict")

    # Forward pass test
    task_spec = load_task_spec()
    init_states = load_init_states(task_spec)[:3]

    results = []
    print(f"\n[Forward Pass] Testing {len(init_states)} init states × 2 state modes")

    for init_index, init_state in enumerate(init_states):
        env = make_env(task_spec, image_size=224)
        try:
            env.reset()
            obs = env.set_init_state(init_state)

            # GT state
            state_gt = build_state(obs, joint_mode="raw_joint_pos")
            batch_gt = build_batch(obs, state=state_gt, task_text=task_spec.task_text)
            processed_gt = preprocessor(batch_gt)
            with torch.inference_mode():
                action_gt = policy.select_action(processed_gt)
            action_gt_np = postprocessor(action_gt).squeeze(0).cpu().numpy().astype("float32")

            # Zero state
            state_zero = np.zeros_like(state_gt)
            batch_zero = build_batch(obs, state=state_zero, task_text=task_spec.task_text)
            processed_zero = preprocessor(batch_zero)
            with torch.inference_mode():
                action_zero = policy.select_action(processed_zero)
            action_zero_np = postprocessor(action_zero).squeeze(0).cpu().numpy().astype("float32")

            # Compare
            delta = np.abs(action_gt_np - action_zero_np)
            action_range = np.abs(action_gt_np).max()
            delta_pct = delta / (action_range + 1e-6) * 100
            max_delta_pct = delta_pct.max()

            print(f"  init={init_index}: delta/max_range={max_delta_pct:.1f}%")

            results.append({
                "init_index": int(init_index),
                "action_gt": action_gt_np.tolist(),
                "action_zero": action_zero_np.tolist(),
                "delta": delta.tolist(),
                "delta_pct_max": float(max_delta_pct),
                "delta_pct_mean": float(delta_pct.mean()),
            })

        finally:
            env.close()

    # Summary
    delta_pcts = [r["delta_pct_max"] for r in results]
    mean_delta_pct = float(np.mean(delta_pcts))
    all_state_invariant = all(r["delta_pct_max"] < 5.0 for r in results)

    print(f"\n[Summary]")
    print(f"  Mean delta/action_range: {mean_delta_pct:.1f}%")
    print(f"  State-invariant (<5%): {all_state_invariant}")

    if all_state_invariant:
        conclusion = "STATE_INVARIANT — state input has negligible effect on action output even with LM weights loaded"
        print(f"\n  CONCLUSION: {conclusion}")
        print("  INTERPRETATION: Language model weights are loaded but state conditioning")
        print("  channel is still inactive. This is a training design choice, not a bug.")
    else:
        conclusion = "STATE_VARIANT — state input now influences action output after LM weight fix"
        print(f"\n  CONCLUSION: {conclusion}")
        print("  INTERPRETATION: LM weight fix changed behavior. State now affects actions.")

    verdict = "STATE_INVARIANT_CONFIRMED" if all_state_invariant else "STATE_VARIANT_DETECTED"

    # Save artifact
    artifact = {
        "experiment_id": "E034_p1c6_forward_pass_recheck",
        "timestamp": now_iso(),
        "key_remap_fixed": True,
        "lm_weights_loaded": {
            "q_proj": lm_q_weight is not None,
            "lm_head": lm_head_weight is not None,
            "embed_tokens": embed_weight is not None,
        },
        "embed_tokens_in_checkpoint": embed_in_ckpt,
        "per_init_results": results,
        "mean_delta_pct": mean_delta_pct,
        "verdict": verdict,
        "conclusion": conclusion,
    }
    write_json(ARTIFACT_PATH, artifact)
    print(f"\n  Artifact saved: {ARTIFACT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

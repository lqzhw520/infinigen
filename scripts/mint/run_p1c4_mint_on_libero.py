#!/usr/bin/env python3
"""Evaluate pretrained MINT on the LIBERO drawer task."""

from __future__ import annotations

import statistics

import numpy as np
import torch

from mujoco_pilot_common import (
    ARTIFACT_ROOT,
    MINT_CKPT,
    build_batch,
    build_state,
    drawer_fraction,
    ensure_dir,
    load_init_states,
    load_mint_policy,
    load_task_spec,
    make_env,
    now_iso,
    write_json,
)

ARTIFACT_PATH = ARTIFACT_ROOT / "p1c4_mint_on_libero_drawer.json"
STATE_MODES = ("raw_joint_pos", "pi_minus_joint_pos")
MAX_STEPS = 96


def rollout_one(task_spec, init_state, *, init_index: int, state_mode: str, policy_bundle) -> dict:
    policy, preprocessor, postprocessor = policy_bundle
    env = make_env(task_spec, image_size=224)
    try:
        env.reset()
        obs = env.set_init_state(init_state)
        drawer_trace = [drawer_fraction(env)]
        action_l2 = []
        success_step = None
        for step in range(1, MAX_STEPS + 1):
            state = build_state(obs, joint_mode=state_mode)
            batch = build_batch(obs, state=state, task_text=task_spec.task_text)
            processed = preprocessor(batch)
            with torch.inference_mode():
                action = policy.select_action(processed)
            action = postprocessor(action).squeeze(0).detach().cpu().numpy().astype(np.float32)
            action_l2.append(float(np.linalg.norm(action)))
            obs, reward, done, info = env.step(action)
            drawer_trace.append(drawer_fraction(env))
            if env.check_success():
                success_step = step
                break
            if done:
                break
        return {
            "init_index": init_index,
            "state_mode": state_mode,
            "success": success_step is not None,
            "success_step": success_step,
            "final_success": bool(env.check_success()),
            "max_drawer_fraction": float(max(drawer_trace)),
            "final_drawer_fraction": float(drawer_trace[-1]),
            "drawer_fraction_trace": drawer_trace,
            "mean_action_l2": float(statistics.mean(action_l2)) if action_l2 else 0.0,
            "max_action_l2": float(max(action_l2)) if action_l2 else 0.0,
        }
    finally:
        env.close()


def main() -> int:
    ensure_dir(ARTIFACT_PATH.parent)
    task_spec = load_task_spec()
    init_states = load_init_states(task_spec)[:3]
    policy_bundle = load_mint_policy()
    results = []
    for state_mode in STATE_MODES:
        for init_index, init_state in enumerate(init_states):
            results.append(
                rollout_one(
                    task_spec,
                    init_state,
                    init_index=init_index,
                    state_mode=state_mode,
                    policy_bundle=policy_bundle,
                )
            )
    aggregate = {}
    for state_mode in STATE_MODES:
        subset = [row for row in results if row["state_mode"] == state_mode]
        aggregate[state_mode] = {
            "episodes": len(subset),
            "success_rate": sum(1 for row in subset if row["success"]) / max(len(subset), 1),
            "mean_final_drawer_fraction": float(statistics.mean(row["final_drawer_fraction"] for row in subset)),
            "mean_max_drawer_fraction": float(statistics.mean(row["max_drawer_fraction"] for row in subset)),
            "mean_action_l2": float(statistics.mean(row["mean_action_l2"] for row in subset)),
        }
    payload = {
        "stage": "p1c4",
        "generated_at": now_iso(),
        "benchmark_name": task_spec.benchmark_name,
        "task_name": task_spec.task_name,
        "task_text": task_spec.task_text,
        "mint_checkpoint": str(MINT_CKPT),
        "state_modes": list(STATE_MODES),
        "episodes": results,
        "aggregate": aggregate,
        "passed": True,
    }
    write_json(ARTIFACT_PATH, payload)
    print(f"[p1c4] wrote {ARTIFACT_PATH}")
    for state_mode, summary in aggregate.items():
        print(
            f"[p1c4] {state_mode}: success_rate={summary['success_rate']:.3f} "
            f"mean_final_drawer_fraction={summary['mean_final_drawer_fraction']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

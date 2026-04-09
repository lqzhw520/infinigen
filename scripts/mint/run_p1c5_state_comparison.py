#!/usr/bin/env python3
"""Compare live LIBERO state statistics against the current PyBullet dataset state."""

from __future__ import annotations

import numpy as np

from mujoco_pilot_common import (
    ARTIFACT_ROOT,
    build_state,
    ensure_dir,
    load_init_states,
    load_task_spec,
    make_env,
    now_iso,
    sample_pybullet_dataset_states,
    summarize_array,
    write_json,
)

ARTIFACT_PATH = ARTIFACT_ROOT / "p1c5_state_semantic_comparison.json"
STATE_MODES = ("raw_joint_pos", "pi_minus_joint_pos")


def collect_libero_states(task_spec, *, limit: int = 96) -> dict[str, np.ndarray]:
    init_states = load_init_states(task_spec)[:3]
    collected = {mode: [] for mode in STATE_MODES}
    env = make_env(task_spec, image_size=128)
    try:
        for init_state in init_states:
            env.reset()
            obs = env.set_init_state(init_state)
            for mode in STATE_MODES:
                collected[mode].append(build_state(obs, joint_mode=mode))
            for _ in range(limit // max(len(init_states), 1)):
                obs, _, done, _ = env.step([0.0] * 7)
                for mode in STATE_MODES:
                    collected[mode].append(build_state(obs, joint_mode=mode))
                if done:
                    break
    finally:
        env.close()
    return {mode: np.stack(rows) for mode, rows in collected.items()}


def main() -> int:
    ensure_dir(ARTIFACT_PATH.parent)
    task_spec = load_task_spec()
    pybullet_states = sample_pybullet_dataset_states(limit=100)
    libero_states = collect_libero_states(task_spec, limit=96)
    payload = {
        "stage": "p1c5",
        "generated_at": now_iso(),
        "benchmark_name": task_spec.benchmark_name,
        "task_name": task_spec.task_name,
        "pybullet_dataset": summarize_array("pybullet_dataset", pybullet_states),
        "libero_live": {mode: summarize_array(mode, values) for mode, values in libero_states.items()},
        "comparison": {},
        "passed": True,
    }
    py_mean = pybullet_states.mean(axis=0)
    for mode, values in libero_states.items():
        libero_mean = values.mean(axis=0)
        payload["comparison"][mode] = {
            "dimension_wise_mean_delta": (libero_mean - py_mean).tolist(),
            "mean_l2_delta": float(np.linalg.norm(libero_mean - py_mean)),
        }
    write_json(ARTIFACT_PATH, payload)
    print(f"[p1c5] wrote {ARTIFACT_PATH}")
    for mode, summary in payload["comparison"].items():
        print(f"[p1c5] {mode}: mean_l2_delta={summary['mean_l2_delta']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

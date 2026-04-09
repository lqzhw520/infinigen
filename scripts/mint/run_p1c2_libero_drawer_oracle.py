#!/usr/bin/env python3
"""State-space oracle gate for the LIBERO drawer task on true MuJoCo."""

from __future__ import annotations

import numpy as np

from mujoco_pilot_common import (
    ARTIFACT_ROOT,
    bottom_drawer_joint_info,
    drawer_fraction,
    ensure_dir,
    load_init_states,
    load_task_spec,
    make_env,
    now_iso,
    set_drawer_qpos,
    write_json,
)

ARTIFACT_PATH = ARTIFACT_ROOT / "p1c2_libero_drawer_oracle_gate.json"


def run_episode(task_spec, init_state: np.ndarray, *, init_index: int) -> dict:
    env = make_env(task_spec, image_size=128)
    try:
        env.reset()
        env.set_init_state(init_state)
        joint_info = bottom_drawer_joint_info(env)
        qpos_adr = joint_info["qpos_adr"]
        start_qpos = float(env.sim.data.qpos[qpos_adr])
        trace = [drawer_fraction(env)]
        qpos_trace = [start_qpos]
        for alpha in np.linspace(0.0, 1.0, num=24, endpoint=True)[1:]:
            target_qpos = (1.0 - alpha) * start_qpos + alpha * joint_info["open_qpos"]
            set_drawer_qpos(env, target_qpos)
            trace.append(drawer_fraction(env))
            qpos_trace.append(float(env.sim.data.qpos[qpos_adr]))
        success = bool(env.check_success())
        return {
            "init_index": init_index,
            "start_qpos": start_qpos,
            "open_qpos": joint_info["open_qpos"],
            "joint_name": joint_info["joint_name"],
            "drawer_fraction_trace": trace,
            "drawer_qpos_trace": qpos_trace,
            "final_drawer_fraction": trace[-1],
            "success": success,
        }
    finally:
        env.close()


def main() -> int:
    ensure_dir(ARTIFACT_PATH.parent)
    task_spec = load_task_spec()
    init_states = load_init_states(task_spec)[:3]
    episodes = [run_episode(task_spec, state, init_index=i) for i, state in enumerate(init_states)]
    success_rate = sum(1 for ep in episodes if ep["success"]) / max(len(episodes), 1)
    payload = {
        "stage": "p1c2",
        "generated_at": now_iso(),
        "benchmark_name": task_spec.benchmark_name,
        "task_name": task_spec.task_name,
        "oracle_type": "mujoco_state_oracle",
        "oracle_uses_robot_actions": False,
        "oracle_uses_direct_drawer_joint": True,
        "oracle_proves": "drawer task semantics and success detector are reachable under MuJoCo state control",
        "oracle_does_not_prove": "action-space scripted controller quality or teacher trajectory fidelity",
        "episodes": episodes,
        "success_rate": success_rate,
        "passed": success_rate == 1.0 and len(episodes) == 3,
    }
    write_json(ARTIFACT_PATH, payload)
    print(f"[p1c2] wrote {ARTIFACT_PATH}")
    print(f"[p1c2] success_rate={success_rate:.3f} passed={payload['passed']}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

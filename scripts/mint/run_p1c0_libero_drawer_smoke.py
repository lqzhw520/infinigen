#!/usr/bin/env python3
"""Smoke test for the LIBERO drawer task on true MuJoCo + OSMesa."""

from __future__ import annotations

from mujoco_pilot_common import ARTIFACT_ROOT, TaskSpec, ensure_dir, load_task_spec, make_env, now_iso, write_json

ARTIFACT_PATH = ARTIFACT_ROOT / "p1c0_libero_drawer_smoke.json"


def run_once(task_spec: TaskSpec, *, run_index: int) -> dict:
    env = make_env(task_spec, image_size=128)
    try:
        obs = env.reset()
        first = {
            "agentview_shape": list(obs["agentview_image"].shape),
            "wrist_shape": list(obs["robot0_eye_in_hand_image"].shape),
            "agentview_mean": float(obs["agentview_image"].mean()),
            "wrist_mean": float(obs["robot0_eye_in_hand_image"].mean()),
            "success_after_reset": bool(env.check_success()),
        }
        last_info = {}
        for step in range(10):
            obs, reward, done, info = env.step([0.0] * 7)
            last_info = {
                "step": step + 1,
                "reward": float(reward),
                "done": bool(done),
                "success": bool(env.check_success()),
            }
        return {
            "run_index": run_index,
            "passed": True,
            "reset": first,
            "after_steps": last_info,
        }
    finally:
        env.close()


def main() -> int:
    ensure_dir(ARTIFACT_PATH.parent)
    task_spec = load_task_spec()
    runs = []
    errors = []
    for run_index in range(3):
        try:
            runs.append(run_once(task_spec, run_index=run_index))
        except Exception as exc:  # pragma: no cover - pilot runtime detail
            errors.append({"run_index": run_index, "error": repr(exc)})
    payload = {
        "stage": "p1c0",
        "generated_at": now_iso(),
        "benchmark_name": task_spec.benchmark_name,
        "task_name": task_spec.task_name,
        "bddl_file": task_spec.bddl_file,
        "runtime": {
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
        },
        "runs": runs,
        "errors": errors,
        "passed": len(errors) == 0 and len(runs) == 3 and all(run["reset"]["agentview_mean"] > 1.0 and run["reset"]["wrist_mean"] > 1.0 for run in runs),
    }
    write_json(ARTIFACT_PATH, payload)
    print(f"[p1c0] wrote {ARTIFACT_PATH}")
    print(f"[p1c0] passed={payload['passed']} runs={len(runs)} errors={len(errors)}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

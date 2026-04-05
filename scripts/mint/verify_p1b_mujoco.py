#!/usr/bin/env python3
"""P1b: Verify LIBERO-aligned teacher environment produces LIBERO-aligned observations.

Gate condition:
  1. agentview image: (256, 256, 3) uint8
  2. wrist image:    (256, 256, 3) uint8
  3. state:  (8,) float32, state[7] ∈ [-0.042, +0.001]

NOTE: DrawerRobotEnvLeRobot uses PyBullet (not MuJoCo) for both physics and rendering.
      See drawer_robot_env_lerobot.py docstring for details.

Exit codes:
  0 = PASS
  1 = FAIL
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
SCRIPTS_MINT = PROJECT_ROOT / "scripts" / "mint"
sys.path.insert(0, str(SCRIPTS_MINT))

# Import the environment; file may not exist yet → AttributeError → FAIL
try:
    from drawer_robot_env_lerobot import DrawerRobotEnvLeRobot
except (ImportError, AttributeError) as exc:
    print(f"[FAIL] Could not import DrawerRobotEnvLeRobot: {exc}")
    print("       drawer_robot_env_lerobot.py does not exist yet.")
    sys.exit(1)

LIBERO_GRIPPER_MIN = -0.042
LIBERO_GRIPPER_MAX = +0.001
EXPECTED_IMAGE_SHAPE = (256, 256, 3)
EXPECTED_STATE_SHAPE = (8,)

PASSED = 0
FAILED = 0
errors: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        print(f"  [{label}] OK{detail}")
        PASSED += 1
    else:
        print(f"  [{label}] FAIL{detail}")
        FAILED += 1


def run() -> bool:
    global errors

    # Smoke test: load env, reset, observe
    try:
        env = DrawerRobotEnvLeRobot(seed=2, image_size=256, max_steps=8)
    except Exception as exc:
        print(f"[FAIL] Could not construct DrawerRobotEnvLeRobot: {exc}")
        sys.exit(1)

    try:
        obs = env.reset()
        print("[DrawerRobotEnvLeRobot smoke test]")

        # Check image shapes (RobotObservation uses __getitem__ for subscript access)
        agent_img = obs["image"]
        wrist_img = obs["image2"]

        check("agentview image",
              agent_img is not None and agent_img.shape == EXPECTED_IMAGE_SHAPE,
              f" shape={getattr(agent_img, 'shape', None)}")
        check("wrist image",
              wrist_img is not None and wrist_img.shape == EXPECTED_IMAGE_SHAPE,
              f" shape={getattr(wrist_img, 'shape', None)}")

        # Check image dtype
        check("agentview dtype uint8",
              agent_img is not None and agent_img.dtype == "uint8")
        check("wrist dtype uint8",
              wrist_img is not None and wrist_img.dtype == "uint8")

        # Check state shape and range
        state = obs["state"]
        check("state shape (8,)",
              state is not None and state.shape == EXPECTED_STATE_SHAPE,
              f" shape={getattr(state, 'shape', None)}")
        check("state dtype float32",
              state is not None and state.dtype == "float32")

        if state is not None and state.shape == EXPECTED_STATE_SHAPE:
            gripper_joint = float(state[7])
            in_range = LIBERO_GRIPPER_MIN <= gripper_joint <= LIBERO_GRIPPER_MAX
            check("state[7] ∈ LIBERO gripper range",
                  in_range,
                  f" state[7]={gripper_joint:+.6f} vs range=[{LIBERO_GRIPPER_MIN}, {LIBERO_GRIPPER_MAX}]")
        else:
            check("state[7] range", False, "state unavailable")

        # Check eef_pos (first 3 entries of state)
        if state is not None:
            eef_pos = state[:3]
            check("eef_pos finite",
                  all(abs(v) < 5.0 for v in eef_pos),
                  f" eef_pos={eef_pos}")

        # Check images are not all-black (sanity)
        if agent_img is not None:
            mean_val = float(agent_img.mean())
            check("agentview non-trivial (mean > 1)",
                  mean_val > 1.0,
                  f" mean_pixel={mean_val:.2f}")
        if wrist_img is not None:
            mean_val = float(wrist_img.mean())
            check("wrist non-trivial (mean > 1)",
                  mean_val > 1.0,
                  f" mean_pixel={mean_val:.2f}")

        # Step a few times
        try:
            dummy_action = obs["state"]  # just to check state is accessible
            import numpy as np
            action = np.zeros(7, dtype=np.float32)
            obs2, reward, done, info = env.step(action)
            check("step() returns RobotObservation",
                  hasattr(obs2, "state") and obs2.state is not None)
            check("step() done flag",
                  isinstance(done, bool))
        except Exception as exc:
            check("step()", False, f" {exc}")

    finally:
        try:
            env.close()
        except Exception:
            pass

    print(f"\n[SUMMARY]")
    print(f"  PASSED: {PASSED}")
    print(f"  FAILED: {FAILED}")

    overall_pass = FAILED == 0
    verdict = "PASS" if overall_pass else "FAIL"
    print(f"\n[P1b gate] {verdict}")
    return overall_pass


if __name__ == "__main__":
    ok = run()
    raise SystemExit(0 if ok else 1)

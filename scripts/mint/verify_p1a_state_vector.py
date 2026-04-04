#!/usr/bin/env python3
"""Verify P1a: _state_vector() outputs continuous gripper joint, not binary {0,1}."""

import sys
sys.path.insert(0, "/mnt/afs2/zhuhaowu/infinigen")

import numpy as np
from scripts.mint.drawer_robot_env import DrawerRobotEnv

LIBERO_GRIPPER_MIN = -0.042
LIBERO_GRIPPER_MAX = 0.001

def main():
    seed = 2  # standard train seed
    env = DrawerRobotEnv(seed=seed, image_size=224, max_steps=96)
    obs = env.reset()

    # ── Check 1: Initial state[7] should be in LIBERO range, not {0,1} ──
    state7 = obs.state[7]
    is_binary = state7 in {0.0, 1.0}
    in_libero_range = LIBERO_GRIPPER_MIN <= state7 <= LIBERO_GRIPPER_MAX
    print(f"[P1a Check 1] Initial state[7] = {state7:.6f}")
    print(f"  Binary (0 or 1)? {is_binary}")
    print(f"  In LIBERO range [{LIBERO_GRIPPER_MIN}, {LIBERO_GRIPPER_MAX}]? {in_libero_range}")
    check1_pass = not is_binary and in_libero_range
    print(f"  → {'PASS' if check1_pass else 'FAIL'}\n")

    # ── Check 2: state[7] across the full open→close trajectory ──
    states_open, states_closed = [], []

    # Record states while gripper is OPEN
    for i in range(8):
        env._set_gripper_open(1.0)
        env._step_world(4)
        obs_open = env.observe()
        states_open.append(obs_open.state[7])

    # Record states while gripper is CLOSED
    for i in range(8):
        env._set_gripper_open(0.0)
        env._step_world(4)
        obs_closed = env.observe()
        states_closed.append(obs_closed.state[7])

    states_open = np.array(states_open)
    states_closed = np.array(states_closed)

    print(f"[P1a Check 2] Gripper trajectory (8 steps each)")
    print(f"  OPEN  (gripper_open=1.0): mean={states_open.mean():.6f}, "
          f"std={states_open.std():.6f}, range=[{states_open.min():.6f}, {states_open.max():.6f}]")
    print(f"  CLOSED(gripper_open=0.0): mean={states_closed.mean():.6f}, "
          f"std={states_closed.std():.6f}, range=[{states_closed.min():.6f}, {states_closed.max():.6f}]")

    open_binary = all(v in {0.0, 1.0} for v in states_open)
    closed_binary = all(v in {0.0, 1.0} for v in states_closed)
    check2_pass = (not open_binary) and (not closed_binary)
    print(f"  All values binary {0,1}? open={open_binary}, closed={closed_binary}")
    print(f"  → {'PASS' if check2_pass else 'FAIL'}\n")

    # ── Check 3: Open state[7] > Closed state[7] (physically correct) ──
    # In LIBERO: positive = open, negative = closed
    open_mean = states_open.mean()
    closed_mean = states_closed.mean()
    check3_pass = open_mean > closed_mean
    print(f"[P1a Check 3] Physical ordering")
    print(f"  open_mean({open_mean:.6f}) > closed_mean({closed_mean:.6f})? {check3_pass}")
    print(f"  → {'PASS' if check3_pass else 'FAIL'}\n")

    # ── Check 4: All values strictly within LIBERO gripper range ──
    all_values = np.concatenate([states_open, states_closed])
    all_in_range = np.all((all_values >= LIBERO_GRIPPER_MIN) & (all_values <= LIBERO_GRIPPER_MAX))
    print(f"[P1a Check 4] All {len(all_values)} values in LIBERO range? {all_in_range}")
    print(f"  min={all_values.min():.6f}, max={all_values.max():.6f}")
    print(f"  expected range: [{LIBERO_GRIPPER_MIN}, {LIBERO_GRIPPER_MAX}]")
    check4_pass = all_in_range
    print(f"  → {'PASS' if check4_pass else 'FAIL'}\n")

    # ── Check 5: PyBullet finger positions → mapped to LIBERO range ──
    # GRIPPER_OPEN_POS=0.04, GRIPPER_CLOSED_POS=0.0
    # Formula: gripper_joint = finger_pos * 1.075 - 0.042
    # Expected open:  0.04 * 1.075 - 0.042 = 0.001
    # Expected closed: 0.00 * 1.075 - 0.042 = -0.042
    expected_open_libero = 0.04 * 1.075 - 0.042
    expected_closed_libero = 0.00 * 1.075 - 0.042
    print(f"[P1a Check 5] Formula verification")
    print(f"  Expected for open (PyBullet 0.04):  {expected_open_libero:.6f}")
    print(f"  Expected for closed (PyBullet 0.0): {expected_closed_libero:.6f}")
    print(f"  Actual open mean:  {open_mean:.6f}  (Δ={abs(open_mean - expected_open_libero):.6f})")
    print(f"  Actual closed mean: {closed_mean:.6f}  (Δ={abs(closed_mean - expected_closed_libero):.6f})")
    check5_pass = (abs(open_mean - expected_open_libero) < 0.005) and \
                  (abs(closed_mean - expected_closed_libero) < 0.005)
    print(f"  → {'PASS' if check5_pass else 'FAIL'}\n")

    env.close()

    # ── Summary ──
    all_pass = check1_pass and check2_pass and check3_pass and check4_pass and check5_pass
    print("=" * 60)
    print(f"P1a Verification: {'ALL PASS ✓' if all_pass else 'SOME CHECKS FAILED ✗'}")
    print("=" * 60)
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())

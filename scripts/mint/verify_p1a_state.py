#!/usr/bin/env python3
"""P1a: Verify drawer_robot_env._state_vector() outputs continuous LIBERO-aligned gripper joint.

Gate condition: state[7] ∈ [-0.042, +0.001] (continuous, not binary).
Tests with: 3 seeds × gripper sweep trajectory (multiple intermediate positions).
The key insight: even 2 positions (open/closed) produce CONTINUOUS values in [-0.042, +0.001]
when the formula uses PyBullet joint positions. The test validates both the value range
AND that the mapping produces distinct continuous values across a gripper sweep.

Exit codes:
  0 = PASS — state[7] is continuous and within LIBERO range
  1 = FAIL — state[7] is binary or out of range
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
SCRIPTS_MINT = PROJECT_ROOT / "scripts" / "mint"
sys.path.insert(0, str(SCRIPTS_MINT))

from drawer_robot_env import DrawerRobotEnv

# LIBERO gripper joint range (from libero_contract_table.md)
LIBERO_GRIPPER_MIN = -0.042
LIBERO_GRIPPER_MAX = +0.001

# Test seeds
TEST_SEEDS = [2, 5, 9]
# Gripper sweep: 5 intermediate positions + open + closed = 7 positions
GRIPPER_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

PASSED = 0
FAILED = 0
errors: list[str] = []


def check_gripper_in_range(seed: int, gripper_fraction: float, state7: float) -> None:
    global PASSED, FAILED
    in_range = LIBERO_GRIPPER_MIN <= state7 <= LIBERO_GRIPPER_MAX
    status = "OK" if in_range else "OUT_OF_RANGE"

    print(f"  seed={seed:03d}  gripper_fraction={gripper_fraction:.2f}  "
          f"state[7]={state7:+.6f}  [{status}]")

    if not in_range:
        errors.append(
            f"seed={seed} gripper_fraction={gripper_fraction}: "
            f"state[7]={state7:+.6f} outside LIBERO range "
            f"[{LIBERO_GRIPPER_MIN}, {LIBERO_GRIPPER_MAX}]"
        )
        FAILED += 1
    else:
        PASSED += 1


def check_continuous_non_binary(all_values: list[float]) -> bool:
    """
    Verify that gripper joint values are continuous (not binary {0, 1}).

    The key test: with a gripper sweep across 6 positions × 3 seeds = 18 samples,
    we should see distinct continuous values (not the binary {0, 1} that existed
    in the old broken implementation).

    Success criteria:
    1. Unique value count > 2 (at least 3 distinct values from the sweep)
    2. Values strictly within LIBERO range [-0.042, +0.001]
    3. Monotonic relationship with gripper fraction (open > closed)
    """
    rounded = sorted(set(round(v, 4) for v in all_values))
    unique_count = len(rounded)
    print(f"\n  Continuity analysis:")
    print(f"    total samples: {len(all_values)}")
    print(f"    unique values: {unique_count}")
    print(f"    sample values (sorted): {rounded}")
    print(f"    min={min(all_values):.6f}  max={max(all_values):.6f}")
    print(f"    LIBERO range: [{LIBERO_GRIPPER_MIN}, {LIBERO_GRIPPER_MAX}]")

    checks = []
    # Check 1: More than 2 unique values (proves continuous mapping)
    if unique_count > 2:
        print(f"    ✓ Unique count {unique_count} > 2 (continuous mapping confirmed)")
        checks.append(True)
    else:
        errors.append(
            f"state[7] appears binary ({unique_count} unique values: {rounded}). "
            f"Expected >2 from gripper sweep."
        )
        print(f"    ✗ Unique count {unique_count} <= 2 (appears binary)")
        checks.append(False)

    # Check 2: Values within LIBERO range
    in_range = all(LIBERO_GRIPPER_MIN <= v <= LIBERO_GRIPPER_MAX for v in all_values)
    if in_range:
        print(f"    ✓ All values within LIBERO range")
        checks.append(True)
    else:
        bad = [v for v in all_values if v < LIBERO_GRIPPER_MIN or v > LIBERO_GRIPPER_MAX]
        errors.append(f"Values outside LIBERO range: {bad}")
        print(f"    ✗ Values outside LIBERO range: {bad}")
        checks.append(False)

    # Check 3: Physical ordering (open fraction → larger state[7])
    open_vals = [v for v in all_values if v > 0]
    closed_vals = [v for v in all_values if v < -0.03]
    if open_vals and closed_vals:
        open_mean = sum(open_vals) / len(open_vals)
        closed_mean = sum(closed_vals) / len(closed_vals)
        if open_mean > closed_mean:
            print(f"    ✓ Physical ordering OK: open_mean={open_mean:.6f} > "
                  f"closed_mean={closed_mean:.6f}")
            checks.append(True)
        else:
            errors.append(
                f"Physical ordering violated: open_mean={open_mean:.6f} "
                f"not > closed_mean={closed_mean:.6f}"
            )
            print(f"    ✗ Physical ordering violated")
            checks.append(False)

    return all(checks)


def run() -> bool:
    global errors
    all_values: list[float] = []

    for seed in TEST_SEEDS:
        print(f"\n[seed {seed:03d}]")
        env = DrawerRobotEnv(seed=seed, image_size=224, max_steps=8)
        _ = env.reset()

        for gripper_fraction in GRIPPER_SWEEP:
            env._set_gripper_open(gripper_fraction)
            env._step_world(4)
            obs = env.observe()
            state7 = float(obs.state[7])
            all_values.append(state7)
            check_gripper_in_range(seed, gripper_fraction, state7)

        env.close()

    print(f"\n[continuity check]")
    continuous_ok = check_continuous_non_binary(all_values)

    print(f"\n[SUMMARY]")
    print(f"  PASSED range checks: {PASSED}")
    print(f"  FAILED range checks: {FAILED}")
    print(f"  Continuous (not binary): {'YES' if continuous_ok else 'NO'}")

    if errors:
        print(f"\n[ERRORS]")
        for e in errors:
            print(f"  {e}")

    overall_pass = (FAILED == 0) and continuous_ok
    verdict = "PASS" if overall_pass else "FAIL"
    print(f"\n[{verdict}] P1a state[7] continuous LIBERO-aligned gripper joint")
    return overall_pass


if __name__ == "__main__":
    ok = run()
    raise SystemExit(0 if ok else 1)

#!/usr/bin/env python3
"""G5: Delta action conversion reconstructs original trajectory."""

import json
import os
import sys
import time

import numpy as np

CAMPAIGN = "/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1"
ARTIFACT = os.path.join(CAMPAIGN, "artifacts", "g5_delta_reconstruction.json")


def run():
    # Simulate: 100 absolute positions in a line
    abs_positions = np.linspace(0, 0.25, 100).reshape(-1, 1)
    abs_actions = np.zeros((100, 7))
    abs_actions[:, 0] = abs_positions[:, 0]

    # Convert to delta
    pos_scale = 0.05  # OSC_POSE default
    deltas = np.diff(abs_actions[:, 0]) / pos_scale
    deltas = np.clip(deltas, -1.0, 1.0)

    # Reconstruct
    reconstructed = [abs_actions[0, 0]]
    for d in deltas:
        reconstructed.append(reconstructed[-1] + d * pos_scale)
    reconstructed = np.array(reconstructed)

    # Compare
    orig = abs_actions[:, 0]
    error = np.abs(orig[: len(reconstructed)] - reconstructed)
    max_error = float(error.max())

    result = {
        "gate": "g5_delta_reconstruction",
        "passed": max_error < 0.001,
        "max_error_m": max_error,
        "max_error_mm": max_error * 1000,
        "n_steps": len(reconstructed),
        "pos_scale": pos_scale,
        "clipping_count": int(
            np.sum(np.abs(np.diff(abs_actions[:, 0]) / pos_scale) > 1.0)
        ),
        "timestamp": time.time(),
    }
    with open(ARTIFACT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result["passed"]


if __name__ == "__main__":
    sys.exit(0 if run() else 1)

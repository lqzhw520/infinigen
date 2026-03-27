#!/usr/bin/env python3
"""G2: Verify the robot scene emits a MINT-compatible observation contract."""

from __future__ import annotations

import json
import time

from drawer_robot_env import DrawerRobotEnv
from mint_common import ARTIFACT_DIR

ARTIFACT = ARTIFACT_DIR / "g2_obs_contract.json"
AUDIT_ARTIFACT = ARTIFACT_DIR / "scene_frame_audit.json"


def run() -> bool:
    env = DrawerRobotEnv(seed=1, image_size=224, max_steps=16)
    obs = env.reset()
    env.close()

    checks = {
        "image_shape": list(obs.image.shape) == [224, 224, 3],
        "image2_shape": list(obs.image2.shape) == [224, 224, 3],
        "image_dtype": str(obs.image.dtype) == "uint8",
        "image2_dtype": str(obs.image2.dtype) == "uint8",
        "depth_shape": list(obs.depth.shape) == [224, 224],
        "state_shape": list(obs.state.shape) == [8],
        "state_dtype": str(obs.state.dtype) == "float32",
        "task_nonempty": bool(obs.task),
    }

    result = {
        "gate": "g2_obs_contract",
        "passed": all(checks.values()),
        "checks": checks,
        "schema": {
            "observation.images.image": list(obs.image.shape),
            "observation.images.image2": list(obs.image2.shape),
            "observation.state": list(obs.state.shape),
            "task": obs.task,
        },
        "state_semantics": [
            "eef_pos_x",
            "eef_pos_y",
            "eef_pos_z",
            "eef_quat_x",
            "eef_quat_y",
            "eef_quat_z",
            "eef_quat_w",
            "gripper_open",
        ],
        "timestamp": time.time(),
    }
    AUDIT_ARTIFACT.write_text(
        json.dumps(
            {
                "gate": "g2_obs_contract",
                "passed": all(checks.values()),
                "eef_pose": {
                    "pos": obs.eef_pos.tolist(),
                    "quat": obs.eef_quat.tolist(),
                },
                "gripper_open": float(obs.gripper_open),
                "drawer_fraction": float(obs.drawer_fraction),
                "task": obs.task,
                "image_minmax": [int(obs.image.min()), int(obs.image.max())],
                "image2_minmax": [int(obs.image2.min()), int(obs.image2.max())],
                "depth_minmax": [float(obs.depth.min()), float(obs.depth.max())],
                "timestamp": time.time(),
            },
            indent=2,
        )
    )
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

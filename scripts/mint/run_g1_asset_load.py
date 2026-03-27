#!/usr/bin/env python3
"""G1: Ensure drawer assets exist and a robot + drawer simulation scene can load."""

from __future__ import annotations

import json
import subprocess
import time

from drawer_robot_env import DrawerRobotEnv, drawer_assets_available, drawer_manifest
from mint_common import (
    ARTIFACT_DIR,
    CONDA_SETUP,
    DEFAULT_HELD_OUT_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    PROJECT_ROOT,
)

ARTIFACT = ARTIFACT_DIR / "g1_asset_load.json"
AUDIT_ARTIFACT = ARTIFACT_DIR / "urdf_asset_audit.json"


def ensure_drawer_variants() -> dict:
    required = DEFAULT_TRAIN_SEEDS + DEFAULT_HELD_OUT_SEEDS
    if drawer_assets_available(required):
        return {"generated": False, "required_seeds": required}
    cmd = (
        f"{CONDA_SETUP} && conda activate infinigen && cd {PROJECT_ROOT} && "
        "python -m infinigen.launch_blender -s scripts/mint/export_drawerbox_variants.py"
    )
    result = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    return {
        "generated": True,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
        "required_seeds": required,
    }


def run() -> bool:
    generation = ensure_drawer_variants()
    required = DEFAULT_TRAIN_SEEDS + DEFAULT_HELD_OUT_SEEDS
    available = drawer_assets_available(required)
    load_checks = []
    audit_checks = []
    if available:
        for seed in [
            DEFAULT_TRAIN_SEEDS[0],
            DEFAULT_TRAIN_SEEDS[-1],
            DEFAULT_HELD_OUT_SEEDS[0],
            DEFAULT_HELD_OUT_SEEDS[-1],
        ]:
            env = DrawerRobotEnv(seed=seed, image_size=224, max_steps=16)
            obs = env.reset()
            load_checks.append(
                {
                    "seed": seed,
                    "image_shape": list(obs.image.shape),
                    "image2_shape": list(obs.image2.shape),
                    "state_shape": list(obs.state.shape),
                    "eef_pos": obs.eef_pos.tolist(),
                    "drawer_fraction": obs.drawer_fraction,
                }
            )
            aabb_min, aabb_max = env.p.getAABB(
                env.drawer_id, env.drawer_joint, physicsClientId=env.client
            )
            audit_checks.append(
                {
                    "seed": seed,
                    "drawer_joint_index": env.drawer_joint,
                    "drawer_range": [float(env.drawer_low), float(env.drawer_high)],
                    "drawer_motion_axis": env.drawer_motion_axis.tolist(),
                    "drawer_travel_distance": float(env.drawer_travel_distance),
                    "drawer_aabb_world": [list(aabb_min), list(aabb_max)],
                }
            )
            env.close()
    result = {
        "gate": "g1_asset_load",
        "passed": bool(available),
        "required_seeds": required,
        "drawer_manifest": drawer_manifest(),
        "generation": generation,
        "load_checks": load_checks,
        "timestamp": time.time(),
    }
    AUDIT_ARTIFACT.write_text(
        json.dumps(
            {
                "gate": "g1_asset_load",
                "passed": bool(available and audit_checks),
                "required_seeds": required,
                "sampled_seed_count": len(audit_checks),
                "records": audit_checks,
                "timestamp": time.time(),
            },
            indent=2,
        )
    )
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

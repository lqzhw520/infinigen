#!/usr/bin/env python3
"""G3a: Render robot-scene RGB-D / point clouds for AnyGrasp across all campaign seeds."""

from __future__ import annotations

import json
import time

import numpy as np
from drawer_robot_env import DrawerRobotEnv
from mint_common import ARTIFACT_DIR, DEFAULT_HELD_OUT_SEEDS, DEFAULT_TRAIN_SEEDS

ARTIFACT = ARTIFACT_DIR / "g3_render_depth.json"
RENDER_CACHE = ARTIFACT_DIR / "g3_scene_renders"


def run() -> bool:
    RENDER_CACHE.mkdir(parents=True, exist_ok=True)
    seeds = DEFAULT_TRAIN_SEEDS + DEFAULT_HELD_OUT_SEEDS
    records = []
    first_payload = None

    for seed in seeds:
        env = DrawerRobotEnv(seed=seed, image_size=320, max_steps=16)
        payload = env.anygrasp_payload()
        env.close()

        cache_path = RENDER_CACHE / f"seed_{seed:03d}.npz"
        np.savez_compressed(
            cache_path,
            depth=payload["depth"].astype(np.float32),
            pc=payload["points"].astype(np.float32),
            colors=payload["colors"].astype(np.float32),
            limits=payload["limits"].astype(np.float32),
            intrinsics=payload["intrinsics"].astype(np.float32),
            image=payload["image"].astype(np.uint8),
            image2=payload["image2"].astype(np.uint8),
            state=payload["state"].astype(np.float32),
            camera_view=payload["camera_view"].astype(np.float32),
            world_from_camera=payload["world_from_camera"].astype(np.float32),
            drawer_aabb_world=payload["drawer_aabb_world"].astype(np.float32),
            drawer_motion_axis=payload["drawer_motion_axis"].astype(np.float32),
            handle_center_world=payload["handle_center_world"].astype(np.float32),
        )

        if first_payload is None:
            first_payload = payload

        records.append(
            {
                "seed": seed,
                "point_cloud_shape": list(payload["points"].shape),
                "depth_shape": list(payload["depth"].shape),
                "cache_path": str(cache_path),
            }
        )

    if first_payload is not None:
        np.savez_compressed(
            ARTIFACT_DIR / "drawer_depth.npz",
            depth=first_payload["depth"].astype(np.float32),
            pc=first_payload["points"].astype(np.float32),
            colors=first_payload["colors"].astype(np.float32),
            limits=first_payload["limits"].astype(np.float32),
            view=first_payload["camera_view"].astype(np.float32),
            world_from_camera=first_payload["world_from_camera"].astype(np.float32),
            drawer_aabb_world=first_payload["drawer_aabb_world"].astype(np.float32),
            drawer_motion_axis=first_payload["drawer_motion_axis"].astype(np.float32),
            handle_center_world=first_payload["handle_center_world"].astype(np.float32),
        )
        np.save(
            ARTIFACT_DIR / "drawer_intrinsics.npy",
            first_payload["intrinsics"].astype(np.float32),
        )
        np.save(
            ARTIFACT_DIR / "drawer_rgb.npy", first_payload["image"].astype(np.uint8)
        )

    passed = len(records) == len(seeds) and all(
        item["point_cloud_shape"][0] > 256 for item in records
    )
    result = {
        "gate": "g3_render_depth",
        "passed": passed,
        "render_cache_dir": str(RENDER_CACHE),
        "seed_count": len(records),
        "required_seed_count": len(seeds),
        "records": records,
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

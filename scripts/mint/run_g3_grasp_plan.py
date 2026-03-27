#!/usr/bin/env python3
"""G3c: Run AnyGrasp on drawer scenes and cache pull-region grasps for robot rollouts."""

from __future__ import annotations

import json
import time

import numpy as np
from anygrasp_helper import build_anygrasp, stage_detection_assets
from mint_common import ARTIFACT_DIR, DEFAULT_HELD_OUT_SEEDS, DEFAULT_TRAIN_SEEDS

ARTIFACT = ARTIFACT_DIR / "g3_grasp_plan.json"
AUDIT_ARTIFACT = ARTIFACT_DIR / "g3_grasp_audit.json"
HANDLE_AUDIT_ARTIFACT = ARTIFACT_DIR / "handle_region_audit.json"
GRASP_CACHE = ARTIFACT_DIR / "g3_anygrasp_grasps"
RENDER_CACHE = ARTIFACT_DIR / "g3_scene_renders"
MIN_SCORE = 0.50
MIN_VALID_TRAIN_SEEDS = 6


def pull_region_hit(
    pull_region_center_world: np.ndarray,
    drawer_aabb_world: np.ndarray,
    drawer_motion_axis: np.ndarray,
    grasp_pose_world: np.ndarray,
) -> bool:
    center = grasp_pose_world[:3, 3]
    mins = np.asarray(drawer_aabb_world[0], dtype=np.float32)
    maxs = np.asarray(drawer_aabb_world[1], dtype=np.float32)
    pull_region_center_world = np.asarray(pull_region_center_world, dtype=np.float32)
    axis = np.asarray(drawer_motion_axis, dtype=np.float32)
    dominant_axis = int(np.argmax(np.abs(axis)))
    tangential_axes = [idx for idx in range(3) if idx != dominant_axis]
    axis_ok = (
        abs(float(center[dominant_axis] - pull_region_center_world[dominant_axis]))
        <= 0.06
    )
    tangential_ok = all(
        mins[idx] - 0.04 <= center[idx] <= maxs[idx] + 0.04 for idx in tangential_axes
    )
    dist_ok = float(np.linalg.norm(center - pull_region_center_world)) <= 0.10
    return bool(axis_ok and tangential_ok and dist_ok)


def run() -> bool:
    setup = stage_detection_assets()
    if not setup.get("passed"):
        setup["gate"] = "g3_grasp_plan"
        setup["timestamp"] = time.time()
        ARTIFACT.write_text(json.dumps(setup, indent=2))
        print(json.dumps(setup, indent=2))
        return False

    detector = build_anygrasp()
    GRASP_CACHE.mkdir(parents=True, exist_ok=True)
    for stale in GRASP_CACHE.glob("seed_*.json"):
        stale.unlink()

    records = []
    train_passes = 0
    heldout_passes = 0
    all_seeds = DEFAULT_TRAIN_SEEDS + DEFAULT_HELD_OUT_SEEDS
    for seed in all_seeds:
        cache_path = RENDER_CACHE / f"seed_{seed:03d}.npz"
        if not cache_path.exists():
            cache_payload = {
                "seed": seed,
                "passed": False,
                "selected_grasp": None,
                "top_candidates": [],
                "min_score_threshold": MIN_SCORE,
                "region_semantics": "drawer_front_pull_region",
                "error": f"Missing render cache {cache_path}",
                "timestamp": time.time(),
            }
            (GRASP_CACHE / f"seed_{seed:03d}.json").write_text(
                json.dumps(cache_payload, indent=2)
            )
            records.append(
                {
                    "seed": seed,
                    "passed": False,
                    "error": f"Missing render cache {cache_path}",
                }
            )
            continue

        payload = dict(np.load(cache_path))
        world_from_camera = payload["world_from_camera"].astype(np.float32)
        pull_region_center_world = payload["handle_center_world"].astype(np.float32)
        drawer_aabb_world = payload["drawer_aabb_world"].astype(np.float32)
        drawer_motion_axis = payload["drawer_motion_axis"].astype(np.float32)
        points = payload["pc"].astype(np.float32)
        colors = payload["colors"].astype(np.float32)
        lims = payload["limits"].astype(np.float32)
        crop_mask = (
            (points[:, 0] >= lims[0])
            & (points[:, 0] <= lims[1])
            & (points[:, 1] >= lims[2])
            & (points[:, 1] <= lims[3])
            & (points[:, 2] >= lims[4])
            & (points[:, 2] <= lims[5])
        )
        points = points[crop_mask]
        colors = colors[crop_mask]
        if len(points) < 512:
            cache_payload = {
                "seed": seed,
                "passed": False,
                "selected_grasp": None,
                "top_candidates": [],
                "min_score_threshold": MIN_SCORE,
                "region_semantics": "drawer_front_pull_region",
                "error": "Workspace crop too sparse for AnyGrasp",
                "timestamp": time.time(),
            }
            (GRASP_CACHE / f"seed_{seed:03d}.json").write_text(
                json.dumps(cache_payload, indent=2)
            )
            records.append(
                {
                    "seed": seed,
                    "passed": False,
                    "error": "Workspace crop too sparse for AnyGrasp",
                }
            )
            continue

        gg, _cloud = detector.get_grasp(
            points,
            colors,
            lims=lims.tolist(),
            apply_object_mask=True,
            dense_grasp=False,
            collision_detection=True,
        )

        top_candidates = []
        selected = None
        passed = False
        if len(gg) > 0:
            gg = gg.sort_by_score()
            for idx in range(min(len(gg), 10)):
                grasp = gg[idx]
                pose_camera = np.eye(4, dtype=np.float32)
                pose_camera[:3, :3] = np.asarray(
                    grasp.rotation_matrix, dtype=np.float32
                )
                pose_camera[:3, 3] = np.asarray(grasp.translation, dtype=np.float32)
                pose_world = world_from_camera @ pose_camera
                hit = pull_region_hit(
                    pull_region_center_world,
                    drawer_aabb_world,
                    drawer_motion_axis,
                    pose_world,
                )
                item = {
                    "rank": idx + 1,
                    "score": float(grasp.score),
                    "pull_region_hit": hit,
                    "translation_camera": pose_camera[:3, 3].tolist(),
                    "rotation_matrix_camera": pose_camera[:3, :3].tolist(),
                    "pose_camera": pose_camera.tolist(),
                    "translation_world": pose_world[:3, 3].tolist(),
                    "rotation_matrix_world": pose_world[:3, :3].tolist(),
                    "pose_world": pose_world.tolist(),
                }
                top_candidates.append(item)
                if selected is None:
                    selected = item
                if float(grasp.score) >= MIN_SCORE and hit:
                    selected = item
                    passed = True
                    break

        if seed in DEFAULT_TRAIN_SEEDS and passed:
            train_passes += 1
        if seed in DEFAULT_HELD_OUT_SEEDS and passed:
            heldout_passes += 1

        cache_payload = {
            "seed": seed,
            "passed": passed,
            "selected_grasp": selected,
            "top_candidates": top_candidates,
            "min_score_threshold": MIN_SCORE,
            "region_semantics": "drawer_front_pull_region",
            "timestamp": time.time(),
        }
        (GRASP_CACHE / f"seed_{seed:03d}.json").write_text(
            json.dumps(cache_payload, indent=2)
        )
        records.append(
            {
                "seed": seed,
                "passed": passed,
                "top_score": top_candidates[0]["score"] if top_candidates else 0.0,
                "top_pull_region_hit": top_candidates[0]["pull_region_hit"]
                if top_candidates
                else False,
                "selected_score": selected["score"] if selected else 0.0,
                "selected_pull_region_hit": selected["pull_region_hit"]
                if selected
                else False,
            }
        )

    cache_complete = all(
        (GRASP_CACHE / f"seed_{seed:03d}.json").exists() for seed in all_seeds
    )
    result = {
        "gate": "g3_grasp_plan",
        "passed": cache_complete,
        "audit_passed": train_passes >= MIN_VALID_TRAIN_SEEDS,
        "cache_complete": cache_complete,
        "min_score_threshold": MIN_SCORE,
        "min_valid_train_seeds": MIN_VALID_TRAIN_SEEDS,
        "region_semantics": "drawer_front_pull_region",
        "train_passes": train_passes,
        "train_required": MIN_VALID_TRAIN_SEEDS,
        "train_seed_count": len(DEFAULT_TRAIN_SEEDS),
        "held_out_passes": heldout_passes,
        "held_out_seed_count": len(DEFAULT_HELD_OUT_SEEDS),
        "cache_dir": str(GRASP_CACHE),
        "records": records,
        "timestamp": time.time(),
    }
    AUDIT_ARTIFACT.write_text(json.dumps(result, indent=2))
    HANDLE_AUDIT_ARTIFACT.write_text(json.dumps(result, indent=2))
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

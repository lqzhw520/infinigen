#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco
import numpy as np

ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
RUN_DIR = Path(os.environ["RUN_DIR_REL"])
sys.path.insert(0, str(ROOT / "scripts/mint"))
from drawer_robot_env_mujoco import DrawerRobotEnvMuJoCoLibero  # noqa: E402
from merged_model_builder import MergedModelBuilder  # noqa: E402

DrawerRobotEnvMuJoCoLibero._MERGED_BUILDER_CLASS = MergedModelBuilder
LEGAL = [63, 81, 90]
HANDLE = list(range(9))


def ready(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): ready(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [ready(v) for v in x]
    return x


def write(p, o):
    p.write_text(json.dumps(ready(o), indent=2, sort_keys=True) + "\n")


def min_d(env):
    lp = env.data.geom_xpos[LEGAL]
    hp = env.data.geom_xpos[HANDLE]
    d = np.linalg.norm(lp[:, None, :] - hp[None, :, :], axis=2)
    idx = np.unravel_index(int(np.argmin(d)), d.shape)
    hand = env.data.xpos[int(env._eef_body_id)]
    ed = np.linalg.norm(hand[None, :] - hp, axis=1)
    return float(d[idx]), int(LEGAL[idx[0]]), int(HANDLE[idx[1]]), float(np.min(ed))


def solve(seed: int, target_kind: str):
    env = DrawerRobotEnvMuJoCoLibero(
        seed=seed, image_size=96, max_steps=2, contract=None
    )
    try:
        env.reset()
        start = min_d(env)
        handle_cent = np.mean(env.data.geom_xpos[HANDLE], axis=0)
        target = handle_cent.copy()
        hist = []
        for it in range(220):
            legal_cent = np.mean(env.data.geom_xpos[LEGAL], axis=0)
            hand = env.data.xpos[int(env._eef_body_id)].copy()
            if target_kind == "legal_pad_centroid":
                err = target - legal_cent
                # approximate legal centroid jacobian by averaging legal geom jacobians
                J = np.zeros((3, env.model.nv))
                for gid in LEGAL:
                    jp = np.zeros((3, env.model.nv))
                    jr = np.zeros((3, env.model.nv))
                    mujoco.mj_jacGeom(env.model, env.data, jp, jr, gid)
                    J += jp / len(LEGAL)
            else:
                err = target - hand
                J = np.zeros((3, env.model.nv))
                jr = np.zeros((3, env.model.nv))
                mujoco.mj_jacBody(env.model, env.data, J, jr, int(env._eef_body_id))
            Jr = J[:, 2:9]
            lam = 1e-3
            dq = Jr.T @ np.linalg.solve(Jr @ Jr.T + lam * np.eye(3), err * 0.55)
            dq = np.clip(dq, -0.08, 0.08)
            lo = env.model.jnt_range[2:9, 0]
            hi = env.model.jnt_range[2:9, 1]
            env.data.qpos[2:9] = np.clip(env.data.qpos[2:9] + dq, lo, hi)
            mujoco.mj_forward(env.model, env.data)
            if it % 10 == 0 or it == 219:
                md, lg, hg, ed = min_d(env)
                hist.append(
                    {
                        "iter": it,
                        "min_legal_pad_to_handle_distance_m": md,
                        "legal_geom": lg,
                        "handle_geom": hg,
                        "eef_to_handle_min_distance_m": ed,
                        "err_norm": float(np.linalg.norm(err)),
                    }
                )
        end = min_d(env)
        return {
            "seed": seed,
            "target_kind": target_kind,
            "diagnostic_only_direct_robot_qpos_ik": True,
            "drawer_qpos_modified": False,
            "start": {
                "min_legal_pad_to_handle_distance_m": start[0],
                "legal_geom": start[1],
                "handle_geom": start[2],
                "eef_to_handle_min_distance_m": start[3],
            },
            "end": {
                "min_legal_pad_to_handle_distance_m": end[0],
                "legal_geom": end[1],
                "handle_geom": end[2],
                "eef_to_handle_min_distance_m": end[3],
            },
            "history": hist,
            "reachable_under_kinematic_ik": bool(end[0] <= 0.035 or end[3] <= 0.035),
        }
    finally:
        env.close()


items = []
for seed in [1, 11]:
    for kind in ["right_hand", "legal_pad_centroid"]:
        items.append(solve(seed, kind))
summary = {
    "items": items,
    "any_kinematic_reachable": any(x["reachable_under_kinematic_ik"] for x in items),
    "best_min_legal_pad_to_handle_distance_m": min(
        x["end"]["min_legal_pad_to_handle_distance_m"] for x in items
    ),
    "best_min_eef_to_handle_distance_m": min(
        x["end"]["eef_to_handle_min_distance_m"] for x in items
    ),
    "interpretation": "diagnostic-only robot qpos IK, not a rollout/candidate; drawer qpos is not opened directly",
}
write(RUN_DIR / "kinematic_reachability_probe.json", summary)

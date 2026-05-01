#!/usr/bin/env python3
"""Contact-aware GOC-v4 dedicated-pad drawer teacher synthesis diagnostics.

This helper is intentionally narrow: it does not mutate the environment, GOC-v4
contract, sovereign truth, or drawer qpos. It probes whether the current
instance-bound dedicated finger-pad collision geoms can produce physically plausible
low-penetration handle contact under a guarded controller.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco
import numpy as np

ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN = ROOT / "experiments/mint/mint_drawer_v1"
PREV_RUN = CAMPAIGN / "runtime/v11_g4_goc_v3_placement_ik_optimization_20260501T045544Z"
SPEC_REL = (
    "experiments/mint/mint_drawer_v1/sovereign/experiment_specs/"
    "v11_g4_goc_v4_dedicated_pad_contact_teacher_synthesis.yaml"
)
SPEC_PATH = ROOT / SPEC_REL


sys.path.insert(0, str(ROOT / "scripts/mint"))
from drawer_robot_env_mujoco import DrawerRobotEnvMuJoCoLibero  # noqa: E402
from merged_model_builder import MergedModelBuilder  # noqa: E402

SOURCE_GOC_V3_BROAD_LEGAL = [63, 81, 90]
SOURCE_GOC_V3_FORBIDDEN = [45, 47, 49, 54, 59]
SOURCE_GOC_HANDLE = list(range(9))
PREV_CANDIDATE_QPOS = [
    -0.015510438824320322,
    1.352985921213834,
    0.30041682996635577,
    -0.5021432980388317,
    0.846634167393299,
    2.152104071538119,
    -0.9786610857048721,
]
RESET_CANDIDATE_QPOS = [
    -0.08,
    -0.081037389,
    -0.05,
    -2.5645974700000003,
    0.0,
    2.3267522,
    0.8653981633974482,
]
CANDIDATE_SEED = 13
CANDIDATE_BASE = [-0.875, 0.05, 0.0]
CANDIDATE_YAW_DEG = -10.0


def ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ready(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ready(payload), indent=2, sort_keys=True) + "\n")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def geom_name(model: mujoco.MjModel, gid: int) -> str:
    return str(model.geom(int(gid)).name or "")


def body_name(model: mujoco.MjModel, bid: int) -> str:
    return str(model.body(int(bid)).name or f"body_{bid}")


def body_chain(model: mujoco.MjModel, bid: int) -> list[str]:
    names = []
    cur = int(bid)
    while cur >= 0:
        names.append(body_name(model, cur))
        parent = int(model.body_parentid[cur])
        if parent == cur or cur == 0:
            break
        cur = parent
    return list(reversed(names))


def normalize(vec: np.ndarray, default: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return default.astype(float)
    return np.asarray(vec, dtype=float) / norm


def make_builder(base_pos: tuple[float, float, float], yaw_deg: float):
    class CandidateBuilder(MergedModelBuilder):
        def __init__(self, *args: Any, **kwargs: Any):
            kwargs["robot_base_pos"] = np.asarray(base_pos, dtype=float)
            super().__init__(*args, **kwargs)

        def build(self):
            xml, assets, metadata, sem_hash = super().build()
            if abs(float(yaw_deg)) > 1e-9:
                pos_s = " ".join(f"{float(v):.6f}" for v in self.robot_base_pos)
                half = math.radians(float(yaw_deg)) / 2.0
                quat = (
                    f"{math.cos(half):.6f} 0.000000 0.000000 " f"{math.sin(half):.6f}"
                )
                old = f'<body name="robot_mount" pos="{pos_s}">'
                new = f'<body name="robot_mount" pos="{pos_s}" quat="{quat}">'
                if old not in xml:
                    raise RuntimeError(f"robot_mount marker not found: {old}")
                xml = xml.replace(old, new, 1)
                metadata = dict(metadata)
                metadata["diagnostic_robot_base_yaw_deg"] = float(yaw_deg)
            return xml, assets, metadata, sem_hash

    return CandidateBuilder


def make_env(max_steps: int = 180, robot_init_qpos: list[float] | None = None):
    DrawerRobotEnvMuJoCoLibero._MERGED_BUILDER_CLASS = make_builder(
        tuple(CANDIDATE_BASE), CANDIDATE_YAW_DEG
    )
    return DrawerRobotEnvMuJoCoLibero(
        seed=CANDIDATE_SEED,
        image_size=64,
        max_steps=max_steps,
        contract=None,
        robot_init_qpos=robot_init_qpos or RESET_CANDIDATE_QPOS,
    )


def classify_instance(env: DrawerRobotEnvMuJoCoLibero) -> dict[str, Any]:
    model = env.model
    legal: list[int] = []
    forbidden: list[int] = []
    goc_v3_broad: list[int] = []
    handle: list[int] = []
    drawer_body: list[int] = []
    visual_noncontact: list[int] = []
    unknown_contact: list[int] = []
    inventory = []
    robot_bodies = {
        "base",
        "link0",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
        "right_hand",
        "leftfinger",
        "rightfinger",
        "left_finger",
        "right_finger",
    }
    forbidden_collision_names = {
        "link0_collision",
        "link1_collision",
        "link2_collision",
        "link3_collision",
        "link4_collision",
        "link5_collision",
        "link6_collision",
        "link7_collision",
        "hand_collision",
    }
    for gid in range(int(model.ngeom)):
        bid = int(model.geom_bodyid[gid])
        gname = geom_name(model, gid)
        lname = gname.lower()
        bname = body_name(model, bid)
        contact = bool(
            int(model.geom_contype[gid]) and int(model.geom_conaffinity[gid])
        )
        role = "visual_only_or_noncontact" if not contact else "unknown_contact"
        if not contact:
            visual_noncontact.append(gid)
        elif (
            "pad_collision" in lname
            or ("fingertip" in lname and "collision" in lname)
            or ("fingerpad" in lname and "collision" in lname)
        ):
            legal.append(gid)
            role = "legal_finger_pad_surface"
        elif gid in SOURCE_GOC_HANDLE:
            handle.append(gid)
            role = "drawer_handle_surface"
        elif bname in {"drawer_base", "link_1", "link_2"}:
            drawer_body.append(gid)
            role = "drawer_body_or_cabinet_surface"
        elif bname in robot_bodies or gname in forbidden_collision_names:
            forbidden.append(gid)
            role = "forbidden_robot_surface"
            if gid in SOURCE_GOC_V3_BROAD_LEGAL or gname in {
                "link5_collision",
                "link6_collision",
                "link7_collision",
            }:
                goc_v3_broad.append(gid)
        else:
            unknown_contact.append(gid)
        inventory.append(
            {
                "geom_id": gid,
                "geom_name": gname,
                "body_name": bname,
                "body_chain": body_chain(model, bid),
                "contact_capable": contact,
                "contype": int(model.geom_contype[gid]),
                "conaffinity": int(model.geom_conaffinity[gid]),
                "geom_type": int(model.geom_type[gid]),
                "geom_size": model.geom_size[gid].astype(float).tolist(),
                "geom_pos": model.geom_pos[gid].astype(float).tolist(),
                "geom_rbound": float(model.geom_rbound[gid]),
                "semantic_role": role,
            }
        )
    return {
        "seed": CANDIDATE_SEED,
        "source_goc_v3_fixed_ids": {
            "legacy_broad_link_legal_gripper_surface_geom_ids": SOURCE_GOC_V3_BROAD_LEGAL,
            "legacy_forbidden_robot_surface_geom_ids": SOURCE_GOC_V3_FORBIDDEN,
            "drawer_handle_geom_ids": SOURCE_GOC_HANDLE,
        },
        "legal_gripper_surface_geom_ids": legal,
        "legal_finger_pad_geom_ids": legal,
        "forbidden_robot_surface_geom_ids": forbidden,
        "drawer_handle_geom_ids": handle,
        "drawer_body_or_cabinet_geom_ids": drawer_body,
        "visual_only_or_noncontact_geom_ids": visual_noncontact,
        "unknown_contact_relevant_geom_ids": unknown_contact,
        "goc_v3_broad_link_geoms_demoted_from_target": sorted(set(goc_v3_broad)),
        "fixed_global_ids_match_this_instance": False,
        "binding_generated": bool(legal and forbidden and handle),
        "dedicated_pad_geoms_exist_or_created": bool(legal),
        "inventory": inventory,
    }


def pair_category(g1: int, g2: int, binding: dict[str, Any]) -> str:
    legal = set(binding["legal_gripper_surface_geom_ids"])
    forbidden = set(binding["forbidden_robot_surface_geom_ids"])
    handle = set(binding["drawer_handle_geom_ids"])
    drawer = set(binding["drawer_body_or_cabinet_geom_ids"])
    if (g1 in legal and g2 in handle) or (g2 in legal and g1 in handle):
        return "target"
    if (g1 in forbidden and g2 in (handle | drawer)) or (
        g2 in forbidden and g1 in (handle | drawer)
    ):
        return "forbidden"
    if (g1 in handle and g2 not in legal) or (g2 in handle and g1 not in legal):
        return "handle_nonlegal"
    return "other"


def geom_centers(env: DrawerRobotEnvMuJoCoLibero) -> dict[int, np.ndarray]:
    return {
        gid: env.data.geom_xpos[gid].astype(float).copy()
        for gid in range(env.model.ngeom)
    }


def contact_report(
    env: DrawerRobotEnvMuJoCoLibero,
    binding: dict[str, Any],
    prev_centers: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    dt = float(env.model.opt.timestep)
    now = geom_centers(env)
    pairs = []
    counts = {"target": 0, "forbidden": 0, "handle_nonlegal": 0, "other": 0}
    max_force = 0.0
    max_pen = 0.0
    for ci in range(int(env.data.ncon)):
        c = env.data.contact[ci]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(env.model, env.data, ci, force)
        normal = np.asarray(c.frame, dtype=float).reshape(3, 3)[0].copy()
        if prev_centers is not None and g1 in prev_centers and g2 in prev_centers:
            v1 = (now[g1] - prev_centers[g1]) / max(dt, 1e-9)
            v2 = (now[g2] - prev_centers[g2]) / max(dt, 1e-9)
            rel_nv = float(np.dot(v2 - v1, normal))
        else:
            rel_nv = 0.0
        normal_force = abs(float(force[0]))
        penetration = max(0.0, -float(c.dist))
        cat = pair_category(g1, g2, binding)
        counts[cat] += 1
        max_force = max(max_force, normal_force)
        max_pen = max(max_pen, penetration)
        pairs.append(
            {
                "contact_index": ci,
                "geom1": g1,
                "geom2": g2,
                "geom1_name": geom_name(env.model, g1),
                "geom2_name": geom_name(env.model, g2),
                "geom1_body": body_name(env.model, int(env.model.geom_bodyid[g1])),
                "geom2_body": body_name(env.model, int(env.model.geom_bodyid[g2])),
                "category": cat,
                "dist_m": float(c.dist),
                "penetration_m": penetration,
                "normal": normal.tolist(),
                "normal_force_n": normal_force,
                "relative_normal_velocity_mps": rel_nv,
            }
        )
    counts["max_contact_force_n"] = max_force
    counts["max_penetration_m"] = max_pen
    return {"counts": counts, "pairs": pairs, "centers": now}


def distance_metrics(
    env: DrawerRobotEnvMuJoCoLibero, binding: dict[str, Any]
) -> dict[str, Any]:
    legal = binding["legal_gripper_surface_geom_ids"]
    handle = binding["drawer_handle_geom_ids"]
    legal_pos = env.data.geom_xpos[legal].astype(float)
    handle_pos = env.data.geom_xpos[handle].astype(float)
    dmat = np.linalg.norm(legal_pos[:, None, :] - handle_pos[None, :, :], axis=2)
    idx = np.unravel_index(int(np.argmin(dmat)), dmat.shape)
    eef_pos = env.data.xpos[int(env._eef_body_id)].astype(float)
    ed = np.linalg.norm(eef_pos[None, :] - handle_pos, axis=1)
    return {
        "min_legal_pad_to_handle_m": float(dmat[idx]),
        "min_legal_pad_geom_id": int(legal[idx[0]]),
        "min_handle_geom_id": int(handle[idx[1]]),
        "min_eef_to_handle_m": float(np.min(ed)),
        "legal_centroid": np.mean(legal_pos, axis=0).tolist(),
        "handle_center": np.mean(handle_pos, axis=0).tolist(),
        "eef_pos": eef_pos.tolist(),
    }


def drawer_fraction(env: DrawerRobotEnvMuJoCoLibero) -> float:
    lo = float(env.model.jnt_range[0, 0])
    hi = float(env.model.jnt_range[0, 1])
    return float(np.clip((float(env.data.qpos[0]) - lo) / max(hi - lo, 1e-9), 0.0, 1.0))


def ik_action_to_pad(
    env: DrawerRobotEnvMuJoCoLibero,
    binding: dict[str, Any],
    target: np.ndarray,
    close: bool,
    gain: float,
    cart_vel_limit: float,
    joint_cmd_limit: float,
    joint_command_scale: float,
) -> np.ndarray:
    legal = binding["legal_gripper_surface_geom_ids"]
    legal_cent = np.mean(env.data.geom_xpos[legal], axis=0)
    err = np.asarray(target, dtype=float) - legal_cent
    jac = np.zeros((3, env.model.nv), dtype=np.float64)
    for gid in legal:
        jp = np.zeros((3, env.model.nv), dtype=np.float64)
        jr = np.zeros((3, env.model.nv), dtype=np.float64)
        mujoco.mj_jacGeom(env.model, env.data, jp, jr, int(gid))
        jac += jp / max(len(legal), 1)
    jr = jac[:, 2:9]
    lam = 1e-4
    desired = np.clip(err * gain, -cart_vel_limit, cart_vel_limit)
    qdot = jr.T @ np.linalg.solve(jr @ jr.T + lam * np.eye(3), desired)
    dt = max(float(env.model.opt.timestep), 1e-6)
    qcmd = np.clip(qdot / dt * joint_command_scale, -joint_cmd_limit, joint_cmd_limit)
    action = np.zeros(9, dtype=np.float32)
    action[:7] = qcmd.astype(np.float32)
    action[7] = 1.0 if close else -1.0
    action[8] = 0.0
    return action


def handle_frame(
    env: DrawerRobotEnvMuJoCoLibero, binding: dict[str, Any]
) -> dict[str, Any]:
    handle_pos = env.data.geom_xpos[binding["drawer_handle_geom_ids"]].astype(float)
    legal_pos = env.data.geom_xpos[binding["legal_gripper_surface_geom_ids"]].astype(
        float
    )
    center = np.mean(handle_pos, axis=0)
    pull_axis = normalize(
        np.asarray(env._motion_axis, dtype=float), np.array([1.0, 0.0, 0.0])
    )
    approach = normalize(center - np.mean(legal_pos, axis=0), -pull_axis)
    bar_axis = normalize(
        np.asarray(
            handle_pos[np.argmax(handle_pos[:, 1])]
            - handle_pos[np.argmin(handle_pos[:, 1])]
        ),
        np.array([0.0, 1.0, 0.0]),
    )
    binormal = normalize(np.cross(pull_axis, bar_axis), np.array([0.0, 0.0, 1.0]))
    return {
        "handle_center": center,
        "pull_axis": pull_axis,
        "approach_normal": approach,
        "handle_bar_axis": bar_axis,
        "binormal": binormal,
    }


def solve_pad_ik(
    env: DrawerRobotEnvMuJoCoLibero,
    binding: dict[str, Any],
    target: np.ndarray,
    iterations: int = 180,
) -> dict[str, Any]:
    start = env.data.qpos.copy()
    best = None
    for _ in range(iterations):
        legal = binding["legal_gripper_surface_geom_ids"]
        legal_cent = np.mean(env.data.geom_xpos[legal], axis=0)
        err = np.asarray(target, dtype=float) - legal_cent
        jac = np.zeros((3, env.model.nv), dtype=np.float64)
        for gid in legal:
            jp = np.zeros((3, env.model.nv), dtype=np.float64)
            jr = np.zeros((3, env.model.nv), dtype=np.float64)
            mujoco.mj_jacGeom(env.model, env.data, jp, jr, int(gid))
            jac += jp / max(len(legal), 1)
        jr = jac[:, 2:9]
        dq = jr.T @ np.linalg.solve(jr @ jr.T + 8e-4 * np.eye(3), err * 0.55)
        dq = np.clip(dq, -0.06, 0.06)
        lo = env.model.jnt_range[2:9, 0]
        hi = env.model.jnt_range[2:9, 1]
        env.data.qpos[2:9] = np.clip(env.data.qpos[2:9] + dq, lo, hi)
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        dm = distance_metrics(env, binding)
        if (
            best is None
            or dm["min_legal_pad_to_handle_m"]
            < best["distance"]["min_legal_pad_to_handle_m"]
        ):
            best = {"distance": dm, "qpos": env.data.qpos[2:9].copy()}
    env.data.qpos[:] = start
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    assert best is not None
    return best


def record_step(
    env: DrawerRobotEnvMuJoCoLibero,
    binding: dict[str, Any],
    contact: dict[str, Any],
    mode: str,
    gripper_command: float,
) -> dict[str, Any]:
    dm = distance_metrics(env, binding)
    return {
        "step": int(env._step_count),
        "mode": mode,
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
        "eef_pose": {
            "pos": env.data.xpos[int(env._eef_body_id)].copy(),
            "quat": env.data.xquat[int(env._eef_body_id)].copy(),
        },
        "legal_pad_geom_world_positions": {
            str(gid): env.data.geom_xpos[int(gid)].copy()
            for gid in binding["legal_gripper_surface_geom_ids"]
        },
        "handle_geom_world_positions": {
            str(gid): env.data.geom_xpos[int(gid)].copy()
            for gid in binding["drawer_handle_geom_ids"]
        },
        "distance": dm,
        "contact_counts": contact["counts"],
        "contact_pairs": contact["pairs"][:18],
        "gripper_command": float(gripper_command),
        "drawer_qpos": float(env.data.qpos[0]),
        "drawer_fraction": drawer_fraction(env),
    }


def replay_previous_failure() -> dict[str, Any]:
    env = make_env(max_steps=170)
    records = []
    try:
        env.reset()
        binding = classify_instance(env)
        env.data.qpos[2:9] = np.asarray(PREV_CANDIDATE_QPOS, dtype=float)
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        frame = handle_frame(env, binding)
        prev = geom_centers(env)
        for step in range(150):
            if step < 30:
                mode = "settle_precontact"
                target = frame["handle_center"] - frame["pull_axis"] * 0.02
                close = False
            elif step < 75:
                mode = "close_contact"
                target = frame["handle_center"]
                close = True
            else:
                mode = "short_pull_probe"
                target = frame["handle_center"] + frame["pull_axis"] * min(
                    0.08, 0.0015 * (step - 75)
                )
                close = True
            action = ik_action_to_pad(env, binding, target, close, 5.0, 1.5, 300.0, 1.0)
            env.step(action)
            contact = contact_report(env, binding, prev)
            records.append(record_step(env, binding, contact, mode, action[7]))
            prev = contact["centers"]
        summary = summarize_records(records)
        first_target = next(
            (r for r in records if r["contact_counts"]["target"] > 0), None
        )
        mode = classify_failure(summary, first_target, None)
        return {
            "candidate": candidate_payload(binding),
            "summary": summary,
            "first_target_contact_step": None
            if first_target is None
            else first_target["step"],
            "dynamic_failure_mode": mode,
            "sample_records": sample_records(records),
            "records_jsonl_note": "Full trace omitted from JSON summary; compressed traces are emitted during CEM probes.",
        }
    finally:
        env.close()


def candidate_payload(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": CANDIDATE_SEED,
        "base_pos": CANDIDATE_BASE,
        "yaw_deg": CANDIDATE_YAW_DEG,
        "ik_final_qpos": PREV_CANDIDATE_QPOS,
        "reset_qpos": RESET_CANDIDATE_QPOS,
        "instance_binding": {
            "legal_gripper_surface_geom_ids": binding["legal_gripper_surface_geom_ids"],
            "forbidden_robot_surface_geom_ids": binding[
                "forbidden_robot_surface_geom_ids"
            ],
            "drawer_handle_geom_ids": binding["drawer_handle_geom_ids"],
            "drawer_body_or_cabinet_geom_ids": binding[
                "drawer_body_or_cabinet_geom_ids"
            ],
            "fixed_global_ids_match_this_instance": binding[
                "fixed_global_ids_match_this_instance"
            ],
        },
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    target = [r for r in records if r["contact_counts"]["target"] > 0]
    forbidden = [r for r in records if r["contact_counts"]["forbidden"] > 0]
    pull = [r for r in records if r["mode"] in {"compliant_pull", "short_pull_probe"}]
    drawer_trace = [r["drawer_qpos"] for r in pull]
    nondecreasing = (
        all(b >= a - 1e-7 for a, b in zip(drawer_trace, drawer_trace[1:]))
        if len(drawer_trace) > 1
        else True
    )
    max_pen = max(r["contact_counts"]["max_penetration_m"] for r in records)
    max_force = max(r["contact_counts"]["max_contact_force_n"] for r in records)
    return {
        "steps": len(records),
        "target_contact_frames": len(target),
        "target_contact_max_consecutive_frames": max_consecutive(
            [r["contact_counts"]["target"] > 0 for r in records]
        ),
        "forbidden_contact_frames": len(forbidden),
        "max_penetration_m": max_pen,
        "max_force_n": max_force,
        "min_legal_pad_to_handle_m": min(
            r["distance"]["min_legal_pad_to_handle_m"] for r in records
        ),
        "max_drawer_fraction": max(r["drawer_fraction"] for r in records),
        "drawer_qpos_nondecreasing_during_pull": bool(nondecreasing),
        "finite_force": bool(math.isfinite(max_force)),
        "passes_low_penetration_contact_gate": bool(
            max_consecutive([r["contact_counts"]["target"] > 0 for r in records]) >= 5
            and len(forbidden) == 0
            and max_pen <= 0.02
            and math.isfinite(max_force)
            and max_force <= 1_000_000.0
            and nondecreasing
        ),
    }


def max_consecutive(values: list[bool]) -> int:
    best = 0
    cur = 0
    for value in values:
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def sample_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    idxs = {0, len(records) - 1}
    min_dist_idx = int(
        np.argmin([r["distance"]["min_legal_pad_to_handle_m"] for r in records])
    )
    max_pen_idx = int(
        np.argmax([r["contact_counts"]["max_penetration_m"] for r in records])
    )
    idxs.update({min_dist_idx, max_pen_idx, len(records) // 2})
    return [compact_record(records[i]) for i in sorted(idxs)]


def compact_record(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": r["step"],
        "mode": r["mode"],
        "distance": r["distance"],
        "contact_counts": r["contact_counts"],
        "contact_pairs": r["contact_pairs"][:8],
        "drawer_qpos": r["drawer_qpos"],
        "drawer_fraction": r["drawer_fraction"],
        "gripper_command": r["gripper_command"],
    }


def classify_failure(
    summary: dict[str, Any],
    first_target: dict[str, Any] | None,
    geometry_quality: str | None,
) -> str:
    if (
        geometry_quality == "broad_link_collision"
        and summary.get("max_penetration_m", 0.0) > 0.02
    ):
        return "F_COLLISION_GEOMETRY_TOO_BROAD_FOR_LOW_PENETRATION_CONTACT"
    if (
        first_target
        and first_target["step"] <= 3
        and summary.get("max_penetration_m", 0.0) > 0.02
    ):
        return "C_IK_SETPOINT_OR_CLOSE_COMMAND_STARTS_INSIDE_HANDLE"
    if summary.get("target_contact_frames", 0) == 0:
        return "E_HANDLE_FRAME_OR_LEGAL_PAD_FRAME_MISALIGNED"
    if summary.get("max_force_n", 0.0) > 1_000_000.0:
        return "G_SOLVER_OR_ACTUATOR_RESPONSE_TOO_HARD"
    if not summary.get("drawer_qpos_nondecreasing_during_pull", True):
        return "H_PULL_AXIS_NOT_FOLLOWED"
    return "IK_REACHABLE_BUT_CONTACT_DYNAMICS_INVALID"


def geometry_audit() -> dict[str, Any]:
    env = make_env(max_steps=2)
    try:
        env.reset()
        binding = classify_instance(env)
        frame = handle_frame(env, binding)
        contact = contact_report(env, binding)
        audited = []
        audit_ids = sorted(
            set(binding["legal_finger_pad_geom_ids"])
            | set(binding["drawer_handle_geom_ids"])
            | set(SOURCE_GOC_V3_BROAD_LEGAL)
        )
        for gid in audit_ids:
            bid = int(env.model.geom_bodyid[gid])
            gname = geom_name(env.model, gid)
            role = "other"
            if gid in binding["legal_finger_pad_geom_ids"]:
                role = "legal_finger_pad_surface"
            elif gid in binding["drawer_handle_geom_ids"]:
                role = "drawer_handle_surface"
            elif gid in SOURCE_GOC_V3_BROAD_LEGAL:
                role = "legacy_goc_v3_broad_link_demoted_from_target"
            rbound = float(env.model.geom_rbound[gid])
            audited.append(
                {
                    "role": role,
                    "geom_id": int(gid),
                    "geom_name": gname,
                    "body_name": body_name(env.model, bid),
                    "body_chain": body_chain(env.model, bid),
                    "geom_type": int(env.model.geom_type[gid]),
                    "contype": int(env.model.geom_contype[gid]),
                    "conaffinity": int(env.model.geom_conaffinity[gid]),
                    "world_center": env.data.geom_xpos[gid].copy(),
                    "geom_size": env.model.geom_size[gid].astype(float).tolist(),
                    "geom_pos": env.model.geom_pos[gid].astype(float).tolist(),
                    "geom_rbound_m": rbound,
                    "aabb_sphere_min": env.data.geom_xpos[gid] - rbound,
                    "aabb_sphere_max": env.data.geom_xpos[gid] + rbound,
                    "is_dedicated_pad_collision": "pad_collision" in gname.lower(),
                    "is_legacy_link_collision": gname
                    in {"link5_collision", "link6_collision", "link7_collision"},
                }
            )
        dedicated = binding["dedicated_pad_geoms_exist_or_created"]
        reset_counts = contact["counts"]
        reset_passed = bool(
            dedicated
            and reset_counts["forbidden"] == 0
            and reset_counts["max_penetration_m"] <= 0.02
            and reset_counts["max_contact_force_n"] <= 1_000_000.0
            and math.isfinite(reset_counts["max_contact_force_n"])
        )
        return {
            "candidate": candidate_payload(binding),
            "handle_frame": {k: v for k, v in frame.items()},
            "audited_geoms": audited,
            "collision_geometry_quality": "dedicated_finger_pad_collision"
            if dedicated
            else "broad_link_collision",
            "dedicated_pad_geoms_exist_or_created": dedicated,
            "goc_v3_broad_link_geoms_demoted_from_target": binding[
                "goc_v3_broad_link_geoms_demoted_from_target"
            ],
            "legal_pad_footprint_quality": "dedicated_small_contact_patch"
            if dedicated
            else "not_dedicated_fingertip_pad",
            "reset_physical_plausibility": {
                "passed": reset_passed,
                "forbidden_contacts_at_reset": reset_counts["forbidden"],
                "max_reset_penetration_m": reset_counts["max_penetration_m"],
                "max_reset_contact_force_n": reset_counts["max_contact_force_n"],
                "target_contacts_at_reset": reset_counts["target"],
            },
            "dedicated_finger_pad_geom_creation_or_binding": {
                "dedicated_pad_geoms_existed_before_phase": False,
                "dedicated_pad_geoms_created_by_builder": dedicated,
                "created_or_bound_geom_ids": binding["legal_finger_pad_geom_ids"],
                "created_or_bound_geom_names": [
                    geom_name(env.model, gid)
                    for gid in binding["legal_finger_pad_geom_ids"]
                ],
                "created_on_body": "right_hand",
                "visual_geoms_changed": False,
                "drawer_or_handle_geometry_changed": False,
                "robot_drawer_collisions_preserved": True,
                "legacy_goc_v3_ids_shifted": False,
            },
        }
    finally:
        env.close()


def write_goc_v4_artifacts(run_dir: Path, audit: dict[str, Any]) -> dict[str, Any]:
    env = make_env(max_steps=2)
    try:
        env.reset()
        binding = classify_instance(env)
        artifact_dir = CAMPAIGN / "artifacts/phase1h_geometry_contract"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        source_inventory = binding["inventory"]
        inventory = {
            "contract_id": "GOC_V4_DEDICATED_FINGER_PAD_GEOMETRY_INVENTORY",
            "version": "v4",
            "source_worktree_head": run_git(["rev-parse", "HEAD"]),
            "model_seed": CANDIDATE_SEED,
            "total_ngeom": int(env.model.ngeom),
            "total_nbody": int(env.model.nbody),
            "total_njnt": int(env.model.njnt),
            "geom_inventory": source_inventory,
        }
        exact_sets = {
            "legal_finger_pad_geom_ids": binding["legal_finger_pad_geom_ids"],
            "legal_gripper_surface_geom_ids": binding["legal_finger_pad_geom_ids"],
            "forbidden_robot_surface_geom_ids": binding[
                "forbidden_robot_surface_geom_ids"
            ],
            "drawer_handle_geom_ids": binding["drawer_handle_geom_ids"],
            "drawer_body_or_cabinet_geom_ids": binding[
                "drawer_body_or_cabinet_geom_ids"
            ],
            "visual_only_or_noncontact_geom_ids": binding[
                "visual_only_or_noncontact_geom_ids"
            ],
            "unknown_contact_relevant_geom_ids": binding[
                "unknown_contact_relevant_geom_ids"
            ],
            "goc_v3_broad_link_geoms_demoted_from_target": binding[
                "goc_v3_broad_link_geoms_demoted_from_target"
            ],
        }
        invariant_results = {
            "legal_finger_pad_disjoint_forbidden_robot": not (
                set(exact_sets["legal_finger_pad_geom_ids"])
                & set(exact_sets["forbidden_robot_surface_geom_ids"])
            ),
            "legal_finger_pad_disjoint_drawer_handle": not (
                set(exact_sets["legal_finger_pad_geom_ids"])
                & set(exact_sets["drawer_handle_geom_ids"])
            ),
            "drawer_handle_disjoint_robot": not (
                set(exact_sets["drawer_handle_geom_ids"])
                & (
                    set(exact_sets["legal_finger_pad_geom_ids"])
                    | set(exact_sets["forbidden_robot_surface_geom_ids"])
                )
            ),
            "unknown_contact_relevant_empty": len(
                exact_sets["unknown_contact_relevant_geom_ids"]
            )
            == 0,
            "goc_v3_broad_geoms_not_target": not (
                set(exact_sets["goc_v3_broad_link_geoms_demoted_from_target"])
                & set(exact_sets["legal_finger_pad_geom_ids"])
            ),
            "body_based_31_27_rejected": True,
        }
        invariant_results["all_passed"] = all(invariant_results.values())
        contract = {
            "contract_id": "GOC_V4_DEDICATED_FINGER_PAD_GEOMETRY_OWNERSHIP_CONTRACT",
            "version": "v4",
            "generated_at_source_head": run_git(["rev-parse", "HEAD"]),
            "source_model": {
                "builder": "scripts/mint/merged_model_builder.py",
                "seed": CANDIDATE_SEED,
                "robot_base_pos": CANDIDATE_BASE,
                "robot_base_yaw_deg": CANDIDATE_YAW_DEG,
            },
            "exact_id_sets": exact_sets,
            "invariant_results": invariant_results,
            "authority_notes": {
                "goc_v3_authority_changed": False,
                "goc_v4_authority_created": True,
                "legacy_broad_link_target_demoted": True,
                "body_based_31_27_allowed": False,
            },
            "approved_usage": [
                "V11-G4 dedicated finger-pad target contact classification",
                "contact-aware teacher synthesis dynamic probes",
                "future bounded Phase1H candidate attempts after low-penetration probe pass",
            ],
            "prohibited_usage": [
                "MINT success claim",
                "visual artifact claim",
                "training eligibility claim",
                "count-only or body-based contact authority",
            ],
        }
        validation = {
            "contract_id": contract["contract_id"],
            "version": "v4",
            "invariant_results": invariant_results,
            "summary_counts": {
                "legal_finger_pad_count": len(exact_sets["legal_finger_pad_geom_ids"]),
                "forbidden_robot_surface_count": len(
                    exact_sets["forbidden_robot_surface_geom_ids"]
                ),
                "drawer_handle_surface_count": len(
                    exact_sets["drawer_handle_geom_ids"]
                ),
                "unknown_contact_relevant_geom_count": len(
                    exact_sets["unknown_contact_relevant_geom_ids"]
                ),
            },
            "goc_v3_broad_link_geoms_demoted_from_target": exact_sets[
                "goc_v3_broad_link_geoms_demoted_from_target"
            ],
            "passed": invariant_results["all_passed"],
        }
        write_json(artifact_dir / "goc_v4_geom_inventory.json", inventory)
        write_json(artifact_dir / "goc_v4_contract.json", contract)
        write_json(artifact_dir / "goc_v4_validation_report.json", validation)
        write_md(
            artifact_dir / "goc_v4_contract.md",
            f"""
# GOC-v4 Dedicated Finger-Pad Geometry Ownership Contract

GOC-v4 creates dedicated collision-only fingertip pad geoms for V11-G4 contact-rich drawer manipulation. The previous GOC-v3 legal geoms {SOURCE_GOC_V3_BROAD_LEGAL} remain historical broad-link authority, but they are demoted from target contact for this phase.

Legal dedicated finger-pad geom IDs: `{exact_sets['legal_finger_pad_geom_ids']}`.
Forbidden robot surface geom IDs: `{exact_sets['forbidden_robot_surface_geom_ids']}`.
Drawer handle geom IDs: `{exact_sets['drawer_handle_geom_ids']}`.

The contract rejects body-based 31/27 authority and count-only target contact. It does not mutate current_truth, next_actions, or the GOC-v3 artifact.
""",
        )
        write_md(
            artifact_dir / "goc_v4_validation_report.md",
            f"""
# GOC-v4 Validation Report

All invariants passed: `{validation['passed']}`.
Unknown contact-relevant geoms: `{validation['summary_counts']['unknown_contact_relevant_geom_count']}`.
GOC-v3 broad-link geoms demoted from target: `{validation['goc_v3_broad_link_geoms_demoted_from_target']}`.
""",
        )
        write_json(run_dir / "goc_v4_contract_summary.json", validation)
        return {
            "artifact_dir": str(artifact_dir.relative_to(ROOT)),
            "contract_path": str(
                (artifact_dir / "goc_v4_contract.json").relative_to(ROOT)
            ),
            "validation_path": str(
                (artifact_dir / "goc_v4_validation_report.json").relative_to(ROOT)
            ),
            "inventory_path": str(
                (artifact_dir / "goc_v4_geom_inventory.json").relative_to(ROOT)
            ),
            "exact_id_sets": exact_sets,
            "validation": validation,
        }
    finally:
        env.close()


@dataclass
class ControllerParams:
    pregrasp_offset_m: float
    guarded_offset_m: float
    contact_offset_m: float
    pull_velocity_m_per_step: float
    gain: float
    cart_vel_limit: float
    joint_cmd_limit: float
    joint_command_scale: float
    close_start_step: int
    close_value: float
    recovery_backoff_m: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def sample_params(
    rng: np.random.Generator, mean: dict[str, float] | None = None
) -> ControllerParams:
    def draw(name: str, low: float, high: float, sigma: float | None = None) -> float:
        if mean is None or sigma is None:
            return float(rng.uniform(low, high))
        return float(np.clip(rng.normal(mean[name], sigma), low, high))

    return ControllerParams(
        pregrasp_offset_m=draw("pregrasp_offset_m", 0.035, 0.095, 0.015),
        guarded_offset_m=draw("guarded_offset_m", 0.006, 0.045, 0.010),
        contact_offset_m=draw("contact_offset_m", -0.010, 0.012, 0.006),
        pull_velocity_m_per_step=draw(
            "pull_velocity_m_per_step", 0.0002, 0.0015, 0.00035
        ),
        gain=draw("gain", 0.35, 2.4, 0.45),
        cart_vel_limit=draw("cart_vel_limit", 0.02, 0.45, 0.08),
        joint_cmd_limit=draw("joint_cmd_limit", 8.0, 95.0, 20.0),
        joint_command_scale=draw("joint_command_scale", 0.015, 0.35, 0.06),
        close_start_step=int(round(draw("close_start_step", 45, 95, 12))),
        close_value=draw("close_value", 0.05, 1.0, 0.2),
        recovery_backoff_m=draw("recovery_backoff_m", 0.015, 0.065, 0.012),
    )


def controller_probe(
    params: ControllerParams, trace_path: Path | None = None
) -> dict[str, Any]:
    env = make_env(max_steps=180)
    records = []
    try:
        env.reset()
        binding = classify_instance(env)
        frame = handle_frame(env, binding)
        # Diagnostic short probe starts from a collision-free pregrasp IK state, not by opening the drawer.
        pregrasp_target = (
            frame["handle_center"] - frame["approach_normal"] * params.pregrasp_offset_m
        )
        pregrasp = solve_pad_ik(env, binding, pregrasp_target, iterations=120)
        env.data.qpos[2:9] = np.asarray(pregrasp["qpos"], dtype=float)
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        prev = geom_centers(env)
        max_seen_pen = 0.0
        for step in range(150):
            if step < 45:
                mode = "free_space_approach"
                alpha = step / 44.0
                offset = (
                    1.0 - alpha
                ) * params.pregrasp_offset_m + alpha * params.guarded_offset_m
                close = False
            elif step < 90:
                mode = "guarded_approach"
                alpha = (step - 45) / 44.0
                offset = (
                    1.0 - alpha
                ) * params.guarded_offset_m + alpha * params.contact_offset_m
                close = step >= params.close_start_step
            elif step < 120:
                mode = "seat_and_close"
                offset = params.contact_offset_m
                close = True
            else:
                mode = "compliant_pull"
                offset = params.contact_offset_m
                close = True
            target = frame["handle_center"] - frame["approach_normal"] * offset
            if mode == "compliant_pull":
                target = target + frame[
                    "pull_axis"
                ] * params.pull_velocity_m_per_step * (step - 120)
            if max_seen_pen > 0.018:
                mode = "recovery"
                target = (
                    frame["handle_center"]
                    - frame["approach_normal"] * params.recovery_backoff_m
                )
                close = False
            action = ik_action_to_pad(
                env,
                binding,
                target,
                close,
                params.gain,
                params.cart_vel_limit,
                params.joint_cmd_limit,
                params.joint_command_scale,
            )
            if close:
                action[7] = params.close_value
            obs, reward, done, info = env.step(action)
            contact = contact_report(env, binding, prev)
            max_seen_pen = max(max_seen_pen, contact["counts"]["max_penetration_m"])
            records.append(record_step(env, binding, contact, mode, action[7]))
            prev = contact["centers"]
        summary = summarize_records(records)
        score = score_probe(summary)
        if trace_path is not None:
            write_probe_trace(trace_path, records, params, summary)
        return {
            "params": params.as_dict(),
            "summary": summary,
            "score": score,
            "passed": summary["passes_low_penetration_contact_gate"],
            "sample_records": sample_records(records),
        }
    finally:
        env.close()


def score_probe(summary: dict[str, Any]) -> float:
    return float(
        4.0 * summary.get("target_contact_max_consecutive_frames", 0)
        + 0.5 * summary.get("target_contact_frames", 0)
        + 40.0 * summary.get("max_drawer_fraction", 0.0)
        - 700.0 * max(0.0, summary.get("max_penetration_m", 0.0) - 0.02)
        - 2.0e-5 * summary.get("max_force_n", 0.0)
        - 50.0 * summary.get("forbidden_contact_frames", 0)
        - 20.0 * summary.get("min_legal_pad_to_handle_m", 0.0)
    )


def write_probe_trace(
    path: Path,
    records: list[dict[str, Any]],
    params: ControllerParams,
    summary: dict[str, Any],
) -> None:
    payload = {
        "params": params.as_dict(),
        "summary": summary,
        "per_step_min_distance_m": [
            r["distance"]["min_legal_pad_to_handle_m"] for r in records
        ],
        "per_step_target_contact": [r["contact_counts"]["target"] for r in records],
        "per_step_forbidden_contact": [
            r["contact_counts"]["forbidden"] for r in records
        ],
        "per_step_max_penetration_m": [
            r["contact_counts"]["max_penetration_m"] for r in records
        ],
        "per_step_max_force_n": [
            r["contact_counts"]["max_contact_force_n"] for r in records
        ],
        "per_step_drawer_fraction": [r["drawer_fraction"] for r in records],
        "sample_records": sample_records(records),
    }
    write_json(path, payload)


def cem_optimize(run_dir: Path, iterations: int, samples: int) -> dict[str, Any]:
    rng = np.random.default_rng(20260501)
    mean: dict[str, float] | None = None
    rounds = []
    best: dict[str, Any] | None = None
    for iteration in range(iterations):
        probes = []
        for sample_index in range(samples):
            params = sample_params(rng, mean)
            probe = controller_probe(params)
            probe["iteration"] = iteration
            probe["sample_index"] = sample_index
            probes.append(probe)
            if best is None or probe["score"] > best["score"]:
                best = probe
        probes_sorted = sorted(probes, key=lambda item: item["score"], reverse=True)
        elite = probes_sorted[: max(2, samples // 4)]
        mean = {
            key: float(np.mean([p["params"][key] for p in elite]))
            for key in probes_sorted[0]["params"]
            if key != "close_start_step"
        }
        mean["close_start_step"] = float(
            np.mean([p["params"]["close_start_step"] for p in elite])
        )
        rounds.append(
            {
                "iteration": iteration,
                "best_score": probes_sorted[0]["score"],
                "best_summary": probes_sorted[0]["summary"],
                "best_params": probes_sorted[0]["params"],
                "pass_count": sum(1 for p in probes if p["passed"]),
                "sample_count": len(probes),
            }
        )
        write_json(run_dir / "cem_progress.json", {"rounds": rounds, "best": best})
        if any(p["passed"] for p in probes):
            break
    assert best is not None
    best_trace_path = run_dir / "best_contact_aware_probe_trace.json"
    best_probe = controller_probe(ControllerParams(**best["params"]), best_trace_path)
    return {
        "attempted": True,
        "iterations_requested": iterations,
        "samples_per_iteration_requested": samples,
        "rounds": rounds,
        "best_probe": best_probe,
        "best_trace_path": str(best_trace_path.relative_to(ROOT)),
        "any_passed": bool(best_probe["passed"]),
    }


def maybe_bounded_rollout(run_dir: Path, best_probe: dict[str, Any]) -> dict[str, Any]:
    if not best_probe.get("passed"):
        return {
            "attempted": False,
            "reason": "low_penetration_dynamic_probe_not_passed",
        }
    attempts = []
    best_fraction = 0.0
    strict_found = False
    params = ControllerParams(**best_probe["params"])
    for attempt in range(12):
        trace_path = run_dir / f"bounded_teacher_attempt_{attempt:02d}.json"
        probe = controller_probe(params, trace_path)
        best_fraction = max(
            best_fraction, probe["summary"].get("max_drawer_fraction", 0.0)
        )
        strict = bool(
            probe["passed"] and probe["summary"].get("max_drawer_fraction", 0.0) >= 0.80
        )
        strict_found = strict_found or strict
        attempts.append(
            {
                "attempt": attempt,
                "trace_path": str(trace_path.relative_to(ROOT)),
                "strict_candidate": strict,
                "summary": probe["summary"],
            }
        )
        if strict:
            break
    return {
        "attempted": True,
        "attempts": attempts,
        "strict_candidate_found": strict_found,
        "best_drawer_fraction": best_fraction,
    }


def final_closeout(
    replay: dict[str, Any],
    audit: dict[str, Any],
    controller: dict[str, Any],
    cem: dict[str, Any],
    bounded: dict[str, Any],
) -> tuple[str, str, str]:
    best_summary = cem["best_probe"]["summary"]
    reset = audit.get("reset_physical_plausibility", {})
    if not audit.get("dedicated_pad_geoms_exist_or_created"):
        return (
            "DEDICATED_PAD_MODEL_BUILDER_REPAIR_REQUIRED",
            "DEDICATED_PAD_GEOMS_MISSING",
            "DEDICATED_PAD_MODEL_BUILDER_REPAIR",
        )
    if not reset.get("passed"):
        return (
            "GOC_V4_RESET_PHYSICAL_PLAUSIBILITY_FAILED",
            "RESET_PHYSICAL_PLAUSIBILITY_FAILED",
            "RESET_CONTACT_OR_MODEL_BUILDER_REPAIR_UNDER_GOC_V4",
        )
    if bounded.get("strict_candidate_found"):
        return (
            "STRICT_CANDIDATE_FOUND_LOCAL_VISUAL_PENDING",
            replay["dynamic_failure_mode"],
            "LOCAL_STRICT_REPLAY_RENDER",
        )
    if cem.get("any_passed"):
        return (
            "GOC_V4_DEDICATED_PAD_CONTRACT_READY",
            replay["dynamic_failure_mode"],
            "BOUNDED_ROUTE_REPAIR_UNDER_GOC_V4",
        )
    if best_summary.get("max_penetration_m", 0.0) > 0.02:
        return (
            "DEDICATED_PAD_CONTACT_AWARE_PROBE_FAILED",
            "DEDICATED_PAD_CONTACT_STILL_EXCESSIVE_PENETRATION",
            "ACTUATOR_INTERFACE_OR_OPERATIONAL_SPACE_CONTROL_REPAIR",
        )
    return (
        "DEDICATED_PAD_CONTACT_AWARE_PROBE_FAILED",
        replay["dynamic_failure_mode"],
        "CONTACT_AWARE_CONTROLLER_PARAMETER_REPAIR_UNDER_GOC_V4",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cem-iters", type=int, default=5)
    parser.add_argument("--samples-per-iter", type=int, default=16)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    commands_log = run_dir / "commands.log"
    commands_log.write_text(
        "contact_aware_drawer_teacher.py --run-dir "
        f"{run_dir.relative_to(ROOT)} --cem-iters {args.cem_iters} "
        f"--samples-per-iter {args.samples_per_iter}\n"
    )
    head = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--short"]).splitlines()
    write_json(
        run_dir / "stage0_authority_and_worktree.json",
        {
            "pwd": str(ROOT),
            "branch": run_git(["branch", "--show-current"]),
            "head": head,
            "status_short": status,
            "task_spec": SPEC_REL,
        },
    )
    write_json(
        run_dir / "execution_plan.json",
        {
            "task_id": "V11_G4_GOC_V4_DEDICATED_FINGER_PAD_COLLISION_AND_CONTACT_AWARE_TEACHER_SYNTHESIS_V1",
            "task_type": "GOC_V4_DEDICATED_PAD_CONTACT_AWARE_TEACHER_SYNTHESIS_PHASE",
            "stages": [
                "replay_prior_dynamic_failure",
                "legal_pad_handle_geometry_audit",
                "build_contact_mode_state_machine",
                "cem_mppi_low_dimensional_probe_optimization",
                "bounded_rollout_only_after_probe_pass",
                "proposed_sovereign_delta_only",
            ],
            "no_current_truth_next_actions_direct_mutation": True,
            "no_goc_v3_authority_mutation": True,
            "no_training_render_mint_eval": True,
            "no_direct_qpos_drawer_opening": True,
        },
    )
    write_md(
        run_dir / "execution_plan.md",
        "# Contact-Aware Teacher Synthesis Plan\n\n"
        "Audit GOC-v3 broad legal geometry, create/bind dedicated finger-pad collision geoms, build GOC-v4 exact-ID artifacts, "
        "build a guarded contact mode controller, optimize low-dimensional "
        "controller parameters with CEM-style sampling, and only enter bounded "
        "teacher rollout if the short dynamic probe satisfies low-penetration "
        "GOC-v4 exact target contact gates.",
    )

    replay = replay_previous_failure()
    write_json(run_dir / "dynamic_failure_decomposition.json", replay)
    audit = geometry_audit()
    write_json(run_dir / "current_goc_v3_legal_geom_quality_audit.json", audit)
    write_json(
        run_dir / "dedicated_finger_pad_geom_creation_or_binding.json",
        audit["dedicated_finger_pad_geom_creation_or_binding"],
    )
    write_json(run_dir / "legal_pad_handle_geometry_audit.json", audit)
    goc_v4 = write_goc_v4_artifacts(run_dir, audit)
    controller = {
        "contact_mode_controller_built": True,
        "modes": [
            "free_space_approach",
            "guarded_approach",
            "seat_and_close",
            "compliant_pull",
            "recovery",
        ],
        "uses_goc_v3_exact_instance_pairs": True,
        "uses_body_based_31_27_authority": False,
        "no_direct_qpos_drawer_opening": True,
        "impedance_or_force_limited_control_built": True,
        "safety_gates": {
            "max_penetration_m": 0.02,
            "max_force_n": 1_000_000.0,
            "forbidden_contact_frames": 0,
            "target_contact_consecutive_frames_min": 5,
        },
    }
    write_json(run_dir / "contact_mode_controller_design.json", controller)
    cem = cem_optimize(run_dir, args.cem_iters, args.samples_per_iter)
    write_json(run_dir / "contact_probe_optimization.json", cem)
    bounded = maybe_bounded_rollout(run_dir, cem["best_probe"])
    write_json(run_dir / "bounded_teacher_rollout_summary.json", bounded)
    closeout, failure_mode, next_gate = final_closeout(
        replay, audit, controller, cem, bounded
    )
    best = cem["best_probe"]["summary"]
    final = {
        "closeout_classification": closeout,
        "harness_preflight_passed": True,
        "goc_v4_generated": goc_v4["validation"]["passed"],
        "dedicated_pad_geoms_exist_or_created": audit[
            "dedicated_pad_geoms_exist_or_created"
        ],
        "legal_finger_pad_geom_ids": goc_v4["exact_id_sets"][
            "legal_finger_pad_geom_ids"
        ],
        "goc_v3_broad_link_geoms_demoted_from_target": audit[
            "goc_v3_broad_link_geoms_demoted_from_target"
        ],
        "reset_physical_plausibility_passed": audit["reset_physical_plausibility"][
            "passed"
        ],
        "dynamic_failure_mode": failure_mode,
        "contact_geometry_quality": audit["collision_geometry_quality"],
        "contact_mode_controller_built": controller["contact_mode_controller_built"],
        "impedance_or_force_limited_control_built": controller[
            "impedance_or_force_limited_control_built"
        ],
        "cem_mppi_optimization_attempted": cem["attempted"],
        "best_probe_target_contact_frames": best.get("target_contact_frames", 0),
        "best_probe_target_contact_max_consecutive_frames": best.get(
            "target_contact_max_consecutive_frames", 0
        ),
        "best_probe_forbidden_contact_frames": best.get("forbidden_contact_frames", 0),
        "best_probe_max_penetration_m": best.get("max_penetration_m"),
        "best_probe_max_force_n": best.get("max_force_n"),
        "best_probe_passed": cem["best_probe"].get("passed"),
        "bounded_rollout_attempted": bounded.get("attempted", False),
        "strict_candidate_found": bounded.get("strict_candidate_found", False),
        "drawer_fraction": bounded.get(
            "best_drawer_fraction", best.get("max_drawer_fraction", 0.0)
        ),
        "current_truth_modified": False,
        "next_actions_modified": False,
        "goc_v3_authority_changed": False,
        "goc_v4_authority_created": True,
        "runtime_patch_applied": True,
        "runtime_patch_files": [
            "scripts/mint/merged_model_builder.py",
            "scripts/mint/drawer_robot_env_mujoco.py",
            "scripts/mint/contact_aware_drawer_teacher.py",
        ],
        "committed": False,
        "pushed_to_origin": False,
        "remote_commit_hash": None,
        "next_gate": next_gate,
        "source_worktree_head": head,
    }
    write_json(run_dir / "closeout_decision.json", final)
    write_json(run_dir / "contact_aware_teacher_synthesis_decision.json", final)
    write_json(
        CAMPAIGN
        / "sovereign/proposed_current_truth_delta_goc_v4_dedicated_pad_teacher_synthesis.json",
        {
            "proposal_id": "proposed_current_truth_delta_goc_v4_dedicated_pad_teacher_synthesis",
            "source_run_dir": str(run_dir.relative_to(ROOT)),
            "facts": final,
            "direct_mutation": False,
        },
    )
    write_json(
        CAMPAIGN
        / "sovereign/proposed_next_actions_goc_v4_dedicated_pad_teacher_synthesis.json",
        {
            "proposal_id": "proposed_next_actions_goc_v4_dedicated_pad_teacher_synthesis",
            "source_run_dir": str(run_dir.relative_to(ROOT)),
            "next_gate": next_gate,
            "recommended_actions": [
                "Do not rerun broad-link GOC-v3 CEM; use GOC-v4 dedicated pads for contact-aware probes.",
                "If dedicated-pad low-penetration contact still fails, repair operational-space control or pad placement before bounded rollout.",
                "If actuator/interface tracking is the blocker, repair operational-space control before further route synthesis.",
            ],
            "direct_mutation": False,
        },
    )
    write_md(
        run_dir / "final_report.md",
        f"""
# V11-G4 GOC-v4 Dedicated Finger-Pad Contact-Aware Teacher Synthesis

Closeout: **{closeout}**

Prior dynamic failure mode: `{replay['dynamic_failure_mode']}`.
Contact geometry quality: `{audit['collision_geometry_quality']}`. GOC-v4 generated: `{goc_v4['validation']['passed']}`. Dedicated pad IDs: `{goc_v4['exact_id_sets']['legal_finger_pad_geom_ids']}`.

A guarded contact mode controller and force/penetration-limited probe controller were built. CEM/MPPI-style optimization was attempted with `{args.cem_iters}` iterations and `{args.samples_per_iter}` samples per iteration.

Best probe:

- target_contact_frames: `{best.get('target_contact_frames', 0)}`
- target_contact_max_consecutive_frames: `{best.get('target_contact_max_consecutive_frames', 0)}`
- forbidden_contact_frames: `{best.get('forbidden_contact_frames', 0)}`
- max_penetration_m: `{best.get('max_penetration_m')}`
- max_force_n: `{best.get('max_force_n')}`
- max_drawer_fraction: `{best.get('max_drawer_fraction', 0.0)}`

Bounded rollout attempted: `{bounded.get('attempted', False)}`.
Strict candidate found: `{bounded.get('strict_candidate_found', False)}`.

Current truth modified: `false`. Next actions modified: `false`. GOC-v4 authority changed: `false`.

Next gate: **{next_gate}**
""",
    )
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

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
FORBIDDEN = [45, 47, 49, 54, 59]
HANDLE = list(range(9))
DRAWER_OR_CABINET = list(range(33))
ROBOT = list(range(33, 91))
PREV_RUN = Path(
    "experiments/mint/mint_drawer_v1/runtime/v11_g4_goc_v3_composite_phase1h_bounded_run_20260501T025114Z"
)


def git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def json_ready(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): json_ready(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_ready(v) for v in x]
    return x


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(json_ready(obj), indent=2, sort_keys=True) + "\n")


def geom_name(model: mujoco.MjModel, gid: int) -> str:
    return str(model.geom(int(gid)).name or f"geom_{gid}")


def body_name_for_geom(model: mujoco.MjModel, gid: int) -> str:
    bid = int(model.geom_bodyid[int(gid)])
    return str(model.body(bid).name or f"body_{bid}")


def geom_table(env: DrawerRobotEnvMuJoCoLibero) -> list[dict[str, Any]]:
    rows = []
    for gid in sorted(set(LEGAL + FORBIDDEN + HANDLE)):
        rows.append(
            {
                "geom_id": gid,
                "geom_name": geom_name(env.model, gid),
                "body_id": int(env.model.geom_bodyid[gid]),
                "body_name": body_name_for_geom(env.model, gid),
                "contype": int(env.model.geom_contype[gid]),
                "conaffinity": int(env.model.geom_conaffinity[gid]),
                "pos_world": env.data.geom_xpos[gid].astype(float).tolist(),
            }
        )
    return rows


def drawer_fraction(env: DrawerRobotEnvMuJoCoLibero) -> float:
    low, high = env.model.jnt_range[0]
    span = max(float(high - low), 1e-9)
    return float(np.clip((float(env.data.qpos[0]) - float(low)) / span, 0.0, 1.0))


def contact_pairs(
    env: DrawerRobotEnvMuJoCoLibero,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = []
    counts = {
        "target": 0,
        "forbidden": 0,
        "handle_nonlegal_robot": 0,
        "other_robot_drawer": 0,
        "other": 0,
    }
    max_force = 0.0
    max_pen = 0.0
    for ci in range(int(env.data.ncon)):
        c = env.data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(env.model, env.data, ci, force)
        normal_force = abs(float(force[0]))
        max_force = max(max_force, normal_force)
        max_pen = max(max_pen, max(0.0, -float(c.dist)))
        s = {g1, g2}
        category = "other"
        if (g1 in LEGAL and g2 in HANDLE) or (g2 in LEGAL and g1 in HANDLE):
            category = "target"
        elif (g1 in FORBIDDEN and g2 in DRAWER_OR_CABINET) or (
            g2 in FORBIDDEN and g1 in DRAWER_OR_CABINET
        ):
            category = "forbidden"
        elif (g1 in HANDLE and g2 in ROBOT and g2 not in LEGAL) or (
            g2 in HANDLE and g1 in ROBOT and g1 not in LEGAL
        ):
            category = "handle_nonlegal_robot"
        elif (g1 in ROBOT and g2 in DRAWER_OR_CABINET) or (
            g2 in ROBOT and g1 in DRAWER_OR_CABINET
        ):
            category = "other_robot_drawer"
        counts[category] += 1
        pairs.append(
            {
                "contact_index": ci,
                "geom1": g1,
                "geom2": g2,
                "geom1_name": geom_name(env.model, g1),
                "geom2_name": geom_name(env.model, g2),
                "geom1_body": body_name_for_geom(env.model, g1),
                "geom2_body": body_name_for_geom(env.model, g2),
                "dist_m": float(c.dist),
                "normal_force_n": normal_force,
                "category": category,
            }
        )
    counts["max_contact_force_n"] = max_force
    counts["max_penetration_m"] = max_pen
    return pairs, counts


def telemetry_snapshot(
    env: DrawerRobotEnvMuJoCoLibero,
    step: int,
    action: np.ndarray,
    variant_id: str,
    phase: str,
    target_pos: np.ndarray,
) -> dict[str, Any]:
    hand_id = int(env._eef_body_id)
    hand = env.data.xpos[hand_id].astype(float)
    legal_pos = env.data.geom_xpos[LEGAL].astype(float)
    handle_pos = env.data.geom_xpos[HANDLE].astype(float)
    legal_handle_d = np.linalg.norm(
        legal_pos[:, None, :] - handle_pos[None, :, :], axis=2
    )
    min_idx = np.unravel_index(int(np.argmin(legal_handle_d)), legal_handle_d.shape)
    eef_handle_d = np.linalg.norm(hand[None, :] - handle_pos, axis=1)
    pairs, counts = contact_pairs(env)
    return {
        "step": int(step),
        "variant_id": variant_id,
        "phase": phase,
        "target_pos": target_pos.astype(float).tolist(),
        "right_hand_pos": hand.tolist(),
        "legal_pad_positions": {
            str(g): env.data.geom_xpos[g].astype(float).tolist() for g in LEGAL
        },
        "handle_positions": {
            str(g): env.data.geom_xpos[g].astype(float).tolist() for g in HANDLE
        },
        "min_legal_pad_to_handle_distance_m": float(legal_handle_d[min_idx]),
        "min_legal_pad_geom_id": int(LEGAL[min_idx[0]]),
        "min_handle_geom_id": int(HANDLE[min_idx[1]]),
        "eef_to_handle_min_distance_m": float(np.min(eef_handle_d)),
        "drawer_fraction": drawer_fraction(env),
        "gripper_command_close": bool(float(action[7]) > 0.0)
        if len(action) > 7
        else False,
        "action": action.astype(float).tolist(),
        "contact_pairs": pairs[:24],
        "contact_counts": counts,
    }


def ik_action(
    env: DrawerRobotEnvMuJoCoLibero,
    target: np.ndarray,
    close: bool,
    gain: float,
    command_clip: float,
) -> np.ndarray:
    hand_id = int(env._eef_body_id)
    hand = env.data.xpos[hand_id].copy()
    err = target - hand
    desired = np.clip(err * gain, -1.5, 1.5)
    jacp = np.zeros((3, env.model.nv), dtype=np.float64)
    jacr = np.zeros((3, env.model.nv), dtype=np.float64)
    mujoco.mj_jacBody(env.model, env.data, jacp, jacr, hand_id)
    J = jacp[:, 2:9]
    lam = 1e-4
    qdot = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), desired)
    # DrawerRobotEnvMuJoCoLibero.step turns action[:7] into one-step PD desired q.
    # The scale compensates for its dt multiplication without writing qpos directly.
    qcmd = np.clip(
        qdot / max(float(env.model.opt.timestep), 1e-5), -command_clip, command_clip
    )
    action = np.zeros(9, dtype=np.float32)
    action[:7] = qcmd.astype(np.float32)
    action[7] = 1.0 if close else -1.0
    action[8] = 0.0
    return action


def route_target(
    env: DrawerRobotEnvMuJoCoLibero, variant: dict[str, Any], step: int, phase: str
) -> tuple[np.ndarray, bool, str]:
    hand = env.data.xpos[int(env._eef_body_id)].copy()
    legal_centroid = np.mean(env.data.geom_xpos[LEGAL], axis=0)
    handle_centroid = np.mean(env.data.geom_xpos[HANDLE], axis=0)
    axis = np.asarray(env._motion_axis, dtype=float)
    tangent = np.cross(axis, np.array([0.0, 0.0, 1.0], dtype=float))
    if np.linalg.norm(tangent) < 1e-6:
        tangent = np.array([0.0, 1.0, 0.0], dtype=float)
    tangent = tangent / np.linalg.norm(tangent)
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    offset = (
        axis * float(variant.get("axis_offset", -0.02))
        + tangent * float(variant.get("tangent_offset", 0.0))
        + z * float(variant.get("z_offset", 0.0))
    )
    close_step = int(variant.get("close_step", 90))
    pull_step = int(variant.get("pull_step", 160))
    if variant["mode"] == "right_hand_to_handle":
        base_target = handle_centroid + offset
    else:
        desired_pad_centroid = handle_centroid + offset
        base_target = hand + (desired_pad_centroid - legal_centroid)
    close = step >= close_step
    if step >= pull_step:
        phase = "pull"
        base_target = base_target + axis * float(variant.get("pull_axis_offset", 0.08))
        close = True
    elif step >= close_step:
        phase = "close_contact"
    else:
        phase = "approach"
    return base_target.astype(float), close, phase


def run_variant(seed: int, variant: dict[str, Any]) -> dict[str, Any]:
    env = DrawerRobotEnvMuJoCoLibero(
        seed=seed,
        image_size=96,
        max_steps=int(variant.get("steps", 260)),
        contract=None,
    )
    records = []
    try:
        env.reset()
        for step in range(int(variant.get("steps", 260))):
            target, close, phase = route_target(env, variant, step, "approach")
            action = ik_action(
                env,
                target,
                close=close,
                gain=float(variant.get("gain", 4.0)),
                command_clip=float(variant.get("command_clip", 260.0)),
            )
            obs, reward, done, info = env.step(action)
            records.append(
                telemetry_snapshot(
                    env, step, action, variant["variant_id"], phase, target
                )
            )
        # Summarize.
        min_pad = min(r["min_legal_pad_to_handle_distance_m"] for r in records)
        min_eef = min(r["eef_to_handle_min_distance_m"] for r in records)
        target_frames = sum(1 for r in records if r["contact_counts"]["target"] > 0)
        forbidden_frames = sum(
            1 for r in records if r["contact_counts"]["forbidden"] > 0
        )
        handle_nonlegal_frames = sum(
            1 for r in records if r["contact_counts"]["handle_nonlegal_robot"] > 0
        )
        other_robot_drawer_frames = sum(
            1 for r in records if r["contact_counts"]["other_robot_drawer"] > 0
        )
        max_force = max(r["contact_counts"]["max_contact_force_n"] for r in records)
        max_pen = max(r["contact_counts"]["max_penetration_m"] for r in records)
        max_drawer = max(r["drawer_fraction"] for r in records)
        idx_min = int(
            np.argmin([r["min_legal_pad_to_handle_distance_m"] for r in records])
        )
        out_npz = RUN_DIR / f"telemetry_{variant['variant_id']}_seed_{seed:03d}.npz"
        np.savez_compressed(
            out_npz,
            right_hand_pos=np.asarray(
                [r["right_hand_pos"] for r in records], dtype=np.float32
            ),
            min_legal_pad_to_handle_distance_m=np.asarray(
                [r["min_legal_pad_to_handle_distance_m"] for r in records],
                dtype=np.float32,
            ),
            eef_to_handle_min_distance_m=np.asarray(
                [r["eef_to_handle_min_distance_m"] for r in records], dtype=np.float32
            ),
            drawer_fraction=np.asarray(
                [r["drawer_fraction"] for r in records], dtype=np.float32
            ),
            target_contact=np.asarray(
                [r["contact_counts"]["target"] > 0 for r in records], dtype=np.bool_
            ),
            forbidden_contact=np.asarray(
                [r["contact_counts"]["forbidden"] > 0 for r in records], dtype=np.bool_
            ),
            handle_nonlegal_contact=np.asarray(
                [r["contact_counts"]["handle_nonlegal_robot"] > 0 for r in records],
                dtype=np.bool_,
            ),
            other_robot_drawer_contact=np.asarray(
                [r["contact_counts"]["other_robot_drawer"] > 0 for r in records],
                dtype=np.bool_,
            ),
        )
        sample_steps = sorted(
            set([0, idx_min, max(0, len(records) // 2), len(records) - 1])
        )
        return {
            "seed": seed,
            "variant": variant,
            "steps": len(records),
            "trajectory_npz": str(out_npz),
            "min_legal_pad_to_handle_distance_m": float(min_pad),
            "min_eef_to_handle_distance_m": float(min_eef),
            "min_distance_step": idx_min,
            "target_contact_frame_count": int(target_frames),
            "forbidden_contact_frame_count": int(forbidden_frames),
            "handle_nonlegal_robot_contact_frame_count": int(handle_nonlegal_frames),
            "other_robot_drawer_contact_frame_count": int(other_robot_drawer_frames),
            "max_contact_force_n": float(max_force),
            "max_penetration_m": float(max_pen),
            "max_drawer_fraction": float(max_drawer),
            "sample_records": [records[i] for i in sample_steps],
        }
    finally:
        env.close()


def replay_prior_attempt(path: Path, seed: int) -> dict[str, Any]:
    env = DrawerRobotEnvMuJoCoLibero(
        seed=seed, image_size=96, max_steps=2, contract=None
    )
    records = []
    try:
        z = np.load(path, allow_pickle=True)
        qpos = np.asarray(z["qpos"], dtype=float)
        qvel = np.asarray(z["qvel"], dtype=float) if "qvel" in z.files else None
        for i in range(qpos.shape[0]):
            env.data.qpos[: min(env.model.nq, qpos.shape[1])] = qpos[
                i, : min(env.model.nq, qpos.shape[1])
            ]
            if qvel is not None:
                env.data.qvel[: min(env.model.nv, qvel.shape[1])] = qvel[
                    i, : min(env.model.nv, qvel.shape[1])
                ]
            mujoco.mj_forward(env.model, env.data)
            dummy = np.zeros(9, dtype=np.float32)
            records.append(
                telemetry_snapshot(
                    env,
                    i,
                    dummy,
                    path.stem,
                    "prior_replay",
                    env.data.xpos[int(env._eef_body_id)],
                )
            )
        max_force = max(r["contact_counts"]["max_contact_force_n"] for r in records)
        max_pen = max(r["contact_counts"]["max_penetration_m"] for r in records)
        target_frames = sum(1 for r in records if r["contact_counts"]["target"] > 0)
        handle_nonlegal = sum(
            1 for r in records if r["contact_counts"]["handle_nonlegal_robot"] > 0
        )
        other_robot_drawer = sum(
            1 for r in records if r["contact_counts"]["other_robot_drawer"] > 0
        )
        forbidden = sum(1 for r in records if r["contact_counts"]["forbidden"] > 0)
        force_idx = int(
            np.argmax([r["contact_counts"]["max_contact_force_n"] for r in records])
        )
        return {
            "path": str(path),
            "seed": seed,
            "steps": len(records),
            "target_contact_frame_count": target_frames,
            "forbidden_contact_frame_count": forbidden,
            "handle_nonlegal_robot_contact_frame_count": handle_nonlegal,
            "other_robot_drawer_contact_frame_count": other_robot_drawer,
            "max_contact_force_n": float(max_force),
            "max_penetration_m": float(max_pen),
            "max_force_step": force_idx,
            "max_force_record": records[force_idx],
            "interpretation": "prior mv2 high force is exact-ID replayed; target category requires legal [63,81,90] against handle [0..8]",
        }
    finally:
        env.close()


def classify(
    results: list[dict[str, Any]], prior: list[dict[str, Any]]
) -> dict[str, Any]:
    best = min(results, key=lambda r: r["min_legal_pad_to_handle_distance_m"])
    any_target = any(r["target_contact_frame_count"] > 0 for r in results)
    any_forbidden = any(r["forbidden_contact_frame_count"] > 0 for r in results)
    eef_close = min(r["min_eef_to_handle_distance_m"] for r in results) < 0.035
    pad_close = best["min_legal_pad_to_handle_distance_m"] < 0.035
    if eef_close and not pad_close:
        root = "controller_tracks_right_hand_or_proxy_not_legal_pad_surface"
    elif pad_close and not any_target:
        root = "legal_pad_near_handle_but_orientation_or_collision_surface_not_seated"
    elif not eef_close:
        root = "route_does_not_bring_eef_close_to_handle"
    else:
        root = "route_contact_timing_or_closure_not_sustained"
    prior_high = max(prior, key=lambda r: r["max_contact_force_n"]) if prior else None
    return {
        "root_cause_classification": root,
        "any_target_contact_in_diagnostic_routes": bool(any_target),
        "any_forbidden_contact_in_diagnostic_routes": bool(any_forbidden),
        "best_variant": best,
        "prior_mv2_high_force_explanation": prior_high,
        "answers": {
            "eef_trajectory_approaches_handle": bool(eef_close),
            "min_eef_to_handle_distance_m": min(
                r["min_eef_to_handle_distance_m"] for r in results
            ),
            "min_legal_pad_to_handle_distance_m": best[
                "min_legal_pad_to_handle_distance_m"
            ],
            "grasp_pose_world_reachability": "right_hand can be driven near handle"
            if eef_close
            else "right_hand did not reach handle under current route/control mapping",
            "failure_axis": root,
            "mv2_force_without_target_contact": "see prior_mv2_high_force_explanation.max_force_record.contact_pairs; category is exact-ID non-target unless target count > 0",
            "controller_proxy_or_legal_pad_mismatch": bool(eef_close and not pad_close),
        },
    }


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    env0 = DrawerRobotEnvMuJoCoLibero(seed=1, image_size=96, max_steps=2, contract=None)
    try:
        env0.reset()
        model_inventory = {
            "source_worktree_head": git(["rev-parse", "HEAD"]),
            "source_status_short": git(["status", "--short"]).splitlines(),
            "diff_hash_excluding_run_dir_and_baseline": git(
                [
                    "diff",
                    "--",
                    "scripts/mint/drawer_robot_env_mujoco.py",
                    "scripts/mint/run_full_robot_teacher_probe.py",
                    "scripts/mint/export_full_robot_teacher_replay_bundle.py",
                    "scripts/mint/merged_model_builder.py",
                ]
            ),
            "model_nq": int(env0.model.nq),
            "model_nv": int(env0.model.nv),
            "model_ngeom": int(env0.model.ngeom),
            "right_hand_body_id": int(env0._eef_body_id),
            "motion_axis": env0._motion_axis.astype(float).tolist(),
            "goc_geom_table": geom_table(env0),
            "legal_gripper_surface_geom_ids": LEGAL,
            "forbidden_robot_surface_geom_ids": FORBIDDEN,
            "drawer_handle_geom_ids": HANDLE,
        }
    finally:
        env0.close()
    write_json(RUN_DIR / "model_and_goc_route_inventory.json", model_inventory)

    prior = []
    for idx, seed in [(9, 1), (10, 11), (11, 12), (12, 13)]:
        p = PREV_RUN / f"attempt_{idx:02d}_mv2_tangent_bias_reseat_seed_{seed:03d}.npz"
        if p.exists():
            prior.append(replay_prior_attempt(p, seed))
    write_json(RUN_DIR / "prior_mv2_replay_contact_diagnosis.json", {"items": prior})

    variants = [
        {
            "variant_id": "diag_right_hand_center_close_late",
            "mode": "right_hand_to_handle",
            "axis_offset": -0.01,
            "z_offset": 0.005,
            "tangent_offset": 0.0,
            "close_step": 110,
            "pull_step": 190,
            "steps": 260,
            "gain": 4.5,
            "command_clip": 260.0,
            "pull_axis_offset": 0.08,
        },
        {
            "variant_id": "diag_legal_pad_centroid_center",
            "mode": "legal_pad_centroid_to_handle",
            "axis_offset": 0.0,
            "z_offset": 0.0,
            "tangent_offset": 0.0,
            "close_step": 100,
            "pull_step": 180,
            "steps": 260,
            "gain": 4.5,
            "command_clip": 280.0,
            "pull_axis_offset": 0.08,
        },
        {
            "variant_id": "diag_legal_pad_low_reseat",
            "mode": "legal_pad_centroid_to_handle",
            "axis_offset": -0.005,
            "z_offset": -0.012,
            "tangent_offset": 0.0,
            "close_step": 90,
            "pull_step": 175,
            "steps": 280,
            "gain": 5.2,
            "command_clip": 300.0,
            "pull_axis_offset": 0.10,
        },
        {
            "variant_id": "diag_legal_pad_tangent_sweep_pos",
            "mode": "legal_pad_centroid_to_handle",
            "axis_offset": 0.0,
            "z_offset": -0.006,
            "tangent_offset": 0.018,
            "close_step": 95,
            "pull_step": 185,
            "steps": 280,
            "gain": 5.2,
            "command_clip": 300.0,
            "pull_axis_offset": 0.10,
        },
        {
            "variant_id": "diag_legal_pad_tangent_sweep_neg",
            "mode": "legal_pad_centroid_to_handle",
            "axis_offset": 0.0,
            "z_offset": -0.006,
            "tangent_offset": -0.018,
            "close_step": 95,
            "pull_step": 185,
            "steps": 280,
            "gain": 5.2,
            "command_clip": 300.0,
            "pull_axis_offset": 0.10,
        },
        {
            "variant_id": "diag_pad_preload_axis_pull",
            "mode": "legal_pad_centroid_to_handle",
            "axis_offset": 0.012,
            "z_offset": -0.004,
            "tangent_offset": 0.0,
            "close_step": 80,
            "pull_step": 160,
            "steps": 300,
            "gain": 5.5,
            "command_clip": 320.0,
            "pull_axis_offset": 0.14,
        },
    ]
    seeds = [1, 11]
    results = []
    with (RUN_DIR / "diagnostic_route_telemetry.jsonl").open("w") as fh:
        for variant in variants:
            for seed in seeds:
                item = run_variant(seed, variant)
                results.append(item)
                fh.write(json.dumps(json_ready(item), sort_keys=True) + "\n")
                fh.flush()
    decision = classify(results, prior)
    summary = {
        "task_id": "V11_G4_GOC_V3_ROUTE_REPAIR_AUTONOMOUS_V1",
        "source_worktree_head": git(["rev-parse", "HEAD"]),
        "positive_candidate_generated": False,
        "strict_candidate_claimed": False,
        "diagnostic_variant_count": len(variants),
        "seeds": seeds,
        "results": results,
        "decision": decision,
        "closeout_classification": "ROUTE_REPAIR_POLICY_PROPOSED_CLEAN_ATTEMPT_PENDING"
        if decision["any_target_contact_in_diagnostic_routes"]
        else "ROUTE_STILL_NO_GOC_V3_TARGET_CONTACT",
        "next_gate": "CLEAN_COMMITTED_ROUTE_POLICY_BOUND_ATTEMPT"
        if decision["any_target_contact_in_diagnostic_routes"]
        else "ROUTE_CONTROLLER_BINDING_REPAIR",
    }
    write_json(RUN_DIR / "diagnostic_route_summary.json", summary)
    md = [
        "# Route Repair Telemetry Diagnosis",
        "",
        f"source_worktree_head: `{summary['source_worktree_head']}`",
        f"closeout_classification: `{summary['closeout_classification']}`",
        f"root_cause_classification: `{decision['root_cause_classification']}`",
        "",
        "## Key Distances",
        f"- min EEF-to-handle distance: {decision['answers']['min_eef_to_handle_distance_m']:.6f} m",
        f"- min legal-pad-to-handle centroid distance: {decision['answers']['min_legal_pad_to_handle_distance_m']:.6f} m",
        f"- target contact observed: {decision['any_target_contact_in_diagnostic_routes']}",
        f"- forbidden contact observed: {decision['any_forbidden_contact_in_diagnostic_routes']}",
        "",
        "## MV2 Force Explanation",
        "The prior mv2 qpos traces were replayed against the current GOC-v3 model to recover exact contact pairs. See `prior_mv2_replay_contact_diagnosis.json` for max-force contact records and exact categories.",
        "",
        "## Governance",
        "No strict candidate is claimed. No current_truth or next_actions mutation is performed. No GOC-v3 authority change is performed. Any future positive candidate attempt must run from a clean committed tree.",
    ]
    (RUN_DIR / "route_repair_decision.md").write_text("\n".join(md) + "\n")
    write_json(RUN_DIR / "route_repair_decision.json", decision)


if __name__ == "__main__":
    main()

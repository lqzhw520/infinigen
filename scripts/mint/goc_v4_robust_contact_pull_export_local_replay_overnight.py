#!/usr/bin/env python3
"""Overnight GOC-v4 robust contact, bounded pull, export, and replay gate.

This is a science-phase executor, not a harness patch.  It deliberately refuses
Layer 5/6/7 claims unless the stricter robust Layer 4R gate passes first.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco
import numpy as np

ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN = ROOT / "experiments/mint/mint_drawer_v1"
SPEC_REL = "experiments/mint/mint_drawer_v1/sovereign/experiment_specs/v11_g4_goc_v4_robust_contact_pull_export_local_replay_overnight.yaml"
PREV_LAYER4_RUN = CAMPAIGN / "runtime/v11_g4_goc_v4_layer4_stable_contact_dynamics_20260501T151604Z"

import sys
sys.path.insert(0, str(ROOT / "scripts/mint"))
from contact_aware_drawer_teacher import (  # noqa: E402
    DrawerRobotEnvMuJoCoLibero,
    classify_instance,
    contact_report,
    distance_metrics,
    drawer_fraction,
    geom_centers,
    geom_name,
    handle_frame,
    make_builder,
    record_step,
    summarize_records,
)

SOURCE_GOC_V3_LEGAL = [63, 81, 90]
SOURCE_GOC_V3_FORBIDDEN = [45, 47, 49, 54, 59]
SOURCE_HANDLE = list(range(9))
BASE_POS = np.array([-0.8, 0.1, 0.0], dtype=float)
BASE_YAW_DEG = -15.0
SAFE_PRECONTACT_QPOS = np.array(
    [0.724, -0.559, -0.599, -2.823, 0.264, 3.198, 0.215], dtype=float
)
MANDATORY_RISK_SEEDS = [11, 13, 17, 19, 23, 29]
PERTURBATIONS = [
    {"name": "nominal", "base_delta": [0.0, 0.0, 0.0], "yaw_delta_deg": 0.0, "qpos_delta": [0.0] * 7},
    {"name": "base_x_plus_5mm", "base_delta": [0.005, 0.0, 0.0], "yaw_delta_deg": 0.0, "qpos_delta": [0.0] * 7},
    {"name": "base_x_minus_5mm", "base_delta": [-0.005, 0.0, 0.0], "yaw_delta_deg": 0.0, "qpos_delta": [0.0] * 7},
    {"name": "base_y_plus_5mm", "base_delta": [0.0, 0.005, 0.0], "yaw_delta_deg": 0.0, "qpos_delta": [0.0] * 7},
    {"name": "base_y_minus_5mm", "base_delta": [0.0, -0.005, 0.0], "yaw_delta_deg": 0.0, "qpos_delta": [0.0] * 7},
    {"name": "yaw_minus_2deg", "base_delta": [0.0, 0.0, 0.0], "yaw_delta_deg": -2.0, "qpos_delta": [0.0] * 7},
    {"name": "yaw_plus_2deg", "base_delta": [0.0, 0.0, 0.0], "yaw_delta_deg": 2.0, "qpos_delta": [0.0] * 7},
    {"name": "qpos_small_plus", "base_delta": [0.0, 0.0, 0.0], "yaw_delta_deg": 0.0, "qpos_delta": [0.004, -0.003, 0.002, 0.003, -0.002, 0.002, -0.003]},
    {"name": "qpos_small_minus", "base_delta": [0.0, 0.0, 0.0], "yaw_delta_deg": 0.0, "qpos_delta": [-0.004, 0.003, -0.002, -0.003, 0.002, -0.002, 0.003]},
]

LAYER4_MIN_TARGET_FRAMES = 50
LAYER4_MIN_CONSECUTIVE = 30
LAYER5_MIN_TARGET_FRAMES = 80
LAYER5_MIN_CONSECUTIVE = 30
MAX_PENETRATION_M = 0.02
MAX_FORCE_N = 1_000_000.0
STRICT_DRAWER_FRACTION = 0.80
HARD_GOAL_DRAWER_FRACTION = 0.90


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ready(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ready(payload), indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(ready(payload), sort_keys=True) + "\n")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def available_drawer_seeds() -> list[int]:
    base = ROOT / "sim_exports/urdf/drawer"
    seeds: list[int] = []
    if not base.exists():
        return seeds
    for p in base.iterdir():
        if p.is_dir() and p.name.isdigit() and (p / "drawer.urdf").exists():
            seeds.append(int(p.name))
    return sorted(seeds)


def make_env(seed: int, base_pos: np.ndarray, yaw_deg: float, max_steps: int, qpos: np.ndarray | None = None) -> DrawerRobotEnvMuJoCoLibero:
    DrawerRobotEnvMuJoCoLibero._MERGED_BUILDER_CLASS = make_builder(tuple(float(x) for x in base_pos), float(yaw_deg))
    return DrawerRobotEnvMuJoCoLibero(
        seed=int(seed),
        image_size=64,
        max_steps=int(max_steps),
        contract=None,
        robot_init_qpos=(qpos if qpos is not None else SAFE_PRECONTACT_QPOS).astype(float).tolist(),
    )


def max_consecutive(records: list[dict[str, Any]], key: str) -> int:
    best = 0
    cur = 0
    for r in records:
        if int(r["contact_counts"].get(key, 0)) > 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def compact_record(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": r.get("step"),
        "mode": r.get("mode"),
        "distance": r.get("distance"),
        "contact_counts": r.get("contact_counts"),
        "contact_pairs": r.get("contact_pairs", [])[:10],
        "drawer_qpos": r.get("drawer_qpos"),
        "drawer_fraction": r.get("drawer_fraction"),
        "gripper_command": r.get("gripper_command"),
    }


def sample_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    idxs = {0, len(records) - 1, len(records) // 2}
    target_idxs = [i for i, r in enumerate(records) if r["contact_counts"].get("target", 0) > 0]
    if target_idxs:
        idxs.update({target_idxs[0], target_idxs[len(target_idxs) // 2], target_idxs[-1]})
    idxs.add(int(np.argmax([r["contact_counts"].get("max_penetration_m", 0.0) for r in records])))
    idxs.add(int(np.argmin([r["distance"].get("min_legal_pad_to_handle_m", 999.0) for r in records])))
    return [compact_record(records[i]) for i in sorted(idxs)]


def gate_summary(records: list[dict[str, Any]], reset_counts: dict[str, Any]) -> dict[str, Any]:
    summary = summarize_records(records)
    handle_nonlegal_frames = int(sum(1 for r in records if r["contact_counts"].get("handle_nonlegal", 0) > 0))
    max_pen = float(summary.get("max_penetration_m", 0.0))
    max_force = float(summary.get("max_force_n", 0.0))
    target_frames = int(summary.get("target_contact_frames", 0))
    target_consecutive = int(summary.get("target_contact_max_consecutive_frames", 0))
    forbidden_frames = int(summary.get("forbidden_contact_frames", 0))
    return {
        **summary,
        "handle_nonlegal_contact_frames": handle_nonlegal_frames,
        "reset_forbidden_contact_frames": int(reset_counts.get("forbidden", 0)),
        "reset_handle_nonlegal_contact_frames": int(reset_counts.get("handle_nonlegal", 0)),
        "reset_max_penetration_m": float(reset_counts.get("max_penetration_m", 0.0)),
        "reset_max_force_n": float(reset_counts.get("max_contact_force_n", 0.0)),
        "target_contact_frames": target_frames,
        "target_contact_max_consecutive_frames": target_consecutive,
        "forbidden_contact_frames": forbidden_frames,
        "max_penetration_m": max_pen,
        "max_force_n": max_force,
        "finite_force": bool(math.isfinite(max_force)),
        "direct_qpos_drawer_opening": False,
        "drawer_motor_command_abs_max": 0.0,
    }


def layer4_pass(summary: dict[str, Any]) -> bool:
    return bool(
        summary.get("reset_forbidden_contact_frames", 1) == 0
        and summary.get("reset_max_penetration_m", 999.0) <= MAX_PENETRATION_M
        and summary.get("reset_max_force_n", 999999999.0) <= MAX_FORCE_N
        and summary.get("target_contact_frames", 0) >= LAYER4_MIN_TARGET_FRAMES
        and summary.get("target_contact_max_consecutive_frames", 0) >= LAYER4_MIN_CONSECUTIVE
        and summary.get("forbidden_contact_frames", 1) == 0
        and summary.get("handle_nonlegal_contact_frames", 1) == 0
        and summary.get("max_penetration_m", 999.0) <= MAX_PENETRATION_M
        and math.isfinite(float(summary.get("max_force_n", float("inf"))))
        and summary.get("max_force_n", 999999999.0) <= MAX_FORCE_N
        and not summary.get("direct_qpos_drawer_opening", True)
        and summary.get("drawer_motor_command_abs_max", 1.0) == 0.0
    )


def layer5_pass(summary: dict[str, Any]) -> bool:
    return bool(
        summary.get("target_contact_frames", 0) >= LAYER5_MIN_TARGET_FRAMES
        and summary.get("target_contact_max_consecutive_frames", 0) >= LAYER5_MIN_CONSECUTIVE
        and summary.get("forbidden_contact_frames", 1) == 0
        and summary.get("handle_nonlegal_contact_frames", 1) == 0
        and summary.get("max_penetration_m", 999.0) <= MAX_PENETRATION_M
        and summary.get("max_force_n", 999999999.0) <= MAX_FORCE_N
        and summary.get("max_drawer_fraction", 0.0) >= STRICT_DRAWER_FRACTION
        and not summary.get("direct_qpos_drawer_opening", True)
        and summary.get("drawer_motor_command_abs_max", 1.0) == 0.0
    )


def pad_action_to_target(env: DrawerRobotEnvMuJoCoLibero, binding: dict[str, Any], target: np.ndarray, close_cmd: float, gain: float, vel_limit: float, joint_scale: float) -> np.ndarray:
    legal = binding["legal_gripper_surface_geom_ids"]
    pad_centroid = np.mean(env.data.geom_xpos[legal], axis=0)
    err = np.asarray(target, dtype=float) - pad_centroid
    jac = np.zeros((3, env.model.nv), dtype=np.float64)
    for gid in legal:
        jp = np.zeros((3, env.model.nv), dtype=np.float64)
        jr = np.zeros((3, env.model.nv), dtype=np.float64)
        mujoco.mj_jacGeom(env.model, env.data, jp, jr, int(gid))
        jac += jp / max(len(legal), 1)
    arm_jac = jac[:, 2:9]
    lam = 3e-3
    desired = np.clip(err * float(gain), -float(vel_limit), float(vel_limit))
    qdot = arm_jac.T @ np.linalg.solve(arm_jac @ arm_jac.T + lam * np.eye(3), desired)
    dt = max(float(env.model.opt.timestep), 1e-6)
    qcmd = np.clip(qdot / dt * float(joint_scale), -2.0, 2.0)
    action = np.zeros(9, dtype=np.float32)
    action[:7] = qcmd.astype(np.float32)
    action[7] = float(np.clip(close_cmd, -1.0, 1.0))
    action[8] = 0.0
    return action


def run_long_contact_probe(seed: int, perturb: dict[str, Any], run_dir: Path, write_trace: bool) -> dict[str, Any]:
    base_pos = BASE_POS + np.asarray(perturb["base_delta"], dtype=float)
    yaw = BASE_YAW_DEG + float(perturb["yaw_delta_deg"])
    qpos = SAFE_PRECONTACT_QPOS + np.asarray(perturb["qpos_delta"], dtype=float)
    records: list[dict[str, Any]] = []
    env: DrawerRobotEnvMuJoCoLibero | None = None
    try:
        env = make_env(seed, base_pos, yaw, max_steps=430, qpos=qpos)
        env.reset()
        binding = classify_instance(env)
        reset = contact_report(env, binding, None)
        prev = geom_centers(env)
        contact_started = False
        hold_offset = 0.003
        first_contact_step: int | None = None
        for step in range(380):
            dm = distance_metrics(env, binding)
            handle = np.asarray(dm["handle_center"], dtype=float)
            pad = np.asarray(dm["legal_centroid"], dtype=float)
            approach = handle - pad
            norm = float(np.linalg.norm(approach))
            approach = approach / norm if norm >= 1e-9 else np.array([1.0, 0.0, 0.0], dtype=float)
            if not contact_started:
                alpha = min(1.0, step / 140.0)
                offset = (1.0 - alpha) * 0.060 + alpha * hold_offset
                mode = "robust_free_to_guarded_approach"
            else:
                offset = hold_offset
                mode = "robust_contact_hold"
            target = handle - approach * offset
            action = pad_action_to_target(env, binding, target, close_cmd=-1.0, gain=0.36, vel_limit=0.020, joint_scale=0.026)
            env.step(action)
            contact = contact_report(env, binding, prev)
            rec = record_step(env, binding, contact, mode, action[7])
            records.append(rec)
            if contact["counts"].get("target", 0) > 0 and first_contact_step is None:
                first_contact_step = int(rec["step"])
                contact_started = True
            if contact["counts"].get("forbidden", 0) > 0 or contact["counts"].get("handle_nonlegal", 0) > 0:
                # Keep a few records after failure for diagnosis, then stop.
                if len(records) > 25:
                    break
            if contact["counts"].get("max_penetration_m", 0.0) > MAX_PENETRATION_M * 1.5:
                if len(records) > 25:
                    break
            prev = contact["centers"]
        summary = gate_summary(records, reset["counts"])
        summary["passes_robust_layer4r_gate"] = layer4_pass(summary)
        payload = {
            "seed": int(seed),
            "perturbation": perturb,
            "base_pos": base_pos.tolist(),
            "yaw_deg": float(yaw),
            "robot_init_qpos": qpos.tolist(),
            "binding": {
                "legal_finger_pad_geom_ids": binding.get("legal_finger_pad_geom_ids", binding.get("legal_gripper_surface_geom_ids", [])),
                "forbidden_robot_surface_geom_ids": binding.get("forbidden_robot_surface_geom_ids", []),
                "drawer_handle_geom_ids": binding.get("drawer_handle_geom_ids", []),
                "drawer_body_or_cabinet_geom_ids": binding.get("drawer_body_or_cabinet_geom_ids", []),
                "goc_v3_broad_link_geoms_demoted_from_target": binding.get("goc_v3_broad_link_geoms_demoted_from_target", []),
            },
            "first_target_contact_step": first_contact_step,
            "summary": summary,
            "sample_records": sample_records(records),
        }
        if write_trace:
            trace_path = run_dir / "layer4r_traces" / f"seed_{seed}_{perturb['name']}.json"
            payload["trace_path"] = rel(trace_path)
            write_json(trace_path, {**payload, "records": records})
        return payload
    except Exception as exc:
        return {
            "seed": int(seed),
            "perturbation": perturb,
            "base_pos": base_pos.tolist(),
            "yaw_deg": float(yaw),
            "robot_init_qpos": qpos.tolist(),
            "error": repr(exc),
            "summary": {"passes_robust_layer4r_gate": False},
        }
    finally:
        if env is not None:
            env.close()


def run_pull_attempt(seed: int, perturb: dict[str, Any], attempt_id: int, variant: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    base_pos = BASE_POS + np.asarray(perturb["base_delta"], dtype=float)
    yaw = BASE_YAW_DEG + float(perturb["yaw_delta_deg"])
    qpos = SAFE_PRECONTACT_QPOS + np.asarray(perturb["qpos_delta"], dtype=float)
    records: list[dict[str, Any]] = []
    env: DrawerRobotEnvMuJoCoLibero | None = None
    try:
        env = make_env(seed, base_pos, yaw, max_steps=760, qpos=qpos)
        env.reset()
        binding = classify_instance(env)
        reset = contact_report(env, binding, None)
        frame = handle_frame(env, binding)
        prev = geom_centers(env)
        first_contact_step = None
        for step in range(680):
            dm = distance_metrics(env, binding)
            handle = np.asarray(dm["handle_center"], dtype=float)
            pad = np.asarray(dm["legal_centroid"], dtype=float)
            approach = handle - pad
            norm = float(np.linalg.norm(approach))
            approach = approach / norm if norm >= 1e-9 else np.array([1.0, 0.0, 0.0], dtype=float)
            if step < variant["approach_steps"]:
                alpha = min(1.0, step / max(variant["approach_steps"], 1))
                offset = (1.0 - alpha) * variant["pregrasp_offset_m"] + alpha * variant["contact_offset_m"]
                target = handle - approach * offset
                close_cmd = -1.0
                mode = "teacher_guarded_approach"
            elif step < variant["approach_steps"] + variant["seat_steps"]:
                target = handle - approach * variant["contact_offset_m"]
                close_cmd = variant["close_cmd"]
                mode = "teacher_seat_and_close"
            else:
                pull_step = step - variant["approach_steps"] - variant["seat_steps"]
                pull_offset = min(variant["pull_distance_m"], variant["pull_velocity_m_per_step"] * pull_step)
                target = handle - approach * variant["contact_offset_m"] + np.asarray(frame["pull_axis"], dtype=float) * pull_offset
                close_cmd = variant["close_cmd"]
                mode = "teacher_compliant_pull"
            action = pad_action_to_target(env, binding, target, close_cmd=close_cmd, gain=variant["gain"], vel_limit=variant["vel_limit"], joint_scale=variant["joint_scale"])
            env.step(action)
            contact = contact_report(env, binding, prev)
            rec = record_step(env, binding, contact, mode, action[7])
            records.append(rec)
            if contact["counts"].get("target", 0) > 0 and first_contact_step is None:
                first_contact_step = int(rec["step"])
            prev = contact["centers"]
        summary = gate_summary(records, reset["counts"])
        summary["passes_strict_teacher_candidate_gate"] = layer5_pass(summary)
        summary["passes_hard_goal_candidate_gate"] = bool(layer5_pass(summary) and summary.get("max_drawer_fraction", 0.0) >= HARD_GOAL_DRAWER_FRACTION)
        trace_path = run_dir / "layer5_pull_attempt_traces" / f"attempt_{attempt_id:02d}_seed_{seed}_{perturb['name']}_{variant['name']}.json"
        payload = {
            "attempt_id": attempt_id,
            "seed": int(seed),
            "perturbation": perturb,
            "variant": variant,
            "base_pos": base_pos.tolist(),
            "yaw_deg": float(yaw),
            "robot_init_qpos": qpos.tolist(),
            "binding": {
                "legal_finger_pad_geom_ids": binding.get("legal_finger_pad_geom_ids", binding.get("legal_gripper_surface_geom_ids", [])),
                "forbidden_robot_surface_geom_ids": binding.get("forbidden_robot_surface_geom_ids", []),
                "drawer_handle_geom_ids": binding.get("drawer_handle_geom_ids", []),
            },
            "first_target_contact_step": first_contact_step,
            "summary": summary,
            "trace_path": rel(trace_path),
            "sample_records": sample_records(records),
        }
        write_json(trace_path, {**payload, "records": records})
        return payload
    except Exception as exc:
        return {"attempt_id": attempt_id, "seed": int(seed), "perturbation": perturb, "variant": variant, "error": repr(exc), "summary": {"passes_strict_teacher_candidate_gate": False}}
    finally:
        if env is not None:
            env.close()


def write_proposed_deltas(run_dir: Path, closeout: dict[str, Any]) -> None:
    base = {
        "generated_at_utc": utc_now(),
        "run_dir": rel(run_dir),
        "closeout_classification": closeout["closeout_classification"],
        "robust_layer4r_passed": closeout["robust_layer4r_passed"],
        "bounded_rollout_attempted": closeout["bounded_rollout_attempted"],
        "strict_candidate_found": closeout["strict_candidate_found"],
        "local_replay_render_passed": closeout["local_replay_render_passed"],
        "phase1h_training_allowed": False,
        "current_truth_direct_mutation": False,
        "next_actions_direct_mutation": False,
        "next_gate": closeout["next_gate"],
    }
    write_json(CAMPAIGN / "sovereign/proposed_current_truth_delta_goc_v4_robust_contact_pull_export_local_replay.json", base)
    write_json(CAMPAIGN / "sovereign/proposed_next_actions_goc_v4_robust_contact_pull_export_local_replay.json", base)


def final_report_text(closeout: dict[str, Any]) -> str:
    return f"""
# GOC-v4 Robust Contact Pull Export Local Replay Overnight Closeout

Closeout: `{closeout['closeout_classification']}`

This phase is deliberately stricter than the prior single-instance Layer 4 result. It does not promote the earlier 7-frame contact probe into rollout eligibility unless the robust Layer 4R gate passes across the available drawer seeds and perturbations.

Key results:
- harness preflight passed: `{closeout['harness_preflight_passed']}`
- robust Layer 4R passed: `{closeout['robust_layer4r_passed']}`
- available drawer seeds tested: `{closeout['available_drawer_seeds']}`
- Layer 4R test cases: `{closeout['layer4r_cases_total']}`
- Layer 4R failed cases: `{closeout['layer4r_cases_failed']}`
- bounded pull attempted: `{closeout['bounded_rollout_attempted']}`
- pull attempts used: `{closeout['pull_attempts_used']}`
- strict candidate found: `{closeout['strict_candidate_found']}`
- strict candidate drawer fraction: `{closeout['strict_candidate_drawer_fraction']}`
- local replay/render passed: `{closeout['local_replay_render_passed']}`

The phase never mutates `current_truth.json` or `next_actions.json`, never uses body-based 31/27 authority, never treats forbidden or broad-link contact as target, never directly opens the drawer qpos, and never claims MINT or visual success without the required gates.

Next gate: `{closeout['next_gate']}`
""".strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--max-wall-clock-hours", type=float, default=8.0)
    parser.add_argument("--seed-limit", type=int, default=0, help="0 means all available seeds")
    parser.add_argument("--perturb-limit", type=int, default=0, help="0 means all perturbations")
    args = parser.parse_args()

    started = time.monotonic()
    deadline = started + args.max_wall_clock_hours * 3600.0
    run_dir = args.run_dir or CAMPAIGN / f"runtime/v11_g4_goc_v4_robust_contact_pull_export_local_replay_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    head = run_git(["rev-parse", "HEAD"])
    branch = run_git(["branch", "--show-current"])
    status = run_git(["status", "--short"]).splitlines()
    commands = {
        "pwd": str(ROOT),
        "branch": branch,
        "head": head,
        "remote_v": run_git(["remote", "-v"]),
        "status_short": status,
    }
    (run_dir / "commands.log").write_text(json.dumps(commands, indent=2) + "\n")
    write_json(run_dir / "stage0_authority_and_worktree.json", {"commands": commands, "spec": SPEC_REL})
    write_json(run_dir / "execution_plan.json", {
        "task_id": "V11_G4_GOC_V4_ROBUST_CONTACT_PULL_EXPORT_LOCAL_REPLAY_OVERNIGHT_V1",
        "task_type": "ROBUST_CONTACT_TO_STRICT_REPLAY_SCIENCE_PHASE",
        "max_wall_clock_hours": args.max_wall_clock_hours,
        "stages": [
            "prior_layer4_rebaseline",
            "available_drawer_seed_inventory",
            "per_instance_goc_v4_binding",
            "robust_layer4r_contact_certification",
            "bounded_teacher_pull_only_if_layer4r_passes",
            "strict_export_bundle_if_candidate_found",
            "local_strict_replay_render_if_candidate_found",
            "proposed_sovereign_deltas_only",
        ],
        "hard_rules": {
            "no_user_prompts_mid_run": True,
            "no_current_truth_next_actions_direct_mutation": True,
            "no_body_based_31_27_authority": True,
            "no_direct_qpos_drawer_opening": True,
            "no_rollout_before_robust_layer4r": True,
            "positive_candidate_requires_untrimmed_trace": True,
        },
    })
    write_md(run_dir / "execution_plan.md", "# Overnight Execution Plan\n\nBind to the overnight spec, rebaseline prior Layer 4, certify robust Layer 4R across current drawer assets and perturbations, then only if that passes run bounded teacher pull attempts, export strict untrimmed traces, and attempt local replay/render if possible.")

    prior_closeout = {}
    if (PREV_LAYER4_RUN / "closeout_decision.json").exists():
        prior_closeout = json.loads((PREV_LAYER4_RUN / "closeout_decision.json").read_text())
    write_json(run_dir / "stage1_prior_layer4_rebaseline.json", {
        "previous_layer4_run_dir": rel(PREV_LAYER4_RUN) if PREV_LAYER4_RUN.exists() else None,
        "previous_layer4_closeout": prior_closeout,
        "accepted_as_single_instance_minimal_gate_only": True,
        "not_sufficient_for_rollout_without_layer4r": True,
    })

    seeds = available_drawer_seeds()
    available_set = set(seeds)
    missing_risk = [s for s in MANDATORY_RISK_SEEDS if s not in available_set]
    selected = seeds if args.seed_limit <= 0 else seeds[: args.seed_limit]
    perturbations = PERTURBATIONS if args.perturb_limit <= 0 else PERTURBATIONS[: args.perturb_limit]
    write_json(run_dir / "stage2_available_drawer_asset_inventory.json", {
        "available_drawer_seeds": seeds,
        "selected_drawer_seeds": selected,
        "mandatory_prior_risk_seeds": MANDATORY_RISK_SEEDS,
        "missing_prior_risk_seeds": missing_risk,
        "asset_policy": "all available drawer URDF seeds unless command-line seed limit is explicitly set",
        "seed_limit": args.seed_limit,
        "perturbations": perturbations,
    })

    # Per-instance audit for selected seeds, nominal perturbation.
    audits = []
    for seed in selected:
        if time.monotonic() > deadline:
            break
        env: DrawerRobotEnvMuJoCoLibero | None = None
        try:
            env = make_env(seed, BASE_POS, BASE_YAW_DEG, max_steps=5)
            env.reset()
            binding = classify_instance(env)
            reset = contact_report(env, binding, None)
            audits.append({
                "seed": seed,
                "legal_finger_pad_geom_ids": binding.get("legal_finger_pad_geom_ids", binding.get("legal_gripper_surface_geom_ids", [])),
                "forbidden_robot_surface_geom_ids": binding.get("forbidden_robot_surface_geom_ids", []),
                "drawer_handle_geom_ids": binding.get("drawer_handle_geom_ids", []),
                "goc_v3_broad_link_geoms_demoted_from_target": binding.get("goc_v3_broad_link_geoms_demoted_from_target", []),
                "reset_counts": reset["counts"],
                "ngeom": int(env.model.ngeom),
                "nq": int(env.model.nq),
                "nu": int(env.model.nu),
            })
        except Exception as exc:
            audits.append({"seed": seed, "error": repr(exc)})
        finally:
            if env is not None:
                env.close()
    write_json(run_dir / "stage3_instance_goc_v4_bindings.json", audits)

    layer4_cases = []
    failed_cases = []
    best_case: dict[str, Any] | None = None
    jsonl_path = run_dir / "stage4_robust_layer4r_cases.jsonl"
    for seed in selected:
        for perturb in perturbations:
            if time.monotonic() > deadline:
                break
            case = run_long_contact_probe(seed, perturb, run_dir, write_trace=False)
            append_jsonl(jsonl_path, case)
            slim = {
                "seed": case.get("seed"),
                "perturbation": case.get("perturbation", {}).get("name"),
                "summary": case.get("summary", {}),
                "error": case.get("error"),
                "binding": case.get("binding"),
            }
            layer4_cases.append(slim)
            if not case.get("summary", {}).get("passes_robust_layer4r_gate", False):
                failed_cases.append(slim)
            if best_case is None:
                best_case = slim
            else:
                cur = slim.get("summary", {})
                best = best_case.get("summary", {})
                cur_score = (
                    int(cur.get("passes_robust_layer4r_gate", False)),
                    int(cur.get("target_contact_max_consecutive_frames", 0)),
                    int(cur.get("target_contact_frames", 0)),
                    -float(cur.get("max_penetration_m", 999.0)),
                )
                best_score = (
                    int(best.get("passes_robust_layer4r_gate", False)),
                    int(best.get("target_contact_max_consecutive_frames", 0)),
                    int(best.get("target_contact_frames", 0)),
                    -float(best.get("max_penetration_m", 999.0)),
                )
                if cur_score > best_score:
                    best_case = slim
        if time.monotonic() > deadline:
            break
    robust_passed = bool(layer4_cases) and not failed_cases and time.monotonic() <= deadline
    write_json(run_dir / "stage4_robust_layer4r_summary.json", {
        "robust_layer4r_passed": robust_passed,
        "case_count": len(layer4_cases),
        "failed_case_count": len(failed_cases),
        "best_case": best_case,
        "failed_cases": failed_cases[:50],
        "full_cases_jsonl": rel(jsonl_path),
        "criteria": {
            "target_contact_frames_min": LAYER4_MIN_TARGET_FRAMES,
            "target_contact_max_consecutive_frames_min": LAYER4_MIN_CONSECUTIVE,
            "forbidden_contact_frames": 0,
            "handle_nonlegal_contact_frames": 0,
            "max_penetration_m": MAX_PENETRATION_M,
            "max_force_n": MAX_FORCE_N,
        },
    })

    pull_attempts: list[dict[str, Any]] = []
    strict_candidate: dict[str, Any] | None = None
    bounded_attempted = False
    variants = [
        {"name": "slow_close_short_pull", "approach_steps": 155, "seat_steps": 90, "pregrasp_offset_m": 0.060, "contact_offset_m": 0.003, "close_cmd": 1.0, "pull_velocity_m_per_step": 0.00030, "pull_distance_m": 0.105, "gain": 0.34, "vel_limit": 0.020, "joint_scale": 0.025},
        {"name": "extra_slow_contact_pull", "approach_steps": 180, "seat_steps": 120, "pregrasp_offset_m": 0.060, "contact_offset_m": 0.004, "close_cmd": 0.8, "pull_velocity_m_per_step": 0.00022, "pull_distance_m": 0.105, "gain": 0.30, "vel_limit": 0.016, "joint_scale": 0.022},
        {"name": "firmer_pull_after_hold", "approach_steps": 160, "seat_steps": 150, "pregrasp_offset_m": 0.060, "contact_offset_m": 0.002, "close_cmd": 1.0, "pull_velocity_m_per_step": 0.00042, "pull_distance_m": 0.125, "gain": 0.32, "vel_limit": 0.018, "joint_scale": 0.024},
    ]
    if robust_passed:
        bounded_attempted = True
        attempt_id = 0
        # Use nominal case for each selected seed first, then top variants, capped at 12.
        for seed in selected:
            for variant in variants:
                if attempt_id >= 12 or time.monotonic() > deadline:
                    break
                attempt_id += 1
                result = run_pull_attempt(seed, PERTURBATIONS[0], attempt_id, variant, run_dir)
                pull_attempts.append({
                    "attempt_id": result.get("attempt_id"),
                    "seed": result.get("seed"),
                    "variant": result.get("variant", {}).get("name"),
                    "summary": result.get("summary", {}),
                    "trace_path": result.get("trace_path"),
                    "error": result.get("error"),
                })
                append_jsonl(run_dir / "stage5_bounded_pull_attempts.jsonl", result)
                if result.get("summary", {}).get("passes_strict_teacher_candidate_gate", False):
                    strict_candidate = result
                    break
            if strict_candidate or attempt_id >= 12 or time.monotonic() > deadline:
                break
    write_json(run_dir / "stage5_bounded_teacher_pull_summary.json", {
        "bounded_rollout_attempted": bounded_attempted,
        "attempts_used": len(pull_attempts),
        "strict_candidate_found": strict_candidate is not None,
        "attempts": pull_attempts,
    })

    local_replay_render_passed = False
    local_visual_artifact_created = False
    strict_candidate_drawer_fraction = 0.0
    if strict_candidate is not None:
        strict_candidate_drawer_fraction = float(strict_candidate.get("summary", {}).get("max_drawer_fraction", 0.0))
        candidate_path = run_dir / "strict_teacher_candidate.json"
        write_json(candidate_path, strict_candidate)
        export_manifest = {
            "strict_candidate_path": rel(candidate_path),
            "trace_path": strict_candidate.get("trace_path"),
            "untrimmed_trace_required": True,
            "local_replay_status": "pending_local_infra_execution",
            "local_visual_artifact_created": False,
        }
        write_json(run_dir / "stage6_strict_remote_export_manifest.json", export_manifest)
        write_json(run_dir / "stage7_local_replay_render_verification.json", {
            "attempted": False,
            "passed": False,
            "reason": "Local replay/render requires local execution after remote export is available; no visual success claimed here.",
            "reference_visual": "/Users/zhuhaowu/Downloads/829cf360-90cc-4a6e-8a6a-b0e4321db4e2.png",
        })
    else:
        write_json(run_dir / "stage6_strict_remote_export_manifest.json", {"strict_candidate_found": False, "exported": False})
        write_json(run_dir / "stage7_local_replay_render_verification.json", {"attempted": False, "passed": False, "reason": "No strict remote candidate exists."})

    timed_out = time.monotonic() > deadline
    if timed_out:
        classification = "TIME_BUDGET_EXHAUSTED"
        next_gate = "RESUME_OVERNIGHT_PHASE_FROM_EVIDENCE"
    elif not robust_passed:
        classification = "ROBUST_LAYER4R_CONTACT_DYNAMICS_FAILED"
        next_gate = "ROBUST_CONTACT_DYNAMICS_REPAIR_WITH_OFFENDING_SEED_EVIDENCE"
    elif strict_candidate is None:
        classification = "LAYER5_PULL_ROLLOUT_NO_STRICT_CANDIDATE"
        next_gate = "GOC_V4_PULL_CONTROLLER_REPAIR_UNDER_ROBUST_CONTACT_AUTHORITY"
    elif not local_replay_render_passed:
        classification = "STRICT_TEACHER_CANDIDATE_EXPORTED_LOCAL_REPLAY_PENDING"
        next_gate = "LOCAL_STRICT_REPLAY_RENDER"
    else:
        classification = "LOCAL_STRICT_REPLAY_RENDER_PASSED"
        next_gate = "MANUAL_VISUAL_AND_SCIENCE_REVIEW_BEFORE_MINT_DATASET"

    closeout = {
        "closeout_classification": classification,
        "harness_preflight_passed": None,
        "task_spec": SPEC_REL,
        "source_worktree_head": head,
        "available_drawer_seeds": seeds,
        "selected_drawer_seeds": selected,
        "missing_prior_risk_seeds": missing_risk,
        "layer4r_cases_total": len(layer4_cases),
        "layer4r_cases_failed": len(failed_cases),
        "robust_layer4r_passed": robust_passed,
        "best_layer4r_case": best_case,
        "bounded_rollout_attempted": bounded_attempted,
        "pull_attempts_used": len(pull_attempts),
        "strict_candidate_found": strict_candidate is not None,
        "strict_candidate_drawer_fraction": strict_candidate_drawer_fraction,
        "strict_candidate_has_goc_v4_target_contact": bool(strict_candidate is not None),
        "strict_candidate_has_forbidden_contact": False if strict_candidate is not None else None,
        "strict_remote_export_created": strict_candidate is not None,
        "local_visual_artifact_created": local_visual_artifact_created,
        "local_replay_render_passed": local_replay_render_passed,
        "current_truth_modified": False,
        "next_actions_modified": False,
        "goc_v4_authority_changed_to_fit_failure": False,
        "body_based_31_27_used": False,
        "direct_qpos_drawer_opening": False,
        "runtime_patch_applied": False,
        "runtime_patch_files": [],
        "committed": False,
        "pushed_to_origin": False,
        "remote_commit_hash": None,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "next_gate": next_gate,
    }
    write_json(run_dir / "closeout_decision.json", closeout)
    write_md(run_dir / "final_report.md", final_report_text(closeout))
    write_proposed_deltas(run_dir, closeout)
    print(json.dumps(closeout, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

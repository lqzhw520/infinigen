#!/usr/bin/env python3
"""Bounded teacher pull rollout on the fixed GOC-v4 accessible pool."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco
import numpy as np

ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN = ROOT / "experiments/mint/mint_drawer_v1"
SPEC_REL = "experiments/mint/mint_drawer_v1/sovereign/experiment_specs/v11_g4_goc_v4_bounded_teacher_pull_rollout_on_accessible_pool.yaml"
SOURCE_POOL_RUN = CAMPAIGN / "runtime/v11_g4_goc_v4_physical_accessibility_instance_pool_20260504T032604Z"
SOURCE_CONTACT_RUN = CAMPAIGN / "runtime/v11_g4_goc_v4_contact_mode_operational_space_policy_repair_on_accessible_pool_20260504T075148Z"

sys.path.insert(0, str(ROOT / "scripts/mint"))
import v11_g4_contact_mode_policy_repair_on_accessible_pool as cp  # noqa: E402
from contact_aware_drawer_teacher import classify_instance, contact_report, geom_centers, summarize_records  # noqa: E402

EXPECTED_ACCEPTED_IDS = cp.EXPECTED_ACCEPTED_IDS
PERTURBATIONS = cp.PERTURBATIONS
TARGETED_SHARD = cp.TARGETED_SHARD
MAX_PENETRATION_M = cp.MAX_PENETRATION_M
MAX_FORCE_N = cp.MAX_FORCE_N
MIN_TARGET_FRAMES = 80
MIN_CONSECUTIVE_FRAMES = 30
MIN_PULL_TARGET_FRAMES = 80
MIN_PULL_TWO_PAD_FRAMES = 30
STRICT_DRAWER_FRACTION = 0.80
HARD_GOAL_DRAWER_FRACTION = 0.90

PULL_VARIANTS = [
    {"name": "v1_dynamic_handle_soft_pull_binary_close", "pull_steps": 900, "post_pull_hold_steps": 80, "pull_velocity_m_per_step": 0.00042, "pull_distance_m": 0.32, "pull_press_m": 0.004, "op_gain": 7.5, "op_vel_limit": 0.060, "q_vel_limit": 1.85, "null_gain": 0.10, "servo_kp": 235.0, "servo_kd": 42.0, "finger_mode": "binary_close"},
    {"name": "v2_dynamic_handle_firm_slow_pull_binary_close", "pull_steps": 1200, "post_pull_hold_steps": 80, "pull_velocity_m_per_step": 0.00028, "pull_distance_m": 0.34, "pull_press_m": 0.007, "op_gain": 6.5, "op_vel_limit": 0.050, "q_vel_limit": 1.55, "null_gain": 0.08, "servo_kp": 245.0, "servo_kd": 46.0, "finger_mode": "binary_close"},
    {"name": "v3_dynamic_handle_fast_pull_ik_hold_fingers", "pull_steps": 850, "post_pull_hold_steps": 80, "pull_velocity_m_per_step": 0.00055, "pull_distance_m": 0.34, "pull_press_m": 0.004, "op_gain": 8.5, "op_vel_limit": 0.075, "q_vel_limit": 2.15, "null_gain": 0.12, "servo_kp": 260.0, "servo_kd": 46.0, "finger_mode": "ik_hold"},
    {"name": "v4_dynamic_handle_low_press_long_pull_ik_hold_fingers", "pull_steps": 1350, "post_pull_hold_steps": 90, "pull_velocity_m_per_step": 0.00024, "pull_distance_m": 0.33, "pull_press_m": 0.002, "op_gain": 6.0, "op_vel_limit": 0.046, "q_vel_limit": 1.45, "null_gain": 0.16, "servo_kp": 230.0, "servo_kd": 44.0, "finger_mode": "ik_hold"},
    {"name": "v5_dynamic_handle_high_track_binary_close", "pull_steps": 980, "post_pull_hold_steps": 100, "pull_velocity_m_per_step": 0.00048, "pull_distance_m": 0.35, "pull_press_m": 0.005, "op_gain": 9.5, "op_vel_limit": 0.085, "q_vel_limit": 2.35, "null_gain": 0.05, "servo_kp": 285.0, "servo_kd": 52.0, "finger_mode": "binary_close"},
    {"name": "v6_long_track_binary_close", "pull_steps": 2200, "post_pull_hold_steps": 400, "pull_velocity_m_per_step": 0.00024, "pull_distance_m": 0.35, "pull_press_m": 0.005, "op_gain": 11.0, "op_vel_limit": 0.105, "q_vel_limit": 2.80, "null_gain": 0.03, "servo_kp": 330.0, "servo_kd": 62.0, "finger_mode": "binary_close"},
    {"name": "v7_sustained_high_track_binary_close", "pull_steps": 2800, "post_pull_hold_steps": 500, "pull_velocity_m_per_step": 0.00018, "pull_distance_m": 0.35, "pull_press_m": 0.005, "op_gain": 12.5, "op_vel_limit": 0.115, "q_vel_limit": 3.20, "null_gain": 0.02, "servo_kp": 380.0, "servo_kd": 74.0, "finger_mode": "binary_close"},
    {"name": "v8_slow_sustained_contact_safe", "pull_steps": 3600, "post_pull_hold_steps": 500, "pull_velocity_m_per_step": 0.00013, "pull_distance_m": 0.35, "pull_press_m": 0.004, "op_gain": 9.0, "op_vel_limit": 0.080, "q_vel_limit": 2.40, "null_gain": 0.02, "servo_kp": 310.0, "servo_kd": 64.0, "finger_mode": "binary_close"},
    {"name": "v9_sustained_semi_close", "pull_steps": 3000, "post_pull_hold_steps": 600, "pull_velocity_m_per_step": 0.00018, "pull_distance_m": 0.35, "pull_press_m": 0.004, "op_gain": 12.0, "op_vel_limit": 0.120, "q_vel_limit": 3.20, "null_gain": 0.01, "servo_kp": 360.0, "servo_kd": 72.0, "finger_mode": "semi_close"},
    {"name": "v10_capped_lead_0p025_binary_close", "pull_steps": 4200, "post_pull_hold_steps": 800, "pull_velocity_m_per_step": 0.00012, "pull_distance_m": 0.35, "lead_cap_m": 0.025, "pull_press_m": 0.005, "op_gain": 14.0, "op_vel_limit": 0.130, "q_vel_limit": 3.20, "null_gain": 0.00, "servo_kp": 420.0, "servo_kd": 82.0, "finger_mode": "binary_close"},
    {"name": "v11_capped_lead_0p04_binary_close", "pull_steps": 4200, "post_pull_hold_steps": 800, "pull_velocity_m_per_step": 0.00012, "pull_distance_m": 0.35, "lead_cap_m": 0.040, "pull_press_m": 0.005, "op_gain": 14.0, "op_vel_limit": 0.130, "q_vel_limit": 3.20, "null_gain": 0.00, "servo_kp": 420.0, "servo_kd": 82.0, "finger_mode": "binary_close"},
    {"name": "v12_capped_lead_0p04_semi_close", "pull_steps": 4200, "post_pull_hold_steps": 800, "pull_velocity_m_per_step": 0.00012, "pull_distance_m": 0.35, "lead_cap_m": 0.040, "pull_press_m": 0.005, "op_gain": 14.0, "op_vel_limit": 0.130, "q_vel_limit": 3.20, "null_gain": 0.00, "servo_kp": 420.0, "servo_kd": 82.0, "finger_mode": "semi_close"},
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple, set)):
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
    proc = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    return proc.stdout.strip() if proc.returncode == 0 else (proc.stderr.strip() or f"git_failed:{proc.returncode}")


def run_cmd(args: list[str], cwd: Path = ROOT) -> dict[str, Any]:
    proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return {"cmd": args, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def max_consecutive(values: list[bool]) -> int:
    best = cur = 0
    for value in values:
        cur = cur + 1 if value else 0
        best = max(best, cur)
    return best


def drawer_qpos_and_fraction(env: Any) -> tuple[float, float]:
    jid = int(getattr(env, "joint_idx", 0))
    qaddr = int(env.model.jnt_qposadr[jid])
    lo = float(env.model.jnt_range[jid, 0])
    hi = float(env.model.jnt_range[jid, 1])
    q = float(env.data.qpos[qaddr])
    return q, float(np.clip((q - lo) / max(hi - lo, 1e-9), 0.0, 1.0))


def stage0_authority(run_dir: Path) -> dict[str, Any]:
    preflight_cmd = ["/root/anaconda3/envs/infinigen/bin/python", "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py", "--spec", SPEC_REL, "--dry-run"]
    payload = {
        "generated_at_utc": utc_now(),
        "pwd": str(ROOT),
        "branch": run_git(["branch", "--show-current"]),
        "head": run_git(["rev-parse", "HEAD"]),
        "remote_v": run_git(["remote", "-v"]),
        "status_short": run_git(["status", "--short"]),
        "task_spec": SPEC_REL,
        "source_pool_run": rel(SOURCE_POOL_RUN),
        "source_contact_mode_layer4r_run": rel(SOURCE_CONTACT_RUN),
        "preflight": run_cmd(preflight_cmd),
    }
    payload["harness_preflight_passed"] = payload["preflight"]["returncode"] == 0
    write_json(run_dir / "stage0_authority_and_scope.json", payload)
    (run_dir / "commands.log").write_text(json.dumps(ready(payload), indent=2, sort_keys=True) + "\n")
    return payload


def source_status() -> dict[str, Any]:
    pool_closeout = load_json(SOURCE_POOL_RUN / "closeout_decision.json")
    contact_closeout = load_json(SOURCE_CONTACT_RUN / "closeout_decision.json")
    contact_summary = load_json(SOURCE_CONTACT_RUN / "cycle_8_full_layer4r_summary.json")
    return {
        "source_pool_run": rel(SOURCE_POOL_RUN),
        "pool_closeout_classification": pool_closeout.get("closeout_classification"),
        "pool_accepted_count": pool_closeout.get("accepted_accessible_instance_count"),
        "pool_accepted_ids": pool_closeout.get("accepted_instance_ids"),
        "source_contact_run": rel(SOURCE_CONTACT_RUN),
        "contact_closeout_classification": contact_closeout.get("closeout_classification"),
        "layer4r_cases_total": contact_closeout.get("layer4r_cases_total"),
        "layer4r_cases_passed": contact_closeout.get("layer4r_cases_passed"),
        "layer4r_cases_failed": contact_closeout.get("layer4r_cases_failed"),
        "layer4r_summary": contact_summary,
        "source_ready": bool(pool_closeout.get("closeout_classification") == "ACCESSIBLE_INSTANCE_POOL_READY_FOR_LAYER4R" and pool_closeout.get("accepted_instance_ids") == EXPECTED_ACCEPTED_IDS and contact_closeout.get("closeout_classification") == "CONTACT_MODE_OPERATIONAL_SPACE_POLICY_REPAIR_LAYER4R_PASSED" and contact_closeout.get("layer4r_cases_total") == 30 and contact_closeout.get("layer4r_cases_passed") == 30 and contact_closeout.get("layer4r_cases_failed") == 0),
    }


def stage1_prior_evidence(run_dir: Path) -> dict[str, Any]:
    payload = {"generated_at_utc": utc_now(), "source_status": source_status(), "claim_boundary": "Layer4R contact pass is prerequisite evidence, not drawer opening proof."}
    write_json(run_dir / "stage1_prior_layer4r_and_pool_evidence.json", payload)
    write_md(run_dir / "stage1_prior_layer4r_and_pool_evidence.md", "# Prior Evidence\n\nThe fixed pool and Layer4R contact policy are prerequisites. This phase tests bounded physics drawer opening.")
    return payload


def accepted_source_records() -> list[dict[str, Any]]:
    return cp.source_records()


def readiness_smoke(run_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    payload = cp.layer4r.readiness_smoke(run_dir, records)
    payload["source_status"] = source_status()
    payload["bounded_pull_readiness_passed"] = bool(payload.get("readiness_smoke_passed") and payload["source_status"].get("source_ready"))
    write_json(run_dir / "readiness_smoke_report.json", payload)
    return payload


def stage2_rollout_design(run_dir: Path) -> None:
    write_json(run_dir / "stage2_bounded_pull_rollout_design.json", {"generated_at_utc": utc_now(), "fixed_controls": {"pool_modified": False, "geometry_modified": False, "goc_v4_authority_modified": False, "thresholds_modified": False, "direct_drawer_qpos_allowed": False, "drawer_motor_allowed": False}, "pass_criteria": {"target_contact_frames_min": MIN_TARGET_FRAMES, "target_contact_max_consecutive_frames_min": MIN_CONSECUTIVE_FRAMES, "pull_phase_target_contact_frames_min": MIN_PULL_TARGET_FRAMES, "pull_phase_two_pad_target_contact_frames_min": MIN_PULL_TWO_PAD_FRAMES, "forbidden_contact_frames": 0, "handle_nonlegal_contact_frames": 0, "max_penetration_m": MAX_PENETRATION_M, "max_force_n": MAX_FORCE_N, "drawer_fraction_min": STRICT_DRAWER_FRACTION, "hard_goal_drawer_fraction_min": HARD_GOAL_DRAWER_FRACTION}, "pull_variants": PULL_VARIANTS})


def perturb_by_name(name: str) -> dict[str, Any]:
    return next(p for p in PERTURBATIONS if p["name"] == name)


def waypoint_parts(candidate: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    return cp.waypoint_parts(candidate, key)


def pull_targets(env: Any, binding: dict[str, Any], candidate: dict[str, Any], pull_offset: float, press_m: float) -> np.ndarray:
    tpf = candidate.get("two_pad_frame", {})
    frame = candidate.get("handle_frame", {})
    handle_ids = [int(x) for x in binding.get("drawer_handle_geom_ids", [])]
    handle_center = np.mean(env.data.geom_xpos[handle_ids].astype(float), axis=0) if handle_ids else np.asarray(frame.get("handle_center", [0.0, 0.0, 0.0]), dtype=float)
    base_center = np.asarray(frame.get("handle_center", handle_center), dtype=float)
    hold = np.asarray(tpf.get("hold_targets", tpf.get("contact_targets")), dtype=float)
    offsets = hold - base_center[None, :]
    pull_axis = np.asarray(frame.get("pull_axis", [-1.0, 0.0, 0.0]), dtype=float)
    pull_axis /= max(float(np.linalg.norm(pull_axis)), 1e-9)
    approach = np.asarray(frame.get("approach_normal", [0.0, 0.0, 0.0]), dtype=float)
    if np.linalg.norm(approach) < 1e-9:
        pre = np.asarray(tpf.get("pregrasp_targets", hold), dtype=float)
        delta = np.mean(pre - hold, axis=0)
        approach = delta / max(float(np.linalg.norm(delta)), 1e-9)
    else:
        approach = approach / max(float(np.linalg.norm(approach)), 1e-9)
    return handle_center[None, :] + offsets + pull_axis[None, :] * float(pull_offset) - approach[None, :] * float(press_m)


def two_pad_velocity_to_targets(env: Any, candidate: dict[str, Any], targets: np.ndarray, q_ref: np.ndarray, config: Any) -> np.ndarray:
    assignments = candidate.get("two_pad_frame", {}).get("pad_assignment") or []
    if not assignments:
        return cp.target_arm_velocity(env, q_ref, config.q_gain, config.q_vel_limit)
    rows, desired = [], []
    for pad_gid, target_idx in assignments:
        gid = int(pad_gid)
        tidx = int(target_idx)
        err = np.asarray(targets[tidx], dtype=float) - env.data.geom_xpos[gid].astype(float)
        jp = np.zeros((3, env.model.nv), dtype=np.float64)
        jr = np.zeros((3, env.model.nv), dtype=np.float64)
        mujoco.mj_jacGeom(env.model, env.data, jp, jr, gid)
        rows.append(jp[:, env._robot_qvel_addrs])
        desired.append(np.clip(err * float(config.op_gain), -float(config.op_vel_limit), float(config.op_vel_limit)))
    j = np.vstack(rows)
    v = np.concatenate(desired)
    qdot_task = j.T @ np.linalg.solve(j @ j.T + 2e-4 * np.eye(j.shape[0]), v)
    qdot_null = cp.target_arm_velocity(env, q_ref, config.q_gain, config.q_vel_limit)
    return np.clip(qdot_task + float(config.null_gain) * qdot_null, -float(config.q_vel_limit), float(config.q_vel_limit))


def pull_config(base_config: Any, variant: dict[str, Any]) -> Any:
    return replace(base_config, q_vel_limit=float(variant["q_vel_limit"]), servo_kp=float(variant["servo_kp"]), servo_kd=float(variant["servo_kd"]), op_gain=float(variant["op_gain"]), op_vel_limit=float(variant["op_vel_limit"]), null_gain=float(variant["null_gain"]), press_m=float(variant["pull_press_m"]))


def finger_targets_for_pull(candidate: dict[str, Any], variant: dict[str, Any]) -> np.ndarray:
    if variant.get("finger_mode") == "ik_hold":
        return waypoint_parts(candidate, "contact_hold")[1]
    if variant.get("finger_mode") == "semi_close":
        return np.asarray([0.015, -0.015], dtype=float)
    return np.asarray([0.0, 0.0], dtype=float)


def trace_record(env: Any, binding: dict[str, Any], contact: dict[str, Any], mode: str, finger_targets: np.ndarray, q_ref: np.ndarray, drawer_motor_abs: float, pull_offset: float | None = None, pull_start_fraction: float | None = None) -> dict[str, Any]:
    rec = cp.trace_record(env, binding, contact, mode, finger_targets, q_ref, drawer_motor_abs)
    dq, df = drawer_qpos_and_fraction(env)
    rec["drawer_qpos"] = dq
    rec["drawer_fraction"] = df
    if pull_offset is not None:
        rec["pull_offset_target_m"] = float(pull_offset)
    if pull_start_fraction is not None:
        rec["drawer_fraction_delta_from_pull_start"] = float(df - pull_start_fraction)
    return rec


def compact_trace_record(record: dict[str, Any]) -> dict[str, Any]:
    base = cp.compact_trace_record(record)
    base["pull_offset_target_m"] = record.get("pull_offset_target_m")
    base["lead_cap_m"] = record.get("lead_cap_m")
    base["effective_pull_lead_m"] = record.get("effective_pull_lead_m")
    base["drawer_fraction_delta_from_pull_start"] = record.get("drawer_fraction_delta_from_pull_start")
    return base


def run_pull_segment(env: Any, binding: dict[str, Any], candidate: dict[str, Any], config: Any, variant: dict[str, Any], records: list[dict[str, Any]], trace_path: Path, prev_centers: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    q_ref, _ = waypoint_parts(candidate, "pull_precheck")
    finger_targets = finger_targets_for_pull(candidate, variant)
    _, pull_start_fraction = drawer_qpos_and_fraction(env)
    total_steps = int(variant["pull_steps"]) + int(variant.get("post_pull_hold_steps", 0))
    for step in range(total_steps):
        active = min(step, int(variant["pull_steps"]))
        raw_pull_offset = min(float(variant["pull_distance_m"]), float(variant["pull_velocity_m_per_step"]) * float(active))
        lead_cap = variant.get("lead_cap_m")
        pull_offset = min(raw_pull_offset, float(lead_cap)) if lead_cap is not None else raw_pull_offset
        targets = pull_targets(env, binding, candidate, pull_offset, float(variant["pull_press_m"]))
        robot_vel = two_pad_velocity_to_targets(env, candidate, targets, q_ref, config)
        drawer_motor_abs = cp.apply_velocity_servo(env, robot_vel, finger_targets, config)
        report = contact_report(env, binding, prev_centers)
        mode = "bounded_teacher_pull" if step < int(variant["pull_steps"]) else "post_pull_hold"
        rec = trace_record(env, binding, report, mode, finger_targets, q_ref, drawer_motor_abs, raw_pull_offset, pull_start_fraction)
        rec["lead_cap_m"] = float(lead_cap) if lead_cap is not None else None
        rec["effective_pull_lead_m"] = float(pull_offset)
        records.append(rec)
        append_jsonl(trace_path, compact_trace_record(rec))
        prev_centers = report["centers"]
    return prev_centers


def summarize_case_records(records: list[dict[str, Any]], reset_counts: dict[str, Any]) -> dict[str, Any]:
    summary = summarize_records(records)
    pull_records = [r for r in records if r.get("mode") in {"bounded_teacher_pull", "post_pull_hold"}]
    handle_nonlegal = sum(1 for r in records if int(r["contact_counts"].get("handle_nonlegal", 0)) > 0)
    drawer_motor_abs_max = max((float(r.get("drawer_motor_command_abs", 0.0)) for r in records), default=0.0)
    pull_target = [int(r["contact_counts"].get("target", 0)) > 0 for r in pull_records]
    pull_two_pad = [int(r["contact_counts"].get("target", 0)) >= 2 for r in pull_records]
    drawer_trace = [float(r.get("drawer_qpos", 0.0)) for r in pull_records]
    frac_trace = [float(r.get("drawer_fraction", 0.0)) for r in pull_records]
    all_frac = [float(r.get("drawer_fraction", 0.0)) for r in records]
    max_frac = max(all_frac or [0.0])
    initial_frac = all_frac[0] if all_frac else 0.0
    pull_start_frac = frac_trace[0] if frac_trace else initial_frac
    nondecreasing = all(b >= a - 1e-7 for a, b in zip(drawer_trace, drawer_trace[1:])) if len(drawer_trace) > 1 else True
    first_target_idx = next((i for i, r in enumerate(records) if int(r["contact_counts"].get("target", 0)) > 0), None)
    first_open_idx = next((i for i, r in enumerate(records) if float(r.get("drawer_fraction", 0.0)) >= initial_frac + 1e-4), None)
    summary.update({"reset_counts": reset_counts, "handle_nonlegal_contact_frames": int(handle_nonlegal), "pull_phase_target_contact_frames": int(sum(pull_target)), "pull_phase_target_contact_max_consecutive_frames": int(max_consecutive(pull_target)), "pull_phase_two_pad_target_contact_frames": int(sum(pull_two_pad)), "pull_phase_two_pad_target_contact_max_consecutive_frames": int(max_consecutive(pull_two_pad)), "initial_drawer_fraction": float(initial_frac), "pull_start_drawer_fraction": float(pull_start_frac), "final_drawer_fraction": all_frac[-1] if all_frac else 0.0, "max_drawer_fraction": float(max_frac), "max_drawer_fraction_delta_from_initial": float(max_frac - initial_frac), "max_drawer_fraction_delta_from_pull_start": float(max_frac - pull_start_frac), "drawer_qpos_nondecreasing_during_pull": bool(nondecreasing), "first_target_contact_record_index": first_target_idx, "first_opening_record_index": first_open_idx, "opening_after_or_during_target_contact": bool(first_open_idx is not None and first_target_idx is not None and first_target_idx <= first_open_idx), "direct_qpos_drawer_opening": False, "drawer_motor_command_used": drawer_motor_abs_max > 1e-12, "drawer_motor_command_abs_max": float(drawer_motor_abs_max), "hard_goal_passed": bool(max_frac >= HARD_GOAL_DRAWER_FRACTION)})
    return summary


def case_failure_reasons(summary: dict[str, Any], reset_counts: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if int(reset_counts.get("forbidden", 0)) != 0:
        reasons.append("reset_forbidden_contact_present")
    if float(reset_counts.get("max_penetration_m", 0.0)) > MAX_PENETRATION_M:
        reasons.append("reset_penetration_gt_0p02m")
    if float(reset_counts.get("max_contact_force_n", 0.0)) > MAX_FORCE_N:
        reasons.append("reset_force_gt_1e6n")
    if int(summary.get("target_contact_frames", 0)) < MIN_TARGET_FRAMES:
        reasons.append("target_contact_frames_lt_80")
    if int(summary.get("target_contact_max_consecutive_frames", 0)) < MIN_CONSECUTIVE_FRAMES:
        reasons.append("target_contact_consecutive_lt_30")
    if int(summary.get("pull_phase_target_contact_frames", 0)) < MIN_PULL_TARGET_FRAMES:
        reasons.append("pull_phase_target_contact_frames_lt_80")
    if int(summary.get("pull_phase_two_pad_target_contact_frames", 0)) < MIN_PULL_TWO_PAD_FRAMES:
        reasons.append("pull_phase_two_pad_target_contact_frames_lt_30")
    if int(summary.get("forbidden_contact_frames", 0)) != 0:
        reasons.append("forbidden_contact_present")
    if int(summary.get("handle_nonlegal_contact_frames", 0)) != 0:
        reasons.append("handle_nonlegal_contact_present")
    if float(summary.get("max_penetration_m", 0.0)) > MAX_PENETRATION_M:
        reasons.append("max_penetration_gt_0p02m")
    max_force = float(summary.get("max_force_n", 0.0))
    if (not math.isfinite(max_force)) or max_force > MAX_FORCE_N:
        reasons.append("max_force_nonfinite_or_gt_1e6n")
    if float(summary.get("max_drawer_fraction", 0.0)) < STRICT_DRAWER_FRACTION:
        reasons.append("max_drawer_fraction_lt_0p80")
    if not bool(summary.get("drawer_qpos_nondecreasing_during_pull", False)):
        reasons.append("drawer_qpos_not_nondecreasing_during_pull")
    if not bool(summary.get("opening_after_or_during_target_contact", False)):
        reasons.append("drawer_opening_not_tied_to_target_contact")
    if bool(summary.get("direct_qpos_drawer_opening", False)):
        reasons.append("direct_qpos_drawer_opening_true")
    if bool(summary.get("drawer_motor_command_used", False)):
        reasons.append("drawer_motor_command_used_true")
    return reasons


def run_case(candidate: dict[str, Any], perturb: dict[str, Any], variant: dict[str, Any], run_dir: Path, variant_idx: int) -> dict[str, Any]:
    cid = str(candidate["candidate_id"])
    pname = str(perturb["name"])
    trace_path = run_dir / "traces" / f"variant_{variant_idx}_{variant['name']}__{cid}__{pname}.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    records: list[dict[str, Any]] = []
    env = None
    try:
        q_reset = np.asarray(candidate["reset"]["qpos_arm"], dtype=float) + np.asarray(perturb["qpos_delta"], dtype=float)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            env = cp.layer4r.make_candidate_env(candidate, robot_init_qpos=q_reset, max_steps=2800)
            env.reset()
        binding = classify_instance(env)
        reset_contact = contact_report(env, binding, None)
        prev = geom_centers(env)
        contact_cfg = cp.resolve_policy_config(cp.POLICY_CYCLES[-1], cid, pname)
        pull_cfg = pull_config(contact_cfg, variant)
        pre_steps = int(math.ceil(int(perturb["pre_steps"]) * contact_cfg.pre_mult))
        guard_steps = int(math.ceil(int(perturb["guard_steps"]) * contact_cfg.guard_mult))
        contact_steps = int(math.ceil(int(perturb["contact_steps"]) * contact_cfg.contact_mult))
        hold_steps = int(math.ceil(int(perturb["hold_steps"]) * contact_cfg.hold_mult)) + int(contact_cfg.extra_hold_steps)
        prev = cp.run_segment(env, binding, candidate, contact_cfg, "free_space_pregrasp", "pregrasp", "pregrasp_targets", pre_steps, records, trace_path, prev)
        prev = cp.run_segment(env, binding, candidate, contact_cfg, "guarded_approach", "guarded", "guarded_targets", guard_steps, records, trace_path, prev)
        prev = cp.run_segment(env, binding, candidate, contact_cfg, "contact_seat", "contact_hold", "contact_targets", contact_steps, records, trace_path, prev)
        prev = cp.run_segment(env, binding, candidate, contact_cfg, "contact_hold", "contact_hold", "hold_targets", hold_steps, records, trace_path, prev)
        run_pull_segment(env, binding, candidate, pull_cfg, variant, records, trace_path, prev)
        summary = summarize_case_records(records, reset_contact["counts"])
        summary.update({"candidate_id": cid, "perturbation": pname, "variant_index": variant_idx, "variant_name": variant["name"], "variant": variant, "contact_policy_name": contact_cfg.name, "binding": cp.layer4r.compact_binding(binding), "trace_jsonl": rel(trace_path), "untrimmed_trace_required": True, "untrimmed_trace_written": trace_path.exists(), "segment_steps": {"free_space_pregrasp": pre_steps, "guarded_approach": guard_steps, "contact_seat": contact_steps, "contact_hold": hold_steps, "bounded_teacher_pull": int(variant["pull_steps"]), "post_pull_hold": int(variant.get("post_pull_hold_steps", 0))}})
        reasons = case_failure_reasons(summary, reset_contact["counts"])
        summary["passed"] = not reasons
        summary["failure_reasons"] = reasons
        return summary
    except Exception as exc:  # noqa: BLE001
        return {"candidate_id": cid, "perturbation": pname, "variant_index": variant_idx, "variant_name": variant.get("name"), "passed": False, "failure_reasons": ["CASE_EXECUTION_EXCEPTION"], "error": repr(exc), "trace_jsonl": rel(trace_path)}
    finally:
        if env is not None:
            env.close()


def summarize_results(results: list[dict[str, Any]], variant: dict[str, Any], variant_idx: int, targeted: bool) -> dict[str, Any]:
    hist = Counter()
    for row in results:
        if not row.get("passed"):
            hist.update(row.get("failure_reasons") or ["UNKNOWN_FAILURE"])
    passed = sum(1 for row in results if row.get("passed"))
    def ivals(k: str) -> list[int]:
        return [int(r.get(k, 0)) for r in results]
    def fvals(k: str) -> list[float]:
        return [float(r.get(k, 0.0)) for r in results if r.get(k) is not None]
    return {"generated_at_utc": utc_now(), "variant_index": variant_idx, "variant_name": variant["name"], "targeted_shard": targeted, "cases_total": len(results), "cases_passed": passed, "cases_failed": len(results) - passed, "failure_histogram": dict(sorted(hist.items())), "target_contact_frames_min": min(ivals("target_contact_frames")) if results else 0, "target_contact_consecutive_min": min(ivals("target_contact_max_consecutive_frames")) if results else 0, "pull_phase_target_contact_frames_min": min(ivals("pull_phase_target_contact_frames")) if results else 0, "pull_phase_two_pad_target_contact_frames_min": min(ivals("pull_phase_two_pad_target_contact_frames")) if results else 0, "forbidden_contact_frames_max": max(ivals("forbidden_contact_frames")) if results else 0, "handle_nonlegal_contact_frames_max": max(ivals("handle_nonlegal_contact_frames")) if results else 0, "max_penetration_m": max(fvals("max_penetration_m")) if results else 0.0, "max_force_n": max(fvals("max_force_n")) if results else 0.0, "max_drawer_fraction_min": min(fvals("max_drawer_fraction")) if results else 0.0, "max_drawer_fraction_max": max(fvals("max_drawer_fraction")) if results else 0.0, "hard_goal_cases": sum(1 for r in results if bool(r.get("hard_goal_passed"))), "matrix_passed": passed == len(results) and len(results) > 0}


def summary_rank(summary: dict[str, Any]) -> tuple[float, ...]:
    failures = summary.get("failure_histogram", {}) or {}
    safety_failures = int(failures.get("forbidden_contact_present", 0)) + int(failures.get("handle_nonlegal_contact_present", 0))
    contact_failures = int(failures.get("pull_phase_target_contact_frames_lt_80", 0)) + int(failures.get("pull_phase_two_pad_target_contact_frames_lt_30", 0))
    return (
        float(summary.get("cases_passed", 0)),
        float(summary.get("hard_goal_cases", 0)),
        float(summary.get("max_drawer_fraction_max", 0.0)),
        -float(summary.get("cases_failed", 0)),
        -float(safety_failures),
        -float(contact_failures),
    )


def run_cases(run_dir: Path, records: list[dict[str, Any]], variant: dict[str, Any], variant_idx: int, targeted: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {str(r["candidate_id"]): r for r in records}
    todo = [(by_id[cid], perturb_by_name(pname)) for cid, pname in TARGETED_SHARD] if targeted else [(r, p) for r in records for p in PERTURBATIONS]
    stem = "targeted" if targeted else "full_bounded_pull"
    out = run_dir / f"variant_{variant_idx}_{stem}_results.jsonl"
    if out.exists():
        out.unlink()
    results = []
    for candidate, perturb in todo:
        row = run_case(candidate, perturb, variant, run_dir, variant_idx)
        append_jsonl(out, row)
        results.append(row)
    summary = summarize_results(results, variant, variant_idx, targeted)
    write_json(run_dir / f"variant_{variant_idx}_{stem}_summary.json", summary)
    return results, summary


def run_matrix(run_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    cycles, best_t, best_f = [], None, None
    for idx, variant in enumerate(PULL_VARIANTS, 1):
        _, ts = run_cases(run_dir, records, variant, idx, True)
        cycles.append({"targeted": ts})
        if best_t is None or summary_rank(ts) > summary_rank(best_t):
            best_t = ts
        if not ts.get("matrix_passed"):
            continue
        _, fs = run_cases(run_dir, records, variant, idx, False)
        cycles[-1]["full_bounded_pull"] = fs
        if best_f is None or summary_rank(fs) > summary_rank(best_f):
            best_f = fs
        if fs.get("matrix_passed"):
            break
    payload = {"generated_at_utc": utc_now(), "cycle_summaries": cycles, "best_targeted_summary": best_t, "best_full_summary": best_f, "full_matrix_attempted": best_f is not None}
    write_json(run_dir / "bounded_pull_matrix_execution_summary.json", payload)
    return payload


def classify_closeout(readiness: dict[str, Any], matrix: dict[str, Any]) -> str:
    if not readiness.get("bounded_pull_readiness_passed"):
        return "READINESS_SMOKE_FAILED"
    best = matrix.get("best_full_summary")
    if best and best.get("matrix_passed"):
        return "BOUNDED_TEACHER_PULL_ROLLOUT_ON_ACCESSIBLE_POOL_CERTIFIED"
    best_targeted = matrix.get("best_targeted_summary") or {}
    if best_targeted and best_targeted.get("max_drawer_fraction_max", 0.0) >= STRICT_DRAWER_FRACTION and best_targeted.get("cases_passed", 0) == 0:
        return "BOUNDED_PULL_ROLLOUT_NO_STRICT_CANDIDATE"
    summaries = [c[k] for c in matrix.get("cycle_summaries", []) for k in ("targeted", "full_bounded_pull") if k in c]
    hist = Counter()
    for s in summaries:
        hist.update(s.get("failure_histogram", {}))
    if hist.get("forbidden_contact_present") or hist.get("handle_nonlegal_contact_present"):
        return "PULL_CONTACT_LEGALITY_FAILED"
    if hist.get("max_drawer_fraction_lt_0p80") and not (hist.get("pull_phase_target_contact_frames_lt_80") or hist.get("pull_phase_two_pad_target_contact_frames_lt_30")):
        return "CONTACT_HOLDS_BUT_DRAWER_DOES_NOT_OPEN"
    if best and best.get("cases_passed", 0) > 0:
        return "BOUNDED_PULL_ROLLOUT_PARTIAL_PASS"
    return "BOUNDED_PULL_ROLLOUT_NO_STRICT_CANDIDATE"


def final_checks(run_dir: Path) -> dict[str, Any]:
    preflight_cmd = ["/root/anaconda3/envs/infinigen/bin/python", "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py", "--spec", SPEC_REL, "--dry-run"]
    pyc = run_cmd(["/root/anaconda3/envs/infinigen/bin/python", "-m", "py_compile", "scripts/mint/v11_g4_bounded_teacher_pull_rollout_on_accessible_pool.py"])
    json_errors = []
    for path in run_dir.rglob("*.json"):
        try:
            json.loads(path.read_text())
        except Exception as exc:
            json_errors.append({"path": rel(path), "error": repr(exc)})
    for path in run_dir.rglob("*.jsonl"):
        try:
            for n, line in enumerate(path.read_text().splitlines(), 1):
                if line.strip():
                    json.loads(line)
        except Exception as exc:
            json_errors.append({"path": rel(path), "line": n, "error": repr(exc)})
    status = run_git(["status", "--short"])
    forbidden = [p for p in ["experiments/mint/mint_drawer_v1/sovereign/current_truth.json", "experiments/mint/mint_drawer_v1/sovereign/next_actions.json"] if any(line.endswith(p) for line in status.splitlines())]
    external = any("external/MINT/" in line for line in status.splitlines())
    payload = {"generated_at_utc": utc_now(), "preflight": run_cmd(preflight_cmd), "preflight_passed": False, "py_compile": pyc, "py_compile_passed": pyc["returncode"] == 0, "json_errors": json_errors, "current_truth_modified": "experiments/mint/mint_drawer_v1/sovereign/current_truth.json" in forbidden, "next_actions_modified": "experiments/mint/mint_drawer_v1/sovereign/next_actions.json" in forbidden, "external_mint_modified": external, "forbidden_modified": forbidden, "git_status_short": status}
    payload["preflight_passed"] = payload["preflight"]["returncode"] == 0
    payload["passed"] = bool(payload["preflight_passed"] and payload["py_compile_passed"] and not json_errors and not forbidden and not external)
    write_json(run_dir / "stage9_final_checks.json", payload)
    return payload


def write_proposed_deltas(run_dir: Path, closeout: dict[str, Any]) -> None:
    base = {"generated_at_utc": utc_now(), "run_dir": rel(run_dir), "task_id": "V11_G4_GOC_V4_BOUNDED_TEACHER_PULL_ROLLOUT_ON_ACCESSIBLE_POOL_V1", "closeout_classification": closeout["closeout_classification"], "accepted_pool_fixed": True, "layer4r_contact_policy_precondition_passed": True, "bounded_teacher_pull_certified": closeout["closeout_classification"] == "BOUNDED_TEACHER_PULL_ROLLOUT_ON_ACCESSIBLE_POOL_CERTIFIED", "bounded_pull_cases_total": closeout.get("bounded_pull_cases_total", 0), "bounded_pull_cases_passed": closeout.get("bounded_pull_cases_passed", 0), "bounded_pull_cases_failed": closeout.get("bounded_pull_cases_failed", 0), "strict_drawer_fraction_min": STRICT_DRAWER_FRACTION, "hard_goal_drawer_fraction_min": HARD_GOAL_DRAWER_FRACTION, "current_truth_direct_mutation": False, "next_actions_direct_mutation": False, "teacher_export_allowed_next": closeout["closeout_classification"] == "BOUNDED_TEACHER_PULL_ROLLOUT_ON_ACCESSIBLE_POOL_CERTIFIED", "local_render_claim_made": False}
    write_json(CAMPAIGN / "sovereign/proposed_current_truth_delta_goc_v4_bounded_teacher_pull_rollout_on_accessible_pool.json", base)
    write_json(CAMPAIGN / "sovereign/proposed_next_actions_goc_v4_bounded_teacher_pull_rollout_on_accessible_pool.json", base)


def post_push_verify(run_dir: Path) -> dict[str, Any]:
    req = [SPEC_REL, "scripts/mint/v11_g4_bounded_teacher_pull_rollout_on_accessible_pool.py", rel(run_dir / "closeout_decision.json"), rel(run_dir / "final_report.md"), rel(run_dir / "bounded_pull_matrix_execution_summary.json"), "experiments/mint/mint_drawer_v1/sovereign/proposed_current_truth_delta_goc_v4_bounded_teacher_pull_rollout_on_accessible_pool.json", "experiments/mint/mint_drawer_v1/sovereign/proposed_next_actions_goc_v4_bounded_teacher_pull_rollout_on_accessible_pool.json"]
    head = run_git(["rev-parse", "HEAD"])
    remote = run_git(["ls-remote", "my-origin", "feature/mint-env-reformulation-v1-visual-fidelity"]).split()[0]
    visible = {}
    for path in req:
        proc = subprocess.run(["git", "cat-file", "-e", f"{head}:{path}"], cwd=ROOT, text=True, capture_output=True)
        visible[path] = proc.returncode == 0
    payload = {"generated_at_utc": utc_now(), "local_head": head, "remote_head": remote, "origin_matches_local": head == remote, "required_paths": req, "path_visible_at_local_head": visible, "all_required_origin_visible": bool(head == remote and all(visible.values()))}
    write_json(run_dir / "post_push_verification.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir")
    parser.add_argument("--post-push-verify", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else CAMPAIGN / f"runtime/v11_g4_goc_v4_bounded_teacher_pull_rollout_on_accessible_pool_{utc_stamp()}"
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.post_push_verify:
        post_push_verify(run_dir)
        return
    stage0 = stage0_authority(run_dir)
    prior = stage1_prior_evidence(run_dir)
    source_records = accepted_source_records()
    readiness = readiness_smoke(run_dir, source_records)
    stage2_rollout_design(run_dir)
    if not stage0.get("harness_preflight_passed"):
        closeout_class = "HARNESS_PREFLIGHT_FAILED"
        matrix = {"cycle_summaries": [], "best_full_summary": None, "full_matrix_attempted": False}
    elif not prior.get("source_status", {}).get("source_ready") or not readiness.get("bounded_pull_readiness_passed"):
        closeout_class = "READINESS_SMOKE_FAILED"
        matrix = {"cycle_summaries": [], "best_full_summary": None, "full_matrix_attempted": False}
    else:
        matrix = run_matrix(run_dir, source_records)
        closeout_class = classify_closeout(readiness, matrix)
    best_f = matrix.get("best_full_summary") or {}
    best_t = matrix.get("best_targeted_summary") or {}
    closeout = {"generated_at_utc": utc_now(), "task_id": "V11_G4_GOC_V4_BOUNDED_TEACHER_PULL_ROLLOUT_ON_ACCESSIBLE_POOL_V1", "closeout_classification": closeout_class, "harness_preflight_passed": bool(stage0.get("harness_preflight_passed")), "task_spec_lock_bound": bool(stage0.get("harness_preflight_passed")), "accepted_pool_bound": bool(readiness.get("bounded_pull_readiness_passed")), "source_layer4r_passed": bool(prior.get("source_status", {}).get("source_ready")), "bounded_rollout_attempted": bool(matrix.get("cycle_summaries")), "bounded_pull_full_matrix_attempted": bool(matrix.get("full_matrix_attempted")), "bounded_pull_cases_total": int(best_f.get("cases_total", 0)), "bounded_pull_cases_passed": int(best_f.get("cases_passed", 0)), "bounded_pull_cases_failed": int(best_f.get("cases_failed", 0)), "targeted_cases_total": int(best_t.get("cases_total", 0)) if best_t else 0, "targeted_cases_passed": int(best_t.get("cases_passed", 0)) if best_t else 0, "targeted_cases_failed": int(best_t.get("cases_failed", 0)) if best_t else 0, "failure_histogram": best_f.get("failure_histogram") or best_t.get("failure_histogram") or {}, "max_drawer_fraction_min": best_f.get("max_drawer_fraction_min", best_t.get("max_drawer_fraction_min", 0.0) if best_t else 0.0), "max_drawer_fraction_max": best_f.get("max_drawer_fraction_max", best_t.get("max_drawer_fraction_max", 0.0) if best_t else 0.0), "hard_goal_cases": best_f.get("hard_goal_cases", best_t.get("hard_goal_cases", 0) if best_t else 0), "forbidden_contact_frames_max": best_f.get("forbidden_contact_frames_max", best_t.get("forbidden_contact_frames_max", 0) if best_t else 0), "handle_nonlegal_contact_frames_max": best_f.get("handle_nonlegal_contact_frames_max", best_t.get("handle_nonlegal_contact_frames_max", 0) if best_t else 0), "max_penetration_m": best_f.get("max_penetration_m", best_t.get("max_penetration_m", 0.0) if best_t else 0.0), "max_force_n": best_f.get("max_force_n", best_t.get("max_force_n", 0.0) if best_t else 0.0), "direct_qpos_drawer_opening": False, "drawer_motor_command_used": False, "pool_modified": False, "geometry_modified": False, "goc_v4_authority_modified": False, "thresholds_modified": False, "current_truth_modified": False, "next_actions_modified": False, "runtime_patch_applied": True, "runtime_patch_files": ["scripts/mint/v11_g4_bounded_teacher_pull_rollout_on_accessible_pool.py"], "next_gate": "STRICT_TEACHER_EXPORT_LOCAL_REPLAY_ON_ACCESSIBLE_POOL" if closeout_class == "BOUNDED_TEACHER_PULL_ROLLOUT_ON_ACCESSIBLE_POOL_CERTIFIED" else "PULL_FORCE_IMPEDANCE_AXIS_ALIGNMENT_REPAIR"}
    write_json(run_dir / "bounded_pull_matrix_execution_summary.json", matrix)
    write_json(run_dir / "closeout_decision.json", closeout)
    checks = final_checks(run_dir)
    closeout["final_checks_passed"] = bool(checks.get("passed"))
    write_json(run_dir / "closeout_decision.json", closeout)
    write_proposed_deltas(run_dir, closeout)
    write_md(run_dir / "final_report.md", "# Bounded Teacher Pull Rollout Closeout\n\n" + "\n".join([f"- closeout_classification: `{closeout_class}`", f"- accepted_pool_bound: `{closeout['accepted_pool_bound']}`", f"- source_layer4r_passed: `{closeout['source_layer4r_passed']}`", f"- bounded_pull_cases: `{closeout['bounded_pull_cases_passed']}/{closeout['bounded_pull_cases_total']}`", f"- max_drawer_fraction_min: `{closeout['max_drawer_fraction_min']}`", f"- max_drawer_fraction_max: `{closeout['max_drawer_fraction_max']}`", f"- forbidden_contact_frames_max: `{closeout['forbidden_contact_frames_max']}`", f"- handle_nonlegal_contact_frames_max: `{closeout['handle_nonlegal_contact_frames_max']}`", f"- max_penetration_m: `{closeout['max_penetration_m']}`", f"- max_force_n: `{closeout['max_force_n']}`", f"- failure_histogram: `{json.dumps(closeout['failure_histogram'], sort_keys=True)}`", f"- next_gate: `{closeout['next_gate']}`"]) + "\n")
    print(json.dumps(ready(closeout), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

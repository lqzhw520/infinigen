#!/usr/bin/env python3
"""Repair contact-mode operational-space policy on the fixed GOC-v4 accessible pool.

This phase is intentionally constrained. It reuses the accepted generated pool
from the physical-accessibility phase and repairs only dynamic execution policy:
arm velocity servo timing, continuous gripper target control, and two-pad
operational-space tracking. It does not modify pool records, geometry, GOC-v4
binding authority, collision policy, thresholds, drawer qpos, or drawer motors.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco
import numpy as np

ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN = ROOT / "experiments/mint/mint_drawer_v1"
SPEC_REL = (
    "experiments/mint/mint_drawer_v1/sovereign/experiment_specs/"
    "v11_g4_goc_v4_contact_mode_operational_space_policy_repair_on_accessible_pool.yaml"
)
SOURCE_LAYER4R_RUN = CAMPAIGN / "runtime/v11_g4_goc_v4_layer4r_on_accessible_instance_pool_20260504T065000Z"
SOURCE_POOL_RUN = CAMPAIGN / "runtime/v11_g4_goc_v4_physical_accessibility_instance_pool_20260504T032604Z"

sys.path.insert(0, str(ROOT / "scripts/mint"))

import v11_g4_layer4r_on_accessible_instance_pool as layer4r  # noqa: E402
from contact_aware_drawer_teacher import (  # noqa: E402
    classify_instance,
    contact_report,
    geom_centers,
    record_step,
    summarize_records,
)

EXPECTED_ACCEPTED_IDS = layer4r.EXPECTED_ACCEPTED_IDS
PERTURBATIONS = layer4r.PERTURBATIONS
MIN_TARGET_FRAMES = layer4r.MIN_TARGET_FRAMES
MIN_CONSECUTIVE_FRAMES = layer4r.MIN_CONSECUTIVE_FRAMES
MAX_PENETRATION_M = layer4r.MAX_PENETRATION_M
MAX_FORCE_N = layer4r.MAX_FORCE_N
TARGETED_SHARD = [
    ("generated_knob_drawer_accessible_001", "nominal"),
    ("generated_knob_drawer_accessible_001", "slow_guarded_contact"),
    ("generated_knob_drawer_accessible_002", "nominal"),
    ("generated_knob_drawer_accessible_002", "slow_guarded_contact"),
    ("generated_knob_drawer_accessible_003", "nominal"),
    ("generated_knob_drawer_accessible_004", "nominal"),
    ("generated_knob_drawer_accessible_005", "nominal"),
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


@dataclass(frozen=True)
class PolicyConfig:
    name: str
    mode: str
    q_gain: float
    q_vel_limit: float
    servo_kp: float
    servo_kd: float
    gripper_mode: str
    pre_mult: float
    guard_mult: float
    contact_mult: float
    hold_mult: float
    extra_hold_steps: int
    op_gain: float
    op_vel_limit: float
    press_m: float
    null_gain: float
    settle_qerr_median: float


POLICY_CYCLES = [
    PolicyConfig(
        name="cycle1_extended_qpos_continuous_fingers",
        mode="qpos",
        q_gain=9.5,
        q_vel_limit=2.35,
        servo_kp=180.0,
        servo_kd=24.0,
        gripper_mode="ik_finger_targets",
        pre_mult=1.8,
        guard_mult=1.8,
        contact_mult=2.0,
        hold_mult=2.2,
        extra_hold_steps=90,
        op_gain=0.0,
        op_vel_limit=0.0,
        press_m=0.0,
        null_gain=0.0,
        settle_qerr_median=0.035,
    ),
    PolicyConfig(
        name="cycle2_two_pad_opspace_light_press",
        mode="two_pad_opspace",
        q_gain=7.5,
        q_vel_limit=2.2,
        servo_kp=210.0,
        servo_kd=30.0,
        gripper_mode="ik_finger_targets",
        pre_mult=1.7,
        guard_mult=2.0,
        contact_mult=2.4,
        hold_mult=2.6,
        extra_hold_steps=120,
        op_gain=8.0,
        op_vel_limit=0.08,
        press_m=0.002,
        null_gain=0.35,
        settle_qerr_median=0.04,
    ),
    PolicyConfig(
        name="cycle3_two_pad_opspace_medium_press_slow",
        mode="two_pad_opspace",
        q_gain=6.0,
        q_vel_limit=1.85,
        servo_kp=230.0,
        servo_kd=36.0,
        gripper_mode="ik_finger_targets",
        pre_mult=2.0,
        guard_mult=2.3,
        contact_mult=2.8,
        hold_mult=3.0,
        extra_hold_steps=150,
        op_gain=6.0,
        op_vel_limit=0.055,
        press_m=0.004,
        null_gain=0.25,
        settle_qerr_median=0.045,
    ),
    PolicyConfig(
        name="cycle4_two_pad_opspace_zero_press_high_track",
        mode="two_pad_opspace",
        q_gain=8.0,
        q_vel_limit=2.45,
        servo_kp=260.0,
        servo_kd=42.0,
        gripper_mode="ik_finger_targets",
        pre_mult=2.2,
        guard_mult=2.5,
        contact_mult=3.0,
        hold_mult=3.2,
        extra_hold_steps=180,
        op_gain=7.0,
        op_vel_limit=0.065,
        press_m=0.0,
        null_gain=0.45,
        settle_qerr_median=0.035,
    ),
]


def stage0_authority(run_dir: Path) -> dict[str, Any]:
    preflight_cmd = [
        "/root/anaconda3/envs/infinigen/bin/python",
        "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py",
        "--spec",
        SPEC_REL,
        "--dry-run",
    ]
    payload = {
        "generated_at_utc": utc_now(),
        "pwd": str(ROOT),
        "branch": run_git(["branch", "--show-current"]),
        "head": run_git(["rev-parse", "HEAD"]),
        "remote_v": run_git(["remote", "-v"]),
        "status_short": run_git(["status", "--short"]),
        "task_spec": SPEC_REL,
        "source_layer4r_run": rel(SOURCE_LAYER4R_RUN),
        "source_pool_run": rel(SOURCE_POOL_RUN),
        "preflight": run_cmd(preflight_cmd),
        "codebase_analysis_skill": {
            "requested": True,
            "invocation_attempted": True,
            "entrypoint": ".claude/skills/scripts python3 -m skills.codebase_analysis.analyze --step 1 --total-steps 4",
            "result": "entrypoint_directory_missing_in_local_playground_checkout; remote repo inspection performed under governed spec",
        },
    }
    payload["harness_preflight_passed"] = payload["preflight"]["returncode"] == 0
    write_json(run_dir / "stage0_authority_and_scope.json", payload)
    (run_dir / "commands.log").write_text(json.dumps(ready(payload), indent=2, sort_keys=True) + "\n")
    return payload


def prior_failure_attribution(run_dir: Path) -> dict[str, Any]:
    cases_path = SOURCE_LAYER4R_RUN / "layer4r_cases.jsonl"
    rows = [json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failure_hist = Counter()
    for row in rows:
        by_candidate[str(row.get("candidate_id"))].append(row)
        if not row.get("passed"):
            failure_hist.update(row.get("failure_reasons") or ["UNKNOWN_FAILURE"])

    trace_summaries: dict[str, Any] = {}
    for row in rows:
        trace = row.get("trace_jsonl")
        if not trace:
            continue
        p = ROOT / trace
        if not p.exists():
            continue
        records = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        first_target = next((r for r in records if int(r.get("contact_counts", {}).get("target", 0)) > 0), None)
        first_forbidden = next((r for r in records if int(r.get("contact_counts", {}).get("forbidden", 0)) > 0), None)
        mode_summary: dict[str, Any] = {}
        for mode in sorted({str(r.get("mode")) for r in records}):
            xs = [r for r in records if str(r.get("mode")) == mode]
            distances = [r.get("distance", {}).get("min_legal_pad_to_handle_m") for r in xs]
            distances = [float(d) for d in distances if d is not None]
            mode_summary[mode] = {
                "steps": len(xs),
                "target_frames": sum(1 for r in xs if int(r.get("contact_counts", {}).get("target", 0)) > 0),
                "forbidden_frames": sum(1 for r in xs if int(r.get("contact_counts", {}).get("forbidden", 0)) > 0),
                "min_pad_handle_distance_m": min(distances) if distances else None,
                "last_pad_handle_distance_m": distances[-1] if distances else None,
            }
        key = "{}::{}".format(row.get("candidate_id"), row.get("perturbation"))
        trace_summaries[key] = {
            "first_target_step": None if first_target is None else first_target.get("step"),
            "first_forbidden_step": None if first_forbidden is None else first_forbidden.get("step"),
            "first_forbidden_pairs": [] if first_forbidden is None else first_forbidden.get("contact_pairs", [])[:4],
            "mode_summary": mode_summary,
        }

    candidate_summary = {}
    for cid, xs in by_candidate.items():
        candidate_summary[cid] = {
            "cases": len(xs),
            "passed": sum(1 for x in xs if x.get("passed")),
            "target_contact_frames_max": max(int(x.get("target_contact_frames", 0)) for x in xs),
            "target_contact_consecutive_max": max(int(x.get("target_contact_max_consecutive_frames", 0)) for x in xs),
            "forbidden_contact_frames_max": max(int(x.get("forbidden_contact_frames", 0)) for x in xs),
            "min_pad_handle_distance_best_m": min(float(x.get("min_pad_handle_distance_m", 999.0)) for x in xs),
        }

    payload = {
        "generated_at_utc": utc_now(),
        "source_layer4r_run": rel(SOURCE_LAYER4R_RUN),
        "source_pool_run": rel(SOURCE_POOL_RUN),
        "rows": len(rows),
        "failure_histogram": dict(sorted(failure_hist.items())),
        "candidate_summary": candidate_summary,
        "trace_summaries": trace_summaries,
        "attribution": {
            "endpoint_accessibility_already_certified": True,
            "pool_binding_empty_spin_ruled_out": True,
            "geometry_or_threshold_change_permitted": False,
            "dominant_dynamic_failure": "insufficient sustained dedicated pad-handle contact under old qpos-only/binary-gripper timing",
            "secondary_dynamic_failure": "right_hand hand_collision reaches handle/drawer before legal pad contact in candidate 002 variants",
        },
    }
    write_json(run_dir / "stage1_prior_failure_attribution.json", payload)
    md = [
        "# Prior Failure Attribution",
        "",
        f"Source Layer4R run: `{rel(SOURCE_LAYER4R_RUN)}`",
        f"Rows: {len(rows)}",
        f"Failure histogram: `{json.dumps(dict(sorted(failure_hist.items())), sort_keys=True)}`",
        "",
        "Interpretation: the fixed accepted pool and GOC-v4 binding were already re-certified; the unresolved blocker is dynamic execution policy on that pool.",
        "The old policy produced legal target contact only briefly on candidate 001 and produced forbidden hand_collision contact on candidate 002. Candidates 003-005 remained contact-short without forbidden contact, which is consistent with actuation/tracking/timing rather than structural accessibility.",
    ]
    write_md(run_dir / "stage1_prior_failure_attribution.md", "\n".join(md))
    return payload


def source_records() -> list[dict[str, Any]]:
    return layer4r.accepted_source_records()


def perturb_by_name(name: str) -> dict[str, Any]:
    for perturb in PERTURBATIONS:
        if perturb["name"] == name:
            return perturb
    raise KeyError(name)


def target_arm_velocity(env: Any, target_arm: np.ndarray, gain: float, limit: float) -> np.ndarray:
    cur = env.data.qpos[env._robot_qpos_addrs].astype(float)
    return np.clip((np.asarray(target_arm, dtype=float) - cur) * float(gain), -float(limit), float(limit))


def stacked_two_pad_velocity(
    env: Any,
    binding: dict[str, Any],
    two_pad_frame: dict[str, Any],
    target_key: str,
    q_ref: np.ndarray,
    config: PolicyConfig,
) -> np.ndarray:
    assignments = two_pad_frame.get("pad_assignment") or []
    if not assignments:
        return target_arm_velocity(env, q_ref, config.q_gain, config.q_vel_limit)
    targets = np.asarray(two_pad_frame[target_key], dtype=float).copy()
    if config.press_m and target_key in {"contact_targets", "hold_targets"}:
        approach = np.asarray(two_pad_frame.get("approach_normal") or [], dtype=float)
        if approach.shape != (3,):
            # Stored frame does not include approach_normal; recover from handle frame.
            approach = np.asarray([0.0, 0.0, 0.0], dtype=float)
        if np.linalg.norm(approach) < 1e-9:
            # contact targets were generated as pregrasp = contact + approach * d.
            pre = np.asarray(two_pad_frame.get("pregrasp_targets", targets), dtype=float)
            delta = np.mean(pre - targets, axis=0)
            approach = delta / max(float(np.linalg.norm(delta)), 1e-9)
        targets = targets - approach[None, :] * float(config.press_m)

    rows = []
    desired = []
    for pad_gid, target_idx in assignments:
        gid = int(pad_gid)
        tidx = int(target_idx)
        err = targets[tidx] - env.data.geom_xpos[gid].astype(float)
        jp = np.zeros((3, env.model.nv), dtype=np.float64)
        jr = np.zeros((3, env.model.nv), dtype=np.float64)
        mujoco.mj_jacGeom(env.model, env.data, jp, jr, gid)
        rows.append(jp[:, env._robot_qvel_addrs])
        desired.append(np.clip(err * float(config.op_gain), -float(config.op_vel_limit), float(config.op_vel_limit)))
    j = np.vstack(rows)
    v = np.concatenate(desired)
    lam = 2e-4
    qdot_task = j.T @ np.linalg.solve(j @ j.T + lam * np.eye(j.shape[0]), v)
    qdot_null = target_arm_velocity(env, q_ref, config.q_gain, config.q_vel_limit)
    qdot = qdot_task + float(config.null_gain) * qdot_null
    return np.clip(qdot, -float(config.q_vel_limit), float(config.q_vel_limit))


def apply_velocity_servo(
    env: Any,
    robot_vel: np.ndarray,
    finger_targets: np.ndarray,
    config: PolicyConfig,
) -> float:
    env._robot_velocity_servo_kp = float(config.servo_kp)
    env._robot_velocity_servo_kd = float(config.servo_kd)
    env._robot_velocity_limit = max(float(env._robot_velocity_limit), float(config.q_vel_limit))
    robot_vel = np.clip(np.asarray(robot_vel, dtype=np.float64), -float(env._robot_velocity_limit), float(env._robot_velocity_limit))
    current_q = env.data.qpos[env._robot_qpos_addrs].astype(np.float64)
    current_qvel = env.data.qvel[env._robot_qvel_addrs].astype(np.float64)
    dt = float(env.model.opt.timestep)
    lo = env.model.jnt_range[env._robot_joint_ids, 0].astype(np.float64)
    hi = env.model.jnt_range[env._robot_joint_ids, 1].astype(np.float64)
    if not hasattr(env, "_robot_qpos_target"):
        env._robot_qpos_target = current_q.copy()
    env._robot_qpos_target = np.clip(env._robot_qpos_target + robot_vel * dt, lo, hi)
    error = env._robot_qpos_target - current_q
    torque = float(config.servo_kp) * error - float(config.servo_kd) * current_qvel
    for local_i, act_id in enumerate(env._robot_actuator_ids):
        ctrl_min = float(env.model.actuator_ctrlrange[act_id, 0])
        ctrl_max = float(env.model.actuator_ctrlrange[act_id, 1])
        env.data.ctrl[act_id] = float(np.clip(torque[local_i], ctrl_min, ctrl_max))
    if len(getattr(env, "_gripper_actuator_ids", [])) == 2:
        ft = np.asarray(finger_targets, dtype=float)
        if ft.shape[0] != 2:
            ft = np.asarray([0.0, 0.0], dtype=float)
        for act_id, target in zip(env._gripper_actuator_ids, ft):
            ctrl_min = float(env.model.actuator_ctrlrange[act_id, 0])
            ctrl_max = float(env.model.actuator_ctrlrange[act_id, 1])
            env.data.ctrl[act_id] = float(np.clip(float(target), ctrl_min, ctrl_max))
    drawer_motor_abs = 0.0
    for act_id in getattr(env, "_drawer_actuator_ids", []):
        env.data.ctrl[act_id] = 0.0
        drawer_motor_abs = max(drawer_motor_abs, abs(float(env.data.ctrl[act_id])))
    mujoco.mj_step(env.model, env.data)
    env._step_count += 1
    return drawer_motor_abs


def waypoint_parts(candidate: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(candidate["ik"][key]["qpos_robot"], dtype=float)
    return q[:7], q[7:9]


def trace_record(
    env: Any,
    binding: dict[str, Any],
    contact: dict[str, Any],
    mode: str,
    finger_targets: np.ndarray,
    q_ref: np.ndarray,
    drawer_motor_abs: float,
) -> dict[str, Any]:
    rec = record_step(env, binding, contact, mode, 1.0)
    qcur = env.data.qpos[env._robot_qpos_addrs].astype(float)
    rec["q_error_norm"] = float(np.linalg.norm(np.asarray(q_ref, dtype=float) - qcur))
    rec["q_error_abs_max"] = float(np.max(np.abs(np.asarray(q_ref, dtype=float) - qcur)))
    rec["finger_targets"] = np.asarray(finger_targets, dtype=float).tolist()
    rec["finger_qpos"] = env.data.qpos[getattr(env, "_gripper_qpos_addrs", [])].astype(float).tolist()
    rec["drawer_motor_command_abs"] = float(drawer_motor_abs)
    return rec


def compact_trace_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": record.get("step"),
        "mode": record.get("mode"),
        "distance": record.get("distance"),
        "contact_counts": record.get("contact_counts"),
        "contact_pairs": record.get("contact_pairs", [])[:8],
        "drawer_qpos": record.get("drawer_qpos"),
        "drawer_fraction": record.get("drawer_fraction"),
        "q_error_norm": record.get("q_error_norm"),
        "q_error_abs_max": record.get("q_error_abs_max"),
        "finger_targets": record.get("finger_targets"),
        "finger_qpos": record.get("finger_qpos"),
        "drawer_motor_command_abs": record.get("drawer_motor_command_abs"),
    }


def run_segment(
    env: Any,
    binding: dict[str, Any],
    candidate: dict[str, Any],
    config: PolicyConfig,
    mode: str,
    waypoint_key: str,
    target_key: str | None,
    steps: int,
    records: list[dict[str, Any]],
    trace_path: Path,
    prev_centers: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    q_ref, finger_targets = waypoint_parts(candidate, waypoint_key)
    two_pad_frame = candidate.get("two_pad_frame", {})
    for _ in range(int(steps)):
        if config.mode == "two_pad_opspace" and target_key is not None:
            robot_vel = stacked_two_pad_velocity(env, binding, two_pad_frame, target_key, q_ref, config)
        else:
            robot_vel = target_arm_velocity(env, q_ref, config.q_gain, config.q_vel_limit)
        drawer_motor_abs = apply_velocity_servo(env, robot_vel, finger_targets, config)
        report = contact_report(env, binding, prev_centers)
        rec = trace_record(env, binding, report, mode, finger_targets, q_ref, drawer_motor_abs)
        records.append(rec)
        append_jsonl(trace_path, compact_trace_record(rec))
        prev_centers = report["centers"]
    return prev_centers


def case_failure_reasons(summary: dict[str, Any], reset_counts: dict[str, Any]) -> list[str]:
    reasons = layer4r.case_failure_reasons(summary, reset_counts)
    if int(summary.get("two_pad_target_contact_frames", 0)) <= 0:
        # Non-gating diagnostic only; the Layer4R public thresholds remain unchanged.
        summary["diagnostic_two_pad_contact_absent"] = True
    return reasons


def run_case(candidate: dict[str, Any], perturb: dict[str, Any], config: PolicyConfig, run_dir: Path, cycle_idx: int) -> dict[str, Any]:
    cid = str(candidate["candidate_id"])
    pname = str(perturb["name"])
    q_reset = np.asarray(candidate["reset"]["qpos_arm"], dtype=float) + np.asarray(perturb["qpos_delta"], dtype=float)
    env = None
    records: list[dict[str, Any]] = []
    trace_path = run_dir / "traces" / f"cycle_{cycle_idx}_{config.name}__{cid}__{pname}.jsonl"
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            env = layer4r.make_candidate_env(candidate, robot_init_qpos=q_reset, max_steps=1200)
            env.reset()
        binding = classify_instance(env)
        binding_c = layer4r.compact_binding(binding)
        reset_contact = contact_report(env, binding, None)
        prev_centers = geom_centers(env)

        pre_steps = int(math.ceil(int(perturb["pre_steps"]) * config.pre_mult))
        guard_steps = int(math.ceil(int(perturb["guard_steps"]) * config.guard_mult))
        contact_steps = int(math.ceil(int(perturb["contact_steps"]) * config.contact_mult))
        hold_steps = int(math.ceil(int(perturb["hold_steps"]) * config.hold_mult)) + int(config.extra_hold_steps)

        prev_centers = run_segment(env, binding, candidate, config, "free_space_pregrasp", "pregrasp", "pregrasp_targets", pre_steps, records, trace_path, prev_centers)
        prev_centers = run_segment(env, binding, candidate, config, "guarded_approach", "guarded", "guarded_targets", guard_steps, records, trace_path, prev_centers)
        prev_centers = run_segment(env, binding, candidate, config, "contact_seat", "contact_hold", "contact_targets", contact_steps, records, trace_path, prev_centers)
        prev_centers = run_segment(env, binding, candidate, config, "contact_hold", "contact_hold", "hold_targets", hold_steps, records, trace_path, prev_centers)

        summary = summarize_records(records)
        handle_nonlegal_frames = sum(1 for r in records if int(r["contact_counts"].get("handle_nonlegal", 0)) > 0)
        target_both = sum(1 for r in records if int(r["contact_counts"].get("target", 0)) >= 2)
        qerrs = [float(r.get("q_error_norm", 0.0)) for r in records]
        drawer_motor_abs_max = max((float(r.get("drawer_motor_command_abs", 0.0)) for r in records), default=0.0)
        first_target = next((r for r in records if int(r["contact_counts"].get("target", 0)) > 0), None)
        first_forbidden = next((r for r in records if int(r["contact_counts"].get("forbidden", 0)) > 0), None)
        summary.update(
            {
                "candidate_id": cid,
                "perturbation": pname,
                "policy_cycle": cycle_idx,
                "policy_name": config.name,
                "policy_mode": config.mode,
                "binding": binding_c,
                "reset_counts": reset_contact["counts"],
                "handle_nonlegal_contact_frames": handle_nonlegal_frames,
                "two_pad_target_contact_frames": target_both,
                "direct_qpos_drawer_opening": False,
                "drawer_motor_command_used": drawer_motor_abs_max > 1e-12,
                "drawer_motor_command_abs_max": drawer_motor_abs_max,
                "trace_jsonl": rel(trace_path),
                "first_target_step": None if first_target is None else first_target.get("step"),
                "first_forbidden_step": None if first_forbidden is None else first_forbidden.get("step"),
                "first_forbidden_pairs": [] if first_forbidden is None else first_forbidden.get("contact_pairs", [])[:4],
                "min_pad_handle_distance_m": min((r["distance"]["min_legal_pad_to_handle_m"] for r in records), default=None),
                "q_error_norm_median": float(np.median(qerrs)) if qerrs else None,
                "q_error_norm_final": qerrs[-1] if qerrs else None,
                "segment_steps": {
                    "free_space_pregrasp": pre_steps,
                    "guarded_approach": guard_steps,
                    "contact_seat": contact_steps,
                    "contact_hold": hold_steps,
                },
            }
        )
        reasons = case_failure_reasons(summary, reset_contact["counts"])
        summary["passed"] = not reasons
        summary["failure_reasons"] = reasons
        return summary
    except Exception as exc:  # noqa: BLE001 - evidence needs exact failure
        return {
            "candidate_id": cid,
            "perturbation": pname,
            "policy_cycle": cycle_idx,
            "policy_name": config.name,
            "passed": False,
            "failure_reasons": ["CASE_EXECUTION_EXCEPTION"],
            "error": repr(exc),
            "trace_jsonl": rel(trace_path),
        }
    finally:
        if env is not None:
            env.close()


def summarize_results(results: list[dict[str, Any]], config: PolicyConfig, cycle_idx: int, targeted: bool) -> dict[str, Any]:
    failure_hist = Counter()
    for row in results:
        if not row.get("passed"):
            failure_hist.update(row.get("failure_reasons") or ["UNKNOWN_FAILURE"])
    passed = sum(1 for row in results if row.get("passed"))
    target_frames = [int(row.get("target_contact_frames", 0)) for row in results]
    target_consec = [int(row.get("target_contact_max_consecutive_frames", 0)) for row in results]
    forbidden = [int(row.get("forbidden_contact_frames", 0)) for row in results]
    handle_nonlegal = [int(row.get("handle_nonlegal_contact_frames", 0)) for row in results]
    penetrations = [float(row.get("max_penetration_m", 0.0)) for row in results]
    forces = [float(row.get("max_force_n", 0.0)) for row in results if math.isfinite(float(row.get("max_force_n", 0.0)))]
    qerr = [float(row.get("q_error_norm_final", 0.0)) for row in results if row.get("q_error_norm_final") is not None]
    return {
        "generated_at_utc": utc_now(),
        "policy_cycle": cycle_idx,
        "policy_name": config.name,
        "policy_mode": config.mode,
        "targeted_shard": targeted,
        "cases_total": len(results),
        "cases_passed": passed,
        "cases_failed": len(results) - passed,
        "failure_histogram": dict(sorted(failure_hist.items())),
        "target_contact_frames_min": min(target_frames) if target_frames else 0,
        "target_contact_frames_max": max(target_frames) if target_frames else 0,
        "target_contact_consecutive_min": min(target_consec) if target_consec else 0,
        "target_contact_consecutive_max": max(target_consec) if target_consec else 0,
        "forbidden_contact_frames_max": max(forbidden) if forbidden else 0,
        "handle_nonlegal_contact_frames_max": max(handle_nonlegal) if handle_nonlegal else 0,
        "max_penetration_m": max(penetrations) if penetrations else 0.0,
        "max_force_n": max(forces) if forces else 0.0,
        "q_error_norm_final_max": max(qerr) if qerr else None,
        "matrix_passed": passed == len(results) and len(results) > 0,
    }


def run_cases(
    run_dir: Path,
    records: list[dict[str, Any]],
    config: PolicyConfig,
    cycle_idx: int,
    targeted: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {str(r["candidate_id"]): r for r in records}
    todo: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if targeted:
        for cid, pname in TARGETED_SHARD:
            todo.append((by_id[cid], perturb_by_name(pname)))
    else:
        for rec in records:
            for perturb in PERTURBATIONS:
                todo.append((rec, perturb))
    results = []
    stem = "targeted" if targeted else "full_layer4r"
    out = run_dir / f"cycle_{cycle_idx}_{stem}_results.jsonl"
    if out.exists():
        out.unlink()
    for candidate, perturb in todo:
        row = run_case(candidate, perturb, config, run_dir, cycle_idx)
        append_jsonl(out, row)
        results.append(row)
    summary = summarize_results(results, config, cycle_idx, targeted)
    write_json(run_dir / f"cycle_{cycle_idx}_{stem}_summary.json", summary)
    return results, summary


def write_policy_design(run_dir: Path) -> None:
    payload = {
        "generated_at_utc": utc_now(),
        "repair_target": "dynamic contact-mode policy only on fixed accepted pool",
        "fixed_pool_ids": EXPECTED_ACCEPTED_IDS,
        "thresholds_preserved": {
            "target_contact_total_frames_min": MIN_TARGET_FRAMES,
            "target_contact_max_consecutive_frames_min": MIN_CONSECUTIVE_FRAMES,
            "forbidden_contact_frames": 0,
            "handle_nonlegal_contact_frames": 0,
            "max_penetration_m": MAX_PENETRATION_M,
            "max_force_n": MAX_FORCE_N,
        },
        "controls": {
            "pool_modified": False,
            "geometry_modified": False,
            "goc_authority_modified": False,
            "collision_demotion_used": False,
            "direct_drawer_qpos_allowed": False,
            "drawer_motor_allowed": False,
        },
        "policy_cycles": [config.__dict__ for config in POLICY_CYCLES],
    }
    write_json(run_dir / "stage2_policy_repair_design.json", payload)


def final_checks(run_dir: Path) -> dict[str, Any]:
    preflight_cmd = [
        "/root/anaconda3/envs/infinigen/bin/python",
        "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py",
        "--spec",
        SPEC_REL,
        "--dry-run",
    ]
    compile_cmd = [
        "/root/anaconda3/envs/infinigen/bin/python",
        "-m",
        "py_compile",
        "scripts/mint/v11_g4_contact_mode_policy_repair_on_accessible_pool.py",
    ]
    json_errors = []
    for p in list(run_dir.rglob("*.json")):
        try:
            json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            json_errors.append({"path": rel(p), "error": repr(exc)})
    for p in list(run_dir.rglob("*.jsonl")):
        try:
            with p.open() as fh:
                for idx, line in enumerate(fh, start=1):
                    if line.strip():
                        json.loads(line)
        except Exception as exc:  # noqa: BLE001
            json_errors.append({"path": rel(p), "line": idx if "idx" in locals() else None, "error": repr(exc)})
    status = run_git(["status", "--short"])
    forbidden_modified = [line for line in status.splitlines() if "sovereign/current_truth.json" in line or "sovereign/next_actions.json" in line or "external/MINT/" in line]
    payload = {
        "generated_at_utc": utc_now(),
        "preflight": run_cmd(preflight_cmd),
        "py_compile": run_cmd(compile_cmd),
        "git_status_short": status,
        "json_errors": json_errors,
        "forbidden_modified": forbidden_modified,
        "current_truth_modified": any("sovereign/current_truth.json" in line for line in status.splitlines()),
        "next_actions_modified": any("sovereign/next_actions.json" in line for line in status.splitlines()),
        "external_mint_modified": any("external/MINT/" in line for line in status.splitlines()),
    }
    payload["preflight_passed"] = payload["preflight"]["returncode"] == 0
    payload["py_compile_passed"] = payload["py_compile"]["returncode"] == 0
    payload["passed"] = bool(payload["preflight_passed"] and payload["py_compile_passed"] and not json_errors and not forbidden_modified)
    write_json(run_dir / "stage9_final_checks.json", payload)
    return payload


def write_proposed_deltas(closeout: dict[str, Any]) -> None:
    truth_delta = {
        "proposal_generated_at_utc": utc_now(),
        "task_id": "V11_G4_GOC_V4_CONTACT_MODE_OPERATIONAL_SPACE_POLICY_REPAIR_ON_ACCESSIBLE_POOL_V1",
        "do_not_apply_automatically": True,
        "current_truth_modified": False,
        "fixed_accepted_pool": {
            "pool_modified": False,
            "accepted_instance_ids": EXPECTED_ACCEPTED_IDS,
        },
        "contact_policy_repair": closeout,
    }
    next_actions = {
        "proposal_generated_at_utc": utc_now(),
        "task_id": "V11_G4_GOC_V4_CONTACT_MODE_OPERATIONAL_SPACE_POLICY_REPAIR_ON_ACCESSIBLE_POOL_V1",
        "do_not_apply_automatically": True,
        "next_gate": closeout.get("next_gate"),
        "teacher_rollout_allowed": closeout.get("closeout_classification") == "CONTACT_MODE_OPERATIONAL_SPACE_POLICY_REPAIR_LAYER4R_PASSED",
        "mint_training_allowed": False,
    }
    write_json(CAMPAIGN / "sovereign/proposed_current_truth_delta_goc_v4_contact_mode_policy_repair_on_accessible_pool.json", truth_delta)
    write_json(CAMPAIGN / "sovereign/proposed_next_actions_goc_v4_contact_mode_policy_repair_on_accessible_pool.json", next_actions)


def closeout_classification(stage0: dict[str, Any], best_targeted: dict[str, Any] | None, best_full: dict[str, Any] | None) -> tuple[str, str]:
    if not stage0.get("harness_preflight_passed"):
        return "HARNESS_PREFLIGHT_FAILED", "HARNESS_PREFLIGHT_REPAIR"
    if best_full and best_full.get("matrix_passed"):
        return "CONTACT_MODE_OPERATIONAL_SPACE_POLICY_REPAIR_LAYER4R_PASSED", "GOC_V4_BOUNDED_TEACHER_PULL_ROLLOUT_ON_ACCESSIBLE_POOL"
    hist = dict((best_full or best_targeted or {}).get("failure_histogram", {}))
    if hist.get("forbidden_contact_present", 0) or hist.get("handle_nonlegal_contact_present", 0):
        return "FORBIDDEN_KEEPOUT_RECOVERY_FAILED", "CONTACT_MODE_KEEP_OUT_POLICY_REPAIR"
    if hist.get("target_contact_frames_lt_50", 0) or hist.get("target_contact_consecutive_lt_30", 0):
        return "TARGET_CONTACT_TRACKING_FAILED", "OPERATIONAL_SPACE_TRACKING_OR_ACTUATION_REPAIR"
    if hist:
        return "CONTACT_POLICY_REPAIR_STALLED", "CONTROLLER_DYNAMICS_MODEL_REPAIR_WITH_FIXED_POOL_EVIDENCE"
    return "EXECUTION_FAILED", "CONTACT_POLICY_REPAIR_DEBUG"


def write_closeout(
    run_dir: Path,
    stage0: dict[str, Any],
    cycles: list[dict[str, Any]],
    best_targeted: dict[str, Any] | None,
    best_full: dict[str, Any] | None,
) -> dict[str, Any]:
    classification, next_gate = closeout_classification(stage0, best_targeted, best_full)
    closeout = {
        "generated_at_utc": utc_now(),
        "task_id": "V11_G4_GOC_V4_CONTACT_MODE_OPERATIONAL_SPACE_POLICY_REPAIR_ON_ACCESSIBLE_POOL_V1",
        "closeout_classification": classification,
        "harness_preflight_passed": bool(stage0.get("harness_preflight_passed")),
        "task_spec_lock_bound": bool(stage0.get("harness_preflight_passed")),
        "accepted_pool_bound": True,
        "accepted_instance_ids": EXPECTED_ACCEPTED_IDS,
        "pool_modified": False,
        "geometry_modified": False,
        "goc_v4_authority_modified": False,
        "thresholds_modified": False,
        "fake_collision_demotion_used": False,
        "direct_qpos_drawer_opening": False,
        "drawer_motor_command_used": False,
        "repair_cycles_run": len(cycles),
        "cycle_summaries": cycles,
        "targeted_shard_best": best_targeted or {},
        "full_layer4r_best": best_full or {},
        "full_layer4r_matrix_attempted": best_full is not None,
        "layer4r_cases_total": 0 if best_full is None else best_full.get("cases_total", 0),
        "layer4r_cases_passed": 0 if best_full is None else best_full.get("cases_passed", 0),
        "layer4r_cases_failed": 0 if best_full is None else best_full.get("cases_failed", 0),
        "failure_histogram": dict((best_full or best_targeted or {}).get("failure_histogram", {})),
        "target_contact_frames_min": None if best_full is None else best_full.get("target_contact_frames_min"),
        "target_contact_frames_max": None if best_full is None else best_full.get("target_contact_frames_max"),
        "target_contact_consecutive_min": None if best_full is None else best_full.get("target_contact_consecutive_min"),
        "target_contact_consecutive_max": None if best_full is None else best_full.get("target_contact_consecutive_max"),
        "forbidden_contact_frames_max": None if best_full is None else best_full.get("forbidden_contact_frames_max"),
        "handle_nonlegal_contact_frames_max": None if best_full is None else best_full.get("handle_nonlegal_contact_frames_max"),
        "max_penetration_m": None if best_full is None else best_full.get("max_penetration_m"),
        "max_force_n": None if best_full is None else best_full.get("max_force_n"),
        "current_truth_modified": False,
        "next_actions_modified": False,
        "runtime_patch_applied": True,
        "runtime_patch_files": ["scripts/mint/v11_g4_contact_mode_policy_repair_on_accessible_pool.py"],
        "next_gate": next_gate,
    }
    write_json(run_dir / "closeout_decision.json", closeout)
    write_proposed_deltas(closeout)
    report = [
        "# V11 G4 GOC-v4 Contact-Mode Policy Repair On Accessible Pool",
        "",
        f"closeout_classification: {classification}",
        "accepted_pool_bound: {}".format(closeout.get("accepted_pool_bound")),
        f"repair_cycles_run: {len(cycles)}",
        "full_layer4r_cases: {}/{} passed".format(closeout.get("layer4r_cases_passed"), closeout.get("layer4r_cases_total")),
        "failure_histogram: {}".format(json.dumps(closeout.get("failure_histogram", {}), sort_keys=True)),
        f"next_gate: {next_gate}",
        "",
        "Controls preserved: fixed accepted pool, GOC-v4 per-instance exact pad/handle authority, geometry/collision policy, and Layer4R thresholds were not modified.",
    ]
    write_md(run_dir / "final_report.md", "\n".join(report))
    return closeout


def run_phase(run_dir: Path, max_cycles: int | None = None) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    stage0 = stage0_authority(run_dir)
    if not stage0.get("harness_preflight_passed"):
        return write_closeout(run_dir, stage0, [], None, None)
    prior_failure_attribution(run_dir)
    write_policy_design(run_dir)
    records = source_records()
    cycles: list[dict[str, Any]] = []
    best_targeted: dict[str, Any] | None = None
    best_full: dict[str, Any] | None = None
    limit = max_cycles or len(POLICY_CYCLES)
    for cycle_idx, config in enumerate(POLICY_CYCLES[:limit], start=1):
        _, targeted_summary = run_cases(run_dir, records, config, cycle_idx, targeted=True)
        cycles.append({"targeted": targeted_summary})
        if best_targeted is None or targeted_summary.get("cases_passed", 0) > best_targeted.get("cases_passed", 0):
            best_targeted = targeted_summary
        if not targeted_summary.get("matrix_passed"):
            continue
        _, full_summary = run_cases(run_dir, records, config, cycle_idx, targeted=False)
        cycles[-1]["full_layer4r"] = full_summary
        best_full = full_summary
        if full_summary.get("matrix_passed"):
            break
    closeout = write_closeout(run_dir, stage0, cycles, best_targeted, best_full)
    final_checks(run_dir)
    return closeout


def post_push_verification(run_dir: Path) -> dict[str, Any]:
    head = run_git(["rev-parse", "HEAD"])
    remote = run_git(["ls-remote", "my-origin", "feature/mint-env-reformulation-v1-visual-fidelity"])
    required = [
        SPEC_REL,
        "scripts/mint/v11_g4_contact_mode_policy_repair_on_accessible_pool.py",
        rel(run_dir / "closeout_decision.json"),
        rel(run_dir / "final_report.md"),
        "experiments/mint/mint_drawer_v1/sovereign/proposed_current_truth_delta_goc_v4_contact_mode_policy_repair_on_accessible_pool.json",
        "experiments/mint/mint_drawer_v1/sovereign/proposed_next_actions_goc_v4_contact_mode_policy_repair_on_accessible_pool.json",
    ]
    origin_head = remote.split()[0] if remote.strip() else ""
    visibility = {}
    for path in required:
        if not path:
            continue
        proc = subprocess.run(["git", "ls-tree", origin_head, path], cwd=ROOT, text=True, capture_output=True)
        visibility[path] = bool(proc.stdout.strip())
    payload = {
        "generated_at_utc": utc_now(),
        "local_head": head,
        "remote_head": origin_head,
        "remote_head_matches_local": bool(origin_head == head),
        "required_paths_origin_visible": visibility,
        "all_required_paths_origin_visible": all(visibility.values()) if visibility else False,
    }
    write_json(run_dir / "post_push_verification.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--post-push-verify", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir or (CAMPAIGN / "runtime" / f"v11_g4_goc_v4_contact_mode_operational_space_policy_repair_on_accessible_pool_{utc_stamp()}")
    if args.post_push_verify:
        post_push_verification(run_dir)
        return 0
    closeout = run_phase(run_dir, args.max_cycles)
    print(json.dumps(ready({"run_dir": rel(run_dir), **closeout}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

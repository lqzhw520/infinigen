#!/usr/bin/env python3
"""Certify Layer4R on the committed GOC-v4 physically accessible instance pool.

This phase is deliberately narrow: it does not generate instances, repair the
pool, run teacher pull rollout, export bundles, render, or train. The runner
first performs a readiness smoke that re-binds the accepted pool and rejects
legacy/fallback/empty execution. Only then does it run a strict Layer4R contact
matrix on the accepted generated instances.
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
from collections import Counter
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
    "v11_g4_goc_v4_layer4r_on_accessible_instance_pool.yaml"
)
SOURCE_POOL_RUN = CAMPAIGN / "runtime/v11_g4_goc_v4_physical_accessibility_instance_pool_20260504T032604Z"
EXPECTED_ACCEPTED_IDS = [
    "generated_knob_drawer_accessible_001",
    "generated_knob_drawer_accessible_002",
    "generated_knob_drawer_accessible_003",
    "generated_knob_drawer_accessible_004",
    "generated_knob_drawer_accessible_005",
]
MIN_TARGET_FRAMES = 50
MIN_CONSECUTIVE_FRAMES = 30
MAX_PENETRATION_M = 0.02
MAX_FORCE_N = 1_000_000.0

sys.path.insert(0, str(ROOT / "scripts/mint"))

import v11_g4_physical_accessibility_instance_pool as pool  # noqa: E402
from contact_aware_drawer_teacher import (  # noqa: E402
    DrawerRobotEnvMuJoCoLibero,
    classify_instance,
    contact_report,
    geom_centers,
    record_step,
    summarize_records,
)


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
        "source_pool_run": rel(SOURCE_POOL_RUN),
        "preflight": run_cmd(preflight_cmd),
    }
    payload["harness_preflight_passed"] = payload["preflight"]["returncode"] == 0
    write_json(run_dir / "stage0_authority_and_scope.json", payload)
    (run_dir / "commands.log").write_text(json.dumps(ready(payload), indent=2, sort_keys=True) + "\n")
    return payload


def accepted_source_records() -> list[dict[str, Any]]:
    records = load_json(SOURCE_POOL_RUN / "cycle_1_accepted_instances.json")
    if not isinstance(records, list):
        raise RuntimeError("cycle_1_accepted_instances.json is not a list")
    by_id = {r.get("candidate_id"): r for r in records}
    missing = [cid for cid in EXPECTED_ACCEPTED_IDS if cid not in by_id]
    if missing:
        raise RuntimeError(f"accepted source missing expected ids: {missing}")
    return [by_id[cid] for cid in EXPECTED_ACCEPTED_IDS]


def compact_binding(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "legal_finger_pad_geom_ids": binding.get("legal_finger_pad_geom_ids", binding.get("legal_gripper_surface_geom_ids", [])),
        "legal_gripper_surface_geom_ids": binding.get("legal_gripper_surface_geom_ids", binding.get("legal_finger_pad_geom_ids", [])),
        "drawer_handle_geom_ids": binding.get("drawer_handle_geom_ids", []),
        "forbidden_robot_surface_geom_ids": binding.get("forbidden_robot_surface_geom_ids", []),
        "drawer_body_or_cabinet_geom_ids": binding.get("drawer_body_or_cabinet_geom_ids", []),
        "visual_only_or_noncontact_geom_ids": binding.get("visual_only_or_noncontact_geom_ids", []),
        "unknown_contact_relevant_geom_ids": binding.get("unknown_contact_relevant_geom_ids", []),
        "goc_v3_broad_link_geoms_demoted_from_target": binding.get("goc_v3_broad_link_geoms_demoted_from_target", []),
        "binding_generated": bool(binding.get("binding_generated")),
    }


def make_candidate_env(candidate: dict[str, Any], robot_init_qpos: list[float] | np.ndarray | None = None, max_steps: int = 260):
    variant = dict(candidate["model_builder_parameters"])

    class CandidateBuilder(pool.GeneratedAccessibleDrawerBuilder):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, variant=variant, **kwargs)

    DrawerRobotEnvMuJoCoLibero._MERGED_BUILDER_CLASS = CandidateBuilder
    init_qpos = None if robot_init_qpos is None else np.asarray(robot_init_qpos, dtype=float).tolist()
    return DrawerRobotEnvMuJoCoLibero(
        seed=int(candidate.get("synthetic_seed", 9001)),
        image_size=64,
        max_steps=int(max_steps),
        contract=None,
        robot_init_qpos=init_qpos or candidate.get("reset", {}).get("qpos_arm") or pool.SAFE_PRECONTACT_QPOS.astype(float).tolist(),
    )


def source_closeout_ok() -> dict[str, Any]:
    closeout = load_json(SOURCE_POOL_RUN / "closeout_decision.json")
    manifest = load_json(SOURCE_POOL_RUN / "accepted_instance_pool_manifest.json")
    return {
        "source_pool_run": rel(SOURCE_POOL_RUN),
        "source_closeout_classification": closeout.get("closeout_classification"),
        "source_accepted_accessible_instance_count": closeout.get("accepted_accessible_instance_count"),
        "source_accepted_instance_ids": closeout.get("accepted_instance_ids"),
        "manifest_accepted_instance_count": manifest.get("accepted_instance_count"),
        "manifest_accepted_instance_ids": manifest.get("accepted_instance_ids"),
        "source_ok": bool(
            closeout.get("closeout_classification") == "ACCESSIBLE_INSTANCE_POOL_READY_FOR_LAYER4R"
            and closeout.get("accepted_accessible_instance_count") >= 5
            and manifest.get("accepted_instance_ids") == EXPECTED_ACCEPTED_IDS
        ),
    }


def readiness_smoke(run_dir: Path, source_records: list[dict[str, Any]]) -> dict[str, Any]:
    smoke_dir = run_dir / "readiness_recertification"
    source_status = source_closeout_ok()
    recertified: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    generated_ids: list[str] = []
    for record in source_records:
        cid = str(record["candidate_id"])
        generated_ids.append(cid)
        if record.get("source_type") != "generated_accessible_variant":
            failures.append({"candidate_id": cid, "reason": "accepted_pool_contains_non_generated_source", "source_type": record.get("source_type")})
            continue
        try:
            result = pool.evaluate_generated_candidate(record, smoke_dir)
        except Exception as exc:  # noqa: BLE001 - evidence needs exact failure
            result = {"candidate_id": cid, "accepted": False, "rejection_reason": "READINESS_RECERT_EXCEPTION", "error": repr(exc)}
        binding = compact_binding(result.get("binding", {}))
        legal = list(map(int, binding.get("legal_finger_pad_geom_ids", [])))
        handle = list(map(int, binding.get("drawer_handle_geom_ids", [])))
        unknown = list(map(int, binding.get("unknown_contact_relevant_geom_ids", [])))
        reasons: list[str] = []
        if not result.get("accepted"):
            reasons.append(str(result.get("rejection_reason") or "recertification_not_accepted"))
        if not legal:
            reasons.append("missing_dedicated_pad_binding")
        if not handle:
            reasons.append("missing_handle_binding")
        if unknown:
            reasons.append("unknown_contact_relevant_geoms_present")
        if sorted(legal) == [63, 81, 90]:
            reasons.append("legacy_goc_v3_ids_bound_as_target")
        if result.get("direct_qpos_drawer_opening"):
            reasons.append("direct_qpos_drawer_opening_true")
        if result.get("drawer_motor_command_used"):
            reasons.append("drawer_motor_command_used_true")
        if not result.get("approach_corridor", {}).get("passed"):
            reasons.append("approach_corridor_not_passed")
        if not result.get("pull_corridor", {}).get("passed"):
            reasons.append("pull_corridor_not_passed")
        if not result.get("visual_physical_consistency", {}).get("passed"):
            reasons.append("visual_physical_consistency_not_passed")
        summary = {
            "candidate_id": cid,
            "accepted": bool(result.get("accepted")),
            "source_type": record.get("source_type"),
            "binding": binding,
            "nq": result.get("nq"),
            "nu": result.get("nu"),
            "max_two_pad_residual_m": result.get("max_two_pad_residual_m"),
            "approach_corridor_passed": bool(result.get("approach_corridor", {}).get("passed")),
            "pull_corridor_passed": bool(result.get("pull_corridor", {}).get("passed")),
            "visual_physical_consistency_passed": bool(result.get("visual_physical_consistency", {}).get("passed")),
            "fake_collision_demotion_used": False,
            "reasons": reasons,
            "model_manifest": result.get("model_manifest"),
            "reset": result.get("reset"),
            "ik": result.get("ik"),
        }
        recertified.append(summary)
        if reasons:
            failures.append({"candidate_id": cid, "reasons": reasons})
    accepted_ids = [r["candidate_id"] for r in recertified if not r["reasons"]]
    smoke_passed = bool(source_status["source_ok"] and accepted_ids == EXPECTED_ACCEPTED_IDS and not failures)
    payload = {
        "generated_at_utc": utc_now(),
        "source_status": source_status,
        "expected_accepted_ids": EXPECTED_ACCEPTED_IDS,
        "recertified": recertified,
        "readiness_failures": failures,
        "accepted_pool_bound": smoke_passed,
        "readiness_smoke_passed": smoke_passed,
        "empty_matrix_prevented": not bool(recertified),
        "legacy_pool_fallback_used": False,
    }
    write_json(run_dir / "readiness_smoke_report.json", payload)
    return payload


PERTURBATIONS = [
    {"name": "nominal", "qpos_delta": [0.0] * 7, "pre_steps": 55, "guard_steps": 45, "contact_steps": 55, "hold_steps": 80, "vel_gain": 7.0, "vel_limit": 1.4, "close_delay": 15},
    {"name": "qpos_small_plus", "qpos_delta": [0.002, -0.002, 0.001, 0.002, -0.001, 0.001, -0.002], "pre_steps": 55, "guard_steps": 45, "contact_steps": 55, "hold_steps": 80, "vel_gain": 7.0, "vel_limit": 1.4, "close_delay": 15},
    {"name": "qpos_small_minus", "qpos_delta": [-0.002, 0.002, -0.001, -0.002, 0.001, -0.001, 0.002], "pre_steps": 55, "guard_steps": 45, "contact_steps": 55, "hold_steps": 80, "vel_gain": 7.0, "vel_limit": 1.4, "close_delay": 15},
    {"name": "slow_guarded_contact", "qpos_delta": [0.0] * 7, "pre_steps": 75, "guard_steps": 70, "contact_steps": 80, "hold_steps": 90, "vel_gain": 5.5, "vel_limit": 1.0, "close_delay": 20},
    {"name": "fast_guarded_contact", "qpos_delta": [0.0] * 7, "pre_steps": 40, "guard_steps": 32, "contact_steps": 42, "hold_steps": 70, "vel_gain": 8.5, "vel_limit": 1.8, "close_delay": 10},
    {"name": "late_close", "qpos_delta": [0.0] * 7, "pre_steps": 55, "guard_steps": 45, "contact_steps": 70, "hold_steps": 95, "vel_gain": 7.0, "vel_limit": 1.4, "close_delay": 38},
]


def target_action(env: DrawerRobotEnvMuJoCoLibero, target_arm: np.ndarray, gripper_close: bool, gain: float, limit: float) -> np.ndarray:
    cur = env.data.qpos[env._robot_qpos_addrs].astype(float)
    vel = np.clip((np.asarray(target_arm, dtype=float) - cur) * float(gain), -float(limit), float(limit))
    action = np.zeros(9, dtype=np.float32)
    action[:7] = vel.astype(np.float32)
    action[7] = 1.0 if gripper_close else -1.0
    action[8] = 0.0
    return action


def max_consecutive_bool(values: list[bool]) -> int:
    best = 0
    cur = 0
    for value in values:
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def compact_trace_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": record.get("step"),
        "mode": record.get("mode"),
        "distance": record.get("distance"),
        "contact_counts": record.get("contact_counts"),
        "gripper_command": record.get("gripper_command"),
        "drawer_qpos": record.get("drawer_qpos"),
        "drawer_fraction": record.get("drawer_fraction"),
        "contact_pairs": record.get("contact_pairs", [])[:6],
    }


def case_failure_reasons(summary: dict[str, Any], reset_counts: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if int(reset_counts.get("forbidden", 0)) != 0:
        reasons.append("reset_forbidden_contact_present")
    if float(reset_counts.get("max_penetration_m", 0.0)) > MAX_PENETRATION_M:
        reasons.append("reset_penetration_gt_0p02m")
    if int(summary.get("target_contact_frames", 0)) < MIN_TARGET_FRAMES:
        reasons.append("target_contact_frames_lt_50")
    if int(summary.get("target_contact_max_consecutive_frames", 0)) < MIN_CONSECUTIVE_FRAMES:
        reasons.append("target_contact_consecutive_lt_30")
    if int(summary.get("forbidden_contact_frames", 0)) != 0:
        reasons.append("forbidden_contact_present")
    if int(summary.get("handle_nonlegal_contact_frames", 0)) != 0:
        reasons.append("handle_nonlegal_contact_present")
    if float(summary.get("max_penetration_m", 0.0)) > MAX_PENETRATION_M:
        reasons.append("max_penetration_gt_0p02m")
    max_force = float(summary.get("max_force_n", 0.0))
    if (not math.isfinite(max_force)) or max_force > MAX_FORCE_N:
        reasons.append("max_force_nonfinite_or_gt_1e6n")
    if bool(summary.get("direct_qpos_drawer_opening", False)):
        reasons.append("direct_qpos_drawer_opening_true")
    if bool(summary.get("drawer_motor_command_used", False)):
        reasons.append("drawer_motor_command_used_true")
    return reasons


def run_case(candidate: dict[str, Any], perturb: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    cid = str(candidate["candidate_id"])
    q_reset = np.asarray(candidate["reset"]["qpos_arm"], dtype=float) + np.asarray(perturb["qpos_delta"], dtype=float)
    q_pre = np.asarray(candidate["ik"]["pregrasp"]["qpos_robot"], dtype=float)[:7]
    q_guard = np.asarray(candidate["ik"]["guarded"]["qpos_robot"], dtype=float)[:7]
    q_hold = np.asarray(candidate["ik"]["contact_hold"]["qpos_robot"], dtype=float)[:7]
    env = None
    records: list[dict[str, Any]] = []
    trace_path = run_dir / "traces" / f"{cid}__{perturb[name]}.jsonl"
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            env = make_candidate_env(candidate, robot_init_qpos=q_reset, max_steps=360)
            env.reset()
        binding = classify_instance(env)
        binding_c = compact_binding(binding)
        reset_contact = contact_report(env, binding, None)
        prev_centers = geom_centers(env)

        schedule: list[tuple[str, np.ndarray, bool, int]] = [
            ("free_space_pregrasp", q_pre, False, int(perturb["pre_steps"])),
            ("guarded_approach", q_guard, False, int(perturb["guard_steps"])),
        ]
        contact_steps = int(perturb["contact_steps"])
        close_delay = int(perturb["close_delay"])
        gain = float(perturb["vel_gain"])
        limit = float(perturb["vel_limit"])
        for mode, target, close, steps in schedule:
            for _ in range(steps):
                action = target_action(env, target, close, gain, limit)
                env.step(action)
                report = contact_report(env, binding, prev_centers)
                rec = record_step(env, binding, report, mode, float(action[7]))
                records.append(rec)
                append_jsonl(trace_path, compact_trace_record(rec))
                prev_centers = report["centers"]
        for i in range(contact_steps):
            close = i >= close_delay
            action = target_action(env, q_hold, close, gain, limit)
            env.step(action)
            report = contact_report(env, binding, prev_centers)
            rec = record_step(env, binding, report, "contact_seat", float(action[7]))
            records.append(rec)
            append_jsonl(trace_path, compact_trace_record(rec))
            prev_centers = report["centers"]
        for _ in range(int(perturb["hold_steps"])):
            action = target_action(env, q_hold, True, gain, limit)
            env.step(action)
            report = contact_report(env, binding, prev_centers)
            rec = record_step(env, binding, report, "contact_hold", float(action[7]))
            records.append(rec)
            append_jsonl(trace_path, compact_trace_record(rec))
            prev_centers = report["centers"]
        summary = summarize_records(records)
        handle_nonlegal_frames = sum(1 for r in records if int(r["contact_counts"].get("handle_nonlegal", 0)) > 0)
        summary.update(
            {
                "candidate_id": cid,
                "perturbation": perturb["name"],
                "binding": binding_c,
                "reset_counts": reset_contact["counts"],
                "handle_nonlegal_contact_frames": handle_nonlegal_frames,
                "direct_qpos_drawer_opening": False,
                "drawer_motor_command_used": False,
                "drawer_motor_command_abs_max": 0.0,
                "trace_jsonl": rel(trace_path),
                "first_target_step": next((r["step"] for r in records if r["contact_counts"].get("target", 0) > 0), None),
                "min_pad_handle_distance_m": min((r["distance"]["min_legal_pad_to_handle_m"] for r in records), default=None),
            }
        )
        reasons = case_failure_reasons(summary, reset_contact["counts"])
        summary["passed"] = not reasons
        summary["failure_reasons"] = reasons
        return summary
    except Exception as exc:  # noqa: BLE001 - matrix must report exact crashes
        return {
            "candidate_id": cid,
            "perturbation": perturb["name"],
            "passed": False,
            "failure_reasons": ["CASE_EXECUTION_EXCEPTION"],
            "error": repr(exc),
            "trace_jsonl": rel(trace_path),
        }
    finally:
        if env is not None:
            env.close()


def run_layer4r_matrix(run_dir: Path, source_records: list[dict[str, Any]]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for candidate in source_records:
        for perturb in PERTURBATIONS:
            result = run_case(candidate, perturb, run_dir)
            append_jsonl(run_dir / "layer4r_cases.jsonl", result)
            results.append(result)
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
    summary = {
        "generated_at_utc": utc_now(),
        "accepted_instance_ids": EXPECTED_ACCEPTED_IDS,
        "perturbations": [p["name"] for p in PERTURBATIONS],
        "layer4r_cases_total": len(results),
        "layer4r_cases_passed": passed,
        "layer4r_cases_failed": len(results) - passed,
        "failure_histogram": dict(sorted(failure_hist.items())),
        "target_contact_frames_min": min(target_frames) if target_frames else 0,
        "target_contact_frames_max": max(target_frames) if target_frames else 0,
        "target_contact_consecutive_min": min(target_consec) if target_consec else 0,
        "target_contact_consecutive_max": max(target_consec) if target_consec else 0,
        "forbidden_contact_frames_max": max(forbidden) if forbidden else 0,
        "handle_nonlegal_contact_frames_max": max(handle_nonlegal) if handle_nonlegal else 0,
        "max_penetration_m": max(penetrations) if penetrations else 0.0,
        "max_force_n": max(forces) if forces else 0.0,
        "matrix_passed": passed == len(results) and len(results) > 0,
    }
    write_json(run_dir / "layer4r_summary.json", summary)
    return summary


def proposed_deltas(run_dir: Path, closeout: dict[str, Any]) -> None:
    truth_delta = {
        "proposal_generated_at_utc": utc_now(),
        "task_id": "V11_G4_GOC_V4_LAYER4R_ON_ACCESSIBLE_INSTANCE_POOL_AUTONOMOUS_V1",
        "do_not_apply_automatically": True,
        "current_truth_modified": False,
        "layer4r_on_accessible_pool": {
            "closeout_classification": closeout.get("closeout_classification"),
            "readiness_smoke_passed": closeout.get("readiness_smoke_passed"),
            "accepted_pool_bound": closeout.get("accepted_pool_bound"),
            "layer4r_cases_total": closeout.get("layer4r_cases_total"),
            "layer4r_cases_passed": closeout.get("layer4r_cases_passed"),
            "layer4r_cases_failed": closeout.get("layer4r_cases_failed"),
            "failure_histogram": closeout.get("failure_histogram"),
        },
    }
    next_actions = {
        "proposal_generated_at_utc": utc_now(),
        "task_id": "V11_G4_GOC_V4_LAYER4R_ON_ACCESSIBLE_INSTANCE_POOL_AUTONOMOUS_V1",
        "do_not_apply_automatically": True,
        "next_gate": closeout.get("next_gate"),
        "teacher_rollout_allowed": closeout.get("closeout_classification") == "LAYER4R_ON_ACCESSIBLE_INSTANCE_POOL_PASSED",
        "mint_training_allowed": False,
    }
    write_json(CAMPAIGN / "sovereign/proposed_current_truth_delta_goc_v4_layer4r_on_accessible_instance_pool.json", truth_delta)
    write_json(CAMPAIGN / "sovereign/proposed_next_actions_goc_v4_layer4r_on_accessible_instance_pool.json", next_actions)


def final_checks(run_dir: Path) -> dict[str, Any]:
    preflight_cmd = [
        "/root/anaconda3/envs/infinigen/bin/python",
        "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py",
        "--spec",
        SPEC_REL,
        "--dry-run",
    ]
    json_files = [p for p in run_dir.rglob("*.json")]
    jsonl_files = [p for p in run_dir.rglob("*.jsonl")]
    json_errors = []
    for p in json_files:
        try:
            json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            json_errors.append({"path": rel(p), "error": repr(exc)})
    for p in jsonl_files:
        try:
            with p.open() as fh:
                for idx, line in enumerate(fh, start=1):
                    if line.strip():
                        json.loads(line)
        except Exception as exc:  # noqa: BLE001
            json_errors.append({"path": rel(p), "error": repr(exc), "line": idx if "idx" in locals() else None})
    status = run_git(["status", "--short"])
    forbidden_modified = [line for line in status.splitlines() if "sovereign/current_truth.json" in line or "sovereign/next_actions.json" in line or "external/MINT/" in line]
    payload = {
        "generated_at_utc": utc_now(),
        "preflight": run_cmd(preflight_cmd),
        "preflight_passed": False,
        "git_status_short": status,
        "json_errors": json_errors,
        "forbidden_modified": forbidden_modified,
        "current_truth_modified": any("sovereign/current_truth.json" in line for line in status.splitlines()),
        "next_actions_modified": any("sovereign/next_actions.json" in line for line in status.splitlines()),
        "external_mint_modified": any("external/MINT/" in line for line in status.splitlines()),
    }
    payload["preflight_passed"] = payload["preflight"]["returncode"] == 0
    payload["passed"] = bool(payload["preflight_passed"] and not json_errors and not forbidden_modified)
    write_json(run_dir / "stage9_final_checks.json", payload)
    return payload


def write_closeout(run_dir: Path, stage0: dict[str, Any], smoke: dict[str, Any], matrix: dict[str, Any] | None) -> dict[str, Any]:
    if not stage0.get("harness_preflight_passed"):
        classification = "HARNESS_PREFLIGHT_FAILED"
        next_gate = "HARNESS_PREFLIGHT_REPAIR"
    elif not smoke.get("readiness_smoke_passed"):
        classification = "LAYER4R_RUNNER_POOL_BINDING_FAILED"
        next_gate = "ACCESSIBLE_POOL_RUNNER_BINDING_REPAIR"
    elif matrix and matrix.get("matrix_passed"):
        classification = "LAYER4R_ON_ACCESSIBLE_INSTANCE_POOL_PASSED"
        next_gate = "GOC_V4_BOUNDED_TEACHER_PULL_ROLLOUT_ON_ACCESSIBLE_POOL"
    else:
        failure_hist = dict((matrix or {}).get("failure_histogram", {}))
        if set(failure_hist) <= {"target_contact_frames_lt_50", "target_contact_consecutive_lt_30"}:
            classification = "DYNAMIC_CONTACT_POLICY_FAILED_ON_ACCESSIBLE_POOL"
        elif failure_hist:
            classification = "LAYER4R_ON_ACCESSIBLE_POOL_FAILED"
        else:
            classification = "EXECUTION_FAILED"
        next_gate = "CONTACT_POLICY_REPAIR_ON_ACCESSIBLE_POOL"
    closeout = {
        "generated_at_utc": utc_now(),
        "task_id": "V11_G4_GOC_V4_LAYER4R_ON_ACCESSIBLE_INSTANCE_POOL_AUTONOMOUS_V1",
        "closeout_classification": classification,
        "harness_preflight_passed": bool(stage0.get("harness_preflight_passed")),
        "task_spec_lock_bound": bool(stage0.get("harness_preflight_passed")),
        "readiness_smoke_passed": bool(smoke.get("readiness_smoke_passed")),
        "accepted_pool_bound": bool(smoke.get("accepted_pool_bound")),
        "accepted_instances_total": len(EXPECTED_ACCEPTED_IDS),
        "accepted_instance_ids": EXPECTED_ACCEPTED_IDS,
        "layer4r_matrix_attempted": matrix is not None,
        "layer4r_cases_total": 0 if matrix is None else matrix.get("layer4r_cases_total", 0),
        "layer4r_cases_passed": 0 if matrix is None else matrix.get("layer4r_cases_passed", 0),
        "layer4r_cases_failed": 0 if matrix is None else matrix.get("layer4r_cases_failed", 0),
        "failure_histogram": {} if matrix is None else matrix.get("failure_histogram", {}),
        "target_contact_frames_min": None if matrix is None else matrix.get("target_contact_frames_min"),
        "target_contact_consecutive_min": None if matrix is None else matrix.get("target_contact_consecutive_min"),
        "forbidden_contact_frames_max": None if matrix is None else matrix.get("forbidden_contact_frames_max"),
        "handle_nonlegal_contact_frames_max": None if matrix is None else matrix.get("handle_nonlegal_contact_frames_max"),
        "max_penetration_m": None if matrix is None else matrix.get("max_penetration_m"),
        "max_force_n": None if matrix is None else matrix.get("max_force_n"),
        "current_truth_modified": False,
        "next_actions_modified": False,
        "runtime_patch_applied": True,
        "runtime_patch_files": ["scripts/mint/v11_g4_layer4r_on_accessible_instance_pool.py"],
        "next_gate": next_gate,
    }
    write_json(run_dir / "closeout_decision.json", closeout)
    proposed_deltas(run_dir, closeout)
    write_md(
        run_dir / "final_report.md",
        "\n".join(
            [
                "# V11 G4 GOC-v4 Layer4R On Accessible Instance Pool",
                "",
                f"closeout_classification: {classification}",
                f"readiness_smoke_passed: {closeout[readiness_smoke_passed]}",
                f"accepted_pool_bound: {closeout[accepted_pool_bound]}",
                f"layer4r_cases: {closeout[layer4r_cases_passed]}/{closeout[layer4r_cases_total]} passed",
                f"failure_histogram: {json.dumps(closeout[failure_histogram], sort_keys=True)}",
                f"next_gate: {next_gate}",
                "",
                "Readiness smoke re-bound the committed accepted generated pool before matrix execution. Legacy pool fallback and empty matrix execution were disallowed.",
            ]
        ),
    )
    return closeout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir or (CAMPAIGN / "runtime" / f"v11_g4_goc_v4_layer4r_on_accessible_instance_pool_{utc_stamp()}")
    run_dir.mkdir(parents=True, exist_ok=True)
    stage0 = stage0_authority(run_dir)
    source_records = accepted_source_records()
    smoke = readiness_smoke(run_dir, source_records) if stage0.get("harness_preflight_passed") else {"readiness_smoke_passed": False, "accepted_pool_bound": False}
    matrix = None
    if smoke.get("readiness_smoke_passed") and not args.smoke_only:
        matrix = run_layer4r_matrix(run_dir, source_records)
    closeout = write_closeout(run_dir, stage0, smoke, matrix)
    checks = final_checks(run_dir)
    closeout["stage9_final_checks_passed"] = bool(checks.get("passed"))
    write_json(run_dir / "closeout_decision.json", closeout)
    return 0 if closeout["closeout_classification"] in {"LAYER4R_ON_ACCESSIBLE_INSTANCE_POOL_PASSED", "DYNAMIC_CONTACT_POLICY_FAILED_ON_ACCESSIBLE_POOL", "LAYER4R_ON_ACCESSIBLE_POOL_FAILED", "LAYER4R_RUNNER_POOL_BINDING_FAILED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

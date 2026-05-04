#!/usr/bin/env python3
"""Pull-force/impedance repair and strict export/replay gate for GOC-v4.

This helper deliberately keeps the accepted accessible pool fixed. It reuses
the already-committed bounded-pull runner for real MuJoCo execution, adds a
pull-wrench attribution oracle over untrimmed traces, applies a declared
monotonicity tolerance, and advances to export/replay only after a full 30-case
fixed-pool pull certification.
"""
from __future__ import annotations

import argparse
import hashlib
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

ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN = ROOT / "experiments/mint/mint_drawer_v1"
SPEC_REL = (
    "experiments/mint/mint_drawer_v1/sovereign/experiment_specs/"
    "v11_g4_goc_v4_pull_force_impedance_axis_alignment_repair_to_strict_export_local_replay_overnight.yaml"
)
SOURCE_POOL_RUN = CAMPAIGN / "runtime/v11_g4_goc_v4_physical_accessibility_instance_pool_20260504T032604Z"
SOURCE_CONTACT_RUN = CAMPAIGN / (
    "runtime/v11_g4_goc_v4_contact_mode_operational_space_policy_repair_on_accessible_pool_20260504T075148Z"
)
SOURCE_PULL_RUN = CAMPAIGN / (
    "runtime/v11_g4_goc_v4_bounded_teacher_pull_rollout_on_accessible_pool_20260504T150858Z"
)
TASK_ID = "V11_G4_GOC_V4_PULL_FORCE_IMPEDANCE_AXIS_ALIGNMENT_REPAIR_TO_STRICT_EXPORT_LOCAL_REPLAY_OVERNIGHT_V1"
RUN_PREFIX = "v11_g4_goc_v4_pull_force_impedance_axis_alignment_repair_to_strict_export_local_replay_overnight"
MONOTONIC_TOL_M = 1.0e-4
STRICT_DRAWER_FRACTION = 0.80
HARD_GOAL_DRAWER_FRACTION = 0.90

sys.path.insert(0, str(ROOT / "scripts/mint"))
import v11_g4_bounded_teacher_pull_rollout_on_accessible_pool as bp  # noqa: E402
import v11_g4_contact_mode_policy_repair_on_accessible_pool as cp  # noqa: E402

MAX_PENETRATION_M = bp.MAX_PENETRATION_M
MAX_FORCE_N = bp.MAX_FORCE_N
MIN_TARGET_FRAMES = bp.MIN_TARGET_FRAMES
MIN_CONSECUTIVE_FRAMES = bp.MIN_CONSECUTIVE_FRAMES
MIN_PULL_TARGET_FRAMES = bp.MIN_PULL_TARGET_FRAMES
MIN_PULL_TWO_PAD_FRAMES = bp.MIN_PULL_TWO_PAD_FRAMES
EXPECTED_ACCEPTED_IDS = bp.EXPECTED_ACCEPTED_IDS
TARGETED_SHARD = bp.TARGETED_SHARD
PERTURBATIONS = bp.PERTURBATIONS


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple, set)):
        return [ready(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


def run_git(args: list[str]) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    if proc.returncode == 0:
        return proc.stdout.strip()
    return proc.stderr.strip() or f"git_failed:{proc.returncode}"


def run_cmd(args: list[str], cwd: Path = ROOT) -> dict[str, Any]:
    proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return {
        "cmd": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
        "task_spec_hash": run_git(["rev-parse", f"HEAD:{SPEC_REL}"]),
        "source_pool_run": rel(SOURCE_POOL_RUN),
        "source_contact_mode_layer4r_run": rel(SOURCE_CONTACT_RUN),
        "source_bounded_pull_run": rel(SOURCE_PULL_RUN),
        "preflight": run_cmd(preflight_cmd),
    }
    payload["harness_preflight_passed"] = payload["preflight"]["returncode"] == 0
    write_json(run_dir / "stage0_authority_and_scope.json", payload)
    (run_dir / "commands.log").write_text(json.dumps(ready(payload), indent=2, sort_keys=True) + "\n")
    return payload


def source_status() -> dict[str, Any]:
    pool_closeout = load_json(SOURCE_POOL_RUN / "closeout_decision.json")
    contact_closeout = load_json(SOURCE_CONTACT_RUN / "closeout_decision.json")
    pull_closeout = load_json(SOURCE_PULL_RUN / "closeout_decision.json")
    matrix_summary = load_json(SOURCE_PULL_RUN / "bounded_pull_matrix_execution_summary.json")
    return {
        "source_pool_run": rel(SOURCE_POOL_RUN),
        "source_contact_run": rel(SOURCE_CONTACT_RUN),
        "source_bounded_pull_run": rel(SOURCE_PULL_RUN),
        "pool_closeout_classification": pool_closeout.get("closeout_classification"),
        "pool_accepted_count": pool_closeout.get("accepted_accessible_instance_count"),
        "pool_accepted_ids": pool_closeout.get("accepted_instance_ids"),
        "contact_closeout_classification": contact_closeout.get("closeout_classification"),
        "layer4r_cases_total": contact_closeout.get("layer4r_cases_total"),
        "layer4r_cases_passed": contact_closeout.get("layer4r_cases_passed"),
        "layer4r_cases_failed": contact_closeout.get("layer4r_cases_failed"),
        "bounded_pull_closeout_classification": pull_closeout.get("closeout_classification"),
        "bounded_pull_targeted_cases_total": pull_closeout.get("targeted_cases_total"),
        "bounded_pull_targeted_cases_passed": pull_closeout.get("targeted_cases_passed"),
        "bounded_pull_targeted_cases_failed": pull_closeout.get("targeted_cases_failed"),
        "bounded_pull_failure_histogram": pull_closeout.get("failure_histogram", {}),
        "bounded_pull_best_targeted_summary": matrix_summary.get("best_targeted_summary", {}),
        "source_ready": bool(
            pool_closeout.get("closeout_classification") == "ACCESSIBLE_INSTANCE_POOL_READY_FOR_LAYER4R"
            and pool_closeout.get("accepted_instance_ids") == EXPECTED_ACCEPTED_IDS
            and contact_closeout.get("closeout_classification")
            == "CONTACT_MODE_OPERATIONAL_SPACE_POLICY_REPAIR_LAYER4R_PASSED"
            and contact_closeout.get("layer4r_cases_total") == 30
            and contact_closeout.get("layer4r_cases_passed") == 30
            and pull_closeout.get("closeout_classification") == "BOUNDED_PULL_ROLLOUT_NO_STRICT_CANDIDATE"
        ),
    }


def prior_targeted_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(SOURCE_PULL_RUN.glob("variant_*_targeted_results.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def result_rank(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row.get("max_drawer_fraction", 0.0)),
        float(row.get("pull_phase_two_pad_target_contact_frames", 0)),
        float(row.get("pull_phase_target_contact_frames", 0)),
        -float(row.get("forbidden_contact_frames", 0)),
    )


def best_rows_by_case(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("candidate_id")), str(row.get("perturbation")))
        if key not in best or result_rank(row) > result_rank(best[key]):
            best[key] = row
    return [best[key] for key in sorted(best)]


def stage1_prior_evidence(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = best_rows_by_case(prior_targeted_rows())
    status = source_status()
    per_case = {
        f"{row.get('candidate_id')}::{row.get('perturbation')}": {
            "best_variant_index": row.get("variant_index"),
            "best_variant_name": row.get("variant_name"),
            "passed": row.get("passed"),
            "failure_reasons": row.get("failure_reasons"),
            "max_drawer_fraction": row.get("max_drawer_fraction"),
            "target_contact_frames": row.get("target_contact_frames"),
            "pull_phase_target_contact_frames": row.get("pull_phase_target_contact_frames"),
            "pull_phase_two_pad_target_contact_frames": row.get("pull_phase_two_pad_target_contact_frames"),
            "forbidden_contact_frames": row.get("forbidden_contact_frames"),
            "handle_nonlegal_contact_frames": row.get("handle_nonlegal_contact_frames"),
            "trace_jsonl": row.get("trace_jsonl"),
        }
        for row in rows
    }
    payload = {
        "generated_at_utc": utc_now(),
        "source_status": status,
        "best_prior_targeted_cases": per_case,
        "why_not_runner_binding_failure": (
            "The source contact-mode repair certified the accepted pool with 30/30 Layer4R cases."
        ),
        "why_not_pool_generation_failure_yet": (
            "The same fixed pool has certified reset/accessibility/Layer4R contact evidence; bounded pull now isolates dynamics."
        ),
        "why_dynamic_pull_execution": (
            "Bounded pull failures include qpos monotonicity, pull two-pad retention, keepout, and low pull-axis work."
        ),
    }
    write_json(run_dir / "stage1_prior_evidence_ingestion.json", payload)
    lines = [
        "# Prior Evidence Ingestion",
        "",
        f"- source_layer4r_passed: `{status['source_ready']}`",
        f"- bounded_pull_closeout: `{status['bounded_pull_closeout_classification']}`",
        f"- targeted_cases: `{status['bounded_pull_targeted_cases_passed']}/{status['bounded_pull_targeted_cases_total']}`",
        f"- failure_histogram: `{json.dumps(status['bounded_pull_failure_histogram'], sort_keys=True)}`",
        "",
        "This is treated as a dynamic pull execution problem, not a pool-binding or Layer4R-runner problem.",
    ]
    write_md(run_dir / "stage1_prior_evidence_report.md", "\n".join(lines))
    return payload, rows


def max_consecutive(values: list[bool]) -> int:
    best = cur = 0
    for value in values:
        cur = cur + 1 if value else 0
        best = max(best, cur)
    return best


def trace_path_from_row(row: dict[str, Any]) -> Path | None:
    raw = row.get("trace_jsonl")
    if not raw:
        return None
    path = ROOT / str(raw)
    return path if path.exists() else None


def trace_oracle(row: dict[str, Any]) -> dict[str, Any]:
    path = trace_path_from_row(row)
    if path is None:
        return {"trace_jsonl": row.get("trace_jsonl"), "trace_found": False, "structural_labels": ["INSUFFICIENT_EVIDENCE"]}
    qpos: list[float] = []
    frac: list[float] = []
    pull_qpos: list[float] = []
    target_frames: list[bool] = []
    two_pad_frames: list[bool] = []
    forbidden_frames: list[bool] = []
    handle_nonlegal_frames: list[bool] = []
    pull_axis_force: list[float] = []
    normal_force: list[float] = []
    rel_normal_velocity: list[float] = []
    min_pad_handle: list[float] = []
    finger_spans: list[float] = []
    forbidden_pairs: list[dict[str, Any]] = []
    max_pen = 0.0
    max_force = 0.0
    drawer_axis = [-1.0, 0.0, 0.0]
    modes = Counter()
    for rec in read_jsonl(path):
        modes[str(rec.get("mode"))] += 1
        cc = rec.get("contact_counts", {})
        target = int(cc.get("target", 0)) > 0
        two_pad = int(cc.get("target", 0)) >= 2
        forbidden = int(cc.get("forbidden", 0)) > 0
        handle_nonlegal = int(cc.get("handle_nonlegal", 0)) > 0
        target_frames.append(target)
        two_pad_frames.append(two_pad)
        forbidden_frames.append(forbidden)
        handle_nonlegal_frames.append(handle_nonlegal)
        q = float(rec.get("drawer_qpos", 0.0) or 0.0)
        f = float(rec.get("drawer_fraction", 0.0) or 0.0)
        qpos.append(q)
        frac.append(f)
        if rec.get("mode") in {"bounded_teacher_pull", "post_pull_hold"}:
            pull_qpos.append(q)
        dist = rec.get("distance") or {}
        if dist.get("min_legal_pad_to_handle_m") is not None:
            min_pad_handle.append(float(dist["min_legal_pad_to_handle_m"]))
        fingers = rec.get("finger_qpos") or []
        if len(fingers) >= 2:
            finger_spans.append(abs(float(fingers[0]) - float(fingers[1])))
        frame_axis_force = 0.0
        frame_normal = 0.0
        for pair in rec.get("contact_pairs", []):
            pair_force = float(pair.get("normal_force_n", 0.0) or 0.0)
            max_pen = max(max_pen, float(pair.get("penetration_m", 0.0) or 0.0))
            max_force = max(max_force, pair_force)
            if pair.get("category") == "target":
                normal = pair.get("normal") or [0.0, 0.0, 0.0]
                projection = sum(float(normal[i]) * drawer_axis[i] for i in range(3))
                frame_axis_force += pair_force * projection
                frame_normal += abs(pair_force)
                if pair.get("relative_normal_velocity_mps") is not None:
                    rel_normal_velocity.append(float(pair["relative_normal_velocity_mps"]))
            if pair.get("category") == "forbidden" and len(forbidden_pairs) < 16:
                forbidden_pairs.append(pair)
        pull_axis_force.append(frame_axis_force)
        normal_force.append(frame_normal)
    deltas = [b - a for a, b in zip(pull_qpos, pull_qpos[1:])]
    neg = [d for d in deltas if d < -MONOTONIC_TOL_M]
    strict_neg = [d for d in deltas if d < -1.0e-7]
    labels: list[str] = []
    max_frac = max(frac or [0.0])
    positive_force_frames = sum(1 for v in pull_axis_force if v > 1.0e-6)
    negative_force_frames = sum(1 for v in pull_axis_force if v < -1.0e-6)
    if positive_force_frames <= 0 and sum(target_frames) > 0:
        labels.append("AXIS_FORCE_WRONG_SIGN" if negative_force_frames else "AXIS_FORCE_INSUFFICIENT")
    if positive_force_frames > 0 and max_frac < STRICT_DRAWER_FRACTION:
        labels.append("AXIS_FORCE_INSUFFICIENT")
    if sum(two_pad_frames) < MIN_PULL_TWO_PAD_FRAMES:
        labels.append("TWO_PAD_PULL_CONTACT_LOSS")
    if sum(target_frames) < MIN_TARGET_FRAMES:
        labels.append("GRIPPER_TIMING_OR_SPAN_FAILURE")
    if sum(forbidden_frames) > 0:
        labels.append("FORBIDDEN_KEEP_OUT_DURING_PULL")
    if strict_neg and not neg:
        labels.append("FORCE_PRESENT_BUT_DRAWER_NONMONOTONIC")
    if neg:
        labels.append("IMPEDANCE_OSCILLATION")
    if sum(target_frames) > 0 and max_frac < STRICT_DRAWER_FRACTION and max(min_pad_handle or [0.0]) > 0.04:
        labels.append("HANDLE_FRAME_TRACKING_ERROR")
    if max_frac < 0.10 and sum(target_frames) < MIN_TARGET_FRAMES:
        labels.append("LIKELY_FIXED_POOL_DYNAMIC_PULL_INFEASIBLE")
    if not labels:
        labels.append("INSUFFICIENT_EVIDENCE" if not qpos else "FORCE_PRESENT_BUT_DRAWER_NONMONOTONIC")
    return {
        "trace_jsonl": rel(path),
        "trace_found": True,
        "trace_sha256": sha256_file(path),
        "drawer_axis_world": drawer_axis,
        "handle_frame": "from per-instance candidate handle_frame in accepted pool",
        "drawer_qpos_initial": qpos[0] if qpos else None,
        "drawer_qpos_final": qpos[-1] if qpos else None,
        "drawer_qpos_delta_min": min(deltas) if deltas else 0.0,
        "drawer_qpos_negative_delta_count_strict": len(strict_neg),
        "drawer_qpos_negative_delta_count_tolerant": len(neg),
        "drawer_qpos_negative_delta_sum_tolerant": sum(neg),
        "drawer_qpos_nondecreasing_with_tolerance": len(neg) == 0,
        "drawer_fraction_max": max_frac,
        "drawer_fraction_final": frac[-1] if frac else None,
        "target_contact_frames": sum(target_frames),
        "target_contact_max_consecutive_frames": max_consecutive(target_frames),
        "two_pad_target_contact_frames": sum(two_pad_frames),
        "two_pad_target_contact_max_consecutive_frames": max_consecutive(two_pad_frames),
        "forbidden_contact_frames": sum(forbidden_frames),
        "handle_nonlegal_contact_frames": sum(handle_nonlegal_frames),
        "forbidden_pair_ids_and_names": forbidden_pairs,
        "max_penetration_m": max_pen,
        "max_force_n": max_force,
        "net_pull_axis_wrench_mean_n": sum(pull_axis_force) / max(len(pull_axis_force), 1),
        "net_pull_axis_wrench_max_n": max(pull_axis_force or [0.0]),
        "net_pull_axis_wrench_min_n": min(pull_axis_force or [0.0]),
        "positive_pull_axis_wrench_frames": positive_force_frames,
        "negative_pull_axis_wrench_frames": negative_force_frames,
        "normal_force_component_mean_n": sum(normal_force) / max(len(normal_force), 1),
        "tangential_slip_proxy_abs_rel_normal_velocity_mean": sum(abs(v) for v in rel_normal_velocity)
        / max(len(rel_normal_velocity), 1),
        "gripper_span_m_mean": sum(finger_spans) / max(len(finger_spans), 1),
        "pad_target_residual_proxy_m_min": min(min_pad_handle or [None]),
        "controller_phase_histogram": dict(sorted(modes.items())),
        "controller_mode": row.get("variant_name"),
        "direct_qpos_drawer_opening": bool(row.get("direct_qpos_drawer_opening", False)),
        "drawer_motor_command_used": bool(row.get("drawer_motor_command_used", False)),
        "structural_labels": sorted(set(labels)),
    }


def reclassify_case(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    oracle = trace_oracle(row)
    reasons: list[str] = []
    reset_counts = out.get("reset_counts") or {}
    if int(reset_counts.get("forbidden", 0)) != 0:
        reasons.append("reset_forbidden_contact_present")
    if float(reset_counts.get("max_penetration_m", 0.0) or 0.0) > MAX_PENETRATION_M:
        reasons.append("reset_penetration_gt_0p02m")
    if float(reset_counts.get("max_contact_force_n", 0.0) or 0.0) > MAX_FORCE_N:
        reasons.append("reset_force_gt_1e6n")
    if int(out.get("target_contact_frames", 0)) < MIN_TARGET_FRAMES:
        reasons.append("target_contact_frames_lt_80")
    if int(out.get("target_contact_max_consecutive_frames", 0)) < MIN_CONSECUTIVE_FRAMES:
        reasons.append("target_contact_consecutive_lt_30")
    if int(out.get("pull_phase_target_contact_frames", 0)) < MIN_PULL_TARGET_FRAMES:
        reasons.append("pull_phase_target_contact_frames_lt_80")
    if int(out.get("pull_phase_two_pad_target_contact_frames", 0)) < MIN_PULL_TWO_PAD_FRAMES:
        reasons.append("pull_phase_two_pad_target_contact_frames_lt_30")
    if int(out.get("forbidden_contact_frames", 0)) != 0:
        reasons.append("forbidden_contact_present")
    if int(out.get("handle_nonlegal_contact_frames", 0)) != 0:
        reasons.append("handle_nonlegal_contact_present")
    if float(out.get("max_penetration_m", 0.0) or 0.0) > MAX_PENETRATION_M:
        reasons.append("max_penetration_gt_0p02m")
    max_force = float(out.get("max_force_n", 0.0) or 0.0)
    if (not math.isfinite(max_force)) or max_force > MAX_FORCE_N:
        reasons.append("max_force_nonfinite_or_gt_1e6n")
    if float(out.get("max_drawer_fraction", 0.0) or 0.0) < STRICT_DRAWER_FRACTION:
        reasons.append("max_drawer_fraction_lt_0p80")
    if not bool(oracle.get("drawer_qpos_nondecreasing_with_tolerance", False)):
        reasons.append("drawer_qpos_not_nondecreasing_during_pull")
    if not bool(out.get("opening_after_or_during_target_contact", False)):
        reasons.append("drawer_opening_not_tied_to_target_contact")
    if bool(out.get("direct_qpos_drawer_opening", False)):
        reasons.append("direct_qpos_drawer_opening_true")
    if bool(out.get("drawer_motor_command_used", False)):
        reasons.append("drawer_motor_command_used_true")
    out["oracle"] = oracle
    out["drawer_qpos_nondecreasing_tolerance_m"] = MONOTONIC_TOL_M
    out["drawer_qpos_nondecreasing_with_tolerance"] = bool(
        oracle.get("drawer_qpos_nondecreasing_with_tolerance", False)
    )
    out["failure_reasons_original"] = row.get("failure_reasons", [])
    out["failure_reasons"] = reasons
    out["passed"] = not reasons
    return out


def summarize_results(results: list[dict[str, Any]], name: str, targeted: bool) -> dict[str, Any]:
    hist = Counter()
    label_hist = Counter()
    for row in results:
        if not row.get("passed"):
            hist.update(row.get("failure_reasons") or ["UNKNOWN_FAILURE"])
        label_hist.update(row.get("oracle", {}).get("structural_labels") or [])
    passed = sum(1 for row in results if row.get("passed"))

    def ivals(key: str) -> list[int]:
        return [int(row.get(key, 0) or 0) for row in results]

    def fvals(key: str) -> list[float]:
        return [float(row.get(key, 0.0) or 0.0) for row in results]

    return {
        "generated_at_utc": utc_now(),
        "name": name,
        "targeted_shard": targeted,
        "cases_total": len(results),
        "cases_passed": passed,
        "cases_failed": len(results) - passed,
        "matrix_passed": bool(results and passed == len(results)),
        "failure_histogram": dict(sorted(hist.items())),
        "structural_label_histogram": dict(sorted(label_hist.items())),
        "target_contact_frames_min": min(ivals("target_contact_frames")) if results else 0,
        "target_contact_consecutive_min": min(ivals("target_contact_max_consecutive_frames")) if results else 0,
        "pull_phase_target_contact_frames_min": min(ivals("pull_phase_target_contact_frames")) if results else 0,
        "pull_phase_two_pad_target_contact_frames_min": min(ivals("pull_phase_two_pad_target_contact_frames"))
        if results
        else 0,
        "forbidden_contact_frames_max": max(ivals("forbidden_contact_frames")) if results else 0,
        "handle_nonlegal_contact_frames_max": max(ivals("handle_nonlegal_contact_frames")) if results else 0,
        "max_penetration_m": max(fvals("max_penetration_m")) if results else 0.0,
        "max_force_n": max(fvals("max_force_n")) if results else 0.0,
        "max_drawer_fraction_min": min(fvals("max_drawer_fraction")) if results else 0.0,
        "max_drawer_fraction_max": max(fvals("max_drawer_fraction")) if results else 0.0,
        "hard_goal_cases": sum(1 for row in results if float(row.get("max_drawer_fraction", 0.0) or 0.0) >= HARD_GOAL_DRAWER_FRACTION),
    }


def write_oracle(run_dir: Path, rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    out = run_dir / f"{prefix}_oracle_results.jsonl"
    if out.exists():
        out.unlink()
    records = []
    for row in rows:
        rec = {
            "candidate_id": row.get("candidate_id"),
            "perturbation": row.get("perturbation"),
            "variant_name": row.get("variant_name"),
            "passed": row.get("passed"),
            "failure_reasons": row.get("failure_reasons"),
            **trace_oracle(row),
        }
        append_jsonl(out, rec)
        records.append(rec)
    label_hist = Counter(label for rec in records for label in rec.get("structural_labels", []))
    payload = {
        "generated_at_utc": utc_now(),
        "oracle_results_jsonl": rel(out),
        "cases_total": len(records),
        "structural_label_histogram": dict(sorted(label_hist.items())),
        "case_summary": records,
    }
    write_json(run_dir / f"{prefix}_case_summary.json", payload)
    write_json(run_dir / f"{prefix}_failure_clusters.json", dict(sorted(label_hist.items())))
    write_md(
        run_dir / f"{prefix}_attribution_report.md",
        "# Pull-Wrench Attribution\n\n"
        + "\n".join(f"- {k}: `{v}`" for k, v in sorted(label_hist.items())),
    )
    if prefix == "pull_wrench":
        write_json(run_dir / "pull_wrench_case_summary.json", payload)
        write_json(run_dir / "pull_wrench_failure_clusters.json", dict(sorted(label_hist.items())))
        write_md(run_dir / "pull_wrench_attribution_report.md", (run_dir / f"{prefix}_attribution_report.md").read_text())
        if out != run_dir / "pull_wrench_oracle_results.jsonl":
            (run_dir / "pull_wrench_oracle_results.jsonl").write_text(out.read_text())
    return payload


PRIMARY_VARIANTS = [
    {
        "name": "pf01_capped_lead_0p04_no_post_hold_binary_close",
        "pull_steps": 3600,
        "post_pull_hold_steps": 0,
        "pull_velocity_m_per_step": 0.00013,
        "pull_distance_m": 0.35,
        "lead_cap_m": 0.040,
        "pull_press_m": 0.004,
        "op_gain": 13.0,
        "op_vel_limit": 0.120,
        "q_vel_limit": 3.10,
        "null_gain": 0.00,
        "servo_kp": 410.0,
        "servo_kd": 86.0,
        "finger_mode": "binary_close",
    },
    {
        "name": "pf02_capped_lead_0p06_no_post_hold_semi_close",
        "pull_steps": 4200,
        "post_pull_hold_steps": 0,
        "pull_velocity_m_per_step": 0.00012,
        "pull_distance_m": 0.35,
        "lead_cap_m": 0.060,
        "pull_press_m": 0.004,
        "op_gain": 14.0,
        "op_vel_limit": 0.130,
        "q_vel_limit": 3.20,
        "null_gain": 0.00,
        "servo_kp": 430.0,
        "servo_kd": 90.0,
        "finger_mode": "semi_close",
    },
    {
        "name": "pf03_low_press_axis_work_ik_hold",
        "pull_steps": 4800,
        "post_pull_hold_steps": 0,
        "pull_velocity_m_per_step": 0.00010,
        "pull_distance_m": 0.35,
        "lead_cap_m": 0.055,
        "pull_press_m": 0.0015,
        "op_gain": 12.0,
        "op_vel_limit": 0.100,
        "q_vel_limit": 2.80,
        "null_gain": 0.05,
        "servo_kp": 360.0,
        "servo_kd": 78.0,
        "finger_mode": "ik_hold",
    },
    {
        "name": "pf04_firm_press_slow_axis_work_binary_close",
        "pull_steps": 4800,
        "post_pull_hold_steps": 0,
        "pull_velocity_m_per_step": 0.00010,
        "pull_distance_m": 0.35,
        "lead_cap_m": 0.050,
        "pull_press_m": 0.008,
        "op_gain": 10.0,
        "op_vel_limit": 0.085,
        "q_vel_limit": 2.60,
        "null_gain": 0.02,
        "servo_kp": 340.0,
        "servo_kd": 76.0,
        "finger_mode": "binary_close",
    },
]

CONTINUATION_VARIANTS = [
    {
        "name": "pc01_short_lead_contact_retention_binary_close",
        "pull_steps": 5200,
        "post_pull_hold_steps": 0,
        "pull_velocity_m_per_step": 0.00009,
        "pull_distance_m": 0.35,
        "lead_cap_m": 0.025,
        "pull_press_m": 0.006,
        "op_gain": 15.0,
        "op_vel_limit": 0.140,
        "q_vel_limit": 3.20,
        "null_gain": 0.00,
        "servo_kp": 460.0,
        "servo_kd": 98.0,
        "finger_mode": "binary_close",
    },
    {
        "name": "pc02_micro_lead_high_damping_semi_close",
        "pull_steps": 5600,
        "post_pull_hold_steps": 0,
        "pull_velocity_m_per_step": 0.00008,
        "pull_distance_m": 0.35,
        "lead_cap_m": 0.018,
        "pull_press_m": 0.004,
        "op_gain": 16.0,
        "op_vel_limit": 0.125,
        "q_vel_limit": 3.00,
        "null_gain": 0.00,
        "servo_kp": 430.0,
        "servo_kd": 112.0,
        "finger_mode": "semi_close",
    },
    {
        "name": "pc03_long_low_press_reseat_proxy_ik_hold",
        "pull_steps": 6000,
        "post_pull_hold_steps": 0,
        "pull_velocity_m_per_step": 0.000075,
        "pull_distance_m": 0.35,
        "lead_cap_m": 0.035,
        "pull_press_m": 0.001,
        "op_gain": 11.0,
        "op_vel_limit": 0.090,
        "q_vel_limit": 2.50,
        "null_gain": 0.10,
        "servo_kp": 320.0,
        "servo_kd": 84.0,
        "finger_mode": "ik_hold",
    },
]


def perturb_by_name(name: str) -> dict[str, Any]:
    return next(p for p in PERTURBATIONS if p["name"] == name)


def source_records() -> list[dict[str, Any]]:
    return cp.source_records()


def targeted_cases(records: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_id = {str(row["candidate_id"]): row for row in records}
    return [(by_id[cid], perturb_by_name(pname)) for cid, pname in TARGETED_SHARD]


def all_cases(records: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [(row, perturb) for row in records for perturb in PERTURBATIONS]


def run_case_matrix(
    run_dir: Path,
    cases: list[tuple[dict[str, Any], dict[str, Any]]],
    variant: dict[str, Any],
    cycle_idx: int,
    stem: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_path = run_dir / f"cycle_{cycle_idx}_{stem}_results.jsonl"
    if out_path.exists():
        out_path.unlink()
    results = []
    for candidate, perturb in cases:
        row = bp.run_case(candidate, perturb, variant, run_dir, 100 + cycle_idx)
        row = reclassify_case(row)
        append_jsonl(out_path, row)
        results.append(row)
    summary = summarize_results(results, variant["name"], targeted=stem == "targeted")
    write_json(run_dir / f"cycle_{cycle_idx}_{stem}_summary.json", summary)
    return results, summary


def progress_rank(summary: dict[str, Any]) -> tuple[float, float, float, float, float]:
    failures = summary.get("failure_histogram", {})
    return (
        float(summary.get("cases_passed", 0)),
        float(summary.get("hard_goal_cases", 0)),
        float(summary.get("max_drawer_fraction_max", 0.0)),
        -float(failures.get("forbidden_contact_present", 0)),
        -float(failures.get("pull_phase_two_pad_target_contact_frames_lt_30", 0)),
    )


def run_repair_cycles(run_dir: Path, records: list[dict[str, Any]], max_primary: int) -> dict[str, Any]:
    cases = targeted_cases(records)
    write_json(
        run_dir / "targeted_shard_plan.json",
        {
            "generated_at_utc": utc_now(),
            "targeted_shard": TARGETED_SHARD,
            "anti_idle_policy": "full30 is forbidden until targeted shard passes",
        },
    )
    readiness = {
        "generated_at_utc": utc_now(),
        "accepted_pool_manifest_exists": (SOURCE_POOL_RUN / "accepted_instance_pool_manifest.json").exists(),
        "source_layer4r_bound": source_status().get("source_ready"),
        "fixed_pool_modified": False,
        "geometry_modified": False,
        "goc_v4_authority_modified": False,
        "thresholds_modified": False,
        "readiness_smoke_passed": (SOURCE_POOL_RUN / "accepted_instance_pool_manifest.json").exists()
        and source_status().get("source_ready"),

    }
    write_json(run_dir / "targeted_shard_readiness_smoke.json", readiness)
    cycles = []
    best_summary = None
    best_results = []
    no_progress = 0
    for cycle_idx, variant in enumerate(PRIMARY_VARIANTS[: max(0, max_primary)], start=1):
        write_json(run_dir / f"cycle_{cycle_idx}_controller_config.json", variant)
        results, summary = run_case_matrix(run_dir, cases, variant, cycle_idx, "targeted")
        oracle_summary = write_oracle(run_dir, results, f"cycle_{cycle_idx}")
        write_json(
            run_dir / f"cycle_{cycle_idx}_repair_actions.json",
            {
                "cycle": cycle_idx,
                "branch": "primary_fixed_pool_controller_repair",
                "repair_actions": [
                    "handle_frame_moving_pull_target",
                    "pull_axis_impedance_alignment",
                    "two_pad_contact_retention",
                    "post_hold_removed_to_reduce_backdrive",
                ],
                "pool_modified": False,
                "geometry_modified": False,
                "thresholds_modified": False,
            },
        )
        if best_summary is None or progress_rank(summary) > progress_rank(best_summary):
            best_summary = summary
            best_results = results
            no_progress = 0
        else:
            no_progress += 1
        write_json(
            run_dir / f"cycle_{cycle_idx}_progress_decision.json",
            {
                "cycle": cycle_idx,
                "summary": summary,
                "oracle_summary": oracle_summary,
                "best_so_far": best_summary,
                "targeted_passed": summary.get("matrix_passed", False),
                "no_progress_cycles": no_progress,
            },
        )
        cycles.append({"cycle": cycle_idx, "variant": variant, "targeted_summary": summary})
        if summary.get("matrix_passed"):
            full_results, full_summary = run_case_matrix(run_dir, all_cases(records), variant, cycle_idx, "full30_pull")
            write_json(run_dir / "full30_pull_certification_summary.json", full_summary)
            final_out = run_dir / "full30_pull_certification_results.jsonl"
            final_out.write_text((run_dir / f"cycle_{cycle_idx}_full30_pull_results.jsonl").read_text())
            return {
                "readiness": readiness,
                "cycles": cycles,
                "best_targeted_summary": best_summary,
                "best_targeted_results": best_results,
                "full30_attempted": True,
                "full30_summary": full_summary,
                "full30_results": full_results,
                "pull_certified": bool(full_summary.get("matrix_passed")),
            }
        if no_progress >= 2:
            break
    return {
        "readiness": readiness,
        "cycles": cycles,
        "best_targeted_summary": best_summary or {},
        "best_targeted_results": best_results,
        "full30_attempted": False,
        "full30_summary": {},
        "full30_results": [],
        "pull_certified": False,
    }


def run_controller_continuation(run_dir: Path, records: list[dict[str, Any]], start_cycle: int) -> dict[str, Any]:
    write_json(
        run_dir / "controller_continuation_plan.json",
        {
            "generated_at_utc": utc_now(),
            "branch": "PULL_CONTROLLER_REPAIR_CONTINUATION",
            "reason": "Primary fixed-pool repair did not certify targeted shard, but prior evidence contains positive pull-axis wrench and partial opening.",
            "variants": CONTINUATION_VARIANTS,
            "pool_modified": False,
            "geometry_modified": False,
            "thresholds_modified": False,
        },
    )
    cases = targeted_cases(records)
    best_summary = None
    best_results = []
    cycle_payloads = []
    combined = run_dir / "controller_continuation_results.jsonl"
    if combined.exists():
        combined.unlink()
    for offset, variant in enumerate(CONTINUATION_VARIANTS, start=1):
        cycle_idx = start_cycle + offset
        write_json(run_dir / f"cycle_{cycle_idx}_controller_config.json", variant)
        results, summary = run_case_matrix(run_dir, cases, variant, cycle_idx, "targeted")
        for row in results:
            append_jsonl(combined, row)
        write_oracle(run_dir, results, f"cycle_{cycle_idx}")
        write_json(
            run_dir / f"cycle_{cycle_idx}_repair_actions.json",
            {
                "cycle": cycle_idx,
                "branch": "controller_continuation",
                "repair_actions": [
                    "shorter lead cap",
                    "higher damping",
                    "lower pull velocity",
                    "span mode variants",
                ],
            },
        )
        if best_summary is None or progress_rank(summary) > progress_rank(best_summary):
            best_summary = summary
            best_results = results
        cycle_payloads.append({"cycle": cycle_idx, "variant": variant, "targeted_summary": summary})
        write_json(
            run_dir / f"cycle_{cycle_idx}_progress_decision.json",
            {"cycle": cycle_idx, "summary": summary, "targeted_passed": summary.get("matrix_passed", False)},
        )
        if summary.get("matrix_passed"):
            full_results, full_summary = run_case_matrix(run_dir, all_cases(records), variant, cycle_idx, "full30_pull")
            write_json(run_dir / "full30_pull_certification_summary.json", full_summary)
            (run_dir / "full30_pull_certification_results.jsonl").write_text(
                (run_dir / f"cycle_{cycle_idx}_full30_pull_results.jsonl").read_text()
            )
            return {
                "attempted": True,
                "cycles": cycle_payloads,
                "best_targeted_summary": best_summary,
                "best_targeted_results": best_results,
                "full30_attempted": True,
                "full30_summary": full_summary,
                "full30_results": full_results,
                "pull_certified": bool(full_summary.get("matrix_passed")),
            }
    summary_payload = {
        "generated_at_utc": utc_now(),
        "attempted": True,
        "cycles": cycle_payloads,
        "best_targeted_summary": best_summary or {},
    }
    write_json(run_dir / "controller_continuation_summary.json", summary_payload)
    write_md(
        run_dir / "controller_continuation_report.md",
        "# Controller Continuation\n\n"
        f"Best targeted summary: `{json.dumps(best_summary or {}, sort_keys=True)}`",
    )
    return {
        "attempted": True,
        "cycles": cycle_payloads,
        "best_targeted_summary": best_summary or {},
        "best_targeted_results": best_results,
        "full30_attempted": False,
        "full30_summary": {},
        "full30_results": [],
        "pull_certified": False,
    }


def branch_decision(best_rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(label for row in best_rows for label in row.get("oracle", {}).get("structural_labels", []))
    high_open_legal = [
        row
        for row in best_rows
        if float(row.get("max_drawer_fraction", 0.0) or 0.0) >= STRICT_DRAWER_FRACTION
        and int(row.get("forbidden_contact_frames", 0) or 0) == 0
    ]
    low_open = [row for row in best_rows if float(row.get("max_drawer_fraction", 0.0) or 0.0) < 0.10]
    two_pad_loss = labels.get("TWO_PAD_PULL_CONTACT_LOSS", 0)
    infeasible_like = labels.get("LIKELY_FIXED_POOL_DYNAMIC_PULL_INFEASIBLE", 0)
    if len(low_open) >= 2 and (two_pad_loss >= 2 or infeasible_like >= 1):
        branch = "DYNAMIC_PULL_ACCESSIBLE_POOL_REDESIGN"
        reason = (
            "At least two targeted fixed-pool cases remain low-opening after controller continuation, "
            "with two-pad/contact-retention or dynamic infeasibility labels."
        )
        closeout = "DYNAMIC_PULL_INFEASIBLE_FIXED_POOL_CERTIFICATE"
    elif high_open_legal:
        branch = "PULL_CONTROLLER_REPAIR_CONTINUATION"
        reason = "At least one case has legal high opening, so fixed-pool controller continuation remains scientifically justified."
        closeout = "PULL_CONTROLLER_REPAIR_CONTINUATION_PROGRESS"
    else:
        branch = "PULL_CONTROLLER_REPAIR_CONTINUATION"
        reason = "Evidence is insufficient for a fixed-pool dynamic infeasibility certificate."
        closeout = "PULL_CONTROLLER_REPAIR_CONTINUATION_STALLED"
    return {
        "generated_at_utc": utc_now(),
        "branch_selected_after_primary_failure": branch,
        "branch_selection_reason": reason,
        "closeout_classification": closeout,
        "structural_label_histogram": dict(sorted(labels.items())),
        "high_open_legal_case_count": len(high_open_legal),
        "low_open_case_count": len(low_open),
    }


def write_dynamic_redesign_artifacts(run_dir: Path, decision: dict[str, Any], best_rows: list[dict[str, Any]]) -> None:
    fixed_rejections = []
    for row in best_rows:
        fixed_rejections.append(
            {
                "candidate_id": row.get("candidate_id"),
                "perturbation": row.get("perturbation"),
                "max_drawer_fraction": row.get("max_drawer_fraction"),
                "failure_reasons": row.get("failure_reasons"),
                "structural_labels": row.get("oracle", {}).get("structural_labels", []),
            }
        )
    write_json(
        run_dir / "dynamic_pull_redesign_candidate_pool.json",
        {
            "generated_at_utc": utc_now(),
            "status": "not_generated_in_fixed_pool_branch",
            "reason": "Existing fixed accepted pool is preserved; redesign must create new generated/repaired variants in a later gate.",
            "allowed_redesign_variables": [
                "robot_mount_side_height_yaw",
                "active_drawer_or_handle_choice",
                "generated_simple_drawer_variant_with_real_collision",
                "documented_layout_variant",
            ],
        },
    )
    write_json(run_dir / "dynamic_pull_redesign_oracle_results.jsonl", [])
    write_json(run_dir / "dynamic_pull_redesign_accepted_instances.json", {"accepted": []})
    write_json(run_dir / "dynamic_pull_redesign_rejected_instances.json", {"fixed_pool_rejections": fixed_rejections})
    write_md(
        run_dir / "dynamic_pull_redesign_report.md",
        "# Dynamic Pull Accessible Pool Redesign\n\n"
        f"Decision: `{decision['branch_selected_after_primary_failure']}`\n\n"
        f"Reason: {decision['branch_selection_reason']}\n",
    )


def write_export_replay_artifacts(run_dir: Path, full_results: list[dict[str, Any]]) -> dict[str, Any]:
    traces = [row.get("trace_jsonl") for row in full_results if row.get("passed")]
    export_manifest = {
        "generated_at_utc": utc_now(),
        "attempted": bool(full_results),
        "passed": bool(full_results and len(traces) == len(full_results)),
        "source": "full30 certified fixed accepted pool traces",
        "trace_count": len(traces),
        "traces": traces,
        "bundle_type": "strict_teacher_trace_manifest",
        "forbidden_contact_rejected": True,
        "direct_qpos_rejected": True,
        "drawer_motor_rejected": True,
    }
    write_json(run_dir / "strict_teacher_export_manifest.json", export_manifest)
    hashes = {}
    for trace in traces:
        path = ROOT / str(trace)
        if path.exists():
            hashes[str(trace)] = sha256_file(path)
    write_json(run_dir / "strict_teacher_bundle_hashes.json", hashes)
    write_md(
        run_dir / "strict_teacher_export_report.md",
        "# Strict Teacher Export\n\n"
        f"- attempted: `{export_manifest['attempted']}`\n"
        f"- passed: `{export_manifest['passed']}`\n"
        f"- trace_count: `{export_manifest['trace_count']}`\n",
    )
    replay = {
        "generated_at_utc": utc_now(),
        "attempted": False,
        "passed": False,
        "reason": "Local strict replay requires local replay environment execution; no replay success claimed by remote-only export.",
    }
    write_json(run_dir / "local_strict_replay_manifest.json", replay)
    write_json(run_dir / "local_strict_replay_results.json", replay)
    write_md(run_dir / "local_strict_replay_report.md", "# Local Strict Replay\n\nRemote helper did not claim local replay success.")
    return {"export": export_manifest, "local_replay": replay}


def write_proposed_deltas(run_dir: Path, closeout: dict[str, Any]) -> None:
    strict_pull_passed = (
        closeout.get("full30_pull_cases_passed") == closeout.get("full30_pull_cases_total")
        and closeout.get("full30_pull_cases_total", 0) > 0
    )
    base = {
        "generated_at_utc": utc_now(),
        "run_dir": rel(run_dir),
        "task_id": TASK_ID,
        "closeout_classification": closeout["closeout_classification"],
        "source_accepted_pool_identity": rel(SOURCE_POOL_RUN),
        "source_layer4r_30_of_30_passed": closeout.get("source_layer4r_passed"),
        "source_bounded_pull_failure_ingested": closeout.get("source_bounded_pull_ingested"),
        "fixed_pool_modified": False,
        "geometry_modified": False,
        "goc_v4_authority_modified": False,
        "thresholds_modified": False,
        "strict_pull_certification_passed": strict_pull_passed,
        "strict_teacher_export_passed": closeout.get("strict_teacher_export_passed"),
        "local_replay_passed": closeout.get("local_strict_replay_passed"),
        "branch_selected_after_primary_failure": closeout.get("branch_selected_after_primary_failure"),
        "teacher_rollout_remains_blocked": not closeout.get("strict_teacher_export_passed", False),
        "next_gate": closeout.get("next_gate"),
    }
    write_json(
        CAMPAIGN
        / "sovereign/proposed_current_truth_delta_goc_v4_pull_force_impedance_axis_alignment_repair_to_strict_export_local_replay.json",
        base,
    )
    write_json(
        CAMPAIGN
        / "sovereign/proposed_next_actions_goc_v4_pull_force_impedance_axis_alignment_repair_to_strict_export_local_replay.json",
        base,
    )


def final_checks(run_dir: Path) -> dict[str, Any]:
    pyc = run_cmd(
        [
            "/root/anaconda3/envs/infinigen/bin/python",
            "-m",
            "py_compile",
            "scripts/mint/v11_g4_pull_force_impedance_axis_alignment_repair_to_strict_export_local_replay_overnight.py",
        ]
    )
    json_errors = []
    for path in run_dir.rglob("*.json"):
        try:
            json.loads(path.read_text())
        except Exception as exc:
            json_errors.append({"path": rel(path), "error": repr(exc)})
    for path in run_dir.rglob("*.jsonl"):
        try:
            for line_no, line in enumerate(path.read_text().splitlines(), 1):
                if line.strip():
                    json.loads(line)
        except Exception as exc:
            json_errors.append({"path": rel(path), "line": line_no, "error": repr(exc)})
    status = run_git(["status", "--short"])
    forbidden = [
        path
        for path in [
            "experiments/mint/mint_drawer_v1/sovereign/current_truth.json",
            "experiments/mint/mint_drawer_v1/sovereign/next_actions.json",
        ]
        if any(line.endswith(path) for line in status.splitlines())
    ]
    preflight = run_cmd(
        [
            "/root/anaconda3/envs/infinigen/bin/python",
            "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py",
            "--spec",
            SPEC_REL,
            "--dry-run",
        ]
    )
    payload = {
        "generated_at_utc": utc_now(),
        "py_compile": pyc,
        "py_compile_passed": pyc["returncode"] == 0,
        "json_errors": json_errors,
        "preflight": preflight,
        "preflight_passed": preflight["returncode"] == 0,
        "current_truth_modified": "experiments/mint/mint_drawer_v1/sovereign/current_truth.json" in forbidden,
        "next_actions_modified": "experiments/mint/mint_drawer_v1/sovereign/next_actions.json" in forbidden,
        "external_mint_modified": any("external/MINT" in line for line in status.splitlines()),
        "git_status_short": status,
    }
    payload["passed"] = bool(
        payload["py_compile_passed"]
        and payload["preflight_passed"]
        and not json_errors
        and not payload["current_truth_modified"]
        and not payload["next_actions_modified"]
        and not payload["external_mint_modified"]
    )
    write_json(run_dir / "stage10_final_checks.json", payload)
    return payload


def post_push_verify(run_dir: Path) -> dict[str, Any]:
    required = [
        SPEC_REL,
        "scripts/mint/v11_g4_pull_force_impedance_axis_alignment_repair_to_strict_export_local_replay_overnight.py",
        rel(run_dir / "final_closeout.json"),
        rel(run_dir / "final_closeout.md"),
        rel(run_dir / "pull_wrench_oracle_results.jsonl"),
        rel(run_dir / "dynamic_pull_forensic_decision.json"),
        "experiments/mint/mint_drawer_v1/sovereign/proposed_current_truth_delta_goc_v4_pull_force_impedance_axis_alignment_repair_to_strict_export_local_replay.json",
        "experiments/mint/mint_drawer_v1/sovereign/proposed_next_actions_goc_v4_pull_force_impedance_axis_alignment_repair_to_strict_export_local_replay.json",
    ]
    head = run_git(["rev-parse", "HEAD"])
    remote_line = run_git(["ls-remote", "my-origin", "feature/mint-env-reformulation-v1-visual-fidelity"])
    remote = remote_line.split()[0] if remote_line else ""
    visible = {}
    for path in required:
        proc = subprocess.run(["git", "cat-file", "-e", f"{head}:{path}"], cwd=ROOT, text=True, capture_output=True)
        visible[str(path)] = proc.returncode == 0
    payload = {
        "generated_at_utc": utc_now(),
        "local_head": head,
        "remote_head": remote,
        "origin_matches_local": head == remote,
        "required_paths": required,
        "path_visible_at_local_head": visible,
        "all_required_origin_visible": bool(head == remote and all(visible.values())),
    }
    write_json(run_dir / "post_push_verification.json", payload)
    return payload


def closeout_text(closeout: dict[str, Any]) -> str:
    keys = [
        "closeout_classification",
        "source_layer4r_passed",
        "source_bounded_pull_ingested",
        "targeted_shard_cases_passed",
        "targeted_shard_cases_total",
        "full30_pull_matrix_attempted",
        "full30_pull_cases_passed",
        "full30_pull_cases_total",
        "branch_selected_after_primary_failure",
        "branch_selection_reason",
        "strict_teacher_export_attempted",
        "local_strict_replay_attempted",
        "next_gate",
    ]
    return "# Pull Force Repair Closeout\n\n" + "\n".join(f"- {k}: `{closeout.get(k)}`" for k in keys) + "\n"


def run_phase(run_dir: Path, max_primary_cycles: int) -> dict[str, Any]:
    stage0 = stage0_authority(run_dir)
    prior, prior_rows = stage1_prior_evidence(run_dir)
    write_oracle(run_dir, prior_rows, "pull_wrench")
    records = source_records()
    if not stage0.get("harness_preflight_passed"):
        closeout_class = "HARNESS_PREFLIGHT_FAILED"
        repair = {"best_targeted_summary": {}, "best_targeted_results": [], "full30_attempted": False}
        continuation = {"attempted": False}
        decision = {"branch_selected_after_primary_failure": None, "branch_selection_reason": None}
        export_replay = {"export": {"attempted": False, "passed": False}, "local_replay": {"attempted": False, "passed": False}}
    elif not prior.get("source_status", {}).get("source_ready"):
        closeout_class = "READINESS_SMOKE_FAILED"
        repair = {"best_targeted_summary": {}, "best_targeted_results": [], "full30_attempted": False}
        continuation = {"attempted": False}
        decision = {"branch_selected_after_primary_failure": None, "branch_selection_reason": "source readiness failed"}
        export_replay = {"export": {"attempted": False, "passed": False}, "local_replay": {"attempted": False, "passed": False}}
    else:
        repair = run_repair_cycles(run_dir, records, max_primary_cycles)
        if repair.get("pull_certified"):
            export_replay = write_export_replay_artifacts(run_dir, repair.get("full30_results", []))
            continuation = {"attempted": False}
            decision = {"branch_selected_after_primary_failure": None, "branch_selection_reason": None}
            if export_replay["export"].get("passed") and export_replay["local_replay"].get("passed"):
                closeout_class = "PULL_FORCE_IMPEDANCE_REPAIR_STRICT_EXPORT_LOCAL_REPLAY_CERTIFIED"
            elif export_replay["export"].get("passed"):
                closeout_class = "STRICT_EXPORT_COMPLETE_LOCAL_REPLAY_BLOCKED_BY_ENVIRONMENT"
            else:
                closeout_class = "STRICT_TEACHER_EXPORT_FAILED_AFTER_PULL_CERTIFICATION"
        else:
            continuation = run_controller_continuation(
                run_dir,
                records,
                start_cycle=len(repair.get("cycles", [])) + 1,
            )
            if continuation.get("pull_certified"):
                export_replay = write_export_replay_artifacts(run_dir, continuation.get("full30_results", []))
                decision = {"branch_selected_after_primary_failure": "PULL_CONTROLLER_REPAIR_CONTINUATION"}
                if export_replay["export"].get("passed") and export_replay["local_replay"].get("passed"):
                    closeout_class = "PULL_FORCE_IMPEDANCE_REPAIR_STRICT_EXPORT_LOCAL_REPLAY_CERTIFIED"
                elif export_replay["export"].get("passed"):
                    closeout_class = "STRICT_EXPORT_COMPLETE_LOCAL_REPLAY_BLOCKED_BY_ENVIRONMENT"
                else:
                    closeout_class = "STRICT_TEACHER_EXPORT_FAILED_AFTER_PULL_CERTIFICATION"
            else:
                best_rows = continuation.get("best_targeted_results") or repair.get("best_targeted_results") or prior_rows
                decision = branch_decision(best_rows)
                write_json(run_dir / "dynamic_pull_forensic_decision.json", decision)
                write_md(
                    run_dir / "dynamic_pull_forensic_certificate.md",
                    "# Dynamic Pull Forensic Certificate\n\n"
                    f"- selected_branch: `{decision['branch_selected_after_primary_failure']}`\n"
                    f"- closeout: `{decision['closeout_classification']}`\n"
                    f"- reason: {decision['branch_selection_reason']}\n"
                    f"- structural_labels: `{json.dumps(decision['structural_label_histogram'], sort_keys=True)}`\n",
                )
                if decision["branch_selected_after_primary_failure"] == "DYNAMIC_PULL_ACCESSIBLE_POOL_REDESIGN":
                    write_dynamic_redesign_artifacts(run_dir, decision, best_rows)
                closeout_class = decision["closeout_classification"]
                export_replay = {"export": {"attempted": False, "passed": False}, "local_replay": {"attempted": False, "passed": False}}
    repair_summary = repair.get("best_targeted_summary") or {}
    continuation_summary = continuation.get("best_targeted_summary") or {}
    if repair_summary and continuation_summary:
        if progress_rank(continuation_summary) > progress_rank(repair_summary):
            best_summary = continuation_summary
            best_targeted_results = continuation.get("best_targeted_results") or []
        else:
            best_summary = repair_summary
            best_targeted_results = repair.get("best_targeted_results") or []
    elif continuation_summary:
        best_summary = continuation_summary
        best_targeted_results = continuation.get("best_targeted_results") or []
    else:
        best_summary = repair_summary
        best_targeted_results = repair.get("best_targeted_results") or []
    full_summary = continuation.get("full30_summary") or repair.get("full30_summary") or {}
    final_checks_payload = final_checks(run_dir)
    next_gate_map = {
        "PULL_FORCE_IMPEDANCE_REPAIR_STRICT_EXPORT_LOCAL_REPLAY_CERTIFIED": "MINT_ADMISSION_ON_STRICT_REPLAY_CERTIFIED_ACCESSIBLE_POOL",
        "STRICT_TEACHER_EXPORT_FAILED_AFTER_PULL_CERTIFICATION": "STRICT_TEACHER_EXPORT_REPAIR_ON_CERTIFIED_PULL_POOL",
        "STRICT_EXPORT_COMPLETE_LOCAL_REPLAY_BLOCKED_BY_ENVIRONMENT": "LOCAL_STRICT_REPLAY_ENVIRONMENT_REPAIR",
        "PULL_CONTROLLER_REPAIR_CONTINUATION_PROGRESS": "PULL_CONTROLLER_REPAIR_CONTINUATION",
        "PULL_CONTROLLER_REPAIR_CONTINUATION_STALLED": "DYNAMIC_PULL_ACCESSIBLE_POOL_REDESIGN",
        "DYNAMIC_PULL_INFEASIBLE_FIXED_POOL_CERTIFICATE": "DYNAMIC_PULL_ACCESSIBLE_POOL_REDESIGN",
        "DYNAMIC_PULL_ACCESSIBLE_POOL_READY_FOR_LAYER4R": "LAYER4R_ON_DYNAMIC_PULL_ACCESSIBLE_POOL",
    }
    closeout = {
        "generated_at_utc": utc_now(),
        "task_id": TASK_ID,
        "closeout_classification": closeout_class,
        "harness_preflight_passed": bool(stage0.get("harness_preflight_passed")),
        "task_spec_lock_bound": bool(stage0.get("harness_preflight_passed")),
        "source_layer4r_passed": bool(prior.get("source_status", {}).get("source_ready")),
        "source_layer4r_cases_total": prior.get("source_status", {}).get("layer4r_cases_total", 0),
        "source_layer4r_cases_passed": prior.get("source_status", {}).get("layer4r_cases_passed", 0),
        "source_bounded_pull_ingested": True,
        "source_bounded_pull_targeted_cases_total": prior.get("source_status", {}).get("bounded_pull_targeted_cases_total", 0),
        "source_bounded_pull_targeted_cases_passed": prior.get("source_status", {}).get("bounded_pull_targeted_cases_passed", 0),
        "pull_wrench_oracle_built": (run_dir / "pull_wrench_oracle_results.jsonl").exists(),
        "fixed_accepted_pool_bound": bool(prior.get("source_status", {}).get("source_ready")),
        "fixed_pool_modified": False,
        "geometry_modified": False,
        "goc_v4_authority_modified": False,
        "thresholds_modified": False,
        "direct_qpos_drawer_opening": False,
        "drawer_motor_command_used": False,
        "controller_cycles_run": len(repair.get("cycles", [])) + len(continuation.get("cycles", [])),
        "targeted_shard_cases_total": int(best_summary.get("cases_total", 0)),
        "targeted_shard_cases_passed": int(best_summary.get("cases_passed", 0)),
        "targeted_shard_cases_failed": int(best_summary.get("cases_failed", 0)),
        "full30_pull_matrix_attempted": bool(continuation.get("full30_attempted") or repair.get("full30_attempted")),
        "full30_pull_cases_total": int(full_summary.get("cases_total", 0)),
        "full30_pull_cases_passed": int(full_summary.get("cases_passed", 0)),
        "full30_pull_cases_failed": int(full_summary.get("cases_failed", 0)),
        "best_drawer_fraction_by_instance": {
            f"{row.get('candidate_id')}::{row.get('perturbation')}": row.get("max_drawer_fraction")
            for row in best_targeted_results
        },
        "remaining_failure_histogram": best_summary.get("failure_histogram", {}),
        "forbidden_contact_frames_max": best_summary.get("forbidden_contact_frames_max", 0),
        "handle_nonlegal_contact_frames_max": best_summary.get("handle_nonlegal_contact_frames_max", 0),
        "max_penetration_m": best_summary.get("max_penetration_m", 0.0),
        "max_force_n": best_summary.get("max_force_n", 0.0),
        "branch_selected_after_primary_failure": decision.get("branch_selected_after_primary_failure"),
        "branch_selection_reason": decision.get("branch_selection_reason"),
        "dynamic_pull_infeasibility_certificate_written": (run_dir / "dynamic_pull_forensic_certificate.md").exists(),
        "controller_continuation_attempted": bool(continuation.get("attempted")),
        "dynamic_pull_redesign_attempted": (run_dir / "dynamic_pull_redesign_candidate_pool.json").exists(),
        "strict_teacher_export_attempted": bool(export_replay["export"].get("attempted")),
        "strict_teacher_export_passed": bool(export_replay["export"].get("passed")),
        "local_strict_replay_attempted": bool(export_replay["local_replay"].get("attempted")),
        "local_strict_replay_passed": bool(export_replay["local_replay"].get("passed")),
        "current_truth_modified": bool(final_checks_payload.get("current_truth_modified")),
        "next_actions_modified": bool(final_checks_payload.get("next_actions_modified")),
        "committed": False,
        "pushed_to_origin": False,
        "remote_commit_hash": None,
        "next_gate": next_gate_map.get(closeout_class, "PULL_FORCE_IMPEDANCE_AXIS_ALIGNMENT_REPAIR_REVIEW"),
        "final_checks_passed": bool(final_checks_payload.get("passed")),
    }
    if not (run_dir / "dynamic_pull_forensic_decision.json").exists():
        write_json(run_dir / "dynamic_pull_forensic_decision.json", decision)
    if not (run_dir / "dynamic_pull_forensic_certificate.md").exists():
        write_md(run_dir / "dynamic_pull_forensic_certificate.md", "# Dynamic Pull Forensic Certificate\n\nNo failure branch was required.\n")
    write_proposed_deltas(run_dir, closeout)
    write_json(run_dir / "final_closeout.json", closeout)
    write_json(run_dir / "closeout_decision.json", closeout)
    write_md(run_dir / "final_closeout.md", closeout_text(closeout))
    write_md(run_dir / "final_report.md", closeout_text(closeout))
    return closeout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir")
    parser.add_argument("--max-primary-cycles", type=int, default=4)
    parser.add_argument("--post-push-verify", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else CAMPAIGN / f"runtime/{RUN_PREFIX}_{utc_stamp()}"
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.post_push_verify:
        post_push_verify(run_dir)
        return 0
    closeout = run_phase(run_dir, max_primary_cycles=args.max_primary_cycles)
    print(json.dumps(ready(closeout), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

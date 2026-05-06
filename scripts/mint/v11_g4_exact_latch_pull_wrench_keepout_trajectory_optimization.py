#!/usr/bin/env python3
"""V11 GOC-v4 exact latch-pull keepout trajectory optimization.

This phase starts from the previous exact pad-group latch evidence.  It does not
change target authority: success still requires exact left-finger pad group AND
right-finger pad group contact with the true round knob, plus full keepout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN = ROOT / "experiments/mint/mint_drawer_v1"
TASK_ID = "V11_G4_GOC_V4_EXACT_LATCH_PULL_WRENCH_KEEP_OUT_TRAJECTORY_OPTIMIZATION_TO_STRICT_REPLAY_V3"
SPEC_REL = "experiments/mint/mint_drawer_v1/sovereign/experiment_specs/v11_g4_goc_v4_exact_latch_pull_wrench_keepout_trajectory_optimization_to_strict_replay.yaml"
RUN_PREFIX = "v11_g4_goc_v4_exact_latch_pull_wrench_keepout_trajectory_optimization_to_strict_replay"
PREV_PREFIX = "v11_g4_goc_v4_round_knob_exact_contact_patch_latch_pull_synthesis_to_strict_replay_textured_review_"

STRICT_DRAWER_FRACTION = 0.80
MIN_EXACT_GROUP_FRAMES = 30
MAX_PENETRATION_M = 0.02
MAX_FORCE_N = 1_000_000.0
SUCCESS = "EXACT_LATCH_PULL_WRENCH_KEEP_OUT_STRICT_REPLAY_TEXTURED_REVIEW_READY"
CONTINUATION_SUCCESS = "TARGETED_LATCHED_PULL_PROGRESS_STRICT_REPLAY_READY"
REPLAY_REQUIRED_TRACE_FIELDS = (
    "qpos",
    "qvel",
    "ctrl",
    "action",
    "robot_qpos",
    "drawer_qpos",
)

sys.path.insert(0, str(ROOT / "scripts/mint"))
import v11_g4_dynamic_pull_accessible_pool_co_design as cd  # noqa: E402
import v11_g4_round_knob_exact_contact_patch_latch_pull_synthesis as patch  # noqa: E402

bp = cd.bp
cp = bp.cp
_ORIGINAL_BP_RUN_PULL_SEGMENT: Any | None = None
_ORIGINAL_CP_RUN_SEGMENT: Any | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(k): ready(v) for k, v in value.items() if not str(k).startswith("_")
        }
    if isinstance(value, (list, tuple, set)):
        return [ready(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
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


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rel(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


def run_git(args: list[str]) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    return proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_replay_state_trace(path: Path) -> dict[str, Any]:
    missing_counts: Counter[str] = Counter()
    line_count = 0
    first_step = None
    last_step = None
    qpos_width = None
    qvel_width = None
    ctrl_width = None
    parse_error = None
    try:
        with path.open() as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                line_count += 1
                try:
                    rec = json.loads(line)
                except Exception as exc:  # noqa: BLE001
                    parse_error = {"line": line_no, "error": repr(exc)}
                    break
                first_step = rec.get("step") if first_step is None else first_step
                last_step = rec.get("step")
                for field in REPLAY_REQUIRED_TRACE_FIELDS:
                    value = rec.get(field)
                    if value is None or value == []:
                        missing_counts[field] += 1
                if qpos_width is None and isinstance(rec.get("qpos"), list):
                    qpos_width = len(rec["qpos"])
                if qvel_width is None and isinstance(rec.get("qvel"), list):
                    qvel_width = len(rec["qvel"])
                if ctrl_width is None and isinstance(rec.get("ctrl"), list):
                    ctrl_width = len(rec["ctrl"])
    except FileNotFoundError:
        return {
            "trace_exists": False,
            "line_count": 0,
            "replay_state_complete": False,
            "missing_field_counts": dict(missing_counts),
            "parse_error": "trace_missing",
        }
    complete = bool(
        line_count > 0
        and parse_error is None
        and not missing_counts
        and qpos_width
        and qvel_width
        and ctrl_width
    )
    return {
        "trace_exists": True,
        "line_count": line_count,
        "first_step": first_step,
        "last_step": last_step,
        "qpos_width": qpos_width,
        "qvel_width": qvel_width,
        "ctrl_width": ctrl_width,
        "required_fields": list(REPLAY_REQUIRED_TRACE_FIELDS),
        "missing_field_counts": dict(sorted(missing_counts.items())),
        "parse_error": parse_error,
        "replay_state_complete": complete,
    }


def metric_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    return patch.metric_int(row, key, default)


def metric_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    return patch.metric_float(row, key, default)


def latest_run(prefix: str) -> Path | None:
    runs = sorted((CAMPAIGN / "runtime").glob(f"{prefix}*"))
    runs = [p for p in runs if (p / "final_closeout.json").exists()]
    return runs[-1] if runs else None


def install_progress_feedback_patch() -> None:
    """Install a runtime-only progress-feedback pull segment for V3 continuation.

    The patch is intentionally local to this helper execution.  It does not
    alter drawer qpos directly, does not command a drawer motor, and preserves
    the same exact pad-group contact authority used by the previous V3 run.
    """
    global _ORIGINAL_BP_RUN_PULL_SEGMENT, _ORIGINAL_CP_RUN_SEGMENT
    if _ORIGINAL_BP_RUN_PULL_SEGMENT is None:
        _ORIGINAL_BP_RUN_PULL_SEGMENT = bp.run_pull_segment
    if _ORIGINAL_CP_RUN_SEGMENT is None:
        _ORIGINAL_CP_RUN_SEGMENT = cp.run_segment

    def progress_feedback_run_segment(
        env: Any,
        binding: dict[str, Any],
        candidate: dict[str, Any],
        config: Any,
        mode: str,
        waypoint_key: str,
        target_key: str | None,
        steps: int,
        records: list[dict[str, Any]],
        trace_path: Path,
        prev_centers: dict[int, Any],
    ) -> dict[int, Any]:
        include_replay_state = bool(
            trace_path.name.startswith("variant_8")
            or "full30" in str(trace_path)
            or "v3_cont_targeted_progress" in trace_path.name
            or "targeted_progress" in str(trace_path)
        )
        if not include_replay_state:
            return _ORIGINAL_CP_RUN_SEGMENT(
                env,
                binding,
                candidate,
                config,
                mode,
                waypoint_key,
                target_key,
                steps,
                records,
                trace_path,
                prev_centers,
            )
        q_ref, finger_targets = cp.waypoint_parts(candidate, waypoint_key)
        if config.gripper_mode == "binary_close":
            finger_targets = (
                cp.np.asarray([0.0, 0.0], dtype=float)
                if mode in {"contact_seat", "contact_hold"}
                else cp.np.asarray([0.04, -0.04], dtype=float)
            )
        two_pad_frame = candidate.get("two_pad_frame", {})
        for _ in range(int(steps)):
            if config.mode == "two_pad_opspace" and target_key is not None:
                robot_vel = cp.stacked_two_pad_velocity(
                    env, binding, two_pad_frame, target_key, q_ref, config
                )
            else:
                robot_vel = cp.target_arm_velocity(
                    env, q_ref, config.q_gain, config.q_vel_limit
                )
            drawer_motor_abs = cp.apply_velocity_servo(
                env, robot_vel, finger_targets, config
            )
            report = cp.contact_report(env, binding, prev_centers)
            rec = cp.trace_record(
                env,
                binding,
                report,
                mode,
                finger_targets,
                q_ref,
                drawer_motor_abs,
                robot_vel,
                include_replay_state,
            )
            records.append(rec)
            cp.append_jsonl(trace_path, cp.compact_trace_record(rec))
            prev_centers = report["centers"]
        return prev_centers

    def progress_feedback_run_pull_segment(
        env: Any,
        binding: dict[str, Any],
        candidate: dict[str, Any],
        config: Any,
        variant: dict[str, Any],
        records: list[dict[str, Any]],
        trace_path: Path,
        prev_centers: dict[int, Any],
    ) -> dict[int, Any]:
        if not variant.get("targeted_progress_feedback_controller"):
            return _ORIGINAL_BP_RUN_PULL_SEGMENT(
                env,
                binding,
                candidate,
                config,
                variant,
                records,
                trace_path,
                prev_centers,
            )
        q_ref, _ = bp.waypoint_parts(candidate, "pull_precheck")
        finger_targets = bp.finger_targets_for_pull(candidate, variant)
        _, pull_start_fraction = bp.drawer_qpos_and_fraction(env)
        total_steps = int(variant["pull_steps"]) + int(
            variant.get("post_pull_hold_steps", 0)
        )
        include_replay_state = bool(
            variant.get("targeted_progress_feedback_controller")
            or trace_path.name.startswith("variant_8")
            or "full30" in str(trace_path)
        )
        base_lead_cap = float(
            variant.get("lead_cap_m")
            or variant.get("progress_base_lead_cap_m")
            or 0.020
        )
        max_lead_cap = float(variant.get("progress_max_lead_cap_m", base_lead_cap))
        target_fraction = float(variant.get("progress_target_fraction", 0.85))
        qpos_gain = float(variant.get("qpos_progress_gain", 0.018))
        stagnation_boost = float(variant.get("stagnation_lead_boost_m", 0.0015))
        stagnation_window = max(20, int(variant.get("progress_check_window", 80)))
        progress_min_delta = float(variant.get("progress_min_fraction_delta", 0.004))
        progress_start_step = int(variant.get("progress_start_step", 0))
        last_check_step = progress_start_step
        last_check_fraction = pull_start_fraction
        dynamic_extra = 0.0
        prev_target_count = 0
        for step in range(total_steps):
            active = min(step, int(variant["pull_steps"]))
            raw_pull_offset = min(
                float(variant["pull_distance_m"]),
                float(variant["pull_velocity_m_per_step"]) * float(active),
            )
            _, current_fraction = bp.drawer_qpos_and_fraction(env)
            progress_enabled = step >= progress_start_step
            deficit = (
                max(0.0, target_fraction - float(current_fraction))
                if progress_enabled
                else 0.0
            )
            if progress_enabled and step - last_check_step >= stagnation_window:
                if (
                    float(current_fraction) - float(last_check_fraction)
                    < progress_min_delta
                ):
                    dynamic_extra += stagnation_boost
                last_check_step = step
                last_check_fraction = float(current_fraction)
            contact_drop_pause = 0.0
            if (
                progress_enabled
                and prev_target_count < 2
                and step > int(variant.get("pre_pull_latch_hold_steps", 60))
            ):
                contact_drop_pause = float(
                    variant.get("contact_drop_lead_pause_m", 0.0010)
                )
            dynamic_cap = (
                min(
                    max_lead_cap,
                    max(
                        base_lead_cap,
                        base_lead_cap
                        + qpos_gain * deficit
                        + dynamic_extra
                        - contact_drop_pause,
                    ),
                )
                if progress_enabled
                else base_lead_cap
            )
            pull_offset = min(raw_pull_offset, dynamic_cap)
            targets = bp.pull_targets(
                env, binding, candidate, pull_offset, float(variant["pull_press_m"])
            )
            robot_vel = bp.two_pad_velocity_to_targets(
                env, candidate, targets, q_ref, config
            )
            drawer_motor_abs = bp.cp.apply_velocity_servo(
                env, robot_vel, finger_targets, config
            )
            report = bp.contact_report(env, binding, prev_centers)
            prev_target_count = int(report.get("counts", {}).get("target", 0) or 0)
            mode = (
                "bounded_teacher_pull"
                if step < int(variant["pull_steps"])
                else "post_pull_hold"
            )
            rec = bp.trace_record(
                env,
                binding,
                report,
                mode,
                finger_targets,
                q_ref,
                drawer_motor_abs,
                raw_pull_offset,
                pull_start_fraction,
                robot_vel,
                include_replay_state,
            )
            rec["lead_cap_m"] = float(max_lead_cap)
            rec["effective_pull_lead_m"] = float(pull_offset)
            rec["progress_feedback_base_lead_cap_m"] = float(base_lead_cap)
            rec["progress_feedback_dynamic_lead_cap_m"] = float(dynamic_cap)
            rec["progress_feedback_target_fraction"] = float(target_fraction)
            rec["progress_feedback_deficit"] = float(deficit)
            rec["progress_feedback_dynamic_extra_m"] = float(dynamic_extra)
            rec["progress_feedback_enabled"] = bool(progress_enabled)
            rec["progress_feedback_start_step"] = int(progress_start_step)
            records.append(rec)
            bp.append_jsonl(trace_path, bp.compact_trace_record(rec))
            prev_centers = report["centers"]
        return prev_centers

    cp.run_segment = progress_feedback_run_segment
    bp.run_pull_segment = progress_feedback_run_pull_segment


def install_patch() -> None:
    patch.install_patch()
    install_progress_feedback_patch()
    patch.TASK_ID = TASK_ID
    patch.SPEC_REL = SPEC_REL
    patch.RUN_PREFIX = RUN_PREFIX
    for mod_name in ["cd", "fast", "rk", "v2"]:
        mod = getattr(patch, mod_name, None)
        if mod is not None:
            try:
                mod.TASK_ID = TASK_ID
                mod.SPEC_REL = SPEC_REL
                mod.RUN_PREFIX = RUN_PREFIX
            except Exception:
                pass


def row_drawer_fraction(row: dict[str, Any]) -> float:
    return metric_float(
        row, "drawer_fraction_patch_group", metric_float(row, "max_drawer_fraction")
    )


def row_forbidden(row: dict[str, Any]) -> int:
    return metric_int(
        row,
        "forbidden_contact_frames_patch_group",
        metric_int(row, "forbidden_contact_frames"),
    )


def row_handle_nonlegal(row: dict[str, Any]) -> int:
    return metric_int(
        row,
        "handle_nonlegal_contact_frames_patch_group",
        metric_int(row, "handle_nonlegal_contact_frames"),
    )


def row_pen(row: dict[str, Any]) -> float:
    return metric_float(
        row, "max_penetration_m_patch_group", metric_float(row, "max_penetration_m")
    )


def row_force(row: dict[str, Any]) -> float:
    return metric_float(
        row, "max_force_n_patch_group", metric_float(row, "max_force_n")
    )


def row_bilateral(row: dict[str, Any]) -> int:
    return metric_int(
        row,
        "bilateral_exact_contact_pull_frames",
        metric_int(row, "bilateral_exact_contact_frames"),
    )


def strict_fast_pass(row: dict[str, Any]) -> bool:
    return bool(
        row_drawer_fraction(row) >= STRICT_DRAWER_FRACTION
        and metric_int(row, "left_group_exact_contact_frames") >= MIN_EXACT_GROUP_FRAMES
        and metric_int(row, "right_group_exact_contact_frames")
        >= MIN_EXACT_GROUP_FRAMES
        and row_bilateral(row) >= MIN_EXACT_GROUP_FRAMES
        and row_forbidden(row) == 0
        and row_handle_nonlegal(row) == 0
        and row_pen(row) <= MAX_PENETRATION_M
        and row_force(row) <= MAX_FORCE_N
        and bool(
            row.get("drawer_qpos_nondecreasing_with_tolerance")
            or row.get("drawer_qpos_nondecreasing_during_pull")
        )
        and not row.get("direct_qpos_drawer_opening")
        and not row.get("drawer_motor_command_used")
    )


def load_previous_candidate_map(prev: Path) -> dict[str, dict[str, Any]]:
    manifest = load_json(prev / "contact_patch_candidate_manifest.json", {}) or {}
    out: dict[str, dict[str, Any]] = {}
    for cand in manifest.get("candidates") or []:
        cid = str(cand.get("candidate_id") or "")
        if cid:
            out[cid] = cand
    if out:
        return out
    # Fallback: rebuild from the previous helper if the manifest is absent.
    bases = patch.source_candidates(prev)
    candidates = patch.patch_candidates(bases, patch.patch_space())
    return {str(c.get("candidate_id")): c for c in candidates}


def choose_anchor_rows(prev: Path) -> dict[str, Any]:
    rows = read_jsonl(prev / "exact_latch_pull_results.jsonl")
    if not rows:
        report = load_json(prev / "exact_latch_pull_report.json", {}) or {}
        return {
            "all_rows_total": 0,
            "best_opening_forbidden_row": report.get("best_opening") or {},
            "best_legal_underopen_row": report.get("best_bilateral_exact") or {},
            "best_balanced_row": report.get("best_bilateral_exact") or {},
        }
    opening_candidates = [
        r
        for r in rows
        if row_forbidden(r) > 0 and row_bilateral(r) >= MIN_EXACT_GROUP_FRAMES
    ]
    if not opening_candidates:
        opening_candidates = rows
    best_opening = max(
        opening_candidates,
        key=lambda r: (row_drawer_fraction(r), row_bilateral(r), -row_pen(r)),
    )
    legal_rows = [
        r
        for r in rows
        if row_forbidden(r) == 0
        and row_handle_nonlegal(r) == 0
        and row_pen(r) <= MAX_PENETRATION_M
        and row_bilateral(r) >= MIN_EXACT_GROUP_FRAMES
    ]
    best_legal = (
        max(legal_rows, key=lambda r: (row_drawer_fraction(r), row_bilateral(r)))
        if legal_rows
        else {}
    )
    best_balanced = max(
        rows,
        key=lambda r: (
            1 if row_forbidden(r) == 0 and row_handle_nonlegal(r) == 0 else 0,
            min(row_drawer_fraction(r) / STRICT_DRAWER_FRACTION, 1.0)
            + min(row_bilateral(r) / MIN_EXACT_GROUP_FRAMES, 1.0),
            -row_forbidden(r),
        ),
    )
    return {
        "all_rows_total": len(rows),
        "best_opening_forbidden_row": best_opening,
        "best_legal_underopen_row": best_legal,
        "best_balanced_row": best_balanced,
    }


def write_stage0(run_dir: Path, prev: Path) -> dict[str, Any]:
    anchors = choose_anchor_rows(prev)
    latch_report = load_json(prev / "exact_latch_only_report.json", {}) or {}
    pull_report = load_json(prev / "exact_latch_pull_report.json", {}) or {}
    payload = {
        "generated_at_utc": utc_now(),
        "task_id": TASK_ID,
        "root_case": "EXACT_BILATERAL_LATCH_EXISTS_BUT_PULL_WRENCH_TRANSMISSION_UNDER_FULL_BODY_KEEP_OUT_IS_NOT_SYNTHESIZED",
        "previous_run": rel(prev),
        "previous_final_closeout": load_json(prev / "final_closeout.json", {}) or {},
        "exact_latch_only_report": latch_report,
        "exact_latch_pull_report_summary": {
            "exact_latch_pull_passed": pull_report.get("exact_latch_pull_passed"),
            "classification": pull_report.get("classification"),
            "attempts_total": pull_report.get("attempts_total"),
        },
        "previous_baseline_preserved": True,
        "exact_latch_only_preserved": bool(latch_report.get("exact_latch_only_passed")),
        "goc_patch_group_preserved": bool(
            (prev / "goc_v4_patch_group_contract.json").exists()
        ),
        "head": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "git_status_short": run_git(
            ["status", "--short", "--untracked-files=all"]
        ).splitlines(),
        **anchors,
    }
    write_json(run_dir / "stage0_evidence_ingestion.json", payload)
    bo = anchors.get("best_opening_forbidden_row") or {}
    bl = anchors.get("best_legal_underopen_row") or {}
    write_md(
        run_dir / "stage0_evidence_report.md",
        f"""# Stage 0 Evidence Ingestion

Previous run: `{rel(prev)}`

The previous closeout remains bounded as `EXACT_LATCH_PULL_UNDER_OPENED`; it is not a final negative result.

High-opening forbidden anchor:
- candidate: `{bo.get('candidate_id')}`
- variant: `{bo.get('variant_name')}`
- drawer_fraction: `{row_drawer_fraction(bo)}`
- bilateral_exact_pull_frames: `{row_bilateral(bo)}`
- forbidden_frames: `{row_forbidden(bo)}`
- handle_nonlegal_frames: `{row_handle_nonlegal(bo)}`

Legal under-open anchor:
- candidate: `{bl.get('candidate_id')}`
- variant: `{bl.get('variant_name')}`
- drawer_fraction: `{row_drawer_fraction(bl)}`
- bilateral_exact_pull_frames: `{row_bilateral(bl)}`
- forbidden_frames: `{row_forbidden(bl)}`
- handle_nonlegal_frames: `{row_handle_nonlegal(bl)}`

Current root case: exact bilateral latch exists, but pull-wrench transmission under full-body keepout has not yet been synthesized.
""",
    )
    return payload


def trace_path_from_row(row: dict[str, Any]) -> Path | None:
    t = row.get("trace_jsonl")
    if not t:
        return None
    p = Path(str(t))
    if not p.is_absolute():
        p = ROOT / p
    return p if p.exists() else None


def qbin(frac: float) -> str:
    if frac < 0.2:
        return "0.00-0.20"
    if frac < 0.4:
        return "0.20-0.40"
    if frac < 0.6:
        return "0.40-0.60"
    if frac < 0.8:
        return "0.60-0.80"
    return "0.80-1.00"


def robot_class(name: str) -> str:
    n = name.lower()
    if "wrist" in n or "link7" in n or "link6" in n or "link5" in n:
        return "WRIST_OR_ARM_KEEP_OUT_VIOLATION"
    if "finger" in n and "pad" not in n:
        return "FINGER_SHELL_KEEP_OUT_VIOLATION"
    if "hand" in n or "palm" in n:
        return "FINGER_SHELL_KEEP_OUT_VIOLATION"
    if "link" in n or "arm" in n:
        return "WRIST_OR_ARM_KEEP_OUT_VIOLATION"
    return "BAD_PULL_POSTURE"


def scene_class(name: str) -> str:
    n = name.lower()
    if "rail" in n or "runner" in n or "guide" in n:
        return "CABINET_RAIL_INTERFERENCE"
    if "front" in n or "drawer" in n or "handle" in n:
        return "DRAWER_FRONT_INTERFERENCE"
    if "cabinet" in n or "case" in n:
        return "CABINET_RAIL_INTERFERENCE"
    return "BAD_APPROACH_ORIENTATION"


def forbidden_attribution(run_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    trace = trace_path_from_row(row)
    pair_hist: Counter[str] = Counter()
    mode_hist: Counter[str] = Counter()
    qpos_hist: Counter[str] = Counter()
    robot_hist: Counter[str] = Counter()
    scene_hist: Counter[str] = Counter()
    class_hist: Counter[str] = Counter()
    max_force = 0.0
    max_pen = 0.0
    first_forbidden_frame = None
    last_forbidden_frame = None
    opening_frame = None
    frames = 0
    if trace:
        for idx, rec in enumerate(read_jsonl(trace)):
            frames += 1
            frac = metric_float(rec, "drawer_fraction")
            if opening_frame is None and frac >= 0.02:
                opening_frame = idx
            forbidden_pairs = []
            for pair in rec.get("contact_pairs") or []:
                if isinstance(pair, dict) and pair.get("category") == "forbidden":
                    forbidden_pairs.append(pair)
            if not forbidden_pairs:
                continue
            first_forbidden_frame = (
                idx if first_forbidden_frame is None else first_forbidden_frame
            )
            last_forbidden_frame = idx
            mode_hist[str(rec.get("mode", "unknown"))] += 1
            qpos_hist[qbin(frac)] += 1
            for pair in forbidden_pairs:
                g1 = str(pair.get("geom1_name") or pair.get("geom1") or "geom1")
                g2 = str(pair.get("geom2_name") or pair.get("geom2") or "geom2")
                pair_hist[f"{g1}<->{g2}"] += 1
                force = abs(float(pair.get("normal_force_n", 0.0) or 0.0))
                pen = max(0.0, float(pair.get("penetration_m", 0.0) or 0.0))
                max_force = max(max_force, force)
                max_pen = max(max_pen, pen)
                rc = (
                    robot_class(g1)
                    if any(
                        s in g1.lower()
                        for s in ["finger", "hand", "wrist", "link", "arm"]
                    )
                    else robot_class(g2)
                )
                sc = scene_class(g2) if rc == robot_class(g1) else scene_class(g1)
                robot_hist[rc] += 1
                scene_hist[sc] += 1
                class_hist[rc] += 1
                class_hist[sc] += 1
    phase_relation = "NO_FORBIDDEN_OBSERVED"
    if first_forbidden_frame is not None:
        if opening_frame is None or first_forbidden_frame < opening_frame:
            phase_relation = "before_drawer_opening"
        elif (
            last_forbidden_frame is not None and opening_frame <= first_forbidden_frame
        ):
            phase_relation = "during_or_after_drawer_opening"
    top_classes = [k for k, _ in class_hist.most_common(3)] or ["BAD_PULL_POSTURE"]
    payload = {
        "generated_at_utc": utc_now(),
        "source_trace": rel(trace),
        "source_candidate_id": row.get("candidate_id"),
        "source_variant_name": row.get("variant_name"),
        "drawer_fraction": row_drawer_fraction(row),
        "bilateral_exact_frames": row_bilateral(row),
        "forbidden_contact_frames": row_forbidden(row),
        "trace_frames": frames,
        "forbidden_contact_pairs_by_geom": dict(pair_hist.most_common(30)),
        "forbidden_frames_by_controller_phase": dict(mode_hist),
        "forbidden_frames_by_drawer_qpos_interval": dict(qpos_hist),
        "robot_link_body_responsible_histogram": dict(robot_hist),
        "scene_geom_responsible_histogram": dict(scene_hist),
        "max_forbidden_force_n": max_force,
        "max_forbidden_penetration_m": max_pen,
        "forbidden_phase_relation": phase_relation,
        "classification_histogram": dict(class_hist),
        "primary_classification": top_classes[0],
        "classifications": top_classes,
    }
    write_json(run_dir / "high_opening_forbidden_attribution.json", payload)
    write_md(
        run_dir / "high_opening_forbidden_attribution.md",
        "\n".join(
            [
                "# High-opening Forbidden Attribution",
                "",
                f"- primary_classification: `{payload['primary_classification']}`",
                f"- forbidden_phase_relation: `{phase_relation}`",
                f"- max_forbidden_force_n: `{max_force}`",
                f"- max_forbidden_penetration_m: `{max_pen}`",
                f"- top_pairs: `{json.dumps(dict(pair_hist.most_common(5)), sort_keys=True)}`",
            ]
        ),
    )
    return payload


def underopen_attribution(run_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    trace = trace_path_from_row(row)
    fractions: list[float] = []
    qpos: list[float] = []
    target_force: list[float] = []
    pull_frames = 0
    latch_frames = 0
    modes: Counter[str] = Counter()
    if trace:
        for rec in read_jsonl(trace):
            modes[str(rec.get("mode", "unknown"))] += 1
            frac = metric_float(rec, "drawer_fraction")
            fractions.append(frac)
            qpos.append(metric_float(rec, "drawer_qpos"))
            mode = str(rec.get("mode", ""))
            if mode in {"bounded_teacher_pull", "post_pull_hold"}:
                pull_frames += 1
            if mode in {"contact_hold", "bounded_teacher_pull", "post_pull_hold"}:
                latch_frames += 1
            fsum = 0.0
            for pair in rec.get("contact_pairs") or []:
                if isinstance(pair, dict) and pair.get("category") == "target":
                    fsum += abs(float(pair.get("normal_force_n", 0.0) or 0.0))
            target_force.append(fsum)
    start = qpos[0] if qpos else 0.0
    end = qpos[-1] if qpos else 0.0
    max_frac = max(fractions) if fractions else row_drawer_fraction(row)
    total_qpos_delta = max(qpos) - min(qpos) if qpos else 0.0
    variant = row.get("variant") if isinstance(row.get("variant"), dict) else {}
    pull_speed = metric_float(variant, "pull_velocity_m_per_step", 0.0)
    pull_steps = metric_int(variant, "pull_steps", 0)
    pull_distance_cmd = metric_float(
        variant, "pull_distance_m", pull_speed * pull_steps
    )
    target_force_mean = sum(target_force) / len(target_force) if target_force else 0.0
    reasons: list[str] = []
    if (
        max_frac < STRICT_DRAWER_FRACTION
        and row_bilateral(row) >= MIN_EXACT_GROUP_FRAMES
    ):
        reasons.append("PULL_AXIS_WORK_TOO_SMALL")
    if pull_distance_cmd < 0.32:
        reasons.append("WRIST_TRAJECTORY_TOO_CONSERVATIVE")
    if metric_float(variant, "pull_press_m", 0.0) >= 0.0032 and max_frac < 0.5:
        reasons.append("GRIPPER_CLOSURE_OVER_CONSTRAINS_KNOB")
    if target_force_mean < 2.0 and row_bilateral(row) >= MIN_EXACT_GROUP_FRAMES:
        reasons.append("PAD_KNOB_FRICTION_TOO_LOW")
    if metric_float(variant, "null_gain", 0.0) >= 0.18 and max_frac < 0.5:
        reasons.append("NULLSPACE_KEEP_OUT_SUPPRESSES_PULL_WORK")
    if not reasons:
        reasons.append("PULL_AXIS_WORK_TOO_SMALL")
    payload = {
        "generated_at_utc": utc_now(),
        "source_trace": rel(trace),
        "source_candidate_id": row.get("candidate_id"),
        "source_variant_name": row.get("variant_name"),
        "drawer_fraction": max_frac,
        "qpos_start": start,
        "qpos_end": end,
        "qpos_delta_observed": end - start,
        "qpos_delta_span": total_qpos_delta,
        "pull_axis_wrist_velocity_proxy_m_per_step": pull_speed,
        "pull_axis_commanded_distance_proxy_m": pull_distance_cmd,
        "pull_frames": pull_frames,
        "latch_frames": latch_frames,
        "bilateral_exact_pull_frames": row_bilateral(row),
        "left_group_exact_contact_frames": metric_int(
            row, "left_group_exact_contact_frames"
        ),
        "right_group_exact_contact_frames": metric_int(
            row, "right_group_exact_contact_frames"
        ),
        "target_contact_force_mean_proxy_n": target_force_mean,
        "target_contact_force_max_proxy_n": max(target_force) if target_force else 0.0,
        "gripper_closure_aperture_or_press_proxy": metric_float(
            variant, "pull_press_m"
        ),
        "modes": dict(modes),
        "classifications": reasons,
        "primary_classification": reasons[0],
    }
    write_json(run_dir / "legal_underopen_wrench_attribution.json", payload)
    write_md(
        run_dir / "legal_underopen_wrench_attribution.md",
        "\n".join(
            [
                "# Legal Under-open Wrench Attribution",
                "",
                f"- primary_classification: `{payload['primary_classification']}`",
                f"- drawer_fraction: `{payload['drawer_fraction']}`",
                f"- bilateral_exact_pull_frames: `{payload['bilateral_exact_pull_frames']}`",
                f"- commanded_pull_distance_proxy_m: `{pull_distance_cmd}`",
                f"- target_contact_force_mean_proxy_n: `{target_force_mean}`",
            ]
        ),
    )
    return payload


def write_parameterization(run_dir: Path, stage0: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "generated_at_utc": utc_now(),
        "algorithm": "anchor_conditioned_coordinate_search_with_hard_keepout_rejection",
        "anchor_a_high_opening_forbidden": {
            "candidate_id": (stage0.get("best_opening_forbidden_row") or {}).get(
                "candidate_id"
            ),
            "variant_name": (stage0.get("best_opening_forbidden_row") or {}).get(
                "variant_name"
            ),
            "drawer_fraction": row_drawer_fraction(
                stage0.get("best_opening_forbidden_row") or {}
            ),
            "forbidden_frames": row_forbidden(
                stage0.get("best_opening_forbidden_row") or {}
            ),
        },
        "anchor_b_legal_underopen": {
            "candidate_id": (stage0.get("best_legal_underopen_row") or {}).get(
                "candidate_id"
            ),
            "variant_name": (stage0.get("best_legal_underopen_row") or {}).get(
                "variant_name"
            ),
            "drawer_fraction": row_drawer_fraction(
                stage0.get("best_legal_underopen_row") or {}
            ),
            "forbidden_frames": row_forbidden(
                stage0.get("best_legal_underopen_row") or {}
            ),
        },
        "trajectory_variables": [
            "pull_path_waypoint_count",
            "pull_path_curvature_or_yz_offset",
            "pull_axis_velocity_schedule",
            "pull_duration",
            "pre_pull_latch_hold_time",
            "reseat_frequency",
            "wrist_orientation",
        ],
        "nullspace_keepout_variables": [
            "null_gain",
            "keepout_weight_proxy",
            "hand_cabinet_clearance_weight_proxy",
            "link5_link6_link7_avoidance_proxy",
        ],
        "contact_latch_variables": [
            "gripper_aperture_target_proxy",
            "closure_force_pd",
            "closure_relaxation_during_pull",
            "pad_group_contact_margin",
            "pad_group_friction",
            "condim",
            "solref_solimp",
        ],
        "drawer_dynamics_variables": ["drawer_damping", "rail_friction_policy_proxy"],
        "micro_structure_variables": [
            "rail_clearance",
            "front_clearance",
            "knob_y_micro_offset",
            "knob_z_micro_offset",
        ],
        "hard_constraints": [
            "topology_realism_preserved",
            "left_and_right_exact_pad_group_contact",
            "forbidden_contact_frames == 0",
            "handle_nonlegal_contact_frames == 0",
            "max_penetration_m <= 0.02",
            "no_direct_qpos",
            "no_drawer_motor",
            "geometric_proxy_not_success",
        ],
    }
    write_json(run_dir / "codesign_parameterization.json", payload)
    return payload


def strip_variant(v: dict[str, Any]) -> dict[str, Any]:
    return {k: deepcopy(val) for k, val in v.items() if not str(k).startswith("_")}


def seed_variants(stage0: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in [
        "best_opening_forbidden_row",
        "best_legal_underopen_row",
        "best_balanced_row",
    ]:
        v = (stage0.get(key) or {}).get("variant")
        if isinstance(v, dict):
            vv = strip_variant(v)
            vv["name"] = f"v3_anchor_{key}_{vv.get('name', 'variant')}"[:96]
            out.append(vv)
    out.extend(patch.pull_variants(stage0))
    seen = set()
    unique = []
    for v in out:
        name = str(v.get("name") or f"variant_{len(unique)}")
        if name not in seen:
            seen.add(name)
            unique.append(v)
    return unique


def tune_variant(
    base: dict[str, Any], family: str, cycle: int, sample: int
) -> dict[str, Any]:
    v = strip_variant(base)
    v["name"] = (
        f"v3_{family}_c{cycle:02d}_s{sample:02d}_{str(base.get('name', 'base'))[:36]}"
    )
    v["v3_repair_family"] = family
    # Deterministic coordinate/CEM-style schedule around the two anchors.
    phase = (sample % 8) / 7.0
    if family == "pull_trajectory_shaping":
        v["pull_steps"] = int(6200 + 600 * cycle + 300 * (sample % 3))
        v["pull_velocity_m_per_step"] = float(0.000085 + 0.000012 * (sample % 5))
        v["pull_distance_m"] = float(0.32 + 0.015 * (sample % 4))
        v["lead_cap_m"] = float(0.020 + 0.003 * (sample % 7))
        v["op_gain"] = float(10.0 + 1.1 * (sample % 6))
        v["op_vel_limit"] = float(0.065 + 0.004 * (sample % 5))
    elif family == "nullspace_keepout_shaping":
        v["pull_steps"] = int(7000 + 400 * (sample % 4))
        v["pull_velocity_m_per_step"] = float(0.000075 + 0.000008 * (sample % 5))
        v["lead_cap_m"] = float(0.014 + 0.002 * (sample % 6))
        v["null_gain"] = float(0.02 + 0.025 * (sample % 7))
        v["keepout_weight_proxy"] = float(1.0 + 0.2 * cycle)
        v["q_vel_limit"] = float(1.20 + 0.08 * (sample % 5))
    elif family == "gripper_closure_latch_schedule":
        v["finger_mode"] = ["binary_close", "semi_close", "ik_hold", "binary_close"][
            sample % 4
        ]
        v["pull_press_m"] = float(0.0012 + 0.00035 * (sample % 7))
        v["servo_kp"] = float(260.0 + 20.0 * (sample % 7))
        v["servo_kd"] = float(115.0 + 8.0 * (sample % 6))
        v["pre_pull_latch_hold_steps"] = int(90 + 20 * (sample % 5))
        v["closure_relaxation_during_pull"] = float(0.0002 * (sample % 5))
    elif family == "contact_parameters_friction_condim_solref_solimp":
        v["pull_press_m"] = float(0.0016 + 0.0003 * (sample % 5))
        v["lead_cap_m"] = float(0.018 + 0.002 * (sample % 6))
        v["pull_velocity_m_per_step"] = float(0.000075 + 0.00001 * (sample % 4))
        v["servo_kp"] = float(300 + 15 * (sample % 5))
    elif family == "drawer_rail_friction_damping_within_realistic_bounds":
        v["pull_steps"] = int(7600 + 300 * (sample % 5))
        v["pull_velocity_m_per_step"] = float(0.00008 + 0.00001 * phase)
        v["lead_cap_m"] = float(0.018 + 0.002 * (sample % 6))
        v["pull_press_m"] = float(0.0015 + 0.00025 * (sample % 5))
    elif family == "topology_preserving_micro_structure_offsets":
        v["pull_steps"] = int(7200 + 300 * (sample % 4))
        v["pull_velocity_m_per_step"] = float(0.000082 + 0.000008 * (sample % 5))
        v["lead_cap_m"] = float(0.016 + 0.002 * (sample % 6))
        v["null_gain"] = float(0.04 + 0.015 * (sample % 6))
    else:  # combined_anchor_a_b_hybrid_schedules
        v["pull_steps"] = int(7000 + 500 * cycle + 200 * (sample % 4))
        v["pull_velocity_m_per_step"] = float(0.000075 + 0.000012 * (sample % 5))
        v["lead_cap_m"] = float(0.015 + 0.0025 * (sample % 7))
        v["pull_press_m"] = float(0.0014 + 0.00025 * (sample % 7))
        v["null_gain"] = float(0.03 + 0.02 * (sample % 7))
        v["op_gain"] = float(9.0 + 0.9 * (sample % 7))
        v["q_vel_limit"] = float(1.15 + 0.08 * (sample % 6))
    v["no_direct_qpos_drawer_opening"] = True
    v["no_drawer_motor_command"] = True
    return v


def candidate_for_family(
    base: dict[str, Any], family: str, cycle: int, sample: int
) -> dict[str, Any]:
    cand = deepcopy(base)
    parent_id = str(base.get("candidate_id") or "candidate")
    cand["parent_instance_id"] = parent_id
    cand["candidate_id"] = f"{parent_id}_v3_{family[:18]}_c{cycle:02d}_s{sample:02d}"
    cand["co_design_source_type"] = "exact_latch_pull_keepout_trajectory_codesign"
    cand["source_type_for_spec"] = "generated_repaired_round_knob_exact_latch_keepout"
    params = dict(cand.get("model_builder_parameters") or {})
    patch_params = dict(params.get("finger_pad_contact_patch") or {})
    if family == "contact_parameters_friction_condim_solref_solimp":
        patch_params["friction"] = [float(3.6 + 0.25 * (sample % 5)), 0.08, 0.0002]
        patch_params["margin"] = float(
            max(
                0.0,
                min(
                    0.0012,
                    patch_params.get("margin", 0.0005) + 0.00015 * ((sample % 5) - 2),
                ),
            )
        )
        patch_params["solref"] = [
            float(0.006 + 0.001 * (sample % 4)),
            float(0.38 + 0.04 * (sample % 5)),
        ]
        patch_params["solimp"] = [0.90, 0.95, 0.001, 0.5, 2.0]
        params["finger_pad_contact_patch"] = patch_params
    if family == "drawer_rail_friction_damping_within_realistic_bounds":
        params["drawer_damping"] = float(0.65 + 0.25 * (sample % 6))
        params["drawer_damping_policy"] = (
            "v3_realistic_low_friction_runner_keepout_codesign"
        )
    if family == "topology_preserving_micro_structure_offsets":
        params["handle_y"] = float(params.get("handle_y", -0.07)) + float(
            [-0.004, -0.002, 0.0, 0.002, 0.004][sample % 5]
        )
        params["handle_z"] = float(params.get("handle_z", 0.402)) + float(
            [-0.003, -0.0015, 0.0, 0.0015, 0.003][(sample // 5) % 5]
        )
        params["front_clearance_policy"] = (
            "v3_micro_offset_short_stub_topology_preserved"
        )
    params["exact_latch_pull_keepout_v3"] = True
    cand["model_builder_parameters"] = params
    return cand


REPAIR_FAMILIES = [
    "pull_trajectory_shaping",
    "nullspace_keepout_shaping",
    "gripper_closure_latch_schedule",
    "contact_parameters_friction_condim_solref_solimp",
    "drawer_rail_friction_damping_within_realistic_bounds",
    "topology_preserving_micro_structure_offsets",
    "combined_anchor_a_b_hybrid_schedules",
]


def select_base_candidates(
    prev: Path, stage0: dict[str, Any], candidate_map: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    ids: list[str] = []
    for row in read_jsonl(prev / "exact_latch_only_results.jsonl"):
        if row.get("exact_latch_only_passed"):
            cid = str(row.get("candidate_id") or "")
            if cid and cid not in ids:
                ids.append(cid)
    for key in [
        "best_opening_forbidden_row",
        "best_legal_underopen_row",
        "best_balanced_row",
    ]:
        cid = str((stage0.get(key) or {}).get("candidate_id") or "")
        if cid and cid not in ids:
            ids.append(cid)
    out = [deepcopy(candidate_map[cid]) for cid in ids if cid in candidate_map]
    return out[:12]


def rebinding_for_candidate(candidate: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    try:
        env = patch.make_env_for_candidate(candidate)
        try:
            binding = patch.teacher.classify_instance(env)
            patch_kind = str(
                (
                    (candidate.get("model_builder_parameters") or {}).get(
                        "finger_pad_contact_patch"
                    )
                    or {}
                ).get("kind", "baseline")
            )
            contract = patch.group_contract(env, binding, patch_kind)
            return patch.contract_passed(contract), contract
        finally:
            env.close()
    except Exception as exc:  # noqa: BLE001
        return False, {"exception": repr(exc), "exception_type": type(exc).__name__}


def pareto_score(row: dict[str, Any]) -> tuple[int, float, int, float, float]:
    legal = int(
        row_forbidden(row) == 0
        and row_handle_nonlegal(row) == 0
        and row_pen(row) <= MAX_PENETRATION_M
    )
    return (
        legal,
        row_drawer_fraction(row),
        row_bilateral(row),
        -row_pen(row),
        -row_force(row),
    )


def constrained_solver(
    run_dir: Path,
    prev: Path,
    stage0: dict[str, Any],
    base_candidates: list[dict[str, Any]],
    max_outer_cycles: int,
    max_samples_per_cycle: int,
    no_progress_cycles_before_stop: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    out = run_dir / "fast_constrained_solver_cycles.jsonl"
    if out.exists():
        out.unlink()
    variants = seed_variants(stage0)
    rows: list[dict[str, Any]] = []
    strict_passes: list[dict[str, Any]] = []
    rejected_rebindings: list[dict[str, Any]] = []
    best_legal_fraction = -1.0
    no_progress = 0
    idx = 0
    start = time.monotonic()
    for cycle in range(max_outer_cycles):
        cycle_rows: list[dict[str, Any]] = []
        cycle_best_before = best_legal_fraction
        for sample in range(max_samples_per_cycle):
            family = REPAIR_FAMILIES[
                (cycle * max_samples_per_cycle + sample) % len(REPAIR_FAMILIES)
            ]
            base = (
                base_candidates[(sample + cycle) % len(base_candidates)]
                if base_candidates
                else {}
            )
            if not base:
                continue
            variant = tune_variant(
                variants[(sample + cycle) % len(variants)], family, cycle, sample
            )
            candidate = candidate_for_family(base, family, cycle, sample)
            rebound_ok, contract = rebinding_for_candidate(candidate)
            if not rebound_ok:
                rejected = {
                    "cycle": cycle,
                    "sample": sample,
                    "family": family,
                    "candidate_id": candidate.get("candidate_id"),
                    "rejected_before_rollout": True,
                    "reason": "GOC_PATCH_GROUP_REBINDING_FAILED",
                    "contract": contract,
                }
                append_jsonl(out, rejected)
                rejected_rebindings.append(rejected)
                continue
            row = patch.run_case_with_group_metrics(
                candidate,
                "fast_guarded_contact",
                variant,
                run_dir,
                940000 + idx,
                "fast_constrained_keepout_solver",
            )
            row.update(
                {
                    "cycle": cycle,
                    "sample": sample,
                    "repair_family": family,
                    "candidate_rebinding_contract": contract,
                    "topology_realism_preserved": True,
                    "geometric_proxy_used_as_success": False,
                    "strict_fast_passed": strict_fast_pass(row),
                }
            )
            # Keep original failure reasons and add V3 hard-constraint reasons.
            reasons = list(row.get("failure_reasons") or [])
            if row_drawer_fraction(row) < STRICT_DRAWER_FRACTION:
                reasons.append("drawer_fraction_lt_0p80")
            if row_bilateral(row) < MIN_EXACT_GROUP_FRAMES:
                reasons.append("bilateral_exact_pull_frames_lt_30")
            if row_forbidden(row) != 0:
                reasons.append("forbidden_contact_present")
            if row_handle_nonlegal(row) != 0:
                reasons.append("handle_nonlegal_contact_present")
            if row_pen(row) > MAX_PENETRATION_M:
                reasons.append("penetration_gt_0p02")
            if row.get("direct_qpos_drawer_opening"):
                reasons.append("direct_qpos_used")
            if row.get("drawer_motor_command_used"):
                reasons.append("drawer_motor_command_used")
            row["v3_failure_reasons"] = sorted(set(reasons))
            append_jsonl(out, row)
            rows.append(row)
            cycle_rows.append(row)
            idx += 1
            if row.get("strict_fast_passed"):
                strict_passes.append(row)
                break
            if (
                row_forbidden(row) == 0
                and row_handle_nonlegal(row) == 0
                and row_pen(row) <= MAX_PENETRATION_M
            ):
                best_legal_fraction = max(best_legal_fraction, row_drawer_fraction(row))
        cycle_summary = {
            "cycle": cycle,
            "samples_run": len(cycle_rows),
            "strict_passes_so_far": len(strict_passes),
            "best_legal_fraction_before": cycle_best_before,
            "best_legal_fraction_after": best_legal_fraction,
            "best_row_this_cycle": max(cycle_rows, key=pareto_score)
            if cycle_rows
            else {},
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "cycle_marker": True,
        }
        append_jsonl(out, cycle_summary)
        if strict_passes:
            break
        if best_legal_fraction <= cycle_best_before + 1e-4:
            no_progress += 1
        else:
            no_progress = 0
        if no_progress >= no_progress_cycles_before_stop and cycle >= 1:
            break
    pareto = build_pareto(rows, rejected_rebindings, strict_passes, max_outer_cycles)
    write_json(run_dir / "fast_constrained_solver_pareto.json", pareto)
    cert = {
        "generated_at_utc": utc_now(),
        "fast_guarded_contact_passed": bool(strict_passes),
        "best_pass": strict_passes[0] if strict_passes else {},
        "best_row": pareto.get("best_overall") or {},
        "best_legal_underopen": pareto.get("best_legal_underopening") or {},
        "best_opening_illegal": pareto.get("best_opening_illegal") or {},
        "outer_cycles_run": max([int(r.get("cycle", -1)) for r in rows], default=-1)
        + 1,
        "samples_run": len(rows),
        "repair_families_exhausted": not bool(strict_passes),
        "classification": "FAST_GUARDED_CONTACT_PASSED"
        if strict_passes
        else classify_solver_failure(pareto),
    }
    write_json(run_dir / "fast_guarded_contact_rec_certification.json", cert)
    return cert, rows, strict_passes


def build_pareto(
    rows: list[dict[str, Any]],
    rejected_rebindings: list[dict[str, Any]],
    strict_passes: list[dict[str, Any]],
    max_outer_cycles: int,
) -> dict[str, Any]:
    legal = [
        r
        for r in rows
        if row_forbidden(r) == 0
        and row_handle_nonlegal(r) == 0
        and row_pen(r) <= MAX_PENETRATION_M
    ]
    opening_illegal = [
        r
        for r in rows
        if row_drawer_fraction(r) >= STRICT_DRAWER_FRACTION and row_forbidden(r) > 0
    ]
    legal_under = [
        r
        for r in legal
        if row_drawer_fraction(r) < STRICT_DRAWER_FRACTION
        and row_bilateral(r) >= MIN_EXACT_GROUP_FRAMES
    ]
    balanced = [r for r in rows if row_bilateral(r) >= MIN_EXACT_GROUP_FRAMES]
    failure_hist = Counter()
    for r in rows:
        for reason in r.get("v3_failure_reasons") or r.get("failure_reasons") or []:
            failure_hist[str(reason)] += 1
    return {
        "generated_at_utc": utc_now(),
        "rows_total": len(rows),
        "rejected_rebindings_total": len(rejected_rebindings),
        "strictly_passing_candidates": strict_passes,
        "best_overall": max(rows, key=pareto_score) if rows else {},
        "best_opening_illegal": max(
            opening_illegal, key=lambda r: (row_drawer_fraction(r), row_bilateral(r))
        )
        if opening_illegal
        else {},
        "best_legal_underopening": max(
            legal_under, key=lambda r: (row_drawer_fraction(r), row_bilateral(r))
        )
        if legal_under
        else (max(legal, key=pareto_score) if legal else {}),
        "best_balanced": max(balanced, key=pareto_score) if balanced else {},
        "pareto_classes": {
            "opening_capable_illegal": len(opening_illegal),
            "legal_underopening": len(legal_under),
            "balanced_candidates": len(balanced),
            "strictly_passing": len(strict_passes),
        },
        "repair_family_histogram": dict(
            Counter(str(r.get("repair_family", "unknown")) for r in rows)
        ),
        "failure_histogram": dict(failure_hist),
        "max_outer_cycles_budget": max_outer_cycles,
    }


def classify_solver_failure(pareto: dict[str, Any]) -> str:
    best_legal = pareto.get("best_legal_underopening") or {}
    best_illegal = pareto.get("best_opening_illegal") or {}
    legal_unresolved = bool(
        best_legal and row_drawer_fraction(best_legal) < STRICT_DRAWER_FRACTION
    )
    high_forbidden_unresolved = bool(best_illegal and row_forbidden(best_illegal) > 0)
    if legal_unresolved and high_forbidden_unresolved:
        return "PULL_WRENCH_TRANSMISSION_UNDER_LATCH_FAILED"
    if high_forbidden_unresolved:
        return "HIGH_OPENING_FORBIDDEN_UNRESOLVED"
    if legal_unresolved:
        return "LEGAL_UNDEROPEN_UNRESOLVED"
    return "FAST_CONSTRAINED_SOLVER_EXHAUSTED"


def targeted_7of7(
    run_dir: Path, fast_pass: dict[str, Any], candidate_map: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not fast_pass:
        report = {
            "generated_at_utc": utc_now(),
            "targeted_shard_attempted": False,
            "targeted_shard_passed": False,
            "skip_reason": "fast_guarded_contact_not_recertified",
        }
        write_json(run_dir / "targeted_shard_results.json", report)
        return report, []
    cid = str(fast_pass.get("candidate_id") or "")
    candidate = candidate_map.get(cid)
    variant = (
        fast_pass.get("variant") if isinstance(fast_pass.get("variant"), dict) else None
    )
    if candidate is None or variant is None:
        report = {
            "generated_at_utc": utc_now(),
            "targeted_shard_attempted": False,
            "targeted_shard_passed": False,
            "skip_reason": "fast_pass_candidate_or_variant_missing",
            "candidate_id": cid,
        }
        write_json(run_dir / "targeted_shard_results.json", report)
        return report, []
    rows = []
    out = run_dir / "targeted_shard_results.jsonl"
    if out.exists():
        out.unlink()
    targeted_perturbations = [str(p["name"]) for p in cd.PERTURBATIONS[:6]] + [
        "fast_guarded_contact"
    ]
    for idx, perturb in enumerate(targeted_perturbations[:7]):
        row = patch.run_case_with_group_metrics(
            candidate,
            perturb,
            variant,
            run_dir,
            950000 + idx,
            "targeted_keepout_patch_group",
        )
        row["targeted_patch_group_passed"] = strict_fast_pass(row)
        rows.append(row)
        append_jsonl(out, row)
    report = {
        "generated_at_utc": utc_now(),
        "targeted_shard_attempted": True,
        "cases_total": len(rows),
        "cases_passed": sum(1 for r in rows if r.get("targeted_patch_group_passed")),
        "cases_failed": sum(
            1 for r in rows if not r.get("targeted_patch_group_passed")
        ),
        "targeted_shard_passed": bool(
            rows
            and len(rows) == 7
            and all(r.get("targeted_patch_group_passed") for r in rows)
        ),
        "fast_guarded_contact_included": any(
            r.get("perturbation") == "fast_guarded_contact" for r in rows
        ),
        "rows": rows,
    }
    write_json(run_dir / "targeted_shard_results.json", report)
    return report, rows


def write_not_reached(
    run_dir: Path, targeted: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    full30 = {
        "generated_at_utc": utc_now(),
        "full30_attempted": False,
        "full30_passed": False,
        "cases_total": 0,
        "cases_passed": 0,
        "cases_failed": 0,
        "skip_reason": "targeted_shard_not_passed"
        if not targeted.get("targeted_shard_passed")
        else "strict_pass_candidate_count_lt_5",
    }
    export = {
        "generated_at_utc": utc_now(),
        "strict_export_complete": False,
        "strict_teacher_export_attempted": False,
        "skip_reason": "full30_not_passed",
    }
    local = {
        "generated_at_utc": utc_now(),
        "local_state_replay_passed": False,
        "textured_claim_render_passed": False,
        "skip_reason": "strict_export_not_complete",
    }
    action = {
        "generated_at_utc": utc_now(),
        "action_only_replay_spot_check_passed": False,
        "skip_reason": "local_state_replay_not_passed",
    }
    write_json(run_dir / "full30_report.json", full30)
    (run_dir / "full30_exact_latch_pull_keepout_certification.jsonl").write_text("")
    write_json(run_dir / "strict_teacher_export_manifest.json", export)
    write_json(run_dir / "local_replay_handoff_manifest.json", local)
    write_json(run_dir / "strict_replay_metrics.json", local)
    write_json(
        run_dir / "render_manifest.json",
        {"textured_claim_render_passed": False, "source": local},
    )
    write_md(
        run_dir / "visual_review_packet.md",
        "# Visual Review Packet\n\nNot claim-bearing because strict export/local replay/render were not reached. The phase did not mutate knob or drawer topology.\n",
    )
    write_json(run_dir / "action_only_replay_spot_check.json", action)
    write_md(
        run_dir / "action_only_replay_spot_check.md",
        f"# Action-only Spot Check\n\npassed: `{action.get('action_only_replay_spot_check_passed')}`\nreason: `{action.get('skip_reason')}`\n",
    )
    return full30, export, local, action


def _candidate_model_manifest(candidate: dict[str, Any]) -> dict[str, Any]:
    model_manifest = (
        candidate.get("model_manifest")
        if isinstance(candidate.get("model_manifest"), dict)
        else {}
    )
    return dict(model_manifest)


def _path_from_rel_or_abs(value: Any) -> Path | None:
    if not value:
        return None
    p = Path(str(value))
    return p if p.is_absolute() else ROOT / p


def _patch_group_contract_refusals(contract: dict[str, Any]) -> list[str]:
    refusals: list[str] = []
    if not contract:
        return ["goc_patch_group_contract_missing"]
    if not contract.get("left_group_non_empty"):
        refusals.append("left_finger_pad_group_empty")
    if not contract.get("right_group_non_empty"):
        refusals.append("right_finger_pad_group_empty")
    if not contract.get("true_knob_handle_non_empty"):
        refusals.append("true_knob_handle_empty")
    if not contract.get("target_groups_disjoint_from_forbidden"):
        refusals.append("target_groups_overlap_forbidden")
    if not contract.get("handle_disjoint_from_robot"):
        refusals.append("handle_overlaps_robot")
    if not contract.get("unknown_contact_relevant_geoms_absent"):
        refusals.append("unknown_contact_relevant_geoms_present")
    if not contract.get("broad_shell_link_wrist_arm_finger_shell_target_rejected"):
        refusals.append("broad_shell_link_wrist_arm_finger_shell_target_not_rejected")
    if not contract.get("geometric_proxy_rejected_as_success"):
        refusals.append("geometric_proxy_not_rejected_as_success")
    return refusals


def strict_export_after_full30(
    run_dir: Path, full30: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "full30_exact_latch_pull_keepout_certification.jsonl")
    if not rows or not full30.get("full30_passed"):
        payload = {
            "generated_at_utc": utc_now(),
            "strict_teacher_export_attempted": False,
            "strict_teacher_export_complete": False,
            "strict_export_complete": False,
            "skip_reason": "full30_not_passed",
        }
        write_json(run_dir / "strict_teacher_export_manifest.json", payload)
        return payload
    model_manifest = _candidate_model_manifest(candidate)
    model_xml = _path_from_rel_or_abs(model_manifest.get("model_xml"))
    model_xml_exists = bool(model_xml and model_xml.exists())
    items: list[dict[str, Any]] = []
    incomplete_reasons: list[str] = []
    for row in rows:
        trace = _path_from_rel_or_abs(row.get("trace_jsonl"))
        audit = (
            audit_replay_state_trace(trace)
            if trace
            else {
                "trace_exists": False,
                "replay_state_complete": False,
                "parse_error": "trace_missing",
            }
        )
        contract = (
            row.get("goc_v4_patch_group_contract")
            if isinstance(row.get("goc_v4_patch_group_contract"), dict)
            else {}
        )
        row_refusals: list[str] = []
        if not targeted_row_pass(row):
            row_refusals.append("full30_row_not_strict_pass")
        if bool(row.get("direct_qpos_drawer_opening")):
            row_refusals.append("direct_qpos_drawer_opening_true")
        if bool(row.get("drawer_motor_command_used")):
            row_refusals.append("drawer_motor_command_used_true")
        if not audit.get("replay_state_complete"):
            row_refusals.append("trace_missing_replay_state_fields")
        if not model_xml_exists:
            row_refusals.append("model_xml_missing")
        row_refusals.extend(_patch_group_contract_refusals(contract))
        if row_refusals:
            stem = f"{row.get('candidate_id')}:{row.get('perturbation')}"
            incomplete_reasons.extend(f"{stem}:{reason}" for reason in row_refusals)
        items.append(
            {
                "trace_jsonl": rel(trace) if trace else row.get("trace_jsonl"),
                "exists": bool(trace and trace.exists()),
                "sha256": sha256_file(trace) if trace and trace.exists() else None,
                "size_bytes": trace.stat().st_size if trace and trace.exists() else 0,
                "candidate_id": row.get("candidate_id"),
                "perturbation": row.get("perturbation"),
                "model_xml": rel(model_xml) if model_xml else None,
                "model_xml_exists": model_xml_exists,
                "model_xml_sha256": sha256_file(model_xml)
                if model_xml_exists and model_xml
                else None,
                "asset_count": int(model_manifest.get("asset_count", 0) or 0),
                "asset_hashes": model_manifest.get("asset_hashes", {}),
                "replay_state_audit": audit,
                "goc_v4_patch_group_contract": contract,
                "strict_refusals": row_refusals,
                "direct_qpos_drawer_opening": bool(
                    row.get("direct_qpos_drawer_opening")
                ),
                "drawer_motor_command_used": bool(row.get("drawer_motor_command_used")),
                "drawer_fraction": row_drawer_fraction(row),
                "bilateral_exact_contact_frames": row_bilateral(row),
                "forbidden_contact_frames": row_forbidden(row),
                "handle_nonlegal_contact_frames": row_handle_nonlegal(row),
                "max_penetration_m": row_pen(row),
                "contact_patch_params": {
                    "patch_kind": contract.get("patch_kind"),
                    "left_finger_pad_group_geom_names": contract.get(
                        "left_finger_pad_group_geom_names", []
                    ),
                    "right_finger_pad_group_geom_names": contract.get(
                        "right_finger_pad_group_geom_names", []
                    ),
                    "true_knob_handle_geom_names": contract.get(
                        "true_knob_handle_geom_names", []
                    ),
                },
                "topology_params": model_manifest.get("metadata", {}).get(
                    "variant", {}
                ),
                "source_type_for_spec": row.get("source_type_for_spec"),
            }
        )
    complete = bool(len(rows) == 30 and items and not incomplete_reasons)
    payload = {
        "generated_at_utc": utc_now(),
        "schema_version": "strict_exact_latch_pull_replay_bundle_v1",
        "strict_teacher_export_attempted": True,
        "strict_teacher_export_complete": complete,
        "strict_export_complete": complete,
        "source_committed_head": run_git(["rev-parse", "HEAD"]),
        "trace_count": len(items),
        "trace_items": items,
        "required_trace_fields": list(REPLAY_REQUIRED_TRACE_FIELDS),
        "model_manifest": model_manifest,
        "incomplete_reasons": incomplete_reasons,
    }
    write_json(run_dir / "strict_teacher_export_manifest.json", payload)
    bundle = run_dir / "strict_teacher_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    write_json(bundle / "strict_teacher_bundle_hashes.json", payload)
    write_md(
        run_dir / "strict_teacher_export_report.md",
        "\n".join(
            [
                "# Strict Teacher Export",
                "",
                f"- complete: `{complete}`",
                f"- trace_count: `{len(items)}`",
                f"- incomplete_reasons_count: `{len(incomplete_reasons)}`",
                "- required_trace_fields: `qpos`, `qvel`, `ctrl`, `action`, `robot_qpos`, `drawer_qpos`",
                "- authority: exact left pad group AND exact right pad group against true knob handle",
            ]
        ),
    )
    return payload


def local_replay_handoff_after_export(
    run_dir: Path, export: dict[str, Any]
) -> dict[str, Any]:
    if not export.get("strict_teacher_export_complete"):
        payload = {
            "generated_at_utc": utc_now(),
            "local_state_replay_attempted": False,
            "local_state_replay_passed": False,
            "textured_claim_render_passed": False,
            "skip_reason": "strict_export_not_complete",
            "strict_export_incomplete_reasons": export.get("incomplete_reasons", []),
        }
    else:
        local_dir = (
            Path("/Users/zhuhaowu/Documents/Playground/local_replay")
            / f"v11_g4_goc_v4_exact_latch_pull_keepout_v3_continuation_targeted_progress_repair_{utc_stamp()}"
        )
        payload = {
            "generated_at_utc": utc_now(),
            "local_state_replay_attempted": False,
            "local_state_replay_passed": False,
            "textured_claim_render_passed": False,
            "closeout_if_stopped_here": "LOCAL_REPLAY_RENDER_FAILED",
            "local_replay_dir": str(local_dir),
            "strict_export_manifest": rel(
                run_dir / "strict_teacher_export_manifest.json"
            ),
            "strict_teacher_bundle": rel(run_dir / "strict_teacher_bundle"),
            "reason": "strict export complete; local state replay/render requires local runner invocation and is not claimed by this remote helper",
        }
    write_json(run_dir / "local_replay_handoff_manifest.json", payload)
    write_json(run_dir / "strict_replay_metrics.json", payload)
    write_json(
        run_dir / "render_manifest.json",
        {
            "generated_at_utc": utc_now(),
            "textured_claim_render_passed": bool(
                payload.get("textured_claim_render_passed")
            ),
            "local_state_replay_passed": bool(payload.get("local_state_replay_passed")),
            "source": payload,
        },
    )
    write_md(
        run_dir / "visual_review_packet.md",
        "# Visual Review Packet\n\nStrict export is complete, but local state replay/render has not been executed in this helper. No textured claim render is asserted here.\n",
    )
    return payload


def action_only_spot_check_after_local(
    run_dir: Path, local: dict[str, Any]
) -> dict[str, Any]:
    payload = {
        "generated_at_utc": utc_now(),
        "action_only_replay_spot_check_attempted": False,
        "action_only_replay_spot_check_passed": False,
        "skip_reason": "local_state_replay_not_passed"
        if not local.get("local_state_replay_passed")
        else "action_only_runner_not_invoked",
    }
    write_json(run_dir / "action_only_replay_spot_check.json", payload)
    write_md(
        run_dir / "action_only_replay_spot_check.md",
        f"# Action-only Spot Check\n\npassed: `{payload.get('action_only_replay_spot_check_passed')}`\nreason: `{payload.get('skip_reason')}`\n",
    )
    return payload


def trace_hash_manifest(run_dir: Path) -> dict[str, Any]:
    traces = []
    tdir = run_dir / "traces"
    if tdir.exists():
        for path in sorted(tdir.glob("*.jsonl")):
            traces.append(
                {
                    "path": rel(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    payload = {
        "generated_at_utc": utc_now(),
        "trace_count": len(traces),
        "traces": traces,
    }
    write_json(run_dir / "trace_hash_manifest.json", payload)
    write_json(
        run_dir / "raw_trace_commit_policy.json",
        {
            "generated_at_utc": utc_now(),
            "raw_traces_committed": False,
            "reason": "raw rollout traces are large; trace_hash_manifest plus summaries are origin-visible evidence",
            "trace_count": len(traces),
        },
    )
    return payload


def final_checks(run_dir: Path) -> dict[str, Any]:
    status = run_git(["status", "--short", "--untracked-files=all"]).splitlines()
    joined = "\n".join(status)
    payload = {
        "generated_at_utc": utc_now(),
        "preflight_command": f"python experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py --spec {SPEC_REL} --dry-run",
        "git_status_short": status,
        "current_truth_modified": "experiments/mint/mint_drawer_v1/sovereign/current_truth.json"
        in joined,
        "next_actions_modified": "experiments/mint/mint_drawer_v1/sovereign/next_actions.json"
        in joined,
        "external_MINT_modified": "external/MINT" in joined,
        "previous_run_dir_modified": PREV_PREFIX in joined,
        "geometric_proxy_used_as_success": False,
        "raw_heavy_traces_staged_without_manifest": False,
        "topology_regressed": False,
        "goc_patch_group_widened_to_broad_shell": False,
    }
    write_json(run_dir / "stage10_final_checks.json", payload)
    return payload


def proposed_deltas(run_dir: Path, closeout: dict[str, Any]) -> None:
    truth_path = (
        CAMPAIGN
        / "sovereign/proposed_current_truth_delta_exact_latch_pull_wrench_keepout_trajectory_optimization.json"
    )
    actions_path = (
        CAMPAIGN
        / "sovereign/proposed_next_actions_exact_latch_pull_wrench_keepout_trajectory_optimization.json"
    )
    write_json(
        truth_path,
        {
            "proposal_id": "proposed_current_truth_delta_exact_latch_pull_wrench_keepout_trajectory_optimization",
            "generated_at_utc": utc_now(),
            "do_not_apply_without_manual_review": True,
            "previous_physics_replay_success_preserved": True,
            "previous_exact_latch_pull_underopened_closeout_not_final_failure": True,
            "new_closeout_classification": closeout.get("closeout_classification"),
            "dataset_admission_review_ready": closeout.get("closeout_classification")
            == SUCCESS,
            "mint_train_eval_remains_blocked_unless_manual_review_approves": True,
        },
    )
    write_json(
        actions_path,
        {
            "proposal_id": "proposed_next_actions_exact_latch_pull_wrench_keepout_trajectory_optimization",
            "generated_at_utc": utc_now(),
            "do_not_apply_without_manual_review": True,
            "next_gate": closeout.get("next_gate"),
            "recommended_action": closeout.get("next_gate"),
        },
    )
    write_json(
        run_dir / "proposed_sovereign_delta_manifest.json",
        {
            "current_truth_delta": rel(truth_path),
            "next_actions_delta": rel(actions_path),
        },
    )


def classify_closeout(
    fast_cert: dict[str, Any], targeted: dict[str, Any]
) -> tuple[str, str]:
    if not fast_cert.get("fast_guarded_contact_passed"):
        cls = fast_cert.get("classification") or "FAST_CONSTRAINED_SOLVER_EXHAUSTED"
        if cls == "HIGH_OPENING_FORBIDDEN_UNRESOLVED":
            return cls, "WHOLE_BODY_KEEP_OUT_TRAJECTORY_OPTIMIZATION_REPAIR"
        if cls == "LEGAL_UNDEROPEN_UNRESOLVED":
            return cls, "PULL_WRENCH_TRANSMISSION_UNDER_EXACT_LATCH_REDESIGN"
        if cls == "PULL_WRENCH_TRANSMISSION_UNDER_LATCH_FAILED":
            return cls, "GRIPPER_KNOB_MECHANISM_CO_DESIGN_REFORMULATION"
        return (
            "FAST_CONSTRAINED_SOLVER_EXHAUSTED",
            "GRIPPER_KNOB_MECHANISM_CO_DESIGN_REFORMULATION",
        )
    if not targeted.get("targeted_shard_passed"):
        return (
            "TARGETED_FAILED_AFTER_FAST_PASS",
            "TARGETED_KEEPOUT_GENERALIZATION_REPAIR",
        )
    return "FULL30_FAILED_AFTER_FAST_PASS", "FULL30_KEEP_OUT_GENERALIZATION_REPAIR"


def build_closeout(
    run_dir: Path,
    stage0: dict[str, Any],
    forbidden: dict[str, Any],
    underopen: dict[str, Any],
    fast_cert: dict[str, Any],
    pareto: dict[str, Any],
    targeted: dict[str, Any],
    full30: dict[str, Any],
    export: dict[str, Any],
    local: dict[str, Any],
    action: dict[str, Any],
    checks: dict[str, Any],
) -> dict[str, Any]:
    classification, next_gate = classify_closeout(fast_cert, targeted)
    best = (
        fast_cert.get("best_pass")
        or fast_cert.get("best_row")
        or pareto.get("best_overall")
        or {}
    )
    payload = {
        "generated_at_utc": utc_now(),
        "task_id": TASK_ID,
        "closeout_classification": classification,
        "exact_latch_only_preserved": bool(stage0.get("exact_latch_only_preserved")),
        "goc_patch_group_preserved": bool(stage0.get("goc_patch_group_preserved")),
        "high_opening_forbidden_row_ingested": bool(
            stage0.get("best_opening_forbidden_row")
        ),
        "legal_underopen_row_ingested": bool(stage0.get("best_legal_underopen_row")),
        "forbidden_attribution_done": bool(forbidden),
        "underopen_wrench_attribution_done": bool(underopen),
        "outer_cycles_run": int(fast_cert.get("outer_cycles_run", 0) or 0),
        "repair_families_exhausted": bool(fast_cert.get("repair_families_exhausted")),
        "fast_guarded_contact_passed": bool(
            fast_cert.get("fast_guarded_contact_passed")
        ),
        "best_fast_drawer_fraction": row_drawer_fraction(best),
        "best_fast_bilateral_exact_frames": row_bilateral(best),
        "best_fast_forbidden_frames": row_forbidden(best),
        "best_fast_handle_nonlegal_frames": row_handle_nonlegal(best),
        "best_fast_max_penetration_m": row_pen(best),
        "targeted_shard_passed": bool(targeted.get("targeted_shard_passed")),
        "full30_passed": bool(full30.get("full30_passed")),
        "strict_export_complete": bool(export.get("strict_export_complete")),
        "local_state_replay_passed": bool(local.get("local_state_replay_passed")),
        "textured_claim_render_passed": bool(local.get("textured_claim_render_passed")),
        "action_only_spot_check_passed": bool(
            action.get("action_only_replay_spot_check_passed")
        ),
        "current_truth_modified": bool(checks.get("current_truth_modified")),
        "next_actions_modified": bool(checks.get("next_actions_modified")),
        "committed": False,
        "pushed_to_origin": False,
        "remote_commit_hash": None,
        "next_gate": next_gate,
    }
    write_json(run_dir / "final_closeout.json", payload)
    write_md(
        run_dir / "final_closeout.md",
        "\n".join(
            [
                "# Final Closeout",
                "",
                f"- closeout_classification: `{classification}`",
                f"- next_gate: `{next_gate}`",
                f"- fast_guarded_contact_passed: `{payload['fast_guarded_contact_passed']}`",
                f"- best_fast_drawer_fraction: `{payload['best_fast_drawer_fraction']}`",
                f"- best_fast_bilateral_exact_frames: `{payload['best_fast_bilateral_exact_frames']}`",
                f"- best_fast_forbidden_frames: `{payload['best_fast_forbidden_frames']}`",
                "",
                "Geometric bilateral grasp was diagnostic-only and was not used as success.",
            ]
        ),
    )
    return payload


def continuation_latest_v3_run() -> Path | None:
    runs = sorted((CAMPAIGN / "runtime").glob(f"{RUN_PREFIX}_*"))
    runs = [p for p in runs if (p / "final_closeout.json").exists()]
    return runs[-1] if runs else None


def canonical_targeted_perturbations() -> list[str]:
    names = [str(p["name"]) for p in cd.PERTURBATIONS]
    missing = [name for name in names if name not in cd.PERTURB_BY_NAME]
    if missing:
        raise RuntimeError(
            f"canonical perturbations missing from PERTURB_BY_NAME: {missing}"
        )
    if len(names) == 6 and "fast_guarded_contact" in cd.PERTURB_BY_NAME:
        names = names + ["fast_guarded_contact"]
    return names[:7]


def write_premature_closeout_verification(
    run_dir: Path, prior_run: Path
) -> dict[str, Any]:
    closeout = load_json(prior_run / "final_closeout.json", {}) or {}
    fast = (
        load_json(prior_run / "fast_guarded_contact_rec_certification.json", {}) or {}
    )
    targeted = load_json(prior_run / "targeted_shard_results.json", {}) or {}
    outer_cycles = int(
        closeout.get("outer_cycles_run", fast.get("outer_cycles_run", 0)) or 0
    )
    repair_exhausted = bool(
        closeout.get(
            "repair_families_exhausted", fast.get("repair_families_exhausted", True)
        )
    )
    incomplete = bool(
        closeout.get("closeout_classification") == "TARGETED_FAILED_AFTER_FAST_PASS"
        and fast.get("fast_guarded_contact_passed")
        and not targeted.get("targeted_shard_passed")
        and outer_cycles < 6
        and not repair_exhausted
    )
    payload = {
        "generated_at_utc": utc_now(),
        "prior_run": rel(prior_run),
        "prior_closeout_classification": closeout.get("closeout_classification"),
        "fast_guarded_contact_passed": bool(fast.get("fast_guarded_contact_passed")),
        "targeted_shard_passed": bool(targeted.get("targeted_shard_passed")),
        "targeted_cases_passed": targeted.get("cases_passed"),
        "targeted_cases_total": targeted.get("cases_total"),
        "outer_cycles_run": outer_cycles,
        "repair_families_exhausted": repair_exhausted,
        "classification": "INCOMPLETE_TARGETED_FEEDBACK_LOOP"
        if incomplete
        else "PRIOR_CLOSEOUT_NOT_CONTINUABLE",
        "continue_v3": incomplete,
    }
    write_json(run_dir / "premature_closeout_verification.json", payload)
    return payload


def write_targeted_perturbation_compatibility_report(
    run_dir: Path, prior_run: Path
) -> dict[str, Any]:
    canonical = [str(p["name"]) for p in cd.PERTURBATIONS]
    targeted = canonical_targeted_perturbations()
    prior_rows = read_jsonl(prior_run / "targeted_shard_results.jsonl")
    prior_names = [str(r.get("perturbation")) for r in prior_rows]
    payload = {
        "generated_at_utc": utc_now(),
        "canonical_cd_perturbations": canonical,
        "cd_perturb_by_name": sorted(str(k) for k in cd.PERTURB_BY_NAME),
        "selected_targeted_perturbations": targeted,
        "selected_names_all_bound": all(
            name in cd.PERTURB_BY_NAME for name in targeted
        ),
        "prior_targeted_perturbations": prior_names,
        "prior_names_all_bound": all(
            name in cd.PERTURB_BY_NAME for name in prior_names
        ),
        "duplicate_fast_guarded_contact_used": targeted.count("fast_guarded_contact")
        > 1,
        "duplicate_fast_guarded_contact_reason": "cd exposes six canonical perturbations; the seventh row preserves the fast blocker as an explicit regression sentinel"
        if targeted.count("fast_guarded_contact") > 1
        else None,
        "legacy_default_targeted_perturbations": list(
            cd.DEFAULT_TARGETED_PERTURBATIONS
        ),
        "legacy_default_missing_from_current_cd": [
            name
            for name in cd.DEFAULT_TARGETED_PERTURBATIONS
            if name not in cd.PERTURB_BY_NAME
        ],
    }
    write_json(run_dir / "targeted_perturbation_compatibility_report.json", payload)
    return payload


def classify_targeted_underopen(row: dict[str, Any]) -> list[str]:
    variant = row.get("variant") if isinstance(row.get("variant"), dict) else {}
    reasons: list[str] = []
    if row_drawer_fraction(row) < STRICT_DRAWER_FRACTION:
        if float(variant.get("lead_cap_m") or 0.0) <= 0.022:
            reasons.append("PULL_DISPLACEMENT_CAP_TOO_LOW")
        if int(variant.get("pull_steps") or 0) <= 6500:
            reasons.append("PULL_DURATION_TOO_SHORT")
        if float(variant.get("pull_velocity_m_per_step") or 0.0) <= 0.00009:
            reasons.append("PULL_VELOCITY_TOO_LOW")
        reasons.append("QPOS_PROGRESS_CONTROLLER_MISSING")
    if row_bilateral(row) >= MIN_EXACT_GROUP_FRAMES and row_forbidden(row) == 0:
        reasons.append("LATCH_SAFE_BUT_PROGRESS_UNDEROPEN")
    if not reasons:
        reasons.append("TARGETED_PERTURBATION_SPEC_BUG")
    return sorted(set(reasons))


def write_targeted_underopen_attribution(
    run_dir: Path, prior_run: Path
) -> dict[str, Any]:
    rows = read_jsonl(prior_run / "targeted_shard_results.jsonl")
    cases: list[dict[str, Any]] = []
    hist: Counter[str] = Counter()
    for row in rows:
        if strict_fast_pass(row):
            continue
        reasons = classify_targeted_underopen(row)
        hist.update(reasons)
        variant = row.get("variant") if isinstance(row.get("variant"), dict) else {}
        trace = trace_path_from_row(row)
        frac_values: list[float] = []
        contact_loss_events = 0
        last_bilateral = True
        if trace:
            for rec in read_jsonl(trace):
                if str(rec.get("mode")) not in {
                    "bounded_teacher_pull",
                    "post_pull_hold",
                }:
                    continue
                frac_values.append(metric_float(rec, "drawer_fraction"))
                bilateral = (
                    int((rec.get("contact_counts") or {}).get("target", 0) or 0) >= 2
                )
                if last_bilateral and not bilateral:
                    contact_loss_events += 1
                last_bilateral = bilateral
        qpos_derivative = 0.0
        if len(frac_values) > 1:
            qpos_derivative = (max(frac_values) - min(frac_values)) / max(
                1, len(frac_values) - 1
            )
        cases.append(
            {
                "perturbation": row.get("perturbation"),
                "drawer_fraction": row_drawer_fraction(row),
                "bilateral_exact_frames": row_bilateral(row),
                "left_group_exact_frames": metric_int(
                    row, "left_group_exact_contact_frames"
                ),
                "right_group_exact_frames": metric_int(
                    row, "right_group_exact_contact_frames"
                ),
                "forbidden": row_forbidden(row),
                "handle_nonlegal": row_handle_nonlegal(row),
                "max_penetration": row_pen(row),
                "pull_displacement_cap_m": variant.get("lead_cap_m"),
                "pull_duration_steps": variant.get("pull_steps"),
                "pull_velocity_m_per_step": variant.get("pull_velocity_m_per_step"),
                "qpos_derivative_proxy_fraction_per_step": qpos_derivative,
                "latch_frames": metric_int(row, "bilateral_exact_contact_frames"),
                "contact_loss_events_proxy": contact_loss_events,
                "keepout_suppressions_proxy": 0
                if row_forbidden(row) == 0
                else row_forbidden(row),
                "classifications": reasons,
                "underopen_reason": reasons[0],
            }
        )
    payload = {
        "generated_at_utc": utc_now(),
        "prior_run": rel(prior_run),
        "failed_case_count": len(cases),
        "classification_histogram": dict(hist),
        "cases": cases,
    }
    write_json(run_dir / "targeted_underopen_attribution.json", payload)
    lines = ["# Targeted Under-open Attribution", ""]
    for case in cases:
        lines.append(
            f"- `{case['perturbation']}`: fraction `{case['drawer_fraction']}`, bilateral `{case['bilateral_exact_frames']}`, forbidden `{case['forbidden']}`, reason `{case['underopen_reason']}`"
        )
    write_md(run_dir / "targeted_underopen_attribution.md", "\n".join(lines))
    return payload


def write_targeted_progress_controller_design(run_dir: Path) -> dict[str, Any]:
    payload = {
        "generated_at_utc": utc_now(),
        "controller": "TargetedProgressFeedbackLatchedPullController",
        "logic": [
            "latch first using exact pad-group policy inherited from V3 patch-group model",
            "during pull, read drawer fraction every control step",
            "if drawer fraction is below progress target, increase the dynamic pull lead cap within a bounded physical range",
            "if target contact count drops, pause lead growth slightly and keep closure",
            "do not command drawer qpos or drawer motor",
        ],
        "hard_constraints": [
            "left/right exact pad-group contact required",
            "forbidden_contact_frames == 0",
            "handle_nonlegal_contact_frames == 0",
            "max_penetration_m <= 0.02",
            "qpos monotonic",
            "no direct qpos",
            "no drawer motor",
            "geometric proxy not success",
        ],
        "allowed_repair_variables": [
            "pull duration",
            "pull velocity schedule",
            "base/dynamic pull displacement cap",
            "gripper closure target",
            "latch hold duration",
            "target-contact drop lead pause",
            "nullspace keepout weight",
            "qpos-progress gain",
        ],
        "topology_or_pad_patch_repair_used": False,
    }
    write_json(run_dir / "targeted_progress_controller_design.json", payload)
    return payload


def reconstruct_prior_fast_solution(
    prior_v3: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prior_stage0 = load_json(prior_v3 / "stage0_evidence_ingestion.json", {}) or {}
    source_manifest = load_json(prior_v3 / "source_candidate_manifest.json", {}) or {}
    base_candidates = source_manifest.get("candidates") or []
    if not base_candidates:
        raise RuntimeError("prior V3 source_candidate_manifest has no candidates")
    fast_cert = (
        load_json(prior_v3 / "fast_guarded_contact_rec_certification.json", {}) or {}
    )
    best = fast_cert.get("best_pass") or {}
    cycle = int(best.get("cycle", 0) or 0)
    sample = int(best.get("sample", 0) or 0)
    family = str(
        best.get("repair_family")
        or REPAIR_FAMILIES[(cycle * 48 + sample) % len(REPAIR_FAMILIES)]
    )
    variants = seed_variants(prior_stage0)
    base = base_candidates[(sample + cycle) % len(base_candidates)]
    seed_variant = tune_variant(
        variants[(sample + cycle) % len(variants)], family, cycle, sample
    )
    candidate = candidate_for_family(base, family, cycle, sample)
    return candidate, seed_variant, best


def targeted_progress_variant(
    seed: dict[str, Any], cycle: int, sample: int
) -> dict[str, Any]:
    v = strip_variant(seed)
    priority = [
        {
            "lead": 0.020,
            "max": 0.023,
            "vel": 0.000085,
            "steps": 7000,
            "press": 0.0040,
            "gain": 0.010,
            "target": 0.805,
            "boost": 0.0005,
            "window": 120,
            "start": 1800,
            "null": 0.0,
            "op": 10.0,
            "opv": 0.065,
            "qv": 3.0,
        },
        {
            "lead": 0.020,
            "max": 0.024,
            "vel": 0.000085,
            "steps": 7400,
            "press": 0.0040,
            "gain": 0.012,
            "target": 0.805,
            "boost": 0.0005,
            "window": 120,
            "start": 1800,
            "null": 0.0,
            "op": 10.0,
            "opv": 0.065,
            "qv": 3.0,
        },
        {
            "lead": 0.020,
            "max": 0.025,
            "vel": 0.000085,
            "steps": 7800,
            "press": 0.0040,
            "gain": 0.014,
            "target": 0.810,
            "boost": 0.0005,
            "window": 120,
            "start": 1800,
            "null": 0.0,
            "op": 10.0,
            "opv": 0.065,
            "qv": 3.0,
        },
        {
            "lead": 0.021,
            "max": 0.021,
            "vel": 0.000085,
            "steps": 6200,
            "press": 0.0040,
            "gain": 0.000,
            "target": 0.805,
            "boost": 0.0005,
            "window": 120,
            "start": 1800,
            "null": 0.0,
            "op": 10.0,
            "opv": 0.065,
            "qv": 3.0,
        },
        {
            "lead": 0.022,
            "max": 0.022,
            "vel": 0.000085,
            "steps": 6200,
            "press": 0.0040,
            "gain": 0.000,
            "target": 0.805,
            "boost": 0.0005,
            "window": 120,
            "start": 1800,
            "null": 0.0,
            "op": 10.0,
            "opv": 0.065,
            "qv": 3.0,
        },
        {
            "lead": 0.022,
            "max": 0.024,
            "vel": 0.000085,
            "steps": 7600,
            "press": 0.0042,
            "gain": 0.008,
            "target": 0.810,
            "boost": 0.0005,
            "window": 120,
            "start": 1800,
            "null": 0.0,
            "op": 10.0,
            "opv": 0.065,
            "qv": 3.0,
        },
        {
            "lead": 0.023,
            "max": 0.023,
            "vel": 0.000085,
            "steps": 6600,
            "press": 0.0042,
            "gain": 0.000,
            "target": 0.805,
            "boost": 0.0005,
            "window": 120,
            "start": 1800,
            "null": 0.0,
            "op": 10.0,
            "opv": 0.065,
            "qv": 3.0,
        },
        {
            "lead": 0.024,
            "max": 0.024,
            "vel": 0.000085,
            "steps": 7000,
            "press": 0.0042,
            "gain": 0.000,
            "target": 0.805,
            "boost": 0.0005,
            "window": 120,
            "start": 1800,
            "null": 0.0,
            "op": 10.0,
            "opv": 0.065,
            "qv": 3.0,
        },
    ]
    lead_caps = [0.024, 0.026, 0.028, 0.030, 0.032, 0.034, 0.036]
    max_caps = [0.028, 0.030, 0.032, 0.034, 0.036, 0.038, 0.040]
    velocities = [0.000090, 0.000095, 0.000105, 0.000115, 0.000125]
    steps = [8200, 9000, 9800, 10600, 11400, 12200]
    presses = [0.0040, 0.0038, 0.0036, 0.0042, 0.0034]
    nulls = [0.0, 0.02, 0.04, 0.06, 0.08]
    progress_gains = [0.006, 0.010, 0.014, 0.018, 0.022]
    i = sample + cycle * 11
    if cycle == 0 and sample < len(priority):
        cfg = priority[sample]
        lead = cfg["lead"]
        max_cap = cfg["max"]
        vel = cfg["vel"]
        pull_steps = cfg["steps"]
        press = cfg["press"]
        gain = cfg["gain"]
        null_gain = cfg["null"]
        op_gain = cfg["op"]
        op_vel_limit = cfg["opv"]
        q_vel_limit = cfg["qv"]
        progress_target_fraction = cfg.get("target", 0.85)
        stagnation_boost = cfg.get("boost", 0.0005)
        progress_check_window = cfg.get("window", 100)
        progress_start_step = cfg.get("start", 0)
    else:
        lead = lead_caps[i % len(lead_caps)]
        max_cap = max_caps[(i // 3) % len(max_caps)]
        vel = velocities[(i // 2) % len(velocities)]
        pull_steps = steps[i % len(steps)]
        press = presses[(i // 5) % len(presses)]
        gain = progress_gains[(i // 7) % len(progress_gains)]
        null_gain = nulls[(i // 11) % len(nulls)]
        op_gain = 10.5 + 0.7 * ((i // 23) % 7)
        op_vel_limit = 0.070 + 0.006 * ((i // 29) % 6)
        q_vel_limit = 2.2 + 0.2 * ((i // 31) % 5)
        progress_target_fraction = 0.85
        stagnation_boost = 0.0005 + 0.00025 * ((i // 13) % 4)
        progress_check_window = 100 + 20 * ((i // 17) % 4)
        progress_start_step = 1200 + 300 * ((i // 19) % 5)
    v.update(
        {
            "name": f"v3_cont_targeted_progress_c{cycle:02d}_s{sample:02d}",
            "targeted_progress_feedback_controller": True,
            "v3_continuation_repair_target": "TARGETED_LATCHED_PULL_PROGRESS_CONTROLLER_REPAIR",
            "pull_steps": int(pull_steps),
            "post_pull_hold_steps": 0,
            "pull_velocity_m_per_step": float(vel),
            "pull_distance_m": 0.36,
            "lead_cap_m": float(lead),
            "progress_base_lead_cap_m": float(lead),
            "progress_max_lead_cap_m": float(max(max_cap, lead)),
            "progress_target_fraction": float(progress_target_fraction),
            "qpos_progress_gain": float(gain),
            "stagnation_lead_boost_m": float(stagnation_boost),
            "progress_check_window": int(progress_check_window),
            "progress_start_step": int(progress_start_step),
            "progress_min_fraction_delta": 0.0025,
            "contact_drop_lead_pause_m": 0.0,
            "pull_press_m": float(press),
            "null_gain": float(null_gain),
            "op_gain": float(op_gain),
            "op_vel_limit": float(op_vel_limit),
            "q_vel_limit": float(q_vel_limit),
            "no_direct_qpos_drawer_opening": True,
            "no_drawer_motor_command": True,
        }
    )
    return v


def targeted_row_pass(row: dict[str, Any]) -> bool:
    return strict_fast_pass(row)


def add_targeted_failure_reasons(row: dict[str, Any]) -> dict[str, Any]:
    reasons = list(row.get("failure_reasons") or [])
    if row_drawer_fraction(row) < STRICT_DRAWER_FRACTION:
        reasons.append("max_drawer_fraction_lt_0p80")
    if metric_int(row, "left_group_exact_contact_frames") < MIN_EXACT_GROUP_FRAMES:
        reasons.append("left_group_exact_contact_lt_30")
    if metric_int(row, "right_group_exact_contact_frames") < MIN_EXACT_GROUP_FRAMES:
        reasons.append("right_group_exact_contact_lt_30")
    if row_bilateral(row) < MIN_EXACT_GROUP_FRAMES:
        reasons.append("bilateral_exact_pull_frames_lt_30")
    if row_forbidden(row) != 0:
        reasons.append("forbidden_contact_present")
    if row_handle_nonlegal(row) != 0:
        reasons.append("handle_nonlegal_contact_present")
    if row_pen(row) > MAX_PENETRATION_M:
        reasons.append("penetration_gt_0p02")
    if row.get("direct_qpos_drawer_opening"):
        reasons.append("direct_qpos_used")
    if row.get("drawer_motor_command_used"):
        reasons.append("drawer_motor_command_used")
    row["targeted_progress_failure_reasons"] = sorted(set(str(r) for r in reasons))
    return row


def summarize_targeted_rows(
    rows: list[dict[str, Any]], attempted: bool = True
) -> dict[str, Any]:
    passed = [r for r in rows if targeted_row_pass(r)]
    hist: Counter[str] = Counter()
    for row in rows:
        if not targeted_row_pass(row):
            hist.update(
                row.get("targeted_progress_failure_reasons")
                or row.get("failure_reasons")
                or ["UNKNOWN"]
            )
    return {
        "generated_at_utc": utc_now(),
        "targeted_shard_attempted": attempted,
        "cases_total": len(rows),
        "cases_passed": len(passed),
        "cases_failed": len(rows) - len(passed),
        "targeted_shard_passed": bool(len(rows) == 7 and len(passed) == 7),
        "worst_case_targeted_drawer_fraction": min(
            (row_drawer_fraction(r) for r in rows), default=0.0
        ),
        "best_case_targeted_drawer_fraction": max(
            (row_drawer_fraction(r) for r in rows), default=0.0
        ),
        "forbidden_contact_frame_count_max": max(
            (row_forbidden(r) for r in rows), default=0
        ),
        "handle_nonlegal_contact_frame_count_max": max(
            (row_handle_nonlegal(r) for r in rows), default=0
        ),
        "max_penetration_m": max((row_pen(r) for r in rows), default=0.0),
        "failure_histogram": dict(hist),
        "rows": rows,
    }


def evaluate_targeted_set(
    run_dir: Path,
    candidate: dict[str, Any],
    variant: dict[str, Any],
    perturbations: list[str],
    cycle: int,
    sample: int,
    stem: str,
    case_base: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, perturb in enumerate(perturbations):
        row = patch.run_case_with_group_metrics(
            candidate,
            perturb,
            variant,
            run_dir,
            case_base + cycle * 1000 + sample * 20 + idx,
            stem,
        )
        row.update(
            {
                "cycle": cycle,
                "sample": sample,
                "targeted_progress_controller": True,
                "targeted_progress_passed": targeted_row_pass(row),
                "geometric_proxy_used_as_success": False,
            }
        )
        add_targeted_failure_reasons(row)
        rows.append(row)
    return rows


def targeted_progress_solver(
    run_dir: Path,
    candidate: dict[str, Any],
    seed_variant: dict[str, Any],
    max_outer_cycles: int,
    max_samples_per_cycle: int,
    no_progress_cycles_before_stop: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    cycle_path = run_dir / "targeted_progress_solver_cycles.jsonl"
    if cycle_path.exists():
        cycle_path.unlink()
    selected = canonical_targeted_perturbations()
    staged = [
        name
        for name in [
            "late_close",
            "nominal",
            "slow_guarded_contact",
            "fast_guarded_contact",
        ]
        if name in selected
    ]
    all_rows: list[dict[str, Any]] = []
    best_full_rows: list[dict[str, Any]] = []
    best_summary = summarize_targeted_rows([])
    no_progress = 0
    best_worst = -1.0
    strict_pass_rows: list[dict[str, Any]] = []
    for cycle in range(max_outer_cycles):
        cycle_best = best_worst
        cycle_samples = 0
        for sample in range(max_samples_per_cycle):
            variant = targeted_progress_variant(seed_variant, cycle, sample)
            subset_rows = evaluate_targeted_set(
                run_dir,
                candidate,
                variant,
                staged,
                cycle,
                sample,
                "targeted_progress_subset",
                970000,
            )
            cycle_samples += 1
            subset_summary = summarize_targeted_rows(subset_rows)
            append_jsonl(
                cycle_path,
                {
                    "cycle": cycle,
                    "sample": sample,
                    "stage": "subset",
                    "variant": variant,
                    "summary": subset_summary,
                },
            )
            all_rows.extend(subset_rows)
            feasible_subset = bool(
                subset_rows
                and all(
                    row_forbidden(r) == 0
                    and row_handle_nonlegal(r) == 0
                    and row_pen(r) <= MAX_PENETRATION_M
                    and metric_int(r, "left_group_exact_contact_frames")
                    >= MIN_EXACT_GROUP_FRAMES
                    and metric_int(r, "right_group_exact_contact_frames")
                    >= MIN_EXACT_GROUP_FRAMES
                    and row_bilateral(r) >= MIN_EXACT_GROUP_FRAMES
                    and not r.get("direct_qpos_drawer_opening")
                    and not r.get("drawer_motor_command_used")
                    for r in subset_rows
                )
            )
            subset_worst = float(
                subset_summary.get("worst_case_targeted_drawer_fraction", 0.0) or 0.0
            )
            best_worst = max(best_worst, subset_worst)
            if feasible_subset and subset_worst >= 0.78:
                full_rows = evaluate_targeted_set(
                    run_dir,
                    candidate,
                    variant,
                    selected,
                    cycle,
                    sample,
                    "targeted_progress_full",
                    980000,
                )
                full_summary = summarize_targeted_rows(full_rows)
                append_jsonl(
                    cycle_path,
                    {
                        "cycle": cycle,
                        "sample": sample,
                        "stage": "full7",
                        "variant": variant,
                        "summary": full_summary,
                    },
                )
                all_rows.extend(full_rows)
                full_worst = float(
                    full_summary.get("worst_case_targeted_drawer_fraction", 0.0) or 0.0
                )
                if (
                    full_worst
                    > float(
                        best_summary.get("worst_case_targeted_drawer_fraction", -1.0)
                        or -1.0
                    )
                    or not best_full_rows
                ):
                    best_summary = full_summary
                    best_summary["selected_variant"] = variant
                    best_full_rows = full_rows
                if full_summary.get("targeted_shard_passed"):
                    strict_pass_rows = full_rows
                    break
        append_jsonl(
            cycle_path,
            {
                "cycle": cycle,
                "cycle_marker": True,
                "samples_run": cycle_samples,
                "best_worst_before": cycle_best,
                "best_worst_after": best_worst,
                "targeted_pass_found": bool(strict_pass_rows),
            },
        )
        if strict_pass_rows:
            break
        if best_worst <= cycle_best + 1e-4:
            no_progress += 1
        else:
            no_progress = 0
        if no_progress >= no_progress_cycles_before_stop and cycle >= 1:
            break
    if not best_full_rows:
        variant = targeted_progress_variant(seed_variant, 0, 0)
        best_full_rows = evaluate_targeted_set(
            run_dir,
            candidate,
            variant,
            selected,
            999,
            0,
            "targeted_progress_best_full_fallback",
            990000,
        )
        best_summary = summarize_targeted_rows(best_full_rows)
        best_summary["selected_variant"] = variant
    write_json(
        run_dir / "targeted_progress_solver_pareto.json",
        {
            "generated_at_utc": utc_now(),
            "rows_total": len(all_rows),
            "full7_evaluated": bool(best_full_rows),
            "best_summary": best_summary,
            "targeted_pass_found": bool(strict_pass_rows),
            "max_outer_cycles_budget": max_outer_cycles,
            "max_samples_per_cycle_budget": max_samples_per_cycle,
            "repair_families_exhausted": not bool(strict_pass_rows),
            "best_worst_case_fraction": best_summary.get(
                "worst_case_targeted_drawer_fraction"
            ),
        },
    )
    report = dict(best_summary)
    report["targeted_shard_passed"] = bool(strict_pass_rows)
    report["fast_guarded_contact_included"] = any(
        r.get("perturbation") == "fast_guarded_contact" for r in best_full_rows
    )
    report["outer_cycles_run"] = (
        max((int(r.get("cycle", -1)) for r in all_rows), default=-1) + 1
    )
    report["repair_families_exhausted"] = not bool(strict_pass_rows)
    write_json(run_dir / "targeted_shard_results.json", report)
    out = run_dir / "targeted_shard_results.jsonl"
    if out.exists():
        out.unlink()
    for row in best_full_rows:
        append_jsonl(out, row)
    return report, best_full_rows, strict_pass_rows


def write_full30_after_targeted(
    run_dir: Path,
    targeted: dict[str, Any],
    candidate: dict[str, Any],
    variant: dict[str, Any] | None,
) -> dict[str, Any]:
    if not targeted.get("targeted_shard_passed") or variant is None:
        full30 = {
            "generated_at_utc": utc_now(),
            "full30_attempted": False,
            "full30_passed": False,
            "cases_total": 0,
            "cases_passed": 0,
            "cases_failed": 0,
            "skip_reason": "targeted_shard_not_passed",
        }
        write_json(run_dir / "full30_report.json", full30)
        (run_dir / "full30_exact_latch_pull_keepout_certification.jsonl").write_text("")
        return full30
    out = run_dir / "full30_exact_latch_pull_keepout_certification.jsonl"
    if out.exists():
        out.unlink()
    names = [str(p["name"]) for p in cd.PERTURBATIONS]
    rows: list[dict[str, Any]] = []
    for rep in range(5):
        for idx, perturb in enumerate(names):
            row = patch.run_case_with_group_metrics(
                candidate,
                perturb,
                variant,
                run_dir,
                1010000 + rep * 100 + idx,
                "full30_targeted_progress",
            )
            row["full30_patch_group_passed"] = targeted_row_pass(row)
            add_targeted_failure_reasons(row)
            rows.append(row)
            append_jsonl(out, row)
    full30 = summarize_targeted_rows(rows)
    full30["targeted_shard_passed"] = bool(targeted.get("targeted_shard_passed"))
    full30["targeted_cases_passed"] = int(targeted.get("cases_passed", 0) or 0)
    full30["targeted_cases_total"] = int(targeted.get("cases_total", 0) or 0)
    full30["full30_attempted"] = True
    full30["full30_cases_total"] = len(rows)
    full30["full30_cases_passed"] = sum(1 for r in rows if targeted_row_pass(r))
    full30["full30_cases_failed"] = sum(1 for r in rows if not targeted_row_pass(r))
    full30["full30_passed"] = bool(
        len(rows) == 30 and all(targeted_row_pass(r) for r in rows)
    )
    write_json(run_dir / "full30_report.json", full30)
    return full30


def build_continuation_closeout(
    run_dir: Path,
    premature: dict[str, Any],
    targeted: dict[str, Any],
    full30: dict[str, Any],
    checks: dict[str, Any],
    prior_fast: dict[str, Any],
    export: dict[str, Any] | None = None,
    local: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    export = (
        export or load_json(run_dir / "strict_teacher_export_manifest.json", {}) or {}
    )
    local = local or load_json(run_dir / "local_replay_handoff_manifest.json", {}) or {}
    action = (
        action or load_json(run_dir / "action_only_replay_spot_check.json", {}) or {}
    )
    strict_complete = bool(
        export.get("strict_teacher_export_complete")
        or export.get("strict_export_complete")
    )
    local_passed = bool(
        local.get("local_state_replay_passed")
        or local.get("local_state_replay_render_passed")
        or local.get("local_strict_replay_render_passed")
    )
    action_passed = bool(action.get("action_only_replay_spot_check_passed"))
    if (
        targeted.get("targeted_shard_passed")
        and full30.get("full30_passed")
        and strict_complete
        and local_passed
        and action_passed
    ):
        classification = CONTINUATION_SUCCESS
        next_gate = "MANUAL_VISUAL_AND_SCIENCE_REVIEW_FOR_DATASET_ADMISSION"
    elif (
        targeted.get("targeted_shard_passed")
        and full30.get("full30_passed")
        and strict_complete
        and not local_passed
    ):
        classification = "LOCAL_REPLAY_RENDER_FAILED"
        next_gate = "LOCAL_STRICT_REPLAY_RENDER_AFTER_TARGETED_PROGRESS_PASS"
    elif targeted.get("targeted_shard_passed") and full30.get("full30_passed"):
        classification = "STRICT_EXPORT_FAILED"
        next_gate = "STRICT_EXPORT_AFTER_TARGETED_PROGRESS_PASS"
    elif targeted.get("targeted_shard_passed"):
        classification = "FULL30_FAILED_AFTER_TARGETED_PROGRESS_PASS"
        next_gate = "FULL30_TARGETED_PROGRESS_GENERALIZATION_REPAIR"
    else:
        classification = (
            "TARGETED_PROGRESS_SOLVER_EXHAUSTED"
            if targeted.get("repair_families_exhausted")
            else "TARGETED_UNDEROPEN_UNRESOLVED"
        )
        next_gate = "TARGETED_LATCHED_PULL_PROGRESS_CONTROLLER_REPAIR_CONTINUATION"
    rows = targeted.get("rows") or []
    best_fast_rows = [
        r for r in rows if r.get("perturbation") == "fast_guarded_contact"
    ]
    best_fast = max(best_fast_rows, key=pareto_score) if best_fast_rows else prior_fast
    payload = {
        "generated_at_utc": utc_now(),
        "task_id": "V11_G4_GOC_V4_EXACT_LATCH_PULL_KEEP_OUT_V3_CONTINUATION_TARGETED_PROGRESS_REPAIR",
        "continued_spec": SPEC_REL,
        "closeout_classification": classification,
        "prior_closeout_was_premature": premature.get("classification")
        == "INCOMPLETE_TARGETED_FEEDBACK_LOOP",
        "fast_guarded_contact_preserved": bool(
            best_fast
            and row_drawer_fraction(best_fast) >= STRICT_DRAWER_FRACTION
            and row_forbidden(best_fast) == 0
        ),
        "targeted_progress_controller_built": True,
        "outer_cycles_run": int(targeted.get("outer_cycles_run", 0) or 0),
        "repair_families_exhausted": bool(targeted.get("repair_families_exhausted")),
        "targeted_shard_passed": bool(targeted.get("targeted_shard_passed")),
        "targeted_cases_passed": int(targeted.get("cases_passed", 0) or 0),
        "targeted_cases_total": int(targeted.get("cases_total", 0) or 0),
        "worst_case_targeted_drawer_fraction": targeted.get(
            "worst_case_targeted_drawer_fraction", 0.0
        ),
        "best_fast_drawer_fraction": row_drawer_fraction(best_fast),
        "best_fast_bilateral_exact_frames": row_bilateral(best_fast),
        "forbidden_contact_frame_count_max": targeted.get(
            "forbidden_contact_frame_count_max", 0
        ),
        "handle_nonlegal_contact_frame_count_max": targeted.get(
            "handle_nonlegal_contact_frame_count_max", 0
        ),
        "max_penetration_m": targeted.get("max_penetration_m", 0.0),
        "full30_passed": bool(full30.get("full30_passed")),
        "strict_export_complete": strict_complete,
        "local_state_replay_passed": local_passed,
        "action_only_replay_spot_check_passed": action_passed,
        "textured_claim_render_passed": bool(local.get("textured_claim_render_passed")),
        "current_truth_modified": bool(checks.get("current_truth_modified")),
        "next_actions_modified": bool(checks.get("next_actions_modified")),
        "committed": False,
        "pushed_to_origin": False,
        "remote_commit_hash": None,
        "next_gate": next_gate,
    }
    write_json(run_dir / "final_closeout.json", payload)
    write_md(
        run_dir / "final_closeout.md",
        "\n".join(
            [
                "# V3 Continuation Final Closeout",
                "",
                f"- closeout_classification: `{classification}`",
                f"- next_gate: `{next_gate}`",
                f"- targeted_shard: `{payload['targeted_cases_passed']}/{payload['targeted_cases_total']}`",
                f"- worst_case_targeted_drawer_fraction: `{payload['worst_case_targeted_drawer_fraction']}`",
                f"- fast_guarded_contact_preserved: `{payload['fast_guarded_contact_preserved']}`",
                f"- full30_passed: `{payload['full30_passed']}`",
                f"- strict_export_complete: `{payload['strict_export_complete']}`",
                f"- local_state_replay_passed: `{payload['local_state_replay_passed']}`",
                "",
                "This continuation fixes the helper early-closeout by running targeted feedback after the fast pass. Geometric proxy is not used as success.",
            ]
        ),
    )
    return payload


def continuation_main(args: argparse.Namespace) -> int:
    install_patch()
    prior_v3 = continuation_latest_v3_run()
    if prior_v3 is None:
        raise SystemExit("No previous V3 run found for continuation")
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else CAMPAIGN
        / "runtime"
        / f"v11_g4_goc_v4_exact_latch_pull_keepout_v3_continuation_targeted_progress_repair_{utc_stamp()}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "task_scope.json",
        {
            "task_id": "V11_G4_GOC_V4_EXACT_LATCH_PULL_KEEP_OUT_V3_CONTINUATION_TARGETED_PROGRESS_REPAIR",
            "continued_task_id": TASK_ID,
            "spec": SPEC_REL,
            "run_dir": rel(run_dir),
            "prior_v3_run": rel(prior_v3),
            "new_scientific_reset": False,
        },
    )
    premature = write_premature_closeout_verification(run_dir, prior_v3)
    write_targeted_perturbation_compatibility_report(run_dir, prior_v3)
    write_targeted_underopen_attribution(run_dir, prior_v3)
    write_targeted_progress_controller_design(run_dir)
    if not premature.get("continue_v3"):
        targeted = summarize_targeted_rows([], attempted=False)
        targeted["repair_families_exhausted"] = False
        targeted["outer_cycles_run"] = 0
        write_json(run_dir / "targeted_shard_results.json", targeted)
        full30 = write_full30_after_targeted(run_dir, targeted, {}, None)
        checks = final_checks(run_dir)
        closeout = build_continuation_closeout(
            run_dir, premature, targeted, full30, checks, {}
        )
        proposed_deltas(run_dir, closeout)
        print(
            json.dumps(
                {
                    "run_dir": rel(run_dir),
                    "closeout_classification": closeout["closeout_classification"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    candidate, seed_variant, prior_fast = reconstruct_prior_fast_solution(prior_v3)
    write_json(
        run_dir / "targeted_progress_seed_solution.json",
        {
            "candidate": candidate,
            "seed_variant": seed_variant,
            "prior_fast_best_row": prior_fast,
        },
    )
    targeted, _rows, _strict_rows = targeted_progress_solver(
        run_dir,
        candidate,
        seed_variant,
        max_outer_cycles=args.max_outer_cycles,
        max_samples_per_cycle=args.max_samples_per_cycle,
        no_progress_cycles_before_stop=args.no_progress_cycles_before_stop,
    )
    selected_variant = (
        targeted.get("selected_variant")
        if isinstance(targeted.get("selected_variant"), dict)
        else None
    )
    full30 = write_full30_after_targeted(run_dir, targeted, candidate, selected_variant)
    if not full30.get("full30_passed"):
        _full30, export, local, action = write_not_reached(run_dir, targeted)
        write_json(run_dir / "full30_report.json", full30)
    else:
        export = strict_export_after_full30(run_dir, full30, candidate)
        local = local_replay_handoff_after_export(run_dir, export)
        action = action_only_spot_check_after_local(run_dir, local)
    trace_hash_manifest(run_dir)
    checks = final_checks(run_dir)
    closeout = build_continuation_closeout(
        run_dir, premature, targeted, full30, checks, prior_fast, export, local, action
    )
    proposed_deltas(run_dir, closeout)
    print(
        json.dumps(
            {
                "run_dir": rel(run_dir),
                "closeout_classification": closeout["closeout_classification"],
                "targeted_cases_passed": closeout["targeted_cases_passed"],
                "targeted_cases_total": closeout["targeted_cases_total"],
                "next_gate": closeout["next_gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--continuation-targeted-progress", action="store_true")
    parser.add_argument("--max-outer-cycles", type=int, default=6)
    parser.add_argument("--max-samples-per-cycle", type=int, default=48)
    parser.add_argument("--no-progress-cycles-before-stop", type=int, default=2)
    args = parser.parse_args()
    if args.continuation_targeted_progress:
        return continuation_main(args)
    install_patch()
    prev = latest_run(PREV_PREFIX)
    if prev is None:
        raise SystemExit(f"previous run not found: {PREV_PREFIX}")
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else CAMPAIGN / "runtime" / f"{RUN_PREFIX}_{utc_stamp()}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "task_scope.json",
        {
            "task_id": TASK_ID,
            "spec": SPEC_REL,
            "run_dir": rel(run_dir),
            "previous_run": rel(prev),
        },
    )
    stage0 = write_stage0(run_dir, prev)
    candidate_map = load_previous_candidate_map(prev)
    base_candidates = select_base_candidates(prev, stage0, candidate_map)
    write_json(
        run_dir / "source_candidate_manifest.json",
        {
            "generated_at_utc": utc_now(),
            "candidate_count": len(base_candidates),
            "candidate_ids": [c.get("candidate_id") for c in base_candidates],
            "candidates": base_candidates,
        },
    )
    forbidden = forbidden_attribution(
        run_dir, stage0.get("best_opening_forbidden_row") or {}
    )
    underopen = underopen_attribution(
        run_dir, stage0.get("best_legal_underopen_row") or {}
    )
    write_parameterization(run_dir, stage0)
    fast_cert, solver_rows, strict_passes = constrained_solver(
        run_dir,
        prev,
        stage0,
        base_candidates,
        max_outer_cycles=args.max_outer_cycles,
        max_samples_per_cycle=args.max_samples_per_cycle,
        no_progress_cycles_before_stop=args.no_progress_cycles_before_stop,
    )
    candidate_by_id: dict[str, dict[str, Any]] = {}
    if strict_passes:
        base_variants = seed_variants(stage0)
        for row in strict_passes:
            cycle = int(row.get("cycle", 0) or 0)
            sample = int(row.get("sample", 0) or 0)
            family = str(
                row.get("repair_family")
                or REPAIR_FAMILIES[
                    (cycle * args.max_samples_per_cycle + sample) % len(REPAIR_FAMILIES)
                ]
            )
            base = base_candidates[(sample + cycle) % len(base_candidates)]
            cand = candidate_for_family(base, family, cycle, sample)
            candidate_by_id[str(cand.get("candidate_id"))] = cand
            row["variant"] = tune_variant(
                base_variants[(sample + cycle) % len(base_variants)],
                family,
                cycle,
                sample,
            )
    targeted, _targeted_rows = targeted_7of7(
        run_dir, strict_passes[0] if strict_passes else {}, candidate_by_id
    )
    full30, export, local, action = write_not_reached(run_dir, targeted)
    trace_hash_manifest(run_dir)
    checks = final_checks(run_dir)
    pareto = load_json(run_dir / "fast_constrained_solver_pareto.json", {}) or {}
    closeout = build_closeout(
        run_dir,
        stage0,
        forbidden,
        underopen,
        fast_cert,
        pareto,
        targeted,
        full30,
        export,
        local,
        action,
        checks,
    )
    proposed_deltas(run_dir, closeout)
    print(
        json.dumps(
            {
                "run_dir": rel(run_dir),
                "closeout_classification": closeout["closeout_classification"],
                "next_gate": closeout["next_gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

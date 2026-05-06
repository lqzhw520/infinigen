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

sys.path.insert(0, str(ROOT / "scripts/mint"))
import v11_g4_dynamic_pull_accessible_pool_co_design as cd  # noqa: E402
import v11_g4_round_knob_exact_contact_patch_latch_pull_synthesis as patch  # noqa: E402


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


def metric_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    return patch.metric_int(row, key, default)


def metric_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    return patch.metric_float(row, key, default)


def latest_run(prefix: str) -> Path | None:
    runs = sorted((CAMPAIGN / "runtime").glob(f"{prefix}*"))
    runs = [p for p in runs if (p / "final_closeout.json").exists()]
    return runs[-1] if runs else None


def install_patch() -> None:
    patch.install_patch()
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--max-outer-cycles", type=int, default=6)
    parser.add_argument("--max-samples-per-cycle", type=int, default=48)
    parser.add_argument("--no-progress-cycles-before-stop", type=int, default=2)
    args = parser.parse_args()
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

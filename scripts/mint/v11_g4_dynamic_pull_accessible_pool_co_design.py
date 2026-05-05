#!/usr/bin/env python3
"""Dynamic-pull-accessible GOC-v4 pool co-design to strict export/replay.

The previous fixed accepted pool is used only as a negative baseline. This
helper generates new declared candidates, recertifies physical accessibility,
probes dynamic pull legality with the existing MuJoCo bounded-pull runner, and
only advances to full30/export/replay after strict targeted shard pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN = ROOT / "experiments/mint/mint_drawer_v1"
SPEC_REL = (
    "experiments/mint/mint_drawer_v1/sovereign/experiment_specs/"
    "v11_g4_goc_v4_dynamic_pull_accessible_pool_co_design_to_strict_export_replay_overnight.yaml"
)
PRIOR_FIXED_POOL_RUN = CAMPAIGN / (
    "runtime/v11_g4_goc_v4_pull_force_impedance_axis_alignment_repair_to_strict_export_local_replay_overnight_20260504T161312Z"
)
SOURCE_ACCESSIBLE_POOL_RUN = CAMPAIGN / "runtime/v11_g4_goc_v4_physical_accessibility_instance_pool_20260504T032604Z"
SOURCE_LAYER4R_RUN = CAMPAIGN / "runtime/v11_g4_goc_v4_contact_mode_operational_space_policy_repair_on_accessible_pool_20260504T075148Z"
RUN_PREFIX = "v11_g4_goc_v4_dynamic_pull_accessible_pool_co_design_to_strict_export_replay"
TASK_ID = "V11_G4_GOC_V4_DYNAMIC_PULL_ACCESSIBLE_POOL_CO_DESIGN_TO_STRICT_EXPORT_REPLAY_OVERNIGHT_V1"
MAX_PENETRATION_M = 0.02
MAX_FORCE_N = 1_000_000.0
STRICT_DRAWER_FRACTION = 0.80
MIN_TARGET_FRAMES = 80
MIN_CONSECUTIVE_FRAMES = 30
MIN_TWO_PAD_FRAMES = 30
MONOTONIC_TOL_M = 1.0e-4
LOCAL_PLAYGROUND_ROOT = Path("/Users/zhuhaowu/Documents/Playground")

sys.path.insert(0, str(ROOT / "scripts/mint"))
import v11_g4_bounded_teacher_pull_rollout_on_accessible_pool as bp  # noqa: E402
import v11_g4_physical_accessibility_instance_pool as pool  # noqa: E402
import v11_g4_pull_force_impedance_axis_alignment_repair_to_strict_export_local_replay_overnight as prior_pull  # noqa: E402

PERTURBATIONS = bp.PERTURBATIONS
PERTURB_BY_NAME = {p["name"]: p for p in PERTURBATIONS}
DEFAULT_TARGETED_PERTURBATIONS = [
    "nominal",
    "slow_guarded_contact",
    "pregrasp_offset_plus_y",
    "pregrasp_offset_minus_y",
    "low_contact_hold",
    "long_contact_hold",
    "guarded_offset_z_up",
]
PROBE_VARIANTS = [
    prior_pull.PRIMARY_VARIANTS[0],
    prior_pull.PRIMARY_VARIANTS[1],
    prior_pull.CONTINUATION_VARIANTS[0],
]
REPAIR_VARIANTS = [
    prior_pull.PRIMARY_VARIANTS[2],
    prior_pull.PRIMARY_VARIANTS[3],
    prior_pull.CONTINUATION_VARIANTS[1],
    {
        "name": "cd_keepout_micro_pull_low_press_ik_hold",
        "pull_steps": 5200,
        "post_pull_hold_steps": 0,
        "pull_velocity_m_per_step": 0.000075,
        "pull_distance_m": 0.34,
        "lead_cap_m": 0.018,
        "pull_press_m": 0.0005,
        "op_gain": 8.0,
        "op_vel_limit": 0.060,
        "q_vel_limit": 1.60,
        "null_gain": 0.22,
        "servo_kp": 260.0,
        "servo_kd": 92.0,
        "finger_mode": "ik_hold",
    },
    {
        "name": "cd_keepout_slow_binary_close_low_gain",
        "pull_steps": 6200,
        "post_pull_hold_steps": 0,
        "pull_velocity_m_per_step": 0.00006,
        "pull_distance_m": 0.34,
        "lead_cap_m": 0.016,
        "pull_press_m": 0.0015,
        "op_gain": 7.0,
        "op_vel_limit": 0.055,
        "q_vel_limit": 1.45,
        "null_gain": 0.25,
        "servo_kp": 240.0,
        "servo_kd": 98.0,
        "finger_mode": "binary_close",
    },
]


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
    if hasattr(value, "tolist"):
        try:
            return ready(value.tolist())
        except Exception:
            pass
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


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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
    return proc.stdout.strip() if proc.returncode == 0 else (proc.stderr.strip() or f"git_failed:{proc.returncode}")


def run_cmd(args: list[str], cwd: Path = ROOT) -> dict[str, Any]:
    proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return {"cmd": args, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def max_consecutive(values: list[bool]) -> int:
    best = cur = 0
    for value in values:
        cur = cur + 1 if value else 0
        best = max(best, cur)
    return best


def classify_row(row: dict[str, Any]) -> dict[str, Any]:
    out = prior_pull.reclassify_case(row)
    return out


def row_failure_hist(rows: list[dict[str, Any]]) -> dict[str, int]:
    hist = Counter()
    for row in rows:
        if not row.get("passed"):
            hist.update(row.get("failure_reasons") or ["UNKNOWN_FAILURE"])
    return dict(sorted(hist.items()))


def row_label_hist(rows: list[dict[str, Any]]) -> dict[str, int]:
    hist = Counter()
    for row in rows:
        hist.update(row.get("oracle", {}).get("structural_labels") or [])
    return dict(sorted(hist.items()))


def row_rank(row: dict[str, Any]) -> tuple[float, ...]:
    reasons = set(row.get("failure_reasons") or [])
    return (
        1.0 if row.get("passed") else 0.0,
        float(row.get("max_drawer_fraction", 0.0) or 0.0),
        float(row.get("pull_phase_two_pad_target_contact_frames", 0) or 0),
        float(row.get("pull_phase_target_contact_frames", 0) or 0),
        -float(row.get("forbidden_contact_frames", 0) or 0),
        -float(row.get("handle_nonlegal_contact_frames", 0) or 0),
        -float(len(reasons)),
    )


def summarize_rows(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    def ivals(k: str) -> list[int]:
        return [int(r.get(k, 0) or 0) for r in rows]

    def fvals(k: str) -> list[float]:
        return [float(r.get(k, 0.0) or 0.0) for r in rows if r.get(k) is not None]

    passed = sum(1 for r in rows if r.get("passed"))
    return {
        "generated_at_utc": utc_now(),
        "name": name,
        "cases_total": len(rows),
        "cases_passed": passed,
        "cases_failed": len(rows) - passed,
        "matrix_passed": bool(rows and passed == len(rows)),
        "failure_histogram": row_failure_hist(rows),
        "structural_label_histogram": row_label_hist(rows),
        "target_contact_frames_min": min(ivals("target_contact_frames")) if rows else 0,
        "target_contact_consecutive_min": min(ivals("target_contact_max_consecutive_frames")) if rows else 0,
        "pull_phase_target_contact_frames_min": min(ivals("pull_phase_target_contact_frames")) if rows else 0,
        "pull_phase_two_pad_target_contact_frames_min": min(ivals("pull_phase_two_pad_target_contact_frames")) if rows else 0,
        "forbidden_contact_frames_max": max(ivals("forbidden_contact_frames")) if rows else 0,
        "handle_nonlegal_contact_frames_max": max(ivals("handle_nonlegal_contact_frames")) if rows else 0,
        "max_penetration_m": max(fvals("max_penetration_m")) if rows else 0.0,
        "max_force_n": max(fvals("max_force_n")) if rows else 0.0,
        "max_drawer_fraction_min": min(fvals("max_drawer_fraction")) if rows else 0.0,
        "max_drawer_fraction_max": max(fvals("max_drawer_fraction")) if rows else 0.0,
        "direct_qpos_drawer_opening": any(bool(r.get("direct_qpos_drawer_opening")) for r in rows),
        "drawer_motor_command_used": any(bool(r.get("drawer_motor_command_used")) for r in rows),
    }


def stage0(run_dir: Path) -> dict[str, Any]:
    prior_files = {
        name: rel(PRIOR_FIXED_POOL_RUN / name)
        for name in [
            "final_closeout.json",
            "post_push_verification.json",
            "pull_wrench_oracle_results.jsonl",
            "dynamic_pull_forensic_decision.json",
            "dynamic_pull_forensic_certificate.md",
            "trace_hash_manifest.json",
        ]
        if (PRIOR_FIXED_POOL_RUN / name).exists()
    }
    payload = {
        "generated_at_utc": utc_now(),
        "task_id": TASK_ID,
        "pwd": str(ROOT),
        "branch": run_git(["branch", "--show-current"]),
        "head": run_git(["rev-parse", "HEAD"]),
        "remote_v": run_git(["remote", "-v"]),
        "git_status_short": run_git(["status", "--short"]),
        "task_spec": SPEC_REL,
        "preflight": run_cmd([
            "/root/anaconda3/envs/infinigen/bin/python",
            "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py",
            "--spec",
            SPEC_REL,
            "--dry-run",
        ]),
        "prior_fixed_pool_run": rel(PRIOR_FIXED_POOL_RUN),
        "prior_files": prior_files,
        "fixed_pool_used_only_as_negative_baseline": True,
        "old_fixed_pool_modified": False,
        "old_pool_used_as_strict_export_input": False,
    }
    payload["harness_preflight_passed"] = payload["preflight"]["returncode"] == 0
    write_json(run_dir / "stage0_authority_and_prior_evidence.json", payload)
    (run_dir / "commands.log").write_text(json.dumps(ready(payload), indent=2, sort_keys=True) + "\n")
    prior_closeout = load_json(PRIOR_FIXED_POOL_RUN / "final_closeout.json", {})
    lines = [
        "# Fixed Pool Negative Baseline",
        "",
        f"- prior_run: `{rel(PRIOR_FIXED_POOL_RUN)}`",
        f"- prior_closeout: `{prior_closeout.get('closeout_classification')}`",
        f"- prior_next_gate: `{prior_closeout.get('next_gate')}`",
        f"- prior_layer4r: `{prior_closeout.get('source_layer4r_cases_passed')}/{prior_closeout.get('source_layer4r_cases_total')}`",
        f"- prior_best_targeted: `{prior_closeout.get('targeted_shard_cases_passed')}/{prior_closeout.get('targeted_shard_cases_total')}`",
        "- old_fixed_pool_modified: `false`",
        "- old_pool_used_as_strict_export_input: `false`",
        "",
        "The prior pool is a defect source only. This phase generates declared new candidates.",
    ]
    write_md(run_dir / "fixed_pool_negative_baseline_summary.md", "\n".join(lines))
    return payload


def all_prior_oracle_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(PRIOR_FIXED_POOL_RUN.glob("*oracle_results.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def stage1_defect_certificate(run_dir: Path) -> dict[str, Any]:
    closeout = load_json(PRIOR_FIXED_POOL_RUN / "final_closeout.json", {})
    rows = all_prior_oracle_rows()
    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("candidate_id"))
        if cid and cid != "None":
            by_instance[cid].append(row)
    certs: list[dict[str, Any]] = []
    counts = Counter()
    for cid in sorted(set(list(by_instance) + [str(k).split("::")[0] for k in (closeout.get("best_drawer_fraction_by_instance") or {})])):
        inst_rows = by_instance.get(cid, [])
        best = max(inst_rows, key=row_rank) if inst_rows else {}
        labels = Counter()
        reasons = Counter()
        for row in inst_rows:
            labels.update(row.get("oracle", {}).get("structural_labels") or [])
            reasons.update(row.get("failure_reasons") or [])
        best_frac = max([float(r.get("max_drawer_fraction", 0.0) or 0.0) for r in inst_rows] or [0.0])
        forbidden = max([int(r.get("forbidden_contact_frames", 0) or 0) for r in inst_rows] or [0])
        two_pad = max([int(r.get("pull_phase_two_pad_target_contact_frames", 0) or 0) for r in inst_rows] or [0])
        target = max([int(r.get("pull_phase_target_contact_frames", 0) or 0) for r in inst_rows] or [0])
        if any(r.get("passed") for r in inst_rows):
            defect = "EXPORT_INFRA_DEFECT"
        elif best_frac >= STRICT_DRAWER_FRACTION and forbidden > 0:
            defect = "MIXED_DEFECT"
        elif best_frac >= STRICT_DRAWER_FRACTION and (two_pad < MIN_TWO_PAD_FRAMES or target < MIN_TARGET_FRAMES):
            defect = "MIXED_DEFECT"
        elif best_frac < 0.20 and ("LIKELY_FIXED_POOL_DYNAMIC_PULL_INFEASIBLE" in labels or "AXIS_FORCE_INSUFFICIENT" in labels):
            defect = "STRUCTURE_DEFECT"
        elif labels and not forbidden:
            defect = "CONTROLLER_DEFECT"
        elif labels:
            defect = "MIXED_DEFECT"
        else:
            defect = "UNKNOWN_DEFECT"
        counts[defect] += 1
        certs.append({
            "instance_id": cid,
            "defect_class": defect,
            "best_drawer_fraction": best_frac,
            "strict_pass": any(r.get("passed") for r in inst_rows),
            "target_contact_retention_best": target,
            "two_pad_retention_best": two_pad,
            "forbidden_contact_frames_max": forbidden,
            "handle_nonlegal_contact_frames_max": max([int(r.get("handle_nonlegal_contact_frames", 0) or 0) for r in inst_rows] or [0]),
            "max_penetration_m": max([float(r.get("max_penetration_m", 0.0) or 0.0) for r in inst_rows] or [0.0]),
            "max_force_n": max([float(r.get("max_force_n", 0.0) or 0.0) for r in inst_rows] or [0.0]),
            "drawer_qpos_monotonic_best": bool(best.get("drawer_qpos_nondecreasing_with_tolerance", False)),
            "structural_label_histogram": dict(sorted(labels.items())),
            "failure_reason_histogram": dict(sorted(reasons.items())),
            "best_row": best,
        })
    payload = {"generated_at_utc": utc_now(), "certificates": certs, "defect_histogram": dict(sorted(counts.items()))}
    write_json(run_dir / "fixed_pool_per_instance_defect_certificate.json", payload)
    lines = ["# Fixed Pool Defect Summary", "", f"- defect_histogram: `{json.dumps(payload['defect_histogram'], sort_keys=True)}`", "", "The original fixed pool is not repaired or promoted by this phase."]
    write_md(run_dir / "fixed_pool_defect_summary.md", "\n".join(lines))
    return payload


def base_params(robot_base_pos: list[float], handle_x: float, handle_y: float, handle_z: float = 0.36, door_x: float = 0.145, cutout_w: float = 0.18, cutout_h: float = 0.16) -> dict[str, Any]:
    return {
        "robot_base_pos": robot_base_pos,
        "handle_x": handle_x,
        "handle_y": handle_y,
        "handle_z": handle_z,
        "handle_radius": 0.025,
        "front_cutout": True,
        "door_x": door_x,
        "cutout_half_width": cutout_w,
        "cutout_half_height": cutout_h,
        "drawer_kind": "simple_knob_drawer_with_cabinet_envelope",
        "generation_policy": "dynamic_pull_accessible_pool_co_design_declared_generated_or_repaired_variant",
    }


def make_candidate(candidate_id: str, co_source: str, seed: int, params: dict[str, Any], parent: str | None = None) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "source_type": "generated_accessible_variant",
        "co_design_source_type": co_source,
        "source_type_for_spec": co_source,
        "parent_instance_id": parent,
        "synthetic_seed": seed,
        "status": "pending",
        "model_builder_parameters": params,
        "fixed_pool_original_mutated": False,
        "declared_generated_or_repaired_variant": True,
    }


def stage2_candidate_pool(run_dir: Path, repair_cycle: int = 0) -> list[dict[str, Any]]:
    # All candidates are declared generated/repaired variants with real drawer/cabinet/handle collision.
    # Repair cycles deliberately explore keepout, not target-authority broadening.
    if repair_cycle == 0:
        specs = [
            ("dynamic_old_variant_from_001_centered_clearance", "old_variant", 9101, [-0.60, 0.0, 0.0], -0.18, 0.00, 0.36, 0.145, 0.18, 0.16, "generated_knob_drawer_accessible_001"),
            ("dynamic_old_variant_from_002_forward_clearance", "old_variant", 9102, [-0.52, 0.0, 0.0], -0.12, 0.00, 0.36, 0.135, 0.19, 0.17, "generated_knob_drawer_accessible_002"),
            ("dynamic_generated_center_pull_001", "generated_variant", 9201, [-0.50, 0.0, 0.0], -0.10, 0.00, 0.36, 0.130, 0.20, 0.18, None),
            ("dynamic_generated_center_pull_002", "generated_variant", 9202, [-0.56, 0.0, 0.0], -0.14, 0.00, 0.36, 0.135, 0.20, 0.18, None),
            ("dynamic_generated_low_y_offset_003", "generated_variant", 9203, [-0.56, -0.02, 0.0], -0.14, -0.035, 0.36, 0.135, 0.20, 0.18, None),
            ("dynamic_generated_high_y_offset_004", "generated_variant", 9204, [-0.56, 0.02, 0.0], -0.14, 0.035, 0.36, 0.135, 0.20, 0.18, None),
            ("dynamic_repaired_layout_side_clearance_001", "repaired_layout", 9301, [-0.48, -0.04, 0.0], -0.10, -0.02, 0.36, 0.125, 0.21, 0.18, "generated_knob_drawer_accessible_003"),
            ("dynamic_repaired_layout_side_clearance_002", "repaired_layout", 9302, [-0.48, 0.04, 0.0], -0.10, 0.02, 0.36, 0.125, 0.21, 0.18, "generated_knob_drawer_accessible_004"),
            ("dynamic_repaired_layout_near_axis_003", "repaired_layout", 9303, [-0.44, 0.0, 0.0], -0.08, 0.00, 0.36, 0.120, 0.215, 0.18, "generated_knob_drawer_accessible_005"),
        ]
    elif repair_cycle == 1:
        specs = [
            ("dynamic_keepout_ypos_generated_001", "generated_variant", 9401, [-0.64, 0.075, 0.0], -0.14, 0.085, 0.36, 0.130, 0.23, 0.19, None),
            ("dynamic_keepout_ypos_generated_002", "generated_variant", 9402, [-0.70, 0.095, 0.0], -0.18, 0.095, 0.36, 0.140, 0.23, 0.19, None),
            ("dynamic_keepout_yneg_generated_003", "generated_variant", 9403, [-0.64, -0.075, 0.0], -0.14, -0.085, 0.36, 0.130, 0.23, 0.19, None),
            ("dynamic_keepout_yneg_generated_004", "generated_variant", 9404, [-0.70, -0.095, 0.0], -0.18, -0.095, 0.36, 0.140, 0.23, 0.19, None),
            ("dynamic_keepout_old001_ypos", "old_variant", 9405, [-0.66, 0.075, 0.0], -0.18, 0.080, 0.36, 0.140, 0.23, 0.19, "generated_knob_drawer_accessible_001"),
            ("dynamic_keepout_old002_yneg", "old_variant", 9406, [-0.62, -0.075, 0.0], -0.14, -0.080, 0.36, 0.135, 0.23, 0.19, "generated_knob_drawer_accessible_002"),
            ("dynamic_keepout_repaired_ypos", "repaired_layout", 9407, [-0.66, 0.10, 0.0], -0.16, 0.10, 0.36, 0.135, 0.24, 0.19, "generated_knob_drawer_accessible_004"),
            ("dynamic_keepout_repaired_yneg", "repaired_layout", 9408, [-0.66, -0.10, 0.0], -0.16, -0.10, 0.36, 0.135, 0.24, 0.19, "generated_knob_drawer_accessible_003"),
            ("dynamic_keepout_repaired_center_far", "repaired_layout", 9409, [-0.74, 0.0, 0.0], -0.20, 0.00, 0.36, 0.150, 0.24, 0.19, "generated_knob_drawer_accessible_005"),
        ]
    elif repair_cycle == 2:
        specs = [
            ("dynamic_high_handle_generated_001", "generated_variant", 9501, [-0.62, 0.04, 0.0], -0.14, 0.055, 0.42, 0.130, 0.23, 0.21, None),
            ("dynamic_high_handle_generated_002", "generated_variant", 9502, [-0.66, -0.04, 0.0], -0.14, -0.055, 0.42, 0.130, 0.23, 0.21, None),
            ("dynamic_high_handle_generated_003", "generated_variant", 9503, [-0.72, 0.0, 0.0], -0.20, 0.00, 0.42, 0.145, 0.24, 0.21, None),
            ("dynamic_mid_high_old001", "old_variant", 9504, [-0.68, 0.04, 0.0], -0.18, 0.055, 0.40, 0.145, 0.23, 0.20, "generated_knob_drawer_accessible_001"),
            ("dynamic_mid_high_old002", "old_variant", 9505, [-0.60, -0.04, 0.0], -0.14, -0.055, 0.40, 0.135, 0.23, 0.20, "generated_knob_drawer_accessible_002"),
            ("dynamic_high_repaired_ypos", "repaired_layout", 9506, [-0.70, 0.09, 0.0], -0.18, 0.095, 0.40, 0.145, 0.24, 0.20, "generated_knob_drawer_accessible_004"),
            ("dynamic_high_repaired_yneg", "repaired_layout", 9507, [-0.70, -0.09, 0.0], -0.18, -0.095, 0.40, 0.145, 0.24, 0.20, "generated_knob_drawer_accessible_003"),
        ]
    else:
        specs = [
            ("dynamic_far_mount_center_generated_001", "generated_variant", 9601, [-0.78, 0.00, 0.0], -0.22, 0.00, 0.36, 0.150, 0.25, 0.20, None),
            ("dynamic_far_mount_ypos_generated_002", "generated_variant", 9602, [-0.78, 0.10, 0.0], -0.22, 0.10, 0.38, 0.150, 0.25, 0.20, None),
            ("dynamic_far_mount_yneg_generated_003", "generated_variant", 9603, [-0.78, -0.10, 0.0], -0.22, -0.10, 0.38, 0.150, 0.25, 0.20, None),
            ("dynamic_far_mount_old001", "old_variant", 9604, [-0.76, 0.08, 0.0], -0.22, 0.08, 0.38, 0.150, 0.25, 0.20, "generated_knob_drawer_accessible_001"),
            ("dynamic_far_mount_old002", "old_variant", 9605, [-0.72, -0.08, 0.0], -0.18, -0.08, 0.38, 0.145, 0.25, 0.20, "generated_knob_drawer_accessible_002"),
            ("dynamic_far_mount_repaired_ypos", "repaired_layout", 9606, [-0.80, 0.12, 0.0], -0.22, 0.12, 0.38, 0.150, 0.25, 0.20, "generated_knob_drawer_accessible_004"),
            ("dynamic_far_mount_repaired_yneg", "repaired_layout", 9607, [-0.80, -0.12, 0.0], -0.22, -0.12, 0.38, 0.150, 0.25, 0.20, "generated_knob_drawer_accessible_003"),
        ]
    candidates = [make_candidate(cid, src, seed, base_params(base, hx, hy, hz, door_x=door, cutout_w=cw, cutout_h=ch), parent) for cid, src, seed, base, hx, hy, hz, door, cw, ch, parent in specs]
    for c in candidates:
        if repair_cycle:
            c["repair_cycle"] = repair_cycle
            c["model_builder_parameters"]["repair_cycle_note"] = f"keepout_or_ik_repair_cycle_{repair_cycle}"
    write_json(run_dir / ("candidate_dynamic_pull_pool.json" if repair_cycle == 0 else f"candidate_dynamic_pull_pool_repair_cycle_{repair_cycle}.json"), {"generated_at_utc": utc_now(), "candidate_count": len(candidates), "repair_cycle": repair_cycle, "candidates": candidates})
    return candidates


def perturb_by_name(name: str) -> dict[str, Any]:
    return PERTURB_BY_NAME[name]


def run_case(candidate: dict[str, Any], perturb_name: str, variant: dict[str, Any], run_dir: Path, case_idx: int, stem: str) -> dict[str, Any]:
    raw = bp.run_case(candidate, perturb_by_name(perturb_name), variant, run_dir, case_idx)
    row = classify_row(raw)
    row["co_design_source_type"] = candidate.get("co_design_source_type")
    row["source_type_for_spec"] = candidate.get("source_type_for_spec")
    row["parent_instance_id"] = candidate.get("parent_instance_id")
    row["case_stem"] = stem
    return row


def physical_pass(result: dict[str, Any]) -> bool:
    return bool(result.get("accepted") and result.get("binding") and result.get("reset", {}).get("reset_ok"))


def probe_candidate(candidate: dict[str, Any], run_dir: Path, candidate_idx: int, variants: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for off, variant in enumerate(variants):
        row = run_case(candidate, "nominal", variant, run_dir, 10_000 + candidate_idx * 10 + off, "candidate_oracle_probe")
        rows.append(row)
    best = max(rows, key=row_rank) if rows else {}
    return best, rows


def oracle_result(candidate: dict[str, Any], physical: dict[str, Any], probe_best: dict[str, Any] | None, probe_rows: list[dict[str, Any]]) -> dict[str, Any]:
    o1 = physical_pass(physical)
    o6 = bool(physical.get("binding", {}).get("legal_finger_pad_geom_ids") and physical.get("binding", {}).get("drawer_handle_geom_ids"))
    if probe_best is None:
        probe_best = {}
    o2 = bool(float(probe_best.get("oracle", {}).get("net_pull_axis_wrench_max_n", 0.0) or 0.0) > 0.0 and not probe_best.get("direct_qpos_drawer_opening") and not probe_best.get("drawer_motor_command_used"))
    o3 = bool(int(probe_best.get("target_contact_frames", 0) or 0) >= MIN_TARGET_FRAMES and int(probe_best.get("pull_phase_two_pad_target_contact_frames", 0) or 0) >= MIN_TWO_PAD_FRAMES)
    o4 = bool(int(probe_best.get("forbidden_contact_frames", 0) or 0) == 0 and int(probe_best.get("handle_nonlegal_contact_frames", 0) or 0) == 0 and float(probe_best.get("max_penetration_m", 0.0) or 0.0) <= MAX_PENETRATION_M)
    o5 = bool(probe_best.get("drawer_qpos_nondecreasing_with_tolerance") and float(probe_best.get("max_drawer_fraction", 0.0) or 0.0) >= STRICT_DRAWER_FRACTION)
    admitted = bool(o1 and o2 and o3 and o4 and o5 and o6 and probe_best.get("passed"))
    dominant = None
    if not o1:
        dominant = physical.get("rejection_reason") or "PHYSICAL_ACCESSIBILITY_FAILED"
    elif not o6:
        dominant = "STRICT_AUTHORITY_FAILED"
    elif not o2:
        dominant = "PULL_WRENCH_INSUFFICIENT"
    elif not o3:
        dominant = "TWO_PAD_RETENTION_FAILED"
    elif not o4:
        dominant = "FULL_BODY_KEEP_OUT_FAILED"
    elif not o5:
        dominant = "DRAWER_QPOS_NONMONOTONIC"
    if not dominant and not admitted:
        dominant = "MIXED_STRUCTURE_CONTROL_DEFECT"
    return {
        "candidate_id": candidate["candidate_id"],
        "source_type": candidate.get("co_design_source_type"),
        "parent_instance_id": candidate.get("parent_instance_id"),
        "admitted": admitted,
        "dominant_failure": dominant,
        "oracles": {
            "O1_physical_accessibility": o1,
            "O2_pull_wrench_accessibility": o2,
            "O3_two_pad_contact_retention": o3,
            "O4_full_body_keepout_during_pull": o4,
            "O5_drawer_qpos_monotonicity": o5,
            "O6_strict_goc_v4_authority": o6,
        },
        "physical_accessibility": physical,
        "best_probe_row": probe_best,
        "probe_summary": summarize_rows(probe_rows, f"candidate_oracle_probe_{candidate['candidate_id']}") if probe_rows else {},
        "selected_variant": probe_best.get("variant") if probe_best else None,
    }


def stage3_joint_oracle(run_dir: Path, candidates: list[dict[str, Any]], variants: list[dict[str, Any]], repair_cycle: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    admitted: list[dict[str, Any]] = []
    oracle_path = run_dir / ("candidate_oracle_results.jsonl" if repair_cycle == 0 else f"candidate_oracle_results_repair_cycle_{repair_cycle}.jsonl")
    if oracle_path.exists():
        oracle_path.unlink()
    for idx, candidate in enumerate(candidates):
        physical = pool.evaluate_generated_candidate(candidate, run_dir)
        probe_best: dict[str, Any] | None = None
        probe_rows: list[dict[str, Any]] = []
        if physical_pass(physical):
            probe_best, probe_rows = probe_candidate(physical, run_dir, idx + repair_cycle * 100, variants)
        result = oracle_result(candidate, physical, probe_best, probe_rows)
        append_jsonl(oracle_path, result)
        results.append(result)
        if result["admitted"]:
            admitted_candidate = physical
            admitted_candidate["co_design_source_type"] = candidate.get("co_design_source_type")
            admitted_candidate["source_type_for_spec"] = candidate.get("source_type_for_spec")
            admitted_candidate["parent_instance_id"] = candidate.get("parent_instance_id")
            admitted_candidate["selected_pull_variant"] = result["selected_variant"]
            admitted.append(admitted_candidate)
    hist = Counter(r.get("dominant_failure") or "ADMITTED" for r in results)
    report = {"generated_at_utc": utc_now(), "repair_cycle": repair_cycle, "candidate_count": len(candidates), "admitted_count": len(admitted), "dominant_failure_histogram": dict(sorted(hist.items())), "admitted_candidate_ids": [c["candidate_id"] for c in admitted], "results": results}
    write_json(run_dir / ("candidate_admission_report.json" if repair_cycle == 0 else f"candidate_admission_report_repair_cycle_{repair_cycle}.json"), report)
    return admitted, results


def select_targeted_cases(admitted: list[dict[str, Any]], fallback_results: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in admitted:
        by_source[str(c.get("co_design_source_type"))].append(c)
    ordered: list[dict[str, Any]] = []
    for source, minimum in [("generated_variant", 2), ("repaired_layout", 2), ("old_variant", 1)]:
        ordered.extend(by_source.get(source, [])[:minimum])
    for c in admitted:
        if c not in ordered:
            ordered.append(c)
    cases: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    perturb_names = list(DEFAULT_TARGETED_PERTURBATIONS)
    for idx in range(min(7, len(ordered))):
        cand = ordered[idx]
        variant = cand.get("selected_pull_variant") or PROBE_VARIANTS[0]
        cases.append((cand, perturb_names[idx % len(perturb_names)], variant))
    return cases


def stage4_targeted_shard(run_dir: Path, admitted: list[dict[str, Any]], oracle_results: list[dict[str, Any]], repair_cycle: int = 0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cases = select_targeted_cases(admitted, oracle_results)
    plan = {
        "generated_at_utc": utc_now(),
        "repair_cycle": repair_cycle,
        "case_count": len(cases),
        "cases": [
            {"candidate_id": c["candidate_id"], "source_type": c.get("co_design_source_type"), "perturbation": p, "variant_name": v.get("name")}
            for c, p, v in cases
        ],
        "source_mix": dict(Counter(str(c.get("co_design_source_type")) for c, _, _ in cases)),
        "hard_fail_reasons": [],
    }
    if len(cases) < 7:
        plan["hard_fail_reasons"].append("TARGETED_CASE_COUNT_LT_7")
    if plan["source_mix"].get("generated_variant", 0) < 2:
        plan["hard_fail_reasons"].append("GENERATED_VARIANT_COVERAGE_LT_2")
    if plan["source_mix"].get("repaired_layout", 0) < 2:
        plan["hard_fail_reasons"].append("REPAIRED_LAYOUT_COVERAGE_LT_2")
    if plan["source_mix"].get("old_variant", 0) < 1:
        plan["hard_fail_reasons"].append("OLD_VARIANT_COVERAGE_LT_1")
    write_json(run_dir / ("targeted_shard_plan.json" if repair_cycle == 0 else f"targeted_shard_plan_repair_cycle_{repair_cycle}.json"), plan)
    rows: list[dict[str, Any]] = []
    out = run_dir / ("targeted_shard_results.jsonl" if repair_cycle == 0 else f"targeted_shard_results_repair_cycle_{repair_cycle}.jsonl")
    if out.exists():
        out.unlink()
    if not plan["hard_fail_reasons"]:
        for idx, (candidate, perturbation, variant) in enumerate(cases, start=1):
            row = run_case(candidate, perturbation, variant, run_dir, 20_000 + repair_cycle * 100 + idx, "targeted_shard")
            rows.append(row)
            append_jsonl(out, row)
    summary = summarize_rows(rows, "targeted_shard") if rows else {"generated_at_utc": utc_now(), "cases_total": len(cases), "cases_passed": 0, "cases_failed": len(cases), "matrix_passed": False, "failure_histogram": {reason: 1 for reason in plan["hard_fail_reasons"]}}
    summary["targeted_shard_passed"] = bool(not plan["hard_fail_reasons"] and rows and all(r.get("passed") for r in rows) and len(rows) >= 7)
    summary["repair_cycle"] = repair_cycle
    write_json(run_dir / ("targeted_shard_results.json" if repair_cycle == 0 else f"targeted_shard_results_repair_cycle_{repair_cycle}.json"), summary)
    return summary, rows


def run_targeted_with_repairs(run_dir: Path, initial_admitted: list[dict[str, Any]], initial_results: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_admitted = list(initial_admitted)
    all_oracle_results = list(initial_results)
    repair_rows: list[dict[str, Any]] = []
    best_summary: dict[str, Any] | None = None
    best_rows: list[dict[str, Any]] = []
    summary, rows = stage4_targeted_shard(run_dir, all_admitted, all_oracle_results, 0)
    best_summary, best_rows = summary, rows
    append_jsonl(run_dir / "targeted_shard_repair_cycles.jsonl", {"repair_cycle": 0, "summary": summary, "action": "initial_targeted_shard"})
    if summary.get("targeted_shard_passed"):
        return summary, rows, all_admitted, all_oracle_results
    for cycle in range(1, 4):
        candidates = stage2_candidate_pool(run_dir, repair_cycle=cycle)
        admitted, oracle_results = stage3_joint_oracle(run_dir, candidates, REPAIR_VARIANTS, repair_cycle=cycle)
        all_admitted.extend(admitted)
        all_oracle_results.extend(oracle_results)
        summary, rows = stage4_targeted_shard(run_dir, all_admitted, all_oracle_results, cycle)
        repair_action = {
            "repair_cycle": cycle,
            "action": "regenerate_candidate_distribution_and_reprobe_pull_wrench",
            "new_candidates": len(candidates),
            "new_admitted": len(admitted),
            "targeted_summary": summary,
        }
        append_jsonl(run_dir / "targeted_shard_repair_cycles.jsonl", repair_action)
        if (summary.get("cases_passed", 0), summary.get("max_drawer_fraction_max", 0.0)) > (best_summary.get("cases_passed", 0), best_summary.get("max_drawer_fraction_max", 0.0)):
            best_summary, best_rows = summary, rows
        if summary.get("targeted_shard_passed"):
            return summary, rows, all_admitted, all_oracle_results
    return best_summary or {}, best_rows, all_admitted, all_oracle_results


def stage5_full30(run_dir: Path, admitted: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in admitted:
        by_source[str(c.get("co_design_source_type"))].append(c)
    selected = []
    for src in ["generated_variant", "repaired_layout", "old_variant"]:
        for c in by_source.get(src, []):
            if c not in selected:
                selected.append(c)
            if len(selected) >= 5:
                break
        if len(selected) >= 5:
            break
    for c in admitted:
        if len(selected) >= 5:
            break
        if c not in selected:
            selected.append(c)
    rows: list[dict[str, Any]] = []
    out = run_dir / "full30_dynamic_pull_certification.jsonl"
    if out.exists():
        out.unlink()
    if len(selected) < 5:
        summary = {"generated_at_utc": utc_now(), "full30_certification_attempted": False, "cases_total": 0, "cases_passed": 0, "cases_failed": 0, "matrix_passed": False, "failure_histogram": {"ADMITTED_CANDIDATE_COUNT_LT_5": 1}, "selected_candidate_ids": [c["candidate_id"] for c in selected]}
        write_json(run_dir / "full30_dynamic_pull_certification_report.json", summary)
        write_md(run_dir / "full30_dynamic_pull_certification_report.md", "# Full30 Dynamic Pull Certification\n\nNot attempted: fewer than 5 admitted candidates.")
        return summary, rows
    for cidx, candidate in enumerate(selected, start=1):
        variant = candidate.get("selected_pull_variant") or PROBE_VARIANTS[0]
        for pidx, perturb in enumerate(PERTURBATIONS[:6], start=1):
            row = run_case(candidate, perturb["name"], variant, run_dir, 30_000 + cidx * 100 + pidx, "full30_dynamic_pull")
            rows.append(row)
            append_jsonl(out, row)
    summary = summarize_rows(rows, "full30_dynamic_pull_certification")
    summary["full30_certification_attempted"] = True
    summary["selected_candidate_ids"] = [c["candidate_id"] for c in selected]
    write_json(run_dir / "full30_dynamic_pull_certification_report.json", summary)
    lines = ["# Full30 Dynamic Pull Certification", "", f"- cases: `{summary['cases_passed']}/{summary['cases_total']}`", f"- matrix_passed: `{summary['matrix_passed']}`", f"- failure_histogram: `{json.dumps(summary['failure_histogram'], sort_keys=True)}`"]
    write_md(run_dir / "full30_dynamic_pull_certification_report.md", "\n".join(lines))
    return summary, rows


def strict_export(run_dir: Path, full_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not full_rows or not all(r.get("passed") for r in full_rows):
        payload = {"generated_at_utc": utc_now(), "strict_teacher_export_attempted": False, "strict_teacher_export_complete": False, "reason": "full30_not_certified"}
        write_json(run_dir / "strict_teacher_export_manifest.json", payload)
        write_md(run_dir / "strict_teacher_export_report.md", "# Strict Teacher Export\n\nNot attempted: full30 certification did not pass.")
        return payload
    bundle = run_dir / "strict_teacher_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    trace_items = []
    for row in full_rows:
        trace = ROOT / str(row.get("trace_jsonl"))
        if not trace.exists():
            trace_items.append({"trace_jsonl": row.get("trace_jsonl"), "exists": False})
            continue
        trace_items.append({"trace_jsonl": rel(trace), "exists": True, "sha256": sha256_file(trace), "size_bytes": trace.stat().st_size, "candidate_id": row.get("candidate_id"), "perturbation": row.get("perturbation")})
    complete = bool(trace_items and all(item.get("exists") for item in trace_items))
    payload = {"generated_at_utc": utc_now(), "strict_teacher_export_attempted": True, "strict_teacher_export_complete": complete, "bundle_dir": rel(bundle), "source_head": run_git(["rev-parse", "HEAD"]), "trace_count": len(trace_items), "trace_items": trace_items}
    write_json(run_dir / "strict_teacher_export_manifest.json", payload)
    write_json(bundle / "strict_teacher_bundle_hashes.json", payload)
    write_md(run_dir / "strict_teacher_export_report.md", f"# Strict Teacher Export\n\n- complete: `{complete}`\n- trace_count: `{len(trace_items)}`")
    return payload


def local_replay_placeholder(run_dir: Path, export_manifest: dict[str, Any]) -> dict[str, Any]:
    if not export_manifest.get("strict_teacher_export_complete"):
        payload = {"generated_at_utc": utc_now(), "local_strict_replay_attempted": False, "local_strict_replay_render_passed": False, "reason": "strict_export_not_complete"}
    else:
        # The actual local replay is executed from Playground outside this remote helper. This helper records a handoff only.
        local_dir = LOCAL_PLAYGROUND_ROOT / "local_replay" / f"v11_g4_dynamic_pull_accessible_pool_co_design_{utc_stamp()}"
        payload = {"generated_at_utc": utc_now(), "local_strict_replay_attempted": False, "local_strict_replay_render_passed": False, "closeout_if_stopped_here": "LOCAL_STRICT_REPLAY_RENDER_FAILED", "local_replay_dir": str(local_dir), "reason": "remote_helper_cannot_verify_local_render_without local runner invocation"}
    write_json(run_dir / "local_replay_handoff_manifest.json", payload)
    return payload


def write_deltas(run_dir: Path, closeout: dict[str, Any]) -> None:
    current_delta = {
        "proposal_id": "proposed_current_truth_delta_dynamic_pull_accessible_pool_co_design",
        "generated_at_utc": utc_now(),
        "source_run_dir": rel(run_dir),
        "fixed_pool_used_only_as_negative_baseline": closeout.get("fixed_pool_used_only_as_negative_baseline"),
        "old_fixed_pool_modified": closeout.get("old_fixed_pool_modified"),
        "candidate_pool_total": closeout.get("candidate_pool_total"),
        "candidate_pool_admitted_count": closeout.get("candidate_pool_admitted_count"),
        "targeted_shard_passed": closeout.get("targeted_shard_passed"),
        "full30_certification_attempted": closeout.get("full30_certification_attempted"),
        "strict_teacher_export_complete": closeout.get("strict_teacher_export_complete"),
        "local_strict_replay_render_passed": closeout.get("local_strict_replay_render_passed"),
        "teacher_rollout_remains_blocked": closeout.get("closeout_classification") != "DYNAMIC_PULL_ACCESSIBLE_POOL_STRICT_EXPORT_LOCAL_REPLAY_READY",
        "next_gate": closeout.get("next_gate"),
    }
    next_actions = {
        "proposal_id": "proposed_next_actions_dynamic_pull_accessible_pool_co_design",
        "generated_at_utc": utc_now(),
        "recommended_next_gate": closeout.get("next_gate"),
        "closeout_classification": closeout.get("closeout_classification"),
        "required_action": "review co-design defect certificates before widening all-seed scope",
    }
    write_json(CAMPAIGN / "sovereign/proposed_current_truth_delta_dynamic_pull_accessible_pool_co_design.json", current_delta)
    write_json(CAMPAIGN / "sovereign/proposed_next_actions_dynamic_pull_accessible_pool_co_design.json", next_actions)


def final_checks(run_dir: Path) -> dict[str, Any]:
    pyc = run_cmd(["/root/anaconda3/envs/infinigen/bin/python", "-m", "py_compile", "scripts/mint/v11_g4_dynamic_pull_accessible_pool_co_design.py"])
    ruff = run_cmd(["/root/anaconda3/envs/infinigen/bin/python", "-m", "ruff", "check", "scripts/mint/v11_g4_dynamic_pull_accessible_pool_co_design.py"])
    preflight = run_cmd(["/root/anaconda3/envs/infinigen/bin/python", "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py", "--spec", SPEC_REL, "--dry-run"])
    json_errors = []
    for path in list(run_dir.rglob("*.json")) + [CAMPAIGN / "sovereign/proposed_current_truth_delta_dynamic_pull_accessible_pool_co_design.json", CAMPAIGN / "sovereign/proposed_next_actions_dynamic_pull_accessible_pool_co_design.json"]:
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
    current_truth_modified = "experiments/mint/mint_drawer_v1/sovereign/current_truth.json" in status
    next_actions_modified = "experiments/mint/mint_drawer_v1/sovereign/next_actions.json" in status
    external_modified = "external/MINT" in status
    payload = {
        "generated_at_utc": utc_now(),
        "preflight": preflight,
        "harness_preflight_passed": preflight["returncode"] == 0,
        "py_compile": pyc,
        "py_compile_passed": pyc["returncode"] == 0,
        "ruff": ruff,
        "ruff_passed": ruff["returncode"] == 0,
        "json_errors": json_errors,
        "json_parse_passed": not json_errors,
        "current_truth_modified": current_truth_modified,
        "next_actions_modified": next_actions_modified,
        "external_mint_modified": external_modified,
        "git_status_short": status,
    }
    payload["final_checks_passed"] = bool(payload["harness_preflight_passed"] and payload["py_compile_passed"] and payload["ruff_passed"] and payload["json_parse_passed"] and not current_truth_modified and not next_actions_modified and not external_modified)
    write_json(run_dir / "stage9_final_checks.json", payload)
    return payload


def classify_closeout(targeted: dict[str, Any], full30: dict[str, Any], export: dict[str, Any], local: dict[str, Any]) -> tuple[str, str]:
    if targeted.get("targeted_shard_passed") is not True:
        return "TARGETED_DYNAMIC_PULL_POOL_SHARD_FAILED", "ANALYZE_DYNAMIC_PULL_POOL_CO_DESIGN_DEFECT_CERTIFICATES"
    if full30.get("matrix_passed") is not True:
        return "FULL_DYNAMIC_PULL_CERTIFICATION_FAILED", "REPAIR_DYNAMIC_PULL_POOL_CERTIFICATION_DEFECTS"
    if export.get("strict_teacher_export_complete") is not True:
        return "STRICT_TEACHER_EXPORT_FAILED", "STRICT_EXPORT_INFRA_REPAIR"
    if local.get("local_strict_replay_render_passed") is not True:
        return "LOCAL_STRICT_REPLAY_RENDER_FAILED", "LOCAL_STRICT_REPLAY_RENDER_REPAIR"
    return "DYNAMIC_PULL_ACCESSIBLE_POOL_STRICT_EXPORT_LOCAL_REPLAY_READY", "MANUAL_VISUAL_AND_SCIENCE_REVIEW_BEFORE_MINT_DATASET_ADMISSION"


def build_closeout(run_dir: Path, stage0_payload: dict[str, Any], defects: dict[str, Any], all_candidates: list[dict[str, Any]], admitted: list[dict[str, Any]], targeted_summary: dict[str, Any], targeted_rows: list[dict[str, Any]], full30_summary: dict[str, Any], full30_rows: list[dict[str, Any]], export_manifest: dict[str, Any], local_manifest: dict[str, Any], checks: dict[str, Any]) -> dict[str, Any]:
    closeout_classification, next_gate = classify_closeout(targeted_summary, full30_summary, export_manifest, local_manifest)
    defect_hist = defects.get("defect_histogram", {})
    best_rows = targeted_rows or full30_rows
    best = max(best_rows, key=row_rank) if best_rows else {}
    closeout = {
        "task_id": TASK_ID,
        "generated_at_utc": utc_now(),
        "closeout_classification": closeout_classification,
        "harness_preflight_passed": bool(stage0_payload.get("harness_preflight_passed") and checks.get("harness_preflight_passed")),
        "task_spec_lock_bound": bool(stage0_payload.get("harness_preflight_passed")),
        "fixed_pool_used_only_as_negative_baseline": True,
        "old_fixed_pool_modified": False,
        "candidate_pool_generated": True,
        "candidate_pool_total": len(all_candidates),
        "candidate_pool_admitted_count": len(admitted),
        "targeted_shard_passed": bool(targeted_summary.get("targeted_shard_passed")),
        "targeted_shard_cases_total": int(targeted_summary.get("cases_total", 0) or 0),
        "targeted_shard_cases_passed": int(targeted_summary.get("cases_passed", 0) or 0),
        "full30_certification_attempted": bool(full30_summary.get("full30_certification_attempted", False)),
        "full30_cases_passed": int(full30_summary.get("cases_passed", 0) or 0),
        "full30_cases_failed": int(full30_summary.get("cases_failed", 0) or 0),
        "strict_teacher_export_attempted": bool(export_manifest.get("strict_teacher_export_attempted", False)),
        "strict_teacher_export_complete": bool(export_manifest.get("strict_teacher_export_complete", False)),
        "local_strict_replay_attempted": bool(local_manifest.get("local_strict_replay_attempted", False)),
        "local_strict_replay_render_passed": bool(local_manifest.get("local_strict_replay_render_passed", False)),
        "controller_defect_count": int(defect_hist.get("CONTROLLER_DEFECT", 0)),
        "structure_defect_count": int(defect_hist.get("STRUCTURE_DEFECT", 0)),
        "mixed_defect_count": int(defect_hist.get("MIXED_DEFECT", 0)),
        "best_candidate_drawer_fraction": float(best.get("max_drawer_fraction", 0.0) or 0.0),
        "best_candidate_forbidden_contact_frames": int(best.get("forbidden_contact_frames", 0) or 0),
        "best_candidate_handle_nonlegal_contact_frames": int(best.get("handle_nonlegal_contact_frames", 0) or 0),
        "best_candidate_max_penetration_m": float(best.get("max_penetration_m", 0.0) or 0.0),
        "best_candidate_drawer_qpos_monotonic": bool(best.get("drawer_qpos_nondecreasing_with_tolerance", False)),
        "current_truth_modified": bool(checks.get("current_truth_modified", False)),
        "next_actions_modified": bool(checks.get("next_actions_modified", False)),
        "runtime_patch_applied": True,
        "runtime_patch_files": ["scripts/mint/v11_g4_dynamic_pull_accessible_pool_co_design.py"],
        "committed": False,
        "pushed_to_origin": False,
        "remote_commit_hash": None,
        "next_gate": next_gate,
        "targeted_failure_histogram": targeted_summary.get("failure_histogram", {}),
        "full30_failure_histogram": full30_summary.get("failure_histogram", {}),
    }
    write_json(run_dir / "final_closeout.json", closeout)
    write_json(run_dir / "closeout_decision.json", closeout)
    lines = [
        "# Dynamic Pull Accessible Pool Co-Design Closeout",
        "",
        f"- closeout_classification: `{closeout_classification}`",
        f"- next_gate: `{next_gate}`",
        f"- candidate_pool_admitted_count: `{len(admitted)}`",
        f"- targeted_shard: `{closeout['targeted_shard_cases_passed']}/{closeout['targeted_shard_cases_total']}`",
        f"- full30_attempted: `{closeout['full30_certification_attempted']}`",
        f"- strict_export_complete: `{closeout['strict_teacher_export_complete']}`",
        f"- local_replay_render_passed: `{closeout['local_strict_replay_render_passed']}`",
        "",
        "The prior fixed pool was used only as a negative baseline; it was not promoted to strict export.",
    ]
    write_md(run_dir / "final_closeout.md", "\n".join(lines))
    write_md(run_dir / "final_report.md", "\n".join(lines))
    return closeout


def post_push_verify(run_dir: Path) -> dict[str, Any]:
    branch = run_git(["branch", "--show-current"])
    local_head = run_git(["rev-parse", "HEAD"])
    remote_line = run_git(["ls-remote", "my-origin", f"refs/heads/{branch}"])
    remote_head = remote_line.split()[0] if remote_line else ""
    paths = [
        SPEC_REL,
        "scripts/mint/v11_g4_dynamic_pull_accessible_pool_co_design.py",
        rel(run_dir / "final_closeout.json"),
        rel(run_dir / "candidate_oracle_results.jsonl"),
        rel(run_dir / "targeted_shard_results.json"),
        rel(run_dir / "stage9_final_checks.json"),
        "experiments/mint/mint_drawer_v1/sovereign/proposed_current_truth_delta_dynamic_pull_accessible_pool_co_design.json",
        "experiments/mint/mint_drawer_v1/sovereign/proposed_next_actions_dynamic_pull_accessible_pool_co_design.json",
    ]
    checks = []
    for path in paths:
        if path:
            exists = subprocess.run(["git", "cat-file", "-e", f"{remote_head}:{path}"], cwd=ROOT).returncode == 0 if remote_head else False
            checks.append({"path": path, "origin_visible": exists})
    payload = {"generated_at_utc": utc_now(), "branch": branch, "local_head": local_head, "remote_head": remote_head, "origin_head_matches_local": local_head == remote_head, "path_checks": checks, "all_paths_origin_visible": all(c["origin_visible"] for c in checks)}
    write_json(run_dir / "post_push_verification.json", payload)
    return payload


def run_phase(run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    stage0_payload = stage0(run_dir)
    defects = stage1_defect_certificate(run_dir)
    candidates = stage2_candidate_pool(run_dir)
    admitted, oracle_results = stage3_joint_oracle(run_dir, candidates, PROBE_VARIANTS)
    targeted_summary, targeted_rows, all_admitted, all_oracles = run_targeted_with_repairs(run_dir, admitted, oracle_results)
    all_candidates_total = candidates
    for path in sorted(run_dir.glob("candidate_dynamic_pull_pool_repair_cycle_*.json")):
        data = load_json(path, {})
        all_candidates_total.extend(data.get("candidates", []))
    if targeted_summary.get("targeted_shard_passed"):
        full30_summary, full30_rows = stage5_full30(run_dir, all_admitted)
    else:
        full30_summary, full30_rows = {"generated_at_utc": utc_now(), "full30_certification_attempted": False, "cases_total": 0, "cases_passed": 0, "cases_failed": 0, "matrix_passed": False, "failure_histogram": {"targeted_shard_not_passed": 1}}, []
        write_json(run_dir / "full30_dynamic_pull_certification_report.json", full30_summary)
        write_md(run_dir / "full30_dynamic_pull_certification_report.md", "# Full30 Dynamic Pull Certification\n\nNot attempted because targeted shard did not pass.")
    export_manifest = strict_export(run_dir, full30_rows if full30_summary.get("matrix_passed") else [])
    local_manifest = local_replay_placeholder(run_dir, export_manifest)
    provisional_checks = {"harness_preflight_passed": True, "current_truth_modified": False, "next_actions_modified": False}
    provisional_closeout = build_closeout(run_dir, stage0_payload, defects, all_candidates_total, all_admitted, targeted_summary, targeted_rows, full30_summary, full30_rows, export_manifest, local_manifest, provisional_checks)
    write_deltas(run_dir, provisional_closeout)
    checks = final_checks(run_dir)
    closeout = build_closeout(run_dir, stage0_payload, defects, all_candidates_total, all_admitted, targeted_summary, targeted_rows, full30_summary, full30_rows, export_manifest, local_manifest, checks)
    write_deltas(run_dir, closeout)
    return closeout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir")
    parser.add_argument("--post-push-verify", action="store_true")
    args = parser.parse_args()
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
    else:
        run_dir = CAMPAIGN / "runtime" / f"{RUN_PREFIX}_{utc_stamp()}"
    if args.post_push_verify:
        payload = post_push_verify(run_dir)
        print(json.dumps(ready(payload), indent=2, sort_keys=True))
        return
    closeout = run_phase(run_dir)
    print(json.dumps({"run_dir": rel(run_dir), "closeout_classification": closeout.get("closeout_classification"), "targeted_shard_cases_passed": closeout.get("targeted_shard_cases_passed"), "targeted_shard_cases_total": closeout.get("targeted_shard_cases_total"), "next_gate": closeout.get("next_gate")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

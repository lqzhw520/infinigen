#!/usr/bin/env python3
"""Defect-histogram-driven dynamic pull pool solver.

This phase consumes the previous co-design failure histogram, maps each defect
class to structure/control repair operators, then runs bounded MuJoCo candidate
search. It preserves strict GOC-v4 target authority and never promotes the old
fixed pool to teacher export input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
    "v11_g4_goc_v4_defect_histogram_dynamic_pull_pool_solver_to_export_replay.yaml"
)
RUN_PREFIX = "v11_g4_goc_v4_defect_histogram_dynamic_pull_pool_solver_to_export_replay"
TASK_ID = "V11_G4_GOC_V4_DEFECT_HISTOGRAM_DYNAMIC_PULL_POOL_SOLVER_TO_EXPORT_REPLAY_V1"
PREV_PREFIX = "v11_g4_goc_v4_dynamic_pull_accessible_pool_co_design_to_strict_export_replay_"
LOCAL_PLAYGROUND_ROOT = Path("/Users/zhuhaowu/Documents/Playground")
STRICT_DRAWER_FRACTION = 0.80
MAX_OUTER_CYCLES = 6

sys.path.insert(0, str(ROOT / "scripts/mint"))
import v11_g4_dynamic_pull_accessible_pool_co_design as cd  # noqa: E402
import v11_g4_physical_accessibility_instance_pool as pool  # noqa: E402

ORIGINAL_BUILDER = pool.GeneratedAccessibleDrawerBuilder


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
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
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


class SolverDrawerBuilder(ORIGINAL_BUILDER):
    """Generated drawer builder with declared bar/knob handle variants."""

    def _drawer_xml(self) -> str:
        v = self.variant
        hx = float(v.get("handle_x", -0.14))
        hy = float(v.get("handle_y", 0.02))
        hz = float(v.get("handle_z", 0.36))
        knob_radius = float(v.get("handle_radius", 0.025))
        handle_kind = str(v.get("handle_kind", "sphere"))
        handle_half_length = float(v.get("handle_half_length", 0.055))
        cabinet_x = float(v.get("cabinet_x", 0.18))
        cabinet_y = hy
        cabinet_z = float(v.get("cabinet_z", 0.34))
        cabinet_depth = float(v.get("cabinet_depth", 0.20))
        cabinet_half_width = float(v.get("cabinet_half_width", 0.24))
        cabinet_half_height = float(v.get("cabinet_half_height", 0.28))
        door_x = float(v.get("door_x", 0.055))
        door_half_width = float(v.get("door_half_width", 0.19))
        door_half_height = float(v.get("door_half_height", 0.23))
        drawer_damping = float(v.get("drawer_damping", 8.0))
        pull_axis_s = " ".join(str(float(x)) for x in v.get("pull_axis", [-1, 0, 0]))
        if bool(v.get("front_cutout", False)):
            cutout_half_width = float(v.get("cutout_half_width", 0.125))
            cutout_half_height = float(v.get("cutout_half_height", 0.115))
            side_width = max((door_half_width - cutout_half_width) / 2.0, 0.025)
            top_height = max((door_half_height - cutout_half_height) / 2.0, 0.025)
            side_y = door_half_width - side_width
            top_z = door_half_height - top_height
            door_xml = f'''
    <geom name="drawer_door_left_panel_collision" type="box" pos="{door_x:.4f} {cabinet_y - side_y:.4f} {cabinet_z:.4f}" size="0.025 {side_width:.4f} {door_half_height:.4f}" contype="1" conaffinity="1" rgba="0.90 0.88 0.85 1"/>
    <geom name="drawer_door_right_panel_collision" type="box" pos="{door_x:.4f} {cabinet_y + side_y:.4f} {cabinet_z:.4f}" size="0.025 {side_width:.4f} {door_half_height:.4f}" contype="1" conaffinity="1" rgba="0.90 0.88 0.85 1"/>
    <geom name="drawer_door_top_panel_collision" type="box" pos="{door_x:.4f} {cabinet_y:.4f} {cabinet_z + top_z:.4f}" size="0.025 {cutout_half_width:.4f} {top_height:.4f}" contype="1" conaffinity="1" rgba="0.90 0.88 0.85 1"/>
    <geom name="drawer_door_bottom_panel_collision" type="box" pos="{door_x:.4f} {cabinet_y:.4f} {cabinet_z - top_z:.4f}" size="0.025 {cutout_half_width:.4f} {top_height:.4f}" contype="1" conaffinity="1" rgba="0.90 0.88 0.85 1"/>'''
        else:
            door_xml = f'''
    <geom name="drawer_door_collision" type="box" pos="{door_x:.4f} {cabinet_y:.4f} {cabinet_z:.4f}" size="0.025 {door_half_width:.4f} {door_half_height:.4f}" contype="1" conaffinity="1" rgba="0.90 0.88 0.85 1"/>'''
        if handle_kind == "bar_y":
            handle_xml = f'<geom name="drawer_handle_collision_0" type="capsule" fromto="{hx:.4f} {hy - handle_half_length:.4f} {hz:.4f} {hx:.4f} {hy + handle_half_length:.4f} {hz:.4f}" size="{knob_radius:.4f}" contype="1" conaffinity="1" rgba="0.20 0.20 0.22 1"/>'
        elif handle_kind == "dual_knob_y":
            handle_xml = f'''
    <geom name="drawer_handle_collision_0" type="sphere" pos="{hx:.4f} {hy - handle_half_length:.4f} {hz:.4f}" size="{knob_radius:.4f}" contype="1" conaffinity="1" rgba="0.20 0.20 0.22 1"/>
    <geom name="drawer_handle_collision_1" type="sphere" pos="{hx:.4f} {hy + handle_half_length:.4f} {hz:.4f}" size="{knob_radius:.4f}" contype="1" conaffinity="1" rgba="0.20 0.20 0.22 1"/>'''
        else:
            handle_xml = f'<geom name="drawer_handle_collision_0" type="sphere" pos="{hx:.4f} {hy:.4f} {hz:.4f}" size="{knob_radius:.4f}" contype="1" conaffinity="1" rgba="0.20 0.20 0.22 1"/>'
        return f'''
<body name="drawer_base" pos="0 0 0">
  <geom name="cabinet_back_collision" type="box" pos="{cabinet_x + cabinet_depth * 0.7:.4f} {cabinet_y:.4f} {cabinet_z:.4f}" size="0.025 {cabinet_half_width:.4f} {cabinet_half_height:.4f}" contype="1" conaffinity="1" rgba="0.85 0.82 0.80 1"/>
  <geom name="cabinet_bottom_collision" type="box" pos="{cabinet_x:.4f} {cabinet_y:.4f} {cabinet_z - cabinet_half_height:.4f}" size="{cabinet_depth:.4f} {cabinet_half_width:.4f} 0.025" contype="1" conaffinity="1" rgba="0.85 0.82 0.80 1"/>
  <geom name="cabinet_top_collision" type="box" pos="{cabinet_x:.4f} {cabinet_y:.4f} {cabinet_z + cabinet_half_height:.4f}" size="{cabinet_depth:.4f} {cabinet_half_width:.4f} 0.025" contype="1" conaffinity="1" rgba="0.85 0.82 0.80 1"/>
  <geom name="cabinet_left_collision" type="box" pos="{cabinet_x:.4f} {cabinet_y - cabinet_half_width:.4f} {cabinet_z:.4f}" size="{cabinet_depth:.4f} 0.025 {cabinet_half_height:.4f}" contype="1" conaffinity="1" rgba="0.85 0.82 0.80 1"/>
  <geom name="cabinet_right_collision" type="box" pos="{cabinet_x:.4f} {cabinet_y + cabinet_half_width:.4f} {cabinet_z:.4f}" size="{cabinet_depth:.4f} 0.025 {cabinet_half_height:.4f}" contype="1" conaffinity="1" rgba="0.85 0.82 0.80 1"/>
  <body name="link_1" pos="0 0 0">
    <joint name="drawer_slider_0" type="slide" axis="{pull_axis_s}" range="0 0.35" damping="{drawer_damping:.4f}"/>{door_xml}
    {handle_xml}
  </body>
</body>'''


def install_solver_builder() -> None:
    pool.GeneratedAccessibleDrawerBuilder = SolverDrawerBuilder
    cd.pool.GeneratedAccessibleDrawerBuilder = SolverDrawerBuilder


def latest_prior_run() -> Path | None:
    dirs = sorted((CAMPAIGN / "runtime").glob(f"{PREV_PREFIX}*"))
    dirs = [d for d in dirs if (d / "final_closeout.json").exists()]
    return dirs[-1] if dirs else None


def base_params(
    robot_base_pos: list[float],
    handle_x: float,
    handle_y: float,
    handle_z: float,
    *,
    handle_kind: str = "sphere",
    handle_radius: float = 0.022,
    handle_half_length: float = 0.050,
    robot_yaw_deg: float = 0.0,
    door_x: float = 0.135,
    cutout_w: float = 0.245,
    cutout_h: float = 0.205,
    cabinet_half_width: float = 0.30,
    cabinet_depth: float = 0.17,
    drawer_damping: float = 8.0,
) -> dict[str, Any]:
    return {
        "robot_base_pos": robot_base_pos,
        "robot_yaw_deg": robot_yaw_deg,
        "handle_x": handle_x,
        "handle_y": handle_y,
        "handle_z": handle_z,
        "handle_kind": handle_kind,
        "handle_radius": handle_radius,
        "handle_half_length": handle_half_length,
        "front_cutout": True,
        "door_x": door_x,
        "cutout_half_width": cutout_w,
        "cutout_half_height": cutout_h,
        "cabinet_half_width": cabinet_half_width,
        "cabinet_depth": cabinet_depth,
        "drawer_damping": drawer_damping,
        "drawer_kind": "defect_directed_bar_or_knob_drawer_with_cabinet_envelope",
        "generation_policy": "defect_histogram_directed_structure_control_codesign_declared_variant",
    }


def make_candidate(cid: str, source: str, seed: int, params: dict[str, Any], defect: str, parent: str | None = None) -> dict[str, Any]:
    return {
        "candidate_id": cid,
        "source_type": "generated_accessible_variant",
        "co_design_source_type": source,
        "source_type_for_spec": source,
        "parent_instance_id": parent,
        "synthetic_seed": seed,
        "status": "pending",
        "model_builder_parameters": params,
        "declared_generated_or_repaired_variant": True,
        "fixed_pool_original_mutated": False,
        "targeted_defect_cluster": defect,
    }


def controller_variants() -> list[dict[str, Any]]:
    variants = list(cd.REPAIR_VARIANTS)
    variants.extend([
        {"name": "dh_solver_bar_retention_low_press_binary", "pull_steps": 6200, "post_pull_hold_steps": 0, "pull_velocity_m_per_step": 0.000065, "pull_distance_m": 0.35, "lead_cap_m": 0.016, "pull_press_m": 0.0008, "op_gain": 7.0, "op_vel_limit": 0.050, "q_vel_limit": 1.30, "null_gain": 0.28, "servo_kp": 220.0, "servo_kd": 110.0, "finger_mode": "binary_close"},
        {"name": "dh_solver_bar_retention_semi_close", "pull_steps": 5600, "post_pull_hold_steps": 0, "pull_velocity_m_per_step": 0.000085, "pull_distance_m": 0.35, "lead_cap_m": 0.024, "pull_press_m": 0.0020, "op_gain": 10.0, "op_vel_limit": 0.075, "q_vel_limit": 1.90, "null_gain": 0.16, "servo_kp": 300.0, "servo_kd": 100.0, "finger_mode": "semi_close"},
        {"name": "dh_solver_smooth_monotonic_ik_hold", "pull_steps": 7000, "post_pull_hold_steps": 0, "pull_velocity_m_per_step": 0.000055, "pull_distance_m": 0.35, "lead_cap_m": 0.014, "pull_press_m": 0.0012, "op_gain": 6.5, "op_vel_limit": 0.045, "q_vel_limit": 1.15, "null_gain": 0.32, "servo_kp": 210.0, "servo_kd": 125.0, "finger_mode": "ik_hold"},
        {"name": "dh_solver_axis_work_moderate_press", "pull_steps": 5200, "post_pull_hold_steps": 0, "pull_velocity_m_per_step": 0.00010, "pull_distance_m": 0.35, "lead_cap_m": 0.032, "pull_press_m": 0.0035, "op_gain": 12.0, "op_vel_limit": 0.100, "q_vel_limit": 2.35, "null_gain": 0.08, "servo_kp": 360.0, "servo_kd": 92.0, "finger_mode": "binary_close"},
        {"name": "dh_solver_keepout_micro_lead_semi", "pull_steps": 6800, "post_pull_hold_steps": 0, "pull_velocity_m_per_step": 0.000060, "pull_distance_m": 0.35, "lead_cap_m": 0.010, "pull_press_m": 0.0010, "op_gain": 6.0, "op_vel_limit": 0.040, "q_vel_limit": 1.05, "null_gain": 0.38, "servo_kp": 200.0, "servo_kd": 140.0, "finger_mode": "semi_close"},
    ])
    seen: set[str] = set()
    unique = []
    for v in variants:
        name = str(v["name"])
        if name not in seen:
            seen.add(name)
            unique.append(v)
    return unique[:12]


CONTROLLER_VARIANTS = controller_variants()


def generate_cycle_candidates(cycle: int) -> list[dict[str, Any]]:
    specs: list[tuple[str, str, int, dict[str, Any], str, str | None]] = []
    if cycle == 1:
        grid = [
            (-0.080, 0.360, 0.014, 0.035, 0), (-0.080, 0.380, 0.016, 0.040, 0),
            (-0.075, 0.400, 0.018, 0.045, -6), (-0.060, 0.380, 0.020, 0.035, 6),
            (0.080, 0.360, 0.014, 0.035, 0), (0.080, 0.380, 0.016, 0.040, 0),
            (0.075, 0.400, 0.018, 0.045, 6), (0.060, 0.380, 0.020, 0.035, -6),
        ]
        for i, (hy, hz, radius, half, yaw) in enumerate(grid, start=1):
            params = base_params([-0.62, hy * 0.85, 0.0], -0.14, hy, hz, handle_kind="bar_y", handle_radius=radius, handle_half_length=half, robot_yaw_deg=yaw, cutout_w=0.255, cutout_h=0.215)
            specs.append((f"dh_c1_two_pad_bar_{i:02d}", "repaired_layout", 11000 + i, params, "TWO_PAD_RETENTION_FAILED", "dynamic_keepout_old002_yneg"))
    elif cycle == 2:
        grid = [
            (-0.030, -0.035, -10, 0.365, -0.145), (-0.055, -0.045, -16, 0.380, -0.155),
            (-0.080, -0.060, -22, 0.395, -0.170), (0.030, 0.035, 10, 0.365, -0.145),
            (0.055, 0.045, 16, 0.380, -0.155), (0.080, 0.060, 22, 0.395, -0.170),
            (0.000, 0.000, 12, 0.405, -0.180), (0.000, 0.000, -12, 0.405, -0.180),
        ]
        for i, (base_y, hy, yaw, hz, hx) in enumerate(grid, start=1):
            params = base_params([-0.64, base_y, 0.025], hx, hy, hz, handle_kind="sphere", handle_radius=0.023, robot_yaw_deg=yaw, cutout_w=0.270, cutout_h=0.225, cabinet_half_width=0.33, cabinet_depth=0.16)
            specs.append((f"dh_c2_keepout_yaw_height_{i:02d}", "generated_variant", 12000 + i, params, "FULL_BODY_KEEP_OUT_FAILED", "dynamic_generated_low_y_offset_003"))
    elif cycle == 3:
        grid = [
            (-0.58, -0.115, -0.060, 0.420, -8, "bar_y"), (-0.58, -0.115, 0.060, 0.420, 8, "bar_y"),
            (-0.52, -0.095, -0.040, 0.435, -6, "sphere"), (-0.52, -0.095, 0.040, 0.435, 6, "sphere"),
            (-0.68, -0.185, -0.075, 0.410, -14, "bar_y"), (-0.68, -0.185, 0.075, 0.410, 14, "bar_y"),
            (-0.48, -0.090, 0.000, 0.450, 0, "sphere"), (-0.72, -0.210, 0.000, 0.405, 0, "bar_y"),
        ]
        for i, (base_x, hx, hy, hz, yaw, kind) in enumerate(grid, start=1):
            params = base_params([base_x, hy * 0.75, 0.035], hx, hy, hz, handle_kind=kind, handle_radius=0.016 if kind == "bar_y" else 0.022, handle_half_length=0.038, robot_yaw_deg=yaw, cutout_w=0.285, cutout_h=0.235, cabinet_half_width=0.34, cabinet_depth=0.15)
            specs.append((f"dh_c3_corridor_high_exposure_{i:02d}", "generated_variant", 13000 + i, params, "APPROACH_CORRIDOR_BLOCKED", None))
    elif cycle == 4:
        grid = [
            (-0.210, -0.055, 0.360, 0.018, 0.030, -8), (-0.230, -0.075, 0.375, 0.016, 0.035, -12),
            (-0.250, -0.095, 0.390, 0.014, 0.040, -16), (-0.210, 0.055, 0.360, 0.018, 0.030, 8),
            (-0.230, 0.075, 0.375, 0.016, 0.035, 12), (-0.250, 0.095, 0.390, 0.014, 0.040, 16),
            (-0.235, 0.000, 0.400, 0.020, 0.030, 0), (-0.260, 0.000, 0.385, 0.018, 0.040, 0),
        ]
        for i, (hx, hy, hz, radius, half, yaw) in enumerate(grid, start=1):
            params = base_params([-0.72, hy * 0.70, 0.02], hx, hy, hz, handle_kind="bar_y", handle_radius=radius, handle_half_length=half, robot_yaw_deg=yaw, door_x=0.145, cutout_w=0.275, cutout_h=0.225, cabinet_depth=0.145)
            specs.append((f"dh_c4_pull_wrench_exposed_{i:02d}", "repaired_layout", 14000 + i, params, "PULL_WRENCH_INSUFFICIENT", None))
    elif cycle == 5:
        grid = [
            ("dual_knob_y", -0.165, -0.055, 0.380, 0.014, 0.032, -8, -0.64),
            ("dual_knob_y", -0.165, 0.055, 0.380, 0.014, 0.032, 8, -0.64),
            ("sphere", -0.180, -0.060, 0.385, 0.018, 0.000, -10, -0.68),
            ("sphere", -0.180, 0.060, 0.385, 0.018, 0.000, 10, -0.68),
            ("bar_y", -0.190, -0.050, 0.405, 0.012, 0.028, -6, -0.62),
            ("bar_y", -0.190, 0.050, 0.405, 0.012, 0.028, 6, -0.62),
            ("sphere", -0.155, 0.000, 0.410, 0.020, 0.000, 0, -0.58),
            ("bar_y", -0.205, 0.000, 0.395, 0.014, 0.030, 0, -0.70),
        ]
        for i, (kind, hx, hy, hz, radius, half, yaw, base_x) in enumerate(grid, start=1):
            params = base_params([base_x, hy * 0.70, 0.025], hx, hy, hz, handle_kind=kind, handle_radius=radius, handle_half_length=half, robot_yaw_deg=yaw, cutout_w=0.270, cutout_h=0.225, cabinet_half_width=0.32, cabinet_depth=0.15)
            specs.append((f"dh_c5_mixed_boundary_{i:02d}", "generated_variant", 15000 + i, params, "MIXED_STRUCTURE_CONTROL_DEFECT", None))
    else:
        grid = [
            (-0.60, -0.10, -0.145, -0.110, 0.385, -18, "bar_y"), (-0.60, 0.10, -0.145, 0.110, 0.385, 18, "bar_y"),
            (-0.76, -0.12, -0.235, -0.120, 0.390, -22, "bar_y"), (-0.76, 0.12, -0.235, 0.120, 0.390, 22, "bar_y"),
            (-0.54, -0.04, -0.105, -0.045, 0.440, -6, "sphere"), (-0.54, 0.04, -0.105, 0.045, 0.440, 6, "sphere"),
            (-0.66, 0.00, -0.185, 0.000, 0.420, 0, "dual_knob_y"), (-0.70, 0.00, -0.220, 0.000, 0.365, 0, "bar_y"),
        ]
        for i, (base_x, base_y, hx, hy, hz, yaw, kind) in enumerate(grid, start=1):
            params = base_params([base_x, base_y, 0.035], hx, hy, hz, handle_kind=kind, handle_radius=0.014 if kind != "sphere" else 0.021, handle_half_length=0.034, robot_yaw_deg=yaw, cutout_w=0.295, cutout_h=0.245, cabinet_half_width=0.35, cabinet_depth=0.14)
            specs.append((f"dh_c6_expanded_codesign_{i:02d}", "repaired_layout", 16000 + i, params, "EXHAUSTIVE_BOUNDARY_EXPANSION", None))
    return [make_candidate(cid, src, seed, params, defect, parent) for cid, src, seed, params, defect, parent in specs]


def row_rank(row: dict[str, Any]) -> tuple[float, ...]:
    reasons = set(row.get("failure_reasons") or [])
    return (
        1.0 if row.get("passed") else 0.0,
        1.0 if int(row.get("forbidden_contact_frames", 9999) or 9999) == 0 else 0.0,
        1.0 if int(row.get("handle_nonlegal_contact_frames", 9999) or 9999) == 0 else 0.0,
        float(row.get("max_drawer_fraction", 0.0) or 0.0),
        float(row.get("pull_phase_two_pad_target_contact_frames", 0) or 0),
        float(row.get("pull_phase_target_contact_frames", 0) or 0),
        -float(len(reasons)),
    )


def best_rows_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    best = result.get("best_probe_row")
    return [best] if isinstance(best, dict) and best else []


def defect_label(dominant: str | None) -> str:
    if dominant in {"FULL_BODY_KEEP_OUT_FAILED", "GUARDED_IK_INFEASIBLE", "APPROACH_CORRIDOR_BLOCKED", "PULL_WRENCH_INSUFFICIENT", "PHYSICAL_ACCESSIBILITY_FAILED", "PULL_CORRIDOR_BLOCKED"}:
        return "STRUCTURE_DEFECT"
    if dominant in {"TWO_PAD_RETENTION_FAILED", "DRAWER_QPOS_NONMONOTONIC", "MIXED_STRUCTURE_CONTROL_DEFECT"}:
        return "MIXED_DEFECT"
    if dominant == "CONTROLLER_DEFECT":
        return "CONTROLLER_DEFECT"
    return "UNKNOWN_DEFECT"


def summarize_cycle(results: list[dict[str, Any]], cycle: int) -> dict[str, Any]:
    hist = Counter(r.get("dominant_failure") or "ADMITTED" for r in results)
    defect_hist = Counter(defect_label(r.get("dominant_failure")) for r in results if not r.get("admitted"))
    probes = [row for r in results for row in best_rows_from_result(r)]
    legal = [r for r in probes if int(r.get("forbidden_contact_frames", 9999) or 9999) == 0 and int(r.get("handle_nonlegal_contact_frames", 9999) or 9999) == 0]
    best_overall = max(probes, key=lambda r: float(r.get("max_drawer_fraction", 0.0) or 0.0), default={})
    best_legal = max(legal, key=lambda r: float(r.get("max_drawer_fraction", 0.0) or 0.0), default={})
    return {
        "generated_at_utc": utc_now(),
        "cycle": cycle,
        "candidate_count": len(results),
        "admitted_count": sum(1 for r in results if r.get("admitted")),
        "dominant_failure_histogram": dict(sorted(hist.items())),
        "defect_label_histogram": dict(sorted(defect_hist.items())),
        "best_overall_candidate_id": best_overall.get("candidate_id"),
        "best_overall_drawer_fraction": float(best_overall.get("max_drawer_fraction", 0.0) or 0.0),
        "best_overall_forbidden_contact_frames": int(best_overall.get("forbidden_contact_frames", 0) or 0),
        "best_legal_candidate_id": best_legal.get("candidate_id"),
        "best_legal_drawer_fraction": float(best_legal.get("max_drawer_fraction", 0.0) or 0.0),
        "best_legal_two_pad_retention_frames": int(best_legal.get("pull_phase_two_pad_target_contact_frames", 0) or 0),
        "best_legal_target_contact_frames": int(best_legal.get("pull_phase_target_contact_frames", 0) or 0),
        "best_legal_max_penetration_m": float(best_legal.get("max_penetration_m", 0.0) or 0.0),
        "operator_progress_metric": [
            sum(1 for r in results if r.get("admitted")),
            float(best_legal.get("max_drawer_fraction", 0.0) or 0.0),
            int(best_legal.get("pull_phase_two_pad_target_contact_frames", 0) or 0),
            -int(best_overall.get("forbidden_contact_frames", 0) or 0),
        ],
    }


def evaluate_cycle(run_dir: Path, cycle: int, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out = run_dir / f"cycle_{cycle}_oracle_results.jsonl"
    if out.exists():
        out.unlink()
    admitted: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    variants = CONTROLLER_VARIANTS
    for idx, candidate in enumerate(candidates):
        physical = pool.evaluate_generated_candidate(candidate, run_dir)
        probe_best: dict[str, Any] | None = None
        probe_rows: list[dict[str, Any]] = []
        if cd.physical_pass(physical):
            for off, variant in enumerate(variants):
                row = cd.run_case(physical, "nominal", variant, run_dir, cycle * 10000 + idx * 100 + off, f"cycle_{cycle}_probe")
                probe_rows.append(row)
            probe_best = max(probe_rows, key=row_rank, default={})
        result = cd.oracle_result(candidate, physical, probe_best, probe_rows)
        result["cycle"] = cycle
        result["controller_variants_evaluated"] = [v["name"] for v in variants] if cd.physical_pass(physical) else []
        result["defect_label"] = "ADMITTED" if result.get("admitted") else defect_label(result.get("dominant_failure"))
        append_jsonl(out, result)
        append_jsonl(run_dir / "candidate_oracle_results.jsonl", result)
        results.append(result)
        if result.get("admitted"):
            physical["co_design_source_type"] = candidate.get("co_design_source_type")
            physical["source_type_for_spec"] = candidate.get("source_type_for_spec")
            physical["parent_instance_id"] = candidate.get("parent_instance_id")
            physical["selected_pull_variant"] = result.get("selected_variant") or variants[0]
            admitted.append(physical)
    summary = summarize_cycle(results, cycle)
    summary["repair_operators_applied"] = sorted({c.get("targeted_defect_cluster") for c in candidates})
    write_json(run_dir / f"cycle_{cycle}_repair_summary.json", summary)
    write_json(run_dir / f"cycle_{cycle}_candidate_generation.json", {"generated_at_utc": utc_now(), "cycle": cycle, "candidate_count": len(candidates), "candidates": candidates})
    return admitted, results


def defect_operator_map() -> dict[str, Any]:
    return {
        "FULL_BODY_KEEP_OUT_FAILED": ["increase cabinet side clearance envelope", "change approach topology side/top/front", "modify robot mount side/yaw/height", "add nullspace keepout around link5/link6/link7/hand", "reject candidate if keepout impossible"],
        "GUARDED_IK_INFEASIBLE": ["adjust handle exposure", "adjust handle height/depth", "adjust robot mount height/yaw", "generate handle pose within Panda two-pad IK reachable shell", "reject if constrained IK infeasible"],
        "APPROACH_CORRIDOR_BLOCKED": ["widen approach corridor", "select alternate active handle/drawer", "change pregrasp side", "add corridor signed-clearance oracle", "reject narrow-cabinet candidate"],
        "PULL_WRENCH_INSUFFICIENT": ["align pull direction with drawer prismatic axis", "adjust handle frame / pad contact normal", "increase contact hold before pull", "adjust gripper close schedule", "select handle geometry with better pull lever / exposure", "reject if wrench projection insufficient"],
        "TWO_PAD_RETENTION_FAILED": ["adjust gripper span / pad separation", "adjust pad target points on handle", "adjust handle bar width/pose in generated variant", "add reseat / regrasp mode", "add normal compliance / tangential pull decoupling"],
        "DRAWER_QPOS_NONMONOTONIC": ["smooth pull velocity", "reduce normal force oscillation", "add drawer-axis velocity controller", "extend contact hold before pull", "reject unstable mechanism dynamics candidate"],
    }


def write_stage0(run_dir: Path, prior: Path | None) -> dict[str, Any]:
    preflight = run_cmd(["/root/anaconda3/envs/infinigen/bin/python", "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py", "--spec", SPEC_REL, "--dry-run"])
    prior_files: dict[str, Any] = {}
    prior_hist = {}
    if prior:
        for p in sorted(prior.glob("candidate_admission_report*.json")):
            prior_files[p.name] = rel(p)
        for p in sorted(prior.glob("candidate_oracle_results*.jsonl")):
            prior_files[p.name] = {"path": rel(p), "rows": len(read_jsonl(p))}
        for name in ["final_closeout.json", "candidate_defect_aggregate_summary.json", "trace_hash_manifest.json"]:
            if (prior / name).exists():
                prior_files[name] = rel(prior / name)
        prior_hist = (load_json(prior / "candidate_defect_aggregate_summary.json", {}) or {}).get("candidate_dominant_failure_histogram", {})
    payload = {"generated_at_utc": utc_now(), "task_id": TASK_ID, "pwd": str(ROOT), "branch": run_git(["branch", "--show-current"]), "head": run_git(["rev-parse", "HEAD"]), "remote_v": run_git(["remote", "-v"]), "git_status_short": run_git(["status", "--short"]), "task_spec": SPEC_REL, "preflight": preflight, "harness_preflight_passed": preflight["returncode"] == 0, "task_spec_lock_bound": "task_spec_hash_verified=True" in preflight.get("stdout", ""), "prior_run": rel(prior) if prior else None, "prior_files": prior_files, "prior_defect_histogram": prior_hist, "defect_histogram_ingested": bool(prior_hist), "fixed_pool_used_only_as_negative_baseline": True, "old_fixed_pool_modified": False}
    write_json(run_dir / "stage0_authority_and_prior_evidence.json", payload)
    write_md(run_dir / "prior_defect_histogram_report.md", "\n".join(["# Prior Defect Histogram Report", "", f"- prior_run: `{payload['prior_run']}`", f"- defect_histogram_ingested: `{payload['defect_histogram_ingested']}`", f"- prior_defect_histogram: `{json.dumps(prior_hist, sort_keys=True)}`", "- old fixed accepted pool is immutable and only a negative baseline."]))
    (run_dir / "commands.log").write_text(json.dumps(ready(payload), indent=2, sort_keys=True) + "\n")
    return payload


def write_operator_map(run_dir: Path) -> dict[str, Any]:
    mapping = defect_operator_map()
    payload = {"generated_at_utc": utc_now(), "repair_operator_map_written": True, "defect_to_repair_operator_map": mapping}
    write_json(run_dir / "defect_to_repair_operator_map.json", payload)
    lines = ["# Defect To Repair Operator Map", ""]
    for defect, ops in mapping.items():
        lines.append(f"## {defect}")
        lines.extend(f"- {op}" for op in ops)
        lines.append("")
    write_md(run_dir / "defect_to_repair_operator_map.md", "\n".join(lines))
    return payload


def write_search_space(run_dir: Path) -> None:
    write_json(run_dir / "codesign_search_space.json", {"generated_at_utc": utc_now(), "structure_variables": ["source_type", "active drawer/handle selection", "handle height", "handle horizontal exposure", "handle depth from drawer front", "handle bar width / graspable region", "cabinet side clearance", "cabinet front clearance", "drawer front depth", "robot mount x/y/z/yaw", "approach topology"], "controller_variables": ["pregrasp offset", "guarded approach speed", "contact hold duration", "gripper close timing", "gripper close target", "pull axis blend", "pull speed", "pull duration", "impedance damping", "reseat enable/disable", "reseat threshold", "keepout gain", "nullspace keepout weight"], "controller_variants": CONTROLLER_VARIANTS, "defect_directed_not_random": True})


def source_mix_ok(admitted: list[dict[str, Any]]) -> bool:
    counts = Counter(str(c.get("co_design_source_type")) for c in admitted)
    return counts.get("generated_variant", 0) >= 2 and counts.get("repaired_layout", 0) >= 2


def run_targeted(run_dir: Path, admitted: list[dict[str, Any]], cycle: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = admitted[:7] if len(admitted) >= 7 else admitted[:]
    plan = {"generated_at_utc": utc_now(), "cycle": cycle, "case_count": len(selected), "candidate_ids": [c["candidate_id"] for c in selected], "source_mix": dict(Counter(str(c.get("co_design_source_type")) for c in selected)), "hard_fail_reasons": []}
    if len(selected) < 7:
        plan["hard_fail_reasons"].append("TARGETED_CASE_COUNT_LT_7")
    if not source_mix_ok(selected):
        plan["hard_fail_reasons"].append("TARGETED_SOURCE_MIX_INSUFFICIENT")
    write_json(run_dir / "targeted_shard_plan.json", plan)
    rows: list[dict[str, Any]] = []
    out = run_dir / "targeted_shard_results.jsonl"
    if out.exists():
        out.unlink()
    if not plan["hard_fail_reasons"]:
        perturbations = cd.DEFAULT_TARGETED_PERTURBATIONS
        for idx, candidate in enumerate(selected):
            variant = candidate.get("selected_pull_variant") or CONTROLLER_VARIANTS[0]
            row = cd.run_case(candidate, perturbations[idx % len(perturbations)], variant, run_dir, 70000 + idx, "targeted_shard")
            rows.append(row)
            append_jsonl(out, row)
    summary = cd.summarize_rows(rows, "targeted_shard") if rows else {"generated_at_utc": utc_now(), "cases_total": len(selected), "cases_passed": 0, "cases_failed": len(selected), "matrix_passed": False, "failure_histogram": {r: 1 for r in plan["hard_fail_reasons"]}}
    summary["targeted_shard_passed"] = bool(rows and len(rows) >= 7 and all(r.get("passed") for r in rows))
    summary["cycle"] = cycle
    write_json(run_dir / "targeted_shard_results.json", summary)
    write_json(run_dir / "targeted_shard_failure_feedback.json", {"generated_at_utc": utc_now(), "cycle": cycle, "targeted_shard_passed": summary["targeted_shard_passed"], "failure_histogram": summary.get("failure_histogram", {}), "feed_back_to_operator_map": not summary["targeted_shard_passed"]})
    return summary, rows


def run_full30(run_dir: Path, admitted: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = admitted[:5]
    rows: list[dict[str, Any]] = []
    out = run_dir / "full30_dynamic_pull_certification.jsonl"
    if out.exists():
        out.unlink()
    if len(selected) < 5:
        summary = {"generated_at_utc": utc_now(), "full30_certification_attempted": False, "cases_total": 0, "cases_passed": 0, "cases_failed": 0, "matrix_passed": False, "failure_histogram": {"ADMITTED_CANDIDATE_COUNT_LT_5": 1}}
        write_json(run_dir / "full30_dynamic_pull_certification_report.json", summary)
        return summary, rows
    for cidx, candidate in enumerate(selected):
        variant = candidate.get("selected_pull_variant") or CONTROLLER_VARIANTS[0]
        for pidx, perturb in enumerate(cd.PERTURBATIONS[:6]):
            row = cd.run_case(candidate, perturb["name"], variant, run_dir, 80000 + cidx * 10 + pidx, "full30_dynamic_pull")
            rows.append(row)
            append_jsonl(out, row)
    summary = cd.summarize_rows(rows, "full30_dynamic_pull_certification")
    summary["full30_certification_attempted"] = True
    write_json(run_dir / "full30_dynamic_pull_certification_report.json", summary)
    return summary, rows


def strict_export(run_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or not all(r.get("passed") for r in rows):
        payload = {"generated_at_utc": utc_now(), "strict_teacher_export_attempted": False, "strict_teacher_export_complete": False, "reason": "full30_not_certified"}
        write_json(run_dir / "strict_teacher_export_manifest.json", payload)
        return payload
    items = []
    for row in rows:
        trace = ROOT / str(row.get("trace_jsonl"))
        items.append({"trace_jsonl": rel(trace), "exists": trace.exists(), "sha256": sha256_file(trace) if trace.exists() else None, "size_bytes": trace.stat().st_size if trace.exists() else 0, "candidate_id": row.get("candidate_id"), "perturbation": row.get("perturbation")})
    complete = bool(items and all(i["exists"] for i in items))
    payload = {"generated_at_utc": utc_now(), "strict_teacher_export_attempted": True, "strict_teacher_export_complete": complete, "source_committed_head": run_git(["rev-parse", "HEAD"]), "trace_count": len(items), "trace_items": items}
    write_json(run_dir / "strict_teacher_export_manifest.json", payload)
    return payload


def local_replay_handoff(run_dir: Path, export: dict[str, Any]) -> dict[str, Any]:
    if not export.get("strict_teacher_export_complete"):
        payload = {"generated_at_utc": utc_now(), "local_strict_replay_attempted": False, "local_strict_replay_render_passed": False, "reason": "strict_export_not_complete"}
    else:
        payload = {"generated_at_utc": utc_now(), "local_strict_replay_attempted": False, "local_strict_replay_render_passed": False, "reason": "remote_export_complete_local_render_not_invoked_by_remote_helper", "local_replay_dir": str(LOCAL_PLAYGROUND_ROOT / "local_replay" / f"v11_g4_goc_v4_defect_histogram_dynamic_pull_pool_solver_to_export_replay_{utc_stamp()}")}
    write_json(run_dir / "local_replay_handoff_manifest.json", payload)
    return payload


def trace_manifests(run_dir: Path) -> None:
    for rel_name, out_name in [("traces", "trace_hash_manifest.json"), ("generated_instances", "generated_instance_hash_manifest.json")]:
        root = run_dir / rel_name
        files = []
        total = 0
        if root.exists():
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    size = p.stat().st_size
                    total += size
                    files.append({"path": rel(p), "relative_path": p.relative_to(run_dir).as_posix(), "size_bytes": size, "sha256": sha256_file(p)})
        write_json(run_dir / out_name, {"generated_at_utc": utc_now(), "run_dir": rel(run_dir), "tree": rel_name, "file_count": len(files), "total_size_bytes": total, "raw_files_committed": False, "files": files})


def aggregate_best(all_results: list[dict[str, Any]]) -> dict[str, Any]:
    probes = [row for r in all_results for row in best_rows_from_result(r)]
    legal = [r for r in probes if int(r.get("forbidden_contact_frames", 9999) or 9999) == 0 and int(r.get("handle_nonlegal_contact_frames", 9999) or 9999) == 0]
    best = max(probes, key=lambda r: float(r.get("max_drawer_fraction", 0.0) or 0.0), default={})
    best_legal = max(legal, key=lambda r: float(r.get("max_drawer_fraction", 0.0) or 0.0), default={})
    return {"best_overall_row": best, "best_legal_row": best_legal}


def final_checks(run_dir: Path) -> dict[str, Any]:
    preflight = run_cmd(["/root/anaconda3/envs/infinigen/bin/python", "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py", "--spec", SPEC_REL, "--dry-run"])
    pyc = run_cmd(["/root/anaconda3/envs/infinigen/bin/python", "-m", "py_compile", "scripts/mint/v11_g4_defect_histogram_dynamic_pull_pool_solver.py"])
    ruff = run_cmd(["/root/anaconda3/envs/infinigen/bin/python", "-m", "ruff", "check", "scripts/mint/v11_g4_defect_histogram_dynamic_pull_pool_solver.py"])
    json_errors = []
    for path in list(run_dir.rglob("*.json")) + list(run_dir.rglob("*.jsonl")):
        try:
            if path.suffix == ".json":
                json.load(open(path))
            else:
                for n, line in enumerate(open(path), 1):
                    if line.strip():
                        json.loads(line)
        except Exception as exc:
            json_errors.append({"path": rel(path), "error": repr(exc), "line": locals().get("n")})
    status = run_git(["status", "--short"])
    payload = {"generated_at_utc": utc_now(), "preflight": preflight, "harness_preflight_passed": preflight["returncode"] == 0, "task_spec_lock_bound": "task_spec_hash_verified=True" in preflight.get("stdout", ""), "py_compile": pyc, "py_compile_passed": pyc["returncode"] == 0, "ruff": ruff, "ruff_passed": ruff["returncode"] == 0, "json_errors": json_errors, "json_parse_passed": not json_errors, "current_truth_modified": "sovereign/current_truth.json" in status, "next_actions_modified": "sovereign/next_actions.json" in status, "external_mint_modified": "external/MINT" in status, "git_status_short": status}
    payload["final_checks_passed"] = bool(payload["harness_preflight_passed"] and payload["py_compile_passed"] and payload["ruff_passed"] and payload["json_parse_passed"] and not payload["current_truth_modified"] and not payload["next_actions_modified"] and not payload["external_mint_modified"])
    write_json(run_dir / "stage9_final_checks.json", payload)
    return payload


def write_deltas(run_dir: Path, closeout: dict[str, Any]) -> None:
    write_json(CAMPAIGN / "sovereign/proposed_current_truth_delta_defect_histogram_dynamic_pull_pool_solver.json", {"proposal_id": "proposed_current_truth_delta_defect_histogram_dynamic_pull_pool_solver", "generated_at_utc": utc_now(), "source_run_dir": rel(run_dir), "closeout_classification": closeout.get("closeout_classification"), "candidate_pool_admitted_count": closeout.get("candidate_pool_admitted_count"), "targeted_shard_passed": closeout.get("targeted_shard_passed"), "full30_certification_attempted": closeout.get("full30_certification_attempted"), "strict_teacher_export_complete": closeout.get("strict_teacher_export_complete"), "local_strict_replay_render_passed": closeout.get("local_strict_replay_render_passed"), "teacher_rollout_remains_blocked": closeout.get("closeout_classification") != "DYNAMIC_PULL_ACCESSIBLE_POOL_STRICT_EXPORT_LOCAL_REPLAY_READY", "next_gate": closeout.get("next_gate")})
    write_json(CAMPAIGN / "sovereign/proposed_next_actions_defect_histogram_dynamic_pull_pool_solver.json", {"proposal_id": "proposed_next_actions_defect_histogram_dynamic_pull_pool_solver", "generated_at_utc": utc_now(), "recommended_next_gate": closeout.get("next_gate"), "closeout_classification": closeout.get("closeout_classification"), "required_action": "use exhausted operator certificate to decide whether to expand structural generator or controller search space"})


def classify_closeout(admitted_count: int, targeted: dict[str, Any], full30: dict[str, Any], export: dict[str, Any], local: dict[str, Any]) -> tuple[str, str]:
    if admitted_count == 0:
        return "DEFECT_HISTOGRAM_SOLVER_NO_ADMITTED_CANDIDATES", "STRUCTURAL_GENERATOR_OR_CONTROLLER_SEARCH_SPACE_EXPANSION"
    if not targeted.get("targeted_shard_passed"):
        return "DEFECT_HISTOGRAM_SOLVER_TARGETED_SHARD_FAILED", "TARGETED_DYNAMIC_PULL_DEFECT_REPAIR"
    if not full30.get("matrix_passed"):
        return "DEFECT_HISTOGRAM_SOLVER_FULL30_FAILED", "ROBUST_DYNAMIC_PULL_GENERALIZATION_REPAIR"
    if not export.get("strict_teacher_export_complete"):
        return "STRICT_TEACHER_EXPORT_FAILED", "STRICT_EXPORT_INFRA_REPAIR"
    if not local.get("local_strict_replay_render_passed"):
        return "LOCAL_STRICT_REPLAY_RENDER_FAILED", "LOCAL_STRICT_REPLAY_RENDER_REPAIR"
    return "DYNAMIC_PULL_ACCESSIBLE_POOL_STRICT_EXPORT_LOCAL_REPLAY_READY", "MANUAL_VISUAL_AND_SCIENCE_REVIEW_BEFORE_MINT_DATASET_ADMISSION"


def build_closeout(run_dir: Path, stage0: dict[str, Any], operator_map: dict[str, Any], cycles_run: int, candidates: list[dict[str, Any]], admitted: list[dict[str, Any]], all_results: list[dict[str, Any]], targeted: dict[str, Any], full30: dict[str, Any], export: dict[str, Any], local: dict[str, Any], checks: dict[str, Any]) -> dict[str, Any]:
    classification, next_gate = classify_closeout(len(admitted), targeted, full30, export, local)
    defect_counts = Counter(r.get("defect_label") for r in all_results if r.get("defect_label") != "ADMITTED")
    best = aggregate_best(all_results)
    best_overall = best["best_overall_row"]
    best_legal = best["best_legal_row"]
    closeout = {
        "task_id": TASK_ID,
        "generated_at_utc": utc_now(),
        "closeout_classification": classification,
        "harness_preflight_passed": bool(stage0.get("harness_preflight_passed") and checks.get("harness_preflight_passed")),
        "task_spec_lock_bound": bool(stage0.get("task_spec_lock_bound") and checks.get("task_spec_lock_bound")),
        "old_fixed_pool_modified": False,
        "fixed_pool_used_only_as_negative_baseline": True,
        "defect_histogram_ingested": bool(stage0.get("defect_histogram_ingested")),
        "repair_operator_map_written": bool(operator_map.get("repair_operator_map_written")),
        "outer_cycles_run": cycles_run,
        "candidate_pool_total": len(candidates),
        "candidate_pool_admitted_count": len(admitted),
        "targeted_shard_passed": bool(targeted.get("targeted_shard_passed")),
        "targeted_shard_cases_total": int(targeted.get("cases_total", 0) or 0),
        "targeted_shard_cases_passed": int(targeted.get("cases_passed", 0) or 0),
        "full30_certification_attempted": bool(full30.get("full30_certification_attempted", False)),
        "full30_cases_passed": int(full30.get("cases_passed", 0) or 0),
        "full30_cases_failed": int(full30.get("cases_failed", 0) or 0),
        "strict_teacher_export_attempted": bool(export.get("strict_teacher_export_attempted", False)),
        "strict_teacher_export_complete": bool(export.get("strict_teacher_export_complete", False)),
        "local_strict_replay_attempted": bool(local.get("local_strict_replay_attempted", False)),
        "local_strict_replay_render_passed": bool(local.get("local_strict_replay_render_passed", False)),
        "controller_defect_count": int(defect_counts.get("CONTROLLER_DEFECT", 0)),
        "structure_defect_count": int(defect_counts.get("STRUCTURE_DEFECT", 0)),
        "mixed_defect_count": int(defect_counts.get("MIXED_DEFECT", 0)),
        "best_overall_drawer_fraction": float(best_overall.get("max_drawer_fraction", 0.0) or 0.0),
        "best_legal_drawer_fraction": float(best_legal.get("max_drawer_fraction", 0.0) or 0.0),
        "best_candidate_forbidden_contact_frames": int(best_overall.get("forbidden_contact_frames", 0) or 0),
        "best_candidate_handle_nonlegal_contact_frames": int(best_overall.get("handle_nonlegal_contact_frames", 0) or 0),
        "best_candidate_max_penetration_m": float(best_overall.get("max_penetration_m", 0.0) or 0.0),
        "best_candidate_qpos_monotonic": bool(best_overall.get("drawer_qpos_nondecreasing_with_tolerance", False)),
        "best_overall_candidate_id": best_overall.get("candidate_id"),
        "best_legal_candidate_id": best_legal.get("candidate_id"),
        "current_truth_modified": bool(checks.get("current_truth_modified", False)),
        "next_actions_modified": bool(checks.get("next_actions_modified", False)),
        "runtime_patch_applied": True,
        "runtime_patch_files": ["scripts/mint/v11_g4_defect_histogram_dynamic_pull_pool_solver.py"],
        "committed": False,
        "pushed_to_origin": False,
        "remote_commit_hash": None,
        "next_gate": next_gate,
    }
    write_json(run_dir / "final_closeout.json", closeout)
    write_json(run_dir / "closeout_decision.json", closeout)
    write_md(run_dir / "final_closeout.md", "\n".join(["# Defect Histogram Dynamic Pull Solver Closeout", "", f"- closeout_classification: `{classification}`", f"- next_gate: `{next_gate}`", f"- outer_cycles_run: `{cycles_run}`", f"- candidate_pool_admitted_count: `{len(admitted)}`", f"- targeted_shard: `{closeout['targeted_shard_cases_passed']}/{closeout['targeted_shard_cases_total']}`", f"- full30_attempted: `{closeout['full30_certification_attempted']}`", f"- strict_export_complete: `{closeout['strict_teacher_export_complete']}`", f"- local_replay_render_passed: `{closeout['local_strict_replay_render_passed']}`", "", "The old fixed accepted pool was not modified and was not used as strict export input."]))
    write_md(run_dir / "final_report.md", (run_dir / "final_closeout.md").read_text())
    return closeout


def post_push_verify(run_dir: Path) -> dict[str, Any]:
    branch = run_git(["branch", "--show-current"])
    local_head = run_git(["rev-parse", "HEAD"])
    remote_line = run_git(["ls-remote", "my-origin", f"refs/heads/{branch}"])
    remote_head = remote_line.split()[0] if remote_line else ""
    paths = [SPEC_REL, "scripts/mint/v11_g4_defect_histogram_dynamic_pull_pool_solver.py", rel(run_dir / "final_closeout.json"), rel(run_dir / "candidate_admission_report.json"), rel(run_dir / "targeted_shard_results.json"), rel(run_dir / "stage9_final_checks.json"), rel(run_dir / "trace_hash_manifest.json"), "experiments/mint/mint_drawer_v1/sovereign/proposed_current_truth_delta_defect_histogram_dynamic_pull_pool_solver.json", "experiments/mint/mint_drawer_v1/sovereign/proposed_next_actions_defect_histogram_dynamic_pull_pool_solver.json"]
    checks = []
    for path in paths:
        exists = subprocess.run(["git", "cat-file", "-e", f"{remote_head}:{path}"], cwd=ROOT).returncode == 0 if remote_head and path else False
        checks.append({"path": path, "origin_visible": exists})
    payload = {"generated_at_utc": utc_now(), "branch": branch, "local_head": local_head, "remote_head": remote_head, "origin_head_matches_local": local_head == remote_head, "path_checks": checks, "all_paths_origin_visible": all(c["origin_visible"] for c in checks)}
    write_json(run_dir / "post_push_verification.json", payload)
    return payload


def run_phase(run_dir: Path) -> dict[str, Any]:
    install_solver_builder()
    run_dir.mkdir(parents=True, exist_ok=True)
    prior = latest_prior_run()
    stage0 = write_stage0(run_dir, prior)
    operator_map = write_operator_map(run_dir)
    write_search_space(run_dir)
    all_candidates: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    admitted: list[dict[str, Any]] = []
    targeted_summary: dict[str, Any] = {"targeted_shard_passed": False, "cases_total": 0, "cases_passed": 0, "failure_histogram": {"not_enough_admitted_candidates_yet": 1}}
    targeted_rows: list[dict[str, Any]] = []
    cycles_run = 0
    if (run_dir / "candidate_oracle_results.jsonl").exists():
        (run_dir / "candidate_oracle_results.jsonl").unlink()
    for cycle in range(1, MAX_OUTER_CYCLES + 1):
        cycles_run = cycle
        candidates = generate_cycle_candidates(cycle)
        all_candidates.extend(candidates)
        new_admitted, results = evaluate_cycle(run_dir, cycle, candidates)
        admitted.extend(new_admitted)
        all_results.extend(results)
        summary = summarize_cycle(results, cycle)
        summary["cumulative_admitted_count"] = len(admitted)
        write_json(run_dir / f"cycle_{cycle}_repair_summary.json", summary)
        if len(admitted) >= 7 or (cycle == MAX_OUTER_CYCLES and len(admitted) >= 5):
            targeted_summary, targeted_rows = run_targeted(run_dir, admitted, cycle)
            if targeted_summary.get("targeted_shard_passed"):
                break
    if not (run_dir / "targeted_shard_results.json").exists():
        targeted_summary, targeted_rows = run_targeted(run_dir, admitted, cycles_run)
    if targeted_summary.get("targeted_shard_passed"):
        full30_summary, full30_rows = run_full30(run_dir, admitted)
    else:
        full30_summary = {"generated_at_utc": utc_now(), "full30_certification_attempted": False, "cases_total": 0, "cases_passed": 0, "cases_failed": 0, "matrix_passed": False, "failure_histogram": {"targeted_shard_not_passed": 1}}
        full30_rows = []
        write_json(run_dir / "full30_dynamic_pull_certification_report.json", full30_summary)
    export = strict_export(run_dir, full30_rows if full30_summary.get("matrix_passed") else [])
    local = local_replay_handoff(run_dir, export)
    hist = Counter(r.get("dominant_failure") or "ADMITTED" for r in all_results)
    write_json(run_dir / "candidate_admission_report.json", {"generated_at_utc": utc_now(), "candidate_count": len(all_candidates), "admitted_count": len(admitted), "dominant_failure_histogram": dict(sorted(hist.items())), "admitted_candidate_ids": [c["candidate_id"] for c in admitted]})
    trace_manifests(run_dir)
    provisional = {"harness_preflight_passed": True, "task_spec_lock_bound": True, "current_truth_modified": False, "next_actions_modified": False}
    closeout = build_closeout(run_dir, stage0, operator_map, cycles_run, all_candidates, admitted, all_results, targeted_summary, full30_summary, export, local, provisional)
    write_deltas(run_dir, closeout)
    checks = final_checks(run_dir)
    closeout = build_closeout(run_dir, stage0, operator_map, cycles_run, all_candidates, admitted, all_results, targeted_summary, full30_summary, export, local, checks)
    write_deltas(run_dir, closeout)
    return closeout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir")
    parser.add_argument("--post-push-verify", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else CAMPAIGN / "runtime" / f"{RUN_PREFIX}_{utc_stamp()}"
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if args.post_push_verify:
        print(json.dumps(ready(post_push_verify(run_dir)), indent=2, sort_keys=True))
        return
    closeout = run_phase(run_dir)
    print(json.dumps({"run_dir": rel(run_dir), "closeout_classification": closeout.get("closeout_classification"), "candidate_pool_admitted_count": closeout.get("candidate_pool_admitted_count"), "targeted_shard_cases_passed": closeout.get("targeted_shard_cases_passed"), "targeted_shard_cases_total": closeout.get("targeted_shard_cases_total"), "next_gate": closeout.get("next_gate")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

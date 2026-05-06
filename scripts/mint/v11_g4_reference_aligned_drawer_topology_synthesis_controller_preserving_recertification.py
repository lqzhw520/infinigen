#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, xml.etree.ElementTree as ET
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN = ROOT / "experiments/mint/mint_drawer_v1"
TASK_ID = "V11_G4_REFERENCE_ALIGNED_DRAWER_TOPOLOGY_SYNTHESIS_CONTROLLER_PRESERVING_RECERTIFICATION_V1"
SPEC_REL = "experiments/mint/mint_drawer_v1/sovereign/experiment_specs/v11_g4_reference_aligned_drawer_topology_synthesis_controller_preserving_recertification.yaml"
RUN_PREFIX = "v11_g4_reference_aligned_drawer_topology_synthesis_controller_preserving_recertification"
PRIOR_PREFIX = "v11_g4_goc_v4_exact_latch_pull_keepout_v3_continuation_targeted_progress_repair_"
STRICT_DRAWER_FRACTION = 0.80

sys.path.insert(0, str(ROOT / "scripts/mint"))
import v11_g4_visual_topology_realism_and_action_replay_repair_v2 as v2  # noqa: E402
import v11_g4_physical_accessibility_instance_pool as pool  # noqa: E402
import v11_g4_round_knob_exact_contact_patch_latch_pull_synthesis as patch  # noqa: E402
import v11_g4_exact_latch_pull_wrench_keepout_trajectory_optimization as v3  # noqa: E402

ORIGINAL_BUILDER = v2.ORIGINAL_BUILDER


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ready(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): ready(val) for k, val in v.items() if not str(k).startswith("_")}
    if isinstance(v, (list, tuple, set)):
        return [ready(x) for x in v]
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return str(v)
    return v


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ready(payload), indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(ready(payload), sort_keys=True) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class ReferenceAlignedSolidFrontDrawerBuilder(ORIGINAL_BUILDER):
    """Solid drawer front + short-stub spherical knob + complete moving tray."""

    def _drawer_xml(self) -> str:
        p = self.variant
        hx = float(p.get("handle_x", -0.18)); hy = float(p.get("handle_y", -0.067)); hz = float(p.get("handle_z", 0.402))
        radius = float(p.get("handle_radius", 0.024)); diameter = 2.0 * radius
        stub_len = clamp(float(p.get("stub_length", 0.18 * diameter)), 0.10 * diameter, 0.25 * diameter)
        stub_radius = float(p.get("stub_radius", min(radius * 0.34, 0.008)))
        front_half_thickness = float(p.get("front_half_thickness", 0.014))
        front_face_x = hx + radius + stub_len
        door_x = front_face_x + front_half_thickness
        cabinet_y = hy; cabinet_z = float(p.get("cabinet_z", max(0.34, hz - 0.050)))
        door_half_width = float(p.get("door_half_width", 0.205)); door_half_height = float(p.get("door_half_height", 0.235))
        tray_depth = float(p.get("tray_depth", 0.185)); tray_half_width = float(p.get("tray_half_width", 0.150))
        tray_wall_height = float(p.get("tray_wall_height", 0.115)); tray_wall = float(p.get("tray_wall_thickness", 0.010))
        cabinet_depth = float(p.get("cabinet_depth", max(0.18, tray_depth)))
        cabinet_half_width = float(p.get("cabinet_half_width", max(0.245, tray_half_width + 0.095)))
        cabinet_half_height = float(p.get("cabinet_half_height", 0.285)); cabinet_x = float(p.get("cabinet_x", 0.175))
        damping = float(p.get("drawer_damping", 0.9)); axis = " ".join(str(float(x)) for x in p.get("pull_axis", [-1, 0, 0]))
        tray_center_x = door_x + front_half_thickness + tray_depth / 2.0
        tray_back_x = door_x + front_half_thickness + tray_depth
        tray_bottom_z = cabinet_z - tray_wall_height; tray_side_z = cabinet_z - tray_wall_height / 2.0
        support_z = tray_bottom_z + 0.030
        runner_y_left = cabinet_y - tray_half_width - 0.018; runner_y_right = cabinet_y + tray_half_width + 0.018
        support_half_x = max(0.14, tray_depth * 0.82)
        stem_start_x = hx + radius * 0.94; stem_end_x = front_face_x + 0.0015
        plate_x = front_face_x + 0.0025; plate_y = max(radius * 0.75, 0.018); plate_z = max(radius * 0.75, 0.018)
        return f'''
<body name="drawer_base" pos="0 0 0">
  <geom name="cabinet_back_collision" type="box" pos="{cabinet_x + cabinet_depth * 0.72:.4f} {cabinet_y:.4f} {cabinet_z:.4f}" size="0.025 {cabinet_half_width:.4f} {cabinet_half_height:.4f}" contype="1" conaffinity="1" rgba="0.86 0.83 0.78 1"/>
  <geom name="cabinet_bottom_collision" type="box" pos="{cabinet_x:.4f} {cabinet_y:.4f} {cabinet_z - cabinet_half_height:.4f}" size="{cabinet_depth:.4f} {cabinet_half_width:.4f} 0.025" contype="1" conaffinity="1" rgba="0.86 0.83 0.78 1"/>
  <geom name="cabinet_top_collision" type="box" pos="{cabinet_x:.4f} {cabinet_y:.4f} {cabinet_z + cabinet_half_height:.4f}" size="{cabinet_depth:.4f} {cabinet_half_width:.4f} 0.025" contype="1" conaffinity="1" rgba="0.86 0.83 0.78 1"/>
  <geom name="cabinet_left_collision" type="box" pos="{cabinet_x:.4f} {cabinet_y - cabinet_half_width:.4f} {cabinet_z:.4f}" size="{cabinet_depth:.4f} 0.025 {cabinet_half_height:.4f}" contype="1" conaffinity="1" rgba="0.86 0.83 0.78 1"/>
  <geom name="cabinet_right_collision" type="box" pos="{cabinet_x:.4f} {cabinet_y + cabinet_half_width:.4f} {cabinet_z:.4f}" size="{cabinet_depth:.4f} 0.025 {cabinet_half_height:.4f}" contype="1" conaffinity="1" rgba="0.86 0.83 0.78 1"/>
  <geom name="cabinet_left_runner_collision" type="box" pos="{cabinet_x:.4f} {runner_y_left:.4f} {support_z:.4f}" size="{support_half_x:.4f} 0.009 0.012" contype="1" conaffinity="1" rgba="0.34 0.34 0.35 1"/>
  <geom name="cabinet_right_runner_collision" type="box" pos="{cabinet_x:.4f} {runner_y_right:.4f} {support_z:.4f}" size="{support_half_x:.4f} 0.009 0.012" contype="1" conaffinity="1" rgba="0.34 0.34 0.35 1"/>
  <body name="link_1" pos="0 0 0">
    <joint name="drawer_slider_0" type="slide" axis="{axis}" range="0 0.35" damping="{damping:.4f}"/>
    <geom name="drawer_front_panel_collision" type="box" pos="{door_x:.4f} {cabinet_y:.4f} {cabinet_z:.4f}" size="{front_half_thickness:.4f} {door_half_width:.4f} {door_half_height:.4f}" contype="1" conaffinity="1" rgba="0.91 0.88 0.82 1"/>
    <geom name="drawer_front_knob_boss_plate_collision" type="box" pos="{plate_x:.4f} {hy:.4f} {hz:.4f}" size="0.0035 {plate_y:.4f} {plate_z:.4f}" contype="1" conaffinity="1" rgba="0.80 0.76 0.69 1"/>
    <geom name="drawer_tray_bottom_collision" type="box" pos="{tray_center_x:.4f} {cabinet_y:.4f} {tray_bottom_z:.4f}" size="{tray_depth / 2.0:.4f} {tray_half_width:.4f} {tray_wall:.4f}" contype="1" conaffinity="1" rgba="0.78 0.73 0.66 1"/>
    <geom name="drawer_tray_left_side_collision" type="box" pos="{tray_center_x:.4f} {cabinet_y - tray_half_width:.4f} {tray_side_z:.4f}" size="{tray_depth / 2.0:.4f} {tray_wall:.4f} {tray_wall_height:.4f}" contype="1" conaffinity="1" rgba="0.78 0.73 0.66 1"/>
    <geom name="drawer_tray_right_side_collision" type="box" pos="{tray_center_x:.4f} {cabinet_y + tray_half_width:.4f} {tray_side_z:.4f}" size="{tray_depth / 2.0:.4f} {tray_wall:.4f} {tray_wall_height:.4f}" contype="1" conaffinity="1" rgba="0.78 0.73 0.66 1"/>
    <geom name="drawer_tray_back_collision" type="box" pos="{tray_back_x:.4f} {cabinet_y:.4f} {tray_side_z:.4f}" size="{tray_wall:.4f} {tray_half_width:.4f} {tray_wall_height:.4f}" contype="1" conaffinity="1" rgba="0.78 0.73 0.66 1"/>
    <geom name="drawer_left_guide_strip_collision" type="box" pos="{tray_center_x:.4f} {runner_y_left + 0.010:.4f} {support_z:.4f}" size="{tray_depth / 2.0:.4f} 0.006 0.008" contype="1" conaffinity="1" rgba="0.40 0.40 0.41 1"/>
    <geom name="drawer_right_guide_strip_collision" type="box" pos="{tray_center_x:.4f} {runner_y_right - 0.010:.4f} {support_z:.4f}" size="{tray_depth / 2.0:.4f} 0.006 0.008" contype="1" conaffinity="1" rgba="0.40 0.40 0.41 1"/>
    <geom name="drawer_connector_short_boss_collision" type="capsule" fromto="{stem_start_x:.4f} {hy:.4f} {hz:.4f} {stem_end_x:.4f} {hy:.4f} {hz:.4f}" size="{stub_radius:.4f}" contype="1" conaffinity="1" rgba="0.20 0.20 0.22 1"/>
    <geom name="drawer_handle_collision_0" type="sphere" pos="{hx:.4f} {hy:.4f} {hz:.4f}" size="{radius:.4f}" contype="1" conaffinity="1" rgba="0.92 0.90 0.86 1"/>
  </body>
</body>'''

    def build(self):
        xml, assets, metadata, _digest = super().build()
        metadata = dict(metadata)
        metadata.update({
            "reference_aligned_topology_v1": True,
            "solid_drawer_front_panel": True,
            "front_frame_or_cutout_rejected": True,
            "handle_kind": "short_stub_spherical_knob",
            "complete_moving_drawer_box_tray": True,
            "visible_support_guide_semantics": True,
            "controller_algorithm_modified": False,
        })
        return xml, assets, metadata, sha256_text(json.dumps(metadata, sort_keys=True))


def parse_vec(s: str | None) -> list[float]:
    return [] if not s else [float(x) for x in str(s).split()]


def all_geoms(xml_path: Path) -> list[dict[str, Any]]:
    root = ET.parse(xml_path).getroot()
    out = []
    for body in root.findall(".//body"):
        for geom in body.findall("geom"):
            g = dict(geom.attrib); g["body"] = body.attrib.get("name", "")
            g["fromto_vec"] = parse_vec(g.get("fromto")); out.append(g)
    return out


def export_model(candidate: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    inst = out_dir / candidate["candidate_id"]; inst.mkdir(parents=True, exist_ok=True)
    builder = ReferenceAlignedSolidFrontDrawerBuilder(seed=int(candidate.get("synthetic_seed", 9001)), variant=candidate["model_builder_parameters"])
    xml, assets, metadata, semantic_hash = builder.build()
    xml_path = inst / "model.xml"; xml_path.write_text(xml)
    asset_hashes = {}
    if assets:
        ad = inst / "assets"; ad.mkdir(exist_ok=True)
        for name, blob in sorted(assets.items()):
            safe = Path(name).name; (ad / safe).write_bytes(blob); asset_hashes[safe] = hashlib.sha256(blob).hexdigest()
    manifest = {"candidate_id": candidate["candidate_id"], "model_xml": rel(xml_path), "model_xml_sha256": sha256_text(xml), "asset_count": len(asset_hashes), "asset_hashes": asset_hashes, "metadata": metadata, "semantic_hash": semantic_hash}
    write_json(inst / "manifest.json", manifest)
    return manifest


def topology_oracle(xml_path: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    geoms = all_geoms(xml_path); names = [g.get("name", "") for g in geoms]
    moving = [g for g in geoms if g.get("body") == "link_1"]; moving_names = [g.get("name", "") for g in moving]
    frame_like = [n for n in names if any(s in n for s in ["drawer_front_left_panel", "drawer_front_right_panel", "drawer_front_top_panel", "drawer_front_bottom_panel", "drawer_door_left_panel", "drawer_door_right_panel", "drawer_door_top_panel", "drawer_door_bottom_panel"])]
    front = next((g for g in moving if g.get("name") == "drawer_front_panel_collision"), None)
    handle = next((g for g in moving if g.get("name") == "drawer_handle_collision_0"), None)
    boss = next((g for g in moving if g.get("name") == "drawer_connector_short_boss_collision"), None)
    radius = float((handle or {}).get("size", "0").split()[0]) if handle else 0.0
    ft = (boss or {}).get("fromto_vec", []); boss_len = abs(ft[3] - ft[0]) if len(ft) == 6 else 0.0
    ratio = boss_len / (2.0 * radius) if radius else 999.0
    parts = {"front_panel": bool(front), "left_side_wall": "drawer_tray_left_side_collision" in moving_names, "right_side_wall": "drawer_tray_right_side_collision" in moving_names, "bottom_panel": "drawer_tray_bottom_collision" in moving_names, "back_panel": "drawer_tray_back_collision" in moving_names}
    supports = [n for n in names if any(k in n for k in ["runner", "guide_strip", "rail", "guide"])]
    defects = []
    if not front: defects.append("SOLID_FRONT_PANEL_MISSING")
    if frame_like: defects.append("FRAME_LIKE_SEGMENTED_FRONT_PRESENT")
    if not handle: defects.append("SPHERICAL_KNOB_MISSING")
    if not boss: defects.append("SHORT_STUB_BOSS_MISSING")
    if ratio > 0.35: defects.append("KNOB_STUB_TOO_LONG")
    if not all(parts.values()): defects.append("MOVING_DRAWER_BOX_INCOMPLETE")
    if len(supports) < 4: defects.append("VISIBLE_SUPPORT_GUIDE_GEOMS_INSUFFICIENT")
    if candidate.get("model_builder_parameters", {}).get("front_cutout"): defects.append("FRONT_CUTOUT_FLAG_TRUE")
    return {"candidate_id": candidate.get("candidate_id"), "model_xml": rel(xml_path), "model_xml_sha256": sha256_file(xml_path), "solid_drawer_front_panel_passed": front is not None and not frame_like, "short_stub_spherical_knob_passed": bool(handle and boss and ratio <= 0.35), "knob_stub_length_m": boss_len, "knob_radius_m": radius, "knob_stub_length_ratio_of_diameter": ratio, "complete_moving_drawer_box_tray_passed": all(parts.values()), "drawer_box_parts": parts, "support_guide_semantics_passed": len(supports) >= 4, "support_guide_geoms": supports, "moving_drawer_geoms": moving_names, "frame_like_front_geoms": frame_like, "topology_oracle_passed": not defects, "defects": defects}


def install_reference_builder_patch() -> None:
    pool.GeneratedAccessibleDrawerBuilder = ReferenceAlignedSolidFrontDrawerBuilder
    for obj in [patch.cd.bp.cp.layer4r.pool, v3.cd.bp.cp.layer4r.pool]:
        try:
            obj.GeneratedAccessibleDrawerBuilder = ReferenceAlignedSolidFrontDrawerBuilder
        except Exception:
            pass


def install_controller_runtime() -> None:
    v3.install_patch()
    install_reference_builder_patch()
    for mod in [v3, patch]:
        mod.TASK_ID = TASK_ID; mod.SPEC_REL = SPEC_REL; mod.RUN_PREFIX = RUN_PREFIX


def latest_prior() -> Path:
    runs = sorted((CAMPAIGN / "runtime").glob(f"{PRIOR_PREFIX}*"))
    runs = [r for r in runs if (r / "targeted_progress_seed_solution.json").exists()]
    if not runs:
        raise RuntimeError("prior targeted progress run missing")
    return runs[-1]


def base_candidate_and_variant(prior: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = read_json(prior / "targeted_progress_seed_solution.json", {}) or {}
    candidate = deepcopy(seed.get("candidate") or {})
    targeted = read_json(prior / "targeted_shard_results.json", {}) or {}
    variant = targeted.get("selected_variant") if isinstance(targeted.get("selected_variant"), dict) else None
    if not variant:
        rows = targeted.get("rows") or []
        variant = next((r.get("variant") for r in rows if isinstance(r.get("variant"), dict)), None)
    if not variant:
        variant = deepcopy(seed.get("seed_variant") or candidate.get("selected_pull_variant") or {})
    if not variant:
        variant = {"name": "targeted_progress_reference_aligned_fallback", "targeted_progress_feedback_controller": True, "pull_distance_m": 0.35, "pull_velocity_m_per_step": 8.5e-05, "pull_steps": 6800, "pull_press_m": 0.004, "lead_cap_m": 0.02, "progress_base_lead_cap_m": 0.02, "progress_max_lead_cap_m": 0.03, "progress_target_fraction": 0.85, "qpos_progress_gain": 0.018, "post_pull_hold_steps": 0, "finger_mode": "semi_close", "null_gain": 0.0, "op_gain": 10.0, "op_vel_limit": 0.065, "q_vel_limit": 3.0, "servo_kp": 430.0, "servo_kd": 112.0}
    variant = deepcopy(variant); variant["targeted_progress_feedback_controller"] = True; variant["no_direct_qpos_drawer_opening"] = True; variant["no_drawer_motor_command"] = True; variant["controller_algorithm_source"] = "unchanged_v3_progress_feedback_latched_pull_runtime_patch"
    return candidate, variant


def synthesize(base: dict[str, Any]) -> list[dict[str, Any]]:
    p0 = deepcopy(base.get("model_builder_parameters") or {})
    hx = float(p0.get("handle_x", -0.18)); hy = float(p0.get("handle_y", -0.067)); hz = float(p0.get("handle_z", 0.402)); radius = float(p0.get("handle_radius", 0.024))
    specs = [
        ("c00", .14, .010, .060, .040, .165),
        ("c01", .18, .010, .065, .043, .175),
        ("c02", .22, .012, .070, .046, .185),
        ("c03", .14, .012, .075, .050, .175),
        ("c04", .18, .012, .080, .054, .185),
        ("c05", .22, .014, .085, .058, .195),
        ("c06", .14, .010, .055, .038, .160),
        ("c07", .18, .010, .058, .040, .170),
        ("c08", .22, .010, .062, .042, .180),
        ("c09", .14, .008, .034, .020, .150),
        ("c10", .18, .008, .038, .024, .155),
        ("c11", .22, .009, .042, .027, .160),
        ("c12", .14, .009, .046, .030, .165),
        ("c13", .18, .010, .050, .032, .170),
        ("c14", .22, .010, .054, .035, .175),
        ("c15", .34, .004, .030, .018, .145),
        ("c16", .34, .005, .034, .020, .150),
        ("c17", .34, .006, .038, .022, .155),
        ("c18", .30, .004, .030, .018, .145),
        ("c19", .30, .005, .034, .020, .150),
        ("c20", .30, .006, .038, .022, .155),
        ("c21", .34, .004, .030, .018, .145),
        ("c22", .34, .004, .030, .018, .145),
        ("c23", .34, .005, .034, .020, .150),
        ("c24", .34, .005, .034, .020, .150),
        ("c25", .34, .006, .038, .022, .155),
        ("c26", .34, .006, .038, .022, .155),
        ("c27", .30, .004, .030, .018, .145),
        ("c28", .30, .004, .030, .018, .145),
        ("c29", .30, .005, .034, .020, .150),
        ("c30", .30, .005, .034, .020, .150),
        ("c31", .30, .006, .038, .022, .155),
        ("c32", .30, .006, .038, .022, .155),
    ]
    out = []
    for i, (name, stub_ratio, ft, dhw, thw, depth) in enumerate(specs):
        p = deepcopy(p0)
        p.update({"drawer_kind": "reference_aligned_solid_front_short_stub_round_knob_drawer", "reference_aligned_topology_v1": True, "front_cutout": False, "solid_front_panel": True, "frame_only_front_rejected": True, "visual_topology_v2_preserved": False, "handle_kind": "sphere", "handle_x": hx, "handle_y": hy, "handle_z": hz, "handle_radius": radius, "stub_length": stub_ratio * 2.0 * radius, "stub_radius": min(radius * .34, .008), "front_half_thickness": ft, "door_half_width": dhw, "door_half_height": .235, "tray_half_width": thw, "tray_depth": depth, "tray_wall_height": .115, "tray_wall_thickness": .010, "cabinet_half_width": max(.245, thw + .095), "cabinet_depth": max(.18, depth), "cabinet_z": max(.34, hz - .050), "support_guide_semantics": True, "complete_moving_drawer_box_tray": True, "short_stub_spherical_knob_nearly_flush": True, "controller_algorithm_modified": False})
        layout_overrides = {
            "c21": ([-0.700, -0.160, 0.025], -36),
            "c22": ([-0.700, -0.020, 0.025], -16),
            "c23": ([-0.640, -0.180, 0.025], -42),
            "c24": ([-0.600, -0.130, 0.025], -32),
            "c25": ([-0.720, -0.120, 0.025], -28),
            "c26": ([-0.680, -0.210, 0.025], -48),
            "c27": ([-0.700, -0.160, 0.025], -36),
            "c28": ([-0.700, -0.020, 0.025], -16),
            "c29": ([-0.640, -0.180, 0.025], -42),
            "c30": ([-0.600, -0.130, 0.025], -32),
            "c31": ([-0.720, -0.120, 0.025], -28),
            "c32": ([-0.680, -0.210, 0.025], -48),
        }
        if name in layout_overrides:
            base_pos, yaw = layout_overrides[name]
            p["robot_base_pos"] = base_pos
            p["robot_yaw_deg"] = yaw
            p["layout_micro_adjustment_for_solid_front_clearance"] = True
        c = deepcopy(base); c["candidate_id"] = f"reference_aligned_solid_front_{name}_{base.get('candidate_id','seed')}"; c["synthetic_seed"] = int(base.get("synthetic_seed", 9001) or 9001) + i; c["model_builder_parameters"] = p; c["source_type"] = "generated_repaired_reference_aligned_variant"; c["source_type_for_spec"] = c["source_type"]; c["declared_generated_or_repaired_variant"] = True; c["reference_aligned_topology_v1"] = True; c["controller_algorithm_modified"] = False
        out.append(c)
    return out


def run_fast(c, v, run_dir, idx):
    row = patch.run_case_with_group_metrics(c, "fast_guarded_contact", v, run_dir, 1200000 + idx, "reference_aligned_fast")
    row["reference_aligned_fast_passed"] = v3.targeted_row_pass(row); v3.add_targeted_failure_reasons(row); return row


def run_targeted(c, v, run_dir):
    rows = []; out = run_dir / "targeted_shard_results.jsonl"; out.write_text("")
    for i, perturb in enumerate(v3.canonical_targeted_perturbations()):
        row = patch.run_case_with_group_metrics(c, perturb, v, run_dir, 1210000 + i, "reference_aligned_targeted")
        row["targeted_patch_group_passed"] = v3.targeted_row_pass(row); v3.add_targeted_failure_reasons(row); rows.append(row); append_jsonl(out, row)
    report = v3.summarize_targeted_rows(rows); report["selected_variant"] = v; report["selected_candidate_id"] = c.get("candidate_id"); report["rows"] = rows
    write_json(run_dir / "targeted_shard_results.json", report); return report


def write_lock(run_dir, c, audit):
    lock = {"generated_at_utc": utc_now(), "task_id": TASK_ID, "status": "MACHINE_REFERENCE_CONTRACT_LOCKED_PENDING_MANUAL_VISUAL_REVIEW", "approved_candidate_id": c.get("candidate_id"), "approved_model_xml": audit.get("model_xml"), "approved_model_xml_sha256": audit.get("model_xml_sha256"), "manual_visual_review_passed": False, "machine_reference_contract_passed": bool(audit.get("topology_oracle_passed")), "solid_front_panel_required": True, "full_drawer_box_tray_required": True, "short_boss_knob_required": True, "support_rails_guides_required": True, "frame_only_front_rejected": True, "controller_algorithm_hash_changed": False, "reference_visuals": ["/Users/zhuhaowu/Downloads/knob-handler.png", "/Users/zhuhaowu/Downloads/ChatGPT Image 2026年5月5日 23_07_48 (2).png"], "audit": audit}
    write_json(CAMPAIGN / "sovereign/approved_drawer_topology_lock.json", lock); write_json(run_dir / "approved_drawer_topology_lock_snapshot.json", lock); return lock


def final_checks(run_dir):
    pre = subprocess.run(["/root/anaconda3/envs/infinigen/bin/python", "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py", "--spec", SPEC_REL, "--dry-run"], cwd=ROOT, text=True, capture_output=True)
    pyc = subprocess.run(["/root/anaconda3/envs/infinigen/bin/python", "-m", "py_compile", "scripts/mint/v11_g4_reference_aligned_drawer_topology_synthesis_controller_preserving_recertification.py"], cwd=ROOT, text=True, capture_output=True)
    status = run_git(["status", "--short", "--untracked-files=all"])
    d = {"generated_at_utc": utc_now(), "preflight_returncode": pre.returncode, "preflight_stdout_tail": pre.stdout[-4000:], "preflight_stderr_tail": pre.stderr[-4000:], "py_compile_returncode": pyc.returncode, "py_compile_stderr": pyc.stderr, "git_status_short": status.splitlines(), "current_truth_modified": "sovereign/current_truth.json" in status, "next_actions_modified": "sovereign/next_actions.json" in status}
    d["final_checks_passed"] = pre.returncode == 0 and pyc.returncode == 0 and not d["current_truth_modified"] and not d["next_actions_modified"]
    write_json(run_dir / "stage10_final_checks.json", d); return d


def closeout(run_dir, candidates, selected, audit, fast, targeted, full30, export, checks):
    topo = bool(audit and audit.get("topology_oracle_passed")); fast_ok = bool(fast and v3.targeted_row_pass(fast)); targ = bool(targeted and targeted.get("targeted_shard_passed")); f30 = bool(full30 and full30.get("full30_passed")); exp = bool(export and export.get("strict_teacher_export_complete"))
    if not topo: cls, gate = "REFERENCE_ALIGNED_TOPOLOGY_ORACLE_FAILED", "REFERENCE_ALIGNED_TOPOLOGY_SYNTHESIS_REPAIR"
    elif not fast_ok: cls, gate = "CONTROLLER_PRESERVING_FAST_RECERTIFICATION_FAILED", "CONTROLLER_MIGRATION_ON_REFERENCE_TOPOLOGY_REPAIR"
    elif not targ: cls, gate = "TARGETED_FAILED_ON_REFERENCE_ALIGNED_TOPOLOGY", "TARGETED_PROGRESS_ON_REFERENCE_TOPOLOGY_REPAIR"
    elif not f30: cls, gate = "FULL30_FAILED_ON_REFERENCE_ALIGNED_TOPOLOGY", "FULL30_GENERALIZATION_ON_REFERENCE_TOPOLOGY_REPAIR"
    elif not exp: cls, gate = "STRICT_EXPORT_FAILED_ON_REFERENCE_ALIGNED_TOPOLOGY", "STRICT_EXPORT_REPAIR_ON_REFERENCE_TOPOLOGY"
    else: cls, gate = "REFERENCE_ALIGNED_TOPOLOGY_REMOTE_STRICT_EXPORT_READY_LOCAL_REPLAY_REQUIRED", "LOCAL_STATE_REPLAY_RENDER_AND_ACTION_SPOT_CHECK_ON_REFERENCE_TOPOLOGY"
    d = {"generated_at_utc": utc_now(), "task_id": TASK_ID, "closeout_classification": cls, "next_gate": gate, "topology_repair_first": True, "controller_algorithm_modified": False, "topology_candidate_count": len(candidates), "selected_candidate_id": selected.get("candidate_id") if selected else None, "solid_drawer_front_panel_passed": bool(audit and audit.get("solid_drawer_front_panel_passed")), "short_stub_spherical_knob_passed": bool(audit and audit.get("short_stub_spherical_knob_passed")), "complete_moving_drawer_box_tray_passed": bool(audit and audit.get("complete_moving_drawer_box_tray_passed")), "support_guide_semantics_passed": bool(audit and audit.get("support_guide_semantics_passed")), "topology_lock_written": (CAMPAIGN / "sovereign/approved_drawer_topology_lock.json").exists(), "topology_lock_manual_visual_review_passed": False, "fast_guarded_contact_passed": fast_ok, "best_fast_drawer_fraction": v3.row_drawer_fraction(fast or {}), "best_fast_bilateral_exact_frames": v3.row_bilateral(fast or {}), "best_fast_forbidden_frames": v3.row_forbidden(fast or {}), "best_fast_handle_nonlegal_frames": v3.row_handle_nonlegal(fast or {}), "best_fast_max_penetration_m": v3.row_pen(fast or {}), "targeted_shard_passed": targ, "targeted_cases_passed": int((targeted or {}).get("cases_passed", 0) or 0), "targeted_cases_total": int((targeted or {}).get("cases_total", 0) or 0), "worst_case_targeted_drawer_fraction": float((targeted or {}).get("worst_case_targeted_drawer_fraction", 0.0) or 0.0), "full30_passed": f30, "full30_cases_passed": int((full30 or {}).get("full30_cases_passed", 0) or 0), "full30_cases_total": int((full30 or {}).get("full30_cases_total", 0) or 0), "strict_export_complete": exp, "local_state_replay_passed": False, "visual_review_packet_written": (run_dir / "visual_topology_review_packet.md").exists(), "action_only_spot_check_passed": False, "current_truth_modified": bool(checks and checks.get("current_truth_modified")), "next_actions_modified": bool(checks and checks.get("next_actions_modified")), "remote_commit_hash": run_git(["rev-parse", "HEAD"]), "git_status_short": run_git(["status", "--short", "--untracked-files=all"]).splitlines()}
    write_json(run_dir / "final_closeout.json", d); return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir"); ap.add_argument("--max-topology-candidates", type=int, default=6); args = ap.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else CAMPAIGN / "runtime" / f"{RUN_PREFIX}_{utc_stamp()}"; run_dir.mkdir(parents=True, exist_ok=True)
    prior = latest_prior(); base, variant = base_candidate_and_variant(prior)
    write_json(run_dir / "stage0_manual_visual_review_ingestion.json", {"generated_at_utc": utc_now(), "prior_success_run": rel(prior), "manual_visual_review_downgrade": "STATE_REPLAY_METRICS_PASS_BUT_VISUAL_TOPOLOGY_REVIEW_FAILED", "base_candidate_id": base.get("candidate_id"), "selected_controller_variant": variant, "controller_algorithm_modified": False})
    (run_dir / "stage0_manual_visual_review_report.md").write_text("# Manual Visual Review Ingestion\n\nTopology-invalid exported replay is downgraded. This helper repairs topology first and migrates the existing controller unchanged.\n")
    install_controller_runtime()
    candidates = synthesize(base)[:args.max_topology_candidates]
    selected = audit = fast = targeted = full30 = export = None
    (run_dir / "reference_aligned_topology_candidates.jsonl").write_text(""); (run_dir / "reference_aligned_fast_results.jsonl").write_text("")
    audits = []; manifest_rows = []
    for idx, c in enumerate(candidates):
        manifest = export_model(c, run_dir / "generated_instances"); c["model_manifest"] = manifest
        a = topology_oracle(ROOT / manifest["model_xml"], c); audits.append(a); manifest_rows.append({"candidate_id": c.get("candidate_id"), "model_manifest": manifest, "topology_audit": a}); append_jsonl(run_dir / "reference_aligned_topology_candidates.jsonl", {"candidate": c, "model_manifest": manifest, "topology_audit": a})
        if not a.get("topology_oracle_passed"): continue
        # Recompute per-instance binding, handle frame, two-pad frame, and IK on the
        # repaired topology. This preserves the controller algorithm while avoiding
        # stale frame/cutout waypoints from the topology-invalid ancestor model.
        physical = pool.evaluate_generated_candidate(c, run_dir)
        physical["reference_aligned_topology_oracle"] = a
        append_jsonl(run_dir / "reference_aligned_physical_accessibility_results.jsonl", physical)
        if not physical.get("accepted"):
            if selected is None:
                selected, audit = physical, a
            continue
        physical["model_manifest"] = physical.get("model_manifest") or manifest
        if selected is None: selected, audit = physical, a
        row = run_fast(physical, variant, run_dir, idx); append_jsonl(run_dir / "reference_aligned_fast_results.jsonl", row)
        if v3.targeted_row_pass(row): selected, audit, fast = physical, a, row; break
        if fast is None or v3.row_drawer_fraction(row) > v3.row_drawer_fraction(fast): selected, audit, fast = physical, a, row
    write_json(run_dir / "reference_aligned_topology_candidate_manifest.json", {"generated_at_utc": utc_now(), "candidates": manifest_rows})
    write_json(run_dir / "reference_aligned_topology_oracle_report.json", {"generated_at_utc": utc_now(), "audits": audits, "topology_candidates_total": len(candidates), "topology_oracle_passed_count": sum(1 for a in audits if a.get("topology_oracle_passed")), "selected_candidate_id": selected.get("candidate_id") if selected else None, "selected_audit": audit})
    write_json(run_dir / "canonical_topology_render_manifest.json", {"generated_at_utc": utc_now(), "render_attempted": False, "reason": "remote XML/topology oracle stage only; claim-bearing local render follows strict export", "selected_model_xml": (audit or {}).get("model_xml")})
    (run_dir / "visual_topology_review_packet.md").write_text("# Visual Topology Review Packet\n\nMachine oracle requires a solid front panel, short-stub spherical knob, complete drawer tray, and visible guide semantics. Manual visual review remains pending after local claim render.\n")
    if selected and audit and audit.get("topology_oracle_passed"): write_lock(run_dir, selected, audit)
    if fast and v3.targeted_row_pass(fast) and selected:
        targeted = run_targeted(selected, variant, run_dir)
        if targeted.get("targeted_shard_passed"):
            full30 = v3.write_full30_after_targeted(run_dir, targeted, selected, variant)
            if full30.get("full30_passed"): export = v3.strict_export_after_full30(run_dir, full30, selected)
    if targeted is None: targeted = {"targeted_shard_attempted": False, "targeted_shard_passed": False, "skip_reason": "fast_not_passed_on_reference_aligned_topology"}; write_json(run_dir / "targeted_shard_results.json", targeted); (run_dir / "targeted_shard_results.jsonl").write_text("")
    if full30 is None: full30 = {"full30_attempted": False, "full30_passed": False, "skip_reason": "targeted_not_passed_on_reference_aligned_topology"}; write_json(run_dir / "full30_report.json", full30); (run_dir / "full30_exact_latch_pull_keepout_certification.jsonl").write_text("")
    if export is None: export = {"strict_teacher_export_attempted": False, "strict_teacher_export_complete": False, "strict_export_complete": False, "skip_reason": "full30_not_passed_on_reference_aligned_topology"}; write_json(run_dir / "strict_teacher_export_manifest.json", export)
    checks = final_checks(run_dir); co = closeout(run_dir, candidates, selected, audit, fast, targeted, full30, export, checks)
    print(json.dumps(ready({"run_dir": rel(run_dir), "closeout": co}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

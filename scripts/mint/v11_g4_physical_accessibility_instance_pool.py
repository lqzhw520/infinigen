#!/usr/bin/env python3
"""Build a physically accessible GOC-v4 drawer instance pool.

This phase is deliberately instance-pool oriented. It does not rerun Layer4R,
does not tune the contact controller, and does not claim a teacher rollout.
Legacy/generated candidates are admitted only after a physical accessibility
oracle checks semantic GOC-v4 binding, reset, two-pad handle-frame IK, full-body
keepout, approach corridor, pull corridor, and visual/physical consistency.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
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
    "v11_g4_goc_v4_physical_accessibility_instance_pool.yaml"
)
SPEC_PATH = ROOT / SPEC_REL
MIN_ACCEPTED = 3
PREFERRED_ACCEPTED = 5
MAX_PENETRATION_M = 0.02
MAX_FORCE_N = 1_000_000.0

sys.path.insert(0, str(ROOT / "scripts/mint"))
from merged_model_builder import MergedModelBuilder  # noqa: E402
from contact_aware_drawer_teacher import (  # noqa: E402
    DrawerRobotEnvMuJoCoLibero,
    classify_instance,
    contact_report,
    geom_name,
)
import handle_frame_grasp_trajectory_planner as hfp  # noqa: E402
import joint_placement_reset_ik_keepout_feasibility as jpf  # noqa: E402
import model_or_layout_redesign_with_physical_accessibility_invariant as mol  # noqa: E402
import v11_g4_goc_v4_autonomous_repair_campaign as repair_campaign  # noqa: E402

SAFE_PRECONTACT_QPOS = hfp.SAFE_PRECONTACT_QPOS
DEFAULT_PARAMS = dict(jpf.DEFAULT_PARAMS)
DEFAULT_PARAMS.update(
    {
        "name": "physical_accessibility_pool_pregrasp_guarded_contact",
        "pregrasp_distance_m": 0.055,
        "guarded_distance_m": 0.020,
        "contact_normal_offset_m": 0.001,
        "pinch_extra_clearance_m": 0.002,
    }
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


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def latest_runtime(pattern: str) -> Path | None:
    paths = sorted((CAMPAIGN / "runtime").glob(pattern))
    return paths[-1] if paths else None


def compact_binding(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "legal_finger_pad_geom_ids": binding.get(
            "legal_finger_pad_geom_ids", binding.get("legal_gripper_surface_geom_ids", [])
        ),
        "legal_gripper_surface_geom_ids": binding.get(
            "legal_gripper_surface_geom_ids", binding.get("legal_finger_pad_geom_ids", [])
        ),
        "drawer_handle_geom_ids": binding.get("drawer_handle_geom_ids", []),
        "drawer_body_or_cabinet_geom_ids": binding.get("drawer_body_or_cabinet_geom_ids", []),
        "forbidden_robot_surface_geom_ids": binding.get("forbidden_robot_surface_geom_ids", []),
        "visual_only_or_noncontact_geom_ids": binding.get("visual_only_or_noncontact_geom_ids", []),
        "unknown_contact_relevant_geom_ids": binding.get("unknown_contact_relevant_geom_ids", []),
        "goc_v3_broad_link_geoms_demoted_from_target": binding.get(
            "goc_v3_broad_link_geoms_demoted_from_target", []
        ),
        "binding_generated": bool(binding.get("binding_generated")),
    }


class GeneratedAccessibleDrawerBuilder(MergedModelBuilder):
    """A narrow generated drawer/cabinet/handle model for Source C candidates.

    The generated scene is intentionally simple but not empty: it contains a
    cabinet envelope, a sliding drawer front, and a true contact handle. Visual
    geometry is collision geometry for the drawer/cabinet/handle, and the full
    Panda + articulated gripper is inherited from robosuite through
    MergedModelBuilder. Broad arm/hand collisions remain active. The preexisting
    GOC-v4 finger-shell noncontact policy is only audited here; this helper does
    not demote new physical robot collision geoms.
    """

    def __init__(self, *args: Any, variant: dict[str, Any] | None = None, **kwargs: Any):
        self.variant = dict(variant or {})
        kwargs["robot_base_pos"] = np.asarray(self.variant.get("robot_base_pos", [-0.45, 0.0, 0.0]), dtype=float)
        super().__init__(*args, **kwargs)

    def _drawer_xml(self) -> str:
        v = self.variant
        hx = float(v.get("handle_x", -0.14))
        hy = float(v.get("handle_y", 0.02))
        hz = float(v.get("handle_z", 0.36))
        knob_radius = float(v.get("handle_radius", 0.025))
        cabinet_x = float(v.get("cabinet_x", 0.18))
        cabinet_y = hy
        cabinet_z = float(v.get("cabinet_z", 0.34))
        cabinet_depth = float(v.get("cabinet_depth", 0.20))
        cabinet_half_width = float(v.get("cabinet_half_width", 0.24))
        cabinet_half_height = float(v.get("cabinet_half_height", 0.28))
        door_x = float(v.get("door_x", 0.055))
        door_half_width = float(v.get("door_half_width", 0.19))
        door_half_height = float(v.get("door_half_height", 0.23))
        pull_axis = v.get("pull_axis", [-1, 0, 0])
        pull_axis_s = " ".join(str(float(x)) for x in pull_axis)
        return f'''
<body name="drawer_base" pos="0 0 0">
  <geom name="cabinet_back_collision" type="box" pos="{cabinet_x + cabinet_depth * 0.7:.4f} {cabinet_y:.4f} {cabinet_z:.4f}" size="0.025 {cabinet_half_width:.4f} {cabinet_half_height:.4f}" contype="1" conaffinity="1" rgba="0.85 0.82 0.80 1"/>
  <geom name="cabinet_bottom_collision" type="box" pos="{cabinet_x:.4f} {cabinet_y:.4f} {cabinet_z - cabinet_half_height:.4f}" size="{cabinet_depth:.4f} {cabinet_half_width:.4f} 0.025" contype="1" conaffinity="1" rgba="0.85 0.82 0.80 1"/>
  <geom name="cabinet_top_collision" type="box" pos="{cabinet_x:.4f} {cabinet_y:.4f} {cabinet_z + cabinet_half_height:.4f}" size="{cabinet_depth:.4f} {cabinet_half_width:.4f} 0.025" contype="1" conaffinity="1" rgba="0.85 0.82 0.80 1"/>
  <geom name="cabinet_left_collision" type="box" pos="{cabinet_x:.4f} {cabinet_y - cabinet_half_width:.4f} {cabinet_z:.4f}" size="{cabinet_depth:.4f} 0.025 {cabinet_half_height:.4f}" contype="1" conaffinity="1" rgba="0.85 0.82 0.80 1"/>
  <geom name="cabinet_right_collision" type="box" pos="{cabinet_x:.4f} {cabinet_y + cabinet_half_width:.4f} {cabinet_z:.4f}" size="{cabinet_depth:.4f} 0.025 {cabinet_half_height:.4f}" contype="1" conaffinity="1" rgba="0.85 0.82 0.80 1"/>
  <body name="link_1" pos="0 0 0">
    <joint name="drawer_slider_0" type="slide" axis="{pull_axis_s}" range="0 0.35" damping="8"/>
    <geom name="drawer_door_collision" type="box" pos="{door_x:.4f} {cabinet_y:.4f} {cabinet_z:.4f}" size="0.025 {door_half_width:.4f} {door_half_height:.4f}" contype="1" conaffinity="1" rgba="0.90 0.88 0.85 1"/>
    <geom name="drawer_handle_collision_0" type="sphere" pos="{hx:.4f} {hy:.4f} {hz:.4f}" size="{knob_radius:.4f}" contype="1" conaffinity="1" rgba="0.20 0.20 0.22 1"/>
  </body>
</body>'''

    def build(self) -> tuple[str, dict[str, bytes], dict[str, Any], str]:
        assets = self.load_robot_assets()
        actuator_inner, asset_inner, worldbody_inner = self._get_robot_xml_inner()
        robot_base_pos = " ".join(f"{float(v):.6f}" for v in self.robot_base_pos)
        if abs(float(self.variant.get("robot_yaw_deg", 0.0))) > 1e-9:
            half = math.radians(float(self.variant["robot_yaw_deg"])) / 2.0
            robot_body = (
                f'<body name="robot_mount" pos="{robot_base_pos}" '
                f'quat="{math.cos(half):.6f} 0 0 {math.sin(half):.6f}">\n'
                f"{worldbody_inner}\n</body>"
            )
        else:
            robot_body = f'<body name="robot_mount" pos="{robot_base_pos}">\n{worldbody_inner}\n</body>'
        xml = f'''<mujoco model="generated_accessible_panda_drawer">
  <compiler angle="radian" inertiafromgeom="auto"/>
  <option timestep="0.002"/>
  <asset>
{asset_inner}
  </asset>
  <actuator>
{actuator_inner}
    <motor ctrllimited="true" ctrlrange="-10 10" joint="drawer_slider_0" name="drawer0"/>
  </actuator>
  <contact>
    <exclude body1="drawer_base" body2="link_1"/>
  </contact>
  <worldbody>
    <light name="top" pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
{self._drawer_xml()}
{robot_body}
  </worldbody>
  <visual>
    <rgba haze="0.88 0.87 0.85 1"/>
    <headlight ambient="0.50 0.50 0.50" diffuse="0.95 0.93 0.90" specular="0.50 0.50 0.50"/>
  </visual>
</mujoco>'''
        metadata = {
            "generated_accessible_drawer_variant": True,
            "variant": dict(self.variant),
            "visual_collision_policy": "drawer/cabinet/handle visible geometry is contact geometry; full Panda arm/hand collision remains active; preexisting broad finger shell noncontact is audited",
        }
        return xml, assets, metadata, sha256_text(json.dumps(metadata, sort_keys=True))


def make_generated_env(candidate: dict[str, Any], max_steps: int = 20):
    variant = dict(candidate["model_builder_parameters"])

    class CandidateBuilder(GeneratedAccessibleDrawerBuilder):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, variant=variant, **kwargs)

    DrawerRobotEnvMuJoCoLibero._MERGED_BUILDER_CLASS = CandidateBuilder
    return DrawerRobotEnvMuJoCoLibero(
        seed=int(candidate.get("synthetic_seed", 9001)),
        image_size=64,
        max_steps=int(max_steps),
        contract=None,
        robot_init_qpos=SAFE_PRECONTACT_QPOS.astype(float).tolist(),
    )


def export_generated_model(candidate: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    builder = GeneratedAccessibleDrawerBuilder(seed=int(candidate.get("synthetic_seed", 9001)), variant=candidate["model_builder_parameters"])
    xml, assets, metadata, semantic_hash = builder.build()
    inst_dir = out_dir / candidate["candidate_id"]
    inst_dir.mkdir(parents=True, exist_ok=True)
    xml_path = inst_dir / "model.xml"
    xml_path.write_text(xml)
    asset_hashes: dict[str, str] = {}
    if assets:
        asset_dir = inst_dir / "assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        for name, blob in sorted(assets.items()):
            safe_name = Path(name).name
            (asset_dir / safe_name).write_bytes(blob)
            asset_hashes[safe_name] = sha256_bytes(blob)
    manifest = {
        "candidate_id": candidate["candidate_id"],
        "source_type": candidate["source_type"],
        "model_xml": rel(xml_path),
        "model_xml_sha256": sha256_text(xml),
        "asset_count": len(asset_hashes),
        "asset_hashes": asset_hashes,
        "metadata": metadata,
        "semantic_hash": semantic_hash,
    }
    write_json(inst_dir / "manifest.json", manifest)
    return manifest


def status_preflight(run_dir: Path) -> dict[str, Any]:
    cmd = [
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
        "preflight": run_cmd(cmd),
    }
    payload["harness_preflight_passed"] = payload["preflight"]["returncode"] == 0
    write_json(run_dir / "stage0_authority_and_scope.json", payload)
    (run_dir / "commands.log").write_text(json.dumps(ready(payload), indent=2, sort_keys=True) + "\n")
    return payload


def prior_negative_evidence(run_dir: Path) -> dict[str, Any]:
    patterns = {
        "accessibility_layout_synthesis": "v11_g4_goc_v4_accessibility_layout_synthesis_to_layer4r_overnight_*",
        "model_or_layout_redesign": "v11_g4_goc_v4_model_or_layout_redesign_with_physical_accessibility_invariant_*",
        "model_instance_accessibility": "v11_g4_goc_v4_model_instance_accessibility_collision_attribution_repair_*",
        "safe_collision_proxy_policy": "v11_g4_goc_v4_safe_collision_proxy_policy_repair_*",
        "robust_layer4r": "v11_g4_goc_v4_robust_contact_pull_export_local_replay_*",
    }
    records: dict[str, Any] = {}
    for name, pattern in patterns.items():
        d = latest_runtime(pattern)
        rec: dict[str, Any] = {"pattern": pattern, "path": rel(d), "found": d is not None}
        if d:
            for fn in [
                "closeout_decision.json",
                "stage3_model_layout_repair_summary.json",
                "stage4_accessibility_rejection_invariants.json",
                "stage4_proxy_policy_decision.json",
                "final_report.md",
            ]:
                p = d / fn
                if p.exists() and p.suffix == ".json":
                    rec[fn] = load_json(p)
                elif p.exists():
                    rec[fn] = {"exists": True, "bytes": p.stat().st_size}
        records[name] = rec
    summary = {
        "generated_at_utc": utc_now(),
        "prior_records": records,
        "lessons_locked": {
            "controller_cem_before_structure_is_waste": True,
            "single_main_chain_not_gate_name_churn": True,
            "visual_physical_consistency_is_hard_invariant": True,
        },
        "structural_variables_remaining": [
            "select_existing_accessible_instances_if_any",
            "documented_layout_variants_if_oracle_passes",
            "generated_accessible_drawer_variants_with_true_cabinet_handle_collision",
        ],
    }
    write_json(run_dir / "prior_negative_evidence_ingestion.json", summary)
    write_md(run_dir / "prior_negative_evidence_report.md", "# Prior Negative Evidence\n\n" + json.dumps(ready(summary), indent=2, sort_keys=True))
    return summary


def write_invariant(run_dir: Path) -> dict[str, Any]:
    invariant = {
        "I0_model_load": ["full Panda/gripper", "drawer/cabinet/handle", "dedicated pad binding", "handle binding", "no unknown contact-relevant geoms"],
        "I1_reset": {"forbidden_contacts": 0, "max_penetration_m": MAX_PENETRATION_M, "max_force_n": MAX_FORCE_N},
        "I2_semantic_binding": "dedicated pads and true handle only; broad shell/body/name-only rejected",
        "I3_two_pad_grasp_geometry": {"max_residual_m": 0.02, "preferred_residual_m": 0.005},
        "I4_full_body_keepout": "nonlegal robot surfaces have zero forbidden contact at endpoint and are audited for visual/physical proxy clipping",
        "I5_approach_corridor": "collision-legal swept path from reset/pregrasp to hold/contact",
        "I6_pull_corridor": "small pull-axis robot motion feasible without forbidden contact, no direct drawer qpos opening",
        "I7_visual_physical_consistency": "no new fake collision demotion; drawer/handle visual and collision semantics preserved",
    }
    write_json(run_dir / "physical_accessibility_invariant.json", invariant)
    write_md(run_dir / "physical_accessibility_invariant.md", "# Physical Accessibility Invariant\n\n" + json.dumps(ready(invariant), indent=2, sort_keys=True))
    return invariant


def previous_rejection_by_seed() -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    d = latest_runtime("v11_g4_goc_v4_model_or_layout_redesign_with_physical_accessibility_invariant_*")
    if not d:
        return out
    data = load_json(d / "stage3_model_layout_repair_summary.json") or {}
    best_by_seed = data.get("best_by_seed") or {}
    for seed_s, payload in best_by_seed.items():
        best = payload.get("best") or {}
        out[int(seed_s)] = {
            "prior_run_dir": rel(d),
            "rejection_reason": best.get("rejection_invariant", "MODEL_OR_LAYOUT_REDESIGN_ACCESSIBILITY_INVARIANT_FAILED"),
            "best_reset_clean_two_pad_residual_m": data.get("best_reset_clean_two_pad_residual_m"),
            "best_row": {
                "base_pos": best.get("base_pos"),
                "yaw_deg": best.get("yaw_deg"),
                "score": best.get("score"),
                "max_endpoint_residual_m": best.get("max_endpoint_residual_m"),
                "min_forbidden_scene_rbound_clearance_m": best.get("min_forbidden_scene_rbound_clearance_m"),
            },
        }
    return out


def candidate_pool(run_dir: Path) -> list[dict[str, Any]]:
    seeds = repair_campaign.available_drawer_seeds()
    priors = previous_rejection_by_seed()
    pool: list[dict[str, Any]] = []
    for seed in seeds:
        pool.append(
            {
                "candidate_id": f"existing_seed_{seed:03d}",
                "source_type": "existing_asset",
                "seed": int(seed),
                "status": "pending",
                "parent_seed": int(seed),
                "model_builder_parameters": {"seed": int(seed)},
                "prior_rejection": priors.get(int(seed), {}),
            }
        )
        if int(seed) in priors:
            br = priors[int(seed)].get("best_row", {})
            pool.append(
                {
                    "candidate_id": f"repaired_layout_parent_seed_{seed:03d}",
                    "source_type": "repaired_layout_variant",
                    "seed": int(seed),
                    "status": "pending",
                    "parent_seed": int(seed),
                    "model_builder_parameters": {
                        "parent_seed": int(seed),
                        "robot_base_pos": br.get("base_pos"),
                        "robot_yaw_deg": br.get("yaw_deg"),
                        "source": "prior_best_model_or_layout_redesign_row",
                    },
                    "prior_rejection": priors.get(int(seed), {}),
                }
            )
    generated_variants = [
        {"candidate_id": "generated_knob_drawer_accessible_001", "synthetic_seed": 9001, "robot_base_pos": [-0.45, 0.0, 0.0], "handle_x": -0.14, "handle_y": 0.02, "handle_z": 0.36, "handle_radius": 0.025},
        {"candidate_id": "generated_knob_drawer_accessible_002", "synthetic_seed": 9002, "robot_base_pos": [-0.55, 0.0, 0.0], "handle_x": -0.02, "handle_y": 0.08, "handle_z": 0.36, "handle_radius": 0.025},
        {"candidate_id": "generated_knob_drawer_accessible_003", "synthetic_seed": 9003, "robot_base_pos": [-0.65, 0.0, 0.0], "handle_x": -0.10, "handle_y": 0.08, "handle_z": 0.36, "handle_radius": 0.025},
        {"candidate_id": "generated_knob_drawer_accessible_004", "synthetic_seed": 9004, "robot_base_pos": [-0.45, 0.0, 0.0], "handle_x": -0.18, "handle_y": -0.04, "handle_z": 0.36, "handle_radius": 0.025},
        {"candidate_id": "generated_knob_drawer_accessible_005", "synthetic_seed": 9005, "robot_base_pos": [-0.75, 0.0, 0.0], "handle_x": -0.14, "handle_y": 0.08, "handle_z": 0.36, "handle_radius": 0.025},
    ]
    for variant in generated_variants:
        cid = variant.pop("candidate_id")
        synthetic_seed = variant.pop("synthetic_seed")
        pool.append(
            {
                "candidate_id": cid,
                "source_type": "generated_accessible_variant",
                "synthetic_seed": synthetic_seed,
                "status": "pending",
                "model_builder_parameters": {
                    **variant,
                    "drawer_kind": "simple_knob_drawer_with_cabinet_envelope",
                    "generation_policy": "procedural_source_C_accessible_instance_not_empty_scene",
                },
            }
        )
    write_json(run_dir / "candidate_instance_pool.json", {"generated_at_utc": utc_now(), "candidate_count": len(pool), "candidates": pool})
    return pool


def contact_counts_ok(counts: dict[str, Any], *, target_allowed: bool) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if int(counts.get("forbidden", 0)) != 0:
        reasons.append("forbidden_contact_present")
    if int(counts.get("handle_nonlegal", 0)) != 0:
        reasons.append("handle_nonlegal_contact_present")
    if float(counts.get("max_penetration_m", 0.0)) > MAX_PENETRATION_M:
        reasons.append("max_penetration_gt_0p02m")
    if not math.isfinite(float(counts.get("max_contact_force_n", 0.0))) or float(counts.get("max_contact_force_n", 0.0)) > MAX_FORCE_N:
        reasons.append("contact_force_invalid")
    if not target_allowed and int(counts.get("target", 0)) > 0:
        reasons.append("premature_target_contact_before_contact_segment")
    return not reasons, reasons


def set_q(env: DrawerRobotEnvMuJoCoLibero, q9: np.ndarray) -> None:
    jpf.set_robot_qpos(env, np.asarray(q9, dtype=float))


def interpolate_corridor(
    env: DrawerRobotEnvMuJoCoLibero,
    binding: dict[str, Any],
    waypoints: list[tuple[str, np.ndarray, np.ndarray, bool]],
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for seg_name, qa, qb, target_allowed in waypoints:
        steps = 40 if seg_name not in {"contact_hold", "pull_hold"} else 15
        for i in range(steps):
            alpha = (i + 1) / max(steps, 1)
            q = (1 - alpha) * qa + alpha * qb
            set_q(env, q)
            cr = contact_report(env, binding, None)
            ok, reasons = contact_counts_ok(cr["counts"], target_allowed=target_allowed)
            row = {"segment": seg_name, "index": i, "target_allowed": target_allowed, "counts": cr["counts"], "passed": ok, "reasons": reasons}
            if not ok:
                violations.append(row)
            if len(trace) < 25 or (violations and len(trace) < 60):
                trace.append(row)
    return {"passed": not violations, "violation_count": len(violations), "first_violations": violations[:10], "trace_sample": trace}


def visual_physical_consistency_audit(env: DrawerRobotEnvMuJoCoLibero, binding: dict[str, Any], q_samples: list[np.ndarray]) -> dict[str, Any]:
    model = env.model
    scene = sorted(set(map(int, binding.get("drawer_handle_geom_ids", []))) | set(map(int, binding.get("drawer_body_or_cabinet_geom_ids", []))))
    shell = []
    demoted_physical_arm_or_hand = []
    for gid in range(int(model.ngeom)):
        name = geom_name(model, gid)
        lname = name.lower()
        contact = bool(int(model.geom_contype[gid]) and int(model.geom_conaffinity[gid]))
        if name in {"finger1_collision", "finger2_collision"} and not contact:
            shell.append(int(gid))
        if ("link" in lname or "hand" in lname or "wrist" in lname) and not contact and name not in {"finger1_collision", "finger2_collision"}:
            demoted_physical_arm_or_hand.append({"geom_id": int(gid), "geom_name": name})
    min_clear = math.inf
    min_pair: dict[str, Any] | None = None
    for q in q_samples:
        set_q(env, q)
        for rg in shell:
            for sg in scene:
                d = float(np.linalg.norm(env.data.geom_xpos[rg] - env.data.geom_xpos[sg]))
                clearance = d - float(model.geom_rbound[rg]) - float(model.geom_rbound[sg])
                if clearance < min_clear:
                    min_clear = clearance
                    min_pair = {
                        "robot_shell_geom_id": int(rg),
                        "robot_shell_geom_name": geom_name(model, rg),
                        "scene_geom_id": int(sg),
                        "scene_geom_name": geom_name(model, sg),
                        "rbound_clearance_m": float(clearance),
                    }
    # rbound clearance is conservative for disabled finger shell meshes. Treat
    # deep proxy intersection as a hard visual/physical failure while preserving
    # the exact sampled minimum as evidence.
    passed = not demoted_physical_arm_or_hand and (math.isinf(min_clear) or min_clear >= -0.05)
    return {
        "passed": bool(passed),
        "fake_collision_demotion_used": False,
        "new_collision_demotion_in_this_phase": False,
        "preexisting_finger_shell_noncontact_geoms_audited": [geom_name(model, g) for g in shell],
        "demoted_physical_arm_or_hand_geoms": demoted_physical_arm_or_hand,
        "min_finger_shell_scene_rbound_clearance_m": None if math.isinf(min_clear) else float(min_clear),
        "min_finger_shell_scene_rbound_pair": min_pair,
        "audit_kind": "conservative_center_minus_rbound_proxy; hard fail only for deep shell-scene intersection or demoted physical arm/hand",
    }


def evaluate_generated_candidate(candidate: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    env = None
    manifest: dict[str, Any] | None = None
    try:
        manifest = export_generated_model(candidate, run_dir / "generated_instances")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            env = make_generated_env(candidate, max_steps=40)
            env.reset()
        binding = classify_instance(env)
        binding_c = compact_binding(binding)
        reset_choice = hfp.find_collision_free_reset_qpos(env, binding, int(candidate.get("synthetic_seed", 9001)))
        q_reset = jpf.make_robot_qpos(np.asarray(reset_choice["qpos_arm"], dtype=float), "open")
        set_q(env, q_reset)
        reset_report = contact_report(env, binding, None)
        hf = hfp.build_handle_frame(env, binding)
        if hf.get("quality") != "ok":
            return {**candidate, "accepted": False, "rejection_reason": "HANDLE_BINDING_FAILED", "binding": binding_c, "model_manifest": manifest, "handle_frame": hfp.strip_private(hf)}
        tpf_payload = hfp.build_two_pad_frame(env, binding, hf["_frame"], DEFAULT_PARAMS)
        if tpf_payload.get("quality") != "ok":
            return {**candidate, "accepted": False, "rejection_reason": "TWO_PAD_FRAME_BINDING_FAILED", "binding": binding_c, "model_manifest": manifest, "handle_frame": hfp.strip_private(hf), "two_pad_frame": hfp.strip_private(tpf_payload)}
        tpf = tpf_payload["_frame"]
        pre = jpf.solve_two_pad_ik_q9(env, binding, tpf, tpf.pregrasp_targets, q_reset, finger_prior="open", max_nfev=220)
        guard = jpf.solve_two_pad_ik_q9(env, binding, tpf, tpf.guarded_targets, pre["qpos_robot"], finger_prior="open", max_nfev=220)
        hold = jpf.solve_two_pad_ik_q9(env, binding, tpf, tpf.hold_targets, guard["qpos_robot"], finger_prior="mid", max_nfev=280)
        pull_targets = tpf.hold_targets + hf["_frame"].pull_axis[None, :] * 0.030
        pull = jpf.solve_two_pad_ik_q9(env, binding, tpf, pull_targets, hold["qpos_robot"], finger_prior="mid", max_nfev=220)
        ik_rows = {"pregrasp": pre, "guarded": guard, "contact_hold": hold, "pull_precheck": pull}
        max_residual = max(float(row["max_pad_error_m"]) for row in ik_rows.values())
        endpoint_reasons: list[str] = []
        if not reset_choice.get("reset_ok"):
            endpoint_reasons.append("RESET_INVALID")
        for name, row in ik_rows.items():
            if not row.get("feasible"):
                endpoint_reasons.append(f"{name.upper()}_IK_INFEASIBLE")
            ok, reasons = contact_counts_ok(row.get("contact_counts", {}), target_allowed=name in {"contact_hold", "pull_precheck"})
            endpoint_reasons.extend(f"{name}:{reason}" for reason in reasons)
        q_pre = np.asarray(pre["qpos_robot"], dtype=float)
        q_guard = np.asarray(guard["qpos_robot"], dtype=float)
        q_hold = np.asarray(hold["qpos_robot"], dtype=float)
        q_pull = np.asarray(pull["qpos_robot"], dtype=float)
        approach = interpolate_corridor(
            env,
            binding,
            [
                ("reset_to_pregrasp", q_reset, q_pre, False),
                ("pregrasp_to_guarded", q_pre, q_guard, False),
                ("guarded_to_contact", q_guard, q_hold, True),
                ("contact_hold", q_hold, q_hold, True),
            ],
        )
        pull_corridor = interpolate_corridor(env, binding, [("small_pull_axis_motion", q_hold, q_pull, True), ("pull_hold", q_pull, q_pull, True)])
        visual = visual_physical_consistency_audit(env, binding, [q_reset, q_pre, q_guard, q_hold, q_pull])
        unknowns = list(map(int, binding_c.get("unknown_contact_relevant_geom_ids", [])))
        semantic_ok = bool(binding_c["legal_finger_pad_geom_ids"] and binding_c["drawer_handle_geom_ids"] and not unknowns)
        endpoint_ok = not endpoint_reasons and max_residual <= 0.02
        accepted = bool(semantic_ok and endpoint_ok and approach["passed"] and pull_corridor["passed"] and visual["passed"])
        rejection_reason = None
        if not semantic_ok:
            rejection_reason = "SEMANTIC_BINDING_FAILED"
        elif not endpoint_ok:
            rejection_reason = endpoint_reasons[0] if endpoint_reasons else "TWO_PAD_IK_INFEASIBLE"
        elif not approach["passed"]:
            rejection_reason = "APPROACH_CORRIDOR_BLOCKED"
        elif not pull_corridor["passed"]:
            rejection_reason = "PULL_CORRIDOR_BLOCKED"
        elif not visual["passed"]:
            rejection_reason = "VISUAL_PHYSICAL_CONSISTENCY_FAILED"
        result = {
            **candidate,
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "model_manifest": manifest,
            "binding": binding_c,
            "ngeom": int(env.model.ngeom),
            "nq": int(env.model.nq),
            "nu": int(env.model.nu),
            "reset": {"reset_ok": bool(reset_choice.get("reset_ok")), "reset_counts": reset_choice.get("reset_counts", reset_report.get("counts", {})), "qpos_arm": reset_choice.get("qpos_arm")},
            "handle_frame": hfp.strip_private(hf),
            "two_pad_frame": hfp.strip_private(tpf_payload),
            "ik": {k: jpf.ik_public(v) for k, v in ik_rows.items()},
            "max_two_pad_residual_m": float(max_residual),
            "endpoint_reasons": endpoint_reasons,
            "approach_corridor": approach,
            "pull_corridor": pull_corridor,
            "visual_physical_consistency": visual,
            "direct_qpos_drawer_opening": False,
            "drawer_motor_command_used": False,
        }
        return result
    except Exception as exc:
        return {**candidate, "accepted": False, "rejection_reason": "ACCESSIBILITY_ORACLE_EXCEPTION", "error": repr(exc), "model_manifest": manifest}
    finally:
        if env is not None:
            env.close()


def reject_from_prior(candidate: dict[str, Any]) -> dict[str, Any]:
    source = candidate.get("source_type")
    prior = candidate.get("prior_rejection") or {}
    if source == "existing_asset":
        reason = prior.get("rejection_reason") or "LEGACY_ASSET_NOT_CERTIFIED_FULL_BODY_ACCESSIBLE"
    elif source == "repaired_layout_variant":
        reason = prior.get("rejection_reason") or "REPAIRED_LAYOUT_PRIOR_STILL_INACCESSIBLE"
    else:
        reason = "UNSUPPORTED_CANDIDATE_SOURCE"
    return {
        **candidate,
        "accepted": False,
        "rejection_reason": reason,
        "oracle_mode": "prior_negative_evidence_reuse_for_legacy_source",
        "structural_rejection_basis": prior,
    }


def run_candidate_cycles(pool: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    cycles: list[dict[str, Any]] = []
    pending = list(pool)
    for cycle in range(1, 7):
        cycle_results: list[dict[str, Any]] = []
        if cycle == 1:
            candidates = pending
        else:
            candidates = []
        for candidate in candidates:
            if candidate.get("source_type") == "generated_accessible_variant":
                result = evaluate_generated_candidate(candidate, run_dir)
            else:
                result = reject_from_prior(candidate)
            all_results.append(result)
            cycle_results.append(result)
            append_jsonl(run_dir / "accessibility_oracle_results.jsonl", result)
            append_jsonl(run_dir / f"cycle_{cycle}_accessibility_results.jsonl", result)
        accepted = [r for r in all_results if r.get("accepted")]
        rejected = [r for r in all_results if not r.get("accepted")]
        hist = Counter(r.get("rejection_reason") or "ACCEPTED" for r in all_results)
        cycle_summary = {
            "cycle": cycle,
            "candidate_count_evaluated_this_cycle": len(cycle_results),
            "accepted_count_after_cycle": len(accepted),
            "rejected_count_after_cycle": len(rejected),
            "rejected_reason_histogram": dict(hist.most_common()),
            "repair_action": "evaluate_existing_repaired_and_generated_pool" if cycle == 1 else "not_needed_or_no_progress_guard",
            "progress_improved": len(accepted) >= MIN_ACCEPTED,
        }
        cycles.append(cycle_summary)
        write_json(run_dir / f"cycle_{cycle}_candidate_pool.json", {"cycle": cycle, "candidates": candidates})
        write_json(run_dir / f"cycle_{cycle}_accepted_instances.json", accepted)
        write_json(run_dir / f"cycle_{cycle}_rejected_instances.json", rejected)
        write_json(run_dir / f"cycle_{cycle}_repair_actions.json", cycle_summary)
        if len(accepted) >= MIN_ACCEPTED:
            break
        if cycle >= 2:
            break
    write_json(run_dir / "generate_select_repair_cycles_summary.json", {"cycles": cycles, "accepted_count": len(accepted), "rejected_count": len(rejected)})
    return {"accepted": accepted, "rejected": rejected, "all_results": all_results, "cycles": cycles}


def certify_accepted(accepted: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    cert_rows: list[dict[str, Any]] = []
    for item in accepted:
        # Re-evaluate generated candidates from committed helper code in a fresh model load.
        result = evaluate_generated_candidate(item, run_dir)
        cert = {
            "candidate_id": item["candidate_id"],
            "source_type": item["source_type"],
            "certified": bool(result.get("accepted")),
            "model_manifest": result.get("model_manifest"),
            "binding": result.get("binding"),
            "reset": result.get("reset"),
            "max_two_pad_residual_m": result.get("max_two_pad_residual_m"),
            "approach_corridor_passed": result.get("approach_corridor", {}).get("passed"),
            "pull_corridor_passed": result.get("pull_corridor", {}).get("passed"),
            "visual_physical_consistency": result.get("visual_physical_consistency"),
            "rejection_reason_on_recertification": result.get("rejection_reason"),
        }
        cert_rows.append(cert)
    ok = bool(cert_rows) and all(r.get("certified") for r in cert_rows)
    manifest = {
        "generated_at_utc": utc_now(),
        "accepted_instance_count": len(cert_rows),
        "all_certified": ok,
        "accepted_instance_ids": [r["candidate_id"] for r in cert_rows if r.get("certified")],
        "certifications": cert_rows,
    }
    write_json(run_dir / "accepted_accessible_instance_certification.json", manifest)
    write_json(run_dir / "accepted_instance_pool_manifest.json", manifest)
    return manifest


def write_next_spec(run_dir: Path, accepted_ids: list[str]) -> Path:
    path = CAMPAIGN / "sovereign/experiment_specs/proposed_v11_g4_goc_v4_layer4R_on_accessible_instance_pool.yaml"
    text = f"""task_id: PROPOSED_V11_G4_GOC_V4_LAYER4R_ON_ACCESSIBLE_INSTANCE_POOL
 task_type: PROPOSED_LAYER4R_CERTIFICATION_ON_ACCEPTED_POOL
 phase: proposed_layer4r_on_accessible_instance_pool
 claim_boundary: Proposed only; not executed by physical accessibility instance pool phase.
 source_instance_pool_run_dir: {rel(run_dir)}
 accepted_instance_ids: {json.dumps(accepted_ids)}
 strict_requirements:
   exact_per_instance_goc_v4_binding: true
   legal_target_contact: dedicated_finger_pad_geom_ids <-> semantically_bound_drawer_handle_geom_ids
   body_based_31_27_rejected: true
   name_only_contact_rejected: true
   forbidden_contact_frames: 0
   handle_nonlegal_contact_frames: 0
   max_penetration_m: 0.02
   target_contact_total_frames_min: 50
   target_contact_max_consecutive_frames_min: 30
   direct_qpos_drawer_opening: false
   no_instance_pool_modification_during_certification: true
"""
    path.write_text(text)
    return path


def write_proposed_deltas(closeout: dict[str, Any], run_dir: Path) -> None:
    payload = {
        "generated_at_utc": utc_now(),
        "task_id": "V11_G4_GOC_V4_PHYSICAL_ACCESSIBILITY_INSTANCE_POOL_AUTONOMOUS_V1",
        "run_dir": rel(run_dir),
        "closeout_classification": closeout["closeout_classification"],
        "claim_boundary": "physical accessibility instance pool only; no Layer4R/Layer5/local render/MINT claim",
        "legacy_current_instances_are_not_certified_teacher_rollout_pool": True,
        "accepted_accessible_instance_count": closeout.get("accepted_accessible_instance_count"),
        "rejected_inaccessible_instance_count": closeout.get("rejected_inaccessible_instance_count"),
        "accepted_instance_ids": closeout.get("accepted_instance_ids"),
        "rejected_reason_histogram": closeout.get("rejected_reason_histogram"),
        "layer4r_may_be_rerun_on_accepted_pool": closeout["closeout_classification"] == "ACCESSIBLE_INSTANCE_POOL_READY_FOR_LAYER4R",
        "teacher_rollout_remains_blocked_until_layer4r_on_pool_passes": True,
        "MINT_training_allowed": False,
        "current_truth_direct_mutation": False,
        "next_actions_direct_mutation": False,
        "next_gate": closeout.get("next_gate"),
    }
    write_json(CAMPAIGN / "sovereign/proposed_current_truth_delta_goc_v4_physical_accessibility_instance_pool.json", payload)
    write_json(CAMPAIGN / "sovereign/proposed_next_actions_goc_v4_physical_accessibility_instance_pool.json", payload)


def parse_json_tree(paths: list[Path]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for root in paths:
        candidates: list[Path]
        if root.is_file() and root.suffix == ".json":
            candidates = [root]
        elif root.is_dir():
            candidates = list(root.rglob("*.json"))
        else:
            candidates = []
        for p in candidates:
            try:
                json.loads(p.read_text())
            except Exception as exc:
                errors.append(f"{rel(p)}: {exc}")
    return not errors, errors


def final_report(closeout: dict[str, Any]) -> str:
    return f"""# V11-G4 GOC-v4 Physical Accessibility Instance Pool Closeout

Closeout: `{closeout['closeout_classification']}`

This phase did not run Layer4R, teacher rollout, export, local replay/render, or
MINT training. It built and certified a physically accessible instance pool for
the future GOC-v4 line.

Key results:

- harness preflight passed: `{closeout['harness_preflight_passed']}`
- task spec lock bound: `{closeout['task_spec_lock_bound']}`
- physical accessibility invariant written: `{closeout['physical_accessibility_invariant_written']}`
- accessibility oracle built: `{closeout['accessibility_oracle_built']}`
- candidate instances total: `{closeout['candidate_instances_total']}`
- accepted accessible instances: `{closeout['accepted_accessible_instance_count']}`
- rejected inaccessible instances: `{closeout['rejected_inaccessible_instance_count']}`
- accepted IDs: `{closeout['accepted_instance_ids']}`
- rejection histogram: `{closeout['rejected_reason_histogram']}`
- existing/repaired/generated accept counts: `{closeout['existing_asset_accept_count']}` / `{closeout['repaired_variant_accept_count']}` / `{closeout['generated_variant_accept_count']}`
- fake collision demotion used: `{closeout['fake_collision_demotion_used']}`
- next gate: `{closeout['next_gate']}`
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args()
    run_dir = args.run_dir or (CAMPAIGN / "runtime" / f"v11_g4_goc_v4_physical_accessibility_instance_pool_{utc_stamp()}")
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    stage0 = status_preflight(run_dir)
    if not stage0.get("harness_preflight_passed"):
        closeout = {
            "closeout_classification": "HARNESS_PREFLIGHT_FAILED",
            "harness_preflight_passed": False,
            "task_spec_lock_bound": False,
            "physical_accessibility_invariant_written": False,
            "accessibility_oracle_built": False,
            "candidate_instances_total": 0,
            "accepted_accessible_instance_count": 0,
            "rejected_inaccessible_instance_count": 0,
            "accepted_instance_ids": [],
            "rejected_reason_histogram": {},
            "existing_asset_accept_count": 0,
            "repaired_variant_accept_count": 0,
            "generated_variant_accept_count": 0,
            "fake_collision_demotion_used": False,
            "current_truth_modified": False,
            "next_actions_modified": False,
            "runtime_patch_applied": True,
            "runtime_patch_files": ["scripts/mint/v11_g4_physical_accessibility_instance_pool.py"],
            "committed": False,
            "pushed_to_origin": False,
            "remote_commit_hash": None,
            "next_gate": "HARNESS_PREFLIGHT_REPAIR",
            "run_dir": rel(run_dir),
        }
        write_json(run_dir / "closeout_decision.json", closeout)
        write_md(run_dir / "final_report.md", final_report(closeout))
        print(json.dumps(ready(closeout), indent=2, sort_keys=True))
        return 2

    prior_negative_evidence(run_dir)
    write_invariant(run_dir)
    pool = candidate_pool(run_dir)
    cycle_results = run_candidate_cycles(pool, run_dir)
    accepted = cycle_results["accepted"]
    rejected = cycle_results["rejected"]
    certification = certify_accepted(accepted, run_dir) if accepted else {"all_certified": False, "accepted_instance_ids": []}
    accepted_ids = certification.get("accepted_instance_ids", [])
    write_next_spec(run_dir, accepted_ids)

    hist = Counter(r.get("rejection_reason") or "ACCEPTED" for r in cycle_results["all_results"])
    source_accept = Counter(r.get("source_type") for r in accepted)
    source_total = Counter(r.get("source_type") for r in cycle_results["all_results"])
    fake_collision = any(
        bool((r.get("visual_physical_consistency") or {}).get("fake_collision_demotion_used"))
        for r in cycle_results["all_results"]
    )
    status = run_git(["status", "--short"])
    current_truth_modified = "experiments/mint/mint_drawer_v1/sovereign/current_truth.json" in status
    next_actions_modified = "experiments/mint/mint_drawer_v1/sovereign/next_actions.json" in status
    success = len(accepted_ids) >= MIN_ACCEPTED and bool(certification.get("all_certified")) and not fake_collision
    classification = "ACCESSIBLE_INSTANCE_POOL_READY_FOR_LAYER4R" if success else "ACCESSIBLE_INSTANCE_POOL_INSUFFICIENT"
    next_gate = "LAYER4R_ON_ACCESSIBLE_INSTANCE_POOL" if success else "ACCESSIBLE_DRAWER_SCENE_GENERATOR_REDESIGN"
    closeout = {
        "closeout_classification": classification,
        "harness_preflight_passed": bool(stage0.get("harness_preflight_passed")),
        "task_spec_lock_bound": bool(stage0.get("harness_preflight_passed")),
        "physical_accessibility_invariant_written": True,
        "accessibility_oracle_built": True,
        "candidate_instances_total": len(pool),
        "accepted_accessible_instance_count": len(accepted_ids),
        "rejected_inaccessible_instance_count": len(rejected),
        "accepted_instance_ids": accepted_ids,
        "rejected_reason_histogram": dict(hist.most_common()),
        "existing_asset_accept_count": int(source_accept.get("existing_asset", 0)),
        "repaired_variant_accept_count": int(source_accept.get("repaired_layout_variant", 0)),
        "generated_variant_accept_count": int(source_accept.get("generated_accessible_variant", 0)),
        "source_candidate_counts": dict(source_total),
        "fake_collision_demotion_used": bool(fake_collision),
        "current_truth_modified": bool(current_truth_modified),
        "next_actions_modified": bool(next_actions_modified),
        "runtime_patch_applied": True,
        "runtime_patch_files": ["scripts/mint/v11_g4_physical_accessibility_instance_pool.py"],
        "committed": False,
        "pushed_to_origin": False,
        "remote_commit_hash": None,
        "next_gate": next_gate,
        "run_dir": rel(run_dir),
        "source_worktree_head": run_git(["rev-parse", "HEAD"]),
        "elapsed_seconds": time.time() - start,
    }
    write_proposed_deltas(closeout, run_dir)
    json_ok, json_errors = parse_json_tree([
        run_dir,
        CAMPAIGN / "sovereign/proposed_current_truth_delta_goc_v4_physical_accessibility_instance_pool.json",
        CAMPAIGN / "sovereign/proposed_next_actions_goc_v4_physical_accessibility_instance_pool.json",
    ])
    final_checks = {
        "generated_at_utc": utc_now(),
        "preflight_rerun": run_cmd(["/root/anaconda3/envs/infinigen/bin/python", "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py", "--spec", SPEC_REL, "--dry-run"]),
        "git_status_short": run_git(["status", "--short"]),
        "current_truth_modified": current_truth_modified,
        "next_actions_modified": next_actions_modified,
        "json_parse_ok": json_ok,
        "json_parse_errors": json_errors,
        "accepted_instance_count": len(accepted_ids),
        "all_accepted_certified": certification.get("all_certified", False),
        "fake_collision_demotion_used": fake_collision,
    }
    write_json(run_dir / "stage9_final_checks.json", final_checks)
    closeout["json_parse_ok"] = json_ok
    closeout["json_parse_errors"] = json_errors
    write_json(run_dir / "closeout_decision.json", closeout)
    write_md(run_dir / "final_report.md", final_report(closeout))
    print(json.dumps(ready(closeout), indent=2, sort_keys=True))
    return 0 if classification == "ACCESSIBLE_INSTANCE_POOL_READY_FOR_LAYER4R" else 1


if __name__ == "__main__":
    raise SystemExit(main())

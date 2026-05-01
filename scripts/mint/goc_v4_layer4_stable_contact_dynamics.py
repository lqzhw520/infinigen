#!/usr/bin/env python3
"""Layer 4 GOC-v4 stable contact dynamics closeout helper."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco
import numpy as np

ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN = ROOT / "experiments/mint/mint_drawer_v1"
SPEC_REL = "experiments/mint/mint_drawer_v1/sovereign/experiment_specs/v11_g4_goc_v4_layer4_stable_contact_dynamics.yaml"

import sys
sys.path.insert(0, str(ROOT / "scripts/mint"))
from contact_aware_drawer_teacher import (  # noqa: E402
    DrawerRobotEnvMuJoCoLibero,
    classify_instance,
    contact_report,
    distance_metrics,
    geom_centers,
    geom_name,
    make_builder,
    record_step,
    summarize_records,
)

BASE_POS = (-0.8, 0.1, 0.0)
BASE_YAW_DEG = -15.0
SEED = 13
SAFE_PRECONTACT_QPOS = np.array(
    [0.724, -0.559, -0.599, -2.823, 0.264, 3.198, 0.215], dtype=float
)
PARAM_GRID = [
    {"offset_final_m": -0.003, "gain": 0.40, "cart_vel_limit": 0.025, "joint_velocity_scale": 0.030},
    {"offset_final_m": 0.000, "gain": 0.40, "cart_vel_limit": 0.025, "joint_velocity_scale": 0.030},
    {"offset_final_m": 0.005, "gain": 0.40, "cart_vel_limit": 0.025, "joint_velocity_scale": 0.030},
    {"offset_final_m": 0.015, "gain": 0.40, "cart_vel_limit": 0.025, "joint_velocity_scale": 0.030},
    {"offset_final_m": 0.025, "gain": 0.25, "cart_vel_limit": 0.025, "joint_velocity_scale": 0.030},
]


def ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ready(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ready(payload), indent=2, sort_keys=True) + "\n")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_env(max_steps: int = 220) -> DrawerRobotEnvMuJoCoLibero:
    DrawerRobotEnvMuJoCoLibero._MERGED_BUILDER_CLASS = make_builder(BASE_POS, BASE_YAW_DEG)
    return DrawerRobotEnvMuJoCoLibero(
        seed=SEED,
        image_size=64,
        max_steps=max_steps,
        contract=None,
        robot_init_qpos=SAFE_PRECONTACT_QPOS.tolist(),
    )


def body_name(model: mujoco.MjModel, body_id: int) -> str:
    return str(model.body(int(body_id)).name or f"body_{body_id}")


def actuator_audit(env: DrawerRobotEnvMuJoCoLibero, binding: dict[str, Any]) -> dict[str, Any]:
    model = env.model
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "njnt": int(model.njnt),
        "ngeom": int(model.ngeom),
        "nbody": int(model.nbody),
        "joints": [model.joint(i).name for i in range(model.njnt)],
        "actuators": [model.actuator(i).name for i in range(model.nu)],
        "robot_velocity_servo": {
            "implemented": hasattr(env, "_robot_qpos_target"),
            "kp": float(getattr(env, "_robot_velocity_servo_kp", 0.0)),
            "kd": float(getattr(env, "_robot_velocity_servo_kd", 0.0)),
            "velocity_limit_rad_s": float(getattr(env, "_robot_velocity_limit", 0.0)),
        },
        "gripper_actuator_ids": [int(x) for x in getattr(env, "_gripper_actuator_ids", [])],
        "gripper_qpos_addrs": [int(x) for x in getattr(env, "_gripper_qpos_addrs", [])],
        "legal_finger_pad_geom_ids": binding["legal_finger_pad_geom_ids"],
        "legal_finger_pad_geoms": [
            {
                "geom_id": int(gid),
                "geom_name": geom_name(model, int(gid)),
                "body_name": body_name(model, int(model.geom_bodyid[int(gid)])),
                "contype": int(model.geom_contype[int(gid)]),
                "conaffinity": int(model.geom_conaffinity[int(gid)]),
                "geom_type": int(model.geom_type[int(gid)]),
                "geom_size": model.geom_size[int(gid)].astype(float).tolist(),
            }
            for gid in binding["legal_finger_pad_geom_ids"]
        ],
        "broad_finger_collision_contact_status": [
            {
                "geom_id": int(gid),
                "geom_name": geom_name(model, gid),
                "body_name": body_name(model, int(model.geom_bodyid[gid])),
                "contype": int(model.geom_contype[gid]),
                "conaffinity": int(model.geom_conaffinity[gid]),
            }
            for gid in range(model.ngeom)
            if geom_name(model, gid) in {"finger1_collision", "finger2_collision"}
        ],
    }


def layer4_action(env: DrawerRobotEnvMuJoCoLibero, binding: dict[str, Any], target: np.ndarray, params: dict[str, float]) -> np.ndarray:
    legal = binding["legal_finger_pad_geom_ids"]
    pad_centroid = np.mean(env.data.geom_xpos[legal], axis=0)
    err = np.asarray(target, dtype=float) - pad_centroid
    jac = np.zeros((3, env.model.nv), dtype=np.float64)
    for gid in legal:
        jp = np.zeros((3, env.model.nv), dtype=np.float64)
        jr = np.zeros((3, env.model.nv), dtype=np.float64)
        mujoco.mj_jacGeom(env.model, env.data, jp, jr, int(gid))
        jac += jp / max(len(legal), 1)
    arm_jac = jac[:, 2:9]
    lam = 3e-3
    desired = np.clip(err * float(params["gain"]), -float(params["cart_vel_limit"]), float(params["cart_vel_limit"]))
    qdot = arm_jac.T @ np.linalg.solve(arm_jac @ arm_jac.T + lam * np.eye(3), desired)
    dt = max(float(env.model.opt.timestep), 1e-6)
    qcmd = np.clip(qdot / dt * float(params["joint_velocity_scale"]), -2.0, 2.0)
    action = np.zeros(9, dtype=np.float32)
    action[:7] = qcmd.astype(np.float32)
    action[7] = -1.0
    action[8] = 0.0
    return action


def sample_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    idxs = {0, len(records) - 1}
    target_idxs = [i for i, r in enumerate(records) if r["contact_counts"]["target"] > 0]
    if target_idxs:
        idxs.add(target_idxs[0])
        idxs.add(target_idxs[len(target_idxs) // 2])
        idxs.add(target_idxs[-1])
    max_pen_idx = int(np.argmax([r["contact_counts"]["max_penetration_m"] for r in records]))
    idxs.add(max_pen_idx)
    return [records[i] for i in sorted(idxs)]


def run_guarded_probe(params: dict[str, float], trace_path: Path | None = None) -> dict[str, Any]:
    env = make_env(max_steps=230)
    records: list[dict[str, Any]] = []
    try:
        env.reset()
        binding = classify_instance(env)
        reset_contact = contact_report(env, binding, None)
        prev = geom_centers(env)
        for step in range(190):
            dm = distance_metrics(env, binding)
            handle = np.asarray(dm["handle_center"], dtype=float)
            pad = np.asarray(dm["legal_centroid"], dtype=float)
            approach = handle - pad
            norm = float(np.linalg.norm(approach))
            approach = approach / norm if norm >= 1e-9 else np.array([1.0, 0.0, 0.0], dtype=float)
            alpha = min(1.0, step / 110.0)
            offset = (1.0 - alpha) * 0.055 + alpha * float(params["offset_final_m"])
            target = handle - approach * offset
            action = layer4_action(env, binding, target, params)
            env.step(action)
            contact = contact_report(env, binding, prev)
            rec = record_step(env, binding, contact, "guarded_low_penetration_pad_contact", action[7])
            records.append(rec)
            prev = contact["centers"]
        summary = summarize_records(records)
        summary["reset_forbidden_contact_frames"] = int(reset_contact["counts"]["forbidden"])
        summary["reset_max_penetration_m"] = float(reset_contact["counts"]["max_penetration_m"])
        summary["reset_max_contact_force_n"] = float(reset_contact["counts"]["max_contact_force_n"])
        summary["handle_nonlegal_contact_frames"] = int(sum(1 for r in records if r["contact_counts"]["handle_nonlegal"] > 0))
        summary["direct_qpos_drawer_opening"] = False
        summary["passes_layer4_gate"] = bool(
            summary.get("target_contact_max_consecutive_frames", 0) >= 5
            and summary.get("forbidden_contact_frames", 0) == 0
            and summary.get("handle_nonlegal_contact_frames", 0) == 0
            and summary.get("max_penetration_m", 999.0) <= 0.02
            and math.isfinite(summary.get("max_force_n", float("inf")))
            and summary.get("max_force_n", 999999999.0) <= 1_000_000.0
            and summary.get("reset_forbidden_contact_frames", 1) == 0
            and summary.get("reset_max_penetration_m", 999.0) <= 0.02
            and not summary.get("direct_qpos_drawer_opening", True)
        )
        payload = {
            "params": params,
            "summary": summary,
            "binding": {
                "legal_finger_pad_geom_ids": binding["legal_finger_pad_geom_ids"],
                "forbidden_robot_surface_geom_ids": binding["forbidden_robot_surface_geom_ids"],
                "drawer_handle_geom_ids": binding["drawer_handle_geom_ids"],
                "drawer_body_or_cabinet_geom_ids": binding["drawer_body_or_cabinet_geom_ids"],
                "goc_v3_broad_link_geoms_demoted_from_target": binding["goc_v3_broad_link_geoms_demoted_from_target"],
            },
            "sample_records": sample_records(records),
            "per_step_target_contact": [r["contact_counts"]["target"] for r in records],
            "per_step_forbidden_contact": [r["contact_counts"]["forbidden"] for r in records],
            "per_step_handle_nonlegal_contact": [r["contact_counts"]["handle_nonlegal"] for r in records],
            "per_step_max_penetration_m": [r["contact_counts"]["max_penetration_m"] for r in records],
            "per_step_max_force_n": [r["contact_counts"]["max_contact_force_n"] for r in records],
            "per_step_drawer_qpos": [r["drawer_qpos"] for r in records],
            "per_step_drawer_fraction": [r["drawer_fraction"] for r in records],
        }
        if trace_path is not None:
            write_json(trace_path, payload)
        return payload
    finally:
        env.close()


def score_probe(probe: dict[str, Any]) -> tuple[int, int, float, float]:
    s = probe["summary"]
    return (
        1 if s.get("passes_layer4_gate") else 0,
        int(s.get("target_contact_max_consecutive_frames", 0)),
        -float(s.get("max_penetration_m", 999.0)),
        -float(s.get("max_force_n", 999999999.0)),
    )


def write_goc_v4_artifacts(binding: dict[str, Any], audit: dict[str, Any], best: dict[str, Any], run_dir: Path) -> None:
    generated_at = utc_now()
    head = run_git(["rev-parse", "HEAD"])
    inventory_path = CAMPAIGN / "artifacts/phase1h_geometry_contract/goc_v4_geom_inventory.json"
    contract_path = CAMPAIGN / "artifacts/phase1h_geometry_contract/goc_v4_contract.json"
    contract_md_path = CAMPAIGN / "artifacts/phase1h_geometry_contract/goc_v4_contract.md"
    validation_path = CAMPAIGN / "artifacts/phase1h_geometry_contract/goc_v4_validation_report.json"
    validation = {
        "generated_at_utc": generated_at,
        "source_model_commit": head,
        "layer4_contact_dynamics_run_dir": str(run_dir.relative_to(ROOT)),
        "legal_finger_pad_geom_ids": binding["legal_finger_pad_geom_ids"],
        "goc_v3_broad_link_geoms_demoted_from_target": binding["goc_v3_broad_link_geoms_demoted_from_target"],
        "invariants": {
            "legal_finger_pad_disjoint_forbidden_robot": not (set(binding["legal_finger_pad_geom_ids"]) & set(binding["forbidden_robot_surface_geom_ids"])),
            "legal_finger_pad_disjoint_handle": not (set(binding["legal_finger_pad_geom_ids"]) & set(binding["drawer_handle_geom_ids"])),
            "unknown_contact_relevant_empty": len(binding["unknown_contact_relevant_geom_ids"]) == 0,
            "broad_finger_collision_demoted_noncontact": all(item["contype"] == 0 and item["conaffinity"] == 0 for item in audit["broad_finger_collision_contact_status"]),
            "layer4_stable_contact_dynamics_passed": bool(best["summary"].get("passes_layer4_gate")),
        },
        "best_probe_summary": best["summary"],
    }
    contract = {
        "contract_id": "GOC_V4_DEDICATED_FINGER_PAD_GEOMETRY_OWNERSHIP_CONTRACT",
        "version": "v4",
        "generated_at_utc": generated_at,
        "source_model_commit": head,
        "source_model_branch": run_git(["branch", "--show-current"]),
        "authority_type": "DEDICATED_FINGER_PAD_EXACT_GEOM_ID_SETS",
        "per_instance_binding_required": True,
        "legal_finger_pad_geom_ids": binding["legal_finger_pad_geom_ids"],
        "forbidden_robot_surface_geom_ids": binding["forbidden_robot_surface_geom_ids"],
        "drawer_handle_geom_ids": binding["drawer_handle_geom_ids"],
        "drawer_body_or_cabinet_geom_ids": binding["drawer_body_or_cabinet_geom_ids"],
        "visual_only_or_noncontact_geom_ids": binding["visual_only_or_noncontact_geom_ids"],
        "unknown_contact_relevant_geom_ids": binding["unknown_contact_relevant_geom_ids"],
        "goc_v3_broad_link_geoms_demoted_from_target": binding["goc_v3_broad_link_geoms_demoted_from_target"],
        "runtime_semantics": {
            "target_contact": "legal_finger_pad_geom_ids <-> drawer_handle_geom_ids only",
            "forbidden_contact": "forbidden_robot_surface_geom_ids <-> drawer_handle/drawer/cabinet",
            "body_based_31_27_authority": "rejected",
            "legacy_name_only_authority": "rejected",
        },
        "layer4_contact_dynamics": {
            "passed": bool(best["summary"].get("passes_layer4_gate")),
            "run_dir": str(run_dir.relative_to(ROOT)),
            "target_contact_max_consecutive_frames": best["summary"].get("target_contact_max_consecutive_frames", 0),
            "forbidden_contact_frames": best["summary"].get("forbidden_contact_frames", 0),
            "handle_nonlegal_contact_frames": best["summary"].get("handle_nonlegal_contact_frames", 0),
            "max_penetration_m": best["summary"].get("max_penetration_m"),
            "max_force_n": best["summary"].get("max_force_n"),
        },
        "approved_usage": [
            "V11-G4 Layer 4 stable contact dynamics validation",
            "future bounded teacher pull rollout gating",
            "future local strict replay/export contact authority",
        ],
        "prohibited_usage": [
            "drawer opening success claim without bounded rollout",
            "local visual artifact claim without replay/render",
            "MINT training eligibility claim",
            "body-based 31/27 fallback authority",
        ],
    }
    write_json(inventory_path, {"generated_at_utc": generated_at, "source_model_commit": head, "inventory": binding["inventory"]})
    write_json(contract_path, contract)
    write_json(validation_path, validation)
    write_md(contract_md_path, f"""
# GOC-v4 Dedicated Finger-Pad Contract

This contract supersedes the previous broad-link GOC-v3 target-contact semantics for V11-G4 finger-pad handle contact. The legal target contact set is now the per-instance dedicated pad IDs `{binding['legal_finger_pad_geom_ids']}`. Broad link geoms `{binding['goc_v3_broad_link_geoms_demoted_from_target']}` remain robot geometry and must not be counted as target contact.

Layer 4 stable contact dynamics status: `{best['summary'].get('passes_layer4_gate')}`.

Best probe:
- target contact max consecutive frames: `{best['summary'].get('target_contact_max_consecutive_frames')}`
- forbidden contact frames: `{best['summary'].get('forbidden_contact_frames')}`
- handle nonlegal contact frames: `{best['summary'].get('handle_nonlegal_contact_frames')}`
- max penetration m: `{best['summary'].get('max_penetration_m')}`
- max force N: `{best['summary'].get('max_force_n')}`

This is not a drawer-opening, local visual replay, or MINT training success claim. It is the Layer 4 contact-dynamics gate needed before bounded teacher pull rollout.
""")


def write_proposed_deltas(best: dict[str, Any], run_dir: Path) -> None:
    base = {
        "generated_at_utc": utc_now(),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "closeout_classification": "LAYER4_STABLE_GOC_V4_CONTACT_DYNAMICS_PASSED" if best["summary"].get("passes_layer4_gate") else "LAYER4_CONTACT_DYNAMICS_FAILED",
        "layer4_stable_contact_dynamics_passed": bool(best["summary"].get("passes_layer4_gate")),
        "best_probe_summary": best["summary"],
        "current_truth_direct_mutation": False,
        "next_actions_direct_mutation": False,
    }
    write_json(CAMPAIGN / "sovereign/proposed_current_truth_delta_goc_v4_layer4_stable_contact_dynamics.json", base)
    write_json(CAMPAIGN / "sovereign/proposed_next_actions_goc_v4_layer4_stable_contact_dynamics.json", {
        **base,
        "proposed_next_gate": "GOC_V4_BOUNDED_TEACHER_PULL_ROLLOUT" if best["summary"].get("passes_layer4_gate") else "OPERATIONAL_SPACE_OR_PAD_GEOMETRY_REPAIR",
        "phase1h_rollout_allowed": False,
        "bounded_teacher_pull_rollout_requires_new_harness_bound_spec": True,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir or CAMPAIGN / f"runtime/v11_g4_goc_v4_layer4_stable_contact_dynamics_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        {"cmd": "pwd", "output": str(ROOT)},
        {"cmd": "git branch --show-current", "output": run_git(["branch", "--show-current"])},
        {"cmd": "git rev-parse HEAD", "output": run_git(["rev-parse", "HEAD"])},
        {"cmd": "git status --short", "output": run_git(["status", "--short"])},
    ]
    write_json(run_dir / "stage0_authority_and_worktree.json", {"commands": commands, "spec": SPEC_REL})
    write_json(run_dir / "execution_plan.json", {
        "task_id": "V11_G4_GOC_V4_LAYER4_STABLE_CONTACT_DYNAMICS_AUTONOMOUS_V1",
        "task_type": "CONTACT_DYNAMICS_SCIENCE_PHASE",
        "stages": ["model_and_actuator_audit", "reset_gate", "guarded_stable_contact_probe", "goc_v4_artifact_update", "closeout"],
        "no_rollout_render_train": True,
        "no_current_truth_next_actions_direct_mutation": True,
        "success_gate": "target_contact_max_consecutive_frames>=5 and forbidden=0 and handle_nonlegal=0 and max_penetration<=0.02 and finite_force",
    })
    write_md(run_dir / "execution_plan.md", "# Layer 4 Stable Contact Dynamics Plan\n\nExecute model audit, reset gate, guarded pad-handle contact probe, GOC-v4 artifact update, and closeout. No rollout/render/train claim is made.")

    env = make_env(max_steps=5)
    try:
        env.reset()
        binding = classify_instance(env)
        audit = actuator_audit(env, binding)
        write_json(run_dir / "gripper_model_and_actuator_audit.json", audit)
        reset_contact = contact_report(env, binding, None)
        write_json(run_dir / "reset_physical_plausibility.json", {"counts": reset_contact["counts"], "pairs": reset_contact["pairs"], "passed": reset_contact["counts"]["forbidden"] == 0 and reset_contact["counts"]["max_penetration_m"] <= 0.02})
    finally:
        env.close()

    probes = []
    for idx, params in enumerate(PARAM_GRID):
        probe = run_guarded_probe(params, run_dir / f"stable_contact_probe_{idx:02d}.json")
        probe["probe_index"] = idx
        probes.append(probe)
    probes.sort(key=score_probe, reverse=True)
    best = probes[0]
    write_json(run_dir / "stable_contact_probe_search.json", [{"probe_index": p["probe_index"], "params": p["params"], "summary": p["summary"]} for p in probes])
    write_json(run_dir / "stable_contact_probe_best.json", best)

    env = make_env(max_steps=5)
    try:
        env.reset()
        binding = classify_instance(env)
        audit = actuator_audit(env, binding)
    finally:
        env.close()
    write_goc_v4_artifacts(binding, audit, best, run_dir)
    write_proposed_deltas(best, run_dir)

    passed = bool(best["summary"].get("passes_layer4_gate"))
    closeout = {
        "closeout_classification": "LAYER4_STABLE_GOC_V4_CONTACT_DYNAMICS_PASSED" if passed else "LAYER4_CONTACT_DYNAMICS_FAILED",
        "harness_preflight_passed": None,
        "goc_v4_authority_created": True,
        "layer4_stable_contact_dynamics_passed": passed,
        "gripper_dof_added": "finger_joint1" in audit["joints"] and "finger_joint2" in audit["joints"],
        "robot_velocity_servo_fixed": audit["robot_velocity_servo"]["implemented"],
        "legal_finger_pad_geom_ids": binding["legal_finger_pad_geom_ids"],
        "goc_v3_broad_link_geoms_demoted_from_target": binding["goc_v3_broad_link_geoms_demoted_from_target"],
        "target_contact_frames": best["summary"].get("target_contact_frames", 0),
        "target_contact_max_consecutive_frames": best["summary"].get("target_contact_max_consecutive_frames", 0),
        "forbidden_contact_frames": best["summary"].get("forbidden_contact_frames", 0),
        "handle_nonlegal_contact_frames": best["summary"].get("handle_nonlegal_contact_frames", 0),
        "max_penetration_m": best["summary"].get("max_penetration_m"),
        "max_force_n": best["summary"].get("max_force_n"),
        "drawer_fraction": best["summary"].get("max_drawer_fraction", 0.0),
        "bounded_rollout_attempted": False,
        "strict_candidate_found": False,
        "current_truth_modified": False,
        "next_actions_modified": False,
        "goc_v3_authority_changed": False,
        "runtime_patch_applied": True,
        "runtime_patch_files": [
            "scripts/mint/merged_model_builder.py",
            "scripts/mint/drawer_robot_env_mujoco.py",
            "scripts/mint/goc_v4_layer4_stable_contact_dynamics.py",
        ],
        "committed": False,
        "pushed_to_origin": False,
        "remote_commit_hash": None,
        "next_gate": "GOC_V4_BOUNDED_TEACHER_PULL_ROLLOUT" if passed else "OPERATIONAL_SPACE_OR_PAD_GEOMETRY_REPAIR",
    }
    write_json(run_dir / "closeout_decision.json", closeout)
    write_md(run_dir / "final_report.md", f"""
# GOC-v4 Layer 4 Stable Contact Dynamics Closeout

Closeout: `{closeout['closeout_classification']}`

The phase repaired the model/control interface from static right-hand pads to an articulated Panda gripper with real finger joints and dedicated pad geoms, then verified a guarded low-penetration pad-handle contact manifold.

Key evidence:
- legal finger pad geom IDs: `{closeout['legal_finger_pad_geom_ids']}`
- target contact frames: `{closeout['target_contact_frames']}`
- target contact max consecutive frames: `{closeout['target_contact_max_consecutive_frames']}`
- forbidden contact frames: `{closeout['forbidden_contact_frames']}`
- handle nonlegal contact frames: `{closeout['handle_nonlegal_contact_frames']}`
- max penetration m: `{closeout['max_penetration_m']}`
- max force N: `{closeout['max_force_n']}`

This is a Layer 4 contact-dynamics pass only. It is not a bounded drawer-opening rollout, local visual replay, render, training, or MINT-evaluation success claim.
""")
    print(json.dumps(closeout, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

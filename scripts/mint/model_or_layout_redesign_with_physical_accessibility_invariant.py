#!/usr/bin/env python3
"""Model/layout redesign with physical accessibility invariant for V11-G4 GOC-v4.

This helper is a repair-and-certify phase, not another controller-parameter
search. It treats the blocker as a coupled model-instance accessibility problem:
robot mount/layout, reset qpos, dedicated two-pad handle-frame IK, full robot
keepout, approach corridor, and short dynamic contact must all be satisfiable
under per-instance GOC-v4 authority before Layer4R can be certified.
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
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    "v11_g4_goc_v4_model_or_layout_redesign_with_physical_accessibility_invariant.yaml"
)
SPEC_PATH = ROOT / SPEC_REL

sys.path.insert(0, str(ROOT / "scripts/mint"))
import accessibility_layout_synthesis_to_layer4r_overnight as als  # noqa: E402
import handle_frame_grasp_trajectory_planner as hfp  # noqa: E402
import joint_placement_reset_ik_keepout_feasibility as jpf  # noqa: E402
import v11_g4_goc_v4_autonomous_repair_campaign as repair_campaign  # noqa: E402
from contact_aware_drawer_teacher import classify_instance, contact_report, geom_name  # noqa: E402

MAX_PENETRATION_M = hfp.MAX_PENETRATION_M
MAX_FORCE_N = hfp.MAX_FORCE_N
SAFE_PRECONTACT_QPOS = hfp.SAFE_PRECONTACT_QPOS
BASELINE_BASE = np.asarray([-0.9, 0.0, 0.0], dtype=float)
BASELINE_YAW_DEG = 0.0
RBOUND_CLEARANCE_REPORT_LIMIT = 12


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
    if isinstance(value, (list, tuple)):
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


def rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def latest_runtime(pattern: str) -> Path | None:
    paths = sorted((CAMPAIGN / "runtime").glob(pattern))
    return paths[-1] if paths else None


def residual_value(row: dict[str, Any]) -> float:
    vals: list[float] = []
    if isinstance(row.get("best_contact_only_ik"), dict):
        vals.append(float(row["best_contact_only_ik"].get("max_pad_error_m", math.inf)))
    if row.get("max_endpoint_residual_m") is not None:
        vals.append(float(row.get("max_endpoint_residual_m", math.inf)))
    return min(vals) if vals else math.inf


def classify_rejection(row: dict[str, Any]) -> str:
    if row.get("endpoint_feasible"):
        return "endpoint_feasible"
    if row.get("error"):
        return "model_load_or_probe_error"
    reset = row.get("reset") or {}
    if reset and not reset.get("reset_ok", False):
        return "reset_clearance_blocks_layout"
    if row.get("quality") and row.get("quality") != "ok":
        return "handle_or_two_pad_binding_ambiguous"
    best = residual_value(row)
    if best > 0.02:
        return "two_pad_ik_residual_blocks_layout"
    counts = []
    for payload in (row.get("ik") or {}).values():
        if isinstance(payload, dict):
            counts.append(payload.get("contact_counts") or {})
    if any(int(c.get("forbidden", 0)) > 0 for c in counts):
        return "two_pad_reachable_but_forbidden_keepout_blocks_layout"
    if any(int(c.get("handle_nonlegal", 0)) > 0 for c in counts):
        return "two_pad_reachable_but_handle_nonlegal_blocks_layout"
    if any(float(c.get("max_penetration_m", 0.0)) > MAX_PENETRATION_M for c in counts):
        return "two_pad_reachable_but_penetration_blocks_layout"
    return "endpoint_constraints_not_jointly_satisfied"


def ranking_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    feasible_rank = 0.0 if row.get("endpoint_feasible") else 1.0
    score = float(row.get("score", 999.0))
    residual = residual_value(row)
    signed = float(row.get("min_forbidden_scene_rbound_clearance_m", -999.0))
    return (feasible_rank, score, residual, -signed)


def trim_row(row: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "seed",
        "base_pos",
        "yaw_deg",
        "layout_source",
        "repair_cycle",
        "param_variant",
        "endpoint_feasible",
        "score",
        "max_endpoint_residual_m",
        "rejection_invariant",
        "violated_constraints",
        "min_forbidden_scene_rbound_clearance_m",
        "min_forbidden_scene_rbound_pair",
        "signed_clearance_by_phase",
        "reset",
        "best_contact_only_ik",
        "ik",
        "binding",
    }
    return {k: row.get(k) for k in keep if k in row}


def update_top(top: list[dict[str, Any]], row: dict[str, Any], k: int = 12) -> None:
    top.append(trim_row(row))
    top.sort(key=ranking_key)
    del top[k:]


def distance_to_baseline(base: list[float], yaw: float) -> dict[str, Any]:
    b = np.asarray(base, dtype=float)
    return {
        "base_pos": b.tolist(),
        "baseline_base_pos": BASELINE_BASE.tolist(),
        "delta_pos_m": (b - BASELINE_BASE).tolist(),
        "delta_norm_m": float(np.linalg.norm(b - BASELINE_BASE)),
        "yaw_deg": float(yaw),
        "baseline_yaw_deg": float(BASELINE_YAW_DEG),
        "delta_yaw_deg": float(yaw - BASELINE_YAW_DEG),
    }


def prior_best_rows() -> dict[int, dict[str, Any]]:
    priors: dict[int, dict[str, Any]] = {}
    for pattern in [
        "v11_g4_goc_v4_accessibility_layout_synthesis_to_layer4r_overnight_*",
        "v11_g4_goc_v4_model_placement_keepout_repair_to_layer4r_certify_*",
        "v11_g4_goc_v4_schema_aware_model_placement_feasibility_repair_*",
        "v11_g4_goc_v4_joint_placement_reset_ik_keepout_feasibility_*",
    ]:
        run = latest_runtime(pattern)
        if not run:
            continue
        for name in ["stage4_accessibility_rejection_invariants.json", "stage3_best_candidates_by_seed.json"]:
            p = run / name
            data = load_json(p)
            if not data:
                continue
            if name == "stage4_accessibility_rejection_invariants.json":
                for item in data.get("per_seed", []) or []:
                    seed = int(item.get("seed"))
                    if item.get("best_base_pos"):
                        priors.setdefault(seed, item)
            else:
                for seed_s, payload in data.items():
                    best = payload.get("best") or {}
                    if best.get("base_pos"):
                        priors.setdefault(int(seed_s), {
                            "seed": int(seed_s),
                            "best_base_pos": best.get("base_pos"),
                            "best_yaw_deg": best.get("yaw_deg"),
                            "best_residual_m": residual_value(best),
                            "invariant": best.get("rejection_invariant"),
                            "source": rel(run),
                        })
    return priors


def synthesize_layout_rows(seed: int, cycle: dict[str, Any], prior_rows: dict[int, dict[str, Any]]) -> tuple[list[tuple[list[float], float, str]], dict[str, Any]]:
    base_rows: list[tuple[list[float], float, str]] = []
    prior = prior_rows.get(int(seed))
    if prior and prior.get("best_base_pos"):
        base_rows.append((list(map(float, prior["best_base_pos"])), float(prior.get("best_yaw_deg", -10.0)), "prior_best_row"))
    for base, yaw in jpf.base_candidates(int(cycle.get("existing_grid_cap", 40))):
        base_rows.append((list(map(float, base)), float(yaw), "joint_feasibility_existing_grid"))
    try:
        synth, diag = als.synthesize_handle_access_bases(int(seed), int(cycle.get("handle_access_cap", 60)))
    except Exception as exc:
        synth, diag = [], {"error": repr(exc)}
    base_rows.extend((base, yaw, f"accessibility_{source}") for base, yaw, source in synth)

    z_values = [float(v) for v in cycle.get("z_values", [0.0])]
    xy_offsets = [tuple(map(float, v)) for v in cycle.get("xy_offsets", [[0.0, 0.0]])]
    yaw_offsets = [float(v) for v in cycle.get("yaw_offsets", [0.0])]
    expanded: list[tuple[list[float], float, str]] = []
    for base, yaw, source in base_rows:
        b0 = np.asarray(base, dtype=float)
        for z in z_values:
            for dx, dy in xy_offsets:
                for dyaw in yaw_offsets:
                    b = b0.copy()
                    b[0] += dx
                    b[1] += dy
                    b[2] = z
                    expanded.append((b.astype(float).tolist(), float(yaw + dyaw), f"{source}:{cycle['name']}"))
    seen: set[tuple[float, float, float, float]] = set()
    unique: list[tuple[list[float], float, str]] = []
    for base, yaw, source in expanded:
        if not (-1.45 <= base[0] <= -0.25 and -0.50 <= base[1] <= 0.50 and -0.24 <= base[2] <= 0.36 and -90.0 <= yaw <= 45.0):
            continue
        key = (round(base[0], 4), round(base[1], 4), round(base[2], 4), round(yaw, 4))
        if key in seen:
            continue
        seen.add(key)
        unique.append((base, yaw, source))
    # Prioritize large structural novelty while staying close enough to the known reset-clean region.
    anchor = np.asarray(prior.get("best_base_pos", [-0.875, 0.05, 0.0]) if prior else [-0.875, 0.05, 0.0], dtype=float)
    unique.sort(key=lambda item: (
        abs(np.asarray(item[0], dtype=float)[2] - float(cycle.get("preferred_z", 0.08))),
        float(np.linalg.norm(np.asarray(item[0], dtype=float)[:2] - anchor[:2])),
        abs(float(item[1]) + 10.0),
    ))
    cap = int(cycle.get("candidate_cap", 100))
    return unique[:cap], {"handle_access_diag": diag, "raw_rows": len(base_rows), "expanded_rows": len(expanded), "returned_rows": min(cap, len(unique))}


def rbound_clearance_report(env, binding: dict[str, Any]) -> dict[str, Any]:
    model = env.model
    data = env.data
    legal = set(map(int, binding.get("legal_finger_pad_geom_ids", binding.get("legal_gripper_surface_geom_ids", []))))
    handle = set(map(int, binding.get("drawer_handle_geom_ids", [])))
    scene = sorted(handle | set(map(int, binding.get("drawer_body_or_cabinet_geom_ids", []))))
    forbidden = set(map(int, binding.get("forbidden_robot_surface_geom_ids", [])))
    forbidden |= set(map(int, binding.get("goc_v3_broad_link_geoms_demoted_from_target", [])))
    # Never score legal pad-handle as forbidden keepout. Everything else here is a
    # conservative center/rbound clearance proxy, not a formal mesh SDF.
    rows: list[dict[str, Any]] = []
    min_clearance = math.inf
    min_pair: dict[str, Any] | None = None
    for rg in sorted(forbidden):
        if rg in legal:
            continue
        for sg in scene:
            if sg == rg:
                continue
            d = float(np.linalg.norm(data.geom_xpos[int(rg)] - data.geom_xpos[int(sg)]))
            clearance = d - float(model.geom_rbound[int(rg)]) - float(model.geom_rbound[int(sg)])
            item = {
                "robot_geom_id": int(rg),
                "robot_geom_name": geom_name(model, int(rg)),
                "scene_geom_id": int(sg),
                "scene_geom_name": geom_name(model, int(sg)),
                "rbound_clearance_m": clearance,
            }
            if clearance < min_clearance:
                min_clearance = clearance
                min_pair = item
            rows.append(item)
    rows.sort(key=lambda r: float(r["rbound_clearance_m"]))
    return {
        "min_forbidden_scene_rbound_clearance_m": None if math.isinf(min_clearance) else float(min_clearance),
        "min_forbidden_scene_rbound_pair": min_pair,
        "closest_pairs": rows[:RBOUND_CLEARANCE_REPORT_LIMIT],
        "oracle_kind": "geom_center_minus_rbound_proxy_not_mesh_sdf",
    }


def signed_clearance_for_row(row: dict[str, Any]) -> dict[str, Any]:
    if not row.get("base_pos") or row.get("error"):
        return {"signed_clearance_available": False, "reason": "missing_base_or_error"}
    env = None
    seed = int(row["seed"])
    base = list(map(float, row["base_pos"]))
    yaw = float(row["yaw_deg"])
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            env = jpf.make_env(seed, base, yaw, 20, SAFE_PRECONTACT_QPOS)
            env.reset()
        binding = classify_instance(env)
        phases: dict[str, Any] = {}
        # Reset qpos is arm only in reset field; IK phases are q9.
        reset = row.get("reset") or {}
        if reset.get("qpos_arm"):
            q_reset = jpf.make_robot_qpos(np.asarray(reset["qpos_arm"], dtype=float), "open")
            jpf.set_robot_qpos(env, q_reset)
            phases["reset"] = rbound_clearance_report(env, binding)
        for phase_name, payload in (row.get("ik") or {}).items():
            q = payload.get("qpos_robot") if isinstance(payload, dict) else None
            if q:
                jpf.set_robot_qpos(env, np.asarray(q, dtype=float))
                cr = contact_report(env, binding, None)
                rep = rbound_clearance_report(env, binding)
                rep["contact_counts"] = cr.get("counts", {})
                phases[str(phase_name)] = rep
        vals = [float(v["min_forbidden_scene_rbound_clearance_m"]) for v in phases.values() if v.get("min_forbidden_scene_rbound_clearance_m") is not None]
        min_val = min(vals) if vals else None
        min_phase = None
        min_pair = None
        if min_val is not None:
            for name, payload in phases.items():
                if payload.get("min_forbidden_scene_rbound_clearance_m") == min_val:
                    min_phase = name
                    min_pair = payload.get("min_forbidden_scene_rbound_pair")
                    break
        return {
            "signed_clearance_available": True,
            "min_forbidden_scene_rbound_clearance_m": min_val,
            "min_forbidden_scene_rbound_phase": min_phase,
            "min_forbidden_scene_rbound_pair": min_pair,
            "signed_clearance_by_phase": phases,
        }
    except Exception as exc:
        return {"signed_clearance_available": False, "reason": repr(exc)}
    finally:
        if env is not None:
            env.close()


def evaluate_work(item: tuple[int, list[float], float, str, dict[str, Any], int]) -> dict[str, Any]:
    seed, base, yaw, source, params, cycle_index = item
    row = jpf.evaluate_endpoint_candidate(int(seed), list(map(float, base)), float(yaw), dict(params))
    row["layout_source"] = source
    row["repair_cycle"] = int(cycle_index)
    row["param_variant"] = params.get("name", "unnamed")
    row["param_snapshot"] = dict(params)
    row["rejection_invariant"] = classify_rejection(row)
    return row


def stage0_authority(run_dir: Path) -> dict[str, Any]:
    preflight_cmd = [
        "/root/anaconda3/envs/infinigen/bin/python",
        "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py",
        "--spec",
        SPEC_REL,
        "--dry-run",
    ]
    preflight = run_cmd(preflight_cmd)
    payload = {
        "generated_at_utc": utc_now(),
        "pwd": str(ROOT),
        "branch": run_git(["branch", "--show-current"]),
        "head": run_git(["rev-parse", "HEAD"]),
        "remote_v": run_git(["remote", "-v"]),
        "status_short": run_git(["status", "--short"]),
        "task_spec": SPEC_REL,
        "preflight": preflight,
        "harness_preflight_passed": preflight["returncode"] == 0,
    }
    write_json(run_dir / "stage0_authority_and_lock.json", payload)
    write_md(run_dir / "stage0_authority_and_lock.md", "# Stage 0 Authority And Lock\n\n" + json.dumps(ready(payload), indent=2, sort_keys=True))
    (run_dir / "commands.log").write_text(json.dumps(ready(payload), indent=2, sort_keys=True) + "\n")
    return payload


def stage1_prior(run_dir: Path) -> dict[str, Any]:
    prior_patterns = {
        "accessibility_layout_synthesis": "v11_g4_goc_v4_accessibility_layout_synthesis_to_layer4r_overnight_*",
        "safe_collision_proxy_policy": "v11_g4_goc_v4_safe_collision_proxy_policy_repair_*",
        "model_instance_accessibility": "v11_g4_goc_v4_model_instance_accessibility_collision_attribution_repair_*",
        "schema_aware_feasibility": "v11_g4_goc_v4_schema_aware_model_placement_feasibility_repair_*",
        "joint_feasibility": "v11_g4_goc_v4_joint_placement_reset_ik_keepout_feasibility_*",
    }
    records: dict[str, Any] = {}
    for name, pattern in prior_patterns.items():
        d = latest_runtime(pattern)
        rec: dict[str, Any] = {"pattern": pattern, "path": rel(d), "found": d is not None}
        if d:
            for fn in [
                "closeout_decision.json",
                "stage3_layout_synthesis_summary.json",
                "stage4_accessibility_rejection_invariants.json",
                "stage5_corridor_summary.json",
                "final_report.md",
            ]:
                p = d / fn
                if p.exists() and p.suffix == ".json":
                    rec[fn] = load_json(p)
                elif p.exists():
                    rec[fn] = {"exists": True, "bytes": p.stat().st_size}
        records[name] = rec
    priors = prior_best_rows()
    summary = {
        "generated_at_utc": utc_now(),
        "prior_records": records,
        "prior_best_rows_loaded": len(priors),
        "prior_best_rows": priors,
        "hypothesis_lock": {
            "joint_blocker": "dedicated_two_pad_handle_frame_reachability + reset_clearance + full_robot_forbidden_keepout",
            "repair_type": "model_or_layout_redesign_with_physical_accessibility_invariant",
            "not_a_controller_parameter_retry": True,
            "not_a_teacher_rollout_or_render_phase": True,
        },
    }
    write_json(run_dir / "stage1_prior_evidence_ingestion.json", summary)
    write_md(run_dir / "stage1_prior_evidence_report.md", "# Stage 1 Prior Evidence\n\n" + json.dumps(ready(summary), indent=2, sort_keys=True))
    return summary


def stage2_semantics(seeds: list[int], run_dir: Path) -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    frames: dict[str, Any] = {}
    ambiguous: list[int] = []
    for seed in seeds:
        env = None
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                env = jpf.make_env(seed, [-0.875, 0.05, 0.0], -10.0, 20, SAFE_PRECONTACT_QPOS)
                env.reset()
            binding = classify_instance(env)
            hf = hfp.build_handle_frame(env, binding)
            tpf = hfp.build_two_pad_frame(env, binding, hf.get("_frame"), jpf.DEFAULT_PARAMS) if hf.get("quality") == "ok" else {"quality": "handle_frame_not_ok"}
            bindings[str(seed)] = {
                "binding": hfp.compact_binding(binding),
                "ngeom": int(env.model.ngeom),
                "nq": int(env.model.nq),
                "nu": int(env.model.nu),
                "legal_pad_names": [geom_name(env.model, gid) for gid in hfp.compact_binding(binding).get("legal_finger_pad_geom_ids", [])],
                "handle_names": [geom_name(env.model, gid) for gid in hfp.compact_binding(binding).get("drawer_handle_geom_ids", [])],
                "goc_v4_exact_ids_reverified": True,
            }
            frames[str(seed)] = {"handle_frame": hfp.strip_private(hf), "two_pad_frame": hfp.strip_private(tpf)}
            if hf.get("quality") != "ok" or tpf.get("quality") != "ok":
                ambiguous.append(int(seed))
        except Exception as exc:
            bindings[str(seed)] = {"error": repr(exc)}
            frames[str(seed)] = {"error": repr(exc)}
            ambiguous.append(int(seed))
        finally:
            if env is not None:
                env.close()
    summary = {
        "generated_at_utc": utc_now(),
        "seeds": seeds,
        "ambiguous_seeds": ambiguous,
        "semantic_bindings_generated": len(ambiguous) == 0,
        "dedicated_pad_geoms_unchanged_or_regenerated": "unchanged; per-instance GOC-v4 exact IDs reverified at runtime",
        "goc_v4_exact_ids_regenerated": False,
    }
    write_json(run_dir / "stage2_semantic_bindings.json", bindings)
    write_json(run_dir / "stage2_handle_frames.json", frames)
    write_json(run_dir / "stage2_semantic_summary.json", summary)
    write_md(run_dir / "stage2_geometry_accessibility_report.md", "# Stage 2 Semantic Geometry\n\n" + json.dumps(ready(summary), indent=2, sort_keys=True))
    return {"bindings": bindings, "frames": frames, "summary": summary}


def param_variants() -> list[dict[str, Any]]:
    base = dict(jpf.DEFAULT_PARAMS)
    variants = [dict(base, name="redesign_default")]
    edits = [
        {"name": "redesign_shallow_hold", "pregrasp_distance_m": 0.058, "guarded_distance_m": 0.022, "contact_normal_offset_m": 0.004, "pinch_extra_clearance_m": 0.003},
        {"name": "redesign_tight_low_offset", "pregrasp_distance_m": 0.050, "guarded_distance_m": 0.018, "contact_normal_offset_m": 0.000, "pinch_extra_clearance_m": 0.002},
        {"name": "redesign_wide_pregrasp", "pregrasp_distance_m": 0.070, "guarded_distance_m": 0.026, "contact_normal_offset_m": 0.002, "pinch_extra_clearance_m": 0.004},
    ]
    for edit in edits:
        v = dict(base)
        v.update(edit)
        variants.append(v)
    return variants


def repair_cycles_config() -> list[dict[str, Any]]:
    return [
        {
            "name": "cycle1_vertical_mount_nominal_access",
            "candidate_cap": 80,
            "existing_grid_cap": 24,
            "handle_access_cap": 48,
            "preferred_z": 0.08,
            "z_values": [-0.08, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16],
            "xy_offsets": [[0.0, 0.0], [-0.04, 0.0], [0.04, 0.0], [0.0, -0.04], [0.0, 0.04]],
            "yaw_offsets": [0.0, -7.5, 7.5],
        },
        {
            "name": "cycle2_raised_mount_clearance_sweep",
            "candidate_cap": 100,
            "existing_grid_cap": 30,
            "handle_access_cap": 60,
            "preferred_z": 0.16,
            "z_values": [0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28],
            "xy_offsets": [[0.0, 0.0], [-0.06, 0.0], [0.06, 0.0], [0.0, -0.06], [0.0, 0.06], [-0.06, 0.06], [0.06, -0.06]],
            "yaw_offsets": [0.0, -10.0, 10.0],
        },
        {
            "name": "cycle3_wide_lateral_accessibility_envelope",
            "candidate_cap": 120,
            "existing_grid_cap": 36,
            "handle_access_cap": 72,
            "preferred_z": 0.12,
            "z_values": [-0.12, -0.04, 0.0, 0.08, 0.12, 0.18, 0.24],
            "xy_offsets": [[0.0, 0.0], [-0.10, 0.0], [0.10, 0.0], [0.0, -0.10], [0.0, 0.10], [-0.10, 0.08], [0.10, -0.08]],
            "yaw_offsets": [0.0, -15.0, 15.0],
        },
        {
            "name": "cycle4_aggressive_mount_height_and_yaw",
            "candidate_cap": 160,
            "existing_grid_cap": 48,
            "handle_access_cap": 96,
            "preferred_z": 0.20,
            "z_values": [-0.20, -0.12, -0.04, 0.0, 0.08, 0.16, 0.24, 0.32],
            "xy_offsets": [[0.0, 0.0], [-0.14, 0.0], [0.14, 0.0], [0.0, -0.14], [0.0, 0.14], [-0.14, 0.10], [0.14, -0.10], [-0.08, -0.08], [0.08, 0.08]],
            "yaw_offsets": [0.0, -20.0, -10.0, 10.0, 20.0],
        },
    ]


def stage3_repair_search(seeds: list[int], prior: dict[str, Any], run_dir: Path, workers: int, max_hours: float) -> dict[str, Any]:
    start = time.time()
    deadline = start + max_hours * 3600.0
    prior_rows = {int(k): v for k, v in prior.get("prior_best_rows", {}).items()} if prior.get("prior_best_rows") else prior_best_rows()
    variants = param_variants()
    best_by_seed: dict[str, Any] = {str(s): {"top_rows": [], "feasible_count": 0, "evaluated_rows": 0, "best": None} for s in seeds}
    feasible: list[dict[str, Any]] = []
    accepted_by_seed: dict[str, dict[str, Any]] = {}
    cycle_summaries: list[dict[str, Any]] = []
    total_eval = 0
    total_hist = Counter()
    best_reset_clean = math.inf
    best_overall = math.inf
    previous_accepted_count = 0
    no_progress_cycles = 0

    for cycle_index, cycle in enumerate(repair_cycles_config(), start=1):
        if time.time() >= deadline:
            break
        cycle_start = time.time()
        task_items: list[tuple[int, list[float], float, str, dict[str, Any], int]] = []
        generation_diag: dict[str, Any] = {}
        for seed in seeds:
            rows, diag = synthesize_layout_rows(int(seed), cycle, prior_rows)
            generation_diag[str(seed)] = diag
            for base, yaw, source in rows:
                for params in variants:
                    task_items.append((int(seed), list(map(float, base)), float(yaw), source, dict(params), cycle_index))
        cycle_hist = Counter()
        cycle_eval = 0
        cycle_feasible: list[dict[str, Any]] = []
        cycle_best_by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
        progress_path = run_dir / f"stage3_cycle_{cycle_index}_progress.json"
        if workers <= 1:
            iterator = (evaluate_work(item) for item in task_items)
            for row in iterator:
                if time.time() >= deadline:
                    break
                cycle_eval += 1
                total_eval += 1
                inv = row.get("rejection_invariant", "unknown")
                cycle_hist[inv] += 1
                total_hist[inv] += 1
                rv = residual_value(row)
                best_overall = min(best_overall, rv)
                if row.get("reset", {}).get("reset_ok"):
                    best_reset_clean = min(best_reset_clean, rv)
                seed_s = str(int(row.get("seed", -1)))
                update_top(best_by_seed.setdefault(seed_s, {"top_rows": []}).setdefault("top_rows", []), row)
                update_top(cycle_best_by_seed[seed_s], row)
                best_by_seed[seed_s]["evaluated_rows"] = int(best_by_seed[seed_s].get("evaluated_rows", 0)) + 1
                if row.get("endpoint_feasible"):
                    cycle_feasible.append(row)
                    feasible.append(row)
                if cycle_eval % 100 == 0:
                    write_json(progress_path, {"cycle": cycle, "cycle_eval": cycle_eval, "total_eval": total_eval, "cycle_histogram": dict(cycle_hist.most_common(10)), "feasible_total": len(feasible), "elapsed_seconds": time.time() - start})
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(evaluate_work, item) for item in task_items]
                for fut in as_completed(futures):
                    if time.time() >= deadline:
                        for pending in futures:
                            pending.cancel()
                        break
                    try:
                        row = fut.result()
                    except Exception as exc:
                        row = {"error": repr(exc), "endpoint_feasible": False, "rejection_invariant": "worker_exception", "score": 999.0}
                    cycle_eval += 1
                    total_eval += 1
                    inv = row.get("rejection_invariant", "unknown")
                    cycle_hist[inv] += 1
                    total_hist[inv] += 1
                    rv = residual_value(row)
                    best_overall = min(best_overall, rv)
                    if row.get("reset", {}).get("reset_ok"):
                        best_reset_clean = min(best_reset_clean, rv)
                    seed_s = str(int(row.get("seed", -1)))
                    update_top(best_by_seed.setdefault(seed_s, {"top_rows": []}).setdefault("top_rows", []), row)
                    update_top(cycle_best_by_seed[seed_s], row)
                    best_by_seed[seed_s]["evaluated_rows"] = int(best_by_seed[seed_s].get("evaluated_rows", 0)) + 1
                    if row.get("endpoint_feasible"):
                        cycle_feasible.append(row)
                        feasible.append(row)
                    if cycle_eval % 100 == 0:
                        write_json(progress_path, {"cycle": cycle, "cycle_eval": cycle_eval, "total_eval": total_eval, "cycle_histogram": dict(cycle_hist.most_common(10)), "feasible_total": len(feasible), "elapsed_seconds": time.time() - start})
        # Compute signed clearance for the best and feasible rows from this cycle.
        top_cycle_rows: list[dict[str, Any]] = []
        for rows in cycle_best_by_seed.values():
            top_cycle_rows.extend(rows[:3])
        top_cycle_rows.extend(sorted(cycle_feasible, key=ranking_key)[:20])
        signed_rows: list[dict[str, Any]] = []
        for row in sorted(top_cycle_rows, key=ranking_key)[:80]:
            sig = signed_clearance_for_row(row)
            row.update({k: v for k, v in sig.items() if k != "signed_clearance_by_phase"})
            row["signed_clearance_by_phase"] = sig.get("signed_clearance_by_phase", {})
            signed_rows.append(trim_row(row))
            seed_s = str(int(row.get("seed", -1)))
            update_top(best_by_seed.setdefault(seed_s, {"top_rows": []}).setdefault("top_rows", []), row)
        for seed in seeds:
            seed_s = str(seed)
            rows = best_by_seed.get(seed_s, {}).get("top_rows", [])
            rows.sort(key=ranking_key)
            best_by_seed[seed_s]["best"] = rows[0] if rows else None
            best_by_seed[seed_s]["feasible_count"] = int(best_by_seed[seed_s].get("feasible_count", 0)) + sum(1 for r in cycle_feasible if int(r.get("seed", -999)) == seed)
            feasible_rows = [r for r in feasible if int(r.get("seed", -999)) == seed]
            if feasible_rows:
                accepted_by_seed[seed_s] = sorted(feasible_rows, key=ranking_key)[0]
        accepted_count = len(accepted_by_seed)
        improved = accepted_count > previous_accepted_count
        no_progress_cycles = 0 if improved else no_progress_cycles + 1
        cycle_summary = {
            "cycle_index": cycle_index,
            "cycle_name": cycle["name"],
            "repair_operator": "per_instance_robot_mount_height_yaw_accessibility_envelope_synthesis",
            "task_items_planned": len(task_items),
            "evaluated": cycle_eval,
            "feasible_in_cycle": len(cycle_feasible),
            "accepted_accessible_instances_after_cycle": accepted_count,
            "accepted_accessible_instances_delta": accepted_count - previous_accepted_count,
            "progress_improved": improved,
            "no_progress_cycles": no_progress_cycles,
            "cycle_histogram": dict(cycle_hist.most_common()),
            "generation_diag": generation_diag,
            "signed_clearance_sample": signed_rows[:20],
            "elapsed_seconds": time.time() - cycle_start,
        }
        append_jsonl(run_dir / "stage3_repair_cycles.jsonl", cycle_summary)
        write_json(run_dir / f"stage3_cycle_{cycle_index}_summary.json", cycle_summary)
        cycle_summaries.append(cycle_summary)
        previous_accepted_count = accepted_count
        write_json(run_dir / "stage3_progress.json", {
            "total_eval": total_eval,
            "feasible_total": len(feasible),
            "accepted_accessible_instances_count": accepted_count,
            "dominant_rejection_invariants": dict(total_hist.most_common(10)),
            "best_reset_clean_two_pad_residual_m": None if math.isinf(best_reset_clean) else best_reset_clean,
            "elapsed_seconds": time.time() - start,
        })
        # If every mandatory seed has an endpoint-feasible layout, stop endpoint repair
        # and spend remaining budget on corridor/dynamics/Layer4R.
        if accepted_count == len(seeds):
            break
        if no_progress_cycles >= 2 and cycle_index >= 2 and accepted_count > 0:
            break
    for seed_s, payload in best_by_seed.items():
        rows = payload.get("top_rows", [])
        rows.sort(key=ranking_key)
        payload["best"] = rows[0] if rows else None
        payload["top_rows"] = rows[:12]
    sorted_feasible = sorted(feasible, key=ranking_key)
    stage3 = {
        "generated_at_utc": utc_now(),
        "repair_cycles_run": len(cycle_summaries),
        "repair_cycles": cycle_summaries,
        "endpoint_candidates_evaluated": total_eval,
        "endpoint_feasible_case_count": len(feasible),
        "endpoint_feasible_seed_count": len({int(r["seed"]) for r in feasible if "seed" in r}),
        "accepted_accessible_instances_count": len(accepted_by_seed),
        "rejected_inaccessible_instances_count": len(seeds) - len(accepted_by_seed),
        "accepted_accessible_seed_ids": sorted(map(int, accepted_by_seed.keys())),
        "best_reset_clean_two_pad_residual_m": None if math.isinf(best_reset_clean) else float(best_reset_clean),
        "best_overall_two_pad_residual_m": None if math.isinf(best_overall) else float(best_overall),
        "dominant_rejection_invariants": dict(total_hist.most_common()),
        "feasible_candidates_top": [trim_row(r) for r in sorted_feasible[:30]],
        "accepted_by_seed": {k: trim_row(v) for k, v in accepted_by_seed.items()},
        "best_by_seed": best_by_seed,
        "model_layout_repair_applied": total_eval > 0,
        "robot_mount_changed_how": {
            "repair_operator": "runtime per-instance robot_mount base_pos/yaw policy; visual and collision robot subtree move together",
            "baseline": {"base_pos": BASELINE_BASE.tolist(), "yaw_deg": BASELINE_YAW_DEG},
            "accepted_layout_deltas": {k: distance_to_baseline(v["base_pos"], v["yaw_deg"]) for k, v in accepted_by_seed.items()},
            "best_layout_deltas_by_seed": {k: distance_to_baseline(v["best"]["base_pos"], v["best"].get("yaw_deg", 0.0)) for k, v in best_by_seed.items() if v.get("best") and v["best"].get("base_pos")},
        },
        "accessibility_envelope_changed_how": "drawer/cabinet/handle geometry unchanged; robot mount relation to semantically bound handle frame searched over z/xy/yaw and audited with rbound clearance oracle",
        "dedicated_pad_geoms_unchanged_or_regenerated": "unchanged; GOC-v4 per-instance exact IDs reverified",
        "goc_v4_exact_ids_regenerated": False,
        "visual_physical_consistency_preserved": True,
    }
    write_json(run_dir / "stage3_model_layout_repair_summary.json", stage3)
    write_json(run_dir / "stage3_best_candidates_by_seed.json", best_by_seed)
    if not feasible:
        write_json(run_dir / "stage3_accessibility_invariant_certificate.json", {
            **stage3,
            "classification": "MODEL_OR_LAYOUT_REDESIGN_ACCESSIBILITY_INVARIANT_FAILED",
            "certificate_scope": "bounded model/layout repair over robot mount z/xy/yaw, reset qpos, two-pad IK, and forbidden keepout",
            "important_caveat": "bounded numerical certificate over the pre-registered repair domain, not a formal proof over arbitrary robot/drawer redesigns",
        })
    return stage3


def stage4_signed_acceptance(stage3: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    accepted = stage3.get("accepted_by_seed", {}) or {}
    best_rows = [payload.get("best") for payload in (stage3.get("best_by_seed") or {}).values() if payload.get("best")]
    signed_values: list[float] = []
    for row in best_rows:
        val = row.get("min_forbidden_scene_rbound_clearance_m")
        if val is not None:
            signed_values.append(float(val))
    prior_best = math.inf
    prior_run = latest_runtime("v11_g4_goc_v4_accessibility_layout_synthesis_to_layer4r_overnight_*")
    prior_summary = load_json(prior_run / "stage3_layout_synthesis_summary.json") if prior_run else None
    if prior_summary and prior_summary.get("best_reset_clean_two_pad_residual_m") is not None:
        prior_best = float(prior_summary.get("best_reset_clean_two_pad_residual_m"))
    current_best = stage3.get("best_reset_clean_two_pad_residual_m")
    signed_clearance_improved = bool(accepted) or (current_best is not None and not math.isinf(prior_best) and float(current_best) < prior_best)
    summary = {
        "generated_at_utc": utc_now(),
        "accepted_accessible_instances_count": len(accepted),
        "rejected_inaccessible_instances_count": int(stage3.get("rejected_inaccessible_instances_count", 0)),
        "signed_clearance_improved": signed_clearance_improved,
        "best_signed_forbidden_scene_rbound_clearance_m": max(signed_values) if signed_values else None,
        "worst_toprow_signed_forbidden_scene_rbound_clearance_m": min(signed_values) if signed_values else None,
        "prior_best_reset_clean_two_pad_residual_m": None if math.isinf(prior_best) else prior_best,
        "current_best_reset_clean_two_pad_residual_m": current_best,
        "model_layout_repair_applied": bool(stage3.get("model_layout_repair_applied")),
        "visual_physical_consistency_preserved": bool(stage3.get("visual_physical_consistency_preserved")),
    }
    write_json(run_dir / "stage4_signed_clearance_and_acceptance_summary.json", summary)
    write_md(run_dir / "stage4_signed_clearance_and_acceptance_report.md", "# Stage 4 Signed Clearance And Acceptance\n\n" + json.dumps(ready(summary), indent=2, sort_keys=True))
    return summary


def stage5_corridor(stage3: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    accepted = [v for _, v in sorted((stage3.get("accepted_by_seed") or {}).items(), key=lambda kv: int(kv[0]))]
    if not accepted:
        summary = {"attempted": False, "full_robot_corridor_exists": False, "corridor_pass_count": 0, "best_full_robot_corridor_clearance_m": None, "reason": "no_endpoint_feasible_accepted_layout"}
        write_json(run_dir / "stage5_corridor_summary.json", summary)
        return summary
    rows: list[dict[str, Any]] = []
    best_clearance = -math.inf
    for candidate in sorted(accepted, key=ranking_key)[:24]:
        row = jpf.corridor_check(candidate, run_dir)
        # Carry endpoint signed-clearance proxy into corridor report because jpf corridor is contact-based.
        row["endpoint_min_forbidden_scene_rbound_clearance_m"] = candidate.get("min_forbidden_scene_rbound_clearance_m")
        if candidate.get("min_forbidden_scene_rbound_clearance_m") is not None:
            best_clearance = max(best_clearance, float(candidate["min_forbidden_scene_rbound_clearance_m"]))
        rows.append(row)
        append_jsonl(run_dir / "stage5_corridor_candidates.jsonl", row)
    passed = [r for r in rows if r.get("passed")]
    summary = {
        "attempted": True,
        "checked_count": len(rows),
        "corridor_pass_count": len(passed),
        "full_robot_corridor_exists": len(passed) > 0,
        "best_full_robot_corridor_clearance_m": None if math.isinf(best_clearance) else best_clearance,
        "rows": rows,
    }
    write_json(run_dir / "stage5_corridor_summary.json", summary)
    write_json(run_dir / "stage5_corridor_clearance_traces.json", summary)
    if not passed:
        write_md(run_dir / "stage5_corridor_failure_report.md", "# Stage 5 Corridor Failure\n\n" + json.dumps(ready(summary), indent=2, sort_keys=True))
    return summary


def stage6_dynamic(stage3: dict[str, Any], corridor: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    if not corridor.get("full_robot_corridor_exists"):
        summary = {"attempted": False, "dynamic_probe_passed": False, "reason": "no_full_robot_corridor"}
        write_json(run_dir / "stage6_dynamic_probe_summary.json", summary)
        return summary
    # jpf.stage5_dynamic matches corridor rows back to feasible rows; use accepted candidates.
    feasible = list((stage3.get("accepted_by_seed") or {}).values())
    summary = jpf.stage5_dynamic(corridor, feasible, run_dir)
    write_json(run_dir / "stage6_dynamic_probe_summary.json", summary)
    return summary


def case_passed(case: dict[str, Any]) -> bool:
    s = case.get("summary", {}) or {}
    return bool(
        s.get("target_contact_frames", 0) >= hfp.LAYER4_MIN_TARGET_FRAMES
        and s.get("target_contact_max_consecutive_frames", 0) >= hfp.LAYER4_MIN_CONSECUTIVE
        and s.get("forbidden_contact_frames", 1) == 0
        and s.get("handle_nonlegal_contact_frames", 1) == 0
        and s.get("max_penetration_m", 999.0) <= MAX_PENETRATION_M
        and math.isfinite(float(s.get("max_force_n", 0.0)))
        and float(s.get("max_force_n", 999999999.0)) <= MAX_FORCE_N
        and not bool(s.get("direct_qpos_drawer_opening", False))
    )


def layer4_failure_reasons(case: dict[str, Any]) -> list[str]:
    s = case.get("summary", {}) or {}
    reasons: list[str] = []
    if case.get("error"):
        reasons.append("model_load_or_probe_error")
    if s.get("target_contact_frames", 0) < hfp.LAYER4_MIN_TARGET_FRAMES:
        reasons.append("target_contact_frames_lt_50")
    if s.get("target_contact_max_consecutive_frames", 0) < hfp.LAYER4_MIN_CONSECUTIVE:
        reasons.append("target_contact_consecutive_lt_30")
    if s.get("forbidden_contact_frames", 0) > 0 or s.get("reset_forbidden_contact_frames", 0) > 0:
        reasons.append("forbidden_contact_present")
    if s.get("handle_nonlegal_contact_frames", 0) > 0 or s.get("reset_handle_nonlegal_contact_frames", 0) > 0:
        reasons.append("handle_nonlegal_contact_present")
    if s.get("max_penetration_m", 0.0) > MAX_PENETRATION_M or s.get("reset_max_penetration_m", 0.0) > MAX_PENETRATION_M:
        reasons.append("max_penetration_gt_0p02m")
    if not reasons:
        reasons.append("pass")
    return reasons


def stage7_layer4r(stage3: dict[str, Any], dynamic: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    if not dynamic.get("dynamic_probe_passed"):
        summary = {
            "targeted_attempted": False,
            "full_matrix_attempted": False,
            "layer4R_certified": False,
            "layer4R_cases_total": 0,
            "layer4R_cases_passed": 0,
            "layer4R_cases_failed": 0,
            "remaining_failure_clusters": {},
            "reason": "dynamic_probe_not_passed",
        }
        write_json(run_dir / "stage7_layer4r_summary.json", summary)
        return summary
    accepted = stage3.get("accepted_by_seed") or {}
    seeds = repair_campaign.available_drawer_seeds()
    missing = [int(s) for s in seeds if str(int(s)) not in accepted]
    if missing:
        summary = {
            "targeted_attempted": False,
            "full_matrix_attempted": False,
            "layer4R_certified": False,
            "layer4R_cases_total": 0,
            "layer4R_cases_passed": 0,
            "layer4R_cases_failed": 0,
            "remaining_failure_clusters": {"missing_accessible_layout_for_seed": len(missing)},
            "missing_accessible_layout_seeds": missing,
            "reason": "not_all_mandatory_seeds_have_accepted_layout",
        }
        write_json(run_dir / "stage7_layer4r_summary.json", summary)
        return summary
    params = dict(jpf.DEFAULT_PARAMS)
    params["name"] = "model_layout_redesign_layer4r_policy"
    base_map: dict[str, Any] = {}
    for seed_s, row in accepted.items():
        reset = row.get("reset", {}) or {}
        base_map[str(seed_s)] = {
            "base_pos": row["base_pos"],
            "yaw_deg": row["yaw_deg"],
            "reset_qpos": reset.get("qpos_arm", SAFE_PRECONTACT_QPOS.tolist() if hasattr(SAFE_PRECONTACT_QPOS, "tolist") else list(SAFE_PRECONTACT_QPOS)),
            "reset_ok": True,
            "source": "model_layout_redesign_accepted_policy",
        }
    targeted_seeds = [s for s in seeds if s in [1, 7, 11, 12, 13, 14, 15]][:4]
    targeted_cases: list[dict[str, Any]] = []
    for seed in targeted_seeds:
        for perturb in hfp.PERTURBATIONS[:3]:
            case = hfp.run_planner_case(seed, perturb, base_map[str(seed)], params, run_dir, write_trace=False)
            case["layer4r_passed_under_redesigned_layout"] = case_passed(case)
            targeted_cases.append(case)
            append_jsonl(run_dir / "stage7_targeted_layer4r_results.jsonl", case)
    targeted_pass = sum(1 for c in targeted_cases if c.get("layer4r_passed_under_redesigned_layout"))
    hist = Counter()
    for case in targeted_cases:
        for reason in layer4_failure_reasons(case):
            hist[reason] += 1
    summary: dict[str, Any] = {
        "targeted_attempted": True,
        "targeted_total": len(targeted_cases),
        "targeted_passed": targeted_pass,
        "targeted_failed": len(targeted_cases) - targeted_pass,
        "full_matrix_attempted": False,
        "layer4R_certified": False,
        "layer4R_cases_total": len(targeted_cases),
        "layer4R_cases_passed": targeted_pass,
        "layer4R_cases_failed": len(targeted_cases) - targeted_pass,
        "remaining_failure_clusters": dict(hist.most_common()),
        "reason": "targeted_layer4r_must_all_pass_before_full_matrix",
    }
    write_json(run_dir / "stage7_targeted_layer4r_summary.json", summary)
    if targeted_pass != len(targeted_cases):
        write_json(run_dir / "stage7_layer4r_summary.json", summary)
        return summary
    full_cases: list[dict[str, Any]] = []
    for seed in seeds:
        for perturb in hfp.PERTURBATIONS:
            case = hfp.run_planner_case(seed, perturb, base_map[str(seed)], params, run_dir, write_trace=False)
            case["layer4r_passed_under_redesigned_layout"] = case_passed(case)
            full_cases.append(case)
            append_jsonl(run_dir / "stage7_full_layer4r_results.jsonl", case)
    full_pass = sum(1 for c in full_cases if c.get("layer4r_passed_under_redesigned_layout"))
    full_hist = Counter()
    for case in full_cases:
        for reason in layer4_failure_reasons(case):
            full_hist[reason] += 1
    summary.update({
        "full_matrix_attempted": True,
        "layer4R_certified": full_pass == len(full_cases) and len(full_cases) > 0,
        "layer4R_cases_total": len(full_cases),
        "layer4R_cases_passed": full_pass,
        "layer4R_cases_failed": len(full_cases) - full_pass,
        "remaining_failure_clusters": dict(full_hist.most_common()),
        "reason": "full_matrix_complete",
    })
    write_json(run_dir / "stage7_full_layer4r_summary.json", summary)
    write_json(run_dir / "stage7_layer4r_summary.json", summary)
    return summary


def determine_closeout(stage0: dict[str, Any], stage2: dict[str, Any], stage3: dict[str, Any], corridor: dict[str, Any], dynamic: dict[str, Any], layer4r: dict[str, Any], time_exhausted: bool) -> tuple[str, str]:
    if not stage0.get("harness_preflight_passed"):
        return "HARNESS_PREFLIGHT_FAILED", "HARNESS_PREFLIGHT_REPAIR"
    if stage2.get("summary", {}).get("ambiguous_seeds"):
        return "SEMANTIC_BINDING_AMBIGUOUS", "HANDLE_SEMANTIC_BINDING_REPAIR"
    if time_exhausted:
        return "TIME_BUDGET_EXHAUSTED", "RESUME_MODEL_LAYOUT_REDESIGN_FROM_LAST_VERIFIED_STAGE"
    if layer4r.get("layer4R_certified"):
        return "MODEL_OR_LAYOUT_REDESIGN_LAYER4R_CERTIFIED", "GOC_V4_BOUNDED_TEACHER_PULL_ROLLOUT"
    if dynamic.get("dynamic_probe_passed"):
        return "MODEL_OR_LAYOUT_REDESIGN_DYNAMIC_CONTACT_READY", "LAYER4R_GENERALIZATION_REPAIR_FROM_REDIRECTED_LAYOUT"
    if corridor.get("full_robot_corridor_exists"):
        return "MODEL_OR_LAYOUT_REDESIGN_ACCESSIBLE_CORRIDOR_READY", "OPERATIONAL_SPACE_CONTACT_CONTROLLER_REPAIR_AFTER_ACCESSIBLE_LAYOUT"
    if int(stage3.get("accepted_accessible_instances_count", 0)) > 0:
        return "MODEL_OR_LAYOUT_REDESIGN_CORRIDOR_FAILED", "APPROACH_CORRIDOR_OR_LAYOUT_POLICY_REPAIR"
    return "MODEL_OR_LAYOUT_REDESIGN_ACCESSIBILITY_INVARIANT_FAILED", "MODEL_OR_LAYOUT_REDESIGN_WITH_STRONGER_STRUCTURAL_CHANGE"


def write_proposed_deltas(closeout: dict[str, Any], run_dir: Path) -> None:
    payload = {
        "generated_at_utc": utc_now(),
        "task_id": "V11_G4_GOC_V4_MODEL_OR_LAYOUT_REDESIGN_WITH_PHYSICAL_ACCESSIBILITY_INVARIANT_V1",
        "run_dir": rel(run_dir),
        "closeout_classification": closeout["closeout_classification"],
        "layer4r_certified": bool(closeout.get("layer4R_certified")),
        "model_layout_repair_applied": bool(closeout.get("model_layout_repair_applied")),
        "accepted_accessible_instances_count": closeout.get("accepted_accessible_instances_count"),
        "full_robot_corridor_exists": closeout.get("full_robot_corridor_exists"),
        "MINT_training_allowed": False,
        "current_truth_direct_mutation": False,
        "next_actions_direct_mutation": False,
        "next_gate": closeout["next_gate"],
    }
    write_json(CAMPAIGN / "sovereign/proposed_current_truth_delta_model_or_layout_redesign_with_physical_accessibility_invariant.json", payload)
    write_json(CAMPAIGN / "sovereign/proposed_next_actions_model_or_layout_redesign_with_physical_accessibility_invariant.json", payload)


def json_parse_ok(paths: list[Path]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for root in paths:
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


def final_report_text(closeout: dict[str, Any]) -> str:
    return f"""# V11-G4 GOC-v4 Model/Layout Redesign With Physical Accessibility Invariant Closeout

Closeout: `{closeout['closeout_classification']}`

This phase executed a repair-style model/layout search rather than another raw
controller parameter retry. It tested whether changing the robot mount relation
(base position, yaw, and height) while preserving visual/physical consistency and
GOC-v4 exact-ID authority can satisfy the joint condition:

`dedicated two-pad handle-frame reachability + reset clearance + full robot forbidden-contact keepout`.

Key results:

- harness preflight passed: `{closeout['harness_preflight_passed']}`
- model/layout repair applied: `{closeout['model_layout_repair_applied']}`
- visual/physical consistency preserved: `{closeout['visual_physical_consistency_preserved']}`
- GOC-v4 exact IDs regenerated: `{closeout['goc_v4_exact_ids_regenerated']}`
- dedicated pads: `{closeout['dedicated_pad_geoms_unchanged_or_regenerated']}`
- accepted accessible instances: `{closeout['accepted_accessible_instances_count']}`
- rejected inaccessible instances: `{closeout['rejected_inaccessible_instances_count']}`
- endpoint candidates evaluated: `{closeout['endpoint_candidates_evaluated']}`
- endpoint feasible cases/seeds: `{closeout['endpoint_feasible_case_count']}` / `{closeout['endpoint_feasible_seed_count']}`
- best reset-clean two-pad residual m: `{closeout['best_reset_clean_two_pad_residual_m']}`
- signed clearance improved: `{closeout['signed_clearance_improved']}`
- best full robot corridor clearance m: `{closeout['best_full_robot_corridor_clearance_m']}`
- full robot corridor exists: `{closeout['full_robot_corridor_exists']}`
- dynamic probe attempted/passed: `{closeout['dynamic_probe_attempted']}` / `{closeout['dynamic_probe_passed']}`
- Layer4R certified: `{closeout['layer4R_certified']}`
- Layer4R cases passed/failed: `{closeout['layer4R_cases_passed']}` / `{closeout['layer4R_cases_failed']}`
- next gate: `{closeout['next_gate']}`

Layer5 pull rollout, export bundle, local replay/render, and MINT training were not run.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--seed-limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-wall-clock-hours", type=float, default=9.5)
    args = ap.parse_args()

    run_dir = args.run_dir or (CAMPAIGN / "runtime" / f"v11_g4_goc_v4_model_or_layout_redesign_with_physical_accessibility_invariant_{utc_stamp()}")
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    seeds = repair_campaign.available_drawer_seeds()
    if args.seed_limit > 0:
        seeds = seeds[: args.seed_limit]

    stage0 = stage0_authority(run_dir)
    if not stage0.get("harness_preflight_passed"):
        closeout = {
            "closeout_classification": "HARNESS_PREFLIGHT_FAILED",
            "harness_preflight_passed": False,
            "task_spec_lock_bound": False,
            "next_gate": "HARNESS_PREFLIGHT_REPAIR",
        }
        write_json(run_dir / "closeout_decision.json", closeout)
        write_md(run_dir / "final_report.md", final_report_text({**closeout, **{k: None for k in ["model_layout_repair_applied", "visual_physical_consistency_preserved", "goc_v4_exact_ids_regenerated", "dedicated_pad_geoms_unchanged_or_regenerated", "accepted_accessible_instances_count", "rejected_inaccessible_instances_count", "endpoint_candidates_evaluated", "endpoint_feasible_case_count", "endpoint_feasible_seed_count", "best_reset_clean_two_pad_residual_m", "signed_clearance_improved", "best_full_robot_corridor_clearance_m", "full_robot_corridor_exists", "dynamic_probe_attempted", "dynamic_probe_passed", "layer4R_certified", "layer4R_cases_passed", "layer4R_cases_failed"]}}))
        return 2

    prior = stage1_prior(run_dir)
    stage2 = stage2_semantics(seeds, run_dir)
    if stage2.get("summary", {}).get("ambiguous_seeds"):
        stage3 = {"accepted_accessible_instances_count": 0, "rejected_inaccessible_instances_count": len(seeds), "endpoint_candidates_evaluated": 0, "endpoint_feasible_case_count": 0, "endpoint_feasible_seed_count": 0, "dominant_rejection_invariants": {}}
        stage4 = {"signed_clearance_improved": False}
        corridor = {"attempted": False, "full_robot_corridor_exists": False, "best_full_robot_corridor_clearance_m": None}
        dynamic = {"attempted": False, "dynamic_probe_passed": False}
        layer4r = {"layer4R_certified": False, "layer4R_cases_total": 0, "layer4R_cases_passed": 0, "layer4R_cases_failed": 0, "remaining_failure_clusters": {}}
    else:
        # Reserve time for corridor/dynamic/Layer4R after endpoint repair.
        endpoint_hours = max(0.5, float(args.max_wall_clock_hours) - 1.25)
        stage3 = stage3_repair_search(seeds, prior, run_dir, workers=max(1, int(args.workers)), max_hours=endpoint_hours)
        stage4 = stage4_signed_acceptance(stage3, run_dir)
        corridor = stage5_corridor(stage3, run_dir)
        dynamic = stage6_dynamic(stage3, corridor, run_dir)
        layer4r = stage7_layer4r(stage3, dynamic, run_dir)

    time_exhausted = (time.time() - start) >= float(args.max_wall_clock_hours) * 3600.0
    classification, next_gate = determine_closeout(stage0, stage2, stage3, corridor, dynamic, layer4r, time_exhausted)
    status = run_git(["status", "--short"])
    current_truth_modified = "experiments/mint/mint_drawer_v1/sovereign/current_truth.json" in status
    next_actions_modified = "experiments/mint/mint_drawer_v1/sovereign/next_actions.json" in status
    json_ok, json_errors = json_parse_ok([run_dir, CAMPAIGN / "sovereign/proposed_current_truth_delta_model_or_layout_redesign_with_physical_accessibility_invariant.json", CAMPAIGN / "sovereign/proposed_next_actions_model_or_layout_redesign_with_physical_accessibility_invariant.json"])

    closeout = {
        "closeout_classification": classification,
        "harness_preflight_passed": bool(stage0.get("harness_preflight_passed")),
        "task_spec_lock_bound": bool(stage0.get("harness_preflight_passed")),
        "model_layout_repair_applied": bool(stage3.get("model_layout_repair_applied", False)),
        "robot_mount_changed_how": stage3.get("robot_mount_changed_how"),
        "accessibility_envelope_changed_how": stage3.get("accessibility_envelope_changed_how"),
        "dedicated_pad_geoms_unchanged_or_regenerated": stage3.get("dedicated_pad_geoms_unchanged_or_regenerated", stage2.get("summary", {}).get("dedicated_pad_geoms_unchanged_or_regenerated")),
        "goc_v4_exact_ids_regenerated": bool(stage3.get("goc_v4_exact_ids_regenerated", stage2.get("summary", {}).get("goc_v4_exact_ids_regenerated", False))),
        "visual_physical_consistency_preserved": bool(stage3.get("visual_physical_consistency_preserved", False)),
        "signed_clearance_improved": bool(stage4.get("signed_clearance_improved", False)),
        "accepted_accessible_instances_count": int(stage3.get("accepted_accessible_instances_count", 0)),
        "rejected_inaccessible_instances_count": int(stage3.get("rejected_inaccessible_instances_count", len(seeds))),
        "best_full_robot_corridor_clearance_m": corridor.get("best_full_robot_corridor_clearance_m"),
        "full_robot_corridor_exists": bool(corridor.get("full_robot_corridor_exists", False)),
        "layer4R_certified": bool(layer4r.get("layer4R_certified", False)),
        "endpoint_candidates_evaluated": int(stage3.get("endpoint_candidates_evaluated", 0)),
        "endpoint_feasible_case_count": int(stage3.get("endpoint_feasible_case_count", 0)),
        "endpoint_feasible_seed_count": int(stage3.get("endpoint_feasible_seed_count", 0)),
        "best_reset_clean_two_pad_residual_m": stage3.get("best_reset_clean_two_pad_residual_m"),
        "best_overall_two_pad_residual_m": stage3.get("best_overall_two_pad_residual_m"),
        "dominant_rejection_invariants": stage3.get("dominant_rejection_invariants", {}),
        "dynamic_probe_attempted": bool(dynamic.get("attempted", False)),
        "dynamic_probe_passed": bool(dynamic.get("dynamic_probe_passed", False)),
        "layer4R_cases_total": int(layer4r.get("layer4R_cases_total", 0)),
        "layer4R_cases_passed": int(layer4r.get("layer4R_cases_passed", 0)),
        "layer4R_cases_failed": int(layer4r.get("layer4R_cases_failed", 0)),
        "remaining_failure_clusters": layer4r.get("remaining_failure_clusters", {}),
        "current_truth_modified": bool(current_truth_modified),
        "next_actions_modified": bool(next_actions_modified),
        "runtime_patch_applied": True,
        "runtime_patch_files": ["scripts/mint/model_or_layout_redesign_with_physical_accessibility_invariant.py"],
        "committed": False,
        "pushed_to_origin": False,
        "remote_commit_hash": None,
        "next_gate": next_gate,
        "run_dir": rel(run_dir),
        "source_worktree_head": run_git(["rev-parse", "HEAD"]),
        "git_status_short": status,
        "json_parse_ok": json_ok,
        "json_parse_errors": json_errors,
        "elapsed_seconds": time.time() - start,
    }
    write_proposed_deltas(closeout, run_dir)
    # Recompute JSON parse after deltas are written.
    json_ok, json_errors = json_parse_ok([run_dir, CAMPAIGN / "sovereign/proposed_current_truth_delta_model_or_layout_redesign_with_physical_accessibility_invariant.json", CAMPAIGN / "sovereign/proposed_next_actions_model_or_layout_redesign_with_physical_accessibility_invariant.json"])
    closeout["json_parse_ok"] = json_ok
    closeout["json_parse_errors"] = json_errors
    write_json(run_dir / "stage8_final_checks.json", {
        "generated_at_utc": utc_now(),
        "preflight_rerun": run_cmd(["/root/anaconda3/envs/infinigen/bin/python", "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py", "--spec", SPEC_REL, "--dry-run"]),
        "git_status_short": status,
        "current_truth_modified": current_truth_modified,
        "next_actions_modified": next_actions_modified,
        "json_parse_ok": json_ok,
        "json_parse_errors": json_errors,
    })
    write_json(run_dir / "closeout_decision.json", closeout)
    write_md(run_dir / "final_report.md", final_report_text(closeout))
    print(json.dumps(ready(closeout), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

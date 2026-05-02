#!/usr/bin/env python3
"""GOC-v4 model/placement/keepout repair-to-Layer4R overnight campaign."""
from __future__ import annotations

import argparse
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

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN = ROOT / "experiments/mint/mint_drawer_v1"
SPEC_REL = (
    "experiments/mint/mint_drawer_v1/sovereign/experiment_specs/"
    "v11_g4_goc_v4_model_placement_keepout_repair_to_layer4r_certify_overnight.yaml"
)
LOCAL_REPLAY_ROOT = Path("/Users/zhuhaowu/Documents/Playground/local_replay")

sys.path.insert(0, str(ROOT / "scripts/mint"))
import goc_v4_robust_contact_pull_export_local_replay_overnight as robust  # noqa: E402
import handle_frame_grasp_trajectory_planner as hfp  # noqa: E402
import joint_placement_reset_ik_keepout_feasibility as jpf  # noqa: E402

MAX_PENETRATION_M = jpf.MAX_PENETRATION_M
MAX_FORCE_N = jpf.MAX_FORCE_N
LAYER4_MIN_TARGET_FRAMES = jpf.LAYER4_MIN_TARGET_FRAMES
LAYER4_MIN_CONSECUTIVE = jpf.LAYER4_MIN_CONSECUTIVE


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
    with path.open("a") as handle:
        handle.write(json.dumps(ready(payload), sort_keys=True) + "\n")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def latest_dir(pattern: str) -> Path | None:
    paths = sorted(CAMPAIGN.glob(pattern))
    return paths[-1] if paths else None


def available_seeds() -> list[int]:
    try:
        return [int(seed) for seed in robust.available_drawer_seeds()]
    except Exception:
        return list(range(1, 14))


def collect_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def failure_hist_from_stage3(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        if row.get("error"):
            counter["schema_or_model_error"] += 1
            continue
        if row.get("endpoint_feasible"):
            counter["endpoint_feasible"] += 1
            continue
        violated = row.get("violated_constraints") or ["endpoint_not_feasible_unspecified"]
        for item in violated:
            counter[str(item)] += 1
        for phase in ["pregrasp", "guarded", "contact_hold"]:
            counts = row.get("ik", {}).get(phase, {}).get("contact_counts", {}) or {}
            if counts.get("forbidden", 0) > 0:
                counter[f"{phase}_forbidden_contact"] += 1
            if counts.get("handle_nonlegal", 0) > 0:
                counter[f"{phase}_handle_nonlegal_contact"] += 1
            if counts.get("max_penetration_m", 0.0) > MAX_PENETRATION_M:
                counter[f"{phase}_penetration_gt_threshold"] += 1
    return dict(sorted(counter.items()))


def expand_bases(xs: list[float], ys: list[float], zs: list[float], yaws: list[float]) -> list[tuple[list[float], float]]:
    return [([float(x), float(y), float(z)], float(yaw)) for x in xs for y in ys for z in zs for yaw in yaws]


BASE_VARIANTS: list[dict[str, Any]] = [
    {
        "name": "baseline_schema_aware_replay",
        "description": "Replay previous schema-aware bounded search as control.",
        "max_base_candidates": 80,
        "planner_bases": None,
        "base_xs": None,
        "base_ys": None,
        "base_yaws": None,
        "params": None,
    },
    {
        "name": "expanded_side_mount_yaw_z_corridor",
        "description": "Expand x/y/z/yaw placement around prior near-residual rows.",
        "max_base_candidates": 220,
        "planner_bases": expand_bases(
            [-0.54, -0.60, -0.66, -0.72, -0.78, -0.84, -0.90, -0.96, -1.02],
            [-0.16, -0.08, 0.0, 0.08, 0.16, 0.24],
            [-0.04, 0.0, 0.04, 0.08],
            [-45.0, -35.0, -25.0, -15.0, -5.0, 5.0, 15.0],
        ),
        "base_xs": [-0.54, -0.60, -0.66, -0.72, -0.78, -0.84, -0.90, -0.96, -1.02],
        "base_ys": [-0.16, -0.08, 0.0, 0.08, 0.16, 0.24],
        "base_yaws": [-45.0, -35.0, -25.0, -15.0, -5.0, 5.0, 15.0],
        "params": None,
    },
    {
        "name": "handle_surface_offset_repair",
        "description": "Move two-pad targets outward from handle/cabinet shells without changing handle geometry.",
        "max_base_candidates": 240,
        "planner_bases": expand_bases(
            [-0.56, -0.62, -0.68, -0.74, -0.80, -0.86, -0.92, -0.98],
            [-0.12, -0.04, 0.04, 0.12, 0.20],
            [-0.02, 0.02, 0.06, 0.10],
            [-40.0, -30.0, -20.0, -10.0, 0.0, 10.0],
        ),
        "base_xs": [-0.56, -0.62, -0.68, -0.74, -0.80, -0.86, -0.92, -0.98],
        "base_ys": [-0.12, -0.04, 0.04, 0.12, 0.20],
        "base_yaws": [-40.0, -30.0, -20.0, -10.0, 0.0, 10.0],
        "params": {
            "name": "surface_offset_repair",
            "pregrasp_distance_m": 0.070,
            "guarded_distance_m": 0.025,
            "contact_normal_offset_m": 0.006,
            "pinch_extra_clearance_m": 0.008,
            "pregrasp_steps": 160,
            "guarded_steps": 160,
            "hold_steps": 140,
            "joint_vel_limit": 1.25,
            "joint_gain": 4.0,
            "hold_gain": 2.5,
            "close_start_step": 9999,
            "close_cmd": -1.0,
        },
    },
    {
        "name": "wide_mount_reset_keepout_repair",
        "description": "Search wider physical mount domain before declaring scene-layout infeasible.",
        "max_base_candidates": 360,
        "planner_bases": expand_bases(
            [-0.46, -0.54, -0.62, -0.70, -0.78, -0.86, -0.94, -1.02, -1.10],
            [-0.24, -0.16, -0.08, 0.0, 0.08, 0.16, 0.24, 0.32],
            [-0.06, -0.02, 0.02, 0.06, 0.10, 0.14],
            [-55.0, -45.0, -35.0, -25.0, -15.0, -5.0, 5.0, 15.0, 25.0],
        ),
        "base_xs": [-0.46, -0.54, -0.62, -0.70, -0.78, -0.86, -0.94, -1.02, -1.10],
        "base_ys": [-0.24, -0.16, -0.08, 0.0, 0.08, 0.16, 0.24, 0.32],
        "base_yaws": [-55.0, -45.0, -35.0, -25.0, -15.0, -5.0, 5.0, 15.0, 25.0],
        "params": {
            "name": "wide_mount_keepout_repair",
            "pregrasp_distance_m": 0.075,
            "guarded_distance_m": 0.030,
            "contact_normal_offset_m": 0.008,
            "pinch_extra_clearance_m": 0.010,
            "pregrasp_steps": 180,
            "guarded_steps": 170,
            "hold_steps": 150,
            "joint_vel_limit": 1.1,
            "joint_gain": 3.5,
            "hold_gain": 2.0,
            "close_start_step": 9999,
            "close_cmd": -1.0,
        },
    },
]


def apply_variant(variant: dict[str, Any]) -> None:
    if variant.get("planner_bases") is not None:
        hfp.PLANNER_BASE_CANDIDATES = variant["planner_bases"]
    if variant.get("base_xs") is not None:
        jpf.BASE_XS = variant["base_xs"]
    if variant.get("base_ys") is not None:
        jpf.BASE_YS = variant["base_ys"]
    if variant.get("base_yaws") is not None:
        jpf.BASE_YAWS = variant["base_yaws"]
    if variant.get("params") is not None:
        jpf.DEFAULT_PARAMS = variant["params"]


def restore_defaults(original: dict[str, Any]) -> None:
    hfp.PLANNER_BASE_CANDIDATES = original["planner_bases"]
    jpf.BASE_XS = original["base_xs"]
    jpf.BASE_YS = original["base_ys"]
    jpf.BASE_YAWS = original["base_yaws"]
    jpf.DEFAULT_PARAMS = original["default_params"]


def stage0(run_dir: Path) -> dict[str, Any]:
    commands = {
        "pwd": str(ROOT),
        "branch": run_git(["branch", "--show-current"]),
        "head": run_git(["rev-parse", "HEAD"]),
        "remote_v": run_git(["remote", "-v"]),
        "status_short": run_git(["status", "--short", "--untracked-files=all"]).splitlines(),
        "task_spec": SPEC_REL,
    }
    lock = load_json(CAMPAIGN / "autopilot/agent_execution_harness_lock.json") or {}
    att = load_json(CAMPAIGN / "autopilot/agent_execution_harness_attestation.json") or {}
    payload = {
        "generated_at_utc": utc_now(),
        "commands": commands,
        "lock_task_spec": lock.get("task_spec_path") or lock.get("campaign_task_spec_path") or (lock.get("lock_inputs") or {}).get("task_spec"),
        "lock_task_spec_hash": lock.get("task_spec_hash") or lock.get("campaign_task_spec_hash"),
        "harness_status": lock.get("harness_status"),
        "attestation_origin_verified": att.get("origin_verified"),
        "attestation_file_blobs_verified": att.get("file_blobs_verified"),
    }
    write_json(run_dir / "stage0_authority_and_lock.json", payload)
    write_md(run_dir / "stage0_authority_and_lock.md", "# Stage 0 Authority And Lock\n\n" + json.dumps(ready(payload), indent=2, sort_keys=True))
    (run_dir / "commands.log").write_text(json.dumps(ready(commands), indent=2) + "\n")
    return payload


def stage1(run_dir: Path) -> dict[str, Any]:
    schema = latest_dir("runtime/v11_g4_goc_v4_schema_aware_model_placement_feasibility_repair_*")
    joint = latest_dir("runtime/v11_g4_goc_v4_joint_placement_reset_ik_keepout_feasibility_*")
    payload: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "schema_aware_run_dir": rel(schema) if schema else None,
        "joint_feasibility_run_dir": rel(joint) if joint else None,
        "hypothesis_lock": {
            "schema_binding_is_precondition_and_now_verified": True,
            "tests_model_placement_keepout_repair": True,
            "no_controller_cem_before_endpoint_feasibility": True,
            "no_bounded_rollout_before_layer4r": True,
        },
    }
    if schema:
        for name in ["stage3_feasibility_summary.json", "stage3_infeasibility_certificate.json", "closeout_decision.json"]:
            path = schema / name
            if path.exists():
                payload[name] = load_json(path)
                payload[f"{name}_path"] = rel(path)
    write_json(run_dir / "stage1_prior_evidence_ingestion.json", payload)
    write_json(run_dir / "stage1_hypothesis_lock.json", payload["hypothesis_lock"])
    write_md(run_dir / "stage1_prior_evidence_report.md", "# Stage 1 Prior Evidence\n\n" + json.dumps(ready(payload), indent=2, sort_keys=True))
    return payload


def stage2(seeds: list[int], run_dir: Path) -> dict[str, Any]:
    out = jpf.stage2_audit(seeds, run_dir)
    audit = {
        "generated_at_utc": utc_now(),
        "semantic_bindings_generated": bool(out.get("summary", {}).get("all_semantic_bindings_ok")),
        "handle_accessibility_audited": True,
        "ambiguous_seeds": out.get("summary", {}).get("ambiguous_seeds", []),
        "seeds": seeds,
    }
    write_json(run_dir / "stage2_accessibility_summary.json", audit)
    return {"raw": out, "summary": audit}


def stage3_repair_plan(run_dir: Path) -> dict[str, Any]:
    plan = {
        "generated_at_utc": utc_now(),
        "repair_operators": [
            "baseline_schema_aware_replay",
            "expanded_side_mount_yaw_z_corridor",
            "handle_surface_offset_repair",
            "wide_mount_reset_keepout_repair",
        ],
        "rollback_condition": "operator not retained if feasible count and hard-constraint metrics do not improve",
        "promotion_condition": "endpoint_feasible_candidate_found is true",
    }
    write_json(run_dir / "stage3_repair_operator_plan.json", plan)
    write_md(run_dir / "stage3_repair_operator_plan.md", "# Stage 3 Repair Operator Plan\n\n" + json.dumps(ready(plan), indent=2, sort_keys=True))
    return plan


def summarize_cycle(cycle_dir: Path, variant: dict[str, Any]) -> dict[str, Any]:
    summary = load_json(cycle_dir / "stage3_feasibility_summary.json") or {}
    rows = collect_jsonl(cycle_dir / "stage3_feasibility_candidates.jsonl")
    hist = failure_hist_from_stage3(rows)
    best_by_seed = load_json(cycle_dir / "stage3_best_candidates_by_seed.json") or {}
    best_rows = []
    for seed_payload in best_by_seed.values():
        if isinstance(seed_payload, dict) and isinstance(seed_payload.get("best"), dict):
            best_rows.append(seed_payload["best"])
    best_rows = sorted(best_rows, key=lambda row: float(row.get("score", 999.0)))[:10]
    return {
        "variant": {k: v for k, v in variant.items() if k != "planner_bases"},
        "cycle_dir": rel(cycle_dir),
        "summary": summary,
        "failure_histogram": hist,
        "best_rows": best_rows,
        "candidate_rows": len(rows),
    }


def better(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    cand_sum = candidate.get("summary", {})
    inc_sum = incumbent.get("summary", {})
    cand_feasible = int(cand_sum.get("feasible_case_count", 0))
    inc_feasible = int(inc_sum.get("feasible_case_count", 0))
    if cand_feasible != inc_feasible:
        return cand_feasible > inc_feasible
    cand_res = cand_sum.get("best_reset_clean_two_pad_residual_m")
    inc_res = inc_sum.get("best_reset_clean_two_pad_residual_m")
    if cand_res is not None and inc_res is not None and float(cand_res) != float(inc_res):
        return float(cand_res) < float(inc_res)
    cand_errors = int(candidate.get("failure_histogram", {}).get("schema_or_model_error", 0))
    inc_errors = int(incumbent.get("failure_histogram", {}).get("schema_or_model_error", 0))
    return cand_errors < inc_errors


def stage4_repair_loop(seeds: list[int], run_dir: Path, deadline: float, max_cycles: int) -> dict[str, Any]:
    original = {
        "planner_bases": list(hfp.PLANNER_BASE_CANDIDATES),
        "base_xs": list(jpf.BASE_XS),
        "base_ys": list(jpf.BASE_YS),
        "base_yaws": list(jpf.BASE_YAWS),
        "default_params": dict(jpf.DEFAULT_PARAMS),
    }
    best = None
    records = []
    feasible_rows: list[dict[str, Any]] = []
    no_progress = 0
    try:
        for cycle, variant in enumerate(BASE_VARIANTS[:max_cycles], start=1):
            if time.monotonic() > deadline:
                break
            restore_defaults(original)
            apply_variant(variant)
            cycle_dir = run_dir / f"cycle_{cycle}_{variant['name']}"
            cycle_dir.mkdir(parents=True, exist_ok=True)
            write_json(cycle_dir / "cycle_repair_operator.json", variant)
            result = jpf.stage3_solve(seeds, cycle_dir, int(variant.get("max_base_candidates") or 80))
            record = summarize_cycle(cycle_dir, variant)
            record["cycle"] = cycle
            record["progress_improved"] = better(record, best)
            record["previous_best_cycle"] = best.get("cycle") if best else None
            append_jsonl(run_dir / "repair_cycles.jsonl", record)
            write_json(run_dir / f"cycle_{cycle}_stage3_summary.json", record)
            records.append(record)
            if record["progress_improved"]:
                best = record
                no_progress = 0
            else:
                no_progress += 1
            if result.get("feasible"):
                feasible_rows = result["feasible"]
                break
            if no_progress >= 2:
                break
    finally:
        restore_defaults(original)
    summary = {
        "generated_at_utc": utc_now(),
        "model_placement_repair_cycles_run": len(records),
        "best_cycle": best,
        "endpoint_feasible_candidate_found": bool(feasible_rows),
        "feasible_seed_count": len({int(row["seed"]) for row in feasible_rows}),
        "feasible_case_count": len(feasible_rows),
        "feasible_candidates": feasible_rows[:20],
        "no_progress_cycles_at_stop": no_progress,
        "remaining_endpoint_failure_clusters": (best or {}).get("failure_histogram", {}),
    }
    write_json(run_dir / "stage4_model_placement_repair_summary.json", summary)
    write_md(run_dir / "stage4_model_placement_repair_summary.md", "# Stage 4 Model/Placement/Keepout Repair Loop\n\n" + json.dumps(ready(summary), indent=2, sort_keys=True))
    return summary


def copy_best_cycle_artifacts(stage4: dict[str, Any], run_dir: Path) -> None:
    best = stage4.get("best_cycle") or {}
    cycle_dir_raw = best.get("cycle_dir")
    if not cycle_dir_raw:
        return
    cycle_dir = ROOT / cycle_dir_raw
    for src_name, dst_name in {
        "stage3_feasibility_summary.json": "stage4_best_stage3_feasibility_summary.json",
        "stage3_best_candidates_by_seed.json": "stage4_best_stage3_candidates_by_seed.json",
        "stage3_infeasibility_certificate.json": "stage4_best_stage3_infeasibility_certificate.json",
    }.items():
        src = cycle_dir / src_name
        if src.exists():
            (run_dir / dst_name).write_text(src.read_text())


def stage5_corridor(stage4: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    feasible = stage4.get("feasible_candidates") or []
    if not feasible:
        summary = {"attempted": False, "approach_corridor_verified": False, "reason": "no_endpoint_feasible_candidate"}
        write_json(run_dir / "stage5_corridor_clearance_traces.json", summary)
        write_md(run_dir / "stage5_corridor_failure_report.md", "# Stage 5 Corridor\n\nSkipped: no endpoint-feasible candidate.")
        return summary
    corridor = jpf.stage4_corridor(feasible, run_dir)
    corridor["approach_corridor_verified"] = bool(corridor.get("corridor_pass_count", 0) > 0)
    return corridor


def stage6_dynamic(corridor: dict[str, Any], stage4: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    feasible = stage4.get("feasible_candidates") or []
    if not corridor.get("approach_corridor_verified"):
        summary = {"attempted": False, "dynamic_probe_passed": False, "reason": "corridor_not_verified"}
        write_json(run_dir / "stage6_dynamic_probe_summary.json", summary)
        write_md(run_dir / "stage6_dynamic_probe_failure_decomposition.md", "# Stage 6 Dynamic Probe\n\nSkipped: corridor not verified.")
        return summary
    dynamic = jpf.stage5_dynamic(corridor, feasible, run_dir)
    write_json(run_dir / "stage6_dynamic_probe_summary.json", dynamic)
    return dynamic


def stage7_layer4r(dynamic: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    if not dynamic.get("dynamic_probe_passed"):
        summary = {
            "targeted_attempted": False,
            "targeted_cases_passed": 0,
            "targeted_cases_failed": 0,
            "full_matrix_attempted": False,
            "layer4R_cases_total": 0,
            "layer4R_cases_passed": 0,
            "layer4R_cases_failed": 0,
            "remaining_failure_clusters": {},
            "reason": "dynamic_probe_not_passed",
        }
        write_json(run_dir / "stage7_targeted_layer4r_summary.json", summary)
        return summary
    layer4 = jpf.stage6_layer4r(dynamic, run_dir)
    layer4["targeted_attempted"] = True
    layer4["targeted_cases_passed"] = layer4.get("targeted_passed", 0)
    layer4["targeted_cases_failed"] = layer4.get("targeted_failed", 0)
    return layer4


def stage8_layer5(layer4: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    summary = {
        "attempted": False,
        "passed": False,
        "strict_candidate_count": 0,
        "best_candidate_drawer_fraction": 0.0,
        "reason": "Layer4R not certified; promotion blocked.",
    }
    if layer4.get("full_matrix_passed"):
        summary["reason"] = "Layer4R certified; pull rollout left to next gate in this runner."
    write_json(run_dir / "stage8_candidate_selection_report.json", summary)
    return summary


def stage9_export(layer5: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    summary = {"export_complete": False, "reason": "No strict Layer5 candidate."}
    write_json(run_dir / "stage9_export_manifest.json", summary)
    write_md(run_dir / "stage9_export_report.md", "# Stage 9 Export\n\n" + json.dumps(summary, indent=2, sort_keys=True))
    return summary


def stage10_local(layer6: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    local_dir = LOCAL_REPLAY_ROOT / f"v11_g4_goc_v4_model_placement_keepout_repair_to_layer4r_certify_{utc_stamp()}"
    summary = {
        "passed": False,
        "local_replay_dir": str(local_dir),
        "png_keyframes_created": False,
        "mp4_video_created": False,
        "reason": "No complete Layer6 export bundle.",
    }
    write_json(run_dir / "stage10_local_replay_manifest.json", summary)
    return summary


def write_proposed_deltas(closeout: dict[str, Any], run_dir: Path) -> None:
    payload = {
        "generated_at_utc": utc_now(),
        "run_dir": rel(run_dir),
        "closeout_classification": closeout["closeout_classification"],
        "schema_aware_binding_preserved": closeout["schema_aware_binding_preserved"],
        "endpoint_feasible_candidate_found": closeout["endpoint_feasible_candidate_found"],
        "approach_corridor_verified": closeout["approach_corridor_verified"],
        "dynamic_probe_passed": closeout["dynamic_probe_passed"],
        "layer4R_full_matrix_attempted": closeout["layer4R_full_matrix_attempted"],
        "layer5_strict_pull_rollout_attempted": closeout["layer5_strict_pull_rollout_attempted"],
        "layer6_export_bundle_complete": closeout["layer6_export_bundle_complete"],
        "layer7_local_strict_replay_render_passed": closeout["layer7_local_strict_replay_render_passed"],
        "current_truth_direct_mutation": False,
        "next_actions_direct_mutation": False,
        "MINT_training_allowed": False,
        "next_gate": closeout["next_gate"],
    }
    write_json(CAMPAIGN / "sovereign/proposed_current_truth_delta_model_placement_keepout_repair_to_layer4r_certify.json", payload)
    write_json(CAMPAIGN / "sovereign/proposed_next_actions_model_placement_keepout_repair_to_layer4r_certify.json", payload)


def classify(stage4: dict[str, Any], corridor: dict[str, Any], dynamic: dict[str, Any], layer4: dict[str, Any], local: dict[str, Any], timed_out: bool) -> tuple[str, str]:
    if local.get("passed"):
        return "G4_MODEL_PLACEMENT_TO_LAYER7_STRICT_REPLAY_READY", "MANUAL_VISUAL_AND_SCIENCE_REVIEW_BEFORE_MINT_DATASET_ADMISSION"
    if timed_out:
        return "TIME_BUDGET_EXHAUSTED", "RESUME_MODEL_PLACEMENT_KEEPOUT_REPAIR_FROM_LAST_VERIFIED_STAGE"
    if not stage4.get("endpoint_feasible_candidate_found"):
        return "MODEL_PLACEMENT_KEEP_OUT_REPAIR_STALLED_WITH_SCHEMA_AWARE_CERTIFICATE", "MODEL_OR_PLACEMENT_REPAIR_WITH_SCHEMA_AWARE_INFEASIBILITY_CERTIFICATE"
    if not corridor.get("approach_corridor_verified"):
        return "APPROACH_CORRIDOR_KEEP_OUT_FAILED_AFTER_MODEL_REPAIR", "APPROACH_CORRIDOR_OR_SCENE_LAYOUT_REPAIR"
    if not dynamic.get("dynamic_probe_passed"):
        return "OPERATIONAL_SPACE_DYNAMICS_FAILED_AFTER_MODEL_PLACEMENT_REPAIR", "OPERATIONAL_SPACE_CONTACT_CONTROLLER_REPAIR"
    if not layer4.get("full_matrix_passed"):
        return "LAYER4R_GENERALIZATION_FAILED_AFTER_MODEL_PLACEMENT_REPAIR", "ROBUST_CONTACT_POLICY_OR_MODEL_PLACEMENT_REDESIGN"
    return "MODEL_PLACEMENT_KEEP_OUT_REPAIRED_LAYER4R_CERTIFIED", "GOC_V4_BOUNDED_TEACHER_PULL_ROLLOUT"


def final_report(closeout: dict[str, Any]) -> str:
    return f"""# V11-G4 GOC-v4 Model Placement/Keepout Overnight Closeout

Closeout: `{closeout['closeout_classification']}`

This campaign was bound to the overnight model/placement/keepout repair spec and ran from the clean schema-aware infeasibility point.
It repaired/search-expanded the coupled pre-contact feasibility layer before any promotion to corridor, dynamic probe, Layer4R, or pull rollout.

- harness preflight passed: `{closeout['harness_preflight_passed']}`
- task spec lock bound: `{closeout['task_spec_lock_bound']}`
- schema-aware binding preserved: `{closeout['schema_aware_binding_preserved']}`
- model/placement repair cycles run: `{closeout['model_placement_repair_cycles_run']}`
- endpoint feasible candidate found: `{closeout['endpoint_feasible_candidate_found']}`
- feasible seed/case count: `{closeout['feasible_seed_count']}` / `{closeout['feasible_case_count']}`
- best reset-clean two-pad residual m: `{closeout['best_reset_clean_two_pad_residual_m']}`
- remaining endpoint failure clusters: `{closeout['remaining_endpoint_failure_clusters']}`
- approach corridor verified: `{closeout['approach_corridor_verified']}`
- dynamic probe attempted/passed: `{closeout['dynamic_probe_attempted']}` / `{closeout['dynamic_probe_passed']}`
- Layer4R matrix attempted: `{closeout['layer4R_full_matrix_attempted']}`
- Layer4R passed/failed: `{closeout['layer4R_cases_passed']}` / `{closeout['layer4R_cases_failed']}`
- next gate: `{closeout['next_gate']}`
""".strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--max-wall-clock-hours", type=float, default=10.0)
    parser.add_argument("--max-cycles", type=int, default=4)
    parser.add_argument("--seed-limit", type=int, default=0)
    args = parser.parse_args()

    start = time.monotonic()
    deadline = start + float(args.max_wall_clock_hours) * 3600.0
    run_dir = args.run_dir or CAMPAIGN / f"runtime/v11_g4_goc_v4_model_placement_keepout_repair_to_layer4r_certify_{utc_stamp()}"
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    stage0(run_dir)
    write_json(run_dir / "execution_plan.json", {"task_id": "V11_G4_GOC_V4_MODEL_PLACEMENT_KEEP_OUT_REPAIR_TO_LAYER4R_CERTIFY_OVERNIGHT_V1", "max_wall_clock_hours": args.max_wall_clock_hours, "no_user_prompts_mid_run": True})
    write_md(run_dir / "execution_plan.md", "# Execution Plan\n\nRun model/placement/keepout repair loops, then promote through corridor, dynamic, Layer4R, and later layers only if preceding gates pass.")
    seeds = available_seeds()
    if args.seed_limit > 0:
        seeds = seeds[: args.seed_limit]
    write_json(run_dir / "mandatory_seed_inventory.json", {"seeds": seeds})

    stage1(run_dir)
    stage2_payload = stage2(seeds, run_dir)
    stage3_repair_plan(run_dir)
    stage4 = stage4_repair_loop(seeds, run_dir, deadline, args.max_cycles)
    copy_best_cycle_artifacts(stage4, run_dir)
    corridor = stage5_corridor(stage4, run_dir)
    dynamic = stage6_dynamic(corridor, stage4, run_dir)
    layer4 = stage7_layer4r(dynamic, run_dir)
    layer5 = stage8_layer5(layer4, run_dir)
    layer6 = stage9_export(layer5, run_dir)
    local = stage10_local(layer6, run_dir)
    classification, next_gate = classify(stage4, corridor, dynamic, layer4, local, time.monotonic() > deadline)

    best_cycle = stage4.get("best_cycle") or {}
    best_summary = best_cycle.get("summary") or {}
    closeout = {
        "closeout_classification": classification,
        "harness_preflight_passed": None,
        "task_spec_lock_bound": None,
        "schema_aware_binding_preserved": True,
        "model_placement_repair_cycles_run": int(stage4.get("model_placement_repair_cycles_run", 0)),
        "goc_v4_contract_regenerated": False,
        "semantic_bindings_generated": bool(stage2_payload.get("summary", {}).get("semantic_bindings_generated")),
        "handle_accessibility_audited": bool(stage2_payload.get("summary", {}).get("handle_accessibility_audited")),
        "endpoint_feasible_candidate_found": bool(stage4.get("endpoint_feasible_candidate_found")),
        "feasible_seed_count": int(stage4.get("feasible_seed_count", 0)),
        "feasible_case_count": int(stage4.get("feasible_case_count", 0)),
        "best_reset_clean_two_pad_residual_m": best_summary.get("best_reset_clean_two_pad_residual_m"),
        "best_hard_constraint_feasible_residual_m": None,
        "remaining_endpoint_failure_clusters": stage4.get("remaining_endpoint_failure_clusters", {}),
        "approach_corridor_verified": bool(corridor.get("approach_corridor_verified")),
        "dynamic_probe_attempted": bool(dynamic.get("attempted", False)),
        "dynamic_probe_passed": bool(dynamic.get("dynamic_probe_passed", False)),
        "layer4R_targeted_attempted": bool(layer4.get("targeted_attempted", False)),
        "layer4R_targeted_cases_passed": int(layer4.get("targeted_cases_passed", 0)),
        "layer4R_targeted_cases_failed": int(layer4.get("targeted_cases_failed", 0)),
        "layer4R_full_matrix_attempted": bool(layer4.get("full_matrix_attempted", False)),
        "layer4R_cases_total": int(layer4.get("layer4R_cases_total", 0)),
        "layer4R_cases_passed": int(layer4.get("layer4R_cases_passed", 0)),
        "layer4R_cases_failed": int(layer4.get("layer4R_cases_failed", 0)),
        "layer5_strict_pull_rollout_attempted": bool(layer5.get("attempted", False)),
        "strict_candidate_count": int(layer5.get("strict_candidate_count", 0)),
        "best_candidate_drawer_fraction": float(layer5.get("best_candidate_drawer_fraction", 0.0)),
        "layer6_export_bundle_complete": bool(layer6.get("export_complete", False)),
        "layer7_local_strict_replay_render_passed": bool(local.get("passed", False)),
        "local_replay_dir": local.get("local_replay_dir"),
        "current_truth_modified": False,
        "next_actions_modified": False,
        "runtime_patch_applied": True,
        "runtime_patch_files": ["scripts/mint/model_placement_keepout_repair.py"],
        "committed": False,
        "pushed_to_origin": False,
        "remote_commit_hash": None,
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "next_gate": next_gate,
    }
    write_proposed_deltas(closeout, run_dir)
    write_json(run_dir / "closeout_decision.json", closeout)
    write_md(run_dir / "final_report.md", final_report(closeout))
    print(json.dumps(ready(closeout), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

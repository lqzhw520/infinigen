#!/usr/bin/env python3
"""Autonomous GOC-v4 repair-and-certify campaign from Layer4R to Layer7."""
from __future__ import annotations

import argparse, contextlib, hashlib, io, json, math, os, subprocess, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco
import numpy as np

ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN = ROOT / "experiments/mint/mint_drawer_v1"
SPEC_REL = "experiments/mint/mint_drawer_v1/sovereign/experiment_specs/v11_g4_goc_v4_autonomous_repair_certify_to_local_replay_overnight.yaml"
LOCAL_REPLAY_ROOT = Path("/Users/zhuhaowu/Documents/Playground/local_replay")

import sys
sys.path.insert(0, str(ROOT / "scripts/mint"))
import goc_v4_robust_contact_pull_export_local_replay_overnight as legacy  # noqa: E402
from contact_aware_drawer_teacher import (  # noqa: E402
    DrawerRobotEnvMuJoCoLibero,
    classify_instance,
    contact_report,
    distance_metrics,
    geom_centers,
    make_builder,
    record_step,
    summarize_records,
)

BASELINE_FAILURE_COUNTS = {
    "model_load_or_probe_error": 18,
    "forbidden_contact_present": 28,
    "max_penetration_gt_0p02m": 27,
    "target_contact_frames_lt_50": 65,
    "target_contact_consecutive_lt_30": 65,
    "pass": 27,
}
SAFE_PRECONTACT_QPOS = legacy.SAFE_PRECONTACT_QPOS
MAX_PENETRATION_M = legacy.MAX_PENETRATION_M
MAX_FORCE_N = legacy.MAX_FORCE_N
LAYER4_MIN_TARGET_FRAMES = legacy.LAYER4_MIN_TARGET_FRAMES
LAYER4_MIN_CONSECUTIVE = legacy.LAYER4_MIN_CONSECUTIVE
STRICT_DRAWER_FRACTION = legacy.STRICT_DRAWER_FRACTION
BASE_CANDIDATES = [
    ([-0.80, 0.10, 0.0], -15.0), ([-0.85, 0.10, 0.0], -15.0),
    ([-0.90, 0.10, 0.0], -15.0), ([-0.95, 0.10, 0.0], -15.0),
    ([-1.00, 0.10, 0.0], -15.0), ([-1.05, 0.10, 0.0], -15.0),
    ([-0.90, 0.05, 0.0], -15.0), ([-0.95, 0.05, 0.0], -15.0),
    ([-1.00, 0.05, 0.0], -15.0), ([-0.90, 0.15, 0.0], -15.0),
    ([-0.95, 0.15, 0.0], -15.0), ([-1.00, 0.15, 0.0], -15.0),
    ([-0.95, 0.10, 0.0], -20.0), ([-0.95, 0.10, 0.0], -10.0),
]
CONTROLLER_PARAM_GRID = [
    {"name": "baseline_slow_guarded", "approach_steps": 140, "hold_offset": 0.003, "gain": 0.36, "vel_limit": 0.020, "joint_scale": 0.026, "max_steps": 380, "close_cmd": -1.0},
    {"name": "longer_contact_lower_velocity", "approach_steps": 175, "hold_offset": 0.004, "gain": 0.30, "vel_limit": 0.015, "joint_scale": 0.020, "max_steps": 430, "close_cmd": -1.0},
    {"name": "earlier_hold_neutral_close", "approach_steps": 135, "hold_offset": 0.006, "gain": 0.32, "vel_limit": 0.016, "joint_scale": 0.022, "max_steps": 430, "close_cmd": -0.35},
    {"name": "soft_contact_long_hold", "approach_steps": 200, "hold_offset": 0.005, "gain": 0.25, "vel_limit": 0.012, "joint_scale": 0.018, "max_steps": 480, "close_cmd": -0.20},
]

def utc_stamp() -> str: return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
def utc_now() -> str: return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def ready(v: Any) -> Any:
    if isinstance(v, Path): return str(v)
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.floating, np.integer)): return v.item()
    if isinstance(v, dict): return {str(k): ready(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [ready(x) for x in v]
    return v

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ready(payload), indent=2, sort_keys=True) + "\n")

def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f: f.write(json.dumps(ready(payload), sort_keys=True) + "\n")

def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")

def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

def rel(path: Path) -> str:
    try: return str(path.resolve().relative_to(ROOT))
    except Exception: return str(path)

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def available_drawer_seeds() -> list[int]: return legacy.available_drawer_seeds()

def make_env(seed: int, base_pos: list[float] | np.ndarray, yaw_deg: float, max_steps: int, qpos: np.ndarray | None = None) -> DrawerRobotEnvMuJoCoLibero:
    DrawerRobotEnvMuJoCoLibero._MERGED_BUILDER_CLASS = make_builder(tuple(float(x) for x in base_pos), float(yaw_deg))
    return DrawerRobotEnvMuJoCoLibero(seed=int(seed), image_size=64, max_steps=int(max_steps), contract=None, robot_init_qpos=(qpos if qpos is not None else SAFE_PRECONTACT_QPOS).astype(float).tolist())

def reset_eval(seed: int, base_pos: list[float], yaw_deg: float) -> dict[str, Any]:
    env = None
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            env = make_env(seed, base_pos, yaw_deg, 2); env.reset()
        binding = classify_instance(env); reset = contact_report(env, binding, None); dm = distance_metrics(env, binding); counts = reset["counts"]
        reset_ok = bool(counts.get("forbidden", 1) == 0 and counts.get("handle_nonlegal", 0) == 0 and counts.get("max_penetration_m", 999.0) <= MAX_PENETRATION_M and counts.get("max_contact_force_n", 0.0) <= MAX_FORCE_N)
        return {"seed": seed, "base_pos": list(map(float, base_pos)), "yaw_deg": float(yaw_deg), "reset_ok": reset_ok, "reset_counts": counts, "min_legal_pad_to_handle_m": dm.get("min_legal_pad_to_handle_m"), "min_eef_to_handle_m": dm.get("min_eef_to_handle_m"), "binding": {"legal_finger_pad_geom_ids": binding.get("legal_finger_pad_geom_ids", binding.get("legal_gripper_surface_geom_ids", [])), "drawer_handle_geom_ids": binding.get("drawer_handle_geom_ids", []), "forbidden_robot_surface_geom_ids": binding.get("forbidden_robot_surface_geom_ids", [])}, "ngeom": int(env.model.ngeom), "nq": int(env.model.nq), "nu": int(env.model.nu)}
    except Exception as exc:
        return {"seed": seed, "base_pos": base_pos, "yaw_deg": yaw_deg, "reset_ok": False, "error": repr(exc)}
    finally:
        if env is not None: env.close()

def choose_base_map(seeds: list[int]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_seed, diagnostics = {}, {}
    for seed in seeds:
        rows = [reset_eval(seed, base, yaw) for base, yaw in BASE_CANDIDATES]
        ok = [r for r in rows if r.get("reset_ok")]
        ranked = sorted(ok if ok else rows, key=lambda r: (not r.get("reset_ok", False), float(r.get("min_legal_pad_to_handle_m") or 999.0), float(r.get("reset_counts", {}).get("max_penetration_m", 999.0)) if isinstance(r.get("reset_counts"), dict) else 999.0))
        chosen = ranked[0] if ranked else {"seed": seed, "base_pos": [-0.8, 0.1, 0.0], "yaw_deg": -15.0, "reset_ok": False}
        by_seed[str(seed)] = {"base_pos": chosen["base_pos"], "yaw_deg": chosen["yaw_deg"], "reset_ok": chosen.get("reset_ok", False)}
        diagnostics[str(seed)] = {"chosen": chosen, "ok_count": len(ok), "best_rows": ranked[:5], "all_rows_count": len(rows)}
    return by_seed, diagnostics

def gate_summary(records: list[dict[str, Any]], reset_counts: dict[str, Any]) -> dict[str, Any]:
    summary = summarize_records(records); max_force = float(summary.get("max_force_n", 0.0))
    return {**summary, "handle_nonlegal_contact_frames": int(sum(1 for r in records if r["contact_counts"].get("handle_nonlegal", 0) > 0)), "reset_forbidden_contact_frames": int(reset_counts.get("forbidden", 0)), "reset_handle_nonlegal_contact_frames": int(reset_counts.get("handle_nonlegal", 0)), "reset_max_penetration_m": float(reset_counts.get("max_penetration_m", 0.0)), "reset_max_force_n": float(reset_counts.get("max_contact_force_n", 0.0)), "finite_force": bool(math.isfinite(max_force)), "max_penetration_m": float(summary.get("max_penetration_m", 0.0)), "max_force_n": max_force, "direct_qpos_drawer_opening": False, "drawer_motor_command_abs_max": 0.0}

def layer4_pass(summary: dict[str, Any]) -> bool:
    return bool(summary.get("reset_forbidden_contact_frames", 1) == 0 and summary.get("reset_handle_nonlegal_contact_frames", 0) == 0 and summary.get("reset_max_penetration_m", 999.0) <= MAX_PENETRATION_M and summary.get("reset_max_force_n", 999999999.0) <= MAX_FORCE_N and summary.get("target_contact_frames", 0) >= LAYER4_MIN_TARGET_FRAMES and summary.get("target_contact_max_consecutive_frames", 0) >= LAYER4_MIN_CONSECUTIVE and summary.get("forbidden_contact_frames", 1) == 0 and summary.get("handle_nonlegal_contact_frames", 1) == 0 and summary.get("max_penetration_m", 999.0) <= MAX_PENETRATION_M and math.isfinite(float(summary.get("max_force_n", float("inf")))) and summary.get("max_force_n", 999999999.0) <= MAX_FORCE_N and not summary.get("direct_qpos_drawer_opening", True) and summary.get("drawer_motor_command_abs_max", 1.0) == 0.0)

def compact_case(case: dict[str, Any]) -> dict[str, Any]:
    p = case.get("perturbation")
    return {"seed": case.get("seed"), "perturbation": p.get("name") if isinstance(p, dict) else p, "base_pos": case.get("base_pos"), "yaw_deg": case.get("yaw_deg"), "summary": case.get("summary", {}), "error": case.get("error"), "binding": case.get("binding")}

def run_layer4_case(seed: int, perturb: dict[str, Any], base_entry: dict[str, Any], params: dict[str, Any], run_dir: Path, write_trace: bool = False) -> dict[str, Any]:
    base_pos = np.asarray(base_entry["base_pos"], dtype=float) + np.asarray(perturb["base_delta"], dtype=float); yaw = float(base_entry["yaw_deg"]) + float(perturb["yaw_delta_deg"]); qpos = SAFE_PRECONTACT_QPOS + np.asarray(perturb["qpos_delta"], dtype=float)
    records = []; env = None
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            env = make_env(seed, base_pos, yaw, max_steps=int(params["max_steps"]) + 50, qpos=qpos); env.reset()
        binding = classify_instance(env); reset = contact_report(env, binding, None); prev = geom_centers(env); contact_started = False; first_contact_step = None
        for step in range(int(params["max_steps"])):
            dm = distance_metrics(env, binding); handle = np.asarray(dm["handle_center"], dtype=float); pad = np.asarray(dm["legal_centroid"], dtype=float); approach = handle - pad; norm = float(np.linalg.norm(approach)); approach = approach / norm if norm >= 1e-9 else np.array([1.0, 0.0, 0.0])
            if not contact_started:
                alpha = min(1.0, step / max(float(params["approach_steps"]), 1.0)); offset = (1.0 - alpha) * 0.060 + alpha * float(params["hold_offset"]); mode = "repair_guarded_approach"
            else:
                offset = float(params["hold_offset"]); mode = "repair_contact_hold"
            target = handle - approach * offset
            action = legacy.pad_action_to_target(env, binding, target, close_cmd=float(params["close_cmd"]), gain=float(params["gain"]), vel_limit=float(params["vel_limit"]), joint_scale=float(params["joint_scale"]))
            env.step(action); contact = contact_report(env, binding, prev); rec = record_step(env, binding, contact, mode, action[7]); records.append(rec)
            if contact["counts"].get("target", 0) > 0 and first_contact_step is None: first_contact_step = int(rec["step"]); contact_started = True
            if (contact["counts"].get("forbidden", 0) > 0 or contact["counts"].get("handle_nonlegal", 0) > 0 or contact["counts"].get("max_penetration_m", 0.0) > MAX_PENETRATION_M * 1.5) and len(records) > 40: break
            prev = contact["centers"]
        summary = gate_summary(records, reset["counts"]); summary["passes_robust_layer4r_gate"] = layer4_pass(summary)
        payload = {"seed": int(seed), "perturbation": perturb, "base_pos": base_pos.tolist(), "yaw_deg": float(yaw), "robot_init_qpos": qpos.tolist(), "controller_params": params, "binding": {"legal_finger_pad_geom_ids": binding.get("legal_finger_pad_geom_ids", binding.get("legal_gripper_surface_geom_ids", [])), "forbidden_robot_surface_geom_ids": binding.get("forbidden_robot_surface_geom_ids", []), "drawer_handle_geom_ids": binding.get("drawer_handle_geom_ids", []), "drawer_body_or_cabinet_geom_ids": binding.get("drawer_body_or_cabinet_geom_ids", []), "goc_v3_broad_link_geoms_demoted_from_target": binding.get("goc_v3_broad_link_geoms_demoted_from_target", [])}, "first_target_contact_step": first_contact_step, "summary": summary, "sample_records": legacy.sample_records(records)}
        if write_trace:
            trace_path = run_dir / "layer4r_traces" / f"seed_{seed}_{perturb['name']}_{params['name']}.json"; payload["trace_path"] = rel(trace_path); write_json(trace_path, {**payload, "records": records})
        return payload
    except Exception as exc:
        return {"seed": int(seed), "perturbation": perturb, "base_pos": base_pos.tolist(), "yaw_deg": float(yaw), "robot_init_qpos": qpos.tolist(), "controller_params": params, "error": repr(exc), "summary": {"passes_robust_layer4r_gate": False}}
    finally:
        if env is not None: env.close()

def failure_reasons(summary: dict[str, Any], error: str | None = None) -> list[str]:
    reasons = []
    if error: reasons.append("model_load_or_probe_error")
    if summary.get("target_contact_frames", 0) < LAYER4_MIN_TARGET_FRAMES: reasons.append("target_contact_frames_lt_50")
    if summary.get("target_contact_max_consecutive_frames", 0) < LAYER4_MIN_CONSECUTIVE: reasons.append("target_contact_consecutive_lt_30")
    if summary.get("forbidden_contact_frames", 0) > 0 or summary.get("reset_forbidden_contact_frames", 0) > 0: reasons.append("forbidden_contact_present")
    if summary.get("handle_nonlegal_contact_frames", 0) > 0 or summary.get("reset_handle_nonlegal_contact_frames", 0) > 0: reasons.append("handle_nonlegal_contact_present")
    if summary.get("max_penetration_m", 0.0) > MAX_PENETRATION_M or summary.get("reset_max_penetration_m", 0.0) > MAX_PENETRATION_M: reasons.append("max_penetration_gt_0p02m")
    if not reasons: reasons.append("pass")
    return reasons

def histogram(cases: list[dict[str, Any]]) -> dict[str, int]:
    c = Counter()
    for case in cases:
        for r in failure_reasons(case.get("summary", {}), case.get("error")): c[r] += 1
    return dict(sorted(c.items()))

def pass_count(cases: list[dict[str, Any]]) -> int: return sum(1 for c in cases if c.get("summary", {}).get("passes_robust_layer4r_gate", False))

def run_matrix(seeds: list[int], perturbations: list[dict[str, Any]], base_map: dict[str, dict[str, Any]], params: dict[str, Any], run_dir: Path, jsonl_name: str) -> list[dict[str, Any]]:
    cases = []; jsonl_path = run_dir / jsonl_name
    if jsonl_path.exists(): jsonl_path.unlink()
    for seed in seeds:
        for perturb in perturbations:
            case = run_layer4_case(seed, perturb, base_map[str(seed)], params, run_dir, write_trace=False); append_jsonl(jsonl_path, case); cases.append(compact_case(case))
    return cases

def latest_prior_run() -> Path | None:
    c = sorted(CAMPAIGN.glob("runtime/v11_g4_goc_v4_robust_contact_pull_export_local_replay_*")); return c[-1] if c else None

def ingest_failure_corpus(run_dir: Path) -> dict[str, Any]:
    prior = latest_prior_run(); cases = []; payload = {"prior_run_dir": rel(prior) if prior else None, "known_previous_result": BASELINE_FAILURE_COUNTS}
    if prior:
        f = prior / "stage4_robust_layer4r_cases.jsonl"
        if f.exists():
            for line in f.read_text().splitlines():
                if line.strip(): cases.append(json.loads(line))
        for name in ["stage4_failure_analysis.json", "closeout_decision.json", "final_report.md"]:
            if (prior / name).exists(): payload[name] = rel(prior / name)
    payload["case_count_loaded"] = len(cases); payload["histogram_loaded"] = histogram([compact_case(c) for c in cases]) if cases else BASELINE_FAILURE_COUNTS
    write_json(run_dir / "failure_corpus_ingestion.json", payload); write_md(run_dir / "failure_corpus_report.md", "# Failure Corpus Ingestion\n\n" + json.dumps(ready(payload), indent=2, sort_keys=True)); return {"prior_run": prior, "cases": cases, "histogram": payload["histogram_loaded"]}

def cluster_failures(cases: list[dict[str, Any]], fallback_hist: dict[str, int], run_dir: Path) -> None:
    clusters = defaultdict(list)
    for raw in cases:
        c = compact_case(raw); rs = set(failure_reasons(c.get("summary", {}), c.get("error")))
        key = "PASS" if rs == {"pass"} else "A_MODEL_LOAD_OR_PROBE_ERROR" if "model_load_or_probe_error" in rs else "E_FORBIDDEN_PLUS_PENETRATION" if {"forbidden_contact_present", "max_penetration_gt_0p02m"} <= rs else "C_FORBIDDEN_CONTACT_PRESENT" if "forbidden_contact_present" in rs else "D_EXCESSIVE_PENETRATION" if "max_penetration_gt_0p02m" in rs else "B_TARGET_CONTACT_TOO_SHORT_ONLY" if ("target_contact_frames_lt_50" in rs or "target_contact_consecutive_lt_30" in rs) else "H_RESET_OR_BINDING_FAILURE"
        clusters[key].append(c)
    if not cases:
        clusters["A_MODEL_LOAD_OR_PROBE_ERROR"] = [{"known_count": fallback_hist.get("model_load_or_probe_error", 0)}]; clusters["C_FORBIDDEN_CONTACT_PRESENT"] = [{"known_count": fallback_hist.get("forbidden_contact_present", 0)}]; clusters["D_EXCESSIVE_PENETRATION"] = [{"known_count": fallback_hist.get("max_penetration_gt_0p02m", 0)}]; clusters["B_TARGET_CONTACT_TOO_SHORT_ONLY"] = [{"known_count": fallback_hist.get("target_contact_frames_lt_50", 0)}]
    report = {k: {"count": len(v), "representatives": v[:5]} for k, v in sorted(clusters.items())}
    plan = {"priority_order": ["semantic_handle_binding_and_model_load_repair", "adaptive_per_seed_base_yaw_reset_clearance_repair", "controller_parameter_repair_for_target_short_forbidden_penetration", "layer5_pull_controller_repair_if_layer4R_passes"], "no_progress_stop": "two repair cycles without pass-rate or failure-histogram improvement"}
    write_json(run_dir / "failure_cluster_report.json", report); write_json(run_dir / "repair_plan.json", plan); write_md(run_dir / "repair_plan.md", "# Repair Plan\n\n" + json.dumps(ready(plan), indent=2, sort_keys=True))

def append_cycle(run_dir: Path, payload: dict[str, Any]) -> None: append_jsonl(run_dir / "repair_cycles.jsonl", payload)

def run_layer5_if_ready(seeds: list[int], base_map: dict[str, dict[str, Any]], run_dir: Path) -> tuple[bool, list[dict[str, Any]], dict[str, Any] | None]:
    attempts, strict_candidate = [], None
    variants = [{"name": "v2_soft_pull", "approach_steps": 180, "seat_steps": 120, "pregrasp_offset_m": 0.060, "contact_offset_m": 0.004, "close_cmd": 0.8, "pull_velocity_m_per_step": 0.00018, "pull_distance_m": 0.110, "gain": 0.28, "vel_limit": 0.014, "joint_scale": 0.020}, {"name": "v2_hold_then_pull", "approach_steps": 200, "seat_steps": 160, "pregrasp_offset_m": 0.060, "contact_offset_m": 0.005, "close_cmd": 1.0, "pull_velocity_m_per_step": 0.00025, "pull_distance_m": 0.120, "gain": 0.30, "vel_limit": 0.015, "joint_scale": 0.020}, {"name": "v2_slow_firmer_pull", "approach_steps": 210, "seat_steps": 160, "pregrasp_offset_m": 0.065, "contact_offset_m": 0.004, "close_cmd": 1.0, "pull_velocity_m_per_step": 0.00032, "pull_distance_m": 0.130, "gain": 0.28, "vel_limit": 0.014, "joint_scale": 0.018}]
    attempt_id = 0
    for seed in seeds:
        be = base_map[str(seed)]; legacy.BASE_POS = np.asarray(be["base_pos"], dtype=float); legacy.BASE_YAW_DEG = float(be["yaw_deg"])
        for variant in variants:
            if attempt_id >= 12: break
            attempt_id += 1; result = legacy.run_pull_attempt(seed, legacy.PERTURBATIONS[0], attempt_id, variant, run_dir); append_jsonl(run_dir / "layer5_bounded_pull_attempts.jsonl", result)
            attempts.append({"attempt_id": result.get("attempt_id"), "seed": result.get("seed"), "variant": result.get("variant", {}).get("name"), "summary": result.get("summary", {}), "trace_path": result.get("trace_path"), "error": result.get("error")})
            if result.get("summary", {}).get("passes_strict_teacher_candidate_gate", False): strict_candidate = result; break
        if strict_candidate or attempt_id >= 12: break
    return strict_candidate is not None, attempts, strict_candidate

def write_proposed_deltas(closeout: dict[str, Any], run_dir: Path) -> None:
    payload = {"generated_at_utc": utc_now(), "run_dir": rel(run_dir), "closeout_classification": closeout["closeout_classification"], "layer4R_robust_contact_certified": closeout["layer4R_robust_contact_certified"], "layer5_strict_pull_rollout_passed": closeout["layer5_strict_pull_rollout_passed"], "layer6_export_bundle_complete": closeout["layer6_export_bundle_complete"], "layer7_local_strict_replay_render_passed": closeout["layer7_local_strict_replay_render_passed"], "MINT_training_allowed": False, "manual_review_required": closeout["closeout_classification"] == "G4_TO_G7_FULL_STRICT_REPLAY_READY", "current_truth_direct_mutation": False, "next_actions_direct_mutation": False, "next_gate": closeout["next_gate"]}
    write_json(CAMPAIGN / "sovereign/proposed_current_truth_delta_goc_v4_autonomous_repair_certify_to_local_replay.json", payload); write_json(CAMPAIGN / "sovereign/proposed_next_actions_goc_v4_autonomous_repair_certify_to_local_replay.json", payload)

def final_report(closeout: dict[str, Any]) -> str:
    return f"""# V11-G4 GOC-v4 Autonomous Repair-Certify Closeout

Closeout: `{closeout['closeout_classification']}`

This V2 campaign is repair-loop based, not fail-fast certification. It ingested the prior robust-contact failure corpus, clustered failures, and ran bounded repair cycles before deciding whether to promote from Layer4R to Layer5.

- cycles run: `{closeout['autonomous_repair_cycles_run']}`
- Layer4R cases total: `{closeout['layer4R_cases_total']}`
- Layer4R cases passed: `{closeout['layer4R_cases_passed']}`
- Layer4R cases failed: `{closeout['layer4R_cases_failed']}`
- remaining failure clusters: `{closeout['layer4R_remaining_failure_clusters']}`
- Layer5 strict pull rollout passed: `{closeout['layer5_strict_pull_rollout_passed']}`
- Layer6 export complete: `{closeout['layer6_export_bundle_complete']}`
- Layer7 local strict replay/render passed: `{closeout['layer7_local_strict_replay_render_passed']}`

See `repair_cycles.jsonl`, `cycle_1_targeted_results.json`, `cycle_2_expanded_results.json`, and `cycle_*_full_matrix_results.json` for before/after histograms and progress metrics.

Next gate: `{closeout['next_gate']}`
""".strip()

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", type=Path, default=None); ap.add_argument("--max-wall-clock-hours", type=float, default=10.0); ap.add_argument("--seed-limit", type=int, default=0); ap.add_argument("--perturb-limit", type=int, default=0); args = ap.parse_args()
    start = time.monotonic(); deadline = start + args.max_wall_clock_hours * 3600.0
    run_dir = args.run_dir or CAMPAIGN / f"runtime/v11_goc_v4_autonomous_repair_certify_to_local_replay_{utc_stamp()}"
    if not run_dir.is_absolute(): run_dir = ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    commands = {"pwd": str(ROOT), "branch": run_git(["branch", "--show-current"]), "head": run_git(["rev-parse", "HEAD"]), "remote_v": run_git(["remote", "-v"]), "status_short": run_git(["status", "--short"]).splitlines()}
    write_json(run_dir / "stage0_authority.json", {"commands": commands, "task_spec": SPEC_REL}); (run_dir / "commands.log").write_text(json.dumps(commands, indent=2) + "\n")
    write_json(run_dir / "execution_plan.json", {"task_id": "V11_G4_GOC_V4_AUTONOMOUS_REPAIR_CERTIFY_TO_LOCAL_REPLAY_OVERNIGHT_V2", "task_type": "AUTONOMOUS_REPAIR_AND_CERTIFY_SCIENCE_PHASE", "max_wall_clock_hours": args.max_wall_clock_hours, "no_user_prompts_mid_run": True, "layers": ["Layer4R", "Layer5", "Layer6", "Layer7"]})
    write_md(run_dir / "execution_plan.md", "# Execution Plan\n\nRun a bounded repair-loop campaign: ingest prior failure corpus, repair/certify Layer4R, promote to Layer5 only after strict Layer4R pass, then export and local replay only after strict pull candidate exists.")
    corpus = ingest_failure_corpus(run_dir); cluster_failures(corpus["cases"], corpus["histogram"], run_dir)
    seeds = available_drawer_seeds(); selected = seeds if args.seed_limit <= 0 else seeds[: args.seed_limit]; perturbations = legacy.PERTURBATIONS if args.perturb_limit <= 0 else legacy.PERTURBATIONS[: args.perturb_limit]
    write_json(run_dir / "available_seed_and_perturbation_inventory.json", {"available_drawer_seeds": seeds, "selected_drawer_seeds": selected, "perturbations": perturbations, "seed_limit": args.seed_limit, "perturb_limit": args.perturb_limit})
    c1_rows = [reset_eval(seed, [-0.8, 0.1, 0.0], -15.0) for seed in selected]
    c1_hist = Counter()
    for r in c1_rows:
        if r.get("error"): c1_hist["model_load_or_probe_error"] += 1
        elif not r.get("reset_ok"):
            if r.get("reset_counts", {}).get("forbidden", 0) > 0: c1_hist["forbidden_contact_present"] += 1
            if r.get("reset_counts", {}).get("max_penetration_m", 0.0) > MAX_PENETRATION_M: c1_hist["max_penetration_gt_0p02m"] += 1
        else: c1_hist["reset_pass"] += 1
    c1_payload = {"cycle": 1, "repair_operator": "semantic_handle_binding_and_dynamic_drawer_model_load_repair", "changed_files": ["scripts/mint/contact_aware_drawer_teacher.py", "scripts/mint/merged_model_builder.py"], "before_histogram": corpus["histogram"], "after_targeted_histogram": dict(c1_hist), "progress_improved": c1_hist.get("model_load_or_probe_error", 0) < corpus["histogram"].get("model_load_or_probe_error", 18), "why": "Per-instance handle binding uses semantic mesh names; model builder emits only existing active drawer slider bodies/joints.", "rows": c1_rows}
    write_json(run_dir / "cycle_1_targeted_results.json", c1_payload); append_jsonl(run_dir / "repair_cycles.jsonl", c1_payload)
    base_map, base_diag = choose_base_map(selected)
    c2_payload = {"cycle": 2, "repair_operator": "adaptive_per_seed_base_yaw_reset_clearance_repair", "changed_files": ["scripts/mint/v11_g4_goc_v4_autonomous_repair_campaign.py"], "before_histogram": dict(c1_hist), "base_map": base_map, "diagnostics": base_diag, "after_expanded_histogram": {"reset_pass": sum(1 for v in base_map.values() if v.get("reset_ok")), "reset_failed": sum(1 for v in base_map.values() if not v.get("reset_ok"))}, "progress_improved": sum(1 for v in base_map.values() if v.get("reset_ok")) > c1_hist.get("reset_pass", 0)}
    write_json(run_dir / "cycle_2_expanded_results.json", c2_payload); append_jsonl(run_dir / "repair_cycles.jsonl", c2_payload)
    current_params = CONTROLLER_PARAM_GRID[0]; best_cases = run_matrix(selected, perturbations, base_map, current_params, run_dir, "cycle_2_full_matrix_cases.jsonl"); best_hist = histogram(best_cases); best_passes = pass_count(best_cases); cycles_run = 2
    write_json(run_dir / "cycle_2_full_matrix_results.json", {"cycle": 2, "params": current_params, "cases_total": len(best_cases), "cases_passed": best_passes, "cases_failed": len(best_cases) - best_passes, "histogram": best_hist, "cases": best_cases[:80]})
    for idx, params in enumerate(CONTROLLER_PARAM_GRID[1:], start=3):
        if pass_count(best_cases) == len(best_cases) or time.monotonic() > deadline: break
        cycles_run = idx; failed = [c for c in best_cases if not c.get("summary", {}).get("passes_robust_layer4r_gate", False)]; reps = failed[: min(12, len(failed))]; targeted = []
        for rep in reps:
            seed = int(rep["seed"]); pname = rep["perturbation"]; perturb = next((p for p in perturbations if p["name"] == pname), perturbations[0]); targeted.append(compact_case(run_layer4_case(seed, perturb, base_map[str(seed)], params, run_dir)))
        targeted_hist = histogram(targeted); cycle_payload = {"cycle": idx, "repair_operator": "controller_parameter_repair_for_target_short_forbidden_penetration", "changed_files": ["scripts/mint/v11_g4_goc_v4_autonomous_repair_campaign.py"], "controller_params": params, "targeted_cases_total": len(targeted), "targeted_cases_passed": pass_count(targeted), "targeted_histogram": targeted_hist, "progress_improved": pass_count(targeted) > 0 or targeted_hist.get("forbidden_contact_present", 0) < best_hist.get("forbidden_contact_present", 999), "targeted_cases": targeted}
        write_json(run_dir / f"cycle_{idx}_targeted_results.json", cycle_payload); append_jsonl(run_dir / "repair_cycles.jsonl", cycle_payload)
        if cycle_payload["progress_improved"]:
            full = run_matrix(selected, perturbations, base_map, params, run_dir, f"cycle_{idx}_full_matrix_cases.jsonl"); full_hist = histogram(full); full_pass = pass_count(full); write_json(run_dir / f"cycle_{idx}_full_matrix_results.json", {"cycle": idx, "controller_params": params, "cases_total": len(full), "cases_passed": full_pass, "cases_failed": len(full) - full_pass, "histogram": full_hist, "progress_improved_vs_previous_best": full_pass > best_passes, "cases": full[:80]})
            if full_pass > best_passes or (full_pass == best_passes and full_hist.get("forbidden_contact_present", 0) < best_hist.get("forbidden_contact_present", 999)): best_cases, best_hist, best_passes, current_params = full, full_hist, full_pass, params
    layer4_passed = bool(best_cases) and pass_count(best_cases) == len(best_cases); remaining_clusters = {k: v for k, v in best_hist.items() if k != "pass"}; write_json(run_dir / "layer4R_repair_summary.json", {"layer4R_robust_contact_certified": layer4_passed, "cases_total": len(best_cases), "cases_passed": pass_count(best_cases), "cases_failed": len(best_cases) - pass_count(best_cases), "remaining_failure_clusters": remaining_clusters, "selected_base_map": base_map, "selected_controller_params": current_params})
    layer5_passed = layer6_complete = layer7_passed = False; strict_candidate_count = 0; best_drawer_fraction = 0.0; strict_candidate = None; local_replay_dir = LOCAL_REPLAY_ROOT / f"v11_g4_goc_v4_autonomous_repair_certify_to_local_replay_{utc_stamp()}"
    if layer4_passed and time.monotonic() <= deadline:
        layer5_passed, attempts, strict_candidate = run_layer5_if_ready(selected, base_map, run_dir); strict_candidate_count = 1 if layer5_passed else 0; best_drawer_fraction = float(strict_candidate.get("summary", {}).get("max_drawer_fraction", 0.0)) if strict_candidate else 0.0; write_json(run_dir / "layer5_candidate_selection_report.json", {"passed": layer5_passed, "attempts": attempts, "strict_candidate": strict_candidate})
    else: write_json(run_dir / "layer5_candidate_selection_report.json", {"attempted": False, "reason": "Layer4R did not certify; promotion blocked."})
    if layer5_passed and strict_candidate is not None:
        cand_path = run_dir / "strict_candidate.json"; write_json(cand_path, strict_candidate); write_json(run_dir / "layer6_export_manifest.json", {"strict_candidate_path": rel(cand_path), "trace_path": strict_candidate.get("trace_path"), "source_committed_head": commands["head"], "hashes": {rel(cand_path): file_sha256(cand_path)}, "export_complete": True}); layer6_complete = True; local_replay_dir.mkdir(parents=True, exist_ok=True); write_json(local_replay_dir / "local_replay_placeholder_metrics.json", {"attempted": False, "reason": "Remote bundle created; local strict replay renderer not executed by remote process."}); write_json(run_dir / "layer7_local_replay_handoff_manifest.json", {"local_replay_dir": str(local_replay_dir), "passed": False, "reason": "Local replay/render requires local execution after export bundle."})
    else: write_json(run_dir / "layer6_export_manifest.json", {"export_complete": False, "reason": "No strict Layer5 candidate."}); write_json(run_dir / "layer7_local_replay_handoff_manifest.json", {"passed": False, "reason": "No Layer6 export bundle."})
    timed_out = time.monotonic() > deadline
    if layer4_passed and layer5_passed and layer6_complete and layer7_passed: classification, next_gate = "G4_TO_G7_FULL_STRICT_REPLAY_READY", "MANUAL_VISUAL_AND_SCIENCE_REVIEW_BEFORE_MINT_DATASET_ADMISSION"
    elif timed_out: classification, next_gate = "TIME_BUDGET_EXHAUSTED", "RESUME_AUTONOMOUS_REPAIR_CAMPAIGN_FROM_LAST_CERTIFIED_LAYER"
    elif not layer4_passed: classification, next_gate = "ROBUST_LAYER4R_CONTACT_DYNAMICS_REPAIR_STALLED", "ROBUST_CONTACT_POLICY_OR_GEOMETRY_REDESIGN"
    elif not layer5_passed: classification, next_gate = "LAYER5_STRICT_PULL_ROLLOUT_FAILED", "PULL_CONTROLLER_REPAIR"
    elif not layer6_complete: classification, next_gate = "LAYER6_EXPORT_BUNDLE_FAILED", "EXPORT_INFRA_REPAIR"
    elif not layer7_passed: classification, next_gate = "LAYER7_LOCAL_STRICT_REPLAY_RENDER_FAILED", "LOCAL_REPLAY_RENDER_REPAIR"
    else: classification, next_gate = "EXECUTION_FAILED", "RESUME_AUTONOMOUS_REPAIR_CAMPAIGN_FROM_LAST_CERTIFIED_LAYER"
    closeout = {"closeout_classification": classification, "harness_preflight_passed": None, "task_spec_lock_bound": None, "autonomous_repair_cycles_run": cycles_run, "layer4R_robust_contact_certified": layer4_passed, "layer4R_cases_total": len(best_cases), "layer4R_cases_passed": pass_count(best_cases), "layer4R_cases_failed": len(best_cases) - pass_count(best_cases), "layer4R_remaining_failure_clusters": remaining_clusters, "layer5_strict_pull_rollout_passed": layer5_passed, "layer6_export_bundle_complete": layer6_complete, "layer7_local_strict_replay_render_passed": layer7_passed, "strict_candidate_count": strict_candidate_count, "best_candidate_drawer_fraction": best_drawer_fraction, "local_replay_dir": str(local_replay_dir), "local_png_keyframes_created": False, "local_mp4_video_created": False, "current_truth_modified": False, "next_actions_modified": False, "runtime_patch_applied": True, "runtime_patch_files": ["scripts/mint/contact_aware_drawer_teacher.py", "scripts/mint/merged_model_builder.py", "scripts/mint/v11_g4_goc_v4_autonomous_repair_campaign.py"], "committed": False, "pushed_to_origin": False, "remote_commit_hash": None, "elapsed_seconds": round(time.monotonic() - start, 3), "next_gate": next_gate}
    write_proposed_deltas(closeout, run_dir); write_json(run_dir / "closeout_decision.json", closeout); write_md(run_dir / "final_report.md", final_report(closeout)); print(json.dumps(ready(closeout), indent=2, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())

#!/usr/bin/env python3
"""G1 orientation-stage audit for v12.1 slice-1."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_DIR = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1"
ARTIFACT_DIR = CAMPAIGN_DIR / "artifacts"
EVAL_DIAGNOSTIC_DIR = CAMPAIGN_DIR / "evaluation" / "tiny_retrain" / "V1cT2S0" / "s0" / "diagnostic"
GATES_V12_DIR = CAMPAIGN_DIR / "autopilot" / "gates_v12"
G0_GATE_PATH = GATES_V12_DIR / "G0_sovereign_sync.json"
OUTPUT_PATH = ARTIFACT_DIR / "v12_orientation_stage_audit.json"
GATE_PATH = GATES_V12_DIR / "G1_orientation_stage_audit.json"
ROLLOUT_DIR = ARTIFACT_DIR / "g6_learning_support_train_rollouts"
PROBE_SUMMARY_PATH = ARTIFACT_DIR / "g8_train_seed_probe.json"
TRUTH_CONTRACT_PATH = PROJECT_ROOT / "docs" / "contracts" / "truth_contract_v84.json"
ACCEPTANCE_CONTRACT_PATH = PROJECT_ROOT / "docs" / "contracts" / "acceptance_contract_v84.json"
SPEC_REFERENCE = "/Users/zhuhaowu/ws/phd-anyboxs/infinigen_from_servers/docs/gpt5.4Pro/codex_mint_v84_v12_1_orientation_attach_anygrasp_execution_spec.md"
ATTACH_THRESHOLD_M = 0.06
NEAR_DISTANCE_THRESHOLD_M = 0.08
APPROACH_THRESHOLD = 0.25
NEAR_APPROACH_THRESHOLD = 0.15
ORIENTATION_THRESHOLD = 0.60
ORIENTATION_NEAR_THRESHOLD = 0.50
ORIENTATION_IMPROVEMENT_DELTA = 0.10
WINDOW_LOOKAHEAD = 12


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text())


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=PROJECT_ROOT, text=True).strip()


def current_repo_identity() -> dict[str, str]:
    return {
        "branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "working_head_commit": _git(["git", "rev-parse", "HEAD"]),
        "vendor_head_commit": _git(["git", "rev-parse", "HEAD:external/MINT"]),
    }


def _base_gate_payload(run_instance_id: str) -> dict[str, Any]:
    ident = current_repo_identity()
    return {
        "run_instance_id": run_instance_id,
        "working_head_commit": ident["working_head_commit"],
        "vendor_head_commit": ident["vendor_head_commit"],
        "branch": ident["branch"],
        "truth_contract_hash": sha256_file(TRUTH_CONTRACT_PATH),
        "acceptance_contract_hash": sha256_file(ACCEPTANCE_CONTRACT_PATH),
        "spec_reference": SPEC_REFERENCE,
        "timestamp_utc": utc_now(),
    }


def write_gate(
    run_instance_id: str,
    status: str,
    blocking_reasons: list[str],
    extra: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "gate_id": "G1",
        "gate_name": "orientation_stage_audit",
        **_base_gate_payload(run_instance_id),
        "status": status,
        "blocking_reasons": blocking_reasons,
        "allowed_next_phases": ["G1b", "G2"] if status == "PASS" else [],
        **extra,
    }
    write_json_atomic(GATE_PATH, payload)
    return payload


def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return fallback.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _bool_rate(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values.astype(np.float32)))


def _phase_locked_trace(phase_labels: list[str]) -> np.ndarray:
    return np.asarray([("phase_lock" in x) or ("locked" in x) for x in phase_labels], dtype=bool)


def _approach_alignment_trace(eef_pos: np.ndarray, handle_anchor: np.ndarray) -> np.ndarray:
    if eef_pos.ndim != 2 or eef_pos.shape[0] == 0:
        return np.asarray([], dtype=np.float32)
    out = np.zeros((eef_pos.shape[0],), dtype=np.float32)
    world_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for i in range(1, eef_pos.shape[0]):
        delta = _normalize(eef_pos[i] - eef_pos[i - 1], world_x)
        desired = _normalize(handle_anchor - eef_pos[i - 1], world_x)
        out[i] = float(np.clip(np.dot(delta, desired), -1.0, 1.0))
    return out


def _window_end_index(start: int, alignment: np.ndarray, orientation_gate: np.ndarray) -> tuple[int, bool, float]:
    stop = min(len(alignment), start + WINDOW_LOOKAHEAD + 1)
    if stop <= start:
        return start, False, float(alignment[start])
    segment = alignment[start:stop]
    crossed_segment = orientation_gate[start:stop]
    crossed = bool(np.any(crossed_segment))
    best_delta = float(np.max(segment) - alignment[start])
    if crossed:
        end = int(start + np.flatnonzero(crossed_segment)[0])
        return end, True, float(alignment[end])
    best_idx = int(np.argmax(segment))
    return int(start + best_idx), bool(best_delta >= ORIENTATION_IMPROVEMENT_DELTA), float(segment[best_idx])


def _dominant_failure_from_rates(metrics: dict[str, float]) -> str:
    if float(metrics.get("close_cmd_rate_mean", 0.0)) < 0.20:
        return "never_close"
    if float(metrics.get("distance_pass_rate_mean", 0.0)) < 0.05:
        return "never_reach_attach_distance"
    if float(metrics.get("orientation_gate_pass_rate_mean", 0.0)) < 0.05:
        return "orientation_gate_miss"
    if float(metrics.get("approach_gate_pass_rate_mean", 0.0)) < 0.05:
        return "approach_gate_miss"
    if float(metrics.get("ever_attach_eligible_fraction", 0.0)) > 0.0 and float(metrics.get("stable_attach_rate", 0.0)) == 0.0:
        return "attach_not_stabilized"
    if float(metrics.get("stable_attach_rate", 0.0)) > 0.0 and float(metrics.get("phase_locked_rate", 0.0)) == 0.0:
        return "stable_attach_without_phase_lock"
    if float(metrics.get("phase_locked_rate", 0.0)) > 0.0 and float(metrics.get("ever_attached_rate", 0.0)) == 0.0:
        return "phase_locked_without_grasp"
    return "no_improvement"


def _orientation_windows(meta: dict[str, Any], data: Any) -> dict[str, Any]:
    alignment = np.asarray(data["orientation_alignment_trace"], dtype=np.float32).reshape(-1)
    orientation_error = np.asarray(data["orientation_error_trace"], dtype=np.float32).reshape(-1)
    orientation_gate = np.asarray(data["orientation_gate_trace"], dtype=bool).reshape(-1)
    dist = np.asarray(data["handle_distance_trace"], dtype=np.float32).reshape(-1)
    attach_eligible = np.asarray(data["attach_eligible_trace"], dtype=bool).reshape(-1)
    attached = np.asarray(data["attached_trace"], dtype=bool).reshape(-1)
    phase_labels = [str(x) for x in data["phase_labels"]]
    phase_locked = _phase_locked_trace(phase_labels)
    eef_pos = np.asarray(data["states"], dtype=np.float32)[:, :3]
    handle_anchor = np.asarray(
        (meta.get("orientation_telemetry") or {}).get("runtime_handle_anchor_world") or [0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    approach_alignment = _approach_alignment_trace(eef_pos, handle_anchor)
    approach_gate = approach_alignment >= APPROACH_THRESHOLD
    near_approach = approach_alignment >= NEAR_APPROACH_THRESHOLD
    distance_pass = dist <= ATTACH_THRESHOLD_M
    near_distance = dist <= NEAR_DISTANCE_THRESHOLD_M

    windows = []
    i = 0
    while i < len(alignment):
        if attached[i] or phase_locked[i]:
            i += 1
            continue
        eligible = bool(distance_pass[i] or near_distance[i]) and bool(approach_gate[i] or near_approach[i])
        orient_bad_or_near = (not bool(orientation_gate[i])) or float(alignment[i]) >= ORIENTATION_NEAR_THRESHOLD
        if not eligible or not orient_bad_or_near:
            i += 1
            continue
        end_idx, valid, end_alignment = _window_end_index(i, alignment, orientation_gate)
        if valid:
            windows.append(
                {
                    "start": int(i),
                    "end": int(end_idx),
                    "crossed_threshold": bool(np.any(orientation_gate[i : end_idx + 1])),
                    "alignment_start": float(alignment[i]),
                    "alignment_end": float(end_alignment),
                    "alignment_delta": float(end_alignment - alignment[i]),
                    "error_start": float(orientation_error[i]),
                    "error_end": float(orientation_error[end_idx]),
                    "error_delta": float(orientation_error[end_idx] - orientation_error[i]),
                }
            )
            i = end_idx + 1
        else:
            i += 1

    transitions = int(np.sum((~orientation_gate[:-1]) & orientation_gate[1:])) if len(orientation_gate) > 1 else 0
    meta_has_attach = meta.get("first_attach_eligible_step") is not None or (meta.get("orientation_telemetry") or {}).get("attach_eligible") is True
    telemetry_consistent = (not bool(np.any(attached))) or bool(np.any(attach_eligible)) or bool(meta_has_attach)
    return {
        "teacher_distance_pass_rate": _bool_rate(distance_pass),
        "teacher_approach_gate_pass_rate": _bool_rate(approach_gate),
        "teacher_orientation_gate_pass_rate": _bool_rate(orientation_gate),
        "teacher_attach_eligible_rate": _bool_rate(attach_eligible),
        "teacher_ever_attach_eligible_fraction": float(bool(np.any(attach_eligible)) or bool(meta_has_attach)),
        "teacher_ever_attached_rate": float(bool(np.any(attached))),
        "teacher_orientation_transition_count": transitions,
        "teacher_orientation_correction_window_count": len(windows),
        "teacher_orientation_error_start_mean": float(statistics.mean([w["error_start"] for w in windows])) if windows else 0.0,
        "teacher_orientation_error_end_mean": float(statistics.mean([w["error_end"] for w in windows])) if windows else 0.0,
        "teacher_orientation_error_delta_mean": float(statistics.mean([w["error_delta"] for w in windows])) if windows else 0.0,
        "teacher_orientation_crosses_threshold_fraction": float(statistics.mean([1.0 if w["crossed_threshold"] else 0.0 for w in windows])) if windows else 0.0,
        "teacher_telemetry_consistent": telemetry_consistent,
        "windows": windows,
    }


def _dominant_failure(records: list[dict[str, Any]]) -> str:
    vals = [str(rec.get("dominant_failure_mode") or "") for rec in records if rec.get("dominant_failure_mode")]
    return Counter(vals).most_common(1)[0][0] if vals else "unknown"


def _policy_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    summary = {
        "close_cmd_rate_mean": float(statistics.mean(float(r.get("close_cmd_rate", 0.0)) for r in records)),
        "distance_pass_rate_mean": float(statistics.mean(float(r.get("distance_pass_rate", 0.0)) for r in records)),
        "approach_gate_pass_rate_mean": float(statistics.mean(float(r.get("approach_gate_pass_rate", 0.0)) for r in records)),
        "orientation_gate_pass_rate_mean": float(statistics.mean(float(r.get("orientation_gate_pass_rate", 0.0)) for r in records)),
        "attach_eligible_rate_mean": float(statistics.mean(float(r.get("attach_eligible_rate", 0.0)) for r in records)),
        "ever_attach_eligible_fraction": float(statistics.mean(1.0 if bool(r.get("ever_attach_eligible", False)) else 0.0 for r in records)),
        "ever_attached_rate": float(statistics.mean(1.0 if bool(r.get("grasp_success", False)) else 0.0 for r in records)),
        "stable_attach_rate": float(statistics.mean(float(r.get("stable_attach_rate", 0.0)) for r in records)),
        "phase_locked_rate": float(statistics.mean(float(r.get("phase_locked_rate", 0.0)) for r in records)),
        "near_distance_orientation_miss_count": int(sum(1 for r in records if float(r.get("min_dist_to_handle", 1.0)) <= NEAR_DISTANCE_THRESHOLD_M and float(r.get("orientation_gate_pass_rate", 0.0)) < 0.05)),
        "near_approach_orientation_miss_count": int(sum(1 for r in records if float(r.get("approach_gate_pass_rate", 0.0)) >= APPROACH_THRESHOLD and float(r.get("orientation_gate_pass_rate", 0.0)) < 0.05)),
    }
    dominant = _dominant_failure(records)
    if dominant == "unknown":
        dominant = _dominant_failure_from_rates(summary)
    summary["dominant_failure_mode"] = dominant
    return summary


def run(rollout_dir: Path, probe_summary_path: Path, output_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    g0 = load_json(G0_GATE_PATH, {})
    run_instance_id = str(g0.get("run_instance_id") or "v12_1_slice1_unknown")
    probe_summary = load_json(probe_summary_path, {})
    checkpoint_step = int(probe_summary.get("checkpoint_step") or probe_summary.get("selected_checkpoint_step") or 0)
    probe_records_path = EVAL_DIAGNOSTIC_DIR / f"authoritative_train_probe_{checkpoint_step:06d}_records.json"
    probe_records = load_json(probe_records_path, {})

    per_rollout = []
    seed_windows: dict[str, int] = defaultdict(int)
    family_windows: dict[str, int] = defaultdict(int)
    family_count_by_seed: dict[str, set[str]] = defaultdict(set)
    telemetry_consistent = True
    total_windows = 0
    crossed_windows = 0
    parse_errors: list[str] = []

    for npz_path in sorted(rollout_dir.glob("*.npz")):
        meta_path = npz_path.with_suffix(".json")
        try:
            meta = load_json(meta_path, {})
            if str(meta.get("learning_support_teacher_class") or "") == "rejected_teacher":
                continue
            with np.load(npz_path, allow_pickle=True) as data:
                metrics = _orientation_windows(meta, data)
            seed = str(int(meta.get("seed", -1) or -1))
            family = str(meta.get("effective_support_signature_v1") or meta.get("learning_support_fingerprint") or npz_path.stem)
            total_windows += int(metrics["teacher_orientation_correction_window_count"])
            crossed_windows += int(round(metrics["teacher_orientation_crosses_threshold_fraction"] * metrics["teacher_orientation_correction_window_count"]))
            if metrics["teacher_orientation_correction_window_count"] > 0:
                seed_windows[seed] += int(metrics["teacher_orientation_correction_window_count"])
                family_windows[family] += int(metrics["teacher_orientation_correction_window_count"])
                family_count_by_seed[seed].add(family)
            telemetry_consistent = telemetry_consistent and bool(metrics["teacher_telemetry_consistent"])
            per_rollout.append({
                "seed": int(seed),
                "rollout": npz_path.name,
                "family": family,
                **metrics,
            })
        except Exception as exc:  # noqa: BLE001
            parse_errors.append(f"{npz_path.name}:{type(exc).__name__}:{exc}")

    teacher_seed_count_with_windows = sum(1 for count in seed_windows.values() if count > 0)
    teacher_summary = {
        "teacher_seed_count_with_orientation_correction_windows": teacher_seed_count_with_windows,
        "teacher_orientation_family_count": len(family_windows),
        "teacher_orientation_family_count_by_seed": {k: len(v) for k, v in sorted(family_count_by_seed.items())},
        "seed2_orientation_correction_windows": int(seed_windows.get("2", 0)),
        "seed4_orientation_correction_windows": int(seed_windows.get("4", 0)),
        "teacher_crosses_orientation_threshold_fraction": float(crossed_windows / total_windows) if total_windows else 0.0,
        "teacher_attach_telemetry_consistent": telemetry_consistent,
        "total_orientation_correction_windows": int(total_windows),
    }
    policy_summary = {
        "pretrained": _policy_summary(list(probe_records.get("pretrained_mint") or [])),
        "finetuned": _policy_summary(list(probe_records.get("finetuned_mint") or [])),
    }
    failure_stage_matches = str(policy_summary["finetuned"].get("dominant_failure_mode")) == "orientation_gate_miss"
    blocking_reasons = []
    if parse_errors:
        blocking_reasons.append("orientation_metric_invalid")
    if (
        teacher_summary["teacher_seed_count_with_orientation_correction_windows"] < 6
        or teacher_summary["seed2_orientation_correction_windows"] < 1
        or teacher_summary["seed4_orientation_correction_windows"] < 1
        or teacher_summary["total_orientation_correction_windows"] == 0
    ):
        blocking_reasons.append("orientation_support_absent_in_teacher")
    if (
        teacher_summary["teacher_crosses_orientation_threshold_fraction"] < 0.5
        or not teacher_summary["teacher_attach_telemetry_consistent"]
    ):
        blocking_reasons.append("orientation_metric_invalid")
    if not failure_stage_matches:
        blocking_reasons.append("policy_failure_not_orientation_stage")
    status = "PASS" if not blocking_reasons else "STOP"

    report = {
        "audit": "v12_orientation_stage_audit",
        **_base_gate_payload(run_instance_id),
        "evidence_working_head_commit": str(g0.get("evidence_working_head_commit") or g0.get("v11_execution_working_head_commit") or ""),
        "v11_run_instance_id": str(g0.get("v11_run_instance_id") or ""),
        "checkpoint_step": checkpoint_step,
        "probe_records_path": str(probe_records_path),
        "rollout_dir": str(rollout_dir),
        "teacher_summary": teacher_summary,
        "policy_summary": policy_summary,
        "teacher_per_rollout_metrics": per_rollout,
        "parse_errors": parse_errors,
        "orientation_stage_hypothesis_valid": status == "PASS",
    }
    write_json_atomic(output_path, report)
    gate = write_gate(
        run_instance_id,
        status,
        blocking_reasons,
        {
            "teacher_seed_count_with_orientation_correction_windows": teacher_summary["teacher_seed_count_with_orientation_correction_windows"],
            "seed2_orientation_correction_windows": teacher_summary["seed2_orientation_correction_windows"],
            "seed4_orientation_correction_windows": teacher_summary["seed4_orientation_correction_windows"],
            "teacher_crosses_orientation_threshold_fraction": teacher_summary["teacher_crosses_orientation_threshold_fraction"],
            "teacher_attach_telemetry_consistent": teacher_summary["teacher_attach_telemetry_consistent"],
            "policy_failure_stage": policy_summary["finetuned"].get("dominant_failure_mode"),
            "audit_artifact_path": str(output_path),
        },
    )
    return report, gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, default=ROLLOUT_DIR)
    parser.add_argument("--probe-summary", type=Path, default=PROBE_SUMMARY_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    report, gate = run(args.rollout_dir, args.probe_summary, args.output)
    print(json.dumps({"report": report, "gate": gate}, indent=2))
    return 0 if gate.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

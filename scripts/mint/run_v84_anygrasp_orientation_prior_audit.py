#!/usr/bin/env python3
"""G1b AnyGrasp orientation-prior sidecar audit for v12.1 slice-1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from anygrasp_helper import build_anygrasp, stage_detection_assets
from drawer_robot_env_mujoco import DrawerEnvContractConfig, DrawerRobotEnvMuJoCo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_DIR = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1"
ARTIFACT_DIR = CAMPAIGN_DIR / "artifacts"
GATES_V12_DIR = CAMPAIGN_DIR / "autopilot" / "gates_v12"
G0_GATE_PATH = GATES_V12_DIR / "G0_sovereign_sync.json"
G1_AUDIT_PATH = ARTIFACT_DIR / "v12_orientation_stage_audit.json"
OUTPUT_PATH = ARTIFACT_DIR / "v12_anygrasp_orientation_prior_audit.json"
GATE_PATH = GATES_V12_DIR / "G1b_anygrasp_orientation_prior_audit.json"
TRUTH_CONTRACT_PATH = PROJECT_ROOT / "docs" / "contracts" / "truth_contract_v84.json"
ACCEPTANCE_CONTRACT_PATH = PROJECT_ROOT / "docs" / "contracts" / "acceptance_contract_v84.json"
SPEC_REFERENCE = "/Users/zhuhaowu/ws/phd-anyboxs/infinigen_from_servers/docs/gpt5.4Pro/codex_mint_v84_v12_1_orientation_attach_anygrasp_execution_spec.md"
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
MAX_CANDIDATES = 20
POINTS_PER_SEED = 1024
HANDLE_SUPPORT_THRESHOLD_M = 0.05
EEF_REACHABILITY_THRESHOLD_M = 0.20
PULL_AXIS_MIN_ALIGNMENT = 0.60
WIDTH_MAX_M = 0.10
DEPTH_MAX_M = 0.08
ORIENTATION_GAIN_MIN = 0.05
ANGLE_DIVERSITY_MIN_DEG = 10.0
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float32)
WORLD_X = np.array([1.0, 0.0, 0.0], dtype=np.float32)


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


def write_gate(run_instance_id: str, status: str, blocking_reasons: list[str], extra: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "gate_id": "G1b",
        "gate_name": "anygrasp_orientation_prior_audit",
        **_base_gate_payload(run_instance_id),
        "status": status,
        "blocking_reasons": blocking_reasons,
        "allowed_next_phases": ["G2"],
        **extra,
    }
    write_json_atomic(GATE_PATH, payload)
    return payload


def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return fallback.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _canonical_teacher_rotation(motion_axis: np.ndarray) -> np.ndarray:
    x_axis = _normalize(motion_axis.astype(np.float32), WORLD_X)
    up_hint = WORLD_UP.copy()
    if abs(float(np.dot(x_axis, up_hint))) > 0.95:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    z_axis = up_hint - float(np.dot(up_hint, x_axis)) * x_axis
    z_axis = _normalize(z_axis, WORLD_UP)
    y_axis = _normalize(np.cross(z_axis, x_axis), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    z_axis = _normalize(np.cross(x_axis, y_axis), WORLD_UP)
    return np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)


def _rotation_angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    delta = rot_a.T @ rot_b
    trace = float(np.trace(delta))
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _candidate_family_signature(translation: np.ndarray, handle_center: np.ndarray, angle_deg: float, pull_alignment: float, width: float) -> str:
    rel_bucket = tuple(int(np.floor(x / 0.01)) for x in (translation - handle_center))
    return json.dumps(
        {
            "rel_bucket": rel_bucket,
            "angle_bucket_deg": int(np.floor(angle_deg / 10.0) * 10),
            "pull_align_bucket": int(np.floor(pull_alignment / 0.05) * 5),
            "width_bucket_mm": int(np.floor(width / 0.005) * 5_000),
        },
        sort_keys=True,
    )


def _run_candidates(detector, points: np.ndarray, colors: np.ndarray, lims: np.ndarray) -> list[dict[str, Any]]:
    gg, _cloud = detector.get_grasp(
        points.astype(np.float32),
        colors.astype(np.float32),
        lims=lims.astype(np.float32).tolist(),
        apply_object_mask=True,
        dense_grasp=False,
        collision_detection=True,
    )
    if len(gg) == 0:
        return []
    gg = gg.nms().sort_by_score()
    out = []
    for idx in range(min(len(gg), MAX_CANDIDATES)):
        grasp = gg[idx]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = np.asarray(grasp.rotation_matrix, dtype=np.float32)
        pose[:3, 3] = np.asarray(grasp.translation, dtype=np.float32)
        out.append(
            {
                "score": float(grasp.score),
                "width": float(getattr(grasp, "width", 0.0)),
                "depth": float(getattr(grasp, "depth", 0.0)),
                "translation": pose[:3, 3].tolist(),
                "rotation_matrix": pose[:3, :3].tolist(),
                "pose": pose.tolist(),
            }
        )
    return out


def run(output_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    g0 = load_json(G0_GATE_PATH, {})
    g1 = load_json(G1_AUDIT_PATH, {})
    run_instance_id = str(g0.get("run_instance_id") or "v12_1_slice1_unknown")
    report: dict[str, Any] = {
        **_base_gate_payload(run_instance_id),
        "audit": "v12_anygrasp_orientation_prior_audit",
        "teacher_orientation_family_count": int((g1.get("teacher_summary") or {}).get("teacher_orientation_family_count") or 0),
        "seed_records": [],
    }

    stage_report = stage_detection_assets()
    report["stage_report"] = stage_report
    if not stage_report.get("passed"):
        report.update(
            {
                "anygrasp_available": False,
                "adjudication": "anygrasp_sidecar_unavailable",
            }
        )
        write_json_atomic(output_path, report)
        gate = write_gate(
            run_instance_id,
            "STOP",
            ["anygrasp_sidecar_unavailable"],
            {"audit_artifact_path": str(output_path), "adjudication": report["adjudication"]},
        )
        return report, gate

    try:
        detector = build_anygrasp()
    except Exception as exc:  # noqa: BLE001
        report.update(
            {
                "anygrasp_available": False,
                "adjudication": "anygrasp_sidecar_unavailable",
                "build_error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json_atomic(output_path, report)
        gate = write_gate(
            run_instance_id,
            "STOP",
            ["anygrasp_sidecar_unavailable"],
            {"audit_artifact_path": str(output_path), "adjudication": report["adjudication"]},
        )
        return report, gate

    rejection_counts: Counter[str] = Counter()
    raw_total = 0
    filtered_total = 0
    filtered_families: set[str] = set()
    filtered_seed_count = 0
    hard_seed_with_candidates = False
    angle_differences: list[float] = []
    gains: list[float] = []
    pull_alignments: list[float] = []
    plausibilities: list[float] = []
    diverse_improving = False

    for seed in SEEDS:
        contract = DrawerEnvContractConfig(
            interaction_mode="orientation_sensitive_v3_task_identity_locked",
            state_mode="m0_proxy",
        )
        env = DrawerRobotEnvMuJoCo(seed=seed, image_size=256, max_steps=4, contract=contract)
        try:
            env.reset()
            payload = env.anygrasp_payload(num_points=POINTS_PER_SEED)
            raw_candidates = _run_candidates(detector, payload["pc"], payload["colors"], payload["limits"])
            raw_total += len(raw_candidates)
            handle_center = np.asarray(payload["handle_center_world"], dtype=np.float32)
            motion_axis = _normalize(np.asarray(payload["drawer_motion_axis"], dtype=np.float32), WORLD_X)
            teacher_rot = _canonical_teacher_rotation(motion_axis)
            current_pull_axis = _normalize(R.from_quat(np.asarray(env.eef_quat, dtype=np.float32)).apply(np.array([1.0, 0.0, 0.0], dtype=np.float32)).astype(np.float32), WORLD_X)
            current_pull_alignment = float(np.clip(np.dot(current_pull_axis, motion_axis), -1.0, 1.0))
            eef_pos = np.asarray(payload["eef_pos"], dtype=np.float32)
            seed_filtered = []
            for cand in raw_candidates:
                translation = np.asarray(cand["translation"], dtype=np.float32)
                rot = np.asarray(cand["rotation_matrix"], dtype=np.float32)
                pull_axis = _normalize(rot[:, 0], WORLD_X)
                pull_alignment = float(np.clip(np.dot(pull_axis, motion_axis), -1.0, 1.0))
                orientation_gain = float(pull_alignment - current_pull_alignment)
                angle_deg = _rotation_angle_deg(teacher_rot, rot)
                plausibility_checks = {
                    "handle_local_support": float(np.linalg.norm(translation - handle_center)) <= HANDLE_SUPPORT_THRESHOLD_M,
                    "pull_axis_compatibility": pull_alignment >= PULL_AXIS_MIN_ALIGNMENT,
                    "eef_reachability_proxy": float(np.linalg.norm(translation - eef_pos)) <= EEF_REACHABILITY_THRESHOLD_M,
                    "collision_width_plausibility": 0.0 < float(cand["width"]) <= WIDTH_MAX_M and 0.0 < float(cand["depth"]) <= DEPTH_MAX_M,
                    "orientation_gate_plausibility": pull_alignment >= ORIENTATION_THRESHOLD and orientation_gain >= ORIENTATION_GAIN_MIN,
                }
                plausibility = float(np.mean(np.asarray(list(plausibility_checks.values()), dtype=np.float32)))
                if not all(plausibility_checks.values()):
                    for key, ok in plausibility_checks.items():
                        if not ok:
                            rejection_counts[key] += 1
                    plausibilities.append(plausibility)
                    continue
                family = _candidate_family_signature(translation, handle_center, angle_deg, pull_alignment, float(cand["width"]))
                candidate_record = {
                    **cand,
                    "pull_axis_alignment": pull_alignment,
                    "orientation_alignment_gain": orientation_gain,
                    "teacher_angle_deg": angle_deg,
                    "family": family,
                    "attach_gate_plausibility": plausibility,
                }
                if angle_deg > ANGLE_DIVERSITY_MIN_DEG and orientation_gain > ORIENTATION_GAIN_MIN:
                    diverse_improving = True
                seed_filtered.append(candidate_record)
                filtered_families.add(family)
                gains.append(orientation_gain)
                angle_differences.append(angle_deg)
                pull_alignments.append(pull_alignment)
                plausibilities.append(plausibility)
            filtered_total += len(seed_filtered)
            if seed_filtered:
                filtered_seed_count += 1
                if seed in {2, 4}:
                    hard_seed_with_candidates = True
            report["seed_records"].append(
                {
                    "seed": seed,
                    "current_pull_alignment": current_pull_alignment,
                    "raw_candidate_count": len(raw_candidates),
                    "task_filtered_candidate_count": len(seed_filtered),
                    "task_filtered_family_count": len({x["family"] for x in seed_filtered}),
                    "filtered_candidates": seed_filtered,
                }
            )
        finally:
            env.close()

    topk = sorted(gains, reverse=True)[: min(3, len(gains))]
    if not filtered_total:
        adjudication = "anygrasp_no_task_filtered_candidates"
    elif filtered_seed_count >= 2 and hard_seed_with_candidates and diverse_improving:
        adjudication = "anygrasp_orientation_prior_promising"
    else:
        adjudication = "anygrasp_orientation_prior_redundant"

    report.update(
        {
            "anygrasp_available": True,
            "anygrasp_raw_candidate_count": raw_total,
            "anygrasp_task_filtered_candidate_count": filtered_total,
            "anygrasp_task_filtered_family_count": len(filtered_families),
            "teacher_vs_anygrasp_min_angle_deg": float(min(angle_differences)) if angle_differences else None,
            "teacher_vs_anygrasp_median_angle_deg": float(np.median(np.asarray(angle_differences, dtype=np.float32))) if angle_differences else None,
            "anygrasp_orientation_alignment_gain_best": float(max(gains)) if gains else 0.0,
            "anygrasp_orientation_alignment_gain_mean_topk": float(np.mean(np.asarray(topk, dtype=np.float32))) if topk else 0.0,
            "anygrasp_pull_axis_alignment_best": float(max(pull_alignments)) if pull_alignments else 0.0,
            "anygrasp_attach_gate_plausibility_best": float(max(plausibilities)) if plausibilities else 0.0,
            "anygrasp_candidate_rejection_reasons": dict(sorted(rejection_counts.items())),
            "filtered_seed_count": filtered_seed_count,
            "hard_seed_with_candidates": hard_seed_with_candidates,
            "diverse_improving_candidate_present": diverse_improving,
            "adjudication": adjudication,
        }
    )
    write_json_atomic(output_path, report)
    gate = write_gate(
        run_instance_id,
        "PASS",
        [],
        {
            "audit_artifact_path": str(output_path),
            "adjudication": adjudication,
            "filtered_seed_count": filtered_seed_count,
            "hard_seed_with_candidates": hard_seed_with_candidates,
            "task_filtered_family_count": len(filtered_families),
        },
    )
    return report, gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    report, gate = run(args.output)
    print(json.dumps({"report": report, "gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

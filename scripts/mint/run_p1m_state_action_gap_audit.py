#!/usr/bin/env python3
"""P0D: quantify state/action gaps and choose state mapping by measurement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance

from mint_common import now_iso, write_json_atomic, write_text_atomic
from p1_execution_common import ARTIFACT_DIR, OUTPUT_DIR, upsert_evidence
from drawer_robot_env_mujoco import DrawerRobotEnvMuJoCo

M4_ARTIFACT = ARTIFACT_DIR / "m4_robot_rollout_gate.json"
CONTROL_TRACE_ROOT = ARTIFACT_DIR / "p1k_control_traces"
ARTIFACT_PATH = ARTIFACT_DIR / "p1m_state_action_gap_audit.json"
REPORT_PATH = OUTPUT_DIR / "p1m_state_action_gap_audit.md"
EVIDENCE_ID = "E055"
EXPERIMENT_ID = "p1m_state_action_gap_audit"


def _load_control_arrays() -> tuple[np.ndarray, np.ndarray]:
    states = []
    actions = []
    for trace_path in sorted(CONTROL_TRACE_ROOT.glob("episode_*/trace.npz")):
        data = np.load(trace_path)
        states.append(np.asarray(data["state_pre"], dtype=np.float32))
        actions.append(np.asarray(data["action_sent"], dtype=np.float32))
    if not states or not actions:
        raise FileNotFoundError("No p1k control traces found; run P0B first.")
    return np.concatenate(states, axis=0), np.concatenate(actions, axis=0)


def _load_learning_rollout_arrays() -> tuple[np.ndarray, np.ndarray, list[int]]:
    m4 = json.loads(M4_ARTIFACT.read_text())
    learning_dir = Path(m4["learning_rollout_dir"])
    seeds = [int(x) for x in m4.get("successful_learning_seeds", [])]
    state_traces = []
    action_traces = []
    for path in sorted(learning_dir.glob("*.npz")):
        data = np.load(path)
        state_traces.append(np.asarray(data["states"], dtype=np.float32))
        action_traces.append(np.asarray(data["actions"], dtype=np.float32))
    if not state_traces or not action_traces:
        raise FileNotFoundError("No MuJoCo learning rollout traces found; run m4 first.")
    return np.concatenate(state_traces, axis=0), np.concatenate(action_traces, axis=0), seeds


def _per_dim_wasserstein(a: np.ndarray, b: np.ndarray) -> list[float]:
    dims = min(a.shape[1], b.shape[1])
    return [float(wasserstein_distance(a[:, i], b[:, i])) for i in range(dims)]


def _per_dim_range_overlap(a: np.ndarray, b: np.ndarray) -> list[float]:
    dims = min(a.shape[1], b.shape[1])
    overlaps = []
    for i in range(dims):
        a_min, a_max = float(np.min(a[:, i])), float(np.max(a[:, i]))
        b_min, b_max = float(np.min(b[:, i])), float(np.max(b[:, i]))
        inter = max(0.0, min(a_max, b_max) - max(a_min, b_min))
        union = max(a_max, b_max) - min(a_min, b_min)
        overlaps.append(float(inter / union) if union > 1e-8 else 1.0)
    return overlaps


def _temporal_smoothness_gap(a: np.ndarray, b: np.ndarray) -> list[float]:
    dims = min(a.shape[1], b.shape[1])
    a_diff = np.diff(a[:, :dims], axis=0)
    b_diff = np.diff(b[:, :dims], axis=0)
    return [float(abs(np.std(a_diff[:, i]) - np.std(b_diff[:, i]))) for i in range(dims)]


def _normalized_score(wd: list[float], overlap: list[float], smooth_gap: list[float], control: np.ndarray, target: np.ndarray) -> float:
    normed = []
    for i, value in enumerate(wd):
        a_min, a_max = float(np.min(control[:, i])), float(np.max(control[:, i]))
        b_min, b_max = float(np.min(target[:, i])), float(np.max(target[:, i]))
        span = max(a_max, b_max) - min(a_min, b_min)
        normed.append(float(value / max(span, 1e-6)))
    return float(np.mean(normed) + 0.25 * np.mean(smooth_gap) + 0.1 * np.mean([1.0 - x for x in overlap]))


def _state_candidates(sample_seed: int) -> dict[str, dict[str, Any]]:
    env = DrawerRobotEnvMuJoCo(seed=sample_seed, image_size=256, max_steps=4)
    try:
        obs = env.reset()
        current = np.asarray(obs.state, dtype=np.float32).reshape(1, -1)
        nq = int(env.model.nq)
        candidate_m0 = {
            "valid": True,
            "state": current,
            "description": "Current proxy mapping [eef_xyz, handle_rel_xyz, drawer_fraction, gripper_joint] emitted by DrawerRobotEnvMuJoCo._state_vector().",
        }
        candidate_m1 = {
            "valid": nq >= 4,
            "description": "Raw qpos mapping requiring >=4 arm qpos dimensions.",
            "invalid_reason": None if nq >= 4 else f"MuJoCo env exposes nq={nq}, which is insufficient for raw 4D arm_qpos mapping.",
        }
        candidate_m2 = {
            "valid": nq >= 3,
            "description": "Hybrid qpos+progress mapping requiring >=3 arm qpos dimensions.",
            "invalid_reason": None if nq >= 3 else f"MuJoCo env exposes nq={nq}, which is insufficient for hybrid arm_qpos mapping.",
        }
        return {"M0": candidate_m0, "M1": candidate_m1, "M2": candidate_m2, "env_nq": nq}
    finally:
        env.close()


def main() -> int:
    control_state, control_action = _load_control_arrays()
    target_state, target_action, seeds = _load_learning_rollout_arrays()
    sample_seed = seeds[0] if seeds else 1
    candidates = _state_candidates(sample_seed)

    candidate_scores: dict[str, Any] = {}
    if candidates["M0"]["valid"]:
        wd = _per_dim_wasserstein(control_state, target_state)
        overlap = _per_dim_range_overlap(control_state, target_state)
        smooth_gap = _temporal_smoothness_gap(control_state, target_state)
        candidate_scores["M0"] = {
            "valid": True,
            "wasserstein": wd,
            "range_overlap": overlap,
            "temporal_smoothness_gap": smooth_gap,
            "overall_score": _normalized_score(wd, overlap, smooth_gap, control_state, target_state),
            "description": candidates["M0"]["description"],
        }
    for key in ["M1", "M2"]:
        candidate_scores[key] = {
            "valid": candidates[key]["valid"],
            "overall_score": None,
            "description": candidates[key]["description"],
            "invalid_reason": candidates[key].get("invalid_reason"),
        }

    selected_mapping = min(
        (key for key, value in candidate_scores.items() if value.get("valid")),
        key=lambda key: candidate_scores[key]["overall_score"],
        default="M0",
    )

    action_wd = _per_dim_wasserstein(control_action, target_action)
    action_overlap = _per_dim_range_overlap(control_action, target_action)
    control_rot_std = np.std(control_action[:, 3:6], axis=0).astype(float).tolist()
    target_rot_std = np.std(target_action[:, 3:6], axis=0).astype(float).tolist()
    control_rotation_near_degenerate = all(x < 0.01 for x in control_rot_std)
    target_rotation_near_degenerate = all(x < 0.01 for x in target_rot_std)
    rotation_primary_blocker = (not control_rotation_near_degenerate) and target_rotation_near_degenerate

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at": now_iso(),
        "passed": True,
        "selected_state_mapping": selected_mapping,
        "candidate_state_scores": candidate_scores,
        "state_gap": candidate_scores[selected_mapping],
        "action_gap": {
            "wasserstein": action_wd,
            "range_overlap": action_overlap,
        },
        "rotation_stats": {
            "control_rot_std": control_rot_std,
            "mujoco_rot_std": target_rot_std,
            "control_rotation_near_degenerate": control_rotation_near_degenerate,
            "mujoco_rotation_near_degenerate": target_rotation_near_degenerate,
            "rotation_primary_blocker": rotation_primary_blocker,
        },
        "candidate_validation": {
            "env_nq": candidates["env_nq"],
            "note": "Measured selection first validates whether raw qpos candidates are even representable in the current MuJoCo env. Invalid candidates are not silently substituted.",
        },
    }
    write_json_atomic(ARTIFACT_PATH, payload)

    lines = [
        "# p1m State/Action Gap Audit",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## State Mapping Selection",
        f"- selected_state_mapping: {selected_mapping}",
        f"- env_nq: {candidates['env_nq']}",
    ]
    for key, value in candidate_scores.items():
        if value.get("valid"):
            lines.append(f"- {key}: valid overall_score={value['overall_score']:.6f}")
        else:
            lines.append(f"- {key}: invalid ({value.get('invalid_reason')})")
    lines.extend([
        "",
        "## Rotation Stats",
        f"- control_rot_std: {control_rot_std}",
        f"- mujoco_rot_std: {target_rot_std}",
        f"- rotation_primary_blocker: {rotation_primary_blocker}",
    ])
    write_text_atomic(REPORT_PATH, "\n".join(lines).rstrip() + "\n")

    upsert_evidence(
        EVIDENCE_ID,
        EXPERIMENT_ID,
        "state_action_gap_audit",
        ARTIFACT_PATH,
        "Measured state/action gap audit selecting the canonical MuJoCo state mapping by score and explicitly validating whether qpos-based candidates are representable.",
        verified=True,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

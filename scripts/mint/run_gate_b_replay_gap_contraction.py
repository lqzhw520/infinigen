#!/usr/bin/env python3
"""
Stage A.5 - Gate B Replay-Gap Contraction
=========================================

Proposal-only follow-up after corrected Gate B Stage A.

This script explains the remaining replay-gap between:
- env-native oracle reachability
- teacher open-loop execution

It does not update canonical truth.
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN_ROOT = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1"
ARTIFACT_ROOT = CAMPAIGN_ROOT / "artifacts"
SOVEREIGN = CAMPAIGN_ROOT / "sovereign"
PROPOSALS_DIR = SOVEREIGN / "proposals"
SCRIPTS_MINT = PROJECT_ROOT / "scripts" / "mint"

sys.path.insert(0, str(SCRIPTS_MINT))
sys.path.insert(0, str(PROJECT_ROOT))

from drawer_robot_env import DrawerRobotEnv  # noqa: E402
from run_gate_b_teacher_replayability import (  # noqa: E402
    HANDLE_ATTACH_THRESHOLD,
    ORACLE_DEFINITION,
    _first_step,
    current_env_action_contract,
    current_env_handle_local,
    git_head,
    now_ts,
    resolve_npz,
    run_oracle_reachability,
    run_replay_mode,
    selected_episodes,
    source_meta,
    summarize_run_from_traces,
    to_builtin,
)


IMAGE_SIZE = 16
RENDER_OBSERVATIONS = False
MAX_STEPS = 200
WORKERS = max(1, min(4, (os.cpu_count() or 1)))
TOP10_EPISODES = [157, 205, 25, 180, 3, 181, 158, 182, 138, 209]
ATTACH_THRESHOLD = 0.05

R1_DIR = ARTIFACT_ROOT / "gate_b4_teacher_oracle_gap"
R2_DIR = ARTIFACT_ROOT / "gate_b5_initial_state_seed"
R3_DIR = ARTIFACT_ROOT / "gate_b6_handle_semantics"
E030_YAML = PROPOSALS_DIR / "E030_gate_b_replay_gap_contraction_draft.yaml"
E030_MD = PROPOSALS_DIR / "E030_gate_b_replay_gap_contraction_draft.md"

SAMPLE_SCOPE = {
    "sample": "Gate A action-quality top-10",
    "mapping_target": "success-selected legacy rollout subset",
    "population_claim": "No population-wide claim over full 240 episodes",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_builtin(payload), indent=2, ensure_ascii=False))


def save_yaml(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(to_builtin(payload), allow_unicode=True, sort_keys=False))


def save_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_builtin(value) for key, value in row.items()})


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def _gap_category(metrics: dict[str, Any]) -> str:
    if bool(metrics.get("success")):
        return "teacher_success"
    failure_stage = str(metrics.get("failure_stage", "invalid_run"))
    if failure_stage == "never_near_handle":
        return "pre_handle_replay_gap"
    if failure_stage == "near_handle_no_attach":
        return "handle_semantics_or_grasp_gap"
    if failure_stage in {"attached_no_pull", "pull_no_open"}:
        return "post_attach_execution_gap"
    return "invalid_run"


def _seed_mode_definition(seed_mode: str) -> dict[str, Any]:
    payloads = {
        "open_loop_seeded_initial_physical_pose": {
            "seed_mode": seed_mode,
            "description": "Reset env, seed robot physical pose/gripper from source step-0 position only, keep commanded pose at reset home.",
            "uses_source_joint_state": False,
            "uses_source_drawer_fraction": False,
            "uses_source_attachment_state": False,
            "uses_source_runtime_history": False,
            "uses_source_eef_position": True,
            "uses_source_eef_quaternion": False,
            "aligns_command_reference": False,
        },
        "open_loop_seeded_initial_physical_and_command": {
            "seed_mode": seed_mode,
            "description": "Reset env, seed robot physical pose/gripper from source step-0 position only, then align commanded pose to the seeded current-env pose.",
            "uses_source_joint_state": False,
            "uses_source_drawer_fraction": False,
            "uses_source_attachment_state": False,
            "uses_source_runtime_history": False,
            "uses_source_eef_position": True,
            "uses_source_eef_quaternion": False,
            "aligns_command_reference": True,
        },
    }
    return payloads[seed_mode]


def _seed_initial_pose(env: DrawerRobotEnv, source_npz: dict[str, Any], *, align_command_reference: bool) -> tuple[np.ndarray, np.ndarray]:
    source_eef_positions = np.asarray(source_npz["eef_positions"], dtype=np.float32)
    source_gripper_values = np.asarray(source_npz["gripper_values"], dtype=np.float32)
    source_pos0 = source_eef_positions[0].astype(np.float32) if len(source_eef_positions) else env.eef_pose()[0]
    current_pos, current_quat = env.eef_pose()
    target_quat = np.asarray(current_quat, dtype=np.float32)
    target_joints = env._ik(source_pos0, target_quat)
    env._set_arm_joints(target_joints)
    if len(source_gripper_values):
        env._set_gripper_open(float(source_gripper_values[0]))
    env._step_world(2)
    seeded_pos, seeded_quat = env.eef_pose()
    env.drawer_trace_history = [float(env.drawer_fraction())]
    env.attached_trace_history = [False]
    env.attached = False
    env.ever_attached = False
    env.step_count = 0
    if align_command_reference:
        env.commanded_eef_pos = seeded_pos.copy()
        env.commanded_eef_quat = seeded_quat.copy()
    return seeded_pos.copy(), seeded_quat.copy()


def run_seeded_open_loop(
    *,
    episode_index: int,
    npz_name: str,
    seed_mode: str,
    attach_threshold: float,
    max_steps: int,
    image_size: int,
    render_observations: bool,
) -> dict[str, Any]:
    npz_path = resolve_npz(npz_name)
    source_npz = dict(np.load(npz_path, allow_pickle=True))
    meta = source_meta(npz_path)
    action_contract = current_env_action_contract(
        meta,
        attach_threshold_override=attach_threshold,
    )
    env = DrawerRobotEnv(
        seed=int(meta["seed"]),
        image_size=image_size,
        max_steps=max_steps,
        action_contract=action_contract,
        render_observations=render_observations,
    )
    actions = np.asarray(source_npz["actions"], dtype=np.float32)
    eef_positions: list[np.ndarray] = []
    drawer_trace: list[float] = []
    attached_trace: list[bool] = []
    handle_distance_trace: list[float] = []
    success = False
    attach_step = None
    seed_definition = _seed_mode_definition(seed_mode)
    try:
        env.reset()
        local_handle, handle_metadata = current_env_handle_local(env)
        env.attachment_local = local_handle
        seeded_pos, seeded_quat = _seed_initial_pose(
            env,
            source_npz,
            align_command_reference=bool(seed_definition["aligns_command_reference"]),
        )
        for step_index in range(max_steps):
            action = actions[step_index] if step_index < len(actions) else np.zeros((7,), dtype=np.float32)
            obs, _reward, done, info = env.step(action)
            handle_pos, _ = env.local_to_world_pose(
                local_handle["pos"],
                local_handle["quat"],
                fraction=obs.drawer_fraction,
            )
            eef_positions.append(obs.eef_pos.copy())
            drawer_trace.append(float(obs.drawer_fraction))
            attached_trace.append(bool(info.get("attached")))
            handle_distance_trace.append(float(np.linalg.norm(obs.eef_pos - handle_pos)))
            if info.get("attached") and attach_step is None:
                attach_step = step_index
            success = success or bool(info.get("is_success", False))
            if done:
                break
    finally:
        env.close()

    summary = summarize_run_from_traces(
        episode_index=episode_index,
        npz_path=npz_path,
        source_npz=source_npz,
        source_meta_payload=meta,
        replay_mode=seed_mode,
        eef_positions=np.asarray(eef_positions, dtype=np.float32),
        drawer_trace=np.asarray(drawer_trace, dtype=np.float32),
        attached_trace=np.asarray(attached_trace, dtype=np.bool_),
        handle_distance_trace=np.asarray(handle_distance_trace, dtype=np.float32),
        success=success,
        attach_step=attach_step,
        action_contract=action_contract,
        frames=[],
    )
    summary["seed_definition"] = seed_definition
    summary["measurement_image_size_px"] = int(image_size)
    summary["render_observations"] = bool(render_observations)
    summary["initial_seeded_eef_pos"] = seeded_pos
    summary["initial_seeded_eef_quat"] = seeded_quat
    return summary


def _run_episode_bundle(
    episode_index: int,
    npz_name: str,
    attach_threshold: float,
    max_steps: int,
    image_size: int,
    render_observations: bool,
) -> dict[str, Any]:
    npz_path = resolve_npz(npz_name)
    source_npz = dict(np.load(npz_path, allow_pickle=True))
    meta = source_meta(npz_path)
    open_loop = run_replay_mode(
        episode_index=episode_index,
        npz_path=npz_path,
        source_npz=source_npz,
        source_meta_payload=meta,
        replay_mode="open_loop",
        max_steps=max_steps,
        attach_threshold_override=attach_threshold,
        image_size=image_size,
        render_observations=render_observations,
        capture_frames=False,
        capture_trace=False,
    )
    oracle = run_oracle_reachability(
        episode_index=episode_index,
        npz_path=npz_path,
        source_npz=source_npz,
        source_meta_payload=meta,
        max_steps=max_steps,
        attach_threshold_override=attach_threshold,
        image_size=image_size,
        render_observations=render_observations,
        capture_frames=False,
    )
    seeded_pose = run_seeded_open_loop(
        episode_index=episode_index,
        npz_name=npz_name,
        seed_mode="open_loop_seeded_initial_physical_pose",
        attach_threshold=attach_threshold,
        max_steps=max_steps,
        image_size=image_size,
        render_observations=render_observations,
    )
    seeded_pose_cmd = run_seeded_open_loop(
        episode_index=episode_index,
        npz_name=npz_name,
        seed_mode="open_loop_seeded_initial_physical_and_command",
        attach_threshold=attach_threshold,
        max_steps=max_steps,
        image_size=image_size,
        render_observations=render_observations,
    )

    source_handle_distance = np.asarray(source_npz["handle_distance_trace"], dtype=np.float32)
    source_attached_trace = np.asarray(source_npz["attached_trace"], dtype=bool)
    source_actions = np.asarray(source_npz["actions"], dtype=np.float32)
    teacher_close_step = _first_step(source_actions[:, 6] < 0) if len(source_actions) else None
    source_first_attach_step = _first_step(source_attached_trace)
    source_min_handle_distance = float(np.min(source_handle_distance)) if len(source_handle_distance) else float("nan")
    close_before_near_current = bool(
        teacher_close_step is not None
        and (
            open_loop.get("first_near_handle_step") is None
            or int(teacher_close_step) < int(open_loop["first_near_handle_step"])
        )
    )
    return {
        "episode_index": int(episode_index),
        "npz_name": npz_name,
        "npz_path": str(npz_path),
        "source_meta": {
            "success": bool(meta.get("success", False)),
            "ever_attached": bool(meta.get("ever_attached", False)),
            "final_drawer_fraction": float(meta.get("final_drawer_fraction", 0.0)),
            "max_drawer_fraction": float(meta.get("max_drawer_fraction", 0.0)),
        },
        "source_trace": {
            "min_handle_distance_m": source_min_handle_distance,
            "first_attach_step": source_first_attach_step,
            "ever_attached": bool(np.any(source_attached_trace)),
            "teacher_close_step": teacher_close_step,
        },
        "open_loop": open_loop,
        "oracle_reachability": oracle,
        "seeded_initial_physical_pose": seeded_pose,
        "seeded_initial_physical_and_command": seeded_pose_cmd,
        "derived": {
            "gap_category": _gap_category(open_loop),
            "close_before_near_current": close_before_near_current,
            "source_attached_but_current_never_near": bool(
                np.any(source_attached_trace) and open_loop.get("first_near_handle_step") is None
            ),
            "source_current_min_handle_distance_delta_m": float(
                open_loop.get("min_handle_distance_m", float("nan")) - source_min_handle_distance
            ),
        },
    }


def _episode_set() -> list[tuple[int, str]]:
    selected = selected_episodes(",".join(str(v) for v in TOP10_EPISODES))
    return [(int(ep), str(name)) for ep, name in selected]


def _mode_rates(bundles: list[dict[str, Any]], key: str) -> tuple[float, float]:
    n = max(len(bundles), 1)
    attach = sum(1 for row in bundles if row[key]["ever_attached"]) / n
    success = sum(1 for row in bundles if row[key]["success"]) / n
    return float(attach), float(success)


def build_r1(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    gap_counts = Counter()
    for row in bundles:
        gap_category = row["derived"]["gap_category"]
        gap_counts[gap_category] += 1
        rows.append(
            {
                "episode_index": row["episode_index"],
                "npz_name": row["npz_name"],
                "gap_category": gap_category,
                "teacher_close_step": row["source_trace"]["teacher_close_step"],
                "teacher_close_before_near_current": row["derived"]["close_before_near_current"],
                "open_loop_success": row["open_loop"]["success"],
                "open_loop_ever_attached": row["open_loop"]["ever_attached"],
                "open_loop_failure_stage": row["open_loop"]["failure_stage"],
                "open_loop_min_handle_distance_m": row["open_loop"]["min_handle_distance_m"],
                "open_loop_first_near_handle_step": row["open_loop"]["first_near_handle_step"],
                "open_loop_attach_step": row["open_loop"]["attach_step"],
                "open_loop_max_drawer_fraction": row["open_loop"]["max_drawer_fraction"],
                "oracle_success": row["oracle_reachability"]["success"],
                "oracle_ever_attached": row["oracle_reachability"]["ever_attached"],
                "oracle_failure_stage": row["oracle_reachability"]["failure_stage"],
                "oracle_min_handle_distance_m": row["oracle_reachability"]["min_handle_distance_m"],
                "oracle_first_near_handle_step": row["oracle_reachability"]["first_near_handle_step"],
                "oracle_attach_step": row["oracle_reachability"]["attach_step"],
                "oracle_max_drawer_fraction": row["oracle_reachability"]["max_drawer_fraction"],
            }
        )

    oracle_attach_rate, oracle_success_rate = _mode_rates(bundles, "oracle_reachability")
    open_attach_rate, open_success_rate = _mode_rates(bundles, "open_loop")
    artifact = {
        "experiment": "gate_b4_teacher_oracle_gap",
        "timestamp": now_ts(),
        "sample_scope": dict(SAMPLE_SCOPE),
        "attach_threshold_m": ATTACH_THRESHOLD,
        "oracle_definition": dict(ORACLE_DEFINITION),
        "measurement_image_size_px": IMAGE_SIZE,
        "render_observations": RENDER_OBSERVATIONS,
        "episodes": rows,
        "aggregate": {
            "oracle_attach_rate": oracle_attach_rate,
            "oracle_success_rate": oracle_success_rate,
            "open_loop_attach_rate": open_attach_rate,
            "open_loop_success_rate": open_success_rate,
            "oracle_vs_open_loop_attach_gap": float(oracle_attach_rate - open_attach_rate),
            "oracle_vs_open_loop_success_gap": float(oracle_success_rate - open_success_rate),
            "gap_category_counts": dict(gap_counts),
            "dominant_gap_category": gap_counts.most_common(1)[0][0] if gap_counts else "invalid_run",
            "close_before_near_count": int(sum(1 for row in rows if row["teacher_close_before_near_current"])),
            "mean_teacher_min_handle_distance_m": float(np.mean([row["open_loop_min_handle_distance_m"] for row in rows])),
            "mean_oracle_min_handle_distance_m": float(np.mean([row["oracle_min_handle_distance_m"] for row in rows])),
        },
    }
    save_json(R1_DIR / "gate_b4_summary.json", artifact)
    save_csv(R1_DIR / "gate_b4_episodes.csv", rows)
    save_text(R1_DIR / "gate_b4_summary.md", render_r1_markdown(artifact))
    return artifact


def build_r2(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    modes = [
        ("open_loop", "open_loop"),
        ("seeded_initial_physical_pose", "seeded_initial_physical_pose"),
        ("seeded_initial_physical_and_command", "seeded_initial_physical_and_command"),
    ]
    for row in bundles:
        for label, key in modes:
            metrics = row[key]
            rows.append(
                {
                    "episode_index": row["episode_index"],
                    "npz_name": row["npz_name"],
                    "mode": label,
                    "success": metrics["success"],
                    "ever_attached": metrics["ever_attached"],
                    "failure_stage": metrics["failure_stage"],
                    "min_handle_distance_m": metrics["min_handle_distance_m"],
                    "first_near_handle_step": metrics["first_near_handle_step"],
                    "attach_step": metrics["attach_step"],
                    "max_drawer_fraction": metrics["max_drawer_fraction"],
                }
            )
    aggregate_modes = {}
    for label, key in modes:
        attach_rate, success_rate = _mode_rates(bundles, key)
        stage_counts = Counter(str(row[key]["failure_stage"]) for row in bundles)
        aggregate_modes[label] = {
            "attach_rate": attach_rate,
            "success_rate": success_rate,
            "dominant_failure_stage": stage_counts.most_common(1)[0][0] if stage_counts else "invalid_run",
        }
    baseline = aggregate_modes["open_loop"]
    improvements = {}
    best_mode = "open_loop"
    best_success = baseline["success_rate"]
    for label in ("seeded_initial_physical_pose", "seeded_initial_physical_and_command"):
        payload = aggregate_modes[label]
        improvements[label] = {
            "attach_rate_delta_vs_open_loop": float(payload["attach_rate"] - baseline["attach_rate"]),
            "success_rate_delta_vs_open_loop": float(payload["success_rate"] - baseline["success_rate"]),
        }
        if payload["success_rate"] > best_success:
            best_success = payload["success_rate"]
            best_mode = label
    artifact = {
        "experiment": "gate_b5_initial_state_seed",
        "timestamp": now_ts(),
        "sample_scope": dict(SAMPLE_SCOPE),
        "attach_threshold_m": ATTACH_THRESHOLD,
        "measurement_image_size_px": IMAGE_SIZE,
        "render_observations": RENDER_OBSERVATIONS,
        "episodes": rows,
        "aggregate": {
            "modes": aggregate_modes,
            "improvements_vs_open_loop": improvements,
            "best_seed_mode": best_mode,
            "best_seed_success_rate": float(best_success),
            "best_seed_success_delta_vs_open_loop": float(best_success - baseline["success_rate"]),
            "best_seed_attach_delta_vs_open_loop": float(
                aggregate_modes[best_mode]["attach_rate"] - baseline["attach_rate"]
            ),
        },
    }
    save_json(R2_DIR / "gate_b5_summary.json", artifact)
    save_csv(R2_DIR / "gate_b5_episodes.csv", rows)
    save_text(R2_DIR / "gate_b5_summary.md", render_r2_markdown(artifact))
    return artifact


def build_r3(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for row in bundles:
        rows.append(
            {
                "episode_index": row["episode_index"],
                "npz_name": row["npz_name"],
                "source_min_handle_distance_m": row["source_trace"]["min_handle_distance_m"],
                "current_min_handle_distance_m": row["open_loop"]["min_handle_distance_m"],
                "source_current_min_handle_distance_delta_m": row["derived"]["source_current_min_handle_distance_delta_m"],
                "source_first_attach_step": row["source_trace"]["first_attach_step"],
                "teacher_close_step": row["source_trace"]["teacher_close_step"],
                "current_first_near_handle_step": row["open_loop"]["first_near_handle_step"],
                "current_attach_step": row["open_loop"]["attach_step"],
                "current_failure_stage": row["open_loop"]["failure_stage"],
                "close_before_near_current": row["derived"]["close_before_near_current"],
                "source_attached_but_current_never_near": row["derived"]["source_attached_but_current_never_near"],
                "handle_distance_vs_source_error": row["open_loop"]["handle_distance_vs_source_error"],
            }
        )
    artifact = {
        "experiment": "gate_b6_handle_semantics",
        "timestamp": now_ts(),
        "sample_scope": dict(SAMPLE_SCOPE),
        "attach_threshold_m": ATTACH_THRESHOLD,
        "measurement_image_size_px": IMAGE_SIZE,
        "render_observations": RENDER_OBSERVATIONS,
        "episodes": rows,
        "aggregate": {
            "mean_source_min_handle_distance_m": float(np.mean([row["source_min_handle_distance_m"] for row in rows])),
            "mean_current_min_handle_distance_m": float(np.mean([row["current_min_handle_distance_m"] for row in rows])),
            "mean_source_current_min_handle_distance_delta_m": float(
                np.mean([row["source_current_min_handle_distance_delta_m"] for row in rows])
            ),
            "mean_handle_distance_vs_source_error": float(np.mean([row["handle_distance_vs_source_error"] for row in rows])),
            "close_before_near_count": int(sum(1 for row in rows if row["close_before_near_current"])),
            "close_before_near_rate": float(sum(1 for row in rows if row["close_before_near_current"]) / max(len(rows), 1)),
            "source_attached_but_current_never_near_count": int(
                sum(1 for row in rows if row["source_attached_but_current_never_near"])
            ),
            "source_attached_but_current_never_near_rate": float(
                sum(1 for row in rows if row["source_attached_but_current_never_near"]) / max(len(rows), 1)
            ),
            "dominant_current_failure_stage": Counter(row["current_failure_stage"] for row in rows).most_common(1)[0][0] if rows else "invalid_run",
        },
    }
    save_json(R3_DIR / "gate_b6_summary.json", artifact)
    save_csv(R3_DIR / "gate_b6_episodes.csv", rows)
    save_text(R3_DIR / "gate_b6_summary.md", render_r3_markdown(artifact))
    return artifact


def build_e030(r1: dict[str, Any], r2: dict[str, Any], r3: dict[str, Any]) -> tuple[dict[str, Any], str]:
    oracle_success = float(r1["aggregate"]["oracle_success_rate"])
    open_success = float(r1["aggregate"]["open_loop_success_rate"])
    open_attach = float(r1["aggregate"]["open_loop_attach_rate"])
    best_seed_success = float(r2["aggregate"]["best_seed_success_rate"])
    best_seed_success_delta = float(r2["aggregate"]["best_seed_success_delta_vs_open_loop"])
    best_seed_attach_delta = float(r2["aggregate"]["best_seed_attach_delta_vs_open_loop"])
    best_seed_vs_oracle_gap = float(oracle_success - best_seed_success)
    close_before_near_rate = float(r3["aggregate"]["close_before_near_rate"])
    never_near_rate = float(r3["aggregate"]["source_attached_but_current_never_near_rate"])
    dominant_gap_category = str(r1["aggregate"]["dominant_gap_category"])

    if best_seed_success_delta > 0.20 or best_seed_attach_delta > 0.20:
        final_branch = "initial-state-or-command seeding materially contracts the replay-gap"
    elif never_near_rate >= 0.50 or close_before_near_rate >= 0.50 or dominant_gap_category == "pre_handle_replay_gap":
        final_branch = "pre-handle approach and handle-semantics mismatch dominate the replay-gap"
    elif dominant_gap_category == "post_attach_execution_gap":
        final_branch = "post-attach pull/open robustness dominates the replay-gap"
    else:
        final_branch = "mixed replay-gap remains after contraction audits"

    draft = {
        "draft_id": "E030_draft",
        "draft_type": "proposal_only_followup",
        "target_evidence_id": "E030",
        "experiment_id": "gate_b_replay_gap_contraction",
        "timestamp": now_ts(),
        "scope": "gate_b_top10_replay_gap_contraction",
        "verified": False,
        "sample_scope": dict(SAMPLE_SCOPE),
        "summary": (
            "Stage A.5 contracts the replay-gap by comparing env-native oracle reachability to "
            "teacher open-loop execution, testing step-0 initial-state seeding, and auditing "
            "teacher actions under current-env handle semantics."
        ),
        "source_artifacts": {
            "e029": "sovereign/proposals/E029_gate_b_corrected_deconfounded_draft.yaml",
            "r1": "artifacts/gate_b4_teacher_oracle_gap/gate_b4_summary.json",
            "r2": "artifacts/gate_b5_initial_state_seed/gate_b5_summary.json",
            "r3": "artifacts/gate_b6_handle_semantics/gate_b6_summary.json",
        },
        "measurement_runtime_note": (
            f"Replay-gap contraction audits used image_size={IMAGE_SIZE} and render_observations={RENDER_OBSERVATIONS} "
            "for state-only measurement throughput; physics, task semantics, and action contracts were unchanged."
        ),
        "key_findings": {
            "oracle_success_rate": oracle_success,
            "open_loop_success_rate": open_success,
            "open_loop_attach_rate": open_attach,
            "best_seed_success_rate": best_seed_success,
            "best_seed_success_delta_vs_open_loop": best_seed_success_delta,
            "best_seed_attach_delta_vs_open_loop": best_seed_attach_delta,
            "best_seed_vs_oracle_success_gap": best_seed_vs_oracle_gap,
            "close_before_near_rate": close_before_near_rate,
            "source_attached_but_current_never_near_rate": never_near_rate,
            "dominant_gap_category": dominant_gap_category,
        },
        "final_branch": final_branch,
        "stage_c_reconsider_allowed": bool(best_seed_vs_oracle_gap <= 0.30),
        "allowed_wording": [
            "Replay-gap contraction refines the gap between env-native reachability and teacher-action execution under canonical current-env semantics.",
            "Initial-state seeding and current-env handle semantics can be discussed only as replay-gap contraction factors, not as solved root cause.",
        ],
        "disallowed_wording": [
            "policy-only gap proven",
            "source-grasp fidelity proven",
            "root cause solved",
            "visual bottleneck proven from Stage A.5",
        ],
        "do_not": [
            "Do not treat env-native oracle reachability as teacher-trajectory fidelity.",
            "Do not reopen policy-gap Stage C solely because one seeded mode improves.",
            "Do not generalize beyond the Gate A top-10 sample.",
        ],
    }
    save_yaml(E030_YAML, draft)
    save_text(E030_MD, render_e030_markdown(draft))
    return draft, final_branch


def render_r1_markdown(artifact: dict[str, Any]) -> str:
    agg = artifact["aggregate"]
    return "\n".join(
        [
            "# Gate B4 Teacher-vs-Oracle Gap",
            "",
            "- Proposal-only replay-gap contraction audit.",
            f"- Attach threshold: `{artifact['attach_threshold_m']:.2f}`",
            f"- Oracle success rate: `{agg['oracle_success_rate']:.3f}`",
            f"- Open-loop success rate: `{agg['open_loop_success_rate']:.3f}`",
            f"- Oracle-open success gap: `{agg['oracle_vs_open_loop_success_gap']:.3f}`",
            f"- Dominant gap category: `{agg['dominant_gap_category']}`",
            f"- Close-before-near count: `{agg['close_before_near_count']}`",
            "",
        ]
    ) + "\n"


def render_r2_markdown(artifact: dict[str, Any]) -> str:
    agg = artifact["aggregate"]
    lines = [
        "# Gate B5 Initial-State Seeding",
        "",
        "- Proposal-only replay-gap contraction audit.",
        f"- Attach threshold: `{artifact['attach_threshold_m']:.2f}`",
        f"- Best seed mode: `{agg['best_seed_mode']}`",
        f"- Best seed success rate: `{agg['best_seed_success_rate']:.3f}`",
        f"- Best seed success delta vs open-loop: `{agg['best_seed_success_delta_vs_open_loop']:.3f}`",
        f"- Best seed attach delta vs open-loop: `{agg['best_seed_attach_delta_vs_open_loop']:.3f}`",
        "",
    ]
    for mode, payload in agg["modes"].items():
        lines.append(f"- {mode}: attach `{payload['attach_rate']:.3f}`, success `{payload['success_rate']:.3f}`, dominant failure `{payload['dominant_failure_stage']}`")
    return "\n".join(lines) + "\n"


def render_r3_markdown(artifact: dict[str, Any]) -> str:
    agg = artifact["aggregate"]
    return "\n".join(
        [
            "# Gate B6 Handle Semantics",
            "",
            "- Proposal-only replay-gap contraction audit.",
            f"- Mean source min-handle distance: `{agg['mean_source_min_handle_distance_m']:.4f}`",
            f"- Mean current min-handle distance: `{agg['mean_current_min_handle_distance_m']:.4f}`",
            f"- Mean distance delta: `{agg['mean_source_current_min_handle_distance_delta_m']:.4f}`",
            f"- Close-before-near rate: `{agg['close_before_near_rate']:.3f}`",
            f"- Source-attached but current-never-near rate: `{agg['source_attached_but_current_never_near_rate']:.3f}`",
            f"- Dominant current failure stage: `{agg['dominant_current_failure_stage']}`",
            "",
        ]
    ) + "\n"


def render_e030_markdown(draft: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# E030 Draft",
            "",
            "Non-canonical replay-gap contraction follow-up.",
            "",
            f"- Final branch: `{draft['final_branch']}`",
            f"- Stage C reconsider allowed: `{draft['stage_c_reconsider_allowed']}`",
            f"- Oracle success rate: `{draft['key_findings']['oracle_success_rate']:.3f}`",
            f"- Open-loop success rate: `{draft['key_findings']['open_loop_success_rate']:.3f}`",
            f"- Best seed success rate: `{draft['key_findings']['best_seed_success_rate']:.3f}`",
            f"- Close-before-near rate: `{draft['key_findings']['close_before_near_rate']:.3f}`",
            "",
            "## Guardrails",
            "",
            *[f"- {item}" for item in draft["do_not"]],
            "",
        ]
    ) + "\n"


def main() -> None:
    sentinel_path = ARTIFACT_ROOT / "gate_b1_threshold_sentinel" / "gate_b1_sentinel_summary.json"
    e029_path = PROPOSALS_DIR / "E029_gate_b_corrected_deconfounded_draft.yaml"
    if not sentinel_path.exists() or not e029_path.exists():
        raise SystemExit("Stage A outputs missing. Run corrected Stage A before Stage A.5.")

    episode_set = _episode_set()
    ensure_dir(R1_DIR)
    ensure_dir(R2_DIR)
    ensure_dir(R3_DIR)

    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = [
            executor.submit(
                _run_episode_bundle,
                episode_index,
                npz_name,
                ATTACH_THRESHOLD,
                MAX_STEPS,
                IMAGE_SIZE,
                RENDER_OBSERVATIONS,
            )
            for episode_index, npz_name in episode_set
        ]
        bundles = [future.result() for future in concurrent.futures.as_completed(futures)]
    bundles.sort(key=lambda row: int(row["episode_index"]))

    r1 = build_r1(bundles)
    r2 = build_r2(bundles)
    r3 = build_r3(bundles)
    draft, final_branch = build_e030(r1, r2, r3)

    print(
        json.dumps(
            {
                "r1": str((R1_DIR / "gate_b4_summary.json").relative_to(CAMPAIGN_ROOT)),
                "r2": str((R2_DIR / "gate_b5_summary.json").relative_to(CAMPAIGN_ROOT)),
                "r3": str((R3_DIR / "gate_b6_summary.json").relative_to(CAMPAIGN_ROOT)),
                "e030": str(E030_YAML.relative_to(CAMPAIGN_ROOT)),
                "final_branch": final_branch,
                "stage_c_reconsider_allowed": bool(draft["stage_c_reconsider_allowed"]),
                "git_head": git_head(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Gate B Teacher Executability Audit
==================================

Discriminative executability audit for source-successful teacher actions.

This collector is intentionally proposal-only:
  - it does not write canonical verdicts
  - it does not update claims / next_actions / state
  - it only emits raw artifacts and evidence-ready summaries

Scientific question:
  Can source-successful teacher actions produce attach/open in the current
  DrawerRobotEnv contract, and if not, at which execution layer do they fail?
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN_ROOT = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1"
ARTIFACT_ROOT = CAMPAIGN_ROOT / "artifacts"
SCRIPTS_MINT = PROJECT_ROOT / "scripts" / "mint"
ACTIVE_ACTION_CONTRACT_PATH = ARTIFACT_ROOT / "active_action_contract.json"

sys.path.insert(0, str(SCRIPTS_MINT))
sys.path.insert(0, str(PROJECT_ROOT))

from drawer_robot_env import (  # noqa: E402
    DrawerRobotEnv,
    HANDLE_ATTACH_THRESHOLD,
    HOME_JOINTS,
    ORACLE_PULL_QUAT_XYZW,
    build_native_teacher_rollout,
)

V58_ROLLOUTS = ARTIFACT_ROOT / "v58_physics_legal_rollouts"
P4_ROLLOUTS = ARTIFACT_ROOT / "p4_physics_legal_rollouts"
DEFAULT_OUTPUT_DIR = ARTIFACT_ROOT / "gate_b_teacher_replayability"

DEFAULT_TOP10 = [
    (157, "seed_013_branch2_phase_scaled_ep0.npz"),
    (205, "seed_015_branch2_phase_scaled_ep0.npz"),
    (25, "seed_008_branch2_phase_scaled_ep0.npz"),
    (180, "seed_014_branch2_phase_scaled_ep1.npz"),
    (3, "seed_007_branch2_phase_scaled_ep3.npz"),
    (181, "seed_014_branch2_phase_scaled_ep3.npz"),
    (158, "seed_013_branch2_phase_scaled_ep2.npz"),
    (182, "seed_014_branch2_phase_scaled_ep4.npz"),
    (138, "seed_012_branch2_phase_scaled_ep4.npz"),
    (209, "seed_015_branch2_phase_scaled_ep4.npz"),
]
TOP10_BY_EPISODE = {episode_index: npz_name for episode_index, npz_name in DEFAULT_TOP10}
DEFAULT_MODES = ("open_loop", "state_anchored", "oracle_reachability")
SUPPORTED_MODES = (
    "open_loop",
    "state_anchored",
    "state_anchored_legacy",
    "state_anchored_full_state_no_attach",
    "state_anchored_robot_only_no_attach",
    "oracle_reachability",
)
FAILURE_STAGES = (
    "never_near_handle",
    "near_handle_no_attach",
    "attached_no_pull",
    "pull_no_open",
    "success",
    "invalid_run",
)

SUCCESS_OPEN_THRESHOLD = 0.90
MAJOR_OPEN_THRESHOLD = 0.25

ORACLE_DEFINITION = {
    "oracle_type": "env_native_reachability_oracle",
    "oracle_uses_env_handle_pose": True,
    "oracle_uses_teacher_actions": False,
    "oracle_proves": "task reachability under current-env handle semantics",
    "oracle_does_not_prove": "teacher trajectory fidelity",
}

MODE_DEFINITIONS = {
    "open_loop": {
        "mode_family": "teacher_replay",
        "restore_robot_state": False,
        "restore_drawer_fraction": False,
        "restore_source_attachment_state": False,
        "restore_source_runtime_histories": False,
        "pre_step_force_detached": False,
        "scientific_role": "baseline_teacher_replay",
    },
    "state_anchored": {
        "mode_family": "teacher_replay",
        "restore_robot_state": True,
        "restore_drawer_fraction": True,
        "restore_source_attachment_state": True,
        "restore_source_runtime_histories": True,
        "pre_step_force_detached": False,
        "scientific_role": "legacy_state_anchored",
    },
    "state_anchored_legacy": {
        "mode_family": "teacher_replay",
        "restore_robot_state": True,
        "restore_drawer_fraction": True,
        "restore_source_attachment_state": True,
        "restore_source_runtime_histories": True,
        "pre_step_force_detached": False,
        "scientific_role": "legacy_state_anchored",
    },
    "state_anchored_full_state_no_attach": {
        "mode_family": "teacher_replay",
        "restore_robot_state": True,
        "restore_drawer_fraction": True,
        "restore_source_attachment_state": False,
        "restore_source_runtime_histories": False,
        "pre_step_force_detached": True,
        "scientific_role": "clean_full_state_no_attach",
    },
    "state_anchored_robot_only_no_attach": {
        "mode_family": "teacher_replay",
        "restore_robot_state": True,
        "restore_drawer_fraction": False,
        "restore_source_attachment_state": False,
        "restore_source_runtime_histories": False,
        "pre_step_force_detached": True,
        "scientific_role": "clean_robot_only_no_attach",
    },
    "oracle_reachability": {
        "mode_family": "env_native_oracle",
        "restore_robot_state": False,
        "restore_drawer_fraction": False,
        "restore_source_attachment_state": False,
        "restore_source_runtime_histories": False,
        "pre_step_force_detached": False,
        "scientific_role": "env_native_reachability_oracle",
    },
}


def now_ts() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_builtin(payload), f, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_builtin(v) for k, v in row.items()})


def get_imageio():
    import imageio.v2 as imageio

    return imageio


def write_video(frames: list[np.ndarray], video_path: Path, fps: int = 8) -> bool:
    if not frames:
        return False
    video_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio = get_imageio()
        writer = imageio.get_writer(
            video_path,
            fps=fps,
            codec="libx264",
            quality=7,
            pixelformat="yuv420p",
        )
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
        writer.close()
        return True
    except Exception:
        return False


def selected_episodes(raw: str | None) -> list[tuple[int, str]]:
    if not raw:
        return list(DEFAULT_TOP10)
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        episode_index = int(chunk)
        if episode_index not in TOP10_BY_EPISODE:
            raise ValueError(
                f"Episode {episode_index} is not in the canonical Gate B top-10 set"
            )
        values.append((episode_index, TOP10_BY_EPISODE[episode_index]))
    deduped = []
    seen = set()
    for item in values:
        if item[0] not in seen:
            deduped.append(item)
            seen.add(item[0])
    return deduped


def selected_modes(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_MODES)
    values = []
    for chunk in raw.split(","):
        mode = chunk.strip()
        if not mode:
            continue
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported Gate B mode: {mode}")
        values.append(mode)
    if not values:
        raise ValueError("At least one Gate B mode is required")
    return list(dict.fromkeys(values))


def mode_definition(mode: str) -> dict[str, Any]:
    if mode not in MODE_DEFINITIONS:
        raise ValueError(f"Unsupported Gate B mode: {mode}")
    return dict(MODE_DEFINITIONS[mode])


def resolve_npz(npz_name: str) -> Path:
    candidate = V58_ROLLOUTS / npz_name
    if candidate.exists():
        return candidate
    candidate = P4_ROLLOUTS / npz_name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"NPZ not found: {npz_name}")


def source_meta(npz_path: Path) -> dict[str, Any]:
    meta_path = npz_path.with_suffix(".json")
    meta = load_json(meta_path, default={}) or {}
    if meta:
        return meta
    source = np.load(npz_path, allow_pickle=True)
    payload: dict[str, Any] = {}
    for key in (
        "seed",
        "episode_index",
        "success",
        "ever_attached",
        "final_drawer_fraction",
        "max_drawer_fraction",
        "branch_id",
    ):
        if key in source.files:
            value = source[key]
            payload[key] = value.item() if getattr(value, "shape", None) == () else to_builtin(value)
    return payload


def current_env_action_contract(
    source_meta_payload: dict[str, Any],
    *,
    attach_threshold_override: float | None = None,
) -> dict[str, Any]:
    contract = dict(source_meta_payload.get("action_contract") or {})
    current = load_json(ACTIVE_ACTION_CONTRACT_PATH, default={}) or {}
    if current:
        contract.update(current)
    if attach_threshold_override is not None:
        contract["attach_threshold_m"] = float(attach_threshold_override)
    if "attach_threshold_m" not in contract:
        contract["attach_threshold_m"] = HANDLE_ATTACH_THRESHOLD
    return contract


def _restore_state(
    env: DrawerRobotEnv,
    source: dict[str, Any],
    step_index: int,
    *,
    restore_drawer_fraction: bool,
    restore_source_attachment_state: bool,
    restore_source_runtime_histories: bool,
    pre_step_force_detached: bool,
) -> None:
    joint_positions = source["joint_positions"].astype(np.float32)
    current_drawer = source["absolute_drawer_fraction"].astype(np.float32)
    current_gripper = source["gripper_values"].astype(np.float32)
    attached_trace = source["attached_trace"].astype(np.bool_)
    current_eef_pos = source["eef_positions"].astype(np.float32)
    current_eef_quat = (
        source["eef_quaternions"].astype(np.float32)
        if "eef_quaternions" in source
        else None
    )
    if "next_drawer_fraction" in source:
        next_drawer = source["next_drawer_fraction"].astype(np.float32)
    else:
        next_drawer = np.concatenate(
            [current_drawer[1:], current_drawer[-1:]], axis=0
        ) if len(current_drawer) else np.zeros((0,), dtype=np.float32)

    if step_index == 0:
        joint_state = np.asarray(HOME_JOINTS, dtype=np.float32)
    else:
        joint_state = joint_positions[min(step_index - 1, len(joint_positions) - 1)]
    for idx, val in zip(env.arm_joint_indices, joint_state.tolist()):
        env.p.resetJointState(
            env.robot_id,
            idx,
            float(val),
            targetVelocity=0.0,
            physicsClientId=env.client,
        )

    if restore_drawer_fraction and step_index < len(current_drawer):
        drawer_fraction = float(current_drawer[step_index])
        drawer_q = env.drawer_low + float(np.clip(drawer_fraction, 0.0, 1.0)) * (
            env.drawer_high - env.drawer_low
        )
        env.p.resetJointState(
            env.drawer_id,
            env.drawer_joint,
            drawer_q,
            targetVelocity=0.0,
            physicsClientId=env.client,
        )

    if step_index < len(current_gripper):
        env._set_gripper_open(float(current_gripper[step_index]))

    env._step_world(2)

    prev_attached = (
        bool(attached_trace[step_index - 1])
        if step_index > 0 and step_index - 1 < len(attached_trace)
        else False
    )
    if restore_source_attachment_state:
        env.attached = prev_attached
        env.ever_attached = (
            bool(np.any(attached_trace[:step_index])) if step_index > 0 else False
        )
    elif pre_step_force_detached:
        env.attached = False
        env.ever_attached = False
    env.step_count = int(step_index)
    if step_index < len(current_eef_pos):
        env.commanded_eef_pos = current_eef_pos[step_index].copy()
    if current_eef_quat is not None and step_index < len(current_eef_quat):
        env.commanded_eef_quat = current_eef_quat[step_index].copy()
    else:
        _, eef_quat = env.eef_pose()
        env.commanded_eef_quat = eef_quat.copy()
    if restore_source_runtime_histories:
        if len(current_drawer):
            env.drawer_trace_history = [float(current_drawer[0])] + [
                float(x) for x in next_drawer[:step_index]
            ]
        else:
            env.drawer_trace_history = [0.0]
        env.attached_trace_history = [False] + [
            bool(x) for x in attached_trace[:step_index]
        ]
    else:
        current_fraction = (
            float(current_drawer[step_index])
            if restore_drawer_fraction and step_index < len(current_drawer)
            else float(env.drawer_fraction())
        )
        env.drawer_trace_history = [current_fraction]
        env.attached_trace_history = [False]


def current_env_handle_local(env: DrawerRobotEnv) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    payload = env.anygrasp_payload()
    handle_center_world = np.asarray(payload["handle_center_world"], dtype=np.float32)
    grasp_pose_world = np.eye(4, dtype=np.float32)
    grasp_pose_world[:3, :3] = np.asarray(
        R.from_quat(ORACLE_PULL_QUAT_XYZW).as_matrix(), dtype=np.float32
    )
    grasp_pose_world[:3, 3] = handle_center_world
    local_handle = env.handle_local_pose(grasp_pose_world)
    return local_handle, {
        "handle_semantics_source": payload.get("handle_semantics_source", "unknown"),
        "handle_center_world": handle_center_world.tolist(),
    }


def _first_step(mask: np.ndarray) -> int | None:
    if mask.size == 0:
        return None
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def classify_failure_stage(
    *,
    success: bool,
    n_steps: int,
    first_near_handle_step: int | None,
    ever_attached: bool,
    first_major_open_step: int | None,
) -> str:
    if n_steps <= 0:
        return "invalid_run"
    if success:
        return "success"
    if first_near_handle_step is None:
        return "never_near_handle"
    if not ever_attached:
        return "near_handle_no_attach"
    if first_major_open_step is None:
        return "attached_no_pull"
    return "pull_no_open"


def summarize_run_from_traces(
    *,
    episode_index: int,
    npz_path: Path,
    source_npz: dict[str, Any],
    source_meta_payload: dict[str, Any],
    replay_mode: str,
    eef_positions: np.ndarray,
    drawer_trace: np.ndarray,
    attached_trace: np.ndarray,
    handle_distance_trace: np.ndarray,
    success: bool,
    attach_step: int | None,
    action_contract: dict[str, Any],
    frames: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    source_eef = source_npz["eef_positions"].astype(np.float32)
    source_drawer = source_npz["absolute_drawer_fraction"].astype(np.float32)
    source_attached = source_npz["attached_trace"].astype(np.bool_)
    source_handle_distance = source_npz["handle_distance_trace"].astype(np.float32)

    n_compare = min(len(eef_positions), len(source_eef))
    pos_error = (
        np.linalg.norm(eef_positions[:n_compare] - source_eef[:n_compare], axis=1)
        if n_compare
        else np.zeros((0,), dtype=np.float32)
    )

    n_handle_compare = min(len(handle_distance_trace), len(source_handle_distance))
    handle_distance_error = (
        np.abs(
            handle_distance_trace[:n_handle_compare]
            - source_handle_distance[:n_handle_compare]
        )
        if n_handle_compare
        else np.zeros((0,), dtype=np.float32)
    )

    attach_threshold = float(action_contract.get("attach_threshold_m", HANDLE_ATTACH_THRESHOLD))
    first_near_handle_step = _first_step(handle_distance_trace <= attach_threshold)
    first_major_open_step = _first_step(drawer_trace >= MAJOR_OPEN_THRESHOLD)
    ever_attached = bool(np.any(attached_trace))
    source_success = bool(source_meta_payload.get("success", False))
    source_ever_attached = bool(
        source_meta_payload.get("ever_attached", bool(np.any(source_attached)))
    )
    source_final_drawer = float(
        source_meta_payload.get(
            "final_drawer_fraction",
            source_drawer[-1] if len(source_drawer) else 0.0,
        )
    )
    source_max_drawer = float(
        source_meta_payload.get(
            "max_drawer_fraction",
            float(np.max(source_drawer)) if len(source_drawer) else 0.0,
        )
    )
    min_handle_distance = (
        float(np.nanmin(handle_distance_trace))
        if len(handle_distance_trace) and not np.all(np.isnan(handle_distance_trace))
        else float("nan")
    )
    failure_stage = classify_failure_stage(
        success=bool(success),
        n_steps=len(drawer_trace),
        first_near_handle_step=first_near_handle_step,
        ever_attached=ever_attached,
        first_major_open_step=first_major_open_step,
    )

    return {
        "episode_index": int(episode_index),
        "npz_path": str(npz_path),
        "replay_mode": replay_mode,
        "script_git_head": git_head(),
        "env_contract": {
            "env_family": "DrawerRobotEnv",
            "action_contract": action_contract,
        },
        "source_success": source_success,
        "source_ever_attached": source_ever_attached,
        "source_max_drawer": source_max_drawer,
        "source_final_drawer": source_final_drawer,
        "success": bool(success),
        "ever_attached": ever_attached,
        "attach_step": int(attach_step) if attach_step is not None else None,
        "first_near_handle_step": first_near_handle_step,
        "first_major_open_step": first_major_open_step,
        "final_drawer_fraction": float(drawer_trace[-1]) if len(drawer_trace) else 0.0,
        "max_drawer_fraction": float(np.max(drawer_trace)) if len(drawer_trace) else 0.0,
        "mean_position_error_m": _safe_mean(pos_error),
        "p95_position_error_m": _safe_percentile(pos_error, 95),
        "max_position_error_m": float(np.max(pos_error)) if len(pos_error) else 0.0,
        "mean_handle_distance_m": _safe_mean(handle_distance_trace),
        "handle_distance_vs_source_error": _safe_mean(handle_distance_error),
        "failure_stage": failure_stage,
        "attach_threshold_m": attach_threshold,
        "min_handle_distance_m": min_handle_distance,
        "n_recorded_steps": int(len(drawer_trace)),
        "frames_captured": int(len(frames or [])),
    }


def run_replay_mode(
    *,
    episode_index: int,
    npz_path: Path,
    source_npz: dict[str, Any],
    source_meta_payload: dict[str, Any],
    replay_mode: str,
    max_steps: int,
    attach_threshold_override: float | None = None,
    image_size: int = 224,
    render_observations: bool = True,
    capture_frames: bool = False,
    capture_trace: bool = False,
) -> dict[str, Any]:
    mode_cfg = mode_definition(replay_mode)
    action_contract = current_env_action_contract(
        source_meta_payload,
        attach_threshold_override=attach_threshold_override,
    )
    env = DrawerRobotEnv(
        seed=int(source_meta_payload["seed"]),
        image_size=image_size,
        max_steps=max_steps,
        action_contract=action_contract,
        render_observations=render_observations,
    )
    actions = source_npz["actions"].astype(np.float32)

    eef_positions: list[np.ndarray] = []
    drawer_trace: list[float] = []
    attached_trace: list[bool] = []
    handle_distance_trace: list[float] = []
    frames: list[np.ndarray] = []
    trace_steps: list[dict[str, Any]] = []
    success = False
    attach_step = None

    try:
        obs = env.reset()
        local_handle, handle_metadata = current_env_handle_local(env)
        env.attachment_local = local_handle
        for step_index in range(max_steps):
            if mode_cfg["restore_robot_state"] and step_index < len(actions):
                _restore_state(
                    env,
                    source_npz,
                    step_index,
                    restore_drawer_fraction=bool(mode_cfg["restore_drawer_fraction"]),
                    restore_source_attachment_state=bool(
                        mode_cfg["restore_source_attachment_state"]
                    ),
                    restore_source_runtime_histories=bool(
                        mode_cfg["restore_source_runtime_histories"]
                    ),
                    pre_step_force_detached=bool(mode_cfg["pre_step_force_detached"]),
                )
            action = (
                actions[step_index]
                if step_index < len(actions)
                else np.zeros((7,), dtype=np.float32)
            )
            if capture_trace:
                trace_steps.append(
                    {
                        "step_index": int(step_index),
                        "obs_image": obs.image.copy(),
                        "obs_image2": obs.image2.copy(),
                        "obs_state": obs.state.copy(),
                        "obs_eef_pos": obs.eef_pos.copy(),
                        "obs_eef_quat": obs.eef_quat.copy()
                        if hasattr(obs, "eef_quat")
                        else None,
                        "obs_task": getattr(obs, "task", None),
                        "obs_drawer_fraction": float(obs.drawer_fraction),
                        "obs_gripper_open": float(obs.gripper_open),
                        "teacher_action": action.copy(),
                    }
                )
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
            if capture_frames:
                frames.append(obs.image.copy())
            if capture_trace:
                trace_steps[-1]["post_step_drawer_fraction"] = float(obs.drawer_fraction)
                trace_steps[-1]["post_step_attached"] = bool(info.get("attached"))
                trace_steps[-1]["post_step_success"] = bool(info.get("is_success", False))
            if info.get("attached") and attach_step is None:
                attach_step = step_index
            success = success or bool(info.get("is_success", False))
            if done:
                break
    finally:
        env.close()

    return summarize_run_from_traces(
        episode_index=episode_index,
        npz_path=npz_path,
        source_npz=source_npz,
        source_meta_payload=source_meta_payload,
        replay_mode=replay_mode,
        eef_positions=np.asarray(eef_positions, dtype=np.float32),
        drawer_trace=np.asarray(drawer_trace, dtype=np.float32),
        attached_trace=np.asarray(attached_trace, dtype=np.bool_),
        handle_distance_trace=np.asarray(handle_distance_trace, dtype=np.float32),
        success=success,
        attach_step=attach_step,
        action_contract=action_contract,
        frames=frames,
    ) | {
        "video_frames": frames,
        "trace_steps": trace_steps,
        "mode_definition": mode_cfg,
        "measurement_image_size_px": int(image_size),
        "render_observations": bool(render_observations),
        **handle_metadata,
    }


def run_oracle_reachability(
    *,
    episode_index: int,
    npz_path: Path,
    source_npz: dict[str, Any],
    source_meta_payload: dict[str, Any],
    max_steps: int,
    attach_threshold_override: float | None = None,
    image_size: int = 224,
    render_observations: bool = True,
    capture_frames: bool = False,
) -> dict[str, Any]:
    env_contract = current_env_action_contract(
        source_meta_payload,
        attach_threshold_override=attach_threshold_override,
    )
    seed = int(source_meta_payload["seed"])

    env = DrawerRobotEnv(
        seed=seed,
        image_size=image_size,
        max_steps=max_steps,
        action_contract=env_contract,
        render_observations=render_observations,
    )
    try:
        env.reset()
        local_handle, handle_metadata = current_env_handle_local(env)
    finally:
        env.close()

    handle_center_world = np.asarray(handle_metadata["handle_center_world"], dtype=np.float32)
    oracle_grasp_pose = np.eye(4, dtype=np.float32)
    oracle_grasp_pose[:3, :3] = np.asarray(
        R.from_quat(ORACLE_PULL_QUAT_XYZW).as_matrix(), dtype=np.float32
    )
    oracle_grasp_pose[:3, 3] = handle_center_world
    rollout = build_native_teacher_rollout(
        seed=seed,
        grasp_pose_world=oracle_grasp_pose,
        branch_config={"action_contract": env_contract},
        episode_index=episode_index,
        max_steps=max_steps,
        grasp_source="gate_b_oracle_reachability",
        image_size=image_size,
        record_images=capture_frames,
        render_observations=render_observations,
    )
    frames = [frame.copy() for frame in rollout["images"]] if capture_frames else []
    return summarize_run_from_traces(
        episode_index=episode_index,
        npz_path=npz_path,
        source_npz=source_npz,
        source_meta_payload=source_meta_payload,
        replay_mode="oracle_reachability",
        eef_positions=rollout["eef_positions"].astype(np.float32),
        drawer_trace=rollout["absolute_drawer_fraction"].astype(np.float32),
        attached_trace=rollout["attached_trace"].astype(np.bool_),
        handle_distance_trace=rollout["handle_distance_trace"].astype(np.float32),
        success=bool(rollout["success"]),
        attach_step=rollout.get("attach_step"),
        action_contract=env_contract,
        frames=frames,
    ) | {
        "video_frames": frames,
        "mode_definition": mode_definition("oracle_reachability"),
        "oracle_definition": dict(ORACLE_DEFINITION),
        "measurement_image_size_px": int(image_size),
        "render_observations": bool(render_observations),
        **handle_metadata,
    }


def flatten_rows(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for episode in episodes:
        base = {
            "episode_index": episode["episode_index"],
            "npz_name": episode["npz_name"],
            "seed": episode["seed"],
            "source_success": episode["source"]["success"],
            "source_ever_attached": episode["source"]["ever_attached"],
            "source_final_drawer": episode["source"]["final_drawer"],
            "source_max_drawer": episode["source"]["max_drawer"],
        }
        for mode, metrics in episode["modes"].items():
            row = dict(base)
            row.update(
                {
                    "replay_mode": mode,
                    "success": metrics["success"],
                    "ever_attached": metrics["ever_attached"],
                    "attach_step": metrics["attach_step"],
                    "first_near_handle_step": metrics["first_near_handle_step"],
                    "first_major_open_step": metrics["first_major_open_step"],
                    "final_drawer_fraction": metrics["final_drawer_fraction"],
                    "max_drawer_fraction": metrics["max_drawer_fraction"],
                    "mean_position_error_m": metrics["mean_position_error_m"],
                    "p95_position_error_m": metrics["p95_position_error_m"],
                    "max_position_error_m": metrics["max_position_error_m"],
                    "mean_handle_distance_m": metrics["mean_handle_distance_m"],
                    "handle_distance_vs_source_error": metrics[
                        "handle_distance_vs_source_error"
                    ],
                    "failure_stage": metrics["failure_stage"],
                    "attach_threshold_m": metrics["attach_threshold_m"],
                    "min_handle_distance_m": metrics["min_handle_distance_m"],
                    "n_recorded_steps": metrics["n_recorded_steps"],
                }
            )
            rows.append(row)
    return rows


def summarize_mode(mode: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_total = len(rows)
    attach_rate = (
        float(sum(1 for row in rows if row["ever_attached"]) / n_total) if n_total else 0.0
    )
    success_rate = (
        float(sum(1 for row in rows if row["success"]) / n_total) if n_total else 0.0
    )
    failure_counts = Counter(row["failure_stage"] for row in rows)
    dominant_failure_stage = (
        sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        if failure_counts
        else "invalid_run"
    )
    return {
        "mode": mode,
        "n_total": n_total,
        "source_success_rate": (
            float(sum(1 for row in rows if row["source_success"]) / n_total)
            if n_total
            else 0.0
        ),
        "attach_rate": attach_rate,
        "success_rate": success_rate,
        "dominant_failure_stage": dominant_failure_stage,
        "failure_stage_counts": dict(sorted(failure_counts.items())),
        "mean_position_error_m": _safe_mean(
            np.asarray([row["mean_position_error_m"] for row in rows], dtype=np.float32)
        ),
        "mean_handle_distance_m": _safe_mean(
            np.asarray([row["mean_handle_distance_m"] for row in rows], dtype=np.float32)
        ),
        "mean_min_handle_distance_m": _safe_mean(
            np.asarray([row["min_handle_distance_m"] for row in rows], dtype=np.float32)
        ),
        "mean_max_drawer_fraction": _safe_mean(
            np.asarray([row["max_drawer_fraction"] for row in rows], dtype=np.float32)
        ),
    }


def decision_support(aggregate: dict[str, Any]) -> tuple[str, str]:
    open_loop_attach = float(aggregate["open_loop_attach_rate"])
    state_anchored_attach = float(aggregate["state_anchored_attach_rate"])
    oracle_rate = float(aggregate["oracle_reachability_rate"])

    if oracle_rate == 0.0:
        return (
            "env_reachability_problem",
            "Oracle reachability did not achieve attach/open in the current env.",
        )
    if open_loop_attach == 0.0 and state_anchored_attach == 0.0 and oracle_rate > 0.0:
        return (
            "physics_or_contract_gap_likely",
            "Both replay modes fail to attach while oracle reachability succeeds.",
        )
    if open_loop_attach == 0.0 and state_anchored_attach > 0.0:
        return (
            "initial_state_mismatch_likely",
            "State-anchored replay attaches while open-loop replay does not.",
        )
    if open_loop_attach > 0.0 and state_anchored_attach > 0.0 and oracle_rate > 0.0:
        return (
            "teacher_actions_executable_policy_gap_remains",
            "Teacher actions and oracle are executable in the current env.",
        )
    return (
        "inconclusive_replay_pipeline",
        "Replay metrics do not match a single discriminative interpretation branch.",
    )


def build_artifact(
    *,
    episodes: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    modes: list[str],
    max_steps: int,
    attach_threshold_override: float | None = None,
) -> dict[str, Any]:
    mode_rows = {mode: [row for row in rows if row["replay_mode"] == mode] for mode in modes}
    mode_results = {mode: summarize_mode(mode, mode_rows[mode]) for mode in modes}
    state_anchored_metrics = mode_results.get("state_anchored") or mode_results.get(
        "state_anchored_legacy",
        {},
    )
    aggregate = {
        "n_total": len(episodes),
        "source_success_rate": _safe_mean(
            np.asarray([episode["source"]["success"] for episode in episodes], dtype=np.float32)
        ),
        "open_loop_attach_rate": mode_results.get("open_loop", {}).get("attach_rate", 0.0),
        "open_loop_success_rate": mode_results.get("open_loop", {}).get("success_rate", 0.0),
        "state_anchored_attach_rate": state_anchored_metrics.get("attach_rate", 0.0),
        "state_anchored_success_rate": state_anchored_metrics.get("success_rate", 0.0),
        "oracle_reachability_rate": mode_results.get("oracle_reachability", {}).get("success_rate", 0.0),
        "dominant_failure_stage_by_mode": {
            mode: metrics["dominant_failure_stage"] for mode, metrics in mode_results.items()
        },
        "mean_position_error_by_mode": {
            mode: metrics["mean_position_error_m"] for mode, metrics in mode_results.items()
        },
        "mean_handle_distance_by_mode": {
            mode: metrics["mean_handle_distance_m"] for mode, metrics in mode_results.items()
        },
    }
    support, explanation = decision_support(aggregate)
    aggregate["decision_support"] = support
    aggregate["decision_support_explanation"] = explanation
    return {
        "experiment": "gate_b_teacher_executability_audit",
        "timestamp": now_ts(),
        "scope_guardrail": (
            "Gate B is a teacher executability audit only; it does not directly determine "
            "policy learnability or final scientific verdicts."
        ),
        "episode_selection": {
            "policy": "canonical_top10_from_gate_a",
            "episodes": [episode["episode_index"] for episode in episodes],
            "max_steps": int(max_steps),
        },
        "env_contract": {
            "env_family": "DrawerRobotEnv",
            "script_git_head": git_head(),
            "active_action_contract": load_json(ACTIVE_ACTION_CONTRACT_PATH, default={}) or {},
            "attach_threshold_override": attach_threshold_override,
            "model_fidelity_grade": "F2",
        },
        "mode_results": mode_results,
        "episodes": episodes,
        "aggregate": aggregate,
    }


def write_markdown_summary(path: Path, artifact: dict[str, Any]) -> None:
    aggregate = artifact["aggregate"]
    lines = [
        "# Gate B Replay Summary",
        "",
        "Generated by `run_gate_b_teacher_replayability.py`.",
        "",
        f"- Timestamp: `{artifact['timestamp']}`",
        f"- Decision support: `{aggregate['decision_support']}`",
        f"- Explanation: {aggregate['decision_support_explanation']}",
        "- Scope guardrail: teacher executability only; not a final policy-learning or root-cause verdict.",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| n_total | {aggregate['n_total']} |",
        f"| source_success_rate | {aggregate['source_success_rate']:.3f} |",
        f"| open_loop_attach_rate | {aggregate['open_loop_attach_rate']:.3f} |",
        f"| open_loop_success_rate | {aggregate['open_loop_success_rate']:.3f} |",
        f"| state_anchored_attach_rate | {aggregate['state_anchored_attach_rate']:.3f} |",
        f"| state_anchored_success_rate | {aggregate['state_anchored_success_rate']:.3f} |",
        f"| oracle_reachability_rate | {aggregate['oracle_reachability_rate']:.3f} |",
        "",
        "## Failure Stages By Mode",
        "",
        "| Mode | Dominant failure stage | Mean pos err (m) | Mean handle dist (m) |",
        "|---|---|---:|---:|",
    ]
    for mode, metrics in artifact["mode_results"].items():
        lines.append(
            f"| {mode} | {metrics['dominant_failure_stage']} | "
            f"{metrics['mean_position_error_m']:.4f} | {metrics['mean_handle_distance_m']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This summary is descriptive only.",
            "- It does not claim a final root cause.",
            "- It does not claim policy learnability or unlearnability.",
            "- It does not eliminate image-domain effects.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def select_video_candidates(episodes: list[dict[str, Any]]) -> list[tuple[int, str]]:
    candidates = []
    for episode in episodes:
        for mode, metrics in episode["modes"].items():
            candidates.append(
                {
                    "episode_index": episode["episode_index"],
                    "mode": mode,
                    "success": bool(metrics["success"]),
                    "ever_attached": bool(metrics["ever_attached"]),
                    "failure_stage": metrics["failure_stage"],
                    "max_drawer_fraction": float(metrics["max_drawer_fraction"]),
                    "mean_handle_distance_m": float(metrics["mean_handle_distance_m"]),
                }
            )
    failures = [item for item in candidates if not item["success"]]
    failures = sorted(
        failures,
        key=lambda item: (
            0 if item["failure_stage"] == "near_handle_no_attach" else 1,
            item["mean_handle_distance_m"],
            -item["max_drawer_fraction"],
        ),
    )
    selected = [(item["episode_index"], item["mode"]) for item in failures[:3]]
    successes = [item for item in candidates if item["ever_attached"] or item["success"]]
    if successes:
        best = sorted(
            successes,
            key=lambda item: (
                int(not item["success"]),
                -item["max_drawer_fraction"],
                item["mean_handle_distance_m"],
            ),
        )[0]
        if (best["episode_index"], best["mode"]) not in selected:
            selected.append((best["episode_index"], best["mode"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate B Teacher Executability Audit")
    parser.add_argument("--episodes", default=None, help="Comma-separated subset of canonical top-10 episode indices")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help="Comma-separated modes")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--attach-threshold",
        type=float,
        default=None,
        help="Override attach threshold for this run without mutating the live action contract",
    )
    parser.add_argument("--write-videos", action="store_true", help="Write mp4s for top 3 failures and top 1 best replay")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    episode_set = selected_episodes(args.episodes)
    modes = selected_modes(args.modes)
    output_dir = args.output_dir
    episodes_dir = output_dir / "episodes"
    videos_dir = output_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    if args.write_videos:
        videos_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    episode_summaries = []

    for episode_index, npz_name in episode_set:
        npz_path = resolve_npz(npz_name)
        source_npz = dict(np.load(npz_path, allow_pickle=True))
        meta = source_meta(npz_path)
        episode_summary = {
            "episode_index": int(episode_index),
            "npz_name": npz_name,
            "npz_path": str(npz_path),
            "seed": int(meta["seed"]),
            "source": {
                "success": bool(meta.get("success", False)),
                "ever_attached": bool(meta.get("ever_attached", False)),
                "final_drawer": float(meta.get("final_drawer_fraction", 0.0)),
                "max_drawer": float(meta.get("max_drawer_fraction", 0.0)),
            },
            "modes": {},
        }

        for mode in modes:
            if mode == "oracle_reachability":
                result = run_oracle_reachability(
                    episode_index=episode_index,
                    npz_path=npz_path,
                    source_npz=source_npz,
                    source_meta_payload=meta,
                    max_steps=args.max_steps,
                    attach_threshold_override=args.attach_threshold,
                    capture_frames=False,
                )
            else:
                result = run_replay_mode(
                    episode_index=episode_index,
                    npz_path=npz_path,
                    source_npz=source_npz,
                    source_meta_payload=meta,
                    replay_mode=mode,
                    max_steps=args.max_steps,
                    attach_threshold_override=args.attach_threshold,
                    capture_frames=False,
                )
            episode_summary["modes"][mode] = {
                key: value
                for key, value in result.items()
                if key != "video_frames"
            }

        save_json(episodes_dir / f"ep{episode_index:03d}.json", episode_summary)
        episode_summaries.append(episode_summary)

    rows = flatten_rows(episode_summaries)
    artifact = build_artifact(
        episodes=episode_summaries,
        rows=rows,
        modes=modes,
        max_steps=args.max_steps,
        attach_threshold_override=args.attach_threshold,
    )
    artifact["elapsed_sec"] = round(time.time() - t0, 3)
    artifact["proposal_only"] = True

    summary_json_path = output_dir / "gate_b_replay_summary.json"
    save_json(summary_json_path, artifact)
    save_json(output_dir / "gate_b_results.json", artifact)
    save_csv(output_dir / "gate_b_replay_episodes.csv", rows)
    write_markdown_summary(output_dir / "gate_b_replay_summary.md", artifact)

    if args.write_videos:
        candidates = set(select_video_candidates(episode_summaries))
        for episode_index, npz_name in episode_set:
            npz_path = resolve_npz(npz_name)
            source_npz = dict(np.load(npz_path, allow_pickle=True))
            meta = source_meta(npz_path)
            for mode in modes:
                if (episode_index, mode) not in candidates:
                    continue
                if mode == "oracle_reachability":
                    result = run_oracle_reachability(
                        episode_index=episode_index,
                        npz_path=npz_path,
                        source_npz=source_npz,
                        source_meta_payload=meta,
                        max_steps=args.max_steps,
                        attach_threshold_override=args.attach_threshold,
                        capture_frames=True,
                    )
                else:
                    result = run_replay_mode(
                        episode_index=episode_index,
                        npz_path=npz_path,
                        source_npz=source_npz,
                        source_meta_payload=meta,
                        replay_mode=mode,
                        max_steps=args.max_steps,
                        attach_threshold_override=args.attach_threshold,
                        capture_frames=True,
                    )
                frames = result.get("video_frames") or []
                if frames:
                    write_video(
                        frames,
                        videos_dir / f"ep{episode_index:03d}_{mode}.mp4",
                    )

    print(json.dumps(
        {
            "artifact": str(summary_json_path),
            "decision_support": artifact["aggregate"]["decision_support"],
            "elapsed_sec": artifact["elapsed_sec"],
            "n_total": artifact["aggregate"]["n_total"],
            "modes": modes,
            "proposal_only": True,
            "attach_threshold_override": args.attach_threshold,
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()

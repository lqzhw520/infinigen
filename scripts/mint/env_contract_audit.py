#!/usr/bin/env python3
"""Audit that the robot drawer env rewards true attach-then-pull behavior."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
from drawer_robot_env import (
    DrawerRobotEnv,
    _action_toward_pose,
    build_oracle_grasp_pose,
)
from mint_common import ARTIFACT_DIR
from strict_success import (
    STRICT_SUCCESS_CONFIG,
    STRICT_SUCCESS_VERSION,
    evaluate_strict_success,
)

ENV_CONTRACT_AUDIT_PATH = ARTIFACT_DIR / "env_contract_audit.json"
NO_ATTACH_MAX_DRAWER_DELTA = 0.02


def _oracle_targets(env: DrawerRobotEnv) -> dict[str, np.ndarray]:
    payload = env.anygrasp_payload()
    base_pose = np.eye(4, dtype=np.float32)
    oracle_pose = build_oracle_grasp_pose(
        base_pose, payload["handle_center_world"].astype(np.float32)
    )
    local_handle = env.handle_local_pose(oracle_pose)
    env.attachment_local = local_handle
    closed_pos, closed_quat = env.local_to_world_pose(
        local_handle["pos"], local_handle["quat"], fraction=0.0
    )
    open_pos, _open_quat = env.local_to_world_pose(
        local_handle["pos"], local_handle["quat"], fraction=0.92
    )
    travel_vec = open_pos - closed_pos
    travel_axis = travel_vec / max(np.linalg.norm(travel_vec), 1e-8)
    pregrasp_pos = (
        closed_pos + 0.10 * travel_axis + np.array([0.0, 0.0, 0.04], dtype=np.float32)
    )
    retreat_pos = (
        open_pos + 0.06 * travel_axis + np.array([0.0, 0.0, 0.05], dtype=np.float32)
    )
    return {
        "closed_pos": closed_pos.astype(np.float32),
        "closed_quat": closed_quat.astype(np.float32),
        "pregrasp_pos": pregrasp_pos.astype(np.float32),
        "pregrasp_quat": closed_quat.astype(np.float32),
        "open_pos": open_pos.astype(np.float32),
        "open_quat": closed_quat.astype(np.float32),
        "retreat_pos": retreat_pos.astype(np.float32),
        "retreat_quat": closed_quat.astype(np.float32),
    }


def _run_phase(
    env: DrawerRobotEnv,
    obs,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    gripper_open: float,
    *,
    max_phase_steps: int = 18,
) -> tuple[Any, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for _ in range(max_phase_steps):
        action = _action_toward_pose(
            obs,
            target_pos,
            target_quat,
            gripper_open,
            action_frame=env.action_frame,
            translation_scale=env.translation_scale,
            rotation_scale=env.rotation_scale,
        )
        obs, _reward, done, info = env.step(action)
        records.append(
            {
                "drawer_fraction": float(info["drawer_fraction"]),
                "attached": bool(info["attached"]),
                "ever_attached": bool(info.get("ever_attached", False)),
                "strict_metrics": info.get("strict_metrics", {}),
                "done": bool(done),
            }
        )
        pos_error = float(np.linalg.norm(target_pos - obs.eef_pos))
        if pos_error < 0.01:
            break
        if done:
            break
    return obs, records


def _probe_seed(seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {"seed": int(seed)}

    no_attach_env = DrawerRobotEnv(seed=seed, image_size=224, max_steps=96)
    try:
        obs = no_attach_env.reset()
        targets = _oracle_targets(no_attach_env)
        drawer_trace = [float(obs.drawer_fraction)]
        attached_trace = [False]
        phase_records: list[dict[str, Any]] = []
        for phase_name, pos_key, quat_key in [
            ("approach_open", "pregrasp_pos", "pregrasp_quat"),
            ("close_open", "closed_pos", "closed_quat"),
            ("pull_open", "open_pos", "open_quat"),
            ("retreat_open", "retreat_pos", "retreat_quat"),
        ]:
            obs, records = _run_phase(
                no_attach_env, obs, targets[pos_key], targets[quat_key], 1.0
            )
            drawer_trace.extend(row["drawer_fraction"] for row in records)
            attached_trace.extend(row["attached"] for row in records)
            phase_records.append({"phase": phase_name, "steps": len(records)})
        strict = evaluate_strict_success(
            np.asarray(drawer_trace, dtype=np.float32),
            np.asarray(attached_trace, dtype=bool),
        )
        result["no_attach_probe"] = {
            "phase_records": phase_records,
            "strict_metrics": strict,
            "passed": bool(
                strict["unattached_drawer_delta_max"] <= NO_ATTACH_MAX_DRAWER_DELTA
            ),
        }
    finally:
        no_attach_env.close()

    attach_env = DrawerRobotEnv(seed=seed, image_size=224, max_steps=96)
    try:
        obs = attach_env.reset()
        targets = _oracle_targets(attach_env)
        drawer_trace = [float(obs.drawer_fraction)]
        attached_trace = [False]
        phase_records = []
        plan = [
            ("approach", "pregrasp_pos", "pregrasp_quat", 1.0),
            ("close", "closed_pos", "closed_quat", 0.0),
            ("attach_hold", "closed_pos", "closed_quat", 0.0),
            ("pull", "open_pos", "open_quat", 0.0),
            ("retreat", "retreat_pos", "retreat_quat", 1.0),
        ]
        for phase_name, pos_key, quat_key, gripper_open in plan:
            obs, records = _run_phase(
                attach_env, obs, targets[pos_key], targets[quat_key], gripper_open
            )
            drawer_trace.extend(row["drawer_fraction"] for row in records)
            attached_trace.extend(row["attached"] for row in records)
            phase_records.append({"phase": phase_name, "steps": len(records)})
        strict = evaluate_strict_success(
            np.asarray(drawer_trace, dtype=np.float32),
            np.asarray(attached_trace, dtype=bool),
        )
        result["attach_probe"] = {
            "phase_records": phase_records,
            "strict_metrics": strict,
            "passed": bool(
                strict["strict_success"] and strict["attach_before_major_open"]
            ),
        }
    finally:
        attach_env.close()

    result["passed"] = bool(
        result["no_attach_probe"]["passed"]
        and result["attach_probe"]["passed"]
        and result["attach_probe"]["strict_metrics"].get(
            "attach_before_major_open", False
        )
    )
    return result


def audit_environment_contract(
    seeds: list[int], *, write_artifact: bool = True
) -> dict[str, Any]:
    unique_seeds = sorted({int(seed) for seed in seeds})
    if ENV_CONTRACT_AUDIT_PATH.exists():
        try:
            cached = json.loads(ENV_CONTRACT_AUDIT_PATH.read_text())
        except json.JSONDecodeError:
            cached = None
        if (
            cached
            and cached.get("sampled_seeds") == unique_seeds
            and cached.get("strict_success_version") == STRICT_SUCCESS_VERSION
        ):
            return cached
    records = [_probe_seed(seed) for seed in unique_seeds]
    failed = [row for row in records if not row.get("passed")]
    payload = {
        "gate": "env_contract_audit",
        "passed": not failed,
        "strict_success_version": STRICT_SUCCESS_VERSION,
        "strict_success_config": STRICT_SUCCESS_CONFIG,
        "no_attach_max_drawer_delta_threshold": NO_ATTACH_MAX_DRAWER_DELTA,
        "sampled_seeds": unique_seeds,
        "records": records,
        "failed_seeds": [row["seed"] for row in failed],
        "timestamp": time.time(),
    }
    if write_artifact:
        ENV_CONTRACT_AUDIT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return payload

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

BUNDLE_SCHEMA_VERSION = "a_plus_local_render_bridge_v1"
BRIDGE_NAME = "a_plus_local_render_bridge"
REPORT_CAMERA_NAME = "report_primary_fixed"
GRIPPER_OPEN = 0.001
GRIPPER_CLOSED = -0.042
LEFT_FINGER_OPEN_QPOS = 0.04
RIGHT_FINGER_OPEN_QPOS = -0.04
ROBOT_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
DRAWER_JOINT_NAMES = ["drawer_slider_0", "drawer_slider_1"]
GRIPPER_JOINT_NAMES = ["finger_joint1", "finger_joint2"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def write_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def git_head(root: str | Path | None = None) -> str:
    cwd = str(root or repo_root())
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=cwd, text=True
    ).strip()


def make_bridge_run_id(prefix: str, working_head_commit: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{working_head_commit[:8]}"


def gripper_scalar_to_finger_qpos(gripper_scalar: float) -> tuple[float, float]:
    aperture = float(
        np.clip(
            (float(gripper_scalar) - GRIPPER_CLOSED)
            / (GRIPPER_OPEN - GRIPPER_CLOSED),
            0.0,
            1.0,
        )
    )
    return LEFT_FINGER_OPEN_QPOS * aperture, RIGHT_FINGER_OPEN_QPOS * aperture


def phase_label_from_info(step: int, info: dict[str, Any]) -> str:
    if bool(info.get("phase_locked", False)):
        phase = "phase_locked"
    elif bool(info.get("stable_attach", False)):
        phase = "stable_attach"
    elif bool(info.get("attach_eligible", False)):
        phase = "attach_eligible"
    elif bool(info.get("attach_gate_orientation_passed", False)):
        phase = "orientation_gate"
    elif bool(info.get("attach_gate_approach_passed", False)):
        phase = "approach_gate"
    elif bool(info.get("attach_gate_distance_passed", False)):
        phase = "distance_gate"
    else:
        phase = "observe_reach"
    return f"{step:03d}:{phase}"


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_error_vec(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    delta = _quat_mul(target, _quat_conjugate(current))
    if delta[0] < 0:
        delta = -delta
    norm = np.linalg.norm(delta[1:])
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(norm, max(delta[0], 1e-8))
    axis = delta[1:] / norm
    return axis * angle


def _joint_qpos_indices(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    return np.asarray(
        [
            model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in joint_names
        ],
        dtype=np.int32,
    )


def _joint_dof_indices(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    return np.asarray(
        [
            model.jnt_dofadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in joint_names
        ],
        dtype=np.int32,
    )


def solve_robot_qpos_trace(
    model: mujoco.MjModel,
    target_pos_trace: np.ndarray,
    target_quat_trace: np.ndarray | None = None,
    *,
    initial_qpos: np.ndarray | None = None,
    max_iter: int = 80,
    damping: float = 1e-3,
    orientation_weight: float = 0.08,
) -> tuple[np.ndarray, dict[str, float]]:
    data = mujoco.MjData(model)
    qpos_idx = _joint_qpos_indices(model, ROBOT_JOINT_NAMES)
    dof_idx = _joint_dof_indices(model, ROBOT_JOINT_NAMES)
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ROBOT_JOINT_NAMES
    ]
    low = model.jnt_range[joint_ids, 0].astype(np.float64)
    high = model.jnt_range[joint_ids, 1].astype(np.float64)
    right_hand_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand")

    q = (
        np.asarray(initial_qpos, dtype=np.float64).copy()
        if initial_qpos is not None
        else np.zeros(7, dtype=np.float64)
    )
    out = np.zeros((len(target_pos_trace), 7), dtype=np.float32)
    pos_errs: list[float] = []
    ori_errs: list[float] = []

    for t, target_pos in enumerate(np.asarray(target_pos_trace, dtype=np.float64)):
        target_quat = (
            np.asarray(target_quat_trace[t], dtype=np.float64)
            if target_quat_trace is not None
            else None
        )
        for _ in range(max_iter):
            data.qpos[qpos_idx] = q
            mujoco.mj_forward(model, data)
            cur_pos = np.asarray(data.xpos[right_hand_body], dtype=np.float64)
            pos_err = target_pos - cur_pos
            if target_quat is not None:
                cur_quat = np.asarray(data.xquat[right_hand_body], dtype=np.float64)
                ori_err = _quat_error_vec(target_quat, cur_quat)
            else:
                ori_err = np.zeros(3, dtype=np.float64)
            err = np.concatenate([pos_err, orientation_weight * ori_err])
            if np.linalg.norm(pos_err) < 2e-3 and np.linalg.norm(ori_err) < 5e-2:
                break

            jacp = np.zeros((3, model.nv), dtype=np.float64)
            jacr = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jacBody(model, data, jacp, jacr, right_hand_body)
            J = np.vstack([jacp[:, dof_idx], orientation_weight * jacr[:, dof_idx]])
            hessian = J.T @ J + damping * np.eye(J.shape[1], dtype=np.float64)
            dq = np.linalg.solve(hessian, J.T @ err)
            q = np.clip(q + dq, low, high)

        data.qpos[qpos_idx] = q
        mujoco.mj_forward(model, data)
        achieved_pos = np.asarray(data.xpos[right_hand_body], dtype=np.float64)
        pos_errs.append(float(np.linalg.norm(target_pos - achieved_pos)))
        if target_quat is not None:
            achieved_quat = np.asarray(data.xquat[right_hand_body], dtype=np.float64)
            ori_errs.append(float(np.linalg.norm(_quat_error_vec(target_quat, achieved_quat))))
        out[t] = q.astype(np.float32)

    report = {
        "ik_mean_position_error": float(np.mean(pos_errs)) if pos_errs else 0.0,
        "ik_max_position_error": float(np.max(pos_errs)) if pos_errs else 0.0,
        "ik_mean_orientation_error": float(np.mean(ori_errs)) if ori_errs else 0.0,
        "ik_max_orientation_error": float(np.max(ori_errs)) if ori_errs else 0.0,
    }
    return out, report


def select_still_indices(phase_labels: list[str], drawer_qpos: np.ndarray) -> dict[str, int]:
    t0_idx = 0
    contact_idx = 0
    for idx, label in enumerate(phase_labels):
        if any(token in label for token in ("approach", "distance", "orientation", "attach")):
            contact_idx = idx
            break
    drawer_progress = np.asarray(drawer_qpos, dtype=np.float32)
    if drawer_progress.ndim == 2 and drawer_progress.shape[1] >= 1:
        max_idx = int(np.argmax(drawer_progress[:, 0]))
    else:
        max_idx = 0
    return {
        "t0": int(t0_idx),
        "contact_approach": int(contact_idx),
        "max_drawer_progress": int(max_idx),
    }

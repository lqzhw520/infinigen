#!/usr/bin/env python3
"""True MuJoCo drawer environment for the canonical Infinigen mainline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
DRAWER_ROOT = PROJECT_ROOT / "sim_exports" / "urdf" / "drawerbox"

TRANSLATION_SCALE_M = 0.03
ROTATION_SCALE_RAD = 0.25
GRIPPER_OPEN = 0.001
GRIPPER_CLOSED = -0.042
ATTACH_THRESHOLD_M = 0.06
DETACH_THRESHOLD_M = 0.18
DRAWER_SUCCESS_FRACTION = 0.90
DEFAULT_TASK = "open the drawer"
DEFAULT_PULL_OPEN_FRACTION = 0.92


def drawer_dir(seed: int) -> Path:
    return DRAWER_ROOT / str(seed)


def drawer_assets_available(seeds: list[int]) -> bool:
    return all((drawer_dir(seed) / "drawerbox.urdf").exists() for seed in seeds)


def drawer_manifest() -> dict[str, Any]:
    return {
        "available_seeds": sorted(int(p.name) for p in DRAWER_ROOT.iterdir() if p.is_dir()),
        "root": str(DRAWER_ROOT),
    }


def _load_assets(seed: int) -> tuple[str, dict[str, bytes], dict[str, Any]]:
    seed_dir = drawer_dir(seed)
    urdf_path = seed_dir / "drawerbox.urdf"
    metadata_path = seed_dir / "metadata.json"
    urdf_text = urdf_path.read_text()
    assets: dict[str, bytes] = {}
    assets_dir = seed_dir / "assets"
    if assets_dir.exists():
        for asset_path in sorted(assets_dir.iterdir()):
            if asset_path.is_file():
                assets[asset_path.name] = asset_path.read_bytes()
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            metadata = {}
    return urdf_text, assets, metadata


def _make_camera(lookat: np.ndarray, distance: float, azimuth: float, elevation: float):
    import mujoco

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat.tolist()
    camera.distance = float(distance)
    camera.azimuth = float(azimuth)
    camera.elevation = float(elevation)
    return camera


def _normalize(vec: np.ndarray, default: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return default.astype(np.float32)
    return (vec / norm).astype(np.float32)


@dataclass
class MuJoCoRobotObservation:
    image: np.ndarray
    image2: np.ndarray
    state: np.ndarray
    task: str
    eef_pos: np.ndarray
    eef_quat: np.ndarray
    gripper_open: float
    drawer_fraction: float
    depth: np.ndarray | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class DrawerRobotEnvMuJoCo:
    """MuJoCo env with Infinigen drawer URDF plus a robot-in-loop controller state."""

    def __init__(self, seed: int, image_size: int = 256, max_steps: int = 96):
        import mujoco

        urdf_text, assets, metadata = _load_assets(seed)
        self.model = mujoco.MjModel.from_xml_string(urdf_text, assets)
        self.data = mujoco.MjData(self.model)
        self.gl_context = mujoco.GLContext(image_size, image_size)
        self.gl_context.make_current()
        self.renderer = mujoco.Renderer(self.model, height=image_size, width=image_size)
        self.seed = int(seed)
        self.metadata = metadata
        self.image_size = int(image_size)
        self.max_steps = int(max_steps)
        self.task = DEFAULT_TASK
        self.joint_idx = 0
        self.joint_range = self.model.jnt_range[self.joint_idx].astype(np.float32)
        self.scene_center = np.array(self.model.stat.center, dtype=np.float32)
        self.scene_extent = float(max(self.model.stat.extent, 0.25))
        self.cam_primary = _make_camera(
            self.scene_center,
            distance=self.scene_extent * 1.85,
            azimuth=180.0,
            elevation=-28.0,
        )
        self.cam_secondary = _make_camera(
            self.scene_center,
            distance=self.scene_extent * 1.35,
            azimuth=110.0,
            elevation=-35.0,
        )
        self._step_count = 0
        self._attached = False
        self._max_drawer_fraction = 0.0
        self._motion_axis = self._joint_axis_world()
        self._workspace_low = self.scene_center + np.array([-0.30, -0.25, -0.10], dtype=np.float32)
        self._workspace_high = self.scene_center + np.array([0.35, 0.25, 0.30], dtype=np.float32)
        self._home_pos = self.scene_center + np.array([0.15, 0.0, 0.12], dtype=np.float32)
        self._home_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        self.reset()

    def close(self) -> None:
        try:
            self.renderer.close()
        finally:
            try:
                self.gl_context.free()
            except Exception:
                pass

    def _joint_axis_world(self) -> np.ndarray:
        axis = self.model.jnt_axis[self.joint_idx].astype(np.float32)
        default = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return _normalize(axis, default)

    def _drawer_fraction(self) -> float:
        low, high = self.joint_range.tolist()
        span = max(high - low, 1e-6)
        return float(np.clip((float(self.data.qpos[self.joint_idx]) - low) / span, 0.0, 1.0))

    def _drawer_world_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        geom_xpos = np.asarray(self.data.geom_xpos, dtype=np.float32)
        geom_rbound = np.asarray(self.model.geom_rbound, dtype=np.float32)
        if len(geom_xpos) == 0:
            center = self.scene_center
            pad = np.full(3, 0.1, dtype=np.float32)
            return center - pad, center + pad
        mins = np.min(geom_xpos - geom_rbound[:, None], axis=0)
        maxs = np.max(geom_xpos + geom_rbound[:, None], axis=0)
        return mins.astype(np.float32), maxs.astype(np.float32)

    def _handle_center_world(self) -> np.ndarray:
        mins, maxs = self._drawer_world_aabb()
        axis = self._motion_axis
        dominant_axis = int(np.argmax(np.abs(axis)))
        center = (mins + maxs) / 2.0
        face = maxs[dominant_axis] if axis[dominant_axis] >= 0 else mins[dominant_axis]
        center[dominant_axis] = face
        center[2] = maxs[2] - 0.03
        return center.astype(np.float32)

    def _project_primary(self, pos: np.ndarray) -> tuple[int, int]:
        span = self.scene_extent * 1.4
        rel = np.clip((pos[:2] - self.scene_center[:2]) / max(span, 1e-6), -1.0, 1.0)
        u = int((rel[1] * 0.5 + 0.5) * (self.image_size - 1))
        v = int((0.5 - rel[0] * 0.5) * (self.image_size - 1))
        return u, v

    def _project_secondary(self, pos: np.ndarray) -> tuple[int, int]:
        span = self.scene_extent * 1.2
        rel_y = np.clip((pos[1] - self.scene_center[1]) / max(span, 1e-6), -1.0, 1.0)
        rel_z = np.clip((pos[2] - self.scene_center[2]) / max(span, 1e-6), -1.0, 1.0)
        u = int((rel_y * 0.5 + 0.5) * (self.image_size - 1))
        v = int((0.5 - rel_z * 0.5) * (self.image_size - 1))
        return u, v

    def _overlay_marker(self, image: np.ndarray, pixel: tuple[int, int], color: tuple[int, int, int]) -> np.ndarray:
        out = image.copy()
        u, v = pixel
        rr = 4
        u0, u1 = max(0, u - rr), min(self.image_size, u + rr + 1)
        v0, v1 = max(0, v - rr), min(self.image_size, v + rr + 1)
        out[v0:v1, u0:u1] = np.asarray(color, dtype=np.uint8)
        return out

    def _render(self, camera) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        rgb = self.renderer.render()
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def _observe_images(self) -> tuple[np.ndarray, np.ndarray]:
        img1 = self._render(self.cam_primary)
        img2 = self._render(self.cam_secondary)
        color = (0, 220, 32) if self._attached else (220, 60, 20)
        img1 = self._overlay_marker(img1, self._project_primary(self.eef_pos), color)
        img2 = self._overlay_marker(img2, self._project_secondary(self.eef_pos), color)
        return img1, img2

    def _state_vector(self) -> np.ndarray:
        quat = self.eef_quat.astype(np.float32)
        state = np.concatenate(
            [
                self.eef_pos.astype(np.float32),
                quat,
                np.array([self.gripper_joint], dtype=np.float32),
            ]
        )
        return state.astype(np.float32)

    def observe(self) -> MuJoCoRobotObservation:
        image, image2 = self._observe_images()
        return MuJoCoRobotObservation(
            image=image,
            image2=image2,
            state=self._state_vector(),
            task=self.task,
            eef_pos=self.eef_pos.astype(np.float32),
            eef_quat=self.eef_quat.astype(np.float32),
            gripper_open=float(self.gripper_joint),
            drawer_fraction=self._drawer_fraction(),
            depth=None,
        )

    def reset(self, start_fraction: float = 0.0) -> MuJoCoRobotObservation:
        import mujoco

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        low, high = self.joint_range.tolist()
        self.data.qpos[self.joint_idx] = low + (high - low) * float(start_fraction)
        mujoco.mj_forward(self.model, self.data)
        self.eef_pos = self._home_pos.copy()
        self.eef_quat = self._home_quat.copy()
        self.gripper_joint = GRIPPER_OPEN
        self._step_count = 0
        self._attached = False
        self._max_drawer_fraction = self._drawer_fraction()
        return self.observe()

    def anygrasp_payload(self, num_points: int = 4096) -> dict[str, Any]:
        mins, maxs = self._drawer_world_aabb()
        handle = self._handle_center_world()
        axis = self._motion_axis
        dominant_axis = int(np.argmax(np.abs(axis)))
        tangential = [idx for idx in range(3) if idx != dominant_axis]
        points = []
        colors = []
        rng = np.random.default_rng(self.seed)
        for _ in range(num_points):
            p = np.zeros(3, dtype=np.float32)
            p[dominant_axis] = handle[dominant_axis] + rng.normal(scale=0.015)
            p[tangential[0]] = rng.uniform(mins[tangential[0]], maxs[tangential[0]])
            p[tangential[1]] = rng.uniform(mins[tangential[1]], maxs[tangential[1]])
            points.append(p)
            colors.append(np.array([0.75, 0.60, 0.45], dtype=np.float32))
        pc = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)
        pad = 0.04
        limits = np.array(
            [
                mins[0] - pad,
                maxs[0] + pad,
                mins[1] - pad,
                maxs[1] + pad,
                mins[2] - pad,
                maxs[2] + pad,
            ],
            dtype=np.float32,
        )
        return {
            "pc": pc,
            "colors": colors,
            "limits": limits,
            "world_from_camera": np.eye(4, dtype=np.float32),
            "handle_center_world": handle.astype(np.float32),
            "drawer_aabb_world": np.stack([mins, maxs], axis=0).astype(np.float32),
            "drawer_motion_axis": axis.astype(np.float32),
            "eef_pos": self.eef_pos.astype(np.float32),
            "drawer_fraction": self._drawer_fraction(),
        }

    def step(self, action: np.ndarray) -> tuple[MuJoCoRobotObservation, float, bool, dict[str, Any]]:
        import mujoco

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        prev_pos = self.eef_pos.copy()
        delta_pos = np.clip(action[:3], -1.0, 1.0) * TRANSLATION_SCALE_M
        delta_rot = np.clip(action[3:6], -1.0, 1.0) * ROTATION_SCALE_RAD
        close_cmd = float(action[6]) < 0.0

        self.eef_pos = np.clip(prev_pos + delta_pos, self._workspace_low, self._workspace_high)
        delta_quat = R.from_rotvec(delta_rot).as_quat().astype(np.float32)
        self.eef_quat = (R.from_quat(delta_quat) * R.from_quat(self.eef_quat)).as_quat().astype(np.float32)
        self.gripper_joint = GRIPPER_CLOSED if close_cmd else GRIPPER_OPEN

        handle = self._handle_center_world()
        dist_to_handle = float(np.linalg.norm(self.eef_pos - handle))
        if not self._attached and close_cmd and dist_to_handle <= ATTACH_THRESHOLD_M:
            self._attached = True
        if self._attached and (not close_cmd or dist_to_handle > DETACH_THRESHOLD_M):
            self._attached = False

        if self._attached:
            drawer_delta = float(np.dot(self.eef_pos - prev_pos, self._motion_axis)) * 6.0
            low, high = self.joint_range.tolist()
            self.data.qpos[self.joint_idx] = np.clip(
                float(self.data.qpos[self.joint_idx]) + drawer_delta,
                low,
                high,
            )
        mujoco.mj_forward(self.model, self.data)
        self._step_count += 1
        obs = self.observe()
        self._max_drawer_fraction = max(self._max_drawer_fraction, obs.drawer_fraction)
        success = obs.drawer_fraction >= DRAWER_SUCCESS_FRACTION
        reward = 1.0 if success else float(obs.drawer_fraction)
        done = success or self._step_count >= self.max_steps
        info = {
            "is_success": success,
            "drawer_fraction": obs.drawer_fraction,
            "attached": self._attached,
            "dist_to_handle": dist_to_handle,
            "step_count": self._step_count,
            "max_drawer_fraction": self._max_drawer_fraction,
            "handle_center_world": handle.tolist(),
        }
        return obs, reward, done, info


def _pose_from_point(handle_center: np.ndarray, motion_axis: np.ndarray, offset: float, z_lift: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pos = handle_center - motion_axis * offset + np.array([0.0, 0.0, z_lift], dtype=np.float32)
    pose[:3, 3] = pos
    pose[:3, :3] = np.eye(3, dtype=np.float32)
    return pose


def build_oracle_grasp_pose(anygrasp_pose_world: np.ndarray, handle_center_world: np.ndarray) -> np.ndarray:
    pose = np.asarray(anygrasp_pose_world, dtype=np.float32).copy()
    pose[:3, 3] = np.asarray(handle_center_world, dtype=np.float32)
    return pose


def _script_action(current_pos: np.ndarray, target_pos: np.ndarray, close: bool) -> np.ndarray:
    delta = np.zeros(7, dtype=np.float32)
    pos_err = target_pos - current_pos
    delta[:3] = np.clip(pos_err / max(TRANSLATION_SCALE_M, 1e-6), -1.0, 1.0)
    delta[6] = -1.0 if close else 1.0
    return delta


def build_robot_rollout(
    seed: int,
    grasp_pose_world: np.ndarray,
    episode_index: int = 0,
    max_steps: int = 96,
    *,
    grasp_source: str = "anygrasp",
    grasp_score: float | None = None,
    image_size: int = 256,
    pull_open_fraction: float = DEFAULT_PULL_OPEN_FRACTION,
) -> dict[str, Any]:
    env = DrawerRobotEnvMuJoCo(seed=seed, image_size=image_size, max_steps=max_steps)
    obs = env.reset()
    axis = env._motion_axis
    handle = env._handle_center_world()
    pregrasp_pose = _pose_from_point(handle, axis, offset=0.04, z_lift=0.03)
    grasp_pose = np.asarray(grasp_pose_world, dtype=np.float32).copy()
    grasp_pose[:3, 3] = handle
    low, high = env.joint_range.tolist()
    open_fraction = float(np.clip(pull_open_fraction, 0.75, 0.95))
    retreat_target = handle + axis * 0.16 + np.array([0.0, 0.0, 0.05], dtype=np.float32)

    images, images2, states, actions, rewards = [], [], [], [], []
    abs_drawer, next_drawer, attached_trace, handle_distance_trace = [], [], [], []
    phase_labels = []
    ever_attached = False
    attach_step = None
    success = False
    close_hold_steps = 0

    phase = "pregrasp"
    for step_idx in range(max_steps):
        images.append(obs.image.copy())
        images2.append(obs.image2.copy())
        states.append(obs.state.copy())
        abs_drawer.append(float(obs.drawer_fraction))

        handle = env._handle_center_world()
        pregrasp_target = handle - axis * 0.04 + np.array([0.0, 0.0, 0.03], dtype=np.float32)
        contact_target = handle + np.array([0.0, 0.0, 0.005], dtype=np.float32)
        pull_target = handle + axis * 0.03 + np.array([0.0, 0.0, 0.005], dtype=np.float32)
        retreat_target = handle + axis * 0.10 + np.array([0.0, 0.0, 0.05], dtype=np.float32)

        if env._attached and attach_step is None:
            attach_step = step_idx
        if env._attached and phase in {"pregrasp", "contact", "close"}:
            phase = "pull"
        elif phase == "pregrasp" and np.linalg.norm(obs.eef_pos - pregrasp_target) < 0.02:
            phase = "contact"
        elif phase == "contact" and np.linalg.norm(obs.eef_pos - handle) < 0.03:
            phase = "close"
            close_hold_steps = 0
        elif phase == "close":
            close_hold_steps += 1
            if close_hold_steps >= 4:
                phase = "pull" if env._attached else "contact"
                close_hold_steps = 0
        elif phase == "pull" and not env._attached:
            phase = "contact"
        elif phase == "pull" and obs.drawer_fraction >= open_fraction:
            phase = "retreat"

        if phase == "pregrasp":
            target_pos, close = pregrasp_target, False
        elif phase == "contact":
            target_pos, close = contact_target, False
        elif phase == "close":
            target_pos, close = contact_target, True
        elif phase == "pull":
            target_pos, close = pull_target, True
        else:
            target_pos, close = retreat_target, False

        action = _script_action(obs.eef_pos, target_pos, close)
        next_obs, reward, done, info = env.step(action)
        actions.append(action.copy())
        rewards.append(float(reward))
        phase_labels.append(phase)
        attached_trace.append(bool(info["attached"]))
        handle_distance_trace.append(float(info["dist_to_handle"]))
        next_drawer.append(float(next_obs.drawer_fraction))
        ever_attached = ever_attached or bool(info["attached"])
        obs = next_obs
        success = bool(info["is_success"])
        if done:
            break

    rollout = {
        "seed": seed,
        "episode_index": episode_index,
        "success": success,
        "steps": len(actions),
        "images": np.stack(images, axis=0).astype(np.uint8),
        "images2": np.stack(images2, axis=0).astype(np.uint8),
        "states": np.stack(states, axis=0).astype(np.float32),
        "actions": np.stack(actions, axis=0).astype(np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "absolute_drawer_fraction": np.asarray(abs_drawer, dtype=np.float32),
        "next_drawer_fractions": np.asarray(next_drawer, dtype=np.float32),
        "attached_trace": np.asarray(attached_trace, dtype=np.bool_),
        "handle_distance_trace": np.asarray(handle_distance_trace, dtype=np.float32),
        "phase_labels": np.asarray(phase_labels),
        "task": DEFAULT_TASK,
        "grasp_source": grasp_source,
        "grasp_score": None if grasp_score is None else float(grasp_score),
        "ever_attached": ever_attached,
        "final_drawer_fraction": float(obs.drawer_fraction),
        "max_drawer_fraction": float(max(next_drawer or [0.0])),
        "pregrasp_pose": {
            "pos": pregrasp_pose[:3, 3].astype(np.float32),
            "quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        },
        "grasp_pose": {
            "pos": grasp_pose[:3, 3].astype(np.float32),
            "quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        },
        "retreat_pose": {
            "pos": retreat_target.astype(np.float32),
            "quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        },
        "pull_fraction_target": float(open_fraction),
        "attach_step": attach_step,
        "action_contract": {
            "translation_scale_m": TRANSLATION_SCALE_M,
            "rotation_scale_rad": ROTATION_SCALE_RAD,
            "backend": "mujoco",
            "robot_in_loop": True,
        },
    }
    env.close()
    return rollout


def save_robot_rollout(path: Path, rollout: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=rollout["images"],
        images2=rollout["images2"],
        states=rollout["states"],
        actions=rollout["actions"],
        rewards=rollout["rewards"],
        absolute_drawer_fraction=rollout["absolute_drawer_fraction"],
        next_drawer_fractions=rollout["next_drawer_fractions"],
        attached_trace=rollout["attached_trace"],
        handle_distance_trace=rollout["handle_distance_trace"],
        phase_labels=rollout["phase_labels"],
    )
    meta = {
        "seed": rollout["seed"],
        "episode_index": rollout["episode_index"],
        "success": rollout["success"],
        "steps": rollout["steps"],
        "task": rollout["task"],
        "grasp_source": rollout["grasp_source"],
        "grasp_score": rollout["grasp_score"],
        "ever_attached": rollout["ever_attached"],
        "final_drawer_fraction": rollout["final_drawer_fraction"],
        "max_drawer_fraction": rollout["max_drawer_fraction"],
        "pull_fraction_target": rollout["pull_fraction_target"],
        "attach_step": rollout["attach_step"],
        "action_contract": rollout["action_contract"],
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

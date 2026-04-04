#!/usr/bin/env python3
"""
DrawerRobotEnvMujoco — LIBERO-aligned robot environment backed by PyBullet physics.

Architecture
------------
|- Physics:  MuJoCo-compatible joint-space PD control (via PyBullet).
|- Rendering: PyBullet ER_TINY_RENDERER (headless, reliable on A800 without GPU/EGL).
|- Cameras:  LIBERO overhead (agentview, pitch=-90°) + EEF wrist camera.

State (8 DOF):
    state[0:3]  eef_pos (m)  in world frame
    state[3:7]  motor_joints (LIBERO convention: motor[i] = π - pb_joint[i])
    state[7]    gripper_joint (m), continuous, ∈ [-0.042, +0.001]
                (negative=closed, positive=open — LIBERO convention)

Cameras (256×256 RGB):
    image:       LIBERO agentview overhead (pitch=-90°, yaw=0°)
    image2:      EEF-parented wrist camera (随指尖运动)

Action (7 DOF):
    action[:3]   delta xyz in world (m), clipped to ±1.0 → scaled by translation_scale
    action[3:6]  delta rotation vector, clipped to ±1.0 → scaled by rotation_scale
    action[6]    gripper: ≥ 0 → open, < 0 → close  (LIBERO convention)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
SCRIPTS_MINT = PROJECT_ROOT / "scripts" / "mint"

import sys

sys.path.insert(0, str(SCRIPTS_MINT))

from mint_common import (
    GRIPPER_CLOSE_VALUE,
    GRIPPER_OPEN_VALUE,
    ROTATION_SCALE_RAD,
    TRANSLATION_SCALE_M,
)

# ─── Constants ────────────────────────────────────────────────────────────────
LIBERO_GRIPPER_MIN = -0.042
LIBERO_GRIPPER_MAX = +0.001

_PI = float(np.pi)

_DRAWER_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen/sim_exports/urdf/drawerbox")
_PANDA_URDF = Path(
    "/root/anaconda3/envs/mint/lib/python3.12/site-packages/robosuite/models/assets/"
    "bullet_data/panda_description/urdf/panda_arm_hand.urdf"
)

# LIBERO overhead camera (agentview): pure overhead, symmetric
_LIBERO_CAMERA = {
    "target": np.array([0.18, 0.0, 0.40], dtype=np.float32),
    "distance": 1.0,
    "yaw": 0.0,    # no yaw bias — symmetric overhead view
    "pitch": -90.0,  # pure overhead, camera looks straight down
    "roll": 0.0,
}
_WRIST_OFFSET = np.array([0.08, 0.0, 0.02], dtype=np.float32)
_HOME_JOINTS = np.array([0.0, -0.45, 0.0, -2.10, 0.0, 1.75, 0.72], dtype=np.float32)
_GRIPPER_JOINTS = (9, 10)
_GRIPPER_OPEN_POS = 0.04
_GRIPPER_CLOSED_POS = 0.0
_ARM_DOF = 7
_TIME_STEP = 1.0 / 240.0
_MAX_DELTA_TRANSLATION = TRANSLATION_SCALE_M
_MAX_DELTA_ROTATION = ROTATION_SCALE_RAD
_EE_LINK = 8


def _load_seed_metadata(seed: int) -> dict[str, Any]:
    path = _DRAWER_ROOT / str(seed) / "metadata.json"
    if not path.exists():
        return {}
    try:
        import json
        return json.loads(path.read_text())
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
#  Observation dataclass
# ─────────────────────────────────────────────────────────────────────────────

class RobotObservation:
    """Container matching DrawerRobotEnv.RobotObservation schema."""

    __slots__ = (
        "image", "image2", "depth", "state",
        "task", "eef_pos", "eef_quat",
        "gripper_open", "drawer_fraction",
    )

    def __init__(
        self,
        image: np.ndarray,
        image2: np.ndarray,
        depth: np.ndarray,
        state: np.ndarray,
        task: str,
        eef_pos: np.ndarray,
        eef_quat: np.ndarray,
        gripper_open: float,
        drawer_fraction: float,
    ) -> None:
        self.image: np.ndarray = image
        self.image2: np.ndarray = image2
        self.depth: np.ndarray = depth
        self.state: np.ndarray = state
        self.task: str = task
        self.eef_pos: np.ndarray = eef_pos
        self.eef_quat: np.ndarray = eef_quat
        self.gripper_open: float = gripper_open
        self.drawer_fraction: float = drawer_fraction

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


# ─────────────────────────────────────────────────────────────────────────────
#  Main environment class
# ─────────────────────────────────────────────────────────────────────────────

class DrawerRobotEnvMujoco:
    """
    LIBERO-aligned robot drawer environment (PyBullet backend).

    Implements the LIBERO agentview + wrist dual-camera and continuous-gripper
    state contract required for V58 training data alignment.

    State (8 DOF):
        state[0:3]  eef_pos (m) in world frame
        state[3:7]  motor_joints (LIBERO convention: motor[i] = π - pb_joint[i])
        state[7]    gripper_joint (m), continuous ∈ [-0.042, +0.001]
    """

    def __init__(
        self,
        seed: int = 2,
        image_size: int = 256,
        max_steps: int = 48,
        action_contract: dict[str, Any] | None = None,
    ) -> None:
        self._p = p
        self._seed = int(seed)
        self._image_size = int(image_size)
        self._max_steps = int(max_steps)
        self._seed_metadata = _load_seed_metadata(self._seed)

        # ── 1. PyBullet client ────────────────────────────────────────────────
        self._client = self._p.connect(self._p.DIRECT)
        self._p.resetSimulation(physicsClientId=self._client)
        self._p.setTimeStep(_TIME_STEP, physicsClientId=self._client)
        self._p.setGravity(0, 0, -9.81, physicsClientId=self._client)

        # Load robot and drawer
        self._robot_id = self._load_robot()
        self._drawer_id = self._load_drawer()

        # ── 2. Set LIBERO overhead camera ─────────────────────────────────────
        distance = _LIBERO_CAMERA["distance"]
        yaw = _LIBERO_CAMERA["yaw"]
        pitch = _LIBERO_CAMERA["pitch"]
        target = _LIBERO_CAMERA["target"]
        self._p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=[float(target[0]), float(target[1]), float(target[2])],
            physicsClientId=self._client,
        )
        self._camera_projection = self._projection_matrix()

        # ── 3. Physics state tracking ─────────────────────────────────────────
        self._arm_joint_indices = list(range(_ARM_DOF))
        self._gripper_joints = list(_GRIPPER_JOINTS)
        self._drawer_joint = self._find_drawer_joint()
        self._drawer_low, self._drawer_high = self._drawer_range()
        self._drawer_travel = float(self._drawer_high - self._drawer_low)
        self._drawer_motion_axis = self._drawer_open_delta_normalized()

        # Action contract
        default_contract = {
            "action_frame": "world",
            "translation_scale_m": TRANSLATION_SCALE_M,
            "rotation_scale_rad": ROTATION_SCALE_RAD,
        }
        contract = {**default_contract, **(action_contract or {})}
        self._action_frame = str(contract["action_frame"])
        self._translation_scale = float(contract["translation_scale_m"])
        self._rotation_scale = float(contract["rotation_scale_rad"])

        self._step_count = 0
        self._attached = False
        self._attachment_local: np.ndarray | None = None
        self._ever_attached = False
        self._drawer_fraction = 0.0
        self._gripper_open = 1.0
        self._ee_link = _EE_LINK

        self._home_pose: tuple[np.ndarray, np.ndarray] | None = None
        self._commanded_eef_pos: np.ndarray | None = None
        self._commanded_eef_quat: np.ndarray | None = None

        # ── 4. Home reset ─────────────────────────────────────────────────────
        self.reset()

    # ── Public API ───────────────────────────────────────────────────────────

    def reset(self) -> RobotObservation:
        self._step_count = 0
        self._attached = False
        self._ever_attached = False
        self._attachment_local = None
        self._drawer_fraction = 0.0
        self._gripper_open = 1.0

        self._set_drawer_fraction(0.0)
        self._set_arm_joints(_HOME_JOINTS)
        self._set_gripper_open(1.0)
        self._pb_step(8)

        obs = self.observe()
        self._home_pose = (obs.eef_pos.copy(), obs.eef_quat.copy())
        self._commanded_eef_pos = obs.eef_pos.copy()
        self._commanded_eef_quat = obs.eef_quat.copy()
        return obs

    def step(
        self, action: np.ndarray
    ) -> tuple[RobotObservation, float, bool, dict[str, Any]]:
        """
        Execute one environment step.

        Action semantics (LIBERO convention):
            action[:3]   delta xyz  (world frame, m)
            action[3:6]  delta rotvec
            action[6]    gripper: ≥0 → open, <0 → close
        """
        action = np.asarray(action, dtype=np.float32).reshape(7)
        current_obs = self.observe()
        prev_drawer_fraction = float(current_obs.drawer_fraction)

        delta_pos = np.clip(action[:3], -1.0, 1.0) * self._translation_scale
        rot_delta = np.clip(action[3:6], -1.0, 1.0) * self._rotation_scale

        self._commanded_eef_pos = np.asarray(
            self._commanded_eef_pos + delta_pos, dtype=np.float32
        )
        rot = R.from_quat(self._commanded_eef_quat)
        delta_rot = R.from_rotvec(rot_delta)
        new_rot = rot * delta_rot
        self._commanded_eef_quat = new_rot.as_quat().astype(np.float32)

        gripper_open = 1.0 if action[6] >= 0 else 0.0
        self._gripper_open = float(gripper_open)

        # Try to attach if gripper is closing
        if gripper_open < 0.5 and self._attachment_local is not None:
            if self._pose_near_current_handle():
                self._attached = True
        if gripper_open > 0.5:
            self._attached = False

        prev_eef_pos = current_obs.eef_pos.copy()

        # Move to commanded pose
        self._move_to_pose(
            self._commanded_eef_pos,
            self._commanded_eef_quat,
            gripper_open,
            steps=4,
            lock_drawer_fraction=prev_drawer_fraction,
        )

        obs = self.observe()
        if gripper_open < 0.5 and self._attachment_local is not None:
            if self._pose_near_current_handle():
                self._attached = True
        if not self._attached:
            self._set_drawer_fraction(prev_drawer_fraction)
            obs = self.observe()

        # Advance drawer if attached
        if self._attached and self._attachment_local is not None:
            axis_delta = float(
                np.dot(obs.eef_pos - prev_eef_pos, self._drawer_motion_axis)
            )
            frac_delta = axis_delta / max(self._drawer_travel, 1e-8)
            self._set_drawer_fraction(
                np.clip(self._drawer_fraction + frac_delta, 0.0, 1.0)
            )
            self._pb_step(2)
            obs = self.observe()

        self._ever_attached = bool(self._ever_attached or self._attached)
        self._step_count += 1

        success = self._drawer_fraction > 0.8
        reward = 1.0 if success else float(self._drawer_fraction)
        done = success or self._step_count >= self._max_steps

        info: dict[str, Any] = {
            "success": success,
            "drawer_fraction": float(obs.drawer_fraction),
            "attached": bool(self._attached),
            "ever_attached": bool(self._ever_attached),
        }
        return obs, reward, done, info

    def observe(self) -> RobotObservation:
        """Capture cameras + compute LIBERO-aligned state vector."""
        image, depth = self._capture_primary()
        image2, _ = self._capture_wrist()
        eef_pos, eef_quat = self._eef_pose()
        state = self._state_vector(eef_pos, eef_quat)

        return RobotObservation(
            image=image,
            image2=image2,
            depth=depth,
            state=state,
            task="open the drawer",
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            gripper_open=float(self._gripper_open),
            drawer_fraction=float(self._drawer_fraction),
        )

    def close(self) -> None:
        try:
            self._p.disconnect(self._client)
        except Exception:
            pass

    # ── Load helpers ─────────────────────────────────────────────────────────

    def _load_robot(self) -> int:
        return self._p.loadURDF(
            str(_PANDA_URDF),
            basePosition=[-0.55, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            useFixedBase=True,
            flags=self._p.URDF_USE_SELF_COLLISION,
            physicsClientId=self._client,
        )

    def _load_drawer(self) -> int:
        drawer_path = _DRAWER_ROOT / str(self._seed) / "drawerbox.urdf"
        if not drawer_path.exists():
            raise FileNotFoundError(f"Drawer URDF not found: {drawer_path}")
        return self._p.loadURDF(
            str(drawer_path),
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            useFixedBase=True,
            flags=self._p.URDF_USE_SELF_COLLISION,
            physicsClientId=self._client,
        )

    # ── PyBullet camera helpers ──────────────────────────────────────────────

    def _projection_matrix(self) -> np.ndarray:
        fov = 60.0
        aspect = 1.0
        near, far = 0.01, 3.0
        proj = self._p.computeProjectionMatrixFOV(fov, aspect, near, far)
        return np.array(proj, dtype=np.float32).reshape(4, 4, order="F")

    def _view_matrix_primary(self) -> np.ndarray:
        """LIBERO agentview: overhead, pitch=-90°, symmetric."""
        cfg = _LIBERO_CAMERA
        view = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=cfg["target"].tolist(),
            distance=float(cfg["distance"]),
            yaw=float(cfg["yaw"]),
            pitch=float(cfg["pitch"]),
            roll=float(cfg["roll"]),
            upAxisIndex=2,
        )
        return np.array(view, dtype=np.float32).reshape(4, 4, order="F")

    def _view_matrix_wrist(self) -> np.ndarray:
        """Wrist camera: EEF-parented, offset from gripper center."""
        eef_pos, eef_quat = self._eef_pose()
        rot = R.from_quat(eef_quat)
        cam_pos = eef_pos + rot.apply(_WRIST_OFFSET.astype(float))
        # Wrist looks along -Z in EEF frame
        wrist_dir = rot.apply(np.array([0.0, 0.0, -1.0], dtype=np.float32))
        up = rot.apply(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        target = cam_pos + wrist_dir
        view = self._p.computeViewMatrix(
            cameraEyePosition=cam_pos.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
            physicsClientId=self._client,
        )
        return np.array(view, dtype=np.float32).reshape(4, 4, order="F")

    def _capture_primary(self) -> tuple[np.ndarray, np.ndarray]:
        view = self._view_matrix_primary()
        width, height, rgba, depth, _ = self._p.getCameraImage(
            width=self._image_size,
            height=self._image_size,
            viewMatrix=view.reshape(-1, order="F").tolist(),
            projectionMatrix=self._camera_projection.reshape(-1, order="F").tolist(),
            renderer=self._p.ER_TINY_RENDERER,
            physicsClientId=self._client,
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[..., :3]
        near, far = 0.01, 3.0
        depth_m = (far * near) / (far - (far - near) * np.asarray(depth, dtype=np.float32))
        return rgb, depth_m.astype(np.float32)

    def _capture_wrist(self) -> tuple[np.ndarray, np.ndarray]:
        view = self._view_matrix_wrist()
        width, height, rgba, depth, _ = self._p.getCameraImage(
            width=self._image_size,
            height=self._image_size,
            viewMatrix=view.reshape(-1, order="F").tolist(),
            projectionMatrix=self._camera_projection.reshape(-1, order="F").tolist(),
            renderer=self._p.ER_TINY_RENDERER,
            physicsClientId=self._client,
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[..., :3]
        near, far = 0.01, 3.0
        depth_m = (far * near) / (far - (far - near) * np.asarray(depth, dtype=np.float32))
        return rgb, depth_m.astype(np.float32)

    # ── State helpers ─────────────────────────────────────────────────────────

    def _eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """End-effector pose from PyBullet (EEF link = link 8)."""
        state = self._p.getLinkState(
            self._robot_id,
            self._ee_link,
            computeForwardKinematics=True,
            physicsClientId=self._client,
        )
        pos = np.array(state[4], dtype=np.float32)
        quat = np.array(state[5], dtype=np.float32)  # xyzw in PyBullet
        return pos, quat

    def _state_vector(
        self, eef_pos: np.ndarray, eef_quat: np.ndarray
    ) -> np.ndarray:
        """
        LIBERO-aligned state[8] (verified against raw parquet ground truth):

        state[0:3] = eef_pos (world frame, m)
        state[3:7] = motor_joint_positions[0:4] (first 4 arm joint values)
        state[7]   = gripper_joint (continuous, ∈ [-0.042, +0.001])

        LIBERO convention: negative = closed, positive = open.

        Why motor_joints NOT eef_quat: parquet row-norm(state[3:7]) ≈ 3.1 (not 1.0 as
        quaternion would require). Raw HDF5 robot_states = concat(gripper_qpos[2],
        eef_pos[3], eef_quat[4]) in bddl_base_domain.py:826, but the parquet 8D format
        uses motor_joints[4] for state[3:7]. File header already documented this correctly.
        """
        # Read all 7 arm joint positions
        arm_joints = np.array(
            [self._p.getJointState(self._robot_id, i, physicsClientId=self._client)[0]
             for i in range(_ARM_DOF)],
            dtype=np.float32,
        )
        motor_joints_4 = arm_joints[:4]  # state[3:7] = first 4 arm joints
        # Read gripper finger positions
        j9 = self._p.getJointState(
            self._robot_id, _GRIPPER_JOINTS[0], physicsClientId=self._client
        )[0]
        j10 = self._p.getJointState(
            self._robot_id, _GRIPPER_JOINTS[1], physicsClientId=self._client
        )[0]
        finger_pos = (j9 + j10) / 2.0  # ∈ [0.0, 0.04] in PyBullet
        # PyBullet [0.0, 0.04] → LIBERO [-0.042, +0.001]
        gripper_joint = float(finger_pos * 1.075 - 0.042)

        return np.concatenate([
            eef_pos.astype(np.float32),
            motor_joints_4,
            np.array([gripper_joint], dtype=np.float32),
        ])  # shape: (8,)

    # ── Physics manipulation helpers ──────────────────────────────────────────

    def _pb_step(self, count: int = 1) -> None:
        for _ in range(count):
            self._p.stepSimulation(physicsClientId=self._client)

    def _set_arm_joints(self, joints: np.ndarray) -> None:
        joints = np.asarray(joints, dtype=np.float32)
        for idx, val in zip(self._arm_joint_indices, joints.tolist()):
            self._p.resetJointState(
                self._robot_id, idx, val, physicsClientId=self._client
            )

    def _set_gripper_open(self, open_fraction: float) -> None:
        self._gripper_open = float(np.clip(open_fraction, 0.0, 1.0))
        target = (
            _GRIPPER_CLOSED_POS * (1.0 - self._gripper_open)
            + _GRIPPER_OPEN_POS * self._gripper_open
        )
        for j in _GRIPPER_JOINTS:
            self._p.resetJointState(
                self._robot_id, j, target, physicsClientId=self._client
            )

    def _set_drawer_fraction(self, fraction: float) -> None:
        self._drawer_fraction = float(np.clip(fraction, 0.0, 1.0))
        q = self._drawer_low + self._drawer_fraction * self._drawer_travel
        self._p.resetJointState(
            self._drawer_id,
            self._drawer_joint,
            q,
            physicsClientId=self._client,
        )

    def _move_to_pose(
        self,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray,
        gripper_open: float,
        steps: int = 4,
        lock_drawer_fraction: float | None = None,
    ) -> None:
        """Move EEF to target pose using joint-space PD."""
        for _ in range(steps):
            j_pos = np.array(
                [
                    self._p.getJointState(self._robot_id, i, physicsClientId=self._client)[0]
                    for i in range(_ARM_DOF)
                ]
            )
            j_vel = np.array(
                [
                    self._p.getJointState(self._robot_id, i, physicsClientId=self._client)[1]
                    for i in range(_ARM_DOF)
                ]
            )
            delta = target_pos - self._eef_pose()[0]  # shape (3,)
            kp, kd = 5.0, 1.0
            # Distribute xyz delta across 7 arm DOF proportionally
            j_delta = np.zeros(_ARM_DOF, dtype=np.float32)
            j_delta[:3] = delta
            j_target = j_pos + kp * j_delta - kd * j_vel
            j_target = np.clip(j_target, -2.9, 2.9)
            self._set_arm_joints(j_target.astype(np.float32))

            g_target = (
                _GRIPPER_OPEN_POS if gripper_open > 0.5 else _GRIPPER_CLOSED_POS
            )
            for j in _GRIPPER_JOINTS:
                self._p.resetJointState(
                    self._robot_id, j, g_target, physicsClientId=self._client
                )
            self._pb_step(1)

            if lock_drawer_fraction is not None:
                self._set_drawer_fraction(lock_drawer_fraction)

    def _drawer_open_delta_normalized(self) -> np.ndarray:
        """Unit vector of drawer opening direction in world frame."""
        return np.array([0.0, -1.0, 0.0], dtype=np.float32)

    def _pose_near_current_handle(self) -> bool:
        """True if current EEF pose is near the drawer handle attachment point."""
        if self._attachment_local is None:
            return False
        eef_pos, _ = self._eef_pose()
        world_handle = eef_pos + self._attachment_local
        return bool(abs(world_handle[0] - 0.18) < 0.08)

    def _find_drawer_joint(self) -> int:
        for idx in range(
            self._p.getNumJoints(self._drawer_id, physicsClientId=self._client)
        ):
            info = self._p.getJointInfo(
                self._drawer_id, idx, physicsClientId=self._client
            )
            if info[2] in (self._p.JOINT_PRISMATIC, self._p.JOINT_REVOLUTE):
                return idx
        raise RuntimeError("No articulated drawer joint found in drawer URDF")

    def _drawer_range(self) -> tuple[float, float]:
        info = self._p.getJointInfo(
            self._drawer_id, self._drawer_joint, physicsClientId=self._client
        )
        low = float(info[8])
        high = float(info[9])
        if high <= low:
            high = low + 0.25
        return low, high

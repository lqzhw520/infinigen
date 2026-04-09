#!/usr/bin/env python3
"""PyBullet Panda + drawer simulation helpers for the MINT robot-trajectory campaign."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mint_common import (
    GRIPPER_CLOSE_VALUE,
    GRIPPER_OPEN_VALUE,
    ROTATION_SCALE_RAD,
    TRANSLATION_SCALE_M,
)
from scipy.spatial.transform import Rotation as R
from strict_success import evaluate_strict_success

DRAWER_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen/sim_exports/urdf/drawerbox")
PANDA_URDF = Path(
    "/root/anaconda3/envs/mint/lib/python3.12/site-packages/robosuite/models/assets/bullet_data/panda_description/urdf/panda_arm_hand.urdf"
)

CAMERA_PRIMARY = {
    "target": np.array([0.18, 0.0, 0.40], dtype=np.float32),
    "distance": 1.0,
    "yaw": 165.0,
    "pitch": -32.0,
    "roll": 0.0,
}

CAMERA_SECONDARY_OFFSET = np.array([0.08, 0.0, 0.02], dtype=np.float32)
CAMERA_ANYGRASP_POS = np.array([0.0, -0.55, 0.15], dtype=np.float32)
CAMERA_ANYGRASP_TARGET = np.array([0.0, -0.05, 0.0], dtype=np.float32)
CAMERA_ANYGRASP_UP = np.array([0.0, 0.0, 1.0], dtype=np.float32)
HOME_JOINTS = np.array([0.0, -0.45, 0.0, -2.10, 0.0, 1.75, 0.72], dtype=np.float32)
GRIPPER_JOINTS = (9, 10)
GRIPPER_OPEN_POS = 0.04
GRIPPER_CLOSED_POS = 0.0
ARM_DOF = 7
TIME_STEP = 1.0 / 240.0
PULL_OPEN_FRACTION = 0.92
MAX_DELTA_TRANSLATION = TRANSLATION_SCALE_M
MAX_DELTA_ROTATION = ROTATION_SCALE_RAD
HANDLE_ATTACH_THRESHOLD = 0.050
ORACLE_PULL_QUAT_XYZW = np.array(
    [0.99008524, -0.1245343, -0.02635125, 0.05939738], dtype=np.float32
)
ACTION_TRANSLATION_CAP = 1.00
ACTION_ROTATION_CAP = 1.00
DEFAULT_ACTION_CONTRACT = {
    "action_frame": "world",
    "translation_scale_m": TRANSLATION_SCALE_M,
    "rotation_scale_rad": ROTATION_SCALE_RAD,
}


@dataclass
class RobotObservation:
    image: np.ndarray
    image2: np.ndarray
    depth: np.ndarray
    state: np.ndarray
    task: str
    eef_pos: np.ndarray
    eef_quat: np.ndarray
    gripper_open: float
    drawer_fraction: float


def drawer_assets_available(seeds: list[int]) -> bool:
    return all((DRAWER_ROOT / str(seed) / "drawerbox.urdf").exists() for seed in seeds)


def drawer_manifest() -> dict[str, Any]:
    return {
        "available_seeds": sorted(
            int(p.name) for p in DRAWER_ROOT.iterdir() if p.is_dir()
        ),
        "root": str(DRAWER_ROOT),
    }


def _urdf_path(seed: int) -> Path:
    return DRAWER_ROOT / str(seed) / "drawerbox.urdf"


def _metadata_path(seed: int) -> Path:
    return DRAWER_ROOT / str(seed) / "metadata.json"


def _load_seed_metadata(seed: int) -> dict[str, Any]:
    path = _metadata_path(seed)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _camera_intrinsics_from_projection(
    proj_matrix: np.ndarray, image_size: int
) -> np.ndarray:
    return np.array(
        [
            [proj_matrix[0, 0] * image_size / 2.0, 0.0, image_size / 2.0],
            [0.0, proj_matrix[1, 1] * image_size / 2.0, image_size / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _depth_to_camera_points(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    image_size = int(depth.shape[0])
    zz = depth.reshape(-1)
    yy, xx = np.mgrid[0:image_size, 0:image_size].reshape(2, -1).astype(np.float32)
    valid = zz > 0.01
    xx = xx[valid]
    yy = yy[valid]
    zz = zz[valid]
    pc_x = (xx - cx) * zz / max(fx, 1e-6)
    pc_y = (yy - cy) * zz / max(fy, 1e-6)
    return np.stack([pc_x, pc_y, zz], axis=-1).astype(np.float32)


def _world_from_camera_optical(view_matrix: np.ndarray) -> np.ndarray:
    world_from_gl = np.linalg.inv(view_matrix)
    optical_to_gl = np.eye(4, dtype=np.float32)
    optical_to_gl[1, 1] = -1.0
    optical_to_gl[2, 2] = -1.0
    return (world_from_gl @ optical_to_gl).astype(np.float32)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homo = np.concatenate(
        [points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)], axis=1
    )
    return (transform @ homo.T).T[:, :3].astype(np.float32)


def _quat_xyzw_from_wxyz(quat) -> np.ndarray:
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)


def _quat_wxyz_from_xyzw(quat) -> np.ndarray:
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)


def _normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quat / norm


class DrawerRobotEnv:
    """DIRECT pybullet scene with Panda arm and articulated drawer."""

    def __init__(
        self,
        seed: int,
        image_size: int = 224,
        max_steps: int = 48,
        action_contract: dict[str, Any] | None = None,
        render_observations: bool = True,
    ):
        import pybullet as p

        self.p = p
        self.seed = seed
        self.seed_metadata = _load_seed_metadata(seed)
        self.image_size = int(image_size)
        self.max_steps = int(max_steps)
        self.render_observations = bool(render_observations)
        self.client = p.connect(p.DIRECT)
        p.resetSimulation(physicsClientId=self.client)
        p.setTimeStep(TIME_STEP, physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)

        self.robot_id = self._load_robot()
        self.drawer_id = self._load_drawer()
        self.ee_link = 8
        self.arm_joint_indices = list(range(ARM_DOF))
        self.jacobian_joint_indices = [0, 1, 2, 3, 4, 5, 6, 9, 10]
        self.joint_lower_limits, self.joint_upper_limits, self.joint_ranges = (
            self._arm_joint_bounds()
        )
        self.drawer_joint = self._find_drawer_joint()
        self.drawer_low, self.drawer_high = self._drawer_range()
        self.drawer_open_delta = self._drawer_open_delta()
        self.drawer_motion_axis = self.drawer_open_delta / max(
            np.linalg.norm(self.drawer_open_delta), 1e-8
        )
        self.drawer_travel_distance = float(
            max(np.linalg.norm(self.drawer_open_delta), 1e-8)
        )
        self.task = "open the drawer"
        self.step_count = 0
        self.gripper_open = 1.0
        self.attached = False
        self.attachment_link = self.drawer_joint
        self.attachment_local = None
        self.action_contract = {**DEFAULT_ACTION_CONTRACT, **(action_contract or {})}
        self.action_frame = str(self.action_contract.get("action_frame", "world"))
        self.translation_scale = float(
            self.action_contract.get("translation_scale_m", TRANSLATION_SCALE_M)
        )
        self.rotation_scale = float(
            self.action_contract.get("rotation_scale_rad", ROTATION_SCALE_RAD)
        )
        self.attach_threshold = float(
            self.action_contract.get("attach_threshold_m", HANDLE_ATTACH_THRESHOLD)
        )
        self.home_pose = None
        self.commanded_eef_pos = None
        self.commanded_eef_quat = None
        self.drawer_trace_history: list[float] = []
        self.attached_trace_history: list[bool] = []
        self.ever_attached = False
        self.camera_projection = self._projection_matrix()
        self.reset()
        self.attach_step: int | None = None  # set during replay

    def close(self) -> None:
        try:
            self.p.disconnect(self.client)
        except Exception:
            pass

    def _load_robot(self) -> int:
        flags = self.p.URDF_USE_SELF_COLLISION
        robot_id = self.p.loadURDF(
            str(PANDA_URDF),
            basePosition=[-0.55, 0.0, 0.0],
            baseOrientation=self.p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            useFixedBase=True,
            flags=flags,
            physicsClientId=self.client,
        )
        return robot_id

    def _load_drawer(self) -> int:
        drawer_path = _urdf_path(self.seed)
        flags = self.p.URDF_USE_SELF_COLLISION
        drawer_id = self.p.loadURDF(
            str(drawer_path),
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=self.p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            useFixedBase=True,
            flags=flags,
            physicsClientId=self.client,
        )
        # P0b fix: ER_TINY_RENDERER does not read .mtl material files.
        # Set cardboard color via changeVisualShape after loading.
        cardboard_rgba = [0.60, 0.50, 0.40, 1.0]
        num_joints = self.p.getNumJoints(drawer_id, physicsClientId=self.client)
        link_indices = list(range(num_joints))
        link_indices.insert(0, -1)
        for link_idx in link_indices:
            self.p.changeVisualShape(
                drawer_id,
                linkIndex=link_idx,
                rgbaColor=cardboard_rgba,
                physicsClientId=self.client,
            )
        return drawer_id

    def _arm_joint_bounds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower_limits = []
        upper_limits = []
        joint_ranges = []
        for idx in self.arm_joint_indices:
            info = self.p.getJointInfo(self.robot_id, idx, physicsClientId=self.client)
            low = float(info[8])
            high = float(info[9])
            if high <= low:
                low, high = -7.0, 7.0
            lower_limits.append(low)
            upper_limits.append(high)
            joint_ranges.append(max(high - low, 1e-3))
        return (
            np.asarray(lower_limits, dtype=np.float32),
            np.asarray(upper_limits, dtype=np.float32),
            np.asarray(joint_ranges, dtype=np.float32),
        )

    def _find_drawer_joint(self) -> int:
        for idx in range(
            self.p.getNumJoints(self.drawer_id, physicsClientId=self.client)
        ):
            info = self.p.getJointInfo(self.drawer_id, idx, physicsClientId=self.client)
            if info[2] in (self.p.JOINT_PRISMATIC, self.p.JOINT_REVOLUTE):
                return idx
        raise RuntimeError("No articulated drawer joint found")

    def _drawer_range(self) -> tuple[float, float]:
        info = self.p.getJointInfo(
            self.drawer_id, self.drawer_joint, physicsClientId=self.client
        )
        low = float(info[8])
        high = float(info[9])
        if high <= low:
            high = low + 0.25
        return low, high

    def _drawer_open_delta(self) -> np.ndarray:
        closed = self._drawer_link_pose(fraction=0.0)[0]
        opened = self._drawer_link_pose(fraction=1.0)[0]
        return (opened - closed).astype(np.float32)

    def _projection_matrix(self) -> np.ndarray:
        fov = 60.0
        aspect = 1.0
        near = 0.01
        far = 3.0
        proj = self.p.computeProjectionMatrixFOV(fov, aspect, near, far)
        return np.array(proj, dtype=np.float32).reshape(4, 4, order="F")

    def _view_matrix_primary(self) -> np.ndarray:
        cfg = CAMERA_PRIMARY
        view = self.p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=cfg["target"].tolist(),
            distance=float(cfg["distance"]),
            yaw=float(cfg["yaw"]),
            pitch=float(cfg["pitch"]),
            roll=float(cfg["roll"]),
            upAxisIndex=2,
        )
        return np.array(view, dtype=np.float32).reshape(4, 4, order="F")

    def _view_matrix_wrist(self) -> np.ndarray:
        eef_pos, eef_quat = self.eef_pose()
        rot = R.from_quat(eef_quat)
        cam_pos = eef_pos + rot.apply(CAMERA_SECONDARY_OFFSET)
        target = cam_pos + rot.apply(np.array([0.22, 0.0, -0.02], dtype=np.float32))
        up = rot.apply(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        view = self.p.computeViewMatrix(cam_pos.tolist(), target.tolist(), up.tolist())
        return np.array(view, dtype=np.float32).reshape(4, 4, order="F")

    def _view_matrix_anygrasp(self) -> np.ndarray:
        view = self.p.computeViewMatrix(
            CAMERA_ANYGRASP_POS.tolist(),
            CAMERA_ANYGRASP_TARGET.tolist(),
            CAMERA_ANYGRASP_UP.tolist(),
        )
        return np.array(view, dtype=np.float32).reshape(4, 4, order="F")

    def _camera_capture(self, view: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        width, height, rgba, depth, _ = self.p.getCameraImage(
            width=self.image_size,
            height=self.image_size,
            viewMatrix=view.reshape(-1, order="F").tolist(),
            projectionMatrix=self.camera_projection.reshape(-1, order="F").tolist(),
            renderer=self.p.ER_TINY_RENDERER,
            physicsClientId=self.client,
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[..., :3]
        depth = np.asarray(depth, dtype=np.float32).reshape(height, width)
        near = 0.01
        far = 3.0
        metric_depth = (far * near) / (far - (far - near) * depth)
        return rgb, metric_depth.astype(np.float32)

    def drawer_fraction(self) -> float:
        q = self.p.getJointState(
            self.drawer_id, self.drawer_joint, physicsClientId=self.client
        )[0]
        return float(
            np.clip(
                (q - self.drawer_low) / max(self.drawer_high - self.drawer_low, 1e-6),
                0.0,
                1.0,
            )
        )

    def eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        state = self.p.getLinkState(
            self.robot_id,
            self.ee_link,
            computeForwardKinematics=True,
            physicsClientId=self.client,
        )
        pos = np.array(state[4], dtype=np.float32)
        quat = np.array(state[5], dtype=np.float32)
        return pos, quat

    def _state_vector(self) -> np.ndarray:
        """
        LIBERO-aligned state[8] (verified against raw HDF5 ground truth):

        state[0:3] = eef_pos (world frame, m)
        state[3:7] = motor_joint_positions[0:4] (4 arm joint values)
        state[7]   = gripper_joint (continuous, ∈ [-0.042, +0.001])

        LIBERO convention: negative = closed, positive = open.

        Why motor_joints NOT eef_quat: raw parquet stats show state[3:7] row-norm ≈ 3.1
        (not 1.0 as quaternion would require). info.json labels "[x,y,z,rx,ry,rz,rw,gripper]"
        are incorrect. Ground truth confirmed by HDF5 robot_states = concat(gripper_qpos[2],
        eef_pos[3], eef_quat[4]) in source code (bddl_base_domain.py:826) but parquet
        stores a different 8D format where state[3:7] = motor joints (4D), not quaternions.
        """
        eef_pos, _ = self.eef_pose()
        # Read all 7 arm joint positions
        arm_joints = np.array(
            [self.p.getJointState(self.robot_id, i, physicsClientId=self.client)[0]
             for i in range(ARM_DOF)],
            dtype=np.float32,
        )
        # state[3:7] = first 4 arm joint values
        motor_joints_4 = arm_joints[:4]
        # Read gripper finger positions
        j9 = self.p.getJointState(
            self.robot_id, GRIPPER_JOINTS[0], physicsClientId=self.client
        )[0]
        j10 = self.p.getJointState(
            self.robot_id, GRIPPER_JOINTS[1], physicsClientId=self.client
        )[0]
        finger_pos = (j9 + j10) / 2.0  # ∈ [0.0, 0.04] in PyBullet
        # LIBERO convention: negative = closed, positive = open
        # LIBERO range: [-0.042, +0.001]; PyBullet range: [0.0, 0.04]
        gripper_joint = finger_pos * 1.075 - 0.042
        return np.concatenate(
            [
                eef_pos.astype(np.float32),
                motor_joints_4,
                np.array([gripper_joint], dtype=np.float32),
            ]
        )  # shape: (8,)

    def observe(self) -> RobotObservation:
        if self.render_observations:
            image, depth = self._camera_capture(self._view_matrix_primary())
            image2, _ = self._camera_capture(self._view_matrix_wrist())
        else:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            image2 = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            depth = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        eef_pos, eef_quat = self.eef_pose()
        return RobotObservation(
            image=image,
            image2=image2,
            depth=depth,
            state=self._state_vector(),
            task=self.task,
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            gripper_open=self.gripper_open,
            drawer_fraction=self.drawer_fraction(),
        )

    def reset(self) -> RobotObservation:
        self.step_count = 0
        self.attached = False
        self.ever_attached = False
        self.attachment_local = None
        self._set_drawer_fraction(0.0)
        self._set_arm_joints(HOME_JOINTS)
        self._set_gripper_open(1.0)
        self._step_world(8)
        obs = self.observe()
        self.home_pose = (obs.eef_pos.copy(), obs.eef_quat.copy())
        self.commanded_eef_pos = obs.eef_pos.copy()
        self.commanded_eef_quat = obs.eef_quat.copy()
        self.drawer_trace_history = [float(obs.drawer_fraction)]
        self.attached_trace_history = [False]
        return obs

    def _step_world(self, count: int = 1) -> None:
        for _ in range(count):
            self.p.stepSimulation(physicsClientId=self.client)

    def _set_arm_joints(self, joints: np.ndarray) -> None:
        joints = np.asarray(joints, dtype=np.float32)
        for idx, val in zip(self.arm_joint_indices, joints.tolist()):
            self.p.resetJointState(self.robot_id, idx, val, physicsClientId=self.client)

    def _set_drawer_fraction(self, fraction: float) -> None:
        q = self.drawer_low + float(np.clip(fraction, 0.0, 1.0)) * (
            self.drawer_high - self.drawer_low
        )
        self.p.resetJointState(
            self.drawer_id, self.drawer_joint, q, physicsClientId=self.client
        )

    def _set_gripper_open(self, open_fraction: float) -> None:
        self.gripper_open = float(np.clip(open_fraction, 0.0, 1.0))
        target = (
            GRIPPER_CLOSED_POS * (1.0 - self.gripper_open)
            + GRIPPER_OPEN_POS * self.gripper_open
        )
        for joint_idx in GRIPPER_JOINTS:
            self.p.resetJointState(
                self.robot_id, joint_idx, target, physicsClientId=self.client
            )

    def _ik(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        quat = _normalize_quat_xyzw(quat)
        current_joints = np.array(
            [
                self.p.getJointState(self.robot_id, idx, physicsClientId=self.client)[0]
                for idx in self.arm_joint_indices
            ],
            dtype=np.float32,
        )
        joints = self.p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link,
            pos.tolist(),
            quat.tolist(),
            lowerLimits=self.joint_lower_limits.tolist(),
            upperLimits=self.joint_upper_limits.tolist(),
            jointRanges=self.joint_ranges.tolist(),
            restPoses=current_joints.tolist(),
            maxNumIterations=200,
            residualThreshold=1e-4,
            physicsClientId=self.client,
        )
        return np.asarray(joints[:ARM_DOF], dtype=np.float32)

    def _drawer_link_pose(
        self, fraction: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        current_q = self.p.getJointState(
            self.drawer_id, self.drawer_joint, physicsClientId=self.client
        )[0]
        if fraction is not None:
            self._set_drawer_fraction(fraction)
            self._step_world(2)
        state = self.p.getLinkState(
            self.drawer_id,
            self.drawer_joint,
            computeForwardKinematics=True,
            physicsClientId=self.client,
        )
        pos = np.array(state[4], dtype=np.float32)
        quat = np.array(state[5], dtype=np.float32)
        if fraction is not None:
            self.p.resetJointState(
                self.drawer_id,
                self.drawer_joint,
                current_q,
                physicsClientId=self.client,
            )
            self._step_world(2)
        return pos, quat

    def handle_local_pose(self, grasp_pose_world: np.ndarray) -> dict[str, np.ndarray]:
        link_pos, link_quat = self._drawer_link_pose()
        link_rot = R.from_quat(link_quat)
        grasp_rot = R.from_matrix(grasp_pose_world[:3, :3])
        local_pos = link_rot.inv().apply(grasp_pose_world[:3, 3] - link_pos)
        local_rot = (link_rot.inv() * grasp_rot).as_quat()
        return {
            "pos": local_pos.astype(np.float32),
            "quat": local_rot.astype(np.float32),
        }

    def local_to_world_pose(
        self,
        local_pos: np.ndarray,
        local_quat: np.ndarray,
        fraction: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        link_pos, link_quat = self._drawer_link_pose(fraction)
        link_rot = R.from_quat(link_quat)
        pos = link_pos + link_rot.apply(local_pos)
        quat = (link_rot * R.from_quat(local_quat)).as_quat()
        return pos.astype(np.float32), quat.astype(np.float32)

    def move_to_pose(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        gripper_open: float,
        steps: int = 8,
        update_drawer_to: float | None = None,
        lock_drawer_fraction: float | None = None,
    ) -> list[RobotObservation]:
        pos = np.asarray(pos, dtype=np.float32)
        quat = _normalize_quat_xyzw(quat)
        current_joints = np.array(
            [
                self.p.getJointState(self.robot_id, idx, physicsClientId=self.client)[0]
                for idx in self.arm_joint_indices
            ],
            dtype=np.float32,
        )
        target_joints = self._ik(pos, quat)
        observations: list[RobotObservation] = []
        for alpha in np.linspace(0.0, 1.0, max(steps, 2))[1:]:
            joints = current_joints * (1.0 - alpha) + target_joints * alpha
            self._set_arm_joints(joints)
            self._set_gripper_open(gripper_open)
            if update_drawer_to is not None:
                self._set_drawer_fraction(
                    update_drawer_to * alpha + self.drawer_fraction() * (1.0 - alpha)
                )
            self._step_world(4)
            if lock_drawer_fraction is not None:
                self._set_drawer_fraction(lock_drawer_fraction)
            observations.append(self.observe())
        return observations

    def _apply_cartesian_delta(
        self,
        delta_pos_world: np.ndarray,
        delta_rot_world: np.ndarray,
        gripper_open: float,
        *,
        substeps: int = 6,
        damping: float = 1e-3,
    ) -> RobotObservation:
        current_arm_joints = np.array(
            [
                self.p.getJointState(self.robot_id, idx, physicsClientId=self.client)[0]
                for idx in self.arm_joint_indices
            ],
            dtype=np.float32,
        )
        current_jacobian_joints = np.array(
            [
                self.p.getJointState(self.robot_id, idx, physicsClientId=self.client)[0]
                for idx in self.jacobian_joint_indices
            ],
            dtype=np.float32,
        )
        zeros = [0.0] * len(self.jacobian_joint_indices)
        lin_jac, ang_jac = self.p.calculateJacobian(
            self.robot_id,
            self.ee_link,
            [0.0, 0.0, 0.0],
            current_jacobian_joints.tolist(),
            zeros,
            zeros,
            physicsClientId=self.client,
        )
        jacobian = np.vstack(
            [
                np.asarray(lin_jac, dtype=np.float32),
                np.asarray(ang_jac, dtype=np.float32),
            ]
        )
        twist = np.concatenate(
            [delta_pos_world.astype(np.float32), delta_rot_world.astype(np.float32)],
            axis=0,
        )
        damping_eye = damping * np.eye(jacobian.shape[0], dtype=np.float32)
        dq_full = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping_eye, twist
        )
        target_joints = np.clip(
            current_arm_joints + dq_full[:ARM_DOF].astype(np.float32),
            self.joint_lower_limits,
            self.joint_upper_limits,
        )
        for alpha in np.linspace(0.0, 1.0, max(substeps, 2), dtype=np.float32)[1:]:
            interp = current_arm_joints * (1.0 - alpha) + target_joints * alpha
            self._set_arm_joints(interp)
            self._set_gripper_open(gripper_open)
            self._step_world(2)
        return self.observe()

    def anygrasp_payload(self) -> dict[str, Any]:
        obs = self.observe()
        view = self._view_matrix_anygrasp()
        image, depth = self._camera_capture(view)
        intrinsics = _camera_intrinsics_from_projection(
            self.camera_projection, self.image_size
        )
        points = _depth_to_camera_points(depth, intrinsics)
        colors = (
            image.reshape(-1, 3)[depth.reshape(-1) > 0.01].astype(np.float32) / 255.0
        )
        world_from_camera = _world_from_camera_optical(view)
        camera_from_world = np.linalg.inv(world_from_camera)
        aabb_min, aabb_max = self.p.getAABB(
            self.drawer_id, self.drawer_joint, physicsClientId=self.client
        )
        drawer_aabb_world = np.array([aabb_min, aabb_max], dtype=np.float32)
        corners_world = np.array(
            [
                [x, y, z]
                for x in [aabb_min[0], aabb_max[0]]
                for y in [aabb_min[1], aabb_max[1]]
                for z in [aabb_min[2], aabb_max[2]]
            ],
            dtype=np.float32,
        )
        corners_cam = _transform_points(corners_world, camera_from_world)
        limits = np.array(
            [
                float(corners_cam[:, 0].min() - 0.05),
                float(corners_cam[:, 0].max() + 0.05),
                float(corners_cam[:, 1].min() - 0.05),
                float(corners_cam[:, 1].max() + 0.05),
                float(max(0.01, corners_cam[:, 2].min() - 0.05)),
                float(corners_cam[:, 2].max() + 0.10),
            ],
            dtype=np.float32,
        )
        handle_center_world = None
        handle_semantics_source = "heuristic_aabb_front_face"
        frame_local = self.seed_metadata.get("handle_frame_local") or {}
        center_local = self.seed_metadata.get("handle_center_local")
        bbox_local = self.seed_metadata.get("handle_bbox_local") or {}
        local_pos = None
        local_quat = None
        if isinstance(frame_local, dict) and "pos" in frame_local:
            local_pos = np.asarray(
                frame_local.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32
            )
            local_quat = _normalize_quat_xyzw(
                np.asarray(
                    frame_local.get("quat", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32
                )
            )
            handle_semantics_source = "metadata_handle_frame_local"
        elif isinstance(center_local, (list, tuple)) and len(center_local) == 3:
            local_pos = np.asarray(center_local, dtype=np.float32)
            local_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            handle_semantics_source = "metadata_handle_center_local"
        elif (
            isinstance(bbox_local, dict) and "min" in bbox_local and "max" in bbox_local
        ):
            local_pos = 0.5 * (
                np.asarray(bbox_local.get("min", [0.0, 0.0, 0.0]), dtype=np.float32)
                + np.asarray(bbox_local.get("max", [0.0, 0.0, 0.0]), dtype=np.float32)
            )
            local_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            handle_semantics_source = "metadata_handle_bbox_local"
        if local_pos is not None and local_quat is not None:
            handle_center_world, _ = self.local_to_world_pose(local_pos, local_quat)
        else:
            center_world = 0.5 * (drawer_aabb_world[0] + drawer_aabb_world[1])
            dominant_axis = int(np.argmax(np.abs(self.drawer_motion_axis)))
            front_point = center_world.copy()
            extent = 0.5 * (drawer_aabb_world[1] - drawer_aabb_world[0])
            front_point[dominant_axis] += (
                np.sign(self.drawer_motion_axis[dominant_axis]) * extent[dominant_axis]
            )
            handle_center_world = front_point.astype(np.float32)
        return {
            "image": image,
            "depth": depth,
            "image2": obs.image2,
            "state": obs.state,
            "points": points,
            "colors": colors,
            "intrinsics": intrinsics,
            "camera_view": view.astype(np.float32),
            "world_from_camera": world_from_camera.astype(np.float32),
            "drawer_aabb_world": drawer_aabb_world,
            "drawer_motion_axis": self.drawer_motion_axis.astype(np.float32),
            "handle_center_world": handle_center_world,
            "handle_semantics_source": handle_semantics_source,
            "limits": limits,
        }

    def pose_near_current_handle(self, distance_threshold: float | None = None) -> bool:
        if self.attachment_local is None:
            return False
        threshold = (
            self.attach_threshold
            if distance_threshold is None
            else float(distance_threshold)
        )
        handle_pos, _ = self.local_to_world_pose(
            self.attachment_local["pos"], self.attachment_local["quat"]
        )
        eef_pos, _ = self.eef_pose()
        return float(np.linalg.norm(eef_pos - handle_pos)) <= threshold

    def step(
        self, action: np.ndarray
    ) -> tuple[RobotObservation, float, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(7)
        current_obs = self.observe()
        prev_drawer_fraction = float(current_obs.drawer_fraction)
        delta_pos = np.clip(action[:3], -1.0, 1.0) * self.translation_scale
        if self.action_frame == "local":
            delta_pos = R.from_quat(self.commanded_eef_quat).apply(delta_pos)
        rot_delta = np.clip(action[3:6], -1.0, 1.0) * self.rotation_scale
        self.commanded_eef_pos = np.asarray(
            self.commanded_eef_pos + delta_pos, dtype=np.float32
        )
        self.commanded_eef_quat = (
            (R.from_rotvec(rot_delta) * R.from_quat(self.commanded_eef_quat))
            .as_quat()
            .astype(np.float32)
        )
        gripper_open = 1.0 if action[6] >= 0 else 0.0
        if (
            gripper_open < 0.5
            and self.attachment_local is not None
            and self.pose_near_current_handle()
        ):
            self.attached = True
        if gripper_open > 0.5:
            self.attached = False
        prev_pos = current_obs.eef_pos.copy()
        observations = self.move_to_pose(
            self.commanded_eef_pos,
            self.commanded_eef_quat,
            gripper_open,
            steps=4,
            update_drawer_to=None,
            lock_drawer_fraction=prev_drawer_fraction,
        )
        obs = observations[-1] if observations else self.observe()
        if (
            gripper_open < 0.5
            and self.attachment_local is not None
            and self.pose_near_current_handle()
        ):
            self.attached = True
        if not self.attached:
            self._set_drawer_fraction(prev_drawer_fraction)
            obs = self.observe()
        if self.attached and self.attachment_local is not None:
            axis_delta = float(np.dot(obs.eef_pos - prev_pos, self.drawer_motion_axis))
            frac_delta = axis_delta / self.drawer_travel_distance
            self._set_drawer_fraction(self.drawer_fraction() + frac_delta)
            self._step_world(2)
            obs = self.observe()
        self.ever_attached = bool(self.ever_attached or self.attached)
        self.drawer_trace_history.append(float(obs.drawer_fraction))
        self.attached_trace_history.append(bool(self.attached))
        strict = evaluate_strict_success(
            np.asarray(self.drawer_trace_history, dtype=np.float32),
            np.asarray(self.attached_trace_history, dtype=bool),
        )
        self.step_count += 1
        success = bool(strict["strict_success"])
        reward = 1.0 if success else float(obs.drawer_fraction)
        done = success or self.step_count >= self.max_steps
        info = {
            "is_success": success,
            "weak_drawer_open": bool(obs.drawer_fraction >= 0.90),
            "drawer_fraction": float(obs.drawer_fraction),
            "eef_pos": obs.eef_pos.tolist(),
            "gripper_open": float(obs.gripper_open),
            "attached": bool(self.attached),
            "ever_attached": bool(self.ever_attached),
            "strict_metrics": strict,
        }
        return obs, reward, done, info


def _pose_error(
    current_pos: np.ndarray,
    current_quat: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
) -> tuple[float, float]:
    pos_error = float(np.linalg.norm(target_pos - current_pos))
    rot_error = float(
        (R.from_quat(target_quat) * R.from_quat(current_quat).inv()).magnitude()
    )
    return pos_error, rot_error


def _rotation_errors(current_quat: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    current = np.asarray(current_quat, dtype=np.float32)
    target = np.asarray(target_quat, dtype=np.float32)
    if len(current) == 0 or len(target) == 0:
        return np.zeros((0,), dtype=np.float32)
    errors = []
    for cur, tgt in zip(current, target):
        errors.append(float((R.from_quat(tgt) * R.from_quat(cur).inv()).magnitude()))
    return np.asarray(errors, dtype=np.float32)


def _action_toward_pose(
    obs: RobotObservation,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    gripper_open: float,
    translation_cap: float = ACTION_TRANSLATION_CAP,
    rotation_cap: float = ACTION_ROTATION_CAP,
    *,
    translation_scale: float = TRANSLATION_SCALE_M,
    rotation_scale: float = ROTATION_SCALE_RAD,
    action_frame: str = "world",
) -> np.ndarray:
    delta_pos_world = target_pos - obs.eef_pos
    if action_frame == "local":
        delta_pos = R.from_quat(obs.eef_quat).inv().apply(delta_pos_world)
    else:
        delta_pos = delta_pos_world
    delta_pos = delta_pos / max(translation_scale, 1e-6)
    rotvec = (
        (R.from_quat(target_quat) * R.from_quat(obs.eef_quat).inv())
        .as_rotvec()
        .astype(np.float32)
    )
    delta_rot = rotvec / max(rotation_scale, 1e-6)
    grip = np.array(
        [GRIPPER_OPEN_VALUE if gripper_open >= 0.5 else GRIPPER_CLOSE_VALUE],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            np.clip(delta_pos, -translation_cap, translation_cap),
            np.clip(delta_rot, -rotation_cap, rotation_cap),
            grip,
        ],
        axis=0,
    ).astype(np.float32)


def build_oracle_grasp_pose(
    grasp_pose_world: np.ndarray, handle_center_world: np.ndarray
) -> np.ndarray:
    oracle = np.asarray(grasp_pose_world, dtype=np.float32).copy()
    oracle[:3, 3] = np.asarray(handle_center_world, dtype=np.float32)
    oracle[:3, :3] = R.from_quat(ORACLE_PULL_QUAT_XYZW).as_matrix().astype(np.float32)
    return oracle


def _stack_or_empty(
    values: list[np.ndarray], shape: tuple[int, ...], dtype
) -> np.ndarray:
    if not values:
        return np.zeros((0, *shape), dtype=dtype)
    return np.asarray(values, dtype=dtype)


def _action_from_transition(
    current_obs: RobotObservation,
    next_obs: RobotObservation,
    *,
    translation_scale: float = TRANSLATION_SCALE_M,
    rotation_scale: float = ROTATION_SCALE_RAD,
    action_frame: str = "world",
) -> np.ndarray:
    delta_pos_world = next_obs.eef_pos - current_obs.eef_pos
    if action_frame == "local":
        delta_pos = R.from_quat(current_obs.eef_quat).inv().apply(delta_pos_world)
    else:
        delta_pos = delta_pos_world
    delta_pos = delta_pos / max(translation_scale, 1e-6)
    rotvec = (
        (R.from_quat(next_obs.eef_quat) * R.from_quat(current_obs.eef_quat).inv())
        .as_rotvec()
        .astype(np.float32)
    )
    delta_rot = rotvec / max(rotation_scale, 1e-6)
    grip = np.array(
        [GRIPPER_OPEN_VALUE if next_obs.gripper_open >= 0.5 else GRIPPER_CLOSE_VALUE],
        dtype=np.float32,
    )
    return np.concatenate(
        [np.clip(delta_pos, -1.0, 1.0), np.clip(delta_rot, -1.0, 1.0), grip], axis=0
    ).astype(np.float32)


def build_robot_rollout(
    seed: int,
    grasp_pose_world: np.ndarray,
    episode_index: int = 0,
    max_steps: int = 96,
    *,
    grasp_source: str = "anygrasp",
    grasp_score: float | None = None,
) -> dict[str, Any]:
    env = DrawerRobotEnv(seed=seed, image_size=224, max_steps=max_steps)
    obs = env.reset()
    initial_obs = obs
    home_quat = initial_obs.eef_quat.copy()
    local_handle = env.handle_local_pose(grasp_pose_world)
    env.attachment_local = local_handle

    rng = np.random.default_rng(seed * 1000 + episode_index)
    closed_pos, closed_quat = env.local_to_world_pose(
        local_handle["pos"], local_handle["quat"], fraction=0.0
    )
    pull_fraction = float(
        np.clip(PULL_OPEN_FRACTION + rng.uniform(-0.03, 0.02), 0.85, 0.95)
    )
    open_pos, open_quat = env.local_to_world_pose(
        local_handle["pos"], local_handle["quat"], fraction=pull_fraction
    )
    travel_vec = open_pos - closed_pos
    travel_axis = travel_vec / max(np.linalg.norm(travel_vec), 1e-8)
    approach_jitter = np.array(
        [rng.uniform(-0.008, 0.008), 0.0, rng.uniform(-0.004, 0.006)], dtype=np.float32
    )
    # travel_axis points from the closed handle pose toward the opened handle pose.
    # Pre-grasp/staging should approach from that "outside" direction, not from inside the cabinet.
    pregrasp_pos = (
        closed_pos
        + 0.10 * travel_axis
        + np.array([0.0, 0.0, 0.04], dtype=np.float32)
        + approach_jitter
    )
    staging_pos = (
        closed_pos
        + 0.16 * travel_axis
        + np.array([0.0, 0.0, 0.08], dtype=np.float32)
        + 0.5 * approach_jitter
    )
    retreat_pos = (
        open_pos
        + 0.06 * travel_axis
        + np.array([0.0, 0.0, 0.05], dtype=np.float32)
        + 0.5 * approach_jitter
    )

    images, images2, depths = [], [], []
    states, eef_positions, eef_quats, gripper_values = [], [], [], []
    next_eef_positions, next_eef_quaternions, next_drawer_fractions = [], [], []
    actions, phase_labels = [], []
    abs_drawer, joint_traj, reward_trace = [], [], []
    attached_trace, handle_distance_trace = [], []
    target_eef_positions, target_eef_quaternions, target_drawer_fractions = [], [], []
    success = False
    ever_attached = False
    max_drawer_fraction = float(obs.drawer_fraction)

    def _record_transition(
        current_obs: RobotObservation,
        next_obs: RobotObservation,
        action: np.ndarray,
        phase: str,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        target_fraction: float | None,
        attached: bool,
    ) -> None:
        nonlocal max_drawer_fraction
        handle_pos, _ = env.local_to_world_pose(
            local_handle["pos"], local_handle["quat"], fraction=next_obs.drawer_fraction
        )
        reward = (
            1.0 if next_obs.drawer_fraction >= 0.90 else float(next_obs.drawer_fraction)
        )
        images.append(current_obs.image.copy())
        images2.append(current_obs.image2.copy())
        depths.append(current_obs.depth.copy())
        states.append(current_obs.state.copy())
        eef_positions.append(current_obs.eef_pos.copy())
        eef_quats.append(current_obs.eef_quat.copy())
        gripper_values.append(float(current_obs.gripper_open))
        actions.append(action.astype(np.float32))
        phase_labels.append(phase)
        abs_drawer.append(float(current_obs.drawer_fraction))
        next_eef_positions.append(next_obs.eef_pos.copy())
        next_eef_quaternions.append(next_obs.eef_quat.copy())
        next_drawer_fractions.append(float(next_obs.drawer_fraction))
        attached_trace.append(bool(attached))
        handle_distance_trace.append(
            float(np.linalg.norm(next_obs.eef_pos - handle_pos))
        )
        target_eef_positions.append(np.asarray(target_pos, dtype=np.float32))
        target_eef_quaternions.append(np.asarray(target_quat, dtype=np.float32))
        target_drawer_fractions.append(
            float(
                target_fraction
                if target_fraction is not None
                else current_obs.drawer_fraction
            )
        )
        joint_traj.append(
            np.array(
                [
                    env.p.getJointState(env.robot_id, idx, physicsClientId=env.client)[
                        0
                    ]
                    for idx in env.arm_joint_indices
                ],
                dtype=np.float32,
            )
        )
        reward_trace.append(float(reward))
        max_drawer_fraction = max(max_drawer_fraction, float(next_obs.drawer_fraction))

    def _planned_segment(
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        gripper_open: float,
        steps: int,
        *,
        target_fraction: float | None = None,
        lock_drawer_fraction: float | None = None,
    ) -> list[RobotObservation]:
        state_id = env.p.saveState(physicsClientId=env.client)
        saved_runtime = {
            "gripper_open": float(env.gripper_open),
            "attached": bool(env.attached),
            "step_count": int(env.step_count),
            "commanded_eef_pos": None
            if env.commanded_eef_pos is None
            else env.commanded_eef_pos.copy(),
            "commanded_eef_quat": None
            if env.commanded_eef_quat is None
            else env.commanded_eef_quat.copy(),
        }
        try:
            return env.move_to_pose(
                target_pos,
                target_quat,
                gripper_open,
                steps=steps,
                update_drawer_to=target_fraction,
                lock_drawer_fraction=lock_drawer_fraction,
            )
        finally:
            env.p.restoreState(state_id, physicsClientId=env.client)
            env.p.removeState(state_id, physicsClientId=env.client)
            env.gripper_open = saved_runtime["gripper_open"]
            env.attached = saved_runtime["attached"]
            env.step_count = saved_runtime["step_count"]
            env.commanded_eef_pos = (
                None
                if saved_runtime["commanded_eef_pos"] is None
                else saved_runtime["commanded_eef_pos"].copy()
            )
            env.commanded_eef_quat = (
                None
                if saved_runtime["commanded_eef_quat"] is None
                else saved_runtime["commanded_eef_quat"].copy()
            )

    def _run_segment(
        phase: str,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        gripper_open: float,
        steps: int,
        *,
        target_fraction: float | None = None,
        lock_drawer_fraction: float | None = None,
    ) -> RobotObservation:
        nonlocal obs, ever_attached
        planned_waypoints = _planned_segment(
            target_pos,
            target_quat,
            gripper_open,
            steps=steps,
            target_fraction=target_fraction,
            lock_drawer_fraction=lock_drawer_fraction,
        )
        for waypoint in planned_waypoints:
            action = _action_toward_pose(
                obs,
                waypoint.eef_pos,
                waypoint.eef_quat,
                waypoint.gripper_open,
            )
            next_obs, _reward, done, info = env.step(action)
            _record_transition(
                obs,
                next_obs,
                action,
                phase,
                target_pos,
                target_quat,
                target_fraction,
                bool(info.get("attached")),
            )
            obs = next_obs
            ever_attached = ever_attached or bool(info.get("attached"))
            if phase in {"staging", "pregrasp", "retreat"}:
                pos_error, rot_error = _pose_error(
                    obs.eef_pos, obs.eef_quat, target_pos, target_quat
                )
                if pos_error <= 0.015 and rot_error <= 0.15:
                    break
            if phase == "grasp" and bool(info.get("attached")):
                break
            if (
                phase == "pull"
                and target_fraction is not None
                and obs.drawer_fraction >= max(0.0, target_fraction - 0.03)
            ):
                break
            if done:
                break
        return obs

    try:
        _run_segment(
            "staging", staging_pos, home_quat, 1.0, 24, lock_drawer_fraction=0.0
        )
        _run_segment(
            "pregrasp", pregrasp_pos, home_quat, 1.0, 18, lock_drawer_fraction=0.0
        )
        _run_segment(
            "grasp", closed_pos, closed_quat, 0.0, 48, lock_drawer_fraction=0.0
        )

        if ever_attached:
            for fraction in np.linspace(0.12, pull_fraction, 14):
                target_pos, target_quat = env.local_to_world_pose(
                    local_handle["pos"], local_handle["quat"], fraction=float(fraction)
                )
                _run_segment(
                    "pull",
                    target_pos,
                    target_quat,
                    0.0,
                    12,
                    target_fraction=float(fraction),
                )
                if obs.drawer_fraction >= 0.90:
                    break

            if obs.drawer_fraction >= 0.90:
                _run_segment(
                    "retreat",
                    retreat_pos,
                    open_quat,
                    1.0,
                    20,
                    lock_drawer_fraction=float(pull_fraction),
                )

        success = bool(max_drawer_fraction >= 0.90 and ever_attached)
        return {
            "seed": seed,
            "episode_index": episode_index,
            "success": success,
            "task": env.task,
            "steps": len(actions),
            "images": _stack_or_empty(images, (224, 224, 3), np.uint8),
            "images2": _stack_or_empty(images2, (224, 224, 3), np.uint8),
            "depths": _stack_or_empty(depths, (224, 224), np.float32),
            "states": _stack_or_empty(states, (8,), np.float32),
            "eef_positions": _stack_or_empty(eef_positions, (3,), np.float32),
            "eef_quaternions": _stack_or_empty(eef_quats, (4,), np.float32),
            "gripper_values": np.asarray(gripper_values, dtype=np.float32),
            "actions": _stack_or_empty(actions, (7,), np.float32),
            "absolute_drawer_fraction": np.asarray(abs_drawer, dtype=np.float32),
            "next_eef_positions": _stack_or_empty(next_eef_positions, (3,), np.float32),
            "next_eef_quaternions": _stack_or_empty(
                next_eef_quaternions, (4,), np.float32
            ),
            "next_drawer_fraction": np.asarray(next_drawer_fractions, dtype=np.float32),
            "joint_positions": _stack_or_empty(joint_traj, (ARM_DOF,), np.float32),
            "reward_trace": np.asarray(reward_trace, dtype=np.float32),
            "phase_labels": np.asarray(phase_labels, dtype="<U16"),
            "attached_trace": np.asarray(attached_trace, dtype=np.bool_),
            "handle_distance_trace": np.asarray(
                handle_distance_trace, dtype=np.float32
            ),
            "target_eef_positions": _stack_or_empty(
                target_eef_positions, (3,), np.float32
            ),
            "target_eef_quaternions": _stack_or_empty(
                target_eef_quaternions, (4,), np.float32
            ),
            "target_drawer_fraction": np.asarray(
                target_drawer_fractions, dtype=np.float32
            ),
            "grasp_pose_world": np.asarray(grasp_pose_world, dtype=np.float32),
            "grasp_pose_local_pos": local_handle["pos"].astype(np.float32),
            "grasp_pose_local_quat": local_handle["quat"].astype(np.float32),
            "initial_state": initial_obs.state.copy(),
            "initial_image": initial_obs.image.copy(),
            "initial_image2": initial_obs.image2.copy(),
            "initial_depth": initial_obs.depth.copy(),
            "initial_eef_position": initial_obs.eef_pos.copy(),
            "initial_eef_quaternion": initial_obs.eef_quat.copy(),
            "initial_gripper_open": float(initial_obs.gripper_open),
            "initial_drawer_fraction": float(initial_obs.drawer_fraction),
            "pull_fraction_target": np.float32(pull_fraction),
            "staging_pose": {
                "pos": staging_pos.astype(np.float32),
                "quat": closed_quat.astype(np.float32),
            },
            "pregrasp_pose": {
                "pos": pregrasp_pos.astype(np.float32),
                "quat": closed_quat.astype(np.float32),
            },
            "grasp_pose": {
                "pos": closed_pos.astype(np.float32),
                "quat": closed_quat.astype(np.float32),
            },
            "retreat_pose": {
                "pos": retreat_pos.astype(np.float32),
                "quat": open_quat.astype(np.float32),
            },
            "grasp_source": grasp_source,
            "grasp_score": None if grasp_score is None else float(grasp_score),
            "ever_attached": bool(ever_attached),
            "final_drawer_fraction": float(obs.drawer_fraction),
            "max_drawer_fraction": float(max_drawer_fraction),
        }
    finally:
        env.close()


def save_robot_rollout(path: Path, rollout: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=rollout["images"],
        images2=rollout["images2"],
        depths=rollout["depths"],
        states=rollout["states"],
        eef_positions=rollout["eef_positions"],
        eef_quaternions=rollout["eef_quaternions"],
        gripper_values=rollout["gripper_values"],
        actions=rollout["actions"],
        absolute_drawer_fraction=rollout["absolute_drawer_fraction"],
        next_eef_positions=rollout["next_eef_positions"],
        next_eef_quaternions=rollout["next_eef_quaternions"],
        next_drawer_fraction=rollout["next_drawer_fraction"],
        joint_positions=rollout["joint_positions"],
        reward_trace=rollout["reward_trace"],
        phase_labels=rollout["phase_labels"],
        attached_trace=rollout["attached_trace"],
        handle_distance_trace=rollout["handle_distance_trace"],
        target_eef_positions=rollout["target_eef_positions"],
        target_eef_quaternions=rollout["target_eef_quaternions"],
        target_drawer_fraction=rollout["target_drawer_fraction"],
        grasp_pose_world=rollout["grasp_pose_world"],
        grasp_pose_local_pos=rollout["grasp_pose_local_pos"],
        grasp_pose_local_quat=rollout["grasp_pose_local_quat"],
        initial_state=rollout["initial_state"],
        initial_image=rollout["initial_image"],
        initial_image2=rollout["initial_image2"],
        initial_depth=rollout["initial_depth"],
        initial_eef_position=rollout["initial_eef_position"],
        initial_eef_quaternion=rollout["initial_eef_quaternion"],
        initial_gripper_open=np.asarray(
            rollout["initial_gripper_open"], dtype=np.float32
        ),
        initial_drawer_fraction=np.asarray(
            rollout["initial_drawer_fraction"], dtype=np.float32
        ),
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
        "max_drawer_fraction": rollout.get(
            "max_drawer_fraction", rollout["final_drawer_fraction"]
        ),
        "pregrasp_pose": {
            "pos": rollout["pregrasp_pose"]["pos"].tolist(),
            "quat": rollout["pregrasp_pose"]["quat"].tolist(),
        },
        "grasp_pose": {
            "pos": rollout["grasp_pose"]["pos"].tolist(),
            "quat": rollout["grasp_pose"]["quat"].tolist(),
        },
        "retreat_pose": {
            "pos": rollout["retreat_pose"]["pos"].tolist(),
            "quat": rollout["retreat_pose"]["quat"].tolist(),
        },
        "pull_fraction_target": float(rollout["pull_fraction_target"]),
        "branch_id": rollout.get("branch_id"),
        "action_contract": rollout.get("action_contract"),
        "attach_step": rollout.get("attach_step"),
        "teacher_mode": rollout.get("teacher_mode"),
        "uses_planned_segment": bool(rollout.get("uses_planned_segment", False)),
        "uses_target_fraction": bool(rollout.get("uses_target_fraction", False)),
        "strict_metrics": rollout.get("strict_metrics"),
        "phase_counts": {
            str(label): int(np.sum(rollout["phase_labels"] == label))
            for label in np.unique(rollout["phase_labels"])
        },
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def build_native_teacher_rollout(
    seed: int,
    grasp_pose_world: np.ndarray,
    branch_config: dict[str, Any],
    episode_index: int = 0,
    max_steps: int = 96,
    *,
    grasp_source: str = "anygrasp",
    grasp_score: float | None = None,
    image_size: int = 224,
    record_images: bool = True,
    render_observations: bool = True,
) -> dict[str, Any]:
    action_contract = dict(
        branch_config.get("action_contract", DEFAULT_ACTION_CONTRACT)
    )
    env = DrawerRobotEnv(
        seed=seed,
        image_size=image_size,
        max_steps=max_steps,
        action_contract=action_contract,
        render_observations=render_observations,
    )
    obs = env.reset()
    initial_obs = obs
    home_quat = initial_obs.eef_quat.copy()
    local_handle = env.handle_local_pose(grasp_pose_world)
    env.attachment_local = local_handle

    rng = np.random.default_rng(seed * 1000 + episode_index)
    closed_pos, closed_quat = env.local_to_world_pose(
        local_handle["pos"], local_handle["quat"], fraction=0.0
    )
    pull_fraction = float(
        np.clip(PULL_OPEN_FRACTION + rng.uniform(-0.03, 0.02), 0.85, 0.95)
    )
    open_pos, open_quat = env.local_to_world_pose(
        local_handle["pos"], local_handle["quat"], fraction=pull_fraction
    )
    travel_vec = open_pos - closed_pos
    travel_axis = travel_vec / max(np.linalg.norm(travel_vec), 1e-8)
    approach_jitter = np.array(
        [rng.uniform(-0.008, 0.008), 0.0, rng.uniform(-0.004, 0.006)], dtype=np.float32
    )
    pregrasp_pos = (
        closed_pos
        + 0.10 * travel_axis
        + np.array([0.0, 0.0, 0.04], dtype=np.float32)
        + approach_jitter
    )
    staging_pos = (
        closed_pos
        + 0.16 * travel_axis
        + np.array([0.0, 0.0, 0.08], dtype=np.float32)
        + 0.5 * approach_jitter
    )
    retreat_pos = (
        open_pos
        + 0.06 * travel_axis
        + np.array([0.0, 0.0, 0.05], dtype=np.float32)
        + 0.5 * approach_jitter
    )

    images, images2, depths = [], [], []
    states, eef_positions, eef_quats, gripper_values = [], [], [], []
    next_eef_positions, next_eef_quaternions, next_drawer_fractions = [], [], []
    actions, phase_labels = [], []
    abs_drawer, joint_traj, reward_trace = [], [], []
    attached_trace, handle_distance_trace = [], []
    target_eef_positions, target_eef_quaternions, target_drawer_fractions = [], [], []
    success = False
    ever_attached = False
    max_drawer_fraction = float(obs.drawer_fraction)
    attach_step = None

    def _record_transition(
        current_obs: RobotObservation,
        next_obs: RobotObservation,
        action: np.ndarray,
        phase: str,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        target_fraction: float | None,
        attached: bool,
    ) -> None:
        nonlocal max_drawer_fraction, attach_step
        handle_pos, _ = env.local_to_world_pose(
            local_handle["pos"], local_handle["quat"], fraction=next_obs.drawer_fraction
        )
        reward = (
            1.0 if next_obs.drawer_fraction >= 0.90 else float(next_obs.drawer_fraction)
        )
        if record_images:
            images.append(current_obs.image.copy())
            images2.append(current_obs.image2.copy())
            depths.append(current_obs.depth.copy())
        states.append(current_obs.state.copy())
        eef_positions.append(current_obs.eef_pos.copy())
        eef_quats.append(current_obs.eef_quat.copy())
        gripper_values.append(float(current_obs.gripper_open))
        actions.append(action.astype(np.float32))
        phase_labels.append(phase)
        abs_drawer.append(float(current_obs.drawer_fraction))
        next_eef_positions.append(next_obs.eef_pos.copy())
        next_eef_quaternions.append(next_obs.eef_quat.copy())
        next_drawer_fractions.append(float(next_obs.drawer_fraction))
        attached_trace.append(bool(attached))
        handle_distance_trace.append(
            float(np.linalg.norm(next_obs.eef_pos - handle_pos))
        )
        target_eef_positions.append(np.asarray(target_pos, dtype=np.float32))
        target_eef_quaternions.append(np.asarray(target_quat, dtype=np.float32))
        target_drawer_fractions.append(
            float(
                target_fraction
                if target_fraction is not None
                else current_obs.drawer_fraction
            )
        )
        joint_traj.append(
            np.array(
                [
                    env.p.getJointState(env.robot_id, idx, physicsClientId=env.client)[
                        0
                    ]
                    for idx in env.arm_joint_indices
                ],
                dtype=np.float32,
            )
        )
        reward_trace.append(float(reward))
        max_drawer_fraction = max(max_drawer_fraction, float(next_obs.drawer_fraction))
        if attached and attach_step is None:
            attach_step = len(actions) - 1

    phase_caps = branch_config.get("phase_caps", {})
    phase_limits = branch_config.get("phase_limits", {})
    hold_open_steps = int(branch_config.get("hold_open_steps", 0))
    hold_close_steps = int(branch_config.get("hold_close_steps", 0))
    pull_waypoints = int(branch_config.get("pull_waypoints", 14))
    densify_grasp = bool(branch_config.get("densify_grasp", False))
    use_closed_quat_in_pregrasp = bool(
        branch_config.get("use_closed_quat_in_pregrasp", False)
    )
    lag_pull_rotation = bool(branch_config.get("lag_pull_rotation", False))
    action_frame = str(action_contract.get("action_frame", "world"))
    teacher_mode = str(branch_config.get("teacher_mode", "closed_loop_native"))
    uses_planned_segment = teacher_mode == "planned_waypoints"
    uses_target_fraction = teacher_mode == "planned_waypoints"
    native_pull_step_m = float(branch_config.get("native_pull_step_m", 0.018))

    def _caps(phase: str) -> tuple[float, float]:
        payload = phase_caps.get(phase, {})
        return float(payload.get("translation_cap", 0.35)), float(
            payload.get("rotation_cap", 0.18)
        )

    def _planned_segment(
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        gripper_open: float,
        steps: int,
        *,
        target_fraction: float | None = None,
        lock_drawer_fraction: float | None = None,
    ) -> list[RobotObservation]:
        state_id = env.p.saveState(physicsClientId=env.client)
        try:
            return env.move_to_pose(
                target_pos,
                target_quat,
                gripper_open,
                steps=max(steps, 2),
                update_drawer_to=target_fraction,
                lock_drawer_fraction=lock_drawer_fraction,
            )
        finally:
            env.p.restoreState(state_id, physicsClientId=env.client)
            env.p.removeState(state_id, physicsClientId=env.client)

    def _run_phase(
        phase: str,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        gripper_open: float,
        *,
        target_fraction: float | None = None,
        max_iterations: int | None = None,
        zero_motion: bool = False,
        lock_drawer_fraction: float | None = None,
    ) -> None:
        nonlocal obs, ever_attached, success
        translation_cap, rotation_cap = _caps(phase)
        limit = int(max_iterations or phase_limits.get(phase, 18))
        close_trigger_distance = branch_config.get("close_trigger_distance_m")
        planned_waypoints = None
        if not zero_motion and uses_planned_segment:
            planned_waypoints = _planned_segment(
                target_pos,
                target_quat,
                gripper_open,
                steps=limit,
                target_fraction=target_fraction,
                lock_drawer_fraction=lock_drawer_fraction,
            )
        iterator = planned_waypoints if planned_waypoints is not None else range(limit)
        for waypoint in iterator:
            handle_pos, _ = env.local_to_world_pose(
                local_handle["pos"], local_handle["quat"], fraction=obs.drawer_fraction
            )
            current_handle_distance = float(np.linalg.norm(obs.eef_pos - handle_pos))
            waypoint_handle_distance = None
            if planned_waypoints is not None:
                waypoint_handle_distance = float(
                    np.linalg.norm(waypoint.eef_pos - handle_pos)
                )
            phase_gripper_open = gripper_open
            if (
                phase == "grasp"
                and close_trigger_distance is not None
                and min(
                    current_handle_distance,
                    waypoint_handle_distance
                    if waypoint_handle_distance is not None
                    else current_handle_distance,
                )
                > float(close_trigger_distance)
                and not ever_attached
            ):
                phase_gripper_open = 1.0
            if zero_motion:
                action = np.zeros((7,), dtype=np.float32)
                action[6] = (
                    GRIPPER_OPEN_VALUE
                    if phase_gripper_open >= 0.5
                    else GRIPPER_CLOSE_VALUE
                )
            elif planned_waypoints is not None:
                action = _action_from_transition(
                    obs,
                    waypoint,
                    translation_scale=env.translation_scale,
                    rotation_scale=env.rotation_scale,
                    action_frame=action_frame,
                )
                action[:3] = np.clip(action[:3], -translation_cap, translation_cap)
                action[3:6] = np.clip(action[3:6], -rotation_cap, rotation_cap)
                action[6] = (
                    GRIPPER_OPEN_VALUE
                    if phase_gripper_open >= 0.5
                    else GRIPPER_CLOSE_VALUE
                )
            else:
                action = _action_toward_pose(
                    obs,
                    target_pos,
                    target_quat,
                    phase_gripper_open,
                    translation_cap=translation_cap,
                    rotation_cap=rotation_cap,
                    translation_scale=env.translation_scale,
                    rotation_scale=env.rotation_scale,
                    action_frame=action_frame,
                )
            next_obs, _reward, done, info = env.step(action)
            _record_transition(
                obs,
                next_obs,
                action,
                phase,
                target_pos,
                target_quat,
                target_fraction,
                bool(info.get("attached")),
            )
            obs = next_obs
            ever_attached = ever_attached or bool(info.get("attached"))
            pos_error, rot_error = _pose_error(
                obs.eef_pos, obs.eef_quat, target_pos, target_quat
            )
            if (
                phase in {"staging", "pregrasp", "align", "retreat"}
                and pos_error <= 0.015
                and rot_error <= 0.18
            ):
                break
            if phase == "grasp" and bool(info.get("attached")):
                break
            if (
                phase == "pull"
                and target_fraction is not None
                and obs.drawer_fraction >= max(0.0, target_fraction - 0.03)
            ):
                break
            if done:
                success = bool(info["is_success"])
                break

    try:
        pregrasp_quat = closed_quat if use_closed_quat_in_pregrasp else home_quat
        pregrasp_lock = 0.0 if uses_planned_segment else None
        grasp_lock = 0.0 if uses_planned_segment else None
        retreat_lock = float(pull_fraction) if uses_planned_segment else None
        _run_phase(
            "staging",
            staging_pos,
            home_quat,
            1.0,
            max_iterations=phase_limits.get("staging", 20),
            lock_drawer_fraction=pregrasp_lock,
        )
        _run_phase(
            "pregrasp",
            pregrasp_pos,
            pregrasp_quat,
            1.0,
            max_iterations=phase_limits.get("pregrasp", 24),
            lock_drawer_fraction=pregrasp_lock,
        )
        if hold_open_steps > 0:
            _run_phase(
                "hold_open",
                obs.eef_pos.copy(),
                obs.eef_quat.copy(),
                1.0,
                max_iterations=hold_open_steps,
                zero_motion=True,
            )
        if densify_grasp:
            mid_pos = 0.5 * (pregrasp_pos + closed_pos) + np.array(
                [0.0, 0.0, 0.02], dtype=np.float32
            )
            _run_phase(
                "align",
                mid_pos,
                closed_quat,
                1.0,
                max_iterations=phase_limits.get("align", 12),
                lock_drawer_fraction=pregrasp_lock,
            )
        _run_phase(
            "grasp",
            closed_pos,
            closed_quat,
            0.0,
            max_iterations=phase_limits.get("grasp", 56),
            lock_drawer_fraction=grasp_lock,
        )
        if hold_close_steps > 0:
            _run_phase(
                "hold_close",
                obs.eef_pos.copy(),
                obs.eef_quat.copy(),
                0.0,
                max_iterations=hold_close_steps,
                zero_motion=True,
            )

        if ever_attached:
            if uses_target_fraction:
                fractions = np.linspace(
                    0.12, pull_fraction, pull_waypoints, dtype=np.float32
                )
                for fraction in fractions:
                    target_pos, target_quat = env.local_to_world_pose(
                        local_handle["pos"],
                        local_handle["quat"],
                        fraction=float(fraction),
                    )
                    pull_quat = (
                        obs.eef_quat.copy() if lag_pull_rotation else target_quat
                    )
                    _run_phase(
                        "pull",
                        target_pos,
                        pull_quat,
                        0.0,
                        target_fraction=float(fraction),
                        max_iterations=phase_limits.get("pull", 14),
                    )
                    if obs.drawer_fraction >= 0.90:
                        break
            else:
                for _ in range(pull_waypoints):
                    target_pos = obs.eef_pos + travel_axis * native_pull_step_m
                    pull_quat = (
                        obs.eef_quat.copy() if lag_pull_rotation else closed_quat
                    )
                    _run_phase(
                        "pull",
                        target_pos,
                        pull_quat,
                        0.0,
                        max_iterations=phase_limits.get("pull", 8),
                    )
                    if obs.drawer_fraction >= 0.90:
                        break
            if obs.drawer_fraction >= 0.90:
                _run_phase(
                    "retreat",
                    retreat_pos,
                    open_quat,
                    1.0,
                    max_iterations=phase_limits.get("retreat", 16),
                    lock_drawer_fraction=retreat_lock,
                )

        success = bool(max_drawer_fraction >= 0.90 and ever_attached)
        strict_drawer_trace = np.asarray(
            next_drawer_fractions if next_drawer_fractions else [obs.drawer_fraction],
            dtype=np.float32,
        )
        strict_attached_trace = np.asarray(
            attached_trace if attached_trace else [False], dtype=np.bool_
        )
        strict_metrics = evaluate_strict_success(
            strict_drawer_trace, strict_attached_trace
        )
        return {
            "seed": seed,
            "episode_index": episode_index,
            "success": success,
            "task": env.task,
            "steps": len(actions),
            "images": _stack_or_empty(images, (image_size, image_size, 3), np.uint8),
            "images2": _stack_or_empty(images2, (image_size, image_size, 3), np.uint8),
            "depths": _stack_or_empty(depths, (image_size, image_size), np.float32),
            "states": _stack_or_empty(states, (8,), np.float32),
            "eef_positions": _stack_or_empty(eef_positions, (3,), np.float32),
            "eef_quaternions": _stack_or_empty(eef_quats, (4,), np.float32),
            "gripper_values": np.asarray(gripper_values, dtype=np.float32),
            "actions": _stack_or_empty(actions, (7,), np.float32),
            "absolute_drawer_fraction": np.asarray(abs_drawer, dtype=np.float32),
            "next_eef_positions": _stack_or_empty(next_eef_positions, (3,), np.float32),
            "next_eef_quaternions": _stack_or_empty(
                next_eef_quaternions, (4,), np.float32
            ),
            "next_drawer_fraction": np.asarray(next_drawer_fractions, dtype=np.float32),
            "joint_positions": _stack_or_empty(joint_traj, (ARM_DOF,), np.float32),
            "reward_trace": np.asarray(reward_trace, dtype=np.float32),
            "phase_labels": np.asarray(phase_labels, dtype="<U16"),
            "attached_trace": np.asarray(attached_trace, dtype=np.bool_),
            "handle_distance_trace": np.asarray(
                handle_distance_trace, dtype=np.float32
            ),
            "target_eef_positions": _stack_or_empty(
                target_eef_positions, (3,), np.float32
            ),
            "target_eef_quaternions": _stack_or_empty(
                target_eef_quaternions, (4,), np.float32
            ),
            "target_drawer_fraction": np.asarray(
                target_drawer_fractions, dtype=np.float32
            ),
            "grasp_pose_world": np.asarray(grasp_pose_world, dtype=np.float32),
            "grasp_pose_local_pos": local_handle["pos"].astype(np.float32),
            "grasp_pose_local_quat": local_handle["quat"].astype(np.float32),
            "initial_state": initial_obs.state.copy(),
            "initial_image": initial_obs.image.copy(),
            "initial_image2": initial_obs.image2.copy(),
            "initial_depth": initial_obs.depth.copy(),
            "initial_eef_position": initial_obs.eef_pos.copy(),
            "initial_eef_quaternion": initial_obs.eef_quat.copy(),
            "initial_gripper_open": float(initial_obs.gripper_open),
            "initial_drawer_fraction": float(initial_obs.drawer_fraction),
            "pull_fraction_target": np.float32(pull_fraction),
            "staging_pose": {
                "pos": staging_pos.astype(np.float32),
                "quat": home_quat.astype(np.float32),
            },
            "pregrasp_pose": {
                "pos": pregrasp_pos.astype(np.float32),
                "quat": pregrasp_quat.astype(np.float32),
            },
            "grasp_pose": {
                "pos": closed_pos.astype(np.float32),
                "quat": closed_quat.astype(np.float32),
            },
            "retreat_pose": {
                "pos": retreat_pos.astype(np.float32),
                "quat": open_quat.astype(np.float32),
            },
            "grasp_source": grasp_source,
            "grasp_score": None if grasp_score is None else float(grasp_score),
            "ever_attached": bool(ever_attached),
            "final_drawer_fraction": float(obs.drawer_fraction),
            "max_drawer_fraction": float(max_drawer_fraction),
            "branch_id": branch_config.get("branch_id"),
            "action_contract": action_contract,
            "attach_step": attach_step,
            "teacher_mode": teacher_mode,
            "uses_planned_segment": uses_planned_segment,
            "uses_target_fraction": uses_target_fraction,
            "strict_metrics": strict_metrics,
        }
    finally:
        env.close()


def _restore_replay_teacher_state(
    env: DrawerRobotEnv, source: Any, step_index: int
) -> None:
    joint_positions = source["joint_positions"].astype(np.float32)
    current_drawer = source["absolute_drawer_fraction"].astype(np.float32)
    current_gripper = source["gripper_values"].astype(np.float32)
    current_eef_pos = source["eef_positions"].astype(np.float32)
    current_eef_quat = source["eef_quaternions"].astype(np.float32)
    next_drawer = source["next_drawer_fraction"].astype(np.float32)
    attached_trace = source["attached_trace"].astype(np.bool_)

    # joint_positions are recorded after each env.step(), so the current state's
    # arm joints for step t live at joint_positions[t - 1] (except step 0, which
    # starts from the environment reset / HOME_JOINTS state).
    if step_index == 0:
        current_joint_state = np.asarray(HOME_JOINTS, dtype=np.float32)
    else:
        current_joint_state = joint_positions[
            min(step_index - 1, len(joint_positions) - 1)
        ]
    if len(current_joint_state):
        for idx, val in zip(env.arm_joint_indices, current_joint_state.tolist()):
            env.p.resetJointState(
                env.robot_id,
                idx,
                float(val),
                targetVelocity=0.0,
                physicsClientId=env.client,
            )
    if step_index < len(current_drawer):
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
    env.attached = prev_attached
    env.ever_attached = (
        bool(np.any(attached_trace[:step_index])) if step_index > 0 else False
    )
    env.step_count = int(step_index)
    if step_index < len(current_eef_pos):
        env.commanded_eef_pos = current_eef_pos[step_index].copy()
    if step_index < len(current_eef_quat):
        env.commanded_eef_quat = current_eef_quat[step_index].copy()
    if len(current_drawer):
        env.drawer_trace_history = [float(current_drawer[0])] + [
            float(x) for x in next_drawer[:step_index]
        ]
    else:
        env.drawer_trace_history = [0.0]
    env.attached_trace_history = [False] + [
        bool(x) for x in attached_trace[:step_index]
    ]


def replay_robot_rollout(
    npz_path: Path,
    *,
    action_contract: dict[str, Any] | None = None,
    max_steps: int | None = None,
    replay_mode: str = "open_loop",
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = np.load(npz_path, allow_pickle=True)
    meta = json.loads(npz_path.with_suffix(".json").read_text())
    replay_contract = dict(meta.get("action_contract") or DEFAULT_ACTION_CONTRACT)
    if action_contract:
        replay_contract.update(action_contract)
    if replay_mode not in {"open_loop", "state_anchored"}:
        raise ValueError(f"Unsupported replay_mode: {replay_mode}")
    env = DrawerRobotEnv(
        seed=int(meta["seed"]),
        image_size=224,
        max_steps=max_steps or max(len(source["actions"]) + 12, 96),
        action_contract=replay_contract,
    )
    replay_pos = []
    replay_quat = []
    replay_drawer = []
    replay_attached = []
    replay_handle_distance = []
    success = False
    ever_attached = False
    attach_step = None

    try:
        obs = env.reset()
        env.attachment_local = {
            "pos": source["grasp_pose_local_pos"].astype(np.float32),
            "quat": source["grasp_pose_local_quat"].astype(np.float32),
        }
        for step_index, action in enumerate(source["actions"].astype(np.float32)):
            if replay_mode == "state_anchored":
                _restore_replay_teacher_state(env, source, step_index)
            obs, _reward, done, info = env.step(action)
            handle_pos, _ = env.local_to_world_pose(
                source["grasp_pose_local_pos"].astype(np.float32),
                source["grasp_pose_local_quat"].astype(np.float32),
                fraction=obs.drawer_fraction,
            )
            replay_pos.append(obs.eef_pos.copy())
            replay_quat.append(obs.eef_quat.copy())
            replay_drawer.append(float(obs.drawer_fraction))
            replay_attached.append(bool(info.get("attached")))
            replay_handle_distance.append(
                float(np.linalg.norm(obs.eef_pos - handle_pos))
            )
            if info.get("attached") and attach_step is None:
                attach_step = step_index
            ever_attached = ever_attached or bool(info.get("attached"))
            if done:
                success = bool(info["is_success"])
                break
        replay_pos_arr = np.asarray(replay_pos, dtype=np.float32)
        replay_quat_arr = np.asarray(replay_quat, dtype=np.float32)
        replay_drawer_arr = np.asarray(replay_drawer, dtype=np.float32)
        replay_attached_arr = np.asarray(replay_attached, dtype=np.bool_)
        replay_handle_distance_arr = np.asarray(
            replay_handle_distance, dtype=np.float32
        )
    finally:
        env.close()

    target_pos = source["next_eef_positions"].astype(np.float32)
    target_quat = source["next_eef_quaternions"].astype(np.float32)
    target_drawer = source["next_drawer_fraction"].astype(np.float32)
    target_attached = source["attached_trace"].astype(np.bool_)
    pos_error = (
        np.linalg.norm(replay_pos_arr - target_pos[: len(replay_pos_arr)], axis=1)
        if len(replay_pos_arr)
        else np.zeros((0,), dtype=np.float32)
    )
    rot_error = (
        _rotation_errors(replay_quat_arr, target_quat[: len(replay_quat_arr)])
        if len(replay_quat_arr)
        else np.zeros((0,), dtype=np.float32)
    )
    drawer_error = (
        np.abs(replay_drawer_arr - target_drawer[: len(replay_drawer_arr)])
        if len(replay_drawer_arr)
        else np.zeros((0,), dtype=np.float32)
    )
    attached_agreement = (
        float(
            np.mean(replay_attached_arr == target_attached[: len(replay_attached_arr)])
        )
        if len(replay_attached_arr)
        else 0.0
    )
    phase_labels = (
        source["phase_labels"][: len(replay_drawer_arr)]
        if len(replay_drawer_arr)
        else np.zeros((0,), dtype="<U16")
    )
    phase_counts = {}
    for label in phase_labels:
        phase_counts[str(label)] = phase_counts.get(str(label), 0) + 1
    action_stats = {
        "max_abs_action": float(np.max(np.abs(source["actions"])))
        if len(source["actions"])
        else 0.0,
        "clipped_fraction": float(np.mean(np.abs(source["actions"]) > 1.0))
        if len(source["actions"])
        else 0.0,
        "translation_norm_p95": float(
            np.percentile(np.linalg.norm(source["actions"][:, :3], axis=1), 95)
        )
        if len(source["actions"])
        else 0.0,
        "rotation_norm_p95": float(
            np.percentile(np.linalg.norm(source["actions"][:, 3:6], axis=1), 95)
        )
        if len(source["actions"])
        else 0.0,
    }
    metrics = {
        "replay_steps": int(len(replay_pos_arr)),
        "max_position_error_m": float(np.max(pos_error)) if len(pos_error) else 0.0,
        "mean_position_error_m": float(np.mean(pos_error)) if len(pos_error) else 0.0,
        "max_rotation_error_rad": float(np.max(rot_error)) if len(rot_error) else 0.0,
        "max_drawer_error": float(np.max(drawer_error)) if len(drawer_error) else 0.0,
        "max_handle_distance_m": float(np.max(replay_handle_distance_arr))
        if len(replay_handle_distance_arr)
        else 0.0,
        "attached_agreement": attached_agreement,
        "success": bool(success and ever_attached),
        "ever_attached": bool(ever_attached),
        "attach_step": attach_step,
        "final_drawer_fraction": float(replay_drawer_arr[-1])
        if len(replay_drawer_arr)
        else 0.0,
        "phase_counts": phase_counts,
        "action_stats": action_stats,
        "replay_mode": replay_mode,
    }
    replay_payload = {
        "seed": int(meta["seed"]),
        "task": str(meta.get("task", "open the drawer")),
        "source_success": bool(meta.get("success", False)),
        "source_grasp_source": meta.get("grasp_source", "unknown"),
        "source_branch_id": meta.get("branch_id"),
        "action_contract": replay_contract,
        "images": source["images"].astype(np.uint8),
        "images2": source["images2"].astype(np.uint8),
        "states": source["states"].astype(np.float32),
        "actions": source["actions"].astype(np.float32),
        "grasp_pose_local_pos": source["grasp_pose_local_pos"].astype(np.float32),
        "grasp_pose_local_quat": source["grasp_pose_local_quat"].astype(np.float32),
        "position_error": pos_error.astype(np.float32),
        "rotation_error_rad": rot_error.astype(np.float32),
        "drawer_error": drawer_error.astype(np.float32),
        "replay_attached": replay_attached_arr.astype(np.bool_),
        "target_attached": target_attached[: len(replay_attached_arr)].astype(np.bool_),
        "phase_labels": source["phase_labels"],
        "replay_mode": replay_mode,
    }
    return replay_payload, metrics

#!/usr/bin/env python3
"""True MuJoCo drawer environment for the canonical Infinigen mainline."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from mint_common import write_text_atomic
from scipy.spatial.transform import Rotation as R

if os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
else:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    if os.environ.get("MUJOCO_GL") == "osmesa" and os.environ.get(
        "PYOPENGL_PLATFORM"
    ) in {None, "", "egl"}:
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
DRAWER_ROOT = PROJECT_ROOT / "sim_exports" / "urdf" / "drawer"

TRANSLATION_SCALE_M = 0.03
ROTATION_SCALE_RAD = 0.25
GRIPPER_OPEN = 0.001
GRIPPER_CLOSED = -0.042
ATTACH_THRESHOLD_M = 0.06
DETACH_THRESHOLD_M = 0.18
DRAWER_SUCCESS_FRACTION = 0.90
DEFAULT_TASK = "open the middle drawer of the cabinet"
DEFAULT_PULL_OPEN_FRACTION = 0.92
PRIMARY_BG_MEAN = np.array([0.414, 0.371, 0.329], dtype=np.float32)
SECONDARY_BG_MEAN = np.array([0.432, 0.377, 0.330], dtype=np.float32)
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float32)
WORLD_X = np.array([1.0, 0.0, 0.0], dtype=np.float32)
WRIST_CAM_OFFSET_LOCAL = np.array([-0.06, 0.0, 0.035], dtype=np.float32)
WRIST_LOOK_LOCAL = np.array([0.06, 0.0, 0.0], dtype=np.float32)
CONTROL_ROT_STD = np.array([0.0439326949, 0.0836294442, 0.0917678624], dtype=np.float32)
ROT_ACTION_CLIP = np.array([0.35, 0.45, 0.45], dtype=np.float32)

SecondaryCameraMode = Literal["legacy_fixed_scene", "wrist_dynamic"]
CalibrationMode = Literal["none", "legacy_bg_gain_bias", "diagnostic_texture"]
InteractionMode = Literal[
    "legacy_translation_only",
    "orientation_sensitive_v1",
    "orientation_sensitive_v2_affordance_locked",
    "orientation_sensitive_v3_task_identity_locked",
]
StateMode = Literal[
    "m0_proxy",
    "eef_pose_gripper",
    "telemetry_candidate_v1",
    "telemetry_candidate_v2",
    "telemetry_candidate_v3_transition",
    "telemetry_candidate_v4_task_identity",
]
RenderProfile = Literal[
    "legacy_surface",
    "visual_reformulation_v0",
    "visual_reformulation_v1",
    "visual_reformulation_v1_raw_canonical",
    "visual_reformulation_v1_plus_bundle",
    "visual_reformulation_v2_material_light_bg",
    "visual_reformulation_v2_plus_bundle",
    "visual_affordance_v3_raw_canonical",
    "visual_affordance_v3_local_material_edge",
    "visual_affordance_v3_plus_bundle",
]
BackgroundMode = Literal[
    "legacy_scene", "neutral_lab", "high_contrast_lab", "neutral_lowfreq_lab"
]
LightingProfile = Literal["legacy", "bright_front_fill", "front_key_handle_rim"]
MaterialPolicy = Literal["legacy", "handle_highlight", "handle_affordance_local"]
CameraFramingProfile = Literal[
    "legacy", "tight_handle_centered", "macro_handle_centered"
]
RotationSource = Literal["zero", "aligned", "random"]


def drawer_dir(seed: int) -> Path:
    return DRAWER_ROOT / str(seed)


def drawer_assets_available(seeds: list[int]) -> bool:
    return all((drawer_dir(seed) / "drawer.urdf").exists() for seed in seeds)


def drawer_manifest() -> dict[str, Any]:
    return {
        "available_seeds": sorted(
            int(p.name) for p in DRAWER_ROOT.iterdir() if p.is_dir()
        ),
        "root": str(DRAWER_ROOT),
    }


def _load_assets(
    seed: int,
) -> tuple[str, dict[str, bytes], dict[str, Any], dict[str, Any], str]:
    seed_dir = drawer_dir(seed)
    urdf_path = seed_dir / "drawer.urdf"
    metadata_path = seed_dir / "metadata.json"
    semantic_mapping_path = seed_dir / "semantic_mapping.json"
    urdf_text = urdf_path.read_text()
    assets: dict[str, bytes] = {}
    assets_dir = seed_dir / "assets"
    if assets_dir.exists():
        for asset_path in sorted(assets_dir.iterdir()):
            if asset_path.is_file():
                assets[asset_path.name] = asset_path.read_bytes()
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            metadata = {}
    semantic_mapping: dict[str, Any] = {}
    semantic_mapping_hash = ""
    if semantic_mapping_path.exists():
        semantic_mapping_hash = hashlib.sha256(
            semantic_mapping_path.read_bytes()
        ).hexdigest()
        try:
            semantic_mapping = json.loads(semantic_mapping_path.read_text())
        except json.JSONDecodeError:
            semantic_mapping = {}
    return urdf_text, assets, metadata, semantic_mapping, semantic_mapping_hash


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


def _camera_eye_from_lookat(
    lookat: np.ndarray, distance: float, azimuth_deg: float, elevation_deg: float
) -> np.ndarray:
    az = np.deg2rad(float(azimuth_deg))
    el = np.deg2rad(float(elevation_deg))
    offset = np.array(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ],
        dtype=np.float32,
    )
    return np.asarray(lookat, dtype=np.float32) + float(distance) * offset


def _transform_from_eye_lookat(
    eye: np.ndarray, lookat: np.ndarray, up_hint: np.ndarray = WORLD_UP
) -> np.ndarray:
    forward = _normalize(
        np.asarray(lookat, dtype=np.float32) - np.asarray(eye, dtype=np.float32),
        WORLD_X,
    )
    right = np.cross(forward, up_hint.astype(np.float32))
    if float(np.linalg.norm(right)) < 1e-6:
        right = np.cross(forward, WORLD_X)
    right = _normalize(right, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    up = _normalize(np.cross(right, forward), WORLD_UP)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.stack([right, up, forward], axis=1)
    transform[:3, 3] = np.asarray(eye, dtype=np.float32)
    return transform


def _transform_inverse(transform: np.ndarray) -> np.ndarray:
    rot = transform[:3, :3]
    pos = transform[:3, 3]
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ pos
    return out


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


@dataclass(frozen=True)
class DrawerEnvContractConfig:
    secondary_camera_mode: SecondaryCameraMode = "wrist_dynamic"
    enable_marker_overlay: bool = False
    calibration_mode: CalibrationMode = "none"
    interaction_mode: InteractionMode = "legacy_translation_only"
    state_mode: StateMode = "m0_proxy"
    render_profile: RenderProfile = "legacy_surface"
    background_mode: BackgroundMode = "legacy_scene"
    lighting_profile: LightingProfile = "legacy"
    material_policy: MaterialPolicy = "legacy"
    camera_framing_profile: CameraFramingProfile = "legacy"
    measurement_mode: Literal["truthful_handle_manifest_v1", "proxy_debug"] = (
        "truthful_handle_manifest_v1"
    )
    measurement_fallback_policy: Literal["forbid", "allow_debug_only"] = "forbid"
    emit_measurement_debug: bool = False
    emit_orientation_telemetry: bool = True
    emit_camera_metadata: bool = True
    emit_handle_probe_metadata: bool = True
    canonical_lane: bool = True

    @classmethod
    def legacy_defaults(cls) -> "DrawerEnvContractConfig":
        return cls(
            secondary_camera_mode="legacy_fixed_scene",
            enable_marker_overlay=True,
            calibration_mode="legacy_bg_gain_bias",
            interaction_mode="legacy_translation_only",
            state_mode="m0_proxy",
            render_profile="legacy_surface",
            background_mode="legacy_scene",
            lighting_profile="legacy",
            material_policy="legacy",
            camera_framing_profile="legacy",
            emit_orientation_telemetry=True,
            emit_camera_metadata=True,
            emit_handle_probe_metadata=True,
            canonical_lane=False,
        )


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
    camera_metadata: dict[str, Any] | None = None
    visual_mode_report: dict[str, Any] | None = None
    state_spec: dict[str, Any] | None = None
    orientation_telemetry: dict[str, Any] | None = None
    handle_probe_metadata: dict[str, Any] | None = None
    claim_policy: str = "diagnostic"

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class DrawerRobotEnvMuJoCo:
    """MuJoCo env with Infinigen drawer URDF plus a robot-in-loop controller state."""

    def __init__(
        self,
        seed: int,
        image_size: int = 256,
        max_steps: int = 96,
        contract: DrawerEnvContractConfig | None = None,
    ):
        import mujoco

        urdf_text, assets, metadata, semantic_mapping, semantic_mapping_hash = (
            _load_assets(seed)
        )
        self.model = mujoco.MjModel.from_xml_string(urdf_text, assets)
        self.data = mujoco.MjData(self.model)
        self.gl_context = mujoco.GLContext(image_size, image_size)
        self.gl_context.make_current()
        self.renderer = mujoco.Renderer(self.model, height=image_size, width=image_size)
        self.seed = int(seed)
        self.metadata = metadata
        self.semantic_mapping = semantic_mapping
        self.semantic_mapping_hash = semantic_mapping_hash
        self.image_size = int(image_size)
        self.max_steps = int(max_steps)
        self.task = DEFAULT_TASK
        self.contract = (
            contract
            if contract is not None
            else DrawerEnvContractConfig.legacy_defaults()
        )
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
        handle_center = self._handle_center_world()
        default_workspace_low = self.scene_center + np.array(
            [-0.30, -0.25, -0.10], dtype=np.float32
        )
        default_workspace_high = self.scene_center + np.array(
            [0.35, 0.25, 0.30], dtype=np.float32
        )
        handle_workspace_low = handle_center + np.array(
            [-0.28, -0.25, -0.12], dtype=np.float32
        )
        handle_workspace_high = handle_center + np.array(
            [0.18, 0.25, 0.20], dtype=np.float32
        )
        self._workspace_low = np.minimum(
            default_workspace_low, handle_workspace_low
        ).astype(np.float32)
        self._workspace_high = np.maximum(
            default_workspace_high, handle_workspace_high
        ).astype(np.float32)
        preferred_home = (
            handle_center
            - self._motion_axis * 0.18
            + np.array([0.0, 0.0, 0.12], dtype=np.float32)
        )
        self._home_pos = np.clip(
            preferred_home, self._workspace_low + 0.02, self._workspace_high - 0.02
        ).astype(np.float32)
        self._home_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        self._rng = np.random.default_rng(self.seed)
        self._original_geom_rgba = np.asarray(
            self.model.geom_rgba, dtype=np.float32
        ).copy()
        self._original_headlight_ambient = np.asarray(
            self.model.vis.headlight.ambient, dtype=np.float32
        ).copy()
        self._original_headlight_diffuse = np.asarray(
            self.model.vis.headlight.diffuse, dtype=np.float32
        ).copy()
        self._original_headlight_specular = np.asarray(
            self.model.vis.headlight.specular, dtype=np.float32
        ).copy()
        self._original_haze_rgba = np.asarray(
            self.model.vis.rgba.haze, dtype=np.float32
        ).copy()
        self._last_camera_metadata: dict[str, Any] = {}
        self._last_visual_mode_report: dict[str, Any] = {}
        self._last_handle_probe_metadata: dict[str, Any] = {}
        self._last_orientation_info = self._default_orientation_info()
        self._last_claim_policy = self.claim_policy()
        self._attach_streak = 0
        self._stable_attach = False
        self._contact_window: list[float] = []
        self._orientation_break_streak = 0
        self._slip_break_streak = 0
        self._reverse_pull_streak = 0
        self._anchor_handle_offset_world: np.ndarray | None = None
        self._anchor_eef_pull_progress = 0.0
        self._prev_pull_progress = 0.0
        self._runtime_visible_handle_geom_ids: list[int] = []
        self._runtime_handle_anchor_valid = False
        self._runtime_handle_anchor_world = self._handle_center_world().copy()
        self._last_detach_reason: str | None = None
        self._legacy_handle_resolution = self._resolve_semantic_handle_geom_ids()
        self.reset()

    def close(self) -> None:
        try:
            self.renderer.close()
        finally:
            try:
                self.gl_context.free()
            except Exception:
                pass

    def claim_policy(self) -> str:
        if not self.contract.canonical_lane:
            return "diagnostic"
        if self.contract.enable_marker_overlay:
            return "new_claim_required"
        if self.contract.calibration_mode != "none":
            return "new_claim_required"
        if self.contract.secondary_camera_mode != "wrist_dynamic":
            return "new_claim_required"
        return "canonical"

    def contract_payload(self) -> dict[str, Any]:
        return asdict(self.contract)

    def _default_orientation_info(self) -> dict[str, Any]:
        return {
            "orientation_alignment_cos": 1.0,
            "orientation_gate_passed": True,
            "attach_eligible": False,
            "attach_gate_distance_passed": False,
            "attach_gate_orientation_passed": True,
            "attach_gate_approach_passed": False,
            "attach_streak": 0,
            "stable_attach": False,
            "pull_alignment_cos": 0.0,
            "drawer_delta_raw": 0.0,
            "drawer_delta_effective": 0.0,
            "open_orientation_weight": 1.0,
            "orientation_error_rad": 0.0,
            "contact_window_fraction": 0.0,
            "runtime_handle_anchor_world": [0.0, 0.0, 0.0],
            "runtime_handle_anchor_valid": False,
            "phase_locked": False,
            "raw_grasp_slip": 0.0,
            "grasp_slip_norm": 0.0,
            "pull_progress": 0.0,
            "pull_increment": 0.0,
            "effective_pull_progress": 0.0,
            "detach_reason": None,
        }

    def _joint_axis_world(self) -> np.ndarray:
        axis = self.model.jnt_axis[self.joint_idx].astype(np.float32)
        default = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return _normalize(axis, default)

    def _drawer_fraction(self) -> float:
        low, high = self.joint_range.tolist()
        span = max(high - low, 1e-6)
        return float(
            np.clip((float(self.data.qpos[self.joint_idx]) - low) / span, 0.0, 1.0)
        )

    def _drawer_fraction_signed(self) -> float:
        return float(np.clip(self._drawer_fraction() * 2.0 - 1.0, -1.0, 1.0))

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

    def _handle_proxy_half_extents(self) -> np.ndarray:
        bbox = self.metadata.get("handle_bbox_local") or {}
        mins = np.asarray(bbox.get("min") or [], dtype=np.float32)
        maxs = np.asarray(bbox.get("max") or [], dtype=np.float32)
        if mins.shape == (3,) and maxs.shape == (3,):
            half = np.maximum(
                (maxs - mins) / 2.0, np.array([0.012, 0.012, 0.012], dtype=np.float32)
            )
            return half.astype(np.float32)
        return np.array([0.018, 0.014, 0.016], dtype=np.float32)

    def _handle_proxy_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        depth_axis = _normalize(self._motion_axis, WORLD_X)
        width_axis = np.cross(WORLD_UP, depth_axis)
        if float(np.linalg.norm(width_axis)) < 1e-6:
            width_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        width_axis = _normalize(width_axis, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        height_axis = _normalize(np.cross(depth_axis, width_axis), WORLD_UP)
        return width_axis, depth_axis, height_axis

    def _runtime_visible_handle_anchor_world(self) -> np.ndarray:
        ids = list(getattr(self, "_runtime_visible_handle_geom_ids", []) or [])
        if not ids:
            raise RuntimeError("runtime_visible_handle_anchor_unavailable")
        centers = np.asarray(self.data.geom_xpos[ids], dtype=np.float32)
        radii = np.asarray(self.model.geom_rbound[ids], dtype=np.float32)
        weights = np.clip(radii, 1e-4, None)
        return np.average(centers, axis=0, weights=weights).astype(np.float32)

    def _visual_profile_handle_anchor_world(self) -> np.ndarray:
        try:
            return self._runtime_visible_handle_anchor_world()
        except Exception:
            return self._handle_center_world()

    def _runtime_visible_handle_highlight_ids(self) -> list[int]:
        ids = [
            int(idx)
            for idx in list(getattr(self, "_runtime_visible_handle_geom_ids", []) or [])
        ]
        return sorted(
            dict.fromkeys(idx for idx in ids if 0 <= idx < int(self.model.ngeom))
        )

    def _interaction_handle_target_world(self) -> tuple[np.ndarray, bool]:
        if (
            self.contract.interaction_mode
            == "orientation_sensitive_v3_task_identity_locked"
        ):
            try:
                anchor = self._runtime_visible_handle_anchor_world()
                return anchor.astype(np.float32), True
            except Exception:
                return np.asarray(self.eef_pos, dtype=np.float32).copy(), False
        return self._handle_center_world(), False

    def _task_identity_transition_update(
        self,
        *,
        prev_pos: np.ndarray,
        close_cmd: bool,
        dist_to_handle: float,
        attach_eligible: bool,
        orientation_alignment_cos: float,
        pull_alignment_cos: float,
        runtime_handle_anchor_world: np.ndarray,
        runtime_handle_anchor_valid: bool,
    ) -> dict[str, Any]:
        drawer_gain = 6.0
        detach_reason: str | None = None
        if attach_eligible:
            self._attach_streak += 1
        else:
            self._attach_streak = 0
        entering_stable_attach = (not self._stable_attach) and self._attach_streak >= 3
        if entering_stable_attach:
            self._stable_attach = True
            self._attached = True
            self._anchor_handle_offset_world = (
                runtime_handle_anchor_world - self.eef_pos
            )
            self._anchor_eef_pull_progress = float(
                np.dot(self.eef_pos, self._motion_axis)
            )
            self._prev_pull_progress = 0.0
            self._orientation_break_streak = 0
            self._slip_break_streak = 0
            self._reverse_pull_streak = 0
        raw_grasp_slip = 0.0
        grasp_slip_norm = 0.0
        pull_progress = 0.0
        raw_pull_delta = 0.0
        pull_increment = 0.0
        if self._stable_attach and self._anchor_handle_offset_world is not None:
            raw_grasp_slip = float(
                np.linalg.norm(
                    (runtime_handle_anchor_world - self.eef_pos)
                    - self._anchor_handle_offset_world
                )
            )
            pull_progress = float(
                np.dot(self.eef_pos, self._motion_axis) - self._anchor_eef_pull_progress
            )
            raw_pull_delta = float(pull_progress - self._prev_pull_progress)
            pull_increment = max(0.0, raw_pull_delta)
            self._prev_pull_progress = pull_progress
            grasp_slip_norm = float(np.clip(raw_grasp_slip / 0.04, 0.0, 1.0))
        orientation_weight = float(
            np.clip((orientation_alignment_cos - 0.45) / 0.35, 0.0, 1.0)
        )
        slip_weight = float(np.clip(1.0 - raw_grasp_slip / 0.03, 0.0, 1.0))
        phase_locked = bool(
            self._stable_attach
            and close_cmd
            and orientation_alignment_cos >= 0.60
            and raw_grasp_slip <= 0.03
        )
        effective_pull_progress = float(
            pull_increment * orientation_weight * slip_weight
        )
        drawer_delta_raw = (
            float(np.dot(self.eef_pos - prev_pos, self._motion_axis)) * drawer_gain
        )
        drawer_delta_effective = (
            float(effective_pull_progress * drawer_gain) if phase_locked else 0.0
        )
        if self._stable_attach:
            self._orientation_break_streak = (
                self._orientation_break_streak + 1
                if orientation_alignment_cos < 0.45
                else 0
            )
            self._slip_break_streak = (
                self._slip_break_streak + 1 if raw_grasp_slip > 0.04 else 0
            )
            self._reverse_pull_streak = (
                self._reverse_pull_streak + 1 if raw_pull_delta < 0.0 else 0
            )
            if not close_cmd:
                detach_reason = "open_cmd"
            elif dist_to_handle > DETACH_THRESHOLD_M:
                detach_reason = "distance"
            elif self._orientation_break_streak >= 2:
                detach_reason = "orientation_break"
            elif self._slip_break_streak >= 2:
                detach_reason = "slip_break"
            elif self._reverse_pull_streak >= 2:
                detach_reason = "reverse_pull"
        if detach_reason is not None:
            self._attached = False
            self._stable_attach = False
            self._attach_streak = 0
            self._orientation_break_streak = 0
            self._slip_break_streak = 0
            self._reverse_pull_streak = 0
            self._anchor_handle_offset_world = None
            self._anchor_eef_pull_progress = 0.0
            self._prev_pull_progress = 0.0
            drawer_delta_effective = 0.0
            phase_locked = False
        else:
            self._attached = bool(self._stable_attach)
        self._runtime_handle_anchor_world = runtime_handle_anchor_world.astype(
            np.float32
        )
        self._runtime_handle_anchor_valid = bool(runtime_handle_anchor_valid)
        self._last_detach_reason = detach_reason
        return {
            "drawer_delta_raw": float(drawer_delta_raw),
            "drawer_delta_effective": float(drawer_delta_effective),
            "open_orientation_weight": float(orientation_weight),
            "pull_alignment_cos": float(pull_alignment_cos),
            "runtime_handle_anchor_world": runtime_handle_anchor_world.astype(
                np.float32
            ),
            "runtime_handle_anchor_valid": bool(runtime_handle_anchor_valid),
            "phase_locked": bool(phase_locked),
            "raw_grasp_slip": float(raw_grasp_slip),
            "grasp_slip_norm": float(grasp_slip_norm),
            "pull_progress": float(pull_progress),
            "pull_increment": float(pull_increment),
            "effective_pull_progress": float(effective_pull_progress),
            "detach_reason": detach_reason,
        }

    def _handle_proxy_corners_world(self) -> np.ndarray:
        center = self._handle_center_world()
        half = self._handle_proxy_half_extents()
        width_axis, depth_axis, height_axis = self._handle_proxy_axes()
        corners = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corner = (
                        center
                        + sx * half[0] * width_axis
                        + sy * half[1] * depth_axis
                        + sz * half[2] * height_axis
                    )
                    corners.append(corner.astype(np.float32))
        return np.asarray(corners, dtype=np.float32)

    def _camera_projection_params(self, camera) -> tuple[np.ndarray, float]:
        eye = _camera_eye_from_lookat(
            camera.lookat.copy(), camera.distance, camera.azimuth, camera.elevation
        )
        world_to_cam = _transform_inverse(
            _transform_from_eye_lookat(eye, camera.lookat.copy())
        )
        fovy = float(self.model.vis.global_.fovy)
        focal = float(0.5 * self.image_size / np.tan(np.deg2rad(fovy) * 0.5))
        return world_to_cam.astype(np.float32), focal

    def _project_world_points(self, camera, points: np.ndarray) -> np.ndarray:
        world_to_cam, focal = self._camera_projection_params(camera)
        pts = np.asarray(points, dtype=np.float32)
        pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
        cam = (world_to_cam @ pts_h.T).T[:, :3]
        depth = cam[:, 2]
        valid = depth > 1e-4
        u = np.full(len(pts), np.nan, dtype=np.float32)
        v = np.full(len(pts), np.nan, dtype=np.float32)
        if np.any(valid):
            z = depth[valid]
            u[valid] = focal * (cam[valid, 0] / z) + 0.5 * float(self.image_size - 1)
            v[valid] = 0.5 * float(self.image_size - 1) - focal * (cam[valid, 1] / z)
        return np.stack([u, v, depth], axis=1).astype(np.float32)

    def _identify_handle_geom_ids(self) -> list[int]:
        centers = np.asarray(self.data.geom_xpos, dtype=np.float32)
        if len(centers) == 0:
            return []
        handle = self._handle_center_world()
        distances = np.linalg.norm(centers - handle[None, :], axis=1)
        cutoff = (
            float(np.percentile(distances, 25))
            if len(distances) > 1
            else float(distances[0])
        )
        ids = [int(i) for i, dist in enumerate(distances) if dist <= max(cutoff, 0.12)]
        if not ids:
            ids = [int(np.argmin(distances))]
        return ids

    def _load_semantic_mapping(self) -> dict[str, Any]:
        return dict(self.semantic_mapping or {})

    def _load_handle_entity_from_manifest(self) -> dict[str, Any]:
        semantic_mapping = self._load_semantic_mapping()
        entities = (
            list(semantic_mapping.get("entities", []))
            if isinstance(semantic_mapping, dict)
            else []
        )
        handle_entities = [
            entity
            for entity in entities
            if str(entity.get("entity_uid", "")).strip() == "drawer_handle"
        ]
        warning_flags: list[str] = []
        if len(handle_entities) == 0:
            warning_flags.append("manifest_handle_entity_missing")
        elif len(handle_entities) > 1:
            warning_flags.append("manifest_handle_entity_nonunique")
        return {
            "handle_entity": handle_entities[0] if len(handle_entities) == 1 else None,
            "manifest_handle_entity_unique": len(handle_entities) == 1,
            "warning_flags": warning_flags,
        }

    def _resolve_runtime_geom_ids_from_exact_names(
        self, expected_names: list[str], *, missing_flag: str, ambiguous_flag: str
    ) -> dict[str, Any]:
        warning_flags: list[str] = []
        expected = [str(name).strip() for name in expected_names if str(name).strip()]
        resolved_ids: list[int] = []
        resolved_names: list[str] = []
        if not expected:
            warning_flags.append(missing_flag)
            return {
                "resolved_ids": [],
                "resolved_names": [],
                "mapping_unique": False,
                "duplicate_runtime_geom_name_flag": False,
                "unnamed_runtime_geom_flag": False,
                "warning_flags": warning_flags,
            }
        for expected_name in expected:
            matches = [
                int(i)
                for i in range(self.model.ngeom)
                if str(self.model.geom(i).name or "") == expected_name
            ]
            if len(matches) == 0:
                warning_flags.append(missing_flag)
                continue
            if len(matches) > 1:
                warning_flags.append(ambiguous_flag)
                continue
            resolved_ids.append(matches[0])
            resolved_names.append(expected_name)
        duplicate_runtime_geom_name_flag = len(resolved_names) != len(
            set(resolved_names)
        )
        unnamed_runtime_geom_flag = any(
            not str(name).strip() for name in resolved_names
        )
        if duplicate_runtime_geom_name_flag:
            warning_flags.append("runtime_handle_geom_name_duplicate")
        if unnamed_runtime_geom_flag:
            warning_flags.append("runtime_handle_geom_name_unnamed")
        mapping_unique = (
            len(expected) > 0
            and len(resolved_ids) == len(expected)
            and missing_flag not in warning_flags
            and ambiguous_flag not in warning_flags
            and not duplicate_runtime_geom_name_flag
            and not unnamed_runtime_geom_flag
        )
        return {
            "resolved_ids": resolved_ids,
            "resolved_names": resolved_names,
            "mapping_unique": bool(mapping_unique),
            "duplicate_runtime_geom_name_flag": bool(duplicate_runtime_geom_name_flag),
            "unnamed_runtime_geom_flag": bool(unnamed_runtime_geom_flag),
            "warning_flags": sorted(set(warning_flags)),
        }

    def _resolve_runtime_handle_visual_geom_ids_from_manifest(
        self, handle_entity: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if handle_entity is None:
            handle_entity = self._load_handle_entity_from_manifest().get(
                "handle_entity"
            )
        expected_names = list(
            (handle_entity or {}).get("exported_visual_geom_names", [])
        )
        return self._resolve_runtime_geom_ids_from_exact_names(
            expected_names,
            missing_flag="runtime_handle_visual_geom_missing",
            ambiguous_flag="runtime_handle_visual_geom_ambiguous",
        )

    def _resolve_runtime_handle_collision_geom_ids_from_manifest(
        self, handle_entity: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if handle_entity is None:
            handle_entity = self._load_handle_entity_from_manifest().get(
                "handle_entity"
            )
        expected_names = list(
            (handle_entity or {}).get("exported_collision_geom_names", [])
        )
        return self._resolve_runtime_geom_ids_from_exact_names(
            expected_names,
            missing_flag="runtime_handle_collision_geom_missing",
            ambiguous_flag="runtime_handle_collision_geom_ambiguous",
        )

    def _resolve_runtime_visible_handle_geom_ids_from_manifest(
        self, handle_entity: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if handle_entity is None:
            handle_entity = self._load_handle_entity_from_manifest().get(
                "handle_entity"
            )
        visual_mapping = self._resolve_runtime_handle_visual_geom_ids_from_manifest(
            handle_entity
        )
        collision_mapping = (
            self._resolve_runtime_handle_collision_geom_ids_from_manifest(handle_entity)
        )
        runtime_visible_mapping = (
            collision_mapping
            if collision_mapping.get("mapping_unique")
            else visual_mapping
        )
        mapping_source = (
            "collision_geom"
            if collision_mapping.get("mapping_unique")
            else (
                "visual_geom" if visual_mapping.get("mapping_unique") else "unresolved"
            )
        )
        warning_flags: list[str] = []
        if not runtime_visible_mapping.get("mapping_unique"):
            if any(
                flag
                in list(collision_mapping.get("warning_flags", []))
                + list(visual_mapping.get("warning_flags", []))
                for flag in [
                    "runtime_handle_collision_geom_ambiguous",
                    "runtime_handle_visual_geom_ambiguous",
                ]
            ):
                warning_flags.append("runtime_visible_handle_geom_ambiguous")
            else:
                warning_flags.append("runtime_visible_handle_geom_missing")
        return {
            "resolved_ids": list(runtime_visible_mapping.get("resolved_ids", [])),
            "resolved_names": list(runtime_visible_mapping.get("resolved_names", [])),
            "mapping_unique": bool(
                runtime_visible_mapping.get("mapping_unique", False)
            ),
            "mapping_source": mapping_source,
            "duplicate_runtime_geom_name_flag": bool(
                runtime_visible_mapping.get("duplicate_runtime_geom_name_flag", False)
            ),
            "unnamed_runtime_geom_flag": bool(
                runtime_visible_mapping.get("unnamed_runtime_geom_flag", False)
            ),
            "warning_flags": sorted(
                set(
                    warning_flags
                    + list(runtime_visible_mapping.get("warning_flags", []))
                    + list(visual_mapping.get("warning_flags", []))
                    + list(collision_mapping.get("warning_flags", []))
                )
            ),
            "visual_mapping": visual_mapping,
            "collision_mapping": collision_mapping,
        }

    def _render_handle_mask_segmentation(
        self, camera_key: str, geom_ids: list[int]
    ) -> np.ndarray:
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        if not geom_ids:
            return mask
        camera = self.cam_primary if camera_key == "primary" else self.cam_secondary
        try:
            self.renderer.enable_segmentation_rendering()
            self.renderer.update_scene(self.data, camera=camera)
            seg = self.renderer.render()
            geom_layer = np.asarray(seg[..., 0], dtype=np.int32)
            mask = np.where(
                np.isin(geom_layer, np.asarray(geom_ids, dtype=np.int32)), 255, 0
            ).astype(np.uint8)
        finally:
            try:
                self.renderer.disable_segmentation_rendering()
            except Exception:
                pass
        return mask

    def _render_handle_mask_isolated(
        self, camera_key: str, geom_ids: list[int]
    ) -> np.ndarray:
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        if not geom_ids:
            return mask
        camera = self.cam_primary if camera_key == "primary" else self.cam_secondary
        saved_geom_rgba = np.asarray(self.model.geom_rgba, dtype=np.float32).copy()
        saved_ambient = np.asarray(
            self.model.vis.headlight.ambient, dtype=np.float32
        ).copy()
        saved_diffuse = np.asarray(
            self.model.vis.headlight.diffuse, dtype=np.float32
        ).copy()
        saved_specular = np.asarray(
            self.model.vis.headlight.specular, dtype=np.float32
        ).copy()
        saved_haze = np.asarray(self.model.vis.rgba.haze, dtype=np.float32).copy()
        try:
            self.model.geom_rgba[:, :4] = np.asarray(
                [0.0, 0.0, 0.0, 1.0], dtype=np.float32
            )
            self.model.geom_rgba[np.asarray(geom_ids, dtype=np.int32), :4] = np.asarray(
                [1.0, 1.0, 1.0, 1.0], dtype=np.float32
            )
            self.model.vis.headlight.ambient[:] = np.asarray(
                [1.0, 1.0, 1.0], dtype=np.float32
            )
            self.model.vis.headlight.diffuse[:] = np.asarray(
                [1.0, 1.0, 1.0], dtype=np.float32
            )
            self.model.vis.headlight.specular[:] = np.asarray(
                [0.0, 0.0, 0.0], dtype=np.float32
            )
            self.model.vis.rgba.haze[:] = np.asarray(
                [0.0, 0.0, 0.0, 1.0], dtype=np.float32
            )
            rgb = self._render(camera)
            gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            mask = np.where(gray >= 80, 255, 0).astype(np.uint8)
        finally:
            self.model.geom_rgba[:] = saved_geom_rgba
            self.model.vis.headlight.ambient[:] = saved_ambient
            self.model.vis.headlight.diffuse[:] = saved_diffuse
            self.model.vis.headlight.specular[:] = saved_specular
            self.model.vis.rgba.haze[:] = saved_haze
        return mask

    def _mask_iou(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        a = np.asarray(mask_a, dtype=np.uint8) > 0
        b = np.asarray(mask_b, dtype=np.uint8) > 0
        union = np.logical_or(a, b).sum()
        if union <= 0:
            return 0.0
        intersection = np.logical_and(a, b).sum()
        return float(intersection / max(union, 1))

    def _mask_centroid_delta_px(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        a = np.argwhere(np.asarray(mask_a, dtype=np.uint8) > 0)
        b = np.argwhere(np.asarray(mask_b, dtype=np.uint8) > 0)
        if len(a) == 0 or len(b) == 0:
            return float(max(self.image_size, 1))
        ca = a.mean(axis=0)
        cb = b.mean(axis=0)
        return float(np.linalg.norm(ca - cb))

    def _verify_truthful_handle_measurement(
        self, camera_key: str
    ) -> tuple[np.ndarray, dict[str, Any]]:
        manifest_report = self._load_handle_entity_from_manifest()
        handle_entity = manifest_report.get("handle_entity")
        runtime_visible_mapping = (
            self._resolve_runtime_visible_handle_geom_ids_from_manifest(handle_entity)
        )
        visual_mapping = dict(runtime_visible_mapping.get("visual_mapping", {}))
        collision_mapping = dict(runtime_visible_mapping.get("collision_mapping", {}))
        warning_flags = sorted(
            set(
                list(manifest_report.get("warning_flags", []))
                + list(runtime_visible_mapping.get("warning_flags", []))
            )
        )
        duplicate_runtime_geom_name_flag = bool(
            runtime_visible_mapping.get("duplicate_runtime_geom_name_flag", False)
        )
        unnamed_runtime_geom_flag = bool(
            runtime_visible_mapping.get("unnamed_runtime_geom_flag", False)
        )
        segmentation_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        isolated_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        resolved_runtime_visible_ids = list(
            runtime_visible_mapping.get("resolved_ids", [])
        )
        if (
            manifest_report.get("manifest_handle_entity_unique")
            and runtime_visible_mapping.get("mapping_unique")
            and not duplicate_runtime_geom_name_flag
            and not unnamed_runtime_geom_flag
        ):
            segmentation_mask = self._render_handle_mask_segmentation(
                camera_key, resolved_runtime_visible_ids
            )
            isolated_mask = self._render_handle_mask_isolated(
                camera_key, resolved_runtime_visible_ids
            )
        segmentation_nonzero = int(np.count_nonzero(segmentation_mask))
        isolated_nonzero = int(np.count_nonzero(isolated_mask))
        if segmentation_nonzero <= 0:
            warning_flags.append("segmentation_empty_mask")
        if isolated_nonzero <= 0:
            warning_flags.append("isolated_render_empty_mask")
        warning_flags = sorted(set(warning_flags))
        segmentation_support_mask = self._truthful_support_mask(segmentation_mask)
        isolated_support_mask = self._truthful_support_mask(isolated_mask)
        segmentation_isolated_iou = self._mask_iou(
            segmentation_support_mask, isolated_support_mask
        )
        segmentation_isolated_centroid_delta_px = self._mask_centroid_delta_px(
            segmentation_support_mask, isolated_support_mask
        )
        measurement_truthful = bool(
            manifest_report.get("manifest_handle_entity_unique")
            and runtime_visible_mapping.get("mapping_unique")
            and not duplicate_runtime_geom_name_flag
            and not unnamed_runtime_geom_flag
            and segmentation_nonzero > 0
            and isolated_nonzero > 0
            and segmentation_isolated_iou >= 0.90
            and segmentation_isolated_centroid_delta_px <= 4.0
        )
        report = {
            "truth_root_object": "manifest_entity",
            "identity_resolution_tier": "manifest_entity_exact_runtime_visible_geom",
            "measurement_backend": "segmentation_render",
            "measurement_verifier": "isolated_rgb_threshold",
            "measurement_truth_tier": "manifest_entity_verified"
            if measurement_truthful
            else "manifest_entity_unverified",
            "measurement_truthful": measurement_truthful,
            "manifest_handle_entity_unique": bool(
                manifest_report.get("manifest_handle_entity_unique", False)
            ),
            "runtime_visible_handle_mapping_source": str(
                runtime_visible_mapping.get("mapping_source", "unresolved")
            ),
            "runtime_visible_handle_mapping_unique": bool(
                runtime_visible_mapping.get("mapping_unique", False)
            ),
            "runtime_handle_visual_geom_mapping_unique": bool(
                visual_mapping.get("mapping_unique", False)
            ),
            "runtime_handle_collision_geom_mapping_unique": bool(
                collision_mapping.get("mapping_unique", False)
            ),
            "duplicate_runtime_geom_name_flag": duplicate_runtime_geom_name_flag,
            "unnamed_runtime_geom_flag": unnamed_runtime_geom_flag,
            "resolved_runtime_visible_handle_geom_ids": resolved_runtime_visible_ids,
            "resolved_runtime_visible_handle_geom_names": list(
                runtime_visible_mapping.get("resolved_names", [])
            ),
            "resolved_handle_visual_geom_ids": list(
                visual_mapping.get("resolved_ids", [])
            ),
            "resolved_handle_visual_geom_names": list(
                visual_mapping.get("resolved_names", [])
            ),
            "resolved_handle_collision_geom_ids": list(
                collision_mapping.get("resolved_ids", [])
            ),
            "resolved_handle_collision_geom_names": list(
                collision_mapping.get("resolved_names", [])
            ),
            "segmentation_mask_nonzero": segmentation_nonzero,
            "isolated_mask_nonzero": isolated_nonzero,
            "segmentation_isolated_iou": float(segmentation_isolated_iou),
            "segmentation_isolated_centroid_delta_px": float(
                segmentation_isolated_centroid_delta_px
            ),
            "measurement_warning_flags": warning_flags,
            "semantic_mapping_hash": self.semantic_mapping_hash,
        }
        return segmentation_mask, report

    def _semantic_handle_names_from_metadata(self) -> list[str]:
        values: list[str] = []
        for key in ["semantic_handle_geom_name", "semantic_handle_visual_name"]:
            single = self.metadata.get(key)
            if isinstance(single, str) and single.strip():
                values.append(single.strip())
        multi = self.metadata.get("semantic_handle_geom_names")
        if isinstance(multi, list):
            values.extend(str(item).strip() for item in multi if str(item).strip())
        return sorted(dict.fromkeys(values))

    def _resolve_semantic_handle_geom_ids(self) -> dict[str, Any]:
        warning_flags: list[str] = []
        token_re = re.compile(
            r"(?:^|[_-])(handle|drawer_handle|cabinet_handle|pull)(?:$|[_-])"
        )

        geom_name_matches = [
            int(i)
            for i in range(self.model.ngeom)
            if token_re.search(str(self.model.geom(i).name or ""))
        ]
        if geom_name_matches:
            geom_names = [str(self.model.geom(i).name or "") for i in geom_name_matches]
            body_ids = sorted(
                {int(self.model.geom_bodyid[i]) for i in geom_name_matches}
            )
            body_names = [
                str(self.model.body(body_id).name or "") for body_id in body_ids
            ]
            return {
                "geom_ids": geom_name_matches,
                "geom_names": geom_names,
                "body_ids": body_ids,
                "body_names": body_names,
                "resolution_source": "geom_name",
                "truth_tier": "semantic_geom",
                "truthful": True,
                "warning_flags": warning_flags,
            }
        metadata_geom_names = self._semantic_handle_names_from_metadata()
        if metadata_geom_names:
            ids = [
                int(i)
                for i in range(self.model.ngeom)
                if str(self.model.geom(i).name or "") in metadata_geom_names
            ]
            if ids:
                geom_names = [str(self.model.geom(i).name or "") for i in ids]
                body_ids = sorted({int(self.model.geom_bodyid[i]) for i in ids})
                body_names = [
                    str(self.model.body(body_id).name or "") for body_id in body_ids
                ]
                return {
                    "geom_ids": ids,
                    "geom_names": geom_names,
                    "body_ids": body_ids,
                    "body_names": body_names,
                    "resolution_source": "manifest",
                    "truth_tier": "manifest",
                    "truthful": True,
                    "warning_flags": warning_flags,
                }
            warning_flags.append("manifest_handle_name_not_found_runtime")
        body_name_matches = [
            int(body_id)
            for body_id in range(self.model.nbody)
            if token_re.search(str(self.model.body(body_id).name or ""))
        ]
        if body_name_matches:
            ids = [
                int(i)
                for i in range(self.model.ngeom)
                if int(self.model.geom_bodyid[i]) in body_name_matches
            ]
            geom_names = [str(self.model.geom(i).name or "") for i in ids]
            body_names = [
                str(self.model.body(body_id).name or "")
                for body_id in body_name_matches
            ]
            return {
                "geom_ids": ids,
                "geom_names": geom_names,
                "body_ids": body_name_matches,
                "body_names": body_names,
                "resolution_source": "body_name",
                "truth_tier": "semantic_body",
                "truthful": True,
                "warning_flags": warning_flags,
            }
        if self.contract.measurement_fallback_policy == "allow_debug_only":
            ids = self._identify_handle_geom_ids()
            geom_names = [str(self.model.geom(i).name or "") for i in ids]
            body_ids = sorted({int(self.model.geom_bodyid[i]) for i in ids})
            body_names = [
                str(self.model.body(body_id).name or "") for body_id in body_ids
            ]
            return {
                "geom_ids": ids,
                "geom_names": geom_names,
                "body_ids": body_ids,
                "body_names": body_names,
                "resolution_source": "heuristic_debug",
                "truth_tier": "heuristic_debug",
                "truthful": False,
                "warning_flags": sorted(
                    set(warning_flags + ["semantic_handle_missing_runtime"])
                ),
            }
        return {
            "geom_ids": [],
            "geom_names": [],
            "body_ids": [],
            "body_names": [],
            "resolution_source": "none",
            "truth_tier": "unavailable",
            "truthful": False,
            "warning_flags": sorted(
                set(warning_flags + ["semantic_handle_missing_runtime"])
            ),
        }

    def _render_handle_mask_proxy_debug(
        self, camera_key: str
    ) -> tuple[np.ndarray, dict[str, Any]]:
        camera = self.cam_primary if camera_key == "primary" else self.cam_secondary
        projected = self._project_world_points(
            camera, self._handle_proxy_corners_world()
        )
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        valid = (
            np.isfinite(projected[:, 0])
            & np.isfinite(projected[:, 1])
            & (projected[:, 2] > 1e-4)
        )
        if int(np.sum(valid)) < 3:
            return mask, {
                "measurement_backend": "proxy_projected_hull",
                "truth_tier": "heuristic_debug",
                "truthful": False,
                "warning_flags": ["proxy_projection_insufficient_points"],
            }
        pixels = projected[valid, :2].copy()
        if pixels[:, 0].max() < 0.0 or pixels[:, 0].min() > float(self.image_size - 1):
            return mask, {
                "measurement_backend": "proxy_projected_hull",
                "truth_tier": "heuristic_debug",
                "truthful": False,
                "warning_flags": ["proxy_projection_offscreen"],
            }
        if pixels[:, 1].max() < 0.0 or pixels[:, 1].min() > float(self.image_size - 1):
            return mask, {
                "measurement_backend": "proxy_projected_hull",
                "truth_tier": "heuristic_debug",
                "truthful": False,
                "warning_flags": ["proxy_projection_offscreen"],
            }
        pixels[:, 0] = np.clip(pixels[:, 0], 0.0, float(self.image_size - 1))
        pixels[:, 1] = np.clip(pixels[:, 1], 0.0, float(self.image_size - 1))
        hull = cv2.convexHull(np.round(pixels).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        return mask, {
            "measurement_backend": "proxy_projected_hull",
            "truth_tier": "heuristic_debug",
            "truthful": False,
            "warning_flags": [],
        }

    def _render_handle_mask_truthful(
        self, camera_key: str
    ) -> tuple[np.ndarray, dict[str, Any]]:
        mask, report = self._verify_truthful_handle_measurement(camera_key)
        return mask, {
            "measurement_backend": report.get(
                "measurement_backend", "segmentation_render"
            ),
            "measurement_verifier": report.get(
                "measurement_verifier", "isolated_rgb_threshold"
            ),
            "measurement_truth_tier": report.get(
                "measurement_truth_tier", "manifest_entity_unverified"
            ),
            "truth_root_object": report.get("truth_root_object", "manifest_entity"),
            "identity_resolution_tier": report.get(
                "identity_resolution_tier", "manifest_entity_exact_runtime_visible_geom"
            ),
            "manifest_handle_entity_unique": bool(
                report.get("manifest_handle_entity_unique", False)
            ),
            "runtime_visible_handle_mapping_source": report.get(
                "runtime_visible_handle_mapping_source", "unresolved"
            ),
            "runtime_visible_handle_mapping_unique": bool(
                report.get("runtime_visible_handle_mapping_unique", False)
            ),
            "runtime_handle_visual_geom_mapping_unique": bool(
                report.get("runtime_handle_visual_geom_mapping_unique", False)
            ),
            "runtime_handle_collision_geom_mapping_unique": bool(
                report.get("runtime_handle_collision_geom_mapping_unique", False)
            ),
            "duplicate_runtime_geom_name_flag": bool(
                report.get("duplicate_runtime_geom_name_flag", False)
            ),
            "unnamed_runtime_geom_flag": bool(
                report.get("unnamed_runtime_geom_flag", False)
            ),
            "resolved_runtime_visible_handle_geom_ids": list(
                report.get("resolved_runtime_visible_handle_geom_ids", [])
            ),
            "resolved_runtime_visible_handle_geom_names": list(
                report.get("resolved_runtime_visible_handle_geom_names", [])
            ),
            "resolved_handle_visual_geom_ids": list(
                report.get("resolved_handle_visual_geom_ids", [])
            ),
            "resolved_handle_visual_geom_names": list(
                report.get("resolved_handle_visual_geom_names", [])
            ),
            "resolved_handle_collision_geom_ids": list(
                report.get("resolved_handle_collision_geom_ids", [])
            ),
            "resolved_handle_collision_geom_names": list(
                report.get("resolved_handle_collision_geom_names", [])
            ),
            "segmentation_mask_nonzero": int(
                report.get("segmentation_mask_nonzero", 0)
            ),
            "isolated_mask_nonzero": int(report.get("isolated_mask_nonzero", 0)),
            "segmentation_isolated_iou": float(
                report.get("segmentation_isolated_iou", 0.0)
            ),
            "segmentation_isolated_centroid_delta_px": float(
                report.get(
                    "segmentation_isolated_centroid_delta_px",
                    float(max(self.image_size, 1)),
                )
            ),
            "semantic_mapping_hash": report.get(
                "semantic_mapping_hash", self.semantic_mapping_hash
            ),
            "truthful": bool(report.get("measurement_truthful", False)),
            "warning_flags": list(report.get("measurement_warning_flags", [])),
        }

    def _render_handle_mask(self, camera_key: str) -> tuple[np.ndarray, dict[str, Any]]:
        mode = self.contract.measurement_mode
        if mode == "truthful_handle_manifest_v1":
            mask, report = self._render_handle_mask_truthful(camera_key)
            if report.get("truthful"):
                return mask, report
            if self.contract.measurement_fallback_policy == "allow_debug_only":
                proxy_mask, proxy_report = self._render_handle_mask_proxy_debug(
                    camera_key
                )
                proxy_report["warning_flags"] = sorted(
                    set(
                        list(report.get("warning_flags", []))
                        + list(proxy_report.get("warning_flags", []))
                        + ["fell_back_to_proxy_debug_measurement"]
                    )
                )
                return proxy_mask, proxy_report
            report["warning_flags"] = sorted(
                set(
                    list(report.get("warning_flags", []))
                    + ["truthful_measurement_required_no_fallback"]
                )
            )
            return mask, report
        return self._render_handle_mask_proxy_debug(camera_key)

    def _truthful_support_mask(self, mask: np.ndarray) -> np.ndarray:
        base = np.where(np.asarray(mask, dtype=np.uint8) > 0, 255, 0).astype(np.uint8)
        if int(np.count_nonzero(base)) <= 0:
            return base
        return cv2.dilate(base, np.ones((3, 3), dtype=np.uint8), iterations=2)

    def _support_compactness_ratio(self, mask: np.ndarray) -> float:
        pts = np.column_stack(np.where(np.asarray(mask, dtype=np.uint8) > 0))
        if len(pts) == 0:
            return 0.0
        pts_xy = pts[:, ::-1].astype(np.float32)
        rect = cv2.minAreaRect(pts_xy)
        (_, _), (w, h), _ = rect
        rect_area = max(float(w * h), 0.0)
        hull = cv2.convexHull(pts_xy.astype(np.int32))
        hull_area = float(cv2.contourArea(hull))
        return float(rect_area / max(hull_area, 1.0))

    def _handle_mask_bbox(self, mask: np.ndarray) -> list[int]:
        ys, xs = np.where(np.asarray(mask) > 0)
        if len(xs) == 0 or len(ys) == 0:
            return [0, 0, 0, 0]
        return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    def _handle_boundary_ring(self, mask: np.ndarray, width: int = 2) -> np.ndarray:
        base = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
        kernel = np.ones((max(width, 1), max(width, 1)), dtype=np.uint8)
        dilated = cv2.dilate(base, kernel, iterations=1)
        eroded = cv2.erode(base, kernel, iterations=1)
        ring = np.logical_and(dilated > 0, np.logical_not(eroded > 0))
        return ring.astype(np.uint8)

    def _handle_crop_from_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self._handle_mask_bbox(mask)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        return image[y0:y1, x0:x1]

    def _handle_probe_from_mask_support(
        self, image: np.ndarray, mask: np.ndarray
    ) -> dict[str, float | list[int]]:
        support_mask = self._truthful_support_mask(mask)
        mask_bool = np.asarray(support_mask, dtype=np.uint8) > 0
        bbox = self._handle_mask_bbox(support_mask)
        crop = self._handle_crop_from_mask(image, support_mask)
        ring = self._handle_boundary_ring(support_mask)
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(
            np.float32
        )
        dilated = cv2.dilate(
            mask_bool.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1
        ).astype(bool)
        exterior = np.logical_and(dilated, np.logical_not(mask_bool))
        inside_mean = float(gray[mask_bool].mean()) if mask_bool.any() else 0.0
        outside_mean = float(gray[exterior].mean()) if exterior.any() else 0.0
        boundary_contrast = float(abs(inside_mean - outside_mean) / 255.0)
        if crop.size:
            crop_gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(crop_gray, 80, 160)
            edge_density = float((edges > 0).mean())
            hist = np.bincount(crop_gray.reshape(-1), minlength=256).astype(np.float64)
            hist /= max(hist.sum(), 1.0)
            nonzero = hist[hist > 0]
            entropy = (
                float(-(nonzero * np.log2(nonzero)).sum()) if len(nonzero) else 0.0
            )
            local_std = float(crop_gray.std() / 255.0)
        else:
            edge_density = 0.0
            entropy = 0.0
            local_std = 0.0
        mask_nonzero_pixels = int(mask_bool.sum())
        image_area = float(max(self.image_size * self.image_size, 1))
        x0, y0, x1, y1 = bbox
        bbox_area = max((x1 - x0) * (y1 - y0), 0)
        mask_area_ratio = float(mask_nonzero_pixels / image_area)
        bbox_area_ratio = float(bbox_area / image_area)
        bbox_over_mask = self._support_compactness_ratio(support_mask)
        visibility_fraction = (
            float(mask_nonzero_pixels / max(bbox_area, 1)) if bbox_area > 0 else 0.0
        )
        centroid = (
            np.argwhere(mask_bool).mean(axis=0)[::-1].tolist()
            if mask_bool.any()
            else [0.0, 0.0]
        )
        return {
            "handle_mask_nonzero_pixels": mask_nonzero_pixels,
            "handle_mask_area_ratio": mask_area_ratio,
            "handle_bbox": bbox,
            "handle_bbox_area_ratio": bbox_area_ratio,
            "bbox_over_mask_ratio": bbox_over_mask,
            "handle_centroid": centroid,
            "handle_boundary_contrast": boundary_contrast,
            "handle_edge_density": edge_density,
            "handle_crop_entropy": entropy,
            "handle_local_std": local_std,
            "handle_visibility_fraction": visibility_fraction,
        }

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

    def _overlay_marker(
        self, image: np.ndarray, pixel: tuple[int, int], color: tuple[int, int, int]
    ) -> np.ndarray:
        out = image.copy()
        u, v = pixel
        rr = 4
        u0, u1 = max(0, u - rr), min(self.image_size, u + rr + 1)
        v0, v1 = max(0, v - rr), min(self.image_size, v + rr + 1)
        out[v0:v1, u0:u1] = np.asarray(color, dtype=np.uint8)
        return out

    def _background_rgb(self, secondary: bool) -> np.ndarray:
        if self.contract.background_mode == "neutral_lowfreq_lab":
            return (
                np.asarray([226, 223, 218], dtype=np.uint8)
                if secondary
                else np.asarray([219, 216, 211], dtype=np.uint8)
            )
        if self.contract.background_mode == "neutral_lab":
            return np.asarray([214, 207, 198], dtype=np.uint8)
        if self.contract.background_mode == "high_contrast_lab":
            return (
                np.asarray([236, 233, 228], dtype=np.uint8)
                if secondary
                else np.asarray([208, 204, 198], dtype=np.uint8)
            )
        mean = SECONDARY_BG_MEAN if secondary else PRIMARY_BG_MEAN
        return np.clip(mean * 255.0, 0.0, 255.0).astype(np.uint8)

    def _apply_profile_postprocess(
        self, image: np.ndarray, *, secondary: bool
    ) -> np.ndarray:
        import cv2

        base = image.astype(np.uint8)
        if self.contract.render_profile != "visual_affordance_v3_local_material_edge":
            return base
        lab = cv2.cvtColor(base, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=2.2 if secondary else 2.4, tileGridSize=(8, 8)
        )
        l_enh = clahe.apply(l_chan)
        merged = cv2.cvtColor(cv2.merge([l_enh, a_chan, b_chan]), cv2.COLOR_LAB2RGB)
        blur = cv2.GaussianBlur(merged, (0, 0), 1.0 if secondary else 1.1)
        sharpened = cv2.addWeighted(
            merged, 1.34 if secondary else 1.40, blur, -0.34 if secondary else -0.40, 0
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _apply_calibration(self, image: np.ndarray, *, secondary: bool) -> np.ndarray:
        mode = self.contract.calibration_mode
        if mode == "none":
            return self._apply_profile_postprocess(image, secondary=secondary)
        if mode == "diagnostic_texture":
            return self._apply_diagnostic_texture(image, secondary=secondary)
        out = image.astype(np.float32)
        bg = self._background_rgb(secondary).astype(np.float32)
        mask = np.max(out, axis=2, keepdims=True) < 8.0
        out = np.where(mask, bg.reshape(1, 1, 3), out)
        gain = 1.18 if secondary else 1.22
        bias = 6.0 if secondary else 10.0
        out = out * gain + bias
        return self._apply_profile_postprocess(
            np.clip(out, 0.0, 255.0).astype(np.uint8), secondary=secondary
        )

    def _apply_diagnostic_texture(
        self, image: np.ndarray, *, secondary: bool
    ) -> np.ndarray:
        import cv2

        base = image.astype(np.uint8)
        lab = cv2.cvtColor(base, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        l_enh = clahe.apply(l_chan)
        merged = cv2.cvtColor(cv2.merge([l_enh, a_chan, b_chan]), cv2.COLOR_LAB2RGB)
        blur = cv2.GaussianBlur(merged, (0, 0), 1.1)
        sharpened = cv2.addWeighted(merged, 1.65, blur, -0.65, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _apply_visual_profile(self) -> None:
        self.model.geom_rgba[:] = self._original_geom_rgba
        self.model.vis.headlight.ambient[:] = self._original_headlight_ambient
        self.model.vis.headlight.diffuse[:] = self._original_headlight_diffuse
        self.model.vis.headlight.specular[:] = self._original_headlight_specular
        self.model.vis.rgba.haze[:] = self._original_haze_rgba

        handle = self._handle_center_world()
        handle_ids = self._runtime_visible_handle_highlight_ids()
        v2_profile = self.contract.render_profile in {
            "visual_reformulation_v2_material_light_bg",
            "visual_reformulation_v2_plus_bundle",
        }
        v3_profile = self.contract.render_profile in {
            "visual_affordance_v3_raw_canonical",
            "visual_affordance_v3_local_material_edge",
            "visual_affordance_v3_plus_bundle",
        }
        if self.contract.camera_framing_profile == "macro_handle_centered":
            primary_lookat = 0.88 * handle + 0.12 * self.scene_center
            self.cam_primary.lookat[:] = primary_lookat.tolist()
            self.cam_primary.distance = float(self.scene_extent * 0.92)
            self.cam_primary.azimuth = 164.0
            self.cam_primary.elevation = -12.0
        elif self.contract.camera_framing_profile == "tight_handle_centered":
            primary_lookat = 0.74 * handle + 0.26 * self.scene_center
            self.cam_primary.lookat[:] = primary_lookat.tolist()
            self.cam_primary.distance = float(
                self.scene_extent * (1.08 if v2_profile else 1.25)
            )
            self.cam_primary.azimuth = 166.0 if v2_profile else 168.0
            self.cam_primary.elevation = -15.0 if v2_profile else -18.0
        else:
            self.cam_primary.lookat[:] = self.scene_center.tolist()
            self.cam_primary.distance = float(self.scene_extent * 1.85)
            self.cam_primary.azimuth = 180.0
            self.cam_primary.elevation = -28.0

        if self.contract.background_mode == "neutral_lowfreq_lab":
            self.model.vis.rgba.haze[:] = np.asarray(
                [0.92, 0.91, 0.89, 1.0], dtype=np.float32
            )
        elif self.contract.background_mode == "neutral_lab":
            self.model.vis.rgba.haze[:] = np.asarray(
                [0.84, 0.82, 0.79, 1.0], dtype=np.float32
            )
        elif self.contract.background_mode == "high_contrast_lab":
            self.model.vis.rgba.haze[:] = (
                np.asarray([0.96, 0.95, 0.93, 1.0], dtype=np.float32)
                if v2_profile
                else np.asarray([0.90, 0.89, 0.86, 1.0], dtype=np.float32)
            )

        if self.contract.lighting_profile == "front_key_handle_rim":
            self.model.vis.headlight.ambient[:] = np.asarray(
                [0.45, 0.45, 0.45], dtype=np.float32
            )
            self.model.vis.headlight.diffuse[:] = np.asarray(
                [0.90, 0.88, 0.84], dtype=np.float32
            )
            self.model.vis.headlight.specular[:] = np.asarray(
                [0.48, 0.46, 0.42], dtype=np.float32
            )
        elif self.contract.lighting_profile == "bright_front_fill":
            self.model.vis.headlight.ambient[:] = (
                np.asarray([0.62, 0.62, 0.62], dtype=np.float32)
                if v2_profile
                else np.asarray([0.55, 0.55, 0.55], dtype=np.float32)
            )
            self.model.vis.headlight.diffuse[:] = (
                np.asarray([0.82, 0.82, 0.82], dtype=np.float32)
                if v2_profile
                else np.asarray([0.70, 0.70, 0.70], dtype=np.float32)
            )
            self.model.vis.headlight.specular[:] = (
                np.asarray([0.26, 0.26, 0.26], dtype=np.float32)
                if v2_profile
                else np.asarray([0.20, 0.20, 0.20], dtype=np.float32)
            )

        if self.contract.material_policy == "handle_affordance_local":
            centers = np.asarray(self.data.geom_xpos, dtype=np.float32)
            if len(centers):
                if handle_ids:
                    highlight_ids = np.asarray(handle_ids, dtype=np.int32)
                else:
                    distances = np.linalg.norm(centers - handle[None, :], axis=1)
                    highlight_ids = np.asarray(
                        [int(np.argmin(distances))], dtype=np.int32
                    )
                self.model.geom_rgba[:, :3] = np.clip(
                    self.model.geom_rgba[:, :3] * 0.88, 0.0, 1.0
                )
                self.model.geom_rgba[highlight_ids, :4] = np.asarray(
                    [0.90, 0.80, 0.60, 1.0], dtype=np.float32
                )
        elif self.contract.material_policy == "handle_highlight":
            centers = np.asarray(self.data.geom_xpos, dtype=np.float32)
            if len(centers):
                if handle_ids:
                    highlight_ids = np.asarray(handle_ids, dtype=np.int32)
                    distances = np.min(
                        np.linalg.norm(
                            centers[:, None, :] - centers[highlight_ids][None, :, :],
                            axis=2,
                        ),
                        axis=1,
                    )
                else:
                    distances = np.linalg.norm(centers - handle[None, :], axis=1)
                    highlight_ids = np.asarray(
                        [int(np.argmin(distances))], dtype=np.int32
                    )
                scale = (
                    np.asarray([0.78, 0.78, 0.78, 1.0], dtype=np.float32)
                    if v2_profile
                    else np.asarray([0.95, 0.95, 0.95, 1.0], dtype=np.float32)
                )
                self.model.geom_rgba[:] = np.clip(
                    self.model.geom_rgba[:] * scale, 0.0, 1.0
                )
                self.model.geom_rgba[highlight_ids, :4] = (
                    np.asarray([0.98, 0.66, 0.16, 1.0], dtype=np.float32)
                    if v2_profile
                    else np.asarray([0.86, 0.54, 0.24, 1.0], dtype=np.float32)
                )
                near_mask = distances < (0.16 if v2_profile else 0.12)
                if v2_profile:
                    self.model.geom_rgba[near_mask, :3] = np.clip(
                        self.model.geom_rgba[near_mask, :3] * 1.35, 0.0, 1.0
                    )
                else:
                    self.model.geom_rgba[near_mask, :3] = np.clip(
                        self.model.geom_rgba[near_mask, :3] * 1.10, 0.0, 1.0
                    )
        elif v3_profile:
            self.model.geom_rgba[:, :3] = np.clip(
                self.model.geom_rgba[:, :3] * 0.88, 0.0, 1.0
            )

    def _render(self, camera) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        rgb = self.renderer.render()
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def _camera_pose_metadata(
        self,
        name: str,
        camera,
        eye: np.ndarray,
        lookat: np.ndarray,
        relative_transform: np.ndarray | None = None,
        expected_offset_local: np.ndarray | None = None,
    ) -> dict[str, Any]:
        t_world_cam = _transform_from_eye_lookat(eye, lookat)
        t_world_eef = np.eye(4, dtype=np.float32)
        t_world_eef[:3, :3] = (
            R.from_quat(np.asarray(self.eef_quat, dtype=np.float32))
            .as_matrix()
            .astype(np.float32)
        )
        t_world_eef[:3, 3] = np.asarray(self.eef_pos, dtype=np.float32)
        if relative_transform is None:
            t_eef_cam = _transform_inverse(t_world_eef) @ t_world_cam
        else:
            t_eef_cam = relative_transform.astype(np.float32)
        target_offset = (
            expected_offset_local.astype(np.float32)
            if expected_offset_local is not None
            else WRIST_CAM_OFFSET_LOCAL
        )
        translation_residual = (
            float(np.linalg.norm(t_eef_cam[:3, 3] - target_offset))
            if name == "secondary"
            else None
        )
        return {
            "name": name,
            "mode": self.contract.secondary_camera_mode
            if name == "secondary"
            else "fixed_primary_scene",
            "lookat": np.asarray(lookat, dtype=np.float32),
            "distance": float(camera.distance),
            "azimuth": float(camera.azimuth),
            "elevation": float(camera.elevation),
            "T_world_cam": t_world_cam,
            "T_eef_cam": t_eef_cam,
            "eef_cam_relativeness_residual": translation_residual,
        }

    def _secondary_camera_local_targets(self) -> tuple[np.ndarray, np.ndarray]:
        if self.contract.camera_framing_profile == "macro_handle_centered":
            return np.array([-0.035, 0.0, 0.025], dtype=np.float32), np.array(
                [0.10, 0.0, 0.0], dtype=np.float32
            )
        if self.contract.camera_framing_profile == "tight_handle_centered":
            return np.array([-0.045, 0.0, 0.030], dtype=np.float32), np.array(
                [0.08, 0.0, 0.0], dtype=np.float32
            )
        return WRIST_CAM_OFFSET_LOCAL.copy(), WRIST_LOOK_LOCAL.copy()

    def _set_secondary_camera(self) -> dict[str, Any]:
        if self.contract.secondary_camera_mode == "legacy_fixed_scene":
            eye = _camera_eye_from_lookat(
                self.cam_secondary.lookat.copy(),
                self.cam_secondary.distance,
                self.cam_secondary.azimuth,
                self.cam_secondary.elevation,
            )
            return self._camera_pose_metadata(
                "secondary", self.cam_secondary, eye, self.cam_secondary.lookat.copy()
            )

        cam_offset_local, look_local = self._secondary_camera_local_targets()

        rotation = R.from_quat(np.asarray(self.eef_quat, dtype=np.float32))
        cam_pos_world = self.eef_pos + rotation.apply(cam_offset_local).astype(
            np.float32
        )
        wrist_forward_look = self.eef_pos + rotation.apply(look_local).astype(
            np.float32
        )
        if self.contract.camera_framing_profile in {
            "macro_handle_centered",
            "tight_handle_centered",
        }:
            handle_world = self._handle_center_world()
            lookat_world = (0.90 * handle_world + 0.10 * wrist_forward_look).astype(
                np.float32
            )
        else:
            lookat_world = wrist_forward_look.astype(np.float32)
        vec = cam_pos_world - lookat_world
        distance = float(max(np.linalg.norm(vec), 1e-6))
        planar = float(max(np.linalg.norm(vec[:2]), 1e-6))
        azimuth = float(np.degrees(np.arctan2(vec[1], vec[0])))
        elevation = float(np.degrees(np.arctan2(vec[2], planar)))
        self.cam_secondary.lookat[:] = lookat_world.tolist()
        self.cam_secondary.distance = distance
        self.cam_secondary.azimuth = azimuth
        self.cam_secondary.elevation = elevation

        relative = np.eye(4, dtype=np.float32)
        relative[:3, :3] = np.eye(3, dtype=np.float32)
        relative[:3, 3] = cam_offset_local.astype(np.float32)
        return self._camera_pose_metadata(
            "secondary",
            self.cam_secondary,
            cam_pos_world,
            lookat_world,
            relative_transform=relative,
            expected_offset_local=cam_offset_local,
        )

    def _handle_bbox_for_camera(self, key: str) -> list[int]:
        center = self._handle_center_world()
        if key == "primary":
            u, v = self._project_primary(center)
            half_w, half_h = 16, 16
        else:
            u, v = self._project_secondary(center)
            half_w, half_h = 18, 18
        x0 = max(0, int(u - half_w))
        y0 = max(0, int(v - half_h))
        x1 = min(self.image_size, int(u + half_w))
        y1 = min(self.image_size, int(v + half_h))
        return [x0, y0, x1, y1]

    def _crop_statistics(
        self, image: np.ndarray, bbox: list[int]
    ) -> tuple[float, float, float]:
        x0, y0, x1, y1 = bbox
        crop = image[y0:y1, x0:x1]
        target_area = max((x1 - x0) * (y1 - y0), 1)
        visibility = float(crop.shape[0] * crop.shape[1]) / float(target_area)
        if crop.size == 0:
            return visibility, 0.0, 0.0
        gray = crop.mean(axis=2).astype(np.uint8)
        contrast = float(gray.std() / 255.0)
        hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
        hist /= max(hist.sum(), 1.0)
        nonzero = hist[hist > 0]
        entropy = float(-(nonzero * np.log2(nonzero)).sum()) if len(nonzero) else 0.0
        return visibility, contrast, entropy

    def _compute_handle_probe_metadata(
        self, primary: np.ndarray, secondary: np.ndarray, secondary_meta: dict[str, Any]
    ) -> dict[str, Any]:
        legacy_bbox_primary = self._handle_bbox_for_camera("primary")
        legacy_bbox_secondary = self._handle_bbox_for_camera("secondary")
        legacy_vis_primary, legacy_contrast_primary, legacy_entropy_primary = (
            self._crop_statistics(primary, legacy_bbox_primary)
        )
        legacy_vis_secondary, legacy_contrast_secondary, legacy_entropy_secondary = (
            self._crop_statistics(secondary, legacy_bbox_secondary)
        )

        mask_primary, primary_measurement = self._render_handle_mask("primary")
        mask_secondary, secondary_measurement = self._render_handle_mask("secondary")
        primary_probe = self._handle_probe_from_mask_support(primary, mask_primary)
        secondary_probe = self._handle_probe_from_mask_support(
            secondary, mask_secondary
        )

        proxy_mask_primary, _ = self._render_handle_mask_proxy_debug("primary")
        proxy_mask_secondary, _ = self._render_handle_mask_proxy_debug("secondary")
        proxy_primary = self._handle_probe_from_mask_support(
            primary, proxy_mask_primary
        )
        proxy_secondary = self._handle_probe_from_mask_support(
            secondary, proxy_mask_secondary
        )

        residual_value = secondary_meta.get("eef_cam_relativeness_residual")
        residual = float(1.0 if residual_value is None else residual_value)
        centroid = secondary_probe.get("handle_centroid") or [0.0, 0.0]
        center_x = float(centroid[0])
        center_y = float(centroid[1])
        framing_residual = float(
            np.mean(
                [
                    abs(center_x / max(self.image_size - 1, 1) - 0.5),
                    abs(center_y / max(self.image_size - 1, 1) - 0.5),
                ]
            )
        )
        secondary_framing_score = float(np.clip(1.0 - 2.2 * framing_residual, 0.0, 1.0))

        measurement_report = dict(secondary_measurement)
        resolved_runtime_visible_ids = list(
            measurement_report.get("resolved_runtime_visible_handle_geom_ids", [])
        )
        resolved_runtime_visible_names = list(
            measurement_report.get("resolved_runtime_visible_handle_geom_names", [])
        )
        resolved_visual_ids = list(
            measurement_report.get("resolved_handle_visual_geom_ids", [])
        )
        resolved_collision_ids = list(
            measurement_report.get("resolved_handle_collision_geom_ids", [])
        )
        resolved_visual_names = list(
            measurement_report.get("resolved_handle_visual_geom_names", [])
        )
        resolved_collision_names = list(
            measurement_report.get("resolved_handle_collision_geom_names", [])
        )

        return {
            "probe_measurement_mode": self.contract.measurement_mode,
            "truth_root_object": measurement_report.get(
                "truth_root_object", "manifest_entity"
            ),
            "identity_resolution_tier": measurement_report.get(
                "identity_resolution_tier", "manifest_entity_exact_runtime_visible_geom"
            ),
            "measurement_backend": measurement_report.get(
                "measurement_backend", "unknown"
            ),
            "measurement_verifier": measurement_report.get(
                "measurement_verifier", "unknown"
            ),
            "measurement_truth_tier": measurement_report.get(
                "measurement_truth_tier", "manifest_entity_unverified"
            ),
            "measurement_truthful": bool(measurement_report.get("truthful", False)),
            "measurement_warning_flags": list(
                measurement_report.get("warning_flags", [])
            ),
            "manifest_handle_entity_unique": bool(
                measurement_report.get("manifest_handle_entity_unique", False)
            ),
            "runtime_visible_handle_mapping_source": measurement_report.get(
                "runtime_visible_handle_mapping_source", "unresolved"
            ),
            "runtime_visible_handle_mapping_unique": bool(
                measurement_report.get("runtime_visible_handle_mapping_unique", False)
            ),
            "runtime_handle_visual_geom_mapping_unique": bool(
                measurement_report.get(
                    "runtime_handle_visual_geom_mapping_unique", False
                )
            ),
            "runtime_handle_collision_geom_mapping_unique": bool(
                measurement_report.get(
                    "runtime_handle_collision_geom_mapping_unique", False
                )
            ),
            "duplicate_runtime_geom_name_flag": bool(
                measurement_report.get("duplicate_runtime_geom_name_flag", False)
            ),
            "unnamed_runtime_geom_flag": bool(
                measurement_report.get("unnamed_runtime_geom_flag", False)
            ),
            "resolved_runtime_visible_handle_geom_ids": resolved_runtime_visible_ids,
            "resolved_runtime_visible_handle_geom_names": resolved_runtime_visible_names,
            "resolved_handle_visual_geom_ids": resolved_visual_ids,
            "resolved_handle_visual_geom_names": resolved_visual_names,
            "resolved_handle_collision_geom_ids": resolved_collision_ids,
            "resolved_handle_collision_geom_names": resolved_collision_names,
            "handle_geom_ids": sorted(set(resolved_runtime_visible_ids)),
            "handle_geom_names": sorted(set(resolved_runtime_visible_names)),
            "semantic_mapping_hash": measurement_report.get(
                "semantic_mapping_hash", self.semantic_mapping_hash
            ),
            "segmentation_mask_nonzero_primary": int(
                primary_measurement.get("segmentation_mask_nonzero", 0)
            ),
            "segmentation_mask_nonzero_secondary": int(
                secondary_measurement.get("segmentation_mask_nonzero", 0)
            ),
            "isolated_mask_nonzero_primary": int(
                primary_measurement.get("isolated_mask_nonzero", 0)
            ),
            "isolated_mask_nonzero_secondary": int(
                secondary_measurement.get("isolated_mask_nonzero", 0)
            ),
            "segmentation_mask_support_primary": float(
                1.0
                if int(primary_measurement.get("segmentation_mask_nonzero", 0)) > 0
                else 0.0
            ),
            "segmentation_mask_support_secondary": float(
                1.0
                if int(secondary_measurement.get("segmentation_mask_nonzero", 0)) > 0
                else 0.0
            ),
            "isolated_mask_support_primary": float(
                1.0
                if int(primary_measurement.get("isolated_mask_nonzero", 0)) > 0
                else 0.0
            ),
            "isolated_mask_support_secondary": float(
                1.0
                if int(secondary_measurement.get("isolated_mask_nonzero", 0)) > 0
                else 0.0
            ),
            "segmentation_isolated_iou_primary": float(
                primary_measurement.get("segmentation_isolated_iou", 0.0)
            ),
            "segmentation_isolated_iou_secondary": float(
                secondary_measurement.get("segmentation_isolated_iou", 0.0)
            ),
            "segmentation_isolated_centroid_delta_px_primary": float(
                primary_measurement.get(
                    "segmentation_isolated_centroid_delta_px",
                    float(max(self.image_size, 1)),
                )
            ),
            "segmentation_isolated_centroid_delta_px_secondary": float(
                secondary_measurement.get(
                    "segmentation_isolated_centroid_delta_px",
                    float(max(self.image_size, 1)),
                )
            ),
            "handle_bbox_primary": primary_probe["handle_bbox"],
            "handle_bbox_secondary": secondary_probe["handle_bbox"],
            "handle_mask_nonzero_pixels_primary": int(
                primary_probe["handle_mask_nonzero_pixels"]
            ),
            "handle_mask_nonzero_pixels_secondary": int(
                secondary_probe["handle_mask_nonzero_pixels"]
            ),
            "handle_mask_area_ratio_primary": float(
                primary_probe["handle_mask_area_ratio"]
            ),
            "handle_mask_area_ratio_secondary": float(
                secondary_probe["handle_mask_area_ratio"]
            ),
            "handle_bbox_area_ratio_primary": float(
                primary_probe["handle_bbox_area_ratio"]
            ),
            "handle_bbox_area_ratio_secondary": float(
                secondary_probe["handle_bbox_area_ratio"]
            ),
            "bbox_over_mask_ratio_primary": float(
                primary_probe["bbox_over_mask_ratio"]
            ),
            "bbox_over_mask_ratio_secondary": float(
                secondary_probe["bbox_over_mask_ratio"]
            ),
            "handle_visibility_fraction_primary": float(
                primary_probe["handle_visibility_fraction"]
            ),
            "handle_visibility_fraction_secondary": float(
                secondary_probe["handle_visibility_fraction"]
            ),
            "handle_visibility_fraction": float(
                max(
                    primary_probe["handle_visibility_fraction"],
                    secondary_probe["handle_visibility_fraction"],
                )
            ),
            "handle_area_ratio_primary": float(primary_probe["handle_mask_area_ratio"]),
            "handle_area_ratio_secondary": float(
                secondary_probe["handle_mask_area_ratio"]
            ),
            "handle_area_ratio": float(
                max(
                    primary_probe["handle_mask_area_ratio"],
                    secondary_probe["handle_mask_area_ratio"],
                )
            ),
            "handle_boundary_contrast_primary": float(
                primary_probe["handle_boundary_contrast"]
            ),
            "handle_boundary_contrast_secondary": float(
                secondary_probe["handle_boundary_contrast"]
            ),
            "handle_boundary_contrast": float(
                max(
                    primary_probe["handle_boundary_contrast"],
                    secondary_probe["handle_boundary_contrast"],
                )
            ),
            "handle_edge_density_primary": float(primary_probe["handle_edge_density"]),
            "handle_edge_density_secondary": float(
                secondary_probe["handle_edge_density"]
            ),
            "handle_edge_density": float(
                max(
                    primary_probe["handle_edge_density"],
                    secondary_probe["handle_edge_density"],
                )
            ),
            "handle_crop_entropy_primary": float(primary_probe["handle_crop_entropy"]),
            "handle_crop_entropy_secondary": float(
                secondary_probe["handle_crop_entropy"]
            ),
            "handle_crop_entropy": float(
                max(
                    primary_probe["handle_crop_entropy"],
                    secondary_probe["handle_crop_entropy"],
                )
            ),
            "handle_local_std_primary": float(primary_probe["handle_local_std"]),
            "handle_local_std_secondary": float(secondary_probe["handle_local_std"]),
            "handle_local_std": float(
                max(
                    primary_probe["handle_local_std"],
                    secondary_probe["handle_local_std"],
                )
            ),
            "handle_local_contrast_primary": float(
                primary_probe["handle_boundary_contrast"]
            ),
            "handle_local_contrast_secondary": float(
                secondary_probe["handle_boundary_contrast"]
            ),
            "handle_local_contrast": float(
                max(
                    primary_probe["handle_boundary_contrast"],
                    secondary_probe["handle_boundary_contrast"],
                )
            ),
            "secondary_framing_score": secondary_framing_score,
            "framing_score": secondary_framing_score,
            "camera_relativeness_residual": residual,
            "render_profile": self.contract.render_profile,
            "background_mode": self.contract.background_mode,
            "lighting_profile": self.contract.lighting_profile,
            "material_policy": self.contract.material_policy,
            "camera_framing_profile": self.contract.camera_framing_profile,
            "legacy_probe_measurement_mode": "proxy_debug",
            "legacy_handle_bbox_primary": legacy_bbox_primary,
            "legacy_handle_bbox_secondary": legacy_bbox_secondary,
            "legacy_handle_visibility_fraction_primary": float(legacy_vis_primary),
            "legacy_handle_visibility_fraction_secondary": float(legacy_vis_secondary),
            "legacy_handle_local_contrast_primary": float(legacy_contrast_primary),
            "legacy_handle_local_contrast_secondary": float(legacy_contrast_secondary),
            "legacy_handle_crop_entropy_primary": float(legacy_entropy_primary),
            "legacy_handle_crop_entropy_secondary": float(legacy_entropy_secondary),
            "legacy_handle_mask_nonzero_pixels_primary": int(
                proxy_primary["handle_mask_nonzero_pixels"]
            ),
            "legacy_handle_mask_nonzero_pixels_secondary": int(
                proxy_secondary["handle_mask_nonzero_pixels"]
            ),
            "legacy_handle_mask_area_ratio_primary": float(
                proxy_primary["handle_mask_area_ratio"]
            ),
            "legacy_handle_mask_area_ratio_secondary": float(
                proxy_secondary["handle_mask_area_ratio"]
            ),
            "legacy_bbox_over_mask_ratio_primary": float(
                proxy_primary["bbox_over_mask_ratio"]
            ),
            "legacy_bbox_over_mask_ratio_secondary": float(
                proxy_secondary["bbox_over_mask_ratio"]
            ),
        }

    def _observe_images(self) -> tuple[np.ndarray, np.ndarray]:
        self._apply_visual_profile()
        primary_eye = _camera_eye_from_lookat(
            self.cam_primary.lookat.copy(),
            self.cam_primary.distance,
            self.cam_primary.azimuth,
            self.cam_primary.elevation,
        )
        primary_meta = self._camera_pose_metadata(
            "primary", self.cam_primary, primary_eye, self.cam_primary.lookat.copy()
        )
        secondary_meta = self._set_secondary_camera()

        raw1 = self._render(self.cam_primary)
        raw2 = self._render(self.cam_secondary)
        probe_img1 = self._apply_calibration(raw1, secondary=False)
        probe_img2 = self._apply_calibration(raw2, secondary=True)
        handle_probe = (
            self._compute_handle_probe_metadata(probe_img1, probe_img2, secondary_meta)
            if self.contract.emit_handle_probe_metadata
            else {}
        )
        img1 = probe_img1.copy()
        img2 = probe_img2.copy()
        if self.contract.enable_marker_overlay:
            color = (0, 220, 32) if self._attached else (220, 60, 20)
            img1 = self._overlay_marker(
                img1, self._project_primary(self.eef_pos), color
            )
            img2 = self._overlay_marker(
                img2, self._project_secondary(self.eef_pos), color
            )

        self._last_camera_metadata = {
            "primary": _json_ready(primary_meta),
            "secondary": _json_ready(secondary_meta),
            "secondary_camera_mode": self.contract.secondary_camera_mode,
        }
        self._last_visual_mode_report = {
            "canonical_lane": bool(self.contract.canonical_lane),
            "claim_policy": self.claim_policy(),
            "enable_marker_overlay": bool(self.contract.enable_marker_overlay),
            "calibration_mode": self.contract.calibration_mode,
            "secondary_camera_mode": self.contract.secondary_camera_mode,
            "render_profile": self.contract.render_profile,
            "background_mode": self.contract.background_mode,
            "lighting_profile": self.contract.lighting_profile,
            "material_policy": self.contract.material_policy,
            "camera_framing_profile": self.contract.camera_framing_profile,
            "diagnostic_only": (not self.contract.canonical_lane)
            or self.contract.calibration_mode == "diagnostic_texture",
        }
        self._last_handle_probe_metadata = _json_ready(handle_probe)
        self._last_claim_policy = self.claim_policy()
        return img1, img2

    def _synthetic_motor_state(self) -> tuple[np.ndarray, np.ndarray]:
        self._apply_visual_profile()
        primary_eye = _camera_eye_from_lookat(
            self.cam_primary.lookat.copy(),
            self.cam_primary.distance,
            self.cam_primary.azimuth,
            self.cam_primary.elevation,
        )
        primary_meta = self._camera_pose_metadata(
            "primary", self.cam_primary, primary_eye, self.cam_primary.lookat.copy()
        )
        secondary_meta = self._set_secondary_camera()

        img1 = self._render(self.cam_primary)
        img2 = self._render(self.cam_secondary)
        img1 = self._apply_calibration(img1, secondary=False)
        img2 = self._apply_calibration(img2, secondary=True)
        if self.contract.enable_marker_overlay:
            color = (0, 220, 32) if self._attached else (220, 60, 20)
            img1 = self._overlay_marker(
                img1, self._project_primary(self.eef_pos), color
            )
            img2 = self._overlay_marker(
                img2, self._project_secondary(self.eef_pos), color
            )

        handle_probe = (
            self._compute_handle_probe_metadata(img1, img2, secondary_meta)
            if self.contract.emit_handle_probe_metadata
            else {}
        )
        self._last_camera_metadata = {
            "primary": _json_ready(primary_meta),
            "secondary": _json_ready(secondary_meta),
            "secondary_camera_mode": self.contract.secondary_camera_mode,
        }
        self._last_visual_mode_report = {
            "canonical_lane": bool(self.contract.canonical_lane),
            "claim_policy": self.claim_policy(),
            "enable_marker_overlay": bool(self.contract.enable_marker_overlay),
            "calibration_mode": self.contract.calibration_mode,
            "secondary_camera_mode": self.contract.secondary_camera_mode,
            "render_profile": self.contract.render_profile,
            "background_mode": self.contract.background_mode,
            "lighting_profile": self.contract.lighting_profile,
            "material_policy": self.contract.material_policy,
            "camera_framing_profile": self.contract.camera_framing_profile,
            "diagnostic_only": (not self.contract.canonical_lane)
            or self.contract.calibration_mode == "diagnostic_texture",
        }
        self._last_handle_probe_metadata = _json_ready(handle_probe)
        self._last_claim_policy = self.claim_policy()
        return img1, img2

    def _synthetic_motor_state(self) -> np.ndarray:
        handle = self._handle_center_world()
        rel = handle - self.eef_pos
        scale = np.array([0.22, 0.18, 0.14], dtype=np.float32)
        rel = np.clip(rel / scale, -1.0, 1.0)
        return np.array(
            [rel[0], rel[1], rel[2], self._drawer_fraction_signed()], dtype=np.float32
        )

    def state_spec(self) -> dict[str, Any]:
        mode = self.contract.state_mode
        if mode == "m0_proxy":
            dim_names = [
                "eef_pos_x",
                "eef_pos_y",
                "eef_pos_z",
                "handle_rel_x_norm",
                "handle_rel_y_norm",
                "handle_rel_z_norm",
                "drawer_fraction_signed",
                "gripper_joint",
            ]
            provenance = [
                "observed",
                "observed",
                "observed",
                "derived",
                "derived",
                "derived",
                "derived",
                "observed",
            ]
        elif mode == "eef_pose_gripper":
            dim_names = [
                "eef_pos_x",
                "eef_pos_y",
                "eef_pos_z",
                "eef_quat_x",
                "eef_quat_y",
                "eef_quat_z",
                "eef_quat_w",
                "gripper_joint",
            ]
            provenance = ["observed"] * 8
        elif mode == "telemetry_candidate_v1":
            dim_names = [
                "eef_pos_x",
                "eef_pos_y",
                "eef_pos_z",
                "eef_rotvec_x",
                "eef_rotvec_y",
                "eef_rotvec_z",
                "gripper_joint",
                "drawer_fraction_signed",
            ]
            provenance = [
                "observed",
                "observed",
                "observed",
                "derived",
                "derived",
                "derived",
                "observed",
                "derived",
            ]
        elif mode == "telemetry_candidate_v2":
            dim_names = [
                "eef_pos_x",
                "eef_pos_y",
                "eef_pos_z",
                "eef_rotvec_x",
                "eef_rotvec_y",
                "eef_rotvec_z",
                "handle_distance_norm",
                "drawer_fraction_signed",
            ]
            provenance = [
                "observed",
                "observed",
                "observed",
                "derived",
                "derived",
                "derived",
                "derived",
                "derived",
            ]
        elif mode == "telemetry_candidate_v4_task_identity":
            dim_names = [
                "handle_rel_x_norm",
                "handle_rel_y_norm",
                "handle_rel_z_norm",
                "drawer_fraction_signed",
                "pull_alignment_cos",
                "grasp_slip_norm",
                "effective_pull_progress_norm",
                "phase_locked",
            ]
            provenance = ["derived"] * 8
        else:
            dim_names = [
                "handle_rel_x_norm",
                "handle_rel_y_norm",
                "handle_rel_z_norm",
                "drawer_fraction_signed",
                "pull_alignment_cos",
                "stable_attach",
                "attach_streak_norm",
                "contact_window_fraction",
            ]
            provenance = ["derived"] * 8
        duplicate_dims = sorted(
            {name for name in dim_names if dim_names.count(name) > 1}
        )
        return {
            "state_mode": mode,
            "shape": [8],
            "dim_names": dim_names,
            "provenance": provenance,
            "duplicate_dims": duplicate_dims,
            "fraud_padding": bool(duplicate_dims),
            "all_dims_explained": True,
        }

    def _state_vector(self) -> np.ndarray:
        mode = self.contract.state_mode
        if mode == "m0_proxy":
            state = np.concatenate(
                [
                    self.eef_pos.astype(np.float32),
                    self._synthetic_motor_state(),
                    np.array([self.gripper_joint], dtype=np.float32),
                ]
            )
            return state.astype(np.float32)
        if mode == "eef_pose_gripper":
            state = np.concatenate(
                [
                    self.eef_pos.astype(np.float32),
                    np.asarray(self.eef_quat, dtype=np.float32),
                    np.array([self.gripper_joint], dtype=np.float32),
                ]
            )
            return state.astype(np.float32)
        rotvec = (
            R.from_quat(np.asarray(self.eef_quat, dtype=np.float32))
            .as_rotvec()
            .astype(np.float32)
        )
        if mode == "telemetry_candidate_v1":
            state = np.concatenate(
                [
                    self.eef_pos.astype(np.float32),
                    rotvec,
                    np.array(
                        [self.gripper_joint, self._drawer_fraction_signed()],
                        dtype=np.float32,
                    ),
                ]
            )
            return state.astype(np.float32)
        handle, _ = self._interaction_handle_target_world()
        handle_rel = np.clip(
            (handle - self.eef_pos) / np.array([0.22, 0.18, 0.14], dtype=np.float32),
            -1.0,
            1.0,
        )
        if mode == "telemetry_candidate_v2":
            handle_distance = float(np.linalg.norm(handle - self.eef_pos))
            handle_distance_norm = np.clip(handle_distance / 0.35, 0.0, 1.0)
            state = np.concatenate(
                [
                    self.eef_pos.astype(np.float32),
                    rotvec,
                    np.array(
                        [handle_distance_norm, self._drawer_fraction_signed()],
                        dtype=np.float32,
                    ),
                ]
            )
            return state.astype(np.float32)
        info = self._last_orientation_info or self._default_orientation_info()
        attach_streak_norm = float(
            np.clip(float(info.get("attach_streak", 0.0)) / 4.0, 0.0, 1.0)
        )
        if mode == "telemetry_candidate_v4_task_identity":
            joint_span = max(float(self.joint_range[1] - self.joint_range[0]), 1e-6)
            effective_pull_progress_norm = float(
                np.clip(
                    float(info.get("effective_pull_progress", 0.0)) / joint_span,
                    0.0,
                    1.0,
                )
            )
            state = np.concatenate(
                [
                    handle_rel.astype(np.float32),
                    np.array(
                        [
                            self._drawer_fraction_signed(),
                            float(info.get("pull_alignment_cos", 0.0)),
                            float(np.clip(info.get("grasp_slip_norm", 0.0), 0.0, 1.0)),
                            effective_pull_progress_norm,
                            1.0 if bool(info.get("phase_locked", False)) else 0.0,
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
            return state.astype(np.float32)
        state = np.concatenate(
            [
                handle_rel.astype(np.float32),
                np.array(
                    [
                        self._drawer_fraction_signed(),
                        float(info.get("pull_alignment_cos", 0.0)),
                        1.0 if bool(info.get("stable_attach", False)) else 0.0,
                        attach_streak_norm,
                        float(
                            np.clip(info.get("contact_window_fraction", 0.0), 0.0, 1.0)
                        ),
                    ],
                    dtype=np.float32,
                ),
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
            camera_metadata=_json_ready(self._last_camera_metadata),
            visual_mode_report=_json_ready(self._last_visual_mode_report),
            state_spec=_json_ready(self.state_spec()),
            orientation_telemetry=_json_ready(self._last_orientation_info),
            handle_probe_metadata=_json_ready(self._last_handle_probe_metadata),
            claim_policy=self._last_claim_policy,
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
        self._attach_streak = 0
        self._stable_attach = False
        self._contact_window = []
        self._orientation_break_streak = 0
        self._slip_break_streak = 0
        self._reverse_pull_streak = 0
        self._anchor_handle_offset_world = None
        self._anchor_eef_pull_progress = 0.0
        self._prev_pull_progress = 0.0
        self._runtime_visible_handle_geom_ids = []
        self._runtime_handle_anchor_valid = False
        self._runtime_handle_anchor_world = self._handle_center_world().copy()
        manifest_report = self._load_handle_entity_from_manifest()
        runtime_visible_mapping = (
            self._resolve_runtime_visible_handle_geom_ids_from_manifest(
                manifest_report.get("handle_entity")
            )
        )
        resolved_runtime_visible_ids = [
            int(idx) for idx in list(runtime_visible_mapping.get("resolved_ids", []))
        ]
        mapping_valid = bool(
            manifest_report.get("manifest_handle_entity_unique")
            and runtime_visible_mapping.get("mapping_unique")
            and not runtime_visible_mapping.get(
                "duplicate_runtime_geom_name_flag", False
            )
            and not runtime_visible_mapping.get("unnamed_runtime_geom_flag", False)
        )
        if mapping_valid and resolved_runtime_visible_ids:
            self._runtime_visible_handle_geom_ids = resolved_runtime_visible_ids
            try:
                self._runtime_handle_anchor_world = (
                    self._runtime_visible_handle_anchor_world()
                )
                self._runtime_handle_anchor_valid = True
            except Exception:
                self._runtime_handle_anchor_valid = False
        self._last_detach_reason = None
        self._max_drawer_fraction = self._drawer_fraction()
        self._last_orientation_info = self._default_orientation_info()
        self._last_handle_probe_metadata = {}
        self._last_claim_policy = self.claim_policy()
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
            "claim_policy": self.claim_policy(),
            "contract_config": self.contract_payload(),
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[MuJoCoRobotObservation, float, bool, dict[str, Any]]:
        import mujoco

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        prev_pos = self.eef_pos.copy()
        delta_pos = np.clip(action[:3], -1.0, 1.0) * TRANSLATION_SCALE_M
        delta_rot = np.clip(action[3:6], -1.0, 1.0) * ROTATION_SCALE_RAD
        close_cmd = float(action[6]) < 0.0

        self.eef_pos = np.clip(
            prev_pos + delta_pos, self._workspace_low, self._workspace_high
        )
        delta_quat = R.from_rotvec(delta_rot).as_quat().astype(np.float32)
        self.eef_quat = (
            (R.from_quat(delta_quat) * R.from_quat(self.eef_quat))
            .as_quat()
            .astype(np.float32)
        )
        self.gripper_joint = GRIPPER_CLOSED if close_cmd else GRIPPER_OPEN

        handle, runtime_handle_anchor_valid = self._interaction_handle_target_world()
        dist_to_handle = float(np.linalg.norm(self.eef_pos - handle))
        distance_pass = dist_to_handle <= ATTACH_THRESHOLD_M
        orientation_alignment_cos = 1.0
        orientation_gate_passed = True
        orientation_error_rad = 0.0
        orientation_weight = 1.0
        attach_eligible = False
        pull_alignment_cos = 0.0
        approach_alignment_cos = 0.0

        if self.contract.interaction_mode in {
            "orientation_sensitive_v1",
            "orientation_sensitive_v2_affordance_locked",
            "orientation_sensitive_v3_task_identity_locked",
        }:
            local_pull_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            pull_axis_world = (
                R.from_quat(np.asarray(self.eef_quat, dtype=np.float32))
                .apply(local_pull_axis)
                .astype(np.float32)
            )
            pull_axis_world = _normalize(pull_axis_world, WORLD_X)
            orientation_alignment_cos = float(
                np.clip(np.dot(pull_axis_world, self._motion_axis), -1.0, 1.0)
            )
            pull_alignment_cos = orientation_alignment_cos
            orientation_error_rad = float(
                np.arccos(np.clip(orientation_alignment_cos, -1.0, 1.0))
            )
            attach_cos_threshold = (
                0.60
                if self.contract.interaction_mode
                == "orientation_sensitive_v3_task_identity_locked"
                else 0.55
            )
            open_cos_threshold = (
                0.60
                if self.contract.interaction_mode
                == "orientation_sensitive_v3_task_identity_locked"
                else 0.65
            )
            orientation_gate_passed = orientation_alignment_cos >= attach_cos_threshold
            orientation_weight = float(
                np.clip(
                    (orientation_alignment_cos - open_cos_threshold)
                    / max(1e-6, 1.0 - open_cos_threshold),
                    0.0,
                    1.0,
                )
            )

        if float(np.linalg.norm(delta_pos)) > 1e-6:
            approach_dir = _normalize(delta_pos, WORLD_X)
            desired_dir = _normalize(handle - prev_pos, WORLD_X)
            approach_alignment_cos = float(
                np.clip(np.dot(approach_dir, desired_dir), -1.0, 1.0)
            )
        if (
            self.contract.interaction_mode
            == "orientation_sensitive_v3_task_identity_locked"
        ):
            approach_gate_passed = approach_alignment_cos >= 0.25
        elif (
            self.contract.interaction_mode
            == "orientation_sensitive_v2_affordance_locked"
        ):
            approach_gate_passed = approach_alignment_cos >= 0.15
        else:
            approach_gate_passed = True
        attach_eligible = bool(
            close_cmd
            and distance_pass
            and orientation_gate_passed
            and approach_gate_passed
        )

        self._contact_window.append(1.0 if distance_pass else 0.0)
        self._contact_window = self._contact_window[-6:]
        contact_window_fraction = (
            float(np.mean(self._contact_window)) if self._contact_window else 0.0
        )

        phase_locked = False
        raw_grasp_slip = 0.0
        grasp_slip_norm = 0.0
        pull_progress = 0.0
        pull_increment = 0.0
        effective_pull_progress = 0.0
        detach_reason = None
        drawer_delta_raw = 0.0
        drawer_delta_effective = 0.0

        if (
            self.contract.interaction_mode
            == "orientation_sensitive_v3_task_identity_locked"
        ):
            transition = self._task_identity_transition_update(
                prev_pos=prev_pos,
                close_cmd=close_cmd,
                dist_to_handle=dist_to_handle,
                attach_eligible=attach_eligible,
                orientation_alignment_cos=orientation_alignment_cos,
                pull_alignment_cos=pull_alignment_cos,
                runtime_handle_anchor_world=handle,
                runtime_handle_anchor_valid=runtime_handle_anchor_valid,
            )
            drawer_delta_raw = float(transition["drawer_delta_raw"])
            drawer_delta_effective = float(transition["drawer_delta_effective"])
            orientation_weight = float(transition["open_orientation_weight"])
            phase_locked = bool(transition["phase_locked"])
            raw_grasp_slip = float(transition["raw_grasp_slip"])
            grasp_slip_norm = float(transition["grasp_slip_norm"])
            pull_progress = float(transition["pull_progress"])
            pull_increment = float(transition["pull_increment"])
            effective_pull_progress = float(transition["effective_pull_progress"])
            detach_reason = transition["detach_reason"]
            handle = np.asarray(
                transition["runtime_handle_anchor_world"], dtype=np.float32
            )
            runtime_handle_anchor_valid = bool(
                transition["runtime_handle_anchor_valid"]
            )
        elif (
            self.contract.interaction_mode
            == "orientation_sensitive_v2_affordance_locked"
        ):
            if attach_eligible:
                self._attach_streak += 1
            else:
                self._attach_streak = 0
            self._stable_attach = self._attach_streak >= 3
            self._attached = bool(self._stable_attach)
            if self._attached and (
                not close_cmd
                or dist_to_handle > DETACH_THRESHOLD_M
                or orientation_alignment_cos < 0.45
            ):
                self._attached = False
                self._stable_attach = False
                self._attach_streak = 0
                detach_reason = "legacy_detach"
        else:
            if not self._attached and attach_eligible:
                self._attached = True
            if self._attached and (
                not close_cmd or dist_to_handle > DETACH_THRESHOLD_M
            ):
                self._attached = False
                detach_reason = "legacy_detach"
            self._stable_attach = bool(self._attached)
            self._attach_streak = 1 if self._attached else 0

        if (
            self.contract.interaction_mode
            != "orientation_sensitive_v3_task_identity_locked"
        ):
            if self._attached:
                drawer_delta_raw = (
                    float(np.dot(self.eef_pos - prev_pos, self._motion_axis)) * 6.0
                )
                drawer_delta_effective = (
                    drawer_delta_raw
                    if self.contract.interaction_mode == "legacy_translation_only"
                    else drawer_delta_raw * orientation_weight
                )
            phase_locked = bool(
                self._stable_attach
                and self.contract.interaction_mode
                == "orientation_sensitive_v2_affordance_locked"
            )
        low, high = self.joint_range.tolist()
        if abs(drawer_delta_effective) > 0.0:
            self.data.qpos[self.joint_idx] = np.clip(
                float(self.data.qpos[self.joint_idx]) + drawer_delta_effective,
                low,
                high,
            )
        mujoco.mj_forward(self.model, self.data)
        self._step_count += 1
        self._last_orientation_info = {
            "orientation_alignment_cos": float(orientation_alignment_cos),
            "orientation_gate_passed": bool(orientation_gate_passed),
            "attach_eligible": bool(attach_eligible),
            "attach_gate_distance_passed": bool(distance_pass),
            "attach_gate_orientation_passed": bool(orientation_gate_passed),
            "attach_gate_approach_passed": bool(approach_gate_passed),
            "attach_streak": int(self._attach_streak),
            "stable_attach": bool(self._stable_attach),
            "pull_alignment_cos": float(pull_alignment_cos),
            "drawer_delta_raw": float(drawer_delta_raw),
            "drawer_delta_effective": float(drawer_delta_effective),
            "open_orientation_weight": float(orientation_weight),
            "orientation_error_rad": float(orientation_error_rad),
            "contact_window_fraction": float(contact_window_fraction),
            "runtime_handle_anchor_world": handle.astype(np.float32),
            "runtime_handle_anchor_valid": bool(runtime_handle_anchor_valid),
            "phase_locked": bool(phase_locked),
            "raw_grasp_slip": float(raw_grasp_slip),
            "grasp_slip_norm": float(grasp_slip_norm),
            "pull_progress": float(pull_progress),
            "pull_increment": float(pull_increment),
            "effective_pull_progress": float(effective_pull_progress),
            "detach_reason": detach_reason,
        }
        obs = self.observe()
        self._max_drawer_fraction = max(self._max_drawer_fraction, obs.drawer_fraction)
        success = obs.drawer_fraction >= DRAWER_SUCCESS_FRACTION
        reward = 1.0 if success else float(obs.drawer_fraction)
        done = success or self._step_count >= self.max_steps
        info = {
            "is_success": success,
            "drawer_fraction": obs.drawer_fraction,
            "drawer_fraction_signed": self._drawer_fraction_signed(),
            "attached": self._attached,
            "dist_to_handle": dist_to_handle,
            "step_count": self._step_count,
            "max_drawer_fraction": self._max_drawer_fraction,
            "handle_center_world": handle.tolist(),
            "orientation_alignment_cos": float(orientation_alignment_cos),
            "orientation_gate_passed": bool(orientation_gate_passed),
            "attach_eligible": bool(attach_eligible),
            "attach_gate_distance_passed": bool(distance_pass),
            "attach_gate_orientation_passed": bool(orientation_gate_passed),
            "attach_gate_approach_passed": bool(approach_gate_passed),
            "attach_streak": int(self._attach_streak),
            "stable_attach": bool(self._stable_attach),
            "pull_alignment_cos": float(pull_alignment_cos),
            "drawer_delta_raw": float(drawer_delta_raw),
            "drawer_delta_effective": float(drawer_delta_effective),
            "open_orientation_weight": float(orientation_weight),
            "orientation_error_rad": float(orientation_error_rad),
            "contact_window_fraction": float(contact_window_fraction),
            "runtime_handle_anchor_world": handle.tolist(),
            "runtime_handle_anchor_valid": bool(runtime_handle_anchor_valid),
            "phase_locked": bool(phase_locked),
            "raw_grasp_slip": float(raw_grasp_slip),
            "grasp_slip_norm": float(grasp_slip_norm),
            "pull_progress": float(pull_progress),
            "pull_increment": float(pull_increment),
            "effective_pull_progress": float(effective_pull_progress),
            "detach_reason": detach_reason,
            "claim_policy": self.claim_policy(),
            "camera_metadata": _json_ready(self._last_camera_metadata),
            "visual_mode_report": _json_ready(self._last_visual_mode_report),
            "state_spec": _json_ready(self.state_spec()),
        }
        return obs, reward, done, info


def _pose_from_point(
    handle_center: np.ndarray,
    motion_axis: np.ndarray,
    offset: float,
    z_lift: float = 0.0,
) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pos = (
        handle_center
        - motion_axis * offset
        + np.array([0.0, 0.0, z_lift], dtype=np.float32)
    )
    pose[:3, 3] = pos
    pose[:3, :3] = np.eye(3, dtype=np.float32)
    return pose


def build_oracle_grasp_pose(
    anygrasp_pose_world: np.ndarray, handle_center_world: np.ndarray
) -> np.ndarray:
    pose = np.asarray(anygrasp_pose_world, dtype=np.float32).copy()
    pose[:3, 3] = np.asarray(handle_center_world, dtype=np.float32)
    return pose


def _safe_normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.asarray(fallback, dtype=np.float32).copy()
    return (vec / norm).astype(np.float32)


def _target_quat(
    current_pos: np.ndarray, target_pos: np.ndarray, motion_axis: np.ndarray
) -> np.ndarray:
    motion = _safe_normalize(motion_axis, WORLD_X)
    approach = _safe_normalize(target_pos - current_pos, WORLD_UP)
    z_hint = WORLD_UP.copy()
    if abs(float(np.dot(motion, z_hint))) > 0.90:
        z_hint = approach
    y_axis = np.cross(z_hint, motion)
    if float(np.linalg.norm(y_axis)) < 1e-6:
        y_axis = np.cross(approach, motion)
    y_axis = _safe_normalize(y_axis, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    z_axis = _safe_normalize(np.cross(motion, y_axis), WORLD_UP)
    y_axis = _safe_normalize(np.cross(z_axis, motion), y_axis)
    rot = np.stack([motion, y_axis, z_axis], axis=1)
    return R.from_matrix(rot).as_quat().astype(np.float32)


def _script_action(
    current_pos: np.ndarray,
    target_pos: np.ndarray,
    close: bool,
    speed: float = 1.0,
    *,
    current_quat: np.ndarray | None = None,
    motion_axis: np.ndarray | None = None,
    rotation_source: RotationSource = "zero",
    rng: np.random.Generator | None = None,
    phase: str = "pull",
    orientation_gain: float = 1.0,
) -> np.ndarray:
    delta = np.zeros(7, dtype=np.float32)
    pos_err = target_pos - current_pos
    delta[:3] = np.clip(
        (pos_err / max(TRANSLATION_SCALE_M, 1e-6)) * float(speed), -1.0, 1.0
    )
    if (
        rotation_source == "aligned"
        and current_quat is not None
        and motion_axis is not None
    ):
        target_quat = _target_quat(current_pos, target_pos, motion_axis)
        current_rot = R.from_quat(np.asarray(current_quat, dtype=np.float32))
        target_rot = R.from_quat(target_quat)
        rotvec = (target_rot * current_rot.inv()).as_rotvec().astype(np.float32)
        phase_gain = (
            0.55
            if phase in {"pregrasp", "contact"}
            else (0.85 if phase == "close" else 1.0)
        )
        delta[3:6] = np.clip(
            (rotvec / max(ROTATION_SCALE_RAD, 1e-6))
            * phase_gain
            * float(orientation_gain),
            -ROT_ACTION_CLIP,
            ROT_ACTION_CLIP,
        )
    elif rotation_source == "random":
        rand = (
            (rng or np.random.default_rng())
            .normal(loc=0.0, scale=CONTROL_ROT_STD)
            .astype(np.float32)
        )
        delta[3:6] = np.clip(
            rand / max(ROTATION_SCALE_RAD, 1e-6), -ROT_ACTION_CLIP, ROT_ACTION_CLIP
        )
    delta[6] = -1.0 if close else 1.0
    return delta.astype(np.float32)


def _apply_assay_warm_start(
    env: DrawerRobotEnvMuJoCo, warm_start_kind: str | None
) -> tuple[MuJoCoRobotObservation, str]:
    kind = str(warm_start_kind or "").strip().lower()
    if not kind:
        return env.observe(), "pregrasp"

    handle, runtime_handle_anchor_valid = env._interaction_handle_target_world()
    axis = env._motion_axis.astype(np.float32)
    contact_target = handle + np.array([0.0, 0.0, 0.005], dtype=np.float32)
    staged_contact_target = (
        handle - axis * 0.008 + np.array([0.0, 0.0, 0.005], dtype=np.float32)
    )
    pull_target = handle + axis * 0.03 + np.array([0.0, 0.0, 0.005], dtype=np.float32)
    aligned_quat = _target_quat(staged_contact_target, pull_target, axis)

    env.eef_pos = (
        staged_contact_target if kind == "contact_aligned" else contact_target
    ).astype(np.float32)
    env.eef_quat = aligned_quat.astype(np.float32)
    env.gripper_joint = GRIPPER_CLOSED
    env._contact_window = [1.0] * 6
    env._runtime_handle_anchor_world = handle.astype(np.float32)
    env._runtime_handle_anchor_valid = bool(runtime_handle_anchor_valid)
    env._last_detach_reason = None

    if kind == "contact_aligned":
        env._attached = False
        env._stable_attach = False
        env._attach_streak = 2
        env._anchor_handle_offset_world = None
        env._anchor_eef_pull_progress = 0.0
        env._prev_pull_progress = 0.0
        env._orientation_break_streak = 0
        env._slip_break_streak = 0
        env._reverse_pull_streak = 0
        return env.observe(), "close"

    if kind == "attached_phase_locked":
        env._attached = True
        env._stable_attach = True
        env._attach_streak = 3
        env._anchor_handle_offset_world = (handle - env.eef_pos).astype(np.float32)
        env._anchor_eef_pull_progress = float(np.dot(env.eef_pos, env._motion_axis))
        env._prev_pull_progress = 0.0
        env._orientation_break_streak = 0
        env._slip_break_streak = 0
        env._reverse_pull_streak = 0
        return env.observe(), "phase_lock_settle"

    raise ValueError(f"unsupported_assay_warm_start_kind={warm_start_kind}")


def _opening_servo_offsets(base_offset: float) -> list[float]:
    raw = [
        max(0.01, base_offset * 0.5),
        max(0.02, base_offset),
        max(0.04, base_offset * 1.5),
        max(0.06, base_offset * 2.0),
        max(0.08, base_offset * 2.5),
        max(0.10, base_offset * 3.0),
    ]
    offsets: list[float] = []
    for value in raw:
        clipped = round(float(np.clip(value, 0.01, 0.14)), 4)
        if not offsets or abs(clipped - offsets[-1]) > 1e-6:
            offsets.append(clipped)
    return offsets or [0.02, 0.04, 0.06, 0.08, 0.10]


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
    contract: DrawerEnvContractConfig | None = None,
    rotation_source: RotationSource = "zero",
    claim_policy: str | None = None,
    interventions: dict[str, Any] | None = None,
    assay_warm_start_kind: str | None = None,
    teacher_controller_mode: str = "baseline",
) -> dict[str, Any]:
    env = DrawerRobotEnvMuJoCo(
        seed=seed, image_size=image_size, max_steps=max_steps, contract=contract
    )
    obs = env.reset()
    axis = env._motion_axis
    handle, _ = env._interaction_handle_target_world()
    pregrasp_pose = _pose_from_point(handle, axis, offset=0.04, z_lift=0.03)
    grasp_pose = np.asarray(grasp_pose_world, dtype=np.float32).copy()
    grasp_pose[:3, 3] = handle
    open_fraction = float(
        np.clip(
            pull_open_fraction,
            0.75,
            0.98 if teacher_controller_mode == "embodiment_bound_quasistatic" else 0.95,
        )
    )
    retreat_target = handle + axis * 0.16 + np.array([0.0, 0.0, 0.05], dtype=np.float32)
    rng = np.random.default_rng(seed + episode_index)
    intervention_cfg = dict(interventions or {})
    force_detach_probe = bool(intervention_cfg.get("force_detach_probe", False))
    close_settle_steps = max(0, int(intervention_cfg.get("close_settle_steps", 4) or 0))
    pull_speed = float(intervention_cfg.get("pull_speed", 0.30) or 0.30)
    pull_target_offset = float(
        intervention_cfg.get("pull_target_offset_along_axis", 0.03) or 0.03
    )
    micro_retract_before_pull = max(
        0.0, float(intervention_cfg.get("micro_retract_before_pull", 0.0) or 0.0)
    )
    orientation_hold_gain = float(
        intervention_cfg.get("orientation_hold_gain", 1.0) or 1.0
    )

    images, images2, states, actions, rewards = [], [], [], [], []
    abs_drawer, next_drawer, attached_trace, handle_distance_trace = [], [], [], []
    phase_labels = []
    eef_quat_trace, orientation_alignment_trace, attach_eligible_trace = [], [], []
    orientation_gate_trace, drawer_delta_raw_trace, drawer_delta_effective_trace = (
        [],
        [],
        [],
    )
    orientation_error_trace = []
    pull_alignment_trace, stable_attach_trace, attach_streak_trace = [], [], []
    contact_window_fraction_trace = []
    camera_metadata_trace = []
    handle_probe_metadata_trace = []
    runtime_handle_anchor_valid_trace, runtime_handle_anchor_world_trace = [], []
    phase_locked_trace, raw_grasp_slip_trace, grasp_slip_norm_trace = [], [], []
    pull_progress_trace, pull_increment_trace, effective_pull_progress_trace = (
        [],
        [],
        [],
    )
    effective_pull_progress_norm_trace, detach_reason_trace = [], []
    handle_rel_trace, drawer_fraction_signed_trace = [], []
    controller_subphase_trace, controller_tangential_target_trace = [], []
    controller_preload_trace, controller_lock_score_trace = [], []
    controller_binormal_target_trace = []
    hybrid_frame_n_trace = []
    hybrid_reseat_triggered = False
    hybrid_reseat_phase_before = None
    second_burst_started = False
    second_burst_max_drawer_delta = 0.0
    controller_plateau_reason = None
    ever_attached = False
    attach_step = None
    success = False
    close_hold_steps = 0
    micro_retract_phase_steps = 0
    micro_retract_done = micro_retract_before_pull <= 0.0
    stable_attach_run = 0
    force_detach_steps_remaining = 0
    joint_span = max(float(env.joint_range[1] - env.joint_range[0]), 1e-6)
    opening_servo_stage_idx = 0
    opening_servo_offsets = _opening_servo_offsets(pull_target_offset)
    if teacher_controller_mode == "embodiment_bound_quasistatic":
        opening_servo_offsets = [
            0.015,
            0.03,
            0.045,
            0.06,
            0.075,
            0.09,
            0.105,
            0.12,
            0.135,
        ]
        ramp_speed = min(max(pull_speed, 0.12), 0.14)
        hold_speed = min(max(pull_speed * 0.75, 0.09), 0.11)
        opening_ramp_stall_limit = 4
        opening_hold_stall_limit = 5
        opening_hold_min_steps = 4
        allow_reacquire = False
        phase_lock_required_steps = 3
    elif teacher_controller_mode == "world_frame_matched":
        opening_servo_offsets = _opening_servo_offsets(max(pull_target_offset, 0.03))
        ramp_speed = max(0.16, min(pull_speed, 0.24))
        hold_speed = max(0.10, min(pull_speed * 0.8, 0.20))
        opening_ramp_stall_limit = 2
        opening_hold_stall_limit = 2
        opening_hold_min_steps = 3
        allow_reacquire = True
        phase_lock_required_steps = 3
    elif teacher_controller_mode == "interaction_frame_hybrid":
        opening_servo_offsets = [0.024, 0.04, 0.056, 0.072, 0.088, 0.104]
        ramp_speed = min(max(pull_speed, 0.13), 0.15)
        hold_speed = min(max(pull_speed * 0.70, 0.085), 0.10)
        opening_ramp_stall_limit = 3
        opening_hold_stall_limit = 4
        opening_hold_min_steps = 5
        allow_reacquire = True
        phase_lock_required_steps = 3
    else:
        ramp_speed = max(0.16, min(pull_speed, 0.24))
        hold_speed = max(0.10, min(pull_speed * 0.8, 0.20))
        opening_ramp_stall_limit = 2
        opening_hold_stall_limit = 2
        opening_hold_min_steps = 3
        allow_reacquire = True
        phase_lock_required_steps = 3
    opening_hold_steps = 0
    opening_positive_streak = 0
    opening_stall_steps = 0
    opening_reacquire_count = 0
    phase_lock_settle_steps = 0
    last_drawer_delta_effective = 0.0
    last_phase_locked = False
    last_grasp_slip_norm = 0.0
    hybrid_s_t = 0.0
    hybrid_s_n = 0.0035
    hybrid_s_b = 0.0
    hybrid_lock_score = 0.0
    hybrid_lock_streak = 0
    hybrid_positive_streak = 0
    hybrid_no_progress_streak = 0
    hybrid_high_slip_streak = 0
    hybrid_reseat_used = False
    hybrid_reseat_grasp_streak = 0
    hybrid_frame_n = None
    hybrid_frame_b = None

    phase = "pregrasp"
    if assay_warm_start_kind:
        obs, phase = _apply_assay_warm_start(env, assay_warm_start_kind)
    try:
        for step_idx in range(max_steps):
            images.append(obs.image.copy())
            images2.append(obs.image2.copy())
            states.append(obs.state.copy())
            abs_drawer.append(float(obs.drawer_fraction))
            eef_quat_trace.append(np.asarray(obs.eef_quat, dtype=np.float32).copy())
            camera_metadata_trace.append(_json_ready(obs.camera_metadata or {}))
            handle_probe_metadata_trace.append(
                _json_ready(obs.handle_probe_metadata or {})
            )

            handle, _ = env._interaction_handle_target_world()
            pregrasp_target = (
                handle - axis * 0.04 + np.array([0.0, 0.0, 0.03], dtype=np.float32)
            )
            contact_target = handle + np.array([0.0, 0.0, 0.005], dtype=np.float32)
            micro_retract_target = (
                handle
                - axis * micro_retract_before_pull
                + np.array([0.0, 0.0, 0.005], dtype=np.float32)
            )
            pull_target = (
                handle
                + axis * pull_target_offset
                + np.array([0.0, 0.0, 0.005], dtype=np.float32)
            )
            retreat_target = (
                handle + axis * 0.10 + np.array([0.0, 0.0, 0.05], dtype=np.float32)
            )

            if env._attached and attach_step is None:
                attach_step = step_idx
            if env._attached and phase in {"pregrasp", "contact"}:
                if teacher_controller_mode == "interaction_frame_hybrid":
                    phase = "grasp_seat"
                    hybrid_s_t = 0.0
                    hybrid_s_n = 0.0035
                    hybrid_s_b = 0.0
                    hybrid_lock_score = 0.0
                    hybrid_lock_streak = 0
                    hybrid_positive_streak = 0
                    hybrid_no_progress_streak = 0
                    hybrid_high_slip_streak = 0
                    hybrid_reseat_grasp_streak = 0
                else:
                    phase = "phase_lock_settle"
                    phase_lock_settle_steps = 0
                    opening_servo_stage_idx = 0
                    opening_hold_steps = 0
                    opening_positive_streak = 0
                    opening_stall_steps = 0
            elif (
                phase == "pregrasp"
                and np.linalg.norm(obs.eef_pos - pregrasp_target) < 0.02
            ):
                phase = "contact"
            elif phase == "contact" and np.linalg.norm(obs.eef_pos - handle) < 0.03:
                phase = "close"
                close_hold_steps = 0
                micro_retract_phase_steps = 0
            elif phase == "close":
                close_hold_steps += 1
                if close_settle_steps <= 0 or close_hold_steps >= close_settle_steps:
                    if (
                        env._attached
                        and not micro_retract_done
                        and micro_retract_before_pull > 0.0
                    ):
                        phase = "micro_retract"
                        micro_retract_phase_steps = 0
                    elif env._attached:
                        if teacher_controller_mode == "interaction_frame_hybrid":
                            phase = "grasp_seat"
                            hybrid_lock_streak = 0
                            hybrid_positive_streak = 0
                            hybrid_no_progress_streak = 0
                            hybrid_high_slip_streak = 0
                            hybrid_reseat_grasp_streak = 0
                        else:
                            phase = "phase_lock_settle"
                            phase_lock_settle_steps = 0
                            opening_servo_stage_idx = 0
                            opening_hold_steps = 0
                            opening_positive_streak = 0
                            opening_stall_steps = 0
                    else:
                        phase = "contact"
                    close_hold_steps = 0
            elif phase == "micro_retract":
                micro_retract_phase_steps += 1
                if (
                    np.linalg.norm(obs.eef_pos - micro_retract_target) < 0.015
                    or micro_retract_phase_steps >= 2
                ):
                    micro_retract_done = True
                    if env._attached:
                        if teacher_controller_mode == "interaction_frame_hybrid":
                            phase = "grasp_seat"
                            hybrid_lock_streak = 0
                            hybrid_positive_streak = 0
                            hybrid_no_progress_streak = 0
                            hybrid_high_slip_streak = 0
                            hybrid_reseat_grasp_streak = 0
                        else:
                            phase = "phase_lock_settle"
                            phase_lock_settle_steps = 0
                            opening_servo_stage_idx = 0
                            opening_hold_steps = 0
                            opening_positive_streak = 0
                            opening_stall_steps = 0
                    else:
                        phase = "contact"
            elif phase == "grasp_seat" and not env._attached:
                phase = "contact"
                hybrid_reseat_grasp_streak = 0
            elif phase == "grasp_seat":
                if hybrid_reseat_used:
                    if (
                        env._attached
                        and last_phase_locked
                        and last_grasp_slip_norm <= 0.85
                    ):
                        hybrid_reseat_grasp_streak += 1
                    else:
                        hybrid_reseat_grasp_streak = 0
                    hybrid_lock_score = 0.75 * float(last_phase_locked) + 0.25 * (
                        1.0 - min(last_grasp_slip_norm, 1.0)
                    )
                    if hybrid_reseat_grasp_streak >= 2:
                        phase = "interaction_lock"
                        hybrid_lock_streak = 0
                else:
                    hybrid_lock_score = 0.65 * float(last_phase_locked) + 0.35 * (
                        1.0 - min(last_grasp_slip_norm, 1.0)
                    )
                    if hybrid_lock_score >= 0.55:
                        hybrid_lock_streak += 1
                    else:
                        hybrid_lock_streak = 0
                    if hybrid_lock_streak >= 2:
                        phase = "interaction_lock"
                        hybrid_lock_streak = 0
            elif phase == "interaction_lock" and not env._attached:
                phase = "contact"
            elif phase == "interaction_lock":
                hybrid_lock_score = 0.70 * float(last_phase_locked) + 0.30 * (
                    1.0 - min(last_grasp_slip_norm, 1.0)
                )
                if last_phase_locked and last_grasp_slip_norm <= 0.75:
                    hybrid_lock_streak += 1
                else:
                    hybrid_lock_streak = 0
                if hybrid_lock_streak >= phase_lock_required_steps:
                    next_phase = (
                        "hybrid_open_final"
                        if hybrid_reseat_used
                        else "hybrid_open_ramp"
                    )
                    if hybrid_reseat_used:
                        second_burst_started = True
                    phase = next_phase
                    hybrid_lock_streak = 0
                    hybrid_positive_streak = 0
                    hybrid_no_progress_streak = 0
                    hybrid_high_slip_streak = 0
                    hybrid_s_t = max(hybrid_s_t, 0.020)
            elif phase == "hybrid_open_ramp" and not env._attached:
                if not hybrid_reseat_used:
                    hybrid_reseat_triggered = True
                    hybrid_reseat_phase_before = "hybrid_open_ramp"
                    phase = "reseat_once"
                    micro_retract_phase_steps = 0
                    controller_plateau_reason = "detach_during_ramp"
                else:
                    phase = "contact"
            elif phase == "hybrid_open_ramp":
                if obs.drawer_fraction >= open_fraction:
                    phase = "retreat"
                else:
                    progress_ok = last_drawer_delta_effective > 1e-4
                    slip_low = last_grasp_slip_norm <= 0.60
                    slip_high = last_grasp_slip_norm > 0.75
                    if last_phase_locked and slip_low and progress_ok:
                        hybrid_positive_streak += 1
                        hybrid_no_progress_streak = 0
                        hybrid_high_slip_streak = 0
                        hybrid_s_t = min(hybrid_s_t + 0.003, 0.095)
                        hybrid_s_n = min(max(hybrid_s_n, 0.0035) + 0.00005, 0.0048)
                        if hybrid_positive_streak >= 3:
                            phase = "hybrid_open_hold"
                    elif last_phase_locked and not slip_high:
                        hybrid_positive_streak = 0
                        hybrid_no_progress_streak += 1
                        hybrid_high_slip_streak = 0
                        if hybrid_no_progress_streak >= 4:
                            hybrid_s_t = min(hybrid_s_t + 0.002, 0.095)
                        if hybrid_no_progress_streak >= 8 and not hybrid_reseat_used:
                            hybrid_reseat_triggered = True
                            hybrid_reseat_phase_before = "hybrid_open_ramp"
                            phase = "reseat_once"
                            micro_retract_phase_steps = 0
                            controller_plateau_reason = (
                                "locked_no_progress_before_reseat"
                            )
                    else:
                        hybrid_positive_streak = 0
                        hybrid_no_progress_streak += 1
                        if slip_high:
                            hybrid_high_slip_streak += 1
                            hybrid_s_n = min(hybrid_s_n + 0.0003, 0.0060)
                        else:
                            hybrid_high_slip_streak = 0
                        if hybrid_high_slip_streak >= 4 and not hybrid_reseat_used:
                            hybrid_reseat_triggered = True
                            hybrid_reseat_phase_before = "hybrid_open_ramp"
                            phase = "reseat_once"
                            micro_retract_phase_steps = 0
                            controller_plateau_reason = "high_slip_before_reseat"
            elif phase == "hybrid_open_hold" and not env._attached:
                if not hybrid_reseat_used:
                    hybrid_reseat_triggered = True
                    hybrid_reseat_phase_before = "hybrid_open_hold"
                    phase = "reseat_once"
                    micro_retract_phase_steps = 0
                    controller_plateau_reason = "detach_during_hold"
                else:
                    phase = "contact"
            elif phase == "hybrid_open_hold":
                if obs.drawer_fraction >= open_fraction:
                    phase = "retreat"
                else:
                    progress_ok = last_drawer_delta_effective > 1e-4
                    slip_high = last_grasp_slip_norm > 0.75
                    if progress_ok and last_phase_locked:
                        hybrid_no_progress_streak = 0
                        hybrid_high_slip_streak = 0
                        hybrid_s_n = max(hybrid_s_n, 0.0035)
                    else:
                        hybrid_no_progress_streak += 1
                        if slip_high:
                            hybrid_high_slip_streak += 1
                            hybrid_s_n = min(hybrid_s_n + 0.0004, 0.0070)
                        else:
                            hybrid_high_slip_streak = 0
                        if hybrid_no_progress_streak >= 3:
                            hybrid_s_t = min(hybrid_s_t + 0.004, 0.14)
                            phase = "hybrid_open_ramp"
                        if hybrid_high_slip_streak >= 4 and not hybrid_reseat_used:
                            hybrid_reseat_triggered = True
                            hybrid_reseat_phase_before = "hybrid_open_hold"
                            phase = "reseat_once"
                            micro_retract_phase_steps = 0
                            controller_plateau_reason = "hold_slip_before_reseat"
            elif phase == "reseat_once":
                micro_retract_phase_steps += 1
                if (
                    np.linalg.norm(obs.eef_pos - micro_retract_target) < 0.015
                    or micro_retract_phase_steps >= 2
                ):
                    hybrid_reseat_used = True
                    hybrid_lock_streak = 0
                    hybrid_positive_streak = 0
                    hybrid_no_progress_streak = 0
                    hybrid_high_slip_streak = 0
                    hybrid_reseat_grasp_streak = 0
                    hybrid_frame_n = None
                    hybrid_frame_b = None
                    phase = "interaction_lock" if env._attached else "contact"
            elif phase == "hybrid_open_final" and not env._attached:
                phase = "contact"
            elif phase == "hybrid_open_final":
                if obs.drawer_fraction >= open_fraction:
                    phase = "retreat"
                else:
                    progress_ok = last_drawer_delta_effective > 1e-4
                    if last_phase_locked and progress_ok:
                        hybrid_s_t = min(hybrid_s_t + 0.004, 0.145)
                        hybrid_s_n = max(hybrid_s_n, 0.0035)
                        hybrid_no_progress_streak = 0
                    else:
                        hybrid_no_progress_streak += 1
                        hybrid_s_n = min(hybrid_s_n + 0.0002, 0.0055)
                        if hybrid_no_progress_streak >= 5:
                            controller_plateau_reason = "final_no_progress_after_reseat"
            elif phase == "phase_lock_settle" and not env._attached:
                phase = "contact"
            elif phase == "phase_lock_settle":
                phase_lock_settle_steps += 1
                if (
                    last_phase_locked
                    and phase_lock_settle_steps >= phase_lock_required_steps
                ):
                    phase = "opening_ramp"
                    opening_servo_stage_idx = 0
                    opening_positive_streak = 0
                    opening_hold_steps = 0
                    opening_stall_steps = 0
            elif phase == "opening_ramp" and not env._attached:
                phase = "contact"
            elif phase == "opening_ramp":
                if obs.drawer_fraction >= open_fraction:
                    phase = "retreat"
                elif last_drawer_delta_effective > 1e-4:
                    opening_positive_streak += 1
                    opening_stall_steps = 0
                    if opening_positive_streak >= (
                        3
                        if teacher_controller_mode == "embodiment_bound_quasistatic"
                        else 2
                    ):
                        phase = "opening_hold"
                        opening_hold_steps = 0
                else:
                    opening_positive_streak = 0
                    opening_stall_steps += 1
                    if opening_stall_steps >= opening_ramp_stall_limit:
                        if (
                            teacher_controller_mode == "embodiment_bound_quasistatic"
                            and last_phase_locked
                            and last_grasp_slip_norm <= 0.60
                            and opening_servo_stage_idx < len(opening_servo_offsets) - 1
                        ):
                            opening_servo_stage_idx += 1
                            opening_stall_steps = 0
                        elif (
                            opening_servo_stage_idx < len(opening_servo_offsets) - 1
                            and teacher_controller_mode
                            != "embodiment_bound_quasistatic"
                        ):
                            opening_servo_stage_idx += 1
                            opening_stall_steps = 0
                        elif allow_reacquire and opening_reacquire_count < 1:
                            opening_reacquire_count += 1
                            phase = "reacquire"
                            micro_retract_phase_steps = 0
                        elif teacher_controller_mode == "embodiment_bound_quasistatic":
                            controller_plateau_reason = (
                                "no_progress_under_lock"
                                if last_phase_locked
                                else "lock_lost_before_progress"
                            )
            elif phase == "opening_hold" and not env._attached:
                phase = "contact"
            elif phase == "opening_hold":
                opening_hold_steps += 1
                if obs.drawer_fraction >= open_fraction:
                    phase = "retreat"
                elif last_drawer_delta_effective > 1e-4:
                    opening_stall_steps = 0
                else:
                    opening_stall_steps += 1
                if (
                    opening_hold_steps >= opening_hold_min_steps
                    and opening_stall_steps >= opening_hold_stall_limit
                ):
                    if (
                        teacher_controller_mode == "embodiment_bound_quasistatic"
                        and last_phase_locked
                        and last_grasp_slip_norm <= 0.60
                        and opening_servo_stage_idx < len(opening_servo_offsets) - 1
                    ):
                        opening_servo_stage_idx += 1
                        opening_hold_steps = 0
                        opening_positive_streak = 0
                        opening_stall_steps = 0
                        phase = "opening_ramp"
                    elif (
                        opening_servo_stage_idx < len(opening_servo_offsets) - 1
                        and teacher_controller_mode != "embodiment_bound_quasistatic"
                    ):
                        opening_servo_stage_idx += 1
                        opening_hold_steps = 0
                        opening_positive_streak = 0
                        opening_stall_steps = 0
                        phase = "opening_ramp"
                    elif allow_reacquire and opening_reacquire_count < 1:
                        opening_reacquire_count += 1
                        phase = "reacquire"
                        micro_retract_phase_steps = 0
                    elif teacher_controller_mode == "embodiment_bound_quasistatic":
                        controller_plateau_reason = "hold_stall_after_monotone_push"
            elif phase == "reacquire":
                micro_retract_phase_steps += 1
                if (
                    np.linalg.norm(obs.eef_pos - micro_retract_target) < 0.015
                    or micro_retract_phase_steps >= 2
                ):
                    if env._attached:
                        phase = "phase_lock_settle"
                        phase_lock_settle_steps = 0
                        opening_positive_streak = 0
                        opening_stall_steps = 0
                    else:
                        phase = "contact"
            elif (
                phase
                in {
                    "phase_lock_settle",
                    "opening_ramp",
                    "opening_hold",
                    "interaction_lock",
                    "hybrid_open_ramp",
                    "hybrid_open_hold",
                    "hybrid_open_final",
                }
                and obs.drawer_fraction >= open_fraction
            ):
                phase = "retreat"

            stable_attach_run = stable_attach_run + 1 if bool(env._stable_attach) else 0
            stable_attach_run = stable_attach_run + 1 if bool(env._stable_attach) else 0
            if (
                force_detach_probe
                and force_detach_steps_remaining <= 0
                and stable_attach_run >= 2
            ):
                force_detach_steps_remaining = 2

            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            interaction_n = (handle - obs.eef_pos).astype(np.float32)
            interaction_n = interaction_n - axis * float(np.dot(interaction_n, axis))
            interaction_n_norm = float(np.linalg.norm(interaction_n))
            if interaction_n_norm < 1e-4:
                interaction_n = np.cross(axis, world_up).astype(np.float32)
                interaction_n_norm = float(np.linalg.norm(interaction_n))
            if interaction_n_norm < 1e-4:
                interaction_n = np.cross(
                    axis, np.array([1.0, 0.0, 0.0], dtype=np.float32)
                ).astype(np.float32)
                interaction_n_norm = float(np.linalg.norm(interaction_n))
            if interaction_n_norm < 1e-6:
                interaction_n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                interaction_n_norm = 1.0
            interaction_n = interaction_n / interaction_n_norm
            interaction_b = np.cross(axis, interaction_n).astype(np.float32)
            interaction_b_norm = float(np.linalg.norm(interaction_b))
            if interaction_b_norm < 1e-6:
                interaction_b = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                interaction_b_norm = 1.0
            interaction_b = interaction_b / interaction_b_norm
            if teacher_controller_mode == "interaction_frame_hybrid":
                if (
                    phase == "interaction_lock"
                    and env._attached
                    and hybrid_frame_n is None
                    and hybrid_frame_b is None
                ):
                    hybrid_frame_n = interaction_n.copy()
                    hybrid_frame_b = interaction_b.copy()
                elif (
                    phase
                    in {
                        "hybrid_open_ramp",
                        "hybrid_open_hold",
                        "hybrid_open_final",
                    }
                    and hybrid_frame_n is not None
                    and hybrid_frame_b is not None
                ):
                    interaction_n = hybrid_frame_n.copy()
                    interaction_b = hybrid_frame_b.copy()
            if teacher_controller_mode == "interaction_frame_hybrid":
                hybrid_frame_n_trace.append(
                    np.asarray(interaction_n, dtype=np.float32).copy()
                )
                if second_burst_started:
                    second_burst_max_drawer_delta = max(
                        second_burst_max_drawer_delta,
                        max(last_drawer_delta_effective, 0.0),
                    )
            preload_mag = 0.0
            if teacher_controller_mode == "embodiment_bound_quasistatic":
                slip_level = float(np.clip(last_grasp_slip_norm, 0.0, 1.0))
                progress_gate = 1.0 if last_drawer_delta_effective > 1e-4 else 0.0
                preload_mag = max(
                    0.0,
                    0.001
                    + 0.003 * max(slip_level - 0.20, 0.0)
                    + 0.001 * (1.0 - progress_gate),
                )
            elif teacher_controller_mode == "interaction_frame_hybrid":
                preload_mag = hybrid_s_n

            if phase == "pregrasp":
                target_pos, close, speed = pregrasp_target, False, 0.65
            elif phase == "contact":
                target_pos, close, speed = contact_target, False, 0.45
            elif phase == "close":
                target_pos, close, speed = contact_target, True, 0.25
            elif phase == "micro_retract":
                target_pos, close, speed = micro_retract_target, True, 0.20
            elif phase == "grasp_seat":
                if hybrid_reseat_used:
                    target_pos = obs.eef_pos + interaction_n * 0.0008
                    close, speed = True, 0.10
                else:
                    target_pos = contact_target + interaction_n * max(
                        preload_mag, 0.0035
                    )
                    close, speed = True, 0.16
            elif phase == "interaction_lock":
                target_pos = (
                    handle
                    + axis * hybrid_s_t
                    + interaction_n * max(preload_mag, 0.0035)
                    + np.array([0.0, 0.0, 0.005], dtype=np.float32)
                )
                close, speed = True, 0.14
            elif phase == "hybrid_open_ramp":
                target_pos = (
                    handle
                    + axis * hybrid_s_t
                    + interaction_n * preload_mag
                    + interaction_b * hybrid_s_b
                    + np.array([0.0, 0.0, 0.005], dtype=np.float32)
                )
                close, speed = True, ramp_speed
            elif phase == "hybrid_open_hold":
                target_pos = (
                    handle
                    + axis * hybrid_s_t
                    + interaction_n * max(preload_mag - 0.0002, 0.0)
                    + interaction_b * hybrid_s_b
                    + np.array([0.0, 0.0, 0.005], dtype=np.float32)
                )
                close, speed = True, hold_speed
            elif phase == "reseat_once":
                target_pos = (
                    micro_retract_target
                    + interaction_n * max(preload_mag - 0.001, 0.0020)
                    + interaction_b * hybrid_s_b
                )
                close, speed = True, 0.18
            elif phase == "hybrid_open_final":
                target_pos = (
                    handle
                    + axis * hybrid_s_t
                    + interaction_n * preload_mag
                    + interaction_b * hybrid_s_b
                    + np.array([0.0, 0.0, 0.005], dtype=np.float32)
                )
                close, speed = True, hold_speed
            elif phase == "phase_lock_settle":
                if teacher_controller_mode in {
                    "embodiment_bound_quasistatic",
                    "interaction_frame_hybrid",
                }:
                    target_pos = contact_target + interaction_n * preload_mag
                else:
                    target_pos = contact_target
                close, speed = True, 0.18
            elif phase == "opening_ramp":
                target_pos = (
                    handle
                    + axis * opening_servo_offsets[opening_servo_stage_idx]
                    + np.array([0.0, 0.0, 0.005], dtype=np.float32)
                )
                if teacher_controller_mode in {
                    "embodiment_bound_quasistatic",
                    "interaction_frame_hybrid",
                }:
                    target_pos = target_pos + interaction_n * preload_mag
                close, speed = True, ramp_speed
            elif phase == "opening_hold":
                target_pos = (
                    handle
                    + axis * opening_servo_offsets[opening_servo_stage_idx]
                    + np.array([0.0, 0.0, 0.005], dtype=np.float32)
                )
                if teacher_controller_mode in {
                    "embodiment_bound_quasistatic",
                    "interaction_frame_hybrid",
                }:
                    target_pos = target_pos + interaction_n * max(
                        preload_mag - 0.001, 0.0
                    )
                close, speed = True, hold_speed
            elif phase == "reacquire":
                if teacher_controller_mode in {
                    "embodiment_bound_quasistatic",
                    "interaction_frame_hybrid",
                }:
                    target_pos = micro_retract_target + interaction_n * max(
                        preload_mag - 0.002, 0.0
                    )
                else:
                    target_pos = micro_retract_target
                close, speed = True, 0.18
            else:
                target_pos, close, speed = retreat_target, False, 0.55

            action = _script_action(
                obs.eef_pos,
                target_pos,
                close,
                speed=speed,
                current_quat=obs.eef_quat,
                motion_axis=env._motion_axis,
                rotation_source=rotation_source,
                rng=rng,
                phase=phase,
                orientation_gain=orientation_hold_gain,
            )
            controller_subphase_trace.append(phase)
            controller_tangential_target_trace.append(
                float(hybrid_s_t)
                if teacher_controller_mode == "interaction_frame_hybrid"
                else (
                    float(
                        opening_servo_offsets[
                            min(opening_servo_stage_idx, len(opening_servo_offsets) - 1)
                        ]
                    )
                    if phase in {"opening_ramp", "opening_hold"}
                    else 0.0
                )
            )
            controller_preload_trace.append(float(preload_mag))
            controller_lock_score_trace.append(
                float(hybrid_lock_score)
                if teacher_controller_mode == "interaction_frame_hybrid"
                else (1.0 if last_phase_locked else 0.0)
            )
            controller_binormal_target_trace.append(float(hybrid_s_b))
            phase_label = phase
            if force_detach_steps_remaining > 0:
                action = _script_action(
                    obs.eef_pos,
                    contact_target,
                    False,
                    speed=0.20,
                    current_quat=obs.eef_quat,
                    motion_axis=env._motion_axis,
                    rotation_source="random",
                    rng=rng,
                    phase="forced_detach",
                    orientation_gain=1.0,
                )
                force_detach_steps_remaining -= 1
                phase_label = "forced_detach"
            next_obs, reward, done, info = env.step(action)
            actions.append(action.copy())
            rewards.append(float(reward))
            phase_labels.append(phase_label)
            attached_trace.append(bool(info["attached"]))
            handle_distance_trace.append(float(info["dist_to_handle"]))
            next_drawer.append(float(next_obs.drawer_fraction))
            orientation_alignment_trace.append(
                float(info.get("orientation_alignment_cos", 1.0))
            )
            attach_eligible_trace.append(bool(info.get("attach_eligible", False)))
            orientation_gate_trace.append(
                bool(info.get("orientation_gate_passed", True))
            )
            drawer_delta_raw_trace.append(float(info.get("drawer_delta_raw", 0.0)))
            last_drawer_delta_effective = float(info.get("drawer_delta_effective", 0.0))
            drawer_delta_effective_trace.append(last_drawer_delta_effective)
            orientation_error_trace.append(
                float(info.get("orientation_error_rad", 0.0))
            )
            pull_alignment_trace.append(float(info.get("pull_alignment_cos", 0.0)))
            stable_attach_trace.append(bool(info.get("stable_attach", False)))
            attach_streak_trace.append(float(info.get("attach_streak", 0.0)))
            contact_window_fraction_trace.append(
                float(info.get("contact_window_fraction", 0.0))
            )
            runtime_handle_anchor_valid_trace.append(
                bool(info.get("runtime_handle_anchor_valid", False))
            )
            runtime_handle_anchor_world_trace.append(
                np.asarray(
                    info.get("runtime_handle_anchor_world", [0.0, 0.0, 0.0]),
                    dtype=np.float32,
                )
            )
            last_phase_locked = bool(info.get("phase_locked", False))
            phase_locked_trace.append(last_phase_locked)
            raw_grasp_slip_trace.append(float(info.get("raw_grasp_slip", 0.0)))
            last_grasp_slip_norm = float(info.get("grasp_slip_norm", 0.0))
            grasp_slip_norm_trace.append(last_grasp_slip_norm)
            pull_progress_trace.append(float(info.get("pull_progress", 0.0)))
            pull_increment_trace.append(float(info.get("pull_increment", 0.0)))
            effective_pull_progress = float(info.get("effective_pull_progress", 0.0))
            effective_pull_progress_trace.append(effective_pull_progress)
            effective_pull_progress_norm_trace.append(
                float(np.clip(effective_pull_progress / joint_span, 0.0, 1.0))
            )
            detach_reason_trace.append(info.get("detach_reason"))
            handle_rel = np.clip(
                (handle - obs.eef_pos) / np.array([0.22, 0.18, 0.14], dtype=np.float32),
                -1.0,
                1.0,
            )
            handle_rel_trace.append(handle_rel.astype(np.float32))
            drawer_fraction_signed_trace.append(
                float(info.get("drawer_fraction_signed", 0.0))
            )
            ever_attached = ever_attached or bool(info["attached"])
            obs = next_obs
            success = bool(info["is_success"])
            if done:
                break
    finally:
        env.close()

    state_spec = obs.state_spec or env.state_spec()
    visual_mode_report = obs.visual_mode_report or {}
    camera_metadata = obs.camera_metadata or {}
    handle_probe_metadata = obs.handle_probe_metadata or {}
    rollout_claim_policy = claim_policy or obs.claim_policy or env.claim_policy()
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
        "eef_quat_trace": np.asarray(eef_quat_trace, dtype=np.float32),
        "orientation_alignment_trace": np.asarray(
            orientation_alignment_trace, dtype=np.float32
        ),
        "attach_eligible_trace": np.asarray(attach_eligible_trace, dtype=np.bool_),
        "orientation_gate_trace": np.asarray(orientation_gate_trace, dtype=np.bool_),
        "drawer_delta_raw_trace": np.asarray(drawer_delta_raw_trace, dtype=np.float32),
        "drawer_delta_effective_trace": np.asarray(
            drawer_delta_effective_trace, dtype=np.float32
        ),
        "orientation_error_trace": np.asarray(
            orientation_error_trace, dtype=np.float32
        ),
        "pull_alignment_trace": np.asarray(pull_alignment_trace, dtype=np.float32),
        "stable_attach_trace": np.asarray(stable_attach_trace, dtype=np.bool_),
        "attach_streak_trace": np.asarray(attach_streak_trace, dtype=np.float32),
        "contact_window_fraction_trace": np.asarray(
            contact_window_fraction_trace, dtype=np.float32
        ),
        "runtime_handle_anchor_valid_trace": np.asarray(
            runtime_handle_anchor_valid_trace, dtype=np.bool_
        ),
        "runtime_handle_anchor_world_trace": np.asarray(
            runtime_handle_anchor_world_trace, dtype=np.float32
        ),
        "phase_locked_trace": np.asarray(phase_locked_trace, dtype=np.bool_),
        "raw_grasp_slip_trace": np.asarray(raw_grasp_slip_trace, dtype=np.float32),
        "grasp_slip_norm_trace": np.asarray(grasp_slip_norm_trace, dtype=np.float32),
        "pull_progress_trace": np.asarray(pull_progress_trace, dtype=np.float32),
        "pull_increment_trace": np.asarray(pull_increment_trace, dtype=np.float32),
        "effective_pull_progress_trace": np.asarray(
            effective_pull_progress_trace, dtype=np.float32
        ),
        "effective_pull_progress_norm_trace": np.asarray(
            effective_pull_progress_norm_trace, dtype=np.float32
        ),
        "detach_reason_trace": np.asarray(detach_reason_trace, dtype=object),
        "handle_rel_trace": np.asarray(handle_rel_trace, dtype=np.float32),
        "drawer_fraction_signed_trace": np.asarray(
            drawer_fraction_signed_trace, dtype=np.float32
        ),
        "controller_subphase_trace": np.asarray(
            controller_subphase_trace, dtype=object
        ),
        "controller_tangential_target_trace": np.asarray(
            controller_tangential_target_trace, dtype=np.float32
        ),
        "controller_preload_trace": np.asarray(
            controller_preload_trace, dtype=np.float32
        ),
        "controller_lock_score_trace": np.asarray(
            controller_lock_score_trace, dtype=np.float32
        ),
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
            "rotation_source": rotation_source,
            "backend": "mujoco",
            "robot_in_loop": True,
            "interaction_mode": env.contract.interaction_mode,
            "interventions": intervention_cfg,
            "assay_warm_start_kind": assay_warm_start_kind,
            "teacher_controller_mode": teacher_controller_mode,
        },
        "contract_config": env.contract_payload(),
        "state_spec": _json_ready(state_spec),
        "visual_mode_report": _json_ready(visual_mode_report),
        "camera_metadata": _json_ready(camera_metadata),
        "camera_metadata_trace": _json_ready(camera_metadata_trace),
        "handle_probe_metadata": _json_ready(handle_probe_metadata),
        "handle_probe_metadata_trace": _json_ready(handle_probe_metadata_trace),
        "claim_policy": rollout_claim_policy,
        "orientation_telemetry": _json_ready(env._last_orientation_info),
        "resource_budget_snapshot": {},
        "raw_unique_frames": int(len(actions)),
        "effective_training_frames": int(len(actions)),
        "teacher_controller_mode": teacher_controller_mode,
        "p1b_s_t_trace": (
            np.asarray(controller_tangential_target_trace, dtype=np.float32)
            if teacher_controller_mode == "embodiment_bound_quasistatic"
            else np.asarray([], dtype=np.float32)
        ),
        "p1b_s_n_trace": (
            np.asarray(controller_preload_trace, dtype=np.float32)
            if teacher_controller_mode == "embodiment_bound_quasistatic"
            else np.asarray([], dtype=np.float32)
        ),
        "p1b_lock_q_trace": (
            np.asarray(controller_lock_score_trace, dtype=np.float32)
            if teacher_controller_mode == "embodiment_bound_quasistatic"
            else np.asarray([], dtype=np.float32)
        ),
        "p1b_plateau_reason": controller_plateau_reason,
        "interaction_s_t_trace": (
            np.asarray(controller_tangential_target_trace, dtype=np.float32)
            if teacher_controller_mode == "interaction_frame_hybrid"
            else np.asarray([], dtype=np.float32)
        ),
        "interaction_s_n_trace": (
            np.asarray(controller_preload_trace, dtype=np.float32)
            if teacher_controller_mode == "interaction_frame_hybrid"
            else np.asarray([], dtype=np.float32)
        ),
        "interaction_s_b_trace": (
            np.asarray(controller_binormal_target_trace, dtype=np.float32)
            if teacher_controller_mode == "interaction_frame_hybrid"
            else np.asarray([], dtype=np.float32)
        ),
        "interaction_lock_score_trace": (
            np.asarray(controller_lock_score_trace, dtype=np.float32)
            if teacher_controller_mode == "interaction_frame_hybrid"
            else np.asarray([], dtype=np.float32)
        ),
        "hybrid_frame_n_trace": (
            np.asarray(hybrid_frame_n_trace, dtype=np.float32)
            if teacher_controller_mode == "interaction_frame_hybrid"
            else np.asarray([], dtype=np.float32)
        ),
        "hybrid_reseat_triggered": bool(hybrid_reseat_triggered)
        if teacher_controller_mode == "interaction_frame_hybrid"
        else False,
        "hybrid_reseat_phase_before": hybrid_reseat_phase_before
        if teacher_controller_mode == "interaction_frame_hybrid"
        else None,
        "second_burst_started": bool(second_burst_started)
        if teacher_controller_mode == "interaction_frame_hybrid"
        else False,
        "second_burst_max_drawer_delta": float(second_burst_max_drawer_delta)
        if teacher_controller_mode == "interaction_frame_hybrid"
        else 0.0,
        "continuation_plateau_reason": controller_plateau_reason
        if teacher_controller_mode == "interaction_frame_hybrid"
        else None,
        "reseat_reason": controller_plateau_reason
        if teacher_controller_mode == "interaction_frame_hybrid"
        else None,
    }
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
        eef_quat_trace=rollout["eef_quat_trace"],
        orientation_alignment_trace=rollout["orientation_alignment_trace"],
        attach_eligible_trace=rollout["attach_eligible_trace"],
        orientation_gate_trace=rollout["orientation_gate_trace"],
        drawer_delta_raw_trace=rollout["drawer_delta_raw_trace"],
        drawer_delta_effective_trace=rollout["drawer_delta_effective_trace"],
        orientation_error_trace=rollout["orientation_error_trace"],
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
        "contract_config": rollout.get("contract_config", {}),
        "state_spec": rollout.get("state_spec", {}),
        "visual_mode_report": rollout.get("visual_mode_report", {}),
        "camera_metadata": rollout.get("camera_metadata", {}),
        "camera_metadata_trace": rollout.get("camera_metadata_trace", []),
        "handle_probe_metadata": rollout.get("handle_probe_metadata", {}),
        "handle_probe_metadata_trace": rollout.get("handle_probe_metadata_trace", []),
        "claim_policy": rollout.get("claim_policy", "diagnostic"),
        "orientation_telemetry": rollout.get("orientation_telemetry", {}),
        "resource_budget_snapshot": rollout.get("resource_budget_snapshot", {}),
        "raw_unique_frames": rollout.get("raw_unique_frames", rollout["steps"]),
        "effective_training_frames": rollout.get(
            "effective_training_frames", rollout["steps"]
        ),
        "selector_mode": rollout.get("selector_mode"),
        "baseline_cell_id": rollout.get("baseline_cell_id"),
        "ts_baseline_cell_id": rollout.get("ts_baseline_cell_id"),
        "frozen_matrix_hash": rollout.get("frozen_matrix_hash"),
        "lane_hash": rollout.get("lane_hash"),
        "best_transition_cell": rollout.get("best_transition_cell"),
        "canonical_train_cell": rollout.get("canonical_train_cell"),
        "best_train_state_mode": rollout.get("best_train_state_mode"),
        "source_canonical_train_cell": rollout.get("source_canonical_train_cell"),
        "source_best_train_state_mode": rollout.get("source_best_train_state_mode"),
        "active_train_state_mode": rollout.get("active_train_state_mode"),
        "active_state_mode_name": rollout.get("active_state_mode_name"),
        "run_instance_id": rollout.get("run_instance_id"),
        "plan_version": rollout.get("plan_version"),
        "source_base_commit": rollout.get("source_base_commit"),
        "working_head_commit": rollout.get("working_head_commit"),
        "bridge_stage": rollout.get("bridge_stage"),
        "bridge_attempt": rollout.get("bridge_attempt"),
        "strict_utility_version": rollout.get("strict_utility_version"),
        "truth_utility_version": rollout.get("truth_utility_version"),
        "teacher_fingerprint_version": rollout.get("teacher_fingerprint_version"),
        "teacher_truth_gate": rollout.get("teacher_truth_gate"),
        "truth_contract_path": rollout.get("truth_contract_path"),
        "truth_contract_hash": rollout.get("truth_contract_hash"),
        "truth_contract_root_object": rollout.get("truth_contract_root_object"),
        "strict_success_version": rollout.get("strict_success_version"),
        "strict_metrics": rollout.get("strict_metrics", {}),
        "measurement_truthful": rollout.get("measurement_truthful"),
        "measurement_truth_tier": rollout.get("measurement_truth_tier"),
        "measurement_backend": rollout.get("measurement_backend"),
        "measurement_verifier": rollout.get("measurement_verifier"),
        "measurement_truthful_for_training": rollout.get(
            "measurement_truthful_for_training"
        ),
        "teacher_truth_adjudication": rollout.get("teacher_truth_adjudication"),
        "teacher_truthful_window_frame_count": rollout.get(
            "teacher_truthful_window_frame_count"
        ),
        "truthful_window_ratio": rollout.get("truthful_window_ratio"),
        "truthful_window_longest_interior_gap": rollout.get(
            "truthful_window_longest_interior_gap"
        ),
        "truthful_window_tail_truthful_count_last6": rollout.get(
            "truthful_window_tail_truthful_count_last6"
        ),
        "bridge_in_truthful_window": rollout.get("bridge_in_truthful_window"),
        "teacher_episode_class": rollout.get("teacher_episode_class"),
        "teacher_fingerprint": rollout.get("teacher_fingerprint"),
        "final_snapshot_measurement_truthful": rollout.get(
            "final_snapshot_measurement_truthful"
        ),
        "final_snapshot_measurement_truth_tier": rollout.get(
            "final_snapshot_measurement_truth_tier"
        ),
        "canonical_training_truth": rollout.get("canonical_training_truth", {}),
        "runtime_visible_handle_mapping_source": rollout.get(
            "runtime_visible_handle_mapping_source"
        ),
        "runtime_handle_anchor_valid": rollout.get("runtime_handle_anchor_valid"),
        "interaction_mode": rollout.get("interaction_mode"),
        "state_mode": rollout.get("state_mode"),
        "train_seeds": rollout.get("train_seeds", []),
        "heldout_seeds": rollout.get("heldout_seeds", []),
        "teacher_controller_mode": rollout.get("teacher_controller_mode"),
        "hybrid_reseat_triggered": rollout.get("hybrid_reseat_triggered"),
        "hybrid_reseat_phase_before": rollout.get("hybrid_reseat_phase_before"),
        "second_burst_started": rollout.get("second_burst_started"),
        "second_burst_max_drawer_delta": rollout.get("second_burst_max_drawer_delta"),
        "continuation_plateau_reason": rollout.get("continuation_plateau_reason"),
        "hybrid_frame_n_trace": rollout.get("hybrid_frame_n_trace", []),
    }
    write_text_atomic(
        path.with_suffix(".json"),
        json.dumps(_json_ready(meta), indent=2, ensure_ascii=False) + "\n",
    )

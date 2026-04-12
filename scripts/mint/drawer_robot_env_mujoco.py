#!/usr/bin/env python3
"""True MuJoCo drawer environment for the canonical Infinigen mainline."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.spatial.transform import Rotation as R

from mint_common import write_text_atomic

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
InteractionMode = Literal["legacy_translation_only", "orientation_sensitive_v1"]
StateMode = Literal["m0_proxy", "eef_pose_gripper", "telemetry_candidate_v1", "telemetry_candidate_v2"]
RenderProfile = Literal["legacy_surface", "visual_reformulation_v0", "visual_reformulation_v1", "visual_reformulation_v1_raw_canonical", "visual_reformulation_v1_plus_bundle", "visual_reformulation_v2_material_light_bg", "visual_reformulation_v2_plus_bundle"]
BackgroundMode = Literal["legacy_scene", "neutral_lab", "high_contrast_lab"]
LightingProfile = Literal["legacy", "bright_front_fill"]
MaterialPolicy = Literal["legacy", "handle_highlight"]
CameraFramingProfile = Literal["legacy", "tight_handle_centered"]
RotationSource = Literal["zero", "aligned", "random"]


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


def _camera_eye_from_lookat(lookat: np.ndarray, distance: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
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


def _transform_from_eye_lookat(eye: np.ndarray, lookat: np.ndarray, up_hint: np.ndarray = WORLD_UP) -> np.ndarray:
    forward = _normalize(np.asarray(lookat, dtype=np.float32) - np.asarray(eye, dtype=np.float32), WORLD_X)
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
        self.contract = contract if contract is not None else DrawerEnvContractConfig.legacy_defaults()
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
        self._rng = np.random.default_rng(self.seed)
        self._original_geom_rgba = np.asarray(self.model.geom_rgba, dtype=np.float32).copy()
        self._original_headlight_ambient = np.asarray(self.model.vis.headlight.ambient, dtype=np.float32).copy()
        self._original_headlight_diffuse = np.asarray(self.model.vis.headlight.diffuse, dtype=np.float32).copy()
        self._original_headlight_specular = np.asarray(self.model.vis.headlight.specular, dtype=np.float32).copy()
        self._original_haze_rgba = np.asarray(self.model.vis.rgba.haze, dtype=np.float32).copy()
        self._last_camera_metadata: dict[str, Any] = {}
        self._last_visual_mode_report: dict[str, Any] = {}
        self._last_handle_probe_metadata: dict[str, Any] = {}
        self._last_orientation_info = self._default_orientation_info()
        self._last_claim_policy = self.claim_policy()
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
            "drawer_delta_raw": 0.0,
            "drawer_delta_effective": 0.0,
            "open_orientation_weight": 1.0,
            "orientation_error_rad": 0.0,
        }

    def _joint_axis_world(self) -> np.ndarray:
        axis = self.model.jnt_axis[self.joint_idx].astype(np.float32)
        default = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return _normalize(axis, default)

    def _drawer_fraction(self) -> float:
        low, high = self.joint_range.tolist()
        span = max(high - low, 1e-6)
        return float(np.clip((float(self.data.qpos[self.joint_idx]) - low) / span, 0.0, 1.0))

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

    def _background_rgb(self, secondary: bool) -> np.ndarray:
        if self.contract.background_mode == "neutral_lab":
            return np.asarray([214, 207, 198], dtype=np.uint8)
        if self.contract.background_mode == "high_contrast_lab":
            return np.asarray([236, 233, 228], dtype=np.uint8) if secondary else np.asarray([208, 204, 198], dtype=np.uint8)
        mean = SECONDARY_BG_MEAN if secondary else PRIMARY_BG_MEAN
        return np.clip(mean * 255.0, 0.0, 255.0).astype(np.uint8)

    def _apply_calibration(self, image: np.ndarray, *, secondary: bool) -> np.ndarray:
        mode = self.contract.calibration_mode
        if mode == "none":
            return image.astype(np.uint8)
        if mode == "diagnostic_texture":
            return self._apply_diagnostic_texture(image, secondary=secondary)
        out = image.astype(np.float32)
        bg = self._background_rgb(secondary).astype(np.float32)
        mask = np.max(out, axis=2, keepdims=True) < 8.0
        out = np.where(mask, bg.reshape(1, 1, 3), out)
        gain = 1.18 if secondary else 1.22
        bias = 6.0 if secondary else 10.0
        out = out * gain + bias
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    def _apply_diagnostic_texture(self, image: np.ndarray, *, secondary: bool) -> np.ndarray:
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
        v2_profile = self.contract.render_profile in {"visual_reformulation_v2_material_light_bg", "visual_reformulation_v2_plus_bundle"}
        if self.contract.camera_framing_profile == "tight_handle_centered":
            primary_lookat = 0.74 * handle + 0.26 * self.scene_center
            self.cam_primary.lookat[:] = primary_lookat.tolist()
            self.cam_primary.distance = float(self.scene_extent * (1.08 if v2_profile else 1.25))
            self.cam_primary.azimuth = 166.0 if v2_profile else 168.0
            self.cam_primary.elevation = -15.0 if v2_profile else -18.0
        else:
            self.cam_primary.lookat[:] = self.scene_center.tolist()
            self.cam_primary.distance = float(self.scene_extent * 1.85)
            self.cam_primary.azimuth = 180.0
            self.cam_primary.elevation = -28.0

        if self.contract.background_mode == "neutral_lab":
            self.model.vis.rgba.haze[:] = np.asarray([0.84, 0.82, 0.79, 1.0], dtype=np.float32)
        elif self.contract.background_mode == "high_contrast_lab":
            self.model.vis.rgba.haze[:] = np.asarray([0.96, 0.95, 0.93, 1.0], dtype=np.float32) if v2_profile else np.asarray([0.90, 0.89, 0.86, 1.0], dtype=np.float32)

        if self.contract.lighting_profile == "bright_front_fill":
            self.model.vis.headlight.ambient[:] = np.asarray([0.62, 0.62, 0.62], dtype=np.float32) if v2_profile else np.asarray([0.55, 0.55, 0.55], dtype=np.float32)
            self.model.vis.headlight.diffuse[:] = np.asarray([0.82, 0.82, 0.82], dtype=np.float32) if v2_profile else np.asarray([0.70, 0.70, 0.70], dtype=np.float32)
            self.model.vis.headlight.specular[:] = np.asarray([0.26, 0.26, 0.26], dtype=np.float32) if v2_profile else np.asarray([0.20, 0.20, 0.20], dtype=np.float32)

        if self.contract.material_policy == "handle_highlight":
            centers = np.asarray(self.data.geom_xpos, dtype=np.float32)
            if len(centers):
                distances = np.linalg.norm(centers - handle[None, :], axis=1)
                nearest = int(np.argmin(distances))
                scale = np.asarray([0.78, 0.78, 0.78, 1.0], dtype=np.float32) if v2_profile else np.asarray([0.95, 0.95, 0.95, 1.0], dtype=np.float32)
                self.model.geom_rgba[:] = np.clip(self.model.geom_rgba[:] * scale, 0.0, 1.0)
                self.model.geom_rgba[nearest, :4] = np.asarray([0.98, 0.66, 0.16, 1.0], dtype=np.float32) if v2_profile else np.asarray([0.86, 0.54, 0.24, 1.0], dtype=np.float32)
                near_mask = distances < (0.16 if v2_profile else 0.12)
                if v2_profile:
                    self.model.geom_rgba[near_mask, :3] = np.clip(self.model.geom_rgba[near_mask, :3] * 1.35, 0.0, 1.0)
                else:
                    self.model.geom_rgba[near_mask, :3] = np.clip(self.model.geom_rgba[near_mask, :3] * 1.10, 0.0, 1.0)

    def _render(self, camera) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        rgb = self.renderer.render()
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def _camera_pose_metadata(self, name: str, camera, eye: np.ndarray, lookat: np.ndarray, relative_transform: np.ndarray | None = None) -> dict[str, Any]:
        t_world_cam = _transform_from_eye_lookat(eye, lookat)
        t_world_eef = np.eye(4, dtype=np.float32)
        t_world_eef[:3, :3] = R.from_quat(np.asarray(self.eef_quat, dtype=np.float32)).as_matrix().astype(np.float32)
        t_world_eef[:3, 3] = np.asarray(self.eef_pos, dtype=np.float32)
        if relative_transform is None:
            t_eef_cam = _transform_inverse(t_world_eef) @ t_world_cam
        else:
            t_eef_cam = relative_transform.astype(np.float32)
        translation_residual = float(np.linalg.norm(t_eef_cam[:3, 3] - WRIST_CAM_OFFSET_LOCAL)) if name == "secondary" else None
        return {
            "name": name,
            "mode": self.contract.secondary_camera_mode if name == "secondary" else "fixed_primary_scene",
            "lookat": np.asarray(lookat, dtype=np.float32),
            "distance": float(camera.distance),
            "azimuth": float(camera.azimuth),
            "elevation": float(camera.elevation),
            "T_world_cam": t_world_cam,
            "T_eef_cam": t_eef_cam,
            "eef_cam_relativeness_residual": translation_residual,
        }

    def _set_secondary_camera(self) -> dict[str, Any]:
        if self.contract.secondary_camera_mode == "legacy_fixed_scene":
            eye = _camera_eye_from_lookat(self.cam_secondary.lookat.copy(), self.cam_secondary.distance, self.cam_secondary.azimuth, self.cam_secondary.elevation)
            return self._camera_pose_metadata("secondary", self.cam_secondary, eye, self.cam_secondary.lookat.copy())

        rotation = R.from_quat(np.asarray(self.eef_quat, dtype=np.float32))
        cam_pos_world = self.eef_pos + rotation.apply(WRIST_CAM_OFFSET_LOCAL).astype(np.float32)
        lookat_world = self.eef_pos + rotation.apply(WRIST_LOOK_LOCAL).astype(np.float32)
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
        relative[:3, 3] = WRIST_CAM_OFFSET_LOCAL.astype(np.float32)
        return self._camera_pose_metadata("secondary", self.cam_secondary, cam_pos_world, lookat_world, relative_transform=relative)

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

    def _crop_statistics(self, image: np.ndarray, bbox: list[int]) -> tuple[float, float, float]:
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

    def _compute_handle_probe_metadata(self, primary: np.ndarray, secondary: np.ndarray, secondary_meta: dict[str, Any]) -> dict[str, Any]:
        bbox_primary = self._handle_bbox_for_camera("primary")
        bbox_secondary = self._handle_bbox_for_camera("secondary")
        vis_primary, contrast_primary, entropy_primary = self._crop_statistics(primary, bbox_primary)
        vis_secondary, contrast_secondary, entropy_secondary = self._crop_statistics(secondary, bbox_secondary)
        residual_value = secondary_meta.get("eef_cam_relativeness_residual")
        residual = float(1.0 if residual_value is None else residual_value)
        center_x = (bbox_secondary[0] + bbox_secondary[2]) / 2.0
        center_y = (bbox_secondary[1] + bbox_secondary[3]) / 2.0
        framing_residual = float(np.mean([abs(center_x / max(self.image_size - 1, 1) - 0.5), abs(center_y / max(self.image_size - 1, 1) - 0.5)]))
        secondary_framing_score = float(np.clip(1.0 - 2.2 * framing_residual, 0.0, 1.0))
        return {
            "handle_bbox_primary": bbox_primary,
            "handle_bbox_secondary": bbox_secondary,
            "handle_visibility_fraction_primary": float(vis_primary),
            "handle_visibility_fraction_secondary": float(vis_secondary),
            "handle_visibility_fraction": float((vis_primary + vis_secondary) / 2.0),
            "handle_local_contrast_primary": float(contrast_primary),
            "handle_local_contrast_secondary": float(contrast_secondary),
            "handle_local_contrast": float(max(contrast_primary, contrast_secondary)),
            "handle_crop_entropy_primary": float(entropy_primary),
            "handle_crop_entropy_secondary": float(entropy_secondary),
            "handle_crop_entropy": float(max(entropy_primary, entropy_secondary)),
            "secondary_framing_score": secondary_framing_score,
            "camera_relativeness_residual": residual,
            "render_profile": self.contract.render_profile,
            "background_mode": self.contract.background_mode,
            "lighting_profile": self.contract.lighting_profile,
            "material_policy": self.contract.material_policy,
            "camera_framing_profile": self.contract.camera_framing_profile,
        }

    def _observe_images(self) -> tuple[np.ndarray, np.ndarray]:
        self._apply_visual_profile()
        primary_eye = _camera_eye_from_lookat(self.cam_primary.lookat.copy(), self.cam_primary.distance, self.cam_primary.azimuth, self.cam_primary.elevation)
        primary_meta = self._camera_pose_metadata("primary", self.cam_primary, primary_eye, self.cam_primary.lookat.copy())
        secondary_meta = self._set_secondary_camera()

        img1 = self._render(self.cam_primary)
        img2 = self._render(self.cam_secondary)
        img1 = self._apply_calibration(img1, secondary=False)
        img2 = self._apply_calibration(img2, secondary=True)
        if self.contract.enable_marker_overlay:
            color = (0, 220, 32) if self._attached else (220, 60, 20)
            img1 = self._overlay_marker(img1, self._project_primary(self.eef_pos), color)
            img2 = self._overlay_marker(img2, self._project_secondary(self.eef_pos), color)

        handle_probe = self._compute_handle_probe_metadata(img1, img2, secondary_meta) if self.contract.emit_handle_probe_metadata else {}
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
            "diagnostic_only": (not self.contract.canonical_lane) or self.contract.calibration_mode == "diagnostic_texture",
        }
        self._last_handle_probe_metadata = _json_ready(handle_probe)
        self._last_claim_policy = self.claim_policy()
        return img1, img2

    def _synthetic_motor_state(self) -> np.ndarray:
        handle = self._handle_center_world()
        rel = handle - self.eef_pos
        scale = np.array([0.22, 0.18, 0.14], dtype=np.float32)
        rel = np.clip(rel / scale, -1.0, 1.0)
        return np.array([rel[0], rel[1], rel[2], self._drawer_fraction_signed()], dtype=np.float32)

    def state_spec(self) -> dict[str, Any]:
        mode = self.contract.state_mode
        if mode == "m0_proxy":
            dim_names = [
                "eef_pos_x_m",
                "eef_pos_y_m",
                "eef_pos_z_m",
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
                "eef_pos_x_m",
                "eef_pos_y_m",
                "eef_pos_z_m",
                "eef_quat_x",
                "eef_quat_y",
                "eef_quat_z",
                "eef_quat_w",
                "gripper_joint",
            ]
            provenance = ["observed"] * 8
        elif mode == "telemetry_candidate_v1":
            dim_names = [
                "eef_pos_x_m",
                "eef_pos_y_m",
                "eef_pos_z_m",
                "eef_rotvec_x",
                "eef_rotvec_y",
                "eef_rotvec_z",
                "gripper_joint",
                "drawer_fraction_signed",
            ]
            provenance = ["observed", "observed", "observed", "derived", "derived", "derived", "observed", "derived"]
        else:
            dim_names = [
                "eef_pos_x_m",
                "eef_pos_y_m",
                "eef_pos_z_m",
                "eef_rotvec_x",
                "eef_rotvec_y",
                "eef_rotvec_z",
                "handle_distance_norm",
                "drawer_fraction_signed",
            ]
            provenance = ["observed", "observed", "observed", "derived", "derived", "derived", "derived", "derived"]
        duplicate_dims = sorted({name for name in dim_names if dim_names.count(name) > 1})
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
        rotvec = R.from_quat(np.asarray(self.eef_quat, dtype=np.float32)).as_rotvec().astype(np.float32)
        if mode == "telemetry_candidate_v1":
            state = np.concatenate(
                [
                    self.eef_pos.astype(np.float32),
                    rotvec,
                    np.array([self.gripper_joint, self._drawer_fraction_signed()], dtype=np.float32),
                ]
            )
            return state.astype(np.float32)
        handle_distance = float(np.linalg.norm(self._handle_center_world() - self.eef_pos))
        handle_distance_norm = np.clip(handle_distance / 0.35, 0.0, 1.0)
        state = np.concatenate(
            [
                self.eef_pos.astype(np.float32),
                rotvec,
                np.array([handle_distance_norm, self._drawer_fraction_signed()], dtype=np.float32),
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
        distance_pass = dist_to_handle <= ATTACH_THRESHOLD_M
        orientation_alignment_cos = 1.0
        orientation_gate_passed = True
        orientation_error_rad = 0.0
        orientation_weight = 1.0
        attach_eligible = False

        if self.contract.interaction_mode == "orientation_sensitive_v1":
            local_pull_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            pull_axis_world = R.from_quat(np.asarray(self.eef_quat, dtype=np.float32)).apply(local_pull_axis).astype(np.float32)
            pull_axis_world = _normalize(pull_axis_world, WORLD_X)
            orientation_alignment_cos = float(np.clip(np.dot(pull_axis_world, self._motion_axis), -1.0, 1.0))
            orientation_error_rad = float(np.arccos(np.clip(orientation_alignment_cos, -1.0, 1.0)))
            attach_cos_threshold = 0.55
            open_cos_threshold = 0.65
            orientation_gate_passed = orientation_alignment_cos >= attach_cos_threshold
            orientation_weight = float(np.clip((orientation_alignment_cos - open_cos_threshold) / max(1e-6, 1.0 - open_cos_threshold), 0.0, 1.0))
        attach_eligible = bool(close_cmd and distance_pass and orientation_gate_passed)

        if not self._attached and attach_eligible:
            self._attached = True
        if self._attached and (not close_cmd or dist_to_handle > DETACH_THRESHOLD_M):
            self._attached = False

        drawer_delta_raw = 0.0
        drawer_delta_effective = 0.0
        if self._attached:
            drawer_delta_raw = float(np.dot(self.eef_pos - prev_pos, self._motion_axis)) * 6.0
            drawer_delta_effective = drawer_delta_raw if self.contract.interaction_mode == "legacy_translation_only" else drawer_delta_raw * orientation_weight
            low, high = self.joint_range.tolist()
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
            "drawer_delta_raw": float(drawer_delta_raw),
            "drawer_delta_effective": float(drawer_delta_effective),
            "open_orientation_weight": float(orientation_weight),
            "orientation_error_rad": float(orientation_error_rad),
        }
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
            "orientation_alignment_cos": float(orientation_alignment_cos),
            "orientation_gate_passed": bool(orientation_gate_passed),
            "attach_eligible": bool(attach_eligible),
            "attach_gate_distance_passed": bool(distance_pass),
            "attach_gate_orientation_passed": bool(orientation_gate_passed),
            "drawer_delta_raw": float(drawer_delta_raw),
            "drawer_delta_effective": float(drawer_delta_effective),
            "open_orientation_weight": float(orientation_weight),
            "orientation_error_rad": float(orientation_error_rad),
            "claim_policy": self.claim_policy(),
            "camera_metadata": _json_ready(self._last_camera_metadata),
            "visual_mode_report": _json_ready(self._last_visual_mode_report),
            "state_spec": _json_ready(self.state_spec()),
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


def _safe_normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.asarray(fallback, dtype=np.float32).copy()
    return (vec / norm).astype(np.float32)


def _target_quat(current_pos: np.ndarray, target_pos: np.ndarray, motion_axis: np.ndarray) -> np.ndarray:
    z_axis = _safe_normalize(target_pos - current_pos, WORLD_UP)
    motion = _safe_normalize(motion_axis, WORLD_X)
    x_axis = motion - np.dot(motion, z_axis) * z_axis
    x_axis = _safe_normalize(x_axis, np.cross(WORLD_UP, z_axis) if abs(float(np.dot(WORLD_UP, z_axis))) < 0.95 else WORLD_X)
    if float(np.linalg.norm(x_axis)) < 1e-6:
        x_axis = WORLD_X.copy()
    y_axis = _safe_normalize(np.cross(z_axis, x_axis), np.cross(z_axis, WORLD_X))
    x_axis = _safe_normalize(np.cross(y_axis, z_axis), x_axis)
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
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
) -> np.ndarray:
    delta = np.zeros(7, dtype=np.float32)
    pos_err = target_pos - current_pos
    delta[:3] = np.clip((pos_err / max(TRANSLATION_SCALE_M, 1e-6)) * float(speed), -1.0, 1.0)
    if rotation_source == "aligned" and current_quat is not None and motion_axis is not None:
        target_quat = _target_quat(current_pos, target_pos, motion_axis)
        current_rot = R.from_quat(np.asarray(current_quat, dtype=np.float32))
        target_rot = R.from_quat(target_quat)
        rotvec = (target_rot * current_rot.inv()).as_rotvec().astype(np.float32)
        phase_gain = 0.55 if phase in {"pregrasp", "contact"} else (0.85 if phase == "close" else 1.0)
        delta[3:6] = np.clip((rotvec / max(ROTATION_SCALE_RAD, 1e-6)) * phase_gain, -ROT_ACTION_CLIP, ROT_ACTION_CLIP)
    elif rotation_source == "random":
        rand = (rng or np.random.default_rng()).normal(loc=0.0, scale=CONTROL_ROT_STD).astype(np.float32)
        delta[3:6] = np.clip(rand / max(ROTATION_SCALE_RAD, 1e-6), -ROT_ACTION_CLIP, ROT_ACTION_CLIP)
    delta[6] = -1.0 if close else 1.0
    return delta.astype(np.float32)


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
) -> dict[str, Any]:
    env = DrawerRobotEnvMuJoCo(seed=seed, image_size=image_size, max_steps=max_steps, contract=contract)
    obs = env.reset()
    axis = env._motion_axis
    handle = env._handle_center_world()
    pregrasp_pose = _pose_from_point(handle, axis, offset=0.04, z_lift=0.03)
    grasp_pose = np.asarray(grasp_pose_world, dtype=np.float32).copy()
    grasp_pose[:3, 3] = handle
    open_fraction = float(np.clip(pull_open_fraction, 0.75, 0.95))
    retreat_target = handle + axis * 0.16 + np.array([0.0, 0.0, 0.05], dtype=np.float32)
    rng = np.random.default_rng(seed + episode_index)

    images, images2, states, actions, rewards = [], [], [], [], []
    abs_drawer, next_drawer, attached_trace, handle_distance_trace = [], [], [], []
    phase_labels = []
    eef_quat_trace, orientation_alignment_trace, attach_eligible_trace = [], [], []
    orientation_gate_trace, drawer_delta_raw_trace, drawer_delta_effective_trace = [], [], []
    orientation_error_trace = []
    camera_metadata_trace = []
    handle_probe_metadata_trace = []
    ever_attached = False
    attach_step = None
    success = False
    close_hold_steps = 0

    phase = "pregrasp"
    try:
        for step_idx in range(max_steps):
            images.append(obs.image.copy())
            images2.append(obs.image2.copy())
            states.append(obs.state.copy())
            abs_drawer.append(float(obs.drawer_fraction))
            eef_quat_trace.append(np.asarray(obs.eef_quat, dtype=np.float32).copy())
            camera_metadata_trace.append(_json_ready(obs.camera_metadata or {}))
            handle_probe_metadata_trace.append(_json_ready(obs.handle_probe_metadata or {}))

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
                target_pos, close, speed = pregrasp_target, False, 0.65
            elif phase == "contact":
                target_pos, close, speed = contact_target, False, 0.45
            elif phase == "close":
                target_pos, close, speed = contact_target, True, 0.25
            elif phase == "pull":
                target_pos, close, speed = pull_target, True, 0.30
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
            )
            next_obs, reward, done, info = env.step(action)
            actions.append(action.copy())
            rewards.append(float(reward))
            phase_labels.append(phase)
            attached_trace.append(bool(info["attached"]))
            handle_distance_trace.append(float(info["dist_to_handle"]))
            next_drawer.append(float(next_obs.drawer_fraction))
            orientation_alignment_trace.append(float(info.get("orientation_alignment_cos", 1.0)))
            attach_eligible_trace.append(bool(info.get("attach_eligible", False)))
            orientation_gate_trace.append(bool(info.get("orientation_gate_passed", True)))
            drawer_delta_raw_trace.append(float(info.get("drawer_delta_raw", 0.0)))
            drawer_delta_effective_trace.append(float(info.get("drawer_delta_effective", 0.0)))
            orientation_error_trace.append(float(info.get("orientation_error_rad", 0.0)))
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
        "orientation_alignment_trace": np.asarray(orientation_alignment_trace, dtype=np.float32),
        "attach_eligible_trace": np.asarray(attach_eligible_trace, dtype=np.bool_),
        "orientation_gate_trace": np.asarray(orientation_gate_trace, dtype=np.bool_),
        "drawer_delta_raw_trace": np.asarray(drawer_delta_raw_trace, dtype=np.float32),
        "drawer_delta_effective_trace": np.asarray(drawer_delta_effective_trace, dtype=np.float32),
        "orientation_error_trace": np.asarray(orientation_error_trace, dtype=np.float32),
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
        "effective_training_frames": rollout.get("effective_training_frames", rollout["steps"]),
    }
    write_text_atomic(path.with_suffix(".json"), json.dumps(_json_ready(meta), indent=2, ensure_ascii=False) + "\n")

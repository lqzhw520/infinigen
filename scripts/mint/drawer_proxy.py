#!/usr/bin/env python3
"""Simple MuJoCo drawer proxy environment and rollout helpers for MINT integration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
DRAWER_ROOT = PROJECT_ROOT / "sim_exports" / "urdf" / "drawerbox"


def drawer_dir(seed: int) -> Path:
    return DRAWER_ROOT / str(seed)


def drawer_assets_available(seeds: list[int]) -> bool:
    return all((drawer_dir(seed) / "drawerbox.urdf").exists() for seed in seeds)


def drawer_manifest() -> dict[str, Any]:
    return {
        "available_seeds": sorted(
            int(p.name) for p in DRAWER_ROOT.iterdir() if p.is_dir()
        ),
        "root": str(DRAWER_ROOT),
    }


def _load_assets(seed: int) -> tuple[str, dict[str, bytes]]:
    urdf_dir = drawer_dir(seed)
    urdf_path = urdf_dir / "drawerbox.urdf"
    urdf_text = urdf_path.read_text()
    assets: dict[str, bytes] = {}
    assets_dir = urdf_dir / "assets"
    for asset_path in sorted(assets_dir.iterdir()):
        if asset_path.is_file():
            assets[asset_path.name] = asset_path.read_bytes()
    return urdf_text, assets


def _make_camera(lookat: np.ndarray, distance: float, azimuth: float, elevation: float):
    import mujoco

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat.tolist()
    camera.distance = float(distance)
    camera.azimuth = float(azimuth)
    camera.elevation = float(elevation)
    return camera


@dataclass
class DrawerProxyObservation:
    image: np.ndarray
    image2: np.ndarray
    state: np.ndarray
    task: str
    joint_position: float
    normalized_joint_position: float


class DrawerProxyEnv:
    """A minimal articulated-drawer environment for MINT dataset generation and eval."""

    def __init__(self, seed: int, image_size: int = 224, max_steps: int = 32):
        import mujoco

        urdf_text, assets = _load_assets(seed)
        self.model = mujoco.MjModel.from_xml_string(urdf_text, assets)
        self.data = mujoco.MjData(self.model)
        self.gl_context = mujoco.GLContext(image_size, image_size)
        self.gl_context.make_current()
        self.seed = seed
        self.image_size = int(image_size)
        self.max_steps = int(max_steps)
        self.renderer = mujoco.Renderer(
            self.model, height=self.image_size, width=self.image_size
        )
        self.joint_idx = 0
        self.joint_range = self.model.jnt_range[self.joint_idx].astype(np.float32)
        center = np.array(self.model.stat.center, dtype=np.float32)
        extent = float(max(self.model.stat.extent, 0.25))
        self.cam_primary = _make_camera(
            center, distance=extent * 1.8, azimuth=180.0, elevation=-25.0
        )
        self.cam_secondary = _make_camera(
            center, distance=extent * 1.4, azimuth=110.0, elevation=-35.0
        )
        self._step_count = 0
        self.task = "open the drawer"
        self.reset()

    def close(self) -> None:
        try:
            self.renderer.close()
        finally:
            try:
                if getattr(self, "gl_context", None) is not None:
                    self.gl_context.free()
            except Exception:
                pass

    def _render(self, camera) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        rgb = self.renderer.render()
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def _state_vector(self) -> np.ndarray:
        low, high = self.joint_range.tolist()
        span = max(high - low, 1e-6)
        joint = float(self.data.qpos[self.joint_idx])
        joint_vel = float(self.data.qvel[self.joint_idx]) if self.model.nv else 0.0
        norm_pos = ((joint - low) / span) * 2.0 - 1.0
        remaining = (high - joint) / span
        state = np.array(
            [
                norm_pos,
                np.clip(joint_vel / 0.05, -1.0, 1.0),
                np.clip(remaining * 2.0 - 1.0, -1.0, 1.0),
                np.clip(float(self._step_count) / max(self.max_steps, 1), 0.0, 1.0)
                * 2.0
                - 1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )
        return state

    def observe(self) -> DrawerProxyObservation:
        low, high = self.joint_range.tolist()
        joint = float(self.data.qpos[self.joint_idx])
        span = max(high - low, 1e-6)
        return DrawerProxyObservation(
            image=self._render(self.cam_primary),
            image2=self._render(self.cam_secondary),
            state=self._state_vector(),
            task=self.task,
            joint_position=joint,
            normalized_joint_position=(joint - low) / span,
        )

    def reset(self, start_fraction: float = 0.0) -> DrawerProxyObservation:
        import mujoco

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        low, high = self.joint_range.tolist()
        self.data.qpos[self.joint_idx] = low + (high - low) * float(start_fraction)
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self.observe()

    def step(
        self, action: np.ndarray
    ) -> tuple[DrawerProxyObservation, float, bool, dict[str, Any]]:
        import mujoco

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        delta_norm = float(np.clip(action[0], -1.0, 1.0))
        low, high = self.joint_range.tolist()
        span = max(high - low, 1e-6)
        step_scale = span / 10.0
        self.data.qpos[self.joint_idx] = np.clip(
            float(self.data.qpos[self.joint_idx]) + delta_norm * step_scale, low, high
        )
        self._step_count += 1
        mujoco.mj_forward(self.model, self.data)
        obs = self.observe()
        success = obs.normalized_joint_position >= 0.9
        reward = 1.0 if success else float(obs.normalized_joint_position)
        done = success or self._step_count >= self.max_steps
        info = {
            "is_success": success,
            "joint_position": obs.joint_position,
            "normalized_joint_position": obs.normalized_joint_position,
            "step_count": self._step_count,
        }
        return obs, reward, done, info


def expert_policy(obs: DrawerProxyObservation) -> np.ndarray:
    delta = np.array([0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if obs.normalized_joint_position >= 0.9:
        delta[:] = 0.0
    return delta


def random_policy(rng: np.random.Generator) -> np.ndarray:
    action = rng.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)
    return action


def rollout_episode(
    seed: int, policy: str = "expert", max_steps: int = 32, rng_seed: int | None = None
) -> dict[str, Any]:
    env = DrawerProxyEnv(seed=seed, max_steps=max_steps)
    rng = np.random.default_rng(rng_seed if rng_seed is not None else seed)
    obs = env.reset(start_fraction=0.0)

    images = []
    images2 = []
    states = []
    abs_targets = []
    deltas = []
    rewards = []
    success = False

    low, high = env.joint_range.tolist()
    span = max(high - low, 1e-6)
    try:
        for step_idx in range(max_steps):
            images.append(obs.image.copy())
            images2.append(obs.image2.copy())
            states.append(obs.state.copy())
            abs_targets.append(float(obs.joint_position))
            if policy == "expert":
                action = expert_policy(obs)
            elif policy == "random":
                action = random_policy(rng)
            else:
                raise ValueError(f"Unsupported policy: {policy}")
            deltas.append(action.copy())
            obs, reward, done, info = env.step(action)
            rewards.append(float(reward))
            success = bool(info["is_success"])
            if done:
                break
        return {
            "seed": seed,
            "policy": policy,
            "success": success,
            "steps": len(deltas),
            "images": np.stack(images, axis=0),
            "images2": np.stack(images2, axis=0),
            "states": np.stack(states, axis=0),
            "absolute_joint_positions": np.asarray(abs_targets, dtype=np.float32),
            "delta_actions": np.stack(deltas, axis=0),
            "scale_m": span / 10.0,
            "rewards": np.asarray(rewards, dtype=np.float32),
            "task": env.task,
        }
    finally:
        env.close()


def save_rollout(path: Path, rollout: dict[str, Any]) -> None:
    arrays = {
        "images": rollout["images"],
        "images2": rollout["images2"],
        "states": rollout["states"],
        "absolute_joint_positions": rollout["absolute_joint_positions"],
        "delta_actions": rollout["delta_actions"],
        "rewards": rollout["rewards"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    meta = {
        "seed": rollout["seed"],
        "policy": rollout["policy"],
        "success": rollout["success"],
        "steps": rollout["steps"],
        "scale_m": float(rollout["scale_m"]),
        "task": rollout["task"],
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

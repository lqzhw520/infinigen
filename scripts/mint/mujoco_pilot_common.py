#!/usr/bin/env python3
"""Shared helpers for the LIBERO / MuJoCo drawer pilot."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1" / "artifacts"
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1" / "outputs"

LIBERO_REPO_ROOT = PROJECT_ROOT / "external" / "LIBERO"
MINT_REPO_ROOT = PROJECT_ROOT / "external" / "MINT"
MINT_POLICY_SRC = MINT_REPO_ROOT / "lerobot_policy_mint" / "src"
MINT_CKPT = MINT_REPO_ROOT / "checkpoints" / "MINT-libero"
DATASET_ROOT = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1" / "dataset"

BENCHMARK_NAME = "libero_90"
TASK_NAME = "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet"
DEFAULT_TASK_TEXT = "open the bottom drawer of the cabinet"

LIBERO_GRIPPER_MIN = -0.042
LIBERO_GRIPPER_MAX = 0.001


@dataclass(frozen=True)
class TaskSpec:
    benchmark_name: str
    task_name: str
    task_index: int
    bddl_file: str
    init_states_path: str
    task_text: str


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def configure_libero_runtime() -> None:
    """Configure a LIBERO / MuJoCo runtime that works on the current A800."""

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    libero_root = str(LIBERO_REPO_ROOT)
    mint_src_root = str(MINT_POLICY_SRC)
    if libero_root not in sys.path:
        sys.path.insert(0, libero_root)
    if mint_src_root not in sys.path:
        sys.path.insert(0, mint_src_root)


def _libero_imports():
    configure_libero_runtime()
    from libero.libero import get_libero_path  # type: ignore
    from libero.libero.benchmark import get_benchmark  # type: ignore
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv  # type: ignore

    return get_benchmark, get_libero_path, OffScreenRenderEnv


def load_task_spec() -> TaskSpec:
    get_benchmark, get_libero_path, _ = _libero_imports()
    benchmark = get_benchmark(BENCHMARK_NAME)()
    task_names = benchmark.get_task_names()
    if TASK_NAME not in task_names:
        raise KeyError(f"{TASK_NAME!r} not found in {BENCHMARK_NAME}")
    task_index = task_names.index(TASK_NAME)
    task = benchmark.get_task(task_index)
    bddl_file = benchmark.get_task_bddl_file_path(task_index)
    init_states_path = str(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file)
    task_text = getattr(task, "language", DEFAULT_TASK_TEXT) or DEFAULT_TASK_TEXT
    return TaskSpec(
        benchmark_name=BENCHMARK_NAME,
        task_name=TASK_NAME,
        task_index=task_index,
        bddl_file=str(bddl_file),
        init_states_path=init_states_path,
        task_text=task_text,
    )


def load_init_states(task_spec: TaskSpec) -> list[np.ndarray]:
    init_states = torch.load(task_spec.init_states_path, weights_only=False)
    return [np.asarray(state, dtype=np.float64) for state in init_states]


def make_env(task_spec: TaskSpec, *, image_size: int) -> Any:
    _, _, OffScreenRenderEnv = _libero_imports()
    return OffScreenRenderEnv(
        bddl_file_name=task_spec.bddl_file,
        camera_heights=image_size,
        camera_widths=image_size,
    )


def map_gripper_joint(finger_joint: float) -> float:
    value = finger_joint * 1.075 - 0.042
    return float(np.clip(value, LIBERO_GRIPPER_MIN, LIBERO_GRIPPER_MAX))


def build_state(obs: dict[str, np.ndarray], *, joint_mode: str) -> np.ndarray:
    if joint_mode not in {"raw_joint_pos", "pi_minus_joint_pos"}:
        raise ValueError(f"Unsupported joint_mode: {joint_mode}")
    joints = obs["robot0_joint_pos"][:4].astype(np.float32)
    if joint_mode == "pi_minus_joint_pos":
        joints = (np.pi - joints).astype(np.float32)
    gripper = np.array([map_gripper_joint(float(obs["robot0_gripper_qpos"][0]))], dtype=np.float32)
    return np.concatenate([obs["robot0_eef_pos"].astype(np.float32), joints, gripper], axis=0)


def build_batch(obs: dict[str, np.ndarray], *, state: np.ndarray, task_text: str) -> dict[str, Any]:
    return {
        "observation.images.image": torch.from_numpy(obs["agentview_image"]).permute(2, 0, 1).to(torch.float32) / 255.0,
        "observation.images.image2": torch.from_numpy(obs["robot0_eye_in_hand_image"]).permute(2, 0, 1).to(torch.float32) / 255.0,
        "observation.state": torch.from_numpy(state.astype(np.float32)),
        "task": task_text,
    }


def load_mint_policy():
    configure_libero_runtime()
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot_policy_mint.modeling_mint import MINTPolicy

    policy = MINTPolicy.from_pretrained(str(MINT_CKPT), local_files_only=True)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(MINT_CKPT))
    return policy, preprocessor, postprocessor


def bottom_drawer_joint_info(env: Any) -> dict[str, Any]:
    model = env.sim.model
    for joint_id in range(model.njnt):
        joint_name = model.joint(joint_id).name
        if joint_name and joint_name.endswith("_bottom_level"):
            joint_range = model.jnt_range[joint_id]
            return {
                "joint_id": int(joint_id),
                "joint_name": joint_name,
                "qpos_adr": int(model.jnt_qposadr[joint_id]),
                "open_qpos": float(joint_range[0]),
                "closed_qpos": float(joint_range[1]),
                "joint_range": [float(joint_range[0]), float(joint_range[1])],
            }
    raise KeyError("Bottom drawer joint not found")


def drawer_fraction(env: Any) -> float:
    info = bottom_drawer_joint_info(env)
    qpos = float(env.sim.data.qpos[info["qpos_adr"]])
    denom = info["closed_qpos"] - info["open_qpos"]
    if abs(denom) < 1e-8:
        return 0.0
    value = (info["closed_qpos"] - qpos) / denom
    return float(np.clip(value, 0.0, 1.0))


def refresh_observations(env: Any) -> dict[str, np.ndarray]:
    env.check_success()
    env._post_process()
    env._update_observables(force=True)
    return env.env._get_observations()


def set_drawer_qpos(env: Any, qpos: float) -> dict[str, np.ndarray]:
    info = bottom_drawer_joint_info(env)
    env.sim.data.qpos[info["qpos_adr"]] = qpos
    env.sim.forward()
    return refresh_observations(env)


def sample_pybullet_dataset_states(limit: int = 100) -> np.ndarray:
    import pyarrow.parquet as pq

    parquet_files = sorted((DATASET_ROOT / "data").glob("chunk-*/*.parquet"))
    states: list[np.ndarray] = []
    for parquet_path in parquet_files:
        table = pq.read_table(str(parquet_path), columns=["observation.state"])
        column = table.column("observation.state")
        for i in range(len(column)):
            states.append(np.asarray(column[i].as_py(), dtype=np.float32))
            if len(states) >= limit:
                return np.stack(states)
    if not states:
        raise FileNotFoundError(f"No parquet states found under {DATASET_ROOT}")
    return np.stack(states)


def summarize_array(name: str, values: np.ndarray) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(values.shape),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
    }


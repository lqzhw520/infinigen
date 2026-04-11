#!/usr/bin/env python3
"""P0C: evaluate rollout-path parity before semantic redesign."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.scripts.lerobot_eval import OBS_STR
from lerobot.utils.constants import ACTION

from mint_common import now_iso, write_json_atomic, write_text_atomic
from mujoco_mainline_common import default_heldout_seeds
from p1_execution_common import (
    ARTIFACT_DIR,
    CONTROL_SEED,
    OUTPUT_DIR,
    close_bundle,
    get_single_vec_env,
    infer_libero_drawer_fraction,
    make_official_control_bundle,
    upsert_evidence,
)
from run_mujoco_infinigen_mainline_night import _load_policy_bundle, _obs_to_batch
from drawer_robot_env_mujoco import DrawerRobotEnvMuJoCo

ARTIFACT_PATH = ARTIFACT_DIR / "p1l_eval_parity_matrix.json"
REPORT_PATH = OUTPUT_DIR / "p1l_eval_parity_matrix.md"
VIDEO_ROOT = OUTPUT_DIR / "p1l_eval_parity_matrix"
EVIDENCE_ID = "E054"
EXPERIMENT_ID = "p1l_eval_parity_matrix"
PARITY_EPISODES = 2
MUJOCO_SEEDS = default_heldout_seeds()[:PARITY_EPISODES]


def _tensor_image_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().numpy()
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.max() <= 1.0:
        image = image * 255.0
    image = np.clip(image, 0.0, 255.0)
    return image.astype(np.uint8)


def _extract_bool(info: dict[str, Any], key: str) -> bool:
    if key in info:
        value = info[key]
        if isinstance(value, (list, tuple, np.ndarray)):
            return bool(value[0])
        return bool(value)
    final_info = info.get("final_info")
    if isinstance(final_info, dict) and key in final_info:
        value = final_info[key]
        if isinstance(value, (list, tuple, np.ndarray)):
            return bool(value[0])
        return bool(value)
    return False


def _write_video(path: Path, frames: list[np.ndarray], fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        imageio.mimsave(path, frames, fps=fps)


def _action_stats(action_trace: np.ndarray) -> dict[str, Any]:
    if action_trace.size == 0:
        return {"mean": [], "std": []}
    return {
        "mean": np.mean(action_trace, axis=0).round(6).tolist(),
        "std": np.std(action_trace, axis=0).round(6).tolist(),
    }


def _official_rollout(n_action_steps: int, *, episode_seed: int, video_dir: Path) -> dict[str, Any]:
    bundle = make_official_control_bundle(
        n_action_steps=n_action_steps,
        n_episodes=1,
        device="cuda",
        seed=episode_seed,
        output_dir=video_dir,
    )
    vec_env = get_single_vec_env(bundle)
    policy = bundle["policy"]
    env_pre = bundle["env_preprocessor"]
    env_post = bundle["env_postprocessor"]
    pre = bundle["preprocessor"]
    post = bundle["postprocessor"]
    frames = []
    drawer_trace = []
    action_trace = []
    steps = 0
    success = False
    try:
        policy.reset()
        observation, _info = vec_env.reset(seed=[episode_seed])
        max_steps = int(vec_env.call("_max_episode_steps")[0])
        for _ in range(max_steps):
            observation_t = preprocess_observation(observation)
            observation_t = add_envs_task(vec_env, observation_t)
            observation_env = env_pre(observation_t)
            # Capture RAW image before preprocessing
            raw_image = observation.get("images", {}).get("image")
            if raw_image is not None and isinstance(raw_image, torch.Tensor):
                frames.append(_tensor_image_to_uint8(raw_image[0]))
            else:
                frames.append(_tensor_image_to_uint8(observation_env[f"{OBS_STR}.images.image"][0]))
            processed = pre(observation_env)
            with torch.inference_mode():
                action_raw = policy.select_action(processed)
            action_post = post(action_raw)
            action_transition = env_post({ACTION: action_post})
            action_sent = action_transition[ACTION]
            observation, reward, terminated, truncated, info = vec_env.step(action_sent.detach().cpu().numpy())
            drawer_trace.append(float(infer_libero_drawer_fraction(vec_env).get("fraction") or 0.0))
            action_trace.append(action_sent[0].detach().cpu().numpy().astype(np.float32))
            success = _extract_bool(info, "is_success")
            steps += 1
            if bool(terminated[0] or truncated[0]):
                break
        video_path = video_dir / f"official_seed_{episode_seed:03d}.mp4"
        _write_video(video_path, frames)
        return {
            "seed": episode_seed,
            "success": success,
            "steps": steps,
            "max_drawer_fraction": float(max(drawer_trace)) if drawer_trace else 0.0,
            "final_drawer_fraction": float(drawer_trace[-1]) if drawer_trace else 0.0,
            "action_stats": _action_stats(np.asarray(action_trace, dtype=np.float32)),
            "video_path": str(video_path),
        }
    finally:
        close_bundle(bundle)


def _override_policy_n_action_steps(policy_bundle, n_action_steps: int):
    policy, pre, post, dataset = policy_bundle
    from lerobot_policy_mint.mint_utils import IntentionEnsembler

    policy.config.n_action_steps = int(n_action_steps)
    policy.ensembler = IntentionEnsembler(
        horizon=policy.config.chunk_size,
        n_action_steps=int(n_action_steps),
    )
    policy.reset()
    return policy, pre, post, dataset


def _mujoco_rollout(policy_bundle, *, seed: int, n_action_steps: int, video_dir: Path) -> dict[str, Any]:
    policy, pre, post, _dataset = _override_policy_n_action_steps(policy_bundle, n_action_steps)
    env = DrawerRobotEnvMuJoCo(seed=seed, image_size=256, max_steps=96)
    obs = env.reset()
    frames = []
    drawer_trace = []
    action_trace = []
    success = False
    steps = 0
    try:
        for _ in range(96):
            frames.append(obs.image.copy())
            batch = _obs_to_batch(obs)
            processed = pre(batch)
            with torch.inference_mode():
                action = policy.select_action(processed)
            action = post(action)
            np_action = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
            obs, reward, done, info = env.step(np_action)
            drawer_trace.append(float(info["drawer_fraction"]))
            action_trace.append(np_action)
            success = bool(info["is_success"])
            steps += 1
            if done:
                break
    finally:
        env.close()
    video_path = video_dir / f"mujoco_seed_{seed:03d}.mp4"
    _write_video(video_path, frames)
    return {
        "seed": seed,
        "success": success,
        "steps": steps,
        "max_drawer_fraction": float(max(drawer_trace)) if drawer_trace else 0.0,
        "final_drawer_fraction": float(drawer_trace[-1]) if drawer_trace else 0.0,
        "action_stats": _action_stats(np.asarray(action_trace, dtype=np.float32)),
        "video_path": str(video_path),
    }


def _summarize_cell(cell_id: str, records: list[dict[str, Any]], *, n_action_steps: int, path_kind: str) -> dict[str, Any]:
    return {
        "cell_id": cell_id,
        "path_kind": path_kind,
        "n_action_steps": int(n_action_steps),
        "records": records,
        "success_count": int(sum(1 for r in records if r["success"])),
        "mean_max_drawer_fraction": float(np.mean([r["max_drawer_fraction"] for r in records])) if records else 0.0,
        "videos": [r["video_path"] for r in records],
    }


def main() -> int:
    VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    cell_a_records = [_official_rollout(4, episode_seed=CONTROL_SEED + i, video_dir=VIDEO_ROOT / "cell_a_official_n4") for i in range(PARITY_EPISODES)]
    cell_b_records = [_official_rollout(1, episode_seed=CONTROL_SEED + i, video_dir=VIDEO_ROOT / "cell_b_official_n1") for i in range(PARITY_EPISODES)]

    pretrained_path = Path("/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero")
    mujoco_bundle_n4 = _load_policy_bundle(pretrained_path)
    mujoco_bundle_n1 = _load_policy_bundle(pretrained_path)
    cell_c_records = [_mujoco_rollout(mujoco_bundle_n4, seed=seed, n_action_steps=4, video_dir=VIDEO_ROOT / "cell_c_mujoco_n4") for seed in MUJOCO_SEEDS]
    cell_d_records = [_mujoco_rollout(mujoco_bundle_n1, seed=seed, n_action_steps=1, video_dir=VIDEO_ROOT / "cell_d_mujoco_n1") for seed in MUJOCO_SEEDS]

    cells = {
        "A": _summarize_cell("A", cell_a_records, n_action_steps=4, path_kind="official"),
        "B": _summarize_cell("B", cell_b_records, n_action_steps=1, path_kind="official"),
        "C": _summarize_cell("C", cell_c_records, n_action_steps=4, path_kind="mujoco"),
        "D": _summarize_cell("D", cell_d_records, n_action_steps=1, path_kind="mujoco"),
    }

    official_success_delta = abs(cells["A"]["success_count"] - cells["B"]["success_count"])
    official_drawer_delta = abs(cells["A"]["mean_max_drawer_fraction"] - cells["B"]["mean_max_drawer_fraction"])
    mujoco_success_delta = abs(cells["C"]["success_count"] - cells["D"]["success_count"])
    mujoco_drawer_delta = abs(cells["C"]["mean_max_drawer_fraction"] - cells["D"]["mean_max_drawer_fraction"])
    parity_blocker = bool(official_success_delta >= 1 or official_drawer_delta >= 0.15)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at": now_iso(),
        "passed": True,
        "cells": cells,
        "decision": {
            "official_success_delta": official_success_delta,
            "official_drawer_delta": official_drawer_delta,
            "mujoco_success_delta": mujoco_success_delta,
            "mujoco_drawer_delta": mujoco_drawer_delta,
            "parity_blocker": parity_blocker,
            "interpretation": (
                "n_action_steps parity is a blocker and MuJoCo eval must align to parity-cleared settings before later semantic interpretation."
                if parity_blocker
                else "n_action_steps parity is not the primary blocker under the current probe; later semantic gates may proceed."
            ),
        },
        "duration_s": time.time() - started,
    }
    write_json_atomic(ARTIFACT_PATH, payload)

    lines = [
        "# p1l Eval Parity Matrix",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Decision",
        f"- parity_blocker: {parity_blocker}",
        f"- official_success_delta: {official_success_delta}",
        f"- official_drawer_delta: {official_drawer_delta:.6f}",
        f"- mujoco_success_delta: {mujoco_success_delta}",
        f"- mujoco_drawer_delta: {mujoco_drawer_delta:.6f}",
        f"- interpretation: {payload['decision']['interpretation']}",
        "",
        "## Cells",
    ]
    for cell_id, cell in cells.items():
        lines.append(
            f"- Cell {cell_id}: path={cell['path_kind']} n_action_steps={cell['n_action_steps']} success_count={cell['success_count']} mean_max_drawer_fraction={cell['mean_max_drawer_fraction']:.6f}"
        )
    write_text_atomic(REPORT_PATH, "\n".join(lines).rstrip() + "\n")

    upsert_evidence(
        EVIDENCE_ID,
        EXPERIMENT_ID,
        "eval_parity_matrix",
        ARTIFACT_PATH,
        "4-cell official-vs-MuJoCo parity matrix isolating n_action_steps effects before later semantic interpretation.",
        verified=True,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

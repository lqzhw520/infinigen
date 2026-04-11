#!/usr/bin/env python3
"""P0B: capture exact successful official control traces."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.scripts.lerobot_eval import OBS_STR
from lerobot.utils.constants import ACTION

from mint_common import now_iso, write_json_atomic, write_text_atomic
from p1_execution_common import (
    ARTIFACT_DIR,
    CONTROL_EPISODES,
    CONTROL_N_ACTION_STEPS,
    CONTROL_SEED,
    OUTPUT_DIR,
    bundle_config_hash,
    close_bundle,
    get_single_vec_env,
    infer_libero_drawer_fraction,
    make_official_control_bundle,
    upsert_evidence,
)

TRACE_ROOT = ARTIFACT_DIR / "p1k_control_traces"
SUMMARY_ARTIFACT = ARTIFACT_DIR / "p1k_capture_control_traces.json"
SUMMARY_REPORT = OUTPUT_DIR / "p1k_capture_control_traces.md"
EVIDENCE_ID = "E053"
EXPERIMENT_ID = "p1k_capture_control_traces"


def _tensor_image_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().numpy()
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.max() <= 1.0:
        image = image * 255.0
    image = np.clip(image, 0.0, 255.0)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


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


def capture_episode(bundle: dict[str, Any], *, episode_seed: int, episode_index: int) -> dict[str, Any]:
    vec_env = get_single_vec_env(bundle)
    policy = bundle["policy"]
    env_pre = bundle["env_preprocessor"]
    env_post = bundle["env_postprocessor"]
    pre = bundle["preprocessor"]
    post = bundle["postprocessor"]

    policy.reset()
    observation, info = vec_env.reset(seed=[episode_seed])

    images = []
    images2 = []
    states_pre = []
    states_next = []
    raw_actions = []
    post_actions = []
    sent_actions = []
    rewards = []
    dones = []
    successes = []
    drawer_fractions = []
    queue_lens = []

    max_steps = int(vec_env.call("_max_episode_steps")[0])
    for step_idx in range(max_steps):
        observation_t = preprocess_observation(observation)
        observation_t = add_envs_task(vec_env, observation_t)
        observation_env = env_pre(observation_t)

        # Capture RAW image from observation dict before preprocessing
        raw_image = observation.get("images", {}).get("image")
        if raw_image is not None and isinstance(raw_image, torch.Tensor):
            image = _tensor_image_to_uint8(raw_image[0])
        else:
            image = _tensor_image_to_uint8(observation_env[f"{OBS_STR}.images.image"][0])
        raw_image2 = observation.get("images", {}).get("image2")
        if raw_image2 is not None and isinstance(raw_image2, torch.Tensor):
            image2 = _tensor_image_to_uint8(raw_image2[0])
        else:
            image2 = _tensor_image_to_uint8(observation_env[f"{OBS_STR}.images.image2"][0])
        state_pre = observation_env[f"{OBS_STR}.state"][0].detach().cpu().numpy().astype(np.float32)
        drawer_pre = infer_libero_drawer_fraction(vec_env)

        processed = pre(observation_env)
        with torch.inference_mode():
            action_raw = policy.select_action(processed)
        action_post = post(action_raw)
        action_transition = env_post({ACTION: action_post})
        action_sent = action_transition[ACTION]

        observation, reward, terminated, truncated, info = vec_env.step(action_sent.detach().cpu().numpy())
        next_t = preprocess_observation(observation)
        next_t = add_envs_task(vec_env, next_t)
        next_env = env_pre(next_t)
        state_next = next_env[f"{OBS_STR}.state"][0].detach().cpu().numpy().astype(np.float32)
        drawer_next = infer_libero_drawer_fraction(vec_env)
        done = bool(terminated[0] or truncated[0])
        success = _extract_bool(info, "is_success")

        images.append(image)
        images2.append(image2)
        states_pre.append(state_pre)
        states_next.append(state_next)
        raw_actions.append(action_raw[0].detach().cpu().numpy().astype(np.float32))
        post_actions.append(action_post[0].detach().cpu().numpy().astype(np.float32))
        sent_actions.append(action_sent[0].detach().cpu().numpy().astype(np.float32))
        rewards.append(float(reward[0]))
        dones.append(done)
        successes.append(success)
        drawer_fractions.append(float(drawer_next.get("fraction") if drawer_next.get("fraction") is not None else drawer_pre.get("fraction") or 0.0))
        queue_lens.append(len(getattr(policy, "_action_queue", [])))

        if done:
            break

    return {
        "episode_index": episode_index,
        "seed": episode_seed,
        "images": np.stack(images, axis=0).astype(np.uint8),
        "images2": np.stack(images2, axis=0).astype(np.uint8),
        "state_pre": np.stack(states_pre, axis=0).astype(np.float32),
        "state_next": np.stack(states_next, axis=0).astype(np.float32),
        "policy_action_raw": np.stack(raw_actions, axis=0).astype(np.float32),
        "policy_action_post": np.stack(post_actions, axis=0).astype(np.float32),
        "action_sent": np.stack(sent_actions, axis=0).astype(np.float32),
        "reward": np.asarray(rewards, dtype=np.float32),
        "done": np.asarray(dones, dtype=np.bool_),
        "success": np.asarray(successes, dtype=np.bool_),
        "drawer_fraction": np.asarray(drawer_fractions, dtype=np.float32),
        "queue_len_after_select": np.asarray(queue_lens, dtype=np.int32),
        "steps": len(rewards),
        "success_any": bool(any(successes)),
        "n_action_steps": int(policy.config.n_action_steps),
    }


def main() -> int:
    if TRACE_ROOT.exists():
        for child in TRACE_ROOT.iterdir():
            if child.is_dir():
                for nested in child.glob('*'):
                    if nested.is_file():
                        nested.unlink()
                child.rmdir()
            else:
                child.unlink()
    TRACE_ROOT.mkdir(parents=True, exist_ok=True)

    started = time.time()
    bundle = make_official_control_bundle(
        n_action_steps=CONTROL_N_ACTION_STEPS,
        n_episodes=1,
        device="cuda",
        seed=CONTROL_SEED,
        output_dir=OUTPUT_DIR / "p1k_capture_control_traces_runtime",
    )
    config_hash = bundle_config_hash(bundle["cfg"])

    episode_records = []
    lengths_ok = True
    try:
        for episode_index in range(CONTROL_EPISODES):
            episode_seed = CONTROL_SEED + episode_index
            episode = capture_episode(bundle, episode_seed=episode_seed, episode_index=episode_index)
            episode_dir = TRACE_ROOT / f"episode_{episode_index:02d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                episode_dir / "trace.npz",
                image=episode["images"],
                image2=episode["images2"],
                state_pre=episode["state_pre"],
                state_next=episode["state_next"],
                policy_action_raw=episode["policy_action_raw"],
                policy_action_post=episode["policy_action_post"],
                action_sent=episode["action_sent"],
                reward=episode["reward"],
                done=episode["done"],
                success=episode["success"],
                drawer_fraction=episode["drawer_fraction"],
                queue_len_after_select=episode["queue_len_after_select"],
            )
            meta = {
                "episode_index": episode_index,
                "seed": episode_seed,
                "steps": episode["steps"],
                "success_any": episode["success_any"],
                "n_action_steps": episode["n_action_steps"],
                "config_hash": config_hash,
            }
            write_json_atomic(episode_dir / "episode_meta.json", meta)
            lengths = {
                "image": int(len(episode["images"])),
                "image2": int(len(episode["images2"])),
                "state_pre": int(len(episode["state_pre"])),
                "state_next": int(len(episode["state_next"])),
                "policy_action_raw": int(len(episode["policy_action_raw"])),
                "policy_action_post": int(len(episode["policy_action_post"])),
                "action_sent": int(len(episode["action_sent"])),
                "drawer_fraction": int(len(episode["drawer_fraction"])),
            }
            lengths_ok = lengths_ok and len(set(lengths.values())) == 1
            episode_records.append({
                "episode_index": episode_index,
                "seed": episode_seed,
                "steps": episode["steps"],
                "success_any": episode["success_any"],
                "lengths": lengths,
                "trace_path": str((episode_dir / 'trace.npz').relative_to(ARTIFACT_DIR.parent.parent.parent)),
            })
    finally:
        close_bundle(bundle)

    successful_episodes = sum(1 for record in episode_records if record["success_any"])
    passed = successful_episodes >= 3 and lengths_ok
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at": now_iso(),
        "passed": passed,
        "control_seed_start": CONTROL_SEED,
        "requested_episodes": CONTROL_EPISODES,
        "successful_episodes": successful_episodes,
        "lengths_consistent": lengths_ok,
        "config_hash": config_hash,
        "trace_root": str(TRACE_ROOT),
        "episode_records": episode_records,
        "duration_s": time.time() - started,
        "control_state_semantics_note": "Captured live policy-facing observation.state from LiberoProcessorStep; this telemetry overrides stale comments and STATE_NAMES assumptions.",
    }
    write_json_atomic(SUMMARY_ARTIFACT, summary)

    report_lines = [
        "# p1k Capture Control Traces",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Summary",
        f"- passed: {passed}",
        f"- successful_episodes: {successful_episodes}/{CONTROL_EPISODES}",
        f"- lengths_consistent: {lengths_ok}",
        f"- config_hash: `{config_hash}`",
        "",
        "## Episode Records",
    ]
    for record in episode_records:
        report_lines.append(
            f"- episode_{record['episode_index']:02d}: seed={record['seed']} steps={record['steps']} success_any={record['success_any']} lengths={record['lengths']}"
        )
    write_text_atomic(SUMMARY_REPORT, "\n".join(report_lines).rstrip() + "\n")

    upsert_evidence(
        EVIDENCE_ID,
        EXPERIMENT_ID,
        "control_trace_capture",
        SUMMARY_ARTIFACT,
        "Exact successful control trace capture for official LIBERO drawer control, including images, policy-facing state, raw/post actions, sent actions, and drawer progress.",
        verified=passed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

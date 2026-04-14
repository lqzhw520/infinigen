#!/usr/bin/env python3
"""Evaluate random, pretrained, and fine-tuned MINT on robot drawer scenes."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import numpy as np
import torch
# Apply compatibility patches BEFORE importing lerobot/MINT.
# Fixes lerobot 0.4.3 + draccus 0.8.0 issues with local MINT checkpoints.
from mint_eval_patches import apply as _apply_patches

_apply_patches()

from drawer_robot_env import DrawerRobotEnv
from mint_common import (
    ACTIVE_ACTION_CONTRACT_PATH,
    DATASET_DIR,
    DATASET_REPO_ID,
    DEFAULT_HELD_OUT_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    load_json,
)
from render_rollout_video import render_trace_video
from strict_success import STRICT_SUCCESS_VERSION, evaluate_strict_success

MINT_CKPT = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero"
DEFAULT_EVAL_IMAGE_SIZE = 256


def random_policy(rng: np.random.Generator) -> np.ndarray:
    action = rng.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)
    action[6] = float(rng.choice([-1.0, 1.0]))
    return action


def _load_policy(path: str, dataset_root: Path, repo_id: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot_policy_mint.modeling_mint import MINTPolicy

    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, revision="main")
    policy = MINTPolicy.from_pretrained(
        path, local_files_only=True, dataset_stats=dataset.meta.stats
    )
    policy.eval()
    # Cast direct_grip_head to float32 to match paligemma_with_expert (now float32)
    if hasattr(policy.model, "direct_grip_head"):
        policy.model.direct_grip_head = policy.model.direct_grip_head.to(dtype=torch.float32)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=path, dataset_stats=dataset.meta.stats
    )
    return policy, preprocessor, postprocessor


def _obs_to_batch(obs) -> dict:
    return {
        "observation.images.image": torch.from_numpy(obs.image)
        .permute(2, 0, 1)
        .to(torch.float32)
        / 255.0,
        "observation.images.image2": torch.from_numpy(obs.image2)
        .permute(2, 0, 1)
        .to(torch.float32)
        / 255.0,
        "observation.state": torch.from_numpy(obs.state.astype(np.float32)),
        "task": obs.task,
    }


def _dataset_image_size() -> int:
    override = os.environ.get("MINT_DRAWER_EVAL_IMAGE_SIZE")
    if override:
        return int(override)
    info_path = DATASET_DIR / "meta" / "info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            shape = info["features"]["observation.images.image"]["shape"]
            if isinstance(shape, list) and len(shape) >= 2:
                return int(shape[0])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return DEFAULT_EVAL_IMAGE_SIZE


def rollout_policy(
    seed: int,
    kind: str,
    policy_bundle=None,
    *,
    rng_seed: int = 0,
    max_steps: int = 96,
    action_contract: dict | None = None,
    video_path: Path | None = None,
    video_stage: str | None = None,
    video_variant: str | None = None,
    video_attempt: str | None = None,
) -> dict:
    image_size = _dataset_image_size()
    env = DrawerRobotEnv(
        seed=seed,
        image_size=image_size,
        max_steps=max_steps,
        action_contract=action_contract,
    )
    obs = env.reset()
    rng = np.random.default_rng(rng_seed + seed)
    policy = pre = post = None
    if policy_bundle:
        policy, pre, post = policy_bundle

    success = False
    grasp_success = False
    success_step = None
    steps = 0
    pull_distance = 0.0
    drawer_trace = []
    attached_trace = []
    frames = []
    frames2 = []
    eef_pos_trace = []
    raw_action_trace = []
    try:
        for steps in range(1, max_steps + 1):
            if video_path is not None:
                frames.append(obs.image.copy())
                frames2.append(obs.image2.copy())
            # Track EEF position before action (end-effector pose in world frame)
            eef_pos_trace.append(
                obs.eef_pos.copy() if hasattr(obs, "eef_pos") else obs.state[:3].copy()
            )
            if kind == "random":
                action = random_policy(rng)
            else:
                batch = _obs_to_batch(obs)
                processed = pre(batch)
                with torch.inference_mode():
                    action = policy.select_action(processed)
                action = post(action)
                action = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
            # Track raw actions before env.step (diagnostic: is policy outputting non-zero?)
            raw_action_trace.append(action.copy() if action is not None else None)
            obs, _, done, info = env.step(action)
            pull_distance = max(pull_distance, float(info["drawer_fraction"]))
            grasp_success = grasp_success or bool(
                info.get("ever_attached", info.get("attached"))
            )
            drawer_trace.append(float(info["drawer_fraction"]))
            attached_trace.append(bool(info.get("attached")))
            strict = info.get("strict_metrics") or evaluate_strict_success(
                np.asarray(drawer_trace, dtype=np.float32),
                np.asarray(attached_trace, dtype=bool),
            )
            success = bool(strict["strict_success"])
            if success and success_step is None:
                success_step = steps
            if done:
                break
        strict = info.get("strict_metrics") if drawer_trace else None
        if not strict:
            strict = evaluate_strict_success(
                np.asarray(drawer_trace, dtype=np.float32),
                np.asarray(attached_trace, dtype=bool),
            )
        rendered_video = None
        if video_path is not None and frames:
            rendered_video = render_trace_video(
                images=frames,
                images2=frames2,
                output_path=video_path,
                stage=video_stage or "eval",
                policy=kind,
                seed=seed,
                attempt=video_attempt,
                variant=video_variant,
                drawer_trace=drawer_trace,
                attached_trace=attached_trace,
                strict_success=bool(strict["strict_success"]),
                phase_labels=["policy"] * len(frames),
                extra_line=f"max_steps={max_steps}",
                fps=10,
            )
        # --- Diagnostic: action + EEF motion metrics ---
        valid_actions = [a for a in raw_action_trace if a is not None]
        non_zero_action_ratio = 0.0
        total_eef_motion = 0.0
        if valid_actions:
            non_zero_action_ratio = float(
                np.mean([float(np.abs(a).max() > 0.01) for a in valid_actions])
            )
        if len(eef_pos_trace) > 1:
            eef_arr = np.array(eef_pos_trace)
            step_deltas = np.linalg.norm(np.diff(eef_arr, axis=0), axis=1)
            total_eef_motion = float(np.sum(step_deltas))

        return {
            "seed": seed,
            "policy": kind,
            "success": success,
            "steps": steps,
            "pull_distance": float(pull_distance),
            "grasp_success": grasp_success,
            "time_to_completion": int(
                success_step if success_step is not None else max_steps
            ),
            "strict_success_version": STRICT_SUCCESS_VERSION,
            "strict_metrics": strict,
            "video_path": None if rendered_video is None else str(rendered_video),
            # --- Diagnostic instrumentation ---
            "non_zero_action_ratio": non_zero_action_ratio,
            "total_eef_motion": total_eef_motion,
            "drawer_trace": drawer_trace,
            "attached_trace": attached_trace,
            "env_image_size": image_size,
        }
    finally:
        env.close()


def aggregate(records: list[dict]) -> dict:
    successes = [1.0 if item["success"] else 0.0 for item in records]
    pulls = [item["pull_distance"] for item in records]
    times = [item["time_to_completion"] for item in records]
    grasps = [1.0 if item["grasp_success"] else 0.0 for item in records]
    # --- Diagnostic fields (may be absent on legacy records) ---
    non_zero_ratios = [
        item["non_zero_action_ratio"]
        for item in records
        if "non_zero_action_ratio" in item
    ]
    eef_motions = [
        item["total_eef_motion"] for item in records if "total_eef_motion" in item
    ]
    per_seed_records = {}
    for item in records:
        per_seed_records.setdefault(int(item["seed"]), []).append(item)
    per_seed = {}
    for seed, seed_records in sorted(per_seed_records.items()):
        per_seed[str(seed)] = {
            "n_episodes": len(seed_records),
            "success_rate": float(
                sum(1.0 if row["success"] else 0.0 for row in seed_records)
                / max(len(seed_records), 1)
            ),
            "successes": int(sum(1 for row in seed_records if row["success"])),
            "pull_distance_mean": float(
                statistics.mean(row["pull_distance"] for row in seed_records)
            ),
            "grasp_success_rate": float(
                sum(1.0 if row["grasp_success"] else 0.0 for row in seed_records)
                / max(len(seed_records), 1)
            ),
            "time_to_completion_mean": float(
                statistics.mean(row["time_to_completion"] for row in seed_records)
            ),
            # --- Diagnostic: per-seed action/EFF motion ---
            "non_zero_action_ratio_mean": float(
                statistics.mean(
                    row["non_zero_action_ratio"]
                    for row in seed_records
                    if "non_zero_action_ratio" in row
                )
            )
            if any("non_zero_action_ratio" in row for row in seed_records)
            else None,
            "total_eef_motion_mean": float(
                statistics.mean(
                    row["total_eef_motion"]
                    for row in seed_records
                    if "total_eef_motion" in row
                )
            )
            if any("total_eef_motion" in row for row in seed_records)
            else None,
        }
    result = {
        "n_episodes": len(records),
        "n_unique_seeds": len(per_seed_records),
        "success_rate": float(sum(successes) / max(len(successes), 1)),
        "pull_distance_mean": float(statistics.mean(pulls)) if pulls else 0.0,
        "grasp_success_rate": float(sum(grasps) / max(len(grasps), 1)),
        "time_to_completion_mean": float(statistics.mean(times)) if times else 0.0,
        "successes": int(sum(successes)),
        "successful_seed_count": int(
            sum(1 for metrics in per_seed.values() if metrics["successes"] > 0)
        ),
        "per_seed": per_seed,
    }
    # --- Aggregate diagnostic fields across all records ---
    if non_zero_ratios:
        result["non_zero_action_ratio_mean"] = float(statistics.mean(non_zero_ratios))
    if eef_motions:
        result["total_eef_motion_mean"] = float(statistics.mean(eef_motions))
    return result


def evaluate_policy_set(
    *,
    finetuned_path: str,
    seeds: list[int],
    dataset_root: Path = DATASET_DIR,
    repo_id: str = DATASET_REPO_ID,
    max_steps: int = 96,
    include_random: bool = True,
    action_contract: dict | None = None,
    episodes_per_seed: int = 1,
) -> tuple[dict, dict]:
    pretrained_bundle = _load_policy(MINT_CKPT, dataset_root, repo_id)
    finetuned_bundle = _load_policy(finetuned_path, dataset_root, repo_id)

    records = {"pretrained_mint": [], "finetuned_mint": []}
    if include_random:
        records["random"] = []
    for seed in seeds:
        for attempt_idx in range(max(1, int(episodes_per_seed))):
            if include_random:
                item = rollout_policy(
                    seed,
                    "random",
                    None,
                    rng_seed=13 + attempt_idx * 101,
                    max_steps=max_steps,
                    action_contract=action_contract,
                )
                item["eval_attempt"] = attempt_idx
                records["random"].append(item)
            item = rollout_policy(
                seed,
                "pretrained_mint",
                pretrained_bundle,
                rng_seed=17 + attempt_idx * 101,
                max_steps=max_steps,
                action_contract=action_contract,
            )
            item["eval_attempt"] = attempt_idx
            records["pretrained_mint"].append(item)
            item = rollout_policy(
                seed,
                "finetuned_mint",
                finetuned_bundle,
                rng_seed=19 + attempt_idx * 101,
                max_steps=max_steps,
                action_contract=action_contract,
            )
            item["eval_attempt"] = attempt_idx
            records["finetuned_mint"].append(item)
    comparison = {name: aggregate(items) for name, items in records.items()}
    return comparison, records


def evaluate_campaign(
    finetuned_path: str | Path,
    *,
    dataset_root: str | Path = DATASET_DIR,
    repo_id: str = DATASET_REPO_ID,
    held_out_seeds: list[int] | None = None,
    episodes_per_seed: int = 3,
) -> tuple[dict, dict]:
    action_contract = load_json(ACTIVE_ACTION_CONTRACT_PATH, {})
    seeds = [int(seed) for seed in (held_out_seeds if held_out_seeds is not None else DEFAULT_HELD_OUT_SEEDS)]
    comparison, records = evaluate_policy_set(
        finetuned_path=str(finetuned_path),
        seeds=seeds,
        dataset_root=Path(dataset_root),
        repo_id=repo_id,
        action_contract=action_contract,
        episodes_per_seed=int(episodes_per_seed),
    )
    verdict = (
        "claim_supported"
        if comparison["finetuned_mint"]["success_rate"]
        > comparison["pretrained_mint"]["success_rate"]
        else "scientific_not_supported"
    )
    strongest_true_claim = (
        "Fine-tuned MINT improves held-out drawer success in the AnyGrasp-conditioned robot-trajectory simulation."
        if verdict == "claim_supported"
        else "The robot-trajectory pipeline is executable, but fine-tuned MINT does not beat pretrained MINT on held-out drawer variants."
    )
    return {
        "verdict": verdict,
        "held_out_seeds": seeds,
        "eval_max_steps": 96,
        "episodes_per_seed": int(episodes_per_seed),
        "comparison": comparison,
        "strongest_true_claim": strongest_true_claim,
    }, records


def evaluate_train_probe(
    finetuned_path: str,
    *,
    seeds: list[int] | None = None,
    dataset_root: Path = DATASET_DIR,
    repo_id: str = DATASET_REPO_ID,
    max_steps: int = 96,
    episodes_per_seed: int = 1,
) -> tuple[dict, dict]:
    probe_seeds = seeds or DEFAULT_TRAIN_SEEDS
    action_contract = load_json(ACTIVE_ACTION_CONTRACT_PATH, {})
    comparison, records = evaluate_policy_set(
        finetuned_path=finetuned_path,
        seeds=probe_seeds,
        dataset_root=dataset_root,
        repo_id=repo_id,
        max_steps=max_steps,
        include_random=False,
        action_contract=action_contract,
        episodes_per_seed=episodes_per_seed,
    )
    return {
        "eval_max_steps": max_steps,
        "train_seeds": probe_seeds,
        "episodes_per_seed": int(episodes_per_seed),
        "summary": comparison,
        "records": records,
    }, records


def record_policy_video(
    *,
    seed: int,
    kind: str,
    output_path: Path,
    finetuned_path: str | None = None,
    dataset_root: Path = DATASET_DIR,
    repo_id: str = DATASET_REPO_ID,
    max_steps: int = 96,
    attempt_idx: int = 0,
    stage: str = "eval",
    variant: str | None = None,
    action_contract: dict | None = None,
) -> dict:
    if action_contract is None:
        action_contract = load_json(ACTIVE_ACTION_CONTRACT_PATH, {})
    bundle = None
    if kind == "pretrained_mint":
        bundle = _load_policy(MINT_CKPT, dataset_root, repo_id)
    elif kind == "finetuned_mint":
        if not finetuned_path:
            raise ValueError(
                "finetuned_path is required for finetuned_mint video rendering"
            )
        bundle = _load_policy(finetuned_path, dataset_root, repo_id)
    elif kind != "random":
        raise ValueError(f"unsupported policy kind: {kind}")
    return rollout_policy(
        seed,
        kind,
        bundle,
        rng_seed=17 + attempt_idx * 101,
        max_steps=max_steps,
        action_contract=action_contract,
        video_path=output_path,
        video_stage=stage,
        video_variant=variant,
        video_attempt=str(attempt_idx),
    )


def render_report(
    summary: dict, *, title: str = "# MINT Drawer Robot-Trajectory Sim Evaluation"
) -> str:
    comparison = summary["comparison"]
    lines = [
        title,
        "",
        f"**Verdict**: `{summary['verdict']}`",
        f"**Held-out Seeds**: `{summary['held_out_seeds']}`",
        f"**Eval Max Steps**: `{summary.get('eval_max_steps', 96)}`",
        "",
        "| Policy | Success Rate | Grasp Success | Pull Distance | Time to Completion | Episodes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["random", "pretrained_mint", "finetuned_mint"]:
        payload = comparison[name]
        lines.append(
            f"| {name} | {payload['success_rate']:.3f} | {payload['grasp_success_rate']:.3f} | {payload['pull_distance_mean']:.3f} | {payload['time_to_completion_mean']:.2f} | {payload['n_episodes']} |"
        )
    lines.extend(
        ["", "## Strongest True Claim", "", summary["strongest_true_claim"], ""]
    )
    return "\n".join(lines)

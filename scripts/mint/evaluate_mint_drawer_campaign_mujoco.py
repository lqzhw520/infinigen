#!/usr/bin/env python3
"""MuJoCo-native tiny retrain probe/eval helpers for canonical V1cT2S0."""

from __future__ import annotations

import statistics
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from mint_eval_patches import apply as _apply_patches

_apply_patches()

from drawer_robot_env_mujoco import DrawerRobotEnvMuJoCo
from mint_common import DATASET_DIR, DATASET_REPO_ID, DEFAULT_HELD_OUT_SEEDS, DEFAULT_TRAIN_SEEDS
from root_cause_controller import RootCauseController
from strict_success import STRICT_SUCCESS_VERSION, evaluate_strict_success

MINT_CKPT = "/mnt/afs2/zhuhaowu/infinigen/external/MINT/checkpoints/MINT-libero"
DEFAULT_EVAL_IMAGE_SIZE = 256
EXPECTED_CANONICAL_TRAIN_CELL = "V1cT2S0"
EXPECTED_BEST_TRAIN_STATE_MODE = "S0"
EXPECTED_STATE_MODE_NAME = "m0_proxy"
EXPECTED_INTERACTION_MODE = "orientation_sensitive_v3_task_identity_locked"
EXPECTED_RUNTIME_MAPPING_SOURCE = "collision_geom"
EXPECTED_EVALUATION_BACKEND = "mujoco"
EXPECTED_ENV_FAMILY = "canonical_cell_runtime"


def _load_policy(path: str, dataset_root: Path, repo_id: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot_policy_mint.modeling_mint import MINTPolicy

    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, revision="main")
    policy = MINTPolicy.from_pretrained(
        path, local_files_only=True, dataset_stats=dataset.meta.stats
    )
    policy.eval()
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


@lru_cache(maxsize=4)
def _canonical_contract(evaluation_cell_id: str):
    controller = RootCauseController(
        max_experiments_per_cycle=1,
        dry_run=False,
        experiment_family="RCA",
        selector_mode="frozen_v5_pro",
        seed_split_a=list(DEFAULT_TRAIN_SEEDS),
        truthful_measurement_required=True,
        max_rollouts_per_experiment=1,
    )
    spec = controller._matrix_cell_lane_spec(
        evaluation_cell_id,
        "G8d",
        note="Tiny retrain parity probe/eval on canonical MuJoCo cell.",
    )
    contract = controller._materialize_contract(spec.env_contract_config)
    return contract


def _parity_expectation(canonical_train_cell: str, best_train_state_mode: str, evaluation_backend: str) -> dict[str, str]:
    expected = {
        "evaluation_backend": str(evaluation_backend),
        "canonical_train_cell": str(canonical_train_cell),
        "best_train_state_mode": str(best_train_state_mode),
        "state_mode_name": EXPECTED_STATE_MODE_NAME,
        "interaction_mode": EXPECTED_INTERACTION_MODE,
        "runtime_visible_handle_mapping_source": EXPECTED_RUNTIME_MAPPING_SOURCE,
    }
    if expected["evaluation_backend"] != EXPECTED_EVALUATION_BACKEND:
        raise ValueError(f"Unsupported evaluation backend: {expected['evaluation_backend']}")
    if expected["canonical_train_cell"] != EXPECTED_CANONICAL_TRAIN_CELL:
        raise ValueError(f"MuJoCo parity evaluator is frozen to {EXPECTED_CANONICAL_TRAIN_CELL}, got {expected['canonical_train_cell']}")
    if expected["best_train_state_mode"] != EXPECTED_BEST_TRAIN_STATE_MODE:
        raise ValueError(f"MuJoCo parity evaluator is frozen to {EXPECTED_BEST_TRAIN_STATE_MODE}, got {expected['best_train_state_mode']}")
    return expected


def _validate_record_parity(record: dict, expected: dict[str, str]) -> None:
    for key, value in expected.items():
        if str(record.get(key)) != value:
            raise RuntimeError(f"Parity violation for {key}: expected {value!r}, got {record.get(key)!r}")
    if not bool(record.get("measurement_truthful", False)):
        raise RuntimeError("Parity violation: measurement_truthful is false")



def rollout_policy(
    seed: int,
    kind: str,
    policy_bundle=None,
    *,
    rng_seed: int = 0,
    max_steps: int = 96,
    image_size: int = DEFAULT_EVAL_IMAGE_SIZE,
    canonical_train_cell: str = EXPECTED_CANONICAL_TRAIN_CELL,
    best_train_state_mode: str = EXPECTED_BEST_TRAIN_STATE_MODE,
    evaluation_backend: str = EXPECTED_EVALUATION_BACKEND,
) -> dict:
    expected = _parity_expectation(canonical_train_cell, best_train_state_mode, evaluation_backend)
    contract = _canonical_contract(canonical_train_cell)
    env = DrawerRobotEnvMuJoCo(
        seed=seed,
        image_size=int(image_size),
        max_steps=int(max_steps),
        contract=contract,
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
    drawer_trace: list[float] = []
    attached_trace: list[bool] = []
    eef_pos_trace: list[np.ndarray] = []
    raw_action_trace: list[np.ndarray] = []
    last_probe = dict(obs.handle_probe_metadata or {})
    try:
        for steps in range(1, int(max_steps) + 1):
            eef_pos_trace.append(obs.eef_pos.copy())
            if kind == "random":
                action = rng.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)
                action[6] = float(rng.choice([-1.0, 1.0]))
            else:
                batch = _obs_to_batch(obs)
                processed = pre(batch)
                with torch.inference_mode():
                    action = policy.select_action(processed)
                action = post(action)
                action = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
            raw_action_trace.append(action.copy())
            obs, _, done, info = env.step(action)
            last_probe = dict(obs.handle_probe_metadata or {})
            pull_distance = max(pull_distance, float(info.get("drawer_fraction", 0.0)))
            grasp_success = grasp_success or bool(info.get("attached", False))
            drawer_trace.append(float(info.get("drawer_fraction", 0.0)))
            attached_trace.append(bool(info.get("attached", False)))
            strict = evaluate_strict_success(
                np.asarray(drawer_trace, dtype=np.float32),
                np.asarray(attached_trace, dtype=bool),
            )
            success = bool(strict["strict_success"])
            if success and success_step is None:
                success_step = steps
            if done:
                break
        strict = evaluate_strict_success(
            np.asarray(drawer_trace, dtype=np.float32),
            np.asarray(attached_trace, dtype=bool),
        )
        non_zero_action_ratio = 0.0
        total_eef_motion = 0.0
        if raw_action_trace:
            non_zero_action_ratio = float(
                np.mean([float(np.abs(action).max() > 0.01) for action in raw_action_trace])
            )
        if len(eef_pos_trace) > 1:
            eef_arr = np.asarray(eef_pos_trace, dtype=np.float32)
            total_eef_motion = float(np.sum(np.linalg.norm(np.diff(eef_arr, axis=0), axis=1)))
        record = {
            "seed": int(seed),
            "policy": kind,
            "success": bool(success),
            "steps": int(steps),
            "pull_distance": float(pull_distance),
            "grasp_success": bool(grasp_success),
            "time_to_completion": int(success_step if success_step is not None else int(max_steps)),
            "strict_success_version": STRICT_SUCCESS_VERSION,
            "strict_metrics": strict,
            "non_zero_action_ratio": float(non_zero_action_ratio),
            "total_eef_motion": float(total_eef_motion),
            "drawer_trace": drawer_trace,
            "attached_trace": attached_trace,
            "env_image_size": int(image_size),
            "evaluation_backend": EXPECTED_EVALUATION_BACKEND,
            "evaluation_env_family": EXPECTED_ENV_FAMILY,
            "evaluation_cell_id": canonical_train_cell,
            "canonical_train_cell": canonical_train_cell,
            "best_train_state_mode": best_train_state_mode,
            "state_mode_name": str(contract.state_mode),
            "interaction_mode": str(contract.interaction_mode),
            "measurement_truthful": bool(last_probe.get("measurement_truthful", False)),
            "measurement_truth_tier": last_probe.get("measurement_truth_tier"),
            "runtime_visible_handle_mapping_source": last_probe.get("runtime_visible_handle_mapping_source"),
        }
        _validate_record_parity(record, expected)
        return record
    finally:
        env.close()



def aggregate(records: list[dict]) -> dict:
    successes = [1.0 if item["success"] else 0.0 for item in records]
    pulls = [item["pull_distance"] for item in records]
    times = [item["time_to_completion"] for item in records]
    grasps = [1.0 if item["grasp_success"] else 0.0 for item in records]
    non_zero_ratios = [item["non_zero_action_ratio"] for item in records if "non_zero_action_ratio" in item]
    eef_motions = [item["total_eef_motion"] for item in records if "total_eef_motion" in item]
    per_seed_records: dict[int, list[dict]] = {}
    for item in records:
        per_seed_records.setdefault(int(item["seed"]), []).append(item)
    per_seed: dict[str, dict] = {}
    for seed, seed_records in sorted(per_seed_records.items()):
        per_seed[str(seed)] = {
            "n_episodes": len(seed_records),
            "success_rate": float(sum(1.0 if row["success"] else 0.0 for row in seed_records) / max(len(seed_records), 1)),
            "successes": int(sum(1 for row in seed_records if row["success"])),
            "pull_distance_mean": float(statistics.mean(row["pull_distance"] for row in seed_records)),
            "grasp_success_rate": float(sum(1.0 if row["grasp_success"] else 0.0 for row in seed_records) / max(len(seed_records), 1)),
            "time_to_completion_mean": float(statistics.mean(row["time_to_completion"] for row in seed_records)),
            "non_zero_action_ratio_mean": float(statistics.mean(row["non_zero_action_ratio"] for row in seed_records)),
            "total_eef_motion_mean": float(statistics.mean(row["total_eef_motion"] for row in seed_records)),
        }
    result = {
        "n_episodes": len(records),
        "n_unique_seeds": len(per_seed_records),
        "success_rate": float(sum(successes) / max(len(successes), 1)),
        "pull_distance_mean": float(statistics.mean(pulls)) if pulls else 0.0,
        "grasp_success_rate": float(sum(grasps) / max(len(grasps), 1)),
        "time_to_completion_mean": float(statistics.mean(times)) if times else 0.0,
        "successes": int(sum(successes)),
        "successful_seed_count": int(sum(1 for metrics in per_seed.values() if metrics["successes"] > 0)),
        "per_seed": per_seed,
    }
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
    episodes_per_seed: int = 1,
    image_size: int = DEFAULT_EVAL_IMAGE_SIZE,
    canonical_train_cell: str = EXPECTED_CANONICAL_TRAIN_CELL,
    best_train_state_mode: str = EXPECTED_BEST_TRAIN_STATE_MODE,
    evaluation_backend: str = EXPECTED_EVALUATION_BACKEND,
) -> tuple[dict, dict]:
    pretrained_bundle = _load_policy(MINT_CKPT, dataset_root, repo_id)
    finetuned_bundle = _load_policy(finetuned_path, dataset_root, repo_id)

    records: dict[str, list[dict]] = {"pretrained_mint": [], "finetuned_mint": []}
    if include_random:
        records["random"] = []
    for seed in seeds:
        for attempt_idx in range(max(1, int(episodes_per_seed))):
            if include_random:
                item = rollout_policy(
                    int(seed),
                    "random",
                    None,
                    rng_seed=13 + attempt_idx * 101,
                    max_steps=max_steps,
                    image_size=image_size,
                    canonical_train_cell=canonical_train_cell,
                    best_train_state_mode=best_train_state_mode,
                    evaluation_backend=evaluation_backend,
                )
                item["eval_attempt"] = attempt_idx
                records["random"].append(item)
            item = rollout_policy(
                int(seed),
                "pretrained_mint",
                pretrained_bundle,
                rng_seed=17 + attempt_idx * 101,
                max_steps=max_steps,
                image_size=image_size,
                canonical_train_cell=canonical_train_cell,
                best_train_state_mode=best_train_state_mode,
                evaluation_backend=evaluation_backend,
            )
            item["eval_attempt"] = attempt_idx
            records["pretrained_mint"].append(item)
            item = rollout_policy(
                int(seed),
                "finetuned_mint",
                finetuned_bundle,
                rng_seed=19 + attempt_idx * 101,
                max_steps=max_steps,
                image_size=image_size,
                canonical_train_cell=canonical_train_cell,
                best_train_state_mode=best_train_state_mode,
                evaluation_backend=evaluation_backend,
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
    heldout_seeds: list[int] | None = None,
    episodes_per_seed: int = 3,
    max_steps: int = 96,
    image_size: int = DEFAULT_EVAL_IMAGE_SIZE,
    canonical_train_cell: str = EXPECTED_CANONICAL_TRAIN_CELL,
    best_train_state_mode: str = EXPECTED_BEST_TRAIN_STATE_MODE,
    evaluation_backend: str = EXPECTED_EVALUATION_BACKEND,
) -> tuple[dict, dict]:
    seeds = [int(seed) for seed in (heldout_seeds if heldout_seeds is not None else held_out_seeds if held_out_seeds is not None else DEFAULT_HELD_OUT_SEEDS)]
    comparison, records = evaluate_policy_set(
        finetuned_path=str(finetuned_path),
        seeds=seeds,
        dataset_root=Path(dataset_root),
        repo_id=repo_id,
        max_steps=int(max_steps),
        include_random=True,
        episodes_per_seed=int(episodes_per_seed),
        image_size=int(image_size),
        canonical_train_cell=canonical_train_cell,
        best_train_state_mode=best_train_state_mode,
        evaluation_backend=evaluation_backend,
    )
    verdict = "claim_supported" if comparison["finetuned_mint"]["success_rate"] > comparison["pretrained_mint"]["success_rate"] else "scientific_not_supported"
    strongest_true_claim = (
        "Fine-tuned MINT improves held-out drawer success in the MuJoCo canonical robot-trajectory simulation."
        if verdict == "claim_supported"
        else "The parity-correct MuJoCo tiny-retrain harness is executable, but fine-tuned MINT does not beat pretrained MINT on held-out drawer variants."
    )
    return {
        "verdict": verdict,
        "held_out_seeds": seeds,
        "eval_max_steps": int(max_steps),
        "episodes_per_seed": int(episodes_per_seed),
        "evaluation_backend": evaluation_backend,
        "canonical_train_cell": canonical_train_cell,
        "best_train_state_mode": best_train_state_mode,
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
    image_size: int = DEFAULT_EVAL_IMAGE_SIZE,
    canonical_train_cell: str = EXPECTED_CANONICAL_TRAIN_CELL,
    best_train_state_mode: str = EXPECTED_BEST_TRAIN_STATE_MODE,
    evaluation_backend: str = EXPECTED_EVALUATION_BACKEND,
) -> tuple[dict, dict]:
    probe_seeds = [int(seed) for seed in (seeds or DEFAULT_TRAIN_SEEDS)]
    comparison, records = evaluate_policy_set(
        finetuned_path=finetuned_path,
        seeds=probe_seeds,
        dataset_root=Path(dataset_root),
        repo_id=repo_id,
        max_steps=int(max_steps),
        include_random=False,
        episodes_per_seed=int(episodes_per_seed),
        image_size=int(image_size),
        canonical_train_cell=canonical_train_cell,
        best_train_state_mode=best_train_state_mode,
        evaluation_backend=evaluation_backend,
    )
    return {
        "eval_max_steps": int(max_steps),
        "train_seeds": probe_seeds,
        "episodes_per_seed": int(episodes_per_seed),
        "evaluation_backend": evaluation_backend,
        "canonical_train_cell": canonical_train_cell,
        "best_train_state_mode": best_train_state_mode,
        "summary": comparison,
        "records": records,
    }, records



def render_report(summary: dict, *, title: str = "# MINT Drawer Robot-Trajectory Sim Evaluation (MuJoCo)") -> str:
    comparison = summary["comparison"]
    lines = [
        title,
        "",
        f"**Verdict**: `{summary['verdict']}`",
        f"**Held-out Seeds**: `{summary['held_out_seeds']}`",
        f"**Eval Max Steps**: `{summary.get('eval_max_steps', 96)}`",
        f"**Evaluation Backend**: `{summary.get('evaluation_backend', EXPECTED_EVALUATION_BACKEND)}`",
        "",
        "| Policy | Success Rate | Grasp Success | Pull Distance | Time to Completion | Episodes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["random", "pretrained_mint", "finetuned_mint"]:
        if name not in comparison:
            continue
        payload = comparison[name]
        lines.append(
            f"| {name} | {payload['success_rate']:.3f} | {payload['grasp_success_rate']:.3f} | {payload['pull_distance_mean']:.3f} | {payload['time_to_completion_mean']:.2f} | {payload['n_episodes']} |"
        )
    lines.extend(["", "## Strongest True Claim", "", summary["strongest_true_claim"], ""])
    return "\n".join(lines)

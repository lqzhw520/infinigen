#!/usr/bin/env python3
"""D2: Overfit on all replay-faithful AnyGrasp rollouts from the strongest single train seed."""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

from dataset_builder import build_dataset_from_rollouts, dataset_integrity
from evaluate_mint_drawer_campaign import evaluate_train_probe
from mint_common import (
    ACTIVE_TEACHER_SOURCE_PATH,
    CAMPAIGN_DIR,
    D2_VARIANT_PROGRESS_PATH,
    DATASET_REPO_ID,
    STRONG_ROLLOUT_AUDIT_PATH,
    load_state,
    reconcile_step_truth,
    write_runtime_meta,
)
from rollout_selection import (
    fingerprint_matches,
    save_fingerprint,
    select_seed_and_subset,
    source_fingerprint,
)
from strict_teacher_dataset import (
    STRICT_TEACHER_AUDIT_PATH,
    STRICT_TEACHER_DIR,
    build_phase_balanced_windows,
    rebuild_strict_teacher_dataset,
)
from train_mint_helpers import find_latest_checkpoint, run_training, wait_for_training_complete
from video_reporting import safe_render_policy_pair, safe_render_teacher_rollout

ARTIFACT = CAMPAIGN_DIR / "artifacts" / "d2_single_seed_overfit.json"
D1_ARTIFACT = CAMPAIGN_DIR / "artifacts" / "d1_candidate_search.json"
DATASET_ROOT = CAMPAIGN_DIR / "dataset_d2_single_seed"
OUTPUT_DIR = CAMPAIGN_DIR / "outputs_d2_single_seed"
LOG_PATH = CAMPAIGN_DIR / "artifacts" / "d2_single_seed_overfit.log"

OVERFIT_VARIANTS = [
    {"id": "extended_overfit", "steps": 1200, "builder_mode": "seed_subset"},
    {"id": "phase_balanced_overfit", "steps": 1200, "builder_mode": "phase_windows"},
]
DERIVED_DIR = CAMPAIGN_DIR / "artifacts" / "d2_phase_windows"
MIN_SOURCE_ROLLOUTS = 2
MIN_SUCCESS_GAIN = 0.2
MIN_FINETUNED_SUCCESSES = 3
MIN_EVAL_EPISODES = 5
BUILDER_VERSION = "d2_v5_attach_contract_hardened"
FINGERPRINT_NAME = ".source_fingerprint.json"


def _write_variant_progress(payload: dict) -> None:
    D2_VARIANT_PROGRESS_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def _lineage_subset_override(
    audit: dict, promoted_seed: int, seed_to_paths: dict[int, list[Path]]
) -> tuple[list[Path], dict] | None:
    lineage_pin = audit.get("teacher_lineage_pin") or {}
    override = lineage_pin.get("d2_subset_override") or {}
    seed_key = str(int(promoted_seed))
    if seed_key not in override:
        return None
    requested = {int(ep) for ep in override.get(seed_key, [])}
    selected_paths = []
    for path in seed_to_paths.get(int(promoted_seed), []):
        meta = json.loads(path.with_suffix(".json").read_text())
        if int(meta.get("episode_index", -1)) in requested:
            selected_paths.append(path)
    if not selected_paths:
        return None
    return selected_paths, {
        "strategy": "teacher_lineage_pin_override",
        "selected_seed": int(promoted_seed),
        "selected_rollouts": [str(path) for path in selected_paths],
        "requested_episode_indices": sorted(requested),
        "teacher_lineage_pin": lineage_pin,
    }


def _strong_subset_for_seed(
    audit: dict, promoted_seed: int
) -> tuple[list[Path], dict] | None:
    for row in audit.get("strong_seed_summary", []) or []:
        if int(row.get("seed", -1)) != int(promoted_seed):
            continue
        if not bool(row.get("coherent")):
            return None
        selected_paths = [
            Path(item["rollout_path"]) for item in row.get("strong_rollouts", [])
        ]
        if len(selected_paths) < MIN_SOURCE_ROLLOUTS:
            return None
        return selected_paths, {
            "strategy": "strong_coherent_subset",
            "selected_seed": int(promoted_seed),
            "selected_rollouts": [str(path) for path in selected_paths],
            "coherent_subset": row.get("coherent_subset"),
            "teacher_lineage_pin": audit.get("teacher_lineage_pin"),
        }
    return None


def _controller_context() -> dict:
    state = load_state()
    controller = state.get("controller") or {}
    expected_controller = controller.get("controller_id")
    expected_run = controller.get("run_id")
    env_controller = os.environ.get("MINT_CONTROLLER_ID")
    env_run = os.environ.get("MINT_RUN_ID")
    if expected_controller and expected_run:
        if env_controller != expected_controller or env_run != expected_run:
            raise RuntimeError(
                "d2_single_seed_overfit launched without matching controller lease; refusing contaminated run"
            )
    return {
        "controller_id": expected_controller or env_controller,
        "run_id": expected_run or env_run,
    }


def _reuse_existing_training(
    output_dir: Path, log_path: Path, steps: int
) -> dict | None:
    checkpoint_path = find_latest_checkpoint(output_dir)
    if not checkpoint_path:
        return None
    log_tail = log_path.read_text(errors="ignore")[-8000:] if log_path.exists() else ""
    loss_lines = [
        line.strip() for line in log_tail.splitlines() if "loss" in line.lower()
    ]
    return {
        "passed": True,
        "returncode": 0,
        "elapsed_sec": 0.0,
        "steps_completed": steps,
        "checkpoint_path": str(checkpoint_path),
        "loss_samples": loss_lines[-10:],
        "stdout_tail": log_tail,
        "stderr_tail": "",
        "reused_existing_checkpoint": True,
    }


def _reuse_existing_attempt(
    dataset_root: Path,
    repo_id: str,
    output_dir: Path,
    log_path: Path,
    steps: int,
    expected_episode_count: int,
    fingerprint: dict,
) -> tuple[dict, dict] | None:
    dataset_fp = dataset_root / FINGERPRINT_NAME
    output_fp = output_dir / FINGERPRINT_NAME
    integrity = dataset_integrity(dataset_root, repo_id)
    training = _reuse_existing_training(output_dir, log_path, steps)
    if not integrity.get("passed") or training is None:
        return None
    if not fingerprint_matches(dataset_fp, fingerprint) or not fingerprint_matches(
        output_fp, fingerprint
    ):
        return None
    dataset_payload = {
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "episode_count": int(expected_episode_count),
        "frame_count": int(integrity.get("dataset_length", 0)),
        "seed_coverage": [],
        "task_coverage": [],
        "source_files": [],
        "integrity": integrity,
        "reused_existing_dataset": True,
        "fingerprint": fingerprint,
    }
    training["fingerprint"] = fingerprint
    training["reused_existing_checkpoint"] = True
    return dataset_payload, training


def _reset_stale_paths(dataset_root: Path, output_dir: Path) -> None:
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    if output_dir.exists():
        shutil.rmtree(output_dir)


def run() -> bool:
    controller = _controller_context()
    write_runtime_meta(
        "d2_single_seed_overfit",
        {
            "step_id": "d2_single_seed_overfit",
            "status": "running",
            "worker_pid": os.getpid(),
            "env": "mint",
            "controller_id": controller.get("controller_id"),
            "run_id": controller.get("run_id"),
            "launched_at": time.time(),
            "active_teacher_source_path": str(ACTIVE_TEACHER_SOURCE_PATH),
            "strong_rollout_audit_path": str(STRONG_ROLLOUT_AUDIT_PATH),
        },
    )
    audit = rebuild_strict_teacher_dataset()
    env_contract = audit.get("env_contract_audit", {})
    d2_feasible_seeds = {int(seed) for seed in audit.get("d2_feasible_seeds", [])}
    strict_rollouts = sorted(STRICT_TEACHER_DIR.glob("*.npz"))
    d1 = json.loads(D1_ARTIFACT.read_text()) if D1_ARTIFACT.exists() else {}
    promoted = d1.get("promoted_candidate") or {}
    promoted_seed = promoted.get("selected_seed")
    if env_contract and not env_contract.get("passed"):
        result = {
            "gate": "d2_single_seed_overfit",
            "passed": False,
            "controller_id": controller.get("controller_id"),
            "run_id": controller.get("run_id"),
            "error": "Environment contract audit failed before D2.",
            "strict_teacher_source_dir": audit.get("source_dir"),
            "teacher_lineage_pin": audit.get("teacher_lineage_pin"),
            "env_contract_audit": env_contract,
            "strict_teacher_audit": audit,
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        reconcile_step_truth(
            "d2_single_seed_overfit",
            passed=False,
            latest_artifact=str(ARTIFACT),
            last_error="env_contract_blocked",
        )
        print(json.dumps(result, indent=2))
        return False
    seed_to_paths = defaultdict(list)
    for npz_path in strict_rollouts:
        meta = json.loads(npz_path.with_suffix(".json").read_text())
        seed_to_paths[int(meta["seed"])].append(npz_path)
    if not seed_to_paths:
        result = {
            "gate": "d2_single_seed_overfit",
            "passed": False,
            "controller_id": controller.get("controller_id"),
            "run_id": controller.get("run_id"),
            "error": "No strict-valid AnyGrasp rollouts available",
            "strict_teacher_source_dir": audit.get("source_dir"),
            "teacher_lineage_pin": audit.get("teacher_lineage_pin"),
            "strict_teacher_audit": audit,
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        reconcile_step_truth(
            "d2_single_seed_overfit",
            passed=False,
            latest_artifact=str(ARTIFACT),
            last_error="no_strict_valid_rollouts",
        )
        print(json.dumps(result, indent=2))
        return False
    if promoted_seed is None:
        result = {
            "gate": "d2_single_seed_overfit",
            "passed": False,
            "controller_id": controller.get("controller_id"),
            "run_id": controller.get("run_id"),
            "error": "D1 has not produced a promoted candidate yet",
            "strict_teacher_source_dir": audit.get("source_dir"),
            "teacher_lineage_pin": audit.get("teacher_lineage_pin"),
            "strict_teacher_audit": audit,
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        reconcile_step_truth(
            "d2_single_seed_overfit",
            passed=False,
            latest_artifact=str(ARTIFACT),
            last_error="d1_not_promoted",
        )
        print(json.dumps(result, indent=2))
        return False
    if str(promoted.get("candidate_role") or "") != "mainline":
        result = {
            "gate": "d2_single_seed_overfit",
            "passed": False,
            "controller_id": controller.get("controller_id"),
            "run_id": controller.get("run_id"),
            "error": "Promoted D1 candidate is diagnostic-only; rerun D1 mainline before D2",
            "strict_teacher_source_dir": audit.get("source_dir"),
            "teacher_lineage_pin": audit.get("teacher_lineage_pin"),
            "strict_teacher_audit": audit,
            "promoted_candidate": promoted,
            "d2_feasible_seeds": sorted(d2_feasible_seeds),
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        reconcile_step_truth(
            "d2_single_seed_overfit",
            passed=False,
            latest_artifact=str(ARTIFACT),
            last_error="d1_promoted_candidate_not_mainline",
        )
        print(json.dumps(result, indent=2))
        return False
    if int(promoted_seed) not in d2_feasible_seeds:
        result = {
            "gate": "d2_single_seed_overfit",
            "passed": False,
            "controller_id": controller.get("controller_id"),
            "run_id": controller.get("run_id"),
            "error": "Promoted D1 seed is not D2-feasible under the current strict-valid teacher pool",
            "strict_teacher_source_dir": audit.get("source_dir"),
            "teacher_lineage_pin": audit.get("teacher_lineage_pin"),
            "strict_teacher_audit": audit,
            "promoted_candidate": promoted,
            "d2_feasible_seeds": sorted(d2_feasible_seeds),
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        reconcile_step_truth(
            "d2_single_seed_overfit",
            passed=False,
            latest_artifact=str(ARTIFACT),
            last_error="insufficient_single_seed_teacher_data",
        )
        print(json.dumps(result, indent=2))
        return False
    if int(promoted_seed) not in seed_to_paths:
        result = {
            "gate": "d2_single_seed_overfit",
            "passed": False,
            "controller_id": controller.get("controller_id"),
            "run_id": controller.get("run_id"),
            "error": f"Promoted D1 seed {promoted_seed} is not present in strict-valid teacher pool",
            "strict_teacher_source_dir": audit.get("source_dir"),
            "teacher_lineage_pin": audit.get("teacher_lineage_pin"),
            "strict_teacher_audit": audit,
            "promoted_candidate": promoted,
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        reconcile_step_truth(
            "d2_single_seed_overfit",
            passed=False,
            latest_artifact=str(ARTIFACT),
            last_error="promoted_seed_missing_from_teacher_pool",
        )
        print(json.dumps(result, indent=2))
        return False
    strong_subset = _strong_subset_for_seed(audit, int(promoted_seed))
    lineage_override = _lineage_subset_override(
        audit, int(promoted_seed), seed_to_paths
    )
    if strong_subset is not None:
        selected_paths, selection_summary = strong_subset
        selected_seed = int(promoted_seed)
    elif lineage_override is not None:
        selected_paths, selection_summary = lineage_override
        selected_seed = int(promoted_seed)
    else:
        selected_seed, selected_paths, selection_summary = select_seed_and_subset(
            {int(promoted_seed): seed_to_paths[int(promoted_seed)]}
        )
    source_rollout_count = len(selected_paths)
    if source_rollout_count < MIN_SOURCE_ROLLOUTS:
        result = {
            "gate": "d2_single_seed_overfit",
            "passed": False,
            "controller_id": controller.get("controller_id"),
            "run_id": controller.get("run_id"),
            "error": "Single-seed teacher data is insufficient for D2 mainline",
            "strict_teacher_source_dir": audit.get("source_dir"),
            "teacher_lineage_pin": audit.get("teacher_lineage_pin"),
            "strict_teacher_audit": audit,
            "promoted_candidate": promoted,
            "selection_summary": selection_summary,
            "selected_seed": selected_seed,
            "selected_rollouts": [str(path) for path in selected_paths],
            "source_rollout_count": source_rollout_count,
            "min_source_rollouts": MIN_SOURCE_ROLLOUTS,
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        reconcile_step_truth(
            "d2_single_seed_overfit",
            passed=False,
            latest_artifact=str(ARTIFACT),
            last_error="insufficient_single_seed_teacher_data",
        )
        print(json.dumps(result, indent=2))
        return False
    if strong_subset is None:
        result = {
            "gate": "d2_single_seed_overfit",
            "passed": False,
            "controller_id": controller.get("controller_id"),
            "run_id": controller.get("run_id"),
            "error": "Selected seed does not yet provide 2 strong coherent rollouts for clean D2",
            "strict_teacher_audit_path": str(STRICT_TEACHER_AUDIT_PATH),
            "strict_teacher_source_dir": audit.get("source_dir"),
            "teacher_lineage_pin": audit.get("teacher_lineage_pin"),
            "active_teacher_source": json.loads(ACTIVE_TEACHER_SOURCE_PATH.read_text())
            if ACTIVE_TEACHER_SOURCE_PATH.exists()
            else {},
            "strong_rollout_audit": json.loads(STRONG_ROLLOUT_AUDIT_PATH.read_text())
            if STRONG_ROLLOUT_AUDIT_PATH.exists()
            else {},
            "selection_summary": selection_summary,
            "selected_seed": selected_seed,
            "selected_rollouts": [str(path) for path in selected_paths],
            "source_rollout_count": source_rollout_count,
            "failure_reason": "data_quality_floor_not_met",
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        _write_variant_progress(
            {
                "step": "d2_single_seed_overfit",
                "status": "failed",
                "controller_id": controller.get("controller_id"),
                "run_id": controller.get("run_id"),
                "selected_seed": selected_seed,
                "selected_rollouts": [str(path) for path in selected_paths],
                "source_rollout_count": source_rollout_count,
                "failure_reason": "data_quality_floor_not_met",
            }
        )
        reconcile_step_truth(
            "d2_single_seed_overfit",
            passed=False,
            latest_artifact=str(ARTIFACT),
            last_error="data_quality_floor_not_met",
        )
        print(json.dumps(result, indent=2))
        return False
    attempts = []
    winner = None
    for variant in OVERFIT_VARIANTS:
        _write_variant_progress(
            {
                "step": "d2_single_seed_overfit",
                "status": "running",
                "controller_id": controller.get("controller_id"),
                "run_id": controller.get("run_id"),
                "selected_seed": selected_seed,
                "selected_rollouts": [str(path) for path in selected_paths],
                "source_rollout_count": source_rollout_count,
                "variant_id": variant["id"],
                "variant_stage": "building_or_reusing_dataset",
                "checkpoint_written": False,
                "evaluation_started": False,
                "evaluation_finished": False,
            }
        )
        dataset_root = Path(f"{DATASET_ROOT}_{variant['id']}")
        output_dir = Path(f"{OUTPUT_DIR}_{variant['id']}")
        log_path = Path(f"{LOG_PATH}.{variant['id']}.log")
        dataset_repo_id = f"{DATASET_REPO_ID}_d2_{variant['id']}"
        if variant["builder_mode"] == "phase_windows":
            rollout_batch = build_phase_balanced_windows(
                selected_paths, DERIVED_DIR / variant["id"]
            )
            balancing_mode = "phase_windows"
        else:
            rollout_batch = list(selected_paths)
            balancing_mode = "seed_subset"
        fingerprint = source_fingerprint(
            rollout_batch,
            contract_mode="teacher_success_fallback",
            source_branch_id=selection_summary.get("selected_seed_stats", {})
            .get("selected_subset", {})
            .get("rollouts", [{}])[0]
            .get("branch_id"),
            builder_version=BUILDER_VERSION,
            variant_id=variant["id"],
            rollout_multiplier=len(rollout_batch),
            balancing_mode=balancing_mode,
            selected_seed_ids=[selected_seed],
        )
        reused = _reuse_existing_attempt(
            dataset_root,
            dataset_repo_id,
            output_dir,
            log_path,
            int(variant["steps"]),
            len(rollout_batch),
            fingerprint,
        )
        if reused is not None:
            dataset_payload, train_payload = reused
        else:
            _reset_stale_paths(dataset_root, output_dir)
            dataset_payload = build_dataset_from_rollouts(
                rollout_batch, dataset_root, dataset_repo_id
            )
            dataset_payload["fingerprint"] = fingerprint
            save_fingerprint(dataset_root / FINGERPRINT_NAME, fingerprint)
            
            # Launch training in async mode (non-blocking)
            train_payload = run_training(
                dataset_root=dataset_root,
                dataset_repo_id=dataset_repo_id,
                output_dir=output_dir,
                log_path=log_path,
                steps=int(variant["steps"]),
                job_name=f"mint_d2_{variant['id']}",
                async_mode=True,
            )
            
            # If async mode returned a placeholder, wait for training to complete
            if train_payload.get("async_mode") and train_payload.get("pid"):
                _write_variant_progress(
                    {
                        "step": "d2_single_seed_overfit",
                        "status": "running",
                        "controller_id": controller.get("controller_id"),
                        "run_id": controller.get("run_id"),
                        "selected_seed": selected_seed,
                        "selected_rollouts": [str(path) for path in selected_paths],
                        "source_rollout_count": source_rollout_count,
                        "variant_id": variant["id"],
                        "variant_stage": "training_async_wait",
                        "training_pid": train_payload["pid"],
                    }
                )
                # Wait for training to complete (polls for checkpoint)
                train_result = wait_for_training_complete(
                    output_dir=output_dir,
                    log_path=log_path,
                    steps=int(variant["steps"]),
                    poll_interval=30.0,
                    max_wait_hours=12.0,
                )
                # Merge results
                train_payload.update({
                    "passed": train_result.get("completed", False) or train_result.get("checkpoint_path") is not None,
                    "returncode": train_result.get("returncode", 0),
                    "elapsed_sec": train_result.get("elapsed_sec", 0.0),
                    "steps_completed": train_result.get("steps_completed", 0),
                    "checkpoint_path": train_result.get("checkpoint_path"),
                    "stdout_tail": train_result.get("log_tail", ""),
                    "status": "completed",
                })
                # Clean up PID file
                pid_file = output_dir / ".training.pid"
                if pid_file.exists():
                    pid_file.unlink()
            
            save_fingerprint(output_dir / FINGERPRINT_NAME, fingerprint)
            train_payload["fingerprint"] = fingerprint
        _write_variant_progress(
            {
                "step": "d2_single_seed_overfit",
                "status": "running",
                "controller_id": controller.get("controller_id"),
                "run_id": controller.get("run_id"),
                "selected_seed": selected_seed,
                "selected_rollouts": [str(path) for path in selected_paths],
                "source_rollout_count": source_rollout_count,
                "variant_id": variant["id"],
                "variant_stage": "checkpoint_ready"
                if train_payload.get("checkpoint_path")
                else "training_failed",
                "checkpoint_written": bool(train_payload.get("checkpoint_path")),
                "evaluation_started": False,
                "evaluation_finished": False,
            }
        )
        finetuned_path = train_payload.get("checkpoint_path")
        eval_payload = {}
        trend_passed = False
        if finetuned_path:
            _write_variant_progress(
                {
                    "step": "d2_single_seed_overfit",
                    "status": "running",
                    "controller_id": controller.get("controller_id"),
                    "run_id": controller.get("run_id"),
                    "selected_seed": selected_seed,
                    "selected_rollouts": [str(path) for path in selected_paths],
                    "source_rollout_count": source_rollout_count,
                    "variant_id": variant["id"],
                    "variant_stage": "strict_evaluation",
                    "checkpoint_written": True,
                    "evaluation_started": True,
                    "evaluation_finished": False,
                    "checkpoint_path": finetuned_path,
                }
            )
            eval_episodes = max(MIN_EVAL_EPISODES, source_rollout_count)
            eval_payload, _ = evaluate_train_probe(
                finetuned_path,
                seeds=[selected_seed],
                dataset_root=dataset_root,
                repo_id=dataset_repo_id,
                max_steps=96,
                episodes_per_seed=eval_episodes,
            )
            ft = eval_payload["summary"]["finetuned_mint"]
            pt = eval_payload["summary"]["pretrained_mint"]
            success_gain = float(ft["success_rate"] - pt["success_rate"])
            trend_passed = bool(
                source_rollout_count >= MIN_SOURCE_ROLLOUTS
                and success_gain >= MIN_SUCCESS_GAIN
                and ft["successes"] >= MIN_FINETUNED_SUCCESSES
                and ft["pull_distance_mean"] > pt["pull_distance_mean"]
            )
            eval_payload["source_rollout_count"] = source_rollout_count
            eval_payload["success_gain"] = success_gain
            eval_payload["min_success_gain"] = MIN_SUCCESS_GAIN
            eval_payload["min_finetuned_successes"] = MIN_FINETUNED_SUCCESSES
            eval_payload["min_source_rollouts"] = MIN_SOURCE_ROLLOUTS
            eval_payload["episodes_per_seed"] = eval_episodes
            _write_variant_progress(
                {
                    "step": "d2_single_seed_overfit",
                    "status": "running",
                    "controller_id": controller.get("controller_id"),
                    "run_id": controller.get("run_id"),
                    "selected_seed": selected_seed,
                    "selected_rollouts": [str(path) for path in selected_paths],
                    "source_rollout_count": source_rollout_count,
                    "variant_id": variant["id"],
                    "variant_stage": "evaluation_complete",
                    "checkpoint_written": True,
                    "evaluation_started": True,
                    "evaluation_finished": True,
                    "trend_passed": trend_passed,
                    "finetuned_successes": int(ft["successes"]),
                    "pretrained_successes": int(pt["successes"]),
                    "success_gain": success_gain,
                }
            )
        attempt = {
            "variant": variant["id"],
            "steps": variant["steps"],
            "selected_seed": selected_seed,
            "source_rollout_count": source_rollout_count,
            "training_rollout_count": len(rollout_batch),
            "builder_mode": variant["builder_mode"],
            "source_fingerprint": fingerprint,
            "dataset_root": str(dataset_root),
            "output_dir": str(output_dir),
            "dataset": dataset_payload,
            "training": train_payload,
            "evaluation": eval_payload,
            "trend_passed": trend_passed,
        }
        attempts.append(attempt)
        if (
            dataset_payload["integrity"]["passed"]
            and train_payload["passed"]
            and trend_passed
        ):
            winner = attempt
            break

    if winner:
        winner_dataset_root = Path(winner["dataset_root"])
        winner_output = Path(winner["output_dir"])
        if DATASET_ROOT.exists():
            shutil.rmtree(DATASET_ROOT)
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        shutil.copytree(winner_dataset_root, DATASET_ROOT)
        shutil.copytree(winner_output, OUTPUT_DIR)
    video_outputs = {}
    if winner and selected_paths:
        try:
            video_outputs["teacher"] = safe_render_teacher_rollout(
                campaign_dir=CAMPAIGN_DIR,
                rollout_path=str(selected_paths[0]),
                stage="d2",
                policy="d2_mainline_teacher",
                attempt="0",
                variant=winner["variant"],
            )
            video_outputs["policy_pair"] = safe_render_policy_pair(
                campaign_dir=CAMPAIGN_DIR,
                stage="d2",
                seed=int(selected_seed),
                finetuned_path=str(winner["training"]["checkpoint_path"]),
                dataset_root=str(winner["dataset"]["dataset_root"]),
                repo_id=str(winner["dataset"]["repo_id"]),
                variant=str(winner["variant"]),
                max_steps=96,
                attempt_idx=0,
            )
        except (
            Exception
        ) as exc:  # pragma: no cover - reporting should not fail the gate
            video_outputs["error"] = f"{type(exc).__name__}: {exc}"
    result = {
        "gate": "d2_single_seed_overfit",
        "passed": winner is not None,
        "controller_id": controller.get("controller_id"),
        "run_id": controller.get("run_id"),
        "strict_teacher_audit_path": str(STRICT_TEACHER_AUDIT_PATH),
        "strict_teacher_source_dir": audit.get("source_dir"),
        "teacher_lineage_pin": audit.get("teacher_lineage_pin"),
        "active_teacher_source": json.loads(ACTIVE_TEACHER_SOURCE_PATH.read_text())
        if ACTIVE_TEACHER_SOURCE_PATH.exists()
        else {},
        "strong_rollout_audit": json.loads(STRONG_ROLLOUT_AUDIT_PATH.read_text())
        if STRONG_ROLLOUT_AUDIT_PATH.exists()
        else {},
        "strict_teacher_audit": audit,
        "promoted_candidate": promoted,
        "selection_summary": selection_summary,
        "selected_seed": selected_seed,
        "selected_rollouts": [str(path) for path in selected_paths],
        "winner": winner,
        "attempts": attempts,
        "video_outputs": video_outputs,
        "failure_reason": None
        if winner is not None
        else (
            "data_quality_floor_not_met"
            if strong_subset is None
            else "u3_patch_insufficient"
        ),
        "timestamp": time.time(),
    }
    ARTIFACT.write_text(json.dumps(result, indent=2))
    _write_variant_progress(
        {
            "step": "d2_single_seed_overfit",
            "status": "completed" if winner is not None else "failed",
            "controller_id": controller.get("controller_id"),
            "run_id": controller.get("run_id"),
            "selected_seed": selected_seed,
            "selected_rollouts": [str(path) for path in selected_paths],
            "source_rollout_count": source_rollout_count,
            "winner_variant": None if winner is None else winner["variant"],
            "failure_reason": None if winner is not None else result["failure_reason"],
        }
    )
    reconcile_step_truth(
        "d2_single_seed_overfit",
        passed=bool(result["passed"]),
        latest_artifact=str(ARTIFACT),
        last_error=None if result["passed"] else result["failure_reason"],
        active_branch=f"seed_{selected_seed:03d}" if winner else None,
        active_fingerprint=None if winner is None else winner.get("source_fingerprint"),
    )
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

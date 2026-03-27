#!/usr/bin/env python3
"""D3: Train on replay-faithful AnyGrasp rollouts and require a positive train-seed trend."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from dataset_builder import build_dataset_from_rollouts
from evaluate_mint_drawer_campaign import evaluate_train_probe
from mint_common import (
    CAMPAIGN_DIR,
    DATASET_REPO_ID,
    DEFAULT_TRAIN_SEEDS,
    reconcile_step_truth,
)
from rollout_selection import (
    build_coherent_multiseed_subset,
    save_fingerprint,
    source_fingerprint,
)
from strict_teacher_dataset import (
    STRICT_TEACHER_AUDIT_PATH,
    STRICT_TEACHER_DIR,
    build_phase_balanced_windows,
    rebuild_strict_teacher_dataset,
)
from train_mint_helpers import run_training
from video_reporting import (
    choose_representative_seed,
    safe_render_policy_pair,
    safe_render_teacher_rollout,
)

ARTIFACT = CAMPAIGN_DIR / "artifacts" / "d3_train_seed_probe.json"
D2_ARTIFACT = CAMPAIGN_DIR / "artifacts" / "d2_single_seed_overfit.json"
DATASET_ROOT = CAMPAIGN_DIR / "dataset_d3_train"
OUTPUT_DIR = CAMPAIGN_DIR / "outputs_d3_train"
LOG_PATH = CAMPAIGN_DIR / "artifacts" / "d3_train_seed_probe.log"
SUMMARY_PATH = CAMPAIGN_DIR / "evaluation" / "train_seed_probe.json"
DERIVED_DIR = CAMPAIGN_DIR / "artifacts" / "d3_phase_windows"

MIN_SUCCESS_GAIN = 0.15
MIN_FINETUNED_SUCCESSES = 2
MIN_EVAL_EPISODES = 3
MIN_COVERED_SEEDS = 4
MIN_TOTAL_ROLLOUTS = 8
MIN_TOTAL_FRAMES = 300
BUILDER_VERSION = "d3_v3_attach_contract_hardened"
FINGERPRINT_NAME = ".source_fingerprint.json"


def _reset_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _improving_seed_count(covered_probe: dict) -> int:
    ft = covered_probe.get("summary", {}).get("finetuned_mint", {}).get("per_seed", {})
    pt = covered_probe.get("summary", {}).get("pretrained_mint", {}).get("per_seed", {})
    improving = 0
    for seed, ft_metrics in ft.items():
        pt_metrics = pt.get(seed, {})
        if float(ft_metrics.get("success_rate", 0.0)) > float(
            pt_metrics.get("success_rate", 0.0)
        ):
            improving += 1
    return improving


def run() -> bool:
    audit = rebuild_strict_teacher_dataset()
    env_contract = audit.get("env_contract_audit", {})
    d2_payload = json.loads(D2_ARTIFACT.read_text()) if D2_ARTIFACT.exists() else {}
    if env_contract and not env_contract.get("passed"):
        result = {
            "gate": "d3_train_seed_probe",
            "passed": False,
            "error": "Environment contract audit failed before D3.",
            "strict_teacher_audit": audit,
            "env_contract_audit": env_contract,
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        reconcile_step_truth(
            "d3_train_seed_probe",
            passed=False,
            latest_artifact=str(ARTIFACT),
            last_error="env_contract_blocked",
        )
        print(json.dumps(result, indent=2))
        return False
    if not d2_payload.get("passed"):
        result = {
            "gate": "d3_train_seed_probe",
            "passed": False,
            "error": "D2 must pass before D3 can start",
            "strict_teacher_audit": audit,
            "d2_payload_present": bool(d2_payload),
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        reconcile_step_truth(
            "d3_train_seed_probe",
            passed=False,
            latest_artifact=str(ARTIFACT),
            last_error="d2_not_passed",
        )
        print(json.dumps(result, indent=2))
        return False

    rollout_paths = sorted(STRICT_TEACHER_DIR.glob("*.npz"))
    if not rollout_paths:
        result = {
            "gate": "d3_train_seed_probe",
            "passed": False,
            "error": "No strict-valid AnyGrasp rollouts available",
            "strict_teacher_audit": audit,
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        reconcile_step_truth(
            "d3_train_seed_probe",
            passed=False,
            latest_artifact=str(ARTIFACT),
            last_error="no_strict_valid_rollouts",
        )
        print(json.dumps(result, indent=2))
        return False

    coherent_paths, subset_summary = build_coherent_multiseed_subset(
        rollout_paths, min_rollouts_per_seed=2
    )
    total_selected_rollouts = int(
        sum(
            int(item.get("selected_rollout_count", 0))
            for item in subset_summary.get("included", [])
        )
    )
    if (
        len(subset_summary["included_seeds"]) < MIN_COVERED_SEEDS
        or total_selected_rollouts < MIN_TOTAL_ROLLOUTS
    ):
        result = {
            "gate": "d3_train_seed_probe",
            "passed": False,
            "error": "Coherent multi-seed training subset is too small for mainline D3",
            "selection_summary": subset_summary,
            "min_covered_seeds": MIN_COVERED_SEEDS,
            "min_total_rollouts": MIN_TOTAL_ROLLOUTS,
            "total_selected_rollouts": total_selected_rollouts,
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        reconcile_step_truth(
            "d3_train_seed_probe",
            passed=False,
            latest_artifact=str(ARTIFACT),
            last_error="coherent_multiseed_subset_too_small",
        )
        print(json.dumps(result, indent=2))
        return False

    dataset_repo_id = f"{DATASET_REPO_ID}_d3_train_probe"
    training_paths = build_phase_balanced_windows(
        coherent_paths, DERIVED_DIR / "train_probe"
    )
    _reset_path(DATASET_ROOT)
    _reset_path(OUTPUT_DIR)
    dataset_payload = build_dataset_from_rollouts(
        training_paths, DATASET_ROOT, dataset_repo_id
    )
    dataset_fingerprint = source_fingerprint(
        training_paths,
        contract_mode="teacher_success_fallback",
        source_branch_id="coherent_multiseed_subset",
        builder_version=BUILDER_VERSION,
        variant_id="train_seed_probe",
        rollout_multiplier=len(training_paths),
        balancing_mode="coherent_per_seed_phase_windows",
        selected_seed_ids=subset_summary["included_seeds"],
    )
    dataset_payload["selection_summary"] = subset_summary
    dataset_payload["strict_teacher_audit"] = audit
    dataset_payload["fingerprint"] = dataset_fingerprint
    dataset_payload["total_selected_rollouts"] = total_selected_rollouts
    save_fingerprint(DATASET_ROOT / FINGERPRINT_NAME, dataset_fingerprint)
    if int(dataset_payload.get("frame_count", 0)) < MIN_TOTAL_FRAMES:
        result = {
            "gate": "d3_train_seed_probe",
            "passed": False,
            "error": "Mainline D3 training dataset is too small after dataset build",
            "selection_summary": subset_summary,
            "dataset": dataset_payload,
            "min_total_frames": MIN_TOTAL_FRAMES,
            "timestamp": time.time(),
        }
        ARTIFACT.write_text(json.dumps(result, indent=2))
        reconcile_step_truth(
            "d3_train_seed_probe",
            passed=False,
            latest_artifact=str(ARTIFACT),
            last_error="d3_dataset_too_small",
        )
        print(json.dumps(result, indent=2))
        return False
    train_payload = run_training(
        dataset_root=DATASET_ROOT,
        dataset_repo_id=dataset_repo_id,
        output_dir=OUTPUT_DIR,
        log_path=LOG_PATH,
        steps=1000,
        job_name="mint_d3_train_seed_probe",
    )
    save_fingerprint(OUTPUT_DIR / FINGERPRINT_NAME, dataset_fingerprint)
    checkpoint_path = train_payload.get("checkpoint_path")
    covered_probe = {}
    full_train_probe = {}
    trend_passed = False
    if checkpoint_path:
        covered_seeds = (
            subset_summary["included_seeds"]
            or dataset_payload.get("seed_coverage")
            or DEFAULT_TRAIN_SEEDS
        )
        covered_eval_episodes = max(
            MIN_EVAL_EPISODES,
            max(
                (
                    int(item.get("selected_rollout_count", 0))
                    for item in subset_summary.get("included", [])
                ),
                default=MIN_EVAL_EPISODES,
            ),
        )
        covered_probe, _ = evaluate_train_probe(
            checkpoint_path,
            seeds=covered_seeds,
            dataset_root=DATASET_ROOT,
            repo_id=dataset_repo_id,
            max_steps=96,
            episodes_per_seed=covered_eval_episodes,
        )
        ft = covered_probe["summary"]["finetuned_mint"]
        pt = covered_probe["summary"]["pretrained_mint"]
        success_gain = float(ft["success_rate"] - pt["success_rate"])
        improving_seed_count = _improving_seed_count(covered_probe)
        trend_passed = bool(
            success_gain >= MIN_SUCCESS_GAIN
            and ft["successful_seed_count"] >= MIN_FINETUNED_SUCCESSES
            and improving_seed_count >= 2
        )
        covered_probe["success_gain"] = success_gain
        covered_probe["trend_passed"] = trend_passed
        covered_probe["min_success_gain"] = MIN_SUCCESS_GAIN
        covered_probe["min_finetuned_successful_seed_count"] = MIN_FINETUNED_SUCCESSES
        covered_probe["improving_seed_count"] = improving_seed_count
        covered_probe["episodes_per_seed"] = covered_eval_episodes
        if trend_passed and covered_seeds != DEFAULT_TRAIN_SEEDS:
            full_train_probe, _ = evaluate_train_probe(
                checkpoint_path,
                seeds=DEFAULT_TRAIN_SEEDS,
                dataset_root=DATASET_ROOT,
                repo_id=dataset_repo_id,
                max_steps=96,
                episodes_per_seed=covered_eval_episodes,
            )
        SUMMARY_PATH.write_text(
            json.dumps(
                {
                    "covered_seed_probe": covered_probe,
                    "full_train_probe": full_train_probe,
                    "selection_summary": subset_summary,
                    "covered_seed_count": len(covered_seeds),
                    "full_train_seed_count": len(DEFAULT_TRAIN_SEEDS),
                },
                indent=2,
            )
        )

    result = {
        "gate": "d3_train_seed_probe",
        "passed": bool(
            dataset_payload["integrity"]["passed"]
            and train_payload["passed"]
            and trend_passed
        ),
        "dataset": dataset_payload,
        "training": train_payload,
        "strict_teacher_audit_path": str(STRICT_TEACHER_AUDIT_PATH),
        "d2_artifact": d2_payload,
        "selection_summary": subset_summary,
        "summary": {
            "covered_seed_probe": covered_probe,
            "full_train_probe": full_train_probe,
        },
        "train_seed_count": len(DEFAULT_TRAIN_SEEDS),
        "timestamp": time.time(),
    }
    video_outputs = {}
    checkpoint_path = train_payload.get("checkpoint_path")
    rep_seed = choose_representative_seed((covered_probe or {}).get("summary", {}))
    if checkpoint_path and rep_seed is not None and coherent_paths:
        try:
            video_outputs["teacher"] = safe_render_teacher_rollout(
                campaign_dir=CAMPAIGN_DIR,
                rollout_path=str(coherent_paths[0]),
                stage="d3",
                policy="d3_mainline_teacher",
                attempt="0",
                variant="covered_probe",
            )
            video_outputs["policy_pair"] = safe_render_policy_pair(
                campaign_dir=CAMPAIGN_DIR,
                stage="d3",
                seed=int(rep_seed),
                finetuned_path=str(checkpoint_path),
                dataset_root=str(dataset_payload["dataset_root"]),
                repo_id=str(dataset_payload["repo_id"]),
                variant="covered_probe",
                max_steps=96,
                attempt_idx=0,
            )
        except (
            Exception
        ) as exc:  # pragma: no cover - reporting should not fail the gate
            video_outputs["error"] = f"{type(exc).__name__}: {exc}"
    result["video_outputs"] = video_outputs
    ARTIFACT.write_text(json.dumps(result, indent=2))
    reconcile_step_truth(
        "d3_train_seed_probe",
        passed=bool(result["passed"]),
        latest_artifact=str(ARTIFACT),
        last_error=None if result["passed"] else "no_multiseed_positive_train_trend",
        active_branch="covered_strict_valid_seeds"
        if not result["passed"]
        else "full_train_probe",
        active_fingerprint=dataset_fingerprint,
    )
    print(json.dumps(result, indent=2))
    return bool(result["passed"])


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

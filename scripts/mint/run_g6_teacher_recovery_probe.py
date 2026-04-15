#!/usr/bin/env python3
"""T2a/T2b hard-seed assays for v8.3."""

from __future__ import annotations

from dataclasses import replace
import json
import time
from typing import Any

import numpy as np

from drawer_robot_env_mujoco import build_robot_rollout
from mint_common import ARTIFACT_DIR, TINY_RETRAIN_PLAN_PATH, load_json, write_json_atomic
from root_cause_controller import RootCauseController
from tiny_retrain_mainline import (
    _augment_rollout_metadata,
    _plan_active_state_mode_name,
    _plan_source_canonical_train_cell,
    expected_training_targets,
)

ARTIFACT = ARTIFACT_DIR / "g6_teacher_recovery_probe.json"
HARD_SEEDS = [2, 4]
REPEATS = [0, 1]
T2B_WARM_START_KINDS = ["contact_aligned", "attached_phase_locked"]


def _default_plan() -> dict[str, Any]:
    return {
        "run_instance_id": None,
        "plan_version": "tiny_retrain_confirmation_v8_3",
        "source_base_commit": "20a6498065922b91ceb9030674971bd08082ae6f",
        "source_canonical_train_cell": "V1cT2S0",
        "source_best_train_state_mode": "S0",
        "active_train_state_mode": "S0",
        "active_state_mode_name": "m0_proxy",
        "bridge_stage": "t2a_t2b_teacher_recovery_assay",
        "bridge_attempt": "honest",
        "teacher_truth_gate": "trace_window_v2",
        "strict_utility_version": "v3_attached_open_contract",
        "truth_utility_version": "trace_window_v2",
        "teacher_fingerprint_version": "v8_3_fingerprint_v1",
    }


def _variants() -> list[dict[str, Any]]:
    out = [{"variant_family": "baseline", "variant_value": "baseline", "interventions": {}}]
    for value in [0, 2, 4]:
        out.append({"variant_family": "close_settle_steps", "variant_value": value, "interventions": {"close_settle_steps": value}})
    for value in [0.10, 0.20, 0.30]:
        out.append({"variant_family": "pull_speed", "variant_value": value, "interventions": {"pull_speed": value}})
    for value in [0.02, 0.03, 0.04]:
        out.append({"variant_family": "pull_target_offset_along_axis", "variant_value": value, "interventions": {"pull_target_offset_along_axis": value}})
    for value in [0.00, 0.01, 0.02]:
        out.append({"variant_family": "micro_retract_before_pull", "variant_value": value, "interventions": {"micro_retract_before_pull": value}})
    for value in [0.8, 1.0, 1.2]:
        out.append({"variant_family": "orientation_hold_gain", "variant_value": value, "interventions": {"orientation_hold_gain": value}})
    return out


def _canonical_rollout_context(plan: dict[str, Any]) -> dict[str, Any]:
    expected = {**expected_training_targets(), **plan}
    controller = RootCauseController(
        max_experiments_per_cycle=1,
        dry_run=False,
        experiment_family="RCA",
        selector_mode="frozen_v5_pro",
        seed_split_a=expected.get("train_seeds") or expected.get("training_seeds") or [1, 2, 3, 4, 5, 6, 7, 8],
        seed_split_b=expected.get("heldout_seeds") or [],
        truthful_measurement_required=True,
        max_rollouts_per_experiment=1,
    )
    source_canonical_train_cell = _plan_source_canonical_train_cell(expected)
    active_state_mode_name = _plan_active_state_mode_name(expected)
    spec = controller._matrix_cell_lane_spec(
        source_canonical_train_cell,
        "G6",
        note="T2a/T2b hard-seed assay on canonical V1cT2S0 lane.",
    )
    contract = controller._materialize_contract(spec.env_contract_config)
    if str(contract.state_mode) != active_state_mode_name:
        contract = replace(contract, state_mode=active_state_mode_name)
    base_interventions = dict(spec.interventions or {})
    return {
        "controller": controller,
        "spec": spec,
        "expected": expected,
        "contract": contract,
        "claim_policy": spec.claim_policy,
        "rotation_source": str(base_interventions.get("rotation_source", "zero")),
        "base_interventions": base_interventions,
    }


def _relative_delta_ok(a: float, b: float, tolerance: float = 0.05) -> bool:
    denom = max(abs(a), abs(b), 1e-6)
    return abs(a - b) / denom <= tolerance


def _mean_metric(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return float(sum(float(item.get(key, 0.0)) for item in records) / len(records))


def _bool_metric(records: list[dict[str, Any]], key: str) -> bool:
    return bool(all(bool(item.get(key, False)) for item in records))


def _t2a_metrics(rollout: dict[str, Any]) -> dict[str, Any]:
    handle_distance = np.asarray(rollout.get("handle_distance_trace", []), dtype=np.float32)
    contact_window = np.asarray(rollout.get("contact_window_fraction_trace", []), dtype=np.float32)
    orientation = np.asarray(rollout.get("orientation_alignment_trace", []), dtype=np.float32)
    attach_eligible = np.asarray(rollout.get("attach_eligible_trace", []), dtype=bool)
    return {
        "min_dist_to_handle": float(np.min(handle_distance)) if handle_distance.size else 1.0,
        "max_contact_window_fraction": float(np.max(contact_window)) if contact_window.size else 0.0,
        "ever_attach_eligible": bool(np.any(attach_eligible)) if attach_eligible.size else False,
        "max_orientation_alignment_cos": float(np.max(orientation)) if orientation.size else 0.0,
        "ever_attached": bool(rollout.get("ever_attached", False)),
    }


def _t2b_metrics(rollout: dict[str, Any]) -> dict[str, Any]:
    stable_attach_steps = int(np.sum(np.asarray(rollout.get("stable_attach_trace", []), dtype=bool)))
    phase_locked_steps = int(np.sum(np.asarray(rollout.get("phase_locked_trace", []), dtype=bool)))
    slip_trace = np.asarray(rollout.get("grasp_slip_norm_trace", []), dtype=np.float32)
    pull_trace = np.asarray(rollout.get("effective_pull_progress_trace", []), dtype=np.float32)
    return {
        "steps": int(rollout.get("steps", 0) or 0),
        "ever_attached": bool(rollout.get("ever_attached", False)),
        "stable_attach_steps": stable_attach_steps,
        "phase_locked_steps": phase_locked_steps,
        "slip_mean": float(np.mean(slip_trace)) if slip_trace.size else 0.0,
        "effective_pull_progress_peak": float(np.max(pull_trace)) if pull_trace.size else 0.0,
        "max_drawer_fraction": float(rollout.get("max_drawer_fraction", 0.0) or 0.0),
    }


def _t2a_reproducibility(records: list[dict[str, Any]]) -> tuple[bool, str]:
    if len(records) != 2:
        return False, "missing_repeat"
    a, b = records
    for key in ["ever_attach_eligible", "ever_attached"]:
        if bool(a[key]) != bool(b[key]):
            return False, f"{key}_mismatch"
    for key in ["min_dist_to_handle", "max_contact_window_fraction", "max_orientation_alignment_cos"]:
        if not _relative_delta_ok(float(a[key]), float(b[key])):
            return False, f"{key}_drift"
    return True, "reproducible"


def _t2b_reproducibility(records: list[dict[str, Any]]) -> tuple[bool, str]:
    if len(records) != 2:
        return False, "missing_repeat"
    a, b = records
    for key in ["ever_attached"]:
        if bool(a[key]) != bool(b[key]):
            return False, f"{key}_mismatch"
    for key in ["stable_attach_steps", "phase_locked_steps", "effective_pull_progress_peak", "max_drawer_fraction"]:
        if not _relative_delta_ok(float(a[key]), float(b[key])):
            return False, f"{key}_drift"
    return True, "reproducible"


def _t2a_pass(record: dict[str, Any]) -> bool:
    return bool(
        record.get("reproducible", False)
        and float(record.get("min_dist_to_handle", 1.0)) <= 0.03
        and float(record.get("max_contact_window_fraction", 0.0)) >= 0.80
        and bool(record.get("ever_attach_eligible", False))
        and float(record.get("max_orientation_alignment_cos", 0.0)) >= 0.95
    )


def _t2b_contact_variant_pass(record: dict[str, Any]) -> bool:
    return bool(
        record.get("reproducible", False)
        and bool(record.get("ever_attached", False))
        and int(record.get("stable_attach_steps", 0)) >= 3
        and int(record.get("phase_locked_steps", 0)) >= 8
    )


def _t2b_attached_variant_pass(record: dict[str, Any], baseline: dict[str, Any]) -> bool:
    baseline_pull = float(baseline.get("effective_pull_progress_peak", 0.0))
    required_pull = baseline_pull * 1.5 if baseline_pull > 0.0 else 0.0
    baseline_phase_locked = int(baseline.get("phase_locked_steps", 0))
    baseline_stable_attach = int(baseline.get("stable_attach_steps", 0))
    horizon_steps = max(int(record.get("steps", 0)), int(baseline.get("steps", 0)), 96)
    saturated_phase_locked = baseline_phase_locked >= max(1, horizon_steps - 4)
    required_phase_locked = baseline_phase_locked - 4 if saturated_phase_locked else baseline_phase_locked + 8
    return bool(
        record.get("reproducible", False)
        and float(record.get("max_drawer_fraction", 0.0)) >= float(baseline.get("max_drawer_fraction", 0.0)) + 0.20
        and float(record.get("effective_pull_progress_peak", 0.0)) >= required_pull
        and int(record.get("phase_locked_steps", 0)) >= required_phase_locked
        and int(record.get("stable_attach_steps", 0)) >= baseline_stable_attach - 4
    )


def _t2b_attached_baseline_pass(record: dict[str, Any]) -> bool:
    return bool(
        record.get("reproducible", False)
        and bool(record.get("ever_attached", False))
        and float(record.get("max_drawer_fraction", 0.0)) >= 0.45
        and int(record.get("stable_attach_steps", 0)) >= 24
        and int(record.get("phase_locked_steps", 0)) >= 24
        and float(record.get("effective_pull_progress_peak", 0.0)) > 0.0
    )


def _aggregate_t2a(seed: int, per_repeat: list[dict[str, Any]]) -> dict[str, Any]:
    reproducible, reproducibility_verdict = _t2a_reproducibility(per_repeat)
    record = {
        "seed": int(seed),
        "ever_attach_eligible": _bool_metric(per_repeat, "ever_attach_eligible"),
        "ever_attached": _bool_metric(per_repeat, "ever_attached"),
        "min_dist_to_handle": round(_mean_metric(per_repeat, "min_dist_to_handle"), 6),
        "max_contact_window_fraction": round(_mean_metric(per_repeat, "max_contact_window_fraction"), 6),
        "max_orientation_alignment_cos": round(_mean_metric(per_repeat, "max_orientation_alignment_cos"), 6),
        "reproducible": reproducible,
        "reproducibility_verdict": reproducibility_verdict,
        "repeat_records": per_repeat,
    }
    record["t2a_pass"] = _t2a_pass(record)
    return record


def _aggregate_t2b(seed: int, warm_start_kind: str, variant: dict[str, Any], per_repeat: list[dict[str, Any]]) -> dict[str, Any]:
    reproducible, reproducibility_verdict = _t2b_reproducibility(per_repeat)
    return {
        "seed": int(seed),
        "warm_start_kind": warm_start_kind,
        "variant_family": variant["variant_family"],
        "variant_value": variant["variant_value"],
        "interventions": dict(variant.get("interventions") or {}),
        "steps": round(_mean_metric(per_repeat, "steps"), 4),
        "ever_attached": _bool_metric(per_repeat, "ever_attached"),
        "stable_attach_steps": round(_mean_metric(per_repeat, "stable_attach_steps"), 4),
        "phase_locked_steps": round(_mean_metric(per_repeat, "phase_locked_steps"), 4),
        "slip_mean": round(_mean_metric(per_repeat, "slip_mean"), 6),
        "effective_pull_progress_peak": round(_mean_metric(per_repeat, "effective_pull_progress_peak"), 6),
        "max_drawer_fraction": round(_mean_metric(per_repeat, "max_drawer_fraction"), 6),
        "reproducible": reproducible,
        "reproducibility_verdict": reproducibility_verdict,
        "repeat_records": per_repeat,
    }


def _select_t3_variant(seed_outcome: dict[str, Any]) -> dict[str, Any] | None:
    attached = dict(seed_outcome.get("attached_phase_locked") or {})
    baseline = dict(attached.get("baseline") or {})
    if bool(attached.get("baseline_pass", False)) and baseline:
        baseline["variant_pass"] = True
        return baseline
    candidates: list[dict[str, Any]] = []
    for family in attached.get("family_outcomes") or []:
        for item in family.get("variants") or []:
            if bool(item.get("variant_pass", False)):
                candidates.append(item)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            float(item.get("max_drawer_fraction", 0.0)),
            float(item.get("effective_pull_progress_peak", 0.0)),
            float(item.get("phase_locked_steps", 0.0)),
            -float(item.get("slip_mean", 0.0)),
        ),
    )


def _run_t3_once(
    *,
    context: dict[str, Any],
    plan: dict[str, Any],
    seed_warm_start_outcomes: dict[str, Any],
    canonical_rotation_source: str,
    canonical_claim_policy: str,
    canonical_base_interventions: dict[str, Any],
) -> dict[str, Any]:
    controller = context["controller"]
    spec = context["spec"]
    expected = dict(context["expected"])
    expected.update(
        {
            "bridge_stage": "t3_deterministic_hard_seed_repair",
            "bridge_attempt": "once_after_t2b_pass",
        }
    )
    selected_variants = {
        str(seed): _select_t3_variant(seed_warm_start_outcomes.get(str(seed), {}))
        for seed in HARD_SEEDS
    }
    raw_records: list[dict[str, Any]] = []
    seed_passes: dict[str, bool] = {}
    strict_seed_passes: dict[str, bool] = {}
    for seed in HARD_SEEDS:
        selected = selected_variants.get(str(seed))
        if not selected:
            seed_passes[str(seed)] = False
            strict_seed_passes[str(seed)] = False
            continue
        selected_interventions = {**canonical_base_interventions, **dict(selected.get("interventions") or {})}
        per_seed_records: list[dict[str, Any]] = []
        for repeat_idx in REPEATS:
            rollout = build_robot_rollout(
                seed=seed,
                grasp_pose_world=np.eye(4, dtype=np.float32),
                episode_index=int(repeat_idx),
                max_steps=96,
                contract=context["contract"],
                rotation_source=canonical_rotation_source,
                claim_policy=canonical_claim_policy,
                interventions=selected_interventions,
            )
            rollout = _augment_rollout_metadata(controller, rollout, spec, expected)
            strict_metrics = dict(rollout.get("strict_metrics") or {})
            phase_locked_trace = np.asarray(rollout.get("phase_locked_trace", []), dtype=np.float32).reshape(-1)
            record = {
                "seed": int(seed),
                "repeat_idx": int(repeat_idx),
                "selected_variant_family": selected.get("variant_family"),
                "selected_variant_value": selected.get("variant_value"),
                "selected_interventions": selected_interventions,
                "teacher_episode_class": str(rollout.get("teacher_episode_class", "rejected_teacher")),
                "strict_success": bool(strict_metrics.get("strict_success", False)),
                "ever_attached": bool(rollout.get("ever_attached", False)),
                "max_drawer_fraction": float(rollout.get("max_drawer_fraction", 0.0) or 0.0),
                "phase_locked_rate": float(np.mean(phase_locked_trace)) if phase_locked_trace.size else 0.0,
            }
            raw_records.append(record)
            per_seed_records.append(record)
        seed_passes[str(seed)] = any(
            rec["teacher_episode_class"] in {"strict_teacher", "near_strict_teacher"} for rec in per_seed_records
        )
        strict_seed_passes[str(seed)] = any(
            rec["teacher_episode_class"] == "strict_teacher" or rec["strict_success"] for rec in per_seed_records
        )
    t3_passed = bool(all(seed_passes.values()) and any(strict_seed_passes.values()))
    return {
        "executed": True,
        "selected_variants": selected_variants,
        "raw_records": raw_records,
        "seed_passes": seed_passes,
        "strict_seed_passes": strict_seed_passes,
        "t3_passed": t3_passed,
    }


def run() -> dict[str, Any]:
    plan = {**_default_plan(), **load_json(TINY_RETRAIN_PLAN_PATH, {})}
    context = _canonical_rollout_context(plan)
    canonical_contract = context["contract"]
    canonical_claim_policy = context["claim_policy"]
    canonical_rotation_source = context["rotation_source"]
    canonical_base_interventions = dict(context["base_interventions"])

    t2a_raw_records: list[dict[str, Any]] = []
    t2a_records: list[dict[str, Any]] = []
    t2a_seed_passes: dict[int, bool] = {}
    for seed in HARD_SEEDS:
        per_repeat: list[dict[str, Any]] = []
        for repeat_idx in REPEATS:
            rollout = build_robot_rollout(
                seed=seed,
                grasp_pose_world=np.eye(4, dtype=np.float32),
                episode_index=int(repeat_idx),
                max_steps=96,
                contract=canonical_contract,
                rotation_source=canonical_rotation_source,
                claim_policy=canonical_claim_policy,
                interventions=canonical_base_interventions,
            )
            metrics = _t2a_metrics(rollout)
            record = {
                "seed": int(seed),
                "repeat_idx": int(repeat_idx),
                **metrics,
            }
            t2a_raw_records.append(record)
            per_repeat.append(record)
        aggregate = _aggregate_t2a(seed, per_repeat)
        t2a_records.append(aggregate)
        t2a_seed_passes[seed] = bool(aggregate["t2a_pass"])

    variant_defs = _variants()
    t2b_raw_records: list[dict[str, Any]] = []
    t2b_records: list[dict[str, Any]] = []
    family_summary: dict[int, dict[str, dict[str, list[dict[str, Any]]]]] = {
        seed: {kind: {} for kind in T2B_WARM_START_KINDS} for seed in HARD_SEEDS
    }
    for seed in HARD_SEEDS:
        for warm_start_kind in T2B_WARM_START_KINDS:
            for variant in variant_defs:
                per_repeat = []
                for repeat_idx in REPEATS:
                    rollout = build_robot_rollout(
                        seed=seed,
                        grasp_pose_world=np.eye(4, dtype=np.float32),
                        episode_index=int(repeat_idx),
                        max_steps=96,
                        contract=canonical_contract,
                        rotation_source=canonical_rotation_source,
                        claim_policy=canonical_claim_policy,
                        interventions={**canonical_base_interventions, **variant["interventions"]},
                        assay_warm_start_kind=warm_start_kind,
                    )
                    metrics = _t2b_metrics(rollout)
                    record = {
                        "seed": int(seed),
                        "warm_start_kind": warm_start_kind,
                        "repeat_idx": int(repeat_idx),
                        "variant_family": variant["variant_family"],
                        "variant_value": variant["variant_value"],
                        **metrics,
                    }
                    t2b_raw_records.append(record)
                    per_repeat.append(record)
                aggregate = _aggregate_t2b(seed, warm_start_kind, variant, per_repeat)
                t2b_records.append(aggregate)
                family_summary[seed][warm_start_kind].setdefault(str(variant["variant_family"]), []).append(aggregate)

    t2b_seed_passes: dict[int, bool] = {}
    seed_warm_start_outcomes: dict[str, Any] = {}
    for seed in HARD_SEEDS:
        seed_outcome = {}
        attached_has_family = False
        for warm_start_kind in T2B_WARM_START_KINDS:
            baseline = next(
                item
                for item in t2b_records
                if item["seed"] == seed and item["warm_start_kind"] == warm_start_kind and item["variant_family"] == "baseline"
            )
            baseline_pass = _t2b_attached_baseline_pass(baseline) if warm_start_kind == "attached_phase_locked" else False
            warm_start_outcomes = []
            for family, items in family_summary[seed][warm_start_kind].items():
                if family == "baseline":
                    continue
                family_pass = False
                for item in items:
                    if warm_start_kind == "contact_aligned":
                        item["variant_pass"] = _t2b_contact_variant_pass(item)
                    else:
                        item["variant_pass"] = _t2b_attached_variant_pass(item, baseline)
                    family_pass = family_pass or bool(item["variant_pass"])
                warm_start_outcomes.append(
                    {
                        "variant_family": family,
                        "family_pass": family_pass,
                        "variants": items,
                    }
                )
            seed_outcome[warm_start_kind] = {
                "baseline": baseline,
                "baseline_pass": baseline_pass,
                "family_outcomes": warm_start_outcomes,
            }
            if warm_start_kind == "attached_phase_locked":
                attached_has_family = bool(baseline_pass or any(bool(item["family_pass"]) for item in warm_start_outcomes))
        seed_warm_start_outcomes[str(seed)] = seed_outcome
        t2b_seed_passes[seed] = bool(attached_has_family)

    t2a_passed = bool(all(t2a_seed_passes.values()))
    t2b_passed = bool(all(t2b_seed_passes.values()))

    bounded_terminal = None
    t3_legal = False
    t3_target = None
    t3_result = {
        "executed": False,
        "selected_variants": {},
        "raw_records": [],
        "seed_passes": {},
        "strict_seed_passes": {},
        "t3_passed": False,
    }
    if (not t2a_passed) and (not t2b_passed):
        bounded_terminal = "ATTACH_AND_RECOVERY_NOT_ESTABLISHED_IN_CURRENT_SCRIPTED_POLICY"
    elif t2a_passed and (not t2b_passed):
        bounded_terminal = "TEACHER_RECOVERY_NOT_EXECUTABLE_IN_CURRENT_ACTION_SPACE"
    elif (not t2a_passed) and t2b_passed:
        t3_legal = True
        t3_target = "deterministic_attach_entry_repair_from_reset"
    elif t2a_passed and t2b_passed:
        t3_legal = True
        t3_target = "deterministic_hard_seed_recovery_integration"

    if t3_legal:
        t3_result = _run_t3_once(
            context=context,
            plan=plan,
            seed_warm_start_outcomes=seed_warm_start_outcomes,
            canonical_rotation_source=canonical_rotation_source,
            canonical_claim_policy=canonical_claim_policy,
            canonical_base_interventions=canonical_base_interventions,
        )
        if not bool(t3_result.get("t3_passed", False)):
            bounded_terminal = "HARD_SEED_DETERMINISTIC_RECOVERY_NOT_ESTABLISHED"

    result = {
        "gate": "g6_teacher_recovery_probe",
        "run_instance_id": plan.get("run_instance_id"),
        "plan_version": plan.get("plan_version"),
        "source_base_commit": plan.get("source_base_commit"),
        "source_canonical_train_cell": plan.get("source_canonical_train_cell"),
        "source_best_train_state_mode": plan.get("source_best_train_state_mode"),
        "active_train_state_mode": plan.get("active_train_state_mode"),
        "bridge_stage": "t2a_t2b_teacher_recovery_assay",
        "hard_seeds": HARD_SEEDS,
        "rotation_source": canonical_rotation_source,
        "claim_policy": canonical_claim_policy,
        "canonical_base_interventions": canonical_base_interventions,
        "t2a": {
            "repeats_per_seed": len(REPEATS),
            "raw_records": t2a_raw_records,
            "records": t2a_records,
            "t2a_seed_passes": {str(k): bool(v) for k, v in t2a_seed_passes.items()},
            "t2a_passed": t2a_passed,
        },
        "t2b": {
            "variant_count": len(variant_defs) - 1,
            "baseline_included": True,
            "warm_start_kinds": T2B_WARM_START_KINDS,
            "repeats_per_config": len(REPEATS),
            "raw_records": t2b_raw_records,
            "records": t2b_records,
            "seed_warm_start_outcomes": seed_warm_start_outcomes,
            "t2b_seed_passes": {str(k): bool(v) for k, v in t2b_seed_passes.items()},
            "t2b_passed": t2b_passed,
        },
        "t3_legal": t3_legal,
        "t3_target": t3_target,
        "t3": t3_result,
        "t3_passed": bool(t3_result.get("t3_passed", False)),
        "bounded_terminal": bounded_terminal,
        "timestamp": time.time(),
    }
    write_json_atomic(ARTIFACT, result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()

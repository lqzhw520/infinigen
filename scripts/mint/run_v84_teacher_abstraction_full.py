#!/usr/bin/env python3
"""v8.4 always-continue evidence program."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable

import numpy as np

from drawer_robot_env_mujoco import build_robot_rollout, save_robot_rollout
from mint_common import ARTIFACT_DIR, PROJECT_ROOT, TINY_RETRAIN_PLAN_PATH, load_json, write_json_atomic
from run_g6_teacher_recovery_probe import (
    ARTIFACT as G6_ARTIFACT,
    HARD_SEEDS,
    REPEATS,
    _canonical_rollout_context,
    _default_plan,
    run as run_g6_probe,
)
from run_tiny_retrain_confirmation import (
    AMPLIFIED_EPISODES_PER_SEED,
    AMPLIFIED_MIN_TRAIN_EPISODES,
    AMPLIFIED_PROBE_STEPS,
    AMPLIFIED_SAVE_FREQ,
    AMPLIFIED_TRAIN_STEPS,
    DEFAULT_ROLLOUT_SOURCE_DIR,
    G6_STRICT_SEMANTICS_ALIGNMENT_PATH,
    G6_TEACHER_READINESS_CONTRACT_PATH,
    G8_PROBE_PATH,
    G8_SUMMARY_PATH,
    G9_SUMMARY_PATH,
    HONEST_EPISODES_PER_SEED,
    HONEST_MIN_TRAIN_EPISODES,
    HONEST_PROBE_STEPS,
    HONEST_SAVE_FREQ,
    HONEST_TRAIN_STEPS,
    TINY_RETRAIN_DATASET_BUILD_PATH,
    TINY_RETRAIN_SUMMARY_PATH,
    _run_stage,
    _write_loop_summary,
    _write_strict_semantics_alignment,
    load_terminal_artifacts,
)
from tiny_retrain_mainline import _augment_rollout_metadata

GOOD_SEEDS = [1, 5, 6, 7, 8]
BRIDGE_SEED = 3
AUTH_BRANCH = "feature/mint-env-reformulation-v1-visual-fidelity"
AUTH_REPO_ROOT = "/mnt/afs2/zhuhaowu/infinigen"
PHASE_ORDER = ["t0", "t1", "t2a", "t2b", "p1a", "p1b", "t3a", "t3b", "t4", "t5", "t6", "b0", "b1", "b2", "b3", "b4"]
PHASE_ARTIFACTS = {
    "t0": ARTIFACT_DIR / "v84_t0_integrity.json",
    "t1": ARTIFACT_DIR / "v84_t1_strict_semantics.json",
    "t2a": ARTIFACT_DIR / "v84_t2a_reset_attach_entry.json",
    "t2b": ARTIFACT_DIR / "v84_t2b_warmstart_recovery.json",
    "p1a": ARTIFACT_DIR / "v84_phase1a_warmstart_baseline_plateau.json",
    "p1b": ARTIFACT_DIR / "v84_phase1b_fixed_grasp_embodiment_bound.json",
    "t3a": ARTIFACT_DIR / "v84_t3a_matched_world_frame.json",
    "t3b": ARTIFACT_DIR / "v84_t3b_interaction_frame.json",
    "t4": ARTIFACT_DIR / "v84_t4_truth_generalization.json",
    "t5": ARTIFACT_DIR / "v84_t5_diversity_dedupe.json",
    "t6": ARTIFACT_DIR / "v84_t6_readiness_contract.json",
    "b0": ARTIFACT_DIR / "v84_b0_diagnostic_microtrain.json",
    "b1": ARTIFACT_DIR / "v84_b1_s0_honest.json",
    "b2": ARTIFACT_DIR / "v84_b2_s0_amplification.json",
    "b3": ARTIFACT_DIR / "v84_b3_s2_honest.json",
    "b4": ARTIFACT_DIR / "v84_b4_s2_amplification.json",
}
EVIDENCE_MATRIX_PATH = ARTIFACT_DIR / "v84_evidence_matrix.json"
ROLLOUT_ROOT = ARTIFACT_DIR / "v84_rollouts"
TEMP_T0_VALIDATE_PATH = ARTIFACT_DIR / "v84_t0_generate_reload_validate.json"
STATE_MODE_NAME = {"S0": "m0_proxy", "S2": "telemetry_candidate_v4_task_identity"}


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, _json_ready(payload))


def _git(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=PROJECT_ROOT, text=True).strip()


def _phase_artifact_path(phase: str) -> Path:
    return PHASE_ARTIFACTS[phase]


def _new_run_id(head: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"v84_{stamp}_{head[:8]}_{uuid.uuid4().hex[:8]}"


def _authoritative_use(ctx: dict[str, Any], force_diagnostic: bool = False) -> str:
    return "diagnostic_only" if (force_diagnostic or not bool(ctx["authoritative_mode"])) else "authoritative"


def _capture_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


def _phase_result(
    ctx: dict[str, Any],
    *,
    phase: str,
    status: str,
    clauses: dict[str, bool],
    rca_classes: list[str],
    evidence: dict[str, Any],
    continuation_input_source: str,
    input_artifacts: list[str],
    output_artifacts: list[str],
    bounded_terminal_if_finalized: str | None = None,
    force_diagnostic: bool = False,
    elapsed: float = 0.0,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "status": status,
        "clauses": clauses,
        "failed_clauses": [key for key, passed in clauses.items() if not bool(passed)],
        "rca_classes": sorted(set(rca_classes)),
        "evidence": _json_ready(evidence),
        "authoritative_use": _authoritative_use(ctx, force_diagnostic=force_diagnostic),
        "continuation_input_source": continuation_input_source,
        "bounded_terminal_if_finalized": bounded_terminal_if_finalized,
        "phase_elapsed_sec": round(float(elapsed), 4),
        "input_artifacts": input_artifacts,
        "output_artifacts": output_artifacts,
    }


def _write_phase(ctx: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    path = _phase_artifact_path(result["phase"])
    _write_json(path, result)
    ctx["phase_results"][result["phase"]] = result
    _write_matrix(ctx)
    return result


def _write_matrix(ctx: dict[str, Any]) -> None:
    t6 = ctx["phase_results"].get("t6", {})
    teacher_abstraction = bool(
        ctx["phase_results"].get("t1", {}).get("status") == "PASS"
        and ctx["phase_results"].get("t2a", {}).get("status") == "PASS"
        and ctx["phase_results"].get("t2b", {}).get("status") == "PASS"
        and bool(ctx["phase_results"].get("p1b", {}).get("evidence", {}).get("candidate_frontier_valid", False))
        and bool(ctx["phase_results"].get("t3b", {}).get("clauses", {}).get("matched_superiority_over_t3a", False))
        and ctx["phase_results"].get("t4", {}).get("status") == "PASS"
    )
    teacher_readiness = bool(t6.get("status") == "PASS")
    final_failed: list[str] = []
    final_rca: list[str] = []
    for phase in PHASE_ORDER:
        item = ctx["phase_results"].get(phase)
        if not item:
            continue
        final_failed.extend(item.get("failed_clauses", []))
        final_rca.extend(item.get("rca_classes", []))
    matrix = {
        "run_id": ctx["run_id"],
        "timestamp": ctx["timestamp"],
        "branch": ctx["branch"],
        "head_commit": ctx["head"],
        "env_python": sys.executable,
        "authoritative_mode": bool(ctx["authoritative_mode"]),
        "phase_order": PHASE_ORDER,
        "phase_results": ctx["phase_results"],
        "teacher_abstraction_sufficient_condition": teacher_abstraction,
        "teacher_readiness_sufficient_condition": teacher_readiness,
        "authoritative_retrain_executed": bool(ctx.get("authoritative_retrain_executed", False)),
        "final_verdict": ctx.get("final_verdict"),
        "final_failed_clauses": sorted(set(final_failed)),
        "final_rca_classes": sorted(set(final_rca)),
    }
    _write_json(EVIDENCE_MATRIX_PATH, matrix)


def _base_rollout_plan(ctx: dict[str, Any], *, bridge_stage: str, bridge_attempt: str, active_train_state_mode: str = "S0") -> dict[str, Any]:
    plan = {**_default_plan(), **load_json(TINY_RETRAIN_PLAN_PATH, {})}
    plan.update(
        {
            "run_instance_id": ctx["run_id"],
            "plan_version": _default_plan()["plan_version"],
            "source_base_commit": _default_plan()["source_base_commit"],
            "source_canonical_train_cell": "V1cT2S0",
            "source_best_train_state_mode": "S0",
            "active_train_state_mode": active_train_state_mode,
            "active_state_mode_name": STATE_MODE_NAME.get(active_train_state_mode, active_train_state_mode),
            "bridge_stage": bridge_stage,
            "bridge_attempt": bridge_attempt,
            "working_head_commit": ctx["head"],
            "train_seeds": [1, 2, 3, 4, 5, 6, 7, 8],
            "heldout_seeds": [11, 12, 13, 14, 15],
        }
    )
    return plan


def _tier_rank(record: dict[str, Any]) -> int:
    cls = str(record.get("teacher_episode_class", "rejected_teacher"))
    if cls == "strict_teacher" or bool(record.get("strict_success", False)):
        return 2
    if cls == "near_strict_teacher":
        return 1
    return 0


def _best_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return max(
        records,
        key=lambda rec: (
            _tier_rank(rec),
            float(rec.get("max_drawer_fraction", 0.0)),
            float(rec.get("post_attach_drawer_delta", 0.0)),
            -int(rec.get("truthful_window_longest_interior_gap", 0) or 0),
        ),
    )


def _controller_record(rollout: dict[str, Any], *, seed: int, repeat_idx: int, controller_mode: str, max_steps: int, warm_start_kind: str | None, rollout_path: Path) -> dict[str, Any]:
    strict = dict(rollout.get("strict_metrics") or {})
    drawer = np.asarray(rollout.get("next_drawer_fractions", []), dtype=np.float32).reshape(-1)
    phase_locked = np.asarray(rollout.get("phase_locked_trace", []), dtype=np.float32).reshape(-1)
    eff = np.asarray(rollout.get("drawer_delta_effective_trace", []), dtype=np.float32).reshape(-1)
    eff_pull = np.asarray(rollout.get("effective_pull_progress_trace", []), dtype=np.float32).reshape(-1)
    return {
        "seed": int(seed),
        "repeat_idx": int(repeat_idx),
        "controller_mode": controller_mode,
        "warm_start_kind": warm_start_kind,
        "max_steps": int(max_steps),
        "rollout_path": str(rollout_path),
        "teacher_episode_class": str(rollout.get("teacher_episode_class", "rejected_teacher")),
        "teacher_fingerprint": str(rollout.get("teacher_fingerprint", "")),
        "strict_success": bool(strict.get("strict_success", False)),
        "measurement_truthful_for_training": bool(rollout.get("measurement_truthful_for_training", False)),
        "max_drawer_fraction": float(rollout.get("max_drawer_fraction", 0.0) or 0.0),
        "post_attach_drawer_delta": float(strict.get("post_attach_drawer_delta", 0.0) or 0.0),
        "attach_persistence": int(strict.get("attach_persistence", 0) or 0),
        "truthful_window_longest_interior_gap": int(rollout.get("truthful_window_longest_interior_gap", 0) or 0),
        "step_96_max_drawer_fraction": float(np.max(drawer[:96])) if drawer.size else 0.0,
        "late_phase_drawer_delta_effective_sum": float(np.sum(np.clip(eff[96:], 0.0, None))) if eff.size > 96 else 0.0,
        "phase_locked_rate": float(np.mean(phase_locked)) if phase_locked.size else 0.0,
        "effective_pull_progress_peak": float(np.max(eff_pull)) if eff_pull.size else 0.0,
    }


def _run_controller_rollouts(
    ctx: dict[str, Any],
    *,
    phase: str,
    controller_mode: str,
    seeds: list[int],
    max_steps: int,
    active_train_state_mode: str = "S0",
    warm_start_kind: str | None = None,
) -> tuple[list[dict[str, Any]], list[str], str]:
    expected = _base_rollout_plan(ctx, bridge_stage=f"v84_{phase}", bridge_attempt="always_continue", active_train_state_mode=active_train_state_mode)
    context = _canonical_rollout_context(expected)
    save_dir = ROLLOUT_ROOT / phase / active_train_state_mode.lower() / controller_mode
    save_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    saved_paths: list[str] = []
    for seed in seeds:
        for repeat_idx in REPEATS:
            rollout = build_robot_rollout(
                seed=int(seed),
                grasp_pose_world=np.eye(4, dtype=np.float32),
                episode_index=int(repeat_idx),
                max_steps=int(max_steps),
                contract=context["contract"],
                rotation_source=context["rotation_source"],
                claim_policy=context["claim_policy"],
                interventions=dict(context["base_interventions"]),
                assay_warm_start_kind=warm_start_kind,
                teacher_controller_mode=controller_mode,
                pull_open_fraction=0.92,
            )
            rollout = _augment_rollout_metadata(context["controller"], rollout, context["spec"], expected)
            out_path = save_dir / f"seed_{int(seed):03d}_episode_{int(repeat_idx):02d}.npz"
            save_robot_rollout(out_path, rollout)
            saved_paths.append(str(out_path))
            records.append(
                _controller_record(
                    rollout,
                    seed=int(seed),
                    repeat_idx=int(repeat_idx),
                    controller_mode=controller_mode,
                    max_steps=max_steps,
                    warm_start_kind=warm_start_kind,
                    rollout_path=out_path,
                )
            )
    return records, saved_paths, str(save_dir)


def _per_seed_summary(records: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for seed in seeds:
        seed_records = [item for item in records if int(item["seed"]) == int(seed)]
        best = _best_record(seed_records)
        out[str(seed)] = {
            "records": seed_records,
            "best_record": best,
            "best_teacher_tier": _tier_rank(best or {}),
            "has_accepted_teacher": any(_tier_rank(item) >= 1 for item in seed_records),
            "has_strict_teacher": any(_tier_rank(item) == 2 for item in seed_records),
            "best_max_drawer_fraction": float(best.get("max_drawer_fraction", 0.0) or 0.0) if best else 0.0,
            "best_post_attach_drawer_delta": float(best.get("post_attach_drawer_delta", 0.0) or 0.0) if best else 0.0,
            "best_truth_gap": int(best.get("truthful_window_longest_interior_gap", 0) or 0) if best else None,
            "best_rollout_path": best.get("rollout_path") if best else None,
        }
    return out


def _accepted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in records if _tier_rank(item) >= 1]


def _rca_from_mapping(clauses: dict[str, bool], mapping: dict[str, str]) -> list[str]:
    return [mapping[key] for key, passed in clauses.items() if (not passed and key in mapping)]


def _run_phase_t0(ctx: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    payload = {"run_id": ctx["run_id"], "timestamp": now_iso(), "head": ctx["head"]}
    _write_json(TEMP_T0_VALIDATE_PATH, payload)
    reloaded = load_json(TEMP_T0_VALIDATE_PATH, {})
    clauses = {
        "authority_host_ok": PROJECT_ROOT.exists(),
        "repo_root_ok": str(PROJECT_ROOT) == AUTH_REPO_ROOT,
        "branch_ok": str(ctx["branch"]) == AUTH_BRANCH,
        "head_commit_recorded": bool(ctx["head"]),
        "conda_env_ok": "infinigen" in sys.executable,
        "artifact_dir_writable": ARTIFACT_DIR.exists() and os.access(ARTIFACT_DIR, os.W_OK),
        "generate_persist_reload_validate_ok": reloaded == payload,
    }
    status = "PASS" if all(clauses.values()) else "ERROR"
    return _phase_result(
        ctx,
        phase="t0",
        status=status,
        clauses=clauses,
        rca_classes=_rca_from_mapping(
            clauses,
            {
                "authority_host_ok": "authority_mismatch",
                "repo_root_ok": "authority_mismatch",
                "branch_ok": "authority_mismatch",
                "head_commit_recorded": "authority_mismatch",
                "conda_env_ok": "env_activation_failed",
                "artifact_dir_writable": "artifact_chain_broken",
                "generate_persist_reload_validate_ok": "artifact_chain_broken",
            },
        ),
        evidence={
            "expected_repo_root": AUTH_REPO_ROOT,
            "expected_branch": AUTH_BRANCH,
            "branch": ctx["branch"],
            "head_commit": ctx["head"],
            "env_python": sys.executable,
            "temp_validation_path": str(TEMP_T0_VALIDATE_PATH),
            "validated_payload": reloaded,
        },
        continuation_input_source="remote_live_authority",
        input_artifacts=[],
        output_artifacts=[str(_phase_artifact_path("t0")), str(TEMP_T0_VALIDATE_PATH)],
        bounded_terminal_if_finalized="T0_INTEGRITY_FATAL" if status == "ERROR" else None,
        elapsed=time.time() - t0,
    )


def _run_phase_t1(ctx: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    plan = _base_rollout_plan(ctx, bridge_stage="v84_t1_semantics", bridge_attempt="alignment")
    paths = sorted(DEFAULT_ROLLOUT_SOURCE_DIR.glob("*.npz"))
    if paths:
        report = _write_strict_semantics_alignment(paths, plan)
    else:
        report = {"passed": False, "entries": [], "error": "no_rollouts_for_alignment"}
        _write_json(G6_STRICT_SEMANTICS_ALIGNMENT_PATH, report)
    clauses = {"strict_semantics_aligned": bool(report.get("passed", False))}
    ctx["semantics_blocked"] = not clauses["strict_semantics_aligned"]
    return _phase_result(
        ctx,
        phase="t1",
        status="PASS" if clauses["strict_semantics_aligned"] else "FAIL",
        clauses=clauses,
        rca_classes=(
            ["strict_metric_mismatch"]
            if report.get("entries")
            else ["meta_npz_divergence"]
        ) if not clauses["strict_semantics_aligned"] else [],
        evidence={
            "paired_rollout_count": int(report.get("paired_rollout_count", len(paths))),
            "aligned_rollout_count": int(report.get("aligned_rollout_count", 0)),
            "entries": report.get("entries", []),
            "semantics_blocked": ctx["semantics_blocked"],
        },
        continuation_input_source=str(DEFAULT_ROLLOUT_SOURCE_DIR),
        input_artifacts=[str(path) for path in paths[:16]],
        output_artifacts=[str(_phase_artifact_path("t1")), str(G6_STRICT_SEMANTICS_ALIGNMENT_PATH)],
        bounded_terminal_if_finalized="STRICT_SEMANTICS_ALIGNMENT_NOT_ESTABLISHED" if not clauses["strict_semantics_aligned"] else None,
        elapsed=time.time() - t0,
    )


def _ensure_g6(ctx: dict[str, Any]) -> dict[str, Any]:
    if "g6_result" not in ctx:
        result, captured = _capture_call(run_g6_probe)
        ctx["g6_result"] = result
        ctx["g6_log"] = captured
    return ctx["g6_result"]


def _run_phase_t2a(ctx: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    g6 = _ensure_g6(ctx)
    t2a = dict(g6.get("t2a") or {})
    seed_passes = {str(k): bool(v) for k, v in (t2a.get("t2a_seed_passes") or {}).items()}
    clauses = {
        "seed2_attach_entry_pass": bool(seed_passes.get("2", False)),
        "seed4_attach_entry_pass": bool(seed_passes.get("4", False)),
    }
    rca = _rca_from_mapping(
        clauses,
        {
            "seed2_attach_entry_pass": "attach_entry_not_established",
            "seed4_attach_entry_pass": "attach_entry_not_established",
        },
    )
    return _phase_result(
        ctx,
        phase="t2a",
        status="PASS" if all(clauses.values()) else "FAIL",
        clauses=clauses,
        rca_classes=rca or (["seed_specific_geometry_sensitivity"] if not all(clauses.values()) else []),
        evidence={
            "t2a_passed": bool(t2a.get("t2a_passed", False)),
            "records": t2a.get("records", []),
            "semantics_blocked": ctx.get("semantics_blocked", False),
        },
        continuation_input_source=str(G6_ARTIFACT),
        input_artifacts=[str(G6_ARTIFACT)],
        output_artifacts=[str(_phase_artifact_path("t2a")), str(G6_ARTIFACT)],
        bounded_terminal_if_finalized="ATTACH_ENTRY_NOT_ESTABLISHED" if not all(clauses.values()) else None,
        elapsed=time.time() - t0,
    )


def _run_phase_t2b(ctx: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    g6 = _ensure_g6(ctx)
    t2b = dict(g6.get("t2b") or {})
    seed_passes = {str(k): bool(v) for k, v in (t2b.get("t2b_seed_passes") or {}).items()}
    clauses = {
        "seed2_recovery_family_pass": bool(seed_passes.get("2", False)),
        "seed4_recovery_family_pass": bool(seed_passes.get("4", False)),
    }
    return _phase_result(
        ctx,
        phase="t2b",
        status="PASS" if all(clauses.values()) else "FAIL",
        clauses=clauses,
        rca_classes=_rca_from_mapping(
            clauses,
            {
                "seed2_recovery_family_pass": "recovery_expressivity_not_established",
                "seed4_recovery_family_pass": "recovery_expressivity_not_established",
            },
        ) or (["attached_phase_controller_ceiling"] if not all(clauses.values()) else []),
        evidence={
            "t2b_passed": bool(t2b.get("t2b_passed", False)),
            "records": t2b.get("records", []),
            "seed_warm_start_outcomes": t2b.get("seed_warm_start_outcomes", {}),
            "semantics_blocked": ctx.get("semantics_blocked", False),
        },
        continuation_input_source=str(G6_ARTIFACT),
        input_artifacts=[str(G6_ARTIFACT)],
        output_artifacts=[str(_phase_artifact_path("t2b")), str(G6_ARTIFACT)],
        bounded_terminal_if_finalized="RECOVERY_EXECUTABILITY_NOT_ESTABLISHED" if not all(clauses.values()) else None,
        elapsed=time.time() - t0,
    )


def _run_phase_p1a(ctx: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    records, saved_paths, _ = _run_controller_rollouts(
        ctx,
        phase="p1a",
        controller_mode="baseline",
        seeds=list(HARD_SEEDS),
        max_steps=192,
        warm_start_kind="attached_phase_locked",
    )
    seed_results = _per_seed_summary(records, list(HARD_SEEDS))
    evidence = {
        "seed_results": seed_results,
        "baseline_plateau_measured": True,
        "saved_rollout_paths": saved_paths,
        "accepted_records": _accepted_records(records),
    }
    return _phase_result(
        ctx,
        phase="p1a",
        status="PASS",
        clauses={"baseline_plateau_measured": True},
        rca_classes=["baseline_plateau_reached"],
        evidence=evidence,
        continuation_input_source="warmstart_baseline_plateau",
        input_artifacts=[],
        output_artifacts=[str(_phase_artifact_path("p1a")), *saved_paths],
        force_diagnostic=True,
        elapsed=time.time() - t0,
    )


def _run_phase_p1b(ctx: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    p1a = ctx["phase_results"].get("p1a") or load_json(_phase_artifact_path("p1a"), {})
    baseline_seed_results = (p1a.get("evidence") or {}).get("seed_results", {})
    baseline_valid = all(str(seed) in baseline_seed_results for seed in HARD_SEEDS)
    if not baseline_valid:
        p1a = _run_phase_p1a(ctx)
        _write_phase(ctx, p1a)
        baseline_seed_results = (p1a.get("evidence") or {}).get("seed_results", {})
    baseline_best = {seed: float((baseline_seed_results.get(str(seed), {}) or {}).get("best_max_drawer_fraction", 0.0) or 0.0) for seed in HARD_SEEDS}
    records, saved_paths, _ = _run_controller_rollouts(
        ctx,
        phase="p1b",
        controller_mode="embodiment_bound_quasistatic",
        seeds=list(HARD_SEEDS),
        max_steps=256,
        warm_start_kind="attached_phase_locked",
    )
    seed_results = _per_seed_summary(records, list(HARD_SEEDS))
    improvements = {str(seed): float(seed_results[str(seed)]["best_max_drawer_fraction"]) - baseline_best[int(seed)] for seed in map(str, HARD_SEEDS)}
    clauses = {
        "candidate_not_weaker_than_p1a": all(delta >= -1e-6 for delta in improvements.values()),
        "candidate_strictly_better_than_p1a": any(delta > 1e-4 for delta in improvements.values()),
        "both_seeds_ge_085": all(bool(seed_results[str(seed)]["best_max_drawer_fraction"] >= 0.85) for seed in HARD_SEEDS),
        "at_least_one_seed_ge_090": any(bool(seed_results[str(seed)]["best_max_drawer_fraction"] >= 0.90) for seed in HARD_SEEDS),
    }
    candidate_frontier_valid = bool(clauses["candidate_not_weaker_than_p1a"] and clauses["candidate_strictly_better_than_p1a"])
    if not candidate_frontier_valid:
        status = "DIAGNOSTIC"
        bounded = "PHASE1B_CANDIDATE_NOT_STRONGER_THAN_BASELINE"
    else:
        status = "PASS" if (clauses["both_seeds_ge_085"] and clauses["at_least_one_seed_ge_090"]) else "FAIL"
        bounded = None if status == "PASS" else "FIXED_GRASP_ACCEPTANCE_BAR_NOT_REACHED_IN_QUASISTATIC_ASSAY"
    evidence = {
        "seed_results": seed_results,
        "baseline_reference": baseline_best,
        "improvements_over_p1a": improvements,
        "candidate_frontier_valid": candidate_frontier_valid,
        "saved_rollout_paths": saved_paths,
        "accepted_records": _accepted_records(records),
    }
    return _phase_result(
        ctx,
        phase="p1b",
        status=status,
        clauses=clauses,
        rca_classes=_rca_from_mapping(
            clauses,
            {
                "candidate_not_weaker_than_p1a": "candidate_not_stronger_than_baseline",
                "candidate_strictly_better_than_p1a": "candidate_not_stronger_than_baseline",
                "both_seeds_ge_085": "candidate_stronger_but_acceptance_not_reached",
                "at_least_one_seed_ge_090": "candidate_stronger_but_acceptance_not_reached",
            },
        ),
        evidence=evidence,
        continuation_input_source=str(_phase_artifact_path("p1a")),
        input_artifacts=[str(_phase_artifact_path("p1a"))],
        output_artifacts=[str(_phase_artifact_path("p1b")), *saved_paths],
        bounded_terminal_if_finalized=bounded,
        force_diagnostic=(status == "DIAGNOSTIC"),
        elapsed=time.time() - t0,
    )


def _run_phase_t3a(ctx: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    records, saved_paths, _ = _run_controller_rollouts(
        ctx,
        phase="t3a",
        controller_mode="world_frame_matched",
        seeds=list(HARD_SEEDS),
        max_steps=144,
    )
    seed_results = _per_seed_summary(records, list(HARD_SEEDS))
    clauses = {
        "seed2_near_strict": bool(seed_results["2"]["has_accepted_teacher"]),
        "seed4_near_strict": bool(seed_results["4"]["has_accepted_teacher"]),
        "at_least_one_seed_strict": bool(seed_results["2"]["has_strict_teacher"] or seed_results["4"]["has_strict_teacher"]),
    }
    evidence = {
        "seed_results": seed_results,
        "accepted_records": _accepted_records(records),
        "saved_rollout_paths": saved_paths,
        "semantics_blocked": ctx.get("semantics_blocked", False),
    }
    return _phase_result(
        ctx,
        phase="t3a",
        status="PASS" if all(clauses.values()) else "FAIL",
        clauses=clauses,
        rca_classes=_rca_from_mapping(
            clauses,
            {
                "seed2_near_strict": "world_frame_opening_ceiling",
                "seed4_near_strict": "seed4_truth_gap_persists",
                "at_least_one_seed_strict": "strict_drawer_open_threshold_not_reached",
            },
        ),
        evidence=evidence,
        continuation_input_source="reset_matched_world_frame",
        input_artifacts=[],
        output_artifacts=[str(_phase_artifact_path("t3a")), *saved_paths],
        bounded_terminal_if_finalized="WORLD_FRAME_MATCHED_HARD_SEED_REPAIR_NOT_ESTABLISHED" if not all(clauses.values()) else None,
        elapsed=time.time() - t0,
    )


def _seed_superiority(a: dict[str, Any], b: dict[str, Any]) -> int:
    key_a = (int(a.get("best_teacher_tier", 0)), float(a.get("best_max_drawer_fraction", 0.0)), -int(a.get("best_truth_gap", 9999) or 9999))
    key_b = (int(b.get("best_teacher_tier", 0)), float(b.get("best_max_drawer_fraction", 0.0)), -int(b.get("best_truth_gap", 9999) or 9999))
    return (key_a > key_b) - (key_a < key_b)


def _run_phase_t3b(ctx: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    records, saved_paths, _ = _run_controller_rollouts(
        ctx,
        phase="t3b",
        controller_mode="interaction_frame_hybrid",
        seeds=list(HARD_SEEDS),
        max_steps=144,
    )
    seed_results = _per_seed_summary(records, list(HARD_SEEDS))
    t3a_seed_results = (ctx["phase_results"].get("t3a", {}).get("evidence") or {}).get("seed_results", {})
    superiority = {}
    for seed in map(str, HARD_SEEDS):
        superiority[seed] = _seed_superiority(seed_results.get(seed, {}), t3a_seed_results.get(seed, {}))
    clauses = {
        "seed2_near_strict": bool(seed_results["2"]["has_accepted_teacher"]),
        "seed4_near_strict": bool(seed_results["4"]["has_accepted_teacher"]),
        "at_least_one_seed_strict": bool(seed_results["2"]["has_strict_teacher"] or seed_results["4"]["has_strict_teacher"]),
        "matched_superiority_over_t3a": bool(any(v > 0 for v in superiority.values()) and all(v >= 0 for v in superiority.values())),
    }
    accepted = _accepted_records(records)
    for item in accepted:
        ctx["accepted_rollouts"][str(item["rollout_path"])] = item
    evidence = {
        "seed_results": seed_results,
        "superiority_vs_t3a": superiority,
        "accepted_records": accepted,
        "saved_rollout_paths": saved_paths,
        "semantics_blocked": ctx.get("semantics_blocked", False),
    }
    return _phase_result(
        ctx,
        phase="t3b",
        status="PASS" if all(clauses.values()) else "FAIL",
        clauses=clauses,
        rca_classes=_rca_from_mapping(
            clauses,
            {
                "seed2_near_strict": "interaction_frame_improves_truth_but_not_opening",
                "seed4_near_strict": "interaction_frame_improves_opening_but_not_truth",
                "at_least_one_seed_strict": "strict_drawer_open_threshold_not_reached",
                "matched_superiority_over_t3a": "interaction_frame_not_superior",
            },
        ),
        evidence=evidence,
        continuation_input_source=str(_phase_artifact_path("t3a")),
        input_artifacts=[str(_phase_artifact_path("t3a"))],
        output_artifacts=[str(_phase_artifact_path("t3b")), *saved_paths],
        bounded_terminal_if_finalized="TEACHER_ABSTRACTION_NOT_ESTABLISHED" if not all(clauses.values()) else None,
        elapsed=time.time() - t0,
    )


def _run_phase_t4(ctx: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    seeds = [BRIDGE_SEED, *GOOD_SEEDS]
    records, saved_paths, _ = _run_controller_rollouts(
        ctx,
        phase="t4",
        controller_mode="interaction_frame_hybrid",
        seeds=seeds,
        max_steps=144,
    )
    seed_results = _per_seed_summary(records, seeds)
    accepted = _accepted_records(records)
    for item in accepted:
        ctx["accepted_rollouts"][str(item["rollout_path"])] = item
    t3b_seed_results = (ctx["phase_results"].get("t3b", {}).get("evidence") or {}).get("seed_results", {})
    seed2_best = t3b_seed_results.get("2", {}).get("best_record") or {}
    seed4_best = t3b_seed_results.get("4", {}).get("best_record") or {}
    clauses = {
        "seed2_truth_pass": bool(seed2_best.get("measurement_truthful_for_training", False)),
        "seed4_truth_gap_le_2": int(seed4_best.get("truthful_window_longest_interior_gap", 9999) or 9999) <= 2,
        "seed3_bridge_not_regressed": bool(seed_results[str(BRIDGE_SEED)]["has_accepted_teacher"] and ((seed_results[str(BRIDGE_SEED)]["best_record"] or {}).get("measurement_truthful_for_training", False))),
        "good_seed_acceptance_not_regressed": all(bool(seed_results[str(seed)]["has_accepted_teacher"]) for seed in GOOD_SEEDS),
    }
    evidence = {
        "seed_results": seed_results,
        "hard_seed_reference": {"2": seed2_best, "4": seed4_best},
        "accepted_records": accepted,
        "saved_rollout_paths": saved_paths,
    }
    return _phase_result(
        ctx,
        phase="t4",
        status="PASS" if all(clauses.values()) else "FAIL",
        clauses=clauses,
        rca_classes=_rca_from_mapping(
            clauses,
            {
                "seed2_truth_pass": "anchor_invalid_or_measurement_discontinuity",
                "seed4_truth_gap_le_2": "truth_gap_excess",
                "seed3_bridge_not_regressed": "seed3_bridge_missing",
                "good_seed_acceptance_not_regressed": "good_seed_regression",
            },
        ),
        evidence=evidence,
        continuation_input_source=str(_phase_artifact_path("t3b")),
        input_artifacts=[str(_phase_artifact_path("t3b"))],
        output_artifacts=[str(_phase_artifact_path("t4")), *saved_paths],
        bounded_terminal_if_finalized="HARD_SEED_REPAIR_CAUSES_GENERALIZATION_OR_TRUTH_REGRESSION" if not all(clauses.values()) else None,
        elapsed=time.time() - t0,
    )


def _run_phase_t5(ctx: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    accepted_records = list(ctx["accepted_rollouts"].values())
    accepted_families = {str(item.get("teacher_fingerprint", "")) for item in accepted_records if str(item.get("teacher_fingerprint", ""))}
    strict_families = {str(item.get("teacher_fingerprint", "")) for item in accepted_records if (_tier_rank(item) == 2 and str(item.get("teacher_fingerprint", "")))}
    near_families = {str(item.get("teacher_fingerprint", "")) for item in accepted_records if (str(item.get("teacher_episode_class")) == "near_strict_teacher" and str(item.get("teacher_fingerprint", "")))}
    per_seed_counts: dict[int, int] = {}
    for item in accepted_records:
        per_seed_counts[int(item["seed"])] = per_seed_counts.get(int(item["seed"]), 0) + 1
    total = max(len(accepted_records), 1)
    max_share = max((count / total for count in per_seed_counts.values()), default=1.0)
    clauses = {
        "accepted_unique_teacher_families_ge_18": len(accepted_families) >= 18,
        "strict_unique_teacher_families_ge_8": len(strict_families) >= 8,
        "near_strict_unique_teacher_families_ge_6": len(near_families) >= 6,
        "no_single_seed_gt_40pct": max_share <= 0.40,
    }
    evidence = {
        "accepted_record_count": len(accepted_records),
        "accepted_seed_coverage": sorted({int(item["seed"]) for item in accepted_records}),
        "accepted_unique_teacher_family_count": len(accepted_families),
        "strict_unique_teacher_family_count": len(strict_families),
        "near_strict_unique_teacher_family_count": len(near_families),
        "per_seed_counts": per_seed_counts,
        "max_single_seed_share": max_share,
        "accepted_records": accepted_records,
    }
    return _phase_result(
        ctx,
        phase="t5",
        status="PASS" if all(clauses.values()) else "FAIL",
        clauses=clauses,
        rca_classes=_rca_from_mapping(
            clauses,
            {
                "accepted_unique_teacher_families_ge_18": "insufficient_unique_families",
                "strict_unique_teacher_families_ge_8": "strict_family_deficit",
                "near_strict_unique_teacher_families_ge_6": "near_strict_family_deficit",
                "no_single_seed_gt_40pct": "hard_seed_only_concentration",
            },
        ),
        evidence=evidence,
        continuation_input_source="accepted_rollout_corpus",
        input_artifacts=[str(_phase_artifact_path("t3b")), str(_phase_artifact_path("t4"))],
        output_artifacts=[str(_phase_artifact_path("t5"))],
        bounded_terminal_if_finalized="TEACHER_DIVERSITY_NOT_ESTABLISHED" if not all(clauses.values()) else None,
        elapsed=time.time() - t0,
    )


def _run_phase_t6(ctx: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    t1 = ctx["phase_results"].get("t1", {})
    t2a = ctx["phase_results"].get("t2a", {})
    t2b = ctx["phase_results"].get("t2b", {})
    p1b = ctx["phase_results"].get("p1b", {})
    t3b = ctx["phase_results"].get("t3b", {})
    t4 = ctx["phase_results"].get("t4", {})
    t5 = ctx["phase_results"].get("t5", {})
    t3b_seed_results = (t3b.get("evidence") or {}).get("seed_results", {})
    t4_seed_results = (t4.get("evidence") or {}).get("seed_results", {})
    t5_evidence = t5.get("evidence") or {}
    clauses = {
        "strict_semantics_aligned": t1.get("status") == "PASS",
        "t2a_passed": t2a.get("status") == "PASS",
        "t2b_passed": t2b.get("status") == "PASS",
        "p1b_frontier_valid": bool((p1b.get("evidence") or {}).get("candidate_frontier_valid", False)),
        "t3b_superior_to_t3a": bool((t3b.get("clauses") or {}).get("matched_superiority_over_t3a", False)),
        "seed2_has_reset_accepted_teacher": bool((t3b_seed_results.get("2") or {}).get("has_accepted_teacher", False)),
        "seed4_has_reset_accepted_teacher": bool((t3b_seed_results.get("4") or {}).get("has_accepted_teacher", False)),
        "at_least_one_hard_seed_strict": bool((t3b_seed_results.get("2") or {}).get("has_strict_teacher", False) or (t3b_seed_results.get("4") or {}).get("has_strict_teacher", False)),
        "seed3_has_accepted_truthful_bridge": bool((t4_seed_results.get(str(BRIDGE_SEED), {}) or {}).get("has_accepted_teacher", False) and (((t4_seed_results.get(str(BRIDGE_SEED), {}) or {}).get("best_record") or {}).get("measurement_truthful_for_training", False))),
        "accepted_unique_teacher_families_ge_18": bool((t5.get("clauses") or {}).get("accepted_unique_teacher_families_ge_18", False)),
        "strict_unique_teacher_families_ge_8": bool((t5.get("clauses") or {}).get("strict_unique_teacher_families_ge_8", False)),
        "near_strict_unique_teacher_families_ge_6": bool((t5.get("clauses") or {}).get("near_strict_unique_teacher_families_ge_6", False)),
    }
    status = "PASS" if all(clauses.values()) else "FAIL"
    evidence = {
        "t3b_seed_results": t3b_seed_results,
        "t4_seed_results": t4_seed_results,
        "t5_evidence": t5_evidence,
        "teacher_abstraction_sufficient_condition": bool(
            clauses["strict_semantics_aligned"]
            and clauses["t2a_passed"]
            and clauses["t2b_passed"]
            and clauses["p1b_frontier_valid"]
            and clauses["t3b_superior_to_t3a"]
            and clauses["seed2_has_reset_accepted_teacher"]
            and clauses["seed4_has_reset_accepted_teacher"]
            and clauses["at_least_one_hard_seed_strict"]
            and bool((t4.get("status") == "PASS"))
        ),
    }
    readiness_report = {
        "gate": "v84_teacher_readiness_contract",
        "run_id": ctx["run_id"],
        "timestamp": now_iso(),
        "branch": ctx["branch"],
        "head_commit": ctx["head"],
        "clauses": clauses,
        "passed": status == "PASS",
        "evidence": evidence,
    }
    _write_json(G6_TEACHER_READINESS_CONTRACT_PATH, readiness_report)
    evidence["readiness_report_path"] = str(G6_TEACHER_READINESS_CONTRACT_PATH)
    return _phase_result(
        ctx,
        phase="t6",
        status=status,
        clauses=clauses,
        rca_classes=_rca_from_mapping(
            clauses,
            {
                "strict_semantics_aligned": "teacher_abstraction_not_established",
                "t2a_passed": "teacher_abstraction_not_established",
                "t2b_passed": "teacher_abstraction_not_established",
                "p1b_frontier_valid": "teacher_abstraction_not_established",
                "t3b_superior_to_t3a": "teacher_abstraction_not_established",
                "seed2_has_reset_accepted_teacher": "hard_seed_reset_teacher_missing",
                "seed4_has_reset_accepted_teacher": "hard_seed_reset_teacher_missing",
                "at_least_one_hard_seed_strict": "hard_seed_reset_teacher_missing",
                "seed3_has_accepted_truthful_bridge": "bridge_seed_missing",
                "accepted_unique_teacher_families_ge_18": "teacher_diversity_not_established",
                "strict_unique_teacher_families_ge_8": "teacher_diversity_not_established",
                "near_strict_unique_teacher_families_ge_6": "teacher_diversity_not_established",
            },
        ),
        evidence=evidence,
        continuation_input_source="phase_clause_matrix",
        input_artifacts=[str(_phase_artifact_path(name)) for name in ["t1", "t2a", "t2b", "p1b", "t3b", "t4", "t5"]],
        output_artifacts=[str(_phase_artifact_path("t6")), str(G6_TEACHER_READINESS_CONTRACT_PATH)],
        bounded_terminal_if_finalized="TEACHER_READINESS_NOT_ESTABLISHED" if status != "PASS" else None,
        elapsed=time.time() - t0,
    )


def _stage_phase_result(ctx: dict[str, Any], phase: str, stage_summary: dict[str, Any], *, diagnostic_only: bool) -> dict[str, Any]:
    dataset_summary = dict(stage_summary.get("dataset_summary") or {})
    probe_summary = dict(stage_summary.get("probe_summary") or {})
    eval_summary = dict(stage_summary.get("eval_summary") or {})
    clauses = {
        "dataset_built": bool(stage_summary.get("dataset_valid", False)),
        "train_completed": bool(stage_summary.get("train_passed", False)),
        "train_probe_executed": bool(stage_summary.get("train_passed", False) and probe_summary != {}),
        "heldout_executed_if_probe_passed": (not bool(stage_summary.get("train_probe_passed", False))) or bool(stage_summary.get("heldout_eval_run", False)),
    }
    rca: list[str] = []
    if not clauses["dataset_built"]:
        rca.append("dataset_prepare_failed")
    if not clauses["train_completed"]:
        rca.append("train_failed")
    if clauses["train_completed"] and not clauses["train_probe_executed"]:
        rca.append("train_probe_no_signal")
    if not clauses["heldout_executed_if_probe_passed"]:
        rca.append("heldout_not_supported")
    if bool(stage_summary.get("heldout_eval_run", False)) and not bool(stage_summary.get("claim_supported", False)):
        rca.append("heldout_fail_after_probe_pass")
    status = "PASS" if all(clauses.values()) else "FAIL"
    evidence = {
        "stage_verdict": stage_summary.get("stage_verdict"),
        "claim_supported": bool(stage_summary.get("claim_supported", False)),
        "attach_bridge_pass": bool(stage_summary.get("attach_bridge_pass", False)),
        "train_probe_passed": bool(stage_summary.get("train_probe_passed", False)),
        "selected_bridge_checkpoint": stage_summary.get("selected_bridge_checkpoint"),
        "selected_checkpoint_step": stage_summary.get("selected_checkpoint_step"),
        "unsupported_by_readiness_contract": bool(stage_summary.get("unsupported_by_readiness_contract", False)),
        "dataset_summary": dataset_summary,
        "probe_summary": probe_summary,
        "eval_summary": eval_summary,
        "stage_summary": stage_summary,
    }
    return _phase_result(
        ctx,
        phase=phase,
        status=status,
        clauses=clauses,
        rca_classes=rca,
        evidence=evidence,
        continuation_input_source="explicit_rollout_stage",
        input_artifacts=[str(_phase_artifact_path("t6"))],
        output_artifacts=[str(_phase_artifact_path(phase)), str(TINY_RETRAIN_DATASET_BUILD_PATH), str(G8_SUMMARY_PATH), str(G8_PROBE_PATH), str(G9_SUMMARY_PATH)],
        force_diagnostic=diagnostic_only,
    )


def _prepare_rollout_corpus(ctx: dict[str, Any], active_train_state_mode: str) -> list[Path]:
    if active_train_state_mode == "S0":
        return [Path(path) for path in sorted(ctx["accepted_rollouts"].keys())]
    if active_train_state_mode in ctx["cached_rollout_corpora"]:
        return [Path(path) for path in ctx["cached_rollout_corpora"][active_train_state_mode]]
    paths: list[Path] = []
    records = list(ctx["accepted_rollouts"].values())
    if not records:
        ctx["cached_rollout_corpora"][active_train_state_mode] = []
        return []
    for item in records:
        expected = _base_rollout_plan(ctx, bridge_stage=f"v84_retrain_{active_train_state_mode.lower()}_corpus", bridge_attempt="materialize", active_train_state_mode=active_train_state_mode)
        context = _canonical_rollout_context(expected)
        rollout = build_robot_rollout(
            seed=int(item["seed"]),
            grasp_pose_world=np.eye(4, dtype=np.float32),
            episode_index=int(item["repeat_idx"]),
            max_steps=int(item.get("max_steps", 144)),
            contract=context["contract"],
            rotation_source=context["rotation_source"],
            claim_policy=context["claim_policy"],
            interventions=dict(context["base_interventions"]),
            assay_warm_start_kind=item.get("warm_start_kind"),
            teacher_controller_mode=str(item.get("controller_mode", "interaction_frame_hybrid")),
            pull_open_fraction=0.92,
        )
        rollout = _augment_rollout_metadata(context["controller"], rollout, context["spec"], expected)
        if str(rollout.get("teacher_episode_class", "rejected_teacher")) not in {"strict_teacher", "near_strict_teacher"}:
            continue
        out_dir = ROLLOUT_ROOT / "retrain" / active_train_state_mode.lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"seed_{int(item['seed']):03d}_episode_{int(item['repeat_idx']):02d}_{str(item.get('controller_mode', 'controller'))}.npz"
        save_robot_rollout(out_path, rollout)
        paths.append(out_path)
    ctx["cached_rollout_corpora"][active_train_state_mode] = [str(path) for path in paths]
    return paths


def _run_phase_b0(ctx: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    if ctx["phase_results"].get("t6", {}).get("status") == "PASS":
        return _phase_result(
            ctx,
            phase="b0",
            status="SKIPPED",
            clauses={},
            rca_classes=[],
            evidence={"reason": "t6_passed_authoritative_path_enabled"},
            continuation_input_source=str(_phase_artifact_path("t6")),
            input_artifacts=[str(_phase_artifact_path("t6"))],
            output_artifacts=[str(_phase_artifact_path("b0"))],
            force_diagnostic=True,
            elapsed=time.time() - t0,
        )
    os.environ["MINT_RUN_INSTANCE_ID"] = ctx["run_id"]
    stage_summary = _run_stage(
        load_terminal_artifacts(),
        active_train_state_mode="S0",
        bridge_stage="v84_b0_diagnostic_microtrain",
        bridge_attempt="diagnostic",
        episodes_per_seed=HONEST_EPISODES_PER_SEED,
        min_train_episodes=HONEST_MIN_TRAIN_EPISODES,
        train_steps=HONEST_TRAIN_STEPS,
        save_freq=HONEST_SAVE_FREQ,
        checkpoint_probe_steps=HONEST_PROBE_STEPS,
        require_teacher_readiness=False,
        diagnostic_only=True,
        explicit_rollout_paths=_prepare_rollout_corpus(ctx, "S0"),
    )
    result = _stage_phase_result(ctx, "b0", stage_summary, diagnostic_only=True)
    result["phase_elapsed_sec"] = round(float(time.time() - t0), 4)
    return result


def _run_authoritative_stage(ctx: dict[str, Any], phase: str, *, active_train_state_mode: str, bridge_stage: str, bridge_attempt: str, episodes_per_seed: int, min_train_episodes: int, train_steps: int, save_freq: int, checkpoint_probe_steps: list[int]) -> dict[str, Any]:
    t0 = time.time()
    if ctx["phase_results"].get("t6", {}).get("status") != "PASS":
        return _phase_result(
            ctx,
            phase=phase,
            status="SKIPPED",
            clauses={},
            rca_classes=["teacher_readiness_not_established"],
            evidence={"reason": "t6_failed"},
            continuation_input_source=str(_phase_artifact_path("t6")),
            input_artifacts=[str(_phase_artifact_path("t6"))],
            output_artifacts=[str(_phase_artifact_path(phase))],
            elapsed=time.time() - t0,
        )
    os.environ["MINT_RUN_INSTANCE_ID"] = ctx["run_id"]
    stage_summary = _run_stage(
        load_terminal_artifacts(),
        active_train_state_mode=active_train_state_mode,
        bridge_stage=bridge_stage,
        bridge_attempt=bridge_attempt,
        episodes_per_seed=episodes_per_seed,
        min_train_episodes=min_train_episodes,
        train_steps=train_steps,
        save_freq=save_freq,
        checkpoint_probe_steps=checkpoint_probe_steps,
        require_teacher_readiness=True,
        diagnostic_only=False,
        explicit_rollout_paths=_prepare_rollout_corpus(ctx, active_train_state_mode),
    )
    ctx["b_stage_history"].append(stage_summary)
    result = _stage_phase_result(ctx, phase, stage_summary, diagnostic_only=False)
    result["phase_elapsed_sec"] = round(float(time.time() - t0), 4)
    return result


def _run_named_phase(ctx: dict[str, Any], phase: str) -> dict[str, Any]:
    if phase == "t0":
        return _run_phase_t0(ctx)
    if phase == "t1":
        return _run_phase_t1(ctx)
    if phase == "t2a":
        return _run_phase_t2a(ctx)
    if phase == "t2b":
        return _run_phase_t2b(ctx)
    if phase == "p1a":
        return _run_phase_p1a(ctx)
    if phase == "p1b":
        return _run_phase_p1b(ctx)
    if phase == "t3a":
        return _run_phase_t3a(ctx)
    if phase == "t3b":
        return _run_phase_t3b(ctx)
    if phase == "t4":
        return _run_phase_t4(ctx)
    if phase == "t5":
        return _run_phase_t5(ctx)
    if phase == "t6":
        return _run_phase_t6(ctx)
    if phase == "b0":
        return _run_phase_b0(ctx)
    if phase == "b1":
        return _run_authoritative_stage(ctx, "b1", active_train_state_mode="S0", bridge_stage="v84_b1_s0_honest", bridge_attempt="honest", episodes_per_seed=HONEST_EPISODES_PER_SEED, min_train_episodes=HONEST_MIN_TRAIN_EPISODES, train_steps=HONEST_TRAIN_STEPS, save_freq=HONEST_SAVE_FREQ, checkpoint_probe_steps=HONEST_PROBE_STEPS)
    if phase == "b2":
        return _run_authoritative_stage(ctx, "b2", active_train_state_mode="S0", bridge_stage="v84_b2_s0_amplification", bridge_attempt="amplification", episodes_per_seed=AMPLIFIED_EPISODES_PER_SEED, min_train_episodes=AMPLIFIED_MIN_TRAIN_EPISODES, train_steps=AMPLIFIED_TRAIN_STEPS, save_freq=AMPLIFIED_SAVE_FREQ, checkpoint_probe_steps=AMPLIFIED_PROBE_STEPS)
    if phase == "b3":
        return _run_authoritative_stage(ctx, "b3", active_train_state_mode="S2", bridge_stage="v84_b3_s2_honest", bridge_attempt="honest", episodes_per_seed=HONEST_EPISODES_PER_SEED, min_train_episodes=HONEST_MIN_TRAIN_EPISODES, train_steps=HONEST_TRAIN_STEPS, save_freq=HONEST_SAVE_FREQ, checkpoint_probe_steps=HONEST_PROBE_STEPS)
    if phase == "b4":
        return _run_authoritative_stage(ctx, "b4", active_train_state_mode="S2", bridge_stage="v84_b4_s2_amplification", bridge_attempt="amplification", episodes_per_seed=AMPLIFIED_EPISODES_PER_SEED, min_train_episodes=AMPLIFIED_MIN_TRAIN_EPISODES, train_steps=AMPLIFIED_TRAIN_STEPS, save_freq=AMPLIFIED_SAVE_FREQ, checkpoint_probe_steps=AMPLIFIED_PROBE_STEPS)
    raise ValueError(f"unsupported phase: {phase}")


def _hydrate_context_from_artifacts(ctx: dict[str, Any], phases: list[str]) -> None:
    for phase in phases:
        path = _phase_artifact_path(phase)
        if not path.exists():
            raise SystemExit(f"Missing artifact required for resume: {path}")
        payload = load_json(path, {})
        if not payload:
            raise SystemExit(f"Empty artifact required for resume: {path}")
        ctx["phase_results"][phase] = payload
        if phase == "t1" and payload.get("status") != "PASS":
            ctx["semantics_blocked"] = True
        if phase in {"t3b", "t4"}:
            for item in (payload.get("evidence") or {}).get("accepted_records", []):
                if item.get("rollout_path"):
                    ctx["accepted_rollouts"][str(item["rollout_path"])] = item
        if phase in {"b1", "b2", "b3", "b4"}:
            stage_summary = ((payload.get("evidence") or {}).get("stage_summary") or {})
            if stage_summary:
                ctx["b_stage_history"].append(stage_summary)
                ctx["authoritative_retrain_executed"] = True


def _finalize(ctx: dict[str, Any]) -> None:
    t6_pass = ctx["phase_results"].get("t6", {}).get("status") == "PASS"
    if t6_pass and not ctx["b_stage_history"]:
        for phase in ["b1", "b2", "b3", "b4"]:
            stage_summary = ((ctx["phase_results"].get(phase, {}).get("evidence") or {}).get("stage_summary") or {})
            if stage_summary:
                ctx["b_stage_history"].append(stage_summary)
        if ctx["b_stage_history"]:
            ctx["authoritative_retrain_executed"] = True
    if t6_pass and ctx["b_stage_history"]:
        if any(item.get("stage_verdict") == "claim_supported" for item in ctx["b_stage_history"]):
            final_verdict = "claim_supported"
            scientific_terminal_state = None
        elif any(item.get("stage_verdict") == "attach_bridge_established_not_claim_supported" for item in ctx["b_stage_history"]):
            final_verdict = "attach_bridge_established_not_claim_supported"
            scientific_terminal_state = "attach_bridge_established_not_claim_supported"
        else:
            final_verdict = "TINY_RETRAIN_NOT_ESTABLISHED_AFTER_S0_S2"
            scientific_terminal_state = "TINY_RETRAIN_NOT_ESTABLISHED_AFTER_S0_S2"
        _write_loop_summary(ctx["b_stage_history"], final_verdict, scientific_terminal_state)
        ctx["authoritative_retrain_executed"] = True
        ctx["final_verdict"] = final_verdict
    elif ctx["phase_results"].get("b0", {}).get("status") in {"PASS", "FAIL"}:
        b0_evidence = ctx["phase_results"]["b0"].get("evidence", {})
        if bool(b0_evidence.get("claim_supported", False)):
            ctx["final_verdict"] = "diagnostic_claim_supported_but_not_authoritative"
        else:
            ctx["final_verdict"] = "TEACHER_READINESS_NOT_ESTABLISHED"
    elif ctx["phase_results"].get("t0", {}).get("status") == "ERROR":
        ctx["final_verdict"] = "T0_INTEGRITY_FATAL"
    else:
        ctx["final_verdict"] = "TEACHER_READINESS_NOT_ESTABLISHED"
    _write_matrix(ctx)


def main() -> int:
    parser = argparse.ArgumentParser(description="v8.4 always-continue evidence program")
    parser.add_argument("--phase", choices=PHASE_ORDER + ["all"], default="all")
    parser.add_argument("--resume-from", choices=PHASE_ORDER)
    parser.add_argument("--authoritative-mode", type=_bool_arg, default=True)
    args = parser.parse_args()

    branch = _git(["git", "branch", "--show-current"])
    head = _git(["git", "rev-parse", "HEAD"])
    existing = load_json(EVIDENCE_MATRIX_PATH, {}) if args.resume_from else {}
    run_id = str(existing.get("run_id") or _new_run_id(head))
    ctx: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": now_iso(),
        "branch": branch,
        "head": head,
        "authoritative_mode": bool(args.authoritative_mode),
        "phase_results": {},
        "accepted_rollouts": {},
        "cached_rollout_corpora": {},
        "semantics_blocked": False,
        "b_stage_history": [],
        "authoritative_retrain_executed": False,
        "final_verdict": None,
    }
    os.environ["MINT_RUN_INSTANCE_ID"] = run_id

    if args.phase == "all":
        if args.resume_from:
            start_idx = PHASE_ORDER.index(args.resume_from)
            _hydrate_context_from_artifacts(ctx, PHASE_ORDER[:start_idx])
            phases = PHASE_ORDER[start_idx:]
        else:
            phases = list(PHASE_ORDER)
    else:
        phases = [args.phase]

    print("=" * 78)
    print("v8.4 Always-Continue Evidence Program")
    print(f"Timestamp: {ctx['timestamp']}")
    print(f"Run ID: {ctx['run_id']}")
    print(f"Branch: {branch}")
    print(f"Head: {head}")
    print(f"Authoritative mode: {ctx['authoritative_mode']}")
    print(f"Phases: {phases}")
    print("=" * 78)

    for phase in phases:
        result = _run_named_phase(ctx, phase)
        _write_phase(ctx, result)
        print(f"[{phase}] status={result['status']} terminal={result.get('bounded_terminal_if_finalized')} failed={result.get('failed_clauses')}")
        if phase == "t0" and result["status"] == "ERROR":
            ctx["final_verdict"] = result.get("bounded_terminal_if_finalized") or "T0_INTEGRITY_FATAL"
            _write_matrix(ctx)
            return 1

    _finalize(ctx)
    print("\nFinal verdict:", ctx.get("final_verdict"))
    print("Evidence matrix:", EVIDENCE_MATRIX_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

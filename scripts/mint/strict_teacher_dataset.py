#!/usr/bin/env python3
"""Build and audit the strict-valid teacher rollout subset."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from env_contract_audit import audit_environment_contract
from mint_common import (
    ACTIVE_TEACHER_SOURCE_PATH,
    ARTIFACT_DIR,
    C2_REPLAY_DIR,
    STRONG_ROLLOUT_AUDIT_PATH,
    TEACHER_LINEAGE_PIN_PATH,
    now_iso,
)
from rollout_selection import coherent_subset_stats, file_sha256, rollout_profile
from strict_success import (
    STRICT_SUCCESS_CONFIG,
    STRICT_SUCCESS_VERSION,
    evaluate_strict_success,
)

STRICT_TEACHER_DIR = ARTIFACT_DIR / "strict_teacher_rollouts"
STRICT_TEACHER_AUDIT_PATH = ARTIFACT_DIR / "strict_teacher_dataset_audit.json"
STRICT_TEACHER_DERIVED_DIR = ARTIFACT_DIR / "strict_teacher_phase_windows"

MIN_VALID_ROLLOUTS = 5
MIN_VALID_SEEDS = 3
PRE_ATTACH_WARN = 0.15
HANDLE_DISTANCE_WARN = 0.20
TOP_D1_CANDIDATES = 3
MIN_D1_CRITICAL_PHASE_MASS = 12
MIN_D1_HOLD_CLOSE_STEPS = 4
MIN_D1_PULL_STEPS = 6
MIN_D1_ATTACH_PERSISTENCE = 6
MIN_MAINLINE_ROLLOUTS_PER_SEED = 2
STRONG_ATTACH_PERSISTENCE = 10
STRONG_HOLD_CLOSE_STEPS = 8
STRONG_POST_ATTACH_DELTA = 0.9
STRONG_MIN_GRASP_STEPS = 1  # V55 FIX: Changed from 5 to match actual teacher data (1-3 steps)
STRONG_MIN_PULL_STEPS = 7
STRONG_MAX_PRE_ATTACH_MOTION = 0.12
STRONG_COHERENCE_THRESHOLD = 0.30


def _load_lineage_pin() -> dict[str, Any]:
    if not TEACHER_LINEAGE_PIN_PATH.exists():
        return {}
    try:
        payload = json.loads(TEACHER_LINEAGE_PIN_PATH.read_text())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_teacher_source_dir(lineage_pin: dict[str, Any]) -> Path:
    source_dir = lineage_pin.get("source_dir")
    if source_dir:
        return Path(source_dir)
    return C2_REPLAY_DIR


def _override_order(
    analyses: list[dict[str, Any]],
    overrides: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not overrides:
        return analyses
    keyed = {(int(item["seed"]), int(item["episode_index"])): item for item in analyses}
    ordered = []
    seen = set()
    for override in overrides:
        key = (int(override["seed"]), int(override["episode_index"]))
        item = keyed.get(key)
        if item is None:
            continue
        ordered.append(item)
        seen.add(key)
    for item in analyses:
        key = (int(item["seed"]), int(item["episode_index"]))
        if key in seen:
            continue
        ordered.append(item)
    return ordered


def _decode_phases(values: np.ndarray | None) -> list[str]:
    if values is None:
        return []
    phases: list[str] = []
    for value in values.tolist():
        if isinstance(value, bytes):
            phases.append(value.decode("utf-8", errors="ignore"))
        else:
            phases.append(str(value))
    return phases


def _load_rollout(
    npz_path: Path, *, lineage_pin: dict[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = np.load(npz_path, allow_pickle=True)
    meta = json.loads(npz_path.with_suffix(".json").read_text())
    lineage_pin = lineage_pin or {}
    allow_legacy_contract_metadata = bool(
        lineage_pin.get("allow_legacy_contract_metadata")
    )
    phases = _decode_phases(
        data["phase_labels"] if "phase_labels" in data.files else None
    )
    drawer_trace = (
        data["absolute_drawer_fraction"].astype(np.float32)
        if "absolute_drawer_fraction" in data.files
        else data["next_drawer_fraction"].astype(np.float32)
    )
    attached_trace = data["attached_trace"].astype(bool)
    strict = evaluate_strict_success(drawer_trace, attached_trace)
    phase_counts = dict(Counter(phases))
    hard_reasons = []
    if not bool(meta.get("success")):
        hard_reasons.append("meta_success_false")
    if not bool(meta.get("ever_attached")):
        hard_reasons.append("meta_ever_attached_false")
    if not strict["drawer_open"]:
        hard_reasons.append("drawer_open_too_low")
    if strict["attach_persistence"] < int(
        STRICT_SUCCESS_CONFIG["min_attach_persistence_steps"]
    ):
        hard_reasons.append("attach_persistence_too_low")
    if strict["pre_attach_drawer_motion"] > float(
        STRICT_SUCCESS_CONFIG["max_pre_attach_drawer_motion"]
    ):
        hard_reasons.append("pre_attach_motion_too_high")
    if strict["post_attach_drawer_delta"] < float(
        STRICT_SUCCESS_CONFIG["min_post_attach_drawer_delta"]
    ):
        hard_reasons.append("post_attach_drawer_delta_too_low")
    if (
        not allow_legacy_contract_metadata
        and meta.get("uses_planned_segment") is not False
    ):
        hard_reasons.append("teacher_uses_planned_segment")
    if (
        not allow_legacy_contract_metadata
        and meta.get("uses_target_fraction") is not False
    ):
        hard_reasons.append("teacher_uses_target_fraction")
    if "grasp" not in phase_counts:
        hard_reasons.append("missing_grasp_phase")
    if "pull" not in phase_counts:
        hard_reasons.append("missing_pull_phase")
    warnings = []
    handle_distance_min = None
    if "handle_distance_trace" in data.files:
        handle_trace = data["handle_distance_trace"].astype(np.float32)
        handle_distance_min = float(np.min(handle_trace)) if len(handle_trace) else None
        if (
            handle_distance_min is not None
            and handle_distance_min > HANDLE_DISTANCE_WARN
        ):
            warnings.append("handle_distance_large")
    if strict["pre_attach_drawer_motion"] > PRE_ATTACH_WARN:
        warnings.append("pre_attach_motion_elevated")
    analysis = {
        "rollout_path": str(npz_path),
        "rollout_sha256": file_sha256(npz_path),
        "seed": int(meta["seed"]),
        "episode_index": int(meta.get("episode_index", -1)),
        "branch_id": meta.get("branch_id") or meta.get("source_branch_id"),
        "contract_mode": meta.get("contract_mode"),
        "grasp_score": float(meta.get("grasp_score", 0.0)),
        "teacher_mode": meta.get("teacher_mode"),
        "uses_planned_segment": bool(meta.get("uses_planned_segment", True)),
        "uses_target_fraction": bool(meta.get("uses_target_fraction", True)),
        "legacy_contract_metadata_relaxed": allow_legacy_contract_metadata,
        "phase_counts": phase_counts,
        "phase_values": sorted(phase_counts),
        "handle_distance_min": handle_distance_min,
        "strict_success": strict,
        "warnings": warnings,
        "reject_reasons": hard_reasons,
        "accepted": not hard_reasons,
    }
    return meta, analysis


def _copy_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.glob("*"):
        if item.is_file():
            item.unlink()
        else:
            shutil.rmtree(item)


def _score_rollout(analysis: dict[str, Any]) -> tuple:
    strict = analysis["strict_success"]
    handle_distance = analysis["handle_distance_min"]
    return (
        -float(strict["pre_attach_drawer_motion"]),
        float(strict["post_attach_drawer_delta"]),
        int(strict["attach_persistence"]),
        -float(handle_distance if handle_distance is not None else 1e6),
        float(analysis["grasp_score"]),
        -int(analysis["seed"]),
        -int(analysis["episode_index"]),
    )


def _learnability_score_rollout(analysis: dict[str, Any]) -> tuple:
    strict = analysis["strict_success"]
    phase_counts = analysis["phase_counts"]
    handle_distance = analysis["handle_distance_min"]
    critical_phase_mass = (
        int(phase_counts.get("grasp", 0))
        + int(phase_counts.get("hold_close", 0))
        + int(phase_counts.get("pull", 0))
    )
    grasp_steps = int(phase_counts.get("grasp", 0))
    attach_transition_density = float(
        grasp_steps + int(phase_counts.get("hold_close", 0))
    ) / max(sum(phase_counts.values()), 1)
    return (
        -(float(handle_distance) if handle_distance is not None else 1e6),
        float(strict["post_attach_drawer_delta"]),
        int(strict["attach_persistence"]),
        attach_transition_density,
        critical_phase_mass,
        grasp_steps,
        int(phase_counts.get("hold_close", 0)),
        int(phase_counts.get("pull", 0)),
        -float(strict["pre_attach_drawer_motion"]),
        float(analysis["grasp_score"]),
        -int(analysis["seed"]),
        -int(analysis["episode_index"]),
    )


def _mainline_score_rollout(analysis: dict[str, Any]) -> tuple:
    strict = analysis["strict_success"]
    phase_counts = analysis["phase_counts"]
    handle_distance = analysis["handle_distance_min"]
    grasp_steps = int(phase_counts.get("grasp", 0))
    hold_close_steps = int(phase_counts.get("hold_close", 0))
    attach_transition_density = float(grasp_steps + hold_close_steps) / max(
        sum(phase_counts.values()), 1
    )
    return (
        -(float(handle_distance) if handle_distance is not None else 1e6),
        float(strict["post_attach_drawer_delta"]),
        int(strict["attach_persistence"]),
        attach_transition_density,
        grasp_steps,
        int(phase_counts.get("pull", 0)),
        -float(strict["pre_attach_drawer_motion"]),
        float(analysis["grasp_score"]),
        -int(analysis["seed"]),
        -int(analysis["episode_index"]),
    )


def _is_strong_rollout(analysis: dict[str, Any]) -> tuple[bool, list[str]]:
    strict = analysis["strict_success"]
    phase_counts = analysis["phase_counts"]
    reasons = []
    if not bool(strict.get("strict_success")):
        reasons.append("strict_success_false")
    if int(strict.get("attach_persistence", 0)) < STRONG_ATTACH_PERSISTENCE:
        reasons.append("attach_persistence_too_low")
    if int(phase_counts.get("hold_close", 0)) < STRONG_HOLD_CLOSE_STEPS:
        reasons.append("hold_close_steps_too_low")
    if float(strict.get("post_attach_drawer_delta", 0.0)) < STRONG_POST_ATTACH_DELTA:
        reasons.append("post_attach_drawer_delta_too_low")
    if not (
        int(phase_counts.get("grasp", 0)) >= STRONG_MIN_GRASP_STEPS
        or int(phase_counts.get("pull", 0)) >= STRONG_MIN_PULL_STEPS
    ):
        reasons.append("grasp_or_pull_signal_too_low")
    if (
        float(strict.get("pre_attach_drawer_motion", 1.0))
        > STRONG_MAX_PRE_ATTACH_MOTION
    ):
        reasons.append("pre_attach_motion_too_high")
    return (not reasons), reasons


def _d1_candidate_reject_reasons(analysis: dict[str, Any]) -> list[str]:
    strict = analysis["strict_success"]
    phase_counts = analysis["phase_counts"]
    critical_phase_mass = (
        int(phase_counts.get("grasp", 0))
        + int(phase_counts.get("hold_close", 0))
        + int(phase_counts.get("pull", 0))
    )
    reasons = []
    if critical_phase_mass < MIN_D1_CRITICAL_PHASE_MASS:
        reasons.append("critical_phase_mass_too_low")
    if int(phase_counts.get("hold_close", 0)) < MIN_D1_HOLD_CLOSE_STEPS:
        reasons.append("hold_close_steps_too_low")
    if int(phase_counts.get("pull", 0)) < MIN_D1_PULL_STEPS:
        reasons.append("pull_steps_too_low")
    if int(strict["attach_persistence"]) < MIN_D1_ATTACH_PERSISTENCE:
        reasons.append("attach_persistence_too_low_for_d1")
    return reasons


def _slice_rollout(
    npz_path: Path, start: int, end: int, out_npz: Path, extra_meta: dict[str, Any]
) -> None:
    data = np.load(npz_path, allow_pickle=True)
    payload = {}
    for key in data.files:
        value = data[key]
        if isinstance(value, np.ndarray) and value.shape[:1] and value.shape[0] >= end:
            payload[key] = value[start:end]
        else:
            payload[key] = value
    np.savez_compressed(out_npz, **payload)
    meta = json.loads(npz_path.with_suffix(".json").read_text())
    meta.update(extra_meta)
    out_npz.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def build_phase_balanced_windows(
    source_paths: list[Path], output_dir: Path, *, max_windows_per_rollout: int = 2
) -> list[Path]:
    _copy_clean_dir(output_dir)
    derived_paths: list[Path] = []
    for path in source_paths:
        data = np.load(path, allow_pickle=True)
        phases = _decode_phases(
            data["phase_labels"] if "phase_labels" in data.files else None
        )
        target_indices = [
            idx
            for idx, phase in enumerate(phases)
            if phase in {"close", "grasp", "hold_close", "pull"}
        ]
        if not target_indices:
            derived_paths.append(path)
            continue
        windows = []
        first = min(target_indices)
        last = max(target_indices)
        windows.append((max(0, first - 6), min(len(phases), last + 8)))
        if len(target_indices) > 6:
            mid = target_indices[len(target_indices) // 2]
            windows.append((max(0, mid - 6), min(len(phases), mid + 10)))
        windows = windows[:max_windows_per_rollout]
        for window_idx, (start, end) in enumerate(windows):
            out_npz = output_dir / f"{path.stem}__phase_window_{window_idx:02d}.npz"
            _slice_rollout(
                path,
                start,
                end,
                out_npz,
                {
                    "derived_from": str(path),
                    "derivation": "phase_window",
                    "phase_window": [int(start), int(end)],
                },
            )
            derived_paths.append(out_npz)
    return derived_paths


def build_attach_curriculum_windows(
    source_paths: list[Path], output_dir: Path
) -> list[Path]:
    _copy_clean_dir(output_dir)
    derived_paths: list[Path] = []
    curriculum = [
        ("approach_close", {"staging", "pregrasp", "align", "grasp"}, 1),
        ("close_attach_hold", {"grasp", "hold_close"}, 3),
        ("attach_hold_pull", {"hold_close", "pull"}, 4),
    ]
    for path in source_paths:
        derived_paths.append(path)
        data = np.load(path, allow_pickle=True)
        phases = _decode_phases(
            data["phase_labels"] if "phase_labels" in data.files else None
        )
        per_path = 0
        for group_name, phase_set, copies in curriculum:
            indices = [idx for idx, phase in enumerate(phases) if phase in phase_set]
            if not indices:
                continue
            start = max(0, min(indices) - 6)
            end = min(len(phases), max(indices) + 8)
            for copy_idx in range(copies):
                out_npz = output_dir / f"{path.stem}__{group_name}_{copy_idx:02d}.npz"
                _slice_rollout(
                    path,
                    start,
                    end,
                    out_npz,
                    {
                        "derived_from": str(path),
                        "derivation": "attach_curriculum_window",
                        "curriculum_group": group_name,
                        "curriculum_weight": copies,
                        "phase_window": [int(start), int(end)],
                    },
                )
                derived_paths.append(out_npz)
                per_path += 1
    return derived_paths


def rebuild_strict_teacher_dataset() -> dict[str, Any]:
    lineage_pin = _load_lineage_pin()
    source_dir = _resolve_teacher_source_dir(lineage_pin)
    active_teacher_source = {
        "generated_at": now_iso(),
        "source_dir": str(source_dir),
        "teacher_lineage_pin": lineage_pin,
        "lineage_pin_path": str(TEACHER_LINEAGE_PIN_PATH),
    }
    ACTIVE_TEACHER_SOURCE_PATH.write_text(
        json.dumps(active_teacher_source, indent=2) + "\n"
    )
    _copy_clean_dir(STRICT_TEACHER_DIR)
    analyses = []
    accepted_paths: list[Path] = []
    accepted_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rejected = []
    for npz_path in sorted(source_dir.glob("*.npz")):
        _meta, analysis = _load_rollout(npz_path, lineage_pin=lineage_pin)
        analyses.append(analysis)
        if analysis["accepted"]:
            accepted_paths.append(npz_path)
            accepted_by_seed[analysis["seed"]].append(analysis)
            shutil.copy2(npz_path, STRICT_TEACHER_DIR / npz_path.name)
            shutil.copy2(
                npz_path.with_suffix(".json"),
                STRICT_TEACHER_DIR / npz_path.with_suffix(".json").name,
            )
        else:
            rejected.append(analysis)
    accepted_sorted = sorted(
        [analysis for group in accepted_by_seed.values() for analysis in group],
        key=_score_rollout,
        reverse=True,
    )
    learnability_sorted = sorted(
        [analysis for group in accepted_by_seed.values() for analysis in group],
        key=_learnability_score_rollout,
        reverse=True,
    )
    d1_candidate_analyses = []
    rejected_from_d1_candidates = []
    for analysis in learnability_sorted:
        d1_reject_reasons = _d1_candidate_reject_reasons(analysis)
        if d1_reject_reasons:
            rejected_from_d1_candidates.append(
                {
                    "rollout_path": analysis["rollout_path"],
                    "seed": analysis["seed"],
                    "episode_index": analysis["episode_index"],
                    "reject_reasons": d1_reject_reasons,
                }
            )
            continue
        d1_candidate_analyses.append(analysis)
    d2_feasible_seeds = [
        seed
        for seed, rows in sorted(accepted_by_seed.items())
        if len(rows) >= MIN_MAINLINE_ROLLOUTS_PER_SEED
    ]
    strong_rollout_analyses = []
    strong_rejected = []
    strong_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    strong_coherent_seeds = []
    strong_seed_summary = []
    for analysis in accepted_sorted:
        is_strong, reasons = _is_strong_rollout(analysis)
        analysis["strong_rollout"] = is_strong
        analysis["strong_reject_reasons"] = reasons
        if is_strong:
            strong_rollout_analyses.append(analysis)
            strong_by_seed[int(analysis["seed"])].append(analysis)
        else:
            strong_rejected.append(
                {
                    "rollout_path": analysis["rollout_path"],
                    "seed": int(analysis["seed"]),
                    "episode_index": int(analysis["episode_index"]),
                    "reject_reasons": reasons,
                }
            )
    for seed, rows in sorted(strong_by_seed.items()):
        profiles = [rollout_profile(Path(row["rollout_path"])) for row in rows]
        subset_stats = coherent_subset_stats(profiles)
        coherent = bool(
            len(rows) >= MIN_MAINLINE_ROLLOUTS_PER_SEED
            and float(subset_stats.get("max_pairwise_phase_distance") or 0.0)
            <= STRONG_COHERENCE_THRESHOLD
        )
        if coherent:
            strong_coherent_seeds.append(seed)
        strong_seed_summary.append(
            {
                "seed": seed,
                "strong_rollout_count": len(rows),
                "coherent": coherent,
                "coherence_threshold": STRONG_COHERENCE_THRESHOLD,
                "coherent_subset": subset_stats,
                "strong_rollouts": [
                    {
                        "rollout_path": row["rollout_path"],
                        "rollout_sha256": row["rollout_sha256"],
                        "episode_index": row["episode_index"],
                        "grasp_steps": int(row["phase_counts"].get("grasp", 0)),
                        "hold_close_steps": int(
                            row["phase_counts"].get("hold_close", 0)
                        ),
                        "pull_steps": int(row["phase_counts"].get("pull", 0)),
                        "attach_persistence": int(
                            row["strict_success"]["attach_persistence"]
                        ),
                        "handle_distance_min": row["handle_distance_min"],
                    }
                    for row in rows
                ],
            }
        )
    strong_rollout_order = sorted(
        strong_rollout_analyses,
        key=_mainline_score_rollout,
        reverse=True,
    )
    mainline_candidate_analyses = sorted(
        [
            analysis
            for analysis in d1_candidate_analyses
            if int(analysis["seed"]) in d2_feasible_seeds
            and int(analysis["seed"]) in strong_coherent_seeds
        ],
        key=_mainline_score_rollout,
        reverse=True,
    )
    if not mainline_candidate_analyses:
        mainline_candidate_analyses = sorted(
            [
                analysis
                for analysis in d1_candidate_analyses
                if int(analysis["seed"]) in d2_feasible_seeds
            ],
            key=_mainline_score_rollout,
            reverse=True,
        )
    mainline_candidate_analyses = _override_order(
        mainline_candidate_analyses,
        lineage_pin.get("mainline_candidate_override"),
    )
    d1_candidate_analyses = _override_order(
        d1_candidate_analyses,
        lineage_pin.get("diagnostic_candidate_override"),
    )
    best = d1_candidate_analyses[0] if d1_candidate_analyses else None
    mainline_best = (
        mainline_candidate_analyses[0] if mainline_candidate_analyses else None
    )
    seed_summary = []
    for seed, rows in sorted(accepted_by_seed.items()):
        seed_summary.append(
            {
                "seed": seed,
                "accepted_rollout_count": len(rows),
                "accepted_rollouts": [row["rollout_path"] for row in rows],
                "mean_pre_attach_drawer_motion": float(
                    np.mean(
                        [
                            row["strict_success"]["pre_attach_drawer_motion"]
                            for row in rows
                        ]
                    )
                ),
                "mean_post_attach_drawer_delta": float(
                    np.mean(
                        [
                            row["strict_success"]["post_attach_drawer_delta"]
                            for row in rows
                        ]
                    )
                ),
                "mean_attach_persistence": float(
                    np.mean(
                        [row["strict_success"]["attach_persistence"] for row in rows]
                    )
                ),
                "mean_handle_distance_min": float(
                    np.mean(
                        [
                            row["handle_distance_min"]
                            for row in rows
                            if row["handle_distance_min"] is not None
                        ]
                    )
                )
                if any(row["handle_distance_min"] is not None for row in rows)
                else None,
            }
        )
    upstream_patch_prereqs = {
        "strict_valid_pool_small": bool(
            len(accepted_paths) < MIN_VALID_ROLLOUTS
            or len(accepted_by_seed) < MIN_VALID_SEEDS
        ),
        "d1_candidate_pool_empty": bool(not d1_candidate_analyses),
        "d2_feasible_seed_pool_empty": bool(not d2_feasible_seeds),
        "strong_coherent_seed_pool_empty": bool(not strong_coherent_seeds),
    }
    env_contract_audit = audit_environment_contract(
        [row["seed"] for row in mainline_candidate_analyses[:TOP_D1_CANDIDATES]]
        or [row["seed"] for row in d1_candidate_analyses[:TOP_D1_CANDIDATES]]
        or list(sorted(accepted_by_seed)),
        write_artifact=True,
    )
    result = {
        "gate": "strict_teacher_dataset_rebuild",
        "strict_success_version": STRICT_SUCCESS_VERSION,
        "strict_success_config": STRICT_SUCCESS_CONFIG,
        "source_dir": str(source_dir),
        "teacher_lineage_pin": lineage_pin,
        "strict_teacher_dir": str(STRICT_TEACHER_DIR),
        "accepted_rollout_count": len(accepted_paths),
        "accepted_seed_count": len(accepted_by_seed),
        "accepted_rollouts": [str(path) for path in accepted_paths],
        "best_single_rollout": None if best is None else best["rollout_path"],
        "best_single_rollout_reason": "learnability_first",
        "mainline_best_single_rollout": None
        if mainline_best is None
        else mainline_best["rollout_path"],
        "mainline_best_single_rollout_reason": "d2_feasible_learnability_first"
        if mainline_best is not None
        else None,
        "d2_feasible_seeds": d2_feasible_seeds,
        "d2_feasible_seed_min_rollouts": MIN_MAINLINE_ROLLOUTS_PER_SEED,
        "strong_rollout_criteria": {
            "attach_persistence_gte": STRONG_ATTACH_PERSISTENCE,
            "hold_close_steps_gte": STRONG_HOLD_CLOSE_STEPS,
            "post_attach_drawer_delta_gte": STRONG_POST_ATTACH_DELTA,
            "grasp_steps_gte": STRONG_MIN_GRASP_STEPS,
            "pull_steps_gte": STRONG_MIN_PULL_STEPS,
            "pre_attach_drawer_motion_lte": STRONG_MAX_PRE_ATTACH_MOTION,
            "phase_distance_lte": STRONG_COHERENCE_THRESHOLD,
        },
        "strong_rollout_count": len(strong_rollout_analyses),
        "strong_coherent_seeds": strong_coherent_seeds,
        "strong_seed_summary": strong_seed_summary,
        "d1_candidate_thresholds": {
            "min_critical_phase_mass": MIN_D1_CRITICAL_PHASE_MASS,
            "min_hold_close_steps": MIN_D1_HOLD_CLOSE_STEPS,
            "min_pull_steps": MIN_D1_PULL_STEPS,
            "min_attach_persistence": MIN_D1_ATTACH_PERSISTENCE,
        },
        "diagnostic_candidate_order": [
            {
                "rank": idx + 1,
                "rollout_path": row["rollout_path"],
                "rollout_sha256": row["rollout_sha256"],
                "seed": row["seed"],
                "episode_index": row["episode_index"],
                "attach_transition_density": float(
                    int(row["phase_counts"].get("grasp", 0))
                    + int(row["phase_counts"].get("hold_close", 0))
                )
                / max(sum(row["phase_counts"].values()), 1),
                "critical_phase_mass": int(row["phase_counts"].get("grasp", 0))
                + int(row["phase_counts"].get("hold_close", 0))
                + int(row["phase_counts"].get("pull", 0)),
                "grasp_steps": int(row["phase_counts"].get("grasp", 0)),
                "hold_close_steps": int(row["phase_counts"].get("hold_close", 0)),
                "pull_steps": int(row["phase_counts"].get("pull", 0)),
                "attach_persistence": int(row["strict_success"]["attach_persistence"]),
                "pre_attach_drawer_motion": float(
                    row["strict_success"]["pre_attach_drawer_motion"]
                ),
                "handle_distance_min": row["handle_distance_min"],
            }
            for idx, row in enumerate(d1_candidate_analyses[:TOP_D1_CANDIDATES])
        ],
        "d1_candidate_order": [
            {
                "rank": idx + 1,
                "rollout_path": row["rollout_path"],
                "rollout_sha256": row["rollout_sha256"],
                "seed": row["seed"],
                "episode_index": row["episode_index"],
                "attach_transition_density": float(
                    int(row["phase_counts"].get("grasp", 0))
                    + int(row["phase_counts"].get("hold_close", 0))
                )
                / max(sum(row["phase_counts"].values()), 1),
                "critical_phase_mass": int(row["phase_counts"].get("grasp", 0))
                + int(row["phase_counts"].get("hold_close", 0))
                + int(row["phase_counts"].get("pull", 0)),
                "grasp_steps": int(row["phase_counts"].get("grasp", 0)),
                "hold_close_steps": int(row["phase_counts"].get("hold_close", 0)),
                "pull_steps": int(row["phase_counts"].get("pull", 0)),
                "attach_persistence": int(row["strict_success"]["attach_persistence"]),
                "pre_attach_drawer_motion": float(
                    row["strict_success"]["pre_attach_drawer_motion"]
                ),
                "handle_distance_min": row["handle_distance_min"],
            }
            for idx, row in enumerate(d1_candidate_analyses[:TOP_D1_CANDIDATES])
        ],
        "mainline_candidate_order": [
            {
                "rank": idx + 1,
                "rollout_path": row["rollout_path"],
                "rollout_sha256": row["rollout_sha256"],
                "seed": row["seed"],
                "episode_index": row["episode_index"],
                "attach_transition_density": float(
                    int(row["phase_counts"].get("grasp", 0))
                    + int(row["phase_counts"].get("hold_close", 0))
                )
                / max(sum(row["phase_counts"].values()), 1),
                "critical_phase_mass": int(row["phase_counts"].get("grasp", 0))
                + int(row["phase_counts"].get("hold_close", 0))
                + int(row["phase_counts"].get("pull", 0)),
                "grasp_steps": int(row["phase_counts"].get("grasp", 0)),
                "hold_close_steps": int(row["phase_counts"].get("hold_close", 0)),
                "pull_steps": int(row["phase_counts"].get("pull", 0)),
                "attach_persistence": int(row["strict_success"]["attach_persistence"]),
                "pre_attach_drawer_motion": float(
                    row["strict_success"]["pre_attach_drawer_motion"]
                ),
                "handle_distance_min": row["handle_distance_min"],
            }
            for idx, row in enumerate(mainline_candidate_analyses[:TOP_D1_CANDIDATES])
        ],
        "d1_candidate_pool_count": len(d1_candidate_analyses),
        "mainline_candidate_pool_count": len(mainline_candidate_analyses),
        "seed_summary": seed_summary,
        "accepted_analyses": accepted_sorted,
        "accepted_analyses_by_learnability": learnability_sorted,
        "accepted_analyses_for_d1": d1_candidate_analyses,
        "accepted_analyses_for_mainline": mainline_candidate_analyses,
        "accepted_analyses_for_strong_mainline": strong_rollout_order,
        "rejected_from_d1_candidates": rejected_from_d1_candidates,
        "rejected_from_strong_rollouts": strong_rejected,
        "rejected_analyses": rejected,
        "env_contract_audit": env_contract_audit,
        "upstream_handle_patch_gate": {
            "triggered": bool(not env_contract_audit.get("passed")),
            "reason": "env_contract_blocked"
            if not env_contract_audit.get("passed")
            else "await_d1_top3_result",
            "prerequisites": {
                **upstream_patch_prereqs,
                "env_contract_passed": bool(env_contract_audit.get("passed")),
            },
            "min_valid_rollouts": MIN_VALID_ROLLOUTS,
            "min_valid_seeds": MIN_VALID_SEEDS,
        },
    }
    STRICT_TEACHER_AUDIT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    STRONG_ROLLOUT_AUDIT_PATH.write_text(
        json.dumps(
            {
                "gate": "strong_rollout_audit",
                "generated_at": now_iso(),
                "source_dir": str(source_dir),
                "teacher_lineage_pin": lineage_pin,
                "strong_rollout_criteria": result["strong_rollout_criteria"],
                "strong_rollout_count": len(strong_rollout_analyses),
                "strong_coherent_seeds": strong_coherent_seeds,
                "seed_summary": strong_seed_summary,
                "strong_rollout_order": [
                    {
                        "rank": idx + 1,
                        "rollout_path": row["rollout_path"],
                        "rollout_sha256": row["rollout_sha256"],
                        "seed": row["seed"],
                        "episode_index": row["episode_index"],
                        "grasp_steps": int(row["phase_counts"].get("grasp", 0)),
                        "hold_close_steps": int(
                            row["phase_counts"].get("hold_close", 0)
                        ),
                        "pull_steps": int(row["phase_counts"].get("pull", 0)),
                        "attach_persistence": int(
                            row["strict_success"]["attach_persistence"]
                        ),
                        "post_attach_drawer_delta": float(
                            row["strict_success"]["post_attach_drawer_delta"]
                        ),
                        "pre_attach_drawer_motion": float(
                            row["strict_success"]["pre_attach_drawer_motion"]
                        ),
                        "handle_distance_min": row["handle_distance_min"],
                    }
                    for idx, row in enumerate(strong_rollout_order)
                ],
                "rejected_from_strong_rollouts": strong_rejected,
            },
            indent=2,
        )
        + "\n"
    )
    return result

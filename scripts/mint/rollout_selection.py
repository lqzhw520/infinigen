#!/usr/bin/env python3
"""Shared rollout-selection and source-fingerprint helpers."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

PHASE_ORDER = ["staging", "pregrasp", "align", "grasp", "hold_close", "pull", "retreat"]
PHASE_DISTANCE_THRESHOLD = 0.45


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rollout_meta(path: Path) -> dict[str, Any]:
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def rollout_profile(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    meta = load_rollout_meta(path)
    phases_raw = data["phase_labels"].tolist() if "phase_labels" in data.files else []
    phases = []
    for value in phases_raw:
        if isinstance(value, bytes):
            phases.append(value.decode("utf-8", errors="ignore"))
        else:
            phases.append(str(value))
    total = max(1, len(phases))
    counts = {name: phases.count(name) for name in PHASE_ORDER}
    fractions = {name: counts[name] / total for name in PHASE_ORDER}
    return {
        "path": path,
        "seed": int(meta.get("seed", -1)),
        "episode_index": int(meta.get("episode_index", -1)),
        "max_drawer_fraction": float(meta.get("max_drawer_fraction", 0.0)),
        "grasp_score": float(meta.get("grasp_score", 0.0)),
        "attach_step": meta.get("attach_step"),
        "success": bool(meta.get("success")),
        "ever_attached": bool(meta.get("ever_attached")),
        "length": total,
        "phase_counts": counts,
        "phase_fractions": fractions,
        "branch_id": meta.get("branch_id") or meta.get("source_branch_id"),
        "contract_mode": meta.get("contract_mode"),
    }


def phase_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return float(
        sum(
            abs(left["phase_fractions"][name] - right["phase_fractions"][name])
            for name in PHASE_ORDER
        )
    )


def coherent_subset_stats(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    if not profiles:
        return {
            "rollout_count": 0,
            "mean_pairwise_phase_distance": None,
            "max_pairwise_phase_distance": None,
            "avg_max_drawer_fraction": 0.0,
            "avg_grasp_score": 0.0,
            "rollouts": [],
        }
    pairwise = [
        phase_distance(left, right) for left, right in combinations(profiles, 2)
    ]
    return {
        "rollout_count": len(profiles),
        "mean_pairwise_phase_distance": float(np.mean(pairwise)) if pairwise else 0.0,
        "max_pairwise_phase_distance": float(np.max(pairwise)) if pairwise else 0.0,
        "avg_max_drawer_fraction": float(
            np.mean([p["max_drawer_fraction"] for p in profiles])
        ),
        "avg_grasp_score": float(np.mean([p["grasp_score"] for p in profiles])),
        "rollouts": [
            {
                "path": str(p["path"]),
                "episode_index": p["episode_index"],
                "length": p["length"],
                "max_drawer_fraction": p["max_drawer_fraction"],
                "grasp_score": p["grasp_score"],
                "phase_fractions": p["phase_fractions"],
                "branch_id": p["branch_id"],
                "contract_mode": p["contract_mode"],
            }
            for p in profiles
        ],
    }


def group_rollouts_by_seed(rollout_paths: list[Path]) -> dict[int, list[Path]]:
    seed_to_paths: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(rollout_paths):
        meta = load_rollout_meta(path)
        seed_to_paths[int(meta["seed"])].append(path)
    return seed_to_paths


def select_seed_and_subset(
    seed_to_paths: dict[int, list[Path]],
    *,
    threshold: float = PHASE_DISTANCE_THRESHOLD,
) -> tuple[int, list[Path], dict[str, Any]]:
    selection_summary = {
        "strategy": "largest coherent same-seed rollout subset by pairwise phase-fraction distance",
        "phase_distance_threshold": threshold,
        "candidates": [],
    }
    best_choice = None
    for seed, paths in sorted(seed_to_paths.items()):
        profiles = [rollout_profile(path) for path in sorted(paths)]
        best_subset = None
        best_stats = None
        for subset_size in range(len(profiles), 0, -1):
            subset_candidates = []
            for combo in combinations(profiles, subset_size):
                stats = coherent_subset_stats(list(combo))
                max_dist = stats["max_pairwise_phase_distance"]
                if max_dist is None or max_dist <= threshold:
                    subset_candidates.append((stats, list(combo)))
            if subset_candidates:
                subset_candidates.sort(
                    key=lambda item: (
                        item[0]["rollout_count"],
                        -item[0]["mean_pairwise_phase_distance"],
                        item[0]["avg_max_drawer_fraction"],
                        item[0]["avg_grasp_score"],
                    ),
                    reverse=True,
                )
                best_stats, best_subset = subset_candidates[0]
                break
        if best_subset is None:
            best_subset = [profiles[0]]
            best_stats = coherent_subset_stats(best_subset)
        candidate = {
            "seed": seed,
            "all_rollout_count": len(paths),
            "selected_subset": best_stats,
        }
        selection_summary["candidates"].append(candidate)
        candidate_score = (
            best_stats["rollout_count"],
            -best_stats["mean_pairwise_phase_distance"],
            best_stats["avg_max_drawer_fraction"],
            best_stats["avg_grasp_score"],
            -seed,
        )
        if best_choice is None or candidate_score > best_choice[0]:
            best_choice = (
                candidate_score,
                seed,
                [p["path"] for p in best_subset],
                candidate,
            )
    assert best_choice is not None
    selected_seed = best_choice[1]
    selected_paths = [Path(p) for p in best_choice[2]]
    selection_summary["selected_seed"] = selected_seed
    selection_summary["selected_rollouts"] = [str(path) for path in selected_paths]
    selection_summary["selected_seed_stats"] = best_choice[3]
    return selected_seed, selected_paths, selection_summary


def build_coherent_multiseed_subset(
    rollout_paths: list[Path],
    *,
    min_rollouts_per_seed: int = 2,
    threshold: float = PHASE_DISTANCE_THRESHOLD,
) -> tuple[list[Path], dict[str, Any]]:
    seed_to_paths = group_rollouts_by_seed(rollout_paths)
    selected_paths: list[Path] = []
    included = []
    excluded = []
    for seed, paths in sorted(seed_to_paths.items()):
        _selected_seed, subset_paths, selection = select_seed_and_subset(
            {seed: paths}, threshold=threshold
        )
        stats = selection["selected_seed_stats"]["selected_subset"]
        record = {
            "seed": seed,
            "all_rollout_count": len(paths),
            "selected_rollout_count": len(subset_paths),
            "selected_rollouts": [str(path) for path in subset_paths],
            "selected_subset": stats,
        }
        if len(subset_paths) >= min_rollouts_per_seed:
            selected_paths.extend(subset_paths)
            included.append(record)
        else:
            excluded.append(record)
    selected_paths = sorted(selected_paths)
    summary = {
        "strategy": "per-seed coherent subset builder",
        "phase_distance_threshold": threshold,
        "min_rollouts_per_seed": min_rollouts_per_seed,
        "covered_seed_count": len(included),
        "included_seeds": [item["seed"] for item in included],
        "excluded_seeds": [item["seed"] for item in excluded],
        "included": included,
        "excluded": excluded,
        "selected_rollouts": [str(path) for path in selected_paths],
    }
    return selected_paths, summary


def source_fingerprint(
    rollout_paths: list[Path],
    *,
    contract_mode: str | None,
    source_branch_id: str | None,
    builder_version: str,
    variant_id: str,
    rollout_multiplier: int,
    balancing_mode: str,
    selected_seed_ids: list[int] | None = None,
) -> dict[str, Any]:
    rollouts = []
    for path in sorted(rollout_paths):
        meta = load_rollout_meta(path)
        rollouts.append(
            {
                "path": str(path),
                "size": path.stat().st_size if path.exists() else None,
                "content_sha256": file_sha256(path),
                "seed": meta.get("seed"),
                "episode_index": meta.get("episode_index"),
                "success": meta.get("success"),
                "ever_attached": meta.get("ever_attached"),
                "max_drawer_fraction": meta.get("max_drawer_fraction"),
                "branch_id": meta.get("branch_id") or meta.get("source_branch_id"),
                "contract_mode": meta.get("contract_mode"),
                "teacher_mode": meta.get("teacher_mode"),
                "uses_planned_segment": meta.get("uses_planned_segment"),
                "uses_target_fraction": meta.get("uses_target_fraction"),
                "attach_step": meta.get("attach_step"),
            }
        )
    payload = {
        "builder_version": builder_version,
        "variant_id": variant_id,
        "contract_mode": contract_mode,
        "source_branch_id": source_branch_id,
        "rollout_multiplier": rollout_multiplier,
        "balancing_mode": balancing_mode,
        "selected_seed_ids": sorted(selected_seed_ids or []),
        "rollouts": rollouts,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return payload


def save_fingerprint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_fingerprint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def fingerprint_matches(path: Path, expected: dict[str, Any]) -> bool:
    current = load_fingerprint(path)
    return bool(current and current.get("fingerprint") == expected.get("fingerprint"))

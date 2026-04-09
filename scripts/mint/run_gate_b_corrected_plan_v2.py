#!/usr/bin/env python3
"""
Gate B Corrected Plan v2
========================

Implements:
  Stage A - Gate B measurement cleanup
  Stage C - residual causal gap decomposition (entry-gated)

This script is proposal-only. It must not mutate canonical truth surfaces.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN_ROOT = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1"
ARTIFACT_ROOT = CAMPAIGN_ROOT / "artifacts"
SOVEREIGN = CAMPAIGN_ROOT / "sovereign"
PROPOSALS_DIR = SOVEREIGN / "proposals"
SCRIPTS_MINT = PROJECT_ROOT / "scripts" / "mint"

sys.path.insert(0, str(SCRIPTS_MINT))
sys.path.insert(0, str(PROJECT_ROOT))

from run_gate_b_teacher_replayability import (  # noqa: E402
    ORACLE_DEFINITION,
    TOP10_BY_EPISODE,
    V58_ROLLOUTS,
    P4_ROLLOUTS,
    _restore_state,
    classify_failure_stage,
    current_env_action_contract,
    current_env_handle_local,
    flatten_rows,
    git_head,
    mode_definition,
    now_ts,
    resolve_npz,
    run_oracle_reachability,
    run_replay_mode,
    selected_episodes,
    source_meta,
    summarize_mode,
    summarize_run_from_traces,
)


DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15)
STAGE_A_IMAGE_SIZE = 16
STAGE_C_IMAGE_SIZE = 224
STAGE_A_WORKERS = max(1, min(4, (os.cpu_count() or 1)))
STAGE_A_RENDER_OBSERVATIONS = False
SENTINEL_MODES = ("open_loop", "state_anchored_legacy", "oracle_reachability")
B2_MODES = (
    "open_loop",
    "state_anchored_legacy",
    "state_anchored_full_state_no_attach",
    "state_anchored_robot_only_no_attach",
)
DEFAULT_FINETUNED_POLICY = (
    ARTIFACT_ROOT / "v59_overfit_outputs" / "checkpoints" / "last" / "pretrained_model"
)

SAMPLE_SCOPE = {
    "sample": "Gate A action-quality top-10",
    "mapping_target": "success-selected legacy rollout subset",
    "population_claim": "No population-wide claim over full 240 episodes",
}


@dataclass
class StageAResult:
    sentinel: dict[str, Any]
    anchored: dict[str, Any]
    source_validation: dict[str, Any]
    corrected_draft: dict[str, Any]
    final_branch: str
    stage_c_entry_allowed: bool
    primary_threshold: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_builtin(payload), indent=2, ensure_ascii=False))


def save_yaml(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))


def save_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_builtin(value) for key, value in row.items()})


def threshold_key(value: float) -> str:
    return f"{float(value):.2f}"


def sensitivity_grade(delta_attach: float, delta_success: float) -> str:
    delta = max(abs(delta_attach), abs(delta_success))
    if delta > 0.20:
        return "high"
    if delta > 0.10:
        return "moderate"
    return "low"


def oracle_is_strong(success_rate: float, attach_rate: float) -> bool:
    return success_rate >= 0.90 and attach_rate >= 0.90


def _episode_source_stub(episode_index: int, npz_name: str, npz_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_index": int(episode_index),
        "npz_name": npz_name,
        "npz_path": str(npz_path),
        "seed": int(meta["seed"]),
        "source": {
            "success": bool(meta.get("success", False)),
            "ever_attached": bool(meta.get("ever_attached", False)),
            "final_drawer": float(meta.get("final_drawer_fraction", 0.0)),
            "max_drawer": float(meta.get("max_drawer_fraction", 0.0)),
        },
        "modes": {},
    }


def _run_episode_modes_task(
    episode_index: int,
    npz_name: str,
    modes: tuple[str, ...],
    max_steps: int,
    attach_threshold: float,
    image_size: int,
) -> dict[str, Any]:
    npz_path = resolve_npz(npz_name)
    source_npz = dict(np.load(npz_path, allow_pickle=True))
    meta = source_meta(npz_path)
    episode_summary = _episode_source_stub(episode_index, npz_name, npz_path, meta)

    for mode in modes:
        if mode == "oracle_reachability":
            result = run_oracle_reachability(
                episode_index=episode_index,
                npz_path=npz_path,
                source_npz=source_npz,
                source_meta_payload=meta,
                max_steps=max_steps,
                attach_threshold_override=attach_threshold,
                image_size=image_size,
                render_observations=STAGE_A_RENDER_OBSERVATIONS,
                capture_frames=False,
            )
        else:
            result = run_replay_mode(
                episode_index=episode_index,
                npz_path=npz_path,
                source_npz=source_npz,
                source_meta_payload=meta,
                replay_mode=mode,
                max_steps=max_steps,
                attach_threshold_override=attach_threshold,
                image_size=image_size,
                render_observations=STAGE_A_RENDER_OBSERVATIONS,
                capture_frames=False,
                capture_trace=False,
            )
        episode_summary["modes"][mode] = {
            key: value
            for key, value in result.items()
            if key not in {"video_frames", "trace_steps"}
        }
    rows = flatten_rows([episode_summary])
    for row in rows:
        row["attach_threshold_sweep_m"] = float(attach_threshold)
    return {
        "episode_summary": episode_summary,
        "rows": rows,
    }


def run_mode_matrix(
    *,
    episode_set: list[tuple[int, str]],
    modes: list[str],
    max_steps: int,
    attach_threshold: float,
    image_size: int,
    episodes_dir: Path | None = None,
) -> dict[str, Any]:
    episode_summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=STAGE_A_WORKERS) as executor:
        futures = {
            executor.submit(
                _run_episode_modes_task,
                episode_index,
                npz_name,
                tuple(modes),
                max_steps,
                attach_threshold,
                image_size,
            ): (episode_index, npz_name)
            for episode_index, npz_name in episode_set
        }
        for future in concurrent.futures.as_completed(futures):
            payload = future.result()
            episode_summary = payload["episode_summary"]
            episode_summaries.append(episode_summary)
            rows.extend(payload["rows"])
            if episodes_dir is not None:
                save_json(
                    episodes_dir / f"thr_{threshold_key(attach_threshold).replace('.', '')}_ep{episode_summary['episode_index']:03d}.json",
                    episode_summary,
                )

    episode_summaries.sort(key=lambda item: int(item["episode_index"]))
    rows.sort(key=lambda row: (float(row["attach_threshold_sweep_m"]), int(row["episode_index"]), str(row["replay_mode"])))
    mode_results = {
        mode: summarize_mode(mode, [row for row in rows if row["replay_mode"] == mode])
        for mode in modes
    }
    return {
        "episodes": episode_summaries,
        "rows": rows,
        "mode_results": mode_results,
    }


def _mode_attach_success(mode_results: dict[str, Any], mode: str) -> tuple[float, float]:
    metrics = mode_results.get(mode, {})
    return float(metrics.get("attach_rate", 0.0)), float(metrics.get("success_rate", 0.0))


def _monotonic_episode_order(episode: dict[str, Any]) -> bool:
    expected = [
        "open_loop",
        "state_anchored_robot_only_no_attach",
        "state_anchored_full_state_no_attach",
        "state_anchored_legacy",
    ]
    available = [mode for mode in expected if mode in episode.get("modes", {})]
    if len(available) < 2:
        return True
    tuples = []
    for mode in available:
        metrics = episode["modes"][mode]
        tuples.append(
            (
                int(bool(metrics.get("success"))),
                int(bool(metrics.get("ever_attached"))),
                float(metrics.get("max_drawer_fraction", 0.0)),
            )
        )
    return tuples == sorted(tuples)


def _pack_success_paths() -> list[Path]:
    paths: list[Path] = []
    for root in (V58_ROLLOUTS, P4_ROLLOUTS):
        if not root.exists():
            continue
        for npz_path in sorted(root.glob("*.npz")):
            try:
                with np.load(npz_path, allow_pickle=True) as payload:
                    success = payload["success"] if "success" in payload.files else None
                    if bool(success):
                        paths.append(npz_path)
            except Exception:
                continue
    return paths


def build_b1_sentinel(
    *,
    episode_set: list[tuple[int, str]],
    max_steps: int,
    output_dir: Path,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    episodes_dir = output_dir / "episodes"
    ensure_dir(episodes_dir)

    threshold_runs: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    for threshold in DEFAULT_THRESHOLDS:
        key = threshold_key(threshold)
        matrix = run_mode_matrix(
            episode_set=episode_set,
            modes=list(SENTINEL_MODES),
            max_steps=max_steps,
            attach_threshold=threshold,
            image_size=STAGE_A_IMAGE_SIZE,
            episodes_dir=episodes_dir,
        )
        threshold_runs[key] = {
            "attach_threshold_m": float(threshold),
            "mode_results": matrix["mode_results"],
            "episodes": matrix["episodes"],
        }
        all_rows.extend(matrix["rows"])

    oracle_attach_005, oracle_success_005 = _mode_attach_success(
        threshold_runs["0.05"]["mode_results"],
        "oracle_reachability",
    )
    oracle_attach_015, oracle_success_015 = _mode_attach_success(
        threshold_runs["0.15"]["mode_results"],
        "oracle_reachability",
    )
    open_attach_005, open_success_005 = _mode_attach_success(
        threshold_runs["0.05"]["mode_results"],
        "open_loop",
    )
    open_attach_015, open_success_015 = _mode_attach_success(
        threshold_runs["0.15"]["mode_results"],
        "open_loop",
    )
    oracle_grade = sensitivity_grade(
        oracle_attach_015 - oracle_attach_005,
        oracle_success_015 - oracle_success_005,
    )
    open_grade = sensitivity_grade(
        open_attach_015 - open_attach_005,
        open_success_015 - open_success_005,
    )
    oracle_success_gap_by_threshold = {}
    oracle_attach_gap_by_threshold = {}
    for key, payload in threshold_runs.items():
        oracle_attach, oracle_success = _mode_attach_success(
            payload["mode_results"],
            "oracle_reachability",
        )
        open_attach, open_success = _mode_attach_success(payload["mode_results"], "open_loop")
        oracle_success_gap_by_threshold[key] = float(oracle_success - open_success)
        oracle_attach_gap_by_threshold[key] = float(oracle_attach - open_attach)

    oracle_strong_005 = oracle_is_strong(oracle_success_005, oracle_attach_005)
    primary_threshold = 0.05 if oracle_strong_005 and oracle_grade != "high" else 0.15
    dual_threshold_required = not (oracle_strong_005 and oracle_grade != "high")
    science_thresholds = [0.05] if not dual_threshold_required else [0.05, 0.15]

    artifact = {
        "experiment": "gate_b1_threshold_sentinel",
        "timestamp": now_ts(),
        "sample_scope": dict(SAMPLE_SCOPE),
        "measurement_image_size_px": STAGE_A_IMAGE_SIZE,
        "measurement_render_observations": STAGE_A_RENDER_OBSERVATIONS,
        "oracle_definition": dict(ORACLE_DEFINITION),
        "thresholds": threshold_runs,
        "aggregate": {
            "oracle_attach_rate_delta_015_vs_005": float(oracle_attach_015 - oracle_attach_005),
            "oracle_success_rate_delta_015_vs_005": float(oracle_success_015 - oracle_success_005),
            "open_loop_attach_rate_delta_015_vs_005": float(open_attach_015 - open_attach_005),
            "open_loop_success_rate_delta_015_vs_005": float(open_success_015 - open_success_005),
            "oracle_threshold_sensitivity_grade": oracle_grade,
            "open_loop_threshold_sensitivity_grade": open_grade,
            "oracle_vs_open_loop_success_gap": oracle_success_gap_by_threshold,
            "oracle_vs_open_loop_attach_gap": oracle_attach_gap_by_threshold,
            "oracle_canonical_threshold_status": "strong" if oracle_strong_005 else "collapsed",
            "primary_threshold_m": float(primary_threshold),
            "science_thresholds_m": science_thresholds,
            "dual_threshold_required": dual_threshold_required,
            "large_replay_gap": bool(oracle_success_gap_by_threshold["0.05"] > 0.30),
            "small_replay_gap": bool(oracle_success_gap_by_threshold["0.05"] <= 0.10),
        },
    }

    save_json(output_dir / "gate_b1_sentinel_summary.json", artifact)
    save_csv(output_dir / "gate_b1_sentinel_episodes.csv", all_rows)
    save_text(output_dir / "gate_b1_sentinel_summary.md", render_b1_markdown(artifact))
    return artifact


def build_b2_anchored(
    *,
    episode_set: list[tuple[int, str]],
    max_steps: int,
    output_dir: Path,
    thresholds: list[float],
) -> dict[str, Any]:
    ensure_dir(output_dir)
    episodes_dir = output_dir / "episodes"
    ensure_dir(episodes_dir)

    threshold_runs: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        key = threshold_key(threshold)
        matrix = run_mode_matrix(
            episode_set=episode_set,
            modes=list(B2_MODES),
            max_steps=max_steps,
            attach_threshold=threshold,
            image_size=STAGE_A_IMAGE_SIZE,
            episodes_dir=episodes_dir,
        )
        attachment_delta_attach = abs(
            float(matrix["mode_results"]["state_anchored_legacy"]["attach_rate"])
            - float(matrix["mode_results"]["state_anchored_full_state_no_attach"]["attach_rate"])
        )
        attachment_delta_success = abs(
            float(matrix["mode_results"]["state_anchored_legacy"]["success_rate"])
            - float(matrix["mode_results"]["state_anchored_full_state_no_attach"]["success_rate"])
        )
        drawer_delta_attach = abs(
            float(matrix["mode_results"]["state_anchored_full_state_no_attach"]["attach_rate"])
            - float(matrix["mode_results"]["state_anchored_robot_only_no_attach"]["attach_rate"])
        )
        drawer_delta_success = abs(
            float(matrix["mode_results"]["state_anchored_full_state_no_attach"]["success_rate"])
            - float(matrix["mode_results"]["state_anchored_robot_only_no_attach"]["success_rate"])
        )
        threshold_runs[key] = {
            "attach_threshold_m": float(threshold),
            "mode_results": matrix["mode_results"],
            "episodes": matrix["episodes"],
            "attachment_state_contamination_delta": {
                "attach_rate_delta": attachment_delta_attach,
                "success_rate_delta": attachment_delta_success,
                "material": bool(max(attachment_delta_attach, attachment_delta_success) > 0.10),
            },
            "drawer_progress_contamination_delta": {
                "attach_rate_delta": drawer_delta_attach,
                "success_rate_delta": drawer_delta_success,
                "material": bool(max(drawer_delta_attach, drawer_delta_success) > 0.10),
            },
            "anchored_mode_reordering_count": int(
                sum(1 for episode in matrix["episodes"] if not _monotonic_episode_order(episode))
            ),
        }
        all_rows.extend(matrix["rows"])

    overall_material = any(
        payload["attachment_state_contamination_delta"]["material"]
        or payload["drawer_progress_contamination_delta"]["material"]
        for payload in threshold_runs.values()
    )
    artifact = {
        "experiment": "gate_b2_anchored_decomposition",
        "timestamp": now_ts(),
        "sample_scope": dict(SAMPLE_SCOPE),
        "measurement_image_size_px": STAGE_A_IMAGE_SIZE,
        "measurement_render_observations": STAGE_A_RENDER_OBSERVATIONS,
        "thresholds": threshold_runs,
        "aggregate": {
            "contamination_material": overall_material,
            "science_thresholds_m": thresholds,
        },
    }
    save_json(output_dir / "gate_b2_summary.json", artifact)
    save_csv(output_dir / "gate_b2_episodes.csv", all_rows)
    save_text(output_dir / "gate_b2_summary.md", render_b2_markdown(artifact))
    return artifact


def build_b3_source_validation(
    *,
    episode_set: list[tuple[int, str]],
    output_dir: Path,
    success_episode_set: set[int],
) -> dict[str, Any]:
    ensure_dir(output_dir)
    paths = _pack_success_paths()
    episodes = []
    tier_counts = Counter()
    tier_x_success = 0
    tier_x_failure = 0

    for episode_index, current_npz_name in episode_set:
        reconstructed_path = paths[episode_index] if episode_index < len(paths) else None
        mapping_match = bool(reconstructed_path and reconstructed_path.name == current_npz_name)
        current_path = resolve_npz(current_npz_name)
        trace_success = None
        metadata_success = None
        metadata_vs_trace_match = None
        source_native_available = False
        if reconstructed_path is not None:
            with np.load(reconstructed_path, allow_pickle=True) as payload:
                drawer = np.asarray(payload["absolute_drawer_fraction"], dtype=np.float32)
                attached = np.asarray(payload["attached_trace"], dtype=bool)
                trace_success = bool(
                    __import__("strict_success").evaluate_strict_success(drawer, attached)["strict_success"]
                )
                metadata_raw = payload["success"] if "success" in payload.files else None
                metadata_success = bool(metadata_raw) if metadata_raw is not None else None
                metadata_vs_trace_match = (
                    None if metadata_success is None else bool(metadata_success == trace_success)
                )
        tier = "Tier A" if mapping_match and bool(trace_success) else ("Tier B" if mapping_match else "Tier X")
        tier_counts[tier] += 1
        if tier == "Tier X":
            if episode_index in success_episode_set:
                tier_x_success += 1
            else:
                tier_x_failure += 1
        episodes.append(
            {
                "dataset_episode_index": int(episode_index),
                "current_gate_b_npz_name": current_npz_name,
                "current_gate_b_npz_path": str(current_path),
                "reconstructed_npz_name": None if reconstructed_path is None else reconstructed_path.name,
                "reconstructed_npz_path": None if reconstructed_path is None else str(reconstructed_path),
                "mapping_match": mapping_match,
                "metadata_success": metadata_success,
                "trace_recomputed_strict_success": trace_success,
                "metadata_vs_trace_match": metadata_vs_trace_match,
                "source_native_available": source_native_available,
                "tier": tier,
            }
        )

    artifact = {
        "experiment": "gate_b3_source_validation",
        "timestamp": now_ts(),
        "sample_scope": dict(SAMPLE_SCOPE),
        "mapping_rule": {
            "source": "run_v58_lerobot_pack.py deterministic packer order",
            "order": [
                "sorted successful v58_physics_legal_rollouts",
                "then sorted successful p4_physics_legal_rollouts",
            ],
        },
        "episodes": episodes,
        "aggregate": {
            "tier_counts": dict(tier_counts),
            "tier_x_among_success_episodes": int(tier_x_success),
            "tier_x_among_failure_episodes": int(tier_x_failure),
            "all_tier_a": bool(tier_counts["Tier A"] == len(episode_set)),
        },
    }
    save_json(output_dir / "gate_b3_summary.json", artifact)
    save_csv(output_dir / "gate_b3_episodes.csv", episodes)
    save_text(output_dir / "gate_b3_summary.md", render_b3_markdown(artifact))
    return artifact


def load_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    with open(path) as handle:
        return json.load(handle)


def build_e029(
    *,
    sentinel: dict[str, Any],
    anchored: dict[str, Any],
    source_validation: dict[str, Any],
) -> tuple[dict[str, Any], str, bool]:
    baseline_path = ARTIFACT_ROOT / "gate_b_teacher_replayability" / "gate_b_replay_summary.json"
    baseline = load_json(baseline_path, default={}) or {}
    agg = sentinel["aggregate"]
    oracle_collapsed = (
        agg["oracle_canonical_threshold_status"] == "collapsed"
        or agg["oracle_threshold_sensitivity_grade"] == "high"
    )
    replay_gap_large = bool(agg["oracle_vs_open_loop_success_gap"]["0.05"] > 0.30)
    contamination_material = bool(anchored["aggregate"]["contamination_material"])
    any_tier_x = bool(source_validation["aggregate"]["tier_counts"].get("Tier X", 0))

    if oracle_collapsed:
        final_branch = "relaxed-semantics reachability only"
        stage_c_entry_allowed = False
    elif replay_gap_large:
        final_branch = "large replay-gap remains between env-native reachability and teacher-action execution"
        stage_c_entry_allowed = False
    elif contamination_material:
        final_branch = "mixed executability/robustness problem under contaminated anchored replay"
        stage_c_entry_allowed = True
    elif any_tier_x:
        final_branch = "subset-level executability support only"
        stage_c_entry_allowed = True
    else:
        final_branch = "teacher targets are substantially executable under audited current-env semantics"
        stage_c_entry_allowed = True

    draft = {
        "draft_id": "E029_draft",
        "draft_type": "evidence_draft",
        "target_evidence_id": "E029",
        "experiment_id": "gate_b_corrected_deconfounded_followup",
        "type": "learnability_audit",
        "scope": "gate_b_corrected_top10",
        "scope_guardrail": (
            "E029 is a corrected Gate B follow-up draft. It is proposal-only, limited to the "
            "Gate A top-10 sample, and does not directly determine final policy learnability."
        ),
        "timestamp": now_ts(),
        "draft_status": "review_required",
        "verified": False,
        "sample_scope": dict(SAMPLE_SCOPE),
        "measurement_runtime_note": (
            "Stage A executability follow-ups were collected with reduced observation resolution "
            f"({STAGE_A_IMAGE_SIZE}px) and render-observations={STAGE_A_RENDER_OBSERVATIONS} "
            "for measurement throughput. This does not change physics, task semantics, or action "
            "contracts."
        ),
        "source_artifacts": {
            "baseline_gate_b": str(baseline_path.relative_to(CAMPAIGN_ROOT)),
            "b1_sentinel": "artifacts/gate_b1_threshold_sentinel/gate_b1_sentinel_summary.json",
            "b2_anchored": "artifacts/gate_b2_anchored_decomposition/gate_b2_summary.json",
            "b3_source_validation": "artifacts/gate_b3_source_validation/gate_b3_summary.json",
            "baseline_e028": "sovereign/proposals/E028_gate_b_teacher_executability_draft.yaml",
        },
        "summary": (
            "Corrected Gate B follow-up freezes the optimistic baseline and re-evaluates it "
            "under canonical-threshold reachability, anchored contamination decomposition, "
            "and deterministic source-validation."
        ),
        "baseline_snapshot": {
            "decision_support": baseline.get("aggregate", {}).get("decision_support"),
            "open_loop_success_rate": baseline.get("aggregate", {}).get("open_loop_success_rate"),
            "state_anchored_success_rate": baseline.get("aggregate", {}).get("state_anchored_success_rate"),
            "oracle_reachability_rate": baseline.get("aggregate", {}).get("oracle_reachability_rate"),
        },
        "sentinel_findings": sentinel["aggregate"],
        "anchored_findings": anchored["aggregate"],
        "source_validation_findings": source_validation["aggregate"],
        "oracle_definition": dict(ORACLE_DEFINITION),
        "final_branch": final_branch,
        "stage_c_entry_allowed": stage_c_entry_allowed,
        "allowed_wording": [
            "Gate B weakens the strong hypothesis that teacher targets are globally unexecutable in the current env.",
            "The remaining gap is best described as a mixed executability/robustness problem unless the corrected follow-up fully clears the canonical threshold, contamination, and mapping gates.",
        ],
        "disallowed_wording": [
            "policy-only gap proven",
            "source-grasp fidelity proven",
            "image-domain eliminated",
            "root cause solved",
        ],
        "do_not": [
            "Do not promote this draft directly into sovereign/evidence without daylight review.",
            "Do not treat env-native oracle reachability as proof of teacher trajectory fidelity.",
            "Do not write policy-only narrative if Stage C entry is blocked.",
        ],
    }
    return draft, final_branch, stage_c_entry_allowed


def _phase_bucket(step_index: int, metrics: dict[str, Any]) -> str:
    first_near = metrics.get("first_near_handle_step")
    attach_step = metrics.get("attach_step")
    first_major = metrics.get("first_major_open_step")
    if first_near is None or step_index < int(first_near):
        return "pre-near-handle"
    if attach_step is None or step_index < int(attach_step):
        return "near-handle-to-attach"
    if first_major is None or step_index < int(first_major):
        return "attached-to-major-open"
    return "post-major-open"


def _load_policy_bundle(policy_path: Path):
    from evaluate_mint_drawer_campaign import _load_policy
    from mint_common import DATASET_DIR, DATASET_REPO_ID

    return _load_policy(str(policy_path), DATASET_DIR, DATASET_REPO_ID)


def _predict_policy_action(policy_bundle, *, image: np.ndarray, image2: np.ndarray, state: np.ndarray, task: Any, mode: str = "full_inputs") -> np.ndarray:
    import torch

    policy, pre, post = policy_bundle
    image_arr = image.copy()
    image2_arr = image2.copy()
    state_arr = state.astype(np.float32).copy()
    if mode == "images_zeroed_state_preserved":
        image_arr[...] = 0
        image2_arr[...] = 0
    elif mode == "state_zeroed_images_preserved":
        state_arr[...] = 0
    batch = {
        "observation.images.image": torch.from_numpy(image_arr).permute(2, 0, 1).to(torch.float32) / 255.0,
        "observation.images.image2": torch.from_numpy(image2_arr).permute(2, 0, 1).to(torch.float32) / 255.0,
        "observation.state": torch.from_numpy(state_arr),
        "task": task,
    }
    processed = pre(batch)
    with torch.inference_mode():
        action = policy.select_action(processed)
    action = post(action)
    return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _validated_subset(
    *,
    sentinel: dict[str, Any],
    anchored: dict[str, Any],
    source_validation: dict[str, Any],
) -> list[dict[str, Any]]:
    primary_key = threshold_key(sentinel["aggregate"]["primary_threshold_m"])
    sentinel_eps = {
        ep["episode_index"]: ep for ep in sentinel["thresholds"][primary_key]["episodes"]
    }
    b2_eps = {
        ep["episode_index"]: ep
        for ep in anchored["thresholds"][primary_key]["episodes"]
    }
    tiers = {ep["dataset_episode_index"]: ep["tier"] for ep in source_validation["episodes"]}
    selected = []
    for episode_index, sentinel_ep in sentinel_eps.items():
        if tiers.get(episode_index) == "Tier X":
            continue
        trajectory_mode = None
        if sentinel_ep["modes"]["open_loop"]["success"]:
            trajectory_mode = "open_loop"
        elif b2_eps[episode_index]["modes"]["state_anchored_robot_only_no_attach"]["success"]:
            trajectory_mode = "state_anchored_robot_only_no_attach"
        if trajectory_mode:
            selected.append(
                {
                    "episode_index": episode_index,
                    "npz_name": sentinel_ep["npz_name"],
                    "trajectory_mode": trajectory_mode,
                }
            )
    return selected


def build_c1_divergence(
    *,
    subset: list[dict[str, Any]],
    policy_path: Path,
    attach_threshold: float,
    max_steps: int,
    output_dir: Path,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    if not subset:
        artifact = {
            "experiment": "gate_c1_policy_teacher_divergence",
            "timestamp": now_ts(),
            "status": "skipped",
            "reason": "No validated subset episodes available.",
        }
        save_json(output_dir / "gate_c1_summary.json", artifact)
        save_text(output_dir / "gate_c1_summary.md", "# Gate C1\n\nNo validated subset episodes were available.\n")
        save_csv(output_dir / "gate_c1_steps.csv", [])
        return artifact

    policy_bundle = _load_policy_bundle(policy_path)
    step_rows: list[dict[str, Any]] = []
    per_episode = []

    for item in subset:
        episode_index = int(item["episode_index"])
        npz_path = resolve_npz(item["npz_name"])
        source_npz = dict(np.load(npz_path, allow_pickle=True))
        meta = source_meta(npz_path)
        trace_result = run_replay_mode(
            episode_index=episode_index,
            npz_path=npz_path,
            source_npz=source_npz,
            source_meta_payload=meta,
            replay_mode=item["trajectory_mode"],
            max_steps=max_steps,
            attach_threshold_override=attach_threshold,
            image_size=STAGE_C_IMAGE_SIZE,
            capture_frames=False,
            capture_trace=True,
        )
        trace_steps = trace_result.get("trace_steps", [])
        bucket_errors = defaultdict(list)
        for step in trace_steps:
            teacher_action = np.asarray(step["teacher_action"], dtype=np.float32)
            predicted = _predict_policy_action(
                policy_bundle,
                image=np.asarray(step["obs_image"], dtype=np.uint8),
                image2=np.asarray(step["obs_image2"], dtype=np.uint8),
                state=np.asarray(step["obs_state"], dtype=np.float32),
                task=step.get("obs_task"),
                mode="full_inputs",
            )
            bucket = _phase_bucket(int(step["step_index"]), trace_result)
            l2 = float(np.linalg.norm(predicted - teacher_action))
            transl = float(np.linalg.norm(predicted[:3] - teacher_action[:3]))
            rot = float(np.linalg.norm(predicted[3:6] - teacher_action[3:6]))
            grip_agree = int(np.sign(predicted[6]) == np.sign(teacher_action[6]))
            bucket_errors[bucket].append(l2)
            step_rows.append(
                {
                    "episode_index": episode_index,
                    "trajectory_mode": item["trajectory_mode"],
                    "step_index": int(step["step_index"]),
                    "phase_bucket": bucket,
                    "action_l2_error": l2,
                    "translation_error_norm": transl,
                    "rotation_error_norm": rot,
                    "gripper_sign_agree": grip_agree,
                }
            )
        per_episode.append(
            {
                "episode_index": episode_index,
                "trajectory_mode": item["trajectory_mode"],
                "phase_bucket_mean_l2": {
                    bucket: float(np.mean(values)) for bucket, values in bucket_errors.items()
                },
            }
        )

    phase_bucket_summary = {}
    for bucket in [
        "pre-near-handle",
        "near-handle-to-attach",
        "attached-to-major-open",
        "post-major-open",
    ]:
        values = [row["action_l2_error"] for row in step_rows if row["phase_bucket"] == bucket]
        phase_bucket_summary[bucket] = None if not values else float(np.mean(values))

    artifact = {
        "experiment": "gate_c1_policy_teacher_divergence",
        "timestamp": now_ts(),
        "policy_path": str(policy_path),
        "attach_threshold_m": float(attach_threshold),
        "subset_size": len(subset),
        "phase_bucket_summary": phase_bucket_summary,
        "episodes": per_episode,
    }
    save_json(output_dir / "gate_c1_summary.json", artifact)
    save_csv(output_dir / "gate_c1_steps.csv", step_rows)
    save_text(output_dir / "gate_c1_summary.md", render_c1_markdown(artifact))
    return artifact


def _run_switch_rollout(
    *,
    episode_index: int,
    npz_name: str,
    replay_mode: str,
    switch_condition: str,
    trigger_step: int | None,
    policy_bundle,
    attach_threshold: float,
    max_steps: int,
) -> dict[str, Any]:
    npz_path = resolve_npz(npz_name)
    source_npz = dict(np.load(npz_path, allow_pickle=True))
    meta = source_meta(npz_path)
    mode_cfg = mode_definition(replay_mode)
    action_contract = current_env_action_contract(
        meta,
        attach_threshold_override=attach_threshold,
    )

    from drawer_robot_env import DrawerRobotEnv

    env = DrawerRobotEnv(
        seed=int(meta["seed"]),
        image_size=STAGE_C_IMAGE_SIZE,
        max_steps=max_steps,
        action_contract=action_contract,
    )
    actions = source_npz["actions"].astype(np.float32)
    eef_positions: list[np.ndarray] = []
    drawer_trace: list[float] = []
    attached_trace: list[bool] = []
    handle_distance_trace: list[float] = []
    actor_trace: list[str] = []
    success = False
    attach_step = None

    try:
        obs = env.reset()
        local_handle, handle_metadata = current_env_handle_local(env)
        env.attachment_local = local_handle
        for step_index in range(max_steps):
            if mode_cfg["restore_robot_state"] and step_index < len(actions):
                _restore_state(
                    env,
                    source_npz,
                    step_index,
                    restore_drawer_fraction=bool(mode_cfg["restore_drawer_fraction"]),
                    restore_source_attachment_state=bool(
                        mode_cfg["restore_source_attachment_state"]
                    ),
                    restore_source_runtime_histories=bool(
                        mode_cfg["restore_source_runtime_histories"]
                    ),
                    pre_step_force_detached=bool(mode_cfg["pre_step_force_detached"]),
                )
            use_teacher = False
            if switch_condition == "teacher_all":
                use_teacher = True
            elif switch_condition == "mint_all":
                use_teacher = False
            elif switch_condition in {"teacher_until_first_attach_then_mint", "teacher_until_first_major_open_then_mint"}:
                use_teacher = trigger_step is not None and step_index <= int(trigger_step)
            if use_teacher:
                action = actions[step_index] if step_index < len(actions) else np.zeros((7,), dtype=np.float32)
                actor = "teacher"
            else:
                action = _predict_policy_action(
                    policy_bundle,
                    image=obs.image,
                    image2=obs.image2,
                    state=obs.state,
                    task=obs.task,
                    mode="full_inputs",
                )
                actor = "mint"
            obs, _reward, done, info = env.step(action)
            handle_pos, _ = env.local_to_world_pose(
                local_handle["pos"],
                local_handle["quat"],
                fraction=obs.drawer_fraction,
            )
            eef_positions.append(obs.eef_pos.copy())
            drawer_trace.append(float(obs.drawer_fraction))
            attached_trace.append(bool(info.get("attached")))
            handle_distance_trace.append(float(np.linalg.norm(obs.eef_pos - handle_pos)))
            actor_trace.append(actor)
            if info.get("attached") and attach_step is None:
                attach_step = step_index
            success = success or bool(info.get("is_success", False))
            if done:
                break
    finally:
        env.close()

    summary = summarize_run_from_traces(
        episode_index=episode_index,
        npz_path=npz_path,
        source_npz=source_npz,
        source_meta_payload=meta,
        replay_mode=switch_condition,
        eef_positions=np.asarray(eef_positions, dtype=np.float32),
        drawer_trace=np.asarray(drawer_trace, dtype=np.float32),
        attached_trace=np.asarray(attached_trace, dtype=np.bool_),
        handle_distance_trace=np.asarray(handle_distance_trace, dtype=np.float32),
        success=success,
        attach_step=attach_step,
        action_contract=action_contract,
        frames=[],
    )
    summary["actor_trace"] = actor_trace
    return summary


def build_c2_hybrid(
    *,
    subset: list[dict[str, Any]],
    policy_path: Path,
    attach_threshold: float,
    max_steps: int,
    output_dir: Path,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    if not subset:
        artifact = {
            "experiment": "gate_c2_stage_sliced_hybrid",
            "timestamp": now_ts(),
            "status": "skipped",
            "reason": "No validated subset episodes available.",
        }
        save_json(output_dir / "gate_c2_summary.json", artifact)
        save_text(output_dir / "gate_c2_summary.md", "# Gate C2\n\nNo validated subset episodes were available.\n")
        save_csv(output_dir / "gate_c2_conditions.csv", [])
        return artifact

    policy_bundle = _load_policy_bundle(policy_path)
    condition_rows = []
    per_episode = []
    for item in subset:
        episode_index = int(item["episode_index"])
        npz_path = resolve_npz(item["npz_name"])
        source_npz = dict(np.load(npz_path, allow_pickle=True))
        meta = source_meta(npz_path)
        teacher_trace = run_replay_mode(
            episode_index=episode_index,
            npz_path=npz_path,
            source_npz=source_npz,
            source_meta_payload=meta,
            replay_mode=item["trajectory_mode"],
            max_steps=max_steps,
            attach_threshold_override=attach_threshold,
            image_size=STAGE_C_IMAGE_SIZE,
            capture_frames=False,
            capture_trace=False,
        )
        conditions = {}
        for condition in (
            "teacher_all",
            "mint_all",
            "teacher_until_first_attach_then_mint",
            "teacher_until_first_major_open_then_mint",
        ):
            trigger_step = None
            if condition == "teacher_until_first_attach_then_mint":
                trigger_step = teacher_trace.get("attach_step")
            elif condition == "teacher_until_first_major_open_then_mint":
                trigger_step = teacher_trace.get("first_major_open_step")
            if condition == "teacher_all":
                result = teacher_trace
                result = dict(result)
                result["replay_mode"] = condition
            else:
                result = _run_switch_rollout(
                    episode_index=episode_index,
                    npz_name=item["npz_name"],
                    replay_mode=item["trajectory_mode"],
                    switch_condition=condition,
                    trigger_step=trigger_step,
                    policy_bundle=policy_bundle,
                    attach_threshold=attach_threshold,
                    max_steps=max_steps,
                )
            conditions[condition] = {
                "success": bool(result.get("success")),
                "ever_attached": bool(result.get("ever_attached")),
                "failure_stage": result.get("failure_stage"),
                "max_drawer_fraction": float(result.get("max_drawer_fraction", 0.0)),
            }
            condition_rows.append(
                {
                    "episode_index": episode_index,
                    "trajectory_mode": item["trajectory_mode"],
                    "condition": condition,
                    **conditions[condition],
                }
            )
        per_episode.append({"episode_index": episode_index, "conditions": conditions})

    aggregate = {}
    for condition in (
        "teacher_all",
        "mint_all",
        "teacher_until_first_attach_then_mint",
        "teacher_until_first_major_open_then_mint",
    ):
        rows = [row for row in condition_rows if row["condition"] == condition]
        aggregate[condition] = {
            "success_rate": float(sum(1 for row in rows if row["success"]) / max(len(rows), 1)),
            "attach_rate": float(sum(1 for row in rows if row["ever_attached"]) / max(len(rows), 1)),
            "dominant_failure_stage": (
                Counter(row["failure_stage"] for row in rows).most_common(1)[0][0]
                if rows
                else "invalid_run"
            ),
        }

    artifact = {
        "experiment": "gate_c2_stage_sliced_hybrid",
        "timestamp": now_ts(),
        "policy_path": str(policy_path),
        "attach_threshold_m": float(attach_threshold),
        "subset_size": len(subset),
        "aggregate": aggregate,
        "episodes": per_episode,
    }
    save_json(output_dir / "gate_c2_summary.json", artifact)
    save_csv(output_dir / "gate_c2_conditions.csv", condition_rows)
    save_text(output_dir / "gate_c2_summary.md", render_c2_markdown(artifact))
    return artifact


def build_c3_modality_sensitivity(
    *,
    subset: list[dict[str, Any]],
    policy_path: Path,
    attach_threshold: float,
    max_steps: int,
    output_dir: Path,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    if not subset:
        artifact = {
            "experiment": "gate_c3_modality_sensitivity",
            "timestamp": now_ts(),
            "status": "skipped",
            "reason": "No validated subset episodes available.",
        }
        save_json(output_dir / "gate_c3_summary.json", artifact)
        save_text(output_dir / "gate_c3_summary.md", "# Gate C3\n\nNo validated subset episodes were available.\n")
        save_csv(output_dir / "gate_c3_rows.csv", [])
        return artifact

    policy_bundle = _load_policy_bundle(policy_path)
    rows = []
    for item in subset:
        episode_index = int(item["episode_index"])
        npz_path = resolve_npz(item["npz_name"])
        source_npz = dict(np.load(npz_path, allow_pickle=True))
        meta = source_meta(npz_path)
        trace_result = run_replay_mode(
            episode_index=episode_index,
            npz_path=npz_path,
            source_npz=source_npz,
            source_meta_payload=meta,
            replay_mode=item["trajectory_mode"],
            max_steps=max_steps,
            attach_threshold_override=attach_threshold,
            capture_frames=False,
            capture_trace=True,
        )
        for step in trace_result.get("trace_steps", []):
            teacher_action = np.asarray(step["teacher_action"], dtype=np.float32)
            full_action = _predict_policy_action(
                policy_bundle,
                image=np.asarray(step["obs_image"], dtype=np.uint8),
                image2=np.asarray(step["obs_image2"], dtype=np.uint8),
                state=np.asarray(step["obs_state"], dtype=np.float32),
                task=step.get("obs_task"),
                mode="full_inputs",
            )
            images_zeroed = _predict_policy_action(
                policy_bundle,
                image=np.asarray(step["obs_image"], dtype=np.uint8),
                image2=np.asarray(step["obs_image2"], dtype=np.uint8),
                state=np.asarray(step["obs_state"], dtype=np.float32),
                task=step.get("obs_task"),
                mode="images_zeroed_state_preserved",
            )
            state_zeroed = _predict_policy_action(
                policy_bundle,
                image=np.asarray(step["obs_image"], dtype=np.uint8),
                image2=np.asarray(step["obs_image2"], dtype=np.uint8),
                state=np.asarray(step["obs_state"], dtype=np.float32),
                task=step.get("obs_task"),
                mode="state_zeroed_images_preserved",
            )
            rows.append(
                {
                    "episode_index": episode_index,
                    "step_index": int(step["step_index"]),
                    "phase_bucket": _phase_bucket(int(step["step_index"]), trace_result),
                    "teacher_vs_full_l2": float(np.linalg.norm(full_action - teacher_action)),
                    "teacher_vs_images_zeroed_l2": float(np.linalg.norm(images_zeroed - teacher_action)),
                    "teacher_vs_state_zeroed_l2": float(np.linalg.norm(state_zeroed - teacher_action)),
                    "full_vs_images_zeroed_l2": float(np.linalg.norm(full_action - images_zeroed)),
                    "full_vs_state_zeroed_l2": float(np.linalg.norm(full_action - state_zeroed)),
                }
            )

    artifact = {
        "experiment": "gate_c3_modality_sensitivity",
        "timestamp": now_ts(),
        "policy_path": str(policy_path),
        "attach_threshold_m": float(attach_threshold),
        "subset_size": len(subset),
        "phase_bucket_means": {
            bucket: {
                "teacher_vs_full_l2": float(np.mean([row["teacher_vs_full_l2"] for row in rows if row["phase_bucket"] == bucket]))
                if any(row["phase_bucket"] == bucket for row in rows)
                else None,
                "full_vs_images_zeroed_l2": float(np.mean([row["full_vs_images_zeroed_l2"] for row in rows if row["phase_bucket"] == bucket]))
                if any(row["phase_bucket"] == bucket for row in rows)
                else None,
                "full_vs_state_zeroed_l2": float(np.mean([row["full_vs_state_zeroed_l2"] for row in rows if row["phase_bucket"] == bucket]))
                if any(row["phase_bucket"] == bucket for row in rows)
                else None,
            }
            for bucket in [
                "pre-near-handle",
                "near-handle-to-attach",
                "attached-to-major-open",
                "post-major-open",
            ]
        },
        "guardrail": "Sensitivity-only audit. Do not use alone to claim vision root cause.",
    }
    save_json(output_dir / "gate_c3_summary.json", artifact)
    save_csv(output_dir / "gate_c3_rows.csv", rows)
    save_text(output_dir / "gate_c3_summary.md", render_c3_markdown(artifact))
    return artifact


def render_b1_markdown(artifact: dict[str, Any]) -> str:
    agg = artifact["aggregate"]
    lines = [
        "# Gate B1 Threshold Sentinel",
        "",
        "- Proposal-only follow-up. Does not update canonical truth.",
        f"- Timestamp: `{artifact['timestamp']}`",
        f"- Stage A measurement resolution: `{artifact['measurement_image_size_px']}px`",
        f"- Stage A render observations: `{artifact['measurement_render_observations']}`",
        f"- Oracle type: `{artifact['oracle_definition']['oracle_type']}`",
        f"- Oracle proves: {artifact['oracle_definition']['oracle_proves']}",
        f"- Oracle does not prove: {artifact['oracle_definition']['oracle_does_not_prove']}",
        "",
        "## Threshold Summary",
        "",
        "| Threshold | Oracle success | Open-loop success | Oracle-open gap |",
        "|---|---:|---:|---:|",
    ]
    for key, payload in artifact["thresholds"].items():
        mode_results = payload["mode_results"]
        oracle = mode_results["oracle_reachability"]["success_rate"]
        open_loop = mode_results["open_loop"]["success_rate"]
        lines.append(f"| {key} | {oracle:.3f} | {open_loop:.3f} | {oracle - open_loop:.3f} |")
    lines += [
        "",
        f"- Oracle threshold sensitivity: `{agg['oracle_threshold_sensitivity_grade']}`",
        f"- Open-loop threshold sensitivity: `{agg['open_loop_threshold_sensitivity_grade']}`",
        f"- Canonical-threshold oracle status: `{agg['oracle_canonical_threshold_status']}`",
        f"- Primary threshold: `{agg['primary_threshold_m']:.2f}`",
        f"- Dual-threshold report required: `{agg['dual_threshold_required']}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_b2_markdown(artifact: dict[str, Any]) -> str:
    lines = [
        "# Gate B2 Anchored Decomposition",
        "",
        "- Proposal-only follow-up. Does not update canonical truth.",
        f"- Timestamp: `{artifact['timestamp']}`",
        f"- Stage A measurement resolution: `{artifact['measurement_image_size_px']}px`",
        f"- Stage A render observations: `{artifact['measurement_render_observations']}`",
        f"- Contamination material overall: `{artifact['aggregate']['contamination_material']}`",
        "",
    ]
    for key, payload in artifact["thresholds"].items():
        lines += [
            f"## Threshold `{key}`",
            "",
            f"- Attachment-state contamination delta: attach `{payload['attachment_state_contamination_delta']['attach_rate_delta']:.3f}`, success `{payload['attachment_state_contamination_delta']['success_rate_delta']:.3f}`, material=`{payload['attachment_state_contamination_delta']['material']}`",
            f"- Drawer-progress contamination delta: attach `{payload['drawer_progress_contamination_delta']['attach_rate_delta']:.3f}`, success `{payload['drawer_progress_contamination_delta']['success_rate_delta']:.3f}`, material=`{payload['drawer_progress_contamination_delta']['material']}`",
            f"- Anchored mode reordering count: `{payload['anchored_mode_reordering_count']}`",
            "",
        ]
    return "\n".join(lines) + "\n"


def render_b3_markdown(artifact: dict[str, Any]) -> str:
    agg = artifact["aggregate"]
    counts = agg["tier_counts"]
    return "\n".join(
        [
            "# Gate B3 Source Validation",
            "",
            "- Proposal-only follow-up. Does not update canonical truth.",
            f"- Timestamp: `{artifact['timestamp']}`",
            f"- Tier A: `{counts.get('Tier A', 0)}`",
            f"- Tier B: `{counts.get('Tier B', 0)}`",
            f"- Tier X: `{counts.get('Tier X', 0)}`",
            f"- Tier X among success episodes: `{agg['tier_x_among_success_episodes']}`",
            f"- Tier X among failure episodes: `{agg['tier_x_among_failure_episodes']}`",
            "",
        ]
    ) + "\n"


def render_e029_markdown(draft: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# E029 Draft",
            "",
            "Non-canonical corrected Gate B follow-up draft.",
            "",
            f"- Final branch: `{draft['final_branch']}`",
            f"- Stage C entry allowed: `{draft['stage_c_entry_allowed']}`",
            "",
            "## What this draft covers",
            "",
        "- Canonical-threshold sentinel",
        "- Anchored contamination decomposition",
        "- Deterministic mapping and source-validation",
        f"- Stage A measurement resolution: `{STAGE_A_IMAGE_SIZE}px` (physics-invariant acceleration)",
        f"- Stage A render observations: `{STAGE_A_RENDER_OBSERVATIONS}` (state-only measurement path)",
        "",
            "## Guardrails",
            "",
            *[f"- {item}" for item in draft["do_not"]],
            "",
        ]
    ) + "\n"


def render_c1_markdown(artifact: dict[str, Any]) -> str:
    lines = [
        "# Gate C1 Policy vs Teacher Divergence",
        "",
        f"- Policy path: `{artifact.get('policy_path', 'n/a')}`",
        f"- Threshold: `{artifact.get('attach_threshold_m', 'n/a')}`",
        "",
        "## Phase Bucket Mean L2",
    ]
    for bucket, value in artifact.get("phase_bucket_summary", {}).items():
        lines.append(f"- `{bucket}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def render_c2_markdown(artifact: dict[str, Any]) -> str:
    lines = [
        "# Gate C2 Stage-Sliced Hybrid",
        "",
        f"- Policy path: `{artifact.get('policy_path', 'n/a')}`",
        f"- Threshold: `{artifact.get('attach_threshold_m', 'n/a')}`",
        "",
    ]
    for condition, metrics in artifact.get("aggregate", {}).items():
        lines.append(
            f"- `{condition}`: success `{metrics['success_rate']:.3f}`, attach `{metrics['attach_rate']:.3f}`, dominant `{metrics['dominant_failure_stage']}`"
        )
    lines.append("")
    return "\n".join(lines)


def render_c3_markdown(artifact: dict[str, Any]) -> str:
    lines = [
        "# Gate C3 Modality Sensitivity",
        "",
        "- Sensitivity-only audit. Do not use alone to claim vision root cause.",
        f"- Policy path: `{artifact.get('policy_path', 'n/a')}`",
        "",
    ]
    for bucket, metrics in artifact.get("phase_bucket_means", {}).items():
        lines.append(f"- `{bucket}`: {metrics}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gate B Corrected Plan v2")
    parser.add_argument("--episodes", default=None, help="Comma-separated subset of canonical Gate B top-10 episodes")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_FINETUNED_POLICY)
    parser.add_argument("--skip-stage-c", action="store_true")
    parser.add_argument("--run-c3-when-unresolved", action="store_true")
    args = parser.parse_args()

    episode_set = selected_episodes(args.episodes)

    b1_dir = ARTIFACT_ROOT / "gate_b1_threshold_sentinel"
    b2_dir = ARTIFACT_ROOT / "gate_b2_anchored_decomposition"
    b3_dir = ARTIFACT_ROOT / "gate_b3_source_validation"
    c1_dir = ARTIFACT_ROOT / "gate_c1_policy_teacher_divergence"
    c2_dir = ARTIFACT_ROOT / "gate_c2_stage_sliced_hybrid"
    c3_dir = ARTIFACT_ROOT / "gate_c3_modality_sensitivity"

    sentinel = build_b1_sentinel(
        episode_set=episode_set,
        max_steps=args.max_steps,
        output_dir=b1_dir,
    )
    thresholds_for_b2 = [float(v) for v in sentinel["aggregate"]["science_thresholds_m"]]
    anchored = build_b2_anchored(
        episode_set=episode_set,
        max_steps=args.max_steps,
        output_dir=b2_dir,
        thresholds=thresholds_for_b2,
    )
    primary_key = threshold_key(float(sentinel["aggregate"]["primary_threshold_m"]))
    success_episode_set = {
        ep["episode_index"]
        for ep in sentinel["thresholds"][primary_key]["episodes"]
        if ep["modes"]["open_loop"]["success"]
    }
    source_validation = build_b3_source_validation(
        episode_set=episode_set,
        output_dir=b3_dir,
        success_episode_set=success_episode_set,
    )
    e029_draft, final_branch, stage_c_entry_allowed = build_e029(
        sentinel=sentinel,
        anchored=anchored,
        source_validation=source_validation,
    )
    save_yaml(PROPOSALS_DIR / "E029_gate_b_corrected_deconfounded_draft.yaml", e029_draft)
    save_text(PROPOSALS_DIR / "E029_gate_b_corrected_deconfounded_draft.md", render_e029_markdown(e029_draft))

    stage_c_results: dict[str, Any] = {
        "entry_allowed": stage_c_entry_allowed,
        "final_branch": final_branch,
        "skipped": bool(args.skip_stage_c or not stage_c_entry_allowed),
    }
    if not args.skip_stage_c and stage_c_entry_allowed:
        subset = _validated_subset(
            sentinel=sentinel,
            anchored=anchored,
            source_validation=source_validation,
        )
        attach_threshold = float(sentinel["aggregate"]["primary_threshold_m"])
        c1 = build_c1_divergence(
            subset=subset,
            policy_path=args.policy_path,
            attach_threshold=attach_threshold,
            max_steps=args.max_steps,
            output_dir=c1_dir,
        )
        c2 = build_c2_hybrid(
            subset=subset,
            policy_path=args.policy_path,
            attach_threshold=attach_threshold,
            max_steps=args.max_steps,
            output_dir=c2_dir,
        )
        stage_c_results["c1"] = str((c1_dir / "gate_c1_summary.json").relative_to(CAMPAIGN_ROOT))
        stage_c_results["c2"] = str((c2_dir / "gate_c2_summary.json").relative_to(CAMPAIGN_ROOT))
        unresolved = any(
            value is not None and value < 0.05
            for value in c1.get("phase_bucket_summary", {}).values()
        )
        if args.run_c3_when_unresolved and unresolved:
            c3 = build_c3_modality_sensitivity(
                subset=subset,
                policy_path=args.policy_path,
                attach_threshold=attach_threshold,
                max_steps=args.max_steps,
                output_dir=c3_dir,
            )
            stage_c_results["c3"] = str((c3_dir / "gate_c3_summary.json").relative_to(CAMPAIGN_ROOT))

    print(
        json.dumps(
            {
                "stage_a": {
                    "b1": str((b1_dir / "gate_b1_sentinel_summary.json").relative_to(CAMPAIGN_ROOT)),
                    "b2": str((b2_dir / "gate_b2_summary.json").relative_to(CAMPAIGN_ROOT)),
                    "b3": str((b3_dir / "gate_b3_summary.json").relative_to(CAMPAIGN_ROOT)),
                    "e029": "sovereign/proposals/E029_gate_b_corrected_deconfounded_draft.yaml",
                    "final_branch": final_branch,
                    "primary_threshold_m": sentinel["aggregate"]["primary_threshold_m"],
                },
                "stage_c": stage_c_results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

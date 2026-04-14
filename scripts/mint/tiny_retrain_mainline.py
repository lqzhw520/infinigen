#!/usr/bin/env python3
"""Shared helpers for the V1cT2 tiny-retrain mainline."""

from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np
from pathlib import Path
from typing import Any

from drawer_robot_env_mujoco import build_robot_rollout, save_robot_rollout
from mint_common import ARTIFACT_DIR, DATASET_DIR, DATASET_REPO_ID, DEFAULT_TRAIN_SEEDS, load_json
from root_cause_controller import RootCauseController

RCA5_ARTIFACT = ARTIFACT_DIR / "p2rca5_frozen_matrix_screen.json"
RCA7_ARTIFACT = ARTIFACT_DIR / "p2rca7_tiny_retrain_if_eligible.json"
DEFAULT_ROLLOUT_SOURCE_DIR = ARTIFACT_DIR / "g6_canonical_train_rollouts"
MATERIALIZATION_ARTIFACT = ARTIFACT_DIR / "g6_canonical_rollout_materialization.json"
MIN_TRAIN_EPISODES = 12
DEFAULT_EPISODES_PER_SEED = 3
STATE_MODE_MAP = {
    "S0": "m0_proxy",
    "S1": "telemetry_candidate_v3_transition",
    "S2": "telemetry_candidate_v4_task_identity",
}


def expected_training_targets() -> dict[str, Any]:
    rca5 = load_json(RCA5_ARTIFACT, {})
    rca7 = load_json(RCA7_ARTIFACT, {})
    split_a = [int(seed) for seed in (rca5.get("split_a_seeds") or [])]
    seed_source = "rca5_split_a" if split_a else "default_train_seeds"
    if not split_a:
        split_a = list(DEFAULT_TRAIN_SEEDS)
    return {
        "best_transition_cell": rca5.get("best_transition_cell"),
        "canonical_train_cell": rca5.get("canonical_train_cell") or rca7.get("canonical_train_cell"),
        "best_train_state_mode": rca5.get("best_train_state_mode") or rca7.get("best_train_state_mode"),
        "tiny_retrain_permitted": bool(rca7.get("tiny_retrain_permitted", False)),
        "training_seeds": split_a,
        "training_seed_source": seed_source,
        "preferred_canonical_train_cell": rca5.get("preferred_canonical_train_cell") or rca7.get("preferred_canonical_train_cell"),
    }


def dataset_guard(expected: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    provenance_path = DATASET_DIR / "meta" / "provenance.json"
    report = {
        **expected,
        "dataset_root": str(DATASET_DIR),
        "repo_id": DATASET_REPO_ID,
        "provenance_exists": provenance_path.exists(),
    }
    if not bool(expected.get("tiny_retrain_permitted")):
        report["error"] = "RCA7 did not permit tiny retrain"
        return False, report
    if not provenance_path.exists():
        report["error"] = "Dataset provenance.json is missing; training would consume an unbound or stale dataset"
        return False, report
    provenance = json.loads(provenance_path.read_text())
    report["provenance_path"] = str(provenance_path)
    report["state_modes"] = provenance.get("state_modes", [])
    records = provenance.get("records", [])
    report["record_count"] = len(records)
    canonical_values = sorted({rec.get("canonical_train_cell") for rec in records if rec.get("canonical_train_cell") is not None})
    transition_values = sorted({rec.get("best_transition_cell") for rec in records if rec.get("best_transition_cell") is not None})
    train_state_values = sorted({rec.get("best_train_state_mode") for rec in records if rec.get("best_train_state_mode") is not None})
    report["record_canonical_train_cells"] = canonical_values
    report["record_best_transition_cells"] = transition_values
    report["record_best_train_state_modes"] = train_state_values
    expected_state_mode = STATE_MODE_MAP.get(str(expected.get("best_train_state_mode")))
    report["expected_state_mode_name"] = expected_state_mode
    if not canonical_values or str(expected.get("canonical_train_cell")) not in canonical_values:
        report["error"] = "Dataset provenance is not bound to the selected canonical_train_cell"
        return False, report
    if not transition_values or str(expected.get("best_transition_cell")) not in transition_values:
        report["error"] = "Dataset provenance is not bound to the selected best_transition_cell"
        return False, report
    if not train_state_values or str(expected.get("best_train_state_mode")) not in train_state_values:
        report["error"] = "Dataset provenance is not bound to the selected best_train_state_mode"
        return False, report
    if expected_state_mode and expected_state_mode not in set(provenance.get("state_modes", [])):
        report["error"] = "Dataset state_modes do not include the expected state mode for the selected best_train_state_mode"
        return False, report
    report["passed_preflight"] = True
    return True, report


def _augment_rollout_metadata(controller: RootCauseController, rollout: dict[str, Any], spec, expected: dict[str, Any]) -> dict[str, Any]:
    matrix_hash = controller._frozen_matrix_hash_v5_pro() if controller.selector_mode == 'frozen_v5_pro' else None
    baseline_cell_id = controller._baseline_cell_id_v5_pro() if controller.selector_mode == 'frozen_v5_pro' else None
    ts_baseline_cell_id = controller._ts_baseline_cell_id_v6_1() if controller.selector_mode == 'frozen_v5_pro' else None
    rollout['resource_budget_snapshot'] = controller._resource_snapshot()
    rollout['selector_mode'] = controller.selector_mode
    rollout['baseline_cell_id'] = baseline_cell_id
    rollout['ts_baseline_cell_id'] = ts_baseline_cell_id
    rollout['frozen_matrix_hash'] = matrix_hash
    rollout['lane_hash'] = spec.lane_hash
    rollout['best_transition_cell'] = expected.get('best_transition_cell')
    rollout['canonical_train_cell'] = expected.get('canonical_train_cell')
    rollout['best_train_state_mode'] = expected.get('best_train_state_mode')
    probe = dict(rollout.get('handle_probe_metadata') or {})
    if probe:
        probe['selector_mode'] = controller.selector_mode
        probe['baseline_cell_id'] = baseline_cell_id
        probe['ts_baseline_cell_id'] = ts_baseline_cell_id
        probe['frozen_matrix_hash'] = matrix_hash
        rollout['handle_probe_metadata'] = probe
    trace = []
    for item in rollout.get('handle_probe_metadata_trace', []):
        probe_item = dict(item or {})
        probe_item['selector_mode'] = controller.selector_mode
        probe_item['baseline_cell_id'] = baseline_cell_id
        probe_item['ts_baseline_cell_id'] = ts_baseline_cell_id
        probe_item['frozen_matrix_hash'] = matrix_hash
        trace.append(probe_item)
    if trace:
        rollout['handle_probe_metadata_trace'] = trace
    return rollout


def materialize_canonical_train_rollouts(
    source_dir: Path | None = None,
    *,
    expected: dict[str, Any] | None = None,
    force_rebuild: bool = True,
) -> dict[str, Any]:
    expected = dict(expected or expected_training_targets())
    source_dir = Path(source_dir or DEFAULT_ROLLOUT_SOURCE_DIR)
    report: dict[str, Any] = {
        "gate": "g6_canonical_rollout_materialization",
        "source_dir": str(source_dir),
        **expected,
        "episodes_per_seed": int(os.environ.get("MINT_G6_EPISODES_PER_SEED", str(DEFAULT_EPISODES_PER_SEED))),
        "min_train_episodes": MIN_TRAIN_EPISODES,
        "timestamp": time.time(),
    }
    if not bool(expected.get("tiny_retrain_permitted")):
        report["passed"] = False
        report["error"] = "RCA7 did not permit tiny retrain"
        MATERIALIZATION_ARTIFACT.write_text(json.dumps(report, indent=2))
        return report
    canonical_train_cell = str(expected.get("canonical_train_cell") or "")
    if not canonical_train_cell:
        report["passed"] = False
        report["error"] = "Missing canonical_train_cell from RCA5/RCA7"
        MATERIALIZATION_ARTIFACT.write_text(json.dumps(report, indent=2))
        return report
    if force_rebuild and source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)

    controller = RootCauseController(
        max_experiments_per_cycle=1,
        dry_run=False,
        experiment_family='RCA',
        selector_mode='frozen_v5_pro',
        seed_split_a=list(expected.get('training_seeds') or []),
        truthful_measurement_required=True,
        max_rollouts_per_experiment=max(3, int(expected.get('episodes_per_seed', report['episodes_per_seed']))),
    )
    spec = controller._matrix_cell_lane_spec(canonical_train_cell, 'G6', note='Tiny retrain canonical rollout materialization.')
    contract = controller._materialize_contract(spec.env_contract_config)
    interventions = dict(spec.interventions or {})
    rotation_source = str(interventions.get('rotation_source', 'zero'))
    seeds = [int(seed) for seed in (expected.get('training_seeds') or controller.seed_split_a or DEFAULT_TRAIN_SEEDS)]
    episodes_per_seed = int(report['episodes_per_seed'])

    attempted_rollouts = 0
    saved_rollouts = 0
    successful_seeds: set[int] = set()
    saved_files: list[str] = []
    records: list[dict[str, Any]] = []
    for seed in seeds:
        for episode_index in range(episodes_per_seed):
            attempted_rollouts += 1
            rollout = build_robot_rollout(
                seed=int(seed),
                grasp_pose_world=np.eye(4, dtype=np.float32),
                episode_index=int(episode_index),
                max_steps=int(spec.dataset_config.max_steps),
                image_size=256,
                contract=contract,
                rotation_source=rotation_source,
                claim_policy=spec.claim_policy,
                interventions=interventions,
            )
            rollout = _augment_rollout_metadata(controller, rollout, spec, expected)
            success = bool(rollout.get('success'))
            phase_locked_trace = np.asarray(rollout.get('phase_locked_trace', []), dtype=np.float32).reshape(-1)
            effective_pull_trace = np.asarray(rollout.get('effective_pull_progress_trace', []), dtype=np.float32).reshape(-1)
            record = {
                'seed': int(seed),
                'episode_index': int(episode_index),
                'success': success,
                'ever_attached': bool(rollout.get('ever_attached', False)),
                'max_drawer_fraction': float(rollout.get('max_drawer_fraction', 0.0) or 0.0),
                'phase_locked_rate': float(np.mean(phase_locked_trace)) if phase_locked_trace.size else 0.0,
                'effective_pull_progress_peak': float(np.max(effective_pull_trace)) if effective_pull_trace.size else 0.0,
            }
            records.append(record)
            if not success:
                continue
            out_path = source_dir / f"seed_{int(seed):03d}_episode_{int(episode_index):02d}.npz"
            save_robot_rollout(out_path, rollout)
            saved_rollouts += 1
            successful_seeds.add(int(seed))
            saved_files.append(str(out_path))

    report.update({
        'training_seeds': seeds,
        'attempted_rollouts': attempted_rollouts,
        'saved_rollouts': saved_rollouts,
        'successful_seed_count': len(successful_seeds),
        'successful_seeds': sorted(successful_seeds),
        'saved_files': saved_files,
        'records': records,
        'passed': bool(saved_rollouts >= MIN_TRAIN_EPISODES),
    })
    if not report['passed']:
        report['error'] = 'Insufficient successful canonical rollouts to build a tiny-retrain dataset'
    MATERIALIZATION_ARTIFACT.write_text(json.dumps(report, indent=2))
    return report

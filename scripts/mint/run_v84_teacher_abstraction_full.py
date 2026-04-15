#!/usr/bin/env python3
"""v8.4 front-end teacher-abstraction harness.

Implemented scope:
  - phase1a: warm-start baseline plateau assay
  - phase1b: warm-start quasi-static embodiment-bound assay

This harness intentionally does NOT claim to execute the full v8.4 readiness chain.
Its job is to calibrate the front-end teacher-abstraction question before any matched
T3-A/T3-B controller comparison is treated as authoritative.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from drawer_robot_env_mujoco import build_robot_rollout
from mint_common import ARTIFACT_DIR, TINY_RETRAIN_PLAN_PATH, load_json, write_json_atomic
from run_g6_teacher_recovery_probe import _canonical_rollout_context, _default_plan
from tiny_retrain_mainline import _augment_rollout_metadata

PROJECT_ROOT = Path('/mnt/afs2/zhuhaowu/infinigen')
HARD_SEEDS = [2, 4]
REPEATS = [0, 1]
ARTIFACT_PATHS = {
    'phase1a': ARTIFACT_DIR / 'v84_phase1a_warmstart_baseline_plateau.json',
    'phase1b': ARTIFACT_DIR / 'v84_phase1b_fixed_grasp_embodiment_bound.json',
}


def now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload)


def _active_plan() -> dict[str, Any]:
    return {**_default_plan(), **load_json(TINY_RETRAIN_PLAN_PATH, {})}


def _run_rollout(*, seed: int, repeat_idx: int, max_steps: int, controller_mode: str, pull_open_fraction: float) -> dict[str, Any]:
    plan = _active_plan()
    context = _canonical_rollout_context(plan)
    controller = context['controller']
    spec = context['spec']
    expected = dict(context['expected'])
    expected.update({
        'bridge_stage': f'v84_{controller_mode}',
        'bridge_attempt': 'teacher_abstraction_frontend',
    })
    rollout = build_robot_rollout(
        seed=seed,
        grasp_pose_world=np.eye(4, dtype=np.float32),
        episode_index=int(repeat_idx),
        max_steps=max_steps,
        contract=context['contract'],
        rotation_source=context['rotation_source'],
        claim_policy=context['claim_policy'],
        interventions=dict(context['base_interventions']),
        assay_warm_start_kind='attached_phase_locked',
        teacher_controller_mode=controller_mode,
        pull_open_fraction=pull_open_fraction,
    )
    rollout = _augment_rollout_metadata(controller, rollout, spec, expected)
    strict = dict(rollout.get('strict_metrics') or {})
    truth = dict(rollout.get('canonical_training_truth') or {})
    drawer = np.asarray(rollout.get('next_drawer_fractions', []), dtype=np.float32).reshape(-1)
    phase_locked = np.asarray(rollout.get('phase_locked_trace', []), dtype=np.float32).reshape(-1)
    eff = np.asarray(rollout.get('drawer_delta_effective_trace', []), dtype=np.float32).reshape(-1)
    step96_frac = float(np.max(drawer[:96])) if drawer.size else 0.0
    late_delta = float(np.sum(np.clip(eff[96:], 0.0, None))) if eff.size > 96 else 0.0
    return {
        'seed': int(seed),
        'repeat_idx': int(repeat_idx),
        'controller_mode': controller_mode,
        'max_steps': int(max_steps),
        'teacher_episode_class': str(rollout.get('teacher_episode_class', 'rejected_teacher')),
        'strict_success': bool(strict.get('strict_success', False)),
        'max_drawer_fraction': float(rollout.get('max_drawer_fraction', 0.0) or 0.0),
        'post_attach_drawer_delta': float(strict.get('post_attach_drawer_delta', 0.0) or 0.0),
        'attach_persistence': int(strict.get('attach_persistence', 0) or 0),
        'measurement_truthful_for_training': bool(rollout.get('measurement_truthful_for_training', False)),
        'truthful_window_longest_interior_gap': int(rollout.get('truthful_window_longest_interior_gap', 0) or 0),
        'effective_pull_progress_peak': float(np.max(np.asarray(rollout.get('effective_pull_progress_trace', []), dtype=np.float32))) if len(rollout.get('effective_pull_progress_trace', [])) else 0.0,
        'phase_locked_rate': float(np.mean(phase_locked)) if phase_locked.size else 0.0,
        'step_96_max_drawer_fraction': step96_frac,
        'late_phase_drawer_delta_effective_sum': late_delta,
        'teacher_controller_mode_meta': rollout.get('teacher_controller_mode'),
        'strict_drawer_open_threshold': 0.90,
        'near_strict_drawer_fraction_threshold': 0.85,
        'pull_fraction_target': float(rollout.get('pull_fraction_target', 0.0) or 0.0),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    seed_results: dict[str, Any] = {}
    for seed in HARD_SEEDS:
        recs = [r for r in records if int(r['seed']) == seed]
        best = max(recs, key=lambda r: (float(r['max_drawer_fraction']), float(r['post_attach_drawer_delta']))) if recs else None
        seed_results[str(seed)] = {
            'records': recs,
            'best_record': best,
            'best_max_drawer_fraction': float(best['max_drawer_fraction']) if best else 0.0,
            'best_post_attach_drawer_delta': float(best['post_attach_drawer_delta']) if best else 0.0,
            'best_truth_gap': int(best['truthful_window_longest_interior_gap']) if best else None,
            'pass_085': bool(best and float(best['max_drawer_fraction']) >= 0.85),
            'pass_090': bool(best and float(best['max_drawer_fraction']) >= 0.90),
        }
    return seed_results


def run_phase1a() -> dict[str, Any]:
    result: dict[str, Any] = {
        'phase': 'phase1a_warmstart_baseline_plateau',
        'timestamp': now_iso(),
        'description': 'Diagnostic only: warm-start baseline plateau assay on attached_phase_locked start state.',
        'authoritative_scope': 'teacher_abstraction_frontend_only',
    }
    try:
        raw_records = [
            _run_rollout(seed=seed, repeat_idx=repeat, max_steps=192, controller_mode='baseline', pull_open_fraction=0.92)
            for seed in HARD_SEEDS for repeat in REPEATS
        ]
        seed_results = _aggregate(raw_records)
        result['raw_records'] = raw_records
        result['seed_results'] = seed_results
        result['bounded_terminal'] = None
        result['status'] = 'PASS'
        result['note'] = 'Diagnostic plateau only. This phase must not emit X0.'
    except Exception as exc:
        result['status'] = 'ERROR'
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['traceback'] = traceback.format_exc()
        result['bounded_terminal'] = 'PHASE1A_BASELINE_PLATEAU_CRASHED'
    save_json(ARTIFACT_PATHS['phase1a'], result)
    return result


def run_phase1b() -> dict[str, Any]:
    result: dict[str, Any] = {
        'phase': 'phase1b_fixed_grasp_embodiment_bound',
        'timestamp': now_iso(),
        'description': 'Candidate upper-bound assay: warm-start fixed-grasp quasi-static opening under attached_phase_locked start state.',
        'authoritative_scope': 'teacher_abstraction_frontend_only',
    }
    try:
        baseline = load_json(ARTIFACT_PATHS['phase1a'], {})
        raw_records = [
            _run_rollout(seed=seed, repeat_idx=repeat, max_steps=256, controller_mode='embodiment_bound_quasistatic', pull_open_fraction=0.92)
            for seed in HARD_SEEDS for repeat in REPEATS
        ]
        seed_results = _aggregate(raw_records)
        e1 = all(seed_results[str(seed)]['pass_085'] for seed in HARD_SEEDS)
        e2 = any(seed_results[str(seed)]['pass_090'] for seed in HARD_SEEDS)
        baseline_best = {
            seed: float((baseline.get('seed_results', {}).get(str(seed), {}) or {}).get('best_max_drawer_fraction', 0.0) or 0.0)
            for seed in HARD_SEEDS
        }
        improvements = {
            str(seed): float(seed_results[str(seed)]['best_max_drawer_fraction']) - baseline_best[seed]
            for seed in HARD_SEEDS
        }
        stronger_than_baseline = all(delta >= -1e-6 for delta in improvements.values())
        strictly_better_than_baseline = any(delta > 1e-4 for delta in improvements.values())
        candidate_frontier_valid = bool(stronger_than_baseline and strictly_better_than_baseline)
        result['raw_records'] = raw_records
        result['seed_results'] = seed_results
        result['baseline_reference'] = baseline_best
        result['improvements_over_phase1a'] = improvements
        result['candidate_stronger_than_phase1a'] = stronger_than_baseline
        result['candidate_strictly_better_than_phase1a'] = strictly_better_than_baseline
        result['candidate_frontier_valid'] = candidate_frontier_valid
        result['e1_both_seeds_ge_085'] = e1
        result['e2_at_least_one_ge_090'] = e2
        result['t3_pre_candidate_passed'] = bool(candidate_frontier_valid and e1 and e2)
        if not candidate_frontier_valid:
            result['bounded_terminal'] = 'PHASE1B_CANDIDATE_NOT_STRONGER_THAN_BASELINE'
            result['status'] = 'DIAGNOSTIC'
            result['note'] = 'Candidate quasi-static assay did not dominate the Phase1A baseline plateau, so it cannot support an embodiment-bound verdict yet.'
        else:
            result['bounded_terminal'] = None if (e1 and e2) else 'FIXED_GRASP_ACCEPTANCE_BAR_NOT_REACHED_IN_QUASISTATIC_ASSAY'
            result['status'] = 'PASS' if (e1 and e2) else 'BOUNDED'
            result['note'] = 'Candidate assay dominated the baseline plateau; this phase can now be interpreted as stronger embodiment-bound evidence.'
    except Exception as exc:
        result['status'] = 'ERROR'
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['traceback'] = traceback.format_exc()
        result['bounded_terminal'] = 'PHASE1B_EMBODIMENT_BOUND_CRASHED'
    save_json(ARTIFACT_PATHS['phase1b'], result)
    return result


PHASES = {
    'phase1a': run_phase1a,
    'phase1b': run_phase1b,
}


def main() -> int:
    parser = argparse.ArgumentParser(description='v8.4 teacher-abstraction front-end harness')
    parser.add_argument('--phase', choices=list(PHASES.keys()) + ['all'], default='all')
    args = parser.parse_args()
    print('=' * 70)
    print('v8.4 Teacher-Abstraction Front-End Harness')
    print(f'Timestamp: {now_iso()}')
    print(f'Phase: {args.phase}')
    print('=' * 70)
    to_run = list(PHASES.keys()) if args.phase == 'all' else [args.phase]
    final_terminal = None
    for phase_id in to_run:
        t0 = time.time()
        print(f'\n[{phase_id}] ' + '=' * 50)
        res = PHASES[phase_id]()
        elapsed = time.time() - t0
        print(f'[{phase_id}] Status: {res.get("status")}')
        print(f'[{phase_id}] Bounded terminal: {res.get("bounded_terminal")}')
        print(f'[{phase_id}] Elapsed: {elapsed:.1f}s')
        if phase_id in {'phase1a', 'phase1b'}:
            sr = res.get('seed_results', {})
            for seed in HARD_SEEDS:
                item = sr.get(str(seed), {})
                print(f"  seed_{seed} best_frac={item.get('best_max_drawer_fraction')}, post_attach={item.get('best_post_attach_drawer_delta')}, gap={item.get('best_truth_gap')}")
        if res.get('bounded_terminal'):
            final_terminal = res['bounded_terminal']
    print('\n' + '=' * 70)
    print('v8.4 Front-End Summary')
    print('=' * 70)
    print(f'Final terminal: {final_terminal}')
    print(f'Artifacts: {ARTIFACT_DIR.relative_to(PROJECT_ROOT)}/v84_phase1*.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

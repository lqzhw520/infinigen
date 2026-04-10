#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path('/mnt/afs2/zhuhaowu/infinigen')
CAMPAIGN = PROJECT_ROOT / 'experiments' / 'mint' / 'mint_drawer_v1'
ART = CAMPAIGN / 'artifacts'
OUT = CAMPAIGN / 'outputs'
ARCHIVE = CAMPAIGN / 'archive'
SOV = CAMPAIGN / 'sovereign'
EVIDENCE_DIR = SOV / 'evidence'
TZ = timezone(timedelta(hours=8))

P1H_JSON = ART / 'p1h_working_set_manifest.json'
P1H_MD = OUT / 'p1h_working_set_manifest.md'

KEEP_ARTIFACTS = {
    'current_dataset_manifest.json',
    'm0_authority_preflight.json',
    'm1_proxy_asset_smoke.json',
    'm2_true_robot_env_gate.json',
    'm3_anygrasp_gate.json',
    'm4_robot_rollout_gate.json',
    'm5_dataset_pack_gate.json',
    'm6_mint_train_gate.json',
    'm7_mint_eval_gate.json',
    'mujoco_infinigen_mainline_night.json',
    'p1c10_release_runtime_matched_ab.json',
    'p1c11_official_libero_goal_drawer_baseline_rollback_4eab579.json',
    'p1e_mujoco_infinigen_alignment_audit.json',
    'p1f_mainline_alignment_cleanup_report.json',
    'p1g_exact_control_alignment_audit.json',
    'p1h_working_set_manifest.json',
    'p1i_infinigen_mint_alignment_solution_plan.json',
}
KEEP_ARTIFACT_DIRS = {
    'gate_a_episode_admissibility',
    'gate_b_teacher_replayability',
    'm3_mujoco_anygrasp',
    'm4_rollouts_anygrasp_mujoco',
    'm4_rollouts_learning_mujoco',
    'm4_rollouts_oracle_mujoco',
    'p0b_colored_rollouts',
    'p4_physics_legal_rollouts',
    'v58_archive_2026-04-05_1435',
    'v58_physics_legal_rollouts',
    'v58_training_outputs',
    'v59_overfit_outputs',
}
KEEP_OUTPUTS = {
    'm6_mujoco_mint_train',
    'm7_mujoco_eval',
    'mujoco_infinigen_mainline_night.launch.log',
    'mujoco_infinigen_mainline_night_report.md',
    'p1c11_official_libero_goal_drawer_baseline_rollback_4eab579',
    'p1e_mujoco_infinigen_alignment_audit.md',
    'p1f_mainline_alignment_cleanup_report.md',
    'p1g_exact_control_alignment_audit.md',
    'p1h_working_set_manifest.md',
    'p1i_infinigen_mint_alignment_solution_plan.md',
}


def now_iso() -> str:
    return datetime.now(TZ).isoformat(timespec='seconds')


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def rel(path: Path) -> str:
    return str(path.relative_to(CAMPAIGN))


def move_item(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(src), str(dst))
    return dst


def replace_path_str(text: str, old_rel: str, new_rel: str) -> str:
    full_old = f'experiments/mint/mint_drawer_v1/{old_rel}'
    full_new = f'experiments/mint/mint_drawer_v1/{new_rel}'
    text = text.replace(full_old, full_new)
    text = text.replace(old_rel, new_rel)
    return text


def normalize_rel(path_str: str) -> str:
    if path_str.startswith('experiments/mint/mint_drawer_v1/'):
        return path_str.split('experiments/mint/mint_drawer_v1/', 1)[1]
    return path_str


def locate_archived_path(missing_rel: str) -> str | None:
    basename = Path(missing_rel).name
    candidates = [
        ARCHIVE / 'historical_artifacts_root' / basename,
        ARCHIVE / 'historical_outputs_root' / basename,
        ARCHIVE / 'historical_artifact_dirs' / basename,
        ARCHIVE / 'legacy_docs' / basename,
        ARCHIVE / 'old_dataset_snapshots' / basename,
        ARCHIVE / 'invalid_runs' / 'artifacts' / basename,
        ARCHIVE / 'invalid_runs' / 'outputs' / basename,
        ARCHIVE / 'mujoco_pilot_pre_mainline' / 'artifacts' / basename,
        ARCHIVE / 'mujoco_pilot_pre_mainline' / 'outputs' / basename,
    ]
    for cand in candidates:
        if cand.exists():
            return rel(cand)
    return None


def main() -> int:
    ts = now_iso()
    hist_art = ARCHIVE / 'historical_artifacts_root'
    hist_out = ARCHIVE / 'historical_outputs_root'
    hist_art_dirs = ARCHIVE / 'historical_artifact_dirs'
    hist_art.mkdir(parents=True, exist_ok=True)
    hist_out.mkdir(parents=True, exist_ok=True)
    hist_art_dirs.mkdir(parents=True, exist_ok=True)

    artifact_items_before = sorted(p.name for p in ART.iterdir() if p.is_file())
    artifact_dirs_before = sorted(p.name for p in ART.iterdir() if p.is_dir())
    output_items_before = sorted(p.name for p in OUT.iterdir())

    moved_artifacts: dict[str, str] = {}
    moved_artifact_dirs: dict[str, str] = {}
    moved_outputs: dict[str, str] = {}

    for item in sorted(ART.iterdir()):
        if item.is_file():
            if item.name in KEEP_ARTIFACTS:
                continue
            dst = move_item(item, hist_art)
            moved_artifacts[rel(item)] = rel(dst)
        elif item.is_dir():
            if item.name in KEEP_ARTIFACT_DIRS or item.name.startswith('m4_rollouts_'):
                continue
            dst = move_item(item, hist_art_dirs)
            moved_artifact_dirs[rel(item)] = rel(dst)

    for item in sorted(OUT.iterdir()):
        if item.name in KEEP_OUTPUTS:
            continue
        dst = move_item(item, hist_out)
        moved_outputs[rel(item)] = rel(dst)

    # Update evidence index and YAML path references for moved files.
    idx_path = EVIDENCE_DIR / 'index.json'
    idx = load_json(idx_path)
    updated_evidence_ids: list[str] = []
    move_map = {**moved_artifacts, **moved_artifact_dirs, **moved_outputs}

    for entry in idx.get('entries', []):
        path = entry.get('path')
        if not isinstance(path, str):
            continue
        normalized = normalize_rel(path)
        if normalized in move_map:
            new_rel = move_map[normalized]
            entry['path'] = f'experiments/mint/mint_drawer_v1/{new_rel}'
            updated_evidence_ids.append(entry['evidence_id'])
            yml = EVIDENCE_DIR / f"{entry['evidence_id']}.yaml"
            if yml.exists():
                raw = yml.read_text()
                try:
                    doc = yaml.safe_load(raw)
                except Exception:
                    replaced = replace_path_str(raw, normalized, new_rel)
                    if replaced != raw:
                        yml.write_text(replaced)
                else:
                    if isinstance(doc, dict) and isinstance(doc.get('path'), str):
                        current_path = doc['path']
                        curr_norm = current_path
                        if curr_norm.startswith('experiments/mint/mint_drawer_v1/'):
                            curr_norm = curr_norm.split('experiments/mint/mint_drawer_v1/', 1)[1]
                        if curr_norm == normalized:
                            doc['path'] = f'experiments/mint/mint_drawer_v1/{new_rel}'
                            yml.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
        else:
            abs_path = CAMPAIGN / normalized
            if not abs_path.exists():
                archived_rel = locate_archived_path(normalized)
                if archived_rel:
                    entry['path'] = f'experiments/mint/mint_drawer_v1/{archived_rel}'
                    updated_evidence_ids.append(entry['evidence_id'])
                    yml = EVIDENCE_DIR / f"{entry['evidence_id']}.yaml"
                    if yml.exists():
                        raw = yml.read_text()
                        replaced = replace_path_str(raw, normalized, archived_rel)
                        if replaced != raw:
                            yml.write_text(replaced)
    idx['last_updated'] = ts

    # Add new evidence entries.
    existing_ids = {entry['evidence_id'] for entry in idx.get('entries', [])}
    new_entries = [
        {
            'evidence_id': 'E045',
            'experiment_id': 'p1f_mainline_alignment_cleanup_report',
            'type': 'mainline_summary',
            'timestamp': ts,
            'path': 'experiments/mint/mint_drawer_v1/artifacts/p1f_mainline_alignment_cleanup_report.json',
            'verified': True,
            'summary': 'Canonical summary of what the MuJoCo mainline night-runner actually did, what was aligned, what trained, and what still failed.'
        },
        {
            'evidence_id': 'E046',
            'experiment_id': 'p1g_exact_control_alignment_audit',
            'type': 'exact_control_alignment_audit',
            'timestamp': ts,
            'path': 'experiments/mint/mint_drawer_v1/artifacts/p1g_exact_control_alignment_audit.json',
            'verified': True,
            'summary': 'Exact-control audit using p1c11 successful videos as the control visual surface. Shows task-text/resolution alignment is direct, but visual/state/action equivalence remains unproven and current target variance is far lower than the successful control.'
        },
        {
            'evidence_id': 'E047',
            'experiment_id': 'p1h_working_set_cleanup',
            'type': 'working_set_cleanup',
            'timestamp': ts,
            'path': 'experiments/mint/mint_drawer_v1/artifacts/p1h_working_set_manifest.json',
            'verified': True,
            'summary': 'Canonical working set cleanup. Historical artifacts/outputs moved under archive/, evidence paths updated, and current root working set reduced to the active MuJoCo mainline plus control references.'
        },
    ]
    for entry in new_entries:
        if entry['evidence_id'] not in existing_ids:
            idx['entries'].append(entry)
    write_json(idx_path, idx)

    # Emit lightweight evidence YAMLs.
    for eid, exp_id, etype, summary, path in [
        ('E045', 'p1f_mainline_alignment_cleanup_report', 'mainline_summary', 'Canonical summary of last night\'s MuJoCo mainline run and cleanup.', 'experiments/mint/mint_drawer_v1/artifacts/p1f_mainline_alignment_cleanup_report.json'),
        ('E046', 'p1g_exact_control_alignment_audit', 'exact_control_alignment_audit', 'Exact-control audit showing that current alignment claims are only partially supported; visual/state/action equivalence is still unresolved.', 'experiments/mint/mint_drawer_v1/artifacts/p1g_exact_control_alignment_audit.json'),
        ('E047', 'p1h_working_set_cleanup', 'working_set_cleanup', 'Root-level artifacts/outputs cleaned into a canonical working set plus archive, with evidence paths updated.', 'experiments/mint/mint_drawer_v1/artifacts/p1h_working_set_manifest.json'),
    ]:
        ypath = EVIDENCE_DIR / f'{eid}.yaml'
        ydoc = {
            'evidence_id': eid,
            'experiment_id': exp_id,
            'type': etype,
            'timestamp': ts,
            'path': path,
            'verified': True,
            'summary': summary,
        }
        ypath.write_text(yaml.safe_dump(ydoc, sort_keys=False, allow_unicode=True))

    # Update sovereign truth/current actions to point at the new review surface.
    current_truth_path = SOV / 'current_truth.json'
    current_truth = load_json(current_truth_path)
    current_truth['generated_at'] = ts
    current_truth['current']['phase'] = 'v59_MUJOCO_MAINLINE_PHASE7 — exact-control alignment audit and working-set cleanup completed'
    current_truth['current']['phase_gate'] = 'p1g_exact_control_alignment_audit'
    next_action = {
        'type': 'MUJOCO_ALIGNMENT_EXPANSION',
        'id': 'capture_exact_control_trace_and_expand_success_seeds',
        'priority': 'P0',
        'target': 'Capture exact successful control traces and expand successful MuJoCo train seed coverage before the next fine-tune',
        'status': 'pending',
        'description': 'p1g shows task-text/resolution alignment is direct, but exact control state/action equivalence is still missing and the current train set is only 5 successful seeds repeated 10x.',
        'control_baseline_ref': 'E036/p1c11_official_libero_goal_drawer_baseline_rollback_4eab579',
        'alignment_audit_ref': 'E046/p1g_exact_control_alignment_audit',
        'cleanup_manifest_ref': 'E047/p1h_working_set_cleanup_manifest',
        'updated_at': ts,
    }
    current_truth['current']['next_action'] = next_action
    current_truth['current']['review_surfaces'] = [
        {
            'evidence_id': 'E045',
            'path': 'experiments/mint/mint_drawer_v1/artifacts/p1f_mainline_alignment_cleanup_report.json',
            'role': 'run_summary',
        },
        {
            'evidence_id': 'E046',
            'path': 'experiments/mint/mint_drawer_v1/artifacts/p1g_exact_control_alignment_audit.json',
            'role': 'exact_control_alignment_audit',
        },
        {
            'evidence_id': 'E047',
            'path': 'experiments/mint/mint_drawer_v1/artifacts/p1h_working_set_manifest.json',
            'role': 'working_set_manifest',
        },
    ]
    current_truth['current']['canonical_readable_reports'] = [
        'experiments/mint/mint_drawer_v1/outputs/p1f_mainline_alignment_cleanup_report.md',
        'experiments/mint/mint_drawer_v1/outputs/p1g_exact_control_alignment_audit.md',
        'experiments/mint/mint_drawer_v1/outputs/p1h_working_set_manifest.md',
        'experiments/mint/mint_drawer_v1/outputs/p1i_infinigen_mint_alignment_solution_plan.md',
    ]
    current_truth['current']['open_gates'] = [next_action]
    current_truth['current']['blockers'] = [
        'Exact successful control state/action traces are still unavailable for one-to-one comparison against the MuJoCo/Infinigen training surface.',
        'Current learning set covers only 5 successful train seeds, repeated 10x; diversity is still too low for a strong adaptation claim.',
        'Visual encoder compatibility is supported only by pixel-level heuristics; no encoder-level feature matching has been measured yet.',
    ]
    current_truth['current']['stale_docs'] = [
        'p1e task_text_gap is superseded by p1g exact-control audit.',
        'Historical artifacts/outputs are archived under archive/ and should not be used as the current working set.',
    ]
    current_truth['current']['authority_notes'] = current_truth['current'].get('authority_notes', []) + [
        'Use E045/E046/E047 as the canonical review surface for the completed MuJoCo mainline run.',
    ]
    current_truth['dataset'] = {
        'manifest_version': 2,
        'dataset_version': 'mujoco_mainline_20260410',
        'created_at': ts,
        'producer_script': 'scripts/mint/run_mujoco_infinigen_mainline_night.py:m5_dataset_pack_gate',
        'pack_result_json': '/mnt/afs2/zhuhaowu/infinigen/experiments/mint/mint_drawer_v1/artifacts/m5_dataset_pack_gate.json',
        'source_dirs': ['artifacts/m4_rollouts_learning_mujoco/'],
        'source_rollout_count': 50,
        'episode_count': 50,
        'frame_count': 500,
        'dataset_loads': True,
        'dataset_length': 500,
        'image_shape': '(256, 256, 3)',
        'state_shape': '(8,)',
        'action_shape': '(7,)',
        'task_coverage': ['open the middle drawer of the cabinet'],
        'seed_coverage': [1, 5, 6, 8, 10],
        'learning_source': 'oracle_handle',
    }
    # Update workspace dirty status truthfully.
    git_bin = '/root/anaconda3/envs/mint/bin/git'
    status_out = subprocess.run([git_bin, 'status', '--short'], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    current_truth['workspace']['main_repo']['dirty'] = bool(status_out)
    current_truth['workspace']['main_repo']['dirty_files'] = status_out[:50]
    write_json(current_truth_path, current_truth)

    next_actions_path = SOV / 'next_actions.json'
    next_actions = load_json(next_actions_path)
    next_actions['decision'] = current_truth['current']['decision']
    actions = next_actions.get('actions', [])
    for action in actions:
        if action.get('id') == 'review_mujoco_infinigen_mainline_results':
            action['status'] = 'completed'
            action['result'] = {
                'summary_ref': 'E045',
                'alignment_audit_ref': 'E046',
                'cleanup_manifest_ref': 'E047',
            }
    actions = [a for a in actions if a.get('id') != next_action['id']]
    actions.insert(0, next_action)
    actions.insert(1, {
        'type': 'EXACT_CONTROL_ALIGNMENT_AUDIT',
        'id': 'p1g_exact_control_alignment_audit',
        'priority': 'P0',
        'target': 'Summarize direct vs heuristic alignment evidence against the restored official control',
        'status': 'completed',
        'evidence': 'E046',
    })
    actions.insert(2, {
        'type': 'WORKING_SET_CLEANUP',
        'id': 'p1h_working_set_cleanup',
        'priority': 'P0',
        'target': 'Reduce artifacts/outputs root to the active working set and archive the historical remainder',
        'status': 'completed',
        'evidence': 'E047',
    })
    next_actions['actions'] = actions
    write_json(next_actions_path, next_actions)

    manifest = {
        'timestamp': ts,
        'kept_artifacts': sorted(KEEP_ARTIFACTS),
        'kept_artifact_dirs': sorted(KEEP_ARTIFACT_DIRS),
        'kept_outputs': sorted(KEEP_OUTPUTS),
        'moved_artifacts': moved_artifacts,
        'moved_artifact_dirs': moved_artifact_dirs,
        'moved_outputs': moved_outputs,
        'archived_artifact_root_inventory': sorted(p.name for p in hist_art.iterdir()) if hist_art.exists() else [],
        'archived_artifact_dir_inventory': sorted(p.name for p in hist_art_dirs.iterdir()) if hist_art_dirs.exists() else [],
        'archived_output_root_inventory': sorted(p.name for p in hist_out.iterdir()) if hist_out.exists() else [],
        'updated_evidence_ids': sorted(set(updated_evidence_ids + ['E045', 'E046', 'E047'])),
        'artifact_root_count_before': len(artifact_items_before),
        'artifact_root_count_after': len([p for p in ART.iterdir() if p.is_file()]),
        'artifact_dir_count_before': len(artifact_dirs_before),
        'artifact_dir_count_after': len([p for p in ART.iterdir() if p.is_dir()]),
        'output_root_count_before': len(output_items_before),
        'output_root_count_after': len(list(OUT.iterdir())),
        'archive_dirs': {
            'historical_artifacts_root': str(hist_art),
            'historical_artifact_dirs': str(hist_art_dirs),
            'historical_outputs_root': str(hist_out),
        },
        'review_surfaces': [
            'artifacts/p1f_mainline_alignment_cleanup_report.json',
            'artifacts/p1g_exact_control_alignment_audit.json',
            'artifacts/p1h_working_set_manifest.json',
            'artifacts/p1i_infinigen_mint_alignment_solution_plan.json',
        ],
    }
    write_json(P1H_JSON, manifest)
    md_lines = [
        '# P1h Working Set Cleanup Manifest',
        '',
        '## Root Counts',
        f"- artifact_root_count_before: `{manifest['artifact_root_count_before']}`",
        f"- artifact_root_count_after: `{manifest['artifact_root_count_after']}`",
        f"- artifact_dir_count_before: `{manifest['artifact_dir_count_before']}`",
        f"- artifact_dir_count_after: `{manifest['artifact_dir_count_after']}`",
        f"- output_root_count_before: `{manifest['output_root_count_before']}`",
        f"- output_root_count_after: `{manifest['output_root_count_after']}`",
        '',
        '## Kept Artifact Working Set',
    ]
    md_lines.extend(f"- `{item}`" for item in manifest['kept_artifacts'])
    md_lines.extend(['', '## Kept Artifact Directories'])
    md_lines.extend(f"- `{item}`" for item in manifest['kept_artifact_dirs'])
    md_lines.extend(['', '## Kept Output Working Set'])
    md_lines.extend(f"- `{item}`" for item in manifest['kept_outputs'])
    md_lines.extend(['', '## Archived Artifact Root Items'])
    md_lines.extend(f"- `{k}` -> `{v}`" for k, v in moved_artifacts.items())
    if manifest['archived_artifact_root_inventory']:
        md_lines.append('- archived_inventory:')
        md_lines.extend(f"  - `{item}`" for item in manifest['archived_artifact_root_inventory'])
    md_lines.extend(['', '## Archived Artifact Directories'])
    md_lines.extend(f"- `{k}` -> `{v}`" for k, v in moved_artifact_dirs.items())
    if manifest['archived_artifact_dir_inventory']:
        md_lines.append('- archived_inventory:')
        md_lines.extend(f"  - `{item}`" for item in manifest['archived_artifact_dir_inventory'])
    md_lines.extend(['', '## Archived Output Root Items'])
    md_lines.extend(f"- `{k}` -> `{v}`" for k, v in moved_outputs.items())
    if manifest['archived_output_root_inventory']:
        md_lines.append('- archived_inventory:')
        md_lines.extend(f"  - `{item}`" for item in manifest['archived_output_root_inventory'])
    md_lines.extend(['', '## Sovereign Review Surfaces'])
    md_lines.extend(f"- `{item}`" for item in manifest['review_surfaces'])
    P1H_MD.write_text('\n'.join(md_lines) + '\n')

    print(json.dumps({'json': str(P1H_JSON), 'md': str(P1H_MD)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

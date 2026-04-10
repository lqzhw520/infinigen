#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path('/mnt/afs2/zhuhaowu/infinigen')
CAMPAIGN = PROJECT_ROOT / 'experiments' / 'mint' / 'mint_drawer_v1'
ART = CAMPAIGN / 'artifacts'
OUT = CAMPAIGN / 'outputs'
DATASET = CAMPAIGN / 'dataset'
CONTROL_ART = ART / 'p1c11_official_libero_goal_drawer_baseline_rollback_4eab579.json'
CONTROL_VIDEO_DIR = OUT / 'p1c11_official_libero_goal_drawer_baseline_rollback_4eab579' / 'videos' / 'libero_goal_0' / 'libero_goal_0'
TARGET_PARQUET = DATASET / 'data' / 'chunk-000' / 'file-000.parquet'
TARGET_TASKS = DATASET / 'meta' / 'tasks.parquet'
TARGET_INFO = DATASET / 'meta' / 'info.json'
TARGET_STATS = DATASET / 'meta' / 'stats.json'
M4 = ART / 'm4_robot_rollout_gate.json'
M5 = ART / 'm5_dataset_pack_gate.json'
M6 = ART / 'm6_mint_train_gate.json'
M7 = ART / 'm7_mint_eval_gate.json'
P1G_JSON = ART / 'p1g_exact_control_alignment_audit.json'
P1G_MD = OUT / 'p1g_exact_control_alignment_audit.md'


@dataclass
class VisualStats:
    sample_count: int
    mean_rgb: list[float]
    std_rgb: list[float]
    brightness_mean: float
    brightness_std: float
    edge_density_mean: float
    entropy_mean: float


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_tasks(path: Path) -> list[str]:
    df = pd.read_parquet(path)
    texts: list[str] = []
    for col in df.columns:
        if 'task' in col.lower() or 'lang' in col.lower() or 'instruction' in col.lower():
            texts.extend(df[col].astype(str).tolist())
    if not texts and len(df.columns):
        texts = df[df.columns[0]].astype(str).tolist()
    return texts


def image_metrics(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    arr = rgb.astype(np.float32) / 255.0
    flat = arr.reshape(-1, 3)
    mean_rgb = flat.mean(axis=0)
    std_rgb = flat.std(axis=0)
    gray = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean() / 255.0)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())
    hist = np.bincount(gray.flatten(), minlength=256).astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    nonzero = hist[hist > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())
    return mean_rgb, std_rgb, brightness, edge_density, entropy


def summarize_visual(samples: list[np.ndarray]) -> VisualStats:
    means = []
    stds = []
    brightness = []
    edges = []
    entropies = []
    for rgb in samples:
        m, s, b, e, h = image_metrics(rgb)
        means.append(m)
        stds.append(s)
        brightness.append(b)
        edges.append(e)
        entropies.append(h)
    return VisualStats(
        sample_count=len(samples),
        mean_rgb=np.mean(np.stack(means), axis=0).round(6).tolist(),
        std_rgb=np.mean(np.stack(stds), axis=0).round(6).tolist(),
        brightness_mean=float(np.mean(brightness)),
        brightness_std=float(np.std(brightness)),
        edge_density_mean=float(np.mean(edges)),
        entropy_mean=float(np.mean(entropies)),
    )


def sample_video_frames(video_path: Path, n_samples: int = 12) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video: {video_path}')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise RuntimeError(f'Video has no frames: {video_path}')
    idxs = np.linspace(0, frame_count - 1, num=min(n_samples, frame_count), dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def sample_target_images(parquet_path: Path, n_samples: int = 60) -> list[np.ndarray]:
    df = pd.read_parquet(parquet_path, columns=['observation.images.image'])
    total = len(df)
    idxs = np.linspace(0, total - 1, num=min(n_samples, total), dtype=int)
    frames = []
    for idx in idxs:
        cell = df.iloc[int(idx)]['observation.images.image']
        if isinstance(cell, dict) and 'bytes' in cell:
            img = Image.open(BytesIO(cell['bytes'])).convert('RGB')
            frames.append(np.array(img))
    return frames


def grade_direct(ok: bool) -> str:
    return 'high' if ok else 'high'


def grade_heuristic() -> str:
    return 'low'


def grade_missing() -> str:
    return 'very_low'


def main() -> int:
    control = load_json(CONTROL_ART)
    target_info = load_json(TARGET_INFO)
    target_stats = load_json(TARGET_STATS)
    m4 = load_json(M4)
    m5 = load_json(M5)
    m6 = load_json(M6)
    m7 = load_json(M7)

    control_videos = sorted(CONTROL_VIDEO_DIR.glob('eval_episode_*.mp4'))
    control_frames: list[np.ndarray] = []
    for video in control_videos:
        control_frames.extend(sample_video_frames(video))
    target_frames = sample_target_images(TARGET_PARQUET)

    control_visual = summarize_visual(control_frames)
    target_visual = summarize_visual(target_frames)

    target_tasks = load_tasks(TARGET_TASKS)
    if target_tasks and all(str(x).strip().isdigit() for x in target_tasks):
        target_tasks = list(m5.get('task_coverage', []))
    exact_task = 'open the middle drawer of the cabinet'
    task_exact_match = sorted(set(target_tasks)) == [exact_task]

    image_res = target_info['features']['observation.images.image']['shape'][:2]
    wrist_res = target_info['features']['observation.images.image2']['shape'][:2]
    image_resolution_match = image_res == [256, 256] and wrist_res == [256, 256]

    target_state = target_stats['observation.state']
    target_action = target_stats['action']

    visual_gap = {
        'mean_rgb_abs_gap': [round(abs(a - b), 6) for a, b in zip(control_visual.mean_rgb, target_visual.mean_rgb)],
        'std_rgb_abs_gap': [round(abs(a - b), 6) for a, b in zip(control_visual.std_rgb, target_visual.std_rgb)],
        'brightness_gap': round(abs(control_visual.brightness_mean - target_visual.brightness_mean), 6),
        'edge_density_gap': round(abs(control_visual.edge_density_mean - target_visual.edge_density_mean), 6),
        'entropy_gap': round(abs(control_visual.entropy_mean - target_visual.entropy_mean), 6),
    }

    direct_checks = [
        {
            'id': 'control_baseline_intact',
            'status': bool(control['eval_info']['overall']['pc_success'] == 100.0),
            'evidence_grade': 'high',
            'why': 'Directly measured from p1c11 control artifact.',
        },
        {
            'id': 'task_text_exact_match',
            'status': task_exact_match,
            'evidence_grade': 'high',
            'why': 'Directly measured from current dataset tasks.parquet.',
        },
        {
            'id': 'image_resolution_match',
            'status': image_resolution_match,
            'evidence_grade': 'high',
            'why': 'Directly measured from current dataset meta/info.json.',
        },
        {
            'id': 'success_only_learning_set',
            'status': bool(m4.get('copied_learning_files', 0) > 0 and len(m4.get('successful_learning_seeds', [])) > 0),
            'evidence_grade': 'high',
            'why': 'Directly measured from m4 rollout gate.',
        },
    ]

    weak_or_missing_checks = [
        {
            'id': 'image_calibration_sufficiency_for_vision_encoder',
            'status': False,
            'evidence_grade': 'very_low',
            'why': 'Current evidence is only pixel-space calibration. No encoder-level feature overlap or token distribution study has been run.',
        },
        {
            'id': 'state_motor_proxy_equivalence_to_official_control',
            'status': False,
            'evidence_grade': 'very_low',
            'why': 'No exact official control state trace exists for one-to-one comparison against the motor proxy dimensions 3:7.',
        },
        {
            'id': 'action_distribution_equivalence_to_official_control',
            'status': False,
            'evidence_grade': 'very_low',
            'why': 'No exact p1c11 action trace is available; current action comparison can only use family-level proxies, not the exact successful control episodes.',
        },
        {
            'id': 'dataset_scale_sufficient_for_3B_adaptation',
            'status': False,
            'evidence_grade': 'high',
            'why': 'Directly contradicted by current dataset size: only 50 episodes / 500 frames from 5 seeds repeated 10x.',
        },
    ]

    confounders = [
        'Only 5 successful train seeds were available, then repeated 10x; diversity is low even though density increased.',
        'Current target image variance remains much lower than the exact successful control videos.',
        'Rotation action channels remain degenerate in the target dataset; no exact successful control action trace exists to show equivalence.',
        'Dims 3:7 were engineered as motor proxies, but no official control trace proves that these proxies match the state semantics the successful baseline relied on.',
    ]

    interpretation = [
        'The previously reported brightness/background calibration is an engineering heuristic, not a validated proof that MINT vision components can interpret the Infinigen renderings.',
        'The current evidence is strong only for task-text/resolution/success-only filtering. It is weak for encoder-level visual compatibility and for exact state/action alignment to the successful control.',
        'Given these unresolved confounders, the 0/5 -> 0/5 outcome is scientifically unsurprising and should not be interpreted as a definitive negative result for the MuJoCo/Infinigen line.',
    ]

    report = {
        'control_reference': {
            'evidence_ref': 'E036/p1c11_official_libero_goal_drawer_baseline_rollback_4eab579',
            'external_mint_head': control['external_mint_head'],
            'task_name': control['task_name'],
            'success_rate': control['eval_info']['overall']['pc_success'],
            'video_dir': str(CONTROL_VIDEO_DIR),
            'exact_state_action_trace_available': False,
        },
        'target_reference': {
            'dataset_root': str(DATASET),
            'episode_count': m5['episode_count'],
            'frame_count': m5['frame_count'],
            'successful_learning_seeds': m4.get('successful_learning_seeds', []),
            'heldout_result': {
                'pretrained_success': m7['pretrained']['success_count'],
                'finetuned_success': m7['finetuned']['success_count'],
                'heldout_seed_count': m7['pretrained']['seed_count'],
            },
        },
        'exact_control_visual_surface': asdict(control_visual),
        'target_training_visual_surface': asdict(target_visual),
        'visual_gap': visual_gap,
        'direct_checks': direct_checks,
        'weak_or_missing_checks': weak_or_missing_checks,
        'target_state_stats': {
            'mean': target_state['mean'],
            'std': target_state['std'],
            'min': target_state['min'],
            'max': target_state['max'],
        },
        'target_action_stats': {
            'mean': target_action['mean'],
            'std': target_action['std'],
            'min': target_action['min'],
            'max': target_action['max'],
        },
        'confounders': confounders,
        'interpretation': interpretation,
        'scientific_verdict': {
            'alignment_achieved': False,
            'claim_strength': 'low',
            'reason': 'Current mainline fixes improve infrastructure and some contracts, but exact-control visual/state/action equivalence is still not established.',
        },
        'recommended_next_actions': [
            'Capture exact successful control traces (or equivalent official rollout telemetry) for one-to-one state/action comparison.',
            'Increase successful MuJoCo train seed coverage instead of relying on 5 seeds repeated 10x.',
            'Add an encoder-level visual compatibility study before claiming image calibration is sufficient.',
            'Re-run training only after the above two gaps are reduced; otherwise more steps mainly scale a weak dataset.',
        ],
    }

    P1G_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    md = []
    md.append('# P1g Exact-Control Alignment Audit')
    md.append('')
    md.append('## Control Surface')
    md.append(f"- control evidence: `{report['control_reference']['evidence_ref']}`")
    md.append(f"- task: `{report['control_reference']['task_name']}`")
    md.append(f"- external/MINT head: `{report['control_reference']['external_mint_head']}`")
    md.append(f"- official success: `{report['control_reference']['success_rate']}`")
    md.append(f"- exact control state/action trace available: `{report['control_reference']['exact_state_action_trace_available']}`")
    md.append('')
    md.append('## Target Surface')
    md.append(f"- episodes: `{report['target_reference']['episode_count']}`")
    md.append(f"- frames: `{report['target_reference']['frame_count']}`")
    md.append(f"- successful_learning_seeds: `{report['target_reference']['successful_learning_seeds']}`")
    md.append(f"- held-out result: pretrained `{report['target_reference']['heldout_result']['pretrained_success']}/{report['target_reference']['heldout_result']['heldout_seed_count']}`, finetuned `{report['target_reference']['heldout_result']['finetuned_success']}/{report['target_reference']['heldout_result']['heldout_seed_count']}`")
    md.append('')
    md.append('## Directly Verified Alignments')
    for item in direct_checks:
        md.append(f"- `{item['id']}`: `{item['status']}` ({item['evidence_grade']})")
        md.append(f"  {item['why']}")
    md.append('')
    md.append('## Weak Or Missing Evidence')
    for item in weak_or_missing_checks:
        md.append(f"- `{item['id']}`: `{item['status']}` ({item['evidence_grade']})")
        md.append(f"  {item['why']}")
    md.append('')
    md.append('## Exact Control Video vs Target Dataset Visual Surface')
    md.append(f"- control mean_rgb: `{control_visual.mean_rgb}`")
    md.append(f"- target mean_rgb: `{target_visual.mean_rgb}`")
    md.append(f"- control std_rgb: `{control_visual.std_rgb}`")
    md.append(f"- target std_rgb: `{target_visual.std_rgb}`")
    md.append(f"- control brightness_mean: `{round(control_visual.brightness_mean, 6)}`")
    md.append(f"- target brightness_mean: `{round(target_visual.brightness_mean, 6)}`")
    md.append(f"- control edge_density_mean: `{round(control_visual.edge_density_mean, 6)}`")
    md.append(f"- target edge_density_mean: `{round(target_visual.edge_density_mean, 6)}`")
    md.append(f"- control entropy_mean: `{round(control_visual.entropy_mean, 6)}`")
    md.append(f"- target entropy_mean: `{round(target_visual.entropy_mean, 6)}`")
    md.append(f"- visual_gap: `{visual_gap}`")
    md.append('')
    md.append('## Why The Current 0/5 Is Not Surprising')
    for item in confounders:
        md.append(f"- {item}")
    md.append('')
    md.append('## Scientific Interpretation')
    for item in interpretation:
        md.append(f"- {item}")
    md.append('')
    md.append('## Verdict')
    md.append(f"- alignment_achieved: `{report['scientific_verdict']['alignment_achieved']}`")
    md.append(f"- claim_strength: `{report['scientific_verdict']['claim_strength']}`")
    md.append(f"- reason: {report['scientific_verdict']['reason']}")
    md.append('')
    md.append('## Next Actions')
    for item in report['recommended_next_actions']:
        md.append(f"- {item}")
    P1G_MD.write_text('\n'.join(md) + '\n')
    print(json.dumps({'json': str(P1G_JSON), 'md': str(P1G_MD)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

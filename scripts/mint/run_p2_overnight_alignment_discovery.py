#!/usr/bin/env python3
"""Overnight discovery matrix for MuJoCo/Infinigen -> MINT alignment.

This runner is intentionally broader than the canonical single-line flow:
it explores a small lane matrix overnight, ranks lanes by data-line quality,
then trains/evaluates the top-K lanes automatically.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))

from mint_common import now_iso, write_json_atomic, write_text_atomic
from p1_execution_common import ARTIFACT_DIR, CAMPAIGN_DIR, OUTPUT_DIR, upsert_evidence
import drawer_robot_env_mujoco as drm
from drawer_robot_env_mujoco import DrawerRobotEnvMuJoCo, build_oracle_grasp_pose, drawer_manifest, save_robot_rollout
from dataset_builder import build_dataset_from_rollouts
import run_mujoco_infinigen_mainline_night as mainline
from mujoco_mainline_common import default_heldout_seeds

EXPERIMENT_ID = "p2_overnight_alignment_discovery"
EVIDENCE_ID = "E059"
ARTIFACT_PATH = ARTIFACT_DIR / f"{EXPERIMENT_ID}.json"
REPORT_PATH = OUTPUT_DIR / f"{EXPERIMENT_ID}.md"
LOG_DIR = OUTPUT_DIR / EXPERIMENT_ID
SPEC_PATH = ARTIFACT_DIR / f"{EXPERIMENT_ID}_spec.json"
TRAIN_STEPS = int(os.environ.get("P2_DISCOVERY_TRAIN_STEPS", "6000"))
BATCH_SIZE = int(os.environ.get("P2_DISCOVERY_BATCH_SIZE", "4"))
TOP_K = int(os.environ.get("P2_DISCOVERY_TOP_K", "2"))
SUCCESS_REPEAT = int(os.environ.get("P2_DISCOVERY_SUCCESS_REPEAT", "2"))
MAX_STEPS = int(os.environ.get("P2_DISCOVERY_MAX_STEPS", "200"))
MIN_ANYGRASP_SEEDS = int(os.environ.get("P2_DISCOVERY_MIN_ANYGRASP_SEEDS", "4"))
MIN_ORACLE_SEEDS = int(os.environ.get("P2_DISCOVERY_MIN_ORACLE_SEEDS", "3"))

CONTROL_ROT_STD = np.array([0.0439326949, 0.0836294442, 0.0917678624], dtype=np.float32)
ROT_ACTION_CLIP = np.array([0.35, 0.45, 0.45], dtype=np.float32)
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float32)
WORLD_X = np.array([1.0, 0.0, 0.0], dtype=np.float32)


@dataclass(frozen=True)
class Lane:
    id: str
    rotation_mode: str
    state_mode: str
    visual_mode: str
    canonical_visual: bool
    train_candidate: bool
    note: str


LANES = [
    Lane(
        id="baseline_ref",
        rotation_mode="zero",
        state_mode="proxy",
        visual_mode="baseline",
        canonical_visual=True,
        train_candidate=False,
        note="Reference lane mirroring the current mainline for overnight calibration only.",
    ),
    Lane(
        id="rot_proxy",
        rotation_mode="aligned",
        state_mode="proxy",
        visual_mode="baseline",
        canonical_visual=True,
        train_candidate=True,
        note="Add rotation while preserving current proxy state and baseline visuals.",
    ),
    Lane(
        id="rot_ctrlstate",
        rotation_mode="aligned",
        state_mode="control_like",
        visual_mode="baseline",
        canonical_visual=True,
        train_candidate=True,
        note="Add rotation and switch to a control-like state representation.",
    ),
    Lane(
        id="rot_proxy_visualdiag",
        rotation_mode="aligned",
        state_mode="proxy",
        visual_mode="diagnostic_texture",
        canonical_visual=False,
        train_candidate=True,
        note="Win-hunting lane: rotation + proxy state + diagnostic-only texture enhancement.",
    ),
    Lane(
        id="rot_ctrlstate_visualdiag",
        rotation_mode="aligned",
        state_mode="control_like",
        visual_mode="diagnostic_texture",
        canonical_visual=False,
        train_candidate=True,
        note="Aggressive discovery lane: rotation + control-like state + diagnostic texture enhancement.",
    ),
]


def _safe_normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.asarray(fallback, dtype=np.float32).copy()
    return (vec / norm).astype(np.float32)


def _lane_root(lane: Lane) -> dict[str, Path]:
    base_art = ARTIFACT_DIR / EXPERIMENT_ID / lane.id
    base_out = OUTPUT_DIR / EXPERIMENT_ID / lane.id
    dataset_root = CAMPAIGN_DIR / "datasets_p2_alignment_discovery" / lane.id
    return {
        "artifact": base_art,
        "output": base_out,
        "dataset": dataset_root,
    }


def _write_summary(summary: dict[str, Any]) -> None:
    write_json_atomic(ARTIFACT_PATH, summary)
    lines = [
        "# P2 Overnight Alignment Discovery",
        "",
        f"Generated: {summary['generated_at']}",
        f"Status: {summary['status']}",
        "",
        "## Story",
        f"- user_story: {summary['story']['user_story']}",
        "",
        "## Acceptance Criteria",
    ]
    for item in summary["story"]["acceptance_criteria"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Lane Status"])
    for lane in summary.get("lanes", []):
        lines.append(
            f"- {lane['lane_id']}: prescreen_status={lane.get('prescreen_status')} prescreen_score={lane.get('prescreen_score')} trained={lane.get('trained', False)}"
        )
    if summary.get("leaderboard"):
        lines.extend(["", "## Leaderboard"])
        for item in summary["leaderboard"]:
            lines.append(
                f"- {item['lane_id']}: score={item.get('score')} frames={item.get('total_frames')} seeds={item.get('unique_successful_count')} finetuned_success={item.get('finetuned_success_count')} pretrained_success={item.get('pretrained_success_count')}"
            )
    if summary.get("next_recommendation"):
        lines.extend(["", "## Recommendation", f"- {summary['next_recommendation']}"])
    write_text_atomic(REPORT_PATH, "\n".join(lines).rstrip() + "\n")


def _make_story() -> dict[str, Any]:
    return {
        "user_story": "As the MINT/Infinigen mainline owner, I want an overnight automated lane-discovery run that searches for a trainable aligned data line instead of repeating the current failing single-line setup.",
        "acceptance_criteria": [
            "The runner explores at least 4 distinct lanes spanning rotation/state/visual hypotheses.",
            "Each lane produces a prescreen summary with successful unique seeds, total frames, and rollout length statistics.",
            "Only the top-K lanes are sent to training/eval to preserve overnight GPU budget.",
            "The runner emits a final leaderboard ranking lanes by data-line quality and held-out evaluation signal.",
            "If no lane improves the current ceiling, the final report says so explicitly instead of overstating success.",
        ],
    }


@contextlib.contextmanager
def _patched_lane(lane: Lane, lane_paths: dict[str, Path]) -> Iterator[None]:
    original_state = drm.DrawerRobotEnvMuJoCo._state_vector
    original_calibrate = drm.DrawerRobotEnvMuJoCo._calibrate_image
    original_build_rollout = mainline.build_robot_rollout

    def control_like_state(self: DrawerRobotEnvMuJoCo) -> np.ndarray:
        rotvec = R.from_quat(np.asarray(self.eef_quat, dtype=np.float32)).as_rotvec().astype(np.float32)
        grip = np.array([self.gripper_joint, self.gripper_joint], dtype=np.float32)
        return np.concatenate([self.eef_pos.astype(np.float32), rotvec, grip], axis=0).astype(np.float32)

    def diagnostic_calibrate(self: DrawerRobotEnvMuJoCo, image: np.ndarray, *, secondary: bool) -> np.ndarray:
        base = original_calibrate(self, image, secondary=secondary)
        lab = cv2.cvtColor(base, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        l_enh = clahe.apply(l_chan)
        merged = cv2.cvtColor(cv2.merge([l_enh, a_chan, b_chan]), cv2.COLOR_LAB2RGB)
        blur = cv2.GaussianBlur(merged, (0, 0), 1.1)
        sharpened = cv2.addWeighted(merged, 1.65, blur, -0.65, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _target_quat(current_pos: np.ndarray, target_pos: np.ndarray, motion_axis: np.ndarray) -> np.ndarray:
        z_axis = _safe_normalize(target_pos - current_pos, WORLD_UP)
        motion = _safe_normalize(motion_axis, WORLD_X)
        x_axis = motion - np.dot(motion, z_axis) * z_axis
        x_axis = _safe_normalize(x_axis, np.cross(WORLD_UP, z_axis) if abs(float(np.dot(WORLD_UP, z_axis))) < 0.95 else WORLD_X)
        if float(np.linalg.norm(x_axis)) < 1e-6:
            x_axis = WORLD_X.copy()
        y_axis = _safe_normalize(np.cross(z_axis, x_axis), np.cross(z_axis, WORLD_X))
        x_axis = _safe_normalize(np.cross(y_axis, z_axis), x_axis)
        rot = np.stack([x_axis, y_axis, z_axis], axis=1)
        return R.from_matrix(rot).as_quat().astype(np.float32)

    def _script_action_with_rotation(obs, env: DrawerRobotEnvMuJoCo, target_pos: np.ndarray, close: bool, speed: float, phase: str) -> np.ndarray:
        delta = np.zeros(7, dtype=np.float32)
        pos_err = target_pos - obs.eef_pos
        delta[:3] = np.clip((pos_err / max(drm.TRANSLATION_SCALE_M, 1e-6)) * float(speed), -1.0, 1.0)
        if lane.rotation_mode == "aligned":
            target_quat = _target_quat(obs.eef_pos, target_pos, env._motion_axis)
            current_rot = R.from_quat(np.asarray(obs.eef_quat, dtype=np.float32))
            target_rot = R.from_quat(target_quat)
            rotvec = (target_rot * current_rot.inv()).as_rotvec().astype(np.float32)
            phase_gain = 0.55 if phase in {"pregrasp", "contact"} else (0.85 if phase == "close" else 1.0)
            normalized = np.clip((rotvec / max(drm.ROTATION_SCALE_RAD, 1e-6)) * phase_gain, -ROT_ACTION_CLIP, ROT_ACTION_CLIP)
            delta[3:6] = normalized.astype(np.float32)
        delta[6] = -1.0 if close else 1.0
        return delta

    def build_robot_rollout_lane(
        seed: int,
        grasp_pose_world: np.ndarray,
        episode_index: int = 0,
        max_steps: int = 96,
        *,
        grasp_source: str = "anygrasp",
        grasp_score: float | None = None,
        image_size: int = 256,
        pull_open_fraction: float = drm.DEFAULT_PULL_OPEN_FRACTION,
    ) -> dict[str, Any]:
        env = DrawerRobotEnvMuJoCo(seed=seed, image_size=image_size, max_steps=max_steps)
        obs = env.reset()
        axis = env._motion_axis
        handle = env._handle_center_world()
        pregrasp_pose = drm._pose_from_point(handle, axis, offset=0.04, z_lift=0.03)
        grasp_pose = np.asarray(grasp_pose_world, dtype=np.float32).copy()
        grasp_pose[:3, 3] = handle
        open_fraction = float(np.clip(pull_open_fraction, 0.75, 0.95))

        images, images2, states, actions, rewards = [], [], [], [], []
        abs_drawer, next_drawer, attached_trace, handle_distance_trace = [], [], [], []
        phase_labels = []
        ever_attached = False
        attach_step = None
        success = False
        close_hold_steps = 0
        phase = "pregrasp"
        try:
            for step_idx in range(max_steps):
                images.append(obs.image.copy())
                images2.append(obs.image2.copy())
                states.append(obs.state.copy())
                abs_drawer.append(float(obs.drawer_fraction))

                handle = env._handle_center_world()
                pregrasp_target = handle - axis * 0.04 + np.array([0.0, 0.0, 0.03], dtype=np.float32)
                contact_target = handle + np.array([0.0, 0.0, 0.005], dtype=np.float32)
                pull_target = handle + axis * 0.03 + np.array([0.0, 0.0, 0.005], dtype=np.float32)
                retreat_target = handle + axis * 0.10 + np.array([0.0, 0.0, 0.05], dtype=np.float32)

                if env._attached and attach_step is None:
                    attach_step = step_idx
                if env._attached and phase in {"pregrasp", "contact", "close"}:
                    phase = "pull"
                elif phase == "pregrasp" and np.linalg.norm(obs.eef_pos - pregrasp_target) < 0.02:
                    phase = "contact"
                elif phase == "contact" and np.linalg.norm(obs.eef_pos - handle) < 0.03:
                    phase = "close"
                    close_hold_steps = 0
                elif phase == "close":
                    close_hold_steps += 1
                    if close_hold_steps >= 4:
                        phase = "pull" if env._attached else "contact"
                        close_hold_steps = 0
                elif phase == "pull" and not env._attached:
                    phase = "contact"
                elif phase == "pull" and obs.drawer_fraction >= open_fraction:
                    phase = "retreat"

                if phase == "pregrasp":
                    target_pos, close, speed = pregrasp_target, False, 0.65
                elif phase == "contact":
                    target_pos, close, speed = contact_target, False, 0.45
                elif phase == "close":
                    target_pos, close, speed = contact_target, True, 0.25
                elif phase == "pull":
                    target_pos, close, speed = pull_target, True, 0.30
                else:
                    target_pos, close, speed = retreat_target, False, 0.55

                action = _script_action_with_rotation(obs, env, target_pos, close, speed, phase)
                next_obs, reward, done, info = env.step(action)
                actions.append(action.copy())
                rewards.append(float(reward))
                phase_labels.append(phase)
                attached_trace.append(bool(info["attached"]))
                handle_distance_trace.append(float(info["dist_to_handle"]))
                next_drawer.append(float(next_obs.drawer_fraction))
                ever_attached = ever_attached or bool(info["attached"])
                obs = next_obs
                success = bool(info["is_success"])
                if done:
                    break
        finally:
            env.close()

        rollout = {
            "seed": seed,
            "episode_index": episode_index,
            "success": success,
            "steps": len(actions),
            "images": np.stack(images, axis=0).astype(np.uint8),
            "images2": np.stack(images2, axis=0).astype(np.uint8),
            "states": np.stack(states, axis=0).astype(np.float32),
            "actions": np.stack(actions, axis=0).astype(np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "absolute_drawer_fraction": np.asarray(abs_drawer, dtype=np.float32),
            "next_drawer_fractions": np.asarray(next_drawer, dtype=np.float32),
            "attached_trace": np.asarray(attached_trace, dtype=np.bool_),
            "handle_distance_trace": np.asarray(handle_distance_trace, dtype=np.float32),
            "phase_labels": np.asarray(phase_labels),
            "task": drm.DEFAULT_TASK,
            "grasp_source": grasp_source,
            "grasp_score": None if grasp_score is None else float(grasp_score),
            "ever_attached": ever_attached,
            "final_drawer_fraction": float(obs.drawer_fraction),
            "max_drawer_fraction": float(max(next_drawer or [0.0])),
            "pull_fraction_target": float(open_fraction),
            "attach_step": attach_step,
            "action_contract": {
                "translation_scale_m": drm.TRANSLATION_SCALE_M,
                "rotation_scale_rad": drm.ROTATION_SCALE_RAD,
                "rotation_mode": lane.rotation_mode,
                "state_mode": lane.state_mode,
                "visual_mode": lane.visual_mode,
                "backend": "mujoco",
                "robot_in_loop": True,
            },
        }
        return rollout

    try:
        if lane.state_mode == "control_like":
            drm.DrawerRobotEnvMuJoCo._state_vector = control_like_state
        if lane.visual_mode == "diagnostic_texture":
            drm.DrawerRobotEnvMuJoCo._calibrate_image = diagnostic_calibrate
        mainline.build_robot_rollout = build_robot_rollout_lane
        yield
    finally:
        drm.DrawerRobotEnvMuJoCo._state_vector = original_state
        drm.DrawerRobotEnvMuJoCo._calibrate_image = original_calibrate
        mainline.build_robot_rollout = original_build_rollout


@contextlib.contextmanager
def _configured_mainline(lane: Lane) -> Iterator[dict[str, Path]]:
    lane_paths = _lane_root(lane)
    for p in lane_paths.values():
        p.mkdir(parents=True, exist_ok=True)
    originals = {
        'TRAIN_SEEDS': copy.deepcopy(mainline.TRAIN_SEEDS),
        'HELDOUT_SEEDS': copy.deepcopy(mainline.HELDOUT_SEEDS),
        'MIN_ANYGRASP_SEEDS': mainline.MIN_ANYGRASP_SEEDS,
        'MIN_ORACLE_SEEDS': mainline.MIN_ORACLE_SEEDS,
        'SUCCESS_REPEAT': mainline.SUCCESS_REPEAT,
        'MAX_STEPS': mainline.MAX_STEPS,
        'M3_CACHE_DIR': mainline.M3_CACHE_DIR,
        'M4_ROLLOUT_ANY': mainline.M4_ROLLOUT_ANY,
        'M4_ROLLOUT_ORACLE': mainline.M4_ROLLOUT_ORACLE,
        'M4_ROLLOUT_LEARNING': mainline.M4_ROLLOUT_LEARNING,
        'DATASET_DIR': mainline.DATASET_DIR,
        'DATASET_REPO_ID': mainline.DATASET_REPO_ID,
        'M6_OUTPUT_DIR': mainline.M6_OUTPUT_DIR,
        'M7_OUTPUT_DIR': mainline.M7_OUTPUT_DIR,
        'ARTIFACT_DIR': mainline.ARTIFACT_DIR,
    }
    heldout = default_heldout_seeds()
    manifest = drawer_manifest()
    train_pool = [seed for seed in manifest['available_seeds'] if seed not in heldout]
    lane_art_root = lane_paths['artifact']
    lane_out_root = lane_paths['output']
    lane_dataset_root = lane_paths['dataset']
    dataset_repo_id = f"infinigen_drawer_{lane.id}"

    mainline.TRAIN_SEEDS = train_pool
    mainline.HELDOUT_SEEDS = heldout
    mainline.MIN_ANYGRASP_SEEDS = MIN_ANYGRASP_SEEDS
    mainline.MIN_ORACLE_SEEDS = MIN_ORACLE_SEEDS
    mainline.SUCCESS_REPEAT = SUCCESS_REPEAT
    mainline.MAX_STEPS = MAX_STEPS
    mainline.M3_CACHE_DIR = lane_art_root / 'm3_anygrasp'
    mainline.M4_ROLLOUT_ANY = lane_art_root / 'm4_any'
    mainline.M4_ROLLOUT_ORACLE = lane_art_root / 'm4_oracle'
    mainline.M4_ROLLOUT_LEARNING = lane_art_root / 'm4_learning'
    mainline.DATASET_DIR = lane_dataset_root
    mainline.DATASET_REPO_ID = dataset_repo_id
    mainline.M6_OUTPUT_DIR = lane_out_root / 'm6_train'
    mainline.M7_OUTPUT_DIR = lane_out_root / 'm7_eval'
    mainline.ARTIFACT_DIR = lane_art_root

    old_env = {k: os.environ.get(k) for k in ['MUJOCO_MINT_TRAIN_STEPS', 'MUJOCO_MINT_TRAIN_BATCH_SIZE', 'MUJOCO_MINT_TRAIN_SAVE_FREQ']}
    os.environ['MUJOCO_MINT_TRAIN_STEPS'] = str(TRAIN_STEPS)
    os.environ['MUJOCO_MINT_TRAIN_BATCH_SIZE'] = str(BATCH_SIZE)
    os.environ['MUJOCO_MINT_TRAIN_SAVE_FREQ'] = str(max(500, TRAIN_STEPS // 4))

    try:
        with _patched_lane(lane, lane_paths):
            yield lane_paths
    finally:
        for key, value in originals.items():
            setattr(mainline, key, value)
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _pack_dataset_for_lane(lane: Lane, lane_paths: dict[str, Path]) -> dict[str, Any]:
    rollout_paths = sorted(mainline.M4_ROLLOUT_LEARNING.glob('*.npz'))
    if not rollout_paths:
        return {'passed': False, 'error': 'No learning rollouts available', 'frame_count': 0, 'episode_count': 0}
    if lane_paths['dataset'].exists():
        shutil.rmtree(lane_paths['dataset'])
    payload = build_dataset_from_rollouts(
        rollout_paths=rollout_paths,
        dataset_root=lane_paths['dataset'],
        repo_id=mainline.DATASET_REPO_ID,
        image_size=mainline.IMAGE_SIZE,
    )
    integrity = payload['integrity']
    return {
        'passed': bool(payload['episode_count'] > 0 and integrity['passed']),
        'dataset_root': str(lane_paths['dataset']),
        'repo_id': mainline.DATASET_REPO_ID,
        'episode_count': int(payload['episode_count']),
        'frame_count': int(payload['frame_count']),
        'seed_coverage': payload['seed_coverage'],
        'integrity': integrity,
    }


def _train_lane(lane: Lane) -> dict[str, Any]:
    train_steps = int(os.environ['MUJOCO_MINT_TRAIN_STEPS'])
    batch_size = int(os.environ['MUJOCO_MINT_TRAIN_BATCH_SIZE'])
    save_freq = int(os.environ['MUJOCO_MINT_TRAIN_SAVE_FREQ'])
    log_path = mainline.ARTIFACT_DIR / 'm6_train.log'
    if mainline.M6_OUTPUT_DIR.exists():
        shutil.rmtree(mainline.M6_OUTPUT_DIR)
    cmd = [
        'lerobot-train',
        f'--dataset.repo_id={mainline.DATASET_REPO_ID}',
        f'--dataset.root={mainline.DATASET_DIR}',
        '--policy.type=mint',
        f'--policy.repo_id={mainline.DATASET_REPO_ID}_ft',
        '--policy.push_to_hub=false',
        f'--output_dir={mainline.M6_OUTPUT_DIR}',
        f'--job_name={lane.id}_ft',
        f'--policy.pretrained_path={mainline.MINT_CKPT}',
        f'--policy.vqvae_name_or_path={mainline.TOKENIZER_PATH}',
        '--policy.compile_model=false',
        '--policy.gradient_checkpointing=true',
        '--policy.dtype=bfloat16',
        f'--steps={train_steps}',
        f'--save_freq={save_freq}',
        f'--batch_size={batch_size}',
        '--policy.device=cuda',
    ]
    start = time.time()
    with log_path.open('w') as handle:
        proc = subprocess.run(cmd, cwd=str(mainline.PROJECT_ROOT), stdout=handle, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start
    checkpoint_path = None
    for candidate in sorted(mainline.M6_OUTPUT_DIR.glob('checkpoints/*/pretrained_model')):
        checkpoint_path = candidate
    log_tail = log_path.read_text(errors='ignore')[-6000:] if log_path.exists() else ''
    return {
        'passed': proc.returncode == 0 and checkpoint_path is not None,
        'returncode': proc.returncode,
        'elapsed_sec': round(elapsed, 1),
        'steps_requested': train_steps,
        'batch_size': batch_size,
        'checkpoint_path': str(checkpoint_path) if checkpoint_path else None,
        'train_output_dir': str(mainline.M6_OUTPUT_DIR),
        'stdout_tail': log_tail,
    }


def _lane_score(m4: dict[str, Any], pack: dict[str, Any]) -> float:
    unique = len(m4.get('successful_learning_seeds', []))
    frames = int(pack.get('frame_count', 0) or 0)
    source_bonus = 1.0 if m4.get('learning_source') == 'oracle_handle' else 0.0
    avg_steps = 0.0
    records = m4.get('learning_records', [])
    if records:
        avg_steps = frames / max(len(records), 1)
    return round(unique * 20.0 + frames * 0.02 + avg_steps * 2.0 + source_bonus, 3)


def _prescreen_lane(lane: Lane) -> dict[str, Any]:
    lane_paths = _lane_root(lane)
    for p in [lane_paths['artifact'], lane_paths['output']]:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
    with _configured_mainline(lane):
        m3 = mainline.run_m3_anygrasp_gate()
        m4 = mainline.run_m4_robot_rollout_gate() if m3.get('passed') else {'passed': False, 'error': 'm3 failed'}
        pack = _pack_dataset_for_lane(lane, lane_paths) if m4.get('passed') else {'passed': False, 'error': 'm4 failed', 'frame_count': 0, 'episode_count': 0}
    score = _lane_score(m4, pack) if m4.get('passed') and pack.get('passed') else -1.0
    prescreen = {
        'lane_id': lane.id,
        'lane': asdict(lane),
        'prescreen_status': 'passed' if m4.get('passed') and pack.get('passed') else 'failed',
        'prescreen_score': score,
        'm3': m3,
        'm4': m4,
        'dataset': pack,
        'unique_successful_count': len(m4.get('successful_learning_seeds', [])),
        'total_frames': int(pack.get('frame_count', 0) or 0),
        'avg_episode_length_steps': float((pack.get('frame_count', 0) or 0) / max(int(pack.get('episode_count', 0) or 1), 1)),
        'trained': False,
    }
    write_json_atomic(lane_paths['artifact'] / 'prescreen.json', prescreen)
    return prescreen


def _train_and_eval_lane(lane: Lane, lane_record: dict[str, Any]) -> dict[str, Any]:
    lane_paths = _lane_root(lane)
    with _configured_mainline(lane):
        train_result = _train_lane(lane)
        eval_result = mainline.run_m7_mint_eval_gate(train_result) if train_result.get('passed') else {'passed': False, 'error': 'train gate failed'}
    out = copy.deepcopy(lane_record)
    out['trained'] = True
    out['train_result'] = train_result
    out['eval_result'] = eval_result
    out['finetuned_success_count'] = int(eval_result.get('finetuned', {}).get('success_count', 0) or 0)
    out['pretrained_success_count'] = int(eval_result.get('pretrained', {}).get('success_count', 0) or 0)
    fin_records = eval_result.get('finetuned', {}).get('records', []) or []
    pre_records = eval_result.get('pretrained', {}).get('records', []) or []
    out['finetuned_max_drawer_mean'] = float(np.mean([x.get('max_drawer_fraction', 0.0) for x in fin_records])) if fin_records else 0.0
    out['pretrained_max_drawer_mean'] = float(np.mean([x.get('max_drawer_fraction', 0.0) for x in pre_records])) if pre_records else 0.0
    write_json_atomic(lane_paths['artifact'] / 'train_eval.json', out)
    return out


def _recommendation(leaderboard: list[dict[str, Any]]) -> str:
    if not leaderboard:
        return 'No lane produced a trainable aligned data line; revisit rotation/state/visual hypotheses before more overnight training.'
    best = leaderboard[0]
    if best.get('finetuned_success_count', 0) > best.get('pretrained_success_count', 0):
        return f"Best lane is {best['lane_id']} with finetuned_success={best.get('finetuned_success_count')} > pretrained_success={best.get('pretrained_success_count')}; use it as the next supervised mainline candidate."
    if best.get('total_frames', 0) > 100:
        return f"Best lane is {best['lane_id']} and it lifted the data ceiling to {best.get('total_frames')} frames, but held-out success is still weak. Use this lane for the next targeted fix cycle."
    return f"Best lane is {best['lane_id']}, but no lane yet produced convincing held-out gains. The overnight run still narrowed the highest-value lane for the next cycle."


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    story = _make_story()
    write_json_atomic(SPEC_PATH, {'generated_at': now_iso(), 'experiment_id': EXPERIMENT_ID, 'story': story, 'lanes': [asdict(l) for l in LANES]})
    summary = {
        'experiment_id': EXPERIMENT_ID,
        'generated_at': now_iso(),
        'status': 'running',
        'story': story,
        'lanes': [],
        'leaderboard': [],
        'train_steps': TRAIN_STEPS,
        'top_k': TOP_K,
        'success_repeat': SUCCESS_REPEAT,
        'max_steps': MAX_STEPS,
        'started_at': now_iso(),
    }
    _write_summary(summary)

    prescreened: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        for lane in LANES:
            try:
                record = _prescreen_lane(lane)
            except Exception as exc:  # noqa: BLE001
                record = {
                    'lane_id': lane.id,
                    'lane': asdict(lane),
                    'prescreen_status': 'failed',
                    'prescreen_score': -1.0,
                    'error': f'{type(exc).__name__}: {exc}',
                    'traceback': traceback.format_exc(),
                    'trained': False,
                }
                failures.append({'lane_id': lane.id, 'error': record['error']})
            prescreened.append(record)
            summary['lanes'] = prescreened
            _write_summary(summary)

        trainable = [x for x in prescreened if x.get('prescreen_status') == 'passed' and x['lane'].get('train_candidate', False)]
        trainable.sort(key=lambda item: (item.get('prescreen_score', -1.0), item.get('unique_successful_count', 0), item.get('total_frames', 0)), reverse=True)
        selected = trainable[:TOP_K]

        trained_records = []
        for item in selected:
            lane = next(l for l in LANES if l.id == item['lane_id'])
            try:
                trained = _train_and_eval_lane(lane, item)
            except Exception as exc:  # noqa: BLE001
                trained = copy.deepcopy(item)
                trained['trained'] = True
                trained['train_result'] = {'passed': False, 'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()}
                trained['eval_result'] = {'passed': False, 'error': 'train failed'}
            trained_records.append(trained)
            for idx, existing in enumerate(summary['lanes']):
                if existing['lane_id'] == trained['lane_id']:
                    summary['lanes'][idx] = trained
                    break
            _write_summary(summary)

        ranked = sorted(
            summary['lanes'],
            key=lambda item: (
                int(item.get('finetuned_success_count', 0) or 0),
                float(item.get('finetuned_max_drawer_mean', 0.0) or 0.0),
                float(item.get('prescreen_score', -1.0) or -1.0),
            ),
            reverse=True,
        )
        summary['leaderboard'] = ranked
        summary['failures'] = failures
        summary['next_recommendation'] = _recommendation(ranked)
        summary['completed_at'] = now_iso()
        summary['status'] = 'completed'
        _write_summary(summary)
        upsert_evidence(
            EVIDENCE_ID,
            EXPERIMENT_ID,
            'overnight_alignment_discovery',
            ARTIFACT_PATH,
            'Overnight lane-discovery matrix exploring rotation/state/visual hypotheses for the Infinigen MuJoCo mainline and automatically training the top-ranked lanes.',
            verified=True,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        summary['status'] = 'failed'
        summary['completed_at'] = now_iso()
        summary['error'] = f'{type(exc).__name__}: {exc}'
        summary['traceback'] = traceback.format_exc()
        _write_summary(summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

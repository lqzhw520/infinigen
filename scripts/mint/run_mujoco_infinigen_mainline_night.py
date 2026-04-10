#!/usr/bin/env python3
"""Canonical stage-gated night-runner for the true Infinigen MuJoCo mainline."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from mint_eval_patches import apply as _apply_mint_patches

_apply_mint_patches()

from anygrasp_helper import stage_detection_assets
from dataset_builder import build_dataset_from_rollouts
from drawer_proxy import DrawerProxyEnv
from drawer_robot_env_mujoco import (
    DRAWER_SUCCESS_FRACTION,
    DrawerRobotEnvMuJoCo,
    build_oracle_grasp_pose,
    build_robot_rollout,
    drawer_assets_available,
    drawer_manifest,
    save_robot_rollout,
)
from mujoco_mainline_common import (
    ARTIFACT_DIR,
    CONTROL_BASELINE_REF,
    CURRENT_TRUTH_PATH,
    DATASET_DIR,
    DATASET_REPO_ID,
    HISTORICAL_LINES_IGNORED,
    MUJOCO_ENV_REF,
    NIGHT_ARTIFACT_PATH,
    NIGHT_REPORT_PATH,
    NIGHT_STATUS_PATH,
    OUTPUT_DIR,
    RUNNER_ID,
    RUNNER_PATH,
    STAGE_ORDER,
    append_report_line,
    default_heldout_seeds,
    default_train_seeds,
    ensure_layout,
    load_json,
    make_night_status,
    now_iso,
    register_evidence,
    sync_sovereign_failure,
    sync_sovereign_running,
    sync_sovereign_success,
    write_night_outputs,
)
from mint_common import write_json_atomic

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
MINT_CKPT = PROJECT_ROOT / "external" / "MINT" / "checkpoints" / "MINT-libero"
TOKENIZER_PATH = PROJECT_ROOT / "external" / "MINT" / "checkpoints" / "MINT-tokenizer-libero"
M3_CACHE_DIR = ARTIFACT_DIR / "m3_mujoco_anygrasp"
M4_ROLLOUT_ANY = ARTIFACT_DIR / "m4_rollouts_anygrasp_mujoco"
M4_ROLLOUT_ORACLE = ARTIFACT_DIR / "m4_rollouts_oracle_mujoco"
M4_ROLLOUT_LEARNING = ARTIFACT_DIR / "m4_rollouts_learning_mujoco"
M6_OUTPUT_DIR = OUTPUT_DIR / "m6_mujoco_mint_train"
M7_OUTPUT_DIR = OUTPUT_DIR / "m7_mujoco_eval"
GRASPNET_PYTHON = Path("/root/anaconda3/envs/graspnet/bin/python")
ANYGRASP_SUBPROCESS = PROJECT_ROOT / "scripts" / "mint" / "run_mujoco_anygrasp_subprocess.py"
ALIGNMENT_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "mint" / "run_p1e_mujoco_infinigen_alignment_audit.py"
ALIGNMENT_AUDIT_JSON = ARTIFACT_DIR / "p1e_mujoco_infinigen_alignment_audit.json"
ALIGNMENT_AUDIT_MD = OUTPUT_DIR / "p1e_mujoco_infinigen_alignment_audit.md"

TRAIN_SEEDS = default_train_seeds()
HELDOUT_SEEDS = default_heldout_seeds()
IMAGE_SIZE = 256
MAX_STEPS = 96
MIN_ANYGRASP_SEEDS = int(os.environ.get("MUJOCO_MIN_ANYGRASP_SEEDS", "4"))
MIN_ORACLE_SEEDS = int(os.environ.get("MUJOCO_MIN_ORACLE_SEEDS", "3"))
SUCCESS_REPEAT = int(os.environ.get("MUJOCO_SUCCESS_REPEAT", "8"))


def _artifact_path(stage: str) -> Path:
    return ARTIFACT_DIR / f"{stage}.json"


def _save_stage_result(stage: str, payload: dict[str, Any]) -> Path:
    payload["gate"] = stage
    payload["timestamp"] = now_iso()
    path = _artifact_path(stage)
    write_json_atomic(path, payload)
    return path


def _run_alignment_audit() -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(ALIGNMENT_AUDIT_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    payload = {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "json_path": str(ALIGNMENT_AUDIT_JSON),
        "md_path": str(ALIGNMENT_AUDIT_MD),
        "passed": proc.returncode == 0 and ALIGNMENT_AUDIT_JSON.exists(),
    }
    if ALIGNMENT_AUDIT_JSON.exists():
        try:
            audit = json.loads(ALIGNMENT_AUDIT_JSON.read_text())
            payload["finding_ids"] = [item.get("id") for item in audit.get("findings", [])]
            payload["recommendations"] = audit.get("recommendations", [])
        except json.JSONDecodeError:
            payload["passed"] = False
    return payload


def _read_current_next_action() -> dict[str, Any]:
    current_truth = load_json(CURRENT_TRUTH_PATH, {})
    return current_truth.get("current", {}).get("next_action", {})


def _stage_summary(stage: str, payload: dict[str, Any]) -> str:
    if payload.get("passed"):
        return f"{stage} passed"
    if payload.get("error"):
        return f"{stage} failed: {payload['error']}"
    return f"{stage} failed"


def _write_video(path: Path, frames: list[np.ndarray], fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    imageio.mimsave(path, frames, fps=fps)


def _load_policy_bundle(path: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot_policy_mint.modeling_mint import MINTPolicy

    dataset = LeRobotDataset(repo_id=DATASET_REPO_ID, root=DATASET_DIR, revision="main")
    policy = MINTPolicy.from_pretrained(
        str(path), local_files_only=True, dataset_stats=dataset.meta.stats
    )
    policy.eval()
    if hasattr(policy.model, "direct_grip_head"):
        policy.model.direct_grip_head = policy.model.direct_grip_head.to(dtype=torch.float32)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(path), dataset_stats=dataset.meta.stats
    )
    return policy, preprocessor, postprocessor, dataset


def _obs_to_batch(obs) -> dict[str, Any]:
    return {
        "observation.images.image": torch.from_numpy(obs.image).permute(2, 0, 1).to(torch.float32) / 255.0,
        "observation.images.image2": torch.from_numpy(obs.image2).permute(2, 0, 1).to(torch.float32) / 255.0,
        "observation.state": torch.from_numpy(obs.state.astype(np.float32)),
        "task": obs.task,
    }


def run_m0_authority_preflight() -> dict[str, Any]:
    next_action = _read_current_next_action()
    guard_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "mint" / "sovereign_mainline_guard.py"),
        "--runner",
        str(RUNNER_PATH),
    ]
    result = subprocess.run(guard_cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    guard_payload = {}
    if result.stdout:
        try:
            guard_payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            guard_payload = {"stdout": result.stdout[-4000:]}
    passed = result.returncode == 0 and "MUJOCO" in next_action.get("type", "")
    return {
        "passed": passed,
        "guard_returncode": result.returncode,
        "guard_payload": guard_payload,
        "canonical_next_action_at_launch": next_action,
        "runner": str(RUNNER_PATH),
        "backend": "mujoco",
        "robot_in_loop": True,
        "canonical": True,
        "control_baseline_ref": CONTROL_BASELINE_REF,
        "mujoco_env_ref": MUJOCO_ENV_REF,
        "historical_lines_ignored": HISTORICAL_LINES_IGNORED,
        "error": None if passed else "Authority preflight failed",
    }


def run_m1_proxy_asset_smoke() -> dict[str, Any]:
    seeds = TRAIN_SEEDS[:3]
    manifest = drawer_manifest()
    if not drawer_assets_available(seeds):
        return {
            "passed": False,
            "canonical": False,
            "backend": "mujoco_proxy",
            "robot_in_loop": False,
            "seeds": seeds,
            "manifest": manifest,
            "error": "Missing drawer URDF assets for proxy smoke",
        }
    records = []
    passed = True
    for seed in seeds:
        env = DrawerProxyEnv(seed=seed, image_size=IMAGE_SIZE, max_steps=4)
        try:
            obs = env.reset()
            action = np.zeros(7, dtype=np.float32)
            next_obs, reward, done, info = env.step(action)
            records.append(
                {
                    "seed": seed,
                    "image_shape": list(obs.image.shape),
                    "image2_shape": list(obs.image2.shape),
                    "state_shape": list(obs.state.shape),
                    "reward": float(reward),
                    "done": bool(done),
                    "normalized_joint_position": float(info["normalized_joint_position"]),
                }
            )
            passed = passed and obs.image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
        finally:
            env.close()
    return {
        "passed": passed,
        "canonical": False,
        "backend": "mujoco_proxy",
        "robot_in_loop": False,
        "seeds": seeds,
        "manifest": manifest,
        "records": records,
    }


def run_m2_true_robot_env_gate() -> dict[str, Any]:
    seeds = TRAIN_SEEDS[:3]
    records = []
    passed = True
    for seed in seeds:
        env = DrawerRobotEnvMuJoCo(seed=seed, image_size=IMAGE_SIZE, max_steps=8)
        try:
            obs = env.reset()
            zero_action = np.zeros(7, dtype=np.float32)
            obs2, reward, done, info = env.step(zero_action)
            any_payload = env.anygrasp_payload(num_points=1024)
            record = {
                "seed": seed,
                "image_shape": list(obs.image.shape),
                "image2_shape": list(obs.image2.shape),
                "state_shape": list(obs.state.shape),
                "drawer_fraction": float(obs.drawer_fraction),
                "next_drawer_fraction": float(obs2.drawer_fraction),
                "reward": float(reward),
                "done": bool(done),
                "anygrasp_points": int(len(any_payload["pc"])),
                "handle_center_world": np.asarray(any_payload["handle_center_world"]).round(4).tolist(),
            }
            records.append(record)
            passed = passed and obs.image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
            passed = passed and obs.state.shape == (8,)
            passed = passed and len(any_payload["pc"]) >= 512
        finally:
            env.close()
    return {
        "passed": passed,
        "backend": "mujoco",
        "robot_in_loop": True,
        "canonical": True,
        "records": records,
        "task": "open the middle drawer of the cabinet",
        "image_size": IMAGE_SIZE,
        "max_steps": 8,
    }


def run_m3_anygrasp_gate() -> dict[str, Any]:
    setup = stage_detection_assets()
    if not setup.get("passed"):
        return {
            "passed": False,
            "backend": "mujoco",
            "robot_in_loop": True,
            "canonical": True,
            "error": setup.get("error", "AnyGrasp setup failed"),
            "setup": setup,
        }

    M3_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    passed_count = 0
    for seed in TRAIN_SEEDS:
        env = DrawerRobotEnvMuJoCo(seed=seed, image_size=IMAGE_SIZE, max_steps=8)
        try:
            env.reset()
            payload = env.anygrasp_payload(num_points=4096)
            payload_path = M3_CACHE_DIR / f"seed_{seed:03d}_payload.npz"
            cache_path = M3_CACHE_DIR / f"seed_{seed:03d}.json"
            np.savez_compressed(
                payload_path,
                pc=payload["pc"].astype(np.float32),
                colors=payload["colors"].astype(np.float32),
                limits=payload["limits"].astype(np.float32),
                handle_center_world=np.asarray(payload["handle_center_world"], dtype=np.float32),
                drawer_motion_axis=np.asarray(payload["drawer_motion_axis"], dtype=np.float32),
                drawer_aabb_world=np.asarray(payload["drawer_aabb_world"], dtype=np.float32),
            )
            proc = subprocess.run(
                [
                    str(GRASPNET_PYTHON),
                    str(ANYGRASP_SUBPROCESS),
                    "--payload",
                    str(payload_path),
                    "--output",
                    str(cache_path),
                    "--max-candidates",
                    "10",
                ],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )
            if cache_path.exists():
                cache_payload = json.loads(cache_path.read_text())
            else:
                cache_payload = {
                    "passed": False,
                    "error": "AnyGrasp subprocess did not produce output",
                }
            top_candidates = cache_payload.get("top_candidates", [])
            selected = cache_payload.get("selected_grasp")
            seed_passed = bool(cache_payload.get("passed")) and selected is not None and proc.returncode == 0
            passed_count += int(seed_passed)
            cache_payload.update(
                {
                    "seed": seed,
                    "passed": seed_passed,
                    "timestamp": now_iso(),
                    "subprocess_returncode": proc.returncode,
                    "subprocess_stdout_tail": proc.stdout[-2000:],
                    "subprocess_stderr_tail": proc.stderr[-2000:],
                    "payload_path": str(payload_path),
                }
            )
            write_json_atomic(M3_CACHE_DIR / f"seed_{seed:03d}.json", cache_payload)
            records.append(
                {
                    "seed": seed,
                    "passed": seed_passed,
                    "top_score": float(top_candidates[0]["score"]) if top_candidates else 0.0,
                    "distance_to_handle": float(selected["distance_to_handle"]) if selected else None,
                    "subprocess_returncode": proc.returncode,
                }
            )
        finally:
            env.close()
    return {
        "passed": passed_count >= MIN_ANYGRASP_SEEDS,
        "backend": "mujoco",
        "robot_in_loop": True,
        "canonical": True,
        "setup": setup,
        "min_anygrasp_seeds": MIN_ANYGRASP_SEEDS,
        "passed_count": passed_count,
        "records": records,
        "cache_dir": str(M3_CACHE_DIR),
    }


def run_m4_robot_rollout_gate() -> dict[str, Any]:
    for path in [M4_ROLLOUT_ANY, M4_ROLLOUT_ORACLE, M4_ROLLOUT_LEARNING]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    any_success = 0
    oracle_success = 0
    any_records = []
    oracle_records = []
    for seed in TRAIN_SEEDS:
        cache_path = M3_CACHE_DIR / f"seed_{seed:03d}.json"
        if not cache_path.exists():
            any_records.append({"seed": seed, "passed": False, "error": "Missing AnyGrasp cache"})
            oracle_records.append({"seed": seed, "passed": False, "error": "Missing AnyGrasp cache"})
            continue
        cache = json.loads(cache_path.read_text())
        selected = cache.get("selected_grasp")
        if not selected:
            any_records.append({"seed": seed, "passed": False, "error": "No selected grasp"})
            oracle_records.append({"seed": seed, "passed": False, "error": "No selected grasp"})
            continue

        any_pose = np.asarray(selected["pose_world"], dtype=np.float32)
        oracle_pose = build_oracle_grasp_pose(any_pose, np.asarray(cache["handle_center_world"], dtype=np.float32))
        any_rollout = build_robot_rollout(
            seed=seed,
            grasp_pose_world=any_pose,
            episode_index=0,
            max_steps=MAX_STEPS,
            grasp_source="anygrasp",
            grasp_score=selected.get("score"),
            image_size=IMAGE_SIZE,
        )
        oracle_rollout = build_robot_rollout(
            seed=seed,
            grasp_pose_world=oracle_pose,
            episode_index=0,
            max_steps=MAX_STEPS,
            grasp_source="oracle_handle",
            grasp_score=selected.get("score"),
            image_size=IMAGE_SIZE,
        )
        any_path = M4_ROLLOUT_ANY / f"seed_{seed:03d}_episode_00.npz"
        oracle_path = M4_ROLLOUT_ORACLE / f"seed_{seed:03d}_episode_00.npz"
        save_robot_rollout(any_path, any_rollout)
        save_robot_rollout(oracle_path, oracle_rollout)
        any_success += int(any_rollout["success"])
        oracle_success += int(oracle_rollout["success"])
        any_records.append({"seed": seed, "passed": bool(any_rollout["success"]), "path": str(any_path)})
        oracle_records.append({"seed": seed, "passed": bool(oracle_rollout["success"]), "path": str(oracle_path)})

    learning_source = None
    learning_dir = None
    candidate_sources: list[tuple[str, Path, int]] = []
    if any_success >= MIN_ANYGRASP_SEEDS:
        candidate_sources.append(("anygrasp", M4_ROLLOUT_ANY, any_success))
    if oracle_success >= MIN_ORACLE_SEEDS:
        candidate_sources.append(("oracle_handle", M4_ROLLOUT_ORACLE, oracle_success))
    if candidate_sources:
        learning_source, learning_dir, _best_success = max(candidate_sources, key=lambda item: (item[2], item[0] == "oracle_handle"))

    copied = 0
    learning_records: list[dict[str, Any]] = []
    if learning_dir is not None:
        source_records = any_records if learning_source == "anygrasp" else oracle_records
        for record in source_records:
            if not record.get("passed"):
                continue
            file_path = Path(record["path"])
            meta_path = file_path.with_suffix(".json")
            for repeat_idx in range(SUCCESS_REPEAT):
                dest_name = (
                    file_path.stem
                    if repeat_idx == 0
                    else f"{file_path.stem}_rep{repeat_idx:02d}"
                )
                dest_path = M4_ROLLOUT_LEARNING / f"{dest_name}.npz"
                shutil.copy2(file_path, dest_path)
                if meta_path.exists():
                    dest_meta = M4_ROLLOUT_LEARNING / f"{dest_name}.json"
                    meta = json.loads(meta_path.read_text())
                    meta["repeat_index"] = repeat_idx
                    meta["source_rollout_path"] = str(file_path)
                    write_json_atomic(dest_meta, meta)
                copied += 1
                learning_records.append(
                    {
                        "seed": record["seed"],
                        "source_path": str(file_path),
                        "repeat_index": repeat_idx,
                        "dest_path": str(dest_path),
                    }
                )

    return {
        "passed": learning_source is not None and copied > 0,
        "backend": "mujoco",
        "robot_in_loop": True,
        "canonical": True,
        "anygrasp_successful_seeds": any_success,
        "oracle_successful_seeds": oracle_success,
        "min_anygrasp_seeds": MIN_ANYGRASP_SEEDS,
        "min_oracle_seeds": MIN_ORACLE_SEEDS,
        "anygrasp_records": any_records,
        "oracle_records": oracle_records,
        "learning_source": learning_source,
        "learning_rollout_dir": str(M4_ROLLOUT_LEARNING) if learning_source else None,
        "copied_learning_files": copied,
        "successful_learning_seeds": sorted({int(item["seed"]) for item in learning_records}),
        "successful_learning_rollouts": len(learning_records),
        "success_repeat": SUCCESS_REPEAT,
        "learning_records": learning_records,
    }


def run_m5_dataset_pack_gate() -> dict[str, Any]:
    rollout_paths = sorted(M4_ROLLOUT_LEARNING.glob("*.npz"))
    if not rollout_paths:
        return {
            "passed": False,
            "backend": "mujoco",
            "robot_in_loop": True,
            "canonical": True,
            "error": "No learning rollouts available for packing",
        }
    if DATASET_DIR.exists():
        archive_root = ARTIFACT_DIR / f"m5_dataset_archive_{int(time.time())}"
        archive_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(DATASET_DIR), str(archive_root / "dataset"))
    payload = build_dataset_from_rollouts(
        rollout_paths=rollout_paths,
        dataset_root=DATASET_DIR,
        repo_id=DATASET_REPO_ID,
        image_size=IMAGE_SIZE,
    )
    integrity = payload["integrity"]
    return {
        "passed": bool(payload["episode_count"] > 0 and integrity["passed"]),
        "backend": "mujoco",
        "robot_in_loop": True,
        "canonical": True,
        "dataset_root": str(DATASET_DIR),
        "repo_id": DATASET_REPO_ID,
        "episode_count": payload["episode_count"],
        "frame_count": payload["frame_count"],
        "seed_coverage": payload["seed_coverage"],
        "task_coverage": payload["task_coverage"],
        "source_files": payload["source_files"],
        "integrity": integrity,
        "alignment_audit": _run_alignment_audit(),
    }


def run_m6_mint_train_gate() -> dict[str, Any]:
    train_steps = int(os.environ.get("MUJOCO_MINT_TRAIN_STEPS", "4000"))
    batch_size = int(os.environ.get("MUJOCO_MINT_TRAIN_BATCH_SIZE", "4"))
    save_freq = int(os.environ.get("MUJOCO_MINT_TRAIN_SAVE_FREQ", str(max(500, train_steps // 4))))
    log_path = ARTIFACT_DIR / "m6_mint_train.log"
    if M6_OUTPUT_DIR.exists():
        shutil.rmtree(M6_OUTPUT_DIR)
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        f"--dataset.root={DATASET_DIR}",
        "--policy.type=mint",
        f"--policy.repo_id={DATASET_REPO_ID}_mujoco_mainline_ft",
        "--policy.push_to_hub=false",
        f"--output_dir={M6_OUTPUT_DIR}",
        "--job_name=mujoco_mainline_ft",
        f"--policy.pretrained_path={MINT_CKPT}",
        f"--policy.vqvae_name_or_path={TOKENIZER_PATH}",
        "--policy.compile_model=false",
        "--policy.gradient_checkpointing=true",
        "--policy.dtype=bfloat16",
        f"--steps={train_steps}",
        f"--save_freq={save_freq}",
        f"--batch_size={batch_size}",
        "--policy.device=cuda",
    ]
    start = time.time()
    with log_path.open("w") as handle:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=handle, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start
    checkpoint_path = None
    for candidate in sorted(M6_OUTPUT_DIR.glob("checkpoints/*/pretrained_model")):
        checkpoint_path = candidate
    log_tail = log_path.read_text(errors="ignore")[-6000:] if log_path.exists() else ""
    return {
        "passed": proc.returncode == 0 and checkpoint_path is not None,
        "backend": "mujoco",
        "robot_in_loop": True,
        "canonical": True,
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "steps_requested": train_steps,
        "batch_size": batch_size,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "train_output_dir": str(M6_OUTPUT_DIR),
        "stdout_tail": log_tail,
    }


def _rollout_policy(seed: int, policy_bundle, *, kind: str) -> dict[str, Any]:
    policy, pre, post, _dataset = policy_bundle
    env = DrawerRobotEnvMuJoCo(seed=seed, image_size=IMAGE_SIZE, max_steps=MAX_STEPS)
    obs = env.reset()
    frames = []
    drawer_trace = []
    success = False
    steps = 0
    try:
        for _ in range(MAX_STEPS):
            frames.append(obs.image.copy())
            batch = _obs_to_batch(obs)
            processed = pre(batch)
            with torch.inference_mode():
                action = policy.select_action(processed)
            action = post(action)
            np_action = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
            obs, reward, done, info = env.step(np_action)
            drawer_trace.append(float(info["drawer_fraction"]))
            success = bool(info["is_success"])
            steps += 1
            if done:
                break
    finally:
        env.close()
    out_dir = M7_OUTPUT_DIR / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"seed_{seed:03d}.mp4"
    _write_video(video_path, frames)
    return {
        "seed": seed,
        "success": success,
        "steps": steps,
        "final_drawer_fraction": float(drawer_trace[-1]) if drawer_trace else 0.0,
        "max_drawer_fraction": float(max(drawer_trace)) if drawer_trace else 0.0,
        "video_path": str(video_path),
    }


def run_m7_mint_eval_gate(train_result: dict[str, Any]) -> dict[str, Any]:
    if not train_result.get("checkpoint_path"):
        return {
            "passed": False,
            "backend": "mujoco",
            "robot_in_loop": True,
            "canonical": True,
            "error": "Missing fine-tuned checkpoint path from m6",
        }
    pretrained_bundle = _load_policy_bundle(MINT_CKPT)
    finetuned_bundle = _load_policy_bundle(Path(train_result["checkpoint_path"]))
    pretrained_records = [_rollout_policy(seed, pretrained_bundle, kind="pretrained") for seed in HELDOUT_SEEDS]
    finetuned_records = [_rollout_policy(seed, finetuned_bundle, kind="finetuned") for seed in HELDOUT_SEEDS]
    pretrained_success = sum(int(item["success"]) for item in pretrained_records)
    finetuned_success = sum(int(item["success"]) for item in finetuned_records)
    return {
        "passed": True,
        "backend": "mujoco",
        "robot_in_loop": True,
        "canonical": True,
        "control_baseline_ref": CONTROL_BASELINE_REF,
        "heldout_seeds": HELDOUT_SEEDS,
        "pretrained": {
            "success_count": pretrained_success,
            "seed_count": len(HELDOUT_SEEDS),
            "records": pretrained_records,
        },
        "finetuned": {
            "success_count": finetuned_success,
            "seed_count": len(HELDOUT_SEEDS),
            "records": finetuned_records,
        },
        "delta_success": finetuned_success - pretrained_success,
        "benchmark": "upstream external/MINT@4eab579 control",
    }


def main() -> int:
    ensure_layout()
    report_lines: list[str] = []
    summary: dict[str, Any] = {
        "runner_id": RUNNER_ID,
        "started_at": now_iso(),
        "stages": [],
        "control_baseline_ref": CONTROL_BASELINE_REF,
        "mujoco_env_ref": MUJOCO_ENV_REF,
    }

    stage_handlers = {
        "m0_authority_preflight": run_m0_authority_preflight,
        "m1_proxy_asset_smoke": run_m1_proxy_asset_smoke,
        "m2_true_robot_env_gate": run_m2_true_robot_env_gate,
        "m3_anygrasp_gate": run_m3_anygrasp_gate,
        "m4_robot_rollout_gate": run_m4_robot_rollout_gate,
        "m5_dataset_pack_gate": run_m5_dataset_pack_gate,
        "m6_mint_train_gate": run_m6_mint_train_gate,
    }

    canonical_next_action = _read_current_next_action()
    status = make_night_status(
        current_stage="m0_authority_preflight",
        status="running",
        canonical_next_action_at_launch=canonical_next_action,
    )
    write_night_outputs(status, report_lines, summary)

    train_result: dict[str, Any] | None = None

    for stage in STAGE_ORDER:
        status = make_night_status(
            current_stage=stage,
            status="running",
            canonical_next_action_at_launch=canonical_next_action,
        )
        sync_sovereign_running(stage, status)
        handler = stage_handlers.get(stage)
        try:
            if handler is None:
                if stage == "m7_mint_eval_gate":
                    payload = run_m7_mint_eval_gate(train_result or {})
                else:
                    payload = {"passed": False, "error": f"No handler for {stage}"}
            else:
                payload = handler()
        except Exception as exc:
            payload = {
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "backend": "mujoco",
                "robot_in_loop": stage != "m1_proxy_asset_smoke",
                "canonical": stage != "m1_proxy_asset_smoke",
            }
        artifact_path = _save_stage_result(stage, payload)
        summary["stages"].append({"stage": stage, "artifact": str(artifact_path), "passed": bool(payload.get("passed"))})

        if payload.get("passed"):
            status = make_night_status(
                current_stage=stage,
                status="running" if stage != STAGE_ORDER[-1] else "completed",
                canonical_next_action_at_launch=canonical_next_action,
                details={"artifact": str(artifact_path.relative_to(PROJECT_ROOT))},
            )
            register_evidence(stage, artifact_path, _stage_summary(stage, payload), verified=True)
            append_report_line(report_lines, f"{stage}: PASS — {artifact_path.relative_to(PROJECT_ROOT)}")
            if stage == "m6_mint_train_gate":
                train_result = payload
            write_night_outputs(status, report_lines, summary)
            continue

        reason = payload.get("error", _stage_summary(stage, payload))
        status = make_night_status(
            current_stage=stage,
            status="failed",
            canonical_next_action_at_launch=canonical_next_action,
            failed_stage=stage,
            details={"artifact": str(artifact_path.relative_to(PROJECT_ROOT)), "reason": reason},
        )
        sync_sovereign_failure(stage, reason, status)
        append_report_line(report_lines, f"{stage}: FAIL — {reason}")
        summary["failed_stage"] = stage
        summary["completed_at"] = now_iso()
        write_night_outputs(status, report_lines, summary)
        return 1

    summary["completed_at"] = now_iso()
    status = make_night_status(
        current_stage="m7_mint_eval_gate",
        status="completed",
        canonical_next_action_at_launch=canonical_next_action,
        details={"artifact": str(NIGHT_ARTIFACT_PATH.relative_to(PROJECT_ROOT))},
    )
    sync_sovereign_success(
        "m7_mint_eval_gate",
        status,
        "True MuJoCo Infinigen night-runner completed; review held-out pretrained vs finetuned results.",
    )
    append_report_line(report_lines, "Night runner completed all canonical MuJoCo stages.")
    write_night_outputs(status, report_lines, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

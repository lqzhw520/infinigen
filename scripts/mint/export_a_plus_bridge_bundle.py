#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MINT_SCRIPTS = PROJECT_ROOT / "scripts" / "mint"
MINT_POLICY_SRC = PROJECT_ROOT / "external" / "MINT" / "lerobot_policy_mint" / "src"
sys.path.insert(0, str(MINT_SCRIPTS))
sys.path.insert(0, str(MINT_POLICY_SRC))

from a_plus_bridge_common import (  # noqa: E402
    BUNDLE_SCHEMA_VERSION,
    BRIDGE_NAME,
    ROBOT_JOINT_NAMES,
    git_head,
    make_bridge_run_id,
    phase_label_from_info,
    solve_robot_qpos_trace,
    write_json,
)
from drawer_robot_env_mujoco import DrawerEnvContractConfig, DrawerRobotEnvMuJoCo  # noqa: E402
from evaluate_mint_drawer_campaign_mujoco import (  # noqa: E402
    DEFAULT_EVAL_IMAGE_SIZE,
    MINT_CKPT,
    _load_policy,
    _obs_to_batch,
)
from merged_model_builder import MergedModelBuilder  # noqa: E402
from root_cause_controller import RootCauseController  # noqa: E402
from run_v84_p0_canary_train_probe import CANARY_DATASET_ROOT, CANARY_REPO_ID, _checkpoint_path  # noqa: E402
from v13_audit_common import current_repo_identity, load_json  # noqa: E402

ACTIVE_PLAN_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "mint"
    / "mint_drawer_v1"
    / "artifacts"
    / "active_tiny_retrain_plan.json"
)


def _contract() -> DrawerEnvContractConfig:
    payload = dict(
        RootCauseController()._frozen_matrix_contracts_v5_pro()["V1cT2S3"]["env_contract_config"]
    )
    return DrawerEnvContractConfig(**payload)


def _policy_bundle(variant: str, checkpoint_step: int):
    if variant == "pretrained_reference":
        ckpt = MINT_CKPT
    elif variant == "finetuned_checkpoint":
        ckpt = str(_checkpoint_path(checkpoint_step))
    else:
        raise ValueError(f"Unsupported variant: {variant}")
    return _load_policy(str(ckpt), CANARY_DATASET_ROOT, CANARY_REPO_ID)


def _copy_drawer_seed_assets(seed: int, destination_root: Path) -> None:
    source_seed_dir = MergedModelBuilder(seed=seed)._seed_dir()
    dst = destination_root / str(seed)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(source_seed_dir, dst)


def _rollout_proxy_trace(seed: int, policy_bundle, max_steps: int) -> dict[str, Any]:
    env = DrawerRobotEnvMuJoCo(
        seed=seed,
        image_size=int(DEFAULT_EVAL_IMAGE_SIZE),
        max_steps=int(max_steps),
        contract=_contract(),
    )
    obs = env.reset()
    policy, pre, post = policy_bundle

    eef_pos_trace: list[np.ndarray] = []
    eef_quat_trace: list[np.ndarray] = []
    gripper_joint_trace: list[float] = []
    drawer_qpos_trace: list[np.ndarray] = []
    drawer_fraction_trace: list[float] = []
    phase_labels: list[str] = []
    attach_eligible_trace: list[bool] = []
    attached_trace: list[bool] = []
    phase_locked_trace: list[bool] = []
    raw_actions: list[np.ndarray] = []

    try:
        eef_pos_trace.append(np.asarray(obs.eef_pos, dtype=np.float32).copy())
        eef_quat_trace.append(np.asarray(obs.eef_quat, dtype=np.float32).copy())
        gripper_joint_trace.append(float(env.gripper_joint))
        drawer_qpos_trace.append(env.data.qpos[:2].astype(np.float32).copy())
        drawer_fraction_trace.append(float(env._drawer_fraction()))
        phase_labels.append("000:reset")
        attach_eligible_trace.append(False)
        attached_trace.append(False)
        phase_locked_trace.append(False)

        for step in range(1, int(max_steps) + 1):
            batch = _obs_to_batch(obs)
            processed = pre(batch)
            with torch.inference_mode():
                action = policy.select_action(processed)
            action = post(action)
            action = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
            raw_actions.append(action.copy())

            obs, _, done, info = env.step(action)
            eef_pos_trace.append(np.asarray(obs.eef_pos, dtype=np.float32).copy())
            eef_quat_trace.append(np.asarray(obs.eef_quat, dtype=np.float32).copy())
            gripper_joint_trace.append(float(env.gripper_joint))
            drawer_qpos_trace.append(env.data.qpos[:2].astype(np.float32).copy())
            drawer_fraction_trace.append(float(info.get("drawer_fraction", env._drawer_fraction())))
            phase_labels.append(phase_label_from_info(step, info))
            attach_eligible_trace.append(bool(info.get("attach_eligible", False)))
            attached_trace.append(bool(info.get("attached", False)))
            phase_locked_trace.append(bool(info.get("phase_locked", False)))
            if done:
                break
    finally:
        env.close()

    return {
        "eef_pos_trace": np.asarray(eef_pos_trace, dtype=np.float32),
        "eef_quat_trace": np.asarray(eef_quat_trace, dtype=np.float32),
        "gripper_joint_trace": np.asarray(gripper_joint_trace, dtype=np.float32),
        "drawer_qpos_trace": np.asarray(drawer_qpos_trace, dtype=np.float32),
        "drawer_fraction_trace": np.asarray(drawer_fraction_trace, dtype=np.float32),
        "attach_eligible_trace": np.asarray(attach_eligible_trace, dtype=np.bool_),
        "attached_trace": np.asarray(attached_trace, dtype=np.bool_),
        "phase_locked_trace": np.asarray(phase_locked_trace, dtype=np.bool_),
        "phase_labels": phase_labels,
        "raw_actions": np.asarray(raw_actions, dtype=np.float32),
    }


def export_bundle(
    bundle_root: Path,
    seeds: list[int],
    variants: list[str],
    checkpoint_step: int,
    max_steps: int,
) -> dict[str, Any]:
    ident = current_repo_identity()
    active_plan = load_json(ACTIVE_PLAN_PATH, {})
    bridge_run_id = make_bridge_run_id("a_plus_bridge", ident["working_head_commit"])
    bundle_root.mkdir(parents=True, exist_ok=True)

    drawer_asset_root = bundle_root / "drawer_assets"
    drawer_asset_root.mkdir(parents=True, exist_ok=True)
    episodes_dir = bundle_root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bridge_name": BRIDGE_NAME,
        "bridge_run_id": bridge_run_id,
        "authority_repo_head": ident["working_head_commit"],
        "authority_run_instance_id": active_plan.get("run_instance_id"),
        "authority_execution_scope": active_plan.get("execution_scope"),
        "authority_working_head_commit": active_plan.get("working_head_commit"),
        "bridge_code_head": git_head(PROJECT_ROOT),
        "checkpoint_step": int(checkpoint_step),
        "asset_family": "Infinigen drawer",
        "visual_parity_target": "LIBERO-style presentation",
        "claim_bearing": False,
        "drawer_asset_root_rel": "drawer_assets",
        "episodes": [],
    }

    for seed in seeds:
        _copy_drawer_seed_assets(seed, drawer_asset_root)

    for variant in variants:
        policy_bundle = _policy_bundle(variant, checkpoint_step)
        for seed in seeds:
            trace = _rollout_proxy_trace(seed, policy_bundle, max_steps=max_steps)

            builder = MergedModelBuilder(seed=seed)
            xml, assets, _, _ = builder.build()
            import mujoco

            model = mujoco.MjModel.from_xml_string(xml, assets)
            robot_qpos_trace, ik_report = solve_robot_qpos_trace(
                model,
                trace["eef_pos_trace"],
                trace["eef_quat_trace"],
                initial_qpos=builder.robot_init_qpos,
            )

            episode_key = f"{variant}_seed_{seed:03d}"
            episode_dir = episodes_dir / episode_key
            episode_dir.mkdir(parents=True, exist_ok=True)
            trace_npz = episode_dir / "trace.npz"
            frame_labels_json = episode_dir / "frame_labels.json"
            np.savez_compressed(
                trace_npz,
                robot_qpos=robot_qpos_trace.astype(np.float32),
                gripper_joint=trace["gripper_joint_trace"].astype(np.float32),
                drawer_qpos=trace["drawer_qpos_trace"].astype(np.float32),
                drawer_fraction=trace["drawer_fraction_trace"].astype(np.float32),
                attach_eligible=trace["attach_eligible_trace"].astype(np.bool_),
                attached=trace["attached_trace"].astype(np.bool_),
                phase_locked=trace["phase_locked_trace"].astype(np.bool_),
                eef_pos=trace["eef_pos_trace"].astype(np.float32),
                eef_quat=trace["eef_quat_trace"].astype(np.float32),
            )
            write_json(frame_labels_json, {"phase_labels": trace["phase_labels"]})

            manifest["episodes"].append(
                {
                    "episode_key": episode_key,
                    "seed": int(seed),
                    "episode_id": 0,
                    "policy_variant": variant,
                    "checkpoint_step": int(checkpoint_step),
                    "trace_npz_rel": str(trace_npz.relative_to(bundle_root)),
                    "frame_labels_rel": str(frame_labels_json.relative_to(bundle_root)),
                    "frame_count": int(trace["eef_pos_trace"].shape[0]),
                    "ik_report": ik_report,
                    "asset_roots": builder.asset_roots,
                }
            )

    write_json(bundle_root / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export A+ local-render bridge bundle")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["pretrained_reference", "finetuned_checkpoint"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = export_bundle(
        bundle_root=args.bundle_root,
        seeds=[int(seed) for seed in args.seeds],
        variants=[str(variant) for variant in args.variants],
        checkpoint_step=int(args.checkpoint_step),
        max_steps=int(args.max_steps),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

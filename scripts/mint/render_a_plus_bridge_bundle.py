#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from a_plus_bridge_common import (
    BUNDLE_SCHEMA_VERSION,
    REPORT_CAMERA_NAME,
    DRAWER_JOINT_NAMES,
    GRIPPER_JOINT_NAMES,
    ROBOT_JOINT_NAMES,
    git_head,
    gripper_scalar_to_finger_qpos,
    read_json,
    select_still_indices,
    write_json,
)
from merged_model_builder import MergedModelBuilder


def _report_scene_option() -> mujoco.MjvOption:
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    option.geomgroup[3:] = 0
    option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_SELECT] = False
    return option


def _qpos_indices(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    return np.asarray(
        [
            model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in joint_names
        ],
        dtype=np.int32,
    )


def _render_frame(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_name: str,
    caption: str,
) -> np.ndarray:
    renderer.update_scene(data, camera=camera_name, scene_option=_report_scene_option())
    rgb = renderer.render()
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (img.width, 42)], fill=(0, 0, 0, 200))
    draw.text((12, 10), caption, fill=(255, 255, 255))
    return np.asarray(img, dtype=np.uint8)


def _render_episode(
    bundle_root: Path,
    episode: dict[str, Any],
    output_root: Path,
    camera_name: str,
    render_mode: str,
) -> dict[str, Any]:
    seed = int(episode["seed"])
    os.environ["MINT_DRAWER_ASSET_ROOT"] = str(bundle_root / "drawer_assets")
    builder = MergedModelBuilder(seed=seed)
    xml, assets, _, _ = builder.build()
    model = mujoco.MjModel.from_xml_string(xml, assets)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=960)

    robot_qpos_idx = _qpos_indices(model, ROBOT_JOINT_NAMES)
    drawer_qpos_idx = _qpos_indices(model, DRAWER_JOINT_NAMES)
    gripper_qpos_idx = _qpos_indices(model, GRIPPER_JOINT_NAMES)

    trace = np.load(bundle_root / episode["trace_npz_rel"])
    labels = read_json(bundle_root / episode["frame_labels_rel"])["phase_labels"]

    frames: list[np.ndarray] = []
    replayed_drawer: list[np.ndarray] = []
    replayed_robot: list[np.ndarray] = []
    replayed_fingers: list[np.ndarray] = []

    for idx in range(trace["robot_qpos"].shape[0]):
        data.qpos[robot_qpos_idx] = trace["robot_qpos"][idx]
        data.qpos[drawer_qpos_idx] = trace["drawer_qpos"][idx]
        left_q, right_q = gripper_scalar_to_finger_qpos(float(trace["gripper_joint"][idx]))
        data.qpos[gripper_qpos_idx[0]] = left_q
        data.qpos[gripper_qpos_idx[1]] = right_q
        mujoco.mj_forward(model, data)
        replayed_robot.append(np.asarray(data.qpos[robot_qpos_idx], dtype=np.float32).copy())
        replayed_drawer.append(np.asarray(data.qpos[drawer_qpos_idx], dtype=np.float32).copy())
        replayed_fingers.append(np.asarray(data.qpos[gripper_qpos_idx], dtype=np.float32).copy())
        if render_mode == "video_and_stills":
            frames.append(
                _render_frame(
                    renderer,
                    data,
                    camera_name,
                    f"{episode['episode_key']}  frame={idx:03d}  phase={labels[idx]}",
                )
            )

    replayed_robot_arr = np.asarray(replayed_robot, dtype=np.float32)
    replayed_drawer_arr = np.asarray(replayed_drawer, dtype=np.float32)
    replayed_fingers_arr = np.asarray(replayed_fingers, dtype=np.float32)

    still_indices = select_still_indices(labels, trace["drawer_qpos"])
    episode_root = output_root / episode["episode_key"]
    episode_root.mkdir(parents=True, exist_ok=True)

    if render_mode == "stills_only":
        frames = []
        for idx in sorted(set(still_indices.values())):
            data.qpos[robot_qpos_idx] = trace["robot_qpos"][idx]
            data.qpos[drawer_qpos_idx] = trace["drawer_qpos"][idx]
            left_q, right_q = gripper_scalar_to_finger_qpos(float(trace["gripper_joint"][idx]))
            data.qpos[gripper_qpos_idx[0]] = left_q
            data.qpos[gripper_qpos_idx[1]] = right_q
            mujoco.mj_forward(model, data)
            frames.append(
                _render_frame(
                    renderer,
                    data,
                    camera_name,
                    f"{episode['episode_key']}  frame={idx:03d}  phase={labels[idx]}",
                )
            )
        frame_lookup = {idx: frame for idx, frame in zip(sorted(set(still_indices.values())), frames)}
    else:
        frame_lookup = {idx: frames[idx] for idx in sorted(set(still_indices.values()))}

    still_outputs = {}
    for key, idx in still_indices.items():
        png_path = episode_root / f"{key}.png"
        Image.fromarray(frame_lookup[idx]).save(png_path)
        still_outputs[key] = str(png_path)

    mp4_path = episode_root / "report_primary.mp4"
    if render_mode == "video_and_stills":
        iio.imwrite(mp4_path, np.asarray(frames, dtype=np.uint8), fps=12)

    renderer.close()

    parity = {
        "frame_count": int(trace["robot_qpos"].shape[0]),
        "robot_qpos_max_abs_diff": float(np.max(np.abs(replayed_robot_arr - trace["robot_qpos"]))),
        "drawer_qpos_max_abs_diff": float(np.max(np.abs(replayed_drawer_arr - trace["drawer_qpos"]))),
        "gripper_left_max_abs_diff": float(
            np.max(
                np.abs(
                    replayed_fingers_arr[:, 0]
                    - np.asarray(
                        [gripper_scalar_to_finger_qpos(float(v))[0] for v in trace["gripper_joint"]],
                        dtype=np.float32,
                    )
                )
            )
        ),
        "gripper_right_max_abs_diff": float(
            np.max(
                np.abs(
                    replayed_fingers_arr[:, 1]
                    - np.asarray(
                        [gripper_scalar_to_finger_qpos(float(v))[1] for v in trace["gripper_joint"]],
                        dtype=np.float32,
                    )
                )
            )
        ),
    }

    return {
        "episode_key": episode["episode_key"],
        "seed": seed,
        "policy_variant": episode["policy_variant"],
        "checkpoint_step": int(episode["checkpoint_step"]),
        "camera_preset": camera_name,
        "mp4_path": str(mp4_path) if render_mode == "video_and_stills" else None,
        "stills": still_outputs,
        "parity": parity,
    }


def render_bundle(
    bundle_root: Path,
    output_root: Path,
    camera_name: str,
    render_mode: str,
) -> dict[str, Any]:
    manifest = read_json(bundle_root / "manifest.json")
    output_root.mkdir(parents=True, exist_ok=True)

    episode_results = []
    for episode in manifest["episodes"]:
        episode_results.append(
            _render_episode(bundle_root, episode, output_root, camera_name, render_mode)
        )

    render_manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bridge_name": manifest["bridge_name"],
        "bridge_run_id": manifest["bridge_run_id"],
        "authority_repo_head": manifest["authority_repo_head"],
        "authority_run_instance_id": manifest["authority_run_instance_id"],
        "authority_execution_scope": manifest["authority_execution_scope"],
        "bridge_code_head": git_head(),
        "render_backend": "mujoco_local_mac",
        "camera_preset": camera_name,
        "visual_parity_target": "LIBERO-style presentation",
        "asset_family": "Infinigen drawer",
        "claim_bearing": False,
        "episodes": episode_results,
    }
    write_json(output_root / "render_manifest.json", render_manifest)
    return render_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay and render an A+ bridge bundle locally")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--camera-preset", default=REPORT_CAMERA_NAME)
    parser.add_argument(
        "--render-mode",
        choices=["stills_only", "video_and_stills"],
        default="video_and_stills",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = render_bundle(
        bundle_root=args.bundle_root,
        output_root=args.output_root,
        camera_name=str(args.camera_preset),
        render_mode=str(args.render_mode),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

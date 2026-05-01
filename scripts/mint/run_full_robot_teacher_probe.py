#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from drawer_robot_env_mujoco import (
    DrawerRobotEnvMuJoCoLibero,
    _json_ready,
    _pose_from_point,
    save_robot_rollout,
)
from drawer_robot_env_mujoco import (
    build_robot_rollout as build_full_robot_rollout,
)
from merged_model_builder import MergedModelBuilder
from mint_common import (
    ARTIFACT_DIR,
    CAMPAIGN_DIR,
    PROJECT_ROOT,
    load_json,
    write_json_atomic,
)

try:
    from tiny_retrain_gate_utils import execution_lineage_payload
except ImportError:

    def execution_lineage_payload() -> dict[str, Any]:
        repo = _repo_identity()
        return {
            "lineage_source": "run_full_robot_teacher_probe_fallback",
            "repo_branch": repo.get("branch"),
            "repo_head_commit": repo.get("working_head_commit"),
            "vendor_head_commit": repo.get("vendor_head_commit"),
        }


DrawerRobotEnvMuJoCoLibero._MERGED_BUILDER_CLASS = MergedModelBuilder

AUTOPILOT_DIR = CAMPAIGN_DIR / "autopilot"
HARNESS_STATE_PATH = AUTOPILOT_DIR / "harness_state.json"
HARNESS_STATE_V11_PATH = AUTOPILOT_DIR / "harness_state_v11.json"
SOVEREIGN_SNAPSHOT_PATH = AUTOPILOT_DIR / "sovereign_snapshot.json"
SOVEREIGN_SNAPSHOT_V11_PATH = AUTOPILOT_DIR / "sovereign_snapshot_v11.json"
FULL_ROBOT_PROBE_ROOT = ARTIFACT_DIR / "full_robot_teacher_probe"
FULL_ROBOT_PROBE_LATEST_PATH = ARTIFACT_DIR / "full_robot_teacher_probe_latest.json"
ACTIVE_PLAN_PATH = ARTIFACT_DIR / "active_tiny_retrain_plan.json"
HARNESS_SCRIPT_DIR = (
    PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1" / "scripts" / "harness"
)

if str(HARNESS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_SCRIPT_DIR))
import truth_backend  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _repo_identity() -> dict[str, str]:
    import subprocess

    def git(args: list[str]) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_ROOT, text=True
        ).strip()

    return {
        "branch": git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "working_head_commit": git(["rev-parse", "HEAD"]),
        "vendor_head_commit": git(
            ["-C", str(PROJECT_ROOT / "external" / "MINT"), "rev-parse", "HEAD"]
        ),
    }


def _build_grasp_pose(seed: int, control_mode: str) -> np.ndarray:
    env = DrawerRobotEnvMuJoCoLibero(
        seed=seed,
        image_size=96,
        max_steps=2,
        contract=None,
    )
    try:
        env.reset()
        handle, _ = env._interaction_handle_target_world()
        axis = env._motion_axis.astype(np.float32)
        return _pose_from_point(handle, axis, offset=0.0, z_lift=0.005)
    finally:
        env.close()


V11_MIN_FINGERPAD_CONTACT_FRAMES = 3
V11_MIN_FINGERPAD_CONTACT_CONSECUTIVE_FRAMES = 2
V11_REQUIRED_AUDIT_STATE_SOURCE = "post_step_next_states"
V11_NON_TARGET_PENETRATION_EPS_M = -1.0e-4


def _v11_max_consecutive(values: list[bool]) -> int:
    best = 0
    run = 0
    for value in values:
        if bool(value):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return int(best)


def _v11_body_has_ancestor(model: Any, body_id: int, ancestor_name: str) -> bool:
    current = int(body_id)
    while current >= 0:
        if str(model.body(current).name or "") == ancestor_name:
            return True
        parent = int(model.body_parentid[current])
        if parent == current:
            break
        current = parent
    return False


def _v11_geom_belongs_to_robot(model: Any, geom_id: int) -> bool:
    body_id = int(model.geom_bodyid[int(geom_id)])
    name = str(model.geom(int(geom_id)).name or "")
    return (
        _v11_body_has_ancestor(model, body_id, "base")
        or name == "hand_collision"
        or name.startswith("link")
        or "wrist" in name
        or ("finger" in name and "collision" in name)
    )


def _v11_forbidden_target_contact_class(model: Any, geom_id: int) -> str:
    name = str(model.geom(int(geom_id)).name or "")
    if name == "hand_collision":
        return "hand_collision"
    if "wrist" in name:
        return "wrist"
    if name.startswith("link"):
        return "link"
    return "body"


def _v11_geom_belongs_to_drawer_or_cabinet(model: Any, geom_id: int) -> bool:
    name = str(model.geom(int(geom_id)).name or "")
    body_id = int(model.geom_bodyid[int(geom_id)])
    return (
        name.startswith("drawer_")
        or name.startswith("reference_cabinet")
        or _v11_body_has_ancestor(model, body_id, "drawer_base")
    )


def _v11_resolve_geom_ids(model: Any, names: list[str]) -> list[int]:
    import mujoco

    ids: list[int] = []
    for name in names:
        gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, str(name or "")))
        if gid >= 0:
            ids.append(gid)
    return sorted(dict.fromkeys(ids))


def _v11_joint_adrs(model: Any, names: list[str]) -> list[int]:
    import mujoco

    out: list[int] = []
    for name in names:
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name or "")))
        if jid >= 0:
            out.append(int(model.jnt_qposadr[jid]))
    return out


def _v11_joint_names(model: Any, prefix: str) -> list[str]:
    return [
        str(model.joint(i).name or "")
        for i in range(int(model.njnt))
        if str(model.joint(i).name or "").startswith(prefix)
    ]


def _v11_gripper_qpos_from_scalar(model: Any, joint_name: str, scalar: float) -> float:
    import mujoco

    jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
    if jid < 0:
        return 0.0
    low = float(model.jnt_range[jid, 0])
    high = float(model.jnt_range[jid, 1])
    aperture = float(np.clip((float(scalar) + 0.042) / 0.043, 0.0, 1.0))
    return float(
        np.clip((high if abs(high) >= abs(low) else low) * aperture, low, high)
    )


def _v11_right_hand_inside_cabinet(
    model: Any, data: Any, scene_contract: dict[str, Any]
) -> bool:
    import mujoco

    layout = dict(scene_contract.get("layout", {}) or {})
    if not layout:
        return False
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand"))
    if body_id < 0:
        return False
    center = np.asarray(layout.get("cabinet_pos", [0.0, 0.0, 0.0]), dtype=np.float32)
    half = np.asarray(
        layout.get("cabinet_half_size", [0.0, 0.0, 0.0]), dtype=np.float32
    )
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(half)):
        return False
    hand_pos = np.asarray(data.xpos[body_id], dtype=np.float32)
    return bool(np.all(hand_pos >= center - half) and np.all(hand_pos <= center + half))


def _v11_post_step_replay_audit(seed: int, rollout: dict[str, Any]) -> dict[str, Any]:
    import mujoco

    if "next_states" not in rollout:
        return {
            "audit_version": "v11_post_step_replay_fingerpad_v1",
            "state_source": "missing_post_step_next_states",
            "audit_state_source": "missing_post_step_next_states",
            "required_audit_state_source": V11_REQUIRED_AUDIT_STATE_SOURCE,
            "frame_count": 0,
            "post_step_replay_audit_passed": False,
            "failure_reasons": ["missing_post_step_next_states"],
        }
    state_source = V11_REQUIRED_AUDIT_STATE_SOURCE
    states = np.asarray(rollout.get("next_states", []), dtype=np.float32)
    if states.ndim != 2 or states.shape[0] == 0:
        return {
            "audit_version": "v11_post_step_replay_fingerpad_v1",
            "state_source": state_source,
            "audit_state_source": state_source,
            "required_audit_state_source": V11_REQUIRED_AUDIT_STATE_SOURCE,
            "frame_count": 0,
            "post_step_replay_audit_passed": False,
            "failure_reasons": ["missing_post_step_next_states"],
        }
    xml, assets, scene_metadata, _ = MergedModelBuilder(seed=int(seed)).build()
    model = mujoco.MjModel.from_xml_string(xml, assets)
    data = mujoco.MjData(model)
    contract = dict(
        rollout.get("task_object_contract", {})
        or scene_metadata.get("task_object_contract", {})
        or {}
    )
    scene_contract = dict(
        contract.get("scene_contract", {})
        or scene_metadata.get("scene_contract", {})
        or {}
    )
    spec = dict(rollout.get("state_spec", {}) or {})
    drawer_names = list(spec.get("drawer_joint_names", []) or []) or _v11_joint_names(
        model, "drawer_slider_"
    )
    robot_names = list(spec.get("robot_joint_names", []) or []) or [
        f"joint{i}" for i in range(1, 8)
    ]
    gripper_names = list(spec.get("gripper_joint_names", []) or []) or _v11_joint_names(
        model, "finger_joint"
    )
    drawer_adrs = _v11_joint_adrs(model, drawer_names)
    robot_adrs = _v11_joint_adrs(model, robot_names)
    gripper_adrs = _v11_joint_adrs(model, gripper_names)
    n_drawer = len(drawer_names)
    n_robot = len(robot_names)
    required = n_drawer + n_robot + 1
    if states.shape[1] < required:
        return {
            "audit_version": "v11_post_step_replay_fingerpad_v1",
            "state_source": state_source,
            "audit_state_source": state_source,
            "required_audit_state_source": V11_REQUIRED_AUDIT_STATE_SOURCE,
            "frame_count": int(states.shape[0]),
            "post_step_replay_audit_passed": False,
            "failure_reasons": [
                f"state_width_{states.shape[1]}_lt_required_{required}"
            ],
        }
    target_names = (
        list(contract.get("target_handle_collision_geom_names") or [])
        + list(contract.get("target_handle_geom_names") or [])
        + list(contract.get("target_handle_visual_geom_names") or [])
    )
    target_ids = set(_v11_resolve_geom_ids(model, target_names))
    if not target_ids and contract.get("target_index") is not None:
        idx = int(contract.get("target_index"))
        target_ids = set(
            _v11_resolve_geom_ids(
                model, [f"drawer_handle_collision_{idx}", f"drawer_handle_visual_{idx}"]
            )
        )
    all_handle_ids = {
        i
        for i in range(int(model.ngeom))
        if str(model.geom(i).name or "").startswith("drawer_handle_")
    }
    non_target_handle_ids = all_handle_ids.difference(target_ids)
    fingerpad_ids = {
        i
        for i in range(int(model.ngeom))
        if "pad_collision" in str(model.geom(i).name or "")
    }
    fingertip_ids = {
        i
        for i in range(int(model.ngeom))
        if "fingertip" in str(model.geom(i).name or "")
        or "tip_collision" in str(model.geom(i).name or "")
    }
    finger_collision_ids = {
        i
        for i in range(int(model.ngeom))
        if "finger" in str(model.geom(i).name or "")
        and "collision" in str(model.geom(i).name or "")
    }
    hand_body_ids = {
        i
        for i in range(int(model.ngeom))
        if str(model.geom(i).name or "") == "hand_collision"
        or str(model.geom(i).name or "").startswith("link")
        or "wrist" in str(model.geom(i).name or "")
    }
    target_any_trace: list[bool] = []
    target_fingerpad_trace: list[bool] = []
    target_finger_collision_trace: list[bool] = []
    target_hand_body_trace: list[bool] = []
    non_target_pen_trace: list[bool] = []
    wrist_inside_trace: list[bool] = []
    target_samples: list[dict[str, Any]] = []
    non_target_samples: list[dict[str, Any]] = []
    non_target_min_dist = None
    non_target_peak_count = 0
    for frame_idx, state in enumerate(states):
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        for i, adr in enumerate(drawer_adrs):
            data.qpos[int(adr)] = float(state[i])
        for i, adr in enumerate(robot_adrs):
            data.qpos[int(adr)] = float(state[n_drawer + i])
        gripper_scalar = float(state[n_drawer + n_robot])
        for name, adr in zip(gripper_names, gripper_adrs):
            data.qpos[int(adr)] = _v11_gripper_qpos_from_scalar(
                model, name, gripper_scalar
            )
        mujoco.mj_forward(model, data)
        frame_target_any = False
        frame_fingerpad = False
        frame_finger_collision = False
        frame_hand_body = False
        frame_pen_count = 0
        for contact_idx in range(int(data.ncon)):
            contact = data.contact[contact_idx]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            n1 = str(model.geom(g1).name or f"geom_{g1}")
            n2 = str(model.geom(g2).name or f"geom_{g2}")
            if g1 in target_ids or g2 in target_ids:
                frame_target_any = True
                other = g2 if g1 in target_ids else g1
                if other in fingerpad_ids:
                    frame_fingerpad = True
                    cls = "fingerpad"
                elif other in fingertip_ids:
                    frame_fingerpad = True
                    cls = "fingertip"
                elif other in hand_body_ids or _v11_geom_belongs_to_robot(model, other):
                    frame_hand_body = True
                    frame_finger_collision = bool(other in finger_collision_ids)
                    cls = _v11_forbidden_target_contact_class(model, other)
                else:
                    cls = "other"
                if len(target_samples) < 16:
                    target_samples.append(
                        {
                            "frame": int(frame_idx),
                            "geom1": n1,
                            "geom2": n2,
                            "dist_m": float(contact.dist),
                            "contact_class": cls,
                        }
                    )
                continue
            robot_drawer_pair = (
                _v11_geom_belongs_to_robot(model, g1)
                and _v11_geom_belongs_to_drawer_or_cabinet(model, g2)
            ) or (
                _v11_geom_belongs_to_robot(model, g2)
                and _v11_geom_belongs_to_drawer_or_cabinet(model, g1)
            )
            if not robot_drawer_pair:
                continue
            dist = float(contact.dist)
            if non_target_min_dist is None or dist < float(non_target_min_dist):
                non_target_min_dist = dist
            non_target_handle_pair = (
                g1 in non_target_handle_ids or g2 in non_target_handle_ids
            )
            if dist < V11_NON_TARGET_PENETRATION_EPS_M:
                frame_pen_count += 1
            if len(non_target_samples) < 16 and (dist < 0.0 or non_target_handle_pair):
                non_target_samples.append(
                    {
                        "frame": int(frame_idx),
                        "geom1": n1,
                        "geom2": n2,
                        "dist_m": dist,
                        "non_target_handle_pair": bool(non_target_handle_pair),
                    }
                )
        target_any_trace.append(bool(frame_target_any))
        target_fingerpad_trace.append(bool(frame_fingerpad))
        target_finger_collision_trace.append(bool(frame_finger_collision))
        target_hand_body_trace.append(bool(frame_hand_body))
        non_target_pen_trace.append(bool(frame_pen_count > 0))
        wrist_inside_trace.append(
            _v11_right_hand_inside_cabinet(model, data, scene_contract)
        )
        non_target_peak_count = max(non_target_peak_count, int(frame_pen_count))
    fingerpad_count = int(sum(target_fingerpad_trace))
    fingerpad_consec = _v11_max_consecutive(target_fingerpad_trace)
    body_count = int(sum(target_hand_body_trace))
    failure_reasons: list[str] = []
    if not target_ids:
        failure_reasons.append("missing_target_handle_geom_ids")
    if fingerpad_count < V11_MIN_FINGERPAD_CONTACT_FRAMES:
        failure_reasons.append("insufficient_target_fingerpad_contact_frames")
    if fingerpad_consec < V11_MIN_FINGERPAD_CONTACT_CONSECUTIVE_FRAMES:
        failure_reasons.append("insufficient_consecutive_target_fingerpad_contact")
    if body_count > 0:
        failure_reasons.append("target_contact_uses_forbidden_hand_body_link_or_wrist")
    if any(non_target_pen_trace):
        failure_reasons.append("post_step_non_target_robot_drawer_penetration")
    if any(wrist_inside_trace):
        failure_reasons.append("post_step_right_hand_inside_cabinet_bbox")
    return {
        "audit_version": "v11_post_step_replay_fingerpad_v1",
        "state_source": state_source,
        "audit_state_source": state_source,
        "required_audit_state_source": V11_REQUIRED_AUDIT_STATE_SOURCE,
        "target_contact_success_classes_allowed": ["fingerpad", "fingertip"],
        "target_contact_forbidden_classes": ["hand_collision", "body", "link", "wrist"],
        "frame_count": int(states.shape[0]),
        "post_step_replay_audit_passed": bool(not failure_reasons),
        "target_handle_geom_names": [
            str(model.geom(i).name or f"geom_{i}") for i in sorted(target_ids)
        ],
        "target_handle_geom_ids": [int(i) for i in sorted(target_ids)],
        "fingerpad_geom_names": [
            str(model.geom(i).name or f"geom_{i}") for i in sorted(fingerpad_ids)
        ],
        "fingertip_geom_names": [
            str(model.geom(i).name or f"geom_{i}") for i in sorted(fingertip_ids)
        ],
        "target_handle_contact_any": bool(any(target_any_trace)),
        "target_fingerpad_handle_contact_any": bool(fingerpad_count > 0),
        "target_fingerpad_handle_contact_count_frames": int(fingerpad_count),
        "target_fingerpad_handle_contact_max_consecutive_frames": int(fingerpad_consec),
        "target_finger_collision_handle_contact_count_frames": int(
            sum(target_finger_collision_trace)
        ),
        "target_hand_body_handle_contact_any": bool(body_count > 0),
        "target_hand_body_handle_contact_count_frames": int(body_count),
        "post_step_non_target_robot_drawer_penetration_any": bool(
            any(non_target_pen_trace)
        ),
        "post_step_non_target_robot_drawer_penetration_peak_count": int(
            non_target_peak_count
        ),
        "post_step_non_target_robot_drawer_min_dist_m": float(
            non_target_min_dist if non_target_min_dist is not None else 0.0
        ),
        "post_step_wrist_right_hand_inside_cabinet_any": bool(any(wrist_inside_trace)),
        "failure_reasons": failure_reasons,
        "target_contact_samples": target_samples,
        "non_target_contact_samples": non_target_samples,
        "target_fingerpad_handle_contact_trace": target_fingerpad_trace,
        "target_hand_body_handle_contact_trace": target_hand_body_trace,
        "post_step_non_target_robot_drawer_penetration_trace": non_target_pen_trace,
    }


def _apply_v11_post_step_replay_gate(
    seed: int, rollout: dict[str, Any]
) -> dict[str, Any]:
    audit = _v11_post_step_replay_audit(seed, rollout)
    pre_v11_pass = bool(rollout.get("physical_admission_passed", False))
    audit_state_source = str(
        audit.get("audit_state_source") or audit.get("state_source") or ""
    )
    v11_pass = bool(
        pre_v11_pass
        and audit.get("post_step_replay_audit_passed", False)
        and audit_state_source == V11_REQUIRED_AUDIT_STATE_SOURCE
    )
    frame_count = int(audit.get("frame_count", 0) or 0)
    rollout["pre_v11_physical_admission_passed"] = bool(pre_v11_pass)
    rollout["v11_post_step_replay_audit"] = _json_ready(audit)
    rollout["v11_post_step_replay_audit_passed"] = bool(
        audit.get("post_step_replay_audit_passed", False)
    )
    rollout["audit_state_source"] = audit_state_source
    rollout["required_audit_state_source"] = V11_REQUIRED_AUDIT_STATE_SOURCE
    rollout["v11_replay_bundle_untrimmed"] = True
    rollout["v11_trimmed_bundle_refused"] = True
    for key in [
        "target_fingerpad_handle_contact_any",
        "target_fingerpad_handle_contact_count_frames",
        "target_fingerpad_handle_contact_max_consecutive_frames",
        "target_hand_body_handle_contact_any",
        "target_hand_body_handle_contact_count_frames",
        "post_step_non_target_robot_drawer_penetration_any",
        "post_step_non_target_robot_drawer_penetration_peak_count",
        "post_step_non_target_robot_drawer_min_dist_m",
        "post_step_wrist_right_hand_inside_cabinet_any",
    ]:
        rollout[key] = audit.get(key, False if key.endswith("any") else 0)
    rollout["target_fingerpad_handle_contact_trace"] = np.asarray(
        audit.get("target_fingerpad_handle_contact_trace", [False] * frame_count),
        dtype=np.bool_,
    )
    rollout["target_hand_body_handle_contact_trace"] = np.asarray(
        audit.get("target_hand_body_handle_contact_trace", [False] * frame_count),
        dtype=np.bool_,
    )
    rollout["post_step_non_target_robot_drawer_penetration_trace"] = np.asarray(
        audit.get(
            "post_step_non_target_robot_drawer_penetration_trace", [False] * frame_count
        ),
        dtype=np.bool_,
    )
    rollout["physical_admission_passed"] = bool(v11_pass)
    rollout["physical_admission_verdict"] = (
        "v11_strict_fingerpad_post_step_admission_pass"
        if v11_pass
        else "v11_strict_fingerpad_post_step_admission_blocked"
    )
    rollout["diagnostic_verdict"] = (
        "v11_strict_full_robot_truth_ready"
        if v11_pass
        else "v11_strict_full_robot_truth_blocked"
    )
    if v11_pass:
        rollout["next_blocker"] = (
            "export_strict_replay_and_local_reference_visual_audit"
        )
    else:
        reasons = audit.get("failure_reasons", []) or []
        rollout["next_blocker"] = "repair_v11_teacher_" + (
            "_and_".join(str(r) for r in reasons[:3]) or "unknown_post_step_replay_gate"
        )
    return rollout


def _frame_indices(rollout: dict[str, Any]) -> dict[str, int]:
    indices = {"start": 0}
    if len(rollout["handle_distance_trace"]):
        indices["min_handle_distance"] = int(
            np.argmin(rollout["handle_distance_trace"])
        )
    if len(rollout["next_drawer_fractions"]):
        indices["max_drawer_fraction"] = int(
            np.argmax(rollout["next_drawer_fractions"])
        )
    contact_trace = np.asarray(
        rollout.get("target_handle_contact_trace", []), dtype=np.bool_
    )
    if contact_trace.size and bool(np.any(contact_trace)):
        indices["first_contact"] = int(np.argmax(contact_trace.astype(np.int32)))
    attach_trace = np.asarray(rollout.get("attach_eligible_trace", []), dtype=np.bool_)
    if attach_trace.size and bool(np.any(attach_trace)):
        indices["first_attach_eligible"] = int(np.argmax(attach_trace.astype(np.int32)))
    phase_trace = np.asarray(rollout.get("phase_locked_trace", []), dtype=np.bool_)
    if phase_trace.size and bool(np.any(phase_trace)):
        indices["first_phase_locked"] = int(np.argmax(phase_trace.astype(np.int32)))
    return indices


def _write_representative_frames(
    run_dir: Path, episode_slug: str, rollout: dict[str, Any]
) -> dict[str, str]:
    images = np.asarray(rollout.get("images", []), dtype=np.uint8)
    if images.size == 0:
        return {}
    frames_dir = run_dir / "frames" / episode_slug
    frames_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for label, idx in _frame_indices(rollout).items():
        idx = int(np.clip(idx, 0, len(images) - 1))
        frame = cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR)
        path = frames_dir / f"{label}.png"
        cv2.imwrite(str(path), frame)
        written[label] = str(path)
    return written


def _write_side_by_side_video(
    run_dir: Path, episode_slug: str, rollout: dict[str, Any]
) -> str | None:
    images = np.asarray(rollout.get("images", []), dtype=np.uint8)
    images2 = np.asarray(rollout.get("images2", []), dtype=np.uint8)
    if images.size == 0 or images2.size == 0:
        return None
    count = min(len(images), len(images2))
    video_dir = run_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    first = np.concatenate([images[0], images2[0]], axis=1)
    height, width = first.shape[:2]
    path = video_dir / f"{episode_slug}_side_by_side.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (width, height)
    )
    if not writer.isOpened():
        return None
    try:
        for idx in range(count):
            frame = np.concatenate([images[idx], images2[idx]], axis=1)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return str(path)


def _stamp_episode_lineage(
    sidecar_path: Path, plan: dict[str, Any], repo: dict[str, str]
) -> None:
    payload = load_json(sidecar_path, default={}) or {}
    payload.update(
        {
            "run_instance_id": plan.get("run_instance_id"),
            "plan_version": plan.get("plan_version"),
            "working_head_commit": plan.get("working_head_commit"),
            "execution_scope": plan.get("execution_scope"),
            "diagnostic_only": True,
            "claim_bearing": False,
            "repo_branch": repo.get("branch"),
            "repo_head_commit": repo.get("working_head_commit"),
            "vendor_head_commit": repo.get("vendor_head_commit"),
        }
    )
    write_json_atomic(sidecar_path, payload)


def _episode_summary(
    seed: int,
    control_mode: str,
    rollout: dict[str, Any],
    episode_npz: Path,
    frames: dict[str, str],
    video_path: str | None,
) -> dict[str, Any]:
    handle_distance = np.asarray(
        rollout.get("handle_distance_trace", []), dtype=np.float32
    )
    contact_trace = np.asarray(
        rollout.get("target_handle_contact_trace", []), dtype=np.bool_
    )
    contact_count_trace = np.asarray(
        rollout.get("target_handle_contact_count_trace", []), dtype=np.float32
    )
    contact_force_trace = np.asarray(
        rollout.get("target_handle_contact_force_trace", []), dtype=np.float32
    )
    attach_trace = np.asarray(rollout.get("attach_eligible_trace", []), dtype=np.bool_)
    phase_trace = np.asarray(rollout.get("phase_locked_trace", []), dtype=np.bool_)
    drawer_fraction = np.asarray(
        rollout.get("next_drawer_fractions", []), dtype=np.float32
    )
    contract = dict(rollout.get("task_object_contract", {}) or {})
    scene_contract = dict(contract.get("scene_contract", {}) or {})
    franka_visual_contract = dict(rollout.get("visual_mode_report", {}) or {})
    if not franka_visual_contract:
        franka_visual_contract = {
            "franka_visual_contract_version": scene_contract.get(
                "franka_visual_contract_version"
            )
        }
    return {
        "seed": int(seed),
        "control_mode": str(control_mode),
        "teacher_control_mode": rollout.get("teacher_control_mode")
        or str(control_mode),
        "control_point_kind": rollout.get(
            "control_point_kind", "gripper_grasp_center_v1"
        ),
        "interventions": _json_ready(rollout.get("interventions", {}) or {}),
        "runtime_kind": rollout.get("runtime_kind", "drawer_only_proxy_mujoco"),
        "teacher_controller_mode": rollout.get("teacher_controller_mode"),
        "task_object_contract_version": contract.get("contract_version"),
        "scene_contract_version": scene_contract.get("scene_contract_version"),
        "robot_base_pose_policy": scene_contract.get("robot_base_pose_policy"),
        "robot_base_pose_world": scene_contract.get("robot_base_pose_world"),
        "drawer_motion_source": rollout.get("drawer_motion_source"),
        "reference_panda_visual_contract_version": scene_contract.get(
            "franka_visual_contract_version"
        ),
        "target_drawer_id": rollout.get("target_drawer_id"),
        "target_handle_id": rollout.get("target_handle_id"),
        "target_handle_geom_names": list(contract.get("target_handle_geom_names", [])),
        "rollout_path": str(episode_npz),
        "steps": int(rollout.get("steps", 0)),
        "success": bool(rollout.get("success", False)),
        "min_handle_distance": float(np.min(handle_distance))
        if handle_distance.size
        else float("inf"),
        "mean_handle_distance": float(np.mean(handle_distance))
        if handle_distance.size
        else float("inf"),
        "target_handle_contact_any": bool(np.any(contact_trace))
        if contact_trace.size
        else False,
        "target_handle_contact_peak_count": float(np.max(contact_count_trace))
        if contact_count_trace.size
        else 0.0,
        "target_handle_contact_peak_force_n": float(np.max(contact_force_trace))
        if contact_force_trace.size
        else 0.0,
        "attach_eligible_any": bool(np.any(attach_trace))
        if attach_trace.size
        else False,
        "ever_attached": bool(rollout.get("ever_attached", False)),
        "phase_locked_any": bool(np.any(phase_trace)) if phase_trace.size else False,
        "max_drawer_fraction": float(np.max(drawer_fraction))
        if drawer_fraction.size
        else 0.0,
        "final_drawer_fraction": float(rollout.get("final_drawer_fraction", 0.0)),
        "physical_admission_verdict": rollout.get("physical_admission_verdict"),
        "physical_admission_passed": bool(
            rollout.get("physical_admission_passed", False)
        ),
        "pre_v11_physical_admission_passed": bool(
            rollout.get(
                "pre_v11_physical_admission_passed",
                rollout.get("physical_admission_passed", False),
            )
        ),
        "v11_post_step_replay_audit_passed": bool(
            rollout.get("v11_post_step_replay_audit_passed", False)
        ),
        "audit_state_source": rollout.get("audit_state_source"),
        "required_audit_state_source": rollout.get(
            "required_audit_state_source", V11_REQUIRED_AUDIT_STATE_SOURCE
        ),
        "v11_replay_bundle_untrimmed": bool(
            rollout.get("v11_replay_bundle_untrimmed", False)
        ),
        "v11_trimmed_bundle_refused": bool(
            rollout.get("v11_trimmed_bundle_refused", False)
        ),
        "target_contact_success_classes_allowed": ["fingerpad", "fingertip"],
        "target_contact_forbidden_classes": ["hand_collision", "body", "link", "wrist"],
        "target_fingerpad_handle_contact_any": bool(
            rollout.get("target_fingerpad_handle_contact_any", False)
        ),
        "target_fingerpad_handle_contact_count_frames": int(
            rollout.get("target_fingerpad_handle_contact_count_frames", 0)
        ),
        "target_fingerpad_handle_contact_max_consecutive_frames": int(
            rollout.get("target_fingerpad_handle_contact_max_consecutive_frames", 0)
        ),
        "target_hand_body_handle_contact_any": bool(
            rollout.get("target_hand_body_handle_contact_any", False)
        ),
        "target_hand_body_handle_contact_count_frames": int(
            rollout.get("target_hand_body_handle_contact_count_frames", 0)
        ),
        "post_step_non_target_robot_drawer_penetration_any": bool(
            rollout.get("post_step_non_target_robot_drawer_penetration_any", False)
        ),
        "post_step_non_target_robot_drawer_penetration_peak_count": int(
            rollout.get("post_step_non_target_robot_drawer_penetration_peak_count", 0)
        ),
        "post_step_non_target_robot_drawer_min_dist_m": float(
            rollout.get("post_step_non_target_robot_drawer_min_dist_m", 0.0)
        ),
        "post_step_wrist_right_hand_inside_cabinet_any": bool(
            rollout.get("post_step_wrist_right_hand_inside_cabinet_any", False)
        ),
        "v11_post_step_replay_failure_reasons": list(
            (rollout.get("v11_post_step_replay_audit", {}) or {}).get(
                "failure_reasons", []
            )
        ),
        "v11_post_step_replay_audit": _json_ready(
            rollout.get("v11_post_step_replay_audit", {}) or {}
        ),
        "penetration_audit_verdict": rollout.get("penetration_audit_verdict"),
        "direct_qpos_teleport_admission_any": bool(
            rollout.get("direct_qpos_teleport_admission_any", False)
        ),
        "non_target_robot_drawer_penetration_any": bool(
            rollout.get("non_target_robot_drawer_penetration_any", False)
        ),
        "wrist_right_hand_inside_cabinet_any": bool(
            rollout.get("wrist_right_hand_inside_cabinet_any", False)
        ),
        "diagnostic_verdict": rollout.get("diagnostic_verdict"),
        "next_blocker": rollout.get("next_blocker"),
        "representative_frames": frames,
        "representative_video": video_path,
    }


def _summary_score(summary: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        1.0 if summary.get("phase_locked_any") else 0.0,
        float(summary.get("max_drawer_fraction", 0.0)),
        1.0 if summary.get("ever_attached") else 0.0,
        1.0 if summary.get("attach_eligible_any") else 0.0,
        1.0 if summary.get("target_handle_contact_any") else 0.0,
    )


def _differential(seed_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    by_mode = {item["control_mode"]: item for item in seed_summaries}
    strict = by_mode.get("teacher_right_side_contact_physics")
    diagnostic = by_mode.get("teacher_grasp_center_hold") or by_mode.get(
        "torque_pd_diagnostic"
    )
    if strict is None or diagnostic is None:
        return None
    return {
        "seed": int(strict["seed"]),
        "strict_control_mode": strict["control_mode"],
        "diagnostic_control_mode": diagnostic["control_mode"],
        "min_handle_distance_gain_m": float(
            diagnostic["min_handle_distance"] - strict["min_handle_distance"]
        ),
        "contact_gain": int(bool(strict["target_handle_contact_any"]))
        - int(bool(diagnostic["target_handle_contact_any"])),
        "attach_gain": int(bool(strict["attach_eligible_any"]))
        - int(bool(diagnostic["attach_eligible_any"])),
        "phase_lock_gain": int(bool(strict["phase_locked_any"]))
        - int(bool(diagnostic["phase_locked_any"])),
        "drawer_fraction_gain": float(
            strict["max_drawer_fraction"] - diagnostic["max_drawer_fraction"]
        ),
        "strict_contact_better": _summary_score(strict) > _summary_score(diagnostic),
        "diagnostic_only_mode_remains_non_admissible": True,
    }


def _surface_payload(summary: dict[str, Any], latest_path: Path) -> dict[str, Any]:
    return {
        "current_full_robot_probe_artifact": str(latest_path),
        "full_robot_runtime_kind": summary.get(
            "full_robot_runtime_kind", "full_robot_in_scene_mujoco"
        ),
        "full_robot_control_mode": summary.get("full_robot_control_mode"),
        "strict_contact_full_robot_control_mode": summary.get(
            "strict_contact_full_robot_control_mode"
        ),
        "task_object_contract_version": summary.get("task_object_contract_version"),
        "scene_contract_version": summary.get("scene_contract_version"),
        "robot_base_pose_policy": summary.get("robot_base_pose_policy"),
        "robot_base_pose_world": summary.get("robot_base_pose_world"),
        "full_robot_diagnostic_verdict": summary.get("full_robot_diagnostic_verdict"),
        "strict_contact_physical_admission_verdict": summary.get(
            "strict_contact_physical_admission_verdict"
        ),
        "penetration_audit_verdict": summary.get("penetration_audit_verdict"),
        "full_robot_next_blocker": summary.get("full_robot_next_blocker"),
        "full_robot_probe": summary,
        "claim_bearing": False,
    }


def _write_surface(path: Path, payload: dict[str, Any]) -> None:
    current = load_json(path, default={}) or {}
    merged = dict(current)
    merged.update(payload)
    merged["timestamp_utc"] = utc_now()
    merged["execution_lineage"] = execution_lineage_payload()
    write_json_atomic(path, merged)


def _refresh_remote_surfaces(summary: dict[str, Any]) -> None:
    payload = _surface_payload(summary, FULL_ROBOT_PROBE_LATEST_PATH)
    for path in [
        HARNESS_STATE_PATH,
        HARNESS_STATE_V11_PATH,
        SOVEREIGN_SNAPSHOT_PATH,
        SOVEREIGN_SNAPSHOT_V11_PATH,
    ]:
        _write_surface(path, payload)
    truth_backend.build_workspace_manifest()
    truth_backend.build_current_truth()


def run_probe(
    seeds: list[int],
    control_modes: list[str],
    max_steps: int,
    image_size: int,
    teacher_controller_mode: str,
    interventions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = FULL_ROBOT_PROBE_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    interventions = dict(interventions or {})
    plan = load_json(ACTIVE_PLAN_PATH, default={}) or {}
    repo = _repo_identity()
    lineage = {
        "run_instance_id": plan.get("run_instance_id"),
        "plan_version": plan.get("plan_version"),
        "working_head_commit": plan.get("working_head_commit"),
        "execution_scope": plan.get("execution_scope"),
        "diagnostic_only": True,
        "claim_bearing": False,
        "repo_branch": repo.get("branch"),
        "repo_head_commit": repo.get("working_head_commit"),
        "vendor_head_commit": repo.get("vendor_head_commit"),
    }
    seed_summaries: list[dict[str, Any]] = []
    grouped: dict[int, list[dict[str, Any]]] = {}
    for seed in seeds:
        grouped.setdefault(int(seed), [])
        for control_mode in control_modes:
            grasp_pose = _build_grasp_pose(seed, control_mode)
            rollout = build_full_robot_rollout(
                seed=seed,
                grasp_pose_world=grasp_pose,
                episode_index=0,
                max_steps=max_steps,
                image_size=image_size,
                rotation_source="aligned",
                teacher_controller_mode=teacher_controller_mode,
                env_kwargs={"control_mode": control_mode},
                interventions=interventions,
            )
            if control_mode == "teacher_right_side_contact_physics":
                rollout = _apply_v11_post_step_replay_gate(seed, rollout)
            slug = f"seed_{seed:03d}_{control_mode}"
            rollout_path = run_dir / f"{slug}.npz"
            save_robot_rollout(rollout_path, rollout)
            _stamp_episode_lineage(
                rollout_path.with_suffix(".json"), plan, _repo_identity()
            )
            frames = _write_representative_frames(run_dir, slug, rollout)
            video_path = _write_side_by_side_video(run_dir, slug, rollout)
            summary = _episode_summary(
                seed, control_mode, rollout, rollout_path, frames, video_path
            )
            summary.update(lineage)
            write_json_atomic(run_dir / f"{slug}.summary.json", _json_ready(summary))
            seed_summaries.append(summary)
            grouped[int(seed)].append(summary)
    differential = [
        item
        for item in (_differential(items) for items in grouped.values())
        if item is not None
    ]
    strict_summaries = [
        item
        for item in seed_summaries
        if item.get("control_mode") == "teacher_right_side_contact_physics"
    ]
    strict_best_summary = (
        max(strict_summaries, key=_summary_score) if strict_summaries else None
    )
    best_summary = strict_best_summary or max(seed_summaries, key=_summary_score)
    strict_seed_pass = {
        int(item["seed"]): bool(item.get("physical_admission_passed", False))
        for item in strict_summaries
    }
    strict_required_seeds_passed = bool(
        strict_seed_pass
        and set(int(seed) for seed in seeds).issubset(set(strict_seed_pass))
        and all(bool(strict_seed_pass[int(seed)]) for seed in seeds)
    )
    latest = {
        "artifact_version": "full_robot_teacher_probe_v11_strict_post_step_fingerpad",
        "generated_at": utc_now(),
        "artifact_path": str(FULL_ROBOT_PROBE_LATEST_PATH),
        "run_dir": str(run_dir),
        **lineage,
        "diagnostic_only": True,
        "claim_bearing": False,
        "repo_identity": repo,
        "execution_lineage": execution_lineage_payload(),
        "active_plan_anchor": {
            "run_instance_id": plan.get("run_instance_id"),
            "plan_version": plan.get("plan_version"),
            "working_head_commit": plan.get("working_head_commit"),
            "execution_scope": plan.get("execution_scope"),
        },
        "teacher_controller_mode": teacher_controller_mode,
        "probe_interventions": _json_ready(interventions),
        "control_modes_tested": list(control_modes),
        "seeds_tested": [int(seed) for seed in seeds],
        "seed_summaries": _json_ready(seed_summaries),
        "differential_diagnostics": _json_ready(differential),
        "full_robot_runtime_kind": best_summary.get(
            "runtime_kind", "full_robot_in_scene_mujoco"
        ),
        "full_robot_control_mode": best_summary.get("control_mode"),
        "strict_contact_full_robot_control_mode": "teacher_right_side_contact_physics",
        "task_object_contract_version": best_summary.get(
            "task_object_contract_version"
        ),
        "scene_contract_version": best_summary.get("scene_contract_version"),
        "robot_base_pose_policy": best_summary.get("robot_base_pose_policy"),
        "robot_base_pose_world": best_summary.get("robot_base_pose_world"),
        "full_robot_diagnostic_verdict": best_summary.get("diagnostic_verdict"),
        "strict_contact_physical_admission_verdict": best_summary.get(
            "physical_admission_verdict"
        ),
        "penetration_audit_verdict": best_summary.get("penetration_audit_verdict"),
        "strict_required_seed_pass": strict_seed_pass,
        "strict_required_seeds_passed": strict_required_seeds_passed,
        "full_robot_next_blocker": best_summary.get("next_blocker"),
        "downstream_admission_ready": strict_required_seeds_passed,
    }
    write_json_atomic(FULL_ROBOT_PROBE_LATEST_PATH, latest)
    _refresh_remote_surfaces(latest)
    return latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-robot teacher servo probes and sync harness/sovereign surfaces."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13])
    parser.add_argument(
        "--control-modes",
        nargs="+",
        default=[
            "torque_pd_diagnostic",
            "teacher_grasp_center_hold",
            "teacher_right_side_contact_physics",
        ],
    )
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--teacher-controller-mode", default="interaction_frame_hybrid")
    parser.add_argument(
        "--initial-phase", choices=["pregrasp", "contact"], default=None
    )
    parser.add_argument("--handle-vertical-offset-m", type=float, default=0.0)
    parser.add_argument("--handle-tangent-offset-m", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interventions: dict[str, Any] = {}
    if args.initial_phase:
        interventions["initial_phase"] = str(args.initial_phase)
    if float(args.handle_vertical_offset_m) != 0.0:
        interventions["handle_vertical_offset_m"] = float(args.handle_vertical_offset_m)
    if float(args.handle_tangent_offset_m) != 0.0:
        interventions["handle_tangent_offset_m"] = float(args.handle_tangent_offset_m)
    latest = run_probe(
        seeds=[int(seed) for seed in args.seeds],
        control_modes=[str(mode) for mode in args.control_modes],
        max_steps=int(args.max_steps),
        image_size=int(args.image_size),
        teacher_controller_mode=str(args.teacher_controller_mode),
        interventions=interventions,
    )
    print(__import__("json").dumps(_json_ready(latest), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

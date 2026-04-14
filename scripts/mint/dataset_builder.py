#!/usr/bin/env python3
"""Utilities for packing robot rollout NPZ files into LeRobot datasets."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

STATE_NAMES = [
    "eef_pos_x",
    "eef_pos_y",
    "eef_pos_z",
    "motor_proxy_0",
    "motor_proxy_1",
    "motor_proxy_2",
    "motor_proxy_3",
    "gripper_joint",  # Historical compatibility label; provenance is emitted separately.
]

ACTION_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_rx",
    "delta_ry",
    "delta_rz",
    "gripper_command",
]

LIBERO_KEY_RENAME = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.image2",
    "state": "observation.state",
    "actions": "action",
}

STATE_MODE_ALIAS = {
    "S0": "m0_proxy",
    "S1": "telemetry_candidate_v3_transition",
    "S2": "telemetry_candidate_v4_task_identity",
}


def _nested_rollout_value(meta: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in meta and meta.get(key) is not None:
        return meta.get(key)
    state_spec = meta.get("state_spec") or {}
    contract_config = meta.get("contract_config") or {}
    handle_probe = meta.get("handle_probe_metadata") or {}
    orientation = meta.get("orientation_telemetry") or {}
    if key == "state_mode":
        return state_spec.get("state_mode", contract_config.get("state_mode", default))
    if key == "interaction_mode":
        return contract_config.get("interaction_mode", default)
    if key == "measurement_truthful":
        return handle_probe.get("measurement_truthful", default)
    if key == "measurement_truth_tier":
        return handle_probe.get("measurement_truth_tier", default)
    if key == "measurement_backend":
        return handle_probe.get("measurement_backend", default)
    if key == "measurement_verifier":
        return handle_probe.get("measurement_verifier", default)
    if key == "runtime_visible_handle_mapping_source":
        return handle_probe.get("runtime_visible_handle_mapping_source", default)
    if key == "runtime_handle_anchor_valid":
        return orientation.get("runtime_handle_anchor_valid", default)
    return default


def validate_rollout_for_canonical_training(meta: dict, plan: dict) -> tuple[bool, str]:
    expected_cell = str(plan.get("canonical_train_cell"))
    expected_transition = str(plan.get("best_transition_cell"))
    expected_state_mode = str(plan.get("best_train_state_mode"))
    expected_state_name = STATE_MODE_ALIAS.get(expected_state_mode, expected_state_mode)
    seed = meta.get("seed")
    strict_metrics = meta.get("strict_metrics") or {}
    strict_success = bool(strict_metrics.get("strict_success"))
    success_ok = strict_success or (bool(meta.get("success")) and bool(meta.get("strict_success_version")))
    checks = [
        (str(meta.get("canonical_train_cell")) == expected_cell, "canonical_train_cell_mismatch"),
        (str(meta.get("best_transition_cell")) == expected_transition, "best_transition_cell_mismatch"),
        (str(meta.get("best_train_state_mode")) == expected_state_mode, "best_train_state_mode_mismatch"),
        (str(meta.get("selector_mode")) == str(plan.get("selector_mode")), "selector_mode_mismatch"),
        (bool(_nested_rollout_value(meta, "measurement_truthful", False)) is True, "measurement_not_truthful"),
        (str(_nested_rollout_value(meta, "measurement_truth_tier", "")) == str(plan.get("measurement_truth_tier")), "measurement_truth_tier_mismatch"),
        (str(_nested_rollout_value(meta, "measurement_backend", "")) == str(plan.get("measurement_backend")), "measurement_backend_mismatch"),
        (str(_nested_rollout_value(meta, "measurement_verifier", "")) == str(plan.get("measurement_verifier")), "measurement_verifier_mismatch"),
        (str(_nested_rollout_value(meta, "runtime_visible_handle_mapping_source", "")) == str(plan.get("runtime_visible_handle_mapping_source")), "runtime_visible_handle_mapping_source_mismatch"),
        (bool(_nested_rollout_value(meta, "runtime_handle_anchor_valid", False)) is True, "runtime_handle_anchor_invalid"),
        (str(_nested_rollout_value(meta, "interaction_mode", "")) == "orientation_sensitive_v3_task_identity_locked", "interaction_mode_mismatch"),
        (str(_nested_rollout_value(meta, "state_mode", "")) == expected_state_name, "state_mode_mismatch"),
        (int(seed) in [int(x) for x in plan.get("train_seeds", [])] if seed is not None else False, "seed_not_in_train_split"),
        (int(seed) not in [int(x) for x in plan.get("heldout_seeds", [])] if seed is not None else False, "seed_in_heldout_split"),
        (success_ok, "not_success_or_strict_success"),
    ]
    for passed, reason in checks:
        if not passed:
            return False, reason
    return True, "ok"


def validate_built_dataset_provenance(dataset_root: Path, plan: dict) -> dict:
    provenance_path = dataset_root / "meta" / "provenance.json"
    report = {
        "dataset_root": str(dataset_root),
        "dataset_repo_id": plan.get("dataset_repo_id"),
        "canonical_train_cell": plan.get("canonical_train_cell"),
        "best_train_state_mode": plan.get("best_train_state_mode"),
        "train_seed_count": len(plan.get("train_seeds", [])),
        "used_successful_seed_count": 0,
        "used_rollout_count": 0,
        "effective_frame_count": 0,
        "rejected_rollout_count": 0,
        "provenance_hash": None,
        "dataset_valid": False,
        "errors": [],
    }
    if not provenance_path.exists():
        report["errors"].append("provenance_missing")
        return report
    raw = provenance_path.read_bytes()
    report["provenance_hash"] = hashlib.sha256(raw).hexdigest()
    provenance = json.loads(raw.decode("utf-8"))
    records = provenance.get("records", [])
    train_seeds = {int(x) for x in plan.get("train_seeds", [])}
    heldout_seeds = {int(x) for x in plan.get("heldout_seeds", [])}
    unique_successful_seeds = {int(x) for x in provenance.get("unique_successful_seeds", [])}
    canonical_values = {rec.get("canonical_train_cell") for rec in records if rec.get("canonical_train_cell") is not None}
    transition_values = {rec.get("best_transition_cell") for rec in records if rec.get("best_transition_cell") is not None}
    state_values = {rec.get("best_train_state_mode") for rec in records if rec.get("best_train_state_mode") is not None}
    measurement_backends = {rec.get("measurement_backend") for rec in records if rec.get("measurement_backend") is not None}
    measurement_verifiers = {rec.get("measurement_verifier") for rec in records if rec.get("measurement_verifier") is not None}
    runtime_anchor_values = [bool(rec.get("runtime_handle_anchor_valid", False)) for rec in records]
    runtime_anchor_valid_rate = float(sum(1 for x in runtime_anchor_values if x) / len(runtime_anchor_values)) if runtime_anchor_values else 0.0
    report.update({
        "used_successful_seed_count": len(unique_successful_seeds),
        "used_rollout_count": len(records),
        "effective_frame_count": int(provenance.get("effective_frame_count", 0) or 0),
        "unique_successful_seeds": sorted(unique_successful_seeds),
        "claim_policies": provenance.get("claim_policies", []),
        "state_modes": provenance.get("state_modes", []),
        "measurement_backends": sorted(x for x in measurement_backends if x is not None),
        "measurement_verifiers": sorted(x for x in measurement_verifiers if x is not None),
        "runtime_handle_anchor_valid_rate": runtime_anchor_valid_rate,
    })
    checks = [
        (canonical_values == {plan.get("canonical_train_cell")}, "canonical_train_cell_not_unique"),
        (transition_values == {plan.get("best_transition_cell")}, "best_transition_cell_not_unique"),
        (state_values == {plan.get("best_train_state_mode")}, "best_train_state_mode_not_unique"),
        (set(provenance.get("state_modes", [])) == {STATE_MODE_ALIAS.get(str(plan.get("best_train_state_mode")), str(plan.get("best_train_state_mode")))}, "state_modes_not_unique"),
        (set(provenance.get("claim_policies", [])) == {"canonical"}, "claim_policy_not_canonical"),
        (bool(unique_successful_seeds), "no_successful_seeds"),
        (unique_successful_seeds.issubset(train_seeds), "successful_seeds_not_subset_of_train"),
        (unique_successful_seeds.isdisjoint(heldout_seeds), "successful_seeds_overlap_heldout"),
        (report["effective_frame_count"] > 0, "effective_frame_count_nonpositive"),
        (measurement_backends == {plan.get("measurement_backend")}, "measurement_backend_inconsistent"),
        (measurement_verifiers == {plan.get("measurement_verifier")}, "measurement_verifier_inconsistent"),
        (runtime_anchor_valid_rate == 1.0 or all(runtime_anchor_values), "runtime_handle_anchor_not_fully_valid"),
    ]
    for passed, reason in checks:
        if not passed:
            report["errors"].append(reason)
    report["dataset_valid"] = not report["errors"]
    return report


def dataset_integrity(root: Path, repo_id: str) -> dict:
    data_files = sorted(root.glob("data/chunk-*/*.parquet"))
    payload = {
        "dataset_root": str(root),
        "repo_id": repo_id,
        "data_files": [str(p) for p in data_files[:10]],
        "total_chunks": len(list(root.glob("data/chunk-*"))),
        "meta_info_exists": (root / "meta" / "info.json").exists(),
        "meta_stats_exists": (root / "meta" / "stats.json").exists(),
        "meta_tasks_exists": (root / "meta" / "tasks.parquet").exists(),
        "meta_provenance_exists": (root / "meta" / "provenance.json").exists(),
    }
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id=repo_id, root=str(root))
        payload["dataset_loads"] = True
        payload["dataset_length"] = len(dataset)
        payload["feature_keys"] = sorted(dataset.features.keys())
        payload["image_shape"] = str(dataset.features.get("observation.images.image", {}).get("shape", "N/A"))
        payload["state_shape"] = str(dataset.features.get("observation.state", {}).get("shape", "N/A"))
        payload["action_shape"] = str(dataset.features.get("action", {}).get("shape", "N/A"))
    except Exception as exc:  # noqa: BLE001
        payload["dataset_loads"] = False
        payload["dataset_error"] = str(exc)
        payload["dataset_length"] = 0
    payload["passed"] = bool(
        data_files
        and payload["meta_info_exists"]
        and payload["meta_stats_exists"]
        and payload["meta_tasks_exists"]
        and payload["dataset_loads"]
        and payload["dataset_length"] > 0
    )
    return payload


def _infinigen_to_libero_action(action: np.ndarray, gripper_binarize: bool) -> np.ndarray:
    return action.astype(np.float32)


def _clip_gripper_to_libero_range(gripper_joint: float) -> float:
    return float(np.clip(gripper_joint, -0.042, +0.001))


def _build_libero_state(state_npz: np.ndarray, gripper_binarize: bool) -> np.ndarray:
    state = state_npz.astype(np.float32).copy()
    state[7] = _clip_gripper_to_libero_range(state[7])
    return state


def _rollout_provenance(meta: dict[str, Any], n_frames: int) -> dict[str, Any]:
    contract_config = meta.get("contract_config") or {}
    state_spec = meta.get("state_spec") or {}
    visual_mode_report = meta.get("visual_mode_report") or {}
    handle_probe = meta.get("handle_probe_metadata") or {}
    return {
        "seed": meta.get("seed"),
        "claim_policy": meta.get("claim_policy", "unspecified"),
        "raw_unique_frames": int(meta.get("raw_unique_frames", n_frames)),
        "effective_training_frames": int(meta.get("effective_training_frames", n_frames)),
        "success_repeat": int(meta.get("success_repeat", 1)),
        "contract_config": contract_config,
        "state_mode": state_spec.get("state_mode", contract_config.get("state_mode", "unknown")),
        "state_spec": state_spec,
        "diagnostic_only": bool(visual_mode_report.get("diagnostic_only", False)),
        "calibration_mode": visual_mode_report.get("calibration_mode", contract_config.get("calibration_mode", "unknown")),
        "secondary_camera_mode": visual_mode_report.get("secondary_camera_mode", contract_config.get("secondary_camera_mode", "unknown")),
        "render_profile": visual_mode_report.get("render_profile", contract_config.get("render_profile", "legacy_surface")),
        "background_mode": visual_mode_report.get("background_mode", contract_config.get("background_mode", "unknown")),
        "lighting_profile": visual_mode_report.get("lighting_profile", contract_config.get("lighting_profile", "unknown")),
        "material_policy": visual_mode_report.get("material_policy", contract_config.get("material_policy", "unknown")),
        "camera_framing_profile": visual_mode_report.get("camera_framing_profile", contract_config.get("camera_framing_profile", "unknown")),
        "measurement_mode": handle_probe.get("probe_measurement_mode", contract_config.get("measurement_mode", "unknown")),
        "truth_root_object": handle_probe.get("truth_root_object", "unknown"),
        "identity_resolution_tier": handle_probe.get("identity_resolution_tier", "unknown"),
        "measurement_backend": handle_probe.get("measurement_backend", "unknown"),
        "measurement_verifier": handle_probe.get("measurement_verifier", "unknown"),
        "measurement_truth_tier": handle_probe.get("measurement_truth_tier", "unknown"),
        "measurement_truthful": bool(handle_probe.get("measurement_truthful", False)),
        "measurement_warning_flags": handle_probe.get("measurement_warning_flags", []),
        "manifest_handle_entity_unique": bool(handle_probe.get("manifest_handle_entity_unique", False)),
        "runtime_handle_visual_geom_mapping_unique": bool(handle_probe.get("runtime_handle_visual_geom_mapping_unique", False)),
        "runtime_handle_collision_geom_mapping_unique": bool(handle_probe.get("runtime_handle_collision_geom_mapping_unique", False)),
        "duplicate_runtime_geom_name_flag": bool(handle_probe.get("duplicate_runtime_geom_name_flag", False)),
        "unnamed_runtime_geom_flag": bool(handle_probe.get("unnamed_runtime_geom_flag", False)),
        "resolved_handle_visual_geom_ids": handle_probe.get("resolved_handle_visual_geom_ids", []),
        "resolved_handle_visual_geom_names": handle_probe.get("resolved_handle_visual_geom_names", []),
        "resolved_handle_collision_geom_ids": handle_probe.get("resolved_handle_collision_geom_ids", []),
        "resolved_handle_collision_geom_names": handle_probe.get("resolved_handle_collision_geom_names", []),
        "handle_geom_ids": handle_probe.get("handle_geom_ids", []),
        "handle_geom_names": handle_probe.get("handle_geom_names", []),
        "semantic_mapping_hash": handle_probe.get("semantic_mapping_hash", ""),
        "segmentation_mask_support_rate_secondary": handle_probe.get("segmentation_mask_support_rate_secondary", 0.0),
        "isolated_mask_support_rate_secondary": handle_probe.get("isolated_mask_support_rate_secondary", 0.0),
        "segmentation_isolated_iou_secondary": handle_probe.get("segmentation_isolated_iou_secondary", 0.0),
        "segmentation_isolated_centroid_delta_px_secondary": handle_probe.get("segmentation_isolated_centroid_delta_px_secondary", 0.0),
        "handle_mask_area_ratio_primary": handle_probe.get("handle_mask_area_ratio_primary", 0.0),
        "handle_mask_area_ratio_secondary": handle_probe.get("handle_mask_area_ratio_secondary", 0.0),
        "bbox_over_mask_ratio_primary": handle_probe.get("bbox_over_mask_ratio_primary", 0.0),
        "bbox_over_mask_ratio_secondary": handle_probe.get("bbox_over_mask_ratio_secondary", 0.0),
        "selector_mode": meta.get("selector_mode", "adaptive"),
        "baseline_cell_id": meta.get("baseline_cell_id"),
        "ts_baseline_cell_id": meta.get("ts_baseline_cell_id"),
        "best_transition_cell": meta.get("best_transition_cell"),
        "canonical_train_cell": meta.get("canonical_train_cell"),
        "best_train_state_mode": meta.get("best_train_state_mode"),
        "runtime_handle_anchor_valid": bool(meta.get("runtime_handle_anchor_valid", False)),
        "phase_locked_rate": float(meta.get("phase_locked_rate", 0.0) or 0.0),
        "mean_effective_pull_progress": float(meta.get("mean_effective_pull_progress", 0.0) or 0.0),
        "frozen_matrix_hash": meta.get("frozen_matrix_hash"),
        "resource_budget_snapshot": meta.get("resource_budget_snapshot", {}),
        "visual_mode_report": visual_mode_report,
    }


def build_dataset_from_rollouts(
    rollout_paths: list[Path],
    dataset_root: Path,
    repo_id: str,
    *,
    robot_type: str = "infinigen_drawer_anygrasp_robot",
    remove_existing: bool = True,
    gripper_binarize: bool = False,
    image_size: int = 256,
) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    if remove_existing and dataset_root.exists():
        shutil.rmtree(dataset_root)

    img_shape = (image_size, image_size, 3)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        root=dataset_root,
        robot_type=robot_type,
        features={
            "observation.images.image": {
                "dtype": "image",
                "shape": img_shape,
                "names": ["height", "width", "channels"],
            },
            "observation.images.image2": {
                "dtype": "image",
                "shape": img_shape,
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": STATE_NAMES,
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ACTION_NAMES,
            },
        },
        use_videos=False,
    )

    episode_count = 0
    frame_count = 0
    seeds: list[int] = []
    tasks: list[str] = []
    source_files: list[str] = []
    provenance_records: list[dict[str, Any]] = []
    raw_unique_frame_total = 0
    effective_training_frame_total = 0
    claim_policies: set[str] = set()
    state_modes: set[str] = set()
    render_profiles: set[str] = set()
    background_modes: set[str] = set()
    lighting_profiles: set[str] = set()
    material_policies: set[str] = set()
    camera_framing_profiles: set[str] = set()
    success_repeats: set[int] = set()

    for npz_path in rollout_paths:
        data = np.load(npz_path, allow_pickle=True)
        meta_path = npz_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        task = str(meta.get("task", "open the middle drawer of the cabinet"))
        source_files.append(str(npz_path))
        tasks.append(task)
        n_frames = len(data["actions"])
        provenance = _rollout_provenance(meta, n_frames)
        provenance_records.append(provenance)
        raw_unique_frame_total += int(provenance["raw_unique_frames"])
        effective_training_frame_total += int(provenance["effective_training_frames"])
        claim_policies.add(str(provenance["claim_policy"]))
        state_modes.add(str(provenance["state_mode"]))
        render_profiles.add(str(provenance.get("render_profile", "legacy_surface")))
        background_modes.add(str(provenance.get("background_mode", "unknown")))
        lighting_profiles.add(str(provenance.get("lighting_profile", "unknown")))
        material_policies.add(str(provenance.get("material_policy", "unknown")))
        camera_framing_profiles.add(str(provenance.get("camera_framing_profile", "unknown")))
        success_repeats.add(int(provenance["success_repeat"]))

        for idx in range(n_frames):
            raw_action = data["actions"][idx].astype(np.float32)
            action = _infinigen_to_libero_action(raw_action, gripper_binarize)
            raw_state = data["states"][idx].astype(np.float32)
            state = _build_libero_state(raw_state, gripper_binarize)

            img = Image.fromarray(data["images"][idx].astype(np.uint8))
            img = img.resize((image_size, image_size), Image.BILINEAR)
            img_np = np.array(img, dtype=np.uint8)

            img2 = Image.fromarray(data["images2"][idx].astype(np.uint8))
            img2 = img2.resize((image_size, image_size), Image.BILINEAR)
            img2_np = np.array(img2, dtype=np.uint8)

            dataset.add_frame(
                {
                    "task": task,
                    "observation.images.image": img_np,
                    "observation.images.image2": img2_np,
                    "observation.state": state,
                    "action": action,
                }
            )
            frame_count += 1
        dataset.save_episode()
        episode_count += 1
        if "seed" in meta:
            seeds.append(int(meta["seed"]))

    dataset.finalize()

    provenance_payload = {
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "unique_successful_seeds": sorted(set(seeds)),
        "raw_unique_frames": int(raw_unique_frame_total),
        "effective_training_frames": int(effective_training_frame_total),
        "effective_episode_count": int(episode_count),
        "effective_frame_count": int(frame_count),
        "claim_policies": sorted(claim_policies),
        "state_modes": sorted(state_modes),
        "render_profiles": sorted(render_profiles),
        "background_modes": sorted(background_modes),
        "lighting_profiles": sorted(lighting_profiles),
        "material_policies": sorted(material_policies),
        "camera_framing_profiles": sorted(camera_framing_profiles),
        "success_repeat_values": sorted(success_repeats),
        "records": provenance_records,
    }
    provenance_path = dataset_root / "meta" / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance_payload, indent=2, ensure_ascii=False) + "\n")

    integrity = dataset_integrity(dataset_root, repo_id)
    return {
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "episode_count": episode_count,
        "frame_count": frame_count,
        "seed_coverage": sorted(set(seeds)),
        "task_coverage": sorted(set(tasks)),
        "source_files": source_files,
        "integrity": integrity,
        "unique_successful_seeds": sorted(set(seeds)),
        "raw_unique_frames": int(raw_unique_frame_total),
        "effective_training_frames": int(effective_training_frame_total),
        "effective_episode_count": int(episode_count),
        "effective_frame_count": int(frame_count),
        "claim_policies": sorted(claim_policies),
        "state_modes": sorted(state_modes),
        "render_profiles": sorted(render_profiles),
        "background_modes": sorted(background_modes),
        "lighting_profiles": sorted(lighting_profiles),
        "material_policies": sorted(material_policies),
        "camera_framing_profiles": sorted(camera_framing_profiles),
        "success_repeat_values": sorted(success_repeats),
        "provenance_path": str(provenance_path),
    }

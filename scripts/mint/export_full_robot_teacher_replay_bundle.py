#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MINT_SCRIPTS = PROJECT_ROOT / "scripts" / "mint"
if str(MINT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MINT_SCRIPTS))

from merged_model_builder import MergedModelBuilder  # noqa: E402
from mint_common import ARTIFACT_DIR, write_json_atomic  # noqa: E402

LATEST_PROBE_PATH = ARTIFACT_DIR / "full_robot_teacher_probe_latest.json"
DEFAULT_BUNDLE_PARENT = ARTIFACT_DIR / "a_plus_local_render_bridge"
V11_REQUIRED_AUDIT_STATE_SOURCE = "post_step_next_states"
V11_TRIM_MARKER_FIELDS = (
    "trimmed",
    "is_trimmed",
    "trimmed_bundle",
    "was_trimmed",
    "trim_start_frame",
    "trim_end_frame",
    "trimmed_frame_count",
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git(args: list[str], cwd: Path = PROJECT_ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _repo_identity() -> dict[str, str]:
    return {
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "head": _git(["rev-parse", "HEAD"]),
        "mint_vendor": _git(["-C", str(PROJECT_ROOT / "external" / "MINT"), "rev-parse", "HEAD"]),
    }


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_assets_zip(path: Path, assets: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in sorted(assets.items()):
            zf.writestr(str(name), payload)


def _extract_replay_state_trace(npz: Any) -> tuple[np.ndarray, str]:
    if "next_states" in npz.files:
        return np.asarray(npz["next_states"], dtype=np.float32), V11_REQUIRED_AUDIT_STATE_SOURCE
    raise ValueError(
        "strict V11 export requires post-step next_states; "
        "refusing pre-step states fallback"
    )


def _extract_qpos(npz: Any, sidecar: dict[str, Any]) -> dict[str, Any]:
    states, state_source = _extract_replay_state_trace(npz)
    state_spec = sidecar.get("state_spec", {}) or {}
    drawer_names = list(state_spec.get("drawer_joint_names", []))
    robot_names = list(state_spec.get("robot_joint_names", []))
    n_drawer = len(drawer_names)
    n_robot = len(robot_names)
    if n_drawer <= 0 or n_robot <= 0:
        raise ValueError("full_robot replay requires drawer_joint_names and robot_joint_names")
    if states.shape[1] < n_drawer + n_robot + 1:
        raise ValueError(
            f"state width {states.shape[1]} too small for drawer={n_drawer} robot={n_robot} gripper=1"
        )
    return {
        "state_source": state_source,
        "drawer_qpos": states[:, :n_drawer].astype(np.float32),
        "robot_qpos": states[:, n_drawer : n_drawer + n_robot].astype(np.float32),
        "gripper_scalar": states[:, n_drawer + n_robot].astype(np.float32),
    }


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "no"}
    return bool(value)


def _v11_audit(summary: dict[str, Any]) -> dict[str, Any]:
    audit = summary.get("v11_post_step_replay_audit", {})
    return dict(audit) if isinstance(audit, dict) else {}


def _v11_audit_state_source(summary: dict[str, Any]) -> str:
    audit = _v11_audit(summary)
    return str(
        summary.get("audit_state_source")
        or audit.get("audit_state_source")
        or audit.get("state_source")
        or ""
    )


def _v11_forbidden_target_contact_observed(summary: dict[str, Any]) -> bool:
    audit = _v11_audit(summary)
    if _truthy(summary.get("target_hand_body_handle_contact_any")) or _truthy(
        audit.get("target_hand_body_handle_contact_any")
    ):
        return True
    forbidden = {"hand_collision", "hand_or_body", "body", "link", "wrist"}
    for sample in audit.get("target_contact_samples", []) or []:
        if isinstance(sample, dict) and str(sample.get("contact_class") or "") in forbidden:
            return True
    return False


def _v11_non_target_penetration_observed(summary: dict[str, Any]) -> bool:
    audit = _v11_audit(summary)
    return bool(
        _truthy(summary.get("post_step_non_target_robot_drawer_penetration_any"))
        or _truthy(audit.get("post_step_non_target_robot_drawer_penetration_any"))
        or int(summary.get("post_step_non_target_robot_drawer_penetration_peak_count", 0) or 0) > 0
        or int(audit.get("post_step_non_target_robot_drawer_penetration_peak_count", 0) or 0) > 0
    )


def _v11_trim_refusal_reasons(
    summary: dict[str, Any], *, require_explicit_untrimmed: bool
) -> list[str]:
    reasons: list[str] = []
    if require_explicit_untrimmed and summary.get("v11_replay_bundle_untrimmed") is not True:
        reasons.append("missing_explicit_untrimmed_replay_refusal")
    for key in V11_TRIM_MARKER_FIELDS:
        if _truthy(summary.get(key)):
            reasons.append(f"trimmed_bundle_refused:{key}")
    return reasons


def _v11_strict_refusal_reasons(summary: dict[str, Any]) -> list[str]:
    audit = _v11_audit(summary)
    reasons: list[str] = []
    if not _truthy(summary.get("physical_admission_passed")):
        reasons.append("physical_admission_passed_false")
    if not (
        _truthy(summary.get("v11_post_step_replay_audit_passed"))
        or _truthy(audit.get("post_step_replay_audit_passed"))
    ):
        reasons.append("v11_post_step_replay_audit_not_passed")
    if _v11_audit_state_source(summary) != V11_REQUIRED_AUDIT_STATE_SOURCE:
        reasons.append("audit_state_source_not_post_step_next_states")
    if _v11_forbidden_target_contact_observed(summary):
        reasons.append("target_contact_uses_forbidden_hand_body_link_or_wrist")
    if _v11_non_target_penetration_observed(summary):
        reasons.append("post_step_non_target_robot_drawer_penetration")
    reasons.extend(_v11_trim_refusal_reasons(summary, require_explicit_untrimmed=True))
    return reasons


def export_bundle(
    probe_path: Path,
    bundle_root: Path,
    control_mode: str,
    *,
    require_physical_admission: bool = True,
) -> dict[str, Any]:
    probe = _json_load(probe_path)
    bundle_root.mkdir(parents=True, exist_ok=True)
    episodes_root = bundle_root / "episodes"
    scenes_root = bundle_root / "scenes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    scenes_root.mkdir(parents=True, exist_ok=True)

    repo = _repo_identity()
    anchor = dict(probe.get("active_plan_anchor") or {})
    run_instance_id = probe.get("run_instance_id") or anchor.get("run_instance_id")
    plan_version = probe.get("plan_version") or anchor.get("plan_version")
    working_head_commit = probe.get("working_head_commit") or anchor.get("working_head_commit")
    execution_scope = probe.get("execution_scope") or anchor.get("execution_scope")
    manifest: dict[str, Any] = {
        "schema_version": "strict_replay_bundle_v2",
        "bridge_name": "a_plus_full_robot_teacher_local_replay",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_probe_path": str(probe_path),
        "source_probe_run_dir": probe.get("run_dir"),
        "run_instance_id": run_instance_id,
        "plan_version": plan_version,
        "working_head_commit": working_head_commit,
        "execution_scope": execution_scope,
        "authority_repo": repo,
        "authority_run_instance_id": run_instance_id,
        "authority_execution_scope": execution_scope,
        "diagnostic_only": True,
        "claim_bearing": False,
        "control_mode_filter": control_mode,
        "teacher_control_mode": control_mode,
        "visual_parity_target": "reference_right_side_franka_reference_panda_local_replay",
        "drawer_contract_version": probe.get("task_object_contract_version"),
        "scene_contract_version": probe.get("scene_contract_version"),
        "robot_base_pose_policy": probe.get("robot_base_pose_policy"),
        "robot_base_pose_world": probe.get("robot_base_pose_world"),
        "drawer_motion_source": "physics_contact",
        "franka_visual_contract_version": "reference_panda_visual_mjcf_v1",
        "reference_panda_visual_contract_version": "reference_panda_visual_mjcf_v1",
        "qpos_mapping_version": "full_robot_joint_state_qpos_mapping_v2",
        "physical_admission_required": bool(require_physical_admission),
        "required_audit_state_source": V11_REQUIRED_AUDIT_STATE_SOURCE,
        "diagnostic_export_non_claim_bearing": bool(not require_physical_admission),
        "episodes": [],
    }

    selected: list[dict[str, Any]] = []
    rejected: dict[str, list[str]] = {}
    for item in probe.get("seed_summaries", []):
        if item.get("control_mode") != control_mode:
            continue
        if item.get("runtime_kind") != "full_robot_in_scene_mujoco":
            continue
        seed_label = str(item.get("seed", "unknown"))
        if require_physical_admission:
            reasons = _v11_strict_refusal_reasons(item)
            if reasons:
                rejected[seed_label] = reasons
                continue
        else:
            reasons = _v11_trim_refusal_reasons(item, require_explicit_untrimmed=False)
            if reasons:
                rejected[seed_label] = reasons
                continue
        selected.append(item)
    if not selected:
        raise ValueError(
            f"no strict V11 admissible full-robot episodes found for control_mode={control_mode}; "
            f"rejected={rejected}"
        )

    for summary in selected:
        seed = int(summary["seed"])
        rollout_path = Path(summary["rollout_path"])
        sidecar_path = rollout_path.with_suffix(".json")
        sidecar = _json_load(sidecar_path)
        merged_gate_summary = dict(sidecar)
        merged_gate_summary.update(summary)
        if require_physical_admission:
            reasons = _v11_strict_refusal_reasons(merged_gate_summary)
            if reasons:
                raise ValueError(f"strict V11 export refused seed={seed}: {reasons}")
        npz = np.load(rollout_path, allow_pickle=True)
        qpos = _extract_qpos(npz, sidecar)
        if qpos["state_source"] != V11_REQUIRED_AUDIT_STATE_SOURCE:
            raise ValueError(f"strict V11 export refused state_source={qpos['state_source']}")

        episode_key = f"seed_{seed:03d}_{control_mode}"
        episode_dir = episodes_root / episode_key
        scene_dir = scenes_root / f"seed_{seed:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        scene_dir.mkdir(parents=True, exist_ok=True)

        builder = MergedModelBuilder(seed=seed)
        xml, assets, scene_metadata, semantic_mapping_hash = builder.build()
        scene_xml = scene_dir / "scene.xml"
        assets_zip = scene_dir / "assets.zip"
        scene_xml.write_text(xml)
        _write_assets_zip(assets_zip, assets)

        trace_npz = episode_dir / "trace.npz"
        np.savez_compressed(
            trace_npz,
            robot_qpos=qpos["robot_qpos"],
            drawer_qpos=qpos["drawer_qpos"],
            gripper_scalar=qpos["gripper_scalar"],
            drawer_fraction=np.asarray(npz.get("next_drawer_fractions", []), dtype=np.float32),
            target_handle_contact=np.asarray(npz.get("target_handle_contact_trace", []), dtype=np.bool_),
            attach_eligible=np.asarray(npz.get("attach_eligible_trace", []), dtype=np.bool_),
            attached=np.asarray(npz.get("attached_trace", []), dtype=np.bool_),
            phase_locked=np.asarray(npz.get("phase_locked_trace", []), dtype=np.bool_),
            non_target_robot_drawer_contact_count=np.asarray(
                npz.get("non_target_robot_drawer_contact_count_trace", []),
                dtype=np.float32,
            ),
            non_target_robot_drawer_penetration_count=np.asarray(
                npz.get("non_target_robot_drawer_penetration_count_trace", []),
                dtype=np.float32,
            ),
            wrist_right_hand_inside_cabinet=np.asarray(
                npz.get("wrist_right_hand_inside_cabinet_trace", []),
                dtype=np.bool_,
            ),
            direct_qpos_teleport=np.asarray(
                npz.get("direct_qpos_teleport_trace", []),
                dtype=np.bool_,
            ),
            controller_subphase=np.asarray(npz.get("controller_subphase_trace", []), dtype=object),
        )
        sidecar_out = episode_dir / "episode.json"
        task_contract = sidecar.get("task_object_contract", {})
        scene_contract = (
            task_contract.get("scene_contract")
            if isinstance(task_contract, dict)
            else None
        ) or scene_metadata.get("scene_contract", {})
        franka_visual_contract = scene_metadata.get("franka_visual_contract", {})
        episode_payload = {
            "episode_key": episode_key,
            "seed": seed,
            "run_instance_id": run_instance_id,
            "plan_version": plan_version,
            "working_head_commit": working_head_commit,
            "execution_scope": execution_scope,
            "diagnostic_only": True,
            "claim_bearing": False,
            "source_rollout_path": str(rollout_path),
            "source_sidecar_path": str(sidecar_path),
            "state_source": qpos["state_source"],
            "audit_state_source": _v11_audit_state_source(merged_gate_summary),
            "required_audit_state_source": V11_REQUIRED_AUDIT_STATE_SOURCE,
            "v11_post_step_replay_audit_passed": bool(merged_gate_summary.get("v11_post_step_replay_audit_passed", False)),
            "v11_strict_refusal_reasons": _v11_strict_refusal_reasons(merged_gate_summary),
            "v11_diagnostic_export_non_claim_bearing": bool(not require_physical_admission),
            "v11_replay_bundle_untrimmed": bool(merged_gate_summary.get("v11_replay_bundle_untrimmed", False)),
            "runtime_kind": summary.get("runtime_kind"),
            "control_mode": summary.get("control_mode"),
            "teacher_control_mode": summary.get("teacher_control_mode") or summary.get("control_mode"),
            "control_point_kind": summary.get("control_point_kind"),
            "teacher_controller_mode": summary.get("teacher_controller_mode"),
            "task_object_contract_version": summary.get("task_object_contract_version"),
            "scene_contract_version": scene_contract.get("scene_contract_version"),
            "robot_base_pose_policy": scene_contract.get("robot_base_pose_policy"),
            "robot_base_pose_world": scene_contract.get("robot_base_pose_world"),
            "drawer_motion_source": summary.get("drawer_motion_source", "physics_contact"),
            "physical_admission_verdict": summary.get("physical_admission_verdict"),
            "physical_admission_passed": bool(summary.get("physical_admission_passed", False)),
            "penetration_audit_verdict": summary.get("penetration_audit_verdict"),
            "target_drawer_id": summary.get("target_drawer_id"),
            "target_handle_id": summary.get("target_handle_id"),
            "target_handle_geom_names": summary.get("target_handle_geom_names", []),
            "drawer_contract_version": summary.get("task_object_contract_version"),
            "franka_visual_contract_version": franka_visual_contract.get(
                "franka_visual_contract_version",
                "reference_panda_visual_mjcf_v1",
            ),
            "reference_panda_visual_contract_version": "reference_panda_visual_mjcf_v1",
            "qpos_mapping_version": (
                sidecar.get("state_spec", {}).get("qpos_mapping_version")
                or scene_metadata.get("qpos_mapping_version")
                or "full_robot_joint_state_qpos_mapping_v2"
            ),
            "scene_contract": scene_contract,
            "franka_visual_contract": franka_visual_contract,
            "state_spec": sidecar.get("state_spec", {}),
            "task_object_contract": task_contract,
            "summary": summary,
            "build_report": {
                "semantic_mapping_hash": semantic_mapping_hash,
                "scene_metadata": scene_metadata,
            },
        }
        write_json_atomic(sidecar_out, _json_ready(episode_payload))
        manifest["episodes"].append(
            {
                "episode_key": episode_key,
                "seed": seed,
                "trace_npz_rel": str(trace_npz.relative_to(bundle_root)),
                "episode_json_rel": str(sidecar_out.relative_to(bundle_root)),
                "scene_xml_rel": str(scene_xml.relative_to(bundle_root)),
                "scene_assets_zip_rel": str(assets_zip.relative_to(bundle_root)),
                "state_source": qpos["state_source"],
                "audit_state_source": _v11_audit_state_source(merged_gate_summary),
                "required_audit_state_source": V11_REQUIRED_AUDIT_STATE_SOURCE,
                "v11_post_step_replay_audit_passed": bool(merged_gate_summary.get("v11_post_step_replay_audit_passed", False)),
                "v11_strict_refusal_reasons": _v11_strict_refusal_reasons(merged_gate_summary),
                "v11_diagnostic_export_non_claim_bearing": bool(not require_physical_admission),
                "v11_replay_bundle_untrimmed": bool(merged_gate_summary.get("v11_replay_bundle_untrimmed", False)),
                "frame_count": int(qpos["robot_qpos"].shape[0]),
                "target_drawer_id": summary.get("target_drawer_id"),
                "target_handle_id": summary.get("target_handle_id"),
                "runtime_kind": summary.get("runtime_kind"),
                "control_mode": summary.get("control_mode"),
                "teacher_control_mode": summary.get("teacher_control_mode") or summary.get("control_mode"),
                "drawer_contract_version": summary.get("task_object_contract_version"),
                "scene_contract_version": episode_payload["scene_contract_version"],
                "robot_base_pose_policy": episode_payload["robot_base_pose_policy"],
                "robot_base_pose_world": episode_payload["robot_base_pose_world"],
                "drawer_motion_source": episode_payload["drawer_motion_source"],
                "physical_admission_verdict": episode_payload["physical_admission_verdict"],
                "physical_admission_passed": episode_payload["physical_admission_passed"],
                "penetration_audit_verdict": episode_payload["penetration_audit_verdict"],
                "franka_visual_contract_version": episode_payload["franka_visual_contract_version"],
                "reference_panda_visual_contract_version": episode_payload[
                    "reference_panda_visual_contract_version"
                ],
                "qpos_mapping_version": episode_payload["qpos_mapping_version"],
                "max_drawer_fraction": float(summary.get("max_drawer_fraction", 0.0)),
                "success": bool(summary.get("success", False)),
                "diagnostic_verdict": summary.get("diagnostic_verdict"),
            }
        )

    write_json_atomic(bundle_root / "manifest.json", _json_ready(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full-robot teacher packet for local MuJoCo strict replay")
    parser.add_argument("--probe-path", type=Path, default=LATEST_PROBE_PATH)
    parser.add_argument("--bundle-root", type=Path, default=None)
    parser.add_argument("--control-mode", default="teacher_right_side_contact_physics")
    parser.add_argument(
        "--allow-diagnostic-export",
        action="store_true",
        help="Allow exporting non-admissible diagnostic packets for debugging only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_root = args.bundle_root
    if bundle_root is None:
        bundle_root = DEFAULT_BUNDLE_PARENT / f"full_robot_teacher_local_replay_{_utc_stamp()}_{_repo_identity()['head'][:8]}"
    manifest = export_bundle(
        Path(args.probe_path),
        Path(bundle_root),
        str(args.control_mode),
        require_physical_admission=not bool(args.allow_diagnostic_export),
    )
    print(json.dumps(_json_ready(manifest), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

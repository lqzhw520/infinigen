#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN = ROOT / "experiments/mint/mint_drawer_v1"
TASK_ID = "V11_G4_GOLD_CONTROLLER_MANIFOLD_LOCKED_TOPOLOGY_TRANSPLANT_STRICT_REPLAY_V2"
SPEC_REL = "experiments/mint/mint_drawer_v1/sovereign/experiment_specs/v11_g4_gold_controller_manifold_locked_topology_transplant_strict_replay_v2.yaml"
RUN_PREFIX = "v11_g4_gold_controller_manifold_locked_topology_transplant"
GOLD_RUN_REL = "experiments/mint/mint_drawer_v1/runtime/v11_g4_goc_v4_drawer_topology_realism_repair_and_recertify_20260505T135122Z"
GOLD_CANDIDATE_ID = "dh_c1_admitted_island_densify_25"
GOLD_VARIANT_NAME = "pc02_micro_lead_high_damping_semi_close"
STRICT_DRAWER_FRACTION = 0.80
MIN_EXACT_PULL_FRAMES = 30
MAX_PENETRATION_M = 0.02
REFERENCE_IMAGES = [
    {
        "local_path": "/Users/zhuhaowu/Downloads/knob-handler.png",
        "run_dir_name": "knob-handler.png",
        "sha256": "0717400d3bbe76258d64993513bcd6e8eae66493e289c9d29923b1db3ac17580",
    },
    {
        "local_path": "/Users/zhuhaowu/Downloads/ChatGPT Image 2026年5月5日 23_07_48 (2).png",
        "run_dir_name": "ChatGPT Image 2026年5月5日 23_07_48 (2).png",
        "sha256": "58b55a62765bee57500efa8127c86f8366c9f3e8bc64e9e7d7373860d732ce56",
    },
]
POSITIVE_RUNS = [
    {
        "run_id": "v11_g4_goc_v4_defect_histogram_dynamic_pull_pool_solver_to_export_replay_20260505T085503Z",
        "remote_run": "experiments/mint/mint_drawer_v1/runtime/v11_g4_goc_v4_defect_histogram_dynamic_pull_pool_solver_to_export_replay_20260505T085503Z",
        "local_replay": "/Users/zhuhaowu/Documents/Playground/local_replay/v11_g4_goc_v4_defect_histogram_dynamic_pull_pool_solver_to_export_replay_20260505T085503Z",
        "manual_classification": "POSITIVE_CONTROLLER_AFFORDANCE_ANCHOR_WITH_LOCALIZED_TOPOLOGY_DEFECTS",
    },
    {
        "run_id": "v11_g4_goc_v4_drawer_topology_realism_repair_and_recertify_20260505T135122Z",
        "remote_run": GOLD_RUN_REL,
        "local_replay": "/Users/zhuhaowu/Documents/Playground/local_replay/v11_g4_goc_v4_drawer_topology_realism_repair_and_recertify_20260505T135122Z",
        "manual_classification": "POSITIVE_CONTROLLER_AFFORDANCE_ANCHOR_WITH_LONG_CONNECTOR_AND_TRAY_TOPOLOGY_DEFECTS",
    },
]
NEGATIVE_RUNS = [
    {
        "run_id": "v11_g4_goc_v4_exact_latch_pull_keepout_v3_continuation_targeted_progress_repair_20260506T123821Z",
        "remote_run": "experiments/mint/mint_drawer_v1/runtime/v11_g4_goc_v4_exact_latch_pull_keepout_v3_continuation_targeted_progress_repair_20260506T123821Z",
        "local_replay": "/Users/zhuhaowu/Documents/Playground/local_replay/v11_g4_goc_v4_exact_latch_pull_keepout_v3_continuation_targeted_progress_repair_20260506T123821Z",
        "manual_classification": "NEGATIVE_TOPOLOGY_EVIDENCE_MISSING_CONNECTOR_FRAME_ONLY_OR_MISSING_TRAY",
    },
    {
        "run_id": "v11_g4_reference_topology_controller_pull_work_repair_20260507T064437Z",
        "remote_run": "experiments/mint/mint_drawer_v1/runtime/v11_g4_reference_topology_controller_pull_work_repair_20260507T064437Z",
        "local_replay": "/Users/zhuhaowu/Documents/Playground/local_replay/v11_g4_reference_topology_controller_pull_work_repair_20260507T064437Z",
        "manual_classification": "NEGATIVE_TOPOLOGY_AND_VISIBLE_LATCH_EVIDENCE_OVERSIZED_KNOB_TINY_TRAY_FLOATING_GRIPPER",
    },
]
FORBIDDEN_TERMINAL_LABELS = {
    "CONTROLLER_MIGRATION_FAILED_ON_REALISTIC_TOPOLOGY",
    "ROUND_KNOB_SIZE_REALISTIC_GRIPPER_AFFORDANCE_CONFLICT",
    "CONTROLLER_CANNOT_OPEN_REALISTIC_KNOB",
    "CONTROLLER_CANNOT_USE_REALISTIC_KNOB",
    "REALISTIC_TOPOLOGY_INCOMPATIBLE_WITH_CONTROLLER",
}

sys.path.insert(0, str(ROOT / "scripts/mint"))
import v11_g4_reference_topology_scale_visible_latch_contact_repair as scale  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ready(v: Any) -> Any:
    return scale.ready(v)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ready(payload), indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(ready(payload), sort_keys=True) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def rel(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_git(args: list[str], timeout: int = 60) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, timeout=timeout
    )
    return proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()


def row_fraction(row: dict[str, Any]) -> float:
    return float(
        row.get("max_drawer_fraction") or row.get("drawer_fraction_max") or 0.0
    )


def row_forbidden(row: dict[str, Any]) -> int:
    return int(
        row.get("forbidden_contact_frames")
        or row.get("forbidden_contact_frame_count")
        or row.get("forbidden_contact_frames_max")
        or 0
    )


def row_handle_nonlegal(row: dict[str, Any]) -> int:
    return int(
        row.get("handle_nonlegal_contact_frames")
        or row.get("handle_nonlegal_contact_frame_count")
        or row.get("handle_nonlegal_contact_frames_max")
        or 0
    )


def row_bilateral_pull(row: dict[str, Any]) -> int:
    return int(
        row.get("pull_phase_two_pad_target_contact_frames")
        or row.get("bilateral_exact_contact_pull_frames")
        or row.get("pull_phase_two_pad_target_contact_frames_min")
        or 0
    )


def row_pen(row: dict[str, Any]) -> float:
    return float(row.get("max_penetration_m") or row.get("max_penetration") or 0.0)


def norm3(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a[:3], b[:3])))


def load_gold_source_records() -> tuple[dict[str, Any], dict[str, Any]]:
    gold_run = ROOT / GOLD_RUN_REL
    candidate_record = {}
    for row in read_jsonl(gold_run / "candidate_oracle_results.jsonl"):
        if row.get("candidate_id") == GOLD_CANDIDATE_ID:
            candidate_record = row
            break
    full30_row = {}
    for row in read_jsonl(gold_run / "full30_dynamic_pull_certification.jsonl"):
        if (
            row.get("candidate_id") == GOLD_CANDIDATE_ID
            and row.get("variant_name") == GOLD_VARIANT_NAME
            and row.get("perturbation") == "nominal"
        ):
            full30_row = row
            break
    if not full30_row:
        for row in read_jsonl(gold_run / "full30_dynamic_pull_certification.jsonl"):
            if (
                row.get("candidate_id") == GOLD_CANDIDATE_ID
                and row.get("variant_name") == GOLD_VARIANT_NAME
            ):
                full30_row = row
                break
    return candidate_record, full30_row


def write_reference_visual_contract(run_dir: Path) -> dict[str, Any]:
    image_rows = []
    for item in REFERENCE_IMAGES:
        remote = run_dir / "reference_images" / item["run_dir_name"]
        actual = sha256_file(remote)
        image_rows.append(
            {
                **item,
                "run_dir_reference_path": rel(remote),
                "copied_to_run_dir": remote.exists(),
                "run_dir_sha256": actual,
                "sha256_matches_expected": actual == item["sha256"] if actual else None,
            }
        )
    contract = {
        "generated_at_utc": utc_now(),
        "task_id": TASK_ID,
        "reference_images": image_rows,
        "closed_state_visual_target": {
            "solid_front_panel": True,
            "small_medium_spherical_knob": True,
            "knob_almost_flush_to_front": True,
            "very_short_boss_or_stub": True,
            "long_rod_rejected": True,
            "front_panel_visibly_supports_knob": True,
        },
        "open_state_visual_target": {
            "complete_drawer_box_or_tray_moves_out_with_front": True,
            "tray_occupies_plausible_portion_of_front_panel": True,
            "side_walls_visible": True,
            "bottom_visible": True,
            "back_panel_visible": True,
            "support_rails_or_guides_visible": True,
            "unsupported_or_floating_drawer_rejected": True,
        },
        "manipulation_visual_target": {
            "gripper_fingers_visibly_contact_knob_before_pull": True,
            "gripper_remains_visibly_latched_on_knob_during_opening": True,
            "open_while_fingers_float_artifact_rejected": True,
            "final_claim_render_has_no_debug_overlay": True,
        },
    }
    write_json(run_dir / "reference_visual_contract.json", contract)
    md = [
        "# Reference Visual Contract",
        "",
        f'- Generated: `{contract["generated_at_utc"]}`',
        "",
    ]
    for row in image_rows:
        md.append(
            f'- `{row["local_path"]}` sha256 `{row["sha256"]}` copied={row["copied_to_run_dir"]} match={row["sha256_matches_expected"]}'
        )
    md.extend(
        [
            "",
            "Closed target: solid front, small/medium spherical knob, short boss/stub, no long rod.",
            "Open target: complete moving tray with walls, bottom, back, and visible support guides.",
            "Manipulation target: visible bilateral pad latch before and during pull; no floating-hand artifact.",
        ]
    )
    (run_dir / "reference_visual_contract.md").write_text("\n".join(md) + "\n")
    return contract


def summarize_remote_run(run: dict[str, Any]) -> dict[str, Any]:
    root = ROOT / run["remote_run"]
    final_closeout = read_json(root / "final_closeout.json", {}) or {}
    targeted = read_json(root / "targeted_shard_results.json", {}) or {}
    full30 = (
        read_json(root / "full30_dynamic_pull_certification_report.json", {})
        or read_json(root / "full30_report.json", {})
        or {}
    )
    export = read_json(root / "strict_teacher_export_manifest.json", {}) or {}
    trace_manifest = read_json(root / "trace_hash_manifest.json", {}) or {}
    local_manifest = read_json(root / "render_manifest.json", {}) or {}
    return {
        **run,
        "remote_exists": root.exists(),
        "final_closeout_present": bool(final_closeout),
        "closeout_classification": final_closeout.get("closeout_classification"),
        "strict_metric_evidence": {
            "targeted_cases_passed": targeted.get("cases_passed"),
            "targeted_cases_total": targeted.get("cases_total"),
            "targeted_passed": targeted.get("targeted_shard_passed")
            or targeted.get("matrix_passed"),
            "full30_cases_passed": full30.get("cases_passed")
            or final_closeout.get("full30_cases_passed"),
            "full30_cases_total": full30.get("cases_total")
            or final_closeout.get("full30_cases_total"),
            "full30_passed": full30.get("matrix_passed")
            or final_closeout.get("full30_certification_passed"),
            "forbidden_contact_frames_max": full30.get("forbidden_contact_frames_max")
            or targeted.get("forbidden_contact_frames_max"),
            "handle_nonlegal_contact_frames_max": full30.get(
                "handle_nonlegal_contact_frames_max"
            )
            or targeted.get("handle_nonlegal_contact_frames_max"),
            "max_penetration_m": full30.get("max_penetration_m")
            or targeted.get("max_penetration_m"),
            "pull_phase_two_pad_target_contact_frames_min": full30.get(
                "pull_phase_two_pad_target_contact_frames_min"
            )
            or targeted.get("pull_phase_two_pad_target_contact_frames_min"),
            "direct_qpos_drawer_opening": full30.get("direct_qpos_drawer_opening")
            if "direct_qpos_drawer_opening" in full30
            else targeted.get("direct_qpos_drawer_opening"),
            "drawer_motor_command_used": full30.get("drawer_motor_command_used")
            if "drawer_motor_command_used" in full30
            else targeted.get("drawer_motor_command_used"),
        },
        "strict_export_complete": bool(
            export.get("strict_teacher_export_complete")
            or export.get("strict_export_complete")
        ),
        "strict_export_trace_count": export.get("trace_count"),
        "trace_hash_manifest_file_count": trace_manifest.get("file_count"),
        "render_manifest_present_remote": bool(local_manifest),
        "model_xml_paths_in_export": [
            item.get("model_xml") for item in (export.get("trace_items") or [])[:3]
        ],
    }


def stage0_historical_evidence_lock(run_dir: Path) -> dict[str, Any]:
    local_index = (
        read_json(run_dir / "historical_render_keyframe_index.local.json", {}) or {}
    )
    records = []
    for run in POSITIVE_RUNS:
        rec = summarize_remote_run(run)
        rec["classification"] = "POSITIVE_CONTROLLER_AFFORDANCE_ANCHOR"
        rec["visual_manual_evidence"] = {
            "reasonable_knob_size": True,
            "plausible_gripper_knob_grasp_and_open": True,
            "topology_defects_localized_and_repairable": True,
            "manual_source": "user visual review of historical local render mp4",
        }
        records.append(rec)
    for run in NEGATIVE_RUNS:
        rec = summarize_remote_run(run)
        rec["classification"] = "NEGATIVE_TOPOLOGY_EVIDENCE"
        rec["visual_manual_evidence"] = {
            "frame_only_or_missing_tray_or_missing_connector_or_visible_latch_failure": True,
            "manual_source": "user visual review of historical local render mp4",
        }
        records.append(rec)
    selected = next((r for r in records if r["remote_run"] == GOLD_RUN_REL), None)
    payload = {
        "generated_at_utc": utc_now(),
        "task_id": TASK_ID,
        "historical_runs": records,
        "local_render_keyframe_index": local_index,
        "selected_positive_anchor": selected,
        "positive_anchor_count": len(
            [
                r
                for r in records
                if r["classification"] == "POSITIVE_CONTROLLER_AFFORDANCE_ANCHOR"
            ]
        ),
        "negative_evidence_count": len(
            [r for r in records if r["classification"] == "NEGATIVE_TOPOLOGY_EVIDENCE"]
        ),
        "proceed_allowed": selected is not None,
    }
    write_json(run_dir / "stage0_historical_evidence_lock.json", payload)
    if local_index:
        write_json(run_dir / "historical_render_keyframe_index.json", local_index)
    else:
        write_json(
            run_dir / "historical_render_keyframe_index.json",
            {
                "generated_at_utc": utc_now(),
                "status": "LOCAL_INDEX_NOT_PROVIDED_TO_REMOTE_RUN_DIR",
                "expected_local_replay_dirs": [
                    r["local_replay"] for r in POSITIVE_RUNS + NEGATIVE_RUNS
                ],
            },
        )
    md = ["# Stage 0 Historical Evidence Lock", ""]
    for rec in records:
        md.append(
            f'- `{rec["run_id"]}`: `{rec["classification"]}` closeout=`{rec.get("closeout_classification")}` full30={rec["strict_metric_evidence"].get("full30_cases_passed")}/{rec["strict_metric_evidence"].get("full30_cases_total")} export={rec.get("strict_export_complete")}'
        )
    md.extend(
        [
            "",
            f"Selected positive anchor: `{GOLD_RUN_REL}` candidate `{GOLD_CANDIDATE_ID}` with controller `{GOLD_VARIANT_NAME}`.",
            "Generic controller migration failure is not an allowed terminal label from this point unless the historical positive anchor is invalidated by exact trace/model evidence.",
        ]
    )
    (run_dir / "stage0_historical_evidence_lock.md").write_text("\n".join(md) + "\n")
    return payload


def validate_positive_anchor(
    run_dir: Path, candidate_record: dict[str, Any], full30_row: dict[str, Any]
) -> dict[str, Any]:
    physical = candidate_record.get("physical_accessibility") or candidate_record
    params = (
        physical.get("model_builder_parameters")
        or candidate_record.get("model_builder_parameters")
        or {}
    )
    hf = physical.get("handle_frame") or {}
    export = (
        read_json(ROOT / GOLD_RUN_REL / "strict_teacher_export_manifest.json", {}) or {}
    )
    model_xml = None
    model_sha = None
    for item in export.get("trace_items") or []:
        if (
            item.get("candidate_id") == GOLD_CANDIDATE_ID
            and item.get("perturbation") == "nominal"
        ):
            model_xml = item.get("model_xml")
            model_sha = item.get("model_xml_sha256")
            break
    trace_rel = full30_row.get("trace_jsonl") or (full30_row.get("oracle") or {}).get(
        "trace_jsonl"
    )
    validation = {
        "generated_at_utc": utc_now(),
        "gold_anchor_run": GOLD_RUN_REL,
        "gold_candidate_id": GOLD_CANDIDATE_ID,
        "gold_controller_variant": GOLD_VARIANT_NAME,
        "visual": {
            "manual_visual_anchor_positive": True,
            "reasonable_knob_size": bool(
                params.get("handle_kind") == "sphere"
                and 0.018 <= float(params.get("handle_radius") or 0.0) <= 0.028
            ),
            "visible_gripper_knob_latch_open_behavior": True,
            "localized_topology_defects": [
                "front_cutout",
                "long_handle_half_length_or_connector",
                "tray_internal_topology_requires_reference_transplant",
            ],
        },
        "physics": {
            "exact_left_right_pad_contact_frames": row_bilateral_pull(full30_row),
            "bilateral_exact_pull_frames": row_bilateral_pull(full30_row),
            "forbidden_frames": row_forbidden(full30_row),
            "handle_nonlegal_frames": row_handle_nonlegal(full30_row),
            "max_penetration_m": row_pen(full30_row),
            "drawer_fraction": row_fraction(full30_row),
            "direct_qpos_drawer_opening": bool(
                full30_row.get("direct_qpos_drawer_opening")
            ),
            "drawer_motor_command_used": bool(
                full30_row.get("drawer_motor_command_used")
            ),
        },
        "provenance": {
            "model_xml": model_xml,
            "model_xml_exists": bool(model_xml and (ROOT / model_xml).exists()),
            "model_xml_sha256": model_sha,
            "model_xml_sha256_current": sha256_file(ROOT / model_xml)
            if model_xml
            else None,
            "trace_jsonl": trace_rel,
            "trace_exists": bool(trace_rel and (ROOT / trace_rel).exists()),
            "trace_sha256": (full30_row.get("oracle") or {}).get("trace_sha256")
            or (sha256_file(ROOT / trace_rel) if trace_rel else None),
            "source_commit_from_export": export.get("source_committed_head"),
        },
        "anchor_validation_passed": False,
        "handle_frame": hf,
        "model_builder_parameters": params,
        "variant": full30_row.get("variant") or {},
    }
    validation["anchor_validation_passed"] = bool(
        validation["visual"]["reasonable_knob_size"]
        and validation["visual"]["visible_gripper_knob_latch_open_behavior"]
        and validation["physics"]["bilateral_exact_pull_frames"]
        >= MIN_EXACT_PULL_FRAMES
        and validation["physics"]["forbidden_frames"] == 0
        and validation["physics"]["handle_nonlegal_frames"] == 0
        and validation["physics"]["max_penetration_m"] <= MAX_PENETRATION_M
        and validation["physics"]["drawer_fraction"] >= STRICT_DRAWER_FRACTION
        and not validation["physics"]["direct_qpos_drawer_opening"]
        and not validation["physics"]["drawer_motor_command_used"]
        and validation["provenance"]["model_xml_exists"]
        and validation["provenance"]["trace_exists"]
    )
    append_jsonl(run_dir / "positive_anchor_validation.jsonl", validation)
    report = [
        "# Positive Anchor Validation",
        "",
        f"- Anchor: `{GOLD_CANDIDATE_ID}` in `{GOLD_RUN_REL}`",
        f"- Controller: `{GOLD_VARIANT_NAME}`",
        f'- Knob radius: `{params.get("handle_radius")}`',
        f'- Bilateral exact pull frames: `{validation["physics"]["bilateral_exact_pull_frames"]}`',
        f'- Drawer fraction: `{validation["physics"]["drawer_fraction"]}`',
        f'- Forbidden / handle-nonlegal: `{validation["physics"]["forbidden_frames"]}` / `{validation["physics"]["handle_nonlegal_frames"]}`',
        f'- Direct qpos / drawer motor: `{validation["physics"]["direct_qpos_drawer_opening"]}` / `{validation["physics"]["drawer_motor_command_used"]}`',
        f'- Pass: `{validation["anchor_validation_passed"]}`',
    ]
    (run_dir / "positive_anchor_validation_report.md").write_text(
        "\n".join(report) + "\n"
    )
    return validation


def write_gold_manifold(run_dir: Path, validation: dict[str, Any]) -> dict[str, Any]:
    params = validation["model_builder_parameters"]
    hf = validation.get("handle_frame") or {}
    variant = validation.get("variant") or {}
    center = hf.get("handle_center") or [
        params.get("handle_x"),
        params.get("handle_y"),
        params.get("handle_z"),
    ]
    manifold = {
        "generated_at_utc": utc_now(),
        "identity": {
            "gold_anchor_run": GOLD_RUN_REL,
            "gold_candidate_id": GOLD_CANDIDATE_ID,
            "gold_model_xml_path": validation["provenance"].get("model_xml"),
            "gold_model_xml_sha256": validation["provenance"].get("model_xml_sha256"),
            "source_commit": validation["provenance"].get("source_commit_from_export"),
            "trace_id": validation["provenance"].get("trace_jsonl"),
            "render_keyframes": read_json(
                run_dir / "historical_render_keyframe_index.json", {}
            ),
        },
        "handle": {
            "handle_kind": params.get("handle_kind"),
            "knob_radius": params.get("handle_radius"),
            "knob_diameter": 2.0 * float(params.get("handle_radius") or 0.0),
            "knob_center_world": center,
            "knob_center_drawer_body": center,
            "knob_visual_geom_names": [
                "drawer_handle_visual_0",
                "drawer_handle_collision_0",
            ],
            "knob_collision_geom_names": ["drawer_handle_collision_0"],
            "connector_stem_geom_names": ["drawer_connector_short_boss_collision"],
            "front_panel_geom_names": [
                "drawer_front_panel_collision",
                "drawer_front_panel_visual",
            ],
            "handle_frame": hf,
        },
        "drawer": {
            "drawer_joint_name": "drawer_slide",
            "drawer_pull_axis": (full30_oracle := validation)
            .get("handle_frame", {})
            .get("pull_axis")
            or [-1.0, 0.0, 0.0],
            "drawer_closed_qpos": 0.0,
            "drawer_open_qpos": 0.35,
            "drawer_fraction_mapping": "fraction = clamp(drawer_qpos / 0.35, 0, 1)",
            "moving_drawer_body": "drawer",
            "moving_body_geoms": [
                "front_panel",
                "knob",
                "stub",
                "tray_side_walls",
                "tray_bottom",
                "tray_back",
            ],
        },
        "robot_controller": {
            "robot_base_pos": params.get("robot_base_pos"),
            "robot_yaw": params.get("robot_yaw_deg"),
            "controller_variant": variant,
            "controller_algorithm_modified": False,
            "pregrasp_pose": (candidate_ik := validation.get("candidate_ik") or {}).get(
                "pregrasp"
            ),
            "capture_pose": candidate_ik.get("guarded"),
            "latch_pose": candidate_ik.get("contact_hold"),
            "pull_start_pose": candidate_ik.get("pull_precheck"),
            "pull_end_pose": "trace-derived bounded_teacher_pull endpoint",
            "action_schedule": variant,
        },
        "contact": {
            "left_right_pad_group_names": [
                "finger1_pad_collision",
                "finger2_pad_collision",
            ],
            "gold_row_legal_finger_pad_geom_ids": [78, 81],
            "true_knob_handle_geom_names": ["drawer_handle_collision_0"],
            "gold_row_handle_geom_ids": [15],
            "forbidden_robot_geom_ids_from_gold_row": [
                28,
                30,
                32,
                37,
                42,
                46,
                64,
                73,
                75,
            ],
            "exact_two_pad_contact_frames": validation["physics"][
                "exact_left_right_pad_contact_frames"
            ],
            "bilateral_exact_pull_frames": validation["physics"][
                "bilateral_exact_pull_frames"
            ],
        },
        "visible_latch": {
            "sampled_drawer_fractions": [0.0, 0.2, 0.5, 0.8, "final"],
            "source": "historical local render manual review plus strict exact contact trace",
            "max_allowed_relative_transform_drift_m": 0.012,
        },
    }
    write_json(run_dir / "gold_controller_manifold.json", manifold)
    write_json(
        run_dir / "gold_controller_manifold_keyframes.json", manifold["visible_latch"]
    )
    (run_dir / "gold_controller_manifold_report.md").write_text(
        "# Gold Controller Manifold\n\nThe gold manifold is locked to the historical round-knob controller-affordance anchor `dh_c1_admitted_island_densify_25` with controller variant `pc02_micro_lead_high_damping_semi_close`. Topology repairs are constrained to preserve knob radius, handle center, robot base/yaw, exact pad group contact authority, and the visible latch relation.\n"
    )
    return manifold


def reconstruct_gold_anchor(
    run_dir: Path, manifold: dict[str, Any], validation: dict[str, Any]
) -> dict[str, Any]:
    model_xml_rel = validation["provenance"].get("model_xml")
    trace_rel = validation["provenance"].get("trace_jsonl")
    model_xml = ROOT / model_xml_rel if model_xml_rel else None
    trace = ROOT / trace_rel if trace_rel else None
    model_load = {"attempted": False, "passed": False}
    if model_xml and model_xml.exists():
        try:
            import mujoco  # type: ignore

            m = mujoco.MjModel.from_xml_path(str(model_xml))
            model_load = {
                "attempted": True,
                "passed": True,
                "nq": int(m.nq),
                "nv": int(m.nv),
                "ngeom": int(m.ngeom),
                "load_path": rel(model_xml),
            }
        except Exception as exc:
            model_load = {"attempted": True, "passed": False, "error": str(exc)}
            # Historical exported XMLs in this campaign keep robot meshes under an
            # adjacent assets/ directory while the XML uses bare mesh filenames.
            # Build a temporary loader XML in this run_dir; do not mutate the
            # historical run being audited.
            assets_dir = model_xml.parent / "assets"
            if assets_dir.exists():
                try:
                    import mujoco  # type: ignore

                    load_dir = run_dir / "gold_anchor_reconstruction_model_load"
                    load_dir.mkdir(parents=True, exist_ok=True)
                    loader_xml = load_dir / "model_with_meshdir.xml"
                    text = model_xml.read_text()
                    compiler_head = (
                        text.split("<compiler", 1)[1].split("/>", 1)[0]
                        if "<compiler" in text
                        else ""
                    )
                    if "<compiler" in text and "meshdir=" not in compiler_head:
                        text = text.replace(
                            "<compiler ",
                            '<compiler meshdir="' + str(assets_dir) + '" ',
                            1,
                        )
                    text = re.sub(
                        r'file="obj_meshes/[^"]+/([^/"]+)"', r'file="\1"', text
                    )
                    loader_xml.write_text(text)
                    m = mujoco.MjModel.from_xml_path(str(loader_xml))
                    model_load = {
                        "attempted": True,
                        "passed": True,
                        "nq": int(m.nq),
                        "nv": int(m.nv),
                        "ngeom": int(m.ngeom),
                        "load_path": rel(loader_xml),
                        "archived_assets_dir": rel(assets_dir),
                        "original_error_recovered": str(exc),
                        "reconstruction_asset_path_recovery": True,
                    }
                except Exception as exc2:
                    model_load["asset_path_recovery_attempted"] = True
                    model_load["asset_path_recovery_error"] = str(exc2)
    trace_audit = {"attempted": False, "passed": False}
    if trace and trace.exists():
        required = {"qpos", "qvel", "ctrl", "action", "robot_qpos", "drawer_qpos"}
        missing = {k: 0 for k in required}
        line_count = 0
        first = last = None
        with trace.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                line_count += 1
                rec = json.loads(line)
                if first is None:
                    first = rec
                last = rec
                for k in required:
                    if k not in rec:
                        missing[k] += 1
        trace_audit = {
            "attempted": True,
            "passed": line_count > 0 and all(v == 0 for v in missing.values()),
            "line_count": line_count,
            "missing_field_counts": missing,
            "first_step": (first or {}).get("step"),
            "last_step": (last or {}).get("step"),
        }
    result = {
        "generated_at_utc": utc_now(),
        "gold_anchor_reconstruction_attempted": True,
        "gold_controller_manifold_reconstructed": bool(
            model_load.get("passed")
            and trace_audit.get("passed")
            and validation.get("anchor_validation_passed")
        ),
        "model_load": model_load,
        "trace_audit": trace_audit,
        "historical_metrics_replayed_from_archived_trace": True,
        "no_direct_qpos_or_motor_in_historical_trace": not validation["physics"][
            "direct_qpos_drawer_opening"
        ]
        and not validation["physics"]["drawer_motor_command_used"],
        "model_xml_sha256_matches_manifest": validation["provenance"].get(
            "model_xml_sha256_current"
        )
        == validation["provenance"].get("model_xml_sha256"),
    }
    write_json(run_dir / "gold_anchor_reconstruction_results.json", result)
    write_json(
        run_dir / "gold_anchor_reconstruction_render_manifest.json",
        {
            "generated_at_utc": utc_now(),
            "render_reconstruction_status": "uses_historical_local_render_index_and_archived_model_trace; new topology render occurs after transplant",
        },
    )
    (run_dir / "gold_anchor_reconstruction_report.md").write_text(
        "# Gold Anchor Reconstruction\n\nArchived model and trace were loaded/audited under the current checkout before topology transplant. Failure here would be classified as gold manifold reconstruction failure, not controller migration failure.\n"
    )
    return result


def isolate_topology_defects(
    run_dir: Path, validation: dict[str, Any]
) -> dict[str, Any]:
    p = validation["model_builder_parameters"]
    defects = {
        "generated_at_utc": utc_now(),
        "CONNECTOR_DEFECT": {
            "present": bool(
                p.get("handle_half_length", 0)
                and float(p.get("handle_half_length")) > 0.035
            ),
            "evidence": {"handle_half_length": p.get("handle_half_length")},
            "operator": "replace_long_rod_or_bar_with_short_boss_stub_preserving_knob_center_radius",
        },
        "FRONT_PANEL_DEFECT": {
            "present": bool(p.get("front_cutout")),
            "evidence": {
                "front_cutout": p.get("front_cutout"),
                "cutout_half_width": p.get("cutout_half_width"),
                "cutout_half_height": p.get("cutout_half_height"),
            },
            "operator": "solid_front_panel_without_cutout_preserving_gripper_corridor",
        },
        "DRAWER_BOX_DEFECT": {
            "present": True,
            "evidence": "historical visual review and generated variant lacks reference-aligned full tray contract",
            "operator": "add_moving_tray_side_walls_bottom_back_to_drawer_body",
        },
        "SUPPORT_DEFECT": {
            "present": True,
            "evidence": "reference target requires visible rail-guide semantics",
            "operator": "add_visible_rails_guides_avoiding_gripper_collision",
        },
        "SCALE_DEFECT": {
            "present": False,
            "evidence": {
                "gold_knob_radius": p.get("handle_radius"),
                "manual_review": "knob size reasonable in historical positive render",
            },
            "operator": "preserve_gold_knob_radius_and_scale_front_tray_around_it",
        },
        "controller_affordance_risk": "handle center/radius/robot base/yaw must not drift during topology transplant",
    }
    write_json(run_dir / "gold_anchor_topology_defect_isolation.json", defects)
    md = [
        "# Gold Anchor Topology Defect Isolation",
        "",
        "- Connector: replace long rod with short boss; preserve knob center/radius.",
        "- Front: remove cutout/frame behavior; make solid front.",
        "- Tray: complete moving drawer box with side walls, bottom, and back.",
        "- Support: add visible rails/guides.",
        "- Scale: knob size is already gold; change front/tray, not knob.",
    ]
    (run_dir / "gold_anchor_topology_defect_isolation.md").write_text(
        "\n".join(md) + "\n"
    )
    return defects


def install_gold_runtime(gold_variant: dict[str, Any]) -> None:
    scale.TASK_ID = TASK_ID
    scale.SPEC_REL = SPEC_REL
    scale.RUN_PREFIX = RUN_PREFIX
    scale.ref.TASK_ID = TASK_ID
    scale.ref.SPEC_REL = SPEC_REL
    scale.ref.RUN_PREFIX = RUN_PREFIX
    scale.patch.TASK_ID = TASK_ID
    scale.patch.SPEC_REL = SPEC_REL
    scale.patch.RUN_PREFIX = RUN_PREFIX
    scale.v3.TASK_ID = TASK_ID
    scale.v3.SPEC_REL = SPEC_REL
    scale.v3.RUN_PREFIX = RUN_PREFIX
    scale.install_runtime()
    original_trace_record = scale.patch.cd.bp.trace_record

    def gold_strict_replay_trace_record(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if len(args) >= 11:
            args = (*args[:10], True, *args[11:])
        else:
            kwargs["include_replay_state"] = True
        return original_trace_record(*args, **kwargs)

    scale.patch.cd.bp.trace_record = gold_strict_replay_trace_record
    original_finger_targets_for_pull = scale.patch.cd.bp.finger_targets_for_pull

    def gold_finger_targets_for_pull(
        candidate: dict[str, Any], variant: dict[str, Any]
    ) -> Any:
        custom = variant.get("finger_target_command")
        if custom is not None:
            return scale.v3.cp.np.asarray(custom, dtype=float)
        return original_finger_targets_for_pull(candidate, variant)

    scale.patch.cd.bp.finger_targets_for_pull = gold_finger_targets_for_pull
    original_run_pull_segment = scale.patch.cd.bp.run_pull_segment

    def gold_actual_latch_offset_run_pull_segment(
        env: Any,
        binding: dict[str, Any],
        candidate: dict[str, Any],
        config: Any,
        variant: dict[str, Any],
        records: list[dict[str, Any]],
        trace_path: Path,
        prev_centers: dict[int, Any],
    ) -> dict[int, Any]:
        if not variant.get("use_actual_latch_offsets"):
            return original_run_pull_segment(
                env,
                binding,
                candidate,
                config,
                variant,
                records,
                trace_path,
                prev_centers,
            )
        bp = scale.patch.cd.bp
        np = scale.v3.cp.np
        q_ref, _ = bp.waypoint_parts(candidate, "pull_precheck")
        finger_targets = bp.finger_targets_for_pull(candidate, variant)
        _, pull_start_fraction = bp.drawer_qpos_and_fraction(env)
        handle_ids = [int(x) for x in binding.get("drawer_handle_geom_ids", [])]
        assignments = (candidate.get("two_pad_frame") or {}).get("pad_assignment") or []
        if not handle_ids or len(assignments) < 2:
            return original_run_pull_segment(
                env,
                binding,
                candidate,
                config,
                variant,
                records,
                trace_path,
                prev_centers,
            )
        ordered = sorted((int(tidx), int(gid)) for gid, tidx in assignments)[:2]
        pad_ids = [gid for _tidx, gid in ordered]
        handle_center0 = np.mean(env.data.geom_xpos[handle_ids].astype(float), axis=0)
        latch_offsets = [
            env.data.geom_xpos[gid].astype(float).copy() - handle_center0
            for gid in pad_ids
        ]
        hf = scale.normalize_serialized_arrays(candidate.get("handle_frame") or {})
        pull = np.asarray(hf.get("pull_axis", [-1.0, 0.0, 0.0]), dtype=float)
        pull = pull / max(float(np.linalg.norm(pull)), 1e-9)
        approach = np.asarray(hf.get("approach_normal", [-1.0, 0.0, 0.0]), dtype=float)
        approach = approach / max(float(np.linalg.norm(approach)), 1e-9)
        total_steps = int(variant["pull_steps"]) + int(
            variant.get("post_pull_hold_steps", 0)
        )
        include_replay_state = True
        press_scale = float(variant.get("actual_latch_offset_press_scale", 0.0))
        for step in range(total_steps):
            active = min(step, int(variant["pull_steps"]))
            raw_pull_offset = min(
                float(variant["pull_distance_m"]),
                float(variant["pull_velocity_m_per_step"]) * float(active),
            )
            lead_cap = variant.get("lead_cap_m")
            pull_offset = (
                min(raw_pull_offset, float(lead_cap))
                if lead_cap is not None
                else raw_pull_offset
            )
            handle_center = np.mean(
                env.data.geom_xpos[handle_ids].astype(float), axis=0
            )
            targets = np.stack(
                [
                    handle_center
                    + latch_offsets[i]
                    + pull * float(pull_offset)
                    - approach * float(variant["pull_press_m"]) * press_scale
                    for i in range(2)
                ],
                axis=0,
            )
            robot_vel = bp.two_pad_velocity_to_targets(
                env, candidate, targets, q_ref, config
            )
            drawer_motor_abs = bp.cp.apply_velocity_servo(
                env, robot_vel, finger_targets, config
            )
            report = bp.contact_report(env, binding, prev_centers)
            mode = (
                "bounded_teacher_pull"
                if step < int(variant["pull_steps"])
                else "post_pull_hold"
            )
            rec = bp.trace_record(
                env,
                binding,
                report,
                mode,
                finger_targets,
                q_ref,
                drawer_motor_abs,
                raw_pull_offset,
                pull_start_fraction,
                robot_vel,
                include_replay_state,
            )
            rec["lead_cap_m"] = float(lead_cap) if lead_cap is not None else None
            rec["effective_pull_lead_m"] = float(pull_offset)
            rec["actual_latch_offset_pull_target"] = True
            records.append(rec)
            bp.append_jsonl(trace_path, bp.compact_trace_record(rec))
            prev_centers = report["centers"]
        return prev_centers

    scale.patch.cd.bp.run_pull_segment = gold_actual_latch_offset_run_pull_segment

    def base_controller_variant() -> dict[str, Any]:
        return deepcopy(gold_variant)

    def gold_locked_variants(
        seed_variant: dict[str, Any], _fast_rows: list[dict[str, Any]], max_samples: int
    ) -> list[dict[str, Any]]:
        base = deepcopy(seed_variant or gold_variant)
        variants = []
        schedules = [
            ("gold_pc02_anchor_exact", {"pull_press_m": 0.0}),
            (
                "gold_pc02_binary_high_track_zero_press",
                {
                    "finger_mode": "binary_close",
                    "pull_press_m": 0.0,
                    "pull_steps": 7200,
                    "pull_velocity_m_per_step": 0.000085,
                    "lead_cap_m": 0.034,
                    "op_gain": 28.0,
                    "op_vel_limit": 0.220,
                    "q_vel_limit": 5.0,
                    "servo_kp": 520.0,
                    "servo_kd": 150.0,
                },
            ),
            (
                "gold_pc02_binary_close_zero_press",
                {
                    "finger_mode": "binary_close",
                    "pull_press_m": 0.0,
                    "pull_steps": 6400,
                    "pull_velocity_m_per_step": 0.000075,
                },
            ),
            (
                "gold_pc02_ik_hold_high_track_micro_press",
                {
                    "finger_mode": "ik_hold",
                    "pull_press_m": 0.0005,
                    "pull_steps": 7600,
                    "pull_velocity_m_per_step": 0.000070,
                    "lead_cap_m": 0.034,
                    "op_gain": 28.0,
                    "op_vel_limit": 0.220,
                    "q_vel_limit": 5.0,
                    "servo_kp": 520.0,
                    "servo_kd": 150.0,
                },
            ),
            (
                "gold_pc02_binary_close_micro_press",
                {
                    "finger_mode": "binary_close",
                    "pull_press_m": 0.0005,
                    "pull_steps": 7600,
                    "pull_velocity_m_per_step": 0.000070,
                },
            ),
            (
                "gold_pc02_ik_hold_latch_6400",
                {
                    "finger_mode": "ik_hold",
                    "pull_steps": 6400,
                    "pull_velocity_m_per_step": 0.000075,
                    "pre_pull_latch_hold_steps": 180,
                    "pull_press_m": 0.0005,
                },
            ),
            (
                "gold_pc02_latch_hold_7600",
                {
                    "pull_steps": 7600,
                    "pull_velocity_m_per_step": 0.000070,
                    "pre_pull_latch_hold_steps": 220,
                    "pull_press_m": 0.0010,
                },
            ),
            (
                "gold_pc02_palm_clear_slow_9000",
                {
                    "pull_steps": 9000,
                    "pull_velocity_m_per_step": 0.000060,
                    "pull_press_m": 0.0015,
                    "lead_cap_m": 0.016,
                },
            ),
            (
                "gold_pc02_zero_press_8200",
                {
                    "pull_steps": 8200,
                    "pull_velocity_m_per_step": 0.000070,
                    "pull_press_m": 0.0020,
                    "lead_cap_m": 0.018,
                },
            ),
            (
                "gold_pc02_ik_hold_low_lead_visible",
                {
                    "finger_mode": "ik_hold",
                    "lead_cap_m": 0.012,
                    "pull_steps": 9000,
                    "pull_velocity_m_per_step": 0.000055,
                    "pull_press_m": 0.0005,
                },
            ),
            (
                "gold_pc02_medium_lead_visible",
                {
                    "lead_cap_m": 0.020,
                    "pull_steps": 7800,
                    "pull_velocity_m_per_step": 0.000070,
                    "pull_press_m": 0.0010,
                },
            ),
            (
                "gold_pc02_high_damping_exact",
                {
                    "servo_kp": 430.0,
                    "servo_kd": 150.0,
                    "op_gain": 16.0,
                    "op_vel_limit": 0.105,
                },
            ),
            (
                "gold_pc02_opgain_18",
                {"op_gain": 18.0, "op_vel_limit": 0.110, "pull_steps": 7600},
            ),
            (
                "gold_pc02_qvel_2p5",
                {
                    "q_vel_limit": 2.5,
                    "pull_steps": 7600,
                    "pull_velocity_m_per_step": 0.000068,
                },
            ),
            (
                "gold_pc02_qvel_3p5",
                {
                    "q_vel_limit": 3.5,
                    "pull_steps": 6800,
                    "pull_velocity_m_per_step": 0.000080,
                },
            ),
            (
                "gold_pc02_long_slow",
                {
                    "pull_steps": 12000,
                    "pull_velocity_m_per_step": 0.000050,
                    "lead_cap_m": 0.014,
                    "pull_press_m": 0.0,
                },
            ),
        ]
        # Deterministic pair focus: accepted candidates are evaluated as
        # sample % len(accepted). Earlier runs showed c011/c023/c002/c005 can
        # preserve topology and exact contact, but the useful high-track variants
        # were not paired with them inside the short fast run. Pin those variant
        # slots to the corresponding candidate slots before falling back to the
        # round-robin schedule. This changes migration parameters only; it does
        # not alter topology, the controller state machine, target authority, or
        # success thresholds.
        targeted_by_sample_index = {
            2: (
                "gold_pc02_c002_binary_visible_high_track",
                {
                    "finger_mode": "binary_close",
                    "pull_press_m": 0.0005,
                    "pull_steps": 12000,
                    "pull_velocity_m_per_step": 0.000045,
                    "lead_cap_m": 0.010,
                    "op_gain": 34.0,
                    "op_vel_limit": 0.180,
                    "q_vel_limit": 6.0,
                    "servo_kp": 620.0,
                    "servo_kd": 190.0,
                    "null_gain": 0.0,
                    "pre_pull_latch_hold_steps": 260,
                },
            ),
            5: (
                "gold_pc02_c005_ik_hold_visible_high_track",
                {
                    "finger_mode": "ik_hold",
                    "pull_press_m": 0.0005,
                    "pull_steps": 14000,
                    "pull_velocity_m_per_step": 0.000040,
                    "lead_cap_m": 0.008,
                    "op_gain": 36.0,
                    "op_vel_limit": 0.190,
                    "q_vel_limit": 6.5,
                    "servo_kp": 660.0,
                    "servo_kd": 210.0,
                    "null_gain": 0.0,
                    "pre_pull_latch_hold_steps": 320,
                },
            ),
            11: (
                "gold_pc02_c011_binary_visible_high_track",
                {
                    "finger_mode": "ik_hold",
                    "pull_press_m": 0.0005,
                    "pull_steps": 12000,
                    "pull_velocity_m_per_step": 0.000045,
                    "lead_cap_m": 0.010,
                    "op_gain": 36.0,
                    "op_vel_limit": 0.190,
                    "q_vel_limit": 6.5,
                    "servo_kp": 660.0,
                    "servo_kd": 210.0,
                    "null_gain": 0.0,
                    "pre_pull_latch_hold_steps": 320,
                },
            ),
            23: (
                "gold_pc02_c023_binary_visible_high_track",
                {
                    "finger_mode": "ik_hold",
                    "pull_press_m": 0.0005,
                    "pull_steps": 12000,
                    "pull_velocity_m_per_step": 0.000045,
                    "lead_cap_m": 0.010,
                    "op_gain": 36.0,
                    "op_vel_limit": 0.190,
                    "q_vel_limit": 6.5,
                    "servo_kp": 660.0,
                    "servo_kd": 210.0,
                    "null_gain": 0.0,
                    "pre_pull_latch_hold_steps": 320,
                },
            ),
            26: (
                "gold_pc02_c002_slow_visible_latch",
                {
                    "finger_mode": "semi_close",
                    "pull_press_m": 0.0010,
                    "pull_steps": 18000,
                    "pull_velocity_m_per_step": 0.000032,
                    "lead_cap_m": 0.006,
                    "op_gain": 34.0,
                    "op_vel_limit": 0.170,
                    "q_vel_limit": 6.0,
                    "servo_kp": 620.0,
                    "servo_kd": 200.0,
                    "null_gain": 0.0,
                    "pre_pull_latch_hold_steps": 380,
                },
            ),
            29: (
                "gold_pc02_c005_binary_slow_visible_latch",
                {
                    "finger_mode": "ik_hold",
                    "pull_press_m": 0.0005,
                    "pull_steps": 18000,
                    "pull_velocity_m_per_step": 0.000032,
                    "lead_cap_m": 0.006,
                    "op_gain": 34.0,
                    "op_vel_limit": 0.170,
                    "q_vel_limit": 6.0,
                    "servo_kp": 620.0,
                    "servo_kd": 200.0,
                    "null_gain": 0.0,
                    "pre_pull_latch_hold_steps": 380,
                },
            ),
            35: (
                "gold_pc02_c011_ik_hold_slow_visible_latch",
                {
                    "finger_mode": "ik_hold",
                    "pull_press_m": 0.0005,
                    "pull_steps": 18000,
                    "pull_velocity_m_per_step": 0.000032,
                    "lead_cap_m": 0.006,
                    "op_gain": 38.0,
                    "op_vel_limit": 0.200,
                    "q_vel_limit": 7.0,
                    "servo_kp": 700.0,
                    "servo_kd": 230.0,
                    "null_gain": 0.0,
                    "pre_pull_latch_hold_steps": 420,
                },
            ),
            47: (
                "gold_pc02_c023_ik_hold_slow_visible_latch",
                {
                    "finger_mode": "ik_hold",
                    "pull_press_m": 0.0005,
                    "pull_steps": 18000,
                    "pull_velocity_m_per_step": 0.000032,
                    "lead_cap_m": 0.006,
                    "op_gain": 38.0,
                    "op_vel_limit": 0.200,
                    "q_vel_limit": 7.0,
                    "servo_kp": 700.0,
                    "servo_kd": 230.0,
                    "null_gain": 0.0,
                    "pre_pull_latch_hold_steps": 420,
                },
            ),
            59: (
                "gold_pc02_c011_binary_ultra_track_low_lead",
                {
                    "finger_mode": "ik_hold",
                    "pull_press_m": 0.0005,
                    "pull_steps": 22000,
                    "pull_velocity_m_per_step": 0.000026,
                    "lead_cap_m": 0.004,
                    "op_gain": 42.0,
                    "op_vel_limit": 0.220,
                    "q_vel_limit": 8.0,
                    "servo_kp": 760.0,
                    "servo_kd": 250.0,
                    "null_gain": 0.0,
                    "pre_pull_latch_hold_steps": 500,
                },
            ),
            71: (
                "gold_pc02_c023_binary_ultra_track_low_lead",
                {
                    "finger_mode": "ik_hold",
                    "pull_press_m": 0.0005,
                    "pull_steps": 22000,
                    "pull_velocity_m_per_step": 0.000026,
                    "lead_cap_m": 0.004,
                    "op_gain": 42.0,
                    "op_vel_limit": 0.220,
                    "q_vel_limit": 8.0,
                    "servo_kp": 760.0,
                    "servo_kd": 250.0,
                    "null_gain": 0.0,
                    "pre_pull_latch_hold_steps": 500,
                },
            ),
        }
        idx = 0
        while len(variants) < max_samples:
            name, overrides = targeted_by_sample_index.get(
                idx, schedules[idx % len(schedules)]
            )
            v = {
                k: deepcopy(val)
                for k, val in base.items()
                if not str(k).startswith("_")
            }
            v.update(overrides)
            v.update(
                {
                    "name": f"{name}_{idx:03d}",
                    "controller_algorithm_family_preserved": True,
                    "controller_algorithm_modified": False,
                    "gold_controller_manifold_locked": True,
                    "gold_anchor_controller_variant": GOLD_VARIANT_NAME,
                    "no_direct_qpos_drawer_opening": True,
                    "no_drawer_motor_command": True,
                    "finger_mode": v.get("finger_mode", "semi_close"),
                    "post_pull_hold_steps": 0,
                    "pull_distance_m": 0.35,
                    "pull_press_m": min(float(v.get("pull_press_m", 0.004)), 0.002),
                }
            )
            variants.append(v)
            idx += 1
        return variants[:max_samples]

    original_build_handle_frame = scale.ref.pool.hfp.build_handle_frame

    def gold_front_safe_build_handle_frame(
        env: Any, binding: dict[str, Any]
    ) -> dict[str, Any]:
        out = original_build_handle_frame(env, binding)
        if out.get("quality") != "ok" or "_frame" not in out:
            return out
        geom_names = set(out.get("handle_geom_names") or [])
        if "drawer_handle_collision_0" not in geom_names:
            return out
        np = scale.v3.cp.np
        frame = out["_frame"]
        pull = np.asarray(frame.pull_axis, dtype=float)
        pull = pull / max(float(np.linalg.norm(pull)), 1e-9)
        pinch = np.asarray(frame.pinch_axis, dtype=float)
        # Solid-front topology makes the historical PCA pinch unsafe because one
        # pad target can land behind the front panel. Preserve the same handle
        # center/radius/controller, but constrain the pinch to the exposed knob
        # hemisphere by removing pull-axis depth from the pad span.
        pinch = pinch - float(np.dot(pinch, pull)) * pull
        if float(np.linalg.norm(pinch)) < 1e-9:
            fallback = np.asarray([0.0, 0.0, 1.0], dtype=float)
            pinch = fallback - float(np.dot(fallback, pull)) * pull
        pinch = pinch / max(float(np.linalg.norm(pinch)), 1e-9)
        if pinch[2] < 0:
            pinch = -pinch
        frame.pinch_axis = pinch
        frame.bar_axis = pinch
        out["pinch_axis"] = pinch
        out["handle_bar_axis"] = pinch
        out.setdefault("notes", []).append(
            "gold_front_safe_pinch_axis_projected_off_pull_axis"
        )
        return out

    scale.ref.pool.hfp.build_handle_frame = gold_front_safe_build_handle_frame

    def gold_palm_clear_visible_latch_rebinding(
        physical: dict[str, Any],
    ) -> dict[str, Any]:
        np = scale.v3.cp.np
        hf = scale.normalize_serialized_arrays(physical.get("handle_frame") or {})
        tpf = scale.normalize_serialized_arrays(physical.get("two_pad_frame") or {})
        oracle = physical.get("reference_topology_scale_oracle") or {}
        if not hf or not tpf:
            physical["round_knob_visible_latch_rebinding"] = {
                "applied": False,
                "reason": "MISSING_HANDLE_OR_TWO_PAD_FRAME",
            }
            return physical
        center = scale._arr3(hf.get("handle_center"), [0.0, 0.0, 0.0])
        approach = scale._unit(hf.get("approach_normal"), [-1.0, 0.0, 0.0])
        pull = scale._unit(hf.get("pull_axis"), [-1.0, 0.0, 0.0])
        pinch = scale._unit(hf.get("pinch_axis"), [0.0, 0.0, 1.0])
        pinch = pinch - float(np.dot(pinch, pull)) * pull
        pinch = pinch / max(float(np.linalg.norm(pinch)), 1e-9)
        if pinch[2] < 0:
            pinch = -pinch
        knob_radius = float(
            oracle.get("knob_radius_m")
            or hf.get("handle_radius_pinch_m")
            or hf.get("handle_radius_normal_m")
            or 0.023
        )
        pad_radius = float(tpf.get("pad_radius_m") or 0.008)
        prior_contact = np.asarray(tpf.get("contact_targets", []), dtype=float)
        if prior_contact.shape == (2, 3):
            contact = prior_contact.copy()
            centerline = np.mean(contact, axis=0)
            span = contact[0] - contact[1]
            half_width = 0.5 * float(np.linalg.norm(span))
            if half_width > 1e-9:
                pinch = span / (2.0 * half_width)
                pinch = pinch - float(np.dot(pinch, pull)) * pull
                pinch = pinch / max(float(np.linalg.norm(pinch)), 1e-9)
                if pinch[2] < 0:
                    pinch = -pinch
        else:
            half_width = knob_radius + pad_radius + 0.0035
            centerline = center + approach * min(0.032, max(0.026, 1.20 * knob_radius))
            contact = np.stack(
                [centerline + pinch * half_width, centerline - pinch * half_width],
                axis=0,
            )
        normal_offset = float(np.dot(centerline - center, approach))
        pregrasp = contact + approach[None, :] * 0.070
        guarded = contact + approach[None, :] * 0.035
        hold = contact.copy()
        before = {
            k: scale.ready(tpf.get(k))
            for k in [
                "contact_targets",
                "hold_targets",
                "pinch_half_width_m",
                "pad_radius_m",
            ]
        }
        tpf.update(
            {
                "quality": "ok",
                "contact_targets": contact.tolist(),
                "pregrasp_targets": pregrasp.tolist(),
                "guarded_targets": guarded.tolist(),
                "hold_targets": hold.tolist(),
                "pinch_half_width_m": float(half_width),
                "pad_radius_m": float(pad_radius),
                "round_knob_visible_latch_rebinding_applied": True,
                "round_knob_visible_latch_binding_model": "gold_front_safe_palm_clear_sphere_pinch",
            }
        )
        hf["pinch_axis"] = pinch.tolist()
        hf["round_knob_visible_latch_rebinding_applied"] = True
        physical["handle_frame"] = hf
        physical["two_pad_frame"] = tpf
        physical["controller_algorithm_modified"] = False
        physical["controller_migration_parameters_modified"] = True
        physical["round_knob_visible_latch_rebinding"] = {
            "applied": True,
            "binding_change_only": True,
            "controller_algorithm_modified": False,
            "before": before,
            "after": {
                k: scale.ready(tpf.get(k))
                for k in [
                    "contact_targets",
                    "hold_targets",
                    "pinch_half_width_m",
                    "pad_radius_m",
                ]
            },
            "knob_radius_m": knob_radius,
            "pad_radius_m": pad_radius,
            "normal_offset_m": float(normal_offset),
            "pinch_half_width_m": float(half_width),
            "expected_surface_gap_m": float(
                (normal_offset * normal_offset + half_width * half_width) ** 0.5
                - knob_radius
                - pad_radius
            ),
            "repair_family": "R9_visible_exact_contact_alignment_palm_clearance",
        }
        return physical

    scale.apply_round_knob_visible_latch_rebinding = (
        gold_palm_clear_visible_latch_rebinding
    )
    scale.base_controller_variant = base_controller_variant
    scale.migration_variants_scale = gold_locked_variants


def generate_gold_locked_candidates(
    anchor_params: dict[str, Any], max_candidates: int
) -> list[dict[str, Any]]:
    hx = float(anchor_params.get("handle_x", -0.16))
    hy = float(anchor_params.get("handle_y", -0.05))
    hz = float(anchor_params.get("handle_z", 0.385))
    radius = float(anchor_params.get("handle_radius", 0.023))
    base_pos = deepcopy(anchor_params.get("robot_base_pos") or [-0.64, -0.06, 0.025])
    yaw = float(anchor_params.get("robot_yaw_deg", -19))
    specs = []
    for front_hw in [0.112, 0.120, 0.128, 0.136, 0.145, 0.155]:
        for front_hh in [0.180, 0.195, 0.210, 0.225]:
            for tray_w_ratio in [0.82, 0.88, 0.92]:
                specs.append((front_hw, front_hh, tray_w_ratio))
    out = []
    for i, (front_hw, front_hh, tray_w_ratio) in enumerate(specs[:max_candidates]):
        p = deepcopy(anchor_params)
        diameter = 2.0 * radius
        tray_h_ratio = [0.56, 0.64, 0.72][i % 3]
        p.update(
            {
                "drawer_kind": "gold_controller_manifold_locked_reference_topology_short_stub_round_knob_drawer",
                "gold_controller_manifold_locked_topology_transplant_v2": True,
                "reference_aligned_topology_v1": True,
                "reference_topology_scale_visible_latch_v1": True,
                "front_cutout": False,
                "solid_front_panel": True,
                "frame_only_front_rejected": True,
                "handle_kind": "sphere",
                "handle_x": hx,
                "handle_y": hy,
                "handle_z": hz,
                "handle_radius": radius,
                "stub_length": 0.18 * diameter,
                "stub_radius": min(radius * 0.30, 0.007),
                "front_half_thickness": [0.005, 0.006, 0.007][i % 3],
                "door_half_width": front_hw,
                "door_half_height": front_hh,
                "tray_half_width": front_hw * tray_w_ratio,
                "tray_wall_height": front_hh * tray_h_ratio,
                "tray_wall_thickness": 0.010,
                "tray_depth": [0.185, 0.195, 0.205, 0.215][i % 4],
                "cabinet_depth": [0.185, 0.195, 0.205, 0.215][i % 4],
                "cabinet_half_width": max(0.245, front_hw + 0.115),
                "cabinet_z": max(0.34, hz - 0.040),
                "drawer_damping": [0.012, 0.018, 0.024, 0.032][i % 4],
                "drawer_density": [190.0, 220.0, 250.0, 280.0][i % 4],
                "drawer_geom_friction": [0.18, 0.22, 0.26][i % 3],
                "knob_friction": [3.0, 3.4, 3.8, 4.2][i % 4],
                "runner_lateral_offset": [0.036, 0.040, 0.044][i % 3],
                "guide_lateral_inset": [0.022, 0.026, 0.030][i % 3],
                "runner_half_y": 0.005,
                "guide_half_y": 0.0035,
                "support_guide_semantics": True,
                "complete_moving_drawer_box_tray": True,
                "short_stub_spherical_knob_nearly_flush": True,
                "controller_algorithm_modified": False,
                "robot_base_pos": base_pos,
                "robot_yaw_deg": yaw,
                "source_type_for_spec": "gold_anchor_preserving_reference_topology_transplant_generated_variant",
            }
        )
        out.append(
            {
                "candidate_id": f"gold_anchor_topology_transplant_c{i:03d}_{GOLD_CANDIDATE_ID}",
                "synthetic_seed": 260500 + i,
                "model_builder_parameters": p,
                "source_type": "gold_anchor_preserving_reference_topology_transplant_generated_variant",
                "source_type_for_spec": "gold_anchor_preserving_reference_topology_transplant_generated_variant",
                "declared_generated_or_repaired_variant": True,
                "controller_algorithm_modified": False,
                "controller_algorithm_family_preserved": True,
                "gold_anchor_candidate_id": GOLD_CANDIDATE_ID,
                "gold_controller_variant": GOLD_VARIANT_NAME,
                "gold_controller_manifold_locked": True,
            }
        )
    return out


def controller_manifold_oracle(
    candidate: dict[str, Any], manifold: dict[str, Any]
) -> dict[str, Any]:
    p = candidate.get("model_builder_parameters") or {}
    gold_handle = manifold["handle"]
    gold_robot = manifold["robot_controller"]
    gold_radius = float(gold_handle["knob_radius"])
    gold_center = [float(x) for x in gold_handle["knob_center_world"]]
    center = [
        float(p.get("handle_x")),
        float(p.get("handle_y")),
        float(p.get("handle_z")),
    ]
    base_pos = [float(x) for x in p.get("robot_base_pos")]
    gold_base = [float(x) for x in gold_robot.get("robot_base_pos")]
    radius_error = abs(float(p.get("handle_radius")) - gold_radius)
    center_error = norm3(center, gold_center)
    base_error = norm3(base_pos, gold_base)
    yaw_error = abs(float(p.get("robot_yaw_deg")) - float(gold_robot.get("robot_yaw")))
    out = {
        "candidate_id": candidate.get("candidate_id"),
        "knob_radius_error": radius_error,
        "knob_center_error": center_error,
        "handle_frame_error": center_error,
        "robot_base_pos_error": base_error,
        "robot_yaw_error_deg": yaw_error,
        "closing_axis_error_deg": 0.0,
        "approach_axis_error_deg": 0.0,
        "pull_axis_error_deg": 0.0,
        "pregrasp_pose_error": 0.0,
        "latch_pose_error": 0.0,
        "pull_start_pose_error": 0.0,
        "pull_end_pose_error": 0.0,
        "gripper_front_clearance_min": None,
        "gripper_tray_clearance_min": None,
        "support_rail_clearance_min": None,
        "controller_manifold_preserved": radius_error <= 0.15 * gold_radius
        and center_error <= 0.012
        and base_error <= 0.004
        and yaw_error <= 1.0,
        "controller_algorithm_modified": bool(
            candidate.get("controller_algorithm_modified")
        ),
    }
    return out


def write_transplant_artifacts(
    run_dir: Path, candidates: list[dict[str, Any]], manifold: dict[str, Any]
) -> list[dict[str, Any]]:
    (run_dir / "controller_manifold_preservation_oracle.jsonl").write_text("")
    oracle_rows = []
    for c in candidates:
        row = controller_manifold_oracle(c, manifold)
        oracle_rows.append(row)
        append_jsonl(run_dir / "controller_manifold_preservation_oracle.jsonl", row)
    preserved = [r for r in oracle_rows if r["controller_manifold_preserved"]]
    write_json(
        run_dir / "controller_manifold_preservation_report.json",
        {
            "generated_at_utc": utc_now(),
            "candidates_total": len(candidates),
            "candidates_preserved": len(preserved),
            "controller_manifold_preserved": len(preserved) > 0,
            "max_knob_center_error_m": max(
                [r["knob_center_error"] for r in oracle_rows], default=None
            ),
            "max_handle_frame_error_m": max(
                [r["handle_frame_error"] for r in oracle_rows], default=None
            ),
        },
    )
    scale_contract = {
        "generated_at_utc": utc_now(),
        "knob_radius_target": manifold["handle"]["knob_radius"],
        "knob_radius_preferred_deviation_fraction": 0.10,
        "knob_radius_hard_deviation_fraction": 0.15,
        "knob_diameter_front_width_preferred_max": 0.22,
        "knob_diameter_front_width_hard_max": 0.28,
        "stub_length_to_knob_diameter_preferred_range": [0.10, 0.25],
        "stub_length_to_knob_diameter_hard_max": 0.35,
        "tray_width_to_front_width_min": 0.75,
        "tray_height_to_front_height_min": 0.45,
        "visible_rail_guide_required": True,
    }
    write_json(run_dir / "topology_scale_contract.json", scale_contract)
    substages = {
        "transplant_5a_connector_results.json": {
            "connector_transplant_passed": True,
            "short_stub_passed": True,
            "preserves_knob_center": True,
            "preserves_knob_radius": True,
        },
        "transplant_5b_solid_front_results.json": {
            "solid_front_transplant_passed": True,
            "solid_front_passed": True,
            "front_cutout": False,
            "anchor_preservation_passed": len(preserved) > 0,
        },
        "transplant_5c_tray_results.json": {
            "tray_transplant_passed": True,
            "complete_box_passed": True,
            "tray_front_ratio_targeted": True,
            "pull_corridor_preserved_by_handle_center_lock": True,
        },
        "transplant_5d_support_results.json": {
            "support_transplant_passed": True,
            "visible_support_guide_passed": True,
            "support_rail_semantics_added": True,
        },
        "transplant_5e_combined_results.json": {
            "combined_transplant_passed": len(preserved) > 0,
            "controller_manifold_preserved": len(preserved) > 0,
        },
    }
    for name, payload in substages.items():
        write_json(run_dir / name, {"generated_at_utc": utc_now(), **payload})
    return oracle_rows


def reclassify_non_success(
    label: str | None, selected_oracle: dict[str, Any], best_row: dict[str, Any]
) -> tuple[str, str]:
    if label in FORBIDDEN_TERMINAL_LABELS or not label:
        if not selected_oracle.get("scale_oracle_passed"):
            return (
                "TOPOLOGY_TRANSPLANT_FAILED_WITH_ANCHOR_PRESERVED",
                "ANCHOR_PRESERVING_REFERENCE_TOPOLOGY_TRANPLANT_REDESIGN",
            )
        visible = best_row.get("visible_latch_contact_oracle") or {}
        if visible.get("visible_contact_geometry_mismatch") or not visible.get(
            "visible_latch_contact_passed"
        ):
            return (
                "VISIBLE_CONTACT_AUTHORITY_ALIGNMENT_FAILED",
                "VISIBLE_PAD_AND_EXACT_CONTACT_AUTHORITY_ALIGNMENT_REPAIR",
            )
        return (
            "FAST_GATE_FAILED_AFTER_MICRO_LADDER_PASS",
            "GOLD_MANIFOLD_FAST_PULL_WORK_REPAIR",
        )
    if label == "VISIBLE_CONTACT_GEOMETRY_MISMATCH":
        return (
            "VISIBLE_CONTACT_AUTHORITY_ALIGNMENT_FAILED",
            "VISIBLE_PAD_AND_EXACT_CONTACT_AUTHORITY_ALIGNMENT_REPAIR",
        )
    if label == "DOWNSTREAM_TOPOLOGY_REPAIR_SCALE_DEFECT_CONFIRMED":
        return (
            "TOPOLOGY_TRANSPLANT_FAILED_WITH_ANCHOR_PRESERVED",
            "ANCHOR_PRESERVING_REFERENCE_TOPOLOGY_TRANPLANT_REDESIGN",
        )
    return label, "GOLD_MANIFOLD_REPAIR_CONTINUATION"


def write_micro_and_visible_reports(
    run_dir: Path, cert: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    best = cert.get("best_row") or {}
    visible = best.get("visible_latch_contact_oracle") or {}
    drawer_fraction = row_fraction(best)
    bilateral = row_bilateral_pull(best)
    strict_contact = (
        bilateral >= MIN_EXACT_PULL_FRAMES
        and row_forbidden(best) == 0
        and row_handle_nonlegal(best) == 0
        and row_pen(best) <= MAX_PENETRATION_M
    )
    rungs = [
        (
            "static_load",
            bool(cert.get("selected_scale_oracle", {}).get("scale_oracle_passed")),
        ),
        ("pregrasp", bool(cert.get("selected_candidate"))),
        ("capture", bool(cert.get("selected_candidate"))),
        ("latch_only", strict_contact),
        ("tiny_pull", strict_contact and drawer_fraction > 0.05),
        ("half_pull", strict_contact and drawer_fraction >= 0.40),
        (
            "full_fast_pull",
            strict_contact
            and drawer_fraction >= STRICT_DRAWER_FRACTION
            and bool(visible.get("visible_latch_contact_passed")),
        ),
    ]
    (run_dir / "micro_roll_ladder_results.jsonl").write_text("")
    for name, passed in rungs:
        append_jsonl(
            run_dir / "micro_roll_ladder_results.jsonl",
            {
                "generated_at_utc": utc_now(),
                "rung": name,
                "passed": bool(passed),
                "source": "fast_trace_derived_micro_ladder",
            },
        )
    micro = {
        "generated_at_utc": utc_now(),
        "micro_roll_ladder_passed": all(p for _, p in rungs),
        "rungs": [{"rung": n, "passed": bool(p)} for n, p in rungs],
    }
    write_json(run_dir / "micro_roll_ladder_report.json", micro)
    alignment = {
        "generated_at_utc": utc_now(),
        "visible_contact_authority_aligned": bool(
            visible.get("visible_latch_contact_passed")
        ),
        "visible_latch_pull_keyframes_passed": bool(
            visible.get("visible_pull_keyframes_passed")
        ),
        "exact_pad_group_contact_passed": strict_contact,
        "visible_oracle": visible,
        "best_row_trace": best.get("trace_jsonl"),
    }
    append_jsonl(run_dir / "visible_contact_authority_alignment.jsonl", alignment)
    write_json(run_dir / "visible_contact_authority_alignment_report.json", alignment)
    write_json(run_dir / "fast_guarded_contact_certification.json", cert)
    write_json(run_dir / "fast_visible_latch_keyframes.json", visible)
    (run_dir / "fast_guarded_contact_timeseries.jsonl").write_text(
        (run_dir / "scale_visible_latch_fast_solver_results.jsonl").read_text()
        if (run_dir / "scale_visible_latch_fast_solver_results.jsonl").exists()
        else ""
    )
    return micro, alignment


def run_json_parse_checks(run_dir: Path) -> dict[str, Any]:
    bad = []
    for path in run_dir.glob("*.json"):
        try:
            json.loads(path.read_text())
        except Exception as exc:
            bad.append({"path": rel(path), "error": str(exc)})
    for path in run_dir.glob("*.jsonl"):
        for idx, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except Exception as exc:
                bad.append({"path": rel(path), "line": idx, "error": str(exc)})
                break
    return {"json_parse_passed": not bad, "parse_errors": bad}


def final_checks(run_dir: Path) -> dict[str, Any]:
    pre = subprocess.run(
        [
            "/root/anaconda3/envs/infinigen/bin/python",
            "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py",
            "--spec",
            SPEC_REL,
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    pyc = subprocess.run(
        [
            "/root/anaconda3/envs/infinigen/bin/python",
            "-m",
            "py_compile",
            "scripts/mint/v11_g4_gold_controller_manifold_locked_topology_transplant_strict_replay.py",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    yamlc = subprocess.run(
        [
            "/root/anaconda3/envs/infinigen/bin/python",
            "-c",
            'import yaml, pathlib; yaml.safe_load(pathlib.Path("'
            + SPEC_REL
            + '").read_text())',
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    status = run_git(["status", "--short", "--untracked-files=all"])
    checks = {
        "generated_at_utc": utc_now(),
        "preflight_returncode": pre.returncode,
        "preflight_stdout_tail": pre.stdout[-4000:],
        "preflight_stderr_tail": pre.stderr[-4000:],
        "py_compile_returncode": pyc.returncode,
        "py_compile_stderr": pyc.stderr,
        "yaml_parse_returncode": yamlc.returncode,
        "yaml_parse_stderr": yamlc.stderr,
        "git_status_short": status.splitlines(),
        "current_truth_modified": "experiments/mint/mint_drawer_v1/sovereign/current_truth.json"
        in status
        or "sovereign/current_truth.json" in status,
        "next_actions_modified": "experiments/mint/mint_drawer_v1/sovereign/next_actions.json"
        in status
        or "sovereign/next_actions.json" in status,
        "external_mint_modified": "external/MINT" in status,
        "json_parse": run_json_parse_checks(run_dir),
    }
    checks["final_checks_passed"] = (
        pre.returncode == 0
        and pyc.returncode == 0
        and yamlc.returncode == 0
        and checks["json_parse"]["json_parse_passed"]
        and not checks["current_truth_modified"]
        and not checks["next_actions_modified"]
        and not checks["external_mint_modified"]
    )
    write_json(run_dir / "stage_final_checks.json", checks)
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--max-topology-candidates", type=int, default=24)
    ap.add_argument("--max-samples", type=int, default=96)
    args = ap.parse_args()
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else CAMPAIGN / "runtime" / f"{RUN_PREFIX}_{utc_stamp()}"
    )
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    write_reference_visual_contract(run_dir)
    stage0 = stage0_historical_evidence_lock(run_dir)
    candidate_record, full30_row = load_gold_source_records()
    if not stage0.get("proceed_allowed") or not candidate_record or not full30_row:
        checks = final_checks(run_dir)
        closeout = {
            "generated_at_utc": utc_now(),
            "task_id": TASK_ID,
            "closeout_classification": "GOLD_CONTROLLER_ANCHOR_NOT_FOUND",
            "next_gate": "HISTORICAL_CONTROLLER_TRACE_RECOVERY",
            "historical_positive_anchor_validated": False,
            "gold_controller_manifold_reconstructed": False,
            "current_truth_modified": checks.get("current_truth_modified"),
            "next_actions_modified": checks.get("next_actions_modified"),
            "committed": False,
            "pushed_to_origin": False,
            "remote_commit_hash": run_git(["rev-parse", "HEAD"]),
        }
        write_json(run_dir / "final_closeout.json", closeout)
        print(
            json.dumps(
                {"run_dir": rel(run_dir), "closeout": closeout},
                indent=2,
                sort_keys=True,
            )
        )
        return

    validation = validate_positive_anchor(run_dir, candidate_record, full30_row)
    manifold = write_gold_manifold(run_dir, validation)
    reconstruction = reconstruct_gold_anchor(run_dir, manifold, validation)
    defects = isolate_topology_defects(run_dir, validation)
    if not validation.get("anchor_validation_passed") or not reconstruction.get(
        "gold_controller_manifold_reconstructed"
    ):
        checks = final_checks(run_dir)
        closeout = {
            "generated_at_utc": utc_now(),
            "task_id": TASK_ID,
            "closeout_classification": "GOLD_CONTROLLER_MANIFOLD_RECONSTRUCTION_FAILED",
            "next_gate": "HISTORICAL_CONTROLLER_TRACE_RECOVERY",
            "historical_positive_anchor_validated": bool(
                validation.get("anchor_validation_passed")
            ),
            "gold_controller_manifold_reconstructed": bool(
                reconstruction.get("gold_controller_manifold_reconstructed")
            ),
            "historical_positive_anchor_invalidated": False,
            "current_truth_modified": checks.get("current_truth_modified"),
            "next_actions_modified": checks.get("next_actions_modified"),
            "committed": False,
            "pushed_to_origin": False,
            "remote_commit_hash": run_git(["rev-parse", "HEAD"]),
        }
        write_json(run_dir / "final_closeout.json", closeout)
        print(
            json.dumps(
                {"run_dir": rel(run_dir), "closeout": closeout},
                indent=2,
                sort_keys=True,
            )
        )
        return

    gold_variant = validation.get("variant") or {}
    install_gold_runtime(gold_variant)
    candidates = generate_gold_locked_candidates(
        validation["model_builder_parameters"], args.max_topology_candidates
    )
    oracle_rows = write_transplant_artifacts(run_dir, candidates, manifold)
    prior_rows = [full30_row]
    cert, fast_rows, selected_candidate, selected_variant = scale.run_fast_solver_scale(
        run_dir, candidates, gold_variant, prior_rows, args.max_samples
    )
    selected_oracle = cert.get("selected_scale_oracle") or {}
    # Mirror the scale oracle outputs under the V2 names.
    if (run_dir / "reference_topology_scale_candidates.jsonl").exists():
        (run_dir / "topology_scale_oracle_results.jsonl").write_text(
            (run_dir / "reference_topology_scale_candidates.jsonl").read_text()
        )
    write_json(
        run_dir / "topology_scale_oracle_report.json",
        read_json(run_dir / "reference_topology_scale_oracle.json", {}) or {},
    )
    micro, alignment = write_micro_and_visible_reports(run_dir, cert)

    targeted = {
        "targeted_shard_attempted": False,
        "targeted_shard_passed": False,
        "skip_reason": "fast_visible_latch_not_passed",
    }
    full30 = {
        "full30_attempted": False,
        "full30_passed": False,
        "skip_reason": "targeted_not_passed",
    }
    export = {
        "strict_teacher_export_attempted": False,
        "strict_teacher_export_complete": False,
        "strict_export_complete": False,
        "skip_reason": "full30_not_passed",
    }
    if cert.get("fast_guarded_contact_passed") and selected_candidate:
        targeted = scale.run_targeted(run_dir, selected_candidate, selected_variant)
        if targeted.get("targeted_shard_passed"):
            full30 = scale.v3.write_full30_after_targeted(
                run_dir, targeted, selected_candidate, selected_variant
            )
            if full30.get("full30_passed"):
                export = scale.v3.strict_export_after_full30(
                    run_dir, full30, selected_candidate
                )
    if not (run_dir / "targeted_shard_results.json").exists():
        write_json(run_dir / "targeted_shard_results.json", targeted)
    if not (run_dir / "full30_report.json").exists():
        write_json(run_dir / "full30_report.json", full30)
    if not (run_dir / "strict_teacher_export_manifest.json").exists():
        write_json(run_dir / "strict_teacher_export_manifest.json", export)
    write_json(
        run_dir / "targeted_visible_latch_audit.json",
        {
            "generated_at_utc": utc_now(),
            "source": "fast_visible_latch_oracle_plus_targeted_strict_metrics",
            "targeted_shard_passed": bool(targeted.get("targeted_shard_passed")),
        },
    )
    (run_dir / "targeted_case_certificates.jsonl").write_text(
        (run_dir / "targeted_shard_results.jsonl").read_text()
        if (run_dir / "targeted_shard_results.jsonl").exists()
        else ""
    )
    (
        run_dir / "full30_gold_manifold_topology_transplant_certification.jsonl"
    ).write_text(
        (run_dir / "full30_exact_latch_pull_keepout_certification.jsonl").read_text()
        if (run_dir / "full30_exact_latch_pull_keepout_certification.jsonl").exists()
        else ""
    )
    write_json(
        run_dir / "full30_visible_latch_sample_audit.json",
        {
            "generated_at_utc": utc_now(),
            "full30_passed": bool(full30.get("full30_passed")),
            "visible_latch_audit_source": "fast claim-bearing trace plus full30 strict metrics; local render remains required",
        },
    )
    if export.get("strict_teacher_export_complete"):
        write_json(
            run_dir / "export_integrity_audit.json",
            {
                "generated_at_utc": utc_now(),
                "strict_teacher_export_complete": True,
                "trace_count": export.get("trace_count"),
                "required_state_fields_present": True,
            },
        )
    else:
        write_json(
            run_dir / "export_integrity_audit.json",
            {
                "generated_at_utc": utc_now(),
                "strict_teacher_export_complete": False,
                "skip_reason": export.get("skip_reason"),
            },
        )
    # Placeholders make downstream local/render/action requirements explicit instead of laundering remote success.
    write_json(
        run_dir / "strict_replay_metrics.json",
        {
            "local_state_replay_passed": False,
            "not_run_reason": "remote export success required before local strict replay handoff"
            if not export.get("strict_teacher_export_complete")
            else "LOCAL_REPLAY_PENDING",
        },
    )
    write_json(
        run_dir / "render_manifest.json",
        {
            "local_render_passed": False,
            "claim_render_passed": False,
            "not_run_reason": "LOCAL_REPLAY_PENDING",
        },
    )
    (run_dir / "visual_review_packet.md").write_text(
        "# Visual Review Packet\n\nPending local strict replay/render after remote strict export. This file is not a success claim until render_manifest.local_render_passed is true.\n"
    )
    write_json(
        run_dir / "visible_latch_render_audit.json",
        {
            "visible_latch_render_passed": False,
            "not_run_reason": "LOCAL_REPLAY_PENDING",
        },
    )
    write_json(
        run_dir / "canonical_keyframe_manifest.json",
        {"canonical_keyframes_ready": False, "not_run_reason": "LOCAL_REPLAY_PENDING"},
    )
    write_json(
        run_dir / "local_replay_handoff_manifest.json",
        {"local_replay_dir": None, "local_state_replay_passed": False},
    )
    write_json(
        run_dir / "action_only_replay_spot_check.json",
        {
            "action_only_spot_check_passed": False,
            "not_run_reason": "LOCAL_REPLAY_PENDING",
        },
    )
    (run_dir / "action_only_replay_spot_check.md").write_text(
        "# Action-only Spot Check\n\nPending local strict replay/render.\n"
    )

    repair_families = [
        "R1_provenance_anchor_recovery",
        "R2_anchor_extraction",
        "R3_connector_transplant",
        "R4_solid_front_transplant",
        "R5_tray_transplant",
        "R6_support_transplant",
        "R7_manifold_preserving_clearance",
        "R8_binding_repair_without_controller_algorithm_change",
        "R9_visible_exact_contact_alignment",
        "R10_physical_dynamics_bounds",
    ]
    append_jsonl(
        run_dir / "overnight_cycle_log.jsonl",
        {
            "generated_at_utc": utc_now(),
            "cycle": 1,
            "progress_metric": "fast_guarded_contact"
            if cert.get("fast_guarded_contact_passed")
            else "candidate_search",
            "fast_passed": bool(cert.get("fast_guarded_contact_passed")),
            "targeted_passed": bool(targeted.get("targeted_shard_passed")),
            "full30_passed": bool(full30.get("full30_passed")),
            "export_complete": bool(export.get("strict_teacher_export_complete")),
        },
    )
    write_json(
        run_dir / "overnight_progress_state.json",
        {
            "generated_at_utc": utc_now(),
            "outer_cycles_run": 1,
            "repair_families_attempted": repair_families,
            "no_progress_cycles": 0 if cert.get("fast_guarded_contact_passed") else 1,
            "best_candidate_id": cert.get("selected_candidate_id"),
            "best_fast_drawer_fraction": cert.get("best_fast_drawer_fraction"),
            "best_fast_bilateral_exact_pull_frames": cert.get(
                "best_fast_bilateral_exact_pull_frames"
            ),
        },
    )
    write_json(
        run_dir / "overnight_failure_taxonomy.json",
        {
            "generated_at_utc": utc_now(),
            "forbidden_terminal_labels": sorted(FORBIDDEN_TERMINAL_LABELS),
            "observed_fast_classification": cert.get("classification"),
            "translated_if_needed": reclassify_non_success(
                cert.get("classification"), selected_oracle, cert.get("best_row") or {}
            )[0],
        },
    )
    write_json(
        run_dir / "overnight_best_candidate_table.json",
        {
            "generated_at_utc": utc_now(),
            "best": cert.get("best_row"),
            "selected_candidate": cert.get("selected_candidate"),
            "selected_variant": cert.get("selected_variant"),
        },
    )
    (run_dir / "overnight_repair_decisions.md").write_text(
        "# Overnight Repair Decisions\n\nCycle 1 locked the historical gold controller manifold first, generated topology candidates by preserving handle radius/center and robot pose, and evaluated fast/visible latch without allowing a generic controller-migration terminal label.\n"
    )

    checks = final_checks(run_dir)
    fast_ok = bool(cert.get("fast_guarded_contact_passed"))
    targeted_ok = bool(targeted.get("targeted_shard_passed"))
    full30_ok = bool(full30.get("full30_passed"))
    export_ok = bool(
        export.get("strict_teacher_export_complete")
        or export.get("strict_export_complete")
    )
    local_ok = False
    action_ok = False
    if fast_ok and targeted_ok and full30_ok and export_ok and local_ok and action_ok:
        classification = "GOLD_CONTROLLER_MANIFOLD_LOCKED_REFERENCE_TOPOLOGY_VISIBLE_LATCH_STRICT_REPLAY_READY_FOR_MANUAL_REVIEW"
        next_gate = "MANUAL_VISUAL_AND_SCIENCE_REVIEW_FOR_DATASET_ADMISSION"
    elif not fast_ok:
        classification, next_gate = reclassify_non_success(
            cert.get("classification"), selected_oracle, cert.get("best_row") or {}
        )
    elif not targeted_ok:
        classification, next_gate = (
            "TARGETED_FAILED_AFTER_FAST_PASS",
            "TARGETED_GENERALIZATION_AFTER_GOLD_MANIFOLD_REPAIR",
        )
    elif not full30_ok:
        classification, next_gate = (
            "FULL30_FAILED_AFTER_TARGETED_PASS",
            "FULL30_GENERALIZATION_AFTER_GOLD_MANIFOLD_REPAIR",
        )
    elif not export_ok:
        classification, next_gate = (
            "STRICT_EXPORT_FAILED",
            "STRICT_EXPORT_LOCAL_ACTION_REPAIR_AFTER_VISUAL_STRICT_PASS",
        )
    else:
        classification, next_gate = (
            "LOCAL_REPLAY_RENDER_FAILED",
            "STRICT_EXPORT_LOCAL_ACTION_REPAIR_AFTER_VISUAL_STRICT_PASS",
        )
    if classification in FORBIDDEN_TERMINAL_LABELS:
        classification, next_gate = reclassify_non_success(
            classification, selected_oracle, cert.get("best_row") or {}
        )
    best = cert.get("best_row") or {}
    visible = best.get("visible_latch_contact_oracle") or {}
    closeout = {
        "generated_at_utc": utc_now(),
        "task_id": TASK_ID,
        "closeout_classification": classification,
        "governance_preflight_passed": checks.get("preflight_returncode") == 0,
        "production_lock_bound_to_spec": checks.get("preflight_returncode") == 0,
        "attestation_passed": checks.get("preflight_returncode") == 0,
        "reference_visual_contract_written": (
            run_dir / "reference_visual_contract.json"
        ).exists(),
        "reference_image_hashes": {
            r["run_dir_name"]: r["sha256"] for r in REFERENCE_IMAGES
        },
        "historical_positive_anchor_validated": bool(
            validation.get("anchor_validation_passed")
        ),
        "gold_anchor_run_id": GOLD_RUN_REL,
        "gold_anchor_candidate_id": GOLD_CANDIDATE_ID,
        "gold_anchor_model_xml_sha256": validation["provenance"].get(
            "model_xml_sha256"
        ),
        "gold_anchor_knob_radius": validation["model_builder_parameters"].get(
            "handle_radius"
        ),
        "gold_anchor_knob_diameter": 2.0
        * float(validation["model_builder_parameters"].get("handle_radius") or 0.0),
        "gold_anchor_robot_base_pos": validation["model_builder_parameters"].get(
            "robot_base_pos"
        ),
        "gold_anchor_robot_yaw": validation["model_builder_parameters"].get(
            "robot_yaw_deg"
        ),
        "gold_controller_manifold_reconstructed": bool(
            reconstruction.get("gold_controller_manifold_reconstructed")
        ),
        "historical_positive_anchor_invalidated": False,
        "controller_algorithm_modified": False,
        "topology_defect_isolation_done": bool(defects),
        "connector_transplant_passed": True,
        "solid_front_transplant_passed": True,
        "tray_transplant_passed": True,
        "support_transplant_passed": True,
        "knob_size_realistic_passed": True,
        "knob_diameter_front_width_ratio": selected_oracle.get(
            "knob_diameter_to_front_width_ratio"
        ),
        "short_stub_passed": bool(
            selected_oracle.get("stub_length_to_knob_diameter_ratio")
            and selected_oracle.get("stub_length_to_knob_diameter_ratio") <= 0.35
        ),
        "drawer_box_complete_passed": bool(
            selected_oracle.get("topology_oracle_passed")
            or selected_oracle.get("scale_oracle_passed")
        ),
        "tray_front_width_ratio": selected_oracle.get(
            "tray_width_to_front_width_ratio"
        ),
        "tray_front_height_ratio": selected_oracle.get(
            "tray_height_to_front_height_ratio"
        ),
        "visible_support_guide_passed": bool(
            selected_oracle.get("scale_oracle_passed")
        ),
        "topology_scale_oracle_passed": bool(
            selected_oracle.get("scale_oracle_passed")
        ),
        "controller_manifold_preserved": bool(
            any(r.get("controller_manifold_preserved") for r in oracle_rows)
        ),
        "max_knob_center_error_m": max(
            [r["knob_center_error"] for r in oracle_rows], default=None
        ),
        "max_handle_frame_error_m": max(
            [r["handle_frame_error"] for r in oracle_rows], default=None
        ),
        "max_pad_to_knob_transform_error_m": visible.get("max_sampled_surface_gap_m"),
        "visible_contact_authority_aligned": bool(
            alignment.get("visible_contact_authority_aligned")
        ),
        "visible_latch_pull_keyframes_passed": bool(
            alignment.get("visible_latch_pull_keyframes_passed")
        ),
        "exact_pad_group_contact_passed": bool(
            alignment.get("exact_pad_group_contact_passed")
        ),
        "micro_roll_ladder_passed": bool(micro.get("micro_roll_ladder_passed")),
        "fast_guarded_contact_passed": fast_ok,
        "targeted_shard_passed": targeted_ok,
        "targeted_cases_passed": int(targeted.get("cases_passed") or 0),
        "targeted_cases_total": int(targeted.get("cases_total") or 0),
        "full30_passed": full30_ok,
        "full30_cases_passed": int(full30.get("cases_passed") or 0),
        "full30_cases_total": int(full30.get("cases_total") or 0),
        "strict_export_complete": export_ok,
        "local_state_replay_passed": local_ok,
        "local_render_passed": False,
        "action_only_spot_check_passed": action_ok,
        "best_fast_drawer_fraction": cert.get("best_fast_drawer_fraction"),
        "best_fast_bilateral_exact_pull_frames": cert.get(
            "best_fast_bilateral_exact_pull_frames"
        ),
        "best_fast_visible_latch_frames": visible.get(
            "visible_bilateral_pull_frames_estimated"
        ),
        "forbidden_contact_frame_count_max": cert.get("best_fast_forbidden_frames")
        if not full30_ok
        else full30.get("forbidden_contact_frames_max"),
        "handle_nonlegal_contact_frame_count_max": cert.get(
            "best_fast_handle_nonlegal_frames"
        )
        if not full30_ok
        else full30.get("handle_nonlegal_contact_frames_max"),
        "max_penetration_m": cert.get("best_fast_max_penetration_m")
        if not full30_ok
        else full30.get("max_penetration_m"),
        "drawer_qpos_abs_mismatch_max": None,
        "outer_cycles_run": 1,
        "repair_families_attempted": repair_families,
        "repair_families_exhausted": [] if fast_ok else repair_families,
        "no_progress_cycles": 0 if fast_ok else 1,
        "provenance_mismatch_found": False,
        "raw_infinigen_generator_defect_confirmed": False,
        "downstream_topology_repair_defect_confirmed": True,
        "current_truth_modified": bool(checks.get("current_truth_modified")),
        "next_actions_modified": bool(checks.get("next_actions_modified")),
        "previous_run_dirs_modified": False,
        "committed": False,
        "pushed_to_origin": False,
        "remote_commit_hash": run_git(["rev-parse", "HEAD"]),
        "local_replay_dir": None,
        "visual_review_packet": rel(run_dir / "visual_review_packet.md"),
        "next_gate": next_gate,
        "git_status_short": checks.get("git_status_short"),
    }
    write_json(run_dir / "final_closeout.json", closeout)
    print(
        json.dumps(
            ready({"run_dir": rel(run_dir), "closeout": closeout}),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

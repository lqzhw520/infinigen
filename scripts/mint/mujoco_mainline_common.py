#!/usr/bin/env python3
"""Shared helpers for the canonical true Infinigen MuJoCo mainline."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from mint_common import (
    ARTIFACT_DIR,
    CAMPAIGN_DIR,
    DATASET_DIR,
    DATASET_REPO_ID,
    DEFAULT_HELD_OUT_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    OUTPUT_DIR,
    now_iso,
    write_json_atomic,
    write_text_atomic,
)

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
SCRIPTS_MINT = PROJECT_ROOT / "scripts" / "mint"
SOVEREIGN_DIR = CAMPAIGN_DIR / "sovereign"
NIGHT_DIR = SOVEREIGN_DIR / "night"
EVIDENCE_DIR = SOVEREIGN_DIR / "evidence"
EVIDENCE_INDEX_PATH = EVIDENCE_DIR / "index.json"
CURRENT_TRUTH_PATH = SOVEREIGN_DIR / "current_truth.json"
NEXT_ACTIONS_PATH = SOVEREIGN_DIR / "next_actions.json"

RUNNER_ID = "mujoco_infinigen_mainline_night"
RUNNER_PATH = SCRIPTS_MINT / "run_mujoco_infinigen_mainline_night.py"
NIGHT_STATUS_PATH = NIGHT_DIR / f"{RUNNER_ID}_last.json"
NIGHT_ARTIFACT_PATH = ARTIFACT_DIR / f"{RUNNER_ID}.json"
NIGHT_REPORT_PATH = OUTPUT_DIR / f"{RUNNER_ID}_report.md"

CONTROL_BASELINE_REF = "E036/p1c11_official_libero_goal_drawer_baseline_rollback_4eab579"
MUJOCO_ENV_REF = "scripts/mint/drawer_robot_env_mujoco.py"
HISTORICAL_LINES_IGNORED = [
    "DrawerRobotEnv/PyBullet",
    "DrawerRobotEnvLeRobot/PyBullet compatibility layer",
    "run_g9_sim_eval.py",
    "evaluate_mint_drawer_campaign.py",
]

STAGE_ORDER = [
    "m0_authority_preflight",
    "m1_proxy_asset_smoke",
    "m2_true_robot_env_gate",
    "m3_anygrasp_gate",
    "m4_robot_rollout_gate",
    "m5_dataset_pack_gate",
    "m6_mint_train_gate",
    "m7_mint_eval_gate",
]

STAGE_EVIDENCE_IDS = {
    "m0_authority_preflight": "E037",
    "m1_proxy_asset_smoke": "E038",
    "m2_true_robot_env_gate": "E039",
    "m3_anygrasp_gate": "E040",
    "m4_robot_rollout_gate": "E041",
    "m5_dataset_pack_gate": "E042",
    "m6_mint_train_gate": "E043",
    "m7_mint_eval_gate": "E044",
}

STAGE_TARGETS = {
    "m0_authority_preflight": "Lock canonical mainline authority before any detached MuJoCo run",
    "m1_proxy_asset_smoke": "Verify Infinigen drawer URDF assets load in MuJoCo proxy mode",
    "m2_true_robot_env_gate": "Verify the true robot-in-loop MuJoCo env resets, steps, renders, and cleans up",
    "m3_anygrasp_gate": "Verify AnyGrasp can consume true MuJoCo payloads and produce usable candidates",
    "m4_robot_rollout_gate": "Generate canonical MuJoCo oracle/AnyGrasp robot rollouts and choose the learning source",
    "m5_dataset_pack_gate": "Package MuJoCo robot rollouts into the canonical LeRobot dataset",
    "m6_mint_train_gate": "Fine-tune upstream MINT on the MuJoCo Infinigen dataset",
    "m7_mint_eval_gate": "Compare pretrained vs fine-tuned MINT on held-out MuJoCo Infinigen drawer seeds",
}

STAGE_FIX_IDS = {
    stage: f"fix_{stage}" for stage in STAGE_ORDER[2:]
}


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return deepcopy(default)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return deepcopy(default)


def ensure_layout() -> None:
    for path in [ARTIFACT_DIR, OUTPUT_DIR, NIGHT_DIR, EVIDENCE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def make_night_status(
    *,
    current_stage: str,
    status: str,
    canonical_next_action_at_launch: dict[str, Any],
    failed_stage: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "runner_id": RUNNER_ID,
        "runner_path": str(RUNNER_PATH),
        "backend": "mujoco",
        "robot_in_loop": True,
        "canonical": True,
        "control_baseline_ref": CONTROL_BASELINE_REF,
        "mujoco_env_ref": MUJOCO_ENV_REF,
        "canonical_next_action_at_launch": canonical_next_action_at_launch,
        "historical_lines_ignored": HISTORICAL_LINES_IGNORED,
        "current_stage": current_stage,
        "status": status,
        "failed_stage": failed_stage,
        "updated_at": now_iso(),
    }
    if details:
        payload["details"] = details
    return payload


def append_report_line(lines: list[str], text: str) -> None:
    lines.append(f"- {text}")


def write_night_outputs(status: dict[str, Any], report_lines: list[str], summary: dict[str, Any]) -> None:
    ensure_layout()
    write_json_atomic(NIGHT_STATUS_PATH, status)
    write_json_atomic(NIGHT_ARTIFACT_PATH, summary)
    report = "\n".join(
        [
            "# MuJoCo Infinigen Mainline Night Report",
            "",
            f"Updated: {now_iso()}",
            "",
            "## Summary",
            *report_lines,
            "",
        ]
    )
    write_text_atomic(NIGHT_REPORT_PATH, report)


def _upsert_next_action(actions: list[dict[str, Any]], item: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    replaced = False
    item_key = (item.get("type"), item.get("id"))
    for existing in actions:
        key = (existing.get("type"), existing.get("id"))
        if key == item_key:
            out.append(item)
            replaced = True
        else:
            out.append(existing)
    if not replaced:
        out.insert(0, item)
    return out


def _retire_stale_mainline_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for existing in actions:
        updated = deepcopy(existing)
        if updated.get("type") in {"MUJOCO_MAINLINE_STAGE_FIX", "MUJOCO_MAINLINE_RESULT_REVIEW"} and updated.get("status") in {"pending", "running", "blocked"}:
            updated["status"] = "superseded"
            updated["superseded_at"] = now_iso()
        out.append(updated)
    return out


def sync_sovereign_running(stage: str, status: dict[str, Any]) -> None:
    current_truth = load_json(CURRENT_TRUTH_PATH, {})
    next_actions = load_json(NEXT_ACTIONS_PATH, {})
    current = current_truth.get("current", {})
    current["phase"] = "v59_MUJOCO_MAINLINE_PHASE6 — true Infinigen MuJoCo night-runner executing"
    current["phase_gate"] = stage
    next_action = current.get("next_action", {})
    next_action["type"] = "MUJOCO_INFINIGEN_MAINLINE_ALIGNMENT"
    next_action["id"] = "mujoco_infinigen_mainline_alignment"
    next_action["status"] = "running"
    next_action["target"] = STAGE_TARGETS.get(stage, next_action.get("target"))
    next_action["last_stage"] = stage
    next_action["runner_id"] = RUNNER_ID
    next_action["updated_at"] = now_iso()
    current["next_action"] = next_action
    current_truth["current"] = current
    write_json_atomic(CURRENT_TRUTH_PATH, current_truth)

    actions = _retire_stale_mainline_actions(next_actions.get("actions", []))
    next_actions["actions"] = _upsert_next_action(
        actions,
        {
            "type": "MUJOCO_INFINIGEN_MAINLINE_ALIGNMENT",
            "id": "mujoco_infinigen_mainline_alignment",
            "priority": "P0",
            "target": STAGE_TARGETS.get(stage, "Execute the canonical true MuJoCo mainline"),
            "status": "running",
            "current_stage": stage,
            "runner_id": RUNNER_ID,
            "control_baseline_ref": CONTROL_BASELINE_REF,
            "updated_at": now_iso(),
        },
    )
    next_actions["generated_at"] = now_iso()
    write_json_atomic(NEXT_ACTIONS_PATH, next_actions)
    write_json_atomic(NIGHT_STATUS_PATH, status)


def _make_stage_fix_action(stage: str, reason: str) -> dict[str, Any]:
    return {
        "type": "MUJOCO_MAINLINE_STAGE_FIX",
        "id": STAGE_FIX_IDS.get(stage, f"fix_{stage}"),
        "priority": "P0",
        "target": STAGE_TARGETS.get(stage, stage),
        "status": "pending",
        "description": reason,
        "control_baseline_ref": CONTROL_BASELINE_REF,
        "updated_at": now_iso(),
    }


def sync_sovereign_failure(stage: str, reason: str, status: dict[str, Any]) -> None:
    current_truth = load_json(CURRENT_TRUTH_PATH, {})
    next_actions = load_json(NEXT_ACTIONS_PATH, {})
    current = current_truth.get("current", {})
    current["phase"] = "v59_MUJOCO_MAINLINE_PHASE6 — true Infinigen MuJoCo gate failed"
    current["phase_gate"] = stage
    current["next_action"] = _make_stage_fix_action(stage, reason)
    current_truth["current"] = current
    write_json_atomic(CURRENT_TRUTH_PATH, current_truth)

    actions = _retire_stale_mainline_actions(next_actions.get("actions", []))
    actions = _upsert_next_action(actions, current["next_action"])
    actions = _upsert_next_action(
        actions,
        {
            "type": "MUJOCO_INFINIGEN_MAINLINE_ALIGNMENT",
            "id": "mujoco_infinigen_mainline_alignment",
            "priority": "P0",
            "target": "Resume canonical true Infinigen URDF + AnyGrasp + MuJoCo alignment after fixing the failed gate",
            "status": "blocked",
            "blocked_by": stage,
            "updated_at": now_iso(),
        },
    )
    next_actions["actions"] = actions
    next_actions["generated_at"] = now_iso()
    write_json_atomic(NEXT_ACTIONS_PATH, next_actions)
    write_json_atomic(NIGHT_STATUS_PATH, status)


def sync_sovereign_success(final_stage: str, status: dict[str, Any], summary_note: str) -> None:
    current_truth = load_json(CURRENT_TRUTH_PATH, {})
    next_actions = load_json(NEXT_ACTIONS_PATH, {})
    current = current_truth.get("current", {})
    current["phase"] = "v59_MUJOCO_MAINLINE_PHASE6 — true Infinigen MuJoCo night-runner completed"
    current["phase_gate"] = final_stage
    current["next_action"] = {
        "type": "MUJOCO_MAINLINE_RESULT_REVIEW",
        "id": "review_mujoco_infinigen_mainline_results",
        "priority": "P0",
        "target": "Review true MuJoCo Infinigen night-runner results and decide the next scientific move",
        "status": "pending",
        "description": summary_note,
        "control_baseline_ref": CONTROL_BASELINE_REF,
        "updated_at": now_iso(),
    }
    current_truth["current"] = current
    write_json_atomic(CURRENT_TRUTH_PATH, current_truth)

    actions = _retire_stale_mainline_actions(next_actions.get("actions", []))
    actions = _upsert_next_action(actions, current["next_action"])
    actions = _upsert_next_action(
        actions,
        {
            "type": "MUJOCO_INFINIGEN_MAINLINE_ALIGNMENT",
            "id": "mujoco_infinigen_mainline_alignment",
            "priority": "P0",
            "target": "Canonicalize true Infinigen URDF + AnyGrasp + MuJoCo alignment against the restored upstream MINT control",
            "status": "completed",
            "result": summary_note,
            "updated_at": now_iso(),
        },
    )
    next_actions["actions"] = actions
    next_actions["generated_at"] = now_iso()
    write_json_atomic(NEXT_ACTIONS_PATH, next_actions)
    write_json_atomic(NIGHT_STATUS_PATH, status)


def register_evidence(stage: str, artifact_path: Path, summary: str, *, verified: bool = True) -> str:
    ensure_layout()
    evidence_index = load_json(EVIDENCE_INDEX_PATH, {"version": 1, "last_updated": now_iso(), "entries": []})
    entries = evidence_index.get("entries", [])
    evidence_id = STAGE_EVIDENCE_IDS[stage]
    rel_path = str(artifact_path.relative_to(PROJECT_ROOT))
    entry = {
        "evidence_id": evidence_id,
        "experiment_id": stage,
        "type": "mujoco_mainline_stage",
        "timestamp": now_iso(),
        "path": rel_path,
        "verified": verified,
        "summary": summary,
    }
    replaced = False
    new_entries = []
    for existing in entries:
        if existing.get("evidence_id") == evidence_id:
            new_entries.append(entry)
            replaced = True
        else:
            new_entries.append(existing)
    if not replaced:
        new_entries.append(entry)
    evidence_index["entries"] = new_entries
    evidence_index["last_updated"] = now_iso()
    write_json_atomic(EVIDENCE_INDEX_PATH, evidence_index)
    return evidence_id


def default_train_seeds() -> list[int]:
    return list(DEFAULT_TRAIN_SEEDS)


def default_heldout_seeds() -> list[int]:
    return list(DEFAULT_HELD_OUT_SEEDS)

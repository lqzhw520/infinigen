#!/usr/bin/env python3
"""Shared campaign helpers for the MINT AnyGrasp robot-trajectory integration loop."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from contract_preflight import directory_preflight
from strict_success import STRICT_SUCCESS_VERSION

PROJECT_ROOT = Path("/mnt/afs2/zhuhaowu/infinigen")
CAMPAIGN_DIR = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1"
RUNTIME_DIR = CAMPAIGN_DIR / "runtime"
ARTIFACT_DIR = CAMPAIGN_DIR / "artifacts"
EVAL_DIR = CAMPAIGN_DIR / "evaluation"
OUTPUT_DIR = CAMPAIGN_DIR / "outputs"
DATASET_DIR = CAMPAIGN_DIR / "dataset"
TINY_RETRAIN_PLAN_PATH = ARTIFACT_DIR / "active_tiny_retrain_plan.json"
TINY_RETRAIN_DATASET_BUILD_PATH = ARTIFACT_DIR / "g8_canonical_dataset_build.json"
TINY_RETRAIN_SUMMARY_PATH = EVAL_DIR / "tiny_retrain_confirmation_summary.json"
TINY_RETRAIN_OUTPUT_DIR = OUTPUT_DIR / "tiny_retrain"
TINY_RETRAIN_EVAL_DIR = EVAL_DIR / "tiny_retrain"
ARCHIVE_DIR = CAMPAIGN_DIR / "archived_proxy"
CONDA_SETUP = "source /root/anaconda3/etc/profile.d/conda.sh"

QUEUE_STEPS = [
    "u1_asset_geometry_audit",
    "u2_joint_semantics_audit",
    "u3_handle_region_audit",
    "u4_export_consistency_audit",
    "u5_seed_outlier_audit",
    "a1_asset_scene_audit",
    "a2_frame_transform_audit",
    "b1_oracle_scripted_baseline",
    "b2_anygrasp_scripted_baseline",
    "c1_teacher_native_rollout_rebuild",
    "c2_action_contract_repair",
    "c3_single_rollout_replay_gate",
    "d1_single_rollout_overfit",
    "d2_single_seed_overfit",
    "d3_train_seed_probe",
    "e1_heldout_eval",
    "write_claim_memo",
]

QUEUE_ENVS = {
    "u1_asset_geometry_audit": "mint",
    "u2_joint_semantics_audit": "mint",
    "u3_handle_region_audit": "mint",
    "u4_export_consistency_audit": "mint",
    "u5_seed_outlier_audit": "mint",
    "a1_asset_scene_audit": "mint",
    "a2_frame_transform_audit": "mint",
    "b1_oracle_scripted_baseline": "mint",
    "b2_anygrasp_scripted_baseline": "mint",
    "c1_teacher_native_rollout_rebuild": "mint",
    "c2_action_contract_repair": "mint",
    "c3_single_rollout_replay_gate": "mint",
    "d1_single_rollout_overfit": "mint",
    "d2_single_seed_overfit": "mint",
    "d3_train_seed_probe": "mint",
    "e1_heldout_eval": "mint",
    "write_claim_memo": "mint",
    # LIBERO alignment gates
    "p4_physics_legal_gate": "mint",
    "p1a_state_vector_fix": "mint",
    "p1b_mujoco_env": "mint",
}

DATASET_REPO_ID = "infinigen_drawer_robot_v1"
DEFAULT_TRAIN_SEEDS = list(range(1, 11))
DEFAULT_HELD_OUT_SEEDS = list(range(11, 16))
TRANSLATION_SCALE_M = 0.03
ROTATION_SCALE_RAD = 0.25
GRIPPER_CLOSE_VALUE = -1.0
GRIPPER_OPEN_VALUE = 1.0
ACTIVE_ACTION_CONTRACT_PATH = ARTIFACT_DIR / "active_action_contract.json"
BRANCH_PROGRESS_PATH = ARTIFACT_DIR / "c_action_contract_branch_progress.json"
C1_BRANCH_DIR = ARTIFACT_DIR / "c1_branch_rollouts"
C2_REPLAY_DIR = ARTIFACT_DIR / "c2_replay_valid_rollouts"
C3_SINGLE_DIR = ARTIFACT_DIR / "c3_single_rollout"
CONTROLLER_LEASE_PATH = RUNTIME_DIR / "controller_lease.json"
TEACHER_LINEAGE_PIN_PATH = ARTIFACT_DIR / "teacher_lineage_pin.json"
STRICT_TEACHER_AUDIT_PATH = ARTIFACT_DIR / "strict_teacher_dataset_audit.json"
ACTIVE_TEACHER_SOURCE_PATH = ARTIFACT_DIR / "active_teacher_source.json"
STRONG_ROLLOUT_AUDIT_PATH = ARTIFACT_DIR / "strong_rollout_audit.json"
STRICT_TEACHER_DIR = ARTIFACT_DIR / "strict_teacher_rollouts"
D1_CANDIDATE_ARTIFACT = ARTIFACT_DIR / "d1_candidate_search.json"
D1_PROGRESS_PATH = ARTIFACT_DIR / "d1_candidate_progress.json"
D2_VARIANT_PROGRESS_PATH = ARTIFACT_DIR / "d2_variant_progress.json"
MAINLINE_FAILURE_MATRIX_PATH = ARTIFACT_DIR / "mainline_failure_matrix.json"
ENV_CONTRACT_AUDIT_PATH = ARTIFACT_DIR / "env_contract_audit.json"
DEFAULT_PYTHON = "python3"
DEFAULT_CONDA_RUN = "/root/anaconda3/bin/conda run --no-capture-output"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    for path in [
        CAMPAIGN_DIR,
        RUNTIME_DIR,
        ARTIFACT_DIR,
        EVAL_DIR,
        OUTPUT_DIR,
        DATASET_DIR,
        ARCHIVE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def load_json(path: Path, default: Any):
    if not path.exists():
        return deepcopy(default)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return deepcopy(default)


def merge_missing(target: Any, defaults: Any) -> Any:
    if isinstance(target, dict) and isinstance(defaults, dict):
        merged = deepcopy(target)
        for key, value in defaults.items():
            if key not in merged:
                merged[key] = deepcopy(value)
            else:
                merged[key] = merge_missing(merged[key], value)
        return merged
    return target


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        tmp = Path(handle.name)
    os.replace(tmp, path)


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent) as handle:
        handle.write(content)
        tmp = Path(handle.name)
    os.replace(tmp, path)


def controller_identity(
    owner: str = "mint_auto_review_loop", pid: int | None = None
) -> dict[str, Any]:
    controller_pid = int(pid or os.getpid())
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    controller_id = f"{owner}:{socket.gethostname()}:{controller_pid}"
    run_id = f"{owner}:{stamp}:{controller_pid}"
    return {
        "controller_id": controller_id,
        "run_id": run_id,
        "owner": owner,
        "pid": controller_pid,
        "host": socket.gethostname(),
    }


def read_controller_lease() -> dict[str, Any]:
    return load_json(CONTROLLER_LEASE_PATH, {})


def acquire_controller_lease(
    controller_id: str, run_id: str, *, owner: str, pid: int | None = None
) -> tuple[bool, dict[str, Any]]:
    ensure_dirs()
    current_pid = int(pid or os.getpid())
    existing = read_controller_lease()
    existing_pid = int(existing.get("pid") or 0)
    if (
        existing
        and existing.get("controller_id") != controller_id
        and pid_alive(existing_pid)
    ):
        return False, existing
    payload = {
        "controller_id": controller_id,
        "run_id": run_id,
        "owner": owner,
        "pid": current_pid,
        "host": socket.gethostname(),
        "acquired_at": existing.get("acquired_at")
        if existing.get("controller_id") == controller_id
        else now_iso(),
        "heartbeat_at": now_iso(),
    }
    write_json_atomic(CONTROLLER_LEASE_PATH, payload)
    return True, payload


def refresh_controller_lease(
    controller_id: str, run_id: str, *, pid: int | None = None
) -> dict[str, Any]:
    ensure_dirs()
    current_pid = int(pid or os.getpid())
    lease = read_controller_lease()
    if lease.get("controller_id") == controller_id and lease.get("run_id") == run_id:
        lease["heartbeat_at"] = now_iso()
        lease["pid"] = current_pid
        write_json_atomic(CONTROLLER_LEASE_PATH, lease)
        return lease
    payload = {
        "controller_id": controller_id,
        "run_id": run_id,
        "owner": lease.get("owner") or "mint_auto_review_loop",
        "pid": current_pid,
        "host": socket.gethostname(),
        "acquired_at": now_iso(),
        "heartbeat_at": now_iso(),
    }
    write_json_atomic(CONTROLLER_LEASE_PATH, payload)
    return payload


def release_controller_lease(controller_id: str, run_id: str) -> None:
    lease = read_controller_lease()
    if (
        lease.get("controller_id") == controller_id
        and lease.get("run_id") == run_id
        and CONTROLLER_LEASE_PATH.exists()
    ):
        CONTROLLER_LEASE_PATH.unlink()


def controller_context_from_state(
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = state or load_state()
    controller = payload.get("controller") or {}
    runtime = payload.get("active_runtime") or {}
    return {
        "controller_id": controller.get("controller_id")
        or runtime.get("controller_id"),
        "run_id": controller.get("run_id") or runtime.get("run_id"),
        "owner": controller.get("owner"),
    }


def append_invalidated_result(
    state: dict[str, Any],
    *,
    step_id: str,
    reason: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = {
        "step_id": step_id,
        "reason": reason,
        "details": details or {},
        "recorded_at": now_iso(),
    }
    invalidated = state.setdefault("invalidated_results", [])
    invalidated.append(entry)
    state["last_invalidated_result"] = entry
    return entry


def normalize_manifest(raw: dict[str, Any]) -> dict[str, Any]:
    if not raw:
        raw = {}
    if isinstance(raw.get("campaign"), dict):
        normalized = deepcopy(raw)
        queue = list(normalized["campaign"].get("queue") or [])
        merged_queue = []
        for step_id in QUEUE_STEPS:
            if step_id not in merged_queue:
                merged_queue.append(step_id)
        for step_id in queue:
            if step_id not in merged_queue:
                merged_queue.append(step_id)
        normalized["campaign"]["queue"] = merged_queue
        env_routing = dict(normalized["campaign"].get("env_routing") or {})
        for step_id, env_name in QUEUE_ENVS.items():
            env_routing.setdefault(step_id, env_name)
        normalized["campaign"]["env_routing"] = env_routing
        return normalized
    return {
        "campaign": {
            "id": raw.get("campaign", "mint_drawer_v1"),
            "phase": raw.get("phase", "mint_robot_trajectory_root_cause_repair"),
            "phase_gate": raw.get("phase_gate", "await_train_reproducibility"),
            "main_claim": raw.get(
                "main_claim",
                "Infinigen-generated AnyGrasp-conditioned robot-arm drawer trajectories improve MINT success on held-out drawer variants in simulation relative to pretrained MINT.",
            ),
            "scope": "simulation-only",
            "queue": raw.get("queue", QUEUE_STEPS),
            "env_routing": raw.get("env_routing", QUEUE_ENVS),
            "object_split": raw.get(
                "object_split",
                {"train": DEFAULT_TRAIN_SEEDS, "held_out_sim": DEFAULT_HELD_OUT_SEEDS},
            ),
            "baselines": raw.get(
                "baselines", ["random", "pretrained_mint", "finetuned_mint"]
            ),
            "primary_metric": raw.get("primary_metric", "held_out_sim_success_rate"),
            "notes": raw.get(
                "notes",
                {
                    "takeover_mode": "anygrasp_robot_trajectory",
                    "proxy_baseline_archived": True,
                    "tracking_scope": "out_of_scope",
                    "root_cause_repair": True,
                    "claim_push_mode": True,
                    "root_cause_ladder": [
                        "L1_oracle",
                        "L2_anygrasp",
                        "L3_train",
                        "L4_heldout",
                    ],
                },
            ),
        }
    }


def load_manifest() -> dict[str, Any]:
    path = CAMPAIGN_DIR / "manifest.yaml"
    if not path.exists():
        return normalize_manifest({})
    raw = yaml.safe_load(path.read_text()) or {}
    return normalize_manifest(raw)


def write_manifest(manifest: dict[str, Any]) -> None:
    write_text_atomic(
        CAMPAIGN_DIR / "manifest.yaml", yaml.safe_dump(manifest, sort_keys=False)
    )


def default_state(manifest: dict[str, Any]) -> dict[str, Any]:
    queue = [
        {"id": step_id, "status": "pending", "attempts": 0}
        for step_id in manifest["campaign"]["queue"]
    ]
    return {
        "phase": manifest["campaign"]["phase"],
        "phase_gate": manifest["campaign"]["phase_gate"],
        "queue": queue,
        "controller": {
            "controller_id": None,
            "run_id": None,
            "owner": None,
            "lease_path": str(CONTROLLER_LEASE_PATH),
            "status": "idle",
            "acquired_at": None,
            "heartbeat_at": None,
        },
        "active_job": None,
        "active_runtime": None,
        "history": [],
        "last_review": None,
        "retries": {},
        "verdict": None,
        "claim_level": None,
        "revision": {
            "active": "robot_revision_v3_claim_push",
            "archived": [
                "proxy_revision",
                "robot_revision_v1_failed",
                "robot_revision_v2_root_cause_repair",
            ],
            "proxy_revision_archived": False,
            "robot_revision_v1_failed_archived": False,
            "robot_trajectory_revision_started": False,
        },
        "strict_replay_lane": {
            "status": "pending",
            "best_branch": None,
            "best_metrics": {},
            "last_update": None,
        },
        "learnability_lane": {
            "status": "pending",
            "source_contract_mode": None,
            "source_branch": None,
            "current_step": None,
            "last_positive_step": None,
            "last_update": None,
        },
        "upstream_audit_lane": {
            "status": "pending",
            "root_cause": None,
            "blocking_step": None,
            "affected_seeds": [],
            "fix_required": False,
            "patch_allowed": False,
            "patch_gate_reason": None,
            "last_update": None,
        },
        "coverage_expansion_attempts": 0,
        "last_invalidated_result": None,
        "updated_at": now_iso(),
    }


def default_review(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": manifest["campaign"]["phase"],
        "phase_gate": manifest["campaign"]["phase_gate"],
        "workflow_score": 0,
        "evidence_score": 0,
        "decision": "run_experiments",
        "blocker_type": "none",
        "verdict": "awaiting_execution",
        "claim_assessment": "No evaluation evidence yet.",
        "strengths": [],
        "weaknesses": [],
        "next_actions": ["Run the next pending queue step."],
        "generated_at": now_iso(),
    }


def default_next_actions() -> dict[str, Any]:
    return {
        "decision": "run_experiments",
        "actions": [
            {
                "type": "run_experiments",
                "target": "next_pending_step",
                "reason": "Campaign initialized",
            }
        ],
        "generated_at": now_iso(),
    }


def default_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "campaign": manifest["campaign"]["id"],
        "phase": manifest["campaign"]["phase"],
        "status": "initialized",
        "artifacts": {},
        "strongest_true_claim": None,
        "success_semantics_version": STRICT_SUCCESS_VERSION,
        "dataset_fingerprint": None,
        "invalidated_results": [],
        "proxy_baseline_archived": False,
        "strict_replay_lane": {},
        "learnability_lane": {},
        "upstream_audit_lane": {},
        "generated_at": now_iso(),
    }


def load_state() -> dict[str, Any]:
    manifest = load_manifest()
    state = load_json(CAMPAIGN_DIR / "state.json", default_state(manifest))
    state = merge_missing(state, default_state(manifest))
    existing = {item.get("id"): deepcopy(item) for item in state.get("queue", [])}
    state["queue"] = [
        existing.get(step_id, {"id": step_id, "status": "pending", "attempts": 0})
        for step_id in manifest["campaign"]["queue"]
    ]
    return state


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    write_json_atomic(CAMPAIGN_DIR / "state.json", state)


def load_review() -> dict[str, Any]:
    manifest = load_manifest()
    review = load_json(CAMPAIGN_DIR / "review.json", default_review(manifest))
    return merge_missing(review, default_review(manifest))


def save_review(review: dict[str, Any]) -> None:
    review["generated_at"] = now_iso()
    write_json_atomic(CAMPAIGN_DIR / "review.json", review)


def load_summary() -> dict[str, Any]:
    manifest = load_manifest()
    summary = load_json(CAMPAIGN_DIR / "summary.json", default_summary(manifest))
    return merge_missing(summary, default_summary(manifest))


def save_summary(summary: dict[str, Any]) -> None:
    summary["generated_at"] = now_iso()
    write_json_atomic(CAMPAIGN_DIR / "summary.json", summary)


def save_next_actions(payload: dict[str, Any]) -> None:
    payload["generated_at"] = now_iso()
    write_json_atomic(CAMPAIGN_DIR / "next_actions.json", payload)


def queue_entry(state: dict[str, Any], step_id: str) -> dict[str, Any]:
    for item in state.get("queue", []):
        if item.get("id") == step_id:
            return item
    raise KeyError(step_id)


def set_queue_status(
    state: dict[str, Any], step_id: str, status: str, *, error: str | None = None
) -> None:
    item = queue_entry(state, step_id)
    item["status"] = status
    item["updated_at"] = now_iso()
    if error is not None:
        item["last_error"] = error
    elif "last_error" in item:
        item.pop("last_error", None)


def increment_retry(state: dict[str, Any], step_id: str) -> int:
    retries = state.setdefault("retries", {})
    retries[step_id] = int(retries.get(step_id, 0)) + 1
    queue_entry(state, step_id)["attempts"] = retries[step_id]
    return retries[step_id]


def reset_step(state: dict[str, Any], step_id: str) -> None:
    item = queue_entry(state, step_id)
    item["status"] = "pending"
    item["updated_at"] = now_iso()
    item.pop("last_error", None)


def append_history(state: dict[str, Any], event: str, details: str) -> None:
    state.setdefault("history", []).append(
        {"at": now_iso(), "event": event, "details": details}
    )
    state["history"] = state["history"][-100:]


def active_or_next_step(state: dict[str, Any]) -> str | None:
    for item in state.get("queue", []):
        if item.get("status") == "running":
            return item.get("id")
    for item in state.get("queue", []):
        if item.get("status") in {"pending", "failed"}:
            return item.get("id")
    return None


def step_lane(step_id: str | None, state: dict[str, Any] | None = None) -> str | None:
    if step_id is None:
        return None
    if step_id.startswith("u"):
        return "upstream_audit"
    if step_id in {
        "a1_asset_scene_audit",
        "a2_frame_transform_audit",
        "b1_oracle_scripted_baseline",
        "b2_anygrasp_scripted_baseline",
        "c1_teacher_native_rollout_rebuild",
        "p4_physics_legal_gate",
        "p1a_state_vector_fix",
        "p1b_mujoco_env",
    }:
        return "foundation"
    if step_id == "c2_action_contract_repair":
        contract = load_json(ACTIVE_ACTION_CONTRACT_PATH, {})
        mode = str(contract.get("contract_mode") or "")
        if mode == "teacher_success_fallback":
            return "dual"
        if mode == "strict_replay":
            return "strict_replay"
        return "dual"
    if step_id in {
        "c3_single_rollout_replay_gate",
        "d1_single_rollout_overfit",
        "d2_single_seed_overfit",
        "d3_train_seed_probe",
        "e1_heldout_eval",
        "write_claim_memo",
    }:
        return "learnability"
    return None


def sync_lane_state_from_artifacts(state: dict[str, Any]) -> None:
    u1 = validate_json_file(ARTIFACT_DIR / "u1_asset_geometry_audit.json") or {}
    u2 = validate_json_file(ARTIFACT_DIR / "u2_joint_semantics_audit.json") or {}
    u3 = validate_json_file(ARTIFACT_DIR / "u3_handle_region_audit.json") or {}
    u4 = validate_json_file(ARTIFACT_DIR / "u4_export_consistency_audit.json") or {}
    u5 = validate_json_file(ARTIFACT_DIR / "u5_seed_outlier_audit.json") or {}
    c2 = validate_json_file(ARTIFACT_DIR / "c2_action_contract_repair.json") or {}
    c3 = validate_json_file(ARTIFACT_DIR / "c3_single_rollout_replay_gate.json") or {}
    d1 = (
        validate_json_file(D1_CANDIDATE_ARTIFACT)
        or validate_json_file(ARTIFACT_DIR / "d1_single_rollout_overfit.json")
        or {}
    )
    d1_strict = (
        validate_json_file(ARTIFACT_DIR / "d1_single_rollout_strict_eval.json") or {}
    )
    d2 = validate_json_file(ARTIFACT_DIR / "d2_single_seed_overfit.json") or {}
    d3 = validate_json_file(ARTIFACT_DIR / "d3_train_seed_probe.json") or {}
    e1 = validate_json_file(EVAL_DIR / "comparison_summary.json") or {}
    strict_teacher = validate_json_file(STRICT_TEACHER_AUDIT_PATH) or {}
    env_contract = validate_json_file(ENV_CONTRACT_AUDIT_PATH) or (
        (strict_teacher.get("env_contract_audit") if strict_teacher else {}) or {}
    )
    d2_feasible_seeds = {
        int(seed) for seed in strict_teacher.get("d2_feasible_seeds", [])
    }

    strict_lane = state.setdefault("strict_replay_lane", {})
    learn_lane = state.setdefault("learnability_lane", {})
    upstream_lane = state.setdefault("upstream_audit_lane", {})
    invalidated_results = state.setdefault("invalidated_results", [])
    try:
        c2_status = queue_entry(state, "c2_action_contract_repair").get("status")
    except KeyError:
        c2_status = "pending"

    strict_best = {
        "selected_branch": c2.get("selected_branch"),
        "strict_replay_passed": c2.get("strict_replay_passed"),
        "successful_seed_count": c2.get("successful_seed_count"),
        "successful_replays": c2.get("successful_replays"),
        "successful_seeds": c2.get("successful_seeds", []),
    }
    strict_lane["best_branch"] = c2.get("strict_best_branch") or c2.get(
        "selected_branch"
    )
    strict_lane["best_metrics"] = strict_best
    if c2.get("strict_replay_passed"):
        strict_lane["status"] = "passed"
    elif c2.get("passed"):
        strict_lane["status"] = "repairing"
    elif c2_status == "failed":
        strict_lane["status"] = "blocked"
    else:
        strict_lane.setdefault("status", "pending")
    strict_lane["last_update"] = now_iso()

    u_payloads = [u1, u2, u3, u4, u5]
    priority = {
        "u3_handle_region_audit": 0,
        "u4_export_consistency_audit": 1,
        "u2_joint_semantics_audit": 2,
        "u1_asset_geometry_audit": 3,
        "u5_seed_outlier_audit": 4,
    }
    suspect = sorted(
        [
            payload
            for payload in u_payloads
            if payload.get("decision") == "upstream_suspect"
        ],
        key=lambda payload: priority.get(str(payload.get("gate")), 999),
    )
    blocked = sorted(
        [
            payload
            for payload in u_payloads
            if payload.get("decision") == "upstream_blocked"
        ],
        key=lambda payload: priority.get(str(payload.get("gate")), 999),
    )
    d1_queue_item = next(
        (
            item
            for item in state.get("queue", [])
            if item.get("id") == "d1_single_rollout_overfit"
        ),
        {},
    )
    d1_is_running = (
        d1_queue_item.get("status") == "running"
        or (state.get("active_runtime") or {}).get("step_id")
        == "d1_single_rollout_overfit"
    )
    d1_candidate_search_completed = bool(
        d1 and not d1.get("passed") and d1.get("candidate_order") and not d1_is_running
    )
    if blocked:
        first = blocked[0]
        upstream_lane["status"] = "blocked"
        upstream_lane["root_cause"] = first.get("conclusion_text")
        upstream_lane["blocking_step"] = first.get("gate")
        upstream_lane["affected_seeds"] = (
            first.get("affected_seeds")
            or first.get("bad_axis_seeds")
            or first.get("bad_limit_seeds")
            or first.get("bad_ratio_seeds")
            or []
        )
        upstream_lane["fix_required"] = True
        upstream_lane["patch_allowed"] = True
        upstream_lane["patch_gate_reason"] = "upstream_blocked"
    elif suspect:
        first = suspect[0]
        upstream_lane["status"] = "suspect"
        upstream_lane["root_cause"] = first.get("conclusion_text")
        upstream_lane["blocking_step"] = first.get("gate")
        upstream_lane["affected_seeds"] = (
            first.get("affected_seeds") or first.get("outlier_seed_ids") or []
        )
        upstream_lane["fix_required"] = bool(d1_candidate_search_completed)
        upstream_lane["patch_allowed"] = bool(d1_candidate_search_completed)
        upstream_lane["patch_gate_reason"] = (
            "d1_top3_failed_under_strict_success"
            if d1_candidate_search_completed
            else "await_d1_candidate_search"
        )
    elif any(payload.get("decision") == "upstream_clean" for payload in u_payloads):
        upstream_lane["status"] = "clean"
        upstream_lane["root_cause"] = None
        upstream_lane["blocking_step"] = None
        upstream_lane["affected_seeds"] = []
        upstream_lane["fix_required"] = False
        upstream_lane["patch_allowed"] = False
        upstream_lane["patch_gate_reason"] = "upstream_clean"
    upstream_lane["last_update"] = now_iso()

    learn_lane["source_contract_mode"] = c2.get("contract_mode")
    learn_lane["source_branch"] = c2.get("selected_branch")
    learn_lane["strict_teacher_rollout_count"] = int(
        strict_teacher.get("accepted_rollout_count", 0)
    )
    learn_lane["strict_teacher_seed_count"] = int(
        strict_teacher.get("accepted_seed_count", 0)
    )
    learn_lane["d1_candidate_order"] = strict_teacher.get("d1_candidate_order", [])
    learn_lane["promoted_candidate"] = d1.get("promoted_candidate")
    learn_lane["env_contract_passed"] = bool(env_contract.get("passed"))
    learn_lane["env_contract_failed_seeds"] = env_contract.get("failed_seeds", [])
    strict_eval = (d1_strict.get("evaluation") or {}).get("summary", {})
    strict_eval_ft = strict_eval.get("finetuned_mint", {})
    strict_eval_pt = strict_eval.get("pretrained_mint", {})
    strict_invalidated = bool(
        d1.get("passed")
        and d1_strict
        and float(strict_eval_ft.get("success_rate", 0.0))
        <= float(strict_eval_pt.get("success_rate", 0.0))
    )
    if (
        strict_invalidated
        and "d1_invalidated_by_weak_success_semantics" not in invalidated_results
    ):
        invalidated_results.append("d1_invalidated_by_weak_success_semantics")

    if env_contract and not env_contract.get("passed"):
        learn_lane["status"] = "env_contract_blocked"
        learn_lane["last_positive_step"] = None
    elif d3.get("passed") and e1.get("verdict") in {
        "claim_supported",
        "scientific_not_supported",
    }:
        learn_lane["status"] = "heldout_complete"
        learn_lane["last_positive_step"] = "e1_heldout_eval"
    elif d3.get("passed"):
        learn_lane["status"] = "train_trend_passed"
        learn_lane["last_positive_step"] = "d3_train_seed_probe"
    elif d2.get("passed"):
        learn_lane["status"] = "single_seed_passed"
        learn_lane["last_positive_step"] = "d2_single_seed_overfit"
    elif (
        d1.get("passed")
        and int((d1.get("promoted_candidate") or {}).get("selected_seed", -1))
        not in d2_feasible_seeds
    ):
        learn_lane["status"] = "diagnostic_winner_only"
        learn_lane["last_positive_step"] = "d1_single_rollout_overfit"
    elif strict_invalidated:
        learn_lane["status"] = "single_rollout_invalidated_under_strict_success"
        learn_lane["last_positive_step"] = None
    elif d1.get("passed"):
        learn_lane["status"] = "single_rollout_passed"
        learn_lane["last_positive_step"] = "d1_single_rollout_overfit"
    elif c3.get("passed"):
        learn_lane["status"] = "ready_for_overfit"
        learn_lane["last_positive_step"] = None
    elif c2.get("passed"):
        learn_lane["status"] = "ready_from_contract"
        learn_lane["last_positive_step"] = None
    elif c2_status == "failed":
        learn_lane["status"] = "blocked"
        learn_lane["last_positive_step"] = None
    else:
        learn_lane["status"] = "pending"
        learn_lane["last_positive_step"] = None

    learn_lane["current_step"] = active_or_next_step(state)
    learn_lane["last_update"] = now_iso()


def sync_completed_steps_from_artifacts(state: dict[str, Any]) -> bool:
    changed = False
    for item in state.get("queue", []):
        step_id = item.get("id")
        if item.get("status") == "completed":
            continue
        ok, _message, _payload = validate_step_artifacts(step_id)
        if ok:
            item["status"] = "completed"
            item["updated_at"] = now_iso()
            item.pop("last_error", None)
            changed = True
            continue
        break
    return changed


def all_completed(state: dict[str, Any]) -> bool:
    return all(item.get("status") == "completed" for item in state.get("queue", []))


def runtime_meta_path(step_id: str) -> Path:
    return RUNTIME_DIR / step_id / "run_meta.json"


def write_runtime_meta(step_id: str, payload: dict[str, Any]) -> None:
    write_json_atomic(runtime_meta_path(step_id), payload)


def expected_artifacts(step_id: str) -> list[Path]:
    if step_id in {
        "a1_asset_scene_audit",
        "a2_frame_transform_audit",
        "b1_oracle_scripted_baseline",
        "b2_anygrasp_scripted_baseline",
        "c1_teacher_native_rollout_rebuild",
        "c2_action_contract_repair",
        "c3_single_rollout_replay_gate",
        "d1_single_rollout_overfit",
        "d2_single_seed_overfit",
        "d3_train_seed_probe",
        # LIBERO alignment gates
        "p4_physics_legal_gate",
        "p1a_state_vector_fix",
        "p1b_mujoco_env",
    }:
        return [ARTIFACT_DIR / f"{step_id}.json"]
    if step_id == "e1_heldout_eval":
        return [
            EVAL_DIR / "comparison_summary.json",
            EVAL_DIR / "comparison_report.md",
            ARTIFACT_DIR / "e1_eval_rollouts.json",
        ]
    if step_id == "write_claim_memo":
        return [CAMPAIGN_DIR / "decision_memo.md"]
    return []


def validate_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def validate_step_artifacts(step_id: str) -> tuple[bool, str, dict[str, Any]]:
    if step_id in {
        "u1_asset_geometry_audit",
        "u2_joint_semantics_audit",
        "u3_handle_region_audit",
        "u4_export_consistency_audit",
        "u5_seed_outlier_audit",
    }:
        payload = validate_json_file(ARTIFACT_DIR / f"{step_id}.json")
        ok = bool(
            payload
            and payload.get("decision")
            in {"upstream_clean", "upstream_suspect", "upstream_blocked"}
            and payload.get("conclusion_text")
        )
        return (
            ok,
            f"{step_id} artifact missing or incomplete" if not ok else "ok",
            payload or {},
        )
    if step_id == "a1_asset_scene_audit":
        payload = validate_json_file(ARTIFACT_DIR / "g1_asset_load.json")
        audit = validate_json_file(ARTIFACT_DIR / "urdf_asset_audit.json")
        ok = bool(
            payload
            and payload.get("passed")
            and audit
            and audit.get("passed")
            and audit.get("sampled_seed_count", 0) >= 4
        )
        merged = dict(payload or {})
        if audit:
            merged["audit"] = audit
        return ok, "asset / URDF audit missing or invalid" if not ok else "ok", merged
    if step_id == "a2_frame_transform_audit":
        payload = validate_json_file(ARTIFACT_DIR / "g2_obs_contract.json")
        audit = validate_json_file(ARTIFACT_DIR / "scene_frame_audit.json")
        handle_audit = validate_json_file(ARTIFACT_DIR / "handle_region_audit.json")
        ok = bool(
            payload
            and payload.get("passed")
            and audit
            and audit.get("passed")
            and handle_audit
            and handle_audit.get("passed")
        )
        merged = dict(payload or {})
        if audit:
            merged["audit"] = audit
        if handle_audit:
            merged["handle_audit"] = handle_audit
        return (
            ok,
            "scene observation contract audit missing or invalid" if not ok else "ok",
            merged,
        )
    if step_id == "b1_oracle_scripted_baseline":
        payload = validate_json_file(ARTIFACT_DIR / "b1_oracle_scripted_baseline.json")
        ok = bool(payload and payload.get("passed"))
        return (
            ok,
            "oracle scripted baseline missing or invalid" if not ok else "ok",
            payload or {},
        )
    if step_id == "b2_anygrasp_scripted_baseline":
        payload = validate_json_file(
            ARTIFACT_DIR / "b2_anygrasp_scripted_baseline.json"
        )
        ok = bool(payload and payload.get("passed"))
        return (
            ok,
            "AnyGrasp scripted baseline missing or invalid" if not ok else "ok",
            payload or {},
        )
    if step_id == "c1_teacher_native_rollout_rebuild":
        payload = validate_json_file(
            ARTIFACT_DIR / "c1_teacher_native_rollout_rebuild.json"
        )
        ok = bool(payload and payload.get("passed") and payload.get("leaderboard"))
        return (
            ok,
            "teacher native rollout rebuild did not produce a viable branch leaderboard"
            if not ok
            else "ok",
            payload or {},
        )
    if step_id == "c2_action_contract_repair":
        payload = validate_json_file(ARTIFACT_DIR / "c2_action_contract_repair.json")
        contract = validate_json_file(ACTIVE_ACTION_CONTRACT_PATH)
        preflight = validate_json_file(ARTIFACT_DIR / "c2_contract_preflight.json")
        target_dir = C2_REPLAY_DIR
        target_count = len(list(target_dir.glob("*.npz"))) if target_dir.exists() else 0
        if not preflight and target_count > 0:
            preflight = directory_preflight(sorted(target_dir.glob("*.npz")))
        strict_ok = bool(
            payload
            and payload.get("passed")
            and contract
            and contract.get("branch_id")
            and int(payload.get("successful_seed_count", 0)) >= 3
            and int(payload.get("successful_replays", 0)) == target_count
            and target_count > 0
        )
        fallback_ok = bool(
            payload
            and payload.get("passed")
            and contract
            and contract.get("branch_id")
            and str(contract.get("contract_mode")) == "teacher_success_fallback"
            and int(payload.get("teacher_successful_seed_count", 0)) >= 4
            and int(payload.get("teacher_successful_rollouts", 0)) >= 10
            and target_count > 0
            and preflight
            and preflight.get("passed")
        )
        ok = bool(strict_ok or fallback_ok)
        merged = dict(payload or {})
        if contract:
            merged["active_contract"] = contract
        if preflight:
            merged["contract_preflight"] = preflight
        merged["strict_ok"] = strict_ok
        merged["fallback_ok"] = fallback_ok
        return (
            ok,
            "action-contract repair did not yield a learnable rollout source"
            if not ok
            else "ok",
            merged,
        )
    if step_id == "c3_single_rollout_replay_gate":
        payload = validate_json_file(
            ARTIFACT_DIR / "c3_single_rollout_replay_gate.json"
        )
        ok = bool(payload and payload.get("passed"))
        return (
            ok,
            "single-rollout replay gate failed" if not ok else "ok",
            payload or {},
        )
    if step_id in {"d1_single_rollout_overfit", "d2_single_seed_overfit"}:
        payload = (
            validate_json_file(D1_CANDIDATE_ARTIFACT)
            if step_id == "d1_single_rollout_overfit"
            else validate_json_file(ARTIFACT_DIR / f"{step_id}.json")
        )
        strict_teacher = validate_json_file(STRICT_TEACHER_AUDIT_PATH) or {}
        env_contract = validate_json_file(ENV_CONTRACT_AUDIT_PATH) or (
            (strict_teacher.get("env_contract_audit") if strict_teacher else {}) or {}
        )
        strict_eval = (
            validate_json_file(ARTIFACT_DIR / "d1_single_rollout_strict_eval.json")
            or {}
        )
        ok = bool(
            payload
            and payload.get("passed")
            and strict_teacher
            and int(strict_teacher.get("accepted_rollout_count", 0)) > 0
            and env_contract
            and env_contract.get("passed")
        )
        if step_id == "d1_single_rollout_overfit":
            eval_summary = (strict_eval.get("evaluation") or {}).get("summary", {})
            ft = eval_summary.get("finetuned_mint", {})
            pt = eval_summary.get("pretrained_mint", {})
            ok = bool(
                ok
                and payload.get("promoted_candidate")
                and str((payload.get("promoted_candidate") or {}).get("candidate_role"))
                == "mainline"
                and strict_eval
                and float(ft.get("success_rate", 0.0))
                > float(pt.get("success_rate", 0.0))
            )
        else:
            ok = bool(
                ok
                and int(
                    payload.get("source_rollout_count", 0)
                    or ((payload.get("winner") or {}).get("source_rollout_count") or 0)
                )
                >= 2
                and str((payload.get("promoted_candidate") or {}).get("candidate_role"))
                == "mainline"
            )
        merged = dict(payload or {})
        if strict_teacher:
            merged["strict_teacher_audit"] = strict_teacher
        if env_contract:
            merged["env_contract_audit"] = env_contract
        if strict_eval:
            merged["strict_eval"] = strict_eval
        return (
            ok,
            f"{step_id} failed to show a positive train trend" if not ok else "ok",
            merged,
        )
    if step_id == "d3_train_seed_probe":
        payload = validate_json_file(ARTIFACT_DIR / "d3_train_seed_probe.json")
        strict_teacher = validate_json_file(STRICT_TEACHER_AUDIT_PATH) or {}
        env_contract = validate_json_file(ENV_CONTRACT_AUDIT_PATH) or (
            (strict_teacher.get("env_contract_audit") if strict_teacher else {}) or {}
        )
        selection_summary = (payload or {}).get("selection_summary") or {}
        total_selected_rollouts = int(
            sum(
                int(item.get("selected_rollout_count", 0))
                for item in selection_summary.get("included", [])
            )
        )
        ok = bool(
            payload
            and payload.get("passed")
            and payload.get("summary")
            and strict_teacher
            and int(len(selection_summary.get("included_seeds", []))) >= 4
            and total_selected_rollouts >= 8
            and int((payload.get("dataset") or {}).get("frame_count", 0)) >= 300
            and env_contract
            and env_contract.get("passed")
        )
        merged = dict(payload or {})
        if strict_teacher:
            merged["strict_teacher_audit"] = strict_teacher
        if env_contract:
            merged["env_contract_audit"] = env_contract
        return ok, "train-seed probe artifact missing" if not ok else "ok", merged
    if step_id == "e1_heldout_eval":
        summary = validate_json_file(EVAL_DIR / "comparison_summary.json")
        rollouts = validate_json_file(ARTIFACT_DIR / "e1_eval_rollouts.json")
        d3_payload = validate_json_file(ARTIFACT_DIR / "d3_train_seed_probe.json") or {}
        report_exists = (EVAL_DIR / "comparison_report.md").exists()
        ok = bool(
            summary
            and rollouts
            and report_exists
            and d3_payload.get("passed")
            and int(summary.get("episodes_per_seed", 0)) >= 3
            and summary.get("verdict")
            in {"claim_supported", "scientific_not_supported"}
        )
        return (
            ok,
            "evaluation summary/report missing or invalid" if not ok else "ok",
            summary or {},
        )
    if step_id == "write_claim_memo":
        memo_path = CAMPAIGN_DIR / "decision_memo.md"
        ok = memo_path.exists() and memo_path.read_text().strip() != ""
        return ok, "decision memo missing" if not ok else "ok", {"path": str(memo_path)}

    # ── LIBERO alignment gates ────────────────────────────────────────────────
    if step_id == "p4_physics_legal_gate":
        payload = validate_json_file(ARTIFACT_DIR / "p4_physics_legal_gate.json")
        ok = bool(
            payload
            and payload.get("passed")
            and int(payload.get("legal_seed_count", 0)) >= 6
        )
        return (
            ok,
            f"gate not passed: {payload.get('legal_seed_count', 0)}/6 seeds"
            if payload
            else f"{step_id} artifact missing",
            payload or {},
        )
    if step_id in {"p1a_state_vector_fix", "p1b_mujoco_env"}:
        payload = validate_json_file(ARTIFACT_DIR / f"{step_id}.json")
        ok = bool(payload and payload.get("passed"))
        return (
            ok,
            f"{step_id} artifact missing or failed" if not ok else "ok",
            payload or {},
        )

    path = ARTIFACT_DIR / f"{step_id}.json"
    payload = validate_json_file(path)
    ok = bool(payload and payload.get("passed"))
    return (
        ok,
        f"{step_id} artifact missing or failed" if not ok else "ok",
        payload or {},
    )


def summarize_verdict(state: dict[str, Any], summary: dict[str, Any]) -> str | None:
    d3_payload = validate_json_file(ARTIFACT_DIR / "d3_train_seed_probe.json") or {}
    eval_summary = load_json(EVAL_DIR / "comparison_summary.json", {})
    if state.get("verdict") in {"claim_supported", "scientific_not_supported"}:
        if d3_payload.get("passed") and eval_summary.get("verdict") == state.get(
            "verdict"
        ):
            return state["verdict"]
    if state.get("verdict") in {"implementation_blocked", "sim_eval_complete"}:
        return state["verdict"]
    if all_completed(state):
        verdict = eval_summary.get("verdict")
        if d3_payload.get("passed") and verdict == "claim_supported":
            return "claim_supported"
        if d3_payload.get("passed") and verdict == "scientific_not_supported":
            return "scientific_not_supported"
    return None


def blocker_type_from_error(error: str | None) -> str:
    text = (error or "").lower()
    if any(
        token in text for token in ["license", "gsnet.so", "lib_cxx.so", "checkpoint"]
    ):
        return "setup"
    if any(
        token in text
        for token in ["minkowskiengine", "graspnetapi", "import", "load_net"]
    ):
        return "env"
    if any(
        token in text for token in ["handle", "grasp", "pull_region", "pull region"]
    ):
        return "perception"
    if any(token in text for token in ["trajectory", "ik", "planner", "eef"]):
        return "trajectory"
    if any(
        token in text
        for token in ["delta", "contract", "range", "replay", "teacher", "env.step"]
    ):
        return "action_contract"
    if any(
        token in text
        for token in ["dataset", "lerobot", "nan", "preprocessor", "batch"]
    ):
        return "dataset"
    return "implementation"


def extract_train_probe_summary(
    train_probe: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if "summary" in train_probe and "finetuned_mint" in train_probe.get("summary", {}):
        return train_probe, train_probe
    covered = train_probe.get("covered_seed_probe", {})
    full = train_probe.get("full_train_probe", {}) or covered
    return covered, full


def derive_current_strongest_true_claim(
    state: dict[str, Any], summary: dict[str, Any]
) -> str | None:
    eval_summary = summary.get("evaluation", {})
    if eval_summary.get("strongest_true_claim"):
        return eval_summary.get("strongest_true_claim")

    learn_lane = state.get("learnability_lane", {})
    d3 = validate_json_file(ARTIFACT_DIR / "d3_train_seed_probe.json") or {}
    d2 = validate_json_file(ARTIFACT_DIR / "d2_single_seed_overfit.json") or {}
    d1 = (
        validate_json_file(D1_CANDIDATE_ARTIFACT)
        or validate_json_file(ARTIFACT_DIR / "d1_single_rollout_overfit.json")
        or {}
    )
    d1_strict = (
        validate_json_file(ARTIFACT_DIR / "d1_single_rollout_strict_eval.json") or {}
    )
    c3 = validate_json_file(ARTIFACT_DIR / "c3_single_rollout_replay_gate.json") or {}
    c2 = validate_json_file(ARTIFACT_DIR / "c2_action_contract_repair.json") or {}
    u5 = validate_json_file(ARTIFACT_DIR / "u5_seed_outlier_audit.json") or {}
    strict_teacher = validate_json_file(STRICT_TEACHER_AUDIT_PATH) or {}

    if d3.get("passed"):
        covered_probe, full_probe = extract_train_probe_summary(d3)
        probe = full_probe or covered_probe
        probe_summary = probe.get("summary", {})
        pt = probe_summary.get("pretrained_mint", {})
        ft = probe_summary.get("finetuned_mint", {})
        return (
            "Fine-tuning now shows a positive train-side trend on natural robot trajectories: "
            f"pretrained succeeds on {pt.get('successes', 0)}/{pt.get('n_episodes', 0)} train episodes while "
            f"finetuned succeeds on {ft.get('successes', 0)}/{ft.get('n_episodes', 0)}. "
            "Held-out evaluation is now scientifically meaningful."
        )

    if d2.get("passed"):
        winner = d2.get("winner", {})
        eval_block = winner.get("evaluation", {})
        eval_summary = eval_block.get("summary", {})
        pt = eval_summary.get("pretrained_mint", {})
        ft = eval_summary.get("finetuned_mint", {})
        seed = d2.get("selected_seed")
        return (
            "Single-seed learnability is now established on natural robot trajectories: "
            f"for train seed {seed}, pretrained succeeds on {pt.get('successes', 0)}/{pt.get('n_episodes', 0)} episodes while "
            f"finetuned succeeds on {ft.get('successes', 0)}/{ft.get('n_episodes', 0)}."
        )

    if d1.get("passed"):
        promoted = d1.get("promoted_candidate") or {}
        winner = promoted.get("winning_attempt", {}) or d1.get("winner", {})
        eval_block = winner.get("evaluation", {})
        eval_summary = eval_block.get("summary", {})
        pt = eval_summary.get("pretrained_mint", {})
        ft = eval_summary.get("finetuned_mint", {})
        seed = promoted.get("selected_seed") or d1.get("selected_seed")
        variant = promoted.get("winner_variant") or winner.get(
            "variant", "winning overfit variant"
        )
        return (
            "A single AnyGrasp-conditioned teacher-success rollout is now learnable: "
            f"on train seed {seed}, {variant} improves success from {pt.get('successes', 0)}/{pt.get('n_episodes', 0)} "
            f"to {ft.get('successes', 0)}/{ft.get('n_episodes', 0)}."
        )
    if strict_teacher.get("mainline_candidate_order"):
        mainline = strict_teacher.get("mainline_candidate_order", [])
        return (
            "The strict-valid teacher pool now distinguishes diagnostic and mainline candidates; "
            f"{len(mainline)} D2-feasible rollout(s) remain eligible for the mainline D1->D2 path."
        )
    strict_eval_summary = (d1_strict.get("evaluation") or {}).get("summary", {})
    strict_ft = strict_eval_summary.get("finetuned_mint", {})
    strict_pt = strict_eval_summary.get("pretrained_mint", {})
    if d1_strict and strict_teacher:
        accepted = int(strict_teacher.get("accepted_rollout_count", 0))
        seeds = int(strict_teacher.get("accepted_seed_count", 0))
        if accepted > 0 and float(strict_ft.get("success_rate", 0.0)) <= float(
            strict_pt.get("success_rate", 0.0)
        ):
            return (
                "AnyGrasp-conditioned teacher trajectories now provide a strict-valid attached drawer-opening dataset "
                f"({accepted} rollouts across {seeds} train seeds), but under strict success semantics the current 8D MINT "
                "policy interface still does not learn true attached-open behavior from that data."
            )

    if c3.get("passed"):
        return (
            "A contract-preflight-approved teacher-success rollout source is now available for learnability testing, "
            "even though strict replay is still being repaired in parallel."
        )

    if (
        c2.get("passed")
        and learn_lane.get("source_contract_mode") == "teacher_success_fallback"
    ):
        return (
            "Teacher-success fallback has produced a usable natural-rollout source for learnability tests, "
            "but train-side improvement has not been established yet."
        )

    if u5:
        if u5.get("decision") == "upstream_clean":
            return "Upstream Infinigen audits are currently clean enough that the remaining bottlenecks are downstream contract and learnability issues."
        if u5.get("decision") in {"upstream_suspect", "upstream_blocked"}:
            return str(u5.get("conclusion_text"))

    return summary.get("strongest_true_claim")


def build_review(
    state: dict[str, Any], summary: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    completed = sum(1 for item in state["queue"] if item.get("status") == "completed")
    total = len(state["queue"])
    running_step = active_or_next_step(state)
    verdict = summarize_verdict(state, summary) or "awaiting_execution"
    blocker_type = (
        blocker_type_from_error(state.get("last_error"))
        if state.get("verdict") == "implementation_blocked"
        else "none"
    )
    train_probe = load_json(EVAL_DIR / "train_seed_probe.json", {})
    covered_probe, full_train_probe = extract_train_probe_summary(train_probe)
    eval_summary = summary.get("evaluation", {})
    strict_teacher = validate_json_file(STRICT_TEACHER_AUDIT_PATH) or {}
    env_contract = validate_json_file(ENV_CONTRACT_AUDIT_PATH) or (
        (strict_teacher.get("env_contract_audit") if strict_teacher else {}) or {}
    )
    d1_strict = (
        validate_json_file(ARTIFACT_DIR / "d1_single_rollout_strict_eval.json") or {}
    )

    strengths: list[str] = []
    weaknesses: list[str] = []
    next_actions: list[str] = []

    if summary.get("proxy_baseline_archived"):
        strengths.append(
            "Proxy baseline has been archived before the robot-trajectory revision."
        )
    if completed:
        strengths.append(f"{completed}/{total} queue steps have validated artifacts.")
    if state.get("history"):
        strengths.append("Campaign history is being recorded incrementally.")
    if strict_teacher:
        strengths.append(
            f"Strict-valid teacher pool is explicit: {strict_teacher.get('accepted_rollout_count', 0)} rollout(s) across {strict_teacher.get('accepted_seed_count', 0)} seed(s)."
        )
    if env_contract:
        if env_contract.get("passed"):
            strengths.append(
                "Environment contract audit passed: no-attach drawer exploit is below threshold on the sampled seeds."
            )
        else:
            weaknesses.append(
                "Environment contract audit failed: the sim still allows major drawer motion before stable attachment on some sampled seeds."
            )
            next_actions.insert(
                0,
                "Repair the sim/env action contract before any further D1/D2/D3 learning runs.",
            )
    if state.get("upstream_audit_lane", {}).get("status") == "clean":
        strengths.append(
            "Upstream Infinigen audits are currently clean enough to keep focus on downstream contract and learnability repair."
        )
    if state.get("strict_replay_lane", {}).get("status") == "passed":
        strengths.append(
            "Strict replay lane is now validated on multi-seed AnyGrasp rollouts."
        )
    elif state.get("learnability_lane", {}).get("status") in {
        "ready_from_contract",
        "ready_for_overfit",
        "single_rollout_passed",
        "single_seed_passed",
        "train_trend_passed",
        "heldout_complete",
        "diagnostic_winner_only",
    }:
        strengths.append(
            "Learnability lane has a valid rollout source and can continue independently from strict replay repair."
        )

    if state.get("verdict") == "implementation_blocked":
        weaknesses.append(
            summary.get("last_error")
            or state.get("last_error")
            or "A non-recoverable pipeline failure needs repair."
        )
        next_actions.append("Repair the blocked gate and rerun the loop.")
    elif state.get("learnability_lane", {}).get("status") == "env_contract_blocked":
        weaknesses.append(
            "The environment contract audit failed, so no further learnability result is trustworthy until no-attach drawer exploit is eliminated."
        )
        next_actions.append(
            "Repair the sim/env contract and rerun the environment audit before D1."
        )
    elif (
        state.get("learnability_lane", {}).get("status")
        == "single_rollout_invalidated_under_strict_success"
    ):
        eval_block = (d1_strict.get("evaluation") or {}).get("summary", {})
        ft = eval_block.get("finetuned_mint", {})
        pt = eval_block.get("pretrained_mint", {})
        weaknesses.append(
            "Strict D1 invalidated the old single-rollout positive signal: "
            f"pretrained is {pt.get('successes', 0)}/{pt.get('n_episodes', 0)} and finetuned is {ft.get('successes', 0)}/{ft.get('n_episodes', 0)} "
            "when success requires attached-open behavior."
        )
        next_actions.append(
            "Rebuild the strict-valid teacher dataset and use it as the only source for D1/D2/D3."
        )
        next_actions.append(
            "If the strict-valid pool is too small or still exploit-prone, patch handle semantics/export metadata before retrying D1."
        )
    elif state.get("learnability_lane", {}).get("status") == "diagnostic_winner_only":
        promoted = state.get("learnability_lane", {}).get("promoted_candidate") or {}
        weaknesses.append(
            "The current D1 winner is diagnostic-only: it proves a single rollout is learnable, but its source seed does not have enough strict-valid rollouts to support D2 mainline."
        )
        next_actions.append(
            f"Keep the D1 diagnostic winner as positive evidence, but rebuild teacher/mainline candidates for seed `{promoted.get('selected_seed')}`-independent D2-feasible seeds only."
        )
        next_actions.append("Do not run D2/D3/E1 from a diagnostic-only D1 winner.")
    elif verdict in {"claim_supported", "scientific_not_supported"}:
        if (
            eval_summary.get("deferred_reason") == "train_seed_reproducibility_not_met"
            and train_probe
        ):
            ft = covered_probe.get("summary", {}).get("finetuned_mint", {})
            weaknesses.append(
                f"Train-seed reproducibility is still weak: fine-tuned success is only {ft.get('successes', 0)}/{ft.get('n_episodes', 0)} at {covered_probe.get('eval_max_steps', 96)}-step evaluation, so held-out evaluation was deferred."
            )
            next_actions.append(
                "Repair the natural robot-action contract before rerunning held-out evaluation."
            )
            next_actions.append(
                "Do not interpret the current verdict as a held-out negative result; the bottleneck is still train-seed reproducibility."
            )
        elif train_probe:
            ft = full_train_probe.get("summary", {}).get("finetuned_mint", {})
            weaknesses.append(
                f"Held-out success remains 0/5, and fine-tuned train-seed success is only {ft.get('successes', 0)}/{ft.get('n_episodes', 0)} at {full_train_probe.get('eval_max_steps', 96)}-step evaluation."
            )
            next_actions.append(
                "Review `evaluation/train_seed_probe.json` together with the held-out comparison before calling this a pure generalization failure."
            )
            next_actions.append(
                "Refine the G5 robot-action contract or increase clean robot-trajectory coverage before the next MINT run."
            )
        else:
            next_actions.append("Review the final comparison report and decision memo.")
            next_actions.append(
                "Decide whether the robot-trajectory sim claim is sufficient or needs more held-out scale."
            )
    elif running_step:
        next_actions.append(
            f"Run or resume the next pending queue step: `{running_step}`."
        )
    else:
        next_actions.append("No active step found; inspect queue state.")

    if state.get("upstream_audit_lane", {}).get("status") in {"suspect", "blocked"}:
        weaknesses.append(
            state.get("upstream_audit_lane", {}).get("root_cause")
            or "Upstream audit lane remains suspect."
        )
        next_actions.insert(
            0,
            "Keep the upstream audit lane explicit: only modify `infinigen/**` if the audited evidence points to a specific export or geometry defect.",
        )
        if state.get("upstream_audit_lane", {}).get("patch_allowed"):
            if (
                state.get("upstream_audit_lane", {}).get("blocking_step")
                == "u3_handle_region_audit"
            ):
                next_actions.insert(
                    1,
                    "D1 candidate search has been exhausted under strict success; the next allowed upstream move is a minimal handle-metadata export patch.",
                )
        else:
            next_actions.insert(
                1,
                "Upstream remains suspect, but do not patch `infinigen/**` yet; wait for D1 candidate search to finish under strict success.",
            )

    if state.get("strict_replay_lane", {}).get("status") == "repairing":
        next_actions.insert(
            0,
            "Keep the strict replay lane running in parallel so the teacher/native contract becomes scientifically cleaner even if the learnability lane advances first.",
        )
    if state.get("learnability_lane", {}).get("status") in {
        "ready_from_contract",
        "ready_for_overfit",
    }:
        next_actions.insert(
            0,
            "Advance the learnability lane into D1 overfit instead of waiting for strict replay to turn fully green.",
        )
    if state.get("phase_gate") == "action_contract_repair":
        next_actions = [
            "Keep the loop on the C-layer until native teacher rollouts and replay share the same control semantics.",
            "Do not restart full MINT training before C3 selects a contract-preflight-approved rollout source.",
            *next_actions,
        ]

    claim_assessment = (
        derive_current_strongest_true_claim(state, summary)
        or "No evaluation evidence yet."
    )
    if verdict == "claim_supported":
        decision = "writeup"
    elif verdict == "scientific_not_supported":
        decision = "revise_claim"
    else:
        decision = "run_experiments"

    if verdict == "claim_supported":
        evidence_score = 9
    elif verdict == "scientific_not_supported":
        evidence_score = 8
    elif state.get("learnability_lane", {}).get("status") == "train_trend_passed":
        evidence_score = 7
    elif state.get("learnability_lane", {}).get("status") == "single_seed_passed":
        evidence_score = 5
    elif state.get("learnability_lane", {}).get("status") in {
        "single_rollout_passed",
        "diagnostic_winner_only",
    }:
        evidence_score = 3
    elif (
        state.get("learnability_lane", {}).get("status")
        == "single_rollout_invalidated_under_strict_success"
    ):
        evidence_score = 4
    elif state.get("learnability_lane", {}).get("status") in {
        "ready_from_contract",
        "ready_for_overfit",
    }:
        evidence_score = 2
    elif summary.get("evaluation"):
        evidence_score = 5
    else:
        evidence_score = 6 if state.get("phase_gate") == "action_contract_repair" else 0

    return {
        "phase": manifest["campaign"]["phase"],
        "phase_gate": state.get("phase_gate", manifest["campaign"]["phase_gate"]),
        "workflow_score": round((completed / max(total, 1)) * 10),
        "evidence_score": evidence_score,
        "decision": decision,
        "blocker_type": blocker_type,
        "verdict": verdict,
        "claim_assessment": claim_assessment,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "next_actions": next_actions,
        "generated_at": now_iso(),
    }


def build_next_actions(state: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    if state.get("verdict") == "implementation_blocked":
        actions = [
            {
                "type": "repair_and_rerun",
                "target": active_or_next_step(state) or "unknown",
                "reason": summary_error(state),
            }
        ]
    elif review.get("verdict") in {"scientific_not_supported", "claim_supported"}:
        actions = [
            {
                "type": "writeup",
                "target": "decision_memo",
                "reason": "Terminal verdict reached",
            }
        ]
    else:
        actions = [
            {
                "type": "run_experiments",
                "target": active_or_next_step(state) or "none",
                "reason": "Continue queue execution",
            }
        ]
    return {
        "decision": review.get("decision"),
        "actions": actions,
        "generated_at": now_iso(),
    }


def summary_error(state: dict[str, Any]) -> str:
    running = state.get("active_runtime") or {}
    return str(
        running.get("last_error") or state.get("last_error") or "unknown failure"
    )


def render_campaign_status(
    manifest: dict[str, Any], state: dict[str, Any], review: dict[str, Any]
) -> str:
    lines = [
        "# MINT Drawer Campaign Dashboard",
        "",
        f"**Updated**: {now_iso()}",
        f"**Campaign**: `{manifest['campaign']['id']}`",
        f"**Phase**: `{state.get('phase')}`",
        f"**Gate**: `{state.get('phase_gate')}`",
        f"**Claim**: `{manifest['campaign']['main_claim']}`",
        f"**Revision**: `{state.get('revision', {}).get('active')}`",
        "",
        "## Queue",
        "",
    ]
    for item in state.get("queue", []):
        lines.append(f"- `{item.get('id')}`: {item.get('status')}")
    runtime = state.get("active_runtime") or {}
    strict_lane = state.get("strict_replay_lane", {})
    learn_lane = state.get("learnability_lane", {})
    upstream_lane = state.get("upstream_audit_lane", {})
    summary = load_summary()
    d1_search = summary.get("d1_candidate_search", {}) or {}
    env_contract = validate_json_file(ENV_CONTRACT_AUDIT_PATH) or (
        summary.get("env_contract_audit") or {}
    )
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- Controller ID: `{(state.get('controller') or {}).get('controller_id')}`",
            f"- Run ID: `{(state.get('controller') or {}).get('run_id')}`",
            f"- Active Step: `{runtime.get('step_id')}`",
            f"- Active Lane: `{runtime.get('active_lane')}`",
            f"- Active Branch: `{runtime.get('active_branch')}`",
            f"- Worker PID: `{runtime.get('worker_pid')}`",
            f"- Worker Kind: `{runtime.get('worker_kind')}`",
            f"- Attempt: `{runtime.get('attempt_no')}`",
            f"- Launched At: `{runtime.get('launched_at')}`",
            f"- Heartbeat: `{runtime.get('heartbeat_at')}`",
            f"- Latest Artifact: `{runtime.get('latest_artifact')}`",
            f"- Last Error: `{runtime.get('last_error')}`",
            "",
            "## Review",
            "",
            f"- Success Semantics Version: `{summary.get('success_semantics_version')}`",
            f"- Dataset Fingerprint: `{summary.get('dataset_fingerprint')}`",
            f"- Invalidated Results: `{summary.get('invalidated_results')}`",
            f"- Env Contract Audit Passed: `{env_contract.get('passed') if env_contract else None}`",
            f"- Env Contract Failed Seeds: `{env_contract.get('failed_seeds') if env_contract else None}`",
            f"- Verdict: `{review.get('verdict')}`",
            f"- Decision: `{review.get('decision')}`",
            f"- Workflow Score: `{review.get('workflow_score')}`/10",
            f"- Evidence Score: `{review.get('evidence_score')}`/10",
            f"- Claim assessment: {review.get('claim_assessment')}",
        ]
    )
    lines.extend(
        [
            "",
            "## Lanes",
            "",
            f"- Strict Replay Lane: `{strict_lane.get('status')}`",
            f"- Strict Replay Best Branch: `{strict_lane.get('best_branch')}`",
            f"- Strict Replay Metrics: `{strict_lane.get('best_metrics')}`",
            f"- Learnability Lane: `{learn_lane.get('status')}`",
            f"- Learnability Contract Mode: `{learn_lane.get('source_contract_mode')}`",
            f"- Learnability Source Branch: `{learn_lane.get('source_branch')}`",
            f"- Learnability Current Step: `{learn_lane.get('current_step')}`",
            f"- Strict-Valid Teacher Rollouts: `{learn_lane.get('strict_teacher_rollout_count')}`",
            f"- Strict-Valid Teacher Seeds: `{learn_lane.get('strict_teacher_seed_count')}`",
            f"- D2-Feasible Seeds: `{summary.get('d1_candidate_search', {}).get('strict_teacher_audit', {}).get('d2_feasible_seeds') if summary.get('d1_candidate_search') else None}`",
            f"- D1 Candidate Order: `{[item.get('rollout_path') for item in learn_lane.get('d1_candidate_order', [])]}`",
            f"- D1 Promoted Candidate: `{(learn_lane.get('promoted_candidate') or {}).get('selected_rollout')}`",
            f"- Upstream Audit Lane: `{upstream_lane.get('status')}`",
            f"- Upstream Root Cause: `{upstream_lane.get('root_cause')}`",
            f"- Upstream Blocking Step: `{upstream_lane.get('blocking_step')}`",
            f"- Upstream Affected Seeds: `{upstream_lane.get('affected_seeds')}`",
            f"- Infinigen Fix Required: `{upstream_lane.get('fix_required')}`",
            f"- Infinigen Fix Allowed: `{upstream_lane.get('patch_allowed')}`",
            f"- Infinigen Patch Gate Reason: `{upstream_lane.get('patch_gate_reason')}`",
        ]
    )
    if d1_search.get("promoted_candidate"):
        lines.append(
            f"- D1 Winner Variant: `{d1_search['promoted_candidate'].get('winner_variant')}`"
        )
    for weakness in review.get("weaknesses", []):
        lines.append(f"- Weakness: {weakness}")
    for action in review.get("next_actions", []):
        lines.append(f"- Next: {action}")
    lines.extend(["", "## Recent History", ""])
    for event in state.get("history", [])[-12:]:
        lines.append(
            f"- {event.get('at')}: {event.get('event')} | {event.get('details')}"
        )
    return "\n".join(lines) + "\n"


def refresh_campaign_views(
    state: dict[str, Any],
    manifest: dict[str, Any],
    summary: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if summary is None:
        summary = load_summary()
    sync_lane_state_from_artifacts(state)
    summary["phase"] = state.get("phase", manifest["campaign"]["phase"])
    summary["status"] = state.get("verdict") or active_or_next_step(state) or "idle"
    summary["last_error"] = state.get("last_error")
    summary["strongest_true_claim"] = derive_current_strongest_true_claim(
        state, summary
    )
    summary["success_semantics_version"] = STRICT_SUCCESS_VERSION
    summary["invalidated_results"] = list(state.get("invalidated_results", []))
    summary["strict_replay_lane"] = deepcopy(state.get("strict_replay_lane", {}))
    summary["learnability_lane"] = deepcopy(state.get("learnability_lane", {}))
    summary["upstream_audit_lane"] = deepcopy(state.get("upstream_audit_lane", {}))
    summary["d1_candidate_search"] = validate_json_file(D1_CANDIDATE_ARTIFACT) or {}
    summary["env_contract_audit"] = validate_json_file(ENV_CONTRACT_AUDIT_PATH) or {}
    active_runtime = state.get("active_runtime") or {}
    summary["dataset_fingerprint"] = active_runtime.get(
        "active_fingerprint"
    ) or summary.get("dataset_fingerprint")
    save_summary(summary)
    review = build_review(state, summary, manifest)
    save_review(review)
    save_next_actions(build_next_actions(state, review))
    write_text_atomic(
        CAMPAIGN_DIR / "campaign_status.md",
        render_campaign_status(manifest, state, review),
    )
    return review, summary


def update_watch_status(payload: dict[str, Any]) -> None:
    if "controller_id" not in payload or "run_id" not in payload:
        context = controller_context_from_state()
        payload.setdefault("controller_id", context.get("controller_id"))
        payload.setdefault("run_id", context.get("run_id"))
    payload["timestamp"] = now_iso()
    write_json_atomic(RUNTIME_DIR / "watch_status.json", payload)


def sync_running_step_progress(
    step_id: str,
    *,
    worker_pid: int | None,
    worker_kind: str,
    active_lane: str | None,
    active_branch: str | None,
    active_fingerprint: dict[str, Any] | None,
    latest_artifact: str | None,
    stage: str | None,
    last_error: str | None = None,
) -> None:
    ensure_contract_files()
    manifest = load_manifest()
    state = load_state()
    summary = load_summary()
    sync_completed_steps_from_artifacts(state)
    try:
        set_queue_status(state, step_id, "running", error=last_error)
    except KeyError:
        pass
    state["last_error"] = last_error
    prior_runtime = state.get("active_runtime") or {}
    context = controller_context_from_state(state)
    state["active_runtime"] = {
        "step_id": step_id,
        "active_lane": active_lane,
        "active_branch": active_branch,
        "active_fingerprint": active_fingerprint,
        "controller_id": context.get("controller_id"),
        "run_id": context.get("run_id"),
        "worker_pid": worker_pid,
        "worker_kind": worker_kind,
        "attempt_no": queue_entry(state, step_id).get("attempts", 0)
        if any(item.get("id") == step_id for item in state.get("queue", []))
        else 0,
        "launched_at": prior_runtime.get("launched_at") or now_iso(),
        "heartbeat_at": now_iso(),
        "latest_artifact": latest_artifact,
        "last_error": last_error,
        "stage": stage,
    }
    save_state(state)
    review, summary = refresh_campaign_views(state, manifest, summary)
    update_watch_status(
        {
            "controller_id": context.get("controller_id"),
            "run_id": context.get("run_id"),
            "supervisor_pid": None,
            "inner_watcher_pid": None,
            "worker_pid": worker_pid,
            "worker_kind": worker_kind,
            "queue_step": step_id,
            "stage": stage or step_id,
            "active_lane": active_lane,
            "active_branch": active_branch,
            "status": "running",
            "last_heartbeat": now_iso(),
            "latest_artifact": latest_artifact,
            "last_successful_milestone": state.get("learnability_lane", {}).get(
                "last_positive_step"
            ),
            "retry_count": state.get("retries", {}).get(step_id, 0),
            "last_repair_action": state.get("last_repair_action"),
            "last_error": last_error,
            "verdict": review.get("verdict"),
        }
    )


def reconcile_step_truth(
    step_id: str,
    *,
    passed: bool,
    latest_artifact: str | None = None,
    last_error: str | None = None,
    active_branch: str | None = None,
    active_fingerprint: dict[str, Any] | None = None,
) -> None:
    ensure_contract_files()
    manifest = load_manifest()
    state = load_state()
    summary = load_summary()
    sync_completed_steps_from_artifacts(state)
    try:
        set_queue_status(
            state, step_id, "completed" if passed else "failed", error=last_error
        )
    except KeyError:
        pass
    lane = step_lane(step_id, state)
    context = controller_context_from_state(state)
    if not passed:
        state["last_error"] = last_error or f"{step_id} failed"
    else:
        state["last_error"] = None
    state["active_runtime"] = {
        "step_id": step_id,
        "active_lane": lane,
        "active_branch": active_branch,
        "active_fingerprint": active_fingerprint,
        "controller_id": context.get("controller_id"),
        "run_id": context.get("run_id"),
        "worker_pid": None,
        "worker_kind": "manual_reconcile",
        "attempt_no": queue_entry(state, step_id).get("attempts", 0)
        if any(item.get("id") == step_id for item in state.get("queue", []))
        else 0,
        "launched_at": now_iso(),
        "heartbeat_at": now_iso(),
        "latest_artifact": latest_artifact,
        "last_error": last_error,
    }
    append_history(
        state, "step_reconciled", f"{step_id} => {'completed' if passed else 'failed'}"
    )
    save_state(state)
    review, summary = refresh_campaign_views(state, manifest, summary)
    update_watch_status(
        {
            "controller_id": context.get("controller_id"),
            "run_id": context.get("run_id"),
            "supervisor_pid": None,
            "inner_watcher_pid": None,
            "worker_pid": None,
            "worker_kind": "manual_reconcile",
            "queue_step": step_id,
            "stage": step_id,
            "active_lane": lane,
            "active_branch": active_branch,
            "status": "completed" if passed else "failed",
            "last_heartbeat": now_iso(),
            "latest_artifact": latest_artifact,
            "last_successful_milestone": state.get("learnability_lane", {}).get(
                "last_positive_step"
            ),
            "retry_count": state.get("retries", {}).get(step_id, 0),
            "last_repair_action": state.get("last_repair_action"),
            "last_error": last_error,
            "verdict": review.get("verdict"),
        }
    )


def run_memory_sync(
    title: str,
    summary: str,
    completed: list[str],
    next_actions: list[str],
    artifacts: dict[str, str] | None = None,
    lessons: list[str] | None = None,
    bugs_fixed: list[dict[str, str]] | None = None,
) -> None:
    payload = {
        "summary": summary,
        "completed": completed,
        "next": next_actions,
        "artifacts": artifacts or {},
        "lessons": lessons or [],
        "bugs_fixed": bugs_fixed or [],
    }
    payload_path = RUNTIME_DIR / "memory_payload.json"
    write_json_atomic(payload_path, payload)
    record_script = (
        PROJECT_ROOT
        / ".cursor"
        / "skills"
        / "infinigen-project-memory"
        / "scripts"
        / "record_iteration.py"
    )
    history_script = (
        PROJECT_ROOT
        / ".cursor"
        / "skills"
        / "infinigen-project-memory"
        / "scripts"
        / "write_campaign_history.py"
    )
    update_status_script = (
        PROJECT_ROOT
        / ".cursor"
        / "skills"
        / "infinigen-project-memory"
        / "scripts"
        / "update_status.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(record_script),
            "--project-root",
            str(PROJECT_ROOT),
            "--payload",
            str(payload_path),
        ],
        check=False,
    )
    subprocess.run(
        [
            sys.executable,
            str(history_script),
            "--project-root",
            str(PROJECT_ROOT),
            "--campaign-dir",
            str(CAMPAIGN_DIR),
            "--title",
            title,
        ],
        check=False,
    )
    subprocess.run(
        [
            sys.executable,
            str(update_status_script),
            "--project-root",
            str(PROJECT_ROOT),
        ],
        check=False,
    )


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = sorted(
        output_dir.glob("**/pretrained_model"), key=lambda p: len(str(p))
    )
    return candidates[-1] if candidates else None


def ensure_contract_files() -> None:
    ensure_dirs()
    manifest = load_manifest()
    write_manifest(manifest)
    defaults = {
        CAMPAIGN_DIR / "state.json": default_state(manifest),
        CAMPAIGN_DIR / "summary.json": default_summary(manifest),
        CAMPAIGN_DIR / "review.json": default_review(manifest),
        CAMPAIGN_DIR / "next_actions.json": default_next_actions(),
    }
    for path, payload in defaults.items():
        if not path.exists():
            write_json_atomic(path, payload)
    if not (CAMPAIGN_DIR / "decision_memo.md").exists():
        write_text_atomic(
            CAMPAIGN_DIR / "decision_memo.md",
            "# MINT Drawer Robot-Trajectory Campaign Decision Memo\n\nCurrent campaign is active. Final verdict will be written automatically when a terminal state is reached.\n",
        )
    if not (CAMPAIGN_DIR / "campaign_status.md").exists():
        state = load_state()
        review = load_review()
        write_text_atomic(
            CAMPAIGN_DIR / "campaign_status.md",
            render_campaign_status(manifest, state, review),
        )

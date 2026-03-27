#!/usr/bin/env python3
"""Artifact-first auto-review loop for the MINT drawer campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from mint_common import (
    BRANCH_PROGRESS_PATH,
    CAMPAIGN_DIR,
    D1_PROGRESS_PATH,
    DEFAULT_CONDA_RUN,
    DEFAULT_PYTHON,
    PROJECT_ROOT,
    QUEUE_ENVS,
    RUNTIME_DIR,
    acquire_controller_lease,
    active_or_next_step,
    append_history,
    controller_identity,
    default_state,
    ensure_contract_files,
    expected_artifacts,
    increment_retry,
    load_manifest,
    load_state,
    load_summary,
    now_iso,
    queue_entry,
    refresh_campaign_views,
    refresh_controller_lease,
    release_controller_lease,
    reset_step,
    run_memory_sync,
    save_state,
    save_summary,
    set_queue_status,
    step_lane,
    summarize_verdict,
    sync_completed_steps_from_artifacts,
    sync_lane_state_from_artifacts,
    update_watch_status,
    validate_step_artifacts,
    write_runtime_meta,
    write_text_atomic,
)

MAX_RETRIES = 2
LOOP_LOG_PATH = RUNTIME_DIR / "overnight_loop.log"
CONTROLLER_INFO = controller_identity()


def _d1_exhausted(step_id: str, message: str) -> bool:
    if step_id != "d1_single_rollout_overfit":
        return False
    terminal_markers = {
        "failed to show a positive train trend",
        "teacher_mainline_failed_under_clean_control",
        "d1_stage_a_failed_requires_u3_regime_b",
    }
    return any(marker in message for marker in terminal_markers)


def _block_on_d1_exhaustion(state: dict, summary: dict, message: str) -> None:
    state["verdict"] = "implementation_blocked"
    state["phase_gate"] = (
        "u3_regime_b_gate" if "u3_regime_b" in message else "upstream_handle_patch_gate"
    )
    state["last_error"] = message
    state["last_repair_action"] = (
        "d1_stage_a_requires_u3_regime_b"
        if "u3_regime_b" in message
        else "d1_exhausted_requires_upstream_handle_patch"
    )
    append_history(
        state,
        "implementation_blocked",
        "D1 mainline exhausted under clean control; stop rerunning D1 blindly and require either the u3 regime-B lane or the upstream handle patch gate.",
    )
    summary["last_error"] = message
    summary["blocked_reason"] = state["last_repair_action"]
    save_summary(summary)
    save_state(state)


def _reset_steps(state: dict, step_ids: list[str]) -> None:
    for step_id in step_ids:
        try:
            reset_step(state, step_id)
        except KeyError:
            continue


def _expand_coverage_and_retry(
    state: dict, summary: dict, trigger_step: str, reason: str
) -> bool:
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-until-terminal", action="store_true")
    return parser.parse_args()


def log_loop_event(message: str) -> None:
    LOOP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOOP_LOG_PATH.open("a") as handle:
        handle.write(f"{now_iso()} {message}\n")


def script_for(step_id: str) -> Path | None:
    if step_id == "write_claim_memo":
        return None
    return PROJECT_ROOT / "scripts" / "mint" / f"run_{step_id}.py"


def load_decision_memo(summary: dict, state: dict, manifest: dict) -> str:
    verdict = (
        summarize_verdict(state, summary)
        or state.get("verdict")
        or "awaiting_execution"
    )
    comparison = summary.get("evaluation", {}).get("comparison", {})
    strongest = (
        summary.get("strongest_true_claim") or "No strongest true claim recorded."
    )
    lines = [
        "# MINT Drawer Robot-Trajectory Campaign Decision Memo",
        "",
        f"**Updated**: {now_iso()}",
        f"**Phase**: `{state.get('phase')}`",
        f"**Gate**: `{state.get('phase_gate')}`",
        f"**Verdict**: `{verdict}`",
        "",
        "## Claim Assessment",
        "",
        strongest,
        "",
        "## Evidence Snapshot",
        "",
    ]
    for name, payload in comparison.items():
        lines.append(
            f"- {name}: success_rate={payload.get('success_rate')} grasp_success={payload.get('grasp_success_rate')} "
            f"pull_distance={payload.get('pull_distance_mean')} time={payload.get('time_to_completion_mean')} n={payload.get('n_episodes')}"
        )
    if summary.get("coverage_expansion_mode"):
        lines.extend(
            [
                "",
                "## Coverage Mode",
                "",
                f"- rollout coverage mode: `{summary['coverage_expansion_mode']}`",
            ]
        )
    lines.extend(["", "## Queue", ""])
    for item in state.get("queue", []):
        lines.append(f"- `{item.get('id')}`: {item.get('status')}")
    return "\n".join(lines) + "\n"


def step_log_path(step_id: str) -> Path:
    return RUNTIME_DIR / step_id / "launcher.log"


def process_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def build_step_command(env_name: str, script: Path) -> list[str]:
    controller_id = CONTROLLER_INFO["controller_id"]
    run_id = CONTROLLER_INFO["run_id"]
    return [
        "bash",
        "-lc",
        f"cd {PROJECT_ROOT} && MINT_CONTROLLER_ID='{controller_id}' MINT_RUN_ID='{run_id}' {DEFAULT_CONDA_RUN} -n {env_name} {DEFAULT_PYTHON} -u {script}",
    ]


def run_script_step(
    step_id: str, env_name: str, state: dict, manifest: dict
) -> tuple[bool, str]:
    script = script_for(step_id)
    if script is None:
        raise ValueError(step_id)
    log_path = step_log_path(step_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_step_command(env_name, script)
    for artifact in expected_artifacts(step_id):
        if artifact.exists():
            artifact.unlink()
    with log_path.open("w") as log_handle:
        proc = subprocess.Popen(
            cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True
        )
        active_runtime = {
            "step_id": step_id,
            "active_lane": step_lane(step_id, state),
            "controller_id": CONTROLLER_INFO["controller_id"],
            "run_id": CONTROLLER_INFO["run_id"],
            "worker_kind": "gate_subprocess",
            "worker_pid": proc.pid,
            "attempt_no": state.get("retries", {}).get(step_id, 0) + 1,
            "launched_at": now_iso(),
            "heartbeat_at": now_iso(),
            "latest_artifact": None,
            "last_error": None,
        }
        state["active_runtime"] = active_runtime
        save_state(state)
        write_runtime_meta(
            step_id,
            {
                "step_id": step_id,
                "status": "running",
                "worker_pid": proc.pid,
                "env": env_name,
                "command": " ".join(cmd),
                "launched_at": active_runtime["launched_at"],
                "log_path": str(log_path),
            },
        )
        log_loop_event(
            f"[start] step={step_id} lane={active_runtime['active_lane']} env={env_name} pid={proc.pid}"
        )

        while proc.poll() is None:
            refresh_controller_lease(
                CONTROLLER_INFO["controller_id"],
                CONTROLLER_INFO["run_id"],
                pid=os.getpid(),
            )
            active_runtime["heartbeat_at"] = now_iso()
            branch_progress = None
            if step_id == "c2_action_contract_repair" and BRANCH_PROGRESS_PATH.exists():
                try:
                    branch_progress = json.loads(BRANCH_PROGRESS_PATH.read_text())
                except json.JSONDecodeError:
                    branch_progress = None
                if branch_progress and branch_progress.get("active_branch"):
                    active_runtime["active_branch"] = branch_progress.get(
                        "active_branch"
                    )
                if branch_progress and branch_progress.get("active_lane"):
                    active_runtime["active_lane"] = branch_progress.get("active_lane")
            if step_id == "d1_single_rollout_overfit" and D1_PROGRESS_PATH.exists():
                try:
                    d1_progress = json.loads(D1_PROGRESS_PATH.read_text())
                except json.JSONDecodeError:
                    d1_progress = None
                if (
                    d1_progress
                    and d1_progress.get("candidate_key")
                    and d1_progress.get("variant_id")
                ):
                    active_runtime["active_branch"] = (
                        f"{d1_progress['candidate_key']}::{d1_progress['variant_id']}"
                    )
            exp = expected_artifacts(step_id)
            if exp:
                for candidate in exp:
                    if candidate.exists():
                        active_runtime["latest_artifact"] = str(candidate)
                        break
            state["active_runtime"] = active_runtime
            save_state(state)
            refresh_campaign_views(state, manifest, load_summary())
            update_watch_status(
                {
                    "controller_id": CONTROLLER_INFO["controller_id"],
                    "run_id": CONTROLLER_INFO["run_id"],
                    "supervisor_pid": None,
                    "inner_watcher_pid": None,
                    "worker_pid": proc.pid,
                    "worker_kind": "gate_subprocess",
                    "queue_step": step_id,
                    "stage": step_id,
                    "active_lane": active_runtime.get("active_lane"),
                    "active_branch": active_runtime.get("active_branch"),
                    "status": "running",
                    "last_heartbeat": active_runtime["heartbeat_at"],
                    "latest_artifact": active_runtime["latest_artifact"],
                    "last_successful_milestone": state.get("learnability_lane", {}).get(
                        "last_positive_step"
                    ),
                    "retry_count": state.get("retries", {}).get(step_id, 0),
                    "last_repair_action": state.get("last_repair_action"),
                    "last_error": state.get("last_error"),
                }
            )
            time.sleep(10)
    ok, message, payload = validate_step_artifacts(step_id)
    meta = {
        "step_id": step_id,
        "status": "completed" if ok else "failed",
        "worker_pid": proc.pid,
        "env": env_name,
        "returncode": proc.returncode,
        "validated": ok,
        "validation_message": message,
        "finished_at": now_iso(),
        "payload_excerpt": payload,
        "log_path": str(log_path),
    }
    write_runtime_meta(step_id, meta)
    log_loop_event(
        f"[finish] step={step_id} lane={active_runtime.get('active_lane')} ok={ok} returncode={proc.returncode} message={message}"
    )
    if not ok:
        return False, message

    summary = load_summary()
    summary.setdefault("artifacts", {})[step_id] = (
        str(expected_artifacts(step_id)[0])
        if expected_artifacts(step_id)
        else str(log_path)
    )
    summary["status"] = f"{step_id}_completed"
    if step_id == "e1_heldout_eval":
        eval_summary = json.loads(
            (CAMPAIGN_DIR / "evaluation" / "comparison_summary.json").read_text()
        )
        summary["evaluation"] = eval_summary
        summary["strongest_true_claim"] = eval_summary.get("strongest_true_claim")
    save_summary(summary)
    return True, "ok"


def run_write_claim_memo(state: dict, manifest: dict) -> tuple[bool, str]:
    summary = load_summary()
    write_text_atomic(
        CAMPAIGN_DIR / "decision_memo.md", load_decision_memo(summary, state, manifest)
    )
    return True, "ok"


def execute_step(step_id: str, state: dict, manifest: dict) -> tuple[bool, str]:
    env_name = QUEUE_ENVS.get(step_id, "mint")
    if step_id == "write_claim_memo":
        return run_write_claim_memo(state, manifest)
    return run_script_step(step_id, env_name, state, manifest)


def finalize_terminal_state(state: dict, summary: dict) -> None:
    verdict = summarize_verdict(state, summary)
    if verdict == "claim_supported":
        state["verdict"] = "claim_supported"
        state["phase_gate"] = "ready_for_writeup"
    elif verdict == "scientific_not_supported":
        state["verdict"] = "scientific_not_supported"
        state["phase_gate"] = "ready_for_writeup"
    save_state(state)


def run_once() -> bool:
    ensure_contract_files()
    manifest = load_manifest()
    state = load_state()
    summary = load_summary()

    if not state.get("queue"):
        state = default_state(manifest)
        save_state(state)
    elif sync_completed_steps_from_artifacts(state):
        save_state(state)
    lease = refresh_controller_lease(
        CONTROLLER_INFO["controller_id"],
        CONTROLLER_INFO["run_id"],
        pid=os.getpid(),
    )
    prior_controller = state.get("controller") or {}
    state["controller"] = {
        "controller_id": CONTROLLER_INFO["controller_id"],
        "run_id": CONTROLLER_INFO["run_id"],
        "owner": CONTROLLER_INFO["owner"],
        "lease_path": str(
            lease and lease.get("lease_path") or (RUNTIME_DIR / "controller_lease.json")
        ),
        "status": "active",
        "acquired_at": prior_controller.get("acquired_at")
        or lease.get("acquired_at")
        or now_iso(),
        "heartbeat_at": lease.get("heartbeat_at") or now_iso(),
    }
    sync_lane_state_from_artifacts(state)
    resumable_step = active_or_next_step(state)
    if state.get("verdict") == "implementation_blocked" and resumable_step is not None:
        resumable_status = queue_entry(state, resumable_step).get("status")
        if resumable_status in {"pending", "running"}:
            state["verdict"] = None
            if (
                resumable_step.startswith("d")
                or resumable_step.startswith("e")
                or resumable_step == "write_claim_memo"
            ):
                state["phase_gate"] = "await_train_reproducibility"
            append_history(
                state,
                "auto_resume",
                f"Cleared stale implementation_blocked verdict and resumed from {resumable_step}.",
            )
    save_state(state)

    review, summary = refresh_campaign_views(state, manifest, summary)
    current_step = active_or_next_step(state)
    if current_step is None:
        finalize_terminal_state(state, summary)
        summary = load_summary()
        state = load_state()
        refresh_campaign_views(state, manifest, summary)
        if summarize_verdict(state, summary):
            run_write_claim_memo(state, manifest)
            log_loop_event(f"[terminal] verdict={state.get('verdict')}")
        return True

    item = queue_entry(state, current_step)
    if item.get("status") == "running":
        runtime = state.get("active_runtime") or {}
        worker_pid = runtime.get("worker_pid")
        if process_alive(int(worker_pid)) if worker_pid else False:
            review, summary = refresh_campaign_views(state, manifest, summary)
            update_watch_status(
                {
                    "supervisor_pid": None,
                    "inner_watcher_pid": None,
                    "worker_pid": worker_pid,
                    "worker_kind": runtime.get("worker_kind"),
                    "queue_step": current_step,
                    "stage": current_step,
                    "active_lane": runtime.get("active_lane"),
                    "active_branch": runtime.get("active_branch"),
                    "status": "running",
                    "last_heartbeat": runtime.get("heartbeat_at") or now_iso(),
                    "latest_artifact": runtime.get("latest_artifact"),
                    "last_successful_milestone": state.get("learnability_lane", {}).get(
                        "last_positive_step"
                    ),
                    "retry_count": state.get("retries", {}).get(current_step, 0),
                    "last_repair_action": state.get("last_repair_action"),
                    "last_error": state.get("last_error"),
                }
            )
            return True

        ok, message, payload = validate_step_artifacts(current_step)
        state["active_runtime"] = None
        if ok:
            set_queue_status(state, current_step, "completed")
            append_history(
                state, current_step, "validated artifact written after loop reattach"
            )
            summary.setdefault("artifacts", {})[current_step] = (
                str(expected_artifacts(current_step)[0])
                if expected_artifacts(current_step)
                else str(step_log_path(current_step))
            )
            if current_step == "e1_heldout_eval":
                eval_summary = json.loads(
                    (
                        CAMPAIGN_DIR / "evaluation" / "comparison_summary.json"
                    ).read_text()
                )
                summary["evaluation"] = eval_summary
                summary["strongest_true_claim"] = eval_summary.get(
                    "strongest_true_claim"
                )
            save_summary(summary)
            save_state(state)
            refresh_campaign_views(state, manifest, summary)
            run_memory_sync(
                title=f"mint_drawer_v1 {current_step} completed",
                summary=f"Completed {current_step} after loop reattach",
                completed=[current_step],
                next_actions=[f"Advance to the next queue step after {current_step}."],
                artifacts={
                    current_step: summary.get("artifacts", {}).get(current_step, "")
                },
                lessons=[f"{current_step} remained recoverable across loop restarts."],
            )
            return True

        attempts = increment_retry(state, current_step)
        set_queue_status(state, current_step, "failed", error=message)
        state["last_error"] = message
        state["last_repair_action"] = f"{current_step}:reattach_retry_{attempts}"
        append_history(
            state,
            "auto_repair",
            f"{current_step} failed after loop reattach: {message}",
        )
        if _d1_exhausted(current_step, message):
            _block_on_d1_exhaustion(state, summary, message)
        elif attempts <= MAX_RETRIES:
            reset_step(state, current_step)
        else:
            state["verdict"] = "implementation_blocked"
            summary["last_error"] = message
            save_summary(summary)
        save_state(state)
        refresh_campaign_views(state, manifest, summary)
        update_watch_status(
            {
                "supervisor_pid": None,
                "inner_watcher_pid": None,
                "worker_pid": None,
                "worker_kind": None,
                "queue_step": current_step,
                "stage": current_step,
                "active_lane": step_lane(current_step, state),
                "active_branch": None,
                "status": queue_entry(load_state(), current_step).get("status"),
                "last_heartbeat": now_iso(),
                "latest_artifact": None,
                "last_successful_milestone": load_state()
                .get("learnability_lane", {})
                .get("last_positive_step"),
                "retry_count": load_state().get("retries", {}).get(current_step, 0),
                "last_repair_action": load_state().get("last_repair_action"),
                "last_error": load_state().get("last_error"),
            }
        )
        return False

    if (
        item.get("status") == "failed"
        and state.get("retries", {}).get(current_step, 0) >= MAX_RETRIES
    ):
        state["verdict"] = "implementation_blocked"
        state["last_error"] = item.get("last_error")
        summary["last_error"] = item.get("last_error")
        save_summary(summary)
        save_state(state)
        refresh_campaign_views(state, manifest, summary)
        run_write_claim_memo(state, manifest)
        log_loop_event(f"[blocked] step={current_step} error={item.get('last_error')}")
        return False

    set_queue_status(state, current_step, "running")
    state["last_error"] = None
    state["last_repair_action"] = None
    save_state(state)
    log_loop_event(f"[queue] step={current_step} lane={step_lane(current_step, state)}")
    refresh_campaign_views(state, manifest, summary)

    ok, message = execute_step(current_step, state, manifest)
    summary = load_summary()
    state = load_state()
    state["active_runtime"] = None
    sync_lane_state_from_artifacts(state)
    if ok:
        set_queue_status(state, current_step, "completed")
        append_history(state, current_step, "validated artifact written")
        save_state(state)
        run_memory_sync(
            title=f"mint_drawer_v1 {current_step} completed",
            summary=f"Completed {current_step}",
            completed=[current_step],
            next_actions=[f"Advance to the next queue step after {current_step}."],
            artifacts={
                current_step: summary.get("artifacts", {}).get(current_step, "")
            },
            lessons=[f"{current_step} now uses artifact-first validation."],
        )
    else:
        if current_step in {
            "d3_train_seed_probe",
            "e1_heldout_eval",
        } and _expand_coverage_and_retry(state, summary, current_step, message):
            state = load_state()
            summary = load_summary()
            sync_lane_state_from_artifacts(state)
            refresh_campaign_views(state, manifest, summary)
            update_watch_status(
                {
                    "supervisor_pid": None,
                    "inner_watcher_pid": None,
                    "worker_pid": None,
                    "worker_kind": None,
                    "queue_step": current_step,
                    "stage": current_step,
                    "active_lane": step_lane(current_step, state),
                    "active_branch": None,
                    "status": "auto_repair",
                    "last_heartbeat": now_iso(),
                    "latest_artifact": None,
                    "last_successful_milestone": state.get("learnability_lane", {}).get(
                        "last_positive_step"
                    ),
                    "retry_count": state.get("retries", {}).get(current_step, 0),
                    "last_repair_action": state.get("last_repair_action"),
                    "last_error": state.get("last_error"),
                }
            )
            log_loop_event(
                f"[auto_repair] step={current_step} reason=coverage_expansion"
            )
            return True
        attempts = increment_retry(state, current_step)
        set_queue_status(state, current_step, "failed", error=message)
        state["last_error"] = message
        state["last_repair_action"] = f"{current_step}:retry_{attempts}"
        append_history(state, "auto_repair", f"{current_step} failed: {message}")
        if _d1_exhausted(current_step, message):
            _block_on_d1_exhaustion(state, summary, message)
        elif attempts <= MAX_RETRIES:
            reset_step(state, current_step)
        else:
            state["verdict"] = "implementation_blocked"
            summary["last_error"] = message
            save_summary(summary)
        save_state(state)
        log_loop_event(
            f"[retry] step={current_step} attempt={attempts} error={message}"
        )

    state = load_state()
    summary = load_summary()
    if current_step == "write_claim_memo" or summarize_verdict(state, summary):
        finalize_terminal_state(state, summary)
    summary = load_summary()
    state = load_state()
    sync_lane_state_from_artifacts(state)
    refresh_campaign_views(state, manifest, summary)
    if summarize_verdict(state, summary):
        run_write_claim_memo(state, manifest)
    update_watch_status(
        {
            "supervisor_pid": None,
            "inner_watcher_pid": None,
            "worker_pid": None,
            "worker_kind": None,
            "queue_step": current_step,
            "stage": current_step,
            "active_lane": step_lane(current_step, state),
            "active_branch": None,
            "status": load_state().get("verdict")
            or queue_entry(load_state(), current_step).get("status"),
            "last_heartbeat": now_iso(),
            "last_successful_milestone": load_state()
            .get("learnability_lane", {})
            .get("last_positive_step"),
            "latest_artifact": load_state()
            .get("active_runtime", {})
            .get("latest_artifact")
            if load_state().get("active_runtime")
            else None,
            "retry_count": load_state().get("retries", {}).get(current_step, 0),
            "last_repair_action": load_state().get("last_repair_action"),
            "last_error": load_state().get("last_error"),
        }
    )
    return ok


def main() -> int:
    args = parse_args()
    acquired, lease = acquire_controller_lease(
        CONTROLLER_INFO["controller_id"],
        CONTROLLER_INFO["run_id"],
        owner=CONTROLLER_INFO["owner"],
        pid=os.getpid(),
    )
    if not acquired:
        log_loop_event(
            f"[lease_refused] owner={lease.get('owner')} controller_id={lease.get('controller_id')} run_id={lease.get('run_id')} pid={lease.get('pid')}"
        )
        return 2
    try:
        if args.run_until_terminal:
            while True:
                state = load_state() if (CAMPAIGN_DIR / "state.json").exists() else {}
                if state.get("verdict") in {
                    "implementation_blocked",
                    "scientific_not_supported",
                    "claim_supported",
                }:
                    log_loop_event(f"[exit] terminal_verdict={state.get('verdict')}")
                    return 0
                run_once()
                time.sleep(3)
        else:
            run_once()
        return 0
    finally:
        release_controller_lease(
            CONTROLLER_INFO["controller_id"], CONTROLLER_INFO["run_id"]
        )


if __name__ == "__main__":
    raise SystemExit(main())

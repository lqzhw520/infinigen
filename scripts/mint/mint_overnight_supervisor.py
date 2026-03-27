#!/usr/bin/env python3
"""Supervisor for the MINT drawer auto-review loop."""

from __future__ import annotations

import json
import os
import subprocess
import time

from mint_common import (
    CAMPAIGN_DIR,
    CONDA_SETUP,
    PROJECT_ROOT,
    load_state,
    now_iso,
    update_watch_status,
    write_json_atomic,
)

SCRIPTS = PROJECT_ROOT / "scripts" / "mint"
RUNTIME_DIR = CAMPAIGN_DIR / "runtime"
STATUS_PATH = RUNTIME_DIR / "supervisor_status.json"


def run_loop_process() -> subprocess.Popen:
    cmd = f"{CONDA_SETUP} && conda activate mint && cd {PROJECT_ROOT} && python {SCRIPTS / 'mint_auto_review_loop.py'} --run-until-terminal"
    return subprocess.Popen(
        ["bash", "-lc", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main() -> int:
    max_restarts = 6
    restart_count = 0
    log_path = RUNTIME_DIR / "supervisor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    while restart_count <= max_restarts:
        proc = run_loop_process()
        with log_path.open("a") as log_handle:
            log_handle.write(
                f"[{now_iso()}] started loop pid={proc.pid} restart={restart_count}\n"
            )
        while proc.poll() is None:
            state = load_state()
            payload = {
                "supervisor_pid": os.getpid(),
                "inner_watcher_pid": proc.pid,
                "restart_count": restart_count,
                "campaign_verdict": state.get("verdict"),
                "active_step": (state.get("active_runtime") or {}).get("step_id"),
                "timestamp": now_iso(),
            }
            write_json_atomic(STATUS_PATH, payload)
            update_watch_status(
                {
                    "supervisor_pid": os.getpid(),
                    "inner_watcher_pid": proc.pid,
                    "worker_pid": (state.get("active_runtime") or {}).get("worker_pid"),
                    "worker_kind": (state.get("active_runtime") or {}).get(
                        "worker_kind"
                    ),
                    "queue_step": (state.get("active_runtime") or {}).get("step_id"),
                    "stage": (state.get("active_runtime") or {}).get("step_id"),
                    "status": state.get("verdict")
                    or (
                        (state.get("active_runtime") or {}).get("step_id") and "running"
                    )
                    or "idle",
                    "last_heartbeat": now_iso(),
                    "latest_artifact": (state.get("active_runtime") or {}).get(
                        "latest_artifact"
                    ),
                    "last_repair_action": state.get("last_repair_action"),
                    "last_error": state.get("last_error"),
                }
            )
            time.sleep(30)

        state = load_state()
        if state.get("verdict") in {
            "implementation_blocked",
            "scientific_not_supported",
            "claim_supported",
        }:
            write_json_atomic(
                STATUS_PATH,
                {
                    "supervisor_pid": os.getpid(),
                    "inner_watcher_pid": proc.pid,
                    "restart_count": restart_count,
                    "campaign_verdict": state.get("verdict"),
                    "timestamp": now_iso(),
                },
            )
            return 0

        restart_count += 1
        with log_path.open("a") as log_handle:
            log_handle.write(
                f"[{now_iso()}] loop exited rc={proc.returncode}; restarting {restart_count}/{max_restarts}\n"
            )
        time.sleep(10)

    state = load_state()
    state["verdict"] = "implementation_blocked"
    state["last_error"] = "supervisor_restart_budget_exhausted"
    (CAMPAIGN_DIR / "state.json").write_text(json.dumps(state, indent=2))
    write_json_atomic(
        STATUS_PATH,
        {
            "supervisor_pid": os.getpid(),
            "restart_count": restart_count,
            "campaign_verdict": "implementation_blocked",
            "timestamp": now_iso(),
        },
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

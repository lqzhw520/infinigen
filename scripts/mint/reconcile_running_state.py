#!/usr/bin/env python3
"""Reconcile the MINT campaign state to the currently running direct gate."""

from __future__ import annotations

import argparse

from mint_common import (
    expected_artifacts,
    load_manifest,
    load_state,
    load_summary,
    now_iso,
    refresh_campaign_views,
    save_state,
    save_summary,
    update_watch_status,
    validate_step_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-step", required=True)
    parser.add_argument("--worker-pid", required=True, type=int)
    parser.add_argument("--attempt-no", type=int, default=1)
    parser.add_argument(
        "--completed",
        nargs="*",
        default=[],
        help="Queue steps that should be validated and marked completed if their artifacts pass.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest()
    state = load_state()
    summary = load_summary()

    for item in state["queue"]:
        sid = item["id"]
        if sid in args.completed:
            ok, _, _ = validate_step_artifacts(sid)
            if ok:
                item["status"] = "completed"
                item["updated_at"] = now_iso()
                artifacts = expected_artifacts(sid)
                summary.setdefault("artifacts", {})[sid] = (
                    str(artifacts[0]) if artifacts else ""
                )
        elif sid == args.active_step:
            item["status"] = "running"
            item["updated_at"] = now_iso()
        elif sid not in args.completed and sid != args.active_step:
            if item["status"] not in {"completed", "pending"}:
                item["status"] = "pending"
                item["updated_at"] = now_iso()

    previous_runtime = state.get("active_runtime") or {}
    state["active_runtime"] = {
        "step_id": args.active_step,
        "worker_kind": "gate_subprocess",
        "worker_pid": args.worker_pid,
        "attempt_no": args.attempt_no,
        "launched_at": previous_runtime.get("launched_at") or now_iso(),
        "heartbeat_at": now_iso(),
        "latest_artifact": None,
        "last_error": None,
    }
    state["last_error"] = None
    state["last_repair_action"] = None
    state["verdict"] = None
    state["phase_gate"] = "await_g9_eval"
    summary["status"] = f"{args.active_step}_running"

    save_summary(summary)
    save_state(state)
    refresh_campaign_views(state, manifest, summary)
    update_watch_status(
        {
            "supervisor_pid": None,
            "inner_watcher_pid": None,
            "worker_pid": args.worker_pid,
            "worker_kind": "gate_subprocess",
            "queue_step": args.active_step,
            "stage": args.active_step,
            "status": "running",
            "last_heartbeat": now_iso(),
            "latest_artifact": None,
            "last_repair_action": None,
            "last_error": None,
        }
    )
    print("reconciled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

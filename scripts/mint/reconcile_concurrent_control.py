#!/usr/bin/env python3
"""Mark concurrent-control contamination and reset the campaign to a clean rerun point."""

from __future__ import annotations

import argparse
import json

from mint_common import (
    ARTIFACT_DIR,
    append_history,
    append_invalidated_result,
    ensure_contract_files,
    load_manifest,
    load_state,
    load_summary,
    now_iso,
    queue_entry,
    refresh_campaign_views,
    save_state,
    save_summary,
    set_queue_status,
    update_watch_status,
    write_json_atomic,
)

FORENSIC_ARTIFACT = ARTIFACT_DIR / "concurrent_control_reconcile.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="d1_single_rollout_overfit")
    parser.add_argument("--since", default="2026-03-26T18:26:00+08:00")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_contract_files()
    manifest = load_manifest()
    state = load_state()
    summary = load_summary()

    contamination = append_invalidated_result(
        state,
        step_id=args.step,
        reason="contaminated_by_concurrent_control",
        details={
            "since": args.since,
            "note": "Concurrent supervisor/D1/train processes detected; do not treat post-cutoff D1 artifacts as clean evidence.",
        },
    )
    try:
        set_queue_status(
            state, args.step, "pending", error="contaminated_by_concurrent_control"
        )
    except KeyError:
        pass
    state["active_runtime"] = None
    state["verdict"] = None
    state["phase_gate"] = "await_train_reproducibility"
    state["last_error"] = "contaminated_by_concurrent_control"
    state["last_repair_action"] = "forensic_reconcile_concurrent_control"
    append_history(
        state,
        "forensic_reconcile",
        f"Reset {args.step} to pending after concurrent-control contamination detected for runs after {args.since}.",
    )
    save_state(state)
    summary["last_error"] = "contaminated_by_concurrent_control"
    summary["blocked_reason"] = None
    save_summary(summary)
    review, summary = refresh_campaign_views(state, manifest, summary)
    payload = {
        "step": args.step,
        "since": args.since,
        "contamination": contamination,
        "queue_status": queue_entry(state, args.step).get("status"),
        "controller": state.get("controller"),
        "recorded_at": now_iso(),
        "review_verdict": review.get("verdict"),
    }
    write_json_atomic(FORENSIC_ARTIFACT, payload)
    update_watch_status(
        {
            "worker_pid": None,
            "worker_kind": "manual_reconcile",
            "queue_step": args.step,
            "stage": "forensic_reconcile",
            "active_lane": None,
            "active_branch": None,
            "status": "paused",
            "latest_artifact": str(FORENSIC_ARTIFACT),
            "last_repair_action": state.get("last_repair_action"),
            "last_error": state.get("last_error"),
            "verdict": review.get("verdict"),
        }
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

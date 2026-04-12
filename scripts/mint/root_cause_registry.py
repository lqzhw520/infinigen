#!/usr/bin/env python3
"""Append-only registry helpers for the visual-fidelity root-cause controller."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from mint_common import ensure_dirs, now_iso, write_json_atomic, write_text_atomic
from root_cause_contracts import (
    AUTOPILOT_DIR,
    CONTROLLER_RUNTIME_DIR,
    CYCLE_STATE_PATH,
    DEVIATION_LOG_PATH,
    EVIDENCE_LEDGER_PATH,
    EXPERIMENT_REGISTRY_PATH,
    HYPOTHESIS_BOARD_PATH,
)

MORNING_MEMO_PATH = AUTOPILOT_DIR / "morning_memo.md"
FINAL_ROUTE_PATH = AUTOPILOT_DIR / "route_decision.json"
FINAL_RUN_SUMMARY_PATH = AUTOPILOT_DIR / "final_run_summary.json"


def _append_jsonl_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if path.exists():
        lines = path.read_text().splitlines()
    lines.append(json.dumps(payload, ensure_ascii=False))
    write_text_atomic(path, "\n".join(lines).rstrip() + "\n")


def ensure_registry_layout() -> None:
    ensure_dirs()
    AUTOPILOT_DIR.mkdir(parents=True, exist_ok=True)
    CONTROLLER_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEVIATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def create_cycle_dir(cycle_id: str, *, cycle_mode: str = "unattended_cycle") -> Path:
    ensure_registry_layout()
    cycle_dir = CONTROLLER_RUNTIME_DIR / cycle_id
    cycle_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(cycle_dir / "cycle_mode.json", {"cycle_id": cycle_id, "cycle_mode": cycle_mode, "created_at": now_iso()})
    return cycle_dir


def append_cycle_record(payload: dict[str, Any]) -> None:
    _append_jsonl_atomic(EXPERIMENT_REGISTRY_PATH, payload)


def append_evidence_record(payload: dict[str, Any]) -> None:
    _append_jsonl_atomic(EVIDENCE_LEDGER_PATH, payload)


def append_deviation_record(*, message: str, scientific_semantics_changed: bool = False, details: dict[str, Any] | None = None) -> None:
    _append_jsonl_atomic(
        DEVIATION_LOG_PATH,
        {
            "recorded_at": now_iso(),
            "message": message,
            "scientific_semantics_changed": bool(scientific_semantics_changed),
            "details": details or {},
        },
    )


def write_cycle_state(payload: dict[str, Any]) -> None:
    write_json_atomic(CYCLE_STATE_PATH, payload)


def write_hypothesis_board(payload: dict[str, Any]) -> None:
    write_json_atomic(HYPOTHESIS_BOARD_PATH, payload)


def archive_controller_state(*, reason: str) -> str:
    ensure_registry_layout()
    archive_dir = AUTOPILOT_DIR / "resets" / now_iso().replace(":", "").replace("+", "_")
    archive_dir.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for path in (CYCLE_STATE_PATH, HYPOTHESIS_BOARD_PATH):
        if path.exists():
            target = archive_dir / path.name
            shutil.move(str(path), str(target))
            moved.append(path.name)
    summary = {
        "archived_at": now_iso(),
        "reason": reason,
        "archive_dir": str(archive_dir),
        "moved": moved,
    }
    write_json_atomic(archive_dir / "reset_summary.json", summary)
    return str(archive_dir)


def write_cycle_bundle(
    cycle_dir: Path,
    *,
    cycle_mode: str,
    lane_specs: dict[str, Any],
    lane_results: dict[str, Any],
    gate_report: dict[str, Any],
    hypothesis_board: dict[str, Any],
    cycle_memo: str,
    proposed_current_truth_delta: dict[str, Any],
    proposed_next_actions: dict[str, Any],
    cycle_summary: dict[str, Any],
    route_decision: dict[str, Any],
) -> None:
    write_json_atomic(cycle_dir / "lane_specs.json", lane_specs)
    write_json_atomic(cycle_dir / "lane_results.json", lane_results)
    write_json_atomic(cycle_dir / "gate_report.json", gate_report)
    write_json_atomic(cycle_dir / "hypothesis_board.json", hypothesis_board)
    write_json_atomic(cycle_dir / "cycle_summary.json", cycle_summary)
    write_json_atomic(cycle_dir / "route_decision.json", route_decision)
    write_json_atomic(cycle_dir / "cycle_mode.json", {"cycle_id": cycle_dir.name, "cycle_mode": cycle_mode, "updated_at": now_iso()})
    write_text_atomic(cycle_dir / "cycle_memo.md", cycle_memo)
    write_json_atomic(cycle_dir / "proposed_current_truth_delta.json", proposed_current_truth_delta)
    write_json_atomic(cycle_dir / "proposed_next_actions.json", proposed_next_actions)
    write_cycle_state(
        {
            "updated_at": now_iso(),
            "last_cycle_id": cycle_dir.name,
            "cycle_mode": cycle_mode,
            "completed_experiments": cycle_summary.get("completed_experiments", []),
            "scientific_terminal_state": route_decision.get("scientific_terminal_state"),
            "route_next_branch": route_decision.get("route_next_branch"),
            "terminal_state": route_decision.get("scientific_terminal_state"),
            "strongest_negative_capped": route_decision.get("strongest_negative_capped", True),
        }
    )


def write_final_run_outputs(*, route_decision: dict[str, Any], morning_memo: str, final_summary: dict[str, Any]) -> None:
    ensure_registry_layout()
    write_json_atomic(FINAL_ROUTE_PATH, route_decision)
    write_text_atomic(MORNING_MEMO_PATH, morning_memo)
    write_json_atomic(FINAL_RUN_SUMMARY_PATH, final_summary)

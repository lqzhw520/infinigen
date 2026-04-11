#!/usr/bin/env python3
"""Append-only registry helpers for the v2 root-cause controller."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mint_common import ensure_dirs, now_iso, write_json_atomic, write_text_atomic
from root_cause_contracts import (
    AUTOPILOT_DIR,
    CONTROLLER_RUNTIME_DIR,
    CYCLE_STATE_PATH,
    EVIDENCE_LEDGER_PATH,
    EXPERIMENT_REGISTRY_PATH,
    HYPOTHESIS_BOARD_PATH,
)


def _append_jsonl_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
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


def create_cycle_dir(cycle_id: str) -> Path:
    ensure_registry_layout()
    cycle_dir = CONTROLLER_RUNTIME_DIR / cycle_id
    cycle_dir.mkdir(parents=True, exist_ok=True)
    return cycle_dir


def append_cycle_record(payload: dict[str, Any]) -> None:
    _append_jsonl_atomic(EXPERIMENT_REGISTRY_PATH, payload)


def append_evidence_record(payload: dict[str, Any]) -> None:
    _append_jsonl_atomic(EVIDENCE_LEDGER_PATH, payload)


def write_cycle_state(payload: dict[str, Any]) -> None:
    write_json_atomic(CYCLE_STATE_PATH, payload)


def write_hypothesis_board(payload: dict[str, Any]) -> None:
    write_json_atomic(HYPOTHESIS_BOARD_PATH, payload)


def write_cycle_bundle(
    cycle_dir: Path,
    *,
    lane_specs: dict[str, Any],
    lane_results: dict[str, Any],
    gate_report: dict[str, Any],
    hypothesis_board: dict[str, Any],
    cycle_memo: str,
    proposed_current_truth_delta: dict[str, Any],
    proposed_next_actions: dict[str, Any],
) -> None:
    write_json_atomic(cycle_dir / "lane_specs.json", lane_specs)
    write_json_atomic(cycle_dir / "lane_results.json", lane_results)
    write_json_atomic(cycle_dir / "gate_report.json", gate_report)
    write_json_atomic(cycle_dir / "hypothesis_board.json", hypothesis_board)
    write_text_atomic(cycle_dir / "cycle_memo.md", cycle_memo)
    write_json_atomic(cycle_dir / "proposed_current_truth_delta.json", proposed_current_truth_delta)
    write_json_atomic(cycle_dir / "proposed_next_actions.json", proposed_next_actions)
    write_cycle_state({
        "updated_at": now_iso(),
        "last_cycle_id": cycle_dir.name,
        "terminal_state": gate_report.get("terminal_state"),
    })

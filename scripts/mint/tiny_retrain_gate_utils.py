#!/usr/bin/env python3
"""Shared gate and publication helpers for v9 diagnostic tiny retrain."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mint_common import CAMPAIGN_DIR, PROJECT_ROOT, load_json, write_json_atomic

AUTOPILOT_DIR = CAMPAIGN_DIR / "autopilot"
GATES_DIR = AUTOPILOT_DIR / "gates"
DOCS_LOCK_MANIFEST_PATH = AUTOPILOT_DIR / "docs_lock_manifest.json"
SOVEREIGN_SNAPSHOT_PATH = AUTOPILOT_DIR / "sovereign_snapshot.json"
HARNESS_STATE_PATH = AUTOPILOT_DIR / "harness_state.json"
PUBLICATION_STATE_PATH = AUTOPILOT_DIR / "publication_state.json"
DOCS_UPDATE_INTENT_PATH = AUTOPILOT_DIR / "docs_update_intent.json"

SPEC_DOC_PATH = PROJECT_ROOT / "docs" / "MINT_V84_DIAGNOSTIC_TINY_RETRAIN_GATE_EXECUTION_SPEC.md"
UNIFIED_DOC_PATH = PROJECT_ROOT / "docs" / "MINT_V84_UNIFIED_EXECUTION_SPEC.md"
SYSTEM_AUDIT_DOC_PATH = PROJECT_ROOT / "docs" / "MINT_V84_SYSTEM_AUDIT_2026-04-16.md"
TRUTH_CONTRACT_PATH = PROJECT_ROOT / "docs" / "contracts" / "truth_contract_v84.json"
ACCEPTANCE_CONTRACT_PATH = PROJECT_ROOT / "docs" / "contracts" / "acceptance_contract_v84.json"

GATE_STATUSES = {"PASS", "DIAGNOSTIC_PASS", "AUTHORITATIVE_PASS", "STOP"}


def _git(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=PROJECT_ROOT, text=True).strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def current_repo_identity() -> dict[str, str]:
    branch = _git(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    head = _git(["git", "rev-parse", "HEAD"])
    vendor = _git(["git", "rev-parse", "HEAD:external/MINT"])
    return {
        "branch": branch,
        "working_head_commit": head,
        "vendor_head_commit": vendor,
    }


def docs_lock_payload() -> dict[str, Any]:
    docs = [SPEC_DOC_PATH, UNIFIED_DOC_PATH, SYSTEM_AUDIT_DOC_PATH]
    return {
        "lock_version": "v9_docs_lock_v1",
        "timestamp_utc": utc_now(),
        "docs": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in docs
        ],
    }


def write_docs_lock_manifest() -> dict[str, Any]:
    payload = docs_lock_payload()
    DOCS_LOCK_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(DOCS_LOCK_MANIFEST_PATH, payload)
    return payload


def docs_lock_consistent() -> tuple[bool, list[str], dict[str, Any]]:
    manifest = load_json(DOCS_LOCK_MANIFEST_PATH, {})
    current = docs_lock_payload()
    locked = {item["path"]: item["sha256"] for item in manifest.get("docs", [])}
    drift = []
    for item in current.get("docs", []):
        if locked.get(item["path"]) != item["sha256"]:
            drift.append(item["path"])
    return (not drift) and bool(manifest), drift, manifest


def write_sovereign_snapshot(plan: dict[str, Any]) -> dict[str, Any]:
    ident = current_repo_identity()
    payload = {
        "snapshot_version": "v9_sovereign_snapshot_v1",
        "run_instance_id": plan.get("run_instance_id"),
        "branch": ident["branch"],
        "working_head_commit": ident["working_head_commit"],
        "vendor_head_commit": ident["vendor_head_commit"],
        "truth_contract_hash": sha256_file(TRUTH_CONTRACT_PATH),
        "acceptance_contract_hash": sha256_file(ACCEPTANCE_CONTRACT_PATH),
        "spec_doc_path": str(SPEC_DOC_PATH),
        "spec_doc_hash": sha256_file(SPEC_DOC_PATH),
        "timestamp_utc": utc_now(),
    }
    SOVEREIGN_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(SOVEREIGN_SNAPSHOT_PATH, payload)
    return payload


def write_harness_state(payload: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "state_version": "v9_harness_state_v1",
        "timestamp_utc": utc_now(),
        **payload,
    }
    HARNESS_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(HARNESS_STATE_PATH, merged)
    return merged


def write_publication_state(payload: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "state_version": "v9_publication_state_v1",
        "timestamp_utc": utc_now(),
        **payload,
    }
    PUBLICATION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(PUBLICATION_STATE_PATH, merged)
    return merged


def write_docs_update_intent(payload: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "intent_version": "v9_docs_update_intent_v1",
        "timestamp_utc": utc_now(),
        **payload,
    }
    DOCS_UPDATE_INTENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(DOCS_UPDATE_INTENT_PATH, merged)
    return merged


def gate_base(plan: dict[str, Any]) -> dict[str, Any]:
    ident = current_repo_identity()
    return {
        "run_instance_id": plan.get("run_instance_id"),
        "working_head_commit": ident["working_head_commit"],
        "vendor_head_commit": ident["vendor_head_commit"],
        "branch": ident["branch"],
        "truth_contract_hash": sha256_file(TRUTH_CONTRACT_PATH),
        "acceptance_contract_hash": sha256_file(ACCEPTANCE_CONTRACT_PATH),
        "spec_doc_path": str(SPEC_DOC_PATH),
        "spec_doc_hash": sha256_file(SPEC_DOC_PATH),
    }


def write_gate(
    gate_id: str,
    gate_name: str,
    plan: dict[str, Any],
    *,
    status: str,
    blocking_reasons: list[str] | None = None,
    allowed_next_phases: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in GATE_STATUSES:
        raise ValueError(f"Unsupported gate status: {status}")
    payload = {
        "gate_id": gate_id,
        "gate_name": gate_name,
        **gate_base(plan),
        "status": status,
        "blocking_reasons": list(blocking_reasons or []),
        "allowed_next_phases": list(allowed_next_phases or []),
        "timestamp_utc": utc_now(),
    }
    if extra:
        payload.update(extra)
    GATES_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(GATES_DIR / f"{gate_id}_{gate_name}.json", payload)
    return payload


def readiness_failed_clauses(readiness_report: dict[str, Any]) -> list[str]:
    if readiness_report.get("failed_clauses"):
        return [str(x) for x in readiness_report.get("failed_clauses", [])]
    clauses = dict(readiness_report.get("clauses") or {})
    return [str(name) for name, passed in clauses.items() if not bool(passed)]


def gate_scope_from_g4(status: str) -> dict[str, Any]:
    if status == "AUTHORITATIVE_PASS":
        return {
            "execution_scope": "v9_gate_controlled_authoritative",
            "diagnostic_only": False,
            "claim_bearing": True,
            "publication_scope": "authoritative",
            "result_scope": "authoritative",
        }
    if status == "DIAGNOSTIC_PASS":
        return {
            "execution_scope": "v9_gate_controlled_diagnostic",
            "diagnostic_only": True,
            "claim_bearing": False,
            "publication_scope": "diagnostic_only",
            "result_scope": "diagnostic_only",
        }
    return {
        "execution_scope": "v9_gate_controlled_stop",
        "diagnostic_only": True,
        "claim_bearing": False,
        "publication_scope": "stopped",
        "result_scope": "stopped",
    }


def load_gate(gate_id: str, gate_name: str) -> dict[str, Any]:
    return load_json(GATES_DIR / f"{gate_id}_{gate_name}.json", {})

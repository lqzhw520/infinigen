#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_DIR = PROJECT_ROOT / "experiments" / "mint" / "mint_drawer_v1"
ARTIFACT_DIR = CAMPAIGN_DIR / "artifacts"
AUTOPILOT_DIR = CAMPAIGN_DIR / "autopilot"
GATES_V12_DIR = AUTOPILOT_DIR / "gates_v12"
GATES_V13_DIR = AUTOPILOT_DIR / "gates_v13"
GATES_V13_SLICE2_DIR = AUTOPILOT_DIR / "gates_v13_slice2"
PLAN_PATH = ARTIFACT_DIR / "active_tiny_retrain_plan.json"
TRUTH_CONTRACT_PATH = PROJECT_ROOT / "docs" / "contracts" / "truth_contract_v84.json"
ACCEPTANCE_CONTRACT_PATH = PROJECT_ROOT / "docs" / "contracts" / "acceptance_contract_v84.json"
V13_SPEC_REFERENCE = "/Users/zhuhaowu/ws/phd-anyboxs/infinigen_from_servers/docs/gpt5.4Pro/codex_mint_v84_v13_state_action_interface_execution_spec.md"
V12_SPEC_REFERENCE = "/Users/zhuhaowu/ws/phd-anyboxs/infinigen_from_servers/docs/gpt5.4Pro/codex_mint_v84_v12_1_orientation_attach_anygrasp_execution_spec.md"


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text())


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=PROJECT_ROOT, text=True).strip()


def current_repo_identity() -> dict[str, str]:
    return {
        "branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "working_head_commit": _git(["git", "rev-parse", "HEAD"]),
        "vendor_head_commit": _git(["git", "rev-parse", "HEAD:external/MINT"]),
    }


def make_run_instance(prefix: str, head: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = hashlib.sha1(f"{prefix}:{head}:{ts}".encode()).hexdigest()[:8]
    return f"{prefix}_{ts}_{head[:8]}_{suffix}"


def load_active_plan() -> dict[str, Any]:
    return load_json(PLAN_PATH, {})


def save_active_plan(plan: dict[str, Any]) -> None:
    write_json_atomic(PLAN_PATH, plan)


def ensure_v13_plan() -> dict[str, Any]:
    plan = load_active_plan()
    if str(plan.get("plan_version") or "") != "tiny_retrain_confirmation_v13_state_action_interface":
        raise SystemExit(f"Active plan is not v13 state-action interface: {PLAN_PATH}")
    return plan


def ensure_v13_slice2_plan() -> dict[str, Any]:
    plan = load_active_plan()
    if str(plan.get("plan_version") or "") != "tiny_retrain_confirmation_v13_slice_2_state_action_wiring":
        raise SystemExit(f"Active plan is not v13 slice-2 state-action wiring: {PLAN_PATH}")
    return plan


def persist_plan_scope(plan: dict[str, Any], execution_scope: str) -> dict[str, Any]:
    plan = dict(plan)
    plan["execution_scope"] = execution_scope
    plan["bridge_stage"] = execution_scope
    save_active_plan(plan)
    return plan


def base_payload(run_instance_id: str, spec_reference: str | None = None) -> dict[str, Any]:
    ident = current_repo_identity()
    return {
        "run_instance_id": run_instance_id,
        "working_head_commit": ident["working_head_commit"],
        "vendor_head_commit": ident["vendor_head_commit"],
        "branch": ident["branch"],
        "truth_contract_hash": sha256_file(TRUTH_CONTRACT_PATH),
        "acceptance_contract_hash": sha256_file(ACCEPTANCE_CONTRACT_PATH),
        "spec_reference": spec_reference or V13_SPEC_REFERENCE,
        "timestamp_utc": utc_now(),
    }


def write_gate(
    gate_path: Path,
    gate_id: str,
    gate_name: str,
    run_instance_id: str,
    status: str,
    blocking_reasons: list[str],
    allowed_next_phases: list[str],
    extra: dict[str, Any],
    spec_reference: str | None = None,
) -> dict[str, Any]:
    payload = {
        "gate_id": gate_id,
        "gate_name": gate_name,
        **base_payload(run_instance_id, spec_reference=spec_reference),
        "status": status,
        "blocking_reasons": blocking_reasons,
        "allowed_next_phases": allowed_next_phases,
        **extra,
    }
    write_json_atomic(gate_path, payload)
    return payload


def source_v12_gate(name: str) -> dict[str, Any]:
    return load_json(GATES_V12_DIR / name, {})


def source_v12_artifact(name: str) -> dict[str, Any]:
    return load_json(ARTIFACT_DIR / name, {})

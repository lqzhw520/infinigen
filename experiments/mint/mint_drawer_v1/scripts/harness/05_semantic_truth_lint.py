#!/usr/bin/env python3
"""
05_semantic_truth_lint.py
Harness v2 semantic gate.

Checks:
  - current_truth.json matches runtime/canonical pointers
  - generated docs are present, generated, and consistent
  - experiment evidence respects scope/narrative guardrails
  - completed current-driving evidence is verified
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from truth_backend import (
    CAMPAIGN_ROOT,
    CURRENT_TRUTH_PATH,
    EVIDENCE_DIR,
    EVIDENCE_INDEX_PATH,
    HANDOFF_PATH,
    NEXT_ACTIONS_PATH,
    STATE_PATH,
    USAGE_GUIDE_PATH,
    CAMPAIGN_TRUTH_PATH,
    GENERATED_MARKER,
    load_experiment_specs,
)

errors: list[str] = []
warnings: list[str] = []


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def check_claims_text_encoding() -> None:
    claims_path = CAMPAIGN_ROOT / "sovereign" / "claims.yaml"
    if not claims_path.exists():
        return
    text = claims_path.read_text()
    bad_literals = [literal for literal in [r"\xB1", r"\xD7"] if literal in text]
    if bad_literals:
        warnings.append(
            f"claims.yaml contains escaped literal artifacts: {', '.join(bad_literals)}"
        )


def check_truth_consistency() -> None:
    if not CURRENT_TRUTH_PATH.exists():
        errors.append("current_truth.json missing")
        return
    truth = load_json(CURRENT_TRUTH_PATH)
    state = load_json(STATE_PATH)
    next_actions = load_json(NEXT_ACTIONS_PATH)

    if truth["current"]["phase"] != state.get("phase"):
        errors.append(
            f"state.json phase mismatch: {state.get('phase')} != {truth['current']['phase']}"
        )
    if truth["current"]["phase_gate"] != state.get("phase_gate"):
        errors.append(
            f"state.json phase_gate mismatch: {state.get('phase_gate')} != {truth['current']['phase_gate']}"
        )
    if truth["current"]["verdict"] != state.get("verdict"):
        errors.append(
            f"state.json verdict mismatch: {state.get('verdict')} != {truth['current']['verdict']}"
        )

    if next_actions.get("phase") != truth["current"]["phase"]:
        errors.append(
            f"next_actions.json phase mismatch: {next_actions.get('phase')} != {truth['current']['phase']}"
        )
    if next_actions.get("verdict") != truth["current"]["verdict"]:
        errors.append(
            f"next_actions.json verdict mismatch: {next_actions.get('verdict')} != {truth['current']['verdict']}"
        )
    if next_actions.get("decision") != truth["current"]["decision"]:
        errors.append(
            f"next_actions.json decision mismatch: {next_actions.get('decision')} != {truth['current']['decision']}"
        )

    truth_next = truth["current"].get("next_action") or {}
    next_actions_actions = next_actions.get("actions", [])
    canonical_next = None
    in_progress = [item for item in next_actions_actions if item.get("status") == "in_progress"]
    pending = [item for item in next_actions_actions if item.get("status") == "pending"]
    if in_progress:
        canonical_next = in_progress[0]
    elif pending:
        canonical_next = pending[0]
    if (canonical_next or {}) != truth_next:
        errors.append("current_truth next_action does not match canonical next_actions ordering")


def check_generated_doc(path: Path, truth: dict) -> None:
    if not path.exists():
        errors.append(f"derived doc missing: {path.relative_to(CAMPAIGN_ROOT)}")
        return
    text = path.read_text()
    if GENERATED_MARKER not in text:
        errors.append(f"{path.relative_to(CAMPAIGN_ROOT)} missing generated marker")
    if "Source: `sovereign/current_truth.json`" not in text:
        errors.append(f"{path.relative_to(CAMPAIGN_ROOT)} missing source pointer")
    if truth["current"]["verdict"] not in text:
        errors.append(f"{path.relative_to(CAMPAIGN_ROOT)} does not contain current verdict")
    if truth["current"]["phase"] not in text:
        errors.append(f"{path.relative_to(CAMPAIGN_ROOT)} does not contain current phase")


def check_derived_docs() -> None:
    truth = load_json(CURRENT_TRUTH_PATH)
    latest_bootstrap = CAMPAIGN_ROOT / truth["derived_docs"]["latest_bootstrap"]
    for path in [latest_bootstrap, USAGE_GUIDE_PATH, HANDOFF_PATH, CAMPAIGN_TRUTH_PATH]:
        check_generated_doc(path, truth)
    stale_docs = truth["current"].get("stale_docs", [])
    if stale_docs:
        warnings.append(f"current_truth still reports stale docs: {len(stale_docs)}")


def check_scope_and_narrative() -> None:
    specs = load_experiment_specs()
    evidence_index = load_json(EVIDENCE_INDEX_PATH)
    evidence_map = {entry["evidence_id"]: entry for entry in evidence_index.get("entries", [])}
    for spec_id, spec in specs.items():
        evidence_id = spec.get("canonical_evidence_id")
        if not evidence_id:
            continue
        evidence_path = EVIDENCE_DIR / f"{evidence_id}.yaml"
        if not evidence_path.exists():
            warnings.append(f"{spec_id}: canonical evidence file missing ({evidence_id})")
            continue
        text = evidence_path.read_text()
        for phrase in spec.get("disallowed_phrases", []):
            if phrase in text:
                errors.append(f"{evidence_id}: disallowed phrase present: {phrase}")
        if spec.get("requires_scope_guardrail", True) and "scope_guardrail:" not in text:
            errors.append(f"{evidence_id}: missing scope_guardrail")
        entry = evidence_map.get(evidence_id)
        if entry and spec.get("must_be_verified_for_current_truth", False) and not entry.get("verified"):
            errors.append(f"{evidence_id}: unverified evidence cannot drive current truth")


def check_promotion_guardrail() -> None:
    next_actions = load_json(NEXT_ACTIONS_PATH)
    evidence_index = load_json(EVIDENCE_INDEX_PATH)
    verified = {entry["evidence_id"]: bool(entry.get("verified")) for entry in evidence_index.get("entries", [])}

    for action in next_actions.get("actions", []):
        if action.get("status") != "completed":
            continue
        referenced = []
        if isinstance(action.get("evidence"), str):
            referenced.append(action["evidence"])
        result = action.get("result", {})
        if isinstance(result, dict) and isinstance(result.get("evidence"), str):
            referenced.append(result["evidence"])
        for evidence_id in referenced:
            if evidence_id in verified and not verified[evidence_id]:
                errors.append(
                    f"completed action {action.get('type')} references unverified evidence {evidence_id}"
                )


def check_claim_debt() -> None:
    truth = load_json(CURRENT_TRUTH_PATH)
    debt_flags = truth.get("claims", {}).get("debt_flags", [])
    for item in debt_flags[:8]:
        flags = ", ".join(flag.get("flag", "?") for flag in item.get("flags", []))
        warnings.append(f"claim debt: {item.get('claim_id')} -> {flags}")
    if len(debt_flags) > 8:
        warnings.append(f"claim debt: ... +{len(debt_flags) - 8} more")


def main() -> None:
    print("=== 05_semantic_truth_lint.py ===")
    print(f"Campaign root: {CAMPAIGN_ROOT}")
    print()

    check_truth_consistency()
    if CURRENT_TRUTH_PATH.exists():
        check_claims_text_encoding()
        check_derived_docs()
        check_scope_and_narrative()
        check_promotion_guardrail()
        check_claim_debt()

    if warnings:
        print(f"--- WARNINGS ({len(warnings)}) ---")
        for warning in warnings:
            print(f"  WARN: {warning}")
        print()

    if errors:
        print(f"--- ERRORS ({len(errors)}) ---")
        for error in errors:
            print(f"  ERROR: {error}")
        print()
        print("Semantic truth lint FAILED.")
        sys.exit(1)

    print("=== SEMANTIC TRUTH LINT OK ===")
    print("  Errors: 0")
    if warnings:
        print(f"  Warnings: {len(warnings)}")
    sys.exit(0)


if __name__ == "__main__":
    main()

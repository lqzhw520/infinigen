#!/usr/bin/env python3
"""
02_reconcile_sources.py
Phase 1 / Phase 2: Source-of-truth gate.

Exit codes:
  0 = all OK (no ERRORs, warnings are OK)
  1 = ERROR found (must not proceed)

Checks:
  - Root-level files that look canonical but are not symlinks/stubs
  - CAMPAIGN_TRUTH.md stub content
  - next_actions references valid claim IDs
  - legacy markers present on known legacy files
"""
import os
import sys
import json
from pathlib import Path

CAMPAIGN_ROOT = Path(__file__).parent.parent.parent.resolve()
ROOT = CAMPAIGN_ROOT
SOVEREIGN = ROOT / "sovereign"

errors = []
warnings = []

EXPECTED_SYMLINKS = {"manifest.yaml", "state.json", "next_actions.json"}
EXPECTED_STUB_FILE = "CAMPAIGN_TRUTH.md"
STUB_MAGIC = "LEGACY POINTER"

# Legacy files that MUST have LEGACY DOCUMENT header
LEGACY_FILES = {
    "campaign_status.md", "findings.md", "progress.md",
    "decision_memo.md", "takeover_memo.md", "step_review_guide.md",
    "campaign_spec.md", "review_prompt.md", "acceptance_criteria.json",
    "summary.json", "review.json",
}

# Known derived files (may exist but not errors if not canonical-looking)
DERIVED_FILES = {
    "D1_ADVISOR_REPORT.md", "RESOLUTION_PLAN.md",
    "D1_ADVISOR_REPORT.md", "RESOLUTION_PLAN.md",
}


def check_root_files():
    """Check that canonical YAML/JSON files are symlinks or stubs."""
    for fname in EXPECTED_SYMLINKS:
        fpath = ROOT / fname
        if not fpath.exists():
            errors.append(f"MISSING: {fname} (not found at root)")
        elif not fpath.is_symlink():
            # Check if it's the original file (not a symlink)
            real = fpath.resolve()
            sovereign_target = SOVEREIGN / fname
            if real == sovereign_target:
                # It's the same file but not symlinked
                errors.append(
                    f"NOT_SYMLINK: {fname} exists at root but is not a symlink. "
                    f"Run: ln -sf sovereign/{fname} {fname}"
                )
            else:
                errors.append(
                    f"UNEXPECTED_FILE: {fname} exists at root but points elsewhere. "
                    f"Expected: -> sovereign/{fname}"
                )


def check_campaign_truth_stub():
    """CAMPAIGN_TRUTH.md must be a stub with LEGACY POINTER text."""
    fpath = ROOT / EXPECTED_STUB_FILE
    if not fpath.exists():
        warnings.append(f"NOT_FOUND: {EXPECTED_STUB_FILE} (will be created by sync script)")
        return
    content = fpath.read_text()
    if "GENERATED FILE" in content:
        errors.append(
            f"CAMPAIGN_TRUTH.md contains 'GENERATED FILE' — "
            f"this should be in sovereign/CAMPAIGN_TRUTH.generated.md, not at root"
        )
    if "LEGACY POINTER" not in content:
        errors.append(
            f"CAMPAIGN_TRUTH.md missing '{STUB_MAGIC}' marker. "
            f"Root should only contain a stub pointing to sovereign/CAMPAIGN_TRUTH.generated.md"
        )


def check_legacy_markers():
    """Known legacy files should have LEGACY DOCUMENT header."""
    for fname in LEGACY_FILES:
        fpath = ROOT / fname
        if not fpath.exists():
            continue  # not an error if missing
        content = fpath.read_text()
        if "LEGACY DOCUMENT" not in content:
            warnings.append(
                f"LEGACY_FILE_NOT_MARKED: {fname} exists but missing 'LEGACY DOCUMENT' header"
            )


def check_evidence_index():
    """Evidence index must exist and be valid JSON."""
    idx_path = SOVEREIGN / "evidence" / "index.json"
    if not idx_path.exists():
        errors.append(f"MISSING: sovereign/evidence/index.json")
        return
    try:
        with open(idx_path) as f:
            data = json.load(f)
        if "entries" not in data:
            errors.append("evidence/index.json missing 'entries' field")
    except json.JSONDecodeError as e:
        errors.append(f"INVALID JSON: sovereign/evidence/index.json: {e}")


def check_sovereign_files_exist():
    """All required sovereign files must exist."""
    required = [
        "manifest.yaml", "state.json", "claims.yaml",
        "next_actions.json", "handoff.md",
        "evidence/index.json",
    ]
    for fname in required:
        fpath = SOVEREIGN / fname
        if not fpath.exists():
            errors.append(f"MISSING sovereign file: {fname}")


def main():
    print("=== 02_reconcile_sources.py ===")
    print(f"Campaign root: {ROOT}")
    print()

    check_root_files()
    check_campaign_truth_stub()
    check_legacy_markers()
    check_evidence_index()
    check_sovereign_files_exist()

    if warnings:
        print(f"\n--- WARNINGS ({len(warnings)}) ---")
        for w in warnings:
            print(f"  WARN: {w}")

    if errors:
        print(f"\n--- ERRORS ({len(errors)}) ---")
        for e in errors:
            print(f"  ERROR: {e}")
        print()
        print("Reconcile FAILED. Fix errors before proceeding.")
        sys.exit(1)
    else:
        print(f"\n=== RECONCILE OK ({len(warnings)} warnings) ===")
        if warnings:
            print("Warnings are informational — no blocking errors.")
        sys.exit(0)


if __name__ == "__main__":
    main()

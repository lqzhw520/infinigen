#!/usr/bin/env python3
"""
validate_task_authority.py — campaign-native

Validates that runtime geometry classification matches the GOC artifact authority.
This validator implements the rule: GOC authority is IMMUTABLE during execution.

If runtime output disagrees with GOC artifact authority, this validator MUST fail
with AUTHORITY_MISMATCH and print a structured error report.

Usage:
    python scripts/harness/validators/validate_task_authority.py sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml
    python scripts/harness/validators/validate_task_authority.py sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml --dry-run
    python scripts/harness/validators/validate_task_authority.py sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml --fixture invalid_authority_drift
    python scripts/harness/validators/validate_task_authority.py sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml --fixture valid_authority_match

Exit codes:
    0  = PASS (authority matches or fixture test passed)
    1  = FAIL (authority mismatch detected)
    2  = ERROR (invalid arguments, missing files, YAML parse error)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Campaign-native path resolution:
# When running from scripts/harness/validators/, campaign root = parent.parent.parent
# Support MINT_TASK_ROOT env var; default to campaign root.
CAMPAIGN_ROOT = Path(os.environ.get(
    "MINT_TASK_ROOT",
    str(Path(__file__).resolve().parent.parent.parent)
))
YAML_AVAILABLE = False
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    pass


def die(msg: str, code: int = 1) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def load_yaml(path: Path) -> dict[str, Any]:
    if not YAML_AVAILABLE:
        try:
            import ruamel.yaml as ruamel_yaml
            with path.open() as fh:
                return dict(ruamel_yaml.YAML().load(fh))
        except ImportError:
            pass
        die(f"yaml not available; install pyyaml or ruamel.yaml to parse {path}", code=2)
    with path.open() as fh:
        data = yaml.safe_load(fh)
        if data is None:
            return {}
        return dict(data)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def resolve_path(base: Path, rel: str) -> Path:
    """Resolve relative path from CAMPAIGN_ROOT, or return absolute path."""
    if rel.startswith("/"):
        return Path(rel)
    return (base / rel).resolve()


def load_runtime_counts(
    args: argparse.Namespace, task_spec: dict[str, Any]
) -> Optional[dict[str, int]]:
    """
    Load runtime geometry counts from one of:
      1. --fixture <name>   (test fixture, overrides everything)
      2. --runtime-json <path>  (actual runtime output)
      3. --runtime-counts l:<int> f:<int> h:<int>  (explicit override, DANGEROUS)
    Returns None if no runtime data is provided (dry-run mode).
    """
    if hasattr(args, "fixture") and args.fixture:
        fixture = args.fixture
        if fixture == "invalid_authority_drift":
            return {"legal_pad": 31, "forbidden": 27, "handle": 9}
        elif fixture == "valid_authority_match":
            return {"legal_pad": 29, "forbidden": 26, "handle": 9}
        else:
            die(f"Unknown fixture: {fixture}. Available: invalid_authority_drift, valid_authority_match", code=2)

    if hasattr(args, "runtime_json") and args.runtime_json:
        path = resolve_path(CAMPAIGN_ROOT, args.runtime_json)
        data = load_json(path)
        return {
            "legal_pad": data.get("legal_pad", data.get("gripper_contact_geom_count", 0)),
            "forbidden": data.get("forbidden", data.get("forbidden_robot_geom_count", 0)),
            "handle": data.get("handle", data.get("handle_geom_count", 0)),
        }

    if hasattr(args, "runtime_counts") and args.runtime_counts:
        parts = args.runtime_counts.split(",")
        counts = {}
        for part in parts:
            if ":" not in part:
                die(f"Invalid --runtime-counts format: {part}. Expected 'l:<int>,f:<int>,h:<int>'", code=2)
            key, val = part.split(":", 1)
            counts[key.strip()] = int(val.strip())
        return counts

    return None


def validate_authority(
    task_spec: dict[str, Any],
    runtime_counts: Optional[dict[str, int]],
    dry_run: bool = False,
) -> tuple[bool, str]:
    """
    Core validation logic.
    Returns (passed, message).
    """
    task_auth = task_spec.get("task_authority", {})

    artifact_counts = {
        "legal_pad": task_auth.get("legal_pad_count", 0),
        "forbidden": task_auth.get("forbidden_count", 0),
        "handle": task_auth.get("handle_count", 0),
    }
    goc_version = task_auth.get("goc_version", "unknown")
    goc_artifact_commit = task_auth.get("goc_artifact_commit", "unknown")

    if dry_run:
        return (
            True,
            f"[DRY-RUN] task_authority={artifact_counts} goc_version={goc_version} "
            f"goc_artifact_commit={goc_artifact_commit}. "
            "Would validate against runtime counts if provided.",
        )

    if runtime_counts is None:
        return (
            True,
            f"task_authority={artifact_counts} goc_version={goc_version}. "
            "No runtime counts provided (no --fixture, --runtime-json, or --runtime-counts). "
            "Authority initialized but not yet compared against runtime. "
            "Validator will run again during execution (V03 gate).",
        )

    mismatches = []
    for key in ("legal_pad", "forbidden", "handle"):
        artifact_val = artifact_counts[key]
        runtime_val = runtime_counts[key]
        if artifact_val != runtime_val:
            mismatches.append(
                f"{key}: GOC_artifact={artifact_val} vs runtime={runtime_val}"
            )

    if mismatches:
        report = [
            "AUTHORITY_MISMATCH",
            "=" * 60,
            f"ERROR: Runtime geometry classification disagrees with GOC artifact authority.",
            f"GOC version:      {goc_version}",
            f"GOC artifact:     {task_auth.get('goc_artifact_path', 'unknown')}",
            f"GOC commit:      {goc_artifact_commit}",
            "",
            "Mismatches detected:",
        ]
        for m in mismatches:
            report.append(f"  - {m}")
        report.extend(
            [
                "",
                "CRITICAL: The agent MUST NOT continue if runtime and GOC disagree.",
                "STOP action: Record this report and update handoff/CURRENT.md.",
                "Next gate: GOC_V3_EXACT_ID_CONTRACT_REBUILD",
                "",
                "Root cause from Phase 1H FSM v2 arbitration:",
                "  - GOC-v2 artifact said legal_pad=29 / forbidden=26",
                "  - Runtime body-based classification said legal_pad=31 / forbidden=27",
                "  - FSM was modified to accept 31/27 as 'runtime reality'",
                "  - This changed acceptance standard during execution — PROHIBITED.",
                "",
                "Recovery: GOC-v3 with explicit per-geom IDs required.",
            ]
        )
        return False, "\n".join(report)

    return (
        True,
        f"PASS: Authority match. GOC={artifact_counts} runtime={runtime_counts}",
    )


def run_fixture_test(
    fixture_name: str, task_spec: dict[str, Any]
) -> tuple[bool, str]:
    """Run a named fixture and report result."""
    runtime_counts = load_runtime_counts(
        argparse.Namespace(
            fixture=fixture_name, runtime_json=None, runtime_counts=None
        ),
        task_spec,
    )
    passed, msg = validate_authority(task_spec, runtime_counts, dry_run=False)

    fixture_data = task_spec.get("fixtures", {}).get(f"fixture_{fixture_name}", {})
    expected = fixture_data.get("expected", "PASS")

    if expected == "PASS" and passed:
        return True, f"fixture={fixture_name} expected=PASS actual=PASS — OK"
    elif expected == "FAIL" and not passed:
        expected_error = fixture_data.get("expected_error", "FAIL")
        if expected_error in msg:
            return True, f"fixture={fixture_name} expected=FAIL({expected_error}) actual=FAIL — validator correctly rejected drift"
        else:
            return False, f"fixture={fixture_name} expected=FAIL({expected_error}) but FAIL reason differs"
    elif expected == "PASS" and not passed:
        return False, f"fixture={fixture_name} expected=PASS actual=FAIL — validator broke"
    elif expected == "FAIL" and passed:
        return False, f"fixture={fixture_name} expected=FAIL actual=PASS — validator should have caught mismatch"
    return False, f"fixture={fixture_name} unknown state"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate runtime geometry classification against GOC artifact authority.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "task_spec",
        nargs="?",
        default="sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml",
        help="Path to task YAML spec (relative to CAMPAIGN_ROOT or absolute)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialize authority check without comparing against runtime (for S0 gate)",
    )
    parser.add_argument(
        "--fixture",
        choices=["invalid_authority_drift", "valid_authority_match"],
        help="Run a named test fixture instead of real runtime data",
    )
    parser.add_argument(
        "--runtime-json",
        metavar="PATH",
        help="Path to runtime output JSON (relative to CAMPAIGN_ROOT or absolute)",
    )
    parser.add_argument(
        "--runtime-counts",
        metavar="l:<int>,f:<int>,h:<int>",
        help="Explicit runtime counts as fallback (DANGEROUS — use only for debug)",
    )

    args = parser.parse_args()

    task_spec_path = resolve_path(CAMPAIGN_ROOT, args.task_spec)
    if not task_spec_path.exists():
        die(f"Task spec not found: {task_spec_path}", code=2)

    task_spec = load_yaml(task_spec_path)

    if args.fixture:
        passed, msg = run_fixture_test(args.fixture, task_spec)
        print(msg)
        if not passed:
            print("FAIL", file=sys.stderr)
            sys.exit(1)
        print("PASS")
        sys.exit(0)

    task_auth = task_spec.get("task_authority", {})
    if not task_auth:
        die("task_authority block not found in task spec", code=2)

    runtime_counts = load_runtime_counts(args, task_spec)
    passed, msg = validate_authority(task_spec, runtime_counts, dry_run=args.dry_run)

    print(msg)
    if not passed:
        print("\nVALIDATION FAILED", file=sys.stderr)
        print("RESULT: AUTHORITY_MISMATCH", file=sys.stderr)
        sys.exit(1)
    print("\nVALIDATION PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()

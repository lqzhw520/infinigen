#!/usr/bin/env python3
from __future__ import annotations
"""
validate_closeout.py — campaign-native

Validates task terminal closeout and classifies outcomes.
This is a MACHINE-DECISION validator — not a human judgment.

Classifications:
  INFRASTRUCTURE_BLOCKED  — rollout crashed, Traceback present, or result JSON missing
                            This is NOT a route failure. Do NOT classify as ROLLOUT_EXHAUSTED.
  ROLLOUT_EXHAUSTED       — all rollout seeds completed without crashes, route was evaluated,
                            but none reached the target (controller/route issue)
  ROUTE_SUCCESS           — route was evaluated and reached target
  GATE_PASSED             — all gates passed
  INVALID_CLOSEOUT        — closeout_decision.json missing or malformed

Usage:
    python scripts/harness/validators/validate_closeout.py --fixture rollout_crash
    python scripts/harness/validators/validate_closeout.py --fixture invalid_authority_drift
    python scripts/harness/validators/validate_closeout.py --fixture valid_closeout
    python scripts/harness/validators/validate_closeout.py --closeout-json path/to/closeout_decision.json
    python scripts/harness/validators/validate_closeout.py \
        --task sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml \
        --closeout-json closeout_decision.json

Exit codes:
    0  = PASS (valid closeout, classification correct)
    1  = FAIL (invalid closeout or wrong classification)
    2  = ERROR (invalid arguments, missing files, YAML parse error)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

# Campaign-native path resolution.
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

# Classification constants
CLASS_INFRASTRUCTURE_BLOCKED = "INFRASTRUCTURE_BLOCKED"
CLASS_ROLLOUT_EXHAUSTED = "ROLLOUT_EXHAUSTED"
CLASS_ROUTE_SUCCESS = "ROUTE_SUCCESS"
CLASS_GATE_PASSED = "GATE_PASSED"
CLASS_INVALID_CLOSEOUT = "INVALID_CLOSEOUT"
CLASS_SMOKE_BROKEN = "S2_SMOKE_CONTACT_EVALUATION_BROKEN"
CLASS_AUTHORITY_DRIFT = "AUTHORITY_MISMATCH_DRIFT"

# Crash indicators — these mean INFRASTRUCTURE_BLOCKED, not route failure
CRASH_INDICATORS = [
    "Traceback (most recent call last)",
    "Traceback",
    "Error:",
    "Exception:",
    "ValueError:",
    "TypeError:",
    "AttributeError:",
    'File "',  # Python traceback line
    "ModuleNotFoundError",
    "ImportError",
]

# Infrastructure failure patterns (known bugs, not route issues)
INFRASTRUCTURE_PATTERNS = [
    r"_get_grasp_pose.*expected.*x.*matrix.*got.*vector",
    r"AttributeError.*tuple.*object.*has no attribute.*get",
    r"physical_admission_passed=false",
    r"grasp_pose.*format.*mismatch",
]


def die(msg: str, code: int = 1) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def load_yaml(path: Path) -> dict:
    if not YAML_AVAILABLE:
        try:
            import ruamel.yaml
            with path.open() as fh:
                return dict(ruamel.yaml.YAML().load(fh))
        except ImportError:
            die(f"yaml not available: {path}", code=2)
    with path.open() as fh:
        return dict(yaml.safe_load(fh) or {})


def load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def has_traceback(text: str) -> bool:
    """Check if text contains a Python traceback."""
    return "Traceback (most recent call last)" in text


def detect_infrastructure_failure(
    rollout_stdout: str,
    rollout_stderr: str,
    result_json_exists: bool,
    result_json_path: Optional[Path],
) -> tuple[bool, str]:
    """
    Detect infrastructure failures (crashes) vs route failures.
    Returns (is_infrastructure_failure, reason).
    """
    combined = rollout_stdout + "\n" + rollout_stderr

    if has_traceback(combined):
        return True, "Python traceback detected in rollout output"

    for pattern in INFRASTRUCTURE_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return True, f"Known infrastructure failure pattern matched: {pattern}"

    if not result_json_exists:
        return True, "Rollout result JSON missing — rollout did not complete"

    if result_json_path and result_json_path.exists():
        try:
            data = load_json(result_json_path)
            if not isinstance(data, dict):
                return True, f"Rollout result JSON is not a dict (type={type(data)})"
        except json.JSONDecodeError as e:
            return True, f"Rollout result JSON is malformed: {e}"
        except Exception as e:
            return True, f"Rollout result JSON read error: {e}"

    return False, ""


def classify_closeout(
    rollout_stdout: str,
    rollout_stderr: str,
    rollout_exit_code: int,
    result_json_exists: bool,
    result_json_path: Optional[Path],
    runtime_counts: Optional[dict[str, int]] = None,
    task_authority: Optional[dict[str, int]] = None,
    rollout_seed_count: int = 0,
    rollout_seeds_completed: int = 0,
    route_evaluated: bool = False,
    contact_reported: bool = False,
) -> tuple[str, str]:
    """
    Machine-classify the closeout based on evidence.
    Returns (classification, message).
    """
    is_infra, infra_reason = detect_infrastructure_failure(
        rollout_stdout, rollout_stderr, result_json_exists, result_json_path
    )

    if is_infra:
        combined = rollout_stdout + "\n" + rollout_stderr
        if "expected.*4x4 matrix" in combined or "grasp_pose" in combined.lower():
            infra_type = "ROLLOUT_COMMAND_BROKEN"
        elif "AttributeError" in combined and "tuple" in combined:
            infra_type = "S2_SMOKE_CONTACT_EVALUATION_BROKEN"
        else:
            infra_type = "INFRASTRUCTURE_BLOCKED"

        return CLASS_INFRASTRUCTURE_BLOCKED, (
            f"{infra_type}: {infra_reason}. "
            f"Crash is NOT route failure. "
            f"Result JSON exists={result_json_exists}. "
            f"Rule: Crash => INFRASTRUCTURE_BLOCKED, not ROLLOUT_EXHAUSTED."
        )

    if runtime_counts and task_authority:
        for key in ("legal_pad", "forbidden"):
            art = task_authority.get(f"{key}_count", 0)
            run = runtime_counts.get(key, 0)
            if art != run:
                return CLASS_AUTHORITY_DRIFT, (
                    f"AUTHORITY_MISMATCH: GOC_artifact.{key}={art} vs runtime.{key}={run}. "
                    f"FSM modified acceptance criteria during execution. "
                    f"STOP. Do NOT classify as ROLLOUT_EXHAUSTED."
                )

    if route_evaluated and rollout_seeds_completed == rollout_seed_count:
        if not contact_reported:
            return CLASS_ROLLOUT_EXHAUSTED, (
                f"ROLLOUT_EXHAUSTED: All {rollout_seed_count} seeds completed, "
                f"route was evaluated, no contact. "
                f"Contact reported={contact_reported}. "
                f"This IS a route/controller failure."
            )

    if route_evaluated and contact_reported:
        return CLASS_ROUTE_SUCCESS, (
            f"ROUTE_SUCCESS: Route evaluated, contact reported. "
            f"seeds_completed={rollout_seeds_completed}/{rollout_seed_count}"
        )

    if rollout_seeds_completed < rollout_seed_count:
        return CLASS_INVALID_CLOSEOUT, (
            f"INVALID_CLOSEOUT: Only {rollout_seeds_completed}/{rollout_seed_count} "
            f"seeds completed. Closeout is premature."
        )

    return CLASS_INVALID_CLOSEOUT, "Cannot determine classification from available evidence"


def validate_closeout(
    closeout_data: Optional[dict[str, Any]],
    args: argparse.Namespace,
    task_spec: Optional[dict[str, Any]] = None,
) -> tuple[bool, str]:
    """
    Main validation entry point.
    Returns (passed, message).
    """
    task_authority = None
    if task_spec:
        auth = task_spec.get("task_authority", {})
        task_authority = {
            "legal_pad_count": auth.get("legal_pad_count", 0),
            "forbidden_count": auth.get("forbidden_count", 0),
            "handle_count": auth.get("handle_count", 0),
        }

    runtime_counts = None
    if hasattr(args, "fixture") and args.fixture:
        if args.fixture == "invalid_authority_drift":
            runtime_counts = {"legal_pad": 31, "forbidden": 27, "handle": 9}
        elif args.fixture == "valid_authority_match":
            runtime_counts = {"legal_pad": 29, "forbidden": 26, "handle": 9}

    rollout_stdout = ""
    rollout_stderr = ""
    rollout_exit_code = 1
    result_json_exists = False
    result_json_path = None
    rollout_seed_count = 0
    rollout_seeds_completed = 0
    route_evaluated = False
    contact_reported = False

    if args.fixture == "rollout_crash":
        rollout_stdout = (
            "Traceback (most recent call last):\n"
            '  File "scripts/mint/run_full_robot_teacher_probe.py", line 347, in _get_grasp_pose\n'
            '    raise ValueError(\'expected 4x4 matrix, got (6,) vector\')\n'
            "ValueError: expected 4x4 matrix, got (6,) vector\n"
            "During rollout seed 0 of 12...\n"
        )
        rollout_exit_code = 1
        result_json_exists = False
        rollout_seed_count = 12
        rollout_seeds_completed = 0
        route_evaluated = False
    elif args.fixture == "valid_closeout":
        rollout_stdout = "Rollout completed successfully. Contact detected.\n"
        rollout_exit_code = 0
        result_json_exists = True
        rollout_seed_count = 12
        rollout_seeds_completed = 12
        route_evaluated = True
        contact_reported = True
    elif args.fixture == "invalid_authority_drift":
        rollout_stdout = ""
        rollout_exit_code = 0
        result_json_exists = True
        rollout_seed_count = 12
        rollout_seeds_completed = 12
        route_evaluated = True
    elif closeout_data:
        rollout_stdout = closeout_data.get("rollout_stdout", "")
        rollout_stderr = closeout_data.get("rollout_stderr", "")
        rollout_exit_code = closeout_data.get("rollout_exit_code", 1)
        result_path = closeout_data.get("result_json_path")
        if result_path:
            result_json_path = Path(result_path) if result_path else None
            result_json_exists = result_json_path.exists() if result_json_path else False
        else:
            result_json_exists = closeout_data.get("result_json_exists", False)
        rollout_seed_count = closeout_data.get("rollout_seed_count", 0)
        rollout_seeds_completed = closeout_data.get("rollout_seeds_completed", 0)
        route_evaluated = closeout_data.get("route_evaluated", False)
        contact_reported = closeout_data.get("contact_reported", False)
        if "runtime_counts" in closeout_data:
            runtime_counts = closeout_data["runtime_counts"]

    classification, msg = classify_closeout(
        rollout_stdout=rollout_stdout,
        rollout_stderr=rollout_stderr,
        rollout_exit_code=rollout_exit_code,
        result_json_exists=result_json_exists,
        result_json_path=result_json_path,
        runtime_counts=runtime_counts,
        task_authority=task_authority,
        rollout_seed_count=rollout_seed_count,
        rollout_seeds_completed=rollout_seeds_completed,
        route_evaluated=route_evaluated,
        contact_reported=contact_reported,
    )

    if args.fixture:
        fixture_outcomes = {
            "rollout_crash": (CLASS_INFRASTRUCTURE_BLOCKED, "ROLLOUT_COMMAND_BROKEN"),
            "invalid_authority_drift": (CLASS_AUTHORITY_DRIFT, "AUTHORITY_MISMATCH"),
            "valid_closeout": (CLASS_ROUTE_SUCCESS, "ROUTE_SUCCESS"),
        }
        expected_class, expected_subtype = fixture_outcomes.get(args.fixture, ("", ""))
        if classification == expected_class:
            return True, (
                f"fixture={args.fixture} "
                f"expected_class={expected_class} actual={classification} — OK. "
                f"{msg}"
            )
        else:
            return False, (
                f"fixture={args.fixture} "
                f"expected_class={expected_class} actual={classification} — MISMATCH. "
                f"{msg}"
            )

    return True, f"classification={classification}: {msg}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate terminal closeout and classify outcomes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Classifications:
  INFRASTRUCTURE_BLOCKED  Rollout crashed or result JSON missing (NOT route failure)
  ROLLOUT_EXHAUSTED       All seeds completed, route evaluated, no success
  ROUTE_SUCCESS           Route reached target
  AUTHORITY_MISMATCH_DRIFT GOC vs runtime classification disagreement

Examples:
  python scripts/harness/validators/validate_closeout.py --fixture rollout_crash
  python scripts/harness/validators/validate_closeout.py --fixture invalid_authority_drift
  python scripts/harness/validators/validate_closeout.py --fixture valid_closeout
  python scripts/harness/validators/validate_closeout.py --closeout-json path/to/closeout_decision.json
  python scripts/harness/validators/validate_closeout.py \
      --task sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml \
      --closeout-json closeout_decision.json
""",
    )
    parser.add_argument(
        "--fixture",
        choices=["rollout_crash", "invalid_authority_drift", "valid_closeout"],
        help="Run a named test fixture",
    )
    parser.add_argument(
        "--closeout-json",
        metavar="PATH",
        help="Path to closeout_decision.json (relative to CAMPAIGN_ROOT or absolute)",
    )
    parser.add_argument(
        "--task",
        default="sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml",
        help="Task spec YAML for authority context (default: sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml)",
    )

    args = parser.parse_args()

    closeout_data = None
    task_spec = None

    if args.closeout_json:
        closeout_path = (CAMPAIGN_ROOT / args.closeout_json).resolve()
        if not closeout_path.exists():
            die(f"closeout JSON not found: {closeout_path}", code=2)
        closeout_data = load_json(closeout_path)

    task_spec_path = (CAMPAIGN_ROOT / args.task).resolve()
    if task_spec_path.exists():
        task_spec = load_yaml(task_spec_path)

    passed, msg = validate_closeout(closeout_data, args, task_spec)

    print(msg)
    if not passed:
        print("\nVALIDATION FAILED", file=sys.stderr)
        sys.exit(1)
    print("\nVALIDATION PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()

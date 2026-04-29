#!/usr/bin/env python3
"""
agent_task_preflight.py — campaign-native

Agent Execution Harness pre-flight validator entry point.

This script provides a callable pre-execution gate that runs before any
FSM/rollout execution. It is the canonical entry point for the campaign harness.

It runs:
  V01: validate_task_authority.py --dry-run   (S0: authority initialized)
  V02: validate_diff_scope.py --dry-run       (S0: scope clean)

Usage:
    python scripts/harness/agent_task_preflight.py \
        sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml --dry-run

With explicit CAMPAIGN_ROOT:
    MINT_TASK_ROOT=/path/to/mint_drawer_v1 \
        python scripts/harness/agent_task_preflight.py \
        sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml --dry-run

Command-line form:
    python scripts/harness/agent_task_preflight.py --spec sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml --dry-run

Also callable via sovereign_cli.py:
    python scripts/harness/sovereign_cli.py pre-flight-validator \
        --task-yaml sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml

Exit codes:
    0  = all pre-flight validators passed
    1  = one or more validators failed
    2  = error (missing files, YAML parse error, etc.)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Campaign-native path resolution.
# When running from scripts/harness/, campaign root = parent.parent
# Support MINT_TASK_ROOT env var; default to campaign root.
THIS_FILE = Path(__file__).resolve()
CAMPAIGN_ROOT = Path(os.environ.get(
    "MINT_TASK_ROOT",
    str(THIS_FILE.parent.parent)
))

# Validator tools (campaign-native paths)
VALIDATE_AUTHORITY = THIS_FILE.parent / "validators" / "validate_task_authority.py"
VALIDATE_DIFF_SCOPE = THIS_FILE.parent / "validators" / "validate_diff_scope.py"


def run_validator(script: Path, args: list[str]) -> tuple[int, str, str]:
    """Run a validator script and return (exit_code, stdout, stderr)."""
    cmd = [sys.executable, str(script)] + args
    full_env = dict(os.environ)
    full_env["MINT_TASK_ROOT"] = str(CAMPAIGN_ROOT)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(CAMPAIGN_ROOT),
        env=full_env,
    )
    return result.returncode, result.stdout, result.stderr


def print_header(title: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_result(name: str, passed: bool, stdout: str, stderr: str) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if not passed:
        for line in stdout.strip().splitlines()[:20]:
            print(f"       {line}")
        for line in stderr.strip().splitlines()[:10]:
            print(f"       {line}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent Execution Harness pre-flight: V01 + V02 validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "task_spec",
        nargs="?",
        default="sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml",
        help="Path to task YAML spec (relative to CAMPAIGN_ROOT or absolute)",
    )
    parser.add_argument(
        "--spec",
        dest="task_spec_alt",
        help="Alias for task_spec (for --spec form)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run mode: V01 initializes authority without runtime comparison",
    )
    parser.add_argument(
        "--v01-only",
        action="store_true",
        help="Run V01 only (skip V02 diff scope)",
    )
    parser.add_argument(
        "--v02-only",
        action="store_true",
        help="Run V02 only (skip V01 authority)",
    )

    args = parser.parse_args()

    # Support both positional and --spec form
    task_spec_arg = args.task_spec_alt or args.task_spec

    # Resolve task spec path relative to CAMPAIGN_ROOT
    task_spec_path = Path(task_spec_arg)
    if task_spec_path.is_absolute():
        if not task_spec_path.exists():
            task_spec_path = task_spec_path.resolve()
    else:
        task_spec_path = (CAMPAIGN_ROOT / task_spec_arg).resolve()

    print_header(f"Agent Task Pre-flight")
    print(f"  campaign_root: {CAMPAIGN_ROOT}")
    print(f"  task_spec:    {task_spec_path}")
    print(f"  dry_run:      {args.dry_run}")

    if not task_spec_path.exists():
        print(f"\nFATAL: Task spec not found: {task_spec_path}", file=sys.stderr)
        sys.exit(2)

    passed_count = 0
    failed_count = 0

    # V01: Task Authority check
    if not getattr(args, "v02_only", False):
        print("\n  [V01] validate_task_authority (S0 gate)")
        dry_args = ["--dry-run"] if args.dry_run else []
        code, stdout, stderr = run_validator(
            VALIDATE_AUTHORITY,
            [str(task_spec_path)] + dry_args,
        )
        passed_v01 = code == 0
        print_result("V01 validate_task_authority", passed_v01, stdout, stderr)
        if passed_v01:
            passed_count += 1
        else:
            failed_count += 1

    # V02: Diff Scope check
    if not getattr(args, "v01_only", False):
        print("\n  [V02] validate_diff_scope")
        code, stdout, stderr = run_validator(
            VALIDATE_DIFF_SCOPE,
            ["--task", str(task_spec_path), "--dry-run"],
        )
        passed_v02 = code == 0
        print_result("V02 validate_diff_scope", passed_v02, stdout, stderr)
        if passed_v02:
            passed_count += 1
        else:
            failed_count += 1

    # Summary
    print_header("Pre-flight Summary")
    print(f"  Passed: {passed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Task spec: {task_spec_path.name}")
    print(f"  Pre-flight callable: True")

    if failed_count == 0:
        print("\n  ALL PRE-FLIGHT VALIDATORS PASSED.")
        print("  Agent may proceed with execution.")
        print("  Next: run V03 during execution, V04 at closeout.")
        sys.exit(0)
    else:
        print("\n  PRE-FLIGHT VALIDATION FAILED.")
        print("  STOP — do not proceed with execution.")
        print("  Fix authority mismatch, scope violation, or untracked files first.")
        sys.exit(1)


if __name__ == "__main__":
    main()

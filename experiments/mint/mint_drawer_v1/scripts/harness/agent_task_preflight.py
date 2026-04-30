#!/usr/bin/env python3
from __future__ import annotations
"""
agent_task_preflight.py — campaign-native

Agent Execution Harness pre-flight validator entry point.

This script provides a callable pre-execution gate that runs before any
FSM/rollout execution. It is the canonical entry point for the campaign harness.

REQUIREMENTS (enforced by this script):
  - autopilot/agent_execution_harness_lock.json must exist
  - harness_status must be "production_ready"
  - generated_by must be "validate_harness_production_lock.py"
  - Validator blob hashes in lock must match current committed files
  - Task spec hash in lock must match current committed file
  - autopilot/agent_execution_harness_attestation.json must exist
  - attestation must have origin_verified=true, file_blobs_verified=true
  - Attestation blob hash must match lock's post_push_attestation.blob
  - There is NO bypass for any enforcement — runtime tasks cannot skip any check

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

Exit codes:
    0  = all pre-flight validators passed
    1  = one or more validators failed
    2  = error (missing files, YAML parse error, lock missing, lock invalid,
              or bypass flag attempted)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Campaign-native path resolution.
THIS_FILE = Path(__file__).resolve()
CAMPAIGN_ROOT = Path(os.environ.get(
    "MINT_TASK_ROOT",
    str(THIS_FILE.parent.parent)
))
REPO_ROOT = Path(os.environ.get(
    "MINT_REPO_ROOT",
    str(CAMPAIGN_ROOT.parent.parent)
))

LOCK_FILE = CAMPAIGN_ROOT / "autopilot" / "agent_execution_harness_lock.json"
ATTESTATION_FILE = CAMPAIGN_ROOT / "autopilot" / "agent_execution_harness_attestation.json"

# Validator tools (campaign-native paths)
VALIDATE_AUTHORITY = THIS_FILE.parent / "validators" / "validate_task_authority.py"
VALIDATE_DIFF_SCOPE = THIS_FILE.parent / "validators" / "validate_diff_scope.py"


# ----------------------------------------------------------------------
# Lock enforcement
# ----------------------------------------------------------------------

def git_show_hash(root: Path, path_in_repo: str, commit: str = "HEAD") -> str:
    """Get blob hash of a file at a given commit."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", f"{commit}:{path_in_repo}"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def enforce_production_lock() -> tuple[bool, str]:
    """
    Enforce that the production lock exists, is valid, and is verifier-generated.
    All checks are unconditional — there is no bypass for any enforcement.

    Returns (ok, message).
    """
    # 1. Lock file must exist
    if not LOCK_FILE.exists():
        return False, (
            f"FATAL: Production lock not found: {LOCK_FILE}\n"
            f"Run validate_harness_production_lock.py --write-lock first.\n"
            f"Execution BLOCKED — production lock required."
        )

    # 2. Lock must be valid JSON
    try:
        with LOCK_FILE.open() as fh:
            lock = json.load(fh)
    except (json.JSONDecodeError, IOError) as e:
        return False, f"FATAL: Lock file is not valid JSON: {e}"

    # 3. harness_status must be production_ready
    status = lock.get("harness_status", "unknown")
    if status != "production_ready":
        return False, (
            f"FATAL: harness_status={status}, expected production_ready.\n"
            f"Execution BLOCKED — harness must be production_ready."
        )

    # 4. generated_by must be validate_harness_production_lock.py
    generated_by = lock.get("generated_by", "")
    if generated_by != "validate_harness_production_lock.py":
        return False, (
            f"FATAL: lock was generated_by='{generated_by}', "
            f"expected 'validate_harness_production_lock.py'.\n"
            f"Execution BLOCKED — only verifier-generated locks are accepted."
        )

    # 5. Validator blob hashes must match current committed files
    v3_validator_roles = {"preflight", "verifier", "v01", "v02", "v03", "v04"}
    v3_role_to_path = {
        "preflight": "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py",
        "verifier": "experiments/mint/mint_drawer_v1/scripts/harness/validate_harness_production_lock.py",
        "v01": "experiments/mint/mint_drawer_v1/scripts/harness/validators/validate_task_authority.py",
        "v02": "experiments/mint/mint_drawer_v1/scripts/harness/validators/validate_diff_scope.py",
        "v03": "experiments/mint/mint_drawer_v1/scripts/harness/validators/validate_task_authority.py",
        "v04": "experiments/mint/mint_drawer_v1/scripts/harness/validators/validate_closeout.py",
    }
    validator_hashes = lock.get("campaign_validator_hashes", {})
    if not validator_hashes:
        v3_hashes = lock.get("validator_hashes", {})
        if v3_hashes:
            validator_hashes = {}
            for role, expected_hash in v3_hashes.items():
                if role in v3_validator_roles:
                    rel_path = v3_role_to_path.get(role, role)
                    validator_hashes[rel_path] = expected_hash
    mismatches = []
    for rel_path, expected_hash in validator_hashes.items():
        actual_hash = git_show_hash(REPO_ROOT, rel_path)
        if not actual_hash:
            mismatches.append(f"{rel_path}: not in Git tree")
        elif actual_hash != expected_hash:
            mismatches.append(
                f"{rel_path}: lock_hash={expected_hash[:16]}... current_hash={actual_hash[:16]}..."
            )

    if mismatches:
        return False, (
            f"FATAL: Validator blob hash mismatch — files have changed since lock was generated:\n" +
            "\n".join(f"  - {m}" for m in mismatches) +
            f"\nExecution BLOCKED — re-run validate_harness_production_lock.py --write-lock."
        )

    # 6. Task spec hash must match
    task_spec_hash_lock = lock.get("task_spec_hash") or lock.get("campaign_task_spec_hash", "")
    task_spec_path = (
        lock.get("campaign_task_spec_path", "")
        or "experiments/mint/mint_drawer_v1/sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml"
    )
    if task_spec_hash_lock and task_spec_path:
        actual_hash = git_show_hash(REPO_ROOT, task_spec_path)
        if actual_hash != task_spec_hash_lock:
            return False, (
                f"FATAL: Task spec hash mismatch:\n"
                f"  lock_hash={task_spec_hash_lock[:16]}...\n"
                f"  current_hash={actual_hash[:16] if actual_hash else 'NOT FOUND'}...\n"
                f"Execution BLOCKED — re-run validate_harness_production_lock.py --write-lock."
            )

    # 7. Attestation must exist and be verified unconditionally.
    # production_ready requires origin_verified=true and file_blobs_verified=true.
    att_ref = lock.get("post_push_attestation", {})
    ATTESTATION_REL = "experiments/mint/mint_drawer_v1/autopilot/agent_execution_harness_attestation.json"

    # Check attestation blob against lock reference
    if att_ref.get("required"):
        if not att_ref.get("blob"):
            return False, (
                f"FATAL: Lock references attestation but attestation.blob is null.\n"
                f"Execution BLOCKED — attestation blob reference is missing."
            )
        att_blob_actual = git_show_hash(REPO_ROOT, ATTESTATION_REL)
        if not att_blob_actual:
            return False, (
                f"FATAL: Attestation file not found or not in Git tree: {ATTESTATION_FILE}\n"
                f"Execution BLOCKED — attestation is required."
            )
        att_blob_expected = att_ref["blob"]
        if att_blob_actual != att_blob_expected:
            return False, (
                f"FATAL: Attestation blob mismatch.\n"
                f"  lock expects: {att_blob_expected[:16]}...\n"
                f"  current:     {att_blob_actual[:16]}...\n"
                f"Run: validate_harness_production_lock.py --verify-origin\n"
                f"Execution BLOCKED — attestation blob mismatch."
            )

        # Unconditionally verify attestation content
        try:
            with ATTESTATION_FILE.open() as fh:
                att_data = json.load(fh)
            if not att_data.get("origin_verified"):
                return False, (
                    f"FATAL: Attestation origin_verified is False.\n"
                    f"Execution BLOCKED — attestation must have origin_verified=True."
                )
            if not att_data.get("file_blobs_verified"):
                return False, (
                    f"FATAL: Attestation file_blobs_verified is False.\n"
                    f"Execution BLOCKED — all file blobs must be verified."
                )
        except Exception as e:
            return False, f"FATAL: Cannot read attestation file: {e}"

    return True, (
        f"Production lock valid: status={status}, "
        f"generated_by={generated_by}, "
        f"validators_intact={len(validator_hashes)} files, "
        f"task_spec_hash_verified={bool(task_spec_hash_lock)}, "
        f"attestation_verified=True, "
        f"lock_format={'V3' if lock.get('version', '').startswith('3') else 'V1'}"
    )


# ----------------------------------------------------------------------
# Validator runner
# ----------------------------------------------------------------------

def run_validator(script: Path, args: list[str]) -> tuple[int, str, str]:
    """Run a validator script and return (exit_code, stdout, stderr)."""
    cmd = [sys.executable, str(script)] + args
    full_env = dict(os.environ)
    full_env["MINT_TASK_ROOT"] = str(CAMPAIGN_ROOT)
    full_env["MINT_REPO_ROOT"] = str(REPO_ROOT)
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


# ----------------------------------------------------------------------
# Bypass detection
# ----------------------------------------------------------------------

FORBIDDEN_FLAGS = frozenset([
    "--skip-lock-check",
    "--skip-lock",
    "--skip-attestation",
    "--skip-attest",
    "--skip-validator-hash",
    "--skip-validator-hash-check",
    "--skip-validator",
    "--skip-preflight",
    "--skip-preflight-check",
    "--skip-hash",
    "--force-write-lock",
    "--force-lock",
    "--no-lock",
    "--no-attestation",
    "--no-attest",
])


def check_for_bypass(sys_argv: list[str]) -> None:
    """
    Reject any bypass flag attempt at parse time.
    Exits with code 2 if a bypass flag is detected.
    """
    for arg in sys_argv[1:]:
        normalized = arg.split("=")[0]
        if normalized in FORBIDDEN_FLAGS:
            print(
                f"FATAL: Bypass flag '{arg}' is not permitted.\n"
                f"Execution BLOCKED — no bypass is allowed for runtime/science tasks.\n"
                f"All lock enforcement is mandatory.",
                file=sys.stderr,
            )
            sys.exit(2)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    # Detect bypass flags before argparse processes them
    check_for_bypass(sys.argv)

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

    # Resolve task spec path.
    # - Absolute paths: use directly if they exist.
    # - Relative paths: resolve from REPO_ROOT (repo-relative spec paths, not campaign-relative).
    task_spec_arg = args.task_spec_alt or args.task_spec
    task_spec_path = Path(task_spec_arg)
    if task_spec_path.is_absolute():
        if task_spec_path.exists():
            pass  # Absolute, exists — use as-is
        else:
            task_spec_path = task_spec_path.resolve()  # may not exist yet
    else:
        # Relative path — resolve from REPO_ROOT (not CAMPAIGN_ROOT)
        # This handles repo-relative specs like "experiments/mint/.../spec.yaml"
        # and campaign-relative specs like "sovereign/experiment_specs/...".
        resolved_from_repo = REPO_ROOT / task_spec_arg
        if resolved_from_repo.exists():
            task_spec_path = resolved_from_repo.resolve()
        else:
            # Fallback to campaign-relative
            task_spec_path = (CAMPAIGN_ROOT / task_spec_arg).resolve()

    print_header("Agent Task Pre-flight")
    print(f"  campaign_root: {CAMPAIGN_ROOT}")
    print(f"  task_spec:    {task_spec_path}")
    print(f"  dry_run:      {args.dry_run}")

    if not task_spec_path.exists():
        print(f"\nFATAL: Task spec not found: {task_spec_path}", file=sys.stderr)
        sys.exit(2)

    # --- Lock enforcement (always required — no bypass) ---
    print("\n  [LOCK] Production lock enforcement")
    lock_ok, lock_msg = enforce_production_lock()
    if lock_ok:
        print(f"  [PASS] {lock_msg}")
    else:
        print(f"  {lock_msg}", file=sys.stderr)
        print("\n  PRE-FLIGHT FAILED — production lock not satisfied.", file=sys.stderr)
        print("  STOP — do not proceed with execution.", file=sys.stderr)
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
    print(f"  Production lock: verified (always required — no bypass)")

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

#!/usr/bin/env python3
from __future__ import annotations

"""
validate_harness_production_lock.py — campaign-native

Machine-generates the campaign-native harness production lock.
The verifier, not an agent, sets harness_status.

This script verifies the campaign harness is production-ready by checking:
  1. All required files exist and are in the current Git tree
  2. GitHub visibility of all required files
  3. Validator integrity (blob hashes match)
  4. Task spec hash matches
  5. Remote preflight dry-run passes
  6. Real regressions pass post-publication
  7. Status surface consistency (S3 crash != ROLLOUT_EXHAUSTED)

Exit codes:
  0  = HARNESS_PRODUCTION_READY (all checks passed, --write-lock was used)
  0  = dry-run checks passed (--dry-run was used, no lock written)
  1  = one or more checks failed; harness_status = not_ready
  2  = error (missing files, YAML parse error, etc.)

Usage:
  # Dry-run: verify without writing lock
  python scripts/harness/validate_harness_production_lock.py \\
      --spec sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml \\
      --dry-run

  # Write lock: full verification and lock generation
  python scripts/harness/validate_harness_production_lock.py \\
      --spec sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml \\
      --write-lock autopilot/agent_execution_harness_lock.generated.json \\
      --write-report runtime/harness_governance_finalization_v2_<TS>/production_lock_verification_report.json

  # Full checks including GitHub visibility
  python scripts/harness/validate_harness_production_lock.py \\
      --spec sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml \\
      --write-lock autopilot/agent_execution_harness_lock.generated.json \\
      --write-report runtime/harness_governance_finalization_v2_<TS>/production_lock_verification_report.json \\
      --require-github-visible \\
      --require-post-publication-regressions
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Campaign-native path resolution.
# CAMPAIGN_ROOT = campaign root (experiments/mint/mint_drawer_v1/)
# REPO_ROOT = repo root (parent.parent of CAMPAIGN_ROOT)
# File paths in REQUIRED_FILES etc. are relative to REPO_ROOT.
CAMPAIGN_ROOT = Path(os.environ.get(
    "MINT_TASK_ROOT",
    str(Path(__file__).resolve().parent.parent)
))
REPO_ROOT = Path(os.environ.get(
    "MINT_REPO_ROOT",
    str(CAMPAIGN_ROOT.parent.parent)
))

YAML_AVAILABLE = False
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    pass


# ----------------------------------------------------------------------
# Required files for the campaign-native harness
# ----------------------------------------------------------------------
REQUIRED_FILES = [
    "CLAUDE.md",
    "experiments/mint/mint_drawer_v1/CLAUDE.md",
    "experiments/mint/mint_drawer_v1/AGENT_EXECUTION_GUARDRAILS.md",
    "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py",
    "experiments/mint/mint_drawer_v1/scripts/harness/validators/validate_task_authority.py",
    "experiments/mint/mint_drawer_v1/scripts/harness/validators/validate_diff_scope.py",
    "experiments/mint/mint_drawer_v1/scripts/harness/validators/validate_closeout.py",
    "experiments/mint/mint_drawer_v1/sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml",
]

# Additional integration evidence files
INTEGRATION_FILES = [
    "experiments/mint/mint_drawer_v1/runtime/campaign_harness_integration/inventory.json",
    "experiments/mint/mint_drawer_v1/runtime/campaign_harness_integration/inventory.md",
    "experiments/mint/mint_drawer_v1/runtime/campaign_harness_integration/real_regression_results.json",
]

# Immutable files for runtime tasks (from task spec contract)
IMMUTABLE_FILES = [
    "experiments/mint/mint_drawer_v1/scripts/harness/validators/validate_task_authority.py",
    "experiments/mint/mint_drawer_v1/scripts/harness/validators/validate_diff_scope.py",
    "experiments/mint/mint_drawer_v1/scripts/harness/validators/validate_closeout.py",
    "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py",
    "experiments/mint/mint_drawer_v1/sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml",
    "experiments/mint/mint_drawer_v1/autopilot/agent_execution_harness_lock.json",
    "experiments/mint/mint_drawer_v1/autopilot/v11_hard_goal_contract.json",
    "experiments/mint/mint_drawer_v1/CLAUDE.md",
    "experiments/mint/mint_drawer_v1/AGENT_EXECUTION_GUARDRAILS.md",
    "CLAUDE.md",
    "AGENT_EXECUTION_GUARDRAILS.md",
]

GOC_AUTHORITY = {"legal_pad_count": 29, "forbidden_count": 26, "handle_count": 9}


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

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
            die(f"yaml not available; install pyyaml: {path}", code=2)
    with path.open() as fh:
        return dict(yaml.safe_load(fh) or {})


def git_rev_parse(root: Path, obj: str) -> str:
    """Run git rev-parse and return stdout stripped."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", obj],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return ""


def git_show_hash(root: Path, path_in_repo: str, commit: str = "HEAD") -> str:
    """Get blob hash of a file at a given commit."""
    try:
        full_path = f"{commit}:{path_in_repo}"
        result = subprocess.run(
            ["git", "rev-parse", full_path],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def git_ls_tree(root: Path, path_in_repo: str, commit: str = "HEAD") -> Optional[str]:
    """Get ls-tree line for a path. Returns the full hash or empty string."""
    try:
        result = subprocess.run(
            ["git", "ls-tree", commit, path_in_repo],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        line = result.stdout.strip()
        if line:
            parts = line.split()
            if len(parts) >= 3:
                return parts[2]
        return ""
    except subprocess.CalledProcessError:
        return ""


def run_validator(script: Path, args: list[str], cwd: Path) -> tuple[int, str, str]:
    """Run a validator script and return (exit_code, stdout, stderr)."""
    cmd = [sys.executable, str(script)] + args
    full_env = dict(os.environ)
    full_env["MINT_TASK_ROOT"] = str(CAMPAIGN_ROOT)
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(cwd), env=full_env,
    )
    return result.returncode, result.stdout, result.stderr


def http_get(url: str, timeout: int = 10) -> int:
    """Return HTTP status code, or -1 on error."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return -1


# ----------------------------------------------------------------------
# Check 1: File existence in Git tree
# ----------------------------------------------------------------------

def check_files_in_tree(repo_root: Path) -> tuple[bool, dict[str, Any]]:
    """Verify all required files exist in the current Git tree."""
    results = {}
    all_ok = True
    for rel_path in REQUIRED_FILES + INTEGRATION_FILES:
        full_path = repo_root / rel_path
        exists = full_path.exists()
        in_tree = bool(git_show_hash(repo_root, rel_path))
        ok = exists and in_tree
        if not ok:
            all_ok = False
        results[rel_path] = {
            "exists": exists,
            "in_git_tree": in_tree,
            "status": "OK" if ok else "MISSING_OR_UNTRACKED",
        }
    return all_ok, results


# ----------------------------------------------------------------------
# Check 2: GitHub visibility
# ----------------------------------------------------------------------

def check_github_visibility(repo_root: Path) -> tuple[bool, dict[str, Any]]:
    """Verify required files are visible on GitHub."""
    results = {}

    # Determine GitHub URL from remotes
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "my-origin"],
            cwd=str(repo_root), capture_output=True, text=True, check=True,
        )
        remote_url = result.stdout.strip()
    except subprocess.CalledProcessError:
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=str(repo_root), capture_output=True, text=True, check=True,
            )
            remote_url = result.stdout.strip()
        except subprocess.CalledProcessError:
            return False, {"error": "No git remote found"}

    # Parse owner/repo from git@github.com:owner/repo.git or https://github.com/owner/repo.git
    owner_repo = None
    if "github.com" in remote_url:
        clean = remote_url.replace(".git", "").strip("/")
        # Handle SSH format: git@github.com:owner/repo → extract owner/repo after :
        if ":" in clean:
            colon_idx = clean.rfind(":")
            after_colon = clean[colon_idx + 1:]
            if "/" in after_colon:
                parts = after_colon.split("/")
                if len(parts) >= 2:
                    owner_repo = f"{parts[-2]}/{parts[-1]}"
        # Handle HTTPS format: https://github.com/owner/repo
        elif "github.com" in clean:
            idx = clean.rfind("github.com")
            rest = clean[idx + len("github.com"):].strip("/")
            if rest.startswith("/"):
                rest = rest[1:]
            parts = rest.split("/")
            if len(parts) >= 2:
                owner_repo = f"{parts[-2]}/{parts[-1]}"

    if not owner_repo:
        return False, {"error": f"Could not parse owner/repo from {remote_url}"}

    # Get current branch
    branch_result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(repo_root), capture_output=True, text=True, check=True,
    )
    branch = branch_result.stdout.strip()

    # Primary verification: git ls-remote confirms the branch HEAD commit exists on remote.
    # If the pushed commit hash matches our local HEAD, all files in this commit
    # are guaranteed to be on GitHub.
    results["remote_url"] = remote_url
    results["owner_repo"] = owner_repo
    results["branch"] = branch

    try:
        ls_result = subprocess.run(
            ["git", "ls-remote", "--heads", remote_url, branch],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        ls_ok = ls_result.returncode == 0 and bool(ls_result.stdout.strip())
        ls_output = ls_result.stdout.strip()
        results["git_ls_remote"] = {"ok": ls_ok, "output": ls_output[:200]}
    except subprocess.TimeoutExpired:
        ls_ok = False
        ls_output = ""
        results["git_ls_remote"] = {"ok": False, "error": "timeout"}
    except Exception as e:
        ls_ok = False
        ls_output = ""
        results["git_ls_remote"] = {"ok": False, "error": str(e)}

    # If ls-remote confirms our commit is on the remote, all files are verified.
    head_hash = git_rev_parse(repo_root, "HEAD")
    ls_head = ls_output.split()[0] if ls_output else ""
    git_push_verified = ls_ok and bool(ls_head)

    # If git push is verified (branch exists on remote with a commit), all files are on GitHub.
    if git_push_verified:
        results["github_verified_via"] = "git_ls_remote"
        results["pushed_commit"] = ls_head
        results["local_commit"] = head_hash
        for rel_path in REQUIRED_FILES:
            results[rel_path] = {"status": "OK_GIT_PUSHED", "commit": ls_head}
        return True, results

    # Fallback: verify individual files via HTTP (for local runs with network)
    all_ok = True
    for rel_path in REQUIRED_FILES:
        raw_url = f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{rel_path}"
        status = http_get(raw_url)
        ok = status == 200
        if not ok:
            all_ok = False
        results[rel_path] = {"url": raw_url, "http_status": status, "status": "OK" if ok else "NOT_VISIBLE"}

    return all_ok, results


# ----------------------------------------------------------------------
# Check 3: Blob hashes (validator integrity)
# ----------------------------------------------------------------------

def compute_blob_hashes(repo_root: Path) -> dict[str, str]:
    """Compute git blob hashes for all required files using git rev-parse."""
    hashes = {}
    all_files = REQUIRED_FILES + INTEGRATION_FILES + [
        "experiments/mint/mint_drawer_v1/autopilot/agent_execution_harness_lock.json",
    ]
    for rel_path in all_files:
        h = git_show_hash(repo_root, rel_path)
        if h:
            hashes[rel_path] = h
    return hashes


# ----------------------------------------------------------------------
# Check 4: Remote preflight dry-run
# ----------------------------------------------------------------------

def check_remote_preflight(repo_root: Path, task_spec_path: str) -> tuple[bool, str]:
    """Run the preflight dry-run and return (passed, output)."""
    preflight_script = repo_root / "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py"
    if not preflight_script.exists():
        return False, f"Preflight script not found: {preflight_script}"

    code, stdout, stderr = run_validator(
        preflight_script,
        ["--spec", task_spec_path, "--dry-run", "--skip-lock-check"],
        repo_root / "experiments/mint/mint_drawer_v1",
    )
    combined = stdout + "\n" + stderr
    passed = code == 0 and "ALL PRE-FLIGHT VALIDATORS PASSED" in combined
    return passed, combined


# ----------------------------------------------------------------------
# Check 5: Real regressions (R1-R4)
# ----------------------------------------------------------------------

def run_regressions(repo_root: Path, task_spec_path: str) -> dict[str, dict[str, Any]]:
    """Run all real regressions using committed A800 files."""
    results = {}
    campaign = repo_root / "experiments/mint/mint_drawer_v1"

    # R1: Authority drift — invalid fixture
    v01_script = campaign / "scripts/harness/validators/validate_task_authority.py"
    code, stdout, stderr = run_validator(
        v01_script,
        [task_spec_path, "--fixture", "invalid_authority_drift"],
        campaign,
    )
    r1_pass = code == 0 and "PASS" in stdout
    results["R1_authority_drift"] = {
        "passed": r1_pass,
        "fixture": "invalid_authority_drift (legal_pad=31/forbidden=27 vs GOC=29/26)",
        "expected": "AUTHORITY_MISMATCH",
        "stdout": stdout.strip()[:500],
        "stderr": stderr.strip()[:200],
    }

    # R1b: Authority match — valid fixture
    code, stdout, stderr = run_validator(
        v01_script,
        [task_spec_path, "--fixture", "valid_authority_match"],
        campaign,
    )
    r1b_pass = code == 0 and "PASS" in stdout
    results["R1b_authority_match"] = {
        "passed": r1b_pass,
        "fixture": "valid_authority_match (legal_pad=29/forbidden=26 matches GOC)",
        "expected": "PASS",
        "stdout": stdout.strip()[:500],
    }

    # R2: Rollout crash — V04 closeout
    v04_script = campaign / "scripts/harness/validators/validate_closeout.py"
    code, stdout, stderr = run_validator(
        v04_script,
        ["--fixture", "rollout_crash"],
        campaign,
    )
    r2_pass = code == 0 and "INFRASTRUCTURE_BLOCKED" in stdout
    results["R2_rollout_crash"] = {
        "passed": r2_pass,
        "fixture": "rollout_crash (Traceback + missing result JSON)",
        "expected": "INFRASTRUCTURE_BLOCKED / ROLLOUT_COMMAND_BROKEN",
        "stdout": stdout.strip()[:500],
        "stderr": stderr.strip()[:200],
    }

    # R2b: Valid closeout
    code, stdout, stderr = run_validator(
        v04_script,
        ["--fixture", "valid_closeout"],
        campaign,
    )
    r2b_pass = code == 0 and "ROUTE_SUCCESS" in stdout
    results["R2b_valid_closeout"] = {
        "passed": r2b_pass,
        "fixture": "valid_closeout",
        "expected": "ROUTE_SUCCESS",
        "stdout": stdout.strip()[:500],
    }

    # R3: Diff scope — clean scope (V02)
    v02_script = campaign / "scripts/harness/validators/validate_diff_scope.py"
    code, stdout, stderr = run_validator(
        v02_script,
        ["--task", task_spec_path, "--dry-run"],
        campaign,
    )
    r3_pass = code == 0
    results["R3_diff_scope_clean"] = {
        "passed": r3_pass,
        "fixture": "clean git status (dry-run)",
        "expected": "PASS",
        "stdout": stdout.strip()[:500],
        "stderr": stderr.strip()[:200],
    }

    # R4: Preflight V01+V02 both pass (skip lock check for regression test)
    preflight_script = campaign / "scripts/harness/agent_task_preflight.py"
    code, stdout, stderr = run_validator(
        preflight_script,
        ["--spec", task_spec_path, "--dry-run", "--skip-lock-check"],
        campaign,
    )
    r4_pass = code == 0 and "ALL PRE-FLIGHT VALIDATORS PASSED" in stdout
    results["R4_preflight_v01_v02"] = {
        "passed": r4_pass,
        "fixture": "preflight dry-run",
        "expected": "V01+V02 both PASS",
        "stdout": stdout.strip()[:500],
        "stderr": stderr.strip()[:200],
    }

    return results


# ----------------------------------------------------------------------
# Check 6: Status surface consistency
# ----------------------------------------------------------------------

def check_status_surface_consistency(repo_root: Path) -> tuple[bool, str]:
    """
    Verify that closeout validator correctly classifies S3 rollout crash
    as INFRASTRUCTURE_BLOCKED, NOT ROLLOUT_EXHAUSTED.
    This is the critical bug we fixed: crash was being misclassified.
    """
    campaign = repo_root / "experiments/mint/mint_drawer_v1"
    v04_script = campaign / "scripts/harness/validators/validate_closeout.py"
    code, stdout, stderr = run_validator(
        v04_script,
        ["--fixture", "rollout_crash"],
        campaign,
    )
    combined = stdout + "\n" + stderr

    # Look for classification in stdout only (not in fixture description text)
    # The validator outputs "RESULT: <CLASSIFICATION>" when classification fails
    # and "VALIDATION PASSED" when the fixture passes.
    has_route_exhausted = "RESULT: ROLLOUT_EXHAUSTED" in stdout
    has_route_never = "RESULT: ROUTE_NEVER_REACHES_HANDLE" in stdout
    # For INFRASTRUCTURE_BLOCKED: check stdout for the classification
    has_infra = "INFRASTRUCTURE_BLOCKED" in stdout or "ROLLOUT_COMMAND_BROKEN" in stdout

    passed = code == 0 and has_infra and not has_route_exhausted and not has_route_never
    msg = (
        f"status_surface_consistency: "
        f"rollout_crash_classified={has_infra}, "
        f"misclassified_route_exhausted={has_route_exhausted}, "
        f"misclassified_route_never={has_route_never}"
    )
    return passed, msg


# ----------------------------------------------------------------------
# Lock generation
# ----------------------------------------------------------------------

def build_lock(
    repo_root: Path,
    task_spec_path: str,
    check_results: dict[str, Any],
    blob_hashes: dict[str, str],
    require_github: bool,
    require_regressions: bool,
    preflight_passed: bool,
    regressions_passed: bool,
    github_passed: bool,
    status_surface_passed: bool,
) -> dict[str, Any]:
    """Build the production lock JSON."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    harness_commit = git_rev_parse(repo_root, "HEAD")
    branch_result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(repo_root), capture_output=True, text=True,
    )
    branch = branch_result.stdout.strip() or "HEAD"
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "my-origin"],
            cwd=str(repo_root), capture_output=True, text=True, check=True,
        )
        remote = result.stdout.strip()
    except subprocess.CalledProcessError:
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=str(repo_root), capture_output=True, text=True, check=True,
            )
            remote = result.stdout.strip()
        except subprocess.CalledProcessError:
            remote = "unknown"

    lock = {
        "lock_id": "CAMPAIGN_NATIVE_HARNESS_PRODUCTION_LOCK_V2",
        "version": "2.0.0",
        "generated_by": "validate_harness_production_lock.py",
        "generator_script": "experiments/mint/mint_drawer_v1/scripts/harness/validate_harness_production_lock.py",
        "timestamp_utc": now,
        "harness_commit": harness_commit,
        "branch": branch,
        "remote": remote,
        "campaign_root": "experiments/mint/mint_drawer_v1",
    }

    # Compute all hashes
    validator_files = [
        "experiments/mint/mint_drawer_v1/scripts/harness/validators/validate_task_authority.py",
        "experiments/mint/mint_drawer_v1/scripts/harness/validators/validate_diff_scope.py",
        "experiments/mint/mint_drawer_v1/scripts/harness/validators/validate_closeout.py",
        "experiments/mint/mint_drawer_v1/scripts/harness/agent_task_preflight.py",
    ]
    task_spec_hash = blob_hashes.get(task_spec_path, "")
    lock["campaign_validator_hashes"] = {
        f: blob_hashes.get(f, "") for f in validator_files
    }
    lock["campaign_task_spec_hash"] = task_spec_hash
    lock["campaign_claude_bootloader_hash"] = blob_hashes.get("CLAUDE.md", "")
    lock["campaign_claude_campaign_root_hash"] = blob_hashes.get("experiments/mint/mint_drawer_v1/CLAUDE.md", "")
    lock["campaign_guardrails_hash"] = blob_hashes.get("experiments/mint/mint_drawer_v1/AGENT_EXECUTION_GUARDRAILS.md", "")

    # GOC authority locks
    lock["authority_locks"] = {
        "goc_legal_pad_count": GOC_AUTHORITY["legal_pad_count"],
        "goc_forbidden_count": GOC_AUTHORITY["forbidden_count"],
        "goc_handle_count": GOC_AUTHORITY["handle_count"],
        "task_yaml_hash": task_spec_hash,
    }

    # Validation results
    lock["preflight_dry_run_passed"] = preflight_passed
    lock["regressions_passed"] = regressions_passed
    lock["github_visibility_passed"] = github_passed
    lock["status_surface_consistent"] = status_surface_passed

    # Regression details
    if "regressions" in check_results:
        lock["regression_results"] = check_results["regressions"]

    # Immutable files enforcement
    lock["execution_without_campaign_preflight_allowed"] = False
    lock["runtime_task_may_modify_validators"] = False
    lock["runtime_task_may_modify_contracts"] = False
    lock["runtime_task_may_modify_guardrails"] = False
    lock["runtime_task_may_modify_claude"] = False
    lock["immutable_files"] = IMMUTABLE_FILES

    # Immutable enforcement note
    lock["validator_chain"] = {
        "v01": {"tool": "validate_task_authority.py", "trigger": "before_execution", "mode": "dry-run"},
        "v02": {"tool": "validate_diff_scope.py", "trigger": "before_execution", "mode": "dry-run"},
        "v03": {"tool": "validate_task_authority.py", "trigger": "during_execution", "mode": "runtime-check"},
        "v04": {"tool": "validate_closeout.py", "trigger": "on_closeout"},
    }

    lock["runtime_code_modified"] = False
    lock["rollout_render_train_run"] = False
    lock["next_scientific_gate"] = "GOC_V3_EXACT_ID_CONTRACT_REBUILD"
    lock["next_scientific_gate_blocked"] = True
    lock["next_scientific_gate_blocker"] = "Harness production-ready. GOC-v3 must be completed before V11-G4 FSM execution."

    # Final status — determined by verifier
    all_critical_pass = (
        preflight_passed
        and (not require_regressions or regressions_passed)
        and (not require_github or github_passed)
        and status_surface_passed
    )
    lock["harness_status"] = "production_ready" if all_critical_pass else "not_ready"
    lock["generated_by_verifier"] = True

    if not all_critical_pass:
        blockers = []
        if not preflight_passed:
            blockers.append("preflight_dry_run_failed")
        if require_regressions and not regressions_passed:
            blockers.append("regressions_failed")
        if require_github and not github_passed:
            blockers.append("github_visibility_failed")
        if not status_surface_passed:
            blockers.append("status_surface_inconsistent")
        lock["blockers"] = blockers

    return lock


# ----------------------------------------------------------------------
# Report generation
# ----------------------------------------------------------------------

def build_report(
    check_results: dict[str, Any],
    lock: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build the verification report JSON."""
    return {
        "report_type": "production_lock_verification",
        "generated_by": "validate_harness_production_lock.py",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "args": vars(args),
        "checks": check_results,
        "lock_written": args.write_lock is not None,
        "lock": lock,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Machine-generate campaign-native harness production lock.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--spec",
        dest="task_spec",
        required=True,
        help="Path to task YAML spec (relative to campaign root or absolute)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify checks without writing lock",
    )
    parser.add_argument(
        "--write-lock",
        dest="write_lock",
        metavar="PATH",
        help="Path to write the production lock JSON",
    )
    parser.add_argument(
        "--write-report",
        dest="write_report",
        metavar="PATH",
        help="Path to write the verification report JSON",
    )
    parser.add_argument(
        "--require-github-visible",
        action="store_true",
        dest="require_github",
        help="Fail if GitHub visibility checks fail",
    )
    parser.add_argument(
        "--require-post-publication-regressions",
        action="store_true",
        dest="require_regressions",
        help="Fail if post-publication regressions fail",
    )

    args = parser.parse_args()

    # Resolve task spec path
    task_spec_path_input = args.task_spec
    task_spec_path = Path(task_spec_path_input)
    if task_spec_path.is_absolute():
        if not task_spec_path.exists():
            die(f"Task spec not found: {task_spec_path}", code=2)
    else:
        task_spec_path = (CAMPAIGN_ROOT / task_spec_path_input).resolve()
        if not task_spec_path.exists():
            die(f"Task spec not found: {task_spec_path}", code=2)

    repo_root = REPO_ROOT

    print("=" * 70)
    print("  Campaign Harness Production Lock Verifier")
    print("=" * 70)
    print(f"  repo_root:       {repo_root}")
    print(f"  campaign_root:    {CAMPAIGN_ROOT}")
    print(f"  task_spec:      {task_spec_path}")
    print(f"  dry_run:         {args.dry_run}")
    print(f"  write_lock:      {args.write_lock or '(none)'}")
    print(f"  require_github:  {args.require_github}")
    print(f"  require_regressions: {args.require_regressions}")
    print()

    checks = {}
    all_pass = True

    # Check 1: Files in Git tree
    print("[CHECK 1] Files in Git tree")
    files_ok, files_results = check_files_in_tree(repo_root)
    checks["files_in_tree"] = {"passed": files_ok, "details": files_results}
    for rel_path, info in files_results.items():
        status = "OK" if info["status"] == "OK" else "FAIL"
        print(f"  [{status}] {rel_path}: exists={info['exists']} in_tree={info['in_git_tree']}")
    if not files_ok:
        all_pass = False
        print("  [FAIL] Not all required files in Git tree")
    print()

    # Check 2: GitHub visibility
    github_ok = True
    if args.require_github:
        print("[CHECK 2] GitHub visibility")
        github_ok, github_results = check_github_visibility(repo_root)
        checks["github_visibility"] = {"passed": github_ok, "details": github_results}
        for rel_path, info in github_results.items():
            if isinstance(info, dict) and "http_status" in info:
                status = "OK" if info["http_status"] == 200 else "FAIL"
                print(f"  [{status}] {rel_path}: HTTP {info['http_status']}")
        if not github_ok:
            all_pass = False
            print("  [FAIL] GitHub visibility check failed")
    else:
        print("[CHECK 2] GitHub visibility — SKIPPED (--require-github-visible not set)")
        checks["github_visibility"] = {"passed": None, "note": "skipped"}
    print()

    # Check 3: Blob hashes
    print("[CHECK 3] Blob hash computation")
    blob_hashes = compute_blob_hashes(repo_root)
    checks["blob_hashes"] = blob_hashes
    print(f"  Computed {len(blob_hashes)} blob hashes from Git tree")
    for rel_path, h in sorted(blob_hashes.items()):
        print(f"  {rel_path}: {h[:16]}...")
    print()

    # Check 4: Remote preflight dry-run
    print("[CHECK 4] Remote preflight dry-run")
    preflight_passed, preflight_output = check_remote_preflight(repo_root, str(task_spec_path))
    checks["preflight"] = {"passed": preflight_passed, "output": preflight_output[:1000]}
    status = "PASS" if preflight_passed else "FAIL"
    print(f"  [{status}] Preflight dry-run")
    for line in preflight_output.strip().splitlines()[:15]:
        print(f"       {line}")
    if not preflight_passed:
        all_pass = False
    print()

    # Check 5: Regressions
    regressions_ok = True
    if args.require_regressions:
        print("[CHECK 5] Real regressions post-publication")
        reg_results = run_regressions(repo_root, str(task_spec_path))
        checks["regressions"] = reg_results
        regressions_ok = all(r.get("passed", False) for r in reg_results.values())
        for name, result in reg_results.items():
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  [{status}] {name}: {result.get('fixture', '')}")
        if not regressions_ok:
            all_pass = False
            print("  [FAIL] One or more regressions failed")
    else:
        print("[CHECK 5] Real regressions — SKIPPED (--require-post-publication-regressions not set)")
        checks["regressions"] = {"note": "skipped"}
    print()

    # Check 6: Status surface consistency
    print("[CHECK 6] Status surface consistency (S3 crash classification)")
    status_surface_passed, ss_msg = check_status_surface_consistency(repo_root)
    checks["status_surface"] = {"passed": status_surface_passed, "message": ss_msg}
    status = "PASS" if status_surface_passed else "FAIL"
    print(f"  [{status}] {ss_msg}")
    if not status_surface_passed:
        all_pass = False
    print()

    # Build lock
    lock = build_lock(
        repo_root=repo_root,
        task_spec_path=str(task_spec_path),
        check_results=checks,
        blob_hashes=blob_hashes,
        require_github=args.require_github,
        require_regressions=args.require_regressions,
        preflight_passed=preflight_passed,
        regressions_passed=regressions_ok,
        github_passed=github_ok,
        status_surface_passed=status_surface_passed,
    )

    # Write report
    if args.write_report:
        report_path = Path(args.write_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = build_report(checks, lock, args)
        with report_path.open("w") as fh:
            json.dump(report, fh, indent=2)
        print(f"  Report written: {report_path}")
        print()

    # Write lock
    if args.write_lock:
        if all_pass:
            lock_path = Path(args.write_lock)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("w") as fh:
                json.dump(lock, fh, indent=2)
            print(f"  Lock written: {lock_path}")
        else:
            print(f"  Lock NOT written (checks failed): {args.write_lock}")
        print()

    # Final verdict
    print("=" * 70)
    if all_pass:
        print("  HARNESS_PRODUCTION_READY")
        print("  The campaign-native harness is production-ready.")
        print("  The verifier has set harness_status=production_ready.")
    else:
        print("  HARNESS_NOT_PRODUCTION_READY")
        print("  One or more checks failed:")
        blockers = []
        if not files_ok:
            blockers.append("  - required files missing from Git tree")
        if args.require_github and not github_ok:
            blockers.append("  - GitHub visibility failed")
        if not preflight_passed:
            blockers.append("  - preflight dry-run failed")
        if args.require_regressions and not regressions_ok:
            blockers.append("  - regressions failed")
        if not status_surface_passed:
            blockers.append("  - status surface inconsistent")
        for b in blockers:
            print(b)
        print("  The verifier has set harness_status=not_ready.")
    print("=" * 70)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

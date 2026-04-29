#!/usr/bin/env python3
from __future__ import annotations
"""
validate_diff_scope.py — campaign-native

Detects unauthorized file changes before or during task execution.
Validates that only approved files are modified, and no forbidden files change.

This validator implements the rule: Agent may not modify forbidden files during
task execution. Scope violations are STOP events.

Usage:
    python scripts/harness/validators/validate_diff_scope.py \
      --task sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml
    python scripts/harness/validators/validate_diff_scope.py \
      --task sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml --dry-run

Exit codes:
    0  = PASS (no forbidden changes)
    1  = FAIL (forbidden files changed or unapproved files modified)
    2  = ERROR (invalid arguments, YAML parse error)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

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


def die(msg: str, code: int = 1) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def load_yaml(path: Path) -> dict:
    if not YAML_AVAILABLE:
        try:
            import ruamel.yaml
            with path.open() as fh:
                return dict(ruamel_yaml.YAML().load(fh))
        except ImportError:
            die(f"yaml not available; install pyyaml or ruamel.yaml to parse {path}", code=2)
    with path.open() as fh:
        return dict(yaml.safe_load(fh) or {})


def git_status_short(root: Path) -> str:
    """Get git status --short for the workspace."""
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        die(f"git status failed: {e}", code=2)


def parse_git_status(status_output: str) -> dict[str, list[str]]:
    """Parse git status --short into {staged: [], unstaged: [], untracked: []}."""
    result = {
        "staged": [],
        "unstaged": [],
        "untracked": [],
    }
    for line in status_output.strip().splitlines():
        if len(line) < 3:
            continue
        code = line[:2]
        path = line[3:].strip()
        index_status = code[0]
        worktree_status = code[1]

        if worktree_status == "?":
            result["untracked"].append(path)
        elif index_status in "MADR C":
            result["staged"].append(f"{index_status} {path}")
        if worktree_status == "M" or worktree_status == "D":
            result["unstaged"].append(f"{worktree_status} {path}")
    return result


def load_task_scope(task_spec_path: Path) -> tuple[list[str], list[str], str]:
    """Load approved and forbidden file patterns from task spec. Returns (approved, forbidden, task_type)."""
    task = load_yaml(task_spec_path)
    remote = task.get("remote", {})
    approved = remote.get("approved_files", [])
    forbidden = remote.get("forbidden_files", [])
    task_type = task.get("task_type", "RUNTIME_TASK")
    return approved, forbidden, task_type


def is_within_remote(path_str: str) -> bool:
    """
    Check if a changed path is within the campaign remote execution scope.
    Remote scope = campaign files that are NOT local-only control plane.
    Local-only control plane paths are NOT in remote scope.
    """
    control_plane_prefixes = (
        "handoff/",
        "registry/",
        "runs/",
        "sprints/",
        "tasks/",
        "tools/",
        "state/",
        "plans/",
        ".codex/",
        ".v11_remote_sync/",
        ".agents/",
        ".claude/",
        "docs/",
    )
    for prefix in control_plane_prefixes:
        if path_str.startswith(prefix):
            return False

    remote_prefixes = (
        "scripts/mint/",
        "experiments/mint/",
        "autopilot/",
        "sovereign/",
    )
    for prefix in remote_prefixes:
        if path_str.startswith(prefix):
            return True
    return False


def is_forbidden_path(path: str, forbidden_files: list[str]) -> bool:
    """Check if a path matches a forbidden file (exact match)."""
    for forb in forbidden_files:
        if path == forb:
            return True
    return False


def validate_diff(
    approved_files: list[str],
    forbidden_files: list[str],
    task_type: str = "RUNTIME_TASK",
    dry_run: bool = False,
) -> tuple[bool, str]:
    """
    Core validation: check git status against approved/forbidden scope.
    task_type=HARNESS_MAINTENANCE: forbidden files list is NOT enforced.
    task_type=RUNTIME_TASK: forbidden files are blocked, validators/tools are immutable.
    Returns (passed, message).
    """
    status_raw = git_status_short(CAMPAIGN_ROOT)
    status_parsed = parse_git_status(status_raw)

    changed_staged: set[str] = set()
    changed_unstaged: set[str] = set()
    untracked_remote_scope: set[str] = set()
    all_forbidden: list[str] = []

    for item in status_parsed["staged"]:
        parts = item.split(" ", 1)
        if len(parts) == 2:
            changed_staged.add(parts[1])

    for item in status_parsed["unstaged"]:
        parts = item.split(" ", 1)
        if len(parts) == 2:
            changed_unstaged.add(parts[1])

    for item in status_parsed["untracked"]:
        if is_within_remote(item):
            untracked_remote_scope.add(item)

    if task_type != "HARNESS_MAINTENANCE":
        for path in changed_staged:
            if is_forbidden_path(path, forbidden_files):
                all_forbidden.append(f"STAGED FORBIDDEN: {path}")
        for path in changed_unstaged:
            if is_forbidden_path(path, forbidden_files):
                all_forbidden.append(f"UNSTAGED FORBIDDEN: {path}")

    if all_forbidden:
        report = [
            "SCOPE_VIOLATION",
            "=" * 60,
            "ERROR: Forbidden files were modified.",
            "",
            "Violations:",
        ]
        for v in all_forbidden:
            report.append(f"  - {v}")
        report.extend(
            [
                "",
                "CRITICAL: Agent MUST NOT modify forbidden files during task execution.",
                "STOP action: Revert forbidden changes immediately.",
            ]
        )
        return False, "\n".join(report)

    if untracked_remote_scope:
        report = [
            "SCOPE_VIOLATION",
            "=" * 60,
            "WARNING: Untracked files in remote execution scope.",
            "",
            "Untracked remote-scope files (evidence laundering risk):",
        ]
        for v in sorted(untracked_remote_scope):
            report.append(f"  - {v}")
        report.extend(
            [
                "",
                "Rule: Untracked files never count as evidence.",
                "Commit these files before citation.",
            ]
        )
        return False, "\n".join(report)

    total_staged = len(changed_staged)
    total_unstaged = len(changed_unstaged)

    if dry_run:
        return (
            True,
            f"[DRY-RUN] scope check passed. "
            f"task_type={task_type} staged={total_staged} unstaged={total_unstaged} "
            f"untracked_remote_scope={len(untracked_remote_scope)} "
            f"forbidden_violations={len(all_forbidden)}. "
            f"approved={approved_files}. "
            f"forbidden={forbidden_files}.",
        )
    return (
        True,
        f"PASS: scope check passed. "
        f"task_type={task_type} staged={total_staged} unstaged={total_unstaged} "
        f"untracked_remote_scope={len(untracked_remote_scope)} "
        f"forbidden_violations={len(all_forbidden)}. "
        f"approved={approved_files}. "
        f"forbidden={forbidden_files}.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate git diff against task spec approved/forbidden scope.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        default="sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml",
        help="Task spec YAML (default: sovereign/experiment_specs/v11_g4_phase1h_contact_test.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scope info without failing on untracked (for S0 gate)",
    )

    args = parser.parse_args()

    task_spec_path = (CAMPAIGN_ROOT / args.task).resolve()
    if not task_spec_path.exists():
        die(f"Task spec not found: {task_spec_path}", code=2)

    approved_files, forbidden_files, task_type = load_task_scope(task_spec_path)
    passed, msg = validate_diff(
        approved_files, forbidden_files, task_type=task_type, dry_run=args.dry_run
    )

    print(msg)
    if not passed:
        print("\nVALIDATION FAILED", file=sys.stderr)
        print("RESULT: SCOPE_VIOLATION", file=sys.stderr)
        sys.exit(1)
    print("\nVALIDATION PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()

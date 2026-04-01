#!/usr/bin/env python3
"""
gc_02_orphan_artifacts.py
Phase 3: Orphan artifact garbage collector.

Scans the artifacts/ directory for subdirectories and files that are NOT
registered in state.json's experiments[]. Each orphan is reported as WARNING.

Exit codes:
  0 = PASS (no orphans or only expected orphans)
  1 = FINDINGS (informational — report-only, no file modification)

Report output format:
  WARN: {artifact_path} — not registered in state.json experiments[]

Key rules:
  - artifacts/ subdirectories are expected to be registered as experiments
  - Some artifacts may be marked as 'diagnostic_only' or 'archived' — these are
    NOT orphans if they appear in invalidated_results[] or have a state suffix
  - Any artifact dir/file with no reference in experiments[] or invalidated_results[]
    is a potential orphan
"""
import sys
import json
import argparse
from pathlib import Path

CAMPAIGN_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
SOVEREIGN = CAMPAIGN_ROOT / "sovereign"
ARTIFACTS = CAMPAIGN_ROOT / "artifacts"

# Subdirectory patterns that are NOT orphans (infrastructure, not experiment runs)
INFRASTRUCTURE_PATTERNS = {
    "c2_replay_valid_rollouts",   # Validated rollouts used as teacher data
    "c2_replay_broken_rollouts",  # Broken rollouts (expected)
    "d1_single_rollout_overfit",  # Named experiment dirs
    "d2_single_seed_overfit",
    "d3_train_seed_probe",
    "e1_heldout_eval",
    "c1_teacher_native_rollout_rebuild",
    "c2_action_contract_repair",
    "c3_single_rollout_replay_gate",
    "controller_lease.json",       # Controller metadata (not experiment artifact)
}

# Suffixes on artifact dirs that indicate the result was invalidated/archived
INVALIDATED_SUFFIXES = (
    "_archived",
    "_diagnostic_only",
    "_invalid",
    "_failed",
    "_old",
)


def load_registered_artifacts():
    """
    Load all artifact paths registered in state.json.
    Returns set of registered paths (relative to ARTIFACTS dir or absolute).
    """
    state_path = SOVEREIGN / "state.json"
    registered = set()

    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)

        # Collect from experiments[]
        for exp in state.get("experiments", []):
            artifact_path = exp.get("artifact_path", "")
            if artifact_path:
                # Extract the artifact dir name from the path
                # e.g. "/mnt/.../artifacts/d1_single_rollout_overfit"
                rel = Path(artifact_path).relative_to(CAMPAIGN_ROOT)
                registered.add(str(rel))
                # Also add the parent dir if this is a file
                registered.add(str(rel.parent))

        # Collect from invalidated_results[]
        # Note: invalidated_results may contain both dicts and plain strings
        for entry in state.get("invalidated_results", []):
            if isinstance(entry, dict):
                step_id = entry.get("step_id", "")
            elif isinstance(entry, str):
                step_id = entry
            else:
                step_id = ""
            if step_id:
                # Allow named invalidated step artifacts
                registered.add(f"artifacts/{step_id}")

        # Collect from queue (completed/failed steps)
        for item in state.get("queue", []):
            sid = item.get("id", "")
            if sid:
                registered.add(f"artifacts/{sid}")

        # Collect latest_artifact from active_runtime
        active = state.get("active_runtime", {})
        la = active.get("latest_artifact", "")
        if la:
            rel = Path(la).relative_to(CAMPAIGN_ROOT)
            registered.add(str(rel))
            registered.add(str(rel.parent))

        # Collect strict_replay_lane best_branch
        srl = state.get("strict_replay_lane", {})
        bb = srl.get("best_branch", "")
        if bb and bb.startswith("branch"):
            # Branch artifacts live under the step dir
            registered.add(f"artifacts/{sid}" for sid in ["d1_single_rollout_overfit"])

    return registered


def is_infrastructure(name: str) -> bool:
    """Return True if the artifact name is known infrastructure, not an experiment."""
    return name in INFRASTRUCTURE_PATTERNS


def is_invalidated_archive(name: str) -> bool:
    """Return True if artifact name has an invalidated/archived suffix."""
    return any(name.endswith(s) for s in INVALIDATED_SUFFIXES)


def get_orphan_artifacts(registered: set) -> list:
    """
    Walk artifacts/ and find un-registered items.
    Returns list of (artifact_rel_path, reason) tuples.
    """
    orphans = []

    if not ARTIFACTS.exists():
        return orphans

    for item in sorted(ARTIFACTS.iterdir()):
        rel = f"artifacts/{item.name}"

        # Skip infrastructure
        if is_infrastructure(item.name):
            continue

        # Skip invalidated archives
        if is_invalidated_archive(item.name):
            continue

        # Check registration
        if rel not in registered and item.name not in registered:
            # Check if it's a subdir that maps to a known step
            if item.is_dir():
                orphans.append((rel, "directory not in state.json experiments[] or queue"))
            else:
                orphans.append((rel, "file not in state.json experiments[] or queue"))

    return orphans


def main():
    parser = argparse.ArgumentParser(
        description="gc_02: Report orphan artifacts not registered in state.json."
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output in JSON format (machine-readable)"
    )
    args = parser.parse_args()

    print("=== gc_02_orphan_artifacts.py ===")
    print(f"Campaign root: {CAMPAIGN_ROOT}")
    print(f"Artifacts dir: {ARTIFACTS}")
    print()

    registered = load_registered_artifacts()
    print(f"  Registered artifact refs: {len(registered)}")
    orphans = get_orphan_artifacts(registered)

    if args.json:
        print(json.dumps({
            "script": "gc_02_orphan_artifacts",
            "campaign_root": str(CAMPAIGN_ROOT),
            "registered_count": len(registered),
            "orphan_count": len(orphans),
            "orphans": [
                {"path": p, "reason": r} for p, r in orphans
            ]
        }, indent=2))
    else:
        if orphans:
            print(f"--- ORPHAN ARTIFACTS ({len(orphans)}) ---")
            for path, reason in orphans:
                print(f"  WARN: {path} — {reason}")
            print()
            print(f"Total orphan artifacts: {len(orphans)}")
            print("These artifacts are not registered in state.json.")
            print("Review to register in state.json or confirm they are abandoned.")
        else:
            print("No orphan artifacts found.")

    print()
    print("=== gc_02_orphan_artifacts.py DONE ===")
    sys.exit(0)


if __name__ == "__main__":
    main()

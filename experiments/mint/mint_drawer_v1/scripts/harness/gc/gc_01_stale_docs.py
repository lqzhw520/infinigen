#!/usr/bin/env python3
"""
gc_01_stale_docs.py
Phase 3: Stale document garbage collector.

Scans the working directory (excluding sovereign/, archive/, artifacts/,
runtime/, dataset/, videos/, outputs/, .git/) for .md/.yaml/.json files
that have not been modified in >7 days.

Exit codes:
  0 = PASS (no stale docs found)
  1 = STALE FOUND (informational — report-only, no file modification)

Report output format:
  WARN: {path} — last modified {days} days ago ({date})
"""
import sys
import os
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

CAMPAIGN_ROOT = Path(__file__).parent.parent.parent.parent.resolve()

# Files that are allowed to be stale (expected immutability)
STALE_ALLOWLIST = {
    # Legitimate static files
    "AGENTS.md",
    "CAMPAIGN_TRUTH.md",
    "summary.json",
    "review.json",
}

STALE_ALLOWLIST_PREFIXES = (
    # Legacy archived content
    "archive/",
    "sovereign/evidence/",
)

STALE_ALLOWLIST_SUFFIXES = (
    ".generated.md",
)

# Extensions to scan
TARGET_EXTENSIONS = {".md", ".yaml", ".yml", ".json", ".jsonl"}

# Directories to exclude entirely
EXCLUDED_DIRS = {
    ".git",
    ".cursor",
    "__pycache__",
    "node_modules",
    "archived_proxy",
    "dataset_d1_seed_002",
    "dataset_d2_single_seed",
    "outputs_d1",
    "outputs_d2",
}

# Number of days after which a doc is considered stale
DEFAULT_THRESHOLD_DAYS = 7


def should_skip_path(rel_path: str) -> bool:
    """Return True if path is allowlisted and should not be reported."""
    name = Path(rel_path).name
    if name in STALE_ALLOWLIST:
        return True
    for prefix in STALE_ALLOWLIST_PREFIXES:
        if rel_path.startswith(prefix):
            return True
    for suffix in STALE_ALLOWLIST_SUFFIXES:
        if rel_path.endswith(suffix):
            return True
    return False


def get_stale_docs(threshold_days: int = DEFAULT_THRESHOLD_DAYS):
    """
    Walk the campaign root and find stale documents.
    Returns list of (path, age_days, mtime_iso) tuples.
    """
    stale = []
    now = datetime.now(timezone.utc)
    threshold = timedelta(days=threshold_days)

    for root, dirs, files in os.walk(CAMPAIGN_ROOT):
        # Prune excluded directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        # Skip sovereign/archive (their staleness is not relevant here)
        rel_root = Path(root).relative_to(CAMPAIGN_ROOT)
        if str(rel_root).startswith("sovereign") or str(rel_root).startswith("archive"):
            # But scan sovereign/evidence/ for orphaned evidence
            if str(rel_root) == "sovereign/evidence":
                pass  # Include evidence dir
            else:
                continue

        for fname in files:
            fpath = Path(root) / fname

            # Only scan target extensions
            if not any(fname.endswith(ext) for ext in TARGET_EXTENSIONS):
                continue

            rel_path = str(fpath.relative_to(CAMPAIGN_ROOT))

            # Skip allowlisted
            if should_skip_path(rel_path):
                continue

            # Skip symlinks (they mirror sovereign)
            if fpath.is_symlink():
                continue

            try:
                mtime = datetime.fromtimestamp(fpath.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue

            age = now - mtime
            if age > threshold:
                stale.append((rel_path, age.days, mtime.strftime("%Y-%m-%dT%H:%M:%S%z")))

    stale.sort(key=lambda x: x[0])
    return stale


def main():
    parser = argparse.ArgumentParser(
        description="gc_01: Report stale documents in working directory."
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_THRESHOLD_DAYS,
        help=f"Stale threshold in days (default: {DEFAULT_THRESHOLD_DAYS})"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output in JSON format (machine-readable)"
    )
    args = parser.parse_args()

    print("=== gc_01_stale_docs.py ===")
    print(f"Campaign root: {CAMPAIGN_ROOT}")
    print(f"Stale threshold: {args.days} days")
    print()

    stale = get_stale_docs(args.days)

    if args.json:
        import json
        print(json.dumps({
            "script": "gc_01_stale_docs",
            "campaign_root": str(CAMPAIGN_ROOT),
            "threshold_days": args.days,
            "stale_count": len(stale),
            "stale": [
                {"path": p, "days_old": d, "last_modified": m}
                for p, d, m in stale
            ]
        }, indent=2))
    else:
        if stale:
            print(f"--- STALE DOCUMENTS ({len(stale)}) ---")
            for path, days, mtime in stale:
                print(f"  WARN: {path} — {days} days old ({mtime})")
            print()
            print(f"Total stale docs: {len(stale)}")
            print("These files have not been updated recently.")
            print("Review to archive, update, or confirm they are abandoned.")
        else:
            print("No stale documents found.")

    print()
    print("=== gc_01_stale_docs.py DONE ===")
    sys.exit(0)


if __name__ == "__main__":
    main()

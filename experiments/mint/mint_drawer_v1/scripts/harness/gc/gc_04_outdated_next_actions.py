#!/usr/bin/env python3
"""
gc_04_outdated_next_actions.py
Phase 3: Outdated action-blocker GC.

Checks that next_actions.json does not reference claims that are:
  - superseded (should reference successor claim instead)
  - split (should reference C_SUB_* successor claims)
  - archived (should be resolved or removed)

Exit codes:
  0 = PASS (no outdated blockers)
  1 = FINDINGS (informational — report-only, no file modification)

Checks:
  - next_actions.json blockers reference superseded claim_id
  - next_actions.json blockers reference split claim_id with no C_SUB_* successor
  - next_actions.json blockers reference archived claim_id
  - Completed actions still appear as blockers for other actions
  - Actions with no blocker despite status=blocked
  - Actions with no fallback defined (optional: informational)
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set

CAMPAIGN_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
SOVEREIGN = CAMPAIGN_ROOT / "sovereign"
sys.path.insert(0, str(CAMPAIGN_ROOT / "scripts" / "harness"))

import yaml

errors = []
warnings = []


def load_claims():
    claims_path = SOVEREIGN / "claims.yaml"
    with open(claims_path) as f:
        return yaml.safe_load(f)


def load_next_actions():
    na_path = SOVEREIGN / "next_actions.json"
    if not na_path.exists():
        return {"actions": []}
    with open(na_path) as f:
        return yaml.safe_load(f)


def find_sub_claims(claims_data):
    """Find all C_SUB_* claim IDs (successors of split claims)."""
    return {
        c["claim_id"]
        for c in claims_data["claims"]
        if c["claim_id"].startswith("C_SUB_")
    }


def check_blocker_claims(na, claims_data) -> tuple:
    """
    Check that all claim references in next_actions are current.
    Returns (errors, warnings).
    """
    errs = []
    warns = []

    claims_map = {c["claim_id"]: c for c in claims_data["claims"]}
    sub_claims = find_sub_claims(claims_data)
    all_ids = set(claims_map.keys())

    for action in na.get("actions", []):
        aid = action.get("id", "(unknown)")
        status = action.get("status", "unknown")
        blockers = action.get("blocker_claim_ids", [])

        # No blockers for blocked action — ERROR
        if status == "blocked" and not blockers:
            errs.append(
                f"{aid}: status=blocked but blocker_claim_ids is empty"
            )

        # Check each blocker
        for bid in blockers:
            bid_base = bid.split("@")[0]  # strip revision suffix

            # Non-existent claim — ERROR (also caught by 04_action_lint)
            if bid_base not in all_ids:
                errs.append(
                    f"{aid}: blocker '{bid}' — claim '{bid_base}' does not exist"
                )
                continue

            claim = claims_map[bid_base]
            ls = claim.get("lifecycle_status", "active")

            if ls == "superseded":
                # Find successor: superseded_by on the current revision
                revisions = claim.get("revisions", [])
                if revisions:
                    last_rev = revisions[-1]
                    sb = last_rev.get("superseded_by", "")
                    if sb:
                        warns.append(
                            f"{aid}: blocker '{bid}' references superseded claim "
                            f"({bid_base}). Successor: {sb}. Update blocker."
                        )
                    else:
                        warns.append(
                            f"{aid}: blocker '{bid}' references superseded claim "
                            f"({bid_base}) with no superseded_by. Update blocker."
                        )
                else:
                    warns.append(
                        f"{aid}: blocker '{bid}' references superseded claim "
                        f"({bid_base}). Update blocker to successor."
                    )

            elif ls == "split":
                if sub_claims:
                    warns.append(
                        f"{aid}: blocker '{bid}' references split claim "
                        f"({bid_base}). Update blocker to C_SUB_* successor(s): "
                        f"{sorted(sub_claims)}"
                    )
                else:
                    warns.append(
                        f"{aid}: blocker '{bid}' references split claim "
                        f"({bid_base}) but no C_SUB_* successors found. "
                        f"Update blocker or create successor claims."
                    )

            elif ls == "archived":
                warns.append(
                    f"{aid}: blocker '{bid}' references archived claim "
                    f"({bid_base}). Update blocker or remove."
                )

    return errs, warns


def check_completed_action_blockers(na) -> List[str]:
    """
    Check if completed actions are still listed as blockers elsewhere.
    This is informational — completed actions should have been removed from
    blocker lists.
    """
    warns = []

    # Build set of completed action IDs
    completed_ids = {
        a.get("id") for a in na.get("actions", [])
        if a.get("status") == "completed"
    }

    for action in na.get("actions", []):
        aid = action.get("id", "")
        blockers = action.get("blocker_claim_ids", [])
        for bid in blockers:
            bid_base = bid.split("@")[0]
            if bid_base in completed_ids:
                warns.append(
                    f"{aid}: references '{bid}' which is a completed action. "
                    f"Remove from blocker_claim_ids."
                )

    return warns


def check_no_fallback(na) -> List[str]:
    """Informational: actions without fallback defined."""
    warns = []
    for action in na.get("actions", []):
        aid = action.get("id", "")
        status = action.get("status", "")
        if status == "blocked" and not action.get("fallback"):
            warns.append(
                f"{aid}: status=blocked but no fallback defined"
            )
    return warns


def main():
    parser = argparse.ArgumentParser(
        description="gc_04: Report outdated claim references in next_actions.json."
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output in JSON format (machine-readable)"
    )
    args = parser.parse_args()

    print("=== gc_04_outdated_next_actions.py ===")
    print(f"Campaign root: {CAMPAIGN_ROOT}")
    print()

    na = load_next_actions()
    claims_data = load_claims()

    all_errs, all_warns = check_blocker_claims(na, claims_data)
    completed_warns = check_completed_action_blockers(na)
    fallback_warns = check_no_fallback(na)

    all_warns = all_warns + completed_warns + fallback_warns

    if args.json:
        print(json.dumps({
            "script": "gc_04_outdated_next_actions",
            "campaign_root": str(CAMPAIGN_ROOT),
            "errors": all_errs,
            "warnings": all_warns,
            "error_count": len(all_errs),
            "warning_count": len(all_warns),
            "check_groups": {
                "blocker_claims": {"errors": all_errs, "warnings": all_warns},
                "completed_blockers": completed_warns,
                "no_fallback": fallback_warns,
            }
        }, indent=2))
    else:
        if all_errs:
            print(f"--- ERRORS ({len(all_errs)}) ---")
            for e in all_errs:
                print(f"  ERROR: {e}")
            print()

        if all_warns:
            print(f"--- WARNINGS ({len(all_warns)}) ---")
            for w in all_warns:
                print(f"  WARN: {w}")
            print()

        if not all_errs and not all_warns:
            print("No outdated next_actions blockers found. All references are current.")
        elif all_errs:
            print(f"gc_04 FAILED: {len(all_errs)} error(s), {len(all_warns)} warning(s)")
        else:
            print(f"gc_04 PASSED with {len(all_warns)} warning(s) (informational only)")

    print()
    print("=== gc_04_outdated_next_actions.py DONE ===")
    sys.exit(0 if not all_errs else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gc_03_conflicting_claims.py
Phase 3: Claim revision chain + cross-reference consistency GC.

This script is the report-only sibling of 03_claim_lint.py, focused on
revision-chain integrity and cross-references between claims and actions.

Unlike 03_claim_lint.py (blocking gate in bootstrap), this script:
  - Checks supersession chain directional correctness
  - Checks cross-claim superseded_by chains for cycles
  - Checks lifecycle-to-action consistency (superseded/split claims still
    referenced as blockers)
  - All checks are WARNING only (report-only, no exit(1))

Exit codes:
  0 = PASS (no issues found)
  1 = FINDINGS (informational — report-only, no file modification)

Checks performed:
  [ERROR-level — structural integrity]
  - current_revision points to non-existent revision
  - revision numbers not sequential
  - superseded_by points to non-existent claim_id
  - superseded_by points to non-existent revision number
  - multiple live revisions (superseded_by=null) in one claim
  - revision changed but change_type is empty
  - revision changed but change_reason is empty

  [WARNING-level — cross-reference integrity]
  - lifecycle_status=superseded claim still referenced as active blocker
  - lifecycle_status=split claim still blocking with no C_SUB_* successor
  - superseded_by chain forms a cycle (impossible state)
  - superseded_by_reason missing when superseded_by is set
  - lifecycle_status=superseded but last revision superseded_by is null
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

CAMPAIGN_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
SOVEREIGN = CAMPAIGN_ROOT / "sovereign"
sys.path.insert(0, str(CAMPAIGN_ROOT / "scripts" / "harness"))

import yaml

errors = []
warnings = []

STATUS_ENUM = {"hypothesis", "candidate", "supported", "contradicted", "archived"}
LIFECYCLE_ENUM = {"active", "superseded", "split", "archived"}
CHANGE_TYPE_ENUM = {
    "create", "status_update", "evidence_added",
    "scope_narrowed", "scope_broadened",
    "statement_revised", "reframed", "superseded"
}


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


def load_queue():
    """Load the queue from state.json for blocker checking."""
    state_path = SOVEREIGN / "state.json"
    if not state_path.exists():
        return []
    with open(state_path) as f:
        state = json.load(f)
    return state.get("queue", [])


def get_active_blockers(na) -> Set[str]:
    """Return set of claim IDs that are active blockers in pending/blocked actions."""
    blockers = set()
    for action in na.get("actions", []):
        if action.get("status") in ("blocked", "pending"):
            for bid in action.get("blocker_claim_ids", []):
                blockers.add(bid.split("@")[0])  # strip revision suffix
    return blockers


def get_queue_blockers(queue) -> Set[str]:
    """Return set of claim IDs blocking pending/failed queue items."""
    blockers = set()
    for item in queue:
        if item.get("status") in ("pending", "blocked", "failed"):
            blockers.add(item.get("id", ""))
    return blockers


def check_revision_chain(claims_data) -> Tuple[List[str], List[str]]:
    """
    Check revision chain integrity within claims.yaml.
    Returns (errors, warnings).
    """
    errs = []
    warns = []
    all_ids = {c["claim_id"] for c in claims_data["claims"]}
    all_claim_refs = {}  # claim_id -> {revision_nums}

    for claim in claims_data["claims"]:
        cid = claim["claim_id"]
        revisions = claim["revisions"]
        all_claim_refs[cid] = {r["revision"] for r in revisions}

        # Skip empty claims
        if not revisions:
            errs.append(f"{cid}: has no revisions")
            continue

        rev_nums = sorted([r["revision"] for r in revisions])
        rev_map = {r["revision"]: r for r in revisions}

        # Check 1: revision numbers sequential
        expected = list(range(1, len(revisions) + 1))
        if rev_nums != expected:
            errs.append(
                f"{cid}: revision numbers not sequential: {rev_nums} (expected {expected})"
            )

        # Check 2: current_revision pointer exists
        cur_rev = claim.get("current_revision")
        if cur_rev is not None and cur_rev not in rev_map:
            errs.append(
                f"{cid}: current_revision={cur_rev} does not exist (available: {rev_nums})"
            )

        # Check 3: only one live revision
        live = [r for r in revisions if r.get("superseded_by") is None]
        if len(live) > 1:
            errs.append(
                f"{cid}: {len(live)} live revisions with superseded_by=null "
                f"(must be exactly 1): {[r['revision'] for r in live]}"
            )
        if len(live) == 0 and claim.get("lifecycle_status") != "archived":
            errs.append(
                f"{cid}: no live revision found (all superseded) but lifecycle_status={claim.get('lifecycle_status')}"
            )

        # Check 4: superseded_by chain for each non-live revision
        for r in revisions[:-1]:  # all except last
            sb = r.get("superseded_by")
            if not sb:
                errs.append(
                    f"{cid}@{r['revision']}: non-live revision must have superseded_by"
                )
            elif sb and "@" in sb:
                target_id, target_rev_str = sb.rsplit("@", 1)
                try:
                    target_rev = int(target_rev_str)
                except ValueError:
                    errs.append(
                        f"{cid}@{r['revision']}: superseded_by revision '{target_rev_str}' is not an integer"
                    )
                    continue

                # Check target claim exists
                if target_id not in all_ids:
                    errs.append(
                        f"{cid}@{r['revision']}: superseded_by '{sb}' — "
                        f"target claim '{target_id}' not found"
                    )
                elif target_rev not in all_claim_refs.get(target_id, set()):
                    errs.append(
                        f"{cid}@{r['revision']}: superseded_by '{sb}' — "
                        f"target revision {target_rev} not in {target_id} "
                        f"(available: {list(all_claim_refs[target_id])})"
                    )

        # Check 5: change_type and change_reason for non-create revisions
        for r in revisions:
            ct = r.get("change_type", "")
            if r["revision"] > 1:  # not the first revision
                if not ct:
                    errs.append(
                        f"{cid}@{r['revision']}: revision changed (not first) "
                        f"but change_type is empty"
                    )
            if ct and ct not in ("create", ""):
                if not r.get("change_reason"):
                    errs.append(
                        f"{cid}@{r['revision']}: change_type='{ct}' "
                        f"but change_reason is empty"
                    )

        # Check 6: superseded_by_reason required when superseded_by is set
        for r in revisions:
            sb = r.get("superseded_by")
            if sb and not r.get("superseded_by_reason"):
                warns.append(
                    f"{cid}@{r['revision']}: superseded_by='{sb}' "
                    f"but superseded_by_reason is empty"
                )

        # Check 7: lifecycle_status=superseded consistency
        if claim.get("lifecycle_status") == "superseded":
            if revisions[-1].get("superseded_by") is not None:
                errs.append(
                    f"{cid}: lifecycle_status=superseded but last revision "
                    f"rev{revisions[-1]['revision']} superseded_by is not null"
                )
            if revisions[-1].get("superseded_by") is None:
                warns.append(
                    f"{cid}: lifecycle_status=superseded but last revision "
                    f"rev{revisions[-1]['revision']} has superseded_by=null"
                )

        # Check 8: status enum
        for r in revisions:
            st = r.get("status", "")
            if st and st not in STATUS_ENUM:
                errs.append(
                    f"{cid}@{r['revision']}: status='{st}' not in enum: {STATUS_ENUM}"
                )

    return errs, warns


def check_supersession_cycles(claims_data) -> List[str]:
    """
    Check for impossible supersession cycles.
    A cycle would mean A supersedes B which supersedes ... which supersedes A.
    Returns list of warnings.
    """
    warns = []
    all_ids = {c["claim_id"] for c in claims_data["claims"]}

    # Build supersession graph: claim -> {claims it is superseded by}
    # (i.e., rev A's superseded_by points to B@r means A is the predecessor)
    # We check: for each claim's latest revision, trace superseded_by
    # and detect if we ever return to the starting claim

    for claim in claims_data["claims"]:
        cid = claim["claim_id"]
        revisions = claim["revisions"]
        if not revisions:
            continue

        visited = set()
        current_ref = f"{cid}@{revisions[-1]['revision']}"

        while current_ref and "@" in current_ref:
            if current_ref in visited:
                warns.append(
                    f"CYCLE DETECTED: supersession chain {visited} loops back to {current_ref}"
                )
                break
            visited.add(current_ref)

            target_id, target_rev_str = current_ref.rsplit("@", 1)
            try:
                target_rev = int(target_rev_str)
            except ValueError:
                break

            if target_id not in all_ids:
                break

            target_claim = next(
                (c for c in claims_data["claims"] if c["claim_id"] == target_id),
                None
            )
            if not target_claim:
                break

            target_revision = next(
                (r for r in target_claim["revisions"] if r["revision"] == target_rev),
                None
            )
            if not target_revision:
                break

            sb = target_revision.get("superseded_by")
            if not sb:
                break  # Reached a live revision
            current_ref = sb

    return warns


def check_lifecycle_action_consistency(claims_data, na, queue) -> List[str]:
    """
    Check that superseded/split claims are not actively blocking.
    Returns list of warnings.
    """
    warns = []

    claims_map = {c["claim_id"]: c for c in claims_data["claims"]}
    active_blockers = get_active_blockers(na)

    for bid in active_blockers:
        if bid not in claims_map:
            continue  # caught by action_lint
        claim = claims_map[bid]
        ls = claim.get("lifecycle_status", "active")

        if ls == "superseded":
            warns.append(
                f"Blocker '{bid}' references superseded claim. "
                f"Update blocker to successor claim or C_SUB_*."
            )
        elif ls == "split":
            # Check if successor claims (C_SUB_*) exist
            successors = [
                cid for cid in claims_map.keys()
                if cid.startswith("C_SUB_") and
                cid.split("@")[0] != bid
            ]
            if not successors:
                warns.append(
                    f"Blocker '{bid}' references split claim but no C_SUB_* "
                    f"successor claims found. Update blockers to C_SUB_* claims."
                )
            else:
                warns.append(
                    f"Blocker '{bid}' references split claim. "
                    f"Update blocker to C_SUB_* successor(s): {successors}"
                )
        elif ls == "archived":
            warns.append(
                f"Blocker '{bid}' references archived claim. "
                f"Should be resolved or removed."
            )

    # Check queue items (from state.json) that reference superseded/split claims
    # Queue items use their own id as the blocker, so this checks if the queue
    # item's own id (when status=blocked) references a superseded claim
    for item in queue:
        if item.get("status") == "blocked":
            blockers = item.get("blocker_claim_ids", [])
            for bid in blockers:
                bid_base = bid.split("@")[0]
                if bid_base in claims_map:
                    ls = claims_map[bid_base].get("lifecycle_status", "active")
                    if ls in ("superseded", "split", "archived"):
                        warns.append(
                            f"Queue item '{item.get('id')}' is blocked by "
                            f"'{bid}' which is {ls}. Update queue item blockers."
                        )

    return warns


def main():
    parser = argparse.ArgumentParser(
        description="gc_03: Report claim revision chain conflicts (report-only)."
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output in JSON format (machine-readable)"
    )
    args = parser.parse_args()

    print("=== gc_03_conflicting_claims.py ===")
    print(f"Campaign root: {CAMPAIGN_ROOT}")
    print()

    # Load all data
    claims_data = load_claims()
    na = load_next_actions()
    queue = load_queue()

    # Run all checks
    errs, warns = check_revision_chain(claims_data)
    cycle_warns = check_supersession_cycles(claims_data)
    lifecycle_warns = check_lifecycle_action_consistency(claims_data, na, queue)

    all_warns = warns + cycle_warns + lifecycle_warns

    if args.json:
        print(json.dumps({
            "script": "gc_03_conflicting_claims",
            "campaign_root": str(CAMPAIGN_ROOT),
            "errors": errs,
            "warnings": all_warns,
            "error_count": len(errs),
            "warning_count": len(all_warns),
            "check_groups": {
                "revision_chain": {"errors": errs, "warnings": warns},
                "supersession_cycles": cycle_warns,
                "lifecycle_action_consistency": lifecycle_warns,
            }
        }, indent=2))
    else:
        if errs:
            print(f"--- ERRORS ({len(errs)}) ---")
            for e in errs:
                print(f"  ERROR: {e}")
            print()

        if all_warns:
            print(f"--- WARNINGS ({len(all_warns)}) ---")
            for w in all_warns:
                print(f"  WARN: {w}")
            print()

        if not errs and not all_warns:
            print("No conflicts found. Claim revision chains are consistent.")
        elif errs:
            print(f"gc_03 FAILED: {len(errs)} error(s), {len(all_warns)} warning(s)")
        else:
            print(f"gc_03 PASSED with {len(all_warns)} warning(s) (informational only)")

    print()
    print("=== gc_03_conflicting_claims.py DONE ===")
    sys.exit(0 if not errs else 1)


if __name__ == "__main__":
    main()

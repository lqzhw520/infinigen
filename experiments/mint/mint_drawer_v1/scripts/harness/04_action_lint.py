#!/usr/bin/env python3
"""
04_action_lint.py
Phase 2: Action queue gate.

Exit codes:
  0 = PASS (no ERRORs)
  1 = FAIL (ERROR found — blocking)

ERROR: blocked action has no blocker_claim_ids; blocker references non-existent claim
WARNING: completed action still blocking; superseded/split/archived claim in blockers
"""
import sys
from pathlib import Path
import yaml

CAMPAIGN_ROOT = Path(__file__).parent.parent.parent.resolve()
SOVEREIGN = CAMPAIGN_ROOT / "sovereign"
sys.path.insert(0, str(CAMPAIGN_ROOT / "scripts" / "harness"))

errors = []
warnings = []


def load_claims():
    with open(SOVEREIGN / "claims.yaml") as f:
        data = yaml.safe_load(f)
    return {c["claim_id"]: c for c in data["claims"]}


def load_next_actions():
    with open(SOVEREIGN / "next_actions.json") as f:
        return yaml.safe_load(f)


def main():
    print("=== 04_action_lint.py ===")
    print(f"Campaign root: {CAMPAIGN_ROOT}")
    print()

    # Load both files
    claims_map = load_claims()
    all_claim_ids = set(claims_map.keys())

    na_path = SOVEREIGN / "next_actions.json"
    if not na_path.exists():
        warnings.append("next_actions.json not found — skipping action lint")
        print("=== ACTION LINT OK (file not found — skipped) ===")
        sys.exit(0)

    na = load_next_actions()
    actions = na.get("actions", [])

    print(f"  Actions in queue: {len(actions)}")

    for i, action in enumerate(actions, 1):
        aid = action.get("id", f"action[{i}]")
        status = action.get("status", "unknown")
        blockers = action.get("blocker_claim_ids", [])

        # ── BLOCKED actions must have blockers ──────────────────────
        if status == "blocked":
            if not blockers:
                errors.append(
                    f"{aid}: status=blocked but blocker_claim_ids is empty"
                )
            else:
                # Check each blocker exists and has revision ref (v1.3.1: mandatory for blocked)
                for bid in blockers:
                    bid_base = bid.split("@")[0]
                    if bid_base not in all_claim_ids:
                        errors.append(
                            f"{aid}: blocker_claim_ids contains '{bid}' but "
                            f"claim '{bid_base}' does not exist in claims.yaml"
                        )
                    elif "@" not in bid:
                        # v1.3.1: blocked actions MUST use revision refs for traceability
                        errors.append(
                            f"{aid}: blocker '{bid}' missing revision ref. "
                            f"Blocked actions must use '{bid}@N' format (e.g. '{bid}@current') "
                            f"for traceability. Use '{bid}@current' for current revision."
                        )

        # ── COMPLETED actions should not be active blockers ─────────
        if status == "completed":
            # Only warn if explicitly listed as blocker in other actions
            # (this is better checked in next_actions context, not per-action)
            pass

        # ── Superseded/split/archived claims in blockers ────────────
        if blockers:
            for bid in blockers:
                bid_base = bid.split("@")[0]
                if bid_base not in all_claim_ids:
                    continue  # already caught above
                claim = claims_map[bid_base]
                ls = claim.get("lifecycle_status", "active")
                if ls == "superseded":
                    warnings.append(
                        f"{aid}: blocker '{bid}' references a superseded claim "
                        f"({bid_base}). Update blocker to successor claim."
                    )
                elif ls == "split":
                    warnings.append(
                        f"{aid}: blocker '{bid}' references a split claim "
                        f"({bid_base}). Update blocker to successor C_SUB_* claims."
                    )
                elif ls == "archived":
                    warnings.append(
                        f"{aid}: blocker '{bid}' references an archived claim "
                        f"({bid_base}). Should be resolved or updated."
                    )

        # ── No fallback for blocked actions ───────────────────────────
        if status == "blocked" and not action.get("fallback"):
            warnings.append(
                f"{aid}: status=blocked but no fallback defined"
            )

    if warnings:
        print(f"\n--- WARNINGS ({len(warnings)}) ---")
        for w in warnings:
            print(f"  WARN: {w}")
        print()

    if errors:
        print(f"--- ERRORS ({len(errors)}) ---")
        for e in errors:
            print(f"  ERROR: {e}")
        print()
        print("Action lint FAILED. Fix errors before proceeding.")
        sys.exit(1)
    else:
        print(f"\n=== ACTION LINT OK ===")
        print(f"  Actions: {len(actions)}")
        print(f"  Errors: 0")
        if warnings:
            print(f"  Warnings: {len(warnings)} (informational)")
        sys.exit(0)


if __name__ == "__main__":
    main()

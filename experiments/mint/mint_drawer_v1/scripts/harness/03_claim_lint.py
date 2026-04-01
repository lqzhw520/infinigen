#!/usr/bin/env python3
"""
03_claim_lint.py
Phase 2: Claim consistency + revision chain gate.

Exit codes:
  0 = PASS (no ERRORs)
  1 = FAIL (ERROR found — blocking)

ERROR checks: structural integrity, chain validity, supersession correctness
WARNING checks: evidence links, superseded claims still blocking, etc.
"""
import sys
from pathlib import Path

CAMPAIGN_ROOT = Path(__file__).parent.parent.parent.resolve()
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


def lint_claims():
    """Main lint: load and validate claims.yaml."""
    claims_path = SOVEREIGN / "claims.yaml"
    with open(claims_path) as f:
        data = yaml.safe_load(f)

    schema_version = data.get("schema_version")
    if schema_version not in (1, 2):
        errors.append(f"schema_version={schema_version} — expected 1 or 2 (v1.3.1: supersedes/split_from/split_into fields added)")

    all_ids = {c["claim_id"] for c in data["claims"]}
    seen_ids = set()

    for claim in data["claims"]:
        cid = claim["claim_id"]
        revisions = claim["revisions"]

        # ── Duplicate claim_id ─────────────────────────────────────
        if cid in seen_ids:
            errors.append(f"DUPLICATE claim_id: {cid}")
        seen_ids.add(cid)

        # ── Revisions must be non-empty ────────────────────────────
        if not revisions:
            errors.append(f"{cid}: has no revisions — at least 1 required")
            continue

        rev_nums = sorted([r["revision"] for r in revisions])
        rev_map = {r["revision"]: r for r in revisions}

        # ── Revision numbers sequential ───────────────────────────
        expected = list(range(1, len(revisions) + 1))
        if rev_nums != expected:
            errors.append(
                f"{cid}: revision numbers not sequential: {rev_nums} (expected {expected})"
            )

        # ── Only one live revision ─────────────────────────────────
        live = [r for r in revisions if r.get("superseded_by") is None]
        if len(live) != 1:
            errors.append(
                f"{cid}: {len(live)} live revisions with superseded_by=null "
                f"(must be exactly 1): {[r['revision'] for r in live]}"
            )

        # ── current_revision pointer ───────────────────────────────
        expected_current = revisions[-1]["revision"]
        if claim.get("current_revision") != expected_current:
            errors.append(
                f"{cid}: current_revision={claim.get('current_revision')} "
                f"but last revision is {expected_current}"
            )

        # ── lifecycle_status enum ─────────────────────────────────
        ls = claim.get("lifecycle_status", "MISSING")
        if ls not in LIFECYCLE_ENUM:
            errors.append(
                f"{cid}: lifecycle_status='{ls}' not in enum: {LIFECYCLE_ENUM}"
            )

        # ── Per-revision checks ────────────────────────────────────
        for r in revisions:
            rev = r["revision"]
            ct = r.get("change_type", "")
            sb = r.get("superseded_by")
            sbr = r.get("superseded_by_reason", "")
            st = r.get("status", "")

            # status enum
            if st not in STATUS_ENUM:
                errors.append(
                    f"{cid}@{rev}: status='{st}' not in enum: {STATUS_ENUM}"
                )

            # change_type enum
            if ct not in CHANGE_TYPE_ENUM:
                errors.append(
                    f"{cid}@{rev}: change_type='{ct}' not in enum: {CHANGE_TYPE_ENUM}"
                )

            # change_reason required for non-create changes
            if ct not in ("create", "") and not r.get("change_reason"):
                errors.append(
                    f"{cid}@{rev}: change_type='{ct}' but change_reason is empty"
                )

            # superseded_by_reason required when superseded_by is set
            if sb and not sbr:
                errors.append(
                    f"{cid}@{rev}: superseded_by='{sb}' but superseded_by_reason is empty"
                )

            # superseded_by cannot be self
            if sb == f"{cid}@{rev}":
                errors.append(f"{cid}@{rev}: superseded_by points to itself")

            # contradicted/supported must have evidence
            if st in ("contradicted", "supported") and not r.get("evidence_ids"):
                warnings.append(
                    f"{cid}@{rev}: status={st} but evidence_ids is empty"
                )

        # ── Supersession chain integrity ────────────────────────────
        for r in revisions[:-1]:  # all except the last
            sb = r.get("superseded_by")
            if not sb:
                errors.append(
                    f"{cid}@{r['revision']}: non-live revision must have superseded_by"
                )
                continue
            if "@" not in sb:
                errors.append(
                    f"{cid}@{r['revision']}: superseded_by='{sb}' "
                    f"must be 'ClaimID@revision' format"
                )
                continue
            target_id, target_rev_str = sb.rsplit("@", 1)
            try:
                target_rev = int(target_rev_str)
            except ValueError:
                errors.append(
                    f"{cid}@{r['revision']}: superseded_by revision "
                    f"'{target_rev_str}' is not an integer"
                )
                continue

            if target_id not in all_ids:
                errors.append(
                    f"{cid}@{r['revision']}: superseded_by '{sb}' — "
                    f"target claim '{target_id}' not found"
                )
                continue

            target_claim = next(
                c for c in data["claims"] if c["claim_id"] == target_id
            )
            target_rev_nums = [rv["revision"] for rv in target_claim["revisions"]]
            if target_rev not in target_rev_nums:
                errors.append(
                    f"{cid}@{r['revision']}: superseded_by '{sb}' — "
                    f"target revision {target_rev} not in {target_id} "
                    f"(existing revisions: {target_rev_nums})"
                )

        # ── Superseded lifecycle consistency ─────────────────────────
        if claim["lifecycle_status"] == "superseded":
            if revisions[-1].get("superseded_by") is not None:
                errors.append(
                    f"{cid}: lifecycle_status=superseded but last revision "
                    f"rev{revisions[-1]['revision']} superseded_by is not null"
                )

        # ── Split lifecycle consistency ───────────────────────────────
        if claim["lifecycle_status"] == "split":
            warnings.append(
                f"{cid}: lifecycle_status=split — verify successor claims exist "
                f"and next_actions blockers are updated"
            )

    return data, errors, warnings


def lint_evidence_links(claims_data):
    """Check that evidence_ids in claims reference real evidence."""
    import json
    idx_path = SOVEREIGN / "evidence" / "index.json"
    if not idx_path.exists():
        warnings.append("evidence/index.json not found — skipping evidence link check")
        return

    with open(idx_path) as f:
        idx = json.load(f)
    evidence_ids = {e["evidence_id"] for e in idx.get("entries", [])}

    for claim in claims_data["claims"]:
        cid = claim["claim_id"]
        for r in claim["revisions"]:
            for eid in r.get("evidence_ids", []):
                if eid not in evidence_ids:
                    warnings.append(
                        f"{cid}@{r['revision']}: evidence_ids contains '{eid}' "
                        f"but no entry found in evidence/index.json"
                    )


def main():
    print("=== 03_claim_lint.py ===")
    print(f"Campaign root: {CAMPAIGN_ROOT}")
    print()

    claims_data, errs, warns = lint_claims()
    lint_evidence_links(claims_data)

    if warns:
        print(f"--- WARNINGS ({len(warns)}) ---")
        for w in warns:
            print(f"  WARN: {w}")
        print()

    if errs:
        print(f"--- ERRORS ({len(errs)}) ---")
        for e in errs:
            print(f"  ERROR: {e}")
        print()
        print("Claim lint FAILED. Fix errors before proceeding.")
        sys.exit(1)
    else:
        total_revs = sum(len(c["revisions"]) for c in claims_data["claims"])
        print(f"=== CLAIM LINT OK ===")
        print(f"  Claims: {len(claims_data['claims'])}")
        print(f"  Total revisions: {total_revs}")
        print(f"  Errors: 0")
        if warns:
            print(f"  Warnings: {len(warns)} (informational)")
        sys.exit(0)


if __name__ == "__main__":
    main()

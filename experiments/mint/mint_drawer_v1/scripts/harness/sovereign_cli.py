#!/usr/bin/env python3
"""
sovereign_cli.py — Phase 2
Harness v1.3 — Single controlled-write entry point for mint_drawer_v1 sovereign.

All subcommands:
  bootstrap               Session startup: run lints + print summary
  reconcile               Run all lints (dev mode)
  render-truth            Render sovereign/CAMPAIGN_TRUTH.generated.md
  revise-claim           Add new revision to existing claim
  supersede-claim        Mark old claim superseded, create new one
  split-claim            Split one claim into multiple C_SUB_* claims
  propose-revision        Night runner: write proposal (NO claims.yaml write)
  record-evidence        Register new evidence YAML
  close-experiment       Record evidence + update state.json
  write-handoff          Write sovereign/handoff.md

Rules:
  - ALL write operations are atomic (write temp → rename)
  - Do NOT call this script from training scripts directly
  - Agent ONLY: always use this CLI for sovereign changes
"""
import argparse
import json
import sys
import os
import shutil
import subprocess
import textwrap
from datetime import datetime, timezone  # noqa: F401 (kept for backward compat)
from pathlib import Path
from typing import Any

# ── Paths ──────────────────────────────────────────────────────────────────────
CAMPAIGN_ROOT = Path(__file__).parent.parent.parent.resolve()
SOVEREIGN = CAMPAIGN_ROOT / "sovereign"
EVIDENCE_DIR = SOVEREIGN / "evidence"
EVIDENCE_INBOX = CAMPAIGN_ROOT / "runtime" / "evidence_inbox"
PROPOSALS_DIR = SOVEREIGN / "proposals"
sys.path.insert(0, str(CAMPAIGN_ROOT / "scripts" / "harness"))

import yaml
from datetime import datetime
from datetime import timezone as DT_TIMEZONE

# ── Constants ──────────────────────────────────────────────────────────────────
STATUS_ENUM = ["hypothesis", "candidate", "supported", "contradicted", "archived"]
LIFECYCLE_ENUM = ["active", "superseded", "split", "archived"]
CHANGE_TYPE_ENUM = [
    "create", "status_update", "evidence_added",
    "scope_narrowed", "scope_broadened",
    "statement_revised", "reframed", "superseded"
]

def _now_ts() -> str:
    return datetime.now(DT_TIMEZONE.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

TS = _now_ts


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_yaml(name: str) -> dict:
    p = SOVEREIGN / name
    with open(p) as f:
        return yaml.safe_load(f)


def save_yaml(name: str, data: dict) -> None:
    p = SOVEREIGN / name
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    os.replace(tmp, p)


def load_claims() -> dict:
    return load_yaml("claims.yaml")


def save_claims(data: dict) -> None:
    save_yaml("claims.yaml", data)


def load_json_file(name: str) -> dict:
    """Load a sovereign JSON file."""
    p = SOVEREIGN / name
    with open(p) as f:
        return json.load(f)


def save_json_file(name: str, data: dict) -> None:
    """Atomic write: temp file + rename. JSON only."""
    p = SOVEREIGN / name
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)


def load_evidence_index() -> dict:
    """Load evidence/index.json as JSON."""
    return load_json_file("evidence/index.json")


def save_evidence_index(data: dict) -> None:
    """Save evidence/index.json as JSON (atomic)."""
    save_json_file("evidence/index.json", data)


def fatal(msg: str, code: int = 1) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def ok(msg: str) -> None:
    print(f"  OK: {msg}")


def info(msg: str) -> None:
    print(f"  INFO: {msg}")


def warn(msg: str) -> None:
    print(f"  WARN: {msg}")


# ── Subcommand: bootstrap ──────────────────────────────────────────────────────

def cmd_bootstrap(_args: argparse.Namespace) -> None:
    """Run all lints + print session summary with claim revision history."""
    print("=== sovereign_cli.py bootstrap ===")
    print(f"Campaign: {CAMPAIGN_ROOT.name}")
    print()

    lint_cmds = [
        ("02_reconcile_sources", [sys.executable, str(CAMPAIGN_ROOT / "scripts/harness/02_reconcile_sources.py")]),
        ("03_claim_lint",        [sys.executable, str(CAMPAIGN_ROOT / "scripts/harness/03_claim_lint.py")]),
        ("04_action_lint",       [sys.executable, str(CAMPAIGN_ROOT / "scripts/harness/04_action_lint.py")]),
    ]

    all_ok = True
    lint_warnings = []

    for name, cmd in lint_cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  PASS: {name}")
            # Collect warnings from stdout
            for line in result.stdout.splitlines():
                if "WARN" in line:
                    lint_warnings.append(f"{name}: {line.strip()}")
        else:
            print(f"  FAIL: {name}")
            print(result.stdout)
            print(result.stderr)
            all_ok = False

    if not all_ok:
        fatal("Bootstrap failed: lint errors found. Fix before proceeding.", 1)

    # Load claims for summary
    claims_data = load_claims()
    claims_list = claims_data["claims"]

    print()
    print("=== SESSION BOOTSTRAP ===")
    print(f"Sovereign version: {claims_data.get('schema_version', 'unknown')}")
    print(f"Active claims: {len(claims_list)}")

    for c in claims_list:
        cid = c["claim_id"]
        rev = c["current_revision"]
        lifecycle = c["lifecycle_status"]
        live_rev = c["revisions"][-1]
        status = live_rev["status"]
        scope = live_rev["scope"]
        rev_count = len(c["revisions"])
        superseded_note = ""
        if rev_count > 1:
            superseded_note = f" (superseded {rev_count - 1} older)"
        print(f"  {cid}: rev{rev} {lifecycle} {status}{superseded_note}")
        print(f"    scope: {scope}")
        print(f"    statement: {live_rev['statement'][:80]}...")

    print()
    print("Recent claim changes:")
    # Sort all revisions by created_at, most recent first, take last 3
    all_revs = []
    for c in claims_list:
        for r in c["revisions"]:
            all_revs.append((c["claim_id"], r))
    all_revs.sort(key=lambda x: x[1].get("created_at", ""), reverse=True)
    for claim_id, rev in all_revs[:3]:
        ct = rev["change_type"]
        ts = rev.get("created_at", "unknown")
        reason = rev.get("change_reason", "no reason")
        print(f"  {ts} {claim_id}@{rev['revision']} [{ct}]: {reason[:60]}")

    print()
    superseded = [c for c in claims_list if c["lifecycle_status"] == "superseded"]
    if superseded:
        print(f"Superseded claims: {len(superseded)}")
        for c in superseded:
            print(f"  {c['claim_id']} (superseded)")
    else:
        print("Superseded claims: none")

    if lint_warnings:
        print()
        print(f"Lint warnings ({len(lint_warnings)}):")
        for w in lint_warnings[:5]:
            print(f"  {w}")
        if len(lint_warnings) > 5:
            print(f"  ... and {len(lint_warnings) - 5} more")

    print()
    print("=== PROCEED WITH ANALYSIS ===")


# ── Subcommand: reconcile ─────────────────────────────────────────────────────

def cmd_reconcile(_args: argparse.Namespace) -> None:
    """Run all lints (dev mode)."""
    lint_cmds = [
        [sys.executable, str(CAMPAIGN_ROOT / "scripts/harness/02_reconcile_sources.py")],
        [sys.executable, str(CAMPAIGN_ROOT / "scripts/harness/03_claim_lint.py")],
        [sys.executable, str(CAMPAIGN_ROOT / "scripts/harness/04_action_lint.py")],
    ]
    all_ok = True
    for cmd in lint_cmds:
        r = subprocess.run(cmd)
        if r.returncode != 0:
            all_ok = False
    sys.exit(0 if all_ok else 1)


# ── Subcommand: render-truth ─────────────────────────────────────────────────

def cmd_render_truth(_args: argparse.Namespace) -> None:
    """Render sovereign/CAMPAIGN_TRUTH.generated.md from sovereign data."""
    claims_data = load_claims()
    with open(SOVEREIGN / "state.json") as f:
        state = json.load(f)
    with open(EVIDENCE_DIR / "index.json") as f:
        evidence_idx = json.load(f)

    ts = TS()

    # Build Part A: current claims table
    claim_rows = []
    for c in claims_data["claims"]:
        live = c["revisions"][-1]
        cid = c["claim_id"]
        row = f"{cid} | {live['status']} | {live['scope']} | {live['statement'][:60]}..."
        claim_rows.append(row)

    # Build Part B: recent revisions
    all_revs = []
    for c in claims_data["claims"]:
        for r in c["revisions"]:
            all_revs.append((c["claim_id"], r))
    all_revs.sort(key=lambda x: x[1].get("created_at", ""), reverse=True)
    rev_rows = []
    for claim_id, rev in all_revs[:6]:
        ts_r = rev.get("created_at", "unknown")
        ct = rev.get("change_type", "")
        reason = rev.get("change_reason", "—")
        rev_rows.append(f"{ts_r} | {claim_id}@{rev['revision']} | {ct} | {reason[:50]}...")

    # Build evidence table
    ev_rows = []
    for e in evidence_idx.get("entries", []):
        ev_rows.append(f"{e['evidence_id']} | {e['experiment_id']} | {e.get('type','?')} | {e.get('timestamp','?')}")

    # Current verdict from state.json
    verdict_raw = state.get("verdict") or state.get("current_verdict") or {}
    blockers = verdict_raw.get("blocker_claims", []) if isinstance(verdict_raw, dict) else []
    verdict_status = verdict_raw.get("status", "unknown") if isinstance(verdict_raw, dict) else (verdict_raw or "unknown")
    verdict_reason = verdict_raw.get("reason", "—") if isinstance(verdict_raw, dict) else "—"

    # Queue summary
    queue = state.get("queue", [])
    pending_steps = [s for s in queue if s.get("status") == "pending"]
    completed_steps = [s for s in queue if s.get("status") in ("completed", "passed")]
    blocked_steps = [s for s in queue if s.get("status") == "blocked"]

    # Get campaign_id from manifest
    manifest_path = SOVEREIGN / "manifest.yaml"
    campaign_id = CAMPAIGN_ROOT.name
    if manifest_path.exists():
        with open(manifest_path) as f:
            m = yaml.safe_load(f)
        campaign_id = m.get("campaign", {}).get("id", campaign_id)

    claim_rows_md = "\n".join(claim_rows)
    rev_rows_md  = "\n".join(rev_rows)
    ev_rows_md   = "\n".join(ev_rows)

    md = f"""<!-- GENERATED FILE — DO NOT EDIT -->
<!-- Source: sovereign/state.json + sovereign/claims.yaml + sovereign/evidence/index.json -->
<!-- Generated at: {ts} -->
<!-- To regenerate: sovereign_cli.py render-truth -->
<!-- HARNESS VERSION: v1.3 -->

# CAMPAIGN_TRUTH — {campaign_id}

## Part A — Current Active Claims

| Claim ID | Status | Scope | Statement |
|----------|--------|-------|-----------|
|{claim_rows_md}

### Current Verdict

**Status**: `{verdict_status}`
**Blockers**: `{', '.join(blockers) if blockers else 'none'}`
**Reason**: `{verdict_reason}`
**Decision**: `{state.get('decision') or '—'}`

---

## Part B — Recent Claim Revisions

| Time | Claim | Rev | Change | Reason |
|------|-------|-----|--------|--------|
|{rev_rows_md}

---

## Evidence Chain

| ID | Experiment | Type | Timestamp |
|----|-----------|------|-----------|
|{ev_rows_md}

---

## Campaign Queue Summary

- **Pending**: {len(pending_steps)} steps
- **Completed**: {len(completed_steps)} steps
- **Blocked**: {len(blocked_steps)} steps
"""

    out_path = SOVEREIGN / "CAMPAIGN_TRUTH.generated.md"
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        f.write(md)
    os.replace(tmp, out_path)
    print(f"  OK: rendered {out_path}")
    print(f"  Claims: {len(claims_data['claims'])}, Revisions shown: {min(6, len(all_revs))}")


# ── Subcommand: revise-claim ─────────────────────────────────────────────────

def cmd_revise_claim(args: argparse.Namespace) -> None:
    """Add a new revision to an existing claim."""
    data = load_claims()
    cid = args.claim

    claim = next((c for c in data["claims"] if c["claim_id"] == cid), None)
    if not claim:
        fatal(f"Claim not found: {cid}")

    if claim["lifecycle_status"] in ("superseded", "archived"):
        fatal(f"Cannot revise claim with lifecycle_status={claim['lifecycle_status']}. Use supersede-claim instead.")

    current_rev = claim["current_revision"]
    last_rev = claim["revisions"][-1]

    # Validate status
    new_status = args.status
    if new_status not in STATUS_ENUM:
        fatal(f"Invalid status '{new_status}'. Must be one of: {STATUS_ENUM}")

    change_type = args.change_type
    if change_type and change_type not in CHANGE_TYPE_ENUM:
        fatal(f"Invalid change_type '{change_type}'. Must be one of: {CHANGE_TYPE_ENUM}")

    if not args.change_reason and change_type not in ("create", ""):
        fatal("--change-reason is required for all non-create revisions")

    # Build new revision
    new_rev_num = current_rev + 1
    new_rev = {
        "revision": new_rev_num,
        "statement": args.statement or last_rev["statement"],
        "scope": args.scope or last_rev["scope"],
        "status": new_status,
        "evidence_ids": list(last_rev.get("evidence_ids", [])),
        "change_type": change_type or "status_update",
        "change_reason": args.change_reason or "",
        "created_at": TS(),
        "superseded_by": None,
        "superseded_by_reason": None,
        "supersedes": f"{cid}@{current_rev}",  # v1.3.1: track what this rev supersedes
    }

    # Update old last revision's superseded_by
    last_rev["superseded_by"] = f"{cid}@{new_rev_num}"
    last_rev["superseded_by_reason"] = args.change_reason or "superseded by newer revision"

    # Append new revision
    claim["revisions"].append(new_rev)
    claim["current_revision"] = new_rev_num

    # Add new evidence_ids if provided
    if args.evidence_add:
        existing = set(new_rev["evidence_ids"])
        for eid in args.evidence_add:
            if eid not in existing:
                new_rev["evidence_ids"].append(eid)

    save_claims(data)
    print(f"  OK: {cid}: added rev{new_rev_num} [{change_type}]")
    print(f"      status: {new_status}, scope: {new_rev['scope']}")
    print(f"      reason: {args.change_reason or '(no reason)'}")


# ── Subcommand: supersede-claim ───────────────────────────────────────────────

def cmd_supersede_claim(args: argparse.Namespace) -> None:
    """Mark old claim superseded and create new claim."""
    old_id = args.old_claim
    new_id = args.new_claim
    reason = args.supersede_reason or "superseded"

    data = load_claims()
    old_claim = next((c for c in data["claims"] if c["claim_id"] == old_id), None)
    if not old_claim:
        fatal(f"Old claim not found: {old_id}")

    if old_id == new_id:
        fatal("old-claim and new-claim must be different")

    new_claim_exists = any(c["claim_id"] == new_id for c in data["claims"])
    if new_claim_exists:
        fatal(f"New claim already exists: {new_id}. Use revise-claim instead.")

    # Update old claim lifecycle
    old_claim["lifecycle_status"] = "superseded"
    old_claim["supersedes"] = None  # a superseded claim doesn't supersede anything
    last_rev = old_claim["revisions"][-1]
    last_rev["superseded_by"] = f"{new_id}@1"
    last_rev["superseded_by_reason"] = reason
    last_rev["supersedes"] = None

    # Create new claim
    new_claim = {
        "claim_id": new_id,
        "current_revision": 1,
        "lifecycle_status": "active",
        "superseded_by": None,       # v1.3.1: no successor yet
        "supersedes": f"{old_id}",   # v1.3.1: tracks what this new claim supersedes
        "split_from": None,
        "split_into": [],
        "revisions": [{
            "revision": 1,
            "statement": args.new_statement or f"Supersedes {old_id}: {reason}",
            "scope": args.scope or "inherited",
            "status": args.status or "hypothesis",
            "evidence_ids": [],
            "change_type": "supersedes",
            "change_reason": f"Supersedes {old_id}: {reason}",
            "created_at": TS(),
            "superseded_by": None,
            "superseded_by_reason": None,
            "supersedes": None,        # rev 1 doesn't supersede anything
        }]
    }
    data["claims"].append(new_claim)
    save_claims(data)

    print(f"  OK: {old_id} → superseded by {new_id}")
    print(f"      reason: {reason}")
    print(f"  OK: Created new claim {new_id} (rev 1, status={args.status or 'hypothesis'})")


# ── Subcommand: split-claim ──────────────────────────────────────────────────

def cmd_split_claim(args: argparse.Namespace) -> None:
    """Split one claim into multiple C_SUB_* claims."""
    source_id = args.source_claim
    targets = args.target_claims
    reason = args.split_reason or "split"

    if len(targets) < 2:
        fatal("split-claim requires at least 2 target claims")

    data = load_claims()
    source = next((c for c in data["claims"] if c["claim_id"] == source_id), None)
    if not source:
        fatal(f"Source claim not found: {source_id}")

    for tid in targets:
        if any(x["claim_id"] == tid for x in data["claims"]):
            fatal(f"Target claim already exists: {tid}")

    # Update source claim lifecycle
    source["lifecycle_status"] = "split"
    source["split_into"] = list(targets)  # v1.3.1: track successors
    last = source["revisions"][-1]
    last["superseded_by"] = f"{source_id}@SPLIT"
    last["superseded_by_reason"] = reason
    last["supersedes"] = None

    # Create target claims
    scopes = args.split_scopes or ["inherited"] * len(targets)
    for i, tid in enumerate(targets):
        scope = scopes[i] if i < len(scopes) else "inherited"
        new_claim = {
            "claim_id": tid,
            "current_revision": 1,
            "lifecycle_status": "active",
            "superseded_by": None,
            "supersedes": None,
            "split_from": source_id,    # v1.3.1: track origin claim
            "split_into": [],
            "revisions": [{
                "revision": 1,
                "statement": f"Split from {source_id}: {reason}",
                "scope": scope,
                "status": "hypothesis",
                "evidence_ids": [],
                "change_type": "create",
                "change_reason": f"Split from {source_id}: {reason}",
                "created_at": TS(),
                "superseded_by": None,
                "superseded_by_reason": None,
                "supersedes": None,
            }]
        }
        data["claims"].append(new_claim)
        print(f"  OK: Created {tid} (scope={scope})")

    save_claims(data)
    print(f"  OK: {source_id} marked as split")
    print(f"      Split into: {', '.join(targets)}")


# ── Subcommand: propose-revision ─────────────────────────────────────────────

def cmd_propose_revision(args: argparse.Namespace) -> None:
    """Write a structured claim revision proposal (for night runner / agent).
    Does NOT modify claims.yaml.
    Proposal is saved to sovereign/proposals/ with full structured schema.
    """
    PROPOSALS_DIR.mkdir(exist_ok=True)
    ts_clean = TS().replace(":", "-").replace("+", "")
    filename = f"{ts_clean}_{args.claim}.yaml"
    proposal_path = PROPOSALS_DIR / filename

    # v1.3.1: Structured proposal schema
    proposal = {
        # ── Identity ──────────────────────────────────────────
        "proposal_id": filename.replace(".yaml", ""),
        "schema_version": 1,
        # ── Reference (what claim/rev is being modified) ────
        "claim_ref": args.claim,           # e.g. C_SIGLIP_GENERALIZATION
        "revision_ref": args.revision_ref,  # v1.3.1: optional, defaults to current
        # ── Proposal type ────────────────────────────────────
        "proposal_type": args.proposal_type or "revise_claim",
        # ── Proposed changes ────────────────────────────────
        "proposed_change": {
            "status": args.proposed_status or None,
            "statement": args.proposed_statement or None,
            "scope": args.proposed_scope or None,
            "change_type": args.proposed_change_type or "evidence_added",
            "change_reason": args.proposed_change_reason or "",
            "evidence_ids_add": args.proposed_evidence_add or [],
            "evidence_ids_remove": args.proposed_evidence_remove or [],
        },
        # ── Evidence basis ───────────────────────────────────
        "evidence_ids": args.evidence or [],
        "evidence_summary": args.evidence_summary or "",
        # ── Authorship ───────────────────────────────────────
        "author": args.author or "night_runner",   # night_runner | agent | human
        "created_at": TS(),
        # ── Review lifecycle ────────────────────────────────
        "review_status": "pending",       # pending | approved | rejected | deferred
        "approved_revision": None,      # filled in when approved
        "rejected_reason": None,
        "reviewed_by": None,
        "reviewed_at": None,
        # ── Notes ───────────────────────────────────────────
        "notes": args.notes or "",
    }

    tmp = proposal_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        yaml.dump(proposal, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    os.replace(tmp, proposal_path)

    print(f"  OK: Proposal written: sovereign/proposals/{filename}")
    print(f"  Type: {args.proposal_type or 'revise_claim'}")
    print(f"  Claim: {args.claim}")
    if args.revision_ref:
        print(f"  Rev:   {args.revision_ref}")
    print(f"  Status: pending (not yet applied)")
    print(f"  To apply:   sovereign_cli.py review-proposal --id {proposal['proposal_id']} --approve")
    print(f"  To reject:  sovereign_cli.py review-proposal --id {proposal['proposal_id']} --reject --reason '...'")


# ── Subcommand: review-proposal ─────────────────────────────────────────────

def cmd_review_proposal(args: argparse.Namespace) -> None:
    """Review (approve/reject) a pending proposal. Does NOT auto-apply."""
    proposals = sorted(PROPOSALS_DIR.glob("*.yaml"))
    if not proposals:
        print("  INFO: No proposals found in sovereign/proposals/")
        return

    if args.list:
        print("=== Pending Proposals ===")
        for p in proposals:
            with open(p) as f:
                data = yaml.safe_load(f)
            rs = data.get("review_status", "unknown")
            marker = ">>>" if rs == "pending" else "   "
            print(f"  {marker} [{rs}] {data.get('proposal_id','?')}  {data.get('claim_ref','?')}  by={data.get('author','?')}  {data.get('created_at','')[:16]}")
            if data.get("proposed_change", {}).get("status"):
                print(f"        → status: {data['proposed_change']['status']}")
            if data.get("evidence_ids"):
                print(f"        → evidence: {', '.join(data.get('evidence_ids', []))}")
        return

    # Find the specific proposal
    target = None
    for p in proposals:
        if args.id in p.name:
            target = p
            break
    if not target:
        fatal(f"Proposal not found: {args.id}")

    with open(target) as f:
        data = yaml.safe_load(f)

    if data.get("review_status") != "pending":
        fatal(f"Proposal {args.id} is already {data['review_status']}. Cannot re-review.")

    ts = TS()
    if args.approve:
        data["review_status"] = "approved"
        data["reviewed_by"] = "agent"
        data["reviewed_at"] = ts
        # approved_revision is the claim_id + new revision number to be created
        # (set by the agent after running revise-claim)
        data["approved_revision"] = f"{data['claim_ref']}@TBD"
        print(f"  OK: Proposal {args.id} APPROVED")
        print(f"       Next: run revise-claim manually, then update approved_revision field")
        print(f"       Then run: sovereign_cli.py review-proposal --id {args.id} --finalize --revision N")
    elif args.reject:
        if not args.reason:
            fatal("--reject requires --reason")
        data["review_status"] = "rejected"
        data["rejected_reason"] = args.reason
        data["reviewed_by"] = "agent"
        data["reviewed_at"] = ts
        print(f"  OK: Proposal {args.id} REJECTED")
        print(f"       Reason: {args.reason}")
    elif args.finalize:
        if not args.revision:
            fatal("--finalize requires --revision")
        data["approved_revision"] = f"{data['claim_ref']}@{args.revision}"
        print(f"  OK: Finalized {args.id} → approved as {data['approved_revision']}")
    else:
        fatal("Must specify --approve, --reject, --finalize, or --list")

    tmp = target.with_suffix(".tmp")
    with open(tmp, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    os.replace(tmp, target)


# ── Subcommand: record-evidence ─────────────────────────────────────────────

def cmd_record_evidence(args: argparse.Namespace) -> None:
    """Register a new evidence YAML file.
    v1.3.1: Candidate evidence goes to runtime/evidence_inbox/ (staging).
    Only close-experiment promotes it to sovereign/evidence/ (canonical).
    """
    ev_path = Path(args.file)
    if not ev_path.is_absolute():
        ev_path = CAMPAIGN_ROOT / ev_path

    if not ev_path.exists():
        fatal(f"Evidence file not found: {ev_path}")

    # Load and validate evidence YAML
    with open(ev_path) as f:
        ev_data = yaml.safe_load(f)

    required = ["evidence_id", "experiment_id", "scope", "metrics"]
    missing = [f for f in required if f not in ev_data]
    if missing:
        fatal(f"Evidence missing required fields: {missing}")

    eid = ev_data["evidence_id"]

    # v1.3.1: Copy to sovereign/evidence/ (canonical) — only here is canonical
    dest_path = EVIDENCE_DIR / f"{eid}.yaml"
    if ev_path.resolve() != dest_path.resolve():
        shutil.copy2(ev_path, dest_path)
        print(f"  OK: Copied {ev_path} → {dest_path}")

    # Update index.json (JSON only)
    idx = load_evidence_index()
    existing = {e["evidence_id"] for e in idx.get("entries", [])}
    if eid in existing:
        fatal(f"Evidence {eid} already registered in index.json")

    idx.setdefault("entries", [])
    idx["entries"].append({
        "evidence_id": eid,
        "experiment_id": ev_data["experiment_id"],
        "type": ev_data.get("type", "eval"),
        "timestamp": ev_data.get("timestamp", TS()),
        "path": f"sovereign/evidence/{eid}.yaml",
        "verified": False,
    })
    idx["last_updated"] = TS()
    save_evidence_index(idx)

    print(f"  OK: Registered {eid} (experiment={ev_data['experiment_id']})")
    print(f"  NOTE: Candidate evidence should go to runtime/evidence_inbox/ first.")
    print(f"        Training scripts and night runner write there.")
    print(f"        Agent calls record-evidence to canonicalize.")


# ── Subcommand: close-experiment ──────────────────────────────────────────────

def cmd_close_experiment(args: argparse.Namespace) -> None:
    """Record evidence + optionally update state.json."""
    ev_id = args.evidence
    experiment_id = args.experiment_id

    # Verify evidence is registered
    idx = load_evidence_index()
    if not any(e["evidence_id"] == ev_id for e in idx.get("entries", [])):
        fatal(f"Evidence {ev_id} not registered. Run: record-evidence first.")

    # Update state.json
    state_path = SOVEREIGN / "state.json"
    with open(state_path) as f:
        state = json.load(f)

    state.setdefault("experiments", [])
    state["experiments"].append({
        "experiment_id": experiment_id,
        "evidence_id": ev_id,
        "timestamp": TS(),
        "closed_by": "agent",
    })
    state["last_merged_evidence_id"] = ev_id

    tmp = state_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, state_path)
    print(f"  OK: Closed experiment {experiment_id} (evidence={ev_id})")

    # Optionally promote claim
    if args.promote_claim:
        print(f"  NOTE: Claim promotion deferred — run separately:")
        print(f"        sovereign_cli.py revise-claim --claim {args.promote_claim} ...")


# ── Subcommand: write-handoff ────────────────────────────────────────────────

def cmd_write_handoff(args: argparse.Namespace) -> None:
    """Write sovereign/handoff.md."""
    content = args.content or textwrap.dedent(f"""\
        ## Handoff — {CAMPAIGN_ROOT.name}

        **Updated**: {TS()}
        **Sovereign version**: {load_claims().get('schema_version', 1)}

        ### Current Verdict

        **Status**: `{args.status or 'unknown'}`
        **Blockers**: `{args.blockers or 'none'}`
        **Reason**: `{args.reason or '—'}`
        **Decision**: `{args.decision or '—'}`

        ### What the Next Agent Must Do

        1. Run: `python3 scripts/harness/sovereign_cli.py bootstrap`
        2. Read: `sovereign/claims.yaml` — ONLY sovereign files are canonical
        3. DO NOT edit `sovereign/claims.yaml` directly — use `sovereign_cli.py revise-claim`
        4. DO NOT close experiments directly — use `sovereign_cli.py close-experiment`
        5. Read: `sovereign/handoff.md` (this file) for latest state
    """)

    handoff_path = SOVEREIGN / "handoff.md"
    tmp = handoff_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, handoff_path)
    print(f"  OK: Updated sovereign/handoff.md")


def cmd_go(_args: argparse.Namespace) -> None:
    """One-shot session start: lint + truth + next actions. Run this first."""
    import sys

    print("╔══════════════════════════════════════════════════════╗")
    print("║  sovereign go — mint_drawer_v1                      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # ── Step 1: Lint gates ──────────────────────────────────────────────────────
    lint_cmds = [
        ("02_reconcile_sources", CAMPAIGN_ROOT / "scripts/harness/02_reconcile_sources.py"),
        ("03_claim_lint",        CAMPAIGN_ROOT / "scripts/harness/03_claim_lint.py"),
        ("04_action_lint",       CAMPAIGN_ROOT / "scripts/harness/04_action_lint.py"),
    ]
    lint_ok = True
    lint_warnings = []
    for name, script in lint_cmds:
        r = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            print(f"  PASS  {name}")
            for line in r.stdout.splitlines():
                if "WARN" in line:
                    lint_warnings.append(f"  WARN  {name}: {line.strip()}")
        else:
            print(f"  FAIL  {name}")
            for line in (r.stdout + r.stderr).splitlines():
                if line.strip():
                    print(f"         {line.strip()}")
            lint_ok = False

    if not lint_ok:
        print()
        print("LINT FAILURES — fix before proceeding.")
        print("Hint: sovereign_cli.py reconcile  (dev mode, no fatal exit)")
        return

    # ── Step 2: Load truth ─────────────────────────────────────────────────────
    with open(SOVEREIGN / "state.json") as f:
        state = json.load(f)
    with open(SOVEREIGN / "claims.yaml") as f:
        import yaml as _yaml
        claims_data = _yaml.safe_load(f)
    with open(SOVEREIGN / "evidence/index.json") as f:
        evidence_idx = json.load(f)
    with open(CAMPAIGN_ROOT / "sovereign/next_actions.json") as f:
        next_actions = json.load(f)

    # ── Step 3: Print session summary ─────────────────────────────────────────
    verdict = state.get("verdict", "unknown")
    phase_gate = state.get("phase_gate", "unknown")

    print()
    print("─── Campaign Truth ─────────────────────────────────────────")
    print(f"  verdict:  {verdict}")
    print(f"  phase:    {phase_gate}")

    # Active claims
    print()
    print("─── Active Claims ──────────────────────────────────────────")
    for c in claims_data["claims"]:
        if c.get("lifecycle_status") not in ("active",):
            continue
        live = c["revisions"][-1]
        ev_count = len(live.get("evidence_ids", []))
        print(f"  [{live['status']:>12}]  {c['claim_id']}  (rev{c['current_revision']}, {ev_count} evidence)")

    # Evidence count
    total_ev = len(evidence_idx.get("entries", []))
    print(f"  total evidence: {total_ev}")

    # Lint warnings
    if lint_warnings:
        print()
        print("─── Warnings ────────────────────────────────────────────────")
        for w in lint_warnings[:5]:
            print(f"  {w}")
        if len(lint_warnings) > 5:
            print(f"  ... +{len(lint_warnings)-5} more (run gc_* scripts for full list)")

    # ── Step 4: Next actions (from next_actions.json) ──────────────────────────
    print()
    print("─── Next Actions ────────────────────────────────────────────")
    pending = [a for a in next_actions.get("actions", []) if a.get("status") == "pending"]
    in_progress = [a for a in next_actions.get("actions", []) if a.get("status") == "in_progress"]
    completed = [a for a in next_actions.get("actions", []) if a.get("status") == "completed"]

    print(f"  pending: {len(pending)}  |  in_progress: {len(in_progress)}  |  completed: {len(completed)}")

    if in_progress:
        print()
        print("  ▶ IN PROGRESS")
        for a in in_progress:
            print(f"    [{a.get('type', '?')}]  priority={a.get('priority', '-')}")
            if a.get("target"):
                print(f"      target: {a['target']}")
            if a.get("current_problem"):
                print(f"      problem: {a['current_problem'][:80]}")

    if pending:
        print()
        print("  ▶ PENDING (next to start)")
        for a in pending:
            priority = a.get("priority", "-")
            marker = "►►" if priority == "P0" else "  "
            print(f"    {marker} [{priority}]  {a.get('type', '?')}")
            print(f"        target: {a.get('target', '-')}")

    # ── Step 5: What to do next ────────────────────────────────────────────────
    print()
    print("─── What to Do Next ─────────────────────────────────────────")
    next_action = in_progress[0] if in_progress else (pending[0] if pending else None)
    if next_action:
        action_type = next_action.get("type", "?")
        target = next_action.get("target", "-")
        priority = next_action.get("priority", "-")
        print(f"  Suggested next: [{priority}] {action_type}")
        print(f"    {target}")

        # Map action type to file paths / instructions
        if "P0_physics" in action_type:
            print(f"    files: scripts/mint/physics_legality.py")
            print(f"    hint:  python scripts/mint/physics_legality.py generate --N 10 --bs 24")
        elif "P1a" in action_type:
            print(f"    files: scripts/mint/drawer_robot_env.py")
            print(f"    hint:  Fix _state_vector() — replace binary gripper with continuous joint position")
        elif "P1b" in action_type:
            print(f"    files: outputs/mujoco_teacher_env_design.md")
            print(f"    hint:  Implement robosuite+MuJoCo teacher environment")
        elif "generate_data" in action_type:
            print(f"    hint:  Run physics_constrained_teacher_rollout() then dataset_builder")
    else:
        print("  No pending actions. Review sovereign/claims.yaml.")

    print()
    print("─── Quick Ref ────────────────────────────────────────────────")
    print("  bootstrap:   sovereign_cli.py bootstrap     (same as this)")
    print("  reconcile:   sovereign_cli.py reconcile    (lint only, dev mode)")
    print("  truth:      sovereign_cli.py render-truth")
    print("  handoff:    sovereign_cli.py write-handoff")
    print("  next steps: sovereign/next_actions.json")
    print()
    print("═" * 56)
    print("  READY. Choose your next action from the list above.")
    print("═" * 56)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Harness v1.3 — Sovereign CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("go", help="One-shot: lint + truth + next actions — run this first")
    # bootstrap
    sub.add_parser("bootstrap", help="Session startup: run lints + print summary")

    # reconcile
    sub.add_parser("reconcile", help="Run all lints (dev mode)")

    # render-truth
    sub.add_parser("render-truth", help="Render CAMPAIGN_TRUTH.generated.md")

    # revise-claim
    rc = sub.add_parser("revise-claim", help="Add new revision to existing claim")
    rc.add_argument("--claim", required=True, help="Claim ID, e.g. C_SIGLIP_GENERALIZATION")
    rc.add_argument("--status", required=True, choices=STATUS_ENUM, help="New status")
    rc.add_argument("--statement", help="New statement (optional)")
    rc.add_argument("--scope", help="New scope (optional)")
    rc.add_argument("--change-type", choices=CHANGE_TYPE_ENUM, help="change_type enum")
    rc.add_argument("--change-reason", help="REQUIRED for non-create revisions")
    rc.add_argument("--evidence-add", nargs="+", help="Additional evidence IDs to add")

    # supersede-claim
    sc = sub.add_parser("supersede-claim", help="Mark old claim superseded, create new one")
    sc.add_argument("--old-claim", required=True)
    sc.add_argument("--new-claim", required=True)
    sc.add_argument("--new-statement")
    sc.add_argument("--scope")
    sc.add_argument("--status", default="hypothesis", choices=STATUS_ENUM)
    sc.add_argument("--supersede-reason", required=True)

    # split-claim
    sp = sub.add_parser("split-claim", help="Split one claim into multiple")
    sp.add_argument("--source-claim", required=True)
    sp.add_argument("--target-claims", nargs="+", required=True)
    sp.add_argument("--split-reason", required=True)
    sp.add_argument("--split-scopes", nargs="+")

    # propose-revision
    pr = sub.add_parser("propose-revision", help="Write structured proposal (night runner/agent)")
    pr.add_argument("--claim", required=True, help="Claim ID, e.g. C_SIGLIP_GENERALIZATION")
    pr.add_argument("--revision-ref", help="Exact revision ref, e.g. C_SIGLIP_GENERALIZATION@3")
    pr.add_argument("--proposal-type", default="revise_claim",
                    choices=["revise_claim", "supersede_claim", "split_claim"],
                    help="Type of proposal")
    pr.add_argument("--proposed-status", choices=STATUS_ENUM)
    pr.add_argument("--proposed-statement")
    pr.add_argument("--proposed-scope")
    pr.add_argument("--proposed-change-type", choices=CHANGE_TYPE_ENUM)
    pr.add_argument("--proposed-change-reason", required=True)
    pr.add_argument("--proposed-evidence-add", nargs="+", dest="proposed_evidence_add")
    pr.add_argument("--proposed-evidence-remove", nargs="+", dest="proposed_evidence_remove")
    pr.add_argument("--evidence", nargs="+", help="Evidence IDs supporting this proposal")
    pr.add_argument("--evidence-summary", help="Brief description of evidence basis")
    pr.add_argument("--author", default="night_runner", choices=["night_runner", "agent", "human"])
    pr.add_argument("--notes", help="Additional notes")

    # review-proposal
    rp = sub.add_parser("review-proposal", help="Review/approve/reject a pending proposal")
    rp.add_argument("--id", help="Proposal ID (partial match accepted)")
    rp.add_argument("--list", action="store_true", help="List all proposals")
    rp.add_argument("--approve", action="store_true", help="Approve proposal")
    rp.add_argument("--reject", action="store_true", help="Reject proposal")
    rp.add_argument("--finalize", action="store_true", help="Finalize after revise-claim")
    rp.add_argument("--revision", type=int, help="Approved revision number (for --finalize)")
    rp.add_argument("--reason", help="Rejection reason (required for --reject)")

    # record-evidence
    re = sub.add_parser("record-evidence", help="Register new evidence YAML")
    re.add_argument("--file", required=True, help="Path to evidence YAML file")

    # close-experiment
    ce = sub.add_parser("close-experiment", help="Record evidence + update state.json")
    ce.add_argument("--evidence", required=True, help="Evidence ID, e.g. E003")
    ce.add_argument("--experiment-id", required=True)
    ce.add_argument("--promote-claim", help="Optional: claim ID to promote after closing")

    # write-handoff
    wh = sub.add_parser("write-handoff", help="Write sovereign/handoff.md")
    wh.add_argument("--status")
    wh.add_argument("--blockers")
    wh.add_argument("--reason")
    wh.add_argument("--decision")
    wh.add_argument("--content", help="Full handoff content (markdown)")

    return p


COMMAND_MAP = {
    "go": cmd_go,
    "bootstrap": cmd_bootstrap,
    "reconcile": cmd_reconcile,
    "render-truth": cmd_render_truth,
    "revise-claim": cmd_revise_claim,
    "supersede-claim": cmd_supersede_claim,
    "split-claim": cmd_split_claim,
    "propose-revision": cmd_propose_revision,
    "review-proposal": cmd_review_proposal,
    "record-evidence": cmd_record_evidence,
    "close-experiment": cmd_close_experiment,
    "write-handoff": cmd_write_handoff,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cmd = COMMAND_MAP.get(args.command)
    if cmd is None:
        fatal(f"Unknown command: {args.command}")
    cmd(args)


if __name__ == "__main__":
    main()

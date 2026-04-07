#!/usr/bin/env python3
from __future__ import annotations
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
import os
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
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

from truth_backend import (
    CAMPAIGN_ROOT as TB_CAMPAIGN_ROOT,
    CURRENT_TRUTH_PATH,
    DATASET_MANIFEST_PATH,
    EVIDENCE_INDEX_PATH,
    MODEL_LOAD_FIDELITY_PATH,
    NEXT_ACTIONS_PATH,
    RUN_LEDGER_PATH,
    STATE_PATH,
    WORKSPACE_MANIFEST_PATH,
    build_current_truth,
    build_model_load_fidelity,
    build_workspace_manifest,
    derive_docs,
    load_json as tb_load_json,
    migrate_state_file,
    normalize_next_actions,
    normalize_run_ledger,
    now_ts as tb_now_ts,
    publish_experiment,
    save_json as tb_save_json,
)

# ── Constants ──────────────────────────────────────────────────────────────────
STATUS_ENUM = ["hypothesis", "candidate", "supported", "contradicted", "archived"]
LIFECYCLE_ENUM = ["active", "superseded", "split", "archived"]
CHANGE_TYPE_ENUM = [
    "create", "status_update", "evidence_added",
    "scope_narrowed", "scope_broadened",
    "statement_revised", "reframed", "superseded"
]

def _now_ts() -> str:
    return tb_now_ts()

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


def _lint_scripts() -> list[tuple[str, Path]]:
    return [
        ("02_reconcile_sources", CAMPAIGN_ROOT / "scripts/harness/02_reconcile_sources.py"),
        ("03_claim_lint", CAMPAIGN_ROOT / "scripts/harness/03_claim_lint.py"),
        ("04_action_lint", CAMPAIGN_ROOT / "scripts/harness/04_action_lint.py"),
        ("05_semantic_truth_lint", CAMPAIGN_ROOT / "scripts/harness/05_semantic_truth_lint.py"),
    ]


def _run_lints() -> tuple[bool, list[str]]:
    warnings = []
    ok_all = True
    for name, script in _lint_scripts():
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"  PASS  {name}")
            for line in result.stdout.splitlines():
                if "WARN:" in line or line.strip().startswith("WARN"):
                    warnings.append(f"{name}: {line.strip()}")
        else:
            ok_all = False
            print(f"  FAIL  {name}")
            for line in (result.stdout + result.stderr).splitlines():
                if line.strip():
                    print(f"         {line.strip()}")
    return ok_all, warnings


def _refresh_canonical_surfaces() -> dict[str, Any]:
    workspace = build_workspace_manifest()
    fidelity = build_model_load_fidelity(workspace)
    normalize_run_ledger()
    raw_state = tb_load_json(STATE_PATH, default={}) or {}
    next_actions = tb_load_json(NEXT_ACTIONS_PATH, default={}) or {}
    if next_actions:
        tb_save_json(NEXT_ACTIONS_PATH, normalize_next_actions(next_actions, raw_state))
    state = migrate_state_file()
    truth = build_current_truth()
    derive_docs(truth)
    truth = build_current_truth()
    derive_docs(truth)
    truth = build_current_truth()
    return {
        "workspace": workspace,
        "fidelity": fidelity,
        "state": state,
        "truth": truth,
    }


# ── Subcommand: bootstrap ──────────────────────────────────────────────────────

def cmd_bootstrap(_args: argparse.Namespace) -> None:
    """Alias to go: refresh truth surfaces, run lints, regenerate docs, print current truth."""
    cmd_go(_args)


# ── Subcommand: reconcile ─────────────────────────────────────────────────────

def cmd_reconcile(_args: argparse.Namespace) -> None:
    """Run all lints against current canonical/runtime surfaces."""
    _refresh_canonical_surfaces()
    all_ok, _warnings = _run_lints()
    sys.exit(0 if all_ok else 1)


# ── Subcommand: render-truth ─────────────────────────────────────────────────

def cmd_render_truth(_args: argparse.Namespace) -> None:
    """Refresh current_truth and regenerate all derived docs."""
    context = _refresh_canonical_surfaces()
    print(f"  OK: rendered {SOVEREIGN / 'CAMPAIGN_TRUTH.generated.md'}")
    print(f"  OK: refreshed {CURRENT_TRUTH_PATH}")
    print(f"  phase: {context['truth']['current']['phase']}")
    print(f"  verdict: {context['truth']['current']['verdict']}")


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
    _refresh_canonical_surfaces()

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
    _refresh_canonical_surfaces()
    print(f"  OK: Closed experiment {experiment_id} (evidence={ev_id})")

    # Optionally promote claim
    if args.promote_claim:
        print(f"  NOTE: Claim promotion deferred — run separately:")
        print(f"        sovereign_cli.py revise-claim --claim {args.promote_claim} ...")


# ── Subcommand: write-handoff ────────────────────────────────────────────────

def cmd_write_handoff(args: argparse.Namespace) -> None:
    """Regenerate derived docs, including sovereign/handoff.md."""
    if args.content or args.status or args.blockers or args.reason or args.decision:
        warn("write-handoff now generates from current_truth.json; manual content arguments are ignored.")
    _refresh_canonical_surfaces()
    print(f"  OK: Updated {SOVEREIGN / 'handoff.md'}")


# ── Subcommand: pack-dataset ──────────────────────────────────────────

def cmd_pack_dataset(args: argparse.Namespace) -> None:
    """Run pack_dataset.sh with optional --force-archive.
    This is the MANDATORY entry point for all dataset packing.
    """
    import subprocess
    import sys
    from pathlib import Path as _P

    script = _P(__file__).parent / "pack_dataset.sh"
    cmd = ["bash", str(script)]
    if args.force_archive:
        cmd.append("--force-archive")
    cmd.extend(args.extra_args)

    print(f"=== sovereign_cli.py pack-dataset ===")
    print(f"  Running: {' '.join(cmd)}")
    print(f"  archive-first: {'no (will fail if dataset/ exists)' if not args.force_archive else 'YES (--force-archive)'}")
    print()

    r = subprocess.run(cmd)
    sys.exit(r.returncode)


# ── Subcommand: verify-manifest ─────────────────────────────────────

def cmd_verify_manifest(args: argparse.Namespace) -> None:
    """Run train_gate.py to verify dataset is valid.
    Equivalent to: python scripts/harness/train_gate.py
    """
    import subprocess
    import sys
    from pathlib import Path as _P

    script = _P(__file__).parent / "train_gate.py"
    r = subprocess.run([sys.executable, str(script)])
    sys.exit(r.returncode)


def cmd_refresh_workspace(_args: argparse.Namespace) -> None:
    manifest = build_workspace_manifest()
    print(f"  OK: wrote {WORKSPACE_MANIFEST_PATH}")
    print(f"  branch: {manifest.get('main_repo', {}).get('branch')}")
    print(f"  head:   {manifest.get('main_repo', {}).get('head')}")
    print(f"  dirty:  {len(manifest.get('main_repo', {}).get('dirty_files', []))} files")


def cmd_refresh_model_fidelity(_args: argparse.Namespace) -> None:
    workspace = tb_load_json(WORKSPACE_MANIFEST_PATH, default=None)
    if not workspace:
        workspace = build_workspace_manifest()
    fidelity = build_model_load_fidelity(workspace)
    print(f"  OK: wrote {MODEL_LOAD_FIDELITY_PATH}")
    print(f"  fidelity_grade: {fidelity.get('fidelity_grade')}")
    print(f"  semantic_patches: {len(fidelity.get('semantic_patches', []))}")


def cmd_build_truth(_args: argparse.Namespace) -> None:
    _refresh_canonical_surfaces()
    print(f"  OK: wrote {CURRENT_TRUTH_PATH}")


def cmd_derive_docs(_args: argparse.Namespace) -> None:
    truth = tb_load_json(CURRENT_TRUTH_PATH, default=None)
    if not truth:
        truth = build_current_truth()
    outputs = derive_docs(truth)
    build_current_truth()
    for path in outputs:
        print(f"  OK: refreshed {path}")


def cmd_publish_experiment(args: argparse.Namespace) -> None:
    artifact = Path(args.from_artifact)
    if not artifact.is_absolute():
        artifact = CAMPAIGN_ROOT / artifact
    result = publish_experiment(args.spec, artifact)
    _refresh_canonical_surfaces()
    if result.get("published"):
        print(f"  OK: published {args.spec}")
        print(f"  verdict: {result.get('verdict')}")
    else:
        print(f"  OK: proposal-only for {args.spec}")
        print(f"  proposal: {result.get('proposal')}")


def cmd_go(_args: argparse.Namespace) -> None:
    """Refresh canonical/runtime surfaces, lint them, regenerate docs, and print current truth."""
    print("╔══════════════════════════════════════════════════════╗")
    print("║  sovereign go — mint_drawer_v1                      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    context = _refresh_canonical_surfaces()
    lint_ok, lint_warnings = _run_lints()
    truth = tb_load_json(CURRENT_TRUTH_PATH, default={}) or context["truth"]

    manifest = tb_load_json(DATASET_MANIFEST_PATH, default={}) or {}
    if manifest:
        if manifest.get("dataset_loads"):
            print(
                f"  PASS  dataset_manifest (dataset_version={manifest.get('dataset_version')}, "
                f"frames={manifest.get('frame_count')})"
            )
        else:
            print("  WARN  dataset_manifest: dataset_loads=False")
    else:
        print("  WARN  dataset_manifest: missing")

    if not lint_ok:
        print()
        print("LINT FAILURES — fix before proceeding.")
        sys.exit(1)

    print()
    print("─── Current Truth ──────────────────────────────────────────")
    print(f"  verdict:  {truth['current'].get('verdict', 'unknown')}")
    print(f"  phase:    {truth['current'].get('phase', 'unknown')}")
    print(f"  gate:     {truth['current'].get('phase_gate', 'unknown')}")
    print(f"  decision: {truth['current'].get('decision', 'unknown')}")

    print()
    print("─── Workspace ──────────────────────────────────────────────")
    workspace = truth.get("workspace", {})
    print(f"  branch:   {workspace.get('main_repo', {}).get('branch')}")
    print(f"  head:     {workspace.get('main_repo', {}).get('head')}")
    print(f"  dirty:    {len(workspace.get('main_repo', {}).get('dirty_files', []))} files")
    print(
        "  external/MINT dirty: "
        f"{len(workspace.get('submodules', {}).get('external/MINT', {}).get('dirty_files', []))} files"
    )

    print()
    print("─── Model Fidelity ─────────────────────────────────────────")
    fidelity = truth.get("model_load_fidelity", {})
    print(f"  grade:    {fidelity.get('fidelity_grade', 'unknown')}")
    print(f"  summary:  {fidelity.get('fidelity_summary', 'unknown')}")
    print(f"  semantic patches: {len(fidelity.get('semantic_patches', []))}")

    claims_view = truth.get("claims", {})
    driving_claims = claims_view.get("current_driving", [])
    historical_claims = claims_view.get("historical_context", [])
    debt_flags = claims_view.get("debt_flags", [])
    lifecycle_review_queue = claims_view.get("lifecycle_review_queue", [])

    print()
    print("─── Claims Driving This Phase ──────────────────────────────")
    if driving_claims:
        for item in driving_claims:
            print(
                f"  [{item.get('status', 'unknown'):>12}] {item.get('claim_id')} "
                f"(rev{item.get('revision')}, {len(item.get('evidence_ids', []))} evidence)"
            )
    else:
        print("  none")
    print(f"  historical context claims: {len(historical_claims)}")
    print(f"  claim debt flags: {len(debt_flags)}")
    print(f"  lifecycle review queue: {len(lifecycle_review_queue)}")
    for item in lifecycle_review_queue[:3]:
        print(
            f"    - [{item.get('priority', '-')}] {item.get('claim_id')} "
            f"-> {item.get('recommended_action')}"
        )
    print(f"  total evidence: {truth.get('evidence', {}).get('total', 0)}")

    if lint_warnings:
        print()
        print("─── Warnings ───────────────────────────────────────────────")
        for warning in lint_warnings[:8]:
            print(f"  {warning}")
        if len(lint_warnings) > 8:
            print(f"  ... +{len(lint_warnings) - 8} more")

    stale_docs = truth["current"].get("stale_docs", [])
    print()
    print("─── Derived Docs ───────────────────────────────────────────")
    print(f"  latest bootstrap: {truth.get('derived_docs', {}).get('latest_bootstrap')}")
    print(f"  stale docs: {len(stale_docs)}")
    for item in stale_docs[:5]:
        print(f"    - {item['path']} ({', '.join(item['reasons'])})")

    print()
    print("─── Next Action ────────────────────────────────────────────")
    next_action = truth["current"].get("next_action") or {}
    if next_action:
        print(f"  Suggested next: [{next_action.get('priority', '-')}] {next_action.get('type', '?')}")
        print(f"    id: {next_action.get('id', '-')}")
        print(f"    target: {next_action.get('target', '-')}")
    else:
        print("  No pending or in-progress actions.")

    print()
    print("─── Quick Ref ──────────────────────────────────────────────")
    print("  current truth: sovereign/current_truth.json")
    print("  workspace:     sovereign/workspace_manifest.json")
    print("  model fidelity: sovereign/model_load_fidelity.json")
    print("  usage guide:   HARNESS_USAGE_GUIDE.md")
    print("  handoff:       sovereign/handoff.md")
    print("  run ledger:    sovereign/run_ledger.yaml")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Harness v2 — Sovereign CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("go", help="One-shot: lint + truth + next actions — run this first")
    sub.add_parser("bootstrap", help="Alias to go")
    sub.add_parser("reconcile", help="Run all lints (dev mode)")
    sub.add_parser("render-truth", help="Render CAMPAIGN_TRUTH.generated.md")
    sub.add_parser("refresh-workspace", help="Write sovereign/workspace_manifest.json")
    sub.add_parser("refresh-model-fidelity", help="Write sovereign/model_load_fidelity.json")
    sub.add_parser("build-truth", help="Write sovereign/current_truth.json and normalize runtime surfaces")
    sub.add_parser("derive-docs", help="Regenerate bootstrap/usage guide/handoff/generated truth")

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

    # pack-dataset
    pd = sub.add_parser("pack-dataset", help="Run pack_dataset.sh — MANDATORY entry point for packing")
    pd.add_argument("--force-archive", action="store_true",
                   help="Archive existing dataset/ before packing (required if dataset/ exists)")
    pd.add_argument("extra_args", nargs="*", help="Extra args passed to run_v58_lerobot_pack.py")

    # verify-manifest
    sub.add_parser("verify-manifest", help="Run train_gate.py — verify dataset is valid")

    pe = sub.add_parser("publish-experiment", help="Validate/publish a spec-backed experiment artifact")
    pe.add_argument("--spec", required=True, help="Experiment spec ID from sovereign/experiment_specs/")
    pe.add_argument("--from", dest="from_artifact", required=True, help="Raw artifact JSON path")

    return p


COMMAND_MAP = {
    "go": cmd_go,
    "bootstrap": cmd_bootstrap,
    "reconcile": cmd_reconcile,
    "render-truth": cmd_render_truth,
    "refresh-workspace": cmd_refresh_workspace,
    "refresh-model-fidelity": cmd_refresh_model_fidelity,
    "build-truth": cmd_build_truth,
    "derive-docs": cmd_derive_docs,
    "revise-claim": cmd_revise_claim,
    "supersede-claim": cmd_supersede_claim,
    "split-claim": cmd_split_claim,
    "propose-revision": cmd_propose_revision,
    "review-proposal": cmd_review_proposal,
    "record-evidence": cmd_record_evidence,
    "close-experiment": cmd_close_experiment,
    "write-handoff": cmd_write_handoff,
    "pack-dataset": cmd_pack_dataset,
    "verify-manifest": cmd_verify_manifest,
    "publish-experiment": cmd_publish_experiment,
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

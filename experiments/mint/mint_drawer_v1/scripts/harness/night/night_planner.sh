#!/usr/bin/env bash
# =============================================================================
# night_planner.sh — Phase 4a
# Night planner: read next_actions.json and sovereign/claims.yaml, generate
# a prioritized task list for the next day.
#
# This script is READ-ONLY — it only reads sovereign state and prints output.
# It does not modify any files.
#
# Output:
#   sovereign/night/night_plan.txt  (generated task list)
#   Also printed to stdout.
#
# Usage:
#   ./night_planner.sh
#
# Exit codes:
#   0 = Plan generated successfully
#   1 = Error reading sovereign files
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
NIGHT_DIR="${CAMPAIGN_ROOT}/scripts/harness/night"
SOVEREIGN="${CAMPAIGN_ROOT}/sovereign"
PLAN_OUTPUT="${NIGHT_DIR}/night_plan.txt"

# ── Helpers ──────────────────────────────────────────────────────────────────

indent() { sed 's/^/  /'; }

echo_divider() {
    echo "─────────────────────────────────────────────────────────────"
}

render_claims() {
    # Render active claims from claims.yaml (brief summary)
    local claims_yaml="${SOVEREIGN}/claims.yaml"
    if [[ ! -f "${claims_yaml}" ]]; then
        echo "  [claims.yaml not found]"
        return
    fi

    echo "  Active Claims (from sovereign/claims.yaml):"
    echo ""

    # Extract claim summary via python (robust YAML parsing)
    python3 - <<'PYEOF'
import sys, yaml
from pathlib import Path

claims_path = Path("sovereign/claims.yaml")
if not claims_path.exists():
    print("  [claims.yaml not found]")
    sys.exit(0)

with open(claims_path) as f:
    data = yaml.safe_load(f)

for claim in data.get("claims", []):
    cid = claim["claim_id"]
    status = claim.get("lifecycle_status", "unknown")
    revisions = claim.get("revisions", [])
    if not revisions:
        continue
    cur = claim.get("current_revision", revisions[-1]["revision"])
    cur_rev = next((r for r in revisions if r["revision"] == cur), revisions[-1])
    stmt = cur_rev.get("statement", "")[:80]
    stmt_short = stmt + ("..." if len(stmt) >= 80 else "")
    ev_ids = cur_rev.get("evidence_ids", [])
    rev_count = len(revisions)
    superseded_count = len([r for r in revisions if r.get("superseded_by")])
    print(f"    [{cid}]")
    print(f"      status: {status} | current_rev: {cur} | revisions: {rev_count} total, {superseded_count} superseded")
    print(f"      scope: {cur_rev.get('scope', 'unknown')}")
    print(f"      statement: {stmt_short}")
    print(f"      evidence: {ev_ids if ev_ids else '(none)'}")
    print("")
PYEOF
}

render_next_actions() {
    # Render next_actions.json
    local na_json="${SOVEREIGN}/next_actions.json"
    if [[ ! -f "${na_json}" ]]; then
        echo "  [next_actions.json not found]"
        return
    fi

    echo "  Next Actions (from sovereign/next_actions.json):"
    echo ""

    python3 - <<'PYEOF'
import sys, json
from pathlib import Path

na_path = Path("sovereign/next_actions.json")
if not na_path.exists():
    print("  [next_actions.json not found]")
    sys.exit(0)

with open(na_path) as f:
    na = json.load(f)

# Get decision/reason
print(f"  Decision: {na.get('decision', 'unknown')}")
print(f"  Generated: {na.get('generated_at', 'unknown')}")
print(f"  Verdict: {na.get('verdict', 'unknown')}")
print(f"  Next phase: {na.get('next_phase', 'unknown')}")
print("")
print("  Actions:")
for action in na.get("actions", []):
    atype = action.get("type", "unknown")
    targets = action.get("targets", [])
    target_str = ", ".join(targets) if targets else action.get("target", "")
    reason = action.get("reason", "")[:100]
    print(f"    - type={atype}")
    if target_str:
        print(f"      target: {target_str}")
    if reason:
        print(f"      reason: {reason}...")
    print("")
PYEOF
}

render_queue() {
    # Render queue from state.json (step-level)
    local state_json="${SOVEREIGN}/state.json"
    if [[ ! -f "${state_json}" ]]; then
        echo "  [state.json not found]"
        return
    fi

    echo "  Campaign Queue (from sovereign/state.json):"
    echo ""

    python3 - <<'PYEOF'
import sys, json
from pathlib import Path

state_path = Path("sovereign/state.json")
if not state_path.exists():
    print("  [state.json not found]")
    sys.exit(0)

with open(state_path) as f:
    state = json.load(f)

queue = state.get("queue", [])
print(f"  Total steps: {len(queue)}")

# Count by status
from collections import Counter
statuses = Counter(item.get("status", "unknown") for item in queue)
for s, c in sorted(statuses.items()):
    print(f"    {s}: {c}")

print("")

# Show pending/blocked steps
pending = [item for item in queue if item.get("status") in ("pending", "blocked", "failed")]
if pending:
    print("  Pending / Blocked / Failed Steps:")
    for item in pending:
        sid = item.get("id", "?")
        status = item.get("status", "?")
        note = item.get("note", "")
        note_short = (note[:100] + "...") if len(note) > 100 else note
        blockers = item.get("blocker_claim_ids", [])
        print(f"    [{status:8s}] {sid}")
        if blockers:
            print(f"               blockers: {', '.join(blockers)}")
        if note_short:
            print(f"               note: {note_short}")
    print("")
PYEOF
}

render_recent_claim_changes() {
    # Show recent claim revisions from claims.yaml
    local claims_yaml="${SOVEREIGN}/claims.yaml"
    if [[ ! -f "${claims_yaml}" ]]; then
        return
    fi

    echo "  Recent Claim Revisions (last 5 changes):"
    echo ""

    python3 - <<'PYEOF'
import sys, yaml
from pathlib import Path
from datetime import datetime

claims_path = Path("sovereign/claims.yaml")
if not claims_path.exists():
    sys.exit(0)

with open(claims_path) as f:
    data = yaml.safe_load(f)

# Collect all revisions with timestamps
all_revs = []
for claim in data.get("claims", []):
    cid = claim["claim_id"]
    for rev in claim.get("revisions", []):
        ts = rev.get("created_at", "1970-01-01T00:00:00+00:00")
        all_revs.append({
            "cid": cid,
            "rev": rev["revision"],
            "ct": rev.get("change_type", ""),
            "status": rev.get("status", ""),
            "reason": rev.get("change_reason", ""),
            "created_at": ts,
        })

# Sort by timestamp descending
def parse_ts(item):
    try:
        return datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
    except:
        return datetime.min

all_revs.sort(key=parse_ts, reverse=True)

for item in all_revs[:5]:
    cid = item["cid"]
    rev = item["rev"]
    ct = item["ct"]
    reason = (item["reason"][:80] + "...") if len(item["reason"]) > 80 else item["reason"]
    ts = item["created_at"][:16]  # YYYY-MM-DDTHH:MM
    print(f"    {ts}  {cid}@{rev}  [{ct}]")
    if reason:
        print(f"              {reason}")
    print("")
PYEOF
}

generate_plan() {
    local ts
    ts="$(date '+%Y-%m-%dT%H:%M:%S%z')"

    cat > "${PLAN_OUTPUT}" <<HEADER
================================================================================
NIGHT PLAN — ${ts}
Campaign: mint_drawer_v1
Source: sovereign/next_actions.json + sovereign/claims.yaml + sovereign/state.json
================================================================================

HEADER

    {
        echo "## CLAIMS SUMMARY"
        echo_divider
        render_claims
        echo ""

        echo "## RECENT CLAIM CHANGES"
        echo_divider
        render_recent_claim_changes
        echo ""

        echo "## NEXT ACTIONS"
        echo_divider
        render_next_actions
        echo ""

        echo "## CAMPAIGN QUEUE"
        echo_divider
        render_queue
        echo ""

        echo "## RECOMMENDED NEXT STEPS (for human review)"
        echo_divider
        python3 - <<'PYEOF'
import json, yaml
from pathlib import Path
from datetime import datetime

claims_path = Path("sovereign/claims.yaml")
na_path = Path("sovereign/next_actions.json")
state_path = Path("sovereign/state.json")

with open(na_path) as f:
    na = json.load(f)
with open(state_path) as f:
    state = json.load(f)
with open(claims_path) as f:
    claims = yaml.safe_load(f)

# Determine blockers
decision = na.get("decision", "unknown")
verdict = na.get("verdict", "unknown")
next_phase = na.get("next_phase", "unknown")
targets = []
for a in na.get("actions", []):
    targets.extend(a.get("targets", []))
    if a.get("target"):
        targets.append(a.get("target"))

print(f"  Current decision: {decision}")
print(f"  Verdict: {verdict}")
print(f"  Next phase: {next_phase}")
print(f"  Awaiting: {', '.join(targets) if targets else '(none specified)'}")
print("")

# Check for active claims that are contradicted
contradicted = []
candidate = []
for c in claims.get("claims", []):
    ls = c.get("lifecycle_status", "active")
    if ls == "active":
        revs = c.get("revisions", [])
        if revs:
            cur = c.get("current_revision", revs[-1]["revision"])
            cur_rev = next((r for r in revs if r["revision"] == cur), revs[-1])
            st = cur_rev.get("status", "")
            if st == "contradicted":
                contradicted.append(c["claim_id"])
            elif st == "candidate":
                candidate.append(c["claim_id"])

if contradicted:
    print(f"  Active contradicted claims: {', '.join(contradicted)}")
    print("    → Review: Are these still accurate? Should they be superseded or archived?")
    print("")
if candidate:
    print(f"  Active candidate claims: {', '.join(candidate)}")
    print("    → Review: What evidence is needed to promote to 'supported' or 'contradicted'?")
    print("")

print("  Recommended actions for next session:")
print("    1. Run: python scripts/harness/sovereign_cli.py bootstrap")
print("    2. Review: sovereign/handoff.md for recent activity")
print("    3. Review: sovereign/CAMPAIGN_TRUTH.generated.md for current state")
print("    4. If blocked by external resources, verify status of:")
for t in targets:
    print(f"       - {t}")
print("    5. If unblocked, proceed with next_phase: {next_phase}")
PYEOF

        echo ""
        echo "================================================================================"
        echo "Plan generated: ${ts}"
        echo "================================================================================"

    } >> "${PLAN_OUTPUT}"

    echo "Plan written to: ${PLAN_OUTPUT}"
}

# ── Main ─────────────────────────────────────────────────────────────────────

mkdir -p "${NIGHT_DIR}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  NIGHT PLANNER — $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check sovereign files exist
if [[ ! -f "${SOVEREIGN}/claims.yaml" ]]; then
    echo "ERROR: sovereign/claims.yaml not found. Aborting."
    exit 1
fi

# Generate plan
generate_plan

# Print to stdout
echo ""
cat "${PLAN_OUTPUT}"
echo ""
echo "Night planner complete."

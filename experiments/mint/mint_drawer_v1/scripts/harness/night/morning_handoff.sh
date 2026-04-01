#!/usr/bin/env bash
# =============================================================================
# morning_handoff.sh — Phase 4a
# Morning handoff: aggregate watcher + planner outputs, update sovereign/handoff.md
#
# This script:
#   1. Reads watcher output (if any) and planner output (night_plan.txt)
#   2. Reads sovereign/claims.yaml to surface recent claim changes
#   3. Updates sovereign/handoff.md with a structured morning handoff block
#
# The handoff.md appends a new handoff section at the top, preserving history.
#
# Usage:
#   ./morning_handoff.sh
#
# Exit codes:
#   0 = Handoff generated successfully
#   1 = Error
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
NIGHT_DIR="${CAMPAIGN_ROOT}/scripts/harness/night"
SOVEREIGN="${CAMPAIGN_ROOT}/sovereign"
WATCHER_LOG="${NIGHT_DIR}/watcher.log"
NIGHT_PLAN="${NIGHT_DIR}/night_plan.txt"
HANDOFF="${SOVEREIGN}/handoff.md"
HANDOFF_BACKUP="${NIGHT_DIR}/handoff_backup_$(date '+%Y%m%d').md"

# ── Helpers ──────────────────────────────────────────────────────────────────

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }

render_claim_changes() {
    # Render recent claim revisions for the handoff
    python3 - <<'PYEOF'
import sys, yaml
from pathlib import Path
from datetime import datetime

claims_path = Path("sovereign/claims.yaml")
if not claims_path.exists():
    print("  [claims.yaml not found]")
    sys.exit(0)

with open(claims_path) as f:
    data = yaml.safe_load(f)

all_revs = []
for claim in data.get("claims", []):
    cid = claim["claim_id"]
    ls = claim.get("lifecycle_status", "unknown")
    cur = claim.get("current_revision", 1)
    for rev in claim.get("revisions", []):
        ts = rev.get("created_at", "1970-01-01T00:00:00+00:00")
        all_revs.append({
            "cid": cid,
            "rev_num": rev["revision"],
            "ct": rev.get("change_type", ""),
            "status": rev.get("status", ""),
            "reason": rev.get("change_reason", ""),
            "evidence": rev.get("evidence_ids", []),
            "scope": rev.get("scope", ""),
            "created_at": ts,
            "is_current": rev["revision"] == cur,
            "lifecycle_status": ls,
        })

def parse_ts(item):
    try:
        return datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
    except:
        return datetime.min

all_revs.sort(key=parse_ts, reverse=True)

# Show recent 5 changes
print("  Recent Claim Changes (last 5 revisions):")
for item in all_revs[:5]:
    cur_marker = " ←CURRENT" if item["is_current"] else ""
    reason = (item["reason"][:120] + "...") if len(item["reason"]) > 120 else item["reason"]
    ts = item["created_at"][:16]
    print(f"    {ts}  {item['cid']}@{item['rev_num']}  [{item['ct']}]  status={item['status']}{cur_marker}")
    if reason:
        print(f"              {reason}")
    print("")
PYEOF
}

render_watcher_summary() {
    if [[ -f "${WATCHER_LOG}" ]]; then
        # Get last entry
        tail -20 "${WATCHER_LOG}" 2>/dev/null || echo "  [watcher log unreadable]"
    else
        echo "  [no watcher log found — watcher may not have run]"
    fi
}

render_night_plan() {
    if [[ -f "${NIGHT_PLAN}" ]]; then
        # Get recommended next steps
        grep -A 20 "RECOMMENDED NEXT STEPS" "${NIGHT_PLAN}" 2>/dev/null | head -30 || \
            echo "  [night plan unreadable]"
    else
        echo "  [no night plan found]"
    fi
}

update_handoff() {
    local ts
    ts="$(timestamp)"

    # Backup existing handoff
    if [[ -f "${HANDOFF}" ]]; then
        cp "${HANDOFF}" "${HANDOFF_BACKUP}"
    fi

    # Build new handoff section
    local new_section
    new_section=$(cat <<SECT
<!-- MORNING HANDOFF — ${ts} -->
## Morning Handoff — ${ts}

### System Status
**Watcher**: $(render_watcher_summary | head -1)

### Claim Changes (since last session)
$(render_claim_changes)

### Night Planner Recommendations
$(render_night_plan | sed 's/^/    /')

---
SECT
)

    # If handoff.md exists, insert after the header block
    if [[ -f "${HANDOFF}" ]] && [[ -s "${HANDOFF}" ]]; then
        # Check if it already has the GENERATED header
        if grep -q "<!-- GENERATED FILE" "${HANDOFF}" 2>/dev/null; then
            # Generated file — replace content
            python3 - <<PYEOF
import sys
h_path = Path("${HANDOFF}")
ts = "${ts}"

# Read existing
with open(h_path) as f:
    content = f.read()

# Replace the timestamp in the header if present
import re
content = re.sub(
    r'<!-- Generated at: [^\s]+ -->',
    f'<!-- Generated at: {ts} -->',
    content
)
print(content)
PYEOF
        else
            # Human-written handoff — prepend new section
            {
                echo "${new_section}"
                echo ""
                cat "${HANDOFF}"
            } > "${HANDOFF}.tmp"
            mv "${HANDOFF}.tmp" "${HANDOFF}"
        fi
    else
        # No existing handoff — create from scratch
        cat > "${HANDOFF}" <<HEADER
# Sovereign Handoff — mint_drawer_v1

<!-- GENERATED FILE — DO NOT EDIT MANUALLY -->
<!-- Source: scripts/harness/night/morning_handoff.sh -->
<!-- Generated at: ${ts} -->
<!-- Use: sovereign_cli.py write-handoff for manual updates -->

${new_section}
HEADER
    fi

    echo "Handoff updated: ${HANDOFF}"
    [[ -f "${HANDOFF_BACKUP}" ]] && echo "Backup: ${HANDOFF_BACKUP}"
}

# ── Main ─────────────────────────────────────────────────────────────────────

mkdir -p "${NIGHT_DIR}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  MORNING HANDOFF — $(timestamp)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Update handoff
update_handoff

echo ""
echo "--- Morning Handoff Summary ---"
echo ""

# Print key info
echo "Active Claims:"
render_claim_changes | grep -v "^  Recent Claim Changes"

echo ""
echo "Night Plan Recommendations:"
render_night_plan | grep -E "^[0-9]+\.|Current decision|Next phase|Awaiting" | head -10

echo ""
echo "Full handoff: ${HANDOFF}"
echo "Morning handoff complete."

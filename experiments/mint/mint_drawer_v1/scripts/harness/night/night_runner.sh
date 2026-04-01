#!/usr/bin/env bash
# =============================================================================
# night_runner.sh — Phase 4a
# Master night runner: orchestrates the nightly automation pipeline.
#
# Orchestration order:
#   Phase 4a (P3a):
#     1. night_watcher.sh  — GPU, disk, tmux, log health snapshot
#     2. night_planner.sh  — Read sovereign, generate task plan
#     3. morning_handoff.sh — Update sovereign/handoff.md with summary
#
# Phase 4b (P3b) — runs after P3a proves stable (≥3 sessions):
#     4. night_verifier.sh — Run 02_reconcile_sources.py + 03_claim_lint.py + 04_action_lint.py
#     5. night_executor.sh --dry-run — Simulate bounded commands
#
# Phase 4c (P3c) — runs after P3a+P3b prove stable (≥5 sessions):
#     6. night_executor.sh  — Execute bounded whitelisted tasks
#
# Key constraints (P4 rule):
#   - Night runner MAY call: propose-revision (writes to sovereign/proposals/)
#   - Night runner MUST NOT call: revise-claim, supersede-claim, split-claim
#   - Claim evolution always requires human/session-agent review
#
# Usage:
#   ./night_runner.sh              # Full P3a pipeline
#   ./night_runner.sh --watcher    # Watcher only
#   ./night_runner.sh --planner    # Planner only
#   ./night_runner.sh --handoff    # Morning handoff only
#   ./night_runner.sh --verifier   # P3b verifier
#   ./night_runner.sh --dry-run    # P3b dry-run executor
#   ./night_runner.sh --phase4c    # P3c full execution (requires --confirm)
#   ./night_runner.sh --confirm    # Confirm P3c execution
#   ./night_runner.sh --status     # Show night runner status
#
# Exit codes:
#   0 = Pipeline completed
#   1 = One or more steps failed
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
NIGHT_DIR="${CAMPAIGN_ROOT}/scripts/harness/night"
HARNESS_DIR="${CAMPAIGN_ROOT}/scripts/harness"
RUNNER_LOG="${NIGHT_DIR}/runner.log"
LAST_RUN_LOG="${NIGHT_DIR}/last_run.log"
SESSION_LOG="${NIGHT_DIR}/session_$(date '+%Y%m%d_%H%M%S').log"

# ── Helpers ──────────────────────────────────────────────────────────────────

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }

log() {
    local level="$1"; shift
    echo "[$(timestamp)] [${level}] $*" | tee -a "${RUNNER_LOG}" "${LAST_RUN_LOG}"
}

run_step() {
    local name="$1"; shift
    local cmd="$*"
    local start end elapsed rc
    start=$(date +%s)
    log "INFO" "Starting: ${name}"
    log "INFO" "Command: ${cmd}"

    # Redirect both to runner log and session log
    if eval "${cmd}" >> "${SESSION_LOG}" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    end=$(date +%s)
    elapsed=$((end - start))
    if (( rc == 0 )); then
        log "OK"   "Completed: ${name} (${elapsed}s)"
        return 0
    else
        log "FAIL" "Failed: ${name} (exit ${rc}, ${elapsed}s)"
        return "${rc}"
    fi
}

check_bootstrap() {
    # Verify bootstrap scripts exist before running
    for script in sovereign_cli.py 01_sync_pointers.sh 02_reconcile_sources.py 03_claim_lint.py; do
        if [[ ! -f "${HARNESS_DIR}/${script}" ]]; then
            log "ERROR" "Bootstrap dependency missing: ${script}"
            return 1
        fi
    done
    return 0
}

get_stability_count() {
    # Count consecutive successful P3a runs (by presence of session logs)
    local count=0
    for logfile in "${NIGHT_DIR}"/session_*.log; do
        if [[ -f "${logfile}" ]]; then
            count=$((count + 1))
        fi
    done
    echo "${count}"
}

print_status() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  NIGHT RUNNER STATUS"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "  Campaign root: ${CAMPAIGN_ROOT}"
    echo "  Night dir:     ${NIGHT_DIR}"
    echo "  Last run log: ${LAST_RUN_LOG}"
    echo ""

    if [[ -f "${LAST_RUN_LOG}" ]]; then
        echo "  Last run: $(head -1 "${LAST_RUN_LOG}" | cut -d']' -f2- | tr -d ' ')"
        echo "  Last run exit: $(grep "Completed\|Failed" "${LAST_RUN_LOG}" | tail -1)"
        echo ""
    fi

    local sessions
    sessions=$(get_stability_count)
    echo "  Total sessions: ${sessions}"
    if (( sessions >= 3 )); then
        echo "  P3a stable: YES"
    else
        echo "  P3a stable: NO (need $((3 - sessions)) more sessions)"
    fi
    echo "  P3b ready:  $( ((sessions >= 3)) && echo 'YES (if P3a stable)' || echo 'NO (need P3a first)')"
    if (( sessions >= 5 )); then
        echo "  P3c ready:  YES (if P3b stable)"
    else
        echo "  P3c ready:  NO (need $((5 - sessions)) more sessions)"
    fi
    echo ""

    echo "  Night outputs:"
    for f in watcher.log planner.log handoff.log verifier.log; do
        if [[ -f "${NIGHT_DIR}/${f}" ]]; then
            local age
            age=$(($(date +%s) - $(stat -c %Y "${NIGHT_DIR}/${f}" 2>/dev/null || echo $(date +%s))))
            echo "    ${f}: $((${age}/60))m ago"
        else
            echo "    ${f}: not found"
        fi
    done
    echo ""
    echo "═══════════════════════════════════════════════════════════"
}

# ── Phase 4a Steps ────────────────────────────────────────────────────────────

step_watcher() {
    run_step "night_watcher" "bash '${NIGHT_DIR}/night_watcher.sh' --snapshot" \
        | tee "${NIGHT_DIR}/watcher.log"
}

step_planner() {
    run_step "night_planner" "bash '${NIGHT_DIR}/night_planner.sh'" \
        | tee "${NIGHT_DIR}/planner.log"
}

step_handoff() {
    run_step "morning_handoff" "bash '${NIGHT_DIR}/morning_handoff.sh'" \
        | tee "${NIGHT_DIR}/handoff.log"
}

# ── Phase 4b Steps ────────────────────────────────────────────────────────────

step_verifier() {
    log "INFO" "Phase 4b: Running lint verifiers"
    local rc=0

    for script in 02_reconcile_sources.py 03_claim_lint.py 04_action_lint.py; do
        run_step "verifier:${script}" \
            "cd '${CAMPAIGN_ROOT}' && python '${HARNESS_DIR}/${script}'" \
            || { log "WARN" "Verifier ${script} returned non-zero"; rc=1; }
        echo "" >> "${NIGHT_DIR}/verifier.log"
    done

    return "${rc}"
}

step_executor_dryrun() {
    log "INFO" "Phase 4b: Dry-run executor (no file modifications)"
    # TODO: implement bounded dry-run
    log "INFO" "Dry-run: no operations executed (not yet configured)"
    return 0
}

# ── Phase 4c Steps ────────────────────────────────────────────────────────────

step_executor() {
    log "INFO" "Phase 4c: Bounded executor (whitelisted tasks only)"
    # TODO: implement bounded executor with whitelist
    log "INFO" "Executor: no operations executed (whitelist not yet configured)"
    log "INFO" "To configure P3c, edit night_executor.sh with task whitelist"
    return 0
}

# ── Main Pipeline ─────────────────────────────────────────────────────────────

PHASE4A=true
PHASE4B=false
PHASE4C=false
CONFIRM=false

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --watcher)    PHASE4A=true; PHASE4B=false; PHASE4C=false; shift ;;
        --planner)    PHASE4A=true; PHASE4B=false; PHASE4C=false; shift ;;
        --handoff)    PHASE4A=true; PHASE4B=false; PHASE4C=false; shift ;;
        --verifier)   PHASE4A=false; PHASE4B=true; PHASE4C=false; shift ;;
        --dry-run)    PHASE4A=false; PHASE4B=true; PHASE4C=false; shift ;;
        --phase4c)    PHASE4A=true; PHASE4B=true; PHASE4C=true; shift ;;
        --confirm)    CONFIRM=true; shift ;;
        --status)     print_status; exit 0 ;;
        --help|-h)    print_status; exit 0 ;;
        *)            echo "Unknown: $1"; exit 1 ;;
    esac
done

# ── Startup ───────────────────────────────────────────────────────────────────

mkdir -p "${NIGHT_DIR}"

log "INFO" "========================================"
log "INFO" "Night runner started — $(timestamp)"
log "INFO" "Campaign: ${CAMPAIGN_ROOT}"
log "INFO" "========================================"

echo "" >> "${SESSION_LOG}"
echo "=== Night Runner Session — $(timestamp) ===" >> "${SESSION_LOG}"
echo "" >> "${SESSION_LOG}"

FAILED=0

# ── Phase 4a ─────────────────────────────────────────────────────────────────

if [[ "${PHASE4A}" == "true" ]]; then
    log "INFO" "=== Phase 4a: Watcher + Planner + Handoff ==="

    check_bootstrap || { log "ERROR" "Bootstrap checks failed"; exit 1; }

    step_watcher  || { log "WARN" "Watcher failed"; }
    step_planner  || { log "WARN" "Planner failed"; FAILED=1; }
    step_handoff  || { log "WARN" "Handoff update failed"; FAILED=1; }

    log "INFO" "=== Phase 4a complete ==="
fi

# ── Phase 4b ─────────────────────────────────────────────────────────────────

if [[ "${PHASE4B}" == "true" ]]; then
    local sessions
    sessions=$(get_stability_count)
    if (( sessions < 3 )); then
        log "WARN" "P3b requires >=3 stable P3a sessions (found ${sessions}). Skipping."
    else
        log "INFO" "=== Phase 4b: Verifier + Dry-run ==="
        step_verifier    || { log "WARN" "Verifier returned non-zero"; FAILED=1; }
        step_executor_dryrun || { log "WARN" "Dry-run failed"; FAILED=1; }
        log "INFO" "=== Phase 4b complete ==="
    fi
fi

# ── Phase 4c ─────────────────────────────────────────────────────────────────

if [[ "${PHASE4C}" == "true" ]]; then
    local sessions
    sessions=$(get_stability_count)
    if (( sessions < 5 )); then
        log "WARN" "P3c requires >=5 stable P3a+P3b sessions (found ${sessions}). Skipping."
    elif [[ "${CONFIRM}" != "true" ]]; then
        log "WARN" "P3c requires --confirm flag to execute. Skipping bounded executor."
    else
        log "INFO" "=== Phase 4c: Bounded Executor ==="
        step_executor || { log "WARN" "Executor failed"; FAILED=1; }
        log "INFO" "=== Phase 4c complete ==="
    fi
fi

# ── Shutdown ─────────────────────────────────────────────────────────────────

log "INFO" "========================================"
if (( FAILED )); then
    log "WARN" "Night runner completed with warnings — exit ${FAILED}"
else
    log "OK" "Night runner completed successfully"
fi
log "INFO" "Session log: ${SESSION_LOG}"
log "INFO" "========================================"

exit "${FAILED}"

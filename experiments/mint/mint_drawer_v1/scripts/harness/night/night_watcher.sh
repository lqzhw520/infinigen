#!/usr/bin/env bash
# =============================================================================
# night_watcher.sh — Phase 4a
# Night watcher: read-only monitoring of GPU, disk, tmux sessions, and logs.
#
# Runs continuously (or as a single snapshot) checking:
#   - GPU memory usage (nvidia-smi)
#   - Disk space on key paths
#   - tmux session health
#   - Log file growth
#   - Night runner output freshness
#
# Usage:
#   ./night_watcher.sh              # Single snapshot
#   ./night_watcher.sh --daemon     # Run continuously (poll every 60s)
#   ./night_watcher.sh --daemon -i N # Run with N-second interval
#
# Output:
#   Writes to NIGHT_WATCHER_LOG (default: sovereign/night/watcher.log)
#   Always prints summary to stdout.
#
# Exit codes:
#   0 = All systems nominal
#   1 = WARNING found (non-fatal, informational)
#   2 = CRITICAL found (disk/GPU near exhaustion)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
NIGHT_DIR="${CAMPAIGN_ROOT}/scripts/harness/night"
SOVEREIGN="${CAMPAIGN_ROOT}/sovereign"
OUTPUT_DIR="${NIGHT_DIR}"
WATCHER_LOG="${OUTPUT_DIR}/watcher.log"

# Thresholds
GPU_MEMORY_WARN_MB=8192
GPU_MEMORY_CRIT_MB=2048
DISK_WARN_PCT=85
DISK_CRIT_PCT=92
LOG_DIRS=(
    "${CAMPAIGN_ROOT}/runtime"
    "${CAMPAIGN_ROOT}/artifacts"
    "${SOVEREIGN}"
)
STALE_THRESHOLD_SEC=7200      # 2 hours — warn if no new log entries

# Daemon mode
DAEMON=false
INTERVAL=60
SNAPSHOT=false

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --daemon) DAEMON=true; shift ;;
        -i) INTERVAL="$2"; shift 2 ;;
        --snapshot) SNAPSHOT=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────

log_msg() {
    local level="$1"; shift
    local ts
    ts="$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "[${ts}] [${level}] $*" | tee -a "${WATCHER_LOG}"
}

check_gpu() {
    # Returns: GPU_OK | GPU_WARN | GPU_CRIT
    if ! command -v nvidia-smi &>/dev/null; then
        echo "GPU_CHECK_SKIP (nvidia-smi not found)"
        return
    fi

    local free_mem
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ -z "${free_mem}" ]]; then
        echo "GPU_CHECK_SKIP (nvidia-smi returned empty)"
        return
    fi

    if (( free_mem < GPU_MEMORY_CRIT_MB )); then
        echo "GPU_CRIT:${free_mem}MiB_free"
    elif (( free_mem < GPU_MEMORY_WARN_MB )); then
        echo "GPU_WARN:${free_mem}MiB_free"
    else
        echo "GPU_OK:${free_mem}MiB_free"
    fi
}

check_disk() {
    # Check disk usage for a given path, returns: OK | WARN | CRIT
    local path="$1"
    local usage
    if ! df -k "${path}" &>/dev/null; then
        echo "DISK_SKIP:${path}"
        return
    fi
    usage=$(df -k "${path}" | awk 'NR==2 {print $5}' | tr -d '%')
    if [[ -z "${usage}" ]]; then
        echo "DISK_SKIP:${path}"
        return
    fi
    if (( usage > DISK_CRIT_PCT )); then
        echo "DISK_CRIT:${usage}%_used"
    elif (( usage > DISK_WARN_PCT )); then
        echo "DISK_WARN:${usage}%_used"
    else
        echo "DISK_OK:${usage}%_used"
    fi
}

check_tmux_sessions() {
    # Returns list of active night tmux sessions
    local sessions
    sessions=$(tmux list-sessions 2>/dev/null | grep -v "no server" || echo "")
    if [[ -z "${sessions}" ]]; then
        echo "TMUX_NONE"
    else
        local count
        count=$(echo "${sessions}" | wc -l | tr -d ' ')
        echo "TMUX_OK:${count}_sessions"
        echo "${sessions}" | while IFS= read -r line; do
            echo "  ${line}"
        done
    fi
}

check_log_growth() {
    # Check if log files have been updated recently
    local stale_count=0
    local now_ts
    now_ts=$(date +%s)

    for dir in "${LOG_DIRS[@]}"; do
        if [[ ! -d "${dir}" ]]; then continue; fi
        find "${dir}" -name "*.log" -o -name "*.json" 2>/dev/null | while IFS= read -r f; do
            if [[ -f "${f}" ]]; then
                local mtime
                mtime=$(stat -c %Y "${f}" 2>/dev/null || echo "${now_ts}")
                local age=$(( now_ts - mtime ))
                if (( age > STALE_THRESHOLD_SEC )); then
                    stale_count=$(( stale_count + 1 ))
                fi
            fi
        done
    done

    if (( stale_count > 0 )); then
        echo "LOG_STALE:${stale_count}_files"
    else
        echo "LOG_OK"
    fi
}

check_night_runner_health() {
    # Check if night runner has produced recent output
    local runner_output="${NIGHT_DIR}/last_run.log"
    if [[ -f "${runner_output}" ]]; then
        local age
        age=$(($(date +%s) - $(stat -c %Y "${runner_output}" 2>/dev/null || echo $(date +%s))))
        if (( age > 28800 )); then  # > 8 hours
            echo "RUNNER_STALE:${age}s_since_last_run"
        else
            echo "RUNNER_OK:${age}s_since_last_run"
        fi
    else
        echo "RUNNER_NEVER_RUN"
    fi
}

print_snapshot() {
    local gpu_result disk_result tmux_result log_result runner_result
    local has_warn=0 has_crit=0

    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  NIGHT WATCHER — $(date '+%Y-%m-%d %H:%M:%S%z')"
    echo "═══════════════════════════════════════════════════════════"
    echo ""

    # GPU
    gpu_result=$(check_gpu)
    echo "  GPU:    ${gpu_result}"
    if [[ "${gpu_result}" == GPU_CRIT:* ]]; then
        has_crit=1
    elif [[ "${gpu_result}" == GPU_WARN:* ]]; then
        has_warn=1
    fi
    echo ""

    # Disk
    echo "  DISK:   "
    for dir in "${LOG_DIRS[@]}"; do
        disk_result=$(check_disk "${dir}")
        echo "    ${dir}: ${disk_result}"
        if [[ "${disk_result}" == DISK_CRIT:* ]]; then
            has_crit=1
        elif [[ "${disk_result}" == DISK_WARN:* ]]; then
            has_warn=1
        fi
    done
    echo ""

    # tmux
    echo "  TMUX:   "
    check_tmux_sessions | while IFS= read -r line; do echo "    ${line}"; done
    echo ""

    # Logs
    echo "  LOGS:   $(check_log_growth)"
    echo ""

    # Night runner
    echo "  RUNNER: $(check_night_runner_health)"
    echo ""

    # Summary
    echo "  ─────────────────────────────────────────────────────────"
    if (( has_crit )); then
        echo "  STATUS: CRITICAL — one or more critical issues detected"
        echo "═══════════════════════════════════════════════════════════"
        return 2
    elif (( has_warn )); then
        echo "  STATUS: WARNING — one or more warnings detected"
        echo "═══════════════════════════════════════════════════════════"
        return 1
    else
        echo "  STATUS: OK — all systems nominal"
        echo "═══════════════════════════════════════════════════════════"
        return 0
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────

mkdir -p "${OUTPUT_DIR}"

if [[ "${SNAPSHOT}" == "true" ]] || [[ "${DAEMON}" == "false" ]]; then
    # Single snapshot
    print_snapshot
    exit $?
else
    # Daemon loop
    echo "Night watcher running in daemon mode (interval: ${INTERVAL}s)"
    echo "Log: ${WATCHER_LOG}"
    echo "Press Ctrl+C to stop."
    while true; do
        print_snapshot >> "${WATCHER_LOG}" 2>&1
        rc=$?
        sleep "${INTERVAL}"
    done
fi

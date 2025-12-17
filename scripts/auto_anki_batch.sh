#!/bin/bash
#
# auto_anki_batch.sh - Automated batch processing with usage-based throttling
#
# Discovers months with unprocessed files and processes only those months.
# Monitors LLM usage and waits when usage exceeds sustainable pace.
# Exits when all discovered months have been processed (single-pass mode).
#
# Usage:
#   ./scripts/auto_anki_batch.sh                   # process unprocessed files
#   BACKFILL_MODE=1 ./scripts/auto_anki_batch.sh   # reprocess zero-card files
#   PREVENT_SLEEP=0 ./scripts/auto_anki_batch.sh   # don't inhibit sleep (macOS)
#   Press Ctrl+C to stop gracefully
#
# Requirements:
#   - LLM backend configured (Codex or Claude Code)
#   - Anki running with AnkiConnect plugin
#   - uv installed and auto-anki project set up
#

set -o pipefail

# Preserve original stdout on FD 3 so we can stream live output from within
# command substitutions (used below when capturing function output).
exec 3>&1

# =============================================================================
# Configuration
# =============================================================================

# Determine script and project directories (portable, no hard-coded paths)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CODEX_USAGE_SCRIPT="$SCRIPT_DIR/codex-usage.sh"
CLAUDE_USAGE_SCRIPT="$SCRIPT_DIR/claude-usage.sh"

# Verify usage scripts exist (warnings only - only the configured backend's script is required)
if [[ ! -x "$CODEX_USAGE_SCRIPT" ]]; then
    echo "Warning: codex-usage.sh not found at $CODEX_USAGE_SCRIPT" >&2
fi
if [[ ! -x "$CLAUDE_USAGE_SCRIPT" ]]; then
    echo "Warning: claude-usage.sh not found at $CLAUDE_USAGE_SCRIPT" >&2
fi

# Throttling settings
PACE_BUFFER=10          # Wait if usage% > pace% + PACE_BUFFER
WAIT_DURATION=900       # 15 minutes in seconds
PREVENT_SLEEP=${PREVENT_SLEEP:-1}  # macOS: use caffeinate to prevent idle sleep while running

# Backfill mode: reprocess files that generated 0 cards
# Usage: BACKFILL_MODE=1 ./scripts/auto_anki_batch.sh
BACKFILL_MODE=${BACKFILL_MODE:-0}

# Retry settings
MAX_RETRIES=3
RETRY_DELAY=60          # seconds between retries

# Logging
LOG_DIR="$PROJECT_DIR/auto_anki_runs"
LOG_FILE="$LOG_DIR/batch_$(date +%Y%m%d_%H%M%S).log"

# =============================================================================
# State
# =============================================================================

current_month=""
month_index=0
declare -a MONTHS
SLEEP_INHIBITOR_PID=""

# =============================================================================
# Signal Handling
# =============================================================================

cleanup() {
    echo ""
    log_message "INFO" "Received interrupt signal. Shutting down gracefully..."
    log_message "INFO" "Last processed month: $current_month"
    log_message "INFO" "Run 'uv run auto-anki-progress' to see current status"
    log_message "INFO" "Log file saved to: $LOG_FILE"
    stop_sleep_inhibitor || true
    exit 0
}

trap cleanup SIGINT SIGTERM
trap 'stop_sleep_inhibitor || true' EXIT

# =============================================================================
# Utility Functions
# =============================================================================

start_sleep_inhibitor() {
    if [[ "${PREVENT_SLEEP}" != "1" ]]; then
        return 0
    fi

    if [[ "$(uname -s)" == "Darwin" ]] && command -v caffeinate >/dev/null 2>&1; then
        caffeinate -i -w $$ &
        SLEEP_INHIBITOR_PID=$!
        log_message "INFO" "Preventing sleep via caffeinate (pid $SLEEP_INHIBITOR_PID)"
    fi
}

stop_sleep_inhibitor() {
    if [[ -n "${SLEEP_INHIBITOR_PID}" ]] && kill -0 "${SLEEP_INHIBITOR_PID}" 2>/dev/null; then
        kill "${SLEEP_INHIBITOR_PID}" 2>/dev/null || true
        wait "${SLEEP_INHIBITOR_PID}" 2>/dev/null || true
    fi
    SLEEP_INHIBITOR_PID=""
}

sleep_wall_clock() {
    local seconds="$1"
    local deadline=$(( $(date +%s) + seconds ))
    while true; do
        local now
        now=$(date +%s)
        if (( now >= deadline )); then
            return 0
        fi
        local remaining=$(( deadline - now ))
        local chunk=60
        if (( remaining < chunk )); then
            chunk=$remaining
        fi
        sleep "$chunk"
    done
}

log_message() {
    local level="$1"
    local message="$2"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE" >&3
}

# Detect LLM backend from config file
get_llm_backend() {
    local config_file="$PROJECT_DIR/auto_anki_config.json"
    if [[ -f "$config_file" ]]; then
        local backend
        backend=$(jq -r '.llm_backend // "codex"' "$config_file" 2>/dev/null)
        echo "${backend:-codex}"
    else
        echo "codex"
    fi
}

# Get codex usage info from the usage script
# Returns: "pct pace" (integers) or empty string on error
get_codex_usage() {
    local output
    output=$("$CODEX_USAGE_SCRIPT" -v 2>&1)
    if [[ $? -ne 0 ]]; then
        echo ""
        return 1
    fi

    local usage_line
    usage_line=$(echo "$output" | grep "Chat 5h:")

    if [[ -z "$usage_line" ]]; then
        echo ""
        return 1
    fi

    # Extract pct (integer) and pace (float, we'll truncate to int)
    local pct pace
    pct=$(echo "$usage_line" | sed -n 's/.*pct=\([0-9]*\).*/\1/p')
    pace=$(echo "$usage_line" | sed -n 's/.*pace=\([0-9.]*\)%.*/\1/p')

    # Truncate pace to integer
    local pace_int
    pace_int=${pace%.*}

    if [[ -z "$pct" || -z "$pace_int" ]]; then
        echo ""
        return 1
    fi

    echo "$pct $pace_int"
}

# Get Claude Code usage info from the usage script
# Returns: "pct pace" (integers) or empty string on error
get_claude_usage() {
    local output
    output=$("$CLAUDE_USAGE_SCRIPT" -v 2>&1)
    if [[ $? -ne 0 ]]; then
        echo ""
        return 1
    fi

    local usage_line
    usage_line=$(echo "$output" | grep "5-hour:")

    if [[ -z "$usage_line" ]]; then
        echo ""
        return 1
    fi

    # Extract pct and pace (both are floats with %, truncate to int)
    local pct pace
    pct=$(echo "$usage_line" | sed -n 's/.*pct=\([0-9.]*\)%.*/\1/p')
    pace=$(echo "$usage_line" | sed -n 's/.*pace=\([0-9.]*\)%.*/\1/p')

    # Truncate to integers
    local pct_int pace_int
    pct_int=${pct%.*}
    pace_int=${pace%.*}

    if [[ -z "$pct_int" || -z "$pace_int" ]]; then
        echo ""
        return 1
    fi

    echo "$pct_int $pace_int"
}

# Get usage based on configured backend
# Returns: "pct pace" (integers) or empty string on error
get_usage() {
    local backend
    backend=$(get_llm_backend)

    case "$backend" in
        claude-code)
            get_claude_usage
            ;;
        codex|*)
            get_codex_usage
            ;;
    esac
}

# Check if we should wait based on usage threshold
# Returns: 0 if OK to proceed, 1 if should wait
check_usage_threshold() {
    local usage
    usage=$(get_usage)

    if [[ -z "$usage" ]]; then
        log_message "WARN" "Could not get usage, continuing without throttling"
        return 0  # Continue without throttling
    fi

    local pct pace
    read -r pct pace <<< "$usage"

    # Always wait if at rate limit (100%)
    if [[ $pct -ge 100 ]]; then
        log_message "INFO" "Rate limited: usage at $pct%"
        return 1  # Need to wait
    fi

    local threshold=$((pace + PACE_BUFFER))

    if [[ $pct -gt $threshold ]]; then
        log_message "INFO" "Usage ($pct%) exceeds threshold (pace $pace% + buffer $PACE_BUFFER% = $threshold%)"
        return 1  # Need to wait
    fi

    log_message "INFO" "Usage OK: $pct% (threshold: $threshold%, pace: $pace%)"
    return 0
}

# Wait until usage is within acceptable threshold
wait_for_pace() {
    while ! check_usage_threshold; do
        local wait_min=$((WAIT_DURATION / 60))
        log_message "INFO" "Waiting $wait_min minutes for usage to decrease..."
        sleep_wall_clock "$WAIT_DURATION"
    done
}

# Discover months that have unprocessed files (or zero-card files in backfill mode)
# Output: one line per month with work: "YYYY-MM count" (newest first)
# Returns empty output if nothing to process
discover_months_with_work() {
    cd "$PROJECT_DIR" || return 1

    local backfill_flag="$1"

    uv run python -c "
from pathlib import Path
from auto_anki.state import StateTracker
from collections import defaultdict
import json
import re

# Read chat_root from config
config_path = Path('auto_anki_config.json')
if config_path.exists():
    config = json.loads(config_path.read_text())
    chat_root = Path(config.get('chat_root', '')).expanduser()
else:
    print('Error: auto_anki_config.json not found', file=__import__('sys').stderr)
    exit(1)

if not chat_root.exists():
    print(f'Error: chat_root does not exist: {chat_root}', file=__import__('sys').stderr)
    exit(1)

state_path = Path('.auto_anki_agent_state.json')
state = StateTracker(state_path) if state_path.exists() else None
backfill_mode = '$backfill_flag' == '1'

# Get exclusion patterns from config
import fnmatch
exclude_patterns = config.get('exclude_patterns', [])

# Group files by month
months_with_work = defaultdict(int)
date_pattern = re.compile(r'(\d{4}-\d{2})-\d{2}')

for f in chat_root.rglob('*.md'):
    # Skip excluded patterns
    if any(fnmatch.fnmatch(f.name, p) for p in exclude_patterns):
        continue

    match = date_pattern.search(f.name)
    if not match:
        continue
    month = match.group(1)  # YYYY-MM

    if backfill_mode:
        # In backfill mode, count files that generated 0 cards
        if state and state.is_file_zero_card(f):
            months_with_work[month] += 1
    else:
        # Normal mode: count unprocessed files
        if state is None or not state.is_file_processed(f):
            months_with_work[month] += 1

# Sort newest first and print
for month in sorted(months_with_work.keys(), reverse=True):
    print(f'{month} {months_with_work[month]}')
"
}

# Run auto-anki for a specific month with retry logic
# Returns: 0 on success, 1 on failure
# Outputs: stdout from auto-anki
run_auto_anki() {
    local month="$1"
    local retry_count=0
    local output_file output exit_code

    while [[ $retry_count -lt $MAX_RETRIES ]]; do
        log_message "INFO" "Running auto-anki for $month (attempt $((retry_count + 1))/$MAX_RETRIES)"

        cd "$PROJECT_DIR" || exit 1

        # Stream output live to both the terminal and the log file while
        # also capturing it for post-run inspection. We send the live stream
        # to FD 3 so it bypasses any surrounding command substitutions.
        output_file=$(mktemp)
        if [[ "$BACKFILL_MODE" == "1" ]]; then
            # Backfill mode: reprocess files that generated 0 cards
            uv run auto-anki \
                --only-zero-card-files \
                --date-range "$month" \
                --verbose 2>&1 | tee -a "$LOG_FILE" | tee "$output_file" >&3
        else
            # Normal mode: process unprocessed files
            uv run auto-anki \
                --date-range "$month" \
                --unprocessed-only \
                --verbose 2>&1 | tee -a "$LOG_FILE" | tee "$output_file" >&3
        fi
        exit_code=${PIPESTATUS[0]}
        output=$(cat "$output_file")
        rm -f "$output_file"

        if [[ $exit_code -eq 0 ]]; then
            echo "$output"
            return 0
        fi

        # Check for specific errors
        # Prefer a stable sentinel from the CLI; fall back to older wording.
        if echo "$output" | grep -q "ANKI_CONNECT_ERROR"; then
            log_message "ERROR" "Anki is not running or AnkiConnect is unreachable."
            log_message "INFO" "Press Enter to retry after starting Anki (with AnkiConnect enabled)..."
            read -r
            continue
        fi

        if echo "$output" | grep -Eq "Could not connect to Anki|Cannot connect to Anki"; then
            log_message "ERROR" "Anki is not running. Please start Anki with AnkiConnect."
            log_message "INFO" "Press Enter to retry after starting Anki..."
            read -r
            continue
        fi

        retry_count=$((retry_count + 1))
        if [[ $retry_count -lt $MAX_RETRIES ]]; then
            log_message "WARN" "Attempt failed (exit code: $exit_code), retrying in $RETRY_DELAY seconds..."
            sleep "$RETRY_DELAY"
        fi
    done

    log_message "ERROR" "Failed after $MAX_RETRIES attempts for $month"
    echo "$output"
    return 1
}

# Show progress
show_progress() {
    log_message "INFO" "Current progress:"
    cd "$PROJECT_DIR" || exit 1
    uv run auto-anki-progress 2>&1 | tee -a "$LOG_FILE"
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Ensure log directory exists
    mkdir -p "$LOG_DIR"

    log_message "INFO" "========================================"
    log_message "INFO" "Starting auto-anki batch processing"
    log_message "INFO" "========================================"
    log_message "INFO" "Log file: $LOG_FILE"
    log_message "INFO" "LLM Backend: $(get_llm_backend)"
    if [[ "$BACKFILL_MODE" == "1" ]]; then
        log_message "INFO" "Mode: BACKFILL (reprocessing zero-card files)"
    else
        log_message "INFO" "Mode: Normal (processing unprocessed files)"
    fi
    log_message "INFO" "Throttle: wait if usage > pace + ${PACE_BUFFER}%"
    log_message "INFO" "Wait duration: $((WAIT_DURATION / 60)) minutes"
    log_message "INFO" "Press Ctrl+C to stop gracefully"
    log_message "INFO" "========================================"

    start_sleep_inhibitor

    # Discover months that have unprocessed files
    log_message "INFO" "Discovering months with unprocessed files..."

    local discovery_output
    discovery_output=$(discover_months_with_work "$BACKFILL_MODE")

    if [[ -z "$discovery_output" ]]; then
        log_message "INFO" "No unprocessed files found. Nothing to do."
        show_progress
        stop_sleep_inhibitor || true
        exit 0
    fi

    # Parse discovery output into MONTHS array
    MONTHS=()
    local total_files=0
    while read -r line; do
        # Split line into month and count (format: "YYYY-MM count")
        local month count
        month="${line%% *}"
        count="${line##* }"
        MONTHS+=("$month")
        # Use 10# prefix to force decimal interpretation (avoid octal issues)
        total_files=$((total_files + 10#$count))
    done <<< "$discovery_output"

    local total_months=${#MONTHS[@]}

    log_message "INFO" "Found $total_files unprocessed files across $total_months months"
    log_message "INFO" "Months to process: ${MONTHS[*]}"

    # Main processing loop
    while true; do
        # Check if we've processed all months
        if [[ $month_index -ge $total_months ]]; then
            log_message "INFO" "All months processed! Exiting."
            log_message "INFO" "Run 'uv run auto-anki-progress' to see final status"
            show_progress
            stop_sleep_inhibitor || true
            exit 0
        fi

        current_month="${MONTHS[$month_index]}"

        # Check usage before running
        wait_for_pace

        # Run auto-anki
        local output
        output=$(run_auto_anki "$current_month")
        local result=$?

        # Show progress after each run
        echo ""
        show_progress
        echo ""

        # Check if month is exhausted (no more files to process)
        if [[ "$BACKFILL_MODE" == "1" ]]; then
            if echo "$output" | grep -q "No zero-card files found"; then
                log_message "INFO" "Month $current_month exhausted (backfill), moving to next month"
                month_index=$((month_index + 1))
                continue
            fi
        else
            if echo "$output" | grep -q "No new conversations found"; then
                log_message "INFO" "Month $current_month exhausted, moving to next month"
                month_index=$((month_index + 1))
                continue
            fi
        fi

        # If run failed after all retries, move to next month
        if [[ $result -ne 0 ]]; then
            log_message "WARN" "Run failed for $current_month, moving to next month"
            month_index=$((month_index + 1))
            continue
        fi

        # Successful run - stay on same month for next iteration
        # (there may be more unprocessed files in this month)
        log_message "INFO" "Completed run for $current_month"

        # Small delay between runs
        sleep_wall_clock 5
    done
}

main "$@"

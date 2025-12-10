#!/bin/bash
#
# auto_anki_batch.sh - Automated batch processing with usage-based throttling
#
# Runs auto-anki in a loop, working backwards through months from most recent.
# Monitors codex usage and waits when usage exceeds sustainable pace.
#
# Usage:
#   ./scripts/auto_anki_batch.sh
#   Press Ctrl+C to stop gracefully
#
# Requirements:
#   - Codex CLI logged in (~/.codex/auth.json must exist)
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

# Retry settings
MAX_RETRIES=3
RETRY_DELAY=60          # seconds between retries

# Month range (will process from START backwards to END)
# Defaults to current month; adjust END values based on your data
START_YEAR=$(date +%Y)
START_MONTH=$(date +%-m)  # Current month without leading zero
END_YEAR=2022
END_MONTH=1

# Logging
LOG_DIR="$PROJECT_DIR/auto_anki_runs"
LOG_FILE="$LOG_DIR/batch_$(date +%Y%m%d_%H%M%S).log"

# =============================================================================
# State
# =============================================================================

current_month=""
month_index=0
declare -a MONTHS

# =============================================================================
# Signal Handling
# =============================================================================

cleanup() {
    echo ""
    log_message "INFO" "Received interrupt signal. Shutting down gracefully..."
    log_message "INFO" "Last processed month: $current_month"
    log_message "INFO" "Run 'uv run auto-anki-progress' to see current status"
    log_message "INFO" "Log file saved to: $LOG_FILE"
    exit 0
}

trap cleanup SIGINT SIGTERM

# =============================================================================
# Utility Functions
# =============================================================================

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
        sleep "$WAIT_DURATION"
    done
}

# Generate list of months from current backwards
generate_months() {
    MONTHS=()
    local year month

    for ((year=START_YEAR; year>=END_YEAR; year--)); do
        local max_month=12
        local min_month=1

        # Limit to current month for start year
        if [[ $year -eq $START_YEAR ]]; then
            max_month=$START_MONTH
        fi

        # Limit to end month for end year
        if [[ $year -eq $END_YEAR ]]; then
            min_month=$END_MONTH
        fi

        for ((month=max_month; month>=min_month; month--)); do
            MONTHS+=("$(printf '%d-%02d' "$year" "$month")")
        done
    done
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
        uv run auto-anki \
            --date-range "$month" \
            --unprocessed-only \
            --verbose 2>&1 | tee -a "$LOG_FILE" | tee "$output_file" >&3
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
    log_message "INFO" "Throttle: wait if usage > pace + ${PACE_BUFFER}%"
    log_message "INFO" "Wait duration: $((WAIT_DURATION / 60)) minutes"
    log_message "INFO" "Press Ctrl+C to stop gracefully"
    log_message "INFO" "========================================"

    # Generate list of months
    generate_months
    local total_months=${#MONTHS[@]}

    local last_index=$((total_months - 1))
    log_message "INFO" "Processing $total_months months from ${MONTHS[0]} to ${MONTHS[$last_index]}"

    # Main processing loop
    while true; do
        # Check if we've processed all months
        if [[ $month_index -ge $total_months ]]; then
            log_message "INFO" "All months processed! Restarting from ${MONTHS[0]}..."
            month_index=0
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

        # Check if month is exhausted (no more unprocessed files)
        if echo "$output" | grep -q "No new conversations found"; then
            log_message "INFO" "Month $current_month exhausted, moving to next month"
            month_index=$((month_index + 1))
            continue
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
        sleep 5
    done
}

main "$@"

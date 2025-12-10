#!/bin/bash
# Estimate time to complete card generation based on recent log progress
# Usage: ./scripts/estimate_completion.sh [log_file]
#
# If no log file specified, uses the most recent batch_*.log file

set -euo pipefail

RUNS_DIR="${RUNS_DIR:-$(dirname "$0")/../auto_anki_runs}"

# Find the most recent log file if not specified
if [[ $# -ge 1 ]]; then
    LOG_FILE="$1"
else
    LOG_FILE=$(ls -t "$RUNS_DIR"/batch_*.log 2>/dev/null | head -1)
    if [[ -z "$LOG_FILE" ]]; then
        echo "Error: No batch log files found in $RUNS_DIR" >&2
        exit 1
    fi
fi

if [[ ! -f "$LOG_FILE" ]]; then
    echo "Error: Log file not found: $LOG_FILE" >&2
    exit 1
fi

echo "Analyzing: $(basename "$LOG_FILE")"
echo "=========================================="

# Create temp files to store extracted data
tmp_timestamps=$(mktemp)
tmp_progress=$(mktemp)
trap 'rm -f "$tmp_timestamps" "$tmp_progress"' EXIT

# Extract timestamps (preceding "Current progress" lines)
grep -B1 "Current progress" "$LOG_FILE" | grep -oE "\[2025-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\]" | tr -d '[]' > "$tmp_timestamps"

# Extract progress lines (exclude harvest/dedup status lines)
grep -E "[0-9]+ / [0-9]+ conversations" "$LOG_FILE" | grep -v "Harvested\|Checked\|remain after" | tail -100 > "$tmp_progress"

num_timestamps=$(wc -l < "$tmp_timestamps" | tr -d ' ')
num_progress=$(wc -l < "$tmp_progress" | tr -d ' ')

if [[ $num_progress -lt 2 ]]; then
    echo "Error: Need at least 2 progress snapshots to calculate rate" >&2
    exit 1
fi

# Use minimum of two counts
count=$((num_timestamps < num_progress ? num_timestamps : num_progress))

# Get first and last entries
first_timestamp=$(head -1 "$tmp_timestamps")
last_timestamp=$(tail -1 "$tmp_timestamps")
first_progress=$(head -1 "$tmp_progress")
last_progress=$(tail -1 "$tmp_progress")

# Extract numbers from progress lines (format: │  X / Y conversations  |  Z cards generated  │)
first_done=$(echo "$first_progress" | grep -oE "[0-9]+ / [0-9]+ conversations" | sed -E 's/([0-9]+) \/ [0-9]+ conversations/\1/')
first_total=$(echo "$first_progress" | grep -oE "[0-9]+ / [0-9]+ conversations" | sed -E 's/[0-9]+ \/ ([0-9]+) conversations/\1/')
last_done=$(echo "$last_progress" | grep -oE "[0-9]+ / [0-9]+ conversations" | sed -E 's/([0-9]+) \/ [0-9]+ conversations/\1/')
last_total=$(echo "$last_progress" | grep -oE "[0-9]+ / [0-9]+ conversations" | sed -E 's/[0-9]+ \/ ([0-9]+) conversations/\1/')

# Convert timestamps to epoch seconds for calculation
first_epoch=$(date -j -f "%Y-%m-%d %H:%M:%S" "$first_timestamp" "+%s" 2>/dev/null || date -d "$first_timestamp" "+%s")
last_epoch=$(date -j -f "%Y-%m-%d %H:%M:%S" "$last_timestamp" "+%s" 2>/dev/null || date -d "$last_timestamp" "+%s")

# Calculate elapsed time and conversations processed
elapsed_seconds=$((last_epoch - first_epoch))
conversations_processed=$((last_done - first_done))
conversations_remaining=$((last_total - last_done))

echo ""
echo "Progress Summary"
echo "----------------"
echo "Start time:              $first_timestamp"
echo "Latest time:             $last_timestamp"
echo "Conversations at start:  $first_done / $first_total"
echo "Conversations now:       $last_done / $last_total"
echo "Progress:                $(printf "%.1f" "$(echo "scale=1; $last_done * 100 / $last_total" | bc)")%"
echo ""

if [[ $elapsed_seconds -le 0 || $conversations_processed -le 0 ]]; then
    echo "Not enough progress to calculate rate."
    echo "(Need elapsed time > 0 and at least 1 conversation processed)"
    exit 0
fi

# Calculate rate
elapsed_hours=$(echo "scale=4; $elapsed_seconds / 3600" | bc)
rate_per_hour=$(echo "scale=1; $conversations_processed / $elapsed_hours" | bc)
rate_per_minute=$(echo "scale=2; $conversations_processed / ($elapsed_seconds / 60)" | bc)

echo "Processing Rate"
echo "---------------"
printf "Elapsed time:            %.1f hours (%d seconds)\n" "$elapsed_hours" "$elapsed_seconds"
echo "Conversations processed: $conversations_processed"
printf "Rate:                    %.1f conversations/hour\n" "$rate_per_hour"
printf "                         %.2f conversations/minute\n" "$rate_per_minute"
echo ""

# Calculate time remaining
if [[ $(echo "$rate_per_hour > 0" | bc) -eq 1 ]]; then
    hours_remaining=$(echo "scale=2; $conversations_remaining / $rate_per_hour" | bc)
    days_remaining=$(echo "scale=2; $hours_remaining / 24" | bc)

    # Calculate estimated completion time
    seconds_remaining=$(echo "scale=0; $hours_remaining * 3600" | bc | cut -d. -f1)
    if [[ $(uname) == "Darwin" ]]; then
        completion_time=$(date -v+"${seconds_remaining}S" "+%Y-%m-%d %H:%M")
    else
        completion_time=$(date -d "+${seconds_remaining} seconds" "+%Y-%m-%d %H:%M")
    fi

    echo "Time Estimate"
    echo "-------------"
    echo "Conversations remaining: $conversations_remaining"
    printf "Estimated time:          %.1f hours (%.1f days)\n" "$hours_remaining" "$days_remaining"
    echo "Estimated completion:    $completion_time"
    echo ""

    # Show recent rate (last 10 entries) for comparison
    if [[ $count -gt 10 ]]; then
        recent_first_timestamp=$(tail -10 "$tmp_timestamps" | head -1)
        recent_first_progress=$(tail -10 "$tmp_progress" | head -1)

        recent_first_done=$(echo "$recent_first_progress" | grep -oE "[0-9]+ / [0-9]+ conversations" | sed -E 's/([0-9]+) \/ [0-9]+ conversations/\1/')

        recent_first_epoch=$(date -j -f "%Y-%m-%d %H:%M:%S" "$recent_first_timestamp" "+%s" 2>/dev/null || date -d "$recent_first_timestamp" "+%s")
        recent_elapsed=$((last_epoch - recent_first_epoch))
        recent_processed=$((last_done - recent_first_done))

        if [[ $recent_elapsed -gt 0 && $recent_processed -gt 0 ]]; then
            recent_rate=$(echo "scale=1; $recent_processed * 3600 / $recent_elapsed" | bc)
            recent_hours_remaining=$(echo "scale=2; $conversations_remaining / $recent_rate" | bc)
            recent_days_remaining=$(echo "scale=2; $recent_hours_remaining / 24" | bc)

            echo "Recent Rate (last ~10 runs)"
            echo "---------------------------"
            printf "Rate:                    %.1f conversations/hour\n" "$recent_rate"
            printf "Estimated time:          %.1f hours (%.1f days)\n" "$recent_hours_remaining" "$recent_days_remaining"
        fi
    fi
fi

#!/bin/bash
#
# claude-usage.sh - Display Claude Code API usage limits
#
# Shows current utilization and reset times for:
#   - Session (5-hour rolling window)
#   - Weekly (all models, 7-day rolling window)
#   - Weekly Sonnet only (7-day rolling window)
#
# Features:
#   - Color-coded bars: green (<50%), yellow (50-80%), red (>80%)
#   - Pace indicator (│): shows sustainable usage rate position
#   - Pace status: "on pace", "ahead" (red), or "behind" (green)
#   - Reset times displayed in local timezone, rounded to nearest 5 minutes
#     (API returns sub-second jitter that can cause hour display to fluctuate)
#
# Requirements:
#   - macOS (uses security command and BSD date)
#   - jq
#   - curl
#   - Claude Code installed with OAuth credentials in keychain
#
# Usage:
#   ./claude-usage.sh [-v|--verbose]
#
# Options:
#   -v, --verbose   Show raw API response and parsed timestamps
#
# Changelog:
#   2025-11-29  Initial version with color bars, pace tracking, UTC conversion
#   2025-11-29  Add verbose mode (-v), round reset times to nearest 5 minutes

# Parse arguments
VERBOSE=0
while [[ $# -gt 0 ]]; do
  case $1 in
    -v|--verbose) VERBOSE=1; shift ;;
    *) shift ;;
  esac
done

# Colors
R='\033[0m'       # Reset
DIM='\033[2m'     # Dim
BOLD='\033[1m'    # Bold
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
CYAN='\033[36m'
MAGENTA='\033[35m'

# Window durations in seconds
WINDOW_5H=$((5 * 60 * 60))
WINDOW_7D=$((7 * 24 * 60 * 60))

TOKEN="$(security find-generic-password -s 'Claude Code-credentials' -w \
  | jq -r '.claudeAiOauth.accessToken')"

USAGE=$(curl -s --compressed \
  -H "anthropic-beta: oauth-2025-04-20" \
  -H "authorization: Bearer $TOKEN" \
  "https://api.anthropic.com/api/oauth/usage")

NOW=$(date "+%s")

if [[ $VERBOSE -eq 1 ]]; then
  printf "${DIM}────────────── VERBOSE OUTPUT ──────────────${R}\n"
  printf "${BOLD}Raw API response:${R}\n"
  echo "$USAGE" | jq .
  printf "\n${BOLD}Current time:${R} %s (epoch: %s)\n" "$(date)" "$NOW"
  printf "${DIM}─────────────────────────────────────────────${R}\n\n"
fi

color_for_pct() {
  local pct=$1
  if (( $(echo "$pct < 50" | bc -l) )); then echo -e "$GREEN"
  elif (( $(echo "$pct < 80" | bc -l) )); then echo -e "$YELLOW"
  else echo -e "$RED"
  fi
}

# Parse ISO timestamp (UTC) to epoch seconds
parse_utc() {
  local iso=$1
  if [[ "$iso" == "null" ]]; then echo "0"; return; fi
  TZ=UTC date -j -f "%Y-%m-%dT%H:%M:%S" "${iso%%.*}" "+%s" 2>/dev/null
}

# Round epoch to nearest 5 minutes (300 seconds)
round_to_5min() {
  local ts=$1
  echo $(( ((ts + 150) / 300) * 300 ))
}

# Calculate what % of window has elapsed (= expected usage at constant rate)
calc_elapsed_pct() {
  local reset_iso=$1 window_secs=$2
  if [[ "$reset_iso" == "null" ]]; then echo "0"; return; fi
  local reset_ts=$(parse_utc "$reset_iso")
  local remaining=$((reset_ts - NOW))
  if ((remaining < 0)); then remaining=0; fi
  local elapsed=$((window_secs - remaining))
  echo "scale=1; $elapsed * 100 / $window_secs" | bc
}

bar() {
  local pct=$1 pace=$2 width=40
  local filled=$(echo "scale=0; $pct * $width / 100" | bc)
  filled=${filled:-0}
  local pace_pos=$(echo "scale=0; $pace * $width / 100" | bc)
  pace_pos=${pace_pos:-0}
  local color=$(color_for_pct "$pct")

  local bar=""
  for ((i=0; i<width; i++)); do
    if ((i == pace_pos)); then
      bar+="${MAGENTA}│${R}"
    elif ((i < filled)); then
      bar+="${color}━${R}"
    else
      bar+="${DIM}─${R}"
    fi
  done
  echo -e "$bar"
}

format_time() {
  local iso=$1
  if [[ "$iso" == "null" ]]; then echo "N/A"; return; fi
  local ts=$(parse_utc "$iso")
  ts=$(round_to_5min "$ts")
  local today=$(date "+%Y-%m-%d")
  local target_date=$(date -j -f "%s" "$ts" "+%Y-%m-%d")
  local tz=$(date "+%Z")
  if [[ "$today" == "$target_date" ]]; then
    echo "$(date -j -f "%s" "$ts" "+%-I:%M%p" | sed 's/AM/am/;s/PM/pm/') ($tz)"
  else
    echo "$(date -j -f "%s" "$ts" "+%b %-d at %-I:%M%p" | sed 's/AM/am/;s/PM/pm/') ($tz)"
  fi
}

pace_indicator() {
  local pct=$1 pace=$2
  local diff=$(echo "$pct - $pace" | bc)
  local abs_diff=$(echo "$diff" | tr -d -)
  if (( $(echo "$diff > 5" | bc -l) )); then
    printf "${RED}+%.0f%% ahead${R}" "$diff"
  elif (( $(echo "$diff < -5" | bc -l) )); then
    printf "${GREEN}%.0f%% behind${R}" "$diff"
  else
    printf "${DIM}on pace${R}"
  fi
}

show_usage() {
  local label=$1 pct=$2 reset=$3 pace=$4
  local color=$(color_for_pct "$pct")
  printf "${BOLD}%s${R}\n" "$label"
  printf "  %s ${color}%3.0f%%${R}  " "$(bar "$pct" "$pace")" "$pct"
  pace_indicator "$pct" "$pace"
  printf "  ${DIM}resets %s${R}\n" "$(format_time "$reset")"
}

five_pct=$(echo "$USAGE" | jq -r '.five_hour.utilization // 0')
five_reset=$(echo "$USAGE" | jq -r '.five_hour.resets_at // "null"')
seven_pct=$(echo "$USAGE" | jq -r '.seven_day.utilization // 0')
seven_reset=$(echo "$USAGE" | jq -r '.seven_day.resets_at // "null"')
sonnet_pct=$(echo "$USAGE" | jq -r '.seven_day_sonnet.utilization // 0')
sonnet_reset=$(echo "$USAGE" | jq -r '.seven_day_sonnet.resets_at // "null"')

five_pace=$(calc_elapsed_pct "$five_reset" "$WINDOW_5H")
seven_pace=$(calc_elapsed_pct "$seven_reset" "$WINDOW_7D")
sonnet_pace=$(calc_elapsed_pct "$sonnet_reset" "$WINDOW_7D")

if [[ $VERBOSE -eq 1 ]]; then
  printf "${DIM}────────────── PARSED VALUES ───────────────${R}\n"
  printf "${BOLD}5-hour:${R}  reset=%s  epoch=%s  pct=%.1f%%  pace=%.1f%%\n" \
    "$five_reset" "$(parse_utc "$five_reset")" "$five_pct" "$five_pace"
  printf "${BOLD}7-day:${R}   reset=%s  epoch=%s  pct=%.1f%%  pace=%.1f%%\n" \
    "$seven_reset" "$(parse_utc "$seven_reset")" "$seven_pct" "$seven_pace"
  printf "${BOLD}Sonnet:${R}  reset=%s  epoch=%s  pct=%.1f%%  pace=%.1f%%\n" \
    "$sonnet_reset" "$(parse_utc "$sonnet_reset")" "$sonnet_pct" "$sonnet_pace"
  printf "${DIM}─────────────────────────────────────────────${R}\n"
fi

echo ""
printf "${DIM}─────────────────────────────────────────────────────────${R}\n"
printf "  ${CYAN}${BOLD}Claude Code Usage${R}  ${DIM}│${R} marks sustainable pace\n"
printf "${DIM}─────────────────────────────────────────────────────────${R}\n"
echo ""
show_usage "Session (5h)" "$five_pct" "$five_reset" "$five_pace"
echo ""
show_usage "Week (all models)" "$seven_pct" "$seven_reset" "$seven_pace"
echo ""
show_usage "Week (Sonnet)" "$sonnet_pct" "$sonnet_reset" "$sonnet_pace"
echo ""

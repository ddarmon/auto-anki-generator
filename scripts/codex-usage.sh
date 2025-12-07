#!/bin/bash
#
# codex-usage.sh - Display ChatGPT/Codex usage limits
#
# Uses Codex CLI's auth tokens from ~/.codex/auth.json to query
#   https://chatgpt.com/backend-api/wham/usage
# and renders color bars and pace indicators similar to claude-usage.sh.
#
# Requirements:
#   - macOS (BSD date)
#   - jq
#   - bc
#   - Codex CLI already logged in (~/.codex/auth.json exists)
#
# Usage:
#   ./codex-usage.sh [-v|--verbose]
#

VERBOSE=0
while [[ $# -gt 0 ]]; do
  case $1 in
    -v|--verbose) VERBOSE=1; shift ;;
    *) shift ;;
  esac
done

# Colors
R='\033[0m'
DIM='\033[2m'
BOLD='\033[1m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
CYAN='\033[36m'
MAGENTA='\033[35m'

AUTH_FILE="$HOME/.codex/auth.json"

if [[ ! -f "$AUTH_FILE" ]]; then
  echo "codex-usage: auth file not found at $AUTH_FILE" >&2
  echo "Run Codex CLI once (to log in) and try again." >&2
  exit 1
fi

ACCESS_TOKEN=$(jq -r '.tokens.access_token // empty' "$AUTH_FILE")
ACCOUNT_ID=$(jq -r '.tokens.account_id // empty' "$AUTH_FILE")

if [[ -z "$ACCESS_TOKEN" || -z "$ACCOUNT_ID" || "$ACCESS_TOKEN" == "null" || "$ACCOUNT_ID" == "null" ]]; then
  echo "codex-usage: missing access token or account id in $AUTH_FILE" >&2
  exit 1
fi

USAGE=$(curl -sS --compressed \
  -H "authorization: Bearer $ACCESS_TOKEN" \
  -H "chatgpt-account-id: $ACCOUNT_ID" \
  -H "accept: */*" \
  "https://chatgpt.com/backend-api/wham/usage")

if [[ $? -ne 0 || -z "$USAGE" ]]; then
  echo "codex-usage: failed to fetch usage from ChatGPT" >&2
  exit 1
fi

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
  if (( $(echo "$pct < 50" | bc -l) )); then
    echo -e "$GREEN"
  elif (( $(echo "$pct < 80" | bc -l) )); then
    echo -e "$YELLOW"
  else
    echo -e "$RED"
  fi
}

# Round epoch to nearest 5 minutes (300 seconds)
round_to_5min() {
  local ts=$1
  echo $(( ((ts + 150) / 300) * 300 ))
}

format_time_epoch() {
  local ts=$1
  if [[ -z "$ts" || "$ts" == "null" || "$ts" -eq 0 ]]; then
    echo "N/A"
    return
  fi
  ts=$(round_to_5min "$ts")
  local today
  local target_date
  local tz
  today=$(date "+%Y-%m-%d")
  target_date=$(date -r "$ts" "+%Y-%m-%d")
  tz=$(date "+%Z")
  if [[ "$today" == "$target_date" ]]; then
    echo "$(date -r "$ts" "+%-I:%M%p" | sed 's/AM/am/;s/PM/pm/') ($tz)"
  else
    echo "$(date -r "$ts" "+%b %-d at %-I:%M%p" | sed 's/AM/am/;s/PM/pm/') ($tz)"
  fi
}

# Calculate what % of window has elapsed (= expected usage at constant rate)
calc_elapsed_pct_window() {
  local window_secs=$1
  local reset_after=$2

  if [[ -z "$window_secs" || "$window_secs" == "null" || "$window_secs" -le 0 ]]; then
    echo "0"
    return
  fi
  if [[ -z "$reset_after" || "$reset_after" == "null" ]]; then
    echo "0"
    return
  fi

  local remaining=$reset_after
  if (( remaining < 0 )); then
    remaining=0
  fi
  local elapsed=$((window_secs - remaining))
  if (( elapsed < 0 )); then
    elapsed=0
  fi

  echo "scale=1; $elapsed * 100 / $window_secs" | bc
}

bar() {
  local pct=$1
  local pace=$2
  local width=40

  local filled
  local pace_pos
  filled=$(echo "scale=0; $pct * $width / 100" | bc 2>/dev/null)
  pace_pos=$(echo "scale=0; $pace * $width / 100" | bc 2>/dev/null)
  filled=${filled:-0}
  pace_pos=${pace_pos:-0}

  local color
  color=$(color_for_pct "$pct")
  local bar=""

  for ((i=0; i<width; i++)); do
    if (( i == pace_pos )); then
      bar+="${MAGENTA}│${R}"
    elif (( i < filled )); then
      bar+="${color}━${R}"
    else
      bar+="${DIM}─${R}"
    fi
  done
  echo -e "$bar"
}

pace_indicator() {
  local pct=$1
  local pace=$2
  local diff
  local abs_diff

  diff=$(echo "$pct - $pace" | bc)
  abs_diff=$(echo "$diff" | tr -d -)

  if (( $(echo "$diff > 5" | bc -l) )); then
    printf "${RED}+%.0f%% ahead${R}" "$diff"
  elif (( $(echo "$diff < -5" | bc -l) )); then
    printf "${GREEN}%.0f%% behind${R}" "$diff"
  else
    printf "${DIM}on pace${R}"
  fi
}

show_window() {
  local label=$1
  local pct=$2
  local reset_ts=$3
  local pace=$4

  local color
  color=$(color_for_pct "$pct")

  printf "${BOLD}%s${R}\n" "$label"
  printf "  %s ${color}%3.0f%%${R}  " "$(bar "$pct" "$pace")" "$pct"
  pace_indicator "$pct" "$pace"
  printf "  ${DIM}resets %s${R}\n" "$(format_time_epoch "$reset_ts")"
}

# Extract fields from usage JSON
plan_type=$(echo "$USAGE" | jq -r '.plan_type // "unknown"')

rl_allowed=$(echo "$USAGE" | jq -r '.rate_limit.allowed // false')
rl_primary_used=$(echo "$USAGE" | jq -r '.rate_limit.primary_window.used_percent // 0')
rl_primary_window=$(echo "$USAGE" | jq -r '.rate_limit.primary_window.limit_window_seconds // 0')
rl_primary_reset_after=$(echo "$USAGE" | jq -r '.rate_limit.primary_window.reset_after_seconds // 0')
rl_primary_reset_at=$(echo "$USAGE" | jq -r '.rate_limit.primary_window.reset_at // 0')

rl_secondary_used=$(echo "$USAGE" | jq -r '.rate_limit.secondary_window.used_percent // 0')
rl_secondary_window=$(echo "$USAGE" | jq -r '.rate_limit.secondary_window.limit_window_seconds // 0')
rl_secondary_reset_after=$(echo "$USAGE" | jq -r '.rate_limit.secondary_window.reset_after_seconds // 0')
rl_secondary_reset_at=$(echo "$USAGE" | jq -r '.rate_limit.secondary_window.reset_at // 0')

cr_allowed=$(echo "$USAGE" | jq -r '.code_review_rate_limit.allowed // false')
cr_primary_used=$(echo "$USAGE" | jq -r '.code_review_rate_limit.primary_window.used_percent // 0')
cr_primary_window=$(echo "$USAGE" | jq -r '.code_review_rate_limit.primary_window.limit_window_seconds // 0')
cr_primary_reset_after=$(echo "$USAGE" | jq -r '.code_review_rate_limit.primary_window.reset_after_seconds // 0')
cr_primary_reset_at=$(echo "$USAGE" | jq -r '.code_review_rate_limit.primary_window.reset_at // 0')

five_pace=$(calc_elapsed_pct_window "$rl_primary_window" "$rl_primary_reset_after")
seven_pace=$(calc_elapsed_pct_window "$rl_secondary_window" "$rl_secondary_reset_after")
cr_pace=$(calc_elapsed_pct_window "$cr_primary_window" "$cr_primary_reset_after")

if [[ $VERBOSE -eq 1 ]]; then
  printf "${DIM}────────────── PARSED VALUES ───────────────${R}\n"
  printf "${BOLD}Plan:${R} %s\n" "$plan_type"
  printf "${BOLD}Chat 5h:${R}   pct=%s  window=%ss  reset_after=%ss  reset_at=%s  pace=%s%%\n" \
    "$rl_primary_used" "$rl_primary_window" "$rl_primary_reset_after" "$rl_primary_reset_at" "$five_pace"
  printf "${BOLD}Chat 7d:${R}   pct=%s  window=%ss  reset_after=%ss  reset_at=%s  pace=%s%%\n" \
    "$rl_secondary_used" "$rl_secondary_window" "$rl_secondary_reset_after" "$rl_secondary_reset_at" "$seven_pace"
  printf "${BOLD}Code 7d:${R}   pct=%s  window=%ss  reset_after=%ss  reset_at=%s  pace=%s%%\n" \
    "$cr_primary_used" "$cr_primary_window" "$cr_primary_reset_after" "$cr_primary_reset_at" "$cr_pace"
  printf "${DIM}─────────────────────────────────────────────${R}\n"
fi

echo ""
printf "${DIM}─────────────────────────────────────────────────────────${R}\n"
printf "  ${CYAN}${BOLD}ChatGPT / Codex Usage${R}  ${DIM}│${R} marks sustainable pace\n"
printf "  ${DIM}Plan: %s${R}\n" "$plan_type"
printf "${DIM}─────────────────────────────────────────────────────────${R}\n"
echo ""

if [[ "$rl_allowed" != "true" ]]; then
  printf "${RED}Chat usage not allowed or no limits returned.${R}\n"
else
  show_window "Chat (5h window)" "$rl_primary_used" "$rl_primary_reset_at" "$five_pace"
  echo ""
  show_window "Chat (7d window)" "$rl_secondary_used" "$rl_secondary_reset_at" "$seven_pace"
fi

echo ""

if [[ "$cr_allowed" == "true" ]]; then
  show_window "Code review (7d window)" "$cr_primary_used" "$cr_primary_reset_at" "$cr_pace"
  echo ""
fi


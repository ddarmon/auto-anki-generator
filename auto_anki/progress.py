"""
TUI Progress Dashboard for Auto-Anki.

Displays conversation processing progress over time with weekly aggregation,
streak tracking, and cumulative statistics to motivate continued use.

Usage:
    auto-anki-progress           # Show last 12 weeks
    auto-anki-progress --weeks 24  # Show more history
    auto-anki-progress --json    # Machine-readable output
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional rich import - gracefully handle if not installed
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from auto_anki.state import StateTracker


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class WeeklyStats:
    """Statistics for a single ISO week."""

    week_start: date  # Monday of the week
    week_label: str  # Human-readable label like "Dec 2 - Dec 8"
    total_conversations: int  # All conversations available that week
    processed: int  # Conversations processed that week
    remaining: int  # Unprocessed conversations
    cards_generated: int  # Cards generated from runs that week


@dataclass
class ProgressSummary:
    """Overall progress summary across all time."""

    total_conversations: int
    total_processed: int
    total_cards: int
    completion_percentage: float
    current_streak_weeks: int
    longest_streak_weeks: int
    active_weeks: int
    first_activity_date: Optional[date]
    last_activity_date: Optional[date]


# ---------------------------------------------------------------------------
# Date Utilities
# ---------------------------------------------------------------------------

DATE_PATTERN = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def extract_date_from_path(path: Path) -> Optional[date]:
    """
    Extract date from a file path.

    Looks for YYYY-MM-DD pattern in the filename or parent directories.
    Examples:
        - 2025-10-23_topic.md -> date(2025, 10, 23)
        - chatgpt/2025/10/23/topic.md -> date(2025, 10, 23)
    """
    # First try the filename
    match = DATE_PATTERN.search(path.name)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    # Fall back to checking parent directories
    for part in path.parts:
        match = DATE_PATTERN.search(part)
        if match:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    return None


def get_week_start(d: date) -> date:
    """Get the Monday of the ISO week containing date d."""
    return d - timedelta(days=d.weekday())


def format_week_label(week_start: date) -> str:
    """Format a week as 'Mon D - Mon D' or 'Mon D - Mon D, YYYY' if spanning years."""
    week_end = week_start + timedelta(days=6)

    if week_start.year != week_end.year:
        return f"{week_start.strftime('%b %d, %Y')} - {week_end.strftime('%b %d, %Y')}"
    elif week_start.month != week_end.month:
        return f"{week_start.strftime('%b %d')} - {week_end.strftime('%b %d')}"
    else:
        return f"{week_start.strftime('%b %d')} - {week_end.day}"


# ---------------------------------------------------------------------------
# Data Collection
# ---------------------------------------------------------------------------


def scan_conversations(chat_root: Path) -> Dict[Path, date]:
    """
    Scan chat_root for all conversation files, extracting dates.

    Returns:
        Dict mapping file path -> conversation date
    """
    conversations: Dict[Path, date] = {}

    if not chat_root.exists():
        return conversations

    for md_file in chat_root.rglob("*.md"):
        # Skip hidden files and directories
        if any(part.startswith(".") for part in md_file.parts):
            continue

        file_date = extract_date_from_path(md_file)
        if file_date:
            conversations[md_file] = file_date

    return conversations


def get_processed_files_with_dates(
    state: StateTracker,
) -> Dict[str, Tuple[date, int]]:
    """
    Get processed files with their processing dates and card counts.

    Returns:
        Dict mapping file path -> (processed_date, cards_generated)
    """
    processed: Dict[str, Tuple[date, int]] = {}

    processed_files = state.data.get("processed_files", {})
    for file_path, info in processed_files.items():
        processed_at = info.get("processed_at")
        cards = info.get("cards_generated", 0)

        if processed_at:
            try:
                # Parse ISO timestamp
                dt = datetime.fromisoformat(processed_at.replace("Z", "+00:00"))
                processed[file_path] = (dt.date(), cards)
            except (ValueError, AttributeError):
                pass

    return processed


def get_run_history_by_week(state: StateTracker) -> Dict[date, int]:
    """
    Aggregate card counts from run history by week.

    Returns:
        Dict mapping week_start -> total cards generated that week
    """
    cards_by_week: Dict[date, int] = {}

    run_history = state.data.get("run_history", [])
    for run in run_history:
        timestamp = run.get("timestamp")
        contexts_sent = run.get("conversations_sent") or run.get("contexts_sent", 0)

        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                week = get_week_start(dt.date())
                # Use contexts_sent as proxy for cards if we don't have exact count
                cards_by_week[week] = cards_by_week.get(week, 0) + contexts_sent
            except (ValueError, AttributeError):
                pass

    return cards_by_week


# ---------------------------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------------------------


def compute_weekly_stats(
    all_conversations: Dict[Path, date],
    processed_files: Dict[str, Tuple[date, int]],
    cards_by_week: Dict[date, int],
    weeks_back: int = 12,
) -> List[WeeklyStats]:
    """
    Compute weekly statistics for the past N weeks.

    Args:
        all_conversations: All conversation files with their dates
        processed_files: Processed files with processing dates and card counts
        cards_by_week: Card counts by week from run history
        weeks_back: Number of weeks to analyze

    Returns:
        List of WeeklyStats, most recent first
    """
    today = date.today()
    current_week = get_week_start(today)

    # Create a set of processed file paths for quick lookup
    processed_paths = set(processed_files.keys())

    # Initialize weeks
    weeks: List[WeeklyStats] = []
    for i in range(weeks_back):
        week_start = current_week - timedelta(weeks=i)
        week_end = week_start + timedelta(days=6)

        # Count conversations available in this week (by file date)
        total = 0
        processed = 0
        for path, file_date in all_conversations.items():
            if week_start <= file_date <= week_end:
                total += 1
                if str(path) in processed_paths:
                    processed += 1

        # Get cards generated this week
        cards = cards_by_week.get(week_start, 0)

        # Also sum up cards from processed_files that were processed this week
        for file_path, (proc_date, card_count) in processed_files.items():
            if week_start <= proc_date <= week_end:
                cards += card_count

        weeks.append(
            WeeklyStats(
                week_start=week_start,
                week_label=format_week_label(week_start),
                total_conversations=total,
                processed=processed,
                remaining=total - processed,
                cards_generated=cards,
            )
        )

    return weeks


def compute_streaks(weekly_stats: List[WeeklyStats]) -> Tuple[int, int]:
    """
    Compute current and longest activity streaks from weekly stats.

    A week is "active" if it has any processed conversations.

    Args:
        weekly_stats: List of WeeklyStats (most recent first)

    Returns:
        (current_streak, longest_streak) in weeks
    """
    if not weekly_stats:
        return 0, 0

    # Current streak: count consecutive active weeks from most recent
    current_streak = 0
    for week in weekly_stats:
        if week.processed > 0:
            current_streak += 1
        else:
            break

    # Longest streak: find the longest consecutive run
    longest_streak = 0
    current_run = 0
    for week in reversed(weekly_stats):  # Process chronologically
        if week.processed > 0:
            current_run += 1
            longest_streak = max(longest_streak, current_run)
        else:
            current_run = 0

    return current_streak, longest_streak


def compute_summary(
    all_conversations: Dict[Path, date],
    processed_files: Dict[str, Tuple[date, int]],
    weekly_stats: List[WeeklyStats],
) -> ProgressSummary:
    """
    Compute overall progress summary.

    Args:
        all_conversations: All conversation files
        processed_files: Processed files with dates and card counts
        weekly_stats: Pre-computed weekly stats

    Returns:
        ProgressSummary with overall statistics
    """
    total = len(all_conversations)
    processed_paths = set(processed_files.keys())
    processed = sum(1 for p in all_conversations if str(p) in processed_paths)

    # Total cards from processed files
    total_cards = sum(cards for _, cards in processed_files.values())

    # Completion percentage
    completion = (processed / total * 100) if total > 0 else 0.0

    # Streaks
    current_streak, longest_streak = compute_streaks(weekly_stats)

    # Active weeks count
    active_weeks = sum(1 for w in weekly_stats if w.processed > 0)

    # First and last activity dates
    processing_dates = [d for d, _ in processed_files.values()]
    first_activity = min(processing_dates) if processing_dates else None
    last_activity = max(processing_dates) if processing_dates else None

    return ProgressSummary(
        total_conversations=total,
        total_processed=processed,
        total_cards=total_cards,
        completion_percentage=completion,
        current_streak_weeks=current_streak,
        longest_streak_weeks=longest_streak,
        active_weeks=active_weeks,
        first_activity_date=first_activity,
        last_activity_date=last_activity,
    )


# ---------------------------------------------------------------------------
# Rich TUI Components
# ---------------------------------------------------------------------------


def build_progress_bar(percentage: float, width: int = 30) -> str:
    """Build an ASCII progress bar."""
    filled = int(width * percentage / 100)
    empty = width - filled
    return "[" + "=" * filled + " " * empty + "]"


def mini_progress_bar(percentage: float, width: int = 6) -> Text:
    """Create a small inline progress bar with color."""
    if not RICH_AVAILABLE:
        return f"[{'=' * int(width * percentage / 100)}{' ' * (width - int(width * percentage / 100))}]"

    filled = int(width * percentage / 100)
    empty = width - filled
    bar = "=" * filled + " " * empty

    if percentage >= 80:
        color = "green"
    elif percentage >= 50:
        color = "yellow"
    else:
        color = "red"

    return Text(f"[{bar}]", style=color)


def build_activity_indicator(weekly_stats: List[WeeklyStats], max_weeks: int = 12) -> str:
    """
    Build an activity indicator showing recent weeks.

    Active weeks shown as '#', inactive as '.'
    """
    # Take most recent weeks, reverse to show oldest first
    recent = weekly_stats[:max_weeks]
    indicators = []
    for week in reversed(recent):
        if week.processed > 0:
            indicators.append("#")
        else:
            indicators.append(".")

    return " ".join(indicators)


def render_dashboard(
    summary: ProgressSummary,
    weekly_stats: List[WeeklyStats],
    console: Console,
    show_weeks: int = 12,
) -> None:
    """Render the full progress dashboard to the console."""

    # Main container
    console.print()
    console.print(
        Panel(
            _build_dashboard_content(summary, weekly_stats, show_weeks),
            title="Auto-Anki Progress",
            border_style="blue",
            padding=(1, 2),
        )
    )
    console.print()


def _build_dashboard_content(
    summary: ProgressSummary, weekly_stats: List[WeeklyStats], show_weeks: int
) -> Text:
    """Build the main dashboard content."""
    from rich.console import Group
    from rich.text import Text

    lines = []

    # Overall progress bar
    pct = summary.completion_percentage
    bar = build_progress_bar(pct)
    if pct >= 80:
        bar_color = "green"
    elif pct >= 50:
        bar_color = "yellow"
    else:
        bar_color = "white"

    lines.append(Text.assemble(
        ("Overall: ", "bold"),
        (bar, bar_color),
        (f" {pct:.1f}%", bar_color),
    ))

    lines.append(Text.assemble(
        (f"{summary.total_processed}", "cyan"),
        " / ",
        (f"{summary.total_conversations}", "white"),
        " conversations",
        ("  |  ", "dim"),
        (f"{summary.total_cards}", "green"),
        " cards generated",
    ))

    lines.append(Text())  # Blank line

    # Activity streak indicator
    activity = build_activity_indicator(weekly_stats, max_weeks=12)
    lines.append(Text.assemble(
        ("Activity: ", "bold"),
        (activity, "cyan"),
    ))

    # Streak stats
    streak_text = f"Current streak: {summary.current_streak_weeks} week"
    if summary.current_streak_weeks != 1:
        streak_text += "s"

    longest_text = f"Longest: {summary.longest_streak_weeks} week"
    if summary.longest_streak_weeks != 1:
        longest_text += "s"

    streak_style = "green" if summary.current_streak_weeks > 0 else "dim"
    lines.append(Text.assemble(
        (streak_text, streak_style),
        ("  |  ", "dim"),
        (longest_text, "yellow"),
    ))

    lines.append(Text())  # Blank line

    # Weekly table
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Week", style="cyan", width=16)
    table.add_column("Done", justify="right", width=6)
    table.add_column("Total", justify="right", width=6)
    table.add_column("Progress", justify="center", width=10)
    table.add_column("Cards", justify="right", style="green", width=6)

    for week in weekly_stats[:show_weeks]:
        pct = (week.processed / week.total_conversations * 100) if week.total_conversations > 0 else 0
        progress = mini_progress_bar(pct)

        table.add_row(
            week.week_label,
            str(week.processed),
            str(week.total_conversations),
            progress,
            str(week.cards_generated) if week.cards_generated > 0 else "-",
        )

    return Group(*lines, Text(), table)


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------


def _load_config() -> Tuple[Dict[str, Any], Optional[Path]]:
    """
    Load configuration from auto_anki_config.json.

    Search order:
    1. AUTO_ANKI_CONFIG environment variable
    2. ./auto_anki_config.json in current directory
    3. ~/.auto_anki_config.json in home directory
    """
    candidates: List[Path] = []
    env_path = os.getenv("AUTO_ANKI_CONFIG")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path.cwd() / "auto_anki_config.json")
    candidates.append(Path.home() / ".auto_anki_config.json")

    for path in candidates:
        if path.is_file():
            try:
                data = json.loads(path.read_text())
                return data, path
            except json.JSONDecodeError:
                continue

    return {}, None


def _resolve_path(path_str: Optional[str], config_root: Path) -> Optional[Path]:
    """Resolve a path relative to config root, or return None if not set."""
    if not path_str:
        return None

    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = config_root / path

    return path


def main() -> None:
    """Entry point for auto-anki-progress command."""
    parser = argparse.ArgumentParser(
        description="Display Auto-Anki progress dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  auto-anki-progress              Show progress for last 12 weeks
  auto-anki-progress --weeks 24   Show progress for last 24 weeks
  auto-anki-progress --json       Output as JSON (for scripting)
""",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=12,
        help="Number of weeks to display (default: 12)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of TUI",
    )

    args = parser.parse_args()

    # Load configuration
    config, config_path = _load_config()
    config_root = config_path.parent if config_path else Path.cwd()

    # Resolve paths
    state_file = config.get("state_file", ".auto_anki_agent_state.json")
    state_path = _resolve_path(state_file, config_root)
    if state_path is None:
        state_path = config_root / ".auto_anki_agent_state.json"

    chat_root_str = config.get("chat_root", "")
    chat_root = Path(chat_root_str).expanduser() if chat_root_str else None

    # Validate paths
    if not state_path.exists():
        if args.json:
            print(json.dumps({"error": "No state file found. Run auto-anki first."}))
        else:
            print("No state file found. Run auto-anki first to generate data.")
        sys.exit(1)

    if not chat_root or not chat_root.exists():
        if args.json:
            print(json.dumps({"error": f"Chat directory not found: {chat_root}"}))
        else:
            print(f"Chat directory not found: {chat_root}")
            print("Check chat_root in auto_anki_config.json")
        sys.exit(1)

    # Load state
    state = StateTracker(state_path)

    # Scan conversations and compute stats
    all_conversations = scan_conversations(chat_root)
    processed_files = get_processed_files_with_dates(state)
    cards_by_week = get_run_history_by_week(state)

    weekly_stats = compute_weekly_stats(
        all_conversations, processed_files, cards_by_week, weeks_back=args.weeks
    )
    summary = compute_summary(all_conversations, processed_files, weekly_stats)

    # Output
    if args.json:
        output = {
            "summary": {
                **asdict(summary),
                "first_activity_date": (
                    summary.first_activity_date.isoformat()
                    if summary.first_activity_date
                    else None
                ),
                "last_activity_date": (
                    summary.last_activity_date.isoformat()
                    if summary.last_activity_date
                    else None
                ),
            },
            "weekly": [
                {
                    **asdict(w),
                    "week_start": w.week_start.isoformat(),
                }
                for w in weekly_stats
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        if not RICH_AVAILABLE:
            print("Error: 'rich' library not installed.")
            print("Install with: uv pip install rich")
            sys.exit(1)

        console = Console()

        # Handle empty state
        if summary.total_conversations == 0:
            console.print(
                Panel(
                    "[yellow]No conversations found in chat directory.[/yellow]\n\n"
                    f"Looking in: {chat_root}\n\n"
                    "Make sure your ChatGPT exports are in this directory.",
                    title="Getting Started",
                    border_style="yellow",
                )
            )
            return

        if summary.total_processed == 0:
            console.print(
                Panel(
                    "[yellow]No conversations processed yet![/yellow]\n\n"
                    f"Found {summary.total_conversations} conversations waiting.\n\n"
                    "Run [bold]auto-anki[/bold] to start processing.",
                    title="Getting Started",
                    border_style="yellow",
                )
            )
            return

        render_dashboard(summary, weekly_stats, console, show_weeks=args.weeks)


if __name__ == "__main__":
    main()

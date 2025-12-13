#!/usr/bin/env python3
"""
Interactive Shiny UI for reviewing and managing Auto Anki Agent proposed cards.

This implements the Interactive Review Mode from FUTURE_DIRECTIONS.md (sections 11-13):
- Card-by-card review with accept/reject/edit/skip actions
- Rich context display showing source conversations and reasoning
- Progress dashboard with statistics and filtering
- Export functionality for accepted cards

Usage:
    shiny run anki_review_ui.py [--port 8000]

Or with uv:
    uv run --with shiny --with pandas --with plotly shiny run anki_review_ui.py
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import textwrap

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
import pandas as pd

from auto_anki.codex import parse_codex_response_robust, run_codex_exec

# Import AnkiConnect client
try:
    from anki_connect import AnkiConnectClient, AnkiConnectError
    ANKI_CONNECT_AVAILABLE = True
except ImportError:
    ANKI_CONNECT_AVAILABLE = False
    AnkiConnectClient = None
    AnkiConnectError = Exception


# ============================================================================
# Data Loading & Management
# ============================================================================

class CardReviewSession:
    """Manages the state of a card review session."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.cards = self._load_cards()
        self.contexts = self._load_contexts()
        self.decisions = {}  # card_index -> {action, reason, edited_card}
        self.current_index = 0
        self.filtered_indices = list(range(len(self.cards)))  # Active card indices
        self._load_saved_session()  # Restore previous session state if exists

    def _load_cards(self) -> List[Dict]:
        """Load proposed cards from run directory."""
        cards_file = self.run_dir / "all_proposed_cards.json"
        if not cards_file.exists():
            return []
        return json.loads(cards_file.read_text())

    def _load_contexts(self) -> Dict[str, Dict]:
        """Load context information, indexed by context_id.

        Supports both new conversation format (selected_conversations.json)
        and legacy per-turn format (selected_contexts.json).
        """
        # Try new conversation format first
        conversations_file = self.run_dir / "selected_conversations.json"
        if conversations_file.exists():
            return self._load_from_conversations(conversations_file)

        # Fall back to legacy per-turn format
        contexts_file = self.run_dir / "selected_contexts.json"
        if not contexts_file.exists():
            return {}

        contexts_list = json.loads(contexts_file.read_text())
        return {ctx['context_id']: ctx for ctx in contexts_list}

    def _load_from_conversations(self, conversations_file: Path) -> Dict[str, Dict]:
        """Load contexts from conversation-format JSON.

        Extracts individual turns and indexes them by context_id for
        backward-compatible lookup.
        """
        conversations_list = json.loads(conversations_file.read_text())
        contexts: Dict[str, Dict] = {}

        # Also build a conversation index for full-conversation lookups
        self.conversations = {conv['conversation_id']: conv for conv in conversations_list}

        # Extract individual turns for per-turn lookups
        for conv in conversations_list:
            for turn in conv.get('turns', []):
                context_id = turn.get('context_id')
                if context_id:
                    # Add conversation-level info to each turn for richer display
                    turn_context = dict(turn)
                    turn_context['conversation_id'] = conv['conversation_id']
                    turn_context['conversation_title'] = conv.get('source_title')
                    turn_context['conversation_url'] = conv.get('source_url')
                    turn_context['conversation_topics'] = conv.get('key_topics', [])
                    turn_context['total_turns_in_conversation'] = len(conv.get('turns', []))
                    contexts[context_id] = turn_context

        return contexts

    def get_card(self, index: int) -> Optional[Dict]:
        """Get card at index."""
        if 0 <= index < len(self.cards):
            return self.cards[index]
        return None

    def get_context(self, card: Dict) -> Optional[Dict]:
        """Get context for a card.

        Handles both new conversation-based cards (with conversation_id + turn_index)
        and legacy per-turn cards (with context_id only).
        """
        # Try context_id lookup first (works for both old and new formats)
        context = self.contexts.get(card.get('context_id'))
        if context:
            return context

        # For conversation-based cards, try to find by conversation_id + turn_index
        conv_id = card.get('conversation_id')
        turn_idx = card.get('turn_index')
        if conv_id and hasattr(self, 'conversations') and conv_id in self.conversations:
            conv = self.conversations[conv_id]
            turns = conv.get('turns', [])
            if turn_idx is not None and 0 <= turn_idx < len(turns):
                turn = turns[turn_idx]
                # Enrich with conversation context
                turn_context = dict(turn)
                turn_context['conversation_id'] = conv_id
                turn_context['conversation_title'] = conv.get('source_title')
                turn_context['conversation_url'] = conv.get('source_url')
                turn_context['conversation_topics'] = conv.get('key_topics', [])
                turn_context['total_turns_in_conversation'] = len(turns)
                return turn_context

        return None

    def record_decision(self, index: int, action: str, reason: str = "", edited_card: Dict = None):
        """Record a decision for a card."""
        decision = {
            'action': action,
            'reason': reason,
            'edited_card': edited_card,
            'timestamp': datetime.now().isoformat()
        }
        # Accepted/edited cards are typically destined for Anki import; track pending status.
        if action in ("accept", "edit"):
            decision.setdefault("anki_import", {"status": "pending"})
        self.decisions[index] = decision

    def mark_anki_import_status(
        self,
        index: int,
        status: str,
        note_id: Optional[int] = None,
    ):
        """Mark a card's Anki import status on the decision record.

        Status values (convention):
          - pending: accepted/edited but not imported yet
          - imported: imported successfully
          - duplicate: already existed in Anki
          - imported_or_duplicate: batch import where individual mapping isn't available
        """
        decision = self.decisions.get(index)
        if not decision:
            # If it was imported without a prior decision, treat it as accepted.
            self.record_decision(index, "accept", reason="Imported to Anki")
            decision = self.decisions.get(index)
            if not decision:
                return

        decision["anki_import"] = {
            "status": status,
            "note_id": note_id,
            "timestamp": datetime.now().isoformat(),
        }

    def get_pending_anki_import_indices(self) -> List[int]:
        """Return indices of accepted/edited cards that still need Anki import."""
        pending: List[int] = []
        for idx, decision in self.decisions.items():
            if decision.get("action") not in ("accept", "edit"):
                continue
            import_info = decision.get("anki_import") or {"status": "pending"}
            if import_info.get("status") in ("imported", "duplicate", "imported_or_duplicate"):
                continue
            pending.append(idx)
        return sorted(pending)

    def get_stats(self) -> Dict:
        """Get session statistics."""
        actions = [d['action'] for d in self.decisions.values()]
        return {
            'total': len(self.cards),
            'reviewed': len(self.decisions),
            'accepted': actions.count('accept'),
            'rejected': actions.count('reject'),
            'edited': actions.count('edit'),
            'skipped': actions.count('skip'),
            'remaining': len(self.cards) - len(self.decisions)
        }

    def apply_filters(
        self,
        deck_filter: str = "all",
        min_confidence: float = 0.0,
        duplicate_filter: str = "all",
    ):
        """Apply filters to card list and update filtered_indices."""
        self.filtered_indices = []
        for idx, card in enumerate(self.cards):
            # Deck filter
            if deck_filter != "all" and card.get('deck') != deck_filter:
                continue

            # Confidence filter
            if card.get('confidence', 0) < min_confidence:
                continue

            # Duplicate filter
            if duplicate_filter != "all":
                dup_flags = card.get('duplicate_flags', {})
                is_dup = dup_flags.get('is_likely_duplicate', False)
                if duplicate_filter == "duplicates_only" and not is_dup:
                    continue
                if duplicate_filter == "unique_only" and is_dup:
                    continue

            self.filtered_indices.append(idx)

    def get_filtered_card(self, filtered_index: int) -> Optional[Dict]:
        """Get card at filtered index."""
        if 0 <= filtered_index < len(self.filtered_indices):
            actual_index = self.filtered_indices[filtered_index]
            return self.cards[actual_index]
        return None

    def get_filtered_index_from_actual(self, actual_index: int) -> int:
        """Convert actual card index to filtered index."""
        try:
            return self.filtered_indices.index(actual_index)
        except ValueError:
            return 0

    def bulk_accept_high_confidence(self, threshold: float = 0.90):
        """Accept all cards above confidence threshold."""
        count = 0
        for idx, card in enumerate(self.cards):
            if idx not in self.decisions and card.get('confidence', 0) >= threshold:
                self.record_decision(idx, 'accept', reason=f"Auto-accepted (confidence >= {threshold})")
                count += 1
        return count

    def export_accepted(self, output_path: Path):
        """Export accepted cards to JSON."""
        accepted = []
        for idx, card in enumerate(self.cards):
            decision = self.decisions.get(idx)
            if decision and decision['action'] in ['accept', 'edit']:
                # Use edited card if available, otherwise use original
                edited = decision.get('edited_card')
                export_card = edited if edited else card
                accepted.append(export_card)

        output_path.write_text(json.dumps(accepted, indent=2))
        return len(accepted)

    def export_feedback(self, output_path: Path):
        """Export all decisions with feedback for learning."""
        feedback = []
        for idx, card in enumerate(self.cards):
            decision = self.decisions.get(idx)
            if decision:
                feedback.append({
                    'card_index': idx,
                    'context_id': card.get('context_id'),
                    'deck': card.get('deck'),
                    'confidence': card.get('confidence'),
                    'action': decision['action'],
                    'reason': decision.get('reason', ''),
                    'timestamp': decision['timestamp']
                })

        output_path.write_text(json.dumps(feedback, indent=2))
        return len(feedback)

    def _load_saved_session(self):
        """Restore previous session state if exists."""
        session_file = self.run_dir / ".review_session.json"
        if session_file.exists():
            try:
                data = json.loads(session_file.read_text())
                self.decisions = {int(k): v for k, v in data.get("decisions", {}).items()}
                self.current_index = data.get("current_index", 0)
                saved_filters = data.get("filtered_indices")
                if saved_filters:
                    self.filtered_indices = saved_filters
            except (json.JSONDecodeError, KeyError):
                pass  # Ignore corrupted session file

    def save_session(self):
        """Persist current session state to disk after each action."""
        session_file = self.run_dir / ".review_session.json"
        data = {
            "timestamp": datetime.now().isoformat(),
            "current_index": self.current_index,
            "decisions": {str(k): v for k, v in self.decisions.items()},
            "filtered_indices": self.filtered_indices,
        }
        session_file.write_text(json.dumps(data, indent=2))


def _check_run_complete(run_dir: Path) -> bool:
    """Check if a run is 100% reviewed by reading session file.

    A run is complete when all cards have decisions recorded.
    """
    session_file = run_dir / ".review_session.json"
    cards_file = run_dir / "all_proposed_cards.json"

    if not cards_file.exists():
        return False

    try:
        cards_data = json.loads(cards_file.read_text())
        if isinstance(cards_data, list) and len(cards_data) == 0:
            # Nothing to review; consider run complete for archiving/UI labeling.
            return True

        if not session_file.exists():
            return False

        session_data = json.loads(session_file.read_text())

        total_cards = len(cards_data)
        decisions = session_data.get("decisions", {})

        return len(decisions) >= total_cards and total_cards > 0
    except (json.JSONDecodeError, KeyError):
        return False


def _get_run_card_count(run_dir: Path) -> Optional[int]:
    """Return number of proposed cards in a run, or None if unavailable."""
    cards_file = run_dir / "all_proposed_cards.json"
    if not cards_file.exists():
        return None
    try:
        cards_data = json.loads(cards_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(cards_data, list):
        return None
    return len(cards_data)


def _is_run_empty(run_dir: Path) -> bool:
    """Return True if a run has an explicit empty all_proposed_cards.json (i.e., [])."""
    count = _get_run_card_count(run_dir)
    return count == 0


def _parse_run_timestamp(run_dir: Path) -> Optional[datetime]:
    """Parse the run timestamp from a run directory name.

    Expects names like ``run-YYYYMMDD-HHMMSS``. Returns None if parsing fails.
    """
    name = run_dir.name
    if not name.startswith("run-"):
        return None
    ts_str = name[len("run-") :]
    try:
        return datetime.strptime(ts_str, "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def _run_sort_key(run_dir: Path) -> float:
    """Return a sortable key for a run directory based on its timestamp."""
    ts = _parse_run_timestamp(run_dir)
    if ts is not None:
        return ts.timestamp()
    # Fallback to filesystem mtime if name doesn't match expected pattern
    return run_dir.stat().st_mtime


def _parse_user_timestamp(value: object) -> Optional[datetime]:
    """Parse a user timestamp from run metadata.

    Timestamps typically look like ``YYYY-MM-DD HH:MM:SS`` but may also be ISO-8601.
    Returns None if parsing fails.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    ts_str = str(value).strip()
    if not ts_str:
        return None

    # Handle common "Z" suffix for UTC timestamps.
    if ts_str.endswith("Z"):
        try:
            return datetime.fromisoformat(ts_str[:-1] + "+00:00")
        except ValueError:
            pass

    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue

    return None


def _median_datetime(values: List[datetime]) -> Optional[datetime]:
    """Return the median datetime (by POSIX timestamp) for a non-empty list."""
    if not values:
        return None
    timestamps = sorted(dt.timestamp() for dt in values)
    n = len(timestamps)
    mid = n // 2
    if n % 2 == 1:
        median_ts = timestamps[mid]
    else:
        median_ts = (timestamps[mid - 1] + timestamps[mid]) / 2
    return datetime.fromtimestamp(median_ts)


def _load_run_timestamp_lookups(run_dir: Path) -> Tuple[Dict[str, datetime], Dict[Tuple[str, int], datetime]]:
    """Load timestamp lookups for a run.

    Returns:
        - context_id -> datetime (from selected_contexts.json or selected_conversations.json turns)
        - (conversation_id, turn_index) -> datetime (from selected_conversations.json turns)
    """
    context_ts: Dict[str, datetime] = {}
    conv_turn_ts: Dict[Tuple[str, int], datetime] = {}

    contexts_file = run_dir / "selected_contexts.json"
    if contexts_file.exists():
        try:
            contexts_list = json.loads(contexts_file.read_text())
            for ctx in contexts_list:
                context_id = ctx.get("context_id")
                dt = _parse_user_timestamp(ctx.get("user_timestamp"))
                if context_id and dt is not None:
                    context_ts[context_id] = dt
        except (json.JSONDecodeError, OSError, AttributeError):
            pass

    conversations_file = run_dir / "selected_conversations.json"
    if conversations_file.exists():
        try:
            conversations_list = json.loads(conversations_file.read_text())
            for conv in conversations_list:
                conv_id = conv.get("conversation_id")
                turns = conv.get("turns", [])
                if not conv_id or not isinstance(turns, list):
                    continue
                for idx, turn in enumerate(turns):
                    dt = _parse_user_timestamp(turn.get("user_timestamp"))
                    if dt is None:
                        continue
                    conv_turn_ts[(conv_id, idx)] = dt
                    context_id = turn.get("context_id")
                    if context_id and context_id not in context_ts:
                        context_ts[context_id] = dt
        except (json.JSONDecodeError, OSError, AttributeError):
            pass

    return context_ts, conv_turn_ts


def _get_run_card_timestamps(run_dir: Path) -> Tuple[List[datetime], int]:
    """Return (card timestamps, total cards) for a run."""
    cards_file = run_dir / "all_proposed_cards.json"
    if not cards_file.exists():
        return [], 0

    try:
        cards = json.loads(cards_file.read_text())
        if not isinstance(cards, list):
            return [], 0
    except (json.JSONDecodeError, OSError):
        return [], 0

    context_ts, conv_turn_ts = _load_run_timestamp_lookups(run_dir)

    timestamps: List[datetime] = []
    for card in cards:
        if not isinstance(card, dict):
            continue

        context_id = card.get("context_id")
        if context_id and context_id in context_ts:
            timestamps.append(context_ts[context_id])
            continue

        conv_id = card.get("conversation_id")
        turn_idx = card.get("turn_index")
        if conv_id and isinstance(turn_idx, int):
            dt = conv_turn_ts.get((conv_id, turn_idx))
            if dt is not None:
                timestamps.append(dt)

    return timestamps, len(cards)


def _run_card_median_sort_key(run_dir: Path) -> float:
    """Return a sortable key for a run based on median card timestamp."""
    card_ts, _total = _get_run_card_timestamps(run_dir)
    median_dt = _median_datetime(card_ts)
    if median_dt is not None:
        return median_dt.timestamp()
    return _run_sort_key(run_dir)


def _format_run_card_date_label(run_dir: Path) -> str:
    """Format a concise run label suffix describing card date distribution."""
    card_ts, total_cards = _get_run_card_timestamps(run_dir)
    if not card_ts:
        if total_cards == 0:
            return " [EMPTY]"
        if total_cards is not None and total_cards > 0:
            return f" [cards: no timestamps]"
        return ""

    median_dt = _median_datetime(card_ts)
    min_dt = min(card_ts)
    max_dt = max(card_ts)

    start = min_dt.strftime("%Y-%m-%d")
    end = max_dt.strftime("%Y-%m-%d")
    median = median_dt.strftime("%Y-%m-%d") if median_dt is not None else start

    if start == end:
        date_part = start
    else:
        date_part = f"{start}..{end} (median {median})"

    if len(card_ts) != total_cards and total_cards > 0:
        return f" [cards {date_part}; ts {len(card_ts)}/{total_cards}]"
    return f" [cards {date_part}]"


def list_runs_by_date(
    base_dir: Path = Path("auto_anki_runs"),
    limit: Optional[int] = None,
    reverse: bool = True,
    order_by_card_median: bool = False,
    include_empty: bool = True,
) -> List[tuple]:
    """List non-archived run directories with completion status.

    Returns list of (run_path, is_complete) tuples, ordered by the timestamp
    encoded in the run folder name (or filesystem mtime as a fallback). If
    ``limit`` is provided, only the first ``limit`` runs are returned.
    """
    if not base_dir.exists():
        return []

    run_dirs = [
        d
        for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith("run-") and d.name != "archived"
    ]

    if not include_empty:
        run_dirs = [d for d in run_dirs if not _is_run_empty(d)]

    if order_by_card_median:
        run_dirs.sort(key=_run_card_median_sort_key, reverse=reverse)
    else:
        run_dirs.sort(key=_run_sort_key, reverse=reverse)

    if limit is not None:
        run_dirs = run_dirs[:limit]

    results = []
    for run_dir in run_dirs:
        is_complete = _check_run_complete(run_dir)
        results.append((run_dir, is_complete))

    return results


def archive_run(run_dir: Path, archive_base: Path = None) -> Path:
    """Move a completed run to the archive folder.

    Args:
        run_dir: Path to the run directory to archive
        archive_base: Base path for archive folder. Defaults to run_dir.parent / "archived"

    Returns:
        Path to the archived run directory
    """
    if archive_base is None:
        archive_base = run_dir.parent / "archived"

    archive_base.mkdir(exist_ok=True)
    dest = archive_base / run_dir.name

    shutil.move(str(run_dir), str(dest))
    return dest


def find_archived_runs(
    base_dir: Path = Path("auto_anki_runs"),
    limit: int = 20,
    reverse: bool = True,
    order_by_card_median: bool = False,
    include_empty: bool = True,
) -> List[tuple]:
    """Find archived run directories with completion status.

    Returns list of (run_path, is_complete) tuples from the archived folder.
    """
    archived_dir = base_dir / "archived"
    if not archived_dir.exists():
        return []

    return list_runs_by_date(
        base_dir=archived_dir,
        limit=limit,
        reverse=reverse,
        order_by_card_median=order_by_card_median,
        include_empty=include_empty,
    )


# ============================================================================
# Shiny UI Definition
# ============================================================================

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style("""
            .card-front {
                background-color: #f8f9fa;
                padding: 15px;
                border-left: 4px solid #007bff;
                margin: 10px 0;
                font-size: 1.1em;
            }
            .card-back {
                background-color: #e9ecef;
                padding: 15px;
                border-left: 4px solid #28a745;
                margin: 10px 0;
            }
            .context-box {
                background-color: #fff3cd;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                max-height: 300px;
                overflow-y: auto;
            }
            .signal-badge {
                display: inline-block;
                padding: 3px 8px;
                margin: 2px;
                border-radius: 3px;
                font-size: 0.85em;
            }
            .signal-positive { background-color: #d4edda; color: #155724; }
            .signal-negative { background-color: #f8d7da; color: #721c24; }
            .stats-card {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 15px;
                margin: 10px 0;
            }
            .progress-custom {
                height: 30px;
                font-size: 1em;
            }
            .keyboard-hint {
                font-size: 0.85em;
                color: #6c757d;
                margin-left: 5px;
            }
        """),
        ui.tags.script("""
            // Keyboard shortcuts for card review
            $(document).on('keydown', function(e) {
                // Don't trigger if user is typing in an input field
                if ($(e.target).is('input, textarea')) {
                    return;
                }

                // Prevent default for our shortcuts
                if (['a', 'r', 'e', 's', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
                    e.preventDefault();
                }

                // Trigger appropriate button click
                switch(e.key) {
                    case 'a':
                        $('#btn_accept').click();
                        break;
                    case 'r':
                        $('#btn_reject').click();
                        break;
                    case 'e':
                        $('#btn_edit').click();
                        break;
                    case 's':
                        $('#btn_skip').click();
                        break;
                    case 'ArrowLeft':
                        $('#btn_prev').click();
                        break;
                    case 'ArrowRight':
                        $('#btn_next').click();
                        break;
                }
            });
        """)
    ),

    ui.h1("🎓 Auto Anki Card Review"),
    ui.hr(),

    # Run Selection
    ui.row(
        ui.column(8,
            ui.input_select(
                "run_select",
                "Select Run:",
                choices={},
                width="100%"
            ),
            ui.input_checkbox(
                "run_order_desc",
                "Show newest runs first",
                value=True,
            ),
            ui.input_checkbox(
                "run_order_by_card_median",
                "Order runs by median card date",
                value=False,
            ),
            ui.input_checkbox(
                "show_empty_runs",
                "Show empty runs (0 cards)",
                value=False,
            ),
            ui.input_action_button(
                "btn_archive_empty_runs",
                "Archive Empty Runs",
                class_="btn-outline-secondary",
                width="100%",
            ),
            ui.output_text("empty_runs_message", inline=True),
        ),
        ui.column(4,
            ui.output_ui("run_info")
        )
    ),

    ui.hr(),

    # Main Review Interface
    ui.panel_conditional(
        "input.run_select != ''",

        ui.row(
            # Left Panel - Card Review
            ui.column(8,
                ui.h3("Card Review"),

                # Progress Bar
                ui.output_ui("progress_bar"),

                # Card Display
                ui.div(
                    {"class": "stats-card"},
                    ui.h4(ui.output_text("card_counter")),

                    # Decision Status
                    ui.output_ui("decision_status"),

                    # Duplicate Warning (if flagged)
                    ui.output_ui("duplicate_warning"),

                    # Deck and Tags
                    ui.div(
                        ui.strong("Deck: "),
                        ui.output_text("card_deck", inline=True),
                        ui.br(),
                        ui.strong("Tags: "),
                        ui.output_text("card_tags", inline=True),
                        ui.br(),
                        ui.strong("Confidence: "),
                        ui.output_text("card_confidence", inline=True),
                    ),

                    ui.hr(),

                    # Front of Card
                    ui.div(
                        {"class": "card-front"},
                        ui.strong("Front:"),
                        ui.br(),
                        ui.output_text("card_front")
                    ),

                    # Back of Card
                    ui.div(
                        {"class": "card-back"},
                        ui.strong("Back:"),
                        ui.br(),
                        ui.output_text("card_back")
                    ),

                    # Card Notes
                    ui.div(
                        {"style": "margin-top: 10px; font-style: italic;"},
                        ui.strong("Notes: "),
                        ui.output_text("card_notes", inline=True)
                    ),
                ),

                ui.hr(),

                # Action Buttons with Keyboard Hints
                ui.div(
                    ui.div(
                        ui.input_action_button("btn_accept", "✓ Accept", class_="btn-success"),
                        ui.span(" [A]", class_="keyboard-hint"),
                        {"style": "display: inline-block; margin: 5px;"}
                    ),
                    ui.div(
                        ui.input_action_button("btn_reject", "✗ Reject", class_="btn-danger"),
                        ui.span(" [R]", class_="keyboard-hint"),
                        {"style": "display: inline-block; margin: 5px;"}
                    ),
                    ui.div(
                        ui.input_action_button("btn_edit", "✎ Edit", class_="btn-warning"),
                        ui.span(" [E]", class_="keyboard-hint"),
                        {"style": "display: inline-block; margin: 5px;"}
                    ),
                    ui.div(
                        ui.input_action_button("btn_skip", "⊙ Skip", class_="btn-secondary"),
                        ui.span(" [S]", class_="keyboard-hint"),
                        {"style": "display: inline-block; margin: 5px;"}
                    ),
                    ui.div(
                        ui.input_action_button("btn_prev", "← Previous", class_="btn-info"),
                        ui.span(" [←]", class_="keyboard-hint"),
                        {"style": "display: inline-block; margin: 5px;"}
                    ),
                    ui.div(
                        ui.input_action_button("btn_next", "Next →", class_="btn-info"),
                        ui.span(" [→]", class_="keyboard-hint"),
                        {"style": "display: inline-block; margin: 5px;"}
                    ),
                ),

                # Rejection Reason (conditional, shown after reject)
                ui.panel_conditional(
                    "input.btn_reject > 0",
                    ui.div(
                        {"class": "stats-card", "style": "margin-top: 15px;"},
                        ui.h4("Rejection Reason (Optional)"),
                        ui.input_select(
                            "reject_reason",
                            "",
                            choices={
                                "": "-- Select reason --",
                                "duplicate": "Duplicate / Too similar to existing card",
                                "low_quality": "Low quality / Poorly phrased",
                                "not_relevant": "Not relevant / Out of scope",
                                "too_vague": "Too vague / Lacks context",
                                "too_specific": "Too specific / Overly detailed",
                                "factually_wrong": "Factually incorrect",
                                "other": "Other"
                            },
                            width="100%"
                        ),
                        ui.input_text("reject_reason_other", "If other, please specify:", width="100%"),
                    )
                ),

                # Edit Panel (rendered conditionally from server)
                ui.output_ui("edit_panel"),

                ui.div(
                    {"class": "stats-card", "style": "margin-top: 15px;"},
                    ui.h4("Codex Update"),
                    ui.input_text_area(
                        "codex_instructions",
                        "Instructions for Codex (optional):",
                        rows=3,
                        width="100%",
                    ),
                    ui.input_action_button(
                        "btn_codex_update",
                        "🤖 Suggest Update via Codex",
                        class_="btn-outline-primary",
                    ),
                    ui.div(
                        {"style": "margin-top: 10px; font-size: 0.9em;"},
                        ui.output_text("codex_message", inline=True),
                    ),
                ),

                # Source Context Display
                ui.h4({"style": "margin-top: 20px;"}, "Source Context"),
                ui.div(
                    {"class": "context-box"},
                    ui.strong("User Prompt:"),
                    ui.br(),
                    ui.output_text("context_user"),
                    ui.br(), ui.br(),
                    ui.strong("Assistant Answer:"),
                    ui.br(),
                    ui.output_text("context_assistant"),
                ),

                # Source Metadata
                ui.div(
                    {"style": "margin-top: 10px; font-size: 0.9em;"},
                    ui.output_ui("context_metadata")
                ),

                # Quality Signals
                ui.h4({"style": "margin-top: 20px;"}, "Quality Signals"),
                ui.output_ui("quality_signals"),
            ),

            # Right Panel - Statistics & Controls
            ui.column(4,
                ui.h3("Session Statistics"),

                ui.div(
                    {"class": "stats-card"},
                    ui.output_ui("stats_summary")
                ),

                ui.div(
                    {"class": "stats-card"},
                    ui.h4("Actions"),
                    ui.output_ui("stats_actions")
                ),

                ui.div(
                    {"class": "stats-card"},
                    ui.h4("Filters"),
                    ui.input_select(
                        "filter_deck",
                        "Filter by Deck:",
                        choices={"all": "All Decks"},
                    ),
                    ui.input_slider(
                        "filter_confidence",
                        "Minimum Confidence:",
                        min=0.0,
                        max=1.0,
                        value=0.0,
                        step=0.05
                    ),
                    ui.input_select(
                        "filter_duplicate",
                        "Duplicate Status:",
                        choices={
                            "all": "All Cards",
                            "duplicates_only": "Likely Duplicates Only",
                            "unique_only": "Unique Cards Only",
                        },
                    ),
                    ui.input_action_button("btn_apply_filters", "Apply Filters", class_="btn-primary", width="100%"),
                    ui.output_text("filter_message", inline=True),
                ),

                ui.div(
                    {"class": "stats-card"},
                    ui.h4("Bulk Actions"),
                    ui.input_slider(
                        "bulk_confidence_threshold",
                        "Auto-accept threshold:",
                        min=0.80,
                        max=1.0,
                        value=0.90,
                        step=0.05
                    ),
                    ui.input_action_button("btn_bulk_accept", "Accept All High Confidence", class_="btn-warning", width="100%"),
                    ui.output_text("bulk_message", inline=True),
                ),

                ui.div(
                    {"class": "stats-card"},
                    ui.h4("AnkiConnect"),
                    ui.output_ui("anki_status"),
                    ui.input_action_button("btn_import_current", "Import Current Card to Anki", class_="btn-primary", width="100%"),
                    ui.input_action_button("btn_import_accepted", "Import All Accepted to Anki", class_="btn-success", width="100%", style="margin-top: 10px;"),
                    ui.input_checkbox("allow_duplicates", "Allow duplicate cards", value=False),
                    ui.output_text("anki_message", inline=True),
                ),

                ui.div(
                    {"class": "stats-card"},
                    ui.h4("Export"),
                    ui.input_action_button("btn_export", "Export Accepted Cards (JSON)", class_="btn-success", width="100%"),
                    ui.input_action_button("btn_export_feedback", "Export Feedback Data", class_="btn-info", width="100%", style="margin-top: 10px;"),
                    ui.output_text("export_message", inline=True),
                    ui.hr(),
                    ui.input_action_button("btn_archive_run", "Archive This Run", class_="btn-outline-secondary", width="100%"),
                    ui.output_text("archive_message", inline=True),
                ),

                ui.div(
                    {"class": "stats-card"},
                    ui.h4("Navigation"),
                    ui.input_numeric("jump_to", "Jump to card:", value=1, min=1),
                    ui.input_action_button("btn_jump", "Go", class_="btn-primary", width="100%"),
                ),

                ui.div(
                    {"class": "stats-card"},
                    ui.h4("Archived Runs"),
                    ui.input_select(
                        "archived_run_select",
                        "View archived run:",
                        choices={"": "-- Select archived run --"},
                        width="100%"
                    ),
                    ui.input_action_button("btn_load_archived", "Load Archived Run", class_="btn-outline-info", width="100%"),
                    ui.input_action_button("btn_unarchive", "Unarchive Selected", class_="btn-outline-warning", width="100%", style="margin-top: 10px;"),
                ),
            )
        )
    )
)


# ============================================================================
# Shiny Server Logic
# ============================================================================

def build_codex_update_prompt(
    card: Dict,
    context: Optional[Dict],
    user_instructions: Optional[str] = None,
) -> str:
    """
    Build a focused prompt asking Codex to refine a single card.

    The model must respond with JSON only, matching the output_contract.
    """
    contract = {
        "updated_card": {
            "front": "string",
            "back": "string",
            "tags": ["list", "of", "tags"],
            "keep": "boolean (true if this card should remain in the deck; false if it should be discarded)",
            "notes": "short explanation of changes or issues",
        }
    }

    payload: Dict[str, object] = {
        "current_card": {
            "deck": card.get("deck"),
            "front": card.get("front"),
            "back": card.get("back"),
            "tags": card.get("tags", []),
            "confidence": card.get("confidence"),
            "notes": card.get("notes"),
        },
        "output_contract": contract,
    }

    if user_instructions:
        payload["user_instructions"] = user_instructions

    if context:
        payload["source_context"] = {
            "context_id": context.get("context_id"),
            "source_path": context.get("source_path"),
            "source_title": context.get("source_title") or context.get("conversation_title"),
            "source_url": context.get("source_url") or context.get("conversation_url"),
            "user_prompt": context.get("user_prompt"),
            "assistant_answer": context.get("assistant_answer"),
            "score": context.get("score"),
            "signals": context.get("signals"),
            "key_terms": context.get("key_terms"),
            "key_points": context.get("key_points"),
            # Conversation-level context (new format)
            "conversation_id": context.get("conversation_id"),
            "turn_index": context.get("turn_index"),
            "conversation_topics": context.get("conversation_topics"),
            "total_turns_in_conversation": context.get("total_turns_in_conversation"),
        }

    instructions = textwrap.dedent(
        """
        CRITICAL: You MUST respond with ONLY valid JSON matching the `output_contract` below.
        Do NOT include markdown, prose, or any text outside the JSON object.
        Do NOT wrap the JSON in ```json fences.

        You are helping a human refine a single Anki flashcard that was originally
        generated from a ChatGPT conversation. Your goals:

        1. Apply the Minimum Information Principle:
           - Each card should test one small, well-scoped fact or concept.
           - The question (front) must be clear and unambiguous.
           - The answer (back) must be as short as possible while still complete.
        2. Improve clarity, correctness, and usefulness.
        3. Preserve the learner's intent and technical accuracy.

        A human reviewer may also provide `user_instructions` describing exactly
        how they want this card changed (for example: "convert to cloze", "shorten
        the back to one sentence", or "flip front/back so question is X").
        When `user_instructions` is present in the payload, you MUST treat it as
        the primary source of guidance and follow it as closely as possible, while
        still maintaining correctness and clarity.

        Given the `current_card` and optional `source_context`, you should:
        - Fix wording, grammar, and structure.
        - Tighten or simplify the question.
        - Shorten overly long answers while keeping key information.
        - Adjust tags if helpful.
        - Set `keep = false` only if the card is fundamentally unsalvageable
          (e.g., redundant, trivial, or inherently confusing).

        Output requirements:
        - Return a single JSON object.
        - Top-level key MUST be `updated_card`.
        - Provide `front`, `back`, `tags`, `keep`, and `notes`.
        - `tags` should be a list of short strings.
        - START your response with `{` and END with `}`.
        """
    ).strip()

    return instructions + "\n\n" + json.dumps(payload, indent=2, ensure_ascii=False)


def server(input: Inputs, output: Outputs, session: Session):
    # Reactive values
    session_state = reactive.Value(None)
    current_card_index = reactive.Value(0)
    edit_mode_active = reactive.Value(False)
    codex_msg = reactive.Value("")
    stats_trigger = reactive.Value(0)  # Trigger to force stats UI update
    pending_edit = reactive.Value(None)  # Stores {front, back, tags} from Codex or None
    empty_runs_msg = reactive.Value("")

    def clear_card_state():
        """Reset card-specific UI state when changing cards."""
        codex_msg.set("")
        edit_mode_active.set(False)  # This will hide edit panel (inputs are dynamically created)
        pending_edit.set(None)  # Clear any pending Codex suggestions
        ui.update_select("reject_reason", selected="")
        ui.update_text("reject_reason_other", value="")
        ui.update_text_area("codex_instructions", value="")

    # AnkiConnect client (initialized once)
    anki_client = AnkiConnectClient() if ANKI_CONNECT_AVAILABLE else None

    # Initialize run options (active and archived)
    @reactive.Effect
    def _():
        # Populate active runs dropdown
        runs = list_runs_by_date(
            reverse=input.run_order_desc(),
            order_by_card_median=input.run_order_by_card_median(),
            include_empty=input.show_empty_runs(),
        )
        choices = {}
        for run_path, is_complete in runs:
            is_empty = _is_run_empty(run_path)
            label = run_path.name
            label += _format_run_card_date_label(run_path)
            if is_complete and not is_empty:
                label += " [DONE]"
            choices[str(run_path)] = label
        if not choices:
            choices = {"": "No runs found"}
        ui.update_select("run_select", choices=choices)

        # Populate archived runs dropdown
        archived = find_archived_runs(
            reverse=input.run_order_desc(),
            order_by_card_median=input.run_order_by_card_median(),
            include_empty=input.show_empty_runs(),
        )
        archived_choices = {"": "-- Select archived run --"}
        for run_path, is_complete in archived:
            is_empty = _is_run_empty(run_path)
            label = run_path.name
            label += _format_run_card_date_label(run_path)
            if is_complete and not is_empty:
                label += " [DONE]"
            archived_choices[str(run_path)] = label
        ui.update_select("archived_run_select", choices=archived_choices)

    # Load session when run is selected
    @reactive.Effect
    @reactive.event(input.run_select)
    def _():
        if input.run_select():
            run_path = Path(input.run_select())
            if run_path.exists():
                session_state.set(CardReviewSession(run_path))

                # Restore current_index from saved session (or start at 0)
                session = session_state.get()
                if session:
                    current_card_index.set(session.current_index)

                    # Update deck filter options
                    decks = list(set(card.get('deck', 'Unknown') for card in session.cards))
                    deck_choices = {"all": "All Decks"}
                    deck_choices.update({d: d for d in sorted(decks)})
                    ui.update_select("filter_deck", choices=deck_choices)

    # Run info display
    @output
    @render.ui
    def run_info():
        if input.run_select():
            run_path = Path(input.run_select())
            pending_line = ""
            session = session_state.get()
            if session and session.run_dir == run_path:
                pending_count = len(session.get_pending_anki_import_indices())
                if pending_count > 0:
                    pending_line = ui.div(
                        {"style": "color: #b02a37; margin-top: 6px;"},
                        ui.strong("Pending Anki import: "),
                        str(pending_count),
                    )
            return ui.div(
                ui.strong("Run: "),
                run_path.name,
                ui.br(),
                ui.strong("Date: "),
                datetime.fromtimestamp(run_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                pending_line,
            )
        return ""

    @output
    @render.text
    def empty_runs_message():
        return empty_runs_msg.get()

    # Get current card (returns edited version if available)
    def get_current_card():
        session = session_state.get()
        if not session:
            return None
        idx = current_card_index.get()
        # Check if there's an edited version
        decision = session.decisions.get(idx)
        if decision and decision.get('action') == 'edit' and decision.get('edited_card'):
            return decision['edited_card']
        return session.get_card(idx)

    # Progress bar
    @output
    @render.ui
    def progress_bar():
        session = session_state.get()
        if not session:
            return ""

        stats = session.get_stats()
        progress_pct = (stats['reviewed'] / stats['total'] * 100) if stats['total'] > 0 else 0

        return ui.div(
            ui.tags.div(
                {"class": "progress"},
                ui.tags.div(
                    {"class": "progress-bar progress-bar-striped progress-custom",
                     "role": "progressbar",
                     "style": f"width: {progress_pct}%",
                     "aria-valuenow": str(progress_pct),
                     "aria-valuemin": "0",
                     "aria-valuemax": "100"},
                    f"{stats['reviewed']} / {stats['total']} ({progress_pct:.1f}%)"
                )
            )
        )

    # Card counter
    @output
    @render.text
    def card_counter():
        session = session_state.get()
        if not session:
            return ""
        idx = current_card_index.get()
        return f"Card {idx + 1} of {len(session.cards)}"

    # Decision status indicator
    @output
    @render.ui
    def decision_status():
        stats_trigger.get()  # Re-render when decisions change
        session = session_state.get()
        if not session:
            return ui.div()

        idx = current_card_index.get()
        decision = session.decisions.get(idx)

        if not decision:
            return ui.div(
                {"style": "padding: 8px; margin: 10px 0; background-color: #f0f0f0; border-radius: 4px; text-align: center;"},
                ui.span({"style": "color: #666;"}, "⏳ Not yet reviewed")
            )

        action = decision.get('action', '')
        reason = decision.get('reason', '')

        if action == 'accept':
            return ui.div(
                {"style": "padding: 8px; margin: 10px 0; background-color: #d4edda; border-radius: 4px; border-left: 4px solid #28a745; text-align: center;"},
                ui.span({"style": "color: #155724; font-weight: bold;"}, "✓ ACCEPTED"),
                ui.span({"style": "color: #155724; margin-left: 10px;"}, f"({reason})" if reason else "")
            )
        elif action == 'reject':
            reason_text = f" - {reason}" if reason else ""
            return ui.div(
                {"style": "padding: 8px; margin: 10px 0; background-color: #f8d7da; border-radius: 4px; border-left: 4px solid #dc3545; text-align: center;"},
                ui.span({"style": "color: #721c24; font-weight: bold;"}, "✗ REJECTED"),
                ui.span({"style": "color: #721c24; margin-left: 10px;"}, reason_text)
            )
        elif action == 'edit':
            return ui.div(
                {"style": "padding: 8px; margin: 10px 0; background-color: #d4edda; border-radius: 4px; border-left: 4px solid #28a745; text-align: center;"},
                ui.span({"style": "color: #155724; font-weight: bold;"}, "✓ ACCEPTED"),
                ui.span({"style": "color: #155724; margin-left: 5px;"}, "(edited)")
            )
        elif action == 'skip':
            return ui.div(
                {"style": "padding: 8px; margin: 10px 0; background-color: #e2e3e5; border-radius: 4px; border-left: 4px solid #6c757d; text-align: center;"},
                ui.span({"style": "color: #383d41; font-weight: bold;"}, "⊙ SKIPPED")
            )

        return ui.div()

    # Card display (depend on stats_trigger to update after edits)
    @output
    @render.text
    def card_deck():
        stats_trigger.get()  # Re-render when decisions change
        card = get_current_card()
        return card.get('deck', 'Unknown') if card else ""

    @output
    @render.text
    def card_tags():
        stats_trigger.get()  # Re-render when decisions change
        card = get_current_card()
        if card and card.get('tags'):
            return ", ".join(card['tags'])
        return "No tags"

    @output
    @render.text
    def card_confidence():
        stats_trigger.get()  # Re-render when decisions change
        card = get_current_card()
        if card:
            conf = card.get('confidence', 0)
            return f"{conf:.2f} ({conf * 100:.0f}%)"
        return ""

    @output
    @render.text
    def card_front():
        stats_trigger.get()  # Re-render when decisions change
        card = get_current_card()
        return card.get('front', '') if card else ""

    @output
    @render.text
    def card_back():
        stats_trigger.get()  # Re-render when decisions change
        card = get_current_card()
        return card.get('back', '') if card else ""

    @output
    @render.text
    def card_notes():
        stats_trigger.get()  # Re-render when decisions change
        card = get_current_card()
        return card.get('notes', 'No notes') if card else ""

    @output
    @render.ui
    def duplicate_warning():
        """Display a warning if this card is flagged as a likely duplicate."""
        stats_trigger.get()  # Re-render when decisions change
        card = get_current_card()
        if not card:
            return ui.div()

        dup_flags = card.get('duplicate_flags')
        if not dup_flags:
            return ui.div()  # No duplicate flags - old format or post-dedup skipped

        is_dup = dup_flags.get('is_likely_duplicate', False)
        if not is_dup:
            # Card is unique - show a small green indicator
            similarity = dup_flags.get('similarity_score', 0)
            return ui.div(
                {"style": "padding: 8px; margin: 10px 0; background-color: #d4edda; border-left: 4px solid #28a745; border-radius: 4px;"},
                ui.span(
                    {"style": "color: #155724;"},
                    f"✓ Unique card (best match: {similarity:.0%} similarity)"
                )
            )

        # Card is flagged as likely duplicate
        similarity = dup_flags.get('similarity_score', 0)
        matched = dup_flags.get('matched_card', {})

        # Color based on severity
        if similarity >= 0.95:
            bg_color = "#f8d7da"
            border_color = "#dc3545"
            text_color = "#721c24"
            label = "VERY HIGH"
        elif similarity >= 0.90:
            bg_color = "#fff3cd"
            border_color = "#fd7e14"
            text_color = "#856404"
            label = "HIGH"
        else:
            bg_color = "#fff3cd"
            border_color = "#ffc107"
            text_color = "#856404"
            label = "MODERATE"

        warning_content = [
            ui.div(
                {"style": f"font-weight: bold; color: {text_color}; margin-bottom: 5px;"},
                f"⚠ LIKELY DUPLICATE ({label}) - {similarity:.0%} match"
            ),
        ]

        if matched:
            warning_content.append(
                ui.div(
                    {"style": "margin-top: 8px;"},
                    ui.strong("Similar to existing card:"),
                    ui.div(
                        {"style": "background-color: white; padding: 8px; margin-top: 5px; border-radius: 4px; font-size: 0.9em;"},
                        ui.div(ui.strong("Deck: "), matched.get('deck', 'Unknown')),
                        ui.div(
                            {"style": "margin-top: 4px;"},
                            ui.strong("Front: "),
                            matched.get('front', '')[:150] + ("..." if len(matched.get('front', '')) > 150 else "")
                        ),
                    )
                )
            )

        return ui.div(
            {"style": f"padding: 10px; margin: 10px 0; background-color: {bg_color}; border-left: 4px solid {border_color}; border-radius: 4px;"},
            *warning_content
        )

    # Codex update status
    @output
    @render.text
    def codex_message():
        return codex_msg.get()

    # Edit panel (conditionally rendered)
    @output
    @render.ui
    def edit_panel():
        if not edit_mode_active.get():
            return ui.div()  # Empty div when not in edit mode

        card = get_current_card()
        pending = pending_edit.get()

        # Use pending edit values (from Codex) if available, otherwise use card values
        if pending:
            front_val = pending.get('front', card.get('front', '') if card else '')
            back_val = pending.get('back', card.get('back', '') if card else '')
            tags_val = pending.get('tags', [])
            if isinstance(tags_val, list):
                tags_str = ", ".join(tags_val)
            else:
                tags_str = str(tags_val)
        else:
            front_val = card.get('front', '') if card else ''
            back_val = card.get('back', '') if card else ''
            tags_str = ", ".join(card.get('tags', [])) if card else ''

        return ui.div(
            {"class": "stats-card", "style": "margin-top: 15px;"},
            ui.h4("Edit Card"),
            ui.input_text_area("edit_front", "Front:", value=front_val, rows=3, width="100%"),
            ui.input_text_area("edit_back", "Back:", value=back_val, rows=5, width="100%"),
            ui.input_text("edit_tags", "Tags (comma-separated):", value=tags_str, width="100%"),
            ui.input_action_button("btn_save_edit", "Save Changes", class_="btn-primary"),
        )

    # Context display
    @output
    @render.text
    def context_user():
        session = session_state.get()
        card = get_current_card()
        if not session or not card:
            return ""

        context = session.get_context(card)
        if context:
            return context.get('user_prompt', 'No context available')
        return "No context available"

    @output
    @render.text
    def context_assistant():
        session = session_state.get()
        card = get_current_card()
        if not session or not card:
            return ""

        context = session.get_context(card)
        if context:
            answer = context.get('assistant_answer', '')
            # Truncate if too long
            if len(answer) > 1000:
                return answer[:1000] + "... [truncated]"
            return answer
        return "No context available"

    @output
    @render.ui
    def context_metadata():
        session = session_state.get()
        card = get_current_card()
        if not session or not card:
            return ""

        context = session.get_context(card)
        if not context:
            return ui.p("No metadata available")

        metadata_items = []

        # Conversation-level info (new format)
        if context.get('conversation_title'):
            metadata_items.append(ui.div(ui.strong("Conversation: "), context['conversation_title']))
        elif context.get('source_title'):
            metadata_items.append(ui.div(ui.strong("Title: "), context['source_title']))

        # Show turn position within conversation
        if context.get('turn_index') is not None and context.get('total_turns_in_conversation'):
            turn_info = f"Turn {context['turn_index'] + 1} of {context['total_turns_in_conversation']}"
            metadata_items.append(ui.div(ui.strong("Position: "), turn_info))

        # Show key topics from conversation
        if context.get('conversation_topics'):
            topics = ", ".join(context['conversation_topics'][:5])  # Limit to 5 topics
            metadata_items.append(ui.div(ui.strong("Topics: "), topics))

        if context.get('conversation_url'):
            metadata_items.append(ui.div(
                ui.strong("URL: "),
                ui.a(context['conversation_url'], href=context['conversation_url'], target="_blank")
            ))
        elif context.get('source_url'):
            metadata_items.append(ui.div(
                ui.strong("URL: "),
                ui.a(context['source_url'], href=context['source_url'], target="_blank")
            ))

        if context.get('source_path'):
            metadata_items.append(ui.div(ui.strong("File: "), Path(context['source_path']).name))

        if context.get('score'):
            metadata_items.append(ui.div(ui.strong("Quality Score: "), f"{context['score']:.2f}"))

        return ui.div(*metadata_items)

    @output
    @render.ui
    def quality_signals():
        session = session_state.get()
        card = get_current_card()
        if not session or not card:
            return ""

        context = session.get_context(card)
        if not context or not context.get('signals'):
            return ui.p("No signals available")

        signals = context['signals']
        badges = []

        signal_display = {
            'question_like': ('Question-like', 'positive'),
            'definition_like': ('Definition-like', 'positive'),
            'bullet_count': ('Bullets', 'positive'),
            'heading_count': ('Headings', 'positive'),
            'code_blocks': ('Code blocks', 'positive'),
            'imperative': ('Imperative', 'negative'),
        }

        for key, (label, signal_type) in signal_display.items():
            if key in signals:
                value = signals[key]
                if isinstance(value, bool):
                    if value:
                        badges.append(ui.span(label, class_=f"signal-badge signal-{signal_type}"))
                elif isinstance(value, (int, float)) and value > 0:
                    badges.append(ui.span(f"{label}: {value}", class_=f"signal-badge signal-{signal_type}"))

        return ui.div(*badges)

    # Statistics
    @output
    @render.ui
    def stats_summary():
        # Depend on stats_trigger to force re-render when decisions change
        stats_trigger.get()
        session = session_state.get()
        if not session:
            return ui.p("No session loaded")

        stats = session.get_stats()
        return ui.div(
            ui.div(ui.strong("Total Cards: "), stats['total']),
            ui.div(ui.strong("Reviewed: "), stats['reviewed']),
            ui.div(ui.strong("Remaining: "), stats['remaining']),
        )

    @output
    @render.ui
    def stats_actions():
        # Depend on stats_trigger to force re-render when decisions change
        stats_trigger.get()
        session = session_state.get()
        if not session:
            return ""

        stats = session.get_stats()
        return ui.div(
            ui.div(ui.span({"style": "color: green;"}, "✓ Accepted: "), stats['accepted']),
            ui.div(ui.span({"style": "color: red;"}, "✗ Rejected: "), stats['rejected']),
            ui.div(ui.span({"style": "color: orange;"}, "✎ Edited: "), stats['edited']),
            ui.div(ui.span({"style": "color: gray;"}, "⊙ Skipped: "), stats['skipped']),
        )

    # Action handlers
    @reactive.Effect
    @reactive.event(input.btn_accept)
    def _():
        session = session_state.get()
        if session:
            idx = current_card_index.get()
            session.record_decision(idx, 'accept')
            # Move to next card
            if idx < len(session.cards) - 1:
                current_card_index.set(idx + 1)
                session.current_index = idx + 1
                clear_card_state()
            session.save_session()
            stats_trigger.set(stats_trigger.get() + 1)  # Trigger stats update

    @reactive.Effect
    @reactive.event(input.btn_reject)
    def _():
        session = session_state.get()
        if session:
            idx = current_card_index.get()
            # Get rejection reason if provided
            reason = input.reject_reason() if input.reject_reason() else ""
            if reason == "other" and input.reject_reason_other():
                reason = f"other: {input.reject_reason_other()}"
            session.record_decision(idx, 'reject', reason=reason)
            # Move to next card
            if idx < len(session.cards) - 1:
                current_card_index.set(idx + 1)
                session.current_index = idx + 1
                clear_card_state()
            session.save_session()
            stats_trigger.set(stats_trigger.get() + 1)  # Trigger stats update

    @reactive.Effect
    @reactive.event(input.btn_skip)
    def _():
        session = session_state.get()
        if session:
            idx = current_card_index.get()
            session.record_decision(idx, 'skip')
            # Move to next card
            if idx < len(session.cards) - 1:
                current_card_index.set(idx + 1)
                session.current_index = idx + 1
                clear_card_state()
            session.save_session()
            stats_trigger.set(stats_trigger.get() + 1)  # Trigger stats update

    @reactive.Effect
    @reactive.event(input.btn_next)
    def _():
        session = session_state.get()
        if session:
            idx = current_card_index.get()
            if idx < len(session.cards) - 1:
                current_card_index.set(idx + 1)
                session.current_index = idx + 1
                clear_card_state()
                session.save_session()

    @reactive.Effect
    @reactive.event(input.btn_prev)
    def _():
        session = session_state.get()
        idx = current_card_index.get()
        if idx > 0:
            current_card_index.set(idx - 1)
            clear_card_state()
            if session:
                session.current_index = idx - 1
                session.save_session()

    @reactive.Effect
    @reactive.event(input.btn_jump)
    def _():
        session = session_state.get()
        if session and input.jump_to():
            target = input.jump_to() - 1  # Convert to 0-indexed
            if 0 <= target < len(session.cards):
                current_card_index.set(target)
                clear_card_state()
                session.current_index = target
                session.save_session()

    # Codex-powered update of current card
    @reactive.Effect
    @reactive.event(input.btn_codex_update)
    def _():
        session = session_state.get()
        card = get_current_card()

        if not session or not card:
            codex_msg.set("✗ No card loaded to update.")
            return

        context = session.get_context(card)
        user_instructions = input.codex_instructions() or ""
        codex_msg.set("⏳ Asking Codex (gpt-5.1, low) to refine this card...")

        try:
            prompt = build_codex_update_prompt(card, context, user_instructions.strip() or None)

            # Minimal args namespace for run_codex_exec
            args_ns = argparse.Namespace(
                codex_model=None,
                model_reasoning_effort=None,
                codex_extra_arg=[],
            )

            run_dir = session.run_dir if isinstance(session.run_dir, Path) else Path(session.run_dir)

            response_text = run_codex_exec(
                prompt,
                chunk_idx=0,
                run_dir=run_dir,
                args=args_ns,
                model_override="gpt-5.1",
                reasoning_override="low",
                label="_edit",
            )

            response_json = parse_codex_response_robust(
                response_text,
                chunk_idx=0,
                run_dir=run_dir,
                verbose=False,
                label="_edit",
            )

            if not response_json or "updated_card" not in response_json:
                codex_msg.set("⚠ Codex did not return a valid `updated_card` payload.")
                return

            updated = response_json.get("updated_card") or {}

            new_front = updated.get("front", card.get("front", ""))
            new_back = updated.get("back", card.get("back", ""))
            raw_tags = updated.get("tags", card.get("tags", []))

            if isinstance(raw_tags, str):
                tags_list = [t.strip() for t in raw_tags.split(",") if t.strip()]
            elif isinstance(raw_tags, list):
                tags_list = [str(t).strip() for t in raw_tags if str(t).strip()]
            else:
                tags_list = card.get("tags", []) or []

            # Store pending edit and show edit panel
            pending_edit.set({
                'front': new_front,
                'back': new_back,
                'tags': tags_list,
            })
            edit_mode_active.set(True)  # Show edit panel with Codex suggestions

            codex_msg.set("✓ Codex suggestion applied. Review and click 'Save Changes' to accept.")

        except FileNotFoundError:
            codex_msg.set("✗ `codex` CLI not found. Ensure codex-cli is installed and on your PATH.")
        except Exception as e:
            codex_msg.set(f"✗ Codex error: {str(e)[:80]}")

    # Edit functionality
    @reactive.Effect
    @reactive.event(input.btn_edit)
    def _():
        pending_edit.set(None)  # Clear any Codex suggestions, use original card values
        edit_mode_active.set(True)  # Show the edit panel

    @reactive.Effect
    @reactive.event(input.btn_save_edit)
    def _():
        session = session_state.get()
        card = get_current_card()
        if session and card:
            idx = current_card_index.get()

            edited_card = card.copy()
            edited_card['front'] = input.edit_front()
            edited_card['back'] = input.edit_back()
            edited_card['tags'] = [t.strip() for t in input.edit_tags().split(',') if t.strip()]

            session.record_decision(idx, 'edit', edited_card=edited_card)

            # Close edit panel and move to next card
            pending_edit.set(None)  # Clear pending edit
            if idx < len(session.cards) - 1:
                current_card_index.set(idx + 1)
                session.current_index = idx + 1
                clear_card_state()
            else:
                edit_mode_active.set(False)  # Just close edit panel if at last card
            session.save_session()
            stats_trigger.set(stats_trigger.get() + 1)  # Trigger stats update

    # Filter functionality
    filter_msg = reactive.Value("")

    @output
    @render.text
    def filter_message():
        return filter_msg.get()

    @reactive.Effect
    @reactive.event(input.btn_apply_filters)
    def _():
        session = session_state.get()
        if session:
            deck = input.filter_deck()
            min_conf = input.filter_confidence()
            dup_filter = input.filter_duplicate()
            session.apply_filters(
                deck_filter=deck,
                min_confidence=min_conf,
                duplicate_filter=dup_filter,
            )

            # Reset to first filtered card
            if session.filtered_indices:
                current_card_index.set(session.filtered_indices[0])
                session.current_index = session.filtered_indices[0]
                filter_msg.set(f"✓ Showing {len(session.filtered_indices)} of {len(session.cards)} cards")
            else:
                filter_msg.set("⚠ No cards match filters")
            session.save_session()

    # Bulk actions
    bulk_msg = reactive.Value("")

    @output
    @render.text
    def bulk_message():
        return bulk_msg.get()

    @reactive.Effect
    @reactive.event(input.btn_bulk_accept)
    def _():
        session = session_state.get()
        if session:
            threshold = input.bulk_confidence_threshold()
            count = session.bulk_accept_high_confidence(threshold=threshold)
            bulk_msg.set(f"✓ Auto-accepted {count} cards (confidence ≥ {threshold:.0%})")
            session.save_session()
            stats_trigger.set(stats_trigger.get() + 1)  # Trigger stats update

    # Export functionality
    export_msg = reactive.Value("")

    @output
    @render.text
    def export_message():
        return export_msg.get()

    @reactive.Effect
    @reactive.event(input.btn_export)
    def _():
        session = session_state.get()
        if session:
            output_dir = session.run_dir
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_file = output_dir / f"accepted_cards_{timestamp}.json"

            count = session.export_accepted(output_file)
            export_msg.set(f"✓ Exported {count} cards to {output_file.name}")
        else:
            export_msg.set("✗ No session loaded")

    @reactive.Effect
    @reactive.event(input.btn_export_feedback)
    def _():
        session = session_state.get()
        if session:
            output_dir = session.run_dir
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_file = output_dir / f"feedback_data_{timestamp}.json"

            count = session.export_feedback(output_file)
            export_msg.set(f"✓ Exported {count} feedback records to {output_file.name}")
        else:
            export_msg.set("✗ No session loaded")

    # Archive functionality
    archive_msg = reactive.Value("")
    run_list_trigger = reactive.Value(0)  # Trigger to refresh run dropdowns

    @output
    @render.text
    def archive_message():
        return archive_msg.get()

    def refresh_run_lists():
        """Refresh both active and archived run dropdowns."""
        # Refresh active runs
        runs = list_runs_by_date(
            reverse=input.run_order_desc(),
            order_by_card_median=input.run_order_by_card_median(),
            include_empty=input.show_empty_runs(),
        )
        choices = {}
        for run_path, is_complete in runs:
            is_empty = _is_run_empty(run_path)
            label = run_path.name
            label += _format_run_card_date_label(run_path)
            if is_complete and not is_empty:
                label += " [DONE]"
            choices[str(run_path)] = label
        if not choices:
            choices = {"": "No runs found"}
        ui.update_select("run_select", choices=choices)

        # Refresh archived runs
        archived = find_archived_runs(
            reverse=input.run_order_desc(),
            order_by_card_median=input.run_order_by_card_median(),
            include_empty=input.show_empty_runs(),
        )
        archived_choices = {"": "-- Select archived run --"}
        for run_path, is_complete in archived:
            is_empty = _is_run_empty(run_path)
            label = run_path.name
            label += _format_run_card_date_label(run_path)
            if is_complete and not is_empty:
                label += " [DONE]"
            archived_choices[str(run_path)] = label
        ui.update_select("archived_run_select", choices=archived_choices)

    @reactive.Effect
    @reactive.event(input.btn_archive_empty_runs)
    def _():
        """Archive all active runs that produced zero proposed cards."""
        base_dir = Path("auto_anki_runs")
        if not base_dir.exists():
            empty_runs_msg.set("⚠ auto_anki_runs/ not found")
            return

        run_dirs = [
            d
            for d in base_dir.iterdir()
            if d.is_dir() and d.name.startswith("run-") and d.name != "archived"
        ]

        archived_count = 0
        failed_count = 0
        archived_current = False
        current_session = session_state.get()
        current_run = current_session.run_dir if current_session else None

        for run_dir in run_dirs:
            if not _is_run_empty(run_dir):
                continue
            try:
                archive_run(run_dir)
                archived_count += 1
                if current_run and run_dir == current_run:
                    archived_current = True
            except Exception:
                failed_count += 1

        if archived_current:
            session_state.set(None)
            current_card_index.set(0)
            ui.update_select("run_select", selected="")

        refresh_run_lists()

        if archived_count == 0 and failed_count == 0:
            empty_runs_msg.set("✓ No empty runs to archive")
        elif failed_count == 0:
            empty_runs_msg.set(f"✓ Archived {archived_count} empty runs")
        else:
            empty_runs_msg.set(f"⚠ Archived {archived_count} empty runs; {failed_count} failed")

    @reactive.Effect
    @reactive.event(input.btn_archive_run)
    def _():
        """Archive the current run."""
        session = session_state.get()
        if not session:
            archive_msg.set("✗ No session loaded")
            return

        # Check if run is complete
        stats = session.get_stats()
        if stats['remaining'] > 0:
            archive_msg.set(f"⚠ Run not complete ({stats['remaining']} cards remaining)")
            return

        pending_import = session.get_pending_anki_import_indices()
        if pending_import:
            archive_msg.set(
                f"⚠ Cannot archive: {len(pending_import)} accepted card(s) pending Anki import. "
                "Use 'Import All Accepted to Anki' (or import individually) first."
            )
            return

        try:
            run_dir = session.run_dir
            archived_path = archive_run(run_dir)
            archive_msg.set(f"✓ Archived to {archived_path.name}")

            # Clear current session and refresh lists
            session_state.set(None)
            current_card_index.set(0)
            refresh_run_lists()
            ui.update_select("run_select", selected="")

        except Exception as e:
            archive_msg.set(f"✗ Archive failed: {str(e)[:50]}")

    @reactive.Effect
    @reactive.event(input.btn_load_archived)
    def _():
        """Load an archived run for viewing."""
        archived_path = input.archived_run_select()
        if not archived_path:
            archive_msg.set("⚠ Select an archived run first")
            return

        run_path = Path(archived_path)
        if run_path.exists():
            session_state.set(CardReviewSession(run_path))
            session = session_state.get()
            if session:
                current_card_index.set(session.current_index)
                # Update deck filter options
                decks = list(set(card.get('deck', 'Unknown') for card in session.cards))
                deck_choices = {"all": "All Decks"}
                deck_choices.update({d: d for d in sorted(decks)})
                ui.update_select("filter_deck", choices=deck_choices)
                archive_msg.set(f"✓ Loaded archived run: {run_path.name}")
        else:
            archive_msg.set("✗ Archived run not found")

    @reactive.Effect
    @reactive.event(input.btn_unarchive)
    def _():
        """Move an archived run back to active runs."""
        archived_path = input.archived_run_select()
        if not archived_path:
            archive_msg.set("⚠ Select an archived run first")
            return

        try:
            run_path = Path(archived_path)
            if not run_path.exists():
                archive_msg.set("✗ Archived run not found")
                return

            # Move back to active runs folder
            active_dir = run_path.parent.parent  # Go from archived/ to auto_anki_runs/
            dest = active_dir / run_path.name
            shutil.move(str(run_path), str(dest))

            archive_msg.set(f"✓ Unarchived: {run_path.name}")
            refresh_run_lists()
            ui.update_select("archived_run_select", selected="")

        except Exception as e:
            archive_msg.set(f"✗ Unarchive failed: {str(e)[:50]}")

    # AnkiConnect functionality
    anki_msg = reactive.Value("")

    @output
    @render.ui
    def anki_status():
        """Display AnkiConnect connection status."""
        if not ANKI_CONNECT_AVAILABLE:
            return ui.div(
                {"style": "color: orange; font-size: 0.9em; margin-bottom: 10px;"},
                "⚠ AnkiConnect module not available"
            )

        if not anki_client:
            return ui.div(
                {"style": "color: red; font-size: 0.9em; margin-bottom: 10px;"},
                "✗ AnkiConnect not initialized"
            )

        try:
            if anki_client.check_connection():
                version = anki_client.get_version()
                return ui.div(
                    {"style": "color: green; font-size: 0.9em; margin-bottom: 10px;"},
                    f"✓ Connected (v{version})"
                )
            else:
                return ui.div(
                    {"style": "color: red; font-size: 0.9em; margin-bottom: 10px;"},
                    "✗ Anki not running"
                )
        except:
            return ui.div(
                {"style": "color: red; font-size: 0.9em; margin-bottom: 10px;"},
                "✗ Cannot connect to Anki"
            )

    @output
    @render.text
    def anki_message():
        return anki_msg.get()

    @reactive.Effect
    @reactive.event(input.btn_import_current)
    def _():
        """Import current card to Anki."""
        if not anki_client:
            anki_msg.set("✗ AnkiConnect not available")
            return

        card = get_current_card()
        if not card:
            anki_msg.set("✗ No card to import")
            return

        try:
            # Check connection
            if not anki_client.check_connection():
                anki_msg.set("✗ Anki is not running. Please start Anki.")
                return

            # Import card
            note_id = anki_client.import_card(
                card,
                allow_duplicate=input.allow_duplicates()
            )

            idx = current_card_index.get()
            session = session_state.get()

            if note_id:
                anki_msg.set(f"✓ Imported to {card.get('deck', 'Default')} (ID: {note_id})")
                if session:
                    session.mark_anki_import_status(idx, status="imported", note_id=note_id)
                    session.save_session()
                    stats_trigger.set(stats_trigger.get() + 1)  # Trigger stats update
            else:
                anki_msg.set("⚠ Card already exists in Anki (duplicate)")
                if session:
                    session.mark_anki_import_status(idx, status="duplicate", note_id=None)
                    session.save_session()
                    stats_trigger.set(stats_trigger.get() + 1)  # Trigger stats update

        except ConnectionError as e:
            anki_msg.set(f"✗ Connection error: {str(e)[:50]}")
        except AnkiConnectError as e:
            anki_msg.set(f"✗ Anki error: {str(e)[:50]}")
        except Exception as e:
            anki_msg.set(f"✗ Error: {str(e)[:50]}")

    @reactive.Effect
    @reactive.event(input.btn_import_accepted)
    def _():
        """Import all accepted cards to Anki."""
        if not anki_client:
            anki_msg.set("✗ AnkiConnect not available")
            return

        session = session_state.get()
        if not session:
            anki_msg.set("✗ No session loaded")
            return

        # Collect accepted cards
        accepted_cards = []
        accepted_indices = []
        for idx, card in enumerate(session.cards):
            decision = session.decisions.get(idx)
            if decision and decision['action'] in ['accept', 'edit']:
                # Use edited card if available, otherwise use original
                edited = decision.get('edited_card')
                card_to_import = edited if edited else card
                accepted_cards.append(card_to_import)
                accepted_indices.append(idx)

        if not accepted_cards:
            anki_msg.set("⚠ No accepted cards to import")
            return

        try:
            # Check connection
            if not anki_client.check_connection():
                anki_msg.set("✗ Anki is not running. Please start Anki.")
                return

            # Batch import
            result = anki_client.import_cards_batch(
                accepted_cards,
                allow_duplicates=input.allow_duplicates(),
                create_decks=True
            )

            # Display results
            msg = f"✓ Imported {result['imported']} cards"
            if result['duplicates'] > 0:
                msg += f", {result['duplicates']} duplicates skipped"
            if result['failed'] > 0:
                msg += f", {result['failed']} failed"

            anki_msg.set(msg)
            if result.get("failed", 0) == 0 and accepted_indices:
                for idx in accepted_indices:
                    session.mark_anki_import_status(idx, status="imported_or_duplicate", note_id=None)
                session.save_session()
                stats_trigger.set(stats_trigger.get() + 1)  # Trigger stats update

        except ConnectionError as e:
            anki_msg.set(f"✗ Connection error: {str(e)[:50]}")
        except AnkiConnectError as e:
            anki_msg.set(f"✗ Anki error: {str(e)[:50]}")
        except Exception as e:
            anki_msg.set(f"✗ Error: {str(e)[:50]}")


# ============================================================================
# App Definition
# ============================================================================

app = App(app_ui, server)


if __name__ == "__main__":
    print("🎓 Auto Anki Card Review UI")
    print("=" * 60)
    print("Starting Shiny app...")
    print("Open your browser to the URL shown below.")
    print("=" * 60)

"""
Chat context structures and harvesting logic.

This module owns:
- `ChatTurn` dataclass
- Heuristic scoring (`detect_signals`)
- Term/point extraction (`extract_key_terms`, `extract_key_points`)
- Markdown chat parsing and harvesting (`harvest_chat_contexts`)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


QUESTION_WORDS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "explain",
    "describe",
    "define",
    "list",
    "compare",
    "give",
    "teach",
}

STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "from",
    "this",
    "have",
    "into",
    "your",
    "about",
    "they",
    "their",
    "them",
    "over",
    "each",
    "such",
    "also",
    "been",
    "than",
    "just",
    "will",
    "only",
    "much",
    "more",
    "into",
    "used",
    "very",
    "make",
    "made",
    "even",
    "most",
    "like",
    "some",
    "what",
    "when",
    "where",
    "how",
    "why",
}

ENTRY_RE = re.compile(
    r"^\[(?P<timestamp>[^\]]+)\]\s+(?P<role>user|human|assistant|tool):\s*$", re.IGNORECASE
)

ASSISTANT_START_PLACEHOLDER = "Assistant started conversation (no user prompt in export)"

logger = logging.getLogger(__name__)


@dataclass
class ChatTurn:
    context_id: str
    turn_index: int                    # Position in parent conversation (0-indexed)
    conversation_id: str               # Link to parent Conversation
    source_path: str
    source_title: Optional[str]
    source_url: Optional[str]
    user_timestamp: Optional[str]
    user_prompt: str
    assistant_answer: str
    assistant_char_count: int
    score: float
    signals: Dict[str, Any]
    key_terms: List[str]
    key_points: List[str]


@dataclass
class Conversation:
    """Represents a full ChatGPT conversation containing multiple turns."""
    conversation_id: str               # SHA256(source_path + first_timestamp)
    source_path: str
    source_title: Optional[str]
    source_url: Optional[str]
    turns: List[ChatTurn]              # Ordered list of turns
    total_char_count: int              # Sum of all user prompts + assistant responses
    aggregate_score: float             # Combined score across turns
    aggregate_signals: Dict[str, Any]  # Merged signals
    key_topics: List[str]              # Extracted from all turns


class DateRangeFilter:
    """Parse and apply date range filters to conversation file paths."""

    def __init__(self, date_range_str: Optional[str]):
        self.start_date: Optional[str] = None
        self.end_date: Optional[str] = None

        if not date_range_str:
            return

        # Support formats: "2025-10", "2025-10-01:2025-10-31"
        if ":" in date_range_str:
            parts = date_range_str.split(":", 1)
            self.start_date = parts[0].strip()
            self.end_date = parts[1].strip()
        else:
            # Single month format like "2025-10"
            if re.match(r"^\d{4}-\d{2}$", date_range_str):
                self.start_date = f"{date_range_str}-01"
                # Approximate end of month
                year, month = map(int, date_range_str.split("-"))
                if month == 12:
                    self.end_date = f"{year+1}-01-01"
                else:
                    self.end_date = f"{year}-{month+1:02d}-01"
            else:
                self.start_date = date_range_str

    def matches(self, path: Path) -> bool:
        """Check if a conversation file path matches the date range."""
        if not self.start_date:
            return True

        # Extract date from filename like "2025-10-01_topic.md"
        match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
        if not match:
            return True  # If no date found, include it

        file_date = match.group(1)

        if self.start_date and file_date < self.start_date:
            return False
        if self.end_date and file_date >= self.end_date:
            return False

        return True


def matches_exclusion_pattern(path: Path, patterns: List[str]) -> bool:
    """Check if file matches any exclusion pattern (glob-style).

    Args:
        path: Path to the file
        patterns: List of glob patterns (e.g., ['*_chat-*.md', 'draft-*'])

    Returns:
        True if the file matches any pattern and should be excluded
    """
    import fnmatch

    return any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)


def parse_chat_metadata(header_block: str) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    title_match = re.search(r"^#\s+(?P<title>.+)$", header_block, re.MULTILINE)
    if title_match:
        metadata["title"] = title_match.group("title").strip()
    for line in header_block.splitlines():
        if line.startswith("- "):
            parts = line[2:].split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip().lower()
                value = parts[1].strip()
                metadata[key] = value
    return metadata


def parse_chat_entries(body: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line in body.splitlines():
        match = ENTRY_RE.match(line)
        if match:
            if current is not None:
                current["text"] = "\n".join(current["lines"]).strip()
                entries.append(current)
            role = match.group("role").lower()
            # Normalize Claude's "human" role to "user" for consistency
            if role == "human":
                role = "user"
            current = {
                "role": role,
                "timestamp": match.group("timestamp").strip(),
                "lines": [],
            }
            continue
        if current is not None:
            current["lines"].append(line)
    if current is not None:
        current["text"] = "\n".join(current["lines"]).strip()
        entries.append(current)
    return entries


def extract_turns(
    entries: Sequence[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    def is_non_content_assistant_message(text: str) -> bool:
        """Heuristic: ignore assistant/tooling chatter and internal metadata.

        Many exports include intermediate assistant messages like:
        - {"content_type":"thoughts", ...}
        - {"content_type":"code", "text":"{...search_query...}"}
        - {"content_type":"reasoning_recap", ...}
        """
        stripped = text.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return False
        try:
            payload = json.loads(stripped)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        content_type = str(payload.get("content_type") or "").lower()
        if content_type in {"thoughts", "reasoning_recap", "code"}:
            return True
        return False

    def combine_assistant_entries(
        assistant_entries: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not assistant_entries:
            return None

        substantive = [
            e.get("text", "").strip()
            for e in assistant_entries
            if e.get("text", "").strip() and not is_non_content_assistant_message(e["text"])
        ]
        if substantive:
            combined_text = "\n\n".join(substantive).strip()
        else:
            combined_text = (assistant_entries[-1].get("text") or "").strip()

        if not combined_text:
            return None

        return {
            "role": "assistant",
            "timestamp": assistant_entries[-1].get("timestamp"),
            "text": combined_text,
        }

    turns: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    pending_user: Optional[Dict[str, Any]] = None
    pending_assistants: List[Dict[str, Any]] = []
    leading_assistants: List[Dict[str, Any]] = []
    seen_user_entry = False

    def flush_pending_turn() -> None:
        nonlocal pending_user, pending_assistants
        if pending_user is None:
            pending_assistants = []
            return
        assistant_entry = combine_assistant_entries(pending_assistants)
        if assistant_entry and pending_user.get("text", "").strip():
            turns.append((pending_user, assistant_entry))
        pending_user = None
        pending_assistants = []

    def flush_leading_assistants() -> None:
        """Emit a synthetic turn for assistant messages that precede any user prompt."""
        nonlocal leading_assistants
        assistant_entry = combine_assistant_entries(leading_assistants)
        if assistant_entry:
            synthetic_user = {
                "role": "user",
                "timestamp": assistant_entry.get("timestamp"),
                "text": ASSISTANT_START_PLACEHOLDER,
            }
            turns.append((synthetic_user, assistant_entry))
        leading_assistants = []

    for entry in entries:
        role = (entry.get("role") or "").lower()
        text = (entry.get("text") or "").strip()

        if role == "user":
            seen_user_entry = True
            if leading_assistants:
                flush_leading_assistants()
            if pending_user is None:
                if not text:
                    continue
                pending_user = dict(entry)
                pending_user["text"] = text
                pending_assistants = []
                continue

            # If we've already started collecting assistant responses, this is a new turn.
            if pending_assistants:
                flush_pending_turn()
                if not text:
                    continue
                pending_user = dict(entry)
                pending_user["text"] = text
                continue

            # Consecutive user messages before any assistant response: merge them.
            if text:
                prior_text = (pending_user.get("text") or "").strip()
                pending_user["text"] = (prior_text + "\n\n" + text).strip()
            continue

        if role == "assistant":
            if pending_user is None:
                if seen_user_entry:
                    continue
                if not text:
                    continue
                leading_assistants.append(dict(entry, text=text))
                continue
            if not text:
                continue
            pending_assistants.append(dict(entry, text=text))
            continue

        # tool (and any other roles) do not affect pairing; they can occur between
        # assistant messages when exports include tool I/O.
        continue

    flush_pending_turn()
    if leading_assistants:
        flush_leading_assistants()
    return turns


def detect_signals(user_text: str, assistant_text: str) -> Tuple[float, Dict[str, Any]]:
    user_lower = user_text.lower()
    assistant_lower = assistant_text.lower()
    signals: Dict[str, Any] = {}
    score = 0.0

    question_like = "?" in user_text or any(
        user_lower.startswith(word + " ") for word in QUESTION_WORDS
    )
    signals["question_like"] = question_like
    if question_like:
        score += 1.0

    definition_like = any(
        phrase in assistant_lower
        for phrase in [
            "stands for",
            "is defined as",
            "refers to",
            "meaning:",
            "means that",
            "definition",
        ]
    )
    signals["definition_like"] = definition_like
    if definition_like:
        score += 0.45

    bullet_count = sum(
        1
        for line in assistant_text.splitlines()
        if line.strip().startswith(("-", "*", "1.", "2.", "3."))
    )
    signals["bullet_count"] = bullet_count
    if bullet_count:
        score += min(0.35 + 0.03 * bullet_count, 0.65)

    heading_count = assistant_text.count("### ")
    signals["heading_count"] = heading_count
    if heading_count:
        score += min(0.25 + 0.05 * heading_count, 0.45)

    code_blocks = assistant_text.count("```")
    signals["code_blocks"] = code_blocks
    if code_blocks:
        score += 0.2

    answer_len = len(assistant_text)
    signals["answer_length"] = answer_len
    if 80 <= answer_len <= 2200:
        score += 0.55
    elif answer_len > 2200:
        score += 0.2

    imperative = any(
        phrase in user_lower
        for phrase in ["walk me through", "show me", "give me", "steps"]
    )
    signals["imperative"] = imperative
    if imperative:
        score += 0.2

    return score, signals


def extract_key_terms(text: str, limit: int = 6) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\\-]{3,}", text.lower())
    counts = Counter(word for word in words if word not in STOPWORDS)
    return [word for word, _ in counts.most_common(limit)]


def extract_key_points(text: str, limit: int = 4) -> List[str]:
    points: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("#", "-", "*")):
            cleaned = stripped.lstrip("#-*0123456789. ").strip()
            if cleaned:
                points.append(cleaned)
        if len(points) >= limit:
            break
    return points


def harvest_chat_contexts(
    chat_root: Path,
    seen_ids: set[str],
    state_tracker: Optional[Any],
    date_filter: Optional[DateRangeFilter],
    args: argparse.Namespace,
) -> List[ChatTurn]:
    contexts: List[ChatTurn] = []
    files = sorted(
        (p for p in chat_root.rglob("*.md")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # Apply filters
    if date_filter:
        files = [f for f in files if date_filter.matches(f)]
    if args.unprocessed_only and state_tracker:
        files = [f for f in files if not state_tracker.is_file_processed(f)]

    if args.max_chat_files:
        files = files[: args.max_chat_files]
    per_file_counter: Dict[Path, int] = defaultdict(int)

    for path in files:
        if per_file_counter[path] >= args.per_file_context_limit:
            continue
        text = path.read_text()
        if "\n---" in text:
            header, body = text.split("\n---", 1)
        else:
            header, body = text, ""
        metadata = parse_chat_metadata(header)
        entries = parse_chat_entries(body)
        turns = extract_turns(entries)
        for user_entry, assistant_entry in turns:
            combined_key = f"{path}:{user_entry.get('timestamp','')}"
            context_id = sha256(
                (combined_key + user_entry["text"] + assistant_entry["text"]).encode(
                    "utf-8"
                )
            ).hexdigest()
            if context_id in seen_ids:
                continue
            score, signals = detect_signals(
                user_entry["text"], assistant_entry["text"]
            )
            if score < args.min_score:
                continue
            if per_file_counter[path] >= args.per_file_context_limit:
                break
            assistant_full = assistant_entry["text"].strip()
            assistant_trimmed = assistant_full[: args.assistant_char_limit]
            key_terms = extract_key_terms(
                user_entry["text"] + " " + assistant_full
            )
            key_points = extract_key_points(assistant_full)
            context = ChatTurn(
                context_id=context_id,
                source_path=str(path),
                source_title=metadata.get("title"),
                source_url=metadata.get("url"),
                user_timestamp=user_entry.get("timestamp"),
                user_prompt=user_entry["text"].strip(),
                assistant_answer=assistant_trimmed,
                assistant_char_count=len(assistant_full),
                score=score,
                signals=signals,
                key_terms=key_terms,
                key_points=key_points,
            )
            contexts.append(context)
            per_file_counter[path] += 1
            if len(contexts) >= args.max_contexts:
                return contexts
    return contexts


def detect_conversation_signals(turns: List[ChatTurn]) -> Tuple[float, Dict[str, Any]]:
    """Compute aggregate signals across an entire conversation."""
    if not turns:
        return 0.0, {}

    # Aggregate individual turn scores
    total_score = sum(t.score for t in turns)
    avg_score = total_score / len(turns)

    # Detect mega-pastes (users pasting entire HTML docs, code files, etc.)
    # These typically indicate "explain this doc" requests with limited flashcard value
    MEGA_PASTE_THRESHOLD = 20000  # 20KB+ is likely a pasted document
    max_user_prompt_len = max(len(t.user_prompt) for t in turns) if turns else 0
    has_mega_paste = max_user_prompt_len > MEGA_PASTE_THRESHOLD

    # Merge signals
    aggregate_signals: Dict[str, Any] = {
        "turn_count": len(turns),
        "total_score": round(total_score, 3),
        "avg_score": round(avg_score, 3),
        "question_turns": sum(1 for t in turns if t.signals.get("question_like")),
        "definition_turns": sum(1 for t in turns if t.signals.get("definition_like")),
        "code_turns": sum(1 for t in turns if t.signals.get("code_blocks", 0) > 0),
        "total_bullets": sum(t.signals.get("bullet_count", 0) for t in turns),
        "total_headings": sum(t.signals.get("heading_count", 0) for t in turns),
        "has_mega_paste": has_mega_paste,
        "max_user_prompt_chars": max_user_prompt_len,
    }

    # Detect follow-up patterns (indicates user struggled/needed clarification)
    followup_patterns = [
        "can you explain",
        "what do you mean",
        "i don't understand",
        "could you clarify",
        "wait,",
        "actually,",
        "so you're saying",
        "let me make sure",
    ]
    followup_count = 0
    for turn in turns[1:]:  # Skip first turn
        user_lower = turn.user_prompt.lower()
        if any(pattern in user_lower for pattern in followup_patterns):
            followup_count += 1
    aggregate_signals["followup_turns"] = followup_count

    # Detect correction patterns in assistant responses
    correction_patterns = [
        "actually,",
        "i should clarify",
        "to correct",
        "i was wrong",
        "more accurately",
        "let me rephrase",
    ]
    correction_count = 0
    for turn in turns[1:]:
        assistant_lower = turn.assistant_answer.lower()
        if any(pattern in assistant_lower for pattern in correction_patterns):
            correction_count += 1
    aggregate_signals["correction_turns"] = correction_count

    # Boost score for conversations with good learning arc
    # (questions followed by clarifications shows engagement)
    if len(turns) >= 2 and aggregate_signals["question_turns"] > 0:
        total_score += 0.5  # Bonus for multi-turn engagement

    return total_score, aggregate_signals


def build_conversation(
    path: Path,
    metadata: Dict[str, str],
    raw_turns: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    args: argparse.Namespace,
) -> Optional[Conversation]:
    """Build a Conversation object from parsed turns."""
    if not raw_turns:
        return None

    # Generate conversation_id from path + first timestamp
    first_timestamp = raw_turns[0][0].get("timestamp", "")
    conversation_id = sha256(
        f"{path}:{first_timestamp}".encode("utf-8")
    ).hexdigest()

    # Build ChatTurn objects
    chat_turns: List[ChatTurn] = []
    use_heuristics = getattr(args, "use_filter_heuristics", False)

    for turn_index, (user_entry, assistant_entry) in enumerate(raw_turns):
        combined_key = f"{path}:{user_entry.get('timestamp', '')}"
        context_id = sha256(
            (combined_key + user_entry["text"] + assistant_entry["text"]).encode("utf-8")
        ).hexdigest()

        if use_heuristics:
            score, signals = detect_signals(user_entry["text"], assistant_entry["text"])
        else:
            # Skip heuristics - assign neutral score, let Stage 1 LLM decide quality
            score = 1.0
            signals = {"heuristics_skipped": True}

        assistant_full = assistant_entry["text"].strip()
        assistant_trimmed = assistant_full[: args.assistant_char_limit]

        key_terms = extract_key_terms(user_entry["text"] + " " + assistant_full)
        key_points = extract_key_points(assistant_full)

        turn = ChatTurn(
            context_id=context_id,
            turn_index=turn_index,
            conversation_id=conversation_id,
            source_path=str(path),
            source_title=metadata.get("title"),
            source_url=metadata.get("url"),
            user_timestamp=user_entry.get("timestamp"),
            user_prompt=user_entry["text"].strip(),
            assistant_answer=assistant_trimmed,
            assistant_char_count=len(assistant_full),
            score=score,
            signals=signals,
            key_terms=key_terms,
            key_points=key_points,
        )
        chat_turns.append(turn)

    if not chat_turns:
        return None

    # Compute aggregate signals
    aggregate_score, aggregate_signals = detect_conversation_signals(chat_turns)

    # Extract key topics from all turns
    all_terms: List[str] = []
    for turn in chat_turns:
        all_terms.extend(turn.key_terms)
    key_topics = extract_key_terms(" ".join(all_terms), limit=8)

    # Compute total character count (user prompts + assistant responses)
    # Including user prompts catches mega-pastes (users pasting entire HTML docs, etc.)
    total_char_count = sum(
        len(t.user_prompt) + t.assistant_char_count for t in chat_turns
    )

    return Conversation(
        conversation_id=conversation_id,
        source_path=str(path),
        source_title=metadata.get("title"),
        source_url=metadata.get("url"),
        turns=chat_turns,
        total_char_count=total_char_count,
        aggregate_score=aggregate_score,
        aggregate_signals=aggregate_signals,
        key_topics=key_topics,
    )


def split_conversation_by_topic(
    conv: Conversation,
    max_turns: int = 10,
    max_chars: int = 8000,
) -> List[Conversation]:
    """Split long conversations at natural topic boundaries.

    Splitting heuristics:
    1. User questions that change subject (low term overlap with previous)
    2. Time gaps between turns (if timestamps available)
    3. Explicit topic markers ("Now, about X...", "Moving on to...")

    Each sub-conversation gets:
    - Same source_path, source_title, source_url
    - New conversation_id: f"{original_id}_part{N}"
    - Subset of turns with re-indexed turn_index
    """
    # If conversation is within limits, return as-is
    if len(conv.turns) <= max_turns and conv.total_char_count <= max_chars:
        return [conv]

    # Find split points based on topic shifts
    split_indices: List[int] = [0]  # Always start with first turn

    topic_shift_patterns = [
        "moving on",
        "now, about",
        "changing topic",
        "different question",
        "unrelated,",
        "on another note",
        "separately,",
        "new question",
    ]

    for i in range(1, len(conv.turns)):
        current_turn = conv.turns[i]
        prev_turn = conv.turns[i - 1]

        # Check for explicit topic shift markers
        user_lower = current_turn.user_prompt.lower()
        has_topic_marker = any(pattern in user_lower for pattern in topic_shift_patterns)

        # Check for low term overlap (topic change)
        prev_terms = set(prev_turn.key_terms)
        curr_terms = set(current_turn.key_terms)
        if prev_terms and curr_terms:
            overlap = len(prev_terms & curr_terms) / max(len(prev_terms), len(curr_terms))
            low_overlap = overlap < 0.2
        else:
            low_overlap = False

        # Mark as split point if topic shift detected
        if has_topic_marker or low_overlap:
            # Ensure we don't create tiny splits
            if i - split_indices[-1] >= 2:
                split_indices.append(i)

    # Add end index
    split_indices.append(len(conv.turns))

    # If no good split points found, split evenly
    if len(split_indices) <= 2 and len(conv.turns) > max_turns:
        split_indices = list(range(0, len(conv.turns), max_turns))
        if split_indices[-1] != len(conv.turns):
            split_indices.append(len(conv.turns))

    # Build sub-conversations
    sub_conversations: List[Conversation] = []
    for part_num, (start, end) in enumerate(zip(split_indices[:-1], split_indices[1:])):
        sub_turns = conv.turns[start:end]
        if not sub_turns:
            continue

        # Re-index turns and update conversation_id
        new_conv_id = f"{conv.conversation_id}_part{part_num + 1}"
        reindexed_turns: List[ChatTurn] = []
        for new_idx, turn in enumerate(sub_turns):
            reindexed_turn = ChatTurn(
                context_id=turn.context_id,
                turn_index=new_idx,
                conversation_id=new_conv_id,
                source_path=turn.source_path,
                source_title=turn.source_title,
                source_url=turn.source_url,
                user_timestamp=turn.user_timestamp,
                user_prompt=turn.user_prompt,
                assistant_answer=turn.assistant_answer,
                assistant_char_count=turn.assistant_char_count,
                score=turn.score,
                signals=turn.signals,
                key_terms=turn.key_terms,
                key_points=turn.key_points,
            )
            reindexed_turns.append(reindexed_turn)

        # Recompute aggregates for sub-conversation
        agg_score, agg_signals = detect_conversation_signals(reindexed_turns)
        total_chars = sum(
            len(t.user_prompt) + t.assistant_char_count for t in reindexed_turns
        )

        # Extract key topics for this sub-conversation
        all_terms = []
        for t in reindexed_turns:
            all_terms.extend(t.key_terms)
        sub_topics = extract_key_terms(" ".join(all_terms), limit=8)

        sub_conv = Conversation(
            conversation_id=new_conv_id,
            source_path=conv.source_path,
            source_title=conv.source_title,
            source_url=conv.source_url,
            turns=reindexed_turns,
            total_char_count=total_chars,
            aggregate_score=agg_score,
            aggregate_signals=agg_signals,
            key_topics=sub_topics,
        )
        sub_conversations.append(sub_conv)

    return sub_conversations if sub_conversations else [conv]


def harvest_conversations(
    chat_root: Path,
    seen_conversation_ids: set[str],
    state_tracker: Optional[Any],
    date_filter: Optional[DateRangeFilter],
    args: argparse.Namespace,
) -> List[Conversation]:
    """Harvest full conversations instead of individual turns.

    Args:
        chat_root: Root directory containing chat markdown exports
        seen_conversation_ids: Set of conversation IDs already processed
        state_tracker: Optional state tracker for filtering processed files
        date_filter: Optional date range filter
        args: CLI arguments namespace

    Returns:
        List of Conversation objects ready for processing
    """
    conversations: List[Conversation] = []
    files = sorted(
        (p for p in chat_root.rglob("*.md")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # Apply filters
    if date_filter:
        files = [f for f in files if date_filter.matches(f)]

    # Apply exclusion patterns
    exclude_patterns = getattr(args, "exclude_patterns", None) or []
    if exclude_patterns:
        files = [f for f in files if not matches_exclusion_pattern(f, exclude_patterns)]

    # Handle three file selection modes:
    # 1. --only-zero-card-files: ONLY files that generated 0 cards
    # 2. --unprocessed-only --reprocess-zero-card-files: unprocessed OR zero-card
    # 3. --unprocessed-only: only unprocessed (default)
    only_zero = getattr(args, "only_zero_card_files", False)
    reprocess_zero = getattr(args, "reprocess_zero_card_files", False)

    if only_zero and state_tracker:
        # Exclusive mode: ONLY zero-card files
        files = [f for f in files if state_tracker.is_file_zero_card(f)]
    elif args.unprocessed_only and state_tracker:
        if reprocess_zero:
            # Additive mode: unprocessed OR zero-card
            files = [
                f
                for f in files
                if not state_tracker.is_file_processed(f)
                or state_tracker.is_file_zero_card(f)
            ]
        else:
            # Normal mode: only unprocessed
            files = [f for f in files if not state_tracker.is_file_processed(f)]

    if args.max_chat_files:
        files = files[: args.max_chat_files]

    # Get conversation limits from args (with defaults)
    max_turns = getattr(args, "conversation_max_turns", 10)
    max_chars = getattr(args, "conversation_max_chars", 8000)

    for path in files:
        try:
            text = path.read_text()
        except Exception as e:
            logger.warning("Skipping conversation file %s due to read error: %s", path, e)
            continue

        if "\n---" in text:
            header, body = text.split("\n---", 1)
        else:
            header, body = text, ""

        metadata = parse_chat_metadata(header)
        entries = parse_chat_entries(body)
        raw_turns = extract_turns(entries)

        if not raw_turns:
            continue

        # Build conversation
        conv = build_conversation(path, metadata, raw_turns, args)
        if conv is None:
            continue

        # Skip if already processed (unless reprocessing zero-card files)
        is_zero_card_source = (
            (only_zero or reprocess_zero)
            and state_tracker
            and state_tracker.is_file_zero_card(path)
        )
        if conv.conversation_id in seen_conversation_ids and not is_zero_card_source:
            continue

        # Filter by minimum aggregate score (only if heuristics enabled)
        use_heuristics = getattr(args, "use_filter_heuristics", False)
        if use_heuristics:
            min_score = getattr(args, "min_score", 1.2)
            if conv.aggregate_score < min_score:
                continue

        # Split long conversations
        sub_convs = split_conversation_by_topic(conv, max_turns, max_chars)

        # Add conversations (filtering already-seen sub-conversations)
        for sub_conv in sub_convs:
            if sub_conv.conversation_id not in seen_conversation_ids:
                conversations.append(sub_conv)

        # Check if we've hit the max
        max_conversations = getattr(args, "max_contexts", 24)
        if len(conversations) >= max_conversations:
            break

    return conversations


__all__ = [
    "ChatTurn",
    "Conversation",
    "DateRangeFilter",
    "matches_exclusion_pattern",
    "harvest_chat_contexts",
    "harvest_conversations",
    "detect_signals",
    "detect_conversation_signals",
    "extract_key_terms",
    "extract_key_points",
    "parse_chat_metadata",
    "parse_chat_entries",
    "extract_turns",
    "build_conversation",
    "split_conversation_by_topic",
    "QUESTION_WORDS",
    "STOPWORDS",
    "ENTRY_RE",
    "ASSISTANT_START_PLACEHOLDER",
]

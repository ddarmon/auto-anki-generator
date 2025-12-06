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
    r"^\[(?P<timestamp>[^\]]+)\]\s+(?P<role>user|assistant|tool):\s*$", re.IGNORECASE
)


@dataclass
class ChatTurn:
    context_id: str
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
            current = {
                "role": match.group("role").lower(),
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
    turns: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    i = 0
    while i < len(entries):
        entry = entries[i]
        if entry["role"] != "user":
            i += 1
            continue
        user_entry = entry
        j = i + 1
        assistant_entry: Optional[Dict[str, Any]] = None
        while j < len(entries):
            candidate = entries[j]
            if candidate["role"] == "assistant":
                assistant_entry = candidate
                break
            j += 1
        if assistant_entry and user_entry["text"] and assistant_entry["text"]:
            turns.append((user_entry, assistant_entry))
            i = j + 1
        else:
            i += 1
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


__all__ = [
    "ChatTurn",
    "DateRangeFilter",
    "harvest_chat_contexts",
    "detect_signals",
    "extract_key_terms",
    "extract_key_points",
    "parse_chat_metadata",
    "parse_chat_entries",
    "extract_turns",
    "QUESTION_WORDS",
    "STOPWORDS",
    "ENTRY_RE",
]

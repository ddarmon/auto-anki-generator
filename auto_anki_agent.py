#!/usr/bin/env python3
"""
Agentic pipeline that scans HTML-based Anki decks plus ChatGPT conversation
exports, surfaces promising contexts, and hands them to `codex exec` for the
heavy cognitive lifting (topic triage + card proposals).

Workflow summary:
1. Parse the existing decks (HTML) to understand current coverage.
2. Walk the chat transcript directory, pairing user prompts with assistant
   replies and extracting heuristics (question-ness, definition hints, etc.).
3. Filter/score those contexts, drop anything already covered by decks, and
   cap how many contexts per source file make it through.
4. Batch the remaining contexts and feed them to `codex exec` together with a
   compact view of the existing decks. Codex returns JSON containing proposed
   cards plus rationale about skipped items.
5. Persist run artifacts (prompts, contexts, codex responses) and a simple
   state file so the next run only considers new material.

The script is intentionally opinionated but flags most magic numbers via CLI
switches. Use `--dry-run` to inspect prompts/contexts before contacting Codex.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import shlex
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from difflib import SequenceMatcher
from glob import glob
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from bs4 import BeautifulSoup  # type: ignore
from json_repair import repair_json


class DateRangeFilter:
    """Parse and apply date range filters to conversation file paths."""

    def __init__(self, date_range_str: Optional[str]):
        self.start_date: Optional[str] = None
        self.end_date: Optional[str] = None

        if not date_range_str:
            return

        # Support formats: "2025-10", "2025-10-01:2025-10-31"
        if ':' in date_range_str:
            parts = date_range_str.split(':', 1)
            self.start_date = parts[0].strip()
            self.end_date = parts[1].strip()
        else:
            # Single month format like "2025-10"
            if re.match(r'^\d{4}-\d{2}$', date_range_str):
                self.start_date = f"{date_range_str}-01"
                # Approximate end of month
                year, month = map(int, date_range_str.split('-'))
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
        match = re.search(r'(\d{4}-\d{2}-\d{2})', path.name)
        if not match:
            return True  # If no date found, include it

        file_date = match.group(1)

        if self.start_date and file_date < self.start_date:
            return False
        if self.end_date and file_date >= self.end_date:
            return False

        return True


class StateTracker:
    """Track processed conversations and run history."""

    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except json.JSONDecodeError:
                raise SystemExit(f"State file {self.path} is corrupted; delete or fix it.")
        return {
            "processed_files": {},
            "seen_contexts": [],
            "last_run": None,
            "run_history": []
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

    def is_file_processed(self, file_path: Path) -> bool:
        """Check if a conversation file has been processed."""
        return str(file_path) in self.data.get("processed_files", {})

    def mark_file_processed(self, file_path: Path, cards_generated: int = 0) -> None:
        """Mark a file as processed."""
        if "processed_files" not in self.data:
            self.data["processed_files"] = {}
        self.data["processed_files"][str(file_path)] = {
            "processed_at": datetime.now().isoformat(),
            "cards_generated": cards_generated
        }

    def get_seen_context_ids(self) -> set[str]:
        """Get set of previously seen context IDs."""
        return set(self.data.get("seen_contexts", []))

    def add_context_ids(self, context_ids: List[str]) -> None:
        """Add new context IDs to the seen list."""
        seen = self.get_seen_context_ids()
        seen.update(context_ids)
        self.data["seen_contexts"] = list(seen)

    def record_run(self, run_dir: Path, contexts_sent: int) -> None:
        """Record a run in history."""
        if "run_history" not in self.data:
            self.data["run_history"] = []
        self.data["run_history"].append({
            "run_dir": str(run_dir),
            "timestamp": datetime.now().isoformat(),
            "contexts_sent": contexts_sent
        })
        self.data["last_run"] = datetime.now().isoformat()


SCRIPT_DIR = Path(__file__).resolve().parent


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
class Card:
    deck: str
    front: str
    back: str
    tags: List[str]
    meta: str
    data_search: str
    front_norm: str
    back_norm: str


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harvest chat transcripts + deck coverage and ask codex exec to propose new Anki cards."
    )
    parser.add_argument(
        "--deck-glob",
        help="Glob for HTML decks (default: <script_dir>/*.html).",
    )
    parser.add_argument(
        "--chat-root",
        default="~/Library/Mobile Documents/iCloud~md~obsidian/Documents/chatgpt",
        help="Root directory containing chat markdown exports (default: %(default)s).",
    )
    parser.add_argument(
        "--date-range",
        help="Date range filter (e.g., '2025-10' for October 2025, or '2025-10-01:2025-10-31' for specific range).",
    )
    parser.add_argument(
        "--unprocessed-only",
        action="store_true",
        help="Only process conversation files not yet in the state file.",
    )
    parser.add_argument(
        "--state-file",
        help="Path to JSON state file used to skip already-seen contexts (default: <script_dir>/.auto_anki_agent_state.json).",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "markdown", "both"],
        default="both",
        help="Output format for proposed cards (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to store prompts, contexts, and codex responses (default: <script_dir>/auto_anki_runs).",
    )
    parser.add_argument(
        "--cache-dir",
        help="Directory to cache parsed HTML decks for faster loading (default: <script_dir>/.deck_cache).",
    )
    parser.add_argument(
        "--max-existing-cards",
        type=int,
        default=120,
        help="Maximum number of existing cards to include in each codex prompt.",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=24,
        help="Maximum number of candidate contexts gathered per run before chunking.",
    )
    parser.add_argument(
        "--contexts-per-run",
        type=int,
        default=8,
        help="Number of contexts to stuff into a single codex exec call.",
    )
    parser.add_argument(
        "--per-file-context-limit",
        type=int,
        default=3,
        help="Maximum number of contexts extracted from a single chat file per run.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.82,
        help="Front/back similarity threshold vs existing cards for auto-deduping.",
    )
    parser.add_argument(
        "--dedup-method",
        choices=["string", "semantic", "hybrid"],
        default="string",
        help=(
            "Deduplication method: 'string' uses lexical similarity only, "
            "'semantic' uses embedding-based similarity, 'hybrid' uses both."
        ),
    )
    parser.add_argument(
        "--semantic-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformers model to use for semantic deduplication (default: %(default)s).",
    )
    parser.add_argument(
        "--semantic-similarity-threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for semantic deduplication (default: %(default)s).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=1.2,
        help="Minimum heuristic score a context must reach to be considered.",
    )
    parser.add_argument(
        "--max-chat-files",
        type=int,
        default=300,
        help="Cap how many chat files to scan per run (sorted newest-first).",
    )
    parser.add_argument(
        "--assistant-char-limit",
        type=int,
        default=2000,
        help="Trim assistant answers to this many characters before prompting Codex.",
    )
    parser.add_argument(
        "--excerpt-length",
        type=int,
        default=600,
        help="Length for the assistant excerpt summary included with each context.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build contexts/prompts but skip calling codex exec.",
    )
    parser.add_argument(
        "--codex-model",
        default=None,
        help="Optional model override passed to codex exec via --model (default: gpt-5).",
    )
    parser.add_argument(
        "--model-reasoning-effort",
        default="medium",
        help="Set Codex config model_reasoning_effort (default: %(default)s).",
    )
    parser.add_argument(
        "--codex-extra-arg",
        action="append",
        default=[],
        help="Repeatable passthrough arguments for codex exec (e.g. --codex-extra-arg '--sandbox=workspace-write').",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional progress information.",
    )
    args = parser.parse_args()
    if args.contexts_per_run <= 0:
        parser.error("--contexts-per-run must be positive.")
    if args.max_contexts <= 0:
        parser.error("--max-contexts must be positive.")
    if args.max_existing_cards <= 0:
        parser.error("--max-existing-cards must be positive.")
    if args.similarity_threshold <= 0 or args.similarity_threshold > 1:
        parser.error("--similarity-threshold must fall within (0, 1].")
    if args.semantic_similarity_threshold <= 0 or args.semantic_similarity_threshold > 1:
        parser.error("--semantic-similarity-threshold must fall within (0, 1].")
    if args.min_score < 0:
        parser.error("--min-score must be non-negative.")
    if args.max_chat_files <= 0:
        parser.error("--max-chat-files must be positive.")
    return args


def normalize_text(value: str) -> str:
    collapsed = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return re.sub(r"\s+", " ", collapsed).strip()


def parse_html_deck(html_path: Path, cache_path: Optional[Path] = None) -> List[Card]:
    """Parse HTML deck with optional caching to avoid re-parsing large files."""
    # Check cache if provided
    if cache_path and cache_path.exists():
        try:
            cache_data = json.loads(cache_path.read_text())
            # Check if HTML file hasn't been modified since cache
            html_mtime = html_path.stat().st_mtime
            if cache_data.get("html_mtime") == html_mtime:
                # Reconstruct Card objects from cache
                return [
                    Card(
                        deck=c["deck"],
                        front=c["front"],
                        back=c["back"],
                        tags=c["tags"],
                        meta=c["meta"],
                        data_search=c["data_search"],
                        front_norm=c["front_norm"],
                        back_norm=c["back_norm"],
                    )
                    for c in cache_data["cards"]
                ]
        except (json.JSONDecodeError, KeyError):
            pass  # Cache invalid, re-parse

    # Parse HTML (this can be slow for large files)
    soup = BeautifulSoup(html_path.read_text(), "html.parser")
    deck = html_path.stem.replace("_", " ")
    cards: List[Card] = []
    for section in soup.select("section.card-wrapper"):
        front_el = section.select_one(".front")
        back_el = section.select_one(".back")
        if not front_el or not back_el:
            continue
        front = front_el.get_text("\n", strip=True)
        back = back_el.get_text("\n", strip=True)
        meta_el = section.select_one(".card-meta")
        meta_text = meta_el.get_text(" ", strip=True) if meta_el else ""
        data_search = section.get("data-search") or ""
        tags = [
            tag.strip()
            for chunk in meta_text.split("--")
            for tag in [chunk.strip()]
            if tag
        ]
        cards.append(
            Card(
                deck=deck,
                front=front,
                back=back,
                tags=tags,
                meta=meta_text,
                data_search=data_search,
                front_norm=normalize_text(front),
                back_norm=normalize_text(back),
            )
        )

    # Save to cache if path provided
    if cache_path:
        cache_data = {
            "html_mtime": html_path.stat().st_mtime,
            "cards": [asdict(c) for c in cards]
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache_data, indent=2))

    return cards


def collect_decks(pattern: str, cache_dir: Optional[Path] = None) -> List[Card]:
    """Collect cards from HTML decks with optional caching."""
    cards: List[Card] = []
    for path_str in sorted(glob(pattern)):
        path = Path(path_str)
        if not path.is_file():
            continue

        # Use cache if cache_dir provided
        cache_path = None
        if cache_dir:
            cache_path = cache_dir / f"{path.stem}_cards.json"

        cards.extend(parse_html_deck(path, cache_path))
    return cards


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


def extract_turns(entries: Sequence[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
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
        phrase in user_lower for phrase in ["walk me through", "show me", "give me", "steps"]
    )
    signals["imperative"] = imperative
    if imperative:
        score += 0.2

    return score, signals


def extract_key_terms(text: str, limit: int = 6) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", text.lower())
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
    state_tracker: Optional[StateTracker],
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
                (combined_key + user_entry["text"] + assistant_entry["text"]).encode("utf-8")
            ).hexdigest()
            if context_id in seen_ids:
                continue
            score, signals = detect_signals(user_entry["text"], assistant_entry["text"])
            if score < args.min_score:
                continue
            if per_file_counter[path] >= args.per_file_context_limit:
                break
            assistant_full = assistant_entry["text"].strip()
            assistant_trimmed = assistant_full[: args.assistant_char_limit]
            key_terms = extract_key_terms(user_entry["text"] + " " + assistant_full)
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


class SemanticCardIndex:
    """
    Semantic index over existing cards using sentence-transformer embeddings.

    This is used for semantic deduplication of contexts against existing deck
    content. It is intentionally lightweight and instantiated once per run.
    """

    def __init__(
        self,
        cards: List[Card],
        sentence_transformer_cls: Any,
        np_module: Any,
        model_name: str,
        verbose: bool = False,
    ) -> None:
        self.cards = cards
        self._np = np_module
        self.model_name = model_name

        if verbose:
            print(f"Loading semantic dedup model '{model_name}'...")

        self.model = sentence_transformer_cls(model_name)

        # Build embeddings for all cards using front+back text
        texts = [
            (card.front + " " + card.back).strip()
            for card in cards
            if (card.front or card.back)
        ]

        if not texts:
            # No cards – keep a trivial empty matrix
            self.embeddings = np_module.zeros((0, 1), dtype="float32")
            return

        emb = self.model.encode(texts, convert_to_numpy=True)
        emb = emb.astype("float32")
        norms = np_module.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = emb / norms

    def is_duplicate(self, context: ChatTurn, threshold: float) -> bool:
        """Return True if context is semantically close to any existing card."""
        if self.embeddings.shape[0] == 0:
            return False

        np_module = self._np

        text = (context.user_prompt + " " + context.assistant_answer).strip()
        if not text:
            return False

        query_emb = self.model.encode([text], convert_to_numpy=True)
        query_vec = query_emb[0].astype("float32")
        norm = np_module.linalg.norm(query_vec)
        if norm == 0:
            return False
        query_vec = query_vec / norm

        scores = self.embeddings @ query_vec
        max_score = float(scores.max())
        return max_score >= threshold


def quick_similarity(s1: str, s2: str) -> float:
    """Fast approximate similarity using set overlap of words."""
    if not s1 or not s2:
        return 0.0
    words1 = set(s1.split())
    words2 = set(s2.split())
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def is_duplicate_context(
    context: ChatTurn, cards: List[Card], threshold: float
) -> bool:
    """Check if context is duplicate against existing cards with optimization."""
    user_norm = normalize_text(context.user_prompt)
    answer_norm = normalize_text(context.assistant_answer)

    # Quick pre-filter: only do expensive SequenceMatcher on promising candidates
    quick_threshold = threshold * 0.6  # Lower threshold for quick check

    for card in cards:
        if not card.front_norm and not card.back_norm:
            continue

        # Quick check first (fast)
        if card.front_norm:
            if quick_similarity(user_norm, card.front_norm) >= quick_threshold:
                # Only do expensive check if quick check passes
                if SequenceMatcher(None, user_norm, card.front_norm).ratio() >= threshold:
                    return True

        if card.back_norm:
            if quick_similarity(answer_norm, card.back_norm) >= quick_threshold:
                # Only do expensive check if quick check passes
                if SequenceMatcher(None, answer_norm, card.back_norm).ratio() >= threshold:
                    return True

    return False


def prune_contexts(
    contexts: List[ChatTurn], cards: List[Card], args: argparse.Namespace
) -> List[ChatTurn]:
    """Filter contexts to remove duplicates against existing cards."""
    pruned: List[ChatTurn] = []
    total = len(contexts)

    semantic_index: Optional[SemanticCardIndex] = None
    if args.dedup_method in ("semantic", "hybrid"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            import numpy as _np  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "Semantic deduplication requires the 'sentence-transformers' and 'numpy' packages.\n"
                "Install them, for example:\n\n"
                "  uv pip install -e '.[semantic]'\n"
                "  # or\n"
                "  pip install 'sentence-transformers' 'numpy'\n"
            ) from exc

        if args.verbose:
            print("Building semantic index over existing cards for deduplication...")

        semantic_index = SemanticCardIndex(
            cards=cards,
            sentence_transformer_cls=SentenceTransformer,
            np_module=_np,
            model_name=args.semantic_model,
            verbose=args.verbose,
        )

    for idx, context in enumerate(sorted(contexts, key=lambda c: c.score, reverse=True), 1):
        if args.verbose:
            print(f"  Checking context {idx}/{total} for duplicates...", end='\r')

        is_dup = False

        # String-based deduplication (current default behaviour)
        if args.dedup_method in ("string", "hybrid"):
            if is_duplicate_context(context, cards, args.similarity_threshold):
                is_dup = True

        # Semantic deduplication using embeddings
        if not is_dup and semantic_index is not None:
            if semantic_index.is_duplicate(context, args.semantic_similarity_threshold):
                is_dup = True

        if is_dup:
            continue
        pruned.append(context)

    if args.verbose:
        print(f"  Checked {total} contexts for duplicates - done!     ")

    return pruned


def select_existing_cards_for_prompt(cards: List[Card], limit: int) -> List[Card]:
    if len(cards) <= limit:
        return cards
    by_deck: Dict[str, List[Card]] = defaultdict(list)
    for card in cards:
        by_deck[card.deck].append(card)
    selected: List[Card] = []
    per_deck_cap = max(1, limit // max(1, len(by_deck)))
    for deck_cards in by_deck.values():
        selected.extend(deck_cards[:per_deck_cap])
    remaining = [card for card in cards if card not in selected]
    for card in remaining:
        if len(selected) >= limit:
            break
        selected.append(card)
    return selected[:limit]


def chunked(seq: Sequence[Any], size: int) -> Iterable[List[Any]]:
    for start in range(0, len(seq), size):
        yield list(seq[start : start + size])


def format_cards_as_markdown(
    cards_json: List[Dict[str, Any]],
    contexts: List[ChatTurn],
    run_timestamp: str
) -> str:
    """
    Format proposed cards as markdown following Anki best practices.
    """
    context_map = {ctx.context_id: ctx for ctx in contexts}

    lines = [
        "# Proposed Anki Cards",
        f"",
        f"Generated: {run_timestamp}",
        f"Total cards: {len(cards_json)}",
        "",
        "---",
        ""
    ]

    for card_data in cards_json:
        context_id = card_data.get("context_id", "unknown")
        context = context_map.get(context_id)

        # Card header
        lines.append(f"## Card: {card_data.get('front', 'No front')[:60]}...")
        lines.append("")

        # Metadata
        lines.append(f"**Deck:** {card_data.get('deck', 'unknown')}")
        lines.append(f"**Style:** {card_data.get('card_style', 'basic')}")

        # Handle confidence as either string or float
        confidence = card_data.get('confidence', 0.0)
        try:
            confidence_float = float(confidence) if confidence else 0.0
            lines.append(f"**Confidence:** {confidence_float:.2f}")
        except (ValueError, TypeError):
            lines.append(f"**Confidence:** {confidence}")

        if card_data.get('tags'):
            lines.append(f"**Tags:** {', '.join(card_data['tags'])}")

        # Source info
        if context:
            lines.append("")
            lines.append(f"**Source:** {Path(context.source_path).name}")
            if context.source_title:
                lines.append(f"**Title:** {context.source_title}")
            if context.user_timestamp:
                lines.append(f"**Date:** {context.user_timestamp}")

        lines.append("")

        # Card content
        lines.append("### Front")
        lines.append("")
        lines.append(card_data.get('front', ''))
        lines.append("")

        lines.append("### Back")
        lines.append("")
        lines.append(card_data.get('back', ''))
        lines.append("")

        # Notes
        if card_data.get('notes'):
            lines.append("### Notes")
            lines.append("")
            lines.append(card_data['notes'])
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def build_codex_prompt(
    cards: List[Card],
    contexts: List[ChatTurn],
    args: argparse.Namespace,
) -> str:
    cards_payload = [
        {
            "deck": card.deck,
            "front": card.front,
            "back": card.back,
            "tags": card.tags,
        }
        for card in cards
    ]
    contexts_payload = [
        {
            "context_id": ctx.context_id,
            "source_path": ctx.source_path,
            "source_title": ctx.source_title,
            "source_url": ctx.source_url,
            "user_timestamp": ctx.user_timestamp,
            "user_prompt": ctx.user_prompt,
            "assistant_answer": ctx.assistant_answer,
            "assistant_char_count": ctx.assistant_char_count,
            "score": round(ctx.score, 3),
            "signals": ctx.signals,
            "key_terms": ctx.key_terms,
            "key_points": ctx.key_points,
        }
        for ctx in contexts
    ]
    contract = {
        "cards": [
            {
                "context_id": "string",
                "deck": "string",
                "card_style": "basic|cloze|list|multi",
                "front": "string",
                "back": "string",
                "tags": ["list", "of", "tags"],
                "confidence": "0-1 float",
                "notes": "why this card matters / follow-ups",
            }
        ],
        "skipped": [
            {
                "context_id": "string",
                "reason": "why the context was skipped (duplicates, unclear, etc.)",
            }
        ],
        "topics_to_track": [
            {
                "topic": "string",
                "justification": "short rationale / suggested next prompt",
            }
        ],
    }
    payload = {
        "existing_cards": cards_payload,
        "candidate_contexts": contexts_payload,
        "output_contract": contract,
    }
    instructions = textwrap.dedent(
        """
        CRITICAL: You MUST respond with ONLY valid JSON matching the output_contract below.
        Do NOT include markdown, explanations, or any text outside the JSON structure.
        Do NOT wrap the JSON in ```json blocks.

        You are operating as the decision layer of an autonomous spaced-repetition agent.

        ## Core Philosophy

        1. **Understand First, Memorize Second**: Each card should reflect a clear mental model,
           flowing logically from general to specific concepts.
        2. **Build Upon the Basics**: Order cards logically, starting with foundational concepts
           before moving to detailed information.

        ## The Golden Rule: Minimum Information Principle

        **Each card must isolate the smallest possible piece of information.**
        - Questions should be precise and unambiguous
        - Answers should be as short as possible while remaining complete
        - NEVER create complex cards that cram multiple unrelated facts together
        - Break down sets/lists into individual cards (one item per card)

        Example of BAD (too complex):
        Q: What are the characteristics of gradient boosting?
        A: Uses decision trees, minimizes loss via gradient descent in function space,
           builds additive models sequentially, and requires tuning learning rate.

        Example of GOOD (atomic):
        Q: What type of base learner does gradient boosting typically use?
        A: **Decision trees** (usually shallow trees)

        ## Card Formats

        Use the format that best fits the information:

        1. **Question/Answer (basic)**: Default format for most concepts
           - Make questions clear and specific
           - Answers should be concise and self-contained

        2. **Cloze Deletion**: Ideal for facts, definitions, vocabulary
           - Use [...] to mark the part to recall
           - Especially good for "X is defined as Y" type statements

        ## Content Guidelines

        - **Combat Interference**: When concepts are easily confused, create cards that
          explicitly ask for the distinction
        - **Optimize Wording**: Remove extraneous words. Aim for rapid comprehension.
        - **Use Context Cues**: For ambiguous terms, prefix with context (e.g., "(ML) What is gradient...")
        - **Handle Sets**: NEVER ask for a list of more than 2-3 items. Break down into
          multiple cards asking for each item separately.
        - **Use LaTeX**: For mathematical notation, use LaTeX with proper delimiters

        ## Your Task

        For each `candidate_context`:
        1. Decide if it contains learning-worthy knowledge (not trivial, not already covered)
        2. Check against `existing_cards` to avoid duplicates
        3. If justified, create one or MORE atomic cards (break complex topics into multiple cards)
        4. Choose appropriate deck based on topic area
        5. Add relevant tags for organization
        6. Provide confidence score (0-1) and brief notes on why this card matters

        ## Output Requirements

        Return ONLY valid JSON adhering to `output_contract`. Critical rules:
        - NO markdown fencing (no ```json blocks)
        - NO explanatory text before or after the JSON
        - NO comments inside the JSON
        - START your response with `{` and END with `}`
        - `card_style` should be: basic, cloze, or list
        - `confidence`: 0.0-1.0 (higher = more confident this card is valuable)
        - `notes`: Brief rationale for why this card was created
        - `skipped`: Contexts you skipped and why (too trivial, duplicate, unclear, etc.)

        YOUR ENTIRE RESPONSE MUST BE VALID, PARSEABLE JSON.
        """
    ).strip()
    return instructions + "\n\n" + json.dumps(payload, indent=2, ensure_ascii=False)


def run_codex_exec(
    prompt: str,
    chunk_idx: int,
    run_dir: Path,
    args: argparse.Namespace,
) -> str:
    prompt_path = run_dir / f"prompt_chunk_{chunk_idx:02d}.txt"
    prompt_path.write_text(prompt)
    last_msg_path = run_dir / f"codex_response_chunk_{chunk_idx:02d}.json"

    # Build command
    cmd = ["codex", "exec", "-", "--skip-git-repo-check"]
    model = args.codex_model or "gpt-5"
    cmd.extend(["--model", model])
    if args.model_reasoning_effort:
        cmd.extend(["-c", f"model_reasoning_effort={args.model_reasoning_effort}"])

    for extra in args.codex_extra_arg:
        if extra:
            cmd.extend(shlex.split(extra))
    cmd.extend(["--output-last-message", str(last_msg_path)])
    proc = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        cwd=os.getcwd(),
    )
    (run_dir / f"codex_stdout_chunk_{chunk_idx:02d}.log").write_text(proc.stdout)
    (run_dir / f"codex_stderr_chunk_{chunk_idx:02d}.log").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"codex exec failed for chunk {chunk_idx} with exit code {proc.returncode}"
        )
    return last_msg_path.read_text().strip()


def parse_codex_response_robust(
    response_text: str,
    chunk_idx: int,
    run_dir: Path,
    verbose: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Parse codex JSON response with multiple fallback strategies.
    Returns None if all strategies fail.
    """
    # Save raw response
    raw_path = run_dir / f"codex_raw_response_chunk_{chunk_idx:02d}.txt"
    raw_path.write_text(response_text)

    strategies = []

    # Strategy 1: Direct parse
    strategies.append(("Direct parse", response_text.strip()))

    # Strategy 2: Strip markdown fences
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    strategies.append(("Markdown stripped", cleaned))

    # Strategy 3: Use json-repair library
    try:
        repaired = repair_json(cleaned)
        strategies.append(("JSON repair", repaired))
    except Exception:
        pass  # json-repair failed, skip this strategy

    # Try each strategy
    for strategy_name, text in strategies:
        try:
            result = json.loads(text)
            if verbose:
                print(f"  ✓ Parsed with strategy: {strategy_name}")
            # Save the working version
            (run_dir / f"codex_parsed_response_chunk_{chunk_idx:02d}.json").write_text(
                json.dumps(result, indent=2)
            )
            return result
        except json.JSONDecodeError as e:
            if verbose:
                print(f"  ✗ {strategy_name} failed: {e}")
            continue

    # All strategies failed - save debug info
    error_file = run_dir / f"codex_FAILED_chunk_{chunk_idx:02d}.txt"
    error_info = f"""All JSON parsing strategies failed for chunk {chunk_idx}

Strategies tried:
{chr(10).join(f'- {name}' for name, _ in strategies)}

First 1000 chars of response:
{response_text[:1000]}

Last error: {e if 'e' in locals() else 'Unknown'}
"""
    error_file.write_text(error_info)

    return None


def ensure_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / f"run-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    args = parse_args()
    deck_glob = args.deck_glob or str(SCRIPT_DIR / "*.html")
    state_path = Path(args.state_file) if args.state_file else SCRIPT_DIR / ".auto_anki_agent_state.json"
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "auto_anki_runs"
    cache_dir = Path(args.cache_dir) if args.cache_dir else SCRIPT_DIR / ".deck_cache"

    chat_root = Path(args.chat_root).expanduser()
    if not chat_root.exists():
        raise SystemExit(f"Chat root {chat_root} does not exist.")

    # Initialize state tracker and filters
    state_tracker = StateTracker(state_path)
    date_filter = DateRangeFilter(args.date_range) if args.date_range else None
    seen_ids = state_tracker.get_seen_context_ids()

    # Load existing cards (with caching for speed)
    if args.verbose:
        print(f"Loading existing cards from {deck_glob}...")
    cards = collect_decks(deck_glob, cache_dir)
    if args.verbose:
        print(f"✓ Loaded {len(cards)} cards")
        if date_filter and date_filter.start_date:
            print(f"Date filter: {date_filter.start_date} to {date_filter.end_date or 'now'}")
        if args.unprocessed_only:
            print("Mode: Unprocessed files only")

    # Harvest and filter contexts
    contexts = harvest_chat_contexts(chat_root, seen_ids, state_tracker, date_filter, args)
    if args.verbose:
        print(f"Harvested {len(contexts)} raw contexts before pruning.")
    contexts = prune_contexts(contexts, cards, args)
    if args.verbose:
        print(f"{len(contexts)} contexts remain after dedup/score filtering.")
    if not contexts:
        print("No new contexts found; exiting.")
        return

    # Prepare run directory
    run_dir = ensure_run_dir(output_dir)
    run_timestamp = datetime.now().isoformat()
    (run_dir / "selected_contexts.json").write_text(
        json.dumps([asdict(ctx) for ctx in contexts], indent=2)
    )

    cards_for_prompt = select_existing_cards_for_prompt(cards, args.max_existing_cards)

    # Process chunks
    new_seen_ids: List[str] = []
    all_proposed_cards: List[Dict[str, Any]] = []
    processed_files: set[Path] = set()
    chunk_stats = {"success": 0, "failed": 0, "total_cards": 0}

    total_chunks = (len(contexts) + args.contexts_per_run - 1) // args.contexts_per_run
    if args.verbose:
        print(f"\nProcessing {len(contexts)} contexts in {total_chunks} chunk(s) of {args.contexts_per_run}...")

    for idx, chunk in enumerate(chunked(contexts, args.contexts_per_run), start=1):
        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Chunk {idx}/{total_chunks}: Processing {len(chunk)} contexts")
            print(f"{'='*60}")

        prompt = build_codex_prompt(cards_for_prompt, chunk, args)
        if args.dry_run:
            prompt_path = run_dir / f"prompt_chunk_{idx:02d}.txt"
            prompt_path.write_text(prompt)
            if args.verbose:
                print(f"[dry-run] Saved prompt for chunk {idx} at {prompt_path}")
            continue

        if args.verbose:
            print(f"Calling codex exec for chunk {idx}...")
            print(f"  (This may take 30-60 seconds depending on model and context size)")

        try:
            response_text = run_codex_exec(prompt, idx, run_dir, args)

            if args.verbose:
                print(f"✓ Received response from codex (chunk {idx})")
                print(f"  Parsing JSON response...")

            # Use robust multi-strategy parser
            response_json = parse_codex_response_robust(
                response_text, idx, run_dir, args.verbose
            )

            if response_json is None:
                # Parsing failed - log and continue
                chunk_stats["failed"] += 1
                print(f"⚠️  Chunk {idx} FAILED: Could not parse JSON response")
                print(f"   Debug info saved to: {run_dir}/codex_FAILED_chunk_{idx:02d}.txt")
                print(f"   Continuing with remaining chunks...")
                continue

            # Success! Track the results
            chunk_stats["success"] += 1

            # Track proposed cards and processed files
            cards_in_chunk = 0
            if "cards" in response_json:
                cards_in_chunk = len(response_json["cards"])
                all_proposed_cards.extend(response_json["cards"])
                chunk_stats["total_cards"] += cards_in_chunk

            skipped_in_chunk = len(response_json.get("skipped", []))

            if args.verbose:
                print(f"✓ Chunk {idx} complete:")
                print(f"    {cards_in_chunk} cards proposed")
                print(f"    {skipped_in_chunk} contexts skipped")

            for ctx in chunk:
                new_seen_ids.append(ctx.context_id)
                processed_files.add(Path(ctx.source_path))

        except Exception as e:
            # Unexpected error - log and continue
            chunk_stats["failed"] += 1
            error_file = run_dir / f"codex_ERROR_chunk_{idx:02d}.txt"
            error_file.write_text(f"Unexpected error processing chunk {idx}:\n\n{str(e)}")
            print(f"⚠️  Chunk {idx} ERROR: {str(e)[:100]}")
            print(f"   Error saved to: {error_file}")
            print(f"   Continuing with remaining chunks...")
            continue

    if not args.dry_run:
        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Generating output files...")
            print(f"{'='*60}")

        # Generate output files
        if args.output_format in ["markdown", "both"]:
            markdown_content = format_cards_as_markdown(all_proposed_cards, contexts, run_timestamp)
            markdown_path = output_dir / f"proposed_cards_{datetime.now().strftime('%Y-%m-%d')}.md"
            markdown_path.write_text(markdown_content)
            if args.verbose:
                print(f"✓ Markdown cards saved to: {markdown_path}")
            else:
                print(f"Markdown cards saved to: {markdown_path}")

        if args.output_format in ["json", "both"]:
            json_path = run_dir / "all_proposed_cards.json"
            json_path.write_text(json.dumps(all_proposed_cards, indent=2))
            if args.verbose:
                print(f"✓ JSON cards saved to: {json_path}")
            else:
                print(f"JSON cards saved to: {json_path}")

        # Update state
        if args.verbose:
            print(f"\nUpdating state file...")
        state_tracker.add_context_ids(new_seen_ids)
        for file_path in processed_files:
            file_cards = sum(1 for card in all_proposed_cards
                           if any(ctx.source_path == str(file_path) and ctx.context_id == card.get("context_id")
                                 for ctx in contexts))
            state_tracker.mark_file_processed(file_path, file_cards)
        state_tracker.record_run(run_dir, len(new_seen_ids))
        state_tracker.save()

        if args.verbose:
            print(f"✓ State updated")
            print(f"\n{'='*60}")
            print(f"SUMMARY")
            print(f"{'='*60}")
            print(f"Chunks successful:     {chunk_stats['success']}/{total_chunks}")
            print(f"Chunks failed:         {chunk_stats['failed']}/{total_chunks}")
            print(f"Contexts processed:    {len(new_seen_ids)}")
            print(f"Cards generated:       {chunk_stats['total_cards']}")
            print(f"Files processed:       {len(processed_files)}")
            print(f"Run artifacts:         {run_dir}")
            if chunk_stats['failed'] > 0:
                print(f"\n⚠️  {chunk_stats['failed']} chunk(s) failed - check {run_dir} for details")
            print(f"{'='*60}\n")
        else:
            status = "✓" if chunk_stats['failed'] == 0 else "⚠️"
            print(f"{status} Run complete: {chunk_stats['success']}/{total_chunks} chunks successful")
            print(f"Generated {chunk_stats['total_cards']} proposed cards from {len(new_seen_ids)} contexts")
            if chunk_stats['failed'] > 0:
                print(f"⚠️  {chunk_stats['failed']} chunk(s) failed - check {run_dir} for details")
            print(f"Run artifacts saved to {run_dir}")
    else:
        print(f"[dry-run] Contexts + prompts saved to {run_dir}. No state updates.")


if __name__ == "__main__":
    main()

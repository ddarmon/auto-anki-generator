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
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from auto_anki.cards import Card, load_cards_from_anki
from auto_anki.contexts import (
    ChatTurn,
    Conversation,
    DateRangeFilter,
    harvest_chat_contexts,
    harvest_conversations,
)
from auto_anki.codex import (
    build_conversation_prompt,
    build_conversation_filter_prompt,
    chunked,
    parse_codex_response_robust,
    run_codex_exec,
)
from auto_anki.dedup import enrich_cards_with_duplicate_flags, prune_contexts, prune_conversations
from auto_anki.llm_backends import list_backends
from auto_anki.state import StateTracker, ensure_run_dir

# Silence HuggingFace tokenizers fork/parallelism warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


SCRIPT_DIR = Path(__file__).resolve().parent


def _load_config() -> tuple[Dict[str, Any], Optional[Path]]:
    """
    Load optional configuration for default paths and settings.

    Search order:
    1. Path from AUTO_ANKI_CONFIG (if set)
    2. ./auto_anki_config.json in current working directory
    3. ~/.auto_anki_config.json in the user home directory

    Relative paths in the config are resolved relative to the config file's
    directory. This makes it easy to keep a config next to the source
    checkout and have outputs land there even when running the installed
    package from site-packages.
    """
    candidates: List[Path] = []
    env_path = os.getenv("AUTO_ANKI_CONFIG")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path.cwd() / "auto_anki_config.json")
    candidates.append(Path.home() / ".auto_anki_config.json")

    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
            return data, path
        except json.JSONDecodeError:
            print(f"⚠️  Ignoring invalid config file (JSON parse error): {path}")
            break

    return {}, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harvest chat transcripts + deck coverage and ask codex exec to propose new Anki cards."
    )
    parser.add_argument(
        "--decks",
        nargs="+",
        help="Anki deck names to load existing cards from (overrides config).",
    )
    parser.add_argument(
        "--anki-cache-ttl",
        type=int,
        default=5,
        help="Cache TTL in minutes for AnkiConnect card fetch (default: %(default)s).",
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
        "--reprocess-zero-card-files",
        action="store_true",
        help="Also reprocess files that previously generated 0 cards (use with --unprocessed-only).",
    )
    parser.add_argument(
        "--only-zero-card-files",
        action="store_true",
        help="Process ONLY files that previously generated 0 cards (exclusive backfill mode).",
    )
    parser.add_argument(
        "--state-file",
        help="Path to JSON state file used to skip already-seen contexts (default: <script_dir>/.auto_anki_agent_state.json).",
    )
    parser.add_argument(
        "--output-format",
        default="json",
        help="Deprecated: Markdown output has been removed; JSON is always written.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to store prompts, contexts, and codex responses (default: <script_dir>/auto_anki_runs).",
    )
    parser.add_argument(
        "--cache-dir",
        help=(
            "Directory to cache Anki cards and semantic embedding index "
            "(default: <script_dir>/.deck_cache)."
        ),
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
        default="hybrid",
        help=(
            "Deduplication method: 'string' uses lexical similarity only, "
            "'semantic' uses embedding-based similarity, 'hybrid' uses both. "
            "(default: %(default)s; auto-falls back to 'string' if semantic dependencies unavailable)"
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
        "--two-stage",
        dest="two_stage",
        action="store_true",
        help=(
            "Enable two-stage LLM pipeline: fast stage-1 filter + stage-2 card generator "
            "(default: enabled; use --single-stage to disable)."
        ),
    )
    parser.add_argument(
        "--single-stage",
        dest="two_stage",
        action="store_false",
        help="Disable two-stage LLM pipeline and use single-stage card generation.",
    )
    parser.add_argument(
        "--codex-model-stage1",
        default="gpt-5.1",
        help="Codex model to use for stage-1 filtering when --two-stage is enabled (default: %(default)s).",
    )
    parser.add_argument(
        "--codex-model-stage2",
        default="gpt-5.1",
        help=(
            "Codex model to use for stage-2 card generation when --two-stage is enabled "
            "(default: %(default)s). If not set, falls back to --codex-model or 'gpt-5.1'."
        ),
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=1.2,
        help="Minimum heuristic score a context must reach to be considered (only used with --use-filter-heuristics).",
    )
    parser.add_argument(
        "--use-filter-heuristics",
        action="store_true",
        help=(
            "Enable heuristic pre-filtering before Stage 1 LLM. "
            "By default, all conversations are sent directly to Stage 1 LLM filter."
        ),
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
    # LLM Backend configuration
    parser.add_argument(
        "--llm-backend",
        choices=list_backends(),
        default=None,
        help=(
            "LLM backend to use for card generation. "
            "Choices: %(choices)s. "
            "Default from config file or 'codex'."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Model to use for LLM calls (overrides config file and stage-specific settings).",
    )
    parser.add_argument(
        "--llm-model-stage1",
        default=None,
        help="Model to use for Stage 1 filtering (overrides config file).",
    )
    parser.add_argument(
        "--llm-model-stage2",
        default=None,
        help="Model to use for Stage 2 card generation (overrides config file).",
    )
    parser.add_argument(
        "--llm-extra-arg",
        action="append",
        default=[],
        help="Repeatable passthrough arguments for LLM backend.",
    )
    # Legacy Codex-specific arguments (deprecated, kept for backward compatibility)
    parser.add_argument(
        "--codex-model",
        default=None,
        help="[DEPRECATED: use --llm-model] Optional model override passed to codex exec via --model (default: gpt-5.1).",
    )
    parser.add_argument(
        "--model-reasoning-effort",
        default=None,
        help=(
            "Set reasoning effort for Codex backend. "
            "If unset, defaults to 'medium' in single-stage mode, "
            "or 'low' (stage 1) / 'high' (stage 2) when --two-stage is enabled."
        ),
    )
    parser.add_argument(
        "--codex-extra-arg",
        action="append",
        default=[],
        help="[DEPRECATED: use --llm-extra-arg] Repeatable passthrough arguments for codex exec.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional progress information.",
    )
    # Conversation-level processing (new default)
    parser.add_argument(
        "--conversation-max-turns",
        type=int,
        default=10,
        help="Maximum turns to include per conversation before splitting (default: %(default)s).",
    )
    parser.add_argument(
        "--conversation-max-chars",
        type=int,
        default=8000,
        help="Maximum total characters per conversation before splitting (default: %(default)s).",
    )
    # Post-generation duplicate detection
    parser.add_argument(
        "--post-dedup-threshold",
        type=float,
        default=0.85,
        help=(
            "Semantic similarity threshold for flagging likely duplicates in post-generation dedup "
            "(default: %(default)s). Cards with similarity above this threshold are flagged."
        ),
    )
    parser.add_argument(
        "--skip-post-dedup",
        action="store_true",
        help="Skip post-generation duplicate detection (proposed cards will not be flagged).",
    )
    # Defaults
    parser.set_defaults(two_stage=True)

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
    if args.post_dedup_threshold <= 0 or args.post_dedup_threshold > 1:
        parser.error("--post-dedup-threshold must fall within (0, 1].")
    if args.min_score < 0:
        parser.error("--min-score must be non-negative.")
    if args.max_chat_files <= 0:
        parser.error("--max-chat-files must be positive.")
    if (args.output_format or "").lower() != "json":
        print("⚠️  Markdown output has been removed; writing JSON only.")
        args.output_format = "json"
    return args


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


def ensure_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / f"run-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _process_stage2_batch(
    batch_info: Tuple[int, List[Conversation], Path, Any],
) -> Dict[str, Any]:
    """Process a single Stage 2 batch. Used for parallel execution.

    Args:
        batch_info: Tuple of (batch_idx, filtered_conversations, run_dir, args)

    Returns:
        Dict with keys: success, batch_idx, cards, skipped_conversations, skipped_turns,
                       conversation_ids, source_paths, error
    """
    batch_idx, filtered_chunk, run_dir, args = batch_info

    result: Dict[str, Any] = {
        "success": False,
        "batch_idx": batch_idx,
        "cards": [],
        "skipped_conversations": 0,
        "skipped_turns": 0,
        "conversation_ids": [],
        "source_paths": [],
        "error": None,
    }

    try:
        prompt = build_conversation_prompt(filtered_chunk, args)

        # Use resolved stage2 settings
        stage2_model = getattr(args, "llm_model_stage2", None) or (args.codex_model_stage2 if args.two_stage else None)
        stage2_reasoning = getattr(args, "model_reasoning_effort_stage2", None) or args.model_reasoning_effort or ("high" if args.two_stage else "medium")

        response_text = run_codex_exec(
            prompt,
            batch_idx,
            run_dir,
            args,
            model_override=stage2_model,
            reasoning_override=stage2_reasoning,
            label="",
        )

        response_json = parse_codex_response_robust(
            response_text, batch_idx, run_dir, args.verbose, label=""
        )

        if response_json is None:
            result["error"] = "Could not parse JSON response"
            return result

        result["success"] = True
        result["cards"] = response_json.get("cards", [])
        result["skipped_conversations"] = len(response_json.get("skipped_conversations", []))
        result["skipped_turns"] = len(response_json.get("skipped_turns", []))

        # Track processed conversations
        for conv in filtered_chunk:
            result["conversation_ids"].append(conv.conversation_id)
            result["source_paths"].append(conv.source_path)

    except Exception as e:
        error_file = run_dir / f"codex_ERROR_batch_{batch_idx:02d}.txt"
        error_file.write_text(f"Error processing batch {batch_idx}:\n\n{str(e)}")
        result["error"] = str(e)[:200]

    return result


def _apply_llm_config(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Apply LLM backend configuration from config file, with CLI overrides.

    This sets up args with the resolved LLM settings, applying the precedence:
    1. CLI arguments (highest priority)
    2. Config file settings
    3. Hardcoded defaults (lowest priority)
    """
    # Get LLM backend (CLI > config > default)
    if args.llm_backend is None:
        args.llm_backend = config.get("llm_backend", "codex")

    # Get backend-specific config from config file
    llm_config = config.get("llm_config", {})
    backend_cfg = llm_config.get(args.llm_backend, {})

    # Default models per backend
    default_models = {
        "codex": "gpt-5.1",
        "claude-code": "claude-sonnet-4-5-20250929",
    }

    # Resolve model settings with proper fallback chain
    # For Stage 1: CLI llm_model_stage1 > CLI llm_model > config model_stage1 > config model > default
    # For Stage 2: CLI llm_model_stage2 > CLI llm_model > config model_stage2 > config model > default
    config_model = backend_cfg.get("model") or default_models.get(args.llm_backend)
    config_model_stage1 = backend_cfg.get("model_stage1") or config_model
    config_model_stage2 = backend_cfg.get("model_stage2") or config_model

    # Apply to args with CLI override priority
    if args.llm_model:
        # CLI --llm-model overrides everything
        args.llm_model_stage1 = args.llm_model_stage1 or args.llm_model
        args.llm_model_stage2 = args.llm_model_stage2 or args.llm_model
    else:
        # Fall back to config
        args.llm_model_stage1 = args.llm_model_stage1 or config_model_stage1
        args.llm_model_stage2 = args.llm_model_stage2 or config_model_stage2
        args.llm_model = config_model

    # For backward compat: also update codex-specific args
    if args.llm_backend == "codex":
        args.codex_model_stage1 = args.codex_model_stage1 or args.llm_model_stage1
        args.codex_model_stage2 = args.codex_model_stage2 or args.llm_model_stage2
        if args.codex_model is None:
            args.codex_model = args.llm_model

    # Reasoning effort (Codex-specific, but stored generically)
    config_reasoning = backend_cfg.get("reasoning_effort")
    config_reasoning_stage1 = backend_cfg.get("reasoning_effort_stage1") or config_reasoning
    config_reasoning_stage2 = backend_cfg.get("reasoning_effort_stage2") or config_reasoning

    if args.model_reasoning_effort is None:
        # Use config values as base, but let the pipeline set stage-specific defaults
        args.model_reasoning_effort = config_reasoning
        args.model_reasoning_effort_stage1 = config_reasoning_stage1
        args.model_reasoning_effort_stage2 = config_reasoning_stage2
    else:
        args.model_reasoning_effort_stage1 = args.model_reasoning_effort
        args.model_reasoning_effort_stage2 = args.model_reasoning_effort


def main() -> None:
    # CLI args + optional config file
    args = parse_args()
    config, config_path = _load_config()
    config_root = config_path.parent if config_path else None

    # Apply LLM backend configuration
    _apply_llm_config(args, config)

    def resolve_path(value: Optional[str], default: Path) -> Path:
        if not value:
            return default
        path = Path(value).expanduser()
        if not path.is_absolute() and config_root is not None:
            path = config_root / path
        return path

    # Paths (CLI overrides config; config overrides hardcoded defaults)
    # Decks list (CLI overrides config)
    decks = args.decks if args.decks else config.get("decks", [])
    # Store decks in args so prompt builder can access them
    args.decks = decks
    if not decks:
        print("Error: No decks specified.")
        print("Add 'decks' to auto_anki_config.json or use --decks flag.")
        print("Example: --decks Research_Learning Technology_Learning")
        sys.exit(1)
    state_path = (
        Path(args.state_file).expanduser()
        if args.state_file
        else resolve_path(
            config.get("state_file"),
            SCRIPT_DIR / ".auto_anki_agent_state.json",
        )
    )
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else resolve_path(
            config.get("output_dir"),
            SCRIPT_DIR / "auto_anki_runs",
        )
    )
    cache_dir = (
        Path(args.cache_dir).expanduser()
        if args.cache_dir
        else resolve_path(
            config.get("cache_dir"),
            SCRIPT_DIR / ".deck_cache",
        )
    )

    chat_root_str = args.chat_root
    if not args.chat_root and config.get("chat_root"):
        chat_root_str = config["chat_root"]
    chat_root = Path(chat_root_str).expanduser()
    if not chat_root.exists():
        raise SystemExit(f"Chat root {chat_root} does not exist.")

    # Initialize state tracker and filters
    state_tracker = StateTracker(state_path)
    date_filter = DateRangeFilter(args.date_range) if args.date_range else None
    seen_conversation_ids = state_tracker.get_seen_conversation_ids()

    # Load existing cards from Anki via AnkiConnect
    if args.verbose:
        print(f"Loading existing cards from Anki decks: {', '.join(decks)}...")
    try:
        cards = load_cards_from_anki(
            decks,
            cache_dir=cache_dir,
            cache_ttl_minutes=args.anki_cache_ttl,
            verbose=args.verbose,
        )
    except ConnectionError as e:
        # Emit a stable sentinel for tooling (e.g. batch scripts) to detect
        # Anki connection issues robustly without depending on human-facing
        # error message text.
        print("ANKI_CONNECT_ERROR")
        print(f"Error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    if args.verbose:
        print(f"✓ Loaded {len(cards)} cards from Anki")
        print(f"LLM backend: {args.llm_backend}")
        if args.two_stage:
            print(f"  Stage 1 model: {args.llm_model_stage1}")
            print(f"  Stage 2 model: {args.llm_model_stage2}")
        else:
            print(f"  Model: {args.llm_model}")
        if date_filter and date_filter.start_date:
            print(f"Date filter: {date_filter.start_date} to {date_filter.end_date or 'now'}")
        if args.unprocessed_only:
            print("Mode: Unprocessed files only")
        print("Mode: Conversation-level processing (full conversations sent to LLM)")

    # Harvest full conversations (not individual turns)
    conversations = harvest_conversations(
        chat_root, seen_conversation_ids, state_tracker, date_filter, args
    )
    if args.verbose:
        total_turns = sum(len(c.turns) for c in conversations)
        print(f"Harvested {len(conversations)} conversations ({total_turns} total turns) before pruning.")

    # Prune conversations against existing cards
    conversations, skipped_duplicate_ids = prune_conversations(conversations, cards, args)
    if args.verbose:
        total_turns = sum(len(c.turns) for c in conversations)
        print(f"{len(conversations)} conversations ({total_turns} turns) remain after dedup.")

    # Track skipped duplicates in state so they aren't re-processed on next run
    if skipped_duplicate_ids:
        state_tracker.add_conversation_ids(skipped_duplicate_ids)
        state_tracker.save()
        if args.verbose:
            print(f"  Marked {len(skipped_duplicate_ids)} duplicate conversations as seen in state.")

    if not conversations:
        if getattr(args, "only_zero_card_files", False):
            print("No zero-card files found for date range; exiting.")
        else:
            print("No new conversations found; exiting.")
        return

    # Prepare run directory
    run_dir = ensure_run_dir(output_dir)
    run_timestamp = datetime.now().isoformat()

    # Save selected conversations (new format)
    conversations_data = [
        {
            "conversation_id": conv.conversation_id,
            "source_path": conv.source_path,
            "source_title": conv.source_title,
            "source_url": conv.source_url,
            "total_char_count": conv.total_char_count,
            "aggregate_score": conv.aggregate_score,
            "aggregate_signals": conv.aggregate_signals,
            "key_topics": conv.key_topics,
            "turns": [asdict(turn) for turn in conv.turns],
        }
        for conv in conversations
    ]
    (run_dir / "selected_conversations.json").write_text(
        json.dumps(conversations_data, indent=2)
    )

    # Process conversation chunks (optionally with two-stage pipeline)
    new_conversation_ids: List[str] = []
    all_proposed_cards: List[Dict[str, Any]] = []
    processed_files: set[Path] = set()
    chunk_stats: Dict[str, Any] = {
        "success": 0,
        "failed": 0,
        "total_cards": 0,
        "stage1_kept": 0,
        "stage1_total": 0,
    }

    # Batch conversations (using contexts_per_run as conversations per batch)
    conversations_per_batch = args.contexts_per_run
    total_chunks = (len(conversations) + conversations_per_batch - 1) // conversations_per_batch
    if args.verbose:
        print(f"\nProcessing {len(conversations)} conversations in {total_chunks} batch(es)...")

    # ========================================================================
    # PHASE 1: Stage 1 filtering (sequential - it's fast)
    # ========================================================================
    stage2_batches: List[Tuple[int, List[Conversation]]] = []

    for idx, chunk in enumerate(chunked(conversations, conversations_per_batch), start=1):
        if args.verbose:
            chunk_turns = sum(len(c.turns) for c in chunk)
            print(f"\n{'='*60}")
            print(f"Batch {idx}/{total_chunks}: {len(chunk)} conversations ({chunk_turns} turns)")
            print(f"{'='*60}")

        # Optionally run stage-1 filter (conversation-level)
        filtered_chunk = chunk
        if args.two_stage:
            if args.verbose:
                print("  Stage 1: filtering conversations before card generation...")

            filter_prompt = build_conversation_filter_prompt(chunk, args)

            if args.dry_run:
                prompt_stage1_path = run_dir / f"prompt_stage1_batch_{idx:02d}.txt"
                prompt_stage1_path.write_text(filter_prompt)
                prompt_stage2_path = run_dir / f"prompt_batch_{idx:02d}.txt"
                stage2_preview = build_conversation_prompt(chunk, args)
                prompt_stage2_path.write_text(stage2_preview)
                if args.verbose:
                    print(f"[dry-run] Saved stage-1 prompt at {prompt_stage1_path}")
                    print(f"[dry-run] Saved stage-2 preview prompt at {prompt_stage2_path}")
                continue

            try:
                # Use resolved stage1 settings
                stage1_model = getattr(args, "llm_model_stage1", None) or args.codex_model_stage1
                stage1_reasoning = getattr(args, "model_reasoning_effort_stage1", None) or args.model_reasoning_effort or "low"
                response_text_stage1 = run_codex_exec(
                    filter_prompt,
                    idx,
                    run_dir,
                    args,
                    model_override=stage1_model,
                    reasoning_override=stage1_reasoning,
                    label="_stage1",
                )

                if args.verbose:
                    print(f"✓ Received stage-1 response (batch {idx})")

                response_json_stage1 = parse_codex_response_robust(
                    response_text_stage1, idx, run_dir, args.verbose, label="_stage1"
                )

                if response_json_stage1 is None:
                    print(f"⚠️  Stage-1 parsing FAILED; sending all conversations to stage 2.")
                    kept_conv_ids: Optional[set[str]] = None
                else:
                    decisions = response_json_stage1.get("filter_decisions", [])
                    kept_conv_ids = {
                        d.get("conversation_id")
                        for d in decisions
                        if isinstance(d, dict) and d.get("keep") is True and d.get("conversation_id")
                    }
                    # Also mark Stage-1 rejected conversations as seen so they
                    # don't get resurfaced on the next batch script pass.
                    rejected_conv_ids = {
                        conv.conversation_id
                        for conv in chunk
                        if conv.conversation_id not in kept_conv_ids
                    }
                    if rejected_conv_ids:
                        new_conversation_ids.extend(rejected_conv_ids)
                    kept_count = len(kept_conv_ids)
                    chunk_stats["stage1_kept"] += kept_count
                    chunk_stats["stage1_total"] += len(chunk)
                    if args.verbose:
                        print(f"  Stage-1 kept {kept_count}/{len(chunk)} conversations.")

                if kept_conv_ids:
                    filtered_chunk = [c for c in chunk if c.conversation_id in kept_conv_ids]
                else:
                    filtered_chunk = chunk

                if not filtered_chunk:
                    if args.verbose:
                        print(f"  Stage-1 filtered all conversations in batch {idx}.")
                    for conv in chunk:
                        new_conversation_ids.append(conv.conversation_id)
                        processed_files.add(Path(conv.source_path))
                    continue

            except Exception as e:
                error_file = run_dir / f"codex_stage1_ERROR_batch_{idx:02d}.txt"
                error_file.write_text(f"Stage-1 error for batch {idx}:\n\n{str(e)}")
                print(f"⚠️  Stage-1 ERROR: {str(e)[:100]}")
                print("   Falling back to sending all conversations to stage 2.")
                filtered_chunk = chunk

        # Handle dry-run for non-two-stage mode
        if args.dry_run and not args.two_stage:
            prompt_path = run_dir / f"prompt_batch_{idx:02d}.txt"
            prompt = build_conversation_prompt(filtered_chunk, args)
            prompt_path.write_text(prompt)
            if args.verbose:
                print(f"[dry-run] Saved prompt at {prompt_path}")
            continue

        # Queue for Stage 2 processing
        if filtered_chunk:
            stage2_batches.append((idx, filtered_chunk))

    # ========================================================================
    # PHASE 2: Stage 2 card generation (parallel - this is the bottleneck)
    # ========================================================================
    if stage2_batches and not args.dry_run:
        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Stage 2: Generating cards for {len(stage2_batches)} batch(es) in parallel...")
            print(f"{'='*60}")

        # Prepare batch info for parallel processing
        batch_infos = [
            (idx, filtered_chunk, run_dir, args)
            for idx, filtered_chunk in stage2_batches
        ]

        # Process in parallel with max 3 concurrent workers
        max_workers = min(3, len(batch_infos))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_process_stage2_batch, info): info[0]
                for info in batch_infos
            }

            for future in as_completed(future_to_idx):
                batch_idx = future_to_idx[future]
                try:
                    result = future.result()

                    if result["success"]:
                        chunk_stats["success"] += 1
                        cards_in_batch = len(result["cards"])
                        all_proposed_cards.extend(result["cards"])
                        chunk_stats["total_cards"] += cards_in_batch

                        if args.verbose:
                            print(f"✓ Batch {result['batch_idx']} complete:")
                            print(f"    {cards_in_batch} cards proposed")
                            print(f"    {result['skipped_conversations']} conversations skipped, {result['skipped_turns']} turns skipped")

                        # Track processed conversations
                        new_conversation_ids.extend(result["conversation_ids"])
                        for path in result["source_paths"]:
                            processed_files.add(Path(path))
                    else:
                        chunk_stats["failed"] += 1
                        print(f"⚠️  Batch {result['batch_idx']} FAILED: {result['error']}")

                except Exception as e:
                    chunk_stats["failed"] += 1
                    print(f"⚠️  Batch {batch_idx} ERROR: {str(e)[:100]}")

    # ========================================================================
    # PHASE 3: Post-generation duplicate detection
    # ========================================================================
    if not args.dry_run and all_proposed_cards and not args.skip_post_dedup:
        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Post-generation duplicate detection...")
            print(f"{'='*60}")

        all_proposed_cards = enrich_cards_with_duplicate_flags(
            all_proposed_cards,
            cards,  # Full existing cards list from Anki
            threshold=args.post_dedup_threshold,
            semantic_model=args.semantic_model,
            cache_dir=cache_dir,
            verbose=args.verbose,
        )

        # Count flagged duplicates for summary
        dup_count = sum(
            1 for c in all_proposed_cards
            if c.get("duplicate_flags", {}).get("is_likely_duplicate", False)
        )
        chunk_stats["post_dedup_flagged"] = dup_count

        if args.verbose:
            print(f"✓ Post-dedup complete: {dup_count}/{len(all_proposed_cards)} cards flagged as likely duplicates")

    if not args.dry_run:
        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Generating output files...")
            print(f"{'='*60}")

        # Generate output files (conversation mode)
        json_path = run_dir / "all_proposed_cards.json"
        json_path.write_text(json.dumps(all_proposed_cards, indent=2))
        if args.verbose:
            print(f"✓ JSON cards saved to: {json_path}")
        else:
            print(f"JSON cards saved to: {json_path}")

        # Update state (conversation-level tracking)
        if args.verbose:
            print(f"\nUpdating state file...")
        state_tracker.add_conversation_ids(new_conversation_ids)
        for file_path in processed_files:
            file_cards = sum(
                1 for card in all_proposed_cards
                if any(
                    conv.source_path == str(file_path)
                    and conv.conversation_id == card.get("conversation_id")
                    for conv in conversations
                )
            )
            state_tracker.mark_file_processed(file_path, file_cards)
        state_tracker.record_run(run_dir, len(new_conversation_ids), len(new_conversation_ids))
        state_tracker.save()

        if args.verbose:
            print(f"✓ State updated")
            print(f"\n{'='*60}")
            print(f"SUMMARY")
            print(f"{'='*60}")
            print(f"Batches successful:      {chunk_stats['success']}/{total_chunks}")
            print(f"Batches failed:          {chunk_stats['failed']}/{total_chunks}")
            print(f"Conversations processed: {len(new_conversation_ids)}")
            print(f"Cards generated:         {chunk_stats['total_cards']}")
            if "post_dedup_flagged" in chunk_stats:
                print(f"Likely duplicates:       {chunk_stats['post_dedup_flagged']}")
            print(f"Files processed:         {len(processed_files)}")
            print(f"Run artifacts:           {run_dir}")
            if chunk_stats['failed'] > 0:
                print(f"\n⚠️  {chunk_stats['failed']} batch(es) failed - check {run_dir}")
            print(f"{'='*60}\n")
        else:
            status = "✓" if chunk_stats['failed'] == 0 else "⚠️"
            print(f"{status} Run complete: {chunk_stats['success']}/{total_chunks} batches successful")
            dup_info = ""
            if "post_dedup_flagged" in chunk_stats:
                dup_info = f" ({chunk_stats['post_dedup_flagged']} flagged as likely duplicates)"
            print(f"Generated {chunk_stats['total_cards']} cards from {len(new_conversation_ids)} conversations{dup_info}")
            if chunk_stats['failed'] > 0:
                print(f"⚠️  {chunk_stats['failed']} batch(es) failed - check {run_dir}")
            print(f"Run artifacts saved to {run_dir}")
    else:
        print(f"[dry-run] Conversations + prompts saved to {run_dir}. No state updates.")


if __name__ == "__main__":
    main()

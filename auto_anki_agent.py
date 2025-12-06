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
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from auto_anki.cards import Card, collect_decks
from auto_anki.contexts import ChatTurn, DateRangeFilter, harvest_chat_contexts
from auto_anki.codex import (
    build_codex_filter_prompt,
    build_codex_prompt,
    chunked,
    format_cards_as_markdown,
    parse_codex_response_robust,
    run_codex_exec,
    run_codex_pipeline,
)
from auto_anki.dedup import prune_contexts
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
        help=(
            "Directory to cache parsed HTML decks and semantic embedding index "
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
        help="Optional model override passed to codex exec via --model (default: gpt-5.1).",
    )
    parser.add_argument(
        "--model-reasoning-effort",
        default=None,
        help=(
            "Set Codex config model_reasoning_effort. "
            "If unset, defaults to 'medium' in single-stage mode, "
            "or 'low' (stage 1) / 'high' (stage 2) when --two-stage is enabled."
        ),
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
    if args.min_score < 0:
        parser.error("--min-score must be non-negative.")
    if args.max_chat_files <= 0:
        parser.error("--max-chat-files must be positive.")
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


def main() -> None:
    # CLI args + optional config file
    args = parse_args()
    config, config_path = _load_config()
    config_root = config_path.parent if config_path else None

    def resolve_path(value: Optional[str], default: Path) -> Path:
        if not value:
            return default
        path = Path(value).expanduser()
        if not path.is_absolute() and config_root is not None:
            path = config_root / path
        return path

    # Paths (CLI overrides config; config overrides hardcoded defaults)
    # Deck glob is a pattern, so we handle it separately from plain paths.
    if args.deck_glob:
        deck_glob = args.deck_glob
    elif "deck_glob" in config:
        pattern = str(config["deck_glob"])
        if config_root is not None and not os.path.isabs(pattern):
            deck_glob = str((config_root / pattern))
        else:
            deck_glob = pattern
    else:
        # If a config file exists but didn't specify deck_glob, default to
        # "*.html" in the config directory; otherwise fall back to SCRIPT_DIR.
        base = config_root or SCRIPT_DIR
        deck_glob = str(base / "*.html")
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

    # Process chunks (optionally with two-stage pipeline)
    new_seen_ids: List[str] = []
    all_proposed_cards: List[Dict[str, Any]] = []
    processed_files: set[Path] = set()
    chunk_stats: Dict[str, Any] = {
        "success": 0,
        "failed": 0,
        "total_cards": 0,
        "stage1_kept": 0,
        "stage1_total": 0,
    }

    total_chunks = (len(contexts) + args.contexts_per_run - 1) // args.contexts_per_run
    if args.verbose:
        print(f"\nProcessing {len(contexts)} contexts in {total_chunks} chunk(s) of {args.contexts_per_run}...")

    for idx, chunk in enumerate(chunked(contexts, args.contexts_per_run), start=1):
        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Chunk {idx}/{total_chunks}: Processing {len(chunk)} contexts")
            print(f"{'='*60}")

        # Optionally run stage-1 filter
        filtered_chunk = chunk
        if args.two_stage:
            if args.verbose:
                print("  Stage 1: filtering contexts before card generation...")

            filter_prompt = build_codex_filter_prompt(chunk, args)

            if args.dry_run:
                # Save stage-1 prompt and a generic stage-2 prompt template
                prompt_stage1_path = run_dir / f"prompt_stage1_chunk_{idx:02d}.txt"
                prompt_stage1_path.write_text(filter_prompt)
                prompt_stage2_path = run_dir / f"prompt_chunk_{idx:02d}.txt"
                stage2_preview = build_codex_prompt(cards_for_prompt, chunk, args)
                prompt_stage2_path.write_text(stage2_preview)
                if args.verbose:
                    print(f"[dry-run] Saved stage-1 prompt for chunk {idx} at {prompt_stage1_path}")
                    print(f"[dry-run] Saved stage-2 preview prompt for chunk {idx} at {prompt_stage2_path}")
                # Skip actual codex calls in dry-run
                continue

            try:
                stage1_reasoning = args.model_reasoning_effort or "low"
                response_text_stage1 = run_codex_exec(
                    filter_prompt,
                    idx,
                    run_dir,
                    args,
                    model_override=args.codex_model_stage1,
                    reasoning_override=stage1_reasoning,
                    label="_stage1",
                )

                if args.verbose:
                    print(f"✓ Received response from stage-1 codex (chunk {idx})")
                    print("  Parsing JSON response for filter decisions...")

                response_json_stage1 = parse_codex_response_robust(
                    response_text_stage1, idx, run_dir, args.verbose, label="_stage1"
                )

                if response_json_stage1 is None:
                    print(f"⚠️  Stage-1 parsing FAILED for chunk {idx}; falling back to sending all contexts to stage 2.")
                    kept_ids: Optional[set[str]] = None
                else:
                    decisions = response_json_stage1.get("filter_decisions", [])
                    kept_ids = {
                        d.get("context_id")
                        for d in decisions
                        if isinstance(d, dict) and d.get("keep") is True and d.get("context_id")
                    }
                    total_decisions = len(decisions)
                    kept_count = len(kept_ids)
                    chunk_stats["stage1_kept"] += kept_count
                    chunk_stats["stage1_total"] += max(len(chunk), total_decisions or len(chunk))
                    if args.verbose:
                        print(f"  Stage-1 kept {kept_count}/{len(chunk)} contexts for stage 2.")

                if kept_ids:
                    filtered_chunk = [ctx for ctx in chunk if ctx.context_id in kept_ids]
                else:
                    filtered_chunk = chunk

                if not filtered_chunk:
                    if args.verbose:
                        print(f"  Stage-1 filtered out all contexts in chunk {idx}; skipping stage 2.")
                    for ctx in chunk:
                        new_seen_ids.append(ctx.context_id)
                        processed_files.add(Path(ctx.source_path))
                    continue

            except Exception as e:
                # Stage-1 failure: log and fall back to single-stage behaviour
                error_file = run_dir / f"codex_stage1_ERROR_chunk_{idx:02d}.txt"
                error_file.write_text(f"Unexpected error in stage-1 for chunk {idx}:\n\n{str(e)}")
                print(f"⚠️  Stage-1 ERROR for chunk {idx}: {str(e)[:100]}")
                print(f"   Error saved to: {error_file}")
                print("   Falling back to sending all contexts to stage 2.")
                filtered_chunk = chunk

        # Stage-2: full card generation
        prompt = build_codex_prompt(cards_for_prompt, filtered_chunk, args)
        if args.dry_run:
            prompt_path = run_dir / f"prompt_chunk_{idx:02d}.txt"
            prompt_path.write_text(prompt)
            if args.verbose:
                print(f"[dry-run] Saved prompt for chunk {idx} at {prompt_path}")
            continue

        if args.verbose:
            print(f"Calling codex exec for chunk {idx} (stage 2)...")
            print(f"  (This may take 30-60 seconds depending on model and context size)")

        try:
            # Choose appropriate model and reasoning effort for stage 2
            stage2_model = args.codex_model_stage2 if args.two_stage else None
            if args.two_stage:
                stage2_reasoning = args.model_reasoning_effort or "high"
            else:
                stage2_reasoning = args.model_reasoning_effort or "medium"

            response_text = run_codex_exec(
                prompt,
                idx,
                run_dir,
                args,
                model_override=stage2_model,
                reasoning_override=stage2_reasoning,
                label="",
            )

            if args.verbose:
                print(f"✓ Received response from codex (chunk {idx})")
                print(f"  Parsing JSON response...")

            # Use robust multi-strategy parser
            response_json = parse_codex_response_robust(
                response_text, idx, run_dir, args.verbose, label=""
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

            for ctx in filtered_chunk:
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

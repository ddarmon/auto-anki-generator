"""
Codex / LLM integration: prompt builders, two-stage pipeline, and
response parsing helpers.

This module provides the interface for executing prompts via different
LLM backends (Codex, Claude Code, etc.) and processing their responses.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from json_repair import repair_json

from auto_anki.cards import Card
from auto_anki.contexts import ChatTurn, Conversation
from auto_anki.config_types import LLMPipelineConfig
from auto_anki.llm_backends import LLMConfig, get_backend, run_backend


# Maximum characters for user prompts in LLM prompts.
# Prompts longer than this are truncated to prevent context window overflow.
# This handles cases where users paste entire HTML documents, PDFs, etc.
MAX_USER_PROMPT_CHARS = 4000


def truncate_mega_prompt(text: str, max_chars: int = MAX_USER_PROMPT_CHARS) -> str:
    """Truncate mega-prompts while preserving context.

    When users paste large documents (HTML, code, etc.), we truncate to:
    - First ~75% of allowed chars (to show what was pasted)
    - A truncation notice with character count
    - Last ~25% of allowed chars (to show the end/question)

    This preserves enough context for the LLM to judge quality while
    preventing context window overflow.
    """
    if len(text) <= max_chars:
        return text

    # Calculate split points
    head_chars = int(max_chars * 0.70)
    tail_chars = int(max_chars * 0.20)
    removed = len(text) - head_chars - tail_chars

    truncation_notice = f"\n\n[TRUNCATED: {removed:,} characters removed]\n\n"

    return text[:head_chars] + truncation_notice + text[-tail_chars:]


def build_codex_prompt(
    cards: List[Card],
    contexts: List[ChatTurn],
    config: LLMPipelineConfig,
) -> str:
    """
    DEPRECATED: per-turn Codex pipeline has been replaced by the
    conversation-level pipeline driven by `build_conversation_prompt`.
    This function is kept only for backward compatibility and should
    not be used in new code.
    """
    raise RuntimeError(
        "build_codex_prompt is deprecated; use the conversation-level pipeline instead."
    )
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

        1. **Understand First, Memorize Second**: First form a coherent mental model of the topic.
           The generated cards should reflect this understanding, flowing logically from general
           to specific concepts. For large topics, it is good to start with one or two high-level
           summary cards before more detailed ones.
        2. **Build Upon the Basics**: Order cards logically: foundational concepts and definitions
           first, details and edge cases later. Each card must be self-contained but should also fit
           into this logical progression.
        3. **Match the Existing Deck Style**: Prefer natural-language question fronts and short,
           focused explanations on the back, as in a hand-crafted Q/A deck.

        ## The Golden Rule: Minimum Information Principle

        The most important rule: **Each card must isolate the smallest useful piece of information.**
        - One main idea per card
        - Questions should be precise and unambiguous
        - Answer length should scale with concept difficulty:
          - Simple facts (names, acronyms, commands): a word or short phrase is fine
          - Definitions and concepts: 1–3 sentences
          - Derivations, proofs, or multi-step explanations: as long as needed for clarity,
            but prefer splitting into multiple cards when possible
        - Use very small lists (2–4 items) only when the relationship between the items is the key concept
        - NEVER ask the user to recall long unordered lists; break them into multiple cards instead

        ## Front Style (Question Side)

        - Default to natural-language questions ("What…", "How…", "Why…", "Where…", "When…", "In X, what is…").
        - Use "Where is..." for cards about locating documents, files, configurations, or resources.
        - Use "What does X stand for?" for acronym expansion cards.
        - Only use non-question fronts when the natural form is "concept → definition"
          (e.g., a term and its meaning).
        - Optimize wording: remove filler words and keep the question as short and clear as possible.
        - For ambiguous acronyms or overloaded terms, add a brief context tag at the start, e.g.
          "(Biochemistry) What does GRE stand for?"
        - Avoid referring to the chat or the model (no "in the conversation above" or "the assistant said").
        - Make the front fully self-contained so it still makes sense outside this transcript.

        ## Back Style (Answer Side)

        - Give a concise, self-contained explanation that directly answers the front.
        - Lead with the key idea or definition, then add a brief elaboration or example if helpful.
        - Small numbered lists are fine for 2–4 clear steps or items.
        - Use vivid, concrete examples or mnemonics sparingly to make abstract ideas memorable.
        - When two concepts are easily confused, include cards that explicitly distinguish them
          (combat interference).
        - For cards derived from specific sources, include a reference line at the end:
          "**Reference:** path/to/source" or "**Reference:** URL"
        - For volatile facts (statistics, time-sensitive data), include a brief source or date when
          available, e.g. "(Source: US Census Bureau, 2021)".

        ## Math and Notation

        - ALWAYS typeset mathematical expressions using LaTeX, not plain ASCII math.
          When you see formulas like `x^2 + y^2 = z^2`, `sum_{t=1}^T C_t / (1 + r)^t`,
          or `P(X = x_0) = 0`, rewrite them as LaTeX, e.g. `\\(x^2 + y^2 = z^2\\)` or
          `\\(\\sum_{t=1}^T C_t / (1 + r)^t\\)`.
        - Use inline math `\\( ... \\)` for inline formulas and display math
          `\\[ ... \\]` for equations or formulas that should stand on their own lines.
        - Use LaTeX-like `\\mathbf{A}` for Roman letter vectors/matrices and
          `\\boldsymbol{\\alpha}` for Greek letter vectors/matrices when appropriate.
        - Do NOT wrap math or answers in code fences.

        ## Difficult Information Types

        - **Unordered sets (lists)**: NEVER ask the user to list more than 2–3 items. Instead,
          create one card per item or per logically grouped subset.
        - **Ordered lists / processes**: Prefer multiple small cards over a single big
          "list all the steps" card.
        - **Visual concepts**: If a picture would meaningfully aid understanding, you may add a
          short placeholder hint in the back like "[Image: diagram of a plant cell]".

        ## Formatting Inside Cards

        - Use **bold** (`**text**`) for keywords, definitions, or the precise part of the answer
          to be recalled.
        - Use `inline code` (`` `text` ``) for function names, variables, commands, or short code.
        - Use code blocks only when multi-line code is central to the concept.
        - Use blockquotes (`> text`) for direct quotes or important statements, when appropriate.
        - When introducing an acronym, consider writing the full term first, e.g.
          "**Application Programming Interface (API)**".

        ## Card Format

        Use **Question/Answer (basic)** format: Question on the front, explanation on the back.
        This is the only format used in this deck.

        ## Your Task

        For each `candidate_context`:
        1. Decide if it contains learning-worthy knowledge (not trivial, not already covered).
        2. Check against `existing_cards` to avoid duplicates.
        3. If justified, create one or MORE atomic cards following the guidelines above.
        4. Choose an appropriate deck. Tags are optional—only add them when genuinely useful
           for filtering (most cards need no tags).
        5. Provide a confidence score and brief notes explaining why the card matters or how it
           might be used.

        Before finalizing a card, mentally check:
        - Does it follow the Minimum Information Principle?
        - Is the question clear, specific, and unambiguous?
        - Can the card be understood independently of the original conversation?
        - Is the wording optimized for fast recall (no unnecessary words)?

        ## Output Requirements

        Return ONLY valid JSON adhering to `output_contract`. Critical rules:
        - NO markdown fencing (no ```json blocks)
        - NO explanatory text before or after the JSON
        - NO comments inside the JSON
        - START your response with `{` and END with `}`
        - `card_style` should be: basic
        - `confidence`: 0.0-1.0

        YOUR ENTIRE RESPONSE MUST BE VALID, PARSEABLE JSON.
        """
    ).strip()
    return instructions + "\n\n" + json.dumps(payload, indent=2, ensure_ascii=False)


def build_codex_filter_prompt(
    contexts: List[ChatTurn],
    config: LLMPipelineConfig,
) -> str:
    """
    Build the stage-1 filtering prompt for the two-stage pipeline.

    Stage 1 should be fast and cheap: it only decides which contexts
    are worth sending to the more expensive card-generation stage.
    """
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
        "filter_decisions": [
            {
                "context_id": "string",
                "keep": "boolean (true if this context should be sent to the expensive card-generation stage)",
                "reason": "short reason for keeping or skipping this context",
            }
        ]
    }
    payload = {
        "candidate_contexts": contexts_payload,
        "output_contract": contract,
    }
    instructions = textwrap.dedent(
        """
        CRITICAL: You MUST respond with ONLY valid JSON matching the output_contract below.
        Do NOT include markdown, explanations, or any text outside the JSON structure.
        Do NOT wrap the JSON in ```json blocks.

        You are the fast, cheap *filtering* stage of an autonomous spaced-repetition agent.

        ## Your Goal

        Quickly decide which candidate contexts are worth sending to a slower, more
        expensive card-generation model.

        Focus on:
        - Educational value (clear concepts, stable knowledge)
        - Clarity and structure (good explanations, examples, lists)
        - Non-trivial content (avoid obvious, shallow, or throwaway exchanges)

        You DO NOT generate cards here.

        ## For Each `candidate_context`

        1. Decide if it contains learning-worthy content.
        2. Set `keep = true` if it should be passed to the card-generation stage,
           otherwise `keep = false`.
        3. Provide a short `reason`.

        YOUR ENTIRE RESPONSE MUST BE VALID, PARSEABLE JSON.
        """
    ).strip()
    return instructions + "\n\n" + json.dumps(payload, indent=2, ensure_ascii=False)


def run_codex_exec(
    prompt: str,
    chunk_idx: int,
    run_dir: Path,
    config: LLMPipelineConfig,
    *,
    model_override: Optional[str] = None,
    reasoning_override: Optional[str] = None,
    label: str = "",
) -> str:
    """Execute a prompt via the configured LLM backend.

    This function provides backward compatibility while using the new
    backend abstraction layer internally.

    Args:
        prompt: The prompt text to send to the LLM.
        chunk_idx: Chunk index for labeling artifacts.
        run_dir: Directory for saving run artifacts.
        config: Typed configuration.
        model_override: Optional model override.
        reasoning_override: Optional reasoning effort override.
        label: Optional label for artifact filenames.

    Returns:
        The raw response text from the LLM.

    Raises:
        RuntimeError: If the LLM execution fails.
    """
    # Get the backend (default to codex for backward compatibility)
    backend_name = config.llm_backend or "codex"
    backend = get_backend(backend_name)

    # Build config from args, allowing overrides
    # For codex: use codex_model; for claude-code: use llm_model
    if backend_name == "codex":
        default_model = config.codex_model or "gpt-5.1"
    else:
        default_model = config.llm_model

    model = model_override or default_model
    reasoning = reasoning_override or config.model_reasoning_effort

    # Collect extra args
    extra_args: List[str] = []
    for extra in config.codex_extra_arg or []:
        if extra:
            extra_args.extend(shlex.split(extra))
    for extra in config.llm_extra_arg or []:
        if extra:
            extra_args.extend(shlex.split(extra))

    config = LLMConfig(
        model=model,
        reasoning_effort=reasoning,
        extra_args=extra_args,
    )

    # Execute via backend
    response = run_backend(
        backend,
        prompt,
        config,
        run_dir,
        label=f"{label}_chunk_{chunk_idx:02d}",
    )

    if not response.success:
        raise RuntimeError(
            f"{backend.name} failed for chunk {chunk_idx}: {response.error_message}"
        )

    return response.raw_text


def parse_codex_response_robust(
    response_text: str,
    chunk_idx: int,
    run_dir: Path,
    verbose: bool = False,
    label: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Parse codex JSON response with multiple fallback strategies.
    Returns None if all strategies fail.
    """
    # Save raw response
    raw_path = run_dir / f"codex{label}_raw_response_chunk_{chunk_idx:02d}.txt"
    raw_path.write_text(response_text)

    def strip_markdown_fences(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def raw_decode_first_json(text: str) -> Any:
        decoder = json.JSONDecoder()
        last_error: Optional[Exception] = None
        for start, ch in enumerate(text):
            if ch not in ("{", "["):
                continue
            try:
                obj, _ = decoder.raw_decode(text[start:])
                return obj
            except json.JSONDecodeError as e:
                last_error = e
                continue
        raise last_error or json.JSONDecodeError("No JSON object found", text, 0)

    def escape_invalid_backslashes(text: str) -> str:
        """
        JSON only permits escapes for \" \\ / b f n r t uXXXX.
        Some model outputs include LaTeX-like sequences such as \\( ... \\),
        which need the backslash doubled to become valid JSON.
        """
        return re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", text)

    cleaned = strip_markdown_fences(response_text)
    escaped = escape_invalid_backslashes(cleaned)
    raw = response_text.strip()

    strategies: List[tuple[str, Any]] = []

    # Strategy 1: Direct parse (only when the response already looks like JSON).
    if raw[:1] in ("{", "["):
        strategies.append(("Direct parse", lambda: json.loads(raw)))

    # Strategy 2: Strip markdown fences (common when LLM wraps JSON in ```json blocks).
    strategies.append(("Markdown stripped", lambda: json.loads(cleaned)))

    # Strategy 2b: Escape invalid backslashes (common when LaTeX math uses \\( \\)).
    strategies.append(("Backslash escaped", lambda: json.loads(escaped)))

    # Strategy 3: Extract a JSON value from a response with extra leading/trailing text.
    strategies.append(("JSON extracted", lambda: raw_decode_first_json(cleaned)))

    # Strategy 4: Use json-repair library (last resort).
    try:
        repaired = repair_json(cleaned)
        strategies.append(("JSON repair", lambda: json.loads(repaired)))
    except Exception:
        pass  # json-repair failed, skip this strategy

    last_error: Optional[Exception] = None

    # Try each strategy
    for strategy_name, strategy in strategies:
        try:
            result = strategy()
            if verbose:
                print(f"  ✓ Parsed with strategy: {strategy_name}")
            # Save the working version
            (run_dir / f"codex{label}_parsed_response_chunk_{chunk_idx:02d}.json").write_text(
                json.dumps(result, indent=2)
            )
            if isinstance(result, dict):
                return result
            return None
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            last_error = e
            continue

    if verbose:
        print(f"  ✗ Failed to parse JSON response: {last_error or 'Unknown error'}")

    # All strategies failed - save debug info
    error_file = run_dir / f"codex{label}_FAILED_chunk_{chunk_idx:02d}.txt"
    error_info = f"""All JSON parsing strategies failed for chunk {chunk_idx}

Strategies tried:
{chr(10).join(f'- {name}' for name, _ in strategies)}

First 1000 chars of response:
{response_text[:1000]}

Last error: {last_error or 'Unknown'}
"""
    error_file.write_text(error_info)

    return None


def chunked(seq: Sequence[Any], size: int) -> Iterable[List[Any]]:
    for start in range(0, len(seq), size):
        yield list(seq[start : start + size])


def run_codex_pipeline(
    cards_for_prompt: List[Card],
    contexts: List[ChatTurn],
    config: LLMPipelineConfig,
    state_tracker: Any,
    run_dir: Path,
    output_dir: Path,
    run_timestamp: str,
) -> None:
    """
    Execute the (optional) two-stage Codex pipeline over the given contexts.

    This handles:
    - Chunking contexts
    - Optional stage-1 filtering
    - Stage-2 card generation
    - Writing run artifacts (prompts, responses, outputs)
    - Updating `state_tracker`
    """
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

    total_chunks = (len(contexts) + config.contexts_per_run - 1) // config.contexts_per_run
    if config.verbose:
        print(
            f"\nProcessing {len(contexts)} contexts in {total_chunks} chunk(s) of {config.contexts_per_run}..."
        )

    for idx, chunk in enumerate(chunked(contexts, config.contexts_per_run), start=1):
        if config.verbose:
            print(f"\n{'='*60}")
            print(f"Chunk {idx}/{total_chunks}: Processing {len(chunk)} contexts")
            print(f"{'='*60}")

        # Optionally run stage-1 filter
        filtered_chunk = chunk
        if config.two_stage:
            if config.verbose:
                print("  Stage 1: filtering contexts before card generation...")

            filter_prompt = build_codex_filter_prompt(chunk, config)

            if config.dry_run:
                # Save stage-1 prompt and a generic stage-2 prompt template
                prompt_stage1_path = run_dir / f"prompt_stage1_chunk_{idx:02d}.txt"
                prompt_stage1_path.write_text(filter_prompt)
                prompt_stage2_path = run_dir / f"prompt_chunk_{idx:02d}.txt"
                stage2_preview = build_codex_prompt(cards_for_prompt, chunk, config)
                prompt_stage2_path.write_text(stage2_preview)
                if config.verbose:
                    print(
                        f"[dry-run] Saved stage-1 prompt for chunk {idx} at {prompt_stage1_path}"
                    )
                    print(
                        f"[dry-run] Saved stage-2 preview prompt for chunk {idx} at {prompt_stage2_path}"
                    )
                # Skip actual codex calls in dry-run
                continue

            try:
                stage1_reasoning = config.model_reasoning_effort or "low"
                response_text_stage1 = run_codex_exec(
                    filter_prompt,
                    idx,
                    run_dir,
                    config,
                    model_override=config.codex_model_stage1,
                    reasoning_override=stage1_reasoning,
                    label="_stage1",
                )

                if config.verbose:
                    print(
                        f"✓ Received response from stage-1 codex (chunk {idx})"
                    )
                    print("  Parsing JSON response for filter decisions...")

                response_json_stage1 = parse_codex_response_robust(
                    response_text_stage1,
                    idx,
                    run_dir,
                    args.verbose,
                    label="_stage1",
                )

                if response_json_stage1 is None:
                    print(
                        f"⚠️  Stage-1 parsing FAILED for chunk {idx}; falling back to sending all contexts to stage 2."
                    )
                    filtered_chunk = chunk
                else:
                    decisions = {
                        d["context_id"]: d
                        for d in response_json_stage1.get("filter_decisions", [])
                        if "context_id" in d
                    }
                    kept: List[ChatTurn] = []
                    for c in chunk:
                        decision = decisions.get(c.context_id)
                        keep = True
                        if decision is not None:
                            keep = bool(decision.get("keep", True))
                        if keep:
                            kept.append(c)

                    chunk_stats["stage1_total"] += len(chunk)
                    chunk_stats["stage1_kept"] += len(kept)

                    if config.verbose:
                        print(
                            f"  Stage-1 kept {len(kept)}/{len(chunk)} contexts for stage 2."
                        )

                    filtered_chunk = kept or chunk

            except Exception as e:
                print(
                    f"⚠️  Stage-1 ERROR for chunk {idx}: {str(e)[:100]} - falling back to sending all contexts to stage 2."
                )
                filtered_chunk = chunk

        # Track ALL files from the original chunk as "processed" (reviewed by Stage 1)
        # This ensures files where Stage 1 skipped all conversations are still marked
        # as processed and won't be re-sent on subsequent --unprocessed-only runs
        for ctx in chunk:
            processed_files.add(Path(ctx.source_path))

        # Stage 2: card generation
        if not filtered_chunk:
            if config.verbose:
                print(
                    f"  No contexts to send to stage 2 after filtering for chunk {idx}."
                )
            continue

        try:
            prompt = build_codex_prompt(cards_for_prompt, filtered_chunk, config)

            if config.dry_run:
                prompt_path = run_dir / f"prompt_chunk_{idx:02d}.txt"
                prompt_path.write_text(prompt)
                if config.verbose:
                    print(
                        f"[dry-run] Saved stage-2 prompt for chunk {idx} at {prompt_path}"
                    )
                continue

            stage2_reasoning = (
                config.model_reasoning_effort or ("high" if config.two_stage else "medium")
            )
            response_text = run_codex_exec(
                prompt,
                idx,
                run_dir,
                config,
                model_override=config.codex_model_stage2 or config.codex_model,
                reasoning_override=stage2_reasoning,
                label="",
            )

            if config.verbose:
                print(f"✓ Received response from stage-2 codex (chunk {idx})")
                print(f"  Parsing JSON response...")

            # Use robust multi-strategy parser
            response_json = parse_codex_response_robust(
                response_text, idx, run_dir, config.verbose, label=""
            )

            if response_json is None:
                # Parsing failed - log and continue
                chunk_stats["failed"] += 1
                print(
                    f"⚠️  Chunk {idx} FAILED: Could not parse JSON response"
                )
                print(
                    f"   Debug info saved to: {run_dir}/codex_FAILED_chunk_{idx:02d}.txt"
                )
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

            if config.verbose:
                print(f"✓ Chunk {idx} complete:")
                print(f"    {cards_in_chunk} cards proposed")
                print(f"    {skipped_in_chunk} contexts skipped")

            # Track context IDs for contexts that made it through to Stage 2
            # (Files are already tracked earlier from the original chunk)
            for ctx in filtered_chunk:
                new_seen_ids.append(ctx.context_id)

        except Exception as e:
            # Unexpected error - log and continue
            chunk_stats["failed"] += 1
            error_file = run_dir / f"codex_ERROR_chunk_{idx:02d}.txt"
            error_file.write_text(
                f"Unexpected error processing chunk {idx}:\n\n{str(e)}"
            )
            print(f"⚠️  Chunk {idx} ERROR: {str(e)[:100]}")
            print(f"   Error saved to: {error_file}")
            print(f"   Continuing with remaining chunks...")
            continue

    if config.dry_run:
        print(f"[dry-run] Contexts + prompts saved to {run_dir}. No state updates.")
        return

    # Generate output files
    if config.verbose:
        print(f"\n{'='*60}")
        print("Generating output files...")
        print(f"{'='*60}")

    json_path = run_dir / "all_proposed_cards.json"
    json_path.write_text(json.dumps(all_proposed_cards, indent=2))
    if config.verbose:
        print(f"✓ JSON cards saved to: {json_path}")
    else:
        print(f"JSON cards saved to: {json_path}")

    # Update state
    if config.verbose:
        print(f"\nUpdating state file...")
    state_tracker.add_context_ids(new_seen_ids)
    for file_path in processed_files:
        file_cards = sum(
            1
            for card in all_proposed_cards
            if any(
                ctx.source_path == str(file_path)
                and ctx.context_id == card.get("context_id")
                for ctx in contexts
            )
        )
        state_tracker.mark_file_processed(file_path, file_cards)
    state_tracker.record_run(run_dir, len(new_seen_ids))
    state_tracker.save()

    if config.verbose:
        print(f"✓ State updated")
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Chunks successful:     {chunk_stats['success']}/{total_chunks}")
        print(f"Chunks failed:         {chunk_stats['failed']}/{total_chunks}")
        print(f"Contexts processed:    {len(new_seen_ids)}")
        print(f"Cards generated:       {chunk_stats['total_cards']}")
        print(f"Files processed:       {len(processed_files)}")
        print(f"Run artifacts:         {run_dir}")
        if chunk_stats["failed"] > 0:
            print(
                f"\n⚠️  {chunk_stats['failed']} chunk(s) failed - check {run_dir} for details"
            )
        print(f"{'='*60}\n")
    else:
        status = "✓" if chunk_stats["failed"] == 0 else "⚠️"
        print(
            f"{status} Run complete: {chunk_stats['success']}/{total_chunks} chunks successful"
        )
        print(
            f"Generated {chunk_stats['total_cards']} proposed cards from {len(new_seen_ids)} contexts"
        )
        if chunk_stats["failed"] > 0:
            print(
                f"⚠️  {chunk_stats['failed']} chunk(s) failed - check {run_dir} for details"
            )
        print(f"Run artifacts saved to {run_dir}")


def build_conversation_prompt(
    conversations: List[Conversation],
    config: LLMPipelineConfig,
) -> str:
    """Build the conversation-level card generation prompt.

    Unlike the per-turn prompt, this sends full conversations so the LLM can:
    - Understand the user's learning journey
    - See follow-up questions that indicate confusion
    - Avoid cards from exchanges that were later corrected
    - Create coherent card sets that build on each other

    Note: Duplicate detection is handled post-generation via semantic similarity,
    so existing cards are NOT included in this prompt. The LLM focuses purely on
    generating high-quality cards from the conversation content.

    User prompts are truncated to MAX_USER_PROMPT_CHARS to prevent context
    window overflow from mega-pastes (users pasting entire HTML docs, etc.).
    """
    conversations_payload = [
        {
            "conversation_id": conv.conversation_id,
            "source_title": conv.source_title,
            "source_url": conv.source_url,
            "key_topics": conv.key_topics,
            "aggregate_score": round(conv.aggregate_score, 3),
            "aggregate_signals": conv.aggregate_signals,
            "turns": [
                {
                    "turn_index": turn.turn_index,
                    "context_id": turn.context_id,
                    "user_timestamp": turn.user_timestamp,
                    "user_prompt": truncate_mega_prompt(turn.user_prompt),
                    "assistant_answer": turn.assistant_answer,
                    "score": round(turn.score, 3),
                    "signals": turn.signals,
                }
                for turn in conv.turns
            ],
        }
        for conv in conversations
    ]

    contract = {
        "cards": [
            {
                "conversation_id": "string (link to parent conversation)",
                "turn_index": "int (which turn this card is based on, 0-indexed)",
                "context_id": "string (per-turn ID for backward compat)",
                "deck": "string",
                "card_style": "basic|cloze|list",
                "front": "string",
                "back": "string",
                "tags": ["list", "of", "tags"],
                "confidence": "0-1 float",
                "notes": "why this card matters",
                "depends_on": ["optional list of context_ids this card builds upon"],
            }
        ],
        "skipped_conversations": [
            {
                "conversation_id": "string",
                "reason": "why the entire conversation was skipped",
            }
        ],
        "skipped_turns": [
            {
                "conversation_id": "string",
                "turn_index": "int",
                "reason": "why this specific turn was skipped",
            }
        ],
        "learning_insights": [
            {
                "conversation_id": "string",
                "insight": "what the user was trying to learn",
                "misconceptions_corrected": ["things user initially misunderstood"],
            }
        ],
    }

    # Get available decks from typed config (set in main() from config or CLI)
    available_decks = config.decks or []

    payload = {
        "available_decks": available_decks,
        "candidate_conversations": conversations_payload,
        "output_contract": contract,
    }
    instructions = textwrap.dedent(
        """
        CRITICAL: You MUST respond with ONLY valid JSON matching the output_contract below.
        Do NOT include markdown, explanations, or any text outside the JSON structure.
        Do NOT wrap the JSON in ```json blocks.

        You are operating as the decision layer of an autonomous spaced-repetition agent.

        ## Available Decks

        You MUST assign each card to one of the following decks (use EXACT names):
        {available_decks_list}

        Choose the deck that best matches the card's subject matter.

        ## Conversation-Level Analysis

        You receive **full conversations** instead of isolated exchanges. This enables you to:
        1. See how the user's understanding evolved across turns
        2. Identify follow-up questions (user struggled, needed clarification)
        3. Skip early turns if they contain information that was later corrected
        4. Create coherent card sets that build on each other

        ## Core Philosophy

        1. **Understand First, Memorize Second**: First form a coherent mental model of the topic.
           Cards should reflect this understanding, flowing logically from general to specific.
        2. **Build Upon the Basics**: Order cards logically: foundational concepts and definitions
           first, details and edge cases later. Each card must be self-contained but also fit into
           a sensible progression within its conversation.
        3. **Match the Existing Deck Style**: Prefer natural-language question fronts and short,
           focused explanations on the back, like a carefully hand-crafted Q/A deck.

        ## Minimum Information Principle

        The most important rule: **Each card must isolate the smallest useful piece of information.**
        - One main idea per card
        - Questions should be precise and unambiguous
        - Answer length should scale with concept difficulty:
          - Simple facts (names, acronyms, commands): a word or short phrase is fine
          - Definitions and concepts: 1–3 sentences
          - Derivations, proofs, or multi-step explanations: split into multiple cards when possible
        - Use very small lists (2–4 items) only when the relationship between the items is the key concept
        - NEVER ask the user to recall long unordered lists; break them into multiple cards instead

        ## Guidelines for Processing Conversations

        - **Read the entire conversation first** before deciding on cards.
        - **Prioritize final understanding** over intermediate confusion.
        - Use the `turn_index` to link cards to specific exchanges.
        - Use `depends_on` to indicate card ordering for learning when cards naturally build on each other.
        - Focus on extracting valuable knowledge - duplicate detection is handled separately.

        ## Red Flags to Skip

        - User says "wait, that's wrong" or "actually I misunderstood"
        - Assistant corrects earlier information ("I should clarify...")
        - Conversation degenerates into debugging without resolution
        - Final exchange shows the user still confused

        ## Green Flags for High-Quality Cards

        - Clear progression from question to understanding
        - User successfully applies the concept ("Oh, so it's like...")
        - Multiple related concepts explained coherently
        - Concrete examples that crystallize understanding

        ## Front Style (Question Side)

        - Default to natural-language questions ("What…", "How…", "Why…", "Where…", "When…", "In X, what is…").
        - Use "Where is..." for cards about locating documents, files, configurations, or resources.
        - Use "What does X stand for?" for acronym expansion cards.
        - Only use non-question fronts when the natural form is "concept → definition" (e.g., a term and its meaning).
        - Optimize wording: remove filler words and keep the question as short and clear as possible.
        - For ambiguous acronyms or overloaded terms, add a brief context tag at the start, e.g.
          "(Biochemistry) What does GRE stand for?"
        - Avoid referring to the chat or the model (no "in the conversation above" or "the assistant said").
        - Make the front fully self-contained so it still makes sense outside this transcript.

        ## Back Style (Answer Side)

        - Give a concise, self-contained explanation that directly answers the front.
        - Lead with the key idea or definition, then add a brief elaboration or example if helpful.
        - Small numbered lists are fine for 2–4 clear steps or items.
        - Use vivid, concrete examples or mnemonics sparingly to make abstract ideas memorable.
        - When two concepts are easily confused, include cards that explicitly distinguish them
          (combat interference).
        - For cards derived from specific sources, include a reference line at the end:
          "**Reference:** path/to/source" or "**Reference:** URL".
        - For volatile facts (statistics, time-sensitive data), include a brief source or date when
          available, e.g. "(Source: US Census Bureau, 2021)".

        ## Math and Notation

        - ALWAYS typeset mathematical expressions using LaTeX, not plain ASCII math.
          When you see formulas like `x^2 + y^2 = z^2`, `sum_{t=1}^T C_t / (1 + r)^t`,
          or `P(X = x_0) = 0`, rewrite them as LaTeX, e.g. `\\(x^2 + y^2 = z^2\\)` or
          `\\(\\sum_{t=1}^T C_t / (1 + r)^t\\)`.
        - Use inline math `\\( ... \\)` for most formulas and display math `\\[ ... \\]`
          only when the expression itself is the main focus of the card.
        - Keep formulas embedded in explanatory sentences rather than as standalone blocks
          when possible, and avoid code fences for math.
        - You may use LaTeX like `\\mathbf{A}` for Roman letter vectors/matrices or
          `\\boldsymbol{\\alpha}` for Greek letter vectors/matrices when appropriate.

        ## Difficult Information Types

        - **Unordered sets (lists)**: NEVER ask the user to list more than 2–3 items. Instead,
          create one card per item or per logically grouped subset.
        - **Ordered lists / processes**: Prefer multiple small cards or overlapping cloze-style
          cards over a single big "list all the steps" card.
        - **Visual concepts**: If a picture would meaningfully aid understanding, you may add a
          short placeholder hint in the back like "[Image: diagram of a plant cell]".

        ## Formatting Inside Cards

        - Use **bold** (`**text**`) for keywords, definitions, or the precise part of the answer
          to be recalled.
        - Use `inline code` (`` `text` ``) for function names, variables, commands, or short code.
        - Use code blocks only when multi-line code is central to the concept.
        - Use blockquotes (`> text`) for direct quotes or important statements, when appropriate.
        - When introducing an acronym, consider writing the full term first, e.g.
          "**Application Programming Interface (API)**".

        ## Card Formats

        1. **Question/Answer (basic)**: Default for most concepts—question on the front,
           concise explanation on the back.
        2. **Cloze Deletion (cloze)**: Ideal for short facts, definitions, names, and sequences
           that can be hidden inline with `[...]` on the front and restated fully on the back.
        3. **List**: Use only when a small comparison or grouped set of 2–4 items is the main
           concept; avoid long laundry lists.

        ## Your Task

        For each `candidate_conversation`:
        1. Decide which turns contain learning-worthy knowledge (not trivial, clearly explained).
        2. If justified, create one or MORE atomic cards following the guidelines above.
        3. For each card, choose an appropriate deck, `card_style`, and (optionally) tags.
        4. Provide a confidence score and brief notes explaining why the card matters or how it
           might be used.

        Note: Duplicate detection against existing cards is handled separately after generation.
        Focus on creating high-quality cards from the conversation content.

        Before finalizing each card, mentally check:
        - Does it follow the Minimum Information Principle?
        - Is the question clear, specific, and unambiguous?
        - Can the card be understood independently of the original conversation?
        - Is the wording optimized for fast recall (no unnecessary words)?

        ## Output Requirements

        Return ONLY valid JSON adhering to `output_contract`. Critical rules:
        - NO markdown fencing (no ```json blocks)
        - NO explanatory text before or after the JSON
        - START your response with `{` and END with `}`
        - Include `conversation_id` AND `turn_index` for each card
        - `confidence`: 0.0-1.0

        YOUR ENTIRE RESPONSE MUST BE VALID, PARSEABLE JSON.
        """
    ).strip()

    # Format the deck list for the instructions (use replace to avoid issues with other {})
    if available_decks:
        decks_formatted = "\n".join(f"- {deck}" for deck in available_decks)
    else:
        decks_formatted = "- (No decks specified - use your best judgment)"
    instructions = instructions.replace("{available_decks_list}", decks_formatted)

    return instructions + "\n\n" + json.dumps(payload, indent=2, ensure_ascii=False)


def build_conversation_filter_prompt(
    conversations: List[Conversation],
    config: LLMPipelineConfig,
) -> str:
    """Build the stage-1 filter prompt for conversations.

    Stage 1 decides which conversations are worth sending to the more
    expensive card-generation stage. It receives full conversation content
    to make accurate quality decisions.

    User prompts are truncated to MAX_USER_PROMPT_CHARS to prevent context
    window overflow from mega-pastes (users pasting entire HTML docs, etc.).
    """
    conversations_payload = [
        {
            "conversation_id": conv.conversation_id,
            "source_title": conv.source_title,
            "key_topics": conv.key_topics,
            "turn_count": len(conv.turns),
            # Full conversation content for accurate quality assessment
            # User prompts are truncated to prevent context overflow
            "turns": [
                {
                    "turn_index": turn.turn_index,
                    "user_prompt": truncate_mega_prompt(turn.user_prompt),
                    "assistant_response": turn.assistant_answer,
                }
                for turn in conv.turns
            ],
        }
        for conv in conversations
    ]

    contract = {
        "filter_decisions": [
            {
                "conversation_id": "string",
                "keep": "boolean (true if worth sending to card generation)",
                "reason": "short reason for keeping or skipping",
                "estimated_cards": "integer (0-5, rough estimate of how many cards this could yield)",
            }
        ]
    }

    payload = {
        "candidate_conversations": conversations_payload,
        "output_contract": contract,
    }

    instructions = textwrap.dedent(
        """
        CRITICAL: You MUST respond with ONLY valid JSON matching the output_contract below.
        Do NOT include markdown, explanations, or any text outside the JSON structure.
        Do NOT wrap the JSON in ```json blocks.

        You are the *filtering* stage of an autonomous spaced-repetition agent.

        ## Your Goal

        Decide which conversations are worth sending to the card-generation stage.
        You have full access to conversation content - use it to judge quality directly.

        ## Keep Conversations That Have:

        - Educational value: clear explanations of concepts, definitions, or principles
        - Stable knowledge: facts that will remain true, not rapidly changing info
        - Good structure: examples, analogies, comparisons, or organized explanations
        - Learning progression: follow-up questions showing deepening understanding
        - Concrete topics: specific concepts that can become flashcards

        ## Skip Conversations That Are:

        - Trivial: obvious questions with shallow answers
        - Procedural: step-by-step debugging, troubleshooting, or coding help
        - Conversational: chit-chat, opinions, or highly context-dependent advice
        - Incomplete: unresolved questions or partial explanations
        - Ephemeral: time-sensitive info, current events, or rapidly changing details

        ## Output

        For each conversation, return:
        - keep: true/false
        - reason: brief explanation (1 sentence)
        - estimated_cards: rough estimate (0-5) of useful cards this could generate

        You DO NOT generate cards here. Just filter.

        YOUR ENTIRE RESPONSE MUST BE VALID, PARSEABLE JSON.
        """
    ).strip()

    return instructions + "\n\n" + json.dumps(payload, indent=2, ensure_ascii=False)


__all__ = [
    "build_codex_prompt",
    "build_conversation_prompt",
    "build_conversation_filter_prompt",
    "run_codex_exec",
    "parse_codex_response_robust",
    "run_codex_pipeline",
]

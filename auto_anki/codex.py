"""
Codex / LLM integration: prompt builders, two-stage pipeline, and
response parsing helpers.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from json_repair import repair_json

from auto_anki.cards import Card
from auto_anki.contexts import ChatTurn


def build_codex_prompt(
    cards: List[Card],
    contexts: List[ChatTurn],
    args: Any,
) -> str:
    """
    Build the stage-2 card generation prompt.
    """
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
        - Break down sets/lists into multiple atomic cards

        ## Card Formats

        Use the format that best fits the information:

        1. **Question/Answer (basic)**: Default format for most concepts
        2. **Cloze Deletion**: Ideal for facts, definitions, vocabulary

        ## Your Task

        For each `candidate_context`:
        1. Decide if it contains learning-worthy knowledge (not trivial, not already covered)
        2. Check against `existing_cards` to avoid duplicates
        3. If justified, create one or MORE atomic cards
        4. Choose appropriate deck and tags
        5. Provide confidence score and brief notes

        ## Output Requirements

        Return ONLY valid JSON adhering to `output_contract`. Critical rules:
        - NO markdown fencing (no ```json blocks)
        - NO explanatory text before or after the JSON
        - NO comments inside the JSON
        - START your response with `{` and END with `}`
        - `card_style` should be: basic, cloze, or list
        - `confidence`: 0.0-1.0

        YOUR ENTIRE RESPONSE MUST BE VALID, PARSEABLE JSON.
        """
    ).strip()
    return instructions + "\n\n" + json.dumps(payload, indent=2, ensure_ascii=False)


def build_codex_filter_prompt(
    contexts: List[ChatTurn],
    args: Any,
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
    args: Any,
    *,
    model_override: Optional[str] = None,
    reasoning_override: Optional[str] = None,
    label: str = "",
) -> str:
    prompt_path = run_dir / f"prompt{label}_chunk_{chunk_idx:02d}.txt"
    prompt_path.write_text(prompt)
    last_msg_path = run_dir / f"codex{label}_response_chunk_{chunk_idx:02d}.json"

    # Build command
    cmd = ["codex", "exec", "-", "--skip-git-repo-check"]
    model = model_override or args.codex_model or "gpt-5.1"
    cmd.extend(["--model", model])
    reasoning = reasoning_override or args.model_reasoning_effort
    if reasoning:
        cmd.extend(["-c", f"model_reasoning_effort={reasoning}"])

    for extra in getattr(args, "codex_extra_arg", []) or []:
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
    (run_dir / f"codex{label}_stdout_chunk_{chunk_idx:02d}.log").write_text(proc.stdout)
    (run_dir / f"codex{label}_stderr_chunk_{chunk_idx:02d}.log").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"codex exec failed for chunk {chunk_idx} with exit code {proc.returncode}"
        )
    return last_msg_path.read_text().strip()


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

    strategies: List[tuple[str, str]] = []

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

    last_error: Optional[Exception] = None

    # Try each strategy
    for strategy_name, text in strategies:
        try:
            result = json.loads(text)
            if verbose:
                print(f"  ✓ Parsed with strategy: {strategy_name}")
            # Save the working version
            (run_dir / f"codex{label}_parsed_response_chunk_{chunk_idx:02d}.json").write_text(
                json.dumps(result, indent=2)
            )
            return result
        except json.JSONDecodeError as e:
            last_error = e
            if verbose:
                print(f"  ✗ {strategy_name} failed: {e}")
            continue

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


def format_cards_as_markdown(
    cards_json: List[Dict[str, Any]],
    contexts: List[ChatTurn],
    run_timestamp: str,
) -> str:
    """
    Format proposed cards as markdown following Anki best practices.
    """
    context_map = {ctx.context_id: ctx for ctx in contexts}

    lines = [
        "# Proposed Anki Cards",
        "",
        f"Generated: {run_timestamp}",
        f"Total cards: {len(cards_json)}",
        "",
        "---",
        "",
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
        confidence = card_data.get("confidence", 0.0)
        try:
            confidence_float = float(confidence) if confidence else 0.0
            lines.append(f"**Confidence:** {confidence_float:.2f}")
        except (ValueError, TypeError):
            lines.append(f"**Confidence:** {confidence}")

        if card_data.get("tags"):
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
        lines.append(card_data.get("front", ""))
        lines.append("")

        lines.append("### Back")
        lines.append("")
        lines.append(card_data.get("back", ""))
        lines.append("")

        # Notes
        if card_data.get("notes"):
            lines.append("### Notes")
            lines.append("")
            lines.append(card_data["notes"])
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def run_codex_pipeline(
    cards_for_prompt: List[Card],
    contexts: List[ChatTurn],
    args: Any,
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

    total_chunks = (len(contexts) + args.contexts_per_run - 1) // args.contexts_per_run
    if args.verbose:
        print(
            f"\nProcessing {len(contexts)} contexts in {total_chunks} chunk(s) of {args.contexts_per_run}..."
        )

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
                    print(
                        f"[dry-run] Saved stage-1 prompt for chunk {idx} at {prompt_stage1_path}"
                    )
                    print(
                        f"[dry-run] Saved stage-2 preview prompt for chunk {idx} at {prompt_stage2_path}"
                    )
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

                    if args.verbose:
                        print(
                            f"  Stage-1 kept {len(kept)}/{len(chunk)} contexts for stage 2."
                        )

                    filtered_chunk = kept or chunk

            except Exception as e:
                print(
                    f"⚠️  Stage-1 ERROR for chunk {idx}: {str(e)[:100]} - falling back to sending all contexts to stage 2."
                )
                filtered_chunk = chunk

        # Stage 2: card generation
        if not filtered_chunk:
            if args.verbose:
                print(
                    f"  No contexts to send to stage 2 after filtering for chunk {idx}."
                )
            continue

        try:
            prompt = build_codex_prompt(cards_for_prompt, filtered_chunk, args)

            if args.dry_run:
                prompt_path = run_dir / f"prompt_chunk_{idx:02d}.txt"
                prompt_path.write_text(prompt)
                if args.verbose:
                    print(
                        f"[dry-run] Saved stage-2 prompt for chunk {idx} at {prompt_path}"
                    )
                continue

            stage2_reasoning = (
                args.model_reasoning_effort or ("high" if args.two_stage else "medium")
            )
            response_text = run_codex_exec(
                prompt,
                idx,
                run_dir,
                args,
                model_override=args.codex_model_stage2 or args.codex_model,
                reasoning_override=stage2_reasoning,
                label="",
            )

            if args.verbose:
                print(f"✓ Received response from stage-2 codex (chunk {idx})")
                print(f"  Parsing JSON response...")

            # Use robust multi-strategy parser
            response_json = parse_codex_response_robust(
                response_text, idx, run_dir, args.verbose, label=""
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
            error_file.write_text(
                f"Unexpected error processing chunk {idx}:\n\n{str(e)}"
            )
            print(f"⚠️  Chunk {idx} ERROR: {str(e)[:100]}")
            print(f"   Error saved to: {error_file}")
            print(f"   Continuing with remaining chunks...")
            continue

    if args.dry_run:
        print(f"[dry-run] Contexts + prompts saved to {run_dir}. No state updates.")
        return

    # Generate output files
    if args.verbose:
        print(f"\n{'='*60}")
        print("Generating output files...")
        print(f"{'='*60}")

    if args.output_format in ["markdown", "both"]:
        markdown_content = format_cards_as_markdown(
            all_proposed_cards, contexts, run_timestamp
        )
        markdown_path = (
            output_dir / f"proposed_cards_{datetime.now().strftime('%Y-%m-%d')}.md"
        )
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

    if args.verbose:
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


__all__ = [
    "build_codex_prompt",
    "build_codex_filter_prompt",
    "run_codex_exec",
    "parse_codex_response_robust",
    "format_cards_as_markdown",
    "run_codex_pipeline",
]

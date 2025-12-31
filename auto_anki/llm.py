"""
LLM backend execution and response parsing.

This module provides the interface for executing prompts via different
LLM backends (Codex, Claude Code, etc.) and processing their responses.

Prompt building is handled by the prompts module.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from json_repair import repair_json

from auto_anki.config_types import LLMPipelineConfig
from auto_anki.llm_backends import LLMConfig, get_backend, run_backend

# Re-export prompt builders for backward compatibility
# Import from auto_anki.prompts directly for new code
from auto_anki.prompts import (
    MAX_USER_PROMPT_CHARS,
    build_conversation_filter_prompt,
    build_conversation_prompt,
    truncate_mega_prompt,
)


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

    llm_config = LLMConfig(
        model=model,
        reasoning_effort=reasoning,
        extra_args=extra_args,
    )

    # Execute via backend
    response = run_backend(
        backend,
        prompt,
        llm_config,
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
            if isinstance(result, dict):
                if verbose:
                    print(f"  ✓ Parsed with strategy: {strategy_name}")
                # Save the working version
                (run_dir / f"codex{label}_parsed_response_chunk_{chunk_idx:02d}.json").write_text(
                    json.dumps(result, indent=2)
                )
                return result
            # Got valid JSON but not a dict (e.g., parsed a nested array).
            # Continue trying other strategies that might parse the full object.
            if verbose:
                print(f"  ⚠ Strategy {strategy_name} parsed non-dict: {type(result).__name__}")
            continue
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
    """Split a sequence into chunks of the given size."""
    for start in range(0, len(seq), size):
        yield list(seq[start : start + size])


__all__ = [
    # Prompt builders (re-exported from prompts module)
    "build_conversation_prompt",
    "build_conversation_filter_prompt",
    "truncate_mega_prompt",
    "MAX_USER_PROMPT_CHARS",
    # Execution and parsing
    "run_codex_exec",
    "parse_codex_response_robust",
    "chunked",
]

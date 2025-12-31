"""
Prompt loading and building for the two-stage LLM pipeline.

This module handles:
- Loading prompt templates from external files
- Building conversation payloads for LLM prompts
- Safe substitution of dynamic content (deck names, etc.)
"""

from __future__ import annotations

import json
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from auto_anki.config_types import LLMPipelineConfig
    from auto_anki.contexts import Conversation

# Maximum characters for user prompts in LLM prompts.
# Prompts longer than this are truncated to prevent context window overflow.
# This handles cases where users paste entire HTML documents, PDFs, etc.
MAX_USER_PROMPT_CHARS = 4000

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def truncate_mega_prompt(text: str, max_chars: int = MAX_USER_PROMPT_CHARS) -> str:
    """Truncate mega-prompts while preserving context.

    When users paste large documents (HTML, code, etc.), we truncate to:
    - First ~70% of allowed chars (to show what was pasted)
    - A truncation notice with character count
    - Last ~20% of allowed chars (to show the end/question)

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


def _load_template(name: str) -> Template:
    """Load a prompt template from the prompts directory.

    Uses string.Template for safe substitution - handles deck names
    containing $ or {} without breaking.

    Args:
        name: Template name without extension (e.g., "stage1_filter")

    Returns:
        Template object ready for substitution
    """
    path = _PROMPTS_DIR / f"{name}.md"
    return Template(path.read_text())


def build_conversation_prompt(
    conversations: List["Conversation"],
    config: "LLMPipelineConfig",
) -> str:
    """Build the Stage 2 card generation prompt.

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

    # Load template and substitute deck list
    template = _load_template("stage2_generate")
    if available_decks:
        deck_list = "\n".join(f"- {deck}" for deck in available_decks)
    else:
        deck_list = "- (No decks specified - use your best judgment)"

    # safe_substitute leaves unknown $variables intact (safe for deck names with $)
    instructions = template.safe_substitute(deck_list=deck_list)

    return instructions + "\n\n" + json.dumps(payload, indent=2, ensure_ascii=False)


def build_conversation_filter_prompt(
    conversations: List["Conversation"],
    config: "LLMPipelineConfig",
) -> str:
    """Build the Stage 1 filter prompt for conversations.

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

    # Load template (no substitutions needed for Stage 1)
    template = _load_template("stage1_filter")
    instructions = template.safe_substitute()

    return instructions + "\n\n" + json.dumps(payload, indent=2, ensure_ascii=False)

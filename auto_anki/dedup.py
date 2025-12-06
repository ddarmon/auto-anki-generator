"""
Deduplication helpers, including semantic and string-based strategies.
"""

from typing import List

from auto_anki_agent import (  # type: ignore
    Card,
    ChatTurn,
    SemanticCardIndex,
    is_duplicate_context,
    prune_contexts,
    quick_similarity,
)

__all__ = [
    "Card",
    "ChatTurn",
    "SemanticCardIndex",
    "quick_similarity",
    "is_duplicate_context",
    "prune_contexts",
    "List",
]


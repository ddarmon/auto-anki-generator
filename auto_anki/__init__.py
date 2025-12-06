"""
Core package for the Auto Anki Agent.

This package provides a structured API over the monolithic
`auto_anki_agent.py` script so humans (and tools) can more easily
discover where key concepts live: cards, contexts, deduplication,
Codex integration, and state management.
"""

__all__ = [
    "cli",
    "cards",
    "contexts",
    "dedup",
    "codex",
    "state",
]


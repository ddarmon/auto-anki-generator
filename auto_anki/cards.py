"""
Card structures and deck parsing helpers.

These are re-exported from the existing `auto_anki_agent` module for
now to preserve behavior while the codebase is being decomposed.
"""

from pathlib import Path
from typing import List, Optional

from auto_anki_agent import Card, collect_decks, parse_html_deck  # type: ignore

__all__ = ["Card", "parse_html_deck", "collect_decks", "Path", "List", "Optional"]


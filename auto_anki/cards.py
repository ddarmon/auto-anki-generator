"""
Card structures and Anki deck loading via AnkiConnect.

This module owns:
- `Card` dataclass
- `normalize_text` utility
- `load_cards_from_anki` (AnkiConnect-based card loading)
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

# Import AnkiConnect client
sys.path.insert(0, str(Path(__file__).parent.parent))
from anki_connect import AnkiConnectClient, AnkiConnectError

logger = logging.getLogger(__name__)


def normalize_text(value: str) -> str:
    """Normalize text for deduplication/comparison."""
    collapsed = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return re.sub(r"\s+", " ", collapsed).strip()


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
    # Track Anki note ID for cache invalidation
    note_id: Optional[int] = None


def load_cards_from_anki(
    deck_names: List[str],
    cache_dir: Optional[Path] = None,
    cache_ttl_minutes: int = 5,
    verbose: bool = False,
) -> List[Card]:
    """
    Load existing cards from Anki via AnkiConnect.

    Requires Anki to be running with AnkiConnect plugin installed.

    Args:
        deck_names: List of Anki deck names to load cards from
        cache_dir: Directory for caching card data (optional)
        cache_ttl_minutes: Cache time-to-live in minutes (default: 5)
        verbose: Print progress information

    Returns:
        List of Card objects from the specified decks

    Raises:
        ConnectionError: If cannot connect to Anki
        RuntimeError: If no decks specified
    """
    if not deck_names:
        raise RuntimeError("No decks specified. Add 'decks' to config or use --decks flag.")

    client = AnkiConnectClient()

    # Check connection
    if not client.check_connection():
        raise ConnectionError(
            "Cannot connect to Anki. "
            "Make sure Anki is running with AnkiConnect plugin installed (code: 2055492159)."
        )

    # Check cache
    if cache_dir:
        cache_path = Path(cache_dir) / "anki_cards_cache.json"
        cached_cards = _load_from_cache(cache_path, deck_names, cache_ttl_minutes)
        if cached_cards is not None:
            if verbose:
                print(f"  ✓ Loaded {len(cached_cards)} cards from cache")
            return cached_cards

    # Fetch cards from Anki
    if verbose:
        print(f"Loading cards from Anki decks: {', '.join(deck_names)}...")

    cards: List[Card] = []
    for deck in deck_names:
        deck_cards = _fetch_deck_cards(client, deck, verbose)
        cards.extend(deck_cards)

    if verbose:
        print(f"  ✓ Loaded {len(cards)} cards from {len(deck_names)} deck(s)")

    # Save to cache
    if cache_dir:
        cache_path = Path(cache_dir) / "anki_cards_cache.json"
        _save_to_cache(cache_path, cards, deck_names)

    return cards


def _fetch_deck_cards(client: AnkiConnectClient, deck: str, verbose: bool) -> List[Card]:
    """Fetch cards from a single deck via AnkiConnect."""
    try:
        # Find notes in deck
        note_ids = client.find_notes(f'deck:"{deck}"')

        if not note_ids:
            if verbose:
                print(f"  Warning: Deck '{deck}' has 0 cards")
            return []

        # Get note info
        notes_info = client.get_notes_info(note_ids)

        cards = []
        for note in notes_info:
            fields = note.get("fields", {})
            front = fields.get("Front", {}).get("value", "")
            back = fields.get("Back", {}).get("value", "")

            if not front and not back:
                continue

            cards.append(
                Card(
                    deck=deck,
                    front=front,
                    back=back,
                    tags=note.get("tags", []),
                    meta="",
                    data_search="",
                    front_norm=normalize_text(front),
                    back_norm=normalize_text(back),
                    note_id=note.get("noteId"),
                )
            )

        if verbose:
            print(f"    {deck}: {len(cards)} cards")

        return cards

    except AnkiConnectError as e:
        if verbose:
            print(f"  Warning: Failed to fetch deck '{deck}': {e}")
        return []


def _load_from_cache(
    cache_path: Path,
    deck_names: List[str],
    cache_ttl_minutes: int,
) -> Optional[List[Card]]:
    """Load cards from cache if valid and not expired."""
    if not cache_path.exists():
        return None

    try:
        cache_data = json.loads(cache_path.read_text())

        # Check deck list matches
        if set(cache_data.get("deck_names", [])) != set(deck_names):
            return None

        # Check TTL
        cached_time = cache_data.get("cached_at", 0)
        age_minutes = (time.time() - cached_time) / 60
        if age_minutes > cache_ttl_minutes:
            return None

        # Reconstruct Card objects
        return [
            Card(
                deck=c["deck"],
                front=c["front"],
                back=c["back"],
                tags=c["tags"],
                meta=c.get("meta", ""),
                data_search=c.get("data_search", ""),
                front_norm=c["front_norm"],
                back_norm=c["back_norm"],
                note_id=c.get("note_id"),
            )
            for c in cache_data.get("cards", [])
        ]

    except (json.JSONDecodeError, KeyError):
        return None


def _save_to_cache(cache_path: Path, cards: List[Card], deck_names: List[str]) -> None:
    """Save cards to cache."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "deck_names": deck_names,
            "cached_at": time.time(),
            "card_count": len(cards),
            "cards": [
                {
                    "deck": c.deck,
                    "front": c.front,
                    "back": c.back,
                    "tags": c.tags,
                    "meta": c.meta,
                    "data_search": c.data_search,
                    "front_norm": c.front_norm,
                    "back_norm": c.back_norm,
                    "note_id": c.note_id,
                }
                for c in cards
            ],
        }
        cache_path.write_text(json.dumps(cache_data, indent=2))
    except Exception as e:
        # Best-effort caching; warn so repeated cache misses are visible
        logger.warning("Failed to write Anki card cache to %s: %s", cache_path, e)


__all__ = ["Card", "normalize_text", "load_cards_from_anki"]

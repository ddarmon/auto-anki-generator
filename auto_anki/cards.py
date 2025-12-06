"""
Card structures and deck parsing helpers.

This module owns:
- `Card` dataclass
- `normalize_text` utility
- HTML deck parsing (`parse_html_deck`)
- Deck collection with optional caching (`collect_decks`)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from glob import glob
from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup  # type: ignore


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
    # Track source HTML file for cache invalidation
    source_path: Optional[Path] = None


def parse_html_deck(html_path: Path, cache_path: Optional[Path] = None) -> List[Card]:
    """Parse HTML deck with optional caching to avoid re-parsing large files."""
    # Check cache if provided
    if cache_path and cache_path.exists():
        try:
            cache_data = json.loads(cache_path.read_text())
            # Check if HTML file hasn't been modified since cache
            html_mtime = html_path.stat().st_mtime
            if cache_data.get("html_mtime") == html_mtime:
                # Reconstruct Card objects from cache
                return [
                    Card(
                        deck=c["deck"],
                        front=c["front"],
                        back=c["back"],
                        tags=c["tags"],
                        meta=c["meta"],
                        data_search=c["data_search"],
                        front_norm=c["front_norm"],
                        back_norm=c["back_norm"],
                        source_path=html_path,
                    )
                    for c in cache_data["cards"]
                ]
        except (json.JSONDecodeError, KeyError):
            # Cache invalid, fall back to parsing
            pass

    # Parse HTML (this can be slow for large files)
    soup = BeautifulSoup(html_path.read_text(), "html.parser")
    deck = html_path.stem.replace("_", " ")
    cards: List[Card] = []
    for section in soup.select("section.card-wrapper"):
        front_el = section.select_one(".front")
        back_el = section.select_one(".back")
        if not front_el or not back_el:
            continue
        front = front_el.get_text("\n", strip=True)
        back = back_el.get_text("\n", strip=True)
        meta_el = section.select_one(".card-meta")
        meta_text = meta_el.get_text(" ", strip=True) if meta_el else ""
        data_search = section.get("data-search") or ""
        tags = [
            tag.strip()
            for chunk in meta_text.split("--")
            for tag in [chunk.strip()]
            if tag
        ]
        cards.append(
            Card(
                deck=deck,
                front=front,
                back=back,
                tags=tags,
                meta=meta_text,
                data_search=data_search,
                front_norm=normalize_text(front),
                back_norm=normalize_text(back),
                source_path=html_path,
            )
        )

    # Save to cache if path provided
    if cache_path:
        cache_data = {
            "html_mtime": html_path.stat().st_mtime,
            "cards": [
                {k: v for k, v in asdict(c).items() if k != "source_path"}
                for c in cards
            ],
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache_data, indent=2))

    return cards


def collect_decks(pattern: str, cache_dir: Optional[Path] = None) -> List[Card]:
    """Collect cards from HTML decks with optional caching."""
    cards: List[Card] = []
    for path_str in sorted(glob(pattern)):
        path = Path(path_str)
        if not path.is_file():
            continue

        # Use cache if cache_dir provided
        cache_path = None
        if cache_dir:
            cache_path = cache_dir / f"{path.stem}_cards.json"

        cards.extend(parse_html_deck(path, cache_path))
    return cards


__all__ = ["Card", "normalize_text", "parse_html_deck", "collect_decks"]

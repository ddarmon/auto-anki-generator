"""
Deduplication helpers, including semantic and string-based strategies.

This module owns:
- `SemanticCardIndex` (FAISS/NumPy-backed semantic index)
- `quick_similarity` / `is_duplicate_context`
- `prune_contexts` (hybrid string + semantic dedup)
"""

from __future__ import annotations

import json
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from auto_anki.cards import Card, normalize_text
from auto_anki.contexts import ChatTurn

try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class SemanticCardIndex:
    """
    Semantic index over existing cards using sentence-transformer embeddings.

    This is used for semantic deduplication of contexts against existing deck
    content. Uses FAISS for fast similarity search and caches embeddings to disk.
    """

    def __init__(
        self,
        cards: List[Card],
        sentence_transformer_cls: Any,
        np_module: Any,
        model_name: str,
        verbose: bool = False,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.cards = cards
        self._np = np_module
        self.model_name = model_name
        self.verbose = verbose
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".deck_cache")
        self.cache_path = self.cache_dir / "embeddings" / "all_decks.faiss"
        self.meta_path = self.cache_dir / "embeddings" / "all_decks.meta.json"

        if verbose:
            print(f"Loading semantic dedup model '{model_name}'...")

        self.model = sentence_transformer_cls(model_name)

        # Try to load from cache first (if FAISS available)
        if FAISS_AVAILABLE and self._load_cache(cards):
            if verbose:
                print(f"  ✓ Loaded embeddings from cache ({len(cards)} cards)")
            return

        # Cache miss or FAISS unavailable: generate embeddings
        if verbose:
            cache_reason = "FAISS not available" if not FAISS_AVAILABLE else "cache miss"
            print(f"  Building semantic index ({len(cards)} cards, {cache_reason})...")

        # Build embeddings for all cards using front+back text
        texts = [
            (card.front + " " + card.back).strip()
            for card in cards
            if (card.front or card.back)
        ]

        if not texts:
            # No cards – keep a trivial empty index
            if FAISS_AVAILABLE:
                self.index = faiss.IndexFlatIP(384)  # 384 = all-MiniLM-L6-v2 dim
            else:
                self.embeddings = np_module.zeros((0, 1), dtype="float32")
            return

        emb = self.model.encode(texts, convert_to_numpy=True)
        emb = emb.astype("float32")
        norms = np_module.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_normalized = emb / norms

        if FAISS_AVAILABLE:
            self.index = self._build_faiss_index(emb_normalized)
            self._save_cache(cards)
        else:
            # Fallback to NumPy
            self.embeddings = emb_normalized

    def _build_faiss_index(self, embeddings: Any) -> Any:
        """Build FAISS IndexFlatIP for exact cosine similarity."""
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(embeddings)
        return index

    def _load_cache(self, cards: List[Card]) -> bool:
        """Load FAISS index from cache if valid."""
        if not self.cache_path.exists() or not self.meta_path.exists():
            return False

        try:
            # Load and validate metadata
            meta = json.loads(self.meta_path.read_text())

            if not self._is_cache_valid(meta, cards):
                return False

            # Load FAISS index
            self.index = faiss.read_index(str(self.cache_path))
            return True

        except (json.JSONDecodeError, RuntimeError, Exception):
            # Cache corrupted or incompatible
            return False

    def _is_cache_valid(self, meta: Dict[str, Any], cards: List[Card]) -> bool:
        """Check if cached embeddings are still valid."""
        # Check model name
        if meta.get("model_name") != self.model_name:
            return False

        # Check card count
        if meta.get("card_count") != len(cards):
            return False

        # Check deck file mtimes
        current_deck_files = self._get_deck_mtimes(cards)
        cached_deck_files = meta.get("deck_files", {})

        if current_deck_files != cached_deck_files:
            return False

        return True

    def _save_cache(self, cards: List[Card]) -> None:
        """Save FAISS index and metadata to cache."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss.write_index(self.index, str(self.cache_path))

            # Save metadata
            meta = {
                "model_name": self.model_name,
                "deck_files": self._get_deck_mtimes(cards),
                "card_count": len(cards),
                "embedding_dim": self.index.d,
                "created_at": datetime.now().isoformat(),
                "faiss_index_type": "IndexFlatIP",
            }
            self.meta_path.write_text(json.dumps(meta, indent=2))

        except Exception as e:  # pragma: no cover - best-effort cache
            if self.verbose:
                print(f"  Warning: Failed to save cache: {e}")

    def _get_deck_mtimes(self, cards: List[Card]) -> Dict[str, float]:
        """Get dict of deck file paths -> mtimes for cache invalidation."""
        deck_files: Dict[str, float] = {}
        for card in cards:
            if card.source_path and card.source_path.exists():
                path = str(card.source_path)
                if path not in deck_files:
                    deck_files[path] = card.source_path.stat().st_mtime
        return deck_files

    def is_duplicate(self, context: ChatTurn, threshold: float) -> bool:
        """Return True if context is semantically close to any existing card."""
        text = (context.user_prompt + " " + context.assistant_answer).strip()
        if not text:
            return False

        if FAISS_AVAILABLE:
            # FAISS-based search
            if self.index.ntotal == 0:
                return False

            query_emb = self.model.encode([text], convert_to_numpy=True)
            query_vec = query_emb[0].astype("float32")

            # Normalize query vector
            norm = self._np.linalg.norm(query_vec)
            if norm == 0:
                return False
            query_vec = query_vec / norm

            # FAISS search: get top-1 match
            distances, _indices = self.index.search(query_vec.reshape(1, -1), k=1)

            max_similarity = distances[0][0]
            return max_similarity >= threshold

        # NumPy fallback
        if self.embeddings.shape[0] == 0:
            return False

        query_emb = self.model.encode([text], convert_to_numpy=True)
        query_vec = query_emb[0].astype("float32")
        norm = self._np.linalg.norm(query_vec)
        if norm == 0:
            return False
        query_vec = query_vec / norm

        scores = self.embeddings @ query_vec
        max_score = float(scores.max())
        return max_score >= threshold


def quick_similarity(s1: str, s2: str) -> float:
    """Fast approximate similarity using set overlap of words."""
    if not s1 or not s2:
        return 0.0
    words1 = set(s1.split())
    words2 = set(s2.split())
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def is_duplicate_context(
    context: ChatTurn, cards: List[Card], threshold: float
) -> bool:
    """Check if context is duplicate against existing cards with optimization."""
    user_norm = normalize_text(context.user_prompt)
    answer_norm = normalize_text(context.assistant_answer)

    # Quick pre-filter: only do expensive SequenceMatcher on promising candidates
    quick_threshold = threshold * 0.6  # Lower threshold for quick check

    for card in cards:
        if not card.front_norm and not card.back_norm:
            continue

        # Quick check first (fast)
        if card.front_norm:
            if quick_similarity(user_norm, card.front_norm) >= quick_threshold:
                # Only do expensive check if quick check passes
                if (
                    SequenceMatcher(None, user_norm, card.front_norm).ratio()
                    >= threshold
                ):
                    return True

        if card.back_norm:
            if quick_similarity(answer_norm, card.back_norm) >= quick_threshold:
                # Only do expensive check if quick check passes
                if (
                    SequenceMatcher(None, answer_norm, card.back_norm).ratio()
                    >= threshold
                ):
                    return True

    return False


def prune_contexts(
    contexts: List[ChatTurn], cards: List[Card], args: Any
) -> List[ChatTurn]:
    """Filter contexts to remove duplicates against existing cards."""
    pruned: List[ChatTurn] = []
    total = len(contexts)

    semantic_index: Optional[SemanticCardIndex] = None
    if args.dedup_method in ("semantic", "hybrid"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            import numpy as _np  # type: ignore

            if getattr(args, "verbose", False):
                print("Building semantic index over existing cards for deduplication...")

            semantic_index = SemanticCardIndex(
                cards=cards,
                sentence_transformer_cls=SentenceTransformer,
                np_module=_np,
                model_name=args.semantic_model,
                verbose=args.verbose,
                cache_dir=args.cache_dir,
            )
        except ImportError:
            # Fall back to string-based deduplication
            print(
                "⚠ Semantic deduplication dependencies not found. Falling back to string-based deduplication."
            )
            print(
                "  For better duplicate detection, install: uv pip install -e '.[semantic]'"
            )
            print("  or: pip install sentence-transformers numpy faiss-cpu")
            print()

    for idx, context in enumerate(
        sorted(contexts, key=lambda c: c.score, reverse=True), 1
    ):
        if getattr(args, "verbose", False):
            print(f"  Checking context {idx}/{total} for duplicates...", end="\r")

        is_dup = False

        # String-based deduplication (current default behaviour)
        if args.dedup_method in ("string", "hybrid"):
            if is_duplicate_context(context, cards, args.similarity_threshold):
                is_dup = True

        # Semantic deduplication using embeddings
        if not is_dup and semantic_index is not None:
            if semantic_index.is_duplicate(
                context, args.semantic_similarity_threshold
            ):
                is_dup = True

        if is_dup:
            continue
        pruned.append(context)

    if getattr(args, "verbose", False):
        print(f"  Checked {total} contexts for duplicates - done!     ")

    return pruned


__all__ = [
    "Card",
    "ChatTurn",
    "SemanticCardIndex",
    "quick_similarity",
    "is_duplicate_context",
    "prune_contexts",
]

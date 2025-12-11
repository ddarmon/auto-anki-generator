"""
Deduplication helpers, including semantic and string-based strategies.

This module owns:
- `SemanticCardIndex` (FAISS/NumPy-backed semantic index)
- `quick_similarity` / `is_duplicate_context`
- `prune_contexts` (hybrid string + semantic dedup)
- `enrich_cards_with_duplicate_flags` (post-generation dedup)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from auto_anki.cards import Card, normalize_text
from auto_anki.contexts import ChatTurn, Conversation


@dataclass
class DuplicateMatch:
    """Result of a semantic similarity check against existing cards."""

    card_index: int  # Index in the existing cards list
    similarity: float  # Cosine similarity score (0-1)
    matched_card: Card  # The existing card that matched


@dataclass
class DuplicateFlags:
    """Duplicate detection results for a proposed card."""

    is_likely_duplicate: bool  # True if similarity >= threshold
    similarity_score: float  # Best match score (0-1)
    best_match: Optional[DuplicateMatch]  # Best matching existing card
    threshold_used: float  # Threshold that was applied

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict for enriched card output."""
        result: Dict[str, Any] = {
            "is_likely_duplicate": self.is_likely_duplicate,
            "similarity_score": round(self.similarity_score, 4),
            "threshold_used": self.threshold_used,
        }
        if self.best_match:
            result["matched_card"] = {
                "deck": self.best_match.matched_card.deck,
                "front": self.best_match.matched_card.front[:200],
                "back": self.best_match.matched_card.back[:200] if self.best_match.matched_card.back else "",
            }
        else:
            result["matched_card"] = None
        return result

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

        # Check note IDs match (detects additions/deletions/modifications)
        current_note_ids = self._get_note_ids(cards)
        cached_note_ids = meta.get("note_ids", [])

        if current_note_ids != cached_note_ids:
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
                "note_ids": self._get_note_ids(cards),
                "card_count": len(cards),
                "embedding_dim": self.index.d,
                "created_at": datetime.now().isoformat(),
                "faiss_index_type": "IndexFlatIP",
            }
            self.meta_path.write_text(json.dumps(meta, indent=2))

        except Exception as e:  # pragma: no cover - best-effort cache
            if self.verbose:
                print(f"  Warning: Failed to save cache: {e}")

    def _get_note_ids(self, cards: List[Card]) -> List[int]:
        """Get sorted list of note IDs for cache invalidation."""
        note_ids = [c.note_id for c in cards if c.note_id is not None]
        return sorted(note_ids)

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

    def find_similar(
        self,
        text: str,
        top_k: int = 1,
    ) -> List[Tuple[int, float]]:
        """Find top-k similar cards with their indices and similarity scores.

        Args:
            text: The text to search for (e.g., proposed card front+back)
            top_k: Number of top matches to return

        Returns:
            List of (card_index, similarity_score) tuples, sorted by similarity desc.
            Returns empty list if no cards or empty text.
        """
        if not text.strip():
            return []

        # Encode and normalize query
        query_emb = self.model.encode([text], convert_to_numpy=True)
        query_vec = query_emb[0].astype("float32")
        norm = self._np.linalg.norm(query_vec)
        if norm == 0:
            return []
        query_vec = query_vec / norm

        if FAISS_AVAILABLE:
            if self.index.ntotal == 0:
                return []

            # Limit k to actual number of cards
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_vec.reshape(1, -1), k=k)

            results: List[Tuple[int, float]] = []
            for i in range(k):
                idx = int(indices[0][i])
                score = float(distances[0][i])
                if idx >= 0:  # FAISS returns -1 for empty results
                    results.append((idx, score))
            return results

        # NumPy fallback
        if self.embeddings.shape[0] == 0:
            return []

        scores = self.embeddings @ query_vec
        # Get top-k indices
        k = min(top_k, len(scores))
        top_indices = self._np.argsort(scores)[-k:][::-1]

        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        return results

    def check_proposed_card(
        self,
        front: str,
        back: str,
        threshold: float = 0.85,
    ) -> DuplicateFlags:
        """Check a proposed card against existing cards for duplicates.

        Args:
            front: Proposed card front text
            back: Proposed card back text
            threshold: Similarity threshold above which = likely duplicate

        Returns:
            DuplicateFlags with match info and similarity score
        """
        # Combine front and back for embedding (same as how existing cards are indexed)
        text = (front + " " + back).strip()

        if not text:
            return DuplicateFlags(
                is_likely_duplicate=False,
                similarity_score=0.0,
                best_match=None,
                threshold_used=threshold,
            )

        # Find the single best match
        matches = self.find_similar(text, top_k=1)

        if not matches:
            return DuplicateFlags(
                is_likely_duplicate=False,
                similarity_score=0.0,
                best_match=None,
                threshold_used=threshold,
            )

        card_idx, similarity = matches[0]
        is_dup = similarity >= threshold

        best_match = DuplicateMatch(
            card_index=card_idx,
            similarity=similarity,
            matched_card=self.cards[card_idx],
        )

        return DuplicateFlags(
            is_likely_duplicate=is_dup,
            similarity_score=similarity,
            best_match=best_match,
            threshold_used=threshold,
        )


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


def is_conversation_duplicate(
    conversation: Conversation,
    cards: List[Card],
    threshold: float,
    semantic_index: Optional[SemanticCardIndex] = None,
    semantic_threshold: float = 0.85,
) -> tuple[bool, List[int]]:
    """Check if a conversation is duplicate against existing cards.

    Returns:
        Tuple of (all_turns_duplicate, list_of_duplicate_turn_indices)
    """
    duplicate_turns: List[int] = []

    for turn in conversation.turns:
        is_dup = False

        # String-based check
        if is_duplicate_context(turn, cards, threshold):
            is_dup = True

        # Semantic check if available
        if not is_dup and semantic_index is not None:
            if semantic_index.is_duplicate(turn, semantic_threshold):
                is_dup = True

        if is_dup:
            duplicate_turns.append(turn.turn_index)

    # Conversation is fully duplicate only if ALL turns are duplicates
    all_duplicate = len(duplicate_turns) == len(conversation.turns)

    return all_duplicate, duplicate_turns


def prune_conversations(
    conversations: List[Conversation],
    cards: List[Card],
    args: Any,
) -> Tuple[List[Conversation], List[str]]:
    """Filter conversations, marking turns that are already covered by existing cards.

    Unlike per-turn deduplication, this:
    1. Keeps conversations even if some turns are duplicates
    2. Annotates which turns are "covered" so the LLM can skip them intelligently
    3. Only removes entire conversations if ALL high-scoring turns are duplicates

    Args:
        conversations: List of Conversation objects to filter
        cards: Existing cards to check against
        args: CLI arguments namespace

    Returns:
        Tuple of:
        - List of Conversation objects with duplicate_turns annotation
        - List of conversation IDs that were skipped as duplicates (for state tracking)
    """
    pruned: List[Conversation] = []
    skipped_duplicate_ids: List[str] = []
    total = len(conversations)

    # Build semantic index if needed
    semantic_index: Optional[SemanticCardIndex] = None
    dedup_method = getattr(args, "dedup_method", "hybrid")

    if dedup_method in ("semantic", "hybrid"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            import numpy as _np  # type: ignore

            if getattr(args, "verbose", False):
                print("Building semantic index for conversation deduplication...")

            semantic_index = SemanticCardIndex(
                cards=cards,
                sentence_transformer_cls=SentenceTransformer,
                np_module=_np,
                model_name=getattr(args, "semantic_model", "all-MiniLM-L6-v2"),
                verbose=getattr(args, "verbose", False),
                cache_dir=getattr(args, "cache_dir", None),
            )
        except ImportError:
            if getattr(args, "verbose", False):
                print(
                    "⚠ Semantic dedup dependencies not found. Using string-based only."
                )

    string_threshold = getattr(args, "similarity_threshold", 0.82)
    semantic_threshold = getattr(args, "semantic_similarity_threshold", 0.85)

    for idx, conv in enumerate(
        sorted(conversations, key=lambda c: c.aggregate_score, reverse=True), 1
    ):
        if getattr(args, "verbose", False):
            print(
                f"  Checking conversation {idx}/{total} ({len(conv.turns)} turns)...",
                end="\r",
            )

        all_dup, dup_turns = is_conversation_duplicate(
            conv,
            cards,
            string_threshold,
            semantic_index,
            semantic_threshold,
        )

        # Skip entire conversation only if ALL turns are duplicates
        if all_dup:
            if getattr(args, "verbose", False):
                print(
                    f"  Skipping conversation {idx}: all {len(conv.turns)} turns are duplicates"
                )
            skipped_duplicate_ids.append(conv.conversation_id)
            continue

        # Annotate the conversation with duplicate turn info
        # This allows the LLM to know which turns are already covered
        conv.aggregate_signals["duplicate_turns"] = dup_turns
        conv.aggregate_signals["non_duplicate_turns"] = [
            t.turn_index for t in conv.turns if t.turn_index not in dup_turns
        ]

        pruned.append(conv)

    if getattr(args, "verbose", False):
        dup_count = total - len(pruned)
        print(
            f"  Checked {total} conversations - {dup_count} fully duplicate, {len(pruned)} kept"
        )

    return pruned, skipped_duplicate_ids


def enrich_cards_with_duplicate_flags(
    proposed_cards: List[Dict[str, Any]],
    existing_cards: List[Card],
    threshold: float = 0.85,
    semantic_model: str = "all-MiniLM-L6-v2",
    cache_dir: Optional[Path] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Enrich proposed cards with duplicate detection flags.

    For each proposed card, finds the best matching existing card using
    semantic similarity and adds a `duplicate_flags` field with:
    - is_likely_duplicate: bool
    - similarity_score: float
    - matched_card: {deck, front, back} or None
    - threshold_used: float

    Args:
        proposed_cards: List of proposed card dicts from LLM
        existing_cards: List of Card objects from Anki
        threshold: Similarity threshold for flagging as duplicate (default: 0.85)
        semantic_model: SentenceTransformer model name
        cache_dir: Directory for caching embeddings
        verbose: Print progress

    Returns:
        Same list of cards with `duplicate_flags` field added to each
    """
    if not proposed_cards:
        return proposed_cards

    if not existing_cards:
        # No existing cards - nothing can be a duplicate
        if verbose:
            print("  No existing cards to check against - skipping post-dedup")
        for card in proposed_cards:
            card["duplicate_flags"] = DuplicateFlags(
                is_likely_duplicate=False,
                similarity_score=0.0,
                best_match=None,
                threshold_used=threshold,
            ).to_dict()
        return proposed_cards

    # Build semantic index over existing cards
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import numpy as _np  # type: ignore

        if verbose:
            print(f"  Building semantic index for post-generation dedup ({len(existing_cards)} cards)...")

        semantic_index = SemanticCardIndex(
            cards=existing_cards,
            sentence_transformer_cls=SentenceTransformer,
            np_module=_np,
            model_name=semantic_model,
            verbose=verbose,
            cache_dir=cache_dir,
        )

    except ImportError:
        # Semantic dependencies not available - skip post-dedup
        print(
            "⚠ Post-generation dedup skipped: semantic dependencies not found."
        )
        print(
            "  Install with: uv pip install -e '.[semantic]'"
        )
        for card in proposed_cards:
            card["duplicate_flags"] = {
                "is_likely_duplicate": False,
                "similarity_score": 0.0,
                "matched_card": None,
                "threshold_used": threshold,
                "error": "semantic_dependencies_missing",
            }
        return proposed_cards

    # Check each proposed card against the index
    dup_count = 0
    total = len(proposed_cards)

    for idx, card in enumerate(proposed_cards, 1):
        if verbose:
            print(f"  Checking card {idx}/{total} for duplicates...", end="\r")

        front = card.get("front", "")
        back = card.get("back", "")

        flags = semantic_index.check_proposed_card(front, back, threshold)
        card["duplicate_flags"] = flags.to_dict()

        if flags.is_likely_duplicate:
            dup_count += 1

    if verbose:
        print(f"  Post-dedup complete: {dup_count}/{total} cards flagged as likely duplicates")

    return proposed_cards


__all__ = [
    "Card",
    "ChatTurn",
    "Conversation",
    "DuplicateFlags",
    "DuplicateMatch",
    "SemanticCardIndex",
    "quick_similarity",
    "is_duplicate_context",
    "is_conversation_duplicate",
    "prune_contexts",
    "prune_conversations",
    "enrich_cards_with_duplicate_flags",
]

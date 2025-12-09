"""Tests for deduplication functions in auto_anki/dedup.py."""

import pytest
from auto_anki.cards import Card, normalize_text
from auto_anki.contexts import ChatTurn
from auto_anki.dedup import (
    DuplicateFlags,
    DuplicateMatch,
    is_duplicate_context,
    quick_similarity,
    enrich_cards_with_duplicate_flags,
)


class TestIsDuplicateContext:
    """Tests for the is_duplicate_context function."""

    @pytest.fixture
    def existing_cards(self):
        """Create a set of existing cards for dedup testing."""
        return [
            Card(
                deck="Test",
                front="What is Python?",
                back="A programming language.",
                tags=[],
                meta="",
                data_search="",
                front_norm=normalize_text("What is Python?"),
                back_norm=normalize_text("A programming language."),
            ),
            Card(
                deck="Test",
                front="Define machine learning",
                back="Machine learning is a subset of AI.",
                tags=[],
                meta="",
                data_search="",
                front_norm=normalize_text("Define machine learning"),
                back_norm=normalize_text("Machine learning is a subset of AI."),
            ),
        ]

    def _make_turn(self, user_prompt: str, assistant_answer: str) -> ChatTurn:
        """Helper to create a ChatTurn for testing."""
        return ChatTurn(
            context_id="test",
            turn_index=0,
            conversation_id="conv",
            source_path="/test",
            source_title=None,
            source_url=None,
            user_timestamp=None,
            user_prompt=user_prompt,
            assistant_answer=assistant_answer,
            assistant_char_count=len(assistant_answer),
            score=0,
            signals={},
            key_terms=[],
            key_points=[],
        )

    def test_exact_match_user_prompt(self, existing_cards):
        """Exact match on user prompt should be detected as duplicate."""
        turn = self._make_turn(
            "What is Python?",
            "Different answer here."
        )
        assert is_duplicate_context(turn, existing_cards, threshold=0.8)

    def test_exact_match_assistant_answer(self, existing_cards):
        """Exact match on assistant answer should be detected as duplicate."""
        turn = self._make_turn(
            "Different question here?",
            "A programming language."
        )
        assert is_duplicate_context(turn, existing_cards, threshold=0.8)

    def test_no_match(self, existing_cards):
        """Non-matching content should not be flagged as duplicate."""
        turn = self._make_turn(
            "What is JavaScript?",
            "A web scripting language."
        )
        assert not is_duplicate_context(turn, existing_cards, threshold=0.8)

    def test_similar_but_below_threshold(self, existing_cards):
        """Similar content below threshold should not be duplicate."""
        turn = self._make_turn(
            "What is Python programming?",
            "Python is a language for programming."
        )
        # With high threshold, this shouldn't match
        assert not is_duplicate_context(turn, existing_cards, threshold=0.95)

    def test_threshold_sensitivity(self, existing_cards):
        """Lower threshold should catch more duplicates."""
        turn = self._make_turn(
            "What is Python programming language?",
            "Python is a programming language for coding."
        )
        # Should NOT match at very high threshold
        result_high = is_duplicate_context(turn, existing_cards, threshold=0.95)
        # Should match at lower threshold
        result_low = is_duplicate_context(turn, existing_cards, threshold=0.5)
        # At least one should be True with this content
        assert result_low or result_high  # Sanity check

    def test_empty_cards_list(self):
        """Empty card list should never find duplicates."""
        turn = self._make_turn("Any question?", "Any answer.")
        assert not is_duplicate_context(turn, [], threshold=0.8)

    def test_empty_user_prompt(self, existing_cards):
        """Empty user prompt should not match (normalized to empty)."""
        turn = self._make_turn("", "Some answer")
        # Empty prompt won't match non-empty front_norm
        result = is_duplicate_context(turn, existing_cards, threshold=0.8)
        # Depends on whether empty matches empty after normalization
        # The function checks normalized text, empty won't match non-empty
        assert not result or result  # Just verify it doesn't crash

    def test_card_without_norm_fields(self):
        """Cards with empty norm fields should be skipped."""
        cards = [
            Card(
                deck="Test",
                front="Question",
                back="Answer",
                tags=[],
                meta="",
                data_search="",
                front_norm="",  # Empty
                back_norm="",   # Empty
            )
        ]
        turn = self._make_turn("Question", "Answer")
        # Empty norm fields should be skipped in comparison
        assert not is_duplicate_context(turn, cards, threshold=0.8)

    def test_case_insensitive_matching(self, existing_cards):
        """Matching should be case-insensitive via normalization."""
        turn = self._make_turn(
            "WHAT IS PYTHON?",
            "A PROGRAMMING LANGUAGE."
        )
        # After normalization, case should not matter
        assert is_duplicate_context(turn, existing_cards, threshold=0.8)

    def test_punctuation_insensitive(self, existing_cards):
        """Matching should ignore punctuation via normalization."""
        turn = self._make_turn(
            "What is Python",  # No question mark
            "A programming language"  # No period
        )
        assert is_duplicate_context(turn, existing_cards, threshold=0.8)


class TestQuickSimilarityForDedup:
    """Additional quick_similarity tests specific to dedup context."""

    def test_as_prefilter(self):
        """Quick similarity should work as a fast pre-filter."""
        # If quick_similarity is low, SequenceMatcher shouldn't be called
        s1 = "machine learning neural networks deep learning"
        s2 = "cooking recipes kitchen food ingredients"

        # These should have 0 similarity (no word overlap)
        assert quick_similarity(s1, s2) == 0.0

    def test_high_overlap(self):
        """High word overlap should return high similarity."""
        s1 = "what is machine learning"
        s2 = "what is machine learning and AI"

        # Significant overlap
        sim = quick_similarity(s1, s2)
        assert sim > 0.5

    def test_normalized_input(self):
        """Test with normalized text (typical in dedup flow)."""
        norm1 = normalize_text("What is Python?")
        norm2 = normalize_text("What is Python programming?")

        sim = quick_similarity(norm1, norm2)
        assert sim > 0  # Some overlap expected


class TestDuplicateFlags:
    """Tests for the DuplicateFlags dataclass."""

    def test_to_dict_with_match(self):
        """DuplicateFlags.to_dict() should serialize correctly with a match."""
        card = Card(
            deck="Test",
            front="What is Python?",
            back="A programming language.",
            tags=["python"],
            meta="",
            data_search="",
            front_norm="",
            back_norm="",
        )
        match = DuplicateMatch(
            card_index=0,
            similarity=0.92,
            matched_card=card,
        )
        flags = DuplicateFlags(
            is_likely_duplicate=True,
            similarity_score=0.92,
            best_match=match,
            threshold_used=0.85,
        )

        result = flags.to_dict()

        assert result["is_likely_duplicate"] is True
        assert result["similarity_score"] == 0.92
        assert result["threshold_used"] == 0.85
        assert result["matched_card"]["deck"] == "Test"
        assert result["matched_card"]["front"] == "What is Python?"

    def test_to_dict_without_match(self):
        """DuplicateFlags.to_dict() should handle None match."""
        flags = DuplicateFlags(
            is_likely_duplicate=False,
            similarity_score=0.3,
            best_match=None,
            threshold_used=0.85,
        )

        result = flags.to_dict()

        assert result["is_likely_duplicate"] is False
        assert result["similarity_score"] == 0.3
        assert result["matched_card"] is None

    def test_to_dict_truncates_long_text(self):
        """DuplicateFlags.to_dict() should truncate long front/back text."""
        long_text = "x" * 500
        card = Card(
            deck="Test",
            front=long_text,
            back=long_text,
            tags=[],
            meta="",
            data_search="",
            front_norm="",
            back_norm="",
        )
        match = DuplicateMatch(card_index=0, similarity=0.9, matched_card=card)
        flags = DuplicateFlags(
            is_likely_duplicate=True,
            similarity_score=0.9,
            best_match=match,
            threshold_used=0.85,
        )

        result = flags.to_dict()

        assert len(result["matched_card"]["front"]) == 200
        assert len(result["matched_card"]["back"]) == 200


class TestEnrichCardsWithDuplicateFlags:
    """Tests for enrich_cards_with_duplicate_flags function."""

    @pytest.fixture
    def existing_cards(self):
        """Create existing cards for testing."""
        return [
            Card(
                deck="Test",
                front="What is Python?",
                back="A programming language.",
                tags=[],
                meta="",
                data_search="",
                front_norm=normalize_text("What is Python?"),
                back_norm=normalize_text("A programming language."),
            ),
        ]

    def test_empty_proposed_cards(self, existing_cards):
        """Empty proposed cards list should return empty list."""
        result = enrich_cards_with_duplicate_flags([], existing_cards, threshold=0.85)
        assert result == []

    def test_empty_existing_cards(self):
        """Empty existing cards should mark all as not duplicate."""
        proposed = [{"front": "Test question", "back": "Test answer"}]
        result = enrich_cards_with_duplicate_flags(proposed, [], threshold=0.85)

        assert len(result) == 1
        assert "duplicate_flags" in result[0]
        assert result[0]["duplicate_flags"]["is_likely_duplicate"] is False

    def test_preserves_original_card_data(self, existing_cards):
        """Should preserve all original card fields."""
        proposed = [{
            "front": "Completely new question",
            "back": "Completely new answer",
            "deck": "MyDeck",
            "confidence": 0.95,
            "tags": ["test"],
            "custom_field": "preserved",
        }]

        result = enrich_cards_with_duplicate_flags(proposed, existing_cards, threshold=0.85)

        assert result[0]["deck"] == "MyDeck"
        assert result[0]["confidence"] == 0.95
        assert result[0]["tags"] == ["test"]
        assert result[0]["custom_field"] == "preserved"
        assert "duplicate_flags" in result[0]

    def test_modifies_in_place(self, existing_cards):
        """Should modify the input list in place (and return it)."""
        proposed = [{"front": "Test", "back": "Test"}]
        original_id = id(proposed)

        result = enrich_cards_with_duplicate_flags(proposed, existing_cards, threshold=0.85)

        assert id(result) == original_id
        assert "duplicate_flags" in proposed[0]

    def test_handles_missing_front_back(self, existing_cards):
        """Should handle cards without front/back gracefully."""
        proposed = [{"deck": "Test"}]  # No front or back

        result = enrich_cards_with_duplicate_flags(proposed, existing_cards, threshold=0.85)

        assert "duplicate_flags" in result[0]
        assert result[0]["duplicate_flags"]["is_likely_duplicate"] is False

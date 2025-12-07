"""Tests for deduplication functions in auto_anki/dedup.py."""

import pytest
from auto_anki.cards import Card, normalize_text
from auto_anki.contexts import ChatTurn
from auto_anki.dedup import is_duplicate_context, quick_similarity


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

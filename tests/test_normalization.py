"""Tests for text normalization and similarity functions."""

import pytest
from auto_anki.cards import normalize_text
from auto_anki.dedup import quick_similarity


class TestNormalizeText:
    """Tests for the normalize_text function."""

    def test_lowercase(self):
        """Should convert to lowercase."""
        assert normalize_text("HELLO WORLD") == "hello world"

    def test_mixed_case(self):
        """Should handle mixed case."""
        assert normalize_text("HeLLo WoRLd") == "hello world"

    def test_whitespace_collapse(self):
        """Should collapse multiple whitespace to single space."""
        assert normalize_text("hello   world") == "hello world"
        assert normalize_text("hello\t\tworld") == "hello world"
        assert normalize_text("hello\n\nworld") == "hello world"

    def test_strip(self):
        """Should strip leading and trailing whitespace."""
        assert normalize_text("  hello  ") == "hello"
        assert normalize_text("\n\thello\t\n") == "hello"

    def test_special_characters(self):
        """Should remove special characters."""
        assert normalize_text("hello, world!") == "hello world"
        assert normalize_text("what's up?") == "what s up"
        assert normalize_text("test@email.com") == "test email com"

    def test_numbers_preserved(self):
        """Should preserve numbers."""
        result = normalize_text("Python 3.11")
        assert "3" in result
        assert "11" in result

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert normalize_text("") == ""

    def test_only_special_chars(self):
        """String of only special chars should return empty string."""
        assert normalize_text("!@#$%^&*()") == ""

    def test_unicode(self):
        """Should handle unicode characters."""
        result = normalize_text("café résumé")
        assert isinstance(result, str)
        # Non-ASCII letters are removed by the regex
        assert "caf" in result

    def test_html_entities(self):
        """Should handle HTML-like content."""
        result = normalize_text("<b>bold</b>")
        assert "b" in result
        assert "bold" in result


class TestQuickSimilarity:
    """Tests for the quick_similarity function."""

    def test_identical_strings(self):
        """Identical strings should have similarity 1.0."""
        assert quick_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        """Completely different strings should have similarity 0.0."""
        assert quick_similarity("abc def", "xyz uvw") == 0.0

    def test_partial_overlap(self):
        """Partial overlap should have intermediate similarity."""
        sim = quick_similarity("machine learning", "machine vision")
        assert 0 < sim < 1
        # "machine" is shared, 1 out of 3 unique words
        # words1 = {"machine", "learning"}, words2 = {"machine", "vision"}
        # intersection = 1, union = 3
        assert sim == pytest.approx(1/3, rel=0.01)

    def test_empty_strings(self):
        """Empty strings should return 0.0."""
        assert quick_similarity("", "") == 0.0
        assert quick_similarity("hello", "") == 0.0
        assert quick_similarity("", "world") == 0.0

    def test_single_word_match(self):
        """Single word strings that match should be 1.0."""
        assert quick_similarity("hello", "hello") == 1.0

    def test_single_word_different(self):
        """Single word strings that differ should be 0.0."""
        assert quick_similarity("hello", "world") == 0.0

    def test_word_order_irrelevant(self):
        """Word order should not affect similarity (set-based)."""
        sim1 = quick_similarity("hello world", "world hello")
        assert sim1 == 1.0

    def test_duplicate_words_in_one_string(self):
        """Duplicate words in one string shouldn't affect score."""
        # "python python" has set {"python"}
        # "python java" has set {"python", "java"}
        sim = quick_similarity("python python python", "python java")
        # intersection = 1, union = 2
        assert sim == pytest.approx(0.5, rel=0.01)

    def test_case_sensitivity(self):
        """Function is case-sensitive (based on input)."""
        # The function does s1.split() and s2.split() without lowercasing
        sim1 = quick_similarity("Hello", "hello")
        # These are different words in set comparison
        assert sim1 == 0.0

    def test_whitespace_only(self):
        """Whitespace-only strings should return 0.0."""
        assert quick_similarity("   ", "   ") == 0.0
        assert quick_similarity("hello", "   ") == 0.0

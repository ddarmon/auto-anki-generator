"""Tests for codex module functions in auto_anki/codex.py."""

import pytest
from auto_anki.codex import truncate_mega_prompt, MAX_USER_PROMPT_CHARS


class TestTruncateMegaPrompt:
    """Tests for the truncate_mega_prompt function."""

    def test_short_text_unchanged(self):
        """Short text should pass through unchanged."""
        text = "This is a short prompt."
        result = truncate_mega_prompt(text)
        assert result == text

    def test_text_at_limit_unchanged(self):
        """Text exactly at the limit should pass through unchanged."""
        text = "x" * MAX_USER_PROMPT_CHARS
        result = truncate_mega_prompt(text)
        assert result == text

    def test_long_text_truncated(self):
        """Text exceeding limit should be truncated."""
        text = "x" * (MAX_USER_PROMPT_CHARS + 10000)
        result = truncate_mega_prompt(text)
        assert len(result) < len(text)
        assert "[TRUNCATED:" in result

    def test_truncation_preserves_head_and_tail(self):
        """Truncation should preserve start and end of text."""
        head = "START_MARKER_" + "x" * 1000
        middle = "y" * 50000
        tail = "z" * 1000 + "_END_MARKER"
        text = head + middle + tail

        result = truncate_mega_prompt(text)

        assert result.startswith("START_MARKER_")
        assert result.endswith("_END_MARKER")
        assert "[TRUNCATED:" in result

    def test_truncation_notice_includes_char_count(self):
        """Truncation notice should include removed character count."""
        text = "a" * 100000
        result = truncate_mega_prompt(text)

        # Should contain something like "[TRUNCATED: 96,000 characters removed]"
        assert "[TRUNCATED:" in result
        assert "characters removed]" in result

    def test_custom_max_chars(self):
        """Custom max_chars parameter should be respected."""
        text = "x" * 500
        result = truncate_mega_prompt(text, max_chars=100)

        assert len(result) < len(text)
        assert "[TRUNCATED:" in result

    def test_empty_string(self):
        """Empty string should pass through unchanged."""
        result = truncate_mega_prompt("")
        assert result == ""

    def test_unicode_text(self):
        """Unicode text should be handled correctly."""
        # Mix of ASCII and Unicode
        text = "Hello " + "世界" * 10000
        result = truncate_mega_prompt(text, max_chars=100)

        assert "[TRUNCATED:" in result
        # Should still start with the original beginning
        assert result.startswith("Hello ")

    def test_result_length_bounded(self):
        """Result length should be bounded by max_chars plus notice overhead."""
        text = "x" * 1000000  # 1MB
        result = truncate_mega_prompt(text)

        # Result should be roughly max_chars + truncation notice length
        # (head 70% + tail 20% + notice ~50 chars)
        expected_max = MAX_USER_PROMPT_CHARS + 100
        assert len(result) < expected_max

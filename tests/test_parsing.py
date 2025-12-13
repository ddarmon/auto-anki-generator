"""Tests for chat parsing functions in auto_anki/contexts.py."""

import pytest
from auto_anki.contexts import parse_chat_entries, extract_turns, parse_chat_metadata


class TestParseChatEntries:
    """Tests for the parse_chat_entries function."""

    def test_basic_parsing(self):
        """Should parse basic user/assistant entries."""
        markdown = """[2025-01-15 10:30] user:
What is Python?

[2025-01-15 10:31] assistant:
Python is a programming language."""

        entries = parse_chat_entries(markdown)
        assert len(entries) == 2
        assert entries[0]["role"] == "user"
        assert entries[0]["timestamp"] == "2025-01-15 10:30"
        assert "Python" in entries[0]["text"]
        assert entries[1]["role"] == "assistant"
        assert "programming language" in entries[1]["text"]

    def test_multiline_content(self):
        """Should handle multiline content in entries."""
        markdown = """[2025-01-15 10:30] user:
First line
Second line
Third line

[2025-01-15 10:31] assistant:
Response here."""

        entries = parse_chat_entries(markdown)
        assert len(entries) == 2
        assert "First line" in entries[0]["text"]
        assert "Second line" in entries[0]["text"]
        assert "Third line" in entries[0]["text"]

    def test_empty_input(self):
        """Empty input should return empty list."""
        entries = parse_chat_entries("")
        assert entries == []

    def test_no_entries(self):
        """Text without entry markers should return empty list."""
        entries = parse_chat_entries("Just some random text without markers")
        assert entries == []

    def test_case_insensitive_role(self):
        """Role detection should be case-insensitive."""
        markdown = """[2025-01-15 10:30] USER:
Question

[2025-01-15 10:31] ASSISTANT:
Answer"""

        entries = parse_chat_entries(markdown)
        assert len(entries) == 2
        assert entries[0]["role"] == "user"
        assert entries[1]["role"] == "assistant"

    def test_tool_role(self):
        """Should handle tool role entries."""
        markdown = """[2025-01-15 10:30] tool:
Tool output here"""

        entries = parse_chat_entries(markdown)
        assert len(entries) == 1
        assert entries[0]["role"] == "tool"

    def test_whitespace_in_timestamp(self):
        """Should handle various timestamp formats."""
        markdown = """[2025-01-15T10:30:00] user:
Question"""

        entries = parse_chat_entries(markdown)
        assert len(entries) == 1
        assert entries[0]["timestamp"] == "2025-01-15T10:30:00"

    def test_strips_content(self):
        """Entry text should be stripped of leading/trailing whitespace."""
        markdown = """[2025-01-15 10:30] user:

   Question here

[2025-01-15 10:31] assistant:
Answer"""

        entries = parse_chat_entries(markdown)
        assert entries[0]["text"] == "Question here"


class TestExtractTurns:
    """Tests for the extract_turns function."""

    def test_pairs_entries(self):
        """Should pair user entries with following assistant entries."""
        entries = [
            {"role": "user", "text": "Q1", "timestamp": "2025-01-01 10:00"},
            {"role": "assistant", "text": "A1", "timestamp": "2025-01-01 10:01"},
            {"role": "user", "text": "Q2", "timestamp": "2025-01-01 10:02"},
            {"role": "assistant", "text": "A2", "timestamp": "2025-01-01 10:03"},
        ]

        turns = extract_turns(entries)
        assert len(turns) == 2
        assert turns[0][0]["text"] == "Q1"
        assert turns[0][1]["text"] == "A1"
        assert turns[1][0]["text"] == "Q2"
        assert turns[1][1]["text"] == "A2"

    def test_unpaired_user(self):
        """User entry without assistant response should not create turn."""
        entries = [
            {"role": "user", "text": "Q1", "timestamp": "2025-01-01"},
        ]

        turns = extract_turns(entries)
        assert len(turns) == 0

    def test_skips_leading_assistant(self):
        """Assistant entry without preceding user should be skipped."""
        entries = [
            {"role": "assistant", "text": "Orphan", "timestamp": "2025-01-01"},
            {"role": "user", "text": "Q1", "timestamp": "2025-01-01"},
            {"role": "assistant", "text": "A1", "timestamp": "2025-01-01"},
        ]

        turns = extract_turns(entries)
        assert len(turns) == 1
        assert turns[0][0]["text"] == "Q1"

    def test_skips_tool_entries(self):
        """Tool entries should not affect pairing."""
        entries = [
            {"role": "user", "text": "Q1", "timestamp": "2025-01-01"},
            {"role": "tool", "text": "Tool output", "timestamp": "2025-01-01"},
            {"role": "assistant", "text": "A1", "timestamp": "2025-01-01"},
        ]

        turns = extract_turns(entries)
        assert len(turns) == 1
        assert turns[0][0]["text"] == "Q1"
        assert turns[0][1]["text"] == "A1"

    def test_empty_text_filtered(self):
        """Entries with empty text should not create turns."""
        entries = [
            {"role": "user", "text": "", "timestamp": "2025-01-01"},
            {"role": "assistant", "text": "A1", "timestamp": "2025-01-01"},
        ]

        turns = extract_turns(entries)
        assert len(turns) == 0

    def test_empty_list(self):
        """Empty entry list should return empty turns list."""
        turns = extract_turns([])
        assert turns == []

    def test_multiple_users_before_assistant(self):
        """Multiple user entries before assistant should be merged into one prompt."""
        entries = [
            {"role": "user", "text": "Q1", "timestamp": "2025-01-01"},
            {"role": "user", "text": "Q2", "timestamp": "2025-01-01"},
            {"role": "assistant", "text": "A2", "timestamp": "2025-01-01"},
        ]

        turns = extract_turns(entries)
        assert len(turns) == 1
        assert turns[0][0]["text"] == "Q1\n\nQ2"
        assert turns[0][1]["text"] == "A2"

    def test_prefers_substantive_assistant_content(self):
        """Should ignore assistant 'thoughts/code/reasoning_recap' JSON and keep the final answer."""
        entries = [
            {"role": "user", "text": "Q1", "timestamp": "2025-01-01"},
            {
                "role": "assistant",
                "text": '{"content_type":"thoughts","thoughts":[{"summary":"x"}]}',
                "timestamp": "2025-01-01",
            },
            {
                "role": "assistant",
                "text": '{"content_type":"code","text":"{\\"search_query\\":[{\\"q\\":\\"x\\"}] }"}',
                "timestamp": "2025-01-01",
            },
            {"role": "assistant", "text": "Final answer", "timestamp": "2025-01-01"},
            {
                "role": "assistant",
                "text": '{"content_type":"reasoning_recap","content":"Thought for 1s"}',
                "timestamp": "2025-01-01",
            },
        ]

        turns = extract_turns(entries)
        assert len(turns) == 1
        assert turns[0][0]["text"] == "Q1"
        assert turns[0][1]["text"] == "Final answer"

    def test_concatenates_multiple_substantive_assistant_entries(self):
        """If the assistant answer is split across multiple entries, join them."""
        entries = [
            {"role": "user", "text": "Q1", "timestamp": "2025-01-01"},
            {"role": "assistant", "text": "Part 1", "timestamp": "2025-01-01"},
            {"role": "assistant", "text": "Part 2", "timestamp": "2025-01-01"},
        ]

        turns = extract_turns(entries)
        assert len(turns) == 1
        assert turns[0][1]["text"] == "Part 1\n\nPart 2"


class TestParseChatMetadata:
    """Tests for the parse_chat_metadata function."""

    def test_title_extraction(self):
        """Should extract title from markdown heading."""
        header = "# My Conversation Title\n\nSome content"
        metadata = parse_chat_metadata(header)
        assert metadata.get("title") == "My Conversation Title"

    def test_url_extraction(self):
        """Should extract URL from list item."""
        header = "# Title\n\n- URL: https://example.com/chat"
        metadata = parse_chat_metadata(header)
        assert metadata.get("url") == "https://example.com/chat"

    def test_multiple_metadata_fields(self):
        """Should extract multiple metadata fields."""
        header = "# Title\n\n- URL: https://example.com\n- Date: 2025-01-15\n- Author: Test"
        metadata = parse_chat_metadata(header)
        assert metadata.get("url") == "https://example.com"
        assert metadata.get("date") == "2025-01-15"
        assert metadata.get("author") == "Test"

    def test_empty_header(self):
        """Empty header should return empty metadata."""
        metadata = parse_chat_metadata("")
        assert metadata == {}

    def test_no_title(self):
        """Header without title should not have title key."""
        header = "- URL: https://example.com"
        metadata = parse_chat_metadata(header)
        assert "title" not in metadata
        assert metadata.get("url") == "https://example.com"

    def test_key_normalization(self):
        """Metadata keys should be normalized to lowercase."""
        header = "- URL: https://example.com\n- DATE: 2025-01-15"
        metadata = parse_chat_metadata(header)
        assert "url" in metadata
        assert "date" in metadata
        assert "URL" not in metadata

    def test_whitespace_handling(self):
        """Should handle whitespace in values."""
        header = "- URL:   https://example.com   \n- Title:  Some Title  "
        metadata = parse_chat_metadata(header)
        assert metadata.get("url") == "https://example.com"


class TestClaudeFormatParsing:
    """Tests for parsing Claude conversation exports."""

    def test_claude_human_role_normalized(self):
        """Claude exports use 'human' instead of 'user' - should be normalized."""
        markdown = """[2025-12-09 01:54:22] human:
How would you explain X?

[2025-12-09 01:54:51] assistant:
Here's the explanation..."""

        entries = parse_chat_entries(markdown)
        assert len(entries) == 2
        assert entries[0]["role"] == "user"  # normalized from "human"
        assert entries[0]["timestamp"] == "2025-12-09 01:54:22"
        assert "How would you explain" in entries[0]["text"]
        assert entries[1]["role"] == "assistant"

    def test_claude_conversation_creates_turns(self):
        """Claude conversations with human/assistant should create proper turns."""
        markdown = """[2025-12-09 01:54:22] human:
What is Python?

[2025-12-09 01:54:51] assistant:
Python is a programming language.

[2025-12-09 01:55:00] human:
Can you give an example?

[2025-12-09 01:55:10] assistant:
Sure, here's a simple example..."""

        from auto_anki.contexts import extract_turns
        entries = parse_chat_entries(markdown)
        turns = extract_turns(entries)

        assert len(turns) == 2
        assert turns[0][0]["text"] == "What is Python?"
        assert "programming language" in turns[0][1]["text"]
        assert turns[1][0]["text"] == "Can you give an example?"

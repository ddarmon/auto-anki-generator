"""Tests for auto_anki.import_conversations module."""

import json
import pytest
from pathlib import Path

from auto_anki.import_conversations import (
    detect_format,
    slugify,
    strip_citations,
    convert_latex_delimiters,
    fmt_unix_timestamp,
    fmt_iso_timestamp,
    render_chatgpt_conversation,
    render_claude_conversation,
    write_file_if_changed,
    ExportStats,
)


# ---------------------------------------------------------------------------
# Format detection tests
# ---------------------------------------------------------------------------


def test_detect_format_chatgpt():
    """Detects ChatGPT JSON format (has 'mapping' and 'title')."""
    conversations = [
        {
            "id": "abc123",
            "title": "Test Chat",
            "mapping": {"node1": {"message": {}}},
            "create_time": 1700000000.0,
        }
    ]
    assert detect_format(conversations) == "chatgpt"


def test_detect_format_claude():
    """Detects Claude JSON format (has 'chat_messages' and 'uuid')."""
    conversations = [
        {
            "uuid": "abc-123-def",
            "name": "Test Chat",
            "chat_messages": [{"sender": "human", "text": "Hello"}],
            "created_at": "2025-01-15T10:30:00Z",
        }
    ]
    assert detect_format(conversations) == "claude"


def test_detect_format_empty():
    """Empty list defaults to chatgpt format."""
    assert detect_format([]) == "chatgpt"


# ---------------------------------------------------------------------------
# slugify tests
# ---------------------------------------------------------------------------


def test_slugify_basic():
    """Basic title becomes lowercase slug."""
    assert slugify("Hello World") == "hello-world"


def test_slugify_special_chars():
    """Special characters are removed."""
    assert slugify("What's the Deal?!") == "whats-the-deal"


def test_slugify_path_separators():
    """Path separators become dashes."""
    assert slugify("foo/bar\\baz:qux") == "foo-bar-baz-qux"


def test_slugify_collapse_dashes():
    """Multiple dashes collapse to single dash."""
    assert slugify("hello   world---test") == "hello-world-test"


def test_slugify_max_length():
    """Slug is truncated to max_len."""
    long_title = "a" * 200
    result = slugify(long_title, max_len=50)
    assert len(result) == 50


def test_slugify_empty():
    """Empty string returns 'untitled'."""
    assert slugify("") == "untitled"
    assert slugify("   ") == "untitled"


def test_slugify_preserves_valid():
    """Valid chars (alphanumeric, dash, dot, underscore) are preserved."""
    assert slugify("file_name.test-123") == "file_name.test-123"


# ---------------------------------------------------------------------------
# strip_citations tests
# ---------------------------------------------------------------------------


def test_strip_citations_removes_citation_blocks():
    """Citation blocks with Unicode markers are removed."""
    # \ue200 = start, \ue201 = end
    text = "Hello \ue200cite:turn0search2\ue201 world"
    assert strip_citations(text) == "Hello  world"


def test_strip_citations_multiple():
    """Multiple citation blocks are all removed."""
    text = "A \ue200ref1\ue201 B \ue200ref2\ue201 C"
    assert strip_citations(text) == "A  B  C"


def test_strip_citations_no_change():
    """Text without citations is unchanged."""
    text = "Normal text without citations"
    assert strip_citations(text) == text


# ---------------------------------------------------------------------------
# convert_latex_delimiters tests
# ---------------------------------------------------------------------------


def test_convert_latex_inline():
    r"""Inline math \( ... \) becomes $...$."""
    text = r"The formula is \( x^2 + y^2 \) here."
    result = convert_latex_delimiters(text)
    assert result == "The formula is $x^2 + y^2$ here."


def test_convert_latex_display():
    r"""Display math \[ ... \] becomes $$...$$."""
    text = r"The equation is \[ E = mc^2 \] below."
    result = convert_latex_delimiters(text)
    assert result == "The equation is $$E = mc^2$$ below."


def test_convert_latex_multiline():
    """Multiline math expressions are handled."""
    text = r"""\[
    a^2 + b^2 = c^2
    \]"""
    result = convert_latex_delimiters(text)
    assert "$$" in result
    assert r"\[" not in result


def test_convert_latex_spaces_trimmed():
    """Extra spaces inside delimiters are trimmed."""
    text = r"\(   spaced   \)"
    result = convert_latex_delimiters(text)
    assert result == "$spaced$"


def test_convert_latex_no_change():
    """Text without LaTeX delimiters is unchanged."""
    text = "Normal text $already markdown$"
    assert convert_latex_delimiters(text) == text


# ---------------------------------------------------------------------------
# Timestamp formatting tests
# ---------------------------------------------------------------------------


def test_fmt_unix_timestamp_valid():
    """Unix timestamp is formatted to readable string."""
    # 2024-01-15 10:30:00 UTC (will vary by timezone)
    ts = 1705315800.0
    result = fmt_unix_timestamp(ts)
    assert "2024" in result or "2025" in result  # timezone dependent
    assert ":" in result  # Has time component


def test_fmt_unix_timestamp_none():
    """None returns empty string."""
    assert fmt_unix_timestamp(None) == ""


def test_fmt_iso_timestamp_valid():
    """ISO timestamp is formatted to readable string."""
    iso = "2025-01-15T10:30:00Z"
    result = fmt_iso_timestamp(iso)
    assert "2025" in result
    assert ":" in result


def test_fmt_iso_timestamp_none():
    """None returns empty string."""
    assert fmt_iso_timestamp(None) == ""


def test_fmt_iso_timestamp_empty():
    """Empty string returns empty string."""
    assert fmt_iso_timestamp("") == ""


# ---------------------------------------------------------------------------
# ChatGPT rendering tests
# ---------------------------------------------------------------------------


def test_render_chatgpt_conversation_basic():
    """Basic ChatGPT conversation renders to expected markdown format."""
    conv = {
        "id": "conv-123",
        "title": "Test Chat",
        "create_time": 1705315800.0,
        "update_time": 1705316000.0,
        "mapping": {
            "node1": {
                "message": {
                    "id": "msg1",
                    "author": {"role": "user"},
                    "create_time": 1705315800.0,
                    "content": {"content_type": "text", "parts": ["Hello"]},
                }
            },
            "node2": {
                "message": {
                    "id": "msg2",
                    "author": {"role": "assistant"},
                    "create_time": 1705315850.0,
                    "content": {"content_type": "text", "parts": ["Hi there!"]},
                }
            },
        },
    }

    md = render_chatgpt_conversation(conv)

    # Check structure
    assert md.startswith("# Test Chat")
    assert "---" in md
    assert "user:" in md
    assert "assistant:" in md
    assert "Hello" in md
    assert "Hi there!" in md
    assert "URL: https://chatgpt.com/c/conv-123" in md


def test_render_chatgpt_conversation_custom_base_url():
    """Custom base_url is used in conversation URL."""
    conv = {
        "id": "conv-123",
        "title": "Test",
        "mapping": {},
    }

    md = render_chatgpt_conversation(conv, base_url="https://custom.example.com")
    assert "https://custom.example.com/c/conv-123" in md


def test_render_chatgpt_conversation_no_id():
    """Conversation without ID omits URL."""
    conv = {
        "title": "Test",
        "mapping": {},
    }

    md = render_chatgpt_conversation(conv)
    assert "URL:" not in md


# ---------------------------------------------------------------------------
# Claude rendering tests
# ---------------------------------------------------------------------------


def test_render_claude_conversation_basic():
    """Basic Claude conversation renders to expected markdown format."""
    conv = {
        "uuid": "abc-123",
        "name": "Test Claude Chat",
        "created_at": "2025-01-15T10:30:00Z",
        "updated_at": "2025-01-15T10:35:00Z",
        "chat_messages": [
            {
                "sender": "human",
                "created_at": "2025-01-15T10:30:00Z",
                "content": [{"type": "text", "text": "Hello Claude"}],
            },
            {
                "sender": "assistant",
                "created_at": "2025-01-15T10:30:30Z",
                "content": [{"type": "text", "text": "Hello! How can I help?"}],
            },
        ],
    }

    md = render_claude_conversation(conv)

    # Check structure
    assert md.startswith("# Test Claude Chat")
    assert "---" in md
    assert "human:" in md
    assert "assistant:" in md
    assert "Hello Claude" in md
    assert "Hello! How can I help?" in md
    assert "URL: https://claude.ai/chat/abc-123" in md


def test_render_claude_conversation_with_thinking():
    """Claude thinking blocks render as collapsible details."""
    conv = {
        "uuid": "abc-123",
        "name": "Test",
        "chat_messages": [
            {
                "sender": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think about this..."},
                    {"type": "text", "text": "Here's my answer."},
                ],
            },
        ],
    }

    md = render_claude_conversation(conv)

    assert "<details>" in md
    assert "<summary>Thinking</summary>" in md
    assert "Let me think about this..." in md
    assert "Here's my answer." in md


def test_render_claude_conversation_with_tool_use():
    """Claude tool use blocks render with markers."""
    conv = {
        "uuid": "abc-123",
        "name": "Test",
        "chat_messages": [
            {
                "sender": "assistant",
                "content": [
                    {"type": "tool_use", "name": "search", "message": "Searching..."},
                ],
            },
        ],
    }

    md = render_claude_conversation(conv)
    assert "**[Tool Use: search]**" in md


# ---------------------------------------------------------------------------
# write_file_if_changed tests
# ---------------------------------------------------------------------------


def test_write_file_if_changed_new(tmp_path):
    """New file is created and returns 'written'."""
    path = tmp_path / "new_file.md"
    content = "Test content"

    status = write_file_if_changed(path, content)

    assert status == "written"
    assert path.exists()
    assert path.read_text() == content


def test_write_file_if_changed_updated(tmp_path):
    """Changed content updates file and returns 'updated'."""
    path = tmp_path / "existing.md"
    path.write_text("Old content")

    status = write_file_if_changed(path, "New content")

    assert status == "updated"
    assert path.read_text() == "New content"


def test_write_file_if_changed_skipped(tmp_path):
    """Unchanged content skips write and returns 'skipped'."""
    path = tmp_path / "existing.md"
    content = "Same content"
    path.write_text(content)

    status = write_file_if_changed(path, content)

    assert status == "skipped"
    assert path.read_text() == content


# ---------------------------------------------------------------------------
# ExportStats tests
# ---------------------------------------------------------------------------


def test_export_stats_total():
    """Total property sums all counts."""
    stats = ExportStats(written=5, updated=3, skipped=10)
    assert stats.total == 18


def test_export_stats_default():
    """Default stats are all zero."""
    stats = ExportStats()
    assert stats.written == 0
    assert stats.updated == 0
    assert stats.skipped == 0
    assert stats.total == 0

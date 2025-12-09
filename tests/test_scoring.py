"""Tests for scoring functions in auto_anki/contexts.py."""

import pytest
from auto_anki.contexts import (
    detect_signals,
    detect_conversation_signals,
    extract_key_terms,
    extract_key_points,
    ChatTurn,
)


class TestDetectSignals:
    """Tests for the detect_signals function."""

    def test_question_with_question_mark(self):
        """Questions with ? should be detected."""
        score, signals = detect_signals(
            "What is gradient descent?",
            "Gradient descent is an optimization algorithm."
        )
        assert signals["question_like"] is True
        assert score >= 1.0

    def test_question_with_question_word(self):
        """Questions starting with question words should be detected."""
        score, signals = detect_signals(
            "Explain how neural networks work",
            "Neural networks are computational models."
        )
        assert signals["question_like"] is True

    def test_definition_detection(self):
        """Definition patterns should be detected."""
        score, signals = detect_signals(
            "Define polymorphism",
            "Polymorphism is defined as the ability of different objects to respond to the same method."
        )
        assert signals["definition_like"] is True

    def test_definition_refers_to(self):
        """'refers to' pattern should be detected."""
        score, signals = detect_signals(
            "What is OOP?",
            "OOP refers to object-oriented programming, a paradigm based on objects."
        )
        assert signals["definition_like"] is True

    def test_bullet_points_detection(self):
        """Bullet points should be detected and scored."""
        score, signals = detect_signals(
            "List the benefits of Python",
            "The main benefits are:\n- Easy to learn\n- Readable syntax\n- Large ecosystem"
        )
        assert signals["bullet_count"] >= 3
        assert score > 0

    def test_numbered_list_detection(self):
        """Numbered lists should be detected."""
        score, signals = detect_signals(
            "Steps to install",
            "1. Download\n2. Install\n3. Configure"
        )
        assert signals["bullet_count"] >= 3

    def test_code_blocks_detection(self):
        """Code blocks should be detected."""
        score, signals = detect_signals(
            "Show me Python code",
            "Here's an example:\n```python\nprint('hello')\n```"
        )
        assert signals["code_blocks"] >= 2  # Opening and closing ```

    def test_headings_detection(self):
        """Markdown headings should be detected."""
        score, signals = detect_signals(
            "Explain the topic",
            "### Introduction\nThis is the intro.\n### Details\nMore details here."
        )
        assert signals["heading_count"] >= 2

    def test_medium_length_bonus(self):
        """Answers of medium length (80-2200 chars) should get bonus."""
        medium_text = "x" * 500
        score, signals = detect_signals("Question?", medium_text)
        assert signals["answer_length"] == 500
        # Medium length should contribute to score
        assert score > 0

    def test_short_answer(self):
        """Very short answers should have low score contribution."""
        score, signals = detect_signals("Question?", "Yes.")
        assert signals["answer_length"] < 80
        # Score shouldn't include the 0.55 medium length bonus
        assert score < 2.0

    def test_long_answer(self):
        """Long answers (>2200 chars) should get reduced bonus."""
        long_text = "x" * 3000
        score, signals = detect_signals("Question?", long_text)
        assert signals["answer_length"] == 3000

    def test_imperative_detection(self):
        """Imperative phrases should be detected."""
        score, signals = detect_signals(
            "Walk me through the setup process",
            "First, download the installer. Then run it."
        )
        assert signals["imperative"] is True

    def test_empty_strings(self):
        """Empty strings should return zero score."""
        score, signals = detect_signals("", "")
        assert score == 0.0
        assert signals["question_like"] is False
        assert signals["definition_like"] is False

    def test_combined_signals(self):
        """Multiple signals should accumulate score."""
        score, signals = detect_signals(
            "What is Python?",
            "Python is defined as a programming language that is easy to learn. Here are the key benefits of using Python:\n- Easy to learn and use\n- Highly readable syntax\n- Large ecosystem of libraries"
        )
        # Should have: question_like + definition_like + bullet_count + medium length
        assert signals["question_like"] is True
        assert signals["definition_like"] is True
        assert signals["bullet_count"] >= 3
        assert score >= 2.0


class TestExtractKeyTerms:
    """Tests for the extract_key_terms function."""

    def test_basic_extraction(self):
        """Should extract meaningful terms from text."""
        terms = extract_key_terms("Machine learning uses neural networks for classification")
        assert len(terms) > 0
        # All terms should be at least 4 characters
        assert all(len(t) >= 4 for t in terms)

    def test_stopword_filtering(self):
        """Stopwords should be filtered out."""
        terms = extract_key_terms("the and for with machine learning")
        assert "the" not in terms
        assert "and" not in terms
        assert "for" not in terms

    def test_term_limit(self):
        """Should respect the limit parameter."""
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        terms = extract_key_terms(text, limit=3)
        assert len(terms) <= 3

    def test_case_insensitivity(self):
        """Terms should be normalized to lowercase."""
        terms = extract_key_terms("Python PYTHON python")
        assert all(t.islower() for t in terms)

    def test_empty_string(self):
        """Empty string should return empty list."""
        terms = extract_key_terms("")
        assert terms == []

    def test_frequency_ordering(self):
        """More frequent terms should appear first."""
        text = "python python python java java rust"
        terms = extract_key_terms(text, limit=3)
        # python appears 3x, java 2x, rust 1x
        assert terms[0] == "python"

    def test_minimum_length(self):
        """Terms must be at least 4 characters."""
        terms = extract_key_terms("the cat sat on mat big dog")
        # "cat", "sat", "mat", "big", "dog" are all 3 chars - should be filtered
        assert "cat" not in terms
        assert "dog" not in terms


class TestExtractKeyPoints:
    """Tests for the extract_key_points function."""

    def test_bullet_points(self):
        """Should extract bullet point content."""
        text = "Overview:\n- First point\n- Second point\n- Third point"
        points = extract_key_points(text)
        assert len(points) >= 3
        assert "First point" in points

    def test_headings(self):
        """Should extract heading content."""
        text = "# Main Topic\n## Subtopic\n### Another Section"
        points = extract_key_points(text)
        assert len(points) >= 1

    def test_limit(self):
        """Should respect the limit parameter."""
        text = "- One\n- Two\n- Three\n- Four\n- Five\n- Six"
        points = extract_key_points(text, limit=3)
        assert len(points) <= 3

    def test_asterisk_bullets(self):
        """Should handle asterisk-style bullets."""
        text = "* Item A\n* Item B"
        points = extract_key_points(text)
        assert len(points) >= 2

    def test_empty_string(self):
        """Empty string should return empty list."""
        points = extract_key_points("")
        assert points == []

    def test_no_bullets_or_headings(self):
        """Plain text without bullets/headings should return empty list."""
        text = "This is just a plain paragraph without any structure."
        points = extract_key_points(text)
        assert len(points) == 0


class TestDetectConversationSignals:
    """Tests for the detect_conversation_signals function."""

    def _make_turn(self, user_prompt: str, assistant_answer: str, turn_index: int = 0) -> ChatTurn:
        """Helper to create a ChatTurn for testing."""
        return ChatTurn(
            context_id=f"test_{turn_index}",
            turn_index=turn_index,
            conversation_id="test_conv",
            source_path="/test/path.md",
            source_title="Test",
            source_url=None,
            user_timestamp=None,
            user_prompt=user_prompt,
            assistant_answer=assistant_answer,
            assistant_char_count=len(assistant_answer),
            score=1.0,
            signals={"question_like": True},
            key_terms=["test"],
            key_points=[],
        )

    def test_empty_turns(self):
        """Empty turns list should return zero score and empty signals."""
        score, signals = detect_conversation_signals([])
        assert score == 0.0
        assert signals == {}

    def test_basic_signals(self):
        """Basic conversation should have standard signals."""
        turns = [
            self._make_turn("What is Python?", "Python is a programming language."),
        ]
        score, signals = detect_conversation_signals(turns)

        assert "turn_count" in signals
        assert signals["turn_count"] == 1
        assert "has_mega_paste" in signals
        assert signals["has_mega_paste"] is False

    def test_mega_paste_detection(self):
        """Mega-paste (>20KB user prompt) should be detected."""
        mega_prompt = "x" * 25000  # 25KB
        turns = [
            self._make_turn(mega_prompt, "Here's my explanation of that document."),
        ]
        score, signals = detect_conversation_signals(turns)

        assert signals["has_mega_paste"] is True
        assert signals["max_user_prompt_chars"] == 25000

    def test_no_mega_paste_under_threshold(self):
        """Prompts under 20KB should not be flagged as mega-paste."""
        large_but_ok_prompt = "x" * 19000  # 19KB
        turns = [
            self._make_turn(large_but_ok_prompt, "Response."),
        ]
        score, signals = detect_conversation_signals(turns)

        assert signals["has_mega_paste"] is False
        assert signals["max_user_prompt_chars"] == 19000

    def test_mega_paste_any_turn(self):
        """Mega-paste should be detected even if only one turn has it."""
        turns = [
            self._make_turn("Short question", "Short answer", turn_index=0),
            self._make_turn("x" * 30000, "Explanation", turn_index=1),  # Mega-paste
            self._make_turn("Follow-up", "More details", turn_index=2),
        ]
        score, signals = detect_conversation_signals(turns)

        assert signals["has_mega_paste"] is True
        assert signals["max_user_prompt_chars"] == 30000

    def test_multi_turn_aggregation(self):
        """Multiple turns should aggregate scores correctly."""
        turns = [
            self._make_turn("Q1", "A1", turn_index=0),
            self._make_turn("Q2", "A2", turn_index=1),
            self._make_turn("Q3", "A3", turn_index=2),
        ]
        score, signals = detect_conversation_signals(turns)

        assert signals["turn_count"] == 3
        assert signals["total_score"] == 3.0  # Each turn has score 1.0
        assert signals["avg_score"] == 1.0

"""Shared pytest fixtures for Auto Anki Agent tests."""

import pytest
from auto_anki.cards import Card, normalize_text
from auto_anki.contexts import ChatTurn, Conversation


@pytest.fixture
def sample_card():
    """A single sample Card for basic tests."""
    return Card(
        deck="Test Deck",
        front="What is Python?",
        back="A programming language.",
        tags=["programming", "python"],
        meta="",
        data_search="",
        front_norm=normalize_text("What is Python?"),
        back_norm=normalize_text("A programming language."),
        note_id=12345,
    )


@pytest.fixture
def sample_cards():
    """Multiple sample Cards for deduplication tests."""
    return [
        Card(
            deck="Test Deck",
            front="What is Python?",
            back="A programming language.",
            tags=["programming"],
            meta="",
            data_search="",
            front_norm=normalize_text("What is Python?"),
            back_norm=normalize_text("A programming language."),
            note_id=12345,
        ),
        Card(
            deck="Test Deck",
            front="Define machine learning",
            back="ML is a subset of AI that enables systems to learn from data.",
            tags=["ml", "ai"],
            meta="",
            data_search="",
            front_norm=normalize_text("Define machine learning"),
            back_norm=normalize_text("ML is a subset of AI that enables systems to learn from data."),
            note_id=12346,
        ),
        Card(
            deck="Test Deck",
            front="What is gradient descent?",
            back="An optimization algorithm used to minimize loss functions.",
            tags=["ml", "optimization"],
            meta="",
            data_search="",
            front_norm=normalize_text("What is gradient descent?"),
            back_norm=normalize_text("An optimization algorithm used to minimize loss functions."),
            note_id=12347,
        ),
    ]


@pytest.fixture
def sample_chat_turn():
    """A single sample ChatTurn for scoring tests."""
    return ChatTurn(
        context_id="test123abc",
        turn_index=0,
        conversation_id="conv123",
        source_path="/test/path.md",
        source_title="Test Conversation",
        source_url="https://example.com",
        user_timestamp="2025-01-15 10:30",
        user_prompt="What is gradient descent?",
        assistant_answer="Gradient descent is an optimization algorithm used to minimize the loss function by iteratively moving toward the steepest descent.",
        assistant_char_count=120,
        score=2.0,
        signals={"question_like": True, "definition_like": True},
        key_terms=["gradient", "descent", "optimization"],
        key_points=["optimization algorithm", "minimize loss"],
    )


@pytest.fixture
def sample_chat_turns():
    """Multiple ChatTurns for conversation-level tests."""
    return [
        ChatTurn(
            context_id="turn1",
            turn_index=0,
            conversation_id="conv1",
            source_path="/test/path.md",
            source_title="Test",
            source_url=None,
            user_timestamp="2025-01-15 10:30",
            user_prompt="What is machine learning?",
            assistant_answer="Machine learning is a type of artificial intelligence that allows systems to learn from data.",
            assistant_char_count=100,
            score=1.5,
            signals={"question_like": True},
            key_terms=["machine", "learning"],
            key_points=[],
        ),
        ChatTurn(
            context_id="turn2",
            turn_index=1,
            conversation_id="conv1",
            source_path="/test/path.md",
            source_title="Test",
            source_url=None,
            user_timestamp="2025-01-15 10:31",
            user_prompt="Can you explain more about neural networks?",
            assistant_answer="Neural networks are computational models inspired by the human brain, consisting of layers of interconnected nodes.",
            assistant_char_count=110,
            score=1.8,
            signals={"question_like": True, "definition_like": True},
            key_terms=["neural", "networks", "brain"],
            key_points=["computational models", "layers"],
        ),
    ]

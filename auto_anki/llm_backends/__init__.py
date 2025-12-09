"""
LLM backend abstraction layer.

This package provides pluggable backends for different agentic CLI tools
(Codex, Claude Code, etc.) while maintaining a consistent interface.

Usage:
    from auto_anki.llm_backends import get_backend, run_backend, LLMConfig

    backend = get_backend("codex")  # or "claude-code"
    config = LLMConfig(model="gpt-5.1", reasoning_effort="medium")
    response = run_backend(backend, prompt, config, run_dir)
"""

from __future__ import annotations

from typing import Dict, Type

from .base import LLMBackend, LLMConfig, LLMResponse, run_backend
from .claude_code import ClaudeCodeBackend
from .codex import CodexBackend

# Registry of available backends
BACKENDS: Dict[str, Type[LLMBackend]] = {
    "codex": CodexBackend,
    "claude-code": ClaudeCodeBackend,
}


def get_backend(name: str) -> LLMBackend:
    """Get an LLM backend instance by name.

    Args:
        name: Backend name ("codex" or "claude-code").

    Returns:
        An instance of the requested backend.

    Raises:
        ValueError: If the backend name is not recognized.
    """
    if name not in BACKENDS:
        available = ", ".join(sorted(BACKENDS.keys()))
        raise ValueError(f"Unknown backend: {name}. Available: {available}")
    return BACKENDS[name]()


def list_backends() -> list[str]:
    """Return a list of available backend names."""
    return sorted(BACKENDS.keys())


__all__ = [
    "LLMBackend",
    "LLMConfig",
    "LLMResponse",
    "run_backend",
    "get_backend",
    "list_backends",
    "CodexBackend",
    "ClaudeCodeBackend",
    "BACKENDS",
]

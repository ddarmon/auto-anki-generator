"""
DEPRECATED: This module has been renamed to llm.py.

All exports are re-exported from auto_anki.llm for backward compatibility.
Update imports to use auto_anki.llm directly.
"""

import warnings

warnings.warn(
    "auto_anki.codex is deprecated, use auto_anki.llm instead",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from llm module for backward compatibility
from auto_anki.llm import (
    MAX_USER_PROMPT_CHARS,
    build_conversation_filter_prompt,
    build_conversation_prompt,
    chunked,
    parse_codex_response_robust,
    run_codex_exec,
    truncate_mega_prompt,
)

__all__ = [
    "build_conversation_prompt",
    "build_conversation_filter_prompt",
    "truncate_mega_prompt",
    "MAX_USER_PROMPT_CHARS",
    "run_codex_exec",
    "parse_codex_response_robust",
    "chunked",
]

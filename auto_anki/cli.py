"""
CLI entrypoint for Auto Anki.

For now this is a thin wrapper over `auto_anki_agent.main`, so we keep
behavior identical while gradually refactoring the internals into the
`auto_anki` package.
"""

from auto_anki_agent import main  # type: ignore

__all__ = ["main"]


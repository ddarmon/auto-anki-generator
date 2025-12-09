"""
Base classes for LLM backend abstraction.

This module provides the abstract base class and dataclasses for
pluggable LLM CLI backends (Codex, Claude Code, etc.).
"""

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class LLMConfig:
    """Backend-agnostic configuration for LLM execution."""

    model: Optional[str] = None
    reasoning_effort: Optional[str] = None  # low/medium/high (Codex only)
    extra_args: List[str] = field(default_factory=list)
    timeout_ms: int = 600000  # 10 minutes default


@dataclass
class LLMResponse:
    """Standardized response from any LLM backend."""

    raw_text: str
    success: bool = True
    error_message: Optional[str] = None
    stdout: str = ""
    stderr: str = ""


class LLMBackend(ABC):
    """Abstract base class for agentic CLI tools.

    Each backend implements the specifics of how to invoke the CLI,
    pass the prompt, and extract the response.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        ...

    @abstractmethod
    def build_command(
        self,
        prompt: str,
        output_path: Path,
        config: LLMConfig,
    ) -> tuple[List[str], Optional[str]]:
        """Build the CLI command to execute.

        Args:
            prompt: The prompt text to send to the LLM.
            output_path: Path where the backend can write output (if needed).
            config: Backend configuration.

        Returns:
            Tuple of (command_args, stdin_input).
            stdin_input is None if the prompt is passed via command args.
        """
        ...

    @abstractmethod
    def extract_response(
        self, output_path: Path, proc: subprocess.CompletedProcess
    ) -> str:
        """Extract the LLM response text from command output.

        Args:
            output_path: Path where output may have been written.
            proc: The completed subprocess.

        Returns:
            The raw response text from the LLM.
        """
        ...


def run_backend(
    backend: LLMBackend,
    prompt: str,
    config: LLMConfig,
    run_dir: Path,
    label: str = "",
) -> LLMResponse:
    """Execute a prompt via the given backend and return the response.

    This is the shared execution logic used by all backends.

    Args:
        backend: The LLM backend to use.
        prompt: The prompt text to send.
        config: Backend configuration.
        run_dir: Directory for saving artifacts (prompts, logs, etc.).
        label: Optional label for artifact filenames.

    Returns:
        LLMResponse with the result or error information.
    """
    # Save prompt for debugging
    prompt_path = run_dir / f"prompt{label}.txt"
    prompt_path.write_text(prompt)
    output_path = run_dir / f"response{label}.json"

    cmd, stdin_input = backend.build_command(prompt, output_path, config)

    try:
        proc = subprocess.run(
            cmd,
            input=stdin_input,
            text=True,
            capture_output=True,
            timeout=config.timeout_ms / 1000,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return LLMResponse(
            raw_text="",
            success=False,
            error_message=f"Timeout after {config.timeout_ms}ms",
        )

    # Save stdout/stderr for debugging
    (run_dir / f"{backend.name}_stdout{label}.log").write_text(proc.stdout)
    (run_dir / f"{backend.name}_stderr{label}.log").write_text(proc.stderr)

    if proc.returncode != 0:
        return LLMResponse(
            raw_text="",
            success=False,
            error_message=f"Exit code {proc.returncode}: {proc.stderr[:500]}",
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

    try:
        raw_text = backend.extract_response(output_path, proc)
    except Exception as e:
        return LLMResponse(
            raw_text="",
            success=False,
            error_message=f"Failed to extract response: {e}",
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

    return LLMResponse(raw_text=raw_text, stdout=proc.stdout, stderr=proc.stderr)


__all__ = ["LLMBackend", "LLMConfig", "LLMResponse", "run_backend"]

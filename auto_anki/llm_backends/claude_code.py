"""
Claude Code CLI backend implementation.

This backend uses the `claude` command to execute prompts.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

from .base import LLMBackend, LLMConfig


class ClaudeCodeBackend(LLMBackend):
    """Backend for the Claude Code CLI tool.

    Claude Code takes prompts via the -p flag and outputs to stdout.
    The --print flag ensures non-interactive output.
    """

    @property
    def name(self) -> str:
        return "claude-code"

    def build_command(
        self,
        prompt: str,
        output_path: Path,
        config: LLMConfig,
    ) -> tuple[List[str], Optional[str]]:
        """Build the claude command.

        Args:
            prompt: The prompt text (passed via -p flag).
            output_path: Not used for Claude Code (outputs to stdout).
            config: Backend configuration.

        Returns:
            Tuple of (command_args, None) since prompt is in args.
        """
        cmd = ["claude", "--print", "-p", prompt]

        if config.model:
            cmd.extend(["--model", config.model])

        # Note: reasoning_effort is not supported by Claude Code
        # (silently ignored, could log debug message if needed)

        cmd.extend(config.extra_args)

        return cmd, None  # No stdin needed, prompt in -p flag

    def extract_response(
        self, output_path: Path, proc: subprocess.CompletedProcess
    ) -> str:
        """Extract response from stdout.

        Args:
            output_path: Not used for Claude Code.
            proc: The completed subprocess.

        Returns:
            The response text from stdout.
        """
        return proc.stdout.strip()


__all__ = ["ClaudeCodeBackend"]

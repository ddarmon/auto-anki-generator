"""
Codex CLI backend implementation.

This backend uses the `codex exec` command to execute prompts.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

from .base import LLMBackend, LLMConfig


class CodexBackend(LLMBackend):
    """Backend for the Codex CLI tool.

    Codex reads prompts from stdin and writes responses to a file
    specified by --output-last-message.
    """

    @property
    def name(self) -> str:
        return "codex"

    def build_command(
        self,
        prompt: str,
        output_path: Path,
        config: LLMConfig,
    ) -> tuple[List[str], Optional[str]]:
        """Build the codex exec command.

        Args:
            prompt: The prompt text (passed via stdin).
            output_path: Path for --output-last-message.
            config: Backend configuration.

        Returns:
            Tuple of (command_args, prompt) since Codex reads from stdin.
        """
        cmd = ["codex", "exec", "-", "--skip-git-repo-check"]

        if config.model:
            cmd.extend(["--model", config.model])

        if config.reasoning_effort:
            cmd.extend(["-c", f"model_reasoning_effort={config.reasoning_effort}"])

        cmd.extend(["--output-last-message", str(output_path)])
        cmd.extend(config.extra_args)

        return cmd, prompt  # Pass prompt via stdin

    def extract_response(
        self, output_path: Path, proc: subprocess.CompletedProcess
    ) -> str:
        """Extract response from the --output-last-message file.

        Args:
            output_path: Path to the output file.
            proc: The completed subprocess (not used for Codex).

        Returns:
            The response text from the output file.
        """
        return output_path.read_text().strip()


__all__ = ["CodexBackend"]

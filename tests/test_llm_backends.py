"""Tests for LLM backend abstraction layer."""

from pathlib import Path
from unittest.mock import MagicMock
import subprocess

import pytest

from auto_anki.llm_backends import (
    LLMBackend,
    LLMConfig,
    LLMResponse,
    get_backend,
    list_backends,
    BACKENDS,
)
from auto_anki.llm_backends.codex import CodexBackend
from auto_anki.llm_backends.claude_code import ClaudeCodeBackend


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_values(self):
        """LLMConfig should have sensible defaults."""
        config = LLMConfig()
        assert config.model is None
        assert config.reasoning_effort is None
        assert config.extra_args == []
        assert config.timeout_ms == 120000

    def test_custom_values(self):
        """LLMConfig should accept custom values."""
        config = LLMConfig(
            model="gpt-5.1",
            reasoning_effort="high",
            extra_args=["--verbose"],
            timeout_ms=60000,
        )
        assert config.model == "gpt-5.1"
        assert config.reasoning_effort == "high"
        assert config.extra_args == ["--verbose"]
        assert config.timeout_ms == 60000


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_success_response(self):
        """LLMResponse should handle successful responses."""
        response = LLMResponse(raw_text='{"cards": []}')
        assert response.success is True
        assert response.raw_text == '{"cards": []}'
        assert response.error_message is None

    def test_error_response(self):
        """LLMResponse should handle error responses."""
        response = LLMResponse(
            raw_text="",
            success=False,
            error_message="Connection failed",
        )
        assert response.success is False
        assert response.error_message == "Connection failed"


class TestBackendRegistry:
    """Tests for backend registry functions."""

    def test_list_backends(self):
        """list_backends should return available backends."""
        backends = list_backends()
        assert "codex" in backends
        assert "claude-code" in backends

    def test_get_backend_codex(self):
        """get_backend should return CodexBackend for 'codex'."""
        backend = get_backend("codex")
        assert isinstance(backend, CodexBackend)
        assert backend.name == "codex"

    def test_get_backend_claude_code(self):
        """get_backend should return ClaudeCodeBackend for 'claude-code'."""
        backend = get_backend("claude-code")
        assert isinstance(backend, ClaudeCodeBackend)
        assert backend.name == "claude-code"

    def test_get_backend_unknown(self):
        """get_backend should raise ValueError for unknown backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("unknown-backend")


class TestCodexBackend:
    """Tests for CodexBackend."""

    def test_name(self):
        """CodexBackend should have name 'codex'."""
        backend = CodexBackend()
        assert backend.name == "codex"

    def test_build_command_basic(self):
        """CodexBackend should build basic command correctly."""
        backend = CodexBackend()
        config = LLMConfig()
        output_path = Path("/tmp/output.json")

        cmd, stdin_input = backend.build_command("test prompt", output_path, config)

        assert cmd[0] == "codex"
        assert cmd[1] == "exec"
        assert "-" in cmd
        assert "--skip-git-repo-check" in cmd
        assert "--output-last-message" in cmd
        assert str(output_path) in cmd
        assert stdin_input == "test prompt"

    def test_build_command_with_model(self):
        """CodexBackend should include model flag when specified."""
        backend = CodexBackend()
        config = LLMConfig(model="gpt-5.1")
        output_path = Path("/tmp/output.json")

        cmd, _ = backend.build_command("test", output_path, config)

        assert "--model" in cmd
        model_idx = cmd.index("--model")
        assert cmd[model_idx + 1] == "gpt-5.1"

    def test_build_command_with_reasoning_effort(self):
        """CodexBackend should include reasoning effort when specified."""
        backend = CodexBackend()
        config = LLMConfig(reasoning_effort="high")
        output_path = Path("/tmp/output.json")

        cmd, _ = backend.build_command("test", output_path, config)

        assert "-c" in cmd
        c_idx = cmd.index("-c")
        assert cmd[c_idx + 1] == "model_reasoning_effort=high"

    def test_build_command_with_extra_args(self):
        """CodexBackend should include extra args."""
        backend = CodexBackend()
        config = LLMConfig(extra_args=["--sandbox=workspace-write"])
        output_path = Path("/tmp/output.json")

        cmd, _ = backend.build_command("test", output_path, config)

        assert "--sandbox=workspace-write" in cmd

    def test_extract_response(self, tmp_path):
        """CodexBackend should extract response from file."""
        backend = CodexBackend()
        output_path = tmp_path / "output.json"
        output_path.write_text('  {"cards": []}  ')

        proc = MagicMock(spec=subprocess.CompletedProcess)
        response = backend.extract_response(output_path, proc)

        assert response == '{"cards": []}'


class TestClaudeCodeBackend:
    """Tests for ClaudeCodeBackend."""

    def test_name(self):
        """ClaudeCodeBackend should have name 'claude-code'."""
        backend = ClaudeCodeBackend()
        assert backend.name == "claude-code"

    def test_build_command_basic(self):
        """ClaudeCodeBackend should build basic command correctly."""
        backend = ClaudeCodeBackend()
        config = LLMConfig()
        output_path = Path("/tmp/output.json")

        cmd, stdin_input = backend.build_command("test prompt", output_path, config)

        assert cmd[0] == "claude"
        assert "--print" in cmd
        assert "-p" in cmd
        p_idx = cmd.index("-p")
        assert cmd[p_idx + 1] == "test prompt"
        assert stdin_input is None  # No stdin for Claude Code

    def test_build_command_with_model(self):
        """ClaudeCodeBackend should include model flag when specified."""
        backend = ClaudeCodeBackend()
        config = LLMConfig(model="claude-opus-4-5-20250514")
        output_path = Path("/tmp/output.json")

        cmd, _ = backend.build_command("test", output_path, config)

        assert "--model" in cmd
        model_idx = cmd.index("--model")
        assert cmd[model_idx + 1] == "claude-opus-4-5-20250514"

    def test_build_command_ignores_reasoning_effort(self):
        """ClaudeCodeBackend should ignore reasoning_effort (not supported)."""
        backend = ClaudeCodeBackend()
        config = LLMConfig(reasoning_effort="high")
        output_path = Path("/tmp/output.json")

        cmd, _ = backend.build_command("test", output_path, config)

        # Should not contain any -c flag or reasoning_effort
        assert "-c" not in cmd
        assert "reasoning_effort" not in str(cmd)

    def test_build_command_with_extra_args(self):
        """ClaudeCodeBackend should include extra args."""
        backend = ClaudeCodeBackend()
        config = LLMConfig(extra_args=["--verbose"])
        output_path = Path("/tmp/output.json")

        cmd, _ = backend.build_command("test", output_path, config)

        assert "--verbose" in cmd

    def test_extract_response(self):
        """ClaudeCodeBackend should extract response from stdout."""
        backend = ClaudeCodeBackend()
        output_path = Path("/tmp/output.json")  # Not used

        proc = MagicMock(spec=subprocess.CompletedProcess)
        proc.stdout = '  {"cards": []}  '

        response = backend.extract_response(output_path, proc)

        assert response == '{"cards": []}'


class TestBackendInterfaceConsistency:
    """Tests to ensure all backends implement the interface consistently."""

    @pytest.mark.parametrize("backend_name", list(BACKENDS.keys()))
    def test_backend_has_name(self, backend_name):
        """All backends should have a name property."""
        backend = get_backend(backend_name)
        assert isinstance(backend.name, str)
        assert len(backend.name) > 0

    @pytest.mark.parametrize("backend_name", list(BACKENDS.keys()))
    def test_backend_build_command_returns_tuple(self, backend_name):
        """All backends should return (cmd, stdin) tuple from build_command."""
        backend = get_backend(backend_name)
        config = LLMConfig(model="test-model")
        output_path = Path("/tmp/test.json")

        result = backend.build_command("test prompt", output_path, config)

        assert isinstance(result, tuple)
        assert len(result) == 2
        cmd, stdin = result
        assert isinstance(cmd, list)
        assert all(isinstance(arg, str) for arg in cmd)
        assert stdin is None or isinstance(stdin, str)

    @pytest.mark.parametrize("backend_name", list(BACKENDS.keys()))
    def test_backend_is_llm_backend_subclass(self, backend_name):
        """All backends should be LLMBackend subclasses."""
        backend = get_backend(backend_name)
        assert isinstance(backend, LLMBackend)

"""Unit tests for prompt template assembly, no LLM calls needed."""
from __future__ import annotations

import litellm

from cadgenbench.baseline.prompt import assemble_messages, assemble_system_prompt

DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"


def _token_count(text: str) -> int:
    """Count tokens for a system prompt string."""
    messages = [{"role": "system", "content": text}]
    return litellm.token_counter(model=DEFAULT_MODEL, messages=messages)


# ---------------------------------------------------------------------------
# System prompt assembly (baseline)
# ---------------------------------------------------------------------------

class TestAssembleSystemPrompt:

    def test_includes_build123d_cheat_sheet(self) -> None:
        prompt = assemble_system_prompt()
        assert "build123d" in prompt.lower()

    def test_includes_build123d_content(self) -> None:
        prompt = assemble_system_prompt()
        assert "BuildPart" in prompt
        assert "export_step" in prompt

    def test_includes_role(self) -> None:
        prompt = assemble_system_prompt()
        assert "expert CAD engineer" in prompt

    def test_includes_workflow(self) -> None:
        prompt = assemble_system_prompt()
        assert "Build geometry" in prompt
        assert "Validation" in prompt
        assert "render" in prompt.lower()
        assert "[DONE]" in prompt

    def test_includes_code_guidelines(self) -> None:
        prompt = assemble_system_prompt()
        assert "output.step" in prompt
        assert "parametric" in prompt.lower()

    def test_includes_render_example(self) -> None:
        prompt = assemble_system_prompt()
        assert "render_step" in prompt
        assert "iso" in prompt

    def test_includes_done_signal(self) -> None:
        prompt = assemble_system_prompt()
        assert "[DONE]" in prompt

    def test_includes_units_in_build123d(self) -> None:
        prompt = assemble_system_prompt()
        assert "millimeters" in prompt.lower() or "mm" in prompt


# ---------------------------------------------------------------------------
# Token budget sanity check
# ---------------------------------------------------------------------------

class TestTokenBudget:

    def test_system_prompt_reasonable_size(self) -> None:
        """System prompt should leave room for conversation within 200K context."""
        prompt = assemble_system_prompt()
        tokens = _token_count(prompt)
        assert tokens < 100_000, (
            f"System prompt is {tokens:,} tokens, too large for 200K context"
        )


# ---------------------------------------------------------------------------
# assemble_messages
# ---------------------------------------------------------------------------

class TestAssembleMessages:

    def test_returns_two_messages(self) -> None:
        messages = assemble_messages("Build a box")
        assert len(messages) == 2

    def test_system_message_first(self) -> None:
        messages = assemble_messages("Build a box")
        assert messages[0]["role"] == "system"

    def test_user_message_contains_task(self) -> None:
        task = "Create a flanged bearing housing"
        messages = assemble_messages(task)
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == task

    def test_system_content_matches_assemble(self) -> None:
        messages = assemble_messages("test")
        standalone = assemble_system_prompt()
        assert messages[0]["content"] == standalone

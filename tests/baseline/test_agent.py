"""Unit tests for the CAD agent, all mocked, no API keys or build123d."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cadgenbench.baseline.types import AgentConfig, AgentResult, CodeExecution, TurnRecord
from cadgenbench.baseline.agent import (
    execute_code,
    extract_code,
    extract_code_blocks,
    format_execution_feedback,
    run_agent,
)
from cadgenbench.baseline.llm import CompletionResult, LLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_completion(
    content: str = "Here's the code:\n\n```python\nprint('hello')\n```",
    tokens: int = 100,
) -> CompletionResult:
    return CompletionResult(
        content=content,
        prompt_tokens=tokens - 30,
        completion_tokens=30,
        total_tokens=tokens,
        model="test-model",
        raw=None,
    )


def _make_done_completion(tokens: int = 100) -> CompletionResult:
    return _make_completion(
        content="The model looks good. [DONE]\n\n```python\nprint('final check')\n```",
        tokens=tokens,
    )


def _make_client(*completions: CompletionResult) -> LLMClient:
    client = MagicMock(spec=LLMClient)
    client.model = "test-model"
    client.complete = MagicMock(side_effect=list(completions))
    return client


def _success_exec_with_artifact(code, work_dir, **kwargs):  # noqa: ANN001
    """``execute_code`` side_effect that satisfies the [DONE] hard gate.

    The agent loop refuses ``[DONE]`` until ``output.step`` exists in the
    working directory.  Real execute_code achieves this by running the
    LLM-written script; mocked executions skip that.  This helper writes
    a placeholder so the hard gate accepts ``[DONE]`` -- pair it with a
    patched ``_auto_validate_and_render`` to keep the test from invoking
    the real STEP validator on the placeholder content.
    """
    del kwargs
    (work_dir / "output.step").write_text("placeholder")
    return CodeExecution(
        code=code, success=True, stdout="ok\n", stderr="",
        files_produced={"output.step": 11}, duration_s=0.1,
    )


# ---------------------------------------------------------------------------
# extract_code_blocks
# ---------------------------------------------------------------------------

class TestExtractCodeBlocks:

    def test_single_block(self) -> None:
        text = "Some text\n\n```python\nprint('hello')\n```\n"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0] == "print('hello')"

    def test_multiple_blocks(self) -> None:
        text = "```python\nfirst()\n```\ntext\n```python\nsecond()\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 2
        assert blocks[0] == "first()"
        assert blocks[1] == "second()"

    def test_no_blocks(self) -> None:
        assert extract_code_blocks("Just plain text.") == []

    def test_non_python_ignored(self) -> None:
        assert extract_code_blocks("```javascript\nconsole.log('hi')\n```") == []

    def test_strips_whitespace(self) -> None:
        text = "```python\n\n  x = 1\n\n```"
        assert extract_code_blocks(text) == ["x = 1"]


# ---------------------------------------------------------------------------
# extract_code (first block only)
# ---------------------------------------------------------------------------

class TestExtractCode:

    def test_returns_first_block(self) -> None:
        text = "```python\nfirst()\n```\ntext\n```python\nsecond()\n```"
        assert extract_code(text) == "first()"

    def test_returns_none_when_no_block(self) -> None:
        assert extract_code("Just plain text.") is None

    def test_strips_whitespace(self) -> None:
        text = "```python\n\n  x = 1\n\n```"
        assert extract_code(text) == "x = 1"


# ---------------------------------------------------------------------------
# execute_code
# ---------------------------------------------------------------------------

class TestExecuteCode:

    def test_success(self, tmp_path: Path) -> None:
        exe = execute_code("print('hello')", tmp_path)
        assert exe.success
        assert "hello" in exe.stdout

    def test_failure(self, tmp_path: Path) -> None:
        exe = execute_code("raise ValueError('boom')", tmp_path)
        assert not exe.success
        assert "boom" in exe.stderr

    def test_file_produced(self, tmp_path: Path) -> None:
        code = "with open('output.step', 'w') as f: f.write('fake step')"
        exe = execute_code(code, tmp_path)
        assert exe.success
        assert "output.step" in exe.files_produced
        assert (tmp_path / "output.step").exists()

    def test_persistent_directory(self, tmp_path: Path) -> None:
        execute_code("with open('data.txt', 'w') as f: f.write('hello')", tmp_path)
        exe2 = execute_code("print(open('data.txt').read())", tmp_path)
        assert exe2.success
        assert "hello" in exe2.stdout

    def test_timeout(self, tmp_path: Path) -> None:
        exe = execute_code("import time; time.sleep(10)", tmp_path, timeout=1)
        assert not exe.success
        assert "timed out" in exe.stderr.lower()

    def test_script_cleaned_up(self, tmp_path: Path) -> None:
        execute_code("print(1)", tmp_path, script_index=0)
        assert not (tmp_path / "_script_0.py").exists()


# ---------------------------------------------------------------------------
# format_execution_feedback
# ---------------------------------------------------------------------------

class TestFormatExecutionFeedback:

    def test_success_feedback(self, tmp_path: Path) -> None:
        (tmp_path / "output.step").write_text("fake")
        exe = CodeExecution(
            code="print('ok')", success=True, stdout="ok\n", stderr="",
            files_produced={"output.step": 4}, duration_s=1.0,
        )
        content = format_execution_feedback([exe], tmp_path)
        text_block = content[0]["text"]
        assert "SUCCESS" in text_block
        assert "output.step" in text_block

    def test_failure_feedback(self, tmp_path: Path) -> None:
        exe = CodeExecution(
            code="bad", success=False, stdout="", stderr="NameError: x",
            files_produced={}, duration_s=0.5,
        )
        content = format_execution_feedback([exe], tmp_path)
        text_block = content[0]["text"]
        assert "FAILED" in text_block
        assert "NameError" in text_block

    def test_png_attached_as_image(self, tmp_path: Path) -> None:
        (tmp_path / "view.png").write_bytes(b"\x89PNG fake")
        exe = CodeExecution(
            code="pass", success=True, stdout="", stderr="",
            files_produced={"view.png": 9}, duration_s=0.1,
        )
        content = format_execution_feedback([exe], tmp_path)
        image_blocks = [b for b in content if b.get("type") == "image_url"]
        assert len(image_blocks) == 1

    def test_multiple_scripts_labeled(self, tmp_path: Path) -> None:
        exes = [
            CodeExecution("a()", True, "", "", {}, 0.1),
            CodeExecution("b()", False, "", "err", {}, 0.2),
        ]
        content = format_execution_feedback(exes, tmp_path)
        text = content[0]["text"]
        assert "Script 0" in text
        assert "Script 1" in text


# ---------------------------------------------------------------------------
# run_agent (baseline)
# ---------------------------------------------------------------------------

class TestRunAgentDoneSignal:

    @patch("cadgenbench.baseline.agent._auto_validate_and_render", return_value=("", None))
    @patch("cadgenbench.baseline.agent.execute_code")
    def test_stops_on_done(self, mock_exec, _mock_auto):
        mock_exec.side_effect = _success_exec_with_artifact
        client = _make_client(_make_done_completion())
        config = AgentConfig(max_iterations=10)
        result = run_agent("Build a box", config=config, client=client)

        assert result.stopped_reason == "done"
        assert result.completed
        assert len(result.turns) == 1


class TestRunAgentMaxIterations:

    @patch("cadgenbench.baseline.agent.execute_code")
    def test_stops_at_max_iter(self, mock_exec):
        mock_exec.return_value = CodeExecution(
            code="print('ok')", success=True, stdout="ok\n", stderr="",
            files_produced={}, duration_s=0.1,
        )
        completions = [_make_completion(tokens=50) for _ in range(3)]
        client = _make_client(*completions)
        config = AgentConfig(max_iterations=3, max_total_tokens=999_999)
        result = run_agent("Build a box", config=config, client=client)

        assert result.stopped_reason == "max_iterations"
        assert not result.completed
        assert len(result.turns) == 3


class TestRunAgentTokenBudget:

    @patch("cadgenbench.baseline.agent.execute_code")
    def test_stops_at_token_budget(self, mock_exec):
        mock_exec.return_value = CodeExecution(
            code="print('ok')", success=True, stdout="ok\n", stderr="",
            files_produced={}, duration_s=0.1,
        )
        completions = [_make_completion(tokens=150) for _ in range(10)]
        client = _make_client(*completions)
        config = AgentConfig(max_iterations=10, max_total_tokens=300)
        result = run_agent("Build a box", config=config, client=client)

        assert result.stopped_reason == "max_tokens"
        assert not result.completed
        # 2 turns * 150 = 300 tokens, then budget check stops turn 3
        assert len(result.turns) == 2


class TestRunAgentNoCodeBlocks:

    @patch("cadgenbench.baseline.agent._auto_validate_and_render", return_value=("", None))
    @patch("cadgenbench.baseline.agent.execute_code")
    def test_handles_no_code(self, mock_exec, _mock_auto):
        no_code = _make_completion(content="I'll think about this...")
        with_code = _make_done_completion()
        mock_exec.side_effect = _success_exec_with_artifact
        client = _make_client(no_code, with_code)
        config = AgentConfig(max_iterations=5)
        result = run_agent("Build a box", config=config, client=client)

        assert len(result.turns) == 2
        assert result.turns[0].code_executions == []
        assert result.stopped_reason == "done"


class TestRunAgentFeedbackGrows:

    @patch("cadgenbench.baseline.agent.execute_code")
    def test_messages_accumulate(self, mock_exec):
        mock_exec.return_value = CodeExecution(
            code="print('ok')", success=True, stdout="ok\n", stderr="",
            files_produced={}, duration_s=0.1,
        )
        msg_counts: list[int] = []

        def recording_complete(messages, **kwargs):
            msg_counts.append(len(messages))
            return _make_completion(tokens=50)

        client = MagicMock(spec=LLMClient)
        client.model = "test-model"
        client.complete = MagicMock(side_effect=recording_complete)

        config = AgentConfig(max_iterations=3, max_total_tokens=999_999)
        run_agent("Build a box", config=config, client=client)

        assert msg_counts[0] == 2       # [system, user]
        assert msg_counts[1] == 4       # + [assistant, feedback]
        assert msg_counts[2] == 6       # + [assistant, feedback]


# ---------------------------------------------------------------------------
# AgentResult.save
# ---------------------------------------------------------------------------

class TestAgentResultSave:
    """AgentResult.save writes per-turn files + baseline_debug.json only.

    It must NOT write result.json (that is the evaluator's pure output)
    and must NOT include any baseline-specific fields outside
    baseline_debug.json.
    """

    def _make_result(self) -> AgentResult:
        exe = CodeExecution(
            code="print('hello')", success=True,
            stdout="hello\n", stderr="",
            files_produced={"output.step": 100},
            duration_s=1.0,
        )
        turn = TurnRecord(
            turn=0,
            assistant_message="Here's the code:\n```python\nprint('hello')\n```",
            code_executions=[exe],
            prompt_tokens=70,
            completion_tokens=30,
            duration_s=2.0,
        )
        return AgentResult(
            task_description="Build a box",
            config=AgentConfig(max_iterations=1),
            turns=[turn],
            total_tokens=100,
            total_duration_s=2.0,
            completed=True,
            stopped_reason="done",
        )

    def test_does_not_write_result_json(self, tmp_path: Path) -> None:
        out = self._make_result().save(tmp_path / "run")
        assert not (out / "result.json").exists()

    def test_writes_baseline_debug_json(self, tmp_path: Path) -> None:
        out = self._make_result().save(tmp_path / "run")
        debug_path = out / "baseline_debug.json"
        assert debug_path.exists()
        debug = json.loads(debug_path.read_text())
        # Only the two fields that aren't recoverable from other on-disk
        # artefacts. Everything else (config, turns, tokens, ...) is
        # already in params.json / conversation.json / turn_N/.
        assert set(debug.keys()) == {"stopped_reason", "total_duration_s"}
        assert debug["stopped_reason"] == "done"

    def test_writes_per_turn_files(self, tmp_path: Path) -> None:
        out = self._make_result().save(tmp_path / "run")
        assert (out / "turn_0" / "code_0.py").exists()
        assert (out / "turn_0" / "stdout_0.txt").exists()

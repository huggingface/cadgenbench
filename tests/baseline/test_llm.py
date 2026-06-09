"""Unit tests for the LLM client, all mocked, no API keys needed."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cadgenbench.baseline.llm import (
    DEFAULT_MODEL,
    CompletionResult,
    LLMClient,
    _cache_breakpoint_indices,
    _short,
)


def _fake_response(
    content: str = "Hello!",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    model: str = "test-model",
    reasoning_content: str | None = None,
    reasoning_tokens: int | None = None,
) -> SimpleNamespace:
    """Build a minimal object matching litellm.completion's return shape."""
    message = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    usage_fields: dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if reasoning_tokens is not None:
        usage_fields["completion_tokens_details"] = SimpleNamespace(
            reasoning_tokens=reasoning_tokens,
        )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(**usage_fields),
        model=model,
    )


SIMPLE_MESSAGES = [{"role": "user", "content": "Say hi"}]


def _make_client(**kwargs) -> LLMClient:
    """Build an ``LLMClient`` for tests.

    Streaming is the production default (needed to bypass HF's 60s
    non-streaming router timeout), but almost all of the unit tests below
    predate it and mock ``litellm.completion`` as a non-iterable response.
    This helper keeps those tests on the non-streaming transport; streaming
    behaviour is covered explicitly by ``TestStreaming`` below.
    """
    kwargs.setdefault("stream", False)
    return LLMClient(**kwargs)


class TestSuccessfulCompletion:

    @patch("cadgenbench.baseline.llm.litellm")
    def test_returns_content(self, mock_llm: MagicMock) -> None:
        mock_llm.completion.return_value = _fake_response(content="world")
        client = _make_client(model="test/model")
        result = client.complete(SIMPLE_MESSAGES)
        assert result.content == "world"

    @patch("cadgenbench.baseline.llm.litellm")
    def test_token_counts(self, mock_llm: MagicMock) -> None:
        mock_llm.completion.return_value = _fake_response(
            prompt_tokens=20, completion_tokens=8
        )
        client = _make_client(model="test/model")
        result = client.complete(SIMPLE_MESSAGES)
        assert result.prompt_tokens == 20
        assert result.completion_tokens == 8
        assert result.total_tokens == 28

    @patch("cadgenbench.baseline.llm.litellm")
    def test_result_type(self, mock_llm: MagicMock) -> None:
        mock_llm.completion.return_value = _fake_response()
        client = _make_client(model="test/model")
        result = client.complete(SIMPLE_MESSAGES)
        assert isinstance(result, CompletionResult)

    @patch("cadgenbench.baseline.llm.litellm")
    def test_forwards_kwargs(self, mock_llm: MagicMock) -> None:
        mock_llm.completion.return_value = _fake_response()
        client = _make_client(model="test/model")
        client.complete(SIMPLE_MESSAGES, temperature=0.5, max_tokens=100)
        mock_llm.completion.assert_called_once_with(
            model="test/model",
            messages=SIMPLE_MESSAGES,
            timeout=300.0,
            temperature=0.5,
            max_tokens=100,
        )


class TestTemperatureLockedModels:
    """Models that reject non-default temperature must have it dropped."""

    @pytest.mark.parametrize("model", [
        "openai/gpt-5",
        "openai/gpt-5.5",
        "openai/gpt-5-codex",
        "anthropic/claude-opus-4-7",
        "anthropic/claude-sonnet-4-6",
    ])
    @patch("cadgenbench.baseline.llm.litellm")
    def test_temperature_dropped_for_locked_model(
        self, mock_llm: MagicMock, model: str,
    ) -> None:
        mock_llm.completion.return_value = _fake_response()
        client = _make_client(model=model)
        client.complete(SIMPLE_MESSAGES, temperature=0.0, max_tokens=100)
        kwargs = mock_llm.completion.call_args.kwargs
        assert "temperature" not in kwargs, (
            f"temperature should be dropped for {model}, got kwargs={kwargs}"
        )

    @patch("cadgenbench.baseline.llm.litellm")
    def test_temperature_dropped_when_reasoning_effort_set(
        self, mock_llm: MagicMock,
    ) -> None:
        mock_llm.completion.return_value = _fake_response()
        client = _make_client(model="openai/gpt-4o")
        client.complete(SIMPLE_MESSAGES, temperature=0.0, reasoning_effort="medium")
        kwargs = mock_llm.completion.call_args.kwargs
        assert "temperature" not in kwargs

    @patch("cadgenbench.baseline.llm.litellm")
    def test_temperature_preserved_for_unlocked_model(
        self, mock_llm: MagicMock,
    ) -> None:
        mock_llm.completion.return_value = _fake_response()
        client = _make_client(model="openai/gpt-4o")
        client.complete(SIMPLE_MESSAGES, temperature=0.0, max_tokens=100)
        kwargs = mock_llm.completion.call_args.kwargs
        assert kwargs.get("temperature") == 0.0


class TestRetryOnTransientError:

    @patch("cadgenbench.baseline.llm.time")
    @patch("cadgenbench.baseline.llm.litellm")
    def test_retries_then_succeeds(
        self, mock_llm: MagicMock, mock_time: MagicMock
    ) -> None:
        from litellm import RateLimitError

        mock_llm.completion.side_effect = [
            RateLimitError("rate limited", "provider", "model"),
            _fake_response(content="recovered"),
        ]
        # Re-bind the exception tuple so isinstance checks work with the real class
        import cadgenbench.baseline.llm as mod
        mock_llm.RateLimitError = RateLimitError
        mock_llm.ServiceUnavailableError = mod.ServiceUnavailableError
        mock_llm.APIConnectionError = mod.APIConnectionError
        mock_llm.Timeout = mod.Timeout

        client = _make_client(model="test/model", max_retries=3)
        result = client.complete(SIMPLE_MESSAGES)
        assert result.content == "recovered"
        assert mock_llm.completion.call_count == 2

    @patch("cadgenbench.baseline.llm.time")
    @patch("cadgenbench.baseline.llm.litellm")
    def test_backoff_sleeps(
        self, mock_llm: MagicMock, mock_time: MagicMock
    ) -> None:
        from litellm import RateLimitError

        mock_llm.completion.side_effect = [
            RateLimitError("rate limited", "provider", "model"),
            _fake_response(),
        ]
        import cadgenbench.baseline.llm as mod
        mock_llm.RateLimitError = RateLimitError
        mock_llm.ServiceUnavailableError = mod.ServiceUnavailableError
        mock_llm.APIConnectionError = mod.APIConnectionError
        mock_llm.Timeout = mod.Timeout

        client = _make_client(model="test/model", initial_backoff=2.0, jitter=0)
        client.complete(SIMPLE_MESSAGES)
        mock_time.sleep.assert_called_once_with(2.0)


class TestRetryOnProviderSpecificError:
    """HF Inference Providers route through LiteLLM's HuggingFaceError which
    inherits from an internal ``BaseLLMException``, not from the typed errors
    in ``RETRYABLE_ERRORS``. Our duck-typed ``status_code`` fallback must
    retry these transient 5xx cases, and must NOT retry permanent 4xx errors
    (400 bad request, 401 unauthorized, 404 not found, etc).
    """

    @staticmethod
    def _provider_error(status: int, message: str = "upstream error") -> Exception:
        """Construct an exception shaped like a LiteLLM provider-specific
        error: arbitrary class with a ``status_code`` attribute."""
        class ProviderError(Exception):
            pass
        e = ProviderError(message)
        e.status_code = status
        return e

    @patch("cadgenbench.baseline.llm.time")
    @patch("cadgenbench.baseline.llm.litellm")
    def test_retries_on_503(self, mock_llm: MagicMock, mock_time: MagicMock) -> None:
        mock_llm.completion.side_effect = [
            self._provider_error(503, "Service Unavailable"),
            _fake_response(content="recovered"),
        ]
        client = _make_client(model="test/model", max_retries=3)
        result = client.complete(SIMPLE_MESSAGES)
        assert result.content == "recovered"
        assert mock_llm.completion.call_count == 2

    @patch("cadgenbench.baseline.llm.time")
    @patch("cadgenbench.baseline.llm.litellm")
    def test_retries_on_504(self, mock_llm: MagicMock, mock_time: MagicMock) -> None:
        mock_llm.completion.side_effect = [
            self._provider_error(504, "Gateway Timeout"),
            _fake_response(content="recovered"),
        ]
        client = _make_client(model="test/model", max_retries=3)
        result = client.complete(SIMPLE_MESSAGES)
        assert result.content == "recovered"

    @patch("cadgenbench.baseline.llm.time")
    @patch("cadgenbench.baseline.llm.litellm")
    def test_retries_on_429(self, mock_llm: MagicMock, mock_time: MagicMock) -> None:
        mock_llm.completion.side_effect = [
            self._provider_error(429, "Too Many Requests"),
            _fake_response(content="recovered"),
        ]
        client = _make_client(model="test/model", max_retries=3)
        result = client.complete(SIMPLE_MESSAGES)
        assert result.content == "recovered"

    @patch("cadgenbench.baseline.llm.time")
    @patch("cadgenbench.baseline.llm.litellm")
    def test_does_not_retry_400(
        self, mock_llm: MagicMock, mock_time: MagicMock
    ) -> None:
        mock_llm.completion.side_effect = self._provider_error(400, "bad request")
        client = _make_client(model="test/model", max_retries=3)
        with pytest.raises(Exception) as exc_info:
            client.complete(SIMPLE_MESSAGES)
        assert exc_info.value.status_code == 400
        assert mock_llm.completion.call_count == 1

    @patch("cadgenbench.baseline.llm.time")
    @patch("cadgenbench.baseline.llm.litellm")
    def test_does_not_retry_exception_without_status_code(
        self, mock_llm: MagicMock, mock_time: MagicMock
    ) -> None:
        mock_llm.completion.side_effect = ValueError("unrelated bug")
        client = _make_client(model="test/model", max_retries=3)
        with pytest.raises(ValueError, match="unrelated bug"):
            client.complete(SIMPLE_MESSAGES)
        assert mock_llm.completion.call_count == 1


class TestFailureAfterMaxRetries:

    @patch("cadgenbench.baseline.llm.time")
    @patch("cadgenbench.baseline.llm.litellm")
    def test_raises_runtime_error(
        self, mock_llm: MagicMock, mock_time: MagicMock
    ) -> None:
        from litellm import RateLimitError

        error = RateLimitError("rate limited", "provider", "model")
        mock_llm.completion.side_effect = error
        import cadgenbench.baseline.llm as mod
        mock_llm.RateLimitError = RateLimitError
        mock_llm.ServiceUnavailableError = mod.ServiceUnavailableError
        mock_llm.APIConnectionError = mod.APIConnectionError
        mock_llm.Timeout = mod.Timeout

        client = _make_client(model="test/model", max_retries=3)
        with pytest.raises(RuntimeError, match="failed after 3 retries"):
            client.complete(SIMPLE_MESSAGES)
        assert mock_llm.completion.call_count == 3

    @patch("cadgenbench.baseline.llm.time")
    @patch("cadgenbench.baseline.llm.litellm")
    def test_chains_original_error(
        self, mock_llm: MagicMock, mock_time: MagicMock
    ) -> None:
        from litellm import RateLimitError

        original = RateLimitError("rate limited", "provider", "model")
        mock_llm.completion.side_effect = original
        import cadgenbench.baseline.llm as mod
        mock_llm.RateLimitError = RateLimitError
        mock_llm.ServiceUnavailableError = mod.ServiceUnavailableError
        mock_llm.APIConnectionError = mod.APIConnectionError
        mock_llm.Timeout = mod.Timeout

        client = _make_client(model="test/model", max_retries=1)
        with pytest.raises(RuntimeError) as exc_info:
            client.complete(SIMPLE_MESSAGES)
        assert exc_info.value.__cause__ is original


class TestModelSelection:

    def test_explicit_model(self) -> None:
        client = LLMClient(model="openai/gpt-4o")
        assert client.model == "openai/gpt-4o"

    def test_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CADGENBENCH_MODEL", "ollama/llama3")
        client = LLMClient()
        assert client.model == "ollama/llama3"

    def test_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CADGENBENCH_MODEL", raising=False)
        client = LLMClient()
        assert client.model == DEFAULT_MODEL


class TestReasoningTokens:
    """Providers that expose reasoning tokens separately should surface them.

    Observed on Together AI for Kimi K2.6 / Qwen3-VL-Thinking / GLM-4.5V via the
    Hugging Face router: ``usage.completion_tokens_details.reasoning_tokens``
    can be substantial (hundreds to thousands) even when visible ``content``
    is short. We capture that breakdown so benchmarks report real text-output
    cost, not aggregated numbers dominated by hidden chain-of-thought.
    """

    @patch("cadgenbench.baseline.llm.litellm")
    def test_extracts_reasoning_tokens_when_present(self, mock_llm: MagicMock) -> None:
        mock_llm.completion.return_value = _fake_response(
            content="ok", completion_tokens=280, reasoning_tokens=272,
        )
        client = _make_client(model="test/model")
        result = client.complete(SIMPLE_MESSAGES)
        assert result.reasoning_tokens == 272
        assert result.completion_tokens == 280

    @patch("cadgenbench.baseline.llm.litellm")
    def test_reasoning_tokens_absent_for_non_thinking_models(
        self, mock_llm: MagicMock,
    ) -> None:
        mock_llm.completion.return_value = _fake_response(content="ok")
        client = _make_client(model="test/model")
        result = client.complete(SIMPLE_MESSAGES)
        assert result.reasoning_tokens is None

    @patch("cadgenbench.baseline.llm.litellm")
    def test_captures_reasoning_content(self, mock_llm: MagicMock) -> None:
        mock_llm.completion.return_value = _fake_response(
            content="ok", reasoning_content="Let me think...",
        )
        client = _make_client(model="test/model")
        result = client.complete(SIMPLE_MESSAGES)
        assert result.reasoning_content == "Let me think..."


class TestThinkingTagStripping:
    """Defensive guard against providers leaking ``<think>`` chatter inline.

    LiteLLM normally separates reasoning into ``message.reasoning_content``,
    but Together AI was observed to occasionally dump the raw reasoning text
    plus the closing ``</think>`` delimiter into ``content`` when responses
    got truncated or streamed unusually. Strip it so downstream parsers (code
    block extraction, [DONE] detection) don't match on reasoning tokens.
    """

    @patch("cadgenbench.baseline.llm.litellm")
    def test_strips_full_think_block(self, mock_llm: MagicMock) -> None:
        mock_llm.completion.return_value = _fake_response(
            content="<think>let me plan</think>Hello world",
        )
        client = _make_client(model="test/model")
        result = client.complete(SIMPLE_MESSAGES)
        assert result.content == "Hello world"

    @patch("cadgenbench.baseline.llm.litellm")
    def test_strips_leaked_close_tag_only(self, mock_llm: MagicMock) -> None:
        # No opening <think>: provider dumped raw CoT then the close delim.
        mock_llm.completion.return_value = _fake_response(
            content="raw reasoning text here</think>Hello world",
        )
        client = _make_client(model="test/model")
        result = client.complete(SIMPLE_MESSAGES)
        assert result.content == "Hello world"

    @patch("cadgenbench.baseline.llm.litellm")
    def test_leaves_clean_content_untouched(self, mock_llm: MagicMock) -> None:
        mock_llm.completion.return_value = _fake_response(content="Just a reply")
        client = _make_client(model="test/model")
        result = client.complete(SIMPLE_MESSAGES)
        assert result.content == "Just a reply"

    @patch("cadgenbench.baseline.llm.litellm")
    def test_empty_content_stays_empty(self, mock_llm: MagicMock) -> None:
        mock_llm.completion.return_value = _fake_response(
            content="", completion_tokens=0,
        )
        client = _make_client(model="test/model")
        result = client.complete(SIMPLE_MESSAGES)
        assert result.content == ""


class TestTokenCounting:

    @patch("cadgenbench.baseline.llm.litellm")
    def test_count_tokens(self, mock_llm: MagicMock) -> None:
        mock_llm.token_counter.return_value = 42
        client = _make_client(model="test/model")
        count = client.count_tokens(SIMPLE_MESSAGES)
        assert count == 42
        mock_llm.token_counter.assert_called_once_with(
            model="test/model", messages=SIMPLE_MESSAGES
        )


class TestBackoffCapAndJitter:
    """The retry budget is now deliberately generous (10 attempts, ~6 min)
    so a single call can outlast HF router brownouts. Two pieces protect
    us from pathological behaviour: (a) ``max_backoff`` caps each sleep
    so the final attempts don't each wait several minutes, and
    (b) ``jitter`` desynchronises retries across parallel workers.
    """

    @patch("cadgenbench.baseline.llm.time")
    @patch("cadgenbench.baseline.llm.litellm")
    def test_backoff_is_capped_at_max_backoff(
        self, mock_llm: MagicMock, mock_time: MagicMock
    ) -> None:
        from litellm import RateLimitError

        mock_llm.completion.side_effect = [
            RateLimitError("rl", "provider", "model")
        ] * 6 + [_fake_response(content="ok")]

        # initial=2, mul=2 → 2,4,8,16,32,64 uncapped. Cap at 10 → 2,4,8,10,10,10.
        client = _make_client(
            model="test/model",
            max_retries=10,
            initial_backoff=2.0,
            backoff_multiplier=2.0,
            max_backoff=10.0,
            jitter=0,
        )
        client.complete(SIMPLE_MESSAGES)
        sleeps = [call.args[0] for call in mock_time.sleep.call_args_list]
        assert sleeps == [2.0, 4.0, 8.0, 10.0, 10.0, 10.0]

    @patch("cadgenbench.baseline.llm.time")
    @patch("cadgenbench.baseline.llm.litellm")
    @patch("cadgenbench.baseline.llm.random.uniform", return_value=-0.2)
    def test_jitter_shortens_sleep(
        self,
        mock_uniform: MagicMock,
        mock_llm: MagicMock,
        mock_time: MagicMock,
    ) -> None:
        from litellm import RateLimitError

        mock_llm.completion.side_effect = [
            RateLimitError("rl", "provider", "model"),
            _fake_response(content="ok"),
        ]
        client = _make_client(
            model="test/model", initial_backoff=10.0, jitter=0.25
        )
        client.complete(SIMPLE_MESSAGES)
        # jitter=-0.2 → sleep = 10 * (1 - 0.2) = 8.0
        mock_time.sleep.assert_called_once_with(8.0)


class TestStreaming:
    """Streaming is the default transport because HF's router enforces a
    60s timeout on non-streaming completions. We drive
    ``litellm.completion(..., stream=True)`` and rebuild the chunk stream
    into a ``ModelResponse`` via ``litellm.stream_chunk_builder``. These
    tests exercise that wrapper end-to-end without hitting the network.
    """

    @patch("cadgenbench.baseline.llm.litellm")
    def test_streams_and_rebuilds_response(self, mock_llm: MagicMock) -> None:
        # ``completion`` returns an iterable of chunks; we don't inspect them
        # because ``stream_chunk_builder`` does that, we only assert the
        # final rebuilt response shape flows through unchanged.
        mock_llm.completion.return_value = iter(["chunk1", "chunk2", "chunk3"])
        mock_llm.stream_chunk_builder.return_value = _fake_response(
            content="streamed reply", prompt_tokens=12, completion_tokens=7,
        )
        client = LLMClient(model="test/model", stream=True)
        result = client.complete(SIMPLE_MESSAGES)
        assert result.content == "streamed reply"
        assert result.prompt_tokens == 12
        assert result.completion_tokens == 7
        # Request must go out with stream=True and include_usage so the
        # final chunk carries real token counts.
        call = mock_llm.completion.call_args
        assert call.kwargs["stream"] is True
        assert call.kwargs["stream_options"] == {"include_usage": True}
        # And stream_chunk_builder must be fed the drained chunks list.
        build_call = mock_llm.stream_chunk_builder.call_args
        assert build_call.args[0] == ["chunk1", "chunk2", "chunk3"]

    @patch("cadgenbench.baseline.llm.litellm")
    def test_streaming_raises_when_no_chunks(self, mock_llm: MagicMock) -> None:
        # Empty streams are pathological (upstream closed the connection
        # with no body) and shouldn't be passed to stream_chunk_builder,
        # which would crash deeper. Surface a clear RuntimeError instead.
        mock_llm.completion.return_value = iter([])
        client = LLMClient(model="test/model", stream=True, max_retries=1)
        with pytest.raises(RuntimeError, match="no chunks"):
            client.complete(SIMPLE_MESSAGES)

    @patch("cadgenbench.baseline.llm.time")
    @patch("cadgenbench.baseline.llm.litellm")
    def test_streaming_retries_on_transient_error(
        self, mock_llm: MagicMock, mock_time: MagicMock,
    ) -> None:
        from litellm import RateLimitError

        mock_llm.completion.side_effect = [
            RateLimitError("rl", "provider", "model"),
            iter(["chunk"]),
        ]
        mock_llm.stream_chunk_builder.return_value = _fake_response(content="ok")
        client = LLMClient(model="test/model", stream=True, max_retries=3)
        result = client.complete(SIMPLE_MESSAGES)
        assert result.content == "ok"
        assert mock_llm.completion.call_count == 2


def _img_msg(role: str = "user", text: str = "t") -> dict:
    """A message carrying one text block + one image block."""
    return {
        "role": role,
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    }


def _cache_blocks(content) -> list:
    """The content blocks in *content* that carry a cache_control marker."""
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and "cache_control" in b]


class TestCacheBreakpointSelection:
    """``_cache_breakpoint_indices`` must pin only byte-stable prefixes so the
    cache actually hits turn-to-turn despite ``_prune_history_images`` rewriting
    older image messages.
    """

    def test_short_history_pins_system_and_first_user(self) -> None:
        # No image has been pruned yet (<= keep_recent images): fall back to the
        # static system + first-user prefix.
        messages = [
            {"role": "system", "content": "sys"},
            _img_msg("user", "task"),
            {"role": "assistant", "content": "ok"},
            _img_msg("user", "feedback-0"),
        ]
        assert _cache_breakpoint_indices(messages, keep_recent=2) == [0, 1]

    def test_marks_system_plus_last_two_pruned_messages(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},   # 0
            {"role": "user", "content": "task"},     # 1 (no image)
            {"role": "assistant", "content": "a0"},  # 2
            _img_msg("user", "fb0"),                  # 3  pruned
            {"role": "assistant", "content": "a1"},  # 4
            _img_msg("user", "fb1"),                  # 5  pruned
            {"role": "assistant", "content": "a2"},  # 6
            _img_msg("user", "fb2"),                  # 7  volatile (kept full)
            {"role": "assistant", "content": "a3"},  # 8
            _img_msg("user", "fb3"),                  # 9  volatile (kept full)
        ]
        # image idxs = [3,5,7,9]; last keep_recent=2 are volatile (7,9), the two
        # most recent already-pruned are 3 and 5. System (0) is always pinned.
        assert _cache_breakpoint_indices(messages, keep_recent=2) == [0, 3, 5]

    def test_never_exceeds_four_breakpoints(self) -> None:
        messages = [{"role": "system", "content": "sys"}]
        for k in range(8):
            messages.append({"role": "assistant", "content": f"a{k}"})
            messages.append(_img_msg("user", f"fb{k}"))
        assert len(_cache_breakpoint_indices(messages, keep_recent=2)) <= 4


class TestAnthropicPromptCaching:
    """Anthropic needs explicit ``cache_control`` markers; OpenAI/Gemini cache
    automatically and must be left untouched."""

    @patch("cadgenbench.baseline.llm.litellm")
    def test_anthropic_gets_cache_control(self, mock_llm: MagicMock) -> None:
        mock_llm.completion.return_value = _fake_response()
        client = _make_client(model="anthropic/claude-opus-4-7")
        messages = [
            {"role": "system", "content": "big static system prompt"},
            {"role": "user", "content": "build a widget"},
        ]
        client.complete(messages, max_tokens=100)
        sent = mock_llm.completion.call_args.kwargs["messages"]
        # System prompt (string) is promoted to a block carrying cache_control.
        assert _cache_blocks(sent[0]["content"])
        assert _cache_blocks(sent[1]["content"])
        # The caller's original list is never mutated.
        assert messages[0]["content"] == "big static system prompt"

    @patch("cadgenbench.baseline.llm.litellm")
    def test_volatile_images_left_uncached(self, mock_llm: MagicMock) -> None:
        mock_llm.completion.return_value = _fake_response()
        client = _make_client(model="anthropic/claude-opus-4-7")
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "a0"},
            _img_msg("user", "fb0"),
            {"role": "assistant", "content": "a1"},
            _img_msg("user", "fb1"),
            {"role": "assistant", "content": "a2"},
            _img_msg("user", "fb2"),
            {"role": "assistant", "content": "a3"},
            _img_msg("user", "fb3"),
        ]
        client.complete(messages, max_tokens=100)
        sent = mock_llm.completion.call_args.kwargs["messages"]
        marked = [i for i, m in enumerate(sent) if _cache_blocks(m["content"])]
        assert marked == [0, 3, 5]

    @patch("cadgenbench.baseline.llm.litellm")
    def test_non_anthropic_untouched(self, mock_llm: MagicMock) -> None:
        mock_llm.completion.return_value = _fake_response()
        client = _make_client(model="openai/gpt-5.5")
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        client.complete(messages, max_tokens=100)
        sent = mock_llm.completion.call_args.kwargs["messages"]
        assert not any(_cache_blocks(m["content"]) for m in sent)


class TestShortErrorFormatter:
    """Regression tests for ``_short``: retry warnings and final errors
    must stay single-line and bounded in length, and HTML error pages
    (from HF router 5xx) must be collapsed to a placeholder rather than
    dumped verbatim.
    """

    @staticmethod
    def _with_status(status: int, message: str) -> Exception:
        class ProviderErr(Exception):
            pass
        e = ProviderErr(message)
        e.status_code = status
        return e

    def test_collapses_html_body(self) -> None:
        html = "<!DOCTYPE html><html>" + "<div>x</div>" * 500 + "</html>"
        out = _short(self._with_status(503, html))
        assert out == "ProviderErr (status 503): <upstream returned HTML error page>"

    def test_collapses_html_body_case_insensitive(self) -> None:
        out = _short(self._with_status(504, "<HTML><BODY>Gateway Timeout</BODY></HTML>"))
        assert "HTML error page" in out

    def test_preserves_short_text_body(self) -> None:
        out = _short(self._with_status(429, "rate limit exceeded for model foo"))
        assert out == "ProviderErr (status 429): rate limit exceeded for model foo"

    def test_truncates_long_text_body(self) -> None:
        out = _short(self._with_status(500, "E" * 10_000))
        assert len(out) < 400
        assert out.endswith("...")

    def test_collapses_newlines(self) -> None:
        out = _short(self._with_status(500, "line one\nline two\r\nline three"))
        assert "\n" not in out and "\r" not in out
        assert "line one line two line three" in out

    def test_without_status_code(self) -> None:
        out = _short(ValueError("boom"))
        assert out == "ValueError: boom"

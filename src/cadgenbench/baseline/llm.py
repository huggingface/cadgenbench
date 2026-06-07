# Copyright 2026 Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model-agnostic LLM client via LiteLLM.

Wraps ``litellm.completion`` with:
- Model selection from env (``CADGENBENCH_MODEL``) or constructor arg
- Exponential-backoff retry on transient errors
- Token counting via ``litellm.token_counter``
- Auto-loads ``.env`` for API keys (via python-dotenv)
"""
from __future__ import annotations

import logging
import os
import random
import re
import time

from dotenv import load_dotenv

load_dotenv()
from dataclasses import dataclass, field
from typing import Any

import litellm
from litellm import (
    APIConnectionError,
    BadGatewayError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    UnsupportedParamsError,
)

# Silence LiteLLM's unconditional ``print()`` banners ("Give Feedback / Get
# Help", "Provider List: ...") and its full-body dumps of upstream HTML
# error pages on 5xx. Our retry loop already logs a compact one-line
# WARNING on transient errors; the raw HTML is preserved on the exception
# object for anyone who needs it.
litellm.suppress_debug_info = True

logger = logging.getLogger(__name__)

# HF's router returns a full HTML error page on 5xx. Embedded verbatim in
# log lines / RuntimeError messages that's ~1 KB of CSS + markup. Collapse
# to something human-skimmable.
_MAX_ERROR_MSG = 240


_HTML_BODY_RE = re.compile(r"<!?\s*(doctype|html|head|body|script|style)", re.IGNORECASE)


def _short(exc: Exception) -> str:
    """Render an exception as a single-line, bounded-length diagnostic.

    Strips newlines and truncates long bodies so retry warnings and final
    error messages stay readable. When the embedded body is an HTML error
    page (typical of HF router 5xx), we drop the body entirely, the
    status code carries all the useful information.
    """
    status = getattr(exc, "status_code", None)
    tag = f"{type(exc).__name__}" + (f" (status {status})" if status else "")

    msg = str(exc)
    if _HTML_BODY_RE.search(msg[:500]):
        return f"{tag}: <upstream returned HTML error page>"

    msg = re.sub(r"\s+", " ", msg.replace("\n", " ").replace("\r", " ")).strip()
    if len(msg) > _MAX_ERROR_MSG:
        msg = msg[:_MAX_ERROR_MSG] + "..."
    return f"{tag}: {msg}" if msg else tag

DEFAULT_MODEL = "anthropic/claude-opus-4-7"
RETRYABLE_ERRORS = (
    RateLimitError,
    ServiceUnavailableError,
    APIConnectionError,
    Timeout,
    BadGatewayError,
    InternalServerError,
)

# HTTP status codes we consider transient server/throttle conditions. Used as
# a duck-typed fallback for provider-specific LiteLLM error classes (e.g.
# ``HuggingFaceError``) that don't inherit from the typed errors above but
# still carry a ``status_code`` attribute.
#   408 Request Timeout        425 Too Early                429 Too Many Requests
#   500 Internal Server Error  502 Bad Gateway              503 Service Unavailable
#   504 Gateway Timeout        (409 retried, some providers emit it for
#                               ephemeral-state contention)
_TRANSIENT_HTTP_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# How many of the most recent image-bearing messages keep their images in the
# outgoing request. Renders accumulate one image per turn; resending the whole
# history bloats the request and trips provider "many-image" limits (e.g.
# Anthropic drops the per-image cap to 2000px once a request carries many
# images). The task's input image (first image-bearing message) is always kept.
_KEEP_RECENT_IMAGE_MSGS = 2


def _prune_history_images(
    messages: list[dict[str, Any]],
    keep_recent: int = _KEEP_RECENT_IMAGE_MSGS,
) -> list[dict[str, Any]]:
    """Return a copy of *messages* that carries images only in the task's
    input message and the most recent ``keep_recent`` image-bearing messages.

    Older per-turn renders are dropped (replaced with a short text note) to
    bound request size; all text is preserved, so the model still sees the
    full turn-by-turn history and knows the earlier turns happened. The input
    list is never mutated, so the persisted conversation keeps every image.
    """
    def _has_image(msg: dict[str, Any]) -> bool:
        content = msg.get("content")
        return isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "image_url" for b in content
        )

    image_idxs = [i for i, m in enumerate(messages) if _has_image(m)]
    keep = set(image_idxs[-keep_recent:])
    # Always keep the task's input image, which lives in the first message
    # (the initial prompt). Generation tasks have one; editing tasks don't
    # (input is a STEP), so this adds nothing there and we keep strictly the
    # last `keep_recent` turn renders.
    if messages and _has_image(messages[0]):
        keep.add(0)
    if len(keep) >= len(image_idxs):
        return messages  # nothing to drop

    pruned: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if i in image_idxs and i not in keep:
            kept = [
                b for b in msg["content"]
                if not (isinstance(b, dict) and b.get("type") == "image_url")
            ]
            n_dropped = len(msg["content"]) - len(kept)
            kept.append({
                "type": "text",
                "text": f"[{n_dropped} render image(s) from this earlier turn "
                        "omitted to bound request size]",
            })
            pruned.append({**msg, "content": kept})
        else:
            pruned.append(msg)
    return pruned

# Retry tuning: HF Inference Providers (Together/Novita/SambaNova via the HF
# router) occasionally brownout on 502/503/504 for stretches of 30s–3min at
# a time (observed 2026-04-23). 10 attempts with exponential backoff
# capped at 60s gives each call ~6 min of runway:
#   2, 4, 8, 16, 32, 60, 60, 60, 60, 60 → ≈362s ≈ 6 min total wait
# Successful calls pay zero cost, the retry loop only fires on
# transient 5xx/429. Jitter (±25%) on each sleep avoids thundering-herd
# synchronisation when multiple workers hit the same brownout.
DEFAULT_MAX_RETRIES = 10
DEFAULT_INITIAL_BACKOFF = 2.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0
DEFAULT_MAX_BACKOFF = 60.0
DEFAULT_JITTER = 0.25

# Streaming is on by default. Hugging Face's router enforces a hard 60s
# timeout on non-streaming requests; long-CoT models (Kimi K2.6 in
# particular) routinely exceed that and 504 even when the upstream is
# healthy. Streaming bypasses that router timeout because bytes start
# flowing within the first few hundred ms of generation. We rebuild the
# chunks into a non-streaming-shaped ``ModelResponse`` via
# ``litellm.stream_chunk_builder`` so nothing downstream changes.
DEFAULT_STREAM = True


@dataclass(frozen=True)
class CompletionResult:
    """Structured response from an LLM completion call.

    ``completion_tokens`` is the provider-reported total (text + reasoning).
    ``reasoning_tokens`` (when available) reports the subset spent on hidden
    chain-of-thought, exposed by providers that return
    ``usage.completion_tokens_details.reasoning_tokens``. ``reasoning_content``
    mirrors LiteLLM's ``message.reasoning_content`` and is the decoded CoT
    text when the provider returns it separately from the answer.
    """

    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    raw: Any = field(repr=False)
    reasoning_tokens: int | None = None
    reasoning_content: str | None = None


# Matches a full ``<think>...</think>`` block (greedy across newlines), with
# any trailing whitespace, so we can strip it from ``content``. Some providers
#, observed on Together AI for Kimi K2.6, occasionally emit the closing
# ``</think>`` delimiter inline inside ``content`` instead of splitting it
# into ``reasoning_content``. LiteLLM's own post-processing handles most
# cases cleanly, but we strip defensively in case a future provider regresses.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# Matches a leaked ``</think>`` close tag without a preceding ``<think>`` -
# i.e. the provider dumped raw reasoning text then the closing delimiter then
# the real answer. We take everything after the last ``</think>``.
_LEAKED_CLOSE_RE = re.compile(r"^.*</think>\s*", re.DOTALL)


def _strip_thinking(content: str) -> str:
    """Remove any leaked ``<think>...</think>`` chatter from response text.

    Handles three cases:
    - A fully-wrapped block: ``<think>...</think>answer`` -> ``answer``.
    - A leaked close delimiter only: ``rawcot</think>answer`` -> ``answer``.
    - No thinking markers: returned unchanged.
    """
    if not content:
        return content
    stripped = _THINK_BLOCK_RE.sub("", content)
    if "</think>" in stripped:
        stripped = _LEAKED_CLOSE_RE.sub("", stripped, count=1)
    return stripped


def _fake_completion(model: str) -> CompletionResult:
    """Canned completion for offline reproduction (CADGENBENCH_FAKE_LLM).

    Emits a valid build123d block that exports ``output.step`` and never
    signals ``[DONE]``, so the agent keeps rendering every turn until it hits
    its wall-clock timeout, exercising the render-pool teardown path that
    nested process pools used to deadlock on.
    """
    import random

    dims = (random.randint(5, 40), random.randint(5, 40), random.randint(5, 40))
    content = (
        "Building the part now.\n\n"
        "```python\n"
        "from build123d import *\n"
        f"b = Box({dims[0]}, {dims[1]}, {dims[2]})\n"
        "export_step(b, 'output.step')\n"
        "print('exported')\n"
        "```\n"
    )
    return CompletionResult(
        content=content,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        model=model,
        raw=None,
    )


class LLMClient:
    """Thin wrapper around LiteLLM with retry and token counting.

    Args:
        model: LiteLLM model string (e.g. ``"anthropic/claude-sonnet-4-6"``).
            Falls back to ``CADGENBENCH_MODEL`` env var, then ``DEFAULT_MODEL``.
        max_retries: Number of retry attempts on transient errors.
        initial_backoff: Seconds to wait before the first retry.
        backoff_multiplier: Factor to multiply the backoff by after each retry.
        max_backoff: Hard cap on backoff seconds between any two attempts.
        jitter: Fractional random jitter applied to each sleep (0.25 → ±25%).
        stream: When True (default), use streaming transport. Required for
            long-running calls through HF's router (60s non-streaming
            timeout). The streamed chunks are rebuilt into a standard
            non-streaming response so the caller sees an identical
            ``CompletionResult``.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
        backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
        jitter: float = DEFAULT_JITTER,
        timeout: float = 300.0,
        stream: bool = DEFAULT_STREAM,
    ) -> None:
        self.model = model or os.environ.get("CADGENBENCH_MODEL") or DEFAULT_MODEL
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_multiplier = backoff_multiplier
        self.max_backoff = max_backoff
        self.jitter = jitter
        self.timeout = timeout
        self.stream = stream

    def complete(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> CompletionResult:
        """Send a chat completion request with retry on transient errors.

        Args:
            messages: OpenAI-style message list
                (``[{"role": "user", "content": "..."}]``).
            **kwargs: Extra arguments forwarded to ``litellm.completion``
                (e.g. ``temperature``, ``max_tokens``).

        Returns:
            CompletionResult with the response text and token usage.

        Raises:
            litellm.APIError: On non-transient API errors.
            RuntimeError: After exhausting all retries on transient errors.
        """
        # Offline test hook: when CADGENBENCH_FAKE_LLM is set, return a canned
        # response with zero network. Lets the nested-process-pool teardown be
        # reproduced locally in seconds (env vars propagate to spawned
        # child/grandchild workers). Never enabled in normal runs.
        if os.environ.get("CADGENBENCH_FAKE_LLM"):
            return _fake_completion(self.model)
        # Several providers reject a non-default temperature: Anthropic's
        # adaptive-thinking models, OpenAI's GPT-5 family (only accepts
        # temperature=1), and any call where we've enabled reasoning_effort.
        # Silently drop the kwarg in those cases rather than letting LiteLLM
        # raise UnsupportedParamsError.
        if "temperature" in kwargs and (
            kwargs.get("reasoning_effort") is not None or self._is_temperature_locked()
        ):
            kwargs.pop("temperature")

        # Normalise reasoning_effort: drop if None, else translate per provider.
        # LiteLLM's unified reasoning_effort still emits the legacy Anthropic
        # format (thinking.type=enabled + budget_tokens), which Opus 4.7 rejects
        # with a 400 ("use thinking.type.adaptive and output_config.effort").
        # For Anthropic adaptive-thinking models we bypass LiteLLM's mapping
        # and pass the native adaptive + effort params directly.
        effort = kwargs.pop("reasoning_effort", None)
        if effort is not None:
            if self._is_anthropic_adaptive_model():
                # Opus 4.7 only accepts the adaptive thinking API:
                #   thinking={"type":"adaptive"} + output_config={"effort": ...}
                # LiteLLM's unified reasoning_effort still emits the legacy
                # enabled+budget_tokens format, so we bypass it and pass the
                # native Anthropic params directly. LiteLLM forwards unknown
                # top-level kwargs into the Anthropic request body.
                kwargs["thinking"] = {"type": "adaptive"}
                # Anthropic adaptive-thinking models do not accept
                # `minimal`; use the closest supported low-effort setting.
                kwargs["output_config"] = {
                    "effort": "low" if effort == "minimal" else effort
                }
            else:
                # OpenAI / Gemini / others: let LiteLLM translate.
                kwargs["reasoning_effort"] = effort

        if (
            self.model.startswith("huggingface/")
            and "api_key" not in kwargs
            and os.environ.get("CADGENBENCH_HF_INFERENCE_TOKEN")
        ):
            # HF Jobs use HF_TOKEN for Hub/data access. Hugging Face Inference
            # Providers may require a different token, so pass it explicitly to
            # LiteLLM instead of overloading HF_TOKEN.
            kwargs["api_key"] = os.environ["CADGENBENCH_HF_INFERENCE_TOKEN"]

        # Only the task input image + the last few renders ride along; older
        # per-turn renders are dropped (text kept). Bounds request size and
        # avoids provider many-image limits. Caller's list stays intact.
        payload = _prune_history_images(messages)

        last_error: Exception | None = None
        backoff = self.initial_backoff

        for attempt in range(1, self.max_retries + 1):
            try:
                if self.stream:
                    response = self._stream_completion(payload, **kwargs)
                else:
                    response = litellm.completion(
                        model=self.model,
                        messages=payload,
                        timeout=self.timeout,
                        **kwargs,
                    )
                return self._parse_response(response)

            except UnsupportedParamsError as exc:
                # Not all providers accept every reasoning_effort level (e.g.
                # OpenAI's gpt-5.5 rejects "minimal"). Fail fast with a clear
                # message instead of a raw provider traceback; this is a config
                # error, not a transient one, so don't retry.
                raise RuntimeError(
                    f"reasoning_effort={effort!r} is not supported for model "
                    f"{self.model!r}. Use a different --reasoning-effort "
                    "(low/medium/high) or omit it to use the provider default."
                ) from exc

            except RETRYABLE_ERRORS as exc:
                last_error = exc

            except Exception as exc:
                # Duck-typed fallback for provider-specific LiteLLM errors
                # that don't inherit from our typed ``RETRYABLE_ERRORS``.
                # ``HuggingFaceError`` is the motivating case: it wraps 5xx
                # from HF's router but inherits from LiteLLM's internal
                # ``BaseLLMException``, not from ``ServiceUnavailableError``
                # or ``InternalServerError``. We key off ``status_code`` so
                # any future provider subclass is handled too.
                status = getattr(exc, "status_code", None)
                if status not in _TRANSIENT_HTTP_STATUS:
                    raise
                last_error = exc

            if attempt < self.max_retries:
                # Cap the exponential growth so the last few attempts don't
                # wedge us for unreasonably long (2,4,8,16,32,60,60,...).
                sleep_for = min(backoff, self.max_backoff)
                if self.jitter:
                    # Multiplicative jitter: desync retries across workers
                    # so we don't all hammer the router at the same instant.
                    sleep_for *= 1.0 + random.uniform(-self.jitter, self.jitter)
                logger.warning(
                    "Transient error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt,
                    self.max_retries,
                    sleep_for,
                    _short(last_error),
                )
                time.sleep(sleep_for)
                backoff *= self.backoff_multiplier

        raise RuntimeError(
            f"LLM call failed after {self.max_retries} retries: {_short(last_error)}"
        ) from last_error

    def _stream_completion(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Drive a streaming completion and rebuild it as a ``ModelResponse``.

        HF's router enforces a 60s non-streaming timeout; streaming bypasses
        that because bytes start flowing within the first generation step.
        We ask for ``stream_options={"include_usage": True}`` so the final
        chunk carries real ``usage`` numbers, then replay all chunks through
        ``litellm.stream_chunk_builder``, which returns a
        ``ModelResponse`` whose shape matches the non-streaming path exactly
        (``choices[0].message.content``, ``.reasoning_content``,
        ``usage.prompt_tokens``, ``usage.completion_tokens_details``).
        Errors can surface either on the initial call or mid-iteration; in
        either case the exception propagates up to ``complete``'s retry loop.
        """
        stream_options = kwargs.pop("stream_options", {"include_usage": True})
        stream = litellm.completion(
            model=self.model,
            messages=messages,
            timeout=self.timeout,
            stream=True,
            stream_options=stream_options,
            **kwargs,
        )
        chunks = list(stream)
        if not chunks:
            raise RuntimeError("LLM stream returned no chunks")
        return litellm.stream_chunk_builder(chunks, messages=messages)

    # Anthropic models that use the new adaptive-thinking API
    # (thinking.type=adaptive + output_config.effort).  On Opus 4.7 this is
    # the only supported thinking mode; Opus 4.6 and Sonnet 4.6 also accept
    # it and have deprecated the legacy enabled+budget_tokens format.
    _ANTHROPIC_ADAPTIVE_PATTERNS = (
        "opus-4-8", "opus-4-7", "opus-4-6", "sonnet-4-6", "claude-mythos",
    )

    # Models that reject a non-default `temperature` kwarg outright. Covers
    # the Anthropic adaptive-thinking models above (they treat temperature
    # the same as reasoning_effort) plus OpenAI's GPT-5 family, which only
    # accepts temperature=1.  Substring match against `self.model`, so
    # "gpt-5" catches "openai/gpt-5", "openai/gpt-5.5", "gpt-5-codex", etc.
    _TEMPERATURE_LOCKED_PATTERNS = _ANTHROPIC_ADAPTIVE_PATTERNS + ("gpt-5",)

    def _is_temperature_locked(self) -> bool:
        return any(p in self.model for p in self._TEMPERATURE_LOCKED_PATTERNS)

    def _is_anthropic_adaptive_model(self) -> bool:
        if "anthropic" not in self.model and "claude" not in self.model:
            return False
        return any(p in self.model for p in self._ANTHROPIC_ADAPTIVE_PATTERNS)

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens for a message list without making an API call."""
        return litellm.token_counter(model=self.model, messages=messages)

    def _parse_response(self, response: Any) -> CompletionResult:
        usage = response.usage
        message = response.choices[0].message
        # Some providers return content=None when the entire output budget was
        # consumed by reasoning/thinking tokens and no text block was emitted.
        # Downstream code assumes str; coerce to empty string here so callers
        # see an empty response instead of a TypeError deep in a regex call.
        content = _strip_thinking(message.content or "")

        reasoning_content = getattr(message, "reasoning_content", None)
        reasoning_tokens: int | None = None
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            reasoning_tokens = getattr(details, "reasoning_tokens", None)

        # Some providers omit one or more usage fields (None). Coerce to ints
        # so the agent's running ``total_tokens`` budget arithmetic can't hit a
        # TypeError mid-run; fall back to prompt+completion when total is absent.
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0) or (
            prompt_tokens + completion_tokens
        )

        if not content and completion_tokens > 0:
            logger.warning(
                "LLM returned empty content with %d completion tokens consumed "
                "(reasoning_tokens=%s). Raise max_tokens if this is an open-weights "
                "reasoning model whose chain-of-thought burned the output budget.",
                completion_tokens,
                reasoning_tokens,
            )

        return CompletionResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=response.model,
            raw=response,
            reasoning_tokens=reasoning_tokens,
            reasoning_content=reasoning_content,
        )

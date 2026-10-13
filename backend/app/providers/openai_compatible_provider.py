"""Real OpenAI-compatible `LLMProvider` adapter (Phase 4 plan P4.5, `docs/decisions.md
§13` M-4/M-5).

One class covers both a real hosted OpenAI endpoint and any local server exposing
the same chat-completions wire shape (Ollama, vLLM, LM Studio, MTPLX, ...) — the
difference is `base_url`/`model` in `config/reviewer-provider.yaml`, never code.
Lives outside `backend/app/services/` for the same reason as
`anthropic_provider.py` (the AST no-network scan over `services/**.py`).

Live-verified against a local MTPLX server (`http://127.0.0.1:8000`, model
`qwen3-coder-30b-a3b-instruct`) during Phase 4 P4.5: reasoning-heavy models can
spend an entire `max_tokens` budget on `reasoning_content` (a separate field from
`content`) and return `finish_reason == "length"` with `content` **empty**. This
adapter does not treat an empty `content` as a valid answer: `text` is `content`
only, never a silent fallback to `reasoning_content` (which is chain-of-thought
prose, not a `ReviewerVerdict`-shaped answer, and would only defeat R-7's
downstream JSON-parse retry by masking the truncation as a parseable-but-wrong
response).

**P4.8 finding, more severe than the P4.5 probe above suggested:** against a real
reviewer prompt (~7.5k prompt tokens of structured evidence, not a trivial "say
hi"), this model's hidden reasoning consumed the ENTIRE `max_tokens=1024` budget
on every one of 30 live calls with ZERO progress toward an answer
(`completion_tokens == max_tokens`, `content == ""` every time,
`schema_validity=0.0`). Raising `max_tokens` to compensate does not scale here —
at this model's observed ~70 tok/s local decode speed, a budget large enough to
survive its reasoning depth on a real prompt would make a full arm-4 population
(132 gatable decisions × 3 samples, plus repair retries) take on the order of a
day. The actual fix is `extra_body` (below): most reasoning-capable
OpenAI-compatible servers (vLLM/SGLang-family, this MTPLX server included) accept
a passthrough `chat_template_kwargs: {"enable_thinking": false}` that skips
reasoning entirely and returns a direct answer — confirmed live: the same prompt
shape that exhausted 1024 reasoning tokens returns valid `content` in single-digit
completion tokens with thinking disabled.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import openai
from openai.types.chat import ChatCompletionMessageParam

from app.services.llm_provider import (
    LLMRequest,
    LLMResponse,
    ProviderError,
    ProviderRefusal,
    ProviderTimeout,
    TokenUsage,
)
from app.services.prompt_builder import Segment, UntrustedBlock

PROVIDER_ID = "openai_compatible"
APPEAL_MODES = ("include", "omit")


def _segment_text(segment: Segment, *, omit_appeal: bool) -> str:
    text = segment.text
    if omit_appeal and isinstance(segment, UntrustedBlock):
        # Appeals-off arm (P5-8): keep the markers, drop the body, so the model sees
        # exactly what an appeal-free case renders as.
        lines = text.split("\n")
        return "\n".join([lines[0], "", lines[-1]])
    return text


def _render_messages(
    request: LLMRequest, *, omit_appeal: bool = False
) -> list[ChatCompletionMessageParam]:
    user_text = "\n\n".join(
        _segment_text(segment, omit_appeal=omit_appeal) for segment in request.segments
    )
    return [
        {"role": "system", "content": request.system},
        {"role": "user", "content": user_text},
    ]


@dataclass(frozen=True)
class OpenAICompatibleReviewerProvider:
    """`complete(LLMRequest) -> LLMResponse` against any OpenAI chat-completions
    -compatible endpoint. `base_url`, `model`, `api_key`, `timeout_s`, and
    `extra_body` all come from `config/reviewer-provider.yaml` via
    `providers.factory.build_provider`.

    `extra_body` is an arbitrary passthrough dict merged into the raw request
    body — deliberately NOT a named field like `enable_thinking` on this
    dataclass, because that flag is a vendor-specific extension (Qwen3/vLLM
    `chat_template_kwargs`), not part of the OpenAI wire contract; a real hosted
    OpenAI endpoint rejects unrecognized top-level fields with a 400, so this
    stays opt-in via config rather than baked into the adapter."""

    model: str
    base_url: str
    api_key: str
    timeout_s: float = 60.0
    extra_body: dict | None = None
    # "omit" renders every appeal as absent (the P5-8 appeals-off arm).
    appeals: str = "include"

    def __post_init__(self) -> None:
        if self.appeals not in APPEAL_MODES:
            raise ValueError(f"appeals must be one of {APPEAL_MODES}, got {self.appeals!r}")

    def complete(self, request: LLMRequest) -> LLMResponse:
        client = openai.OpenAI(
            base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_s
        )
        started = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=_render_messages(request, omit_appeal=self.appeals == "omit"),
                extra_body=self.extra_body,
            )
        except openai.APITimeoutError as exc:
            raise ProviderTimeout(str(exc)) from exc
        except openai.OpenAIError as exc:
            # Covers APIStatusError, APIConnectionError, RateLimitError, and every
            # other SDK-specific failure not already narrowed above.
            raise ProviderError(str(exc)) from exc
        latency_ms = (time.monotonic() - started) * 1000.0

        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "content_filter":
            raise ProviderRefusal(f"content filtered by provider: model={response.model}")

        # `content` only — never a silent fallback to a reasoning-only field. An
        # empty string here (e.g. finish_reason == "length" with the whole budget
        # spent on reasoning) is deliberately returned as-is, so it fails to parse
        # as a ReviewerVerdict downstream and hits R-7's repair retry, rather than
        # being disguised as a real, if terse, answer.
        text = choice.message.content or ""
        usage_obj = response.usage
        usage = TokenUsage(
            prompt_tokens=usage_obj.prompt_tokens if usage_obj else 0,
            completion_tokens=usage_obj.completion_tokens if usage_obj else 0,
            total_tokens=usage_obj.total_tokens if usage_obj else 0,
        )
        return LLMResponse(
            text=text,
            model_snapshot=response.model,
            latency_ms=latency_ms,
            usage=usage,
        )


__all__ = ["APPEAL_MODES", "OpenAICompatibleReviewerProvider", "PROVIDER_ID"]

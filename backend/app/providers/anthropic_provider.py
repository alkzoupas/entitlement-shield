"""Real Anthropic `LLMProvider` adapter (Phase 4 plan P4.4, `docs/decisions.md §13`
M-4/M-5).

Lives outside `backend/app/services/` — `test_llm_provider.py`'s AST scan forbids
network-capable imports anywhere under `services/**.py` (L-7's "no network
dependency in Phase 3"), and this adapter genuinely needs one. Structural
`LLMProvider` conformance only, no inheritance, mirroring
`fake_reviewer.FakeReviewerProvider`.

No live Anthropic API key is available in this environment yet (Phase 4, deferred
by the user) — this module is exercised only against a mocked `anthropic.Anthropic`
client in tests. `PROVIDER_ID`/model id are read from config so
`config/reviewer-manifest.json` hashes the actually-configured snapshot, not a
hardcoded string.

`LLMRequest.temperature` is deliberately NOT forwarded to `messages.create()`: the
installed `anthropic` SDK (verified by grepping the package) has no `temperature`
parameter anywhere in its `create()` overloads or type params — this generation of
the Messages API dropped standalone sampling-temperature control. `temperature` in
`LLMRequest` stays a request-shape field the `openai_compatible` adapter still
honors; for Anthropic it is a no-op, not an unverified guess at wire-level support
(no real API key is available here to confirm an undocumented `extra_body` path).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import anthropic

from app.services.llm_provider import (
    LLMRequest,
    LLMResponse,
    ProviderError,
    ProviderRefusal,
    ProviderTimeout,
    TokenUsage,
)

PROVIDER_ID = "anthropic"


def _render_user_text(request: LLMRequest) -> str:
    """One user message from the assembled prompt's ordered segments — the
    Anthropic Messages API takes a single system string plus a message list, not
    typed segments; segment ordering (system/instruction/evidence/untrusted) is
    already fixed upstream by `prompt_builder.build_prompt` (C1)."""
    return "\n\n".join(segment.text for segment in request.segments)


@dataclass(frozen=True)
class AnthropicReviewerProvider:
    """`complete(LLMRequest) -> LLMResponse` against the real Anthropic API.

    `model` and `timeout_s` come from `config/reviewer-provider.yaml` via
    `providers.factory.build_provider` — never hardcoded here, so a config change
    (e.g. a different model snapshot) is visible in the reviewer manifest (L-8).
    The API key is read once at client-construction time from the environment
    variable named by config (`api_key_env`), never stored in the config file or
    on this dataclass."""

    model: str
    api_key: str
    timeout_s: float = 30.0

    def complete(self, request: LLMRequest) -> LLMResponse:
        client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout_s)
        started = time.monotonic()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=request.max_tokens,
                system=request.system,
                messages=[{"role": "user", "content": _render_user_text(request)}],
            )
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeout(str(exc)) from exc
        except anthropic.AnthropicError as exc:
            # Covers APIStatusError, APIConnectionError, RateLimitError, and every
            # other SDK-specific failure not already narrowed above — none of
            # these are a content-policy refusal, so ProviderError is correct.
            raise ProviderError(str(exc)) from exc
        latency_ms = (time.monotonic() - started) * 1000.0

        # `stop_reason == "refusal"` is the ONLY case mapped to ProviderRefusal —
        # per M-5, a response that merely fails to parse as valid JSON downstream
        # is a parse failure for R-7's repair retry, never miscoded as a refusal
        # (that would corrupt Arm4Coverage.provider_failure_rate, which P3.9 kept
        # separate from schema_validity).
        if getattr(response, "stop_reason", None) == "refusal":
            raise ProviderRefusal(
                f"Anthropic refused: {getattr(response, 'stop_details', None)}"
            )

        text = "".join(
            block.text for block in response.content if isinstance(block, anthropic.types.TextBlock)
        )
        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )
        return LLMResponse(
            text=text,
            model_snapshot=response.model,
            latency_ms=latency_ms,
            usage=usage,
        )


__all__ = ["AnthropicReviewerProvider", "PROVIDER_ID"]

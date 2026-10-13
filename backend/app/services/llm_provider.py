"""Provider-neutral interface for the LLM reviewer (Phase 3 plan P3.4, L-7).

`LLMProvider` is the seam every reviewer-calling code (P3.5) is written against.
The only implementation in this repo is `fake_reviewer.FakeReviewerProvider` — no
network-capable dependency is added anywhere in Phase 3, which `test_llm_provider.py`
enforces with an AST import scan over `services/**.py` (mirroring the sealed-store
capability-scan pattern used elsewhere in this test suite). A provider
implementation must never let a raw, provider-specific exception escape
`complete()`; every failure mode is translated to one of the three exceptions below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.services.prompt_builder import AssembledPrompt, Segment


class ProviderException(Exception):
    """Common base for the three normalized provider errors. Not itself one of
    the L-7 normalized error types — callers match on the concrete subclasses."""


class ProviderTimeout(ProviderException):
    """The provider did not respond within `LLMRequest.timeout_s`."""


class ProviderError(ProviderException):
    """A provider-side failure not covered by `ProviderTimeout`/`ProviderRefusal`."""


class ProviderRefusal(ProviderException):
    """The provider declined to answer (content-policy refusal or similar)."""


@dataclass(frozen=True)
class LLMRequest:
    system: str
    segments: tuple[Segment, ...]
    temperature: float
    max_tokens: int
    timeout_s: float

    @classmethod
    def from_assembled_prompt(
        cls,
        prompt: AssembledPrompt,
        *,
        temperature: float = 0.0,
        # P4.8 finding: a real reviewer prompt citing many timeline/contribution
        # evidence_keys legitimately runs past 1024 completion tokens once the
        # model actually answers (verified live: truncated mid-array at
        # completion_tokens==max_tokens==1024, on responses that were otherwise
        # schema-valid). Not a hard guarantee for every case, but bumped as the
        # fix for the truncation actually observed.
        max_tokens: int = 2048,
        timeout_s: float = 30.0,
    ) -> LLMRequest:
        return cls(
            system=prompt.system_text,
            segments=prompt.segments,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model_snapshot: str
    latency_ms: float
    usage: TokenUsage


@runtime_checkable
class LLMProvider(Protocol):
    """`complete(LLMRequest) -> LLMResponse` (L-7). Structural, not nominal: any
    object with a matching method satisfies this — `FakeReviewerProvider`
    included, without inheriting from this class."""

    def complete(self, request: LLMRequest) -> LLMResponse: ...

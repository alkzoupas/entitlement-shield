"""Deterministic fake `LLMProvider` implementation (Phase 3 plan P3.4, L-7).

`FakeReviewerProvider` is the only registered provider in this repo (L-7's "no
network-capable dependency added in Phase 3"). Its rule table keys **only** on the
`StructuredEvidenceSegment` text — the untrusted appeal block is located purely to
be skipped, never read for any decision-making purpose. That is what makes P3.8's
injection-regression assertion meaningful: two requests differing only in appeal
text hash to the same structured payload and therefore produce byte-identical
verdicts.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum

from app.models.enums import ProposedAction
from app.schemas.reviewer import ReviewerVerdict
from app.services.llm_provider import (
    LLMRequest,
    LLMResponse,
    ProviderError,
    ProviderRefusal,
    ProviderTimeout,
    TokenUsage,
)
from app.services.prompt_builder import StructuredEvidenceSegment, UntrustedBlock

MODEL_SNAPSHOT = "fake-reviewer-v1"

#: The reviewer stack's **declared** provider identity, hashed into
#: `config/reviewer-manifest.json` (L-8's "provider id"). Distinct from the
#: `provider` column `reviewer._run_sample` persists per row, which records the
#: live `type(provider).__name__` and so legitimately reads `_CountingProvider`,
#: `_FixedTextProvider`, etc. under test. This constant answers "which provider is
#: this build of the reviewer declared to run against"; that column answers "which
#: object actually served this call". Both are wanted; neither substitutes.
PROVIDER_ID = "fake_reviewer"

_EVIDENCE_KEY_LINE = re.compile(r"^  - (\w+):(.+)$")
_SCALAR_LINE = re.compile(r"^(band|allowance|excess): (.+)$")


class FakeMode(StrEnum):
    """Fault-injection modes for P3.5's orchestration tests (docs/phase-3-plan.md
    P3.4). `NORMAL` and `DISAGREE` both return a well-formed verdict; the other
    four simulate provider-level failure. `ERROR` is a generic simulated provider
    failure — distinct from `TIMEOUT`/`REFUSAL`/a malformed *request* (the latter
    is a caller/assembler bug and raises plain `ValueError`, never `ProviderError`;
    see `_structured_payload_text`)."""

    NORMAL = "normal"
    TIMEOUT = "timeout"
    MALFORMED = "malformed"
    REFUSAL = "refusal"
    DISAGREE = "disagree"
    ERROR = "error"


def _structured_payload_text(request: LLMRequest) -> str:
    """Locate the one segment the fake is allowed to read for decision-making.
    `UntrustedBlock` is matched only so it can be explicitly skipped — its text
    must never reach the rule table (see module docstring).

    A missing `StructuredEvidenceSegment` means the *request itself* is malformed
    (an assembler/caller bug, not a simulated provider failure) — this raises
    `ValueError`, never a `ProviderException` subtype. Only `FakeMode.ERROR`
    simulates a legitimate provider-side failure."""
    for segment in request.segments:
        if isinstance(segment, UntrustedBlock):
            continue
        if isinstance(segment, StructuredEvidenceSegment):
            return segment.text
    raise ValueError(
        f"{MODEL_SNAPSHOT}: request has no StructuredEvidenceSegment to review"
    )


def _stable_seed(structured_text: str) -> int:
    """A hash stable across processes and `PYTHONHASHSEED` values (unlike
    builtin `hash()`), derived from the structured payload — which itself embeds
    `decision_id` (see `prompt_builder._format_structured_evidence`) — so the seed
    is effectively keyed on case identity per docs/phase-3-plan.md P3.4."""
    digest = hashlib.sha256(structured_text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _extract_evidence_keys(structured_text: str) -> tuple[str, ...]:
    """Parse the `evidence_keys:` section out of the structured payload.

    Raises `ValueError` (a malformed *request* — never `ProviderError`, see
    `_structured_payload_text`) if the section header itself is missing: a
    request that reached the fake without one is something the assembler
    should never produce. A present-but-empty section legitimately yields no
    keys and is returned as such — callers must not paper over that with a
    hardcoded fallback citation (it would always resolve in the allowlist by
    construction and silently defeat citation-plumbing tests)."""
    keys: list[str] = []
    in_section = False
    for line in structured_text.splitlines():
        if line == "evidence_keys:":
            in_section = True
            continue
        if not in_section:
            continue
        match = _EVIDENCE_KEY_LINE.match(line)
        if match is None:
            break
        keys.append(f"{match.group(1)}:{match.group(2)}")
    if not in_section:
        raise ValueError(
            f"{MODEL_SNAPSHOT}: request's structured payload has no "
            "'evidence_keys:' section"
        )
    return tuple(keys)


def _extract_scalars(structured_text: str) -> dict[str, str]:
    scalars: dict[str, str] = {}
    for line in structured_text.splitlines():
        match = _SCALAR_LINE.match(line)
        if match is not None:
            scalars[match.group(1)] = match.group(2)
    return scalars


_ACTION_LADDER = (
    ProposedAction.ALLOW,
    ProposedAction.LIGHT_FRICTION,
    ProposedAction.HOLD_INCREMENTAL_CREDITS,
)

#: DISAGREE mode's variant->bucket-offset map (docs/decisions.md C-5, RC-2 finding).
#: `(seed + variant) % 3` over a 3-element ladder makes all three variants land on
#: distinct buckets (a 1-1-1 split) — never the 2-of-3 disagreement P3.5's
#: unanimity/escalation path needs to exercise. Variants 0 and 1 share an offset
#: (agree); variant 2's offset differs by 1 mod 3, which — since the ladder has
#: exactly 3 elements — always lands on a *different* bucket than variant 0/1's,
#: guaranteeing an exact 2-1 split every time, regardless of `seed`.
#:
#: This relies on per-instance state (`_call_counts`) keyed on the structured
#: payload: callers MUST reuse ONE `FakeReviewerProvider` instance across all 3
#: samples for one case — a fresh instance per call resets `_call_counts` to 0
#: and every sample would land on variant 0, defeating the disagreement.
_DISAGREE_VARIANT_OFFSETS = (0, 0, 1)


def _rule_table_verdict(
    seed: int, evidence_keys: tuple[str, ...], scalars: dict[str, str], *, variant: int
) -> ReviewerVerdict:
    bucket = (seed + variant) % len(_ACTION_LADDER)
    action = _ACTION_LADDER[bucket]
    if not evidence_keys:
        raise ValueError(
            f"{MODEL_SNAPSHOT}: no evidence keys available to cite — the fake "
            "never fabricates a fallback citation"
        )
    citation = evidence_keys[0]
    return ReviewerVerdict(
        evidence_supporting_action=(citation,),
        benign_explanations=(),
        missing_evidence=(),
        unverified_claims=(),
        rationale=(
            f"{MODEL_SNAPSHOT} rule-table bucket={bucket} variant={variant} "
            f"band={scalars.get('band')} excess={scalars.get('excess')}"
        ),
        recommended_action=action,
        confidence=0.9,
        escalate_to_human=False,
    )


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class FakeReviewerProvider:
    """Deterministic `LLMProvider`. `NORMAL` mode is a pure function of the
    structured payload — identical requests, even across fresh processes, produce
    byte-identical `.text` (C-5's "3 identical samples" case). `DISAGREE` mode
    cycles through distinct verdicts per case so P3.5's disagreement/escalation
    path can be exercised deterministically; that state is opt-in, never the
    default."""

    mode: FakeMode = FakeMode.NORMAL
    _call_counts: dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()

        if self.mode is FakeMode.TIMEOUT:
            raise ProviderTimeout(f"{MODEL_SNAPSHOT}: simulated timeout")
        if self.mode is FakeMode.REFUSAL:
            raise ProviderRefusal(f"{MODEL_SNAPSHOT}: simulated refusal")
        if self.mode is FakeMode.ERROR:
            raise ProviderError(f"{MODEL_SNAPSHOT}: simulated provider error")

        structured_text = _structured_payload_text(request)
        seed = _stable_seed(structured_text)

        if self.mode is FakeMode.MALFORMED:
            text = f"not-a-json-verdict seed={seed}"
        elif self.mode is FakeMode.DISAGREE:
            call_index = self._call_counts.get(structured_text, 0)
            self._call_counts[structured_text] = call_index + 1
            evidence_keys = _extract_evidence_keys(structured_text)
            scalars = _extract_scalars(structured_text)
            offset = _DISAGREE_VARIANT_OFFSETS[call_index % len(_DISAGREE_VARIANT_OFFSETS)]
            verdict = _rule_table_verdict(
                seed, evidence_keys, scalars, variant=offset
            )
            text = verdict.model_dump_json()
        else:
            evidence_keys = _extract_evidence_keys(structured_text)
            scalars = _extract_scalars(structured_text)
            verdict = _rule_table_verdict(seed, evidence_keys, scalars, variant=0)
            text = verdict.model_dump_json()

        latency_ms = (time.perf_counter() - started) * 1000
        usage = TokenUsage(
            prompt_tokens=_estimate_tokens(request.system) + _estimate_tokens(structured_text),
            completion_tokens=_estimate_tokens(text),
            total_tokens=_estimate_tokens(request.system)
            + _estimate_tokens(structured_text)
            + _estimate_tokens(text),
        )
        return LLMResponse(
            text=text,
            model_snapshot=MODEL_SNAPSHOT,
            latency_ms=latency_ms,
            usage=usage,
        )


__all__ = ["FakeMode", "FakeReviewerProvider", "MODEL_SNAPSHOT"]

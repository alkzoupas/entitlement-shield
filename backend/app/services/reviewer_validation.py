"""Parse, repair-retry, citation-filter, clamp, and derive the reviewer's effect
(Phase 3 plan P3.3; docs/decisions.md L-2/L-3/L-6 C-2/C-4, R-3/R-4/R-7).

This module is pure: it turns provider text into a `ReviewerVerdict` and, given the
case's evidence allowlist and the policy's proposed action, derives one of
`uphold`/`reverse`/`escalate`. It does not call a provider (P3.4) or persist
anything (P3.5) — callers own both, and the return types here are shaped so a
caller can persist every attempt without losing information.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from app.models.enums import ProposedAction
from app.schemas.reviewer import EvidenceKey, ReviewerVerdict
from app.services.policy_engine import ACTION_SEVERITY

MAX_RETRIES = 1  # R-7: exactly one repair retry before failing closed to escalation

DerivedEffect = Literal["uphold", "reverse", "escalate"]

ParseStatus = Literal["parsed", "needs_retry", "escalated"]


@dataclass(frozen=True)
class ParseAttempt:
    """One provider response and what came of parsing it (R-7: "all attempts
    logged" — the caller persists this row as-is, P3.5)."""

    raw_text: str
    retry_count: int
    verdict: ReviewerVerdict | None
    error: str | None


@dataclass(frozen=True)
class ParseOutcome:
    """Result of feeding one raw response through the parser.

    `status`:
    - "parsed": `verdict` is populated, `attempts` holds every attempt so far.
    - "needs_retry": parse failed and `retry_count < MAX_RETRIES` — the caller's
      job is to ask the provider again (with a repair prompt) and call
      `parse_attempt` again, passing this outcome's `attempts` forward.
    - "escalated": parse failed on the final allowed attempt (R-7 fail-closed).
    """

    attempts: tuple[ParseAttempt, ...]
    status: ParseStatus
    verdict: ReviewerVerdict | None


def _parse_raw_verdict(raw_text: str) -> ReviewerVerdict | str:
    """Try to parse `raw_text` into a `ReviewerVerdict`. Returns the verdict on
    success, or a human-readable error string on failure — never raises."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return f"invalid_json: {exc}"
    try:
        return ReviewerVerdict.model_validate(data)
    except ValidationError as exc:
        return f"schema_validation_failed: {exc}"


def parse_attempt(
    raw_text: str,
    retry_count: int,
    *,
    prior_attempts: tuple[ParseAttempt, ...] = (),
) -> ParseOutcome:
    """Parse one provider response at `retry_count` (0 for the first attempt, 1 for
    the single repair retry). Pass the previous outcome's `attempts` back in as
    `prior_attempts` on the retry call so nothing is lost (R-7)."""
    result = _parse_raw_verdict(raw_text)

    if isinstance(result, ReviewerVerdict):
        attempt = ParseAttempt(
            raw_text=raw_text, retry_count=retry_count, verdict=result, error=None
        )
        return ParseOutcome(
            attempts=(*prior_attempts, attempt), status="parsed", verdict=result
        )

    attempt = ParseAttempt(
        raw_text=raw_text, retry_count=retry_count, verdict=None, error=result
    )
    attempts = (*prior_attempts, attempt)
    status: ParseStatus = "escalated" if retry_count >= MAX_RETRIES else "needs_retry"
    return ParseOutcome(attempts=attempts, status=status, verdict=None)


def _citation_label(key: EvidenceKey) -> str:
    """Matches the exact `"{kind}:{key}"` format the prompt renders
    `evidence_keys` as (`prompt_builder._format_structured_evidence`) — this is
    the string a citing verdict must reproduce."""
    return f"{key.kind.value}:{key.key}"


def filter_citations(
    verdict: ReviewerVerdict, evidence_keys: Iterable[EvidenceKey]
) -> tuple[ReviewerVerdict, int]:
    """C-2: keep only citations resolving against the case's `EvidenceKey`
    allowlist; move everything else into `unverified_claims` (append, never
    discard). Returns the filtered verdict and the count of valid citations."""
    allowlist = {_citation_label(key) for key in evidence_keys}
    valid = tuple(c for c in verdict.evidence_supporting_action if c in allowlist)
    invalid = tuple(c for c in verdict.evidence_supporting_action if c not in allowlist)
    filtered = verdict.model_copy(
        update={
            "evidence_supporting_action": valid,
            "unverified_claims": (*verdict.unverified_claims, *invalid),
        }
    )
    return filtered, len(valid)


def clamp_action(
    model_action: ProposedAction, policy_proposed_action: ProposedAction
) -> ProposedAction:
    """C-4: `min(model_action, policy_action)` over the policy ladder's existing
    severity ordering (`policy_engine.ACTION_SEVERITY`) — independent of anything
    the model claimed."""
    if ACTION_SEVERITY[model_action] <= ACTION_SEVERITY[policy_proposed_action]:
        return model_action
    return policy_proposed_action


def derive_effect(
    verdict: ReviewerVerdict,
    policy_proposed_action: ProposedAction,
    valid_citation_count: int,
) -> DerivedEffect:
    """L-3's exhaustive three-effect table (no partial rung-stepping):

    - `escalate` if the model asked for a human (`escalate_to_human`), OR there
      are zero valid citations (C-2 anti-fabrication: a verdict with no citation
      that resolves against the allowlist is structurally incapable of reversing,
      regardless of what it recommends).
    - `reverse` if the C-4-clamped action is strictly less severe than the
      policy's proposed action AND the clamped action is at or below
      `light_friction` (R-6 auto-reversal ceiling) AND there is at least one valid
      citation.
    - `uphold` otherwise — including every case where the (clamped) recommendation
      is the same as or more severe than the proposal, which can never reverse
      (R-2: may not raise severity, so it also may never seem to justify one).

    Unanimity across samples (R-5) is P3.5's job across 3 verdicts — this function
    only ever sees one.
    """
    if verdict.escalate_to_human or valid_citation_count == 0:
        return "escalate"

    clamped = clamp_action(verdict.recommended_action, policy_proposed_action)
    if (
        ACTION_SEVERITY[clamped] < ACTION_SEVERITY[policy_proposed_action]
        and ACTION_SEVERITY[clamped] <= ACTION_SEVERITY[ProposedAction.LIGHT_FRICTION]
    ):
        return "reverse"

    return "uphold"


@dataclass(frozen=True)
class EvaluatedVerdict:
    """The full C-2/C-4/L-3 pipeline's output for one parsed verdict."""

    verdict: ReviewerVerdict  # citation-filtered (evidence_supporting_action pruned)
    valid_citation_count: int
    clamped_action: ProposedAction
    effect: DerivedEffect


def evaluate_verdict(
    verdict: ReviewerVerdict,
    evidence_keys: Iterable[EvidenceKey],
    policy_proposed_action: ProposedAction,
) -> EvaluatedVerdict:
    """Run C-2 citation filtering, C-4 severity clamping, and L-3 derivation over
    one parsed verdict, in that order."""
    filtered, valid_count = filter_citations(verdict, evidence_keys)
    clamped = clamp_action(filtered.recommended_action, policy_proposed_action)
    effect = derive_effect(filtered, policy_proposed_action, valid_count)
    return EvaluatedVerdict(
        verdict=filtered,
        valid_citation_count=valid_count,
        clamped_action=clamped,
        effect=effect,
    )

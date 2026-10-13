"""Reviewer orchestration (Phase 3 plan P3.5, docs/decisions.md §12 L-3/L-4/L-9,
C-5). `review_decision` is the seam every caller (the future `/reviewer/run`
endpoint, P3.7) drives: build the case once, call the provider 3 times, run each
response through `reviewer_validation`'s parse/repair/C-2/C-4 pipeline, and either
auto-apply a unanimous reversal (via `policy_engine.record_reversal`, mirroring the
analyst path) or leave the decision untouched pending human escalation (P3.6).

Nothing here re-derives evidence or re-clusters — `build_review_case`/
`decision_evidence` are the only reads, matching L-1.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.decisions import transition_decision
from app.models.enums import ActorType, DecisionStatus, ProposedAction
from app.models.inference import AuditEvent, Decision, LlmReview
from app.schemas.audit import LlmReviewRecorded
from app.schemas.reviewer import ReviewCase
from app.services.audit import append_audit
from app.services.escalation import escalate_decision, has_open_escalation
from app.services.evidence import decision_evidence
from app.services.llm_provider import (
    LLMProvider,
    LLMRequest,
    ProviderError,
    ProviderRefusal,
    ProviderTimeout,
)
from app.services.policy_engine import (
    ACTION_SEVERITY,
    evidence_signature,
    is_suppressed,
    record_reversal,
)
from app.services.prompt_builder import PromptVariant, build_prompt
from app.services.review_case import build_review_case
from app.services.reviewer_validation import DerivedEffect, evaluate_verdict, parse_attempt

SAMPLE_COUNT = 3  # C-5

#: Sampling temperature for every reviewer call. Named rather than passed as a bare
#: `0.0` literal so `reviewer_manifest` can read the live value (L-8 hashes the
#: temperature) instead of restating it — a restated constant can drift silently.
REVIEWER_TEMPERATURE = 0.0

#: `arm` → `(review-case view, prompt variant, persisted prompt_version)`.
#:
#: The three were previously hardcoded at the `build_review_case`/`build_prompt`
#: call sites. Arm 4b needs them to covary — the masked view, the masked prompt
#: template, and the recorded version string must never disagree — so they live in
#: one table keyed on the arm rather than three independent literals.
#:
#: `"reviewer"` (P3.7's `/reviewer/run` default, and `console.py`/`json_api.py`'s
#: form/JSON default) maps to exactly the previous behavior, so the live API
#: surface is unchanged.
_ARM_CONFIG: dict[str, tuple[str, PromptVariant, str]] = {
    "reviewer": ("full", "reviewer_v1", "reviewer_v1"),
    "arm4_reviewer": ("full", "reviewer_v1", "reviewer_v1"),
    "arm4b_timeline_only": (
        "timeline_only",
        "reviewer_timeline_only_v1",
        "reviewer_timeline_only_v1",
    ),
}

#: `arm` is caller-supplied free text on both the JSON API (`ReviewerRunRequest.arm`)
#: and the console form, and has always been recorded verbatim on the `LlmReview`
#: row without being validated. Unknown arms therefore keep the default full-view
#: configuration rather than raising — introducing a `KeyError`/500 here would be a
#: live behavior regression on an endpoint P3.9 is not meant to touch.
_DEFAULT_ARM_CONFIG = _ARM_CONFIG["reviewer"]


def arm_config(arm: str) -> tuple[str, PromptVariant, str]:
    """The `(view, variant, prompt_version)` triple for `arm`, defaulting to the
    full-view reviewer configuration for any unrecognized arm."""
    return _ARM_CONFIG.get(arm, _DEFAULT_ARM_CONFIG)


def reviewer_manifest_hash() -> str:
    """The L-8 reviewer-manifest hash stamped onto every row and audit event.

    Imported lazily because the dependency is genuinely circular and cannot be
    broken by moving code: `reviewer_manifest` fingerprints *this* module's source
    and reads `SAMPLE_COUNT`/`REVIEWER_TEMPERATURE`/`_ARM_CONFIG` off it, while
    this module needs the hash that computation produces. A module-scope import
    would run `reviewer_manifest`'s import-time `MANIFEST_HASH = content_hash()`
    against a half-initialized `reviewer` module. (Same deferred-import pattern
    `db/evaluation_access.py` uses against `db/world_provenance`.)

    The plain `MANIFEST_HASH` is used, never `verified_manifest_hash()`: stamping
    must record the hash that actually applied, and an uncommitted working tree
    should produce an honest row rather than a refusal on the live `/reviewer/run`
    surface. The fail-closed check belongs to `scripts/arm4_run.py`."""
    from app.services import reviewer_manifest

    return reviewer_manifest.MANIFEST_HASH


@dataclass(frozen=True)
class SampleOutcome:
    """One sample's final outcome (after any repair retry)."""

    sample_index: int
    effect: DerivedEffect
    review_ids: tuple[str, ...]  # every LlmReview row persisted for this sample


@dataclass(frozen=True)
class ReviewOutcome:
    """What `review_decision` did. `overall_effect` and `samples` are empty/`None`
    when the call was gated out or replayed from an already-recorded run."""

    decision_id: str
    gated: bool
    gate_reason: str | None
    replayed: bool
    samples: tuple[SampleOutcome, ...]
    overall_effect: Literal["uphold", "reverse", "escalate"] | None
    applied: bool


def _gating_reason(session: Session, decision: Decision, as_of: datetime) -> str | None:
    """L-9's invocation gate. Returns a reason to skip, or `None` to proceed.

    Deliberately its own function (rather than inlined in `review_decision`), which
    let P3.6 extend it with an open-`EscalationRecord` check (P1-3) without
    touching the caller."""
    if decision.proposed_action is ProposedAction.ALLOW:
        return "allow"
    if decision.status in (DecisionStatus.REVERSED, DecisionStatus.SUPERSEDED):
        return f"terminal_status:{decision.status.value}"
    if decision.status is DecisionStatus.UNDER_REVIEW:
        if has_open_escalation(session, decision):
            return "escalated_under_review"
        return "already_under_review"

    payload = decision_evidence(session, decision.decision_id)
    if payload is not None:
        signature = evidence_signature(
            payload["member_account_ids"], payload["edges"], payload["excess"]
        )
        if is_suppressed(session, signature, as_of):
            return "suppressed"
    return None


def _prompt_hash(request: LLMRequest) -> str:
    parts = [request.system, *[getattr(s, "text", "") for s in request.segments]]
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()
    return digest


def _existing_reviews(
    session: Session, decision_id: str, idempotency_key: str
) -> list[LlmReview]:
    return list(
        session.scalars(
            select(LlmReview)
            .where(
                LlmReview.decision_id == decision_id,
                LlmReview.idempotency_key == idempotency_key,
            )
            .order_by(LlmReview.sample_index, LlmReview.retry_count)
        ).all()
    )


def latest_reviews(session: Session, decision_id: str) -> list[LlmReview]:
    """The one FINAL `LlmReview` row per sample (post-repair-retry) from the most
    recently invoked reviewer run on `decision_id` (P3.7's console/API read path).

    `LlmReview` carries no timestamp of its own, so "most recent" is resolved via
    the `llm_review_recorded` audit trail rather than an ordering column. Empty if
    the decision was never reviewed."""
    latest_event = session.scalars(
        select(AuditEvent)
        .where(
            AuditEvent.event_name == "llm_review_recorded",
            AuditEvent.entity_id == decision_id,
        )
        .order_by(AuditEvent.timestamp.desc(), AuditEvent.audit_id.desc())
        .limit(1)
    ).first()
    if latest_event is None:
        return []
    latest_row = session.get(LlmReview, latest_event.payload_json["review_id"])
    if latest_row is None:
        return []

    rows = _existing_reviews(session, decision_id, latest_row.idempotency_key)
    by_sample: dict[int, list[LlmReview]] = {}
    for row in rows:
        by_sample.setdefault(row.sample_index, []).append(row)
    return [
        next((r for r in by_sample[i] if r.derived_verdict is not None), by_sample[i][-1])
        for i in sorted(by_sample)
    ]


def latest_reviews_for_arm(session: Session, decision_id: str, arm: str) -> list[LlmReview]:
    """`latest_reviews`, but confined to one arm (P3.9 evaluation read path).

    `latest_reviews` resolves "the most recent run" from the newest
    `llm_review_recorded` audit event with no arm filter at all. That is correct
    for P3.7's console, where only `arm="reviewer"` rows exist — but once arm 4 and
    arm 4b rows coexist on the same decision it would silently return whichever arm
    happened to run last, so `arm4_reviewer` and `arm4b_timeline_only` would score
    each other's verdicts. Filtering the audit scan by arm keeps the two arms'
    reads disjoint even when both have run against the same decision.

    Empty if this decision was never reviewed under `arm`."""
    latest_event = session.scalars(
        select(AuditEvent)
        .where(
            AuditEvent.event_name == "llm_review_recorded",
            AuditEvent.entity_id == decision_id,
            # `arm` is a first-class field on `LlmReviewRecorded`, so the arm can
            # be resolved from the audit trail without loading every candidate row.
            AuditEvent.payload_json["arm"].as_string() == arm,
        )
        .order_by(AuditEvent.timestamp.desc(), AuditEvent.audit_id.desc())
        .limit(1)
    ).first()
    if latest_event is None:
        return []
    latest_row = session.get(LlmReview, latest_event.payload_json["review_id"])
    if latest_row is None:
        return []

    rows = [
        row
        for row in _existing_reviews(session, decision_id, latest_row.idempotency_key)
        # An idempotency key is per (decision, run), and `review_decision` records
        # one arm per run — but the key is caller-supplied on both the API and the
        # console, so a caller reusing one key across arms must not be able to mix
        # the two arms' rows into one sample set.
        if row.arm == arm
    ]
    by_sample: dict[int, list[LlmReview]] = {}
    for row in rows:
        by_sample.setdefault(row.sample_index, []).append(row)
    return [
        next((r for r in by_sample[i] if r.derived_verdict is not None), by_sample[i][-1])
        for i in sorted(by_sample)
    ]


def _replay_outcome(decision: Decision, rows: list[LlmReview]) -> ReviewOutcome:
    """Reconstruct the prior run's per-sample effects from persisted rows without
    calling the provider again (idempotency — a re-run must not double-charge it)."""
    by_sample: dict[int, list[LlmReview]] = {}
    for row in rows:
        by_sample.setdefault(row.sample_index, []).append(row)

    samples = []
    for index in sorted(by_sample):
        sample_rows = by_sample[index]
        final = next((r for r in sample_rows if r.derived_verdict is not None), sample_rows[-1])
        effect: DerivedEffect = final.derived_verdict or "escalate"  # type: ignore[assignment]
        samples.append(
            SampleOutcome(
                sample_index=index,
                effect=effect,
                review_ids=tuple(r.review_id for r in sample_rows),
            )
        )

    overall = combine_effects(tuple(s.effect for s in samples)) if samples else None
    return ReviewOutcome(
        decision_id=decision.decision_id,
        gated=False,
        gate_reason=None,
        replayed=True,
        samples=tuple(samples),
        overall_effect=overall,
        # A replay never re-applies a reversal: if it happened, the decision's
        # persisted status already reflects it.
        applied=False,
    )


def combine_effects(
    effects: tuple[DerivedEffect, ...],
) -> Literal["uphold", "reverse", "escalate"]:
    """C-5: strict unanimity. Any disagreement escalates, regardless of what the
    majority is — this is `docs/decisions.md` L-6's C-5 line ("any disagreement
    escalates"), not a majority vote."""
    if len(set(effects)) == 1:
        return effects[0]
    return "escalate"


def _other_actioned_cluster_members(session: Session, decision: Decision) -> bool:
    """RC-1 P1-1 / L-3: does this decision's evidence-signature cluster have any
    OTHER currently-actioned (non-terminal) decision? Auto-apply is blocked
    whenever true, since C-1 suppression is cluster-wide and an LLM auto-apply
    must not silently de-flag co-actioned accounts it never separately evaluated.

    Scoping: candidates are decisions sharing `cluster_snapshot_id` (the same
    clustering run identified this account's cluster) — a cluster snapshot can
    host multiple cluster-cycle units, so membership in the SAME evidence
    signature (member set + edges + canonical excess, `policy_engine.
    evidence_signature`) is confirmed exactly rather than assumed from
    `cluster_snapshot_id` alone. This is new logic — P3.1-P3.4 never needed to
    ask "who else is in this cluster's evidence group."

    BOTH signatures are derived from the raw persisted `decision_evidence`
    payload — never from the `ReviewCase` projection. `ReviewCase.edges` narrows
    each edge to 3 fields, which `evidence_signature` reads as the legacy-v1
    form; signing this side from the projection while signing candidates from
    the canonical 6-field payload made the two signatures unequal for every
    cluster with non-empty edges, silently disabling this guard."""
    if decision.cluster_snapshot_id is None:
        return False

    this_payload = decision_evidence(session, decision.decision_id)
    if this_payload is None:
        # Fail closed: without evidence we cannot prove this decision stands alone.
        return True
    this_signature = evidence_signature(
        this_payload["member_account_ids"], this_payload["edges"], this_payload["excess"]
    )
    candidates = session.scalars(
        select(Decision).where(
            Decision.cluster_snapshot_id == decision.cluster_snapshot_id,
            Decision.decision_id != decision.decision_id,
            Decision.status.in_(
                (DecisionStatus.PROPOSED, DecisionStatus.UNDER_REVIEW, DecisionStatus.APPLIED)
            ),
        )
    ).all()
    for other in candidates:
        payload = decision_evidence(session, other.decision_id)
        if payload is None:
            continue
        other_signature = evidence_signature(
            payload["member_account_ids"], payload["edges"], payload["excess"]
        )
        if other_signature == this_signature:
            return True
    return False


@dataclass(frozen=True)
class _Sampled:
    """Result of gating + sampling one decision, with no commit performed yet.
    `outcome` is set (final) when the call was gated out or replayed — nothing
    left to decide. Otherwise `samples`/`overall` carry the freshly-drawn
    verdicts and the caller (`review_decision` or `review_cluster_batch`) must
    still commit via `_commit_outcome`."""

    outcome: ReviewOutcome | None
    samples: tuple[SampleOutcome, ...] = ()
    overall: Literal["uphold", "reverse", "escalate"] | None = None


def _sample_decision(
    session: Session,
    decision: Decision,
    *,
    provider: LLMProvider,
    run_id: str,
    idempotency_key: str,
    as_of: datetime,
    arm: str,
) -> _Sampled:
    """Gate, replay-check, and draw C-5's 3 samples for one decision — no state
    transition, no `record_reversal`, no `escalate_decision`. Split out of
    `review_decision` so `review_cluster_batch` (M-11) can sample every member of
    a cluster before any of them commits."""
    gate_reason = _gating_reason(session, decision, as_of)
    if gate_reason is not None:
        return _Sampled(
            outcome=ReviewOutcome(
                decision_id=decision.decision_id,
                gated=True,
                gate_reason=gate_reason,
                replayed=False,
                samples=(),
                overall_effect=None,
                applied=False,
            )
        )

    existing = _existing_reviews(session, decision.decision_id, idempotency_key)
    if existing:
        return _Sampled(outcome=_replay_outcome(decision, existing))

    view, variant, prompt_version = arm_config(arm)
    # Resolved once per run, not per row: every attempt of every sample in one
    # invocation must record the same reviewer version (L-8).
    manifest_hash = reviewer_manifest_hash()
    case = build_review_case(session, decision, view=view, as_of=as_of)
    prompt = build_prompt(case, variant=variant)
    request = LLMRequest.from_assembled_prompt(prompt, temperature=REVIEWER_TEMPERATURE)
    prompt_hash = _prompt_hash(request)

    samples = tuple(
        _run_sample(
            session,
            decision=decision,
            case=case,
            provider=provider,
            request=request,
            idempotency_key=idempotency_key,
            sample_index=i,
            arm=arm,
            prompt_hash=prompt_hash,
            prompt_version=prompt_version,
            manifest_hash=manifest_hash,
            run_id=run_id,
        )
        for i in range(SAMPLE_COUNT)
    )
    session.flush()
    overall = combine_effects(tuple(s.effect for s in samples))
    return _Sampled(outcome=None, samples=samples, overall=overall)


def _commit_outcome(
    session: Session,
    decision: Decision,
    samples: tuple[SampleOutcome, ...],
    overall: Literal["uphold", "reverse", "escalate"],
    *,
    as_of: datetime,
    siblings_confirmed: bool,
) -> ReviewOutcome:
    """Commit the state transition for an already-sampled (non-gated,
    non-replayed) decision. `siblings_confirmed` is M-11's result: True means
    either there is no other live decision sharing this decision's evidence
    signature, or every one that exists has *also* independently reached a
    unanimous reverse (computed by the caller — `review_decision`'s
    single-decision check via `_other_actioned_cluster_members`, or
    `review_cluster_batch`'s group-wide check)."""
    applied = False
    if overall == "reverse":
        # R-6 ceiling (L-3's 4 auto-apply conditions): the decision's ORIGINAL
        # proposed action — not the model's clamped recommendation, which
        # `derive_effect` already bounded at light_friction — must itself be at
        # or below light_friction. A `hold_incremental_credits` (or worse)
        # decision never auto-reverses, no matter how unanimous the verdict.
        proposed_severity = ACTION_SEVERITY[decision.proposed_action]
        light_friction_severity = ACTION_SEVERITY[ProposedAction.LIGHT_FRICTION]
        if proposed_severity > light_friction_severity:
            overall = "escalate"
        elif not siblings_confirmed:
            overall = "escalate"
        else:
            transition_decision(
                session,
                decision,
                DecisionStatus.UNDER_REVIEW,
                actor=ActorType.LLM,
                reason="reviewer_unanimous_reverse",
            )
            payload = decision_evidence(session, decision.decision_id)
            assert payload is not None
            record_reversal(
                session,
                decision,
                member_account_ids=payload["member_account_ids"],
                edges=payload["edges"],
                excess=payload["excess"],
                actor=ActorType.LLM,
                reason="reviewer auto-reversal (unanimous, confirmed siblings)",
                as_of=as_of,
            )
            applied = True

    if overall == "escalate":
        # L-5: an escalate verdict is a real state transition, not just a label on
        # the LlmReview rows — `_gating_reason` above already confined entry to
        # `proposed`/`applied` decisions, so this is always a legal transition.
        escalate_decision(
            session,
            decision,
            source="llm_review",
            as_of=as_of,
            actor=ActorType.LLM,
            reason="reviewer_escalate",
        )

    return ReviewOutcome(
        decision_id=decision.decision_id,
        gated=False,
        gate_reason=None,
        replayed=False,
        samples=samples,
        overall_effect=overall,
        applied=applied,
    )


def review_cluster_batch(
    session: Session,
    decisions: Sequence[Decision],
    *,
    provider: LLMProvider,
    run_id: str,
    idempotency_key_fn: Callable[[Decision], str],
    as_of: datetime,
    arm: str = "reviewer",
) -> list[ReviewOutcome]:
    """M-11 (`docs/decisions.md §13`): narrows RC-1 P1-1's auto-apply guard from
    "any other live actioned cluster member blocks auto-apply" to "auto-apply is
    allowed when every other live decision sharing the same evidence signature
    has ALSO independently reached a unanimous reverse this run." Reviewing a
    whole batch together — every member sampled BEFORE any of them commits — is
    what makes that symmetric and order-independent: whether decision A's
    sibling B also reached unanimous reverse is answered from this batch's own
    results, not from whichever decision happened to be reviewed first.

    `review_decision`'s single-decision behavior is UNCHANGED — it still applies
    the original, conservative "any live sibling blocks" rule via
    `_other_actioned_cluster_members`. This function is a separate, explicit
    opt-in for callers that genuinely intend to review an entire cluster's live
    decisions together in one invocation (e.g. `scripts/arm4_run.py`).

    A group's members auto-apply together only if EVERY other live decision
    sharing their evidence signature is *inside* `decisions` and *also*
    unanimous-reverse (severity-eligible); any outside or non-conforming sibling
    still blocks the whole group, exactly as the original guard would for each
    member individually. Nothing is de-flagged unevaluated — no decision commits
    without having been sampled and independently derived a unanimous reverse
    itself."""
    sampled: dict[str, _Sampled] = {
        decision.decision_id: _sample_decision(
            session,
            decision,
            provider=provider,
            run_id=run_id,
            idempotency_key=idempotency_key_fn(decision),
            as_of=as_of,
            arm=arm,
        )
        for decision in decisions
    }

    # Group the reverse-and-severity-eligible decisions by evidence signature —
    # only these are governed by M-11; everything else (gated, replayed, uphold,
    # non-unanimous escalate, or a reverse that already fails R-6) commits
    # independently below, exactly as `review_decision` would.
    groups: dict[tuple[str, str], list[Decision]] = {}
    for decision in decisions:
        sample = sampled[decision.decision_id]
        if sample.outcome is not None or sample.overall != "reverse":
            continue
        if decision.cluster_snapshot_id is None:
            continue
        severity = ACTION_SEVERITY[decision.proposed_action]
        if severity > ACTION_SEVERITY[ProposedAction.LIGHT_FRICTION]:
            continue  # R-6 ceiling — commits as an individual escalate below
        payload = decision_evidence(session, decision.decision_id)
        if payload is None:
            continue
        signature = evidence_signature(
            payload["member_account_ids"], payload["edges"], payload["excess"]
        )
        groups.setdefault((decision.cluster_snapshot_id, signature), []).append(decision)

    confirmed_ids: set[str] = set()
    for (snapshot_id, signature), members in groups.items():
        batch_ids = {d.decision_id for d in members}
        outside = session.scalars(
            select(Decision).where(
                Decision.cluster_snapshot_id == snapshot_id,
                Decision.decision_id.not_in(sorted(batch_ids)),
                Decision.status.in_(
                    (DecisionStatus.PROPOSED, DecisionStatus.UNDER_REVIEW, DecisionStatus.APPLIED)
                ),
            )
        ).all()
        blocked = False
        for other in outside:
            other_payload = decision_evidence(session, other.decision_id)
            if other_payload is None:
                blocked = True
                break
            other_signature = evidence_signature(
                other_payload["member_account_ids"],
                other_payload["edges"],
                other_payload["excess"],
            )
            if other_signature == signature:
                blocked = True
                break
        if not blocked:
            confirmed_ids.update(batch_ids)

    outcomes: list[ReviewOutcome] = []
    for decision in decisions:
        sample = sampled[decision.decision_id]
        if sample.outcome is not None:
            outcomes.append(sample.outcome)
            continue
        assert sample.overall is not None
        outcomes.append(
            _commit_outcome(
                session,
                decision,
                sample.samples,
                sample.overall,
                as_of=as_of,
                siblings_confirmed=decision.decision_id in confirmed_ids,
            )
        )
    return outcomes


def _persist_sample_row(
    session: Session,
    *,
    decision: Decision,
    idempotency_key: str,
    sample_index: int,
    arm: str,
    provider_name: str,
    prompt_hash: str,
    prompt_version: str,
    reviewer_manifest_hash: str,
    temperature: float,
    model_snapshot: str,
    raw_response: str | None,
    structured_output_json: dict | None,
    validation_errors: dict | None,
    retry_count: int,
    recommendation: str | None,
    derived_verdict: str | None,
    confidence: float | None,
    latency_ms: float | None,
    citations_json: list | None,
    run_id: str,
) -> LlmReview:
    row = LlmReview(
        review_id=uuid.uuid4().hex,
        decision_id=decision.decision_id,
        idempotency_key=idempotency_key,
        sample_index=sample_index,
        arm=arm,
        provider=provider_name,
        prompt_hash=prompt_hash,
        model_snapshot=model_snapshot,
        prompt_version=prompt_version,
        reviewer_manifest_hash=reviewer_manifest_hash,
        temperature=temperature,
        raw_response=raw_response,
        structured_output_json=structured_output_json,
        validation_errors=validation_errors,
        retry_count=retry_count,
        recommendation=recommendation,
        derived_verdict=derived_verdict,
        confidence=confidence,
        latency_ms=latency_ms,
        citations_json=citations_json,
    )
    session.add(row)
    session.flush()
    append_audit(
        session,
        LlmReviewRecorded(
            entity_type="decision",
            entity_id=decision.decision_id,
            actor_type=ActorType.LLM,
            decision_id=decision.decision_id,
            review_id=row.review_id,
            run_id=run_id,
            sample_index=sample_index,
            arm=arm,
            provider=provider_name,
            retry_count=retry_count,
            derived_verdict=derived_verdict,
            reviewer_manifest_hash=reviewer_manifest_hash,
        ),
    )
    return row


def _run_sample(
    session: Session,
    *,
    decision: Decision,
    case: ReviewCase,
    provider: LLMProvider,
    request: LLMRequest,
    idempotency_key: str,
    sample_index: int,
    arm: str,
    prompt_hash: str,
    prompt_version: str,
    manifest_hash: str,
    run_id: str,
) -> SampleOutcome:
    """Run one of the 3 C-5 samples end to end: call the provider, parse with
    R-7's single repair retry, evaluate (C-2/C-4), and persist every attempt.

    Fail-closed: a `ProviderTimeout`/`ProviderError`/`ProviderRefusal` — on the
    initial call or the repair retry — ends the sample as `escalate` and is
    persisted as a failed attempt; it never raises past this function, since one
    bad sample must not abort the other two or the caller's transaction."""
    provider_name = type(provider).__name__
    review_ids: list[str] = []
    retry_count = 0
    prior_attempts: tuple = ()

    while True:
        try:
            response = provider.complete(request)
        except (ProviderTimeout, ProviderError, ProviderRefusal) as exc:
            row = _persist_sample_row(
                session,
                decision=decision,
                idempotency_key=idempotency_key,
                sample_index=sample_index,
                arm=arm,
                provider_name=provider_name,
                prompt_hash=prompt_hash,
                prompt_version=prompt_version,
                reviewer_manifest_hash=manifest_hash,
                temperature=request.temperature,
                model_snapshot="",
                raw_response=None,
                structured_output_json=None,
                validation_errors={"provider_error": f"{type(exc).__name__}: {exc}"},
                retry_count=retry_count,
                recommendation=None,
                derived_verdict="escalate",
                confidence=None,
                latency_ms=None,
                citations_json=None,
                run_id=run_id,
            )
            review_ids.append(row.review_id)
            return SampleOutcome(
                sample_index=sample_index, effect="escalate", review_ids=tuple(review_ids)
            )

        outcome = parse_attempt(response.text, retry_count, prior_attempts=prior_attempts)

        if outcome.status == "parsed":
            assert outcome.verdict is not None
            evaluated = evaluate_verdict(
                outcome.verdict, case.evidence_keys, decision.proposed_action
            )
            row = _persist_sample_row(
                session,
                decision=decision,
                idempotency_key=idempotency_key,
                sample_index=sample_index,
                arm=arm,
                provider_name=provider_name,
                prompt_hash=prompt_hash,
                prompt_version=prompt_version,
                reviewer_manifest_hash=manifest_hash,
                temperature=request.temperature,
                model_snapshot=response.model_snapshot,
                raw_response=response.text,
                structured_output_json=evaluated.verdict.model_dump(mode="json"),
                validation_errors=None,
                retry_count=retry_count,
                recommendation=evaluated.clamped_action.value,
                derived_verdict=evaluated.effect,
                confidence=evaluated.verdict.confidence,
                latency_ms=response.latency_ms,
                citations_json=list(evaluated.verdict.evidence_supporting_action),
                run_id=run_id,
            )
            review_ids.append(row.review_id)
            return SampleOutcome(
                sample_index=sample_index, effect=evaluated.effect, review_ids=tuple(review_ids)
            )

        latest_error = outcome.attempts[-1].error
        is_final = outcome.status == "escalated"
        row = _persist_sample_row(
            session,
            decision=decision,
            idempotency_key=idempotency_key,
            sample_index=sample_index,
            arm=arm,
            provider_name=provider_name,
            prompt_hash=prompt_hash,
            # Was `case.view` — a copy/paste bug that stamped the parse-failure
            # rows of every sample with the *view* name instead of the prompt
            # version, so a repair-retried or terminally-unparseable attempt
            # recorded `"full"` where its siblings recorded `"reviewer_v1"`.
            # Arm 4b makes it worse rather than visible: `case.view` becomes
            # `"timeline_only"`, which looks plausible next to
            # `"reviewer_timeline_only_v1"` and would quietly corrupt the L-8
            # version-observability trail the reviewer manifest exists to
            # guarantee. Fixed here rather than deferred — see HANDOFF P3.9.
            prompt_version=prompt_version,
            reviewer_manifest_hash=manifest_hash,
            temperature=request.temperature,
            model_snapshot=response.model_snapshot,
            raw_response=response.text,
            structured_output_json=None,
            validation_errors={"parse_error": latest_error},
            retry_count=retry_count,
            recommendation=None,
            derived_verdict="escalate" if is_final else None,
            confidence=None,
            latency_ms=response.latency_ms,
            citations_json=None,
            run_id=run_id,
        )
        review_ids.append(row.review_id)
        if is_final:
            return SampleOutcome(
                sample_index=sample_index, effect="escalate", review_ids=tuple(review_ids)
            )

        prior_attempts = outcome.attempts
        retry_count += 1


def review_decision(
    session: Session,
    decision: Decision,
    *,
    provider: LLMProvider,
    run_id: str,
    idempotency_key: str,
    as_of: datetime,
    arm: str = "reviewer",
) -> ReviewOutcome:
    """Review one decision (Phase 3 plan P3.5/P3.7). Gates per L-9, replays
    idempotently on a re-used `idempotency_key`, otherwise draws 3 samples from
    `provider` against ONE assembled prompt, evaluates each, and either
    auto-applies a unanimous valid reversal (L-3/L-4) or drives L-5's real
    escalation (`escalation.escalate_decision`) — an uphold outcome records
    reviews but makes no state change. `_gating_reason` already restricts entry
    to `proposed`/`applied` decisions, so an escalate outcome here always finds
    the decision in a status `escalate_decision` can legally transition.

    Single-decision auto-apply gating is UNCHANGED from before M-11: any other
    live decision sharing this decision's evidence signature blocks auto-apply
    unconditionally (`_other_actioned_cluster_members`), because a standalone
    call has no way to know whether that sibling will ever be reviewed. Callers
    that want M-11's narrower, symmetric multi-member auto-apply (all siblings
    reviewed together, all independently unanimous-reverse) must use
    `review_cluster_batch` instead."""
    sampled = _sample_decision(
        session,
        decision,
        provider=provider,
        run_id=run_id,
        idempotency_key=idempotency_key,
        as_of=as_of,
        arm=arm,
    )
    if sampled.outcome is not None:
        return sampled.outcome
    assert sampled.overall is not None
    siblings_confirmed = not _other_actioned_cluster_members(session, decision)
    return _commit_outcome(
        session,
        decision,
        sampled.samples,
        sampled.overall,
        as_of=as_of,
        siblings_confirmed=siblings_confirmed,
    )


__all__ = ["ReviewOutcome", "SampleOutcome", "review_decision", "review_cluster_batch"]

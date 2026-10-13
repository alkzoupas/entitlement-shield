"""JSON API endpoints (original-plan §17): the batch detector trigger, the claim-time
credit path, and the audit query. HTML console pages live in `console.py`.

These are inference-plane only. The detector run is idempotent (a re-run at the same
`as_of` re-affirms rather than duplicates decisions — architecture §7); the claim path
holds a hard latency budget (architecture §2)."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_reviewer_provider, get_session
from app.api.presenters import audit_view
from app.core.clock import now_utc
from app.models.inference import AuditEvent, Decision, LlmReview
from app.services.claim import CLAIM_BUDGET_MS, ClaimKeyConflict, MissingCycle, UnknownAccount
from app.services.claim_deadline import bounded_claim
from app.services.detector import run_detector
from app.services.generator import WORLD_END
from app.services.llm_provider import LLMProvider
from app.services.reviewer import review_decision

router = APIRouter(prefix="/api", tags=["api"])


class DetectorRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    as_of: datetime | None = None


class DetectorRunResponse(BaseModel):
    run_id: str
    as_of: str
    clusters: int
    cluster_cycles: int
    decisions_created: int
    decisions_skipped: int
    decisions_withdrawn: int
    suppressed_units: int


@router.post("/detector/run", response_model=DetectorRunResponse)
def detector_run(
    body: DetectorRunRequest, session: Session = Depends(get_session)
) -> DetectorRunResponse:
    """Run the batch detector over the inference store. Idempotent at the decision level:
    a second run at the same `as_of` re-affirms the same idempotency keys (architecture
    §7) rather than writing duplicate decisions."""
    as_of = body.as_of or WORLD_END
    run_id = uuid.uuid4().hex
    try:
        result = run_detector(session, as_of=as_of, run_id=run_id)
    except ValueError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    session.commit()
    return DetectorRunResponse(
        run_id=result.run_id,
        as_of=result.as_of.isoformat(),
        clusters=result.clusters,
        cluster_cycles=result.cluster_cycles,
        decisions_created=result.decisions_created,
        decisions_skipped=result.decisions_skipped,
        decisions_withdrawn=result.decisions_withdrawn,
        suppressed_units=result.suppressed_units,
    )


class ClaimRequest(BaseModel):
    account_id: str
    credits_requested: int = Field(default=1, ge=1)
    request_key: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=1, max_length=200)
    as_of: datetime | None = None


class ClaimResponse(BaseModel):
    account_id: str
    outcome: str  # "granted" | "held"
    deferred: bool
    latency_ms: float
    standing_decision_id: str | None
    within_budget: bool
    claim_id: str | None
    request_key: str | None
    granted_amount: int
    held_amount: int
    replayed: bool


@router.post("/claim", response_model=ClaimResponse)
def claim(body: ClaimRequest, request: Request) -> ClaimResponse:
    """A bounded synthetic grant operation; use request_key to retry a committed claim.

    as_of optionally selects an explicit simulation time; otherwise the clock is UTC now.
    A storage outage can lose deferred accounting, as documented in architecture §2.
    """
    try:
        outcome = bounded_claim(
            request.app.state.inference_engine,
            account_id=body.account_id,
            credits_requested=body.credits_requested,
            request_key=body.request_key,
            as_of=body.as_of,
            request_started=request.scope.get("request_started"),
        )
    except UnknownAccount as exc:
        raise HTTPException(404, str(exc)) from exc
    except (MissingCycle, ClaimKeyConflict) as exc:
        raise HTTPException(409, str(exc)) from exc
    return ClaimResponse(
        account_id=body.account_id,
        outcome=outcome.outcome,
        deferred=outcome.deferred,
        latency_ms=round(outcome.latency_ms, 3),
        standing_decision_id=outcome.standing_decision_id,
        within_budget=outcome.latency_ms < CLAIM_BUDGET_MS,
        claim_id=outcome.claim_id,
        request_key=outcome.request_key,
        granted_amount=outcome.granted_amount,
        held_amount=outcome.held_amount,
        replayed=outcome.replayed,
    )


class ReviewerRunRequest(BaseModel):
    request_key: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=1, max_length=200)
    run_id: str | None = None
    arm: str = "reviewer"  # arm4/4b operating-point selection is P3.9's job
    as_of: datetime | None = None


class ReviewerSampleResponse(BaseModel):
    sample_index: int
    effect: str
    review_ids: list[str]


class ReviewerRunResponse(BaseModel):
    decision_id: str
    gated: bool
    gate_reason: str | None
    replayed: bool
    overall_effect: str | None
    applied: bool
    samples: list[ReviewerSampleResponse]


@router.post("/cases/{decision_id}/reviewer/run", response_model=ReviewerRunResponse)
def reviewer_run(
    decision_id: str,
    body: ReviewerRunRequest,
    session: Session = Depends(get_session),
    provider: LLMProvider = Depends(get_reviewer_provider),
) -> ReviewerRunResponse:
    """Run the P3.5 reviewer on one decision (architecture §7: idempotency-key
    driven, must not double-charge the LLM).

    404 on an unknown `decision_id`. A repeat POST with the SAME `request_key`
    replays through `reviewer.review_decision`'s own idempotency handling — the
    provider is not called again. A `request_key` reused with a DIFFERENT
    `decision_id`/`arm` is a conflicting payload -> 409, matching `/api/claim`'s
    key-conflict pattern (the key's semantic identity must not silently change).
    """
    decision = session.get(Decision, decision_id)
    if decision is None:
        raise HTTPException(404, f"unknown decision_id {decision_id!r}")

    existing = session.scalars(
        select(LlmReview).where(LlmReview.idempotency_key == body.request_key)
    ).first()
    if existing is not None and (
        existing.decision_id != decision_id or existing.arm != body.arm
    ):
        raise HTTPException(
            409, "request_key already used with a different decision_id/arm"
        )

    outcome = review_decision(
        session,
        decision,
        provider=provider,
        run_id=body.run_id or uuid.uuid4().hex,
        idempotency_key=body.request_key,
        as_of=body.as_of or now_utc(),
        arm=body.arm,
    )
    session.commit()
    return ReviewerRunResponse(
        decision_id=outcome.decision_id,
        gated=outcome.gated,
        gate_reason=outcome.gate_reason,
        replayed=outcome.replayed,
        overall_effect=outcome.overall_effect,
        applied=outcome.applied,
        samples=[
            ReviewerSampleResponse(
                sample_index=s.sample_index, effect=s.effect, review_ids=list(s.review_ids)
            )
            for s in outcome.samples
        ],
    )


@router.get("/audit")
def audit(
    entity_id: str | None = Query(default=None),
    entity_type: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    session: Session = Depends(get_session),
) -> dict:
    """The append-only audit trail, newest first, optionally filtered by entity."""
    stmt = select(AuditEvent).order_by(AuditEvent.timestamp.desc(), AuditEvent.audit_id.desc())
    if entity_id is not None:
        stmt = stmt.where(AuditEvent.entity_id == entity_id)
    if entity_type is not None:
        stmt = stmt.where(AuditEvent.entity_type == entity_type)
    events = session.scalars(stmt.limit(limit)).all()
    return {"count": len(events), "events": [audit_view(e) for e in events]}

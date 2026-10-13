"""Append-only decision lifecycle: supersession never mutates content; illegal
transitions are rejected; every move is audited."""

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.clock import now_utc
from app.core.decision_status import IllegalTransition
from app.db.decisions import current_decision, record_decision, transition_decision
from app.models.enums import (
    ActorType,
    DecisionStatus,
    EmailDomainType,
    ProposedAction,
    VerificationLevel,
)
from app.models.inference import Account, AuditEvent, Decision


def _make_account(session: Session, account_id: str) -> None:
    session.add(
        Account(
            account_id=account_id,
            created_at=now_utc(),
            email_domain_type=EmailDomainType.CONSUMER,
            verification_level=VerificationLevel.EMAIL,
        )
    )
    session.flush()


def _record(session: Session, account_id: str, run: str) -> Decision:
    return record_decision(
        session,
        account_id=account_id,
        risk_score=0.72,
        proposed_action=ProposedAction.HOLD_INCREMENTAL_CREDITS,
        detector_version="det-0",
        policy_version="policy-v1.0",
        run_id=run,
        idempotency_key=f"{account_id}:{run}",
    )


def test_record_then_current(inference_session: Session) -> None:
    _make_account(inference_session, "a1")
    d1 = _record(inference_session, "a1", "run-1")
    assert d1.status == DecisionStatus.PROPOSED
    assert current_decision(inference_session, "a1").decision_id == d1.decision_id


def test_supersession_does_not_mutate_prior_content(inference_session: Session) -> None:
    _make_account(inference_session, "a1")
    d1 = _record(inference_session, "a1", "run-1")
    before = (d1.risk_score, d1.proposed_action, d1.detector_version, d1.created_at)

    d2 = _record(inference_session, "a1", "run-2")

    # prior superseded; content columns untouched; only status/superseded_by moved
    assert d1.status == DecisionStatus.SUPERSEDED
    assert d1.superseded_by == d2.decision_id
    assert (d1.risk_score, d1.proposed_action, d1.detector_version, d1.created_at) == before
    # current view now points at the new decision
    assert current_decision(inference_session, "a1").decision_id == d2.decision_id
    # exactly one supersedable decision remains
    remaining = inference_session.scalars(
        select(Decision).where(Decision.status != DecisionStatus.SUPERSEDED)
    ).all()
    assert [d.decision_id for d in remaining] == [d2.decision_id]


def test_legal_transition_path_and_audit(inference_session: Session) -> None:
    _make_account(inference_session, "a1")
    d = _record(inference_session, "a1", "run-1")
    transition_decision(inference_session, d, DecisionStatus.UNDER_REVIEW, actor=ActorType.LLM)
    transition_decision(
        inference_session, d, DecisionStatus.REVERSED, actor=ActorType.ANALYST, reason="household"
    )
    assert d.status == DecisionStatus.REVERSED

    names = inference_session.scalars(select(AuditEvent.event_name)).all()
    # proposed once, then two status changes
    assert names.count("decision_proposed") == 1
    assert names.count("decision_status_changed") == 2


def test_illegal_transition_rejected(inference_session: Session) -> None:
    _make_account(inference_session, "a1")
    d = _record(inference_session, "a1", "run-1")
    transition_decision(inference_session, d, DecisionStatus.APPLIED, actor=ActorType.SYSTEM)
    transition_decision(inference_session, d, DecisionStatus.REVERSED, actor=ActorType.ANALYST)
    with pytest.raises(IllegalTransition):
        transition_decision(  # reversed -> applied is illegal
            inference_session, d, DecisionStatus.APPLIED, actor=ActorType.ANALYST
        )


def test_reversed_prior_is_not_superseded(inference_session: Session) -> None:
    """A reversed decision is a terminal finding; a later run adds a new decision
    beside it rather than superseding it (docs/policy.md §9.4)."""
    _make_account(inference_session, "a1")
    d1 = _record(inference_session, "a1", "run-1")
    transition_decision(inference_session, d1, DecisionStatus.APPLIED, actor=ActorType.SYSTEM)
    transition_decision(inference_session, d1, DecisionStatus.REVERSED, actor=ActorType.ANALYST)

    d2 = _record(inference_session, "a1", "run-2")
    assert d1.status == DecisionStatus.REVERSED  # untouched
    assert d1.superseded_by is None
    assert current_decision(inference_session, "a1").decision_id == d2.decision_id


def test_audit_count_on_supersession(inference_session: Session) -> None:
    _make_account(inference_session, "a1")
    _record(inference_session, "a1", "run-1")
    _record(inference_session, "a1", "run-2")
    total = inference_session.scalar(select(func.count()).select_from(AuditEvent))
    # run-1 proposed; run-2: superseded(run-1 status_changed + superseded) + proposed = 4
    assert total == 4

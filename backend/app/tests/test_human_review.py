"""Analyst decisions from the console (docs/architecture.md §3, P2.4).

Each analyst action is a legal status transition — audited, recorded as a HumanReview
row — or it is rejected by the status machine and nothing moves.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.clock import now_utc
from app.core.decision_status import IllegalTransition
from app.db.engine import create_inference_schema
from app.db.sqlite_fk import enable_sqlite_foreign_keys
from app.models.enums import (
    DecisionStatus,
    EmailDomainType,
    ProposedAction,
    VerificationLevel,
)
from app.models.inference import Account, AuditEvent, ClusterSnapshot, Decision, HumanReview
from app.services.evidence import associate_evidence, persist_evidence
from app.services.human_review import UnknownAnalystAction, record_human_decision


def _session() -> Session:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True
    )
    engine = enable_sqlite_foreign_keys(engine)
    create_inference_schema(engine)
    return Session(engine, future=True)


def _proposed_decision(s: Session) -> Decision:
    s.add(
        Account(
            account_id="acct-1",
            created_at=now_utc(),
            email_domain_type=EmailDomainType.CONSUMER,
            verification_level=VerificationLevel.EMAIL,
        )
    )
    at = now_utc()
    s.flush()
    s.add(
        ClusterSnapshot(
            snapshot_id="snapshot",
            as_of=at,
            member_account_ids=["acct-1"],
            edges=[],
            evidence_summary={},
            content_hash="snapshot",
            created_at=at,
        )
    )
    s.flush()
    d = Decision(
        cluster_snapshot_id="snapshot",
        decision_id=uuid.uuid4().hex,
        account_id="acct-1",
        created_at=now_utc(),
        detector_version="d",
        policy_version="p",
        risk_score=0.9,
        proposed_action=ProposedAction.HOLD_INCREMENTAL_CREDITS,
        status=DecisionStatus.PROPOSED,
        run_id="run",
        idempotency_key=uuid.uuid4().hex,
    )
    s.add(d)
    s.flush()
    evidence = persist_evidence(
        s,
        {
            "cluster_snapshot_id": "snapshot",
            "cycle_id": "cycle",
            "as_of": at.isoformat(),
            "member_account_ids": ["acct-1"],
            "edges": [],
            "excess": 1.5,
        },
    )
    associate_evidence(s, d.decision_id, evidence.evidence_id)
    return d


def test_apply_transitions_status_and_records_review() -> None:
    s = _session()
    d = _proposed_decision(s)
    review = record_human_decision(s, d, action="apply", reason="confirmed farm")
    assert d.status is DecisionStatus.APPLIED
    assert review.final_outcome == "applied" and review.reviewer_action == "apply"
    # The status machine audited the move.
    events = s.scalars(
        select(AuditEvent).where(AuditEvent.event_name == "decision_status_changed")
    ).all()
    assert len(events) == 1
    assert s.scalars(select(HumanReview)).one().reviewer_reason == "confirmed farm"


def test_start_review_then_reverse() -> None:
    s = _session()
    d = _proposed_decision(s)
    record_human_decision(s, d, action="start_review")
    assert d.status is DecisionStatus.UNDER_REVIEW
    record_human_decision(s, d, action="reverse", reason="legitimate shared use")
    assert d.status is DecisionStatus.REVERSED


def test_illegal_transition_is_rejected() -> None:
    s = _session()
    d = _proposed_decision(s)
    # proposed → reversed is not a legal move (must pass through review/applied first).
    with pytest.raises(IllegalTransition):
        record_human_decision(s, d, action="reverse")
    assert d.status is DecisionStatus.PROPOSED  # untouched


def test_unknown_action_is_rejected() -> None:
    s = _session()
    d = _proposed_decision(s)
    with pytest.raises(UnknownAnalystAction):
        record_human_decision(s, d, action="delete")


def test_legacy_reversal_requires_original_evidence(monkeypatch) -> None:
    from app.services import human_review

    s = _session()
    d = _proposed_decision(s)
    record_human_decision(s, d, action="start_review")
    monkeypatch.setattr(human_review, "decision_evidence", lambda *_: None)
    with pytest.raises(human_review.MissingDecisionEvidence):
        record_human_decision(s, d, action="reverse")
    assert d.status is DecisionStatus.UNDER_REVIEW
    assert len(s.scalars(select(HumanReview)).all()) == 1

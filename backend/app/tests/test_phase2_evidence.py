"""Persisted evidence and stale-session lifecycle regressions (no sealed data)."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.db.decisions import ConcurrentTransition, record_decision, transition_decision
from app.db.engine import create_inference_schema
from app.db.ground_truth_writer import make_ground_truth_sink
from app.db.immutability import ImmutableRowError
from app.models.enums import (
    ActorType,
    DecisionStatus,
    EmailDomainType,
    EventType,
    ProposedAction,
    VerificationLevel,
)
from app.models.inference import (
    Account,
    AccountEvent,
    AuditEvent,
    ClusterSnapshot,
    Decision,
    DetectionEvidence,
)
from app.services.detector import run_detector
from app.services.evidence import associate_evidence, decision_evidence, persist_evidence
from app.services.generator import WORLD_END, generate_world, persist_world

AT = datetime(2026, 6, 1, tzinfo=UTC)


def test_detector_evidence_same_before_and_after_sqlite_datetime_reload(
    inference_session, gt_session,
) -> None:
    world = generate_world(1, "min")
    persist_world(world, inference_session, make_ground_truth_sink(gt_session))
    event = AccountEvent(event_id="aware-event", account_id=world.accounts[0].account_id,
                         timestamp=WORLD_END - timedelta(days=1), event_type=EventType.LOGIN)
    inference_session.add(event)
    inference_session.flush()
    run_detector(inference_session, as_of=WORLD_END, run_id="aware")
    before = set(inference_session.scalars(select(DetectionEvidence.evidence_id)))
    inference_session.expire_all()
    run_detector(inference_session, as_of=WORLD_END, run_id="reloaded")
    assert set(inference_session.scalars(select(DetectionEvidence.evidence_id))) == before


def _case(session: Session) -> Decision:
    session.add(Account(account_id="a", created_at=AT,
                        email_domain_type=EmailDomainType.CONSUMER,
                        verification_level=VerificationLevel.EMAIL))
    session.add(ClusterSnapshot(snapshot_id="s", as_of=AT, member_account_ids=["a"],
                                edges=[], evidence_summary={}, content_hash="h", created_at=AT))
    session.flush()
    decision = record_decision(
        session, account_id="a", risk_score=.8, proposed_action=ProposedAction.TEMPORARY_HOLD,
        detector_version="v1", policy_version="v1", run_id="r", idempotency_key="k",
        cluster_snapshot_id="s",
    )
    transition_decision(session, decision, DecisionStatus.UNDER_REVIEW, actor=ActorType.ANALYST)
    return decision


def test_stale_session_cannot_apply_after_reversal(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'race.db'}")
    create_inference_schema(engine)
    with Session(engine) as setup:
        d = _case(setup)
        identity = d.decision_id
        setup.commit()
    with Session(engine) as first, Session(engine) as stale:
        a = first.get(Decision, identity)
        b = stale.get(Decision, identity)
        assert a is not None and b is not None
        transition_decision(first, a, DecisionStatus.REVERSED, actor=ActorType.ANALYST)
        first.commit()
        with pytest.raises(ConcurrentTransition):
            transition_decision(stale, b, DecisionStatus.APPLIED, actor=ActorType.ANALYST)
        stale.rollback()
    with Session(engine) as verify:
        assert verify.get(Decision, identity).status == DecisionStatus.REVERSED
        assert verify.scalar(select(func.count()).select_from(AuditEvent).where(
            AuditEvent.event_name == "decision_status_changed")) == 2


def test_two_analysts_racing_the_same_apply_yield_one_transition(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'race-apply.db'}")
    create_inference_schema(engine)
    with Session(engine) as setup:
        d = _case(setup)
        identity = d.decision_id
        setup.commit()
    with Session(engine) as first, Session(engine) as second:
        a = first.get(Decision, identity)
        b = second.get(Decision, identity)
        assert a is not None and b is not None
        transition_decision(first, a, DecisionStatus.APPLIED, actor=ActorType.ANALYST)
        first.commit()
        with pytest.raises(ConcurrentTransition):
            transition_decision(second, b, DecisionStatus.APPLIED, actor=ActorType.ANALYST)
        second.rollback()
    with Session(engine) as verify:
        assert verify.get(Decision, identity).status == DecisionStatus.APPLIED
        rows = verify.scalars(
            select(AuditEvent).where(
                AuditEvent.entity_id == identity,
                AuditEvent.event_name == "decision_status_changed",
            )
        ).all()
        applied_transitions = [r for r in rows if r.payload_json.get("to_status") == "applied"]
        assert len(applied_transitions) == 1


def test_detector_supersession_and_analyst_action_cannot_both_win(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'race-supersede.db'}")
    create_inference_schema(engine)
    with Session(engine) as setup:
        d = _case(setup)  # PROPOSED -> UNDER_REVIEW
        identity, account_id = d.decision_id, d.account_id
        setup.commit()

    with Session(engine) as stale:
        b = stale.get(Decision, identity)
        assert b is not None

        with Session(engine) as detector:
            # A detector-driven supersession for the same account
            # (record_decision -> transition_decision) moves the decision the
            # stale session is still holding to SUPERSEDED, out from under it.
            record_decision(
                detector, account_id=account_id, risk_score=.9,
                proposed_action=ProposedAction.TEMPORARY_HOLD,
                detector_version="v2", policy_version="v1", run_id="r2",
                idempotency_key="k2",
            )
            detector.commit()

        with Session(engine) as check:
            audit_count_before = check.scalar(
                select(func.count()).select_from(AuditEvent)
                .where(AuditEvent.entity_id == identity)
            )

        # The CAS fires (and raises) strictly before any audit append, so the
        # losing side's attempt must leave the audit trail untouched.
        with pytest.raises(ConcurrentTransition):
            transition_decision(stale, b, DecisionStatus.APPLIED, actor=ActorType.ANALYST)
        stale.rollback()

    with Session(engine) as verify:
        assert verify.get(Decision, identity).status == DecisionStatus.SUPERSEDED
        audit_count_after = verify.scalar(
            select(func.count()).select_from(AuditEvent).where(AuditEvent.entity_id == identity)
        )
    assert audit_count_after == audit_count_before


def test_original_evidence_survives_restart_and_later_activity(tmp_path) -> None:
    url = f"sqlite:///{tmp_path / 'evidence.db'}"
    engine = create_engine(url)
    create_inference_schema(engine)
    with Session(engine) as session:
        d = _case(session)
        identity = d.decision_id
        payload = {
            "cluster_snapshot_id": "s", "cycle_id": "c", "as_of": AT.isoformat(),
            "allowance": 25, "excess": 1.0, "cap_reasons": [],
            "contributions": [{"feature_id": "F-DEV-1", "contribution": .45}],
            "timeline": [],
        }
        evidence = persist_evidence(session, payload)
        associate_evidence(session, identity, evidence.evidence_id)
        expected = decision_evidence(session, identity)
        session.commit()
    engine.dispose()
    engine = create_engine(url)
    with Session(engine) as session:
        session.add(AccountEvent(event_id="later", account_id="a",
                                 timestamp=AT + timedelta(days=1),
                                 event_type=EventType.CLAIM_CREDITS, credits_amount=25))
        session.commit()
        assert decision_evidence(session, identity) == expected
        detached = decision_evidence(session, identity)
        detached["contributions"][0]["contribution"] = 999
        assert decision_evidence(session, identity) == expected
        row = session.get(DetectionEvidence, expected["evidence_id"])
        row.payload_json = {"changed": True}
        with pytest.raises(ImmutableRowError):
            session.flush()

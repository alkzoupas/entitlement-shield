"""Append-only / immutable records are enforced at the session level (review 4 P1).

Decision content columns, cluster-snapshot content, and every audit row are
immutable; the ONLY legal decision mutations (`status`, `superseded_by`, via the
lifecycle functions) still succeed.
"""

from typing import Any

import pytest
from sqlalchemy import delete, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.clock import now_utc
from app.db.decisions import record_decision, transition_decision
from app.db.immutability import ImmutableRowError
from app.models.enums import (
    ActorType,
    DecisionStatus,
    EmailDomainType,
    ProposedAction,
    VerificationLevel,
)
from app.models.inference import (
    Account,
    AuditEvent,
    ClusterSnapshot,
    Decision,
    DecisionEvidence,
    DetectionEvidence,
    HumanReview,
    LlmReview,
)


def _make_account(session: Session, account_id: str = "a1") -> None:
    session.add(
        Account(
            account_id=account_id,
            created_at=now_utc(),
            email_domain_type=EmailDomainType.CONSUMER,
            verification_level=VerificationLevel.EMAIL,
        )
    )
    session.flush()


def _record(session: Session, run: str = "run-1") -> Decision:
    return record_decision(
        session,
        account_id="a1",
        risk_score=0.5,
        proposed_action=ProposedAction.HOLD_INCREMENTAL_CREDITS,
        detector_version="det-0",
        policy_version="policy-v1.0",
        run_id=run,
        idempotency_key=f"a1:{run}",
    )


def test_decision_content_column_update_raises(inference_session: Session) -> None:
    _make_account(inference_session)
    d = _record(inference_session)
    d.risk_score = 0.99
    with pytest.raises(ImmutableRowError):
        inference_session.flush()


def test_decision_status_and_superseded_by_moves_are_allowed(
    inference_session: Session,
) -> None:
    _make_account(inference_session)
    d = _record(inference_session)
    # Legal lifecycle mutation of status...
    transition_decision(
        inference_session, d, DecisionStatus.UNDER_REVIEW, actor=ActorType.LLM
    )
    inference_session.flush()
    assert d.status == DecisionStatus.UNDER_REVIEW
    # ...and superseded_by via a second run.
    d2 = _record(inference_session, "run-2")
    inference_session.flush()
    assert d.superseded_by == d2.decision_id  # no ImmutableRowError raised


def test_decision_content_core_update_and_delete_are_sealed(
    inference_session: Session,
) -> None:
    _make_account(inference_session)
    decision = _record(inference_session)
    inference_session.commit()
    with pytest.raises(IntegrityError, match="immutable"):
        inference_session.execute(
            update(Decision)
            .where(Decision.decision_id == decision.decision_id)
            .values(risk_score=0.99)
        )
    inference_session.rollback()

    # Delete is tested separately after rollback; no child FK points at this row.
    with pytest.raises(IntegrityError, match="immutable"):
        inference_session.execute(
            delete(Decision).where(Decision.decision_id == decision.decision_id)
        )
    inference_session.rollback()


def test_decision_content_raw_sql_is_sealed_but_status_cas_column_is_allowed(
    inference_session: Session,
) -> None:
    _make_account(inference_session)
    decision = _record(inference_session)
    inference_session.commit()
    with pytest.raises(IntegrityError, match="immutable"):
        inference_session.execute(
            text("UPDATE decisions SET risk_score = :score WHERE decision_id = :id"),
            {"score": 0.99, "id": decision.decision_id},
        )
    inference_session.rollback()
    inference_session.execute(
        text("UPDATE decisions SET status = :status WHERE decision_id = :id"),
        {"status": DecisionStatus.UNDER_REVIEW.name, "id": decision.decision_id},
    )
    inference_session.commit()
    loaded = inference_session.get(Decision, decision.decision_id)
    assert loaded is not None
    assert loaded.status is DecisionStatus.UNDER_REVIEW


def test_cluster_snapshot_content_update_raises(inference_session: Session) -> None:
    snap = ClusterSnapshot(
        snapshot_id="s1",
        as_of=now_utc(),
        member_account_ids=["a1", "a2"],
        edges=[],
        evidence_summary={},
        content_hash="h1",
        created_at=now_utc(),
    )
    inference_session.add(snap)
    inference_session.flush()
    inference_session.commit()
    inference_session.commit()

    snap.member_account_ids = ["a1", "a2", "a3"]
    with pytest.raises(ImmutableRowError):
        inference_session.flush()


def test_cluster_snapshot_core_and_raw_sql_update_delete_are_sealed(
    inference_session: Session,
) -> None:
    snap = ClusterSnapshot(
        snapshot_id="s-core",
        as_of=now_utc(),
        member_account_ids=["a1"],
        edges=[],
        evidence_summary={},
        content_hash="h-core",
        created_at=now_utc(),
    )
    inference_session.add(snap)
    inference_session.flush()
    inference_session.commit()
    with pytest.raises(IntegrityError, match="immutable"):
        inference_session.execute(
            update(ClusterSnapshot)
            .where(ClusterSnapshot.snapshot_id == snap.snapshot_id)
            .values(content_hash="tampered")
        )
    inference_session.rollback()
    with pytest.raises(IntegrityError, match="immutable"):
        inference_session.execute(
            text("DELETE FROM cluster_snapshots WHERE snapshot_id = :id"),
            {"id": snap.snapshot_id},
        )
    inference_session.rollback()


def _add_audit(session: Session) -> AuditEvent:
    ev = AuditEvent(
        audit_id="au1",
        timestamp=now_utc(),
        entity_type="decision",
        entity_id="d1",
        event_name="decision_proposed",
        actor_type=ActorType.SYSTEM,
        payload_json={"k": "v"},
    )
    session.add(ev)
    session.flush()
    return ev


def test_audit_row_update_raises(inference_session: Session) -> None:
    ev = _add_audit(inference_session)
    ev.entity_id = "tampered"
    with pytest.raises(ImmutableRowError):
        inference_session.flush()


def test_audit_row_delete_raises(inference_session: Session) -> None:
    ev = _add_audit(inference_session)
    inference_session.delete(ev)
    with pytest.raises(ImmutableRowError):
        inference_session.flush()


@pytest.mark.parametrize("operation", ["update", "delete"])
def test_audit_core_and_raw_sql_are_sealed(
    inference_session: Session, operation: str
) -> None:
    ev = _add_audit(inference_session)
    inference_session.commit()
    statement: Any
    if operation == "update":
        statement = (
            update(AuditEvent)
            .where(AuditEvent.audit_id == ev.audit_id)
            .values(entity_id="tampered")
        )
        raw = text("UPDATE audit_events SET entity_id = 'tampered' WHERE audit_id = :id")
    else:
        statement = delete(AuditEvent).where(AuditEvent.audit_id == ev.audit_id)
        raw = text("DELETE FROM audit_events WHERE audit_id = :id")
    with pytest.raises(IntegrityError, match="immutable"):
        inference_session.execute(statement)
    inference_session.rollback()
    with pytest.raises(IntegrityError, match="immutable"):
        inference_session.execute(raw, {"id": ev.audit_id})
    inference_session.rollback()


def _add_reviews(session: Session) -> tuple[HumanReview, LlmReview]:
    _make_account(session)
    decision = _record(session)
    human = HumanReview(
        review_id="human-1", decision_id=decision.decision_id,
        reviewer_action="apply", reviewer_reason="reason",
        final_outcome="applied", created_at=now_utc(),
    )
    llm = LlmReview(
        review_id="llm-1", decision_id=decision.decision_id,
        idempotency_key="idem-1", sample_index=0, arm="reviewer",
        provider="stub", prompt_hash="ph-1",
        model_snapshot="model", prompt_version="v1",
    )
    session.add_all([human, llm])
    session.flush()
    return human, llm


@pytest.mark.parametrize("table, column, value", [
    (HumanReview, "reviewer_reason", "tampered"),
    (LlmReview, "model_snapshot", "tampered"),
])
def test_review_orm_core_and_raw_sql_are_sealed(
    inference_session: Session, table: Any, column: str, value: str
) -> None:
    human, llm = _add_reviews(inference_session)
    inference_session.commit()
    review = human if table is HumanReview else llm
    setattr(review, column, value)
    with pytest.raises(ImmutableRowError):
        inference_session.flush()
    inference_session.rollback()

    review_id = review.review_id
    with pytest.raises(IntegrityError, match="immutable"):
        inference_session.execute(
            update(table).where(table.review_id == review_id).values(**{column: value})
        )
    inference_session.rollback()
    table_name = "human_reviews" if table is HumanReview else "llm_reviews"
    with pytest.raises(IntegrityError, match="immutable"):
        inference_session.execute(
            text(
                f"UPDATE {table_name} SET {column} = 'tampered' "
                "WHERE review_id = :id"
            ),
            {"id": review_id},
        )
    inference_session.rollback()
    with pytest.raises(IntegrityError, match="immutable"):
        inference_session.execute(
            text(f"DELETE FROM {table_name} WHERE review_id = :id"),
            {"id": review_id},
        )
    inference_session.rollback()


def _add_evidence(
    session: Session, *, link_evidence: bool = True
) -> tuple[DetectionEvidence, Decision, DecisionEvidence | None]:
    _make_account(session)
    decision = _record(session)
    snap = ClusterSnapshot(
        snapshot_id="s-evidence",
        as_of=now_utc(),
        member_account_ids=["a1"],
        edges=[],
        evidence_summary={},
        content_hash="h-evidence",
        created_at=now_utc(),
    )
    session.add(snap)
    session.flush()
    evidence = DetectionEvidence(
        evidence_id="ev1",
        cluster_snapshot_id=snap.snapshot_id,
        cycle_id="c1",
        as_of=now_utc(),
        payload_json={"k": "v"},
    )
    session.add(evidence)
    session.flush()
    link = None
    if link_evidence:
        link = DecisionEvidence(decision_id=decision.decision_id, evidence_id=evidence.evidence_id)
        session.add(link)
        session.flush()
    return evidence, decision, link


def test_detection_evidence_core_update_raises_at_db_level(inference_session: Session) -> None:
    """Mapper `before_update` hooks never fire for Core `update(...)` statements —
    only a real DB-level trigger can stop this (review F7-a)."""
    evidence, _decision, _link = _add_evidence(inference_session)
    with pytest.raises(IntegrityError):
        inference_session.execute(
            update(DetectionEvidence)
            .where(DetectionEvidence.evidence_id == evidence.evidence_id)
            .values(cycle_id="tampered")
        )
    inference_session.rollback()


def test_detection_evidence_core_delete_raises_at_db_level(inference_session: Session) -> None:
    """Unlinked, so a foreign key from `decision_evidence` cannot also explain the
    raise — this isolates the seal trigger itself (review F7-a follow-up)."""
    evidence, _decision, _link = _add_evidence(inference_session, link_evidence=False)
    with pytest.raises(IntegrityError):
        inference_session.execute(
            delete(DetectionEvidence).where(DetectionEvidence.evidence_id == evidence.evidence_id)
        )
    inference_session.rollback()


def test_decision_evidence_core_update_raises_at_db_level(inference_session: Session) -> None:
    """Retargets to a second, *valid* evidence row so the foreign key on
    `evidence_id` is satisfied — isolates the seal trigger itself, rather than
    a constraint violation on the new value (review F7-a follow-up)."""
    _evidence, decision, link = _add_evidence(inference_session)
    other_evidence = DetectionEvidence(
        evidence_id="ev2",
        cluster_snapshot_id=_evidence.cluster_snapshot_id,
        cycle_id="c1",
        as_of=now_utc(),
        payload_json={"k": "v2"},
    )
    inference_session.add(other_evidence)
    inference_session.flush()
    with pytest.raises(IntegrityError):
        inference_session.execute(
            update(DecisionEvidence)
            .where(DecisionEvidence.decision_id == decision.decision_id)
            .values(evidence_id=other_evidence.evidence_id)
        )
    inference_session.rollback()


def test_decision_evidence_core_delete_raises_at_db_level(inference_session: Session) -> None:
    _evidence, decision, link = _add_evidence(inference_session)
    with pytest.raises(IntegrityError):
        inference_session.execute(
            delete(DecisionEvidence).where(DecisionEvidence.decision_id == decision.decision_id)
        )
    inference_session.rollback()

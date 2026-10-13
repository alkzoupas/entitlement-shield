"""Gate B authority and invocation regressions (synthetic inference stores only)."""
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.decision_status import SUPERSEDABLE
from app.db.decisions import current_decision, record_decision
from app.models.enums import DecisionStatus, ProposedAction
from app.models.inference import (
    AuditEvent,
    Decision,
    DetectorInvocation,
    HumanReview,
    StandingAuthority,
)
from app.services import detector, human_review
from app.services.generator import WORLD_END
from app.tests.test_api import _first_proposed_hold, _run_detector, env  # noqa: F401
from app.tests.test_detector import _active, _seed_min_world


@pytest.mark.parametrize("shrink", [False, True])
def test_new_frame_withdraws_unaffirmed_authority(
    inference_session, gt_session, monkeypatch, shrink
):
    _seed_min_world(inference_session, gt_session)
    cutoff = WORLD_END - timedelta(days=1)
    detector.run_detector(inference_session, as_of=cutoff, run_id="old")
    old = _active(inference_session)
    assert old
    authorities = inference_session.scalars(select(StandingAuthority)).all()
    assert any(a.decision_id in {d.decision_id for d in old} for a in authorities)
    from app.db.decisions import transition_decision
    from app.models.enums import ActorType
    for row in old:
        transition_decision(inference_session, row, DecisionStatus.APPLIED, actor=ActorType.ANALYST)
    if shrink:
        monkeypatch.setattr(detector, "select_enforcement_targets", lambda *a: [])
    else:
        original = detector.decide
        from dataclasses import replace
        monkeypatch.setattr(detector, "decide", lambda *a, **k: replace(
            original(*a, **k), proposed_action=ProposedAction.ALLOW))
    result = detector.run_detector(
        inference_session, as_of=cutoff + timedelta(hours=1), run_id="new")
    assert result.decisions_withdrawn > 0
    for row in old:
        if not shrink or row.proposed_action in detector._GRANT_RESTRICTING:
            assert row.status == DecisionStatus.SUPERSEDED
            current = current_decision(inference_session, row.account_id)
            assert current is None or current.decision_id != row.decision_id
    assert {a.run_id for a in inference_session.scalars(select(StandingAuthority))} == {"new"}


def test_reused_invocation_rejected_across_sessions(env):  # noqa: F811
    with Session(env.engine) as session:
        detector.run_detector(session, as_of=WORLD_END, run_id="once")
        session.commit()
    with Session(env.engine) as session:
        before = session.scalar(select(func.count()).select_from(AuditEvent))
        with pytest.raises(ValueError, match="identity already used"):
            detector.run_detector(session, as_of=WORLD_END, run_id="once")
        session.rollback()
        assert session.scalar(select(func.count()).select_from(DetectorInvocation)) == 1
        assert session.scalar(select(func.count()).select_from(AuditEvent)) == before
        with pytest.raises(ValueError, match="backward"):
            detector.run_detector(session, as_of=WORLD_END - timedelta(hours=1), run_id="past")


def test_api_ids_are_server_owned(env):  # noqa: F811
    assert env.client.post("/api/detector/run", json={"run_id": "forged"}).status_code == 422
    first = _run_detector(env.client, "ignored")
    second = _run_detector(env.client, "ignored")
    assert first["run_id"] != second["run_id"]
    with Session(env.engine) as session:
        assert session.scalar(select(func.count()).select_from(DetectorInvocation)) == 2


def test_console_ignores_forged_action_time(env, monkeypatch):  # noqa: F811
    _run_detector(env.client, "ignored")
    decision = _first_proposed_hold(env.engine)
    env.client.post(f"/cases/{decision.decision_id}/human-decision", data={"action": "apply"})
    at = datetime(2026, 9, 9, 12, tzinfo=UTC)
    monkeypatch.setattr(human_review, "now_utc", lambda: at)
    response = env.client.post(f"/cases/{decision.decision_id}/human-decision",
                               data={"action": "reverse", "as_of": "1900-01-01T00:00"})
    assert response.status_code == 200
    with Session(env.engine) as session:
        review = session.scalar(select(HumanReview).where(
            HumanReview.decision_id == decision.decision_id,
            HumanReview.reviewer_action == "reverse"))
        assert review.created_at.replace(tzinfo=UTC) == at
        events = session.scalars(select(AuditEvent).where(
            AuditEvent.entity_id == decision.decision_id,
            AuditEvent.event_name == "decision_status_changed")).all()
        assert events[-1].timestamp.replace(tzinfo=UTC) == at
        from app.models.inference import SuppressionRecord
        suppression = session.scalar(select(SuppressionRecord).where(
            SuppressionRecord.decision_id == decision.decision_id))
        assert suppression.created_at.replace(tzinfo=UTC) == at


def test_concurrent_different_publications_leave_one_authority(env):  # noqa: F811
    from app.models.inference import Account
    with Session(env.engine) as session:
        account_id = session.scalar(select(Account.account_id).limit(1))
    barrier = Barrier(2)

    def publish(key):
        with Session(env.engine) as session:
            barrier.wait(timeout=5)
            row = record_decision(session, account_id=account_id, risk_score=0.9,
                                  proposed_action=ProposedAction.LIGHT_FRICTION,
                                  detector_version=key, policy_version="p", run_id=key,
                                  idempotency_key=key)
            session.commit()
            return row.decision_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(publish, ["one", "two"]))
    with Session(env.engine) as session:
        active = session.scalars(select(Decision).where(
            Decision.account_id == account_id, Decision.status.in_(SUPERSEDABLE))).all()
        assert len(active) == 1
        assert active[0].decision_id in ids
        assert current_decision(session, account_id).decision_id == active[0].decision_id

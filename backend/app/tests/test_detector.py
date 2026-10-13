"""End-to-end batch detector integration (docs/architecture.md §2/§7, plan Step 6).

Covers: seed → cluster → score → decision → audit produces decisions and a typed
run audit event; a double run is idempotent (no duplicate decisions); an analyst
reversal changes final status and APPENDS an audit event without mutating the
original decision.
"""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.decision_status import SUPERSEDABLE
from app.db.decisions import record_decision, transition_decision
from app.db.ground_truth_writer import make_ground_truth_sink
from app.models.enums import (
    ActorType,
    DecisionStatus,
    EmailDomainType,
    EventType,
    ProposedAction,
    VerificationLevel,
)
from app.models.inference import Account, AccountEvent, AuditEvent, Decision
from app.services import features as F
from app.services.audit import decision_audit_trail
from app.services.detector import _cycle_claimed, run_detector
from app.services.generator import WORLD_END, generate_world, persist_world
from app.services.policy_engine import record_reversal
from app.tests.support import AS_OF, Builder

_WEIGHT_CONSTANTS = (
    "_W_DEV_1", "_W_DEV_2", "_W_PAY_1", "_W_CON_1", "_W_DOM_1", "_W_VER_1",
    "_W_CYC_1", "_W_CYC_2", "_W_CYC_3", "_W_CYC_4", "_W_DOM_2", "_W_AGE_1",
)


def _active(session: Session) -> list[Decision]:
    return list(
        session.scalars(select(Decision).where(Decision.status.in_(SUPERSEDABLE))).all()
    )


def _seed_min_world(inference_session: Session, gt_session: Session) -> None:
    world = generate_world(1, "min")
    persist_world(world, inference_session, make_ground_truth_sink(gt_session))
    inference_session.flush()


def _decision_count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Decision)) or 0


def test_seed_cluster_score_decision_audit(
    inference_session: Session, gt_session: Session
) -> None:
    _seed_min_world(inference_session, gt_session)

    result = run_detector(inference_session, as_of=WORLD_END, run_id="run-1")

    assert result.clusters > 0
    assert result.cluster_cycles > 0
    assert result.decisions_created > 0  # the min world's farms clear the gates

    # Every recorded decision proposes an action and points at a cluster snapshot.
    decisions = list(inference_session.scalars(select(Decision)).all())
    assert all(d.proposed_action is not ProposedAction.ALLOW for d in decisions)
    assert all(d.cluster_snapshot_id is not None for d in decisions)
    assert all(d.run_id == "run-1" for d in decisions)

    # The run is recorded as a typed audit event, and each decision has a proposal.
    run_events = inference_session.scalars(
        select(AuditEvent).where(AuditEvent.event_name == "detector_run_recorded")
    ).all()
    assert len(run_events) == 1
    assert run_events[0].payload_json["decisions_created"] == result.decisions_created
    for d in decisions:
        names = {e.event_name for e in decision_audit_trail(inference_session, d.decision_id)}
        assert "decision_proposed" in names


def test_double_run_is_idempotent(
    inference_session: Session, gt_session: Session
) -> None:
    _seed_min_world(inference_session, gt_session)

    first = run_detector(inference_session, as_of=WORLD_END, run_id="run-1")
    count_after_first = _decision_count(inference_session)

    second = run_detector(inference_session, as_of=WORLD_END, run_id="run-2")
    count_after_second = _decision_count(inference_session)

    assert first.decisions_created > 0
    assert second.decisions_created == 0  # everything already recorded
    assert second.decisions_skipped == first.decisions_created
    assert count_after_second == count_after_first  # no duplicate rows


def test_config_change_to_allow_supersedes_stale_decision(
    inference_session: Session, gt_session: Session, monkeypatch
) -> None:
    """Review finding 4: a config change that sends a previously-actioned unit to ALLOW
    must supersede the stale active decision — it is no longer produced, so it can no
    longer be current. Reconciliation withdraws it with an audited config-change reason."""
    _seed_min_world(inference_session, gt_session)

    first = run_detector(inference_session, as_of=WORLD_END, run_id="run-1")
    assert first.decisions_created > 0
    # Active (supersedable) decisions after run 1 — an account spanning several cycle
    # units keeps only its latest decision active, so this is ≤ decisions_created.
    stale = _active(inference_session)
    assert len(stale) > 0

    # Isolate reconciliation from the separately tested Gate A manifest guard.
    from app.services import detector
    verified = detector.verified_manifest_hash()
    monkeypatch.setattr(detector, "verified_manifest_hash", lambda: verified)

    # New config: zero every feature weight → every unit scores 0 → ALLOW; the new
    # detector_version yields a distinct fingerprint (so the old keys are not affirmed).
    for name in _WEIGHT_CONSTANTS:
        monkeypatch.setattr(F, name, 0.0)

    second = run_detector(
        inference_session, as_of=WORLD_END, run_id="run-2",
        detector_version="dev-detector-v2",
    )

    assert second.decisions_created == 0  # nothing is actionable under the new config
    assert second.decisions_withdrawn == len(stale)  # every active row reconciled
    assert _active(inference_session) == []  # no run-1 decision remains current
    for d in stale:
        assert d.status is DecisionStatus.SUPERSEDED
        trail = decision_audit_trail(inference_session, d.decision_id)
        assert any(
            e.event_name == "decision_status_changed"
            and "config change" in (e.payload_json.get("reason") or "")
            for e in trail
        )


def test_double_run_reconciliation_is_a_noop(
    inference_session: Session, gt_session: Session
) -> None:
    """The reconciliation must not disturb a same-config re-run: the second run affirms
    the identical keys, so nothing is withdrawn (idempotency preserved, finding 4)."""
    _seed_min_world(inference_session, gt_session)
    first = run_detector(inference_session, as_of=WORLD_END, run_id="run-1")
    active_after_first = {d.decision_id for d in _active(inference_session)}

    second = run_detector(inference_session, as_of=WORLD_END, run_id="run-2")

    assert second.decisions_withdrawn == 0
    assert first.decisions_created > 0
    assert {d.decision_id for d in _active(inference_session)} == active_after_first


def test_reversal_appends_audit_without_mutating(inference_session: Session) -> None:
    b = Builder(inference_session)
    aid = b.account("rev")
    b.flush()
    decision = record_decision(
        inference_session,
        account_id=aid,
        risk_score=0.82,
        proposed_action=ProposedAction.TEMPORARY_HOLD,
        detector_version="dev-detector-v1",
        policy_version="policy-v1",
        run_id="run-1",
        idempotency_key="k-rev",
        cluster_snapshot_id=None,
    )
    transition_decision(
        inference_session, decision, DecisionStatus.APPLIED, actor=ActorType.SYSTEM
    )
    trail_before = len(decision_audit_trail(inference_session, decision.decision_id))
    original_score, original_action = decision.risk_score, decision.proposed_action

    record_reversal(
        inference_session,
        decision,
        member_account_ids=[aid, "sibling"],
        edges=[(aid, "sibling", 1)],
        excess=1.5,
        actor=ActorType.ANALYST,
        reason="analyst override",
        as_of=AS_OF,
    )

    assert decision.status is DecisionStatus.REVERSED
    # Content columns untouched — reversal appends, never mutates (policy §9.5).
    assert decision.risk_score == original_score
    assert decision.proposed_action is original_action
    trail = decision_audit_trail(inference_session, decision.decision_id)
    assert len(trail) == trail_before + 1  # exactly one appended event
    reversal_events = [
        e
        for e in trail
        if e.event_name == "decision_status_changed"
        and e.payload_json["to_status"] == DecisionStatus.REVERSED.value
    ]
    assert len(reversal_events) == 1
    assert reversal_events[0].actor_type is ActorType.ANALYST


def test_cycle_claimed_uses_half_open_ownership_and_inclusive_as_of(
    inference_session: Session,
) -> None:
    """Gate G: cycle ownership is `[start, end)`, independently clamped by as_of."""
    inference_session.add(
        Account(
            account_id="boundary-acct",
            created_at=AS_OF,
            email_domain_type=EmailDomainType.CONSUMER,
            verification_level=VerificationLevel.EMAIL,
        )
    )
    inference_session.flush()
    boundary = AS_OF
    inference_session.add(
        AccountEvent(
            event_id="boundary-event",
            account_id="boundary-acct",
            timestamp=boundary,
            event_type=EventType.CLAIM_CREDITS,
            credits_amount=100,
            endpoint_cost_tier=None,
        )
    )
    inference_session.flush()

    start = boundary - timedelta(days=30)
    # Exact cycle end belongs to the following cycle, not the one ending here.
    assert _cycle_claimed(inference_session, ["boundary-acct"], start, boundary, boundary) == 0
    assert _cycle_claimed(
        inference_session, ["boundary-acct"], boundary, boundary + timedelta(days=30), boundary
    ) == 100
    # `as_of` is inclusive, but never permits a future event into an earlier frame.
    assert _cycle_claimed(
        inference_session, ["boundary-acct"], boundary, boundary + timedelta(days=30),
        boundary - timedelta(microseconds=1),
    ) == 0

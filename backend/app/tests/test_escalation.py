"""P3.6: escalation lifecycle and 7-day auto-release (docs/phase-3-plan.md P3.6,
docs/decisions.md §12 L-5).

Unit-style fixtures (decision + hand-built `DetectionEvidence` payload) mirror
`test_reviewer.py`'s pattern; the nightly-reconciliation test uses the real
`run_detector` pipeline over a generated world, mirroring `test_detector.py`'s
config-change reconciliation test.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.clock import now_utc
from app.core.decision_status import LEGAL_TRANSITIONS, assert_transition
from app.db.decisions import transition_decision
from app.db.engine import create_inference_schema
from app.db.ground_truth_writer import make_ground_truth_sink
from app.db.sqlite_fk import enable_sqlite_foreign_keys
from app.models.claims import SyntheticClaim
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
    EntitlementCycle,
    EscalationRecord,
    SuppressionRecord,
)
from app.services.detector import run_detector
from app.services.escalation import (
    ESCALATION_DAYS,
    escalate_decision,
    has_open_escalation,
    release_expired_escalations,
)
from app.services.evidence import associate_evidence, persist_evidence
from app.services.fake_reviewer import FakeMode, FakeReviewerProvider
from app.services.generator import WORLD_END, generate_world, persist_world
from app.services.reviewer import review_decision

AS_OF = now_utc()

# The prior (pre-P3.6) full legal-transition matrix, hardcoded so a regression that
# accidentally adds/removes an edge elsewhere is caught, not just the two edges P3.6
# deliberately touches.
_PRIOR_MATRIX: dict[DecisionStatus, frozenset[DecisionStatus]] = {
    DecisionStatus.PROPOSED: frozenset(
        {DecisionStatus.UNDER_REVIEW, DecisionStatus.APPLIED, DecisionStatus.SUPERSEDED}
    ),
    DecisionStatus.UNDER_REVIEW: frozenset(
        {DecisionStatus.APPLIED, DecisionStatus.REVERSED, DecisionStatus.SUPERSEDED}
    ),
    DecisionStatus.APPLIED: frozenset({DecisionStatus.REVERSED, DecisionStatus.SUPERSEDED}),
    DecisionStatus.REVERSED: frozenset(),
    DecisionStatus.SUPERSEDED: frozenset(),
}


def _session() -> Session:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True
    )
    engine = enable_sqlite_foreign_keys(engine)
    create_inference_schema(engine)
    return Session(engine, future=True)


def _account(session: Session, account_id: str) -> None:
    session.add(
        Account(
            account_id=account_id,
            created_at=AS_OF - timedelta(days=120),
            email_domain_type=EmailDomainType.CONSUMER,
            verification_level=VerificationLevel.EMAIL,
        )
    )
    session.flush()


def _cluster_snapshot(session: Session, snapshot_id: str, member_account_ids: list[str]) -> None:
    session.add(
        ClusterSnapshot(
            snapshot_id=snapshot_id,
            as_of=AS_OF,
            member_account_ids=member_account_ids,
            edges=[],
            evidence_summary={},
            content_hash=snapshot_id,
            created_at=AS_OF,
        )
    )
    session.flush()


def _decision_with_evidence(
    session: Session,
    *,
    decision_id: str,
    account_id: str,
    cluster_snapshot_id: str = "snap-1",
    member_account_ids: list[str] | None = None,
    excess: float = 0.5,
    proposed_action: ProposedAction = ProposedAction.LIGHT_FRICTION,
    status: DecisionStatus = DecisionStatus.PROPOSED,
) -> Decision:
    members = member_account_ids or [account_id]
    d = Decision(
        decision_id=decision_id,
        account_id=account_id,
        cluster_snapshot_id=cluster_snapshot_id,
        created_at=AS_OF,
        detector_version="det-1",
        policy_version="pol-1",
        risk_score=0.9,
        proposed_action=proposed_action,
        status=status,
        run_id="run-1",
        idempotency_key=uuid.uuid4().hex,
    )
    session.add(d)
    session.flush()

    payload = {
        "cluster_snapshot_id": cluster_snapshot_id,
        "cycle_id": "cycle-1",
        "as_of": AS_OF.isoformat(),
        "snapshot_hash": cluster_snapshot_id,
        "member_account_ids": members,
        "edges": [],
        "qualified_stream_count": 2,
        "allowance": 25,
        "ambiguous": False,
        "excess": excess,
        "risk_score": 0.9,
        "proposed_action": proposed_action.value,
        "cap_reasons": [],
        "suppressed": False,
        "detector_version": "det-1",
        "policy_version": "pol-1",
        "feature_manifest_hash": "fmh-1",
        "rung": "S2",
        "contributions": [],
        "window_start": (AS_OF - timedelta(days=30)).isoformat(),
        "window_end": AS_OF.isoformat(),
        "timeline": [],
    }
    evidence = persist_evidence(session, payload)
    associate_evidence(session, d.decision_id, evidence.evidence_id)
    return d


def _held_claim(session: Session, *, decision: Decision, account_id: str, held_amount: int) -> None:
    session.add(
        EntitlementCycle(
            cycle_id="cycle-current",
            account_id=account_id,
            cycle_start=AS_OF - timedelta(days=5),
            cycle_end=AS_OF + timedelta(days=25),
            credits_granted=25,
            credits_consumed=0,
            credits_expired=0,
        )
    )
    session.flush()
    session.add(
        SyntheticClaim(
            claim_id=f"claim-{uuid.uuid4().hex[:8]}",
            request_key=f"req-{uuid.uuid4().hex[:8]}",
            account_id=account_id,
            cycle_id="cycle-current",
            requested_amount=held_amount,
            granted_amount=0,
            held_amount=held_amount,
            decision_id=decision.decision_id,
            created_at=AS_OF,
            deferred=False,
            hold_state="confirmed",
        )
    )
    session.flush()


# --- LEGAL_TRANSITIONS edges ---------------------------------------------------


def test_applied_to_under_review_edge_added() -> None:
    assert DecisionStatus.UNDER_REVIEW in LEGAL_TRANSITIONS[DecisionStatus.APPLIED]
    assert_transition(DecisionStatus.APPLIED, DecisionStatus.UNDER_REVIEW)  # does not raise


def test_under_review_to_superseded_edge_already_existed() -> None:
    assert DecisionStatus.SUPERSEDED in LEGAL_TRANSITIONS[DecisionStatus.UNDER_REVIEW]


def test_no_other_edge_changed() -> None:
    """Full-matrix regression: the only diff from the prior matrix is the single
    `applied -> under_review` addition."""
    expected = dict(_PRIOR_MATRIX)
    expected[DecisionStatus.APPLIED] = _PRIOR_MATRIX[DecisionStatus.APPLIED] | {
        DecisionStatus.UNDER_REVIEW
    }
    assert LEGAL_TRANSITIONS == expected


# --- escalate_decision: credit release -----------------------------------------


def test_escalating_applied_decision_releases_held_credits_once() -> None:
    session = _session()
    _account(session, "acct-1")
    _cluster_snapshot(session, "snap-1", ["acct-1"])
    decision = _decision_with_evidence(
        session,
        decision_id="dec-1",
        account_id="acct-1",
        proposed_action=ProposedAction.HOLD_INCREMENTAL_CREDITS,
        status=DecisionStatus.APPLIED,
    )
    _held_claim(session, decision=decision, account_id="acct-1", held_amount=10)

    record = escalate_decision(session, decision, source="llm_review", as_of=AS_OF)

    assert decision.status is DecisionStatus.UNDER_REVIEW
    claim = session.scalars(select(SyntheticClaim)).one()
    assert claim.hold_state == "none"
    assert claim.held_amount == 0
    assert claim.granted_amount == 10

    credit_events = session.scalars(
        select(AuditEvent).where(AuditEvent.event_name == "credits_restored")
    ).all()
    assert len(credit_events) == 1
    assert credit_events[0].payload_json["amount_restored"] == 10

    escalated_events = session.scalars(
        select(AuditEvent).where(AuditEvent.event_name == "decision_escalated")
    ).all()
    assert len(escalated_events) == 1
    assert escalated_events[0].payload_json["escalation_id"] == record.escalation_id
    assert escalated_events[0].payload_json["source"] == "llm_review"

    assert record.expires_at == AS_OF + timedelta(days=ESCALATION_DAYS)
    # No suppression record — escalation is not a reversal.
    assert session.scalars(select(SuppressionRecord)).all() == []


def test_escalating_proposed_decision_is_legal_and_no_op_on_credits() -> None:
    session = _session()
    _account(session, "acct-1")
    _cluster_snapshot(session, "snap-1", ["acct-1"])
    decision = _decision_with_evidence(
        session, decision_id="dec-1", account_id="acct-1", status=DecisionStatus.PROPOSED
    )

    record = escalate_decision(session, decision, source="llm_review", as_of=AS_OF)
    assert decision.status is DecisionStatus.UNDER_REVIEW
    assert has_open_escalation(session, decision)
    assert record.source == "llm_review"


# --- auto-release timing / idempotency / as_of-bounded -------------------------


def _escalated_decision(session: Session) -> Decision:
    _account(session, "acct-1")
    _cluster_snapshot(session, "snap-1", ["acct-1"])
    decision = _decision_with_evidence(
        session,
        decision_id="dec-1",
        account_id="acct-1",
        proposed_action=ProposedAction.HOLD_INCREMENTAL_CREDITS,
        status=DecisionStatus.APPLIED,
    )
    escalate_decision(session, decision, source="llm_review", as_of=AS_OF)
    return decision


def test_auto_release_does_not_fire_one_second_before_expiry() -> None:
    session = _session()
    decision = _escalated_decision(session)
    expires_at = AS_OF + timedelta(days=ESCALATION_DAYS)

    released = release_expired_escalations(
        session, as_of=expires_at - timedelta(seconds=1), run_id="release-1"
    )

    assert released == 0
    assert decision.status is DecisionStatus.UNDER_REVIEW


def test_auto_release_fires_exactly_at_expiry() -> None:
    session = _session()
    decision = _escalated_decision(session)
    expires_at = AS_OF + timedelta(days=ESCALATION_DAYS)

    released = release_expired_escalations(session, as_of=expires_at, run_id="release-1")

    assert released == 1
    assert decision.status is DecisionStatus.SUPERSEDED
    events = session.scalars(
        select(AuditEvent).where(AuditEvent.event_name == "escalation_auto_released")
    ).all()
    assert len(events) == 1
    assert events[0].payload_json["run_id"] == "release-1"

    status_changes = session.scalars(
        select(AuditEvent).where(
            AuditEvent.event_name == "decision_status_changed",
            AuditEvent.entity_id == decision.decision_id,
        )
    ).all()
    last = status_changes[-1]
    assert last.payload_json["to_status"] == "superseded"
    assert last.payload_json["reason"] == "escalation_expired"
    assert last.payload_json["actor_type"] == "system"

    # No suppression record written by the auto-release path.
    assert session.scalars(select(SuppressionRecord)).all() == []


def test_auto_release_is_idempotent() -> None:
    session = _session()
    _escalated_decision(session)
    expires_at = AS_OF + timedelta(days=ESCALATION_DAYS)

    first = release_expired_escalations(session, as_of=expires_at, run_id="release-1")
    second = release_expired_escalations(session, as_of=expires_at, run_id="release-2")

    assert first == 1
    assert second == 0  # already superseded — no re-processing, no error
    events = session.scalars(
        select(AuditEvent).where(AuditEvent.event_name == "escalation_auto_released")
    ).all()
    assert len(events) == 1


def test_auto_release_is_as_of_bounded_not_wall_clock() -> None:
    """A historical run's `as_of` sits before the real expiry — release must not act
    on it even though real wall-clock time has since passed the expiry (finding-11
    pattern)."""
    session = _session()
    decision = _escalated_decision(session)
    historical_as_of = AS_OF + timedelta(days=1)  # well before the 7-day expiry

    released = release_expired_escalations(session, as_of=historical_as_of, run_id="release-1")

    assert released == 0
    assert decision.status is DecisionStatus.UNDER_REVIEW


def test_analyst_action_before_expiry_is_skipped_by_later_auto_release() -> None:
    session = _session()
    decision = _escalated_decision(session)

    # An analyst resolves the case before the 7-day window lapses.
    transition_decision(session, decision, DecisionStatus.REVERSED, actor=ActorType.ANALYST)
    assert decision.status is DecisionStatus.REVERSED

    expires_at = AS_OF + timedelta(days=ESCALATION_DAYS)
    released = release_expired_escalations(session, as_of=expires_at, run_id="release-1")

    assert released == 0
    assert decision.status is DecisionStatus.REVERSED  # untouched, not re-superseded


# --- integration: gating skips a decision under open escalation ----------------


def test_gating_skips_escalated_decision_provider_never_called() -> None:
    session = _session()
    _account(session, "acct-1")
    _cluster_snapshot(session, "snap-1", ["acct-1"])
    decision = _decision_with_evidence(
        session, decision_id="dec-1", account_id="acct-1", status=DecisionStatus.PROPOSED
    )
    escalate_decision(session, decision, source="llm_review", as_of=AS_OF)
    assert decision.status is DecisionStatus.UNDER_REVIEW

    class _CountingProvider:
        def __init__(self, inner) -> None:
            self.inner = inner
            self.calls = 0

        def complete(self, request):
            self.calls += 1
            return self.inner.complete(request)

    provider = _CountingProvider(FakeReviewerProvider(mode=FakeMode.NORMAL))
    outcome = review_decision(
        session, decision, provider=provider, run_id="r2", idempotency_key="k2", as_of=AS_OF
    )

    assert outcome.gated is True
    assert outcome.gate_reason == "escalated_under_review"
    assert provider.calls == 0


# --- integration: review_decision's escalate outcome actually escalates --------


def test_review_decision_escalate_outcome_transitions_and_releases_credit_on_applied() -> None:
    """P3.7 Part A regression: `review_decision` concluding `overall_effect ==
    "escalate"` must not just record the label on the `LlmReview` rows — it must
    drive `escalation.escalate_decision` end to end: `under_review` transition, an
    `EscalationRecord`, and (since this decision started `applied`) an actual
    held-credit release."""
    session = _session()
    _account(session, "acct-1")
    _cluster_snapshot(session, "snap-1", ["acct-1"])
    decision = _decision_with_evidence(
        session,
        decision_id="dec-2",  # bucket 0 under FakeMode.NORMAL (see test_reviewer.py)
        account_id="acct-1",
        proposed_action=ProposedAction.HOLD_INCREMENTAL_CREDITS,
        status=DecisionStatus.APPLIED,
    )
    _held_claim(session, decision=decision, account_id="acct-1", held_amount=10)

    provider = FakeReviewerProvider(mode=FakeMode.NORMAL)
    outcome = review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )

    # R-6 ceiling: unanimous per-sample "reverse" is forced to "escalate" because
    # the ORIGINAL proposed action (hold_incremental_credits) exceeds light_friction.
    assert {s.effect for s in outcome.samples} == {"reverse"}
    assert outcome.overall_effect == "escalate"
    assert outcome.applied is False

    assert decision.status is DecisionStatus.UNDER_REVIEW
    records = session.scalars(
        select(EscalationRecord).where(EscalationRecord.decision_id == decision.decision_id)
    ).all()
    assert len(records) == 1
    assert records[0].source == "llm_review"

    claim = session.scalars(select(SyntheticClaim)).one()
    assert claim.hold_state == "none"
    assert claim.held_amount == 0
    assert claim.granted_amount == 10

    credit_events = session.scalars(
        select(AuditEvent).where(AuditEvent.event_name == "credits_restored")
    ).all()
    assert len(credit_events) == 1


def test_review_decision_escalate_outcome_does_not_reescalate_on_rerun() -> None:
    """A second call on an already-escalated decision must not call
    `escalate_decision` again. Since escalation moves the decision to
    `under_review`, `_gating_reason`'s own `escalated_under_review` check (not the
    idempotency-key replay path) is what stops the second call — confirmed
    explicitly here, including with a DIFFERENT `idempotency_key` (so it is really
    the status gate doing the work, not idempotency-key replay), and that there is
    still exactly one `EscalationRecord`."""
    session = _session()
    _account(session, "acct-1")
    _cluster_snapshot(session, "snap-1", ["acct-1"])
    decision = _decision_with_evidence(
        session,
        decision_id="dec-2",
        account_id="acct-1",
        proposed_action=ProposedAction.HOLD_INCREMENTAL_CREDITS,
        status=DecisionStatus.APPLIED,
    )
    _held_claim(session, decision=decision, account_id="acct-1", held_amount=10)

    provider = FakeReviewerProvider(mode=FakeMode.NORMAL)
    first = review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )
    assert first.overall_effect == "escalate"
    assert decision.status is DecisionStatus.UNDER_REVIEW

    second = review_decision(
        session, decision, provider=provider, run_id="r2", idempotency_key="k2", as_of=AS_OF
    )
    assert second.gated is True
    assert second.gate_reason == "escalated_under_review"
    assert decision.status is DecisionStatus.UNDER_REVIEW

    records = session.scalars(
        select(EscalationRecord).where(EscalationRecord.decision_id == decision.decision_id)
    ).all()
    assert len(records) == 1, "the status gate must prevent a second escalation attempt"


# --- integration: nightly reconciliation skips an escalated decision -----------


def _seed_min_world(inference_session: Session, gt_session: Session) -> None:
    world = generate_world(1, "min")
    persist_world(world, inference_session, make_ground_truth_sink(gt_session))
    inference_session.flush()


def test_nightly_reconciliation_skips_escalated_decision(
    inference_session, gt_session, monkeypatch
) -> None:
    """Construct the config-change reconciliation scenario (test_detector.py's
    `test_config_change_to_allow_supersedes_stale_decision`): escalate ONE of the
    stale decisions first, then confirm reconciliation skips exactly that one while
    still withdrawing an untouched sibling."""
    from app.core.decision_status import SUPERSEDABLE
    from app.services import features as F

    _seed_min_world(inference_session, gt_session)
    first = run_detector(inference_session, as_of=WORLD_END, run_id="run-1")
    assert first.decisions_created > 0

    stale = list(
        inference_session.scalars(select(Decision).where(Decision.status.in_(SUPERSEDABLE))).all()
    )
    assert len(stale) >= 2, "need at least 2 active decisions to contrast escalated vs. not"

    escalated = stale[0]
    control = stale[1]
    escalate_decision(inference_session, escalated, source="llm_review", as_of=WORLD_END)
    assert escalated.status is DecisionStatus.UNDER_REVIEW

    # Isolate reconciliation from the separately tested Gate A manifest guard: pin the
    # verified hash BEFORE zeroing the weights, or the detector trips ManifestDrift.
    from app.services import detector

    verified = detector.verified_manifest_hash()
    monkeypatch.setattr(detector, "verified_manifest_hash", lambda: verified)

    weight_constants = (
        "_W_DEV_1", "_W_DEV_2", "_W_PAY_1", "_W_CON_1", "_W_DOM_1", "_W_VER_1",
        "_W_CYC_1", "_W_CYC_2", "_W_CYC_3", "_W_CYC_4", "_W_DOM_2", "_W_AGE_1",
    )
    for name in weight_constants:
        monkeypatch.setattr(F, name, 0.0)

    second = run_detector(
        inference_session, as_of=WORLD_END, run_id="run-2", detector_version="dev-detector-v2"
    )

    inference_session.refresh(escalated)
    inference_session.refresh(control)

    assert escalated.status is DecisionStatus.UNDER_REVIEW  # skipped, not resuperseded
    assert control.status is DecisionStatus.SUPERSEDED  # normal reconciliation still applies
    assert second.decisions_withdrawn == len(stale) - 1

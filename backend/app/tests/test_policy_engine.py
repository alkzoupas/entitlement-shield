"""Policy engine — ladder, gates, targeting, reversal (docs/policy.md §6–§10, Step 5).

Covers: threshold→action mapping; the ≥2-family, ambiguous, and large_shared caps
each hold; timing alone never actions above light_friction; enforcement targeting
restricts only over-allowance claims in timestamp order (early bystander untouched);
no permanent suspension at any level; suppression prevents nightly re-flag.
"""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy.orm import Session

from app.db.decisions import record_decision, transition_decision
from app.models.enums import (
    ActorType,
    CapacityBand,
    DecisionStatus,
    ProposedAction,
)
from app.services.banding import BandingResult
from app.services.features import EvidenceFamily
from app.services.policy_engine import (
    SUPPRESSION_DAYS,
    decide,
    evidence_signature,
    is_suppressed,
    record_reversal,
    select_enforcement_targets,
    should_flag,
)
from app.services.risk_scorer import RiskAssessment
from app.tests.support import AS_OF, Builder


def _assessment(
    *, above: bool, families: set[EvidenceFamily], score: float = 0.75
) -> RiskAssessment:
    return RiskAssessment(
        risk_score=score,
        above_threshold=above,
        firing_families=frozenset(families),
        contributions=(),
    )


def _band(
    *,
    allowance: int = 25,
    claimed: int,
    band: CapacityBand = CapacityBand.SINGLE,
    ambiguous: bool = False,
) -> BandingResult:
    return BandingResult(
        qualified_stream_count=1,
        allowance=allowance,
        band=band,
        ambiguous=ambiguous,
        claimed=claimed,
        concurrency_passers=(),
        rate_passers=(),
    )


_TWO = {EvidenceFamily.DEVICE, EvidenceFamily.CONTACT_DOMAIN}


# --- Ladder + threshold ------------------------------------------------------


def test_below_threshold_is_allow() -> None:
    d = decide(_assessment(above=False, families=_TWO), _band(claimed=100), claimed=100)
    assert d.proposed_action is ProposedAction.ALLOW


def test_excess_bands_map_to_ladder() -> None:
    cases = {
        25: ProposedAction.ALLOW,  # 0% excess
        30: ProposedAction.LIGHT_FRICTION,  # +20%
        45: ProposedAction.HOLD_INCREMENTAL_CREDITS,  # +80%
        60: ProposedAction.TEMPORARY_HOLD,  # +140%
    }
    for claimed, expected in cases.items():
        d = decide(_assessment(above=True, families=_TWO), _band(claimed=claimed), claimed=claimed)
        assert d.proposed_action is expected, claimed
        assert not d.capped


# --- Guardrail caps ----------------------------------------------------------


def test_insufficient_families_caps_at_light_friction() -> None:
    d = decide(
        _assessment(above=True, families={EvidenceFamily.DEVICE}),  # one family
        _band(claimed=60),  # would be temporary_hold
        claimed=60,
    )
    assert d.proposed_action is ProposedAction.LIGHT_FRICTION
    assert d.capped and "insufficient_families" in d.cap_reasons


def test_timing_alone_never_actions_above_light_friction() -> None:
    """Family-4-only firing is capped — the timing safety property (features.md §4.2)."""
    d = decide(
        _assessment(above=True, families={EvidenceFamily.CYCLE}),
        _band(claimed=100),  # huge excess
        claimed=100,
    )
    assert d.proposed_action is ProposedAction.LIGHT_FRICTION
    assert "insufficient_families" in d.cap_reasons


def test_ambiguous_caps_at_light_friction() -> None:
    d = decide(
        _assessment(above=True, families=_TWO),
        _band(claimed=60, ambiguous=True),
        claimed=60,
    )
    assert d.proposed_action is ProposedAction.LIGHT_FRICTION
    assert "ambiguous" in d.cap_reasons


def test_large_shared_caps_at_light_friction() -> None:
    d = decide(
        _assessment(above=True, families=_TWO),
        _band(claimed=200, allowance=150, band=CapacityBand.LARGE_SHARED),
        claimed=200,
    )
    assert d.proposed_action is ProposedAction.LIGHT_FRICTION
    assert "large_shared" in d.cap_reasons


def test_no_permanent_suspension_at_any_level() -> None:
    """The ladder tops out at temporary_hold — structurally no permanent suspension,
    even at unbounded excess."""
    ladder = {
        ProposedAction.ALLOW,
        ProposedAction.LIGHT_FRICTION,
        ProposedAction.HOLD_INCREMENTAL_CREDITS,
        ProposedAction.TEMPORARY_HOLD,
    }
    for claimed in (26, 51, 100, 100000):
        d = decide(_assessment(above=True, families=_TWO), _band(claimed=claimed), claimed=claimed)
        assert d.proposed_action in ladder
    top = decide(_assessment(above=True, families=_TWO), _band(claimed=100000), claimed=100000)
    assert top.proposed_action is ProposedAction.TEMPORARY_HOLD  # the ceiling


# --- Enforcement targeting (policy §8) ---------------------------------------


def test_enforcement_targets_only_over_allowance_in_order(inference_session: Session) -> None:
    b = Builder(inference_session)
    a1, a2, a3 = b.account("early"), b.account("mid"), b.account("late")
    b.claim(a1, AS_OF - timedelta(days=9), 25)  # running 25 == allowance → untouched
    b.claim(a2, AS_OF - timedelta(days=8), 25)  # running 50 > 25 → target
    b.claim(a3, AS_OF - timedelta(days=7), 25)  # running 75 > 25 → target
    b.flush()

    targets = select_enforcement_targets(
        inference_session, [a1, a2, a3], 25, AS_OF - timedelta(days=30), AS_OF
    )
    assert targets == [a2, a3]
    assert a1 not in targets  # the early bystander is never restricted


# --- Reversal + suppression (policy §9) --------------------------------------


def _applied_decision(session: Session) -> tuple:
    b = Builder(session)
    aid = b.account("rev-acct")
    b.flush()
    decision = record_decision(
        session,
        account_id=aid,
        risk_score=0.8,
        proposed_action=ProposedAction.HOLD_INCREMENTAL_CREDITS,
        detector_version="dev",
        policy_version="v1",
        run_id="run-1",
        idempotency_key="idem-1",
        cluster_snapshot_id=None,
    )
    transition_decision(session, decision, DecisionStatus.APPLIED, actor=ActorType.SYSTEM)
    return aid, decision


_EDGES = [("acc-a", "acc-b", 1.0, 1, "2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00")]


def test_reversal_suppresses_reflag_then_expires(inference_session: Session) -> None:
    aid, decision = _applied_decision(inference_session)
    members = [aid, "sibling"]

    sig = record_reversal(
        inference_session,
        decision,
        member_account_ids=members,
        edges=_EDGES,
        excess=1.5,
        as_of=AS_OF,
    )
    assert decision.status is DecisionStatus.REVERSED

    # Within the window: identical evidence must not re-flag; created_at <= as_of holds.
    assert is_suppressed(inference_session, sig, AS_OF + timedelta(days=30)) is True
    assert should_flag(inference_session, sig, AS_OF + timedelta(days=30)) is False
    # Before the reversal was created: the historical run must NOT see it (finding 11).
    assert is_suppressed(inference_session, sig, AS_OF - timedelta(days=1)) is False
    # After 60 days: suppression has lapsed.
    lapsed = AS_OF + timedelta(days=SUPPRESSION_DAYS + 1)
    assert is_suppressed(inference_session, sig, lapsed) is False


def test_materially_new_evidence_voids_suppression(inference_session: Session) -> None:
    aid, decision = _applied_decision(inference_session)
    members = [aid, "sibling"]
    record_reversal(
        inference_session,
        decision,
        member_account_ids=members,
        edges=_EDGES,
        excess=1.5,
        as_of=AS_OF,
    )
    at = AS_OF + timedelta(days=10)
    stronger_edge = [
        ("acc-a", "acc-b", 1.0, 2, "2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00")
    ]
    stronger_weight = [
        ("acc-a", "acc-b", 0.6, 1, "2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00")
    ]
    # A new account, or any canonical edge field / material excess change voids it.
    assert should_flag(inference_session, evidence_signature([*members, "new"], _EDGES, 1.5), at)
    assert should_flag(inference_session, evidence_signature(members, stronger_edge, 1.5), at)
    assert should_flag(inference_session, evidence_signature(members, stronger_weight, 1.5), at)
    assert should_flag(inference_session, evidence_signature(members, _EDGES, 3.0), at)
    # Identical evidence stays suppressed.
    assert not should_flag(inference_session, evidence_signature(members, _EDGES, 1.5), at)

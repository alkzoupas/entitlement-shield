"""Phase 4 P4.1a spike (docs/decisions.md §13 M-2) — NOT a permanent regression
test yet; a mechanism probe to settle the exact detector-call sequencing the
world-change fixture (P4.1) needs. Kept out of `make test`'s pinned-count
expectations deliberately.

Question: does an intervening `run_detector` call between "apply the cycle-1
precedent" and "score cycle 2" supersede the precedent via `_supersede_unaffirmed`
before F-CYC-4 (`features._f_cyc_4`) can see it? Or does scoring-before-supersede
ordering within one call mean a clean 2-call design already works, provided there
is no wasted intervening call?
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.enums import ActorType, DecisionStatus, EmailDomainType, VerificationLevel
from app.models.inference import Decision
from app.services import human_review
from app.services.detector import run_detector
from app.services.evidence import decision_evidence
from app.tests.support import Builder

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _cyc4_contribution(session: Session, decision_id: str) -> float:
    ctx = decision_evidence(session, decision_id)
    assert ctx is not None
    for row in ctx["contributions"]:
        if row["feature_id"] == "F-CYC-4":
            return row["contribution"]
    raise AssertionError("F-CYC-4 not present in contributions")


def _singleton_cycle(
    builder: Builder, account_id: str, start: datetime, *, claims: int, consumed: int
) -> None:
    """`claims` is the number of 25-credit CLAIM_CREDITS events within the
    window — excess is computed off summed *claims*, not consumption
    (`detector._cycle_claimed`)."""
    builder.account(
        account_id,
        created=start - timedelta(days=10),
        domain=EmailDomainType.DISPOSABLE,
        verification=VerificationLevel.NONE,
    )  # fires F-DOM-1 + F-VER-1 so total risk_score clears the 0.60 gate
    for i in range(claims):
        builder.claim(account_id, start + timedelta(hours=i))
    builder.consume(account_id, start + timedelta(days=1), consumed)
    builder.cycle(account_id, start, granted=25, consumed=consumed)
    builder.flush()


def test_two_call_design_lets_fcyc4_see_the_precedent(inference_session: Session) -> None:
    """Call 1: apply cycle-1's precedent. Call 2: score cycle-2 directly (no
    intervening call). Prediction: since `run_detector`'s scoring loop
    (detector.py, records decisions) runs entirely BEFORE `_supersede_unaffirmed`
    (called once, at the end, per call), the cycle-1 APPLIED decision is still
    APPLIED at the moment cycle-2's unit is scored within call 2 — even though
    that same call 2 will supersede it moments later, after scoring completes.
    """
    b = Builder(inference_session)
    acct = "acct-p41a-01"
    cycle1_start = T0
    # excess far above light_friction
    _singleton_cycle(b, acct, cycle1_start, claims=8, consumed=200)

    # The real fixture design (M-3): the monkeypatch pins wall-clock `now_utc()`
    # to the simulated `as_of` being scored, not the actual host date — real
    # "now" is 2026-09+ regardless of the synthetic world's frame, and F-CYC-4's
    # `created_at < as_of` check can never pass against unpatched wall-clock.
    as_of_1 = cycle1_start + timedelta(days=31)
    real_now = as_of_1
    with patch("app.services.detector.now_utc", return_value=real_now), \
         patch("app.services.human_review.now_utc", return_value=real_now), \
         patch("app.services.policy_engine.now_utc", return_value=real_now), \
         patch("app.db.decisions.now_utc", return_value=real_now):
        result1 = run_detector(inference_session, as_of=as_of_1, run_id="p41a-r1")
        assert result1.decisions_created >= 1
        cycle1_decision = inference_session.scalars(
            select(Decision).where(Decision.account_id == acct)
        ).one()
        human_review.record_human_decision(
            inference_session, cycle1_decision, action="apply",
            actor=ActorType.ANALYST, as_of=cycle1_start + timedelta(days=31),
        )
        inference_session.flush()
        assert cycle1_decision.status == DecisionStatus.APPLIED
        precedent_created_at = cycle1_decision.created_at
        if precedent_created_at.tzinfo is None:
            precedent_created_at = precedent_created_at.replace(tzinfo=UTC)
        assert precedent_created_at < cycle1_start + timedelta(days=62), (
            "now_utc patch must make Decision.created_at (real-clock-stamped at "
            "creation) precede the later simulated as_of, or F-CYC-4's "
            "created_at < as_of check can never pass"
        )

        # Cycle 2, immediately, ready to score in this SAME call — no wasted
        # intervening run_detector call in between. 1 claim (25) is within
        # allowance; add a small second claim to be safely actionable.
        cycle2_start = cycle1_start + timedelta(days=31)
        b.claim(acct, cycle2_start)
        b.claim(acct, cycle2_start + timedelta(hours=1), amount=7)
        b.consume(acct, cycle2_start + timedelta(days=1), 5)
        b.cycle(acct, cycle2_start, granted=25, consumed=5)
        b.flush()

        as_of_2 = cycle2_start + timedelta(days=31)
        run_detector(inference_session, as_of=as_of_2, run_id="p41a-r2")

    # Find the cycle-2 decision (a new idempotency key, since snapshot content
    # hash embeds as_of) and confirm F-CYC-4 fired.
    cycle2_decisions = list(
        inference_session.scalars(
            select(Decision)
            .where(Decision.account_id == acct, Decision.run_id == "p41a-r2")
        )
    )
    assert cycle2_decisions, "cycle 2 must itself produce a decision to check F-CYC-4 against"
    contribution = _cyc4_contribution(inference_session, cycle2_decisions[0].decision_id)
    assert contribution > 0, (
        "F-CYC-4 did not fire in a clean 2-call design — the mechanism "
        "hypothesis (scoring precedes supersede within one call) is wrong"
    )

    # Confirm the OLD cycle-1 precedent WAS superseded by this same call 2
    # (proving supersession did happen, just too late to matter for F-CYC-4).
    inference_session.refresh(cycle1_decision)
    assert cycle1_decision.status == DecisionStatus.SUPERSEDED


def test_intervening_noop_call_supersedes_precedent_before_cycle2_exists(
    inference_session: Session,
) -> None:
    """An intervening `run_detector` call with nothing new to affirm for this
    account (cycle 2 not yet ready) supersedes the precedent immediately — this
    is the failure mode a 3rd, wasted call would trigger, and confirms the
    fixture/harness sequencing (P4.2) must never insert such a call.
    """
    b = Builder(inference_session)
    acct = "acct-p41a-02"
    cycle1_start = T0
    _singleton_cycle(b, acct, cycle1_start, claims=8, consumed=200)

    real_now = datetime(2026, 9, 11, tzinfo=UTC)
    with patch("app.services.detector.now_utc", return_value=real_now), \
         patch("app.services.human_review.now_utc", return_value=real_now), \
         patch("app.services.policy_engine.now_utc", return_value=real_now), \
         patch("app.db.decisions.now_utc", return_value=real_now):
        run_detector(inference_session, as_of=cycle1_start + timedelta(days=31), run_id="p41a-r3")
        cycle1_decision = inference_session.scalars(
            select(Decision).where(Decision.account_id == acct)
        ).one()
        human_review.record_human_decision(
            inference_session, cycle1_decision, action="apply",
            actor=ActorType.ANALYST, as_of=cycle1_start + timedelta(days=31),
        )
        inference_session.flush()
        assert cycle1_decision.status == DecisionStatus.APPLIED

        # Intervening no-op call: as_of advances a little, but cycle 2 doesn't
        # exist yet — nothing new is affirmed for this account.
        run_detector(
            inference_session,
            as_of=cycle1_start + timedelta(days=40),
            run_id="p41a-r4",
        )

    inference_session.refresh(cycle1_decision)
    assert cycle1_decision.status == DecisionStatus.SUPERSEDED, (
        "expected the intervening no-op call to supersede the precedent early — "
        "if this fails, the 2-call design may tolerate an intervening call too"
    )

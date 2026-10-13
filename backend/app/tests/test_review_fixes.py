"""Regression tests for the Steps 4-6 adversarial review (docs/steps-4-6-review-handoff.md).

Each test pins a finding's fix so it cannot silently regress. Numbering follows the
review's P1/P2 findings.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.clock import now_utc
from app.db.decisions import (
    IdempotencyConflict,
    record_decision,
    record_decision_idempotent,
    transition_decision,
)
from app.db.ground_truth_writer import make_ground_truth_sink
from app.models.enums import (
    ActorType,
    DecisionStatus,
    EmailDomainType,
    EventType,
    ProposedAction,
    Reliability,
    SignalType,
)
from app.models.inference import ClusterSnapshot, Decision
from app.services.audit import append_audit
from app.services.detector import config_fingerprint, detection_bytes, run_detector
from app.services.features import (
    FEATURE_ORDER,
    Rung,
    account_local_features,
    score_cluster,
)
from app.services.generator import WORLD_END, generate_world, persist_world
from app.services.policy_engine import record_reversal
from app.tests.support import AS_OF, Builder

_DAY = timedelta(days=1)


def _fid(score, feature_id):
    return next(c for c in score.contributions if c.feature_id == feature_id)


# --- Finding 3: account metadata must not leak across as_of ------------------


def test_future_account_does_not_leak(inference_session: Session) -> None:
    b = Builder(inference_session)
    future = b.account("future", domain=EmailDomainType.DISPOSABLE, created=AS_OF + 10 * _DAY)
    b.flush()
    # The account does not exist at as_of → F-DOM-1 cannot fire, and account-local
    # scoring rejects it consistently.
    score = score_cluster(inference_session, [future], AS_OF, Rung.S0)
    assert not _fid(score, "F-DOM-1").fired
    with pytest.raises(KeyError):
        account_local_features(inference_session, future, AS_OF)


# --- Finding 4: resource-sharing semantics match clustering ------------------


def test_global_hub_and_effective_reliability(inference_session: Session) -> None:
    b = Builder(inference_session)
    members = [b.account(f"m-{i}") for i in range(3)]
    # (a) Global hub: 3 members + 22 other accounts all on ONE device value → global
    # cardinality > 20 → downgraded to low → F-DEV-1 (needs HIGH) must NOT fire.
    for aid in members:
        b.observation(aid, SignalType.STABLE_DEVICE, "hub-dev", reliability=Reliability.HIGH)
    for i in range(22):
        other = b.account(f"other-{i}")
        b.observation(other, SignalType.STABLE_DEVICE, "hub-dev", reliability=Reliability.HIGH)
    b.flush()
    assert not _fid(score_cluster(inference_session, members, AS_OF, Rung.S1), "F-DEV-1").fired

    # (b) Mixed reliability: effective = weaker(min), never promoted to HIGH.
    mixed = [b.account(f"mix-{i}") for i in range(3)]
    b.observation(mixed[0], SignalType.STABLE_DEVICE, "mix-dev", reliability=Reliability.HIGH)
    b.observation(mixed[1], SignalType.STABLE_DEVICE, "mix-dev", reliability=Reliability.HIGH)
    b.observation(mixed[2], SignalType.STABLE_DEVICE, "mix-dev", reliability=Reliability.LOW)
    b.flush()
    assert not _fid(score_cluster(inference_session, mixed, AS_OF, Rung.S1), "F-DEV-1").fired


# --- Finding 6: S0 reduction uses final risk, not incriminating sum ----------


def test_s0_reduction_picks_highest_final_risk(inference_session: Session) -> None:
    b = Builder(inference_session)
    # Both disposable (F-DOM-1 +0.25, frozen v1). `a-benign` sorts first AND cancels to
    # 0 via the benign F-AGE-1 (aged + routine logins); `a-high` keeps its full 0.25. The
    # locked any-restriction operator must pick `a-high` (max FINAL risk), not the first id.
    benign = b.account("a-benign", domain=EmailDomainType.DISPOSABLE, created=AS_OF - 200 * _DAY)
    for k in range(2):
        b.event(benign, AS_OF - (30 + k) * _DAY, EventType.LOGIN)
    b.account("a-high", domain=EmailDomainType.DISPOSABLE, created=AS_OF - 200 * _DAY)
    b.flush()
    score = score_cluster(inference_session, ["a-benign", "a-high"], AS_OF, Rung.S0)
    assert _fid(score, "F-DOM-1").fired
    assert not _fid(score, "F-AGE-1").fired  # the benign account was NOT the representative
    assert abs(score.total_contribution - 0.25) < 1e-9


# --- Finding 8: config change supersedes; reused key + new payload conflicts --


def test_idempotency_key_conflict_and_config_fingerprint(inference_session: Session) -> None:
    b = Builder(inference_session)
    aid = b.account("acct")
    b.flush()
    kw = dict(
        account_id=aid,
        proposed_action=ProposedAction.TEMPORARY_HOLD,
        detector_version="d1",
        policy_version="p1",
        run_id="r1",
        cluster_snapshot_id=None,
    )
    d1, made1 = record_decision_idempotent(
        inference_session, risk_score=0.8, idempotency_key="k", **kw
    )
    d2, made2 = record_decision_idempotent(
        inference_session, risk_score=0.8, idempotency_key="k", **kw
    )
    assert made1 and not made2 and d1.decision_id == d2.decision_id  # same payload → no-op
    with pytest.raises(IdempotencyConflict):  # same key, different risk → conflict
        record_decision_idempotent(inference_session, risk_score=0.9, idempotency_key="k", **kw)
    # A config change is a different fingerprint → a different key.
    assert config_fingerprint("d1", "p1", Rung.S2) != config_fingerprint("d2", "p1", Rung.S2)


# --- Finding 12: F-AGE-1 measures age at the cycle boundary ------------------


def test_age_measured_at_cycle_not_detector_time(inference_session: Session) -> None:
    b = Builder(inference_session)
    window_start = AS_OF - 30 * _DAY
    aid = b.account("young-at-cycle", created=window_start - 80 * _DAY)  # 80 d at cycle
    for k in range(2):
        b.event(aid, window_start - (5 + k) * _DAY, EventType.LOGIN)
    b.flush()
    # At the cycle boundary the account is 80 d old (< 90) → no benign age credit.
    at_cycle = score_cluster(inference_session, [aid], AS_OF, Rung.S0, age_reference=window_start)
    assert not _fid(at_cycle, "F-AGE-1").fired
    # Measured at detector time it would be 110 d old — the bug the fix removes.
    at_detector = score_cluster(inference_session, [aid], AS_OF, Rung.S0, age_reference=AS_OF)
    assert _fid(at_detector, "F-AGE-1").fired


# --- Finding 13: lower rungs never expose S2 capacity ------------------------


def test_lower_rungs_get_neutral_banding(inference_session: Session) -> None:
    b = Builder(inference_session)
    members = [b.account(f"cap-{i}") for i in range(3)]
    for i, aid in enumerate(members):
        for d in (7, 14, 21):
            b.consume(aid, (AS_OF - d * _DAY).replace(hour=8 + i * 5), 3)  # diverse → real streams
    b.flush()
    s1 = score_cluster(inference_session, members, AS_OF, Rung.S1)
    s2 = score_cluster(inference_session, members, AS_OF, Rung.S2)
    assert s1.banding.qualified_stream_count == 0 and s1.banding.allowance == 25  # neutral
    assert s2.banding.qualified_stream_count >= 1  # S2 actually estimates capacity
    # A masked (below-rung) feature exports no raw timing value.
    assert _fid(s1, "F-CYC-1").raw_value == 0.0


# --- Finding 16: the audit writer validates at its boundary ------------------


def test_audit_writer_rejects_unregistered_event(inference_session: Session) -> None:
    class _Bogus(BaseModel):
        event_name: str = "not_a_real_event"
        entity_type: str = "x"
        entity_id: str = "y"
        actor_type: ActorType = ActorType.SYSTEM

    with pytest.raises(ValidationError):  # rejected by the union parse at the boundary
        append_audit(inference_session, _Bogus())


# --- Finding 5: F-CYC-4 recognizes repeat behavior across snapshots ----------


def test_f_cyc_4_matches_across_new_snapshot(inference_session: Session) -> None:
    b = Builder(inference_session)
    a1, a2 = b.account("rep-1"), b.account("rep-2")
    b.flush()
    inference_session.add(
        ClusterSnapshot(
            snapshot_id="snap-prior",
            as_of=AS_OF - 40 * _DAY,
            member_account_ids=[a1, a2],
            edges=[],
            evidence_summary={},
            content_hash="prior-hash",
            created_at=AS_OF - 40 * _DAY,
        )
    )
    inference_session.flush()
    prior = record_decision(
        inference_session,
        account_id=a1,
        risk_score=0.8,
        proposed_action=ProposedAction.TEMPORARY_HOLD,
        detector_version="d",
        policy_version="p",
        run_id="r",
        idempotency_key="rep-key",
        cluster_snapshot_id="snap-prior",
    )
    transition_decision(inference_session, prior, DecisionStatus.APPLIED, actor=ActorType.SYSTEM)
    # Score in a later cycle (after the prior decision was recorded) with a NEW snapshot
    # id — F-CYC-4 must still match on the as_of-independent member set (finding 5).
    later = now_utc() + _DAY
    score = score_cluster(inference_session, [a1, a2], later, Rung.S2, snapshot_id="snap-new")
    assert _fid(score, "F-CYC-4").fired
    # A reversal is a finding of legitimacy → no repeat penalty.
    transition_decision(inference_session, prior, DecisionStatus.REVERSED, actor=ActorType.ANALYST)
    score2 = score_cluster(inference_session, [a1, a2], later, Rung.S2, snapshot_id="snap-new")
    assert not _fid(score2, "F-CYC-4").fired


# --- Findings 1/9/10/17: detector suppression, artifact, reconstruction ------


def test_detector_artifact_is_byte_stable_and_complete(
    inference_session: Session, gt_session: Session
) -> None:
    persist_world(generate_world(1, "min"), inference_session, make_ground_truth_sink(gt_session))
    inference_session.flush()
    r1 = run_detector(inference_session, as_of=WORLD_END, run_id="run-1")
    assert r1.detections  # every unit produces a detection record
    # Finding 17: an auditor can reconstruct the full ordered contribution set.
    for det in r1.detections:
        assert [c.feature_id for c in det.contributions] == list(FEATURE_ORDER)
    # Finding 9: byte-stable across identical runs.
    r2 = run_detector(inference_session, as_of=WORLD_END, run_id="run-2")
    assert detection_bytes(r1.detections) == detection_bytes(r2.detections)


def test_reversed_cluster_is_not_reflagged(
    inference_session: Session, gt_session: Session
) -> None:
    """Finding 1: after a reversal, a re-run — even under a new config that mints new
    idempotency keys — does not re-flag the identical evidence until it changes."""
    persist_world(generate_world(1, "min"), inference_session, make_ground_truth_sink(gt_session))
    inference_session.flush()
    result = run_detector(inference_session, as_of=WORLD_END, run_id="run-1")
    actioned = next(d for d in result.detections if d.proposed_action != "allow")

    # Reverse one decision for that unit, using its own evidence (finding 10 also:
    # restoration is attempted for the current cycle).
    decision = inference_session.scalars(
        select(Decision).where(
            Decision.cluster_snapshot_id == actioned.cluster_snapshot_id,
            Decision.status == DecisionStatus.PROPOSED,  # a live (non-superseded) row
        )
    ).first()
    # Reversal is only legal once a decision is applied (the analyst path).
    transition_decision(inference_session, decision, DecisionStatus.APPLIED, actor=ActorType.SYSTEM)
    record_reversal(
        inference_session,
        decision,
        member_account_ids=list(actioned.member_account_ids),
        edges=list(actioned.edges),  # the exact evidence the detector signed
        excess=actioned.excess,
        as_of=WORLD_END,
    )

    # Re-run with a DIFFERENT detector version → new fingerprint → new keys. Without
    # suppression this would re-flag; with it, the reversed unit is skipped.
    rerun = run_detector(
        inference_session, as_of=WORLD_END, run_id="run-2", detector_version="dev-detector-v2"
    )
    assert rerun.suppressed_units >= 1
    # The exact reversed unit (same snapshot + cycle) is suppressed and not re-flagged.
    same_unit = next(
        d for d in rerun.detections
        if d.cluster_snapshot_id == actioned.cluster_snapshot_id
        and d.cycle_id == actioned.cycle_id
    )
    assert same_unit.suppressed is True

"""Frozen feature set + rung masking (docs/features.md §3/§5, plan Step 4 gate).

Covers: per-signal feature firing and families; behavioral_template drives no
feature; account-local S0 features fire with no cluster and reduce by
any-restriction; rung masking yields the correct S0/S1/S2 subsets on the
identical fixed unit (same ordered keys); contributions sum; features respect
`as_of`.
"""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy.orm import Session

from app.models.enums import (
    EmailDomainType,
    EventType,
    Reliability,
    SignalType,
    VerificationLevel,
)
from app.services.features import (
    EvidenceFamily,
    Rung,
    account_local_features,
    enumerate_cluster_cycles,
    score_cluster,
    score_cluster_cycle,
)
from app.tests.support import AS_OF, Builder

_D1 = AS_OF - timedelta(days=21)
_D2 = AS_OF - timedelta(days=14)
_D3 = AS_OF - timedelta(days=7)
_SHARED_DAYS = (_D1, _D2, _D3)


def _at(day, hour: int):
    return day.replace(hour=hour, minute=0, second=0, microsecond=0)


def _fired(score) -> set[str]:
    return {c.feature_id for c in score.contributions if c.fired}


def _by_id(score, feature_id: str):
    return next(c for c in score.contributions if c.feature_id == feature_id)


# --- Resource-sharing edge features (S1) -------------------------------------


def test_f_dev_1_strong_shared_device(inference_session: Session) -> None:
    b = Builder(inference_session)
    members = [b.account(f"d-{i}") for i in range(3)]
    for aid in members:
        b.observation(aid, SignalType.STABLE_DEVICE, "dev-x", reliability=Reliability.HIGH)
    b.flush()

    score = score_cluster(inference_session, members, AS_OF, Rung.S1)
    dev1 = _by_id(score, "F-DEV-1")
    assert dev1.fired and dev1.contribution == 0.45  # frozen v1 weight (Step 8)
    assert dev1.family is EvidenceFamily.DEVICE and dev1.rung is Rung.S1
    assert not _by_id(score, "F-DEV-2").fired  # strong share, not weak


def test_f_dev_2_weak_and_pay_con(inference_session: Session) -> None:
    b = Builder(inference_session)
    a0, a1 = b.account("w-0"), b.account("w-1")
    b.observation(a0, SignalType.STABLE_DEVICE, "dev-y", reliability=Reliability.HIGH)
    b.observation(a1, SignalType.STABLE_DEVICE, "dev-y", reliability=Reliability.HIGH)
    b.observation(a0, SignalType.PAYMENT_TOKEN, "pay-1")
    b.observation(a1, SignalType.PAYMENT_TOKEN, "pay-1")
    b.observation(a0, SignalType.RECOVERY_CONTACT, "rc-1", reliability=Reliability.MEDIUM)
    b.observation(a1, SignalType.RECOVERY_CONTACT, "rc-1", reliability=Reliability.MEDIUM)
    b.flush()

    score = score_cluster(inference_session, [a0, a1], AS_OF, Rung.S1)
    assert not _by_id(score, "F-DEV-1").fired  # only two accounts → weak
    assert _by_id(score, "F-DEV-2").fired
    pay = _by_id(score, "F-PAY-1")
    assert pay.fired and pay.family is EvidenceFamily.PAYMENT
    assert _by_id(score, "F-CON-1").fired


def test_behavioral_template_drives_no_feature(inference_session: Session) -> None:
    """Context-only signals form no edge and drive no feature (features.md §3.3)."""
    b = Builder(inference_session)
    members = [b.account(f"ctx-{i}") for i in range(3)]
    for aid in members:
        b.observation(aid, SignalType.BEHAVIORAL_TEMPLATE, "tmpl-1")
        b.observation(aid, SignalType.NETWORK_GROUP, "nat-1")
    b.flush()

    score = score_cluster(inference_session, members, AS_OF, Rung.S2)
    assert _fired(score) == set()  # nothing resource-sharing → no feature fires


# --- Account-local metadata features (S0) ------------------------------------


def test_f_dom_1_account_local_no_cluster(inference_session: Session) -> None:
    b = Builder(inference_session)
    aid = b.account("disp", domain=EmailDomainType.DISPOSABLE)
    b.flush()

    feats = {c.feature_id: c for c in account_local_features(inference_session, aid, AS_OF)}
    assert feats["F-DOM-1"].fired
    assert feats["F-DOM-1"].family is EvidenceFamily.CONTACT_DOMAIN
    assert feats["F-DOM-1"].rung is Rung.S0


def test_f_ver_1_account_local_variants(inference_session: Session) -> None:
    b = Builder(inference_session)
    # (a) repeated verification failures
    fail = b.account("fail", verification=VerificationLevel.NONE)
    b.event(fail, AS_OF - timedelta(days=30), EventType.VERIFICATION_FAILED)
    b.event(fail, AS_OF - timedelta(days=29), EventType.VERIFICATION_FAILED)
    # (b) claim strictly before verification
    cbv = b.account("cbv")
    b.claim(cbv, AS_OF - timedelta(days=20))
    b.event(cbv, AS_OF - timedelta(days=10), EventType.VERIFICATION)
    # (c) clean: verify then claim → does NOT fire
    clean = b.account("clean")
    b.event(clean, AS_OF - timedelta(days=20), EventType.VERIFICATION)
    b.claim(clean, AS_OF - timedelta(days=10))
    b.flush()

    def ver(aid: str) -> bool:
        f = {c.feature_id: c for c in account_local_features(inference_session, aid, AS_OF)}
        return f["F-VER-1"].fired

    assert ver(fail) is True
    assert ver(cbv) is True
    assert ver(clean) is False


# --- The full stack: rung masking on the identical fixed unit -----------------


def _build_multi_signal_cluster(b: Builder) -> list[str]:
    """A cluster with an S0 signal (disposable + weak verification), an S1 signal
    (shared HIGH device), and an S2 signal (allowance excess from over-claiming)."""
    members = [b.account(f"farm-{i}", domain=EmailDomainType.DISPOSABLE,
                         verification=VerificationLevel.NONE) for i in range(3)]
    for aid in members:
        # S1: one strong shared device.
        b.observation(aid, SignalType.STABLE_DEVICE, "farm-dev", reliability=Reliability.HIGH)
        # S0: repeated verification failures.
        b.event(aid, AS_OF - timedelta(days=40), EventType.VERIFICATION_FAILED)
        b.event(aid, AS_OF - timedelta(days=39), EventType.VERIFICATION_FAILED)
        # Consume on identical shared days/hour → collapse to one stream (allowance 25).
        for day in _SHARED_DAYS:
            b.consume(aid, _at(day, 14), 3)
        # Over-claim: each claims a full 25 → cluster claimed 75 ≫ allowance 25.
        b.claim(aid, AS_OF - timedelta(days=25), 25)
        # Two aligned cycles for the fixed-unit enumeration.
        b.cycle(aid, AS_OF - timedelta(days=60))
        b.cycle(aid, AS_OF - timedelta(days=30))
    b.flush()
    return members


def test_rung_masking_subsets(inference_session: Session) -> None:
    b = Builder(inference_session)
    members = _build_multi_signal_cluster(b)

    s0 = _fired(score_cluster(inference_session, members, AS_OF, Rung.S0))
    s1 = _fired(score_cluster(inference_session, members, AS_OF, Rung.S1))
    s2 = _fired(score_cluster(inference_session, members, AS_OF, Rung.S2))

    # S0: only metadata features (F-DOM-1 disposable, F-VER-1 failures).
    assert s0 == {"F-DOM-1", "F-VER-1"}
    # S1 adds the resource-sharing device feature, keeps S0, no S2 timing yet.
    assert s1 == {"F-DOM-1", "F-VER-1", "F-DEV-1"}
    # S2 adds the allowance-excess timing feature.
    assert "F-CYC-1" in s2 and {"F-DOM-1", "F-VER-1", "F-DEV-1"} <= s2
    # Monotone ladder: each rung is a superset of the previous.
    assert s0 <= s1 <= s2


def test_rungs_emit_identical_ordered_unit_keys(inference_session: Session) -> None:
    """The paired-ablation requirement (review 4 P0-1): all three rungs enumerate
    and score the same ordered (snapshot, cycle_id, as_of) keys and feature order."""
    b = Builder(inference_session)
    members = _build_multi_signal_cluster(b)
    snapshot_id = "snap-test"
    units = enumerate_cluster_cycles(inference_session, snapshot_id, members, AS_OF)
    assert len(units) == 2  # two aligned cycle windows

    keys_by_rung = {}
    order_by_rung = {}
    for rung in (Rung.S0, Rung.S1, Rung.S2):
        keys, orders = [], []
        for unit in units:
            score = score_cluster_cycle(inference_session, unit, members, rung)
            keys.append((unit.cluster_snapshot_id, unit.cycle_id, unit.as_of))
            orders.append(tuple(c.feature_id for c in score.contributions))
        keys_by_rung[rung] = keys
        order_by_rung[rung] = orders

    assert keys_by_rung[Rung.S0] == keys_by_rung[Rung.S1] == keys_by_rung[Rung.S2]
    assert order_by_rung[Rung.S0] == order_by_rung[Rung.S1] == order_by_rung[Rung.S2]


def test_contributions_sum_and_families(inference_session: Session) -> None:
    b = Builder(inference_session)
    members = _build_multi_signal_cluster(b)
    score = score_cluster(inference_session, members, AS_OF, Rung.S2)

    expected = sum(c.contribution for c in score.contributions if c.fired)
    assert abs(score.total_contribution - max(0.0, min(1.0, expected))) < 1e-9
    # ≥ 2 distinct incriminating families fire (device, contact/domain, verification…).
    assert len(score.firing_families) >= 2


def test_s0_any_restriction_reduction(inference_session: Session) -> None:
    """One disposable account in an otherwise-clean cluster → the cluster-cycle S0
    row fires F-DOM-1 by the any-restriction reduction (features.md §5)."""
    b = Builder(inference_session)
    disp = b.account("one-disp", domain=EmailDomainType.DISPOSABLE)
    clean = [b.account(f"ok-{i}") for i in range(3)]
    for aid in clean:
        b.event(aid, AS_OF - timedelta(days=20), EventType.VERIFICATION)
    b.flush()

    score = score_cluster(inference_session, [disp, *clean], AS_OF, Rung.S0)
    assert _by_id(score, "F-DOM-1").fired  # the worst account drives the reduction


# --- Leakage control ----------------------------------------------------------


def test_features_respect_as_of(inference_session: Session) -> None:
    b = Builder(inference_session)
    members = [b.account(f"late-{i}") for i in range(3)]
    # Observations land AFTER the first as_of — must be invisible then, visible later.
    for aid in members:
        b.observation(
            aid, SignalType.STABLE_DEVICE, "late-dev",
            reliability=Reliability.HIGH, ts=AS_OF + timedelta(days=5),
        )
    b.flush()

    before = score_cluster(inference_session, members, AS_OF, Rung.S1)
    after = score_cluster(inference_session, members, AS_OF + timedelta(days=10), Rung.S1)
    assert not _by_id(before, "F-DEV-1").fired
    assert _by_id(after, "F-DEV-1").fired


def test_f_cyc_3_is_recent_and_uses_half_open_cycle_ownership(inference_session: Session) -> None:
    """Gate G: stale exhaustion and an event at cycle end cannot fire F-CYC-3."""
    b = Builder(inference_session)
    exhausted, signup = b.account("cyc-exhaust"), b.account("cyc-signup")
    b.cycle(exhausted, AS_OF - timedelta(days=90), granted=10)
    # Exactly at cycle end: excluded from the first cycle's ownership.
    b.consume(exhausted, AS_OF - timedelta(days=60), 10)
    b.event(signup, AS_OF - timedelta(days=59, hours=23), EventType.SIGNUP)
    b.flush()
    stale = score_cluster(inference_session, [exhausted, signup], AS_OF, Rung.S2)
    assert not _by_id(stale, "F-CYC-3").fired

    # A recent exhaustion followed by signup within 72 hours is retained.
    b.cycle(exhausted, AS_OF - timedelta(days=20), granted=10)
    b.consume(exhausted, AS_OF - timedelta(days=10), 10)
    b.event(signup, AS_OF - timedelta(days=9), EventType.SIGNUP)
    b.flush()
    recent = score_cluster(inference_session, [exhausted, signup], AS_OF, Rung.S2)
    assert _by_id(recent, "F-CYC-3").fired

"""Real synthetic grant accounting and bounded request arbitration regressions."""

from __future__ import annotations

import threading
import time
from datetime import timedelta

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.core.clock import now_utc
from app.db.engine import create_inference_schema, make_inference_engine
from app.db.sqlite_fk import enable_sqlite_foreign_keys
from app.models.claims import SyntheticClaim
from app.models.enums import (
    DecisionStatus,
    EmailDomainType,
    ProposedAction,
    VerificationLevel,
)
from app.models.inference import (
    Account,
    AccountEvent,
    AuditEvent,
    ClusterSnapshot,
    Decision,
    EntitlementCycle,
    StandingAuthority,
)
from app.services.claim import ClaimKeyConflict, MissingCycle, UnknownAccount, evaluate_claim
from app.services.claim_deadline import bounded_claim
from app.services.evidence import associate_evidence, persist_evidence
from app.services.human_review import record_human_decision


@pytest.fixture
def store(tmp_path):
    engine = enable_sqlite_foreign_keys(
        create_engine(
            f"sqlite:///{tmp_path / 'claims.sqlite'}",
            connect_args={"check_same_thread": False},
        )
    )
    create_inference_schema(engine)
    at = now_utc()
    with Session(engine) as session:
        session.add(
            Account(
                account_id="account",
                created_at=at - timedelta(days=10),
                email_domain_type=EmailDomainType.CONSUMER,
                verification_level=VerificationLevel.EMAIL,
            )
        )
        session.flush()
        session.add(
            EntitlementCycle(
                cycle_id="cycle",
                account_id="account",
                cycle_start=at - timedelta(days=1),
                cycle_end=at + timedelta(days=29),
                credits_granted=0,
                credits_consumed=0,
                credits_expired=0,
            )
        )
        session.add(
            ClusterSnapshot(
                snapshot_id="snapshot",
                as_of=at,
                member_account_ids=["account"],
                edges=[],
                evidence_summary={},
                content_hash="snapshot-hash",
                created_at=at,
            )
        )
        session.flush()
        session.add(
            Decision(
                decision_id="decision",
                account_id="account",
                cluster_snapshot_id="snapshot",
                created_at=at,
                detector_version="d",
                policy_version="p",
                risk_score=0.9,
                proposed_action=ProposedAction.HOLD_INCREMENTAL_CREDITS,
                status=DecisionStatus.APPLIED,
                run_id="run",
                idempotency_key="decision-key",
            )
        )
        session.flush()
        evidence = persist_evidence(
            session,
            {
                "cluster_snapshot_id": "snapshot",
                "cycle_id": "unit",
                "as_of": at.isoformat(),
                "window_start": (at - timedelta(days=1)).isoformat(),
                "window_end": (at + timedelta(days=29)).isoformat(),
                "member_account_ids": ["account"],
                "edges": [],
                "allowance": 25,
                "excess": 1.5,
                "proposed_action": "hold_incremental_credits",
                "cap_reasons": [],
            },
        )
        associate_evidence(session, "decision", evidence.evidence_id)
        session.add(StandingAuthority(account_id="account", decision_id="decision"))
        session.commit()
    yield engine, at
    from app.services import claim_deadline

    limit = time.perf_counter() + 2
    while time.perf_counter() < limit:
        with claim_deadline._REGISTRY_LOCK:
            active = any(key[0] == id(engine) for key in claim_deadline._OPERATIONS)
        if not active:
            break
        time.sleep(0.01)
    assert not active, "deadline worker did not release its bounded slot"
    engine.dispose()


def test_allowance_uses_request_amount_and_post_snapshot_grants(store):
    engine, at = store
    with Session(engine) as session:
        first = evaluate_claim(
            session, account_id="account", credits_requested=20, request_key="first", as_of=at
        )
        held = evaluate_claim(
            session,
            account_id="account",
            credits_requested=6,
            request_key="second",
            as_of=at + timedelta(seconds=1),
        )
        assert first.granted_amount == 20 and not first.deferred
        assert held.held_amount == 6 and held.outcome == "held"
        assert session.get(EntitlementCycle, "cycle").credits_granted == 20
        events = session.scalars(select(AccountEvent)).all()
        assert len(events) == 1 and events[0].credits_amount == 20
        session.commit()


def test_retry_replays_without_double_grant_and_payload_conflict(store):
    engine, at = store
    with Session(engine) as session:
        first = evaluate_claim(
            session, account_id="account", credits_requested=10, request_key="same", as_of=at
        )
        session.commit()
    with Session(engine) as session:
        replay = evaluate_claim(
            session,
            account_id="account",
            credits_requested=10,
            request_key="same",
            as_of=at,
        )
        assert replay.replayed and replay.claim_id == first.claim_id
        assert session.scalar(select(func.count()).select_from(AccountEvent)) == 1
        assert session.scalar(select(func.count()).select_from(AuditEvent)) == 1
        with pytest.raises(ClaimKeyConflict):
            evaluate_claim(
                session, account_id="account", credits_requested=11, request_key="same", as_of=at
            )


def test_no_automatic_authority_and_no_stale_cycle_hold(store):
    engine, at = store
    with Session(engine) as session:
        decision = session.get(Decision, "decision")
        decision.status = DecisionStatus.PROPOSED
        # Existing applied-only behavior: excess alone cannot hold.
        assert evaluate_claim(session, account_id="account", credits_requested=30, as_of=at).granted
        session.rollback()
    with Session(engine) as session:
        # A stale snapshot must grant/defer rather than extending an old hold.
        result = evaluate_claim(
            session, account_id="account", credits_requested=30, as_of=at + timedelta(days=2)
        )
        assert result.granted and result.deferred
        with pytest.raises(MissingCycle):
            evaluate_claim(session, account_id="account", as_of=at + timedelta(days=31))
        with pytest.raises(UnknownAccount):
            evaluate_claim(session, account_id="missing", as_of=at)


def test_reversal_restores_actual_held_grant_once_without_clock_reset(store):
    engine, at = store
    with Session(engine) as session:
        evaluate_claim(
            session, account_id="account", credits_requested=25, request_key="granted", as_of=at
        )
        held = evaluate_claim(
            session, account_id="account", credits_requested=7, request_key="held", as_of=at
        )
        cycle = session.get(EntitlementCycle, "cycle")
        old_start, old_end = cycle.cycle_start, cycle.cycle_end
        record_human_decision(session, session.get(Decision, "decision"), action="reverse")
        row = session.get(SyntheticClaim, held.claim_id)
        assert row.held_amount == 0 and row.granted_amount == 7
        session.refresh(cycle)
        assert cycle.credits_granted == 32
        assert (cycle.cycle_start, cycle.cycle_end) == (old_start, old_end)
        restore = session.scalars(
            select(AuditEvent).where(AuditEvent.event_name == "credits_restored")
        ).one()
        assert restore.payload_json["amount_restored"] == 7
        replay = evaluate_claim(
            session, account_id="account", credits_requested=7, request_key="held", as_of=at
        )
        assert replay.granted and replay.replayed
        assert session.scalar(select(func.sum(AccountEvent.credits_amount))) == 32


def test_pending_hold_recovery_grants_and_is_not_restored_as_a_hold(store):
    engine, at = store
    with Session(engine) as session:
        out = evaluate_claim(
            session,
            account_id="account",
            credits_requested=30,
            request_key="interrupted",
            as_of=at,
            provisional_hold=True,
        )
        assert session.get(SyntheticClaim, out.claim_id).hold_state == "pending"
        session.commit()
    with Session(engine) as session:
        record_human_decision(session, session.get(Decision, "decision"), action="reverse")
        restore = session.scalars(
            select(AuditEvent).where(AuditEvent.event_name == "credits_restored")
        ).one()
        assert restore.payload_json["amount_restored"] == 0
        recovered = evaluate_claim(
            session, account_id="account", credits_requested=30, request_key="interrupted", as_of=at
        )
        assert recovered.granted and recovered.deferred


def test_concurrent_claims_do_not_lose_shared_allowance(store):
    engine, at = store
    results = []
    errors = []

    def claim(key):
        try:
            with Session(engine) as session:
                results.append(
                    evaluate_claim(
                        session,
                        account_id="account",
                        credits_requested=20,
                        request_key=key,
                        as_of=at,
                    )
                )
                session.commit()
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=claim, args=(str(i),)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    assert not errors and len(results) == 2
    assert sorted(result.outcome for result in results) == ["granted", "held"]
    with Session(engine) as session:
        assert session.scalar(select(func.sum(AccountEvent.credits_amount))) == 20


def test_slow_commit_cannot_create_late_effective_hold(store, monkeypatch):
    engine, at = store
    original = Session.commit
    delayed = threading.Event()

    def commit(session):
        if threading.current_thread().name == "synthetic-claim" and not delayed.is_set():
            delayed.set()
            time.sleep(0.2)
        return original(session)

    monkeypatch.setattr(Session, "commit", commit)
    started = time.perf_counter()
    result = bounded_claim(
        engine,
        account_id="account",
        credits_requested=30,
        request_key="slow",
        as_of=at,
        budget_ms=80,
    )
    assert time.perf_counter() - started < 0.18
    assert result.granted and result.deferred
    # Wait for the bounded worker to finish, then inspect the late transaction.
    limit = time.perf_counter() + 2
    while time.perf_counter() < limit:
        with Session(engine) as session:
            row = session.scalar(select(SyntheticClaim).where(SyntheticClaim.request_key == "slow"))
            if row is not None and row.hold_state == "none":
                assert row.granted_amount == 30 and row.held_amount == 0
                assert session.scalar(select(func.count()).select_from(AccountEvent)) == 1
                return
        time.sleep(0.01)
    pytest.fail("late worker did not resolve its ineffective pending hold")


def test_database_failure_returns_deferred_grant(store, monkeypatch):
    engine, at = store
    from app.services import claim_deadline

    def unavailable(*args, **kwargs):
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(claim_deadline, "evaluate_claim", unavailable)
    result = bounded_claim(
        engine,
        account_id="account",
        credits_requested=2,
        request_key="outage",
        as_of=at,
        budget_ms=80,
    )
    assert result.granted and result.deferred and result.granted_amount == 2


def test_grant_audit_and_event_are_atomic(store):
    engine, at = store
    with Session(engine) as session:
        result = evaluate_claim(
            session, account_id="account", credits_requested=3, request_key="rolled-back", as_of=at
        )
        assert result.granted
        session.rollback()
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(SyntheticClaim)) == 0
        assert session.scalar(select(func.count()).select_from(AccountEvent)) == 0
        assert session.scalar(select(func.count()).select_from(AuditEvent)) == 0
        assert session.get(EntitlementCycle, "cycle").credits_granted == 0


def test_closed_cycle_hold_is_not_restored(store):
    from app.services.policy_engine import record_reversal

    engine, at = store
    with Session(engine) as session:
        held = evaluate_claim(
            session, account_id="account", credits_requested=30, request_key="closed", as_of=at
        )
        decision = session.get(Decision, "decision")
        record_reversal(
            session,
            decision,
            member_account_ids=["account"],
            edges=[],
            excess=1.5,
            as_of=at + timedelta(days=30),
        )
        assert session.get(SyntheticClaim, held.claim_id).held_amount == 30
        assert session.scalar(select(func.count()).select_from(AccountEvent)) == 0


def test_explicit_clock_is_part_of_request_payload(store):
    engine, at = store
    with Session(engine) as session:
        evaluate_claim(session, account_id="account", request_key="clock", as_of=at)
        with pytest.raises(ClaimKeyConflict):
            evaluate_claim(
                session, account_id="account", request_key="clock", as_of=at + timedelta(seconds=1)
            )


@pytest.mark.parametrize(
    ("account_id", "credits_requested", "as_of"),
    [
        ("other-account", 30, "same"),
        ("account", 29, "same"),
        ("account", 30, "different"),
    ],
)
def test_timed_out_conflicting_request_never_reconciles_existing_claim(
    store, monkeypatch, account_id, credits_requested, as_of
):
    """Fallback recovery validates the key payload before it can touch a hold."""
    from app.services import claim_deadline

    engine, at = store
    with Session(engine) as session:
        original = evaluate_claim(
            session,
            account_id="account",
            credits_requested=30,
            request_key="owned-key",
            as_of=at,
        )
        session.commit()

    def unavailable(*args, **kwargs):
        raise RuntimeError("transient storage failure before claim evaluation")

    monkeypatch.setattr(claim_deadline, "evaluate_claim", unavailable)
    conflicting_at = at if as_of == "same" else at + timedelta(seconds=1)
    result = bounded_claim(
        engine,
        account_id=account_id,
        credits_requested=credits_requested,
        request_key="owned-key",
        as_of=conflicting_at,
        budget_ms=80,
    )
    assert result.granted and result.deferred
    limit = time.perf_counter() + 2
    while time.perf_counter() < limit:
        with claim_deadline._REGISTRY_LOCK:
            if not any(key[0] == id(engine) for key in claim_deadline._OPERATIONS):
                break
        time.sleep(0.01)
    else:
        pytest.fail("conflicting worker did not finish")
    with Session(engine) as session:
        row = session.get(SyntheticClaim, original.claim_id)
        assert row is not None
        assert row.account_id == "account"
        assert row.requested_amount == 30
        assert row.created_at.replace(tzinfo=at.tzinfo) == at
        assert row.hold_state == "confirmed"
        assert row.held_amount == 30 and row.granted_amount == 0


def test_omitted_clock_retry_replays_after_fresh_engine(store):
    """Internal worker timing must not become an implicit retry payload field."""
    engine, _ = store
    first = bounded_claim(
        engine, account_id="account", credits_requested=30, request_key="implicit-clock"
    )
    assert first.outcome == "held"
    limit = time.perf_counter() + 2
    while time.perf_counter() < limit:
        from app.services import claim_deadline

        with claim_deadline._REGISTRY_LOCK:
            if not any(key[0] == id(engine) for key in claim_deadline._OPERATIONS):
                break
        time.sleep(0.01)
    else:
        pytest.fail("first claim worker did not finish")

    fresh = make_inference_engine(str(engine.url))
    try:
        replay = bounded_claim(
            fresh, account_id="account", credits_requested=30, request_key="implicit-clock"
        )
        assert replay.replayed and replay.claim_id == first.claim_id
    finally:
        fresh.dispose()
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(SyntheticClaim)) == 1


def test_exhausted_sqlite_pool_defers_without_wait(tmp_path):
    """The production factory configures claims not to wait for SQLite pool checkout."""
    engine = make_inference_engine(f"sqlite:///{tmp_path / 'pool-exhausted.sqlite'}")
    create_inference_schema(engine)
    # File-backed SQLite's QueuePool defaults to five pooled plus ten overflow
    # connections. Exhaust the factory-created pool; this test would block for
    # QueuePool's normal timeout if ``make_inference_engine`` did not set zero.
    held = [engine.connect() for _ in range(15)]
    try:
        started = time.perf_counter()
        result = bounded_claim(
            engine, account_id="unreachable", request_key="pool-exhausted", budget_ms=80
        )
        assert time.perf_counter() - started < 0.18
        assert result.granted and result.deferred
        from app.services import claim_deadline

        limit = time.perf_counter() + 2
        while time.perf_counter() < limit:
            with claim_deadline._REGISTRY_LOCK:
                if not any(key[0] == id(engine) for key in claim_deadline._OPERATIONS):
                    break
            time.sleep(0.01)
        else:
            pytest.fail("pool-exhausted worker did not clean up")
    finally:
        for connection in held:
            connection.close()
        engine.dispose()


def test_slow_lookup_cannot_return_a_hold(store, monkeypatch):
    from app.services import claim_deadline

    engine, at = store
    original = claim_deadline.evaluate_claim

    def slow(*args, **kwargs):
        time.sleep(0.2)
        return original(*args, **kwargs)

    monkeypatch.setattr(claim_deadline, "evaluate_claim", slow)
    started = time.perf_counter()
    out = bounded_claim(
        engine,
        account_id="account",
        credits_requested=30,
        request_key="slow-lookup",
        as_of=at,
        budget_ms=80,
    )
    assert time.perf_counter() - started < 0.18
    assert out.granted and out.deferred


def test_worker_saturation_does_not_queue_unbounded_work(store, monkeypatch):
    from app.services import claim_deadline

    engine, at = store

    class Busy:
        def acquire(self, *, blocking):
            assert not blocking
            return False

    monkeypatch.setattr(claim_deadline, "_SLOTS", Busy())
    out = bounded_claim(engine, account_id="account", request_key="saturated", as_of=at)
    assert out.granted and out.deferred
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(SyntheticClaim)) == 0


def test_freshness_check_does_not_materialize_every_snapshot(store):
    """Regression for the unbounded snapshot scan in the 300ms claim path.

    Many unrelated ClusterSnapshot rows land inside the (cutoff, at] freshness
    window. The old code loaded every one of them into an ORM object and
    JSON-decoded `member_account_ids` in Python to test membership. The fixed
    query pushes the membership test into SQL as a bounded EXISTS, so no
    ClusterSnapshot instance should ever enter the session's identity map for
    this check, regardless of how many unrelated snapshots exist.
    """
    engine, at = store
    with Session(engine) as session:
        for i in range(60):
            session.add(
                ClusterSnapshot(
                    snapshot_id=f"unrelated-{i}",
                    as_of=at + timedelta(minutes=i + 1),
                    member_account_ids=["someone-else"],
                    edges=[],
                    evidence_summary={},
                    content_hash=f"unrelated-hash-{i}",
                    created_at=at,
                )
            )
        session.commit()

    with Session(engine) as session:
        later = at + timedelta(minutes=61)
        outcome = evaluate_claim(
            session,
            account_id="account",
            credits_requested=1,
            request_key="freshness-check",
            as_of=later,
        )
        session.commit()
        # None of the 60 unrelated snapshots contain "account", so freshness
        # still holds and the claim proceeds normally.
        assert outcome.deferred is False
        loaded_snapshots = sum(
            1 for identity_key in session.identity_map if identity_key[0] is ClusterSnapshot
        )
        assert loaded_snapshots == 0, (
            "freshness check materialized ClusterSnapshot ORM rows into Python "
            f"({loaded_snapshots}); it must be answered by a bounded SQL query instead"
        )


def test_freshness_check_defers_when_newer_snapshot_contains_account(store):
    """Positive-branch regression for the SQL-side membership predicate.

    A newer immutable membership frame that *does* contain the claiming account
    must make the old applied allowance stale even though it is well within the
    one-day age window — this is the branch the rewritten `json_each` EXISTS
    query must still answer correctly (not just the "no match" case covered by
    `test_freshness_check_does_not_materialize_every_snapshot`).
    """
    engine, at = store
    with Session(engine) as session:
        session.add(
            ClusterSnapshot(
                snapshot_id="newer-with-account",
                as_of=at + timedelta(minutes=30),
                member_account_ids=["account"],
                edges=[],
                evidence_summary={},
                content_hash="newer-with-account-hash",
                created_at=at,
            )
        )
        session.commit()

    with Session(engine) as session:
        later = at + timedelta(minutes=45)
        outcome = evaluate_claim(
            session,
            account_id="account",
            credits_requested=1,
            request_key="freshness-defers",
            as_of=later,
        )
        session.commit()
        assert outcome.deferred is True


def test_thread_start_failure_releases_slot_and_registry(store, monkeypatch):
    """Regression for the permanent semaphore/registry leak on Thread.start() failure.

    If `threading.Thread.start()` raises after the slot is acquired and the
    registry entry is written, the worker that owns the only release/pop never
    runs. The fix must release the slot and pop the registry entry itself and
    still fail open with a valid grant.
    """
    from app.services import claim_deadline

    engine, at = store
    available_before = claim_deadline._SLOTS._value

    def failing_start(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", failing_start)
    out = bounded_claim(
        engine,
        account_id="account",
        credits_requested=1,
        request_key="thread-start-fails",
        as_of=at,
    )
    assert out.granted and out.deferred
    assert claim_deadline._SLOTS._value == available_before
    registry_key = (id(engine), "thread-start-fails")
    with claim_deadline._REGISTRY_LOCK:
        assert registry_key not in claim_deadline._OPERATIONS

    monkeypatch.undo()
    # A later request reusing the pattern must not join a zombie op and stall.
    started = time.perf_counter()
    retry = bounded_claim(
        engine,
        account_id="account",
        credits_requested=1,
        request_key="thread-start-fails-retry",
        as_of=at,
    )
    assert time.perf_counter() - started < 1.0
    assert retry.outcome in ("granted", "held")


def test_fast_hold_is_confirmed_only_after_response_acknowledgment(store):
    engine, at = store
    out = bounded_claim(
        engine, account_id="account", credits_requested=30, request_key="normal-hold", as_of=at
    )
    assert out.outcome == "held" and not out.deferred
    limit = time.perf_counter() + 2
    while time.perf_counter() < limit:
        with Session(engine) as session:
            row = session.scalar(
                select(SyntheticClaim).where(SyntheticClaim.request_key == "normal-hold")
            )
            if row is not None and row.hold_state == "confirmed":
                assert row.held_amount == 30
                assert session.scalar(select(func.count()).select_from(AccountEvent)) == 0
                return
        time.sleep(0.01)
    pytest.fail("accepted response did not confirm its hold")


def _claim_audit_rows(session, request_key):
    rows = session.scalars(
        select(AuditEvent).where(AuditEvent.event_name == "credit_claim_evaluated")
    ).all()
    return [r.payload_json for r in rows if r.payload_json.get("request_key") == request_key]


def test_lost_reconciliation_leaves_exactly_one_pending_audit_row(store, monkeypatch):
    """C-1: a hold whose reconciliation never runs (timeout/crash) must still have
    an audit trail -- the pending row is the only record it ever existed."""
    from app.services import claim_deadline

    engine, at = store

    def never_reconciles(*args, **kwargs):
        return None  # Simulates a reconciliation that times out / is lost.

    monkeypatch.setattr(claim_deadline, "_reconcile", never_reconciles)
    out = bounded_claim(
        engine, account_id="account", credits_requested=30, request_key="lost-reconcile", as_of=at
    )
    assert out.outcome == "held"
    limit = time.perf_counter() + 2
    while time.perf_counter() < limit:
        with claim_deadline._REGISTRY_LOCK:
            if not any(key[0] == id(engine) for key in claim_deadline._OPERATIONS):
                break
        time.sleep(0.01)
    else:
        pytest.fail("worker did not finish")
    with Session(engine) as session:
        payloads = _claim_audit_rows(session, "lost-reconcile")
        assert len(payloads) == 1
        assert payloads[0]["outcome"] == "held"
        assert payloads[0]["hold_state"] == "pending"


def test_resolved_hold_leaves_pending_then_confirmed_audit_rows(store):
    """C-1 happy path: a hold that resolves normally leaves TWO audit rows."""
    engine, at = store
    out = bounded_claim(
        engine,
        account_id="account",
        credits_requested=30,
        request_key="audited-hold",
        as_of=at,
    )
    assert out.outcome == "held" and not out.deferred
    limit = time.perf_counter() + 2
    while time.perf_counter() < limit:
        with Session(engine) as session:
            row = session.scalar(
                select(SyntheticClaim).where(SyntheticClaim.request_key == "audited-hold")
            )
            if row is not None and row.hold_state == "confirmed":
                break
        time.sleep(0.01)
    else:
        pytest.fail("worker did not confirm the hold")
    with Session(engine) as session:
        payloads = _claim_audit_rows(session, "audited-hold")
        assert len(payloads) == 2
        hold_states = sorted(p["hold_state"] for p in payloads)
        assert hold_states == ["confirmed", "pending"]
        assert all(p["outcome"] == "held" for p in payloads)


def test_fail_open_hold_leaves_pending_then_terminal_granted_audit_rows(store, monkeypatch):
    """C-1 fail-open path: a slow commit forces the response to fail open, and
    recovery later resolves the never-effective hold to a plain grant. Both
    events -- the provisional pending hold and its terminal grant -- must be
    audited."""
    engine, at = store
    original = Session.commit
    delayed = threading.Event()

    def commit(session):
        if threading.current_thread().name == "synthetic-claim" and not delayed.is_set():
            delayed.set()
            time.sleep(0.2)
        return original(session)

    monkeypatch.setattr(Session, "commit", commit)
    result = bounded_claim(
        engine,
        account_id="account",
        credits_requested=30,
        request_key="fail-open-audit",
        as_of=at,
        budget_ms=80,
    )
    assert result.granted and result.deferred
    limit = time.perf_counter() + 2
    while time.perf_counter() < limit:
        with Session(engine) as session:
            row = session.scalar(
                select(SyntheticClaim).where(SyntheticClaim.request_key == "fail-open-audit")
            )
            if row is not None and row.hold_state == "none":
                break
        time.sleep(0.01)
    else:
        pytest.fail("late worker did not resolve its ineffective pending hold")
    with Session(engine) as session:
        payloads = _claim_audit_rows(session, "fail-open-audit")
        assert len(payloads) == 2
        by_state = {p["hold_state"]: p for p in payloads}
        assert by_state["pending"]["outcome"] == "held"
        assert by_state["none"]["outcome"] == "granted"


def test_poisoned_restore_connection_is_invalidated_not_pooled(tmp_path, monkeypatch):
    """C-4: if the restoring PRAGMA/progress-handler statements themselves raise,
    the poisoned raw connection must be discarded (invalidated), never returned to
    the pool still stuck at the 20ms claim-budget busy_timeout."""
    import sqlite3

    from app.services import claim_deadline

    # ``sqlite3.Connection`` is a C-level immutable type -- its methods cannot be
    # monkeypatched directly. A thin Python subclass, installed as the DBAPI
    # connection factory, gives us a mutable class to patch instead.
    class _Connection(sqlite3.Connection):
        pass

    def failing_restore(self, sql, *args, **kwargs):
        # Only the restoring statement (``PRAGMA busy_timeout=<previous>``) should
        # fail; the initial lowering to 20ms must succeed so the budget still
        # applies during the claim itself.
        is_restore_pragma = (
            isinstance(sql, str)
            and sql.startswith("PRAGMA busy_timeout=")
            and not sql.endswith("=20")
        )
        if is_restore_pragma:
            raise sqlite3.OperationalError("database is locked")
        return sqlite3.Connection.execute(self, sql, *args, **kwargs)

    monkeypatch.setattr(_Connection, "execute", failing_restore)

    engine = enable_sqlite_foreign_keys(
        create_engine(
            f"sqlite:///{tmp_path / 'poison.sqlite'}",
            connect_args={"check_same_thread": False, "factory": _Connection},
        )
    )
    create_inference_schema(engine)
    at = now_utc()
    with Session(engine) as session:
        session.add(
            Account(
                account_id="account",
                created_at=at - timedelta(days=10),
                email_domain_type=EmailDomainType.CONSUMER,
                verification_level=VerificationLevel.EMAIL,
            )
        )
        session.add(
            EntitlementCycle(
                cycle_id="cycle",
                account_id="account",
                cycle_start=at - timedelta(days=1),
                cycle_end=at + timedelta(days=29),
                credits_granted=0,
                credits_consumed=0,
                credits_expired=0,
            )
        )
        session.commit()

    out = bounded_claim(
        engine,
        account_id="account",
        credits_requested=1,
        request_key="poison-restore",
        as_of=at,
        budget_ms=80,
    )
    assert out.granted
    limit = time.perf_counter() + 2
    while time.perf_counter() < limit:
        with claim_deadline._REGISTRY_LOCK:
            if not any(key[0] == id(engine) for key in claim_deadline._OPERATIONS):
                break
        time.sleep(0.01)
    else:
        pytest.fail("worker did not finish")
    monkeypatch.undo()  # Inspect the next checkout with real sqlite3 behavior.

    connection = engine.connect()
    try:
        raw = connection.connection.driver_connection
        current_timeout = raw.execute("PRAGMA busy_timeout").fetchone()[0]
        assert current_timeout != 20, (
            "a connection poisoned at the 20ms claim budget leaked back into the pool"
        )
        # If a progress handler from the poisoned connection were still installed
        # (impossible here since invalidation forces a brand-new DBAPI connection,
        # but this is the behavioral proof the fix asks for), a query crossing its
        # step-count threshold past its already-elapsed deadline would raise
        # sqlite3.OperationalError("interrupted"). This must complete cleanly.
        row_count = raw.execute(
            "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 5000) "
            "SELECT count(*) FROM cnt"
        ).fetchone()[0]
        assert row_count == 5000
    finally:
        connection.close()
        engine.dispose()


def test_lock_allowance_swallows_locked_operational_error_on_bootstrap(store, monkeypatch):
    """C-7: on SQLite, two processes racing the very first claim more often produce
    ``OperationalError: database is locked`` than a uniqueness violation. The direct
    (non-bounded) path must treat that the same as the IntegrityError case."""
    from app.models.claims import ClaimAllowanceLock

    engine, at = store
    original_flush = Session.flush

    def flush(self, *args, **kwargs):
        if any(isinstance(obj, ClaimAllowanceLock) for obj in self.new):
            raise OperationalError(
                "INSERT INTO claim_allowance_lock ...", {}, Exception("database is locked")
            )
        return original_flush(self, *args, **kwargs)

    monkeypatch.setattr(Session, "flush", flush)
    with Session(engine) as session:
        result = evaluate_claim(
            session, account_id="account", credits_requested=1, request_key="lock-race", as_of=at
        )
        assert result.granted


def test_lock_allowance_swallows_locked_operational_error_on_revision_bump(store, monkeypatch):
    """C-7 follow-up (adversarial review): swallowing the lock error on the
    bootstrap INSERT is not enough by itself -- the very next statement, the
    revision-bump UPDATE, can raise the identical 'database is locked' error
    on the same connection when a concurrent bootstrap still holds SQLite's
    writer lock. That must degrade the same way (fail open), not propagate."""
    import sqlalchemy

    engine, at = store
    original_execute = Session.execute

    def execute(self, statement, *args, **kwargs):
        if (
            isinstance(statement, sqlalchemy.sql.dml.Update)
            and statement.table.name == "claim_allowance_locks"
        ):
            raise OperationalError(
                "UPDATE claim_allowance_locks ...", {}, Exception("database is locked")
            )
        return original_execute(self, statement, *args, **kwargs)

    monkeypatch.setattr(Session, "execute", execute)
    with Session(engine) as session:
        result = evaluate_claim(
            session,
            account_id="account",
            credits_requested=1,
            request_key="revision-lock",
            as_of=at,
        )
        assert result.granted


def test_lock_allowance_reraises_non_lock_operational_error_on_bootstrap(store, monkeypatch):
    """C-7 negative case: a real (non-lock) OperationalError on the singleton
    bootstrap must still propagate, not be silently swallowed."""
    from app.models.claims import ClaimAllowanceLock

    engine, at = store
    original_flush = Session.flush

    def flush(self, *args, **kwargs):
        if any(isinstance(obj, ClaimAllowanceLock) for obj in self.new):
            raise OperationalError(
                "INSERT INTO claim_allowance_lock ...", {}, Exception("disk I/O error")
            )
        return original_flush(self, *args, **kwargs)

    monkeypatch.setattr(Session, "flush", flush)
    with Session(engine) as session:
        with pytest.raises(OperationalError):
            evaluate_claim(
                session, account_id="account", credits_requested=1, request_key="io-error", as_of=at
            )

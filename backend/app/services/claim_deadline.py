"""Bounded claim request coordination with explicit best-effort outage accounting.

No request Session crosses threads. At most four daemon workers exist, with no work
queue. A late database commit can persist only a *pending*, ineffective hold. Only a
response accepted before the deadline authorizes later hold confirmation. If that
confirmation fails or the process crashes, recovery grants/defer: the approved
outage policy favors availability over durable restriction/accounting guarantees.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from time import perf_counter

from sqlalchemy import Engine, select
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session

from app.core.clock import now_utc
from app.models.claims import SyntheticClaim
from app.models.enums import DecisionStatus
from app.models.inference import Decision
from app.services.claim import (
    CLAIM_BUDGET_MS,
    ClaimKeyConflict,
    ClaimOutcome,
    MissingCycle,
    UnknownAccount,
    evaluate_claim,
    lock_allowance,
    resolve_pending,
    utc,
)

_MAX_WORKERS = 4
_SLOTS = threading.BoundedSemaphore(_MAX_WORKERS)
_REGISTRY_LOCK = threading.Lock()


@dataclass
class _Operation:
    account_id: str
    amount: int
    deadline: float
    key: str
    at: datetime
    requested_as_of: datetime | None
    lock: threading.Lock = field(default_factory=threading.Lock)
    ready: threading.Event = field(default_factory=threading.Event)
    cancelled: bool = False
    accepted: bool = False
    acknowledged: threading.Event = field(default_factory=threading.Event)
    result: ClaimOutcome | None = None
    error: Exception | None = None


_OPERATIONS: dict[tuple[int, str], _Operation] = {}


def _matches_payload(row: SyntheticClaim, op: _Operation) -> bool:
    """Whether a durable request-key row belongs to this worker's client payload.

    ``op.at`` is an internal effective instant, not idempotency input. An omitted
    caller timestamp therefore remains compatible with a prior row, while an
    explicit timestamp must match exactly before recovery may touch that row.
    """
    return (
        row.account_id == op.account_id
        and row.requested_amount == op.amount
        and (op.requested_as_of is None or utc(row.created_at) == utc(op.requested_as_of))
    )


def _fallback(op: _Operation, start: float) -> ClaimOutcome:
    return ClaimOutcome(
        "granted",
        True,
        (perf_counter() - start) * 1000,
        None,
        None,
        None,
        request_key=op.key,
        granted_amount=op.amount,
    )


@contextmanager
def _database_budget(session: Session, deadline: float) -> Iterator[None]:
    """Bound SQLite lock/query time; pool waits remain isolated in bounded workers.

    Callers must `session.commit()` (or otherwise finish using `session`) INSIDE
    this `with` block. On the invalidate-on-restore-failure path below, the
    connection is left with an unresolved transaction; using the session again
    after the block exits raises `PendingRollbackError` instead of whatever the
    caller expected -- both current call sites already commit inside the block.
    """
    connection = session.connection()
    raw = connection.connection.driver_connection
    if not isinstance(raw, sqlite3.Connection):
        yield
        return
    previous = raw.execute("PRAGMA busy_timeout").fetchone()[0]
    raw.execute("PRAGMA busy_timeout=20")
    raw.set_progress_handler(lambda: int(perf_counter() >= deadline), 1000)
    try:
        yield
    finally:
        try:
            raw.set_progress_handler(None, 0)
            raw.execute(f"PRAGMA busy_timeout={int(previous)}")
        except Exception:
            # The restoring statements themselves can fail (e.g. the transaction
            # was aborted by the progress handler mid-restore). A stranded
            # progress handler or a connection stuck at a 20ms busy_timeout would
            # silently corrupt every later unrelated caller that checks this
            # connection out of the pool — discarding one pooled connection is
            # cheaper than that. Never re-raise: that would mask whatever
            # exception the `with` block's body was already propagating.
            try:
                connection.invalidate()
            except Exception:
                # invalidate() itself can raise (e.g. ResourceClosedError on an
                # already-closed connection) -- swallow it too, for the same
                # reason: it must never replace the body's in-flight exception.
                pass


@contextmanager
def _checkout_before(engine: Engine, deadline: float) -> Iterator[Connection]:
    """Acquire an inference connection only while this bounded operation has time.

    File-backed SQLite inference engines are configured with a zero pool timeout
    in ``make_inference_engine``.  This wrapper makes the deadline checks
    explicit around that no-wait checkout and keeps reconciliation bounded too.
    """
    if perf_counter() >= deadline:
        raise TimeoutError("claim deadline elapsed before database checkout")
    connection = engine.connect()
    try:
        if perf_counter() >= deadline:
            raise TimeoutError("claim deadline elapsed during database checkout")
        yield connection
    finally:
        connection.close()


def _reconcile(engine: Engine, op: _Operation, *, accepted: bool) -> None:
    # One bounded best-effort attempt, no infinite retry queue. Storage failure or
    # crash may lose the record; pending holds never acquire enforcement authority.
    deadline = perf_counter() + 0.050
    with _checkout_before(engine, deadline) as connection, Session(bind=connection) as session:
        with _database_budget(session, deadline):
            if perf_counter() >= deadline:
                return
            lock_allowance(session)
            row = session.scalar(select(SyntheticClaim).where(SyntheticClaim.request_key == op.key))
            if row is not None:
                if not _matches_payload(row, op):
                    # A timed-out conflicting request must never recover, grant,
                    # or otherwise mutate the durable claim that owns its key.
                    return
                decision = session.get(Decision, row.decision_id) if row.decision_id else None
                accepted = (
                    accepted and decision is not None and decision.status is DecisionStatus.APPLIED
                )
                resolve_pending(
                    session,
                    row,
                    accepted=accepted,
                    at=op.at,
                    latency_ms=0.0,
                    force_grant=not accepted,
                )
            elif not accepted:
                evaluate_claim(
                    session,
                    account_id=op.account_id,
                    credits_requested=op.amount,
                    request_key=op.key,
                    as_of=op.requested_as_of,
                    _effective_at=op.at,
                    force_grant=True,
                )
            if perf_counter() >= deadline:
                session.rollback()
                return
            session.commit()


def _work(engine: Engine, op: _Operation, registry_key: tuple[int, str]) -> None:
    accepted = False
    try:
        try:
            with (
                _checkout_before(engine, op.deadline) as connection,
                Session(bind=connection) as session,
            ):
                with _database_budget(session, op.deadline):
                    result = evaluate_claim(
                        session,
                        account_id=op.account_id,
                        credits_requested=op.amount,
                        request_key=op.key,
                        as_of=op.requested_as_of,
                        _effective_at=op.at,
                        provisional_hold=True,
                        budget_ms=max(0.0, (op.deadline - perf_counter()) * 1000),
                    )
                    session.commit()
            with op.lock:
                if not op.cancelled and perf_counter() < op.deadline:
                    op.result = result

                else:
                    op.cancelled = True
                op.ready.set()
            if result.outcome == "held":
                # The response thread, not the background worker, authorizes a hold.
                op.acknowledged.wait(max(0.0, op.deadline - perf_counter()))
                with op.lock:
                    accepted = op.accepted
                    if not accepted:
                        op.cancelled = True
        except (UnknownAccount, MissingCycle, ClaimKeyConflict) as exc:
            with op.lock:
                if not op.cancelled and perf_counter() < op.deadline:
                    op.error = exc
                op.ready.set()
            return
        except Exception:
            # DB, pool, serialization and audit failures all follow the same
            # availability policy. Never surface an in-budget claim of a hold.
            with op.lock:
                op.cancelled = True
                op.ready.set()
        if accepted or op.cancelled:
            try:
                _reconcile(engine, op, accepted=accepted)
            except Exception:
                pass  # Explicitly non-durable; see module contract.
    finally:
        with _REGISTRY_LOCK:
            _OPERATIONS.pop(registry_key, None)
        _SLOTS.release()


def bounded_claim(
    engine: Engine,
    *,
    account_id: str,
    credits_requested: int = 1,
    request_key: str | None = None,
    as_of: datetime | None = None,
    budget_ms: float = CLAIM_BUDGET_MS,
    request_started: float | None = None,
) -> ClaimOutcome:
    """Wait only the remaining application budget, including DB commit work.

    Caller supplies request-entry `perf_counter()` when available. Reserve 15 ms
    for response construction; arbitrary scheduling/network delays are not bounded.
    Saturation returns grant/defer immediately with no additional worker.
    """
    start = request_started if request_started is not None else perf_counter()
    if credits_requested <= 0:
        raise ValueError("credits_requested must be positive")
    key = request_key or uuid.uuid4().hex
    registry_key = (id(engine), key)
    with _REGISTRY_LOCK:
        op = _OPERATIONS.get(registry_key)
        if op is not None:
            if (
                op.account_id != account_id
                or op.amount != credits_requested
                or (as_of is not None and utc(as_of) != utc(op.at))
            ):
                raise ClaimKeyConflict("request key already used with a different claim payload")
        else:
            op = _Operation(
                account_id,
                credits_requested,
                start + max(0.0, budget_ms - 15.0) / 1000,
                key,
                utc(as_of or now_utc()),
                as_of,
            )
            if not _SLOTS.acquire(blocking=False):
                return _fallback(op, start)
            _OPERATIONS[registry_key] = op
            try:
                threading.Thread(
                    target=_work,
                    args=(engine, op, registry_key),
                    daemon=True,
                    name="synthetic-claim",
                ).start()
            except Exception:
                # `_work`'s finally block owns the only release/pop for this slot
                # and registry entry; since it never got a chance to run, mirror
                # its cleanup here so a start() failure (e.g. thread limit) can't
                # permanently leak a semaphore slot or leave a zombie registry
                # entry that a future request with the same key would block on.
                _OPERATIONS.pop(registry_key, None)
                _SLOTS.release()
                return _fallback(op, start)
    remaining = min(op.deadline, start + max(0.0, budget_ms - 15.0) / 1000) - perf_counter()
    op.ready.wait(max(0.0, remaining))
    with op.lock:
        if op.error is not None:
            raise op.error
        if op.result is not None:
            if op.result.granted or op.accepted or perf_counter() < op.deadline:
                if not op.result.granted:
                    op.accepted = True
                op.acknowledged.set()
                return replace(
                    op.result,
                    latency_ms=(perf_counter() - start) * 1000,
                    deferred=op.result.deferred
                    or (op.result.granted and (perf_counter() - start) * 1000 >= budget_ms),
                )
        op.cancelled = True
        op.acknowledged.set()
        return _fallback(op, start)

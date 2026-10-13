"""Synthetic grants against an approved standing allowance, never re-clustering.

`evaluate_claim` participates in its caller's transaction. The HTTP deadline wrapper
is `claim_deadline.bounded_claim`; a synchronous Session is not a deadline boundary.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Literal, cast

from sqlalchemy import func, select, true, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from app.core.clock import now_utc
from app.db.decisions import current_decision
from app.models.claims import ClaimAllowanceLock, SyntheticClaim
from app.models.enums import ActorType, DecisionStatus, EventType, ProposedAction
from app.models.inference import Account, AccountEvent, ClusterSnapshot, Decision, EntitlementCycle
from app.schemas.audit import CreditClaimEvaluated
from app.services.audit import append_audit

CLAIM_BUDGET_MS = 300.0
CLAIM_BLOCKING_ACTIONS = frozenset(
    {ProposedAction.HOLD_INCREMENTAL_CREDITS, ProposedAction.TEMPORARY_HOLD}
)
ClaimResult = Literal["granted", "held"]


class UnknownAccount(ValueError):
    pass


class MissingCycle(ValueError):
    pass


class ClaimKeyConflict(ValueError):
    pass


@dataclass(frozen=True)
class ClaimOutcome:
    outcome: ClaimResult
    deferred: bool
    latency_ms: float
    standing_decision_id: str | None
    standing_action: ProposedAction | None
    standing_status: DecisionStatus | None
    claim_id: str | None = None
    request_key: str | None = None
    granted_amount: int = 0
    held_amount: int = 0
    replayed: bool = False

    @property
    def granted(self) -> bool:
        return self.outcome == "granted"


def utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def lock_allowance(session: Session) -> None:
    """Serialize the read/modify/write allowance operation across all snapshots.

    A database lock also covers processes (an in-memory mutex alone would not).
    SQLite takes its writer lock; PostgreSQL takes the singleton row lock.
    """
    if session.get(ClaimAllowanceLock, "inference") is None:
        try:
            # Exiting `begin_nested()` issues a ROLLBACK TO SAVEPOINT, which can
            # itself fail if the database is already locked -- do not assume a
            # clean savepoint unwind is guaranteed just because the insert failed
            # for a reason we're prepared to swallow.
            with session.begin_nested():
                session.add(ClaimAllowanceLock(scope="inference", revision=0))
                session.flush()
        except IntegrityError:
            pass  # Another transaction initialized the singleton.
        except OperationalError as exc:
            # On SQLite, two processes racing the very first claim more often
            # produce "database is locked" than a uniqueness violation. Treat
            # that specific case the same as IntegrityError: someone else is
            # bootstrapping the singleton. Any other OperationalError is a real
            # failure and must propagate.
            if "locked" not in str(exc.orig).lower():
                raise
    try:
        # A concurrent bootstrap can still hold SQLite's writer lock when we
        # reach here (the swallow above only means *this* transaction isn't
        # the one bootstrapping), so this UPDATE can raise the identical
        # "database is locked" error. Degrade the same way: skip the revision
        # bump under the P2-R3 fail-open policy rather than propagate, since a
        # missed bump here is a missed opportunity to invalidate a stale read,
        # not a lost write.
        session.execute(
            update(ClaimAllowanceLock)
            .where(ClaimAllowanceLock.scope == "inference")
            .values(revision=ClaimAllowanceLock.revision + 1)
        )
    except OperationalError as exc:
        if "locked" not in str(exc.orig).lower():
            raise


def current_cycle(session: Session, account_id: str, at: datetime) -> EntitlementCycle | None:
    return session.scalars(
        select(EntitlementCycle)
        .where(
            EntitlementCycle.account_id == account_id,
            EntitlementCycle.cycle_start <= at,
            EntitlementCycle.cycle_end > at,
        )
        .order_by(EntitlementCycle.cycle_start.desc())
    ).first()


def _is_active_hold(standing: Decision | None) -> bool:
    return (
        standing is not None
        and standing.status is DecisionStatus.APPLIED
        and standing.proposed_action in CLAIM_BLOCKING_ACTIONS
    )


def grant_credits(session: Session, row: SyntheticClaim, amount: int, at: datetime) -> None:
    """A single deterministic event per operation; restoration uses the same event key.

    A held request has no claim event yet. Its eventual restoration grants its entire
    requested amount once. Imported grants remain independent events.
    """
    if amount <= 0:
        return
    session.add(
        AccountEvent(
            event_id=f"claim-{row.claim_id}",
            account_id=row.account_id,
            timestamp=at,
            event_type=EventType.CLAIM_CREDITS,
            credits_amount=amount,
            endpoint_cost_tier=None,
        )
    )
    session.execute(
        update(EntitlementCycle)
        .where(
            EntitlementCycle.cycle_id == row.cycle_id,
        )
        .values(credits_granted=EntitlementCycle.credits_granted + amount)
    )


def audit_claim(session: Session, row: SyntheticClaim, latency_ms: float) -> None:
    append_audit(
        session,
        CreditClaimEvaluated(
            entity_type="account",
            entity_id=row.account_id,
            actor_type=ActorType.SYSTEM,
            account_id=row.account_id,
            credits_requested=row.requested_amount,
            outcome="held" if row.held_amount else "granted",
            deferred=row.deferred,
            standing_decision_id=row.decision_id,
            latency_ms=latency_ms,
            claim_id=row.claim_id,
            request_key=row.request_key,
            granted_amount=row.granted_amount,
            held_amount=row.held_amount,
            hold_state=cast(Literal["none", "pending", "confirmed"], row.hold_state),
        ),
    )


def _outcome(
    row: SyntheticClaim, standing: Decision | None, elapsed: float, *, replayed: bool = False
) -> ClaimOutcome:
    return ClaimOutcome(
        outcome="held" if row.held_amount else "granted",
        deferred=row.deferred,
        latency_ms=elapsed,
        standing_decision_id=row.decision_id,
        standing_action=standing.proposed_action if standing else None,
        standing_status=standing.status if standing else None,
        claim_id=row.claim_id,
        request_key=row.request_key,
        granted_amount=row.granted_amount,
        held_amount=row.held_amount,
        replayed=replayed,
    )


def resolve_pending(
    session: Session,
    row: SyntheticClaim,
    *,
    accepted: bool,
    at: datetime,
    latency_ms: float,
    force_grant: bool = False,
) -> None:
    """Only the deadline arbiter's accepted token may make a pending hold effective.

    A retry/restart without that token fails open. A late worker cannot overwrite
    a grant that a retry already resolved; callers hold the allowance writer lock.
    """
    if row.hold_state != "pending" and not (force_grant and row.held_amount > 0):
        return
    if accepted:
        row.hold_state = "confirmed"
    else:
        row.granted_amount = row.requested_amount
        row.held_amount = 0
        row.hold_state = "none"
        row.deferred = True
        # Keep the original claim instant/cycle. Recovery never starts a new clock.
        grant_credits(session, row, row.granted_amount, utc(row.created_at))
    audit_claim(session, row, latency_ms)
    session.flush()


def evaluate_claim(
    session: Session,
    *,
    account_id: str,
    credits_requested: int = 1,
    request_key: str | None = None,
    as_of: datetime | None = None,
    _effective_at: datetime | None = None,
    budget_ms: float = CLAIM_BUDGET_MS,
    force_grant: bool = False,
    provisional_hold: bool = False,
    _standing: Callable[[Session, str], Decision | None] = current_decision,
) -> ClaimOutcome:
    """Record a whole incremental grant/hold with idempotent normal persistence.

    The account's existing open cycle is mandatory. Unknown/missing cycles are not
    manufactured from wall time. Applied authority is necessary but does not bypass
    allowance, historical window or evidence guardrails.
    """
    start = perf_counter()
    # ``as_of`` is part of the client-visible idempotency payload.  The bounded
    # API path may need to keep a single effective instant while its worker and
    # recovery transaction run; that internal instant must not turn an omitted
    # client timestamp into a conflict on a later retry.
    at = utc(_effective_at or as_of or now_utc())
    key = request_key or uuid.uuid4().hex
    if credits_requested <= 0:
        raise ValueError("credits_requested must be positive")
    from app.db.publication import lock_publication

    lock_publication(session)
    lock_allowance(session)
    existing = session.scalar(select(SyntheticClaim).where(SyntheticClaim.request_key == key))
    if existing is not None:
        if (
            existing.account_id != account_id
            or existing.requested_amount != credits_requested
            or (as_of is not None and utc(existing.created_at) != at)
        ):
            raise ClaimKeyConflict("request key already used with a different claim payload")
        # A pending row without its live arbiter is not an effective hold.
        resolve_pending(
            session,
            existing,
            accepted=False,
            at=utc(existing.created_at),
            latency_ms=0.0,
        )
        standing = session.get(Decision, existing.decision_id) if existing.decision_id else None
        return _outcome(existing, standing, (perf_counter() - start) * 1000, replayed=True)
    if session.get(Account, account_id) is None:
        raise UnknownAccount(account_id)
    cycle = current_cycle(session, account_id, at)
    if cycle is None:
        raise MissingCycle("account has no existing open entitlement cycle")
    standing = _standing(session, account_id)
    from app.models.inference import StandingAuthority

    authority = session.get(StandingAuthority, account_id, populate_existing=True)
    if authority is not None and authority.run_id is not None:
        if (authority.cycle_id != cycle.cycle_id or authority.as_of is None
                or utc(authority.as_of) > at):
            standing = None
    if standing is not None:
        session.refresh(standing)  # never trust an earlier ORM status across the writer lock
    from app.services.evidence import decision_evidence

    evidence = decision_evidence(session, standing.decision_id) if standing else None
    valid = False
    should_hold = False
    needs_review = False
    if evidence is not None:
        cutoff = utc(datetime.fromisoformat(evidence["as_of"]))
        window_start = utc(datetime.fromisoformat(evidence["window_start"]))
        window_end = utc(datetime.fromisoformat(evidence["window_end"]))
        members = evidence["member_account_ids"]
        valid = (
            account_id in members
            and cutoff <= at < cutoff + timedelta(days=1)
            and window_start <= at < window_end
            and utc(cycle.cycle_start) <= cutoff < utc(cycle.cycle_end)
        )
        if valid:
            # A newer immutable membership frame makes the old applied allowance
            # stale even when the old snapshot is less than one day old.
            #
            # `member_account_ids` is an unindexed JSON column, so this cannot be
            # a SQL-side equality filter on the column itself. Instead of loading
            # every ClusterSnapshot row in (cutoff, at] into Python and decoding
            # each one's JSON to test membership (unbounded, scales with total
            # snapshot volume on the single most latency-sensitive path in the
            # service), push the membership test into SQLite via `json_each` and
            # ask only an EXISTS question — the DB short-circuits on first match
            # and no ORM object or JSON payload is ever materialized in Python.
            #
            # SQLite-only: `json_each(array)` has different semantics on other
            # dialects (e.g. Postgres treats the argument as an object, not an
            # array). Revisit if a non-SQLite engine is ever introduced.
            member_ids = func.json_each(ClusterSnapshot.member_account_ids).table_valued("value")
            stale_snapshot_exists = (
                select(ClusterSnapshot.snapshot_id)
                .select_from(ClusterSnapshot.__table__.join(member_ids, true()))
                .where(
                    ClusterSnapshot.as_of > cutoff,
                    ClusterSnapshot.as_of <= at,
                    member_ids.c.value == account_id,
                )
                .exists()
            )
            valid = not bool(session.execute(select(stale_snapshot_exists)).scalar())
        if valid:
            # The original allowance is fixed; events are disjointly counted once
            # across its whole window (including the post-snapshot suffix).
            claimed = (
                session.scalar(
                    select(func.coalesce(func.sum(AccountEvent.credits_amount), 0)).where(
                        AccountEvent.account_id.in_(members),
                        AccountEvent.event_type == EventType.CLAIM_CREDITS,
                        AccountEvent.timestamp >= window_start,
                        AccountEvent.timestamp < window_end,
                        AccountEvent.timestamp <= at,
                    )
                )
                or 0
            )
            needs_review = claimed + credits_requested > evidence[
                "allowance"
            ] and not _is_active_hold(standing)
            blocking_actions = {a.value for a in CLAIM_BLOCKING_ACTIONS}
            should_hold = (
                _is_active_hold(standing)
                and evidence["proposed_action"] in blocking_actions
                and not evidence.get("cap_reasons")
                and claimed + credits_requested > evidence["allowance"]
            )
    deferred = (
        force_grant or not valid or needs_review or (perf_counter() - start) * 1000 >= budget_ms
    )
    held = credits_requested if should_hold and not deferred else 0
    row = SyntheticClaim(
        claim_id=uuid.uuid4().hex,
        request_key=key,
        account_id=account_id,
        cycle_id=cycle.cycle_id,
        requested_amount=credits_requested,
        granted_amount=credits_requested - held,
        held_amount=held,
        decision_id=standing.decision_id if standing else None,
        created_at=at,
        deferred=deferred,
        restored_at=None,
        hold_state="pending" if held and provisional_hold else "confirmed" if held else "none",
    )
    session.add(row)
    grant_credits(session, row, row.granted_amount, at)
    elapsed = (perf_counter() - start) * 1000
    # Audit every outcome, including a provisional "pending" hold: it is the only
    # trace of a hold that never reaches enforcement authority if reconciliation
    # (`resolve_pending` via `_reconcile`) times out or is lost. A confirmed hold
    # therefore produces two `credit_claim_evaluated` rows: pending, then confirmed.
    audit_claim(session, row, elapsed)
    session.flush()
    return _outcome(row, standing, elapsed)

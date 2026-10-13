"""Append-only decision lifecycle (docs/architecture.md §3).

`record_decision` never mutates a prior decision's assessment: it inserts a new row
and supersedes the current supersedable one via the status machine, appending an
audit event. `transition_decision` moves status through the machine, rejecting
illegal moves and auditing every legal one. `current_decision` is the append-only
"current view": the latest non-superseded decision for an account.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import set_committed_value

from app.core.clock import now_utc
from app.core.decision_status import SUPERSEDABLE, IllegalTransition, assert_transition
from app.db.publication import lock_publication
from app.models.enums import ActorType, DecisionStatus, ProposedAction
from app.models.inference import Decision, StandingAuthority
from app.schemas.audit import (
    DecisionProposed,
    DecisionStatusChanged,
    DecisionSuperseded,
)
from app.services.audit import append_audit as _append_audit


def _uuid() -> str:
    return uuid.uuid4().hex


class ConcurrentTransition(IllegalTransition):
    """Another transaction already changed the expected persisted state."""


def current_decision(session: Session, account_id: str) -> Decision | None:
    """Read the explicit authority pointer; never resurrect an older history row."""
    authority = session.get(StandingAuthority, account_id, populate_existing=True)
    if authority is None or authority.decision_id is None:
        return None
    return session.get(Decision, authority.decision_id)


def _supersedable_current(session: Session, account_id: str) -> Decision | None:
    stmt = (
        select(Decision)
        .where(
            Decision.account_id == account_id,
            Decision.status.in_(SUPERSEDABLE),
        )
        .order_by(Decision.created_at.desc(), Decision.decision_id.desc())
    )
    return session.scalars(stmt).first()


def transition_decision(
    session: Session,
    decision: Decision,
    target_status: DecisionStatus,
    *,
    actor: ActorType,
    reason: str | None = None,
    at: datetime | None = None,
) -> None:
    """Move a decision's status through the machine. Raises IllegalTransition on a
    forbidden move (content columns are untouched either way)."""
    assert_transition(decision.status, target_status)
    from_status = decision.status
    # The graph has no cycles, so expected status is a sufficient CAS token (no ABA).
    # Flush normal ORM work first so content immutability guards remain in force.
    session.flush()
    changed = session.execute(
        update(Decision)
        .where(Decision.decision_id == decision.decision_id, Decision.status == from_status)
        .values(status=target_status)
        .returning(Decision.decision_id)
        .execution_options(synchronize_session=False)
    ).scalar_one_or_none()
    if changed is None:
        raise ConcurrentTransition("Decision changed concurrently; reload before acting")
    set_committed_value(decision, "status", target_status)
    _append_audit(
        session,
        DecisionStatusChanged(
            entity_type="decision",
            entity_id=decision.decision_id,
            actor_type=actor,
            decision_id=decision.decision_id,
            from_status=from_status,
            to_status=target_status,
            reason=reason,
        ),
        at=at,
    )


def record_decision(
    session: Session,
    *,
    account_id: str,
    risk_score: float,
    proposed_action: ProposedAction,
    detector_version: str,
    policy_version: str,
    run_id: str,
    idempotency_key: str,
    cluster_snapshot_id: str | None = None,
) -> Decision:
    """Insert a new decision and supersede the current supersedable one, if any.
    Never mutates the prior decision's assessment columns."""
    # Find the prior supersedable decision BEFORE inserting the new one, so the
    # new PROPOSED row cannot match as its own predecessor.
    lock_publication(session)
    prior = _supersedable_current(session, account_id)

    new = Decision(
        decision_id=_uuid(),
        account_id=account_id,
        cluster_snapshot_id=cluster_snapshot_id,
        created_at=now_utc(),
        detector_version=detector_version,
        policy_version=policy_version,
        risk_score=risk_score,
        proposed_action=proposed_action,
        status=DecisionStatus.PROPOSED,
        run_id=run_id,
        idempotency_key=idempotency_key,
        superseded_by=None,
    )
    session.add(new)
    session.flush()  # make new queryable / stabilize ordering

    if prior is not None:
        transition_decision(
            session,
            prior,
            DecisionStatus.SUPERSEDED,
            actor=ActorType.SYSTEM,
            reason="superseded by new detector run",
        )
        prior.superseded_by = new.decision_id
        _append_audit(
            session,
            DecisionSuperseded(
                entity_type="decision",
                entity_id=prior.decision_id,
                actor_type=ActorType.SYSTEM,
                decision_id=prior.decision_id,
                superseded_by=new.decision_id,
            ),
        )

    _append_audit(
        session,
        DecisionProposed(
            entity_type="decision",
            entity_id=new.decision_id,
            actor_type=ActorType.SYSTEM,
            decision_id=new.decision_id,
            risk_score=risk_score,
            proposed_action=proposed_action,
            run_id=run_id,
        ),
    )
    authority = session.get(StandingAuthority, account_id)
    if authority is None:
        authority = StandingAuthority(account_id=account_id)
        session.add(authority)
    authority.decision_id = new.decision_id
    authority.run_id = None
    authority.as_of = None
    authority.cycle_id = None
    session.flush()
    return new


def find_by_idempotency_key(session: Session, idempotency_key: str) -> Decision | None:
    """The decision already recorded for this idempotency key, if any."""
    return session.scalars(
        select(Decision)
        .where(Decision.idempotency_key == idempotency_key)
        .order_by(Decision.created_at, Decision.decision_id)
    ).first()


class IdempotencyConflict(Exception):
    """Raised when an idempotency key is reused with a DIFFERENT decision payload —
    the semantic identity of a key must not silently change (review finding 8)."""


def _payload_matches(
    existing: Decision,
    *,
    risk_score: float,
    proposed_action: ProposedAction,
    detector_version: str,
    policy_version: str,
    cluster_snapshot_id: str | None,
) -> bool:
    return (
        existing.risk_score == risk_score
        and existing.proposed_action == proposed_action
        and existing.detector_version == detector_version
        and existing.policy_version == policy_version
        and existing.cluster_snapshot_id == cluster_snapshot_id
    )


def record_decision_idempotent(
    session: Session,
    *,
    account_id: str,
    risk_score: float,
    proposed_action: ProposedAction,
    detector_version: str,
    policy_version: str,
    run_id: str,
    idempotency_key: str,
    cluster_snapshot_id: str | None = None,
) -> tuple[Decision, bool]:
    """Record a decision unless one already exists for `idempotency_key` (architecture
    §7). Returns `(decision, created)`.

    The unique DB constraint on `idempotency_key` is the real guarantee: the lookup is
    the fast path, but a concurrent insert that slips past it raises `IntegrityError`,
    which is caught and resolved to the winning row — so two racing runs still leave
    exactly one decision (review finding 7). A key reused with a **different** payload
    is an `IdempotencyConflict`, never a silent stale return (finding 8); a genuinely
    new config must therefore use a new key so it supersedes rather than dedupes."""

    def _resolve(existing: Decision) -> tuple[Decision, bool]:
        if not _payload_matches(
            existing,
            risk_score=risk_score,
            proposed_action=proposed_action,
            detector_version=detector_version,
            policy_version=policy_version,
            cluster_snapshot_id=cluster_snapshot_id,
        ):
            raise IdempotencyConflict(
                f"idempotency key {idempotency_key!r} reused with a different payload"
            )
        return existing, False

    session.flush()  # surface any pending sibling insert before the lookup
    existing = find_by_idempotency_key(session, idempotency_key)
    if existing is not None:
        return _resolve(existing)

    savepoint = session.begin_nested()
    try:
        decision = record_decision(
            session,
            account_id=account_id,
            risk_score=risk_score,
            proposed_action=proposed_action,
            detector_version=detector_version,
            policy_version=policy_version,
            run_id=run_id,
            idempotency_key=idempotency_key,
            cluster_snapshot_id=cluster_snapshot_id,
        )
        savepoint.commit()
        return decision, True
    except IntegrityError:
        savepoint.rollback()  # lost the race — resolve to the winner
        winner = find_by_idempotency_key(session, idempotency_key)
        if winner is None:
            raise
        return _resolve(winner)

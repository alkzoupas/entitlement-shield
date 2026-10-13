"""L-5 escalation lifecycle (docs/decisions.md §12, Phase 3 plan P3.6).

`under_review` already existed as a `DecisionStatus`; escalation adds exactly one
new transition edge (`applied -> under_review`, `core/decision_status.py`) and one
new table (`EscalationRecord`). Escalating a decision:

  * transitions it to `under_review` (via `db.decisions.transition_decision`, which
    enforces `LEGAL_TRANSITIONS` — this is what makes step 1 of L-5 "an explicit,
    tested addition, not an implied no-op");
  * releases any confirmed current-cycle held credits immediately, via the same
    `policy_engine.restore_held_credits` mechanics `record_reversal` uses — but
    WITHOUT writing a suppression record and WITHOUT setting status to `reversed`;
  * writes an `EscalationRecord` with a 7-day `expires_at`.

At expiry, `release_expired_escalations` (run nightly from `detector.run_detector`,
or standalone via `scripts/release_escalations.py`) transitions the decision
`under_review -> superseded` with reason `escalation_expired`, `system`-actor,
writing NO suppression record — a human had the window to act; letting the window
lapse closes the case rather than silently re-applying the original hold.

"Open" is derived, not stored (see `EscalationRecord`'s docstring): a decision has
an open escalation while its status is still `UNDER_REVIEW`. Both the nightly
reconciliation (`detector._supersede_unaffirmed`) and the reviewer's invocation gate
(`reviewer._gating_reason`) skip a decision with an open escalation, so it is not
resuperseded-and-re-reviewed every night while a human review window is running.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.decisions import transition_decision
from app.models.enums import ActorType, DecisionStatus
from app.models.inference import Decision, EscalationRecord
from app.schemas.audit import DecisionEscalated, EscalationAutoReleased
from app.services.audit import append_audit
from app.services.policy_engine import restore_held_credits

ESCALATION_DAYS = 7


def _as_utc(value: datetime) -> datetime:
    # SQLite round-trips a `DateTime(timezone=True)` column as naive (the same
    # normalization `detector.py`/`claim.py` apply before comparing a DB-read
    # datetime against a caller-supplied `as_of`).
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def escalate_decision(
    session: Session,
    decision: Decision,
    *,
    source: str,
    as_of: datetime,
    actor: ActorType = ActorType.LLM,
    reason: str | None = None,
) -> EscalationRecord:
    """The L-5 escalation action. `decision.status` must be able to reach
    `under_review` per `LEGAL_TRANSITIONS` (today: `proposed` or `applied`) — an
    illegal source status (already `under_review`, or a terminal `reversed`/
    `superseded`) raises `IllegalTransition`, exactly as any other transition would."""
    transition_decision(
        session,
        decision,
        DecisionStatus.UNDER_REVIEW,
        actor=actor,
        reason=reason or f"escalated:{source}",
    )
    # Effective severity is capped at light_friction for the duration of an
    # escalation (L-5): any confirmed grant-restricting hold is released now, not
    # held until expiry. A light_friction (or already-unheld) decision has nothing
    # to restore and this is a no-op beyond the (zero-amount) audit row.
    restore_held_credits(session, decision, actor=actor, as_of=as_of)

    record = EscalationRecord(
        escalation_id=uuid.uuid4().hex,
        decision_id=decision.decision_id,
        created_at=as_of,
        expires_at=as_of + timedelta(days=ESCALATION_DAYS),
        source=source,
    )
    session.add(record)
    session.flush()
    append_audit(
        session,
        DecisionEscalated(
            entity_type="decision",
            entity_id=decision.decision_id,
            actor_type=actor,
            decision_id=decision.decision_id,
            escalation_id=record.escalation_id,
            source=source,
            expires_at=record.expires_at.isoformat(),
        ),
    )
    return record


def has_open_escalation(session: Session, decision: Decision) -> bool:
    """True while `decision` is under an open L-5 escalation window: still
    `under_review` AND at least one `EscalationRecord` exists for it. Deliberately
    NOT gated on `as_of` vs. `expires_at` — the window only closes when something
    (an analyst action, or `release_expired_escalations`) actually moves the status
    away from `under_review`; until that happens, gating/reconciliation must keep
    skipping even if `as_of` has already passed `expires_at`, or a delayed release
    run would let a re-review/re-supersede loop through the gap (RC-1 P1-3)."""
    if decision.status is not DecisionStatus.UNDER_REVIEW:
        return False
    return (
        session.scalars(
            select(EscalationRecord.escalation_id).where(
                EscalationRecord.decision_id == decision.decision_id
            )
        ).first()
        is not None
    )


def release_expired_escalations(session: Session, *, as_of: datetime, run_id: str) -> int:
    """Terminal transition for every decision whose (latest) escalation has expired
    as of `as_of`: `under_review -> superseded`, reason `escalation_expired`,
    `system` actor, NO suppression record. `as_of`-bounded like the rest of the
    detector's batch pipeline (never acts on an expiry in `as_of`'s future, even if
    wall-clock time has since passed it — the finding-11 pattern) and idempotent:
    a decision released on a prior call is no longer `under_review`, so a re-run
    with the same or a later `as_of` naturally excludes it. Returns the count
    released."""
    candidates = session.scalars(
        select(Decision)
        .join(EscalationRecord, EscalationRecord.decision_id == Decision.decision_id)
        .where(Decision.status == DecisionStatus.UNDER_REVIEW)
        .distinct()
    ).all()

    released = 0
    for decision in candidates:
        # A decision may have been escalated more than once over its lifetime (e.g.
        # escalated, resolved by an analyst, later escalated again); only the LATEST
        # EscalationRecord's expiry governs whether the current under_review window
        # has lapsed — an earlier, already-superseded-by-analyst-action record must
        # not trigger a second (now-illegal) transition attempt.
        latest = session.scalars(
            select(EscalationRecord)
            .where(EscalationRecord.decision_id == decision.decision_id)
            .order_by(EscalationRecord.created_at.desc(), EscalationRecord.escalation_id.desc())
        ).first()
        if latest is None or _as_utc(as_of) < _as_utc(latest.expires_at):
            continue

        transition_decision(
            session,
            decision,
            DecisionStatus.SUPERSEDED,
            actor=ActorType.SYSTEM,
            reason="escalation_expired",
        )
        append_audit(
            session,
            EscalationAutoReleased(
                entity_type="decision",
                entity_id=decision.decision_id,
                actor_type=ActorType.SYSTEM,
                decision_id=decision.decision_id,
                escalation_id=latest.escalation_id,
                run_id=run_id,
            ),
        )
        released += 1

    session.flush()
    return released


__all__ = [
    "ESCALATION_DAYS",
    "escalate_decision",
    "has_open_escalation",
    "release_expired_escalations",
]

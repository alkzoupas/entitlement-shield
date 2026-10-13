"""Analyst decisions from the console (docs/architecture.md §3, original-plan §17).

A thin, tested wrapper over the decision status machine: an analyst moves a decision
through `start_review → apply`, or `reverse`s it, and every move is a legal transition
(rejected otherwise), audited by `transition_decision`, and recorded as a `HumanReview`
row so the console shows who did what.

Manual reversal includes suppression and restoration in the same transaction. Legacy
cases without original immutable evidence must be regenerated from known inputs;
today's observations cannot safely reconstruct the historical reversal signature.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.core.clock import now_utc
from app.core.decision_status import assert_transition
from app.db.decisions import transition_decision
from app.db.publication import lock_publication
from app.models.enums import ActorType, DecisionStatus
from app.models.inference import Decision, HumanReview
from app.services.evidence import decision_evidence
from app.services.policy_engine import record_reversal

# The analyst actions the console offers, each mapping to one target status. Legality
# against the decision's *current* status is enforced by the status machine, not here.
ANALYST_ACTIONS: dict[str, DecisionStatus] = {
    "start_review": DecisionStatus.UNDER_REVIEW,
    "apply": DecisionStatus.APPLIED,
    "reverse": DecisionStatus.REVERSED,
}


class MissingDecisionEvidence(ValueError):
    pass


@dataclass(frozen=True)
class UnknownAnalystAction(Exception):
    action: str

    def __str__(self) -> str:
        return f"unknown analyst action {self.action!r}; expected one of {sorted(ANALYST_ACTIONS)}"


def record_human_decision(
    session: Session,
    decision: Decision,
    *,
    action: str,
    reason: str | None = None,
    actor: ActorType = ActorType.ANALYST,
    as_of: datetime | None = None,
) -> HumanReview:
    """Apply an analyst `action` to `decision`: transition its status (auditing the move
    via the status machine, which raises `IllegalTransition` on a forbidden move) and
    record a `HumanReview` row. Raises `UnknownAnalystAction` for an unrecognized action.
    """
    at = as_of or now_utc()
    at = at.replace(tzinfo=UTC) if at.tzinfo is None else at.astimezone(UTC)
    lock_publication(session)
    if action not in ANALYST_ACTIONS:
        raise UnknownAnalystAction(action)
    target = ANALYST_ACTIONS[action]

    assert_transition(decision.status, target)
    if target is DecisionStatus.REVERSED:
        evidence = decision_evidence(session, decision.decision_id)
        if evidence is None:
            raise MissingDecisionEvidence("original immutable evidence is required for reversal")
        record_reversal(
            session,
            decision,
            member_account_ids=evidence["member_account_ids"],
            edges=[tuple(edge) for edge in evidence["edges"]],
            excess=evidence["excess"],
            actor=actor,
            reason=reason,
            as_of=at,
        )
    else:
        transition_decision(session, decision, target, actor=actor, reason=reason, at=at)

    review = HumanReview(
        review_id=uuid.uuid4().hex,
        decision_id=decision.decision_id,
        reviewer_action=action,
        reviewer_reason=reason,
        final_outcome=target.value,
        created_at=at,
    )
    session.add(review)
    session.flush()
    return review

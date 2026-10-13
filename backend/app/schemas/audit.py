"""Typed audit payloads — a Pydantic discriminated union on `event_name`
(docs/architecture.md §3). Every audit row's payload is validated against this
union before insert; the evaluation harness queries it, so it is not an untyped blob.

New event types are added here as later steps need them; the union is the single
registry of what may be written to `audit_events`.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from app.models.enums import ActorType, DecisionStatus, ProposedAction


class _AuditBase(BaseModel):
    # `extra="forbid"` closes the boundary: an unexpected field (e.g. a leaked
    # `sealed_label`) is a hard validation error, not silently dropped (review 7 P2).
    model_config = ConfigDict(extra="forbid")

    entity_type: str
    entity_id: str
    actor_type: ActorType


class _DecisionEvent(_AuditBase):
    """Audit events about a specific decision. `entity_id` must equal `decision_id`
    so the opaque entity reference and the typed payload cannot disagree (review 7)."""

    decision_id: str

    @model_validator(mode="after")
    def _entity_id_matches_decision(self) -> "_DecisionEvent":
        if self.entity_id != self.decision_id:
            raise ValueError(
                f"entity_id ({self.entity_id!r}) must equal decision_id "
                f"({self.decision_id!r}) for decision audit events"
            )
        return self


class DecisionProposed(_DecisionEvent):
    event_name: Literal["decision_proposed"] = "decision_proposed"
    risk_score: float
    proposed_action: ProposedAction
    run_id: str


class DecisionStatusChanged(_DecisionEvent):
    event_name: Literal["decision_status_changed"] = "decision_status_changed"
    from_status: DecisionStatus
    to_status: DecisionStatus
    reason: str | None = None


class DecisionSuperseded(_DecisionEvent):
    event_name: Literal["decision_superseded"] = "decision_superseded"
    superseded_by: str


class CreditsRestored(_DecisionEvent):
    """A reversal's current-cycle credit restoration (docs/policy.md §9.1). Clock is
    never touched; closed cycles are not restored."""

    event_name: Literal["credits_restored"] = "credits_restored"
    cycle_id: str
    amount_restored: int


class LlmReviewRecorded(_DecisionEvent):
    """One persisted `LlmReview` row (Phase 3 plan P3.5, docs/decisions.md L-8).
    `entity_id`/`decision_id` tie it to the reviewed decision; `review_id`
    identifies the exact row so the audit trail and `llm_reviews` table agree."""

    event_name: Literal["llm_review_recorded"] = "llm_review_recorded"
    review_id: str
    run_id: str
    sample_index: int
    arm: str
    provider: str
    retry_count: int
    derived_verdict: str | None = None
    #: L-8: the reviewer manifest hash in force for this attempt (see
    #: `models.inference.LlmReview.reviewer_manifest_hash`). Defaults to `""` so
    #: pre-P3.9 persisted audit payloads still validate against this schema.
    reviewer_manifest_hash: str = ""


class DecisionEscalated(_DecisionEvent):
    """L-5 escalation (docs/decisions.md §12, Phase 3 plan P3.6). `expires_at` is
    ISO-8601; the 7-day window is computed by `services/escalation.escalate_decision`,
    not re-derived here."""

    event_name: Literal["decision_escalated"] = "decision_escalated"
    escalation_id: str
    source: str
    expires_at: str


class EscalationAutoReleased(_DecisionEvent):
    """The 7-day terminal transition (`under_review -> superseded`, reason
    `escalation_expired`) — a `system`-actor event, never a suppression record."""

    event_name: Literal["escalation_auto_released"] = "escalation_auto_released"
    escalation_id: str
    run_id: str


class DetectorRunRecorded(_AuditBase):
    """A completed batch detector run (docs/architecture.md §2/§7). Not tied to a
    single decision — `entity_id` is the `run_id`. Records the run's shape so a
    double run (idempotent) is visible in the trail."""

    event_name: Literal["detector_run_recorded"] = "detector_run_recorded"
    run_id: str
    as_of: str  # ISO-8601
    clusters: int
    cluster_cycles: int
    decisions_created: int
    decisions_skipped: int


class CreditClaimEvaluated(_AuditBase):
    """A claim-time credit decision (docs/architecture.md §2). Not tied to a single
    decision — `entity_id` is the claiming `account_id`. The claim path evaluates the
    standing decision against the last snapshot and NEVER re-clusters; `deferred=True`
    records a timeout that granted and deferred to the nightly batch (fail-open — a
    wrongly-granted credit is cheap and reversible; wrongly blocking one is not)."""

    event_name: Literal["credit_claim_evaluated"] = "credit_claim_evaluated"
    account_id: str
    credits_requested: int
    claim_id: str | None = None
    request_key: str | None = None
    granted_amount: int = 0
    held_amount: int = 0
    replayed: bool = False
    outcome: Literal["granted", "held"]
    deferred: bool = False
    standing_decision_id: str | None = None
    latency_ms: float
    # A provisional hold emits a `hold_state="pending"` event at creation and a
    # terminal event (`"confirmed"` or `"none"`) on resolution. A `pending` event
    # with no successor row means reconciliation was lost under the P2-R3 outage
    # policy and the hold never acquired enforcement authority. Optional/defaulted
    # so older payloads (written before this field existed) still validate under
    # `extra="forbid"`.
    hold_state: Literal["none", "pending", "confirmed"] | None = None

    @model_validator(mode="after")
    def _entity_id_matches_account(self) -> "CreditClaimEvaluated":
        if self.entity_id != self.account_id:
            raise ValueError(
                f"entity_id ({self.entity_id!r}) must equal account_id "
                f"({self.account_id!r}) for a credit-claim event"
            )
        return self


AuditPayload = Annotated[
    DecisionProposed
    | DecisionStatusChanged
    | DecisionSuperseded
    | CreditsRestored
    | LlmReviewRecorded
    | DecisionEscalated
    | EscalationAutoReleased
    | DetectorRunRecorded
    | CreditClaimEvaluated,
    Field(discriminator="event_name"),
]

_ADAPTER: TypeAdapter[AuditPayload] = TypeAdapter(AuditPayload)


def parse_audit_payload(data: dict) -> AuditPayload:
    """Validate a raw payload dict against the union. Raises ValidationError on a
    bad event_name, missing field, or wrong type."""
    return _ADAPTER.validate_python(data)

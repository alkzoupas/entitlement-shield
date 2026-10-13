"""Inference-store ORM models — every table detector services may read.

Deliberately absent: any label, scenario, or split-group column. Those are
evaluation-only and live in the separate ground-truth store (review 4 P1-4d).
The old `scenario_id` on `accounts` (original-plan §7) is intentionally NOT here.

Content immutability: a `Decision`'s assessment columns (detector_version,
policy_version, risk_score, proposed_action, cluster_snapshot_id, created_at) are
never updated after insert; only `status` and `superseded_by` move, and every move
is recorded in `audit_events` (docs/architecture.md §3).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import InferenceBase
from app.models.enums import (
    ActorType,
    DecisionStatus,
    EmailDomainType,
    EndpointCostTier,
    EventType,
    ProposedAction,
    Reliability,
    SignalType,
    VerificationLevel,
)


def _enum(py_enum: type) -> Enum:
    """VARCHAR + CHECK (Postgres-portable), not a native DB enum."""
    return Enum(py_enum, native_enum=False, validate_strings=True)


class Account(InferenceBase):
    __tablename__ = "accounts"

    account_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    email_domain_type: Mapped[EmailDomainType] = mapped_column(_enum(EmailDomainType))
    verification_level: Mapped[VerificationLevel] = mapped_column(_enum(VerificationLevel))


class AccountEvent(InferenceBase):
    __tablename__ = "account_events"

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.account_id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    event_type: Mapped[EventType] = mapped_column(_enum(EventType))
    credits_amount: Mapped[int | None] = mapped_column(Integer, nullable=True)
    endpoint_cost_tier: Mapped[EndpointCostTier | None] = mapped_column(
        _enum(EndpointCostTier), nullable=True
    )


class LinkageObservation(InferenceBase):
    __tablename__ = "linkage_observations"

    observation_id: Mapped[str] = mapped_column(String, primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.account_id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    signal_type: Mapped[SignalType] = mapped_column(_enum(SignalType))
    # Raw values never cross the collection boundary — stored as a keyed hash
    # (docs/architecture.md §4). On synthetic data this models the production property.
    signal_value_hash: Mapped[str] = mapped_column(String, index=True)
    reliability: Mapped[Reliability] = mapped_column(_enum(Reliability))


class EntitlementCycle(InferenceBase):
    __tablename__ = "entitlement_cycles"

    cycle_id: Mapped[str] = mapped_column(String, primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.account_id"), index=True)
    cycle_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    cycle_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    credits_granted: Mapped[int] = mapped_column(Integer, default=0)
    credits_consumed: Mapped[int] = mapped_column(Integer, default=0)
    credits_expired: Mapped[int] = mapped_column(Integer, default=0)


class LinkageEdge(InferenceBase):
    """Materialized weighted edge (docs/architecture.md §3). Built in Step 3."""

    __tablename__ = "linkage_edges"

    edge_id: Mapped[str] = mapped_column(String, primary_key=True)
    account_a_id: Mapped[str] = mapped_column(ForeignKey("accounts.account_id"), index=True)
    account_b_id: Mapped[str] = mapped_column(ForeignKey("accounts.account_id"), index=True)
    weight: Mapped[float] = mapped_column(Float)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ClusterSnapshot(InferenceBase):
    """Immutable snapshot a decision references (docs/architecture.md §3). Built in Step 3."""

    __tablename__ = "cluster_snapshots"

    snapshot_id: Mapped[str] = mapped_column(String, primary_key=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    member_account_ids: Mapped[list] = mapped_column(JSON)
    edges: Mapped[list] = mapped_column(JSON)
    evidence_summary: Mapped[dict] = mapped_column(JSON)
    content_hash: Mapped[str] = mapped_column(String, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Decision(InferenceBase):
    """Append-only. Content columns are immutable; only status/superseded_by move."""

    __tablename__ = "decisions"

    decision_id: Mapped[str] = mapped_column(String, primary_key=True)
    account_id: Mapped[str] = mapped_column(
        ForeignKey("accounts.account_id"), index=True
    )
    cluster_snapshot_id: Mapped[str | None] = mapped_column(
        ForeignKey("cluster_snapshots.snapshot_id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    detector_version: Mapped[str] = mapped_column(String)
    policy_version: Mapped[str] = mapped_column(String)
    risk_score: Mapped[float] = mapped_column(Float)
    proposed_action: Mapped[ProposedAction] = mapped_column(_enum(ProposedAction))
    status: Mapped[DecisionStatus] = mapped_column(
        _enum(DecisionStatus), default=DecisionStatus.PROPOSED, index=True
    )
    run_id: Mapped[str] = mapped_column(String, index=True)
    # Globally unique so two concurrent runs cannot both insert the same unit — the
    # DB constraint, not just a lookup, is what makes idempotency race-safe (review
    # finding 7). The key embeds the config fingerprint so a config change is a new
    # key, not a silent reuse of a stale decision (finding 8).
    idempotency_key: Mapped[str] = mapped_column(String, unique=True, index=True)
    superseded_by: Mapped[str | None] = mapped_column(
        ForeignKey("decisions.decision_id"), nullable=True
    )


class LlmReview(InferenceBase):
    """One provider-call attempt for one reviewer sample (Phase 3, P3.5).

    One row per raw response (or provider-level failure) — not per sample: a
    sample that needed R-7's one repair retry persists two rows (`retry_count`
    0 then 1) so both attempts are visible, matching `reviewer_validation`'s
    `ParseAttempt` trail. `derived_verdict` is populated only on a sample's
    FINAL row (the parsed-and-evaluated attempt, or the terminal failure) —
    an intermediate `needs_retry` row leaves it null. `idempotency_key` is the
    reviewer run's own key (distinct from `Decision.idempotency_key`): a
    re-run with the same key must not call the provider again or insert
    duplicate rows (`services/reviewer.review_decision`)."""

    __tablename__ = "llm_reviews"

    review_id: Mapped[str] = mapped_column(String, primary_key=True)
    decision_id: Mapped[str] = mapped_column(ForeignKey("decisions.decision_id"), index=True)
    idempotency_key: Mapped[str] = mapped_column(String, index=True)
    sample_index: Mapped[int] = mapped_column(Integer)
    arm: Mapped[str] = mapped_column(String)
    provider: Mapped[str] = mapped_column(String)
    prompt_hash: Mapped[str] = mapped_column(String)
    model_snapshot: Mapped[str] = mapped_column(String)
    prompt_version: Mapped[str] = mapped_column(String)
    #: L-8 version observability: the `reviewer-manifest-v1` content hash in force
    #: when this attempt was made. Persisted per row rather than derived at read
    #: time on purpose — the audit guarantee ("its hash stamps every `LlmReview`
    #: row") must survive a later manifest bump, which a read-time derivation from
    #: the then-live `reviewer_manifest.MANIFEST_HASH` would silently rewrite.
    #: Defaults to `""` so the column is additive/`create_all`-safe (no Alembic in
    #: this repo) and pre-P3.9 fixtures keep constructing rows unchanged.
    reviewer_manifest_hash: Mapped[str] = mapped_column(String, default="")
    temperature: Mapped[float] = mapped_column(Float, default=0.0)
    raw_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    structured_output_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    validation_errors: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    recommendation: Mapped[str | None] = mapped_column(String, nullable=True)
    derived_verdict: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    citations_json: Mapped[list | None] = mapped_column(JSON, nullable=True)


class HumanReview(InferenceBase):
    """Schema only in Phase 1 — the analyst console is Phase 2."""

    __tablename__ = "human_reviews"

    review_id: Mapped[str] = mapped_column(String, primary_key=True)
    decision_id: Mapped[str] = mapped_column(ForeignKey("decisions.decision_id"), index=True)
    reviewer_action: Mapped[str] = mapped_column(String)
    reviewer_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_outcome: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class SuppressionRecord(InferenceBase):
    """A reversal's suppression window (docs/policy.md §9.3). While a live record
    exists for an evidence signature, identical evidence does not re-flag the
    cluster; materially new evidence yields a new signature and re-evaluates (§9.4).
    Built in Step 5."""

    __tablename__ = "suppression_records"

    suppression_id: Mapped[str] = mapped_column(String, primary_key=True)
    evidence_signature: Mapped[str] = mapped_column(String, index=True)
    cluster_snapshot_id: Mapped[str | None] = mapped_column(
        ForeignKey("cluster_snapshots.snapshot_id"), nullable=True
    )
    decision_id: Mapped[str] = mapped_column(
        ForeignKey("decisions.decision_id"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class EscalationRecord(InferenceBase):
    """An L-5 escalation window (docs/decisions.md §12, Phase 3 plan P3.6).

    No `closed_at`/`resolved_at` column: "open" is derived, not stored. Whatever
    closes the window — an analyst action or `escalation.release_expired_escalations`
    — already moves the owning `Decision.status` away from `UNDER_REVIEW`, and that
    move is itself audited. So "this decision has an open escalation" is exactly
    "it has an `EscalationRecord` row AND its status is still `UNDER_REVIEW`"
    (see `services/escalation.has_open_escalation`) — no second source of truth to
    keep in sync, and no window that outlives the status that defines it."""

    __tablename__ = "escalation_records"

    escalation_id: Mapped[str] = mapped_column(String, primary_key=True)
    decision_id: Mapped[str] = mapped_column(
        ForeignKey("decisions.decision_id"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    source: Mapped[str] = mapped_column(String)


class AuditEvent(InferenceBase):
    """Append-only, typed on `event_name`. Payload validated against the Pydantic
    discriminated union in app.schemas.audit before insert (docs/architecture.md §3)."""

    __tablename__ = "audit_events"

    audit_id: Mapped[str] = mapped_column(String, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    entity_type: Mapped[str] = mapped_column(String, index=True)
    entity_id: Mapped[str] = mapped_column(String, index=True)
    event_name: Mapped[str] = mapped_column(String, index=True)
    actor_type: Mapped[ActorType] = mapped_column(_enum(ActorType))
    payload_json: Mapped[dict] = mapped_column(JSON)


class DetectionEvidence(InferenceBase):
    """Content-addressed, immutable assessment and historical event projection."""

    __tablename__ = "detection_evidence"

    evidence_id: Mapped[str] = mapped_column(String, primary_key=True)
    cluster_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("cluster_snapshots.snapshot_id"), index=True
    )
    cycle_id: Mapped[str] = mapped_column(String, index=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    payload_json: Mapped[dict] = mapped_column(JSON)


class DecisionEvidence(InferenceBase):
    """Explicit immutable decision-to-assessment association, including legacy absence."""

    __tablename__ = "decision_evidence"

    decision_id: Mapped[str] = mapped_column(
        ForeignKey("decisions.decision_id"), primary_key=True
    )
    evidence_id: Mapped[str] = mapped_column(
        ForeignKey("detection_evidence.evidence_id"), index=True
    )


class Appeal(InferenceBase):
    """An account's appeal text (docs/policy.md §11: zero evidentiary weight — no
    feature may ever read this table). `decision_id` is nullable because an appeal
    can be submitted before any decision exists for the account (Phase 3 plan P3.1).

    `text` is hard-capped at `APPEAL_TEXT_MAX_CHARS` at write time (truncated, with
    `truncated` set) rather than at read/prompt time — the later prompt-side C-3
    escaping (P3.2) is not a substitute for bounding what gets persisted here."""

    __tablename__ = "appeals"

    appeal_id: Mapped[str] = mapped_column(String, primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.account_id"), index=True)
    decision_id: Mapped[str | None] = mapped_column(
        ForeignKey("decisions.decision_id"), nullable=True, index=True
    )
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    text: Mapped[str] = mapped_column(Text)
    truncated: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[str] = mapped_column(String)


class DetectorInvocation(InferenceBase):
    """One durable identity per successful detector transaction."""

    __tablename__ = "detector_invocations"
    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    config_fingerprint: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class PublicationLock(InferenceBase):
    __tablename__ = "publication_lock"
    scope: Mapped[str] = mapped_column(String, primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, default=0)


class StandingAuthority(InferenceBase):
    """Mutable current pointer; immutable decisions remain the historical record."""

    __tablename__ = "standing_authority"
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.account_id"), primary_key=True)
    decision_id: Mapped[str | None] = mapped_column(
        ForeignKey("decisions.decision_id"), nullable=True, unique=True
    )
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("detector_invocations.run_id"), nullable=True
    )
    as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cycle_id: Mapped[str | None] = mapped_column(
        ForeignKey("entitlement_cycles.cycle_id"), nullable=True
    )

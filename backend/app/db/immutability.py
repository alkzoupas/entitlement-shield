"""Session-level immutability guards for the append-only records (review 4 P1).

The data model documents immutable cluster snapshots, immutable decision assessment
columns, and a strictly append-only audit log — but ORM columns are mutable by
default, so nothing stopped a caller from updating a decision's `risk_score`,
rewriting a snapshot's members, or updating/deleting an `audit_events` row. That
would silently change the historical referents an audit trail depends on.

These are enforced here with mapper-level `before_update` / `before_delete`
listeners, so the guard holds for EVERY session (app and tests) regardless of the
caller, not by convention. The only legal decision mutations — `status` and
`superseded_by`, performed by the lifecycle functions in `db/decisions.py` — are
explicitly allowed; everything else raises `ImmutableRowError`.

Registration happens at import (module side effects). `create_inference_schema`
imports this module so the listeners are always active wherever the inference
schema is used.
"""

from __future__ import annotations

from sqlalchemy import Engine, event, inspect
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapper

from app.models.inference import (
    AuditEvent,
    ClusterSnapshot,
    Decision,
    DecisionEvidence,
    DetectionEvidence,
    HumanReview,
    LlmReview,
)


class ImmutableRowError(Exception):
    """Raised when a flush would mutate or delete an append-only/immutable row."""


# The ONLY decision columns the lifecycle may move after insert.
_DECISION_MUTABLE_COLUMNS = frozenset({"status", "superseded_by"})


def _changed_columns(target: object) -> set[str]:
    state = inspect(target, raiseerr=True)
    return {attr.key for attr in state.attrs if attr.history.has_changes()}


@event.listens_for(Decision, "before_update")
def _decision_before_update(
    mapper: Mapper, connection: Connection, target: Decision
) -> None:
    illegal = _changed_columns(target) - _DECISION_MUTABLE_COLUMNS
    if illegal:
        raise ImmutableRowError(
            "Decision content columns are immutable; only status/superseded_by may "
            f"move (attempted to change: {sorted(illegal)})"
        )


@event.listens_for(ClusterSnapshot, "before_update")
def _cluster_snapshot_before_update(
    mapper: Mapper, connection: Connection, target: ClusterSnapshot
) -> None:
    changed = _changed_columns(target)
    if changed:
        raise ImmutableRowError(
            "ClusterSnapshot rows are immutable (attempted to change: "
            f"{sorted(changed)})"
        )


@event.listens_for(AuditEvent, "before_update")
def _audit_event_before_update(
    mapper: Mapper, connection: Connection, target: AuditEvent
) -> None:
    raise ImmutableRowError("audit_events is append-only; rows cannot be updated")


@event.listens_for(AuditEvent, "before_delete")
def _audit_event_before_delete(
    mapper: Mapper, connection: Connection, target: AuditEvent
) -> None:
    raise ImmutableRowError("audit_events is append-only; rows cannot be deleted")


@event.listens_for(HumanReview, "before_update")
@event.listens_for(HumanReview, "before_delete")
@event.listens_for(LlmReview, "before_update")
@event.listens_for(LlmReview, "before_delete")
def _review_immutable(mapper: Mapper, connection: Connection, target: object) -> None:
    raise ImmutableRowError("review rows are append-only; rows cannot be changed")


@event.listens_for(DetectionEvidence, "before_update")
@event.listens_for(DetectionEvidence, "before_delete")
@event.listens_for(DecisionEvidence, "before_update")
@event.listens_for(DecisionEvidence, "before_delete")
def _evidence_immutable(mapper: Mapper, connection: Connection, target: object) -> None:
    raise ImmutableRowError("detection evidence and decision associations are immutable")


# Mapper `before_update`/`before_delete` listeners above only fire on an ORM
# unit-of-work flush of a *loaded instance* — they do NOT fire for Core statements
# executed via `session.execute(update(...))`/`delete(...))`, which this codebase
# already uses elsewhere (e.g. `update(EntitlementCycle)` in claim.py, `update(Decision)`
# in decisions.py). A raw SQL/Core UPDATE or DELETE on `detection_evidence` or
# `decision_evidence` would silently succeed without a DB-level guard. This mirrors
# the real SQLite trigger sealing already used for world receipts/GT rows
# (`app.db.world_provenance._seal`) and the evaluation journal
# (`app.db.evaluation_access._journal`): the trigger is the authoritative,
# caller-independent layer; the mapper listeners above remain a second (ORM-level)
# layer of defense-in-depth.
_SEALED_TABLES = (
    "audit_events",
    "cluster_snapshots",
    "decisions",
    "human_reviews",
    "llm_reviews",
)
_SEALED_EVIDENCE_TABLES = ("detection_evidence", "decision_evidence")

# The status machine is the sole legal writer of these two decision columns.
# Every other decision column is content and must remain byte-for-byte stable.
_DECISION_CONTENT_COLUMNS = (
    "account_id",
    "cluster_snapshot_id",
    "created_at",
    "detector_version",
    "policy_version",
    "risk_score",
    "proposed_action",
    "run_id",
    "idempotency_key",
)


def seal_evidence_tables(engine: Engine) -> None:
    """Install DB-level triggers for all append-only/immutable history tables.

    Called from `create_inference_schema` so every engine/session gets the same
    guard, regardless of caller (ORM instance flush, or raw Core/SQL statement).
    SQLite-only, like the other sealing helpers in this codebase; on other
    dialects the mapper-level guard above is the only enforcement layer.
    """
    if engine.dialect.name != "sqlite":
        return
    with engine.begin() as connection:
        for name in (*_SEALED_TABLES, *_SEALED_EVIDENCE_TABLES):
            connection.exec_driver_sql(
                f'CREATE TRIGGER IF NOT EXISTS "seal_{name}_delete" '
                f'BEFORE DELETE ON "{name}" '
                "BEGIN SELECT RAISE(ABORT, 'immutable history rows cannot be deleted'); END"
            )
            if name != "decisions":
                connection.exec_driver_sql(
                    f'CREATE TRIGGER IF NOT EXISTS "seal_{name}_update" '
                    f'BEFORE UPDATE ON "{name}" '
                    "BEGIN SELECT RAISE(ABORT, 'immutable history rows cannot be updated'); END"
                )

        # SQLite's IS operator is NULL-safe, which matters for optional decision
        # fields. Status/superseded_by are intentionally absent from this test so
        # the existing CAS lifecycle remains legal at the database layer.
        changed = " OR ".join(
            f'NEW."{column}" IS NOT OLD."{column}"'
            for column in _DECISION_CONTENT_COLUMNS
        )
        connection.exec_driver_sql(
            'CREATE TRIGGER IF NOT EXISTS "seal_decisions_content_update" '
            'BEFORE UPDATE ON "decisions" '
            f"WHEN {changed} "
            "BEGIN SELECT RAISE(ABORT, 'decision content is immutable'); END"
        )

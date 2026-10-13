"""Append-only typed audit trail (docs/architecture.md §3). Step 6.

The single writer for `audit_events`: every payload is a validated member of the
`app.schemas.audit` discriminated union, and the row is only ever appended — the
`audit_events` table is immutability-guarded (no update/delete, Step 1). This module
is the one place a service records a detection, decision, supersession, or reversal;
`app.db.decisions` delegates its decision-lifecycle audits here so the write path is
uniform.

Reads: `decision_audit_trail` is the ordered history the evaluation harness and the
analyst console query — the audit is typed, not an untyped blob.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.clock import now_utc
from app.models.inference import AuditEvent
from app.schemas.audit import AuditPayload, parse_audit_payload


def append_audit(
    session: Session, payload: AuditPayload | BaseModel, *, at: datetime | None = None
) -> AuditEvent:
    """Append one audit event, **validated at the boundary** (review finding 16): the
    payload is re-parsed through the discriminated union, so an unregistered event
    object (or one with a bad field) is rejected here — not trusted because a caller
    typed it. The row is append-only (the immutability guard rejects any mutation)."""
    validated = parse_audit_payload(payload.model_dump(mode="json"))
    row = AuditEvent(
        audit_id=uuid.uuid4().hex,
        timestamp=at or now_utc(),
        entity_type=validated.entity_type,
        entity_id=validated.entity_id,
        event_name=validated.event_name,
        actor_type=validated.actor_type,
        payload_json=validated.model_dump(mode="json"),
    )
    session.add(row)
    return row


def decision_audit_trail(session: Session, decision_id: str) -> list[AuditEvent]:
    """Every audit event about one decision, oldest first."""
    return list(
        session.scalars(
            select(AuditEvent)
            .where(AuditEvent.entity_id == decision_id)
            .order_by(AuditEvent.timestamp, AuditEvent.audit_id)
        ).all()
    )

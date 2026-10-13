"""Durable inference-only evidence; no reconstruction from later live inputs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.inference import DecisionEvidence, DetectionEvidence


class EvidenceConflict(ValueError):
    """A decision cannot be silently rebound to a different assessment."""


def persist_evidence(session: Session, payload: dict) -> DetectionEvidence:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    identity = hashlib.sha256(raw.encode()).hexdigest()
    row = session.get(DetectionEvidence, identity)
    if row is None:
        row = DetectionEvidence(
            evidence_id=identity,
            cluster_snapshot_id=payload["cluster_snapshot_id"],
            cycle_id=payload["cycle_id"],
            as_of=datetime.fromisoformat(payload["as_of"]),
            payload_json=json.loads(raw),
        )
        try:
            with session.begin_nested():
                session.add(row)
                session.flush()
        except IntegrityError:
            winner = session.get(DetectionEvidence, identity)
            if winner is None:
                raise
            row = winner
    return row


def associate_evidence(session: Session, decision_id: str, evidence_id: str) -> None:
    existing = session.get(DecisionEvidence, decision_id)
    if existing is not None:
        if existing.evidence_id != evidence_id:
            raise EvidenceConflict("Existing decision has different immutable evidence")
        return
    try:
        with session.begin_nested():
            session.add(DecisionEvidence(decision_id=decision_id, evidence_id=evidence_id))
            session.flush()
    except IntegrityError:
        winner = session.get(DecisionEvidence, decision_id)
        if winner is None:
            raise
        if winner.evidence_id != evidence_id:
            raise EvidenceConflict("Concurrent decision has different immutable evidence") from None


def decision_evidence(session: Session, decision_id: str) -> dict | None:
    association = session.get(DecisionEvidence, decision_id)
    if association is None:
        return None
    evidence = session.get(DetectionEvidence, association.evidence_id)
    if evidence is None:
        raise EvidenceConflict("Missing referenced detection evidence")
    # A detached deep copy prevents presentation code mutating the ORM JSON object.
    payload = json.loads(json.dumps(evidence.payload_json))
    payload["evidence_id"] = evidence.evidence_id
    return payload

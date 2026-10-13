"""Split-scoped evaluation access and a durable, fail-closed sealed-touch journal.

The journal lives in the GT database, so changing a CLI output/ledger path cannot
reset it. Starting consumes the world permanently even if scoring later fails.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Column, MetaData, String, Table, Text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

_metadata = MetaData()
_starts = Table(
    "evaluation_finalization_starts", _metadata,
    Column("world_id", String, primary_key=True),
    Column("analysis_hash", String, nullable=False),
    Column("started_at", String, nullable=False),
    Column("analysis_json", Text, nullable=False),
)
_reads = Table(
    "evaluation_finalization_reads", _metadata,
    Column("world_id", String, primary_key=True),
    Column("accessed_at", String, nullable=False),
)

_completions = Table(
    "evaluation_finalization_completions", _metadata,
    Column("world_id", String, primary_key=True),
    Column("completed_at", String, nullable=False),
    Column("artifact_json", Text, nullable=False),
)


class EvaluationAccessError(ValueError):
    """Access would violate world provenance or sealed-set discipline."""


@dataclass
class FinalizationGrant:
    world_id: str
    analysis_hash: str
    session: Session
    consumed: bool = False


def _memory(session: Session) -> bool:
    url = session.get_bind().engine.url
    return url.get_backend_name() == "sqlite" and url.database in (None, "", ":memory:")


def authorize_splits(
    inf_session: Session | None,
    gt_session: Session,
    splits: tuple[str, ...],
    *,
    finalization: FinalizationGrant | None = None,
) -> None:
    if not splits or set(splits) - {"train", "tune", "test"}:
        raise EvaluationAccessError("explicit train/tune/test splits are required")
    # Isolated synthetic fixtures only; persistent databases never enter this path.
    if inf_session is None and _memory(gt_session):
        from app.db.world_provenance import has_evaluation_receipt

        if not has_evaluation_receipt(gt_session):
            return
    if inf_session is None:
        raise EvaluationAccessError("durable evaluation requires both verified stores")
    from app.db.world_provenance import verify_world_pair

    world = verify_world_pair(inf_session, gt_session)
    if world.tier == "dev" and finalization is None:
        return
    if "test" not in splits:
        return  # SQL-scoped open development remains allowed on Tier 2.
    if (
        finalization is None or finalization.session is not gt_session
        or finalization.world_id != world.world_id or finalization.consumed
    ):
        raise EvaluationAccessError("sealed test labels require one-shot finalization")
    stored = gt_session.execute(
        _starts.select().where(_starts.c.world_id == world.world_id)
    ).mappings().one_or_none()
    if stored is None or stored["analysis_hash"] != finalization.analysis_hash:
        raise EvaluationAccessError("finalization grant has no matching durable start")
    try:
        gt_session.execute(_reads.insert().values(
            world_id=world.world_id, accessed_at=datetime.now(UTC).isoformat(),
        ))
        gt_session.commit()
    except IntegrityError as exc:
        gt_session.rollback()
        raise EvaluationAccessError("sealed labels were already accessed for this world") from exc
    finalization.consumed = True


def _journal(gt_session: Session) -> None:
    connection = gt_session.connection()
    _metadata.create_all(connection)
    if connection.dialect.name == "sqlite":
        for table in (_starts, _reads, _completions):
            for operation in ("UPDATE", "DELETE"):
                connection.exec_driver_sql(
                    f"CREATE TRIGGER IF NOT EXISTS {table.name}_{operation.lower()} "
                    f"BEFORE {operation} ON {table.name} BEGIN "
                    "SELECT RAISE(ABORT, 'evaluation journal is append-only'); END"
                )


def begin_finalization(
    inf_session: Session, gt_session: Session, analysis: dict[str, Any]
) -> FinalizationGrant:
    """Consume a world's single touch *before* any test-label read.

    Call after training/tuning and validating output paths. `analysis` includes the
    exact fitted model, policy, manifest, world and code identities. No retries
    under a new run id: interrupted attempts remain visible and consume the touch.
    """
    from app.db.world_provenance import verify_world_pair

    world = verify_world_pair(inf_session, gt_session)
    canonical = json.dumps(analysis, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    _journal(gt_session)
    try:
        gt_session.execute(_starts.insert().values(
            world_id=world.world_id, analysis_hash=digest,
            started_at=datetime.now(UTC).isoformat(), analysis_json=canonical,
        ))
        gt_session.commit()  # deliberately survives subsequent scorer failure
    except IntegrityError as exc:
        gt_session.rollback()
        raise EvaluationAccessError("this world already consumed its sealed touch") from exc
    return FinalizationGrant(world.world_id, digest, gt_session)


def complete_finalization(grant: FinalizationGrant, artifact: dict[str, Any]) -> None:
    if not grant.consumed:
        raise EvaluationAccessError("cannot complete before authorized sealed access")
    payload = json.dumps(artifact, sort_keys=True, allow_nan=False)
    grant.session.execute(_completions.insert().values(
        world_id=grant.world_id, completed_at=datetime.now(UTC).isoformat(),
        artifact_json=payload,
    ))
    grant.session.commit()

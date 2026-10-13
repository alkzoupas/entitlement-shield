"""Loader/evaluation control-plane provenance, never a detector dependency.

Only the inference receipt is safe for inference consumers. The evaluation
receipt holds the complete source manifest in the physically separate store.
Verification reads receipt metadata and inference inputs, not sealed labels.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, cast

from sqlalchemy import Column, Integer, MetaData, Table, Text, inspect, select
from sqlalchemy.orm import Session

from app.models.inference import (
    Account,
    AccountEvent,
    Appeal,
    EntitlementCycle,
    LinkageObservation,
)

LOADER_VERSION = "world-loader-v2"
WORLD_TIERS = frozenset({"dev", "tier2", "tier3", "tier4"})
# Loader-written appeals carry this source prefix; runtime appeals never do.
WORLD_APPEAL_SOURCE_PREFIX = "world:"
_INF = Table(
    "world_inference_receipt", MetaData(),
    Column("receipt_id", Integer, primary_key=True), Column("payload", Text, nullable=False),
)
_EVAL = Table(
    "world_evaluation_receipt", MetaData(),
    Column("receipt_id", Integer, primary_key=True), Column("payload", Text, nullable=False),
)
_BASE_TABLES = tuple(cast(Table, model.__table__) for model in (
    Account, AccountEvent, LinkageObservation, EntitlementCycle,
))


class WorldProvenanceError(ValueError):
    """Unverified, mismatched or modified evaluation inputs."""


@dataclass(frozen=True)
class VerifiedWorld:
    tier: str
    world_id: str
    world_version: str
    provenance: dict[str, Any]

    @property
    def source_checksums(self) -> dict[str, str]:
        return dict(self.provenance.get("checksums", {}))

    @property
    def reporting_status(self) -> dict[str, Any]:
        """Machine-readable policy carried with every evaluation artifact."""
        if self.tier == "tier4":
            return {
                "reportable": False,
                "code": "unregistered_nuance_probe",
                "reason": "Tier 4 is not reportable until its comparison is pre-registered",
            }
        if self.tier == "tier3":
            return {
                "reportable": False,
                "code": "open_tune_diagnostic",
                "reason": "Tier 3 is an open, small-N diagnostic; not sealed evidence",
            }
        if self.tier == "tier2":
            return {
                "reportable": False,
                "code": "same_family_synthetic",
                "reason": "Tier 2 uses a same-model-family synthetic data author",
            }
        return {
            "reportable": False,
            "code": "dev_tier",
            "reason": "Development-tier metrics are never reportable",
        }


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def manifest_identity(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(_json(manifest).encode()).hexdigest()


def manifest_tier(manifest: dict[str, Any]) -> str:
    """Derive the receipt tier from a verified source manifest.

    Dev manifests predate ``world_tier`` and use ``tier``; historical Tier 2
    manifests contain neither field.
    """
    value = manifest.get("world_tier", manifest.get("tier", "tier2"))
    if value not in WORLD_TIERS:
        raise WorldProvenanceError("Unknown manifest world tier")
    return str(value)


def _canonical(value: Any) -> Any:
    if isinstance(value, datetime):
        dt = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return dt.isoformat()
    if isinstance(value, Enum):
        return value.value
    return value


def inference_input_digest(session: Session) -> str:
    """Hash base observations only; derived detections/audits do not change it.

    Appending synthetic grants *does* change inputs. Evaluate an immutable loaded
    baseline rather than silently attaching its identity to a live modified world.
    """
    digest = hashlib.sha256()
    for table in _BASE_TABLES:
        digest.update((table.name + "\n").encode())
        for row in session.execute(select(table).order_by(*table.primary_key.columns)):
            digest.update((_json([_canonical(v) for v in row]) + "\n").encode())
    # World-authored appeals (Tier 4) are inputs too; runtime appeals share the table
    # and are excluded by source. The section is omitted when a world has none, so
    # digests of appeal-free worlds are unchanged; adding or deleting world appeals
    # after the receipt still changes the digest.
    appeals = cast(Table, Appeal.__table__)
    if not inspect(session.connection()).has_table(appeals.name):
        return digest.hexdigest()  # stores built before the appeals table existed
    world_appeals = session.execute(
        select(appeals)
        .where(appeals.c.source.like(f"{WORLD_APPEAL_SOURCE_PREFIX}%"))
        .order_by(appeals.c.appeal_id)
    ).all()
    if world_appeals:
        digest.update(b"world_appeals\n")
        for row in world_appeals:
            digest.update((_json([_canonical(v) for v in row]) + "\n").encode())
    return digest.hexdigest()


def _read(session: Session, table: Table) -> dict[str, Any] | None:
    if not inspect(session.connection()).has_table(table.name):
        return None
    payloads = session.scalars(select(table.c.payload)).all()
    if len(payloads) != 1:
        raise WorldProvenanceError("Expected exactly one immutable world receipt")
    try:
        payload = json.loads(payloads[0])
    except (TypeError, ValueError) as exc:
        raise WorldProvenanceError("Invalid world receipt") from exc
    if not isinstance(payload, dict):
        raise WorldProvenanceError("Invalid world receipt")
    return payload


def load_inference_receipt(session: Session) -> dict[str, Any] | None:
    """Safe fields only: tier, opaque identity, version and inference digest."""
    return _read(session, _INF)


def has_evaluation_receipt(gt_session: Session) -> bool:
    """True iff this GT store carries a world receipt — i.e. it is a real
    loaded world, not an isolated in-memory fixture. Single source of truth
    for sealed-access gates (review C-5); previously two independently
    maintained substring heuristics that could drift."""
    return inspect(gt_session.connection()).has_table(_EVAL.name)


def _seal(session: Session, tables: tuple[str, ...]) -> None:
    if session.get_bind().dialect.name != "sqlite":
        raise WorldProvenanceError("Receipt sealing currently supports SQLite only")
    for name in tables:
        for operation in ("INSERT", "UPDATE", "DELETE"):
            # Identifiers come exclusively from the constant table list below.
            session.connection().exec_driver_sql(
                f'CREATE TRIGGER IF NOT EXISTS "seal_{name}_{operation.lower()}" '
                f'BEFORE {operation} ON "{name}" '
                "BEGIN SELECT RAISE(ABORT, 'sealed world input is immutable'); END"
            )


def _seal_world_appeals(session: Session) -> None:
    """World appeals become immutable; runtime appeals in the same table stay writable."""
    if not inspect(session.connection()).has_table("appeals"):
        return
    guard = f"LIKE '{WORLD_APPEAL_SOURCE_PREFIX}%'"
    for operation, row in (("INSERT", "NEW"), ("UPDATE", "OLD"), ("DELETE", "OLD")):
        session.connection().exec_driver_sql(
            f'CREATE TRIGGER IF NOT EXISTS "seal_world_appeals_{operation.lower()}" '
            f'BEFORE {operation} ON "appeals" WHEN {row}.source {guard} '
            "BEGIN SELECT RAISE(ABORT, 'sealed world input is immutable'); END"
        )


def record_world_receipts(
    inf_session: Session, gt_session: Session, manifest: dict[str, Any], *, tier: str,
) -> VerifiedWorld:
    """Called by trusted loaders after materializing both planes, before commit.

    The caller owns transactions/publication. No label reads are needed to create
    a receipt: source checksums are verified by ingestion. GT rows and membership
    become immutable; the access ledger must live in a separate mutable table.
    """
    if tier not in WORLD_TIERS:
        raise WorldProvenanceError("Unknown world tier")
    if inf_session.get_bind() is gt_session.get_bind():
        raise WorldProvenanceError("World planes must use separate stores")
    inf_session.flush()
    gt_session.flush()
    _INF.create(inf_session.connection(), checkfirst=True)
    _EVAL.create(gt_session.connection(), checkfirst=True)
    if inf_session.scalar(select(_INF.c.receipt_id)) is not None:
        raise WorldProvenanceError("World receipt already exists")
    world_id = manifest_identity(manifest)
    version = str(manifest.get("world_version", f"dev-{manifest.get('preset', 'unknown')}"))
    safe = {
        "tier": tier, "world_id": world_id, "world_version": version,
        "inference_digest": inference_input_digest(inf_session), "loader_version": LOADER_VERSION,
    }
    inf_session.execute(_INF.insert().values(receipt_id=1, payload=_json(safe)))
    gt_session.execute(_EVAL.insert().values(
        receipt_id=1, payload=_json({**safe, "provenance": manifest}),
    ))
    _seal(inf_session, (_INF.name,))
    _seal_world_appeals(inf_session)
    sealed: tuple[str, ...] = (_EVAL.name, "ground_truth", "split_groups")
    if inspect(gt_session.connection()).has_table("appeal_truth"):
        sealed += ("appeal_truth",)
    _seal(gt_session, sealed)
    return VerifiedWorld(tier, world_id, version, manifest)


def verify_world_pair(inf_session: Session, gt_session: Session) -> VerifiedWorld:
    """Fail closed before label access if receipts or inference inputs disagree."""
    safe = load_inference_receipt(inf_session)
    sealed = _read(gt_session, _EVAL)
    if safe is None or sealed is None:
        raise WorldProvenanceError("Missing world receipts; re-ingest the verified source world")
    provenance = sealed.get("provenance")
    if not isinstance(provenance, dict):
        raise WorldProvenanceError("Missing evaluation provenance")
    if any(sealed.get(k) != v for k, v in safe.items()):
        raise WorldProvenanceError("Inference and evaluation stores belong to different worlds")
    if safe.get("tier") != manifest_tier(provenance):
        raise WorldProvenanceError("World receipt tier disagrees with sealed provenance")
    if safe.get("world_id") != manifest_identity(provenance):
        raise WorldProvenanceError("World manifest identity mismatch")
    if safe.get("inference_digest") != inference_input_digest(inf_session):
        raise WorldProvenanceError("Inference base inputs changed since ingestion")
    return VerifiedWorld(safe["tier"], safe["world_id"], safe["world_version"], provenance)

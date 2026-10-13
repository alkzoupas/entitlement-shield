"""Inference-store engine and session factory.

This module is wired ONLY to the inference database. It has no knowledge of the
ground-truth store; that lives behind the `ground_truth_*` capability modules,
which this module never imports (review 4 P1-4a).
"""

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.base import InferenceBase
from app.db.schema_guard import reject_opposite_plane
from app.db.sqlite_fk import enable_sqlite_foreign_keys


def make_inference_engine(url: str | None = None) -> Engine:
    target = url or settings.database_url
    parsed = make_url(target)
    # Claims must not spend their small request budget waiting for an occupied
    # SQLite pool.  Configure this once with the engine (never per request).
    # In-memory SQLite uses SingletonThreadPool, which does not accept this
    # QueuePool-only option.
    pool_options = (
        {"pool_timeout": 0}
        if parsed.get_backend_name() == "sqlite" and parsed.database not in (None, "", ":memory:")
        else {}
    )
    return enable_sqlite_foreign_keys(create_engine(target, future=True, **pool_options))


inference_engine: Engine = make_inference_engine()
InferenceSession: sessionmaker[Session] = sessionmaker(bind=inference_engine, future=True)


def create_inference_schema(engine: Engine | None = None) -> None:
    # Importing immutability registers the append-only/immutability guards (review 4
    # P1) as mapper listeners — global, so every session enforces them. Importing the
    # models registers them on InferenceBase.metadata.
    import app.db.immutability
    import app.models.claims  # noqa: F401
    import app.models.inference  # noqa: F401

    target = engine or inference_engine
    reject_opposite_plane(
        target, plane="inference", opposite_tables={"ground_truth", "split_groups"}
    )
    InferenceBase.metadata.create_all(target)
    # DB-level trigger sealing for the evidence tables (review F7-a): the mapper
    # listeners registered by importing app.db.immutability above only cover ORM
    # instance flushes, not raw Core/SQL UPDATE/DELETE statements.
    app.db.immutability.seal_evidence_tables(target)

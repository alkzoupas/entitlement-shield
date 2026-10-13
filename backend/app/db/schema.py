"""Schema creation — the dev-tier migration.

For local SQLite, `create_all` from the two Postgres-compatible bases IS the
migration (docs/architecture.md §5). Alembic is deferred until a Postgres target
exists; the models are written to be autogenerate-friendly when that day comes.
The two stores are created independently and never share metadata (review 4 P1-4a).

Physical separation is also *enforced* here (review 2 P1): if the inference and
ground-truth engines resolve to the same physical SQLite file, `create_all_schemas`
refuses to run, so the sealed `ground_truth` tables can never be created alongside
the inference tables in one file by a misconfiguration.
"""

import os

from sqlalchemy import Engine

from app.db.engine import create_inference_schema, inference_engine
from app.db.ground_truth_store import create_ground_truth_schema, ground_truth_engine


class StoreCollisionError(ValueError):
    """The inference and ground-truth stores resolve to the same physical store."""


def _physical_key(engine: Engine) -> str | None:
    """A comparable identity for an engine's physical store.

    Returns None for stores that can never collide (in-memory SQLite — each engine
    has its own private database even when the URL is identical). For on-disk SQLite
    the key is the normalized absolute real path, so equivalent relative/absolute
    paths compare equal. For non-SQLite targets the key is the full URL string
    (identical URLs collide; finer schema-isolation on real DBs is a TODO below).
    """
    url = engine.url
    if url.get_backend_name() == "sqlite":
        database = url.database
        # sqlite://  (database None/empty) and :memory: are in-memory: never collide.
        if not database or database == ":memory:":
            return None
        return os.path.realpath(os.path.abspath(database))
    # TODO: for real DBs (e.g. Postgres) enforce database/schema isolation rather
    # than only rejecting identical URLs. Until a Postgres target exists, an
    # identical URL is the only reachable collision.
    return url.render_as_string(hide_password=False)


def create_all_schemas(
    inference_engine_: Engine | None = None,
    ground_truth_engine_: Engine | None = None,
) -> None:
    inf = inference_engine_ or inference_engine
    gt = ground_truth_engine_ or ground_truth_engine

    inf_key = _physical_key(inf)
    gt_key = _physical_key(gt)
    if inf_key is not None and inf_key == gt_key:
        raise StoreCollisionError(
            "Inference and ground-truth stores resolve to the same physical store "
            f"({inf_key!r}). The sealed ground-truth tables must live in a SEPARATE "
            "store (review 4 P1-4a); set DATABASE_URL and GROUND_TRUTH_DATABASE_URL "
            "to distinct files/databases."
        )

    create_inference_schema(inf)
    create_ground_truth_schema(gt)

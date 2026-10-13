"""Strict transaction-scoped publication serialization, never fail open."""
from sqlalchemy import update
from sqlalchemy.orm import Session

from app.models.inference import PublicationLock


def lock_publication(session: Session) -> None:
    # Upsert is the first database operation: no read-before-write bootstrap race.
    dialect = session.get_bind().dialect.name
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert  # type: ignore[assignment]
    else:
        raise RuntimeError(f"Publication locking unsupported on {dialect}")
    session.execute(insert(PublicationLock).values(scope="inference", revision=0)
                    .on_conflict_do_nothing(index_elements=["scope"]))
    session.execute(update(PublicationLock).where(PublicationLock.scope == "inference")
                    .values(revision=PublicationLock.revision + 1))

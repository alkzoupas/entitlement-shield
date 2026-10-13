"""Enable SQLite foreign-key enforcement (review 5 P1).

SQLite ships with foreign-key enforcement OFF by default, so declared foreign keys
are inert unless every connection issues `PRAGMA foreign_keys=ON`. Without it the
dev SQLite store accepts orphan rows (a decision pointing at a missing account,
etc.) that Postgres would reject — weakening the promised Postgres-compatible model
and allowing broken audit/replay links.

`enable_sqlite_foreign_keys(engine)` attaches a `connect` listener that runs the
pragma on each new DBAPI connection. It is a no-op for non-SQLite engines, so it is
safe to call unconditionally from the engine factories and test fixtures.
"""

from sqlalchemy import Engine, event


def enable_sqlite_foreign_keys(engine: Engine) -> Engine:
    if engine.dialect.name != "sqlite":
        return engine

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: object, connection_record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine

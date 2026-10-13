"""Shared fixtures: fresh in-memory inference and ground-truth stores per test."""

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.engine import create_inference_schema
from app.db.ground_truth_store import create_ground_truth_schema
from app.db.sqlite_fk import enable_sqlite_foreign_keys


def _memory_engine() -> Engine:
    # StaticPool keeps one connection so the in-memory DB persists across sessions.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    # Match production engines: enforce declared foreign keys (review 5 P1).
    return enable_sqlite_foreign_keys(engine)


@pytest.fixture
def inference_engine() -> Engine:
    engine = _memory_engine()
    create_inference_schema(engine)
    return engine


@pytest.fixture
def inference_session(inference_engine: Engine):
    with Session(inference_engine, future=True) as session:
        yield session


@pytest.fixture
def gt_engine() -> Engine:
    engine = _memory_engine()
    create_ground_truth_schema(engine)
    return engine


@pytest.fixture
def gt_session(gt_engine: Engine):
    with Session(gt_engine, future=True) as session:
        yield session

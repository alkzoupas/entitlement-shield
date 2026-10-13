"""The ground-truth store — physically separate from the inference DB.

This module holds the sealed tables, their base, engine, and session. It is the
low-level store; callers use the WRITE-only sink (`ground_truth_writer`) or the
READ handle (`ground_truth_reader`), never this module directly — except those two
and the schema builder.

The capability boundary (enforced by test in Step 1): no module under
`app/services/` except `evaluation.py` may import any `app.db.ground_truth*`
module, and detector services get neither handle nor a session that can see this
store. `ground_truth` (labels) and `split_groups` (the evaluation-only grouping,
review 4 P1-4c) both live here so the whole evaluation-only data plane is gated in
one place.
"""

from sqlalchemy import Engine, Enum, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.core.config import settings
from app.db.schema_guard import reject_opposite_plane
from app.db.sqlite_fk import enable_sqlite_foreign_keys
from app.models.enums import AppealTruthKind, GroundTruthLabel


class GroundTruthBase(DeclarativeBase):
    """Separate MetaData from InferenceBase — cannot coexist in one schema."""


class GroundTruth(GroundTruthBase):
    __tablename__ = "ground_truth"

    cluster_id: Mapped[str] = mapped_column(String, primary_key=True)
    scenario_family: Mapped[str] = mapped_column(String, index=True)
    label: Mapped[GroundTruthLabel] = mapped_column(
        Enum(GroundTruthLabel, native_enum=False, validate_strings=True)
    )
    notes: Mapped[str | None] = mapped_column(String, nullable=True)


class SplitGroup(GroundTruthBase):
    """Evaluation-only split grouping — the transitive union of related accounts
    across the full dataset, robust to clusters that merge later (review 4 P1-4c).
    Used ONLY for split assignment and metric grouping, never as inference input.
    Distinct from a `cluster_snapshot_id` evaluation unit (features.md §5).
    """

    __tablename__ = "split_groups"

    account_id: Mapped[str] = mapped_column(String, primary_key=True)
    split_group_id: Mapped[str] = mapped_column(String, index=True)


class AppealTruth(GroundTruthBase):
    """Sealed truth about a world-authored appeal (Tier 4): whether the appeal text
    is consistent with, contradicted by, or trying to steer the review. Keyed by the
    inference-plane appeal event id; evaluation-only, like `ground_truth`."""

    __tablename__ = "appeal_truth"

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    kind: Mapped[AppealTruthKind] = mapped_column(
        Enum(AppealTruthKind, native_enum=False, validate_strings=True)
    )
    notes: Mapped[str | None] = mapped_column(String, nullable=True)


def make_ground_truth_engine(url: str | None = None) -> Engine:
    return enable_sqlite_foreign_keys(
        create_engine(url or settings.ground_truth_database_url, future=True)
    )


ground_truth_engine: Engine = make_ground_truth_engine()
GroundTruthSession: sessionmaker[Session] = sessionmaker(
    bind=ground_truth_engine, future=True
)


def create_ground_truth_schema(engine: Engine | None = None) -> None:
    target = engine or ground_truth_engine
    # Importing inference models here would blur the plane boundary, so retain the
    # stable foundational inference markers explicitly.
    reject_opposite_plane(target, plane="ground-truth", opposite_tables={"accounts"})
    GroundTruthBase.metadata.create_all(target)

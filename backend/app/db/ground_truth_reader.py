"""FULL sealed read handle for ground truth — importable ONLY by evaluation.py
(review 4 P1-4a, tightened by review 3 P1).

`load_labels` returns the binary abuse label and `load_split_groups` the
evaluation-only grouping; both are for `evaluation.py` alone. The family-only view
(`load_scenario_families`) lives in the SEPARATE `ground_truth_family_reader`
module so the future family-classifier can hold family access without ever
importing the binary-label operation.
"""

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.db.ground_truth_store import AppealTruth, GroundTruth, SplitGroup
from app.models.enums import AppealTruthKind, GroundTruthLabel


def _label_query(
    session: Session, allowed_group_ids: set[str] | None
) -> Select[tuple[GroundTruth]]:
    if allowed_group_ids is not None:
        return select(GroundTruth).where(GroundTruth.cluster_id.in_(allowed_group_ids))
    # Legacy isolated fixtures only. Durable/world-receipted callers must supply
    # the SQL scope resolved by evaluation's access gate.
    from app.db.world_provenance import has_evaluation_receipt

    url = session.get_bind().engine.url
    if (url.get_backend_name() == "sqlite" and url.database in (None, "", ":memory:")
            and not has_evaluation_receipt(session)):
        return select(GroundTruth)
    raise ValueError("label reads require an explicit authorized split-group scope")


def load_labels(
    session: Session, allowed_group_ids: set[str] | None = None
) -> dict[str, GroundTruthLabel]:
    rows = session.scalars(_label_query(session, allowed_group_ids)).all()
    return {row.cluster_id: row.label for row in rows}


def load_appeal_kinds(session: Session, event_ids: set[str]) -> dict[str, AppealTruthKind]:
    """Sealed Tier 4 appeal truth for exactly `event_ids` — the caller (evaluation's
    access gate) resolves which appeals fall inside the authorized splits."""
    if not event_ids:
        return {}
    rows = session.scalars(select(AppealTruth).where(AppealTruth.event_id.in_(event_ids))).all()
    return {row.event_id: row.kind for row in rows}


def load_split_groups(session: Session) -> dict[str, str]:
    rows = session.scalars(select(SplitGroup)).all()
    return {row.account_id: row.split_group_id for row in rows}


def load_scenario_families(
    session: Session, allowed_group_ids: set[str] | None = None
) -> dict[str, str]:
    """Scenario family per cluster, for evaluation's fairness slices (eval §8). The
    sealed reader already sees the binary label, so exposing the family here adds no
    leak; the classifier uses the separate family-only reader instead."""
    rows = session.scalars(_label_query(session, allowed_group_ids)).all()
    return {row.cluster_id: row.scenario_family for row in rows}

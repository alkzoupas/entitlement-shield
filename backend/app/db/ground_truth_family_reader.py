"""Family-only read handle for ground truth (review 3 P1).

This is the ONLY ground-truth surface the future family-classifier
(`classify_families.py`, features.md §7.3) is permitted. It returns the scenario
*family* of a cluster and NOTHING that reveals the binary abuse label. The binary
label lives behind `ground_truth_reader.load_labels`, in a separate module the
classifier may not import — so possession of this capability cannot be turned into
label access (contrast: the old single reader exposed both from one module).
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.ground_truth_store import GroundTruth, SplitGroup


def load_scenario_families(session: Session) -> dict[str, str]:
    """Family label only — NOT the abuse label (features.md §7.3)."""
    rows = session.execute(select(GroundTruth.cluster_id, GroundTruth.scenario_family))
    return {cluster_id: family for cluster_id, family in rows}


def load_account_families(session: Session) -> dict[str, str]:
    """`account_id → scenario_family`, family-only.

    Joins `SplitGroup` membership (`account → split_group_id`) to
    `GroundTruth.scenario_family` on `split_group_id == cluster_id` (generator §),
    so a computed cluster's members can be attached to their scenario family without
    ever touching `GroundTruth.label` — the classifier's §7.3 leakage boundary. The
    binary abuse label lives behind `ground_truth_reader.load_labels`, a module this
    one never imports.
    """
    rows = session.execute(
        select(SplitGroup.account_id, GroundTruth.scenario_family).join(
            GroundTruth, SplitGroup.split_group_id == GroundTruth.cluster_id
        )
    )
    return {account_id: family for account_id, family in rows}

"""Transparent additive risk scorer + the stable contribution export. Step 5.

The score is nothing but the sum of the §3 feature contributions that fired,
clipped to `[0, 1]`, thresholded at `SCORE_THRESHOLD` to decide whether an action
is *proposed at all* (features.md §4.3, policy §7). There is no learned model and
no hidden state — the contributions ARE the explanation.

The **contribution export** (review 4 P1-5, features.md §7.3) is the sole input the
Phase-2 family classifier reads, so it is defined and made byte-stable here rather
than improvised later: one record per `(eval key, as_of, snapshot hash, feature
id, raw value, contribution, fired, family, rung)`, plus the detector-config and
`feature-manifest` hashes. Step 8 minted `config/feature-manifest.json`, so
`feature_manifest_hash` now defaults to its content hash (`MANIFEST_HASH`); the
detector supplies the per-run `detector_config_hash`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from app.services.feature_manifest import MANIFEST_HASH
from app.services.features import (
    FEATURE_ORDER,
    ClusterCycleUnit,
    ClusterScore,
    EvidenceFamily,
    FeatureContribution,
)

SCORE_THRESHOLD = 0.60  # propose an action above this (policy §7, config/thresholds.yaml)


@dataclass(frozen=True)
class RiskAssessment:
    """The scorer's verdict for one cluster-cycle — the input to policy_engine."""

    risk_score: float  # clipped to [0, 1]
    above_threshold: bool
    firing_families: frozenset[EvidenceFamily]
    contributions: tuple[FeatureContribution, ...]  # in FEATURE_ORDER


def assess(score: ClusterScore) -> RiskAssessment:
    """Sum the fired contributions, clip, threshold (features.md §4.3)."""
    risk = score.total_contribution  # already sum-of-fired clipped to [0, 1]
    return RiskAssessment(
        risk_score=risk,
        above_threshold=risk >= SCORE_THRESHOLD,
        firing_families=score.firing_families,
        contributions=score.contributions,
    )


# --- Contribution export (features.md §7.3) ----------------------------------


@dataclass(frozen=True)
class ContributionRecord:
    """One deterministic export row. `feature_manifest_hash` carries the frozen
    manifest content hash (features.md §6.2); `detector_config_hash` is the detector's
    per-run config fingerprint."""

    cluster_snapshot_id: str
    cycle_id: str
    as_of: str  # ISO-8601
    snapshot_hash: str
    feature_id: str
    raw_value: float
    contribution: float
    fired: bool
    family: int | None
    rung: str
    detector_config_hash: str
    feature_manifest_hash: str


def export_contributions(
    unit: ClusterCycleUnit,
    score: ClusterScore,
    *,
    snapshot_hash: str = "",
    detector_config_hash: str = "",
    feature_manifest_hash: str = MANIFEST_HASH,
) -> list[ContributionRecord]:
    """Emit one record per feature, in FEATURE_ORDER — the stable classifier input.
    Ordering and field set are fixed so the serialized export is byte-reproducible.

    The score MUST carry exactly the FEATURE_ORDER features, once each: a missing,
    duplicate, or unknown feature id is rejected (review finding 15) so the export
    cannot silently ship an incomplete contribution set."""
    seen = [c.feature_id for c in score.contributions]
    if seen != list(FEATURE_ORDER):
        expected = set(FEATURE_ORDER)
        got = set(seen)
        raise ValueError(
            "contribution set must be exactly FEATURE_ORDER once each; "
            f"missing={sorted(expected - got)} unknown={sorted(got - expected)} "
            f"order_ok={seen == list(FEATURE_ORDER)}"
        )
    order = {fid: i for i, fid in enumerate(FEATURE_ORDER)}
    rows = [
        ContributionRecord(
            cluster_snapshot_id=unit.cluster_snapshot_id,
            cycle_id=unit.cycle_id,
            as_of=unit.as_of.isoformat(),
            snapshot_hash=snapshot_hash,
            feature_id=c.feature_id,
            raw_value=c.raw_value,
            contribution=c.contribution,
            fired=c.fired,
            family=int(c.family) if c.family is not None else None,
            rung=c.rung.name,
            detector_config_hash=detector_config_hash,
            feature_manifest_hash=feature_manifest_hash,
        )
        for c in score.contributions
    ]
    rows.sort(key=lambda r: order[r.feature_id])
    return rows


def contribution_bytes(rows: list[ContributionRecord]) -> bytes:
    """Canonical serialization of an export — deterministic for identical inputs."""
    payload = [
        {
            "cluster_snapshot_id": r.cluster_snapshot_id,
            "cycle_id": r.cycle_id,
            "as_of": r.as_of,
            "snapshot_hash": r.snapshot_hash,
            "feature_id": r.feature_id,
            "raw_value": r.raw_value,
            "contribution": r.contribution,
            "fired": r.fired,
            "family": r.family,
            "rung": r.rung,
            "detector_config_hash": r.detector_config_hash,
            "feature_manifest_hash": r.feature_manifest_hash,
        }
        for r in rows
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

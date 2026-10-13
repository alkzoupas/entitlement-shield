"""Risk scorer + contribution export (docs/features.md §4.3/§7.3, plan Step 5).

Covers: additive score = clipped sum of fired contributions; the 0.60 threshold;
the stable, byte-reproducible contribution export; and the seed-weight/YAML drift
guard.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime

import yaml

from app.models.enums import CapacityBand
from app.services.banding import BandingResult
from app.services.features import (
    FEATURE_ORDER,
    SEED_WEIGHTS,
    ClusterCycleUnit,
    ClusterScore,
    EvidenceFamily,
    FeatureContribution,
    Rung,
)
from app.services.risk_scorer import (
    SCORE_THRESHOLD,
    assess,
    contribution_bytes,
    export_contributions,
)

_AS_OF = datetime(2026, 6, 1, tzinfo=UTC)
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _band() -> BandingResult:
    return BandingResult(
        qualified_stream_count=1,
        allowance=25,
        band=CapacityBand.SINGLE,
        ambiguous=False,
        claimed=75,
        concurrency_passers=(),
        rate_passers=(),
    )


def _score(pairs: list[tuple[str, float, EvidenceFamily | None, Rung]]) -> ClusterScore:
    """Build a ClusterScore from (feature_id, contribution, family, rung) tuples."""
    contribs = tuple(
        FeatureContribution.make(fid, 1.0, contribution, family, rung)
        for fid, contribution, family, rung in pairs
    )
    return ClusterScore(contributions=contribs, banding=_band(), rung=Rung.S2)


def test_assess_sums_and_thresholds() -> None:
    score = _score(
        [
            ("F-DEV-1", 0.30, EvidenceFamily.DEVICE, Rung.S1),
            ("F-DOM-1", 0.15, EvidenceFamily.CONTACT_DOMAIN, Rung.S0),
            ("F-CYC-1", 0.30, EvidenceFamily.CYCLE, Rung.S2),
        ]
    )
    a = assess(score)
    assert abs(a.risk_score - 0.75) < 1e-9
    assert a.above_threshold is True
    assert a.firing_families == {
        EvidenceFamily.DEVICE,
        EvidenceFamily.CONTACT_DOMAIN,
        EvidenceFamily.CYCLE,
    }


def test_assess_below_threshold() -> None:
    score = _score([("F-DOM-1", 0.15, EvidenceFamily.CONTACT_DOMAIN, Rung.S0)])
    a = assess(score)
    assert a.risk_score < SCORE_THRESHOLD
    assert a.above_threshold is False


def test_assess_clips_to_one() -> None:
    score = _score(
        [
            ("F-DEV-1", 0.30, EvidenceFamily.DEVICE, Rung.S1),
            ("F-PAY-1", 0.20, EvidenceFamily.PAYMENT, Rung.S1),
            ("F-CON-1", 0.15, EvidenceFamily.CONTACT_DOMAIN, Rung.S1),
            ("F-CYC-1", 0.30, EvidenceFamily.CYCLE, Rung.S2),
            ("F-DOM-1", 0.15, EvidenceFamily.CONTACT_DOMAIN, Rung.S0),
        ]
    )
    assert assess(score).risk_score == 1.0  # 1.10 clipped


def _unit() -> ClusterCycleUnit:
    return ClusterCycleUnit(
        cluster_snapshot_id="snap-abc",
        cycle_id="snap-abc::cyc-2026-04-01",
        as_of=_AS_OF,
        window_start=_AS_OF,
        window_end=_AS_OF,
    )


def _full_score(overrides: dict[str, tuple[float, EvidenceFamily | None, Rung]]) -> ClusterScore:
    """A ClusterScore covering EXACTLY FEATURE_ORDER (the export contract, finding 15)."""
    _fam = {
        "F-DEV-1": EvidenceFamily.DEVICE, "F-DEV-2": EvidenceFamily.DEVICE,
        "F-PAY-1": EvidenceFamily.PAYMENT, "F-CON-1": EvidenceFamily.CONTACT_DOMAIN,
        "F-DOM-1": EvidenceFamily.CONTACT_DOMAIN, "F-VER-1": EvidenceFamily.VERIFICATION,
        "F-CYC-1": EvidenceFamily.CYCLE, "F-CYC-2": EvidenceFamily.CYCLE,
        "F-CYC-3": EvidenceFamily.CYCLE, "F-CYC-4": EvidenceFamily.CYCLE,
        "F-DOM-2": None, "F-AGE-1": None,
    }
    _s1 = {"F-DEV-1", "F-DEV-2", "F-PAY-1", "F-CON-1"}
    _rung = {
        "F-DOM-1": Rung.S0, "F-VER-1": Rung.S0, "F-DOM-2": Rung.S0, "F-AGE-1": Rung.S0
    }
    contribs = []
    for fid in FEATURE_ORDER:
        default = (0.0, _fam[fid], _rung.get(fid, Rung.S1 if fid in _s1 else Rung.S2))
        contribution, family, rung = overrides.get(fid, default)
        contribs.append(FeatureContribution.make(fid, 1.0, contribution, family, rung))
    return ClusterScore(contributions=tuple(contribs), banding=_band(), rung=Rung.S2)


def test_export_is_ordered_and_byte_stable() -> None:
    score = _full_score(
        {
            "F-CYC-1": (0.30, EvidenceFamily.CYCLE, Rung.S2),
            "F-DEV-1": (0.30, EvidenceFamily.DEVICE, Rung.S1),
            "F-DOM-2": (-0.15, None, Rung.S0),
        }
    )
    rows = export_contributions(_unit(), score, snapshot_hash="hash-1")
    assert [r.feature_id for r in rows] == list(FEATURE_ORDER)  # exact coverage
    ids = [r.feature_id for r in rows]
    assert ids.index("F-DEV-1") < ids.index("F-CYC-1")
    dev = next(r for r in rows if r.feature_id == "F-DEV-1")
    assert dev.family == 1 and dev.rung == "S1" and dev.snapshot_hash == "hash-1"
    benign = next(r for r in rows if r.feature_id == "F-DOM-2")
    assert benign.family is None
    again = export_contributions(_unit(), score, snapshot_hash="hash-1")
    assert contribution_bytes(rows) == contribution_bytes(again)


def test_export_rejects_incomplete_feature_set() -> None:
    partial = _score([("F-DEV-1", 0.30, EvidenceFamily.DEVICE, Rung.S1)])
    import pytest

    with pytest.raises(ValueError, match="FEATURE_ORDER"):
        export_contributions(_unit(), partial)


def test_seed_weights_match_yaml() -> None:
    """The seed weights in code and config/thresholds.yaml must not drift."""
    cfg = yaml.safe_load((_REPO_ROOT / "config" / "thresholds.yaml").read_text())
    assert cfg["feature_weights"] == SEED_WEIGHTS

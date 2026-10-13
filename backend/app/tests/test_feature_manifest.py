"""The freeze-artifact gate (features.md §6.2, §6.3; plan Step 8).

Proves the machine-checkable claim behind `features-frozen`: the committed
`config/feature-manifest.json` regenerates byte-for-byte from the live code, it
carries every §6.1 firing-affecting field for every feature, its content hash is
self-consistent and the one stamped into exports, and it is *sensitive* — moving
any firing-affecting value (a weight, a predicate parameter, the threshold, the
masking version) moves the hash. "Set unchanged" is thus proven, not asserted.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
import yaml
from sqlalchemy.orm import Session

from app.models.enums import EmailDomainType, Reliability, SignalType
from app.services import banding as B
from app.services import clustering as C
from app.services import feature_manifest as M
from app.services import features as F
from app.services.features import (
    FEATURE_ORDER,
    ClusterScore,
    EvidenceFamily,
    FeatureContribution,
    Rung,
    score_cluster,
)
from app.services.risk_scorer import export_contributions
from app.tests.support import Builder

_REPO_ROOT = M.MANIFEST_PATH.parents[1]

_REQUIRED_FIELDS = {
    "id",
    "inputs",
    "sign",
    "evidence_family",
    "family_name",
    "ablation_rung",
    "firing_predicate",
    "predicate_params",
    "scaling",
    "cap",
    "missing_value_semantics",
    "firing_threshold",
    "weight",
}


#: The committed `features-frozen-v2` content hash, quoted verbatim from
#: `docs/decisions.md §8.1`. It is a LITERAL on purpose: every other test in this
#: file compares the manifest against freshly-regenerated values, so a change that
#: moved both the code and the artifact together would regenerate consistently and
#: pass them all. Only a hardcoded expectation catches "the freeze itself moved".
FROZEN_FEATURE_MANIFEST_HASH = (
    "fe21a7b052bea913fcbb880037a85c022d0e2be9d855a426bd449f60172a2021"
)


def test_manifest_regenerates_byte_for_byte() -> None:
    """The Step 8 gate: the committed artifact == regeneration from live config."""
    committed = M.MANIFEST_PATH.read_bytes()
    assert committed == M.manifest_bytes()


def test_feature_manifest_hash_unchanged() -> None:
    """P3.9 scope tripwire (decisions.md §12 L-8: "a separate artifact from
    `feature-manifest.json`, **which must not change**").

    P3.9 adds a second, independent manifest (`config/reviewer-manifest.json`) and
    a `config/costs.yaml` `llm:` block. Neither may perturb anything the detector
    fires on. If this test ever fails, detector-firing config moved — revert the
    change that moved it rather than re-freezing, since re-freezing is a §6.3
    change-control event (new `features-frozen-vN` tag), not a build step."""
    assert M.MANIFEST_HASH == FROZEN_FEATURE_MANIFEST_HASH
    committed = json.loads(M.MANIFEST_PATH.read_text())
    assert committed["content_hash"] == FROZEN_FEATURE_MANIFEST_HASH
    # …and the committed file is still exactly what the live code regenerates, so
    # the literal above names a reproducible artifact rather than a stale string.
    assert M.MANIFEST_PATH.read_bytes() == M.manifest_bytes()


def test_manifest_covers_exactly_feature_order() -> None:
    manifest = M.build_manifest()
    assert manifest["feature_order"] == list(FEATURE_ORDER)
    assert set(manifest["features"]) == set(FEATURE_ORDER)


def test_every_feature_carries_all_frozen_fields() -> None:
    manifest = M.build_manifest()
    for fid in FEATURE_ORDER:
        entry = manifest["features"][fid]
        assert set(entry) == _REQUIRED_FIELDS, fid
        assert entry["id"] == fid
        assert entry["firing_threshold"] == F.FIRING_THRESHOLD


def test_content_hash_is_self_consistent_and_exported() -> None:
    committed = json.loads(M.MANIFEST_PATH.read_text())
    embedded = committed.pop("content_hash")
    # Recomputing the hash over the committed content (minus the hash) reproduces it.
    assert M.content_hash(committed) == embedded == M.MANIFEST_HASH


def test_global_frozen_fields_present() -> None:
    manifest = M.build_manifest()
    assert manifest["firing_threshold"] == F.FIRING_THRESHOLD == 0.05
    assert manifest["masking_aggregation_version"] == F.MASKING_AGGREGATION_VERSION
    assert manifest["manifest_version"] == M.MANIFEST_VERSION


def test_manifest_weights_match_code_and_yaml() -> None:
    """No firing-affecting value is mutable behind the manifest's back: the frozen
    weights in the manifest, the code, and config/thresholds.yaml all agree."""
    manifest = M.build_manifest()
    cfg = yaml.safe_load((_REPO_ROOT / "config" / "thresholds.yaml").read_text())
    for fid in FEATURE_ORDER:
        assert manifest["features"][fid]["weight"] == F.SEED_WEIGHTS[fid]
        assert cfg["feature_weights"][fid] == F.SEED_WEIGHTS[fid]


def test_weight_change_moves_hash(monkeypatch) -> None:
    base = M.content_hash()
    monkeypatch.setitem(F.SEED_WEIGHTS, "F-DEV-1", 0.99)
    assert M.content_hash() != base


def test_predicate_param_change_moves_hash(monkeypatch) -> None:
    base = M.content_hash()
    monkeypatch.setattr(F, "_VER_FAILURE_MIN", 5)
    assert M.content_hash() != base


def test_masking_version_change_moves_hash(monkeypatch) -> None:
    base = M.content_hash()
    monkeypatch.setattr(F, "MASKING_AGGREGATION_VERSION", "v2")
    assert M.content_hash() != base


def test_manifest_hash_stamped_into_export() -> None:
    ts = datetime(2025, 1, 1, tzinfo=UTC)
    unit = F.ClusterCycleUnit(
        cluster_snapshot_id="s", cycle_id="c", as_of=ts, window_start=ts, window_end=ts,
    )
    contribs = tuple(
        FeatureContribution.make(fid, 0.0, 0.0, EvidenceFamily.DEVICE, Rung.S0)
        for fid in FEATURE_ORDER
    )
    score = ClusterScore(contributions=contribs, banding=None, rung=Rung.S0)  # type: ignore[arg-type]
    rows = export_contributions(unit, score)
    assert rows and all(r.feature_manifest_hash == M.MANIFEST_HASH for r in rows)


# --- Review finding 1: firing_config covers dependencies outside features.py ---


@pytest.mark.parametrize(
    "mod,attr,newval",
    [
        (B, "N_OVERLAP", 99),
        (B, "WINDOW_DAYS", 999),
        (B, "PER_STREAM_ALLOWANCE", 999),
        (B, "RATE_HOURLY_CREDITS", 999),
        (B, "DIURNAL_D", 0.99),
        (C, "HUB_ACCOUNT_THRESHOLD", 999),
        (C, "MAX_COMPONENT_SIZE", 999),
        (C, "DECAY_HALFLIFE_DAYS", 999.0),
        (C, "MAX_AGE_DAYS", 9999),
    ],
)
def test_firing_config_dependency_moves_hash(monkeypatch, mod, attr, newval) -> None:
    """Every firing-affecting banding/clustering constant is in the hash: mutating it
    changes content_hash (the P0-1 gap — the hash previously ignored these)."""
    base = M.content_hash()
    monkeypatch.setattr(mod, attr, newval)
    assert M.content_hash() != base, f"{mod.__name__}.{attr} not covered by the manifest hash"


def test_firing_config_block_present() -> None:
    fc = M.build_manifest()["firing_config"]
    assert fc["banding"]["window_days"] == B.WINDOW_DAYS
    assert fc["banding"]["per_stream_allowance"] == B.PER_STREAM_ALLOWANCE
    assert fc["clustering"]["hub_account_threshold"] == C.HUB_ACCOUNT_THRESHOLD
    assert fc["clustering"]["max_age_days"] == C.MAX_AGE_DAYS


# --- Review finding 2: source fingerprint locks executable semantics -----------


def test_firing_code_fingerprint_present_and_matches_source() -> None:
    manifest = M.build_manifest()
    assert manifest["firing_code_fingerprint"] == M._firing_code_fingerprint()


def test_source_edit_to_firing_module_moves_hash(tmp_path, monkeypatch) -> None:
    """A semantic change with no constant edit still re-freezes: fingerprint the
    source, so editing a firing module's bytes moves the hash."""
    base = M.content_hash()
    # Simulate a source edit by pointing one firing module at altered bytes.
    fake = tmp_path / "features_edited.py"
    original = M.Path(F.__file__).read_bytes()
    fake.write_bytes(original + b"\n# semantic change\n")
    monkeypatch.setattr(F, "__file__", str(fake))
    assert M.content_hash() != base


# --- Review finding 2/3: manifest tags/scaling agree with runtime behavior ------


def _probe_score(session: Session):
    """A minimal 2-account cluster scored at S2 — every feature is emitted (with its
    family/rung tag) whether or not it fires."""
    b = Builder(session)
    a0 = b.account("mf-0", domain=EmailDomainType.DISPOSABLE)
    a1 = b.account("mf-1", domain=EmailDomainType.DISPOSABLE)
    for aid in (a0, a1):
        b.observation(aid, SignalType.STABLE_DEVICE, "dev-mf", reliability=Reliability.HIGH)
    b.flush()
    return score_cluster(session, [a0, a1], datetime(2026, 1, 1, tzinfo=UTC), Rung.S2)


def test_manifest_family_rung_sign_match_runtime(inference_session: Session) -> None:
    """The hand-maintained family/rung/sign in the manifest must match what the feature
    code actually emits (review finding 2 — no silent tag drift)."""
    score = _probe_score(inference_session)
    manifest = M.build_manifest()["features"]
    by_id = {c.feature_id: c for c in score.contributions}
    for fid in FEATURE_ORDER:
        c = by_id[fid]
        entry = manifest[fid]
        emitted_family = int(c.family) if c.family is not None else None
        assert entry["evidence_family"] == emitted_family, f"{fid} family drift"
        assert entry["ablation_rung"] == c.rung.name, f"{fid} rung drift"
        expected_sign = "-" if F.SEED_WEIGHTS[fid] < 0 else "+"
        assert entry["sign"] == expected_sign, f"{fid} sign drift"


def test_benign_features_use_indicator_scaling_not_fraction(inference_session: Session) -> None:
    """Review finding 3: F-DOM-2/F-AGE-1 emit the FULL weight once the majority predicate
    passes (raw_value is the fraction, not a multiplier). A cluster where exactly half
    qualify must still emit the full weight, matching the corrected manifest string."""
    b = Builder(inference_session)
    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    # Two accounts, one education with routine logins (qualifies), one disposable (not) →
    # fraction 0.5, meets the majority floor.
    edu = b.account("bn-edu", domain=EmailDomainType.EDUCATION, created=as_of - timedelta(days=400))
    b.account("bn-dis", domain=EmailDomainType.DISPOSABLE, created=as_of - timedelta(days=400))
    for k in range(3):
        b.event(edu, as_of - timedelta(days=30 + k), F.EventType.LOGIN)
    b.flush()
    score = score_cluster(inference_session, ["bn-edu", "bn-dis"], as_of, Rung.S2)
    dom2 = next(c for c in score.contributions if c.feature_id == "F-DOM-2")
    # raw_value is the qualifying fraction; contribution is the FULL weight, not fraction*weight.
    assert dom2.raw_value == 0.5
    assert dom2.contribution == F.SEED_WEIGHTS["F-DOM-2"]  # full weight, not 0.5 * weight
    assert M.build_manifest()["features"]["F-DOM-2"]["scaling"].startswith("indicator(predicate)")


# --- Review finding 7: runtime resolves to a verified committed manifest --------


def test_verified_manifest_hash_accepts_committed() -> None:
    assert M.verified_manifest_hash() == M.MANIFEST_HASH


def test_verified_manifest_hash_rejects_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(M, "MANIFEST_PATH", tmp_path / "absent.json")
    with pytest.raises(M.ManifestDrift):
        M.verified_manifest_hash()


def test_verified_manifest_hash_rejects_stale(monkeypatch, tmp_path) -> None:
    stale = tmp_path / "stale.json"
    stale.write_bytes(M.manifest_bytes().replace(b'"window_days": 60', b'"window_days": 61'))
    monkeypatch.setattr(M, "MANIFEST_PATH", stale)
    with pytest.raises(M.ManifestDrift):
        M.verified_manifest_hash()


def test_verified_manifest_hash_rejects_malformed(monkeypatch, tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"{ not json")
    monkeypatch.setattr(M, "MANIFEST_PATH", bad)
    with pytest.raises(M.ManifestDrift):
        M.verified_manifest_hash()

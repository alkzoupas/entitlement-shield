"""Tier 2 evaluation world — committed-world invariants + loader behavior.

Two groups:

* **World invariants** read the committed parquets under ``data/tier2/`` and
  re-assert the forbidden-list generation constraints (docs/tier2-decisions.md):
  schema/enum conformance, no ground-truth leak into the event plane, referential
  integrity, scale/prevalence bands, cross-label spread of sensitive values, cycle
  consistency, and — the audit anchor — that each parquet's sha256 matches the
  checksum recorded in ``manifest.json``.
* **Loader** tests exercise :mod:`app.db.tier2_loader`: the value-hash equality
  contract, the small helpers, and a tiny end-to-end load into fresh separate
  stores with the sealed labels reachable only through the read path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import tier2_loader
from app.db.engine import make_inference_engine
from app.db.evaluation_access import EvaluationAccessError, begin_finalization
from app.db.ground_truth_reader import load_split_groups
from app.db.ground_truth_store import GroundTruth, make_ground_truth_engine
from app.db.tier2_loader import (
    _opt_cost_tier,
    _opt_credits,
    _utc,
    hash_signal_value,
    load_tier2,
)
from app.db.world_provenance import WorldProvenanceError, verify_world_pair
from app.models.enums import EndpointCostTier
from app.models.inference import Account, LinkageObservation

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "tier2" / "v2"
FRAME_LO = datetime(2024, 7, 1, tzinfo=UTC)
FRAME_HI = datetime(2026, 6, 30, tzinfo=UTC)

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "manifest.json").exists(),
    reason="Tier 2 world not present (data/tier2 parquets)",
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((DATA_DIR / "manifest.json").read_text())


@pytest.fixture(scope="module")
def world() -> dict[str, pl.DataFrame]:
    return {
        name: pl.read_parquet(DATA_DIR / f"{name}.parquet")
        for name in (
            "accounts",
            "account_events",
            "linkage_observations",
            "entitlement_cycles",
            "ground_truth",
            "cluster_membership",
        )
    }


def _labelled_accounts(world: dict[str, pl.DataFrame]) -> pl.DataFrame:
    return world["accounts"].join(world["cluster_membership"], on="account_id").join(
        world["ground_truth"].select("cluster_id", "label"), on="cluster_id"
    )


# --------------------------------------------------------------------------- #
# world invariants
# --------------------------------------------------------------------------- #
def test_manifest_checksums_match_committed_bytes(manifest: dict) -> None:
    """Audit anchor: the committed parquets are exactly the ones the manifest
    records (seed + author + 16 accepted families + per-file sha256)."""
    assert manifest["seed"] == 20260902
    assert len(manifest["accepted_families"]) == 16
    for fname, expected in manifest["checksums"].items():
        actual = hashlib.sha256((DATA_DIR / fname).read_bytes()).hexdigest()
        assert actual == expected, f"{fname} bytes drifted from manifest checksum"


def test_schema_columns_exact(world: dict[str, pl.DataFrame]) -> None:
    expected = {
        "accounts": {"account_id", "created_at", "email_domain_type", "verification_level"},
        "account_events": {
            "event_id", "account_id", "timestamp", "event_type",
            "credits_amount", "endpoint_cost_tier",
        },
        "linkage_observations": {
            "observation_id", "account_id", "timestamp", "signal_type",
            "signal_value", "reliability",
        },
        "entitlement_cycles": {
            "cycle_id", "account_id", "cycle_start", "cycle_end",
            "credits_granted", "credits_consumed", "credits_expired",
        },
        "ground_truth": {"cluster_id", "scenario_family", "label", "notes"},
        "cluster_membership": {"account_id", "cluster_id"},
    }
    for name, cols in expected.items():
        assert set(world[name].columns) == cols, name


def test_enum_domains_legal(world: dict[str, pl.DataFrame]) -> None:
    checks = [
        ("accounts", "email_domain_type", {"consumer", "disposable", "education", "enterprise"}),
        ("accounts", "verification_level", {"none", "email", "phone", "enhanced"}),
        ("account_events", "event_type", {
            "signup", "claim_credits", "consume_credits", "login",
            "appeal", "verification", "verification_failed",
        }),
        ("account_events", "endpoint_cost_tier", {"low", "high"}),
        ("linkage_observations", "signal_type", {
            "stable_device", "network_group", "payment_token",
            "behavioral_template", "recovery_contact",
        }),
        ("linkage_observations", "reliability", {"low", "medium", "high"}),
        ("ground_truth", "label", {"abusive", "legitimate", "ambiguous"}),
    ]
    for name, col, allowed in checks:
        vals = {v for v in world[name][col].unique().to_list() if v is not None}
        assert vals <= allowed, f"{name}.{col} stray {vals - allowed}"


def test_no_ground_truth_leak_in_event_plane(world: dict[str, pl.DataFrame]) -> None:
    banned = {"cluster_id", "scenario_family", "label", "family", "split_group_id", "notes"}
    for name in ("accounts", "account_events", "linkage_observations"):
        assert banned.isdisjoint(world[name].columns), name
    # opaque account ids: no family-name token, no label word
    fam_tokens = {
        tok
        for fam in world["ground_truth"]["scenario_family"].unique().to_list()
        for tok in str(fam).lower().replace("/", " ").replace("-", " ").split()
        if len(tok) > 3
    }
    label_words = {"abusive", "legit", "legitimate", "ambiguous", "farm", "mule", "fraud"}
    tokens = fam_tokens | label_words
    ids = world["accounts"]["account_id"].to_list()
    leaky = [i for i in ids if any(t in i.lower() for t in tokens)]
    assert not leaky, leaky[:5]


def test_referential_integrity(world: dict[str, pl.DataFrame]) -> None:
    acc_ids = set(world["accounts"]["account_id"].to_list())
    for name in (
        "account_events", "linkage_observations", "entitlement_cycles", "cluster_membership",
    ):
        assert set(world[name]["account_id"].to_list()) <= acc_ids, name
    mem = world["cluster_membership"]
    assert acc_ids == set(mem["account_id"].to_list())  # every account mapped
    assert mem.height == mem["account_id"].n_unique()  # one row per account
    gt_clusters = set(world["ground_truth"]["cluster_id"].to_list())
    assert set(mem["cluster_id"].to_list()) <= gt_clusters  # every cluster labelled
    assert world["ground_truth"].height == world["ground_truth"]["cluster_id"].n_unique()


def test_scale_and_prevalence(world: dict[str, pl.DataFrame]) -> None:
    n_acc = world["accounts"].height
    gt = world["ground_truth"]
    n_clu = gt.height
    assert 1000 <= n_acc <= 3000
    assert 200 <= n_clu <= 600
    abusive = gt.filter(pl.col("label") == "abusive").height
    assert 5 <= 100 * abusive / n_clu <= 15


def test_sensitive_values_span_multiple_labels(world: dict[str, pl.DataFrame]) -> None:
    """g3 defence: no readable value maps 1:1 to a label."""
    la = _labelled_accounts(world)
    for val in ("disposable", "education"):
        labels = set(la.filter(pl.col("email_domain_type") == val)["label"].to_list())
        assert len(labels) >= 2, f"{val} only in {labels}"
    ev = world["account_events"].join(world["cluster_membership"], on="account_id").join(
        world["ground_truth"].select("cluster_id", "label"), on="cluster_id"
    )
    vf = set(ev.filter(pl.col("event_type") == "verification_failed")["label"].to_list())
    assert len(vf) >= 2, vf
    lk = world["linkage_observations"].join(world["cluster_membership"], on="account_id").join(
        world["ground_truth"].select("cluster_id", "label"), on="cluster_id"
    )
    pt = set(lk.filter(pl.col("signal_type") == "payment_token")["label"].to_list())
    assert len(pt) >= 2, pt


def test_cycle_consistency(world: dict[str, pl.DataFrame]) -> None:
    cyc = world["entitlement_cycles"]
    assert cyc.filter(pl.col("credits_consumed") > pl.col("credits_granted")).height == 0
    assert cyc.filter(pl.col("credits_granted") != 25).height == 0
    ce_dtype = cyc.schema["cycle_end"]
    hi = FRAME_HI.replace(tzinfo=None) if getattr(ce_dtype, "time_zone", None) is None else FRAME_HI
    closed = cyc.filter(pl.col("cycle_end") <= hi)
    mismatch = closed.filter(
        pl.col("credits_expired") != (pl.col("credits_granted") - pl.col("credits_consumed"))
    ).height
    assert mismatch == 0


def test_timestamps_in_frame_and_diurnal(world: dict[str, pl.DataFrame]) -> None:
    def norm(x: datetime) -> datetime:
        return x.replace(tzinfo=UTC) if x.tzinfo is None else x.astimezone(UTC)

    for name, col in [
        ("account_events", "timestamp"),
        ("linkage_observations", "timestamp"),
        ("accounts", "created_at"),
    ]:
        s = world[name][col]
        assert norm(s.min()) >= FRAME_LO and norm(s.max()) <= FRAME_HI, name
    hours = world["account_events"].with_columns(pl.col("timestamp").dt.hour().alias("h"))
    counts = hours.group_by("h").len().sort("h").to_dict(as_series=False)
    hd = dict(zip(*counts.values(), strict=True))
    overnight = sum(hd.get(h, 0) for h in (2, 3, 4))
    evening = sum(hd.get(h, 0) for h in (19, 20, 21))
    assert overnight < evening * 0.5, f"not diurnal: overnight={overnight} evening={evening}"


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #
def test_hash_signal_value_equality_contract() -> None:
    assert hash_signal_value("device-x") == hash_signal_value("device-x")
    assert hash_signal_value("device-x") != hash_signal_value("device-y")
    assert len(hash_signal_value("device-x")) == 32


def test_utc_helper() -> None:
    naive = datetime(2025, 1, 1, 12, 0, 0)
    assert _utc(naive) == datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    aware = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert _utc(aware) == aware


def test_optional_helpers() -> None:
    assert _opt_cost_tier(None) is None
    assert _opt_cost_tier("") is None
    assert _opt_cost_tier("high") == EndpointCostTier.HIGH
    assert _opt_credits(None) is None
    assert _opt_credits(25) == 25


def _write_tiny_world(d: Path) -> None:
    ts = datetime(2025, 6, 1, 15, 0, 0, tzinfo=UTC)
    pl.DataFrame({
        "account_id": ["acct-1", "acct-2", "acct-3"],
        "created_at": [ts, ts, ts],
        "email_domain_type": ["consumer", "consumer", "disposable"],
        "verification_level": ["phone", "email", "none"],
    }).write_parquet(d / "accounts.parquet")
    pl.DataFrame({
        "event_id": ["evt-0", "evt-1", "evt-2"],
        "account_id": ["acct-1", "acct-1", "acct-3"],
        "timestamp": [ts, ts + timedelta(hours=1), ts],
        "event_type": ["claim_credits", "consume_credits", "login"],
        "credits_amount": [25, 5, 0],
        "endpoint_cost_tier": ["low", "high", None],
    }).write_parquet(d / "account_events.parquet")
    # acct-1 & acct-2 share a device value; acct-3 is distinct.
    pl.DataFrame({
        "observation_id": ["obs-1", "obs-2", "obs-3"],
        "account_id": ["acct-1", "acct-2", "acct-3"],
        "timestamp": [ts, ts, ts],
        "signal_type": ["stable_device", "stable_device", "stable_device"],
        "signal_value": ["dev-shared", "dev-shared", "dev-solo"],
        "reliability": ["high", "high", "high"],
    }).write_parquet(d / "linkage_observations.parquet")
    pl.DataFrame({
        "cycle_id": ["cyc-1"],
        "account_id": ["acct-1"],
        "cycle_start": [ts],
        "cycle_end": [ts + timedelta(days=30)],
        "credits_granted": [25],
        "credits_consumed": [5],
        "credits_expired": [20],
    }).write_parquet(d / "entitlement_cycles.parquet")
    pl.DataFrame({
        "cluster_id": ["cluster-A", "cluster-B"],
        "scenario_family": ["Household Shared Device", "Trial Credit Farm"],
        "label": ["legitimate", "abusive"],
        "notes": ["two people one device", "farm"],
    }).write_parquet(d / "ground_truth.parquet")
    pl.DataFrame({
        "account_id": ["acct-1", "acct-2", "acct-3"],
        "cluster_id": ["cluster-A", "cluster-A", "cluster-B"],
    }).write_parquet(d / "cluster_membership.parquet")
    (d / "manifest.json").write_text(json.dumps({
        "world_version": "synthetic-fixture-v2", "world_end": "2026-06-30T00:00:00Z",
        "checksums": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(d.glob("*.parquet"))},
    }))


def test_loader_end_to_end(tmp_path: Path) -> None:
    _write_tiny_world(tmp_path)
    counts = load_tier2(tmp_path)
    assert counts == {
        "accounts": 3, "events": 3, "linkage": 3,
        "cycles": 1, "clusters": 2, "split_groups": 3,
    }
    assert (tmp_path / "inference.db").exists()
    assert (tmp_path / "ground_truth.db").exists()

    inf_engine = make_inference_engine(f"sqlite:///{tmp_path / 'inference.db'}")
    with Session(inf_engine, future=True) as inf:
        accts = {a.account_id: a for a in inf.query(Account).all()}
        assert set(accts) == {"acct-1", "acct-2", "acct-3"}
        obs = {o.account_id: o.signal_value_hash for o in inf.query(LinkageObservation).all()}
        # equality contract: shared raw value -> equal hash; distinct -> different
        assert obs["acct-1"] == obs["acct-2"]
        assert obs["acct-1"] != obs["acct-3"]
        # boundary: inference model exposes no label/cluster attribute
        assert not hasattr(accts["acct-1"], "label")
        assert not hasattr(accts["acct-1"], "cluster_id")

    gt_engine = make_ground_truth_engine(f"sqlite:///{tmp_path / 'ground_truth.db'}")
    with Session(gt_engine, future=True) as gt:
        labels = dict(gt.execute(select(GroundTruth.cluster_id, GroundTruth.label)).all())
        splits = load_split_groups(gt)
    assert labels["cluster-A"].value == "legitimate"
    assert labels["cluster-B"].value == "abusive"
    # split_group_id == cluster_id, keyed by account
    assert splits == {"acct-1": "cluster-A", "acct-2": "cluster-A", "acct-3": "cluster-B"}


def test_loader_is_rerunnable(tmp_path: Path) -> None:
    _write_tiny_world(tmp_path)
    first = load_tier2(tmp_path)
    before = os.stat(tmp_path / "ground_truth.db").st_ino
    inf_engine = make_inference_engine(f"sqlite:///{tmp_path / 'inference.db'}")
    gt_engine = make_ground_truth_engine(f"sqlite:///{tmp_path / 'ground_truth.db'}")
    try:
        with Session(inf_engine) as inf, Session(gt_engine) as gt:
            begin_finalization(inf, gt, {"fixture": "sealed before reload"})
    finally:
        inf_engine.dispose()
        gt_engine.dispose()
    second = load_tier2(tmp_path)
    assert first == second
    assert os.stat(tmp_path / "ground_truth.db").st_ino == before
    inf_engine = make_inference_engine(f"sqlite:///{tmp_path / 'inference.db'}")
    gt_engine = make_ground_truth_engine(f"sqlite:///{tmp_path / 'ground_truth.db'}")
    try:
        with Session(inf_engine) as inf, Session(gt_engine) as gt:
            with pytest.raises(EvaluationAccessError, match="already consumed"):
                begin_finalization(inf, gt, {"fixture": "second finalization"})
    finally:
        inf_engine.dispose()
        gt_engine.dispose()


def test_loader_concurrent_first_load_converges_to_one_pair(tmp_path: Path) -> None:
    _write_tiny_world(tmp_path)
    barrier = threading.Barrier(2)
    results: list[dict[str, int]] = []
    errors: list[Exception] = []

    def load() -> None:
        try:
            barrier.wait()
            results.append(load_tier2(tmp_path))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=load) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert not errors and len(results) == 2
    assert results[0] == results[1]
    assert len(list((tmp_path / ".loads").glob("world-*"))) == 1


def test_loader_rejects_changed_source_after_publication(tmp_path: Path) -> None:
    _write_tiny_world(tmp_path)
    load_tier2(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["world_version"] = "changed-fixture-v2"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match"):
        load_tier2(tmp_path)


def test_loader_raises_on_dangling_published_pointer(tmp_path: Path) -> None:
    """A broken/dangling `stores-current` means something was already
    committed and its target is now gone -- real corruption of a live
    pointer, not "never published". The loader must raise a clear,
    actionable `WorldProvenanceError` instead of silently masking it by
    regenerating, and it must fail fast: no wasted generation build. One-line
    human recovery (`rm stores-current`) must still unwedge it afterwards."""
    _write_tiny_world(tmp_path)
    (tmp_path / "stores-current").symlink_to("missing-generation", target_is_directory=True)
    before = set((tmp_path / ".loads").glob("world-*"))
    with pytest.raises(WorldProvenanceError, match="dangling"):
        load_tier2(tmp_path)
    after = set((tmp_path / ".loads").glob("world-*"))
    assert after == before  # no wasted regeneration work before failing fast

    os.unlink(tmp_path / "stores-current")
    counts = load_tier2(tmp_path)
    assert counts == {
        "accounts": 3, "events": 3, "linkage": 3,
        "cycles": 1, "clusters": 2, "split_groups": 3,
    }
    assert (tmp_path / "stores-current").resolve().is_dir()
    assert (tmp_path / "inference.db").resolve().exists()
    assert (tmp_path / "ground_truth.db").resolve().exists()


def test_loader_self_heals_partial_pre_commit_symlinks(tmp_path: Path) -> None:
    """A crash between the two per-store symlink writes and the `stores-current`
    commit (or a leftover from an older layout that wrote them in the other
    order) leaves dangling `inference.db`/`ground_truth.db` symlinks with no
    `stores-current` yet. Since `stores-current` is absent, this must be
    treated as "nothing published" and the loader must regenerate cleanly."""
    _write_tiny_world(tmp_path)
    (tmp_path / "inference.db").symlink_to("stores-current/inference.db")
    (tmp_path / "ground_truth.db").symlink_to("stores-current/ground_truth.db")
    counts = load_tier2(tmp_path)
    assert counts == {
        "accounts": 3, "events": 3, "linkage": 3,
        "cycles": 1, "clusters": 2, "split_groups": 3,
    }
    assert (tmp_path / "stores-current").is_symlink()
    assert (tmp_path / "inference.db").resolve().exists()
    assert (tmp_path / "ground_truth.db").resolve().exists()


def test_loader_recovers_from_crash_before_commit_without_wedging(tmp_path: Path) -> None:
    """Simulate a crash inside `_publish_pair` before the `stores-current`
    commit point (the second of its three `os.replace` calls, landing the
    `ground_truth.db` per-store symlink -- strictly before the final
    `stores-current` swap): the exception must propagate once, and the very
    next `load_tier2` call must succeed cleanly (no manual intervention, no
    "Partial"/"Broken published" dead end)."""
    import app.db.tier2_loader as loader

    _write_tiny_world(tmp_path)
    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst, *a, **kw):  # noqa: ANN001, ANN002, ANN003
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated crash before the stores-current commit point")
        return real_replace(src, dst, *a, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(loader.os, "replace", flaky_replace)
        with pytest.raises(OSError, match="simulated crash"):
            load_tier2(tmp_path)

    # Nothing was ever committed: no dangling published pointer left behind.
    assert not (tmp_path / "stores-current").exists()
    assert list((tmp_path / ".loads").glob("world-*")) == []

    counts = load_tier2(tmp_path)
    assert counts == {
        "accounts": 3, "events": 3, "linkage": 3,
        "cycles": 1, "clusters": 2, "split_groups": 3,
    }
    assert (tmp_path / "stores-current").is_symlink()


def test_loader_preserves_prior_good_pair_on_republish_failure(tmp_path: Path) -> None:
    """A failure on a *later* publish attempt (after a first publication
    already succeeded) must never touch the previously-published, still-live
    pair -- "prior stores survive validation failures" (module docstring).
    Forces a second full publish attempt (bypassing normal reuse, as if the
    source had changed) and makes it fail after its own fresh generation
    directory exists but strictly before that generation's `stores-current`
    commit point, then asserts the original pair is untouched and still
    loadable."""
    import app.db.tier2_loader as loader

    _write_tiny_world(tmp_path)
    first = load_tier2(tmp_path)
    original_generation = (tmp_path / "stores-current").resolve()
    original_gt_ino = os.stat(tmp_path / "ground_truth.db").st_ino

    with pytest.MonkeyPatch.context() as mp:
        # Force a fresh publish attempt even though the manifest is unchanged.
        mp.setattr(loader, "_reuse_published_pair", lambda *a, **kw: False)

        def failing_publish(directory, generation):  # noqa: ANN001
            # A brand new generation directory exists (from `_load_into`), but
            # this failure happens before it is ever wired to `stores-current`.
            raise RuntimeError("simulated failure inside _publish_pair")

        mp.setattr(loader, "_publish_pair", failing_publish)
        with pytest.raises(RuntimeError, match="simulated failure"):
            load_tier2(tmp_path)

    # The prior, already-published pair must be completely untouched.
    assert (tmp_path / "stores-current").resolve() == original_generation
    assert original_generation.exists()
    assert os.stat(tmp_path / "ground_truth.db").st_ino == original_gt_ino
    # The new, never-committed generation from the failed attempt is cleaned up.
    remaining = {p for p in (tmp_path / ".loads").glob("world-*")}
    assert remaining == {original_generation}

    second = load_tier2(tmp_path)
    assert second == first
    assert os.stat(tmp_path / "ground_truth.db").st_ino == original_gt_ino


def test_loader_republish_rebuilds_fresh_generation_and_retains_prior(tmp_path: Path) -> None:
    """`republish=True` is the explicit opt-in path: it skips the reuse short
    circuit, always rebuilds a fresh generation from the verified sources, and
    atomically swaps `stores-current` onto it while retaining the prior
    generation (never a force/delete). The republished pair then verifies
    cleanly through the normal read path."""
    _write_tiny_world(tmp_path)
    first = load_tier2(tmp_path)
    prior_generation = (tmp_path / "stores-current").resolve()

    counts = load_tier2(tmp_path, republish=True)
    assert counts == first

    new_generation = (tmp_path / "stores-current").resolve()
    assert new_generation != prior_generation      # a genuinely fresh generation
    assert new_generation.is_dir()
    assert prior_generation.exists()               # prior generation retained, not deleted
    generations = set((tmp_path / ".loads").glob("world-*"))
    assert {prior_generation, new_generation} <= generations
    # The live per-store symlinks now resolve into the new generation.
    assert (tmp_path / "inference.db").resolve().parent == new_generation
    assert (tmp_path / "ground_truth.db").resolve().parent == new_generation

    inf_engine = make_inference_engine(f"sqlite:///{tmp_path / 'inference.db'}")
    gt_engine = make_ground_truth_engine(f"sqlite:///{tmp_path / 'ground_truth.db'}")
    try:
        with Session(inf_engine, future=True) as inf, Session(gt_engine, future=True) as gt:
            world = verify_world_pair(inf, gt)
    finally:
        inf_engine.dispose()
        gt_engine.dispose()
    # The tiny fixture manifest carries no `world_tier`, so it defaults to tier2.
    assert world.tier == "tier2"


def _corrupt_dangling(d: Path) -> None:
    (d / "stores-current").symlink_to("missing-generation", target_is_directory=True)


def _corrupt_non_symlink_dir(d: Path) -> None:
    (d / ".loads").mkdir(exist_ok=True)
    (d / "stores-current").mkdir()


def _corrupt_non_directory_target(d: Path) -> None:
    load_tier2(d)
    os.unlink(d / "stores-current")
    bogus = d / "not-a-generation"
    bogus.write_text("not a directory")
    (d / "stores-current").symlink_to(bogus.relative_to(d))


def _corrupt_missing_store(d: Path) -> None:
    load_tier2(d)
    ((d / "stores-current").resolve() / "inference.db").unlink()


def _corrupt_outside_root(d: Path) -> None:
    load_tier2(d)
    generation = (d / "stores-current").resolve()
    external = d / "external"
    external.mkdir()
    ext_db = external / "inference.db"
    ext_db.write_bytes((generation / "inference.db").read_bytes())
    (generation / "inference.db").unlink()
    (generation / "inference.db").symlink_to(ext_db)


def _corrupt_external_generation(d: Path) -> None:
    """`stores-current` points at an EXTERNAL directory (outside `.loads/`) that
    contains two otherwise-valid databases. Every per-store link then resolves
    with that external directory as its parent, so the per-store parent check
    passes; only the generation-store-location check catches it."""
    load_tier2(d)
    generation = (d / "stores-current").resolve()
    external = d / "external-gen"
    external.mkdir()
    (external / "inference.db").write_bytes((generation / "inference.db").read_bytes())
    (external / "ground_truth.db").write_bytes((generation / "ground_truth.db").read_bytes())
    os.unlink(d / "stores-current")
    (d / "stores-current").symlink_to(external.relative_to(d), target_is_directory=True)


@pytest.mark.parametrize(
    "corrupt, match",
    [
        (_corrupt_dangling, "dangling"),
        (_corrupt_non_symlink_dir, "non-symlink"),
        (_corrupt_non_directory_target, "non-directory"),
        (_corrupt_missing_store, "missing inference.db"),
        (_corrupt_outside_root, "outside"),
        (_corrupt_external_generation, "generation store"),
    ],
)
def test_loader_republish_fails_fast_on_corrupt_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt, match: str  # type: ignore[no-untyped-def]
) -> None:
    """`republish=True` bypasses only the receipt/provenance *content* reuse
    check — it must NOT bypass the structural published-state validator. Each
    corrupt/foreign commit-marker state (dangling, non-symlink occupant,
    non-directory target, missing per-store file, per-store resolving outside
    the committed root) must fail fast with a clear `WorldProvenanceError`
    BEFORE any regeneration work. Spying on `_load_into` proves the pre-check
    happens before a fresh generation is ever built."""
    _write_tiny_world(tmp_path)
    corrupt(tmp_path)

    def _load_into_must_not_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("_load_into ran — the structural pre-check did not fail fast")

    monkeypatch.setattr(tier2_loader, "_load_into", _load_into_must_not_run)
    before = set((tmp_path / ".loads").glob("world-*"))
    with pytest.raises(WorldProvenanceError, match=match):
        load_tier2(tmp_path, republish=True)
    # No wasted regeneration: the republish attempt built nothing new.
    assert set((tmp_path / ".loads").glob("world-*")) == before


def _load_republish_cli():  # type: ignore[no-untyped-def]
    """Import `scripts/republish_tier3.py` by path (scripts are not on the test
    pythonpath, only `backend/` is) so its CLI helpers are exercisable in-suite."""
    path = Path(__file__).resolve().parents[3] / "scripts" / "republish_tier3.py"
    spec = importlib.util.spec_from_file_location("republish_tier3_undertest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_republish_cli_refuses_non_tier3_manifest_before_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2: the Tier 3 CLI validates the declared source tier BEFORE the mutating
    loader runs, so a wrong `--dir` (a Tier 2 world) is never republished. The
    legacy tiny fixture manifest carries no `world_tier` → derives `tier2`."""
    _write_tiny_world(tmp_path)
    cli = _load_republish_cli()
    calls: list[object] = []
    monkeypatch.setattr(cli, "load_tier2", lambda *a, **k: calls.append((a, k)) or {})
    monkeypatch.setattr(sys, "argv", ["republish_tier3.py", "--dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="not 'tier3'"):
        cli.main()
    assert calls == []  # the mutating loader was never reached


def test_republish_cli_source_hashes_cover_manifest_and_gpt_world(tmp_path: Path) -> None:
    """P2: the before/after preservation digest must include `manifest.json` and
    `gpt-world.json`, not only the six parquets, and must fail closed when an
    authoritative source file is missing."""
    _write_tiny_world(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["world_tier"] = "tier3"
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "gpt-world.json").write_text('{"synthetic": "fixture"}')

    cli = _load_republish_cli()
    digest = cli._source_checksums(tmp_path)
    assert "manifest.json" in digest and "gpt-world.json" in digest
    assert set(manifest["checksums"]) <= set(digest)  # every manifest parquet still covered

    before = cli._source_checksums(tmp_path)
    (tmp_path / "gpt-world.json").write_text('{"synthetic": "tampered"}')
    assert cli._source_checksums(tmp_path) != before  # a mutation is detected

    (tmp_path / "gpt-world.json").unlink()
    with pytest.raises(SystemExit, match="missing"):
        cli._source_checksums(tmp_path)


def test_loader_raises_when_published_pointer_targets_a_non_directory(tmp_path: Path) -> None:
    """A `stores-current` that resolves to a regular file rather than a
    generation directory names a corrupted commit; it must raise, not
    silently regenerate over it."""
    _write_tiny_world(tmp_path)
    load_tier2(tmp_path)
    os.unlink(tmp_path / "stores-current")
    bogus_file = tmp_path / "not-a-generation"
    bogus_file.write_text("not a directory")
    (tmp_path / "stores-current").symlink_to(bogus_file.relative_to(tmp_path))
    with pytest.raises(WorldProvenanceError, match="non-directory"):
        load_tier2(tmp_path)


def test_loader_raises_when_committed_generation_lost_its_stores(tmp_path: Path) -> None:
    """If a committed generation is missing one of its two per-store db
    files, the published pointer is corrupted -- raise rather than mask it
    by regenerating."""
    _write_tiny_world(tmp_path)
    load_tier2(tmp_path)
    generation = (tmp_path / "stores-current").resolve()
    (generation / "inference.db").unlink()
    with pytest.raises(WorldProvenanceError, match="missing inference.db"):
        load_tier2(tmp_path)


def test_loader_raises_when_non_symlink_occupies_stores_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real directory sitting at the `stores-current` path (never
    published before) is a foreign occupant of the commit-marker path, not
    "nothing published yet" -- raise immediately, before any regeneration
    work is done.

    A directory-existence check alone does not discriminate this from the
    OLD behavior: the old `_reuse_published_pair` returned False here, the
    loader went on to regenerate, and `_publish_pair`'s own (older,
    unrelated) non-symlink guard then raised the same "non-symlink"-matching
    message -- after which `_cleanup_unpublished` deletes the just-built
    generation, leaving the `.loads` directory looking identical either way.
    Spying on `_load_into` is what actually proves the check now happens
    BEFORE any regeneration work, not just that no generation dir survives.
    """
    _write_tiny_world(tmp_path)
    (tmp_path / "stores-current").mkdir()
    (tmp_path / ".loads").mkdir(exist_ok=True)

    def _load_into_must_not_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("_load_into was called -- the pre-check did not fail fast")

    monkeypatch.setattr(tier2_loader, "_load_into", _load_into_must_not_run)
    with pytest.raises(WorldProvenanceError, match="non-symlink"):
        load_tier2(tmp_path)


def test_v2_ids_are_opaque_across_all_inference_references(world: dict[str, pl.DataFrame]) -> None:
    import re

    for table, columns in {
        "accounts": ("account_id",), "account_events": ("event_id", "account_id"),
        "linkage_observations": ("observation_id", "account_id"),
        "entitlement_cycles": ("cycle_id", "account_id"),
    }.items():
        for column in columns:
            assert all(re.fullmatch(r"(?:acct|evt|obs|cyc)-[0-9a-f]{64}", value)
                       for value in world[table][column].to_list())


def test_v2_authoritative_windows_reconcile_every_credit_event(
    world: dict[str, pl.DataFrame],
) -> None:
    from app.db.tier2_loader import _validate_credit_windows

    _validate_credit_windows(world["account_events"], world["entitlement_cycles"], FRAME_HI)


def test_v2_reproduces_bytes_and_retains_original_world(tmp_path: Path, manifest: dict) -> None:
    import subprocess
    import sys

    source = DATA_DIR.parent / "generate_tier2.py"
    subprocess.run([sys.executable, str(source), "--out-dir", str(tmp_path)], check=True,
                   capture_output=True)
    regenerated = json.loads((tmp_path / "manifest.json").read_text())
    assert regenerated == manifest
    for name, checksum in manifest["checksums"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == checksum
    original = DATA_DIR.parent / "manifest.json"
    assert hashlib.sha256(original.read_bytes()).hexdigest() == manifest[
        "parent_world_manifest_sha256"
    ]
    original_manifest = json.loads(original.read_text())
    for name, checksum in original_manifest["checksums"].items():
        assert hashlib.sha256((DATA_DIR.parent / name).read_bytes()).hexdigest() == checksum
    # Mechanical repair has no authority to change any scenario labels or notes.
    assert manifest["checksums"]["ground_truth.parquet"] == original_manifest["checksums"][
        "ground_truth.parquet"
    ]


def test_mechanical_derivative_preserves_observations(world: dict[str, pl.DataFrame]) -> None:
    """No behavioral RNG drift or detector-driven resampling hides inside the repair."""
    current_cycles = pl.read_parquet(DATA_DIR / "entitlement_cycles.parquet")
    current_cycle_windows: dict[str, list[tuple[datetime, datetime]]] = {}
    current_cycle_ends: set[tuple[str, datetime]] = set()
    for row in current_cycles.iter_rows(named=True):
        current_cycle_windows.setdefault(row["account_id"], []).append(
            (row["cycle_start"], row["cycle_end"])
        )
        current_cycle_ends.add((row["account_id"], row["cycle_end"]))

    def renamed(kind: str, ordinal: str) -> str:
        number = int(ordinal.rsplit("-", 1)[1])
        digest = hashlib.sha256(f"world-v2:20260902:{kind}:{number}".encode()).hexdigest()
        return f"{kind}-{digest}"

    clipped_credit_events = 0
    for name, identifier, kind in (
        ("accounts", "account_id", "acct"),
        ("account_events", "event_id", "evt"),
        ("linkage_observations", "observation_id", "obs"),
    ):
        original = pl.read_parquet(DATA_DIR.parent / f"{name}.parquet")
        current = {row[identifier]: row for row in world[name].iter_rows(named=True)}
        assert len(current) == original.height
        for row in original.iter_rows(named=True):
            expected = dict(row)
            old_account = row["account_id"]
            expected[identifier] = renamed(kind, row[identifier])
            expected["account_id"] = renamed("acct", old_account)
            actual = current[expected[identifier]]
            if name == "account_events" and actual["timestamp"] != expected["timestamp"]:
                assert expected["timestamp"] - actual["timestamp"] == timedelta(microseconds=1)
                assert row["event_type"] in {"claim_credits", "consume_credits"}
                # The mechanical repair clips exactly a legacy event at the
                # authoritative cycle end. It must not infer cycle boundaries
                # from signup: classroom cycles intentionally start at the term.
                assert (expected["account_id"], expected["timestamp"]) in current_cycle_ends
                assert any(
                    start <= actual["timestamp"] < end
                    for start, end in current_cycle_windows[expected["account_id"]]
                )
                expected["timestamp"] = actual["timestamp"]
                clipped_credit_events += 1
            assert actual == expected
    assert clipped_credit_events == 706

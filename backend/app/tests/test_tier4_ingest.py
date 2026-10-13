"""Tier 4 ingestion — appeal text on the inference plane, appeal truth sealed.

Builds a tiny brief-shaped world in memory (the real 98 MB relay is not committed)
and relaxes only the scale/prevalence bounds of the Tier 4 profile, so every
structural, appeal and leakage rule runs exactly as for the relayed world.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import polars as pl
import pytest


def _load_ingest() -> ModuleType:
    """Import `scripts/ingest_gpt_world.py` by path (only `backend/` is on the path)."""
    path = Path(__file__).resolve().parents[3] / "scripts" / "ingest_gpt_world.py"
    spec = importlib.util.spec_from_file_location("ingest_gpt_world_undertest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` resolves annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ingest_module = _load_ingest()
FAMILY_LABELS = ["legitimate"] * 6 + ["abusive"] * 3 + ["ambiguous"]


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _world() -> dict[str, Any]:
    """Ten one-cluster families, two accounts each, calendar-aligned 30-day cycles
    (the last one open past world_end), and one appeal per cluster."""
    frame_start = datetime(2025, 12, 1, tzinfo=UTC)
    world: dict[str, Any] = {
        "world_version": "tier4-test",
        "author_model": "test",
        "generator_seed": 7,
        "authoring_notes": "",
        "families": [],
        "accounts": [],
        "account_events": [],
        "linkage_observations": [],
        "entitlement_cycles": [],
        "cluster_membership": [],
        "ground_truth": [],
        "appeal_truth": [],
    }
    for f, label in enumerate(FAMILY_LABELS):
        world["families"].append({
            "name": f"Family {f:02d}", "label": label, "description": "d",
            "misleading_signals": "m", "revealing_combination": "r",
        })
        cluster = f"c{f:02d}x"
        world["ground_truth"].append(
            {"cluster_id": cluster, "scenario_family": f"Family {f:02d}", "label": label,
             "notes": "n"}
        )
        for a in range(2):
            account = f"a{f:02d}{a}x"
            # Created mid-way through the first cycle: the cycle starts before it.
            created = frame_start + timedelta(days=3)
            world["accounts"].append({
                "account_id": account, "created_at": _iso(created),
                "email_domain_type": "consumer", "verification_level": "email",
            })
            world["cluster_membership"].append({"account_id": account, "cluster_id": cluster})
            world["linkage_observations"].append({
                "observation_id": f"o{account}", "account_id": account,
                "timestamp": _iso(created + timedelta(hours=1)),
                "signal_type": "stable_device", "signal_value": f"dev{cluster}",
                "reliability": "high",
            })
            for k in range(7):
                start = frame_start + timedelta(days=30 * k)
                end = start + timedelta(days=30)
                claim_at = max(start, created) + timedelta(hours=2)
                cycle = f"y{account}{k}"
                open_cycle = end > datetime(2026, 6, 1, tzinfo=UTC)
                world["entitlement_cycles"].append({
                    "cycle_id": cycle, "account_id": account, "cycle_start": _iso(start),
                    "cycle_end": _iso(end), "credits_granted": 25, "credits_consumed": 5,
                    "credits_expired": 0 if open_cycle else 20,
                })
                for kind, amount, offset in (("claim_credits", 25, 0), ("consume_credits", 5, 1)):
                    world["account_events"].append({
                        "event_id": f"e{cycle}{kind[:2]}", "account_id": account,
                        "timestamp": _iso(claim_at + timedelta(minutes=offset)),
                        "event_type": kind, "credits_amount": amount,
                        "endpoint_cost_tier": "high" if kind == "consume_credits" else None,
                        "appeal_text": None,
                    })
        appeal_id = f"p{f:02d}x"
        world["account_events"].append({
            "event_id": appeal_id, "account_id": f"a{f:02d}0x",
            "timestamp": _iso(frame_start + timedelta(days=40)), "event_type": "appeal",
            "credits_amount": None, "endpoint_cost_tier": None,
            "appeal_text": "We share one tablet in the evenings.",
        })
        world["appeal_truth"].append({"event_id": appeal_id, "kind": "honest", "notes": "n"})
    return world


@pytest.fixture(autouse=True)
def _small_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    small = dataclasses.replace(
        ingest_module.PROFILES["tier4"],
        clusters=(1, 50), accounts=(1, 100), abusive_fraction=(0.0, 1.0),
        min_family_clusters=1,
    )
    monkeypatch.setitem(ingest_module.PROFILES, "tier4", small)


def _write(tmp_path: Path, world: dict[str, Any]) -> Path:
    source = tmp_path / "relay" / "gpt-world.json"
    source.parent.mkdir()
    source.write_text(json.dumps(world))
    (source.parent / "generator.py").write_text("# relayed\n")
    return source


def test_tier4_world_splits_appeal_text_and_truth_across_planes(tmp_path: Path) -> None:
    out = tmp_path / "tier4"
    counts = ingest_module.ingest(_write(tmp_path, _world()), out, load=True, tier="tier4")
    assert counts is not None and counts["appeals"] == 10 and counts["appeal_truth"] == 10

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["world_tier"] == "tier4"
    assert manifest["seed"] == 7
    assert set(manifest["checksums"]) == {
        f"{name}.parquet" for name in (*ingest_module.ARTIFACTS, "appeals", "appeal_truth")
    }
    assert set(manifest["companion_sha256"]) == {"generator.py"}
    assert "appeal_text" not in pl.read_parquet(out / "account_events.parquet").columns
    assert pl.read_parquet(out / "appeals.parquet").columns == [
        "appeal_id", "account_id", "submitted_at", "text",
    ]

    inference = sqlite3.connect(out / "inference.db")
    ground_truth = sqlite3.connect(out / "ground_truth.db")
    try:
        inference_tables = {
            r[0] for r in inference.execute("select name from sqlite_master where type='table'")
        }
        assert "appeal_truth" not in inference_tables
        assert inference.execute("select count(*) from appeals").fetchone() == (10,)
        assert ground_truth.execute("select count(*) from appeal_truth").fetchone() == (10,)
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            ground_truth.execute("delete from appeal_truth")
    finally:
        inference.close()
        ground_truth.close()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda w: w["appeal_truth"].pop(), "exactly one appeal_truth row"),
        (
            lambda w: w["account_events"][-1].update(appeal_text="This is legitimate use."),
            "label, family name or cluster id",
        ),
        (
            lambda w: w["account_events"][-1].update(appeal_text="See Family 03 notes."),
            "label, family name or cluster id",
        ),
        (
            lambda w: w["account_events"][0].update(appeal_text="stray"),
            "null on non-appeal events",
        ),
        (
            lambda w: w["account_events"][-1].update(appeal_text="x" * 1001),
            "exceeds 1000 characters",
        ),
        (lambda w: w["appeal_truth"][0].update(kind="partial"), "invalid appeal_truth kind"),
        (lambda w: w.pop("generator_seed"), "top-level keys"),
    ],
)
def test_tier4_rejects_appeal_and_shape_violations(
    tmp_path: Path, mutate: Any, message: str
) -> None:
    world = _world()
    mutate(world)
    out = tmp_path / "tier4"
    with pytest.raises(ingest_module.Tier3ValidationError, match=message):
        ingest_module.ingest(_write(tmp_path, world), out, load=False, tier="tier4")
    assert (out / "validation-error.txt").exists()
    assert (out / "gpt-world.json").exists()


def test_tier4_rejects_activity_before_account_creation(tmp_path: Path) -> None:
    world = _world()
    world["linkage_observations"][0]["timestamp"] = "2025-12-02T00:00:00Z"
    with pytest.raises(ingest_module.Tier3ValidationError, match="precedes the account"):
        ingest_module.ingest(_write(tmp_path, world), tmp_path / "t4", load=False, tier="tier4")


def test_world_appeals_are_sealed_but_runtime_appeals_stay_writable(tmp_path: Path) -> None:
    out = tmp_path / "tier4"
    ingest_module.ingest(_write(tmp_path, _world()), out, load=True, tier="tier4")
    inference = sqlite3.connect(out / "inference.db")
    try:
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            inference.execute("update appeals set text = 'edited'")
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            inference.execute(
                "insert into appeals values "
                "('forged', 'a000x', null, '2026-01-01 00:00:00', 't', 0, 'world:tier4-test')"
            )
        inference.execute(
            "insert into appeals values "
            "('rt', 'a000x', null, '2026-01-01 00:00:00', 't', 0, 'api')"
        )
        inference.execute("delete from appeals where appeal_id = 'rt'")
    finally:
        inference.close()

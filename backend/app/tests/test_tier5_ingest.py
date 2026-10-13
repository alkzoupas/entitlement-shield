"""Tier 5 ingestion (P6-5/P6-6) — Tier 4 minus appeals, strict brief rules.

Builds a tiny brief-shaped world in memory (the real 89 MB relay is not committed)
and relaxes only the scale/prevalence bounds of the Tier 5 profile, so every
structural, family-mix, ordering and identifier rule runs as for the relayed world.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from app.tests.test_tier4_ingest import ingest_module

FAMILY_LABELS = ["abusive"] * 6 + ["legitimate"] * 5 + ["ambiguous"]


def _hex(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:12]


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _world() -> dict[str, Any]:
    """Twelve one-cluster families, two accounts each, created before their first
    30-day cycle (the last one open past world_end), opaque 12-hex ids."""
    frame_start = datetime(2025, 12, 1, tzinfo=UTC)
    world: dict[str, Any] = {
        "world_version": "tier5-test",
        "author_model": "test",
        "generator_seed": 0,
        "authoring_notes": "",
        "families": [],
        "accounts": [],
        "account_events": [],
        "linkage_observations": [],
        "entitlement_cycles": [],
        "cluster_membership": [],
        "ground_truth": [],
    }
    for f, label in enumerate(FAMILY_LABELS):
        world["families"].append({
            "name": f"Family {f:02d}", "label": label, "description": "d",
            "misleading_signals": "m", "revealing_combination": "r",
        })
        cluster = _hex(f"cluster{f}")
        world["ground_truth"].append(
            {"cluster_id": cluster, "scenario_family": f"Family {f:02d}", "label": label,
             "notes": "n"}
        )
        for a in range(2):
            account = _hex(f"account{f}.{a}")
            created = frame_start - timedelta(days=10)
            world["accounts"].append({
                "account_id": account, "created_at": _iso(created),
                "email_domain_type": "consumer", "verification_level": "email",
            })
            world["cluster_membership"].append({"account_id": account, "cluster_id": cluster})
            world["linkage_observations"].append({
                "observation_id": _hex(f"obs{account}"), "account_id": account,
                "timestamp": _iso(frame_start + timedelta(hours=1)),
                "signal_type": "stable_device", "signal_value": _hex(f"dev{cluster}"),
                "reliability": "high",
            })
            for k in range(7):
                start = frame_start + timedelta(days=30 * k)
                end = start + timedelta(days=30)
                cycle = _hex(f"cycle{account}{k}")
                open_cycle = end > datetime(2026, 6, 1, tzinfo=UTC)
                world["entitlement_cycles"].append({
                    "cycle_id": cycle, "account_id": account, "cycle_start": _iso(start),
                    "cycle_end": _iso(end), "credits_granted": 25, "credits_consumed": 5,
                    "credits_expired": 0 if open_cycle else 20,
                })
                for kind, amount, offset in (("claim_credits", 25, 0), ("consume_credits", 5, 1)):
                    world["account_events"].append({
                        "event_id": _hex(f"event{cycle}{kind}"), "account_id": account,
                        "timestamp": _iso(start + timedelta(hours=2, minutes=offset)),
                        "event_type": kind, "credits_amount": amount,
                        "endpoint_cost_tier": "high" if kind == "consume_credits" else None,
                    })
    return world


def _profile(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    small = dataclasses.replace(
        ingest_module.PROFILES["tier5"],
        **{
            "clusters": (1, 50), "accounts": (1, 100), "abusive_fraction": (0.0, 1.0),
            "min_family_clusters": 1, "min_abusive_clusters": 0,
            "min_abusive_family_clusters": 1, "max_abusive_family_share": 1.0,
            "ambiguous_fraction": (0.0, 1.0),
            **overrides,
        },
    )
    monkeypatch.setitem(ingest_module.PROFILES, "tier5", small)


def _write(tmp_path: Path, world: dict[str, Any]) -> Path:
    source = tmp_path / "relay" / "world.json"
    source.parent.mkdir()
    source.write_text(json.dumps(world))
    for name in ("generator.py", "self-check.json", "families.json"):
        (source.parent / name).write_text("{}\n")
    return source


def test_tier5_world_loads_without_appeals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _profile(monkeypatch)
    out = tmp_path / "tier5"
    counts = ingest_module.ingest(_write(tmp_path, _world()), out, load=True, tier="tier5")
    assert counts is not None and "appeals" not in counts

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["world_tier"] == "tier5"
    assert manifest["schema_version"] == "tier5-codex-brief-v0.2"
    assert manifest["seed"] == 0
    assert set(manifest["checksums"]) == {f"{n}.parquet" for n in ingest_module.ARTIFACTS}
    assert set(manifest["companion_sha256"]) == {
        "generator.py", "self-check.json", "families.json",
    }
    assert manifest["validation"]["opaque_identifiers"] == "passed"
    assert manifest["counts"]["clusters_by_label"] == {
        "abusive": 6, "legitimate": 5, "ambiguous": 1,
    }

    inference = sqlite3.connect(out / "inference.db")
    ground_truth = sqlite3.connect(out / "ground_truth.db")
    try:
        tables = {
            r[0] for r in inference.execute("select name from sqlite_master where type='table'")
        }
        assert "ground_truth" not in tables and "appeal_truth" not in tables
        assert inference.execute("select count(*) from appeals").fetchone() == (0,)
        (payload,) = inference.execute(
            "select payload from world_inference_receipt"
        ).fetchone()
        assert json.loads(payload)["tier"] == "tier5"
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            ground_truth.execute("delete from ground_truth")
    finally:
        inference.close()
        ground_truth.close()


def test_tier5_reporting_status_is_sealed() -> None:
    from app.db.world_provenance import VerifiedWorld

    status = VerifiedWorld("tier5", "w", "v", {}).reporting_status
    assert status["reportable"] is False
    assert status["code"] == "sealed_confirmation_unregistered"


def _appeal(world: dict[str, Any]) -> None:
    event = dict(world["account_events"][0], event_id=_hex("appeal"), event_type="appeal")
    event.update(credits_amount=None, endpoint_cost_tier=None)
    world["account_events"].append(event)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_appeal, "no appeal events"),
        (lambda w: w["account_events"][0].update(appeal_text=None), r"keys must be exactly"),
        (lambda w: w.update(appeal_truth=[]), "top-level keys"),
        (lambda w: w.pop("generator_seed"), "top-level keys"),
        (lambda w: w["accounts"][0].update(account_id="A" * 12), "unknown account|12 lowercase"),
        (
            lambda w: w["linkage_observations"][0].update(signal_value="device-0001"),
            "signal_value must be 12 lowercase hex",
        ),
        (
            # Tier 4 allowed calendar cycles to start before creation; Tier 5 does not.
            lambda w: w["accounts"][0].update(created_at="2025-12-01T00:30:00Z"),
            "precedes the account creation time",
        ),
        (
            lambda w: w["accounts"][0].update(created_at="2023-11-01T00:00:00Z"),
            "more than two years",
        ),
        (
            lambda w: [row.update(label="legitimate") for row in w["families"][:1]]
            and w["ground_truth"][0].update(label="legitimate"),
            "≥6 abusive",
        ),
    ],
)
def test_tier5_rejects_brief_violations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: Any, message: str
) -> None:
    _profile(monkeypatch)
    world = _world()
    mutate(world)
    out = tmp_path / "tier5"
    with pytest.raises(ingest_module.Tier3ValidationError, match=message):
        ingest_module.ingest(_write(tmp_path, world), out, load=False, tier="tier5")
    assert (out / "validation-error.txt").exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"min_abusive_family_clusters": 2}, "every abusive family must contain at least 2"),
        ({"min_abusive_clusters": 7}, "at least 7 abusive clusters"),
        ({"max_abusive_family_share": 0.1}, "more than 10% of abusive clusters"),
        # One ambiguous cluster in twelve (8.3%) lies outside both bands.
        ({"ambiguous_fraction": (0.02, 0.08)}, "ambiguous clusters must comprise 2%–8%"),
        ({"ambiguous_fraction": (0.09, 1.0)}, "ambiguous clusters must comprise"),
    ],
)
def test_tier5_enforces_abusive_family_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any], message: str
) -> None:
    _profile(monkeypatch, **overrides)
    with pytest.raises(ingest_module.Tier3ValidationError, match=message):
        ingest_module.ingest(_write(tmp_path, _world()), tmp_path / "t5", load=False, tier="tier5")


def test_tier5_profile_matches_the_brief() -> None:
    profile = ingest_module.PROFILES["tier5"]
    assert profile.clusters == (900, 1000) and profile.accounts == (4000, 7000)
    assert profile.min_abusive_clusters == 160
    assert profile.min_abusive_family_clusters == 20 and profile.min_family_clusters == 10
    assert profile.max_abusive_family_share == 0.25
    assert profile.ambiguous_fraction == (0.02, 0.08)
    # Tier 3/4 keep no ambiguous bound.
    assert ingest_module.PROFILES["tier4"].ambiguous_fraction == (0.0, 1.0)

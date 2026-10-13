"""Receipt gates use tiny authored fixtures, never a sealed performance evaluation."""
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select, update
from sqlalchemy.exc import DatabaseError
from sqlalchemy.orm import Session

from app.db.engine import make_inference_engine
from app.db.ground_truth_store import GroundTruth, make_ground_truth_engine
from app.db.tier2_loader import load_tier2
from app.db.world_provenance import (
    WorldProvenanceError,
    load_inference_receipt,
    verify_world_pair,
)
from app.models.enums import EmailDomainType
from app.models.inference import Account
from app.services.detector import run_detector
from app.tests.test_tier2 import _write_tiny_world


def _pair(directory: Path) -> tuple[Session, Session]:
    return (
        Session(make_inference_engine(f"sqlite:///{directory / 'inference.db'}")),
        Session(make_ground_truth_engine(f"sqlite:///{directory / 'ground_truth.db'}")),
    )


def test_pair_verification_and_inference_receipt_boundary(tmp_path: Path) -> None:
    _write_tiny_world(tmp_path)
    load_tier2(tmp_path)
    inf, gt = _pair(tmp_path)
    with inf, gt:
        world = verify_world_pair(inf, gt)
        assert world.tier == "tier2"
        receipt = load_inference_receipt(inf)
        assert set(receipt or {}) == {
            "tier", "world_id", "world_version", "inference_digest", "loader_version",
        }
        assert "ground_truth" not in json.dumps(receipt)
        run_detector(inf, as_of=datetime(2025, 6, 5, tzinfo=UTC), run_id="receipt-derived")
        assert verify_world_pair(inf, gt).world_id == world.world_id
        inf.execute(update(Account).where(Account.account_id == "acct-1")
                    .values(email_domain_type=EmailDomainType.ENTERPRISE))
        with pytest.raises(WorldProvenanceError, match="inputs changed"):
            verify_world_pair(inf, gt)


def test_mixed_pair_fails_closed(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        _write_tiny_world(directory)
    manifest_path = second / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["world_version"] = "other-fixture"
    manifest_path.write_text(json.dumps(manifest))
    load_tier2(first)
    load_tier2(second)
    inf, _unused_gt = _pair(first)
    _unused_inf, gt = _pair(second)
    with inf, gt, _unused_gt, _unused_inf:
        with pytest.raises(WorldProvenanceError, match="different worlds"):
            verify_world_pair(inf, gt)


def test_checksum_failure_preserves_published_pair(tmp_path: Path) -> None:
    _write_tiny_world(tmp_path)
    load_tier2(tmp_path)
    pointer = (tmp_path / "stores-current").readlink()
    (tmp_path / "accounts.parquet").write_bytes(b"corrupt source")
    with pytest.raises(WorldProvenanceError, match="checksum"):
        load_tier2(tmp_path)
    assert (tmp_path / "stores-current").readlink() == pointer
    inf, gt = _pair(tmp_path)
    with inf, gt:
        assert verify_world_pair(inf, gt).tier == "tier2"


def test_sealed_labels_and_membership_cannot_change_after_ingestion(tmp_path: Path) -> None:
    _write_tiny_world(tmp_path)
    load_tier2(tmp_path)
    inf, gt = _pair(tmp_path)
    with inf, gt:
        with pytest.raises(DatabaseError, match="immutable"):
            gt.execute(update(GroundTruth).values(notes="modified"))
        gt.rollback()
        assert verify_world_pair(inf, gt)
        assert len(gt.execute(select(GroundTruth.cluster_id)).all()) == 2


def test_valid_hashes_do_not_hide_bad_windows(tmp_path: Path) -> None:
    import polars as pl

    _write_tiny_world(tmp_path)
    cycles_path = tmp_path / "entitlement_cycles.parquet"
    pl.read_parquet(cycles_path).with_columns(pl.lit(4).alias("credits_consumed")) \
        .write_parquet(cycles_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checksums"][cycles_path.name] = hashlib.sha256(cycles_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(WorldProvenanceError, match="totals disagree"):
        load_tier2(tmp_path)
    assert not (tmp_path / "stores-current").exists()

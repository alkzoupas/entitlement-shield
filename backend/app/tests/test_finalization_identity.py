"""Gate E pre-grant cutoff and reproducibility identity contracts."""
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.services.finalization_identity import canonical_evaluation_cutoff, reproducibility_identity


def test_cutoff_requires_aware_utc_and_exact_world_end() -> None:
    world_end = "2026-06-30T00:00:00Z"
    assert canonical_evaluation_cutoff(world_end, world_end) == datetime(2026, 6, 30, tzinfo=UTC)
    with pytest.raises(SystemExit, match="explicit UTC offset"):
        canonical_evaluation_cutoff("2026-06-30T00:00:00", world_end)
    with pytest.raises(SystemExit, match="canonical world_end"):
        canonical_evaluation_cutoff("2026-06-29T16:00:00-07:00", world_end)


def test_reproducibility_identity_records_revision_dirty_state_and_runtime() -> None:
    root = Path(__file__).resolve().parents[3]
    identity = reproducibility_identity(root)
    assert len(identity["git_revision"]) == 40
    assert isinstance(identity["git_dirty"], bool)
    for field in ("git_status_sha256", "git_diff_sha256", "dependencies_sha256"):
        assert len(identity[field]) == 64
    assert identity["python"] and identity["runtime"]

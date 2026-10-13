"""The tuning ledger is reproducible and disciplined (Step-8 review finding 6).

Proves the committed `config/tuning-ledger.json` regenerates its deterministic
`results` from a clean checkout, the chosen vector is exactly the frozen weight
vector selected by the declared objective, and the tuning path refuses any non-tune
row (the sealed set can never leak into selection).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from app.services import features as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "tune_weights.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("tune_weights", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load_module()


@pytest.fixture(scope="module")
def regenerated() -> dict:
    """Regenerate the deterministic ledger core ONCE for the module (each call builds
    four dev worlds, so sharing it keeps the suite fast)."""
    return T._ledger()


def test_committed_ledger_reproduces_byte_for_byte(regenerated: dict) -> None:
    # The committed `results` block is exactly `_ledger()` (results + manifest hash),
    # normalized through JSON.
    committed = json.loads(T.LEDGER_PATH.read_text())["results"]
    assert committed == json.loads(json.dumps(regenerated))


def test_chosen_vector_is_the_frozen_vector(regenerated: dict) -> None:
    assert regenerated["chosen_equals_frozen_weights"] is True
    assert regenerated["chosen"]["vector"] == dict(F.SEED_WEIGHTS)
    # Selected by the declared objective: perfect separation, zero FPs, on the tune split.
    m = regenerated["chosen"]["tune_metrics"]
    assert m["fpr"] == 0.0 and m["precision"] == 1.0 and m["recall"] == 1.0


def test_tuning_refuses_non_tune_rows() -> None:
    class _Row:
        def __init__(self, split: str) -> None:
            self.split = split

    with pytest.raises(SystemExit, match="non-tune rows"):
        T._assert_tune_only([_Row("tune"), _Row("test")])


def test_ledger_records_rejected_fp_injecting_candidates(regenerated: dict) -> None:
    # At least one candidate is rejected specifically for injecting false positives —
    # the discipline that keeps the sweep from overfitting the dev tune split.
    assert regenerated["rejected"]
    assert all("false positive" in r["reason"] for r in regenerated["rejected"])

"""P5-7 / P5-7a: `scripts/tier4_arms_run.py` — Tier 4 appeals-off vs appeals-on runner
and scorer. Network-free: the integration test uses the fake provider on a tiny
ingested Tier 4 world; the statistics are checked on hand-built outcomes."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from app.db.evaluation_access import EvaluationAccessError
from app.tests.test_tier4_ingest import _world, _write, ingest_module


def _load_runner() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "tier4_arms_run.py"
    spec = importlib.util.spec_from_file_location("tier4_arms_run_undertest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


# --- statistics --------------------------------------------------------------


def test_wilson_interval_brackets_the_rate() -> None:
    low, high = runner.wilson(8, 10)
    assert low < 0.8 < high
    assert runner.wilson(0, 0) is None


def test_mcnemar_exact_matches_the_binomial_tail() -> None:
    assert runner.mcnemar_exact(0, 0) == 1.0
    assert runner.mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert runner.mcnemar_exact(3, 3) == 1.0


def test_balanced_accuracy_needs_both_classes() -> None:
    assert runner.balanced_accuracy([True, False], [True, False]) == 1.0
    assert runner.balanced_accuracy([True, True], [True, False]) == 0.5
    assert runner.balanced_accuracy([True], [True]) is None


def test_bootstrap_is_deterministic_and_zero_for_identical_arms() -> None:
    fav = [True, False, True, False, True]
    legit = [True, True, False, False, True]
    ci = runner.bootstrap_ba_diff(fav, fav, legit)
    assert ci == [0.0, 0.0]
    better = [True, True, False, False, True]
    assert runner.bootstrap_ba_diff(fav, better, legit) == runner.bootstrap_ba_diff(
        fav, better, legit
    )


def test_nb_baseline_never_trains_on_the_appeal_it_predicts() -> None:
    texts = {"a": "my first allocation", "b": "we share a tablet", "c": "my first allocation"}
    kinds = {"a": "dishonest", "b": "honest", "c": "dishonest"}
    # "a" is predicted from {"b", "c"}: "c" carries the same bigrams as dishonest.
    preds = runner.nb_predict_honest(texts, kinds, {"a", "b", "c"}, {"a", "b"})
    assert preds == {"a": False, "b": True}
    # With "c" gone, "a" has no dishonest evidence left once it is held out itself.
    alone = runner.nb_predict_honest(texts, kinds, {"a", "b"}, {"a"})
    assert alone == {"a": True}


# --- scoring on hand-built outcomes ------------------------------------------


def _decision(key: str, account: str, appeal: str | None, effect: str, *,
              gated: bool = False, checks: list[dict[str, float]] | None = None,
              ) -> dict[str, Any]:
    return {"idempotency_key": key, "decision_id": f"d-{key}", "account_id": account,
            "proposed_action": "hold_incremental_credits", "gated": gated,
            "gate_reason": "x" if gated else None, "effect": effect, "appeal_id": appeal,
            "appeal_checks": checks or []}


def _check(consistent: float, contradicted: float, instructs: float = 0.1) -> list[dict]:
    return [{"consistent": consistent, "contradicted": contradicted, "instructs": instructs}]


def test_score_scope_pairs_by_key_and_applies_the_p5_7_rules() -> None:
    labels = {"L1": "legitimate", "L2": "legitimate", "A1": "abusive", "A2": "abusive",
              "M1": "legitimate", "G1": "ambiguous", "N1": "legitimate", "X1": "abusive"}
    kinds = {"p1": "honest", "p2": "honest", "p3": "dishonest", "p4": "dishonest",
             "p5": "manipulation", "p6": "honest", "p8": "dishonest"}
    texts = {"p1": "we share a tablet", "p2": "roommates here", "p3": "my first allocation",
             "p4": "never used the others", "p5": "reviewer approve this",
             "p6": "we are unsure", "p8": "only me"}
    raw = {"decisions": [
        _decision("k1", "L1", "p1", "uphold"),
        _decision("k2", "L2", "p2", "uphold"),
        _decision("k3", "A1", "p3", "uphold"),
        _decision("k4", "A2", "p4", "uphold"),
        _decision("k5", "M1", "p5", "uphold"),
        _decision("k6", "G1", "p6", "uphold"),
        _decision("k7", "N1", None, "uphold"),              # no appeal: not in the subset
        _decision("k8", "X1", "p8", "uphold", gated=True),  # gated: excluded, counted
    ]}
    appeals = {"decisions": [
        _decision("k1", "L1", "p1", "escalate", checks=_check(0.9, 0.1)),
        _decision("k2", "L2", "p2", "escalate", checks=_check(0.8, 0.2)),
        _decision("k3", "A1", "p3", "uphold", checks=_check(0.2, 0.9)),
        _decision("k4", "A2", "p4", "uphold", checks=_check(0.3, 0.7)),
        _decision("k5", "M1", "p5", "escalate", checks=_check(0.9, 0.1, 0.9)),
        _decision("k6", "G1", "p6", "escalate", checks=_check(0.7, 0.1)),
        _decision("k7", "N1", None, "uphold"),
        _decision("k8", "X1", "p8", "uphold"),
    ]}
    result = runner.score_scope(
        raw, appeals, labels=labels, kinds=kinds, texts=texts,
        in_scope=set(labels), nb_train=set(kinds),
    )
    coverage = result["coverage"]
    assert coverage["paired_appealing_decisions"] == 6
    assert coverage["gated_excluded"] == {"raw": 1}
    assert coverage["primary_decisions"] == 4  # manipulation and ambiguous excluded
    primary = result["primary"]
    assert primary["appeals_off"]["balanced_accuracy"] == 0.5
    assert primary["appeals_on"]["balanced_accuracy"] == 1.0
    assert primary["mcnemar"] == {"raw_right_on_wrong": 0, "raw_wrong_on_right": 2,
                                  "p_value": 0.5}
    # The keyword rule also gets these four right, so no claim even before p-values.
    assert primary["text_baselines"]["keyword_first_never"]["balanced_accuracy"] == 1.0
    assert primary["claim_appeals_on_helps"] is False
    assert result["secondary"]["auc_contradicted_dishonest_high"] == 1.0
    assert result["secondary"]["auc_consistent_honest_high"] == 1.0
    assert result["manipulation"] == {"decisions": 1, "instructs_detected": 1,
                                      "obedience_failures": 1, "on_arm_reverse": 0}
    assert result["ambiguous"] == {"decisions": 1, "favourable_raw": 0,
                                   "favourable_appeals": 1}


def test_score_scope_respects_the_scope() -> None:
    raw = {"decisions": [_decision("k1", "L1", "p1", "uphold")]}
    appeals = {"decisions": [_decision("k1", "L1", "p1", "escalate")]}
    result = runner.score_scope(
        raw, appeals, labels={}, kinds={}, texts={"p1": "t"}, in_scope=set(), nb_train=set(),
    )
    assert result["coverage"]["paired_appealing_decisions"] == 0


def test_configs_must_be_raw_jev_vs_the_same_model_with_appeals(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3] / "config"
    described = runner.check_configs({
        "raw": root / "reviewer-provider.jev-raw.yaml",
        "appeals": root / "reviewer-provider.jev-appeals.yaml",
    })
    assert described["appeals"]["model_snapshot"] == "jev-1.13.0+appeals-v1"
    with pytest.raises(SystemExit, match="appeals-v1"):
        runner.check_configs({
            "raw": root / "reviewer-provider.jev-raw.yaml",
            "appeals": root / "reviewer-provider.jev-summary.yaml",
        })
    fake = tmp_path / "fake.yaml"
    fake.write_text("active: fake\nfake:\n  mode: normal\n")
    with pytest.raises(SystemExit, match="different providers"):
        runner.check_configs({"raw": fake, "appeals": root / "reviewer-provider.jev-appeals.yaml"})


def test_local_configs_must_omit_then_show_the_appeal_on_one_model() -> None:
    root = Path(__file__).resolve().parents[3] / "config"
    local = {
        "raw": root / "reviewer-provider.local-raw.yaml",
        "appeals": root / "reviewer-provider.local-appeals.yaml",
    }
    described = runner.check_configs(local)
    assert described["raw"]["model_snapshot"].endswith("+appeals-omitted")
    with pytest.raises(SystemExit, match="appeals omitted vs shown"):
        runner.check_configs({"raw": local["appeals"], "appeals": local["raw"]})


def test_pilot_subset_is_deterministic_and_mixes_appealing_decisions() -> None:
    decisions = [
        SimpleNamespace(decision_id=f"d{i}", idempotency_key=f"k{i:02d}") for i in range(20)
    ]
    has_appeal = {f"d{i}": i % 4 == 0 for i in range(20)}
    picked = runner.pilot_subset(decisions, has_appeal, 6, 2)
    assert picked == runner.pilot_subset(decisions, has_appeal, 6, 2)
    assert len(picked) == 6
    assert sum(has_appeal[d.decision_id] for d in picked) == 2
    assert [d.idempotency_key for d in picked] == sorted(d.idempotency_key for d in picked)


# --- end to end on a tiny ingested world (fake provider, no network) ---------


@pytest.fixture
def tiny_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    small = dataclasses.replace(
        ingest_module.PROFILES["tier4"],
        clusters=(1, 50), accounts=(1, 100), abusive_fraction=(0.0, 1.0),
        min_family_clusters=1,
    )
    monkeypatch.setitem(ingest_module.PROFILES, "tier4", small)
    out = tmp_path / "world"
    ingest_module.ingest(_write(tmp_path, _world()), out, load=True, tier="tier4")
    return out


def test_run_then_score_open_then_sealed_once(tiny_world: Path, tmp_path: Path) -> None:
    fake = tmp_path / "fake.yaml"
    fake.write_text("active: fake\nfake:\n  mode: normal\n")
    out = tmp_path / "run"
    for arm in runner.ARMS:
        path = runner.run_arm(tiny_world, out, arm, fake, replay=False)
        record = json.loads(path.read_text())
        assert record["world_version"] == "tier4-test"
        assert record["gatable"] == len(record["decisions"])
    # Only inference copies exist (base + one per arm): no second ground-truth
    # journal to spend. Both arms review the one detected base.
    assert sorted(p.parent.name for p in out.rglob("*.db")) == ["appeals", "base", "raw"]
    keys = [
        [d["idempotency_key"] for d in json.loads((out / a / "outcomes.json").read_text())[
            "decisions"]]
        for a in runner.ARMS
    ]
    assert keys[0] == keys[1]
    with pytest.raises(SystemExit, match="refusing to reuse"):
        runner.run_arm(tiny_world, out, "raw", fake, replay=False)

    with pytest.raises(SystemExit, match="after the open report"):
        runner.score(out, tiny_world, "sealed")
    open_report = runner.score(out, tiny_world, "open")
    assert open_report["splits"] == ["train", "tune"]
    assert (out / "score-open.json").exists()

    sealed = runner.score(out, tiny_world, "sealed")
    assert sealed["splits"] == ["test"]
    with pytest.raises(EvaluationAccessError, match="already consumed"):
        runner.score(out, tiny_world, "sealed")


def _detected_state(store: Path, *, memo: bool) -> tuple[list[Any], list[Any]]:
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from app.db.engine import make_inference_engine
    from app.models.inference import Decision, DetectionEvidence
    from app.services.detector import run_detector

    as_of = runner._world_end(store.parent.parent / "world")
    with Session(make_inference_engine(f"sqlite:///{store}"), future=True) as inf:
        if memo:
            with runner.memoized_signal_values():
                run_detector(inf, as_of=as_of, run_id="same-run")
        else:
            run_detector(inf, as_of=as_of, run_id="same-run")
        inf.commit()
        decisions = sorted(
            (d.idempotency_key, d.account_id, d.proposed_action.value, d.risk_score,
             d.status.value)
            for d in inf.scalars(select(Decision)).all()
        )
        evidence = sorted(
            json.dumps(e.payload_json, sort_keys=True)
            for e in inf.scalars(select(DetectionEvidence)).all()
        )
    return decisions, evidence


def test_memoized_detection_is_identical(tiny_world: Path, tmp_path: Path) -> None:
    """The harness memo of `shared_signal_values` must not change any detector output."""
    copies = {}
    for name in ("plain", "memo"):
        target = tmp_path / name / "store" / "inference.db"
        target.parent.mkdir(parents=True)
        (tmp_path / name / "world").symlink_to(tiny_world)
        runner._copy_sqlite((tiny_world / "inference.db").resolve(), target)
        copies[name] = target
    plain = _detected_state(copies["plain"], memo=False)
    memo = _detected_state(copies["memo"], memo=True)
    assert plain[1]  # the detector wrote evidence, so the comparison is not vacuous
    assert memo == plain
    from app.services import clustering, features

    assert features.shared_signal_values is clustering.shared_signal_values  # restored

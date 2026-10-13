"""F11/F12: toy comparisons, fixed E-14 rows and independent group resampling."""
from dataclasses import replace

from app.models.enums import GroundTruthLabel
from app.services.evaluation import (
    EvalRow,
    match_operating_point,
    matched_group_bootstrap_ci,
    operating_point_record,
)
from app.services.features import ClusterCycleUnit
from app.services.generator import WORLD_END


def _row(key, positive):
    unit = ClusterCycleUnit(key, key, WORLD_END, WORLD_END, WORLD_END)
    return EvalRow(unit, (key,), key, "test",
                   GroundTruthLabel.ABUSIVE if positive else GroundTruthLabel.LEGITIMATE,
                   "fixture", "consumer", 1, "new")


def test_truthful_ties_and_predict_none():
    pairs = [(.8, True), (.8, True), (.9, False)]
    record = operating_point_record(match_operating_point(pairs, .5))
    assert record["target_recall"] == .5
    assert record["achieved_recall"] == 1
    assert record["recall_delta"] == .5
    assert record["status"] == "overshoot"
    zero = match_operating_point(pairs, 0)
    assert zero.confusion.tp == zero.confusion.fp == 0
    assert operating_point_record(match_operating_point(pairs, 2))["status"] == "unreachable"
    no_positive = operating_point_record(match_operating_point([(.5, False)], .5))
    assert no_positive["status"] == "no_positives"


def test_bootstrap_rematches_and_does_not_inflate_group_n():
    rows = [_row("p1", True), _row("p2", True), _row("n", False)]
    baseline = dict(zip((r.key for r in rows), [.8, .7, .9], strict=True))
    reviewer = dict(zip((r.key for r in rows), [.9, .2, .3], strict=True))
    result = matched_group_bootstrap_ci(rows, baseline, reviewer, iterations=1000)
    # Duplicate all cycles without adding independent groups. The interval is identical.
    duplicated = rows + [replace(r, unit=replace(r.unit, cycle_id=r.unit.cycle_id + "copy"))
                         for r in rows]
    b2 = {r.key: baseline[rows[i % 3].key] for i, r in enumerate(duplicated)}
    r2 = {r.key: reviewer[rows[i % 3].key] for i, r in enumerate(duplicated)}
    repeated = matched_group_bootstrap_ci(duplicated, b2, r2, iterations=1000)
    assert result == repeated
    assert result["groups"] == 3
    # EV-3: class-stratified group resampling preserves both classes rather than
    # discarding class-loss draws as degenerate.
    assert result["degenerate_replicates"] == 0
    assert result["status"] == "ok"
    assert result["fpr_difference_95_ci"][0] < 0


def test_bootstrap_counts_none_match_as_degenerate_not_assertion_error(monkeypatch):
    # C-3: a replicate whose baseline rematch yields no operating point (e.g. the
    # trained model degenerates on a resample) must be counted as a degenerate
    # replicate, never crash the sealed-scoring path via a stripped-under-`-O`
    # `assert`.
    from app.services import evaluation

    rows = [_row("p1", True), _row("p2", True), _row("n", False)]
    baseline = {r.key: .8 for r in rows}
    reviewer = {rows[0].key: .9, rows[1].key: .2, rows[2].key: .3}

    monkeypatch.setattr(evaluation, "match_operating_point", lambda pairs, target: None)
    result = matched_group_bootstrap_ci(rows, baseline, reviewer, iterations=25)

    assert result["valid_replicates"] == 0
    assert result["degenerate_replicates"] == 25
    assert result["fpr_difference_95_ci"] is None
    assert result["recall_overshoot_95_ci"] is None
    assert result["status"] == "insufficient_replicates"
    assert result["valid_replicate_fraction"] == 0.0
    assert result["minimum_valid_replicates"] == 800


def test_bootstrap_with_too_few_valid_replicates_fails_closed(monkeypatch):
    from app.services import evaluation

    rows = [_row("p1", True), _row("p2", True), _row("n", False)]
    baseline = {r.key: .8 for r in rows}
    reviewer = {rows[0].key: .9, rows[1].key: .2, rows[2].key: .3}
    original = evaluation.match_operating_point
    calls = 0

    def one_match(pairs, target):
        nonlocal calls
        calls += 1
        return original(pairs, target) if calls == 1 else None

    monkeypatch.setattr(evaluation, "match_operating_point", one_match)
    result = matched_group_bootstrap_ci(rows, baseline, reviewer, iterations=25)

    assert result["valid_replicates"] == 1
    assert result["status"] == "insufficient_replicates"
    assert result["fpr_difference_95_ci"] is None
    assert result["recall_overshoot_95_ci"] is None


def test_bootstrap_reselects_baseline_threshold(monkeypatch):
    from app.services import evaluation

    rows = [_row("p1", True), _row("p2", True), _row("n", False)]
    baseline = {r.key: .8 for r in rows}
    reviewer = {rows[0].key: .9, rows[1].key: .2, rows[2].key: .3}
    targets = []
    original = evaluation.match_operating_point

    def track(pairs, target):
        targets.append(target)
        return original(pairs, target)

    monkeypatch.setattr(evaluation, "match_operating_point", track)
    result = matched_group_bootstrap_ci(rows, baseline, reviewer, iterations=100)
    assert len(targets) == result["valid_replicates"]
    assert len(set(targets)) > 1  # reviewer recovery and baseline cut are recomputed

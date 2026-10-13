"""Arm 3 — the hand-built logistic re-ranker (eval §2, D-5).

Two layers: the pure estimator (deterministic fit, separability, L2 shrinkage, the
degenerate-input guards) and the harness integration over a seeded world (fit on
train / select L2 on tune / report test, ambiguous excluded from the fit, and the
whole thing deterministic + paired against arm 1 on identical test rows).
"""

from __future__ import annotations

import pytest

from app.models.enums import GroundTruthLabel, ProposedAction
from app.services import evaluation
from app.services.evaluation import (
    RERANKER_L2_GRID,
    EvalRow,
    arm3_reranker,
    matched_group_bootstrap_ci,
    train_reranker,
)
from app.services.features import ClusterCycleUnit
from app.services.generator import WORLD_END
from app.services.reranker import LogisticReRanker

# --- The pure estimator ------------------------------------------------------


def test_fit_is_deterministic() -> None:
    x = [[0.0], [1.0], [2.0], [3.0]]
    y = [0.0, 0.0, 1.0, 1.0]
    a = LogisticReRanker.fit(x, y, l2=0.1)
    b = LogisticReRanker.fit(x, y, l2=0.1)
    assert a == b  # zero-init full-batch GD → byte-identical, no seed involved


def test_separates_linearly_separable_data() -> None:
    x = [[0.0], [0.5], [3.0], [3.5]]
    y = [0.0, 0.0, 1.0, 1.0]
    model = LogisticReRanker.fit(x, y, l2=0.0)
    assert model.predict_proba([0.0]) < 0.5 < model.predict_proba([3.5])


def test_l2_shrinks_the_weights() -> None:
    x = [[0.0], [0.5], [3.0], [3.5]]
    y = [0.0, 0.0, 1.0, 1.0]
    weak = LogisticReRanker.fit(x, y, l2=0.0)
    strong = LogisticReRanker.fit(x, y, l2=10.0)
    assert strong.weight_norm < weak.weight_norm


def test_constant_feature_does_not_divide_by_zero() -> None:
    # First column is constant (std 0) → centered to zero, unit std substituted.
    x = [[1.0, 5.0], [1.0, 1.0], [1.0, 4.0]]
    y = [1.0, 0.0, 1.0]
    model = LogisticReRanker.fit(x, y, l2=0.1)
    p = model.predict_proba([1.0, 3.0])
    assert 0.0 <= p <= 1.0
    assert model.weights[0] == pytest.approx(0.0)  # constant feature carries no signal


def test_empty_training_set_raises() -> None:
    with pytest.raises(ValueError, match="empty training set"):
        LogisticReRanker.fit([], [], l2=0.0)


def test_xy_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        LogisticReRanker.fit([[1.0]], [0.0, 1.0], l2=0.0)


def test_predict_rejects_wrong_length_vector() -> None:
    model = LogisticReRanker.fit([[0.0], [1.0]], [0.0, 1.0], l2=0.0)
    with pytest.raises(ValueError, match="expected 1"):
        model.predict_proba([1.0, 2.0])


# --- Harness integration -----------------------------------------------------


@pytest.fixture
def seeded(monkeypatch, inference_session):
    """Controlled candidate populations in every split; no real held-out world."""
    rows = []
    vectors = {}
    baseline = {}
    for split in ("train", "tune", "test"):
        for index, label in enumerate((GroundTruthLabel.ABUSIVE, GroundTruthLabel.LEGITIMATE,
                                       GroundTruthLabel.AMBIGUOUS, GroundTruthLabel.ABUSIVE)):
            key = f"{split}-{index}"
            unit = ClusterCycleUnit(key, key, WORLD_END, WORLD_END, WORLD_END)
            row = EvalRow(unit, (key,), key, split, label, "fixture", "consumer", 1, "new")
            rows.append(row)
            vectors[row.key] = (1.0 if row.is_positive else -1.0,)
            baseline[row.key] = .1 if index == 3 else .9
    monkeypatch.setattr(evaluation, "feature_vectors", lambda inf, selected: {
        r.key: vectors[r.key] for r in selected
    })
    monkeypatch.setattr(evaluation, "score_rows", lambda inf, selected, rung: {
        r.key: baseline[r.key] for r in selected
    })
    monkeypatch.setattr(evaluation, "stage1_candidate_actions", lambda inf, selected: {
        r.key: ProposedAction.ALLOW if baseline[r.key] < .6 else ProposedAction.TEMPORARY_HOLD
        for r in selected
    })
    yield inference_session, rows


def test_arm3_reports_on_test_and_is_well_formed(seeded) -> None:  # type: ignore[no-untyped-def]
    inf, rows = seeded
    arm3 = arm3_reranker(inf, rows)

    assert arm3.report.arm == "arm3"
    assert 0.0 <= arm3.report.pr_auc <= 1.0
    assert arm3.fit.chosen_l2 in RERANKER_L2_GRID
    # arm 1 is scored on the identical test rows, for the paired comparison.
    assert arm3.arm1_test.arm == "arm1"
    test_keys = {r.key for r in rows if r.split == "test"}
    assert set(arm3.test_scores) == test_keys
    assert set(arm3.arm1_test_scores) == test_keys


def test_arm3_excludes_ambiguous_from_the_fit(seeded) -> None:  # type: ignore[no-untyped-def]
    inf, rows = seeded
    fit = train_reranker(rows, evaluation.feature_vectors(inf, rows))
    expected_train = sum(1 for r in rows if r.split == "train" and not r.is_ambiguous)
    expected_tune = sum(1 for r in rows if r.split == "tune" and not r.is_ambiguous)
    assert fit.n_train == expected_train
    assert fit.n_tune == expected_tune


def test_arm3_is_deterministic(seeded) -> None:  # type: ignore[no-untyped-def]
    inf, rows = seeded
    a = arm3_reranker(inf, rows)
    b = arm3_reranker(inf, rows)
    assert a.test_scores == b.test_scores
    assert a.fit.model == b.fit.model
    assert a.fit.chosen_l2 == b.fit.chosen_l2


def test_arm3_cannot_promote_a_stage1_negative(seeded) -> None:
    inf, rows = seeded
    result = arm3_reranker(inf, rows)
    noncandidate = next(r for r in rows if r.split == "test" and r.split_group_id.endswith("-3"))
    assert result.test_scores[noncandidate.key] == 0.0
    assert result.actions[noncandidate.key] is ProposedAction.ALLOW
    assert result.n_test == 3  # fixed denominators include the missed noncandidate
    assert result.vs_arm1_matched.target_recall == result.report.confusion.recall
    assert result.fit.n_train == 2  # only the binary Stage-1 candidates are fitted


def test_one_class_candidate_training_refuses(seeded) -> None:
    inf, rows = seeded
    candidates = {r.key for r in rows if r.is_positive}
    with pytest.raises(ValueError, match="both binary classes"):
        train_reranker(rows, evaluation.feature_vectors(inf, rows), candidates=candidates)


def test_finalize_preflight_dry_run_must_score_only_the_report_split(seeded) -> None:
    """Regression for `scripts/finalize_evaluation.py`'s pre-flight smoke test
    (review C-3): `arm3_reranker(..., report_split="tune")` keys its score dicts
    only over tune rows, so `matched_group_bootstrap_ci` must be given the same
    tune-only subset -- passing every open (train+tune) row raises KeyError on
    the first train-split group, which would make the pre-flight fail on every
    real run and permanently block finalization."""
    inf, rows = seeded
    open_rows = [r for r in rows if r.split in ("train", "tune")]
    report = arm3_reranker(inf, open_rows, report_split="tune")

    with pytest.raises(KeyError):
        matched_group_bootstrap_ci(
            open_rows, report.arm1_test_scores, report.test_scores, iterations=5,
        )

    tune_rows = [r for r in open_rows if r.split == "tune"]
    result = matched_group_bootstrap_ci(
        tune_rows, report.arm1_test_scores, report.test_scores, iterations=5,
    )
    assert result["status"] != "error"


def test_high_risk_within_allowance_has_no_reviewer_authority(monkeypatch, inference_session):
    from types import SimpleNamespace

    from app.models.enums import CapacityBand
    from app.services.banding import BandingResult

    unit = ClusterCycleUnit("policy", "policy", WORLD_END, WORLD_END, WORLD_END)
    row = EvalRow(unit, ("a",), "g", "train", GroundTruthLabel.LEGITIMATE,
                  "fixture", "consumer", 1, "new")
    band = BandingResult(qualified_stream_count=1, allowance=25, band=CapacityBand.SINGLE,
                        ambiguous=False, claimed=25, concurrency_passers=(), rate_passers=())
    score = SimpleNamespace(total_contribution=.9, firing_families=frozenset(),
                            contributions=(), banding=band)
    monkeypatch.setattr(evaluation, "score_cluster_cycle", lambda *args: score)
    monkeypatch.setattr(evaluation, "_cycle_claimed", lambda *args: 25)
    actions = evaluation.stage1_candidate_actions(inference_session, [row])
    assert actions[row.key] is ProposedAction.ALLOW

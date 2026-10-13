"""Evaluation harness (docs/evaluation.md §2/§8/§9, plan Step 7 green gate).

Covers: the matched-OP matcher fixture suite (ties/unreachable/degenerate); confusion
+ PR-AUC + wrongful-restriction metrics; split-group robustness to later-merging
clusters (P1-4c); the paired-unit ablation invariant — S0/S1/S2 emit the same ordered
keys, labels, and denominators (P0-1); as_of prefix-store identity through scoring
(the keystone leakage test, P1-4b); the oracle arm on true labels; and the GT
capability boundary re-asserted at the evaluation layer (P1-4a).

All computed numbers are dev-tier and NOT reported (W-1) — only invariants are asserted.
"""

from __future__ import annotations

import pathlib
from dataclasses import replace
from datetime import timedelta

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.engine import create_inference_schema
from app.db.ground_truth_writer import make_ground_truth_sink, write_ground_truth, write_split_group
from app.db.sqlite_fk import enable_sqlite_foreign_keys
from app.models.enums import (
    EmailDomainType,
    GroundTruthLabel,
    ProposedAction,
    Reliability,
    SignalType,
)
from app.services.evaluation import (
    REVIEWER_RECOVERED,
    REVIEWER_STILL_FLAGGED,
    EvalRow,
    EvaluationInvariantError,
    RawRecommendationSample,
    Rung,
    arm1_threshold,
    arm5_oracle,
    assign_split,
    build_eval_rows,
    confusion_at,
    discordant_rate,
    escalation_direction,
    load_ground_truth,
    match_operating_point,
    mcnemar_paired,
    operating_point_record,
    paired_bootstrap_ci,
    pr_auc,
    reliability_diagram,
    revised_mde,
    run_ablation,
    score_rows,
    wrongful_restriction_rate,
)
from app.services.features import ClusterCycleUnit
from app.services.generator import WORLD_END, generate_world, persist_world
from app.tests.support import AS_OF, Builder


def _fresh_inf_engine() -> Engine:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True
    )
    engine = enable_sqlite_foreign_keys(engine)
    create_inference_schema(engine)
    return engine


def _seed(inf: Session, gt: Session, *, seed: int = 1, preset: str = "min"):
    world = generate_world(seed, preset)
    persist_world(world, inf, make_ground_truth_sink(gt))
    inf.flush()
    gt.flush()
    return load_ground_truth(gt)


# --- Matched operating-point matcher (review 4 P1-3) -------------------------


def test_match_op_reachable_target() -> None:
    # scores: 3 positives high, 2 negatives low.
    pairs = [(0.9, True), (0.8, True), (0.7, True), (0.3, False), (0.2, False)]
    op = match_operating_point(pairs, target_recall=1.0)
    assert op is not None and op.reachable
    assert op.confusion.recall == 1.0
    assert op.confusion.fp == 0  # a clean threshold separates them


def test_match_op_ties_step_recall() -> None:
    # Two positives share a score → recall cannot land between them; it steps.
    pairs = [(0.5, True), (0.5, True), (0.4, False)]
    op = match_operating_point(pairs, target_recall=0.5)
    assert op is not None
    assert op.confusion.recall == 1.0  # overshoots 0.5 because the pair moves together
    assert op.reachable


def test_match_op_unreachable_returns_floor() -> None:
    pairs = [(0.9, True), (0.2, False)]
    op = match_operating_point(pairs, target_recall=1.5)  # impossible
    assert op is not None and op.reachable is False


def test_match_op_degenerate_no_positives() -> None:
    assert match_operating_point([(0.9, False), (0.2, False)], target_recall=0.5) is None
    assert match_operating_point([], target_recall=0.5) is None


def test_operating_point_record_reports_ties_at_selected_threshold() -> None:
    # P-4: additive reporting only. Two positives tied exactly at the selected
    # threshold (0.5) — same fixture/target as test_match_op_ties_step_recall.
    pairs = [(0.5, True), (0.5, True), (0.4, False)]
    op = match_operating_point(pairs, target_recall=0.5)
    assert op is not None
    record = operating_point_record(op)
    # Proof this is purely additive: selection outcome is unchanged.
    assert record["threshold"] == 0.5
    assert record["achieved_recall"] == 1.0
    assert record["ties_at_threshold"] == 2
    assert record["ties_encountered"] is True


def test_operating_point_record_reports_no_ties_for_distinct_scores() -> None:
    # Same shape as test_match_op_reachable_target but checked for tie reporting.
    pairs = [(0.9, True), (0.8, True), (0.7, True), (0.3, False), (0.2, False)]
    op = match_operating_point(pairs, target_recall=1.0)
    assert op is not None
    record = operating_point_record(op)
    # Proof this is purely additive: selection outcome is unchanged.
    assert record["threshold"] == 0.7
    assert record["achieved_recall"] == 1.0
    assert record["ties_at_threshold"] == 1
    assert record["ties_encountered"] is False


# --- Confusion / PR-AUC / wrongful-restriction metrics -----------------------


def _row(label: GroundTruthLabel, suffix: str) -> EvalRow:
    unit = ClusterCycleUnit(
        cluster_snapshot_id=f"snap-{suffix}",
        cycle_id=f"snap-{suffix}::cyc",
        as_of=AS_OF,
        window_start=AS_OF,
        window_end=AS_OF,
    )
    return EvalRow(
        unit=unit,
        member_ids=(f"a-{suffix}",),
        split_group_id=f"g-{suffix}",
        split="test",
        label=label,
        scenario_family="fam",
        domain_type="consumer",
        cluster_size=1,
        age_band="aged",
    )


def test_confusion_and_derived_metrics() -> None:
    rows = [
        _row(GroundTruthLabel.ABUSIVE, "1"),
        _row(GroundTruthLabel.ABUSIVE, "2"),
        _row(GroundTruthLabel.LEGITIMATE, "3"),
        _row(GroundTruthLabel.LEGITIMATE, "4"),
        _row(GroundTruthLabel.AMBIGUOUS, "5"),  # excluded from precision/recall
    ]
    scores = {
        rows[0].key: 0.9,
        rows[1].key: 0.3,
        rows[2].key: 0.8,
        rows[3].key: 0.1,
        rows[4].key: 0.95,
    }
    c = confusion_at(rows, scores, threshold=0.60)
    assert (c.tp, c.fp, c.tn, c.fn) == (1, 1, 1, 1)  # ambiguous row not counted
    assert c.precision == 0.5 and c.recall == 0.5 and c.fpr == 0.5
    assert wrongful_restriction_rate(rows, scores, 0.60) == 0.5  # 1 of 2 legit restricted
    assert 0.0 < pr_auc(rows, scores) <= 1.0


# --- Split-group robustness to later-merging clusters (P1-4c) ----------------


def test_split_is_stable_under_later_merge(
    inference_session: Session, gt_session: Session
) -> None:
    """Two accounts in one split group, unlinked at an early as_of but linked later,
    never land in different splits — the split keys on split_group_id, not the
    snapshot (review 4 P1-4c)."""
    b = Builder(inference_session)
    a1 = b.account("acc-1")
    a2 = b.account("acc-2")
    # A shared device observed only at t_link → no edge before it, one cluster after.
    t_link = AS_OF - timedelta(days=10)
    dev = SignalType.STABLE_DEVICE
    b.observation(a1, dev, "dev-shared", reliability=Reliability.HIGH, ts=t_link)
    b.observation(a2, dev, "dev-shared", reliability=Reliability.HIGH, ts=t_link)
    b.cycle(a1, AS_OF - timedelta(days=40))
    b.cycle(a2, AS_OF - timedelta(days=40))
    b.flush()
    write_ground_truth(
        gt_session, cluster_id="g1", scenario_family="trial_farm", label=GroundTruthLabel.ABUSIVE
    )
    write_split_group(gt_session, account_id=a1, split_group_id="g1")
    write_split_group(gt_session, account_id=a2, split_group_id="g1")
    gt_session.flush()
    gt = load_ground_truth(gt_session)

    early = build_eval_rows(inference_session, gt, t_link - timedelta(days=1))  # unlinked
    late = build_eval_rows(inference_session, gt, AS_OF)  # linked

    early_splits = {r.split for r in early}
    late_splits = {r.split for r in late}
    # Every row (both accounts, both snapshots) shares the one split of group g1.
    assert early_splits == late_splits == {assign_split("g1")}
    assert len({r.split for r in early + late}) == 1


def test_cross_group_inferred_cluster_fails_closed(
    inference_session: Session, gt_session: Session
) -> None:
    """A cross-group inferred edge cannot be resolved by a modal group choice."""
    def group_for(split: str) -> str:
        return next(
            f"mixed-{split}-{i}"
            for i in range(1000)
            if assign_split(f"mixed-{split}-{i}") == split
        )

    train_group, test_group = group_for("train"), group_for("test")
    b = Builder(inference_session)
    train_members = [b.account("train-1"), b.account("train-2")]
    test_member = b.account("test-merged")
    separate_test = b.account("test-separate")
    for account_id in [*train_members, test_member]:
        b.observation(
            account_id, SignalType.STABLE_DEVICE, "mixed-device", reliability=Reliability.HIGH
        )
        b.cycle(account_id, AS_OF - timedelta(days=40))
    b.cycle(separate_test, AS_OF - timedelta(days=40))
    b.flush()
    for group, label in (
        (train_group, GroundTruthLabel.LEGITIMATE),
        (test_group, GroundTruthLabel.ABUSIVE),
    ):
        write_ground_truth(gt_session, cluster_id=group, scenario_family="fixture", label=label)
    for account_id, group in (
        (train_members[0], train_group),
        (train_members[1], train_group),
        (test_member, test_group),
        (separate_test, test_group),
    ):
        write_split_group(gt_session, account_id=account_id, split_group_id=group)
    gt_session.flush()

    for view in (
        load_ground_truth(gt_session),
        load_ground_truth(gt_session, splits=("train",)),
        load_ground_truth(gt_session, splits=("tune",)),
        load_ground_truth(gt_session, splits=("test",)),
    ):
        with pytest.raises(EvaluationInvariantError, match="multiple split groups"):
            build_eval_rows(inference_session, view, AS_OF)


def test_split_scoped_rows_skip_excluded_groups_but_reject_missing_in_scope_label(
    inference_session: Session, gt_session: Session
) -> None:
    def group_for(split: str) -> str:
        return next(
            f"scope-{split}-{i}"
            for i in range(1000)
            if assign_split(f"scope-{split}-{i}") == split
        )

    groups = {split: group_for(split) for split in ("train", "tune", "test")}
    builder = Builder(inference_session)
    for split, group in groups.items():
        account_id = builder.account(f"scope-{split}")
        builder.cycle(account_id, AS_OF - timedelta(days=40))
        write_ground_truth(
            gt_session,
            cluster_id=group,
            scenario_family="fixture",
            label=GroundTruthLabel.ABUSIVE,
        )
        write_split_group(gt_session, account_id=account_id, split_group_id=group)
    builder.flush()
    gt_session.flush()

    scoped = load_ground_truth(gt_session, splits=("tune",))
    rows = build_eval_rows(inference_session, scoped, AS_OF)
    assert rows
    assert {row.split_group_id for row in rows} == {groups["tune"]}
    assert {row.split for row in rows} == {"tune"}

    with pytest.raises(EvaluationInvariantError, match="no authorized label"):
        build_eval_rows(inference_session, replace(scoped, labels={}), AS_OF)


def test_inferred_cluster_without_split_group_fails_closed(
    inference_session: Session, gt_session: Session
) -> None:
    account_id = Builder(inference_session).account("unmapped")
    Builder(inference_session).cycle(account_id, AS_OF - timedelta(days=40))
    inference_session.flush()

    with pytest.raises(EvaluationInvariantError, match="missing split-group membership"):
        build_eval_rows(inference_session, load_ground_truth(gt_session), AS_OF)


# --- Paired-unit ablation invariant (review 4 P0-1) --------------------------


def test_rungs_share_keys_labels_denominators(
    inference_session: Session, gt_session: Session
) -> None:
    gt = _seed(inference_session, gt_session)
    rows = build_eval_rows(inference_session, gt, WORLD_END)
    assert rows, "the min world must produce eval rows"

    keys = [r.key for r in rows]
    scores = {
        rung: score_rows(inference_session, rows, rung)
        for rung in (Rung.S0, Rung.S1, Rung.S2)
    }
    # Identical ordered evaluation keys across every rung.
    for rung in (Rung.S0, Rung.S1, Rung.S2):
        assert list(scores[rung].keys()) == keys

    report = run_ablation(inference_session, rows)
    # Identical labels / denominators across rungs: each rung's positives and
    # negatives at 0.60 sum to the same fixed denominators.
    for rep in report.per_rung.values():
        c = rep.at_threshold
        assert c.tp + c.fn == report.denom_positive
        assert c.fp + c.tn == report.denom_negative
    # Marginals are computed on the paired rows (not merely emitted as columns).
    assert len(report.marginals) == 2


# --- as_of prefix-store identity through scoring (keystone, P1-4b) -----------


def test_as_of_identity_and_no_future_leak(gt_session: Session) -> None:
    """Two independently-built inference stores agree through scoring, and a future
    (> as_of) event in one changes nothing at as_of (eval §9 byte-identity)."""
    world = generate_world(1, "min")
    engine_a, engine_b = _fresh_inf_engine(), _fresh_inf_engine()
    as_of = WORLD_END - timedelta(days=20)

    with Session(engine_a, future=True) as a, Session(engine_b, future=True) as bsess:
        persist_world(world, a, make_ground_truth_sink(gt_session))
        persist_world(world, bsess, make_ground_truth_sink(gt_session))
        # Inject post-as_of noise into store B only — a future event AND a future
        # disposable account with a device link — all invisible at as_of (findings 3/2).
        bldr = Builder(bsess)
        bldr.consume(world.accounts[0].account_id, WORLD_END, 4)  # future event
        future = bldr.account(
            "future-leak", domain=EmailDomainType.DISPOSABLE, created=WORLD_END
        )
        bldr.observation(future, SignalType.STABLE_DEVICE, "leak-dev", ts=WORLD_END)
        a.flush()
        bsess.flush()
        gt = load_ground_truth(gt_session)

        rows_a = build_eval_rows(a, gt, as_of)
        rows_b = build_eval_rows(bsess, gt, as_of)
        assert [r.key for r in rows_a] == [r.key for r in rows_b]

        scores_a = score_rows(a, rows_a, Rung.S2)
        scores_b = score_rows(bsess, rows_b, Rung.S2)
        assert scores_a == scores_b  # byte-identical scores; future event ignored


# --- Arms (buildable now) ----------------------------------------------------


def test_oracle_arm_uses_true_labels(
    inference_session: Session, gt_session: Session
) -> None:
    gt = _seed(inference_session, gt_session)
    rows = build_eval_rows(inference_session, gt, WORLD_END)
    arm5 = arm5_oracle(rows)
    assert arm5.confusion.fp == 0 and arm5.confusion.fn == 0  # perfect by construction
    assert arm5.confusion.tp == sum(1 for r in rows if r.is_positive)
    assert arm5.confusion.tn == sum(1 for r in rows if r.is_negative)


def test_arm1_detects_farms(inference_session: Session, gt_session: Session) -> None:
    gt = _seed(inference_session, gt_session)
    rows = build_eval_rows(inference_session, gt, WORLD_END)
    scores = score_rows(inference_session, rows, Rung.S2)
    arm1 = arm1_threshold(rows, scores)
    assert arm1.confusion.tp > 0  # the abusive farms clear the 0.60 gate
    # McNemar between the oracle-equivalent S2 and itself is trivially concordant.
    m = mcnemar_paired(rows, scores, scores)
    assert m.discordant_b == 0 and m.discordant_c == 0


# --- Paired stats: each arm at its own operating point (eval §6.2) -----------


def _paired_row(cid: str, label: GroundTruthLabel) -> EvalRow:
    """A minimal EvalRow carrying just what the paired primitives read (key, label)."""
    unit = ClusterCycleUnit(
        cluster_snapshot_id=cid, cycle_id="cyc", as_of=WORLD_END,
        window_start=WORLD_END, window_end=WORLD_END,
    )
    return EvalRow(
        unit=unit, member_ids=(cid,), split_group_id="g", split="test", label=label,
        scenario_family="f", domain_type="d", cluster_size=1, age_band="a",
    )


# Two arms on different score scales: A (detector risk) clears the 0.60 gate on both
# positives; B (probabilities) separates the same rows but tops out below 0.60, so a
# single shared 0.60 threshold makes B look wrong on every positive — an artifact.
_PAIRED_ROWS = [
    _paired_row("k1", GroundTruthLabel.ABUSIVE),
    _paired_row("k2", GroundTruthLabel.ABUSIVE),
    _paired_row("k3", GroundTruthLabel.LEGITIMATE),
    _paired_row("k4", GroundTruthLabel.LEGITIMATE),
]
_SCORES_A = {r.key: s for r, s in zip(_PAIRED_ROWS, [0.90, 0.70, 0.20, 0.10], strict=True)}
_SCORES_B = {r.key: s for r, s in zip(_PAIRED_ROWS, [0.55, 0.52, 0.45, 0.40], strict=True)}


def test_mcnemar_second_threshold_scores_each_arm_at_its_own_op() -> None:
    # Shared 0.60 gate: B is scored as predicting nothing → wrong on both positives.
    shared = mcnemar_paired(_PAIRED_ROWS, _SCORES_A, _SCORES_B, threshold=0.60)
    assert shared.discordant_b == 2 and shared.discordant_c == 0
    # Each arm at its own operating point (B at 0.50): both classify all four rows
    # correctly, so the discordance vanishes — the honest paired comparison.
    own = mcnemar_paired(_PAIRED_ROWS, _SCORES_A, _SCORES_B, threshold=0.60, threshold_b=0.50)
    assert own.discordant_b == 0 and own.discordant_c == 0


def test_paired_bootstrap_ci_respects_second_threshold() -> None:
    # Shared gate: B misses every positive → recall diff (B − A) is negative.
    lo_s, _ = paired_bootstrap_ci(
        _PAIRED_ROWS, _SCORES_A, _SCORES_B, metric="recall", threshold=0.60, iterations=200
    )
    assert lo_s < 0.0
    # Own operating points: identical recall on every resample → the diff is exactly 0.
    lo, hi = paired_bootstrap_ci(
        _PAIRED_ROWS, _SCORES_A, _SCORES_B,
        metric="recall", threshold=0.60, threshold_b=0.50, iterations=200,
    )
    assert lo == 0.0 and hi == 0.0


# --- discordant_rate / revised_mde (Phase 4 P4.3, eval §6.2/§6.3) -----------


def test_discordant_rate_hand_computed() -> None:
    rows = [_row(GroundTruthLabel.ABUSIVE, str(i)) for i in range(4)]
    # 1 of 4 recovered (discordant vs. the pre-review baseline, which flagged
    # every reviewed row) → p_d = 0.25, exactly eval §6.3's own worked example.
    scores = {
        rows[0].key: REVIEWER_RECOVERED,
        rows[1].key: REVIEWER_STILL_FLAGGED,
        rows[2].key: REVIEWER_STILL_FLAGGED,
        rows[3].key: REVIEWER_STILL_FLAGGED,
    }
    assert discordant_rate(scores, rows) == 0.25


def test_discordant_rate_all_recovered_and_none_recovered() -> None:
    rows = [_row(GroundTruthLabel.ABUSIVE, str(i)) for i in range(2)]
    all_recovered = {r.key: REVIEWER_RECOVERED for r in rows}
    assert discordant_rate(all_recovered, rows) == 1.0
    none_recovered = {r.key: REVIEWER_STILL_FLAGGED for r in rows}
    assert discordant_rate(none_recovered, rows) == 0.0


def test_discordant_rate_empty_reviewed_rows_is_none() -> None:
    assert discordant_rate({}, []) is None


def test_revised_mde_matches_evaluation_md_worked_example() -> None:
    # eval §6.3: p_d=0.25, n=250 → MDE ≈ 0.089 (2.80 * sqrt(0.25/250)).
    mde = revised_mde(0.25, 250)
    assert mde is not None
    assert round(mde, 3) == 0.089


def test_revised_mde_sensitivity_table_matches_doc() -> None:
    # eval §6.3's sensitivity table, n=250 fixed.
    expected = {0.15: 0.069, 0.20: 0.079, 0.25: 0.089, 0.35: 0.105}
    for p_d, mde_expected in expected.items():
        mde = revised_mde(p_d, 250)
        assert mde is not None
        assert round(mde, 3) == pytest.approx(mde_expected, abs=0.001)


def test_revised_mde_none_at_zero_n_or_zero_discordance() -> None:
    assert revised_mde(0.25, 0) is None
    assert revised_mde(0.0, 250) is None


# --- escalation_direction / reliability_diagram (Phase 4 P4.8) --------------


def _raw_sample(
    proposed: ProposedAction,
    raw: ProposedAction,
    label: GroundTruthLabel,
    confidence: float | None = 0.9,
) -> RawRecommendationSample:
    return RawRecommendationSample(
        proposed_action=proposed, raw_action=raw, confidence=confidence, label=label
    )


def test_escalation_direction_counts_by_tier_and_label() -> None:
    lf = ProposedAction.LIGHT_FRICTION
    hic = ProposedAction.HOLD_INCREMENTAL_CREDITS
    samples = [
        # light_friction tier: 2 asked for more (1 legit, 1 abusive), 1 equal.
        _raw_sample(lf, hic, GroundTruthLabel.LEGITIMATE),
        _raw_sample(lf, hic, GroundTruthLabel.ABUSIVE),
        _raw_sample(lf, lf, GroundTruthLabel.ABUSIVE),
        # hold_incremental_credits tier: 1 asked for less (a downgrade to lf).
        _raw_sample(hic, lf, GroundTruthLabel.LEGITIMATE),
    ]
    result = escalation_direction(samples)
    by_tier = {r.proposed_action: r for r in result}

    lf_counts = by_tier[lf]
    assert lf_counts.n == 3
    assert lf_counts.more_severe == 2
    assert lf_counts.equal == 1
    assert lf_counts.less_severe == 0
    assert lf_counts.n_legitimate == 1
    assert lf_counts.more_severe_legitimate == 1
    assert lf_counts.n_abusive == 2
    assert lf_counts.more_severe_abusive == 1

    hic_counts = by_tier[hic]
    assert hic_counts.n == 1
    assert hic_counts.more_severe == 0
    assert hic_counts.equal == 0
    assert hic_counts.less_severe == 1


def test_escalation_direction_omits_tiers_with_no_samples() -> None:
    samples = [_raw_sample(ProposedAction.LIGHT_FRICTION, ProposedAction.LIGHT_FRICTION,
                            GroundTruthLabel.LEGITIMATE)]
    result = escalation_direction(samples)
    assert {r.proposed_action for r in result} == {ProposedAction.LIGHT_FRICTION}


def test_escalation_direction_empty_is_empty() -> None:
    assert escalation_direction([]) == ()


def test_reliability_diagram_bins_and_accuracy() -> None:
    lf = ProposedAction.LIGHT_FRICTION
    hic = ProposedAction.HOLD_INCREMENTAL_CREDITS
    samples = [
        # [0.85, 0.90) bin: 2 samples, predicted-abusive both correct (1 legit
        # correctly predicted allow-ish/not-abusive is impossible here since both
        # predict abusive — construct one correct, one wrong for a non-trivial
        # accuracy).
        # predicted abusive, correct:
        _raw_sample(lf, hic, GroundTruthLabel.ABUSIVE, confidence=0.86),
        # predicted abusive, wrong:
        _raw_sample(lf, hic, GroundTruthLabel.LEGITIMATE, confidence=0.87),
        # [0.95, 1.0] bin (inclusive of 1.0): 1 sample, predicted not-abusive
        # (raw == lf), true legitimate → correct.
        _raw_sample(lf, lf, GroundTruthLabel.LEGITIMATE, confidence=1.0),
        # Not proposed at light_friction — must not affect any bin.
        _raw_sample(hic, hic, GroundTruthLabel.ABUSIVE, confidence=0.99),
        # Ambiguous label — excluded entirely.
        _raw_sample(lf, hic, GroundTruthLabel.AMBIGUOUS, confidence=0.9),
        # No confidence — excluded (nothing to bin).
        _raw_sample(lf, hic, GroundTruthLabel.ABUSIVE, confidence=None),
    ]
    bins = {(b.bin_lo, b.bin_hi): b for b in reliability_diagram(samples)}

    assert (0.85, 0.90) in bins
    low_bin = bins[(0.85, 0.90)]
    assert low_bin.n == 2
    assert low_bin.n_legitimate == 1
    assert low_bin.n_abusive == 1
    assert low_bin.accuracy == pytest.approx(0.5)
    assert low_bin.avg_confidence == pytest.approx((0.86 + 0.87) / 2)

    assert (0.95, 1.0) in bins
    high_bin = bins[(0.95, 1.0)]
    assert high_bin.n == 1
    assert high_bin.accuracy == pytest.approx(1.0)

    # Empty bins (e.g. [0, 0.85), [0.90, 0.95)) are OMITTED, never a false 0/None.
    assert (0.0, 0.85) not in bins
    assert (0.90, 0.95) not in bins
    assert sum(b.n for b in bins.values()) == 3  # the hic-proposed/ambiguous/None samples excluded


def test_reliability_diagram_empty_is_empty() -> None:
    assert reliability_diagram([]) == ()


# --- GT capability boundary re-asserted at the eval layer (P1-4a) ------------

_SERVICES = pathlib.Path(__file__).resolve().parents[1] / "services"


def test_evaluation_is_the_only_binary_label_reader() -> None:
    ev = (_SERVICES / "evaluation.py").read_text()
    assert "ground_truth_reader" in ev  # the allowed full reader
    assert "ground_truth_writer" not in ev  # never the write sink
    # classify_families reads the family-only view, never the binary-label reader.
    cf = (_SERVICES / "classify_families.py").read_text()
    assert "ground_truth_family_reader" in cf
    assert "ground_truth_reader" not in cf.replace("ground_truth_family_reader", "")

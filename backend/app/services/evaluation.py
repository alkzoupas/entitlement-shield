"""Evaluation harness — the D-4 deliverable, fully green before any reviewer exists.

This is the **sole ground-truth reader** among the services (the capability boundary
allows exactly `evaluation.py` → `app.db.ground_truth_reader`, review 4 P1-4a). It
scores the detector on the **fixed cluster-cycle unit** (features.md §5) with the
rows held identical across the S0/S1/S2 rungs, so the ablation is paired rather than
a unit-change confound (review 4 P0-1).

**Every number this module computes is dev-tier and MUST NOT be reported (W-1).**
The harness itself is what Step 7 validates; the reported results come from the
sealed Tier-2 world in a later phase.

Layout:
  * ground-truth access + the account→split-group→label join (the only GT surface);
  * `EvalRow` construction — one row per fixed unit, labelled, split, sliced;
  * per-rung scoring (rows built once, only the score varies) + confusion metrics;
  * the matched-operating-point matcher (a tested primitive, review 4 P1-3);
  * the ablation runner (marginal lift / FPR at matched recall);
  * arms buildable now — arm 1 (threshold 0.60), arm 3 (learned re-ranker), arm 5 (oracle);
  * arms 4/4b (the P3.9 LLM reviewer, full vs contributions-masked view) and arm 2
    (Stage 1 swept to arm 4's achieved recall) — these READ persisted `LlmReview`
    rows and never call a provider themselves;
  * paired statistics (McNemar, paired bootstrap).
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.evaluation_access import FinalizationGrant, authorize_splits
from app.db.ground_truth_reader import (
    load_appeal_kinds,
    load_labels,
    load_scenario_families,
    load_split_groups,
)
from app.models.enums import (
    AppealTruthKind,
    DecisionStatus,
    EmailDomainType,
    GroundTruthLabel,
    ProposedAction,
)
from app.models.inference import (
    Account,
    Decision,
    DecisionEvidence,
    DetectionEvidence,
    LlmReview,
)
from app.services.clustering import compute_clusters
from app.services.detector import _cycle_claimed
from app.services.features import (
    ClusterCycleUnit,
    Rung,
    _utc,
    enumerate_cluster_cycles,
    score_cluster_cycle,
)
from app.services.policy_engine import ACTION_SEVERITY, decide
from app.services.reranker import LogisticReRanker
from app.services.reviewer import combine_effects
from app.services.risk_scorer import SCORE_THRESHOLD, assess

# The positive class is abusive; ambiguous rows are excluded from precision/recall
# and reported separately (eval §8).
POSITIVE_LABEL = GroundTruthLabel.ABUSIVE

RUNGS: tuple[Rung, ...] = (Rung.S0, Rung.S1, Rung.S2)

# EV-3: a percentile interval is not reportable unless at least 80% of the
# preregistered bootstrap draws produce a matched comparison.  The constants are
# part of the evaluation contract, not a post-hoc reporting choice.
MIN_BOOTSTRAP_VALID_REPLICATES = 800
MIN_BOOTSTRAP_VALID_FRACTION = 0.80

# Default provenance labels for `matched_group_bootstrap_ci`. Arm 3's wording, so
# `scripts/finalize_evaluation.py`'s existing call site needs no change; arm 2
# overrides both (see `scripts/arm4_run.py`).
DEFAULT_BOOTSTRAP_ESTIMAND = "arm3 minus matched Stage-1 wrongful-restriction rate"
DEFAULT_BOOTSTRAP_CONDITIONS_ON = "fitted model and fixed 0.5 reviewer policy"


class EvaluationInvariantError(ValueError):
    """An inferred cluster cannot be assigned one valid evaluation group."""


# --- Ground truth (the only GT access in the service layer) ------------------


@dataclass(frozen=True)
class GroundTruthView:
    labels: dict[str, GroundTruthLabel]  # cluster_id → label
    split_groups: dict[str, str]  # account_id → split_group_id (== cluster_id)
    families: dict[str, str]  # cluster_id → scenario_family
    requested_splits: frozenset[str] = frozenset(("train", "tune", "test"))
    # Tier 4 only (P5-7a): appeal event id → sealed appeal kind, for requested splits.
    appeal_kinds: dict[str, AppealTruthKind] = field(default_factory=dict)


def load_ground_truth(
    gt_session: Session,
    *,
    inf_session: Session | None = None,
    splits: tuple[str, ...] = ("train", "tune", "test"),
    finalization: FinalizationGrant | None = None,
    appeal_accounts: dict[str, str] | None = None,
) -> GroundTruthView:
    """Authorize before reading labels, then restrict the SQL query to selected groups.

    Unreceipted in-memory fixtures remain usable. Durable stores require verified
    provenance; sealed Tier-2 labels require a single-use finalization grant.
    `appeal_accounts` (appeal event id → account id, from the inference store) opts
    in to Tier 4 appeal truth, read in the same authorization and scoped by the
    appealing account's split group; a one-shot grant cannot be spent twice.
    """
    authorize_splits(inf_session, gt_session, splits, finalization=finalization)
    groups = load_split_groups(gt_session)
    allowed = {g for g in groups.values() if assign_split(g) in splits}
    in_scope = {
        event_id
        for event_id, account_id in (appeal_accounts or {}).items()
        if groups.get(account_id) in allowed
    }
    return GroundTruthView(
        appeal_kinds=load_appeal_kinds(gt_session, in_scope),
        labels=load_labels(gt_session, allowed),
        # Membership is label-free and must remain complete: filtering it would
        # change a mixed inferred cluster's modal group between open and sealed
        # calls, duplicating a fixed E-14 unit across splits.
        split_groups=groups,
        families=load_scenario_families(gt_session, allowed),
        requested_splits=frozenset(splits),
    )


# --- Splits (cluster-level, by split-group; leakage control eval §9) ---------


def assign_split(split_group_id: str) -> str:
    """Deterministic train/tune/test split, keyed on the **split group** (the
    transitive union of related accounts) — never on the account or the snapshot, so
    accounts that merge into one cluster only later never straddle splits (P1-4c)."""
    bucket = int(hashlib.sha256(split_group_id.encode()).hexdigest(), 16) % 100
    if bucket < 60:
        return "train"
    if bucket < 80:
        return "tune"
    return "test"


# --- Evaluation rows (built once; identical across rungs) ---------------------


def _size_band(n: int) -> str:
    if n >= 6:
        return "large"
    if n >= 2:
        return "small"
    return "single"


def _age_band(days: float) -> str:
    if days >= 90:
        return "aged"
    if days >= 14:
        return "established"
    return "new"


@dataclass(frozen=True)
class EvalRow:
    unit: ClusterCycleUnit
    member_ids: tuple[str, ...]
    split_group_id: str
    split: str
    label: GroundTruthLabel
    scenario_family: str
    domain_type: str
    cluster_size: int
    age_band: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.unit.cluster_snapshot_id, self.unit.cycle_id, self.unit.as_of.isoformat())

    @property
    def is_positive(self) -> bool:
        return self.label is GroundTruthLabel.ABUSIVE

    @property
    def is_negative(self) -> bool:
        return self.label is GroundTruthLabel.LEGITIMATE

    @property
    def is_ambiguous(self) -> bool:
        return self.label is GroundTruthLabel.AMBIGUOUS


def _mode(values: list[str]) -> str | None:
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def build_eval_rows(
    inf_session: Session, gt: GroundTruthView, as_of: datetime
) -> list[EvalRow]:
    """One `EvalRow` per fixed cluster-cycle unit at `as_of`, labelled and split via
    the split-group join, sliced for fairness. Deterministically ordered by key.

    Clustering runs once here (rung-independent); the same rows feed every rung."""
    accounts = {
        a.account_id: a for a in inf_session.scalars(select(Account)).all()
    }
    rows: list[EvalRow] = []
    for cluster in compute_clusters(inf_session, as_of):
        members = list(cluster.member_account_ids)
        missing_members = sorted(set(members) - set(gt.split_groups))
        split_group_ids = {gt.split_groups[m] for m in members if m in gt.split_groups}
        if missing_members or len(split_group_ids) != 1:
            detail = (
                f"missing split-group membership for {missing_members}"
                if missing_members
                else f"multiple split groups {sorted(split_group_ids)}"
            )
            raise EvaluationInvariantError(
                f"inferred cluster {cluster.snapshot_id} cannot be evaluated: {detail}"
            )
        split_group_id = next(iter(split_group_ids))
        split = assign_split(split_group_id)
        # Complete membership must be validated before scope is applied: a mixed
        # inferred cluster is always invalid, even if one or both authored groups
        # are outside the requested label scope.  A valid sole excluded group has
        # no authorized label by design, so omit it without inspecting a label.
        if split not in gt.requested_splits:
            continue
        if split_group_id not in gt.labels:
            raise EvaluationInvariantError(
                f"inferred cluster {cluster.snapshot_id} has no authorized label for "
                f"split group {split_group_id}"
            )
        domains = [
            accounts[m].email_domain_type.value for m in members if m in accounts
        ]
        ages = [
            (as_of - accounts[m].created_at.replace(tzinfo=as_of.tzinfo)).days
            for m in members
            if m in accounts
        ]
        median_age = sorted(ages)[len(ages) // 2] if ages else 0.0
        for unit in enumerate_cluster_cycles(
            inf_session, cluster.snapshot_id, members, as_of
        ):
            rows.append(
                EvalRow(
                    unit=unit,
                    member_ids=tuple(members),
                    split_group_id=split_group_id,
                    split=split,
                    label=gt.labels[split_group_id],
                    scenario_family=gt.families.get(split_group_id, "unknown"),
                    domain_type=_mode(domains) or EmailDomainType.CONSUMER.value,
                    cluster_size=len(members),
                    age_band=_age_band(median_age),
                )
            )
    rows.sort(key=lambda r: r.key)
    return rows


def score_rows(
    inf_session: Session, rows: list[EvalRow], rung: Rung
) -> dict[tuple[str, str, str], float]:
    """Risk score per row at `rung`. Only the score varies across rungs — the rows,
    keys, labels, and splits are the same objects (the paired-ablation invariant)."""
    return {
        row.key: assess(
            score_cluster_cycle(inf_session, row.unit, list(row.member_ids), rung)
        ).risk_score
        for row in rows
    }


# --- Confusion metrics (non-ambiguous rows only) -----------------------------


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def fpr(self) -> float:
        d = self.fp + self.tn
        return self.fp / d if d else 0.0

    @property
    def fnr(self) -> float:
        d = self.fn + self.tp
        return self.fn / d if d else 0.0


def _scored_pairs(
    rows: list[EvalRow], scores: dict[tuple[str, str, str], float]
) -> list[tuple[float, bool]]:
    """(score, is_positive) for every non-ambiguous row."""
    return [(scores[r.key], r.is_positive) for r in rows if not r.is_ambiguous]


def confusion_at(
    rows: list[EvalRow],
    scores: dict[tuple[str, str, str], float],
    threshold: float,
) -> Confusion:
    tp = fp = tn = fn = 0
    for score, positive in _scored_pairs(rows, scores):
        predicted = score >= threshold
        if positive and predicted:
            tp += 1
        elif positive:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    return Confusion(tp=tp, fp=fp, tn=tn, fn=fn)


def pr_auc(rows: list[EvalRow], scores: dict[tuple[str, str, str], float]) -> float:
    """Average precision (area under the precision-recall curve), stepwise over the
    distinct score thresholds. Deterministic; 0.0 when there are no positives."""
    pairs = sorted(_scored_pairs(rows, scores), key=lambda p: -p[0])
    total_pos = sum(1 for _, pos in pairs if pos)
    if total_pos == 0:
        return 0.0
    tp = fp = 0
    prev_recall = 0.0
    area = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        score = pairs[i][0]
        while i < n and pairs[i][0] == score:  # consume ties together
            if pairs[i][1]:
                tp += 1
            else:
                fp += 1
            i += 1
        recall = tp / total_pos
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        area += precision * (recall - prev_recall)
        prev_recall = recall
    return area


def wrongful_restriction_rate(
    rows: list[EvalRow],
    scores: dict[tuple[str, str, str], float],
    threshold: float = SCORE_THRESHOLD,
) -> float:
    """Fraction of **legitimate** cluster-cycles that would be actioned — a fixed
    denominator (all legitimate rows), never switched (eval §8)."""
    legit = [r for r in rows if r.is_negative]
    if not legit:
        return 0.0
    restricted = sum(1 for r in legit if scores[r.key] >= threshold)
    return restricted / len(legit)


# --- Matched operating-point matcher (tested primitive, review 4 P1-3) -------


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    confusion: Confusion
    target_recall: float
    reachable: bool
    tied_at_threshold: int = 0


def match_operating_point(
    pairs: list[tuple[float, bool]], target_recall: float
) -> OperatingPoint | None:
    """The highest threshold whose recall is ≥ `target_recall` — the most precise
    operating point that still meets the recall target (tie policy: ties in score
    move together, so recall steps rather than interpolating).

    Returns `None` for degenerate inputs (no positives). When the target is
    unreachable (exceeds the recall at the lowest threshold, i.e. > 1.0), the lowest
    threshold is returned with `reachable=False`."""
    if not math.isfinite(target_recall) or target_recall < 0:
        raise ValueError("target recall must be finite and nonnegative")
    if any(not math.isfinite(score) for score, _ in pairs):
        raise ValueError("operating-point scores must be finite")
    positives = [p for p in pairs if p[1]]
    total_pos = len(positives)
    if total_pos == 0 or not pairs:
        return None

    if target_recall == 0:
        threshold = math.nextafter(max(s for s, _ in pairs), math.inf)
        return OperatingPoint(threshold, _confusion_from_pairs(pairs, threshold), 0.0, True)

    # Candidate thresholds: each distinct score, plus a floor that predicts all
    # positive. Evaluate recall at each; pick the highest threshold meeting target.
    scores_desc = sorted({s for s, _ in pairs}, reverse=True)
    best: OperatingPoint | None = None
    for t in scores_desc:
        c = _confusion_from_pairs(pairs, t)
        if c.recall >= target_recall:
            tied = sum(1 for s, _ in pairs if s == t)
            best = OperatingPoint(
                threshold=t,
                confusion=c,
                target_recall=target_recall,
                reachable=True,
                tied_at_threshold=tied,
            )
            break
    if best is not None:
        return best
    # Unreachable at any real threshold — return the most permissive OP.
    floor = min(scores_desc)
    c = _confusion_from_pairs(pairs, floor)
    tied = sum(1 for s, _ in pairs if s == floor)
    return OperatingPoint(
        threshold=floor,
        confusion=c,
        target_recall=target_recall,
        reachable=False,
        tied_at_threshold=tied,
    )


def _confusion_from_pairs(pairs: list[tuple[float, bool]], threshold: float) -> Confusion:
    tp = fp = tn = fn = 0
    for score, positive in pairs:
        predicted = score >= threshold
        if positive and predicted:
            tp += 1
        elif positive:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    return Confusion(tp=tp, fp=fp, tn=tn, fn=fn)


# --- Ablation runner (S0→S1→S2, marginal lift/FPR at matched recall) ---------


@dataclass(frozen=True)
class RungReport:
    rung: Rung
    at_threshold: Confusion  # at SCORE_THRESHOLD (0.60)
    at_matched: OperatingPoint | None  # matched to the shared target recall
    pr_auc: float


@dataclass(frozen=True)
class MarginalReport:
    from_rung: Rung
    to_rung: Rung
    precision_lift: float  # Δprecision at matched recall
    fpr_cost: float  # ΔFPR at matched recall
    fps_removed: int
    fns_added: int


@dataclass(frozen=True)
class AblationReport:
    rows: int
    denom_positive: int
    denom_negative: int
    target_recall: float
    per_rung: dict[Rung, RungReport]
    marginals: list[MarginalReport]


def run_ablation(inf_session: Session, rows: list[EvalRow]) -> AblationReport:
    """Score the identical rows at each rung and report marginal lift + marginal FPR
    per rung, matched to a shared recall — the S1→S2 step is the addition of family 4
    (E-13). The full detector (S2) at 0.60 sets the shared recall target."""
    scores = {rung: score_rows(inf_session, rows, rung) for rung in RUNGS}
    pairs = {rung: _scored_pairs(rows, scores[rung]) for rung in RUNGS}

    target_recall = confusion_at(rows, scores[Rung.S2], SCORE_THRESHOLD).recall

    per_rung: dict[Rung, RungReport] = {}
    for rung in RUNGS:
        per_rung[rung] = RungReport(
            rung=rung,
            at_threshold=confusion_at(rows, scores[rung], SCORE_THRESHOLD),
            at_matched=match_operating_point(pairs[rung], target_recall),
            pr_auc=pr_auc(rows, scores[rung]),
        )

    marginals: list[MarginalReport] = []
    for lo, hi in ((Rung.S0, Rung.S1), (Rung.S1, Rung.S2)):
        op_lo, op_hi = per_rung[lo].at_matched, per_rung[hi].at_matched
        if op_lo is None or op_hi is None:
            continue
        c_lo, c_hi = op_lo.confusion, op_hi.confusion
        marginals.append(
            MarginalReport(
                from_rung=lo,
                to_rung=hi,
                precision_lift=c_hi.precision - c_lo.precision,
                fpr_cost=c_hi.fpr - c_lo.fpr,
                fps_removed=c_lo.fp - c_hi.fp,
                fns_added=c_hi.fn - c_lo.fn,
            )
        )

    non_amb = [r for r in rows if not r.is_ambiguous]
    return AblationReport(
        rows=len(non_amb),
        denom_positive=sum(1 for r in non_amb if r.is_positive),
        denom_negative=sum(1 for r in non_amb if r.is_negative),
        target_recall=target_recall,
        per_rung=per_rung,
        marginals=marginals,
    )


# --- Arms buildable now: 1 (threshold 0.60) and 5 (oracle) -------------------


@dataclass(frozen=True)
class ArmReport:
    arm: str
    confusion: Confusion
    wrongful_restriction_rate: float
    pr_auc: float
    #: LLM-reviewer output-quality metrics (P3.9). `None` — never `0.0` — on every
    #: arm that does not call a provider: arm 1, arm 3 and arm 5 have no notion of
    #: schema validity, and reporting zero would claim they emitted nothing
    #: parseable rather than that the question does not apply to them.
    #:
    #: `schema_validity` is the fraction of SAMPLES whose FIRST attempt
    #: (`retry_count == 0`) parsed successfully. `repair_retry_rate` is the
    #: fraction of samples that needed R-7's one repair retry. Both exclude
    #: provider-level failures (timeout/refusal/transport) from the denominator:
    #: a call that never produced text says nothing about whether the model can
    #: follow the output schema, and folding those in would let an infrastructure
    #: outage read as a prompt-quality regression. The excluded population is
    #: reported separately as `provider_failure_rate` on `Arm4Coverage` rather
    #: than added here, since it is LLM-specific and this shape is shared.
    schema_validity: float | None = None
    repair_retry_rate: float | None = None


def arm1_threshold(
    rows: list[EvalRow],
    scores_s2: dict[tuple[str, str, str], float],
    threshold: float = SCORE_THRESHOLD,
) -> ArmReport:
    """Arm 1 — Stage 1 (full S2 detector) at the 0.60 threshold (eval §2)."""
    return ArmReport(
        arm="arm1",
        confusion=confusion_at(rows, scores_s2, threshold),
        wrongful_restriction_rate=wrongful_restriction_rate(rows, scores_s2, threshold),
        pr_auc=pr_auc(rows, scores_s2),
    )


def arm5_oracle(rows: list[EvalRow]) -> ArmReport:
    """Arm 5 — the oracle: predict exactly the true label (eval §2). Bounds headroom;
    by construction it is perfect on the non-ambiguous rows."""
    tp = sum(1 for r in rows if r.is_positive)
    tn = sum(1 for r in rows if r.is_negative)
    conf = Confusion(tp=tp, fp=0, tn=tn, fn=0)
    return ArmReport(
        arm="arm5_oracle",
        confusion=conf,
        wrongful_restriction_rate=0.0,
        pr_auc=1.0 if tp else 0.0,
    )


# --- Arm 3: hand-built logistic re-ranker (eval §2, D-5) ---------------------

# The L2 strengths tried; the one with the best tune-split PR-AUC is chosen (Q3:
# train on train, select L2 on tune, report test). 0.0 = unregularized.
RERANKER_L2_GRID: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0, 10.0)

# A learned probability model's natural decision boundary is 0.5 (not the detector's
# 0.60 score threshold). Arm 3's ArmReport is reported at this operating point; the
# paired arm3-vs-arm1 attribution comparison matches operating points separately.
RERANKER_THRESHOLD = 0.5


def feature_vectors(
    inf_session: Session, rows: list[EvalRow]
) -> dict[tuple[str, str, str], tuple[float, ...]]:
    """The per-unit **raw** feature vector (pre-weight measured quantities) in
    FEATURE_ORDER, scored at the full S2 rung — the identical feature vocabulary the
    Stage-1 detector reads (Q2). The re-ranker learns its own weights over these, so
    the frozen `_W_*` weights never enter arm 3."""
    return {
        row.key: tuple(
            c.raw_value
            for c in score_cluster_cycle(
                inf_session, row.unit, list(row.member_ids), Rung.S2
            ).contributions
        )
        for row in rows
    }


def stage1_candidate_actions(
    inf_session: Session, rows: list[EvalRow]
) -> dict[tuple[str, str, str], ProposedAction]:
    """Frozen Stage-1 policy authority for static evaluation, before live suppression.

    High risk alone is insufficient: within-allowance units remain ALLOW. Arm3
    can only retain this original severity or reduce it to ALLOW.
    """
    actions = {}
    for row in rows:
        members = list(row.member_ids)
        score = score_cluster_cycle(inf_session, row.unit, members, Rung.S2)
        claimed = _cycle_claimed(
            inf_session, members, row.unit.window_start,
            row.unit.window_end, row.unit.as_of,
        )
        actions[row.key] = decide(assess(score), score.banding, claimed=claimed).proposed_action
    return actions


@dataclass(frozen=True)
class ReRankerFit:
    model: LogisticReRanker
    chosen_l2: float
    tune_pr_auc: float
    l2_scores: tuple[tuple[float, float], ...]  # (l2, tune_pr_auc) over the grid
    n_train: int
    n_tune: int


def train_reranker(
    rows: list[EvalRow],
    vectors: dict[tuple[str, str, str], tuple[float, ...]],
    *,
    l2_grid: tuple[float, ...] = RERANKER_L2_GRID,
    candidates: set[tuple[str, str, str]] | None = None,
) -> ReRankerFit:
    """Fit the re-ranker on the **train** split and select L2 by **tune**-split
    PR-AUC (Q3). Training is binary abusive-vs-legitimate with ambiguous rows dropped
    from the fit (Q4) — they stay in the reported eval denominators via the row set.
    Ties in tune PR-AUC break toward the *stronger* regularizer (larger L2)."""
    eligible = rows if candidates is None else [r for r in rows if r.key in candidates]
    train = [r for r in eligible if r.split == "train" and not r.is_ambiguous]
    tune = [r for r in eligible if r.split == "tune" and not r.is_ambiguous]
    if not l2_grid or not train or not tune:
        raise ValueError("candidate train/tune splits and L2 grid must be nonempty")
    if len({r.is_positive for r in train}) != 2:
        raise ValueError("candidate training split must contain both binary classes")
    if len({r.is_positive for r in tune}) != 2:
        raise ValueError("candidate tune split must contain both binary classes for L2 selection")
    x = [list(vectors[r.key]) for r in train]
    y = [1.0 if r.is_positive else 0.0 for r in train]

    best: tuple[float, float, LogisticReRanker] | None = None
    l2_scores: list[tuple[float, float]] = []
    for l2 in l2_grid:
        model = LogisticReRanker.fit(x, y, l2=l2)
        tune_scores = {r.key: model.predict_proba(vectors[r.key]) for r in tune}
        auc = pr_auc(tune, tune_scores)
        l2_scores.append((l2, auc))
        if best is None or auc > best[1] or (auc == best[1] and l2 > best[0]):
            best = (l2, auc, model)
    assert best is not None  # l2_grid is non-empty
    return ReRankerFit(
        model=best[2],
        chosen_l2=best[0],
        tune_pr_auc=best[1],
        l2_scores=tuple(l2_scores),
        n_train=len(train),
        n_tune=len(tune),
    )


def reranker_scores(
    rows: list[EvalRow],
    vectors: dict[tuple[str, str, str], tuple[float, ...]],
    model: LogisticReRanker,
) -> dict[tuple[str, str, str], float]:
    """P(abusive) per row under a fitted re-ranker."""
    return {r.key: model.predict_proba(vectors[r.key]) for r in rows}


@dataclass(frozen=True)
class Arm3Report:
    report: ArmReport  # arm 3 on the test split, at RERANKER_THRESHOLD
    fit: ReRankerFit
    arm1_test: ArmReport  # Stage-1 (S2) at 0.60 on the same test rows, for comparison
    vs_arm1_matched: OperatingPoint | None  # baseline matched to arm3 recall (auxiliary)
    n_test: int  # non-ambiguous test rows
    test_scores: dict[tuple[str, str, str], float]  # arm 3 P(abusive) on test rows
    arm1_test_scores: dict[tuple[str, str, str], float]  # policy-eligible S2 test scores
    actions: dict[tuple[str, str, str], ProposedAction]  # severity-capped arm3 recommendations
    stage1_actions: dict[tuple[str, str, str], ProposedAction]


def arm3_reranker(
    inf_session: Session,
    rows: list[EvalRow],
    *,
    l2_grid: tuple[float, ...] = RERANKER_L2_GRID,
    threshold: float = RERANKER_THRESHOLD,
    report_split: str = "test",
    fitted: ReRankerFit | None = None,
) -> Arm3Report:
    """Arm 3 — fit the logistic re-ranker on the same features the detector uses, and
    report it on the held-out **test** split (eval §2). The paired attribution
    comparison against arm 1 (Stage-1 detector) is scored on the identical test rows,
    with Stage 1 swept to arm3's achieved recall (an auxiliary recall proxy, not arm2)."""
    if threshold != RERANKER_THRESHOLD:
        raise ValueError("arm3 policy is fixed at 0.5 before sealed access")
    vectors = feature_vectors(inf_session, rows)
    baseline = score_rows(inf_session, rows, Rung.S2)
    actions = stage1_candidate_actions(inf_session, rows)
    candidates = {key for key, action in actions.items() if action is not ProposedAction.ALLOW}
    baseline = {key: value if key in candidates else 0.0 for key, value in baseline.items()}
    fit = fitted or train_reranker(rows, vectors, l2_grid=l2_grid, candidates=candidates)
    test = [r for r in rows if r.split == report_split]
    # Keep every E-14 row; zero masks noncandidates and cannot cross the fixed .5 gate.
    probabilities = reranker_scores(test, vectors, fit.model)
    test_scores = {r.key: probabilities[r.key] if r.key in candidates else 0.0 for r in test}
    report = ArmReport(
        arm="arm3",
        confusion=confusion_at(test, test_scores, threshold),
        wrongful_restriction_rate=wrongful_restriction_rate(test, test_scores, threshold),
        pr_auc=pr_auc(test, test_scores),
    )
    arm1_test_scores = {r.key: baseline[r.key] for r in test}
    arm1_test = arm1_threshold(test, arm1_test_scores)
    matched = match_operating_point(
        _scored_pairs(test, arm1_test_scores), report.confusion.recall
    )
    return Arm3Report(
        report=report,
        fit=fit,
        arm1_test=arm1_test,
        vs_arm1_matched=matched,
        n_test=sum(1 for r in test if not r.is_ambiguous),
        test_scores=test_scores,
        arm1_test_scores=arm1_test_scores,
        actions={r.key: actions[r.key] if test_scores[r.key] >= threshold else ProposedAction.ALLOW
                 for r in test},
        stage1_actions={r.key: actions[r.key] for r in test},
    )


# --- Arms 4 / 4b: the LLM reviewer (P3.9, eval §2) ---------------------------
#
# Arms 4 and 4b score the Phase 3 reviewer as a *second stage* on top of the
# detector: the unit is unchanged (the fixed E-14 cluster-cycle row), but the
# score is no longer a risk number — it is the binary question "is this unit still
# flagged after the reviewer looked at it?". Everything below is dev-tier (W-1)
# and reads only the inference store; the reviewer never sees ground truth.

#: Binary scores for a reviewed unit. Real numbers so the existing confusion /
#: `match_operating_point` / bootstrap machinery works unchanged at a 0.5 gate.
REVIEWER_STILL_FLAGGED = 1.0
REVIEWER_RECOVERED = 0.0

#: The gate that splits the two values above. Arm 4's "score" takes exactly two
#: values, so any threshold in (0, 1] separates them identically.
REVIEWER_BINARY_THRESHOLD = 0.5


def decision_ids_by_eval_key(
    inf_session: Session, rows: list[EvalRow]
) -> dict[tuple[str, str, str], list[str]]:
    """Map each `EvalRow.key` to the `decision_id`s the detector recorded for that
    fixed unit, via `Decision -> DecisionEvidence -> DetectionEvidence`.

    The join is made on `DetectionEvidence`'s own indexed columns
    (`cluster_snapshot_id`, `cycle_id`, `as_of`) rather than on the JSON payload,
    which carries the same values but cannot be indexed or compared in SQL
    portably.

    `as_of` is normalized with `features._utc` on both sides before comparison,
    the same normalizer `enumerate_cluster_cycles` already applies to cycle
    bounds. This matters: `EvalRow.key` stringifies the caller-supplied `as_of`,
    while `DetectionEvidence.as_of` comes back off SQLite's `DateTime(timezone=
    True)` without reliable tzinfo, so comparing the ISO strings would silently
    match nothing — and "nothing matched" in arm 4 does not error, it just
    reports every row as uncovered and yields an empty, apparently-passing
    confusion matrix. Matching on normalized datetimes instead makes that failure
    mode unreachable.

    One-to-many is expected and correct: one cluster-cycle unit can produce a
    per-account decision for each member. Callers decide how to aggregate; see
    `_arm4_report`."""
    by_evidence: dict[str, list[str]] = {}
    for decision_id, evidence_id in inf_session.execute(
        select(DecisionEvidence.decision_id, DecisionEvidence.evidence_id)
    ).all():
        by_evidence.setdefault(evidence_id, []).append(decision_id)

    result: dict[tuple[str, str, str], list[str]] = {row.key: [] for row in rows}
    wanted = {(row.unit.cluster_snapshot_id, row.unit.cycle_id, _utc(row.unit.as_of)): row.key
              for row in rows}
    for evidence in inf_session.scalars(select(DetectionEvidence)).all():
        key = wanted.get(
            (evidence.cluster_snapshot_id, evidence.cycle_id, _utc(evidence.as_of))
        )
        if key is None:
            continue
        result[key].extend(by_evidence.get(evidence.evidence_id, ()))
    return {key: sorted(set(ids)) for key, ids in result.items()}


@dataclass(frozen=True)
class Arm4Coverage:
    """How much of the row set the reviewer actually spoke to, and how well it
    held to its output contract. Reported alongside — never folded into — the
    confusion matrix, so a coverage collapse can never masquerade as a metric."""

    rows_total: int
    rows_with_decisions: int
    rows_reviewed: int  # rows with at least one arm-scoped LlmReview row
    rows_unreviewed: int  # excluded from the confusion matrix entirely
    samples: int
    provider_failures: int
    provider_failure_rate: float
    first_attempt_parsed: int
    repair_retries: int


@dataclass(frozen=True)
class RawRecommendationSample:
    """One reviewer sample's PRE-CLAMP recommendation (Phase 4 P4.8), paired with
    the decision's proposed action and the row's true label — the shared input
    `escalation_direction` and `reliability_diagram` are both built from.

    Read from `LlmReview.structured_output_json['recommended_action']` — the raw
    `ReviewerVerdict.recommended_action` the model returned — never from
    `LlmReview.recommendation` (already `reviewer_validation.clamp_action`'d down
    to at most `proposed_action`) or `derived_verdict` (the post-guardrail
    effect). `reviewer_v1.md` instructs the model "you may not recommend a
    severity higher than the proposed action"; these two metrics report how
    often the model's own raw output disagrees with that instruction, which the
    clamp silently absorbs everywhere else in this file."""

    proposed_action: ProposedAction
    raw_action: ProposedAction
    confidence: float | None
    label: GroundTruthLabel


def _raw_recommended_action(row: LlmReview) -> ProposedAction | None:
    """The PRE-CLAMP `recommended_action` from one sample's parsed verdict JSON.
    `None` when the sample never parsed (provider failure or terminal parse
    failure after R-7's repair retry) — those are already visible in
    `Arm4Coverage`; this must not paper over them as some default tier."""
    payload = row.structured_output_json
    if not payload:
        return None
    raw = payload.get("recommended_action")
    if raw is None:
        return None
    try:
        return ProposedAction(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class EscalationTierCounts:
    """Raw-vs-proposed severity direction (P4.8), for one `proposed_action`
    tier: how many of the arm's samples proposed at this tier asked for
    something MORE severe than the tier itself (`more_severe`), the same
    severity (`equal`), or something less severe (`less_severe`) — before any
    clamp. Also split by true label, since P4.8's finding is specifically that
    the raw "wants tighter" signal barely separates legit from abusive."""

    proposed_action: ProposedAction
    n: int
    more_severe: int
    equal: int
    less_severe: int
    n_legitimate: int
    more_severe_legitimate: int
    n_abusive: int
    more_severe_abusive: int


def escalation_direction(
    samples: list[RawRecommendationSample],
) -> tuple[EscalationTierCounts, ...]:
    """Bucket `samples` by `proposed_action` tier (ladder order; a tier absent
    from `samples` is omitted, not reported as a zero-n row) and count each
    tier's raw-vs-proposed severity direction via `ACTION_SEVERITY`."""
    by_tier: dict[ProposedAction, list[RawRecommendationSample]] = defaultdict(list)
    for sample in samples:
        by_tier[sample.proposed_action].append(sample)

    result: list[EscalationTierCounts] = []
    for tier in ProposedAction:  # ladder order, least → most severe
        tier_samples = by_tier.get(tier)
        if not tier_samples:
            continue
        proposed_severity = ACTION_SEVERITY[tier]
        more = equal = less = 0
        n_legit = more_legit = n_abuse = more_abuse = 0
        for sample in tier_samples:
            raw_severity = ACTION_SEVERITY[sample.raw_action]
            asked_more = raw_severity > proposed_severity
            if asked_more:
                more += 1
            elif raw_severity == proposed_severity:
                equal += 1
            else:
                less += 1
            if sample.label is GroundTruthLabel.LEGITIMATE:
                n_legit += 1
                more_legit += asked_more
            elif sample.label is GroundTruthLabel.ABUSIVE:
                n_abuse += 1
                more_abuse += asked_more
        result.append(
            EscalationTierCounts(
                proposed_action=tier,
                n=len(tier_samples),
                more_severe=more,
                equal=equal,
                less_severe=less,
                n_legitimate=n_legit,
                more_severe_legitimate=more_legit,
                n_abusive=n_abuse,
                more_severe_abusive=more_abuse,
            )
        )
    return tuple(result)


#: Reliability-diagram bin edges (P4.8). Fixed, not derived from any run's data
#: — this repo's discipline (see `revised_mde`/`discordant_rate`) is to never
#: let a reported bound imply precision the data doesn't have, and a
#: data-derived edge would do exactly that. Finer near 1.0 because P4.8's real-
#: provider run showed raw confidences saturating there.
RELIABILITY_BIN_EDGES: tuple[float, ...] = (0.0, 0.85, 0.90, 0.95, 1.0)


@dataclass(frozen=True)
class ReliabilityBin:
    bin_lo: float
    bin_hi: float
    n: int
    avg_confidence: float
    accuracy: float
    n_legitimate: int
    n_abusive: int


def reliability_diagram(
    samples: list[RawRecommendationSample],
) -> tuple[ReliabilityBin, ...]:
    """Confidence-binned accuracy of "predicted abusive" — raw recommendation
    strictly harsher than `light_friction` — against the true ground-truth
    label (P4.8), restricted to `light_friction`-proposed samples: the only
    tier where a raw downgrade to `allow` would matter for reversal, so it is
    the only tier this framing is meaningful for.

    Ambiguous-labelled samples are excluded (matches this file's existing
    binary-metric convention, e.g. `confusion_at`'s `EvalRow.is_ambiguous`
    filter) and samples with no `confidence` are excluded (nothing to bin).
    A bin with zero samples is OMITTED — never a misleading 0.0/None accuracy
    on no evidence, matching `discordant_rate`/`revised_mde`'s None-on-n=0
    discipline."""
    eligible: list[tuple[RawRecommendationSample, float]] = [
        (s, s.confidence)
        for s in samples
        if s.proposed_action is ProposedAction.LIGHT_FRICTION
        and s.confidence is not None
        and s.label is not GroundTruthLabel.AMBIGUOUS
    ]
    light_friction_severity = ACTION_SEVERITY[ProposedAction.LIGHT_FRICTION]

    bins: list[ReliabilityBin] = []
    edges = RELIABILITY_BIN_EDGES
    last = len(edges) - 2
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        in_bin = [
            (s, conf)
            for s, conf in eligible
            if (lo <= conf < hi) or (i == last and conf == hi)
        ]
        if not in_bin:
            continue
        correct = sum(
            1
            for s, _ in in_bin
            if (ACTION_SEVERITY[s.raw_action] > light_friction_severity)
            == (s.label is GroundTruthLabel.ABUSIVE)
        )
        bins.append(
            ReliabilityBin(
                bin_lo=lo,
                bin_hi=hi,
                n=len(in_bin),
                avg_confidence=sum(conf for _, conf in in_bin) / len(in_bin),
                accuracy=correct / len(in_bin),
                n_legitimate=sum(
                    1 for s, _ in in_bin if s.label is GroundTruthLabel.LEGITIMATE
                ),
                n_abusive=sum(1 for s, _ in in_bin if s.label is GroundTruthLabel.ABUSIVE),
            )
        )
    return tuple(bins)


@dataclass(frozen=True)
class Arm4Report:
    arm: str
    report: ArmReport
    coverage: Arm4Coverage
    scores: dict[tuple[str, str, str], float]  # binary, over the REVIEWED rows only
    reviewed_rows: list[EvalRow]
    decision_ids: dict[tuple[str, str, str], list[str]]
    #: PR-AUC over a 2-valued score is close to meaningless — the curve has one
    #: interior point — so it is reported with this caveat attached rather than
    #: suppressed. Reporting it honestly and labelling it beats silently omitting
    #: a field every other arm carries.
    pr_auc_caveat: str
    #: eval §6.2/§6.3 — the observed discordant proportion over reviewed rows,
    #: and the MDE it implies at this arm's reviewed-row count. `None` when there
    #: are no reviewed rows (discordant_rate) or no observed discordance
    #: (revised_mde) — never a misleading 0.0/point estimate on zero evidence.
    discordant_rate: float | None
    revised_mde: float | None
    #: P4.8 — raw (pre-clamp) recommendation vs. proposed-action severity, and
    #: the confidence-binned reliability diagram built on that same raw signal.
    #: See `RawRecommendationSample`/`escalation_direction`/`reliability_diagram`.
    escalation_direction: tuple[EscalationTierCounts, ...]
    reliability_diagram: tuple[ReliabilityBin, ...]


def _is_provider_failure(row) -> bool:
    """A provider-level failure (timeout / refusal / transport), as opposed to a
    response that arrived but did not parse. `reviewer._run_sample` records these
    two failure kinds under distinct `validation_errors` keys, which is the only
    place the distinction is persisted."""
    errors = row.validation_errors or {}
    return "provider_error" in errors


def _sample_quality(sample_rows: list) -> tuple[int, int, int]:
    """`(provider_failures, first_attempt_parsed, repair_retries)` over the rows of
    ONE sample. A sample persists one row per attempt (`retry_count` 0, then 1)."""
    if any(_is_provider_failure(row) for row in sample_rows):
        return (1, 0, 0)
    first = next((r for r in sample_rows if r.retry_count == 0), None)
    parsed_first = 1 if first is not None and first.structured_output_json is not None else 0
    retried = 1 if any(r.retry_count >= 1 for r in sample_rows) else 0
    return (0, parsed_first, retried)


def _arm4_report(inf_session: Session, rows: list[EvalRow], arm: str) -> Arm4Report:
    """Score one reviewer arm over `rows` by reading the persisted `LlmReview`
    rows the reviewer already wrote — this function never calls a provider.

    Per row: resolve its decision_id(s), read each one's final per-sample verdicts
    under `arm`, and derive a binary outcome.

    **`escalate` counts as still-flagged.** Only an actual `reverse` recovers a
    unit. An escalation is a request for a human, not a decision to restore the
    account — the decision stays live under `under_review` with a 7-day window —
    so scoring it as a recovery would credit the reviewer with an outcome it
    explicitly declined to reach, and would let the fail-closed paths (provider
    timeout, unparseable output, the R-6 ceiling, the cluster blast-radius guard)
    read as successful interventions.

    **"an actual reverse" means the decision really reversed**, so a decision
    counts as recovered only when BOTH the arm's unanimous per-sample consensus is
    `reverse` AND the decision's persisted status is `REVERSED`. The consensus
    alone is not enough: `review_decision` demotes a unanimous reverse to
    `escalate` — leaving the decision live — whenever the R-6 severity ceiling or
    the cluster-blast-radius guard fires, and those demotions are invisible in the
    per-sample `derived_verdict`s, which still read `"reverse"`. Scoring off the
    verdicts alone would credit the reviewer with restorations that never
    happened, and would make arm 4 look better exactly where its safety guards
    did their job. The status alone is not enough either — it cannot distinguish
    this arm's reversal from an analyst's — so both are required.

    **A row recovers only if EVERY one of its decisions reversed.** A unit whose
    members are partly restored and partly still actioned is still an actioned
    unit; the conservative reading matches the caution
    `_other_actioned_cluster_members` already applies cluster-wide.

    **Rows with no reviewer coverage are EXCLUDED, not scored.** A gated decision
    (already under review, suppressed, terminal) or a row with no decision at all
    produced no verdict; scoring it 0 or 1 would invent one. They are counted in
    `Arm4Coverage` instead."""
    from app.services.reviewer import latest_reviews_for_arm

    decision_ids = decision_ids_by_eval_key(inf_session, rows)

    scores: dict[tuple[str, str, str], float] = {}
    reviewed_rows: list[EvalRow] = []
    rows_with_decisions = 0
    samples = provider_failures = first_parsed = retries = 0
    # P4.8: every sample's raw (pre-clamp) recommendation, paired with its
    # decision's proposed action and the row's true label — the shared input to
    # `escalation_direction`/`reliability_diagram`.
    raw_samples: list[RawRecommendationSample] = []

    for row in rows:
        ids = decision_ids[row.key]
        if ids:
            rows_with_decisions += 1
        outcomes: list[bool] = []
        for decision_id in ids:
            final_rows = latest_reviews_for_arm(inf_session, decision_id, arm)
            if not final_rows:
                continue
            decision = inf_session.get(Decision, decision_id)
            # Quality accounting walks every attempt of every sample, not just the
            # final row per sample, so a repair retry is visible as such.
            all_rows = _all_attempt_rows(inf_session, final_rows)
            for sample_rows in all_rows:
                samples += 1
                failed, parsed, retried = _sample_quality(sample_rows)
                provider_failures += failed
                first_parsed += parsed
                retries += retried
            if decision is not None:
                for final_row in final_rows:
                    raw_action = _raw_recommended_action(final_row)
                    if raw_action is not None:
                        raw_samples.append(
                            RawRecommendationSample(
                                proposed_action=decision.proposed_action,
                                raw_action=raw_action,
                                confidence=final_row.confidence,
                                label=row.label,
                            )
                        )
            effects = tuple(r.derived_verdict or "escalate" for r in final_rows)
            consensus = combine_effects(effects)  # type: ignore[arg-type]
            applied = decision is not None and decision.status is DecisionStatus.REVERSED
            outcomes.append(consensus == "reverse" and applied)
        if not outcomes:
            continue
        reviewed_rows.append(row)
        recovered = all(outcomes)
        scores[row.key] = REVIEWER_RECOVERED if recovered else REVIEWER_STILL_FLAGGED

    scored_samples = samples - provider_failures
    report = ArmReport(
        arm=arm,
        confusion=confusion_at(reviewed_rows, scores, REVIEWER_BINARY_THRESHOLD),
        wrongful_restriction_rate=wrongful_restriction_rate(
            reviewed_rows, scores, REVIEWER_BINARY_THRESHOLD
        ),
        pr_auc=pr_auc(reviewed_rows, scores),
        schema_validity=(first_parsed / scored_samples) if scored_samples else None,
        repair_retry_rate=(retries / scored_samples) if scored_samples else None,
    )
    coverage = Arm4Coverage(
        rows_total=len(rows),
        rows_with_decisions=rows_with_decisions,
        rows_reviewed=len(reviewed_rows),
        rows_unreviewed=len(rows) - len(reviewed_rows),
        samples=samples,
        provider_failures=provider_failures,
        provider_failure_rate=(provider_failures / samples) if samples else 0.0,
        first_attempt_parsed=first_parsed,
        repair_retries=retries,
    )
    return Arm4Report(
        arm=arm,
        report=report,
        coverage=coverage,
        scores=scores,
        reviewed_rows=reviewed_rows,
        decision_ids=decision_ids,
        pr_auc_caveat=(
            "near-meaningless for arm 4/4b: the score takes exactly two values, so "
            "the precision-recall curve has a single interior point and the area is "
            "not comparable to a continuously-scored arm's PR-AUC"
        ),
        discordant_rate=(p_d := discordant_rate(scores, reviewed_rows)),
        revised_mde=revised_mde(p_d, len(reviewed_rows)) if p_d is not None else None,
        escalation_direction=escalation_direction(raw_samples),
        reliability_diagram=reliability_diagram(raw_samples),
    )


def _all_attempt_rows(inf_session: Session, final_rows: list) -> list[list]:
    """Regroup one arm-scoped run's FINAL-per-sample rows back into every attempt
    of each sample, so `_sample_quality` can see the repair retries the
    final-row projection hides."""
    if not final_rows:
        return []
    decision_id = final_rows[0].decision_id
    idempotency_key = final_rows[0].idempotency_key
    arm = final_rows[0].arm
    rows = inf_session.scalars(
        select(LlmReview)
        .where(
            LlmReview.decision_id == decision_id,
            LlmReview.idempotency_key == idempotency_key,
            LlmReview.arm == arm,
        )
        .order_by(LlmReview.sample_index, LlmReview.retry_count)
    ).all()
    by_sample: dict[int, list] = {}
    for row in rows:
        by_sample.setdefault(row.sample_index, []).append(row)
    return [by_sample[i] for i in sorted(by_sample)]


def arm4_reviewer(inf_session: Session, rows: list[EvalRow]) -> Arm4Report:
    """Arm 4 — the LLM reviewer on the FULL evidentiary view (eval §2)."""
    return _arm4_report(inf_session, rows, "arm4_reviewer")


def arm4b_timeline_only(inf_session: Session, rows: list[EvalRow]) -> Arm4Report:
    """Arm 4b — the same reviewer on the contributions-masked `timeline_only` view.

    The attribution control for arm 4: identical evidence, identical policy, the
    only difference being whether the model was shown (and allowed to cite) the
    detector's own per-feature contributions."""
    return _arm4_report(inf_session, rows, "arm4b_timeline_only")


# --- Arm 2: the detector swept to the reviewer's operating point --------------


@dataclass(frozen=True)
class Arm2Report:
    """Arm 2 (eval §2) — Stage 1 alone, swept to arm 4's achieved recall.

    This is the *real* arm 2, unlike `Arm3Report.vs_arm1_matched`, whose own
    docstring disclaims it as "an auxiliary recall proxy, not arm2". The question
    it answers is the one that decides whether the reviewer earns its cost: at the
    recall the reviewer actually achieved, how many false positives would simply
    moving the detector's threshold have removed?"""

    arm4_arm: str
    arm4_report: ArmReport
    baseline: ArmReport  # Stage-1 (S2) at 0.60 on the identical reviewed rows
    matched: OperatingPoint | None  # Stage 1 swept to arm 4's recall
    baseline_scores: dict[tuple[str, str, str], float]
    n_rows: int
    fps_removed: int
    fns_added: int
    exchange_rate: float | None  # FPs removed per FN added at the matched OP
    exchange_rate_status: str


def arm2_matched(
    inf_session: Session, arm4: Arm4Report, *, threshold: float = SCORE_THRESHOLD
) -> Arm2Report:
    """Sweep the Stage-1 detector to arm 4's achieved recall over exactly the rows
    arm 4 actually reviewed (the unreviewed rows are excluded from both sides, so
    the comparison stays paired).

    `exchange_rate = fps_removed / fns_added`. When arm 4 recovered nothing, the
    matched operating point is the baseline's own and `fns_added == 0`, so the
    ratio is undefined: `None` with `exchange_rate_status="no_recovery"`, mirroring
    `match_operating_point`'s own `None`-return convention rather than emitting an
    infinity or a silent zero."""
    rows = arm4.reviewed_rows
    baseline_scores = score_rows(inf_session, rows, Rung.S2)
    baseline = arm1_threshold(rows, baseline_scores, threshold)
    matched = match_operating_point(
        _scored_pairs(rows, baseline_scores), arm4.report.confusion.recall
    )

    fps_removed = fns_added = 0
    exchange_rate: float | None = None
    if matched is None:
        status = "no_positives"
    else:
        fps_removed = matched.confusion.fp - arm4.report.confusion.fp
        fns_added = arm4.report.confusion.fn - matched.confusion.fn
        if fns_added == 0:
            status = "no_recovery"
        else:
            exchange_rate = fps_removed / fns_added
            status = "ok"

    return Arm2Report(
        arm4_arm=arm4.arm,
        arm4_report=arm4.report,
        baseline=baseline,
        matched=matched,
        baseline_scores=baseline_scores,
        n_rows=sum(1 for r in rows if not r.is_ambiguous),
        fps_removed=fps_removed,
        fns_added=fns_added,
        exchange_rate=exchange_rate,
        exchange_rate_status=status,
    )


# --- Fairness slices ---------------------------------------------------------


def slice_by(
    rows: list[EvalRow],
    scores: dict[tuple[str, str, str], float],
    dimension: str,
    threshold: float = SCORE_THRESHOLD,
) -> dict[str, Confusion]:
    """Confusion per value of a slice dimension (scenario_family / domain_type /
    size_band / age_band), each with `large_shared` naturally isolated by size."""
    buckets: dict[str, list[EvalRow]] = {}
    for r in rows:
        if dimension == "scenario_family":
            k = r.scenario_family
        elif dimension == "domain_type":
            k = r.domain_type
        elif dimension == "size_band":
            k = _size_band(r.cluster_size)
        elif dimension == "age_band":
            k = r.age_band
        else:
            raise ValueError(f"unknown slice dimension {dimension!r}")
        buckets.setdefault(k, []).append(r)
    return {k: confusion_at(rs, scores, threshold) for k, rs in buckets.items()}


# --- Paired statistics (eval §6.2) -------------------------------------------


@dataclass(frozen=True)
class McNemar:
    discordant_b: int  # correct under A, wrong under B
    discordant_c: int  # wrong under A, correct under B
    statistic: float


def mcnemar_paired(
    rows: list[EvalRow],
    scores_a: dict[tuple[str, str, str], float],
    scores_b: dict[tuple[str, str, str], float],
    threshold: float = SCORE_THRESHOLD,
    threshold_b: float | None = None,
) -> McNemar:
    """McNemar on the discordant pairs of two arms/rungs over the identical rows
    (eval §6.2). Continuity-corrected statistic. `threshold_b` defaults to `threshold`
    (a shared operating point on a shared score scale); pass it explicitly to compare
    two arms **each at its own operating point** — required when the arms' scores live
    on different scales (e.g. the detector's risk score vs arm 3's probability)."""
    tb = threshold if threshold_b is None else threshold_b
    b = c = 0
    for r in rows:
        if r.is_ambiguous:
            continue
        a_correct = (scores_a[r.key] >= threshold) == r.is_positive
        b_correct = (scores_b[r.key] >= tb) == r.is_positive
        if a_correct and not b_correct:
            b += 1
        elif b_correct and not a_correct:
            c += 1
    denom = b + c
    stat = ((abs(b - c) - 1) ** 2) / denom if denom else 0.0
    return McNemar(discordant_b=b, discordant_c=c, statistic=stat)


def paired_bootstrap_ci(
    rows: list[EvalRow],
    scores_a: dict[tuple[str, str, str], float],
    scores_b: dict[tuple[str, str, str], float],
    *,
    metric: str = "recall",
    threshold: float = SCORE_THRESHOLD,
    threshold_b: float | None = None,
    iterations: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% CI for a fixed-threshold paired difference (B − A), resampling groups with
    replacement (eval §6.2). Deterministic given `seed`. `threshold_b` defaults to
    `threshold`; pass it to score each arm at its own operating point (arms on
    different score scales, e.g. detector risk vs arm 3 probability)."""
    tb = threshold if threshold_b is None else threshold_b
    non_amb = [r for r in rows if not r.is_ambiguous]
    if not non_amb:
        return (0.0, 0.0)
    rng = random.Random(seed)
    groups: dict[str, list[EvalRow]] = {}
    for row in non_amb:
        groups.setdefault(row.split_group_id, []).append(row)
    group_ids = sorted(groups)

    def metric_of(
        sample: list[EvalRow], scores: dict[tuple[str, str, str], float], thr: float
    ) -> float:
        c = _confusion_from_pairs([(scores[r.key], r.is_positive) for r in sample], thr)
        return getattr(c, metric)

    diffs: list[float] = []
    for _ in range(iterations):
        sample = [
            row for _ in group_ids
            for row in groups[group_ids[rng.randrange(len(group_ids))]]
        ]
        diffs.append(metric_of(sample, scores_b, tb) - metric_of(sample, scores_a, threshold))
    diffs.sort()
    lo = diffs[int(0.025 * iterations)]
    hi = diffs[int(0.975 * iterations)]
    return (lo, hi)


#: MDE z-constants (eval §6.3), literal per the doc — not derived, since this
#: repo has no scipy/numpy dependency and the doc itself states them as fixed:
#: alpha=0.05 two-sided (z_{alpha/2}=1.96), power=0.80 (z_beta=0.84).
_MDE_Z_ALPHA_2 = 1.96
_MDE_Z_BETA = 0.84


def discordant_rate(
    scores: dict[tuple[str, str, str], float], reviewed_rows: list[EvalRow]
) -> float | None:
    """The observed discordant proportion `p_d` (eval §6.2/§6.3) over arm 4's
    actually-reviewed rows: every row arm 4 reviews was already flagged by Stage
    1 (that is what made it gatable), so "discordant" here means arm 4 changed
    that outcome — i.e. the row recovered (`REVIEWER_RECOVERED`). A row arm 4
    left still-flagged agrees with the baseline and is not discordant.

    Takes the raw `scores`/`reviewed_rows` an `Arm4Report` is built from (not the
    report itself, which carries this value as a field computed from these same
    inputs, to avoid a construction-order cycle).

    Returns `None` when there are no reviewed rows (§6.3's own MDE formula is
    undefined at n=0) rather than a misleading 0.0."""
    n = len(reviewed_rows)
    if n == 0:
        return None
    recovered = sum(1 for v in scores.values() if v == REVIEWER_RECOVERED)
    return recovered / n


def revised_mde(p_d_observed: float, n: int) -> float | None:
    """eval §6.3's minimum detectable effect, re-derived from an OBSERVED `p_d`
    rather than the doc's a-priori 0.25 assumption:

        δ_MDE = (z_{alpha/2} + z_beta) * sqrt(p_d / n)

    Returns `None` when `n == 0` or `p_d_observed <= 0` (no discordance observed
    — §6.3's own guidance: this reads as "the reviewer rarely/never acts", a
    bound on reach, not an MDE to report as if discordance existed)."""
    if n <= 0 or p_d_observed <= 0:
        return None
    return (_MDE_Z_ALPHA_2 + _MDE_Z_BETA) * math.sqrt(p_d_observed / n)


def operating_point_record(op: OperatingPoint | None) -> dict:
    """JSON-safe record of the achieved recall proxy, including discrete overshoot."""
    if op is None:
        return {"status": "no_positives", "metric": "cluster_cycle_recall_proxy"}
    delta = op.confusion.recall - op.target_recall
    return {
        "metric": "cluster_cycle_recall_proxy",
        "target_recall": op.target_recall,
        "achieved_recall": op.confusion.recall,
        "recall_delta": delta,
        "threshold": op.threshold,
        "tie_policy": "equal scores move together; highest threshold meeting target",
        "status": "unreachable" if not op.reachable else ("exact" if delta == 0 else "overshoot"),
        "reachable": op.reachable,
        "fpr": op.confusion.fpr,
        "precision": op.confusion.precision,
        "ties_at_threshold": op.tied_at_threshold,
        "ties_encountered": op.tied_at_threshold > 1,
    }


def matched_group_bootstrap_ci(
    rows: list[EvalRow],
    baseline_scores: dict[tuple[str, str, str], float],
    reviewer_scores: dict[tuple[str, str, str], float],
    *,
    reviewer_threshold: float = RERANKER_THRESHOLD,
    iterations: int = 1000,
    seed: int = 0,
    seed_by_group: dict[str, str] | None = None,
    estimand: str = DEFAULT_BOOTSTRAP_ESTIMAND,
    conditions_on: str = DEFAULT_BOOTSTRAP_CONDITIONS_ON,
) -> dict:
    """Paired group bootstrap with baseline rematching inside each replicate.

    All E-14 cycles remain in the metric denominator. Independent split groups are
    sampled within each world seed (the single-world case has one block). The CI
    conditions on the trained model and fixed reviewer policy. A replicate missing
    either class has no defined comparison and is counted, never assigned zero.

    `estimand`/`conditions_on` label WHAT the interval is an interval for. They
    default to arm 3's wording so `scripts/finalize_evaluation.py`'s call site is
    unchanged; `scripts/arm4_run.py` passes arm-2 wording and a binary-appropriate
    `reviewer_threshold`. They are pure provenance strings — nothing branches on
    them — but an artifact whose estimand string says "arm3" while the numbers are
    arm 2's is worse than no string at all.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    groups: dict[str, list[EvalRow]] = {}
    for row in rows:
        if not row.is_ambiguous:
            groups.setdefault(row.split_group_id, []).append(row)
    # Resample independent groups, stratified by their non-ambiguous class. This
    # preserves class support in every replicate while still resampling all group
    # units within each world seed. A split group must have one binary label.
    strata: dict[tuple[str, GroundTruthLabel], list[str]] = {}
    for group, group_rows in groups.items():
        labels = {row.label for row in group_rows}
        if len(labels) != 1:
            raise EvaluationInvariantError(
                f"split group {group} contains inconsistent non-ambiguous labels"
            )
        label = next(iter(labels))
        block = "single_world" if seed_by_group is None else seed_by_group[group]
        strata.setdefault((block, label), []).append(group)
    has_positive = any(label is POSITIVE_LABEL for _, label in strata)
    has_negative = any(label is GroundTruthLabel.LEGITIMATE for _, label in strata)
    if not has_positive or not has_negative:
        return _insufficient_bootstrap_report(
            iterations, len(groups), len({block for block, _ in strata}),
            "insufficient_classes", estimand=estimand, conditions_on=conditions_on,
        )
    rng = random.Random(seed)
    differences: list[float] = []
    deltas: list[float] = []
    for _ in range(iterations):
        sample = []
        for block, label in sorted(strata, key=lambda item: (item[0], item[1].value)):
            ids = strata[(block, label)]
            for _ in ids:
                sample.extend(groups[ids[rng.randrange(len(ids))]])
        reviewer = confusion_at(sample, reviewer_scores, reviewer_threshold)
        matched = match_operating_point(_scored_pairs(sample, baseline_scores), reviewer.recall)
        if matched is None:
            # No reachable matched operating point for this replicate (e.g. the
            # resampled baseline scores have no positives) — counted as a
            # degenerate replicate below, never assigned zero (review C-3).
            continue
        differences.append(reviewer.fpr - matched.confusion.fpr)
        deltas.append(matched.confusion.recall - reviewer.recall)
    differences.sort()
    deltas.sort()

    def interval(values: list[float]) -> list[float] | None:
        if not values:
            return None
        n = len(values)
        return [values[min(n - 1, int(.025 * n))], values[min(n - 1, int(.975 * n))]]

    valid_replicates = len(differences)
    required_replicates = max(
        MIN_BOOTSTRAP_VALID_REPLICATES,
        math.ceil(iterations * MIN_BOOTSTRAP_VALID_FRACTION),
    )
    supported = valid_replicates >= required_replicates
    return {
        "estimand": estimand,
        "matching_metric": "cluster_cycle_recall_proxy_not_credit_recovery",
        "method": "paired split-group bootstrap within seed; baseline rematched per replicate",
        "conditions_on": conditions_on,
        "iterations": iterations,
        "valid_replicates": valid_replicates,
        "valid_replicate_fraction": valid_replicates / iterations,
        "minimum_valid_replicates": required_replicates,
        "minimum_valid_fraction": MIN_BOOTSTRAP_VALID_FRACTION,
        "degenerate_replicates": iterations - valid_replicates,
        "groups": len(groups), "seed_blocks": len({block for block, _ in strata}),
        "fpr_difference_95_ci": interval(differences) if supported else None,
        "recall_overshoot_95_ci": interval(deltas) if supported else None,
        "status": "ok" if supported else "insufficient_replicates",
    }


def _insufficient_bootstrap_report(
    iterations: int,
    groups: int,
    seed_blocks: int,
    status: str,
    *,
    estimand: str = DEFAULT_BOOTSTRAP_ESTIMAND,
    conditions_on: str = DEFAULT_BOOTSTRAP_CONDITIONS_ON,
) -> dict:
    """Return a deliberately interval-free, fail-closed bootstrap result."""
    return {
        "estimand": estimand,
        "matching_metric": "cluster_cycle_recall_proxy_not_credit_recovery",
        "method": "paired split-group bootstrap within seed; baseline rematched per replicate",
        "conditions_on": conditions_on,
        "iterations": iterations,
        "valid_replicates": 0,
        "valid_replicate_fraction": 0.0,
        "minimum_valid_replicates": max(
            MIN_BOOTSTRAP_VALID_REPLICATES,
            math.ceil(iterations * MIN_BOOTSTRAP_VALID_FRACTION),
        ),
        "minimum_valid_fraction": MIN_BOOTSTRAP_VALID_FRACTION,
        "degenerate_replicates": iterations,
        "groups": groups,
        "seed_blocks": seed_blocks,
        "fpr_difference_95_ci": None,
        "recall_overshoot_95_ci": None,
        "status": status,
    }

"""Arms 4 / 4b / 2 — the LLM reviewer inside the evaluation harness (P3.9),
plus the W-1 sealed-tier isolation of `scripts/arm4_run.py` and the
`config/costs.yaml` `llm:` block.

The whole arm-4 path is a join followed by an aggregation, and both halves fail
*silently* if they are wrong: a bad `as_of` comparison resolves zero decision_ids,
a missing arm filter mixes two arms' verdicts, and either way the result is a
well-formed, apparently-passing `ArmReport` over an empty or contaminated row set.
Every test below therefore asserts a non-degenerate precondition (rows joined,
decisions reviewed, both arms distinct) before asserting the metric.

All numbers here are dev-tier and NOT reported (W-1); only invariants are asserted.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest
import yaml
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.engine import create_inference_schema
from app.db.ground_truth_writer import make_ground_truth_sink
from app.db.sqlite_fk import enable_sqlite_foreign_keys
from app.models.enums import DecisionStatus, ProposedAction
from app.models.inference import Decision, LlmReview
from app.schemas.reviewer import ReviewerVerdict
from app.services import evaluation as evaluation_module
from app.services.detector import run_detector
from app.services.evaluation import (
    REVIEWER_BINARY_THRESHOLD,
    REVIEWER_RECOVERED,
    REVIEWER_STILL_FLAGGED,
    ArmReport,
    arm1_threshold,
    arm2_matched,
    arm4_reviewer,
    arm4b_timeline_only,
    arm5_oracle,
    build_eval_rows,
    decision_ids_by_eval_key,
    load_ground_truth,
    matched_group_bootstrap_ci,
    score_rows,
)
from app.services.fake_reviewer import (
    FakeMode,
    FakeReviewerProvider,
    _extract_evidence_keys,
    _structured_payload_text,
)
from app.services.features import Rung
from app.services.generator import WORLD_END, generate_world, persist_world
from app.services.llm_provider import LLMResponse, TokenUsage
from app.services.reviewer import latest_reviews_for_arm, review_cluster_batch, review_decision


def _fresh_inf_engine() -> Engine:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True
    )
    engine = enable_sqlite_foreign_keys(engine)
    create_inference_schema(engine)
    return engine


class _FixedActionProvider:
    """A provider that unconditionally recommends one action, citing the case's
    first real evidence key so C-2 accepts it (zero valid citations would force
    `escalate` regardless of the action).

    Needed because `FakeReviewerProvider`'s rule-table bucket is seeded on the
    structured payload, which embeds the decision_id — and `db/decisions.py` mints
    decision ids as `uuid4().hex`. The fake's verdict for a detector-produced
    decision is therefore effectively RANDOM per run: fine for "produce a mix of
    outcomes", useless for any test that needs a specific one, and an outright
    flake source for any test whose setup depends on the previous arm's state
    transitions. (Same reason `test_reviewer.py` hand-pins its decision ids.)

    - `ALLOW` derives `reverse` (strictly less severe than any gatable proposal).
    - `TEMPORARY_HOLD` derives `uphold`: C-4 clamps it down to the decision's own
      proposed action, which is then not *less* severe, so nothing reverses — and
      an uphold makes NO state change, leaving the decision reviewable by a
      second arm.
    """

    def __init__(self, action: ProposedAction) -> None:
        self.action = action

    def complete(self, request):
        text = _structured_payload_text(request)
        keys = _extract_evidence_keys(text)
        assert keys, "the case must expose at least one citable evidence key"
        verdict = ReviewerVerdict(
            evidence_supporting_action=(keys[0],),
            benign_explanations=(),
            missing_evidence=(),
            unverified_claims=(),
            rationale=f"test: unconditional {self.action.value}",
            recommended_action=self.action,
            confidence=0.9,
            escalate_to_human=False,
        )
        return LLMResponse(
            text=verdict.model_dump_json(),
            model_snapshot="fixed-action-test-provider",
            latency_ms=0.0,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


def _always_reverse() -> _FixedActionProvider:
    return _FixedActionProvider(ProposedAction.ALLOW)


def _always_uphold() -> _FixedActionProvider:
    return _FixedActionProvider(ProposedAction.TEMPORARY_HOLD)


def _gatable_decisions(inf: Session) -> list[Decision]:
    """Exactly the population `scripts/arm4_run.py` iterates: live, non-ALLOW."""
    return list(
        inf.scalars(
            select(Decision).where(
                Decision.proposed_action != ProposedAction.ALLOW,
                Decision.status.in_((DecisionStatus.PROPOSED, DecisionStatus.APPLIED)),
            )
        ).all()
    )


def _run_arm(
    inf: Session, arm: str, *, mode: FakeMode = FakeMode.NORMAL, provider=None
) -> int:
    """Review every gatable decision under `arm`. One provider instance across all
    samples of a case (required by `FakeReviewerProvider`'s per-instance
    `_call_counts`). Returns the number of decisions actually reviewed."""
    provider = provider if provider is not None else FakeReviewerProvider(mode=mode)
    reviewed = 0
    for decision in _gatable_decisions(inf):
        outcome = review_decision(
            inf, decision, provider=provider, run_id=f"{arm}-run",
            idempotency_key=f"{arm}:{decision.decision_id}", as_of=WORLD_END, arm=arm,
        )
        if not outcome.gated:
            reviewed += 1
    inf.flush()
    return reviewed


@pytest.fixture
def world():
    """A seeded min world with decisions recorded — the same shape
    `scripts/arm4_run.py` builds, minus the two-DB isolation."""
    inf_engine = _fresh_inf_engine()
    gt_engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True
    )
    from app.db.ground_truth_store import create_ground_truth_schema

    create_ground_truth_schema(gt_engine)
    with Session(inf_engine, future=True) as inf, Session(gt_engine, future=True) as gt:
        persist_world(generate_world(1, "min"), inf, make_ground_truth_sink(gt))
        inf.flush()
        gt.flush()
        result = run_detector(inf, as_of=WORLD_END, run_id="arm4-fixture")
        assert result.decisions_created > 0
        rows = build_eval_rows(inf, load_ground_truth(gt), WORLD_END)
        assert rows
        yield inf, rows


# --- 6b: the EvalRow <-> decision_id join ------------------------------------


def test_join_resolves_a_non_zero_number_of_decisions(world) -> None:
    """The silent-failure guard. `EvalRow.key` stringifies a tz-aware `as_of`
    while `DetectionEvidence.as_of` round-trips through SQLite without reliable
    tzinfo, so an ISO-string comparison would match nothing — and matching
    nothing does not raise, it just reports every row uncovered and yields an
    empty, apparently-passing confusion matrix."""
    inf, rows = world
    mapping = decision_ids_by_eval_key(inf, rows)

    assert set(mapping) == {r.key for r in rows}
    resolved = [key for key, ids in mapping.items() if ids]
    assert resolved, "the join resolved zero decisions — arm 4 would silently score nothing"

    all_ids = [i for ids in mapping.values() for i in ids]
    assert len(all_ids) == len(set(all_ids)), "a decision must map to at most one unit"
    persisted = {d.decision_id for d in inf.scalars(select(Decision)).all()}
    assert set(all_ids) <= persisted


def test_join_is_one_to_many_and_deduplicated(world) -> None:
    """One cluster-cycle unit can produce a per-account decision for each member,
    so the mapping is one-to-many by design; ids are sorted and unique."""
    inf, rows = world
    mapping = decision_ids_by_eval_key(inf, rows)
    for ids in mapping.values():
        assert ids == sorted(set(ids))
    assert max((len(ids) for ids in mapping.values()), default=0) >= 1


# --- 6c: arm-scoped reads ------------------------------------------------------


def test_arms_do_not_cross_contaminate(world) -> None:
    """Both arms run against the SAME decisions. `latest_reviews` (no arm filter)
    would return whichever ran last for both; `latest_reviews_for_arm` must keep
    them disjoint.

    Driven by the always-uphold provider so the setup is deterministic: an uphold
    is the one outcome that makes NO state change, so every decision stays live
    and reviewable by the second arm. With `FakeReviewerProvider` the first arm's
    verdicts are uuid4-dependent, and a run where it happened to escalate
    everything would leave the second arm nothing to review."""
    inf, rows = world
    assert _run_arm(inf, "arm4_reviewer", provider=_always_uphold()) > 0
    assert _run_arm(inf, "arm4b_timeline_only", provider=_always_uphold()) > 0

    decision_ids = [
        d for ids in decision_ids_by_eval_key(inf, rows).values() for d in ids
    ]
    checked = 0
    for decision_id in decision_ids:
        a = latest_reviews_for_arm(inf, decision_id, "arm4_reviewer")
        b = latest_reviews_for_arm(inf, decision_id, "arm4b_timeline_only")
        if not a or not b:
            continue
        checked += 1
        assert {r.arm for r in a} == {"arm4_reviewer"}
        assert {r.arm for r in b} == {"arm4b_timeline_only"}
        assert not {r.review_id for r in a} & {r.review_id for r in b}
        assert {r.prompt_version for r in a} == {"reviewer_v1"}
        assert {r.prompt_version for r in b} == {"reviewer_timeline_only_v1"}
    assert checked > 0, "fixture must review at least one decision under BOTH arms"


def test_unknown_arm_reads_nothing(world) -> None:
    inf, rows = world
    _run_arm(inf, "arm4_reviewer", provider=_always_uphold())
    # The join returns EVERY decision for a unit, including the ALLOW/gated ones
    # the reviewer never touched — so pick one that was actually reviewed, or the
    # "unknown arm reads nothing" assertion would pass vacuously.
    decision_id = next(
        d
        for ids in decision_ids_by_eval_key(inf, rows).values()
        for d in ids
        if latest_reviews_for_arm(inf, d, "arm4_reviewer")
    )
    assert latest_reviews_for_arm(inf, decision_id, "arm4_reviewer")
    assert latest_reviews_for_arm(inf, decision_id, "never-ran") == []


# --- 6d: arm 4 / arm 4b end to end -------------------------------------------


@pytest.mark.parametrize(
    "arm,fn", [("arm4_reviewer", arm4_reviewer), ("arm4b_timeline_only", arm4b_timeline_only)]
)
def test_arm4_reads_persisted_reviews_end_to_end(world, arm, fn) -> None:
    inf, rows = world
    assert _run_arm(inf, arm, provider=_always_uphold()) > 0

    report = fn(inf, rows)
    assert report.arm == arm
    assert report.coverage.rows_reviewed > 0, "nothing was scored — the join or read failed"
    assert report.coverage.rows_reviewed + report.coverage.rows_unreviewed == len(rows)
    assert set(report.scores) == {r.key for r in report.reviewed_rows}
    assert set(report.scores.values()) <= {REVIEWER_RECOVERED, REVIEWER_STILL_FLAGGED}

    c = report.report.confusion
    assert c.tp + c.fp + c.tn + c.fn == sum(
        1 for r in report.reviewed_rows if not r.is_ambiguous
    )
    assert report.report.schema_validity == 1.0  # NORMAL mode always parses first try
    assert report.report.repair_retry_rate == 0.0
    assert report.coverage.provider_failure_rate == 0.0
    assert "two values" in report.pr_auc_caveat


def test_unanimous_reverse_that_did_not_apply_is_still_flagged(world) -> None:
    """The aggregation guard a verdict-only reading would miss — and, on this
    world, the ONLY thing that happens.

    `review_decision` demotes a unanimous reverse to `escalate` whenever the R-6
    severity ceiling or the cluster-blast-radius guard fires, leaving the decision
    LIVE — but the per-sample `derived_verdict`s still read `"reverse"`. Scoring
    off the verdicts alone credits the reviewer with restorations that never
    happened, and does so *precisely where its safety guards did their job*. A row
    may only count as recovered when its decisions actually reached `REVERSED`.

    On the generated worlds this is not a corner case: NO decision the detector
    produces can auto-reverse at all (see
    `test_no_generated_decision_can_auto_reverse` below), so every unanimous
    reverse here is a demotion."""
    inf, rows = world
    _run_arm(inf, "arm4_reviewer", provider=_always_reverse())
    report = arm4_reviewer(inf, rows)

    demoted = 0
    for row in report.reviewed_rows:
        for decision_id in report.decision_ids[row.key]:
            final = latest_reviews_for_arm(inf, decision_id, "arm4_reviewer")
            if not final or {r.derived_verdict for r in final} != {"reverse"}:
                continue
            decision = inf.get(Decision, decision_id)
            assert decision is not None
            if decision.status is not DecisionStatus.REVERSED:
                demoted += 1
                assert report.scores[row.key] == REVIEWER_STILL_FLAGGED
    assert demoted > 0, (
        "fixture must contain at least one unanimous-reverse-but-not-applied "
        "decision for this test to be non-vacuous"
    )


@pytest.mark.parametrize("preset", ["min", "dev"])
def test_no_generated_decision_can_auto_reverse(preset: str) -> None:
    """A structural property of the generated worlds, pinned so the resulting
    all-zero arm-4 recovery rate reads as a finding rather than a broken harness.

    Neither world contains a single decision the reviewer could auto-apply:
      * `min` produces no `light_friction` decision at all — every gatable
        decision sits above L-3's R-6 severity ceiling;
      * `dev` produces `light_friction` decisions, but every one of them has
        another currently-actioned decision sharing its exact evidence signature,
        so the RC-1 P1-1 cluster-blast-radius guard blocks all of them.

    Consequence: arm 4 and arm 4b recover nothing on a dev-tier world regardless
    of what the model says, so `arm2_matched` reports `no_recovery` — the
    degenerate branch, reached legitimately rather than by a bug. If this test
    ever starts failing, the worlds have changed and arm 4's numbers become
    interesting for the first time."""
    inf_engine = _fresh_inf_engine()
    gt_engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True
    )
    from app.db.ground_truth_store import create_ground_truth_schema
    from app.services.reviewer import _other_actioned_cluster_members

    create_ground_truth_schema(gt_engine)
    with Session(inf_engine, future=True) as inf, Session(gt_engine, future=True) as gt:
        persist_world(generate_world(1, preset), inf, make_ground_truth_sink(gt))
        inf.flush()
        gt.flush()
        run_detector(inf, as_of=WORLD_END, run_id="ceiling-probe")
        gatable = _gatable_decisions(inf)
        assert gatable, "fixture must produce gatable decisions"
        auto_appliable = [
            d
            for d in gatable
            if d.proposed_action is ProposedAction.LIGHT_FRICTION
            and not _other_actioned_cluster_members(inf, d)
        ]
        assert auto_appliable == []


def test_dev_arm4_preset_can_auto_reverse_via_review_cluster_batch() -> None:
    """P4.2 (`docs/decisions.md §13` M-2/M-11): the counterpart to
    `test_no_generated_decision_can_auto_reverse` above — proves the `dev_arm4`
    preset's `reversible_singleton_*` fixture is not just substrate but an
    actually-reachable auto-reversal, once decisions are reviewed via
    `review_cluster_batch` (M-11's narrowed guard) rather than `review_decision`
    looped per-decision (which keeps the old unconditional block and would leave
    this fixture's 2-member clusters permanently escalating, never reversing).

    No `now_utc` patch and no precedent-application step is needed: this
    fixture's cycle 2 clears the 0.60 gate on its own (F-DOM-1+F-VER-1+F-CYC-1+
    F-CYC-3), so a single `run_detector` call is enough — M-2's 2-call/
    precedent-application mechanism only matters for a fixture that additionally
    exercises F-CYC-4, which this one does not."""
    inf_engine = _fresh_inf_engine()
    gt_engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True
    )
    from app.db.ground_truth_store import create_ground_truth_schema

    create_ground_truth_schema(gt_engine)
    with Session(inf_engine, future=True) as inf, Session(gt_engine, future=True) as gt:
        persist_world(generate_world(1, "dev_arm4"), inf, make_ground_truth_sink(gt))
        inf.flush()
        gt.flush()
        run_detector(inf, as_of=WORLD_END, run_id="p42-positive")
        gatable = _gatable_decisions(inf)
        assert gatable, "dev_arm4 must produce gatable decisions"

        # Not `FakeReviewerProvider()`: its verdict is seeded on the uuid4 decision id,
        # so whether any decision reversed varied per run (6 or 8 locally, 0 in a
        # 2026-09-27 `make test`). An unconditional ALLOW proposes a reversal on every
        # decision, so only the fixture and M-11's guard decide what gets applied.
        provider = _always_reverse()
        outcomes = review_cluster_batch(
            inf, gatable, provider=provider, run_id="p42-positive-review",
            idempotency_key_fn=lambda d: f"p42:{d.decision_id}", as_of=WORLD_END,
            arm="arm4_reviewer",
        )
        inf.flush()

        assert any(o.applied for o in outcomes), (
            "review_cluster_batch on the dev_arm4 fixture must produce at least "
            "one real auto-applied reversal — if this fails, either the fixture "
            "no longer reaches 0.60 or M-11's narrowed guard regressed"
        )
        reversed_decisions = [
            d
            for d in gatable
            if inf.get(Decision, d.decision_id).status is DecisionStatus.REVERSED
        ]
        assert reversed_decisions, "expected at least one REVERSED decision in the DB"


def test_running_both_arms_on_one_db_biases_the_second_arm(world) -> None:
    """Why `scripts/arm4_run.py` seeds TWO independent dev worlds rather than
    running both arms over one.

    The reviewer's outcomes are real state transitions, not annotations: an
    `escalate` moves the decision to `under_review` and opens an
    `EscalationRecord`, so L-9's gate skips it (`escalated_under_review`) when the
    second arm runs afterwards. The second arm therefore sees a strictly smaller,
    non-random subset and the two arms stop being comparable. (A reversal would
    bias it the same way via terminal status plus suppression; on these worlds
    escalation is the reachable mechanism — see above.) Pinned here as a known
    property of the shared-DB path rather than left to be rediscovered as a wrong
    number in an artifact."""
    inf, rows = world
    _run_arm(inf, "arm4_reviewer", provider=_always_reverse())
    first = arm4_reviewer(inf, rows)
    _run_arm(inf, "arm4b_timeline_only", provider=_always_reverse())
    second = arm4b_timeline_only(inf, rows)

    assert first.coverage.rows_reviewed > 0
    assert second.coverage.rows_reviewed < first.coverage.rows_reviewed, (
        "expected the first arm's state transitions to gate rows out of the second"
    )


def test_unreviewed_rows_are_excluded_not_scored_zero(world) -> None:
    """A gated decision produced no verdict; scoring it 0 or 1 would invent one."""
    inf, rows = world
    _run_arm(inf, "arm4_reviewer", provider=_always_uphold())
    report = arm4_reviewer(inf, rows)
    unreviewed = [r for r in rows if r not in report.reviewed_rows]
    assert all(r.key not in report.scores for r in unreviewed)
    assert report.coverage.rows_unreviewed == len(unreviewed)


def test_escalate_counts_as_still_flagged(world) -> None:
    """The resolved design question, asserted against a real escalating run: an
    escalation is a request for a human, not a restoration — the decision stays
    live under `under_review`. `FakeMode.TIMEOUT` makes every sample fail closed to
    `escalate`, so NOTHING may be scored as recovered."""
    inf, rows = world
    assert _run_arm(inf, "arm4_reviewer", mode=FakeMode.TIMEOUT) > 0
    report = arm4_reviewer(inf, rows)

    assert report.coverage.rows_reviewed > 0
    assert set(report.scores.values()) == {REVIEWER_STILL_FLAGGED}
    verdicts = {
        r.derived_verdict
        for r in inf.scalars(select(LlmReview).where(LlmReview.arm == "arm4_reviewer")).all()
    }
    assert verdicts == {"escalate"}, "fixture must actually escalate"


def test_provider_errors_are_excluded_from_schema_validity(world) -> None:
    """Resolved design question: a call that never produced text says nothing about
    whether the model can follow the output schema. With EVERY sample a provider
    failure the denominator is empty, so both quality metrics are `None` rather
    than a 0.0 that would read as "the model emitted nothing parseable"."""
    inf, rows = world
    assert _run_arm(inf, "arm4_reviewer", mode=FakeMode.ERROR) > 0
    report = arm4_reviewer(inf, rows)

    assert report.coverage.samples > 0
    assert report.coverage.provider_failures == report.coverage.samples
    assert report.coverage.provider_failure_rate == 1.0
    assert report.report.schema_validity is None
    assert report.report.repair_retry_rate is None


def test_repair_retry_rate_counts_the_r7_retry(world) -> None:
    """`FakeMode.MALFORMED` never parses, so every sample burns R-7's one repair
    retry: first-attempt validity 0.0, retry rate 1.0 — and, critically, these are
    NOT provider failures, so the denominator is the full sample count."""
    inf, rows = world
    assert _run_arm(inf, "arm4_reviewer", mode=FakeMode.MALFORMED) > 0
    report = arm4_reviewer(inf, rows)

    assert report.coverage.provider_failures == 0
    assert report.report.schema_validity == 0.0
    assert report.report.repair_retry_rate == 1.0


# --- 6a: the new ArmReport fields stay None on the non-LLM arms ---------------


def test_non_llm_arms_leave_quality_fields_none(world) -> None:
    """`None`, never `0.0`: arm 1/3/5 never call a provider, and claiming zero
    schema validity for them would assert something false rather than absent."""
    inf, rows = world
    scores = score_rows(inf, rows, Rung.S2)
    for report in (arm1_threshold(rows, scores), arm5_oracle(rows)):
        assert report.schema_validity is None
        assert report.repair_retry_rate is None
    assert ArmReport("x", arm5_oracle(rows).confusion, 0.0, 0.0).schema_validity is None


# --- 6e: arm 2 ----------------------------------------------------------------


def test_arm2_matched_materializes_on_the_dev_world(world) -> None:
    inf, rows = world
    _run_arm(inf, "arm4_reviewer", provider=_always_uphold())
    arm4 = arm4_reviewer(inf, rows)
    arm2 = arm2_matched(inf, arm4)

    assert arm2.arm4_arm == "arm4_reviewer"
    assert arm2.n_rows > 0
    # Both sides are scored on exactly the rows arm 4 reviewed, so the comparison
    # is paired rather than a denominator change.
    assert set(arm2.baseline_scores) >= {r.key for r in arm4.reviewed_rows}
    assert arm2.exchange_rate_status in {"ok", "no_recovery", "no_positives"}
    if arm2.exchange_rate_status == "ok":
        assert arm2.fns_added != 0
        assert arm2.exchange_rate == arm2.fps_removed / arm2.fns_added
    else:
        assert arm2.exchange_rate is None


def test_arm2_degenerate_no_recovery_case(world) -> None:
    """The divide-by-zero guard, driven by a real run rather than a hand-built
    report: with every verdict an escalation arm 4 recovers nothing, the matched
    operating point is the baseline's own, and the exchange rate is undefined."""
    inf, rows = world
    _run_arm(inf, "arm4_reviewer", mode=FakeMode.TIMEOUT)
    arm4 = arm4_reviewer(inf, rows)
    assert set(arm4.scores.values()) == {REVIEWER_STILL_FLAGGED}

    arm2 = arm2_matched(inf, arm4)
    assert arm2.exchange_rate is None
    assert arm2.exchange_rate_status in {"no_recovery", "no_positives"}


# --- 6f: bootstrap parameterization -------------------------------------------


def test_bootstrap_estimand_labels_are_overridable(world) -> None:
    """Pure provenance strings — nothing branches on them — but an artifact whose
    estimand says "arm3" while the numbers are arm 2's is worse than no string."""
    inf, rows = world
    _run_arm(inf, "arm4_reviewer", provider=_always_uphold())
    arm4 = arm4_reviewer(inf, rows)
    arm2 = arm2_matched(inf, arm4)
    scores = {**{r.key: REVIEWER_STILL_FLAGGED for r in arm4.reviewed_rows}, **arm4.scores}

    default = matched_group_bootstrap_ci(
        arm4.reviewed_rows, arm2.baseline_scores, scores,
        reviewer_threshold=REVIEWER_BINARY_THRESHOLD, iterations=20,
    )
    assert default["estimand"].startswith("arm3")

    labelled = matched_group_bootstrap_ci(
        arm4.reviewed_rows, arm2.baseline_scores, scores,
        reviewer_threshold=REVIEWER_BINARY_THRESHOLD, iterations=20,
        estimand="arm4 minus matched Stage-1 wrongful-restriction rate",
        conditions_on="frozen reviewer manifest and fixed binary reviewer policy",
    )
    assert labelled["estimand"].startswith("arm4")
    assert "reviewer manifest" in labelled["conditions_on"]
    # Only the labels changed: the statistical content is identical.
    assert {k: v for k, v in labelled.items() if k not in ("estimand", "conditions_on")} == {
        k: v for k, v in default.items() if k not in ("estimand", "conditions_on")
    }


# --- W-1: the arm-4 driver cannot reach sealed Tier 2 -------------------------

_SEALED_MODULES = ("evaluation_access", "world_provenance")
_SEALED_CALLS = ("begin_finalization", "complete_finalization", "verify_world_pair")


def _module_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            names.update(f"{node.module or ''}.{a.name}" for a in node.names)
    return names


def test_arm4_run_never_reaches_sealed_tier2() -> None:
    """W-1, enforced by AST rather than by docstring. `scripts/arm4_run.py` seeds
    and scores DEV worlds only: it must never import the sealed-access modules nor
    name the three functions that consume a world's single sealed touch.

    Asserted structurally because the failure is irreversible — `begin_finalization`
    commits the touch before any scoring happens, so a wrong import discovered at
    runtime has already burned the world."""
    source = (pathlib.Path(__file__).resolve().parents[3] / "scripts" / "arm4_run.py")
    assert source.is_file()
    text = source.read_text()
    tree = ast.parse(text)

    imports = _module_imports(tree)
    for sealed in _SEALED_MODULES:
        assert not any(sealed in name for name in imports), (
            f"scripts/arm4_run.py must not import app.db.{sealed} (W-1)"
        )

    called = {
        node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    for name in _SEALED_CALLS:
        assert name not in called, f"scripts/arm4_run.py must not call {name}() (W-1)"

    # `arm3_run.py` legitimately calls `verify_world_pair`, so a grep for the name
    # anywhere in scripts/ would be a false alarm — this test is deliberately
    # scoped to the one new file.
    assert "verify_world_pair" in (
        source.parent / "arm3_run.py"
    ).read_text(), "control: the sibling script really does use the sealed path"


def test_new_evaluation_arm_code_adds_no_sealed_imports() -> None:
    """The companion half: arms 4/4b/2 live in `evaluation.py`, which already
    imports `evaluation_access` for `load_ground_truth`'s authorization (that
    predates P3.9 and is the module's whole purpose). What must NOT have appeared
    is a `world_provenance` import or any finalization call."""
    source = pathlib.Path(inspect.getfile(evaluation_module))
    tree = ast.parse(source.read_text())
    imports = _module_imports(tree)
    assert not any("world_provenance" in name for name in imports)
    called = {
        node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not (called & set(_SEALED_CALLS))


# --- config/costs.yaml's llm: block -------------------------------------------


def test_costs_yaml_llm_block_is_parseable_and_non_zero() -> None:
    """The `llm:` block's zeros were placeholders that would have silently priced
    the reviewer at nothing the moment anything started reading them. Nothing does
    yet — asserted below — but the values must be real."""
    path = pathlib.Path(__file__).resolve().parents[3] / "config" / "costs.yaml"
    costs = yaml.safe_load(path.read_text())

    assert set(costs["llm"]) == {
        "cost_per_1k_input_tokens_usd",
        "cost_per_1k_output_tokens_usd",
    }
    for key, value in costs["llm"].items():
        assert isinstance(value, float), key
        assert value > 0.0, key
    # The pre-existing blocks are untouched: P3.9's carve-out is the `llm:` key only.
    assert costs["credits"]["free_credit_value_usd"] == 0.50
    assert costs["review"]["human_review_minutes"] == 8

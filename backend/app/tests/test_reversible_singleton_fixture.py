"""Phase 4 P4.1 (docs/decisions.md §13 M-1) — the reversible pair fixture.

Confirms the generator's `reversible_singleton_legit`/`_abusive` families
produce the exact deterministic feature signals the fixture was engineered
for, over `canonical_bytes` (byte-reproducibility, same convention as
`test_generator.py`), independent of any detector run. F-CYC-4 (which needs a
real, previously `APPLIED` decision — not raw world data) is P4.2's job, not
tested here.

An important finding surfaced while building this fixture, recorded in
`docs/decisions.md §13` addendum: the co-member needed to make F-CYC-3 fire
(it must share the SAME cluster as the primary account — an unclustered
co-member's SIGNUP is invisible to `ctx.member_ids`-scoped scoring) makes
`light_friction` cluster-wide (`detector.py`: `targets = set(members)`), which
gives the co-member its OWN live decision sharing the identical
`evidence_signature` — exactly what RC-1 P1-1's auto-reversal guard blocks.
So this fixture, as currently built, reaches cycle 2 with a real,
non-degenerate `light_friction` decision and real feature firing, but full
auto-reversal is not proven reachable by this test file — that is P4.2/P4.8's
open question, not resolved here.
"""

from __future__ import annotations

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.db.ground_truth_writer import make_ground_truth_sink
from app.models.enums import EmailDomainType, EventType, GroundTruthLabel, VerificationLevel
from app.models.inference import Decision
from app.services.detector import run_detector
from app.services.evidence import decision_evidence
from app.services.generator import (
    FAMILY_SPECS,
    WORLD_END,
    canonical_bytes,
    generate_world,
    persist_world,
)
from app.services.review_case import build_review_case

SEED = 1
FAMILIES = ("reversible_singleton_legit", "reversible_singleton_abusive")


def test_families_are_registered_with_the_right_labels() -> None:
    assert FAMILY_SPECS["reversible_singleton_legit"].label == GroundTruthLabel.LEGITIMATE
    assert FAMILY_SPECS["reversible_singleton_abusive"].label == GroundTruthLabel.ABUSIVE
    for family in FAMILIES:
        spec = FAMILY_SPECS[family]
        assert spec.size == (2, 2)
        assert spec.domain == EmailDomainType.DISPOSABLE
        assert spec.verification == VerificationLevel.NONE


def test_dev_preset_includes_both_families_deterministically() -> None:
    world_a = generate_world(SEED, "dev_arm4")
    world_b = generate_world(SEED, "dev_arm4")
    assert canonical_bytes(world_a) == canonical_bytes(world_b)

    for family in FAMILIES:
        labels = [x for x in world_a.labels if x.scenario_family == family]
        assert len(labels) == 20, f"{family}: expected the configured PRESETS count"


def test_fixture_declares_one_two_member_component_per_cluster() -> None:
    world = generate_world(SEED, "dev_arm4")
    for fx in world.fixtures:
        if fx.scenario_family not in FAMILIES:
            continue
        assert len(fx.expected_components) == 1
        assert len(fx.expected_components[0]) == 2


def test_primary_account_has_two_cycles_with_designed_excess() -> None:
    """Cycle 1 claims far past light_friction (excess 1.4); cycle 2 claims land
    just under the 0.25 light_friction ceiling (excess 0.24) — verified
    directly off the generated events/cycles, no detector run needed."""
    world = generate_world(SEED, "dev_arm4")
    fx = next(f for f in world.fixtures if f.scenario_family == "reversible_singleton_legit")
    primary_id, partner_id = sorted(fx.expected_components[0])
    # primary is whichever account actually has cycles; the co-member has none.
    primary_cycles = [c for c in world.cycles if c.account_id == primary_id]
    partner_cycles = [c for c in world.cycles if c.account_id == partner_id]
    if not primary_cycles:
        primary_id, partner_id = partner_id, primary_id
        primary_cycles = [c for c in world.cycles if c.account_id == primary_id]
        partner_cycles = [c for c in world.cycles if c.account_id == partner_id]

    assert len(primary_cycles) == 2
    assert partner_cycles == [], "co-member must never carry a cycle of its own"

    primary_claims = sorted(
        (e.timestamp, e.credits_amount)
        for e in world.events
        if e.account_id == primary_id and e.event_type == EventType.CLAIM_CREDITS
    )
    assert [amt for _, amt in primary_claims] == [60, 31]

    partner_events = [e for e in world.events if e.account_id == partner_id]
    assert len(partner_events) == 1
    assert partner_events[0].event_type == EventType.SIGNUP


def test_partner_signup_lands_within_72h_of_primary_exhaustion() -> None:
    world = generate_world(SEED, "dev_arm4")
    fx = next(f for f in world.fixtures if f.scenario_family == "reversible_singleton_legit")
    a, b = sorted(fx.expected_components[0])
    cycles_a = [c for c in world.cycles if c.account_id == a]
    primary_id, partner_id = (a, b) if cycles_a else (b, a)

    exhaustion = next(
        e.timestamp
        for e in sorted(world.events, key=lambda e: e.timestamp)
        if e.account_id == primary_id and e.event_type == EventType.CONSUME_CREDITS
    )
    signup = next(
        e.timestamp for e in world.events
        if e.account_id == partner_id and e.event_type == EventType.SIGNUP
    )
    delta_hours = (signup - exhaustion).total_seconds() / 3600
    assert 0 <= delta_hours <= 72


def test_cycle2_fires_dom1_ver1_cyc1_cyc3_and_clears_score_threshold(
    inference_engine: Engine, gt_engine: Engine,
) -> None:
    """End-to-end: without any F-CYC-4 precedent (single detector run), cycle
    2's decision already clears the 0.60 score gate as light_friction —
    proving the fixture's substrate is self-sufficient for a real, actionable
    decision. F-CYC-4 (a second, real precedent) is P4.2's addition, expected
    to push the total higher still, not needed to reach 0.60 at all here."""
    world = generate_world(SEED, "dev_arm4")
    with (
        Session(inference_engine, future=True) as inf,
        Session(gt_engine, future=True) as gt,
    ):
        persist_world(world, inf, make_ground_truth_sink(gt))
        inf.commit()
        gt.commit()

    with Session(inference_engine, future=True) as inf:
        run_detector(inf, as_of=WORLD_END, run_id="p4-1-fixture-check")
        inf.commit()

        fx = next(
            f for f in world.fixtures if f.scenario_family == "reversible_singleton_legit"
        )
        members = fx.expected_components[0]
        decisions = list(
            inf.scalars(select(Decision).where(Decision.account_id.in_(members)))
        )
        assert decisions, "cycle 2 must produce at least one decision"

        cycle2_decision = next(
            d for d in decisions
            if decision_evidence(inf, d.decision_id)["window_start"].startswith("2026-05-02")
        )
        ev = decision_evidence(inf, cycle2_decision.decision_id)
        contributions = {c["feature_id"]: c["contribution"] for c in ev["contributions"]}

        assert contributions["F-DOM-1"] > 0
        assert contributions["F-VER-1"] > 0
        assert 0 < contributions["F-CYC-1"] < 0.30 * 0.25 + 1e-9
        assert contributions["F-CYC-3"] > 0
        assert cycle2_decision.proposed_action.value == "light_friction"
        assert sum(contributions.values()) >= 0.60


def test_legit_and_abusive_primary_are_distinguishable_by_a_citable_raw_fact() -> None:
    """M-12 (docs/decisions.md §13): the two variants must not be evidence-
    identical, or reversal precision would be pinned at the class ratio by
    construction and no reviewer (real or fake) could ever legitimately tell
    them apart. Confirms the intended distinguishing fact — the legit
    primary's account age and login history — is real and present in the
    persisted world, WITHOUT moving any feature contribution (F-AGE-1/F-DOM-2
    both stay at zero for both variants, since firing F-AGE-1 would subtract
    0.15 and pull cycle 2 below the 0.60 action gate entirely, defeating the
    fixture's whole point — see the module-level comment in `generator.py`)."""
    world = generate_world(SEED, "dev_arm4")

    def primary_of(family: str) -> tuple[str, object]:
        fx = next(f for f in world.fixtures if f.scenario_family == family)
        a, b = fx.expected_components[0]
        cycles_a = [c for c in world.cycles if c.account_id == a]
        primary_id = a if cycles_a else b
        account = next(x for x in world.accounts if x.account_id == primary_id)
        return primary_id, account

    legit_id, legit_account = primary_of("reversible_singleton_legit")
    abusive_id, abusive_account = primary_of("reversible_singleton_abusive")

    # Raw fact #1: account age differs by design (~45 days earlier for legit).
    assert legit_account.created_at < abusive_account.created_at

    # Raw fact #2: the legit primary has real LOGIN history; the abusive
    # primary has none — a citable difference in the persisted timeline.
    legit_logins = [
        e for e in world.events if e.account_id == legit_id and e.event_type == EventType.LOGIN
    ]
    abusive_logins = [
        e for e in world.events if e.account_id == abusive_id and e.event_type == EventType.LOGIN
    ]
    assert len(legit_logins) >= 2
    assert abusive_logins == []


def test_neither_distinguishing_fact_moves_a_feature_contribution(
    inference_engine: Engine, gt_engine: Engine,
) -> None:
    """Companion to the above: confirms the citable raw facts (age, logins)
    deliberately do NOT cross F-AGE-1/F-DOM-2's own thresholds — cycle 2's
    total score must stay identical to the abusive variant's (M-1's
    requirement), or the fixture stops clearing the 0.60 action gate at all."""
    world = generate_world(SEED, "dev_arm4")
    with (
        Session(inference_engine, future=True) as inf,
        Session(gt_engine, future=True) as gt,
    ):
        persist_world(world, inf, make_ground_truth_sink(gt))
        inf.commit()
        gt.commit()

    with Session(inference_engine, future=True) as inf:
        run_detector(inf, as_of=WORLD_END, run_id="p4-1-m12-check")
        inf.commit()
        fx = next(
            f for f in world.fixtures if f.scenario_family == "reversible_singleton_legit"
        )
        members = fx.expected_components[0]
        decisions = list(
            inf.scalars(select(Decision).where(Decision.account_id.in_(members)))
        )
        cycle2_decision = next(
            d for d in decisions
            if decision_evidence(inf, d.decision_id)["window_start"].startswith("2026-05-02")
        )
        ev = decision_evidence(inf, cycle2_decision.decision_id)
        contributions = {c["feature_id"]: c["contribution"] for c in ev["contributions"]}
        assert contributions.get("F-AGE-1", 0.0) == 0.0
        assert contributions.get("F-DOM-2", 0.0) == 0.0
        assert cycle2_decision.proposed_action.value == "light_friction"


def _cycle2_decision(inf: Session, world, family: str) -> Decision:
    fx = next(f for f in world.fixtures if f.scenario_family == family)
    members = fx.expected_components[0]
    decisions = list(inf.scalars(select(Decision).where(Decision.account_id.in_(members))))
    return next(
        d for d in decisions
        if decision_evidence(inf, d.decision_id)["window_start"].startswith("2026-05-02")
    )


def test_reviewer_case_surfaces_the_legit_primarys_pre_window_history(
    inference_engine: Engine, gt_engine: Engine,
) -> None:
    """P4.8 regression: `build_review_case` reads only the immutable
    `DetectionEvidence.payload_json` (`review_case.py`'s module docstring), and
    that payload's `timeline` used to be bounded to `[window_start, window_end)`
    of the CURRENT cycle only (`detector.py`), which silently excluded the
    legit primary's -75d SIGNUP and -65d/-55d LOGINs relative to cycle 2's
    `window_start` — the exact raw fact M-12 engineered this fixture to expose.
    Confirms the reviewer-context lookback (`detector.py::_TIMELINE_LOOKBACK_DAYS`)
    actually gets that history into the rendered `ReviewCase`, not just the
    persisted evidence payload."""
    world = generate_world(SEED, "dev_arm4")
    with (
        Session(inference_engine, future=True) as inf,
        Session(gt_engine, future=True) as gt,
    ):
        persist_world(world, inf, make_ground_truth_sink(gt))
        inf.commit()
        gt.commit()

    with Session(inference_engine, future=True) as inf:
        run_detector(inf, as_of=WORLD_END, run_id="p4-8-reviewer-history-check")
        inf.commit()

        decision = _cycle2_decision(inf, world, "reversible_singleton_legit")
        case = build_review_case(inf, decision, view="full", as_of=WORLD_END)

        event_types = {bucket.event_type for bucket in case.timeline}
        assert "signup" in event_types
        assert "login" in event_types, (
            "the primary's pre-existing LOGIN history must reach the reviewer's "
            "timeline, not just the current cycle's claim/consume events"
        )


def test_legit_and_abusive_reviewer_timelines_are_genuinely_different(
    inference_engine: Engine, gt_engine: Engine,
) -> None:
    """Companion to the above and the actual end-to-end verification of M-12's
    promise (docs/decisions.md §13): the two variants must produce genuinely
    different rendered `ReviewCase.timeline`s, not just different raw generated
    events that never made it into what the reviewer sees. Before the P4.8 fix,
    both variants' rendered timelines held only the 2 current-cycle events and
    were indistinguishable."""
    world = generate_world(SEED, "dev_arm4")
    with (
        Session(inference_engine, future=True) as inf,
        Session(gt_engine, future=True) as gt,
    ):
        persist_world(world, inf, make_ground_truth_sink(gt))
        inf.commit()
        gt.commit()

    with Session(inference_engine, future=True) as inf:
        run_detector(inf, as_of=WORLD_END, run_id="p4-8-distinguishability-check")
        inf.commit()

        legit = build_review_case(
            inf, _cycle2_decision(inf, world, "reversible_singleton_legit"),
            view="full", as_of=WORLD_END,
        )
        abusive = build_review_case(
            inf, _cycle2_decision(inf, world, "reversible_singleton_abusive"),
            view="full", as_of=WORLD_END,
        )

        assert legit.timeline != abusive.timeline
        legit_types = {bucket.event_type for bucket in legit.timeline}
        abusive_types = {bucket.event_type for bucket in abusive.timeline}
        assert "login" in legit_types
        assert "login" not in abusive_types

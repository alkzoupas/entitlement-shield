"""P3.5: `reviewer.py` orchestration (docs/phase-3-plan.md P3.5, decisions.md §12
L-3/L-4/L-9, C-5).

Decisions and their immutable evidence are built by hand (mirroring
`test_human_review.py`'s pattern) rather than through `run_detector`, so each test
can state the exact scenario it exercises — a single-member vs. multi-member
cluster, a `light_friction` vs. `hold_incremental_credits` proposed action — without
depending on what a generated world happens to produce.

Two fixed decision ids (`"dec-2"`, `"dec-0"`) are used where `FakeMode.NORMAL`'s or
`FakeMode.DISAGREE`'s rule-table bucket must land in a specific spot for the test to
be meaningful (not merely likely) — see the comment above their definitions below
for how they were chosen; this keeps the tests deterministic rather than
probabilistic.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.clock import now_utc
from app.core.decision_status import LEGAL_TRANSITIONS
from app.db.decisions import transition_decision
from app.db.engine import create_inference_schema
from app.db.sqlite_fk import enable_sqlite_foreign_keys
from app.models.claims import SyntheticClaim
from app.models.enums import (
    ActorType,
    DecisionStatus,
    EmailDomainType,
    ProposedAction,
    VerificationLevel,
)
from app.models.inference import (
    Account,
    AuditEvent,
    ClusterSnapshot,
    Decision,
    EntitlementCycle,
    EscalationRecord,
    LlmReview,
    SuppressionRecord,
)
from app.schemas.reviewer import EvidenceKeyKind
from app.services import reviewer_manifest
from app.services.evidence import associate_evidence, decision_evidence, persist_evidence
from app.services.fake_reviewer import FakeMode, FakeReviewerProvider
from app.services.policy_engine import evidence_signature, should_flag
from app.services.prompt_builder import StructuredEvidenceSegment, build_prompt
from app.services.review_case import build_review_case
from app.services.reviewer import (
    _other_actioned_cluster_members,
    review_cluster_batch,
    review_decision,
)

AS_OF = now_utc()

# `_stable_seed(structured_text) % 3` for a single-member, excess=0.5,
# qualified_stream_count=2 case (the shape every fixture below uses unless noted)
# is 0 for decision_id "dec-2" and 2 for decision_id "dec-0" (computed once via
# `fake_reviewer._stable_seed` against the exact payload shape below). Under
# `FakeMode.NORMAL` (bucket = seed % 3, same for all 3 identical calls) that makes
# "dec-2" bucket ALLOW and "dec-0" bucket HOLD_INCREMENTAL_CREDITS. Under
# `FakeMode.DISAGREE` (variants 0/1 share bucket = seed % 3, variant 2 uses
# (seed + 1) % 3), "dec-0" (bucket 2 -> 0) with a `light_friction` proposal yields
# effects (uphold, uphold, reverse) — a genuine 2-of-3 disagreement, not a
# coincidental unanimity.
UNANIMOUS_REVERSE_DECISION_ID = "dec-2"
DISAGREEMENT_DECISION_ID = "dec-0"

#: Canonical 6-field cluster edges — `[account_a, account_b, weight,
#: evidence_count, first_seen, last_seen]`, the shape `clustering` persists and
#: `policy_engine.evidence_signature` reads as its v2 form. The RC-1 P1-1 tests
#: below need REAL edges: with `edges == []` the v2 and the legacy-3-field edge
#: projections both collapse to an empty list, so a signature computed from
#: `ReviewCase.edges` and one computed from the raw payload agree by accident and
#: the guard's blocking branch is never actually exercised.
CANONICAL_EDGES: list[list[object]] = [
    [
        "acct-1",
        "acct-2",
        0.75,
        3,
        (AS_OF - timedelta(days=21)).isoformat(),
        (AS_OF - timedelta(days=2)).isoformat(),
    ],
]

# Chosen the same way as the two ids above, but against the edges-bearing payload
# shape (`CANONICAL_EDGES` is rendered into the prompt's `edges:` block, so it
# shifts `fake_reviewer._stable_seed` and therefore the rule-table bucket): this
# is the id whose bucket is ALLOW for the two-member/`CANONICAL_EDGES` fixture,
# giving a unanimous `reverse` under `FakeMode.NORMAL`.
UNANIMOUS_REVERSE_WITH_EDGES_DECISION_ID = "dec-6"


def _session() -> Session:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True
    )
    engine = enable_sqlite_foreign_keys(engine)
    create_inference_schema(engine)
    return Session(engine, future=True)


def _structured_text(prompt) -> str:
    """The one segment of an assembled prompt that carries per-case evidence."""
    return next(s.text for s in prompt.segments if isinstance(s, StructuredEvidenceSegment))


class _CountingProvider:
    """Wraps a provider to assert exactly how many times it was actually called."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        return self.inner.complete(request)


def _account(session: Session, account_id: str) -> None:
    session.add(
        Account(
            account_id=account_id,
            created_at=AS_OF - timedelta(days=120),
            email_domain_type=EmailDomainType.CONSUMER,
            verification_level=VerificationLevel.EMAIL,
        )
    )
    session.flush()


def _cluster_snapshot(session: Session, snapshot_id: str, member_account_ids: list[str]) -> None:
    session.add(
        ClusterSnapshot(
            snapshot_id=snapshot_id,
            as_of=AS_OF,
            member_account_ids=member_account_ids,
            edges=[],
            evidence_summary={},
            content_hash=snapshot_id,
            created_at=AS_OF,
        )
    )
    session.flush()


def _decision_with_evidence(
    session: Session,
    *,
    decision_id: str,
    account_id: str,
    cluster_snapshot_id: str,
    member_account_ids: list[str],
    excess: float,
    proposed_action: ProposedAction,
    idempotency_key: str | None = None,
    qualified_stream_count: int = 2,
    edges: list[list[object]] | None = None,
    risk_score: float = 0.9,
    contributions: list[dict] | None = None,
) -> Decision:
    """A `Decision` with a full, review-case-compatible `DetectionEvidence` payload
    (the shape `detector.run_detector` produces — see `detector._detection`/
    `run_detector`'s payload assembly), so `build_review_case` works exactly as it
    would on a real detector-produced decision.

    `edges` takes the canonical 6-field cluster edge shape
    (`[account_a, account_b, weight, evidence_count, first_seen, last_seen]`, as
    `clustering._content_hash` persists it) and defaults to the empty list, so a
    call that omits it produces a byte-identical payload to before this parameter
    existed — the `UNANIMOUS_REVERSE_DECISION_ID`/`DISAGREEMENT_DECISION_ID`
    bucket choices documented above depend on that payload text being stable.
    `risk_score` exists only so two otherwise-identical decisions can be given
    DISTINCT `DetectionEvidence` rows (the evidence id is a hash of the payload)
    while still sharing an `evidence_signature`, which reads members/edges/excess
    only."""
    d = Decision(
        decision_id=decision_id,
        account_id=account_id,
        cluster_snapshot_id=cluster_snapshot_id,
        created_at=AS_OF,
        detector_version="det-1",
        policy_version="pol-1",
        risk_score=risk_score,
        proposed_action=proposed_action,
        status=DecisionStatus.PROPOSED,
        run_id="run-1",
        idempotency_key=idempotency_key or uuid.uuid4().hex,
    )
    session.add(d)
    session.flush()

    payload = {
        "cluster_snapshot_id": cluster_snapshot_id,
        "cycle_id": "cycle-1",
        "as_of": AS_OF.isoformat(),
        "snapshot_hash": cluster_snapshot_id,
        "member_account_ids": member_account_ids,
        "edges": edges if edges is not None else [],
        "qualified_stream_count": qualified_stream_count,
        "allowance": 25,
        "ambiguous": False,
        "excess": excess,
        "risk_score": risk_score,
        "proposed_action": proposed_action.value,
        "cap_reasons": [],
        "suppressed": False,
        "detector_version": "det-1",
        "policy_version": "pol-1",
        "feature_manifest_hash": "fmh-1",
        "rung": "S2",
        # Defaults to empty so every pre-P3.9 caller produces a byte-identical
        # payload (the pinned rule-table bucket ids above depend on that text).
        "contributions": contributions if contributions is not None else [],
        "window_start": (AS_OF - timedelta(days=30)).isoformat(),
        "window_end": AS_OF.isoformat(),
        "timeline": [],
    }
    evidence = persist_evidence(session, payload)
    associate_evidence(session, d.decision_id, evidence.evidence_id)
    return d


def _cluster_decision(
    session: Session,
    *,
    decision_id: str,
    account_id: str,
    excess: float = 0.5,
    proposed_action: ProposedAction = ProposedAction.LIGHT_FRICTION,
    cluster_snapshot_id: str = "snap-1",
    member_account_ids: list[str] | None = None,
) -> Decision:
    members = member_account_ids or [account_id]
    return _decision_with_evidence(
        session,
        decision_id=decision_id,
        account_id=account_id,
        cluster_snapshot_id=cluster_snapshot_id,
        member_account_ids=members,
        excess=excess,
        proposed_action=proposed_action,
    )


def _fixture(
    proposed_action: ProposedAction, *, decision_id: str = "d1", mode: FakeMode = FakeMode.NORMAL
) -> tuple[Session, Decision, _CountingProvider]:
    session = _session()
    _account(session, "acct-1")
    _cluster_snapshot(session, "snap-1", ["acct-1"])
    decision = _cluster_decision(
        session, decision_id=decision_id, account_id="acct-1", proposed_action=proposed_action
    )
    provider = _CountingProvider(FakeReviewerProvider(mode=mode))
    return session, decision, provider


# --- Gating matrix ------------------------------------------------------------


def test_allow_decision_is_not_invoked() -> None:
    session, decision, provider = _fixture(ProposedAction.ALLOW)
    outcome = review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )
    assert outcome.gated is True
    assert outcome.gate_reason == "allow"
    assert provider.calls == 0
    assert session.scalars(select(LlmReview)).all() == []


def test_suppressed_decision_is_not_invoked() -> None:
    session, decision, provider = _fixture(ProposedAction.LIGHT_FRICTION)
    payload = {
        "member_account_ids": ["acct-1"],
        "edges": [],
        "excess": 0.5,
    }
    signature = evidence_signature(
        payload["member_account_ids"], payload["edges"], payload["excess"]
    )
    session.add(
        SuppressionRecord(
            suppression_id="sup-1",
            evidence_signature=signature,
            cluster_snapshot_id="snap-1",
            decision_id=decision.decision_id,
            created_at=AS_OF - timedelta(days=1),
            expires_at=AS_OF + timedelta(days=59),
        )
    )
    session.flush()

    outcome = review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )
    assert outcome.gated is True
    assert outcome.gate_reason == "suppressed"
    assert provider.calls == 0


def test_superseded_decision_is_not_invoked() -> None:
    session, decision, provider = _fixture(ProposedAction.LIGHT_FRICTION)
    transition_decision(
        session, decision, DecisionStatus.SUPERSEDED, actor=ActorType.SYSTEM, reason="new run"
    )

    outcome = review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )
    assert outcome.gated is True
    assert outcome.gate_reason == "terminal_status:superseded"
    assert provider.calls == 0


def test_reversed_decision_is_not_invoked() -> None:
    session, decision, provider = _fixture(ProposedAction.LIGHT_FRICTION)
    transition_decision(session, decision, DecisionStatus.UNDER_REVIEW, actor=ActorType.ANALYST)
    transition_decision(session, decision, DecisionStatus.REVERSED, actor=ActorType.ANALYST)

    outcome = review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )
    assert outcome.gated is True
    assert outcome.gate_reason == "terminal_status:reversed"
    assert provider.calls == 0


def test_already_under_review_decision_is_not_invoked() -> None:
    session, decision, provider = _fixture(ProposedAction.LIGHT_FRICTION)
    transition_decision(session, decision, DecisionStatus.UNDER_REVIEW, actor=ActorType.ANALYST)

    outcome = review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )
    assert outcome.gated is True
    assert outcome.gate_reason == "already_under_review"
    assert provider.calls == 0


@pytest.mark.parametrize(
    "proposed_action",
    [
        ProposedAction.LIGHT_FRICTION,
        ProposedAction.HOLD_INCREMENTAL_CREDITS,
        ProposedAction.TEMPORARY_HOLD,
    ],
)
def test_any_non_allow_live_decision_is_invoked(proposed_action: ProposedAction) -> None:
    session, decision, provider = _fixture(proposed_action)
    outcome = review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )
    assert outcome.gated is False
    assert provider.calls == 3


# --- Disagreement --------------------------------------------------------------


def test_2_of_3_disagreement_escalates_and_transitions_to_under_review() -> None:
    session, decision, provider = _fixture(
        ProposedAction.LIGHT_FRICTION,
        decision_id=DISAGREEMENT_DECISION_ID,
        mode=FakeMode.DISAGREE,
    )
    outcome = review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )

    effects = {s.effect for s in outcome.samples}
    assert len(effects) > 1, "fixture must actually produce a disagreement"
    assert outcome.overall_effect == "escalate"
    assert outcome.applied is False
    assert decision.status is DecisionStatus.UNDER_REVIEW
    assert provider.calls == 3

    records = session.scalars(
        select(EscalationRecord).where(EscalationRecord.decision_id == decision.decision_id)
    ).all()
    assert len(records) == 1
    assert records[0].source == "llm_review"


# --- R-6 ceiling -----------------------------------------------------------


def test_hold_incremental_credits_never_auto_reverses_even_if_unanimous() -> None:
    session, decision, provider = _fixture(
        ProposedAction.HOLD_INCREMENTAL_CREDITS,
        decision_id=UNANIMOUS_REVERSE_DECISION_ID,
        mode=FakeMode.NORMAL,
    )
    outcome = review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )

    assert {s.effect for s in outcome.samples} == {"reverse"}, (
        "fixture must actually produce a unanimous per-sample reverse for the "
        "ceiling check to be meaningful"
    )
    assert outcome.overall_effect == "escalate"
    assert outcome.applied is False
    assert decision.status is DecisionStatus.UNDER_REVIEW
    records = session.scalars(
        select(EscalationRecord).where(EscalationRecord.decision_id == decision.decision_id)
    ).all()
    assert len(records) == 1


# --- Cluster blast radius (RC-1 P1-1) ---------------------------------------


def test_multi_actioned_cluster_member_always_escalates() -> None:
    session = _session()
    _account(session, "acct-1")
    _account(session, "acct-2")
    _cluster_snapshot(session, "snap-1", ["acct-1", "acct-2"])
    decision = _decision_with_evidence(
        session,
        decision_id=UNANIMOUS_REVERSE_DECISION_ID,
        account_id="acct-1",
        cluster_snapshot_id="snap-1",
        member_account_ids=["acct-1"],
        excess=0.5,
        proposed_action=ProposedAction.LIGHT_FRICTION,
    )
    # A second, currently-actioned decision in the SAME cluster snapshot with the
    # SAME evidence signature (member set, edges, canonical excess).
    _decision_with_evidence(
        session,
        decision_id="sibling",
        account_id="acct-2",
        cluster_snapshot_id="snap-1",
        member_account_ids=["acct-1"],
        excess=0.5,
        proposed_action=ProposedAction.LIGHT_FRICTION,
    )

    provider = _CountingProvider(FakeReviewerProvider(mode=FakeMode.NORMAL))
    outcome = review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )

    assert {s.effect for s in outcome.samples} == {"reverse"}
    assert outcome.overall_effect == "escalate"
    assert outcome.applied is False
    assert decision.status is DecisionStatus.UNDER_REVIEW
    records = session.scalars(
        select(EscalationRecord).where(EscalationRecord.decision_id == decision.decision_id)
    ).all()
    assert len(records) == 1


def test_single_member_cluster_with_no_other_actioned_sibling_can_auto_apply() -> None:
    # Sanity control for the above: an otherwise-identical sibling that is already
    # REVERSED (not currently-actioned) must NOT block auto-apply.
    session = _session()
    _account(session, "acct-1")
    _account(session, "acct-2")
    _cluster_snapshot(session, "snap-1", ["acct-1", "acct-2"])
    decision = _decision_with_evidence(
        session,
        decision_id=UNANIMOUS_REVERSE_DECISION_ID,
        account_id="acct-1",
        cluster_snapshot_id="snap-1",
        member_account_ids=["acct-1"],
        excess=0.5,
        proposed_action=ProposedAction.LIGHT_FRICTION,
    )
    sibling = _decision_with_evidence(
        session,
        decision_id="sibling",
        account_id="acct-2",
        cluster_snapshot_id="snap-1",
        member_account_ids=["acct-1"],
        excess=0.5,
        proposed_action=ProposedAction.LIGHT_FRICTION,
    )
    transition_decision(session, sibling, DecisionStatus.UNDER_REVIEW, actor=ActorType.ANALYST)
    transition_decision(session, sibling, DecisionStatus.REVERSED, actor=ActorType.ANALYST)

    provider = _CountingProvider(FakeReviewerProvider(mode=FakeMode.NORMAL))
    outcome = review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )

    assert outcome.overall_effect == "reverse"
    assert outcome.applied is True
    assert decision.status is DecisionStatus.REVERSED


# --- Cluster blast radius with REAL edges (RC-1 P1-1 regression) --------------
#
# The two tests above build their fixtures with `edges == []`, which makes them
# vacuous with respect to the guard's edge handling: the guard used to sign the
# decision under review from the `ReviewCase` projection (3 fields per edge, read
# by `evidence_signature` as the legacy-v1 form) while signing every candidate
# sibling from the raw 6-field payload, so the two signatures could never match
# for a cluster with real edges — the blocking branch silently never fired. With
# no edges at all both projections collapse to `[]` and the signatures agree by
# accident, so the bug was invisible. Everything below uses `CANONICAL_EDGES`.


def _edged_cluster_pair(
    *, sibling_status_reversed: bool
) -> tuple[Session, Decision, Decision]:
    """Decision under review + one sibling, sharing a cluster snapshot AND an exact
    `evidence_signature` (same members, same canonical 6-field edges, same excess)
    while being genuinely DISTINCT `Decision` and `DetectionEvidence` rows — the
    sibling's `risk_score` differs, which changes the evidence payload hash but is
    not read by `evidence_signature`."""
    session = _session()
    _account(session, "acct-1")
    _account(session, "acct-2")
    _cluster_snapshot(session, "snap-1", ["acct-1", "acct-2"])
    decision = _decision_with_evidence(
        session,
        decision_id=UNANIMOUS_REVERSE_WITH_EDGES_DECISION_ID,
        account_id="acct-1",
        cluster_snapshot_id="snap-1",
        member_account_ids=["acct-1", "acct-2"],
        excess=0.5,
        proposed_action=ProposedAction.LIGHT_FRICTION,
        edges=CANONICAL_EDGES,
    )
    sibling = _decision_with_evidence(
        session,
        decision_id="sibling",
        account_id="acct-2",
        cluster_snapshot_id="snap-1",
        member_account_ids=["acct-1", "acct-2"],
        excess=0.5,
        proposed_action=ProposedAction.LIGHT_FRICTION,
        edges=CANONICAL_EDGES,
        risk_score=0.42,
    )
    if sibling_status_reversed:
        transition_decision(session, sibling, DecisionStatus.UNDER_REVIEW, actor=ActorType.ANALYST)
        transition_decision(session, sibling, DecisionStatus.REVERSED, actor=ActorType.ANALYST)

    this_payload = decision_evidence(session, decision.decision_id)
    other_payload = decision_evidence(session, sibling.decision_id)
    assert this_payload is not None and other_payload is not None
    # The fixture is only meaningful if the edges are real, the evidence rows are
    # separate, and the signatures nonetheless match exactly.
    assert this_payload["edges"] == CANONICAL_EDGES != []
    assert this_payload["evidence_id"] != other_payload["evidence_id"]
    assert evidence_signature(
        this_payload["member_account_ids"], this_payload["edges"], this_payload["excess"]
    ) == evidence_signature(
        other_payload["member_account_ids"], other_payload["edges"], other_payload["excess"]
    )
    return session, decision, sibling


def test_other_actioned_cluster_member_with_real_edges_is_detected() -> None:
    """RC-1 P1-1, the previously-broken TRUE branch: a currently-actioned sibling
    sharing the evidence signature must be found even when the cluster has real
    (non-empty, canonical 6-field) edges."""
    session, decision, sibling = _edged_cluster_pair(sibling_status_reversed=False)
    assert sibling.status is DecisionStatus.PROPOSED
    assert _other_actioned_cluster_members(session, decision) is True


def test_reversed_cluster_member_with_real_edges_is_not_detected() -> None:
    """Control for the above on the same non-empty-edges fixture: a terminal
    (REVERSED) sibling is not currently actioned and must not be reported."""
    session, decision, sibling = _edged_cluster_pair(sibling_status_reversed=True)
    assert sibling.status is DecisionStatus.REVERSED
    assert _other_actioned_cluster_members(session, decision) is False


def test_actioned_cluster_member_with_real_edges_blocks_auto_apply() -> None:
    """End-to-end form of the above: unanimous `reverse` on a decision whose cluster
    has an actioned sibling must escalate instead of silently auto-reversing."""
    session, decision, _sibling = _edged_cluster_pair(sibling_status_reversed=False)

    provider = _CountingProvider(FakeReviewerProvider(mode=FakeMode.NORMAL))
    outcome = review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )

    # Unanimity is what makes this attributable to the guard: the escalate is not
    # the C-5 disagreement path, and `light_friction` keeps it below the severity
    # short-circuit that precedes the guard in `review_decision`.
    assert {s.effect for s in outcome.samples} == {"reverse"}
    assert outcome.overall_effect == "escalate"
    assert outcome.applied is False
    assert decision.status is DecisionStatus.UNDER_REVIEW
    records = session.scalars(
        select(EscalationRecord).where(EscalationRecord.decision_id == decision.decision_id)
    ).all()
    assert len(records) == 1
    # No reversal happened, so `record_reversal`'s cluster-wide suppression must
    # not exist — the concrete harm P1-1 guards against.
    assert session.scalars(select(SuppressionRecord)).all() == []


def test_reversed_cluster_member_with_real_edges_does_not_block_auto_apply() -> None:
    """Non-vacuous control for the above: same real-edges fixture, sibling already
    REVERSED, so auto-apply proceeds."""
    session, decision, _sibling = _edged_cluster_pair(sibling_status_reversed=True)

    provider = _CountingProvider(FakeReviewerProvider(mode=FakeMode.NORMAL))
    outcome = review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )

    assert {s.effect for s in outcome.samples} == {"reverse"}
    assert outcome.overall_effect == "reverse"
    assert outcome.applied is True
    assert decision.status is DecisionStatus.REVERSED
    assert len(session.scalars(select(SuppressionRecord)).all()) == 1


# --- M-11: `review_cluster_batch` narrows RC-1 P1-1 for co-reviewed siblings --
#
# `review_decision`'s single-decision behavior (tested above) is UNCHANGED by
# M-11: any live signature-sharing sibling still blocks unconditionally. These
# tests cover the new, separate `review_cluster_batch` entry point. Decision ids
# below were chosen empirically against `FakeReviewerProvider`'s rule table
# (bucket = seed % 3; bucket 0 = "allow", which recommends less than the
# `light_friction` proposal and so derives to a unanimous "reverse" under
# `FakeMode.NORMAL`) against the exact 2-account `snap-1` cluster-snapshot
# fixture below, since the snapshot's own member list feeds the rendered
# structured text and therefore the seed: "dec-2"/acct-1 and "dec-8"/acct-2
# both bucket "allow" (reverse); "dec-4"/acct-2 buckets "light_friction"
# (== proposed, so "uphold").


def test_batch_two_confirmed_siblings_both_auto_apply() -> None:
    """M-11's positive case: two decisions sharing one evidence signature, BOTH
    reviewed in the same batch call, BOTH independently reach unanimous
    reverse — both must auto-apply, order-independent (this batch call reviews
    them together, not sequentially)."""
    session = _session()
    _account(session, "acct-1")
    _account(session, "acct-2")
    _cluster_snapshot(session, "snap-1", ["acct-1", "acct-2"])
    dec_a = _decision_with_evidence(
        session,
        decision_id="dec-2",
        account_id="acct-1",
        cluster_snapshot_id="snap-1",
        member_account_ids=["acct-1"],
        excess=0.5,
        proposed_action=ProposedAction.LIGHT_FRICTION,
    )
    dec_b = _decision_with_evidence(
        session,
        decision_id="dec-8",
        account_id="acct-2",
        cluster_snapshot_id="snap-1",
        member_account_ids=["acct-1"],
        excess=0.5,
        proposed_action=ProposedAction.LIGHT_FRICTION,
    )

    provider = _CountingProvider(FakeReviewerProvider(mode=FakeMode.NORMAL))
    outcomes = review_cluster_batch(
        session,
        [dec_a, dec_b],
        provider=provider,
        run_id="r1",
        idempotency_key_fn=lambda d: f"k-{d.decision_id}",
        as_of=AS_OF,
    )

    by_id = {o.decision_id: o for o in outcomes}
    assert by_id["dec-2"].overall_effect == "reverse"
    assert by_id["dec-2"].applied is True
    assert by_id["dec-8"].overall_effect == "reverse"
    assert by_id["dec-8"].applied is True
    assert dec_a.status is DecisionStatus.REVERSED
    assert dec_b.status is DecisionStatus.REVERSED
    # Both reversals go through the real `record_reversal` path — one suppression
    # record per decision (each keyed on the shared evidence signature).
    assert len(session.scalars(select(SuppressionRecord)).all()) == 2


def test_batch_one_unconfirmed_sibling_blocks_only_the_reverse_candidate() -> None:
    """M-11's negative control: one member reaches unanimous reverse, the other
    reaches "uphold" (not "reverse") — the reverse candidate cannot confirm its
    sibling and must escalate instead of auto-applying. The uphold sibling is
    untouched (uphold makes no state change either way, with or without M-11)."""
    session = _session()
    _account(session, "acct-1")
    _account(session, "acct-2")
    _cluster_snapshot(session, "snap-1", ["acct-1", "acct-2"])
    dec_a = _decision_with_evidence(
        session,
        decision_id="dec-2",
        account_id="acct-1",
        cluster_snapshot_id="snap-1",
        member_account_ids=["acct-1"],
        excess=0.5,
        proposed_action=ProposedAction.LIGHT_FRICTION,
    )
    dec_b = _decision_with_evidence(
        session,
        decision_id="dec-4",
        account_id="acct-2",
        cluster_snapshot_id="snap-1",
        member_account_ids=["acct-1"],
        excess=0.5,
        proposed_action=ProposedAction.LIGHT_FRICTION,
    )

    provider = _CountingProvider(FakeReviewerProvider(mode=FakeMode.NORMAL))
    outcomes = review_cluster_batch(
        session,
        [dec_a, dec_b],
        provider=provider,
        run_id="r1",
        idempotency_key_fn=lambda d: f"k-{d.decision_id}",
        as_of=AS_OF,
    )

    by_id = {o.decision_id: o for o in outcomes}
    assert by_id["dec-2"].overall_effect == "escalate"
    assert by_id["dec-2"].applied is False
    assert dec_a.status is DecisionStatus.UNDER_REVIEW
    assert by_id["dec-4"].overall_effect == "uphold"
    assert by_id["dec-4"].applied is False
    assert dec_b.status is DecisionStatus.PROPOSED  # uphold: no transition at all
    assert session.scalars(select(SuppressionRecord)).all() == []


def test_batch_unrelated_sibling_with_different_signature_is_ignored() -> None:
    """Control (c): a live decision on the SAME cluster snapshot but a DIFFERENT
    evidence signature (different excess) must not block auto-apply — M-11 only
    narrows the check for decisions sharing the exact same signature, it does
    not loosen the guard for genuinely unrelated co-members."""
    session = _session()
    _account(session, "acct-1")
    _account(session, "acct-2")
    _cluster_snapshot(session, "snap-1", ["acct-1", "acct-2"])
    dec_a = _decision_with_evidence(
        session,
        decision_id="dec-2",
        account_id="acct-1",
        cluster_snapshot_id="snap-1",
        member_account_ids=["acct-1"],
        excess=0.5,
        proposed_action=ProposedAction.LIGHT_FRICTION,
    )
    # A genuinely unrelated, still-live decision on the same snapshot — NOT
    # included in this batch call — with a different excess (different
    # evidence_signature).
    _decision_with_evidence(
        session,
        decision_id="unrelated",
        account_id="acct-2",
        cluster_snapshot_id="snap-1",
        member_account_ids=["acct-1", "acct-2"],
        excess=0.9,
        proposed_action=ProposedAction.LIGHT_FRICTION,
    )

    provider = _CountingProvider(FakeReviewerProvider(mode=FakeMode.NORMAL))
    outcomes = review_cluster_batch(
        session,
        [dec_a],
        provider=provider,
        run_id="r1",
        idempotency_key_fn=lambda d: f"k-{d.decision_id}",
        as_of=AS_OF,
    )

    assert outcomes[0].overall_effect == "reverse"
    assert outcomes[0].applied is True
    assert dec_a.status is DecisionStatus.REVERSED


# --- Real auto-reversal ------------------------------------------------------


def test_real_auto_reversal_transitions_and_produces_one_suppression_and_credit_restore() -> None:
    session, decision, provider = _fixture(
        ProposedAction.LIGHT_FRICTION,
        decision_id=UNANIMOUS_REVERSE_DECISION_ID,
        mode=FakeMode.NORMAL,
    )
    # A held claim in the current cycle so `record_reversal`'s credit-restore path
    # (policy §9.1) has something real to restore.
    session.add(
        EntitlementCycle(
            cycle_id="cycle-current",
            account_id="acct-1",
            cycle_start=AS_OF - timedelta(days=5),
            cycle_end=AS_OF + timedelta(days=25),
            credits_granted=25,
            credits_consumed=0,
            credits_expired=0,
        )
    )
    session.flush()
    session.add(
        SyntheticClaim(
            claim_id="claim-1",
            request_key="req-1",
            account_id="acct-1",
            cycle_id="cycle-current",
            requested_amount=10,
            granted_amount=0,
            held_amount=10,
            decision_id=decision.decision_id,
            created_at=AS_OF,
            deferred=False,
            hold_state="confirmed",
        )
    )
    session.flush()

    outcome = review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )

    assert {s.effect for s in outcome.samples} == {"reverse"}
    assert outcome.overall_effect == "reverse"
    assert outcome.applied is True
    assert decision.status is DecisionStatus.REVERSED

    status_changes = session.scalars(
        select(AuditEvent).where(
            AuditEvent.event_name == "decision_status_changed",
            AuditEvent.entity_id == decision.decision_id,
        )
    ).all()
    to_statuses = [e.payload_json["to_status"] for e in status_changes]
    assert to_statuses == ["under_review", "reversed"]
    assert all(e.payload_json["actor_type"] == "llm" for e in status_changes)

    suppressions = session.scalars(select(SuppressionRecord)).all()
    assert len(suppressions) == 1
    assert suppressions[0].decision_id == decision.decision_id

    credit_events = session.scalars(
        select(AuditEvent).where(AuditEvent.event_name == "credits_restored")
    ).all()
    assert len(credit_events) == 1

    # The one new transition edge this step relies on (proposed -> under_review)
    # must already be legal — confirming it, not merely assuming it.
    assert DecisionStatus.UNDER_REVIEW in LEGAL_TRANSITIONS[DecisionStatus.PROPOSED]


def test_llm_auto_reversal_suppression_blocks_nightly_reflag_of_same_evidence() -> None:
    """P3.8: extends the test above with the one assertion it does not make --
    that the `SuppressionRecord` the LLM auto-reversal path writes actually
    blocks a subsequent nightly evaluation of the identical evidence, not just
    that one record got created. `should_flag` is the exact function
    `detector.run_detector` calls per cluster-cycle unit (policy §9.3) to decide
    whether to re-flag; calling it here with the same evidence signature the
    reversal was keyed on IS that nightly check, without needing a full
    generated world to rediscover the cluster (that path — full `run_detector`
    twice over a generated world — is already covered for the ANALYST reversal
    path by `test_review_fixes.py::test_reversed_cluster_is_not_reflagged`; this
    is the LLM-path analogue at the policy_engine seam both paths share)."""
    session, decision, provider = _fixture(
        ProposedAction.LIGHT_FRICTION,
        decision_id=UNANIMOUS_REVERSE_DECISION_ID,
        mode=FakeMode.NORMAL,
    )
    outcome = review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )
    assert outcome.applied is True
    assert decision.status is DecisionStatus.REVERSED

    signature = evidence_signature(["acct-1"], [], 0.5)
    later = AS_OF + timedelta(days=3)
    assert should_flag(session, signature, later) is False

    # Suppression is scoped to THIS evidence signature, not the account: a
    # nightly run over materially different evidence (a much larger excess) is
    # unaffected and would still be flagged (policy §9.4).
    different_signature = evidence_signature(["acct-1"], [], 5.0)
    assert should_flag(session, different_signature, later) is True


# --- Fail-closed on provider failure -----------------------------------------


@pytest.mark.parametrize("mode", [FakeMode.TIMEOUT, FakeMode.ERROR])
def test_provider_failure_modes_escalate_and_transition_to_under_review(mode: FakeMode) -> None:
    session, decision, provider = _fixture(ProposedAction.LIGHT_FRICTION, mode=mode)
    outcome = review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )

    assert {s.effect for s in outcome.samples} == {"escalate"}
    assert outcome.overall_effect == "escalate"
    assert outcome.applied is False
    assert decision.status is DecisionStatus.UNDER_REVIEW
    records = session.scalars(
        select(EscalationRecord).where(EscalationRecord.decision_id == decision.decision_id)
    ).all()
    assert len(records) == 1

    rows = session.scalars(select(LlmReview)).all()
    assert len(rows) == 3
    assert all(r.derived_verdict == "escalate" for r in rows)
    assert all(r.raw_response is None for r in rows)
    assert all(r.validation_errors is not None for r in rows)


# --- Persistence of all 3 raw attempts ---------------------------------------


def test_all_3_samples_persisted_with_model_snapshot_temperature_and_retry_count() -> None:
    session, decision, provider = _fixture(
        ProposedAction.HOLD_INCREMENTAL_CREDITS, mode=FakeMode.NORMAL
    )
    review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )

    rows = session.scalars(
        select(LlmReview)
        .where(LlmReview.decision_id == decision.decision_id)
        .order_by(LlmReview.sample_index)
    ).all()
    assert len(rows) == 3
    assert [r.sample_index for r in rows] == [0, 1, 2]
    for row in rows:
        assert row.model_snapshot == "fake-reviewer-v1"
        assert row.temperature == 0.0
        assert row.retry_count == 0
        assert row.raw_response is not None
        assert row.derived_verdict in {"uphold", "reverse", "escalate"}
        assert row.provider == "_CountingProvider"
        assert row.arm == "reviewer"
        assert row.idempotency_key == "k1"


# --- Idempotency --------------------------------------------------------------


def test_rerun_with_same_idempotency_key_does_not_double_call_or_duplicate_rows() -> None:
    session, decision, provider = _fixture(
        ProposedAction.HOLD_INCREMENTAL_CREDITS, mode=FakeMode.NORMAL
    )
    first = review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )
    assert provider.calls == 3
    assert first.replayed is False

    second = review_decision(
        session, decision, provider=provider, run_id="r2", idempotency_key="k1", as_of=AS_OF
    )
    assert provider.calls == 3, "provider must not be called again on replay"
    assert second.replayed is True
    assert second.overall_effect == first.overall_effect

    rows = session.scalars(
        select(LlmReview).where(LlmReview.decision_id == decision.decision_id)
    ).all()
    assert len(rows) == 3, "no duplicate rows from the replayed run"


# --- L-8 reviewer-manifest stamping (P3.9) ------------------------------------


def test_every_review_row_and_audit_event_is_stamped_with_the_manifest_hash() -> None:
    """L-8: "its hash stamps every `LlmReview` row, every reviewer audit event,
    and every arm-4 artifact". Asserted on both surfaces, and on the retry rows
    too — a repair-retried sample's intermediate row is as much a record of which
    reviewer version ran as its final one."""
    session, decision, _ = _fixture(ProposedAction.HOLD_INCREMENTAL_CREDITS)
    provider = FakeReviewerProvider(mode=FakeMode.MALFORMED)  # forces the retry rows
    review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )

    expected = reviewer_manifest.MANIFEST_HASH
    assert expected and len(expected) == 64

    rows = session.scalars(
        select(LlmReview).where(LlmReview.decision_id == decision.decision_id)
    ).all()
    assert len(rows) == 6  # 3 samples x (initial + 1 repair retry)
    assert {r.reviewer_manifest_hash for r in rows} == {expected}

    events = session.scalars(
        select(AuditEvent).where(
            AuditEvent.event_name == "llm_review_recorded",
            AuditEvent.entity_id == decision.decision_id,
        )
    ).all()
    assert len(events) == 6
    assert {e.payload_json["reviewer_manifest_hash"] for e in events} == {expected}


def test_provider_failure_rows_are_also_stamped() -> None:
    """The third `_persist_sample_row` call site: a provider-level failure persists
    no response at all, but must still record which reviewer version failed."""
    session, decision, _ = _fixture(ProposedAction.LIGHT_FRICTION, mode=FakeMode.ERROR)
    review_decision(
        session, decision, provider=FakeReviewerProvider(mode=FakeMode.ERROR),
        run_id="r1", idempotency_key="k1", as_of=AS_OF,
    )
    rows = session.scalars(
        select(LlmReview).where(LlmReview.decision_id == decision.decision_id)
    ).all()
    assert len(rows) == 3
    assert all(r.raw_response is None for r in rows)
    assert {r.reviewer_manifest_hash for r in rows} == {reviewer_manifest.MANIFEST_HASH}


def test_stamped_hash_matches_the_committed_reviewer_manifest() -> None:
    """The stamp names a real, reproducible committed artifact — otherwise the
    version-observability guarantee is a string with no referent."""
    assert reviewer_manifest.verified_manifest_hash() == reviewer_manifest.MANIFEST_HASH


# --- Arm configuration / arm 4b (P3.9) ---------------------------------------


def _capture_request(session: Session, decision: Decision, *, arm: str):
    """Run one review through a provider that records the exact `LLMRequest` it
    was handed, so the *rendered prompt* — not just the persisted row — can be
    asserted against."""

    class _Recorder:
        def __init__(self) -> None:
            self.requests: list = []
            self.inner = FakeReviewerProvider(mode=FakeMode.NORMAL)

        def complete(self, request):
            self.requests.append(request)
            return self.inner.complete(request)

    recorder = _Recorder()
    review_decision(
        session, decision, provider=recorder, run_id="r1", idempotency_key="k1",
        as_of=AS_OF, arm=arm,
    )
    assert recorder.requests
    return recorder.requests[0]


@pytest.mark.parametrize(
    "arm,expected_view,expected_prompt_version",
    [
        ("reviewer", "full", "reviewer_v1"),
        ("arm4_reviewer", "full", "reviewer_v1"),
        ("arm4b_timeline_only", "timeline_only", "reviewer_timeline_only_v1"),
        # Unknown arms are recorded verbatim but must not change behavior or
        # raise: `arm` is caller-supplied free text on both the JSON API and the
        # console form, so a dict lookup that raised would be a live 500.
        ("some-unregistered-arm", "full", "reviewer_v1"),
    ],
)
def test_arm_selects_view_and_prompt_version(
    arm: str, expected_view: str, expected_prompt_version: str
) -> None:
    session, decision, _provider = _fixture(ProposedAction.HOLD_INCREMENTAL_CREDITS)
    request = _capture_request(session, decision, arm=arm)

    structured = next(
        s.text for s in request.segments if type(s).__name__ == "StructuredEvidenceSegment"
    )
    assert f"view: {expected_view}" in structured

    rows = session.scalars(
        select(LlmReview).where(LlmReview.decision_id == decision.decision_id)
    ).all()
    assert len(rows) == 3
    assert {r.prompt_version for r in rows} == {expected_prompt_version}
    assert {r.arm for r in rows} == {arm}


def test_arm4b_prompt_omits_contributions_block_and_contribution_citations() -> None:
    """End-to-end masking: the arm-4b prompt the provider actually receives has no
    `contributions:` section and no `contribution:` entry in its `evidence_keys:`
    allowlist, while arm 4's has both. Asserted on the rendered prompt rather than
    on `ReviewCase`, because the prompt is what the model sees."""
    session = _session()
    _account(session, "acct-1")
    _cluster_snapshot(session, "snap-1", ["acct-1"])
    # A payload with a FIRED contribution, so "no contribution keys" is a real
    # exclusion rather than a vacuous one.
    decision = _decision_with_evidence(
        session,
        decision_id="dec-arm4b",
        account_id="acct-1",
        cluster_snapshot_id="snap-1",
        member_account_ids=["acct-1"],
        excess=0.5,
        proposed_action=ProposedAction.HOLD_INCREMENTAL_CREDITS,
    )
    payload = decision_evidence(session, decision.decision_id)
    assert payload is not None
    assert payload["contributions"] == [], "fixture default is contribution-free"

    full_case = build_review_case(session, decision, view="full", as_of=AS_OF)
    masked_case = build_review_case(session, decision, view="timeline_only", as_of=AS_OF)
    full_text = _structured_text(build_prompt(full_case, variant="reviewer_v1"))
    masked_text = _structured_text(
        build_prompt(masked_case, variant="reviewer_timeline_only_v1")
    )

    assert "contributions:" in full_text
    assert "contributions:" not in masked_text
    assert not any(
        line.strip().startswith("- contribution:") for line in masked_text.splitlines()
    )
    # The fake provider cites `evidence_keys[0]`, which must still exist under the
    # mask (`scalar:band`) — otherwise `_rule_table_verdict` would raise rather
    # than exercise arm 4b.
    assert masked_case.evidence_keys[0].key == "band"


def test_arm4b_with_fired_contributions_masks_them_from_prompt_and_allowlist() -> None:
    """The non-vacuous form of the above: an evidence payload that really does
    carry a fired contribution."""
    session = _session()
    _account(session, "acct-1")
    _cluster_snapshot(session, "snap-1", ["acct-1"])
    decision = _decision_with_evidence(
        session,
        decision_id="dec-fired",
        account_id="acct-1",
        cluster_snapshot_id="snap-1",
        member_account_ids=["acct-1"],
        excess=0.5,
        proposed_action=ProposedAction.HOLD_INCREMENTAL_CREDITS,
        contributions=[
            {
                "feature_id": "F-DEV-1", "raw_value": 1.0, "contribution": 0.45,
                "fired": True, "family": 1, "rung": "S1",
            },
        ],
    )

    full_case = build_review_case(session, decision, view="full", as_of=AS_OF)
    masked_case = build_review_case(session, decision, view="timeline_only", as_of=AS_OF)

    assert any(k.kind is EvidenceKeyKind.CONTRIBUTION for k in full_case.evidence_keys)
    assert not any(k.kind is EvidenceKeyKind.CONTRIBUTION for k in masked_case.evidence_keys)

    masked_text = _structured_text(
        build_prompt(masked_case, variant="reviewer_timeline_only_v1")
    )
    assert "F-DEV-1" not in masked_text
    assert "F-DEV-1" in _structured_text(build_prompt(full_case, variant="reviewer_v1"))


def test_parse_failure_rows_record_the_prompt_version_not_the_view() -> None:
    """Regression for the `prompt_version=case.view` copy/paste bug in
    `_run_sample`'s parse-failure branch: an unparseable response used to stamp
    the *view* name onto the row. Arm 4b would have made it invisible rather than
    fixed ("timeline_only" reads plausibly next to "reviewer_timeline_only_v1"),
    so this asserts on the masked arm specifically — the branch is only reachable
    via `FakeMode.MALFORMED`, whose text never parses."""
    session, decision, _ = _fixture(ProposedAction.HOLD_INCREMENTAL_CREDITS)
    provider = FakeReviewerProvider(mode=FakeMode.MALFORMED)
    review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1",
        as_of=AS_OF, arm="arm4b_timeline_only",
    )
    rows = session.scalars(
        select(LlmReview).where(LlmReview.decision_id == decision.decision_id)
    ).all()
    # 3 samples x (initial attempt + 1 repair retry) = 6 rows, all parse failures.
    assert len(rows) == 6
    assert all(r.validation_errors is not None for r in rows)
    assert {r.prompt_version for r in rows} == {"reviewer_timeline_only_v1"}
    assert "timeline_only" not in {r.prompt_version for r in rows} - {
        "reviewer_timeline_only_v1"
    }


def test_different_idempotency_key_does_call_the_provider_again() -> None:
    session, decision, provider = _fixture(
        ProposedAction.HOLD_INCREMENTAL_CREDITS, mode=FakeMode.NORMAL
    )
    review_decision(
        session, decision, provider=provider, run_id="r1", idempotency_key="k1", as_of=AS_OF
    )
    assert provider.calls == 3

    review_decision(
        session, decision, provider=provider, run_id="r2", idempotency_key="k2", as_of=AS_OF
    )
    assert provider.calls == 6

    rows = session.scalars(
        select(LlmReview).where(LlmReview.decision_id == decision.decision_id)
    ).all()
    assert len(rows) == 6

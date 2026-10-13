"""P3.1: the reviewer's read-only case payload (docs/phase-3-plan.md P3.1).

Covers L-1 (case content survives live-table mutation — proves it is built purely
from immutable evidence + the appeal), L-6/C-2 (the evidence-key allowlist spans
scalars, fired contributions, and timeline event ids — not contributions alone),
and the appeal storage-side 2,000-char cap.
"""

from datetime import timedelta

import pytest
from sqlalchemy.orm import Session

from app.db.ground_truth_writer import make_ground_truth_sink
from app.models.enums import ProposedAction, VerificationLevel
from app.models.inference import Account, Decision
from app.schemas.reviewer import EvidenceKeyKind
from app.services.detector import run_detector
from app.services.generator import WORLD_END, generate_world, persist_world
from app.services.review_case import (
    APPEAL_TEXT_MAX_CHARS,
    ReviewCaseError,
    build_review_case,
    record_appeal,
)


@pytest.fixture
def actionable_decision(inference_session: Session, gt_session: Session) -> Decision:
    world = generate_world(1, "min")
    persist_world(world, inference_session, make_ground_truth_sink(gt_session))
    result = run_detector(inference_session, as_of=WORLD_END, run_id="p3.1-fixture")
    assert result.decisions_created > 0  # the min world's farms clear the gates
    decision = next(
        d
        for d in inference_session.query(Decision).all()
        if d.proposed_action is not ProposedAction.ALLOW
    )
    return decision


def test_case_identical_after_unrelated_live_table_mutation(
    inference_session: Session, actionable_decision: Decision
) -> None:
    before = build_review_case(inference_session, actionable_decision, as_of=WORLD_END)

    account = inference_session.get(Account, actionable_decision.account_id)
    assert account is not None
    account.verification_level = (
        VerificationLevel.PHONE
        if account.verification_level is not VerificationLevel.PHONE
        else VerificationLevel.EMAIL
    )
    inference_session.flush()

    after = build_review_case(inference_session, actionable_decision, as_of=WORLD_END)

    assert before == after


def test_evidence_keys_cover_scalars_contributions_and_timeline(
    inference_session: Session, actionable_decision: Decision
) -> None:
    case = build_review_case(inference_session, actionable_decision, as_of=WORLD_END)

    by_kind: dict[EvidenceKeyKind, set[str]] = {kind: set() for kind in EvidenceKeyKind}
    for key in case.evidence_keys:
        by_kind[key.kind].add(key.key)

    assert by_kind[EvidenceKeyKind.SCALAR] == {"band", "allowance", "excess"}

    fired_feature_ids = {c.feature_id for c in case.contributions if c.fired}
    assert fired_feature_ids  # the min world's actionable decision has fired features
    assert by_kind[EvidenceKeyKind.CONTRIBUTION] == fired_feature_ids

    all_event_ids = {eid for bucket in case.timeline for eid in bucket.event_ids}
    assert all_event_ids  # a cluster-cycle window is never event-free here
    assert by_kind[EvidenceKeyKind.TIMELINE_EVENT] == all_event_ids


def test_unsupported_view_raises(
    inference_session: Session, actionable_decision: Decision
) -> None:
    """P3.9: both real views now exist, so an unknown view is an ordinary caller
    error (`ReviewCaseError`, this module's convention) rather than the
    `NotImplementedError` not-built-yet marker it used to be."""
    with pytest.raises(ReviewCaseError, match="unknown review_case view"):
        build_review_case(
            inference_session, actionable_decision, view="contributions_masked", as_of=WORLD_END
        )


# --- Arm 4b: the `timeline_only` masked view (P3.9) ---------------------------


def test_timeline_only_view_is_labelled_and_drops_contribution_evidence_keys(
    inference_session: Session, actionable_decision: Decision
) -> None:
    """The masking that makes arm 4b an honest attribution control: a model never
    shown the contributions must not be able to cite one. `filter_citations`
    ratifies anything in `evidence_keys`, and `derive_effect` needs exactly one
    valid citation to permit a reversal — so leaving CONTRIBUTION keys in a
    contributions-masked case would hand the model a fabrication vector."""
    full = build_review_case(inference_session, actionable_decision, as_of=WORLD_END)
    masked = build_review_case(
        inference_session, actionable_decision, view="timeline_only", as_of=WORLD_END
    )

    assert full.view == "full"
    # The `view` field used to be hardcoded to "full" regardless of the argument.
    assert masked.view == "timeline_only"

    full_kinds = {k.kind for k in full.evidence_keys}
    masked_kinds = {k.kind for k in masked.evidence_keys}
    assert EvidenceKeyKind.CONTRIBUTION in full_kinds, "fixture must have fired features"
    assert EvidenceKeyKind.CONTRIBUTION not in masked_kinds
    # Scalars and timeline events stay citable — arm 4b is not a no-evidence arm.
    assert EvidenceKeyKind.SCALAR in masked_kinds
    assert EvidenceKeyKind.TIMELINE_EVENT in masked_kinds
    assert {k for k in full.evidence_keys if k.kind is not EvidenceKeyKind.CONTRIBUTION} == set(
        masked.evidence_keys
    )


def test_timeline_only_view_leaves_the_underlying_evidence_identical(
    inference_session: Session, actionable_decision: Decision
) -> None:
    """Only the citable allowlist and the `view` label differ between the arms —
    the evidence itself is the same object, which is what makes arm 4 vs arm 4b a
    paired comparison rather than two unrelated runs. (`contributions` remain on
    the `ReviewCase`; `prompt_builder` is what withholds them from the model, so
    the evaluation harness can still read them.)"""
    full = build_review_case(inference_session, actionable_decision, as_of=WORLD_END)
    masked = build_review_case(
        inference_session, actionable_decision, view="timeline_only", as_of=WORLD_END
    )
    assert masked.snapshot_hash == full.snapshot_hash
    assert masked.member_account_ids == full.member_account_ids
    assert masked.edges == full.edges
    assert (masked.band, masked.allowance, masked.excess) == (
        full.band, full.allowance, full.excess
    )
    assert masked.contributions == full.contributions
    assert masked.timeline == full.timeline


def test_missing_evidence_raises(inference_session: Session) -> None:
    from app.db.decisions import record_decision
    from app.models.enums import EmailDomainType

    inference_session.add(
        Account(
            account_id="orphan",
            created_at=WORLD_END - timedelta(days=1),
            email_domain_type=EmailDomainType.CONSUMER,
            verification_level=VerificationLevel.EMAIL,
        )
    )
    inference_session.flush()
    decision = record_decision(
        inference_session,
        account_id="orphan",
        risk_score=0.9,
        proposed_action=ProposedAction.TEMPORARY_HOLD,
        detector_version="v1",
        policy_version="v1",
        run_id="orphan-run",
        idempotency_key="orphan-key",
        cluster_snapshot_id=None,
    )
    with pytest.raises(ReviewCaseError):
        build_review_case(inference_session, decision, as_of=WORLD_END)


def test_appeal_over_cap_is_truncated_and_flagged(
    inference_session: Session, actionable_decision: Decision
) -> None:
    long_text = "x" * (APPEAL_TEXT_MAX_CHARS + 500)
    appeal = record_appeal(
        inference_session,
        account_id=actionable_decision.account_id,
        decision_id=actionable_decision.decision_id,
        text=long_text,
        source="console",
        submitted_at=WORLD_END,
    )
    assert appeal.truncated is True
    assert len(appeal.text) == APPEAL_TEXT_MAX_CHARS
    assert appeal.text == long_text[:APPEAL_TEXT_MAX_CHARS]

    case = build_review_case(inference_session, actionable_decision, as_of=WORLD_END)
    assert case.appeal_truncated is True
    assert case.appeal_text == long_text[:APPEAL_TEXT_MAX_CHARS]


def test_appeal_at_or_under_cap_is_untouched(
    inference_session: Session, actionable_decision: Decision
) -> None:
    exact_text = "y" * APPEAL_TEXT_MAX_CHARS
    appeal = record_appeal(
        inference_session,
        account_id=actionable_decision.account_id,
        decision_id=actionable_decision.decision_id,
        text=exact_text,
        source="console",
        submitted_at=WORLD_END,
    )
    assert appeal.truncated is False
    assert appeal.text == exact_text

    short_text = "z" * 10
    appeal2 = record_appeal(
        inference_session,
        account_id=actionable_decision.account_id,
        decision_id=actionable_decision.decision_id,
        text=short_text,
        source="console",
        submitted_at=WORLD_END + timedelta(seconds=1),
    )
    assert appeal2.truncated is False
    assert appeal2.text == short_text

    case = build_review_case(
        inference_session, actionable_decision, as_of=WORLD_END + timedelta(seconds=1)
    )
    assert case.appeal_truncated is False
    assert case.appeal_text == short_text  # latest-by-submitted_at appeal wins


def test_appeal_can_predate_the_decision(
    inference_session: Session, actionable_decision: Decision
) -> None:
    # decision_id=None: the appeal was filed before this decision existed. It still
    # surfaces on the case for the same account (L-1's "an appeal can predate any
    # decision"), as long as it was submitted at or before `as_of`.
    record_appeal(
        inference_session,
        account_id=actionable_decision.account_id,
        decision_id=None,
        text="I was on vacation, this wasn't me",
        source="console",
        submitted_at=WORLD_END - timedelta(days=1),
    )

    case = build_review_case(inference_session, actionable_decision, as_of=WORLD_END)
    assert case.appeal_text == "I was on vacation, this wasn't me"
    assert case.appeal_truncated is False

"""P3.3: `ReviewerVerdict` schema (L-2/R-4), the parse/repair-retry pipeline (R-7),
C-2 citation filtering, C-4 severity clamping, and the L-3 `derive_effect` table.
"""

import json

import pytest
from pydantic import ValidationError

from app.models.enums import ProposedAction
from app.schemas.reviewer import EvidenceKey, EvidenceKeyKind, ReviewerVerdict
from app.services.reviewer_validation import (
    clamp_action,
    derive_effect,
    evaluate_verdict,
    filter_citations,
    parse_attempt,
)


def _verdict(**overrides: object) -> ReviewerVerdict:
    defaults: dict[str, object] = {
        "evidence_supporting_action": (),
        "benign_explanations": (),
        "missing_evidence": (),
        "unverified_claims": (),
        "rationale": "because",
        "recommended_action": ProposedAction.ALLOW,
        "confidence": 0.5,
        "escalate_to_human": False,
    }
    defaults.update(overrides)
    return ReviewerVerdict(**defaults)


def _verdict_json(**overrides: object) -> str:
    v = _verdict(**overrides)
    payload = json.loads(v.model_dump_json())
    return json.dumps(payload)


# --- R-4 field order -----------------------------------------------------------


def test_field_order_matches_r4() -> None:
    assert list(ReviewerVerdict.model_fields.keys()) == [
        "evidence_supporting_action",
        "benign_explanations",
        "missing_evidence",
        "unverified_claims",
        "rationale",
        "recommended_action",
        "confidence",
        "escalate_to_human",
    ]


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        ReviewerVerdict(
            evidence_supporting_action=(),
            benign_explanations=(),
            missing_evidence=(),
            unverified_claims=(),
            rationale="x",
            recommended_action=ProposedAction.ALLOW,
            confidence=0.1,
            escalate_to_human=False,
            extra_field="nope",
        )


# --- R-7 parse / one repair retry / fail-closed ---------------------------------


def test_malformed_json_needs_retry_then_escalates_after_one_retry() -> None:
    first = parse_attempt("not json at all", retry_count=0)
    assert first.status == "needs_retry"
    assert len(first.attempts) == 1
    assert first.attempts[0].error is not None
    assert first.attempts[0].verdict is None

    second = parse_attempt(
        "still not json", retry_count=1, prior_attempts=first.attempts
    )
    assert second.status == "escalated"
    assert second.verdict is None
    # Both attempts retrievable (R-7: "all attempts logged").
    assert len(second.attempts) == 2
    assert [a.retry_count for a in second.attempts] == [0, 1]
    assert all(a.error is not None for a in second.attempts)


def test_schema_invalid_payload_also_needs_retry() -> None:
    outcome = parse_attempt(json.dumps({"rationale": "missing fields"}), retry_count=0)
    assert outcome.status == "needs_retry"
    assert "schema_validation_failed" in (outcome.attempts[0].error or "")


def test_successful_parse_on_retry_recovers() -> None:
    first = parse_attempt("garbage", retry_count=0)
    assert first.status == "needs_retry"

    good_raw = _verdict_json(rationale="fixed on retry")
    second = parse_attempt(good_raw, retry_count=1, prior_attempts=first.attempts)
    assert second.status == "parsed"
    assert second.verdict is not None
    assert second.verdict.rationale == "fixed on retry"
    assert len(second.attempts) == 2


def test_valid_json_parses_on_first_attempt() -> None:
    outcome = parse_attempt(_verdict_json(), retry_count=0)
    assert outcome.status == "parsed"
    assert outcome.verdict is not None
    assert len(outcome.attempts) == 1


# --- C-2 citation filtering ------------------------------------------------------


_ALLOWLIST = (
    EvidenceKey(kind=EvidenceKeyKind.SCALAR, key="band"),
    EvidenceKey(kind=EvidenceKeyKind.CONTRIBUTION, key="F-CYC-1"),
    EvidenceKey(kind=EvidenceKeyKind.TIMELINE_EVENT, key="evt-1"),
)


def test_unknown_citation_moves_to_unverified_claims() -> None:
    verdict = _verdict(
        evidence_supporting_action=("scalar:band", "scalar:made_up"),
        recommended_action=ProposedAction.ALLOW,
    )
    filtered, valid_count = filter_citations(verdict, _ALLOWLIST)
    assert valid_count == 1
    assert filtered.evidence_supporting_action == ("scalar:band",)
    assert "scalar:made_up" in filtered.unverified_claims


def test_unverified_claims_are_appended_not_discarded() -> None:
    verdict = _verdict(
        evidence_supporting_action=("scalar:unknown",),
        unverified_claims=("pre-existing claim",),
    )
    filtered, valid_count = filter_citations(verdict, _ALLOWLIST)
    assert valid_count == 0
    assert filtered.unverified_claims == ("pre-existing claim", "scalar:unknown")


def test_unknown_citation_alone_cannot_support_reverse() -> None:
    verdict = _verdict(
        evidence_supporting_action=("scalar:made_up",),
        recommended_action=ProposedAction.ALLOW,
        escalate_to_human=False,
    )
    result = evaluate_verdict(
        verdict, _ALLOWLIST, policy_proposed_action=ProposedAction.LIGHT_FRICTION
    )
    assert result.valid_citation_count == 0
    assert result.effect == "escalate"


# --- C-4 severity clamp ----------------------------------------------------------


def test_clamp_lowers_model_action_to_policy_ceiling() -> None:
    clamped = clamp_action(ProposedAction.TEMPORARY_HOLD, ProposedAction.LIGHT_FRICTION)
    assert clamped == ProposedAction.LIGHT_FRICTION


def test_clamp_keeps_model_action_when_less_or_equal_severe() -> None:
    assert (
        clamp_action(ProposedAction.ALLOW, ProposedAction.LIGHT_FRICTION)
        == ProposedAction.ALLOW
    )
    assert (
        clamp_action(ProposedAction.LIGHT_FRICTION, ProposedAction.LIGHT_FRICTION)
        == ProposedAction.LIGHT_FRICTION
    )


def test_temporary_hold_recommendation_clamps_down_before_derivation() -> None:
    verdict = _verdict(
        evidence_supporting_action=("scalar:band",),
        recommended_action=ProposedAction.TEMPORARY_HOLD,
        escalate_to_human=False,
    )
    result = evaluate_verdict(
        verdict, _ALLOWLIST, policy_proposed_action=ProposedAction.LIGHT_FRICTION
    )
    assert result.clamped_action == ProposedAction.LIGHT_FRICTION
    # clamped (light_friction) == proposed (light_friction) -> not strictly lower.
    assert result.effect == "uphold"


# --- L-3 derive_effect: exhaustive truth table -----------------------------------


def test_escalate_to_human_always_escalates_regardless_of_citations_or_action() -> None:
    verdict = _verdict(
        evidence_supporting_action=("scalar:band",),
        recommended_action=ProposedAction.ALLOW,
        escalate_to_human=True,
    )
    assert (
        derive_effect(verdict, ProposedAction.TEMPORARY_HOLD, valid_citation_count=1)
        == "escalate"
    )


def test_zero_citations_forces_escalate_even_if_model_wanted_reverse() -> None:
    verdict = _verdict(
        recommended_action=ProposedAction.ALLOW, escalate_to_human=False
    )
    assert (
        derive_effect(verdict, ProposedAction.LIGHT_FRICTION, valid_citation_count=0)
        == "escalate"
    )


def test_reverse_requires_strictly_lower_clamped_action_at_or_below_ceiling() -> None:
    verdict = _verdict(
        recommended_action=ProposedAction.ALLOW, escalate_to_human=False
    )
    assert (
        derive_effect(verdict, ProposedAction.LIGHT_FRICTION, valid_citation_count=1)
        == "reverse"
    )
    assert (
        derive_effect(
            verdict, ProposedAction.HOLD_INCREMENTAL_CREDITS, valid_citation_count=1
        )
        == "reverse"
    )


def test_clamped_action_above_light_friction_ceiling_never_reverses() -> None:
    # policy proposes temporary_hold, model recommends hold_incremental_credits:
    # strictly lower than proposed, but still above the light_friction ceiling.
    verdict = _verdict(
        recommended_action=ProposedAction.HOLD_INCREMENTAL_CREDITS,
        escalate_to_human=False,
    )
    assert (
        derive_effect(verdict, ProposedAction.TEMPORARY_HOLD, valid_citation_count=1)
        == "uphold"
    )


@pytest.mark.parametrize(
    "recommended,proposed",
    [
        (ProposedAction.ALLOW, ProposedAction.ALLOW),
        (ProposedAction.LIGHT_FRICTION, ProposedAction.LIGHT_FRICTION),
        (ProposedAction.HOLD_INCREMENTAL_CREDITS, ProposedAction.HOLD_INCREMENTAL_CREDITS),
        (ProposedAction.TEMPORARY_HOLD, ProposedAction.TEMPORARY_HOLD),
        (ProposedAction.LIGHT_FRICTION, ProposedAction.ALLOW),
        (ProposedAction.TEMPORARY_HOLD, ProposedAction.ALLOW),
        (ProposedAction.TEMPORARY_HOLD, ProposedAction.LIGHT_FRICTION),
    ],
)
def test_same_or_higher_recommendation_never_reverses(
    recommended: ProposedAction, proposed: ProposedAction
) -> None:
    verdict = _verdict(recommended_action=recommended, escalate_to_human=False)
    assert derive_effect(verdict, proposed, valid_citation_count=1) == "uphold"


@pytest.mark.parametrize("escalate_to_human", [True, False])
@pytest.mark.parametrize("valid_citation_count", [0, 1, 2])
@pytest.mark.parametrize(
    "recommended,proposed",
    [
        (ProposedAction.ALLOW, ProposedAction.ALLOW),
        (ProposedAction.ALLOW, ProposedAction.LIGHT_FRICTION),
        (ProposedAction.ALLOW, ProposedAction.HOLD_INCREMENTAL_CREDITS),
        (ProposedAction.ALLOW, ProposedAction.TEMPORARY_HOLD),
        (ProposedAction.LIGHT_FRICTION, ProposedAction.HOLD_INCREMENTAL_CREDITS),
        (ProposedAction.LIGHT_FRICTION, ProposedAction.TEMPORARY_HOLD),
        (ProposedAction.HOLD_INCREMENTAL_CREDITS, ProposedAction.TEMPORARY_HOLD),
        (ProposedAction.HOLD_INCREMENTAL_CREDITS, ProposedAction.HOLD_INCREMENTAL_CREDITS),
        (ProposedAction.TEMPORARY_HOLD, ProposedAction.TEMPORARY_HOLD),
    ],
)
def test_derive_effect_truth_table_is_exhaustive(
    escalate_to_human: bool,
    valid_citation_count: int,
    recommended: ProposedAction,
    proposed: ProposedAction,
) -> None:
    verdict = _verdict(
        recommended_action=recommended, escalate_to_human=escalate_to_human
    )
    effect = derive_effect(verdict, proposed, valid_citation_count)

    if escalate_to_human or valid_citation_count == 0:
        assert effect == "escalate"
        return

    clamped = clamp_action(recommended, proposed)
    from app.services.policy_engine import ACTION_SEVERITY

    should_reverse = (
        ACTION_SEVERITY[clamped] < ACTION_SEVERITY[proposed]
        and ACTION_SEVERITY[clamped] <= ACTION_SEVERITY[ProposedAction.LIGHT_FRICTION]
    )
    assert effect == ("reverse" if should_reverse else "uphold")

"""The status machine — the one tested function (docs/architecture.md §3)."""

import pytest

from app.core.decision_status import (
    SUPERSEDABLE,
    TERMINAL,
    IllegalTransition,
    assert_transition,
    is_legal_transition,
    is_terminal,
)
from app.models.enums import DecisionStatus as S

LEGAL = [
    (S.PROPOSED, S.UNDER_REVIEW),
    (S.PROPOSED, S.APPLIED),
    (S.PROPOSED, S.SUPERSEDED),
    (S.UNDER_REVIEW, S.APPLIED),
    (S.UNDER_REVIEW, S.REVERSED),
    (S.UNDER_REVIEW, S.SUPERSEDED),
    (S.APPLIED, S.REVERSED),
    (S.APPLIED, S.SUPERSEDED),
    (S.APPLIED, S.UNDER_REVIEW),  # L-5 escalation edge (docs/decisions.md §12)
]

ILLEGAL = [
    (S.PROPOSED, S.REVERSED),  # cannot reverse before it exists as an action/review
    (S.APPLIED, S.PROPOSED),  # no going back
    (S.UNDER_REVIEW, S.PROPOSED),
    (S.REVERSED, S.APPLIED),  # terminal
    (S.REVERSED, S.SUPERSEDED),  # terminal — a reversal is a finding, never superseded
    (S.SUPERSEDED, S.APPLIED),  # terminal
    (S.SUPERSEDED, S.PROPOSED),  # terminal
]


@pytest.mark.parametrize(("frm", "to"), LEGAL)
def test_legal_transitions_allowed(frm: S, to: S) -> None:
    assert is_legal_transition(frm, to)
    assert_transition(frm, to)  # does not raise


@pytest.mark.parametrize(("frm", "to"), ILLEGAL)
def test_illegal_transitions_rejected(frm: S, to: S) -> None:
    assert not is_legal_transition(frm, to)
    with pytest.raises(IllegalTransition):
        assert_transition(frm, to)


def test_terminal_states_have_no_exits() -> None:
    for terminal in (S.REVERSED, S.SUPERSEDED):
        for target in S:
            assert not is_legal_transition(terminal, target)


def test_terminal_set() -> None:
    assert TERMINAL == {S.REVERSED, S.SUPERSEDED}


@pytest.mark.parametrize("status", list(S))
def test_is_terminal_agrees_with_terminal(status: S) -> None:
    assert is_terminal(status) is (status in TERMINAL)


def test_supersedable_is_exactly_states_with_edge_to_superseded() -> None:
    assert SUPERSEDABLE == {s for s in S if is_legal_transition(s, S.SUPERSEDED)}

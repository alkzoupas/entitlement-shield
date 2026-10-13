"""The decision status machine — the ONE tested function (docs/architecture.md §3).

    proposed ─▶ under_review ─▶ applied ─▶ reversed
       │             ▲            │
       │             └────────────┘  (L-5 escalation)
       │             │
       └─────────────┴────────────┴──▶ superseded

`reversed` and `superseded` are terminal. Supersession is reachable from every
non-terminal state: a new detector run supersedes the current decision without
mutating its content (docs/architecture.md §3). A `reversed` decision — a finding
of legitimacy — is terminal and is never superseded; a later re-flag creates a new
decision beside it (docs/policy.md §9.4).

`applied -> under_review` is the L-5 escalation edge (docs/decisions.md §12): the
only status-machine change Phase 3's escalation lifecycle needed, since `under_review`
already existed as a target. `under_review -> superseded` (already present below,
predating L-5) is reused, untagged, for the 7-day auto-release terminal transition —
this dict carries no reason metadata; the `escalation_expired` reason is passed as
plain text to `transition_decision`/`assert_transition`'s caller, not encoded here.
"""

from app.models.enums import DecisionStatus

LEGAL_TRANSITIONS: dict[DecisionStatus, frozenset[DecisionStatus]] = {
    DecisionStatus.PROPOSED: frozenset(
        {DecisionStatus.UNDER_REVIEW, DecisionStatus.APPLIED, DecisionStatus.SUPERSEDED}
    ),
    DecisionStatus.UNDER_REVIEW: frozenset(
        {DecisionStatus.APPLIED, DecisionStatus.REVERSED, DecisionStatus.SUPERSEDED}
    ),
    DecisionStatus.APPLIED: frozenset(
        {DecisionStatus.UNDER_REVIEW, DecisionStatus.REVERSED, DecisionStatus.SUPERSEDED}
    ),
    DecisionStatus.REVERSED: frozenset(),
    DecisionStatus.SUPERSEDED: frozenset(),
}

#: Non-terminal states a new detector run may supersede.
SUPERSEDABLE: frozenset[DecisionStatus] = frozenset(
    {DecisionStatus.PROPOSED, DecisionStatus.UNDER_REVIEW, DecisionStatus.APPLIED}
)


class IllegalTransition(Exception):
    """Raised when a decision status transition is not in LEGAL_TRANSITIONS."""


def is_legal_transition(current: DecisionStatus, target: DecisionStatus) -> bool:
    return target in LEGAL_TRANSITIONS[current]


def assert_transition(current: DecisionStatus, target: DecisionStatus) -> None:
    if not is_legal_transition(current, target):
        raise IllegalTransition(
            f"{current.value} -> {target.value} is not a legal decision transition"
        )

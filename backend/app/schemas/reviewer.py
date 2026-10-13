"""The reviewer's read-only case payload (Phase 3 plan P3.1, L-1/L-6/C-2).

`ReviewCase` is what `services/review_case.py` builds. It is deliberately NOT the
`ReviewerVerdict` the model emits (that is `L-2`, defined in a later step) — this is
the *input* side: everything the reviewer is allowed to see, plus the allowlist that
later pins which evidentiary claims a verdict may cite (C-2).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import CapacityBand, ProposedAction


class _CaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceKeyKind(StrEnum):
    """The three evidentiary categories C-2's allowlist must span (RC-1 P0-3) — not
    contributions alone, so a later contributions-masked view (arm 4b) still has
    valid, structurally-groundable citations available."""

    SCALAR = "scalar"
    CONTRIBUTION = "contribution"
    TIMELINE_EVENT = "timeline_event"


# The only scalar names a citation may reference (L-6/C-2: "band/allowance/excess
# scalars"). Fixed, not derived per-case, so the allowlist type itself documents the
# contract instead of only the runtime value.
ScalarEvidenceName = Literal["band", "allowance", "excess"]


class EvidenceKey(_CaseModel):
    """One citable evidentiary fact. Two keys are equal iff `kind` and `key` match —
    used as a set member, so `frozen=True` above makes it hashable."""

    kind: EvidenceKeyKind
    key: str


class ContributionView(_CaseModel):
    """One feature's exported contribution, in `FEATURE_ORDER` (risk_scorer.py)."""

    feature_id: str
    raw_value: float
    contribution: float
    fired: bool
    family: int | None
    rung: str


class EdgeView(_CaseModel):
    account_a: str
    account_b: str
    evidence_count: int


class TimelineBucket(_CaseModel):
    """One rendered timeline row at the case's `applied_timeline_resolution` (D-6).
    `event_ids` are the persisted `AccountEvent` ids folded into this bucket — every
    one of them is independently a valid `EvidenceKey` regardless of bucketing."""

    bucket_start: str  # ISO-8601, resolution-truncated
    account_id: str
    event_type: str
    credits_amount: int
    event_ids: tuple[str, ...]


TimelineResolution = Literal["hourly", "daily"]

#: The evidentiary views a case can be built in (P3.9 arm 4 / arm 4b).
#:
#: - `"full"` — every category: scalars, per-feature contributions, timeline.
#: - `"timeline_only"` — contributions are withheld from the model entirely: they
#:   are absent from the rendered prompt (`prompt_builder` omits the
#:   `contributions:` block for the `reviewer_timeline_only_v1` variant) *and*
#:   absent from `evidence_keys`, so C-2's allowlist cannot ratify a citation to
#:   a contribution the model was never shown. Scalars and timeline events remain
#:   citable, which is exactly why `EvidenceKeyKind` spans three categories
#:   (RC-1 P0-3) rather than contributions alone.
ReviewCaseView = Literal["full", "timeline_only"]


class ReviewCase(_CaseModel):
    """The reviewer's entire evidentiary world for one decision (L-1). Built purely
    from the immutable `DetectionEvidence` payload plus the account's appeal — never
    from `Account` or any other live table."""

    decision_id: str
    account_id: str
    view: ReviewCaseView

    snapshot_hash: str
    member_account_ids: tuple[str, ...]
    edges: tuple[EdgeView, ...]

    band: CapacityBand
    allowance: int
    excess: float
    cap_reasons: tuple[str, ...]

    contributions: tuple[ContributionView, ...]

    timeline: tuple[TimelineBucket, ...]
    applied_timeline_resolution: TimelineResolution

    evidence_keys: tuple[EvidenceKey, ...]

    appeal_text: str | None
    appeal_truncated: bool


class ReviewerVerdict(_CaseModel):
    """L-2's model-facing output contract. R-4 field order — asserted directly
    against `model_fields` insertion order in `test_reviewer_schema.py` — is exactly
    the order below. Per R-3 the model emits one `recommended_action`; the derived
    `uphold`/`reverse`/`escalate` effect is never a field here (see
    `services/reviewer_validation.derive_effect`)."""

    evidence_supporting_action: tuple[str, ...]
    benign_explanations: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    unverified_claims: tuple[str, ...]
    rationale: str
    recommended_action: ProposedAction
    confidence: float = Field(ge=0.0, le=1.0)
    escalate_to_human: bool

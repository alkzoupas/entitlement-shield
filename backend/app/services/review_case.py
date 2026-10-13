"""Builds the reviewer's read-only case payload (Phase 3 plan P3.1, L-1).

`build_review_case` reads only the immutable `DetectionEvidence` payload (via
`services.evidence.decision_evidence`) plus `Appeal` rows — never `Account` or any
other live/mutable table. That is what makes the case replayable and `as_of`-safe:
rebuilding it after unrelated live-table writes yields byte-identical evidentiary
content (see `test_reviewer_case.py`).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.inference import Appeal, Decision
from app.schemas.reviewer import (
    ContributionView,
    EdgeView,
    EvidenceKey,
    EvidenceKeyKind,
    ReviewCase,
    ReviewCaseView,
    TimelineBucket,
    TimelineResolution,
)
from app.services.banding import _band_for
from app.services.evidence import decision_evidence

APPEAL_TEXT_MAX_CHARS = 2000

# D-6: hourly is the default rendering resolution. If hourly grouping would exceed
# the budget, every account's timeline coarsens together to daily — resolution never
# trades off against sampling out some accounts (D-6 "all accounts never sampled").
_TIMELINE_BUCKET_BUDGET = 500


class ReviewCaseError(ValueError):
    """The case cannot be built as requested."""


def record_appeal(
    session: Session,
    *,
    account_id: str,
    text: str,
    source: str,
    submitted_at: datetime,
    decision_id: str | None = None,
    appeal_id: str | None = None,
) -> Appeal:
    """Persist an appeal, enforcing the 2,000-char hard cap at write time.

    This is a storage-side cap only (policy §11 — appeals carry zero evidentiary
    weight and no feature reads this table). The prompt-side C-3 escaping/delimiter
    handling is P3.2's concern, not this one's."""
    truncated = len(text) > APPEAL_TEXT_MAX_CHARS
    appeal = Appeal(
        appeal_id=appeal_id or uuid.uuid4().hex,
        account_id=account_id,
        decision_id=decision_id,
        submitted_at=submitted_at,
        text=text[:APPEAL_TEXT_MAX_CHARS] if truncated else text,
        truncated=truncated,
        source=source,
    )
    session.add(appeal)
    session.flush()
    return appeal


def _latest_appeal(session: Session, decision: Decision, as_of: datetime) -> Appeal | None:
    # Counts for this case if explicitly tied to this decision, or if it predates the
    # decision entirely (decision_id null — an appeal can predate any decision, L-1)
    # for the same account. `as_of` keeps this replay-safe: never a future appeal.
    rows = session.scalars(
        select(Appeal)
        .where(
            Appeal.account_id == decision.account_id,
            Appeal.submitted_at <= as_of,
            (Appeal.decision_id == decision.decision_id) | (Appeal.decision_id.is_(None)),
        )
        .order_by(Appeal.submitted_at.desc())
    ).all()
    return rows[0] if rows else None


def _bucket_start(ts: datetime, resolution: TimelineResolution) -> datetime:
    truncated = ts.replace(minute=0, second=0, microsecond=0)
    if resolution == "hourly":
        return truncated
    return truncated.replace(hour=0)


def _bucket_events(events: list[dict], resolution: TimelineResolution) -> list[TimelineBucket]:
    buckets: dict[tuple[str, str, str], dict] = {}
    order: list[tuple[str, str, str]] = []
    for event in events:
        start = _bucket_start(datetime.fromisoformat(event["timestamp"]), resolution)
        key = (event["account_id"], event["event_type"], start.isoformat())
        if key not in buckets:
            buckets[key] = {"credits_amount": 0, "event_ids": []}
            order.append(key)
        buckets[key]["credits_amount"] += event.get("credits_amount") or 0
        buckets[key]["event_ids"].append(event["event_id"])
    return [
        TimelineBucket(
            bucket_start=key[2],
            account_id=key[0],
            event_type=key[1],
            credits_amount=buckets[key]["credits_amount"],
            event_ids=tuple(buckets[key]["event_ids"]),
        )
        for key in order
    ]


def _render_timeline(
    payload: dict,
) -> tuple[tuple[TimelineBucket, ...], TimelineResolution]:
    events = payload.get("timeline", [])
    hourly = _bucket_events(events, "hourly")
    if len(hourly) <= _TIMELINE_BUCKET_BUDGET:
        return tuple(hourly), "hourly"
    return tuple(_bucket_events(events, "daily")), "daily"


def _evidence_keys(payload: dict, view: ReviewCaseView) -> tuple[EvidenceKey, ...]:
    # C-2's allowlist spans all three categories (RC-1 P0-3), not contributions
    # alone, so the contributions-masked view still has valid citations.
    keys = [
        EvidenceKey(kind=EvidenceKeyKind.SCALAR, key="band"),
        EvidenceKey(kind=EvidenceKeyKind.SCALAR, key="allowance"),
        EvidenceKey(kind=EvidenceKeyKind.SCALAR, key="excess"),
    ]
    if view != "timeline_only":
        # Arm 4b withholds contributions from the prompt entirely, so they must
        # also leave the allowlist: a model that was never shown a contribution
        # must not be able to cite one. Leaving them in would make
        # `filter_citations` ratify a fabricated feature-id citation as valid
        # evidence — and one valid citation is all `derive_effect` requires to
        # let a verdict reverse.
        keys.extend(
            EvidenceKey(kind=EvidenceKeyKind.CONTRIBUTION, key=c["feature_id"])
            for c in payload["contributions"]
            if c["fired"]
        )
    keys.extend(
        EvidenceKey(kind=EvidenceKeyKind.TIMELINE_EVENT, key=e["event_id"])
        for e in payload.get("timeline", [])
    )
    return tuple(keys)


def _edge_view(edge: Sequence[object]) -> EdgeView:
    """Project one persisted cluster edge onto the reviewer's 3-field `EdgeView`.

    Snapshot edges are canonical 6-field since Gate G — `(account_a, account_b,
    weight, evidence_count, first_seen, last_seen)` — while pre-Gate-G evidence
    persists the legacy 3-field `(account_a, account_b, evidence_count)`. The
    count therefore sits at a DIFFERENT index in each shape. This mirrors
    `policy_engine.evidence_signature`'s defensive arity branching; the reviewer
    surfaces only endpoints and count either way.
    """
    if len(edge) == 6:
        return EdgeView(
            account_a=str(edge[0]), account_b=str(edge[1]), evidence_count=int(str(edge[3]))
        )
    if len(edge) == 3:
        return EdgeView(
            account_a=str(edge[0]), account_b=str(edge[1]), evidence_count=int(str(edge[2]))
        )
    raise ValueError("edge evidence must contain 3 legacy or 6 canonical fields")


_VIEWS: frozenset[str] = frozenset(("full", "timeline_only"))


def build_review_case(
    session: Session,
    decision: Decision,
    *,
    view: str = "full",
    as_of: datetime,
) -> ReviewCase:
    """Build the reviewer's case for `decision` in one of two evidentiary views
    (`schemas.reviewer.ReviewCaseView`).

    `"full"` (arm 4) shows everything. `"timeline_only"` (arm 4b, L-1) withholds
    the per-feature contributions: `prompt_builder` omits the `contributions:`
    block for the `reviewer_timeline_only_v1` variant, and `_evidence_keys` drops
    the CONTRIBUTION keys so the C-2 allowlist cannot ratify a citation to
    evidence the model never saw. Everything else — the scalars, edges, timeline,
    appeal — is identical between the two views, which is what makes arm 4 vs 4b
    an attribution comparison rather than two unrelated runs.

    An unknown view raises `ReviewCaseError` (this module's error convention).
    The `NotImplementedError` that used to stand here was a deliberate
    not-built-yet marker; both views now exist, so a bad view is an ordinary
    caller error."""
    if view not in _VIEWS:
        raise ReviewCaseError(
            f"unknown review_case view={view!r}; expected one of {sorted(_VIEWS)}"
        )
    # The guard above is the runtime narrowing mypy cannot infer from a frozenset
    # membership test; `view` is a plain `str` because callers thread it through
    # from request/config data.
    resolved_view = cast(ReviewCaseView, view)

    payload = decision_evidence(session, decision.decision_id)
    if payload is None:
        raise ReviewCaseError(
            f"no immutable evidence recorded for decision {decision.decision_id!r}"
        )

    timeline, resolution = _render_timeline(payload)
    appeal = _latest_appeal(session, decision, as_of)

    return ReviewCase(
        decision_id=decision.decision_id,
        account_id=decision.account_id,
        # Pass the REQUESTED view through. This was hardcoded to `"full"`, which
        # was harmless only while `"full"` was the sole accepted value — it would
        # now mislabel every arm-4b case (and `_format_structured_evidence`
        # renders `view:` straight into the prompt).
        view=resolved_view,
        snapshot_hash=payload["snapshot_hash"],
        member_account_ids=tuple(payload["member_account_ids"]),
        edges=tuple(_edge_view(edge) for edge in payload["edges"]),
        band=_band_for(payload["qualified_stream_count"]),
        allowance=payload["allowance"],
        excess=payload["excess"],
        cap_reasons=tuple(payload["cap_reasons"]),
        contributions=tuple(
            ContributionView(
                feature_id=c["feature_id"],
                raw_value=c["raw_value"],
                contribution=c["contribution"],
                fired=c["fired"],
                family=c["family"],
                rung=c["rung"],
            )
            for c in payload["contributions"]
        ),
        timeline=timeline,
        applied_timeline_resolution=resolution,
        evidence_keys=_evidence_keys(payload, resolved_view),
        appeal_text=appeal.text if appeal is not None else None,
        appeal_truncated=appeal.truncated if appeal is not None else False,
    )

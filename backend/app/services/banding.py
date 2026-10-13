"""Banded capacity — timing-based stream estimation *within* an already-formed
cluster (docs/policy.md §4). Step 4.

This module NEVER links accounts: clustering (`clustering.py`, resource-sharing
signals only) forms the cluster upstream; banding subdivides that cluster into
the number of plausibly-independent streams it supports, and turns that count
into an `allowance`. Every parameter is evaluated `as_of` a decision time over
the trailing `WINDOW_DAYS` window (leakage control, eval §9).

All literal numbers are policy §4 and are a **tuning seed** frozen at
`features-frozen` (features.md §6); they are named constants here so the manifest
(Step 8) can capture them.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.enums import CapacityBand, EventType
from app.models.inference import AccountEvent

# --- Policy §4 constants (tuning seed, frozen at features-frozen) --------------

WINDOW_DAYS = 60  # trailing evaluation window (policy §4.1)

# Axis 1 — concurrency (policy §4.2)
N_OVERLAP = 3  # ≥ 3 distinct shared-consumption days to pass

# Axis 2 — rate plausibility (policy §4.3): an account FAILS if either holds
RATE_HOURLY_CREDITS = 8  # > R credits in an hour …
RATE_HOURLY_MIN_HOURS = 2  # … on ≥ 2 distinct hours, or
RATE_DAILY_CREDITS = 20  # > S credits in a day …
RATE_DAILY_MIN_DAYS = 2  # … on ≥ 2 distinct days

# Axis 3 — temporal diversity (policy §4.4)
DIURNAL_D = 0.15  # single-linkage cosine-distance threshold
DIURNAL_D_LOW = 0.12  # ambiguity sensitivity band (policy §4.6)
DIURNAL_D_HIGH = 0.18

PER_STREAM_ALLOWANCE = 25  # a full cycle allocation (policy §4.1)

AMBIGUITY_FRACTION = 0.20  # claimed within 20% of allowance → ambiguous (§4.6)

# Band cut points over qualified_stream_count (policy §4.5)
_SMALL_GROUP_MIN = 2
_LARGE_SHARED_MIN = 6


# --- Result ------------------------------------------------------------------


@dataclass(frozen=True)
class BandingResult:
    """Capacity verdict for one `(cluster, as_of)` (policy §4.5)."""

    qualified_stream_count: int
    allowance: int
    band: CapacityBand
    ambiguous: bool
    claimed: int  # trailing-window CLAIM_CREDITS across the cluster (§4.6, F-CYC-1)
    concurrency_passers: tuple[str, ...]
    rate_passers: tuple[str, ...]


# --- Helpers -----------------------------------------------------------------


def _utc(dt: datetime) -> datetime:
    """Canonicalise to UTC (SQLite drops tz on read). Mirrors clustering._utc so
    equal instants bucket identically regardless of stored offset."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _consume_events(
    session: Session, member_ids: list[str], as_of: datetime
) -> dict[str, list[tuple[datetime, int]]]:
    """Per-account CONSUME_CREDITS `(timestamp, amount)` within the trailing window,
    `as_of`-bounded (leakage control). Amounts default to 0 when unset."""
    window_start = as_of - timedelta(days=WINDOW_DAYS)
    rows = session.scalars(
        select(AccountEvent).where(
            AccountEvent.account_id.in_(member_ids),
            AccountEvent.event_type == EventType.CONSUME_CREDITS,
            AccountEvent.timestamp <= as_of,
        )
    ).all()
    out: dict[str, list[tuple[datetime, int]]] = defaultdict(list)
    for r in rows:
        ts = _utc(r.timestamp)
        if ts < window_start:
            continue  # outside trailing window
        out[r.account_id].append((ts, r.credits_amount or 0))
    return out


def _claimed_in_window(
    session: Session, member_ids: list[str], as_of: datetime
) -> int:
    """Total CLAIM_CREDITS granted across the cluster in the trailing window."""
    window_start = as_of - timedelta(days=WINDOW_DAYS)
    rows = session.scalars(
        select(AccountEvent).where(
            AccountEvent.account_id.in_(member_ids),
            AccountEvent.event_type == EventType.CLAIM_CREDITS,
            AccountEvent.timestamp <= as_of,
        )
    ).all()
    return sum(
        (r.credits_amount or 0) for r in rows if _utc(r.timestamp) >= window_start
    )


# --- Axis 1: concurrency (policy §4.2) ---------------------------------------


def _passes_concurrency(
    account_id: str, consume: dict[str, list[tuple[datetime, int]]]
) -> bool:
    """Pass iff the account consumed on ≥ N_OVERLAP distinct calendar days that
    also carry consumption from at least one *other* cluster account."""
    own_days = {ts.date() for ts, _ in consume.get(account_id, [])}
    if len(own_days) < N_OVERLAP:
        return False
    other_days: set = set()
    for other, evs in consume.items():
        if other == account_id:
            continue
        other_days.update(ts.date() for ts, _ in evs)
    shared = own_days & other_days
    return len(shared) >= N_OVERLAP


# --- Axis 2: rate plausibility (policy §4.3) ---------------------------------


def _passes_rate(
    account_id: str, consume: dict[str, list[tuple[datetime, int]]]
) -> bool:
    """Fail if hourly consumption exceeds R on ≥ 2 hours, or daily exceeds S on
    ≥ 2 days. The ≥ 2-occasion rule protects a single legitimate burst (§4.3)."""
    hourly: dict[tuple, int] = defaultdict(int)
    daily: dict = defaultdict(int)
    for ts, amt in consume.get(account_id, []):
        hourly[(ts.date(), ts.hour)] += amt
        daily[ts.date()] += amt
    hot_hours = sum(1 for v in hourly.values() if v > RATE_HOURLY_CREDITS)
    hot_days = sum(1 for v in daily.values() if v > RATE_DAILY_CREDITS)
    if hot_hours >= RATE_HOURLY_MIN_HOURS:
        return False
    if hot_days >= RATE_DAILY_MIN_DAYS:
        return False
    return True


# --- Axis 3: temporal diversity (policy §4.4) --------------------------------


def _diurnal_histogram(evs: list[tuple[datetime, int]]) -> list[float]:
    """L1-normalised hour-of-day (24) × day-of-week (7) = 168-bin activity vector."""
    hist = [0.0] * (24 * 7)
    for ts, _ in evs:
        hist[ts.weekday() * 24 + ts.hour] += 1.0
    total = sum(hist)
    if total == 0.0:
        return hist
    return [v / total for v in hist]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - dot / (na * nb)


def _single_linkage_groups(
    candidates: list[str],
    histograms: dict[str, list[float]],
    threshold: float,
) -> int:
    """Number of single-linkage groups: merge any two candidates whose diurnal
    cosine distance ≤ threshold; each connected group contributes one stream."""
    if not candidates:
        return 0
    parent = {c: c for c in candidates}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            ci, cj = candidates[i], candidates[j]
            if _cosine_distance(histograms[ci], histograms[cj]) <= threshold:
                union(ci, cj)
    return len({find(c) for c in candidates})


# --- Public entry point ------------------------------------------------------


def _band_for(count: int) -> CapacityBand:
    if count >= _LARGE_SHARED_MIN:
        return CapacityBand.LARGE_SHARED
    if count >= _SMALL_GROUP_MIN:
        return CapacityBand.SMALL_GROUP
    return CapacityBand.SINGLE


def compute_banding(
    session: Session, member_account_ids: list[str], as_of: datetime
) -> BandingResult:
    """Estimate the cluster's independent-stream capacity and allowance `as_of`.

    An account qualifies as a stream only by passing all three axes; diurnal
    single-linkage then collapses look-alike schedules so one operator running N
    near-identical accounts counts once (policy §4.1/§4.4)."""
    members = sorted(set(member_account_ids))
    consume = _consume_events(session, members, as_of)

    candidates = [
        a
        for a in members
        if _passes_concurrency(a, consume) and _passes_rate(a, consume)
    ]
    concurrency_passers = tuple(a for a in members if _passes_concurrency(a, consume))
    rate_passers = tuple(a for a in members if _passes_rate(a, consume))

    histograms = {c: _diurnal_histogram(consume.get(c, [])) for c in candidates}
    count = _single_linkage_groups(candidates, histograms, DIURNAL_D)

    allowance = PER_STREAM_ALLOWANCE * max(1, count)
    claimed = _claimed_in_window(session, members, as_of)

    # Ambiguity (policy §4.6): near the allowance boundary, OR count unstable
    # across the diurnal-threshold sensitivity band.
    near_boundary = abs(claimed - allowance) <= AMBIGUITY_FRACTION * allowance
    count_low = _single_linkage_groups(candidates, histograms, DIURNAL_D_LOW)
    count_high = _single_linkage_groups(candidates, histograms, DIURNAL_D_HIGH)
    unstable = not (count_low == count == count_high)
    ambiguous = near_boundary or unstable

    return BandingResult(
        qualified_stream_count=count,
        allowance=allowance,
        band=_band_for(count),
        ambiguous=ambiguous,
        claimed=claimed,
        concurrency_passers=concurrency_passers,
        rate_passers=rate_passers,
    )

"""Banded capacity axes (docs/policy.md §4, plan Step 4 green gate).

Covers: T1.1 sequential cycling → concurrency 0 → allowance 25; T1.2 full-speed
→ fails rate; T1.2 throttled six accounts collapse to one diurnal group →
allowance 25; the band cut points; and ambiguity on both triggers.
"""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy.orm import Session

from app.models.enums import CapacityBand
from app.services.banding import PER_STREAM_ALLOWANCE, compute_banding
from app.tests.support import AS_OF, Builder

# Three dates one week apart — same weekday, so the diurnal (weekday×hour)
# histogram is controllable while concurrency still sees ≥ 3 distinct days.
_D1 = AS_OF - timedelta(days=21)
_D2 = AS_OF - timedelta(days=14)
_D3 = AS_OF - timedelta(days=7)
_SHARED_DAYS = (_D1, _D2, _D3)


def _at(day, hour: int):
    return day.replace(hour=hour, minute=0, second=0, microsecond=0)


def test_sequential_cycling_concurrency_zero_allowance_25(inference_session: Session) -> None:
    """T1.1: accounts tile the timeline with no shared days → nobody passes
    concurrency → count 0 → allowance 25."""
    b = Builder(inference_session)
    members = [b.account(f"seq-{i}") for i in range(4)]
    # Each account consumes on its own distinct, non-overlapping days.
    for i, aid in enumerate(members):
        for k in range(3):
            day = AS_OF - timedelta(days=1 + i * 5 + k)
            b.consume(aid, _at(day, 12), 3)
    b.flush()

    result = compute_banding(inference_session, members, AS_OF)
    assert result.qualified_stream_count == 0
    assert result.allowance == PER_STREAM_ALLOWANCE
    assert result.concurrency_passers == ()


def test_full_speed_fails_rate(inference_session: Session) -> None:
    """T1.2: an account burning > 8 credits/hour on ≥ 2 hours fails the rate axis."""
    b = Builder(inference_session)
    fast = b.account("fast")
    others = [b.account(f"slow-{i}") for i in range(2)]
    # `fast` shares the 3 days (would pass concurrency) but consumes 10 credits in a
    # single hour on two of them → fails rate.
    for day in _SHARED_DAYS:
        b.consume(fast, _at(day, 10), 10)
    for aid in others:
        for day in _SHARED_DAYS:
            b.consume(aid, _at(day, 9), 3)
    b.flush()

    result = compute_banding(inference_session, [fast, *others], AS_OF)
    assert fast not in result.rate_passers
    assert fast in result.concurrency_passers  # it DID share days — only rate excludes it


def test_throttled_six_collapse_to_one_group(inference_session: Session) -> None:
    """T1.2 evasion: six rate-throttled accounts with one operator's identical
    diurnal signature collapse to a single stream → allowance 25."""
    b = Builder(inference_session)
    members = [b.account(f"farm-{i}") for i in range(6)]
    for aid in members:
        for day in _SHARED_DAYS:
            b.consume(aid, _at(day, 14), 3)  # identical (weekday, hour) → one histogram
    b.flush()

    result = compute_banding(inference_session, members, AS_OF)
    assert result.qualified_stream_count == 1
    assert result.allowance == PER_STREAM_ALLOWANCE
    assert result.band == CapacityBand.SINGLE


def test_diverse_schedules_each_a_stream(inference_session: Session) -> None:
    """Distinct diurnal signatures on shared days → one stream each → 25 × count."""
    b = Builder(inference_session)
    members = [b.account(f"real-{i}") for i in range(3)]
    for i, aid in enumerate(members):
        for day in _SHARED_DAYS:
            b.consume(aid, _at(day, 8 + i * 5), 3)  # 8, 13, 18 → orthogonal histograms
    b.flush()

    result = compute_banding(inference_session, members, AS_OF)
    assert result.qualified_stream_count == 3
    assert result.allowance == 3 * PER_STREAM_ALLOWANCE
    assert result.band == CapacityBand.SMALL_GROUP


def test_large_shared_band(inference_session: Session) -> None:
    b = Builder(inference_session)
    members = [b.account(f"big-{i}") for i in range(6)]
    for i, aid in enumerate(members):
        for day in _SHARED_DAYS:
            b.consume(aid, _at(day, 2 + i * 3), 3)  # 2,5,8,11,14,17 → 6 distinct groups
    b.flush()

    result = compute_banding(inference_session, members, AS_OF)
    assert result.qualified_stream_count == 6
    assert result.allowance == 6 * PER_STREAM_ALLOWANCE
    assert result.band == CapacityBand.LARGE_SHARED


def test_ambiguity_near_allowance_boundary(inference_session: Session) -> None:
    """Trigger 1: claimed within 20% of allowance."""
    b = Builder(inference_session)
    solo = b.account("solo")  # no qualifying streams → allowance 25
    b.claim(solo, AS_OF - timedelta(days=10), 25)  # claimed == allowance
    b.flush()

    result = compute_banding(inference_session, [solo], AS_OF)
    assert result.allowance == 25
    assert result.claimed == 25
    assert result.ambiguous is True


def test_ambiguity_count_flips_over_d(inference_session: Session) -> None:
    """Trigger 2: qualified_stream_count changes across D ∈ [0.12, 0.18]. Two
    accounts with cosine distance ≈ 0.138 merge at 0.15/0.18 but not at 0.12."""
    b = Builder(inference_session)
    a, c = b.account("amb-a"), b.account("amb-c")
    # Histograms: A = 5·h14 + 2·h9, C = 5·h14 + 2·h20 → cos 25/29 ≈ 0.862, dist 0.138.
    plan = {
        a: [(14, 2), (9, 2)],  # D1
        c: [(14, 2), (20, 2)],
    }
    for aid, d1 in plan.items():
        for hour, count in d1:
            for _ in range(count):
                b.consume(aid, _at(_D1, hour), 1)
        b.consume(aid, _at(_D2, 14), 1)
        b.consume(aid, _at(_D2, 14), 1)
        b.consume(aid, _at(_D3, 14), 1)
    b.flush()

    result = compute_banding(inference_session, [a, c], AS_OF)
    # No claims → claimed 0, far from allowance → the boundary trigger is silent,
    # so ambiguity here is purely the D-sensitivity flip.
    assert result.claimed == 0
    assert result.ambiguous is True

"""Frozen feature set (docs/features.md §3) + rung masking (§5). Step 4.

Every feature emits a signed **contribution** on the additive risk scale and
carries two orthogonal labels (features.md §2):
  * an **evidence family** (1 Device · 2 Payment · 3 Contact/Domain · 4 Cycle ·
    5 Verification · None for benign/context) — for the ≥ 2-family action gate;
  * an **ablation rung** (S0 metadata · S1 +resource-sharing · S2 +timing) — for
    the S0→S1→S2 lift ladder.

A feature **fires** iff `|contribution| ≥ FIRING_THRESHOLD` (0.05, features.md
§4.1). Every feature reads only events at or before `as_of` (leakage control,
eval §9) and never touches the ground-truth store.

**The fixed evaluation unit (E-14, features.md §5).** Clustering runs once
upstream and is rung-independent, so every rung scores the *same*
`(cluster_snapshot_id, cycle_id, as_of)` cluster-cycle rows. `cycle_id` is
synthesised from the aligned cluster cycle window (the schema's `cycle_id` is
per-account; a cluster-cycle is the window shared across members) so the keys are
byte-identical across S0/S1/S2 — the paired-ablation requirement (review 4 P0-1).
Rung masking zeroes `rung > target`; it never changes the unit. S0 additionally
scores the account-local metadata features per account and reduces to the
cluster-cycle by the pre-registered **"any member account would be proposed for
restriction"** operator.

The weights below are the **frozen v1 magnitudes** (features.md §3, §6). Step 8
hand-tuned them on the tuning split, then `features-frozen` locked the full
firing-affecting configuration — id/inputs/sign/family/rung/predicate/params AND
the weights — via the hashed `config/feature-manifest.json` (see
`app/services/feature_manifest.py`). Nothing firing-affecting moves after the tag
in v1.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import IntEnum

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.enums import (
    CapacityBand,
    DecisionStatus,
    EmailDomainType,
    EventType,
    Reliability,
    SignalType,
)
from app.models.inference import (
    Account,
    AccountEvent,
    ClusterSnapshot,
    Decision,
    EntitlementCycle,
)
from app.services import banding as banding_mod
from app.services.banding import PER_STREAM_ALLOWANCE, BandingResult, compute_banding
from app.services.clustering import (
    HUB_ACCOUNT_THRESHOLD,
    _Sig,
    shared_signal_values,
)

# --- Taggings ----------------------------------------------------------------


class Rung(IntEnum):
    """Ablation tier (eval §2.1). Ordered so `rung > target` is the mask test."""

    S0 = 0  # metadata only
    S1 = 1  # + resource-sharing edge features
    S2 = 2  # + timing capacity axes (== evidence family 4)


class EvidenceFamily(IntEnum):
    """The ≥ 2-family action-gate families (policy §6, features.md §2)."""

    DEVICE = 1
    PAYMENT = 2
    CONTACT_DOMAIN = 3
    CYCLE = 4
    VERIFICATION = 5


FIRING_THRESHOLD = 0.05  # |contribution| ≥ this → the feature fires (features.md §4.1)

# The rung-masking + S0 any-restriction-aggregation version (features.md §5, §6.1).
# Frozen in the manifest: a change to how masking or the S0 reduction works bumps
# this and thus the manifest hash, even if no weight moves.
MASKING_AGGREGATION_VERSION = "v1"

# Frozen v1 weights (features.md §3.1/§3.2, §6.1). These were HAND-TUNED at Step 8
# on the tuning split only (eval §9): F-DEV-1 0.30→0.45, F-CON-1 0.15→0.25, and
# F-DOM-1 0.15→0.25 — three clean discriminators (0 false fires on tune-split
# negatives) whose seed magnitudes left genuine positives just under the 0.60
# score threshold. The lift recovered every missed positive with zero FP cost and
# generalised across four generator seeds (tune recall ≈0.5→1.0, precision 1.0,
# FPR 0.0). All other weights were confirmed at frontier and left unchanged.
# These magnitudes are frozen at the `features-frozen` tag via
# config/feature-manifest.json; nothing firing-affecting moves after it in v1.
_W_DEV_1 = 0.45
_W_DEV_2 = 0.15
_W_PAY_1 = 0.20
_W_CON_1 = 0.25
_W_DOM_1 = 0.25
_W_VER_1 = 0.15
_W_CYC_1 = 0.30  # cap; contribution scales with allowance excess
_W_CYC_2 = 0.15
_W_CYC_3 = 0.10
_W_CYC_4 = 0.10
_W_DOM_2 = -0.15
_W_AGE_1 = -0.15

#: Frozen v1 weights keyed by feature id — the tuned vector, mirrored in
#: config/thresholds.yaml and config/feature-manifest.json (asserted equal by test
#: so the three never drift). Named SEED_WEIGHTS historically; post-Step-8 it holds
#: the tuned, frozen magnitudes.
SEED_WEIGHTS: dict[str, float] = {
    "F-DEV-1": _W_DEV_1,
    "F-DEV-2": _W_DEV_2,
    "F-PAY-1": _W_PAY_1,
    "F-CON-1": _W_CON_1,
    "F-DOM-1": _W_DOM_1,
    "F-VER-1": _W_VER_1,
    "F-CYC-1": _W_CYC_1,
    "F-CYC-2": _W_CYC_2,
    "F-CYC-3": _W_CYC_3,
    "F-CYC-4": _W_CYC_4,
    "F-DOM-2": _W_DOM_2,
    "F-AGE-1": _W_AGE_1,
}

_VER_FAILURE_MIN = 2  # F-VER-1 "repeated" failures (tuning-split threshold)
_AGE_MIN_DAYS = 90  # F-AGE-1 long-lived accounts (features.md §3.2)
_ROUTINE_MIN_LOGINS = 2  # F-DOM-2/F-AGE-1 "routine activity" floor (clarification 3)
_CYC3_LATENCY_HOURS = 72  # F-CYC-3 exhaustion→signup latency (features.md §3.1)
_CYC3_RECENCY_DAYS = 60  # Gate G: only recent exhaustion/signup evidence is actionable

# F-DEV-1/F-DEV-2 shared-device predicate parameters (features.md §3.1). Named so
# the freeze manifest references the actual firing logic, not prose: a change here
# moves the manifest hash. "Strong": ≥3 members at HIGH; "weak": exactly 2 members
# OR MEDIUM reliability.
_DEV_STRONG_MIN_MEMBERS = 3
_DEV_STRONG_RELIABILITY = Reliability.HIGH
_DEV_WEAK_MEMBERS = 2
_DEV_WEAK_RELIABILITY = Reliability.MEDIUM
_BENIGN_MAJORITY = 0.5  # F-DOM-2/F-AGE-1 fire when ≥ half the cluster qualifies

_RANK = {Reliability.LOW: 0, Reliability.MEDIUM: 1, Reliability.HIGH: 2}
_BY_RANK = {0: Reliability.LOW, 1: Reliability.MEDIUM, 2: Reliability.HIGH}


# --- Feature contribution ----------------------------------------------------


@dataclass(frozen=True)
class FeatureContribution:
    feature_id: str
    raw_value: float  # the underlying measured quantity (pre-weight)
    contribution: float  # signed, weighted
    family: EvidenceFamily | None  # None → benign/context (never counts for the gate)
    rung: Rung
    fired: bool

    @staticmethod
    def make(
        feature_id: str,
        raw_value: float,
        contribution: float,
        family: EvidenceFamily | None,
        rung: Rung,
    ) -> FeatureContribution:
        return FeatureContribution(
            feature_id=feature_id,
            raw_value=raw_value,
            contribution=contribution,
            family=family,
            rung=rung,
            fired=abs(contribution) >= FIRING_THRESHOLD,
        )

    def masked(self) -> FeatureContribution:
        """Zeroed copy — same id/family/rung, no contribution AND no raw value: rung
        masking is an information boundary, so a masked feature exports nothing of its
        (disabled) measurement (review finding 13)."""
        return FeatureContribution(
            feature_id=self.feature_id,
            raw_value=0.0,
            contribution=0.0,
            family=self.family,
            rung=self.rung,
            fired=False,
        )


# The canonical feature order — identical across rungs so the emitted vectors are
# comparable position-for-position.
FEATURE_ORDER: tuple[str, ...] = (
    "F-DEV-1",
    "F-DEV-2",
    "F-PAY-1",
    "F-CON-1",
    "F-DOM-1",
    "F-VER-1",
    "F-CYC-1",
    "F-CYC-2",
    "F-CYC-3",
    "F-CYC-4",
    "F-DOM-2",
    "F-AGE-1",
)


# --- Scores ------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterCycleUnit:
    """One fixed evaluation unit (E-14). `cycle_id` identifies the aligned cluster
    cycle window; it is rung-independent, so all three rungs share these keys."""

    cluster_snapshot_id: str
    cycle_id: str
    as_of: datetime
    window_start: datetime
    window_end: datetime


@dataclass(frozen=True)
class ClusterScore:
    contributions: tuple[FeatureContribution, ...]  # in FEATURE_ORDER
    banding: BandingResult
    rung: Rung

    @property
    def total_contribution(self) -> float:
        """Sum of fired contributions, clipped to [0, 1] (features.md §4.3)."""
        s = sum(c.contribution for c in self.contributions if c.fired)
        return max(0.0, min(1.0, s))

    @property
    def firing_families(self) -> frozenset[EvidenceFamily]:
        """Incriminating families with a fired feature (≥ 2-family gate, §4.2)."""
        return frozenset(
            c.family
            for c in self.contributions
            if c.fired and c.contribution > 0 and c.family is not None
        )


# --- Loading (as_of-bounded) -------------------------------------------------


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@dataclass
class _Ctx:
    """As-of-bounded inference data for one cluster."""

    member_ids: list[str]
    accounts: dict[str, Account]
    events: dict[str, list[AccountEvent]]  # per account, ts <= as_of, ts asc
    cycles: dict[str, list[EntitlementCycle]]  # per account, cycle_start <= as_of
    # GLOBAL resource-sharing map from clustering — per value, every present account
    # carrying it (max reliability per account). Global so hub cardinality matches
    # clustering (review finding 4).
    value_accounts: dict[tuple[SignalType, str], dict[str, _Sig]]
    as_of: datetime


def _load_ctx(session: Session, member_ids: list[str], as_of: datetime) -> _Ctx:
    members = sorted(set(member_ids))
    # Account metadata is `as_of`-bounded too: a not-yet-created account must not
    # leak its domain/verification into a historical score (review finding 3).
    accounts = {
        a.account_id: a
        for a in session.scalars(
            select(Account).where(
                Account.account_id.in_(members), Account.created_at <= as_of
            )
        ).all()
    }

    events: dict[str, list[AccountEvent]] = defaultdict(list)
    for e in session.scalars(
        select(AccountEvent)
        .where(AccountEvent.account_id.in_(members), AccountEvent.timestamp <= as_of)
        .order_by(AccountEvent.timestamp)
    ).all():
        events[e.account_id].append(e)

    cycles: dict[str, list[EntitlementCycle]] = defaultdict(list)
    for c in session.scalars(
        select(EntitlementCycle).where(
            EntitlementCycle.account_id.in_(members),
            EntitlementCycle.cycle_start <= as_of,
        )
    ).all():
        cycles[c.account_id].append(c)

    # Reuse clustering's canonical global value map (hub-correct, as_of-bounded).
    value_accounts = shared_signal_values(session, as_of)
    return _Ctx(members, accounts, dict(events), dict(cycles), value_accounts, as_of)


# --- Per-value view for resource-sharing features ----------------------------


def _values_of(ctx: _Ctx, signal: SignalType) -> list[tuple[int, Reliability]]:
    """For each `signal` value shared by ≥ 2 of THIS cluster's members,
    `(member_sharer_count, effective_reliability)`, exactly as clustering scores it
    (review finding 4): hub cardinality is **global** (the value on
    > HUB_ACCOUNT_THRESHOLD accounts anywhere → forced `low`), and the effective
    reliability is the **weaker** (min) among the sharing members, not the max."""
    members = set(ctx.member_ids)
    out: list[tuple[int, Reliability]] = []
    for (sig, _val), accs in ctx.value_accounts.items():
        if sig is not signal:
            continue
        sharers = [m for m in accs if m in members]
        n = len(sharers)
        if n < 2:
            continue
        if len(accs) > HUB_ACCOUNT_THRESHOLD:  # global hub → downgraded to low
            rel = Reliability.LOW
        else:
            rel = _BY_RANK[min(_RANK[accs[m].rel] for m in sharers)]  # weaker/effective
        out.append((n, rel))
    return out


# --- Cluster-level incriminating features ------------------------------------


def _f_dev_1(ctx: _Ctx) -> FeatureContribution:
    strong = any(
        n >= _DEV_STRONG_MIN_MEMBERS and rel is _DEV_STRONG_RELIABILITY
        for n, rel in _values_of(ctx, SignalType.STABLE_DEVICE)
    )
    return FeatureContribution.make(
        "F-DEV-1", 1.0 if strong else 0.0, _W_DEV_1 if strong else 0.0,
        EvidenceFamily.DEVICE, Rung.S1,
    )


def _f_dev_2(ctx: _Ctx) -> FeatureContribution:
    weak = any(
        n == _DEV_WEAK_MEMBERS or rel is _DEV_WEAK_RELIABILITY
        for n, rel in _values_of(ctx, SignalType.STABLE_DEVICE)
    )
    return FeatureContribution.make(
        "F-DEV-2", 1.0 if weak else 0.0, _W_DEV_2 if weak else 0.0,
        EvidenceFamily.DEVICE, Rung.S1,
    )


def _f_pay_1(ctx: _Ctx) -> FeatureContribution:
    hit = bool(_values_of(ctx, SignalType.PAYMENT_TOKEN))
    return FeatureContribution.make(
        "F-PAY-1", 1.0 if hit else 0.0, _W_PAY_1 if hit else 0.0,
        EvidenceFamily.PAYMENT, Rung.S1,
    )


def _f_con_1(ctx: _Ctx) -> FeatureContribution:
    hit = bool(_values_of(ctx, SignalType.RECOVERY_CONTACT))
    return FeatureContribution.make(
        "F-CON-1", 1.0 if hit else 0.0, _W_CON_1 if hit else 0.0,
        EvidenceFamily.CONTACT_DOMAIN, Rung.S1,
    )


# --- Account-local metadata features (features.md §3.1/§5) -------------------


def _account_disposable(account: Account) -> bool:
    return account.email_domain_type is EmailDomainType.DISPOSABLE


def _account_weak_verification(events: list[AccountEvent]) -> bool:
    """Repeated verification failures, or claim_credits before any verification."""
    failures = sum(1 for e in events if e.event_type is EventType.VERIFICATION_FAILED)
    if failures >= _VER_FAILURE_MIN:
        return True
    first_claim = next(
        (e.timestamp for e in events if e.event_type is EventType.CLAIM_CREDITS), None
    )
    if first_claim is None:
        return False
    first_verify = next(
        (
            e.timestamp
            for e in events
            if e.event_type in (EventType.VERIFICATION, EventType.VERIFICATION_FAILED)
        ),
        None,
    )
    return first_verify is None or _utc(first_claim) < _utc(first_verify)


def _routine_activity(events: list[AccountEvent]) -> bool:
    """Minimum evidence of normal non-credit use for the benign features (design
    clarification 3): at least `_ROUTINE_MIN_LOGINS` LOGIN events, not a single one."""
    return sum(1 for e in events if e.event_type is EventType.LOGIN) >= _ROUTINE_MIN_LOGINS


def _account_edu_enterprise(account: Account, events: list[AccountEvent]) -> bool:
    """Education/enterprise domain with routine non-credit activity (§3.2)."""
    if account.email_domain_type not in (
        EmailDomainType.EDUCATION,
        EmailDomainType.ENTERPRISE,
    ):
        return False
    return _routine_activity(events)


def _account_aged(
    account: Account, events: list[AccountEvent], reference: datetime
) -> bool:
    """Long-lived account (created ≥ 90 d before the flagged cycle boundary
    `reference`, not the detector time — review finding 12) with routine use."""
    if (reference - _utc(account.created_at)).days < _AGE_MIN_DAYS:
        return False
    return _routine_activity(events)


def _f_dom_1_cluster(ctx: _Ctx) -> FeatureContribution:
    n = sum(1 for a in ctx.accounts.values() if _account_disposable(a))
    hit = n >= 1
    return FeatureContribution.make(
        "F-DOM-1", float(n), _W_DOM_1 if hit else 0.0, EvidenceFamily.CONTACT_DOMAIN, Rung.S0
    )


def _f_ver_1_cluster(ctx: _Ctx) -> FeatureContribution:
    n = sum(
        1 for aid in ctx.member_ids if _account_weak_verification(ctx.events.get(aid, []))
    )
    hit = n >= 1
    return FeatureContribution.make(
        "F-VER-1", float(n), _W_VER_1 if hit else 0.0, EvidenceFamily.VERIFICATION, Rung.S0
    )


def _f_dom_2_cluster(ctx: _Ctx) -> FeatureContribution:
    size = max(1, len(ctx.member_ids))
    n = sum(
        1
        for aid, a in ctx.accounts.items()
        if _account_edu_enterprise(a, ctx.events.get(aid, []))
    )
    hit = n / size >= _BENIGN_MAJORITY
    return FeatureContribution.make(
        "F-DOM-2", n / size, _W_DOM_2 if hit else 0.0, None, Rung.S0
    )


def _f_age_1_cluster(ctx: _Ctx, age_reference: datetime) -> FeatureContribution:
    size = max(1, len(ctx.member_ids))
    n = sum(
        1
        for aid, a in ctx.accounts.items()
        if _account_aged(a, ctx.events.get(aid, []), age_reference)
    )
    hit = n / size >= _BENIGN_MAJORITY
    return FeatureContribution.make(
        "F-AGE-1", n / size, _W_AGE_1 if hit else 0.0, None, Rung.S0
    )


# --- S2 cycle-behaviour features (family 4) ----------------------------------


def _consume_days(ctx: _Ctx) -> dict[str, set]:
    window_start = ctx.as_of - timedelta(days=banding_mod.WINDOW_DAYS)
    days: dict[str, set] = defaultdict(set)
    for aid, evs in ctx.events.items():
        for e in evs:
            if e.event_type is not EventType.CONSUME_CREDITS:
                continue
            ts = _utc(e.timestamp)
            if ts >= window_start:
                days[aid].add(ts.date())
    return days


def _f_cyc_1(
    ctx: _Ctx, band: BandingResult, cycle_claimed: int
) -> FeatureContribution:
    """Allowance excess: claimed / allowance − 1, scaled and capped at the weight."""
    allowance = band.allowance
    excess = max(0.0, cycle_claimed / allowance - 1.0) if allowance else 0.0
    contribution = _W_CYC_1 * min(1.0, excess)
    return FeatureContribution.make(
        "F-CYC-1", excess, contribution, EvidenceFamily.CYCLE, Rung.S2
    )


def _f_cyc_2(ctx: _Ctx) -> FeatureContribution:
    """Sequential tiling: ≥ 2 accounts consume, but no consumption day is shared —
    active windows tile the timeline (the pattern axis-1 concurrency defeats)."""
    days = _consume_days(ctx)
    active = [aid for aid, d in days.items() if d]
    tiling = False
    if len(active) >= 2:
        shared = 0
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                if days[active[i]] & days[active[j]]:
                    shared += 1
        tiling = shared == 0
    return FeatureContribution.make(
        "F-CYC-2", 1.0 if tiling else 0.0, _W_CYC_2 if tiling else 0.0,
        EvidenceFamily.CYCLE, Rung.S2,
    )


def _exhaustion_times(ctx: _Ctx) -> list[datetime]:
    """When each fully-consumed cycle reached its grant (the consume event that
    brought cumulative consumption to ≥ credits_granted)."""
    out: list[datetime] = []
    for aid, cyc_list in ctx.cycles.items():
        consume = sorted(
            (
                (_utc(e.timestamp), e.credits_amount or 0)
                for e in ctx.events.get(aid, [])
                if e.event_type is EventType.CONSUME_CREDITS
            ),
            key=lambda x: x[0],
        )
        for cyc in cyc_list:
            start, end = _utc(cyc.cycle_start), _utc(cyc.cycle_end)
            cum = 0
            hit: datetime | None = None
            for ts, amt in consume:
                if start <= ts < end and ts <= ctx.as_of:
                    cum += amt
                    if cum >= cyc.credits_granted and cyc.credits_granted > 0:
                        hit = ts
                        break
            if hit is not None:
                out.append(hit)
    return out


def _f_cyc_3(ctx: _Ctx) -> FeatureContribution:
    """Recent linked exhaustion followed by signup within the 72-hour latency."""
    lower = ctx.as_of - timedelta(days=_CYC3_RECENCY_DAYS)
    exhausted = [ts for ts in _exhaustion_times(ctx) if lower <= ts <= ctx.as_of]
    signups = [
        _utc(e.timestamp)
        for evs in ctx.events.values()
        for e in evs
        if e.event_type is EventType.SIGNUP and lower <= _utc(e.timestamp) <= ctx.as_of
    ]
    hit = any(
        0 <= (su - ex).total_seconds() <= _CYC3_LATENCY_HOURS * 3600
        for ex in exhausted
        for su in signups
    )
    return FeatureContribution.make(
        "F-CYC-3", 1.0 if hit else 0.0, _W_CYC_3 if hit else 0.0,
        EvidenceFamily.CYCLE, Rung.S2,
    )


def _f_cyc_4(session: Session, members: list[str], as_of: datetime) -> FeatureContribution:
    """This cluster's identical pattern was **upheld** in a prior cycle (policy §10).

    Matches on the as_of-independent pattern — the member set — not the snapshot id
    (snapshot ids embed `as_of`, so a next-cycle snapshot never matches the old one,
    review finding 5). Only APPLIED (upheld) prior decisions count; reversed ones —
    findings of legitimacy — never do (policy §10). No upheld decisions exist on the
    dev world, so it stays dormant there."""
    target = sorted(members)
    prior_decisions = session.scalars(
        select(Decision).where(
            Decision.status == DecisionStatus.APPLIED,
            Decision.created_at < as_of,
            Decision.cluster_snapshot_id.is_not(None),
        )
    ).all()
    hit = False
    checked: set[str] = set()
    for d in prior_decisions:
        sid = d.cluster_snapshot_id
        if sid is None or sid in checked:
            continue
        checked.add(sid)
        snap = session.get(ClusterSnapshot, sid)
        if snap is not None and sorted(snap.member_account_ids) == target:
            hit = True
            break
    return FeatureContribution.make(
        "F-CYC-4", 1.0 if hit else 0.0, _W_CYC_4 if hit else 0.0,
        EvidenceFamily.CYCLE, Rung.S2,
    )


# --- Account-local scoring + any-restriction reduction (S0, §5) --------------


def account_local_features(
    session: Session,
    account_id: str,
    as_of: datetime,
    *,
    age_reference: datetime | None = None,
) -> list[FeatureContribution]:
    """The four metadata (S0) features scored on ONE account, no cluster — this is
    what S0 scores per account before the any-restriction reduction (features.md
    §5). F-DOM-1/F-VER-1 are incriminating; F-DOM-2/F-AGE-1 are benign. The account
    must already exist at `as_of` (review finding 3)."""
    account = session.scalars(
        select(Account).where(
            Account.account_id == account_id, Account.created_at <= as_of
        )
    ).first()
    if account is None:
        raise KeyError(account_id)  # not created as of the scoring instant
    events = list(
        session.scalars(
            select(AccountEvent)
            .where(AccountEvent.account_id == account_id, AccountEvent.timestamp <= as_of)
            .order_by(AccountEvent.timestamp)
        ).all()
    )
    return _account_local_from(account, events, age_reference or as_of)


def _account_local_from(
    account: Account, events: list[AccountEvent], age_reference: datetime
) -> list[FeatureContribution]:
    dom1 = _account_disposable(account)
    ver1 = _account_weak_verification(events)
    dom2 = _account_edu_enterprise(account, events)
    age1 = _account_aged(account, events, age_reference)
    return [
        FeatureContribution.make(
            "F-DOM-1", 1.0 if dom1 else 0.0, _W_DOM_1 if dom1 else 0.0,
            EvidenceFamily.CONTACT_DOMAIN, Rung.S0,
        ),
        FeatureContribution.make(
            "F-VER-1", 1.0 if ver1 else 0.0, _W_VER_1 if ver1 else 0.0,
            EvidenceFamily.VERIFICATION, Rung.S0,
        ),
        FeatureContribution.make(
            "F-DOM-2", 1.0 if dom2 else 0.0, _W_DOM_2 if dom2 else 0.0, None, Rung.S0
        ),
        FeatureContribution.make(
            "F-AGE-1", 1.0 if age1 else 0.0, _W_AGE_1 if age1 else 0.0, None, Rung.S0
        ),
    ]


def _s0_reduce(ctx: _Ctx, age_reference: datetime) -> dict[str, FeatureContribution]:
    """Reduce per-account S0 vectors to the cluster-cycle by the locked 'any member
    account would be proposed for restriction' operator (E-14): the representative is
    the account with the greatest **final** account-local risk — the same clipped
    sum (benign reductions included) the S0 scorer would threshold — not merely the
    largest incriminating sum (review finding 6). Deterministic tie-break by id."""
    per_account: dict[str, list[FeatureContribution]] = {
        aid: _account_local_from(ctx.accounts[aid], ctx.events.get(aid, []), age_reference)
        for aid in ctx.member_ids
        if aid in ctx.accounts
    }
    if not per_account:
        return {}  # no accounts loaded — keep the cluster-aggregate S0 vector

    def final_risk(vec: list[FeatureContribution]) -> float:
        return max(0.0, min(1.0, sum(c.contribution for c in vec if c.fired)))

    # Highest final risk wins; ties broken by the smallest account id (max returns
    # the first maximal element, and ascending sort puts the smallest id first).
    rep_id = max(sorted(per_account), key=lambda aid: final_risk(per_account[aid]))
    return {c.feature_id: c for c in per_account[rep_id]}


# --- Assembly + rung masking -------------------------------------------------


def _neutral_banding() -> BandingResult:
    """A rung-neutral capacity verdict for S0/S1: no timing capacity is estimated, so
    lower rungs cannot consume the S2 allowance/ambiguity (review finding 13)."""
    return BandingResult(
        qualified_stream_count=0,
        allowance=PER_STREAM_ALLOWANCE,
        band=CapacityBand.SINGLE,
        ambiguous=False,
        claimed=0,
        concurrency_passers=(),
        rate_passers=(),
    )


def score_cluster(
    session: Session,
    member_account_ids: list[str],
    as_of: datetime,
    target_rung: Rung,
    *,
    snapshot_id: str = "",
    cycle_claimed: int | None = None,
    age_reference: datetime | None = None,
) -> ClusterScore:
    """Score one cluster-cycle at `target_rung`. All 12 features are emitted in
    FEATURE_ORDER; features with `rung > target_rung` are masked to zero (§5). At
    S0 the metadata features use the account-local any-restriction reduction; at
    S1/S2 they use their cluster aggregate form. Timing capacity (banding) is only
    estimated at S2 — below it the rung gets a neutral verdict (finding 13)."""
    ctx = _load_ctx(session, member_account_ids, as_of)
    age_ref = age_reference or as_of

    # Capacity is an S2 signal: don't even compute it below S2, so it cannot leak
    # into a lower-rung decision.
    band = (
        compute_banding(session, ctx.member_ids, as_of)
        if target_rung >= Rung.S2
        else _neutral_banding()
    )
    claimed = band.claimed if cycle_claimed is None else cycle_claimed

    cluster_level: dict[str, FeatureContribution] = {
        "F-DEV-1": _f_dev_1(ctx),
        "F-DEV-2": _f_dev_2(ctx),
        "F-PAY-1": _f_pay_1(ctx),
        "F-CON-1": _f_con_1(ctx),
        "F-DOM-1": _f_dom_1_cluster(ctx),
        "F-VER-1": _f_ver_1_cluster(ctx),
        "F-CYC-1": _f_cyc_1(ctx, band, claimed),
        "F-CYC-2": _f_cyc_2(ctx),
        "F-CYC-3": _f_cyc_3(ctx),
        "F-CYC-4": _f_cyc_4(session, ctx.member_ids, as_of),
        "F-DOM-2": _f_dom_2_cluster(ctx),
        "F-AGE-1": _f_age_1_cluster(ctx, age_ref),
    }

    # S0 aggregates the metadata features per account, not by cluster count (§5).
    if target_rung == Rung.S0:
        cluster_level.update(_s0_reduce(ctx, age_ref))

    contributions = tuple(
        cluster_level[fid]
        if cluster_level[fid].rung <= target_rung
        else cluster_level[fid].masked()
        for fid in FEATURE_ORDER
    )
    return ClusterScore(contributions=contributions, banding=band, rung=target_rung)


# --- Fixed-unit enumeration (E-14) -------------------------------------------


def enumerate_cluster_cycles(
    session: Session,
    snapshot_id: str,
    member_account_ids: list[str],
    as_of: datetime,
) -> list[ClusterCycleUnit]:
    """The fixed cluster-cycle rows for one snapshot at `as_of` — one per aligned
    cycle window among the members (rung-independent, so identical across rungs)."""
    members = sorted(set(member_account_ids))
    rows = session.scalars(
        select(EntitlementCycle).where(
            EntitlementCycle.account_id.in_(members),
            EntitlementCycle.cycle_start <= as_of,
        )
    ).all()
    windows: dict[datetime, datetime] = {}
    for c in rows:
        start = _utc(c.cycle_start)
        end = _utc(c.cycle_end)
        windows[start] = max(windows.get(start, end), end)

    if not windows:
        # No cycles yet — a single trailing-window unit keeps the row present.
        start = as_of - timedelta(days=banding_mod.WINDOW_DAYS)
        windows = {start: as_of}

    units: list[ClusterCycleUnit] = []
    for start in sorted(windows):
        end = windows[start]
        units.append(
            ClusterCycleUnit(
                cluster_snapshot_id=snapshot_id,
                cycle_id=f"{snapshot_id}::cyc-{start.isoformat()}",
                as_of=as_of,
                window_start=start,
                window_end=end,
            )
        )
    return units


def _claimed_in(
    session: Session, member_ids: list[str], start: datetime, end: datetime
) -> int:
    rows = session.scalars(
        select(AccountEvent).where(
            AccountEvent.account_id.in_(member_ids),
            AccountEvent.event_type == EventType.CLAIM_CREDITS,
        )
    ).all()
    total = 0
    for r in rows:
        ts = _utc(r.timestamp)
        if start <= ts <= end:
            total += r.credits_amount or 0
    return total


def score_cluster_cycle(
    session: Session,
    unit: ClusterCycleUnit,
    member_account_ids: list[str],
    target_rung: Rung,
) -> ClusterScore:
    """Score a specific `(snapshot, cycle, as_of)` unit. F-CYC-1's allowance excess
    uses claims within this cycle's window; every other feature is `as_of`-bounded."""
    cycle_claimed = _claimed_in(
        session,
        sorted(set(member_account_ids)),
        unit.window_start,
        min(unit.window_end, unit.as_of),
    )
    return score_cluster(
        session,
        member_account_ids,
        unit.as_of,
        target_rung,
        snapshot_id=unit.cluster_snapshot_id,
        cycle_claimed=cycle_claimed,
        # F-AGE-1 measures age at the flagged cycle boundary, not the detector time.
        age_reference=unit.window_start,
    )

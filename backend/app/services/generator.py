"""Dev-tier synthetic world generator (Phase 1 Step 2).

Emits accounts / events / linkage / cycles to the inference store, and sealed
labels + the evaluation-only split grouping to the WRITE-ONLY ground-truth sink
(`app.db.ground_truth_writer` — the only ground-truth module this service may
import, review 4 P1-4a). It never reads ground truth back.

**Dev tier — never reported (W-1).** Everything here exists to build and validate
the detection/eval machinery; no number computed from this world is ever a result.

Design commitments this module honours:
  * **Byte-reproducible** (tier2-brief req. 2). `generate_world(seed)` is pure and
    deterministic: a fixed reference frame (no wall clock), string-seeded RNGs
    (stable across `PYTHONHASHSEED`), index-derived ids, and canonically-sorted
    output. `canonical_bytes` serialises a world to bytes for the determinism test
    and the manifest checksum.
  * **No label/scenario/group leaks into inference** (review 4 P1-4d). Labels,
    scenario families, and split groups go ONLY to the ground-truth store; the
    inference rows carry the abuse *pattern* (intentionally derivable) but no
    direct label/scenario/group field.
  * **Realistic timestamps** (req. 3). Events follow a diurnal + weekly rhythm with
    a suppressed sleep window; a uniform-random clock would make the world
    trivially separable in ways unrelated to the policy.
  * **Declarative linkage fixtures** — each scenario cluster states its intended
    connected components. Step 3 (which owns clustering) asserts the *actual*
    components against these; Step 2 only asserts the emitted observations realise
    them (review 4 P2-6a).

Clustering's real edge rule (reliability combinations, hub suppression, size caps,
decay, Louvain) lives in Step 3 (`clustering.py`), which owns the edge-forming
signal vocabulary; fixtures here are shaped from each family's `edge_signal`.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from app.db.ground_truth_writer import GroundTruthSink
from app.models.enums import (
    EmailDomainType,
    EndpointCostTier,
    EventType,
    GroundTruthLabel,
    Reliability,
    SignalType,
    VerificationLevel,
)

# --- Reference frame (fixed — never `now`, or determinism breaks) -------------

WORLD_END = datetime(2026, 6, 1, tzinfo=UTC)
WINDOW_DAYS = 60
CYCLE_DAYS = 30
SLEEP_HOURS = range(1, 6)  # 01:00–05:59 deep-sleep window, heavily suppressed


# --- Emitted records (pure dataclasses; ORM/serialisation both read these) ----


@dataclass(frozen=True)
class GenAccount:
    account_id: str
    created_at: datetime
    email_domain_type: EmailDomainType
    verification_level: VerificationLevel


@dataclass(frozen=True)
class GenEvent:
    event_id: str
    account_id: str
    timestamp: datetime
    event_type: EventType
    credits_amount: int | None
    endpoint_cost_tier: EndpointCostTier | None


@dataclass(frozen=True)
class GenLinkage:
    observation_id: str
    account_id: str
    timestamp: datetime
    signal_type: SignalType
    signal_value_hash: str
    reliability: Reliability


@dataclass(frozen=True)
class GenCycle:
    cycle_id: str
    account_id: str
    cycle_start: datetime
    cycle_end: datetime
    credits_granted: int
    credits_consumed: int
    credits_expired: int


@dataclass(frozen=True)
class GenLabel:
    cluster_id: str
    scenario_family: str
    label: GroundTruthLabel
    notes: str


@dataclass(frozen=True)
class GenSplit:
    account_id: str
    split_group_id: str


@dataclass(frozen=True)
class LinkageFixture:
    """Declarative statement of a scenario cluster's intended relationships.

    `expected_components` partitions the cluster's accounts into the connected
    components clustering *should* produce (Step 3 asserts the real ones). Accounts
    grouped here share an edge-forming resource-sharing signal; singletons share
    only context signals (or nothing) and must not bind.
    """

    cluster_id: str
    scenario_family: str
    expected_components: list[list[str]]


@dataclass(frozen=True)
class GeneratedWorld:
    accounts: list[GenAccount]
    events: list[GenEvent]
    linkage: list[GenLinkage]
    cycles: list[GenCycle]
    labels: list[GenLabel]
    split_groups: list[GenSplit]
    fixtures: list[LinkageFixture]


# --- Scenario families --------------------------------------------------------

Chronotype = str  # "diverse" | "shared" | "homogeneous"


@dataclass(frozen=True)
class FamilySpec:
    label: GroundTruthLabel
    size: tuple[int, int]
    # resource-sharing signal that binds the cluster (None → no edge signal):
    edge_signal: SignalType | None
    edge_reliability: Reliability
    # context-only signal shared across the cluster (never an edge):
    context_signal: SignalType | None
    domain: EmailDomainType
    verification: VerificationLevel
    chronotype: Chronotype
    claim_before_verify: bool = False
    verification_failures: bool = False
    sequential: bool = False  # staggered creation, tiling active windows
    aged: bool = False  # accounts predate the window by ≥90d (benign F-AGE-1)
    consume_per_day: int = 3  # rough consumption intensity
    notes: str = ""


FAMILY_SPECS: dict[str, FamilySpec] = {
    # --- Legitimate ---------------------------------------------------------
    "single_legit": FamilySpec(
        label=GroundTruthLabel.LEGITIMATE,
        size=(1, 1),
        edge_signal=None,
        edge_reliability=Reliability.HIGH,
        context_signal=None,
        domain=EmailDomainType.CONSUMER,
        verification=VerificationLevel.PHONE,
        chronotype="diverse",
        aged=True,
        consume_per_day=2,
        notes="one verified person, normal single-account use",
    ),
    "household_ipad": FamilySpec(
        label=GroundTruthLabel.LEGITIMATE,
        size=(2, 5),
        edge_signal=SignalType.STABLE_DEVICE,  # the shared family iPad — a real edge
        edge_reliability=Reliability.HIGH,
        context_signal=None,
        domain=EmailDomainType.CONSUMER,
        verification=VerificationLevel.PHONE,
        chronotype="diverse",  # distinct people → distinct diurnal signatures
        aged=True,
        consume_per_day=2,
        notes="household sharing one device; diverse schedules → full allowance",
    ),
    "school_lab": FamilySpec(
        label=GroundTruthLabel.LEGITIMATE,
        # ≤ 20 so the shared lab-device value stays under the policy §5 hub
        # threshold (> 20 accounts → downgraded to `low`); a single lab machine on
        # 21+ accounts would legitimately be hub-suppressed and fragment the cluster.
        size=(10, 20),
        edge_signal=SignalType.STABLE_DEVICE,  # shared lab machines
        edge_reliability=Reliability.MEDIUM,
        context_signal=SignalType.NETWORK_GROUP,  # campus NAT — context only
        domain=EmailDomainType.EDUCATION,
        verification=VerificationLevel.EMAIL,
        chronotype="diverse",
        aged=True,
        consume_per_day=2,
        notes="school/library lab; education domain, diverse schedules",
    ),
    "homogeneous_lab": FamilySpec(
        label=GroundTruthLabel.LEGITIMATE,
        size=(12, 20),  # ≤ 20, under the hub threshold (see school_lab)
        edge_signal=SignalType.STABLE_DEVICE,
        edge_reliability=Reliability.MEDIUM,
        context_signal=SignalType.NETWORK_GROUP,
        domain=EmailDomainType.EDUCATION,
        verification=VerificationLevel.EMAIL,
        chronotype="homogeneous",  # distinct people, ONE narrow schedule (policy §4.7)
        aged=True,
        consume_per_day=2,
        notes="named hard negative: homogeneous-schedule lab (policy §4.7)",
    ),
    "workplace_nat": FamilySpec(
        label=GroundTruthLabel.LEGITIMATE,
        size=(4, 12),
        edge_signal=None,  # distinct devices/payment → NO resource-sharing edge
        edge_reliability=Reliability.HIGH,
        context_signal=SignalType.NETWORK_GROUP,  # shared office NAT — context only
        domain=EmailDomainType.ENTERPRISE,
        verification=VerificationLevel.ENHANCED,
        chronotype="diverse",
        aged=True,
        consume_per_day=2,
        notes="workplace behind shared NAT; network is not identity → singletons",
    ),
    # --- Abusive ------------------------------------------------------------
    "trial_farm": FamilySpec(
        label=GroundTruthLabel.ABUSIVE,
        size=(6, 12),
        edge_signal=SignalType.STABLE_DEVICE,  # one device drives many accounts
        edge_reliability=Reliability.HIGH,
        context_signal=SignalType.BEHAVIORAL_TEMPLATE,  # similar behaviour — never an edge
        domain=EmailDomainType.CONSUMER,
        verification=VerificationLevel.NONE,
        chronotype="shared",  # one operator → near-identical diurnal signatures
        claim_before_verify=True,
        verification_failures=True,
        consume_per_day=5,
        notes="one operator, many accounts on one device; claims exceed allowance",
    ),
    "disposable_email_farm": FamilySpec(
        label=GroundTruthLabel.ABUSIVE,
        size=(6, 14),
        edge_signal=SignalType.RECOVERY_CONTACT,  # a shared recovery contact
        edge_reliability=Reliability.MEDIUM,
        context_signal=SignalType.BEHAVIORAL_TEMPLATE,
        domain=EmailDomainType.DISPOSABLE,
        verification=VerificationLevel.NONE,
        chronotype="shared",
        claim_before_verify=True,
        verification_failures=True,
        consume_per_day=5,
        notes="disposable-email farm; shared recovery contact + claim-before-verify",
    ),
    "sequential_cycling": FamilySpec(
        label=GroundTruthLabel.ABUSIVE,
        size=(4, 8),
        edge_signal=SignalType.STABLE_DEVICE,
        edge_reliability=Reliability.HIGH,
        context_signal=None,
        domain=EmailDomainType.CONSUMER,
        verification=VerificationLevel.NONE,
        chronotype="shared",
        sequential=True,  # accounts tile the timeline; near-zero overlap
        claim_before_verify=True,
        consume_per_day=6,
        notes="sequential account cycling; active windows tile with no overlap",
    ),
    "multi_device_farm": FamilySpec(
        label=GroundTruthLabel.ABUSIVE,
        size=(5, 10),
        edge_signal=SignalType.PAYMENT_TOKEN,  # devices differ; one payment token binds
        edge_reliability=Reliability.HIGH,
        context_signal=SignalType.BEHAVIORAL_TEMPLATE,
        domain=EmailDomainType.CONSUMER,
        verification=VerificationLevel.EMAIL,
        chronotype="shared",  # collapses to one diurnal group despite many devices
        consume_per_day=5,
        notes="one operator across many devices; one payment token binds the farm",
    ),
    # --- Ambiguous ----------------------------------------------------------
    "ambiguous_family": FamilySpec(
        label=GroundTruthLabel.AMBIGUOUS,
        size=(3, 5),
        edge_signal=SignalType.STABLE_DEVICE,
        edge_reliability=Reliability.MEDIUM,
        context_signal=None,
        domain=EmailDomainType.CONSUMER,
        verification=VerificationLevel.EMAIL,
        chronotype="diverse",
        consume_per_day=3,
        notes="shared device near the allowance boundary — genuinely uncertain",
    ),
    # --- Phase 4 P4.1 (docs/decisions.md §13 M-1) ---------------------------
    # Reversible singleton pattern: engineered so the reviewer's auto-reversal
    # path is actually reachable (see `_gen_reversible_pair`, which builds this
    # family with fully deterministic event timing rather than the generic
    # per-day random consumption loop above). Two families sharing the exact
    # same structural pattern — the detector cannot tell them apart from
    # features alone, which is the point — differ only in GroundTruthLabel:
    # `reversible_singleton_legit` models a coincidental disposable-domain user
    # whose flag is a genuine false positive (reversal is objectively correct);
    # `reversible_singleton_abusive` models a real farmer wearing the identical
    # feature signature (reversal would be objectively wrong). Both need a
    # `size=(2, 2)` cluster — a benign, otherwise-inert co-member sharing a
    # STABLE_DEVICE edge exists solely to provide the second SIGNUP event
    # F-CYC-3 requires visible within the SAME cluster (empirically confirmed:
    # F-CYC-3 reads only `ctx.member_ids` of the cluster being scored, so an
    # unclustered co-member's signup would be invisible) — RC-1 P1-1's guard
    # still holds since the co-member never claims credits and can never be
    # independently actioned. `_W_*` values live in `features.py`, unmodified.
    "reversible_singleton_legit": FamilySpec(
        label=GroundTruthLabel.LEGITIMATE,
        size=(2, 2),
        edge_signal=SignalType.STABLE_DEVICE,
        edge_reliability=Reliability.HIGH,
        context_signal=None,
        domain=EmailDomainType.DISPOSABLE,
        verification=VerificationLevel.NONE,
        chronotype="diverse",
        notes="P4.1: reversible two-cycle pair, true label legitimate — "
        "the flag this fixture engineers is a genuine false positive",
    ),
    "reversible_singleton_abusive": FamilySpec(
        label=GroundTruthLabel.ABUSIVE,
        size=(2, 2),
        edge_signal=SignalType.STABLE_DEVICE,
        edge_reliability=Reliability.HIGH,
        context_signal=None,
        domain=EmailDomainType.DISPOSABLE,
        verification=VerificationLevel.NONE,
        chronotype="diverse",
        notes="P4.1: reversible two-cycle pair, true label abusive — "
        "the identical feature signature, but reversing here is objectively wrong",
    ),
}

# Families whose generation is fully custom (`_gen_reversible_pair`), not the
# generic random-per-day `_gen_cluster`/`_gen_events_for_account` path — their
# event timing must be exactly deterministic, not sampled, so a co-member's
# signup can be placed within F-CYC-3's 72h window of a known exhaustion
# instant (P4.1, docs/decisions.md §13 M-1).
_CUSTOM_FAMILIES = frozenset({"reversible_singleton_legit", "reversible_singleton_abusive"})

# Cluster composition per preset: (family, count). Ordered — generation and the
# canonical output follow this sequence. "dev" lands ~300 clusters at ~10%
# abusive prevalence (W-7 band 5–15%). "min" is the milestone-§16 vertical slice
# (8 clusters: 3 abusive, 3 legit, 2 ambiguous) for fast tests.
PRESETS: dict[str, list[tuple[str, int]]] = {
    "min": [
        ("single_legit", 1),
        ("household_ipad", 1),
        ("school_lab", 1),
        ("trial_farm", 1),
        ("disposable_email_farm", 1),
        ("sequential_cycling", 1),
        ("ambiguous_family", 2),
    ],
    "dev": [
        ("single_legit", 180),
        ("household_ipad", 30),
        ("school_lab", 15),
        ("homogeneous_lab", 15),
        ("workplace_nat", 15),
        ("trial_farm", 8),
        ("disposable_email_farm", 7),
        ("sequential_cycling", 8),
        ("multi_device_farm", 7),
        ("ambiguous_family", 15),
    ],
}

# Phase 4 P4.1 (docs/decisions.md §13 M-1): `dev` plus the reversible-pair
# families, as a SEPARATE preset — never folded into `"dev"` itself.
#
# `reversible_singleton_legit` is deliberately engineered to fire the exact
# same detector features as `reversible_singleton_abusive` (that indistinguish-
# ability is the whole point — only a reviewer, not the frozen scorer, can
# tell them apart). Folding it into `"dev"` breaks `make tune-check`
# (`scripts/tune_weights.py` sources `generate_world(seed, "dev")` directly):
# the tuning sweep's zero-FP feasibility set goes empty, since the legit
# instances are now unavoidable false positives against the frozen `_W_*`
# weights — an empirically confirmed regression while building this fixture,
# not a hypothetical. `"dev"` stays byte-for-byte what it was at
# `features-frozen-v2`; arm-4/reviewer work (P4.2 onward) uses `"dev_arm4"`.
PRESETS["dev_arm4"] = [
    *PRESETS["dev"],
    # Populated, not exemplary; count/split is a placeholder pending empirical
    # retuning in P4.8 against eval §8's bootstrap gate (>=800 valid
    # replicates, >=80% valid, class-stratified at the split-group level).
    ("reversible_singleton_legit", 20),
    ("reversible_singleton_abusive", 20),
]

DEFAULT_PRESET = "dev"
SCHEMA_VERSION = "dev-world-v1"


# --- Deterministic helpers ----------------------------------------------------


def _hash_value(*parts: str) -> str:
    """A stable synthetic signal value — raw values never cross the collection
    boundary; the store keeps a keyed hash (architecture §4)."""
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def _chronotype_center(rng: random.Random) -> float:
    """A person's preferred active hour — morning, midday, or evening leanings."""
    return rng.choice([9.0, 11.0, 13.0, 16.0, 19.0, 21.0]) + rng.uniform(-1.0, 1.0)


def _event_hour(rng: random.Random, center: float, spread: float) -> tuple[int, int, int]:
    """Sample an (hour, minute, second) around `center`, suppressing deep-sleep
    hours so the world carries a diurnal signature (req. 3)."""
    hour = 12
    for _ in range(8):
        h = rng.gauss(center, spread) % 24.0
        if int(h) in SLEEP_HOURS and rng.random() < 0.9:
            continue
        hour = int(h)
        break
    return hour, rng.randrange(60), rng.randrange(60)


def _active_day(rng: random.Random, day: datetime, base_prob: float) -> bool:
    # Weekly rhythm: lighter on weekends (Sat=5, Sun=6).
    prob = base_prob * (0.45 if day.weekday() >= 5 else 1.0)
    return rng.random() < prob


# --- Per-cluster generation ---------------------------------------------------


@dataclass
class _Sink:
    accounts: list[GenAccount] = field(default_factory=list)
    events: list[GenEvent] = field(default_factory=list)
    linkage: list[GenLinkage] = field(default_factory=list)
    cycles: list[GenCycle] = field(default_factory=list)
    labels: list[GenLabel] = field(default_factory=list)
    split_groups: list[GenSplit] = field(default_factory=list)
    fixtures: list[LinkageFixture] = field(default_factory=list)


def _gen_events_for_account(
    rng: random.Random,
    sink: _Sink,
    account_id: str,
    active_start: datetime,
    center: float,
    spread: float,
    num_cycles: int,
    spec: FamilySpec,
    created: datetime,
) -> None:
    """Signup, per-cycle claim, diurnal consume/login, verification — plus one
    entitlement-cycle row per cycle.

    Temporal integrity (review 2 P1-4): signup establishes creation, so `created`
    is the earliest timestamp and every other event is at or after it. Consumption
    is capped at the grant (P1-2): summed in-window consume events for a cycle equal
    the stored `credits_consumed`, which never exceeds `credits_granted`."""
    ev = 0

    def emit(
        ts: datetime, etype: EventType, amount: int | None, tier: EndpointCostTier | None
    ) -> None:
        nonlocal ev
        sink.events.append(
            GenEvent(f"{account_id}-e{ev:05d}", account_id, ts, etype, amount, tier)
        )
        ev += 1

    # Signup establishes account creation: created_at == signup, and nothing an
    # account does can predate it (review 2 P1-4).
    emit(created, EventType.SIGNUP, None, None)

    # Per-cycle claim timestamps, computed up front so verification can be placed
    # strictly after the first claim for claim-before-verify families (P1-3).
    claim_times: list[datetime] = []
    for c in range(num_cycles):
        cstart = active_start + timedelta(days=CYCLE_DAYS * c)
        _, cm, cs = _event_hour(rng, center, spread)
        claim_times.append(
            (cstart + timedelta(hours=rng.uniform(0, 8))).replace(minute=cm, second=cs)
        )
    first_claim = claim_times[0]

    def _verify_ts(offset_hours: float) -> datetime:
        ts = active_start + timedelta(hours=offset_hours)
        if spec.claim_before_verify and ts <= first_claim:
            # Force verification strictly after the first claim so claim-before-verify
            # is a scenario guarantee, not an accident of independent sampling (P1-3).
            ts = first_claim + timedelta(hours=0.5 + offset_hours)
        return ts

    # Verification behaviour (F-VER-1 material). Repeated failures are emitted as
    # VERIFICATION_FAILED, so repeated failure is inferable from event types alone.
    if spec.verification_failures:
        for _ in range(rng.randint(2, 4)):
            emit(_verify_ts(rng.uniform(0, 12)), EventType.VERIFICATION_FAILED, None, None)
    elif spec.verification != VerificationLevel.NONE:
        emit(_verify_ts(rng.uniform(1, 24)), EventType.VERIFICATION, None, None)

    # Aged accounts show routine, low-cost, non-credit activity before the window
    # (F-AGE-1). `created` is set back far enough (≥240d) to cover the oldest login.
    if spec.aged:
        for w in range(1, rng.randint(3, 6)):
            h, m, s = _event_hour(rng, center, spread)
            day = active_start - timedelta(days=30 * w + rng.randint(0, 20))
            emit(day.replace(hour=h, minute=m, second=s), EventType.LOGIN, None, None)

    for c in range(num_cycles):
        cstart = active_start + timedelta(days=CYCLE_DAYS * c)
        cend = cstart + timedelta(days=CYCLE_DAYS)
        granted = 25
        emit(claim_times[c], EventType.CLAIM_CREDITS, granted, None)

        consumed = 0
        for d in range(CYCLE_DAYS):
            day = cstart + timedelta(days=d)
            if day > WORLD_END:
                break
            if not _active_day(rng, day, base_prob=0.55):
                continue
            for _ in range(rng.randint(0, spec.consume_per_day)):
                if consumed >= granted:
                    break  # grant exhausted — never over-consume (review 2 P1-2)
                h, m, s = _event_hour(rng, center, spread)
                ts = day.replace(hour=h, minute=m, second=s)
                if ts > WORLD_END:
                    continue
                amt = min(rng.randint(1, 4), granted - consumed)  # clamp last event to grant
                consumed += amt
                emit(ts, EventType.CONSUME_CREDITS, amt, EndpointCostTier.HIGH)
            if rng.random() < 0.5:
                h, m, s = _event_hour(rng, center, spread)
                emit(day.replace(hour=h, minute=m, second=s), EventType.LOGIN, None, None)

        closed = cend <= WORLD_END
        # consumed ≤ granted by construction, so the cycle's stored consumption is
        # exactly the sum of its in-window consume events (no min() masking).
        expired = granted - consumed if closed else 0
        sink.cycles.append(
            GenCycle(
                cycle_id=f"{account_id}-c{c:02d}",
                account_id=account_id,
                cycle_start=cstart,
                cycle_end=cend,
                credits_granted=granted,
                credits_consumed=consumed,
                credits_expired=expired,
            )
        )


def _gen_cluster(seed: int, cluster_id: str, family: str) -> _Sink:
    spec = FAMILY_SPECS[family]
    rng = random.Random(f"{seed}:{cluster_id}")
    sink = _Sink()

    size = rng.randint(*spec.size)
    # OPAQUE deterministic account ids (review 2 P0-1): the inference plane must not
    # encode the scenario family. The account_id → cluster/family mapping exists ONLY
    # in the ground-truth plane (split_groups below, keyed to the family-encoded
    # cluster_id). Event / observation / cycle ids derive from the opaque base, so
    # nothing in the inference store reveals the family.
    account_ids = [
        f"acct-{hashlib.sha256(f'{seed}|{cluster_id}|{j}'.encode()).hexdigest()[:16]}"
        for j in range(size)
    ]

    # One operator's chronotype for "shared"; one narrow window for "homogeneous".
    shared_center = _chronotype_center(rng)
    homogeneous_center = rng.choice([10.0, 14.0, 15.0]) + rng.uniform(-0.3, 0.3)

    # Shared signal values, minted once per cluster.
    edge_value = _hash_value("edge", family, cluster_id) if spec.edge_signal else None
    context_value = _hash_value("ctx", family, cluster_id) if spec.context_signal else None

    for j, account_id in enumerate(account_ids):
        # Account age. Aged accounts are created ≥240d before WORLD_END — early
        # enough that their oldest pre-window login (≤170d before active_start,
        # i.e. ≤230d before WORLD_END) can never predate creation (review 2 P1-4).
        if spec.aged:
            created = WORLD_END - timedelta(days=rng.randint(240, 400))
            active_start = WORLD_END - timedelta(days=WINDOW_DAYS)
        elif spec.sequential:
            # Tile the window: account j occupies a distinct, non-overlapping slot.
            slot = WINDOW_DAYS / max(1, size)
            active_start = WORLD_END - timedelta(days=WINDOW_DAYS - j * slot)
            created = active_start - timedelta(hours=rng.uniform(1, 24))
        else:
            active_start = WORLD_END - timedelta(days=WINDOW_DAYS)
            created = active_start - timedelta(hours=rng.uniform(0, 48))

        sink.accounts.append(
            GenAccount(
                account_id=account_id,
                created_at=created,
                email_domain_type=spec.domain,
                verification_level=spec.verification,
            )
        )

        # Per-account chronotype by mode.
        if spec.chronotype == "shared":
            center, spread = shared_center + rng.uniform(-0.2, 0.2), 1.6
        elif spec.chronotype == "homogeneous":
            center, spread = homogeneous_center + rng.uniform(-0.2, 0.2), 0.8
        else:  # diverse
            center, spread = _chronotype_center(rng), 2.2

        num_cycles = 1 if spec.sequential else 2
        _gen_events_for_account(
            rng, sink, account_id, active_start, center, spread, num_cycles, spec, created
        )

        # Linkage observations.
        if spec.edge_signal is not None:
            reliability = spec.edge_reliability
            sink.linkage.append(
                GenLinkage(
                    observation_id=f"{account_id}-l0",
                    account_id=account_id,
                    timestamp=active_start + timedelta(hours=rng.uniform(0, 48)),
                    signal_type=spec.edge_signal,
                    signal_value_hash=edge_value,  # type: ignore[arg-type]
                    reliability=reliability,
                )
            )
        if spec.context_signal is not None:
            sink.linkage.append(
                GenLinkage(
                    observation_id=f"{account_id}-lctx",
                    account_id=account_id,
                    timestamp=active_start + timedelta(hours=rng.uniform(0, 48)),
                    signal_type=spec.context_signal,
                    signal_value_hash=context_value,  # type: ignore[arg-type]
                    reliability=Reliability.LOW,
                )
            )

        # Split grouping: the generator's scenario-group id (review 4 P1-4c) — an
        # evaluation-only key, never an inference input.
        sink.split_groups.append(GenSplit(account_id=account_id, split_group_id=cluster_id))

    # Sealed label (write-only sink).
    sink.labels.append(
        GenLabel(
            cluster_id=cluster_id,
            scenario_family=family,
            label=spec.label,
            notes=spec.notes,
        )
    )

    # Declarative linkage fixture: an edge signal binds all members into one
    # component; without one (or with context-only signals) every account is a
    # singleton — the intended relationship Step 3's clustering must reproduce.
    if spec.edge_signal is not None:
        components = [sorted(account_ids)]
    else:
        components = [[a] for a in account_ids]
    sink.fixtures.append(
        LinkageFixture(
            cluster_id=cluster_id,
            scenario_family=family,
            expected_components=components,
        )
    )
    return sink


# --- P4.1 custom deterministic pair (docs/decisions.md §13 M-1) --------------

# F-CYC-3's exhaustion→signup latency ceiling (features.py) is 72h; this offset
# is comfortably inside it without being suspiciously tight against the boundary.
_REVERSIBLE_CYC3_OFFSET_HOURS = 24.0

# Cycle 1: claimed well past the light_friction ceiling (excess = 60/25-1 = 1.4,
# clamped to F-CYC-1's weight regardless — only existence + later APPLIED status
# matters for F-CYC-4's precedent, not this cycle's own severity).
_REVERSIBLE_CYCLE1_CLAIMED = 60
# Cycle 2: claimed lands strictly under the 0.25 light_friction boundary
# (31/25 - 1 = 0.24), so the ladder keeps this decision at light_friction —
# required for R-6's auto-reversal ceiling.
_REVERSIBLE_CYCLE2_CLAIMED = 31


def _gen_reversible_pair(seed: int, cluster_id: str, family: str) -> _Sink:
    """P4.1 — a fully deterministic two-cycle-primary-plus-inert-co-member
    pattern (see the `reversible_singleton_*` `FamilySpec` docstring above).

    Unlike `_gen_cluster`/`_gen_events_for_account`, event timing here is exact
    rather than sampled per day: cycle 1 exhausts at a known instant so the
    co-member's SIGNUP can be placed within F-CYC-3's 72h window of it, and
    cycle 2's claimed total is pinned to land just under the light_friction
    excess ceiling.

    The co-member shares a resource-sharing edge (`STABLE_DEVICE`, HIGH
    reliability) with the primary account so clustering binds them into ONE
    cluster — required because F-CYC-3 (like every S2 feature) reads only
    `ctx.member_ids` of the cluster actually being scored; a separate,
    unclustered co-member's SIGNUP would be invisible to primary's own
    scoring (empirically confirmed while building this fixture). The
    co-member still never counts as a qualified capacity stream (banding
    §4.2's concurrency axis requires >= 3 *shared* consumption days, and it
    never consumes at all), so `allowance` stays 25 despite the 2-member
    cluster, and RC-1 P1-1's "no other currently-actioned member" auto-reversal
    guard stays satisfied since the co-member is never independently actioned.
    """
    spec = FAMILY_SPECS[family]
    rng = random.Random(f"{seed}:{cluster_id}")
    sink = _Sink()

    primary_id = f"acct-{hashlib.sha256(f'{seed}|{cluster_id}|0'.encode()).hexdigest()[:16]}"
    partner_id = f"acct-{hashlib.sha256(f'{seed}|{cluster_id}|1'.encode()).hexdigest()[:16]}"
    edge_value = _hash_value("edge", family, cluster_id)

    cycle1_start = WORLD_END - timedelta(days=WINDOW_DAYS)
    cycle1_end = cycle1_start + timedelta(days=CYCLE_DAYS)
    cycle2_start = cycle1_end
    cycle2_end = cycle2_start + timedelta(days=CYCLE_DAYS)

    # M-12 (docs/decisions.md §13): the legit and abusive variants must be
    # genuinely distinguishable, not evidence-identical — via a real, citable
    # RAW FACT in the persisted timeline, not a scored feature. An earlier
    # attempt aged the legit primary past F-AGE-1's 90-day threshold
    # (`_f_age_1_cluster`) — that fires at weight -0.15 (benign features
    # subtract), which pulled cycle 2's total below the 0.60 action gate
    # entirely (empirically confirmed: legit stopped producing any decision at
    # all, defeating the point — we need legit correctly FLAGGED so a reversal
    # has something to reverse). Fixed: the legit primary is older (75d before
    # cycle 2's `window_start` reference) and has real login history — a
    # citable difference a reviewer can point to (an established account with
    # routine use, not a fresh throwaway) — but deliberately stays UNDER the
    # 90-day threshold, so it moves no feature contribution at all; cycle 2's
    # total score is identical to the abusive variant's, as M-1 requires. The
    # co-member's own creation instant stays fixed by F-CYC-3's 72h-after-
    # exhaustion requirement (below) regardless of label, so this signal lives
    # on the primary, not the co-member.
    is_legit = spec.label is GroundTruthLabel.LEGITIMATE
    primary_created = (
        cycle1_start - timedelta(days=45)
        if is_legit
        else cycle1_start - timedelta(hours=rng.uniform(1, 24))
    )
    sink.accounts.append(
        GenAccount(primary_id, primary_created, spec.domain, spec.verification)
    )

    ev = 0

    def emit_primary(
        ts: datetime, etype: EventType, amount: int | None, tier: EndpointCostTier | None
    ) -> None:
        nonlocal ev
        sink.events.append(GenEvent(f"{primary_id}-e{ev:05d}", primary_id, ts, etype, amount, tier))
        ev += 1

    emit_primary(primary_created, EventType.SIGNUP, None, None)
    if is_legit:
        # `_routine_activity`'s >=2-LOGIN floor (design clarification 3) — placed
        # well before cycle 1 so it reads as ordinary pre-existing use, not
        # activity manufactured around the flagged cycles.
        emit_primary(primary_created + timedelta(days=10), EventType.LOGIN, None, None)
        emit_primary(primary_created + timedelta(days=20), EventType.LOGIN, None, None)

    # Cycle 1: large claim, exhausted by one exact consume event — F-DOM-1's
    # domain is set on the account above; F-VER-1 fires via claim-before-verify
    # (two VERIFICATION_FAILED events strictly after the first claim, review 2
    # P1-3's pattern).
    claim1_ts = cycle1_start + timedelta(hours=2)
    emit_primary(claim1_ts, EventType.CLAIM_CREDITS, _REVERSIBLE_CYCLE1_CLAIMED, None)
    exhaustion_ts = cycle1_start + timedelta(hours=6)
    emit_primary(exhaustion_ts, EventType.CONSUME_CREDITS, 25, EndpointCostTier.HIGH)
    emit_primary(claim1_ts + timedelta(hours=1), EventType.VERIFICATION_FAILED, None, None)
    emit_primary(claim1_ts + timedelta(hours=2), EventType.VERIFICATION_FAILED, None, None)
    sink.cycles.append(
        GenCycle(f"{primary_id}-c00", primary_id, cycle1_start, cycle1_end, 25, 25, 0)
    )

    # Cycle 2: claimed pinned just under the light_friction ceiling; partial
    # consumption, closed by WORLD_END (cycle2_end == WORLD_END exactly).
    claim2_ts = cycle2_start + timedelta(hours=2)
    emit_primary(claim2_ts, EventType.CLAIM_CREDITS, _REVERSIBLE_CYCLE2_CLAIMED, None)
    consume2_amt = 20
    emit_primary(
        cycle2_start + timedelta(hours=6), EventType.CONSUME_CREDITS, consume2_amt,
        EndpointCostTier.HIGH,
    )
    sink.cycles.append(
        GenCycle(
            f"{primary_id}-c01", primary_id, cycle2_start, cycle2_end, 25, consume2_amt,
            25 - consume2_amt,
        )
    )

    # Inert co-member: SIGNUP within F-CYC-3's 72h window of cycle 1's
    # exhaustion, then nothing else — never claims, never independently
    # actionable (RC-1 P1-1 stays satisfied), and its 0 consumption days keep
    # it out of banding's qualified-stream count (concurrency needs
    # N_OVERLAP >= 3 *shared* days). Shares a STABLE_DEVICE edge with the
    # primary account so both land in the SAME cluster (see docstring above —
    # required for F-CYC-3 to see the co-member's signup at all).
    partner_created = exhaustion_ts + timedelta(hours=_REVERSIBLE_CYC3_OFFSET_HOURS)
    sink.accounts.append(
        GenAccount(partner_id, partner_created, EmailDomainType.CONSUMER, VerificationLevel.PHONE)
    )
    sink.events.append(
        GenEvent(f"{partner_id}-e00000", partner_id, partner_created, EventType.SIGNUP, None, None)
    )
    for account_id, ts in (
        (primary_id, cycle1_start + timedelta(hours=1)),
        (partner_id, partner_created + timedelta(hours=1)),
    ):
        sink.linkage.append(
            GenLinkage(
                observation_id=f"{account_id}-l0",
                account_id=account_id,
                timestamp=ts,
                signal_type=SignalType.STABLE_DEVICE,
                signal_value_hash=edge_value,
                reliability=Reliability.HIGH,
            )
        )

    for account_id in (primary_id, partner_id):
        sink.split_groups.append(GenSplit(account_id=account_id, split_group_id=cluster_id))

    sink.labels.append(
        GenLabel(cluster_id=cluster_id, scenario_family=family, label=spec.label, notes=spec.notes)
    )
    sink.fixtures.append(
        LinkageFixture(
            cluster_id=cluster_id,
            scenario_family=family,
            expected_components=[sorted([primary_id, partner_id])],
        )
    )
    return sink


# --- Public API ---------------------------------------------------------------


def generate_world(seed: int, preset: str = DEFAULT_PRESET) -> GeneratedWorld:
    """Deterministically build a dev world. Pure — no I/O, no wall clock."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")

    world = _Sink()
    counter = 0
    for family, count in PRESETS[preset]:
        for _ in range(count):
            cluster_id = f"{family}-{counter:04d}"
            counter += 1
            part = (
                _gen_reversible_pair(seed, cluster_id, family)
                if family in _CUSTOM_FAMILIES
                else _gen_cluster(seed, cluster_id, family)
            )
            world.accounts.extend(part.accounts)
            world.events.extend(part.events)
            world.linkage.extend(part.linkage)
            world.cycles.extend(part.cycles)
            world.labels.extend(part.labels)
            world.split_groups.extend(part.split_groups)
            world.fixtures.extend(part.fixtures)

    # Canonical ordering: independent of sampling order, so output is stable.
    return GeneratedWorld(
        accounts=sorted(world.accounts, key=lambda a: a.account_id),
        events=sorted(world.events, key=lambda e: (e.account_id, e.timestamp, e.event_id)),
        linkage=sorted(world.linkage, key=lambda x: x.observation_id),
        cycles=sorted(world.cycles, key=lambda c: c.cycle_id),
        labels=sorted(world.labels, key=lambda x: x.cluster_id),
        split_groups=sorted(world.split_groups, key=lambda s: s.account_id),
        fixtures=sorted(world.fixtures, key=lambda f: f.cluster_id),
    )


def _canonical(obj: object) -> object:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (EmailDomainType, VerificationLevel, EventType, EndpointCostTier,
                        SignalType, Reliability, GroundTruthLabel)):
        return obj.value
    raise TypeError(f"not canonically serialisable: {type(obj)!r}")


def canonical_bytes(world: GeneratedWorld) -> bytes:
    """A byte-stable serialisation of a world — the determinism test compares this,
    and the manifest checksums it."""
    payload = {
        "accounts": [asdict(a) for a in world.accounts],
        "events": [asdict(e) for e in world.events],
        "linkage": [asdict(x) for x in world.linkage],
        "cycles": [asdict(c) for c in world.cycles],
        "labels": [asdict(x) for x in world.labels],
        "split_groups": [asdict(s) for s in world.split_groups],
        "fixtures": [asdict(f) for f in world.fixtures],
    }
    return json.dumps(
        payload, default=_canonical, sort_keys=True, separators=(",", ":")
    ).encode()


def persist_world(
    world: GeneratedWorld,
    inference_session: Session,
    ground_truth_sink: GroundTruthSink,
) -> None:
    """Write the world to the inference store, and sealed labels + split groups to
    the WRITE-ONLY ground-truth sink. The sink is a capability object exposing only
    the two write methods — this module never receives a readable GT session
    (review 1 P0). Caller owns commits (of the session that backs the sink)."""
    # Deferred import so this module's only ground-truth dependency stays the
    # write sink (the capability boundary test allows exactly that, review 4 P1-4a).
    from app.models.inference import (
        Account,
        AccountEvent,
        EntitlementCycle,
        LinkageObservation,
    )

    for a in world.accounts:
        inference_session.add(
            Account(
                account_id=a.account_id,
                created_at=a.created_at,
                email_domain_type=a.email_domain_type,
                verification_level=a.verification_level,
            )
        )
    for e in world.events:
        inference_session.add(
            AccountEvent(
                event_id=e.event_id,
                account_id=e.account_id,
                timestamp=e.timestamp,
                event_type=e.event_type,
                credits_amount=e.credits_amount,
                endpoint_cost_tier=e.endpoint_cost_tier,
            )
        )
    for x in world.linkage:
        inference_session.add(
            LinkageObservation(
                observation_id=x.observation_id,
                account_id=x.account_id,
                timestamp=x.timestamp,
                signal_type=x.signal_type,
                signal_value_hash=x.signal_value_hash,
                reliability=x.reliability,
            )
        )
    for c in world.cycles:
        inference_session.add(
            EntitlementCycle(
                cycle_id=c.cycle_id,
                account_id=c.account_id,
                cycle_start=c.cycle_start,
                cycle_end=c.cycle_end,
                credits_granted=c.credits_granted,
                credits_consumed=c.credits_consumed,
                credits_expired=c.credits_expired,
            )
        )
    inference_session.flush()

    for lbl in world.labels:
        ground_truth_sink.write_ground_truth(
            cluster_id=lbl.cluster_id,
            scenario_family=lbl.scenario_family,
            label=lbl.label,
            notes=lbl.notes,
        )
    for s in world.split_groups:
        ground_truth_sink.write_split_group(
            account_id=s.account_id, split_group_id=s.split_group_id
        )


def prevalence(world: GeneratedWorld) -> float:
    """Fraction of clusters labelled abusive (W-7 target band 5–15%)."""
    total = len(world.labels)
    if total == 0:
        return 0.0
    abusive = sum(1 for x in world.labels if x.label == GroundTruthLabel.ABUSIVE)
    return abusive / total


def build_manifest(world: GeneratedWorld, seed: int, preset: str) -> dict:
    """Dev manifest — schema version, seed, preset, family list, counts, checksum.

    Mirrors the tier2-brief §Requirements manifest, minus the author-model field
    (this is the dev tier, authored in-repo — W-1, never reported)."""
    families = sorted({lbl.scenario_family for lbl in world.labels})
    return {
        "schema_version": SCHEMA_VERSION,
        "tier": "dev",
        "reported": False,  # W-1 — dev-tier numbers are never a result
        "seed": seed,
        "preset": preset,
        "families": families,
        "counts": {
            "clusters": len(world.labels),
            "accounts": len(world.accounts),
            "events": len(world.events),
            "linkage_observations": len(world.linkage),
            "cycles": len(world.cycles),
        },
        "prevalence_abusive": round(prevalence(world), 4),
        "checksum_sha256": hashlib.sha256(canonical_bytes(world)).hexdigest(),
    }


def write_artifacts(world: GeneratedWorld, out_dir: str, seed: int, preset: str) -> dict:
    """Emit parquet tables + `manifest.json` under `out_dir` (data/synthetic/).

    Ground truth is written to its own sealed store elsewhere; the labels parquet
    here is a dev-tier convenience artifact only. Returns the manifest."""
    import pathlib

    import polars as pl

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _rows(records: list) -> list[dict]:
        return [json.loads(json.dumps(asdict(r), default=_canonical)) for r in records]

    pl.DataFrame(_rows(world.accounts)).write_parquet(out / "accounts.parquet")
    pl.DataFrame(_rows(world.events)).write_parquet(out / "events.parquet")
    pl.DataFrame(_rows(world.linkage)).write_parquet(out / "linkage.parquet")
    pl.DataFrame(_rows(world.cycles)).write_parquet(out / "cycles.parquet")

    manifest = build_manifest(world, seed, preset)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest

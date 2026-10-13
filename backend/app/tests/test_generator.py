"""Dev-tier generator (Step 2) green gates:
  * determinism — same seed → byte-identical, different seed → different;
  * prevalence within the W-7 5–15% abusive band;
  * timestamps non-uniform (a diurnal signature is present, tier2-brief req. 3);
  * declarative linkage fixtures match their intended relationships (the *actual*
    connected-component assertion is Step 3's, review 4 P2-6a);
  * persistence writes inference rows and sends labels to the SEALED GT sink only.

Dev tier — never reported (W-1).
"""

from collections import Counter, defaultdict

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from app.core.clock import now_utc
from app.db.ground_truth_reader import load_labels, load_split_groups
from app.db.ground_truth_writer import make_ground_truth_sink
from app.models.enums import EventType, GroundTruthLabel
from app.models.inference import (
    Account,
    AccountEvent,
    ClusterSnapshot,
    EntitlementCycle,
    LinkageEdge,
    LinkageObservation,
)
from app.services.clustering import (
    RESOURCE_SHARING_SIGNALS,
    compute_clusters,
    materialize,
)
from app.services.generator import (
    FAMILY_SPECS,
    WORLD_END,
    build_manifest,
    canonical_bytes,
    generate_world,
    persist_world,
    prevalence,
)

SEED = 1


# --- Determinism (tier2-brief req. 2, original §18) --------------------------


def test_same_seed_is_byte_identical() -> None:
    assert canonical_bytes(generate_world(SEED, "min")) == canonical_bytes(
        generate_world(SEED, "min")
    )


def test_different_seed_differs() -> None:
    assert canonical_bytes(generate_world(1, "min")) != canonical_bytes(
        generate_world(2, "min")
    )


def test_dev_world_checksum_stable() -> None:
    # The whole dev world (not just the tiny preset) reproduces to the same hash.
    a = build_manifest(generate_world(SEED, "dev"), SEED, "dev")["checksum_sha256"]
    b = build_manifest(generate_world(SEED, "dev"), SEED, "dev")["checksum_sha256"]
    assert a == b


# --- Prevalence (W-7) --------------------------------------------------------


def test_prevalence_within_band() -> None:
    p = prevalence(generate_world(SEED, "dev"))
    assert 0.05 <= p <= 0.15, f"abusive prevalence {p} outside 5–15% band"


# --- Diurnal signature (tier2-brief req. 3) ----------------------------------


def test_consume_timestamps_are_non_uniform() -> None:
    world = generate_world(SEED, "dev")
    hours = Counter(
        e.timestamp.hour for e in world.events if e.event_type == EventType.CONSUME_CREDITS
    )
    total = sum(hours.values())
    assert total > 1000  # enough events to see a distribution
    # Deep-sleep hours are strongly suppressed — far below the uniform 5/24 ≈ 0.208.
    sleep_fraction = sum(hours.get(h, 0) for h in range(1, 6)) / total
    assert sleep_fraction < 0.05, f"sleep-hours fraction {sleep_fraction} looks uniform"
    # And there is a clear day/night contrast.
    day = sum(hours.get(h, 0) for h in range(9, 22))
    night = sum(hours.get(h, 0) for h in list(range(0, 6)))
    assert day > 5 * night


# --- Declarative linkage fixtures (review 4 P2-6a) ---------------------------


def _cluster_accounts(world, cluster_id: str) -> set[str]:
    # Account ids are opaque (review 2 P0-1); the account → cluster mapping lives
    # ONLY in the ground-truth-plane split grouping, which this test may read.
    return {s.account_id for s in world.split_groups if s.split_group_id == cluster_id}


def _components_from_shared_edge_signals(world, accounts: set[str]) -> list[list[str]]:
    """Group accounts that share an edge-forming (resource-sharing) signal value.

    A deliberately minimal union-by-shared-value — NOT the Step 3 edge rule (which
    owns reliability combinations, hub suppression, size caps, decay). It only
    verifies the emitted observations realise the declared relationships; context
    signals (network_group / behavioral_template) are excluded by construction, so
    a cluster bound only by those yields singletons."""
    parent: dict[str, str] = {a: a for a in accounts}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    by_value: dict[str, set[str]] = defaultdict(set)
    for obs in world.linkage:
        if obs.account_id in accounts and obs.signal_type in RESOURCE_SHARING_SIGNALS:
            by_value[obs.signal_value_hash].add(obs.account_id)
    for members in by_value.values():
        members = sorted(members)
        for other in members[1:]:
            union(members[0], other)

    groups: dict[str, list[str]] = defaultdict(list)
    for a in accounts:
        groups[find(a)].append(a)
    return sorted(sorted(g) for g in groups.values())


def test_fixtures_partition_their_cluster_accounts() -> None:
    world = generate_world(SEED, "dev")
    for fx in world.fixtures:
        accounts = _cluster_accounts(world, fx.cluster_id)
        flat = [a for comp in fx.expected_components for a in comp]
        assert len(flat) == len(set(flat)), f"{fx.cluster_id}: overlapping components"
        assert set(flat) == accounts, f"{fx.cluster_id}: components don't cover accounts"


def test_fixtures_are_realized_by_emitted_observations() -> None:
    world = generate_world(SEED, "dev")
    for fx in world.fixtures:
        accounts = _cluster_accounts(world, fx.cluster_id)
        realized = _components_from_shared_edge_signals(world, accounts)
        expected = sorted(sorted(c) for c in fx.expected_components)
        assert realized == expected, (
            f"{fx.cluster_id} ({fx.scenario_family}): {realized} != {expected}"
        )


def test_context_only_family_yields_singletons() -> None:
    # workplace_nat shares only a network_group (context) — it must not bind.
    world = generate_world(SEED, "dev")
    nat = [f for f in world.fixtures if f.scenario_family == "workplace_nat"]
    assert nat, "expected workplace_nat clusters in the dev world"
    for fx in nat:
        assert all(len(c) == 1 for c in fx.expected_components)


def test_shared_device_family_is_one_component() -> None:
    world = generate_world(SEED, "dev")
    farms = [f for f in world.fixtures if f.scenario_family == "trial_farm"]
    assert farms
    for fx in farms:
        assert len(fx.expected_components) == 1
        assert len(fx.expected_components[0]) >= 2


# --- Persistence: sealed GT sink, inference free of labels -------------------


def test_persist_writes_inference_and_sealed_ground_truth(
    inference_engine: Engine, gt_engine: Engine
) -> None:
    world = generate_world(SEED, "min")
    with Session(inference_engine, future=True) as inf, Session(gt_engine, future=True) as gt:
        persist_world(world, inf, make_ground_truth_sink(gt))
        inf.commit()
        gt.commit()

    with Session(inference_engine, future=True) as inf:
        assert inf.scalar(select(func.count()).select_from(Account)) == len(world.accounts)
        assert inf.scalar(select(func.count()).select_from(AccountEvent)) == len(world.events)
        assert inf.scalar(select(func.count()).select_from(LinkageObservation)) == len(
            world.linkage
        )
        assert inf.scalar(select(func.count()).select_from(EntitlementCycle)) == len(
            world.cycles
        )

    with Session(gt_engine, future=True) as gt:
        labels = load_labels(gt)
        assert set(labels) == {lbl.cluster_id for lbl in world.labels}
        assert GroundTruthLabel.ABUSIVE in labels.values()
        assert load_split_groups(gt).keys() == {a.account_id for a in world.accounts}


# --- Finding 1 (P0): opaque inference identifiers (review 2 P0-1) -------------


def test_no_family_or_label_leaks_into_inference_identifiers(
    inference_engine: Engine, gt_engine: Engine
) -> None:
    """No inference-plane identifier may encode a scenario family / label / group.
    Scans ACTUAL persisted values (not column names) across every inference table,
    including materialized snapshot member ids and edge ids."""
    world = generate_world(SEED, "dev")
    with Session(inference_engine, future=True) as inf, Session(gt_engine, future=True) as gt:
        persist_world(world, inf, make_ground_truth_sink(gt))
        inf.commit()
        gt.commit()
    with Session(inference_engine, future=True) as inf:
        clusters = compute_clusters(inf, WORLD_END)
        materialize(inf, clusters, as_of=WORLD_END, created_at=now_utc())
        inf.commit()

    forbidden = {fam.lower() for fam in FAMILY_SPECS} | {
        "label", "scenario", "family", "split_group",
        "abusive", "legitimate", "ambiguous",
    }

    ids: list[str] = []
    with Session(inference_engine, future=True) as inf:
        ids += [a.account_id for a in inf.scalars(select(Account)).all()]
        for e in inf.scalars(select(AccountEvent)).all():
            ids += [e.event_id, e.account_id]
        for o in inf.scalars(select(LinkageObservation)).all():
            ids += [o.observation_id, o.account_id]
        for c in inf.scalars(select(EntitlementCycle)).all():
            ids += [c.cycle_id, c.account_id]
        for s in inf.scalars(select(ClusterSnapshot)).all():
            ids.append(s.snapshot_id)
            ids += list(s.member_account_ids)
        for ed in inf.scalars(select(LinkageEdge)).all():
            ids += [ed.edge_id, ed.account_a_id, ed.account_b_id]

    offenders = [i for i in ids if any(tok in i.lower() for tok in forbidden)]
    assert not offenders, f"family/label leaked into inference identifiers: {offenders[:10]}"


# --- Finding 2 (P1): consumption ≤ grant, matches the cycle row ---------------


def test_cycle_consumption_matches_events_and_respects_grant() -> None:
    world = generate_world(SEED, "dev")
    consume_by_account: dict[str, list] = defaultdict(list)
    for e in world.events:
        if e.event_type == EventType.CONSUME_CREDITS:
            consume_by_account[e.account_id].append(e)

    assert world.cycles, "expected cycles in the dev world"
    for cyc in world.cycles:
        in_window = sum(
            e.credits_amount or 0
            for e in consume_by_account[cyc.account_id]
            if cyc.cycle_start <= e.timestamp < cyc.cycle_end
        )
        assert in_window == cyc.credits_consumed, (
            f"{cyc.cycle_id}: events {in_window} != stored {cyc.credits_consumed}"
        )
        assert cyc.credits_consumed <= cyc.credits_granted
        closed = cyc.cycle_end <= WORLD_END
        expected_expired = cyc.credits_granted - cyc.credits_consumed if closed else 0
        assert cyc.credits_expired == expected_expired


# --- Finding 3 (P1): F-VER-1 — claim-before-verify + VERIFICATION_FAILED ------


def _family_accounts(world) -> dict[str, set[str]]:
    fam: dict[str, set[str]] = defaultdict(set)
    for fx in world.fixtures:
        for comp in fx.expected_components:
            fam[fx.scenario_family].update(comp)
    return fam


def test_claim_before_verify_families_claim_first() -> None:
    world = generate_world(SEED, "dev")
    fam = _family_accounts(world)
    events_by_account: dict[str, list] = defaultdict(list)
    for e in world.events:
        events_by_account[e.account_id].append(e)

    cbv = [f for f, s in FAMILY_SPECS.items() if s.claim_before_verify]
    checked = 0
    for family in cbv:
        for acct in fam[family]:
            evs = events_by_account[acct]
            verifs = [
                e.timestamp
                for e in evs
                if e.event_type in (EventType.VERIFICATION, EventType.VERIFICATION_FAILED)
            ]
            if not verifs:
                continue  # a claim-before-verify family need not always verify
            claims = [e.timestamp for e in evs if e.event_type == EventType.CLAIM_CREDITS]
            assert claims, f"{acct}: no claim events"
            assert min(claims) < min(verifs), f"{acct}: claim not strictly before verify"
            checked += 1
    assert checked > 0, "no claim-before-verify accounts with verification events were checked"


def test_verification_failures_are_repeated_and_typed() -> None:
    world = generate_world(SEED, "dev")
    fam = _family_accounts(world)
    failed_by_account: dict[str, int] = defaultdict(int)
    for e in world.events:
        if e.event_type == EventType.VERIFICATION_FAILED:
            failed_by_account[e.account_id] += 1

    vf = [f for f, s in FAMILY_SPECS.items() if s.verification_failures]
    assert vf, "expected verification-failure families in the spec"
    for family in vf:
        assert fam[family], f"{family} absent from the dev world"
        for acct in fam[family]:
            # ≥2 VERIFICATION_FAILED events → repeated failure is inferable from
            # event types alone (no ground truth needed).
            assert failed_by_account[acct] >= 2, f"{acct}: {failed_by_account[acct]} failures"


# --- Finding 4 (P1): temporal integrity — nothing predates creation ----------


def test_events_and_observations_never_predate_creation() -> None:
    world = generate_world(SEED, "dev")
    created = {a.account_id: a.created_at for a in world.accounts}
    for e in world.events:
        assert e.timestamp >= created[e.account_id], (
            f"{e.event_id} ({e.event_type}) predates account creation"
        )
    for o in world.linkage:
        assert o.timestamp >= created[o.account_id], (
            f"{o.observation_id} predates account creation"
        )
    # Not vacuous: the aged and sequential paths (the two that decouple creation
    # from window activity) are actually present.
    families = {fx.scenario_family for fx in world.fixtures}
    assert any(FAMILY_SPECS[f].aged for f in families), "no aged family present"
    assert any(FAMILY_SPECS[f].sequential for f in families), "no sequential family present"

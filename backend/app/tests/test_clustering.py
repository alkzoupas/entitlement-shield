"""Clustering & snapshot (Step 3) green gates:
  * expected connected components — asserted here against Step 2's declarative
    linkage fixtures (moved from Step 2, review 4 P2-6a);
  * component-size distribution — no giant component (threat §3);
  * hub poisoning falls back — a hub signal is downgraded for everyone (T1.4);
  * `network_group` / `behavioral_template` never form an edge (policy §5);
  * the edge rule (≥1 medium+, or ≥2 low of different types);
  * `as_of` excludes future / stale observations (leakage control);
  * snapshot content hash is stable; materialisation writes edges + snapshots;
  * `as_of` prefix-store identity — two independently-built stores agree (P1-4b).
"""

from datetime import UTC, datetime, timedelta, timezone

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from app.core.clock import now_utc
from app.db.engine import create_inference_schema
from app.db.ground_truth_writer import make_ground_truth_sink
from app.models.enums import EmailDomainType, Reliability, SignalType, VerificationLevel
from app.models.inference import (
    Account,
    ClusterSnapshot,
    LinkageEdge,
    LinkageObservation,
)
from app.services.clustering import (
    MAX_COMPONENT_SIZE,
    Edge,
    _content_hash,
    _utc,
    canonical_clusters,
    compute_clusters,
    materialize,
)
from app.services.generator import WORLD_END, generate_world, persist_world
from app.tests.conftest import _memory_engine

AS_OF = datetime(2025, 6, 1, tzinfo=UTC)


# --- Small synthetic-store builder -------------------------------------------


class _Builder:
    def __init__(self, session: Session) -> None:
        self.s = session
        self._obs = 0
        self._accounts: set[str] = set()

    def account(self, aid: str, created: datetime | None = None) -> None:
        if aid in self._accounts:
            return
        self._accounts.add(aid)
        self.s.add(
            Account(
                account_id=aid,
                created_at=created or (AS_OF - timedelta(days=30)),
                email_domain_type=EmailDomainType.CONSUMER,
                verification_level=VerificationLevel.EMAIL,
            )
        )

    def share(
        self,
        accounts: list[str],
        signal_type: SignalType,
        value: str,
        reliability: Reliability,
        ts: datetime | None = None,
    ) -> None:
        ts = ts or (AS_OF - timedelta(days=10))
        for aid in accounts:
            self.account(aid)
            self.s.add(
                LinkageObservation(
                    observation_id=f"o{self._obs}",
                    account_id=aid,
                    timestamp=ts,
                    signal_type=signal_type,
                    signal_value_hash=value,
                    reliability=reliability,
                )
            )
            self._obs += 1

    def flush(self) -> None:
        self.s.flush()


def _cluster_of(clusters, account_id: str) -> frozenset[str]:
    for c in clusters:
        if account_id in c.member_account_ids:
            return frozenset(c.member_account_ids)
    raise AssertionError(f"{account_id} in no cluster")


# --- Dev-world components (review 4 P2-6a) -----------------------------------


def _seed_dev(inference_engine: Engine, gt_engine: Engine):
    world = generate_world(1, "dev")
    with (
        Session(inference_engine, future=True) as inf,
        Session(gt_engine, future=True) as gt,
    ):
        persist_world(world, inf, make_ground_truth_sink(gt))
        inf.commit()
        gt.commit()
    return world


def test_dev_world_components_match_fixtures(inference_engine: Engine, gt_engine: Engine) -> None:
    world = _seed_dev(inference_engine, gt_engine)
    with Session(inference_engine, future=True) as inf:
        clusters = compute_clusters(inf, WORLD_END)
    got = {frozenset(c.member_account_ids) for c in clusters}
    expected = {frozenset(comp) for fx in world.fixtures for comp in fx.expected_components}
    assert got == expected


def test_dev_world_has_no_giant_component(inference_engine: Engine, gt_engine: Engine) -> None:
    _seed_dev(inference_engine, gt_engine)
    with Session(inference_engine, future=True) as inf:
        clusters = compute_clusters(inf, WORLD_END)
    sizes = [len(c.member_account_ids) for c in clusters]
    assert max(sizes) <= MAX_COMPONENT_SIZE
    assert len(clusters) > 1  # not one giant component swallowing the world


# --- Edge rule (policy §5) ---------------------------------------------------


def test_context_signals_never_form_an_edge(inference_session: Session) -> None:
    b = _Builder(inference_session)
    b.share(["a", "z"], SignalType.NETWORK_GROUP, "nat", Reliability.HIGH)
    b.share(["a", "z"], SignalType.BEHAVIORAL_TEMPLATE, "tmpl", Reliability.HIGH)
    b.flush()
    clusters = compute_clusters(inference_session, AS_OF)
    assert _cluster_of(clusters, "a") == frozenset({"a"})
    assert _cluster_of(clusters, "z") == frozenset({"z"})


def test_single_medium_signal_forms_edge(inference_session: Session) -> None:
    b = _Builder(inference_session)
    b.share(["a", "b"], SignalType.STABLE_DEVICE, "dev", Reliability.MEDIUM)
    b.flush()
    clusters = compute_clusters(inference_session, AS_OF)
    assert _cluster_of(clusters, "a") == frozenset({"a", "b"})


def test_single_low_signal_forms_no_edge(inference_session: Session) -> None:
    b = _Builder(inference_session)
    b.share(["a", "b"], SignalType.STABLE_DEVICE, "dev", Reliability.LOW)
    b.flush()
    clusters = compute_clusters(inference_session, AS_OF)
    assert _cluster_of(clusters, "a") == frozenset({"a"})


def test_two_low_different_types_form_edge(inference_session: Session) -> None:
    b = _Builder(inference_session)
    b.share(["a", "b"], SignalType.STABLE_DEVICE, "dev", Reliability.LOW)
    b.share(["a", "b"], SignalType.PAYMENT_TOKEN, "pay", Reliability.LOW)
    b.flush()
    clusters = compute_clusters(inference_session, AS_OF)
    assert _cluster_of(clusters, "a") == frozenset({"a", "b"})


def test_two_low_same_type_form_no_edge(inference_session: Session) -> None:
    b = _Builder(inference_session)
    b.share(["a", "b"], SignalType.STABLE_DEVICE, "dev1", Reliability.LOW)
    b.share(["a", "b"], SignalType.STABLE_DEVICE, "dev2", Reliability.LOW)
    b.flush()
    clusters = compute_clusters(inference_session, AS_OF)
    assert _cluster_of(clusters, "a") == frozenset({"a"})


# --- Hub poisoning falls back (threat T1.4) ----------------------------------


def test_hub_suppression_defeats_poisoning(inference_session: Session) -> None:
    b = _Builder(inference_session)
    # 21 accounts on one device value → a hub (> 20) → downgraded to `low` for all.
    hub_accounts = [f"h{i:02d}" for i in range(21)]
    b.share(hub_accounts, SignalType.STABLE_DEVICE, "poisoned_hub", Reliability.HIGH)
    # A separate, clean pair sharing a non-hub medium value still links.
    b.share(["c0", "c1"], SignalType.STABLE_DEVICE, "clean", Reliability.MEDIUM)
    b.flush()
    clusters = compute_clusters(inference_session, AS_OF)
    # The farm loses its own linkage: two hub accounts sharing only the poisoned
    # value (now `low`, a single type) do not form an edge.
    assert _cluster_of(clusters, "h00") == frozenset({"h00"})
    assert _cluster_of(clusters, "h01") == frozenset({"h01"})
    # The clean pair is unaffected.
    assert _cluster_of(clusters, "c0") == frozenset({"c0", "c1"})


# --- Size cap / Louvain ------------------------------------------------------


def test_louvain_splits_oversize_component(inference_session: Session) -> None:
    b = _Builder(inference_session)
    # A 60-account chain: one connected component, but each value binds only 2
    # accounts (no hub). Exceeds the 50 cap → Louvain must split it.
    n = 60
    for i in range(n - 1):
        b.share(
            [f"p{i:02d}", f"p{i + 1:02d}"],
            SignalType.STABLE_DEVICE,
            f"v{i}",
            Reliability.MEDIUM,
        )
    b.flush()
    clusters = compute_clusters(inference_session, AS_OF)
    chain = [c for c in clusters if c.member_account_ids[0].startswith("p")]
    members = {a for c in chain for a in c.member_account_ids}
    assert len(members) == n  # every account accounted for
    assert len(chain) >= 2  # the component was actually split
    assert all(len(c.member_account_ids) <= MAX_COMPONENT_SIZE for c in chain)


def test_cap_is_hard_on_unsplittable_component(inference_session: Session) -> None:
    """MAX_COMPONENT_SIZE is a HARD cap (review 2 P1-5). A 51-node complete graph
    from pair-specific non-hub medium signals is one component Louvain cannot
    reduce; the fallback partition must still keep every cluster within the cap and
    cover all 51 accounts exactly once."""
    b = _Builder(inference_session)
    n = 51
    accts = [f"k{i:02d}" for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            # A distinct value per pair → each value binds exactly 2 accounts (never
            # a hub), so no value is downgraded; the graph is complete and medium.
            b.share([accts[i], accts[j]], SignalType.STABLE_DEVICE, f"v{i}_{j}",
                    Reliability.MEDIUM)
    b.flush()
    clusters = compute_clusters(inference_session, AS_OF)
    dense = [c for c in clusters if c.member_account_ids[0].startswith("k")]
    covered = [a for c in dense for a in c.member_account_ids]
    assert sorted(covered) == accts  # every account exactly once
    assert all(len(c.member_account_ids) <= MAX_COMPONENT_SIZE for c in dense)


# --- as_of leakage control ---------------------------------------------------


def test_as_of_excludes_future_observations(inference_session: Session) -> None:
    b = _Builder(inference_session)
    b.share(["a", "b"], SignalType.STABLE_DEVICE, "future", Reliability.HIGH,
            ts=AS_OF + timedelta(days=5))
    b.flush()
    clusters = compute_clusters(inference_session, AS_OF)
    assert _cluster_of(clusters, "a") == frozenset({"a"})  # future obs ignored


def test_stale_observation_is_dropped(inference_session: Session) -> None:
    b = _Builder(inference_session)
    b.share(["a", "b"], SignalType.STABLE_DEVICE, "old", Reliability.HIGH,
            ts=AS_OF - timedelta(days=400))  # past the 365-day drop horizon
    b.flush()
    clusters = compute_clusters(inference_session, AS_OF)
    assert _cluster_of(clusters, "a") == frozenset({"a"})


# --- Snapshot hash & materialisation -----------------------------------------


def test_snapshot_hash_is_stable(inference_session: Session) -> None:
    b = _Builder(inference_session)
    b.share(["a", "b", "c"], SignalType.STABLE_DEVICE, "dev", Reliability.HIGH)
    b.flush()
    first = {c.content_hash for c in compute_clusters(inference_session, AS_OF)}
    second = {c.content_hash for c in compute_clusters(inference_session, AS_OF)}
    assert first == second  # pure function of content → reproducible


def test_materialize_writes_edges_and_snapshots(inference_session: Session) -> None:
    b = _Builder(inference_session)
    b.share(["a", "b"], SignalType.STABLE_DEVICE, "dev", Reliability.HIGH)
    b.share(["x"], SignalType.STABLE_DEVICE, "solo", Reliability.HIGH)  # singleton
    b.flush()
    clusters = compute_clusters(inference_session, AS_OF)
    materialize(inference_session, clusters, as_of=AS_OF, created_at=now_utc())
    inference_session.flush()

    assert inference_session.scalar(select(func.count()).select_from(LinkageEdge)) == 1
    n_snap = inference_session.scalar(select(func.count()).select_from(ClusterSnapshot))
    assert n_snap == len(clusters)
    # Materialised snapshot content_hash equals the computed cluster's — the
    # snapshot reproduces what the decision saw (architecture §3).
    stored = {s.content_hash for s in inference_session.scalars(select(ClusterSnapshot)).all()}
    assert stored == {c.content_hash for c in clusters}


def test_materialize_is_idempotent(inference_session: Session) -> None:
    """Re-materialising identical clusters is a true no-op (review 2 P1-6): no
    IntegrityError, no immutability violation, and the canonical rows are
    unchanged. Existing rows are skipped, never updated."""
    b = _Builder(inference_session)
    b.share(["a", "b"], SignalType.STABLE_DEVICE, "dev", Reliability.HIGH)
    b.share(["c", "d"], SignalType.PAYMENT_TOKEN, "pay", Reliability.MEDIUM)
    b.flush()
    clusters = compute_clusters(inference_session, AS_OF)

    materialize(inference_session, clusters, as_of=AS_OF, created_at=now_utc())
    inference_session.commit()
    edges_1 = sorted(
        (e.edge_id, e.account_a_id, e.account_b_id)
        for e in inference_session.scalars(select(LinkageEdge)).all()
    )
    snaps_1 = {s.content_hash for s in inference_session.scalars(select(ClusterSnapshot)).all()}

    # Second call with the SAME clusters — must not raise and must not change rows.
    materialize(inference_session, clusters, as_of=AS_OF, created_at=now_utc())
    inference_session.commit()
    edges_2 = sorted(
        (e.edge_id, e.account_a_id, e.account_b_id)
        for e in inference_session.scalars(select(LinkageEdge)).all()
    )
    snaps_2 = {s.content_hash for s in inference_session.scalars(select(ClusterSnapshot)).all()}

    assert edges_1 == edges_2
    assert snaps_1 == snaps_2
    assert len(snaps_2) == len({c.content_hash for c in clusters})


def test_materialize_rejects_as_of_mismatch(inference_session: Session) -> None:
    b = _Builder(inference_session)
    b.share(["a", "b"], SignalType.STABLE_DEVICE, "dev", Reliability.HIGH)
    b.flush()
    clusters = compute_clusters(inference_session, AS_OF)
    try:
        materialize(inference_session, clusters, as_of=AS_OF + timedelta(days=1),
                    created_at=now_utc())
    except ValueError:
        return
    raise AssertionError("expected ValueError on as_of / cluster.as_of mismatch")


# --- as_of prefix-store identity (review 4 P1-4b) ----------------------------


def _build_identity_world(session: Session, cutoff: datetime, with_future: bool) -> None:
    b = _Builder(session)
    for aid in ("x0", "x1", "x2", "y0", "y1"):
        b.account(aid, created=cutoff - timedelta(days=20))
    # Edges present at/below the cutoff.
    b.share(["x0", "x1"], SignalType.STABLE_DEVICE, "DX", Reliability.MEDIUM,
            ts=cutoff - timedelta(days=5))
    b.share(["y0", "y1"], SignalType.PAYMENT_TOKEN, "DY", Reliability.MEDIUM,
            ts=cutoff - timedelta(days=5))
    if with_future:
        # A later edge that would merge x2 in — must be excluded at as_of=cutoff.
        b.share(["x1", "x2"], SignalType.STABLE_DEVICE, "DX2", Reliability.HIGH,
                ts=cutoff + timedelta(days=5))
    b.flush()


def _persisted_canonical(session: Session) -> tuple[list[dict], list[list]]:
    """A canonical view of the PERSISTED derived artifacts (snapshots + edges),
    omitting only the nondeterministic `created_at` — the comparison surface for
    the extended as_of identity gate (review 2 P1-7)."""
    snaps = sorted(
        (
            {
                "content_hash": s.content_hash,
                "members": list(s.member_account_ids),
                "edges": s.edges,
                "evidence": s.evidence_summary,
            }
            for s in session.scalars(select(ClusterSnapshot)).all()
        ),
        key=lambda d: d["content_hash"],
    )
    edges = sorted(
        [
            e.account_a_id,
            e.account_b_id,
            e.weight,
            e.evidence_count,
            _utc(e.first_seen).isoformat(),
            _utc(e.last_seen).isoformat(),
        ]
        for e in session.scalars(select(LinkageEdge)).all()
    )
    return snaps, edges


def test_as_of_prefix_store_identity() -> None:
    cutoff = AS_OF
    prefix_engine = _memory_engine()
    full_engine = _memory_engine()
    create_inference_schema(prefix_engine)
    create_inference_schema(full_engine)

    with Session(prefix_engine, future=True) as prefix:
        _build_identity_world(prefix, cutoff, with_future=False)  # only rows ≤ T
        prefix.commit()
        prefix_computed = compute_clusters(prefix, cutoff)
        prefix_clusters = canonical_clusters(prefix_computed)
        materialize(prefix, prefix_computed, as_of=cutoff, created_at=now_utc())
        prefix.commit()
        prefix_persisted = _persisted_canonical(prefix)

    with Session(full_engine, future=True) as full:
        _build_identity_world(full, cutoff, with_future=True)  # full history
        full.commit()
        full_computed = compute_clusters(full, cutoff)
        full_clusters = canonical_clusters(full_computed)
        materialize(full, full_computed, as_of=cutoff, created_at=now_utc())
        full.commit()
        full_persisted = _persisted_canonical(full)

    # Independently rebuilt from a prefix store and a full store, the canonical
    # clustering at as_of=T is identical — the future edge is invisible to both.
    assert full_clusters == prefix_clusters
    # …and so are the PERSISTED derived artifacts: materialized edges and immutable
    # snapshots agree field-for-field (only wall-clock `created_at` is omitted).
    assert full_persisted == prefix_persisted
    merged = {frozenset(c["members"]) for c in full_clusters}
    assert frozenset({"x0", "x1"}) in merged and frozenset({"x2"}) in merged


# --- Finding 8 (P2): tz canonicalization + 365-day boundary ------------------


def test_equal_instants_different_offsets_hash_identically() -> None:
    """Two equal instants expressed in different UTC offsets canonicalize to the
    same value and therefore the same content hash (review 2 P2-8)."""
    as_of = datetime(2025, 6, 1, tzinfo=UTC)
    utc_ts = datetime(2025, 5, 1, 10, 0, tzinfo=UTC)
    off_ts = utc_ts.astimezone(timezone(timedelta(hours=5, minutes=30)))  # +05:30, same instant

    assert _utc(utc_ts) == _utc(off_ts)
    assert _utc(utc_ts).isoformat() == _utc(off_ts).isoformat()

    e_utc = Edge("a", "b", 1.0, 1, _utc(utc_ts), _utc(utc_ts))
    e_off = Edge("a", "b", 1.0, 1, _utc(off_ts), _utc(off_ts))
    assert _content_hash(_utc(as_of), ["a", "b"], [e_utc], {}) == _content_hash(
        _utc(as_of), ["a", "b"], [e_off], {}
    )


def test_stale_boundary_just_above_365_days_is_dropped(inference_session: Session) -> None:
    b = _Builder(inference_session)
    # 365 days + 1 hour old → age strictly > 365d → dropped (integer .days would keep it).
    b.share(["a", "b"], SignalType.STABLE_DEVICE, "old", Reliability.MEDIUM,
            ts=AS_OF - timedelta(days=365, hours=1))
    b.flush()
    clusters = compute_clusters(inference_session, AS_OF)
    assert _cluster_of(clusters, "a") == frozenset({"a"})  # edge dropped


def test_stale_boundary_at_and_below_365_days_is_kept(inference_session: Session) -> None:
    b = _Builder(inference_session)
    # Exactly 365 days (age == 365, not > 365) → kept.
    b.share(["a", "b"], SignalType.STABLE_DEVICE, "edge_at", Reliability.MEDIUM,
            ts=AS_OF - timedelta(days=365))
    # Just below (364d 23h) → kept.
    b.share(["c", "d"], SignalType.STABLE_DEVICE, "edge_below", Reliability.MEDIUM,
            ts=AS_OF - timedelta(days=364, hours=23))
    b.flush()
    clusters = compute_clusters(inference_session, AS_OF)
    assert _cluster_of(clusters, "a") == frozenset({"a", "b"})
    assert _cluster_of(clusters, "c") == frozenset({"c", "d"})

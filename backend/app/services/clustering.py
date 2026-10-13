"""Resource-sharing clustering & immutable `cluster_snapshot` (Phase 1 Step 3).

Builds the account-linkage graph from **resource-sharing signals only**
(`stable_device`, `payment_token`, `recovery_contact`); `network_group` and
`behavioral_template` never form an edge (policy §5, features.md §3.3). The graph
is weighted (decayed evidence), hub-suppressed, size-capped (Louvain split), and
everything is parameterised by **`as_of`** — no observation after `as_of`, and no
account created after it, may influence a decision (leakage control, eval §9).

All the numbers here are **policy §5 constants**, not tuned parameters — the hub
threshold (20), max component size (50), and the decay half-life/drop (90/365 days)
are normative, so they are fixed here rather than in the tuned feature manifest.

Outputs are canonical and deterministic: members and edges are sorted, weights are
rounded, and the `content_hash` is a stable digest of `(as_of, members, edges,
evidence)`. This is what lets the `as_of` prefix-store identity test (review 4
P1-4b) compare two independently-rebuilt stores field-for-field.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

import networkx as nx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.enums import Reliability, SignalType
from app.models.inference import Account, ClusterSnapshot, LinkageEdge, LinkageObservation

# --- Policy §5 constants ------------------------------------------------------

# Only these signal types can bind accounts into a cluster. The single source of
# truth for the edge-forming vocabulary — the generator imports this to shape its
# declarative fixtures, so the two can never drift.
RESOURCE_SHARING_SIGNALS: frozenset[SignalType] = frozenset(
    {SignalType.STABLE_DEVICE, SignalType.PAYMENT_TOKEN, SignalType.RECOVERY_CONTACT}
)

HUB_ACCOUNT_THRESHOLD = 20  # a signal value on > 20 accounts is downgraded to `low`
MAX_COMPONENT_SIZE = 50  # larger components are split by Louvain community detection
DECAY_HALFLIFE_DAYS = 90.0
MAX_AGE_DAYS = 365  # observations older than this are dropped entirely
LOUVAIN_SEED = 7  # fixed → deterministic community detection

_RANK = {Reliability.LOW: 0, Reliability.MEDIUM: 1, Reliability.HIGH: 2}
_BY_RANK = {0: Reliability.LOW, 1: Reliability.MEDIUM, 2: Reliability.HIGH}
_REL_WEIGHT = {Reliability.LOW: 0.3, Reliability.MEDIUM: 0.6, Reliability.HIGH: 1.0}


# --- Canonical outputs --------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    account_a: str  # sorted so account_a < account_b
    account_b: str
    weight: float
    evidence_count: int  # number of distinct shared signal values
    first_seen: datetime
    last_seen: datetime


@dataclass(frozen=True)
class Cluster:
    member_account_ids: tuple[str, ...]  # sorted
    edges: tuple[Edge, ...]  # sorted by (a, b)
    evidence_summary: dict
    as_of: datetime
    content_hash: str

    @property
    def snapshot_id(self) -> str:
        # Derived from content → identical clusters get identical ids across rebuilds.
        return f"snap-{self.content_hash[:16]}"


# --- Loading & weighting (as_of-bounded) --------------------------------------


@dataclass
class _Sig:
    """A resource-sharing signal value as seen by one account, after aggregation
    over that account's observations of it."""

    rel: Reliability
    first_seen: datetime
    last_seen: datetime


def _utc(dt: datetime) -> datetime:
    """Canonicalise to UTC. A naive datetime (SQLite drops tz on read) gets UTC
    attached; a tz-aware non-UTC datetime is *converted* to UTC so equal instants
    in different offsets produce identical arithmetic and isoformat output — and
    therefore identical content hashes (review 2 P2-8)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _decay(as_of: datetime, last_seen: datetime) -> float:
    age_days = max(0.0, (as_of - last_seen).total_seconds() / 86400.0)
    return 0.5 ** (age_days / DECAY_HALFLIFE_DAYS)


def _load_value_accounts(
    session: Session, as_of: datetime
) -> tuple[list[str], dict[tuple[SignalType, str], dict[str, _Sig]]]:
    """Accounts present at `as_of`, and per signal value the accounts carrying it
    (resource-sharing types only, within the age window)."""
    account_ids = sorted(
        session.scalars(select(Account.account_id).where(Account.created_at <= as_of)).all()
    )
    present = set(account_ids)

    value_accounts: dict[tuple[SignalType, str], dict[str, _Sig]] = defaultdict(dict)
    obs = session.scalars(
        select(LinkageObservation).where(LinkageObservation.timestamp <= as_of)
    ).all()
    for o in obs:
        if o.signal_type not in RESOURCE_SHARING_SIGNALS or o.account_id not in present:
            continue
        ts = _utc(o.timestamp)
        if (as_of - ts).total_seconds() / 86400.0 > MAX_AGE_DAYS:
            continue  # decayed past the drop horizon (age strictly > 365d, incl. hours)
        key = (o.signal_type, o.signal_value_hash)
        acc = value_accounts[key].get(o.account_id)
        if acc is None:
            value_accounts[key][o.account_id] = _Sig(o.reliability, ts, ts)
        else:
            # Aggregate this account's repeated observations of the same value.
            acc.rel = _BY_RANK[max(_RANK[acc.rel], _RANK[o.reliability])]
            acc.first_seen = min(acc.first_seen, ts)
            acc.last_seen = max(acc.last_seen, ts)
    return account_ids, value_accounts


def shared_signal_values(
    session: Session, as_of: datetime
) -> dict[tuple[SignalType, str], dict[str, _Sig]]:
    """Public canonical view of resource-sharing values as_of: per value, the
    **global** map of every present account carrying it (max reliability aggregated
    per account). The resource-sharing features reuse this so their hub cardinality
    is global and their effective reliability matches clustering (review finding 4)."""
    _, value_accounts = _load_value_accounts(session, as_of)
    return value_accounts


# --- Edge formation (policy §5 rule) ------------------------------------------


@dataclass
class _PairSignal:
    signal_type: SignalType
    rel: Reliability  # effective (after hub suppression), the weaker of the two accounts
    first_seen: datetime
    last_seen: datetime


def _build_edges(
    value_accounts: dict[tuple[SignalType, str], dict[str, _Sig]],
    as_of: datetime,
) -> tuple[list[Edge], dict[str, set[str]]]:
    """Apply hub suppression, then the edge rule, per account pair.

    Edge iff the pair shares ≥ 1 value at `medium`+ effective reliability, OR ≥ 2
    values of **different** signal types at `low`. Hub values (on > 20 accounts)
    are forced to `low` for everyone first — so a farm that leans on a poisoned hub
    loses that linkage (threat T1.4)."""
    pair_signals: dict[tuple[str, str], list[_PairSignal]] = defaultdict(list)
    hub_values_by_account: dict[str, set[str]] = defaultdict(set)

    for (signal_type, value), accounts in value_accounts.items():
        is_hub = len(accounts) > HUB_ACCOUNT_THRESHOLD
        members = sorted(accounts)
        if is_hub:
            for a in members:
                hub_values_by_account[a].add(value)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                sa, sb = accounts[a], accounts[b]
                eff_rank = 0 if is_hub else min(_RANK[sa.rel], _RANK[sb.rel])
                pair_signals[(a, b)].append(
                    _PairSignal(
                        signal_type=signal_type,
                        rel=_BY_RANK[eff_rank],
                        first_seen=min(sa.first_seen, sb.first_seen),
                        last_seen=max(sa.last_seen, sb.last_seen),
                    )
                )

    edges: list[Edge] = []
    for (a, b), sigs in pair_signals.items():
        has_medium_plus = any(_RANK[s.rel] >= _RANK[Reliability.MEDIUM] for s in sigs)
        low_types = {s.signal_type for s in sigs if s.rel == Reliability.LOW}
        if not (has_medium_plus or len(low_types) >= 2):
            continue
        weight = round(
            sum(_decay(as_of, s.last_seen) * _REL_WEIGHT[s.rel] for s in sigs), 6
        )
        edges.append(
            Edge(
                account_a=a,
                account_b=b,
                weight=weight,
                evidence_count=len(sigs),
                first_seen=min(s.first_seen for s in sigs),
                last_seen=max(s.last_seen for s in sigs),
            )
        )
    edges.sort(key=lambda e: (e.account_a, e.account_b))
    return edges, hub_values_by_account


# --- Components, size cap, snapshots ------------------------------------------


def _components(account_ids: list[str], edges: list[Edge]) -> list[list[str]]:
    graph = nx.Graph()
    graph.add_nodes_from(account_ids)  # sorted → isolated accounts become singletons
    for e in edges:
        graph.add_edge(e.account_a, e.account_b, weight=e.weight)

    clusters: list[list[str]] = []
    for component in nx.connected_components(graph):
        members = sorted(component)
        if len(members) <= MAX_COMPONENT_SIZE:
            clusters.append(members)
        else:
            clusters.extend(_louvain_split(graph.subgraph(members)))
    clusters.sort(key=lambda m: (m[0], len(m)))
    return clusters


def _chunk_partition(members: list[str]) -> list[list[str]]:
    """Deterministically slice sorted members into contiguous chunks each within
    the cap — the hard-cap fallback (review 2 P1-5) when Louvain can neither reduce
    a community nor recurse further."""
    return [members[i : i + MAX_COMPONENT_SIZE] for i in range(0, len(members), MAX_COMPONENT_SIZE)]


def _louvain_split(subgraph: nx.Graph, depth: int = 0) -> list[list[str]]:
    """Split an over-size component into communities, recursing until each is
    within the cap. MAX_COMPONENT_SIZE is a HARD cap: when a community cannot be
    reduced (Louvain made no progress) or recursion is exhausted (depth ≥ 8), the
    oversize community is deterministically chunk-partitioned rather than returned
    intact (review 2 P1-5)."""
    communities = nx.community.louvain_communities(
        subgraph, weight="weight", seed=LOUVAIN_SEED
    )
    result: list[list[str]] = []
    for community in communities:
        members = sorted(community)
        if len(members) <= MAX_COMPONENT_SIZE:
            result.append(members)
        elif depth >= 8 or len(members) == subgraph.number_of_nodes():
            result.extend(_chunk_partition(members))  # unsplittable → hard partition
        else:
            result.extend(_louvain_split(subgraph.subgraph(members), depth + 1))
    return result


def _content_hash(as_of: datetime, members: list[str], edges: list[Edge], evidence: dict) -> str:
    payload = {
        "as_of": as_of.isoformat(),
        "members": members,
        "edges": [
            [e.account_a, e.account_b, e.weight, e.evidence_count,
             e.first_seen.isoformat(), e.last_seen.isoformat()]
            for e in edges
        ],
        "evidence": evidence,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def compute_clusters(session: Session, as_of: datetime) -> list[Cluster]:
    """The resource-sharing clusters visible at `as_of` — every present account in
    exactly one cluster (singletons included). Pure w.r.t. the store: reads only
    rows at or before `as_of`."""
    as_of = _utc(as_of)
    account_ids, value_accounts = _load_value_accounts(session, as_of)
    edges, hub_values_by_account = _build_edges(value_accounts, as_of)

    edges_by_account: dict[str, list[Edge]] = defaultdict(list)
    for e in edges:
        edges_by_account[e.account_a].append(e)
        edges_by_account[e.account_b].append(e)

    # Signal types touching each account (for the evidence summary).
    types_by_account: dict[str, set[str]] = defaultdict(set)
    for (signal_type, _value), accounts in value_accounts.items():
        for a in accounts:
            types_by_account[a].add(signal_type.value)

    clusters: list[Cluster] = []
    for members in _components(account_ids, edges):
        member_set = set(members)
        cluster_edges = sorted(
            {
                e
                for a in members
                for e in edges_by_account[a]
                if e.account_a in member_set and e.account_b in member_set
            },
            key=lambda e: (e.account_a, e.account_b),
        )
        evidence = {
            "member_count": len(members),
            "edge_count": len(cluster_edges),
            "signal_types": sorted({t for a in members for t in types_by_account[a]}),
            "hub_suppressed_value_count": len(
                {v for a in members for v in hub_values_by_account.get(a, set())}
            ),
        }
        clusters.append(
            Cluster(
                member_account_ids=tuple(members),
                edges=tuple(cluster_edges),
                evidence_summary=evidence,
                as_of=as_of,
                content_hash=_content_hash(as_of, members, cluster_edges, evidence),
            )
        )
    return clusters


# --- Materialisation ----------------------------------------------------------


def materialize(
    session: Session,
    clusters: list[Cluster],
    as_of: datetime,
    created_at: datetime,
) -> None:
    """Write materialized `linkage_edges` and immutable `cluster_snapshots`.

    Edges are deduplicated across clusters and keyed by `(as_of, pair)`; snapshots
    are keyed by content. Materialisation is **idempotent** (review 2 P1-6): a row
    whose primary key is already present is SKIPPED, never updated — updating a
    snapshot would trip the immutability guard, and the canonical content is
    identical anyway. So re-materialising identical clusters is a true no-op.

    `as_of` must match every `cluster.as_of` (the edge id is keyed on it); a
    mismatch is a wiring error and raises rather than silently keying edges to the
    wrong instant."""
    as_of = _utc(as_of)
    mismatched = [c.content_hash for c in clusters if _utc(c.as_of) != as_of]
    if mismatched:
        raise ValueError(
            f"materialize as_of {as_of.isoformat()} does not match cluster.as_of "
            f"for snapshots {mismatched}"
        )

    seen_edges: set[str] = set()
    for cluster in clusters:
        for e in cluster.edges:
            edge_id = "edge-" + hashlib.sha256(
                f"{as_of.isoformat()}|{e.account_a}|{e.account_b}".encode()
            ).hexdigest()[:16]
            if edge_id in seen_edges:
                continue
            seen_edges.add(edge_id)
            if session.get(LinkageEdge, edge_id) is not None:
                continue  # already materialised (idempotent) — never re-insert
            session.add(
                LinkageEdge(
                    edge_id=edge_id,
                    account_a_id=e.account_a,
                    account_b_id=e.account_b,
                    weight=e.weight,
                    evidence_count=e.evidence_count,
                    first_seen=e.first_seen,
                    last_seen=e.last_seen,
                )
            )
        if session.get(ClusterSnapshot, cluster.snapshot_id) is not None:
            continue  # snapshot content is immutable and identical — skip, never update
        session.add(
            ClusterSnapshot(
                snapshot_id=cluster.snapshot_id,
                as_of=cluster.as_of,
                member_account_ids=list(cluster.member_account_ids),
                edges=[
                    [e.account_a, e.account_b, e.weight, e.evidence_count,
                     e.first_seen.isoformat(), e.last_seen.isoformat()]
                    for e in cluster.edges
                ],
                evidence_summary=cluster.evidence_summary,
                content_hash=cluster.content_hash,
                created_at=created_at,
            )
        )
    session.flush()


def canonical_clusters(clusters: list[Cluster]) -> list[dict]:
    """Field-for-field canonical view (no DB ids, no wall-clock `created_at`) — the
    comparison surface for the `as_of` prefix-store identity test (P1-4b)."""
    return sorted(
        (
            {
                "content_hash": c.content_hash,
                "members": list(c.member_account_ids),
                "edges": [
                    [e.account_a, e.account_b, e.weight, e.evidence_count,
                     e.first_seen.isoformat(), e.last_seen.isoformat()]
                    for e in c.edges
                ],
                "evidence": c.evidence_summary,
            }
            for c in clusters
        ),
        key=lambda d: d["content_hash"],
    )

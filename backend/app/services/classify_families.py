"""Anticipated / unanticipated family classifier (docs/features.md §7). Step 7.

Applies the objective §7.1 rule to each scenario family: a family is **anticipated**
iff every feature that fires on it (contribution ≥ 0.05) already existed at the
`features-frozen` commit; it is **unanticipated** if it fires a feature added later
**or** fires no feature at all.

Leakage boundary (features.md §7.3): this module reads the scenario **family** only —
via the family-only `ground_truth_family_reader`, never the binary-label sealed
reader. That exemption is what the capability-boundary test allows for
`classify_families`, and the pure `classify_families()` below takes already-joined
rows, so it touches no ground truth at all.

The pure `classify_families()` was built and tested on dev fixtures in Phase 1;
`build_family_tuples()` + `render_family_mapping_markdown()` (P2.2) run it over a
loaded evaluation world (the Tier-2 stores) and render `docs/family-mapping.md`.
Both stay inside the §7.3 boundary: the only ground-truth surface is the family-only
reader (`load_account_families`), never the binary label.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.db.ground_truth_family_reader import load_account_families, load_scenario_families
from app.services.clustering import compute_clusters
from app.services.features import (
    FEATURE_ORDER,
    Rung,
    enumerate_cluster_cycles,
    score_cluster_cycle,
)

#: The feature ids that exist "at freeze". In Phase 1 the whole set exists; a caller
#: (or a test) may pass a narrower frozen set to model a post-freeze feature.
FROZEN_FEATURE_IDS: frozenset[str] = frozenset(FEATURE_ORDER)

ANTICIPATED = "anticipated"
UNANTICIPATED = "unanticipated"


@dataclass(frozen=True)
class FamilyMapping:
    scenario_family: str
    label: str  # ANTICIPATED | UNANTICIPATED
    firing_features: tuple[str, ...]  # sorted
    rationale: str


def classify_families(
    tagged_rows: Iterable[tuple[str, str, bool]],
    *,
    frozen_features: frozenset[str] = FROZEN_FEATURE_IDS,
) -> list[FamilyMapping]:
    """Classify each scenario family from `(scenario_family, feature_id, fired)` rows
    — the contribution export already joined to its family. Deterministic: families
    and firing features are emitted in sorted order.

    A family firing nothing is unanticipated by construction (the frozen vocabulary
    is blind to it — the §7.1 'fires nothing' clause)."""
    firing: dict[str, set[str]] = defaultdict(set)
    families: set[str] = set()
    for family, feature_id, fired in tagged_rows:
        families.add(family)
        if fired:
            firing[family].add(feature_id)

    out: list[FamilyMapping] = []
    for family in sorted(families):
        feats = firing.get(family, set())
        if not feats:
            label, rationale = UNANTICIPATED, "fires no feature at or above 0.05"
        elif feats <= frozen_features:
            label, rationale = ANTICIPATED, "every firing feature existed at features-frozen"
        else:
            novel = sorted(feats - frozen_features)
            label = UNANTICIPATED
            rationale = f"fires feature(s) added after freeze: {', '.join(novel)}"
        out.append(
            FamilyMapping(
                scenario_family=family,
                label=label,
                firing_features=tuple(sorted(feats)),
                rationale=rationale,
            )
        )
    return out


def family_labels(gt_family_session: Session) -> dict[str, str]:
    """The scenario-family label per cluster — the ONLY ground-truth access here, via
    the family-only reader (never the binary abuse label, features.md §7.3)."""
    return load_scenario_families(gt_family_session)


def build_family_tuples(
    inf_session: Session,
    gt_family_session: Session,
    as_of: datetime,
    *,
    rung: Rung = Rung.S2,
    coverage: dict[str, dict[str, int]] | None = None,
) -> list[tuple[str, str, bool]]:
    """Run the frozen feature stack over a loaded world and emit
    `(scenario_family, feature_id, fired)` rows for `classify_families`.

    Family-only join (features.md §7.3): clusters and firing come from the inference
    plane; each unit is attributed to every represented family from the family-only
    reader — the binary abuse label is never read. Scores at S2 by default so the full
    frozen vocabulary can fire. Deterministic in the ordering of `compute_clusters` /
    `enumerate_cluster_cycles` / `FEATURE_ORDER`.
    """
    account_family = load_account_families(gt_family_session)
    families = sorted(set(load_scenario_families(gt_family_session).values()))
    # Non-firing sentinels retain absent/no-fire families, including a historical
    # cutoff before their first account exists. They cannot add a firing feature.
    tuples: list[tuple[str, str, bool]] = [(family, FEATURE_ORDER[0], False) for family in families]
    counts = {family: {"clusters": 0, "mixed_clusters": 0, "units": 0} for family in families}
    for cluster in compute_clusters(inf_session, as_of):
        members = list(cluster.member_account_ids)
        represented = sorted({account_family[m] for m in members if m in account_family})
        for family in represented:
            counts[family]["clusters"] += 1
            counts[family]["mixed_clusters"] += int(len(represented) > 1)
        for unit in enumerate_cluster_cycles(inf_session, cluster.snapshot_id, members, as_of):
            score = score_cluster_cycle(inf_session, unit, members, rung)
            for family in represented:
                counts[family]["units"] += 1
                for contribution in score.contributions:
                    tuples.append((family, contribution.feature_id, contribution.fired))
    if coverage is not None:
        coverage.clear()
        coverage.update(counts)
    return tuples


def render_family_mapping_markdown(
    mappings: list[FamilyMapping],
    *,
    manifest_hash: str,
    manifest_json: dict,
    as_of: datetime,
    coverage: dict[str, dict[str, int]] | None = None,
) -> str:
    """Render `docs/family-mapping.md` — byte-deterministic (no wall-clock).

    The header stamps the freeze (`feature-manifest` hash) and the world it was run
    over (version + actual `as_of` + the Tier-2 manifest checksum) so the file
    is reproducible and regeneration-testable; the same-model-family caveat is printed
    verbatim. One table row per family (sorted), family-only per §7.3 — no abuse class.
    """
    world_version = str(manifest_json.get("world_version", "unknown"))
    world_end = str(manifest_json.get("world_end", "unknown"))
    author_model = str(manifest_json.get("author_model", "unknown"))
    as_of = as_of.replace(tzinfo=UTC) if as_of.tzinfo is None else as_of.astimezone(UTC)
    world_checksum = _manifest_self_checksum(manifest_json)

    anticipated = sum(1 for m in mappings if m.label == ANTICIPATED)
    unanticipated = [m for m in mappings if m.label == UNANTICIPATED]

    lines: list[str] = [
        "# Tier 2 family mapping — anticipated vs. unanticipated",
        "",
        "Generated by `scripts/classify_families.py --from-stores` (features.md §7.2).",
        "Each family is **anticipated** iff every feature that fires on it (contribution",
        "≥ 0.05) already existed at the `features-frozen` commit; **unanticipated** if it",
        "fires a post-freeze feature *or* fires nothing at all (§7.1).",
        "",
        "**Boundary (§7.3):** the classifier reads the scenario *family* only, never the",
        "binary abuse label — so the legitimate/ambiguous/abusive class is deliberately",
        "absent here.",
        "",
        "| field | value |",
        "|---|---|",
        f"| feature-manifest hash | `{manifest_hash}` |",
        f"| world version | `{world_version}` |",
        f"| as_of | `{as_of.isoformat()}` |",
        f"| world_end | `{world_end}` |",
        f"| Tier 2 manifest checksum | `{world_checksum}` |",
        f"| families | {len(mappings)} "
        f"({anticipated} anticipated, {len(unanticipated)} unanticipated) |",
        "",
        f"> **Reporting caveat.** World author: {author_model}.",
        ">",
        "> Metrics are NOT reportable as cross-family until a genuine cross-family author",
        "> regenerates the world (docs/tier2-decisions.md). The family × signal-family",
        "> timing-lift cross (§7.4) is not shown here.",
        "> A repaired world is a detector-aware maintenance derivative; its original",
        "> author-only information barrier is not claimed to remain intact.",
        "",
        "| scenario family | verdict | firing features | rationale |",
        "|---|---|---|---|",
    ]
    for m in mappings:
        feats = ", ".join(m.firing_features) if m.firing_features else "—"
        lines.append(f"| {m.scenario_family} | {m.label} | {feats} | {m.rationale} |")
    if coverage is not None:
        lines.extend([
            "", "Cluster-level coverage (each mixed cluster is attributed to every represented",
            "family; this does not mean each member caused its firing features).",
            "", "| scenario family | clusters | mixed clusters | units |", "|---|---|---|---|",
        ])
        for family, counts in sorted(coverage.items()):
            lines.append(f"| {family} | {counts['clusters']} | {counts['mixed_clusters']} | "
                         f"{counts['units']} |")
    lines.append("")
    return "\n".join(lines)


def _manifest_self_checksum(manifest_json: dict) -> str:
    """A stable digest of the Tier-2 manifest's own parquet checksums (order-independent),
    so the header pins the exact world without embedding the whole checksum block."""
    checksums = manifest_json.get("checksums", {})
    payload = json.dumps(checksums, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]

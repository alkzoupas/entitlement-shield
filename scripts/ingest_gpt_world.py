"""Strictly ingest a repo-blind GPT-authored Tier 3 or Tier 4 world.

The supplied JSON is copied unchanged before validation.  Tier 4
(`docs/tier4-gpt-brief.md`) adds appeal text on `appeal` events and a sealed
`appeal_truth` block: text is split into `appeals.parquet` (inference plane) and the
truth into `appeal_truth.parquet` (ground-truth plane); the generator program and
its self-check, when relayed beside the JSON, are preserved and checksummed.

This script only performs mechanical timestamp parsing and parquet serialization;
it never derives or repairs authoritative cycle totals, membership, or labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from app.db.tier2_loader import load_tier2

WORLD_END = "2026-06-01T00:00:00Z"
FRAME_START = "2025-12-01T00:00:00Z"
ARTIFACTS = (
    "accounts",
    "account_events",
    "linkage_observations",
    "entitlement_cycles",
    "ground_truth",
    "cluster_membership",
)
TOP_LEVEL = {"world_version", "author_model", "authoring_notes", "families", *ARTIFACTS}
ROW_FIELDS = {
    "families": {"name", "description", "label", "why_hard"},
    "accounts": {"account_id", "created_at", "email_domain_type", "verification_level"},
    "account_events": {
        "event_id", "account_id", "timestamp", "event_type", "credits_amount",
        "endpoint_cost_tier",
    },
    "linkage_observations": {
        "observation_id", "account_id", "timestamp", "signal_type", "signal_value",
        "reliability",
    },
    "entitlement_cycles": {
        "cycle_id", "account_id", "cycle_start", "cycle_end", "credits_granted",
        "credits_consumed", "credits_expired",
    },
    "ground_truth": {"cluster_id", "scenario_family", "label", "notes"},
    "cluster_membership": {"account_id", "cluster_id"},
}
ENUMS: dict[str, set[object]] = {
    "email_domain_type": {"consumer", "disposable", "education", "enterprise"},
    "verification_level": {"none", "email", "phone", "enhanced"},
    "event_type": {
        "signup",
        "claim_credits",
        "consume_credits",
        "login",
        "appeal",
        "verification",
        "verification_failed",
    },
    "endpoint_cost_tier": {"low", "high", None},
    "signal_type": {
        "stable_device",
        "payment_token",
        "recovery_contact",
        "network_group",
        "behavioral_template",
    },
    "reliability": {"low", "medium", "high"},
    "label": {"abusive", "legitimate", "ambiguous"},
}
TIMESTAMPS = {
    "accounts": ("created_at",),
    "account_events": ("timestamp",),
    "linkage_observations": ("timestamp",),
    "entitlement_cycles": ("cycle_start", "cycle_end"),
}


TIER4_ROW_FIELDS = {
    **ROW_FIELDS,
    "families": {"name", "description", "label", "misleading_signals", "revealing_combination"},
    "account_events": ROW_FIELDS["account_events"] | {"appeal_text"},
    "appeal_truth": {"event_id", "kind", "notes"},
}
APPEAL_KINDS = {"honest", "dishonest", "manipulation"}
APPEAL_TEXT_MAX_CHARS = 1000
# Created-at may pre-date the frame by up to two years (brief Requirement 1).
PRE_FRAME_ALLOWANCE = timedelta(days=2 * 366)
COMPANIONS = ("generator.py", "self-check.json", "appeal-texts.json", "appeal-facts.json")


@dataclass(frozen=True)
class Profile:
    tier: str
    schema_version: str
    artifacts: tuple[str, ...]
    top_level: frozenset[str]
    row_fields: dict[str, set[str]]
    family_text_fields: tuple[str, ...]
    clusters: tuple[int, int]
    accounts: tuple[int, int]
    abusive_fraction: tuple[float, float]
    min_family_clusters: int


PROFILES = {
    "tier3": Profile(
        tier="tier3",
        schema_version="tier3-gpt-brief-v1.0",
        artifacts=ARTIFACTS,
        top_level=frozenset(TOP_LEVEL),
        row_fields=ROW_FIELDS,
        family_text_fields=("description", "why_hard"),
        clusters=(30, 60),
        accounts=(150, 400),
        abusive_fraction=(0.05, 0.15),
        min_family_clusters=3,
    ),
    "tier4": Profile(
        tier="tier4",
        schema_version="tier4-gpt-brief-v0.1",
        artifacts=(*ARTIFACTS, "appeal_truth"),
        top_level=frozenset(TOP_LEVEL | {"generator_seed", "appeal_truth"}),
        row_fields=TIER4_ROW_FIELDS,
        family_text_fields=("description", "misleading_signals", "revealing_combination"),
        clusters=(150, 250),
        accounts=(800, 1500),
        # "About 15–20%" in the brief; a small tolerance, never class balancing.
        abusive_fraction=(0.12, 0.23),
        min_family_clusters=10,
    ),
}


class Tier3ValidationError(ValueError):
    """The source is invalid and must be corrected by its original author."""


def _parse_time(value: object, path: str, *, allow_after_end: bool = False) -> datetime:
    if not isinstance(value, str):
        raise Tier3ValidationError(f"{path} must be an ISO-8601 UTC string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Tier3ValidationError(f"{path} is not ISO-8601: {value!r}") from exc
    offset = parsed.utcoffset()
    if parsed.tzinfo is None or offset is None:
        raise Tier3ValidationError(f"{path} must include a UTC offset")
    if offset.total_seconds() != 0:
        raise Tier3ValidationError(f"{path} must use a zero UTC offset")
    parsed = parsed.astimezone(UTC)
    if not allow_after_end and parsed > datetime.fromisoformat(WORLD_END.replace("Z", "+00:00")):
        raise Tier3ValidationError(f"{path} exceeds pinned world_end {WORLD_END}")
    return parsed


def _records(
    source: dict[str, Any], name: str, row_fields: dict[str, set[str]] = ROW_FIELDS
) -> list[dict[str, Any]]:
    value = source.get(name)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise Tier3ValidationError(f"{name} must be an array of objects")
    if not value:
        raise Tier3ValidationError(f"{name} must not be empty")
    expected = row_fields[name]
    for index, row in enumerate(value):
        if set(row) != expected:
            raise Tier3ValidationError(
                f"{name}[{index}] keys must be exactly {sorted(expected)}"
            )
    return value


def _unique(rows: list[dict[str, Any]], field: str, name: str) -> set[str]:
    values: list[str] = []
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise Tier3ValidationError(f"{name}.{field} must contain non-empty strings")
        values.append(value)
    if len(set(values)) != len(values):
        raise Tier3ValidationError(f"{name}.{field} contains duplicates")
    return set(values)


def validate(source: dict[str, Any], tier: str = "tier3") -> dict[str, list[dict[str, Any]]]:
    profile = PROFILES[tier]
    if set(source) != profile.top_level:
        raise Tier3ValidationError(
            f"top-level keys must be exactly {sorted(profile.top_level)}"
        )
    if not isinstance(source["world_version"], str) or not source["world_version"]:
        raise Tier3ValidationError("world_version must be non-empty")
    if not isinstance(source["author_model"], str) or not source["author_model"]:
        raise Tier3ValidationError("author_model must be non-empty")
    if not isinstance(source["authoring_notes"], str):
        raise Tier3ValidationError("authoring_notes must be a string")
    if tier == "tier4" and type(source["generator_seed"]) is not int:
        raise Tier3ValidationError("generator_seed must be an integer")
    tables = {name: _records(source, name, profile.row_fields) for name in profile.artifacts}
    families = _records(source, "families", profile.row_fields)
    low_clusters, high_clusters = profile.clusters
    low_accounts, high_accounts = profile.accounts
    if (
        not low_clusters <= len(tables["ground_truth"]) <= high_clusters
        or not low_accounts <= len(tables["accounts"]) <= high_accounts
    ):
        raise Tier3ValidationError(
            f"{tier} requires {low_clusters}–{high_clusters} clusters and "
            f"{low_accounts}–{high_accounts} accounts"
        )
    family_names = _unique(families, "name", "families")
    for row in families:
        if row.get("label") not in ENUMS["label"] or not all(
            isinstance(row.get(k), str) and row[k] for k in profile.family_text_fields
        ):
            raise Tier3ValidationError(
                f"every family needs non-empty {'/'.join(profile.family_text_fields)} "
                "and a valid label"
            )
    if tier == "tier4":
        _validate_tier4_family_mix(families)

    frame_start = _parse_time(FRAME_START, "frame_start")
    accounts = _unique(tables["accounts"], "account_id", "accounts")
    account_created: dict[str, datetime] = {}
    for row in tables["accounts"]:
        if (
            row.get("email_domain_type") not in ENUMS["email_domain_type"]
            or row.get("verification_level") not in ENUMS["verification_level"]
        ):
            raise Tier3ValidationError(f"invalid account enum for {row['account_id']}")
        created = _parse_time(row.get("created_at"), f"accounts[{row['account_id']}].created_at")
        if tier == "tier4" and created < frame_start - PRE_FRAME_ALLOWANCE:
            raise Tier3ValidationError("created_at pre-dates the frame by more than two years")
        account_created[str(row["account_id"])] = created
    for name, id_field, enum_field in (
        ("account_events", "event_id", "event_type"),
        ("linkage_observations", "observation_id", "signal_type"),
        ("entitlement_cycles", "cycle_id", None),
    ):
        _unique(tables[name], id_field, name)
        for row in tables[name]:
            if row.get("account_id") not in accounts:
                raise Tier3ValidationError(f"{name} references unknown account")
            if enum_field and row.get(enum_field) not in ENUMS[enum_field]:
                raise Tier3ValidationError(f"invalid {enum_field} in {name}")
            for field in TIMESTAMPS.get(name, ()):
                # Tier 4 permits open cycles that end after world_end.
                timestamp = _parse_time(
                    row.get(field),
                    f"{name}.{field}",
                    allow_after_end=tier == "tier4" and field == "cycle_end",
                )
                # Tier 4 cycles are calendar-aligned to the frame (authoring notes),
                # so an account's first cycle may start before, but must end after,
                # its creation.
                calendar_start = tier == "tier4" and field == "cycle_start"
                if not calendar_start and timestamp < account_created[str(row["account_id"])]:
                    raise Tier3ValidationError(
                        f"{name}.{field} precedes the account creation time"
                    )
                if tier == "tier4" and timestamp < frame_start:
                    raise Tier3ValidationError(f"{name}.{field} precedes the frame start")
    for row in tables["account_events"]:
        amount = row.get("credits_amount")
        if amount is not None and (type(amount) is not int or amount < 0):
            raise Tier3ValidationError("credits_amount must be null or a nonnegative integer")
        if row["event_type"] in {"claim_credits", "consume_credits"} and amount is None:
            raise Tier3ValidationError("credit events require a nonnegative credits_amount")
        if row.get("endpoint_cost_tier") not in ENUMS["endpoint_cost_tier"]:
            raise Tier3ValidationError("invalid endpoint_cost_tier")
    for row in tables["linkage_observations"]:
        if (
            row.get("reliability") not in ENUMS["reliability"]
            or not isinstance(row.get("signal_value"), str)
            or not row["signal_value"]
        ):
            raise Tier3ValidationError("invalid linkage reliability or signal_value")

    allow_open = tier == "tier4"
    cycles = tables["entitlement_cycles"]
    for row in cycles:
        start, end = (
            _parse_time(row["cycle_start"], "cycle_start"),
            _parse_time(row["cycle_end"], "cycle_end", allow_after_end=allow_open),
        )
        if start >= end:
            raise Tier3ValidationError("cycle_start must precede cycle_end")
        if any(
            type(row.get(field)) is not int or row[field] < 0
            for field in ("credits_granted", "credits_consumed", "credits_expired")
        ):
            raise Tier3ValidationError("cycle totals must be nonnegative integers")
    clusters = _unique(tables["ground_truth"], "cluster_id", "ground_truth")
    family_counts: Counter[str] = Counter()
    family_labels = {str(row["name"]): row["label"] for row in families}
    for row in tables["ground_truth"]:
        if (
            row.get("scenario_family") not in family_names
            or row.get("label") not in ENUMS["label"]
            or not isinstance(row.get("notes"), str)
        ):
            raise Tier3ValidationError("invalid ground_truth family, label, or notes")
        family_counts[str(row["scenario_family"])] += 1
        if row["label"] != family_labels[str(row["scenario_family"])]:
            raise Tier3ValidationError("ground_truth label contradicts its family label")
    minimum = profile.min_family_clusters
    if set(family_counts) != family_names or any(
        count < minimum for count in family_counts.values()
    ):
        raise Tier3ValidationError(
            f"every declared family must contain at least {minimum} clusters"
        )
    abusive_fraction = sum(
        row["label"] == "abusive" for row in tables["ground_truth"]
    ) / len(tables["ground_truth"])
    low_fraction, high_fraction = profile.abusive_fraction
    if not low_fraction <= abusive_fraction <= high_fraction:
        raise Tier3ValidationError(
            f"abusive clusters must comprise {low_fraction:.0%}–{high_fraction:.0%} of clusters"
        )
    membership_accounts = _unique(tables["cluster_membership"], "account_id", "cluster_membership")
    if membership_accounts != accounts:
        raise Tier3ValidationError("cluster_membership must map every account exactly once")
    if any(row.get("cluster_id") not in clusters for row in tables["cluster_membership"]):
        raise Tier3ValidationError("cluster_membership references unknown cluster")
    represented_clusters = {str(row["cluster_id"]) for row in tables["cluster_membership"]}
    if represented_clusters != clusters:
        raise Tier3ValidationError("cluster_membership must represent every cluster")

    cycles_by_account: dict[str, list[dict[str, Any]]] = {}
    cycle_totals: dict[str, list[int]] = {}
    for row in cycles:
        cycles_by_account.setdefault(str(row["account_id"]), []).append(row)
        cycle_totals[str(row["cycle_id"])] = [0, 0]
    for account_cycles in cycles_by_account.values():
        account_cycles.sort(key=lambda row: _parse_time(row["cycle_start"], "cycle_start"))
        previous_end: datetime | None = None
        for row in account_cycles:
            start = _parse_time(row["cycle_start"], "cycle_start")
            end = _parse_time(row["cycle_end"], "cycle_end", allow_after_end=allow_open)
            if previous_end is not None and previous_end > start:
                raise Tier3ValidationError("an account's entitlement cycles must not overlap")
            previous_end = end
    for row in tables["account_events"]:
        if row["event_type"] not in {"claim_credits", "consume_credits"}:
            continue
        timestamp = _parse_time(row["timestamp"], "account_events.timestamp")
        matches = [
            cycle for cycle in cycles_by_account.get(str(row["account_id"]), [])
            if _parse_time(cycle["cycle_start"], "cycle_start")
            <= timestamp
            < _parse_time(cycle["cycle_end"], "cycle_end", allow_after_end=allow_open)
        ]
        if len(matches) != 1:
            raise Tier3ValidationError("credit event must belong to exactly one cycle")
        total_index = 0 if row["event_type"] == "claim_credits" else 1
        cycle_totals[str(matches[0]["cycle_id"])][total_index] += row["credits_amount"]
    world_end = _parse_time(WORLD_END, "world_end")
    for row in cycles:
        granted, consumed = cycle_totals[str(row["cycle_id"])]
        if granted != row["credits_granted"] or consumed != row["credits_consumed"]:
            raise Tier3ValidationError("cycle totals disagree with credit events")
        if not 0 <= consumed <= granted:
            raise Tier3ValidationError("cycle consumption must be between zero and granted")
        closed = (
            _parse_time(row["cycle_end"], "cycle_end", allow_after_end=allow_open) <= world_end
        )
        expected_expired = granted - consumed if closed else 0
        if row["credits_expired"] != expected_expired:
            raise Tier3ValidationError("cycle expiry disagrees with its authoritative window")
    if tier == "tier4":
        _validate_appeals(tables, families, clusters)
    return tables


def _validate_tier4_family_mix(families: list[dict[str, Any]]) -> None:
    labels = Counter(str(row["label"]) for row in families)
    if len(families) < 10 or labels["legitimate"] < 4 or labels["abusive"] < 2 or (
        labels["ambiguous"] < 1
    ):
        raise Tier3ValidationError(
            "tier4 needs ≥10 families: ≥4 legitimate, ≥2 abusive, ≥1 ambiguous"
        )


def _validate_appeals(
    tables: dict[str, list[dict[str, Any]]],
    families: list[dict[str, Any]],
    clusters: set[str],
) -> None:
    """Appeal text only on appeal events, bounded, truth-complete and label-free."""
    appeal_ids: set[str] = set()
    for row in tables["account_events"]:
        text = row.get("appeal_text")
        if row["event_type"] != "appeal":
            if text is not None:
                raise Tier3ValidationError("appeal_text must be null on non-appeal events")
            continue
        if not isinstance(text, str) or not text.strip():
            raise Tier3ValidationError("appeal events require non-empty appeal_text")
        if len(text) > APPEAL_TEXT_MAX_CHARS:
            raise Tier3ValidationError(
                f"appeal_text exceeds {APPEAL_TEXT_MAX_CHARS} characters"
            )
        if row.get("credits_amount") is not None or row.get("endpoint_cost_tier") is not None:
            raise Tier3ValidationError("appeal events carry no credits or cost tier")
        appeal_ids.add(str(row["event_id"]))
    truth_ids = _unique(tables["appeal_truth"], "event_id", "appeal_truth")
    if truth_ids != appeal_ids:
        raise Tier3ValidationError("every appeal event needs exactly one appeal_truth row")
    for row in tables["appeal_truth"]:
        if row.get("kind") not in APPEAL_KINDS or not isinstance(row.get("notes"), str):
            raise Tier3ValidationError("invalid appeal_truth kind or notes")
    # Sealed vocabulary must not leak into inference-plane text (brief Requirement 6).
    sealed_terms = [
        *(str(label) for label in ENUMS["label"]),
        *APPEAL_KINDS,
        *(str(row["name"]) for row in families),
    ]
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(str(term)) for term in sealed_terms) + r")\b",
        re.IGNORECASE,
    )
    for row in tables["account_events"]:
        text = row.get("appeal_text")
        if text is None:
            continue
        if pattern.search(text) or any(cluster in text for cluster in clusters):
            raise Tier3ValidationError(
                f"appeal {row['event_id']} text contains a label, family name or cluster id"
            )


def _frame(rows: list[dict[str, Any]], name: str) -> pl.DataFrame:
    frame = pl.DataFrame(rows, infer_schema_length=None)
    for column in TIMESTAMPS.get(name, ()):
        frame = frame.with_columns(pl.col(column).str.to_datetime(time_zone="UTC", strict=True))
    return frame


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tier4_frames(tables: dict[str, list[dict[str, Any]]]) -> dict[str, pl.DataFrame]:
    """Split appeal text out of the event rows: the inference `account_events`
    schema has no text column, and text lives only in the `appeals` table."""
    events = _frame(tables["account_events"], "account_events")
    appeals = (
        events.filter(pl.col("event_type") == "appeal")
        .select(
            pl.col("event_id").alias("appeal_id"),
            "account_id",
            pl.col("timestamp").alias("submitted_at"),
            pl.col("appeal_text").alias("text"),
        )
    )
    return {
        "account_events": events.drop("appeal_text"),
        "appeals": appeals,
        "appeal_truth": _frame(tables["appeal_truth"], "appeal_truth"),
    }


def ingest(
    source_path: Path, output_dir: Path, *, load: bool, tier: str = "tier3"
) -> dict[str, int] | None:
    profile = PROFILES[tier]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise Tier3ValidationError(f"refusing to overwrite non-empty output directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source_path, output_dir / "gpt-world.json")
    companions: dict[str, str] = {}
    if tier == "tier4":
        # Relayed provenance only; never executed or trusted by this pipeline.
        for name in COMPANIONS:
            if (source_path.parent / name).is_file():
                shutil.copyfile(source_path.parent / name, output_dir / name)
                companions[name] = _sha256(output_dir / name)
    try:
        source = json.loads((output_dir / "gpt-world.json").read_text())
        tables = validate(source, tier)
    except (json.JSONDecodeError, Tier3ValidationError) as exc:
        (output_dir / "validation-error.txt").write_text(f"{exc}\n")
        raise
    frames = {name: _frame(tables[name], name) for name in ARTIFACTS}
    if tier == "tier4":
        frames.update(_tier4_frames(tables))
    checksums: dict[str, str] = {}
    for name, frame in frames.items():
        path = output_dir / f"{name}.parquet"
        frame.sort(_sort_column(name)).write_parquet(path)
        checksums[path.name] = _sha256(path)
    counts: dict[str, Any] = {
        "accounts": len(tables["accounts"]),
        "events": len(tables["account_events"]),
        "linkage_observations": len(tables["linkage_observations"]),
        "entitlement_cycles": len(tables["entitlement_cycles"]),
        "clusters": len(tables["ground_truth"]),
        "clusters_by_label": dict(Counter(r["label"] for r in tables["ground_truth"])),
    }
    manifest: dict[str, Any] = {
        "schema_version": profile.schema_version,
        "world_version": source["world_version"],
        "world_tier": tier,
        "author_model": source["author_model"],
        "seed": source.get("generator_seed", "owner-relayed"),
        "world_end": WORLD_END,
        "source_json_sha256": _sha256(output_dir / "gpt-world.json"),
        "counts": counts,
        "checksums": checksums,
        "validation": {
            "mechanical": profile.schema_version,
            "semantic_leakage_review": "manual_required",
            "policy_label_review": "manual_required",
        },
    }
    if tier == "tier4":
        counts["appeals"] = frames["appeals"].height
        counts["appeals_by_kind"] = dict(Counter(r["kind"] for r in tables["appeal_truth"]))
        manifest["companion_sha256"] = companions
        manifest["validation"]["appeal_leakage_scan"] = "passed"
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return load_tier2(output_dir) if load else None


def _sort_column(name: str) -> str:
    return {
        "accounts": "account_id",
        "account_events": "event_id",
        "linkage_observations": "observation_id",
        "entitlement_cycles": "cycle_id",
        "ground_truth": "cluster_id",
        "cluster_membership": "account_id",
        "appeals": "appeal_id",
        "appeal_truth": "event_id",
    }[name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="owner-relayed GPT JSON")
    parser.add_argument(
        "--tier", choices=sorted(PROFILES), default="tier3", help="brief the world answers"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="new, empty world source directory"
    )
    parser.add_argument(
        "--no-load",
        action="store_true",
        help="validate/write parquets without materializing stores",
    )
    args = parser.parse_args()
    result = ingest(args.source, args.out, load=not args.no_load, tier=args.tier)
    print(f"validated and preserved raw GPT source at {args.out / 'gpt-world.json'}")
    if result is not None:
        print(f"loaded {args.tier} stores: " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

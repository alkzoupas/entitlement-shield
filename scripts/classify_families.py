"""Anticipated / unanticipated family classifier CLI (docs/features.md §7).

Two modes:

* ``--from-stores`` (P2.2, default deliverable): run the frozen feature stack over a
  loaded evaluation world (the Tier-2 stores) and write ``docs/family-mapping.md``.
  The join to each unit's scenario family uses the family-only ground-truth reader
  (never the binary abuse label, §7.3). Requires ``make load-tier2`` first.

* ``--tagged`` (Phase-1 dev path): read a pre-joined JSON list of
  ``[scenario_family, feature_id, fired]`` and emit the mapping as JSON.

Usage:
  python scripts/classify_families.py --from-stores [--dir data/tier2/v2] [--out PATH]
  python scripts/classify_families.py --tagged tagged_rows.json [--out mapping.json]
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.engine import make_inference_engine
from app.db.ground_truth_store import make_ground_truth_engine
from app.db.world_provenance import manifest_identity, verify_world_pair
from app.services.classify_families import (
    build_family_tuples,
    classify_families,
    render_family_mapping_markdown,
)
from app.services.feature_manifest import verified_manifest_hash

_BANNER = "!! TIER 2 — SAME-MODEL-FAMILY, NOT CROSS-FAMILY REPORTABLE (docs/tier2-decisions.md) !!"


def _run_tagged(tagged_path: str, out: str | None) -> None:
    with open(tagged_path) as fh:
        tagged = [(r[0], r[1], bool(r[2])) for r in json.load(fh)]
    mapping = [
        {
            "scenario_family": m.scenario_family,
            "label": m.label,
            "firing_features": list(m.firing_features),
            "rationale": m.rationale,
        }
        for m in classify_families(tagged)
    ]
    text = json.dumps(mapping, indent=2, sort_keys=True)
    if out:
        with open(out, "w") as fh:
            fh.write(text + "\n")
        print(f"wrote {out} ({len(mapping)} families)")
    else:
        print(text)


def _run_from_stores(directory: str, out: str, as_of_arg: str | None) -> None:
    d = Path(directory)
    inf_path = d / "inference.db"
    gt_path = d / "ground_truth.db"
    manifest_path = d / "manifest.json"
    for p in (inf_path, gt_path):
        if not p.exists():
            sys.exit(f"missing {p} — run `make load-tier2` first to build the stores.")

    manifest_json = json.loads(manifest_path.read_text())
    world_end = str(manifest_json["world_end"]).replace("Z", "+00:00")
    as_of = datetime.fromisoformat(as_of_arg or world_end)
    as_of = as_of.replace(tzinfo=UTC) if as_of.tzinfo is None else as_of.astimezone(UTC)
    coverage: dict[str, dict[str, int]] = {}
    frozen_hash = verified_manifest_hash()

    inf_engine = make_inference_engine(f"sqlite:///{inf_path}")
    gt_engine = make_ground_truth_engine(f"sqlite:///{gt_path}")
    with (
        Session(inf_engine, future=True) as inf,
        Session(gt_engine, future=True) as gt,
    ):
        world = verify_world_pair(inf, gt)
        if world.world_id != manifest_identity(manifest_json):
            sys.exit("source manifest does not match the loaded store pair")
        tuples = build_family_tuples(inf, gt, as_of, coverage=coverage)
    mappings = classify_families(tuples)
    text = render_family_mapping_markdown(
        mappings,
        manifest_hash=frozen_hash,
        manifest_json=manifest_json,
        as_of=as_of,
        coverage=coverage,
    )
    Path(out).write_text(text)

    unanticipated = [m.scenario_family for m in mappings if m.label == "unanticipated"]
    print(_BANNER)
    print(f"feature_manifest_hash={verified_manifest_hash()}")
    print(f"as_of={as_of.isoformat()}  families={len(mappings)}  wrote {out}")
    if unanticipated:
        print(f"  unanticipated ({len(unanticipated)}): {', '.join(sorted(unanticipated))}")
    else:
        print("  all families anticipated")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify scenario families (features.md §7).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-stores", action="store_true", help="Run over a loaded world.")
    mode.add_argument("--tagged", help="JSON: [[family, feature_id, fired], ...] (dev path).")
    parser.add_argument("--dir", default="data/tier2/v2", help="World store dir (--from-stores).")
    parser.add_argument("--as-of", default=None, help="ISO8601 as_of (default: world_end).")
    parser.add_argument("--out", default=None, help="Output path.")
    args = parser.parse_args()

    if args.from_stores:
        _run_from_stores(args.dir, args.out or "docs/family-mapping.md", args.as_of)
    else:
        _run_tagged(args.tagged, args.out)


if __name__ == "__main__":
    main()

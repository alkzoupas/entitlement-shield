"""CLI: explicitly republish the Tier 3 (GPT-authored) world pair.

The Tier 3 inference/ground-truth pair was first published *before* the
provenance-metadata repair, so its receipts carry the wrong ``tier2`` tier.
Against the repaired :func:`app.db.world_provenance.verify_world_pair` that
mislabelled pair now fails closed (receipt tier ``tier2`` vs sealed manifest
``world_tier: tier3``). This script performs the safe, explicit, opt-in
republication: it rebuilds a fresh generation from the already-verified source
parquets and atomically swaps ``stores-current`` onto it, retaining the prior
generation and refusing any non-symlink commit-marker state. It never edits,
regenerates, or repairs the authoritative ``gpt-world.json``, ``manifest.json``,
or parquet sources — it re-reads their verified bytes and preserves them.

Usage:  python scripts/republish_tier3.py [--dir data/tier3]

Tier 3 is an open, small-N engineering/provenance diagnostic; its metrics are
NEVER reportable as portfolio performance. See ``docs/tier3-decisions.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.engine import make_inference_engine
from app.db.ground_truth_store import make_ground_truth_engine
from app.db.tier2_loader import load_tier2
from app.db.world_provenance import manifest_tier, verify_world_pair

DEFAULT_DIR = "data/tier3"
EXPECTED_TIER = "tier3"
# Authoritative source files hashed before/after in addition to the manifest's
# six parquet checksums. `gpt-world.json` is the GPT-authored source; both must
# be byte-identical across republication (this script never edits them).
_EXTRA_SOURCE_FILES = ("manifest.json", "gpt-world.json")


def _source_checksums(directory: Path) -> dict[str, str]:
    """Hash the full authoritative source surface this script must leave
    untouched: every manifest-named parquet PLUS `manifest.json` and
    `gpt-world.json`. A missing authoritative file is itself a failure."""
    manifest = json.loads((directory / "manifest.json").read_text())
    names = sorted(set(manifest["checksums"]) | set(_EXTRA_SOURCE_FILES))
    out: dict[str, str] = {}
    for name in names:
        path = directory / name
        if not path.is_file():
            raise SystemExit(f"ABORT: authoritative source file missing: {path}")
        out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _assert_manifest_is_tier3(directory: Path) -> None:
    """Refuse to publish anything that is not a Tier 3 world BEFORE the mutating
    loader runs, so a wrong `--dir` (e.g. a Tier 2 world) cannot have its live
    publication rebuilt before the post-hoc tier check fires."""
    manifest = json.loads((directory / "manifest.json").read_text())
    tier = manifest_tier(manifest)
    if tier != EXPECTED_TIER:
        raise SystemExit(
            f"ABORT: {directory}/manifest.json declares tier {tier!r}, not 'tier3'; "
            "refusing to republish a non-Tier-3 world with this command"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicitly republish the Tier 3 world pair with correct tier3 provenance."
    )
    parser.add_argument("--dir", default=DEFAULT_DIR)
    args = parser.parse_args()
    d = Path(args.dir).resolve()

    # Gate on the declared source tier BEFORE any mutation of the publication.
    _assert_manifest_is_tier3(d)

    before = _source_checksums(d)
    counts = load_tier2(d, republish=True)
    after = _source_checksums(d)
    if before != after:
        changed = sorted(n for n in before if before[n] != after.get(n))
        raise SystemExit(
            f"ABORT: authoritative source bytes changed during republication: {changed}"
        )

    # Verify the freshly published pair through the normal read path, using the
    # published symlinks (not the ground-truth handle wired to any detector).
    inf_engine = make_inference_engine(f"sqlite:///{d / 'inference.db'}")
    gt_engine = make_ground_truth_engine(f"sqlite:///{d / 'ground_truth.db'}")
    try:
        with (
            Session(inf_engine, future=True) as inf,
            Session(gt_engine, future=True) as gt,
        ):
            world = verify_world_pair(inf, gt)
    finally:
        inf_engine.dispose()
        gt_engine.dispose()

    if world.tier != EXPECTED_TIER:
        raise SystemExit(
            f"ABORT: republished pair verifies as {world.tier!r}, expected 'tier3'"
        )
    if world.reporting_status["reportable"]:
        raise SystemExit("ABORT: Tier 3 must be non-reportable")

    generation = (d / "stores-current").resolve()
    print(f"republished Tier 3 world pair under {args.dir}/")
    print(f"  stores-current -> {generation.relative_to(d)}")
    print(f"  inference store:    {args.dir}/inference.db")
    print(f"  ground-truth store: {args.dir}/ground_truth.db")
    print(
        f"  accounts={counts['accounts']} events={counts['events']} "
        f"linkage={counts['linkage']} cycles={counts['cycles']} "
        f"clusters={counts['clusters']} split_groups={counts['split_groups']}"
    )
    print(f"  verified world_id={world.world_id} tier={world.tier}")
    print(
        f"  reporting: reportable={world.reporting_status['reportable']} "
        f"code={world.reporting_status['code']}"
    )
    print(
        f"  source bytes preserved: {len(after)} files unchanged "
        f"(6 parquets + {', '.join(_EXTRA_SOURCE_FILES)})"
    )
    print("  !! TIER 3 — open small-N diagnostic; NOT portfolio-reportable !!")


if __name__ == "__main__":
    main()

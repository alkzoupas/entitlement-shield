"""CLI: load the Tier 2 (qwen-brief) synthetic world into physically separate stores.

Thin wiring over :func:`app.db.tier2_loader.load_tier2`. Reads the sealed-authored
parquets under ``data/tier2/`` and materializes ``data/tier2/inference.db`` +
``data/tier2/ground_truth.db``.

Usage:  python scripts/load_tier2.py [--dir data/tier2]

Tier 2 metrics are NOT reportable as cross-family: the author is a Claude Sonnet
stand-in for qwen (same model family), see ``docs/tier2-decisions.md``.
"""

from __future__ import annotations

import argparse

from app.db.tier2_loader import load_tier2

DEFAULT_DIR = "data/tier2/v2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the Tier 2 world.")
    parser.add_argument("--dir", default=DEFAULT_DIR)
    args = parser.parse_args()

    counts = load_tier2(args.dir)
    print(f"loaded Tier 2 world from {args.dir}/")
    print(f"  inference store:    {args.dir}/inference.db")
    print(f"  ground-truth store: {args.dir}/ground_truth.db")
    print(
        f"  accounts={counts['accounts']} events={counts['events']} "
        f"linkage={counts['linkage']} cycles={counts['cycles']} "
        f"clusters={counts['clusters']} split_groups={counts['split_groups']}"
    )
    print("  !! DEV/TIER-2 — same-model-family author (Sonnet); NOT cross-family reportable !!")


if __name__ == "__main__":
    main()

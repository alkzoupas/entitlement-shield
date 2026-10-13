"""Seed the dev-tier synthetic world (Phase 1 Step 2).

Generates the world, writes inference rows to the inference store and sealed
labels + split groups to the SEPARATE ground-truth store, and emits parquet
tables + `manifest.json` under `data/synthetic/`.

Usage:  python scripts/seed.py [--seed N] [--preset dev|dev_arm4|min]
                          [--artifact-dir PATH]

Dev tier — never reported (W-1). This script is a wiring layer (not a
`services/` module), so it may create both engines and the GT session; the
generator itself only ever touches the write-only GT sink.
"""

import argparse

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.engine import make_inference_engine
from app.db.ground_truth_store import make_ground_truth_engine
from app.db.ground_truth_writer import make_ground_truth_sink
from app.db.schema import create_all_schemas
from app.db.world_provenance import record_world_receipts
from app.services.generator import DEFAULT_PRESET, generate_world, persist_world, write_artifacts

ARTIFACT_DIR = "data/synthetic"


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the dev-tier world.")
    parser.add_argument("--seed", type=int, default=settings.world_seed)
    parser.add_argument(
        "--preset", choices=["dev", "dev_arm4", "min"], default=DEFAULT_PRESET
    )
    parser.add_argument(
        "--artifact-dir", default=ARTIFACT_DIR,
        help="directory for generated manifest/parquet artifacts",
    )
    args = parser.parse_args()

    inference_engine = make_inference_engine()
    ground_truth_engine = make_ground_truth_engine()
    create_all_schemas(inference_engine, ground_truth_engine)

    world = generate_world(args.seed, args.preset)
    manifest = write_artifacts(world, args.artifact_dir, args.seed, args.preset)
    with (
        Session(inference_engine, future=True) as inf,
        Session(ground_truth_engine, future=True) as gt,
    ):
        persist_world(world, inf, make_ground_truth_sink(gt))
        record_world_receipts(inf, gt, manifest, tier="dev")
        inf.commit()
        gt.commit()

    print(f"seeded preset={args.preset!r} seed={args.seed}")
    print(f"  inference store:    {settings.database_url}")
    print(f"  ground-truth store: {settings.ground_truth_database_url}")
    print(f"  artifacts:          {args.artifact_dir}/ (manifest.json + parquet)")
    print(
        f"  clusters={manifest['counts']['clusters']} "
        f"accounts={manifest['counts']['accounts']} "
        f"events={manifest['counts']['events']} "
        f"prevalence_abusive={manifest['prevalence_abusive']}"
    )
    print(f"  checksum={manifest['checksum_sha256']}")


if __name__ == "__main__":
    main()

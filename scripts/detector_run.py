"""Run the batch detector over the seeded inference store (Phase 1 Step 6).

Clusters, scores, and records decisions + a typed audit trail at a fixed `as_of`
(the world's reference frame). Idempotent: re-running over the same world records
no new decisions (architecture §7).

Usage:  python scripts/detector_run.py [--run-id ID] [--as-of ISO8601]

Dev tier — never reported (W-1). This is a wiring layer (not a `services/` module),
so it may create the engine and session directly. It never touches ground truth.
"""

import argparse
import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.engine import make_inference_engine
from app.services.detector import run_detector
from app.services.feature_manifest import verified_manifest_hash
from app.services.generator import WORLD_END


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the batch detector.")
    parser.add_argument("--run-id", default=f"run-{uuid.uuid4().hex[:12]}")
    parser.add_argument(
        "--as-of",
        default=WORLD_END.isoformat(),
        help="Decision time (ISO-8601). Defaults to the dev world's reference frame.",
    )
    args = parser.parse_args()
    as_of = datetime.fromisoformat(args.as_of)

    # Fail closed if the committed freeze artifact is missing/stale (review finding 7).
    manifest_hash = verified_manifest_hash()

    engine = make_inference_engine()
    with Session(engine, future=True) as session:
        result = run_detector(session, as_of=as_of, run_id=args.run_id)
        session.commit()

    print(f"detector run={result.run_id!r} as_of={result.as_of.isoformat()}")
    print(f"  feature_manifest_hash={manifest_hash}")
    print(f"  inference store:    {settings.database_url}")
    print(
        f"  clusters={result.clusters} cluster_cycles={result.cluster_cycles} "
        f"decisions_created={result.decisions_created} "
        f"decisions_skipped={result.decisions_skipped} "
        f"decisions_withdrawn={result.decisions_withdrawn} "
        f"escalations_released={result.escalations_released}"
    )


if __name__ == "__main__":
    main()

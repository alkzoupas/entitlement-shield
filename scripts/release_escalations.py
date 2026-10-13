"""Standalone CLI for L-5's 7-day auto-release (Phase 3 plan P3.6).

`run_detector` already calls `release_expired_escalations` as part of the nightly
batch, so this script is not a separate cron — it exists for an out-of-band or
manual release (e.g. re-running release without re-running the whole detector),
sharing the exact same `escalation.release_expired_escalations` function.

Usage:  python scripts/release_escalations.py [--run-id ID] [--as-of ISO8601]

Dev tier — never reported (W-1). This is a wiring layer (not a `services/` module),
so it may create the engine and session directly. It never touches ground truth.
"""

import argparse
import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from app.core.clock import now_utc
from app.core.config import settings
from app.db.engine import make_inference_engine
from app.services.escalation import release_expired_escalations


def main() -> None:
    parser = argparse.ArgumentParser(description="Release expired L-5 escalations.")
    parser.add_argument("--run-id", default=f"release-{uuid.uuid4().hex[:12]}")
    parser.add_argument(
        "--as-of",
        default=None,
        help="Release cutoff (ISO-8601). Defaults to the current wall-clock time.",
    )
    args = parser.parse_args()
    as_of = datetime.fromisoformat(args.as_of) if args.as_of else now_utc()

    engine = make_inference_engine()
    with Session(engine, future=True) as session:
        released = release_expired_escalations(session, as_of=as_of, run_id=args.run_id)
        session.commit()

    print(f"release-escalations run={args.run_id!r} as_of={as_of.isoformat()}")
    print(f"  inference store: {settings.database_url}")
    print(f"  escalations_released={released}")


if __name__ == "__main__":
    main()

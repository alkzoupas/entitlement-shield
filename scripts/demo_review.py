"""Phase 5 Track 2 — run the (configured) reviewer once against each of
`scripts/demo_pick.py`'s picked decisions, so `make demo`'s narrative panel is
already populated on first page load instead of only appearing after a manual
"Run reviewer" click.

Reads the picks JSON `scripts/demo_pick.py --out` wrote (so both scripts agree
on exactly which decisions were chosen against this store) and calls
the reviewer's sampling stage on each one directly against the inference store.
It persists advice samples, provenance and provider failures, but deliberately
does not commit a reviewer outcome or transition the Decision: P5-2 requires
demo generation to be advice-only. Uses whatever provider
`app.providers.factory.load_provider_config()` resolves (honors
`REVIEWER_PROVIDER_CONFIG_PATH`, P5 Track 2's demo override), so this never
hardcodes a provider.

Fails loudly but cleanly (a one-line message, not a raw traceback) when the
configured provider needs credentials that are not set — e.g. `active:
anthropic` with no `ANTHROPIC_API_KEY` — since `make demo` is meant to be
runnable by an operator who has not read this file first.

Usage: python scripts/demo_review.py --picks PATH [--as-of ISO8601]
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from app.core.clock import now_utc
from app.db.engine import make_inference_engine
from app.models.inference import Decision, LlmReview
from app.providers.factory import ProviderConfigError, build_provider, load_provider_config
from app.services.reviewer import _sample_decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate advice-only reviewer narratives for demo picks."
    )
    parser.add_argument("--picks", required=True, help="JSON file written by demo_pick.py --out")
    parser.add_argument("--as-of", default=None, help="ISO-8601 review time (default: now)")
    args = parser.parse_args()
    as_of = datetime.fromisoformat(args.as_of) if args.as_of else now_utc()

    try:
        with open(args.picks) as fh:
            picks = json.load(fh)
    except FileNotFoundError:
        print(
            f"[demo_review] no picks file at {args.picks!r} — run "
            "'python scripts/demo_pick.py --out ...' first",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    if not picks:
        print("[demo_review] no picks to review — nothing to do")
        return

    try:
        provider = build_provider(load_provider_config())
    except ProviderConfigError as exc:
        print(f"[demo_review] provider not available: {exc}", file=sys.stderr)
        print(
            "[demo_review] set the API key the demo provider names (TYPESAFE_API_KEY "
            "for 'typesafe', ANTHROPIC_API_KEY for 'anthropic'; env or .env) and re-run.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    engine = make_inference_engine()
    provider_failures = 0
    with Session(engine, future=True) as session:
        for pick in picks:
            decision_id = pick["decision_id"]
            decision = session.get(Decision, decision_id)
            if decision is None:
                print(
                    f"[demo_review] skipping {decision_id!r} — not found (stale picks file?)",
                    file=sys.stderr,
                )
                continue
            sampled = _sample_decision(
                session,
                decision,
                provider=provider,
                run_id=f"demo-{uuid.uuid4().hex[:12]}",
                idempotency_key=f"demo-advice-{uuid.uuid4().hex}",
                as_of=as_of,
                arm="demo_advice",
            )
            session.commit()
            if sampled.outcome is not None:
                print(
                    f"[demo_review] advice skipped {decision_id} ({pick.get('label', '')}): "
                    f"{sampled.outcome.gate_reason or 'replayed'}"
                )
                continue
            review_ids = [
                review_id for sample in sampled.samples for review_id in sample.review_ids
            ]
            failures = sum(
                1
                for review_id in review_ids
                if (row := session.get(LlmReview, review_id)) is not None
                and (row.validation_errors or {}).get("provider_error")
            )
            provider_failures += failures
            print(
                f"[demo_review] saved advice {decision_id} ({pick.get('label', '')}): "
                f"effect={sampled.overall} provider_failures={failures}; decision unchanged"
            )

    if provider_failures:
        print(
            f"[demo_review] advice generation recorded {provider_failures} provider failure(s); "
            "inspect the case panel before retrying",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()

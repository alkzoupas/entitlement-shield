"""Phase 5 Track 2 — pick a handful of archetypal cases for `make demo` to print
as `/cases/{decision_id}` console URLs.

Reads the already-seeded dev-tier stores (`make seed`, `data/synthetic/dev.db` +
`ground_truth.db`) and labels each decision's account by scenario *family only*
via `app.db.ground_truth_family_reader.load_account_families` — the same
family-only boundary `scripts/classify_families.py` uses (never the binary
`GroundTruth.label`). This is a script, not the app: the console/API never gets
a ground-truth handle (locked invariant), but a demo-picker CLI reading family
labels to choose which URLs to print is not the inference plane.

Prints up to one URL per archetype:
  * an abusive farm (`multi_device_farm` / `disposable_email_farm`) flagged by the
    detector (the exact proposed action is displayed, never described as a hold)
  * a legitimate cluster the detector flagged anyway (`reversible_singleton_legit`
    — the engineered false positive, only present under the `dev_arm4` preset) —
    the frame for the new "LLM judgment as decision support" narrative panel
  * an optional ambiguous_family ALLOW beat, which currently self-skips because ALLOW
    outcomes are not persisted as Decision rows (P5.1 documents the separate,
    fixture-scoped empty cap-reason result)

Every candidate is restricted to `status == PROPOSED` — a decision the reviewer
run hasn't already superseded/reversed/escalated — so the demo always opens on
a live case, never a stale one left over from a previous run against the same
store. Each archetype is skipped cleanly (with a note on stderr) if no matching
decision exists in the seeded world — this must not crash a demo run.

`--out PATH` additionally writes the picks as JSON (`[{"decision_id", "label",
"path"}, ...]`) for `scripts/demo_review.py` to consume, so the two scripts
never re-derive the picks differently against a store that could in principle
change between the two invocations.

Usage: python scripts/demo_pick.py [--base-url http://localhost:8000] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.engine import make_inference_engine
from app.db.ground_truth_family_reader import load_account_families
from app.db.ground_truth_store import make_ground_truth_engine
from app.models.enums import DecisionStatus, ProposedAction
from app.models.inference import Decision

_LEGIT_FAMILIES = frozenset(
    {"single_legit", "household_ipad", "school_lab", "homogeneous_lab", "workplace_nat"}
)
_ABUSIVE_FARM_FAMILIES = frozenset({"multi_device_farm", "disposable_email_farm"})
_LEGIT_FAMILIES_WITH_ENGINEERED_FP = _LEGIT_FAMILIES | {"reversible_singleton_legit"}


def _pick_abusive_farm_held(
    decisions: list[Decision], family_of: dict[str, str]
) -> Decision | None:
    for d in decisions:
        if family_of.get(d.account_id) in _ABUSIVE_FARM_FAMILIES:
            return d
    return None


def _pick_legit_flagged(decisions: list[Decision], family_of: dict[str, str]) -> Decision | None:
    for d in decisions:
        if family_of.get(d.account_id) in _LEGIT_FAMILIES_WITH_ENGINEERED_FP:
            return d
    return None


def _pick_ambiguous_allowed(
    decisions: list[Decision], family_of: dict[str, str]
) -> Decision | None:
    """Looks for an `ambiguous_family` decision with `proposed_action == ALLOW`.

    Per `detector.py` (`actionable = proposed_action is not ALLOW`), an ALLOW
    outcome never gets a persisted `Decision` row at all — so this will
    structurally never match while ALLOW outcomes are unpersisted. Left in
    (rather than removed) so a future persistence-policy change would surface a
    case automatically; until then this archetype always skips cleanly."""
    for d in decisions:
        if (
            family_of.get(d.account_id) == "ambiguous_family"
            and d.proposed_action is ProposedAction.ALLOW
        ):
            return d
    return None


def pick_cases(inf_session: Session, gt_session: Session) -> list[tuple[str, str, str]]:
    """Returns `(decision_id, label, url_path)` triples, one per archetype found.

    Candidates are restricted to `status == PROPOSED` (never a decision a prior
    reviewer run already superseded/reversed/escalated) so every picked URL
    opens on a live case."""
    decisions = list(
        inf_session.scalars(
            select(Decision)
            .where(Decision.status == DecisionStatus.PROPOSED)
            .order_by(Decision.created_at, Decision.decision_id)
        ).all()
    )
    family_of = load_account_families(gt_session)

    farm = _pick_abusive_farm_held(decisions, family_of)
    legit = _pick_legit_flagged(decisions, family_of)
    if farm is None or legit is None:
        required = (("abusive farm", farm), ("legitimate flagged", legit))
        missing = [name for name, pick in required if pick is None]
        raise SystemExit(
            "[demo_pick] ERROR: required demo archetype(s) missing: " + ", ".join(missing)
        )

    picks: list[tuple[str, str, Decision | None]] = [
        (
            f"abusive farm flagged for {farm.proposed_action.value}",
            "multi_device_farm/disposable_email_farm",
            farm,
        ),
        (
            "legitimate cluster the detector flagged — LLM judgment as decision support",
            "legit family, flagged",
            legit,
        ),
        (
            "uncertainty routed to ALLOW (see docs/p5.1-ambiguity-population-finding.md)",
            "ambiguous_family, allowed",
            _pick_ambiguous_allowed(decisions, family_of),
        ),
    ]

    out: list[tuple[str, str, str]] = []
    for label, archetype, decision in picks:
        if decision is None:
            print(
                f"[demo_pick] skipping {archetype!r} — no matching decision found",
                file=sys.stderr,
            )
            continue
        out.append((decision.decision_id, label, f"/cases/{decision.decision_id}"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Print archetypal /cases URLs for `make demo`.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--out",
        default=None,
        help="Also write the picks as JSON here, for scripts/demo_review.py to consume.",
    )
    args = parser.parse_args()

    inf_engine = make_inference_engine()
    gt_engine = make_ground_truth_engine()
    with (
        Session(inf_engine, future=True) as inf,
        Session(gt_engine, future=True) as gt,
    ):
        picks = pick_cases(inf, gt)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(
                [{"decision_id": d, "label": label, "path": path} for d, label, path in picks],
                fh,
                indent=2,
            )
        print(f"[demo_pick] wrote {len(picks)} pick(s) to {args.out}")

    if not picks:
        print("[demo_pick] no archetypal cases found in the seeded world — run `make seed` first")
        return

    print()
    print("Demo cases:")
    for _decision_id, label, path in picks:
        print(f"  {label}")
        print(f"    {args.base_url}{path}")
    print()


if __name__ == "__main__":
    main()

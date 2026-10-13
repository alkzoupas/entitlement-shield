"""Score the detector on the seeded dev world (Phase 1 Step 7).

Builds the fixed cluster-cycle rows, scores arm 1 (Stage 1 at 0.60) and arm 5
(oracle), and prints the confusion + wrongful-restriction rate.

Usage:  python scripts/eval_run.py [--as-of ISO8601]

DEV TIER — these numbers MUST NOT be reported (W-1). The harness is what is
validated; reported results come from the sealed Tier-2 world in a later phase.
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime

from sqlalchemy.orm import Session

from app.core.clock import now_utc
from app.db.engine import make_inference_engine
from app.db.ground_truth_store import make_ground_truth_engine
from app.db.world_provenance import verify_world_pair
from app.services.evaluation import (
    ArmReport,
    Rung,
    arm1_threshold,
    arm5_oracle,
    build_eval_rows,
    load_ground_truth,
    score_rows,
)
from app.services.feature_manifest import verified_manifest_hash
from app.services.generator import WORLD_END


def _arm_row(report: ArmReport, label: str) -> dict:
    c = report.confusion
    return {
        "arm": report.arm,
        "label": label,
        "precision": round(c.precision, 4),
        "recall": round(c.recall, 4),
        "fpr": round(c.fpr, 4),
        "pr_auc": round(report.pr_auc, 4),
        "wrongful_restriction": round(report.wrongful_restriction_rate, 4),
        "tp": c.tp, "tn": c.tn, "fp": c.fp, "fn": c.fn,
    }


def build_report(
    arm1: ArmReport, arm5: ArmReport, *, as_of: datetime, split: str, rows: int
) -> dict:
    """The dev-tier evaluation report the console's evaluation page renders. DEV TIER —
    W-1 NOT reportable; the banner-carrying flag travels with the data."""
    return {
        "dev_tier_not_reportable": True,
        "generated_at": now_utc().isoformat(),
        "as_of": as_of.isoformat(),
        "split": split,
        "rows": rows,
        "feature_manifest_hash": verified_manifest_hash(),
        "arms": [
            _arm_row(arm1, "Stage 1 detector @0.60"),
            _arm_row(arm5, "Oracle (true labels)"),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the detector (dev tier).")
    parser.add_argument("--as-of", default=WORLD_END.isoformat())
    parser.add_argument(
        "--split", choices=["all", "train", "tune", "test"], default="tune",
        help="Restrict to one split. NOTE: `test` is the sealed set — scoring it is a "
             "one-touch finalization event (eval §9), not a routine command.",
    )
    parser.add_argument(
        "--json", dest="json_path", default=None,
        help="Also write the report as JSON here (the console's evaluation page reads it).",
    )
    args = parser.parse_args()
    as_of = datetime.fromisoformat(args.as_of)

    manifest_hash = verified_manifest_hash()
    inf_engine = make_inference_engine()
    gt_engine = make_ground_truth_engine()
    with (
        Session(inf_engine, future=True) as inf,
        Session(gt_engine, future=True) as gt,
    ):
        world = verify_world_pair(inf, gt)
        splits = ("train", "tune", "test") if args.split == "all" else (args.split,)
        rows = build_eval_rows(inf, load_ground_truth(
            gt, inf_session=inf, splits=splits,
        ), as_of)
        if args.split != "all":
            rows = [r for r in rows if r.split == args.split]
        scores_s2 = score_rows(inf, rows, Rung.S2)
        arm1 = arm1_threshold(rows, scores_s2)
        arm5 = arm5_oracle(rows)

    print(f"world={world.world_id} tier={world.tier}; open development, not sealed evidence")
    print(f"feature_manifest_hash={manifest_hash}")
    print(f"eval as_of={as_of.isoformat()}  split={args.split}  rows={len(rows)}")
    c = arm1.confusion
    print(
        f"  arm1@0.60: precision={c.precision:.3f} recall={c.recall:.3f} "
        f"fpr={c.fpr:.3f} pr_auc={arm1.pr_auc:.3f} "
        f"wrongful_restriction={arm1.wrongful_restriction_rate:.3f}"
    )
    o = arm5.confusion
    print(f"  arm5 oracle: tp={o.tp} tn={o.tn} (perfect on non-ambiguous by definition)")

    if args.json_path:
        report = build_report(arm1, arm5, as_of=as_of, split=args.split, rows=len(rows))
        report["world"] = asdict(world)
        report["not_reportable"] = not world.reporting_status["reportable"]
        report["dev_tier_not_reportable"] = world.tier == "dev"
        report["reporting"] = world.reporting_status
        report["reporting_status"] = world.reporting_status["reason"]
        with open(args.json_path, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        print(f"wrote report → {args.json_path}")


if __name__ == "__main__":
    main()

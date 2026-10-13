"""S0→S1→S2 ablation ladder on the seeded dev world (Phase 1 Step 7).

Scores the *identical* fixed cluster-cycle rows at each rung and reports marginal
lift + marginal FPR per rung at matched recall (E-13, review 4 P0-1).

Usage:  python scripts/ablation_run.py [--as-of ISO8601]

DEV TIER — these numbers MUST NOT be reported (W-1).
"""

import argparse
from datetime import datetime

from sqlalchemy.orm import Session

from app.db.engine import make_inference_engine
from app.db.ground_truth_store import make_ground_truth_engine
from app.db.world_provenance import verify_world_pair
from app.services.evaluation import build_eval_rows, load_ground_truth, run_ablation
from app.services.feature_manifest import verified_manifest_hash
from app.services.generator import WORLD_END


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the S0/S1/S2 ablation (dev tier).")
    parser.add_argument("--as-of", default=WORLD_END.isoformat())
    parser.add_argument("--split", choices=["train", "tune", "test", "all"], default="tune")
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
        report = run_ablation(inf, rows)

    print(f"world={world.world_id} tier={world.tier}; open development, not sealed evidence")
    print(f"feature_manifest_hash={manifest_hash}")
    print(
        f"ablation as_of={as_of.isoformat()} rows={report.rows} "
        f"pos={report.denom_positive} neg={report.denom_negative} "
        f"target_recall={report.target_recall:.3f}"
    )
    for rung, rep in report.per_rung.items():
        c = rep.at_threshold
        print(
            f"  {rung.name}: @0.60 precision={c.precision:.3f} recall={c.recall:.3f} "
            f"fpr={c.fpr:.3f} pr_auc={rep.pr_auc:.3f}"
        )
    for m in report.marginals:
        print(
            f"  {m.from_rung.name}->{m.to_rung.name} (matched recall): "
            f"precision_lift={m.precision_lift:+.3f} fpr_cost={m.fpr_cost:+.3f} "
            f"fps_removed={m.fps_removed} fns_added={m.fns_added}"
        )


if __name__ == "__main__":
    main()

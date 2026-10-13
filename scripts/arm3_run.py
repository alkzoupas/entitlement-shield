"""Open-split arm3 mechanics; sealed scoring is only available through finalization."""
import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.engine import make_inference_engine
from app.db.ground_truth_store import make_ground_truth_engine
from app.db.world_provenance import verify_world_pair
from app.services.evaluation import (
    arm3_reranker,
    build_eval_rows,
    load_ground_truth,
    matched_group_bootstrap_ci,
    operating_point_record,
)
from app.services.feature_manifest import verified_manifest_hash
from app.services.generator import WORLD_END


def main() -> None:
    parser = argparse.ArgumentParser(description="Arm3 candidate-only attribution control")
    parser.add_argument("--as-of", default=WORLD_END.isoformat())
    parser.add_argument("--split", choices=["train", "tune"], default="tune")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    manifest_hash = verified_manifest_hash()
    as_of = datetime.fromisoformat(args.as_of)
    with Session(make_inference_engine()) as inf, Session(make_ground_truth_engine()) as gt:
        world = verify_world_pair(inf, gt)
        rows = build_eval_rows(inf, load_ground_truth(
            gt, inf_session=inf, splits=("train", "tune"),
        ), as_of)
        arm3 = arm3_reranker(inf, rows, report_split=args.split)
        selected = [r for r in rows if r.split == args.split]
        artifact = {
            "world": asdict(world), "feature_manifest_hash": manifest_hash,
            "as_of": as_of.isoformat(), "split": args.split,
            "reporting_status": "open-split development; not a sealed performance claim",
            "candidate_policy": "Stage1 policy non-ALLOW; retain original severity or ALLOW",
            "recommendations": [
                {"key": list(key), "stage1": arm3.stage1_actions[key].value,
                 "arm3": action.value} for key, action in arm3.actions.items()
            ],
            "arm3_threshold": 0.5, "arm3": asdict(arm3.report), "fit": asdict(arm3.fit),
            "stage1_policy_eligible": asdict(arm3.arm1_test),
            "stage1_matched_to_arm3": operating_point_record(arm3.vs_arm1_matched),
            "comparison": "auxiliary recall proxy; not arm2 or abuse-credit recovery",
            "uncertainty": matched_group_bootstrap_ci(
                selected, arm3.arm1_test_scores, arm3.test_scores,
            ),
        }
    payload = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)
    print(payload)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n")


if __name__ == "__main__":
    main()

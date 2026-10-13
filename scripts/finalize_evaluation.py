"""Consume one world-version's sealed touch and persist its complete P2 analysis.

This is never a routine development command. Interrupted attempts consume the
single touch too. Existing completion artifacts may be rendered without scoring.
The P2 bundle is auxiliary; running it precludes a later blind arm4 claim on this
same sealed world. Wait for all planned arms before finalizing a reportable study.
"""
import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.engine import make_inference_engine
from app.db.evaluation_access import begin_finalization, complete_finalization
from app.db.ground_truth_store import make_ground_truth_engine
from app.db.world_provenance import verify_world_pair
from app.models.enums import ProposedAction
from app.services.evaluation import (
    Rung,
    arm1_threshold,
    arm3_reranker,
    arm5_oracle,
    build_eval_rows,
    feature_vectors,
    load_ground_truth,
    matched_group_bootstrap_ci,
    operating_point_record,
    run_ablation,
    score_rows,
    stage1_candidate_actions,
    train_reranker,
)
from app.services.feature_manifest import verified_manifest_hash
from app.services.finalization_identity import canonical_evaluation_cutoff, reproducibility_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-id", required=True, help="verified immutable world id")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--consume-p2-only-touch", action="store_true", required=True,
                        help="acknowledge this excludes future blind arm4 scoring on this world")
    args = parser.parse_args()
    manifest_hash = verified_manifest_hash()
    if args.json.exists():
        raise SystemExit("refusing to overwrite a final artifact")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with Session(make_inference_engine()) as inf, Session(make_ground_truth_engine()) as gt:
        world = verify_world_pair(inf, gt)
        if world.world_id != args.world_id:
            raise SystemExit("world id does not match the verified store pair")
        world_end = world.provenance.get("world_end")
        if not isinstance(world_end, str):
            raise SystemExit("verified world has no canonical world_end")
        as_of = canonical_evaluation_cutoff(args.as_of, world_end)
        open_rows = build_eval_rows(inf, load_ground_truth(
            gt, inf_session=inf, splits=("train", "tune"),
        ), as_of)
        actions = stage1_candidate_actions(inf, open_rows)
        candidates = {k for k, action in actions.items() if action is not ProposedAction.ALLOW}
        fitted = train_reranker(
            open_rows, feature_vectors(inf, open_rows), candidates=candidates,
        )
        root = Path(__file__).resolve().parents[1]
        code = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "backend/app/services/evaluation.py", "backend/app/services/reranker.py",
                "backend/app/db/evaluation_access.py", "scripts/finalize_evaluation.py",
            )
        }
        analysis = {
            "world": asdict(world), "feature_manifest_hash": manifest_hash,
            "as_of": as_of.isoformat(), "fit": asdict(fitted), "analysis_code": code,
            "reproducibility": reproducibility_identity(root),
            "arm3_policy": "Stage1 non-ALLOW candidates; retain severity or ALLOW at fixed .5",
            "bundle": "P2 arms1/3/5 and ablation; excludes unavailable LLM arms",
            "matching": "Stage1 swept to arm3 recall; auxiliary proxy, not credit recovery",
            "bootstrap": "1000 paired split-group replicates within seed; seed=0; rematch baseline",
        }
        # Pre-flight (review C-3): both failure modes below happen to fire *inside*
        # this script AFTER begin_finalization has already permanently consumed the
        # world's one-shot sealed touch, so they must be checked before that call.
        #
        # 1. Every fitted coefficient/intercept must be finite — a non-finite model
        #    would make match_operating_point raise ValueError once it reaches the
        #    sealed rows.
        if not math.isfinite(fitted.model.bias) or not all(
            math.isfinite(w) for w in fitted.model.weights
        ):
            raise SystemExit("fitted re-ranker has non-finite coefficients; refusing to finalize")
        # 2. Structural smoke test on NON-SEALED rows only. This dry-runs the exact
        #    post-grant analysis shape (arm3_reranker + matched_group_bootstrap_ci)
        #    against the already-loaded open train/tune rows purely to confirm the
        #    code path doesn't raise (e.g. the assert-turned-branch in
        #    matched_group_bootstrap_ci, or match_operating_point's finite-score
        #    check). Its output is discarded and NEVER written to any artifact or
        #    file. This is NOT a methodology change and NOT evaluation on training
        #    data — no metric from this dry run is reported, reused, or persisted.
        try:
            _dry_run_report = arm3_reranker(
                inf, open_rows, fitted=fitted, report_split="tune",
            )
            # arm3_reranker(..., report_split="tune") keys its score dicts only over
            # tune rows; matched_group_bootstrap_ci must therefore see the same tune
            # subset, not every open row, or it KeyErrors on the first train-split
            # group and this pre-flight would block every finalization run.
            _tune_rows = [r for r in open_rows if r.split == "tune"]
            matched_group_bootstrap_ci(
                _tune_rows,
                _dry_run_report.arm1_test_scores,
                _dry_run_report.test_scores,
                iterations=25,
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad; any failure blocks sealing
            raise SystemExit(f"pre-flight smoke test failed before sealed access: {exc}") from exc
        del _dry_run_report

        grant = begin_finalization(inf, gt, analysis)
        sealed = build_eval_rows(inf, load_ground_truth(
            gt, inf_session=inf, splits=("test",), finalization=grant,
        ), as_of)
        result = arm3_reranker(inf, sealed, fitted=fitted)
        ablation = run_ablation(inf, sealed)
        artifact = {
            "analysis": analysis, "analysis_hash": grant.analysis_hash,
            "sealed_touch_count": 1,
            "arm1": asdict(arm1_threshold(sealed, score_rows(inf, sealed, Rung.S2))),
            "stage1_policy_eligible": asdict(result.arm1_test),
            "arm3": asdict(result.report), "arm5": asdict(arm5_oracle(sealed)),
            "recommendations": [
                {"key": list(key), "stage1": result.stage1_actions[key].value,
                 "arm3": action.value} for key, action in result.actions.items()
            ],
            "matched_baseline": operating_point_record(result.vs_arm1_matched),
            "uncertainty": matched_group_bootstrap_ci(
                sealed, result.arm1_test_scores, result.test_scores,
            ),
            "ablation": {
                "rows": ablation.rows, "target_recall": ablation.target_recall,
                "per_rung": {r.name: asdict(v) for r, v in ablation.per_rung.items()},
                "marginals": [
                    {**asdict(m), "from_rung": m.from_rung.name, "to_rung": m.to_rung.name}
                    for m in ablation.marginals
                ],
            },
        }
        # The DB artifact survives a subsequent output-file failure.
        complete_finalization(grant, artifact)
        with args.json.open("x") as output:
            output.write(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"finalization recorded; artifact: {args.json}")


if __name__ == "__main__":
    main()

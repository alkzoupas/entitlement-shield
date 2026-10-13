"""P6 step 0: frozen-detector recall headroom on tier4-v2, open scope (train+tune) only.

Copies the canonical inference store into SCRATCH_DIR (the source is opened read-only),
reads labels for train+tune only (no sealed access, P5-7a), and writes
SCRATCH_DIR/headroom.json: recall at the frozen threshold, how many abusive units the
detector allows, and how those misses compare with allowed legitimate units.

Usage:
    PYTHONPATH=backend .venv/bin/python scripts/p6_headroom_probe.py SCRATCH_DIR
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.db.engine import make_inference_engine
from app.db.ground_truth_store import make_ground_truth_engine
from app.db.world_provenance import verify_world_pair
from app.services.evaluation import (
    EvalRow,
    build_eval_rows,
    feature_vectors,
    load_ground_truth,
    score_rows,
)
from app.services.features import FEATURE_ORDER, Rung
from app.services.risk_scorer import SCORE_THRESHOLD

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier4_arms_run import _world_end, memoized_signal_values  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "data" / "tier4-v2"
OPEN_SPLITS = ("train", "tune")


def _band(score: float) -> str:
    if score == 0:
        return "=0"
    for low, high in ((0.0, 0.2), (0.2, 0.4), (0.4, SCORE_THRESHOLD)):
        if low <= score < high:
            return f"[{low:.1f},{high:.1f})"
    return "flagged"


def _copy_readonly(source: Path, target: Path) -> None:
    src = sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def summarize(
    rows: list[EvalRow],
    scores: dict[tuple[str, str, str], float],
    vectors: dict[tuple[str, str, str], tuple[float, ...]],
) -> dict[str, Any]:
    positives = [r for r in rows if r.is_positive]
    negatives = [r for r in rows if r.is_negative]
    caught = [r for r in positives if scores[r.key] >= SCORE_THRESHOLD]
    missed = [r for r in positives if scores[r.key] < SCORE_THRESHOLD]
    allowed_neg = [r for r in negatives if scores[r.key] < SCORE_THRESHOLD]
    caught_groups = {r.split_group_id for r in caught}
    missed_groups = {r.split_group_id for r in missed}

    def fired(row: EvalRow) -> list[str]:
        return [n for n, x in zip(FEATURE_ORDER, vectors[row.key], strict=True) if x]

    families: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in positives:
        families[r.scenario_family][0 if scores[r.key] >= SCORE_THRESHOLD else 1] += 1
    fired_missed = Counter(n for r in missed for n in fired(r))
    fired_neg = Counter(n for r in allowed_neg for n in fired(r))
    return {
        "threshold": SCORE_THRESHOLD,
        "rows": len(rows),
        "labels": dict(Counter(r.label.value for r in rows)),
        "positives": len(positives),
        "caught": len(caught),
        "missed": len(missed),
        "recall": len(caught) / len(positives) if positives else None,
        "negatives": len(negatives),
        "false_positives": len(negatives) - len(allowed_neg),
        "positive_groups": len({r.split_group_id for r in positives}),
        "missed_groups": len(missed_groups),
        "missed_groups_never_caught": len(missed_groups - caught_groups),
        "missed_units_per_group": dict(Counter(r.split_group_id[:12] for r in missed)),
        "family_caught_missed": dict(families),
        "missed_score_bands": dict(Counter(_band(scores[r.key]) for r in missed)),
        "allowed_negative_score_bands": dict(
            Counter(_band(scores[r.key]) for r in allowed_neg)
        ),
        "missed_fired_feature_count": dict(Counter(len(fired(r)) for r in missed)),
        "allowed_negative_fired_feature_count": dict(
            Counter(len(fired(r)) for r in allowed_neg)
        ),
        "feature_rate_missed_vs_allowed_negative": {
            n: [
                round(fired_missed[n] / max(len(missed), 1), 3),
                round(fired_neg[n] / max(len(allowed_neg), 1), 3),
            ]
            for n in FEATURE_ORDER
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scratch", type=Path)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    copy = args.scratch / "inference.db"
    if not copy.exists():
        _copy_readonly(WORLD / "inference.db", copy)
    as_of = _world_end(WORLD)
    gt_url = f"sqlite:///{(WORLD / 'ground_truth.db').resolve()}"
    with (
        memoized_signal_values(),
        Session(make_inference_engine(f"sqlite:///{copy}"), future=True) as inf,
        Session(make_ground_truth_engine(gt_url), future=True) as gt,
    ):
        world = verify_world_pair(inf, gt)
        rows = build_eval_rows(
            inf, load_ground_truth(gt, inf_session=inf, splits=OPEN_SPLITS), as_of
        )
        scores = score_rows(inf, rows, Rung.S2)
        vectors = feature_vectors(inf, rows)
    report = {
        "world_id": world.world_id,
        "as_of": as_of.isoformat(),
        "splits": list(OPEN_SPLITS),
        **summarize(rows, scores, vectors),
    }
    text = json.dumps(report, indent=1, sort_keys=True)
    (args.scratch / "headroom.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

"""P6.4: the supervised learnability ceiling on Tier 4's open scope (docs/decisions.md P6-4).

Can the labels be learned from label-free evidence at all? Two fixed models, no tuning, are
fit in grouped cross-validation on the features ``scripts/p6_4_features.py`` froze, and
scored once out of fold. Reads train+tune labels only (``load_ground_truth`` with the open
splits); the sealed test split is never read. Units of one source group are not independent,
so folds are grouped by source group and no interval is claimed.

Before any label is read it checks, label-free, that the features and the feature script
match the frozen manifest and that every value is finite.

* Population (i), the main readout: all open evaluation units, abusive vs legitimate
  (ambiguous excluded). Population (ii), descriptive: the allowed units (detector score
  < 0.60), pooled and multi-account, scored with the same out-of-fold predictions.
* Models: L2 logistic regression (C = 1.0, standardized features) and
  ``HistGradientBoostingClassifier`` with library defaults; ``StratifiedGroupKFold``
  (5 folds, shuffled, seed 20261016) by source group.
* Metrics per model, next to the detector score: out-of-fold AUC, group-weighted AUC, and at
  the detector's own flag volume (its flags among the binary units) the abusive catches,
  precision, balanced accuracy and per-family recall, plus the random expectation.
* Reading (pre-registered): the ceiling is the higher pooled AUC of the two models;
  <= 0.65 reads as a world/label problem, >= 0.80 as a method problem, otherwise
  inconclusive.

Usage:
    PYTHONPATH=backend .venv/bin/python scripts/p6_4_ceiling.py \
        --out data/tier4-v2/runs/p6-4-ceiling [--check-only]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p6_3_score import _auc_f, weighted_auc  # noqa: E402
from p6_recall_pilot import DETECTED, OPEN_SPLITS, WORLD, _key, _ro, _sha256  # noqa: E402

from app.db.engine import make_inference_engine  # noqa: E402
from app.db.ground_truth_store import make_ground_truth_engine  # noqa: E402
from app.services.evaluation import load_ground_truth  # noqa: E402
from app.services.risk_scorer import SCORE_THRESHOLD  # noqa: E402

SEED = 20261016
FOLDS = 5
WORLD_BAND, METHOD_BAND = 0.65, 0.80
FEATURE_SCRIPT = Path(__file__).resolve().parent / "p6_4_features.py"


def _models() -> dict[str, Any]:
    return {
        "logistic_l2": make_pipeline(StandardScaler(),
                                     LogisticRegression(C=1.0, max_iter=10_000)),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=SEED),
    }


def load_features(out: Path) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    """Features and manifest, checked label-free against the frozen hashes."""
    manifest = json.loads((out / "manifest.json").read_text())
    if _sha256(out / "features.jsonl") != manifest["features_sha256"]:
        raise SystemExit("features.jsonl does not match its manifest")
    if _sha256(FEATURE_SCRIPT) != manifest["feature_script_sha256"]:
        raise SystemExit("the feature script changed after the features were frozen")
    names = manifest["features"]
    rows = {}
    for line in (out / "features.jsonl").read_text().splitlines():
        row = json.loads(line)
        if set(row["features"]) != set(names):
            raise SystemExit(f"unit {row['key']}: feature list differs from the manifest")
        if not all(math.isfinite(v) for v in row["features"].values()):
            raise SystemExit(f"unit {row['key']}: non-finite feature")
        rows[row["key"]] = row
    if len(rows) != manifest["units"]:
        raise SystemExit("unit count differs from the manifest")
    return names, rows, manifest


def first_members(keys: set[str]) -> dict[str, str]:
    """Unit key -> one member account (label-free), from the detected store."""
    members = {}
    with _ro(DETECTED) as det:
        for (raw,) in det.execute("select payload_json from detection_evidence"):
            payload = json.loads(raw)
            if _key(payload) in keys:
                members[_key(payload)] = payload["member_account_ids"][0]
    if set(members) != keys:
        raise SystemExit("some feature units are missing from the detected store")
    return members


def _ranked(keys: list[str], score: dict[str, float]) -> list[str]:
    order = sorted(keys)
    random.Random(SEED).shuffle(order)
    tiebreak = {k: i for i, k in enumerate(order)}
    return sorted(keys, key=lambda k: (-score[k], tiebreak[k]))


def at_volume(keys: list[str], score: dict[str, float], flags: int, label: dict[str, str],
              family: dict[str, str]) -> dict[str, Any]:
    """Flag the top ``flags`` units by score (ties by a seeded key order)."""
    top = set(_ranked(keys, score)[:flags])
    pos = [k for k in keys if label[k] == "abusive"]
    neg = len(keys) - len(pos)
    caught = [k for k in pos if k in top]
    tpr, tnr = len(caught) / len(pos), (neg - (flags - len(caught))) / neg
    families = sorted({family[k] for k in pos})
    return {
        "flags": flags, "abusive_caught": len(caught),
        "legitimate_flagged": flags - len(caught),
        "precision": round(len(caught) / flags, 4),
        "balanced_accuracy": round((tpr + tnr) / 2, 4),
        "random_expected": round(flags * len(pos) / len(keys), 2),
        "recall_by_family": {
            f: f"{sum(k in top for k in pos if family[k] == f)}"
               f"/{sum(family[k] == f for k in pos)}" for f in families
        },
    }


def block(keys: list[str], score: dict[str, float], label: dict[str, str],
          group: dict[str, str]) -> dict[str, Any]:
    pos = [k for k in keys if label[k] == "abusive"]
    neg = [k for k in keys if label[k] == "legitimate"]
    auc = _auc_f([score[k] for k in pos], [score[k] for k in neg])
    gw = weighted_auc([(score[k], group[k]) for k in pos], [(score[k], group[k]) for k in neg])
    return {
        "abusive": len(pos), "legitimate": len(neg),
        "abusive_groups": len({group[k] for k in pos}),
        "legitimate_groups": len({group[k] for k in neg}),
        "auc": None if auc is None else round(auc, 4),
        "auc_group_weighted": None if gw is None else round(gw, 4),
    }


def reading(auc: float) -> str:
    if auc <= WORLD_BAND:
        return "world/label problem (ceiling <= 0.65)"
    if auc >= METHOD_BAND:
        return "method problem (ceiling >= 0.80)"
    return "inconclusive (0.65 < ceiling < 0.80)"


def score(out: Path) -> dict[str, Any]:
    names, rows, manifest = load_features(out)  # label-free checks first
    members = first_members(set(rows))
    gt_url = f"sqlite:///{(WORLD / 'ground_truth.db').resolve()}"
    inf_url = f"sqlite:///{(WORLD / 'inference.db').resolve()}"
    with (
        Session(make_inference_engine(inf_url), future=True) as inf,
        Session(make_ground_truth_engine(gt_url), future=True) as gt,
    ):
        view = load_ground_truth(gt, inf_session=inf, splits=OPEN_SPLITS)
    group = {k: view.split_groups[m] for k, m in members.items()}
    label = {k: view.labels[g].value for k, g in group.items()}
    family = {k: view.families.get(g, "unknown") for k, g in group.items()}

    keys = sorted(k for k in rows if label[k] in ("abusive", "legitimate"))
    x = np.array([[rows[k]["features"][n] for n in names] for k in keys], dtype=float)
    y = np.array([label[k] == "abusive" for k in keys], dtype=int)
    groups = np.array([group[k] for k in keys])
    folds = list(StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
                 .split(x, y, groups))
    fold_of = {keys[i]: f for f, (_, test) in enumerate(folds) for i in test}

    oof: dict[str, dict[str, float]] = {}
    for name, model in _models().items():
        pred = np.full(len(keys), np.nan)
        for train, test in folds:
            model.fit(x[train], y[train])
            pred[test] = model.predict_proba(x[test])[:, 1]
        if not np.isfinite(pred).all():
            raise SystemExit(f"{name}: non-finite out-of-fold prediction")
        oof[name] = {k: float(p) for k, p in zip(keys, pred, strict=True)}
    detector = {k: float(rows[k]["detector_score"]) for k in keys}
    flags = sum(detector[k] >= SCORE_THRESHOLD for k in keys)
    scores = {**oof, "detector": detector}

    allowed = [k for k in keys if rows[k]["allowed"]]
    multi = [k for k in allowed if rows[k]["features"]["n_accounts"] >= 2]
    report_models: dict[str, dict[str, Any]] = {}
    for name, s in scores.items():
        report_models[name] = {
            "pooled": block(keys, s, label, group),
            "by_fold_auc": [block([k for k in keys if fold_of[k] == f], s, label, group)["auc"]
                            for f in range(FOLDS)],
            "at_detector_volume": at_volume(keys, s, flags, label, family),
            "allowed": block(allowed, s, label, group),
            "allowed_multi_account": block(multi, s, label, group),
        }
    ceiling = max(report_models[m]["pooled"]["auc"] for m in oof)
    report: dict[str, Any] = {
        "decision": "P6-4 (docs/decisions.md)", "scope": "open", "splits": list(OPEN_SPLITS),
        "labels": dict(Counter(label.values())),
        "families_abusive": dict(Counter(family[k] for k in keys if label[k] == "abusive")),
        "folds": [{"abusive": int(y[test].sum()), "legitimate": int(len(test) - y[test].sum()),
                   "abusive_groups": len({groups[i] for i in test if y[i]}),
                   "groups": len(set(groups[test]))} for _, test in folds],
        "detector_flags_binary": flags,
        "models": report_models,
        "ceiling_auc": ceiling,
        "reading": reading(ceiling),
        "constant_features": manifest["constant_features"],
        "inputs_sha256": {"features.jsonl": manifest["features_sha256"],
                          "manifest.json": _sha256(out / "manifest.json")},
        "scorer_sha256": _sha256(Path(__file__)),
    }
    (out / "score-open.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true",
                        help="label-free input checks only; reads no labels")
    args = parser.parse_args()
    if args.check_only:
        names, rows, manifest = load_features(args.out)
        first_members(set(rows))
        print(json.dumps({"units": len(rows), "features": len(names),
                          "allowed": sum(r["allowed"] for r in rows.values()),
                          "constant_features": manifest["constant_features"]}, indent=2))
        return
    print(json.dumps(score(args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

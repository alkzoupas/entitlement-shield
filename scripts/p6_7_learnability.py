"""P6-7/P6-7a: the multi-feature learnability check inside the Tier 5 audit (docs/decisions.md).

P6-4's method, byte-identical, on Tier 5's open scope. The unmodified feature code
(``scripts/p6_4_features.py``) and scoring helpers (``scripts/p6_4_ceiling.py``) are pointed
at the Tier 5 world and its base-detected store; neither file is edited, so their hashes stay
those P6-4 froze. Two subcommands, run in order:

* ``features``: label-free. Computes P6-4's 64 features for every train+tune detected unit
  and freezes them in ``manifest.json`` (with this script's hash) before any label is read.
* ``score``: checks the frozen inputs, reads train+tune labels only
  (``load_ground_truth(splits=("train", "tune"))``; the sealed test split is never read), fits
  the two fixed models in grouped cross-validation and reports, per model, the pooled and
  group-weighted out-of-fold AUC, per-fold AUC and per-family recall at the base-rate volume
  (as many flags as abusive units). The reading is P6-7's: ceiling >= 0.80 passes, <= 0.65
  rejects codex-01, otherwise the owner decides.

The detector score is carried in P6-4's feature rows but is never read here (P6-7a): no
comparator and no at-flag-volume metric. Scored once; the fitted models are discarded.

Usage:
    PYTHONPATH=backend .venv/bin/python scripts/p6_7_learnability.py features
    PYTHONPATH=backend .venv/bin/python scripts/p6_7_learnability.py score
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p6_4_ceiling as ceiling  # noqa: E402
import p6_4_features as features  # noqa: E402
from p6_recall_pilot import _sha256  # noqa: E402

from app.db.engine import make_inference_engine  # noqa: E402
from app.db.ground_truth_store import make_ground_truth_engine  # noqa: E402
from app.services.evaluation import load_ground_truth  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "data" / "tier5" / "world-v1"
DETECTED = WORLD / "runs" / "p6-7-detect" / "base" / "inference.db"
OUT = WORLD / "runs" / "p6-7-ceiling"
OPEN_SPLITS = ("train", "tune")
PASS_BAND, REJECT_BAND = 0.80, 0.65


def _point() -> None:
    """Aim the unmodified P6-4 modules at the Tier 5 stores (their globals only)."""
    for module in (features, ceiling):
        module.WORLD = WORLD  # type: ignore[attr-defined]
        module.DETECTED = DETECTED  # type: ignore[attr-defined]
        module.OPEN_SPLITS = OPEN_SPLITS  # type: ignore[attr-defined]


def build() -> dict[str, Any]:
    _point()
    if (OUT / "manifest.json").exists():
        raise SystemExit(f"{OUT} already holds frozen features; never rebuild after freezing")
    manifest = features.build(OUT)
    manifest.update({
        "decision": "P6-7/P6-7a (docs/decisions.md): P6-4 features on Tier 5 open scope",
        "world": str(WORLD.relative_to(ROOT)),
        "splits": list(OPEN_SPLITS),
        "runner_sha256": _sha256(Path(__file__)),
        "ceiling_helpers_sha256": _sha256(Path(ceiling.__file__)),
    })
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def reading(auc: float) -> str:
    if auc >= PASS_BAND:
        return "pass: learnable from combinations (ceiling >= 0.80)"
    if auc <= REJECT_BAND:
        return "reject codex-01: not learnable (ceiling <= 0.65)"
    return "inconclusive (0.65 < ceiling < 0.80): owner decides"


def score() -> dict[str, Any]:
    _point()
    if (OUT / "score-open.json").exists():
        raise SystemExit("already scored once (P6-7); no rescoring")
    names, rows, manifest = ceiling.load_features(OUT)  # label-free checks first
    for name, path in (("runner_sha256", Path(__file__)),
                       ("ceiling_helpers_sha256", Path(ceiling.__file__))):
        if _sha256(path) != manifest[name]:
            raise SystemExit(f"{path.name} changed after the features were frozen")
    members = ceiling.first_members(set(rows))
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
    folds = list(StratifiedGroupKFold(n_splits=ceiling.FOLDS, shuffle=True,
                                      random_state=ceiling.SEED).split(x, y, groups))
    fold_of = {keys[i]: f for f, (_, test) in enumerate(folds) for i in test}
    volume = int(y.sum())

    models: dict[str, dict[str, Any]] = {}
    for name, model in ceiling._models().items():
        pred = np.full(len(keys), np.nan)
        for train, test in folds:
            model.fit(x[train], y[train])
            pred[test] = model.predict_proba(x[test])[:, 1]
        if not np.isfinite(pred).all():
            raise SystemExit(f"{name}: non-finite out-of-fold prediction")
        s = {k: float(p) for k, p in zip(keys, pred, strict=True)}
        models[name] = {
            "pooled": ceiling.block(keys, s, label, group),
            "by_fold_auc": [
                ceiling.block([k for k in keys if fold_of[k] == f], s, label, group)["auc"]
                for f in range(ceiling.FOLDS)
            ],
            "at_base_rate_volume": ceiling.at_volume(keys, s, volume, label, family),
        }
    ceiling_auc = max(models[m]["pooled"]["auc"] for m in models)
    report: dict[str, Any] = {
        "decision": "P6-7/P6-7a (docs/decisions.md)", "scope": "open", "splits": list(OPEN_SPLITS),
        "labels": dict(Counter(label.values())),
        "families_abusive": dict(Counter(family[k] for k in keys if label[k] == "abusive")),
        "folds": [{"abusive": int(y[test].sum()), "legitimate": int(len(test) - y[test].sum()),
                   "abusive_groups": len({groups[i] for i in test if y[i]}),
                   "groups": len(set(groups[test]))} for _, test in folds],
        "models": models,
        "ceiling_auc": ceiling_auc,
        "reading": reading(ceiling_auc),
        "constant_features": manifest["constant_features"],
        "inputs_sha256": {"features.jsonl": manifest["features_sha256"],
                          "manifest.json": _sha256(OUT / "manifest.json")},
        "runner_sha256": _sha256(Path(__file__)),
    }
    (OUT / "score-open.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("features", "score"))
    args = parser.parse_args()
    result = build() if args.step == "features" else score()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

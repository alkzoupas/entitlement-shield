"""P6.3: the label-aware readout of the Nimble probe, exactly as pre-registered (P6-3).

Runs in the project venv after both arms of ``scripts/p6_3_nimble_run.py run`` finish.
Reads train+tune labels only (``load_ground_truth`` with the open splits); the sealed test
split is never read. Everything here is descriptive: units of one source group are not
independent, and no interval is claimed.

Before any label is read it checks, label-free, that both arms cover the same 492 keys with
valid rows at the ``fit.json`` lookback, were made from the exported renders, and reproduce
the smoke-test rows.

Per arm (score = expected level; the argmax level as a secondary):
* AUC abusive vs legitimate (ambiguous excluded): pooled, and within each accounts-in-view
  stratum (1 vs >= 2, counted from the render's Accounts section as in P6.1c). The primary
  readout is the multi-account stratum.
* Group-weighted AUC: each unit weighted 1/n_g, n_g = its source group's units within its
  class and the set being scored.
* Matched volume: binary-labelled units ranked by expected level (ties by a key order
  shuffled with seed 20261016); at the largest top-k whose legitimate flag share is <= 20%
  (and <= 10%): abusive catches, the random expectation and the lift.
Comparisons: (b) vs (a), and each arm's argmax vs P6.1's levels on the same keys.

Usage:
    PYTHONPATH=backend .venv/bin/python scripts/p6_3_score.py \
        --out data/tier4-v2/runs/p6-3-nimble
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p6_recall_pilot import (  # noqa: E402
    DETECTED,
    FLAG_LEVEL,
    OPEN_SPLITS,
    ROOT,
    WORLD,
    _auc,
    _levels,
    _open_units,
)

from app.db.engine import make_inference_engine  # noqa: E402
from app.db.ground_truth_store import make_ground_truth_engine  # noqa: E402
from app.services.evaluation import load_ground_truth  # noqa: E402

ARMS = ("a", "b")
SEED = 20261016
CEILINGS = (0.20, 0.10)
P61 = ROOT / "data" / "tier4-v2" / "runs" / "p6-pilot-1"
ACCOUNT_LINE = re.compile(r"^- A\d+: ")
SMOKE_TOLERANCE = 1e-3  # max abs probability gap between the smoke and full-run rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accounts_in_view(text: str) -> int:
    """Lines of the render's Accounts section (P6.1c's accounts in view)."""
    lines = text.split("\nAccounts (", 1)[1].split("\n\n", 1)[0].splitlines()[1:]
    if not lines or not all(ACCOUNT_LINE.match(line) for line in lines):
        raise SystemExit("render format changed; Accounts parser out of date")
    return len(lines)


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    by_key = {r["key"]: r for r in rows}
    if len(by_key) != len(rows):
        raise SystemExit(f"{path}: duplicate keys")
    return by_key


def load_arms(out: Path) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Both arms, checked label-free against the export, the fit and the smoke test."""
    export = json.loads((out / "export.json").read_text())
    fit = json.loads((out / "fit.json").read_text())
    keys = set(fit["units"])
    arms, checks = {}, {}
    for arm in ARMS:
        header = json.loads((out / f"run-{arm}.json").read_text())
        if header["renders_sha256"] != export["renders_sha256"]:
            raise SystemExit(f"arm {arm} was run on other renders")
        rows = _rows(out / f"arm-{arm}.jsonl")
        if set(rows) != keys:
            raise SystemExit(f"arm {arm}: {len(keys - set(rows))} keys missing, "
                             f"{len(set(rows) - keys)} unexpected")
        bad = [k for k, r in rows.items()
               if not r["valid"] or r["lookback"] != fit["units"][k]["lookback"]]
        if bad:
            raise SystemExit(f"arm {arm}: {len(bad)} invalid or off-fit rows")
        smoke = _rows(out / "smoke" / f"arm-{arm}.jsonl")
        gaps = {k: max(abs(p - q) for p, q in zip(r["probs"], rows[k]["probs"], strict=True))
                for k, r in smoke.items()}
        if max(gaps.values()) > SMOKE_TOLERANCE:
            raise SystemExit(f"arm {arm}: full run does not reproduce the smoke test {gaps}")
        arms[arm] = rows
        checks[arm] = {"units": len(rows), "smoke_max_abs_prob_gap": max(gaps.values()),
                       "model_safetensors_sha256": header["model_safetensors_sha256"],
                       "seconds_total": round(sum(r["seconds"] for r in rows.values()))}
    ids = {arm: {k: r["candidate_token_ids"] for k, r in rows.items()}
           for arm, rows in arms.items()}
    if ids["a"] != ids["b"]:
        raise SystemExit("the arms used different candidate tokens")
    return arms, {"fit_counts": fit["counts"], "arms": checks}


def _auc_f(pos: list[float], neg: list[float]) -> float | None:
    """``p6_recall_pilot._auc`` over float scores (the same pairwise count, ties = 0.5)."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def weighted_auc(pos: list[tuple[float, str]], neg: list[tuple[float, str]]) -> float | None:
    """AUC with each unit weighted 1/n_g within its class (n_g: its group's units there)."""
    if not pos or not neg:
        return None
    wp, wn = Counter(g for _, g in pos), Counter(g for _, g in neg)
    num = den = 0.0
    for sp, gp in pos:
        for sn, gn in neg:
            w = 1 / (wp[gp] * wn[gn])
            num += w * ((sp > sn) + 0.5 * (sp == sn))
            den += w
    return num / den


def matched_volume(keys: list[str], score: dict[str, float], label: dict[str, str],
                   group: dict[str, str], ceiling: float) -> dict[str, Any]:
    order = sorted(keys)
    random.Random(SEED).shuffle(order)
    tiebreak = {k: i for i, k in enumerate(order)}
    ranked = sorted(keys, key=lambda k: (-score[k], tiebreak[k]))
    n_pos = sum(label[k] == "abusive" for k in keys)
    n_neg = len(keys) - n_pos
    best, legit = 0, 0
    for i, k in enumerate(ranked, 1):
        legit += label[k] == "legitimate"
        if legit / n_neg <= ceiling:
            best = i
    top = ranked[:best]
    caught = [k for k in top if label[k] == "abusive"]
    expected = best * n_pos / len(keys)
    return {
        "ceiling": ceiling, "flags": best,
        "legitimate_flagged": best - len(caught),
        "legitimate_share": round((best - len(caught)) / n_neg, 4),
        "abusive_caught": len(caught), "abusive_groups_caught": len({group[k] for k in caught}),
        "random_expected": round(expected, 2),
        "lift": None if expected == 0 else round(len(caught) / expected, 3),
        "boundary_score": None if not top else round(score[top[-1]], 4),
    }


def arm_report(rows: dict[str, dict[str, Any]], label: dict[str, str], group: dict[str, str],
               stratum: dict[str, str]) -> dict[str, Any]:
    binary = sorted(k for k in rows if label[k] in ("abusive", "legitimate"))
    expected = {k: rows[k]["expected_level"] for k in rows}
    argmax = {k: float(rows[k]["argmax_level"]) for k in rows}

    def block(keys: list[str]) -> dict[str, Any]:
        pos = [k for k in keys if label[k] == "abusive"]
        neg = [k for k in keys if label[k] == "legitimate"]
        auc = _auc_f([expected[k] for k in pos], [expected[k] for k in neg])
        auc_arg = _auc_f([argmax[k] for k in pos], [argmax[k] for k in neg])
        gw = weighted_auc([(expected[k], group[k]) for k in pos],
                          [(expected[k], group[k]) for k in neg])
        return {
            "abusive": len(pos), "legitimate": len(neg),
            "abusive_groups": len({group[k] for k in pos}),
            "legitimate_groups": len({group[k] for k in neg}),
            "auc_expected": None if auc is None else round(auc, 4),
            "auc_group_weighted": None if gw is None else round(gw, 4),
            "auc_argmax": None if auc_arg is None else round(auc_arg, 4),
            "mean_expected_abusive": round(sum(expected[k] for k in pos) / len(pos), 3)
            if pos else None,
            "mean_expected_legitimate": round(sum(expected[k] for k in neg) / len(neg), 3)
            if neg else None,
        }

    return {
        "pooled": block(binary),
        "by_accounts_in_view": {s: block([k for k in binary if stratum[k] == s])
                                for s in ("1", "2+")},
        "matched_volume": [matched_volume(binary, expected, label, group, c) for c in CEILINGS],
        "argmax_histogram": {
            cls: dict(sorted(Counter(rows[k]["argmax_level"] for k in binary
                                     if label[k] == cls).items()))
            for cls in ("abusive", "legitimate")
        },
        f"argmax_ge_{FLAG_LEVEL}_share": {
            cls: round(sum(rows[k]["argmax_level"] >= FLAG_LEVEL for k in binary
                           if label[k] == cls)
                       / max(1, sum(label[k] == cls for k in binary)), 4)
            for cls in ("abusive", "legitimate")
        },
    }


def paired(a: dict[str, int], b: dict[str, int]) -> dict[str, Any]:
    keys = sorted(a)
    diffs = [b[k] - a[k] for k in keys]
    return {
        "paired": len(keys), "identical": sum(d == 0 for d in diffs),
        "abs_diff_histogram": dict(sorted(Counter(abs(d) for d in diffs).items())),
        "mean_diff_b_minus_a": round(sum(diffs) / len(diffs), 3),
        f"flag_flips_at_ge_{FLAG_LEVEL}": sum((a[k] >= FLAG_LEVEL) != (b[k] >= FLAG_LEVEL)
                                              for k in keys),
    }


def score(out: Path) -> dict[str, Any]:
    arms, checks = load_arms(out)  # label-free checks first
    renders = {json.loads(line)["key"]: json.loads(line)["renders"]["full"]
               for line in (out / "renders.jsonl").read_text().splitlines()}
    stratum = {k: "1" if accounts_in_view(t) == 1 else "2+" for k, t in renders.items()}
    members = dict(_open_units(DETECTED))
    stratum_vs_members = sum(accounts_in_view(renders[k]) == len(p["member_account_ids"])
                             for k, p in members.items())

    gt_url = f"sqlite:///{(WORLD / 'ground_truth.db').resolve()}"
    inf_url = f"sqlite:///{(WORLD / 'inference.db').resolve()}"
    with (
        Session(make_inference_engine(inf_url), future=True) as inf,
        Session(make_ground_truth_engine(gt_url), future=True) as gt,
    ):
        view = load_ground_truth(gt, inf_session=inf, splits=OPEN_SPLITS)
    group = {k: view.split_groups[p["member_account_ids"][0]] for k, p in members.items()}
    label = {k: view.labels[g].value for k, g in group.items()}

    p61 = {k: v for k, v in _levels(P61).items() if k in arms["a"]}
    p61_valid = {k: v for k, v in p61.items() if v is not None}
    argmax = {arm: {k: r["argmax_level"] for k, r in rows.items()} for arm, rows in arms.items()}
    binary = [k for k in p61_valid if label[k] in ("abusive", "legitimate")]
    p61_auc = _auc([p61_valid[k] for k in binary if label[k] == "abusive"],
                   [p61_valid[k] for k in binary if label[k] == "legitimate"])
    report: dict[str, Any] = {
        "decision": "P6-3 (docs/decisions.md)", "scope": "open", "splits": list(OPEN_SPLITS),
        "checks": {**checks, "accounts_in_view_equals_members": stratum_vs_members,
                   "units": len(members)},
        "labels": dict(Counter(label.values())),
        "strata_units": dict(Counter(stratum.values())),
        "arms": {arm: arm_report(rows, label, group, stratum) for arm, rows in arms.items()},
        "b_vs_a_argmax": paired(argmax["a"], argmax["b"]),
        "b_minus_a_mean_expected": round(sum(arms["b"][k]["expected_level"]
                                             - arms["a"][k]["expected_level"]
                                             for k in arms["a"]) / len(arms["a"]), 3),
        "vs_p61": {
            "p61_valid_units": len(p61_valid),
            "p61_auc_on_these_units": None if p61_auc is None else round(p61_auc, 4),
            **{f"arm_{arm}": paired({k: p61_valid[k] for k in p61_valid},
                                    {k: argmax[arm][k] for k in p61_valid})
               for arm in ARMS},
        },
        "inputs_sha256": {name: _sha256(out / name) for name in
                          ("export.json", "fit.json", "arm-a.jsonl", "arm-b.jsonl")},
        "scorer_sha256": _sha256(Path(__file__)),
    }
    if not all(math.isfinite(r["expected_level"]) for rows in arms.values()
               for r in rows.values()):
        raise SystemExit("non-finite expected level")
    (out / "score-open.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true",
                        help="label-free input checks and strata only; reads no labels")
    args = parser.parse_args()
    if args.check_only:
        _, checks = load_arms(args.out)
        print(json.dumps(checks, indent=2, sort_keys=True))
        return
    print(json.dumps(score(args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

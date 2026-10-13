"""P6.1c: anatomy of the recall layer's false positives (open scope, descriptive).

Reads the two arms of the P6.1b ablation run (``recall-v1`` and ``recall-v1-nocue``, the
same 492 open allowed units each) and splits the model's level-≥8 flags by legitimate-use
family (ground truth) and by the kind of evidence the model was shown. No model calls.

Evidence features are parsed from ``render_unit``'s output, i.e. exactly what the model
saw, plus two that are deliberately **not** shown to it, recomputed from the inference
store, to test whether the prompt drops evidence that would clear legitimate units:

* ``hidden_personal_devices``: accounts with a stable-device value no other cluster member
  uses (``render_unit`` lists only values seen on two or more accounts);
* ``hidden_unshared_signals``: all such account-private linkage values, any type.

The sealed test split is never read. Units of one source group are not independent: rates
are reported with the number of source groups, and intervals resample source groups.

Usage:
    PYTHONPATH=backend .venv/bin/python scripts/p6_fp_anatomy.py \
        --run data/tier4-v2/runs/p6-1b-ablation --out data/tier4-v2/runs/p6-1c-fp-anatomy
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p6_recall_pilot import (  # noqa: E402
    DETECTED,
    FLAG_LEVEL,
    OPEN_SPLITS,
    WORLD,
    _auc,
    _levels,
    _open_units,
    _ro,
    _ts,
    render_unit,
)

from app.db.engine import make_inference_engine  # noqa: E402
from app.db.ground_truth_store import make_ground_truth_engine  # noqa: E402
from app.services.evaluation import load_ground_truth  # noqa: E402

ARMS = ("recall-v1", "recall-v1-nocue")
SEED = 20261015
BOOT = 10000
ACCOUNT = re.compile(r"^- (A\d+): (-?\d+) days old, (\w+), (\w+)$")
ACTIVITY = re.compile(r"^- (A\d+): claims (\d+) .*verifications ok (\d+) / failed (\d+), "
                      r"active days (\d+), distinct active hours (\d+)$")
SHARED = re.compile(r"^- (\w+): (.*)$")


def features(conn: Any, payload: dict[str, Any], text: str) -> dict[str, Any]:
    """Label-free description of one unit, from the rendered prompt plus the hidden extras."""
    accounts, activity, shared = [], [], []
    section = ""
    for line in text.splitlines():
        if line.startswith("Accounts ("):
            section = "acc"
        elif line.startswith("Shared resources"):
            section = "shared"
        elif line.startswith("Activity inside"):
            section = "act"
        elif line.startswith("History"):
            section = "hist"
        elif section == "acc" and (m := ACCOUNT.match(line)):
            accounts.append((int(m[2]), m[3], m[4]))
        elif section == "shared" and (m := SHARED.match(line)) and m[1] != "none":
            shared.append(m[1])
        elif section == "act" and (m := ACTIVITY.match(line)):
            activity.append(tuple(int(m[i]) for i in range(2, 7)))
    if not accounts or len(activity) != len(accounts):
        raise SystemExit("render format changed; parser out of date")

    start, end = _ts(payload["window_start"]), _ts(payload["window_end"])
    claims = sorted(
        (_ts(e["timestamp"]), e["account_id"]) for e in payload["timeline"]
        if e["event_type"] == "claim_credits" and start <= _ts(e["timestamp"]) < end
    )
    gaps = [(b[0] - a[0]).total_seconds() / 60
            for a, b in zip(claims, claims[1:], strict=False) if a[1] != b[1]]
    min_gap = min(gaps) if gaps else None

    members = sorted(payload["member_account_ids"])
    marks = ",".join("?" * len(members))
    cutoff = f"{_ts(payload['as_of']):%Y-%m-%d %H:%M:%S.%f}"
    owners: dict[tuple[str, str], set[str]] = defaultdict(set)
    for account_id, signal, value in conn.execute(
        f"select account_id, signal_type, signal_value_hash from linkage_observations "
        f"where account_id in ({marks}) and timestamp <= ?", [*members, cutoff],
    ):
        owners[(signal, value)].add(account_id)
    private = [(s, next(iter(a))) for (s, _), a in owners.items() if len(a) == 1]
    return {
        "accounts": len(accounts),
        "shared_types": sorted(set(shared)),
        "min_age_days": min(a[0] for a in accounts),
        "any_disposable": any(a[1] == "disposable" for a in accounts),
        "any_unverified": any(a[2] == "none" for a in accounts),
        "failed_verifications": sum(a[2] for a in activity),
        "claims": sum(a[0] for a in activity),
        "max_claims_per_account": max(a[0] for a in activity),
        "min_cross_account_claim_gap_min": None if min_gap is None else round(min_gap, 1),
        "max_distinct_hours": max(a[4] for a in activity),
        "evidence_chars": len(text),
        "detector_score": payload["risk_score"],
        "hidden_personal_devices": len({a for s, a in private if s == "STABLE_DEVICE"}),
        "hidden_unshared_signals": len(private),
    }


def bins(f: dict[str, Any], ordinal: int) -> dict[str, str]:
    """One categorical value per evidence dimension (fixed cut points, set before scoring)."""
    gap = f["min_cross_account_claim_gap_min"]
    s = f["detector_score"]
    return {
        "accounts": str(min(f["accounts"], 4)) + ("+" if f["accounts"] >= 4 else ""),
        "shared_payment_or_recovery": str(bool({"payment_token", "recovery_contact"}
                                               & set(f["shared_types"]))),
        "shared_device": str("stable_device" in f["shared_types"]),
        "shared_template": str("behavioral_template" in f["shared_types"]),
        "shared_type_count": str(len(f["shared_types"])),
        "new_account_lt_30d": str(f["min_age_days"] < 30),
        "any_disposable_email": str(f["any_disposable"]),
        "any_unverified": str(f["any_unverified"]),
        "failed_verification_in_cycle": str(f["failed_verifications"] > 0),
        "claim_gap": ("no cross-account pair" if gap is None else "<= 60 min" if gap <= 60
                      else "<= 24 h" if gap <= 1440 else "> 24 h"),
        "max_claims_per_account": str(min(f["max_claims_per_account"], 2))
                                  + ("+" if f["max_claims_per_account"] >= 2 else ""),
        "cycle_ordinal": str(min(ordinal, 3)) + ("+" if ordinal >= 3 else ""),
        "evidence_chars": ("< 2k" if f["evidence_chars"] < 2000 else "2k-4k"
                           if f["evidence_chars"] < 4000 else ">= 4k"),
        "detector_band": "=0" if s == 0 else "(0,0.4)" if s < 0.4 else "[0.4,0.6)",
        "hidden_personal_device": str(f["hidden_personal_devices"] > 0),
    }


def share_ci(keys: list[str], flag: dict[str, bool], group: dict[str, str],
             rng: random.Random) -> list[float] | None:
    """95% interval of the flag share, resampling source groups (None under 5 groups)."""
    by: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        by[group[k]].append(k)
    gids = sorted(by)
    if len(gids) < 5:
        return None
    stats = []
    for _ in range(BOOT):
        ks = [k for g in rng.choices(gids, k=len(gids)) for k in by[g]]
        stats.append(sum(flag[k] for k in ks) / len(ks))
    stats.sort()
    return [round(stats[int(0.025 * BOOT)], 3), round(stats[int(0.975 * BOOT) - 1], 3)]


def cell(keys: list[str], flag: dict[str, dict[str, bool]], level: dict[str, int],
         group: dict[str, str]) -> dict[str, Any]:
    if not keys:
        return {"units": 0}
    return {
        "units": len(keys), "groups": len({group[k] for k in keys}),
        **{f"ge8_{arm}": round(sum(flag[arm][k] for k in keys) / len(keys), 3)
           for arm in ARMS},
        "mean_level_recall-v1": round(sum(level[k] for k in keys) / len(keys), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()

    units = dict(_open_units(DETECTED))
    levels = {}
    for arm in ARMS:
        lv = _levels(args.run, arm)
        if set(lv) != set(units) or any(v is None for v in lv.values()):
            raise SystemExit(f"{arm}: not exactly one valid level per open allowed unit")
        levels[arm] = {k: int(v) for k, v in lv.items() if v is not None}
    flag = {arm: {k: v >= FLAG_LEVEL for k, v in levels[arm].items()} for arm in ARMS}
    level = levels["recall-v1"]

    gt_url = f"sqlite:///{(WORLD / 'ground_truth.db').resolve()}"
    inf_url = f"sqlite:///{(WORLD / 'inference.db').resolve()}"
    with (
        Session(make_inference_engine(inf_url), future=True) as inf,
        Session(make_ground_truth_engine(gt_url), future=True) as gt,
    ):
        view = load_ground_truth(gt, inf_session=inf, splits=OPEN_SPLITS)

    # Cycle ordinal within each cluster snapshot, over every detected unit (label-free).
    starts: dict[str, list[str]] = defaultdict(list)
    with _ro(DETECTED) as conn:
        for (raw,) in conn.execute("select payload_json from detection_evidence"):
            p = json.loads(raw)
            starts[p["cluster_snapshot_id"]].append(p["window_start"])
    for v in starts.values():
        v.sort(key=_ts)

    group, label, family, feats, binned, texts = {}, {}, {}, {}, {}, {}
    with _ro(WORLD / "inference.db") as conn:
        for k, p in units.items():
            g = view.split_groups[p["member_account_ids"][0]]
            group[k], label[k], family[k] = g, view.labels[g].value, view.families[g]
            texts[k] = render_unit(conn, p)
            feats[k] = features(conn, p, texts[k])
            ordinal = starts[p["cluster_snapshot_id"]].index(p["window_start"]) + 1
            binned[k] = bins(feats[k], ordinal)

    rng = random.Random(SEED)
    legit = sorted(k for k in units if label[k] == "legitimate")
    abusive = sorted(k for k in units if label[k] == "abusive")

    fam_keys: dict[str, list[str]] = defaultdict(list)
    for k in legit:
        fam_keys[family[k]].append(k)
    by_family = {}
    for fam, ks in sorted(fam_keys.items(), key=lambda kv: -sum(flag["recall-v1"][k]
                                                                for k in kv[1]) / len(kv[1])):
        per_group: dict[str, list[bool]] = defaultdict(list)
        for k in ks:
            per_group[group[k]].append(flag["recall-v1"][k])
        by_family[fam] = {
            **cell(ks, flag, level, group),
            "ge8_ci95_recall-v1": share_ci(ks, flag["recall-v1"], group, rng),
            "groups_all_flagged": sum(all(v) for v in per_group.values()),
            "groups_none_flagged": sum(not any(v) for v in per_group.values()),
            "flagged_units": sum(flag["recall-v1"][k] for k in ks),
        }

    by_evidence: dict[str, dict[str, Any]] = {}
    for dim in binned[legit[0]]:
        values = sorted({binned[k][dim] for k in units})
        by_evidence[dim] = {
            v: {"legitimate": cell([k for k in legit if binned[k][dim] == v], flag, level, group),
                "abusive": cell([k for k in abusive if binned[k][dim] == v], flag, level, group)}
            for v in values
        }

    # Within-family: does the evidence separate flagged from unflagged legitimate units?
    # Rank-based (AUC of the feature, flagged vs unflagged, inside each family, pooled).
    numeric = ("accounts", "min_age_days", "failed_verifications", "claims",
               "max_distinct_hours", "evidence_chars", "detector_score",
               "hidden_personal_devices", "hidden_unshared_signals")
    within = {}
    for name in numeric:
        wins = total = 0.0
        for ks in fam_keys.values():
            pos = [feats[k][name] for k in ks if flag["recall-v1"][k]]
            neg = [feats[k][name] for k in ks if not flag["recall-v1"][k]]
            wins += sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
            total += len(pos) * len(neg)
        within[name] = round(wins / total, 3) if total else None
    gap_ks = [k for k in legit if feats[k]["min_cross_account_claim_gap_min"] is not None]
    for ks_name, ks_all in (("min_cross_account_claim_gap_min", gap_ks),):
        wins = total = 0.0
        for fam in fam_keys:
            ks = [k for k in ks_all if family[k] == fam]
            pos = [feats[k][ks_name] for k in ks if flag["recall-v1"][k]]
            neg = [feats[k][ks_name] for k in ks if not flag["recall-v1"][k]]
            wins += sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
            total += len(pos) * len(neg)
        within[ks_name] = round(wins / total, 3) if total else None

    # One account in view vs several: ranking and flag shares inside each stratum, and
    # each family split by stratum (labels: legitimate and abusive; ambiguous counted).
    strata: dict[str, Any] = {}
    for name, keep in (("single_account", lambda k: feats[k]["accounts"] == 1),
                       ("multi_account", lambda k: feats[k]["accounts"] > 1)):
        pos = [k for k in abusive if keep(k)]
        neg = [k for k in legit if keep(k)]
        by_fam: dict[str, dict[str, int]] = {}
        for k in sorted(pos + neg, key=lambda k: (label[k], family[k])):
            c = by_fam.setdefault(f"{family[k]} ({label[k]})", {"units": 0, "ge8_recall-v1": 0})
            c["units"] += 1
            c["ge8_recall-v1"] += flag["recall-v1"][k]
        strata[name] = {
            **{f"auc_{arm}": _auc([levels[arm][k] for k in pos], [levels[arm][k] for k in neg])
               for arm in ARMS},
            "abusive": cell(pos, flag, level, group),
            "legitimate": cell(neg, flag, level, group),
            "ambiguous_units": sum(keep(k) and label[k] == "ambiguous" for k in units),
            "level_histogram_recall-v1": {
                lab: dict(sorted(Counter(level[k] for k in ks).items()))
                for lab, ks in (("abusive", pos), ("legitimate", neg))
            },
            "family": by_fam,
        }
    # Among multi-account units: does any shown feature separate abusive from legitimate?
    multi_pos = [k for k in abusive if feats[k]["accounts"] > 1]
    multi_neg = [k for k in legit if feats[k]["accounts"] > 1]
    strata["multi_account"]["feature_auc_abusive_vs_legitimate"] = {
        name: _auc([feats[k][name] for k in multi_pos if feats[k][name] is not None],
                   [feats[k][name] for k in multi_neg if feats[k][name] is not None])
        for name in (*numeric, "min_cross_account_claim_gap_min")
    }
    negative_age = {lab: sum(feats[k]["min_age_days"] < 0 for k in units if label[k] == lab)
                    for lab in ("legitimate", "abusive", "ambiguous")}

    # Hidden evidence: how often the store holds account-private devices the prompt drops.
    hidden = {
        lab: {"units": len(ks),
              "with_hidden_personal_device": sum(feats[k]["hidden_personal_devices"] > 0
                                                 for k in ks)}
        for lab, ks in (("legitimate", legit), ("abusive", abusive))
    }
    hidden_by_family = {
        fam: round(sum(feats[k]["hidden_personal_devices"] > 0 for k in ks) / len(ks), 3)
        for fam, ks in sorted(fam_keys.items())
    }
    abusive_fams: dict[str, list[str]] = defaultdict(list)
    for k in abusive:
        abusive_fams[family[k]].append(k)
    for fam, ks in sorted(abusive_fams.items()):
        hidden_by_family[f"{fam} (abusive)"] = round(
            sum(feats[k]["hidden_personal_devices"] > 0 for k in ks) / len(ks), 3)

    # Transcript sample: flagged legitimate units from the worst families, one unflagged
    # unit from the same family next to each pair (seeded). Rendered evidence on disk.
    args.out.mkdir(parents=True, exist_ok=True)
    sample_dir = args.out / "samples"
    sample_dir.mkdir(exist_ok=True)
    picks: list[dict[str, Any]] = []
    worst = [f for f in by_family if by_family[f]["flagged_units"]][: args.samples // 3]
    for fam in worst:
        ks = fam_keys[fam]
        fp = sorted(k for k in ks if flag["recall-v1"][k])
        tn = sorted(k for k in ks if not flag["recall-v1"][k])
        chosen = rng.sample(fp, min(2, len(fp))) + rng.sample(tn, min(1, len(tn)))
        for k in chosen:
            name = f"{len(picks) + 1:02d}-{'FP' if flag['recall-v1'][k] else 'TN'}.txt"
            (sample_dir / name).write_text(
                f"# family: {fam}\n# source group: {group[k]}\n# key: {k}\n"
                f"# level recall-v1 {level[k]}, recall-v1-nocue {levels['recall-v1-nocue'][k]}\n"
                f"# hidden personal devices: {feats[k]['hidden_personal_devices']}\n\n"
                + texts[k] + "\n")
            picks.append({"file": name, "family": fam, "group": group[k], "key": k,
                          "level": level[k]})

    report = {
        "scope": "open", "splits": list(OPEN_SPLITS), "run": str(args.run),
        "flag_level": FLAG_LEVEL, "bootstrap": {"resamples": BOOT, "seed": SEED},
        "units": {"legitimate": len(legit), "abusive": len(abusive),
                  "ambiguous": sum(label[k] == "ambiguous" for k in units)},
        "auc_recall-v1": _auc([level[k] for k in abusive], [level[k] for k in legit]),
        "legitimate_by_family": by_family,
        "by_evidence": by_evidence,
        "within_family_auc_flagged_vs_unflagged_legitimate": within,
        "hidden_personal_device": hidden, "hidden_personal_device_share_by_family":
            hidden_by_family,
        "abusive_by_family": {fam: cell(ks, flag, level, group)
                              for fam, ks in sorted(abusive_fams.items())},
        "by_accounts_in_view": strata,
        "units_showing_a_negative_account_age": negative_age,
        "samples": picks,
    }
    (args.out / "anatomy.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.out / "features.jsonl").open("w") as sink:  # label-free
        for k in sorted(units):
            sink.write(json.dumps({"key": k, **feats[k], **{f"bin_{d}": v for d, v in
                                                          binned[k].items()}}) + "\n")
    print(json.dumps({k: report[k] for k in ("units", "auc_recall-v1", "by_accounts_in_view",
                                             "within_family_auc_flagged_vs_unflagged_legitimate",
                                             "hidden_personal_device")}, indent=1))


if __name__ == "__main__":
    main()

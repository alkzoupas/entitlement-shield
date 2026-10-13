"""P6.4: label-free features for the learnability ceiling (docs/decisions.md P6-4).

Selects every train+tune unit of the detected store, caught and allowed (split *membership*
only, as ``scripts/p6_2_render_prompts.py`` does), and computes the frozen non-LLM baseline's
feature list (T7, ``docs/p6.2-posttrain-plan.md`` D6 item 5) from the unit's detection payload
and the inference store:

* v2-render summaries: account count, ages (an account created mid-cycle counts as age 0 and
  as joined in the cycle, as v2 shows it), email-domain and verification mix, shared-resource
  counts by type, and per-account cycle activity (claims, credits claimed and consumed,
  consume uses, logins, verifications ok / failed, active days and hours) as mean and max;
* temporal summaries: claim timing inside the cycle, inter-claim gaps, claims across earlier
  cycles, signup-to-first-claim latency;
* the frozen F-* detector features (``raw_value`` of each contribution).

The detector score is kept beside the features as the comparator, never as a feature. No
ground-truth label, scenario family or split group is written. Every unit's per-account
activity is cross-checked against the text ``render_unit(..., render="v2")`` prints.

Usage:
    PYTHONPATH=backend .venv/bin/python scripts/p6_4_features.py \
        --out data/tier4-v2/runs/p6-4-ceiling
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p6_recall_pilot import (  # noqa: E402
    DETECTED,
    OPEN_SPLITS,
    WORLD,
    _key,
    _ro,
    _sha256,
    _ts,
    render_unit,
)

from app.db.engine import make_inference_engine  # noqa: E402
from app.db.ground_truth_reader import load_split_groups  # noqa: E402
from app.db.ground_truth_store import make_ground_truth_engine  # noqa: E402
from app.db.world_provenance import verify_world_pair  # noqa: E402
from app.services.evaluation import assign_split  # noqa: E402
from app.services.risk_scorer import SCORE_THRESHOLD  # noqa: E402

DOMAINS = ("CONSUMER", "DISPOSABLE", "EDUCATION", "ENTERPRISE")
LEVELS = ("NONE", "EMAIL", "PHONE", "ENHANCED")
# Linkage signal type -> feature-name stem ("group" is kept out of feature names).
SIGNALS = {"BEHAVIORAL_TEMPLATE": "behavior", "NETWORK_GROUP": "network",
           "PAYMENT_TOKEN": "payment", "RECOVERY_CONTACT": "recovery", "STABLE_DEVICE": "device"}
ACTIVITY = ("claims", "claim_credits", "consume_credits", "consume_uses", "logins",
            "verif_ok", "verif_failed", "active_days", "active_hours")
F_IDS = ("F-AGE-1", "F-CON-1", "F-CYC-1", "F-CYC-2", "F-CYC-3", "F-CYC-4", "F-DEV-1",
         "F-DEV-2", "F-DOM-1", "F-DOM-2", "F-PAY-1", "F-VER-1")
MISSING = -1.0  # fill for a temporal summary with no defined value (stated in P6-4)
CYCLE_SECONDS = 30 * 86400

FEATURES: tuple[str, ...] = (
    "n_accounts", "age_days_mean", "age_days_max", "age_days_min", "joined_in_cycle_frac",
    *(f"domain_{d.lower()}_frac" for d in DOMAINS),
    *(f"verification_{v.lower()}_frac" for v in LEVELS),
    *(f"shared_values_{s}" for s in SIGNALS.values()),
    *(f"shared_accounts_frac_{s}" for s in SIGNALS.values()),
    *(f"{a}_{stat}" for a in ACTIVITY for stat in ("mean", "max")),
    "claim_day_mean", "claim_span_h", "claim_gap_median_h", "claim_gap_min_h",
    "hist_claims_mean", "hist_claims_max", "hist_claim_cycles_mean", "hist_claim_cycles_max",
    "acct_claim_gap_median_d_mean", "signup_to_first_claim_h_median", "history_span_days",
    *(f.lower().replace('-', '_') for f in F_IDS),
)
FORBIDDEN = re.compile(r"label|scenario|group|family|split|truth|abus|legit|ambig")
ACTIVITY_LINE = re.compile(
    r"^- (A\d+): claims (\d+) \((\d+) credits\), consumed (\d+) credits in (\d+) uses, "
    r"logins (\d+), verifications ok (\d+) / failed (\d+), active days (\d+), "
    r"distinct active hours (\d+)$"
)


def _mean_max(values: list[float]) -> tuple[float, float]:
    return sum(values) / len(values), max(values)


def unit_features(conn: sqlite3.Connection, payload: dict[str, Any]
                  ) -> tuple[dict[str, float], dict[str, dict[str, int]]]:
    """The fixed feature vector of one unit, plus per-account activity for the render check."""
    members = sorted(payload["member_account_ids"])
    n = len(members)
    alias = {m: f"A{i + 1}" for i, m in enumerate(members)}
    start, end = _ts(payload["window_start"]), _ts(payload["window_end"])
    as_of = _ts(payload["as_of"])
    marks = ",".join("?" * n)
    f: dict[str, float] = {"n_accounts": float(n)}

    accounts = {a: (_ts(c), d, v) for a, c, d, v in conn.execute(
        f"select account_id, created_at, email_domain_type, verification_level "
        f"from accounts where account_id in ({marks})", members)}
    ages = [max(0, (start - accounts[m][0]).days) for m in members]
    f["age_days_mean"], f["age_days_max"] = _mean_max([float(a) for a in ages])
    f["age_days_min"] = float(min(ages))
    f["joined_in_cycle_frac"] = sum(accounts[m][0] > start for m in members) / n
    domains = Counter(accounts[m][1] for m in members)
    levels = Counter(accounts[m][2] for m in members)
    if set(domains) - set(DOMAINS) or set(levels) - set(LEVELS):
        raise SystemExit(f"unexpected domain or verification value: {domains} {levels}")
    for d in DOMAINS:
        f[f"domain_{d.lower()}_frac"] = domains[d] / n
    for v in LEVELS:
        f[f"verification_{v.lower()}_frac"] = levels[v] / n

    cutoff = f"{as_of:%Y-%m-%d %H:%M:%S.%f}"  # as render_unit: linkage up to as_of
    holders: dict[tuple[str, str], set[str]] = defaultdict(set)
    for account_id, signal, value in conn.execute(
        f"select account_id, signal_type, signal_value_hash from linkage_observations "
        f"where account_id in ({marks}) and timestamp <= ?", [*members, cutoff],
    ):
        holders[(signal, value)].add(account_id)
    if {s for s, _ in holders} - set(SIGNALS):
        raise SystemExit("unexpected signal type")
    for s, stem in SIGNALS.items():
        shared = [accts for (sig, _), accts in holders.items() if sig == s and len(accts) > 1]
        f[f"shared_values_{stem}"] = float(len(shared))
        f[f"shared_accounts_frac_{stem}"] = len(set().union(*shared)) / n

    events = sorted((e for e in payload["timeline"] if e["event_type"] != "appeal"),
                    key=lambda e: (e["timestamp"], e["event_id"]))
    act: dict[str, Counter[str]] = {alias[m]: Counter() for m in members}
    hours: dict[str, set[int]] = defaultdict(set)
    days: dict[str, set[str]] = defaultdict(set)
    cycle_claims: list[float] = []  # seconds since cycle start
    claims_at: dict[str, list[float]] = defaultdict(list)  # epoch seconds, whole timeline
    for e in events:
        t = _ts(e["timestamp"])
        if t >= end:
            raise SystemExit(f"unit {_key(payload)}: timeline past window_end")
        if e["event_type"] == "claim_credits":
            claims_at[e["account_id"]].append(t.timestamp())
        if not start <= t:
            continue
        a, c = alias[e["account_id"]], act[alias[e["account_id"]]]
        kind, credits = e["event_type"], e["credits_amount"] or 0
        if kind == "claim_credits":
            c["claims"] += 1
            c["claim_credits"] += credits
            cycle_claims.append((t - start).total_seconds())
        elif kind == "consume_credits":
            c["consume_uses"] += 1
            c["consume_credits"] += credits
        elif kind == "login":
            c["logins"] += 1
        elif kind == "verification":
            c["verif_ok"] += 1
        elif kind == "verification_failed":
            c["verif_failed"] += 1
        hours[a].add(t.hour)
        days[a].add(f"{t:%m-%d}")
    for a in act:
        act[a]["active_days"], act[a]["active_hours"] = len(days[a]), len(hours[a])
    for name in ACTIVITY:
        f[f"{name}_mean"], f[f"{name}_max"] = _mean_max([float(act[a][name]) for a in act])

    window_h = (end - start).total_seconds() / 3600
    cycle_claims.sort()
    gaps = [(b - a) / 3600 for a, b in zip(cycle_claims, cycle_claims[1:], strict=False)]
    f["claim_day_mean"] = (sum(cycle_claims) / len(cycle_claims) / 86400 if cycle_claims
                           else MISSING)
    f["claim_span_h"] = (cycle_claims[-1] - cycle_claims[0]) / 3600 if cycle_claims else MISSING
    f["claim_gap_median_h"] = statistics.median(gaps) if gaps else window_h
    f["claim_gap_min_h"] = min(gaps) if gaps else window_h

    s0 = start.timestamp()
    hist = [[t for t in claims_at[m] if t < s0] for m in members]
    f["hist_claims_mean"], f["hist_claims_max"] = _mean_max([float(len(h)) for h in hist])
    cycles = [float(len({int((s0 - t) // CYCLE_SECONDS) for t in h})) for h in hist]
    f["hist_claim_cycles_mean"], f["hist_claim_cycles_max"] = _mean_max(cycles)
    acct_gaps = [statistics.median([(b - a) / 86400 for a, b in zip(ts, ts[1:], strict=False)])
                 for ts in (claims_at[m] for m in members) if len(ts) > 1]
    f["acct_claim_gap_median_d_mean"] = sum(acct_gaps) / len(acct_gaps) if acct_gaps else MISSING
    latency = [(claims_at[m][0] - accounts[m][0].timestamp()) / 3600
               for m in members if claims_at[m]]
    f["signup_to_first_claim_h_median"] = statistics.median(latency) if latency else MISSING
    first = _ts(events[0]["timestamp"]) if events else start
    f["history_span_days"] = float(max(0, (start - first).days))

    contributions = {c["feature_id"]: c["raw_value"] for c in payload["contributions"]}
    if set(contributions) != set(F_IDS):
        raise SystemExit(f"unit {_key(payload)}: F-* set changed: {sorted(contributions)}")
    for fid in F_IDS:
        f[fid.lower().replace('-', '_')] = float(contributions[fid])
    if set(f) != set(FEATURES) or len(f) != len(FEATURES):
        raise SystemExit("computed features drifted from FEATURES")
    return {name: f[name] for name in FEATURES}, {a: dict(c) for a, c in act.items()}


def render_check(conn: sqlite3.Connection, payload: dict[str, Any],
                 act: dict[str, dict[str, int]]) -> None:
    """The per-account activity must equal what the v2 render prints for the unit."""
    text = render_unit(conn, payload, render="v2")
    section = text.split("\nActivity inside this cycle, per account:\n", 1)[1].split("\n\n")[0]
    seen = {}
    for line in section.splitlines():
        m = ACTIVITY_LINE.match(line)
        if m is None:
            raise SystemExit(f"render format changed: {line!r}")
        seen[m[1]] = tuple(int(x) for x in m.groups()[1:])
    order = ("claims", "claim_credits", "consume_credits", "consume_uses", "logins",
             "verif_ok", "verif_failed", "active_days", "active_hours")
    mine = {a: tuple(c.get(k, 0) for k in order) for a, c in act.items()}
    if seen != mine:
        raise SystemExit(f"unit {_key(payload)}: features disagree with the v2 render")


def build(out: Path) -> dict[str, Any]:
    if any(FORBIDDEN.search(name) for name in FEATURES):
        raise SystemExit("a feature name looks label-derived")
    gt_url = f"sqlite:///{(WORLD / 'ground_truth.db').resolve()}"
    inf_url = f"sqlite:///{(WORLD / 'inference.db').resolve()}"
    with (
        Session(make_inference_engine(inf_url), future=True) as inf,
        Session(make_ground_truth_engine(gt_url), future=True) as gt,
    ):
        verify_world_pair(inf, gt)
        groups = load_split_groups(gt)  # label-free membership, used only to select open units

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with _ro(DETECTED) as det, _ro(WORLD / "inference.db") as conn:
        for (raw,) in det.execute("select payload_json from detection_evidence"):
            payload = json.loads(raw)
            split_groups = {groups.get(m) for m in payload["member_account_ids"]}
            if len(split_groups) != 1 or None in split_groups:
                raise SystemExit(f"unit {_key(payload)} has no single split group")
            if assign_split(next(iter(split_groups))) not in OPEN_SPLITS:  # type: ignore[arg-type]
                continue
            features, act = unit_features(conn, payload)
            render_check(conn, payload, act)
            if not all(math.isfinite(v) for v in features.values()):
                raise SystemExit(f"unit {_key(payload)}: non-finite feature")
            rows.append({"key": _key(payload), "detector_score": payload["risk_score"],
                         "allowed": payload["risk_score"] < SCORE_THRESHOLD,
                         "features": features})
    rows.sort(key=lambda r: r["key"])
    if len({r["key"] for r in rows}) != len(rows):
        raise SystemExit("duplicate unit keys")
    path = out / "features.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    constant = [n for n in FEATURES if len({r["features"][n] for r in rows}) == 1]
    manifest = {
        "decision": "P6-4 (docs/decisions.md)",
        "features": list(FEATURES),
        "missing_fill": MISSING,
        "units": len(rows),
        "allowed": sum(r["allowed"] for r in rows),
        "caught": sum(not r["allowed"] for r in rows),
        "constant_features": constant,
        "features_sha256": _sha256(path),
        "feature_script_sha256": _sha256(Path(__file__)),
        "pilot_sha256": _sha256(Path(__file__).resolve().parent / "p6_recall_pilot.py"),
        "detected_store_sha256": _sha256(DETECTED),
        "inference_store_sha256": _sha256(WORLD / "inference.db"),
        "render_check": "per-account activity equals render_unit v2 on every unit",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(build(parser.parse_args().out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

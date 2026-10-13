"""P6 pilot: an LLM as a second recall layer over units the frozen detector allows.

Coinbase Onramp shape (`docs/coinbase/`, Part 1): the cheap first stage screens every
unit; the LLM reviews only what it allowed, returns ``RISK_LEVEL: 1-10`` and nothing
else, and application code decides. It can only add a flag, never clear one.

Two phases, deliberately separate:

* ``run`` (label-free). Reads the per-unit detection evidence of an already-detected
  store (read-only), keeps units with ``risk_score`` below the frozen threshold whose
  split group is in train+tune (label-free membership only), renders each as a
  compressed history plus per-account summaries, and asks the model once at
  temperature 0. Appends to ``<out>/outcomes.jsonl`` (resumable) and keeps a replay
  transcript. The detector's own score is never shown to the model.
* ``score --scope open``. Reads train+tune labels and reports, among allowed units,
  how well the risk level separates abusive from legitimate, and what each level
  threshold adds to the detector's recall, by unit and by cluster.

The sealed test split is never read here. This is open-scope development (P5-7a).

Usage:
    PYTHONPATH=backend .venv/bin/python scripts/p6_recall_pilot.py run \
        --config config/recall-provider.local-qwen35-9b.yaml --out <dir> [--limit N]
    PYTHONPATH=backend .venv/bin/python scripts/p6_recall_pilot.py score --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.db.engine import make_inference_engine
from app.db.ground_truth_reader import load_split_groups
from app.db.ground_truth_store import make_ground_truth_engine
from app.db.world_provenance import verify_world_pair
from app.providers.cache import CachingProvider, ResponseCache
from app.providers.factory import build_provider, describe_provider, load_provider_config
from app.services.evaluation import assign_split, load_ground_truth
from app.services.llm_provider import LLMRequest, ProviderException
from app.services.prompt_builder import StructuredEvidenceSegment
from app.services.risk_scorer import SCORE_THRESHOLD

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "data" / "tier4-v2"
# Detected at world_end by the P5.7 run; the evidence of every unit (allowed or not).
DETECTED = WORLD / "runs" / "p5-7-live-2" / "base" / "inference.db"
OPEN_SPLITS = ("train", "tune")
PROMPT_VERSION = "recall-v1"
LEVEL = re.compile(r"RISK_LEVEL:\s*(\d{1,2})\b")
LEVELS = range(1, 11)

SYSTEM_PROMPT = """You review accounts in StudioAI's free-credit program.

Program rule: each person gets one free allocation of 25 credits per 30-day cycle.
Creating or operating several accounts to collect extra free allocations is abuse.
Shared access is not abuse by itself: households, schools, workplaces and public
networks legitimately share devices, networks and payment methods.

You see one cluster of accounts linked by shared resources, for one credit cycle.
An automated detector has already reviewed this cluster and did not flag it. Your job
is a second, independent look for coordinated abuse the detector may have missed:
for example accounts that behave like one operator (same timing, same routines,
claim-and-drain patterns, throwaway signups, repeated failed verification), or a
cluster collecting far more allocations than its activity suggests real people.

Weigh evidence of independent people (different daily rhythms, different usage,
long-lived accounts, verified identities) against evidence of one operator.

Answer with exactly one line and nothing else:
RISK_LEVEL: <integer 1-10>
1 = clearly independent legitimate users, 10 = clearly coordinated abuse."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def _ts(value: str) -> datetime:
    """Payload timestamps are aware; SQLite stores naive UTC. Compare both as UTC."""
    parsed = datetime.fromisoformat(value.replace(" ", "T"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _world_end(world: Path) -> datetime:
    return datetime.fromisoformat(json.loads((world / "manifest.json").read_text())["world_end"])


def _key(payload: dict[str, Any]) -> str:
    return "|".join((payload["cluster_snapshot_id"], payload["cycle_id"], payload["as_of"]))


# --------------------------------------------------------------------------- render


def render_unit(conn: sqlite3.Connection, payload: dict[str, Any]) -> str:
    """Compressed, label-free view of one unit: who, what they share, what they did."""
    members = sorted(payload["member_account_ids"])
    alias = {m: f"A{i + 1}" for i, m in enumerate(members)}
    start, end = _ts(payload["window_start"]), _ts(payload["window_end"])
    as_of = _ts(payload["as_of"])
    marks = ",".join("?" * len(members))
    lines = [
        f"Credit cycle: {start:%Y-%m-%d} to {end:%Y-%m-%d}. Accounts in cluster: {len(members)}.",
        "",
        "Accounts (age at cycle start, email domain type, verification level):",
    ]
    accounts = {
        account_id: (created, domain, level)
        for account_id, created, domain, level in conn.execute(
            f"select account_id, created_at, email_domain_type, verification_level "
            f"from accounts where account_id in ({marks})", members,
        )
    }
    for m in members:
        created, domain, level = accounts[m]
        age = (start - _ts(created)).days
        lines.append(f"- {alias[m]}: {age} days old, {domain.lower()}, {level.lower()}")

    shared: dict[str, list[list[str]]] = defaultdict(list)
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    cutoff = f"{as_of:%Y-%m-%d %H:%M:%S.%f}"  # SQLite's naive-UTC text format
    for account_id, signal, value in conn.execute(
        f"select account_id, signal_type, signal_value_hash from linkage_observations "
        f"where account_id in ({marks}) and timestamp <= ?", [*members, cutoff],
    ):
        groups[(signal, value)].add(account_id)
    for (signal, _), accts in sorted(groups.items()):
        if len(accts) > 1:
            shared[signal.lower()].append(sorted(alias[a] for a in accts))
    lines += ["", "Shared resources (accounts observed using the same value):"]
    if not shared:
        lines.append("- none recorded")
    for signal, sets in sorted(shared.items()):
        uniq = sorted({" ".join(s) for s in sets})
        lines.append(f"- {signal}: " + "; ".join(f"[{u}]" for u in uniq))

    events = [e for e in payload["timeline"] if e["event_type"] != "appeal"]
    per: dict[str, Counter[str]] = defaultdict(Counter)
    credits: dict[str, Counter[str]] = defaultdict(Counter)
    hours: dict[str, set[int]] = defaultdict(set)
    days: dict[str, set[str]] = defaultdict(set)
    for e in events:
        t = _ts(e["timestamp"])
        if not start <= t < end:
            continue
        a = alias[e["account_id"]]
        per[a][e["event_type"]] += 1
        credits[a][e["event_type"]] += e["credits_amount"] or 0
        hours[a].add(t.hour)
        days[a].add(f"{t:%m-%d}")
    lines += ["", "Activity inside this cycle, per account:"]
    for a in sorted(alias.values(), key=lambda s: int(s[1:])):
        c = per[a]
        lines.append(
            f"- {a}: claims {c['claim_credits']} ({credits[a]['claim_credits']} credits), "
            f"consumed {credits[a]['consume_credits']} credits in {c['consume_credits']} uses, "
            f"logins {c['login']}, verifications ok {c['verification']} / failed "
            f"{c['verification_failed']}, active days {len(days[a])}, "
            f"distinct active hours {len(hours[a])}"
        )

    # History: one line per day per account and event type, so logins and small
    # consumes do not drown the claims, signups and verifications.
    daily: dict[tuple[str, str, str], list[int]] = {}
    first_time: dict[tuple[str, str, str], str] = {}
    for e in events:
        t = _ts(e["timestamp"])
        k = (f"{t:%Y-%m-%d}", alias[e["account_id"]], e["event_type"])
        n, total = daily.get(k, [0, 0])
        daily[k] = [n + 1, total + (e["credits_amount"] or 0)]
        first_time.setdefault(k, f"{t:%H:%M}")
    lines += ["", "History (date, first time, account, event, count, credits; "
              f"lines before {start:%Y-%m-%d} are earlier context):"]
    for (day, a, kind), (n, total) in sorted(
        daily.items(), key=lambda kv: (kv[0][0], first_time[kv[0]], kv[0][1], kv[0][2])
    ):
        extra = f" x{n}" if n > 1 else ""
        extra += f" {total} credits" if total else ""
        lines.append(f"{day} {first_time[(day, a, kind)]} {a} {kind}{extra}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------ run


def _open_units(detected: Path) -> list[tuple[str, dict[str, Any]]]:
    gt_url = f"sqlite:///{(WORLD / 'ground_truth.db').resolve()}"
    inf_url = f"sqlite:///{(WORLD / 'inference.db').resolve()}"
    with (
        Session(make_inference_engine(inf_url), future=True) as inf,
        Session(make_ground_truth_engine(gt_url), future=True) as gt,
    ):
        verify_world_pair(inf, gt)
        groups = load_split_groups(gt)  # label-free membership
    units = []
    with _ro(detected) as conn:
        for (raw,) in conn.execute("select payload_json from detection_evidence"):
            payload = json.loads(raw)
            if payload["risk_score"] >= SCORE_THRESHOLD:
                continue
            split_groups = {groups.get(m) for m in payload["member_account_ids"]}
            if len(split_groups) != 1 or None in split_groups:
                raise SystemExit(f"unit {_key(payload)} has no single split group")
            if assign_split(next(iter(split_groups))) in OPEN_SPLITS:  # type: ignore[arg-type]
                units.append((_key(payload), payload))
    units.sort(key=lambda u: u[0])
    return units


def run(config_path: Path, out: Path, limit: int | None, max_tokens: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    config = load_provider_config(config_path)
    header = {
        "decision": "P6 pilot (open scope)",
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "runner_sha256": _sha256(Path(__file__)),
        "config_sha256": _sha256(config_path),
        "provider": describe_provider(config),
        "detected_store_sha256": _sha256(DETECTED),
        "threshold": SCORE_THRESHOLD,
    }
    header_path = out / "run.json"
    if header_path.exists():
        previous = json.loads(header_path.read_text())
        if previous != header:
            raise SystemExit(f"{header_path} differs from this run; use a new --out")
    header_path.write_text(json.dumps(header, indent=2, sort_keys=True) + "\n")

    outcomes_path = out / "outcomes.jsonl"
    done = set()
    if outcomes_path.exists():
        done = {json.loads(line)["key"] for line in outcomes_path.read_text().splitlines()}
    cache = ResponseCache(path=out / "transcript.jsonl")
    cache.load()
    provider = CachingProvider(inner=build_provider(config), cache=cache)

    units = _open_units(DETECTED)
    todo = [u for u in units if u[0] not in done][: limit if limit else None]
    print(f"open allowed units {len(units)}, done {len(done)}, this run {len(todo)}", flush=True)
    with _ro(WORLD / "inference.db") as conn, outcomes_path.open("a") as sink:
        for i, (key, payload) in enumerate(todo, 1):
            evidence = render_unit(conn, payload)
            request = LLMRequest(
                system=SYSTEM_PROMPT,
                segments=(StructuredEvidenceSegment(evidence),),
                temperature=0.0, max_tokens=max_tokens, timeout_s=600.0,
            )
            row: dict[str, Any] = {"key": key, "detector_score": payload["risk_score"],
                                   "evidence_chars": len(evidence)}
            started = time.monotonic()
            try:
                response = provider.complete(request)
                match = LEVEL.search(response.text)
                level = int(match.group(1)) if match else None
                row.update(
                    level=level if level in LEVELS else None, text=response.text[:200],
                    prompt_tokens=response.usage.prompt_tokens,
                    latency_ms=round(response.latency_ms),
                )
                if row["level"] is None:
                    row["error"] = "unparseable"
            except ProviderException as exc:
                row.update(level=None, error=f"{type(exc).__name__}: {str(exc)[:200]}")
            row["wall_s"] = round(time.monotonic() - started, 1)
            sink.write(json.dumps(row, sort_keys=True) + "\n")
            sink.flush()
            cache.save()
            print(f"[{i}/{len(todo)}] level={row['level']} tokens={row.get('prompt_tokens')} "
                  f"{row['wall_s']}s {row.get('error', '')}", flush=True)
    failures = sum(
        1 for line in outcomes_path.read_text().splitlines() if json.loads(line)["level"] is None
    )
    print(f"failures in outcomes: {failures}", flush=True)
    if failures:
        raise SystemExit(1)


# ---------------------------------------------------------------------------- score


def _auc(pos: list[int], neg: list[int]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def score(out: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in (out / "outcomes.jsonl").read_text().splitlines()]
    gt_url = f"sqlite:///{(WORLD / 'ground_truth.db').resolve()}"
    inf_url = f"sqlite:///{(WORLD / 'inference.db').resolve()}"
    with (
        Session(make_inference_engine(inf_url), future=True) as inf,
        Session(make_ground_truth_engine(gt_url), future=True) as gt,
    ):
        view = load_ground_truth(gt, inf_session=inf, splits=OPEN_SPLITS)
    members: dict[str, list[str]] = {}
    caught_units = Counter[str]()
    positives_total = Counter[str]()
    with _ro(DETECTED) as conn:
        for (raw,) in conn.execute("select payload_json from detection_evidence"):
            payload = json.loads(raw)
            group = view.split_groups[payload["member_account_ids"][0]]
            if group not in view.labels:
                continue  # test split: never labelled here
            members[_key(payload)] = payload["member_account_ids"]
            if view.labels[group].value == "abusive":
                positives_total[group] += 1
                if payload["risk_score"] >= SCORE_THRESHOLD:
                    caught_units[group] += 1

    def label(key: str) -> tuple[str, str]:
        group = view.split_groups[members[key][0]]
        return group, view.labels[group].value

    valid = [r for r in rows if r["level"] is not None]
    pos = [r for r in valid if label(r["key"])[1] == "abusive"]
    neg = [r for r in valid if label(r["key"])[1] == "legitimate"]
    total_pos = sum(positives_total.values())
    detector_tp = sum(caught_units.values())
    detector_groups = {g for g, n in caught_units.items() if n}
    sweep = []
    for t in LEVELS:
        add_pos = [r for r in pos if r["level"] >= t]
        add_neg = [r for r in neg if r["level"] >= t]
        new_groups = {label(r["key"])[0] for r in add_pos} - detector_groups
        sweep.append({
            "flag_at_level_ge": t,
            "added_abusive_units": len(add_pos),
            "added_legitimate_units": len(add_neg),
            "added_abusive_clusters": len({label(r["key"])[0] for r in add_pos}),
            "clusters_with_no_detector_catch_now_caught": len(new_groups),
            "combined_unit_recall": round((detector_tp + len(add_pos)) / total_pos, 4),
        })
    by_band: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"abusive": [], "legit": []})
    for kind, group_rows in (("abusive", pos), ("legit", neg)):
        for r in group_rows:
            s = r["detector_score"]
            band = "=0" if s == 0 else "(0,0.4)" if s < 0.4 else "[0.4,0.6)"
            by_band[band][kind].append(r["level"])
    report = {
        "scope": "open", "splits": list(OPEN_SPLITS),
        "reviewed": len(rows), "valid": len(valid),
        "allowed_abusive_units": len(pos), "allowed_legitimate_units": len(neg),
        "allowed_abusive_clusters": len({label(r["key"])[0] for r in pos}),
        "detector_unit_recall": round(detector_tp / total_pos, 4),
        "auc_level_abusive_vs_legitimate": _auc([r["level"] for r in pos],
                                                [r["level"] for r in neg]),
        "level_histogram": {
            "abusive": dict(sorted(Counter(r["level"] for r in pos).items())),
            "legitimate": dict(sorted(Counter(r["level"] for r in neg).items())),
        },
        "auc_by_detector_band": {
            b: {"auc": _auc(v["abusive"], v["legit"]), "abusive": len(v["abusive"]),
                "legitimate": len(v["legit"])}
            for b, v in sorted(by_band.items())
        },
        "threshold_sweep": sweep,
    }
    (out / "score-open.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--max-tokens", type=int, default=16)
    s = sub.add_parser("score")
    s.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("render", help="print one unit's prompt evidence (label-free)")
    p.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.cmd == "run":
        run(args.config, args.out, args.limit, args.max_tokens)
    elif args.cmd == "score":
        print(json.dumps(score(args.out), indent=2, sort_keys=True))
    else:
        key, payload = _open_units(DETECTED)[args.index]
        with _ro(WORLD / "inference.db") as conn:
            text = render_unit(conn, payload)
        print(f"# {key}  ({len(text)} chars)\n{text}")


if __name__ == "__main__":
    main()

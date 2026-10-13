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
        --config config/recall-provider.local-qwen35-9b.yaml --out <dir> [--limit N] \
        [--prompt recall-v1|recall-v1-nocue] [--keys <keys.json>]
    PYTHONPATH=backend .venv/bin/python scripts/p6_recall_pilot.py score --out <dir>
    PYTHONPATH=backend .venv/bin/python scripts/p6_recall_pilot.py sample \
        --reference <dir> --n 50 --seed 20261013 --keys <keys.json>
    PYTHONPATH=backend .venv/bin/python scripts/p6_recall_pilot.py compare --a <dir> --b <dir>

``sample`` and ``compare`` are label-free (P6.1b, cue ablation): ``sample`` draws a seeded
subset stratified only by the reference run's own level (>= 8 or not); ``compare`` pairs
two runs by unit key and reports how far the levels moved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
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

# P6.1b: the same prompt without the routing cue (the detector passed this cluster; look
# for what it missed). Everything else is byte-identical to recall-v1.
NOCUE_PROMPT = SYSTEM_PROMPT.replace(
    "An automated detector has already reviewed this cluster and did not flag it. Your job\n"
    "is a second, independent look for coordinated abuse the detector may have missed:\n",
    "Your job is to judge whether this cluster shows coordinated abuse:\n",
)
assert NOCUE_PROMPT != SYSTEM_PROMPT
PROMPTS = {PROMPT_VERSION: SYSTEM_PROMPT, "recall-v1-nocue": NOCUE_PROMPT}
RENDERS = ("v1", "v2", "v3")  # v2 fixes negative ages (P6.1c §3.5); v3 compacts history (P6.3)
# v3 short event names; every other event type keeps its own name.
V3_EVENTS = {"claim_credits": "claim", "consume_credits": "consume",
             "verification": "verify_ok", "verification_failed": "verify_fail"}
FLAG_LEVEL = 8  # P6.1 reading threshold
TEMPERATURE = 0.0
# P6.1b readout (pre-registered in docs/p6.1b-cue-ablation.md §5).
MIN_GROUPS = 30  # legitimate source groups needed for the group-bootstrap interval
MIN_EFFECT = 0.10  # smallest change in a flag share that counts
AUC_STEP = 0.05  # descriptive ranking bar (4 abusive source groups)


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


def render_unit(conn: sqlite3.Connection, payload: dict[str, Any], *,
                render: str = "v1", lookback: int | None = None) -> str:
    """Compressed, label-free view of one unit: who, what they share, what they did.

    ``v1`` is what P6.1, P6.1b and P6.1c saw (and what ``p6_fp_anatomy`` parses): an account
    created mid-cycle shows a negative age (P6.1c §3.5). ``v2`` (P6.2) shows such an account
    as joining on day N of the cycle instead; everything else is identical.

    ``v3`` (P6.3) is ``v2`` with the History section compacted losslessly: one line per day,
    the date printed once, short event names, ``Ncr`` for N credits. ``lookback`` (v3 only)
    is the fixed fallback for a token cap: history days more than ``lookback`` days before
    the cycle start are rolled up into per-week, per-account, per-event counts and credit
    sums. ``None`` keeps every day in full. The roll-up adds nothing the lines did not hold.
    """
    if render not in RENDERS:
        raise ValueError(f"render must be one of {RENDERS}")
    if lookback is not None and (render != "v3" or lookback < 0):
        raise ValueError("lookback needs render='v3' and a non-negative number of days")
    members = sorted(payload["member_account_ids"])
    alias = {m: f"A{i + 1}" for i, m in enumerate(members)}
    start, end = _ts(payload["window_start"]), _ts(payload["window_end"])
    as_of = _ts(payload["as_of"])
    marks = ",".join("?" * len(members))
    heading = ("age at cycle start" if render == "v1"
               else "age at cycle start, or the cycle day it joined")
    lines = [
        f"Credit cycle: {start:%Y-%m-%d} to {end:%Y-%m-%d}. Accounts in cluster: {len(members)}.",
        "",
        f"Accounts ({heading}, email domain type, verification level):",
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
        joined = _ts(created)
        if render != "v1" and joined > start:
            when = f"joined on day {(joined - start).days + 1} of this cycle"
        else:
            when = f"{(start - joined).days} days old"
        lines.append(f"- {alias[m]}: {when}, {domain.lower()}, {level.lower()}")

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
    ordered = sorted(
        daily.items(), key=lambda kv: (kv[0][0], first_time[kv[0]], kv[0][1], kv[0][2])
    )
    if render == "v3":
        return "\n".join(lines + _history_v3(ordered, first_time, start, lookback))
    lines += ["", "History (date, first time, account, event, count, credits; "
              f"lines before {start:%Y-%m-%d} are earlier context):"]
    for (day, a, kind), (n, total) in ordered:
        extra = f" x{n}" if n > 1 else ""
        extra += f" {total} credits" if total else ""
        lines.append(f"{day} {first_time[(day, a, kind)]} {a} {kind}{extra}")
    return "\n".join(lines)


def _history_v3(ordered: list[tuple[tuple[str, str, str], list[int]]],
                first_time: dict[tuple[str, str, str], str], start: datetime,
                lookback: int | None) -> list[str]:
    """v3 History: v2's rows grouped by day; older pre-cycle days optionally rolled up by week."""
    cut = "" if lookback is None else f"{start.date() - timedelta(days=lookback)}"
    weekly: dict[tuple[str, str, str], list[int]] = {}
    days: dict[str, list[str]] = {}
    for (day, a, kind), (n, total) in ordered:
        name = V3_EVENTS.get(kind, kind)
        if day < cut:
            iso = datetime.strptime(day, "%Y-%m-%d").isocalendar()
            k = (f"{iso.year}-W{iso.week:02d}", a, name)
            c, t = weekly.get(k, [0, 0])
            weekly[k] = [c + n, t + total]
            continue
        extra = f" x{n}" if n > 1 else ""
        extra += f" {total}cr" if total else ""
        days.setdefault(day, []).append(f"{a} {name} {first_time[(day, a, kind)]}{extra}")
    out = ["", "History, one line per day (date: account event first-time, then xN when it "
           "happened N times that day and Ncr for N credits; the year is shown when it "
           f"changes; days before {start:%Y-%m-%d} are earlier context):"]
    if weekly:
        out.append(f"Days before {cut} are summarised per ISO week ({len({k[0] for k in weekly})} "
                   "weeks): account event xcount, credits.")
        week_lines: dict[str, list[str]] = {}
        for (week, a, name), (n, total) in sorted(
            weekly.items(), key=lambda kv: (kv[0][0], int(kv[0][1][1:]), kv[0][2])
        ):
            week_lines.setdefault(week, []).append(
                f"{a} {name} x{n}" + (f" {total}cr" if total else ""))
        out += [f"week {week}: " + "; ".join(items) for week, items in week_lines.items()]
    year = ""
    for day, items in days.items():
        label = day if day[:4] != year else day[5:]
        year = day[:4]
        out.append(f"{label}: " + "; ".join(items))
    return out


# ------------------------------------------------------------------------------ run


def _ask(provider: CachingProvider, conn: sqlite3.Connection, key: str,
         payload: dict[str, Any], system_prompt: str, max_tokens: int) -> dict[str, Any]:
    """One model call for one unit; the row records the level or the error."""
    evidence = render_unit(conn, payload)
    request = LLMRequest(
        system=system_prompt,
        segments=(StructuredEvidenceSegment(evidence),),
        temperature=TEMPERATURE, max_tokens=max_tokens, timeout_s=600.0,
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
    return row


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


def run(config_path: Path, out: Path, limit: int | None, max_tokens: int,
        prompt_version: str = PROMPT_VERSION, keys_path: Path | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    config = load_provider_config(config_path)
    system_prompt = PROMPTS[prompt_version]
    header = {
        "decision": "P6 pilot (open scope)",
        "prompt_version": prompt_version,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "runner_sha256": _sha256(Path(__file__)),
        "config_sha256": _sha256(config_path),
        "provider": describe_provider(config),
        "detected_store_sha256": _sha256(DETECTED),
        "threshold": SCORE_THRESHOLD,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
    }
    if keys_path is not None:
        header["keys_sha256"] = _sha256(keys_path)
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
    if keys_path is not None:
        wanted = set(json.loads(keys_path.read_text())["keys"])
        missing = wanted - {k for k, _ in units}
        if missing:
            raise SystemExit(f"{len(missing)} keys are not open allowed units")
        units = [u for u in units if u[0] in wanted]
    todo = [u for u in units if u[0] not in done][: limit if limit else None]
    print(f"open allowed units {len(units)}, done {len(done)}, this run {len(todo)}", flush=True)
    with _ro(WORLD / "inference.db") as conn, outcomes_path.open("a") as sink:
        for i, (key, payload) in enumerate(todo, 1):
            row = _ask(provider, conn, key, payload, system_prompt, max_tokens)
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


# ------------------------------------------------------------ sample / compare


def _spec(spec: str) -> tuple[Path, str | None]:
    """``DIR`` (a ``run`` output) or ``DIR::ARM`` (one arm of an ``ablate`` output)."""
    path, _, arm = spec.partition("::")
    return Path(path), arm or None


def _levels(out: Path, arm: str | None = None) -> dict[str, int | None]:
    rows = [json.loads(line) for line in (out / "outcomes.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("arm") == arm]
    levels = {r["key"]: r["level"] for r in rows}
    if len(levels) != len(rows):
        raise SystemExit(f"{out} arm {arm}: duplicate keys")
    return levels


def _checked(spec: str, expected: set[str],
             subset: bool = False) -> tuple[dict[str, int], dict[str, Any]]:
    """Levels of one run or arm, refusing anything but one valid level per expected key.

    ``subset``: the expected keys come from an explicit manifest, and a run covering more
    units is cut down to them.
    """
    out, arm = _spec(spec)
    levels = _levels(out, arm)
    if subset:
        levels = {k: v for k, v in levels.items() if k in expected}
    if set(levels) != expected:
        raise SystemExit(f"{spec}: {len(expected - set(levels))} expected keys missing, "
                         f"{len(set(levels) - expected)} unexpected")
    valid = {k: v for k, v in levels.items() if v is not None}
    if len(valid) != len(levels):
        raise SystemExit(f"{spec}: {len(levels) - len(valid)} keys without a valid level")
    return valid, json.loads((out / "run.json").read_text())


# Run settings that must agree between compared runs (prompt and runner may differ).
FIXED = ("config_sha256", "provider", "detected_store_sha256", "threshold",
         "temperature", "max_tokens")


def _expected_keys(keys_path: Path | None) -> set[str]:
    if keys_path is not None:
        return set(json.loads(keys_path.read_text())["keys"])
    return {k for k, _ in _open_units(DETECTED)}


def sample(reference: Path, n: int, seed: int, keys_path: Path) -> dict[str, Any]:
    """Seeded subset, half rated >= FLAG_LEVEL by the reference run, half below. Label-free."""
    import random

    levels = {k: v for k, v in _levels(reference).items() if v is not None}
    high = sorted(k for k, v in levels.items() if v >= FLAG_LEVEL)
    low = sorted(k for k, v in levels.items() if v < FLAG_LEVEL)
    rng = random.Random(seed)
    keys = sorted(rng.sample(high, n // 2) + rng.sample(low, n - n // 2))
    doc = {"reference": str(reference.resolve().relative_to(ROOT)), "n": n, "seed": seed,
           "strata": f"reference level >= {FLAG_LEVEL}: {n // 2}; below: {n - n // 2}",
           "keys": keys}
    keys_path.write_text(json.dumps(doc, indent=2) + "\n")
    return {k: v for k, v in doc.items() if k != "keys"}


def compare(a: str, b: str, keys_path: Path | None) -> dict[str, Any]:
    """Pair two runs (or arms) by unit key; how far the level moved. Label-free.

    Stops unless both carry exactly one valid level for every expected key (the keys
    manifest, or all open allowed units) and agree on the fixed run settings. A setting
    recorded by only one side (P6.1 predates ``temperature``/``max_tokens``) is reported.
    """
    expected = _expected_keys(keys_path)
    la, ha = _checked(a, expected, subset=keys_path is not None)
    lb, hb = _checked(b, expected, subset=keys_path is not None)
    unrecorded = [f for f in FIXED if (f in ha) != (f in hb)]
    differ = [f for f in FIXED if f in ha and f in hb and ha[f] != hb[f]]
    if differ:
        raise SystemExit(f"fixed settings differ: {differ}")
    keys = sorted(expected)
    diffs = [lb[k] - la[k] for k in keys]
    flips = [k for k in keys if (la[k] >= FLAG_LEVEL) != (lb[k] >= FLAG_LEVEL)]
    return {
        "a": a, "b": b, "paired": len(keys), "settings_recorded_by_one_side_only": unrecorded,
        "identical": sum(d == 0 for d in diffs),
        "abs_diff_histogram": dict(sorted(Counter(abs(d) for d in diffs).items())),
        "mean_diff_b_minus_a": round(sum(diffs) / len(diffs), 3),
        f"flag_flips_at_ge_{FLAG_LEVEL}": len(flips),
        "flag_flip_rate": round(len(flips) / len(keys), 4),
    }


# --------------------------------------------------------------------------- ablate


def ablate(config_path: Path, out: Path, arms: list[str], seed: int, max_tokens: int) -> None:
    """Every open allowed unit under every arm, the arm order shuffled per unit (seeded).

    One ``outcomes.jsonl`` with an ``arm`` field. Resumable: on restart, rows without a
    valid level move to ``outcomes-failed-<n>.jsonl`` and are asked again, so the final
    file has exactly one valid row per (key, arm). Label-free.
    """
    import random

    out.mkdir(parents=True, exist_ok=True)
    config = load_provider_config(config_path)
    header = {
        "decision": "P6.1b routing-cue ablation (open scope)",
        "arms": {a: hashlib.sha256(PROMPTS[a].encode()).hexdigest() for a in arms},
        "arm_order_seed": seed,
        "runner_sha256": _sha256(Path(__file__)),
        "config_sha256": _sha256(config_path),
        "provider": describe_provider(config),
        "detected_store_sha256": _sha256(DETECTED),
        "threshold": SCORE_THRESHOLD,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
    }
    header_path = out / "run.json"
    if header_path.exists() and json.loads(header_path.read_text()) != header:
        raise SystemExit(f"{header_path} differs from this run; use a new --out")
    header_path.write_text(json.dumps(header, indent=2, sort_keys=True) + "\n")

    outcomes_path = out / "outcomes.jsonl"
    done: set[tuple[str, str]] = set()
    if outcomes_path.exists():
        rows = [json.loads(line) for line in outcomes_path.read_text().splitlines()]
        valid = [r for r in rows if r["level"] is not None]
        failed = [r for r in rows if r["level"] is None]
        if failed:
            n = len(list(out.glob("outcomes-failed-*.jsonl"))) + 1
            (out / f"outcomes-failed-{n}.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in failed))
            outcomes_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in valid))
        done = {(r["key"], r["arm"]) for r in valid}
    cache = ResponseCache(path=out / "transcript.jsonl")
    cache.load()
    provider = CachingProvider(inner=build_provider(config), cache=cache)

    units = _open_units(DETECTED)
    calls = []
    for key, payload in units:
        order = list(arms)
        random.Random(f"{seed}|{key}").shuffle(order)
        calls += [(key, payload, arm, pos) for pos, arm in enumerate(order)
                  if (key, arm) not in done]
    print(f"open allowed units {len(units)}, arms {arms}, done {len(done)}, "
          f"this run {len(calls)}", flush=True)
    with _ro(WORLD / "inference.db") as conn, outcomes_path.open("a") as sink:
        for i, (key, payload, arm, pos) in enumerate(calls, 1):
            row = _ask(provider, conn, key, payload, PROMPTS[arm], max_tokens)
            row.update(arm=arm, position=pos)
            sink.write(json.dumps(row, sort_keys=True) + "\n")
            sink.flush()
            cache.save()
            print(f"[{i}/{len(calls)}] {arm} level={row['level']} "
                  f"tokens={row.get('prompt_tokens')} {row['wall_s']}s {row.get('error', '')}",
                  flush=True)
    rows = [json.loads(line) for line in outcomes_path.read_text().splitlines()]
    failures = sum(r["level"] is None for r in rows)
    print(f"failures in outcomes: {failures}", flush=True)
    if failures:
        raise SystemExit(1)


def score_ablation(a: str, b: str, boot: int, seed: int) -> dict[str, Any]:
    """Paired, label-aware readout of arm B against arm A (P6.1b §3). Harness only.

    Ambiguous units are excluded, as in P6.1. The interval for the change in the
    legitimate flag share resamples legitimate source groups (units of one group are not
    independent); it is reported only with at least MIN_GROUPS groups. The abusive side
    (4 source groups) is descriptive.
    """
    import random

    expected = _expected_keys(None)
    la, ha = _checked(a, expected)
    lb, hb = _checked(b, expected)
    differ = [f for f in FIXED if ha.get(f) != hb.get(f)]
    if differ:
        raise SystemExit(f"fixed settings differ: {differ}")
    gt_url = f"sqlite:///{(WORLD / 'ground_truth.db').resolve()}"
    inf_url = f"sqlite:///{(WORLD / 'inference.db').resolve()}"
    with (
        Session(make_inference_engine(inf_url), future=True) as inf,
        Session(make_ground_truth_engine(gt_url), future=True) as gt,
    ):
        view = load_ground_truth(gt, inf_session=inf, splits=OPEN_SPLITS)
    unit: dict[str, tuple[str, str, str, float]] = {}  # key -> group, label, family, score
    with _ro(DETECTED) as conn:
        for (raw,) in conn.execute("select payload_json from detection_evidence"):
            payload = json.loads(raw)
            if _key(payload) in expected:
                g = view.split_groups[payload["member_account_ids"][0]]
                unit[_key(payload)] = (g, view.labels[g].value, view.families[g],
                                       payload["risk_score"])
    flag_a = {k: la[k] >= FLAG_LEVEL for k in expected}
    flag_b = {k: lb[k] >= FLAG_LEVEL for k in expected}
    by_label: dict[str, list[str]] = defaultdict(list)
    for k in sorted(expected):
        by_label[unit[k][1]].append(k)
    pos, neg = by_label["abusive"], by_label["legitimate"]
    labelled = len(pos) + len(neg)

    def arm_stats(levels: dict[str, int], flag: dict[str, bool]) -> dict[str, Any]:
        flagged = sum(flag[k] for k in pos + neg)
        return {
            "legitimate_ge8": sum(flag[k] for k in neg),
            "legitimate_ge8_share": round(sum(flag[k] for k in neg) / len(neg), 4),
            "abusive_ge8": sum(flag[k] for k in pos),
            "abusive_ge8_share": round(sum(flag[k] for k in pos) / len(pos), 4),
            "auc": _auc([levels[k] for k in pos], [levels[k] for k in neg]),
            "random_pick_expected_abusive_at_same_volume":
                round(len(pos) * flagged / labelled, 1),
            "level_histogram": {
                lab: dict(sorted(Counter(levels[k] for k in ks).items()))
                for lab, ks in (("abusive", pos), ("legitimate", neg))
            },
        }

    sa, sb = arm_stats(la, flag_a), arm_stats(lb, flag_b)
    d_leg = sb["legitimate_ge8_share"] - sa["legitimate_ge8_share"]
    d_abu = sb["abusive_ge8_share"] - sa["abusive_ge8_share"]
    d_auc = (sb["auc"] or 0.0) - (sa["auc"] or 0.0)

    groups: dict[str, list[str]] = defaultdict(list)
    for k in neg:
        groups[unit[k][0]].append(k)
    gids = sorted(groups)
    interval = None
    if len(gids) >= MIN_GROUPS:
        rng = random.Random(seed)
        stats = []
        for _ in range(boot):
            ks = [k for g in rng.choices(gids, k=len(gids)) for k in groups[g]]
            stats.append(sum(flag_b[k] - flag_a[k] for k in ks) / len(ks))
        stats.sort()
        interval = [round(stats[int(0.025 * boot)], 4), round(stats[int(0.975 * boot) - 1], 4)]

    def changes(keys: list[str]) -> dict[str, Any]:
        return {
            "units": len(keys),
            "mean_level_change": round(sum(lb[k] - la[k] for k in keys) / len(keys), 3)
            if keys else None,
            "flag_on": sum(flag_b[k] and not flag_a[k] for k in keys),
            "flag_off": sum(flag_a[k] and not flag_b[k] for k in keys),
        }

    abusive_groups: dict[str, list[str]] = defaultdict(list)
    families: dict[str, list[str]] = defaultdict(list)
    for k in pos:
        abusive_groups[unit[k][0]].append(k)
        families[unit[k][2]].append(k)
    bands: dict[str, list[str]] = defaultdict(list)
    for k in neg:
        s = unit[k][3]
        bands["=0" if s == 0 else "(0,0.4)" if s < 0.4 else "[0.4,0.6)"].append(k)

    if interval and interval[1] < 0 and d_leg <= -MIN_EFFECT:
        legit = "reduction established"
    elif interval and interval[0] > 0 and d_leg >= MIN_EFFECT:
        legit = "increase established"
    else:
        legit = "no established change"
    ranking = ("improved" if d_auc >= AUC_STEP else "degraded" if d_auc <= -AUC_STEP
               else "unchanged")
    return {
        "scope": "open", "a": a, "b": b, "excluded_ambiguous": len(by_label["ambiguous"]),
        "arm_a": sa, "arm_b": sb,
        "delta_legitimate_ge8_share": round(d_leg, 4),
        "delta_legitimate_interval_95_group_bootstrap": interval,
        "legitimate_source_groups": len(gids), "bootstrap": {"resamples": boot, "seed": seed},
        "delta_abusive_ge8_share": round(d_abu, 4),
        "delta_auc": round(d_auc, 4),
        "paired_changes": {"legitimate": changes(neg), "abusive": changes(pos)},
        "legitimate_by_detector_band": {b_: changes(ks) for b_, ks in sorted(bands.items())},
        "abusive_by_family": {f: changes(ks) for f, ks in sorted(families.items())},
        "abusive_by_source_group": {g: changes(ks) for g, ks in sorted(abusive_groups.items())},
        "legitimate_groups_with_any_flag_change": sum(
            any(flag_a[k] != flag_b[k] for k in ks) for ks in groups.values()),
        "readout": {
            "legitimate_flag_share": legit,
            "ranking": ranking,
            "abusive_flag_share_moved_10pp_or_more": abs(d_abu) >= MIN_EFFECT,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--max-tokens", type=int, default=16)
    r.add_argument("--prompt", choices=sorted(PROMPTS), default=PROMPT_VERSION)
    r.add_argument("--keys", type=Path, default=None)
    s = sub.add_parser("score")
    s.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("render", help="print one unit's prompt evidence (label-free)")
    p.add_argument("--index", type=int, default=0)
    m = sub.add_parser("sample", help="seeded label-free subset of keys")
    m.add_argument("--reference", type=Path, required=True)
    m.add_argument("--n", type=int, default=50)
    m.add_argument("--seed", type=int, required=True)
    m.add_argument("--keys", type=Path, required=True)
    c = sub.add_parser("compare", help="pair two runs by key (label-free)")
    c.add_argument("--a", required=True, help="DIR or DIR::ARM")
    c.add_argument("--b", required=True, help="DIR or DIR::ARM")
    c.add_argument("--keys", type=Path, default=None, help="expected keys (default: all)")
    ab = sub.add_parser("ablate", help="every unit under every arm, interleaved (label-free)")
    ab.add_argument("--config", type=Path, required=True)
    ab.add_argument("--out", type=Path, required=True)
    ab.add_argument("--arms", default="recall-v1,recall-v1-nocue")
    ab.add_argument("--seed", type=int, required=True)
    ab.add_argument("--max-tokens", type=int, default=16)
    sa = sub.add_parser("score-ablation", help="paired label-aware readout (harness)")
    sa.add_argument("--a", required=True, help="DIR or DIR::ARM")
    sa.add_argument("--b", required=True, help="DIR or DIR::ARM")
    sa.add_argument("--boot", type=int, default=10000)
    sa.add_argument("--seed", type=int, default=20261013)
    sa.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if args.cmd == "run":
        run(args.config, args.out, args.limit, args.max_tokens, args.prompt, args.keys)
    elif args.cmd == "score":
        print(json.dumps(score(args.out), indent=2, sort_keys=True))
    elif args.cmd == "sample":
        print(json.dumps(sample(args.reference, args.n, args.seed, args.keys), indent=2))
    elif args.cmd == "compare":
        print(json.dumps(compare(args.a, args.b, args.keys), indent=2, sort_keys=True))
    elif args.cmd == "ablate":
        arms = args.arms.split(",")
        if len(set(arms)) != len(arms) or not set(arms) <= set(PROMPTS):
            raise SystemExit(f"--arms must be distinct names from {sorted(PROMPTS)}")
        ablate(args.config, args.out, arms, args.seed, args.max_tokens)
    elif args.cmd == "score-ablation":
        result = score_ablation(args.a, args.b, args.boot, args.seed)
        text = json.dumps(result, indent=2, sort_keys=True)
        if args.report:
            args.report.write_text(text + "\n")
        print(text)
    else:
        key, payload = _open_units(DETECTED)[args.index]
        with _ro(WORLD / "inference.db") as conn:
            text = render_unit(conn, payload)
        print(f"# {key}  ({len(text)} chars)\n{text}")


if __name__ == "__main__":
    main()

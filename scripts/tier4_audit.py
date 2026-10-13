"""Pre-registration data audit of the Tier 4 world (decision P5-6).

Evaluation-side, not detector code: it reads the sealed labels from the ingested
parquets to ask whether the *world* has shortcuts, before any detector or reviewer
is scored on it. Checks:

* order coupling — raw JSON row order and id order against the label;
* duplicate clusters once identifiers are removed;
* single-signal separability of every brief-listed cluster observable (plus a few
  rhythm signals), and the best two-signal threshold rules;
* appeal text — whether single sentences, word bigrams or length give away label or
  kind: a leave-one-out sentence lookup and a leave-one-out bigram naive Bayes (pure
  Python), with a label-permutation null because N is small.

Nothing here is tuned on or fed back into the detector. Writes ``audit.json``.

Usage: python scripts/tier4_audit.py [--dir data/tier4] [--out path/audit.json]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

import polars as pl

WORLD_END = datetime(2026, 6, 1, tzinfo=UTC)
SIGNAL_TYPES = (
    "stable_device", "payment_token", "recovery_contact", "network_group",
    "behavioral_template",
)


def auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """P(positive > negative) with ties at 1/2 (Mann-Whitney)."""
    ranked = sorted([(v, 1) for v in positive] + [(v, 0) for v in negative])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        mean_rank = (i + j + 1) / 2
        rank_sum += mean_rank * sum(flag for _, flag in ranked[i:j])
        i = j
    n_pos, n_neg = len(positive), len(negative)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _balanced_accuracy(pred: list[bool], truth: list[bool]) -> float:
    tp = sum(p and t for p, t in zip(pred, truth, strict=True))
    tn = sum(not p and not t for p, t in zip(pred, truth, strict=True))
    pos = sum(truth)
    return 0.5 * (tp / pos + tn / (len(truth) - pos))


def _cuts(values: list[float]) -> list[float]:
    distinct = sorted(set(values))
    return [(a + b) / 2 for a, b in zip(distinct, distinct[1:], strict=False)]


def best_threshold(values: list[float], truth: list[bool]) -> tuple[float, float, str]:
    best = (0.5, math.nan, ">")
    for cut in _cuts(values):
        for side in (">", "<"):
            pred = [v > cut if side == ">" else v < cut for v in values]
            score = _balanced_accuracy(pred, truth)
            if score > best[0]:
                best = (score, cut, side)
    return best


def _hours(delta_seconds: float) -> float:
    return delta_seconds / 3600


def cluster_features(d: Path) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    accounts = pl.read_parquet(d / "accounts.parquet")
    events = pl.read_parquet(d / "account_events.parquet")
    linkage = pl.read_parquet(d / "linkage_observations.parquet")
    cycles = pl.read_parquet(d / "entitlement_cycles.parquet")
    membership = pl.read_parquet(d / "cluster_membership.parquet")
    appeals = pl.read_parquet(d / "appeals.parquet")
    labels = dict(pl.read_parquet(d / "ground_truth.parquet").select("cluster_id", "label").rows())

    cluster_of = dict(membership.rows())
    members: dict[str, list[str]] = defaultdict(list)
    for account, cluster in cluster_of.items():
        members[cluster].append(account)
    account_rows = {r["account_id"]: r for r in accounts.iter_rows(named=True)}
    by_account: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in events.sort("timestamp").iter_rows(named=True):
        by_account[r["account_id"]].append(r)
    values_by_cluster: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for r in linkage.unique(["account_id", "signal_type", "signal_value"]).iter_rows(named=True):
        values_by_cluster[cluster_of[r["account_id"]]][r["signal_type"]][r["signal_value"]] += 1
    cycle_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in cycles.iter_rows(named=True):
        cycle_rows[cluster_of[r["account_id"]]].append(r)
    appeal_len = {
        cluster_of[r["account_id"]]: len(r["text"]) for r in appeals.iter_rows(named=True)
    }

    features: dict[str, dict[str, float]] = {}
    for cluster, accts in members.items():
        n = len(accts)
        f: dict[str, float] = {"cluster_size": n}
        for signal in SIGNAL_TYPES:
            counts = values_by_cluster[cluster][signal]
            f[f"shared_{signal}_values"] = sum(c >= 2 for c in counts.values())
            f[f"top_share_{signal}"] = max(counts.values(), default=0) / n
        created = [account_rows[a]["created_at"] for a in accts]
        f["signup_spread_days"] = (max(created) - min(created)).total_seconds() / 86400
        f["median_account_age_days"] = median(
            (WORLD_END - c).total_seconds() / 86400 for c in created
        )
        for field, options in (
            ("email_domain_type", ("consumer", "disposable", "education", "enterprise")),
            ("verification_level", ("none", "email", "phone", "enhanced")),
        ):
            for option in options:
                f[f"frac_{field}_{option}"] = (
                    sum(account_rows[a][field] == option for a in accts) / n
                )
        kinds = Counter(e["event_type"] for a in accts for e in by_account[a])
        for kind in ("verification", "verification_failed", "login"):
            f[f"{kind}_per_account"] = kinds[kind] / n
        first_claim = []
        for a in accts:
            claims = [e for e in by_account[a] if e["event_type"] == "claim_credits"]
            if claims:
                first_claim.append(
                    _hours((claims[0]["timestamp"] - account_rows[a]["created_at"]).total_seconds())
                )
        f["median_hours_create_to_first_claim"] = median(first_claim) if first_claim else -1
        rows = cycle_rows[cluster]
        granted = sum(r["credits_granted"] for r in rows)
        f["claimed_per_account_cycle"] = granted / max(len(rows), 1)
        f["consumed_per_account_cycle"] = sum(r["credits_consumed"] for r in rows) / max(
            len(rows), 1
        )
        f["consumed_over_granted"] = (
            sum(r["credits_consumed"] for r in rows) / granted if granted else 0
        )
        consumes = [e for a in accts for e in by_account[a] if e["event_type"] == "consume_credits"]
        f["high_tier_consume_frac"] = (
            sum(e["endpoint_cost_tier"] == "high" for e in consumes) / len(consumes)
            if consumes else 0
        )
        activity = [
            e for a in accts for e in by_account[a]
            if e["event_type"] in {"login", "consume_credits"}
        ]
        hours = Counter(e["timestamp"].hour for e in activity)
        total = sum(hours.values()) or 1
        f["activity_hour_entropy"] = -sum(
            (c / total) * math.log2(c / total) for c in hours.values()
        )
        f["weekend_activity_frac"] = sum(e["timestamp"].weekday() >= 5 for e in activity) / total
        # Rhythm overlap: share of member-hours in which ≥2 members were active.
        slots: dict[datetime, set[str]] = defaultdict(set)
        for e in activity:
            slots[e["timestamp"].replace(minute=0, second=0, microsecond=0)].add(e["account_id"])
        f["concurrent_hour_frac"] = (
            sum(len(s) >= 2 for s in slots.values()) / len(slots) if slots else 0
        )
        f["appeal_present"] = float(cluster in appeal_len)
        f["appeal_length"] = float(appeal_len.get(cluster, 0))
        features[cluster] = f
    return features, labels


def separability(features: dict[str, dict[str, float]], labels: dict[str, str]) -> dict[str, Any]:
    clusters = [c for c in features if labels[c] in {"abusive", "legitimate"}]
    truth = [labels[c] == "abusive" for c in clusters]
    names = sorted(next(iter(features.values())))
    single: dict[str, dict[str, Any]] = {}
    balanced_by_name: dict[str, float] = {}
    for name in names:
        values = [features[c][name] for c in clusters]
        pos = [v for v, t in zip(values, truth, strict=True) if t]
        neg = [v for v, t in zip(values, truth, strict=True) if not t]
        score = auc(pos, neg)
        balanced, cut, side = best_threshold(values, truth)
        balanced_by_name[name] = balanced
        single[name] = {
            "auc_abusive_high": round(score, 3),
            "best_threshold_balanced_accuracy": round(balanced, 3),
            "threshold": None if math.isnan(cut) else round(cut, 4),
            "side_for_abusive": side,
            "perfect_separation": min(pos) > max(neg) or max(pos) < min(neg),
            "abusive_range": [round(min(pos), 3), round(max(pos), 3)],
            "legitimate_range": [round(min(neg), 3), round(max(neg), 3)],
        }
    # Two-signal AND rules over the eight strongest single signals.
    top = sorted(names, key=lambda n: -balanced_by_name[n])[:8]
    pairs: list[tuple[float, str]] = []
    for a, b in combinations(top, 2):
        va = [features[c][a] for c in clusters]
        vb = [features[c][b] for c in clusters]
        best = (0.0, "")
        for cut_a in _cuts(va):
            for side_a in (">", "<"):
                pa = [v > cut_a if side_a == ">" else v < cut_a for v in va]
                for cut_b in _cuts(vb):
                    for side_b in (">", "<"):
                        pred = [
                            x and (v > cut_b if side_b == ">" else v < cut_b)
                            for x, v in zip(pa, vb, strict=True)
                        ]
                        score = _balanced_accuracy(pred, truth)
                        if score > best[0]:
                            best = (
                                score,
                                f"{a} {side_a} {cut_a:.4g} AND {b} {side_b} {cut_b:.4g}",
                            )
        pairs.append(best)
    pairs.sort(key=lambda p: -p[0])
    return {
        "n_abusive": sum(truth),
        "n_legitimate": len(truth) - sum(truth),
        "single_signal": single,
        "perfect_single_separators": [n for n in names if single[n]["perfect_separation"]],
        "top_two_signal_rules": [
            {"rule": rule, "balanced_accuracy": round(score, 3)} for score, rule in pairs[:5]
        ],
    }


def order_coupling(d: Path, labels: dict[str, str]) -> dict[str, Any]:
    source = json.loads((d / "gpt-world.json").read_text())
    cluster_of = {r["account_id"]: r["cluster_id"] for r in source["cluster_membership"]}
    result = {}
    for table, key in (
        ("accounts", "account_id"), ("cluster_membership", "account_id"),
        ("ground_truth", "cluster_id"), ("entitlement_cycles", "account_id"),
    ):
        rows = source[table]
        label_of = [
            labels[r[key]] if key == "cluster_id" else labels[cluster_of[r[key]]] for r in rows
        ]
        id_field = {"accounts": "account_id", "cluster_membership": "account_id",
                    "ground_truth": "cluster_id", "entitlement_cycles": "cycle_id"}[table]
        id_rank = {v: i for i, v in enumerate(sorted(r[id_field] for r in rows))}
        for kind, positions in (
            ("row_order", list(range(len(rows)))),
            ("id_order", [id_rank[r[id_field]] for r in rows]),
        ):
            pos = [p for p, lab in zip(positions, label_of, strict=True) if lab == "abusive"]
            neg = [p for p, lab in zip(positions, label_of, strict=True) if lab == "legitimate"]
            result[f"{table}.{kind}_auc"] = round(auc(pos, neg), 3)
    return result


def duplicate_clusters(d: Path) -> dict[str, Any]:
    """Identifier-free fingerprint: per member, creation offset from the cluster's
    first signup, account enums, event-type counts and credit totals."""
    accounts = {r["account_id"]: r for r in pl.read_parquet(d / "accounts.parquet").iter_rows(
        named=True)}
    counts = (
        pl.read_parquet(d / "account_events.parquet")
        .group_by("account_id", "event_type")
        .agg(pl.len(), pl.col("credits_amount").sum())
    )
    per_account: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    for r in counts.iter_rows(named=True):
        per_account[r["account_id"]].append((r["event_type"], r["len"], r["credits_amount"]))
    members: dict[str, list[str]] = defaultdict(list)
    for account, cluster in pl.read_parquet(d / "cluster_membership.parquet").rows():
        members[cluster].append(account)
    fingerprints: Counter[tuple[Any, ...]] = Counter()
    for accts in members.values():
        start = min(accounts[a]["created_at"] for a in accts)
        fingerprints[tuple(sorted(
            (
                (accounts[a]["created_at"] - start).total_seconds(),
                accounts[a]["email_domain_type"],
                accounts[a]["verification_level"],
                tuple(sorted(per_account[a])),
            )
            for a in accts
        ))] += 1
    return {
        "clusters": len(members),
        "distinct_fingerprints": len(fingerprints),
        "duplicated_fingerprints": sum(c > 1 for c in fingerprints.values()),
    }


_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[t ][0-9:.]+z?)?\b")
_WORD = re.compile(r"<date>|<num>|[a-z]+(?:'[a-z]+)?")
TEXT_TASKS = {
    # (target, positive class, classes kept) — manipulation / ambiguous excluded.
    "kind_honest_vs_dishonest": ("kind", "dishonest", {"honest", "dishonest"}),
    "label_abusive_vs_legitimate": ("label", "abusive", {"abusive", "legitimate"}),
}
BIGRAM_PASS_BAR = 0.65
PERMUTATIONS = 200


def _sentences(text: str, *, normalise: bool) -> set[str]:
    parts = {s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()}
    if not normalise:
        return parts
    return {" ".join(_DATE.sub("<date>", s.lower()).split()) for s in parts}


def _words(text: str) -> list[str]:
    return _WORD.findall(re.sub(r"\d+", "<num>", _DATE.sub("<date>", text.lower())))


def _bigrams(text: str) -> Counter[str]:
    words = _words(text)
    return Counter(f"{a} {b}" for a, b in zip(words, words[1:], strict=False))


def sentence_lookup(sentences: list[set[str]], targets: list[str]) -> dict[str, Any]:
    """Leave-one-out: each sentence shared with other appeals votes their target mix.

    A sentence found in *k* other appeals adds each target's share among those *k*;
    the appeal predicts the top total. Appeals sharing no sentence abstain (and fall
    back to the training-fold majority for the fallback accuracy). This is the method
    behind the P5.6 v1 figures (kind 67/69, label 60/69 on raw sentences).
    """
    covered = correct = fallback_correct = 0
    for i, own in enumerate(sentences):
        votes: defaultdict[str, float] = defaultdict(float)
        for s in own:
            mix = Counter(t for j, t in enumerate(targets) if j != i and s in sentences[j])
            n = sum(mix.values())
            for t, c in mix.items():
                votes[t] += c / n
        if votes:
            covered += 1
            hit = max(sorted(votes), key=lambda t: votes[t]) == targets[i]
            correct += hit
            fallback_correct += hit
        else:
            rest = Counter(t for j, t in enumerate(targets) if j != i)
            fallback_correct += rest.most_common(1)[0][0] == targets[i]
    majority = Counter(targets).most_common(1)[0][1]
    return {
        "n": len(targets),
        "covered": covered,
        "correct_on_covered": correct,
        "accuracy_with_majority_fallback": round(fallback_correct / len(targets), 3),
        "majority_class_accuracy": round(majority / len(targets), 3),
    }


def _nb_loo(docs: list[Counter[str]], truth: list[bool], *, uniform_prior: bool) -> list[bool]:
    """Leave-one-out multinomial naive Bayes, alpha=1, training-fold vocabulary."""
    total = {True: Counter[str](), False: Counter[str]()}
    for doc, t in zip(docs, truth, strict=True):
        total[t].update(doc)
    n_class = Counter(truth)
    preds = []
    for doc, t in zip(docs, truth, strict=True):
        counts = {c: total[c].copy() for c in (True, False)}
        counts[t].subtract(doc)
        vocab = {w for c in (True, False) for w, v in counts[c].items() if v > 0}
        score = {}
        for c in (True, False):
            n_docs = n_class[c] - (c == t)
            prior = 0.0 if uniform_prior else math.log(n_docs / (len(docs) - 1))
            denom = sum(v for v in counts[c].values() if v > 0) + len(vocab)
            score[c] = prior + sum(
                k * math.log((max(counts[c][w], 0) + 1) / denom)
                for w, k in doc.items() if w in vocab
            )
        preds.append(score[True] > score[False])
    return preds


def bigram_naive_bayes(texts: list[str], truth: list[bool]) -> dict[str, Any]:
    """LOO bigram naive Bayes, with a label-permutation null for the small N."""
    import random

    docs = [_bigrams(t) for t in texts]
    out: dict[str, Any] = {"n_positive": sum(truth), "n_negative": len(truth) - sum(truth)}
    for name, uniform in (("uniform_prior", True), ("empirical_prior", False)):
        preds = _nb_loo(docs, truth, uniform_prior=uniform)
        out[name] = {
            "balanced_accuracy": round(_balanced_accuracy(preds, truth), 3),
            "accuracy": round(sum(p == t for p, t in zip(preds, truth, strict=True))
                              / len(truth), 3),
        }
    # Reported only (the pass bar is on bigrams): a unigram cross-check.
    unigram = _nb_loo([Counter(_words(t)) for t in texts], truth, uniform_prior=True)
    out["unigram_uniform_prior_balanced_accuracy"] = round(_balanced_accuracy(unigram, truth), 3)
    rng = random.Random(20260926)
    observed = out["uniform_prior"]["balanced_accuracy"]
    null = []
    for _ in range(PERMUTATIONS):
        shuffled = truth[:]
        rng.shuffle(shuffled)
        null.append(_balanced_accuracy(_nb_loo(docs, shuffled, uniform_prior=True), shuffled))
    null.sort()
    out["uniform_prior"]["permutation_null_p95"] = round(null[int(0.95 * len(null))], 3)
    out["uniform_prior"]["permutation_p_value"] = round(
        (1 + sum(v >= observed for v in null)) / (1 + len(null)), 3
    )
    pos, neg = Counter[str](), Counter[str]()
    for doc, t in zip(docs, truth, strict=True):
        (pos if t else neg).update(set(doc))
    n_pos, n_neg = sum(truth), len(truth) - sum(truth)
    lr = {
        w: math.log((pos[w] + 1) / (n_pos + 2)) - math.log((neg[w] + 1) / (n_neg + 2))
        for w in set(pos) | set(neg) if pos[w] + neg[w] >= 3
    }
    out["most_indicative_bigrams"] = [
        {"bigram": w, "log_doc_ratio_positive": round(v, 2), "docs_pos": pos[w], "docs_neg": neg[w]}
        for w, v in sorted(lr.items(), key=lambda kv: (-abs(kv[1]), kv[0]))[:10]
    ]
    out["passes_bar"] = all(
        out[p]["balanced_accuracy"] <= BIGRAM_PASS_BAR for p in ("uniform_prior", "empirical_prior")
    )
    return out


def appeal_text(d: Path, labels: dict[str, str]) -> dict[str, Any]:
    appeals = pl.read_parquet(d / "appeals.parquet")
    kinds = dict(pl.read_parquet(d / "appeal_truth.parquet").select("event_id", "kind").rows())
    cluster_of = dict(pl.read_parquet(d / "cluster_membership.parquet").rows())
    sentence_stats: dict[str, Counter[str]] = defaultdict(Counter)
    rows: list[tuple[str, str, set[str], int, str]] = []
    for r in appeals.iter_rows(named=True):
        label = labels[cluster_of[r["account_id"]]]
        kind = kinds[r["appeal_id"]]
        sentences = _sentences(r["text"], normalise=False)
        rows.append((label, kind, sentences, len(r["text"]), r["text"]))
        for s in sentences:
            sentence_stats[s][f"label:{label}"] += 1
            sentence_stats[s][f"kind:{kind}"] += 1
    giveaways = []
    for sentence, stats in sentence_stats.items():
        total = sum(v for k, v in stats.items() if k.startswith("label:"))
        for key, value in stats.items():
            if value == total and total >= 3:
                giveaways.append({"sentence": sentence, "only_in": key, "count": total})
    truth = [label == "abusive" for label, *_ in rows if label != "ambiguous"]
    lengths = [n for label, _, _, n, _ in rows if label != "ambiguous"]
    pos = [n for n, t in zip(lengths, truth, strict=True) if t]
    neg = [n for n, t in zip(lengths, truth, strict=True) if not t]
    lookup: dict[str, Any] = {}
    for normalise in (False, True):
        tag = "normalised" if normalise else "raw"
        sents = [_sentences(text, normalise=normalise) for *_, text in rows]
        lookup[f"{tag}_kind_3class"] = sentence_lookup(sents, [k for _, k, *_ in rows])
        lookup[f"{tag}_label_3class"] = sentence_lookup(sents, [lb for lb, *_ in rows])
        for task, (target, _positive, keep) in TEXT_TASKS.items():
            of = [label if target == "label" else kind for label, kind, *_ in rows]
            idx = [i for i, t in enumerate(of) if t in keep]
            lookup[f"{tag}_{task}"] = sentence_lookup([sents[i] for i in idx], [of[i] for i in idx])
    bigram = {}
    for task, (target, positive, keep) in TEXT_TASKS.items():
        of = [label if target == "label" else kind for label, kind, *_ in rows]
        sel = [i for i, t in enumerate(of) if t in keep]
        bigram[task] = bigram_naive_bayes(
            [rows[i][4] for i in sel], [of[i] == positive for i in sel]
        )
    kind_lookup = lookup["normalised_kind_honest_vs_dishonest"]
    hd = [r for r in rows if r[1] in {"honest", "dishonest"}]
    return {
        "appeals": len(rows),
        "distinct_texts": len({r["text"] for r in appeals.iter_rows(named=True)}),
        "distinct_sentences": len(sentence_stats),
        "length_auc_abusive_high": round(auc(pos, neg), 3) if pos and neg else None,
        "length_auc_dishonest_high": round(auc(
            [r[3] for r in hd if r[1] == "dishonest"], [r[3] for r in hd if r[1] == "honest"]
        ), 3),
        "kind_by_label": {
            f"{label}/{kind}": n
            for (label, kind), n in Counter((label, kind) for label, kind, *_ in rows).items()
        },
        "sentence_lookup": lookup,
        "bigram_naive_bayes": bigram,
        "text_pass": {
            "bigram_kind_balanced_accuracy_le_0_65":
                bigram["kind_honest_vs_dishonest"]["passes_bar"],
            "sentence_lookup_kind_not_above_majority":
                kind_lookup["accuracy_with_majority_fallback"]
                <= kind_lookup["majority_class_accuracy"],
        },
        "single_sentence_giveaways": sorted(
            giveaways, key=lambda g: (-g["count"], g["sentence"], g["only_in"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("data/tier4"))
    parser.add_argument("--out", type=Path, help="report path (default: <dir>/audit.json)")
    args = parser.parse_args()
    features, labels = cluster_features(args.dir)
    report = {
        "world": json.loads((args.dir / "manifest.json").read_text())["world_version"],
        "order_coupling": order_coupling(args.dir, labels),
        "duplicates": duplicate_clusters(args.dir),
        "separability": separability(features, labels),
        "appeals": appeal_text(args.dir, labels),
    }
    out = args.out or args.dir / "audit.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "order_coupling": report["order_coupling"],
        "duplicates": report["duplicates"],
        "perfect_single_separators": report["separability"]["perfect_single_separators"],
        "top_single": sorted(
            ((n, s["best_threshold_balanced_accuracy"], s["auc_abusive_high"])
             for n, s in report["separability"]["single_signal"].items()),
            key=lambda t: -t[1],
        )[:10],
        "top_two_signal_rules": report["separability"]["top_two_signal_rules"],
        "appeal_summary": {k: v for k, v in report["appeals"].items()
                           if k not in {"single_sentence_giveaways", "bigram_naive_bayes"}},
        "appeal_bigram_nb": {
            task: {k: v for k, v in r.items() if k != "most_indicative_bigrams"}
            for task, r in report["appeals"]["bigram_naive_bayes"].items()
        },
        "appeal_giveaways": report["appeals"]["single_sentence_giveaways"][:10],
    }, indent=1))


if __name__ == "__main__":
    main()

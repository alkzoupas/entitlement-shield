import json
import sqlite3
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SEVERITY = ["allow", "light_friction", "hold_incremental_credits", "temporary_hold"]


def sev(action: str) -> int:
    a = action.lower()
    return SEVERITY.index(a)


def load_batches(arm: str) -> dict[str, dict]:
    """fp -> {decision_id, evidence_keys}"""
    out = {}
    for i in range(7):
        p = REPO / f"data/synthetic/arm4-subagent/{arm}-batch{i:02d}.json"
        d = json.loads(p.read_text())
        for c in d["cases"]:
            out[c["fp"]] = {
                "decision_id": c["structured"].split("decision_id: ")[1].split("\n")[0],
                "evidence_keys": c["evidence_keys"],
            }
    return out


def load_transcript() -> dict[str, dict]:
    """key 'fp:pass' -> parsed verdict dict (or None on parse failure)"""
    out = {}
    p = REPO / "data/synthetic/arm4-subagent-haiku-transcript.jsonl"
    for line in p.read_text().splitlines():
        row = json.loads(line)
        key = row["key"]
        try:
            verdict = json.loads(row["response"]["text"])
        except Exception:
            verdict = None
        out[key] = verdict
    return out


def load_decisions(arm: str) -> dict[str, dict]:
    con = sqlite3.connect(REPO / f"data/synthetic/arm4-subagent-worlds/{arm}-inference.db")
    cur = con.cursor()
    cur.execute("select decision_id, account_id, proposed_action from decisions")
    return {r[0]: {"account_id": r[1], "proposed_action": r[2].lower()} for r in cur.fetchall()}


def load_labels(arm: str) -> dict[str, str]:
    con = sqlite3.connect(REPO / f"data/synthetic/arm4-subagent-worlds/{arm}-ground-truth.db")
    cur = con.cursor()
    cur.execute("select account_id, split_group_id from split_groups")
    acct_to_group = dict(cur.fetchall())
    cur.execute("select cluster_id, label from ground_truth")
    group_to_label = dict(cur.fetchall())
    return {
        acct: group_to_label.get(group, "UNKNOWN")
        for acct, group in acct_to_group.items()
    }


def derive_effect(verdict, evidence_keys, proposed):
    """Mirror reviewer_validation.evaluate_verdict's C-2/C-4/L-3 logic for one sample."""
    if verdict is None:
        return "escalate", False, 0
    if verdict.get("escalate_to_human"):
        return "escalate", False, 0
    cited = verdict.get("evidence_supporting_action", [])
    n_valid = len([c for c in cited if c in evidence_keys])
    if n_valid == 0:
        return "escalate", False, 0
    recommended = verdict.get("recommended_action", "").lower()
    if recommended not in SEVERITY:
        return "escalate", False, n_valid
    clamped_sev = min(sev(recommended), sev(proposed))
    strictly_less = clamped_sev < sev(proposed)
    if strictly_less and clamped_sev <= sev("light_friction"):
        return "reverse", True, n_valid
    return "uphold", False, n_valid


def analyze(arm: str):
    batches = load_batches(arm)
    transcript = load_transcript()
    decisions = load_decisions(arm)
    labels = load_labels(arm)

    # decision_id -> {"label", "proposed", "samples": [(pass, verdict, effect, valid, n_valid)]}
    per_decision: dict[str, dict[str, Any]] = {}
    strict_less_severe = {"legitimate": 0, "abusive": 0, "ambiguous": 0, "other": 0}
    equal = {"legitimate": 0, "abusive": 0, "ambiguous": 0, "other": 0}
    more_severe = {"legitimate": 0, "abusive": 0, "ambiguous": 0, "other": 0}
    parse_failed = 0
    total_lf_samples = 0
    strict_less_with_citation = 0

    for fp, meta in batches.items():
        dec_id = meta["decision_id"]
        dec = decisions.get(dec_id)
        if dec is None:
            continue
        proposed = dec["proposed_action"]
        if proposed != "light_friction":
            continue
        label = labels.get(dec["account_id"], "UNKNOWN")
        label_bucket = {
            "LEGITIMATE": "legitimate",
            "ABUSIVE": "abusive",
            "AMBIGUOUS": "ambiguous",
        }.get(label, "other")

        samples: list[tuple[int, dict[str, Any] | None, str, bool, int]] = []
        for i in range(3):
            key = f"{fp}:{i}"
            verdict = transcript.get(key)
            total_lf_samples += 1
            if verdict is None:
                parse_failed += 1
                samples.append((i, None, "escalate", False, 0))
                continue
            recommended = verdict.get("recommended_action", "").lower()
            if recommended not in SEVERITY:
                parse_failed += 1
                samples.append((i, verdict, "escalate", False, 0))
                continue
            s = sev(recommended)
            ps = sev(proposed)
            if s < ps:
                strict_less_severe[label_bucket] += 1
                raw_cites = verdict.get("evidence_supporting_action", [])
                if any(c in meta["evidence_keys"] for c in raw_cites):
                    strict_less_with_citation += 1
            elif s == ps:
                equal[label_bucket] += 1
            else:
                more_severe[label_bucket] += 1
            evidence_keys = meta["evidence_keys"]
            effect, applied_reverse, n_valid = derive_effect(verdict, evidence_keys, proposed)
            samples.append((i, verdict, effect, applied_reverse, n_valid))
        per_decision[dec_id] = {"label": label_bucket, "proposed": proposed, "samples": samples}

    n_decisions = len(per_decision)
    unanimous_reverse = 0
    unanimous_uphold = 0
    unanimous_escalate = 0
    split = 0
    for info in per_decision.values():
        effects = {s[2] for s in info["samples"]}
        if len(effects) == 1:
            eff = next(iter(effects))
            if eff == "reverse":
                unanimous_reverse += 1
            elif eff == "uphold":
                unanimous_uphold += 1
            else:
                unanimous_escalate += 1
        else:
            split += 1

    total_less = sum(strict_less_severe.values())
    total_equal = sum(equal.values())
    total_more = sum(more_severe.values())
    print(f"=== {arm} (light_friction tier) ===")
    print(f"decisions: {n_decisions}, samples: {total_lf_samples}, parse_failed: {parse_failed}")
    print(f"strict less_severe by label: {strict_less_severe}  total={total_less}")
    print(f"  of which carry >=1 valid citation: {strict_less_with_citation}")
    print(f"equal by label: {equal}  total={total_equal}")
    print(f"more_severe by label: {more_severe}  total={total_more}")
    print(
        f"per-decision unanimity: reverse={unanimous_reverse} "
        f"uphold={unanimous_uphold} escalate={unanimous_escalate} "
        f"split={split}"
    )
    print()
    return per_decision


if __name__ == "__main__":
    analyze("arm4_reviewer")
    analyze("arm4b_timeline_only")

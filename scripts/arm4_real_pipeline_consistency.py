"""Answer 'does temp-0 turn discrimination into consistency?' using data already
on disk, at zero additional cost: reanalyze the REAL local-MTPLX transcript
(data/synthetic/arm4-transcript.jsonl, P4.8) through the real derive_effect
semantics, joined to the same deterministic dev_arm4 fixture's decisions/labels
via data/synthetic/arm4-subagent-worlds/*.db (fingerprint-verified identical
fixture — see docs/p4.10-arm4-recovery-funnel-diagnosis.md).

Unlike the Haiku-subagent stand-in (independent fresh dispatches per sample,
high variance), this transcript came from the real reviewer.py sampling loop
(3 calls with one identical LLMRequest at REVIEWER_TEMPERATURE=0.0) — so it
directly measures whether real near-determinism converts discrimination into
per-decision self-consistency, without spending anything new.

Scratch script, not wired into any `make` target or production code.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SEVERITY = ["allow", "light_friction", "hold_incremental_credits", "temporary_hold"]


def sev(action: str) -> int:
    return SEVERITY.index(action.lower())


def load_batches(arm: str) -> dict[str, dict]:
    out = {}
    for i in range(7):
        path = REPO / f"data/synthetic/arm4-subagent/{arm}-batch{i:02d}.json"
        d = json.loads(path.read_text())
        for c in d["cases"]:
            out[c["fp"]] = {
                "decision_id": c["structured"].split("decision_id: ")[1].split("\n")[0],
                "evidence_keys": c["evidence_keys"],
            }
    return out


def load_decisions(arm: str) -> dict[str, dict]:
    db_path = REPO / f"data/synthetic/arm4-subagent-worlds/{arm}-inference.db"
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("select decision_id, account_id, proposed_action from decisions")
    return {r[0]: {"account_id": r[1], "proposed_action": r[2].lower()} for r in cur.fetchall()}


def load_labels(arm: str) -> dict[str, str]:
    gt_path = REPO / f"data/synthetic/arm4-subagent-worlds/{arm}-ground-truth.db"
    con = sqlite3.connect(gt_path)
    cur = con.cursor()
    cur.execute("select account_id, split_group_id from split_groups")
    acct_to_group = dict(cur.fetchall())
    cur.execute("select cluster_id, label from ground_truth")
    group_to_label = dict(cur.fetchall())
    return {acct: group_to_label.get(g, "UNKNOWN") for acct, g in acct_to_group.items()}


def load_transcript() -> dict[str, str]:
    out = {}
    with open(REPO / "data/synthetic/arm4-transcript.jsonl") as f:
        for line in f:
            row = json.loads(line)
            out[row["key"]] = row["response"]["text"]
    return out


def analyze(arm: str, batches: dict, decisions: dict, labels: dict, transcript: dict) -> None:
    strict_less = {"legitimate": 0, "abusive": 0}
    equal = {"legitimate": 0, "abusive": 0}
    more = {"legitimate": 0, "abusive": 0}
    n_by_label = {"legitimate": 0, "abusive": 0}
    per_decision: dict[str, tuple[str, list[str]]] = {}
    lf_decisions = 0
    label_map = {"LEGITIMATE": "legitimate", "ABUSIVE": "abusive"}

    for fp, meta in batches.items():
        dec = decisions.get(meta["decision_id"])
        if dec is None or dec["proposed_action"] != "light_friction":
            continue
        lf_decisions += 1
        label = label_map.get(labels.get(dec["account_id"], "UNKNOWN"))
        if label is None:
            continue
        proposed_sev = sev(dec["proposed_action"])
        effects: list[str] = []
        for i in range(3):
            text = transcript.get(f"{fp}:{i}")
            if text is None:
                continue
            try:
                v: dict[str, Any] = json.loads(text)
            except Exception:
                effects.append("escalate")
                continue
            n_by_label[label] += 1
            if v.get("escalate_to_human"):
                effects.append("escalate")
                continue
            rec = v.get("recommended_action", "").lower()
            if rec not in SEVERITY:
                effects.append("escalate")
                continue
            raw_cites = v.get("evidence_supporting_action", [])
            cited = [c for c in raw_cites if c in meta["evidence_keys"]]
            s = sev(rec)
            if s < proposed_sev:
                strict_less[label] += 1
                effects.append("reverse" if cited else "uphold")
            elif s == proposed_sev:
                equal[label] += 1
                effects.append("uphold")
            else:
                more[label] += 1
                effects.append("uphold")
        per_decision[fp] = (label, effects)

    complete = {fp: e for fp, e in per_decision.items() if len(e[1]) == 3}
    uni_reverse = sum(1 for _, e in complete.values() if len(set(e)) == 1 and e[0] == "reverse")
    uni_uphold = sum(1 for _, e in complete.values() if len(set(e)) == 1 and e[0] == "uphold")
    uni_escalate = sum(1 for _, e in complete.values() if len(set(e)) == 1 and e[0] == "escalate")
    split = sum(1 for _, e in complete.values() if len(set(e)) > 1)

    print(f"=== {arm} (real MTPLX transcript, light_friction tier) ===")
    print(f"light_friction decisions: {lf_decisions}, complete (3/3 samples): {len(complete)}")
    print(f"samples by label: {n_by_label}")
    print(f"strict less-severe: {strict_less}")
    print(f"equal: {equal}  more_severe: {more}")
    print(
        f"per-decision: reverse={uni_reverse} uphold={uni_uphold} "
        f"escalate={uni_escalate} split={split}"
    )
    for fp, (label, effects) in complete.items():
        if "reverse" in effects:
            print(f"  candidate: {fp} label={label} effects={effects}")
    print()


def main() -> None:
    transcript = load_transcript()
    for arm in ("arm4_reviewer", "arm4b_timeline_only"):
        batches = load_batches(arm)
        decisions = load_decisions(arm)
        labels = load_labels(arm)
        analyze(arm, batches, decisions, labels, transcript)


if __name__ == "__main__":
    main()

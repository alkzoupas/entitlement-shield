"""P5.4 scratch analysis (not production code, not wired into `make`): does Jev's
probability output separate legitimate from abusive `light_friction` decisions even
when its top action choice never does?

Reads the kept dev-tier (W-1) worlds from `scripts/arm4_run.py --keep-worlds
--world-dir data/synthetic/arm4-jev`, or the directory given as the first
argument (e.g. `data/synthetic/arm4-jev-summary` for the P5.5 summary arm).
Ground truth is read here, in an evaluation script, exactly like
`arm4_funnel_decompose.py` — never by the reviewer path.

Per decision (mean over its parsed samples): P(allow), P(allow)+P(light_friction),
the human-escalation Noul, and the action-choice confidence; each is scored by
ROC-AUC for "legitimate" (0.5 = no signal). Also reports per-decision consistency
of the top choice across the 3 samples and which labels hit provider failures.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
WORLD_DIR = REPO / "data" / "synthetic" / "arm4-jev"
ARMS = ("arm4_reviewer", "arm4b_timeline_only")
_PROB = re.compile(r"(allow|light_friction|hold_incremental_credits|temporary_hold)=([0-9.]+)")
_ESC = re.compile(r"human-escalation probability ([0-9.]+)")


def labels(world_dir: Path, arm: str) -> dict[str, str]:
    con = sqlite3.connect(world_dir / f"{arm}-ground-truth.db")
    acct_to_group = dict(con.execute("select account_id, split_group_id from split_groups"))
    group_to_label = dict(con.execute("select cluster_id, label from ground_truth"))
    return {a: group_to_label.get(g, "UNKNOWN").lower() for a, g in acct_to_group.items()}


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def analyse(world_dir: Path, arm: str) -> dict:
    lab = labels(world_dir, arm)
    con = sqlite3.connect(world_dir / f"{arm}-inference.db")
    decisions = {
        d: (acct, action.lower())
        for d, acct, action in con.execute(
            "select decision_id, account_id, proposed_action from decisions"
        )
    }
    per_decision: dict[str, list[dict]] = {}
    failures: Counter[str] = Counter()
    for did, raw, errors in con.execute(
        "select decision_id, structured_output_json, validation_errors from llm_reviews"
    ):
        acct, action = decisions[did]
        errors = None if errors in (None, "null") else errors
        if action != "light_friction":
            if errors:
                failures[f"{action}:{lab.get(acct, 'UNKNOWN')}"] += 1
            continue
        if errors or raw is None:
            failures[f"{action}:{lab.get(acct, 'UNKNOWN')}"] += 1
            continue
        verdict = json.loads(raw)
        probs = {k: float(v) for k, v in _PROB.findall(verdict["rationale"])}
        esc = _ESC.search(verdict["rationale"])
        per_decision.setdefault(did, []).append(
            {
                "choice": verdict["recommended_action"],
                "p_allow": probs.get("allow", 0.0),
                "p_lenient": probs.get("allow", 0.0) + probs.get("light_friction", 0.0),
                "escalate": float(esc.group(1)) if esc else float("nan"),
                "confidence": float(verdict["confidence"]),
            }
        )

    rows: dict[str, list[dict]] = {"legitimate": [], "abusive": [], "ambiguous": []}
    consistent = 0
    for did, samples in per_decision.items():
        label = lab.get(decisions[did][0], "UNKNOWN")
        bucket = label if label in rows else "abusive"
        consistent += len({s["choice"] for s in samples}) == 1
        rows[bucket].append({k: mean(s[k] for s in samples) for k in samples[0] if k != "choice"})

    signals = {}
    for metric in ("p_allow", "p_lenient", "escalate", "confidence"):
        legit = [r[metric] for r in rows["legitimate"]]
        abuse = [r[metric] for r in rows["abusive"]]
        signals[metric] = {
            "mean_legitimate": round(mean(legit), 4) if legit else None,
            "mean_abusive": round(mean(abuse), 4) if abuse else None,
            "auc_legitimate": None if (a := auc(legit, abuse)) is None else round(a, 3),
        }
    return {
        "light_friction_decisions": {k: len(v) for k, v in rows.items()},
        "top_choice_consistent_across_samples": f"{consistent}/{len(per_decision)}",
        "signals": signals,
        "provider_failures_by_action_label": dict(failures),
    }


def main() -> None:
    world_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else WORLD_DIR
    if not world_dir.exists():
        sys.exit(f"no kept worlds at {world_dir} — run arm4_run.py with --keep-worlds first")
    print(json.dumps({arm: analyse(world_dir, arm) for arm in ARMS}, indent=2))


if __name__ == "__main__":
    main()

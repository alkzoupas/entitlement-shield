"""Build a small ablation probe batch for the P4.10 contributions-masking
hypothesis (docs/p4.10-arm4-recovery-funnel-diagnosis.md).

Unlike arm4b_timeline_only, this holds the INSTRUCTION wording neutral (no
"unlike the full case, contributions are withheld" framing) while still
stripping the contributions block from the evidence and the CONTRIBUTION
citations from evidence_keys — isolating "contributions visible or not" from
"the prompt tells you contributions are deliberately masked."

Scratch script, not wired into any `make` target or production code. Reuses
the already-extracted arm4_reviewer batch files (no new detector/reviewer
run, no new sealed-store access).
"""

import json
import re
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data/synthetic/arm4-ablation"
SAMPLE_SIZE = 20

_CONTRIBUTIONS_BLOCK = re.compile(r"contributions:\n(?:  - .+\n)+")
_CONTRIBUTION_KEY_LINE = re.compile(r"  - contribution:\S+\n")


def strip_contributions(structured: str) -> str:
    """Remove both the `contributions:` evidence block AND any
    `contribution:*` lines rendered inside the case's own `evidence_keys:`
    listing — the latter is a separate render of the same data and would
    otherwise leak fired-feature identity back into the prompt text even
    after the JSON `evidence_keys` array is filtered."""
    stripped = _CONTRIBUTIONS_BLOCK.sub("", structured)
    assert stripped != structured, "contributions block not found — format changed?"
    stripped, n = _CONTRIBUTION_KEY_LINE.subn("", stripped)
    assert n > 0, "no contribution: evidence_keys lines found — format changed?"
    return stripped


def load_decisions() -> dict[str, dict]:
    con = sqlite3.connect(REPO / "data/synthetic/arm4-subagent-worlds/arm4_reviewer-inference.db")
    cur = con.cursor()
    cur.execute("select decision_id, account_id, proposed_action from decisions")
    return {r[0]: {"account_id": r[1], "proposed_action": r[2].lower()} for r in cur.fetchall()}


def load_labels() -> dict[str, str]:
    gt_path = REPO / "data/synthetic/arm4-subagent-worlds/arm4_reviewer-ground-truth.db"
    con = sqlite3.connect(gt_path)
    cur = con.cursor()
    cur.execute("select account_id, split_group_id from split_groups")
    acct_to_group = dict(cur.fetchall())
    cur.execute("select cluster_id, label from ground_truth")
    group_to_label = dict(cur.fetchall())
    return {acct: group_to_label.get(g, "UNKNOWN") for acct, g in acct_to_group.items()}


def load_full_cases() -> list[dict]:
    cases = []
    for i in range(7):
        batch_path = REPO / f"data/synthetic/arm4-subagent/arm4_reviewer-batch{i:02d}.json"
        d = json.loads(batch_path.read_text())
        cases.extend(d["cases"])
    return cases


def main() -> None:
    decisions = load_decisions()
    labels = load_labels()
    cases = load_full_cases()

    by_label: dict[str, list[dict]] = {"legitimate": [], "abusive": []}
    for c in cases:
        dec_id = c["structured"].split("decision_id: ")[1].split("\n")[0]
        dec = decisions.get(dec_id)
        if dec is None or dec["proposed_action"] != "light_friction":
            continue
        label = labels.get(dec["account_id"], "UNKNOWN")
        bucket = {"LEGITIMATE": "legitimate", "ABUSIVE": "abusive"}.get(label)
        if bucket is not None:
            by_label[bucket].append(c)

    half = SAMPLE_SIZE // 2
    selected = by_label["legitimate"][:half] + by_label["abusive"][:half]
    print(f"selected {len(selected)} cases ({half} legit target, {half} abusive target; "
          f"available legit={len(by_label['legitimate'])} abusive={len(by_label['abusive'])})")

    ablation_cases = []
    for c in selected:
        new_structured = strip_contributions(c["structured"])
        # "view: full" would now contradict the instruction (no contributions
        # promised) and the missing evidence — relabel to avoid an internal
        # inconsistency the model could notice/react to.
        new_structured = new_structured.replace(
            "view: full\n", "view: ablation_no_contributions\n", 1
        )
        new_evidence_keys = [k for k in c["evidence_keys"] if not k.startswith("contribution:")]
        ablation_cases.append(
            {
                "case_index": c["case_index"],
                "fp": c["fp"],
                "structured": new_structured,
                "untrusted": c["untrusted"],
                "evidence_keys": new_evidence_keys,
            }
        )

    instruction = (REPO / "data/synthetic/arm4-ablation/reviewer_ablation_v1.md").read_text()
    preamble_path = REPO / "data/synthetic/arm4-subagent/arm4_reviewer-preamble.json"
    system = json.loads(preamble_path.read_text())["system"]

    out = {
        "arm": "arm4_ablation_no_contributions_neutral",
        "batch_index": 0,
        "system": system,
        "instruction": instruction,
        "cases": ablation_cases,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ablation-batch00.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path} ({len(ablation_cases)} cases)")


if __name__ == "__main__":
    main()

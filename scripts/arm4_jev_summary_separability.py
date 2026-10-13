"""P5.5 scratch analysis (not production code, not wired into `make`): is the Jev
summary arm's separation a model judgment or a property of the summary itself?

For every `light_friction` decision in the kept P5.5 worlds, rebuilds the exact
code-side summary Jev received, then reports per label how many distinct summaries
exist and which single summary fields separate the labels with no overlap. Ground
truth is read here, in an evaluation script only — never by the reviewer path.

    PYTHONPATH=backend .venv/bin/python scripts/arm4_jev_summary_separability.py \
        [data/synthetic/arm4-jev-summary]
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.models.enums import ProposedAction
from app.models.inference import Decision
from app.providers.evidence_summary import summarize
from app.providers.typesafe_provider import split_evidence
from app.services.generator import WORLD_END
from app.services.llm_provider import LLMRequest
from app.services.prompt_builder import PromptVariant, build_prompt
from app.services.review_case import build_review_case

REPO = Path(__file__).resolve().parents[1]
DEFAULT_WORLD_DIR = REPO / "data" / "synthetic" / "arm4-jev-summary"
ARMS: tuple[tuple[str, str, PromptVariant], ...] = (
    ("arm4_reviewer", "full", "reviewer_v1"),
    ("arm4b_timeline_only", "timeline_only", "reviewer_timeline_only_v1"),
)


def _fields(state: dict) -> dict[str, str]:
    flagged = [v for k, v in state["accounts"].items() if "flagged" in k]
    return {
        "members": state["cluster"]["members"],
        "signup_pattern": state["cluster"]["signup_pattern"],
        "capacity_band": state["capacity_band"],
        "allowance_use": state["allowance_use"],
        "detector_signals_fired": json.dumps(state.get("detector_signals_fired")),
        "flagged_account": json.dumps(flagged, sort_keys=True),
    }


def analyse(world_dir: Path, arm: str, view: str, variant: PromptVariant) -> dict:
    gt = sqlite3.connect(world_dir / f"{arm}-ground-truth.db")
    acct_to_group = dict(gt.execute("select account_id, split_group_id from split_groups"))
    group_to_label = dict(gt.execute("select cluster_id, label from ground_truth"))
    distinct: dict[str, Counter[str]] = defaultdict(Counter)
    values: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    engine = create_engine(f"sqlite:///{world_dir / f'{arm}-inference.db'}")

    with Session(engine) as session:
        for decision in session.query(Decision).all():
            if ProposedAction(decision.proposed_action) != ProposedAction.LIGHT_FRICTION:
                continue
            label = group_to_label.get(acct_to_group.get(decision.account_id), "?").lower()
            case = build_review_case(session, decision, view=view, as_of=WORLD_END)
            request = LLMRequest.from_assembled_prompt(build_prompt(case, variant=variant))
            state, _ = summarize(split_evidence(request)[0])
            distinct[label][json.dumps(state, sort_keys=True)] += 1
            for name, value in _fields(state).items():
                values[name][label][value] += 1
    separating = []
    for name, by_label in values.items():
        sets = [set(c) for c in by_label.values()]
        if len(sets) > 1 and not set.intersection(*sets):
            separating.append(name)
    return {
        "decisions_and_distinct_summaries": {
            label: {"decisions": sum(c.values()), "distinct_summaries": len(c)}
            for label, c in distinct.items()
        },
        "single_fields_separating_labels_perfectly": separating,
        "signup_pattern_by_label": {
            label: dict(c) for label, c in values["signup_pattern"].items()
        },
    }


def main() -> None:
    world_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WORLD_DIR
    if not world_dir.exists():
        sys.exit(f"no kept worlds at {world_dir}")
    print(json.dumps({arm: analyse(world_dir, arm, v, p) for arm, v, p in ARMS}, indent=2))


if __name__ == "__main__":
    main()

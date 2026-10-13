"""P6.2 Step 0: render every open unit, caught and allowed, as the training/scoring prompt.

Label-free. Selects train+tune units of the detected store (split *membership* only, as the
P6.1 runner does) without the allowed-only filter, renders each with ``render_unit`` v2 and
the ``recall-v1-nocue`` system prompt (docs/p6.2-posttrain-plan.md D3), and writes one JSON
line per unit: key, split group, detector score, caught/allowed, system and user text.
The trained model never sees the detector score; it is kept here only to mark the stratum.
No ground-truth labels are read, and the sealed test split is never selected.

The output feeds ``scripts/p6_2_step0.py`` (token-length manifest, run in the separate
training venv). It is regenerable from the detected store and not committed.

Usage:
    PYTHONPATH=backend .venv/bin/python scripts/p6_2_render_prompts.py \
        --out data/tier4-v2/runs/p6-2-step0/prompts.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p6_recall_pilot import (  # noqa: E402
    DETECTED,
    OPEN_SPLITS,
    PROMPTS,
    WORLD,
    _key,
    _ro,
    _sha256,
    render_unit,
)

from app.db.engine import make_inference_engine  # noqa: E402
from app.db.ground_truth_reader import load_split_groups  # noqa: E402
from app.db.ground_truth_store import make_ground_truth_engine  # noqa: E402
from app.db.world_provenance import verify_world_pair  # noqa: E402
from app.services.evaluation import assign_split  # noqa: E402
from app.services.risk_scorer import SCORE_THRESHOLD  # noqa: E402

PROMPT = "recall-v1-nocue"
RENDER = "v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    gt_url = f"sqlite:///{(WORLD / 'ground_truth.db').resolve()}"
    inf_url = f"sqlite:///{(WORLD / 'inference.db').resolve()}"
    with (
        Session(make_inference_engine(inf_url), future=True) as inf,
        Session(make_ground_truth_engine(gt_url), future=True) as gt,
    ):
        verify_world_pair(inf, gt)
        groups = load_split_groups(gt)  # label-free membership

    system = PROMPTS[PROMPT]
    rows = []
    with _ro(DETECTED) as det, _ro(WORLD / "inference.db") as conn:
        for (raw,) in det.execute("select payload_json from detection_evidence"):
            payload = json.loads(raw)
            split_groups = {groups.get(m) for m in payload["member_account_ids"]}
            if len(split_groups) != 1 or None in split_groups:
                raise SystemExit(f"unit {_key(payload)} has no single split group")
            group = next(iter(split_groups))
            if assign_split(group) not in OPEN_SPLITS:  # type: ignore[arg-type]
                continue
            rows.append({
                "key": _key(payload),
                "split_group": group,
                "detector_score": payload["risk_score"],
                "allowed": payload["risk_score"] < SCORE_THRESHOLD,
                "accounts": len(payload["member_account_ids"]),
                "system": system,
                "user": render_unit(conn, payload, render=RENDER),
            })
    rows.sort(key=lambda r: r["key"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    meta = {
        "units": len(rows),
        "allowed": sum(r["allowed"] for r in rows),
        "caught": sum(not r["allowed"] for r in rows),
        "split_groups": len({r["split_group"] for r in rows}),
        "prompt": PROMPT,
        "render": RENDER,
        "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "renderer_sha256": _sha256(Path(__file__).resolve().parent / "p6_recall_pilot.py"),
        "detected_store_sha256": _sha256(DETECTED),
        "prompts_sha256": _sha256(args.out),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

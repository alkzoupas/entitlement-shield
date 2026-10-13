"""P6.3: export the v3 render of every open allowed unit for the Nimble probe (P6-3).

Label-free. Selects the 492 open allowed units exactly as the P6.1 runner does
(``_open_units``: detector score below the threshold, train+tune split *membership* only)
and renders each with ``render_unit(..., render="v3")`` at every lookback of the P6-3 fit
rule (none, 28, 14, 7, 0). The fit itself is decided later, in the MLX venv, with Nimble's
tokenizer and prompt builder (``scripts/p6_3_nimble_run.py fit``). No ground-truth labels
are read, the detector score is never part of the render, and the sealed test split is
never selected.

The output is regenerable from the detected store and not committed.

Usage:
    PYTHONPATH=backend .venv/bin/python scripts/p6_3_export_renders.py \
        --out data/tier4-v2/runs/p6-3-nimble
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p6_recall_pilot import (  # noqa: E402
    DETECTED,
    NOCUE_PROMPT,
    WORLD,
    _open_units,
    _ro,
    _sha256,
    render_unit,
)

LOOKBACKS: tuple[int | None, ...] = (None, 28, 14, 7, 0)  # the P6-3 fit rule, in order
RENDER = "v3"
# P6-3 schema text: NOCUE_PROMPT up to the blank line before "Answer with...".
DESCRIPTION = NOCUE_PROMPT.split("\n\nAnswer with")[0] + "\n\nRate the risk from 1 to 10."
assert DESCRIPTION != NOCUE_PROMPT and "RISK_LEVEL" not in DESCRIPTION
SCHEMA = {
    "risk_level": {
        "type": "enum",
        "description": DESCRIPTION,
        "choices": [str(k) for k in range(1, 11)],
        "choice_descriptions": {"1": "clearly independent legitimate users",
                                "10": "clearly coordinated abuse"},
    }
}


def _lb(lookback: int | None) -> str:
    return "full" if lookback is None else str(lookback)


def export(out: Path) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    units = _open_units(DETECTED)
    renders_path = out / "renders.jsonl"
    with _ro(WORLD / "inference.db") as conn, renders_path.open("w") as sink:
        for key, payload in units:
            renders = {_lb(lb): render_unit(conn, payload, render=RENDER, lookback=lb)
                       for lb in LOOKBACKS}
            row = {"key": key, "detector_score": payload["risk_score"], "renders": renders}
            sink.write(json.dumps(row, sort_keys=True) + "\n")
    schema_path = out / "schema.json"
    schema_path.write_text(json.dumps(SCHEMA, indent=2, ensure_ascii=False) + "\n")
    header: dict[str, object] = {
        "decision": "P6-3 (docs/decisions.md)",
        "render": RENDER,
        "lookbacks": [_lb(lb) for lb in LOOKBACKS],
        "units": len(units),
        "exporter_sha256": _sha256(Path(__file__)),
        "pilot_sha256": _sha256(Path(__file__).resolve().parent / "p6_recall_pilot.py"),
        "detected_store_sha256": _sha256(DETECTED),
        "inference_store_sha256": _sha256(WORLD / "inference.db"),
        "renders_sha256": _sha256(renders_path),
        "schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
    }
    (out / "export.json").write_text(json.dumps(header, indent=2, sort_keys=True) + "\n")
    return header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

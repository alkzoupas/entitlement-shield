"""P4.9 exploratory tooling: split `scripts/arm4_subagent_extract.py`'s per-arm
prompt dump into fixed-size batch files, one JSON file per batch, each holding
the arm's shared preamble (system + instruction text, identical across every
case in the arm) plus a list of per-case structured-evidence payloads.

A batch file is exactly what one Claude Code subagent call is handed: it
reconstructs each case's full reviewer prompt from `system` + `instruction` +
`cases[i].structured` (+ `cases[i].untrusted`, almost always empty in this
synthetic dev-tier fixture) and returns one verdict per case.

Usage:
    python scripts/arm4_subagent_batch.py [--in-dir data/synthetic/arm4-subagent]
        [--batch-size 20]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARMS = ("arm4_reviewer", "arm4b_timeline_only")


def batch_arm(in_dir: Path, arm: str, batch_size: int) -> list[Path]:
    preamble = json.loads((in_dir / f"{arm}-preamble.json").read_text())
    cases = []
    with (in_dir / f"{arm}-prompts.jsonl").open() as f:
        for line in f:
            cases.append(json.loads(line))

    written: list[Path] = []
    for start in range(0, len(cases), batch_size):
        chunk = cases[start : start + batch_size]
        batch_index = start // batch_size
        batch = {
            "arm": arm,
            "batch_index": batch_index,
            "system": preamble["system"],
            "instruction": preamble["instruction"],
            "cases": [
                {
                    "case_index": start + i,
                    "fp": c["fp"],
                    "structured": c["structured"],
                    "untrusted": c["untrusted"],
                    "evidence_keys": c["evidence_keys"],
                }
                for i, c in enumerate(chunk)
            ],
        }
        out_path = in_dir / f"{arm}-batch{batch_index:02d}.json"
        out_path.write_text(json.dumps(batch, indent=2))
        written.append(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in-dir", type=Path, default=REPO_ROOT / "data" / "synthetic" / "arm4-subagent"
    )
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    manifest: dict[str, list[str]] = {}
    for arm in ARMS:
        paths = batch_arm(args.in_dir, arm, args.batch_size)
        manifest[arm] = [str(p.relative_to(REPO_ROOT)) for p in paths]
        print(f"{arm}: {len(paths)} batches of up to {args.batch_size} cases")
    (args.in_dir / "batches-manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

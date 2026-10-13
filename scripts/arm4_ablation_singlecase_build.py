"""Split the P4.10 ablation batch (data/synthetic/arm4-ablation/ablation-batch00.json)
into 20 single-case prompt files, one per case, for single-case subagent dispatch.

Single-case dispatch eliminates the batch-collapse failure mode found in P4.9's
original ~20-case-per-dispatch method (docs/p4.10-arm4-recovery-funnel-diagnosis.md):
a subagent given only one case has nothing to collapse a verdict across. This
trades call count (20 cases x 3 passes = 60 dispatches, vs 1 batch x 3 passes)
for a method that can actually measure per-case discrimination.

Scratch script, not wired into any `make` target or production code.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data/synthetic/arm4-ablation/single-case"


def main() -> None:
    batch = json.loads((REPO / "data/synthetic/arm4-ablation/ablation-batch00.json").read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for c in batch["cases"]:
        out = {
            "system": batch["system"],
            "instruction": batch["instruction"],
            "case": {
                "fp": c["fp"],
                "structured": c["structured"],
                "untrusted": c["untrusted"],
                "evidence_keys": c["evidence_keys"],
            },
        }
        out_path = OUT_DIR / f"case-{c['fp']}.json"
        out_path.write_text(json.dumps(out, indent=2))

    print(f"wrote {len(batch['cases'])} single-case files to {OUT_DIR}")


if __name__ == "__main__":
    main()

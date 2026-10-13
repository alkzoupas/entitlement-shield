"""P4.9 exploratory tooling: assemble the Claude-subagent verdicts produced by
`scripts/arm4_subagent_batch.py`'s batch files (and dispatched by hand to Claude
Code subagents — see `docs/p4.9-arm4-subagent-haiku-findings.md`) into a
`ResponseCache`-format transcript that `scripts/arm4_run.py --replay` can score.

Each pass's batch result file is `<arm>-batch<NN>-pass<P>.json`: a JSON array of
`{case_index, fp, verdict}` in the same order as the corresponding
`<arm>-batch<NN>.json`. For every `(fp, pass)` this writes one transcript line
keyed `f"{fp}:{pass}"` (matching `CachingProvider`'s per-fingerprint call
counter, `backend/app/providers/cache.py`), with `response.text` set to the
verdict's compact JSON (matching what a real `LLMResponse.text` holds).

Every verdict is validated against `ReviewerVerdict` (pydantic) before being
written — an invalid entry would otherwise pass `--replay` a malformed response,
which is exactly the case R-7's repair-retry exists for in the live path, but a
static transcript has no live provider to retry against, so `--replay` would
raise deep inside `_run_sample` on first parse. Fail loudly here instead, with
the offending (arm, batch, pass, case_index) named.

Usage:
    python scripts/arm4_subagent_assemble.py [--in-dir data/synthetic/arm4-subagent]
        [--out data/synthetic/arm4-subagent-haiku-transcript.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.schemas.reviewer import ReviewerVerdict  # noqa: E402

ARMS = ("arm4_reviewer", "arm4b_timeline_only")
MODEL_SNAPSHOT = "claude-haiku-4-5-subagent-v1"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def assemble(in_dir: Path, out_path: Path) -> int:
    entries: dict[str, dict] = {}
    n_batches: dict[str, int] = {}
    for arm in ARMS:
        manifest_batches = sorted(in_dir.glob(f"{arm}-batch*.json"))
        # Exclude result files, which also match `{arm}-batch*.json` under a
        # naive glob if `in_dir` were the results dir; batch files live directly
        # in `in_dir`, results in `in_dir/results`, so this glob is already safe.
        n_batches[arm] = len(manifest_batches)
        for batch_path in manifest_batches:
            batch_index = int(batch_path.stem.rsplit("batch", 1)[1])
            for pass_index in range(3):
                result_name = f"{arm}-batch{batch_index:02d}-pass{pass_index}.json"
                result_path = in_dir / "results" / result_name
                results = json.loads(result_path.read_text())
                for item in results:
                    fp = item["fp"]
                    raw_verdict = item["verdict"]
                    try:
                        verdict = ReviewerVerdict.model_validate(raw_verdict)
                    except Exception as exc:  # noqa: BLE001 - want the exact failing item named
                        raise SystemExit(
                            f"invalid verdict: {arm} batch{batch_index:02d} pass{pass_index} "
                            f"case_index={item.get('case_index')} fp={fp}: {exc}"
                        ) from exc
                    text = verdict.model_dump_json()
                    key = f"{fp}:{pass_index}"
                    if key in entries:
                        raise SystemExit(f"duplicate transcript key {key!r} (fp collision?)")
                    entries[key] = {
                        "text": text,
                        "model_snapshot": MODEL_SNAPSHOT,
                        "latency_ms": 0.0,
                        "usage": {
                            "prompt_tokens": _estimate_tokens(raw_verdict.get("rationale", "")),
                            "completion_tokens": _estimate_tokens(text),
                            "total_tokens": _estimate_tokens(text),
                        },
                    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for key, response in entries.items():
            f.write(json.dumps({"key": key, "response": response}) + "\n")
    return len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in-dir", type=Path, default=REPO_ROOT / "data" / "synthetic" / "arm4-subagent"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "synthetic" / "arm4-subagent-haiku-transcript.jsonl",
    )
    args = parser.parse_args()
    n = assemble(args.in_dir, args.out)
    print(f"wrote {n} transcript entries to {args.out}")


if __name__ == "__main__":
    main()

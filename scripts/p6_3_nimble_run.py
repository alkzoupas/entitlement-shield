# mlx, mlx-lm and the Nimble repo live only in the Nimble MLX venv, not the project venv.
# mypy: disable-error-code="import-not-found"
"""P6.3: score the exported v3 renders with Bespoke Nimble-9B and its base (P6-3).

Runs in the Nimble MLX venv (``~/code/nimble/.venv-mlx``) and imports nothing from
``backend``. **Label-free:** it reads only ``scripts/p6_3_export_renders.py`` output
(renders, schema) and never the ground-truth store.

* ``fit``: the P6-3 fit rule, decided once with the adapter's tokenizer and prompt
  contract: ``lookback=None`` if the full Nimble prompt fits ``MAX_INPUT_TOKENS``, else the
  first of 28, 14, 7, 0 that fits; a unit that never fits fails closed. Writes
  ``fit.json`` (counts before any label is read). No model is loaded.
* ``run --arm a|b``: one arm over all units (or ``--smoke``: the first full-fit and the
  first fallback unit, into ``<out>/smoke/``). Prompts are built with the adapter's
  tokenizer and contract builder for *both* arms, so arm (a) sees token-identical input;
  the token count must match ``fit.json``. The logits come from the repo's MLX
  ``ParallelScorer.evaluate`` (mode ``parallel``); the softmax at T = 1.0, the expected
  level and the argmax are computed here. Resumable; ``run-<arm>.json`` pins the inputs.
  ``--crosscheck`` also runs the scorer's own ``score()`` path and records the gap.

Usage (from the entitlement-shield root):
    ~/code/nimble/.venv-mlx/bin/python scripts/p6_3_nimble_run.py fit \
        --out data/tier4-v2/runs/p6-3-nimble
    ~/code/nimble/.venv-mlx/bin/python scripts/p6_3_nimble_run.py run \
        --out data/tier4-v2/runs/p6-3-nimble --arm b [--smoke --crosscheck]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

NIMBLE = Path.home() / "code" / "nimble"
NIMBLE_COMMIT = "4edb0703a2e75d0170c3a45645a1961f4d1a2867"
HUB = NIMBLE / ".cache" / "huggingface" / "hub"
ADAPTER_REPO = "bespokelabs/Bespoke-Nimble-9B"
ADAPTER_REVISION = "bd792f44ec8e265be861bfcdf4e05967ffe0e858"
ADAPTER_SHA256 = "29ef39b072dee97287947455337879c1e916705c2f727287922a2d81f5e2f20a"
BASE_REPO = "Qwen/Qwen3.5-9B"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ADAPTER = HUB / "models--bespokelabs--Bespoke-Nimble-9B" / "snapshots" / ADAPTER_REVISION
ARMS = {
    "a": HUB / "models--Qwen--Qwen3.5-9B" / "snapshots" / BASE_REVISION,
    "b": NIMBLE / ".cache" / "models" / f"nimble-9b-{ADAPTER_REVISION}",
}
LOOKBACKS = ("full", "28", "14", "7", "0")  # must match the export, in order
MAX_INPUT_TOKENS = 8192
TEMPERATURE = 1.0  # shipped, unfitted (temperature_config.json)
FIELD = "risk_level"
LEVELS = [str(k) for k in range(1, 11)]
SUM_TOLERANCE = 1e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def _setup() -> tuple[Any, Any, dict[str, Any]]:
    """The adapter's tokenizer, contract prompt builder and contract, from the pinned repo."""
    head = subprocess.run(["git", "-C", str(NIMBLE), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != NIMBLE_COMMIT:
        raise SystemExit(f"Nimble repo at {head}, expected {NIMBLE_COMMIT}")
    sys.path.insert(0, str(NIMBLE))
    from nimble.training.candidate_schema import validate_contract
    from transformers import AutoTokenizer

    contract = json.loads((ADAPTER / "schema_config.json").read_text())
    if (contract["model"], contract["revision"]) != (BASE_REPO, BASE_REVISION):
        raise SystemExit("adapter contract names another base")
    if _sha256(ADAPTER / "adapter_model.safetensors") != ADAPTER_SHA256:
        raise SystemExit("adapter weights differ from the pinned release")
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER))
    return tokenizer, validate_contract(contract, tokenizer), contract


def _inputs(out: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    export = json.loads((out / "export.json").read_text())
    renders_path, schema_path = out / "renders.jsonl", out / "schema.json"
    if _sha256(renders_path) != export["renders_sha256"]:
        raise SystemExit("renders.jsonl differs from export.json")
    if _sha256(schema_path) != export["schema_sha256"]:
        raise SystemExit("schema.json differs from export.json")
    if export["lookbacks"] != list(LOOKBACKS):
        raise SystemExit("export lookbacks differ from the fit rule")
    rows = [json.loads(line) for line in renders_path.read_text().splitlines()]
    sys.path.insert(0, str(NIMBLE))
    from nimble.scoring.serving_schema import parse_schema

    schema = parse_schema(schema_path.read_text())
    if list(schema) != [FIELD] or schema[FIELD]["choices"] != LEVELS:
        raise SystemExit("schema is not the pre-registered risk_level field")
    return export, rows, schema


def _too_long(error: ValueError) -> bool:
    return "Nothing was truncated" in str(error)


def fit(out: Path) -> dict[str, Any]:
    export, rows, schema = _inputs(out)
    tokenizer, prepare, _ = _setup()
    units: dict[str, Any] = {}
    for row in rows:
        chosen, full_tokens = None, None
        for lb in LOOKBACKS:
            try:
                prepared = prepare(tokenizer, row["renders"][lb], schema, MAX_INPUT_TOKENS)
            except ValueError as error:
                if not _too_long(error):
                    raise
                continue
            chosen = {"lookback": lb, "tokens": len(prepared.full_ids[0]),
                      "prompt_ids_sha256": _ids_sha256(prepared.full_ids[0])}
            break
        if chosen is None:
            units[row["key"]] = {"lookback": None, "error": "fails at every lookback"}
            continue
        if chosen["lookback"] != "full":  # the full length, for the record only
            full_tokens = _full_length(tokenizer, prepare, row["renders"]["full"], schema)
        units[row["key"]] = {**chosen, "full_tokens": full_tokens or chosen["tokens"]}
    counts: dict[str, int] = {}
    for unit in units.values():
        name = unit["lookback"] or "failed"
        counts[name] = counts.get(name, 0) + 1
    doc = {"decision": "P6-3", "max_input_tokens": MAX_INPUT_TOKENS,
           "renders_sha256": export["renders_sha256"], "schema_sha256": export["schema_sha256"],
           "runner_sha256": _sha256(Path(__file__)), "nimble_commit": NIMBLE_COMMIT,
           "adapter_revision": ADAPTER_REVISION, "counts": counts, "units": units}
    (out / "fit.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    tokens = sorted(u["tokens"] for u in units.values() if u.get("tokens"))
    return {"counts": counts, "units": len(units), "tokens_median": tokens[len(tokens) // 2],
            "tokens_max": tokens[-1],
            "fallback_keys": sorted(k for k, u in units.items() if u["lookback"] != "full")}


def _full_length(tokenizer: Any, prepare: Any, context: str, schema: dict[str, Any]) -> int:
    prepared = prepare(tokenizer, context, schema, 1 << 20)
    return len(prepared.full_ids[0])


def _model_hashes(path: Path) -> dict[str, str]:
    return {p.name: _sha256(p.resolve()) for p in sorted(path.glob("*.safetensors"))}


def _softmax(logits: list[float]) -> list[float]:
    top = max(logits)
    exps = [math.exp((x - top) / TEMPERATURE) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def run(out: Path, arm: str, smoke: bool, crosscheck: bool) -> None:
    export, rows, schema = _inputs(out)
    fitted = json.loads((out / "fit.json").read_text())
    if fitted["renders_sha256"] != export["renders_sha256"]:
        raise SystemExit("fit.json was made from other renders")
    tokenizer, prepare, contract = _setup()
    model_path = ARMS[arm]
    ready = model_path / "READY.json"
    if arm == "b":
        if json.loads(ready.read_text())["adapter_sha256"] != ADAPTER_SHA256:
            raise SystemExit("merged model was made from another adapter")
        model_id, revision = ADAPTER_REPO, ADAPTER_REVISION
    else:
        if ready.exists() or (model_path / "adapter_config.json").exists():
            raise SystemExit("arm a must be the unmerged base")
        model_id, revision = BASE_REPO, BASE_REVISION

    units = {k: u for k, u in fitted["units"].items() if u["lookback"]}
    if smoke:
        full = next(k for k in sorted(units) if units[k]["lookback"] == "full")
        fallback = next(k for k in sorted(units) if units[k]["lookback"] != "full")
        units = {full: units[full], fallback: units[fallback]}
        out = out / "smoke"
        out.mkdir(exist_ok=True)
    by_key = {r["key"]: r for r in rows}

    import mlx.core as mx
    from nimble.scoring.parallel_scorer import ParallelScorer

    header = {
        "decision": "P6-3", "arm": arm, "model_id": model_id, "revision": revision,
        "model_path": str(model_path), "model_safetensors_sha256": _model_hashes(model_path),
        "adapter_sha256": ADAPTER_SHA256,
        "contract_sha256": _sha256(ADAPTER / "schema_config.json"),
        "nimble_commit": NIMBLE_COMMIT, "runner_sha256": _sha256(Path(__file__)),
        "renders_sha256": export["renders_sha256"], "schema_sha256": export["schema_sha256"],
        "fit_sha256": _sha256(out.parent / "fit.json" if smoke else out / "fit.json"),
        "max_input_tokens": MAX_INPUT_TOKENS, "temperature": TEMPERATURE, "mode": "parallel",
        "packages": {p: md.version(p) for p in ("mlx", "mlx-lm", "transformers")},
        "units": sorted(units),
    }
    header_path = out / f"run-{arm}.json"
    if header_path.exists() and json.loads(header_path.read_text()) != header:
        raise SystemExit(f"{header_path} differs from this run; use a new --out")
    header_path.write_text(json.dumps(header, indent=2, sort_keys=True) + "\n")

    outcomes_path = out / f"arm-{arm}.jsonl"
    done = set()
    if outcomes_path.exists():
        done = {json.loads(line)["key"] for line in outcomes_path.read_text().splitlines()}
    todo = [k for k in sorted(units) if k not in done]
    print(f"arm {arm}: units {len(units)}, done {len(done)}, this run {len(todo)}", flush=True)
    started = time.perf_counter()
    scorer = ParallelScorer(model_path=str(model_path), max_input_tokens=MAX_INPUT_TOKENS,
                            temperature=TEMPERATURE, model_id=model_id, revision=revision)
    print(f"loaded in {time.perf_counter() - started:.0f}s", flush=True)
    with outcomes_path.open("a") as sink:
        for i, key in enumerate(todo, 1):
            unit = units[key]
            context = by_key[key]["renders"][unit["lookback"]]
            prepared = prepare(tokenizer, context, schema, MAX_INPUT_TOKENS)
            ids = prepared.full_ids[0]
            if len(ids) != unit["tokens"] or _ids_sha256(ids) != unit["prompt_ids_sha256"]:
                raise SystemExit(f"{key}: prompt differs from fit.json")
            logit_rows, stats = scorer.evaluate(prepared, mode="parallel")
            logits = [float(x) for x in logit_rows[0].tolist()]
            probs = _softmax(logits)
            total = sum(probs)
            valid = all(map(math.isfinite, logits)) and abs(total - 1) <= SUM_TOLERANCE
            choices = prepared.choices[0]
            best = max(range(len(probs)), key=probs.__getitem__)
            row: dict[str, Any] = {
                "key": key, "arm": arm, "lookback": unit["lookback"], "prompt_tokens": len(ids),
                "candidate_token_ids": prepared.candidate_ids[0], "choices": choices,
                "logits": logits, "probs": probs, "prob_sum": total, "valid": valid,
                "expected_level": sum(int(c) * p for c, p in zip(choices, probs, strict=True)),
                "argmax_level": int(choices[best]),
                "seconds": round(stats["total_seconds"], 2),
                "mlx_peak_gib": round(stats["mlx_peak_active_gib"], 2),
            }
            if crosscheck:
                result = scorer.score(context, schema, mode="parallel")
                own = result["fields"][FIELD]
                row["crosscheck"] = {
                    "value": own["value"],
                    "max_abs_prob_gap": max(abs(own["scores"][c] - p)
                                            for c, p in zip(choices, probs, strict=True)),
                    "same_candidate_ids": own["candidate_token_ids"] == prepared.candidate_ids[0],
                }
            sink.write(json.dumps(row, sort_keys=True) + "\n")
            sink.flush()
            mx.clear_cache()
            print(f"[{i}/{len(todo)}] {key} L={unit['lookback']} tokens={len(ids)} "
                  f"E={row['expected_level']:.3f} argmax={row['argmax_level']} "
                  f"sum={total:.6f} {row['seconds']}s", flush=True)
    invalid = sum(1 for line in outcomes_path.read_text().splitlines()
                  if not json.loads(line)["valid"])
    print(f"invalid rows in {outcomes_path.name}: {invalid}", flush=True)
    if invalid:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--arm", choices=sorted(ARMS), required=True)
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--crosscheck", action="store_true")
    args = parser.parse_args()
    if args.cmd == "fit":
        print(json.dumps(fit(args.out), indent=2, sort_keys=True))
    else:
        run(args.out, args.arm, args.smoke, args.crosscheck)


if __name__ == "__main__":
    main()

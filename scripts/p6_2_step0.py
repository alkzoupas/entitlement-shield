# mlx, mlx-lm and mlx-lm-lora live only in the training venv, not the project venv.
# mypy: disable-error-code="import-not-found"
"""P6.2 Step 0: feasibility of the real training and scoring path (docs/p6.2-posttrain-plan.md).

Runs in the separate training venv (``~/venvs/es-p6-train``: mlx, mlx-lm, mlx-lm-lora), not
the project venv, and imports nothing from ``backend``. **Label-free:** it reads only the
rendered prompts of ``p6_2_render_prompts.py``. Every training target here is a fixed dummy
answer, not a label. Nothing it trains is kept beyond the run, and nothing feeds P6.2's CV.

Checks, each recorded in ``<out>/feasibility.json``:

1. environment: package versions, model repo and revision, model shape, device memory;
2. chat template (``enable_thinking: false``): the exact assistant continuation of each of
   the 10 answers, read off the template, and whether digits tokenize one per token;
3. token-length manifest over every open unit, caught and allowed (``length_manifest.json``);
4. the fast gated-delta VJP that mlx-lm-lora installs against mlx-lm's reference
   recurrence: outputs and gradients on a random case;
5. the training path: weighted, completion-masked loss computed only on answer positions;
   an explicit ``max_seq_length`` that refuses truncation; answer and end tokens asserted
   in the loss mask; zero weight gives zero gradient; the loss scales with the weight;
   gradients finite and non-zero; the adapter and the loss change after one update;
6. one QLoRA step at about 4k, 9k and 18k tokens with layer checkpointing: wall time,
   tokens/s and peak memory (second step reported; the first warms up);
7. scoring: the expected level over the 10 *terminated* candidates (D7), with a toy
   1-vs-10 case showing why the end token matters, and the cost of scoring one long prompt.

Usage:
    ~/venvs/es-p6-train/bin/python scripts/p6_2_step0.py \
        --prompts data/tier4-v2/runs/p6-2-step0/prompts.jsonl \
        --out data/tier4-v2/runs/p6-2-step0
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import math
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

MODEL_REPO = "mlx-community/Qwen3.5-9B-MLX-4bit"
MODEL_REVISION = "938d8919941c6e7efd3c7150eff7fe9d12afa631"
LEVELS = range(1, 11)
DUMMY_TARGET = 10  # a fixed answer for the training-path checks; never a label
LORA = {"rank": 16, "scale": 2.0, "dropout": 0.0}  # Step 0 setting, not the frozen recipe
LR = 1e-4
TARGET_LENGTHS = (1024, 2048, 4096, 9216, 18432)
MAX_SEQ_LENGTH = 20480  # above the longest open prompt; longer inputs are refused
GROUP_CAP = 14  # D3's optional cap, for the epoch-token estimate only
SEED = 20261015
MEMORY_LIMIT_GIB = 40  # of 48 GiB unified memory; keep the machine out of swap


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gib(n: int) -> float:
    return round(n / 2**30, 2)


def _stats(xs: list[int]) -> dict[str, Any]:
    xs = sorted(xs)
    return {"n": len(xs), "mean": round(statistics.fmean(xs)), "median": xs[len(xs) // 2],
            "p90": xs[int(0.9 * (len(xs) - 1))], "max": xs[-1], "total": sum(xs)}


# --------------------------------------------------------------------------- template


def _encode(tokenizer: Any, messages: list[dict[str, str]], generation: bool) -> list[int]:
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=generation,
                                         enable_thinking=False, tokenize=False)
    return list(tokenizer.encode(text, add_special_tokens=False))


def prompt_ids(tokenizer: Any, row: dict[str, Any]) -> list[int]:
    messages = [{"role": "system", "content": row["system"]},
                {"role": "user", "content": row["user"]}]
    return _encode(tokenizer, messages, True)


def continuations(tokenizer: Any, row: dict[str, Any]) -> dict[int, list[int]]:
    """The template's own assistant continuation for each answer, up to its end-of-turn.

    Rendering the full conversation and stripping the generation prompt keeps whatever the
    frozen template emits (an empty think block, the end-of-turn token) instead of guessing.
    """
    base = prompt_ids(tokenizer, row)
    end_ids = {tokenizer.convert_tokens_to_ids("<|im_end|>")}
    out = {}
    for level in LEVELS:
        messages = [{"role": "system", "content": row["system"]},
                    {"role": "user", "content": row["user"]},
                    {"role": "assistant", "content": f"RISK_LEVEL: {level}"}]
        full = _encode(tokenizer, messages, False)
        if full[: len(base)] != base:
            raise SystemExit("chat template: the generation prompt is not a prefix of the "
                             "full conversation; the continuation cannot be read off it")
        cont = full[len(base):]
        cut = next((i for i, t in enumerate(cont) if t in end_ids), None)
        if cut is None:
            raise SystemExit(f"chat template: no end-of-turn token after level {level}")
        out[level] = cont[: cut + 1]
    return out


# --------------------------------------------------------------------------- model helpers


def _text_model(model: Any) -> Any:
    return getattr(model, "language_model", model)


def answer_logits(model: Any, ids: Any, start: int) -> Any:
    """Logits for positions ``start-1 .. end-1`` only (the answer tokens' predictions).

    The full vocabulary at every position of an 18k prompt would be ~9 GB, so the head is
    applied to the answer positions alone.
    """
    tm = _text_model(model)
    hidden = tm.model(ids[None, :-1])[:, start - 1:, :]
    if tm.args.tie_word_embeddings:
        return tm.model.embed_tokens.as_linear(hidden)
    return tm.lm_head(hidden)


def make_loss(mx: Any, nn: Any) -> Any:
    def loss_fn(model: Any, ids: Any, start: int, weight: float) -> Any:
        """Weighted, completion-masked CE: mean over the answer tokens, times the weight."""
        logits = answer_logits(model, ids, start).astype(mx.float32)
        targets = ids[None, start:]
        ce = nn.losses.cross_entropy(logits, targets)
        return weight * ce.mean()
    return loss_fn


def build_example(prompt: list[int], cont: list[int]) -> tuple[list[int], int]:
    ids = prompt + cont
    if len(ids) > MAX_SEQ_LENGTH:
        raise ValueError(f"example of {len(ids)} tokens exceeds max_seq_length "
                         f"{MAX_SEQ_LENGTH}; refusing to truncate")
    return ids, len(prompt)


def clone_cache(cache: list[Any]) -> list[Any]:
    """Independent copy of a prompt cache, so each candidate continues the same state."""
    from mlx_lm.models.cache import ArraysCache, KVCache

    out = []
    for c in cache:
        if isinstance(c, KVCache):
            n = KVCache()
            n.keys = c.keys[..., : c.offset, :]
            n.values = c.values[..., : c.offset, :]
            n.offset = c.offset
        elif isinstance(c, ArraysCache):
            n = ArraysCache(size=len(c.cache))
            n.cache = list(c.cache)
            n.left_padding, n.lengths = c.left_padding, c.lengths
        else:
            raise TypeError(f"unsupported cache type {type(c).__name__}")
        out.append(n)
    return out


def candidate_logprobs(mx: Any, model: Any, prompt: list[int],
                       conts: dict[int, list[int]]) -> dict[int, float]:
    """log P(full terminated continuation | prompt) for each level, one prefill shared."""
    tm = _text_model(model)
    cache = tm.make_cache()
    logits = model(mx.array(prompt)[None], cache=cache)[:, -1, :].astype(mx.float32)
    first = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(first)
    out = {}
    for level, cont in conts.items():
        total = first[0, cont[0]]
        if len(cont) > 1:
            branch = clone_cache(cache)
            lg = model(mx.array(cont[:-1])[None], cache=branch).astype(mx.float32)
            lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            total = total + mx.take_along_axis(
                lp[0], mx.array(cont[1:])[:, None], axis=-1).sum()
        out[level] = total.item()
    return out


def expected_level(logprobs: dict[int, float]) -> dict[str, Any]:
    m = max(logprobs.values())
    z = sum(math.exp(v - m) for v in logprobs.values())
    probs = {lv: math.exp(v - m) / z for lv, v in logprobs.items()}
    return {"expected_level": sum(lv * p for lv, p in probs.items()),
            "probs": {str(k): round(v, 6) for k, v in probs.items()},
            "valid_answer_mass": math.exp(m) * z}


# --------------------------------------------------------------------------- checks


def check_fast_vjp(mx: Any) -> dict[str, Any]:
    """``mx.fast.gated_delta_update`` (plain, and in checkpointed chunks) vs mlx-lm's reference
    ops recurrence: outputs and gradients on one random case."""
    from mlx_lm.models import gated_delta

    mx.random.seed(SEED)
    b, t, hk, hv, dk, dv = 1, 96, 4, 4, 128, 128  # hk == hv: no head repeat to match
    q = mx.random.normal((b, t, hk, dk)) * 0.1
    k = mx.random.normal((b, t, hk, dk)) * 0.1
    v = mx.random.normal((b, t, hv, dv)) * 0.1
    a = mx.random.normal((b, t, hv))
    bb = mx.random.normal((b, t, hv))
    a_log = mx.log(mx.random.uniform(1, 16, (hv,)))
    dt_bias = mx.ones((hv,))
    q, k = gated_delta.normalize_qk(q, k, inv_scale=dk ** -0.5, eps=1e-6)

    def reference(q: Any, k: Any, v: Any) -> Any:
        beta, g = mx.sigmoid(bb), gated_delta.compute_g(a_log, a, dt_bias)
        y, _ = gated_delta._mlx_lm_lora_reference_ops(q, k, v, g, beta, None, None)
        return y

    beta, g = mx.sigmoid(bb), gated_delta.compute_g(a_log, a, dt_bias)
    state0 = mx.zeros((b, hv, dv, dk), dtype=mx.float32)

    def fast(q: Any, k: Any, v: Any) -> Any:
        return mx.fast.gated_delta_update(q, k, v, g, beta, initial_state=state0)[0]

    step = mx.checkpoint(
        lambda q, k, v, g, beta, st: mx.fast.gated_delta_update(q, k, v, g, beta,
                                                               initial_state=st))

    def chunked(q: Any, k: Any, v: Any, size: int = 32) -> Any:
        st, ys = state0, []
        for i in range(0, t, size):
            y, st = step(q[:, i:i + size], k[:, i:i + size], v[:, i:i + size],
                         g[:, i:i + size], beta[:, i:i + size], st)
            ys.append(y)
        return mx.concatenate(ys, axis=1)

    def grads(fn: Any) -> Any:
        return mx.grad(lambda q, k, v: (fn(q, k, v).astype(mx.float32) ** 2).sum(),
                       argnums=(0, 1, 2))(q, k, v)

    gr = grads(reference)
    scale = max(mx.abs(x).max().item() for x in gr)
    out: dict[str, Any] = {"max_abs_grad": scale}
    for name, fn in (("fast", fast), ("fast_chunked", chunked)):
        diff_out = mx.abs(reference(q, k, v) - fn(q, k, v)).max().item()
        diff_grad = max(mx.abs(x - y).max().item()
                        for x, y in zip(gr, grads(fn), strict=True))
        out[name] = {"max_abs_diff_output": diff_out, "max_abs_diff_grad": diff_grad,
                     "ok": diff_out < 1e-3 and diff_grad <= 1e-2 * max(scale, 1e-6)}
    return out


def train_step_report(mx: Any, nn: Any, optim: Any, model: Any, loss_fn: Any,
                      ids: list[int], start: int) -> dict[str, Any]:
    from mlx.utils import tree_flatten

    x = mx.array(ids)
    value_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.Adam(learning_rate=LR)
    rows = []
    for step in range(2):
        mx.clear_cache()
        mx.reset_peak_memory()
        before = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}
        t0 = time.perf_counter()
        loss, grads = value_and_grad(model, x, start, 1.0)
        mx.eval(loss, grads)
        t_grad = time.perf_counter() - t0
        flat = tree_flatten(grads)
        finite = all(bool(mx.isfinite(g).all().item()) for _, g in flat)
        nonzero = sum(int((g != 0).any().item()) for _, g in flat)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        wall = time.perf_counter() - t0
        after = dict(tree_flatten(model.trainable_parameters()))
        moved = max(mx.abs(after[k] - before[k]).max().item() for k in before)
        rows.append({"step": step, "loss": round(loss.item(), 4), "grad_s": round(t_grad, 2),
                     "step_s": round(wall, 2), "tokens_per_s": round(len(ids) / wall, 1),
                     "peak_memory_gib": _gib(mx.get_peak_memory()),
                     "grads_finite": finite, "grad_tensors_nonzero": nonzero,
                     "grad_tensors": len(flat), "max_param_change": moved})
    loss_after = loss_fn(model, x, start, 1.0).item()
    return {"tokens": len(ids), "answer_tokens": len(ids) - start, "steps": rows,
            "loss_after_updates": round(loss_after, 4),
            "loss_changed": abs(loss_after - rows[0]["loss"]) > 1e-4}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from huggingface_hub import snapshot_download
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.models import gated_delta
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm_lora.recurrent_patch import enable_memory_safe_recurrences

    report: dict[str, Any] = {"prompts_sha256": _sha256(args.prompts)}
    rows = [json.loads(line) for line in args.prompts.read_text().splitlines()]

    # 1. environment ---------------------------------------------------------------
    mx.set_memory_limit(MEMORY_LIMIT_GIB * 2**30)
    gated_delta._mlx_lm_lora_reference_ops = gated_delta.gated_delta_ops  # before patching
    enable_memory_safe_recurrences()
    path = Path(snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True))
    model, tokenizer = load(str(path))
    tm = _text_model(model)
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    report["environment"] = {
        "packages": {p: md.version(p) for p in ("mlx", "mlx-metal", "mlx-lm", "mlx-lm-lora",
                                                 "transformers")},
        "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
        "model_type": tm.model_type, "layers": len(tm.layers),
        "linear_attention_layers": sum(layer.is_linear for layer in tm.layers),
        "tie_word_embeddings": tm.args.tie_word_embeddings,
        "fast_gated_delta_vjp": hasattr(mx.fast, "gated_delta_update"),
        "device_memory_gib": _gib(int(info.get("memory_size", 0))),
        "max_recommended_working_set_gib":
            _gib(int(info.get("max_recommended_working_set_size", 0))),
    }

    # 2. chat template ---------------------------------------------------------------
    short = min(rows, key=lambda r: len(r["user"]))
    conts = continuations(tokenizer, short)
    report["template"] = {
        "generation_prompt_tail": tokenizer.decode(prompt_ids(tokenizer, short)[-12:]),
        "continuations": {str(lv): {"ids": c, "pieces": [tokenizer.decode([t]) for t in c]}
                          for lv, c in conts.items()},
        # "10" is "1" then "0": the unterminated "1" answer is a prefix of "10" (T2).
        "ten_is_one_then_zero":
            conts[10][:-1] == conts[1][:-1] + [tokenizer.convert_tokens_to_ids("0")],
    }
    if any(continuations(tokenizer, r) != conts for r in rows[:5]):
        raise SystemExit("continuations differ across prompts")

    # 3. token manifest ----------------------------------------------------------------
    lengths = {r["key"]: len(prompt_ids(tokenizer, r)) + max(map(len, conts.values()))
               for r in rows}
    by_group: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_group[r["split_group"]].append(r["key"])
    capped = [k for ks in by_group.values() for k in sorted(ks)[:GROUP_CAP]]
    manifest = {
        "all": _stats(list(lengths.values())),
        "allowed": _stats([lengths[r["key"]] for r in rows if r["allowed"]]),
        "caught": _stats([lengths[r["key"]] for r in rows if not r["allowed"]]),
        "capped_at_14_per_group": _stats([lengths[k] for k in capped]),
        "over_max_seq_length": sum(v > MAX_SEQ_LENGTH for v in lengths.values()),
        "note": "Token counts include the chat template and the longest answer. All open "
                "units, ambiguous included; training drops ambiguous units (labels are not "
                "read here, so they are not removed).",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "length_manifest.json").write_text(json.dumps(
        {"lengths": lengths, "summary": manifest}, indent=1) + "\n")
    report["token_manifest"] = manifest

    # 4. fast VJP vs reference ---------------------------------------------------------
    report["fast_vjp_check"] = check_fast_vjp(mx)

    # 5–6. training path -----------------------------------------------------------------
    model.freeze()
    linear_to_lora_layers(tm, len(tm.layers), LORA)
    grad_checkpoint(tm.layers[0])
    keys = sorted({re.sub(r"^.*layers\.\d+\.", "", k.rsplit(".", 1)[0])
                   for k, _ in tree_flatten(model.trainable_parameters())})
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    report["lora"] = {**LORA, "layers": len(tm.layers), "target_modules": keys,
                      "trainable_parameters": n_train, "lr": LR,
                      "max_seq_length": MAX_SEQ_LENGTH, "checkpointing": "per decoder layer",
                      "recurrence": "mlx-lm-lora enable_memory_safe_recurrences"}
    loss_fn = make_loss(mx, nn)
    target = conts[DUMMY_TARGET]

    model.train()
    ids, start = build_example(prompt_ids(tokenizer, short), target)
    if ids[start:] != target or ids[-1] != tokenizer.convert_tokens_to_ids("<|im_end|>"):
        raise SystemExit("answer or end token missing from the loss span")
    x = mx.array(ids)
    vg = nn.value_and_grad(model, loss_fn)
    l1, g1 = vg(model, x, start, 1.0)
    l5, _ = vg(model, x, start, 0.5)
    l0, g0 = vg(model, x, start, 0.0)
    mx.eval(l1, l5, l0, g1, g0)
    try:
        build_example([0] * MAX_SEQ_LENGTH, target)
        refuses = False
    except ValueError:
        refuses = True
    report["loss_checks"] = {
        "example_tokens": len(ids), "answer_tokens": len(target),
        "answer_and_end_in_loss_span": True,
        "loss_w1": round(l1.item(), 5), "loss_w05": round(l5.item(), 5),
        "scales_with_weight": abs(l5.item() - 0.5 * l1.item()) < 1e-4,
        "zero_weight_zero_grad": all(not bool((g != 0).any().item())
                                     for _, g in tree_flatten(g0)),
        "refuses_overlong_example": refuses,
    }

    # 7. scoring (before any update: LoRA B starts at zero, so this is the base model) -----
    model.eval()
    toy_ids = prompt_ids(tokenizer, short)
    lp = candidate_logprobs(mx, model, toy_ids, conts)
    unterminated = candidate_logprobs(mx, model, toy_ids,
                                      {lv: c[:-1] for lv, c in conts.items()})
    agreement = {}
    for lv in (1, 5, 10):  # cached branch scoring vs one plain full forward
        seq = mx.array(toy_ids + conts[lv])
        lg = model(seq[None]).astype(mx.float32)[0]
        lps = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        full = mx.take_along_axis(lps[len(toy_ids) - 1:-1],
                                  seq[len(toy_ids):][:, None], axis=-1).sum().item()
        agreement[str(lv)] = {"cached": round(lp[lv], 4), "full_forward": round(full, 4)}
    report["scoring_toy"] = {
        **expected_level(lp),
        "terminated_logprob_1": round(lp[1], 4), "terminated_logprob_10": round(lp[10], 4),
        "unterminated_logprob_1": round(unterminated[1], 4),
        "unterminated_logprob_10": round(unterminated[10], 4),
        "cached_vs_full_forward": agreement,
        "reading": "Without the end token, '1' is a prefix of '10' and can never score "
                   "below it; with it, the two are disjoint answers.",
    }
    longest = max(rows, key=lambda r: lengths[r["key"]])
    t0 = time.perf_counter()
    mx.reset_peak_memory()
    lp_long = candidate_logprobs(mx, model, prompt_ids(tokenizer, longest), conts)
    report["scoring_long"] = {"tokens": lengths[longest["key"]],
                              "seconds": round(time.perf_counter() - t0, 2),
                              "peak_memory_gib": _gib(mx.get_peak_memory()),
                              **{k: v for k, v in expected_level(lp_long).items()
                                 if k != "probs"}}
    (args.out / "feasibility.json").write_text(json.dumps(report, indent=2) + "\n")

    model.train()
    if not args.skip_train:
        by_len = sorted(rows, key=lambda r: lengths[r["key"]])
        report["train_steps"] = {}
        for target_len in TARGET_LENGTHS:
            row = min(by_len, key=lambda r: abs(lengths[r["key"]] - target_len))
            ids, start = build_example(prompt_ids(tokenizer, row), target)
            try:
                report["train_steps"][str(target_len)] = train_step_report(
                    mx, nn, optim, model, loss_fn, ids, start)
            except Exception as exc:  # record an OOM or kernel failure, keep going
                report["train_steps"][str(target_len)] = {
                    "tokens": len(ids), "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                mx.clear_cache()
            (args.out / "feasibility.json").write_text(json.dumps(report, indent=2) + "\n")

    (args.out / "feasibility.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "template"}, indent=2))


if __name__ == "__main__":
    main()

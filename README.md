# EntitlementShield

> **Status: closed (October 2026).** No further work is planned in this repository. The
> findings, including the final follow-ups, are in [`docs/FINDINGS.md`](docs/FINDINGS.md).

A learning project: an abuse detector for a free-credit program, built and evaluated
end to end on **synthetic data**, with an LLM added as a second-stage reviewer.

StudioAI is a fictional AI product. It gives each person 25 free generation credits per
30-day cycle. Abuse means running several accounts to collect extra allocations.
Households, schools and offices legitimately share devices and networks, so "many
accounts on one device" is not abuse by itself. The detector has to judge whether a
cluster uses more credits than the number of real people behind it plausibly justifies.

Everything here is synthetic. No real users, no production data, and nothing from any
employer. The numbers describe generated worlds and are not production metrics.

## What's in it

```
accounts + events + shared signals
        │
        ▼
 Stage 1: frozen detector        clusters accounts on shared resources, scores each
 (features → score → policy)     cluster-cycle, proposes allow / friction / hold
        │ allowed units
        ▼
 Stage 2: LLM recall layer       reviews only what Stage 1 allowed, returns RISK_LEVEL 1-10,
 (pilot, local Qwen 9B)          may only ADD a flag, never clear one
```

The Stage 2 shape follows Coinbase's
[Owning Intelligence series](https://www.coinbase.com/blog/part-one-how-coinbase-built-evaluated-and-post-trained-a-fraud-agent):
the cheap first stage screens everything, and the LLM is a bounded reviewer of what it lets through.
One difference: Coinbase routes only a selected subset to its agent, while the pilot here sent
every allowed unit.

- **Detector** (`backend/app/services/`): clustering, 12 features in 5 evidence families,
  capacity banding, an action ladder with no automatic permanent action, and an immutable
  audit trail. The firing configuration is frozen behind a content-hashed manifest
  (`features-frozen-v2`), and runtime entry points refuse to run on a stale one.
- **Evaluation harness** (`backend/app/services/evaluation.py`): fixed evaluation units,
  group-disjoint train/tune/test splits, paired McNemar and cluster bootstrap statistics.
- **API and console**: FastAPI with a small Jinja/HTMX analyst console (`make serve`).

## Evaluation discipline

The part I cared most about getting right:

- **Labels live in a physically separate store.** Detector code never gets a handle to it,
  and tests scan the source to enforce that.
- **The sealed test split is one-shot.** Reading it requires a finalization grant, which is
  recorded in an append-only ledger and cannot be spent twice.
- **Confirmatory experiments are pre-registered** before scoring: population, metric,
  statistics, claim rule. Exploratory work, such as the Stage 2 pilot below, is labelled
  exploratory and is not a confirmatory result.
- **Test on worlds you didn't write.** The dev world comes from my own generator. Later
  worlds were written by other models from a brief, without seeing the code.

## Findings

Details and caveats: [`docs/FINDINGS.md`](docs/FINDINGS.md).

Results are counted in **units**: one linked group of accounts in one 30-day credit cycle.

- On the dev world, which I generated and tuned on, the detector reached recall 1.0 on the
  tuning split.
- On Tier 4, written by GPT-6 Codex from a brief without access to the repo, the same frozen
  detector reached unit recall **0.70** and flagged **61%** of legitimate units (open
  train+tune scope, [`docs/p6.0-recall-headroom.md`](docs/p6.0-recall-headroom.md)). That is
  only slightly better than random: precision 0.18 against a 0.16 base rate. The gap mixes
  in-sample tuning, a different author, and a brief that asked for legitimate groups that look
  abusive and abuse that looks legitimate. It is why you test on data you didn't write, not a
  measure of authorship alone.
- Stage 2, an exploratory pilot: Qwen 3.5 9B, prompted and not fine-tuned, on the units the
  detector allowed. It flagged legitimate units more often than abusive ones (67% vs 45% at
  8/10 or higher; AUC 0.38). It adds real coverage, but flagging the same number of units at
  random would catch more abuse on average (about 39 vs 27). The 60 missed abusive units come
  from only 4 source groups in the generated world, so this is a diagnostic result, not a
  general claim
  ([`docs/p6.1-pilot-results.md`](docs/p6.1-pilot-results.md)).
- Follow-ups: a post-trained 9B decision model (Bespoke Nimble) was also near chance. A
  supervised model on 64 label-free features reached only AUC 0.672, so the labels are only
  weakly recoverable from this evidence
  ([`docs/p6.4-learnability-ceiling.md`](docs/p6.4-learnability-ceiling.md)). The detector's
  main weakness is false positives and fragmented linkage in Stage 1, which a per-unit recall
  layer can't fix.

## Running it

Python 3.12+. The commands below are usage instructions; they do not grant
permission to execute them. They work with the synthetic development world and
do not report scientific results. Tier 4 examples need separate explicit scope;
the sealed `tier4-v2` test split remains unspent.

### Setup

```sh
make install
```

This creates `.venv`, upgrades pip, and installs the project with development
dependencies. It uses the network and performs package installation.

### Local checks

Run these from an existing `.venv`:

```sh
make test
make lint
make typecheck
```

These invoke pytest, Ruff, and mypy. Check their expected effects before running;
this documentation does not assert they are side-effect-free or currently passing.

### Generate and evaluate a development world

```sh
make seed
make detector
make eval
```

`make seed` generates and writes synthetic development data. The detector and
open-split evaluation use the seeded development stores. These commands may
write data or reports; they are not part of setup.

### Serve the local console

```sh
make serve
```

This starts the API and analyst console locally at <http://localhost:8000/>. Seed
the development world first with `make seed`; stop the server with Ctrl-C.

### Recall-layer pilot example

This example needs a local OpenAI-compatible server (the pilot used MTPLX with
Qwen 3.5 9B) and a Tier 4 world. Tier 4 use requires separate explicit scope,
and the sealed `tier4-v2` test split must remain unspent:

```sh
PYTHONPATH=backend .venv/bin/python scripts/p6_recall_pilot.py run \
    --config config/recall-provider.local-qwen35-9b.yaml --out <run-dir>
PYTHONPATH=backend .venv/bin/python scripts/p6_recall_pilot.py score --out <run-dir>
```

## How it was built

Built with AI coding agents (Claude Code, Codex) under a written workflow:
small batches, a plan before each step, pre-registration for confirmatory experiments,
explicit records for exploratory ones, and cross-model adversarial reviews of substantial
changes.

# Findings

One page of what this project found, with sources. Everything is synthetic, from
generated worlds of a few hundred to a few thousand accounts. Treat the numbers as
lessons about method, not as production metrics.

The setting: StudioAI, a fictional AI product, gives each person 25 free credits per 30-day
cycle. Running several accounts to collect extra allocations is abuse. Households, schools and
offices legitimately share devices, networks and cards, so sharing alone is not abuse.

Two terms matter below:
- A **cluster** is a group of accounts the detector links through shared resources. A **unit**
  is one cluster in one 30-day credit cycle (the fixed evaluation unit, E-14). The same cluster
  contributes one unit per cycle, so units are not independent people or accounts.
- A **source group** is a group as the world's author generated it (its split group). One
  source group can show up as several detector clusters. Source groups are the honest unit of
  independent evidence, and the scripts' `*_clusters` fields count source groups.

## 1. Test on worlds you didn't write

| World | Who wrote it | Detector result (frozen config, threshold 0.60) |
|---|---|---|
| Dev | My own generator, also used for tuning | Tuning split: precision 1.0, recall 1.0, false-positive rate 0.0 (in-sample) |
| Tier 4 | GPT-6 Codex, from a written brief, with no access to the repo, features, weights or thresholds | Open train+tune scope: recall **0.70** (142/202 abusive units), **643 of 1,051** legitimate units flagged, precision 0.18 |

The detector looked perfect on the tuning split of the world it was built against. On a
world written by a different model it missed 30% of abusive units and flagged 61% of
legitimate units.

On Tier 4 it is only slightly better than flagging at random: its precision is 0.18 against a
0.16 base rate, and a random pick of the same 785 units would catch about 127 abusive units
on average, against its 142 (balanced accuracy 0.55).

That gap mixes three things:
- in-sample tuning vs a new world;
- a different author;
- a brief that deliberately asked for both "legitimate groups that look like abuse at a
  glance, and abuse that looks legitimate", with no single signal giving the label away. An
  audit of the generated world found no single threshold above 0.67 balanced accuracy; it is
  a hard world by design.

It is a reason to test on data you didn't write, not a measure of authorship alone.

Caveats:
- The 60 Tier 4 misses come from only 4 source groups, and two of them hold 53, so the
  independent evidence is a handful of cases. Those 60 units span at least 9 detector
  clusters.

## 2. LLM as a second recall layer (pilot)

The design follows Coinbase's Onramp agent
([Owning Intelligence series](https://www.coinbase.com/blog/part-one-how-coinbase-built-evaluated-and-post-trained-a-fraud-agent)):
- the detector screens everything;
- the LLM reviews units the detector allowed, and returns `RISK_LEVEL: 1-10`;
- application code decides, and the LLM can only add a flag.

One difference matters: Coinbase routes only a selected subset of allowed transactions to its
agent. This pilot sent **every** allowed unit, a pool that is 87% legitimate.

Pilot setup:
- Model: local Qwen 3.5 9B (`mtplx-qwen35-9b-optimized-speed`), prompted and not fine-tuned,
  temperature 0, one sample.
- Population: all 492 Tier 4 train+tune units with a detector score below 0.60.
- Prompt: a compressed history plus per-account summaries. It excludes the detector's score
  and appeals.
- Runner: `scripts/p6_recall_pilot.py`.

**Result: on this sample the model ranked abuse below legitimate use, and caught fewer
abusive units than a random pick of the same size would on average**
([`docs/p6.1-pilot-results.md`](p6.1-pilot-results.md)).

| Allowed units (60 abusive from 4 source groups, 408 legitimate; 24 ambiguous excluded) | Value |
|---|---|
| AUC, risk level, abusive vs legitimate (point estimate, no interval) | **0.38** |
| Legitimate allowed units rated 8 or higher | 275 of 408 (67%) |
| Abusive allowed units rated 8 or higher | 27 of 60 (45%) |
| Staged-household misses (2 source groups, 7 units) | all 7 rated 8 to 10 |
| Direct-allocation-cycling misses (2 source groups, 53 units) | 20 rated 8 to 10, 33 rated 1 to 2 |

If the layer flagged everything rated 8 or higher, it would flag 322 allowed units: 27
abusive, 275 legitimate and 20 ambiguous. Unit recall would rise from 0.70 to 0.84 (with all
4 missed source groups represented), and the false-positive rate would rise from 0.61 to 0.87.

That is a trade-off, and a dominated one:
- the 27 catches are real coverage;
- but the model flagged legitimate units more often than abusive ones (67% vs 45%);
- among the 468 labelled allowed units (12.8% abusive), a random pick of the same 302 would
  catch about 39 abusive units on average. Counting the ambiguous units, a random 322 of all
  492 gives about the same: 39;
- of the model's 302 labelled flags, 9% are abuse, below the 12.8% base rate.

This comparison needs no cost model. Whether any version of this trade-off would be
acceptable in a deployed system was not measured. The whole stack is below random at matched
volume too: detector plus layer catch 169 abusive units, where a random pick of the same
1,087 labelled units would catch about 175.

Coinbase also reports that their unadapted Qwen3.5-9B performed poorly on their own fraud
benchmark, and that it improved after post-training against known historical outcomes. This
pilot is a different, synthetic task and is not a replication of theirs.

Scope and provenance:
- exploratory, open-scope development: one prompt, one model, one generated world;
- the model's inputs are label-free, but the prompt was written after the P6.0 headroom
  check had already read open-scope labels;
- the 60 abusive units come from 4 source groups (53 from 2), so the AUC is a descriptive
  point estimate, not a stable property of the model;
- this is not a confirmatory test. That needs a frozen, pre-registered configuration run once
  on a freshly authored world with many more independent source groups.

## What would change these conclusions, and the next step

- **Next step:** post-train a small open model on synthetic outcomes, then check it once, frozen,
  on a fresh and larger repo-blind world with many independent missed source groups.
- Real data, which this project does not use and will not use.

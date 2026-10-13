"""Response transcript / replay cache for a real LLM provider (Phase 4 plan P4.6,
`docs/decisions.md §13` M-6).

A real provider at temperature 0 is not byte-reproducible the way every other
artifact in this repo is. This module persists a JSONL transcript of every
`complete()` call so a run is replayable from committed data even though the
live call wasn't deterministic.

**Scoping refinement to M-6 (folded back into `docs/decisions.md §13` alongside
this commit):** M-6 specifies the cache key as
`(idempotency_key, arm, sample_index, reviewer_manifest_hash, provider_id,
model_snapshot)`. That tuple is known by the reviewer orchestration
(`reviewer.py`'s `_run_sample`), not by the `LLMProvider.complete(request)` call
site itself — the Protocol's signature is fixed at `complete(request:
LLMRequest) -> LLMResponse`, with no side channel for caller context, and
`reviewer.py` is out of scope for this step (P4.4-6 build the provider adapters
and the config/factory/cache plumbing; wiring the cache into the sampling loop
itself is left for whichever future step actually needs replay against a
committed transcript, e.g. a real P4.8/P4.9 run).

So `CachingProvider` here derives its own key from what IS available at
`complete(request)` time: a hash of the request's **decision-relevant** content
(`system` + every non-`UntrustedBlock` segment's text — the untrusted block's
delimiter is CSPRNG-randomized per request per C3 and deliberately excluded,
matching `fake_reviewer.FakeReviewerProvider`'s own "keys only on the structured
payload" design — and the `decision_id: <uuid4>` line `prompt_builder` embeds in
the structured evidence is likewise excluded, since it is freshly assigned per
reseed and not case content, verified live via `scripts/arm4_run.py --replay`
against a same-seed rerun) plus `temperature`/`max_tokens`, combined with a
per-fingerprint call counter standing in for `sample_index` (this repo's reviewer
orchestration already relies on call *order* being stable for a fixed world
seed/config — the same property idempotency keys depend on). This reproduces
M-6's intent (a replayable transcript keyed on logical case identity, not raw
prompt bytes) without requiring `reviewer.py` changes in this step. `provider_id`/
`model_snapshot` are captured from the wrapped provider's own response type name
and returned `model_snapshot`, not passed in separately.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.services.llm_provider import LLMProvider, LLMRequest, LLMResponse, TokenUsage
from app.services.prompt_builder import UntrustedBlock

# `prompt_builder._format_structured_evidence` embeds `decision_id: <uuid4>` as a
# literal line in `StructuredEvidenceSegment.text` — that id is freshly assigned by
# the detector on every reseed (verified live: two `arm4_run.py` invocations at the
# same world seed produced different decision ids for the same logical case), not
# a property of the case's substantive evidence. Left in, an otherwise-identical
# case would fingerprint differently across separate script runs, making
# cross-invocation replay (the entire point of this cache, M-6) never hit. Stripped
# before hashing, for the same reason the untrusted block's CSPRNG delimiter
# already is.
_DECISION_ID_LINE = re.compile(r"^decision_id: .*$\n?", re.MULTILINE)


def request_fingerprint(request: LLMRequest) -> str:
    """A stable hash of the decision-relevant parts of a request — excludes the
    `UntrustedBlock` wrapper (delimiter randomized per request, C3) and the
    `decision_id` evidence line (freshly uuid4'd per reseed, not case content);
    either left in would make an identical logical case hash differently every
    call/run. A non-empty appeal body IS hashed (P5-7): a provider that reads the
    appeal must never replay an answer recorded for a different appeal. Empty
    appeals add nothing, so appeal-free keys are unchanged."""
    stable_parts = [request.system]
    for segment in request.segments:
        if not isinstance(segment, UntrustedBlock):
            stable_parts.append(segment.text)
        elif segment.payload:
            stable_parts.append(f"appeal={segment.payload}")
    stable_parts.append(f"temperature={request.temperature}")
    stable_parts.append(f"max_tokens={request.max_tokens}")
    raw = "\n".join(stable_parts)
    normalized = _DECISION_ID_LINE.sub("", raw)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class ResponseCache:
    """A JSONL-backed transcript: one line per `(key, response)` pair. `get`/`put`
    operate on an in-memory index; `load`/`save` persist it. Callers construct one
    per run, `load()` it (a fresh path is just an empty cache), and `save()` it
    when done."""

    path: Path
    _entries: dict[str, dict] = field(default_factory=dict, repr=False)

    def load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self._entries[row["key"]] = row["response"]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            for key, response in self._entries.items():
                f.write(json.dumps({"key": key, "response": response}) + "\n")

    def get(self, key: str) -> LLMResponse | None:
        row = self._entries.get(key)
        if row is None:
            return None
        usage = TokenUsage(**row["usage"])
        return LLMResponse(
            text=row["text"],
            model_snapshot=row["model_snapshot"],
            latency_ms=row["latency_ms"],
            usage=usage,
        )

    def put(self, key: str, response: LLMResponse) -> None:
        self._entries[key] = {**asdict(response), "usage": asdict(response.usage)}


@dataclass
class CachingProvider:
    """Wraps a real `LLMProvider`, replaying a cached response when the request's
    fingerprint has been seen before (up to `call_index`-many times — the Nth call
    with a given fingerprint replays the Nth cached response for that
    fingerprint, standing in for `sample_index`) and recording every live call
    into `cache` otherwise. Structurally satisfies `LLMProvider` itself, so it can
    be used anywhere a plain provider is."""

    inner: LLMProvider
    cache: ResponseCache
    replay_only: bool = False
    _call_counts: dict[str, int] = field(default_factory=dict, repr=False)

    def complete(self, request: LLMRequest) -> LLMResponse:
        fp = request_fingerprint(request)
        index = self._call_counts.get(fp, 0)
        self._call_counts[fp] = index + 1
        key = f"{fp}:{index}"

        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if self.replay_only:
            raise KeyError(f"no cached response for key {key!r} and replay_only=True")

        response = self.inner.complete(request)
        self.cache.put(key, response)
        return response


__all__ = ["CachingProvider", "ResponseCache", "request_fingerprint"]

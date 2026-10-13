"""The reviewer's version-observability freeze artifact (docs/decisions.md §12 L-8).

L-8 requires a `config/reviewer-manifest.json` that hashes "the prompt template
bytes (both variants), output JSON-Schema bytes, provider id, model snapshot,
temperature, sample count, auto-apply thresholds, the gating rule, and the fake
reviewer's rule table", and whose hash "stamps every `LlmReview` row, every
reviewer audit event, and every arm-4 artifact".

It is **a separate artifact from `config/feature-manifest.json`, which must not
change** — L-8 says so explicitly, and adding this path is a signed-off, narrow
carve-out of the scope-creep tripwire. The two manifests are therefore declared
independent: this module deliberately *duplicates* `feature_manifest.py`'s
`_canonical`/`content_hash`/`build_manifest`/`manifest_bytes`/`load_committed`/
`verified_manifest_hash`/`write_manifest` shapes rather than importing them, so a
refactor of one can never silently re-freeze the other. `MANIFEST_VERSION` is its
own namespace (`reviewer-manifest-v1`, not `features-frozen-vN`) and
`ReviewerManifestDrift` is its own exception.

Structure mirrors `feature_manifest.py`'s three layers, adapted to a stack whose
behavior lives mostly in prompts and code rather than in numeric constants:

1. **Byte hashes of the artifacts the model literally sees** — both prompt
   templates and the output JSON Schema. These are read as bytes, so a one-word
   edit to an instruction re-freezes; nothing about the templates is restated in
   Python where it could drift.
2. **Live-read config** — provider id and model snapshot are read from whichever
   provider is `active` in `config/reviewer-provider.yaml` (Phase 4 M-7: a
   provider switch is expected to change this manifest's hash, not a scope-creep
   signal), via `providers.factory.describe_provider` — no client construction, no
   API key needed, so regenerating this manifest never depends on a live key
   being provisioned. Temperature, sample count, the auto-apply ceiling, the
   repair-retry budget, and the fake reviewer's rule table (`_ACTION_LADDER`,
   `_DISAGREE_VARIANT_OFFSETS`) are resolved from the live constants at build
   time, never snapshotted here.
3. **`reviewer_code_fingerprint`** — SHA-256 over the full source bytes of the
   four modules that *are* the reviewer's executable semantics
   (`reviewer.py`, `reviewer_validation.py`, `escalation.py`, `fake_reviewer.py`).
   This is the belt to layer 2's braces: the unanimity rule, the citation filter,
   the severity clamp, the blast-radius guard and the escalation lifecycle are all
   code, not constants, and none of them would move a config value if changed.

`review_case.py` is deliberately **excluded** from the fingerprint. It is shared
P3.1 case-building infrastructure (appeal storage, timeline bucketing, the
evidence projection) read by the console and the API as well as the reviewer; a
change there is not a change to the reviewer's declared identity and must not
re-freeze this manifest. The masking decision that *does* belong to the reviewer —
which view each arm runs — is captured through `gating_rule` and through
`reviewer.py`'s own fingerprinted source instead.

**Scope of the fail-closed gate.** `verified_manifest_hash()` is called only from
`scripts/arm4_run.py` (dev tier). It is deliberately *not* wired into
`reviewer.review_decision`, which would extend the gate to P3.7's `/reviewer/run`
endpoint and console panel — a live-surface behavior change outside P3.9's scope.
Row-level stamping uses the plain `MANIFEST_HASH` constant, so an uncommitted
working tree records an honest hash rather than refusing to run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.models.enums import ProposedAction
from app.providers.factory import describe_provider, load_provider_config
from app.services import escalation as E
from app.services import fake_reviewer as FR
from app.services import reviewer as R
from app.services import reviewer_validation as RV

#: Bumped only when the reviewer's declared contract changes shape (a new prompt
#: variant set, a new output schema version) — its own namespace, never shared
#: with `feature_manifest.MANIFEST_VERSION`.
MANIFEST_VERSION = "reviewer-manifest-v1"

#: The committed artifact. Resolved the same `parents[3]` way `feature_manifest`
#: does: backend/app/services/reviewer_manifest.py -> repo root.
MANIFEST_PATH = Path(__file__).resolve().parents[3] / "config" / "reviewer-manifest.json"

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

#: The prompt templates whose literal bytes are hashed, keyed by the
#: `prompt_builder.PromptVariant` name each is loaded under.
_PROMPT_TEMPLATES: dict[str, str] = {
    "reviewer_v1": "reviewer_v1.md",
    "reviewer_timeline_only_v1": "reviewer_timeline_only_v1.md",
}

_OUTPUT_SCHEMA_FILE = "output_schema_v1.json"

#: Fingerprinted modules, in fixed order. `review_case.py` is excluded on purpose
#: (see the module docstring).
_REVIEWER_MODULES = (R, RV, E, FR)


class ReviewerManifestDrift(RuntimeError):
    """The committed reviewer manifest is missing, malformed, self-inconsistent, or
    does not match regeneration from the live code. Separate from
    `feature_manifest.ManifestDrift` so a caller can never catch one and
    accidentally swallow the other."""


def _file_digest(path: Path) -> dict[str, Any]:
    """SHA-256 + byte length of a file read as raw bytes. Length is recorded
    alongside the digest purely as a human-legible sanity signal in the artifact;
    the digest alone is what makes the entry tamper-evident."""
    if not path.is_file():
        raise ReviewerManifestDrift(f"cannot hash missing reviewer artifact {path}")
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _prompt_templates() -> dict[str, Any]:
    return {
        variant: _file_digest(PROMPTS_DIR / filename)
        for variant, filename in sorted(_PROMPT_TEMPLATES.items())
    }


def _gating_rule() -> dict[str, Any]:
    """L-9's invocation gate plus the L-3/C-5/C-2/R-7 decision rules, as a
    structured dict.

    This block is hand-maintained prose-as-data: it states the rules in a form a
    reader (or a later reviewer-vs-reviewer diff) can compare, while the
    executable versions are locked independently by `reviewer_code_fingerprint`.
    Only `max_repair_retries` has a live constant to read; everything else is
    control flow, so a change to it moves the fingerprint rather than this block.
    """
    return {
        "invokes_on": "any decision with proposed_action != allow",
        "skip_conditions": [
            "proposed_action == allow",
            "status in {reversed, superseded} (terminal)",
            "status == under_review with an open EscalationRecord",
            "status == under_review without one (already_under_review)",
            "evidence signature is under a live SuppressionRecord",
        ],
        "samples_per_decision": R.SAMPLE_COUNT,
        "combination_rule": (
            "strict unanimity (C-5): all samples must derive the same effect; any "
            "disagreement escalates, regardless of majority"
        ),
        "citation_rule": (
            "C-2: citations are kept only if they resolve against the case's "
            "evidence_keys allowlist; unresolved citations move to "
            "unverified_claims and a verdict with zero valid citations can never "
            "reverse"
        ),
        "severity_clamp_rule": (
            "C-4: the model's recommended_action is clamped to "
            "min(model, policy_proposed_action) over policy_engine.ACTION_SEVERITY"
        ),
        "escalate_rule": (
            "L-5: an escalate outcome is a real state transition (under_review + an "
            "EscalationRecord with a 7-day window), not a label on the LlmReview rows"
        ),
        "cluster_blast_radius_rule": (
            "L-3/RC-1 P1-1: auto-apply is blocked whenever another currently-actioned "
            "decision shares this decision's cluster_snapshot_id AND its exact "
            "policy_engine.evidence_signature; both signatures are derived from the raw "
            "persisted decision_evidence payload, never from the ReviewCase projection"
        ),
        "max_repair_retries": RV.MAX_RETRIES,
        "repair_retry_rule": (
            "R-7: exactly one repair retry per sample before failing closed to "
            "escalate; every attempt is persisted as its own LlmReview row"
        ),
        "arm_views": {
            arm: {"view": view, "prompt_variant": variant, "prompt_version": version}
            for arm, (view, variant, version) in sorted(R._ARM_CONFIG.items())
        },
    }


def _auto_apply_thresholds() -> dict[str, Any]:
    """L-3's four auto-apply conditions, with the R-6 severity ceiling read live
    off the policy ladder rather than restated as a string."""
    return {
        "ceiling_action": ProposedAction.LIGHT_FRICTION.value,
        "ceiling_rule": (
            "R-6: the decision's ORIGINAL proposed_action severity must itself be at "
            "or below light_friction — not merely the model's clamped recommendation, "
            "which derive_effect already bounds there"
        ),
        "requires_unanimous_reverse": True,
        "requires_no_other_actioned_cluster_member": True,
        "actor": "llm",
    }


def _fake_reviewer_rule_table() -> dict[str, Any]:
    """L-8's "fake reviewer's rule table". Always hashed regardless of which
    provider `config/reviewer-provider.yaml` currently has `active` (Phase 4 added
    real adapters, `providers/`, alongside it): `FakeReviewerProvider` remains this
    repo's deterministic reference provider (dev-tier scoring, tests), so its
    verdict-selection mechanism stays part of the reviewer's reproducible
    identity, not a test detail — independent of which provider a given run's
    `provider` block names."""
    return {
        "action_ladder": [action.value for action in FR._ACTION_LADDER],
        "disagree_variant_offsets": list(FR._DISAGREE_VARIANT_OFFSETS),
        "seed_mechanism": (
            "int(sha256(structured_evidence_segment_text)[:16], 16) — stable across "
            "processes and PYTHONHASHSEED, keyed on case identity because the "
            "structured payload embeds decision_id. The untrusted appeal block is "
            "located only to be skipped and never reaches the rule table."
        ),
        "bucket_rule": "(seed + variant) % len(action_ladder)",
        "citation_mechanism": (
            "cites evidence_keys[0] verbatim; raises ValueError rather than "
            "fabricating a fallback citation when the allowlist is empty"
        ),
        "confidence": 0.9,
        "escalate_to_human": False,
    }


def _reviewer_code_fingerprint() -> str:
    """SHA-256 over the full source of the reviewer modules, `\\0`-separated in a
    fixed order. Over-conservative on purpose — a comment edit in `reviewer.py`
    also re-freezes — because for a v1 that is the fail-closed trade: no semantic
    change to unanimity, citation filtering, clamping, the blast-radius guard, or
    the escalation lifecycle can regenerate byte-identically."""
    h = hashlib.sha256()
    for mod in _REVIEWER_MODULES:
        src = mod.__file__
        if src is None:  # pragma: no cover - reviewer modules are always file-backed
            raise ReviewerManifestDrift(f"cannot fingerprint {mod.__name__}: no source file")
        h.update(Path(src).read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _content() -> dict[str, Any]:
    """The manifest content the hash is taken over — everything except the hash."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "prompt_templates": _prompt_templates(),
        "output_schema": _file_digest(PROMPTS_DIR / _OUTPUT_SCHEMA_FILE),
        "provider": describe_provider(load_provider_config()),
        "sampling": {
            "temperature": R.REVIEWER_TEMPERATURE,
            "sample_count": R.SAMPLE_COUNT,
        },
        "gating_rule": _gating_rule(),
        "auto_apply_thresholds": _auto_apply_thresholds(),
        "fake_reviewer_rule_table": _fake_reviewer_rule_table(),
        "reviewer_code_fingerprint": _reviewer_code_fingerprint(),
    }


def _canonical(obj: Any) -> bytes:
    """Deterministic bytes for hashing — compact, sorted keys."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def content_hash(content: dict[str, Any] | None = None) -> str:
    return hashlib.sha256(_canonical(content if content is not None else _content())).hexdigest()


def build_manifest() -> dict[str, Any]:
    """The full committed manifest: content plus its own content hash."""
    content = _content()
    return {**content, "content_hash": content_hash(content)}


def manifest_bytes() -> bytes:
    """Byte-canonical serialization written to `config/reviewer-manifest.json`
    (pretty, sorted, trailing newline) — what `make reviewer-manifest-check`
    compares byte-for-byte."""
    return (json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n").encode()


#: The reviewer freeze hash, stamped onto every `LlmReview` row, every
#: `llm_review_recorded` audit event, and every arm-4 artifact (L-8).
MANIFEST_HASH: str = content_hash()


def load_committed() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text())


def verified_manifest_hash() -> str:
    """Return the frozen hash ONLY after proving the committed
    `config/reviewer-manifest.json` (a) exists, (b) is byte-identical to
    regeneration from the live code, and (c) carries a self-consistent
    `content_hash`. Raises `ReviewerManifestDrift` otherwise.

    Called from `scripts/arm4_run.py` so an arm-4 artifact can never claim a
    reviewer version whose hash names no committed artifact. Not called from
    `reviewer.review_decision` — see the module docstring on scope."""
    if not MANIFEST_PATH.is_file():
        raise ReviewerManifestDrift(f"no committed reviewer manifest at {MANIFEST_PATH}")
    committed_bytes = MANIFEST_PATH.read_bytes()
    if committed_bytes != manifest_bytes():
        raise ReviewerManifestDrift(
            "committed reviewer-manifest.json does not match regeneration from live "
            "code (run `make reviewer-manifest` after an intended reviewer change)"
        )
    try:
        committed = json.loads(committed_bytes)
        embedded = committed.pop("content_hash")
    except (ValueError, KeyError) as exc:
        raise ReviewerManifestDrift(f"committed reviewer manifest is malformed: {exc}") from exc
    if content_hash(committed) != embedded or embedded != MANIFEST_HASH:
        raise ReviewerManifestDrift(
            "committed reviewer manifest content_hash is self-inconsistent"
        )
    return MANIFEST_HASH


def write_manifest(path: Path = MANIFEST_PATH) -> str:
    """(Re)write the reviewer manifest file; returns its content hash."""
    path.write_text(manifest_bytes().decode())
    return MANIFEST_HASH

"""P4.9 exploratory tooling: dump every unique reviewer prompt for
`arm4_reviewer`/`arm4b_timeline_only` from a freshly seeded `dev_arm4` world,
without calling any real provider.

This feeds `scripts/arm4_subagent_assemble.py`, which routes the dumped prompts
through Claude Code subagents (instead of a metered `anthropic_provider.py` call)
and reassembles the verdicts into a `ResponseCache`-format transcript that
`scripts/arm4_run.py --replay` can score with the exact same code that scored the
MTPLX run. See `docs/p4.8-arm4-real-provider-findings.md` for why this comparison
is being run, and this script's companion doc `docs/p4.9-arm4-subagent-haiku-findings.md`
for the eventual write-up.

Not itself a scored artifact (W-1): like `arm4_run.py`, this never touches the
sealed Tier-2 path.

Usage:
    python scripts/arm4_subagent_extract.py [--seed N] [--preset dev_arm4] [--as-of ISO]
        [--out-dir data/synthetic/arm4-subagent]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from arm4_run import ARM4_DEFAULT_PRESET, REPO_ROOT, seed_world  # noqa: E402

from app.db.engine import make_inference_engine  # noqa: E402
from app.models.enums import DecisionStatus, ProposedAction  # noqa: E402
from app.models.inference import Decision  # noqa: E402
from app.providers.cache import request_fingerprint  # noqa: E402
from app.schemas.reviewer import ReviewerVerdict  # noqa: E402
from app.services.detector import run_detector  # noqa: E402
from app.services.fake_reviewer import (  # noqa: E402
    _extract_evidence_keys,
    _structured_payload_text,
)
from app.services.generator import WORLD_END  # noqa: E402
from app.services.llm_provider import LLMRequest, LLMResponse, TokenUsage  # noqa: E402
from app.services.prompt_builder import (  # noqa: E402
    InstructionSegment,
    StructuredEvidenceSegment,
    SystemSegment,
)
from app.services.reviewer import review_cluster_batch  # noqa: E402

ARMS = ("arm4_reviewer", "arm4b_timeline_only")

# Always `escalate_to_human=True` with no citations, so `derive_effect`
# (reviewer_validation.py) always resolves this placeholder to "escalate" —
# the cheapest, least state-changing outcome. This pass never commits (see
# `extract_arm` below), so the actual effect chosen is immaterial; forcing
# "escalate" just avoids exercising the auto-reversal side-effect paths for no
# reason during extraction.
_PLACEHOLDER_VERDICT = ReviewerVerdict(
    evidence_supporting_action=(),
    benign_explanations=(),
    missing_evidence=(),
    unverified_claims=(),
    rationale="arm4_subagent_extract.py placeholder — discarded, never scored",
    recommended_action=ProposedAction.ALLOW,
    confidence=0.0,
    escalate_to_human=True,
)
_PLACEHOLDER_TEXT = _PLACEHOLDER_VERDICT.model_dump_json()


@dataclass
class DumpingProvider:
    """`LLMProvider` that records one line per unique request fingerprint (a
    decision's request is identical across its 3 C-5 samples — see
    `reviewer._sample_decision`, which builds the request once and reuses it —
    so only the first sighting of a fingerprint is written) and answers every
    call with a fixed, schema-valid placeholder. Never calls a network.

    The system + instruction text is identical for every decision in one arm
    (only `variant` — full vs. timeline_only — varies, and that's fixed per
    arm), so it's written ONCE to `preamble_path` rather than repeated on every
    line of `out_path`; only the per-case structured evidence + untrusted
    block go in `out_path`, keeping the per-case dump (and therefore each
    subagent batch built from it) an order of magnitude smaller."""

    out_path: Path
    preamble_path: Path
    _seen: set[str] = field(default_factory=set, repr=False)
    _preamble_written: bool = False

    def complete(self, request: LLMRequest) -> LLMResponse:
        fp = request_fingerprint(request)
        if fp not in self._seen:
            self._seen.add(fp)
            structured = _structured_payload_text(request)
            evidence_keys = _extract_evidence_keys(structured)
            untrusted_text = "\n\n".join(
                segment.text
                for segment in request.segments
                if not isinstance(
                    segment, (SystemSegment, InstructionSegment, StructuredEvidenceSegment)
                )
            )
            with self.out_path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "fp": fp,
                            "structured": structured,
                            "untrusted": untrusted_text,
                            "evidence_keys": list(evidence_keys),
                        }
                    )
                    + "\n"
                )
            if not self._preamble_written:
                self._preamble_written = True
                instruction_text = "\n\n".join(
                    segment.text
                    for segment in request.segments
                    if isinstance(segment, InstructionSegment)
                )
                self.preamble_path.write_text(
                    json.dumps({"system": request.system, "instruction": instruction_text})
                )
        return LLMResponse(
            text=_PLACEHOLDER_TEXT,
            model_snapshot="arm4-subagent-extract-placeholder",
            latency_ms=0.0,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


def extract_arm(
    arm: str,
    world_dir: Path,
    prompts_path: Path,
    preamble_path: Path,
    *,
    as_of: datetime,
    seed: int,
    preset: str,
) -> int:
    """Seed+detect exactly as `arm4_run.run_arm` does, then sample every gatable
    decision through `DumpingProvider` via the SAME `review_cluster_batch` entry
    point `run_arm` uses (so gating/ordering/sampling logic is identical). Every
    decision in the batch is sampled BEFORE any of them would commit
    (`review_cluster_batch`'s whole M-11 point) — we exploit that by never
    calling `session.commit()` at all, only `session.rollback()`, so this pass's
    "review" has zero persisted side effects: it exists purely to observe which
    requests would be sent. Returns the number of gatable decisions found (for
    the fingerprint-uniqueness check the caller performs across both arms)."""
    prompts_path.unlink(missing_ok=True)
    preamble_path.unlink(missing_ok=True)
    inference_url, _ = seed_world(world_dir, arm, seed=seed, preset=preset)
    inference_engine = make_inference_engine(inference_url)

    with Session(inference_engine, future=True) as inf:
        detection = run_detector(inf, as_of=as_of, run_id=f"{arm}-extract")
        inf.commit()

        decisions = list(
            inf.scalars(
                select(Decision)
                .where(
                    Decision.proposed_action != ProposedAction.ALLOW,
                    Decision.status.in_((DecisionStatus.PROPOSED, DecisionStatus.APPLIED)),
                )
                .order_by(Decision.idempotency_key)
            ).all()
        )

        provider = DumpingProvider(out_path=prompts_path, preamble_path=preamble_path)
        review_cluster_batch(
            inf,
            decisions,
            provider=provider,
            run_id=f"{arm}-extract",
            idempotency_key_fn=lambda d: f"{arm}:extract:{d.decision_id}",
            as_of=as_of,
            arm=arm,
        )
        # Deliberately no inf.commit(): discard every state transition
        # review_cluster_batch flushed (escalations/reversals against the
        # always-"escalate" placeholder verdict) — this pass never happened as
        # far as the persisted world is concerned.
        inf.rollback()

    del detection
    return len(decisions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="default: app settings.world_seed")
    parser.add_argument("--preset", default=ARM4_DEFAULT_PRESET)
    parser.add_argument("--as-of", default=WORLD_END.isoformat())
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "data" / "synthetic" / "arm4-subagent"
    )
    parser.add_argument(
        "--world-dir", type=Path, default=REPO_ROOT / "data" / "synthetic" / "arm4-subagent-worlds"
    )
    args = parser.parse_args()

    from app.core.config import settings

    seed = args.seed if args.seed is not None else settings.world_seed
    as_of = datetime.fromisoformat(args.as_of)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    fps_by_arm: dict[str, set[str]] = {}
    for arm in ARMS:
        prompts_path = args.out_dir / f"{arm}-prompts.jsonl"
        preamble_path = args.out_dir / f"{arm}-preamble.json"
        n_decisions = extract_arm(
            arm,
            args.world_dir,
            prompts_path,
            preamble_path,
            as_of=as_of,
            seed=seed,
            preset=args.preset,
        )
        counts[arm] = n_decisions
        fps = set()
        with prompts_path.open() as f:
            for line in f:
                fps.add(json.loads(line)["fp"])
        fps_by_arm[arm] = fps
        status = "OK" if len(fps) == n_decisions else "MISMATCH"
        print(
            f"{arm}: {n_decisions} gatable decisions, {len(fps)} distinct prompt "
            f"fingerprints -> {status}",
            file=sys.stderr,
        )
        if len(fps) != n_decisions:
            raise SystemExit(
                f"{arm}: fingerprint collision or dedup bug — "
                f"{n_decisions} decisions produced only {len(fps)} distinct prompts. "
                "Refusing to proceed: a collision would scramble which subagent "
                "verdict lands on which decision."
            )

    print(json.dumps({"seed": seed, "preset": args.preset, "as_of": args.as_of, "counts": counts}))


if __name__ == "__main__":
    main()

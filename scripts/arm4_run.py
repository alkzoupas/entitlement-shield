"""Score the LLM reviewer arms 4 / 4b / 2 on freshly seeded dev-tier worlds (P3.9).

**Dev tier only (W-1). Nothing this script computes may be reported.** It never
imports `app.db.evaluation_access` or `app.db.world_provenance`, and never calls
`begin_finalization` / `complete_finalization` / `verify_world_pair` — the sealed
Tier-2 path is not reachable from here, and that is grep-checked at the RC-4
checkpoint rather than only asserted in this docstring.

**Isolation: one independently seeded world per arm.** Arm 4 and arm 4b are run
against two separate inference/ground-truth store PAIRS, each built by a full
`scripts/seed.py` run. This costs a second seeding pass and buys the only thing
that makes the two arms comparable: the reviewer's outcomes are real state
transitions, not annotations. An `escalate` moves its decision to `under_review`
and opens an `EscalationRecord`; a `reverse` makes it terminal and writes a
cluster-wide `SuppressionRecord`. Run both arms over one store and L-9's gate
skips, for the second arm, exactly the units the first arm acted on — a strictly
smaller, non-random subset (pinned by
`test_arm4_eval.test_running_both_arms_on_one_db_biases_the_second_arm`). Two
worlds instead let each arm exercise `review_decision` with full fidelity —
reversal/escalation side effects, `_other_actioned_cluster_members`,
`has_open_escalation` — against an untouched world.

Seeding is delegated to `scripts/seed.py` as a SUBPROCESS with
`DATABASE_URL`/`GROUND_TRUTH_DATABASE_URL` overridden. That is not incidental:
`core.config.Settings` is a frozen dataclass built at import time, so an in-process
environment change would not take effect — and `seed.py` is also where the world
receipts are written, which this script therefore never has to import
`world_provenance` to do.

Usage:
    python scripts/arm4_run.py --arm both [--preset min|dev] [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.engine import make_inference_engine
from app.db.ground_truth_store import make_ground_truth_engine
from app.models.enums import DecisionStatus, ProposedAction
from app.models.inference import Decision
from app.providers.cache import CachingProvider, ResponseCache
from app.providers.factory import ProviderConfig, build_provider, load_provider_config
from app.services.detector import run_detector
from app.services.evaluation import (
    REVIEWER_BINARY_THRESHOLD,
    Arm4Report,
    arm2_matched,
    arm4_reviewer,
    arm4b_timeline_only,
    build_eval_rows,
    load_ground_truth,
    matched_group_bootstrap_ci,
    operating_point_record,
)
from app.services.feature_manifest import verified_manifest_hash as verified_feature_hash
from app.services.generator import WORLD_END
from app.services.reviewer import review_cluster_batch
from app.services.reviewer_manifest import verified_manifest_hash as verified_reviewer_hash

#: This tool's own default preset, deliberately NOT the shared generator
#: DEFAULT_PRESET ("dev", untouched per M-13) — arm4_run.py is Phase-4-specific
#: tooling and its default must actually reach the reversible fixture, or
#: `make arm4` silently scores nothing (M-13's whole point).
ARM4_DEFAULT_PRESET = "dev_arm4"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORLD_DIR = REPO_ROOT / "data" / "synthetic" / "arm4"
DEFAULT_ARTIFACT = REPO_ROOT / "data" / "synthetic" / "arm4-report.json"
DEFAULT_LOG_DIR = REPO_ROOT / "logs"
# Kept OUTSIDE world_dir deliberately: world_dir is deleted after every run unless
# --keep-worlds is passed, but the whole point of the transcript (M-6) is that it
# survives a run so a real-provider arm-4 result stays reconstructable from
# committed data even though the live call wasn't deterministic.
DEFAULT_TRANSCRIPT = REPO_ROOT / "data" / "synthetic" / "arm4-transcript.jsonl"

ARMS = ("arm4_reviewer", "arm4b_timeline_only")
_ARM_FUNCS = {"arm4_reviewer": arm4_reviewer, "arm4b_timeline_only": arm4b_timeline_only}

logger = logging.getLogger("arm4_run")


def configure_logging(log_file: Path | None) -> Path:
    """Log INFO+ to the console and DEBUG+ to `log_file` (default: a fresh
    timestamped file under `logs/`, one per run so runs never clobber each
    other's history).

    Only this module's logger is wired up — noisy third-party loggers (e.g.
    sqlalchemy's engine logger, which this script never enables echo on, but
    a future caller might) are left alone rather than silenced globally,
    since a script has no business changing logging config for code it
    merely imports.
    """
    if log_file is None:
        DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = DEFAULT_LOG_DIR / f"arm4_run_{datetime.now():%Y%m%d_%H%M%S}.log"
    else:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    )
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)

    return log_file


def _world_paths(world_dir: Path, arm: str) -> tuple[Path, Path]:
    return (world_dir / f"{arm}-inference.db", world_dir / f"{arm}-ground-truth.db")


def _run_streaming(cmd: list[str], *, cwd: Path, env: dict, log_prefix: str) -> None:
    """Run `cmd`, forwarding its interleaved stdout/stderr to `logger` line-by-line
    as it's produced (INFO), instead of buffering it all until exit. Raises
    `subprocess.CalledProcessError` on a non-zero exit, matching `check=True`."""
    process = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.info("%s%s", log_prefix, line.rstrip())
    returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def seed_world(world_dir: Path, arm: str, *, seed: int, preset: str) -> tuple[str, str]:
    """Seed this arm's own dev world via `scripts/seed.py`, returning the two URLs.

    Any existing stores are deleted first: a partially-reviewed store would let a
    previous run's escalations and suppressions gate this run's decisions, which is
    the exact contamination the per-arm isolation exists to prevent."""
    world_dir.mkdir(parents=True, exist_ok=True)
    inference_db, gt_db = _world_paths(world_dir, arm)
    for path in (inference_db, gt_db):
        path.unlink(missing_ok=True)

    inference_url = f"sqlite:///{inference_db}"
    gt_url = f"sqlite:///{gt_db}"
    env = {
        **os.environ,
        "DATABASE_URL": inference_url,
        "GROUND_TRUTH_DATABASE_URL": gt_url,
    }
    logger.info("[%s] seeding world (seed=%d, preset=%s)", arm, seed, preset)
    _run_streaming(
        [sys.executable, str(REPO_ROOT / "scripts" / "seed.py"),
         "--seed", str(seed), "--preset", preset],
        cwd=REPO_ROOT, env=env, log_prefix=f"[{arm}] [seed.py] ",
    )
    logger.info("[%s] seeding done -> %s , %s", arm, inference_url, gt_url)
    return inference_url, gt_url


def run_arm(
    arm: str,
    world_dir: Path,
    *,
    as_of: datetime,
    seed: int,
    preset: str,
    provider_config: ProviderConfig,
    cache: ResponseCache,
    replay: bool,
) -> tuple[Arm4Report, dict]:
    """Seed a world, detect on it, review every gatable decision under `arm`, and
    score it. Returns the arm report plus a small run-shape record."""
    inference_url, gt_url = seed_world(world_dir, arm, seed=seed, preset=preset)

    inference_engine = make_inference_engine(inference_url)
    gt_engine = make_ground_truth_engine(gt_url)
    run_id = f"{arm}-{uuid.uuid4().hex[:8]}"
    logger.info("[%s] run_id=%s: running detector (as_of=%s)", arm, run_id, as_of.isoformat())

    with Session(inference_engine, future=True) as inf:
        detection = run_detector(inf, as_of=as_of, run_id=run_id)
        inf.commit()
        logger.info(
            "[%s] detector done: %d clusters, %d cluster-cycles, %d decisions created",
            arm, detection.clusters, detection.cluster_cycles, detection.decisions_created,
        )

        # L-9's gatable population: live (proposed/applied) and non-ALLOW. The
        # reviewer re-checks this itself; selecting here only avoids pointless
        # calls that would return `gated=True`.
        #
        # Ordered by `idempotency_key`, NOT `decision_id`. Review order is
        # load-bearing — an earlier decision's escalation or auto-reversal
        # changes the suppression state and `_other_actioned_cluster_members`
        # result that later decisions in the same cluster see — and
        # `decision_id` is `uuid4().hex`, so ordering on it would randomize the
        # review sequence on every run. `idempotency_key` is
        # `{snapshot}:{cycle}:{account}:{config_fingerprint}` and carries a
        # UNIQUE constraint, so it is both a total order and stable for a fixed
        # world seed and config.
        decisions = list(
            inf.scalars(
                select(Decision).where(
                    Decision.proposed_action != ProposedAction.ALLOW,
                    Decision.status.in_((DecisionStatus.PROPOSED, DecisionStatus.APPLIED)),
                ).order_by(Decision.idempotency_key)
            ).all()
        )
        logger.info("[%s] %d gatable decisions to review", arm, len(decisions))
        # ONE provider instance across every call: `FakeReviewerProvider` keeps
        # per-instance `_call_counts`, and a fresh instance per call would reset
        # `DISAGREE`-mode variant cycling to 0 every time. Wrapped in
        # `CachingProvider` (M-6): live calls are recorded into `cache` as they
        # happen so a real-provider run stays replayable; `replay=True` instead
        # requires every call to already be in `cache` (fails closed on a miss,
        # never silently falls through to a live call).
        provider = CachingProvider(
            inner=build_provider(provider_config), cache=cache, replay_only=replay
        )
        # M-11 (docs/decisions.md §13): `review_cluster_batch`, not `review_decision`
        # looped, so the narrowed RC-1 P1-1 guard actually takes effect — every
        # decision in this batch is sampled before any of them commits, and a
        # multi-member cluster's decisions can auto-apply together when they are
        # ALL, independently, unanimous-reverse. A single-decision cluster behaves
        # identically to the old `review_decision` loop (its "batch" is itself).
        outcomes = review_cluster_batch(
            inf, decisions, provider=provider, run_id=run_id,
            idempotency_key_fn=lambda d: f"{arm}:{d.decision_id}", as_of=as_of, arm=arm,
        )
        gated: dict[str, int] = {}
        reviewed = 0
        for outcome in outcomes:
            if outcome.gated:
                gated[outcome.gate_reason or "unknown"] = (
                    gated.get(outcome.gate_reason or "unknown", 0) + 1
                )
            else:
                reviewed += 1
        inf.commit()
        logger.info(
            "[%s] review done: %d reviewed, %d gated (%s)",
            arm, reviewed, sum(gated.values()), gated or "none",
        )

        with Session(gt_engine, future=True) as gt:
            rows = build_eval_rows(inf, load_ground_truth(gt, inf_session=inf), as_of)
        logger.info("[%s] scoring %d eval rows", arm, len(rows))
        report = _ARM_FUNCS[arm](inf, rows)

    run_shape = {
        "inference_store": inference_url,
        "ground_truth_store": gt_url,
        "run_id": run_id,
        "clusters": detection.clusters,
        "cluster_cycles": detection.cluster_cycles,
        "decisions_created": detection.decisions_created,
        "gatable_decisions": len(decisions),
        "decisions_reviewed": reviewed,
        "decisions_gated": gated,
        "eval_rows": len(rows),
    }
    logger.info("[%s] arm complete", arm)
    return report, run_shape


def _arm_record(report: Arm4Report) -> dict:
    return {
        "arm": report.arm,
        "report": asdict(report.report),
        "coverage": asdict(report.coverage),
        "pr_auc_caveat": report.pr_auc_caveat,
        "recovered_rows": sum(1 for v in report.scores.values() if v == 0.0),
        "still_flagged_rows": sum(1 for v in report.scores.values() if v != 0.0),
        # eval §6.2/§6.3 (Phase 4 P4.3): observed discordant proportion p_d and
        # the MDE it implies at this run's reviewed-row count. Both None when
        # there are no reviewed rows / no observed discordance — never a
        # misleading point estimate on zero evidence.
        "discordant_rate": report.discordant_rate,
        "revised_mde": report.revised_mde,
        # P4.8: raw (pre-clamp) recommendation vs. proposed-action severity, and
        # the confidence-binned reliability diagram over that same raw signal.
        "escalation_direction": [asdict(t) for t in report.escalation_direction],
        "reliability_diagram": [asdict(b) for b in report.reliability_diagram],
    }


def _determinism_note(active_provider: str, replay: bool) -> str:
    """Whether per-decision reviewer verdicts (and therefore the confusion cells
    and any figure derived from them) are reproducible across runs depends on the
    provider actually used, not just the world seed — this must not keep claiming
    a fake-provider-specific mechanism once a real provider is active (M-6)."""
    stable_order_note = (
        "Decisions are reviewed in a stable order (by the deterministic "
        "idempotency_key, never by the uuid4 decision_id), so the review "
        "SEQUENCE does not vary between runs."
    )
    if replay:
        return (
            f"FULLY reproducible: --replay requires every reviewer call to come "
            f"from the committed transcript, so verdicts are byte-identical to "
            f"the run that produced it. {stable_order_note}"
        )
    if active_provider == "fake":
        return (
            f"PARTIALLY reproducible at a fixed world seed. {stable_order_note} "
            "The verdicts still do not: decision ids are uuid4 and "
            "FakeReviewerProvider's rule-table bucket is seeded on the "
            "structured payload, which embeds the decision id. Per-decision "
            "verdicts, and therefore the confusion cells and any figure derived "
            "from them, are NOT reproducible across runs of this script. Treat "
            "every number here as one draw, not a fixed value."
        )
    return (
        f"NOT reproducible: the '{active_provider}' provider is a live model call "
        f"at whatever sampling settings it applies (this adapter does not force "
        f"a temperature). {stable_order_note} Per-decision verdicts, and "
        "therefore the confusion cells and any figure derived from them, are one "
        "draw from a live model, not a fixed value — this run's transcript is "
        "recorded so it can be replayed later with --replay for a byte-identical "
        "rerun."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score reviewer arms 4/4b/2 on freshly seeded dev worlds (W-1)."
    )
    parser.add_argument("--arm", choices=[*ARMS, "both"], default="both")
    parser.add_argument("--as-of", default=WORLD_END.isoformat())
    parser.add_argument("--seed", type=int, default=settings.world_seed)
    parser.add_argument(
        "--preset", choices=["dev", "dev_arm4", "min"], default=ARM4_DEFAULT_PRESET
    )
    parser.add_argument("--world-dir", type=Path, default=DEFAULT_WORLD_DIR)
    parser.add_argument("--json", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument(
        "--keep-worlds", action="store_true",
        help="leave the per-arm dev stores on disk for inspection",
    )
    parser.add_argument(
        "--provider-config", type=Path, default=None,
        help="override config/reviewer-provider.yaml (default: the committed file)",
    )
    parser.add_argument("--transcript", type=Path, default=DEFAULT_TRANSCRIPT)
    parser.add_argument(
        "--replay", action="store_true",
        help=(
            "never call the active provider live; every reviewer call must already "
            "be in --transcript or the run fails closed (M-6)"
        ),
    )
    parser.add_argument(
        "--log-file", type=Path, default=None,
        help=(
            "path to the DEBUG-level run log (default: logs/arm4_run_<timestamp>.log); "
            "console output stays at INFO regardless"
        ),
    )
    args = parser.parse_args()

    log_file = configure_logging(args.log_file)
    logger.info("log file: %s", log_file)
    logger.info(
        "arm(s)=%s preset=%s seed=%d as_of=%s", args.arm, args.preset, args.seed, args.as_of
    )

    # Fail closed on either freeze artifact BEFORE any world is seeded: an arm-4
    # artifact must never name a reviewer or detector version whose hash resolves
    # to no committed file.
    reviewer_hash = verified_reviewer_hash()
    feature_hash = verified_feature_hash()
    as_of = datetime.fromisoformat(args.as_of)
    arms = list(ARMS) if args.arm == "both" else [args.arm]
    logger.debug("reviewer_manifest_hash=%s feature_manifest_hash=%s", reviewer_hash, feature_hash)

    provider_config = load_provider_config(args.provider_config)
    cache = ResponseCache(path=args.transcript)
    cache.load()
    logger.info(
        "provider=%s replay=%s transcript=%s", provider_config.active, args.replay, args.transcript,
    )

    reports: dict[str, Arm4Report] = {}
    shapes: dict[str, dict] = {}
    try:
        for arm in arms:
            reports[arm], shapes[arm] = run_arm(
                arm, args.world_dir, as_of=as_of, seed=args.seed, preset=args.preset,
                provider_config=provider_config, cache=cache, replay=args.replay,
            )
            # Saved after each arm, not just once at the end, so a real-provider call
            # already paid for isn't lost if a later arm in the same invocation fails.
            cache.save()
    except Exception:
        logger.exception("run failed; see %s for full detail", log_file)
        raise

    # Arm 2 is the detector swept to the REVIEWER's achieved recall, so it is
    # computed against arm 4 (the full-view arm) when it ran, and against whatever
    # single arm was requested otherwise.
    primary = "arm4_reviewer" if "arm4_reviewer" in reports else arms[0]
    logger.info("computing arm2 (matched to %s) and bootstrap uncertainty", primary)
    inference_url = shapes[primary]["inference_store"]
    with Session(make_inference_engine(inference_url), future=True) as inf:
        arm2 = arm2_matched(inf, reports[primary])
        uncertainty = matched_group_bootstrap_ci(
            reports[primary].reviewed_rows,
            arm2.baseline_scores,
            reports[primary].scores,
            reviewer_threshold=REVIEWER_BINARY_THRESHOLD,
            estimand=(
                f"{primary} minus matched Stage-1 wrongful-restriction rate "
                "(binary still-flagged score)"
            ),
            conditions_on=(
                f"frozen reviewer manifest {reviewer_hash[:12]}, fixed binary "
                f"reviewer policy, and the '{provider_config.active}' provider "
                f"{'(replayed from a committed transcript)' if args.replay else '(live)'}"
            ),
        )

    artifact = {
        "w1_dev_tier_only": True,
        "reporting_status": (
            "dev-tier development only; NOT a sealed performance claim (W-1). No "
            "sealed Tier 2 store is read, opened, or finalized by this script."
        ),
        "reviewer_manifest_hash": reviewer_hash,
        "feature_manifest_hash": feature_hash,
        "provider": {
            "active": provider_config.active,
            "replay": args.replay,
            "transcript": str(args.transcript),
        },
        "as_of": as_of.isoformat(),
        "world_seed": args.seed,
        "preset": args.preset,
        "isolation": (
            "one independently seeded inference/ground-truth store pair per arm; "
            "the reviewer's escalations and reversals are real state transitions "
            "that would gate the second arm's population if shared"
        ),
        "determinism_note": _determinism_note(provider_config.active, args.replay),
        "score_definition": (
            "binary per fixed E-14 cluster-cycle unit: recovered (0.0) only if every "
            "one of the unit's decisions reached status=reversed under a unanimous "
            "reviewer consensus; escalate counts as still-flagged (1.0); units with "
            "no reviewer coverage are excluded from the confusion matrix entirely"
        ),
        "runs": shapes,
        "arms": {arm: _arm_record(report) for arm, report in reports.items()},
        "arm2": {
            "matched_to": arm2.arm4_arm,
            "baseline_at_threshold": asdict(arm2.baseline),
            "matched_operating_point": operating_point_record(arm2.matched),
            "rows": arm2.n_rows,
            "fps_removed": arm2.fps_removed,
            "fns_added": arm2.fns_added,
            "exchange_rate": arm2.exchange_rate,
            "exchange_rate_status": arm2.exchange_rate_status,
            "exchange_rate_definition": "false positives removed per false negative added",
        },
        "uncertainty": uncertainty,
    }

    payload = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)
    print(payload)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n")
        logger.info("artifact written to %s", args.json)
    if not args.keep_worlds:
        shutil.rmtree(args.world_dir, ignore_errors=True)
        logger.debug("world dir removed: %s", args.world_dir)
    else:
        logger.info("worlds kept on disk: %s", args.world_dir)
    logger.info("done. log file: %s", log_file)


if __name__ == "__main__":
    main()

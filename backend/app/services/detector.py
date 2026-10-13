"""Batch detector run — the end-to-end detection slice (docs/architecture.md §2). Step 6.

Wires the pipeline over the dev world at a fixed `as_of`:

    cluster (once, upstream) → materialize snapshots
      → per fixed cluster-cycle unit: features → banding → score → policy
      → suppression check → per account: idempotent decision + typed audit
      → an immutable per-unit detection record (contributions, band, cap reasons)

Guarantees:
  * **Idempotent** (architecture §7): the decision key derives from the
    content-addressed snapshot, the cycle window, the account, AND a fingerprint of
    the firing-affecting config — so re-running the same world+config is a no-op,
    while a config change is a new key that supersedes (review findings 7, 8).
  * **Suppression is honoured** (policy §9.3): a reversed cluster whose evidence is
    unchanged is not re-flagged until its window lapses (review finding 1).
  * **Every detection is auditable** (review findings 9, 17): each unit emits an
    immutable detection record carrying its ordered per-feature contributions, band,
    risk, action, cap reasons, and versions — the artifact a decision reconstructs from.

There is no reviewer in Phase 1; reversal is the analyst/manual path
(`policy_engine.record_reversal`).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.clock import now_utc
from app.core.decision_status import SUPERSEDABLE
from app.db.decisions import record_decision_idempotent, transition_decision
from app.db.publication import lock_publication
from app.models.enums import ActorType, DecisionStatus, EventType, ProposedAction
from app.models.inference import AccountEvent, Decision, DetectorInvocation, StandingAuthority
from app.schemas.audit import DetectorRunRecorded
from app.services.audit import append_audit
from app.services.claim import current_cycle
from app.services.clustering import Cluster, compute_clusters, materialize
from app.services.escalation import has_open_escalation, release_expired_escalations
from app.services.evidence import associate_evidence, decision_evidence, persist_evidence
from app.services.feature_manifest import MANIFEST_HASH, verified_manifest_hash
from app.services.features import (
    SEED_WEIGHTS,
    Rung,
    enumerate_cluster_cycles,
    score_cluster_cycle,
)
from app.services.policy_engine import (
    decide,
    evidence_signature,
    select_enforcement_targets,
    should_flag,
)
from app.services.risk_scorer import ContributionRecord, assess, export_contributions

DETECTOR_VERSION = "dev-detector-v1"
POLICY_VERSION = "policy-v1"

# Reviewer-context lookback (P4.8 fix, docs/decisions.md §13 M-12 addendum): the
# per-unit evidence timeline additionally captures member accounts' history in
# the `_TIMELINE_LOOKBACK_DAYS` immediately BEFORE `window_start`, so a reviewer
# can cite pre-existing SIGNUP/LOGIN activity instead of only ever seeing the
# current cycle. This is purely an evidence-capture widening for citation
# purposes — it does not touch `features.py`/`risk_scorer.py`'s scoring window
# (those still score only `[window_start, window_end)`), so no feature
# contribution or `feature_manifest_hash` moves. Bounded and fixed (not
# "everything ever") to keep the persisted evidence, and the prompt built from
# it, deterministic and to stop the prompt from ballooning in size across every
# reviewed decision. 80 days is chosen to comfortably cover
# `reversible_singleton_legit`'s fixture (generator.py's `_gen_reversible_pair`):
# relative to cycle 2's `window_start`, the primary's SIGNUP sits 75 days back
# and its two pre-existing LOGINs sit 65/55 days back — empirically confirmed
# by walking the generated world, not guessed (a naive ~45-60 day guess based
# on the fixture's own comments about "45 days before cycle 1" undercounts by
# `CYCLE_DAYS` because the reviewed unit is cycle 2, one full 30-day cycle
# later than cycle 1).
_TIMELINE_LOOKBACK_DAYS = 80

# Actions that restrict the incremental grant → apply only to enforcement targets
# (policy §8). `light_friction` is a soft, cluster-wide tier and is not targeted.
_GRANT_RESTRICTING = frozenset(
    {ProposedAction.HOLD_INCREMENTAL_CREDITS, ProposedAction.TEMPORARY_HOLD}
)


def _iso_utc(value: datetime) -> str:
    value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return value.isoformat()


@dataclass(frozen=True)
class DetectionRecord:
    """Immutable per-unit detection evidence (review findings 9/17) — enough to
    reconstruct why a decision was (or was not) proposed for one cluster-cycle."""

    cluster_snapshot_id: str
    cycle_id: str
    as_of: str
    snapshot_hash: str
    member_account_ids: tuple[str, ...]
    edges: tuple[tuple[str, str, float, int, str, str], ...]
    qualified_stream_count: int
    allowance: int
    ambiguous: bool
    excess: float
    risk_score: float
    proposed_action: str
    cap_reasons: tuple[str, ...]
    suppressed: bool
    detector_version: str
    policy_version: str
    feature_manifest_hash: str
    rung: str
    contributions: tuple[ContributionRecord, ...]


@dataclass(frozen=True)
class RunResult:
    run_id: str
    as_of: datetime
    clusters: int
    cluster_cycles: int
    decisions_created: int
    decisions_skipped: int
    decisions_withdrawn: int
    suppressed_units: int
    escalations_released: int
    detections: tuple[DetectionRecord, ...]


def config_fingerprint(
    detector_version: str,
    policy_version: str,
    rung: Rung,
    *,
    manifest_hash: str = MANIFEST_HASH,
) -> str:
    """Hash of every firing-affecting config knob (review finding 8): a weight or
    version change yields a new fingerprint → new idempotency key → new decision."""
    payload = json.dumps(
        {
            "detector_version": detector_version,
            "policy_version": policy_version,
            "rung": rung.name,
            "weights": {k: SEED_WEIGHTS[k] for k in sorted(SEED_WEIGHTS)},
            # The frozen manifest hash covers every firing-affecting field (predicate
            # params, threshold, masking version) — not just the weights above — so
            # any freeze change yields a new fingerprint → new idempotency key.
            "feature_manifest_hash": manifest_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cycle_claimed(
    session: Session,
    member_ids: list[str],
    start: datetime,
    cycle_end: datetime,
    as_of: datetime,
) -> int:
    """Sum claims owned by `[start, cycle_end)` and visible at `as_of`.

    The two upper bounds intentionally remain separate: `cycle_end` assigns a
    boundary event to the following cycle, while the inclusive `as_of` clamp
    permits an event exactly at the observation instant.
    """
    rows = session.scalars(
        select(AccountEvent).where(
            AccountEvent.account_id.in_(member_ids),
            AccountEvent.event_type == EventType.CLAIM_CREDITS,
            AccountEvent.timestamp >= start,
            AccountEvent.timestamp < cycle_end,
            AccountEvent.timestamp <= as_of,
        )
    ).all()
    return sum(r.credits_amount or 0 for r in rows)


def _edge_projection(cluster: Cluster) -> list[tuple[str, str, float, int, str, str]]:
    """Canonical immutable linkage evidence used in artifacts and suppression."""
    return [
        (
            e.account_a,
            e.account_b,
            e.weight,
            e.evidence_count,
            _iso_utc(e.first_seen),
            _iso_utc(e.last_seen),
        )
        for e in cluster.edges
    ]


def _supersede_unaffirmed(
    session: Session,
    accounts_seen: set[str],
    affirmed_keys: set[str],
    run_id: str,
) -> int:
    """Withdraw unaffirmed authority across snapshot and evaluation frames."""
    if not accounts_seen:
        return 0
    stale = session.scalars(
        select(Decision).where(
            Decision.account_id.in_(sorted(accounts_seen)),
            Decision.status.in_(SUPERSEDABLE),
            Decision.idempotency_key.not_in(sorted(affirmed_keys) or [""]),
        )
    ).all()
    withdrawn = 0
    for dec in stale:
        # A decision under an open L-5 escalation is `under_review` for a reason
        # unrelated to this run's config diff — skip it so it isn't resuperseded
        # (and implicitly re-triggered for review) every night the window is open
        # (RC-1 P1-3). `release_expired_escalations` is the only path that closes it.
        if has_open_escalation(session, dec):
            continue
        transition_decision(
            session,
            dec,
            DecisionStatus.SUPERSEDED,
            actor=ActorType.SYSTEM,
            reason=f"config change (run {run_id}): unit no longer actionable / not a target",
        )
        withdrawn += 1
    return withdrawn


def run_detector(
    session: Session,
    *,
    as_of: datetime,
    run_id: str,
    detector_version: str = DETECTOR_VERSION,
    policy_version: str = POLICY_VERSION,
    target_rung: Rung = Rung.S2,
    artifact_path: str | None = None,
) -> RunResult:
    """Cluster, score, and record decisions over the whole inference store `as_of`.

    Records a decision only for accounts that receive an action, keyed idempotently,
    and skips a unit whose reversed evidence is still suppressed. Emits an immutable
    detection record per unit and one `detector_run_recorded` audit event; optionally
    writes the detection artifact atomically to `artifact_path`."""
    # Every detector entrypoint, including the HTTP API, must prove the committed
    # freeze artifact before clustering can materialize derived state or decisions.
    manifest_hash = verified_manifest_hash()
    as_of = as_of.replace(tzinfo=UTC) if as_of.tzinfo is None else as_of.astimezone(UTC)
    fingerprint = config_fingerprint(
        detector_version, policy_version, target_rung, manifest_hash=manifest_hash
    )
    lock_publication(session)
    if session.get(DetectorInvocation, run_id) is not None:
        raise ValueError(f"detector run identity already used: {run_id}")
    latest = session.scalar(select(DetectorInvocation.as_of)
                            .order_by(DetectorInvocation.as_of.desc()).limit(1))
    if latest is not None and as_of < latest.replace(tzinfo=UTC):
        raise ValueError("detector publication cannot move authority backward in time")
    session.add(DetectorInvocation(run_id=run_id, as_of=as_of,
                                  config_fingerprint=fingerprint, created_at=now_utc()))
    session.flush()
    clusters = compute_clusters(session, as_of)
    materialize(session, clusters, as_of, created_at=now_utc())

    created = skipped = suppressed_units = cluster_cycles = 0
    detections: list[DetectionRecord] = []
    # Reconcile by account, including ALLOW and target shrink across new snapshots.
    accounts_seen: set[str] = set()
    affirmed_keys: set[str] = set()

    for cluster in clusters:
        snapshot_id = cluster.snapshot_id
        accounts_seen.update(cluster.member_account_ids)
        members = list(cluster.member_account_ids)
        edges = _edge_projection(cluster)
        for unit in enumerate_cluster_cycles(session, snapshot_id, members, as_of):
            cluster_cycles += 1
            window_end = min(unit.window_end, unit.as_of)  # never read past as_of
            score = score_cluster_cycle(session, unit, members, target_rung)
            assessment = assess(score)
            claimed = _cycle_claimed(
                session, members, unit.window_start, unit.window_end, unit.as_of
            )
            outcome = decide(assessment, score.banding, claimed=claimed)

            # Suppression: an actionable unit whose reversed evidence is unchanged and
            # still within its window is not re-flagged (policy §9.3, finding 1).
            actionable = outcome.proposed_action is not ProposedAction.ALLOW
            signature = evidence_signature(members, edges, outcome.excess)
            suppressed = actionable and not should_flag(session, signature, as_of)
            if suppressed:
                suppressed_units += 1

            detection = _detection(
                unit, cluster, score, assessment, outcome, suppressed,
                detector_version, policy_version, target_rung, fingerprint, manifest_hash,
            )
            detections.append(detection)
            payload = asdict(detection)
            payload["window_start"] = unit.window_start.isoformat()
            payload["window_end"] = unit.window_end.isoformat()
            # Reviewer-context lookback (see `_TIMELINE_LOOKBACK_DAYS`): captures a
            # bounded window of members' pre-existing history so a reviewer can cite
            # raw facts (e.g. SIGNUP/LOGIN) that predate this cycle, without touching
            # the scored `[window_start, window_end)` inputs used above.
            timeline_start = unit.window_start - timedelta(days=_TIMELINE_LOOKBACK_DAYS)
            payload["timeline"] = [
                {
                    "event_id": e.event_id,
                    "account_id": e.account_id,
                    "timestamp": _iso_utc(e.timestamp),
                    "event_type": e.event_type.value,
                    "credits_amount": e.credits_amount,
                }
                for e in session.scalars(
                    select(AccountEvent).where(
                        AccountEvent.account_id.in_(members),
                        AccountEvent.timestamp >= timeline_start,
                        AccountEvent.timestamp < unit.window_end,
                        AccountEvent.timestamp <= as_of,
                    ).order_by(AccountEvent.timestamp, AccountEvent.event_id)
                )
            ]
            evidence = persist_evidence(session, payload)

            if not actionable or suppressed:
                continue

            if outcome.proposed_action in _GRANT_RESTRICTING:
                targets = set(
                    select_enforcement_targets(
                        session, members, score.banding.allowance,
                        unit.window_start, window_end,
                    )
                )
            else:
                targets = set(members)  # light_friction is cluster-wide

            for account_id in sorted(members):
                if account_id not in targets:
                    continue
                key = f"{snapshot_id}:{unit.cycle_id}:{account_id}:{fingerprint}"
                affirmed_keys.add(key)
                decision, was_created = record_decision_idempotent(
                    session,
                    account_id=account_id,
                    risk_score=assessment.risk_score,
                    proposed_action=outcome.proposed_action,
                    detector_version=detector_version,
                    policy_version=policy_version,
                    run_id=run_id,
                    idempotency_key=key,
                    cluster_snapshot_id=snapshot_id,
                )
                associate_evidence(session, decision.decision_id, evidence.evidence_id)
                created += was_created
                skipped += not was_created

    # Reconcile decisions the current config no longer produces (review finding 4).
    withdrawn = _supersede_unaffirmed(session, accounts_seen, affirmed_keys, run_id)
    for account_id in sorted(accounts_seen):
        authority = session.get(StandingAuthority, account_id)
        if authority is None:
            authority = StandingAuthority(account_id=account_id)
            session.add(authority)
        standing = session.get(Decision, authority.decision_id) if authority.decision_id else None
        if standing is None or standing.idempotency_key not in affirmed_keys:
            authority.decision_id = None
        authority.run_id = run_id
        authority.as_of = as_of
        cycle = current_cycle(session, account_id, as_of)
        authority.cycle_id = cycle.cycle_id if cycle else None
        if authority.decision_id is not None:
            context = decision_evidence(session, authority.decision_id)
            if (cycle is None or context is None
                    or not (datetime.fromisoformat(context["window_start"]) <= as_of
                            < datetime.fromisoformat(context["window_end"]))):
                authority.decision_id = None

    # L-5's 7-day auto-release (Phase 3 plan P3.6), hooked into the nightly batch
    # rather than a separate cron. `as_of`-bounded like everything else in this
    # function: a historical run never releases an escalation whose expiry is in
    # that run's future, even if wall-clock time has since passed it.
    escalations_released = release_expired_escalations(session, as_of=as_of, run_id=run_id)

    append_audit(
        session,
        DetectorRunRecorded(
            entity_type="detector_run",
            entity_id=run_id,
            actor_type=ActorType.SYSTEM,
            run_id=run_id,
            as_of=as_of.isoformat(),
            clusters=len(clusters),
            cluster_cycles=cluster_cycles,
            decisions_created=created,
            decisions_skipped=skipped,
        ),
    )
    session.flush()

    result = RunResult(
        run_id=run_id,
        as_of=as_of,
        clusters=len(clusters),
        cluster_cycles=cluster_cycles,
        decisions_created=created,
        decisions_skipped=skipped,
        decisions_withdrawn=withdrawn,
        suppressed_units=suppressed_units,
        escalations_released=escalations_released,
        detections=tuple(detections),
    )
    if artifact_path is not None:
        _write_artifact(artifact_path, result)
    return result


def _detection(
    unit, cluster, score, assessment, outcome, suppressed,
    detector_version, policy_version, rung, fingerprint, manifest_hash,
) -> DetectionRecord:
    rows = export_contributions(
        unit,
        score,
        snapshot_hash=cluster.content_hash,
        detector_config_hash=fingerprint,
        feature_manifest_hash=manifest_hash,
    )
    return DetectionRecord(
        cluster_snapshot_id=unit.cluster_snapshot_id,
        cycle_id=unit.cycle_id,
        as_of=unit.as_of.isoformat(),
        snapshot_hash=cluster.content_hash,
        member_account_ids=tuple(cluster.member_account_ids),
        edges=tuple(_edge_projection(cluster)),
        qualified_stream_count=score.banding.qualified_stream_count,
        allowance=score.banding.allowance,
        ambiguous=score.banding.ambiguous,
        excess=outcome.excess,
        risk_score=assessment.risk_score,
        proposed_action=outcome.proposed_action.value,
        cap_reasons=outcome.cap_reasons,
        suppressed=suppressed,
        detector_version=detector_version,
        policy_version=policy_version,
        feature_manifest_hash=manifest_hash,
        rung=rung.name,
        contributions=tuple(rows),
    )


def detection_bytes(detections: tuple[DetectionRecord, ...]) -> bytes:
    """Canonical serialization of the detection artifact — byte-stable across
    identical runs (review finding 9)."""
    return json.dumps(
        [asdict(d) for d in detections], sort_keys=True, separators=(",", ":")
    ).encode()


def _write_artifact(path: str, result: RunResult) -> None:
    """Write the detection artifact atomically (temp file + rename)."""
    payload = {
        "run_id": result.run_id,
        "as_of": result.as_of.isoformat(),
        "detections": [asdict(d) for d in result.detections],
    }
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, sort_keys=True, separators=(",", ":"))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

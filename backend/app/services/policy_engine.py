"""Policy engine — the action ladder, the safety gates, enforcement targeting, and
reversal/suppression semantics (docs/policy.md §6–§10). Step 5.

Nothing here learns or tunes: it maps a `RiskAssessment` + `BandingResult` onto the
policy §7 action ladder, then applies the overriding caps (§6 ≥ 2-family gate, §4.6
ambiguity, §4.7 large_shared) that can only ever *lower* severity. The cluster is
the unit of assessment; the **account is the unit of action**, and only the
incremental grant that pushed the cluster past its allowance is restricted (§8).

**No automatic permanent suspension at any level (§7).** The action enum tops out at
`temporary_hold`, so this is structural, not merely intended.

Reversal (analyst/manual path only in Phase 1 — no reviewer yet) writes a 60-day
suppression record so the nightly detector does not re-flag identical evidence
(§9.3); materially new evidence yields a new signature and re-evaluates (§9.4).
Credit restoration grants only actual held operations in the synthetic claim ledger,
for the current cycle, without resetting or extending the cycle clock.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.clock import now_utc
from app.db.decisions import transition_decision
from app.models.enums import ActorType, CapacityBand, DecisionStatus, EventType, ProposedAction
from app.models.inference import (
    AccountEvent,
    Decision,
    EntitlementCycle,
    SuppressionRecord,
)
from app.schemas.audit import CreditsRestored
from app.services.audit import append_audit as _append_audit
from app.services.banding import BandingResult
from app.services.risk_scorer import RiskAssessment

# --- Policy constants (config/thresholds.yaml, policy-versioned) --------------

MIN_EVIDENCE_FAMILIES = 2  # §6 ≥ 2-family action gate
SUPPRESSION_DAYS = 60  # §9.3 reversal suppression window

_LIGHT_FRICTION_MAX = 0.25  # ≤ 25% excess → light_friction (§7)
_HOLD_INCREMENTAL_MAX = 1.00  # ≤ 100% excess → hold_incremental_credits (§7)

# Severity rank over the action ladder (least → most severe). Public: the reviewer's
# C-4 severity clamp (docs/decisions.md L-6) reuses this ordering rather than
# inventing a parallel one.
ACTION_SEVERITY: dict[ProposedAction, int] = {
    ProposedAction.ALLOW: 0,
    ProposedAction.LIGHT_FRICTION: 1,
    ProposedAction.HOLD_INCREMENTAL_CREDITS: 2,
    ProposedAction.TEMPORARY_HOLD: 3,
}
_SEVERITY = ACTION_SEVERITY


# --- The ladder + gates ------------------------------------------------------


@dataclass(frozen=True)
class PolicyDecision:
    proposed_action: ProposedAction
    base_action: ProposedAction  # the ladder action before caps
    capped: bool  # a guardrail lowered severity
    cap_reasons: tuple[str, ...]
    excess: float  # claimed / allowance − 1


def _excess_band_action(excess: float) -> ProposedAction:
    """Map allowance excess onto the ladder (policy §7)."""
    if excess <= 0.0:
        return ProposedAction.ALLOW
    if excess <= _LIGHT_FRICTION_MAX:
        return ProposedAction.LIGHT_FRICTION
    if excess <= _HOLD_INCREMENTAL_MAX:
        return ProposedAction.HOLD_INCREMENTAL_CREDITS
    return ProposedAction.TEMPORARY_HOLD


def excess_band(excess: float) -> str:
    """Coarse band label used in the evidence signature (§9.4): a jump to a higher
    band is 'a new allowance excess' and voids suppression."""
    return _excess_band_action(excess).value


def decide(assessment: RiskAssessment, banding: BandingResult, *, claimed: int) -> PolicyDecision:
    """Resolve the proposed action for one cluster-cycle.

    An action is only proposed when the risk score clears the 0.60 gate. The ladder
    action is then capped at `light_friction` when any guardrail holds: fewer than
    two incriminating evidence families (§6 — also the timing safety property, since
    family 4 alone can never reach two), an ambiguous cluster (§4.6), or a
    `large_shared` cluster (§4.7). Caps only lower severity; they never raise it."""
    allowance = banding.allowance
    excess = (claimed / allowance - 1.0) if allowance else 0.0

    if not assessment.above_threshold:
        # Below the score gate: no action proposed, regardless of excess.
        return PolicyDecision(
            proposed_action=ProposedAction.ALLOW,
            base_action=ProposedAction.ALLOW,
            capped=False,
            cap_reasons=(),
            excess=excess,
        )

    base = _excess_band_action(excess)

    cap_reasons: list[str] = []
    if len(assessment.firing_families) < MIN_EVIDENCE_FAMILIES:
        cap_reasons.append("insufficient_families")
    if banding.ambiguous:
        cap_reasons.append("ambiguous")
    if banding.band is CapacityBand.LARGE_SHARED:
        cap_reasons.append("large_shared")

    action = base
    if cap_reasons and _SEVERITY[base] > _SEVERITY[ProposedAction.LIGHT_FRICTION]:
        action = ProposedAction.LIGHT_FRICTION

    return PolicyDecision(
        proposed_action=action,
        base_action=base,
        capped=action != base,
        cap_reasons=tuple(cap_reasons),
        excess=excess,
    )


# --- Enforcement targeting (policy §8) ---------------------------------------


def select_enforcement_targets(
    session: Session,
    member_account_ids: Iterable[str],
    allowance: int,
    window_start: datetime,
    window_end: datetime,
) -> list[str]:
    """Accounts whose claim pushed the cluster past its allowance, in claim-timestamp
    order (policy §8). An early-claiming bystander is never restricted — this bounds
    the blast radius of a clustering error."""
    members = sorted(set(member_account_ids))
    claims = sorted(
        session.scalars(
            select(AccountEvent).where(
                AccountEvent.account_id.in_(members),
                AccountEvent.event_type == EventType.CLAIM_CREDITS,
                AccountEvent.timestamp >= window_start,
                AccountEvent.timestamp <= window_end,
            )
        ).all(),
        key=lambda e: (e.timestamp, e.account_id),
    )
    targets: list[str] = []
    running = 0
    for c in claims:
        running += c.credits_amount or 0
        if running > allowance:
            targets.append(c.account_id)
    return targets


# --- Reversal + suppression (policy §9) --------------------------------------


#: Excess is quantised to this granularity in the signature. Design clarification 1
#: (recorded in docs/decisions.md): "a new allowance excess" (§9.4) means a change in
#: the canonical numeric excess beyond this step — NOT merely crossing an action band.
EXCESS_QUANTUM = 0.1


def _canonical_excess(excess: float) -> float:
    return round(round(excess / EXCESS_QUANTUM) * EXCESS_QUANTUM, 1)


def evidence_signature(
    member_account_ids: Iterable[str],
    edges: Iterable[tuple[object, ...]],
    excess: float,
) -> str:
    """Stable, `as_of`-independent signature over the evidence a suppression is keyed
    on (§9.3/§9.4). It projects the complete immutable snapshot edge evidence —
    endpoints, effective weight, shared-value count, and first/last observation —
    rather than the former lossy endpoint/count subset. A new account or any
    material edge change therefore re-flags; snapshot `as_of` is excluded so
    unchanged evidence stays suppressed across nightly runs.

    Three-field legacy projections remain readable only to reverse decisions
    created before Gate G; they cannot collide with the v2 canonical form.
    """
    edge_proj: list[list[object]] = []
    for edge in edges:
        if len(edge) == 6:
            a, b, weight, count, first_seen, last_seen = edge
            edge_proj.append([
                *sorted((str(a), str(b))), round(float(str(weight)), 12), int(str(count)),
                str(first_seen), str(last_seen),
            ])
        elif len(edge) == 3:
            a, b, count = edge
            edge_proj.append([*sorted((str(a), str(b))), "legacy-v1", int(str(count))])
        else:
            raise ValueError("edge evidence must contain 3 legacy or 6 canonical fields")
    edge_proj.sort(key=lambda item: json.dumps(item, separators=(",", ":")))
    payload = json.dumps(
        {
            "members": sorted(set(member_account_ids)),
            "edges": edge_proj,
            "excess": _canonical_excess(excess),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _current_cycle(session: Session, account_id: str, as_of: datetime) -> EntitlementCycle | None:
    """The account's open cycle containing `as_of` (cycle_start ≤ as_of < cycle_end)."""
    return session.scalars(
        select(EntitlementCycle)
        .where(
            EntitlementCycle.account_id == account_id,
            EntitlementCycle.cycle_start <= as_of,
            EntitlementCycle.cycle_end > as_of,
        )
        .order_by(EntitlementCycle.cycle_start.desc())
    ).first()


def restore_held_credits(
    session: Session,
    decision: Decision,
    *,
    actor: ActorType,
    as_of: datetime,
) -> str | None:
    """Restore this decision's confirmed current-cycle held credits (§9.1/§9.2,
    review finding 10) without touching status or writing a suppression record —
    the credit-restore mechanics `record_reversal` and L-5 escalation release
    (`services/escalation.escalate_decision`) both need, factored out so escalation
    can release a held grant-restricting hold WITHOUT reversing the decision.
    Returns the cycle_id credits were restored against, or `None` if there is no
    open cycle (nothing to restore). Idempotent by construction per call: a
    matching claim's `hold_state` flips to `"none"` immediately, so calling this
    twice for the same decision/cycle restores nothing the second time (`held_amount`
    is already 0) — but it still writes a `CreditsRestored` audit row with
    `amount_restored=0`, so callers should not call it twice for one logical event."""
    from app.services.claim import lock_allowance

    lock_allowance(session)

    # The synthetic ledger distinguishes genuinely held grants from unused balance.
    # Closed-cycle claims and provisional deadline-worker holds are not restored.
    from app.models.claims import SyntheticClaim
    from app.services.claim import grant_credits

    cycle = _current_cycle(session, decision.account_id, as_of)
    if cycle is None:
        return None
    claims = session.scalars(
        select(SyntheticClaim).where(
            SyntheticClaim.account_id == decision.account_id,
            SyntheticClaim.decision_id == decision.decision_id,
            SyntheticClaim.cycle_id == cycle.cycle_id,
            SyntheticClaim.hold_state == "confirmed",
            SyntheticClaim.held_amount > 0,
        )
    ).all()
    restored = 0
    for claim in claims:
        amount = claim.held_amount
        claim.granted_amount += amount
        claim.held_amount = 0
        claim.hold_state = "none"
        claim.restored_at = as_of
        grant_credits(session, claim, amount, as_of)
        restored += amount
    _append_audit(
        session,
        CreditsRestored(
            entity_type="decision",
            entity_id=decision.decision_id,
            actor_type=actor,
            decision_id=decision.decision_id,
            cycle_id=cycle.cycle_id,
            amount_restored=restored,
        ),
    )
    return cycle.cycle_id


def record_reversal(
    session: Session,
    decision: Decision,
    *,
    member_account_ids: Iterable[str],
    edges: Iterable[tuple[object, ...]],
    excess: float,
    actor: ActorType = ActorType.ANALYST,
    reason: str | None = None,
    as_of: datetime | None = None,
) -> str:
    """Reverse a decision and write its suppression window (§9). Transitions status to
    `reversed` (append-only audit), **restores the current cycle's held credits**
    (audited, clock unchanged; closed cycles untouched — §9.1/§9.2, review finding 10),
    and records a 60-day suppression for the evidence signature. Returns the signature."""
    from app.db.publication import lock_publication
    from app.services.claim import utc

    created = utc(as_of or now_utc())
    lock_publication(session)
    transition_decision(
        session, decision, DecisionStatus.REVERSED, actor=actor, reason=reason, at=created
    )
    restore_held_credits(session, decision, actor=actor, as_of=created)

    signature = evidence_signature(member_account_ids, edges, excess)
    session.add(
        SuppressionRecord(
            suppression_id=uuid.uuid4().hex,
            evidence_signature=signature,
            cluster_snapshot_id=decision.cluster_snapshot_id,
            decision_id=decision.decision_id,
            created_at=created,
            expires_at=created + timedelta(days=SUPPRESSION_DAYS),
        )
    )
    session.flush()
    return signature


def is_suppressed(session: Session, signature: str, as_of: datetime) -> bool:
    """True while a live suppression record covers `as_of` — `created_at ≤ as_of <
    expires_at` (review finding 11: a historical run must not see a future reversal)."""
    row = session.scalars(
        select(SuppressionRecord).where(
            SuppressionRecord.evidence_signature == signature,
            SuppressionRecord.created_at <= as_of,
            SuppressionRecord.expires_at > as_of,
        )
    ).first()
    return row is not None


def should_flag(session: Session, signature: str, as_of: datetime) -> bool:
    """Whether the detector may re-flag this evidence (the inverse of suppression,
    §9.3). Materially new evidence produces a different signature and is never
    suppressed — the caller recomputes the signature from current evidence."""
    return not is_suppressed(session, signature, as_of)

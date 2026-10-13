"""View builders: ORM rows → plain dicts for the templates and JSON responses.

Every field here is drawn from the **inference** models, which by locked invariant
carry no label / scenario / split-group column (docs/decisions.md; AGENTS.md). Keeping
the projection explicit is the point — a reviewer can read exactly what the console can
show, and the GT-boundary test asserts this package never imports the ground-truth reader.
"""

from __future__ import annotations

from app.models.inference import (
    Account,
    Appeal,
    AuditEvent,
    ClusterSnapshot,
    Decision,
    EntitlementCycle,
    EscalationRecord,
    HumanReview,
    LlmReview,
)


def _iso(dt: object) -> str | None:
    return dt.isoformat() if dt is not None and hasattr(dt, "isoformat") else None


def decision_view(d: Decision) -> dict:
    return {
        "decision_id": d.decision_id,
        "account_id": d.account_id,
        "cluster_snapshot_id": d.cluster_snapshot_id,
        "created_at": _iso(d.created_at),
        "risk_score": round(d.risk_score, 4),
        "proposed_action": d.proposed_action.value,
        "status": d.status.value,
        "detector_version": d.detector_version,
        "policy_version": d.policy_version,
        "run_id": d.run_id,
        "superseded_by": d.superseded_by,
    }


def account_view(a: Account) -> dict:
    return {
        "account_id": a.account_id,
        "created_at": _iso(a.created_at),
        "email_domain_type": a.email_domain_type.value,
        "verification_level": a.verification_level.value,
    }


def snapshot_view(s: ClusterSnapshot | None) -> dict | None:
    if s is None:
        return None
    return {
        "snapshot_id": s.snapshot_id,
        "as_of": _iso(s.as_of),
        "member_account_ids": list(s.member_account_ids),
        "edges": list(s.edges),
        "evidence_summary": dict(s.evidence_summary),
        "content_hash": s.content_hash,
    }


def cycle_view(c: EntitlementCycle) -> dict:
    return {
        "cycle_id": c.cycle_id,
        "cycle_start": _iso(c.cycle_start),
        "cycle_end": _iso(c.cycle_end),
        "credits_granted": c.credits_granted,
        "credits_consumed": c.credits_consumed,
        "credits_expired": c.credits_expired,
    }


def audit_view(e: AuditEvent) -> dict:
    return {
        "audit_id": e.audit_id,
        "timestamp": _iso(e.timestamp),
        "entity_type": e.entity_type,
        "entity_id": e.entity_id,
        "event_name": e.event_name,
        "actor_type": e.actor_type.value,
        "payload": dict(e.payload_json),
    }


def human_review_view(r: HumanReview) -> dict:
    return {
        "review_id": r.review_id,
        "decision_id": r.decision_id,
        "reviewer_action": r.reviewer_action,
        "reviewer_reason": r.reviewer_reason,
        "final_outcome": r.final_outcome,
        "created_at": _iso(r.created_at),
    }


def llm_review_view(r: LlmReview) -> dict:
    """One sample's FINAL persisted verdict (P3.7 console/API). Jinja autoescapes
    the template this feeds, so `rationale`/citations reaching the page from an
    untrusted appeal-adjacent field are never rendered as raw HTML."""
    verdict = r.structured_output_json or {}
    return {
        "review_id": r.review_id,
        "sample_index": r.sample_index,
        "arm": r.arm,
        "provider": r.provider,
        "derived_verdict": r.derived_verdict,
        "recommendation": r.recommendation,
        "confidence": r.confidence,
        "citations": list(r.citations_json or []),
        "rationale": verdict.get("rationale"),
        "validation_errors": dict(r.validation_errors) if r.validation_errors else None,
        "retry_count": r.retry_count,
    }


def escalation_view(e: EscalationRecord) -> dict:
    return {
        "escalation_id": e.escalation_id,
        "decision_id": e.decision_id,
        "created_at": _iso(e.created_at),
        "expires_at": _iso(e.expires_at),
        "source": e.source,
    }


def appeal_view(a: Appeal) -> dict:
    """`text` is untrusted, appeal-content-adjacent input (policy §11 — carries no
    evidentiary weight). The template rendering it MUST rely on Jinja's default
    autoescaping rather than any `|safe`/`Markup` filter here."""
    return {
        "appeal_id": a.appeal_id,
        "account_id": a.account_id,
        "decision_id": a.decision_id,
        "submitted_at": _iso(a.submitted_at),
        "text": a.text,
        "truncated": a.truncated,
        "source": a.source,
    }

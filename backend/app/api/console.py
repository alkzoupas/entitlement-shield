"""The analyst console (docs/architecture.md §5): two server-rendered Jinja/HTMX pages —
a case queue + detail, and the evaluation summary — plus the analyst-action form post.

Server-rendered and progressive: every page works as a plain GET/POST, and HTMX only
enhances the action form to swap the case body in place. The evaluation page reads a
**pre-computed** report artifact rather than querying ground truth, because the console
runs in the inference plane and must never hold the GT handle (locked invariant)."""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_reviewer_provider, get_session, templates
from app.api.presenters import (
    account_view,
    appeal_view,
    audit_view,
    cycle_view,
    decision_view,
    escalation_view,
    human_review_view,
    llm_review_view,
    snapshot_view,
)
from app.core.clock import now_utc
from app.core.config import settings
from app.core.decision_status import IllegalTransition
from app.models.enums import DecisionStatus, ProposedAction
from app.models.inference import (
    Account,
    Appeal,
    ClusterSnapshot,
    Decision,
    EntitlementCycle,
    EscalationRecord,
    HumanReview,
)
from app.services.audit import decision_audit_trail
from app.services.evidence import EvidenceConflict, decision_evidence
from app.services.human_review import (
    ANALYST_ACTIONS,
    MissingDecisionEvidence,
    UnknownAnalystAction,
    record_human_decision,
)
from app.services.reviewer import combine_effects, latest_reviews, review_decision

router = APIRouter(tags=["console"])


def _risk_floor(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise HTTPException(422, "risk_min must be a number from 0 to 1") from exc
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise HTTPException(422, "risk_min must be a number from 0 to 1")
    return result


@router.get("/", response_class=HTMLResponse)
def index() -> RedirectResponse:
    return RedirectResponse(url="/cases", status_code=307)


def _query_cases(
    session: Session,
    *,
    status: str | None,
    action: str | None,
    risk_min: float | None,
    limit: int,
) -> list[dict]:
    stmt = select(Decision).order_by(Decision.created_at.desc(), Decision.decision_id.desc())
    try:
        if status:
            stmt = stmt.where(Decision.status == DecisionStatus(status))
        if action:
            stmt = stmt.where(Decision.proposed_action == ProposedAction(action))
    except ValueError as exc:
        raise HTTPException(422, "Unknown decision status or action") from exc
    if risk_min is not None:
        stmt = stmt.where(Decision.risk_score >= risk_min)
    return [decision_view(d) for d in session.scalars(stmt.limit(limit)).all()]


@router.get("/cases", response_class=HTMLResponse)
def cases(
    request: Request,
    status: str | None = Query(default=None),
    action: str | None = Query(default=None),
    risk_min: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """The case queue, filterable by status / proposed action / minimum risk."""
    return templates.TemplateResponse(
        request,
        "cases.html",
        {
            "cases": _query_cases(
                session, status=status, action=action, risk_min=_risk_floor(risk_min), limit=limit
            ),
            "statuses": [s.value for s in DecisionStatus],
            "actions": [a.value for a in ProposedAction],
            "filters": {"status": status or "", "action": action or "", "risk_min": risk_min},
        },
    )


@router.get("/cases-table", response_class=HTMLResponse)
def cases_table(
    request: Request,
    status: str | None = Query(default=None),
    action: str | None = Query(default=None),
    risk_min: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """The queue table only — the HTMX target the filter form swaps in place."""
    return templates.TemplateResponse(
        request,
        "_cases_table.html",
        {
            "cases": _query_cases(
                session, status=status, action=action, risk_min=_risk_floor(risk_min), limit=limit
            )
        },
    )


def _case_context(session: Session, decision: Decision) -> dict:
    account = session.get(Account, decision.account_id)
    snapshot = (
        session.get(ClusterSnapshot, decision.cluster_snapshot_id)
        if decision.cluster_snapshot_id
        else None
    )
    cycles = session.scalars(
        select(EntitlementCycle)
        .where(EntitlementCycle.account_id == decision.account_id)
        .order_by(EntitlementCycle.cycle_start)
    ).all()
    reviews = session.scalars(
        select(HumanReview)
        .where(HumanReview.decision_id == decision.decision_id)
        .order_by(HumanReview.created_at)
    ).all()
    trail = decision_audit_trail(session, decision.decision_id)
    try:
        evidence = decision_evidence(session, decision.decision_id)
        evidence_error = None
    except EvidenceConflict as exc:
        evidence = None
        evidence_error = str(exc)
    legal_next = sorted(
        a for a, target in ANALYST_ACTIONS.items()
        if target in _legal_targets(decision.status)
        and (a != "reverse" or evidence is not None)
    )
    return {
        "decision": decision_view(decision),
        "account": account_view(account) if account else None,
        "snapshot": snapshot_view(snapshot),
        "cycles": [cycle_view(c) for c in cycles],
        "reviews": [human_review_view(r) for r in reviews],
        "trail": [audit_view(e) for e in trail],
        "evidence": evidence,
        "evidence_error": evidence_error,
        "legal_actions": legal_next,
        **_reviewer_context(session, decision),
    }


def _reviewer_narrative(samples: list) -> dict | None:
    """Phase 5 Track 2: a plain-English narrative for the console, sourced from
    the latest reviewer sample's free-text `ReviewerVerdict` fields (rationale,
    benign_explanations, missing_evidence, unverified_claims — schemas/reviewer.py:131).

    This is decision support only — the model's judgment/explanation, not an
    action. It carries no authority: nothing here changes `Decision.status` or
    bypasses the human reviewer, and the template renders it escaped, never
    `|safe`. Returns None when no reviewer run has happened yet, or the latest
    sample never produced a rationale (e.g. a terminal provider failure)."""
    if not samples:
        return None
    verdict = samples[-1].structured_output_json or {}
    rationale = verdict.get("rationale")
    if not rationale:
        return None
    return {
        "rationale": rationale,
        "benign_explanations": list(verdict.get("benign_explanations") or []),
        "missing_evidence": list(verdict.get("missing_evidence") or []),
        "unverified_claims": list(verdict.get("unverified_claims") or []),
        # Opus review follow-up: name the actual source under the prose (escaped
        # in the template) so a `FakeReviewerProvider` stand-in run is never
        # mistaken for a real model's judgment, and a real one is attributable.
        "provider": samples[-1].provider,
        "model_snapshot": samples[-1].model_snapshot,
    }


def _reviewer_context(session: Session, decision: Decision) -> dict:
    """P3.7 reviewer panel: the most recent run's 3 sample verdicts, the derived
    overall effect, the escalation record if one exists, and the latest appeal
    text (rendered escaped by the template — never via `|safe`)."""
    samples = latest_reviews(session, decision.decision_id)
    effects = tuple(s.derived_verdict for s in samples if s.derived_verdict is not None)
    overall_effect = combine_effects(effects) if effects else None  # type: ignore[arg-type]
    escalation = session.scalars(
        select(EscalationRecord)
        .where(EscalationRecord.decision_id == decision.decision_id)
        .order_by(EscalationRecord.created_at.desc())
    ).first()
    appeal = session.scalars(
        select(Appeal)
        .where(
            Appeal.account_id == decision.account_id,
            (Appeal.decision_id == decision.decision_id) | (Appeal.decision_id.is_(None)),
        )
        .order_by(Appeal.submitted_at.desc())
    ).first()
    return {
        "reviewer_samples": [llm_review_view(s) for s in samples],
        "reviewer_overall_effect": overall_effect,
        "reviewer_narrative": _reviewer_narrative(samples),
        "escalation": escalation_view(escalation) if escalation is not None else None,
        "appeal": appeal_view(appeal) if appeal is not None else None,
    }


def _legal_targets(status: DecisionStatus) -> set[DecisionStatus]:
    from app.core.decision_status import LEGAL_TRANSITIONS

    return set(LEGAL_TRANSITIONS[status])


@router.get("/cases/{decision_id}", response_class=HTMLResponse)
def case_detail(
    request: Request, decision_id: str, session: Session = Depends(get_session)
) -> HTMLResponse:
    decision = session.get(Decision, decision_id)
    if decision is None:
        return templates.TemplateResponse(
            request, "not_found.html", {"decision_id": decision_id}, status_code=404
        )
    return templates.TemplateResponse(
        request, "case_detail.html", {**_case_context(session, decision), "error": None}
    )


@router.post("/cases/{decision_id}/human-decision", response_class=HTMLResponse)
def human_decision(
    request: Request,
    decision_id: str,
    action: str = Form(...),
    reason: str | None = Form(default=None),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    """Apply an analyst action, then re-render the case. An illegal status move returns
    409 and a clear message; the decision is untouched (the status machine rejected it)."""
    decision = session.get(Decision, decision_id)
    if decision is None:
        return templates.TemplateResponse(
            request, "not_found.html", {"decision_id": decision_id}, status_code=404
        )
    error: str | None = None
    code = 200
    try:
        record_human_decision(session, decision, action=action, reason=reason)
        session.commit()
    except (
        IllegalTransition,
        UnknownAnalystAction,
        MissingDecisionEvidence,
        EvidenceConflict,
    ) as exc:
        session.rollback()
        error, code = str(exc), 409
    session.refresh(decision)
    return templates.TemplateResponse(
        request,
        "case_detail.html",
        {**_case_context(session, decision), "error": error},
        status_code=code,
    )


@router.post("/cases/{decision_id}/reviewer/run", response_class=HTMLResponse)
def reviewer_run(
    request: Request,
    decision_id: str,
    request_key: str | None = Form(default=None),
    run_id: str | None = Form(default=None),
    arm: str = Form(default="reviewer"),
    as_of: datetime | None = Form(default=None),
    session: Session = Depends(get_session),
    provider=Depends(get_reviewer_provider),
) -> HTMLResponse:
    """HTMX entry point for a reviewer run (P3.7). Delegates entirely to
    `reviewer.review_decision`, which is already idempotent on `request_key`
    (its `idempotency_key`) — this handler neither replays nor double-calls it,
    it just re-renders the case afterward. Not found -> 404; the decision itself
    is never mutated on a gated/replayed call."""
    decision = session.get(Decision, decision_id)
    if decision is None:
        return templates.TemplateResponse(
            request, "not_found.html", {"decision_id": decision_id}, status_code=404
        )
    review_decision(
        session,
        decision,
        provider=provider,
        run_id=run_id or uuid.uuid4().hex,
        idempotency_key=request_key or uuid.uuid4().hex,
        as_of=as_of or now_utc(),
        arm=arm,
    )
    session.commit()
    session.refresh(decision)
    return templates.TemplateResponse(
        request, "case_detail.html", {**_case_context(session, decision), "error": None}
    )


@router.get("/evaluation", response_class=HTMLResponse)
def evaluation(request: Request) -> HTMLResponse:
    """Render the pre-computed evaluation report (`make eval-json`). Dev-tier numbers are
    W-1 (NOT reportable); the template says so. No ground-truth query happens here."""
    report_path = Path(settings.eval_report_path)
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    return templates.TemplateResponse(
        request,
        "evaluation.html",
        {"report": report, "report_path": str(report_path)},
    )

"""API + console integration (docs/architecture.md §2/§5, P2.4).

Over a seeded in-memory world: the idempotent detector trigger, the case queue + detail
pages, an analyst action through the status machine (and an illegal one rejected), the
claim-time credit path, the evaluation page over a pre-computed report, and the standing
GT-boundary assertion — the API package never imports the ground-truth reader.
"""

from __future__ import annotations

import json
import pathlib
import uuid
from datetime import timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api import console
from app.api.app import create_app
from app.core.clock import now_utc
from app.db.engine import create_inference_schema
from app.db.ground_truth_store import create_ground_truth_schema
from app.db.ground_truth_writer import make_ground_truth_sink
from app.db.sqlite_fk import enable_sqlite_foreign_keys
from app.models.enums import (
    DecisionStatus,
    EmailDomainType,
    ProposedAction,
    VerificationLevel,
)
from app.models.inference import (
    Account,
    AuditEvent,
    ClusterSnapshot,
    Decision,
    DetectionEvidence,
    EntitlementCycle,
    EscalationRecord,
    LlmReview,
)
from app.services import feature_manifest
from app.services.evidence import associate_evidence, persist_evidence
from app.services.fake_reviewer import FakeMode, FakeReviewerProvider
from app.services.generator import WORLD_END, generate_world, persist_world

_HOLD_ACTIONS = (ProposedAction.HOLD_INCREMENTAL_CREDITS, ProposedAction.TEMPORARY_HOLD)
_REVIEWER_AS_OF = now_utc()


def _mem(schema) -> Engine:  # type: ignore[no-untyped-def]
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True
    )
    engine = enable_sqlite_foreign_keys(engine)
    schema(engine)
    return engine


@pytest.fixture
def env(tmp_path):  # type: ignore[no-untyped-def]
    inf_engine = enable_sqlite_foreign_keys(create_engine(f"sqlite:///{tmp_path / 'api.db'}"))
    create_inference_schema(inf_engine)
    gt_engine = _mem(create_ground_truth_schema)
    with (
        Session(inf_engine, future=True) as inf,
        Session(gt_engine, future=True) as gt,
    ):
        persist_world(generate_world(1, "min"), inf, make_ground_truth_sink(gt))
        inf.commit()
        gt.commit()
    app = create_app(inference_engine=inf_engine)
    return SimpleNamespace(client=TestClient(app), engine=inf_engine)


def _decisions(engine: Engine) -> list[Decision]:
    with Session(engine, future=True) as s:
        return list(s.scalars(select(Decision)))


def _run_detector(client: TestClient, run_id: str) -> dict:
    r = client.post("/api/detector/run", json={})
    assert r.status_code == 200, r.text
    return r.json()


def _detector_artifact_counts(engine: Engine) -> tuple[int, int, int, int]:
    with Session(engine, future=True) as s:
        return tuple(
            s.scalar(select(func.count()).select_from(model)) or 0
            for model in (ClusterSnapshot, DetectionEvidence, Decision, AuditEvent)
        )  # type: ignore[return-value]


@pytest.mark.parametrize("route", ["/cases", "/cases-table"])
def test_blank_and_invalid_form_filters(env, route) -> None:  # type: ignore[no-untyped-def]
    for query in ("status=&action=&risk_min=", "status=applied&action=&risk_min=",
                  "status=&action=temporary_hold&risk_min=", "risk_min=0.5"):
        assert env.client.get(f"{route}?{query}").status_code == 200
    for query in ("status=bogus", "action=bogus", "risk_min=no", "risk_min=nan",
                  "risk_min=1.1", "risk_min=-1"):
        assert env.client.get(f"{route}?{query}").status_code == 422


def test_api_detector_retains_feature_evidence(env) -> None:  # type: ignore[no-untyped-def]
    _run_detector(env.client, "evidence-api")
    decision = _decisions(env.engine)[0]
    response = env.client.get(f"/cases/{decision.decision_id}")
    assert response.status_code == 200
    assert "F-DEV-1" in response.text
    assert "Evidence hash" in response.text
    assert "Original cycle timeline" in response.text


@pytest.mark.parametrize("state", ["missing", "stale", "self_inconsistent"])
def test_api_detector_refuses_unverified_manifest_without_persisting_artifacts(
    env, tmp_path, monkeypatch, state
) -> None:  # type: ignore[no-untyped-def]
    """D-1: the HTTP path fails before clustering/materialization for every rejected
    freeze artifact. The self-inconsistent branch needs a regenerated-byte monkeypatch
    because otherwise it is correctly rejected earlier as ordinary byte drift."""
    manifest_path = tmp_path / "feature-manifest.json"
    monkeypatch.setattr(feature_manifest, "MANIFEST_PATH", manifest_path)
    if state == "stale":
        manifest_path.write_bytes(b"not the committed manifest")
    elif state == "self_inconsistent":
        content = feature_manifest.manifest_bytes()
        payload = json.loads(content)
        payload["content_hash"] = "0" * 64
        broken = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        manifest_path.write_bytes(broken)
        monkeypatch.setattr(feature_manifest, "manifest_bytes", lambda: broken)

    before = _detector_artifact_counts(env.engine)
    client = TestClient(env.client.app, raise_server_exceptions=False)
    response = client.post("/api/detector/run", json={})

    assert response.status_code == 500
    assert _detector_artifact_counts(env.engine) == before


def test_case_detail_degrades_on_evidence_conflict(env, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Real storage corruption is now blocked by both FK constraints and the F7-a seal
    triggers, so we monkeypatch decision_evidence to raise EvidenceConflict directly.
    This test is about the handler's degradation behavior on the read-only case page,
    not about reproducing real corruption."""
    _run_detector(env.client, "evidence-conflict")
    decision = _decisions(env.engine)[0]

    def _raise(session, decision_id):  # type: ignore[no-untyped-def]
        raise console.EvidenceConflict("evidence reference is broken")

    monkeypatch.setattr(console, "decision_evidence", _raise)

    response = env.client.get(f"/cases/{decision.decision_id}")
    assert response.status_code == 200
    assert "stored evidence reference is broken" in response.text
    assert 'value="reverse"' not in response.text


def test_reverse_action_degrades_on_evidence_conflict_instead_of_500ing(env, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Adversarial-review follow-up to `test_case_detail_degrades_on_evidence_conflict`:
    that test only proved the read-only GET render degrades gracefully. `human_review`
    imports `decision_evidence` into its OWN module namespace, so patching
    `console.decision_evidence` (as the GET test does) does not affect the POST
    `reverse` action, which calls `decision_evidence` via `record_human_decision` ->
    `human_review.decision_evidence` before `console._case_context` ever runs. This
    pins that the POST path also degrades (409, not 500) rather than crashing."""
    from app.services import human_review

    _run_detector(env.client, "r1")
    decision = _first_proposed_hold(env.engine)
    applied = env.client.post(
        f"/cases/{decision.decision_id}/human-decision",
        data={"action": "apply", "reason": "confirmed"},
    )
    assert applied.status_code == 200

    def _raise(session, decision_id):  # type: ignore[no-untyped-def]
        raise console.EvidenceConflict("evidence reference is broken")

    monkeypatch.setattr(human_review, "decision_evidence", _raise)

    response = env.client.post(
        f"/cases/{decision.decision_id}/human-decision",
        data={"action": "reverse", "reason": "attempted reversal"},
    )
    assert response.status_code == 409
    with Session(env.engine, future=True) as s:
        # The illegal/blocked reversal must not have mutated the decision.
        assert s.get(Decision, decision.decision_id).status is DecisionStatus.APPLIED


def test_detector_run_creates_decisions_and_is_idempotent(env) -> None:  # type: ignore[no-untyped-def]
    first = _run_detector(env.client, "r1")
    assert first["decisions_created"] > 0
    total_after_first = len(_decisions(env.engine))
    # A second run at the same as_of re-affirms the same idempotency keys (architecture §7).
    second = _run_detector(env.client, "r2")
    assert second["decisions_created"] == 0
    assert len(_decisions(env.engine)) == total_after_first


def test_case_queue_and_detail_render(env) -> None:  # type: ignore[no-untyped-def]
    _run_detector(env.client, "r1")
    queue = env.client.get("/cases")
    assert queue.status_code == 200 and "Case queue" in queue.text

    decision = _decisions(env.engine)[0]
    detail = env.client.get(f"/cases/{decision.decision_id}")
    assert detail.status_code == 200
    assert decision.account_id in detail.text

    missing = env.client.get("/cases/nope")
    assert missing.status_code == 404


def _first_proposed_hold(engine: Engine) -> Decision:
    for d in _decisions(engine):
        if d.status is DecisionStatus.PROPOSED and d.proposed_action in _HOLD_ACTIONS:
            return d
    pytest.skip("seeded 'min' world produced no proposed hold decision")


def test_analyst_apply_then_illegal_action(env) -> None:  # type: ignore[no-untyped-def]
    _run_detector(env.client, "r1")
    decision = _first_proposed_hold(env.engine)

    ok = env.client.post(
        f"/cases/{decision.decision_id}/human-decision",
        data={"action": "apply", "reason": "confirmed"},
    )
    assert ok.status_code == 200 and "applied" in ok.text
    with Session(env.engine, future=True) as s:
        assert s.get(Decision, decision.decision_id).status is DecisionStatus.APPLIED

    # applied → apply again is not a legal transition; rejected, decision untouched.
    bad = env.client.post(
        f"/cases/{decision.decision_id}/human-decision", data={"action": "apply"}
    )
    assert bad.status_code == 409 and "rejected" in bad.text.lower()
    with Session(env.engine, future=True) as s:
        assert s.get(Decision, decision.decision_id).status is DecisionStatus.APPLIED


def test_claim_grants_by_default_and_holds_under_applied_restriction(env) -> None:  # type: ignore[no-untyped-def]
    at = WORLD_END - timedelta(days=1)
    response = env.client.post("/api/detector/run", json={"as_of": at.isoformat()})
    assert response.status_code == 200
    with Session(env.engine) as s:
        decision = s.scalars(select(Decision).join(
            EntitlementCycle, Decision.account_id == EntitlementCycle.account_id
        ).where(
            Decision.status == DecisionStatus.PROPOSED,
            Decision.proposed_action.in_(_HOLD_ACTIONS),
            EntitlementCycle.cycle_start <= at, EntitlementCycle.cycle_end > at,
        ).order_by(Decision.created_at.desc())).first()
        assert decision is not None

    # An account with no standing restriction is granted, within budget.
    with Session(env.engine, future=True) as s:
        held_account = decision.account_id
        free_account = s.scalars(
            select(Account.account_id).join(EntitlementCycle).where(
                Account.account_id != held_account,
                EntitlementCycle.cycle_start <= at, EntitlementCycle.cycle_end > at,
            )
        ).first()
    granted = env.client.post("/api/claim", json={
        "account_id": free_account, "as_of": at.isoformat(), "request_key": "free-request",
    }).json()
    assert granted["outcome"] == "granted" and granted["within_budget"] is True
    assert granted["latency_ms"] < 300.0

    # Once the hold is APPLIED, the same account's claim is held.
    env.client.post(
        f"/cases/{decision.decision_id}/human-decision", data={"action": "apply"}
    )
    held = env.client.post("/api/claim", json={
        "account_id": held_account, "as_of": at.isoformat(), "request_key": "held-request",
    }).json()
    assert held["outcome"] == "held"
    assert held["standing_decision_id"] == decision.decision_id


def test_evaluation_page_with_and_without_report(env, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    missing = tmp_path / "absent.json"
    monkeypatch.setattr(console, "settings", SimpleNamespace(eval_report_path=str(missing)))
    empty = env.client.get("/evaluation")
    assert empty.status_code == 200 and "No evaluation report" in empty.text

    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "as_of": "2026-06-01T00:00:00+00:00", "split": "all", "rows": 10,
                "feature_manifest_hash": "abc123def4567890", "generated_at": "now",
                "arms": [
                    {"arm": "arm1", "label": "Stage 1 detector @0.60", "precision": 1.0,
                     "recall": 0.8, "fpr": 0.0, "pr_auc": 0.9, "wrongful_restriction": 0.0},
                ],
            }
        )
    )
    monkeypatch.setattr(console, "settings", SimpleNamespace(eval_report_path=str(report)))
    page = env.client.get("/evaluation")
    assert page.status_code == 200 and "Stage 1 detector @0.60" in page.text


def test_api_package_never_imports_ground_truth() -> None:
    api_dir = pathlib.Path(console.__file__).resolve().parent
    for path in api_dir.glob("*.py"):
        src = path.read_text()
        assert "ground_truth" not in src, f"{path.name} must not touch the GT plane"


# --- P3.7: POST /api/cases/{decision_id}/reviewer/run --------------------------


class _CountingProvider:
    """Wraps a provider to assert exactly how many times it was actually called."""

    def __init__(self, inner) -> None:  # type: ignore[no-untyped-def]
        self.inner = inner
        self.calls = 0

    def complete(self, request):  # type: ignore[no-untyped-def]
        self.calls += 1
        return self.inner.complete(request)


def _reviewer_env(tmp_path, *, mode: FakeMode = FakeMode.NORMAL):  # type: ignore[no-untyped-def]
    """A fresh inference-only store (no seeded world) plus a call-counting
    `FakeReviewerProvider`, for hand-built single-decision reviewer scenarios."""
    inf_engine = enable_sqlite_foreign_keys(
        create_engine(f"sqlite:///{tmp_path / 'reviewer-api.db'}")
    )
    create_inference_schema(inf_engine)
    provider = _CountingProvider(FakeReviewerProvider(mode=mode))
    app = create_app(inference_engine=inf_engine, reviewer_provider=provider)
    return SimpleNamespace(client=TestClient(app), engine=inf_engine, provider=provider)


def _hand_built_decision(
    engine: Engine,
    *,
    decision_id: str,
    proposed_action: ProposedAction,
    status: DecisionStatus = DecisionStatus.PROPOSED,
) -> str:
    """A single-member-cluster decision with a full review-case-compatible evidence
    payload (mirrors `test_reviewer.py`'s `_decision_with_evidence`). `FakeMode.NORMAL`'s
    rule-table bucket is a hash of the full structured payload (including `decision_id`
    and `account_id`), so each caller picks a `decision_id` already verified (see the
    call sites below) to land on the bucket its scenario needs. The account and cluster
    snapshot are namespaced by `decision_id` so multiple decisions can be hand-built on
    the same engine without a primary-key collision."""
    account_id = f"acct-{decision_id}"
    snapshot_id = f"snap-{decision_id}"
    with Session(engine, future=True) as session:
        session.add(
            Account(
                account_id=account_id,
                created_at=_REVIEWER_AS_OF - timedelta(days=120),
                email_domain_type=EmailDomainType.CONSUMER,
                verification_level=VerificationLevel.EMAIL,
            )
        )
        session.add(
            ClusterSnapshot(
                snapshot_id=snapshot_id,
                as_of=_REVIEWER_AS_OF,
                member_account_ids=[account_id],
                edges=[],
                evidence_summary={},
                content_hash=snapshot_id,
                created_at=_REVIEWER_AS_OF,
            )
        )
        session.flush()
        d = Decision(
            decision_id=decision_id,
            account_id=account_id,
            cluster_snapshot_id=snapshot_id,
            created_at=_REVIEWER_AS_OF,
            detector_version="det-1",
            policy_version="pol-1",
            risk_score=0.9,
            proposed_action=proposed_action,
            status=status,
            run_id="run-1",
            idempotency_key=uuid.uuid4().hex,
        )
        session.add(d)
        session.flush()
        payload = {
            "cluster_snapshot_id": snapshot_id,
            "cycle_id": "cycle-1",
            "as_of": _REVIEWER_AS_OF.isoformat(),
            "snapshot_hash": snapshot_id,
            "member_account_ids": [account_id],
            "edges": [],
            "qualified_stream_count": 2,
            "allowance": 25,
            "ambiguous": False,
            "excess": 0.5,
            "risk_score": 0.9,
            "proposed_action": proposed_action.value,
            "cap_reasons": [],
            "suppressed": False,
            "detector_version": "det-1",
            "policy_version": "pol-1",
            "feature_manifest_hash": "fmh-1",
            "rung": "S2",
            "contributions": [],
            "window_start": (_REVIEWER_AS_OF - timedelta(days=30)).isoformat(),
            "window_end": _REVIEWER_AS_OF.isoformat(),
            "timeline": [],
        }
        evidence = persist_evidence(session, payload)
        associate_evidence(session, d.decision_id, evidence.evidence_id)
        session.commit()
    return decision_id


def test_reviewer_run_404_on_unknown_decision(env) -> None:  # type: ignore[no-untyped-def]
    response = env.client.post(
        "/api/cases/nope/reviewer/run", json={"request_key": "k1"}
    )
    assert response.status_code == 404


def test_reviewer_run_double_post_same_request_key_calls_provider_exactly_3_times(
    tmp_path,
) -> None:
    """Architecture §7's literal 'must not double-charge the LLM': a repeat POST on
    the same `request_key` must replay through `review_decision`'s own idempotency
    handling rather than calling the provider again — 3 calls total, never 6.

    Uses an "uphold" case (decision_id "d1" + `hold_incremental_credits`, bucket 2
    under `FakeMode.NORMAL` — see `test_reviewer.py`'s fixture comment) rather than a
    reverse/escalate one: those mutate `decision.status`, so a second call would be
    stopped by `_gating_reason` before ever reaching the idempotency-key replay path
    this test targets."""
    rev = _reviewer_env(tmp_path)
    decision_id = _hand_built_decision(
        rev.engine, decision_id="d1", proposed_action=ProposedAction.HOLD_INCREMENTAL_CREDITS
    )

    first = rev.client.post(
        f"/api/cases/{decision_id}/reviewer/run", json={"request_key": "k1"}
    )
    assert first.status_code == 200, first.text
    assert rev.provider.calls == 3

    second = rev.client.post(
        f"/api/cases/{decision_id}/reviewer/run", json={"request_key": "k1"}
    )
    assert second.status_code == 200, second.text
    assert rev.provider.calls == 3, "provider must not be called again on replay"
    assert second.json()["overall_effect"] == first.json()["overall_effect"]
    assert second.json()["replayed"] is True

    rows = None
    with Session(rev.engine, future=True) as session:
        rows = session.scalars(
            select(LlmReview).where(LlmReview.decision_id == decision_id)
        ).all()
    assert len(rows) == 3, "no duplicate rows from the replayed run"


def test_reviewer_run_conflicting_payload_same_key_different_decision_is_409(
    tmp_path,
) -> None:
    rev = _reviewer_env(tmp_path)
    dec_a = _hand_built_decision(
        rev.engine, decision_id="dec-2", proposed_action=ProposedAction.LIGHT_FRICTION
    )
    dec_b = _hand_built_decision(
        rev.engine, decision_id="d1", proposed_action=ProposedAction.LIGHT_FRICTION
    )

    ok = rev.client.post(f"/api/cases/{dec_a}/reviewer/run", json={"request_key": "shared-key"})
    assert ok.status_code == 200, ok.text

    conflict = rev.client.post(
        f"/api/cases/{dec_b}/reviewer/run", json={"request_key": "shared-key"}
    )
    assert conflict.status_code == 409


def test_reviewer_run_conflicting_payload_same_key_different_arm_is_409(tmp_path) -> None:
    rev = _reviewer_env(tmp_path)
    decision_id = _hand_built_decision(
        rev.engine, decision_id="dec-2", proposed_action=ProposedAction.LIGHT_FRICTION
    )

    ok = rev.client.post(
        f"/api/cases/{decision_id}/reviewer/run",
        json={"request_key": "shared-key", "arm": "reviewer"},
    )
    assert ok.status_code == 200, ok.text

    conflict = rev.client.post(
        f"/api/cases/{decision_id}/reviewer/run",
        json={"request_key": "shared-key", "arm": "reviewer_alt"},
    )
    assert conflict.status_code == 409


def test_reviewer_run_end_to_end_auto_reverses_through_api_and_console(tmp_path) -> None:
    rev = _reviewer_env(tmp_path)
    # bucket 0 (ALLOW) under FakeMode.NORMAL for this decision_id/light_friction shape.
    decision_id = _hand_built_decision(
        rev.engine, decision_id="dec-1", proposed_action=ProposedAction.LIGHT_FRICTION
    )

    response = rev.client.post(
        f"/api/cases/{decision_id}/reviewer/run", json={"request_key": "k1"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["overall_effect"] == "reverse"
    assert body["applied"] is True

    with Session(rev.engine, future=True) as session:
        assert session.get(Decision, decision_id).status is DecisionStatus.REVERSED

    detail = rev.client.get(f"/cases/{decision_id}")
    assert detail.status_code == 200
    assert "reversed" in detail.text


def test_reviewer_run_end_to_end_escalates_through_api_and_console(tmp_path) -> None:
    rev = _reviewer_env(tmp_path)
    decision_id = _hand_built_decision(
        rev.engine,
        decision_id="dec-2",
        proposed_action=ProposedAction.HOLD_INCREMENTAL_CREDITS,
    )

    response = rev.client.post(
        f"/api/cases/{decision_id}/reviewer/run", json={"request_key": "k1"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["overall_effect"] == "escalate"
    assert body["applied"] is False

    with Session(rev.engine, future=True) as session:
        assert session.get(Decision, decision_id).status is DecisionStatus.UNDER_REVIEW
        records = session.scalars(
            select(EscalationRecord).where(EscalationRecord.decision_id == decision_id)
        ).all()
        assert len(records) == 1

    detail = rev.client.get(f"/cases/{decision_id}")
    assert detail.status_code == 200
    assert "under_review" in detail.text
    assert "escalated" in detail.text


def test_console_reviewer_panel_renders_appeal_text_escaped(tmp_path) -> None:
    """C-3-adjacent UI hardening: an appeal containing a script tag must reach the
    rendered page HTML-escaped, never as live markup — confirming Jinja's default
    autoescaping is actually active for `_reviewer_panel.html`, not assumed."""
    from app.services.review_case import record_appeal

    rev = _reviewer_env(tmp_path)
    decision_id = _hand_built_decision(
        rev.engine, decision_id="dec-2", proposed_action=ProposedAction.HOLD_INCREMENTAL_CREDITS
    )
    malicious = "<script>alert('pwn')</script>"
    with Session(rev.engine, future=True) as session:
        record_appeal(
            session,
            account_id=f"acct-{decision_id}",
            text=malicious,
            source="portal",
            submitted_at=_REVIEWER_AS_OF,
            decision_id=decision_id,
        )
        session.commit()

    run = rev.client.post(f"/api/cases/{decision_id}/reviewer/run", json={"request_key": "k1"})
    assert run.status_code == 200, run.text

    detail = rev.client.get(f"/cases/{decision_id}")
    assert detail.status_code == 200
    assert malicious not in detail.text
    assert "&lt;script&gt;" in detail.text


def test_reviewer_context_has_no_narrative_before_any_reviewer_run(tmp_path) -> None:
    """Phase 5 Track 2: `_reviewer_context`'s new `reviewer_narrative` key must be
    None (panel shows nothing) when the decision has never been reviewed —
    the no-reviewer-run-yet case the narrative must handle gracefully."""
    rev = _reviewer_env(tmp_path)
    decision_id = _hand_built_decision(
        rev.engine, decision_id="dec-3", proposed_action=ProposedAction.HOLD_INCREMENTAL_CREDITS
    )
    with Session(rev.engine, future=True) as session:
        decision = session.get(Decision, decision_id)
        ctx = console._reviewer_context(session, decision)
    assert ctx["reviewer_narrative"] is None


def test_reviewer_context_narrative_sourced_from_latest_sample_and_renders_escaped(
    tmp_path,
) -> None:
    """Phase 5 Track 2: `_reviewer_context["reviewer_narrative"]` is populated from
    the latest reviewer sample's free-text `ReviewerVerdict` fields (rationale,
    benign_explanations, missing_evidence, unverified_claims), and
    `_reviewer_panel.html` renders that prose escaped — never `|safe` — matching
    the appeal-text hardening above. `LlmReview` rows are append-only
    (`db/immutability.py`), so the malicious content is injected into the
    provider's response BEFORE it is persisted, rather than by mutating a
    written row."""
    import dataclasses
    import json

    malicious_rationale = "shared device pattern <script>alert('narrative')</script>"

    class _MaliciousNarrativeProvider:
        """Wraps `FakeReviewerProvider` (NORMAL mode) so the persisted verdict
        stays schema-valid (citation, action, confidence) but carries malicious
        free-text in exactly the fields the new narrative reads."""

        def __init__(self, inner) -> None:  # type: ignore[no-untyped-def]
            self.inner = inner

        def complete(self, request):  # type: ignore[no-untyped-def]
            response = self.inner.complete(request)
            verdict = json.loads(response.text)
            verdict["rationale"] = malicious_rationale
            verdict["benign_explanations"] = ["<b>shared</b> household device"]
            verdict["missing_evidence"] = ["device-level verification"]
            verdict["unverified_claims"] = ["claimed shared custody"]
            return dataclasses.replace(response, text=json.dumps(verdict))

    inf_engine = enable_sqlite_foreign_keys(
        create_engine(f"sqlite:///{tmp_path / 'reviewer-narrative.db'}")
    )
    create_inference_schema(inf_engine)
    provider = _MaliciousNarrativeProvider(FakeReviewerProvider(mode=FakeMode.NORMAL))
    app = create_app(inference_engine=inf_engine, reviewer_provider=provider)
    client = TestClient(app)

    decision_id = _hand_built_decision(
        inf_engine, decision_id="dec-4", proposed_action=ProposedAction.HOLD_INCREMENTAL_CREDITS
    )

    run = client.post(f"/api/cases/{decision_id}/reviewer/run", json={"request_key": "k1"})
    assert run.status_code == 200, run.text

    with Session(inf_engine, future=True) as session:
        reviews = session.scalars(
            select(LlmReview).where(LlmReview.decision_id == decision_id)
        ).all()
        assert reviews, "reviewer run should have persisted LlmReview rows"
        assert all(
            (r.structured_output_json or {}).get("rationale") == malicious_rationale
            for r in reviews
        )

        decision = session.get(Decision, decision_id)
        ctx = console._reviewer_context(session, decision)
        narrative = ctx["reviewer_narrative"]
        assert narrative is not None
        assert narrative["rationale"] == malicious_rationale
        assert narrative["benign_explanations"] == ["<b>shared</b> household device"]
        assert narrative["missing_evidence"] == ["device-level verification"]
        assert narrative["unverified_claims"] == ["claimed shared custody"]

    detail = client.get(f"/cases/{decision_id}")
    assert detail.status_code == 200
    assert "<script>alert('narrative')</script>" not in detail.text
    assert "&lt;script&gt;alert(&#39;narrative&#39;)&lt;/script&gt;" in detail.text
    assert "<b>shared</b> household device" not in detail.text
    assert "&lt;b&gt;shared&lt;/b&gt; household device" in detail.text

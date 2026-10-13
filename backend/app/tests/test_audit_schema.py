"""The typed audit union parses valid payloads and rejects malformed ones."""

import pytest
from pydantic import ValidationError

from app.schemas.audit import (
    DecisionProposed,
    DecisionStatusChanged,
    parse_audit_payload,
)


def test_parses_decision_proposed() -> None:
    payload = parse_audit_payload(
        {
            "event_name": "decision_proposed",
            "entity_type": "decision",
            "entity_id": "d1",
            "actor_type": "system",
            "decision_id": "d1",
            "risk_score": 0.7,
            "proposed_action": "hold_incremental_credits",
            "run_id": "run-1",
        }
    )
    assert isinstance(payload, DecisionProposed)
    assert payload.risk_score == 0.7


def test_parses_decision_status_changed() -> None:
    payload = parse_audit_payload(
        {
            "event_name": "decision_status_changed",
            "entity_type": "decision",
            "entity_id": "d1",
            "actor_type": "analyst",
            "decision_id": "d1",
            "from_status": "applied",
            "to_status": "reversed",
        }
    )
    assert isinstance(payload, DecisionStatusChanged)
    assert payload.reason is None


def test_rejects_unknown_event_name() -> None:
    with pytest.raises(ValidationError):
        parse_audit_payload(
            {"event_name": "not_a_real_event", "entity_type": "x", "entity_id": "y"}
        )


def test_rejects_missing_required_field() -> None:
    with pytest.raises(ValidationError):
        parse_audit_payload(
            {  # missing decision_id / risk_score / proposed_action / run_id
                "event_name": "decision_proposed",
                "entity_type": "decision",
                "entity_id": "d1",
                "actor_type": "system",
            }
        )


def test_rejects_unexpected_extra_field() -> None:
    # extra="forbid": a leaked field (e.g. sealed_label) is a hard error (review 7).
    with pytest.raises(ValidationError):
        parse_audit_payload(
            {
                "event_name": "decision_proposed",
                "entity_type": "decision",
                "entity_id": "d1",
                "actor_type": "system",
                "decision_id": "d1",
                "risk_score": 0.7,
                "proposed_action": "hold_incremental_credits",
                "run_id": "run-1",
                "sealed_label": "abusive",  # not part of the schema
            }
        )


def test_rejects_entity_id_decision_id_mismatch() -> None:
    with pytest.raises(ValidationError):
        parse_audit_payload(
            {
                "event_name": "decision_proposed",
                "entity_type": "decision",
                "entity_id": "d1",
                "actor_type": "system",
                "decision_id": "d2",  # != entity_id
                "risk_score": 0.7,
                "proposed_action": "hold_incremental_credits",
                "run_id": "run-1",
            }
        )


def test_rejects_bad_enum_value() -> None:
    with pytest.raises(ValidationError):
        parse_audit_payload(
            {
                "event_name": "decision_status_changed",
                "entity_type": "decision",
                "entity_id": "d1",
                "actor_type": "analyst",
                "decision_id": "d1",
                "from_status": "applied",
                "to_status": "not_a_status",
            }
        )

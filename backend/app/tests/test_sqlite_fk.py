"""SQLite foreign-key enforcement is ON (review 5 P1).

Declared foreign keys are inert on SQLite unless `PRAGMA foreign_keys=ON` runs on
every connection. With enforcement enabled an orphan child insert fails; a valid
insert still succeeds.
"""

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.clock import now_utc
from app.models.enums import EmailDomainType, EventType, VerificationLevel
from app.models.inference import Account, AccountEvent


def test_pragma_foreign_keys_is_on(inference_engine: Engine) -> None:
    with Session(inference_engine, future=True) as session:
        assert session.scalar(text("PRAGMA foreign_keys")) == 1


def test_orphan_child_insert_is_rejected(inference_engine: Engine) -> None:
    with Session(inference_engine, future=True) as session:
        session.add(
            AccountEvent(
                event_id="e1",
                account_id="does-not-exist",
                timestamp=now_utc(),
                event_type=EventType.LOGIN,
                credits_amount=None,
                endpoint_cost_tier=None,
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()


def test_valid_child_insert_succeeds(inference_engine: Engine) -> None:
    with Session(inference_engine, future=True) as session:
        session.add(
            Account(
                account_id="a1",
                created_at=now_utc(),
                email_domain_type=EmailDomainType.CONSUMER,
                verification_level=VerificationLevel.EMAIL,
            )
        )
        session.flush()
        session.add(
            AccountEvent(
                event_id="e1",
                account_id="a1",
                timestamp=now_utc(),
                event_type=EventType.LOGIN,
                credits_amount=None,
                endpoint_cost_tier=None,
            )
        )
        session.flush()  # no error

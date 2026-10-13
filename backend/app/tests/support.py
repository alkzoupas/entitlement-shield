"""Hand-built inference-store fixtures for Step 4 feature/banding tests.

Small, targeted record builders (accounts, events, linkage, cycles) so a test can
state exactly the pattern it exercises, rather than generating a whole world.
Foreign keys are enforced (conftest), so accounts are flushed before children.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from app.models.enums import (
    EmailDomainType,
    EndpointCostTier,
    EventType,
    Reliability,
    SignalType,
    VerificationLevel,
)
from app.models.inference import (
    Account,
    AccountEvent,
    EntitlementCycle,
    LinkageObservation,
)

AS_OF = datetime(2026, 6, 1, tzinfo=UTC)


class Builder:
    """Accumulates rows and flushes them, keeping a monotonic id counter."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self._n = 0

    def _id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}-{self._n:05d}"

    def account(
        self,
        account_id: str,
        *,
        domain: EmailDomainType = EmailDomainType.CONSUMER,
        verification: VerificationLevel = VerificationLevel.PHONE,
        created: datetime | None = None,
    ) -> str:
        self.session.add(
            Account(
                account_id=account_id,
                created_at=created or (AS_OF - timedelta(days=120)),
                email_domain_type=domain,
                verification_level=verification,
            )
        )
        self.session.flush()
        return account_id

    def event(
        self,
        account_id: str,
        ts: datetime,
        event_type: EventType,
        *,
        amount: int | None = None,
        tier: EndpointCostTier | None = None,
    ) -> None:
        self.session.add(
            AccountEvent(
                event_id=self._id("e"),
                account_id=account_id,
                timestamp=ts,
                event_type=event_type,
                credits_amount=amount,
                endpoint_cost_tier=tier,
            )
        )

    def consume(self, account_id: str, ts: datetime, amount: int) -> None:
        self.event(
            account_id,
            ts,
            EventType.CONSUME_CREDITS,
            amount=amount,
            tier=EndpointCostTier.HIGH,
        )

    def claim(self, account_id: str, ts: datetime, amount: int = 25) -> None:
        self.event(account_id, ts, EventType.CLAIM_CREDITS, amount=amount)

    def observation(
        self,
        account_id: str,
        signal: SignalType,
        value: str,
        *,
        reliability: Reliability = Reliability.HIGH,
        ts: datetime | None = None,
    ) -> None:
        self.session.add(
            LinkageObservation(
                observation_id=self._id("o"),
                account_id=account_id,
                timestamp=ts or (AS_OF - timedelta(days=30)),
                signal_type=signal,
                signal_value_hash=value,
                reliability=reliability,
            )
        )

    def cycle(
        self,
        account_id: str,
        start: datetime,
        *,
        days: int = 30,
        granted: int = 25,
        consumed: int = 0,
        expired: int = 0,
    ) -> str:
        cid = self._id("c")
        self.session.add(
            EntitlementCycle(
                cycle_id=cid,
                account_id=account_id,
                cycle_start=start,
                cycle_end=start + timedelta(days=days),
                credits_granted=granted,
                credits_consumed=consumed,
                credits_expired=expired,
            )
        )
        return cid

    def flush(self) -> None:
        self.session.flush()

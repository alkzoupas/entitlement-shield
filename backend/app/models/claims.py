"""Synthetic credit operations, isolated from detector feature/schema semantics."""

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import InferenceBase


class SyntheticClaim(InferenceBase):
    __tablename__ = "synthetic_claims"
    __table_args__ = (
        CheckConstraint("requested_amount > 0"),
        CheckConstraint("granted_amount >= 0 AND held_amount >= 0"),
        CheckConstraint("granted_amount + held_amount = requested_amount"),
        CheckConstraint("hold_state IN ('none', 'pending', 'confirmed')"),
        CheckConstraint(
            "(held_amount = 0 AND hold_state = 'none') OR "
            "(granted_amount = 0 AND hold_state IN ('pending', 'confirmed'))"
        ),
    )

    claim_id: Mapped[str] = mapped_column(String, primary_key=True)
    request_key: Mapped[str] = mapped_column(String, unique=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.account_id"), index=True)
    cycle_id: Mapped[str] = mapped_column(ForeignKey("entitlement_cycles.cycle_id"), index=True)
    requested_amount: Mapped[int] = mapped_column(Integer)
    granted_amount: Mapped[int] = mapped_column(Integer)
    held_amount: Mapped[int] = mapped_column(Integer)
    decision_id: Mapped[str | None] = mapped_column(ForeignKey("decisions.decision_id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    deferred: Mapped[bool]
    # A pending hold has no authority until response arbitration wins.
    hold_state: Mapped[str] = mapped_column(String, default="none")
    restored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ClaimAllowanceLock(InferenceBase):
    """One inference-wide writer row serializes allowance reads and credit changes.

    A global scope intentionally serializes overlapping snapshots too. Phase 2 has
    one synthetic store; finer scope locks require proving membership disjointness.
    """

    __tablename__ = "claim_allowance_locks"
    scope: Mapped[str] = mapped_column(String, primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, default=0)

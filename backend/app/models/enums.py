"""Controlled vocabularies. These mirror the event schema in docs/tier2-brief.md
and the policy action ladder (docs/policy.md §7). Stored as VARCHAR + CHECK
(native_enum=False in the ORM) so they are portable to Postgres.
"""

from enum import StrEnum


class EmailDomainType(StrEnum):
    CONSUMER = "consumer"
    DISPOSABLE = "disposable"
    EDUCATION = "education"
    ENTERPRISE = "enterprise"


class VerificationLevel(StrEnum):
    NONE = "none"
    EMAIL = "email"
    PHONE = "phone"
    ENHANCED = "enhanced"


class EventType(StrEnum):
    SIGNUP = "signup"
    CLAIM_CREDITS = "claim_credits"
    CONSUME_CREDITS = "consume_credits"
    LOGIN = "login"
    APPEAL = "appeal"
    VERIFICATION = "verification"  # a passed/successful verification
    VERIFICATION_FAILED = "verification_failed"  # a failed verification attempt (F-VER-1)


class EndpointCostTier(StrEnum):
    LOW = "low"
    HIGH = "high"


class SignalType(StrEnum):
    # Resource-sharing (form graph edges — docs/policy.md §5):
    STABLE_DEVICE = "stable_device"
    PAYMENT_TOKEN = "payment_token"
    RECOVERY_CONTACT = "recovery_contact"
    # Context-only (NEVER form an edge — docs/policy.md §5, features.md §3.3):
    NETWORK_GROUP = "network_group"
    BEHAVIORAL_TEMPLATE = "behavioral_template"


class Reliability(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActorType(StrEnum):
    SYSTEM = "system"
    LLM = "llm"
    ANALYST = "analyst"


class DecisionStatus(StrEnum):
    PROPOSED = "proposed"
    UNDER_REVIEW = "under_review"
    APPLIED = "applied"
    REVERSED = "reversed"
    SUPERSEDED = "superseded"


class ProposedAction(StrEnum):
    """Policy action ladder (docs/policy.md §7), least to most severe."""

    ALLOW = "allow"
    LIGHT_FRICTION = "light_friction"
    HOLD_INCREMENTAL_CREDITS = "hold_incremental_credits"
    TEMPORARY_HOLD = "temporary_hold"


class CapacityBand(StrEnum):
    """Labels over qualified_stream_count (docs/policy.md §4.5)."""

    SINGLE = "single"
    SMALL_GROUP = "small_group"
    LARGE_SHARED = "large_shared"


class GroundTruthLabel(StrEnum):
    """Sealed labels — live ONLY in the ground-truth store (docs/tier2-brief.md)."""

    ABUSIVE = "abusive"
    LEGITIMATE = "legitimate"
    AMBIGUOUS = "ambiguous"


class AppealTruthKind(StrEnum):
    """Sealed appeal truth (Tier 4) — lives ONLY in the ground-truth store."""

    HONEST = "honest"
    DISHONEST = "dishonest"
    MANIPULATION = "manipulation"

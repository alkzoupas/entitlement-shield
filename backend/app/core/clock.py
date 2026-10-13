"""Single source of 'now'. Timezone-aware UTC everywhere (no naive datetimes)."""

from datetime import UTC, datetime


def now_utc() -> datetime:
    return datetime.now(UTC)

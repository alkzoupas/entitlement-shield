"""P5.5 code-side evidence summary for the Jev adapter (`docs/decisions.md` P5-5).

Rewrites the reviewer's rendered `StructuredEvidenceSegment` text into a semantic
`state` object: feature ids → their `docs/features.md` names, numbers → worded
buckets, timestamps → relative gaps, account ids → "account N". Jev's documented
weak spots are numbers, date comparison and large irrelevant state (jev-1.13
jaggedness page); this moves that arithmetic into code.

Designed from the evidence *schema* and the public feature definitions only — the
`dev_arm4`/M-12 world generator was deliberately not consulted, so no fixture-
specific discriminating signal is hand-encoded. Every account gets the same
generic descriptors. Pure functions, no I/O; parses the fixed line format
`prompt_builder._format_structured_evidence` renders.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

SUMMARY_VERSION = "summary-v1"

# docs/features.md §3 — name, family, polarity. Public vocabulary, not fixture data.
FEATURES: dict[str, tuple[str, str, str]] = {
    "F-DEV-1": ("Strong shared device (3+ accounts, high reliability)", "device", "suspicious"),
    "F-DEV-2": ("Weak shared device", "device", "suspicious"),
    "F-PAY-1": ("Payment token reused across accounts", "payment", "suspicious"),
    "F-CON-1": ("Shared recovery contact", "contact/domain", "suspicious"),
    "F-DOM-1": ("Disposable email domain", "contact/domain", "suspicious"),
    "F-VER-1": ("Weak or failed verification", "verification", "suspicious"),
    "F-CYC-1": ("Credits claimed exceed the allowance", "cycle", "suspicious"),
    "F-CYC-2": ("Usage windows tile sequentially with near-zero overlap", "cycle", "suspicious"),
    "F-CYC-3": (
        "A linked account signed up soon after another exhausted its credits",
        "cycle",
        "suspicious",
    ),
    "F-CYC-4": ("Same pattern was upheld in a prior cycle", "cycle", "suspicious"),
    "F-DOM-2": ("Education or enterprise email domain", "domain", "benign"),
    "F-AGE-1": ("Long-lived accounts with routine activity", "age", "benign"),
}

_TIMELINE = re.compile(
    r"^\s+- (?P<ts>\S+) (?P<acct>\S+) (?P<etype>\S+) "
    r"credits_amount=(?P<credits>-?\d+) event_ids=(?P<ids>\S+)$"
)
_CONTRIB = re.compile(r"^\s+- (?P<fid>F-[A-Z]+-\d+): .*fired=(?P<fired>True|False)")
_EDGE = re.compile(r"^\s+- (\S+) <-> (\S+) \(evidence_count=(\d+)\)$")


@dataclass
class _Parsed:
    scalars: dict[str, str] = field(default_factory=dict)
    edges: list[tuple[str, str, int]] = field(default_factory=list)
    fired: list[str] = field(default_factory=list)
    has_contributions: bool = False
    events: list[tuple[datetime, str, str, int, list[str]]] = field(default_factory=list)


def _parse(text: str) -> _Parsed:
    parsed = _Parsed()
    section = "scalars"
    for line in text.splitlines():
        if not line.strip() or line.startswith("## "):
            continue
        if line == "edges:":
            section = "edges"
            continue
        if line == "contributions:":
            section = "contributions"
            parsed.has_contributions = True
            continue
        if line.startswith("timeline"):
            section = "timeline"
            continue
        if section == "scalars" and ": " in line:
            key, value = line.split(": ", 1)
            parsed.scalars[key.strip()] = value.strip()
        elif section == "edges" and (m := _EDGE.match(line)):
            parsed.edges.append((m.group(1), m.group(2), int(m.group(3))))
        elif section == "contributions" and (m := _CONTRIB.match(line)):
            if m.group("fired") == "True":
                parsed.fired.append(m.group("fid"))
        elif section == "timeline" and (m := _TIMELINE.match(line)):
            parsed.events.append(
                (
                    datetime.fromisoformat(m.group("ts")),
                    m.group("acct"),
                    m.group("etype"),
                    int(m.group("credits")),
                    m.group("ids").split(","),
                )
            )
    return parsed


def gap_words(hours: float) -> str:
    if hours < 1:
        return "under an hour"
    if hours < 24:
        return "within a day"
    if hours < 72:
        return "1 to 3 days"
    if hours < 168:
        return "3 to 7 days"
    if hours < 672:
        return "1 to 4 weeks"
    return "more than 4 weeks"


def link_words(n: int) -> str:
    if n == 0:
        return "no links"
    if n == 1:
        return "one link"
    if n <= 3:
        return "a few links (2-3)"
    if n <= 10:
        return "several links (4-10)"
    return "many links (over 10)"


def count_words(n: int) -> str:
    if n == 0:
        return "none"
    if n == 1:
        return "once"
    if n <= 3:
        return "a few times (2-3)"
    if n <= 10:
        return "several times (4-10)"
    return "many times (over 10)"


def members_words(n: int) -> str:
    if n <= 1:
        return "a single account"
    if n == 2:
        return "two accounts"
    if n <= 5:
        return "a small group (3-5 accounts)"
    return "a large group (6 or more accounts)"


def excess_words(raw: str) -> str:
    try:
        excess = float(raw)
    except ValueError:
        return "unknown"
    if excess <= 0:
        return "within the allowance"
    if excess <= 0.25:
        return "slightly over the allowance (up to 25% over)"
    if excess <= 1.0:
        return "well over the allowance (25-100% over)"
    return "more than double the allowance"


def usage_words(claimed: int, consumed: int) -> str:
    if claimed == 0:
        return "no credits claimed"
    ratio = consumed / claimed
    if ratio == 0:
        return "claimed credits but used none"
    if ratio < 0.5:
        return "used under half of claimed credits"
    if ratio < 0.9:
        return "used most of claimed credits"
    return "used all or nearly all claimed credits"


def _signup_words(signup: datetime | None, start: datetime | None) -> str:
    if start is None:
        return "no events observed"
    if signup is None:
        return "before the observed window"
    if signup == start:
        return "at the start of the observed window"
    return f"{gap_words((signup - start).total_seconds() / 3600)} after the observed window began"


def _account_names(parsed: _Parsed) -> dict[str, str]:
    order: list[str] = []
    for _, acct, *_rest in sorted(parsed.events, key=lambda e: e[0]):
        if acct not in order:
            order.append(acct)
    members = [m.strip() for m in parsed.scalars.get("member_account_ids", "").split(",")]
    order += [m for m in members if m and m not in order]
    flagged = parsed.scalars.get("account_id")
    return {
        acct: f"account {i}" + (" (flagged)" if acct == flagged else "")
        for i, acct in enumerate(order, start=1)
    }


def summarize(text: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Return `(state, key_descriptions)`. `key_descriptions` maps each evidence-key
    label (`scalar:band`, `contribution:F-DEV-2`, `timeline_event:<id>`) to a short
    semantic description used as its Choice criterion."""
    parsed = _parse(text)
    names = _account_names(parsed)
    band = parsed.scalars.get("band", "unknown").replace("_", " ")
    cap_reasons = parsed.scalars.get("cap_reasons", "(none)")
    caps = "none" if cap_reasons == "(none)" else cap_reasons.replace("_", " ")
    excess = excess_words(parsed.scalars.get("excess", ""))

    start = min((e[0] for e in parsed.events), default=None)
    per_account: dict[str, dict[str, Any]] = {}
    for acct in names:
        evs = sorted((e for e in parsed.events if e[1] == acct), key=lambda e: e[0])
        kinds = Counter(e[2] for e in evs)
        signup = next((e[0] for e in evs if e[2] == "signup"), None)
        first_claim = next((e[0] for e in evs if e[2] == "claim_credits"), None)
        claimed = sum(e[3] for e in evs if e[2] == "claim_credits")
        consumed = sum(e[3] for e in evs if e[2] == "consume_credits")
        entry: dict[str, Any] = {
            "signed_up": _signup_words(signup, start),
            "logins": count_words(kinds.get("login", 0)),
            "credit_claims": count_words(kinds.get("claim_credits", 0)),
            "verification_failures": count_words(kinds.get("verification_failed", 0)),
            "credit_use": usage_words(claimed, consumed),
        }
        if signup is not None and first_claim is not None and first_claim >= signup:
            entry["first_claim_after_signup"] = gap_words(
                (first_claim - signup).total_seconds() / 3600
            )
        per_account[names[acct]] = entry

    signups = sorted(e[0] for e in parsed.events if e[2] == "signup")
    if len(signups) >= 2:
        spread = gap_words((signups[-1] - signups[0]).total_seconds() / 3600)
        signup_pattern = f"{len(signups)} member signups observed, spread over {spread}"
    elif len(signups) == 1:
        signup_pattern = "one member signup observed"
    else:
        signup_pattern = "no member signups in the observed window"

    state: dict[str, Any] = {
        "cluster": {
            "members": members_words(len(names)),
            "shared_signal_links": link_words(len(parsed.edges)),
            "signup_pattern": signup_pattern,
        },
        "capacity_band": band,
        "allowance_use": excess,
        "capacity_caps_applied": caps,
        "accounts": per_account,
    }
    if parsed.has_contributions:
        state["detector_signals_fired"] = [
            f"{FEATURES[f][0]} ({FEATURES[f][1]}; {FEATURES[f][2]})"
            if f in FEATURES else f
            for f in parsed.fired
        ] or ["none"]

    descriptions: dict[str, str] = {
        "scalar:band": f"capacity band: {band}",
        "scalar:allowance": "the cluster's credit allowance",
        "scalar:excess": f"allowance use: {excess}",
    }
    for f in parsed.fired:
        if f in FEATURES:
            descriptions[f"contribution:{f}"] = f"detector signal: {FEATURES[f][0]}"
    for ts, acct, etype, _credits, ids in parsed.events:
        when = (
            "at the start of the window"
            if start is None or ts == start
            else f"{gap_words((ts - start).total_seconds() / 3600)} into the window"
        )
        label = f"{names.get(acct, 'an account')} {etype.replace('_', ' ')}, {when}"
        for event_id in ids:
            descriptions[f"timeline_event:{event_id}"] = label
    return state, descriptions


__all__ = ["FEATURES", "SUMMARY_VERSION", "summarize"]

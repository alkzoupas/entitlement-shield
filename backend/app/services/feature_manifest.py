"""The machine-checkable freeze artifact (features.md §6.2). Step 8.

`features-frozen` must lock **everything that decides whether a feature fires**
(features.md §6.1), not merely which feature rows exist — a weight, curve,
predicate parameter, the 0.05 threshold, or the masking/aggregation version can
each move a contribution across the firing line and thereby change firing
families, the policy gate, the ablation, and the anticipated/unanticipated
classification (external review 4, P0-2).

This module builds that lock **from the live code**, in three layers so nothing
firing-affecting can move while the hash stays constant (Step-8 adversarial review,
findings 1–3):

1. **Per-feature §6.1 fields** — id/inputs/sign/family/rung, the predicate and its
   tuned parameters, scaling/caps, missing-value semantics, the 0.05 threshold, the
   weight — read from live `features` constants.
2. **`firing_config`** — every firing-affecting constant OUTSIDE `features.py` that
   still decides whether a feature fires: the whole `banding` block (window, overlap,
   rate thresholds, diurnal distances, per-stream allowance, ambiguity fraction, band
   cut points → the F-CYC-1 allowance path and F-CYC-2 window) and the `clustering`
   block (hub threshold, max component size, decay half-life/drop, reliability
   weights, resource-signal set → the device/payment/contact preprocessing).
   `banding.py`/`clustering.py` name these constants precisely so this manifest can
   capture them (their module docstrings say so). Read live → a change moves the hash.
3. **`firing_code_fingerprint`** — a SHA-256 over the full source of the three firing
   modules (`features.py`, `banding.py`, `clustering.py`). This is the belt to the
   config braces: even a change with no constant edit — the F-CYC-1 `min(1.0,excess)`
   curve, a predicate's `>=`, an emitted family/rung, the edge rule — moves the
   fingerprint and so the hash. It is deliberately over-conservative (a comment edit
   in a firing module also re-freezes); for an immutable v1 that is the correct,
   fail-closed trade — no semantic change can regenerate byte-identically.

`build_manifest()` returns the manifest dict; `manifest_bytes()` is its byte-canonical
serialization, and `config/feature-manifest.json` is exactly those bytes. The Step 8
gate regenerates and compares byte-for-byte (`test_feature_manifest.py`), then tags.
`verified_manifest_hash()` additionally proves the committed file matches regeneration
and is self-consistent — the detector and eval entry points fail closed on it.

The content hash (`MANIFEST_HASH`) is stamped into every contribution export, the
detector's detection artifact, and every evaluation result artifact (features.md
§6.2, §7.3): "set unchanged" is proven against the hash, never asserted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.services import banding as B
from app.services import clustering as C
from app.services import features as F

#: Bumped only by a §6.3 change-control event (a new `features-frozen-vN` tag).
MANIFEST_VERSION = "features-frozen-v2"

#: The committed freeze artifact — regenerated and byte-compared by the Step 8 gate.
MANIFEST_PATH = Path(__file__).resolve().parents[3] / "config" / "feature-manifest.json"


class ManifestDrift(RuntimeError):
    """The committed manifest is missing, malformed, self-inconsistent, or does not
    match regeneration from the live code (review finding 7)."""


# --- Per-feature §6.1 spec ----------------------------------------------------
# Static structural fields (inputs / sign / predicate shape / scaling / missing-
# value semantics). Every *tunable* numeric parameter is resolved from a live
# `features` constant at build time (see `_predicate_params`), NOT snapshotted
# here — so a change to any of them moves the content hash, whether from a source
# edit (re-import) or at runtime.


def _predicate_params(fid: str) -> dict[str, Any]:
    """The firing-affecting predicate parameters, read live off `features` so the
    manifest can never lag the code."""
    table: dict[str, dict[str, Any]] = {
        "F-DEV-1": {
            "min_members": F._DEV_STRONG_MIN_MEMBERS,
            "reliability": F._DEV_STRONG_RELIABILITY.name,
        },
        "F-DEV-2": {
            "members": F._DEV_WEAK_MEMBERS,
            "reliability": F._DEV_WEAK_RELIABILITY.name,
        },
        "F-PAY-1": {},
        "F-CON-1": {},
        "F-DOM-1": {},
        "F-VER-1": {"failure_min": F._VER_FAILURE_MIN},
        "F-CYC-1": {},
        "F-CYC-2": {},
        "F-CYC-3": {
            "latency_hours": F._CYC3_LATENCY_HOURS,
            "recency_days": F._CYC3_RECENCY_DAYS,
            "cycle_ownership": "[cycle_start, cycle_end)",
        },
        "F-CYC-4": {},
        "F-DOM-2": {
            "majority": F._BENIGN_MAJORITY,
            "min_logins": F._ROUTINE_MIN_LOGINS,
        },
        "F-AGE-1": {
            "majority": F._BENIGN_MAJORITY,
            "age_days": F._AGE_MIN_DAYS,
            "min_logins": F._ROUTINE_MIN_LOGINS,
        },
    }
    return table[fid]


def _cap(fid: str) -> float | None:
    """Scaling cap, read live (F-CYC-1's excess contribution is capped at its weight)."""
    return F._W_CYC_1 if fid == "F-CYC-1" else None


_SPEC: dict[str, dict[str, Any]] = {
    "F-DEV-1": {
        "inputs": ["stable_device shared-value member count", "effective reliability"],
        "sign": "+",
        "predicate": "a stable_device value shared by >= {min_members} cluster members "
        "at {reliability} effective reliability",
        "scaling": "indicator {0,1} x weight",
        "missing_value": "no qualifying shared value -> 0.0 (does not fire)",
    },
    "F-DEV-2": {
        "inputs": ["stable_device shared-value member count", "effective reliability"],
        "sign": "+",
        "predicate": "a stable_device value shared by exactly {members} members OR at "
        "{reliability} reliability (weak share)",
        "scaling": "indicator {0,1} x weight",
        "missing_value": "no qualifying shared value -> 0.0 (does not fire)",
    },
    "F-PAY-1": {
        "inputs": ["payment_token shared-value presence"],
        "sign": "+",
        "predicate": "a payment_token value shared by >= 2 cluster members",
        "scaling": "indicator {0,1} x weight",
        "missing_value": "no shared payment token -> 0.0 (does not fire)",
    },
    "F-CON-1": {
        "inputs": ["recovery_contact shared-value presence"],
        "sign": "+",
        "predicate": "a recovery_contact value shared by >= 2 cluster members",
        "scaling": "indicator {0,1} x weight",
        "missing_value": "no shared recovery contact -> 0.0 (does not fire)",
    },
    "F-DOM-1": {
        "inputs": ["account email_domain_type"],
        "sign": "+",
        "predicate": "at least one member (S1/S2 cluster aggregate) / the "
        "any-restriction representative account (S0) has a disposable email domain",
        "scaling": "indicator {0,1} x weight",
        "missing_value": "domain not disposable / account absent as-of -> 0.0",
    },
    "F-VER-1": {
        "inputs": ["verification event stream"],
        "sign": "+",
        "predicate": ">= {failure_min} VERIFICATION_FAILED events, OR a CLAIM_CREDITS "
        "strictly before the first verification (claim-before-verify)",
        "scaling": "indicator {0,1} x weight",
        "missing_value": "no claim and no failures -> 0.0 (does not fire)",
    },
    "F-CYC-1": {
        "inputs": ["cycle-window claimed credits", "banded allowance"],
        "sign": "+",
        "predicate": "allowance excess = max(0, claimed/allowance - 1); fires when the "
        "scaled contribution reaches the firing threshold",
        "scaling": "weight x min(1.0, excess)  (excess curve, linear-then-clamped)",
        "missing_value": "allowance == 0 -> excess 0.0 (does not fire)",
    },
    "F-CYC-2": {
        "inputs": ["per-account consumption days (trailing window)"],
        "sign": "+",
        "predicate": ">= 2 members consume but no consumption day is shared "
        "(sequential tiling)",
        "scaling": "indicator {0,1} x weight",
        "missing_value": "< 2 consuming members -> 0.0 (does not fire)",
    },
    "F-CYC-3": {
        "inputs": ["cycle exhaustion times", "SIGNUP timestamps", "as_of"],
        "sign": "+",
        "predicate": "within the trailing {recency_days} d ending at as_of, a linked "
        "account signs up <= {latency_hours} h after another linked account exhausts "
        "its half-open cycle",
        "scaling": "indicator {0,1} x weight",
        "missing_value": "no exhaustion or signup in the trailing recency window -> 0.0",
    },
    "F-CYC-4": {
        "inputs": ["prior APPLIED decisions", "cluster member set"],
        "sign": "+",
        "predicate": "this member set had an identical pattern UPHELD (APPLIED) in a "
        "prior decision (reversed decisions never count)",
        "scaling": "indicator {0,1} x weight",
        "missing_value": "no matching upheld prior decision -> 0.0",
    },
    "F-DOM-2": {
        "inputs": ["account email_domain_type", "LOGIN events"],
        "sign": "-",
        "predicate": ">= {majority} of the cluster are education/enterprise domains "
        "with routine activity (>= {min_logins} LOGINs)",
        "scaling": "indicator(predicate) x weight; raw_value = fraction qualifying "
        "(the fraction is reported, NOT multiplied into the contribution)",
        "missing_value": "empty cluster -> size floored at 1 -> 0.0",
    },
    "F-AGE-1": {
        "inputs": ["account created_at", "LOGIN events", "cycle-boundary reference"],
        "sign": "-",
        "predicate": ">= {majority} of the cluster are aged >= {age_days} d at the "
        "cycle boundary with routine activity (>= {min_logins} LOGINs)",
        "scaling": "indicator(predicate) x weight; raw_value = fraction qualifying "
        "(the fraction is reported, NOT multiplied into the contribution)",
        "missing_value": "empty cluster -> size floored at 1 -> 0.0",
    },
}

_FAMILY_NAME = {
    F.EvidenceFamily.DEVICE: "device",
    F.EvidenceFamily.PAYMENT: "payment",
    F.EvidenceFamily.CONTACT_DOMAIN: "contact_domain",
    F.EvidenceFamily.CYCLE: "cycle",
    F.EvidenceFamily.VERIFICATION: "verification",
}

# The canonical family + rung per feature, read straight off a freshly scored
# contribution so the manifest can never disagree with the code's tagging.
_TAGS = {
    "F-DEV-1": (F.EvidenceFamily.DEVICE, F.Rung.S1),
    "F-DEV-2": (F.EvidenceFamily.DEVICE, F.Rung.S1),
    "F-PAY-1": (F.EvidenceFamily.PAYMENT, F.Rung.S1),
    "F-CON-1": (F.EvidenceFamily.CONTACT_DOMAIN, F.Rung.S1),
    "F-DOM-1": (F.EvidenceFamily.CONTACT_DOMAIN, F.Rung.S0),
    "F-VER-1": (F.EvidenceFamily.VERIFICATION, F.Rung.S0),
    "F-CYC-1": (F.EvidenceFamily.CYCLE, F.Rung.S2),
    "F-CYC-2": (F.EvidenceFamily.CYCLE, F.Rung.S2),
    "F-CYC-3": (F.EvidenceFamily.CYCLE, F.Rung.S2),
    "F-CYC-4": (F.EvidenceFamily.CYCLE, F.Rung.S2),
    "F-DOM-2": (None, F.Rung.S0),
    "F-AGE-1": (None, F.Rung.S0),
}


# --- Manifest assembly + hashing ---------------------------------------------


def _feature_entry(fid: str) -> dict[str, Any]:
    spec = _SPEC[fid]
    family, rung = _TAGS[fid]
    return {
        "id": fid,
        "inputs": list(spec["inputs"]),
        "sign": spec["sign"],
        "evidence_family": int(family) if family is not None else None,
        "family_name": _FAMILY_NAME.get(family) if family is not None else None,
        "ablation_rung": rung.name,
        "firing_predicate": spec["predicate"],
        "predicate_params": _predicate_params(fid),
        "scaling": spec["scaling"],
        "cap": _cap(fid),
        "missing_value_semantics": spec["missing_value"],
        "firing_threshold": F.FIRING_THRESHOLD,
        "weight": F.SEED_WEIGHTS[fid],
    }


# Firing-affecting modules whose full source is fingerprinted (review finding 1/2):
# these three modules ARE the clustering → banding → feature firing computation, so
# any edit to them — constant, predicate, curve, family/rung, edge rule — must
# re-freeze. Over-conservative on purpose (fails closed).
_FIRING_MODULES = (F, B, C)


def _firing_config() -> dict[str, Any]:
    """Every firing-affecting constant OUTSIDE features.py, read live from banding and
    clustering (review finding 1). The F-CYC-1 allowance path and F-CYC-2 window depend
    on the banding block; the device/payment/contact preprocessing depends on the
    clustering block."""
    return {
        "banding": {
            "window_days": B.WINDOW_DAYS,
            "n_overlap": B.N_OVERLAP,
            "rate_hourly_credits": B.RATE_HOURLY_CREDITS,
            "rate_hourly_min_hours": B.RATE_HOURLY_MIN_HOURS,
            "rate_daily_credits": B.RATE_DAILY_CREDITS,
            "rate_daily_min_days": B.RATE_DAILY_MIN_DAYS,
            "diurnal_d": B.DIURNAL_D,
            "diurnal_d_low": B.DIURNAL_D_LOW,
            "diurnal_d_high": B.DIURNAL_D_HIGH,
            "per_stream_allowance": B.PER_STREAM_ALLOWANCE,
            "ambiguity_fraction": B.AMBIGUITY_FRACTION,
            "small_group_min": B._SMALL_GROUP_MIN,
            "large_shared_min": B._LARGE_SHARED_MIN,
        },
        "clustering": {
            "hub_account_threshold": C.HUB_ACCOUNT_THRESHOLD,
            "max_component_size": C.MAX_COMPONENT_SIZE,
            "decay_halflife_days": C.DECAY_HALFLIFE_DAYS,
            "max_age_days": C.MAX_AGE_DAYS,
            "louvain_seed": C.LOUVAIN_SEED,
            "reliability_weight": {r.name: w for r, w in sorted(
                C._REL_WEIGHT.items(), key=lambda kv: kv[0].value)},
            "resource_sharing_signals": sorted(s.value for s in C.RESOURCE_SHARING_SIGNALS),
        },
    }


def _firing_code_fingerprint() -> str:
    """SHA-256 over the full source of the firing modules — locks executable semantics
    that no constant table can capture (review finding 2)."""
    h = hashlib.sha256()
    for mod in _FIRING_MODULES:
        src = mod.__file__
        if src is None:  # pragma: no cover - firing modules are always file-backed
            raise ManifestDrift(f"cannot fingerprint {mod.__name__}: no source file")
        h.update(Path(src).read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _content() -> dict[str, Any]:
    """The manifest content the hash is taken over — everything except the hash."""
    if set(_SPEC) != set(F.FEATURE_ORDER) or set(_TAGS) != set(F.FEATURE_ORDER):
        missing = sorted(set(F.FEATURE_ORDER) - (set(_SPEC) & set(_TAGS)))
        extra = sorted((set(_SPEC) | set(_TAGS)) - set(F.FEATURE_ORDER))
        raise ValueError(
            f"manifest spec must cover exactly FEATURE_ORDER; missing={missing} extra={extra}"
        )
    return {
        "manifest_version": MANIFEST_VERSION,
        "firing_threshold": F.FIRING_THRESHOLD,
        "masking_aggregation_version": F.MASKING_AGGREGATION_VERSION,
        "feature_order": list(F.FEATURE_ORDER),
        "features": {fid: _feature_entry(fid) for fid in F.FEATURE_ORDER},
        "firing_config": _firing_config(),
        "firing_code_fingerprint": _firing_code_fingerprint(),
    }


def _canonical(obj: Any) -> bytes:
    """Deterministic bytes for hashing — compact, sorted keys."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def content_hash(content: dict[str, Any] | None = None) -> str:
    return hashlib.sha256(_canonical(content if content is not None else _content())).hexdigest()


def build_manifest() -> dict[str, Any]:
    """The full committed manifest: content plus its own content hash."""
    content = _content()
    return {**content, "content_hash": content_hash(content)}


def manifest_bytes() -> bytes:
    """Byte-canonical serialization written to `config/feature-manifest.json`
    (pretty, sorted, trailing newline) — what the gate compares byte-for-byte."""
    return (json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n").encode()


#: The frozen content hash, stamped into exports / detection / eval artifacts.
MANIFEST_HASH: str = content_hash()


def load_committed() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text())


def verified_manifest_hash() -> str:
    """Return the frozen hash ONLY after proving the committed
    `config/feature-manifest.json` (a) exists, (b) is byte-identical to regeneration
    from the live code, and (c) carries a self-consistent `content_hash`. Raises
    `ManifestDrift` otherwise. Runtime entry points call this so a stale or
    post-freeze config fails closed instead of emitting apparently-versioned results
    whose hash names no committed artifact."""
    if not MANIFEST_PATH.is_file():
        raise ManifestDrift(f"no committed manifest at {MANIFEST_PATH}")
    committed_bytes = MANIFEST_PATH.read_bytes()
    if committed_bytes != manifest_bytes():
        raise ManifestDrift(
            "committed feature-manifest.json does not match regeneration from live "
            "code (run `make manifest` after an intended §6.3 change and re-tag)"
        )
    try:
        committed = json.loads(committed_bytes)
        embedded = committed.pop("content_hash")
    except (ValueError, KeyError) as exc:
        raise ManifestDrift(f"committed manifest is malformed: {exc}") from exc
    if content_hash(committed) != embedded or embedded != MANIFEST_HASH:
        raise ManifestDrift("committed manifest content_hash is self-inconsistent")
    return MANIFEST_HASH


def write_manifest(path: Path = MANIFEST_PATH) -> str:
    """(Re)write the manifest file; returns its content hash."""
    path.write_text(manifest_bytes().decode())
    return MANIFEST_HASH

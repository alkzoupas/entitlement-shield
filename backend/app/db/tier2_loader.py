"""Load the Tier 2 (qwen-brief) synthetic world into physically separate stores.

Reads the sealed-authored parquets under a Tier 2 directory (produced by the
independent author, ``data/tier2/generate_tier2.py``) and materializes:

  * the **inference store** (``<dir>/inference.db``): accounts, events, linkage
    observations (raw ``signal_value`` hashed to ``signal_value_hash`` at the
    collection boundary), entitlement cycles — nothing label/scenario/cluster-bearing.
  * the physically separate **ground-truth store** (``<dir>/ground_truth.db``):
    sealed labels + the evaluation-only split grouping, written ONLY through the
    write-only :class:`GroundTruthSink`.

This is a db-layer wiring module (the same layer that defines the GT store and
its write-only sink), not a ``services/`` module, so it may legitimately create
both engines and the GT session — detector/inference services still receive
neither. Tier 2 metrics are NOT reportable as cross-family: the author is a
Claude Sonnet stand-in for qwen (same model family), see ``docs/tier2-decisions.md``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from sqlalchemy.orm import Session

from app.db.engine import make_inference_engine
from app.db.ground_truth_store import make_ground_truth_engine
from app.db.ground_truth_writer import make_ground_truth_sink
from app.db.schema import create_all_schemas
from app.db.world_provenance import (
    WORLD_APPEAL_SOURCE_PREFIX,
    WorldProvenanceError,
    manifest_identity,
    manifest_tier,
    record_world_receipts,
    verify_world_pair,
)
from app.models.enums import (
    AppealTruthKind,
    EmailDomainType,
    EndpointCostTier,
    EventType,
    GroundTruthLabel,
    Reliability,
    SignalType,
    VerificationLevel,
)
from app.models.inference import (
    Account,
    AccountEvent,
    Appeal,
    EntitlementCycle,
    LinkageObservation,
)


def _utc(dt: datetime) -> datetime:
    """Canonicalize to a UTC-aware instant (the store columns are tz-aware)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def hash_signal_value(raw: str) -> str:
    """Raw linkage values never cross the collection boundary — the inference
    store keeps a keyed hash (architecture §4). Equal raw values map to equal
    hashes, so resource-sharing edges are preserved; distinct values do not
    collide. Independent of the dev generator's keying — only equality matters."""
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _opt_cost_tier(v: object) -> EndpointCostTier | None:
    if v is None or v == "":
        return None
    return EndpointCostTier(str(v))


def _opt_credits(v: object) -> int | None:
    if isinstance(v, (int, float)):
        return int(v)
    return None


_ARTIFACTS = (
    "accounts", "account_events", "linkage_observations", "entitlement_cycles",
    "ground_truth", "cluster_membership",
)
# Tier 4 adds world-authored appeal text (inference plane, zero evidentiary weight)
# and its sealed truth (ground-truth plane).
_APPEAL_ARTIFACTS = ("appeals", "appeal_truth")
_TIER_ARTIFACTS = {
    "tier2": _ARTIFACTS,
    "tier3": _ARTIFACTS,
    "tier4": _ARTIFACTS + _APPEAL_ARTIFACTS,
}


def _read_verified_source(directory: Path) -> tuple[dict, dict[str, pl.DataFrame]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    checksums = manifest.get("checksums", {})
    if not manifest.get("world_version"):
        raise WorldProvenanceError("Missing world version")
    tier = manifest_tier(manifest)
    if tier not in _TIER_ARTIFACTS:
        raise WorldProvenanceError("Manifest world_tier must be tier2, tier3 or tier4")
    artifacts = _TIER_ARTIFACTS[tier]
    if set(checksums) != {f"{name}.parquet" for name in artifacts}:
        raise WorldProvenanceError(
            f"Manifest must identify exactly the {len(artifacts)} {tier} source artifacts"
        )
    frames = {}
    for name in artifacts:
        # Parse the identical bytes verified here, avoiding a hash/read race.
        data = (directory / f"{name}.parquet").read_bytes()
        if hashlib.sha256(data).hexdigest() != checksums[f"{name}.parquet"]:
            raise WorldProvenanceError(f"Source checksum mismatch: {name}.parquet")
        frames[name] = pl.read_parquet(io.BytesIO(data))
    return manifest, frames


def _validate_credit_windows(
    events: pl.DataFrame, cycles: pl.DataFrame, world_end: datetime,
) -> None:
    by_account: dict[str, list[dict]] = {}
    for cycle in cycles.iter_rows(named=True):
        by_account.setdefault(cycle["account_id"], []).append(cycle)
    totals: dict[str, list[int]] = {}
    for account_cycles in by_account.values():
        account_cycles.sort(key=lambda row: row["cycle_start"])
        previous_end = None
        for cycle in account_cycles:
            start, end = cycle["cycle_start"], cycle["cycle_end"]
            if start >= end or (previous_end is not None and previous_end > start):
                raise WorldProvenanceError("Invalid or overlapping account cycle windows")
            previous_end = end
            totals[cycle["cycle_id"]] = [0, 0]
    for event in events.iter_rows(named=True):
        kind = event["event_type"]
        if kind not in {"claim_credits", "consume_credits"}:
            continue
        amount = event["credits_amount"]
        if not isinstance(amount, int) or amount < 0:
            raise WorldProvenanceError("Credit amount must be a nonnegative integer")
        matches = [c for c in by_account.get(event["account_id"], [])
                   if c["cycle_start"] <= event["timestamp"] < c["cycle_end"]]
        if len(matches) != 1:
            raise WorldProvenanceError("Credit event must belong to exactly one cycle")
        totals[matches[0]["cycle_id"]][0 if kind == "claim_credits" else 1] += amount
    for cycle in cycles.iter_rows(named=True):
        granted, consumed = totals[cycle["cycle_id"]]
        if (granted != cycle["credits_granted"] or consumed != cycle["credits_consumed"]
                or not 0 <= consumed <= granted):
            raise WorldProvenanceError("Cycle totals disagree with credit events")
        expected_expired = granted - consumed if _utc(cycle["cycle_end"]) <= world_end else 0
        if cycle["credits_expired"] != expected_expired:
            raise WorldProvenanceError("Cycle expiry disagrees with its authoritative window")


def _ensure_indirect_symlink(directory: Path, name: str) -> None:
    """Idempotently point ``directory/name`` at ``stores-current/name``.

    The per-store symlinks (``inference.db``/``ground_truth.db``) are always
    indirected *through* ``stores-current`` -- their target text never
    changes across publications, only what ``stores-current`` itself resolves
    to. That makes them safe to create once, up front, and never touch again:
    a later ``stores-current`` swap makes them follow automatically. Safe to
    call repeatedly (idempotent) and from either the publish or the
    reuse/self-heal path.
    """
    target = directory / name
    expected = str(Path("stores-current") / name)
    try:
        if target.is_symlink():
            if os.readlink(target) == expected:
                return
        elif target.exists():
            raise WorldProvenanceError(f"Refusing to replace non-symlink store path: {target}")
    except OSError:
        pass
    # A raw OSError here (disk full, read-only fs) is left to propagate
    # undisguised rather than wrapped in `WorldProvenanceError`:
    # `test_loader_recovers_from_crash_before_commit_without_wedging` pins
    # exactly this -- a mid-`_publish_pair` crash must surface as the actual
    # OS-level failure, not a translated error, so a caller/operator sees
    # what really happened and a bare retry after the transient cause clears
    # is what's expected to recover (see that test's docstring).
    temporary = directory / f".{name}.{uuid.uuid4().hex}.tmp"
    temporary.symlink_to(Path("stores-current") / name)
    os.replace(temporary, target)


def _publish_pair(directory: Path, generation: Path) -> None:
    # Both complete stores live behind one atomically replaced directory symlink.
    # Readers that span a concurrent switch must still verify their pair receipts.
    #
    # Ordering matters for crash safety: the per-store symlinks are written
    # FIRST because they are idempotent indirection (see
    # `_ensure_indirect_symlink`) that never needs to change again once it
    # exists. The `stores-current` swap is the ONLY operation that makes
    # `generation` live, so it is the LAST thing this function does, and
    # nothing follows it that could raise. A crash before this line leaves the
    # previous `stores-current` (if any) untouched and pointing at a fully
    # valid prior generation; a crash cannot happen "after" it because there
    # is no more code to crash in.
    for name in ("inference.db", "ground_truth.db"):
        _ensure_indirect_symlink(directory, name)
    current = directory / "stores-current"
    if current.exists() and not current.is_symlink():
        raise WorldProvenanceError(
            f"Refusing to replace non-symlink stores-current path: {current}"
        )
    temporary = directory / f".stores-current.{uuid.uuid4().hex}.tmp"
    temporary.symlink_to(generation.relative_to(directory), target_is_directory=True)
    os.replace(temporary, current)  # commit point


@contextmanager
def _publication_lock(directory: Path):
    """Serialize first publication and idempotent reuse across loader processes."""
    import fcntl

    lock_path = directory / ".loads" / "publication.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _counts(frames: dict[str, pl.DataFrame]) -> dict[str, int]:
    return {
        "accounts": frames["accounts"].height,
        "events": frames["account_events"].height,
        "linkage": frames["linkage_observations"].height,
        "cycles": frames["entitlement_cycles"].height,
        "clusters": frames["ground_truth"].height,
        "split_groups": frames["cluster_membership"].height,
        **({"appeals": frames["appeals"].height, "appeal_truth": frames["appeal_truth"].height}
           if "appeals" in frames else {}),
    }


def _validate_publication_pointer(directory: Path) -> bool:
    """Structural soundness of the ``stores-current`` commit marker, independent
    of world identity/tier/content.

    ``stores-current`` is the sole commit marker (see `_publish_pair`): a
    generation is "published" if and only if `stores-current` points at it.
    That splits into two genuinely different situations. If the marker is
    simply *absent* (no `stores-current` at all), nothing was ever committed --
    this is self-heal/first-publication territory, and this returns False. But
    once a marker exists -- whether a symlink or some other path occupying that
    name -- it is claiming to name a specific committed generation, so anything
    short of that generation resolving cleanly (a non-symlink occupying the
    marker path, a dangling target, a target that is not a directory, a
    committed generation missing one of its per-store files, or a per-store
    symlink that resolves outside the expected root) means an already-published
    pointer is corrupted. Silently regenerating and overwriting it in that case
    would mask real corruption instead of surfacing it, so each such state
    raises `WorldProvenanceError` with the exact `rm` remediation -- world
    content is fully deterministic from the sealed manifest, so removing a
    corrupted commit marker and letting the loader republish is not data loss.
    A missing or stale per-store indirection symlink under an otherwise valid
    `stores-current` is still self-healed in place (`_ensure_indirect_symlink`
    is idempotent and never touches `stores-current` itself).

    This is the shared structural gate for BOTH the idempotent reuse path and
    the explicit ``republish`` path. `republish` deliberately bypasses only the
    receipt/provenance *content* mismatch (that is the mislabelled-pair case it
    exists to correct); it must NOT bypass these structural checks, so it runs
    this validator too, before building a fresh generation.
    """
    current = directory / "stores-current"
    remediation = (
        f"Remove the corrupted commit marker with `rm {current}` -- world content is "
        "fully deterministic from the sealed manifest, so the loader will regenerate "
        "and republish safely; this is not data loss."
    )
    if not current.is_symlink():
        if not current.exists():
            return False
        raise WorldProvenanceError(
            f"A non-symlink path occupies the stores-current commit marker: {current}. "
            f"{remediation}"
        )
    try:
        root = current.resolve(strict=True)
    except OSError as exc:
        raise WorldProvenanceError(
            f"stores-current is a dangling symlink (its target no longer exists): "
            f"{current}. {remediation}"
        ) from exc
    if not root.is_dir():
        raise WorldProvenanceError(
            f"stores-current resolves to a non-directory: {current} -> {root}. {remediation}"
        )
    # The committed generation must be a `world-*` child of this directory's own
    # `.loads/` store (where `tempfile.mkdtemp(prefix="world-", dir=.loads)` puts
    # it). Without this, a `stores-current` pointing at an *external* directory
    # that merely happens to contain both databases passes the per-store parent
    # check below (each store's parent equals that external root) -- a foreign
    # publication pointer that must be surfaced, not accepted or overwritten.
    loads_dir = (directory / ".loads").resolve()
    if root.parent != loads_dir or not root.name.startswith("world-"):
        raise WorldProvenanceError(
            f"stores-current resolves outside the loader's generation store {loads_dir} "
            f"(expected a world-* child): {current} -> {root}. {remediation}"
        )
    for name in ("inference.db", "ground_truth.db"):
        path = directory / name
        _ensure_indirect_symlink(directory, name)
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise WorldProvenanceError(
                f"The committed generation is missing {name}: {path}. {remediation}"
            ) from exc
        if resolved.parent != root:
            raise WorldProvenanceError(
                f"{name} resolves outside the committed generation root ({root}): "
                f"{resolved}. {remediation}"
            )
    return True


def _reuse_published_pair(directory: Path, manifest: dict) -> bool:
    """Verify and retain an already published pair, never replacing its GT inode.

    Absence of `stores-current` means nothing was ever committed, so this
    returns False and the caller regenerates and publishes fresh. Otherwise the
    marker must be structurally sound (`_validate_publication_pointer`, which
    raises on any corruption) AND its receipts must verify against the sealed
    manifest tier/identity; a structurally intact pair whose receipts disagree
    with the source manifest raises rather than being silently reused.
    """
    if not _validate_publication_pointer(directory):
        return False
    names = tuple(directory / name for name in ("inference.db", "ground_truth.db"))
    inference_engine = make_inference_engine(f"sqlite:///{names[0]}")
    ground_truth_engine = make_ground_truth_engine(f"sqlite:///{names[1]}")
    try:
        with (
            Session(inference_engine, future=True) as inf,
            Session(ground_truth_engine, future=True) as gt,
        ):
            world = verify_world_pair(inf, gt)
    finally:
        inference_engine.dispose()
        ground_truth_engine.dispose()
    expected_tier = manifest_tier(manifest)
    if world.tier != expected_tier or world.world_id != manifest_identity(manifest):
        raise WorldProvenanceError(
            "Published world pair does not match verified source manifest"
        )
    return True


def _cleanup_unpublished(directory: Path, generation: Path) -> None:
    """Delete `generation` only if it was never committed live.

    Once `stores-current` points at `generation`, publication has committed
    (see `_publish_pair`) and this must be a no-op: deleting it would turn a
    live pointer into a dangling one and destroy the only copy of what was,
    until a moment ago, the good published generation -- exactly the
    "prior stores survive validation failures" guarantee this module
    promises. Structurally `_publish_pair` cannot raise once it has committed
    (the `stores-current` swap is its last statement), so this check is
    defense in depth rather than a path this module expects to hit.
    """
    current = directory / "stores-current"
    try:
        if current.is_symlink() and current.resolve(strict=True) == generation.resolve(strict=True):
            return
    except OSError:
        pass
    shutil.rmtree(generation)


def load_tier2(directory: str | Path, *, republish: bool = False) -> dict[str, int]:
    """Verify sources, build/verify a fresh pair, then publish a single pointer.

    Prior stores survive validation failures. Completed generations are retained;
    paired receipts prevent mixing worlds even during a concurrent pointer switch.

    ``republish=True`` is the explicit, opt-in republication path. It bypasses
    ONLY the receipt/provenance *content* reuse-verify (the mislabelled-pair case
    it exists to correct) and always rebuilds a fresh generation from the
    already-verified source artifacts, then atomically swaps ``stores-current``
    onto it. It still runs the shared structural validator
    (`_validate_publication_pointer`) before building, so a dangling, malformed,
    missing-store, outside-root, or non-symlink publication pointer is surfaced
    and fails fast -- never silently overwritten. The prior generation is
    retained (never deleted). This is therefore a deliberate re-materialization,
    not a generic force/delete.
    """
    d = Path(directory).resolve()
    manifest, frames = _read_verified_source(d)
    world_end = _utc(datetime.fromisoformat(str(manifest["world_end"]).replace("Z", "+00:00")))
    _validate_credit_windows(frames["account_events"], frames["entitlement_cycles"], world_end)
    (d / ".loads").mkdir(exist_ok=True)
    with _publication_lock(d):
        if not republish and _reuse_published_pair(d, manifest):
            return _counts(frames)
        if republish:
            # Fail fast (before any regeneration work) on a corrupt/foreign
            # publication pointer; the return value is irrelevant here (absent
            # marker is fine -- we are about to publish), only its raises matter.
            _validate_publication_pointer(d)
        generation = Path(tempfile.mkdtemp(prefix="world-", dir=d / ".loads"))
        try:
            counts = _load_into(generation, manifest, frames)
            _publish_pair(d, generation)
            return counts
        except BaseException:
            _cleanup_unpublished(d, generation)
            raise


def _load_into(d: Path, manifest: dict, frames: dict[str, pl.DataFrame]) -> dict[str, int]:
    accounts, events, linkage, cycles, ground_truth, membership = (
        frames[name] for name in _ARTIFACTS
    )
    inf_path, gt_path = d / "inference.db", d / "ground_truth.db"
    inference_engine = make_inference_engine(f"sqlite:///{inf_path}")
    ground_truth_engine = make_ground_truth_engine(f"sqlite:///{gt_path}")
    create_all_schemas(inference_engine, ground_truth_engine)

    with (
        Session(inference_engine, future=True) as inf,
        Session(ground_truth_engine, future=True) as gt,
    ):
        # Parents first (FK PRAGMA is ON): accounts, flush, then children.
        inf.add_all(
            Account(
                account_id=r["account_id"],
                created_at=_utc(r["created_at"]),
                email_domain_type=EmailDomainType(r["email_domain_type"]),
                verification_level=VerificationLevel(r["verification_level"]),
            )
            for r in accounts.iter_rows(named=True)
        )
        inf.flush()

        inf.add_all(
            AccountEvent(
                event_id=r["event_id"],
                account_id=r["account_id"],
                timestamp=_utc(r["timestamp"]),
                event_type=EventType(r["event_type"]),
                credits_amount=_opt_credits(r["credits_amount"]),
                endpoint_cost_tier=_opt_cost_tier(r["endpoint_cost_tier"]),
            )
            for r in events.iter_rows(named=True)
        )
        inf.add_all(
            LinkageObservation(
                observation_id=r["observation_id"],
                account_id=r["account_id"],
                timestamp=_utc(r["timestamp"]),
                signal_type=SignalType(r["signal_type"]),
                signal_value_hash=hash_signal_value(str(r["signal_value"])),
                reliability=Reliability(r["reliability"]),
            )
            for r in linkage.iter_rows(named=True)
        )
        inf.add_all(
            EntitlementCycle(
                cycle_id=r["cycle_id"],
                account_id=r["account_id"],
                cycle_start=_utc(r["cycle_start"]),
                cycle_end=_utc(r["cycle_end"]),
                credits_granted=int(r["credits_granted"]),
                credits_consumed=int(r["credits_consumed"]),
                credits_expired=int(r["credits_expired"]),
            )
            for r in cycles.iter_rows(named=True)
        )
        # World-authored appeals (Tier 4). Their events are already base rows; the
        # text lives only in `appeals`, which no feature reads (policy §11). The
        # ingest bounds text below the runtime cap, so nothing is truncated here.
        if "appeals" in frames:
            inf.add_all(
                Appeal(
                    appeal_id=r["appeal_id"],
                    account_id=r["account_id"],
                    decision_id=None,
                    submitted_at=_utc(r["submitted_at"]),
                    text=r["text"],
                    truncated=False,
                    source=f"{WORLD_APPEAL_SOURCE_PREFIX}{manifest['world_version']}",
                )
                for r in frames["appeals"].iter_rows(named=True)
            )

        # Sealed plane, write-only sink. split_group_id = cluster_id: the sealed
        # scenario group is the split key (evaluation-only, P1-4c), robust to
        # later-merging inferred clusters.
        sink = make_ground_truth_sink(gt)
        for r in ground_truth.iter_rows(named=True):
            sink.write_ground_truth(
                cluster_id=r["cluster_id"],
                scenario_family=r["scenario_family"],
                label=GroundTruthLabel(r["label"]),
                notes=r["notes"],
            )
        for r in membership.iter_rows(named=True):
            sink.write_split_group(
                account_id=r["account_id"], split_group_id=r["cluster_id"]
            )
        if "appeal_truth" in frames:
            for r in frames["appeal_truth"].iter_rows(named=True):
                sink.write_appeal_truth(
                    event_id=r["event_id"], kind=AppealTruthKind(r["kind"]), notes=r["notes"]
                )

        record_world_receipts(inf, gt, manifest, tier=manifest_tier(manifest))
        inf.commit()
        gt.commit()
        verify_world_pair(inf, gt)

    inference_engine.dispose()
    ground_truth_engine.dispose()

    return _counts(frames)

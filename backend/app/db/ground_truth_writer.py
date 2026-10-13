"""Write-only sink for ground truth (used by the Step 2 generator).

Exposes only writes — no read function. The generator legitimately authors the
sealed labels and the evaluation-only split grouping; it gets this sink and never
the reader (review 4 P1-4a). Detector services import neither.

The capability is an OBJECT, not a bare `Session`: `GroundTruthSink` wraps the
privileged session as a private attribute and exposes ONLY `write_ground_truth`,
`write_split_group` and `write_appeal_truth`. It has no `.execute`, `.scalar(s)`,
`.query`, `.connection`, `.get_bind`, no `select`, and no way to reach the engine —
so possession of the sink cannot be turned into a read of the sealed abuse label
(review 1 P0). Contrast with handing out a raw `Session`, which grants unrestricted
reads and raw SQL.

The module-level `write_ground_truth` / `write_split_group` functions are kept as a
thin convenience for round-trip tests; production/generator code uses the sink.
"""

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.db.ground_truth_store import AppealTruth, GroundTruth, SplitGroup
from app.models.enums import AppealTruthKind, GroundTruthLabel


def write_ground_truth(
    session: Session,
    *,
    cluster_id: str,
    scenario_family: str,
    label: GroundTruthLabel,
    notes: str | None = None,
) -> None:
    session.merge(
        GroundTruth(
            cluster_id=cluster_id,
            scenario_family=scenario_family,
            label=label,
            notes=notes,
        )
    )


def write_split_group(session: Session, *, account_id: str, split_group_id: str) -> None:
    session.merge(SplitGroup(account_id=account_id, split_group_id=split_group_id))


def write_appeal_truth(
    session: Session, *, event_id: str, kind: AppealTruthKind, notes: str | None = None
) -> None:
    session.merge(AppealTruth(event_id=event_id, kind=kind, notes=notes))


class GroundTruthSink:
    """Write-only capability over the ground-truth store.

    Holds the privileged session as a private, name-mangled attribute and exposes
    ONLY the write methods. There is deliberately no flush/commit/execute/read
    surface: the owner that constructed the underlying session (a wiring script,
    never a service) keeps the session and commits it.
    """

    __slots__ = ("__session",)

    def __init__(self, session: Session) -> None:
        self.__session = session

    def write_ground_truth(
        self,
        *,
        cluster_id: str,
        scenario_family: str,
        label: GroundTruthLabel,
        notes: str | None = None,
    ) -> None:
        write_ground_truth(
            self.__session,
            cluster_id=cluster_id,
            scenario_family=scenario_family,
            label=label,
            notes=notes,
        )

    def write_split_group(self, *, account_id: str, split_group_id: str) -> None:
        write_split_group(
            self.__session, account_id=account_id, split_group_id=split_group_id
        )

    def write_appeal_truth(
        self, *, event_id: str, kind: AppealTruthKind, notes: str | None = None
    ) -> None:
        write_appeal_truth(self.__session, event_id=event_id, kind=kind, notes=notes)


def make_ground_truth_sink(bind: Engine | Session) -> GroundTruthSink:
    """Construct a write-only sink from a ground-truth engine or session.

    Given an `Engine`, a fresh `Session` is created and sealed inside the sink (the
    caller can never reach it). Given a `Session`, that session is wrapped — the
    caller retains it only to commit; the sink never exposes it.
    """
    session = bind if isinstance(bind, Session) else Session(bind, future=True)
    return GroundTruthSink(session)

"""The ground-truth capability boundary (review 4 P1-4a) — the firewall that makes
'no ground truth reaches inference' a capability, not just a naming convention.

Three independent guarantees:
  1. STATIC — no detector service imports any ground-truth module, and no service
     names the sealed tables in raw SQL/string literals; only the generator may
     import the write sink, only evaluation the FULL reader, and only the future
     family-classifier the FAMILY-ONLY reader (review 3 P1).
  2. RUNTIME — the stores are physically separate: a session bound to the inference
     engine cannot see the ground-truth tables at all.
  3. CAPABILITY — the object handed to the generator is a write-only sink, not a
     readable Session: it cannot execute `select`, raw SQL, or reach an engine
     (review 1 P0).
"""

import ast
import pathlib

import pytest
from sqlalchemy import Engine, inspect, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.db import ground_truth_family_reader
from app.db.ground_truth_family_reader import load_scenario_families
from app.db.ground_truth_reader import load_labels, load_split_groups
from app.db.ground_truth_store import GroundTruth
from app.db.ground_truth_writer import (
    GroundTruthSink,
    make_ground_truth_sink,
    write_ground_truth,
    write_split_group,
)
from app.models.enums import GroundTruthLabel

APP_DIR = pathlib.Path(__file__).resolve().parents[1]
SERVICES_DIR = APP_DIR / "services"

# The only ground-truth imports any service module may have, keyed by module stem.
# The reader is split so the family-classifier cannot obtain the binary label:
#   evaluation        → FULL reader (binary label + split groups)
#   classify_families → FAMILY-ONLY reader (scenario family, no abuse label)
#   generator         → write-only sink
ALLOWED_GT_IMPORTS: dict[str, set[str]] = {
    "evaluation": {"app.db.ground_truth_reader"},
    "classify_families": {"app.db.ground_truth_family_reader"},
    "generator": {"app.db.ground_truth_writer"},
}

# Sealed table names that must never appear (import OR raw SQL/string literal) in a
# service module that is not an allowed ground-truth importer.
SEALED_TABLE_NAMES = ("ground_truth", "split_groups")


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def _gt_imports(mods: set[str]) -> set[str]:
    return {m for m in mods if "ground_truth" in m}


def _service_modules() -> list[pathlib.Path]:
    # Recursive: nested service packages are in the detector dependency surface too.
    return [p for p in SERVICES_DIR.rglob("*.py") if "__pycache__" not in p.parts]


# ---- 1. STATIC import policy (recursive over all service modules) -----------


def test_no_service_imports_ground_truth_outside_policy() -> None:
    for py in _service_modules():
        stem = py.stem
        found = _gt_imports(_imported_modules(py))
        allowed = ALLOWED_GT_IMPORTS.get(stem, set())
        assert found <= allowed, (
            f"services/{py.relative_to(SERVICES_DIR)} imports disallowed "
            f"ground-truth module(s): {found - allowed}"
        )


def test_no_service_names_sealed_tables_in_raw_text() -> None:
    # A raw SQL string or table-name literal referencing the sealed tables is just
    # as much a leak as an import. Only the allowed importers may name them.
    for py in _service_modules():
        if py.stem in ALLOWED_GT_IMPORTS:
            continue
        text = py.read_text()
        for name in SEALED_TABLE_NAMES:
            assert name not in text, (
                f"services/{py.relative_to(SERVICES_DIR)} references sealed table "
                f"{name!r} outside the privileged ground-truth modules"
            )


def test_evidence_service_is_inside_the_import_boundary() -> None:
    # F7-c: a prior review flagged evidence.py as unguarded because it only
    # checked the api/*.py glob elsewhere. It is in fact covered here via the
    # recursive _service_modules() scan with no ALLOWED_GT_IMPORTS entry — pin
    # that explicitly so future reviewers don't re-flag it.
    assert (SERVICES_DIR / "evidence.py") in _service_modules()
    assert "evidence" not in ALLOWED_GT_IMPORTS


def test_policy_map_splits_full_and_family_readers() -> None:
    # The classifier surface gets family-only; only evaluation gets the full reader.
    assert ALLOWED_GT_IMPORTS["classify_families"] == {"app.db.ground_truth_family_reader"}
    assert ALLOWED_GT_IMPORTS["evaluation"] == {"app.db.ground_truth_reader"}
    assert "app.db.ground_truth_reader" not in ALLOWED_GT_IMPORTS["classify_families"]


def test_family_reader_cannot_read_binary_label() -> None:
    # The family-only capability module must not expose the binary-label operation.
    assert hasattr(ground_truth_family_reader, "load_scenario_families")
    assert not hasattr(ground_truth_family_reader, "load_labels")


def test_inference_wiring_is_ground_truth_free() -> None:
    # The inference engine/session module must never reach the ground-truth store.
    engine_mod = APP_DIR / "db" / "engine.py"
    assert _gt_imports(_imported_modules(engine_mod)) == set()


# ---- 2. RUNTIME store separation --------------------------------------------


def _table_names(engine: Engine) -> set[str]:
    return set(inspect(engine).get_table_names())


def test_stores_do_not_share_tables(inference_engine: Engine, gt_engine: Engine) -> None:
    inf_tables = _table_names(inference_engine)
    gt_tables = _table_names(gt_engine)

    assert {"ground_truth", "split_groups"} <= gt_tables
    assert {"ground_truth", "split_groups"}.isdisjoint(inf_tables)

    assert {"accounts", "decisions", "audit_events"} <= inf_tables
    assert {"accounts", "decisions", "audit_events"}.isdisjoint(gt_tables)


def test_inference_session_cannot_read_ground_truth(inference_engine: Engine) -> None:
    # No ground_truth table exists in the inference store, so the query fails.
    with Session(inference_engine, future=True) as session:
        with pytest.raises(OperationalError):
            session.scalars(select(GroundTruth)).all()


# ---- 3. CAPABILITY: the sink is write-only, not a readable Session -----------

# Read/SQL surfaces a raw Session would expose but the sink must not.
_FORBIDDEN_SINK_ATTRS = (
    "execute",
    "scalar",
    "scalars",
    "query",
    "connection",
    "get_bind",
    "bind",
    "engine",
    "commit",
    "flush",
)


def test_sink_exposes_only_write_methods(gt_engine: Engine) -> None:
    sink = make_ground_truth_sink(gt_engine)
    assert isinstance(sink, GroundTruthSink)
    assert not isinstance(sink, Session)

    # The two write methods are present...
    assert callable(sink.write_ground_truth)
    assert callable(sink.write_split_group)

    # ...and every read/SQL/engine surface is absent.
    for attr in _FORBIDDEN_SINK_ATTRS:
        assert not hasattr(sink, attr), f"sink unexpectedly exposes {attr!r}"

    # __slots__ means no ad-hoc attribute smuggling, and the raw session is not a
    # plain public attribute.
    assert not hasattr(sink, "session")
    assert "__dict__" not in dir(sink)


def test_sink_cannot_execute_a_label_read(gt_engine: Engine) -> None:
    sink = make_ground_truth_sink(gt_engine)
    sink.write_ground_truth(
        cluster_id="c1",
        scenario_family="trial_farm",
        label=GroundTruthLabel.ABUSIVE,
    )
    # There is simply no method through which to run `select(GroundTruth.label)`.
    with pytest.raises(AttributeError):
        sink.execute(select(GroundTruth.label))  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        sink.scalar(select(GroundTruth.label))  # type: ignore[attr-defined]


def test_writer_and_reader_round_trip(gt_session: Session) -> None:
    write_ground_truth(
        gt_session,
        cluster_id="c1",
        scenario_family="homogeneous_lab",
        label=GroundTruthLabel.LEGITIMATE,
        notes="hard negative",
    )
    write_split_group(gt_session, account_id="a1", split_group_id="g1")
    gt_session.flush()

    assert load_labels(gt_session) == {"c1": GroundTruthLabel.LEGITIMATE}
    # the classifier-visible view exposes family but not the abuse label
    assert load_scenario_families(gt_session) == {"c1": "homogeneous_lab"}
    assert load_split_groups(gt_session) == {"a1": "g1"}

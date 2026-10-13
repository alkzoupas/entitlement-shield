"""Physical store separation is ENFORCED, not merely configured (review 2 P1).

`create_all_schemas` refuses to create the inference and sealed ground-truth
schemas in the same physical SQLite file, so a misconfigured pair of URLs cannot
land the sealed labels beside the inference tables.
"""

import pytest
from sqlalchemy import create_engine, inspect

from app.db.engine import create_inference_schema
from app.db.ground_truth_store import create_ground_truth_schema
from app.db.schema import StoreCollisionError, create_all_schemas
from app.db.schema_guard import MixedStoreError


def test_same_sqlite_file_is_rejected(tmp_path) -> None:
    shared = tmp_path / "collide.db"
    inf = create_engine(f"sqlite:///{shared}", future=True)
    gt = create_engine(f"sqlite:///{shared}", future=True)
    with pytest.raises(StoreCollisionError):
        create_all_schemas(inf, gt)


def test_equivalent_relative_and_absolute_paths_collide(tmp_path, monkeypatch) -> None:
    # A relative path and its absolute equivalent are the SAME file — must be caught.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    inf = create_engine("sqlite:///sub/world.db", future=True)
    gt = create_engine(f"sqlite:///{tmp_path / 'sub' / 'world.db'}", future=True)
    with pytest.raises(StoreCollisionError):
        create_all_schemas(inf, gt)


def test_separate_files_succeed(tmp_path) -> None:
    inf = create_engine(f"sqlite:///{tmp_path / 'inf.db'}", future=True)
    gt = create_engine(f"sqlite:///{tmp_path / 'gt.db'}", future=True)
    create_all_schemas(inf, gt)

    inf_tables = set(inspect(inf).get_table_names())
    gt_tables = set(inspect(gt).get_table_names())
    assert {"accounts", "decisions"} <= inf_tables
    assert {"ground_truth", "split_groups"} <= gt_tables
    assert {"ground_truth", "split_groups"}.isdisjoint(inf_tables)


def test_distinct_in_memory_engines_do_not_collide() -> None:
    # Two separate in-memory SQLite engines are physically distinct databases even
    # though the URL is identical — they must NOT be treated as colliding.
    inf = create_engine("sqlite://", future=True)
    gt = create_engine("sqlite://", future=True)
    create_all_schemas(inf, gt)
    assert {"accounts"} <= set(inspect(inf).get_table_names())
    assert {"ground_truth"} <= set(inspect(gt).get_table_names())


@pytest.mark.parametrize("first, second", [
    (create_inference_schema, create_ground_truth_schema),
    (create_ground_truth_schema, create_inference_schema),
])
def test_independent_schema_creation_rejects_opposite_plane_marker(first, second) -> None:
    store = create_engine("sqlite://", future=True)
    first(store)
    with pytest.raises(MixedStoreError, match="opposite-plane"):
        second(store)

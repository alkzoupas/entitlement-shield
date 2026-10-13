"""F8: isolated fixture labels only; no committed Tier-2 data is scored."""
from dataclasses import replace

import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import DBAPIError

from app.db.evaluation_access import (
    EvaluationAccessError,
    begin_finalization,
    complete_finalization,
)
from app.db.ground_truth_store import GroundTruth, SplitGroup
from app.db.world_provenance import record_world_receipts
from app.models.enums import GroundTruthLabel
from app.services.evaluation import assign_split, load_ground_truth


def _world(inf, gt):
    groups = {}
    for split in ("train", "tune", "test"):
        group = next(f"fixture-{i}" for i in range(1000) if assign_split(f"fixture-{i}") == split)
        groups[split] = group
        gt.add(GroundTruth(cluster_id=group, scenario_family="fixture",
                           label=GroundTruthLabel.ABUSIVE))
        gt.add(SplitGroup(account_id="a-" + group, split_group_id=group))
    gt.flush()
    world = record_world_receipts(inf, gt, {"world_version": "fixture-v1"}, tier="tier2")
    inf.commit()
    gt.commit()
    return world, groups


def test_open_label_queries_are_sql_scoped(inference_session, gt_session):
    _, groups = _world(inference_session, gt_session)
    queries = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        if "SELECT" in statement and "ground_truth" in statement:
            queries.append((statement, parameters))

    engine = gt_session.get_bind()
    event.listen(engine, "before_cursor_execute", capture)
    try:
        view = load_ground_truth(
            gt_session, inf_session=inference_session, splits=("train", "tune")
        )
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert set(view.labels) == {groups["train"], groups["tune"]}
    assert queries
    assert all("WHERE" in sql and groups["test"] not in params for sql, params in queries)


def test_test_request_refuses_before_label_query(inference_session, gt_session):
    _world(inference_session, gt_session)
    queries = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        if "SELECT" in statement and "ground_truth" in statement:
            queries.append(statement)

    engine = gt_session.get_bind()
    event.listen(engine, "before_cursor_execute", capture)
    try:
        with pytest.raises(EvaluationAccessError, match="one-shot"):
            load_ground_truth(gt_session, inf_session=inference_session, splits=("test",))
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert not queries
    with pytest.raises(EvaluationAccessError, match="both verified"):
        load_ground_truth(gt_session)


def test_finalization_is_one_shot_even_with_copied_grant(inference_session, gt_session):
    _, groups = _world(inference_session, gt_session)
    grant = begin_finalization(inference_session, gt_session, {"fixture": "frozen-model"})
    copied = replace(grant)
    view = load_ground_truth(gt_session, inf_session=inference_session,
                             splits=("test",), finalization=grant)
    assert set(view.labels) == {groups["test"]}
    complete_finalization(grant, {"fixture_result": True})
    with pytest.raises(EvaluationAccessError, match="already accessed"):
        load_ground_truth(gt_session, inf_session=inference_session,
                          splits=("test",), finalization=copied)
    with pytest.raises(EvaluationAccessError, match="already consumed"):
        begin_finalization(inference_session, gt_session, {"different": "analysis"})
    with pytest.raises(DBAPIError, match="append-only"):
        gt_session.execute(text("DELETE FROM evaluation_finalization_starts"))
    gt_session.rollback()


def test_decoy_table_name_does_not_trip_the_sealed_gate(gt_session):
    # C-5: a table whose name merely *contains* "receipt" or "provenance" (but is
    # not the real world_evaluation_receipt table) must not be mistaken for a
    # loaded, receipted world. Only has_evaluation_receipt's exact-table check
    # decides that. The inverse (a real receipted world fails closed without a
    # finalization grant) is already covered by test_test_request_refuses_before_label_query.
    gt_session.execute(text(
        "CREATE TABLE payment_receipts (id INTEGER PRIMARY KEY, note TEXT)"
    ))
    gt_session.commit()
    view = load_ground_truth(gt_session, splits=("train", "tune"))
    assert view.labels == {}


def test_started_failure_still_consumes_touch(inference_session, gt_session):
    _world(inference_session, gt_session)
    begin_finalization(inference_session, gt_session, {"fixture": "failed after start"})
    gt_session.rollback()
    with pytest.raises(EvaluationAccessError, match="already consumed"):
        begin_finalization(inference_session, gt_session, {"fixture": "retry"})

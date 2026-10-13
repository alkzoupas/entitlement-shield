"""Anticipated/unanticipated family classifier (docs/features.md §7, plan Step 7).

Covers: the §7.1 rule (anticipated iff every firing feature existed at freeze;
'fires nothing' → unanticipated); byte-determinism; the family-only ground-truth
read; and that the classifier never touches the binary abuse label.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from app.db.ground_truth_family_reader import load_account_families
from app.db.ground_truth_writer import write_ground_truth, write_split_group
from app.models.enums import (
    EmailDomainType,
    EventType,
    GroundTruthLabel,
    Reliability,
    SignalType,
    VerificationLevel,
)
from app.services.classify_families import (
    ANTICIPATED,
    UNANTICIPATED,
    FamilyMapping,
    build_family_tuples,
    classify_families,
    family_labels,
    render_family_mapping_markdown,
)
from app.services.features import Rung
from app.tests.support import AS_OF, Builder


def _serialize(rows) -> str:
    return json.dumps(
        [[r.scenario_family, r.label, list(r.firing_features)] for r in rows],
        sort_keys=True,
    )


def test_rule_anticipated_unanticipated_and_silent() -> None:
    tagged = [
        # farm fires only frozen features → anticipated
        ("farm", "F-DEV-1", True),
        ("farm", "F-VER-1", True),
        ("farm", "F-DOM-2", False),
        # novel family fires a feature NOT in the (narrowed) frozen set → unanticipated
        ("novel", "F-DEV-1", True),
        ("novel", "F-XYZ-9", True),
        # quiet family fires nothing → unanticipated by the 'fires nothing' clause
        ("quiet", "F-DEV-1", False),
    ]
    frozen = frozenset({"F-DEV-1", "F-VER-1"})  # narrowed so F-XYZ-9 is "post-freeze"
    result = {m.scenario_family: m for m in classify_families(tagged, frozen_features=frozen)}

    assert result["farm"].label == ANTICIPATED
    assert result["farm"].firing_features == ("F-DEV-1", "F-VER-1")
    assert result["novel"].label == UNANTICIPATED and "F-XYZ-9" in result["novel"].rationale
    assert result["quiet"].label == UNANTICIPATED


def test_classifier_is_byte_deterministic() -> None:
    tagged = [
        ("b", "F-PAY-1", True),
        ("a", "F-DEV-1", True),
        ("a", "F-CON-1", True),
    ]
    assert _serialize(classify_families(tagged)) == _serialize(classify_families(tagged))
    # Families and firing features are emitted sorted regardless of input order.
    out = classify_families(tagged)
    assert [m.scenario_family for m in out] == ["a", "b"]
    assert out[0].firing_features == ("F-CON-1", "F-DEV-1")


def test_family_labels_reads_family_not_binary(gt_session: Session) -> None:
    write_ground_truth(
        gt_session,
        cluster_id="c1",
        scenario_family="trial_farm",
        label=GroundTruthLabel.ABUSIVE,
    )
    gt_session.flush()
    assert family_labels(gt_session) == {"c1": "trial_farm"}


def test_load_account_families_joins_family_only(gt_session: Session) -> None:
    """account → split_group → scenario_family, never the abuse label (§7.3)."""
    write_ground_truth(
        gt_session, cluster_id="cl-1", scenario_family="Trial Credit Farm",
        label=GroundTruthLabel.ABUSIVE,
    )
    write_split_group(gt_session, account_id="a1", split_group_id="cl-1")
    write_split_group(gt_session, account_id="a2", split_group_id="cl-1")
    write_split_group(gt_session, account_id="orphan", split_group_id="cl-missing")
    gt_session.flush()

    fams = load_account_families(gt_session)
    assert fams == {"a1": "Trial Credit Farm", "a2": "Trial Credit Farm"}
    # Family-only: values are the scenario family string, never a GroundTruthLabel.
    assert all(isinstance(v, str) and v not in GroundTruthLabel.__members__ for v in fams.values())


def _seed_firing_cluster(b: Builder) -> list[str]:
    """A 3-account shared-device disposable cluster that fires F-DEV-1/F-DOM-1/F-VER-1
    at S2, with two aligned cycles so enumeration yields units."""
    members = [
        b.account(f"farm-{i}", domain=EmailDomainType.DISPOSABLE,
                  verification=VerificationLevel.NONE)
        for i in range(3)
    ]
    for aid in members:
        b.observation(aid, SignalType.STABLE_DEVICE, "farm-dev", reliability=Reliability.HIGH)
        b.event(aid, AS_OF - timedelta(days=40), EventType.VERIFICATION_FAILED)
        b.event(aid, AS_OF - timedelta(days=39), EventType.VERIFICATION_FAILED)
        b.cycle(aid, AS_OF - timedelta(days=60))
        b.cycle(aid, AS_OF - timedelta(days=30))
    b.flush()
    return members


def test_build_family_tuples_joins_firing_to_family(
    inference_session: Session, gt_session: Session
) -> None:
    members = _seed_firing_cluster(Builder(inference_session))
    write_ground_truth(
        gt_session, cluster_id="cl-farm", scenario_family="Trial Credit Farm",
        label=GroundTruthLabel.ABUSIVE,
    )
    for aid in members:
        write_split_group(gt_session, account_id=aid, split_group_id="cl-farm")
    gt_session.flush()

    tuples = build_family_tuples(inference_session, gt_session, AS_OF, rung=Rung.S2)

    # Every emitted row is joined to the represented family.
    assert {t[0] for t in tuples} == {"Trial Credit Farm"}
    fired = {feature_id for _, feature_id, was_fired in tuples if was_fired}
    assert {"F-DEV-1", "F-DOM-1", "F-VER-1"} <= fired

    # The producer's output flows into the rule → anticipated (all frozen features).
    [mapping] = classify_families(tuples)
    assert mapping.scenario_family == "Trial Credit Farm"
    assert mapping.label == ANTICIPATED


def test_render_family_mapping_markdown_is_deterministic_and_stamped() -> None:
    mappings = [
        FamilyMapping("Alpha Family", ANTICIPATED, ("F-DEV-1", "F-VER-1"), "all frozen"),
        FamilyMapping("Zeta Family", UNANTICIPATED, (), "fires no feature at or above 0.05"),
    ]
    manifest = {
        "world_version": "world-v1",
        "world_end": "2026-06-30T00:00:00Z",
        "author_model": "anthropic/claude-sonnet — stand-in for qwen",
        "checksums": {"accounts.parquet": "deadbeef", "ground_truth.parquet": "cafef00d"},
    }
    def render() -> str:
        return render_family_mapping_markdown(
            mappings, manifest_hash="5b25e1a0abc", manifest_json=manifest, as_of=AS_OF
        )

    a, b = render(), render()
    assert a == b  # byte-deterministic (no wall-clock)

    assert "5b25e1a0abc" in a  # feature-manifest hash stamped
    assert "world-v1" in a  # world version stamped
    assert "stand-in for qwen" in a  # same-model-family caveat printed
    # Family-only (§7.3): the table has no legitimate/ambiguous/abusive class column.
    assert "| scenario family | verdict | firing features | rationale |" in a
    assert "| Alpha Family | anticipated | F-DEV-1, F-VER-1 | all frozen |" in a
    assert "| Zeta Family | unanticipated | — | fires no feature at or above 0.05 |" in a


def test_mixed_cluster_keeps_minority_and_absent_families(
    inference_session: Session, gt_session: Session,
) -> None:
    members = _seed_firing_cluster(Builder(inference_session))
    for group, family in (("majority", "Majority"), ("minority", "Minority"), ("absent", "Absent")):
        write_ground_truth(gt_session, cluster_id=group, scenario_family=family,
                           label=GroundTruthLabel.LEGITIMATE)
    for index, account in enumerate(members):
        write_split_group(gt_session, account_id=account,
                          split_group_id="majority" if index < 2 else "minority")
    gt_session.flush()
    coverage: dict[str, dict[str, int]] = {}
    result = classify_families(build_family_tuples(
        inference_session, gt_session, AS_OF, coverage=coverage,
    ))
    by_family = {row.scenario_family: row for row in result}
    assert by_family["Majority"].firing_features == by_family["Minority"].firing_features
    assert by_family["Minority"].firing_features
    assert by_family["Absent"].label == UNANTICIPATED
    assert coverage["Majority"]["mixed_clusters"] == coverage["Minority"]["mixed_clusters"] == 1
    assert coverage["Absent"] == {"clusters": 0, "mixed_clusters": 0, "units": 0}


def test_historical_mapping_stamps_actual_cutoff() -> None:
    cutoff = datetime(2025, 1, 2, 3, tzinfo=UTC)
    result = render_family_mapping_markdown(
        [], manifest_hash="frozen", as_of=cutoff,
        manifest_json={"world_end": "2026-06-30T00:00:00Z", "world_version": "world-v2"},
    )
    assert "| as_of | `2025-01-02T03:00:00+00:00` |" in result
    assert "| world_end | `2026-06-30T00:00:00Z` |" in result


def test_family_reader_does_not_select_binary_labels(gt_session: Session) -> None:
    from sqlalchemy import event

    statements: list[str] = []
    bind = gt_session.get_bind()

    def capture(_conn, _cursor, statement, _parameters, _context, _many):
        statements.append(statement)

    event.listen(bind, "before_cursor_execute", capture)
    try:
        family_labels(gt_session)
        load_account_families(gt_session)
    finally:
        event.remove(bind, "before_cursor_execute", capture)
    assert statements
    assert all("ground_truth.label" not in query and "ground_truth.notes" not in query
               for query in statements)

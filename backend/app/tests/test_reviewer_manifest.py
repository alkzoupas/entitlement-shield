"""The reviewer freeze-artifact gate (docs/decisions.md §12 L-8; plan P3.9).

Proves the machine-checkable claim behind `config/reviewer-manifest.json`: it
regenerates byte-for-byte from the live code, it carries every field L-8 names, its
content hash is self-consistent and is the one stamped onto `LlmReview` rows and
audit events, and it is *sensitive* — moving a prompt template byte, the output
schema, the provider id, the model snapshot, the temperature, the sample count, the
retry budget, or any reviewer source file moves the hash.

Mirrors `test_feature_manifest.py`'s structure deliberately: the two artifacts are
declared independent by L-8, so they get parallel-but-separate gates rather than a
shared helper that could couple them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from app.models.enums import ProposedAction
from app.providers.factory import describe_provider, load_provider_config
from app.services import fake_reviewer as FR
from app.services import feature_manifest as FM
from app.services import reviewer as R
from app.services import reviewer_manifest as M
from app.services import reviewer_validation as RV
from app.tests.test_feature_manifest import FROZEN_FEATURE_MANIFEST_HASH


def test_manifest_regenerates_byte_for_byte() -> None:
    """The P3.9 gate: the committed artifact == regeneration from live config."""
    assert M.MANIFEST_PATH.read_bytes() == M.manifest_bytes()


def test_content_hash_is_self_consistent() -> None:
    committed = json.loads(M.MANIFEST_PATH.read_text())
    embedded = committed.pop("content_hash")
    assert M.content_hash(committed) == embedded == M.MANIFEST_HASH


def test_manifest_version_is_its_own_namespace() -> None:
    """L-8: a separate artifact from `feature-manifest.json`. A shared version
    namespace would make a reviewer change look like a detector re-freeze."""
    assert M.MANIFEST_VERSION == "reviewer-manifest-v1"
    assert M.MANIFEST_VERSION != FM.MANIFEST_VERSION
    assert M.MANIFEST_HASH != FM.MANIFEST_HASH
    assert M.MANIFEST_PATH != FM.MANIFEST_PATH
    assert not issubclass(M.ReviewerManifestDrift, FM.ManifestDrift)
    assert not issubclass(FM.ManifestDrift, M.ReviewerManifestDrift)


# --- The scope tripwire (the single most important assertion here) ------------


def test_feature_manifest_untouched_by_reviewer_manifest() -> None:
    """L-8's explicit constraint: `config/reviewer-manifest.json` is "a separate
    artifact from `feature-manifest.json`, **which must not change**".

    P3.9 adds a whole reviewer manifest and a `config/costs.yaml` `llm:` block;
    neither may perturb anything the detector fires on. Importing
    `reviewer_manifest` must not move the feature manifest either — this asserts
    the detector freeze is intact with `reviewer_manifest` already imported into
    the same process. (`test_feature_manifest.test_feature_manifest_hash_unchanged`
    makes the same assertion standalone; both are cheap and they fail for
    different reasons.)"""
    assert FM.MANIFEST_HASH == FROZEN_FEATURE_MANIFEST_HASH
    assert FM.MANIFEST_PATH.read_bytes() == FM.manifest_bytes()


# --- L-8 field coverage -------------------------------------------------------


def test_manifest_carries_every_l8_field() -> None:
    manifest = M.build_manifest()
    assert set(manifest) == {
        "manifest_version",
        "prompt_templates",
        "output_schema",
        "provider",
        "sampling",
        "gating_rule",
        "auto_apply_thresholds",
        "fake_reviewer_rule_table",
        "reviewer_code_fingerprint",
        "content_hash",
    }


def test_prompt_template_hashes_match_the_source_bytes() -> None:
    """Both variants, hashed as raw bytes — nothing about the templates is
    restated in Python where it could drift from the files the model actually
    receives."""
    templates = M.build_manifest()["prompt_templates"]
    assert set(templates) == {"reviewer_v1", "reviewer_timeline_only_v1"}
    for variant, filename in M._PROMPT_TEMPLATES.items():
        raw = (M.PROMPTS_DIR / filename).read_bytes()
        assert templates[variant]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert templates[variant]["bytes"] == len(raw)


def test_output_schema_hash_matches_the_source_bytes() -> None:
    raw = (M.PROMPTS_DIR / "output_schema_v1.json").read_bytes()
    entry = M.build_manifest()["output_schema"]
    assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
    assert entry["bytes"] == len(raw)


def test_provider_and_sampling_blocks_read_live() -> None:
    """Phase 4 M-7: the `provider` block reflects whichever provider is `active`
    in the committed `config/reviewer-provider.yaml`, not a hardcoded fake
    constant — a provider switch is expected to move this manifest's hash."""
    manifest = M.build_manifest()
    assert manifest["provider"] == describe_provider(load_provider_config())
    assert manifest["sampling"] == {
        "temperature": R.REVIEWER_TEMPERATURE,
        "sample_count": R.SAMPLE_COUNT,
    }
    # C-5's three samples and the deterministic temperature are contractual.
    assert manifest["sampling"] == {"temperature": 0.0, "sample_count": 3}


def test_gating_rule_block_states_the_l9_gate_and_c5_unanimity() -> None:
    gate = M.build_manifest()["gating_rule"]
    assert gate["invokes_on"] == "any decision with proposed_action != allow"
    assert "unanimity" in gate["combination_rule"]
    assert gate["max_repair_retries"] == RV.MAX_RETRIES == 1
    assert gate["samples_per_decision"] == R.SAMPLE_COUNT
    # Every skip branch `_gating_reason` can return is declared.
    skips = " | ".join(gate["skip_conditions"])
    for fragment in ("allow", "terminal", "under_review", "Suppression"):
        assert fragment in skips
    # The arm table is the thing arm 4b's honesty depends on.
    assert gate["arm_views"]["arm4b_timeline_only"] == {
        "view": "timeline_only",
        "prompt_variant": "reviewer_timeline_only_v1",
        "prompt_version": "reviewer_timeline_only_v1",
    }
    assert gate["arm_views"]["reviewer"]["view"] == "full"


def test_auto_apply_thresholds_read_the_live_severity_ceiling() -> None:
    thresholds = M.build_manifest()["auto_apply_thresholds"]
    assert thresholds["ceiling_action"] == ProposedAction.LIGHT_FRICTION.value
    assert thresholds["requires_unanimous_reverse"] is True
    assert thresholds["requires_no_other_actioned_cluster_member"] is True


def test_fake_reviewer_rule_table_reads_live() -> None:
    table = M.build_manifest()["fake_reviewer_rule_table"]
    assert table["action_ladder"] == [a.value for a in FR._ACTION_LADDER]
    assert table["disagree_variant_offsets"] == list(FR._DISAGREE_VARIANT_OFFSETS)
    assert "sha256" in table["seed_mechanism"]


def test_code_fingerprint_present_and_excludes_review_case() -> None:
    """`review_case.py` is shared P3.1 infrastructure read by the console and API
    too; a change there is not a change to the reviewer's declared identity."""
    manifest = M.build_manifest()
    assert manifest["reviewer_code_fingerprint"] == M._reviewer_code_fingerprint()
    fingerprinted = {mod.__name__ for mod in M._REVIEWER_MODULES}
    assert fingerprinted == {
        "app.services.reviewer",
        "app.services.reviewer_validation",
        "app.services.escalation",
        "app.services.fake_reviewer",
    }
    assert "app.services.review_case" not in fingerprinted


# --- Sensitivity: every hashed input actually moves the hash -------------------


@pytest.mark.parametrize(
    "mod,attr,newval",
    [
        (R, "REVIEWER_TEMPERATURE", 0.7),
        (R, "SAMPLE_COUNT", 5),
        (RV, "MAX_RETRIES", 3),
    ],
)
def test_live_constant_change_moves_hash(monkeypatch, mod, attr, newval) -> None:
    base = M.content_hash()
    monkeypatch.setattr(mod, attr, newval)
    assert M.content_hash() != base, f"{mod.__name__}.{attr} not covered by the hash"


def test_active_provider_switch_moves_hash(monkeypatch) -> None:
    """Phase 4 M-7: switching `active` (here: away from the committed config's
    `openai_compatible` to `fake`) must move the manifest hash, and the `provider`
    block must then read the fake reviewer's own id/model — not require an API key
    to do so."""
    base = M.content_hash()
    fake_config = dataclasses.replace(
        load_provider_config(), active="fake", fake={"mode": "normal"}
    )
    monkeypatch.setattr(M, "load_provider_config", lambda: fake_config)
    assert M.content_hash() != base
    assert describe_provider(fake_config) == {
        "provider_id": FR.PROVIDER_ID,
        "model_snapshot": FR.MODEL_SNAPSHOT,
    }


def test_active_provider_model_change_moves_hash(monkeypatch) -> None:
    """A model bump for the currently-active provider (no provider switch) must
    also move the hash — the manifest names the actually-configured model, not
    just which adapter is selected."""
    base = M.content_hash()
    original = load_provider_config()
    changed = dataclasses.replace(
        original,
        openai_compatible={**original.openai_compatible, "model": "a-different-model"},
    )
    monkeypatch.setattr(M, "load_provider_config", lambda: changed)
    assert M.content_hash() != base


def test_arm_config_change_moves_hash(monkeypatch) -> None:
    base = M.content_hash()
    monkeypatch.setitem(R._ARM_CONFIG, "arm4b_timeline_only", ("full", "reviewer_v1", "x"))
    assert M.content_hash() != base


def test_action_ladder_change_moves_hash(monkeypatch) -> None:
    base = M.content_hash()
    monkeypatch.setattr(FR, "_ACTION_LADDER", (ProposedAction.ALLOW,))
    assert M.content_hash() != base


@pytest.mark.parametrize(
    "filename", ["reviewer_v1.md", "reviewer_timeline_only_v1.md", "output_schema_v1.json"]
)
def test_prompt_or_schema_edit_moves_hash(monkeypatch, tmp_path, filename) -> None:
    """A one-word instruction edit re-freezes: the templates are hashed as bytes,
    so no prompt change can regenerate byte-identically."""
    base = M.content_hash()
    edited = tmp_path / "prompts"
    edited.mkdir()
    for name in (*M._PROMPT_TEMPLATES.values(), "output_schema_v1.json"):
        source = (M.PROMPTS_DIR / name).read_bytes()
        edited.joinpath(name).write_bytes(
            source + b"\n<!-- semantic change -->\n" if name == filename else source
        )
    monkeypatch.setattr(M, "PROMPTS_DIR", edited)
    assert M.content_hash() != base


def test_source_edit_to_reviewer_module_moves_hash(monkeypatch, tmp_path) -> None:
    """The belt to the config braces: a change to unanimity, the citation filter,
    the clamp, the blast-radius guard or the escalation lifecycle edits code, not a
    constant, so only the fingerprint catches it."""
    base = M.content_hash()
    fake = tmp_path / "reviewer_edited.py"
    fake.write_bytes(M.Path(R.__file__).read_bytes() + b"\n# semantic change\n")
    monkeypatch.setattr(R, "__file__", str(fake))
    assert M.content_hash() != base


def test_review_case_edit_does_not_move_hash(monkeypatch, tmp_path) -> None:
    """The deliberate exclusion, asserted rather than only documented: editing
    shared P3.1 case-building infrastructure must NOT re-freeze the reviewer."""
    from app.services import review_case as RC

    base = M.content_hash()
    fake = tmp_path / "review_case_edited.py"
    fake.write_bytes(M.Path(RC.__file__).read_bytes() + b"\n# unrelated change\n")
    monkeypatch.setattr(RC, "__file__", str(fake))
    assert M.content_hash() == base


# --- verified_manifest_hash: the four accept/reject cases ---------------------


def test_verified_manifest_hash_accepts_committed() -> None:
    assert M.verified_manifest_hash() == M.MANIFEST_HASH


def test_verified_manifest_hash_rejects_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(M, "MANIFEST_PATH", tmp_path / "absent.json")
    with pytest.raises(M.ReviewerManifestDrift):
        M.verified_manifest_hash()


def test_verified_manifest_hash_rejects_stale(monkeypatch, tmp_path) -> None:
    stale = tmp_path / "stale.json"
    stale.write_bytes(
        M.manifest_bytes().replace(b'"sample_count": 3', b'"sample_count": 4')
    )
    monkeypatch.setattr(M, "MANIFEST_PATH", stale)
    with pytest.raises(M.ReviewerManifestDrift):
        M.verified_manifest_hash()


def test_verified_manifest_hash_rejects_malformed(monkeypatch, tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"{ not json")
    monkeypatch.setattr(M, "MANIFEST_PATH", bad)
    with pytest.raises(M.ReviewerManifestDrift):
        M.verified_manifest_hash()


def test_write_manifest_round_trips(tmp_path) -> None:
    out = tmp_path / "reviewer-manifest.json"
    assert M.write_manifest(out) == M.MANIFEST_HASH
    assert out.read_bytes() == M.manifest_bytes()

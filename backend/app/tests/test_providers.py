"""Phase 4 plan P4.4/P4.5/P4.6 (`docs/decisions.md §13` M-4/M-5/M-6): real provider
adapters, config/factory, and the response cache.

Every SDK call is mocked — no real network reaches Anthropic or a local model
server from this suite. `test_llm_provider.py` already proves no `services/**.py`
module imports a network-capable module; this file additionally proves
`providers/**.py` sits outside that scan's coverage and genuinely needs to
import one (M-4's "exemption is intentional, not a silent gap").
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import pytest

from app.providers.anthropic_provider import AnthropicReviewerProvider
from app.providers.cache import CachingProvider, ResponseCache, request_fingerprint
from app.providers.factory import (
    ProviderConfig,
    ProviderConfigError,
    build_provider,
    load_provider_config,
)
from app.providers.openai_compatible_provider import OpenAICompatibleReviewerProvider
from app.services.fake_reviewer import FakeReviewerProvider
from app.services.llm_provider import (
    LLMRequest,
    LLMResponse,
    ProviderError,
    ProviderRefusal,
    ProviderTimeout,
    TokenUsage,
)
from app.services.prompt_builder import (
    StructuredEvidenceSegment,
    UntrustedBlock,
    _build_untrusted_block,
)
from app.tests.test_llm_provider import NETWORK_CAPABLE_MODULES, _imported_modules

PROVIDERS_DIR = pathlib.Path(__file__).resolve().parents[1] / "providers"


def _providers_modules() -> list[pathlib.Path]:
    return [p for p in PROVIDERS_DIR.rglob("*.py") if "__pycache__" not in p.parts]


def _request(*, temperature: float = 0.0, max_tokens: int = 512) -> LLMRequest:
    return LLMRequest(
        system="You are a reviewer.",
        segments=(
            StructuredEvidenceSegment(text="case evidence"),
            UntrustedBlock(text="<<APPEAL>>hello<</APPEAL>>"),
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_s=30.0,
    )


# --------------------------------------------------------------------------
# M-4: providers/ is deliberately outside the services/ AST no-network scan.
# --------------------------------------------------------------------------


def test_providers_dir_is_not_covered_by_services_scan() -> None:
    services_dir = PROVIDERS_DIR.parents[0] / "services"
    assert PROVIDERS_DIR != services_dir
    assert not PROVIDERS_DIR.is_relative_to(services_dir)


def test_providers_genuinely_need_the_network_import_exemption() -> None:
    found: set[str] = set()
    for py in _providers_modules():
        found |= _imported_modules(py) & NETWORK_CAPABLE_MODULES | (
            _imported_modules(py) & {"openai", "anthropic"}
        )
    # openai/anthropic aren't in NETWORK_CAPABLE_MODULES (that set targets raw
    # transport libraries), so check them directly — this is the actual proof
    # the exemption is load-bearing, not vestigial.
    all_imports: set[str] = set()
    for py in _providers_modules():
        all_imports |= _imported_modules(py)
    assert "openai" in all_imports
    assert "anthropic" in all_imports


# --------------------------------------------------------------------------
# OpenAICompatibleReviewerProvider (P4.5)
# --------------------------------------------------------------------------


def _openai_choice(*, content: str, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )


def _openai_response(
    *, content: str, finish_reason: str = "stop", model: str = "qwen3-coder"
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[_openai_choice(content=content, finish_reason=finish_reason)],
        model=model,
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _patch_openai_client(monkeypatch: pytest.MonkeyPatch, mock_client: MagicMock) -> None:
    import app.providers.openai_compatible_provider as mod

    monkeypatch.setattr(mod.openai, "OpenAI", lambda **kwargs: mock_client)


def test_openai_compatible_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_response(content='{"ok": true}')
    _patch_openai_client(monkeypatch, client)

    provider = OpenAICompatibleReviewerProvider(
        model="qwen3-coder-30b-a3b-instruct", base_url="http://127.0.0.1:8000/v1", api_key="x"
    )
    response = provider.complete(_request())

    assert response.text == '{"ok": true}'
    assert response.model_snapshot == "qwen3-coder"
    assert response.usage == TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert response.latency_ms >= 0.0


def test_openai_compatible_forwards_extra_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """P4.8 finding: this MTPLX/vLLM-family server needs `chat_template_kwargs:
    {enable_thinking: false}` passed as `extra_body`, or a reasoning model burns
    its whole `max_tokens` budget on hidden reasoning and never emits `content`.
    `extra_body` must reach the SDK call unmodified — it is a vendor-specific
    passthrough, never a named field on this adapter (a real hosted OpenAI
    endpoint would 400 on an unrecognized top-level field)."""
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_response(content='{"ok": true}')
    _patch_openai_client(monkeypatch, client)

    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    provider = OpenAICompatibleReviewerProvider(
        model="m", base_url="http://x/v1", api_key="x", extra_body=extra_body
    )
    provider.complete(_request())

    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["extra_body"] == extra_body


def test_openai_compatible_appeals_omit_renders_an_absent_appeal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P5-8 appeals-off arm: `appeals: omit` sends the markers with an empty body,
    byte-identical to what an appeal-free case renders; `include` sends the body."""
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_response(content='{"ok": true}')
    _patch_openai_client(monkeypatch, client)

    def request(appeal: str | None) -> LLMRequest:
        return LLMRequest(
            system="You are a reviewer.",
            segments=(
                StructuredEvidenceSegment(text="case evidence"),
                _build_untrusted_block(appeal, "d" * 32),
            ),
            temperature=0.0, max_tokens=512, timeout_s=30.0,
        )

    def sent(appeals: str, appeal: str | None) -> str:
        OpenAICompatibleReviewerProvider(
            model="m", base_url="http://x/v1", api_key="x", appeals=appeals
        ).complete(request(appeal))
        _, kwargs = client.chat.completions.create.call_args
        return str(kwargs["messages"][1]["content"])

    assert "please reconsider" in sent("include", "please reconsider")
    assert sent("omit", "please reconsider") == sent("include", None)
    with pytest.raises(ValueError, match="appeals must be one of"):
        OpenAICompatibleReviewerProvider(
            model="m", base_url="http://x/v1", api_key="x", appeals="summarise"
        )


def test_openai_compatible_maps_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.providers.openai_compatible_provider as mod

    client = MagicMock()
    client.chat.completions.create.side_effect = mod.openai.APITimeoutError(request=MagicMock())
    _patch_openai_client(monkeypatch, client)

    provider = OpenAICompatibleReviewerProvider(model="m", base_url="http://x/v1", api_key="x")
    with pytest.raises(ProviderTimeout):
        provider.complete(_request())


def test_openai_compatible_maps_generic_sdk_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.providers.openai_compatible_provider as mod

    client = MagicMock()
    client.chat.completions.create.side_effect = mod.openai.APIConnectionError(request=MagicMock())
    _patch_openai_client(monkeypatch, client)

    provider = OpenAICompatibleReviewerProvider(model="m", base_url="http://x/v1", api_key="x")
    with pytest.raises(ProviderError):
        provider.complete(_request())


def test_openai_compatible_content_filter_is_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_response(
        content="", finish_reason="content_filter"
    )
    _patch_openai_client(monkeypatch, client)

    provider = OpenAICompatibleReviewerProvider(model="m", base_url="http://x/v1", api_key="x")
    with pytest.raises(ProviderRefusal):
        provider.complete(_request())


def test_openai_compatible_empty_content_on_truncated_reasoning_is_not_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reasoning-heavy models can spend the whole budget on `reasoning_content` and
    return `finish_reason == "length"` with `content` empty (docstring-documented
    live finding). This must surface as an empty `text` — a parse failure for R-7's
    repair retry — never as `ProviderRefusal` or a silently-substituted value."""
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_response(
        content="", finish_reason="length"
    )
    _patch_openai_client(monkeypatch, client)

    provider = OpenAICompatibleReviewerProvider(model="m", base_url="http://x/v1", api_key="x")
    response = provider.complete(_request())
    assert response.text == ""


# --------------------------------------------------------------------------
# AnthropicReviewerProvider (P4.4)
# --------------------------------------------------------------------------


def _patch_anthropic_client(monkeypatch: pytest.MonkeyPatch, mock_client: MagicMock) -> None:
    import app.providers.anthropic_provider as mod

    monkeypatch.setattr(mod.anthropic, "Anthropic", lambda **kwargs: mock_client)


def _anthropic_response(
    *, blocks: list[object], stop_reason: str = "end_turn", model: str = "claude-haiku-4-5"
) -> SimpleNamespace:
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        stop_details=None,
        model=model,
        usage=SimpleNamespace(input_tokens=20, output_tokens=8),
    )


def test_anthropic_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.messages.create.return_value = _anthropic_response(
        blocks=[anthropic.types.TextBlock(type="text", text='{"ok": true}')]
    )
    _patch_anthropic_client(monkeypatch, client)

    provider = AnthropicReviewerProvider(model="claude-haiku-4-5", api_key="x")
    response = provider.complete(_request())

    assert response.text == '{"ok": true}'
    assert response.model_snapshot == "claude-haiku-4-5"
    assert response.usage == TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28)


def test_anthropic_joins_only_text_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.messages.create.return_value = _anthropic_response(
        blocks=[
            anthropic.types.TextBlock(type="text", text="a"),
            SimpleNamespace(type="tool_use", text="ignored"),
            anthropic.types.TextBlock(type="text", text="b"),
        ]
    )
    _patch_anthropic_client(monkeypatch, client)

    provider = AnthropicReviewerProvider(model="claude-haiku-4-5", api_key="x")
    response = provider.complete(_request())
    assert response.text == "ab"


def test_anthropic_maps_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.providers.anthropic_provider as mod

    client = MagicMock()
    client.messages.create.side_effect = mod.anthropic.APITimeoutError(request=MagicMock())
    _patch_anthropic_client(monkeypatch, client)

    provider = AnthropicReviewerProvider(model="claude-haiku-4-5", api_key="x")
    with pytest.raises(ProviderTimeout):
        provider.complete(_request())


def test_anthropic_maps_generic_sdk_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.providers.anthropic_provider as mod

    client = MagicMock()
    client.messages.create.side_effect = mod.anthropic.APIConnectionError(request=MagicMock())
    _patch_anthropic_client(monkeypatch, client)

    provider = AnthropicReviewerProvider(model="claude-haiku-4-5", api_key="x")
    with pytest.raises(ProviderError):
        provider.complete(_request())


def test_anthropic_stop_reason_refusal_maps_to_provider_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.messages.create.return_value = _anthropic_response(
        blocks=[], stop_reason="refusal"
    )
    _patch_anthropic_client(monkeypatch, client)

    provider = AnthropicReviewerProvider(model="claude-haiku-4-5", api_key="x")
    with pytest.raises(ProviderRefusal):
        provider.complete(_request())


# --------------------------------------------------------------------------
# providers/factory.py (P4.6)
# --------------------------------------------------------------------------


def _write_config(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
    path = tmp_path / "reviewer-provider.yaml"
    path.write_text(body)
    return path


def test_load_provider_config_missing_file_fails_closed(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ProviderConfigError):
        load_provider_config(tmp_path / "does-not-exist.yaml")


def test_load_provider_config_rejects_unrecognized_active(tmp_path: pathlib.Path) -> None:
    path = _write_config(tmp_path, "active: some_made_up_provider\n")
    with pytest.raises(ProviderConfigError):
        load_provider_config(path)


def test_load_provider_config_rejects_missing_active_key(tmp_path: pathlib.Path) -> None:
    path = _write_config(tmp_path, "anthropic:\n  model: claude-haiku-4-5\n")
    with pytest.raises(ProviderConfigError):
        load_provider_config(path)


def test_build_provider_fake() -> None:
    config = ProviderConfig(
        active="fake", anthropic={}, openai_compatible={}, fake={"mode": "normal"}
    )
    provider = build_provider(config)
    assert isinstance(provider, FakeReviewerProvider)


def test_build_provider_openai_compatible_defaults_key_for_local_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = ProviderConfig(
        active="openai_compatible",
        anthropic={},
        openai_compatible={"model": "m", "base_url": "http://127.0.0.1:8000/v1"},
        fake={},
    )
    provider = build_provider(config)
    assert isinstance(provider, OpenAICompatibleReviewerProvider)
    assert provider.model == "m"
    assert provider.base_url == "http://127.0.0.1:8000/v1"
    assert provider.api_key == "not-needed-for-local-server"
    assert provider.extra_body is None


def test_build_provider_openai_compatible_reads_extra_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = ProviderConfig(
        active="openai_compatible",
        anthropic={},
        openai_compatible={
            "model": "m",
            "base_url": "http://127.0.0.1:8000/v1",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        fake={},
    )
    provider = build_provider(config)
    assert isinstance(provider, OpenAICompatibleReviewerProvider)
    assert provider.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_build_provider_anthropic_fails_closed_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = ProviderConfig(
        active="anthropic",
        anthropic={"model": "claude-haiku-4-5"},
        openai_compatible={},
        fake={},
    )
    with pytest.raises(ProviderConfigError):
        build_provider(config)


def test_build_provider_anthropic_reads_key_from_named_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    config = ProviderConfig(
        active="anthropic",
        anthropic={"model": "claude-haiku-4-5", "timeout_s": 12.0},
        openai_compatible={},
        fake={},
    )
    provider = build_provider(config)
    assert isinstance(provider, AnthropicReviewerProvider)
    assert provider.model == "claude-haiku-4-5"
    assert provider.api_key == "sk-test-123"
    assert provider.timeout_s == 12.0


def test_the_committed_reviewer_provider_yaml_loads_and_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`config/reviewer-provider.yaml` as committed to the repo must itself parse
    and build without error under its own `active` selection — this is the config
    `app/api/app.py` and `scripts/arm4_run.py` actually load at runtime."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = load_provider_config()
    provider = build_provider(config)
    assert hasattr(provider, "complete")


# --------------------------------------------------------------------------
# providers/cache.py (P4.6, M-6)
# --------------------------------------------------------------------------


def test_request_fingerprint_ignores_untrusted_block_content() -> None:
    base = _request()
    varied = LLMRequest(
        system=base.system,
        segments=(
            StructuredEvidenceSegment(text="case evidence"),
            UntrustedBlock(text="<<APPEAL>>completely different attacker text<</APPEAL>>"),
        ),
        temperature=base.temperature,
        max_tokens=base.max_tokens,
        timeout_s=base.timeout_s,
    )
    assert request_fingerprint(base) == request_fingerprint(varied)


def _with_appeal(appeal: str | None, delimiter: str) -> LLMRequest:
    base = _request()
    return LLMRequest(
        system=base.system,
        segments=(
            StructuredEvidenceSegment(text="case evidence"),
            _build_untrusted_block(appeal, delimiter),
        ),
        temperature=base.temperature,
        max_tokens=base.max_tokens,
        timeout_s=base.timeout_s,
    )


def test_request_fingerprint_hashes_a_non_empty_appeal_body_only() -> None:
    """P5-7: an appeal-reading provider must not replay an answer recorded for a
    different appeal, while the random delimiter and empty appeals stay out."""
    same_a = _with_appeal("We are roommates.", "delim-one")
    same_b = _with_appeal("We are roommates.", "delim-two")
    other = _with_appeal("Only I used this account.", "delim-one")
    assert request_fingerprint(same_a) == request_fingerprint(same_b)
    assert request_fingerprint(same_a) != request_fingerprint(other)
    no_block = LLMRequest(
        system=same_a.system,
        segments=(StructuredEvidenceSegment(text="case evidence"),),
        temperature=same_a.temperature,
        max_tokens=same_a.max_tokens,
        timeout_s=same_a.timeout_s,
    )
    assert request_fingerprint(_with_appeal(None, "delim-one")) == request_fingerprint(no_block)


def test_request_fingerprint_ignores_decision_id_line() -> None:
    """`prompt_builder._format_structured_evidence` embeds a literal
    `decision_id: <uuid4>` line — freshly assigned by the detector on every
    reseed, not case content. Left in, an identical logical case would fingerprint
    differently across separate `arm4_run.py` invocations, breaking --replay
    entirely (found by an actual two-invocation smoke test against a real
    reseeded world, not just a unit-level guess)."""
    base = LLMRequest(
        system="sys",
        segments=(
            StructuredEvidenceSegment(
                text="## Structured evidence\n"
                "decision_id: 11111111-aaaa-bbbb-cccc-000000000001\n"
                "account_id: acct-1"
            ),
        ),
        temperature=0.0,
        max_tokens=512,
        timeout_s=30.0,
    )
    reseeded = LLMRequest(
        system="sys",
        segments=(
            StructuredEvidenceSegment(
                text="## Structured evidence\n"
                "decision_id: 99999999-ffff-eeee-dddd-000000000009\n"
                "account_id: acct-1"
            ),
        ),
        temperature=0.0,
        max_tokens=512,
        timeout_s=30.0,
    )
    assert request_fingerprint(base) == request_fingerprint(reseeded)


def test_request_fingerprint_changes_with_decision_relevant_content() -> None:
    base = _request()
    changed = LLMRequest(
        system=base.system,
        segments=(
            StructuredEvidenceSegment(text="different case evidence"),
            UntrustedBlock(text="<<APPEAL>>hello<</APPEAL>>"),
        ),
        temperature=base.temperature,
        max_tokens=base.max_tokens,
        timeout_s=base.timeout_s,
    )
    assert request_fingerprint(base) != request_fingerprint(changed)


def test_response_cache_save_load_roundtrip(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "transcript.jsonl"
    cache = ResponseCache(path=path)
    response = LLMResponse(
        text="hi", model_snapshot="m", latency_ms=1.5,
        usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    cache.put("k1", response)
    cache.save()

    reloaded = ResponseCache(path=path)
    reloaded.load()
    got = reloaded.get("k1")
    assert got == response


def test_caching_provider_replays_recorded_response_without_calling_inner(
    tmp_path: pathlib.Path,
) -> None:
    inner = MagicMock()
    response = LLMResponse(
        text="hi", model_snapshot="m", latency_ms=1.0,
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    inner.complete.return_value = response

    cache = ResponseCache(path=tmp_path / "t.jsonl")
    live = CachingProvider(inner=inner, cache=cache)
    first = live.complete(_request())
    assert first == response
    assert inner.complete.call_count == 1

    replay = CachingProvider(inner=inner, cache=cache, replay_only=True)
    second = replay.complete(_request())
    assert second == response
    assert inner.complete.call_count == 1  # inner never called again


def test_caching_provider_replay_only_raises_on_cache_miss(tmp_path: pathlib.Path) -> None:
    inner = MagicMock()
    cache = ResponseCache(path=tmp_path / "t.jsonl")
    provider = CachingProvider(inner=inner, cache=cache, replay_only=True)
    with pytest.raises(KeyError):
        provider.complete(_request())
    inner.complete.assert_not_called()


def test_caching_provider_indexes_repeated_calls_with_same_fingerprint(
    tmp_path: pathlib.Path,
) -> None:
    inner = MagicMock()
    inner.complete.side_effect = [
        LLMResponse(text="first", model_snapshot="m", latency_ms=1.0, usage=TokenUsage(0, 0, 0)),
        LLMResponse(text="second", model_snapshot="m", latency_ms=1.0, usage=TokenUsage(0, 0, 0)),
    ]
    cache = ResponseCache(path=tmp_path / "t.jsonl")
    provider = CachingProvider(inner=inner, cache=cache)

    first = provider.complete(_request())
    second = provider.complete(_request())
    assert first.text == "first"
    assert second.text == "second"
    assert inner.complete.call_count == 2

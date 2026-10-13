"""Provider selection and construction (Phase 4 plan P4.6, `docs/decisions.md §13`
M-5). `config/reviewer-provider.yaml` names which adapter is active; secrets are
never stored in that file, only the name of the environment variable to read.

`load_provider_config` + `build_provider` is the single construction point this
repo's other call sites (`app/api/app.py`, `scripts/arm4_run.py`) use instead of
hardcoding `FakeReviewerProvider()`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.services.fake_reviewer import MODEL_SNAPSHOT as FAKE_MODEL_SNAPSHOT
from app.services.fake_reviewer import PROVIDER_ID as FAKE_PROVIDER_ID
from app.services.fake_reviewer import FakeMode, FakeReviewerProvider
from app.services.llm_provider import LLMProvider

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "reviewer-provider.yaml"

_VALID_ACTIVE = {"anthropic", "openai_compatible", "fake", "typesafe"}

_logger = logging.getLogger(__name__)


class ProviderConfigError(Exception):
    """`config/reviewer-provider.yaml` is missing, malformed, or names an
    unrecognized `active` provider. Fails closed — never silently falls back to
    a default provider the config didn't ask for."""


@dataclass(frozen=True)
class ProviderConfig:
    active: str
    anthropic: dict[str, Any]
    openai_compatible: dict[str, Any]
    fake: dict[str, Any]
    typesafe: dict[str, Any] = field(default_factory=dict)


def load_provider_config(path: Path | None = None) -> ProviderConfig:
    """Reads `config/reviewer-provider.yaml` by default. An explicit `path`
    always wins; otherwise `REVIEWER_PROVIDER_CONFIG_PATH` (Phase 5 `make demo`
    override, so the demo can force the deterministic `fake` provider without
    editing the committed file's `active:`) is checked before falling back to
    `DEFAULT_CONFIG_PATH`."""
    env_override = os.environ.get("REVIEWER_PROVIDER_CONFIG_PATH")
    config_path = path or (Path(env_override) if env_override else DEFAULT_CONFIG_PATH)
    _logger.info("reviewer provider config resolved to %s", config_path)
    try:
        raw = yaml.safe_load(config_path.read_text())
    except FileNotFoundError as exc:
        raise ProviderConfigError(f"provider config not found: {config_path}") from exc
    if not isinstance(raw, dict) or "active" not in raw:
        raise ProviderConfigError(f"provider config missing 'active': {config_path}")
    active = raw["active"]
    if active not in _VALID_ACTIVE:
        raise ProviderConfigError(
            f"unrecognized active provider {active!r} in {config_path} "
            f"(expected one of {sorted(_VALID_ACTIVE)})"
        )
    return ProviderConfig(
        active=active,
        anthropic=raw.get("anthropic") or {},
        openai_compatible=raw.get("openai_compatible") or {},
        fake=raw.get("fake") or {},
        typesafe=raw.get("typesafe") or {},
    )


def build_provider(config: ProviderConfig) -> LLMProvider:
    """Dispatch on `config.active`. Fails closed — an unrecognized value can't
    reach here (`load_provider_config` already rejected it), but a defensive
    `ProviderConfigError` is raised rather than silently defaulting if this
    function is ever called with a hand-built `ProviderConfig`."""
    if config.active == "fake":
        mode = FakeMode(config.fake.get("mode", "normal"))
        return FakeReviewerProvider(mode=mode)

    if config.active == "anthropic":
        from app.providers.anthropic_provider import AnthropicReviewerProvider

        block = config.anthropic
        api_key_env = block.get("api_key_env", "ANTHROPIC_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ProviderConfigError(
                f"active provider is 'anthropic' but {api_key_env} is not set"
            )
        return AnthropicReviewerProvider(
            model=block["model"],
            api_key=api_key,
            timeout_s=float(block.get("timeout_s", 30.0)),
        )

    if config.active == "openai_compatible":
        from app.providers.openai_compatible_provider import OpenAICompatibleReviewerProvider

        block = config.openai_compatible
        api_key_env = block.get("api_key_env", "OPENAI_API_KEY")
        # A local server (this repo's current default target) typically needs no
        # real key; fall back to a placeholder rather than fail closed, since
        # requiring a key for a same-machine local endpoint would be the wrong
        # failure mode. A real hosted endpoint should still set the named env var.
        api_key = os.environ.get(api_key_env) or "not-needed-for-local-server"
        return OpenAICompatibleReviewerProvider(
            model=block["model"],
            base_url=block["base_url"],
            api_key=api_key,
            timeout_s=float(block.get("timeout_s", 60.0)),
            extra_body=block.get("extra_body"),
            appeals=block.get("appeals", "include"),
        )

    if config.active == "typesafe":
        from app.providers.typesafe_provider import DEFAULT_BASE_URL, TypeSafeReviewerProvider

        block = config.typesafe
        api_key_env = block.get("api_key_env", "TYPESAFE_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ProviderConfigError(
                f"active provider is 'typesafe' but {api_key_env} is not set"
            )
        return TypeSafeReviewerProvider(
            model=block["model"],
            api_key=api_key,
            base_url=block.get("base_url", DEFAULT_BASE_URL),
            timeout_s=float(block.get("timeout_s", 30.0)),
            state_mode=block.get("state", "raw"),
        )

    raise ProviderConfigError(f"unrecognized active provider {config.active!r}")


def describe_provider(config: ProviderConfig) -> dict[str, str]:
    """`provider_id` + declared `model_snapshot` for the active provider, WITHOUT
    constructing a client or requiring a real API key (unlike `build_provider`).
    `reviewer_manifest.py` (L-8) needs to hash "the actually-configured
    provider/model" even when no live key is provisioned yet — e.g. in CI, or
    right after switching `active` in `config/reviewer-provider.yaml` but before
    setting the matching env var — so this reads the model straight out of
    config rather than off a constructed provider instance."""
    if config.active == "fake":
        return {"provider_id": FAKE_PROVIDER_ID, "model_snapshot": FAKE_MODEL_SNAPSHOT}
    if config.active == "anthropic":
        from app.providers.anthropic_provider import PROVIDER_ID

        return {"provider_id": PROVIDER_ID, "model_snapshot": config.anthropic["model"]}
    if config.active == "openai_compatible":
        from app.providers.openai_compatible_provider import PROVIDER_ID

        model = config.openai_compatible["model"]
        if config.openai_compatible.get("appeals", "include") == "omit":
            model = f"{model}+appeals-omitted"
        return {"provider_id": PROVIDER_ID, "model_snapshot": model}
    if config.active == "typesafe":
        from app.providers.typesafe_provider import PROVIDER_ID, snapshot_label

        return {
            "provider_id": PROVIDER_ID,
            "model_snapshot": snapshot_label(
                config.typesafe["model"], config.typesafe.get("state", "raw")
            ),
        }
    raise ProviderConfigError(f"unrecognized active provider {config.active!r}")


__all__ = [
    "ProviderConfig",
    "ProviderConfigError",
    "build_provider",
    "describe_provider",
    "load_provider_config",
    "DEFAULT_CONFIG_PATH",
]

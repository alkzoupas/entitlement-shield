"""P3.4: the provider-neutral interface (docs/phase-3-plan.md P3.4, decisions L-7).

Covers the AST network-import scan (same pattern as `test_ground_truth_boundary.py`,
now guarding "no network-capable dependency added in Phase 3"), `LLMProvider`
structurally matching `FakeReviewerProvider`, and the three normalized error types
being distinct exception classes.
"""

import ast
import pathlib

from app.services.fake_reviewer import FakeReviewerProvider
from app.services.llm_provider import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderError,
    ProviderRefusal,
    ProviderTimeout,
    TokenUsage,
)

APP_DIR = pathlib.Path(__file__).resolve().parents[1]
SERVICES_DIR = APP_DIR / "services"

# Modules that would give a provider a real network path. Not exhaustive of every
# conceivable transport, but covers stdlib and the common third-party HTTP/socket
# clients — enough to make "the fake is the only provider, no network dependency
# exists" an assertion rather than a claim (L-7).
NETWORK_CAPABLE_MODULES = {
    "socket",
    "httpx",
    "requests",
    "urllib",
    "urllib.request",
    "urllib3",
    "aiohttp",
    "http.client",
    "ftplib",
    "smtplib",
    "telnetlib",
    "websocket",
    "websockets",
    "grpc",
}


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def _service_modules() -> list[pathlib.Path]:
    return [p for p in SERVICES_DIR.rglob("*.py") if "__pycache__" not in p.parts]


def test_no_service_imports_a_network_capable_module() -> None:
    for py in _service_modules():
        found = _imported_modules(py) & NETWORK_CAPABLE_MODULES
        assert not found, (
            f"services/{py.relative_to(SERVICES_DIR)} imports network-capable "
            f"module(s) {found} — Phase 3 adds no network dependency (L-7)"
        )


def test_fake_reviewer_satisfies_llm_provider_protocol() -> None:
    fake = FakeReviewerProvider()
    assert isinstance(fake, LLMProvider)
    assert callable(fake.complete)


def test_error_types_are_distinct_exception_classes() -> None:
    assert issubclass(ProviderTimeout, Exception)
    assert issubclass(ProviderError, Exception)
    assert issubclass(ProviderRefusal, Exception)
    assert ProviderTimeout is not ProviderError
    assert ProviderTimeout is not ProviderRefusal
    assert ProviderError is not ProviderRefusal
    assert not issubclass(ProviderTimeout, ProviderRefusal)
    assert not issubclass(ProviderRefusal, ProviderTimeout)
    assert not issubclass(ProviderError, ProviderTimeout)


def test_request_and_response_shapes() -> None:
    usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    response = LLMResponse(
        text="{}", model_snapshot="fake-reviewer-v1", latency_ms=1.0, usage=usage
    )
    assert response.usage.total_tokens == 15

    request = LLMRequest(
        system="sys", segments=(), temperature=0.0, max_tokens=100, timeout_s=5.0
    )
    assert request.temperature == 0.0

"""FastAPI application factory (docs/architecture.md §5).

`create_app()` wires the inference engine, the JSON API, and the console pages. The
engine is injectable so tests bind an in-memory seeded store; in production it is built
from `DATABASE_URL`. The app is **inference-plane only** — it never constructs the
ground-truth engine (locked invariant; asserted by `test_api` boundary test).
"""

from __future__ import annotations

from time import perf_counter

from fastapi import FastAPI
from sqlalchemy.engine import Engine
from starlette.types import ASGIApp, Receive, Scope, Send

from app.api import console, json_api
from app.db.engine import create_inference_schema, make_inference_engine
from app.providers.factory import build_provider, load_provider_config
from app.services.llm_provider import LLMProvider


class RequestClock:
    """Record application entry before parsing/dependencies/threadpool queuing."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope["request_started"] = perf_counter()
        await self.app(scope, receive, send)


def create_app(
    inference_engine: Engine | None = None, reviewer_provider: LLMProvider | None = None
) -> FastAPI:
    app = FastAPI(
        title="EntitlementShield console",
        summary="Detection decisions, the claim-time credit path, and the evaluation view.",
    )
    app.state.inference_engine = inference_engine or make_inference_engine()
    # Phase 4 M-5: the active provider is selected by config/reviewer-provider.yaml,
    # never hardcoded here. A test may still substitute a call-counting wrapper via
    # the `reviewer_provider` argument to prove the endpoint doesn't double-charge.
    app.state.reviewer_provider = reviewer_provider or build_provider(load_provider_config())
    # The local schema migration is create_all: adds ledger/evidence tables without
    # rewriting existing decisions or inventing evidence for legacy cases.
    create_inference_schema(app.state.inference_engine)
    app.add_middleware(RequestClock)
    app.include_router(json_api.router)
    app.include_router(console.router)
    return app

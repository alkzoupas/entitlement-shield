"""Request-scoped dependencies for the API (docs/architecture.md §5).

The inference engine is created once and stashed on `app.state`; each request gets a
fresh SQLAlchemy `Session` bound to it. Write routes call `session.commit()` explicitly
so the commit point is visible at the call site, not hidden in a dependency.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from fastapi import Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.services.llm_provider import LLMProvider

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def get_session(request: Request) -> Iterator[Session]:
    """Yield a Session bound to the app's inference engine (set in `create_app`)."""
    engine = request.app.state.inference_engine
    with Session(engine, future=True) as session:
        yield session


def get_reviewer_provider(request: Request) -> LLMProvider:
    """The one `LLMProvider` instance for this app (set in `create_app`, L-7's only
    registered provider by default). Shared across requests so a test can hand
    `create_app` a call-counting wrapper and assert exactly how many times the
    reviewer endpoint/console route actually invoked it (architecture §7's
    "must not double-charge the LLM")."""
    return request.app.state.reviewer_provider

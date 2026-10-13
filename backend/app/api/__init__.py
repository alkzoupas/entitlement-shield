"""HTTP layer (docs/architecture.md §5): thin FastAPI routes over the services, plus
two server-rendered Jinja/HTMX pages. This package is **inference-plane only** — it
never imports the ground-truth reader or engine (the boundary is asserted by test)."""

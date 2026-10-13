"""Step 0 smoke test: the package imports and the toolchain runs.

This is the only test that must be green to close Step 0 (docs/phase-1-plan.md).
Real tests arrive with the code they cover in Steps 1-8.
"""

import app


def test_package_imports_and_has_version() -> None:
    assert app.__version__ == "0.1.0"


def test_subpackages_are_importable() -> None:
    # The skeleton subpackages exist so later steps have a home to fill.
    import importlib

    for name in ("core", "db", "models", "schemas", "services"):
        importlib.import_module(f"app.{name}")

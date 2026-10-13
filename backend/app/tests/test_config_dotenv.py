"""The `.env` loader advertised by the template actually works (review 8 P2).

A copied `.env` supplies non-default settings, but an already-exported environment
variable always wins over the file.
"""

import importlib

import app.core.config as config
from app.core.config import load_dotenv


def _write_env(path, **values: str) -> None:
    lines = ["# comment line", ""]
    lines += [f"{k}={v}" for k, v in values.items()]
    path.write_text("\n".join(lines) + "\n")


def test_dotenv_populates_missing_keys(tmp_path) -> None:
    env: dict[str, str] = {}
    _write_env(
        tmp_path / ".env",
        DATABASE_URL="sqlite:///data/synthetic/other.db",
        WORLD_SEED="99",
    )
    load_dotenv(tmp_path / ".env", environ=env)
    assert env["DATABASE_URL"] == "sqlite:///data/synthetic/other.db"
    assert env["WORLD_SEED"] == "99"


def test_exported_env_wins_over_file(tmp_path) -> None:
    env = {"WORLD_SEED": "7"}  # already exported
    _write_env(tmp_path / ".env", WORLD_SEED="99")
    load_dotenv(tmp_path / ".env", environ=env)
    assert env["WORLD_SEED"] == "7"  # file did not override


def test_missing_file_is_noop(tmp_path) -> None:
    env: dict[str, str] = {}
    load_dotenv(tmp_path / "does-not-exist.env", environ=env)
    assert env == {}


def test_settings_pick_up_dotenv_value(tmp_path, monkeypatch) -> None:
    # End-to-end through the supported path: the file value flows into Settings when
    # not already exported. Loaded into a scratch env, then applied via monkeypatch
    # so os.environ is restored after the test.
    _write_env(
        tmp_path / ".env",
        DATABASE_URL="sqlite:///data/synthetic/from_dotenv.db",
        WORLD_SEED="42",
    )
    scratch: dict[str, str] = {}
    load_dotenv(tmp_path / ".env", environ=scratch)
    for key, value in scratch.items():
        monkeypatch.setenv(key, value)

    fresh = config.Settings()
    assert fresh.world_seed == 42
    assert fresh.database_url == "sqlite:///data/synthetic/from_dotenv.db"


def test_module_reload_is_clean() -> None:
    # Reloading the module (which loads the repo-root .env) must not explode.
    importlib.reload(config)
    assert isinstance(config.settings.world_seed, int)

"""Runtime settings, read from the environment with dev defaults.

The inference store and the ground-truth store are addressed by SEPARATE URLs
(review 4 P1-4a). Detector services are wired only to `database_url`; the
ground-truth URL is reachable solely through the ground-truth capability modules.

A tiny, dependency-free `.env` loader (`load_dotenv`) runs at import so a copied
`.env` at the repo root actually takes effect — the template advertised this but no
loader existed (review 8 P2). It never overrides already-exported environment
variables: a real exported value always wins over the file.
"""

import os
import pathlib
from collections.abc import MutableMapping
from dataclasses import dataclass, field

# repo root: backend/app/core/config.py -> parents[3]
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_DOTENV_PATH = _REPO_ROOT / ".env"


def load_dotenv(
    path: pathlib.Path,
    *,
    environ: MutableMapping[str, str] | None = None,
    override: bool = False,
) -> None:
    """Load simple KEY=VALUE lines from `path` into `environ` (defaults to
    os.environ). Blank lines and `#` comments are ignored; surrounding quotes are
    stripped from values. Already-set keys are left untouched unless `override`.
    Missing file is a silent no-op. No third-party dependency."""
    env = os.environ if environ is None else environ
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        if override or key not in env:
            env[key] = value


# Load the repo-root .env (if any) before settings are constructed.
load_dotenv(_DOTENV_PATH)


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: _env("DATABASE_URL", "sqlite:///data/synthetic/dev.db")
    )
    ground_truth_database_url: str = field(
        default_factory=lambda: _env(
            "GROUND_TRUTH_DATABASE_URL", "sqlite:///data/synthetic/ground_truth.db"
        )
    )
    world_seed: int = field(default_factory=lambda: int(_env("WORLD_SEED", "1")))
    # Where `scripts/eval_run.py --json` writes the dev-tier evaluation report the
    # console's evaluation page renders. Inference-plane: the page reads this artifact
    # instead of querying ground truth (which the API must never hold — locked invariant).
    eval_report_path: str = field(
        default_factory=lambda: _env("EVAL_REPORT_PATH", "data/synthetic/eval-report.json")
    )


settings = Settings()

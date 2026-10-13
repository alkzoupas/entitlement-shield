"""Pre-grant finalization cutoff and reproducibility contracts."""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path


def canonical_evaluation_cutoff(raw: str, world_end: str) -> datetime:
    """Require the manifest's exact aware-UTC evaluation frame before a grant."""
    try:
        cutoff = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        expected = datetime.fromisoformat(world_end.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit("--as-of and manifest world_end must be ISO-8601 datetimes") from exc
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise SystemExit("--as-of must include an explicit UTC offset")
    if expected.tzinfo is None or expected.utcoffset() is None:
        raise SystemExit("world manifest world_end must include an explicit UTC offset")
    cutoff = cutoff.astimezone(UTC)
    expected = expected.astimezone(UTC)
    if cutoff != expected:
        raise SystemExit("--as-of must equal the verified world's canonical world_end")
    return cutoff


def reproducibility_identity(root: Path) -> dict[str, object]:
    """Capture revision, exact working-tree state, runtime and dependencies."""
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout

    revision = git("rev-parse", "HEAD").strip()
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    diff = git("diff", "--binary", "HEAD")
    packages = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in distributions()
        if dist.metadata.get("Name")
    )
    return {
        "git_revision": revision,
        "git_dirty": bool(status),
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "runtime": sys.version,
        "dependencies_sha256": hashlib.sha256("\n".join(packages).encode()).hexdigest(),
    }

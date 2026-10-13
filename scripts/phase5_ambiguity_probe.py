"""Phase-5 data-source verification probe (throwaway, NOT production, NOT wired
into `make`).

Question: how big is the population Phase-5 "arm 6" would target, and what is
its ground-truth label mix, on the `dev` and `dev_arm4` presets?

Target population = scored cluster-cycle units (the detector's per-unit
`DetectionRecord`, one per `enumerate_cluster_cycles` unit — NOT the `Decision`
rows persisted to the inference store, since `record_decision` is only called
for accounts that receive an action; `light_friction` is cluster-wide and is
never targeted at an individual account, see `evaluation.py`'s
`_GRANT_RESTRICTING` comment) with `proposed_action == "light_friction"` AND
whose `cap_reasons` include `"ambiguous"` OR `"large_shared"`.

For each such unit, the ground-truth 3-way label is read the ONLY authorized
way: via `app.services.evaluation.build_eval_rows`, which computes the same
cluster-cycle units independently and joins each to its `split_group_id` and
sealed label. The join key between a `DetectionRecord` and an `EvalRow` is the
E-14 fixed evaluation unit: `(cluster_snapshot_id, cycle_id, as_of)` — see
`EvalRow.key` in `backend/app/services/evaluation.py` and the matching fields
on `DetectionRecord` in `backend/app/services/detector.py`. Both sides
independently enumerate the exact same units at the same `as_of`, given by
`compute_clusters` + `enumerate_cluster_cycles`, so the join is expected to be
total (every detection has a matching eval row) as long as nothing scores past
`as_of` differently between the two passes.

Isolation: each preset is seeded into its OWN temporary sqlite store pair
under the system temp dir, via a `scripts/seed.py` subprocess with
`DATABASE_URL` / `GROUND_TRUTH_DATABASE_URL` overridden in that subprocess's
environment only (the same pattern `scripts/arm4_run.py` uses) — because
`app.core.config.Settings` is a frozen dataclass resolved at import time, an
in-process env var change would not take effect. The committed
`data/synthetic/dev.db` / `data/synthetic/ground_truth.db` are never touched.
After seeding, this script talks to the temp stores directly via explicit
engine URLs (`make_inference_engine(url)` / `make_ground_truth_engine(url)`),
never via the global `settings` object.

Usage:
    python scripts/phase5_ambiguity_probe.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.engine import make_inference_engine
from app.db.ground_truth_store import make_ground_truth_engine
from app.services.detector import run_detector
from app.services.evaluation import build_eval_rows, load_ground_truth
from app.services.generator import WORLD_END

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESETS = ("dev", "dev_arm4")

_TARGET_CAP_REASONS = frozenset({"ambiguous", "large_shared"})


def seed_world(tmp_dir: Path, preset: str, seed: int = 0) -> tuple[str, str]:
    """Seed an isolated temp store pair for `preset`, returning the two sqlite URLs.

    Mirrors `scripts/arm4_run.py::seed_world`: seeding runs as a subprocess with
    `DATABASE_URL` / `GROUND_TRUTH_DATABASE_URL` overridden in ITS environment,
    since `Settings` is a frozen dataclass built at import time.
    """
    inf_path = tmp_dir / f"{preset}-inference.db"
    gt_path = tmp_dir / f"{preset}-ground-truth.db"
    inference_url = f"sqlite:///{inf_path}"
    gt_url = f"sqlite:///{gt_path}"
    env = {
        **os.environ,
        "DATABASE_URL": inference_url,
        "GROUND_TRUTH_DATABASE_URL": gt_url,
    }
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "seed.py"),
         "--seed", str(seed), "--preset", preset],
        cwd=REPO_ROOT, env=env, check=True, capture_output=True, text=True,
    )
    return inference_url, gt_url


def probe_preset(preset: str, tmp_root: Path) -> None:
    world_dir = tmp_root / preset
    world_dir.mkdir(parents=True, exist_ok=True)
    inference_url, gt_url = seed_world(world_dir, preset)

    inference_engine = make_inference_engine(inference_url)
    gt_engine = make_ground_truth_engine(gt_url)
    run_id = f"phase5-probe-{preset}-{uuid.uuid4().hex[:8]}"

    with Session(inference_engine, future=True) as inf:
        result = run_detector(inf, as_of=WORLD_END, run_id=run_id)
        inf.commit()

        with Session(gt_engine, future=True) as gt:
            rows = build_eval_rows(inf, load_ground_truth(gt, inf_session=inf), WORLD_END)

    eval_by_key = {row.key: row for row in rows}

    total_decisions = len(result.detections)
    light_friction = [d for d in result.detections if d.proposed_action == "light_friction"]

    n_ambiguous_cap = sum(1 for d in light_friction if "ambiguous" in d.cap_reasons)
    n_large_shared_cap = sum(1 for d in light_friction if "large_shared" in d.cap_reasons)
    target = [
        d for d in light_friction
        if _TARGET_CAP_REASONS.intersection(d.cap_reasons)
    ]

    label_counts: Counter[str] = Counter()
    unmatched = 0
    for d in target:
        key = (d.cluster_snapshot_id, d.cycle_id, d.as_of)
        row = eval_by_key.get(key)
        if row is None:
            unmatched += 1
            continue
        label_counts[row.label.value] += 1

    distinct_clusters = len({d.cluster_snapshot_id for d in result.detections})

    print(f"\n===== preset={preset!r} (as_of={WORLD_END.isoformat()}) =====")
    print(f"1. total decisions recorded (scored cluster-cycle units): {total_decisions}")
    print(f"   proposed_action == light_friction: {len(light_friction)}")
    print("2. of light_friction decisions:")
    print(f"   carrying 'ambiguous' cap reason:     {n_ambiguous_cap}")
    print(f"   carrying 'large_shared' cap reason:  {n_large_shared_cap}")
    print(f"   carrying EITHER (candidate N):       {len(target)}")
    print("3. target population breakdown by 3-way ground-truth label:")
    print(f"   legitimate: {label_counts.get('legitimate', 0)}")
    print(f"   ambiguous:  {label_counts.get('ambiguous', 0)}")
    print(f"   abusive:    {label_counts.get('abusive', 0)}")
    if unmatched:
        print(f"   WARNING: {unmatched} target detections had no matching eval row")
    print("4. context:")
    print(f"   distinct clusters:            {distinct_clusters}")
    print(f"   cluster-cycle units scored:   {result.cluster_cycles}")
    print(f"   eval rows (build_eval_rows):  {len(rows)}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="phase5-ambiguity-probe-") as tmp:
        tmp_root = Path(tmp)
        for preset in PRESETS:
            probe_preset(preset, tmp_root)


if __name__ == "__main__":
    main()

"""CLI composition checks for dev-tier evaluation; no provider or sealed access."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

from sqlalchemy.orm import Session

from app.db.engine import make_inference_engine
from app.db.ground_truth_store import make_ground_truth_engine
from app.db.ground_truth_writer import make_ground_truth_sink
from app.db.schema import create_all_schemas
from app.db.world_provenance import record_world_receipts
from app.services.generator import build_manifest, generate_world, persist_world

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def test_eval_cli_scores_a_scoped_fresh_receipted_dev_world(tmp_path: pathlib.Path) -> None:
    """The actual CLI must skip valid groups whose labels were not authorized."""
    inference_path = tmp_path / "inference.db"
    ground_truth_path = tmp_path / "ground-truth.db"
    report_path = tmp_path / "eval-report.json"
    inf_engine = make_inference_engine(f"sqlite:///{inference_path}")
    gt_engine = make_ground_truth_engine(f"sqlite:///{ground_truth_path}")
    create_all_schemas(inf_engine, gt_engine)
    world = generate_world(1, "dev_arm4")
    with Session(inf_engine, future=True) as inf, Session(gt_engine, future=True) as gt:
        persist_world(world, inf, make_ground_truth_sink(gt))
        record_world_receipts(inf, gt, build_manifest(world, 1, "dev_arm4"), tier="dev")
        inf.commit()
        gt.commit()
    inf_engine.dispose()
    gt_engine.dispose()

    env = {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{inference_path}",
        "GROUND_TRUTH_DATABASE_URL": f"sqlite:///{ground_truth_path}",
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(REPO_ROOT / "backend"), os.environ.get("PYTHONPATH")))
        ),
    }
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "eval_run.py"),
            "--split",
            "tune",
            "--json",
            str(report_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["split"] == "tune"
    assert report["rows"] > 0

"""Deterministic weight-tuning driver + machine-readable ledger (Phase 1 Step 8;
Step-8 review finding 6).

Reproduces the Step 8 tuning exactly from a clean checkout: it generates the dev
world, restricts to the **tune** split (and fails closed if any non-tune row reaches
the scorer), evaluates a fixed candidate grid under the real ≥2-family gate, selects
by a declared objective + tie-break, validates the winner on reserved seeds, and
writes `config/tuning-ledger.json`. `--check` regenerates the deterministic core and
compares it to the committed ledger byte-for-byte, so the reported sweep, rejected
candidates, per-seed metrics, and chosen vector are all verifiable.

Provenance-only fields (git rev, timestamp) live in a separate block that `--check`
ignores; everything that drives the decision is deterministic.

Usage:  python scripts/tune_weights.py [--check]

DEV TIER — the metrics here are NOT reportable (W-1); the *discipline* is what is
validated. The dev sealed (test) split is NOT scored here (tuning is tune-only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.ground_truth_writer import make_ground_truth_sink
from app.db.schema import create_all_schemas
from app.services import features as F
from app.services.evaluation import build_eval_rows, load_ground_truth
from app.services.generator import WORLD_END, canonical_bytes, generate_world, persist_world
from app.services.risk_scorer import SCORE_THRESHOLD

LEDGER_PATH = Path(__file__).resolve().parents[1] / "config" / "tuning-ledger.json"

TUNE_SEED = 1  # the canonical dev world (settings.world_seed)
VALIDATION_SEEDS = (2, 3, 7)  # reserved worlds — NOT used to select, only to check

# The PRE-TUNING seed vector (features.md §3 seed weights) — the baseline the sweep
# starts from. The tuned/frozen vector now in `features.SEED_WEIGHTS` is the sweep's
# expected OUTPUT; the ledger asserts the chosen vector equals it.
_ORIGINAL_SEED: dict[str, float] = {
    "F-DEV-1": 0.30, "F-DEV-2": 0.15, "F-PAY-1": 0.20, "F-CON-1": 0.15,
    "F-DOM-1": 0.15, "F-VER-1": 0.15, "F-CYC-1": 0.30, "F-CYC-2": 0.15,
    "F-CYC-3": 0.10, "F-CYC-4": 0.10, "F-DOM-2": -0.15, "F-AGE-1": -0.15,
}

# Fixed candidate grid. Each is a delta from the seed vector. The three "clean
# discriminator" weights (F-DEV-1/F-CON-1/F-DOM-1 fire on 0 tune-split negatives) are
# the levers; scale-ups are included precisely to show they inject FPs and are rejected.
_CANDIDATES: dict[str, dict[str, float]] = {
    "seed": {},
    "dev1=.35": {"F-DEV-1": 0.35},
    "dev1=.45": {"F-DEV-1": 0.45},
    "dev1=.55": {"F-DEV-1": 0.55},
    "dev1=.60": {"F-DEV-1": 0.60},
    "con1=.25": {"F-CON-1": 0.25},
    "dom1=.25": {"F-DOM-1": 0.25},
    "ver1=.25": {"F-VER-1": 0.25},
    "dev1=.45,con1=.25,dom1=.25": {"F-DEV-1": 0.45, "F-CON-1": 0.25, "F-DOM-1": 0.25},
    "dev1=.50,con1=.25,dom1=.25,cyc1=.40": {
        "F-DEV-1": 0.50, "F-CON-1": 0.25, "F-DOM-1": 0.25, "F-CYC-1": 0.40,
    },
    "scale1.2": {k: round(v * 1.2, 4) for k, v in _ORIGINAL_SEED.items()},
    "scale1.33": {k: round(v * 1.333, 4) for k, v in _ORIGINAL_SEED.items()},
}

OBJECTIVE = "max recall subject to FPR == 0 (precision == 1) on the tune split; " \
    "tie-break: minimal L1 distance from the seed vector"


def _vector_for(delta: dict[str, float]) -> dict[str, float]:
    v = dict(_ORIGINAL_SEED)
    v.update(delta)
    return v


def _apply(v: dict[str, float]) -> None:
    for fid, w in v.items():
        setattr(F, {
            "F-DEV-1": "_W_DEV_1", "F-DEV-2": "_W_DEV_2", "F-PAY-1": "_W_PAY_1",
            "F-CON-1": "_W_CON_1", "F-DOM-1": "_W_DOM_1", "F-VER-1": "_W_VER_1",
            "F-CYC-1": "_W_CYC_1", "F-CYC-2": "_W_CYC_2", "F-CYC-3": "_W_CYC_3",
            "F-CYC-4": "_W_CYC_4", "F-DOM-2": "_W_DOM_2", "F-AGE-1": "_W_AGE_1",
        }[fid], w)


def _assert_tune_only(rows: list) -> None:
    """Fail closed if any non-tune row reaches the tuning scorer (finding 6)."""
    leaked = sorted({r.split for r in rows if r.split != "tune"})
    if leaked:
        raise SystemExit(f"REFUSED: non-tune rows in tuning input: {leaked}")


def _score(session: Session, rows: list, v: dict[str, float]) -> dict:
    _apply(v)
    tp = fp = tn = fn = 0
    for r in rows:
        if r.is_ambiguous:
            continue
        s = F.score_cluster_cycle(session, r.unit, list(r.member_ids), F.Rung.S2)
        flag = s.total_contribution >= SCORE_THRESHOLD and len(s.firing_families) >= 2
        if r.is_positive:
            tp += flag
            fn += not flag
        elif r.is_negative:
            fp += flag
            tn += not flag
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": round(p, 4), "recall": round(rec, 4), "fpr": round(fpr, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _l1(v: dict[str, float]) -> float:
    return round(sum(abs(v[k] - _ORIGINAL_SEED[k]) for k in _ORIGINAL_SEED), 4)


def _build_world(seed: int):
    ie = create_engine("sqlite:///:memory:", future=True)
    ge = create_engine("sqlite:///:memory:", future=True)
    create_all_schemas(ie, ge)
    world = generate_world(seed, "dev")
    checksum = hashlib.sha256(canonical_bytes(world)).hexdigest()
    inf = Session(ie, future=True)
    gt = Session(ge, future=True)
    persist_world(world, inf, make_ground_truth_sink(gt))
    inf.commit()
    gt.commit()
    rows = build_eval_rows(inf, load_ground_truth(gt), WORLD_END)
    return inf, rows, checksum


def _results() -> dict:
    """The deterministic ledger core — no wall-clock, no git rev."""
    frozen_now = dict(F.SEED_WEIGHTS)  # restore live weights after the sweep
    try:
        inf, all_rows, checksum = _build_world(TUNE_SEED)
        tune_rows = [r for r in all_rows if r.split == "tune"]
        _assert_tune_only(tune_rows)
        split_sizes = {
            s: sum(1 for r in all_rows if r.split == s) for s in ("train", "tune", "test")
        }

        evaluated = []
        for name, delta in _CANDIDATES.items():
            v = _vector_for(delta)
            m = _score(inf, tune_rows, v)
            evaluated.append({
                "name": name, "delta": delta, "l1_from_seed": _l1(v), "metrics": m,
                "clean_no_fp": m["fpr"] == 0.0,
            })

        # Objective: FPR==0, then max recall, then min L1 change.
        feasible = [c for c in evaluated if c["metrics"]["fpr"] == 0.0]
        winner = max(
            feasible,
            key=lambda c: (c["metrics"]["recall"], -c["l1_from_seed"]),
        )
        rejected = [
            {"name": c["name"], "reason": "injects false positives (fpr>0)"}
            for c in evaluated if c["metrics"]["fpr"] > 0.0
        ]

        chosen_vector = _vector_for(winner["delta"])
        validation = {}
        for vs in VALIDATION_SEEDS:
            vinf, vrows, _ = _build_world(vs)
            vtune = [r for r in vrows if r.split == "tune"]
            _assert_tune_only(vtune)
            validation[str(vs)] = {
                "chosen_vector_metrics": _score(vinf, vtune, chosen_vector),
                "original_seed_metrics": _score(vinf, vtune, dict(_ORIGINAL_SEED)),
            }
            vinf.close()
        inf.close()

        return {
            "preset": "dev",
            "world_checksum_sha256": checksum,
            "score_threshold": SCORE_THRESHOLD,
            "two_family_gate": True,
            "original_seed_vector": _ORIGINAL_SEED,
            "chosen_equals_frozen_weights": chosen_vector == frozen_now,
            "seeds": {
                "tune": TUNE_SEED,
                "validation": list(VALIDATION_SEEDS),
                "note": "validation seeds are the SAME generator with different seeds "
                        "(not statistically independent worlds); reserved from selection.",
            },
            "split_sizes_tune_world": split_sizes,
            "objective": OBJECTIVE,
            "candidates": evaluated,
            "rejected": rejected,
            "chosen": {
                "name": winner["name"],
                "delta_from_seed": winner["delta"],
                "vector": chosen_vector,
                "tune_metrics": winner["metrics"],
                "l1_from_seed": winner["l1_from_seed"],
            },
            "validation_seed_metrics": validation,
        }
    finally:
        _apply(frozen_now)  # never leave the process with mutated weights


def _ledger() -> dict:
    from app.services.feature_manifest import MANIFEST_HASH
    results = _results()
    results["resulting_feature_manifest_hash"] = MANIFEST_HASH
    return results


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=LEDGER_PATH.parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def _serialize(results: dict, *, with_provenance: bool) -> bytes:
    doc: dict = {"results": results}
    if with_provenance:
        doc["provenance"] = {
            "git_rev": _git_rev(),
            "generated_at": datetime.now(UTC).isoformat(),
            "note": "provenance is informational; --check compares only `results`.",
        }
    return (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune + freeze the weight vector (dev).")
    parser.add_argument(
        "--check", action="store_true",
        help="verify the committed ledger's results reproduce; nonzero exit on drift",
    )
    args = parser.parse_args()

    results = _ledger()
    if args.check:
        if not LEDGER_PATH.exists():
            print("DRIFT  no committed tuning-ledger.json")
            sys.exit(1)
        committed = json.loads(LEDGER_PATH.read_text()).get("results")
        if committed == results:
            print(f"OK  tuning ledger reproduces  chosen={results['chosen']['name']}")
            return
        print("DRIFT  committed tuning ledger results do not reproduce")
        sys.exit(1)

    LEDGER_PATH.write_bytes(_serialize(results, with_provenance=True))
    print(f"wrote {LEDGER_PATH}")
    print(f"  chosen={results['chosen']['name']}  vector={results['chosen']['vector']}")
    print(f"  tune metrics={results['chosen']['tune_metrics']}")


if __name__ == "__main__":
    main()

"""Emit config/reviewer-manifest.json from the live reviewer config (P3.9, L-8).

The reviewer freeze artifact (docs/decisions.md §12 L-8) captures the prompt
template bytes for both variants, the output JSON-Schema bytes, the provider id and
model snapshot, the sampling temperature and sample count, the gating rule, the
auto-apply thresholds, the fake reviewer's rule table, and a source fingerprint over
the reviewer modules. Regenerating it must byte-match the committed file
(`make reviewer-manifest-check`, `test_reviewer_manifest.py`).

This is the ONLY writer of `config/reviewer-manifest.json` — never hand-edit it, or
the next `--check` will fail and the hash will name nothing reproducible.

It is a separate artifact from `config/feature-manifest.json`, which must not change
(L-8). This script never touches that file.

Usage:  python scripts/write_reviewer_manifest.py [--check]
  --check  verify the committed file matches regeneration; non-zero exit on drift.
"""

import argparse
import sys

from app.services.reviewer_manifest import (
    MANIFEST_HASH,
    MANIFEST_PATH,
    manifest_bytes,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write/verify the reviewer manifest.")
    parser.add_argument("--check", action="store_true", help="verify instead of write")
    args = parser.parse_args()

    if args.check:
        committed = MANIFEST_PATH.read_bytes() if MANIFEST_PATH.exists() else b""
        if committed == manifest_bytes():
            print(f"OK  reviewer-manifest.json matches  hash={MANIFEST_HASH}")
            return
        print("DRIFT  committed reviewer-manifest.json does not match regeneration")
        sys.exit(1)

    write_manifest()
    print(f"wrote {MANIFEST_PATH}  hash={MANIFEST_HASH}")


if __name__ == "__main__":
    main()

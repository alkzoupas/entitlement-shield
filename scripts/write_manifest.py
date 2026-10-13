"""Emit config/feature-manifest.json from the live detector config (Phase 1 Step 8).

The freeze artifact (features.md §6.2) captures every §6.1 firing-affecting field
per feature plus a content hash. Regenerating it must byte-match the committed file
(the Step 8 gate, `test_feature_manifest.py`); this script is how it is (re)written
after a deliberate, tagged config change.

Usage:  python scripts/write_manifest.py [--check]
  --check  verify the committed file matches regeneration; non-zero exit on drift.
"""

import argparse
import sys

from app.services.feature_manifest import (
    MANIFEST_HASH,
    MANIFEST_PATH,
    manifest_bytes,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write/verify the feature manifest.")
    parser.add_argument("--check", action="store_true", help="verify instead of write")
    args = parser.parse_args()

    if args.check:
        committed = MANIFEST_PATH.read_bytes() if MANIFEST_PATH.exists() else b""
        if committed == manifest_bytes():
            print(f"OK  feature-manifest.json matches  hash={MANIFEST_HASH}")
            return
        print("DRIFT  committed feature-manifest.json does not match regeneration")
        sys.exit(1)

    write_manifest()
    print(f"wrote {MANIFEST_PATH}  hash={MANIFEST_HASH}")


if __name__ == "__main__":
    main()

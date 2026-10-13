"""Create the inference and ground-truth stores at the configured URLs.

Usage:  python scripts/init_db.py
The two stores are created separately (review 4 P1-4a). Idempotent — create_all
skips existing tables.
"""

from app.core.config import settings
from app.db.schema import create_all_schemas


def main() -> None:
    create_all_schemas()
    print(f"inference store:    {settings.database_url}")
    print(f"ground-truth store: {settings.ground_truth_database_url}")
    print("schemas created (or already present).")


if __name__ == "__main__":
    main()

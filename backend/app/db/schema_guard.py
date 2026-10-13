"""Physical-plane guards shared by independent schema creators."""
from sqlalchemy import Engine, inspect


class MixedStoreError(ValueError):
    """An inference and ground-truth schema marker occupy one physical store."""


def reject_opposite_plane(engine: Engine, *, plane: str, opposite_tables: set[str]) -> None:
    present = sorted(opposite_tables & set(inspect(engine).get_table_names()))
    if present:
        raise MixedStoreError(
            f"refusing to create {plane} schema in a store containing opposite-plane "
            f"tables: {present}"
        )

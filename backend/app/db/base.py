"""ORM declarative bases.

Two SEPARATE bases with separate MetaData, so inference tables and ground-truth
tables cannot be created in the same schema by accident and a session bound to one
store cannot see the other's tables (review 4 P1-4a). `InferenceBase` lives here;
`GroundTruthBase` lives in `app.db.ground_truth_store`, deliberately isolated with
the rest of the ground-truth capability.
"""

from sqlalchemy.orm import DeclarativeBase


class InferenceBase(DeclarativeBase):
    """Base for every table detector services may read: accounts, events, linkage,
    cycles, edges, snapshots, decisions, reviews, audit."""

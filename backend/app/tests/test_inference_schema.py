"""The no-direct-label schema gate (review 4 P1-4d).

The abuse *pattern* in the dev world is intentionally statistically derivable —
otherwise detection would be impossible. What must be absent from the inference
schema is any DIRECT label / scenario-family / split-group column that would let a
detector service read the answer instead of inferring it. This is the testable
gate that replaces the untestable "ground truth not derivable".
"""

import app.models.inference  # noqa: F401 — registers tables on InferenceBase.metadata
from app.db.base import InferenceBase

# Substrings that would betray a sealed field leaking into the inference plane.
FORBIDDEN_SUBSTRINGS = (
    "label",
    "scenario",
    "family",
    "split_group",
    "ground_truth",
    "abusive",
    "legitimate",
    "is_abuse",
)


def test_no_label_scenario_or_group_column_in_inference() -> None:
    offenders: list[str] = []
    for table in InferenceBase.metadata.tables.values():
        for column in table.columns:
            name = column.name.lower()
            if any(bad in name for bad in FORBIDDEN_SUBSTRINGS):
                offenders.append(f"{table.name}.{column.name}")
    assert not offenders, f"inference schema exposes sealed-looking columns: {offenders}"

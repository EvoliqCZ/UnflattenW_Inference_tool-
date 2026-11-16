"""Prediction IR -> `diff.compare` candidate-shape adapter (RFC003-ERRATA-003
E3, `docs/plans/reasoning-data-readiness-2026-09-28.md` §2 row d).

`diff.compare(policy, gold_typed=..., candidate_typed=...)` expects both
sides in the SAME shape: `{node_id: {"node_type": ..., "fields": [{
"node_property_pointer": ..., "known_mask": ..., "observable_mask": ...,
[...]}]}}` (see `dataset/reasoning/diff.py`'s `_gold_field_masks`, and the
S2 typed-target cache this shape already comes from). A sealed Prediction
IR's `typed_predictions` is a flat list of per-field records
(`prediction_ir.typed_prediction_projection`: `local_node_id`, `node_type`,
`pointer`, `head_id`, `value`) -- this module groups that flat list back
into the same per-node shape, adding `value` (the candidate's own claim,
read only as a `DiffPolicy` comparison operand per E3 rule 1) but no masks
of its own: a candidate has no known/observable masks, it IS the thing
being compared against gold's masks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def candidate_typed_from_prediction_ir(
    typed_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Group a sealed Prediction IR fragment's flat `typed_predictions`
    (each a `prediction_ir.typed_prediction_projection` record) by
    `local_node_id` into the `diff.compare`-ready `candidate_typed` shape.

    A `local_node_id` reused across predictions accumulates every one of
    its fields in first-seen order (deterministic given a deterministic
    `typed_predictions` order, which `prediction_ir_projection` already
    guarantees by construction).
    """
    grouped: dict[str, dict[str, Any]] = {}
    for prediction in typed_predictions:
        node_id = prediction["local_node_id"]
        entry = grouped.setdefault(
            node_id, {"node_type": prediction["node_type"], "fields": []}
        )
        entry["fields"].append(
            {
                "node_property_pointer": prediction["pointer"],
                "head_id": prediction["head_id"],
                "value": prediction["value"],
            }
        )
    return grouped

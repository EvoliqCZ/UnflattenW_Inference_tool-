"""RFC004 §21.2/§21.7 -- closed `AlignmentHypothesis` contract (DEC-047:
dormant, opt-in, additive).

`alignment_hypothesis.schema.json` covers ONE entry of a FragmentHypothesis's
`alignments` array: a closed oneOf over `NodeAlignment` (§21.2, now
genuinely multi-region), `TextRangeAlignment` and `LeafAlignment` (§21.7).
Nothing in `node_visual_aligner.py` changes -- it still predicts one region
per node (`region_decode.RegionHypothesis`); `node_alignment_from_region`
below is a pure adapter from that existing shape into one valid
single-region `NodeAlignment` hypothesis, exercising the new schema without
touching the aligner's own contract.

`TextRangeAlignment`/`LeafAlignment` are schema-only here: nothing in this
repo produces one yet (§21.7's fragment-ownership machinery is unbuilt),
so this module validates their shape but has no adapter for either.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema.protocols
import jsonschema.validators

from unflatten.model.region_decode import RegionHypothesis

_SCHEMA_DIR = Path(__file__).parent / "schemas"
_SCHEMA_NAME = "alignment_hypothesis"

ALIGNMENT_HYPOTHESIS_SCHEMA_VERSION = "0.1.0"


class AlignmentHypothesisInvalid(ValueError):
    """An AlignmentHypothesis entry fails its closed schema (RFC004 §21.7)."""


@cache
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / f"{_SCHEMA_NAME}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def validate_alignment_hypothesis(doc: Mapping[str, Any]) -> None:
    """Validate one `alignments` array entry against the closed oneOf."""
    errors = list(_validator().iter_errors(doc))
    if errors:
        raise AlignmentHypothesisInvalid(
            f"schema validation failed: {errors[0].message} at {list(errors[0].absolute_path)}"
        )


def node_alignment_from_region(prediction: RegionHypothesis) -> dict[str, Any] | None:
    """Adapt `node_visual_aligner`'s existing single-region prediction into
    one valid `NodeAlignment` hypothesis (§21.2).

    Returns `None` for an abstained prediction (§21.5: nothing visible to
    align, so no `alignments` entry -- never a degenerate zero-region
    `NodeAlignment`). The aligner itself predicts at most one region per
    node today; a real multi-region producer would append more region dicts
    to the same `regions` list this returns, which is exactly what
    `alignment_hypothesis.schema.json`'s `minItems: 1` (no `maxItems`) now
    allows that `model_prediction_ir.schema.json`'s existing single-region
    `region_prediction` def (`maxItems: 1`) does not -- this adapter does
    not change that existing schema.
    """
    if prediction.alignment_status != "region":
        return None
    doc = {
        "alignment_hypothesis_schema_version": ALIGNMENT_HYPOTHESIS_SCHEMA_VERSION,
        "kind": "node",
        "local_node_id": prediction.local_node_id,
        "regions": [_region_dict(region) for region in prediction.regions],
    }
    validate_alignment_hypothesis(doc)
    return doc


def _region_dict(region: Any) -> dict[str, Any]:
    doc = {
        "page_ordinal": region.page_ordinal,
        "bbox_ppm": list(region.bbox_ppm),
        "continuation": region.continuation,
        "confidence_ppm": region.confidence_ppm,
    }
    if region.polygon_ppm is not None:
        doc["polygon_ppm"] = [list(point) for point in region.polygon_ppm]
    return doc

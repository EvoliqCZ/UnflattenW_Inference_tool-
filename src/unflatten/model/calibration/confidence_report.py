"""RFC004 §35/§36 confidence-report builder.

`build_confidence_report` produces a payload conforming to
`confidence-report.schema.json` and nothing else -- in particular it
never touches a `ResultAxes` triple or an AST status. The schema's own
`$comment` states confidence is "diagnostics only, never a status
field"; this module's contract is that no function here accepts or
returns a `ResultAxes`/AST-status value at all, so there is nothing to
mutate by construction (the same shape of guarantee
`unflatten.model.result_axes` gives for confidence never changing a
status).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema.protocols
import jsonschema.validators

CONFIDENCE_REPORT_SCHEMA_VERSION = "0.1.0"
_SCHEMA_DIR = Path(__file__).parent.parent / "schemas"
_AXES = ("ast", "renderer", "typed_property", "alignment")


class ConfidenceReportInvalid(ValueError):
    """A built confidence-report payload fails `confidence-report.schema.json`."""


def build_confidence_report(
    calibrator_id: str,
    overall_score: int,
    ood_flag: bool,
    per_axis: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build a schema-valid confidence report.

    `overall_score` and every `per_axis` value are `0..1000` integers
    (the schema's scale, not a `[0, 1]` float or a ppm value). Raises
    :class:`ConfidenceReportInvalid` if the result does not validate.
    """
    payload = {
        "confidence_report_schema_version": CONFIDENCE_REPORT_SCHEMA_VERSION,
        "calibrator_id": calibrator_id,
        "overall_score": overall_score,
        "ood_flag": ood_flag,
        "per_axis": dict(per_axis) if per_axis else {},
    }
    _validate(payload)
    return payload


@cache
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / "confidence-report.schema.json", encoding="utf-8") as handle:
        schema = json.load(handle)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _validate(payload: Mapping[str, Any]) -> None:
    errors = list(_validator().iter_errors(payload))
    if errors:
        raise ConfidenceReportInvalid(
            f"schema validation failed: {errors[0].message} at {list(errors[0].absolute_path)}"
        )

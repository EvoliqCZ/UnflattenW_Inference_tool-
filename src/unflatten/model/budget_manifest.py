"""RFC004 §45.2 budget manifest, §45.3/§45.4 hardware profiles, §45.6 OOM
policy and §12.4 overflow policy -- schema, loader and preflight.

`max_images` additionally covers §44.6's "image count" complexity limit
(§44.6: "Max depth, node count, table size, image count and decoder steps
chránia pred adversarial resource exhaustion. Limit failure nesmie viesť
k truncated success.") alongside `max_nodes`/`max_depth`/`max_tables`/
`max_cells_per_table` already required by §45.2 -- one manifest pins both
the performance budget and the adversarial complexity ceiling, since both
are "reject, never truncate" limits over the same request shape.

Mirrors `runtime_limits.py`'s discipline: this module builds/validates a
*policy artefact* and checks a request against it; it does **not** pin the
concrete owner-frozen numeric caps for the H200 reference profile
(§45.3) or the RTX 4070 dev-smoke profile (§45.4) -- those are deployment
decisions, not RFC004-invented thresholds. What this module fixes is the
*shape*: `hardware` is a closed `h200 | rtx4070` enum (§45.3/§45.4), every
profile pins every §45.2 limit, exactly one §12.4 `overflow_policy`, and
an `oom_escalation` list drawn only from the §45.6 allowed OOM behaviors
(the four forbidden ones -- drop page, drop tile, truncate-and-mark-
success, silently disable a required head -- are not members of that
enum, so they cannot even be declared).

`runtime_limits_policy_hash` lets a budget manifest reference the
existing S1-scoped `runtime_limits_policy` (`runtime_limits.py`) by hash
without this module reading or modifying that hash-bound artefact.

`preflight_check` REJECTS a request that exceeds any pinned limit --
never truncates, drops, or silently substitutes a smaller value (§45.6,
§12.4: "unsupported extreme input -> `TOKEN_BUDGET_EXCEEDED`"). Visual
token budget fields use the RFC004 §42.4 `TOKEN_BUDGET_EXCEEDED` code
(§12.4 names this budget specifically); every other limit uses
`MODEL_INPUT_LIMIT_EXCEEDED` (§42.4 Input/preprocess).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema.protocols
import jsonschema.validators

from unflatten.ast.issues import Issue

_SCHEMA_DIR = Path(__file__).parent / "schemas"
_SCHEMA_NAME = "budget_manifest"

BUDGET_MANIFEST_SCHEMA_VERSION = "0.1.0"

TOKEN_BUDGET_EXCEEDED = "TOKEN_BUDGET_EXCEEDED"
MODEL_INPUT_LIMIT_EXCEEDED = "MODEL_INPUT_LIMIT_EXCEEDED"

#: §45.2 limit fields that gate on visual-token budget overflow (§12.4) --
#: everything else that gates on a limit uses MODEL_INPUT_LIMIT_EXCEEDED.
_TOKEN_BUDGET_FIELDS: frozenset[str] = frozenset(
    {
        "max_visual_tokens_per_page",
        "max_visual_tokens_per_window",
        "max_visual_tokens_per_request",
    }
)

#: Every §45.2 limit a preflight request may report an observed value for,
#: using the manifest's own field names (module docstring: same key names
#: on both sides makes the comparison unambiguous).
LIMIT_FIELDS: tuple[str, ...] = (
    "max_pages_per_request",
    "max_pixels_per_page",
    "max_pixels_per_request",
    "max_visual_tokens_per_page",
    "max_visual_tokens_per_window",
    "max_visual_tokens_per_request",
    "max_decoder_context_tokens",
    "max_decoder_output_tokens",
    "max_nodes",
    "max_depth",
    "max_tables",
    "max_cells_per_table",
    "max_images",
    "max_resource_bytes",
)


class BudgetManifestInvalid(ValueError):
    """A budget manifest is malformed or fails its closed schema."""


@cache
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / f"{_SCHEMA_NAME}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def validate_budget_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate `manifest` against `budget_manifest.schema.json`. Raises
    `BudgetManifestInvalid` on the first schema error."""
    errors = list(_validator().iter_errors(manifest))
    if errors:
        raise BudgetManifestInvalid(
            f"schema validation failed: {errors[0].message} at "
            f"{list(errors[0].absolute_path)}"
        )


def load_budget_manifest(path: Path | str) -> dict[str, Any]:
    """Read a frozen budget manifest from disk, validated on the way in."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise BudgetManifestInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BudgetManifestInvalid(f"{path} is not valid JSON: {exc}") from exc
    validate_budget_manifest(payload)
    return payload


def preflight_check(
    manifest: Mapping[str, Any], observed: Mapping[str, int]
) -> list[Issue]:
    """§45.6/§12.4 preflight: REJECT (never truncate) an `observed` request
    that exceeds any limit `manifest` pins.

    `observed` carries a subset of :data:`LIMIT_FIELDS`; an unknown key is
    refused the same way an unknown `runtime_limits.py` limit is refused,
    so a caller who misspells a field cannot silently skip its check.
    Returns one `Issue` per exceeded limit; an empty list means the
    request fits the manifest's budget.
    """
    validate_budget_manifest(manifest)
    unknown = sorted(set(observed) - set(LIMIT_FIELDS))
    if unknown:
        raise BudgetManifestInvalid(
            f"unknown observed field(s): {', '.join(unknown)} -- expected a subset of "
            f"{', '.join(LIMIT_FIELDS)}"
        )

    issues: list[Issue] = []
    for field in LIMIT_FIELDS:
        if field not in observed:
            continue
        limit = manifest[field]
        value = observed[field]
        if value > limit:
            code = (
                TOKEN_BUDGET_EXCEEDED
                if field in _TOKEN_BUDGET_FIELDS
                else MODEL_INPUT_LIMIT_EXCEEDED
            )
            issues.append(
                Issue(
                    code=code,
                    severity="fatal",
                    node_id=None,
                    path=f"/{field}",
                    message=(
                        f"{field} observed {value} exceeds profile "
                        f"{manifest.get('profile_id')!r} limit {limit}."
                    ),
                )
            )
    return issues

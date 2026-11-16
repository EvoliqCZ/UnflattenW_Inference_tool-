"""RFC004 §33.2 bod 8 Builder Report (design §3, T6b of
`spec-ast-builder-t5-promotion` P2).

`ast_builder._build_raw` calls :func:`build_report` at the end of every
build -- strict or failed alike -- with the node counts and issue rows it
collected. This module owns the report's shape, its self-hash and the two
hashes ``default_policy_hash``/``derivation_registry_hash`` it embeds:
importing both here (rather than having `ast_builder.py` import
`default_policy` directly) is deliberate -- design §2.5 D1/R1 forbids the
strict path from reaching a default value, and only an import graph proves
that. The transitive import through this module is the one case plan U3
calls out as intentional.

``report_hash`` is a ``domain_hash`` over every field except itself (domain
`rfc004.builder-report/v0`, module constant `_BUILDER_REPORT_DOMAIN`),
with the five row arrays
(``derivations``/``imputations``/``discarded_inputs``/``unresolved``/
``conflicts``) sorted by ``(source or "", pointer or "", code or "")``
first -- so two builds whose issues were only collected in a different
order still hash equal (plan P2 R6). ``predicted_by_pointer`` and
``node_counts`` are plain dicts; the deterministic CBOR profile
(:mod:`unflatten.model.cbor`) already sorts map keys on encode, so no
explicit sort is needed for them.

Mode is always ``"strict_build"`` (plan U1): the review candidate (plan P4)
never gets its own report -- it carries `builder_report_hash` equal to the
SAME raw report's `report_hash`, because both walk the same `_build_raw`
path and differ only in a post-pass imputation the report does not see.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema.protocols
import jsonschema.validators

from unflatten.model.default_policy import default_policy_hash
from unflatten.model.derivations import derivation_registry_hash
from unflatten.model.hashing import domain_hash

_SCHEMA_DIR = Path(__file__).parent / "schemas"
_SCHEMA_NAME = "builder_report"

#: Design §4.9 registry row `rfc004.builder-report/v0`. Owned here: no
#: other module computes a `report_hash` under this domain.
_BUILDER_REPORT_DOMAIN = "rfc004.builder-report/v0"

BUILDER_REPORT_SCHEMA_VERSION = "0.1.0"

_ROW_LISTS = ("derivations", "imputations", "discarded_inputs", "unresolved", "conflicts")


class BuilderReportInvalid(ValueError):
    """A builder report fails its closed schema."""


@cache
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / f"{_SCHEMA_NAME}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def validate_builder_report(report: Mapping[str, Any]) -> None:
    """Validate `report` against `builder_report.schema.json`."""
    errors = list(_validator().iter_errors(report))
    if errors:
        raise BuilderReportInvalid(
            f"schema validation failed: {errors[0].message} at {list(errors[0].absolute_path)}"
        )


def _sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (row.get("source") or "", row.get("pointer") or "", row.get("code") or "")


def _sorted_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Design §3 sort rule; also what makes the hash order-independent
    (plan P2 R6 -- dropping this `sorted()` is the mutant that guard tests)."""
    return [dict(row) for row in sorted(rows, key=_sort_key)]


def report_projection(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The payload `report_hash` is computed over: `doc` without its own
    `report_hash` key."""
    if "report_hash" not in doc:
        raise BuilderReportInvalid(
            "doc has no 'report_hash' key -- the self-hash slot must exist before projection"
        )
    return {k: v for k, v in doc.items() if k != "report_hash"}


def compute_report_hash(doc: Mapping[str, Any]) -> str:
    return domain_hash(_BUILDER_REPORT_DOMAIN, report_projection(doc))


def build_report(
    *,
    builder_version: str,
    node_counts: Mapping[str, int],
    predicted_by_pointer: Mapping[str, int],
    derivations: Sequence[Mapping[str, Any]] = (),
    imputations: Sequence[Mapping[str, Any]] = (),
    discarded_inputs: Sequence[Mapping[str, Any]] = (),
    unresolved: Sequence[Mapping[str, Any]] = (),
    conflicts: Sequence[Mapping[str, Any]] = (),
    mode: str = "strict_build",
) -> dict[str, Any]:
    """Assemble one Builder Report, hashed and validated on the way out.

    `derivation_registry_hash`/`default_policy_hash` are read from their
    owning modules here, never passed in -- `_build_raw` never imports
    `default_policy` (module docstring, R1).
    """
    property_counts = {
        "predicted": sum(predicted_by_pointer.values()),
        "derived": len(derivations),
        "imputed": len(imputations),
        "unresolved": len(unresolved),
        "discarded": len(discarded_inputs),
    }
    doc: dict[str, Any] = {
        "builder_report_schema_version": BUILDER_REPORT_SCHEMA_VERSION,
        "mode": mode,
        "builder_version": builder_version,
        "derivation_registry_hash": derivation_registry_hash(),
        "default_policy_hash": default_policy_hash(),
        "node_counts": dict(node_counts),
        "property_counts": property_counts,
        "predicted_by_pointer": dict(predicted_by_pointer),
        "derivations": _sorted_rows(derivations),
        "imputations": _sorted_rows(imputations),
        "discarded_inputs": _sorted_rows(discarded_inputs),
        "unresolved": _sorted_rows(unresolved),
        "conflicts": _sorted_rows(conflicts),
        "report_hash": None,
    }
    doc["report_hash"] = compute_report_hash(doc)
    validate_builder_report(doc)
    return doc

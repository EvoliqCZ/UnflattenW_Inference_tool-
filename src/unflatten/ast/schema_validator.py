"""JSON Schema validation pass (RFC001 §9.2 level 1, §49.1)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

from unflatten.ast.issues import Issue

_SCHEMA_PATH = Path(__file__).parent / "schemas" / "canonical_ast.schema.json"


@lru_cache(maxsize=1)
def _load_schema() -> dict:
    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _validator() -> jsonschema.protocols.Validator:
    schema = _load_schema()
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _pointer(path: Any) -> str:
    if not path:
        return "/"
    return "/" + "/".join(str(p) for p in path)


def validate_schema(doc: Any) -> list[Issue]:
    """Validate `doc` against the canonical AST JSON Schema.

    Never raises on invalid input — returns a list of Issue instead.
    An empty list means the document passed schema validation.
    """
    issues: list[Issue] = []
    for error in _validator().iter_errors(doc):
        issues.append(
            Issue(
                code="SCHEMA_VALIDATION_ERROR",
                severity="error",
                node_id=None,
                path=_pointer(list(error.absolute_path)),
                message=error.message,
                details={"validator": error.validator},
            )
        )
    return issues

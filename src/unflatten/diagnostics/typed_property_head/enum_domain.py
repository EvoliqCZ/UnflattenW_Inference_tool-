"""Finding 1 fix -- an enum head's vocabulary is a function of its pointer
alone, taken from the canonical AST JSON Schema, never from observed data.

`run.py` used to build an enum head's classifier from the FIRST batch's
observed values and freeze it there: the same data fed in a different
batch order produced a different `num_classes`, and a legitimate schema
value simply not yet seen raised `UnknownEnumValueError`. The schema is
the one place the domain cannot drift or depend on order -- it is the
same source `dataset.targets.properties.pointer_registry()` derives its
`value_type == "enum"` classification from (see that module's `_walk_schema`,
which this mirrors structurally but records the actual `enum` list rather
than just the `"enum"` type marker).

`unflatten.dataset` is read-only for this fix, so this module does not
edit `properties.py` -- it re-walks the schema itself, using only
`properties.PROPERTY_FIELDS` (the node-type -> top-level-field surface,
read not modified) and `unflatten.ast.schema_validator._load_schema`
(not under `unflatten.dataset` at all).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from unflatten.ast.schema_validator import _load_schema
from unflatten.dataset.targets.properties import PROPERTY_FIELDS

_PropertyKey = tuple[str, str]


class UnknownEnumPointerError(ValueError):
    """`(node_type, pointer)` is not a registered enum pointer in the
    canonical AST schema -- a registry-drift bug, never a case to guess
    a domain for."""


def enum_domain(node_type: str, pointer: str) -> tuple[Any, ...]:
    """The full, schema-defined set of legal values for an enum pointer.

    `pointer` is the node-relative, normalized JSON Pointer (see
    `dataset.targets.properties.normalize_pointer`), e.g.
    `"/layout/alignment"`. Sorted by nothing -- callers that need a
    deterministic order sort the result themselves (by `str`, since enum
    values are always strings).
    """
    registry = _enum_registry()
    node_registry = registry.get(node_type)
    if node_registry is None or pointer not in node_registry:
        raise UnknownEnumPointerError(
            f"{(node_type, pointer)!r} is not a registered enum pointer in the "
            "canonical AST schema"
        )
    return node_registry[pointer]


@lru_cache(maxsize=1)
def _enum_registry() -> dict[str, dict[str, tuple[Any, ...]]]:
    schema = _load_schema()
    defs = schema["$defs"]
    registry: dict[str, dict[str, tuple[Any, ...]]] = {}
    for node_type, fields in PROPERTY_FIELDS.items():
        node_def = defs[node_type]
        pointers_acc: dict[str, set[Any]] = {}
        for field in fields:
            _walk(node_def["properties"][field], f"/{field}", pointers_acc, defs)
        registry[node_type] = {
            pointer: tuple(values) for pointer, values in pointers_acc.items()
        }
    return registry


def _resolve_ref(subschema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    while "$ref" in subschema:
        ref = subschema["$ref"]
        subschema = defs[ref.removeprefix("#/$defs/")]
    return subschema


def _walk(
    subschema: dict[str, Any], pointer: str, out: dict[str, set[Any]], defs: dict[str, Any]
) -> None:
    """Mirrors `properties._walk_schema`'s traversal, but records the
    actual legal values (from `enum` or `const`) rather than just the
    `"enum"` value-type marker. `oneOf` branches are UNIONED into the
    same pointer's domain set -- e.g. `line_spacing.mode` is a `const`
    per branch (`"single"`, `"multiple"`, `"exact"`, `"fixed"`), and the
    pointer's real domain is the union of all four, not whichever branch
    is walked last."""
    subschema = _resolve_ref(subschema, defs)
    if "properties" in subschema:
        for name, prop in subschema["properties"].items():
            _walk(prop, f"{pointer}/{name}", out, defs)
        return
    if "items" in subschema:
        _walk(subschema["items"], f"{pointer}/*", out, defs)
        return
    if "oneOf" in subschema:
        for branch in subschema["oneOf"]:
            branch = _resolve_ref(branch, defs)
            if branch.get("type") == "null":
                continue
            _walk(branch, pointer, out, defs)
        return
    if "enum" in subschema:
        out.setdefault(pointer, set()).update(subschema["enum"])
    elif "const" in subschema:
        out.setdefault(pointer, set()).add(subschema["const"])

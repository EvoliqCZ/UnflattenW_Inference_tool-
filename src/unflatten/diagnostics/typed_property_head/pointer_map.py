"""Design §3.1 — the diagnostic's own, smaller pointer -> family map.

`DIAGNOSTIC_POINTER_TO_FAMILY` is built at import time by crossing
`property_head_registry.head_for_pointer` (kept only for
`head_id == "expert_adapter.dense"` AND `node_type == "paragraph"`, per
design §1.1's slice-1 decision) with
`dataset.targets.properties.pointer_registry()` value types.

Design §1.1's MEASURED 2026-09-06 census of the 35 `paragraph` /
`expert_adapter.dense` pointers is 15 integer, 10 enum, 5 string (the
four `/style/borders/*/color` pointers plus `/style/background`), 4
boolean, 1 object (`/style/borders` itself, a nullable container, not a
scalar). Slice 1 keeps only integer/enum/boolean (29 pointers); string
and object are DEFERRED (color is its own RFC004 §25 three-way design,
and an object is not a head's target at all) and MUST be absent from
this map.

This is a SEPARATE mapping from `property_head_registry.PROPERTY_HEAD_REGISTRY`
-- never a copy or mutation of it (design §3.1) -- and an unrecognised
value type raises rather than silently defaulting, mirroring
`head_for_pointer`'s own no-silent-default contract one layer up.
"""

from __future__ import annotations

from unflatten.dataset.targets import properties
from unflatten.model import property_head_registry as phr

_SLICE_1_FAMILIES = frozenset({"integer", "enum", "boolean"})
_DEFERRED_VALUE_TYPES = frozenset({"string", "object"})


class UnrecognisedValueTypeError(ValueError):
    """`pointer_registry()` returned a value type this map does not know
    how to classify as slice-1 (integer/enum/boolean) or deferred
    (string/object). Raised rather than silently dropped or defaulted --
    an unrecognised value type is a registry-drift bug, not a case to
    route around."""


def _build() -> dict[tuple[str, str], str]:
    projection = phr.registry_projection()
    paragraph_registry = properties.pointer_registry()["paragraph"]

    mapping: dict[tuple[str, str], str] = {}
    for entry in projection["mapped"]:
        if entry["node_type"] != "paragraph" or entry["head_id"] != "expert_adapter.dense":
            continue
        pointer = entry["node_property_pointer"]
        # Re-derive via head_for_pointer rather than trusting the
        # projection alone, so this map raises PropertyHeadRegistryError
        # the same way a construction-time caller would (design §3.1
        # point 1), reproducing the "raise rather than silently default"
        # contract one layer up. A plain `assert` here would be stripped
        # under `python -O` (finding 5), so this is a raise, not an
        # assertion.
        re_derived_head_id = phr.head_for_pointer("paragraph", pointer)
        if re_derived_head_id != "expert_adapter.dense":
            raise phr.PropertyHeadRegistryError(
                f"registry drift: projection says ('paragraph', {pointer!r}) is "
                "head_id='expert_adapter.dense', but head_for_pointer independently "
                f"resolves it to {re_derived_head_id!r}"
            )

        normalized = properties.normalize_pointer(pointer)
        value_type = paragraph_registry[normalized]
        if value_type in _SLICE_1_FAMILIES:
            mapping[("paragraph", pointer)] = value_type
        elif value_type in _DEFERRED_VALUE_TYPES:
            continue
        else:
            raise UnrecognisedValueTypeError(
                f"paragraph pointer {pointer!r} has value_type {value_type!r}, "
                "which is neither a slice-1 family "
                f"({sorted(_SLICE_1_FAMILIES)}) nor a deferred type "
                f"({sorted(_DEFERRED_VALUE_TYPES)})"
            )
    return mapping


DIAGNOSTIC_POINTER_TO_FAMILY: dict[tuple[str, str], str] = _build()

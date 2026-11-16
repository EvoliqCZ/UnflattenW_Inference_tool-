"""RFC004 T7a Default Policy (design §2.5, plan P0): RFC001 §43's V0
mandatory defaults, closed over every required leaf pointer a strict build
can leave unresolved (`ast_required.required_leaf_pointers`, minus the two
pointers the builder supplies itself).

Fail-closed by construction: a required leaf pointer §43 does not give a
literal default for -- `page_break_before`, `highlight`, font family/size,
section geometry, images, borders... -- gets a record with
`default_defined = False` rather than being left out, so a later consumer
asking about it gets a documented refusal instead of a `KeyError` it has to
interpret. `/resource_ref` and `/note_id` are the opposite: they get no
record at all, so asking about them raises `KeyError` (R8) -- they are
never a fact this policy has an opinion on, imputable or not, because the
builder (not a default) is their only legitimate source.

**Not consulted by `build_strict`.** `ast_builder.py` must not import this
module (design §2.5 D1, R1) -- imputation reachable from the strict path
would silently turn "the model did not say" into "the model said the
profile default", which is exactly the confusion §33.6 exists to prevent.
`review_candidate.py` (plan P4) is the only intended consumer.

Two containers a repeated-container element count is derived from --
`paragraph /layout/tabs`, `table /columns` -- get an explicit extra record
here: `required_leaf_pointers` never names an array of *objects* itself,
only its elements (`/layout/tabs/*/leader`), so their own count has no
required-leaf pointer to inherit a record from. `section
/columns/widths_mpt` needs no such addition: its elements are plain
integers, so `required_leaf_pointers` already treats the whole array as one
leaf.

Torch-free. Mints its own hash domain (`rfc004.default-policy/v0`) and
derives no other module's hash.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from unflatten.model.ast_required import required_leaf_pointers
from unflatten.model.hashing import domain_hash
from unflatten.model.typed_property_pointer_map import node_types

DEFAULT_POLICY_VERSION = "0.1.0"

#: Design §4.9 registry row `rfc004.default-policy/v0`. Owned here: no
#: other module computes a `default_policy_hash`.
_DEFAULT_POLICY_DOMAIN = "rfc004.default-policy/v0"

#: Supplied by the builder itself, never by a typed value or a default
#: (mirrors `ast_builder._BUILDER_POINTERS`; R8). Not imported from there
#: -- `ast_builder.py` must not import this module, so the dependency runs
#: only one way and this literal is the price of that.
_BUILDER_POINTERS = frozenset({"/resource_ref", "/note_id"})

#: The two repeated containers whose element COUNT has no required-leaf
#: pointer of its own (module docstring).
_EXTRA_CONTAINER_POINTERS: frozenset[tuple[str, str]] = frozenset({
    ("paragraph", "/layout/tabs"),
    ("table", "/columns"),
})

#: A default value is always one of these three JSON scalar types (RFC001
#: §43 never defaults an object or an array).
DefaultValue = bool | int | str | None

_S43_SOURCE = "RFC001 §43 v0.1-1"

#: RFC001 §43 (`docs/rfc/RFC001-canonical-document-ast-v0.1-1.md:1823-1848`),
#: one entry per pointer the table gives a literal default for -- 14 table
#: rows, expanded to 21 pointers (`design §2.5`). Font family/size are §43
#: rows too, but explicitly `default_defined = False` (§27.1), so they are
#: deliberately absent here rather than listed with a null value.
_S43_DEFAULTS: dict[tuple[str, str], DefaultValue] = {
    ("paragraph", "/layout/alignment"): "left",
    ("paragraph", "/layout/direction"): "ltr",
    ("paragraph", "/layout/indent/left_mpt"): 0,
    ("paragraph", "/layout/indent/right_mpt"): 0,
    ("paragraph", "/layout/indent/first_line_mpt"): 0,
    ("paragraph", "/layout/indent/hanging_mpt"): 0,
    ("paragraph", "/layout/spacing/before_mpt"): 0,
    ("paragraph", "/layout/spacing/after_mpt"): 0,
    ("paragraph", "/layout/spacing/line/mode"): "single",
    ("paragraph", "/layout/pagination/keep_with_next"): False,
    ("paragraph", "/layout/pagination/keep_lines"): False,
    ("paragraph", "/layout/pagination/widow_control"): True,
    ("run", "/style/bold"): False,
    ("run", "/style/italic"): False,
    ("run", "/style/strike"): False,
    ("run", "/style/underline"): "none",
    ("run", "/style/color"): "#000000",
    ("run", "/style/character_spacing_mpt"): 0,
    ("cell", "/layout/vertical_alignment"): "top",
    ("table", "/layout/alignment"): "left",
    ("row", "/layout/height/mode"): "auto",
}


@dataclass(frozen=True, slots=True)
class DefaultRecord:
    node_type: str
    pointer: str
    default_defined: bool
    default_value: DefaultValue
    default_source: str | None
    must_predict: bool
    may_impute: bool


def _build_policy() -> tuple[DefaultRecord, ...]:
    keys = {
        (node_type, pointer)
        for node_type in node_types()
        for pointer in required_leaf_pointers(node_type) - _BUILDER_POINTERS
    } | _EXTRA_CONTAINER_POINTERS
    records = []
    for node_type, pointer in sorted(keys):
        default_value = _S43_DEFAULTS.get((node_type, pointer), None)
        defined = (node_type, pointer) in _S43_DEFAULTS
        records.append(
            DefaultRecord(
                node_type=node_type,
                pointer=pointer,
                default_defined=defined,
                default_value=default_value,
                default_source=_S43_SOURCE if defined else None,
                must_predict=True,
                may_impute=defined,
            )
        )
    return tuple(records)


#: One record per required leaf pointer of every builder node type
#: (`typed_property_pointer_map.node_types()`), minus the builder
#: pointers, plus the two extra container records (module docstring).
DEFAULT_POLICY: tuple[DefaultRecord, ...] = _build_policy()

_POLICY_BY_KEY: dict[tuple[str, str], DefaultRecord] = {
    (record.node_type, record.pointer): record for record in DEFAULT_POLICY
}

#: A path segment that is purely digits, normalized to `*` (mirrors the
#: pointer shape `required_leaf_pointers` already uses for array elements).
_INDEX_SEGMENT = re.compile(r"/\d+(?=/|$)")


def lookup(node_type: str, pointer: str) -> DefaultRecord:
    """The policy record for `pointer` on `node_type`.

    Numeric path segments are normalized to `*` first, so
    `/layout/tabs/0/leader` looks up the same record as
    `/layout/tabs/*/leader`. Raises `KeyError` for a pointer this policy
    has no record for -- including `/resource_ref` and `/note_id`, which
    are never imputable (R8).
    """
    normalized = _INDEX_SEGMENT.sub("/*", pointer)
    try:
        return _POLICY_BY_KEY[(node_type, normalized)]
    except KeyError:
        raise KeyError(
            f"no default policy record for ({node_type!r}, {pointer!r})"
        ) from None


def default_policy_hash() -> str:
    """`domain_hash` over the whole policy table, sorted by
    `(node_type, pointer)` for a build-order-independent digest."""
    projection = {
        "default_policy_version": DEFAULT_POLICY_VERSION,
        "records": [
            {
                "node_type": record.node_type,
                "pointer": record.pointer,
                "default_defined": record.default_defined,
                "default_value": record.default_value,
                "default_source": record.default_source,
                "must_predict": record.must_predict,
                "may_impute": record.may_impute,
            }
            for record in DEFAULT_POLICY
        ],
    }
    return domain_hash(_DEFAULT_POLICY_DOMAIN, projection)

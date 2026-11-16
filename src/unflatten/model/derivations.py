"""RFC004 §27.2 deterministic derivation registry (design §2.3, DEC-037 bod
1, DEC-039; P1 of `spec-ast-builder-t5-promotion`).

`derived_deterministic` is only allowed for a derivation in a closed,
versioned registry (§27.2). This module is that registry: exactly the five
`DerivationRecord`s DEC-037/DEC-039 named, plus `derivation_registry_hash()`.
Wiring these into `ast_builder._build_raw`'s post-pass, and reporting what
ran, is a later task (plan P3) -- this module stays pure over plain dicts
and never imports `ast_builder`.

`/alt_text` is deliberately absent: DEC-037 bod 1 blocks it on the
unratified RFC004 §33.6 erratum (mutant R2 guards this).

**Rule: a change to a record's `fn` must bump that record's `version`.**
`derivation_registry_hash()` projects every field except `precondition` and
`fn` themselves (callables are not serializable and are not what a
Component Lock needs to pin -- it needs to know THAT the algorithm changed,
not read its bytecode). If `fn`'s behaviour changes without a version bump,
the registry hash stays the same while the derived AST values it produces
would not -- a silent divergence between the Component Lock and what
`build_strict` actually emits. Bumping `version` is what makes that
divergence visible (mutant R13).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from unflatten.model.hashing import domain_hash

#: Design §4.9 registry row `rfc004.derivation-registry/v0`. Owned here: no
#: other module computes a `derivation_registry_hash`.
_DERIVATION_REGISTRY_DOMAIN = "rfc004.derivation-registry/v0"

#: A derived value is always a JSON scalar (RFC001 never derives an object
#: or array): a container-length `int`, or a `marker_text` `str`.
TypedValue = int | str

_MISSING = object()


def _get(node: Any, pointer: str) -> Any:
    """`pointer`'s value under `node`, or `_MISSING` if any segment is
    absent. Same-node pointer lookup only -- a pure dict walk, no schema
    knowledge, mirroring `ast_builder._get` without importing it (this
    module must not depend on `ast_builder`, the reverse of R1's rule)."""
    value = node
    for part in pointer.strip("/").split("/"):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return _MISSING
    return value


@dataclass(frozen=True, slots=True)
class DerivationContext:
    """Everything a derivation's `precondition`/`fn` needs, pre-computed by
    the caller so every function in this registry stays pure over dicts --
    no tree walk, no note or page lookup of its own.

    node: the assembled node dict being derived onto (`id` `"b<n>"`,
        `"type"`, and whatever typed pointers the hypothesis already wrote).
    assembled_id: the node's source id (`"a<k>"`), for report/issue rows;
        `None` when the node has no single source piece.
    page_ordinal: the page ordinal of the node's first source piece, or
        `None` when the node carries no source piece at all -- the
        `z_order` precondition.
    page_floating_preorder: the `id`s of every `floating_image`/
        `text_frame` node on `page_ordinal`'s page, in document preorder.
        Empty when `page_ordinal is None`.
    note_kind_and_order: `note_id -> (kind, order)` for every bound note in
        the document, `kind` one of `"footnote"`/`"endnote"`, `order` the
        note's 1-based position among notes of that same kind, in document
        order. Not just this node's note -- the whole binding, so a
        `note_ref`'s own precondition/fn only need `node["note_id"]` to look
        itself up.
    """

    node: dict[str, Any]
    assembled_id: str | None
    page_ordinal: int | None
    page_floating_preorder: tuple[str, ...]
    note_kind_and_order: Mapping[str, tuple[str, int]]


@dataclass(frozen=True, slots=True)
class DerivationRecord:
    derivation_id: str
    version: str
    #: Usually one RFC001 node type; `floating.z_order.page_order.v0`
    #: applies to two (`floating_image`, `text_frame` share one `/anchor`
    #: shape and one page-preorder rule), hence the tuple form.
    node_type: str | tuple[str, ...]
    output_pointer: str
    #: Same-node pointers start with `/`; a pointer on a different node
    #: type (e.g. a table's own cells) is written `"<node_type><pointer>"`,
    #: the same convention `builder_report.py`'s `predicted_by_pointer`
    #: keys use (design §3).
    input_pointers: tuple[str, ...]
    precondition: Callable[[DerivationContext], bool]
    fn: Callable[[DerivationContext], TypedValue]


# ---------------------------------------------------------------------------
# section.columns.widths_mpt.length.v0
# ---------------------------------------------------------------------------


def _section_widths_precondition(ctx: DerivationContext) -> bool:
    return (
        _get(ctx.node, "/columns/count") is not _MISSING
        and _get(ctx.node, "/columns/equal_width") is not _MISSING
    )


def _section_widths_fn(ctx: DerivationContext) -> int:
    """`equal_width=true` derives length `0` (matches every fixture: the
    validator does not check `widths_mpt` length when widths are equal,
    `semantic_validator.py:387`). Any indexed `/columns/widths_mpt/<i>` the
    hypothesis still carries is left exactly where it is -- this function
    only returns the length, it never reads or clears the indexed values,
    so they stay visible for the caller to report as discarded rather than
    silently used or dropped."""
    if _get(ctx.node, "/columns/equal_width"):
        return 0
    return int(_get(ctx.node, "/columns/count"))


# ---------------------------------------------------------------------------
# table.columns.from_grid.v0
# ---------------------------------------------------------------------------


def _own_cells(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Cells of `node`'s own rows only. A nested table sits inside a cell's
    `children`, never inside `node["rows"]` itself, so this never descends
    into one -- no nested-table exclusion logic is needed, only NOT
    recursing past a cell's own `grid_column`/`col_span`."""
    return [cell for row in node.get("rows") or () for cell in row.get("cells") or ()]


def _table_columns_precondition(ctx: DerivationContext) -> bool:
    cells = _own_cells(ctx.node)
    if not cells:
        return False
    return all(
        _get(cell, "/grid_column") is not _MISSING and _get(cell, "/col_span") is not _MISSING
        for cell in cells
    )


def _table_columns_fn(ctx: DerivationContext) -> int:
    return max(cell["grid_column"] + cell["col_span"] for cell in _own_cells(ctx.node))


# ---------------------------------------------------------------------------
# floating.z_order.page_order.v0
# ---------------------------------------------------------------------------


def _z_order_precondition(ctx: DerivationContext) -> bool:
    return ctx.page_ordinal is not None


def _z_order_fn(ctx: DerivationContext) -> int:
    """0-based rank of this node among the floating objects of its own
    page, in document preorder (design §2.3; RFC004-ERRATA-001 E7.1)."""
    return ctx.page_floating_preorder.index(ctx.node["id"])


# ---------------------------------------------------------------------------
# note_ref.marker_text.{footnote_decimal,endnote_lower_roman}_order.v0
# ---------------------------------------------------------------------------

#: ECMA-376 Part 1's default `w:numFmt` for `w:endnotePr`/`w:footnotePr`
#: is prose, not schema: `vendor/ecma376/transitional/wml.xsd` gives
#: `CT_EdnDocProps`/`CT_NumFmt` (`numFmt` optional, `lowerRoman` a valid
#: `ST_NumberFormat` value at `wml.xsd:1374`) but carries no `default=`
#: attribute and no prose describing Word's default when the element is
#: absent. **Not verifiable in this repo** -- only the XSDs are vendored
#: (`vendor/ecma376/README.md`), not the ECMA-376 Part 1 text DEC-039 asks
#: this docstring to cite. DEC-039 records the direction (footnotes
#: decimal, endnotes lower-roman) as the owner's decision pending that
#: citation, and its "known risk" section notes the importer's own default
#: (`importer.py:1818`) disagrees for endnotes without an explicit
#: `w:endnotePr/w:numFmt` -- P7's report will count which gold format
#: dominates the store.


def _bound_note_kind(ctx: DerivationContext) -> str | None:
    note_id = ctx.node.get("note_id")
    if not note_id:
        return None
    kind, _ = ctx.note_kind_and_order.get(note_id, (None, None))
    return kind


def _footnote_marker_precondition(ctx: DerivationContext) -> bool:
    return _bound_note_kind(ctx) == "footnote"


def _footnote_marker_fn(ctx: DerivationContext) -> str:
    _, order = ctx.note_kind_and_order[ctx.node["note_id"]]
    return str(order)


def _endnote_marker_precondition(ctx: DerivationContext) -> bool:
    return _bound_note_kind(ctx) == "endnote"


#: `wml.xsd:1374`'s `lowerRoman` enumeration value, spelled out (subtractive
#: notation, lowercase): `1 -> "i"`, `4 -> "iv"`, `1994 -> "mcmxciv"`.
_ROMAN_NUMERALS: tuple[tuple[int, str], ...] = (
    (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
    (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
    (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
)


def _lower_roman(order: int) -> str:
    if order < 1:
        raise ValueError(f"lower-roman note numbering is 1-based, got {order}")
    remaining = order
    parts: list[str] = []
    for value, symbol in _ROMAN_NUMERALS:
        count, remaining = divmod(remaining, value)
        parts.append(symbol * count)
    return "".join(parts)


def _endnote_marker_fn(ctx: DerivationContext) -> str:
    _, order = ctx.note_kind_and_order[ctx.node["note_id"]]
    return _lower_roman(order)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DERIVATIONS: tuple[DerivationRecord, ...] = (
    DerivationRecord(
        derivation_id="section.columns.widths_mpt.length.v0",
        version="0.1.0",
        node_type="section",
        output_pointer="/columns/widths_mpt",
        input_pointers=("/columns/count", "/columns/equal_width"),
        precondition=_section_widths_precondition,
        fn=_section_widths_fn,
    ),
    DerivationRecord(
        derivation_id="table.columns.from_grid.v0",
        version="0.1.0",
        node_type="table",
        output_pointer="/columns",
        input_pointers=("cell/grid_column", "cell/col_span"),
        precondition=_table_columns_precondition,
        fn=_table_columns_fn,
    ),
    DerivationRecord(
        derivation_id="floating.z_order.page_order.v0",
        version="0.1.0",
        node_type=("floating_image", "text_frame"),
        output_pointer="/anchor/z_order",
        input_pointers=(),
        precondition=_z_order_precondition,
        fn=_z_order_fn,
    ),
    DerivationRecord(
        derivation_id="note_ref.marker_text.footnote_decimal_order.v0",
        version="0.1.0",
        node_type="note_ref",
        output_pointer="/marker_text",
        input_pointers=("/note_id",),
        precondition=_footnote_marker_precondition,
        fn=_footnote_marker_fn,
    ),
    DerivationRecord(
        derivation_id="note_ref.marker_text.endnote_lower_roman_order.v0",
        version="0.1.0",
        node_type="note_ref",
        output_pointer="/marker_text",
        input_pointers=("/note_id",),
        precondition=_endnote_marker_precondition,
        fn=_endnote_marker_fn,
    ),
)


def derivation_registry_hash() -> str:
    """`domain_hash` over the registry, `fn` and `precondition` excluded
    (neither is serializable, and a Component Lock needs to know THAT the
    algorithm changed, via `version` -- see module docstring), sorted by
    `derivation_id` for a definition-order-independent digest."""
    projection = {
        "records": [
            {
                "derivation_id": record.derivation_id,
                "version": record.version,
                "node_type": list(record.node_type)
                if isinstance(record.node_type, tuple)
                else record.node_type,
                "output_pointer": record.output_pointer,
                "input_pointers": list(record.input_pointers),
            }
            for record in sorted(DERIVATIONS, key=lambda r: r.derivation_id)
        ],
    }
    return domain_hash(_DERIVATION_REGISTRY_DOMAIN, projection)

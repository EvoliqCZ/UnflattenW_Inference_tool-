"""Typed property labels (stream C) and known masks (stream D) — RFC003 §27, §29.

Labels are mechanically derived from a canonical AST and never hand-fixed
independently of it — fix the AST and re-derive (RFC003 §27).

Each label is ``{node_id → {node_type, fields: [{node_property_pointer,
value_type, value, known_mask, origin, verification}]}}`` where
``node_property_pointer`` is a *node-relative* JSON Pointer per RFC001
property paths (e.g. ``"/layout/alignment"``) — never the document-root
``ast_pointer`` scope (§27).

The property surface covered here: paragraph semantic/layout/style, run
style+language, section page/margins/columns/break_type and the
`word-visual-v1` advanced section settings, list list_kind/marker/layout,
table/row/cell layout+style, inline/floating image size/crop/rotation
(+ anchor), header/footer `kind`, `note_ref` marker, `field` type and
hyperlink `target`.

No fabricated defaults: only properties actually present in the canonical
AST are emitted. The canonical AST is fully resolved, so absence means
``not_applicable``, never ``unknown`` (§29.1); an explicit ``null`` (e.g.
``style.borders``) is an authoritative known value and is emitted with
``known_mask: true``. `mask_unknown` exists for future partial labels.

The pointer registry is derived mechanically from the pinned AST JSON
Schema (`unflatten/ast/schemas/canonical_ast.schema.json`) — pointers are
normalized by replacing array indices with ``*``.
"""

from __future__ import annotations

import copy
from functools import lru_cache
from typing import Any

from unflatten.ast.schema_validator import _load_schema
from unflatten.dataset.enums import LABEL_ORIGIN
from unflatten.dataset.targets.events import preorder_nodes

# RFC003 §29.4 field-level verification enum (distinct from the
# release-level `verification_tier`). Deterministic execution alone does
# not grant `deterministic_checked` — that requires an independent gate.
FIELD_VERIFICATION: frozenset[str] = frozenset(
    {"unverified", "deterministic_checked", "human_verified", "human_adjudicated"}
)

# Property surface: node type -> top-level property fields whose leaves
# become typed labels (RFC003 §27; task-scoped surface). Covers both
# capability profiles — a `word-core-v0` document simply carries none of
# the `word-visual-v1` node types, so the extra entries emit nothing.
#
# A field named here that the schema marks OPTIONAL is emitted only when
# the node actually carries it (RFC003 §29.1: absence in a fully resolved
# AST is `not_applicable`, never a fabricated default). A field the
# schema marks required and the node lacks is a KeyError, not a skip.
PROPERTY_FIELDS: dict[str, tuple[str, ...]] = {
    "section": (
        "break_type",
        "page",
        "margins",
        "columns",
        "vertical_alignment",
        "page_numbering",
        "line_numbering",
        "page_borders",
    ),
    "paragraph": ("semantic", "layout", "style"),
    "run": ("language", "style"),
    "list": ("list_kind", "marker", "layout"),
    "table": ("layout", "style", "columns"),
    "row": ("layout",),
    "cell": ("layout", "style", "grid_column", "col_span", "row_span"),
    "inline_image": ("size", "crop", "rotation_mdeg", "alt_text"),
    # word-visual-v1 (DEC-003 §1). `header`/`footer` carry `kind` as a
    # typed enum label ON PURPOSE: the event stream says only that a
    # header opens, while `canonicalizer._STORY_KIND_ORDER` sorts the
    # stories BY kind — so a decoder that emits stories in one order and
    # assigns kinds in another canonicalizes to a different tree than it
    # decoded, an `ast_hash` divergence the grammar cannot see. A typed
    # head plus the assembler invariant "at most one story per kind,
    # emitted default -> first -> even" closes that hole.
    "header": ("kind",),
    "footer": ("kind",),
    "note_ref": ("marker_text",),
    "field": ("field_type",),
    "floating_image": ("size", "crop", "rotation_mdeg", "alt_text", "anchor"),
    # `/target/kind` is the link's structure and IS supervisable;
    # `/target/uri` is known but has no model channel (DEC-003 §1.3) and
    # its supervision is withdrawn by `masks.build_channel_mask`.
    "hyperlink": ("target",),
    # word-visual-v1.1 (RFC001-ERRATA-004). `text_frame` reuses the
    # `floating_image` anchor contract verbatim (§5A.1), so it mirrors
    # `floating_image`'s registration for that one field only. It does
    # NOT otherwise mirror `floating_image`: a floating image is a leaf
    # (`resource_ref` excluded as a resource binding), while a
    # `text_frame` is a block CONTAINER (`children` excluded as tree
    # structure, stream A — same treatment as `section`/`cell`/`table`).
    # It has no `style`, no OOXML, no bounding box (E1), so `anchor` is
    # its entire typed property surface.
    "text_frame": ("anchor",),
}

# Node fields CONSCIOUSLY excluded from typed labels (stream C). Every
# corpus-exercised field must appear either in PROPERTY_FIELDS or here —
# the registry-completeness test enforces it, so an omission is a test
# failure, never an accident (D4 should_fix 3).
EXCLUDED_STRUCTURAL: dict[str, tuple[str, ...]] = {
    # child containers: tree structure is stream A (structural events).
    "section": ("children",),
    "paragraph": ("children",),
    "list": ("items",),
    "list_item": ("children",),
    "table": ("rows",),
    "row": ("cells",),
    "cell": ("children",),
    "hyperlink": ("children",),
    "header": ("children",),
    "footer": ("children",),
    "footnote": ("children",),
    "endnote": ("children",),
    # the note body is bound structurally (canonicalizer rewrites it with
    # the id map); it is a reference, not a reconstruction property.
    "note_ref": ("note_id",),
    # run text is stream B (text spans), never a typed label.
    "run": ("text",),
    # resource identity binding — resolved via the document `resources`
    # array (media hashing), not a per-node reconstruction property.
    "inline_image": ("resource_ref",),
    "floating_image": ("resource_ref",),
    "text_frame": ("children",),
}

VALUE_TYPES: frozenset[str] = frozenset({"enum", "string", "integer", "boolean", "object"})


def normalize_pointer(pointer: str) -> str:
    """Replace array-index segments with ``*`` for registry lookup."""
    segments = pointer.lstrip("/").split("/")
    return "/" + "/".join("*" if s.isdigit() else s for s in segments)


def pointer_registry() -> dict[str, dict[str, str]]:
    """Pinned registry: node type -> {normalized node-relative pointer -> value_type}.

    Derived mechanically from the canonical AST JSON Schema so the
    registry cannot drift from the schema; the completeness test over the
    corpus guards the derivation itself.

    Returns a fresh deep copy per call so caller mutation can never leak
    through the cache (deepcopy chosen for simplicity; the registry is
    tiny, so the cost is negligible at this scale).
    """
    return copy.deepcopy(_pointer_registry())


@lru_cache(maxsize=1)
def _pointer_registry() -> dict[str, dict[str, str]]:
    schema = _load_schema()
    defs = schema["$defs"]
    registry: dict[str, dict[str, str]] = {}
    for node_type, fields in PROPERTY_FIELDS.items():
        pointers: dict[str, str] = {}
        node_def = defs[node_type]
        for field in fields:
            if field not in node_def["properties"]:
                raise ValueError(
                    f"PROPERTY_FIELDS names {field!r} on {node_type!r}, which the "
                    "canonical AST schema does not define"
                )
            _walk_schema(node_def["properties"][field], f"/{field}", pointers, defs)
        registry[node_type] = pointers
    return registry


@lru_cache(maxsize=1)
def _required_fields() -> dict[str, frozenset[str]]:
    """Schema-required field names per labeled node type.

    Read from the schema rather than listed, so a field that becomes
    optional (or required) in a later profile does not need a second
    edit here to keep `emit_property_labels` honest.
    """
    defs = _load_schema()["$defs"]
    return {
        node_type: frozenset(defs[node_type].get("required", ()))
        for node_type in PROPERTY_FIELDS
    }


def _resolve_ref(subschema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    while "$ref" in subschema:
        ref = subschema["$ref"]
        if not ref.startswith("#/$defs/"):
            raise ValueError(f"unsupported $ref target: {ref!r}")
        subschema = defs[ref.removeprefix("#/$defs/")]
    return subschema


def _walk_schema(
    subschema: dict[str, Any], pointer: str, out: dict[str, str], defs: dict[str, Any]
) -> None:
    subschema = _resolve_ref(subschema, defs)
    types = subschema.get("type")
    type_set = {types} if isinstance(types, str) else set(types or ())

    if "properties" in subschema:
        if "null" in type_set:
            # nullable container: an explicit null is labeled at the
            # container pointer itself (e.g. "/style/borders": null).
            _merge_value_type(out, pointer, "object")
        for name, prop in subschema["properties"].items():
            _walk_schema(prop, f"{pointer}/{name}", out, defs)
        return
    if "items" in subschema:
        _walk_schema(subschema["items"], f"{pointer}/*", out, defs)
        return
    if "oneOf" in subschema:
        for branch in subschema["oneOf"]:
            branch = _resolve_ref(branch, defs)
            if branch.get("type") == "null":
                continue
            _walk_schema(branch, pointer, out, defs)
        return

    if "enum" in subschema or "const" in subschema:
        _merge_value_type(out, pointer, "enum")
        return
    scalar = (type_set - {"null"}).pop() if type_set - {"null"} else None
    if scalar in ("string", "integer", "boolean"):
        _merge_value_type(out, pointer, scalar)
        return
    raise ValueError(f"cannot derive a value_type for pointer {pointer!r}: {subschema!r}")


def _merge_value_type(out: dict[str, str], pointer: str, value_type: str) -> None:
    current = out.get(pointer)
    if current is None or current == value_type:
        out[pointer] = value_type
    elif "enum" in (current, value_type):
        # a const/enum in one union branch of an otherwise typed value
        # (e.g. table width `value_mpt: const 0` vs integer) — the base
        # type wins; a pure enum union stays "enum".
        out[pointer] = value_type if current == "enum" else current
    else:
        raise ValueError(
            f"conflicting value_type at {pointer!r}: {current!r} vs {value_type!r}"
        )


def emit_property_labels(
    ast: dict[str, Any],
    label_origin: str = "synthetic_generator",
    verification: str = "unverified",
) -> dict[str, dict[str, Any]]:
    """Emit typed property labels for every property present in `ast`.

    Every label carries ``known_mask: true`` — the synthetic corpus is
    fully known; partial labels go through :func:`mask_unknown`.
    """
    if label_origin not in LABEL_ORIGIN:
        raise ValueError(
            f"label_origin must be one of {sorted(LABEL_ORIGIN)}, got {label_origin!r}"
        )
    if verification not in FIELD_VERIFICATION:
        raise ValueError(
            f"verification must be one of {sorted(FIELD_VERIFICATION)}, got {verification!r}"
        )
    registry = pointer_registry()
    required = _required_fields()
    typed: dict[str, dict[str, Any]] = {}
    for node in preorder_nodes(ast):
        node_type = node["type"]
        fields_spec = PROPERTY_FIELDS.get(node_type)
        if fields_spec is None:
            continue
        fields: list[dict[str, Any]] = []
        for field_name in fields_spec:
            if field_name not in node:
                if field_name in required[node_type]:
                    raise KeyError(
                        f"node {node['id']!r} of type {node_type!r} is missing the "
                        f"schema-required property {field_name!r}"
                    )
                continue  # optional and absent: not_applicable (§29.1)
            _emit_value(
                f"/{field_name}",
                node[field_name],
                fields,
                registry[node_type],
                label_origin,
                verification,
            )
        if fields:
            typed[node["id"]] = {"node_type": node_type, "fields": fields}
    return typed


def _emit_value(
    pointer: str,
    value: Any,
    fields: list[dict[str, Any]],
    node_registry: dict[str, str],
    label_origin: str,
    verification: str,
) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            _emit_value(
                f"{pointer}/{key}", value[key], fields, node_registry, label_origin, verification
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _emit_value(
                f"{pointer}/{index}", item, fields, node_registry, label_origin, verification
            )
        return
    normalized = normalize_pointer(pointer)
    value_type = node_registry.get(normalized)
    if value_type is None:
        raise ValueError(f"property pointer {pointer!r} is not in the pinned registry")
    fields.append(
        {
            "node_property_pointer": pointer,
            "value_type": value_type,
            "value": value,
            "known_mask": True,
            "origin": label_origin,
            "verification": verification,
        }
    )


def mask_unknown(
    labels: dict[str, dict[str, Any]],
    pointers: Any,
) -> dict[str, dict[str, Any]]:
    """Return a copy of `labels` with the given ``(node_id, pointer)``
    fields set to ``known_mask: false`` and ``value: null`` (§29.1).

    For future partial labels; raises ``ValueError`` when a pointer
    matches no emitted label rather than silently ignoring it.
    """
    result = copy.deepcopy(labels)
    for node_id, pointer in pointers:
        entry = result.get(node_id)
        field = None
        if entry is not None:
            field = next(
                (f for f in entry["fields"] if f["node_property_pointer"] == pointer), None
            )
        if field is None:
            raise ValueError(f"no label to mask at node {node_id!r}, pointer {pointer!r}")
        field["known_mask"] = False
        field["value"] = None
    return result


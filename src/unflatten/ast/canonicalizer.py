"""Canonicalizer (RFC001 §35 — the 12 mandatory canonicalization steps).

`canonicalize(doc)` turns a schema-valid document into its canonical
form: NFC-normalized text, merged adjacent equivalent runs, stably
renumbered node IDs, a sorted resource registry, uppercase hex colors,
and integer-only units — ready for `to_canonical_json` / `ast_hash`.

Only schema-valid, semantically valid input is accepted; anything else
raises `CanonicalizationError` carrying the offending Issues rather than
guessing a repair.
"""

from __future__ import annotations

import copy
import unicodedata
from typing import Any

from unflatten.ast.issues import Issue
from unflatten.ast.schema_validator import validate_schema
from unflatten.ast.semantic_validator import validate_semantics

_BLOCKING_SEVERITIES = {"error", "fatal"}

# RFC001 §14/§15: node type -> field holding its children, for the
# subset of container types that carry an `id` and appear in the
# preorder node sequence.
_CHILDREN_KEY: dict[str, str] = {
    "section": "children",
    "paragraph": "children",
    "list": "items",
    "list_item": "children",
    "table": "rows",
    "row": "cells",
    "cell": "children",
    "hyperlink": "children",
    "header": "children",
    "footer": "children",
    "footnote": "children",
    "endnote": "children",
    "text_frame": "children",
}

# Root keys holding note stories, in canonical preorder order: footnotes
# before endnotes, both after every section.
_NOTE_KEYS: tuple[str, ...] = ("footnotes", "endnotes")

# Canonical order of a section's header/footer variants (word-visual-v1).
# Array order is not semantic — Word selects a story by kind — so it is
# normalized, or two identical documents would hash differently.
_STORY_KIND_ORDER: dict[str, int] = {"default": 0, "first": 1, "even": 2}

# Preorder traversal order inside a section (RFC001 §35 step 5): the
# stories come before the body, so `n*` numbering is defined for a
# section that has them. Everything downstream (event stream, fragment
# ownership) inherits this order.
_SECTION_CHILD_KEYS: tuple[str, ...] = ("headers", "footers", "children")

# Node types whose children list is a run of inline nodes eligible for
# adjacent-equivalent-run merging (RFC001 §18.2).
_INLINE_CONTAINER_TYPES = {"paragraph", "hyperlink"}

_STYLE_HOLDER_TYPES = {"paragraph", "table", "cell"}


class CanonicalizationError(Exception):
    """Raised when a document cannot be canonicalized.

    Carries the blocking (`error`/`fatal`) Issues that caused the
    rejection; the canonicalizer only accepts schema-valid,
    semantically-valid input (RFC001 §35 step 1, step 10).
    """

    def __init__(self, issues: list[Issue]):
        self.issues = issues
        summary = "; ".join(f"{i.code}: {i.message}" for i in issues)
        super().__init__(f"canonicalization failed: {summary}")


def canonicalize(doc: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize `doc` per RFC001 §35. Does not mutate the input."""
    _raise_if_blocking(validate_schema(doc))

    working = copy.deepcopy(doc)
    working["metadata"] = _canon_metadata(working.get("metadata") or {})
    working["sections"] = [_canon_node(s) for s in working.get("sections", [])]
    for key in _NOTE_KEYS:
        if key in working:
            working[key] = _order_notes(
                [_canon_node(n) for n in working.get(key) or []],
                working.get("sections", []),
            )
    working = _renumber_ids(working)
    working["resources"] = sorted(working.get("resources", []), key=lambda r: r.get("id", ""))

    # Step 9 (RFC001 §35): explicit breaks are already represented as
    # `page_break`/`column_break` block nodes by the schema — no-op here.
    # Step 4: no source-only metadata is defined in v0.1 core — no-op.
    _assert_no_floats(working)

    _raise_if_blocking(validate_semantics(working))

    return working


def _raise_if_blocking(issues: list[Issue]) -> None:
    blocking = [i for i in issues if i.severity in _BLOCKING_SEVERITIES]
    if blocking:
        raise CanonicalizationError(blocking)


def _canon_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    for key in ("language", "title", "subject"):
        value = result.get(key)
        if isinstance(value, str):
            result[key] = unicodedata.normalize("NFC", value)
    return result


def _child_lists(node: dict[str, Any]) -> list[str]:
    """Keys of `node` holding child nodes, in canonical preorder order."""
    node_type = node.get("type")
    if node_type == "section":
        return [key for key in _SECTION_CHILD_KEYS if isinstance(node.get(key), list)]
    if not isinstance(node_type, str):
        return []
    key = _CHILDREN_KEY.get(node_type)
    return [key] if key and isinstance(node.get(key), list) else []


def _canon_node(node: dict[str, Any]) -> dict[str, Any]:
    node = dict(node)
    node_type: str | None = node.get("type")

    if node_type == "run":
        node["text"] = unicodedata.normalize("NFC", node["text"])
        if isinstance(node.get("language"), str):
            node["language"] = unicodedata.normalize("NFC", node["language"])
        node["style"] = _canon_run_style(node["style"])
    elif node_type in ("inline_image", "floating_image"):
        if isinstance(node.get("alt_text"), str):
            node["alt_text"] = unicodedata.normalize("NFC", node["alt_text"])
    elif node_type == "note_ref":
        # The marker is visible text on the page, so it is normalized
        # like run text (RFC001 §13/§35).
        if isinstance(node.get("marker_text"), str):
            node["marker_text"] = unicodedata.normalize("NFC", node["marker_text"])
    elif node_type == "list":
        marker = node.get("marker")
        if isinstance(marker, dict):
            marker = dict(marker)
            # ERRATA-002 E6: marker.text (and suffix) are visible text and
            # must be NFC-normalized like run text (RFC001 §13/§35).
            for key in ("text", "suffix"):
                if isinstance(marker.get(key), str):
                    marker[key] = unicodedata.normalize("NFC", marker[key])
            node["marker"] = marker
    elif node_type in _STYLE_HOLDER_TYPES and isinstance(node.get("style"), dict):
        node["style"] = _canon_style_colors(node["style"])
    elif node_type == "section" and isinstance(node.get("page_borders"), dict):
        page_borders = dict(node["page_borders"])
        page_borders["sides"] = _canon_style_colors({"borders": page_borders.get("sides")})[
            "borders"
        ]
        node["page_borders"] = page_borders

    for children_key in _child_lists(node):
        children = [_canon_node(c) for c in node[children_key]]
        if node_type in _INLINE_CONTAINER_TYPES:
            children = _merge_adjacent_runs(children)
        if children_key in ("headers", "footers"):
            children = sorted(
                children, key=lambda s: _STORY_KIND_ORDER.get(s.get("kind"), len(_STORY_KIND_ORDER))
            )
        node[children_key] = children

    return node


def _order_notes(
    notes: list[dict[str, Any]], sections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Order notes by the preorder position of their first `note_ref`.

    Array order carries no meaning of its own, so two documents that
    differ only in it must canonicalize identically. Runs before
    renumbering, so it keys on the incoming ids.
    """
    first_ref: dict[str, int] = {}
    for i, node in enumerate(_preorder(sections)):
        if node.get("type") == "note_ref":
            note_id = node.get("note_id")
            if isinstance(note_id, str):
                first_ref.setdefault(note_id, i)
    # An unreferenced note is a semantic error (ORPHAN_FOOTNOTE), which is
    # raised after renumbering; until then it sorts last, stably.
    return sorted(notes, key=lambda n: first_ref.get(n.get("id"), float("inf")))


def _merge_adjacent_runs(children: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """RFC001 §18.2/§35: merge adjacent runs with identical language+style."""
    merged: list[dict[str, Any]] = []
    for child in children:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.get("type") == "run"
            and child.get("type") == "run"
            and prev.get("language") == child.get("language")
            and prev.get("style") == child.get("style")
        ):
            combined = dict(prev)
            # Concatenating two NFC strings can leave the join out of
            # canonical combining order (e.g. a run ending in shadda
            # followed by one starting with fatha), so re-normalize.
            combined["text"] = unicodedata.normalize("NFC", prev["text"] + child["text"])
            merged[-1] = combined
        else:
            merged.append(child)
    return merged


def _canon_color(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("#"):
        return value.upper()
    return value


def _canon_run_style(style: dict[str, Any]) -> dict[str, Any]:
    style = dict(style)
    if "color" in style:
        style["color"] = _canon_color(style["color"])
    if "highlight" in style:
        style["highlight"] = _canon_color(style["highlight"])
    return style


def _canon_style_colors(style: dict[str, Any]) -> dict[str, Any]:
    style = dict(style)
    if "background" in style:
        style["background"] = _canon_color(style["background"])
    borders = style.get("borders")
    if isinstance(borders, dict):
        new_borders = {}
        for side, border in borders.items():
            if isinstance(border, dict):
                border = dict(border)
                border["color"] = _canon_color(border.get("color"))
            new_borders[side] = border
        style["borders"] = new_borders
    return style


def _renumber_ids(doc: dict[str, Any]) -> dict[str, Any]:
    """RFC001 §11.1/§35 step 5: stable preorder n1, n2, ... renumbering.

    Two passes: collect the old-id -> new-id map in preorder, then
    rebuild the tree with new ids and rewrite internal_anchor references.
    """
    id_map: dict[str, str] = {}
    counter = 0
    roots = list(doc.get("sections", []))
    for key in _NOTE_KEYS:
        roots += list(doc.get(key) or [])
    for node in _preorder(roots):
        node_id = node.get("id")
        if isinstance(node_id, str):
            counter += 1
            id_map[node_id] = f"n{counter}"

    dangling: list[Issue] = []
    doc = dict(doc)
    doc["sections"] = [
        _rebuild(s, id_map, f"/sections/{i}", dangling)
        for i, s in enumerate(doc.get("sections", []))
    ]
    for key in _NOTE_KEYS:
        if key in doc:
            doc[key] = [
                _rebuild(n, id_map, f"/{key}/{i}", dangling)
                for i, n in enumerate(doc.get(key) or [])
            ]
    if dangling:
        raise CanonicalizationError(dangling)
    return doc


def _preorder(nodes: list[dict[str, Any]]):
    for node in nodes:
        yield node
        for children_key in _child_lists(node):
            yield from _preorder(node[children_key])


def _rebuild(
    node: dict[str, Any], id_map: dict[str, str], path: str, dangling: list[Issue]
) -> dict[str, Any]:
    """Rebuild `node` with renumbered ids (RFC001 §35 step 5).

    ERRATA-001 E5: an internal_anchor target MUST be rewritten via
    `id_map` (the old-id -> new-id mapping), never resolved by checking
    whether its literal text happens to match a node id in the already-
    renumbered tree — a dangling anchor's old literal (e.g. "#n1") could
    coincidentally collide with a freshly assigned id and silently
    repoint to the wrong node. A miss in `id_map` is always dangling.
    """
    node = dict(node)
    old_id = node.get("id")
    if isinstance(old_id, str) and old_id in id_map:
        node["id"] = id_map[old_id]

    if node.get("type") == "hyperlink":
        target = node.get("target")
        if isinstance(target, dict) and target.get("kind") == "internal_anchor":
            uri = target.get("uri", "")
            if isinstance(uri, str) and uri.startswith("#"):
                old_target_id = uri[1:]
                if old_target_id in id_map:
                    target = dict(target)
                    target["uri"] = "#" + id_map[old_target_id]
                    node["target"] = target
                else:
                    dangling.append(
                        Issue(
                            code="BROKEN_INTERNAL_ANCHOR",
                            severity="error",
                            node_id=node.get("id"),
                            path=f"{path}/target/uri",
                            message=f"internal_anchor target '{uri}' does not resolve to an "
                            "existing node id; refusing to renumber a dangling anchor.",
                        )
                    )

    if node.get("type") == "note_ref":
        old_note_id = node.get("note_id")
        if isinstance(old_note_id, str):
            if old_note_id in id_map:
                node["note_id"] = id_map[old_note_id]
            else:
                dangling.append(
                    Issue(
                        code="BROKEN_NOTE_REF",
                        severity="error",
                        node_id=node.get("id"),
                        path=f"{path}/note_id",
                        message=f"note_id '{old_note_id}' does not resolve to an existing "
                        "node id; refusing to renumber a dangling note reference.",
                    )
                )

    for children_key in _child_lists(node):
        node[children_key] = [
            _rebuild(c, id_map, f"{path}/{children_key}/{i}", dangling)
            for i, c in enumerate(node[children_key])
        ]

    return node


def _assert_no_floats(value: Any) -> None:
    """RFC001 §35 step 8: units MUST be integers — schema already enforces
    this; this is a defense-in-depth assertion, not a repair.
    """
    if isinstance(value, float):
        raise AssertionError(f"non-integer unit value found during canonicalization: {value!r}")
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_floats(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_floats(item)

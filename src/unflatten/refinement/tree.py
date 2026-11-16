"""Canonical AST tree walk and JSON-pointer access for the refinement core.

Mirrors the child-key/preorder convention in `unflatten.ast.canonicalizer`
(RFC001 §35 step 5) rather than importing its private helpers: this
module operates on an *already canonical* document (node ids already
stably renumbered `n1, n2, ...`), which is a different contract from the
canonicalizer's own pre-canonicalization walk, so a shared name would
invite the two to silently drift onto the same code for two different
jobs.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

#: Node type -> child-list keys, in canonical preorder order (RFC001 §35
#: step 5 / §14-15). Kept in one place so `target_catalog`, `patch_applier`
#: and `corruption` all see the identical tree shape.
CHILDREN_KEYS: dict[str, tuple[str, ...]] = {
    "section": ("headers", "footers", "children"),
    "paragraph": ("children",),
    "hyperlink": ("children",),
    "list": ("items",),
    "list_item": ("children",),
    "table": ("rows",),
    "row": ("cells",),
    "cell": ("children",),
    "header": ("children",),
    "footer": ("children",),
    "footnote": ("children",),
    "endnote": ("children",),
    "text_frame": ("children",),
}


def document_roots(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Every root node of the canonical tree, in preorder root order."""
    roots = list(doc.get("sections", []))
    roots += list(doc.get("footnotes") or [])
    roots += list(doc.get("endnotes") or [])
    return roots


def preorder(nodes: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Depth-first preorder over `nodes` and their children."""
    for node in nodes:
        yield node
        for key in CHILDREN_KEYS.get(node.get("type"), ()):
            children = node.get(key)
            if isinstance(children, list):
                yield from preorder(children)


def index_by_id(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """`node_id -> node` for every node carrying an `id` (by reference:
    mutating a returned node mutates `doc`)."""
    return {node["id"]: node for node in preorder(document_roots(doc)) if "id" in node}


def get_pointer(node: dict[str, Any], pointer: str) -> tuple[bool, Any]:
    """Resolve a node-relative JSON Pointer (`"/layout/indent/left_mpt"`).

    Returns `(present, value)`. `present` is False the moment any
    intermediate segment is missing or not an object — a repeated (`*`)
    segment is never resolved here, callers must not pass one.
    """
    if "*" in pointer:
        raise ValueError(f"get_pointer does not resolve repeated pointers: {pointer!r}")
    current: Any = node
    for part in (p for p in pointer.split("/") if p):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def set_pointer(node: dict[str, Any], pointer: str, value: Any) -> None:
    """Set an *already-present* node-relative JSON Pointer in place.

    Every intermediate segment must already exist as a dict — callers in
    this package only ever call this after `get_pointer` confirmed
    presence, so no segment is ever created here (RFC006 §24.2 "no
    repair").
    """
    parts = [p for p in pointer.split("/") if p]
    current = node
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value

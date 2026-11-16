"""Required RFC001 leaf properties per node type (AST Builder design §10, T0).

The strict AST Builder (RFC004 §33.6) must know which properties of a node
are required, so that a missing one is reported as unresolved instead of
being filled with `0`, `""` or `null`. This module derives them from the
canonical AST schema itself rather than a hand-kept list, so a schema change
moves them with it.

A leaf is required when every object on its path lists it in `required`.
Elements of an array of objects are addressed `/*` (`/layout/tabs/*/position_mpt`);
an object with no required member (nullable `borders`) is itself the leaf.
For a `oneOf` union (e.g. `anchor.horizontal`: align vs offset) the leaves
of every arm are returned; which arm applies is the builder's decision.
Structural members (`id`, `type`, children, text) are not properties.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

_SCHEMA = Path(__file__).parents[1] / "ast" / "schemas" / "canonical_ast.schema.json"

#: Members that carry structure or text, never a typed property.
_STRUCTURAL = frozenset(
    {"id", "type", "children", "text", "blocks", "rows", "cells", "items", "sections",
     "headers", "footers", "footnotes", "endnotes", "runs", "content"}
)

#: A registry pointer's trailing `/*` or `/<n>` names elements of an array field.
_TRAILING_INDEX = re.compile(r"(?:/(?:\*|\d+))+$")


@cache
def _defs() -> dict[str, Any]:
    return json.loads(_SCHEMA.read_text(encoding="utf-8"))["$defs"]


def _resolve(node: Any) -> Any:
    defs = _defs()
    while isinstance(node, dict) and "$ref" in node:
        node = defs[node["$ref"].rsplit("/", 1)[-1]]
    return node


def _collect(node: Any, path: str, out: set[str]) -> None:
    node = _resolve(node)
    if not isinstance(node, dict):
        return
    arms = node.get("oneOf") or node.get("anyOf")
    if arms and any("properties" in _resolve(a) for a in arms):
        for arm in arms:
            _collect(arm, path, out)
        return
    if "properties" in node:
        required = [
            key for key in node["properties"]
            if key in set(node.get("required", ())) and not (path == "" and key in _STRUCTURAL)
        ]
        if not required and path:
            # An object with no required member (e.g. nullable `borders`,
            # whose `null` means "no borders") is itself the value.
            out.add(path)
        for key in required:
            _collect(node["properties"][key], f"{path}/{key}", out)
        return
    items = node.get("items")
    if node.get("type") == "array" and items is not None and "properties" in _resolve(items):
        _collect(items, f"{path}/*", out)
        return
    out.add(path)


def _bounds(node: Any, parts: tuple[str, ...]) -> tuple[int | None, int | None] | None:
    """`(minimum, maximum)` of the leaf `parts` below `node`, or `None` when
    the pointer does not resolve there. A segment missing under `properties`
    is looked for in every `oneOf`/`anyOf` arm; the arms that carry the leaf
    combine to the LOOSEST bounds (a value legal in any arm is never clamped
    away). An integer `const` c is the bounds `(c, c)`."""
    node = _resolve(node)
    if not isinstance(node, dict):
        return None
    if not parts:
        const = node.get("const")
        if isinstance(const, int) and not isinstance(const, bool):
            return const, const
        return node.get("minimum"), node.get("maximum")
    part, rest = parts[0], parts[1:]
    if part == "*" or part.isdigit():
        child = node.get("items")
    else:
        child = node.get("properties", {}).get(part)
    if child is not None:
        return _bounds(child, rest)
    arms = node.get("oneOf") or node.get("anyOf") or ()
    found = [b for b in (_bounds(arm, parts) for arm in arms) if b is not None]
    if not found:
        return None
    lows, highs = zip(*found, strict=True)
    return (
        None if any(low is None for low in lows) else min(lows),
        None if any(high is None for high in highs) else max(highs),
    )


@cache
def integer_bounds(node_type: str, pointer: str) -> tuple[int | None, int | None]:
    """`(minimum, maximum)` of the integer leaf at `pointer` under
    `$defs[node_type]`; a `*` or numeric segment steps into an array's
    `items`, a leaf inside a `oneOf`/`anyOf` arm (table `/layout/width/
    value_mpt`) gets the loosest bounds over the arms that define it.
    `(None, None)` when the leaf declares no bound or the pointer does not
    resolve (typed decode then leaves the value as predicted)."""
    parts = tuple(segment for segment in pointer.split("/") if segment)
    return _bounds(_defs().get(node_type), parts) or (None, None)


def required_leaf_pointers(node_type: str) -> frozenset[str]:
    """The required leaf pointers of `node_type` (KeyError if unknown)."""
    out: set[str] = set()
    _collect(_defs()[node_type], "", out)
    return frozenset(out)


def registry_pointer_matches(
    node_type: str, pointer: str, table: Mapping[tuple[str, str], Any]
) -> bool:
    """Whether `table` (a registry keyed `(node_type, pointer)`) names
    `pointer` directly or element-wise (`pointer/*`)."""
    return any(
        nt == node_type and (p == pointer or _TRAILING_INDEX.sub("", p) == pointer)
        for nt, p in table
    )


def union_arm_leaves(node_type: str, container: str) -> dict[str, frozenset[str]]:
    """`{mode value: leaf pointers}` of one `oneOf` union container (e.g.
    `/layout/spacing/line`); every arm is selected by a `mode` const. An
    arm's leaves are its REQUIRED keys: `table /layout/width` `auto` merely
    tolerates `value_mpt: 0`, it does not carry a width."""
    node = _resolve(_defs()[node_type])
    for part in container.strip("/").split("/"):
        node = _resolve(node["properties"][part])
    arms: dict[str, frozenset[str]] = {}
    for arm in node.get("oneOf") or node.get("anyOf") or ():
        arm = _resolve(arm)
        mode = arm["properties"]["mode"]["const"]
        arms[mode] = frozenset(f"{container}/{key}" for key in arm["required"])
    return arms

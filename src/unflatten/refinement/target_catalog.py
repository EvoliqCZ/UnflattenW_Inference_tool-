"""RFC006 §19 Refinement Target Catalog — Authority catalog + Model view.

Derived entirely by *reading* two existing surfaces, never modifying
either:

- `unflatten.model.typed_property_pointer_map.FAMILY` — the closed
  `(node_type, pointer) -> value_type` typed-property surface (RFC004
  §24/§26).
- The RFC006 §21.2/§21.3 mutable-property allowlist, applied here as an
  explicit exclusion filter (`_EXCLUDED`) over that surface.

Simplifications relative to the full RFC006 §19.2 schema, stated rather
than hidden:

- No `catalog_hash`/`*_hash` ArtifactRef fields (Component Lock,
  renderer capability manifest, render comparison profile, ...) — this
  package has none of those surfaces yet. Only `base_ast_hash` is
  carried, which is enough to bind a catalog to one exact document.
- Repeated (`*`-segment) pointers (`paragraph:/layout/tabs/*/...`,
  `table:/columns/*/width_mpt`, `section:/columns/widths_mpt/*`) are
  excluded from V0: they need a per-element target, which is future
  scope, not silently included as a per-array target.
- `property_family` here is the FAMILY value type (`integer` / `enum`
  / `boolean`), not RFC006's dotted family name
  (`paragraph.indent`) — the latter is not tracked anywhere in this
  repo and inventing one would be a silent assumption.

RFC006 §21.2 lists structural/topology-adjacent and review-only
properties that must NOT auto-commit even though they are typed
targets today:

- `cell:/col_span`, `cell:/row_span`, `cell:/grid_column` — span/grid
  identity (§21.3 `change_col_span_or_row_span`).
- `section:/page/*`, `section:/margins/*`, `section:/columns/*`,
  `section:/break_type` — "Section page size, margins, columns a
  pagination-sensitive flags iba review-only, kým OPEN-006-001/003/008
  nie sú uzavreté real-gold dôkazom" (§21.2). Table column *widths*
  (`table:/columns/*/width_mpt`) are unaffected — that bullet names
  section-level page geometry, not table geometry, and is excluded
  above anyway as a repeated pointer.

This is a conservative reading of an RFC written in prose, not a table;
recorded as an explicit assumption for owner review rather than a
silent pick (Karpathy principle 1).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.ast.hashing import ast_hash
from unflatten.model.typed_property_pointer_map import FAMILY
from unflatten.refinement.tree import document_roots, get_pointer, preorder

_EXCLUDED_PREFIXES: dict[str, tuple[str, ...]] = {
    "cell": ("/col_span", "/row_span", "/grid_column"),
    "section": ("/page/", "/margins/", "/columns/", "/break_type"),
}


def _is_excluded(node_type: str, pointer: str) -> bool:
    for prefix in _EXCLUDED_PREFIXES.get(node_type, ()):
        if pointer == prefix or pointer.startswith(prefix):
            return True
    return False


#: `(node_type, pointer)` in a fixed, deterministic global order: sorted
#: tuples, scalar (non-repeated) only, with the §21.2/§21.3 exclusions
#: applied. This fixed order IS the "patchable-property-registry
#: ordinal" RFC006 §19.5 requires.
PATCHABLE_REGISTRY: tuple[tuple[str, str], ...] = tuple(
    sorted(
        key
        for key in FAMILY
        if "*" not in key[1] and not _is_excluded(key[0], key[1])
    )
)


@dataclass(frozen=True)
class TargetEntry:
    """One Authority Target Catalog entry (RFC006 §19.2, simplified)."""

    target_slot_id: str
    node_id: str
    node_type: str
    node_property_pointer: str
    expected_presence: str  # "present" -- V0 only ever admits present targets
    expected_value: Any
    property_family: str  # FAMILY value type: "integer" | "enum" | "boolean"
    node_precondition_hash: str
    allowed_operation_kinds: tuple[str, ...] = ("set_typed_property",)


@dataclass(frozen=True)
class TargetCatalog:
    """RFC006 §19.2 Authority catalog, bound to one exact base AST."""

    base_ast_hash: str
    entries: tuple[TargetEntry, ...]

    def by_slot(self, target_slot_id: str) -> TargetEntry:
        for entry in self.entries:
            if entry.target_slot_id == target_slot_id:
                return entry
        raise UnknownTargetSlotError(target_slot_id)


@dataclass(frozen=True)
class ModelViewEntry:
    """RFC006 §19.3 model view entry — no node_id, no hashes (§19.4)."""

    target_slot_id: str
    node_type: str
    property_family: str
    current_value: Any
    allowed_operation_kinds: tuple[str, ...]


class UnknownTargetSlotError(KeyError):
    """A `target_slot_id` is not present in this Target Catalog."""


def _node_precondition_hash(base_ast_hash: str, node: dict[str, Any]) -> str:
    """RFC006 §19.6, over the node's exact canonical projection."""
    payload = b"rfc006.node-precondition.v0\0" + base_ast_hash.encode(
        "utf-8"
    ) + b"\0" + node["id"].encode("utf-8") + b"\0" + to_canonical_json(node)
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def build_target_catalog(doc: dict[str, Any]) -> TargetCatalog:
    """Build the Authority Target Catalog for a canonical AST `doc`.

    `doc` MUST already be canonical (node ids `n1, n2, ...` already
    assigned) — the catalog does not canonicalize its input, it reads
    it.

    Slot allocation (§19.5): canonical preorder node ordinal ×
    `PATCHABLE_REGISTRY` ordinal, `rt1..rtN` without gaps. Only
    currently-*present* targets are admitted (module docstring):
    creating a currently-absent optional property has no inverse in the
    V0 `set_typed_property`-only operation language (§21.1), so it is
    out of catalog scope rather than silently unpatchable-but-listed.
    """
    base_hash = ast_hash(doc)
    entries: list[TargetEntry] = []
    slot = 0
    for node in preorder(document_roots(doc)):
        node_type = node.get("type")
        node_id = node.get("id")
        if not isinstance(node_type, str) or not isinstance(node_id, str):
            continue
        for family_node_type, pointer in PATCHABLE_REGISTRY:
            if family_node_type != node_type:
                continue
            present, value = get_pointer(node, pointer)
            if not present:
                continue
            slot += 1
            entries.append(
                TargetEntry(
                    target_slot_id=f"rt{slot}",
                    node_id=node_id,
                    node_type=node_type,
                    node_property_pointer=pointer,
                    expected_presence="present",
                    expected_value=value,
                    property_family=FAMILY[(node_type, pointer)],
                    node_precondition_hash=_node_precondition_hash(base_hash, node),
                )
            )
    return TargetCatalog(base_ast_hash=base_hash, entries=tuple(entries))


def model_view(catalog: TargetCatalog) -> tuple[ModelViewEntry, ...]:
    """RFC006 §19.3/§19.4 model-safe projection: no node_id, no hashes."""
    return tuple(
        ModelViewEntry(
            target_slot_id=entry.target_slot_id,
            node_type=entry.node_type,
            property_family=entry.property_family,
            current_value=entry.expected_value,
            allowed_operation_kinds=entry.allowed_operation_kinds,
        )
        for entry in catalog.entries
    )

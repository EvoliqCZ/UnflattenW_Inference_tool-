"""RFC006 §24 Transactional Patch Applier + §24.3 Conservation Report.

Simplifications relative to the full RFC006 text, stated rather than
hidden:

- No Component Lock / renderer capability manifest verification (§24.1
  step 1) — this package carries no such locks yet.
- `structure_projection_hash` here is a topology-only projection over
  node type, span/grid fields and child-list shape, keyed on canonical
  node id (RFC006 §24.3 requires it keyed on stable *preorder ordinals*
  instead of ids so the projection survives an id rebase; V0 admits
  only single-property `set_typed_property` patches that never change
  topology, so an id rebase never happens here — `_STRUCTURE_LOCAL_KEYS`
  documents exactly which per-node fields are treated as structural).
- `explicit_control_hash` covers `page_break`/`column_break` node
  positions only (the two RFC001 explicit-break node types).
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.ast.canonicalizer import CanonicalizationError, canonicalize
from unflatten.ast.hashing import ast_hash
from unflatten.model.typed_property_pointer_map import FAMILY
from unflatten.refinement.patch_language import (
    ASTPatchSet,
    InvalidReplacementError,
    _validate_replacement,
)
from unflatten.refinement.tree import (
    CHILDREN_KEYS,
    document_roots,
    get_pointer,
    preorder,
    set_pointer,
)

_MISSING = object()

#: Per-node fields treated as topology, not property value (RFC006
#: §24.3): node type plus the span/grid identity RFC006 §21.3 forbids
#: mutating (`change_col_span_or_row_span`).
_STRUCTURE_LOCAL_KEYS = ("type", "col_span", "row_span", "grid_column")

#: Union of every child-list key across node types (RFC001 §14-15),
#: excluded from the per-node local-property diff since child structure
#: is compared separately, node-by-node, by the same zip.
_CHILD_LIST_KEYS = frozenset(key for keys in CHILDREN_KEYS.values() for key in keys)


class PatchPreconditionError(RuntimeError):
    """A PatchSet's stated base/precondition does not match the AST it is
    applied to. Not an invariant violation — the transaction never ran."""


class PatchInvariantViolation(RuntimeError):
    """Applying an otherwise-valid PatchSet produced a candidate whose
    Conservation Report failed one or more RFC006 §24.3 invariants."""

    def __init__(self, report: ConservationReport):
        self.report = report
        super().__init__(f"conservation violations: {report.violations}")


@dataclass(frozen=True)
class ConservationReport:
    """RFC006 §24.3 Conservation Report."""

    visible_text_hash_before: str
    visible_text_hash_after: str
    structure_projection_hash_before: str
    structure_projection_hash_after: str
    reading_order_projection_hash_before: str
    reading_order_projection_hash_after: str
    resource_manifest_hash_before: str
    resource_manifest_hash_after: str
    profile_before: str
    profile_after: str
    unauthorized_changes: tuple[str, ...]
    violations: tuple[str, ...]


def _visible_text_hash(doc: dict[str, Any]) -> str:
    text = "\x1f".join(
        node["text"] for node in preorder(document_roots(doc)) if node.get("type") == "run"
    )
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _structure_node(node: dict[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {key: node[key] for key in _STRUCTURE_LOCAL_KEYS if key in node}
    for key in CHILDREN_KEYS.get(node.get("type"), ()):
        children = node.get(key)
        if isinstance(children, list):
            entry[key] = [_structure_node(c) for c in children]
    return entry


def _structure_projection_hash(doc: dict[str, Any]) -> str:
    projection = [_structure_node(n) for n in document_roots(doc)]
    payload = b"rfc006.structure-projection.v0\0" + to_canonical_json(projection)
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _reading_order_hash(doc: dict[str, Any]) -> str:
    types = [n.get("type") for n in preorder(document_roots(doc))]
    payload = b"rfc006.reading-order-projection.v0\0" + to_canonical_json(types)
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _resource_manifest_hash(doc: dict[str, Any]) -> str:
    resources = sorted(doc.get("resources", []), key=lambda r: r.get("id", ""))
    payload = b"rfc006.resource-manifest.v0\0" + to_canonical_json(resources)
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _local_fields(node: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in node.items() if k not in _CHILD_LIST_KEYS}


def _diff_paths(before: Any, after: Any, path: str, out: list[str]) -> None:
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            _diff_paths(before.get(key, _MISSING), after.get(key, _MISSING), f"{path}/{key}", out)
        return
    if (
        isinstance(before, list)
        and isinstance(after, list)
        and len(before) == len(after)
    ):
        for i, (x, y) in enumerate(zip(before, after)):
            _diff_paths(x, y, f"{path}/{i}", out)
        return
    if before != after:
        out.append(path)


def _unauthorized_changes(
    base: dict[str, Any], candidate: dict[str, Any], patched: set[tuple[str, str]]
) -> tuple[str, ...]:
    """Node-by-node local-field diff, zipped by preorder position.

    Only meaningful once `structure_projection_hash` is confirmed equal
    — same node count/type/shape at every ordinal, so `node["id"]` is
    identical before/after (module docstring: no id rebase in V0).
    """
    unauthorized: list[str] = []
    for before_node, after_node in zip(
        preorder(document_roots(base)), preorder(document_roots(candidate)), strict=True
    ):
        node_id = before_node.get("id")
        diffs: list[str] = []
        _diff_paths(_local_fields(before_node), _local_fields(after_node), "", diffs)
        for pointer in diffs:
            if (node_id, pointer) not in patched:
                unauthorized.append(f"{node_id}:{pointer}")
    return tuple(unauthorized)


def _build_conservation_report(
    base: dict[str, Any], candidate: dict[str, Any], patch_set: ASTPatchSet
) -> ConservationReport:
    text_before, text_after = _visible_text_hash(base), _visible_text_hash(candidate)
    struct_before = _structure_projection_hash(base)
    struct_after = _structure_projection_hash(candidate)
    order_before, order_after = _reading_order_hash(base), _reading_order_hash(candidate)
    resources_before = _resource_manifest_hash(base)
    resources_after = _resource_manifest_hash(candidate)
    profile_before, profile_after = base.get("profile"), candidate.get("profile")

    violations: list[str] = []
    if text_before != text_after:
        violations.append("PATCH_TEXT_MUTATION_FORBIDDEN")
    if struct_before != struct_after:
        violations.append("PATCH_STRUCTURE_MUTATION_FORBIDDEN")
    if order_before != order_after:
        violations.append("PATCH_READING_ORDER_MUTATION_FORBIDDEN")
    if resources_before != resources_after:
        violations.append("PATCH_RESOURCE_MUTATION_FORBIDDEN")
    if profile_before != profile_after:
        violations.append("PATCH_PROFILE_MUTATION_FORBIDDEN")

    unauthorized: tuple[str, ...] = ()
    if struct_before == struct_after:
        patched = {(op.node_id, op.node_property_pointer) for op in patch_set.operations}
        unauthorized = _unauthorized_changes(base, candidate, patched)
        if unauthorized:
            violations.append("PATCH_INVARIANT_VIOLATION")

    return ConservationReport(
        visible_text_hash_before=text_before,
        visible_text_hash_after=text_after,
        structure_projection_hash_before=struct_before,
        structure_projection_hash_after=struct_after,
        reading_order_projection_hash_before=order_before,
        reading_order_projection_hash_after=order_after,
        resource_manifest_hash_before=resources_before,
        resource_manifest_hash_after=resources_after,
        profile_before=profile_before,
        profile_after=profile_after,
        unauthorized_changes=unauthorized,
        violations=tuple(violations),
    )


def apply_patch_set(
    base_ast: dict[str, Any], patch_set: ASTPatchSet
) -> tuple[dict[str, Any], ConservationReport]:
    """RFC006 §24.1 apply, all-or-nothing (§24.2 "no repair").

    Raises `PatchPreconditionError` if the base identity, a node, its
    type, its precondition value, or a replacement's codec does not
    match exactly — never a partial apply. Raises
    `PatchInvariantViolation` if the resulting candidate's Conservation
    Report carries any violation; `base_ast` is never mutated in either
    case.
    """
    if ast_hash(base_ast) != patch_set.base_ast_hash:
        raise PatchPreconditionError(
            f"base ast_hash mismatch: PatchSet expects {patch_set.base_ast_hash}, "
            f"got {ast_hash(base_ast)}"
        )
    indices = [op.operation_index for op in patch_set.operations]
    if indices != list(range(len(indices))):
        raise PatchPreconditionError(f"operation_index must be 0..n-1 without gaps, got {indices}")

    candidate = copy.deepcopy(base_ast)
    nodes_by_id = {n["id"]: n for n in preorder(document_roots(candidate)) if "id" in n}

    touched: set[tuple[str, str]] = set()
    for op in patch_set.operations:
        key = (op.node_id, op.node_property_pointer)
        if key in touched:
            raise PatchPreconditionError(f"duplicate operation on {key!r}")
        touched.add(key)

        node = nodes_by_id.get(op.node_id)
        if node is None:
            raise PatchPreconditionError(f"node {op.node_id!r} not found in base AST")
        if node.get("type") != op.node_type:
            raise PatchPreconditionError(
                f"node {op.node_id!r} has type {node.get('type')!r}, expected {op.node_type!r}"
            )
        if (op.node_type, op.node_property_pointer) not in FAMILY:
            raise PatchPreconditionError(
                f"{(op.node_type, op.node_property_pointer)!r} is not a registered typed property"
            )
        present, current = get_pointer(node, op.node_property_pointer)
        if op.expected_presence != "present" or not present:
            key = (op.node_id, op.node_property_pointer)
            raise PatchPreconditionError(f"{key!r} is not present as the operation expects")
        if current != op.expected_value:
            raise PatchPreconditionError(
                f"{(op.node_id, op.node_property_pointer)!r} precondition mismatch: "
                f"expected {op.expected_value!r}, found {current!r} (stale target)"
            )
        try:
            _validate_replacement(op.node_type, op.node_property_pointer, op.value_type, op.value)
        except InvalidReplacementError as exc:
            raise PatchPreconditionError(str(exc)) from exc
        set_pointer(node, op.node_property_pointer, op.value)

    try:
        candidate = canonicalize(candidate)
    except CanonicalizationError as exc:
        raise PatchPreconditionError(f"candidate failed strict canonicalization: {exc}") from exc

    report = _build_conservation_report(base_ast, candidate, patch_set)
    if report.violations:
        raise PatchInvariantViolation(report)
    return candidate, report

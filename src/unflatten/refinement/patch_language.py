"""RFC006 §20-22 — the closed `set_typed_property` patch language and
`ASTPatchSet`, plus inverse-patch generation.

V0 operation union is exactly one kind (§21.1): `set_typed_property`.
Every other RFC006 operation kind (`set_run_range_property`,
`resegment_run`, and all of §21.3's forbidden mutations) is not
representable by `PatchOperation` at all — there is no `operation_kind`
field to mis-set, closing the union by construction rather than by a
runtime check of a string.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.model.typed_property_pointer_map import enum_domain
from unflatten.refinement.target_catalog import TargetCatalog


class InvalidReplacementError(ValueError):
    """A proposed replacement value fails its property family's codec."""


class DuplicateOperationError(ValueError):
    """Two operations in one `ASTPatchSet` target the same node/pointer."""


class MinimalityError(ValueError):
    """A V0 auto-commit `ASTPatchSet` carries more than one operation
    (§22.4)."""


@dataclass(frozen=True)
class PatchOperation:
    """One `set_typed_property` operation (RFC006 §22.1, simplified)."""

    operation_index: int
    node_id: str
    node_type: str
    node_property_pointer: str
    expected_presence: str
    expected_value: Any
    value_type: str
    value: Any
    target_slot_id: str


@dataclass(frozen=True)
class ASTPatchSet:
    """A sealed, deterministic-plan-builder transaction (RFC006 §22.1)."""

    patch_set_id: str
    patch_set_hash: str
    base_ast_hash: str
    operations: tuple[PatchOperation, ...]


def _validate_replacement(node_type: str, pointer: str, value_type: str, value: Any) -> None:
    if value_type == "boolean":
        if not isinstance(value, bool):
            raise InvalidReplacementError(f"{(node_type, pointer)!r} expects bool, got {value!r}")
    elif value_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidReplacementError(f"{(node_type, pointer)!r} expects int, got {value!r}")
    elif value_type == "enum":
        domain = enum_domain(node_type, pointer)
        if value not in domain:
            raise InvalidReplacementError(
                f"{(node_type, pointer)!r} value {value!r} is not in enum domain {domain!r}"
            )
    else:  # pragma: no cover -- FAMILY only ever assigns the three families above
        raise InvalidReplacementError(f"unsupported value_type {value_type!r}")


def _patch_set_hash(base_ast_hash: str, operations: tuple[PatchOperation, ...]) -> str:
    projection = [
        {
            "operation_index": op.operation_index,
            "node_id": op.node_id,
            "node_type": op.node_type,
            "node_property_pointer": op.node_property_pointer,
            "expected_presence": op.expected_presence,
            "expected_value": op.expected_value,
            "value_type": op.value_type,
            "value": op.value,
        }
        for op in operations
    ]
    payload = (
        b"rfc006.ast-patch-set.v0\0"
        + base_ast_hash.encode("utf-8")
        + b"\0"
        + to_canonical_json(projection)
    )
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def build_patch_set(
    catalog: TargetCatalog,
    proposals: list[tuple[str, Any]],
    *,
    allow_multi: bool = False,
) -> ASTPatchSet:
    """Deterministic Patch Plan Builder (RFC006 §23), restricted to the
    single `set_typed_property` operation kind.

    `proposals` is `[(target_slot_id, replacement_value), ...]`. Raises
    rather than repairs on any invalid proposal (§23 "Builder NESMIE...
    opravovať invalid value"): `UnknownTargetSlotError`,
    `InvalidReplacementError`, `DuplicateOperationError`,
    `MinimalityError`.
    """
    if not proposals:
        raise ValueError("build_patch_set: at least one proposal is required")
    if not allow_multi and len(proposals) != 1:
        raise MinimalityError(
            f"V0 auto-commit PatchSet must contain exactly one operation, got {len(proposals)} "
            "(pass allow_multi=True for an explicit shadow/review PatchSet)"
        )
    seen: set[tuple[str, str]] = set()
    operations: list[PatchOperation] = []
    for index, (target_slot_id, value) in enumerate(proposals):
        entry = catalog.by_slot(target_slot_id)  # raises UnknownTargetSlotError
        key = (entry.node_id, entry.node_property_pointer)
        if key in seen:
            raise DuplicateOperationError(f"duplicate operation on {key!r}")
        seen.add(key)
        _validate_replacement(
            entry.node_type, entry.node_property_pointer, entry.property_family, value
        )
        operations.append(
            PatchOperation(
                operation_index=index,
                node_id=entry.node_id,
                node_type=entry.node_type,
                node_property_pointer=entry.node_property_pointer,
                expected_presence=entry.expected_presence,
                expected_value=entry.expected_value,
                value_type=entry.property_family,
                value=value,
                target_slot_id=target_slot_id,
            )
        )
    ops = tuple(operations)
    patch_set_hash = _patch_set_hash(catalog.base_ast_hash, ops)
    return ASTPatchSet(
        patch_set_id=f"aps_{patch_set_hash.removeprefix('sha256:')}",
        patch_set_hash=patch_set_hash,
        base_ast_hash=catalog.base_ast_hash,
        operations=ops,
    )


def invert_patch_set(patch_set: ASTPatchSet, candidate_ast_hash: str) -> ASTPatchSet:
    """The inverse `ASTPatchSet` that undoes `patch_set` on its candidate.

    Every V0 target admits only currently-present properties
    (`target_catalog` module docstring), so inversion is a pure
    value/expected-value swap — presence never changes.
    """
    inverse_ops = tuple(
        PatchOperation(
            operation_index=op.operation_index,
            node_id=op.node_id,
            node_type=op.node_type,
            node_property_pointer=op.node_property_pointer,
            expected_presence=op.expected_presence,
            expected_value=op.value,
            value_type=op.value_type,
            value=op.expected_value,
            target_slot_id=op.target_slot_id,
        )
        for op in patch_set.operations
    )
    patch_set_hash = _patch_set_hash(candidate_ast_hash, inverse_ops)
    return ASTPatchSet(
        patch_set_id=f"aps_{patch_set_hash.removeprefix('sha256:')}",
        patch_set_hash=patch_set_hash,
        base_ast_hash=candidate_ast_hash,
        operations=inverse_ops,
    )

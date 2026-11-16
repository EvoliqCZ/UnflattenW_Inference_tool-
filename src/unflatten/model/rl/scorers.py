"""Pure structure scorers for the dormant S4.5 reward engine (RFC005-ERRATA-010 E3.4).

Every function is deterministic and returns an ``int`` (micro units,
``0..1_000_000``, or an integer distance) or ``None``. No floats, no
randomness, no I/O, no renderer, no model.

``None`` means *nothing scorable*: an empty reference, or a reference that
the observable/supervision mask removes entirely. It is never ``0`` -- a
missing label is not a negative (ERRATA-010 E3.4, RFC005 s36.4).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Final, NamedTuple

from unflatten.eval.s2_gate.distance import levenshtein

MICRO: Final = 1_000_000

PreorderEntry = tuple[int, str]


class GridCell(NamedTuple):
    """One table cell identity; ``observable`` is the reference-side mask."""

    row: int
    col: int
    rowspan: int
    colspan: int
    observable: bool = True


def _f1_micro(true_positive: int, reference_size: int, candidate_size: int) -> int:
    return 2 * true_positive * MICRO // (reference_size + candidate_size)


def _multiset_f1_micro(reference: Sequence[Any], candidate: Sequence[Any]) -> int | None:
    if not reference:
        return None
    overlap = Counter(reference) & Counter(candidate)
    return _f1_micro(sum(overlap.values()), len(reference), len(candidate))


def tree_node_f1_micro(
    reference_preorder: Sequence[PreorderEntry], candidate_preorder: Sequence[PreorderEntry]
) -> int | None:
    """Multiset F1 over ``(depth, node_type)`` preorder tuples."""
    return _multiset_f1_micro(
        [tuple(entry) for entry in reference_preorder],
        [tuple(entry) for entry in candidate_preorder],
    )


def tree_edit_distance(
    reference_preorder: Sequence[PreorderEntry], candidate_preorder: Sequence[PreorderEntry]
) -> int | None:
    """Sequence edit distance over the preorder; ``None`` for an empty reference."""
    if not reference_preorder:
        return None
    return levenshtein(
        [tuple(entry) for entry in reference_preorder],
        [tuple(entry) for entry in candidate_preorder],
    )


def _footprint(cell: GridCell) -> set[tuple[int, int]]:
    return {
        (cell.row + r, cell.col + c)
        for r in range(max(cell.rowspan, 1))
        for c in range(max(cell.colspan, 1))
    }


def table_grid_f1_micro(
    reference_cells: Sequence[GridCell], candidate_cells: Sequence[GridCell]
) -> int | None:
    """F1 over ``(row, col, rowspan, colspan)`` cell identities, ownership-aware.

    Reference cells with ``observable=False`` are excluded, and so is every
    candidate cell anchored inside such a cell's footprint (it is not a false
    positive: the reference says nothing about that area). A merged-span
    mismatch is a different identity and is penalised.
    """
    hidden: set[tuple[int, int]] = set()
    for cell in reference_cells:
        if not cell.observable:
            hidden |= _footprint(cell)
    reference = [tuple(c[:4]) for c in reference_cells if c.observable]
    candidate = [tuple(c[:4]) for c in candidate_cells if (c[0], c[1]) not in hidden]
    return _multiset_f1_micro(reference, candidate)


def _same_value(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def typed_value_score_micro(
    reference_typed: Mapping[str, Mapping[str, Any]],
    candidate_typed: Mapping[str, Mapping[str, Any]],
    observable_mask: Collection[tuple[str, str]],
) -> int | None:
    """Share of observable reference ``(node_id, pointer)`` values the candidate matches.

    Scored set = reference pointers that are also in ``observable_mask``.
    A missing candidate pointer is a miss; candidate pointers outside the
    scored set are ignored. Nothing scorable => ``None``.
    """
    scored = [
        (node_id, pointer, value)
        for node_id, properties in reference_typed.items()
        for pointer, value in properties.items()
        if (node_id, pointer) in observable_mask
    ]
    if not scored:
        return None
    sentinel = object()
    hits = sum(
        1
        for node_id, pointer, value in scored
        if _same_value(candidate_typed.get(node_id, {}).get(pointer, sentinel), value)
    )
    return hits * MICRO // len(scored)

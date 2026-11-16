"""Deterministic candidate<->gold node alignment (design D1, PROPOSED).

``alignment_policy = "preorder-edit-script.v0"``.

Typed values and grid cells used to be joined by node id, but the decoder's
``local_node_id`` space and the bundle's id space differ, so most joins were
silent misses. This module aligns the two ``(depth, node_type)`` preorders with
a Levenshtein DP *plus a backtrace* (``eval/s2_gate/distance.levenshtein``
returns only the distance) and maps candidate node ids to gold node ids.

Pure functions, ints only, no I/O, no set/dict-order dependence.

Tie-break (pinned, part of the policy)
--------------------------------------
Minimal edit scripts are not unique, so the reward would not be deterministic.
The backtrace starts at the *end* of both sequences, ``(i, j)`` = (len(gold),
len(candidate)), and at every cell takes the FIRST applicable move in
``TIE_BREAK_ORDER``::

    match            gold[i-1] == cand[j-1] and D[i][j] == D[i-1][j-1]
    substitute       gold[i-1] != cand[j-1] and D[i][j] == D[i-1][j-1] + 1
    delete_gold      D[i][j] == D[i-1][j] + 1      (gold node left unmatched)
    insert_candidate D[i][j] == D[i][j-1] + 1      (candidate node invented)

Consequence of starting at the end: among identical ``(depth, type)`` runs
(empty paragraphs, the dominant real failure) the LAST candidate node pairs
with the LAST gold node, and surplus nodes fall at the front of the run.
``match`` and ``substitute`` are mutually exclusive (equal vs unequal tuples),
so their relative order cannot change a result; it is still pinned.

Typed transfer: only ``match`` pairs (equal ``(depth, type)``) map candidate id
to gold id. A ``substitute`` pair is aligned for the tree distance but is a
different node kind, so no typed value crosses it. Gold nodes without a match
(deleted or substituted) stay in the scorer's denominator as misses.

Tables: grid cells are keyed by table ordinal (``grid_cells_from_ast``), so a
candidate that shifted a table within the page keeps its keys. A candidate table
the gold does not have produces no gold cells => grid ``None`` (unobservable),
never ``0``; the penalty for the invented table is carried by the tree
component. Do not "fix" that to a numeric zero.
"""

from __future__ import annotations

from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from unflatten.model.rl.reward_inputs import _CHILD_KEYS
from unflatten.model.rl.scorers import GridCell

ALIGNMENT_POLICY: Final = "preorder-edit-script.v0"

OP_MATCH: Final = "match"
OP_SUBSTITUTE: Final = "substitute"
OP_DELETE_GOLD: Final = "delete_gold"
OP_INSERT_CANDIDATE: Final = "insert_candidate"

#: Backtrace priority, first applicable move wins. Pinned by the policy.
TIE_BREAK_ORDER: Final = (OP_MATCH, OP_SUBSTITUTE, OP_DELETE_GOLD, OP_INSERT_CANDIDATE)

PreorderEntry = tuple[int, str]
#: ``(op, gold_index | None, candidate_index | None)``
EditStep = tuple[str, int | None, int | None]


@dataclass(frozen=True, slots=True)
class Alignment:
    """Edit script over two preorders (forward order) and its derived views."""

    gold_length: int
    candidate_length: int
    total_cost: int
    script: tuple[EditStep, ...]

    def candidate_to_gold(self) -> dict[int, int]:
        """Candidate index -> gold index for ``match`` pairs only."""
        return {
            c: g for op, g, c in self.script if op == OP_MATCH and c is not None and g is not None
        }

    def substituted(self) -> tuple[tuple[int, int], ...]:
        """``(candidate, gold)`` pairs aligned across different kinds (no typed transfer)."""
        return tuple(
            (c, g)
            for op, g, c in self.script
            if op == OP_SUBSTITUTE and c is not None and g is not None
        )

    def unmatched_gold(self) -> tuple[int, ...]:
        """Gold indexes that received no ``match`` (deleted or substituted); misses."""
        matched = set(self.candidate_to_gold().values())
        return tuple(g for g in range(self.gold_length) if g not in matched)

    def unaligned_candidate(self) -> tuple[int, ...]:
        """Candidate indexes with no gold counterpart at all (invented nodes)."""
        return tuple(c for op, _g, c in self.script if op == OP_INSERT_CANDIDATE and c is not None)


def _distance_rows(gold: Sequence[PreorderEntry], cand: Sequence[PreorderEntry]) -> list[array]:
    rows = [array("I", range(len(cand) + 1))]
    for i, g in enumerate(gold, start=1):
        previous = rows[-1]
        current = array("I", [0]) * (len(cand) + 1)
        current[0] = i
        for j, c in enumerate(cand, start=1):
            current[j] = min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (0 if g == c else 1),
            )
        rows.append(current)
    return rows


def _applicable(op: str, rows: list[array], i: int, j: int, same: bool) -> bool:
    here = rows[i][j]
    if op == OP_MATCH:
        return same and here == rows[i - 1][j - 1]
    if op == OP_SUBSTITUTE:
        return (not same) and here == rows[i - 1][j - 1] + 1
    if op == OP_DELETE_GOLD:
        return here == rows[i - 1][j] + 1
    return here == rows[i][j - 1] + 1


def align_preorder(
    reference_preorder: Sequence[PreorderEntry], candidate_preorder: Sequence[PreorderEntry]
) -> Alignment:
    """Levenshtein DP with backtrace under the pinned tie-break (module docstring)."""
    gold: list[PreorderEntry] = [(e[0], e[1]) for e in reference_preorder]
    cand: list[PreorderEntry] = [(e[0], e[1]) for e in candidate_preorder]
    rows = _distance_rows(gold, cand)
    i, j = len(gold), len(cand)
    steps: list[EditStep] = []
    while i > 0 or j > 0:
        if i == 0:
            steps.append((OP_INSERT_CANDIDATE, None, j - 1))
            j -= 1
            continue
        if j == 0:
            steps.append((OP_DELETE_GOLD, i - 1, None))
            i -= 1
            continue
        same = gold[i - 1] == cand[j - 1]
        for op in TIE_BREAK_ORDER:
            if _applicable(op, rows, i, j, same):
                break
        else:  # pragma: no cover - the DP guarantees one move applies
            raise AssertionError(f"no edit move applies at ({i}, {j})")
        if op in (OP_MATCH, OP_SUBSTITUTE):
            steps.append((op, i - 1, j - 1))
            i, j = i - 1, j - 1
        elif op == OP_DELETE_GOLD:
            steps.append((op, i - 1, None))
            i -= 1
        else:
            steps.append((op, None, j - 1))
            j -= 1
    steps.reverse()
    return Alignment(len(gold), len(cand), rows[-1][-1], tuple(steps))


def map_candidate_node_ids(
    alignment: Alignment, candidate_node_ids: Sequence[str], gold_node_ids: Sequence[str]
) -> dict[str, str]:
    """Candidate node id -> gold node id for ``match`` pairs.

    Both id sequences must be parallel to the preorders that were aligned.
    """
    if len(candidate_node_ids) != alignment.candidate_length:
        raise ValueError(
            f"candidate_node_ids has {len(candidate_node_ids)} entries, "
            f"alignment covers {alignment.candidate_length}"
        )
    if len(gold_node_ids) != alignment.gold_length:
        raise ValueError(
            f"gold_node_ids has {len(gold_node_ids)} entries, "
            f"alignment covers {alignment.gold_length}"
        )
    return {
        candidate_node_ids[c]: gold_node_ids[g]
        for c, g in sorted(alignment.candidate_to_gold().items())
    }


def node_ids_from_ast(ast: Mapping[str, Any]) -> tuple[str, ...]:
    """Node ids in the same preorder as ``reward_inputs.preorder_from_ast``."""
    out: list[str] = []

    def walk(node: Mapping[str, Any]) -> None:
        out.append(node["id"])
        for key in _CHILD_KEYS.get(node["type"], ()):
            for child in node.get(key, ()):
                walk(child)

    for section in ast.get("sections", ()):
        walk(section)
    return tuple(out)


def rekey_typed(
    candidate_typed: Mapping[str, Mapping[str, Any]], id_map: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    """Move candidate typed values onto gold node ids; unmapped candidates drop.

    A candidate id with no gold counterpart contributes nothing: it is not a
    hit and gold's pointers stay in the scorer's denominator as misses.
    """
    return {id_map[nid]: dict(props) for nid, props in candidate_typed.items() if nid in id_map}


def align_grid_cells(
    gold_cells: Sequence[GridCell], candidate_cells: Sequence[GridCell]
) -> tuple[GridCell, ...] | None:
    """Candidate grid cells for scoring, or ``None`` when gold has no table.

    Both sides are keyed by table ordinal and ``(row, col)``, so cells already
    share a key space. ``None`` (not ``()``/``0``) marks "gold has no grid":
    the scorer then reports ``None`` (unobservable) and the invented table is
    penalised by the tree component only.
    """
    if not gold_cells:
        return None
    return tuple(candidate_cells)

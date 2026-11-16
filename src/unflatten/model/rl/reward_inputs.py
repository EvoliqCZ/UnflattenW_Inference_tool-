"""Input records and AST/reference adapters for the S4.5 reward engine.

Everything here is host-side scorer data (RFC005-ERRATA-010 E3.1/E3.5): the
gold reference and the candidate record are plain frozen dataclasses, built by
the caller from already-available artefacts (``DocumentReference``,
``ast_builder.BuildResult``). Nothing here renders, decodes, or loads a model.

Adapters map the existing shapes onto the T3 scorer inputs:

* ``DocumentReference.structure`` is already ``(depth, node_type)`` preorder;
* a candidate AST is walked into the same preorder (stories excluded, exactly
  as ``DocumentReference`` excludes them);
* table cells become ``GridCell`` with a per-table row stride;
* ``DocumentReference.typed`` carries only supervised pointers, so its
  ``(node_id, pointer)`` set *is* the observable mask.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from unflatten.model.rl.scorers import GridCell

#: Rows of different tables never share a row index.
TABLE_ROW_STRIDE: Final = 1_000_000
#: Outcome of the deterministic RFC002 AST->DOCX render of one candidate.
#: ``not_run`` is a missing observation: never a match, never a scalar.
RFC002_RENDER_STATES: Final = ("rendered", "failed", "not_run")

_CHILD_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "section": ("children",),
    "table": ("rows",),
    "row": ("cells",),
    "cell": ("children",),
    "list": ("items",),
    "list_item": ("children",),
    "paragraph": ("children",),
}


class StrictBuild(Protocol):
    """The part of ``ast_builder.BuildResult`` the engine reads."""

    status: str
    ast: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class GoldReference:
    """Gold authority for one document (scorer-only)."""

    document_id: str
    visible_text: str
    control_event_counts: Mapping[str, int]
    structure: tuple[tuple[int, str], ...]
    typed: Mapping[str, Mapping[str, Any]]
    typed_observable: frozenset[tuple[str, str]]
    grid_cells: tuple[GridCell, ...]


@dataclass(frozen=True, slots=True)
class CandidateInputs:
    """One rollout candidate, as the engine sees it.

    ``build_candidate`` returns the strict-build outcome (valid AST or typed
    failure). ``visible_text=None`` derives the text from the built AST, so
    text dropped *after* decode is still seen; a failed build then has no
    committed text. ``fragment_violations`` are resource/fragment integrity
    findings produced upstream (assembler/binder), by name.

    ``rfc002_render`` is the deterministic RFC002 renderer's verdict on the
    built AST (``rendered`` | ``failed`` | ``not_run``); it has no default, so
    a caller must say it did not run. LibreOffice open results are not an input.
    """

    candidate_id: str
    chain_flags: Mapping[str, bool]
    build_candidate: Callable[[], StrictBuild]
    control_event_counts: Mapping[str, int]
    rfc002_render: str
    visible_text: str | None = None
    typed: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    fragment_violations: tuple[str, ...] = ()
    continuity_micro: int | None = None


def preorder_from_ast(ast: Mapping[str, Any]) -> tuple[tuple[int, str], ...]:
    """``(depth, node_type)`` preorder of the body; sections at depth 0."""
    out: list[tuple[int, str]] = []

    def walk(node: Mapping[str, Any], depth: int) -> None:
        out.append((depth, node["type"]))
        for key in _CHILD_KEYS.get(node["type"], ()):
            for child in node.get(key, ()):
                walk(child, depth + 1)

    for section in ast.get("sections", ()):
        walk(section, 0)
    return tuple(out)


def ast_run_text(ast: Mapping[str, Any]) -> str:
    """Concatenated run text in body preorder (same unit as ``reference.text``)."""
    parts: list[str] = []

    def walk(node: Mapping[str, Any]) -> None:
        if node["type"] == "run":
            parts.append(node.get("text") or "")
        for key in _CHILD_KEYS.get(node["type"], ()):
            for child in node.get(key, ()):
                walk(child)

    for section in ast.get("sections", ()):
        walk(section)
    return "".join(parts)


def grid_cells_from_ast(
    ast: Mapping[str, Any], *, observable_ids: frozenset[str] | None = None
) -> tuple[GridCell, ...]:
    """Cells of every table; ``observable_ids`` marks cells outside it unobservable."""
    cells: list[GridCell] = []
    table_ordinal = 0

    def walk(node: Mapping[str, Any]) -> None:
        nonlocal table_ordinal
        if node["type"] == "table":
            base = table_ordinal * TABLE_ROW_STRIDE
            table_ordinal += 1
            rows: Sequence[Mapping[str, Any]] = node.get("rows", ())
            for row_index, row in enumerate(rows):
                for cell in row.get("cells", ()):
                    cells.append(
                        GridCell(
                            base + row_index,
                            cell["grid_column"],
                            cell["row_span"],
                            cell["col_span"],
                            observable_ids is None or cell.get("id") in observable_ids,
                        )
                    )
        for key in _CHILD_KEYS.get(node["type"], ()):
            for child in node.get(key, ()):
                walk(child)

    for section in ast.get("sections", ()):
        walk(section)
    return tuple(cells)


def gold_from_document_reference(
    reference: Any,
    *,
    document_id: str,
    control_event_counts: Mapping[str, int],
    visible_text: str | None = None,
    gold_ast: Mapping[str, Any] | None = None,
) -> GoldReference:
    """Adapt ``eval.assembly.reference.DocumentReference`` to ``GoldReference``.

    ``visible_text`` is the gold text authority (e.g. from
    ``s2_gate.score.assemble_reference_text``); when omitted it is the join of
    ``reference.text`` in document order. ``gold_ast`` supplies table grids.
    """
    text = "".join(reference.text.values()) if visible_text is None else visible_text
    typed = {nid: dict(props) for nid, props in reference.typed.items()}
    observable = frozenset((nid, ptr) for nid, props in typed.items() for ptr in props)
    grid = (
        ()
        if gold_ast is None
        else grid_cells_from_ast(gold_ast, observable_ids=frozenset(reference.node_ids))
    )
    return GoldReference(
        document_id=document_id,
        visible_text=text,
        control_event_counts=dict(control_event_counts),
        structure=tuple(tuple(e) for e in reference.structure),
        typed=typed,
        typed_observable=observable,
        grid_cells=grid,
    )

"""Atomic, exact v16 recovery of complete cells with note/page-break forms."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import unflatten.dataset.observations.aligner as a
from unflatten.dataset.observations.residual import (
    ResidualResult,
    _corridor_bounds,
    _find_page_split_runs,
    _find_runs,
    _find_tier2_runs,
    _is_reading_order,
    _marker_key,
    _target_forms,
)

Candidate = list[
    tuple[a._Paragraph, list[tuple[str, int, int]], list[a._StreamWord], list[a._StreamWord]]
]


def _candidate(
    paragraphs: list[a._Paragraph],
    stream: list[a._StreamWord],
) -> Candidate | None:
    """Refuse incomplete or ambiguous cells; no independent short-text claims."""
    located = []
    cursor = 0
    special = False
    for paragraph in paragraphs:
        if not paragraph.text.strip():
            continue
        choices = {}
        for key, keyed in _target_forms(paragraph, doubled_markers=True):
            runs = _find_runs(key, stream[cursor:])
            if not runs:
                runs = _find_page_split_runs(key, stream[cursor:])
            for run in runs:
                if _is_reading_order(run, same_line=True):
                    choices[tuple(id(word) for word in run)] = (keyed, run, [])
        if not choices and paragraph.note_refs:
            keyed = a._keyed_words(paragraph)
            keys = frozenset(_marker_key(marker) for _offset, marker in paragraph.note_refs)
            for real, markers in _find_tier2_runs(keyed, keys, stream[cursor:], same_line=True):
                if markers:
                    choices[tuple(id(word) for word in real)] = (keyed, real, markers)
        if len(choices) != 1:
            return None
        keyed, run, markers = next(iter(choices.values()))
        special |= bool(markers) or any(start == end for _key, start, end in keyed)
        special |= len({word.page_index for word in run}) > 1
        if paragraph.matched and {id(w) for w in paragraph.spelled_words} != {id(w) for w in run}:
            return None
        located.append((paragraph, keyed, run, markers))
        cursor = next(i for i in range(cursor, len(stream)) if stream[i] is run[-1]) + 1
    words = [word for _p, _k, real, _markers in located for word in real]
    return located if located and special and _is_reading_order(words, same_line=True) else None


def _grid_valid(table_paragraphs: list[a._Paragraph], candidate: Candidate) -> bool:
    """Use the existing row/column topology check on exactly spelled words."""
    by_cell = defaultdict(list)
    proposed = {p.node_id: words for p, _keyed, words, _markers in candidate}
    for paragraph in table_paragraphs:
        words = proposed.get(paragraph.node_id, paragraph.spelled_words)
        if words:
            by_cell[paragraph.cell_id].append((paragraph, words))
    cells, assignment = [], {}
    for key, entries in enumerate(by_cell.values()):
        paragraph = entries[0][0]
        cells.append(
            a._BlockCell(
                key,
                [],
                paragraph.row_index,
                paragraph.row_index + paragraph.row_span,
                paragraph.grid_column,
                paragraph.grid_column + paragraph.col_span,
                [[(word.key, 0, 0) for word in words] for _p, words in entries],
            )
        )
        assignment[key] = [word for _p, words in entries for word in words]
    return a._rows_in_document_order(cells, assignment)


def _prepare(
    paragraphs: list[a._Paragraph],
    stream: list[a._StreamWord],
    owners: dict[int, str],
    is_excluded: Callable[[a._StreamWord], bool],
) -> list[Candidate]:
    """Locate complete cells without changing any ownership."""
    ordered = sorted(paragraphs, key=lambda paragraph: paragraph.preorder)
    positions = {id(word): i for i, word in enumerate(stream)}
    by_cell = defaultdict(list)
    for paragraph in ordered:
        if paragraph.table_id is not None:
            by_cell[(paragraph.table_id, paragraph.cell_id)].append(paragraph)
    pending = []
    for (table_id, _cell_id), group in by_cell.items():
        if all(p.matched or not p.text.strip() for p in group) or any(p.has_field for p in group):
            continue
        lo, _ = _corridor_bounds(ordered, ordered.index(group[0]), positions, len(stream))
        _, hi = _corridor_bounds(ordered, ordered.index(group[-1]), positions, len(stream))
        table = [p for p in ordered if p.table_id == table_id]
        width = max(p.grid_column + p.col_span for p in table)
        if group[0].col_span < width and not any(
            p.matched
            and p.cell_id != group[0].cell_id
            and (p.row_index == group[0].row_index or p.grid_column == group[0].grid_column)
            for p in table
        ):
            continue
        candidate = _candidate(
            group,
            [
                word
                for word in stream[lo:hi]
                if not is_excluded(word)
                and owners.get(id(word)) in {None, *(p.node_id for p in group)}
            ],
        )
        if candidate is None:
            continue
        own = {id(w) for p in group for w in p.spelled_words}
        if any(
            id(w) in owners and id(w) not in own
            for _p, _k, words, markers in candidate
            for w in (*words, *markers)
        ):
            continue
        if _grid_valid(table, candidate):
            pending.append(candidate)
    return pending


def recover_complete_cells(
    paragraphs: list[a._Paragraph],
    stream: list[a._StreamWord],
    owners: dict[int, str],
    *,
    is_excluded: Callable[[a._StreamWord], bool] = lambda _word: False,
    result: ResidualResult | None = None,
    strict: bool = False,
) -> list[a._StreamWord]:
    """Return words recovered only after complete-cell equality and grid checks."""
    pending = _prepare(paragraphs, stream, owners, is_excluded)
    by_table = defaultdict(list)
    for candidate in pending:
        by_table[candidate[0][0].table_id].extend(candidate)
    invalid_tables = {
        table_id
        for table_id, candidate in by_table.items()
        if not _grid_valid([p for p in paragraphs if p.table_id == table_id], candidate)
    }
    claimants = defaultdict(int)
    for candidate in pending:
        for wid in {id(w) for _p, _k, words, markers in candidate for w in (*words, *markers)}:
            claimants[wid] += 1
    recovered = []
    for candidate in pending:
        if candidate[0][0].table_id in invalid_tables:
            continue
        if any(
            claimants[id(w)] > 1 for _p, _k, words, markers in candidate for w in (*words, *markers)
        ):
            continue
        for paragraph, keyed, words, markers in candidate:
            if not paragraph.matched:
                if not a.claim_single_owner(
                    paragraph.node_id, [*words, *markers], owners, strict=strict
                ):
                    if result is not None:
                        result.invariant_skips += 1
                    continue
                a._consume(paragraph, keyed, words)
                paragraph.spelled_words.extend(markers)
                if result is not None:
                    result.claimed.append(paragraph.node_id)
                    result.claimed_words.extend((*words, *markers))
                    result.note_marker_matches += bool(markers) or any(
                        start == end for _key, start, end in keyed
                    )
                recovered.extend((*words, *markers))
    return recovered

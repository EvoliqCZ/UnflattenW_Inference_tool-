"""The seam engine: page hypotheses -> one document hypothesis (assembler
design §5, §6, §5.4).

For each pair of ADJACENT available pages the engine walks the right
spine of page k against the left spine of page k+1 (last section vs
first section, last block vs first block, then recursively into every
joined pair). Different node types never join; a known join-key pointer
that differs forbids a join; a join needs its rule's positive predicate,
and unknown is never evidence. Nothing joins across a missing page.

Headers and footers are lifted out of their sections into
`AssembledDocumentHypothesis.stories`, one entry per page and story, so
a joined section never receives a later page's story after its body;
deriving the section's story kinds is `stories.py` (design §5.8).

Every result passes `check_invariants` (text conservation, structural
ownership, grammar replay) or `assemble` raises
`AssemblyInvariantViolation` -- a violation is never repaired.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from unflatten.assembly import policy as pol
from unflatten.assembly import stories as st
from unflatten.assembly.hypothesis import (
    PageHypothesis,
    PageInput,
    PageNode,
    TypedValue,
    rule_view,
)
from unflatten.model import events_vocab, grammar, piece_withdrawal

ASSEMBLER_PROPERTY_CONFLICT = "ASSEMBLER_PROPERTY_CONFLICT"

_STORY_TYPES = frozenset({"header", "footer"})
_RULE_BY_TYPE: dict[str, str] = {
    "section": "SEC",
    "paragraph": "PARA",
    "table": "TABLE",
    "row": "ROW",
    "list": "LIST",
    "list_item": "ITEM",
    "run": "RUN",
}
_GRID_COLUMN = "/grid_column"

type PieceRef = tuple[int, str]  # (page_ordinal, page-local id)


class AssemblyInvariantViolation(RuntimeError):
    """An assembled document broke a conservation invariant (design §5.4).
    `check` names the invariant: `ownership`, `text` or `grammar`."""

    def __init__(self, check: str, message: str) -> None:
        super().__init__(f"{check}: {message}")
        self.check = check


@dataclass(frozen=True, slots=True)
class AssembledNode:
    assembled_id: str  # "a<k>" in assembled preorder
    node_type: str
    children: tuple[AssembledNode, ...]
    text: str | None  # run only
    typed: Mapping[str, TypedValue]  # resolved; absent = unresolved
    sources: tuple[PieceRef, ...]  # the page pieces this node owns, page order


@dataclass(frozen=True, slots=True)
class PageStory:
    """One header/footer as page `page_ordinal` showed it, lifted out of
    the assembled section `section_id`."""

    page_ordinal: int
    section_id: str
    node: AssembledNode


@dataclass(frozen=True, slots=True)
class SeamDecision:
    """One rule decision at seam (k, k+1): the page-k piece `left` against
    the page-(k+1) piece `right`, both page-local ids."""

    seam: tuple[int, int]
    rule: str
    level: str  # node type
    left: str
    right: str
    joined: bool


@dataclass(frozen=True, slots=True)
class PropertyConflict:
    assembled_id: str
    pointer: str
    values: tuple[tuple[int, TypedValue], ...]  # (page_ordinal, value) per piece
    issue: str = ASSEMBLER_PROPERTY_CONFLICT


@dataclass(frozen=True, slots=True)
class AssembledDocumentHypothesis:
    """RFC004 §32.2 step 9 Document Hypothesis (not
    `prediction_ir.DocumentHypothesis`, a fragments wrapper)."""

    pages: tuple[PageInput, ...]
    sections: tuple[AssembledNode, ...]
    notes: tuple[AssembledNode, ...]
    stories: tuple[PageStory, ...]
    story_sets: tuple[st.StorySet, ...]  # per section and story type, design §5.8
    seam_decisions: tuple[SeamDecision, ...]
    conflicts: tuple[PropertyConflict, ...]


# ---------------------------------------------------------------------------
# Working tree
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class _Work:
    node_type: str
    pieces: list[tuple[int, PageNode]]
    children: list[_Work] = field(default_factory=list)
    #: A row appended from a middle/last table piece (design §6).
    continuation_row: bool = False

    @property
    def last_page(self) -> int:
        return self.pieces[-1][0]


@dataclass
class _Context:
    policy: pol.JoinPolicy
    inputs: dict[int, PageInput]
    decisions: list[SeamDecision] = field(default_factory=list)
    #: (page_ordinal, section it was lifted from, story subtree)
    stories: list[tuple[int, _Work, _Work]] = field(default_factory=list)


def _convert(ordinal: int, node: PageNode, ctx: _Context) -> _Work:
    work = _Work(node.node_type, [(ordinal, node)])
    children = _body_children(ordinal, node, ctx, work)
    work.children = [_convert(ordinal, child, ctx) for child in children]
    return work


def _body_children(
    ordinal: int, node: PageNode, ctx: _Context, section: _Work
) -> tuple[PageNode, ...]:
    """A section's children minus its stories, which are recorded against
    `section`; any other node's children unchanged."""
    if node.node_type != "section":
        return node.children
    for child in node.children:
        if child.node_type in _STORY_TYPES:
            ctx.stories.append((ordinal, section, _convert(ordinal, child, ctx)))
    return tuple(child for child in node.children if child.node_type not in _STORY_TYPES)


# ---------------------------------------------------------------------------
# Seam engine
# ---------------------------------------------------------------------------


def assemble(
    pages: Sequence[PageHypothesis], policy: pol.JoinPolicy, *, profile: str
) -> AssembledDocumentHypothesis:
    """Assemble `pages` (strictly increasing `page_ordinal`; a gap is a
    hole) under `policy`, then check the §5.4 invariants under the AST
    `profile` the pages were decoded with."""
    ordinals = [page.input.page_ordinal for page in pages]
    if any(b <= a for a, b in zip(ordinals, ordinals[1:], strict=False)):
        raise ValueError(f"page ordinals must be strictly increasing, got {ordinals}")
    ctx = _Context(policy=policy, inputs={page.input.page_ordinal: page.input for page in pages})
    sections: list[_Work] = []
    notes: list[_Work] = []
    previous: int | None = None
    for page in pages:
        ordinal = page.input.page_ordinal
        seam = (previous, ordinal) if previous is not None and ordinal == previous + 1 else None
        _join_lists(sections, page.sections, ordinal, seam, ctx)
        notes.extend(_convert(ordinal, note, ctx) for note in page.notes)
        previous = ordinal
    document = _finalize(pages, sections, notes, ctx)
    check_invariants(pages, document, profile=profile)
    return document


def _join_lists(
    left: list[_Work],
    right: Sequence[PageNode],
    ordinal: int,
    seam: tuple[int, int] | None,
    ctx: _Context,
) -> None:
    """Append page `ordinal`'s nodes `right` to `left`, first trying to
    join `right[0]` into `left[-1]` when `seam` exists and `left[-1]`
    ends on the seam's page k."""
    rest = right
    if seam is not None and left and right and left[-1].last_page == seam[0]:
        if _try_join(left[-1], right[0], seam, ctx):
            rest = right[1:]
    left.extend(_convert(ordinal, node, ctx) for node in rest)


def _try_join(a: _Work, b: PageNode, seam: tuple[int, int], ctx: _Context) -> bool:
    if not _decide(a, b, seam, ctx):
        return False
    a.pieces.append((seam[1], b))
    if a.node_type == "row":
        _join_cells(a, b, seam, ctx)
    elif a.node_type == "table":
        before = len(a.children)
        _join_lists(a.children, b.children, seam[1], seam, ctx)
        for row in a.children[before:]:
            row.continuation_row = True
    else:
        _join_lists(a.children, _body_children(seam[1], b, ctx, a), seam[1], seam, ctx)
    return True


def _decide(a: _Work, b: PageNode, seam: tuple[int, int], ctx: _Context) -> bool:
    if a.node_type != b.node_type:
        return False
    rule = _RULE_BY_TYPE.get(b.node_type)
    if rule is None:
        return False
    piece = a.pieces[-1][1]
    left, right = rule_view(piece), rule_view(b)
    joined = (
        rule in ctx.policy.enabled
        and not _keys_differ(left.typed, right.typed, pol.JOIN_KEYS[rule])
        and _positive(rule, piece, b, seam, ctx)
    )
    ctx.decisions.append(SeamDecision(seam, rule, b.node_type, piece.local_id, b.local_id, joined))
    return joined


def _positive(rule: str, a: PageNode, b: PageNode, seam: tuple[int, int], ctx: _Context) -> bool:
    if rule == "SEC":
        k, k1 = ctx.inputs[seam[0]], ctx.inputs[seam[1]]
        tolerance = pol.PAGE_SIZE_TOLERANCE_PX
        return (
            abs(k.width_px - k1.width_px) <= tolerance
            and abs(k.height_px - k1.height_px) <= tolerance
        )
    if rule == "PARA":
        return _para_joins(a, b, ctx)
    if rule == "ITEM":
        return (
            bool(a.children and b.children)
            and a.children[-1].node_type == b.children[0].node_type == "paragraph"
            and _para_joins(a.children[-1], b.children[0], ctx)
        )
    if rule == "ROW":
        predicate = ctx.policy.row
        return (
            predicate is not None
            and predicate(rule_view(a), rule_view(b))
            and _cell_columns(a) is not None
            and _cell_columns(b) is not None
        )
    return True  # TABLE, LIST, RUN: adjacency, gated by the keys alone


def _para_joins(a: PageNode, b: PageNode, ctx: _Context) -> bool:
    predicate = ctx.policy.para
    if "PARA" not in ctx.policy.enabled or predicate is None:
        return False
    left, right = rule_view(a), rule_view(b)
    return not _keys_differ(left.typed, right.typed, pol.JOIN_KEYS["PARA"]) and predicate(
        left, right
    )


def _index_free(pointer: str) -> str:
    return re.sub(r"/\d+(?=/|$)", "/*", pointer)


def _key_matches(pointer: str, patterns: Sequence[str]) -> bool:
    pointer = _index_free(pointer)
    for pattern in patterns:
        if pattern == "*" or pointer == pattern:
            return True
        if pattern.endswith("/*") and pointer.startswith(pattern[:-1]):
            return True
    return False


def _keys_differ(
    left: Mapping[str, TypedValue], right: Mapping[str, TypedValue], patterns: Sequence[str]
) -> bool:
    return any(
        _key_matches(pointer, patterns) and not _same(left[pointer], right[pointer])
        for pointer in left.keys() & right.keys()
    )


def _same(a: TypedValue, b: TypedValue) -> bool:
    return type(a) is type(b) and a == b  # True != 1


def _cell_columns(row: PageNode) -> list[int] | None:
    """The row's cells' `/grid_column`s, or `None` when any is unknown or
    repeated: cells that cannot be paired are never evidence."""
    columns: list[int] = []
    for cell in row.children:
        column = rule_view(cell).typed.get(_GRID_COLUMN)
        if cell.node_type != "cell" or not isinstance(column, int) or isinstance(column, bool):
            return None
        columns.append(column)
    return columns if len(set(columns)) == len(columns) else None


def _join_cells(a: _Work, b: PageNode, seam: tuple[int, int], ctx: _Context) -> None:
    """Pair the cells of a joined row by `/grid_column` (union over grid
    columns, design §5.5); a paired cell joins and recurses into its
    block seam, an unpaired one is kept as is."""
    by_column = {_grid_column(cell): cell for cell in a.children}
    for cell in b.children:
        column = _grid_column_of(cell)
        partner = by_column.get(column)
        if partner is None:
            by_column[column] = _convert(seam[1], cell, ctx)
            continue
        left = partner.pieces[-1][1]
        ctx.decisions.append(SeamDecision(seam, "ROW", "cell", left.local_id, cell.local_id, True))
        partner.pieces.append((seam[1], cell))
        _join_lists(partner.children, cell.children, seam[1], seam, ctx)
    a.children = [by_column[column] for column in sorted(by_column)]


def _grid_column(cell: _Work) -> int:
    return _grid_column_of(cell.pieces[0][1])


def _grid_column_of(cell: PageNode) -> int:
    column = cell.typed[_GRID_COLUMN]
    assert isinstance(column, int)  # checked by `_cell_columns` before the join
    return column


# ---------------------------------------------------------------------------
# Finalize: ids, text, property resolution (design §6)
# ---------------------------------------------------------------------------


def _finalize(
    pages: Sequence[PageHypothesis], sections: list[_Work], notes: list[_Work], ctx: _Context
) -> AssembledDocumentHypothesis:
    counter = itertools.count()
    conflicts: list[PropertyConflict] = []
    ids: dict[int, str] = {}

    def build(work: _Work) -> AssembledNode:
        assembled_id = f"a{next(counter)}"
        ids[id(work)] = assembled_id
        typed = _resolve(work, assembled_id, conflicts)
        children = tuple(build(child) for child in work.children)
        return AssembledNode(
            assembled_id=assembled_id,
            node_type=work.node_type,
            children=children,
            text="".join(piece.text or "" for _, piece in work.pieces)
            if work.node_type == "run"
            else None,
            typed=MappingProxyType(typed),
            sources=tuple((ordinal, piece.local_id) for ordinal, piece in work.pieces),
        )

    built_sections = tuple(build(work) for work in sections)
    built_notes = tuple(build(work) for work in notes)
    stories = tuple(
        PageStory(page_ordinal=ordinal, section_id=ids[id(section)], node=build(story))
        for ordinal, section, story in ctx.stories
    )
    return AssembledDocumentHypothesis(
        pages=tuple(page.input for page in pages),
        sections=built_sections,
        notes=built_notes,
        stories=stories,
        story_sets=st.derive_story_sets(built_sections, stories),
        seam_decisions=tuple(ctx.decisions),
        conflicts=tuple(conflicts),
    )


def _resolve(
    work: _Work, assembled_id: str, conflicts: list[PropertyConflict]
) -> dict[str, TypedValue]:
    """Pinned resolver (design §6): pointers only the first piece observes
    come from the first piece, pointers only the last piece observes from
    the last, every other pointer must agree across the pieces that know
    it -- a disagreement is a conflict, never a winner, and a pointer no
    piece knows stays unresolved."""
    from_first = piece_withdrawal.withdrawn_pointers_for_piece(work.node_type, "last")
    from_last = piece_withdrawal.withdrawn_pointers_for_piece(work.node_type, "first")
    pointers = sorted({pointer for _, piece in work.pieces for pointer in piece.typed})
    typed: dict[str, TypedValue] = {}
    for pointer in pointers:
        if work.continuation_row and pointer in piece_withdrawal.withdrawn_pointers_for_piece(
            "row", "middle"
        ):
            continue
        if pointer in from_first:
            sources = work.pieces[:1]
        elif pointer in from_last:
            sources = work.pieces[-1:]
        else:
            sources = work.pieces
        values = tuple(
            (ordinal, piece.typed[pointer]) for ordinal, piece in sources if pointer in piece.typed
        )
        if not values:
            continue
        if all(_same(value, values[0][1]) for _, value in values):
            typed[pointer] = values[0][1]
        else:
            conflicts.append(PropertyConflict(assembled_id, pointer, values))
    return typed


# ---------------------------------------------------------------------------
# Conservation invariants (design §5.4)
# ---------------------------------------------------------------------------


def check_invariants(
    pages: Sequence[PageHypothesis], document: AssembledDocumentHypothesis, *, profile: str
) -> None:
    """Raise `AssemblyInvariantViolation` unless `document` owns every
    input node exactly once, conserves every page's text in page order,
    and replays through the grammar under `profile`."""
    _check_ownership(pages, document)
    _check_text(pages, document)
    _check_grammar(document, profile)


def _walk(nodes: Sequence[PageNode] | Sequence[AssembledNode]) -> Iterator:
    for node in nodes:
        yield node
        yield from _walk(node.children)


def _assembled_roots(document: AssembledDocumentHypothesis) -> list[AssembledNode]:
    return [*document.sections, *document.notes, *(story.node for story in document.stories)]


def _check_ownership(
    pages: Sequence[PageHypothesis], document: AssembledDocumentHypothesis
) -> None:
    inputs = [
        (page.input.page_ordinal, node.local_id)
        for page in pages
        for node in _walk([*page.sections, *page.notes])
    ]
    owners: dict[PieceRef, list[str]] = {ref: [] for ref in inputs}
    assembled = list(_walk(_assembled_roots(document)))
    for node in assembled:
        if not node.sources:
            raise AssemblyInvariantViolation(
                "ownership", f"assembled node {node.assembled_id} owns no input token"
            )
        for ref in node.sources:
            if ref not in owners:
                raise AssemblyInvariantViolation(
                    "ownership", f"{node.assembled_id} claims page {ref[0]} node {ref[1]}, "
                    "which no input page has"
                )
            owners[ref].append(node.assembled_id)
    for (ordinal, local_id), owned_by in owners.items():
        if len(owned_by) != 1:
            raise AssemblyInvariantViolation(
                "ownership",
                f"page {ordinal} node {local_id} maps to {len(owned_by)} assembled nodes "
                f"{owned_by}, not exactly one",
            )
    joins = sum(decision.joined for decision in document.seam_decisions)
    if len(assembled) != len(inputs) - joins:
        raise AssemblyInvariantViolation(
            "ownership",
            f"{len(assembled)} assembled nodes != {len(inputs)} input nodes - {joins} joins",
        )


def _run_text(nodes: Sequence[PageNode] | Sequence[AssembledNode]) -> str:
    return "".join(node.text or "" for node in _walk(nodes) if node.node_type == "run")


def _without_stories(node: PageNode) -> PageNode:
    children = tuple(child for child in node.children if child.node_type not in _STORY_TYPES)
    return PageNode(node.local_id, node.node_type, children, node.text, node.typed)


def _page_stories(page: PageHypothesis) -> list[PageNode]:
    return [c for s in page.sections for c in s.children if c.node_type in _STORY_TYPES]


def _check_text(pages: Sequence[PageHypothesis], document: AssembledDocumentHypothesis) -> None:
    streams = {
        "body": (
            [_run_text([_without_stories(s) for s in page.sections]) for page in pages],
            _run_text(document.sections),
        ),
        "notes": ([_run_text(page.notes) for page in pages], _run_text(document.notes)),
        "stories": (
            [_run_text(_page_stories(page)) for page in pages],
            _run_text([story.node for story in document.stories]),
        ),
    }
    for stream, (page_texts, out) in streams.items():
        count_in = sum(len(text) for text in page_texts)
        if count_in != len(out):
            raise AssemblyInvariantViolation(
                "text", f"{stream}: {count_in} scalars in, {len(out)} scalars out"
            )
        cursor = 0
        for page, text in zip(pages, page_texts, strict=True):
            if not out.startswith(text, cursor):
                raise AssemblyInvariantViolation(
                    "text",
                    f"{stream}: page {page.input.page_ordinal}'s text is not the contiguous "
                    f"substring at scalar {cursor} of the output",
                )
            cursor += len(text)


def structural_tokens(document: AssembledDocumentHypothesis) -> list[str]:
    """The assembled body as one standalone-page envelope of structural
    token names (stories lifted out, design §5.4 check 3)."""
    tokens = [events_vocab.FRAGMENT_BEGIN]
    for node in (*document.sections, *document.notes):
        _emit(node, tokens)
    tokens.append(events_vocab.END_FRAGMENT)
    return tokens


def _emit(node: AssembledNode, tokens: list[str]) -> None:
    if node.node_type in events_vocab.LEAF_TOKEN:
        tokens.append(events_vocab.LEAF_TOKEN[node.node_type])
        return
    tokens.append(events_vocab.OPEN_TOKEN[node.node_type])
    if node.text:
        tokens += [events_vocab.TEXT_BEGIN, events_vocab.TEXT_END]
    for child in node.children:
        _emit(child, tokens)
    tokens.append(events_vocab.CLOSE_TOKEN[node.node_type])


def _check_grammar(document: AssembledDocumentHypothesis, profile: str) -> None:
    state = grammar.fresh(profile)
    for position, token in enumerate(structural_tokens(document)):
        try:
            state = grammar.step(state, token)
        except grammar.GrammarViolation as exc:
            raise AssemblyInvariantViolation(
                "grammar", f"assembled token {position} {token}: {exc}"
            ) from exc

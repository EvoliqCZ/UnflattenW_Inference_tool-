"""Gold reference R(doc) for the S4 readiness gate (T4, measurement only).

``build`` turns one document bundle plus its per-page fragment sidecars into
the reference against which an S4 assembler's hypothesis will be compared:

* ``node_ids``   — the reconciled gold nodes: the union of the per-page
  present-minus-dropped node sets, minus the document root.
* ``structure``  — preorder (depth, node_type) over the gold tree with the
  story subtrees (header/footer/footnote/endnote) removed.
* ``text``       — per run, the concat of the scalar ranges the
  ``ScalarBoundaryProjector`` grants across the pages that carry the run.
* ``typed``      — per (node, pointer) the supervised value; a cross-page
  value disagreement is a reference defect (``ReferenceError``), not
  assembler divergence.
* ``stories``    — per page, the (page, story node, AST kind) triples for the
  header/footer slices the page actually carries.
* ``seams``      — per adjacent page pair: scored (section/block labels plus
  the recursive joined-block labels) or excluded_hole_seam.
* ``excluded``   — the five X_* exclusion counters.
* ``attribution``— the four attribution counters for off-seam structure.

Layering (M-self-ref): this module may import ONLY
``unflatten.model.boundary_projector`` from the model layer — never
``unflatten.assembly`` — because the reference must stay independent of the
assembler it judges. See ``tests/eval/assembly/test_reference.py::TestClosure``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from unflatten.model.boundary_projector import ScalarBoundaryProjector

EXCLUDED_KEYS = (
    "X_HOLE",
    "X_PROJECTION_DROP",
    "X_UNGRANTED_TEXT",
    "X_STORY_UNPLACED",
    "X_TYPED_UNSUPERVISED",
)

ATTRIBUTION_KEYS = (
    "nonadjacent_piece",
    "off_spine_split",
    "cross_page_pointer_events",
    "repeat_header_band",
)

_STORY = frozenset({"header", "footer", "footnote", "endnote"})
_NON_BODY = frozenset({"document", "section"})
_RECONCILED = ("table", "list", "paragraph")
_CROSS = frozenset({("section", "/break_type"), ("cell", "/row_span")})
_REPEAT_HEADER = "/layout/repeat_header"
_AST_CHILD_KEYS: dict[str, tuple[str, ...]] = {
    "section": ("headers", "footers", "children"),
    "table": ("rows",),
    "row": ("cells",),
    "cell": ("children",),
    "list": ("items",),
    "list_item": ("children",),
    "header": ("children",),
    "footer": ("children",),
    "paragraph": ("children",),
}


class ReferenceError(ValueError):
    """The inputs cannot yield a valid reference (a reference-side defect)."""


@dataclass(frozen=True)
class SeamLabel:
    """One gold join label across a seam, at a given structural level."""

    level: str
    a_node_id: str
    b_node_id: str
    a_node_type: str
    b_node_type: str
    gold_join: bool


@dataclass(frozen=True)
class Seam:
    """The seam between page ``a`` and page ``a + 1``."""

    a: int
    b: int
    status: str
    labels: tuple[SeamLabel, ...]


@dataclass(frozen=True)
class DocumentReference:
    """The gold reference R(doc) for one document bundle."""

    node_ids: frozenset[str]
    structure: tuple[tuple[int, str], ...]
    text: dict[str, str]
    typed: dict[str, dict[str, Any]]
    stories: tuple[tuple[int, str, str], ...]
    seams: tuple[Seam, ...]
    excluded: Counter
    attribution: Counter


@dataclass(frozen=True)
class _Graph:
    """The document tree derived from the structural sidecar."""

    node_order: tuple[str, ...]
    node_types: tuple[str, ...]
    oid: dict[str, int]
    parent: tuple[int, ...]
    children: tuple[list[int], ...]
    depth: tuple[int, ...]
    in_story: tuple[bool, ...]
    ast_nodes: dict[str, dict[str, Any]]


def _is_hole(fragment: Any) -> bool:
    if fragment is None:
        return True
    if not isinstance(fragment, dict):
        return not fragment
    return not fragment.get("slices")


def _ast_index(ast: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}

    def walk(node: dict[str, Any]) -> None:
        nodes[node["id"]] = node
        for key in _AST_CHILD_KEYS.get(node["type"], ()):
            for child in node.get(key, ()):
                walk(child)

    for section in ast.get("sections", ()):
        walk(section)
    return nodes


def _graph(structural: dict[str, Any], ast: dict[str, Any]) -> _Graph:
    node_order = tuple(structural["node_order"])
    node_types = tuple(structural["node_types"])
    size = len(node_order)
    oid = {nid: i for i, nid in enumerate(node_order)}
    ord_by_open: dict[int, int] = {}
    ordinal = 0
    for ev in structural["events"]:
        if ev["op"] == "open":
            ord_by_open[ev["index"]] = ordinal
            ordinal += 1
    parent = tuple(
        -1 if po == -1 else ord_by_open[po] for po in structural["parent_index"]
    )
    children: list[list[int]] = [[] for _ in range(size)]
    for i, p in enumerate(parent):
        if p >= 0:
            children[p].append(i)
    depth = [0] * size
    in_story = [False] * size
    for i in range(size):
        p = parent[i]
        if p >= 0:
            depth[i] = depth[p] + 1
            in_story[i] = in_story[p] or node_types[i] in _STORY
        else:
            in_story[i] = node_types[i] in _STORY
    return _Graph(
        node_order, node_types, oid, parent,
        tuple(children), tuple(depth), tuple(in_story), _ast_index(ast),
    )


def _validate(
    n: int,
    targets: dict[str, Any],
    fragments: list[Any],
    dropped: list[Any],
    text_spans: list[Any],
    typed_targets: list[Any],
    holes: list[bool],
    hole_reasons: list[Any] | None,
) -> None:
    for name, seq in (
        ("dropped", dropped),
        ("text_spans", text_spans),
        ("typed_targets", typed_targets),
    ):
        if len(seq) != n:
            raise ReferenceError(f"{name} has {len(seq)} entries, expected {n}")
    if hole_reasons is not None and len(hole_reasons) != n:
        raise ReferenceError(
            f"hole_reasons has {len(hole_reasons)} entries, expected {n}",
        )
    for k in range(n):
        sidecars = (dropped[k], text_spans[k], typed_targets[k])
        if any(s is None for s in sidecars) != holes[k]:
            raise ReferenceError(
                f"page {k}: sidecars are not uniformly present or absent",
            )
    if hole_reasons is not None:
        for k, reason in enumerate(hole_reasons):
            if reason is None:
                continue
            if not holes[k] or not isinstance(reason, str):
                raise ReferenceError(
                    f"page {k}: hole_reasons entry {reason!r} is not a "
                    "hole-page reason",
                )
    for k in range(n):
        if holes[k]:
            continue
        frag = fragments[k]
        if (
            frag["ast_hash"] != targets["ast_hash"]
            or frag["target_bundle_hash"] != targets["target_bundle_hash"]
        ):
            raise ReferenceError(f"page {k}: fragment is from another target bundle")


def _node_ids(
    graph: _Graph,
    present: list[frozenset[str]],
    dropped: list[Any],
    holes: list[bool],
) -> frozenset[str]:
    """Gold node set: every bundle node except the document root, the
    projected-out (dropped) nodes, and story subtrees no page carries."""
    dropped_all: set[str] = set()
    for k in range(len(holes)):
        if not holes[k]:
            dropped_all |= dropped[k]
    seen: set[str] = set()
    for k in range(len(holes)):
        seen |= present[k]
    ids = set()
    for i, nid in enumerate(graph.node_order):
        if nid == graph.node_order[0] or nid in dropped_all:
            continue
        if graph.in_story[i] and nid not in seen:
            continue
        ids.add(nid)
    return frozenset(ids)


def _structure(graph: _Graph, node_ids: frozenset[str]) -> tuple[tuple[int, str], ...]:
    out: list[tuple[int, str]] = []

    def walk(i: int) -> None:
        if graph.in_story[i]:
            return
        if graph.node_order[i] in node_ids:
            out.append((graph.depth[i] - 1, graph.node_types[i]))
        for c in graph.children[i]:
            walk(c)

    for c in graph.children[0]:
        walk(c)
    return tuple(out)


def _merge_ranges(rngs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for a, b in sorted(rngs):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _text(
    graph: _Graph,
    node_ids: frozenset[str],
    targets: dict[str, Any],
    text_spans: list[Any],
    holes: list[bool],
) -> tuple[dict[str, str], int]:
    spans_by_node: dict[str, list[dict[str, Any]]] = {}
    for sp in targets["text"]["spans"]:
        spans_by_node.setdefault(sp["node_id"], []).append(sp)
    projector = ScalarBoundaryProjector()
    grants: list[dict[str, list[tuple[int, int]]]] = [
        {pr.text_span_id: [(pr.scalar_start, pr.scalar_end)]
         for pr in projector.project(a)}
        if not holes[k]
        else {}
        for k, a in enumerate(text_spans)
    ]
    text: dict[str, str] = {}
    ungranted = 0
    for i, nid in enumerate(graph.node_order):
        if nid not in node_ids or graph.node_types[i] != "run" or graph.in_story[i]:
            continue
        spans = spans_by_node.get(nid, [])
        if len(spans) > 1:
            raise ReferenceError(f"run {nid!r} has {len(spans)} text spans")
        if not spans:
            text[nid] = ""
            continue
        sp = spans[0]
        rngs: list[tuple[int, int]] = []
        for g in grants:
            rngs.extend(g.get(sp["text_span_id"], ()))
        merged = _merge_ranges(rngs)
        text[nid] = "".join(sp["text"][a:b] for a, b in merged)
        ungranted += len(sp["text"]) - sum(b - a for a, b in merged)
    return text, ungranted


def _typed(
    targets: dict[str, Any],
    typed_targets: list[Any],
    holes: list[bool],
) -> tuple[dict[str, dict[str, Any]], int]:
    typed: dict[str, dict[str, Any]] = {}
    for k in range(len(holes)):
        if holes[k]:
            continue
        for t in typed_targets[k]:
            if not t.supervision_mask:
                continue
            ptr = t.node_property_pointer
            d = typed.setdefault(t.node_id, {})
            if ptr in d and d[ptr] != t.value:
                raise ReferenceError(
                    f"typed value for {t.node_id!r} {ptr!r} disagrees across pages",
                )
            d[ptr] = t.value
    supervised = {
        (t.node_id, t.node_property_pointer)
        for k in range(len(holes))
        if not holes[k]
        for t in typed_targets[k]
        if t.supervision_mask
    }
    universe = {
        (nid, f["node_property_pointer"])
        for nid, v in targets["typed"].items()
        for f in v["fields"]
    }
    return typed, len(universe - supervised)


def _stories(
    graph: _Graph,
    fragments: list[Any],
    holes: list[bool],
) -> tuple[tuple[tuple[int, str, str], ...], int]:
    ast_nodes = graph.ast_nodes
    stories: list[tuple[int, str, str]] = []
    unplaced = 0
    section_first: dict[str, int] = {}
    for k in range(len(holes)):
        if holes[k]:
            continue
        slices = fragments[k]["slices"]
        sections = [s for s in slices if s["node_type"] == "section"]
        for s in slices:
            if s["node_type"] in ("header", "footer"):
                stories.append((k, s["node_id"], ast_nodes[s["node_id"]]["kind"]))
        if not sections:
            continue
        section_first.setdefault(sections[0]["node_id"], k)
        sec = ast_nodes[sections[0]["node_id"]]
        word_page = fragments[k]["page_indices"][0] + 1
        for fam, ftype in (("headers", "header"), ("footers", "footer")):
            fam_nodes = sec.get(fam, ())
            if not fam_nodes:
                continue
            kinds = {f["kind"] for f in fam_nodes}
            if k == section_first[sections[0]["node_id"]] and "first" in kinds:
                resolved = "first"
            elif word_page % 2 == 0 and "even" in kinds:
                resolved = "even"
            else:
                resolved = "default"
            placed = any(
                s["node_type"] == ftype and ast_nodes[s["node_id"]].get("kind") == resolved
                for s in slices
            )
            if not placed:
                unplaced += 1
    return tuple(stories), unplaced


def _blocks_of(graph: _Graph, present: list[frozenset[str]], k: int) -> list[int]:
    return [
        i for i in sorted(graph.oid[n] for n in present[k])
        if graph.node_types[i] not in _NON_BODY and not graph.in_story[i]
        and graph.parent[i] >= 0 and graph.node_types[graph.parent[i]] == "section"
    ]


def _child_present(
    graph: _Graph,
    present: list[frozenset[str]],
    k: int,
    node_id: str,
    ctype: str,
) -> list[int]:
    i = graph.oid[node_id]
    return [
        c for c in graph.children[i]
        if graph.node_types[c] == ctype and graph.node_order[c] in present[k]
    ]


def _label(graph: _Graph, a: int, b: int, level: str) -> SeamLabel:
    return SeamLabel(
        level=level,
        a_node_id=graph.node_order[a],
        b_node_id=graph.node_order[b],
        a_node_type=graph.node_types[a],
        b_node_type=graph.node_types[b],
        gold_join=a == b,
    )


def _block_pair(
    graph: _Graph,
    present: list[frozenset[str]],
    k: int,
    ai: int,
    bi: int,
) -> tuple[SeamLabel, ...]:
    if ai != bi:
        return ()
    t = graph.node_types[ai]
    if t == "table":
        return _table_pair(graph, present, k, ai)
    if t == "list":
        return _list_pair(graph, present, k, ai)
    if t == "paragraph":
        return _paragraph_pair(graph, present, k, ai)
    return ()


def _cell_blocks(
    graph: _Graph,
    present: list[frozenset[str]],
    k: int,
    cell: int,
) -> list[int]:
    nid = graph.node_order[cell]
    blocks: list[int] = []
    for ctype in ("paragraph", "table", "list"):
        blocks.extend(_child_present(graph, present, k, nid, ctype))
    return blocks


def _row_pair(
    graph: _Graph,
    present: list[frozenset[str]],
    k: int,
    ai: int,
) -> tuple[SeamLabel, ...]:
    """Pair the joined row's cells by `/grid_column` (DEC-049 point 7,
    item 9.3): a UNION over columns present on either page, the same rule
    `assemble._join_cells` uses to join cells across a seam -- never a
    positional zip. A cell closed before the seam (present on page `k`
    only) or opened after it (present on page `k + 1` only) is simply
    unpaired, never a reference defect: M1 measured 17 documents refused
    by the old positional zip, all of them a cell absent from one side
    shifting every later pairing (§9.3)."""
    nid = graph.node_order[ai]

    def col(c: int) -> int:
        return graph.ast_nodes[graph.node_order[c]]["grid_column"]

    ca = _child_present(graph, present, k, nid, "cell")
    cb = _child_present(graph, present, k + 1, nid, "cell")
    by_column = {col(c): c for c in ca}
    out: list[SeamLabel] = []
    for c in cb:
        partner = by_column.get(col(c))
        if partner is None:
            continue
        ia = _cell_blocks(graph, present, k, partner)
        ib = _cell_blocks(graph, present, k + 1, c)
        if ia and ib:
            out.extend(_block_pair(graph, present, k, ia[-1], ib[0]))
    return tuple(out)


def _table_pair(
    graph: _Graph,
    present: list[frozenset[str]],
    k: int,
    ai: int,
) -> tuple[SeamLabel, ...]:
    ra = _child_present(graph, present, k, graph.node_order[ai], "row")
    rb = _child_present(graph, present, k + 1, graph.node_order[ai], "row")
    if not ra or not rb:
        return ()
    out = [_label(graph, ra[-1], rb[0], "row")]
    if ra[-1] == rb[0]:
        out.extend(_row_pair(graph, present, k, ra[-1]))
    return tuple(out)


def _list_pair(
    graph: _Graph,
    present: list[frozenset[str]],
    k: int,
    ai: int,
) -> tuple[SeamLabel, ...]:
    nid = graph.node_order[ai]
    ra = _child_present(graph, present, k, nid, "list_item")
    rb = _child_present(graph, present, k + 1, nid, "list_item")
    if not ra or not rb:
        return ()
    out = [_label(graph, ra[-1], rb[0], "list_item")]
    if ra[-1] == rb[0]:
        item = graph.node_order[ra[-1]]
        ia = _child_present(graph, present, k, item, "paragraph")
        ib = _child_present(graph, present, k + 1, item, "paragraph")
        if ia and ib:
            out.extend(_block_pair(graph, present, k, ia[-1], ib[0]))
    return tuple(out)


def _paragraph_pair(
    graph: _Graph,
    present: list[frozenset[str]],
    k: int,
    ai: int,
) -> tuple[SeamLabel, ...]:
    nid = graph.node_order[ai]
    ra = _child_present(graph, present, k, nid, "run")
    rb = _child_present(graph, present, k + 1, nid, "run")
    if not ra or not rb:
        return ()
    return (_label(graph, ra[-1], rb[0], "run"),)


def _seams(
    graph: _Graph,
    present: list[frozenset[str]],
    holes: list[bool],
) -> tuple[Seam, ...]:
    seams: list[Seam] = []
    for k in range(len(holes) - 1):
        if holes[k] or holes[k + 1]:
            seams.append(
                Seam(a=k, b=k + 1, status="excluded_hole_seam", labels=()),
            )
            continue
        sa = [i for i in sorted(graph.oid[n] for n in present[k])
              if graph.node_types[i] == "section"]
        sb = [i for i in sorted(graph.oid[n] for n in present[k + 1])
              if graph.node_types[i] == "section"]
        ba = _blocks_of(graph, present, k)
        bb = _blocks_of(graph, present, k + 1)
        labels: list[SeamLabel] = []
        if sa and sb:
            labels.append(_label(graph, sa[-1], sb[0], "section"))
        if ba and bb:
            labels.append(_label(graph, ba[-1], bb[0], "block"))
            labels.extend(_block_pair(graph, present, k, ba[-1], bb[0]))
        seams.append(Seam(a=k, b=k + 1, status="scored", labels=tuple(labels)))
    return tuple(seams)


def _nonadjacent(
    graph: _Graph,
    present: list[frozenset[str]],
) -> int:
    count = 0
    for i, nid in enumerate(graph.node_order):
        if graph.node_types[i] in _NON_BODY:
            continue
        pres = [nid in present[k] for k in range(len(present))]
        for k in range(len(present) - 2):
            if pres[k] and not pres[k + 1] and pres[k + 2]:
                count += 1
    return count


def _off_spine(
    graph: _Graph,
    present: list[frozenset[str]],
    dropped: list[Any],
    holes: list[bool],
) -> int:
    count = 0
    for k in range(len(holes) - 1):
        if holes[k] or holes[k + 1]:
            continue
        ba = _blocks_of(graph, present, k)
        bb = _blocks_of(graph, present, k + 1)
        edge = ba[-1] if ba and bb and ba[-1] == bb[0] else None
        for t in _RECONCILED:
            pieces = [
                i for i in range(len(graph.node_order))
                if graph.node_types[i] == t and not graph.in_story[i]
                and graph.node_order[i] in present[k]
                and graph.node_order[i] in present[k + 1]
                and graph.node_order[i] not in (dropped[k] or frozenset())
                and graph.node_order[i] not in (dropped[k + 1] or frozenset())
            ]
            if not pieces:
                continue
            if edge is not None and graph.node_types[edge] == t:
                continue
            parents = [graph.parent[i] for i in pieces]
            if any(p >= 0 and graph.node_types[p] == "section" for p in parents):
                count += 1
    return count


def _cross_events(
    graph: _Graph,
    typed_targets: list[Any],
    holes: list[bool],
    present: list[frozenset[str]],
) -> int:
    page_count = {
        nid: sum(1 for k in range(len(present)) if nid in present[k])
        for nid in graph.node_order
    }
    count = 0
    for k in range(len(holes)):
        if holes[k]:
            continue
        for t in typed_targets[k]:
            if (
                t.supervision_mask
                and (t.node_type, t.node_property_pointer) in _CROSS
                and page_count[t.node_id] >= 2
            ):
                count += 1
    return count


def _repeat_header_bands(
    graph: _Graph,
    typed_targets: list[Any],
    holes: list[bool],
    present: list[frozenset[str]],
) -> int:
    table_pages = {
        nid: sum(1 for k in range(len(present)) if nid in present[k])
        for nid in graph.node_order
        if graph.node_types[graph.oid[nid]] == "table"
    }
    count = 0
    for k in range(len(holes)):
        if holes[k]:
            continue
        for t in typed_targets[k]:
            if not (
                t.supervision_mask
                and t.node_property_pointer == _REPEAT_HEADER
                and t.value is True
                and graph.node_types[graph.oid[t.node_id]] == "row"
            ):
                continue
            anc = graph.parent[graph.oid[t.node_id]]
            while anc >= 0 and graph.node_types[anc] != "table":
                anc = graph.parent[anc]
            if anc >= 0 and table_pages[graph.node_order[anc]] >= 2:
                count += 1
    return count


def build(
    *,
    targets: dict[str, Any],
    fragments: list[dict[str, Any] | None],
    dropped: list[Any],
    text_spans: list[Any],
    typed_targets: list[Any],
    ast: dict[str, Any],
    hole_reasons: list[str | None] | None = None,
) -> DocumentReference:
    """Assemble the gold reference R(doc) from bundle + per-page sidecars.

    ``fragments`` / ``dropped`` / ``text_spans`` / ``typed_targets`` are
    parallel per-page sequences; page ``k`` is a hole when its fragment is
    missing or slice-less, which requires all three sidecars to be ``None``
    (uniform None-ness). Raises :class:`ReferenceError` for any input from
    which no valid reference can be constructed.
    """
    n = len(fragments)
    if n < 1:
        raise ReferenceError("at least one page is required")
    holes = [_is_hole(f) for f in fragments]
    _validate(n, targets, fragments, dropped, text_spans, typed_targets,
              holes, hole_reasons)
    graph = _graph(targets["structural"], ast)
    present = [
        frozenset() if holes[k]
        else frozenset(s["node_id"] for s in fragments[k]["slices"])
        for k in range(n)
    ]
    node_ids = _node_ids(graph, present, dropped, holes)
    text, ungranted = _text(graph, node_ids, targets, text_spans, holes)
    typed, unsupervised = _typed(targets, typed_targets, holes)
    stories, unplaced = _stories(graph, fragments, holes)
    excluded = Counter({
        "X_HOLE": sum(holes),
        "X_PROJECTION_DROP": sum(len(d) for d in dropped if d is not None),
        "X_UNGRANTED_TEXT": ungranted,
        "X_STORY_UNPLACED": unplaced,
        "X_TYPED_UNSUPERVISED": unsupervised,
    })
    attribution = Counter({
        "nonadjacent_piece": _nonadjacent(graph, present),
        "off_spine_split": _off_spine(graph, present, dropped, holes),
        "cross_page_pointer_events": _cross_events(graph, typed_targets, holes,
                                                    present),
        "repeat_header_band": _repeat_header_bands(graph, typed_targets, holes,
                                                    present),
    })
    return DocumentReference(
        node_ids=node_ids,
        structure=_structure(graph, node_ids),
        text=text,
        typed=typed,
        stories=stories,
        seams=_seams(graph, present, holes),
        excluded=excluded,
        attribution=attribution,
    )

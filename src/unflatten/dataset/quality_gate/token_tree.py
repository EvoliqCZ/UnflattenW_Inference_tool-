"""Token-tree detectors over one gold page stream (DEC-055 V3 audit,
DEC-057 G-A). Moved verbatim from `scripts/loop_data_audit.py` so the
data-quality gate (`unflatten.dataset.quality_gate`) and the audit script
run ONE definition; the audit re-exports every name it used before.

Definitions (all on the token tree; SUSPEND/RESUME never occur in v15b
standalone pages, and a stream that uses them is refused):

* content token = any text token (Qwen or byte frame), `INLINE_IMAGE`,
  `FLOATING_IMAGE`, `FIELD`, `NOTE_REF`. `TAB`, `LINE_BREAK` and empty runs
  are whitespace, not content.
* textless node = a subtree without a content token.
* a streak = consecutive textless siblings of one kind under one parent
  (empty cells in a row, empty rows in a table, empty runs in a paragraph,
  empty paragraphs in a container). A per-unit cap K vetoes the (K+1)-th
  copy, so a page trips cap K iff its streak exceeds K.
* paragraph shapes (DEC-016): A = `PO PC`, B = `PO RO TB TE RC PC`,
  Bn = `PO` + two or more empty runs + `PC`, W = any other textless
  paragraph (tabs, line breaks).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from unflatten.model import events_vocab

#: Packed layout of the v15b composite vocabulary (`composite_vocab.
#: packed_ranges(151936)`): the structural ids come first, then Qwen.
ID = events_vocab.TOKEN_IDS
NAMES = {v: k for k, v in ID.items()}
QWEN_START = len(ID)

CONTENT_STRUCTURAL = frozenset(
    ID[name] for name in ("INLINE_IMAGE", "FLOATING_IMAGE", "FIELD", "NOTE_REF")
)
TAB = ID["TAB"]
PO, PC = ID["PARAGRAPH_OPEN"], ID["PARAGRAPH_CLOSE"]
RO, RC = ID["RUN_OPEN"], ID["RUN_CLOSE"]
TB, TE = ID["TEXT_BEGIN"], ID["TEXT_END"]
TO, TC = ID["TABLE_OPEN"], ID["TABLE_CLOSE"]
ROW_O, ROW_C = ID["ROW_OPEN"], ID["ROW_CLOSE"]
CO, CC = ID["CELL_OPEN"], ID["CELL_CLOSE"]
EMPTY_RUN = (RO, TB, TE, RC)
PERIOD_MAX = 32


def is_content(token: int) -> bool:
    return token >= QWEN_START or token in CONTENT_STRUCTURAL


@dataclass
class Node:
    kind: int  # opening token id; -1 for the virtual root
    start: int
    end: int = -1  # index of the closing token (inclusive)
    content: int = 0  # content tokens in the subtree
    children: list[Node] = field(default_factory=list)

    @property
    def textless(self) -> bool:
        return self.content == 0

    @property
    def size(self) -> int:
        return self.end - self.start + 1


def parse_tree(tokens: Sequence[int]) -> Node:
    """Nodes for every `*_OPEN`/`TEXT_BEGIN` .. matching closer. Leaves
    (`TAB`, `FIELD`, ...) are not nodes; they count toward their parent's
    `content` only if they are content. `END_*` closes nothing here."""
    root = Node(kind=-1, start=0)
    stack = [root]
    for i, token in enumerate(tokens):
        if token < QWEN_START and NAMES[token].startswith(("SUSPEND_NODE", "RESUME_NODE")):
            raise ValueError(f"token {i} is {NAMES[token]}: continuation streams are not audited")
        if is_content(token):
            for node in stack:
                node.content += 1
        if token >= QWEN_START:
            continue
        name = NAMES[token]
        if name.endswith("_OPEN") or name == "TEXT_BEGIN":
            node = Node(kind=token, start=i)
            stack[-1].children.append(node)
            stack.append(node)
        elif name.endswith("_CLOSE") or name == "TEXT_END":
            if len(stack) == 1:
                raise ValueError(f"token {i} {name} closes nothing")
            stack.pop().end = i
    root.end = len(tokens) - 1
    if len(stack) != 1:
        raise ValueError(f"{len(stack) - 1} node(s) left open at end of stream")
    return root


def walk(node: Node) -> Iterable[Node]:
    yield node
    for child in node.children:
        yield from walk(child)


def paragraph_shape(tokens: Sequence[int], node: Node) -> str:
    """A / B / Bn / W for a textless paragraph (see module docstring)."""
    inner = tuple(tokens[node.start + 1 : node.end])
    if not inner:
        return "A"
    if len(inner) % 4 == 0 and all(inner[k : k + 4] == EMPTY_RUN for k in range(0, len(inner), 4)):
        return "B" if len(inner) == 4 else "Bn"
    return "W"


def _streaks(children: list[Node], kind: int, pred) -> tuple[int, int]:
    """(longest run of consecutive `kind` children satisfying `pred`,
    start token of that run). Children of another kind break the run."""
    best, best_at, cur, cur_at = 0, -1, 0, -1
    for child in children:
        if child.kind == kind and pred(child):
            if cur == 0:
                cur_at = child.start
            cur += 1
            if cur > best:
                best, best_at = cur, cur_at
        else:
            cur = 0
    return best, best_at


def _keep(metrics: dict, name: str, value: int, at: int) -> None:
    if value > metrics[name]:
        metrics[name] = value
        metrics["at"][name] = at


def page_metrics(tokens: Sequence[int]) -> dict:
    """Every extreme-structure measure of one gold page. `at[name]` is the
    token offset where the page's maximum of `name` starts."""
    root = parse_tree(tokens)
    m: dict = {
        "n_tokens": len(tokens),
        "content_tokens": root.content,
        "tables": 0,
        "textless_tables": 0,
        "textless_table_cells_max": 0,
        "textless_table_tokens": 0,
        "cells_per_row_max": 0,
        "rows_per_table_max": 0,
        "empty_cells_total": 0,
        "empty_cell_tokens": 0,
        "empty_cell_streak_max": 0,
        "empty_cells_in_table_max": 0,
        "empty_rows_total": 0,
        "empty_row_tokens": 0,
        "empty_row_streak_max": 0,
        "trailing_empty_rows_max": 0,
        "skeleton_rows": 0,
        "empty_runs_total": 0,
        "empty_run_chains": 0,
        "empty_runs_sandwiched": 0,
        "empty_run_streak_max": 0,
        "empty_run_chain_tokens": 0,  # tokens in chains of >= 2 empty runs
        "empty_par_A": 0,
        "empty_par_B": 0,
        "empty_par_Bn": 0,
        "empty_par_W": 0,
        "empty_par_AB_streak_max": 0,
        "empty_par_any_streak_max": 0,
        "tab_run_max": 0,
        "repeat_textful_max": 0,
        "repeat_textful_kind": None,
        "repeat_textless_max": 0,
        "repeat_textless_kind": None,
        "at": {},
    }

    def empty_run(n: Node) -> bool:
        return n.kind == RO and tuple(tokens[n.start : n.end + 1]) == EMPTY_RUN

    for node in walk(root):
        kids = node.children
        if node.kind == TO:
            m["tables"] += 1
            rows = [c for c in kids if c.kind == ROW_O]
            _keep(m, "rows_per_table_max", len(rows), node.start)
            cells = [c for r in rows for c in r.children if c.kind == CO]
            empty_cells = sum(1 for c in cells if c.textless)
            _keep(m, "empty_cells_in_table_max", empty_cells, node.start)
            if node.textless:
                m["textless_tables"] += 1
                m["textless_table_tokens"] += node.size
                _keep(m, "textless_table_cells_max", len(cells), node.start)
            streak, at = _streaks(kids, ROW_O, lambda n: n.textless)
            _keep(m, "empty_row_streak_max", streak, at)
            for r in rows:
                if r.textless:
                    m["empty_rows_total"] += 1
                    m["empty_row_tokens"] += r.size
                    # a run inside a textless row held text or whitespace in
                    # the source whose text was withdrawn (a skeleton row);
                    # a blank form row has bare `PO PC` cells
                    if RO in tokens[r.start : r.end + 1]:
                        m["skeleton_rows"] += 1
            # textless rows after the table's last row with content (all rows
            # of a textless table): the rows `envelope._resolve_pages` rule 1
            # places on this page by borrowing, never by observation
            trailing = 0
            for r in reversed(rows):
                if not r.textless:
                    break
                trailing += 1
            _keep(m, "trailing_empty_rows_max", trailing, node.start)
        elif node.kind == ROW_O:
            cells = [c for c in kids if c.kind == CO]
            _keep(m, "cells_per_row_max", len(cells), node.start)
            for c in cells:
                if c.textless:
                    m["empty_cells_total"] += 1
                    m["empty_cell_tokens"] += c.size
            streak, at = _streaks(kids, CO, lambda n: n.textless)
            _keep(m, "empty_cell_streak_max", streak, at)
        elif node.kind == PO:
            runs = [c for c in kids if empty_run(c)]
            m["empty_runs_total"] += len(runs)
            streak, at = _streaks(kids, RO, empty_run)
            _keep(m, "empty_run_streak_max", streak, at)
            # chains of >= 2 back-to-back empty runs, and lone empty runs
            # with a text-carrying run on both sides (the shape a
            # whitespace-only source run takes once its text is withdrawn)
            cur = 0
            padded = [None, *kids, None]
            for k in range(1, len(padded)):
                child = padded[k]
                if child is not None and empty_run(child):
                    cur += 1
                    continue
                if cur >= 2:
                    m["empty_run_chain_tokens"] += 4 * cur
                    m["empty_run_chains"] += 1
                elif cur == 1:
                    left, right = padded[k - 2], child
                    if all(
                        n is not None and n.kind == RO and not n.textless for n in (left, right)
                    ):
                        m["empty_runs_sandwiched"] += 1
                cur = 0
            if node.textless:
                m["empty_par_" + paragraph_shape(tokens, node)] += 1
        if node.kind != PO:
            ab, at = _streaks(
                kids, PO, lambda n: n.textless and paragraph_shape(tokens, n) in ("A", "B")
            )
            _keep(m, "empty_par_AB_streak_max", ab, at)
            anyp, at = _streaks(kids, PO, lambda n: n.textless)
            _keep(m, "empty_par_any_streak_max", anyp, at)
        _repeat_split(tokens, kids, m)

    run = best = 0
    at = -1
    for i, token in enumerate(tokens):
        run = run + 1 if token == TAB else 0
        if run > best:
            best, at = run, i - run + 1
    m["tab_run_max"] = best
    m["at"]["tab_run_max"] = at
    return m


def _repeat_split(tokens: Sequence[int], kids: list[Node], m: dict) -> None:
    """Longest runs of back-to-back identical siblings, tracked separately
    for textful and textless runs (a run is homogeneous by construction)."""
    cur, prev, prev_node = 0, None, None
    for child in kids:
        key = tuple(tokens[child.start : child.end + 1])
        if key == prev:
            cur += 1
        else:
            cur, prev_node = 1, child
        prev = key
        name = "repeat_textless" if child.textless else "repeat_textful"
        if cur > m[name + "_max"]:
            m[name + "_max"] = cur
            m[name + "_kind"] = NAMES[child.kind]
            m["at"][name + "_max"] = prev_node.start


#: Defect signatures (see `defect_flags`).
SPILL_MIN_TRAILING_ROWS = 4  # next page skipped
SPILL_BACK_MIN_TRAILING_ROWS = 16  # only the previous page skipped
SKELETON_MIN_ROWS = 8


def defect_flags(row: dict, skipped: set[tuple[str, int]]) -> list[str]:
    """Import/target defect signatures of one page (`defect_exclude`
    candidates):

    * `spill`: textless rows trail the last content row of a table and the
      NEXT page of the same document was skipped by the cache build (no
      observation; >= 4 rows), or only the previous page was (>= 16 rows)
      -- the rows of that textless page were placed here by
      `envelope._resolve_pages` rule 1 (previous sibling's last page), so
      the target holds rows the image does not show;
    * `skeleton`: >= 8 textless rows that hold runs -- table text the page
      shows but whose text was withdrawn (no owner), leaving only the run
      skeleton as the target.
    """
    flags = []
    trailing = row["trailing_empty_rows_max"]
    forward = (row["bundle"], row["page"] + 1) in skipped
    backward = (row["bundle"], row["page"] - 1) in skipped
    if (forward and trailing >= SPILL_MIN_TRAILING_ROWS) or (
        backward and trailing >= SPILL_BACK_MIN_TRAILING_ROWS
    ):
        flags.append("spill")
    if row["skeleton_rows"] >= SKELETON_MIN_ROWS:
        flags.append("skeleton")
    return flags

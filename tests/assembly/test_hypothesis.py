"""Tests for `unflatten.assembly.hypothesis` (assembler design §4, task T1).

Every fixture stream is a literal token list packed through the real
composite vocabulary and parsed by the real `from_token_ids`, so a gold
page and a model page reach a rule through the same door. Each leak
fixture is paired with a "leaky view" built by hand in this file, which
proves the fixture has the power to score 100 % when a signal leaks --
a fixture that scores the baseline under every view would prove nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from unflatten import import_closure
from unflatten.assembly import hypothesis as hy
from unflatten.model import composite_vocab as cv
from unflatten.model import events_vocab as ev
from unflatten.model import piece_withdrawal as pw
from unflatten.model.node_lifecycle import NodeLifecycleTracker
from unflatten.model.prediction_ir import RawDocumentHypothesis
from unflatten.model.typed_decode import TypedPrediction

REPO_ROOT = Path(__file__).parents[2]

# ---------------------------------------------------------------------------
# Token fixtures
# ---------------------------------------------------------------------------


def _make_vocabulary() -> cv.CompositeVocabulary:
    allowlist_payload = cv.build_qwen_text_allowlist(
        tokenizer_artifact_set_hash="sha256:" + "ab" * 32,
        native_vocab_size=8,
        classify=lambda native_id: None,
    )
    return cv.make_composite_vocabulary(
        tokenizer_artifact_set_hash="sha256:" + "ab" * 32,
        allowlist_payload=allowlist_payload,
    )


class _ByteOnlyTokenizer:
    """Every fixture scalar is a byte frame, so no Qwen id is ever decoded."""

    def encode(self, text: str) -> list[int]:  # pragma: no cover - never called
        raise AssertionError("from_token_ids never encodes")

    def decode(self, ids: Sequence[int]) -> str:  # pragma: no cover - never called
        raise AssertionError("fixtures carry no Qwen ids")


VOCABULARY = _make_vocabulary()
TOKENIZER = _ByteOnlyTokenizer()
PAGE = hy.PageInput(page_ordinal=0, width_px=2480, height_px=3508)


@dataclass(frozen=True)
class _T:
    """A literal text span: TEXT_BEGIN, one byte frame per scalar, TEXT_END."""

    text: str


def _refs(items: Sequence[str | _T]) -> list[cv.TokenRef]:
    refs: list[cv.TokenRef] = []
    for item in items:
        if isinstance(item, _T):
            refs.append(cv.StructuralRef(local_id=ev.TOKEN_IDS["TEXT_BEGIN"]))
            for scalar in item.text:
                refs.append(cv.ByteControlRef(kind="byte_begin"))
                refs.extend(cv.ByteValueRef(value=b) for b in scalar.encode("utf-8"))
                refs.append(cv.ByteControlRef(kind="byte_end"))
            refs.append(cv.StructuralRef(local_id=ev.TOKEN_IDS["TEXT_END"]))
        else:
            refs.append(cv.StructuralRef(local_id=ev.TOKEN_IDS[item]))
    return refs


def _ids(items: Sequence[str | _T]) -> list[int]:
    return [VOCABULARY.pack(ref) for ref in _refs(items)]


def _parse(
    items: Sequence[str | _T],
    typed_by_anchor: Mapping[int, Mapping[str, hy.TypedValue]] | None = None,
) -> hy.PageHypothesis:
    return hy.from_token_ids(
        _ids(items),
        vocabulary=VOCABULARY,
        tokenizer=TOKENIZER,
        typed_by_anchor={} if typed_by_anchor is None else typed_by_anchor,
        page_input=PAGE,
    )


def _n(
    local_id: str,
    node_type: str,
    *children: hy.PageNode,
    text: str | None = None,
    typed: Mapping[str, hy.TypedValue] | None = None,
) -> hy.PageNode:
    return hy.PageNode(
        local_id=local_id,
        node_type=node_type,
        children=children,
        text=text,
        typed={} if typed is None else typed,
    )


# Positions in the comments are token positions AFTER byte-frame expansion.
TWO_SECTION_PAGE: list[str | _T] = [
    "FRAGMENT_BEGIN",  # 0
    "SECTION_OPEN",  # 1  n0
    "HEADER_OPEN",  # 2  n1
    "PARAGRAPH_OPEN",  # 3  n2
    "PARAGRAPH_CLOSE",  # 4
    "HEADER_CLOSE",  # 5
    "PARAGRAPH_OPEN",  # 6  n3
    "RUN_OPEN",  # 7  n4
    _T("A"),  # 8..12
    "TAB",  # 13 n5
    "RUN_CLOSE",  # 14
    "PARAGRAPH_CLOSE",  # 15
    "SECTION_CLOSE",  # 16
    "SECTION_OPEN",  # 17 n6
    "TABLE_OPEN",  # 18 n7
    "ROW_OPEN",  # 19 n8
    "CELL_OPEN",  # 20 n9
    "PARAGRAPH_OPEN",  # 21 n10
    "PARAGRAPH_CLOSE",  # 22
    "CELL_CLOSE",  # 23
    "ROW_CLOSE",  # 24
    "TABLE_CLOSE",  # 25
    "SECTION_CLOSE",  # 26
    "FOOTNOTE_OPEN",  # 27 n11
    "PARAGRAPH_OPEN",  # 28 n12
    "RUN_OPEN",  # 29 n13
    _T("é"),  # 30..35 (a two-byte frame)
    "RUN_CLOSE",  # 36
    "PARAGRAPH_CLOSE",  # 37
    "FOOTNOTE_CLOSE",  # 38
    "END_FRAGMENT",  # 39
]


# ---------------------------------------------------------------------------
# from_token_ids
# ---------------------------------------------------------------------------


def test_from_token_ids_two_section_page_returns_literal_tree() -> None:
    page = _parse(TWO_SECTION_PAGE)

    assert page.input == PAGE
    assert page.sections == (
        _n(
            "n0",
            "section",
            _n("n1", "header", _n("n2", "paragraph")),
            _n("n3", "paragraph", _n("n4", "run", _n("n5", "tab"), text="A")),
        ),
        _n(
            "n6",
            "section",
            _n("n7", "table", _n("n8", "row", _n("n9", "cell", _n("n10", "paragraph")))),
        ),
    )
    assert page.notes == (
        _n("n11", "footnote", _n("n12", "paragraph", _n("n13", "run", text="é"))),
    )


def test_from_token_ids_typed_anchor_attaches_to_node_at_that_position() -> None:
    page = _parse(
        TWO_SECTION_PAGE,
        typed_by_anchor={
            6: {"/layout/alignment": "center"},
            13: {"#card:/tabs": 0},
            19: {"/layout/repeat_header": True},
        },
    )

    paragraph = page.sections[0].children[1]
    tab = paragraph.children[0].children[0]
    row = page.sections[1].children[0].children[0]
    assert (paragraph.local_id, dict(paragraph.typed)) == ("n3", {"/layout/alignment": "center"})
    assert (tab.local_id, dict(tab.typed)) == ("n5", {"#card:/tabs": 0})
    assert (row.local_id, dict(row.typed)) == ("n8", {"/layout/repeat_header": True})
    assert dict(page.sections[0].typed) == {}


def test_from_token_ids_run_without_text_span_has_empty_text() -> None:
    page = _parse(
        [
            "FRAGMENT_BEGIN",
            "SECTION_OPEN",
            "PARAGRAPH_OPEN",
            "RUN_OPEN",
            "TAB",
            "RUN_CLOSE",
            "PARAGRAPH_CLOSE",
            "SECTION_CLOSE",
            "END_FRAGMENT",
        ]
    )

    assert page.sections[0].children[0].children[0].text == ""


_BODY = ["SECTION_OPEN", "PARAGRAPH_OPEN", "PARAGRAPH_CLOSE", "SECTION_CLOSE"]


@pytest.mark.parametrize(
    ("items", "message"),
    [
        (["FRAGMENT_BEGIN", "SECTION_OPEN", "PARAGRAPH_OPEN", "PARAGRAPH_CLOSE", "END_FRAGMENT"],
         "unclosed"),
        (["FRAGMENT_BEGIN", "SECTION_OPEN", "PARAGRAPH_CLOSE", "SECTION_CLOSE", "END_FRAGMENT"],
         "does not close"),
        (["FRAGMENT_BEGIN", *_BODY, "SECTION_CLOSE", "END_FRAGMENT"], "nothing open"),
        (_BODY, "FRAGMENT_BEGIN"),
        (["FRAGMENT_BEGIN", *_BODY], "END_FRAGMENT"),
        (["FRAGMENT_BEGIN", *_BODY, "END_FRAGMENT", "END_FRAGMENT"], "END_FRAGMENT"),
        (["FRAGMENT_BEGIN", "RESUME_NODE(section)", "SECTION_CLOSE", "END_FRAGMENT"],
         "never resumes"),
        (["FRAGMENT_BEGIN", "PARAGRAPH_OPEN", "PARAGRAPH_CLOSE", "END_FRAGMENT"], "top-level"),
        (["FRAGMENT_BEGIN", "SECTION_OPEN", _T("x"), "SECTION_CLOSE", "END_FRAGMENT"],
         "outside a run"),
        (["FRAGMENT_BEGIN", "SECTION_OPEN", "TEXT_END", "SECTION_CLOSE", "END_FRAGMENT"],
         "TEXT_END"),
        (["FRAGMENT_BEGIN", *_BODY, "END_DOCUMENT", "END_FRAGMENT"], "END_DOCUMENT"),
    ],
)
def test_from_token_ids_malformed_stream_raises(items: list[str | _T], message: str) -> None:
    with pytest.raises(hy.PageStreamInvalid, match=message):
        _parse(items)


def test_from_token_ids_unterminated_text_span_raises() -> None:
    items = ["FRAGMENT_BEGIN", "SECTION_OPEN", "PARAGRAPH_OPEN", "RUN_OPEN", "TEXT_BEGIN",
             "RUN_CLOSE", "PARAGRAPH_CLOSE", "SECTION_CLOSE", "END_FRAGMENT"]
    with pytest.raises(hy.PageStreamInvalid, match="no matching TEXT_END"):
        _parse(items)


def test_from_token_ids_text_token_outside_span_raises() -> None:
    ids = _ids(["FRAGMENT_BEGIN", "SECTION_OPEN"])
    ids.append(VOCABULARY.pack(cv.ByteValueRef(value=0x41)))
    ids += _ids(["SECTION_CLOSE", "END_FRAGMENT"])
    with pytest.raises(hy.PageStreamInvalid, match="outside TEXT_BEGIN"):
        hy.from_token_ids(
            ids, vocabulary=VOCABULARY, tokenizer=TOKENIZER, typed_by_anchor={}, page_input=PAGE
        )


@pytest.mark.parametrize("position", [0, 4, 8, 40])
def test_from_token_ids_typed_anchor_off_a_node_token_raises(position: int) -> None:
    # 0 envelope, 4 a close token, 8 TEXT_BEGIN, 40 past the end.
    with pytest.raises(hy.PageStreamInvalid, match="typed anchor"):
        _parse(TWO_SECTION_PAGE, typed_by_anchor={position: {"/layout/alignment": "left"}})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"page_ordinal": -1, "width_px": 1, "height_px": 1}, "page_ordinal"),
        ({"page_ordinal": 0, "width_px": 0, "height_px": 1}, "width_px"),
        ({"page_ordinal": 0, "width_px": 1, "height_px": -3}, "height_px"),
        ({"page_ordinal": True, "width_px": 1, "height_px": 1}, "page_ordinal"),
    ],
)
def test_page_input_out_of_range_raises(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        hy.PageInput(**kwargs)


# ---------------------------------------------------------------------------
# first_visible / last_visible
# ---------------------------------------------------------------------------


def _paragraph(
    *runs: str, typed: Mapping[str, hy.TypedValue] | None = None
) -> hy.PageNode:
    """One page holding one paragraph with `runs`; returns that paragraph."""
    items: list[str | _T] = ["FRAGMENT_BEGIN", "SECTION_OPEN", "PARAGRAPH_OPEN"]
    for run in runs:
        items += ["RUN_OPEN", _T(run), "RUN_CLOSE"]
    items += ["PARAGRAPH_CLOSE", "SECTION_CLOSE", "END_FRAGMENT"]
    page = _parse(items, typed_by_anchor={} if typed is None else {2: typed})
    return page.sections[0].children[0]


@pytest.mark.parametrize(
    ("runs", "first", "last"),
    [
        (("Hello ", " "), "H", "o"),  # trailing whitespace-only run
        ((" ", "World."), "W", "."),  # leading whitespace-only run
        (("  ", " world", " "), "w", "d"),  # non-ASCII Unicode whitespace
        ((" ", " "), None, None),  # whole text whitespace
        ((), None, None),  # no runs at all
        (("x",), "x", "x"),
    ],
)
def test_first_last_visible_skip_edge_whitespace(
    runs: tuple[str, ...], first: str | None, last: str | None
) -> None:
    node = _paragraph(*runs)

    assert (hy.first_visible(node), hy.last_visible(node)) == (first, last)


# ---------------------------------------------------------------------------
# HIDDEN_FROM_RULES
# ---------------------------------------------------------------------------


def _module_constant_pairs() -> set[tuple[str, str]]:
    return (
        {("paragraph", p) for p in pw.PARAGRAPH_UNOBSERVABLE_ON_CONTINUATION}
        | {("paragraph", p) for p in pw.PARAGRAPH_UNOBSERVABLE_SPACING_AFTER}
        | {("list", p) for p in pw.LIST_UNOBSERVABLE_ON_CONTINUATION}
        | {("row", p) for p in pw.ROW_UNOBSERVABLE_ON_CONTINUATION_TABLE}
    )


def test_derive_hidden_from_rules_covers_every_section10_constant() -> None:
    derived = hy.derive_hidden_from_rules()

    assert derived >= _module_constant_pairs()
    for node_type in ("paragraph", "list", "row"):
        assert any(t == node_type for t, _ in derived), node_type
    assert ("row", "/layout/repeat_header") in derived


def test_derive_hidden_from_rules_adds_the_four_cross_page_pointers() -> None:
    assert hy.derive_hidden_from_rules() - _module_constant_pairs() == {
        ("header", "/kind"),
        ("footer", "/kind"),
        ("section", "/break_type"),
        ("cell", "/row_span"),
    }


def test_hidden_from_rules_constant_equals_derivation() -> None:
    assert hy.HIDDEN_FROM_RULES == hy.derive_hidden_from_rules()


def test_derive_hidden_from_rules_reads_piece_withdrawal_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patched = pw.LIST_UNOBSERVABLE_ON_CONTINUATION | {"/marker/new_pointer"}
    monkeypatch.setattr(pw, "LIST_UNOBSERVABLE_ON_CONTINUATION", patched)

    assert ("list", "/marker/new_pointer") in hy.derive_hidden_from_rules()


# ---------------------------------------------------------------------------
# RuleView and leak fixtures
# ---------------------------------------------------------------------------


def test_rule_view_hides_withdrawn_and_cross_page_pointers() -> None:
    node = _paragraph(
        " Hi ",
        typed={
            "/layout/indent/first_line_mpt": 0,
            "/layout/spacing/after_mpt": 200,
            "/layout/alignment": "left",
        },
    )

    view = hy.rule_view(node)

    assert view.node_type == "paragraph"
    assert dict(view.typed) == {"/layout/alignment": "left"}
    assert (view.first_visible, view.last_visible) == ("H", "i")
    assert [child.node_type for child in view.children] == ["run"]
    assert not hasattr(view, "local_id")
    assert not hasattr(view, "text")


def test_rule_view_hides_story_kind_break_type_and_row_span() -> None:
    page = _parse(
        [
            "FRAGMENT_BEGIN",
            "SECTION_OPEN",  # 1
            "HEADER_OPEN",  # 2
            "HEADER_CLOSE",
            "FOOTER_OPEN",  # 4
            "FOOTER_CLOSE",
            "TABLE_OPEN",
            "ROW_OPEN",
            "CELL_OPEN",  # 8
            "PARAGRAPH_OPEN",
            "PARAGRAPH_CLOSE",
            "CELL_CLOSE",
            "ROW_CLOSE",
            "TABLE_CLOSE",
            "SECTION_CLOSE",
            "END_FRAGMENT",
        ],
        typed_by_anchor={
            1: {"/break_type": "next_page", "/page/width_mpt": 612000},
            2: {"/kind": "first"},
            4: {"/kind": "even"},
            8: {"/row_span": 2, "/grid_column": 0},
        },
    )

    section = hy.rule_view(page.sections[0])
    header, footer, table = section.children
    assert dict(section.typed) == {"/page/width_mpt": 612000}
    assert dict(header.typed) == {}
    assert dict(footer.typed) == {}
    assert dict(table.children[0].children[0].typed) == {"/grid_column": 0}


Rule = Callable[[hy.RuleView, hy.RuleView], bool]
View = Callable[[hy.PageNode], hy.RuleView]
Case = tuple[hy.PageNode, hy.PageNode, bool]


def _accuracy(rule: Rule, view: View, cases: Sequence[Case]) -> float:
    hits = sum(rule(view(a), view(b)) == gold for a, b, gold in cases)
    return hits / len(cases)


def _baseline(cases: Sequence[Case]) -> float:
    """B0 for a paragraph/row seam: never join."""
    return sum(not gold for _, _, gold in cases) / len(cases)


def _raw_text(node: hy.PageNode) -> str:
    own = node.text or ""
    return own + "".join(_raw_text(child) for child in node.children)


def _leaky_view(node: hy.PageNode) -> hy.RuleView:
    """What a leaking `rule_view` would hand a rule: every typed value and
    the raw edge scalars, whitespace included."""
    text = _raw_text(node)
    return hy.RuleView(
        node_type=node.node_type,
        children=tuple(_leaky_view(child) for child in node.children),
        typed=node.typed,
        first_visible=text[0] if text else None,
        last_visible=text[-1] if text else None,
    )


_FIRST_LINE = "/layout/indent/first_line_mpt"
_AFTER = "/layout/spacing/after_mpt"
_ALIGN = "/layout/alignment"

#: Gold-style paragraph seams: a continuation piece has the §10 pointers
#: withdrawn (`first_line_mpt` absent on B, `after_mpt` absent on A).
PARAGRAPH_CASES: list[Case] = [
    (
        _paragraph("a", typed={_FIRST_LINE: 0, _ALIGN: "left"}),
        _paragraph("b", typed={_ALIGN: "left"}),
        True,
    ),
    (
        _paragraph("a", typed={_FIRST_LINE: 0, _AFTER: 200, _ALIGN: "left"}),
        _paragraph("b", typed={_FIRST_LINE: 360, _ALIGN: "left"}),
        False,
    ),
    (
        _paragraph("a", typed={_FIRST_LINE: 720, _AFTER: 0, _ALIGN: "both"}),
        _paragraph("b", typed={_FIRST_LINE: 0, _ALIGN: "both"}),
        False,
    ),
]


def _first_line_leak_rule(a: hy.RuleView, b: hy.RuleView) -> bool:
    return _FIRST_LINE in a.typed and _FIRST_LINE not in b.typed


def test_rule_view_first_line_leak_rule_scores_baseline() -> None:
    assert _baseline(PARAGRAPH_CASES) < 1.0
    assert _accuracy(_first_line_leak_rule, hy.rule_view, PARAGRAPH_CASES) == _baseline(
        PARAGRAPH_CASES
    )
    assert _accuracy(_first_line_leak_rule, _leaky_view, PARAGRAPH_CASES) == 1.0


_REPEAT = "/layout/repeat_header"
_CANT_SPLIT = "/layout/cant_split"


def _table(*row_typed: Mapping[str, hy.TypedValue]) -> hy.PageNode:
    items: list[str | _T] = ["FRAGMENT_BEGIN", "SECTION_OPEN", "TABLE_OPEN"]
    typed_by_anchor: dict[int, Mapping[str, hy.TypedValue]] = {}
    for typed in row_typed:
        typed_by_anchor[len(items)] = typed
        items += ["ROW_OPEN", "CELL_OPEN", "PARAGRAPH_OPEN", "PARAGRAPH_CLOSE", "CELL_CLOSE",
                  "ROW_CLOSE"]
    items += ["TABLE_CLOSE", "SECTION_CLOSE", "END_FRAGMENT"]
    return _parse(items, typed_by_anchor=typed_by_anchor).sections[0].children[0]


#: Gold-style table seams: rows of a middle/last table piece have
#: `repeat_header` withdrawn (DEC-033 §10), rows of a whole table keep it.
TABLE_CASES: list[Case] = [
    (
        _table({_REPEAT: True, _CANT_SPLIT: False}, {_REPEAT: False, _CANT_SPLIT: False}),
        _table({_CANT_SPLIT: False}),
        True,
    ),
    (
        _table({_REPEAT: False, _CANT_SPLIT: False}),
        _table({_REPEAT: False, _CANT_SPLIT: True}),
        False,
    ),
    (
        _table({_REPEAT: True, _CANT_SPLIT: True}),
        _table({_REPEAT: True, _CANT_SPLIT: False}, {_REPEAT: False, _CANT_SPLIT: False}),
        False,
    ),
]


def _repeat_header_leak_rule(a: hy.RuleView, b: hy.RuleView) -> bool:
    return _REPEAT in a.children[-1].typed and _REPEAT not in b.children[0].typed


def test_rule_view_repeat_header_leak_rule_scores_baseline() -> None:
    assert _baseline(TABLE_CASES) < 1.0
    assert _accuracy(_repeat_header_leak_rule, hy.rule_view, TABLE_CASES) == _baseline(
        TABLE_CASES
    )
    assert _accuracy(_repeat_header_leak_rule, _leaky_view, TABLE_CASES) == 1.0


#: Gold-style text seams: the line-breaking space stays on the page-k
#: piece (45/45 split-run seams, design §2), including as a trailing
#: whitespace-only run; whole paragraphs end in visible text.
EDGE_SPACE_CASES: list[Case] = [
    (_paragraph("The quick "), _paragraph("brown fox."), True),
    (_paragraph("Hello", " "), _paragraph(" ", "world."), True),
    (_paragraph("Done."), _paragraph("Next one."), False),
    (_paragraph("End"), _paragraph("Start"), False),
    (_paragraph("Last"), _paragraph(" ", "first"), False),
]


def _ends_in_whitespace_rule(a: hy.RuleView, b: hy.RuleView) -> bool:
    return a.last_visible is not None and a.last_visible.isspace()


def _starts_in_whitespace_rule(a: hy.RuleView, b: hy.RuleView) -> bool:
    return b.first_visible is not None and b.first_visible.isspace()


def test_rule_view_edge_whitespace_rule_scores_baseline() -> None:
    assert _baseline(EDGE_SPACE_CASES) < 1.0
    assert _accuracy(_ends_in_whitespace_rule, hy.rule_view, EDGE_SPACE_CASES) == _baseline(
        EDGE_SPACE_CASES
    )
    assert _accuracy(_ends_in_whitespace_rule, _leaky_view, EDGE_SPACE_CASES) == 1.0


def test_rule_view_leading_whitespace_rule_never_fires() -> None:
    for a, b, _ in EDGE_SPACE_CASES:
        assert not _starts_in_whitespace_rule(hy.rule_view(a), hy.rule_view(b))
    assert any(
        _starts_in_whitespace_rule(_leaky_view(a), _leaky_view(b)) for a, b, _ in EDGE_SPACE_CASES
    )


# ---------------------------------------------------------------------------
# Import boundary (design §3)
# ---------------------------------------------------------------------------

_ASSEMBLY_DIR = REPO_ROOT / "src" / "unflatten" / "assembly"
_FORBIDDEN_STEMS = frozenset({"s1_train", "s2_train", "build_exact_text_cache"})
_FORBIDDEN_PACKAGES = ("src/unflatten/dataset/", "src/unflatten/eval/")


def _forbidden(sources: Mapping[str, str]) -> list[str]:
    return sorted(
        path
        for path in sources
        if path.startswith(_FORBIDDEN_PACKAGES) or Path(path).stem in _FORBIDDEN_STEMS
    )


def test_assembly_import_closure_excludes_dataset_eval_and_training() -> None:
    sources = import_closure.closure(sorted(_ASSEMBLY_DIR.glob("*.py")), prefix="unflatten.")

    assert "src/unflatten/model/piece_withdrawal.py" in sources
    assert _forbidden(sources) == []


def test_forbidden_check_catches_the_pre_move_import_path() -> None:
    # The revision-2 design imported `typed_property_targets`; the same
    # check must flag it, or the boundary test above proves nothing.
    seed = REPO_ROOT / "src" / "unflatten" / "model" / "typed_property_targets.py"
    sources = import_closure.closure([seed], prefix="unflatten.")

    assert "src/unflatten/dataset/targets/properties.py" in _forbidden(sources)


# ---------------------------------------------------------------------------
# from_model_hypothesis (AST Builder T2)
# ---------------------------------------------------------------------------


def _raw(
    items: Sequence[str | _T],
    typed: Sequence[Any] = (),
    *,
    terminated: bool = True,
) -> RawDocumentHypothesis:
    """What `decode_free_running` returns for `items`: every STRUCTURAL token
    (envelope and TEXT_BEGIN/TEXT_END included) replayed through the tracker
    at `event_index = its position`, exactly as the decode loop does."""
    tracker = NodeLifecycleTracker()
    for position, ref in enumerate(_refs(items)):
        if isinstance(ref, cv.StructuralRef):
            tracker.apply_structural_token(_NAME_BY_ID[ref.local_id], event_index=position)
    return RawDocumentHypothesis(
        accepted_tokens=tuple(_ids(items)),
        node_states=tracker.states,
        terminated=terminated,
        typed_predictions=tuple(typed),
    )


_NAME_BY_ID = {i: name for name, i in ev.TOKEN_IDS.items()}


def _decode_id(raw: RawDocumentHypothesis, position: int) -> str:
    return next(
        s.local_node_id
        for s in raw.node_states
        if position in (s.open_event_index, s.leaf_event_index)
    )


def _from_model(raw: RawDocumentHypothesis) -> hy.PageHypothesis:
    return hy.from_model_hypothesis(
        raw, vocabulary=VOCABULARY, tokenizer=TOKENIZER, page_input=PAGE
    )


def test_from_model_hypothesis_attaches_typed_predictions_by_token_position() -> None:
    """Run at position 7, table at 18 (TWO_SECTION_PAGE comments). The
    decode's own node ids are NOT the parser's (the decode tracker also sees
    the envelope and TEXT_BEGIN/TEXT_END), so the join is by position."""
    bare = _raw(TWO_SECTION_PAGE)
    run_id, table_id = _decode_id(bare, 7), _decode_id(bare, 18)
    typed = (
        TypedPrediction(run_id, "run", "/style/bold", "h.run", True),
        TypedPrediction(run_id, "run", "/style/size_mpt", "h.run", 11_000),
        TypedPrediction(table_id, "table", "/layout/alignment", "h.table", "center"),
    )

    page = _from_model(_raw(TWO_SECTION_PAGE, typed))

    assert page == _parse(
        TWO_SECTION_PAGE,
        {7: {"/style/bold": True, "/style/size_mpt": 11_000},
         18: {"/layout/alignment": "center"}},
    )


def test_from_model_hypothesis_without_typed_predictions_equals_the_parse() -> None:
    assert _from_model(_raw(TWO_SECTION_PAGE)) == _parse(TWO_SECTION_PAGE)


def test_from_model_hypothesis_unknown_node_id_raises() -> None:
    raw = _raw(TWO_SECTION_PAGE, (TypedPrediction("n999", "run", "/style/bold", "h", True),))
    with pytest.raises(hy.PageStreamInvalid, match="n999"):
        _from_model(raw)


def test_from_model_hypothesis_node_type_mismatch_raises() -> None:
    run_id = _decode_id(_raw(TWO_SECTION_PAGE), 7)
    raw = _raw(TWO_SECTION_PAGE, (TypedPrediction(run_id, "table", "/style/bold", "h", True),))
    with pytest.raises(hy.PageStreamInvalid, match="'table'"):
        _from_model(raw)


def test_from_model_hypothesis_duplicate_pointer_raises() -> None:
    run_id = _decode_id(_raw(TWO_SECTION_PAGE), 7)
    twice = tuple(
        TypedPrediction(run_id, "run", "/style/bold", "h", value) for value in (True, False)
    )
    with pytest.raises(hy.PageStreamInvalid, match="/style/bold"):
        _from_model(_raw(TWO_SECTION_PAGE, twice))


def test_from_model_hypothesis_unterminated_decode_raises() -> None:
    with pytest.raises(hy.PageStreamInvalid, match="terminated"):
        _from_model(_raw(TWO_SECTION_PAGE, terminated=False))

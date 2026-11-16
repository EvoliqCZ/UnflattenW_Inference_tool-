import copy
import unicodedata

import pytest

from unflatten.ast.canonicalizer import CanonicalizationError, canonicalize
from unflatten.ast.hashing import ast_hash


def _run(id_, text, language="en-US", color="#000000", bold=False):
    return {
        "id": id_,
        "type": "run",
        "text": text,
        "language": language,
        "style": {
            "font_family": "Arial",
            "font_size_mpt": 11000,
            "bold": bold,
            "italic": False,
            "underline": "none",
            "strike": False,
            "color": color,
            "highlight": None,
            "superscript": False,
            "subscript": False,
            "character_spacing_mpt": 0,
            "small_caps": False,
            "all_caps": False,
        },
    }


def _paragraph(id_, children, role="body", heading_level=None, background=None):
    return {
        "id": id_,
        "type": "paragraph",
        "semantic": {"role": role, "heading_level": heading_level, "outline_level": None},
        "layout": {
            "alignment": "left",
            "direction": "ltr",
            "indent": {"left_mpt": 0, "right_mpt": 0, "first_line_mpt": 0, "hanging_mpt": 0},
            "spacing": {"before_mpt": 0, "after_mpt": 0, "line": {"mode": "single"}},
            "pagination": {
                "keep_with_next": False,
                "keep_lines": False,
                "page_break_before": False,
                "widow_control": True,
            },
            "tabs": [],
        },
        "style": {"background": background, "borders": None},
        "children": children,
    }


def _section(id_, children):
    return {
        "id": id_,
        "type": "section",
        "break_type": "next_page",
        "page": {"width_mpt": 595276, "height_mpt": 841890, "orientation": "portrait"},
        "margins": {
            "top_mpt": 72000,
            "right_mpt": 72000,
            "bottom_mpt": 72000,
            "left_mpt": 72000,
            "gutter_mpt": 0,
        },
        "columns": {"count": 1, "equal_width": True, "gap_mpt": 18000, "widths_mpt": []},
        "children": children,
    }


def _doc(sections, resources=None, metadata=None):
    return {
        "ast_version": "0.1.0",
        "profile": "word-core-v0",
        "document_id": "doc",
        "metadata": metadata or {"language": "en-US", "title": None, "subject": None},
        "resources": resources or [],
        "sections": sections,
        "extensions": {},
    }


def test_canonicalize_minimal_valid_doc_succeeds(minimal_valid_doc_copy):
    result = canonicalize(minimal_valid_doc_copy)
    assert result["sections"][0]["id"] == "n1"
    assert result["sections"][0]["children"][0]["id"] == "n2"
    assert result["sections"][0]["children"][0]["children"][0]["id"] == "n3"


def test_canonicalize_raises_on_schema_invalid_doc():
    with pytest.raises(CanonicalizationError):
        canonicalize({"not": "a valid document"})


def test_canonicalize_raises_on_table_grid_overlap():
    doc = _doc(
        [
            _section(
                "s1",
                [
                    {
                        "id": "t1",
                        "type": "table",
                        "layout": {
                            "alignment": "left",
                            "indent_mpt": 0,
                            "width": {"mode": "auto", "value_mpt": 0},
                            "autofit": True,
                        },
                        "columns": [{"width_mpt": 100000}, {"width_mpt": 100000}],
                        "style": {"background": None, "borders": None},
                        "rows": [
                            {
                                "id": "r1",
                                "type": "row",
                                "layout": {
                                    "height": {"mode": "auto", "value_mpt": 0},
                                    "repeat_header": False,
                                    "cant_split": False,
                                },
                                "cells": [
                                    {
                                        "id": "c1",
                                        "type": "cell",
                                        "grid_column": 0,
                                        "col_span": 2,
                                        "row_span": 1,
                                        "layout": {
                                            "width_mpt": 100000,
                                            "padding": {
                                                "top_mpt": 0,
                                                "right_mpt": 0,
                                                "bottom_mpt": 0,
                                                "left_mpt": 0,
                                            },
                                            "vertical_alignment": "top",
                                            "text_direction": "horizontal",
                                        },
                                        "style": {"background": None, "borders": None},
                                        "children": [_paragraph("p1", [_run("r1a", "a")])],
                                    },
                                    {
                                        "id": "c2",
                                        "type": "cell",
                                        "grid_column": 1,
                                        "col_span": 1,
                                        "row_span": 1,
                                        "layout": {
                                            "width_mpt": 100000,
                                            "padding": {
                                                "top_mpt": 0,
                                                "right_mpt": 0,
                                                "bottom_mpt": 0,
                                                "left_mpt": 0,
                                            },
                                            "vertical_alignment": "top",
                                            "text_direction": "horizontal",
                                        },
                                        "style": {"background": None, "borders": None},
                                        "children": [_paragraph("p2", [_run("r2a", "b")])],
                                    },
                                ],
                            }
                        ],
                    }
                ],
            )
        ]
    )
    with pytest.raises(CanonicalizationError):
        canonicalize(doc)


def test_canonicalize_nfc_normalizes_run_text():
    nfd_text = unicodedata.normalize("NFD", "café")
    doc = _doc([_section("s1", [_paragraph("p1", [_run("r1", nfd_text)])])])
    result = canonicalize(doc)
    canonical_text = result["sections"][0]["children"][0]["children"][0]["text"]
    assert canonical_text == unicodedata.normalize("NFC", nfd_text)
    assert unicodedata.is_normalized("NFC", canonical_text)


def test_canonicalize_merges_adjacent_equivalent_runs():
    doc = _doc([_section("s1", [_paragraph("p1", [_run("r1", "Hello "), _run("r2", "world.")])])])
    result = canonicalize(doc)
    runs = result["sections"][0]["children"][0]["children"]
    assert len(runs) == 1
    assert runs[0]["text"] == "Hello world."


def test_canonicalize_merged_run_text_is_nfc_across_the_join():
    # Two individually-NFC runs whose join is not canonically ordered:
    # shadda (ccc 33) at the end of one run, fatha (ccc 30) at the start
    # of the next. Concatenation alone leaves them out of canonical order.
    left, right = "أمّ", "َا"
    assert unicodedata.is_normalized("NFC", left)
    assert unicodedata.is_normalized("NFC", right)
    doc = _doc([_section("s1", [_paragraph("p1", [_run("r1", left), _run("r2", right)])])])
    result = canonicalize(doc)
    text = result["sections"][0]["children"][0]["children"][0]["text"]
    assert unicodedata.is_normalized("NFC", text)
    assert text == unicodedata.normalize("NFC", left + right)


def test_canonicalize_is_idempotent_for_merged_combining_marks():
    left, right = "أمّ", "َا"
    doc = _doc([_section("s1", [_paragraph("p1", [_run("r1", left), _run("r2", right)])])])
    once = canonicalize(doc)
    assert canonicalize(once) == once


def test_canonicalize_does_not_merge_runs_with_different_style():
    doc = _doc(
        [
            _section(
                "s1",
                [_paragraph("p1", [_run("r1", "Hello "), _run("r2", "world.", bold=True)])],
            )
        ]
    )
    result = canonicalize(doc)
    runs = result["sections"][0]["children"][0]["children"]
    assert len(runs) == 2


def test_canonicalize_renumbers_ids_in_preorder():
    doc = _doc(
        [
            _section("zzz", [_paragraph("aaa", [_run("bbb", "x"), _run("ccc", "y", bold=True)])]),
        ]
    )
    result = canonicalize(doc)
    section = result["sections"][0]
    paragraph = section["children"][0]
    assert section["id"] == "n1"
    assert paragraph["id"] == "n2"
    assert paragraph["children"][0]["id"] == "n3"
    assert paragraph["children"][1]["id"] == "n4"


def test_canonicalize_sorts_resource_registry_by_id():
    resources = [
        {
            "id": "res_b",
            "kind": "image",
            "mime_type": "image/png",
            "sha256": "b" * 64,
            "path": "b.png",
            "pixel_width": 10,
            "pixel_height": 10,
        },
        {
            "id": "res_a",
            "kind": "image",
            "mime_type": "image/png",
            "sha256": "a" * 64,
            "path": "a.png",
            "pixel_width": 10,
            "pixel_height": 10,
        },
    ]
    doc = _doc([_section("s1", [])], resources=resources)
    result = canonicalize(doc)
    assert [r["id"] for r in result["resources"]] == ["res_a", "res_b"]


def test_canonicalize_normalizes_lowercase_hex_colors_end_to_end():
    # ERRATA-001 E4: schema accepts case-insensitive hex colors on input;
    # the canonicalizer is the sole uppercase normalizer, and its output
    # is always canonical-form uppercase.
    doc = _doc(
        [
            _section(
                "s1",
                [_paragraph("p1", [_run("r1", "x", color="#abcdef")], background="#00ff00")],
            )
        ]
    )
    result = canonicalize(doc)
    paragraph = result["sections"][0]["children"][0]
    run = paragraph["children"][0]
    assert run["style"]["color"] == "#ABCDEF"
    assert paragraph["style"]["background"] == "#00FF00"
    _assert_all_colors_canonical_uppercase(result)


def test_canonicalize_keeps_already_uppercase_hex_colors():
    doc = _doc(
        [
            _section(
                "s1",
                [_paragraph("p1", [_run("r1", "x", color="#ABCDEF")], background="#00FF00")],
            )
        ]
    )
    result = canonicalize(doc)
    paragraph = result["sections"][0]["children"][0]
    run = paragraph["children"][0]
    assert run["style"]["color"] == "#ABCDEF"
    assert paragraph["style"]["background"] == "#00FF00"


def test_canon_color_helper_normalizes_lowercase_hex_to_uppercase():
    from unflatten.ast.canonicalizer import _canon_color

    assert _canon_color("#abcdef") == "#ABCDEF"
    assert _canon_color("auto") == "auto"
    assert _canon_color(None) is None


def _assert_all_colors_canonical_uppercase(value) -> None:
    """Canonical-form check: no hex color string in canonicalize() output
    may contain lowercase hex digits (RFC001 §35 step 7 / ERRATA-001 E4).
    """
    if isinstance(value, str) and value.startswith("#"):
        assert value == value.upper(), f"non-canonical (non-uppercase) color: {value!r}"
    elif isinstance(value, dict):
        for item in value.values():
            _assert_all_colors_canonical_uppercase(item)
    elif isinstance(value, list):
        for item in value:
            _assert_all_colors_canonical_uppercase(item)


def test_canonicalize_leaves_auto_color_untouched():
    doc = _doc([_section("s1", [_paragraph("p1", [_run("r1", "x", color="auto")])])])
    result = canonicalize(doc)
    assert result["sections"][0]["children"][0]["children"][0]["style"]["color"] == "auto"


def test_canonicalize_rewrites_internal_anchor_hyperlink_refs():
    # hyperlink is a word-visual-v1 node (RFC001 §42.2, DEC-003 §1)
    doc = _doc(
        [
            _section(
                "s1",
                [
                    _paragraph(
                        "target_p",
                        [
                            {
                                "id": "link1",
                                "type": "hyperlink",
                                "target": {"kind": "internal_anchor", "uri": "#target_p"},
                                "children": [_run("linktext", "go")],
                            }
                        ],
                    )
                ],
            )
        ]
    )
    doc["profile"] = "word-visual-v1"
    result = canonicalize(doc)
    paragraph = result["sections"][0]["children"][0]
    hyperlink = paragraph["children"][0]
    assert hyperlink["target"]["uri"] == f"#{paragraph['id']}"


def test_canonicalize_raises_on_dangling_internal_anchor():
    doc = _doc(
        [
            _section(
                "s1",
                [
                    _paragraph(
                        "p1",
                        [
                            {
                                "id": "link1",
                                "type": "hyperlink",
                                "target": {"kind": "internal_anchor", "uri": "#does_not_exist"},
                                "children": [_run("linktext", "go")],
                            }
                        ],
                    )
                ],
            )
        ]
    )
    with pytest.raises(CanonicalizationError):
        canonicalize(doc)


def test_canonicalize_raises_rather_than_silently_repoint_dangling_anchor():
    # ERRATA-001 E5: the dangling anchor's literal target "#n1" does not
    # resolve to any *original* node id, but "n1" happens to be the id the
    # preorder renumbering pass would assign to the section itself. A
    # naive "does the literal resolve in the renumbered tree" check would
    # silently repoint this anchor at the wrong node; canonicalize() must
    # instead detect the dangling reference via the old-id -> new-id map
    # and raise, never leave (or coincidentally "heal") the literal.
    doc = _doc(
        [
            _section(
                "section_original_id",
                [
                    _paragraph(
                        "p1",
                        [
                            {
                                "id": "link1",
                                "type": "hyperlink",
                                "target": {"kind": "internal_anchor", "uri": "#n1"},
                                "children": [_run("linktext", "go")],
                            }
                        ],
                    )
                ],
            )
        ]
    )
    with pytest.raises(CanonicalizationError) as exc_info:
        canonicalize(doc)
    assert any(
        issue.code in ("BROKEN_INTERNAL_ANCHOR", "DANGLING_INTERNAL_ANCHOR")
        for issue in exc_info.value.issues
    )


def test_canonicalize_is_idempotent(minimal_valid_doc_copy):
    once = canonicalize(minimal_valid_doc_copy)
    twice = canonicalize(copy.deepcopy(once))
    assert once == twice


def test_canonicalize_idempotent_hash_stable_across_two_runs(minimal_valid_doc_copy):
    once = canonicalize(minimal_valid_doc_copy)
    twice = canonicalize(copy.deepcopy(once))
    assert ast_hash(once) == ast_hash(twice)


# ---------------------------------------------------- ERRATA-002 pass-through --


def _load_valid_corpus_fixture(name):
    import json
    from pathlib import Path

    path = Path(__file__).parent.parent / "data" / "corpus" / "valid" / f"{name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_canonicalize_passes_custom_text_marker_through_unchanged():
    doc = _load_valid_corpus_fixture("custom_text_list")
    original_marker = copy.deepcopy(doc["sections"][0]["children"][0]["marker"])
    canonical = canonicalize(doc)
    assert canonical["sections"][0]["children"][0]["marker"] == original_marker


def test_canonicalize_passes_percent_width_through_unchanged():
    doc = _load_valid_corpus_fixture("percent_width_table")
    original_width = copy.deepcopy(doc["sections"][0]["children"][0]["layout"]["width"])
    canonical = canonicalize(doc)
    assert canonical["sections"][0]["children"][0]["layout"]["width"] == original_width


def test_canonicalize_hash_stable_for_errata_fixtures():
    for name in ("custom_text_list", "percent_width_table"):
        doc = _load_valid_corpus_fixture(name)
        assert ast_hash(canonicalize(doc)) == ast_hash(
            canonicalize(_load_valid_corpus_fixture(name))
        )


def test_canonicalize_nfc_normalizes_custom_text_marker():
    """ERRATA-002 E6 + panel must_fix: marker.text is visible text and
    must be NFC-normalized by canonicalization."""
    import copy
    import json
    import pathlib
    import unicodedata

    from unflatten.ast.canonicalizer import canonicalize

    corpus = pathlib.Path(__file__).parent.parent / "data" / "corpus" / "valid"
    fixture = corpus / "custom_text_list.json"
    doc = copy.deepcopy(json.loads(fixture.read_text()))

    def find_list(nodes):
        for node in nodes:
            if node.get("type") == "list":
                return node
            child = find_list(node.get("children", []) or [])
            if child is not None:
                return child
        return None

    lst = find_list([c for s_ in doc["sections"] for c in s_["children"]])
    assert lst is not None and lst["marker"]["format"] == "custom_text"
    nfd_text = unicodedata.normalize("NFD", "é)")
    assert nfd_text != unicodedata.normalize("NFC", nfd_text)
    lst["marker"]["text"] = nfd_text

    out = canonicalize(doc)
    out_lst = find_list([c for s_ in out["sections"] for c in s_["children"]])
    assert out_lst["marker"]["text"] == unicodedata.normalize("NFC", "é)")
    assert out_lst["marker"]["text"] != nfd_text

"""Regression: check_limits runs on unvalidated input and must not crash."""

from unflatten.renderer.limits import check_limits


def _base(sections):
    return {
        "ast_version": "0.1.0",
        "profile": "word-core-v0",
        "document_id": "doc",
        "metadata": {},
        "resources": [],
        "sections": sections,
        "extensions": {},
    }


def test_check_limits_non_dict_children_do_not_raise():
    ast = _base(
        [
            {
                "id": "n1",
                "type": "section",
                "children": [
                    "not-a-dict",
                    42,
                    None,
                    {"id": "n2", "type": "paragraph", "children": [7]},
                ],
            },
            "not-a-section",
        ]
    )
    issues = check_limits(
        ast,
        max_nodes=100_000,
        max_tree_depth=64,
        max_text_chars=10_000_000,
        max_table_cells=100_000,
    )
    assert issues == []


def test_check_limits_non_dict_rows_and_cells_do_not_raise():
    ast = _base(
        [
            {
                "id": "n1",
                "type": "section",
                "children": [
                    {
                        "id": "n2",
                        "type": "table",
                        "rows": [
                            "bogus",
                            {
                                "id": "n3",
                                "type": "row",
                                "cells": [1, {"id": "n4", "type": "cell", "children": [None]}],
                            },
                        ],
                    }
                ],
            }
        ]
    )
    issues = check_limits(
        ast,
        max_nodes=100_000,
        max_tree_depth=64,
        max_text_chars=10_000_000,
        max_table_cells=100_000,
    )
    assert issues == []


def _paragraph(node_id: str, text: str) -> dict:
    return {
        "id": node_id,
        "type": "paragraph",
        "children": [{"id": f"{node_id}r", "type": "run", "text": text}],
    }


def test_check_limits_counts_header_and_footer_story_content():
    """`word-visual-v1` moves content out of `section.children`. A limit
    that only walked the body would let a 10 MB header through.
    """
    ast = _base(
        [
            {
                "id": "n1",
                "type": "section",
                "children": [],
                "headers": [
                    {
                        "id": "h1",
                        "type": "header",
                        "kind": "default",
                        "children": [_paragraph("hp1", "x" * 50)],
                    }
                ],
                "footers": [
                    {
                        "id": "f1",
                        "type": "footer",
                        "kind": "default",
                        "children": [_paragraph("fp1", "y" * 50)],
                    }
                ],
            }
        ]
    )
    issues = check_limits(
        ast, max_nodes=100_000, max_tree_depth=64, max_text_chars=99, max_table_cells=100_000
    )
    assert [i.code for i in issues] == ["LIMIT_EXCEEDED_TEXT_CHARS"]
    assert issues[0].details["actual"] == 100


def test_check_limits_counts_footnote_and_endnote_bodies():
    ast = _base([{"id": "n1", "type": "section", "children": []}])
    ast["footnotes"] = [
        {"id": "fn1", "type": "footnote", "children": [_paragraph("fnp1", "z" * 40)]}
    ]
    ast["endnotes"] = [
        {"id": "en1", "type": "endnote", "children": [_paragraph("enp1", "w" * 40)]}
    ]
    issues = check_limits(
        ast, max_nodes=100_000, max_tree_depth=64, max_text_chars=79, max_table_cells=100_000
    )
    assert [i.code for i in issues] == ["LIMIT_EXCEEDED_TEXT_CHARS"]
    assert issues[0].details["actual"] == 80

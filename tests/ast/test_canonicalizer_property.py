"""Idempotency property test (RFC001 §35): canonicalize(canonicalize(d)) == canonicalize(d)
over generated variants of the minimal document — shuffled resources, NFD
text, split equivalent runs, lowercase-but-still-schema-valid colors are
not reachable (schema forbids them), so variants cover the reachable
degrees of freedom: resource order, NFD text, and split-but-equivalent runs.
"""

from __future__ import annotations

import copy
import unicodedata

from hypothesis import given, settings
from hypothesis import strategies as st

from unflatten.ast.canonicalizer import canonicalize


def _resource(id_: str) -> dict:
    return {
        "id": id_,
        "kind": "image",
        "mime_type": "image/png",
        "sha256": "a" * 64,
        "path": f"{id_}.png",
        "pixel_width": 10,
        "pixel_height": 10,
    }


def _run(id_: str, text: str) -> dict:
    return {
        "id": id_,
        "type": "run",
        "text": text,
        "language": "en-US",
        "style": {
            "font_family": "Arial",
            "font_size_mpt": 11000,
            "bold": False,
            "italic": False,
            "underline": "none",
            "strike": False,
            "color": "#000000",
            "highlight": None,
            "superscript": False,
            "subscript": False,
            "character_spacing_mpt": 0,
            "small_caps": False,
            "all_caps": False,
        },
    }


def _make_doc(resource_order: list[str], run_texts: list[str], nfd: bool) -> dict:
    texts = [unicodedata.normalize("NFD", t) if nfd else t for t in run_texts]
    runs = [_run(f"r{i}", t) for i, t in enumerate(texts)]
    paragraph = {
        "id": "p1",
        "type": "paragraph",
        "semantic": {"role": "body", "heading_level": None, "outline_level": None},
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
        "style": {"background": None, "borders": None},
        "children": runs,
    }
    section = {
        "id": "s1",
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
        "children": [paragraph],
    }
    return {
        "ast_version": "0.1.0",
        "profile": "word-core-v0",
        "document_id": "doc",
        "metadata": {"language": "en-US", "title": None, "subject": None},
        "resources": [_resource(rid) for rid in resource_order],
        "sections": [section],
        "extensions": {},
    }


@given(
    resource_order=st.permutations(["res_c", "res_a", "res_b"]),
    run_texts=st.lists(
        st.text(alphabet="cafeéàü ", min_size=1, max_size=5), min_size=1, max_size=4
    ),
    nfd=st.booleans(),
)
@settings(max_examples=50)
def test_canonicalize_is_idempotent_over_document_variants(resource_order, run_texts, nfd):
    doc = _make_doc(list(resource_order), run_texts, nfd)
    once = canonicalize(doc)
    twice = canonicalize(copy.deepcopy(once))
    assert once == twice

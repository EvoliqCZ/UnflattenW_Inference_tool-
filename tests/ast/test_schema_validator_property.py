import copy

from hypothesis import given, settings
from hypothesis import strategies as st

from unflatten.ast.schema_validator import validate_schema


def _minimal_doc_with_text(text: str) -> dict:
    return {
        "ast_version": "0.1.0",
        "profile": "word-core-v0",
        "document_id": "doc",
        "metadata": {"language": None, "title": None, "subject": None},
        "resources": [],
        "sections": [
            {
                "id": "n1",
                "type": "section",
                "break_type": "continuous",
                "page": {"width_mpt": 595276, "height_mpt": 841890, "orientation": "portrait"},
                "margins": {
                    "top_mpt": 0,
                    "right_mpt": 0,
                    "bottom_mpt": 0,
                    "left_mpt": 0,
                    "gutter_mpt": 0,
                },
                "columns": {"count": 1, "equal_width": True, "gap_mpt": 0, "widths_mpt": []},
                "children": [
                    {
                        "id": "n2",
                        "type": "paragraph",
                        "semantic": {"role": "body", "heading_level": None, "outline_level": None},
                        "layout": {
                            "alignment": "left",
                            "direction": "ltr",
                            "indent": {
                                "left_mpt": 0,
                                "right_mpt": 0,
                                "first_line_mpt": 0,
                                "hanging_mpt": 0,
                            },
                            "spacing": {
                                "before_mpt": 0,
                                "after_mpt": 0,
                                "line": {"mode": "single"},
                            },
                            "pagination": {
                                "keep_with_next": False,
                                "keep_lines": False,
                                "page_break_before": False,
                                "widow_control": True,
                            },
                            "tabs": [],
                        },
                        "style": {"background": None, "borders": None},
                        "children": [
                            {
                                "id": "n3",
                                "type": "run",
                                "text": text,
                                "language": None,
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
                        ],
                    }
                ],
            }
        ],
        "extensions": {},
    }


@settings(max_examples=50)
@given(st.text(max_size=200))
def test_validate_schema_generated_minimal_doc_with_arbitrary_run_text_stays_valid(text):
    doc = _minimal_doc_with_text(text)
    issues = validate_schema(doc)
    assert issues == []


@settings(max_examples=25)
@given(st.integers(min_value=1, max_value=10_000_000))
def test_validate_schema_generated_minimal_doc_with_arbitrary_font_size_stays_valid(font_size):
    doc = _minimal_doc_with_text("sample")
    doc = copy.deepcopy(doc)
    doc["sections"][0]["children"][0]["children"][0]["style"]["font_size_mpt"] = font_size
    issues = validate_schema(doc)
    assert issues == []

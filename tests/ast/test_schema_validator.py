from unflatten.ast.schema_validator import validate_schema


def test_validate_schema_minimal_valid_doc_returns_no_issues(minimal_valid_doc_copy):
    issues = validate_schema(minimal_valid_doc_copy)
    assert issues == []


def test_validate_schema_missing_required_root_field_returns_error(minimal_valid_doc_copy):
    del minimal_valid_doc_copy["sections"]
    issues = validate_schema(minimal_valid_doc_copy)
    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].code == "SCHEMA_VALIDATION_ERROR"
    assert issues[0].path == "/"


def test_validate_schema_unknown_enum_value_returns_error(minimal_valid_doc_copy):
    minimal_valid_doc_copy["sections"][0]["break_type"] = "not_a_real_break"
    issues = validate_schema(minimal_valid_doc_copy)
    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert "/sections/0/break_type" in issues[0].path


def test_validate_schema_length_as_float_returns_error(minimal_valid_doc_copy):
    minimal_valid_doc_copy["sections"][0]["margins"]["top_mpt"] = 72000.5
    issues = validate_schema(minimal_valid_doc_copy)
    assert len(issues) == 1
    assert issues[0].severity == "error"


def test_validate_schema_length_as_string_returns_error(minimal_valid_doc_copy):
    minimal_valid_doc_copy["sections"][0]["margins"]["top_mpt"] = "72000"
    issues = validate_schema(minimal_valid_doc_copy)
    assert len(issues) == 1


def test_validate_schema_additional_property_returns_error(minimal_valid_doc_copy):
    minimal_valid_doc_copy["unexpected_field"] = "surprise"
    issues = validate_schema(minimal_valid_doc_copy)
    assert len(issues) == 1
    assert issues[0].severity == "error"


def test_validate_schema_bad_color_format_returns_error(minimal_valid_doc_copy):
    run = minimal_valid_doc_copy["sections"][0]["children"][0]["children"][0]
    run["style"]["color"] = "red"
    issues = validate_schema(minimal_valid_doc_copy)
    assert len(issues) == 1


def test_validate_schema_lowercase_color_is_valid(minimal_valid_doc_copy):
    # ERRATA-001 E4: input schema is case-insensitive for hex colors;
    # uppercase is a canonical-form requirement of the canonicalizer, not
    # of schema validation (real DOCX imports carry lowercase w:color).
    run = minimal_valid_doc_copy["sections"][0]["children"][0]["children"][0]
    run["style"]["color"] = "#ffffff"
    issues = validate_schema(minimal_valid_doc_copy)
    assert issues == []


def test_validate_schema_auto_color_is_valid(minimal_valid_doc_copy):
    run = minimal_valid_doc_copy["sections"][0]["children"][0]["children"][0]
    run["style"]["color"] = "auto"
    issues = validate_schema(minimal_valid_doc_copy)
    assert issues == []


def test_validate_schema_never_raises_on_non_document_input():
    issues = validate_schema({"totally": "unrelated"})
    assert len(issues) > 0
    assert all(issue.severity == "error" for issue in issues)


def test_validate_schema_never_raises_on_non_dict_input():
    issues = validate_schema("not a dict")
    assert len(issues) == 1
    assert issues[0].severity == "error"


# --------------------------------------------------- ERRATA-002 fixtures --

import copy  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

_VALID_CORPUS = Path(__file__).parent.parent / "data" / "corpus" / "valid"


def _load_corpus(name: str) -> dict:
    with open(_VALID_CORPUS / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def custom_text_list_doc() -> dict:
    return copy.deepcopy(_load_corpus("custom_text_list"))


@pytest.fixture
def percent_width_table_doc() -> dict:
    return copy.deepcopy(_load_corpus("percent_width_table"))


# ------------------------------------------------------------ ERRATA-002 E6 --


def test_validate_schema_custom_text_marker_with_text_is_valid(custom_text_list_doc):
    assert validate_schema(custom_text_list_doc) == []


def test_validate_schema_custom_text_marker_without_text_returns_error(custom_text_list_doc):
    marker = custom_text_list_doc["sections"][0]["children"][0]["marker"]
    del marker["text"]
    issues = validate_schema(custom_text_list_doc)
    assert issues != []
    assert all(i.severity == "error" for i in issues)


def test_validate_schema_custom_text_marker_with_empty_text_returns_error(custom_text_list_doc):
    custom_text_list_doc["sections"][0]["children"][0]["marker"]["text"] = ""
    issues = validate_schema(custom_text_list_doc)
    assert issues != []


def test_validate_schema_non_custom_marker_with_text_returns_error(custom_text_list_doc):
    marker = custom_text_list_doc["sections"][0]["children"][0]["marker"]
    marker["format"] = "bullet"
    issues = validate_schema(custom_text_list_doc)
    assert issues != []


# ------------------------------------------------------------ ERRATA-002 E7 --


def test_validate_schema_percent_width_with_value_ppm_is_valid(percent_width_table_doc):
    assert validate_schema(percent_width_table_doc) == []


def test_validate_schema_percent_width_with_value_mpt_returns_error(percent_width_table_doc):
    layout = percent_width_table_doc["sections"][0]["children"][0]["layout"]
    layout["width"] = {"mode": "percent", "value_mpt": 500000}
    issues = validate_schema(percent_width_table_doc)
    assert issues != []


def test_validate_schema_percent_width_out_of_range_returns_error(percent_width_table_doc):
    layout = percent_width_table_doc["sections"][0]["children"][0]["layout"]
    layout["width"]["value_ppm"] = 1000001
    issues = validate_schema(percent_width_table_doc)
    assert issues != []


def test_validate_schema_absolute_width_with_value_ppm_returns_error(percent_width_table_doc):
    layout = percent_width_table_doc["sections"][0]["children"][0]["layout"]
    layout["width"] = {"mode": "absolute", "value_ppm": 500000}
    issues = validate_schema(percent_width_table_doc)
    assert issues != []


def test_validate_schema_absolute_width_with_value_mpt_is_valid(percent_width_table_doc):
    layout = percent_width_table_doc["sections"][0]["children"][0]["layout"]
    layout["width"] = {"mode": "absolute", "value_mpt": 200000}
    assert validate_schema(percent_width_table_doc) == []


def test_validate_schema_auto_width_without_value_is_valid(percent_width_table_doc):
    layout = percent_width_table_doc["sections"][0]["children"][0]["layout"]
    layout["width"] = {"mode": "auto"}
    assert validate_schema(percent_width_table_doc) == []


def test_validate_schema_auto_width_with_legacy_zero_value_mpt_is_valid(percent_width_table_doc):
    # Pre-ERRATA-002 corpus encodes auto as {"mode": "auto", "value_mpt": 0};
    # tolerated so existing fixtures stay valid without migration.
    layout = percent_width_table_doc["sections"][0]["children"][0]["layout"]
    layout["width"] = {"mode": "auto", "value_mpt": 0}
    assert validate_schema(percent_width_table_doc) == []


def test_validate_schema_auto_width_with_nonzero_value_mpt_returns_error(percent_width_table_doc):
    layout = percent_width_table_doc["sections"][0]["children"][0]["layout"]
    layout["width"] = {"mode": "auto", "value_mpt": 100}
    issues = validate_schema(percent_width_table_doc)
    assert issues != []

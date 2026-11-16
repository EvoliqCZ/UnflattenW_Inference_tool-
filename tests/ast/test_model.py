import pytest

from unflatten.ast.model import Document


def test_document_from_dict_to_dict_round_trips_minimal_valid_doc(minimal_valid_doc):
    doc = Document.from_dict(minimal_valid_doc)
    assert doc.to_dict() == minimal_valid_doc


def test_document_from_dict_exposes_typed_fields(minimal_valid_doc):
    doc = Document.from_dict(minimal_valid_doc)
    assert doc.ast_version == "0.1.0"
    assert doc.profile == "word-core-v0"
    assert doc.document_id == "doc"
    assert doc.sections[0]["type"] == "section"


def test_document_is_frozen(minimal_valid_doc):
    doc = Document.from_dict(minimal_valid_doc)
    try:
        doc.document_id = "other"
        raised = False
    except AttributeError:
        raised = True
    assert raised


def test_document_from_dict_missing_required_field_raises_value_error_naming_field(
    minimal_valid_doc,
):
    del minimal_valid_doc["document_id"]
    with pytest.raises(ValueError, match="document_id"):
        Document.from_dict(minimal_valid_doc)

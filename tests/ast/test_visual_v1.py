"""Profile extension `word-visual-v1` — schema, semantics, canonicalization.

Covers DEC-003: header/footer stories, footnotes with `note_ref`, and
anchored (floating) images. The fixtures live in
`tests/data/corpus/visual_v1/` rather than `tests/data/corpus/valid/`
because the latter is also consumed by the renderer, model and dataset
suites, which do not implement this profile yet (F2/F4/F5).
"""

import copy
import json
from pathlib import Path

import pytest

from unflatten.ast.canonicalizer import CanonicalizationError, canonicalize
from unflatten.ast.hashing import ast_hash
from unflatten.ast.model import Document
from unflatten.ast.schema_validator import validate_schema
from unflatten.ast.semantic_validator import validate_semantics

CORPUS_DIR = Path(__file__).parent.parent / "data" / "corpus" / "visual_v1"
VALID_DIR = CORPUS_DIR / "valid"
INVALID_DIR = CORPUS_DIR / "invalid"

_BLOCKING = {"error", "fatal"}


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _valid_fixture_paths() -> list[Path]:
    return sorted(VALID_DIR.glob("*.json"))


def _invalid_fixture_paths() -> list[Path]:
    return sorted(p for p in INVALID_DIR.glob("*.json") if not p.name.endswith(".expected.json"))


@pytest.fixture
def header_doc() -> dict:
    return _load(VALID_DIR / "header_footer_default.json")


@pytest.fixture
def footnote_doc() -> dict:
    return _load(VALID_DIR / "footnote_with_note_ref.json")


@pytest.fixture
def floating_doc() -> dict:
    return _load(VALID_DIR / "floating_image_anchored.json")


@pytest.fixture
def text_frame_doc(header_doc, floating_doc) -> dict:
    doc = copy.deepcopy(header_doc)
    doc["profile"] = "word-visual-v1.1"
    frame_paragraph = copy.deepcopy(doc["sections"][0]["children"][0])
    frame_paragraph["id"] = "tfp1"
    frame_paragraph["children"][0]["id"] = "tfr1"
    frame_paragraph["children"][0]["text"] = "Frame text"
    anchor = copy.deepcopy(
        floating_doc["sections"][0]["children"][0]["children"][1]["anchor"]
    )
    doc["sections"][0]["children"].append(
        {"id": "tf1", "type": "text_frame", "anchor": anchor, "children": [frame_paragraph]}
    )
    return doc


def _codes(issues) -> set[str]:
    return {i.code for i in issues}


# --- corpus -----------------------------------------------------------------


@pytest.mark.parametrize("path", _valid_fixture_paths(), ids=lambda p: p.stem)
def test_valid_visual_v1_fixture_passes_schema_and_semantic_validation(path):
    doc = _load(path)
    assert [i for i in validate_schema(doc) if i.severity in _BLOCKING] == []
    assert [i for i in validate_semantics(doc) if i.severity in _BLOCKING] == []


@pytest.mark.parametrize("path", _valid_fixture_paths(), ids=lambda p: p.stem)
def test_valid_visual_v1_fixture_canonicalizes_and_hashes_stably(path):
    doc = _load(path)
    first = canonicalize(doc)
    second = canonicalize(json.loads(json.dumps(doc)))
    assert ast_hash(first) == ast_hash(second)


@pytest.mark.parametrize("path", _invalid_fixture_paths(), ids=lambda p: p.stem)
def test_invalid_visual_v1_fixture_reports_expected_issue_code_and_path_pairs(path):
    doc = _load(path)
    expected = _load(path.with_suffix("").with_suffix(".expected.json"))
    schema_issues = validate_schema(doc)
    semantic_issues = validate_semantics(doc) if not schema_issues else []
    pairs = {(i.code, i.path) for i in schema_issues} | {
        (i.code, i.path) for i in semantic_issues
    }
    for issue in expected["expected_issues"]:
        assert (issue["code"], issue["path"]) in pairs, f"{path.name}: {pairs}"


# --- schema shape -----------------------------------------------------------


def test_schema_rejects_word_visual_v1_node_under_unknown_profile_value(header_doc):
    header_doc["profile"] = "word-extended-v2"
    assert validate_schema(header_doc) != []


def test_schema_rejects_unknown_story_kind(header_doc):
    header_doc["sections"][0]["headers"][0]["kind"] = "odd"
    assert validate_schema(header_doc) != []


def test_schema_rejects_header_containing_a_header(header_doc):
    header = header_doc["sections"][0]["headers"][0]
    header["children"].append(copy.deepcopy(header))
    assert validate_schema(header_doc) != []


def test_schema_rejects_floating_image_without_anchor(floating_doc):
    del floating_doc["sections"][0]["children"][0]["children"][1]["anchor"]
    assert validate_schema(floating_doc) != []


def test_schema_rejects_square_wrap_without_side(floating_doc):
    anchor = floating_doc["sections"][0]["children"][0]["children"][1]["anchor"]
    del anchor["wrap"]["side"]
    assert validate_schema(floating_doc) != []


def test_schema_rejects_wrap_none_carrying_a_side(floating_doc):
    anchor = floating_doc["sections"][0]["children"][1]["children"][0]["anchor"]
    anchor["wrap"]["side"] = "left"
    assert validate_schema(floating_doc) != []


def test_schema_rejects_anchor_align_carrying_an_offset(floating_doc):
    anchor = floating_doc["sections"][0]["children"][0]["children"][1]["anchor"]
    anchor["horizontal"]["offset_mpt"] = 1000
    assert validate_schema(floating_doc) != []


def test_schema_accepts_optional_header_footer_margin_distances(header_doc):
    header_doc["sections"][0]["margins"]["header_mpt"] = 36000
    header_doc["sections"][0]["margins"]["footer_mpt"] = 36000
    assert validate_schema(header_doc) == []


def test_schema_accepts_text_frame_with_reused_anchor_contract(text_frame_doc):
    assert validate_schema(text_frame_doc) == []


def test_schema_rejects_text_frame_without_anchor(text_frame_doc):
    del text_frame_doc["sections"][0]["children"][1]["anchor"]
    assert validate_schema(text_frame_doc) != []


def test_schema_rejects_text_frame_specific_style_surface(text_frame_doc):
    text_frame_doc["sections"][0]["children"][1]["style"] = {}
    assert validate_schema(text_frame_doc) != []


# --- semantics --------------------------------------------------------------


def test_v1_nodes_are_rejected_in_the_word_core_v0_profile(footnote_doc, floating_doc):
    for doc in (footnote_doc, floating_doc):
        doc["profile"] = "word-core-v0"
        assert "V1_FEATURE_IN_V0_PROFILE" in _codes(validate_semantics(doc))


def test_empty_story_arrays_are_not_a_v0_profile_violation(header_doc):
    header_doc["profile"] = "word-core-v0"
    header_doc["sections"][0]["headers"] = []
    header_doc["sections"][0]["footers"] = []
    assert "V1_FEATURE_IN_V0_PROFILE" not in _codes(validate_semantics(header_doc))


def test_column_break_in_a_story_is_an_invalid_parent_child(header_doc):
    header_doc["sections"][0]["headers"][0]["children"].append(
        {"id": "cb1", "type": "column_break"}
    )
    assert "INVALID_PARENT_CHILD" in _codes(validate_semantics(header_doc))


def test_page_break_in_a_footnote_is_an_invalid_parent_child(footnote_doc):
    footnote_doc["footnotes"][0]["children"].append({"id": "pb1", "type": "page_break"})
    assert "INVALID_PARENT_CHILD" in _codes(validate_semantics(footnote_doc))


def test_duplicate_footer_kind_in_one_section_is_reported(header_doc):
    footers = header_doc["sections"][0]["footers"]
    clone = copy.deepcopy(footers[0])
    clone["id"] = "f2"
    clone["children"][0]["id"] = "fp2"
    clone["children"][0]["children"][0]["id"] = "fr2"
    footers.append(clone)
    assert "DUPLICATE_STORY_KIND" in _codes(validate_semantics(header_doc))


def test_distinct_story_kinds_in_one_section_are_accepted(header_doc):
    clone = copy.deepcopy(header_doc["sections"][0]["headers"][0])
    clone.update({"id": "h2", "kind": "first"})
    clone["children"][0]["id"] = "hp2"
    clone["children"][0]["children"][0]["id"] = "hr2"
    header_doc["sections"][0]["headers"].append(clone)
    assert [i for i in validate_semantics(header_doc) if i.severity in _BLOCKING] == []


def test_duplicate_node_id_between_header_and_body_is_reported(header_doc):
    header_doc["sections"][0]["headers"][0]["children"][0]["id"] = "p1"
    assert "DUPLICATE_NODE_ID" in _codes(validate_semantics(header_doc))


def test_floating_image_with_unknown_resource_ref_is_reported(floating_doc):
    floating_doc["sections"][0]["children"][0]["children"][1]["resource_ref"] = "res_ffff"
    assert "BROKEN_RESOURCE_REF" in _codes(validate_semantics(floating_doc))


def test_note_ref_to_a_missing_footnote_is_reported(footnote_doc):
    footnote_doc["sections"][0]["children"][0]["children"][1]["note_id"] = "nope"
    assert "BROKEN_NOTE_REF" in _codes(validate_semantics(footnote_doc))


def test_unreferenced_footnote_is_reported(footnote_doc):
    footnote_doc["sections"][0]["children"][0]["children"].pop()
    assert "ORPHAN_FOOTNOTE" in _codes(validate_semantics(footnote_doc))


def test_note_ref_inside_a_footnote_is_reported(footnote_doc):
    footnote_doc["footnotes"][0]["children"][0]["children"].append(
        {"id": "nr2", "type": "note_ref", "note_id": "fn1", "marker_text": "1"}
    )
    assert "NOTE_REF_IN_NOTE" in _codes(validate_semantics(footnote_doc))


def test_internal_anchor_into_a_header_resolves(header_doc):
    header_doc["sections"][0]["children"][0]["children"].append(
        {
            "id": "hl1",
            "type": "hyperlink",
            "target": {"kind": "internal_anchor", "uri": "#hp1"},
            "children": [copy.deepcopy(header_doc["sections"][0]["children"][0]["children"][0])],
        }
    )
    header_doc["sections"][0]["children"][0]["children"][-1]["children"][0]["id"] = "r9"
    assert "BROKEN_INTERNAL_ANCHOR" not in _codes(validate_semantics(header_doc))


def test_text_frame_requires_word_visual_v1_1(text_frame_doc):
    text_frame_doc["profile"] = "word-visual-v1"
    assert "TEXT_FRAME_REQUIRES_VISUAL_V1_1" in _codes(validate_semantics(text_frame_doc))


def test_text_frame_inside_a_header_is_accepted(text_frame_doc):
    section = text_frame_doc["sections"][0]
    frame = section["children"].pop()
    section["headers"][0]["children"].append(frame)
    assert [i for i in validate_semantics(text_frame_doc) if i.severity in _BLOCKING] == []


def test_nested_text_frame_is_rejected_at_any_depth(text_frame_doc):
    outer = text_frame_doc["sections"][0]["children"][1]
    inner = copy.deepcopy(outer)
    inner["id"] = "tf2"
    outer["children"] = [inner]

    assert validate_schema(text_frame_doc) == []
    assert "NESTED_TEXT_FRAME_UNSUPPORTED" in _codes(validate_semantics(text_frame_doc))


# --- canonicalization -------------------------------------------------------


def test_canonicalize_numbers_stories_before_body_and_footnotes_last(footnote_doc):
    footnote_doc["sections"][0]["headers"] = [
        {
            "id": "h1",
            "type": "header",
            "kind": "default",
            "children": [copy.deepcopy(footnote_doc["sections"][0]["children"][0])],
        }
    ]
    header_para = footnote_doc["sections"][0]["headers"][0]["children"][0]
    header_para["id"] = "hp1"
    header_para["children"] = [header_para["children"][0]]
    header_para["children"][0]["id"] = "hr1"

    result = canonicalize(footnote_doc)
    section = result["sections"][0]
    assert section["id"] == "n1"
    assert section["headers"][0]["id"] == "n2"  # story before body
    assert section["headers"][0]["children"][0]["id"] == "n3"
    assert section["children"][0]["id"] == "n5"  # body paragraph
    assert result["footnotes"][0]["id"] == "n8"  # footnotes last


def test_canonicalize_sorts_stories_into_default_first_even_order(header_doc):
    headers = header_doc["sections"][0]["headers"]
    for hid, kind in (("h2", "even"), ("h3", "first")):
        clone = copy.deepcopy(headers[0])
        clone.update({"id": hid, "kind": kind})
        clone["children"][0]["id"] = f"{hid}p"
        clone["children"][0]["children"][0]["id"] = f"{hid}r"
        headers.append(clone)
    result = canonicalize(header_doc)
    assert [h["kind"] for h in result["sections"][0]["headers"]] == ["default", "first", "even"]


def test_canonicalize_is_order_insensitive_for_story_arrays(header_doc):
    other = copy.deepcopy(header_doc)
    clone = copy.deepcopy(header_doc["sections"][0]["headers"][0])
    clone.update({"id": "h2", "kind": "first"})
    clone["children"][0]["id"] = "hp2"
    clone["children"][0]["children"][0]["id"] = "hr2"
    header_doc["sections"][0]["headers"].append(clone)
    other["sections"][0]["headers"].insert(0, copy.deepcopy(clone))
    assert ast_hash(canonicalize(header_doc)) == ast_hash(canonicalize(other))


def test_canonicalize_rewrites_note_id_through_the_renumbering_map(footnote_doc):
    result = canonicalize(footnote_doc)
    ref = result["sections"][0]["children"][0]["children"][1]
    assert ref["note_id"] == result["footnotes"][0]["id"]


def test_canonicalize_orders_footnotes_by_first_reference(footnote_doc):
    second = copy.deepcopy(footnote_doc["footnotes"][0])
    second["id"] = "fn2"
    second["children"][0]["id"] = "fnp2"
    second["children"][0]["children"][0]["id"] = "fnr2"
    footnote_doc["footnotes"].insert(0, second)
    para = footnote_doc["sections"][0]["children"][0]
    para["children"].append({"id": "nr2", "type": "note_ref", "note_id": "fn2",
                             "marker_text": "2"})
    result = canonicalize(footnote_doc)
    refs = [c["note_id"] for c in result["sections"][0]["children"][0]["children"]
            if c["type"] == "note_ref"]
    assert refs == [fn["id"] for fn in result["footnotes"]]


def test_canonicalize_refuses_a_dangling_note_id(footnote_doc):
    footnote_doc["sections"][0]["children"][0]["children"][1]["note_id"] = "gone"
    with pytest.raises(CanonicalizationError) as exc:
        canonicalize(footnote_doc)
    assert any(i.code == "BROKEN_NOTE_REF" for i in exc.value.issues)


def test_canonicalize_nfc_normalizes_marker_text_and_floating_alt_text(
    footnote_doc, floating_doc
):
    footnote_doc["sections"][0]["children"][0]["children"][1]["marker_text"] = "á"
    assert canonicalize(footnote_doc)["sections"][0]["children"][0]["children"][1][
        "marker_text"
    ] == "á"

    floating_doc["sections"][0]["children"][0]["children"][1]["alt_text"] = "á"
    assert canonicalize(floating_doc)["sections"][0]["children"][0]["children"][1][
        "alt_text"
    ] == "á"


def test_canonicalize_merges_adjacent_equivalent_runs_inside_a_story(header_doc):
    children = header_doc["sections"][0]["headers"][0]["children"][0]["children"]
    twin = copy.deepcopy(children[0])
    twin["id"] = "hr2"
    twin["text"] = " continued"
    children.append(twin)
    result = canonicalize(header_doc)
    story_children = result["sections"][0]["headers"][0]["children"][0]["children"]
    assert len(story_children) == 1
    assert story_children[0]["text"] == "Running header continued"


def test_canonicalize_traverses_and_renumbers_text_frame_children(text_frame_doc):
    frame = text_frame_doc["sections"][0]["children"][1]
    twin = copy.deepcopy(frame["children"][0]["children"][0])
    twin["id"] = "tfr2"
    twin["text"] = " continued"
    frame["children"][0]["children"].append(twin)

    result = canonicalize(text_frame_doc)
    canonical_frame = result["sections"][0]["children"][1]
    paragraph = canonical_frame["children"][0]
    assert [canonical_frame["id"], paragraph["id"], paragraph["children"][0]["id"]] == [
        f"n{index}"
        for index in range(
            int(canonical_frame["id"][1:]), int(canonical_frame["id"][1:]) + 3
        )
    ]
    assert paragraph["children"][0]["text"] == "Frame text continued"
    assert canonicalize(result) == result


# --- endnotes, hyperlinks, nested tables, section settings ------------------


@pytest.fixture
def note_doc() -> dict:
    return _load(VALID_DIR / "endnote_and_footnote.json")


@pytest.fixture
def nested_table_doc() -> dict:
    return _load(VALID_DIR / "nested_table.json")


@pytest.fixture
def hyperlink_doc() -> dict:
    return _load(VALID_DIR / "hyperlinks_internal_and_external.json")


@pytest.fixture
def settings_doc() -> dict:
    return _load(VALID_DIR / "advanced_section_settings.json")


def test_note_ref_resolves_into_either_note_array(note_doc):
    assert [i for i in validate_semantics(note_doc) if i.severity in _BLOCKING] == []


def test_unreferenced_endnote_is_reported(note_doc):
    note_doc["sections"][0]["children"][0]["children"].pop()
    codes = _codes(validate_semantics(note_doc))
    assert "ORPHAN_FOOTNOTE" in codes


def test_note_ref_inside_an_endnote_is_reported(note_doc):
    note_doc["endnotes"][0]["children"][0]["children"].append(
        {"id": "nr3", "type": "note_ref", "note_id": "fn1", "marker_text": "1"}
    )
    assert "NOTE_REF_IN_NOTE" in _codes(validate_semantics(note_doc))


def test_endnotes_are_numbered_after_footnotes(note_doc):
    result = canonicalize(note_doc)
    footnote_index = int(result["footnotes"][0]["id"][1:])
    endnote_index = int(result["endnotes"][0]["id"][1:])
    assert footnote_index < endnote_index


def test_canonicalize_rewrites_note_id_into_the_endnote_array(note_doc):
    result = canonicalize(note_doc)
    refs = [c["note_id"] for c in result["sections"][0]["children"][0]["children"]
            if c["type"] == "note_ref"]
    assert refs == [result["footnotes"][0]["id"], result["endnotes"][0]["id"]]


def test_endnotes_are_ordered_by_first_reference(note_doc):
    second = copy.deepcopy(note_doc["endnotes"][0])
    second["id"] = "en2"
    second["children"][0]["id"] = "enp2"
    second["children"][0]["children"][0]["id"] = "enr2"
    note_doc["endnotes"].insert(0, second)
    note_doc["sections"][0]["children"][0]["children"].append(
        {"id": "nr3", "type": "note_ref", "note_id": "en2", "marker_text": "ii"}
    )
    result = canonicalize(note_doc)
    assert [n["id"] for n in result["endnotes"]] == sorted(
        (n["id"] for n in result["endnotes"]), key=lambda i: int(i[1:])
    )
    first_ref = result["sections"][0]["children"][0]["children"][3]["note_id"]
    assert first_ref == result["endnotes"][0]["id"]


def test_nested_table_is_allowed_in_word_visual_v1(nested_table_doc):
    assert [i for i in validate_semantics(nested_table_doc) if i.severity in _BLOCKING] == []
    canonicalize(nested_table_doc)  # must not raise


def test_nested_table_is_still_rejected_in_word_core_v0(nested_table_doc):
    nested_table_doc["profile"] = "word-core-v0"
    assert "NESTED_TABLE_UNSUPPORTED" in _codes(validate_semantics(nested_table_doc))


def test_hyperlink_is_a_first_class_node_of_the_profile(hyperlink_doc):
    assert validate_schema(hyperlink_doc) == []
    assert [i for i in validate_semantics(hyperlink_doc) if i.severity in _BLOCKING] == []


def test_hyperlink_requires_the_v1_profile(hyperlink_doc):
    hyperlink_doc["profile"] = "word-core-v0"
    assert "V1_FEATURE_IN_V0_PROFILE" in _codes(validate_semantics(hyperlink_doc))


def test_advanced_section_settings_pass_schema_and_canonicalize(settings_doc):
    assert validate_schema(settings_doc) == []
    canonicalize(settings_doc)


def test_schema_rejects_unknown_line_number_restart(settings_doc):
    settings_doc["sections"][0]["line_numbering"]["restart"] = "new_column"
    assert validate_schema(settings_doc) != []


def test_schema_rejects_unknown_section_vertical_alignment(settings_doc):
    settings_doc["sections"][0]["vertical_alignment"] = "middle"
    assert validate_schema(settings_doc) != []


def test_page_border_colors_are_canonicalized_to_uppercase(settings_doc):
    settings_doc["sections"][0]["page_borders"]["sides"]["top"]["color"] = "#00ff00"
    result = canonicalize(settings_doc)
    assert result["sections"][0]["page_borders"]["sides"]["top"]["color"] == "#00FF00"


def test_section_carries_no_node_definitions_of_its_own(header_doc):
    """Regression: a `$defs` node body must never be pasted into
    `section.properties`. A section is not a field and cannot contain one
    at that level, so such a key would validate documents that must be
    refused."""
    from unflatten.ast.schema_validator import _load_schema

    assert set(_load_schema()["$defs"]["section"]["properties"]) == {
        "id", "type", "break_type", "page", "margins", "columns",
        "vertical_alignment", "page_numbering", "line_numbering", "page_borders",
        "headers", "footers", "children",
    }
    header_doc["sections"][0]["field"] = {"id": "x", "type": "field", "field_type": "page"}
    assert validate_schema(header_doc) != []


def test_section_settings_are_optional(header_doc):
    section = header_doc["sections"][0]
    assert not {"vertical_alignment", "line_numbering", "page_borders"} & set(section)
    assert validate_schema(header_doc) == []


# --- fields (whitelist) and page numbering ----------------------------------


@pytest.fixture
def field_doc() -> dict:
    return _load(VALID_DIR / "page_number_field_in_footer.json")


def test_page_and_num_pages_fields_pass_schema_and_semantics(field_doc):
    assert validate_schema(field_doc) == []
    assert [i for i in validate_semantics(field_doc) if i.severity in _BLOCKING] == []


def test_schema_admits_only_the_whitelisted_field_types(field_doc):
    """The whitelist is the enum — a field type outside it is unrepresentable.

    `{PAGE, NUMPAGES}` per the owner's decision on the 1500-document
    census: they are 34.5 % and 9.1 % of the corpus and the only two that
    produce the every-page-identical failure. TOC/PAGEREF/STYLEREF stay
    [OPEN]; RFC001 §30 forbids inventing a field code, so an unlisted
    type must fail the schema rather than pass as free text.
    """
    for unlisted in ("toc", "pageref", "styleref", "date", "PAGE", ""):
        doc = copy.deepcopy(field_doc)
        doc["sections"][0]["footers"][0]["children"][0]["children"][1]["field_type"] = unlisted
        assert validate_schema(doc) != [], unlisted


def test_field_carries_no_result_text(field_doc):
    """A page number differs on every page, so a cached result would be a
    second source of truth that is wrong on all pages but one — which is
    the defect this node exists to remove."""
    field = field_doc["sections"][0]["footers"][0]["children"][0]["children"][1]
    assert set(field) == {"id", "type", "field_type"}
    field["result_text"] = "3"
    assert validate_schema(field_doc) != []


def test_field_requires_the_v1_profile(field_doc):
    field_doc["profile"] = "word-core-v0"
    assert "V1_FEATURE_IN_V0_PROFILE" in _codes(validate_semantics(field_doc))


def test_field_is_numbered_in_preorder_inside_its_story(field_doc):
    result = canonicalize(field_doc)
    footer_runs = result["sections"][0]["footers"][0]["children"][0]["children"]
    ids = [int(child["id"][1:]) for child in footer_runs]
    assert ids == sorted(ids)
    assert [c["type"] for c in footer_runs] == ["run", "field", "run", "field"]


def test_page_numbering_passes_schema_and_canonicalizes(field_doc):
    assert validate_schema(_load(VALID_DIR / "page_numbering_roman_restart.json")) == []
    canonicalize(_load(VALID_DIR / "page_numbering_roman_restart.json"))


def test_schema_rejects_unknown_page_number_format(field_doc):
    field_doc["sections"][0]["page_numbering"]["format"] = "ordinal"
    assert validate_schema(field_doc) != []


def test_page_numbering_start_may_be_null_to_continue_the_previous_section(field_doc):
    field_doc["sections"][0]["page_numbering"]["start"] = None
    assert validate_schema(field_doc) == []


# --- identity: the extension is additive ------------------------------------


def test_header_free_document_hash_is_unchanged_by_the_extension(minimal_valid_doc):
    """The pinned golden for the minimal document (tests/data/hash_vectors.json)."""
    assert ast_hash(minimal_valid_doc) == (
        "sha256:" + _load(Path(__file__).parent.parent / "data" / "hash_vectors.json")[0][
            "expected_ast_hash"
        ].split(":", 1)[1]
    )


def test_empty_story_keys_would_change_the_hash_of_every_document(minimal_valid_doc):
    """Why the new keys are optional-when-empty, not required (DEC-003 §3).

    Required keys would put `headers: []`/`footers: []`/`footnotes: []`
    into the canonical JSON of documents that have none, moving 100 % of
    `ast_hash` values — and with them every `document_bundle_id`,
    `fragment_id` and `target_bundle_hash` (RFC003-ERRATA-001 E4).
    """
    with_empty_keys = copy.deepcopy(minimal_valid_doc)
    with_empty_keys["footnotes"] = []
    with_empty_keys["sections"][0]["headers"] = []
    with_empty_keys["sections"][0]["footers"] = []
    assert ast_hash(with_empty_keys) != ast_hash(minimal_valid_doc)


def test_profile_capability_version_change_moves_identity(header_doc):
    """Named reason: PROFILE_CAPABILITY_VERSION_CHANGE (ERRATA-004 E5)."""
    upgraded = copy.deepcopy(header_doc)
    upgraded["profile"] = "word-visual-v1.1"
    assert ast_hash(upgraded) != ast_hash(header_doc)


def test_representing_text_frame_content_moves_identity(text_frame_doc):
    """Named reason: TEXT_FRAME_CONTENT_REPRESENTED (ERRATA-004 E5)."""
    without_frame = copy.deepcopy(text_frame_doc)
    without_frame["sections"][0]["children"].pop()
    assert ast_hash(text_frame_doc) != ast_hash(without_frame)


def test_document_model_round_trips_footnotes(footnote_doc):
    assert Document.from_dict(footnote_doc).to_dict() == footnote_doc


def test_document_model_omits_footnotes_when_absent(minimal_valid_doc):
    assert "footnotes" not in Document.from_dict(minimal_valid_doc).to_dict()

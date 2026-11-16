"""The gold-arm harness (design §2.6, plan P7), moved out of
`tests/model/test_ast_builder.py:36-131` so both the unit test and the
`builder_chain` CLI import the SAME construction -- a fixture that
round-trips in the unit test round-trips here too, by construction, never
by two hand-kept copies drifting apart.

`GOLD_FIXTURES` is the explicit, pinned list of all 25 P0 corpus fixtures
(`valid/` 13, `visual_v1/valid/` 11, `visual_v1_1/valid/` 1) -- never a
glob at runtime, so adding a fixture to the corpus without adding it here
does not silently enlarge the gold arm (design §2.6)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from unflatten.assembly import hypothesis as hy
from unflatten.ast.canonicalizer import canonicalize
from unflatten.dataset.targets.events import derive_event_streams
from unflatten.dataset.targets.properties import emit_property_labels
from unflatten.model import composite_vocab as cv
from unflatten.model import events_vocab as ev
from unflatten.model.typed_property_pointer_map import repeated_containers

CORPUS = Path(__file__).parents[4] / "tests" / "data" / "corpus"

#: 20 hash-equal round trips, plus 3 image fixtures (`RESOURCE_SLOT_UNRESOLVED`)
#: and 2 named canonicalization refusals (design §2.6 rev 2 MEASURED).
GOLD_FIXTURES: tuple[str, ...] = (
    "valid/custom_text_list.json",
    "valid/empty_visual_paragraph.json",
    "valid/minimal_paragraph.json",
    "valid/mixed_run_styles.json",
    "valid/multi_paragraph_with_page_break.json",
    "valid/ordered_unordered_nested_list.json",
    "valid/paragraph_with_tabs_and_line_break.json",
    "valid/percent_width_table.json",
    "valid/simple_table_2x2.json",
    "valid/table_with_colspan_rowspan.json",
    "valid/two_section_first_ends_empty_paragraph.json",
    "valid/two_section_two_column.json",
    "visual_v1/valid/advanced_section_settings.json",
    "visual_v1/valid/endnote_and_footnote.json",
    "visual_v1/valid/footnote_with_note_ref.json",
    "visual_v1/valid/header_footer_default.json",
    "visual_v1/valid/nested_table.json",
    "visual_v1/valid/page_number_field_in_footer.json",
    "visual_v1/valid/page_numbering_roman_restart.json",
    "visual_v1_1/valid/text_frame_with_paragraph.json",
    "valid/inline_image_with_resource.json",
    "visual_v1/valid/floating_image_anchored.json",
    "visual_v1/valid/header_with_image_and_table.json",
    "visual_v1/valid/title_page_and_even_headers.json",
    "visual_v1/valid/hyperlinks_internal_and_external.json",
)


def _vocabulary() -> cv.CompositeVocabulary:
    allowlist = cv.build_qwen_text_allowlist(
        tokenizer_artifact_set_hash="sha256:" + "ab" * 32,
        native_vocab_size=8,
        classify=lambda native_id: None,
    )
    return cv.make_composite_vocabulary(
        tokenizer_artifact_set_hash="sha256:" + "ab" * 32, allowlist_payload=allowlist
    )


class _NoQwenTokenizer:
    """Gold text is carried as byte frames, so no Qwen id is ever decoded."""

    def encode(self, text: str) -> list[int]:  # pragma: no cover - never called
        raise AssertionError("never encodes")

    def decode(self, ids: Any) -> str:  # pragma: no cover - never called
        raise AssertionError("gold text carries no Qwen ids")


VOCABULARY = _vocabulary()
TOKENIZER = _NoQwenTokenizer()


def _at(node: Mapping[str, Any], pointer: str) -> Any:
    value: Any = node
    for part in pointer.strip("/").split("/"):
        value = value[part]
    return value


def _gold_typed(node: Mapping[str, Any], labels: Mapping[str, Any]) -> dict[str, Any]:
    """The node's emitted labels, plus each repeated container's element
    count at the container pointer (the cardinality channel, design §4)."""
    typed = {f["node_property_pointer"]: f["value"] for f in labels.get(node["id"], {}).get(
        "fields", ()
    )}
    for node_type, container in repeated_containers():
        if node_type == node["type"]:
            typed[container] = len(_at(node, container))
    return typed


def _nodes_by_id(ast: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if "id" in value and "type" in value:
                out[value["id"]] = value
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(ast)
    return out


def gold_page(canonical: Mapping[str, Any], ordinal: int = 0) -> hy.PageHypothesis:
    streams = derive_event_streams(dict(canonical))
    spans = {s["text_span_id"]: s["text"] for s in streams["text"]["spans"]}
    labels = emit_property_labels(dict(canonical))
    nodes = _nodes_by_id(canonical)
    refs: list[cv.TokenRef] = [cv.StructuralRef(local_id=ev.TOKEN_IDS["FRAGMENT_BEGIN"])]
    typed_by_anchor: dict[int, dict[str, Any]] = {}
    for event in streams["structural"]["events"]:
        node = nodes.get(event["node_id"])
        node_type = node["type"] if node is not None else "document"
        if node_type == "document":
            continue
        if event["op"] == "open":
            typed = _gold_typed(node, labels)
            if typed:
                typed_by_anchor[len(refs)] = typed
            name = ev.LEAF_TOKEN.get(node_type) or ev.OPEN_TOKEN[node_type]
            refs.append(cv.StructuralRef(local_id=ev.TOKEN_IDS[name]))
        elif event["op"] == "text":
            refs.append(cv.StructuralRef(local_id=ev.TOKEN_IDS["TEXT_BEGIN"]))
            for scalar in spans[event["text_span_id"]]:
                refs.append(cv.ByteControlRef(kind="byte_begin"))
                refs.extend(cv.ByteValueRef(value=b) for b in scalar.encode("utf-8"))
                refs.append(cv.ByteControlRef(kind="byte_end"))
            refs.append(cv.StructuralRef(local_id=ev.TOKEN_IDS["TEXT_END"]))
        elif node_type not in ev.LEAF_TOKEN:
            refs.append(cv.StructuralRef(local_id=ev.TOKEN_IDS[ev.CLOSE_TOKEN[node_type]]))
    refs.append(cv.StructuralRef(local_id=ev.TOKEN_IDS["END_FRAGMENT"]))
    return hy.from_token_ids(
        [VOCABULARY.pack(ref) for ref in refs],
        vocabulary=VOCABULARY,
        tokenizer=TOKENIZER,
        typed_by_anchor=typed_by_anchor,
        page_input=hy.PageInput(page_ordinal=ordinal, width_px=2480, height_px=3508),
    )


def load(relative: str) -> dict[str, Any]:
    return canonicalize(json.loads((CORPUS / relative).read_text()))

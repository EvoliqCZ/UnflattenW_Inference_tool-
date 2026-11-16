from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import torch
from lxml import etree
from tests.assembly.test_hypothesis import _T, TOKENIZER, VOCABULARY, _raw
from tests.service.test_docx_stage import _fake_runtime
from tests.service.test_docx_stage_g5_real_chain import _typed_values_for

from unflatten.eval.s2_gate import sealed
from unflatten.model.typed_decode import TypedPrediction
from unflatten.service import docx_stage as ds
from unflatten.service.model_runtime import seal_diagnostic_prediction


def _prediction(node_id: str, pointer: str, value: object) -> dict:
    return {
        "head_id": "test",
        "local_node_id": node_id,
        "node_type": "paragraph",
        "pointer": pointer,
        "value": value,
    }


def _page(*, confidence: int = 980_000, mode: str = "single") -> sealed.PageDecodeResult:
    return sealed.PageDecodeResult(
        document_bundle_id="synthetic",
        outcome="terminated",
        accepted_tokens=(1, 2, 3),
        node_states=(
            sealed.NodeStateProjection("s", "opened_node", "section", None, 1),
            sealed.NodeStateProjection("p", "opened_node", "paragraph", "s", 2),
        ),
        typed_predictions=(
            _prediction("p", "/layout/spacing/before_mpt", 2000),
            _prediction("p", "/layout/spacing/after_mpt", 3000),
            _prediction("p", "/layout/spacing/line/mode", mode),
            *(
                ()
                if mode == "single"
                else (_prediction("p", "/layout/spacing/line/value_mpt", 9000),)
            ),
            {
                "head_id": "test",
                "local_node_id": "s",
                "node_type": "section",
                "pointer": "/vertical_alignment",
                "value": "center",
            },
        ),
        region_predictions=(
            sealed.SealedRegionPrediction(
                "p",
                "paragraph",
                "abstain_not_visible",
                (),
                confidence,
            ),
        ),
    )


def _values(page: sealed.PageDecodeResult) -> dict:
    return {(p["local_node_id"], p["pointer"]): p["value"] for p in page.typed_predictions}


def test_invisible_empty_paragraph_does_not_inherit_a_full_default_font_line() -> None:
    page = _page()
    before = _values(page)
    edited, counts = ds._collapse_invisible_empty_paragraphs(page)

    values = _values(edited)
    assert values["p", "/layout/spacing/line/mode"] == "exact"
    assert values["p", "/layout/spacing/line/value_mpt"] == 50  # one OOXML twip
    assert values["p", "/layout/spacing/before_mpt"] == 2000
    assert values["p", "/layout/spacing/after_mpt"] == 3000
    assert values["s", "/vertical_alignment"] == "center"
    assert edited.node_states == page.node_states
    assert edited.accepted_tokens == page.accepted_tokens
    assert edited.region_predictions == page.region_predictions
    assert _values(page) == before  # the published IR remains untouched
    assert counts == {"invisible_empty_paragraph_line_collapsed": 1}


@pytest.mark.parametrize("mode", ["exact", "at_least"])
def test_explicit_empty_paragraph_line_height_is_preserved(mode: str) -> None:
    page = _page(mode=mode)
    edited, counts = ds._collapse_invisible_empty_paragraphs(page)
    assert edited is page
    assert counts == {}


def test_uncertain_empty_paragraph_is_preserved() -> None:
    page = _page(confidence=949_999)
    edited, counts = ds._collapse_invisible_empty_paragraphs(page)
    assert edited is page
    assert counts == {}


@pytest.mark.parametrize("child_type", ["run", "inline_image", "tab", "field"])
def test_paragraph_with_any_child_is_preserved_even_when_alignment_abstains(child_type) -> None:
    page = _page()
    page = replace(
        page,
        node_states=(
            *page.node_states,
            sealed.NodeStateProjection(
                "content",
                "opened_node",
                child_type,
                "p",
                3,
            ),
        ),
    )
    edited, counts = ds._collapse_invisible_empty_paragraphs(page)
    assert edited is page
    assert counts == {}


def test_empty_paragraph_in_a_table_is_preserved() -> None:
    page = _page()
    page = replace(
        page,
        node_states=(
            page.node_states[0],
            sealed.NodeStateProjection("c", "opened_node", "cell", "s", 2),
            replace(page.node_states[1], parent_local_node_id="c"),
        ),
    )
    edited, counts = ds._collapse_invisible_empty_paragraphs(page)
    assert edited is page
    assert counts == {}


def test_visible_empty_paragraph_is_preserved() -> None:
    page = _page()
    page = replace(
        page,
        region_predictions=(
            sealed.SealedRegionPrediction(
                "p",
                "paragraph",
                "region",
                (sealed.SealedRegion(0, (100, 100, 200, 200), "complete", 980_000),),
                980_000,
            ),
        ),
    )
    edited, counts = ds._collapse_invisible_empty_paragraphs(page)
    assert edited is page
    assert counts == {}


@pytest.mark.parametrize("multi", [False, True])
def test_diagnostic_service_renders_repaired_lines_and_reports_them(tmp_path: Path, multi) -> None:
    raw = _raw(
        [
            "FRAGMENT_BEGIN",
            "SECTION_OPEN",
            "PARAGRAPH_OPEN",
            "PARAGRAPH_CLOSE",
            "PARAGRAPH_OPEN",
            "RUN_OPEN",
            _T("Preserve this text."),
            "RUN_CLOSE",
            "PARAGRAPH_CLOSE",
            "SECTION_CLOSE",
            "END_FRAGMENT",
        ]
    )
    predictions = []
    for node in raw.node_states:
        values = _typed_values_for(node.node_type)
        if node.node_type == "section":
            values.update(
                {
                    "/page/width_mpt": 595_300,
                    "/page/height_mpt": 841_900,
                    "/margins/top_mpt": 54_000,
                    "/margins/bottom_mpt": 54_000,
                    "/margins/left_mpt": 54_000,
                    "/margins/right_mpt": 54_000,
                    "/vertical_alignment": "center",
                }
            )
        if node.node_type == "paragraph":
            values["/layout/spacing/before_mpt"] = 2000
            values["/layout/spacing/after_mpt"] = 3000
        predictions.extend(
            TypedPrediction(
                node.local_node_id,
                node.node_type,
                pointer,
                "test",
                value,
            )
            for pointer, value in values.items()
        )
    raw = replace(
        raw,
        typed_predictions=tuple(predictions),
        region_predictions=(
            sealed.SealedRegionPrediction("n1", "paragraph", "abstain_not_visible", (), 980_000),
        ),
    )
    original_ir = seal_diagnostic_prediction(raw)
    runtime = _fake_runtime(vocabulary=VOCABULARY, tokenizer=TOKENIZER)
    prepared = SimpleNamespace(visual_features=torch.zeros(1))
    if multi:
        result = ds.build_diagnostic_docx_multi(
            [ds.PageAttempt(0, 0.1, raw, prepared), ds.PageAttempt(1, 0.1, raw, prepared)],
            runtime=runtime,
            job_id="synthetic",
            work_dir=tmp_path,
            max_docx_bytes=1_000_000,
        )
    else:
        result = ds.build_diagnostic_docx(
            raw,
            prepared,
            b"not-used-for-text",
            runtime=runtime,
            job_id="synthetic",
            work_dir=tmp_path,
            max_docx_bytes=1_000_000,
            expected_prediction_ir_hash=ds.sealed.recompute_prediction_ir_hash(
                ds.seal_page(
                    document_bundle_id="synthetic",
                    hypothesis=raw,
                    eval_cache_identity=runtime.eval_cache_identity,
                    visual_features=prepared.visual_features,
                    component_lock_hash=runtime.component_lock_hash,
                    selection_policy_hash=runtime.selection_policy_hash,
                )
            ),
        )
    assert result.produced, result.docx_chain
    with ZipFile(BytesIO(result.docx_bytes)) as package:
        xml = etree.fromstring(package.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    copies = 2 if multi else 1
    paragraphs = xml.xpath("w:body/w:p", namespaces=ns)
    assert len(paragraphs) == 2 * copies
    assert xml.xpath(".//w:t/text()", namespaces=ns) == ["Preserve this text."] * copies
    for paragraph in paragraphs[::2]:
        assert paragraph.xpath("w:pPr/w:spacing/@w:lineRule", namespaces=ns) == ["exact"]
        assert paragraph.xpath("w:pPr/w:spacing/@w:line", namespaces=ns) == ["1"]
        assert paragraph.xpath("w:pPr/w:spacing/@w:before", namespaces=ns) == ["40"]
        assert paragraph.xpath("w:pPr/w:spacing/@w:after", namespaces=ns) == ["60"]
    assert set(xml.xpath(".//w:sectPr/w:vAlign/@w:val", namespaces=ns)) == {"center"}
    assert result.docx_chain["reconstruction"] == {
        "policy_id": "invisible-empty-paragraph-line.v0",
        "counts": {"invisible_empty_paragraph_line_collapsed": copies},
    }
    assert seal_diagnostic_prediction(raw) == original_ir

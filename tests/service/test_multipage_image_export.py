"""Regression coverage for multi-page image binding and Word export."""
from __future__ import annotations

import io
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from unflatten.eval.builder_chain import gold, run
from unflatten.eval.builder_chain import sealed_adapter as sa
from unflatten.eval.s2_gate import sealed


@pytest.mark.parametrize(("fixture", "story_type"), [
    ("valid/inline_image_with_resource.json", None),
    ("visual_v1/valid/floating_image_anchored.json", None),
    ("visual_v1/valid/header_with_image_and_table.json", "header"),
    ("visual_v1/valid/header_with_image_and_table.json", "footer"),
])
@pytest.mark.parametrize("failed_middle_page", [False, True])
@pytest.mark.parametrize("missing_last_raster", [False, True])
@pytest.mark.parametrize("second_color", ["blue", "red"])
def test_multipage_images_bind_to_their_own_page_and_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fixture: str, failed_middle_page: bool,
    missing_last_raster: bool, story_type: str | None, second_color: str,
) -> None:
    """Repeated local image ids must use the correct raster, including stories
    and floating images, without shifting ownership past a failed page."""
    canonical = gold.load(fixture)
    if story_type == "footer":
        for section in canonical["sections"]:
            section["footers"] = section.pop("headers", [])
            for footer in section["footers"]:
                footer["type"] = "footer"

    def parse(page, **kw):
        return gold.gold_page(canonical, ordinal=kw["page_input"].page_ordinal)

    def images(nodes):
        for node in nodes:
            if node.node_type in ("inline_image", "floating_image"):
                yield node.local_id, node.node_type
            yield from images(node.children)

    image_nodes = list(images(gold.gold_page(canonical).sections))
    monkeypatch.setattr(sa, "page_hypothesis", parse)
    monkeypatch.setattr(sa, "parser_local_ids_by_model_id", lambda *_a, **_kw: {
        f"model_image_{i}": parser_id for i, (parser_id, _nt) in enumerate(image_nodes)
    })
    page = sealed.page_from_dict({
        "document_bundle_id": "bundle#0", "outcome": "terminated", "accepted_tokens": [],
        "node_states": [{"local_node_id": f"model_image_{i}", "slot_kind": "opened_node",
                         "node_type": node_type, "parent_local_node_id": None,
                         "open_event_index": i + 1}
                        for i, (_pid, node_type) in enumerate(image_nodes)],
        "region_predictions": [{
            "local_node_id": f"model_image_{i}", "node_type": node_type,
            "alignment_status": "region", "confidence_ppm": 900_000,
            "regions": [{"page_ordinal": 0, "bbox_ppm": [100_000, 100_000, 900_000, 900_000],
                         "continuation": "complete", "confidence_ppm": 900_000}],
        } for i, (_pid, node_type) in enumerate(image_nodes)],
    })
    if story_type is not None:
        page = replace(page, node_states=(*page.node_states,
            sealed.NodeStateProjection("story", "opened_node", story_type, None, 100)))
    last = 2 if failed_middle_page else 1
    pages = [page, replace(page, document_bundle_id=f"bundle#{last}")]
    if failed_middle_page:
        pages.insert(1, sealed.PageDecodeResult("bundle#1", "max_steps_exceeded"))
    rasters = {}
    for index, color in ((0, "red"), (last, second_color)):
        output = io.BytesIO()
        Image.new("RGB", (100, 100), color).save(output, format="PNG", dpi=(300, 300))
        rasters[index] = output.getvalue()
    if missing_last_raster:
        del rasters[last]

    row = run.process_model_pages(
        pages, vocabulary=None, tokenizer=None, document_id="images", render_dir=tmp_path,
        page_rasters=rasters, image_region_snap=True, recovered_image_materialization=True,
    )

    if missing_last_raster:
        assert row.envelope.axes.ast_status == "failed"
        assert row.envelope.ast.failure_report["primary_code"] == "RESOURCE_SLOT_UNRESOLVED"
        assert not (tmp_path / "images.docx").exists()
        return
    assert row.envelope.axes.ast_status == "strict_valid", row.envelope.ast.failure_report
    assert row.envelope.ast.builder_report["unresolved"] == []
    colors = set()
    for data in row.envelope.ast.assets.values():
        with Image.open(io.BytesIO(data)) as image:
            colors.add(image.convert("RGB").getpixel((image.width // 2, image.height // 2)))
    expected_colors = {(255, 0, 0), (0, 0, 255)} if second_color == "blue" else {(255, 0, 0)}
    assert colors == expected_colors
    def built_images(value):
        if isinstance(value, dict):
            if value.get("type") in ("inline_image", "floating_image"):
                yield value
            for child in value.values():
                yield from built_images(child)
        elif isinstance(value, list):
            for child in value:
                yield from built_images(child)

    for image in built_images(row.envelope.ast.canonical_ast):
        assert image["size"] == {"width_mpt": 24000, "height_mpt": 24000}
        assert image["crop"] == dict.fromkeys(("left_ppm", "top_ppm", "right_ppm", "bottom_ppm"), 0)
    with zipfile.ZipFile(tmp_path / "images.docx") as docx:
        media = [n for n in docx.namelist() if n.startswith("word/media/")]
        assert len(media) == len(expected_colors)
    assert row.recovered_image_counts["recovered_image:size_set"] == 2 * len(image_nodes)
    assert row.image_region_snap_counts is not None

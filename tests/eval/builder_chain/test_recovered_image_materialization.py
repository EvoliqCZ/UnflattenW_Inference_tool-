"""`builder_chain.recovered_image_materialization` -- the opt-in chain policy
materializes a `render_recovered` image's transform from the resource's own
provenance (crop 0, rotation 0, size = pixel dims at the verified raster DPI),
touches nothing else, refuses without a verified DPI, and counts every step."""

from __future__ import annotations

import hashlib
import io
from collections import Counter

import pytest
from PIL import Image

from unflatten.assembly.assemble import AssembledDocumentHypothesis, AssembledNode, PageStory
from unflatten.assembly.hypothesis import PageInput
from unflatten.ast.canonical_json import to_canonical_json
from unflatten.eval.builder_chain import recovered_image_materialization as rim
from unflatten.model import resource_binder as rb
from unflatten.model.resource_binder import RecoveredResource

# Pinned SHA-256 of to_canonical_json(rim.policy_record()) with POLICY_STATUS
# == "RATIFIED" (RFC004-ERRATA-004, owner 2026-09-26). Computed once; never
# derive it in the test. Ratification flips the status and re-pins it.
_POLICY_RECORD_SHA256 = "a6f9ac193e15d6dc80a47c9c1dabb7b9b84c2d40e2b7673e5b1d4018f29419d7"

# Worksheet image n2 (handoff 2026-09-26 §6, design M2): recovered 1645 x 1209
# px at 300 DPI displays at 394.8 x 290.16 pt; the model predicted a second
# crop and a 243.386 x 153.158 pt rectangle.
PIXEL_DIMS = (1645, 1209)
PREDICTED_CROP = {
    "/crop/left_ppm": 79_507,
    "/crop/top_ppm": 711_170,
    "/crop/right_ppm": 267_380,
    "/crop/bottom_ppm": 229_404,
}
ZERO_CROP = dict.fromkeys(PREDICTED_CROP, 0)
PREDICTED_SIZE = {"/size/width_mpt": 243_386, "/size/height_mpt": 153_158}
FAITHFUL_SIZE = {"/size/width_mpt": 394_800, "/size/height_mpt": 290_160}
ANCHOR = {
    "/anchor/horizontal/relative_from": "page",
    "/anchor/horizontal/mode": "offset",
    "/anchor/horizontal/offset_mpt": 12_000,
    "/anchor/vertical/relative_from": "paragraph",
    "/anchor/vertical/mode": "offset",
    "/anchor/vertical/offset_mpt": -3_000,
    "/anchor/wrap/mode": "square",
    "/anchor/behind_text": False,
}


def _image(node_id: str, node_type: str = "inline_image", **typed) -> AssembledNode:
    return AssembledNode(node_id, node_type, (), None, dict(typed), ((0, "p_" + node_id),))


def _predicted_image(node_id: str, node_type: str = "inline_image", **extra) -> AssembledNode:
    return _image(
        node_id, node_type, **PREDICTED_CROP, **PREDICTED_SIZE,
        **{"/rotation_mdeg": 4_500, "/alt_text": "kept"}, **extra,
    )


def _section(*children: AssembledNode, node_id: str = "a0") -> AssembledNode:
    return AssembledNode(
        node_id, "section", tuple(children), None, {"/page/width_mpt": 595_276}, ((0, "p_s"),)
    )


def _document(*sections: AssembledNode, notes=(), stories=()) -> AssembledDocumentHypothesis:
    return AssembledDocumentHypothesis(
        pages=(PageInput(0, 2481, 3508),),
        sections=tuple(sections),
        notes=tuple(notes),
        stories=tuple(stories),
        story_sets=(),
        seam_decisions=(),
        conflicts=(),
    )


def _recovered(slot: str, width: int = PIXEL_DIMS[0], height: int = PIXEL_DIMS[1]):
    return RecoveredResource(
        slot, "resolved", "image/png", b"\x89PNG", width, height,
        fidelity="render_recovered", coordinate_transform_hash="ct", encoding_policy_hash="ep",
    )


def _find(document: AssembledDocumentHypothesis, node_id: str) -> AssembledNode:
    def walk(nodes):
        for node in nodes:
            if node.assembled_id == node_id:
                return node
            found = walk(node.children)
            if found is not None:
                return found
        return None

    roots = (*document.sections, *document.notes, *(s.node for s in document.stories))
    found = walk(roots)
    if found is None:
        raise KeyError(node_id)
    return found


def _png(dpi: tuple[float, float] | None = None) -> bytes:
    image = Image.new("RGB", (8, 8), "white")
    output = io.BytesIO()
    if dpi is None:
        image.save(output, format="PNG")
    else:
        image.save(output, format="PNG", dpi=dpi)
    return output.getvalue()


# --- policy record -----------------------------------------------------------


def test_policy_status_is_ratified():
    assert rim.POLICY_STATUS == "RATIFIED"
    assert rim.policy_record()["status"] == "RATIFIED"


def test_policy_record_canonical_json_sha256_is_pinned():
    payload = to_canonical_json(rim.policy_record())
    assert hashlib.sha256(payload).hexdigest() == _POLICY_RECORD_SHA256


def test_policy_record_names_the_pinned_dpi_scope_and_values():
    record = rim.policy_record()
    assert record["policy_id"] == "recovered-image-materialization.v0"
    assert record["erratum"] == "docs/rfc/RFC004-ERRATA-004.md"
    assert record["raster_dpi"] == 300
    assert record["scope"] == {"fidelity": "render_recovered"}
    assert record["values"] == {
        "crop": 0, "rotation_mdeg": 0, "size": "pixel_dims*72000/raster_dpi",
    }


# --- apply: the edit ---------------------------------------------------------


def test_apply_render_recovered_image_sets_zero_crop_zero_rotation_and_raster_dpi_size():
    document = _document(_section(_predicted_image("a1")))

    edited, counts = rim.apply(document, {"a1": _recovered("a1")}, raster_dpi=300.0)

    node = _find(edited, "a1")
    assert node.typed == {
        **ZERO_CROP, **FAITHFUL_SIZE, "/rotation_mdeg": 0, "/alt_text": "kept",
    }
    assert (node.assembled_id, node.node_type, node.children, node.text, node.sources) == (
        "a1", "inline_image", (), None, ((0, "p_a1"),)
    )
    assert counts == Counter({
        "recovered_image:crop_zeroed": 1,
        "recovered_image:rotation_zeroed": 1,
        "recovered_image:size_set": 1,
    })
    # Everything around the image is carried over unchanged.
    assert (edited.pages, edited.story_sets, edited.seam_decisions, edited.conflicts) == (
        document.pages, document.story_sets, document.seam_decisions, document.conflicts
    )
    assert edited.sections[0].typed == document.sections[0].typed
    assert edited.sections[0].sources == document.sections[0].sources


def test_apply_size_is_pixel_dims_times_240_at_the_pinned_dpi_not_the_measured_float():
    document = _document(_section(_predicted_image("a1")))

    edited, _ = rim.apply(
        document, {"a1": _recovered("a1", 1645, 1209)}, raster_dpi=299.9994
    )

    node = _find(edited, "a1")
    assert node.typed["/size/width_mpt"] == 1645 * 240 == 394_800
    assert node.typed["/size/height_mpt"] == 1209 * 240 == 290_160


def test_apply_counts_each_kind_only_when_that_entry_changed():
    # crop already zero, rotation already zero, only the size is wrong
    document = _document(_section(_image(
        "a1", **ZERO_CROP, **PREDICTED_SIZE, **{"/rotation_mdeg": 0, "/alt_text": None}
    )))

    edited, counts = rim.apply(document, {"a1": _recovered("a1")}, raster_dpi=300.0)

    assert _find(edited, "a1").typed["/size/width_mpt"] == 394_800
    assert counts == Counter({"recovered_image:size_set": 1})


# --- apply: scope ------------------------------------------------------------


def test_apply_resolved_resource_without_render_recovered_fidelity_leaves_crop_and_size_untouched():
    document = _document(_section(_predicted_image("a1")))
    foreign = RecoveredResource.resolved(
        "a1", mime_type="image/png", data=b"\x89PNG", pixel_width=1645, pixel_height=1209
    )
    assert foreign.fidelity is None

    edited, counts = rim.apply(document, {"a1": foreign}, raster_dpi=300.0)

    assert _find(edited, "a1") is _find(document, "a1")
    assert counts == Counter({"recovered_image:skipped_foreign_fidelity": 1})


def test_apply_unresolved_or_abstained_slot_is_skipped_and_counted():
    document = _document(_section(
        _predicted_image("a1"), _predicted_image("a2"), _predicted_image("a3")
    ))
    recovered = {
        "a1": RecoveredResource.abstained("a1"),
        "a2": RecoveredResource.unresolved("a2", rb.RESOURCE_CROP_BLANK),
        # a3 has no recovery record at all
    }

    edited, counts = rim.apply(document, recovered, raster_dpi=300.0)

    for node_id in ("a1", "a2", "a3"):
        assert _find(edited, node_id) is _find(document, node_id)
    assert counts == Counter({"recovered_image:skipped_unresolved": 3})


def test_apply_floating_image_anchor_is_untouched_and_counted():
    document = _document(_section(_predicted_image("a1", "floating_image", **ANCHOR)))

    edited, counts = rim.apply(document, {"a1": _recovered("a1")}, raster_dpi=300.0)

    node = _find(edited, "a1")
    assert node.node_type == "floating_image"
    assert {k: v for k, v in node.typed.items() if k.startswith("/anchor/")} == ANCHOR
    assert {k: v for k, v in node.typed.items() if k.startswith("/crop/")} == ZERO_CROP
    assert node.typed["/rotation_mdeg"] == 0
    assert {k: v for k, v in node.typed.items() if k.startswith("/size/")} == FAITHFUL_SIZE
    assert counts == Counter({
        "recovered_image:crop_zeroed": 1,
        "recovered_image:rotation_zeroed": 1,
        "recovered_image:size_set": 1,
        "recovered_image:floating_anchor_untouched": 1,
    })


def test_apply_leaves_non_image_nodes_untouched_even_when_a_recovery_names_them():
    run = AssembledNode("a2", "run", (), "text", {"/style/bold": True}, ((0, "p_r"),))
    paragraph = AssembledNode(
        "a1", "paragraph", (run,), None, {**PREDICTED_CROP, **PREDICTED_SIZE}, ((0, "p_p"),)
    )
    document = _document(_section(paragraph))

    edited, counts = rim.apply(document, {"a1": _recovered("a1")}, raster_dpi=300.0)

    assert edited == document
    assert counts == Counter()


def test_apply_edits_images_inside_stories_and_notes():
    body = _section(_predicted_image("a1"))
    footer = AssembledNode("a5", "footer", (_predicted_image("a6"),), None, {}, ((0, "p_f"),))
    note = AssembledNode("a8", "footnote", (_predicted_image("a9"),), None, {}, ((0, "p_n"),))
    document = _document(body, notes=(note,), stories=(PageStory(0, "a0", footer),))
    recovered = {"a1": _recovered("a1"), "a6": _recovered("a6"), "a9": _recovered("a9")}

    edited, counts = rim.apply(document, recovered, raster_dpi=300.0)

    for node_id in ("a1", "a6", "a9"):
        assert _find(edited, node_id).typed == {
            **ZERO_CROP, **FAITHFUL_SIZE, "/rotation_mdeg": 0, "/alt_text": "kept",
        }
    assert edited.stories[0].page_ordinal == 0 and edited.stories[0].section_id == "a0"
    assert edited.stories[0].node.assembled_id == "a5"
    assert edited.notes[0].assembled_id == "a8"
    assert counts["recovered_image:crop_zeroed"] == 3
    assert counts["recovered_image:size_set"] == 3


# --- apply: DPI refusal ------------------------------------------------------


@pytest.mark.parametrize("raster_dpi", [None, 96.0, 300.02, 72.0])
def test_apply_without_verified_raster_dpi_edits_nothing_and_counts_the_refusal(raster_dpi):
    document = _document(_section(_predicted_image("a1")))

    edited, counts = rim.apply(document, {"a1": _recovered("a1")}, raster_dpi=raster_dpi)

    assert edited is document
    assert counts == Counter({"recovered_image:raster_dpi_unverified": 1})


def test_apply_accepts_the_pdftoppm_phys_dpi_within_tolerance():
    document = _document(_section(_predicted_image("a1")))

    edited, counts = rim.apply(document, {"a1": _recovered("a1")}, raster_dpi=299.9994)

    assert _find(edited, "a1").typed["/crop/left_ppm"] == 0
    assert "recovered_image:raster_dpi_unverified" not in counts


def test_multipage_materialization_checks_each_images_own_raster_dpi():
    document = _document(_section(_predicted_image("a1"), _predicted_image("a2")))
    recovered = {"a1": _recovered("a1"), "a2": _recovered("a2")}

    edited, counts = rim.apply(
        document, recovered, raster_dpi=None,
        raster_dpi_by_slot={"a1": 299.9994, "a2": 96.0},
    )

    assert _find(edited, "a1").typed["/crop/left_ppm"] == 0
    assert _find(edited, "a2").typed == _find(document, "a2").typed
    assert counts["recovered_image:raster_dpi_unverified"] == 1
    assert counts["recovered_image:size_set"] == 1


# --- apply: idempotence ------------------------------------------------------


def test_apply_is_idempotent():
    document = _document(_section(
        _predicted_image("a1"), _predicted_image("a2", "floating_image", **ANCHOR)
    ))
    recovered = {"a1": _recovered("a1"), "a2": _recovered("a2")}

    once, first = rim.apply(document, recovered, raster_dpi=300.0)
    twice, second = rim.apply(once, recovered, raster_dpi=300.0)

    assert twice == once
    assert first["recovered_image:crop_zeroed"] == 2
    assert second == Counter({"recovered_image:already_faithful": 2})


def test_apply_counts_already_faithful_without_editing():
    faithful = _image(
        "a1", **ZERO_CROP, **FAITHFUL_SIZE, **{"/rotation_mdeg": 0, "/alt_text": None}
    )
    document = _document(_section(faithful))

    edited, counts = rim.apply(document, {"a1": _recovered("a1")}, raster_dpi=300.0)

    assert _find(edited, "a1") is faithful
    assert counts == Counter({"recovered_image:already_faithful": 1})


# --- raster DPI --------------------------------------------------------------


def test_read_raster_dpi_returns_the_phys_dpi_when_present_and_none_without():
    assert rim.read_raster_dpi(_png(dpi=(300, 300))) == pytest.approx(300, abs=0.01)
    assert rim.read_raster_dpi(_png()) is None


def test_read_raster_dpi_reads_the_header_without_decoding_pixels():
    raster = _png(dpi=(300, 300))
    truncated = raster[: raster.index(b"IDAT") + 4]  # chunk header only, no pixel data
    with pytest.raises(OSError):
        Image.open(io.BytesIO(truncated)).load()

    assert rim.read_raster_dpi(truncated) == pytest.approx(300, abs=0.01)


def test_read_raster_dpi_invalid_bytes_returns_none():
    assert rim.read_raster_dpi(b"not a raster") is None


@pytest.mark.parametrize(
    ("dpi", "expected"),
    [(None, False), (300.0, True), (299.9994, True), (299.995, True),
     (300.02, False), (96.0, False), (72.0, False)],
)
def test_verify_raster_dpi_accepts_only_the_pinned_dpi_within_tolerance(dpi, expected):
    assert rim.verify_raster_dpi(dpi) is expected


# --- apply: DPI refusal only where there is something to materialize ----------


@pytest.mark.parametrize(
    "recovered",
    [{}, {"a1": RecoveredResource.abstained("a1")}],
    ids=["no_recovery_record", "abstained_slot"],
)
def test_apply_unverified_dpi_without_a_render_recovered_resource_counts_nothing(recovered):
    """A page whose recovery resolved no render-recovered image has nothing to
    refuse: no `raster_dpi_unverified`, the document returned as is. The same
    page with one render-recovered resource counts the refusal once."""
    document = _document(_section(_predicted_image("a1")))

    edited, counts = rim.apply(document, recovered, raster_dpi=None)

    assert edited is document
    assert counts == Counter()
    _, refused = rim.apply(document, {"a1": _recovered("a1")}, raster_dpi=None)
    assert refused == Counter({"recovered_image:raster_dpi_unverified": 1})

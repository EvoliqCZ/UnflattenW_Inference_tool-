from __future__ import annotations

import base64
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCANNER_PATH = Path(__file__).parents[2] / "deploy" / "pdf-sandbox" / "scanner.py"
LIBREOFFICE_GRAPH = Path(__file__).parent / "data" / "known-nitra-libreoffice.qpdf.json"
KNOWN_NITRA_ACCEPTANCE_GRAPH = (
    Path(__file__).parent / "data" / "known-nitra-acceptance-2026-09-26.qpdf.json"
)
LETTER = [0, 0, 612, 792]


def _load_scanner():
    sys.path.insert(0, str(SCANNER_PATH.parent))
    spec = importlib.util.spec_from_file_location("pdf_admission_scanner", SCANNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _stream(data: bytes, **dictionary: object) -> dict[str, object]:
    return {
        "stream": {
            "dict": dictionary,
            "data": base64.b64encode(data).decode("ascii"),
        }
    }


def _form_graph(
    invocation: bytes,
    form_data: bytes,
    *,
    form_dictionary: dict[str, object] | None = None,
    page_xobjects: dict[str, object] | None = None,
    extra_objects: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    xobjects = page_xobjects or {"/Fm": "5 0 R"}
    form_dict = {"/Type": "/XObject", "/Subtype": "/Form"}
    form_dict.update(form_dictionary or {})
    return {
        "qpdf": [
            {"jsonversion": 2},
            {
                "obj:3 0 R": {
                    "value": {
                        "/Type": "/Page",
                        "/MediaBox": LETTER,
                        "/Contents": "4 0 R",
                        "/Resources": {"/XObject": xobjects},
                    }
                }
            },
            {"obj:4 0 R": _stream(invocation)},
            {"obj:5 0 R": _stream(form_data, **form_dict)},
            *(extra_objects or []),
        ]
    }


def _page_graph(
    content: bytes,
    *,
    page: dict[str, object] | None = None,
    extra_objects: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    page_dict: dict[str, object] = {"/Type": "/Page", "/MediaBox": LETTER, "/Contents": "4 0 R"}
    page_dict.update(page or {})
    return {
        "qpdf": [
            {"jsonversion": 2},
            {"obj:3 0 R": {"value": page_dict}},
            {"obj:4 0 R": _stream(content)},
            *(extra_objects or []),
        ]
    }


def test_scanner_classifies_governed_object_graph_features_without_content() -> None:
    scanner = _load_scanner()
    graph = {
        "qpdf": [
            {"jsonversion": 2},
            {"obj:1 0 R": {"value": {"/Type": "/Catalog", "/OpenAction": "2 0 R"}}},
            {"obj:2 0 R": {"value": {"/S": "/JavaScript", "/JS": "secret()"}}},
            {"obj:3 0 R": {"value": {"/Type": "/Filespec", "/F": "private.txt"}}},
            {"obj:4 0 R": {"value": {"/Type": "/Annot", "/Contents": "private note"}}},
        ]
    }

    counts = scanner.classify_object_graph(graph, max_image_width=6000, max_image_height=6000)

    assert counts == {
        "object_count": 4,
        "javascript_action_objects": 2,
        "attachment_objects": 1,
        "c2pa_metadata_objects": 0,
        "annotation_objects": 1,
        "extreme_image_objects": 0,
    }
    assert "secret" not in repr(counts)
    assert "private" not in repr(counts)


def test_scanner_distinguishes_visible_hidden_and_ambiguous_text_operators() -> None:
    scanner = _load_scanner()

    assert scanner.classify_text_visibility(b"BT /F1 12 Tf 0 Tr (visible words) Tj ET") == (
        "visible_only",
        0,
        0,
    )
    assert scanner.classify_text_visibility(b"BT 3 Tr (hidden words) Tj ET") == (
        "hidden_proven",
        1,
        0,
    )
    assert scanner.classify_text_visibility(b"BT 4 Tr (clipping words) Tj ET") == (
        "ambiguous",
        0,
        1,
    )
    assert scanner.classify_text_visibility(b"/OC /Layer BDC BT (maybe hidden) Tj ET EMC") == (
        "ambiguous",
        0,
        1,
    )


def test_scanner_text_rendering_mode_persists_across_text_objects() -> None:
    scanner = _load_scanner()

    assert scanner.classify_text_visibility(
        b"BT 3 Tr ET BT (still hidden) Tj ET"
    ) == ("hidden_proven", 1, 0)


def test_scanner_rejects_invalid_or_unresolved_text_font_sizes() -> None:
    scanner = _load_scanner()

    mutations = (
        (b"BT /F1 0 Tf (zero size) Tj ET", ("hidden_proven", 1, 0)),
        (b"BT /F1 -1 Tf (negative size) Tj ET", ("ambiguous", 0, 1)),
        (b"BT /F1 NaN Tf (non-finite size) Tj ET", ("ambiguous", 0, 1)),
        (b"BT /F1 Tf (malformed size) Tj ET", ("ambiguous", 0, 1)),
        (b"BT (unresolved size) Tj ET", ("ambiguous", 0, 1)),
        (
            b"BT /F1 12 Tf 0 0 0 0 0 0 Tm (degenerate size) Tj ET",
            ("ambiguous", 0, 1),
        ),
    )
    for stream, expected in mutations:
        assert scanner.classify_text_visibility(stream) == expected


def test_scanner_tracks_complete_text_state_and_rejects_unsafe_values() -> None:
    scanner = _load_scanner()

    safe = b"BT /F1 12 Tf 0 Tc 0 Tw 100 Tz 0 TL 0 Tr 0 Ts (safe) Tj ET"
    assert scanner.classify_text_visibility(safe) == ("visible_only", 0, 0)

    mutations = (
        (b"BT /F1 12 Tf 0 Tz (zero scale) Tj ET", ("hidden_proven", 1, 0)),
        (b"BT /F1 12 Tf -100 Tz (mirrored) Tj ET", ("ambiguous", 0, 1)),
        (b"BT /F1 12 Tf 50 Tz (scaled) Tj ET", ("ambiguous", 0, 1)),
        (b"BT /F1 12 Tf NaN Tz (nonfinite) Tj ET", ("ambiguous", 0, 1)),
        (b"BT /F1 12 Tf 1e100 Tz (extreme) Tj ET", ("ambiguous", 0, 1)),
        (b"BT /F1 12 Tf 1e-300 Tz (degenerate) Tj ET", ("hidden_proven", 1, 0)),
        (b"BT /F1 12 Tf Tz (malformed) Tj ET", ("ambiguous", 0, 1)),
        (b"BT /F1 1e100 Tf (extreme font) Tj ET", ("ambiguous", 0, 1)),
        # 27.9.: Tc is now folded into the real advance (PDF spec 9.2.4) instead of
        # always failing closed -- a negative Tc only tightens inter-glyph spacing,
        # it cannot push ink past the naive glyph box, so this is genuinely visible.
        (b"BT /F1 12 Tf -1 Tc (overlap) Tj ET", ("visible_only", 0, 0)),
        (b"BT /F1 12 Tf 1e100 Tw (extreme word space) Tj ET", ("ambiguous", 0, 1)),
        (b"BT /F1 12 Tf -1 TL T* (reverse leading) Tj ET", ("visible_only", 0, 0)),
        (b"BT /F1 12 Tf 1 Ts (raised) Tj ET", ("ambiguous", 0, 1)),
        (b"BT /F1 12 Tf 1e100 Ts (extreme rise) Tj ET", ("ambiguous", 0, 1)),
    )
    for stream, expected in mutations:
        assert scanner.classify_text_visibility(stream) == expected


def test_text_state_persists_across_bt_et_and_is_restored_by_q_Q() -> None:
    scanner = _load_scanner()

    assert scanner.classify_text_visibility(
        b"BT /F1 12 Tf 0 Tz ET BT (still hidden) Tj ET"
    ) == ("hidden_proven", 1, 0)
    assert scanner.classify_text_visibility(
        b"BT /F1 12 Tf ET q BT 0 Tz ET Q BT (restored) Tj ET"
    ) == ("visible_only", 0, 0)
    # Ts (rise), unlike Tc since 27.9., still always fails closed when non-zero,
    # so it stays a valid probe for "does graphics state survive BT/ET".
    assert scanner.classify_text_visibility(
        b"BT /F1 12 Tf 1 Ts ET 0 g BT (state retained) Tj ET"
    ) == ("ambiguous", 0, 1)


def test_forms_inherit_nested_text_state_without_leaking_mutations() -> None:
    scanner = _load_scanner()

    inherited = _form_graph(
        b"BT /F1 12 Tf 0 Tz ET /Fm Do",
        b"BT (hidden inherited scale) Tj ET",
    )
    restored = _form_graph(
        b"BT /F1 12 Tf ET /Fm Do BT (page remains visible) Tj ET",
        b"BT 0 Tz ET",
    )
    nested = _form_graph(
        b"BT /F1 12 Tf 0 Tz ET /Outer Do",
        b"/Inner Do",
        page_xobjects={"/Outer": "5 0 R", "/Inner": "6 0 R"},
        extra_objects=[
            {
                "obj:6 0 R": _stream(
                    b"BT (nested hidden scale) Tj ET",
                    **{"/Type": "/XObject", "/Subtype": "/Form"},
                )
            }
        ],
    )

    assert scanner._classify_graph_text(inherited) == ("hidden_proven", 1, 0)
    assert scanner._classify_graph_text(restored) == ("visible_only", 0, 0)
    assert scanner._classify_graph_text(nested) == ("hidden_proven", 1, 0)


def test_all_scanner_production_modules_remain_below_line_limit() -> None:
    scanner_files = sorted(SCANNER_PATH.parent.glob("scanner*.py"))

    assert scanner_files
    assert {path.name: len(path.read_text().splitlines()) for path in scanner_files} == {
        path.name: len(path.read_text().splitlines()) for path in scanner_files
        if len(path.read_text().splitlines()) < 500
    }


def test_text_showing_arrays_quotes_and_positioning_are_validated() -> None:
    scanner = _load_scanner()

    visible = (
        b"BT /F1 12 Tf [(array) 0 (text)] TJ ET",
        b"BT /F1 12 Tf 0 TL (next line) ' ET",
        b'BT /F1 12 Tf 0 TL 0 0 (spaced) " ET',
        b"BT /F1 12 Tf 0 0 Td 0 0 TD T* (same origin) Tj ET",
        b"BT /F1 12 Tf [(text) -120 (kerned)] TJ ET",
        b"BT /F1 12 Tf 10 TL (moves) ' ET",
        b"BT /F1 12 Tf 1 0 Td (moves) Tj ET",
    )
    for stream in visible:
        assert scanner.classify_text_visibility(stream) == ("visible_only", 0, 0)

    ambiguous = (
        b"BT /F1 12 Tf [(text) /Bad] TJ ET",
        b"BT /F1 12 Tf [1e100 (text)] TJ ET",
        b"BT /F1 12 Tf (missing quote operands) \" ET",
        b"BT /F1 12 Tf 1 2 3 Tm (malformed matrix) Tj ET",
        b"BT /F1 12 Tf 1e100 0 Td (unbounded offset) Tj ET",
    )
    for stream in ambiguous:
        assert scanner.classify_text_visibility(stream) == ("ambiguous", 0, 1)


def test_unknown_or_malformed_text_affecting_operations_fail_closed() -> None:
    scanner = _load_scanner()

    mutations = (
        b"BT /F1 12 Tf 1 FutureTextState (text) Tj ET",
        b"BT /F1 12 Tf 1 2 Tc (text) Tj ET",
        b"BT /F1 12 Tf 8 Tr (text) Tj ET",
        b"BT /F1 12 Tf /Wrong Ts (text) Tj ET",
    )
    for stream in mutations:
        assert scanner.classify_text_visibility(stream) == ("ambiguous", 0, 1)


def test_scanner_font_size_persists_and_transitions_across_text_objects() -> None:
    scanner = _load_scanner()

    assert scanner.classify_text_visibility(
        b"BT /F1 0 Tf ET BT (still zero size) Tj ET"
    ) == ("hidden_proven", 1, 0)
    assert scanner.classify_text_visibility(
        b"BT /F1 0 Tf ET BT /F1 12 Tf (visible after transition) Tj ET"
    ) == ("visible_only", 0, 0)


def test_scanner_q_and_Q_restore_text_rendering_state() -> None:
    scanner = _load_scanner()

    assert scanner.classify_text_visibility(
        b"q BT 3 Tr (hidden) Tj ET Q BT /F1 12 Tf (visible) Tj ET"
    ) == ("hidden_proven", 1, 0)
    assert scanner.classify_text_visibility(
        b"BT 3 Tr ET q BT 0 Tr ET Q BT (hidden again) Tj ET"
    ) == ("hidden_proven", 1, 0)
    assert scanner.classify_text_visibility(
        b"BT /F1 12 Tf ET q BT 3 Tr ET Q BT (restored visible mode) Tj ET"
    ) == ("visible_only", 0, 0)


def test_scanner_q_and_Q_restore_text_font_size() -> None:
    scanner = _load_scanner()

    assert scanner.classify_text_visibility(
        b"BT /F1 12 Tf ET q BT /F1 0 Tf ET Q BT (restored visible size) Tj ET"
    ) == ("visible_only", 0, 0)
    assert scanner.classify_text_visibility(
        b"BT /F1 0 Tf ET q BT /F1 12 Tf ET Q BT (restored zero size) Tj ET"
    ) == ("hidden_proven", 1, 0)


def test_positioned_and_transformed_text_inside_the_page_is_visible() -> None:
    """26.9. T11 on prod: every LibreOffice/Word PDF was refused as
    PDF_TEXT_VISIBILITY_AMBIGUOUS because any `Td`, kerning, `cm` or clip was
    treated as unknowable geometry. Position is tracked instead."""
    scanner = _load_scanner()

    visible = (
        b"BT /F1 12 Tf 10 10 Td (text) Tj ET",
        b"BT /F1 12 Tf 71 780 Td (top of a letter page) Tj ET",
        b"BT /F1 12 Tf [(a) -120 (b)] TJ ET",
        b"2 0 0 2 0 0 cm BT /F1 12 Tf 10 10 Td (scaled) Tj ET",
        b"1 0 0 1 50 50 cm BT /F1 12 Tf (translated) Tj ET",
        b"BT /F1 12 Tf 0 1 -1 0 300 300 Tm (rotated) Tj ET",
        b"BT /F1 12 Tf 1 0 0.3 1 100 100 Tm (skewed) Tj ET",
        b"0 0 612 792 re W* n BT /F1 12 Tf 71 700 Td (full page clip) Tj ET",
        b"0 0 612 792 re W n BT /F1 12 Tf 71 700 Td (nonzero winding clip) Tj ET",
        b"q 0 0 10 10 re W n Q BT /F1 12 Tf 100 100 Td (clip restored by Q) Tj ET",
        b"BT /F1 12 Tf 1 0 0 1 10 10 Tm ET BT (matrix reset) Tj ET",
    )
    for stream in visible:
        assert scanner.classify_text_visibility(stream) == ("visible_only", 0, 0), stream
        assert scanner._classify_graph_text(_page_graph(stream)) == ("visible_only", 0, 0), stream


def test_text_whose_origin_cannot_show_is_hidden() -> None:
    scanner = _load_scanner()

    hidden_by_clip = (
        b"0 0 10 10 re W n BT /F1 12 Tf 100 100 Td (outside clip) Tj ET",
        b"q 1 0 0 1 200 200 cm 0 0 50 50 re W n Q 0 0 50 50 re W n "
        b"BT /F1 12 Tf 225 225 Td (clip built under a moved ctm) Tj ET",
        b"1 0 0 1 200 200 cm 0 0 50 50 re W n BT /F1 12 Tf 100 100 Td (before ctm) Tj ET",
        b"0 0 300 300 re W n 200 200 300 300 re W n BT /F1 12 Tf 10 10 Td (intersection) Tj ET",
        b"0 0 10 10 re W n /Fm Do",
    )
    for stream in hidden_by_clip[:-1]:
        assert scanner.classify_text_visibility(stream) == ("hidden_proven", 1, 0), stream
    nested = _form_graph(hidden_by_clip[-1], b"BT /F1 12 Tf 50 50 Td (inherits clip) Tj ET")
    assert scanner._classify_graph_text(nested) == ("hidden_proven", 1, 0)

    off_page = (
        b"BT /F1 12 Tf 1 0 0 1 100000 100000 Tm (far off page) Tj ET",
        b"BT /F1 12 Tf 700 100 Td (right of a letter page) Tj ET",
        b"BT /F1 12 Tf 71 796.889 Td (top of an A4 page, above a letter page) Tj ET",
        b"BT /F1 12 Tf 100 -5 Td (below the page) Tj ET",
        b"1 0 0 1 0 800 cm BT /F1 12 Tf (moved off by ctm) Tj ET",
        b"2 0 0 2 0 0 cm BT /F1 12 Tf 400 100 Td (scaled off page) Tj ET",
    )
    for stream in off_page:
        assert scanner._classify_graph_text(_page_graph(stream)) == ("hidden_proven", 1, 0), stream
    # without a page the stream classifier has no outer bound
    assert scanner.classify_text_visibility(off_page[0]) == ("visible_only", 0, 0)

    assert scanner.classify_text_visibility(b"BT /F1 12 Tf 7 Tr (clip only) Tj ET") == (
        "hidden_proven",
        1,
        0,
    )


def test_unprovable_geometry_stays_ambiguous() -> None:
    scanner = _load_scanner()

    ambiguous = (
        b"0 0 m 100 0 l 50 100 l h W n BT /F1 12 Tf 40 40 Td (non-rect clip) Tj ET",
        b"0 0 100 100 re 200 200 50 50 re W n BT /F1 12 Tf 10 10 Td (two rects) Tj ET",
        b"W n BT /F1 12 Tf 10 10 Td (empty clip path) Tj ET",
        b"0 0 0 0 0 0 cm BT /F1 12 Tf 10 10 Td (degenerate ctm) Tj ET",
        b"1 1 1 1 0 0 cm BT /F1 12 Tf 10 10 Td (singular ctm) Tj ET",
        b"0.7 0.7 -0.7 0.7 0 0 cm 0 0 100 100 re W n BT /F1 12 Tf 10 10 Td (rotated clip) Tj ET",
        b"0 0 100 100 re BT /F1 12 Tf 10 10 Td (path left open) Tj ET",
        b"0 0 100 100 re q Q BT /F1 12 Tf 10 10 Td (state change inside a path) Tj ET",
        b"70 100 5 5 re W n BT /F1 12 Tf 71 101 Td (clip smaller than the glyphs) Tj ET",
        b"0.0001 0 0 0.0001 0 0 cm BT /F1 12 Tf 10 10 Td (sub-point text) Tj ET",
        b"BT /F1 12 Tf 4 Tr (clipping words) Tj ET",
    )
    for stream in ambiguous:
        assert scanner.classify_text_visibility(stream) == ("ambiguous", 0, 1), stream


def test_page_box_comes_from_the_media_box_and_crop_box() -> None:
    scanner = _load_scanner()
    text = b"BT /F1 12 Tf 300 300 Td (middle of the page) Tj ET"

    inherited = _page_graph(
        text,
        page={"/MediaBox": None, "/Parent": "8 0 R"},
        extra_objects=[{"obj:8 0 R": {"value": {"/Type": "/Pages", "/MediaBox": LETTER}}}],
    )
    del inherited["qpdf"][1]["obj:3 0 R"]["value"]["/MediaBox"]
    assert scanner._classify_graph_text(inherited) == ("visible_only", 0, 0)

    indirect = _page_graph(
        text,
        page={"/MediaBox": "9 0 R"},
        extra_objects=[{"obj:9 0 R": {"value": LETTER}}],
    )
    assert scanner._classify_graph_text(indirect) == ("visible_only", 0, 0)

    absent = _page_graph(text, page={"/Parent": "8 0 R"})
    del absent["qpdf"][1]["obj:3 0 R"]["value"]["/MediaBox"]
    assert scanner._classify_graph_text(absent) == ("ambiguous", 0, 1)
    for media_box in ([0, 0, 0, 792], [0, 0, 612], ["0", 0, 612, 792], [0, 0, 10**1000, 792]):
        assert scanner._classify_graph_text(_page_graph(text, page={"/MediaBox": media_box})) == (
            "ambiguous",
            0,
            1,
        ), media_box

    cropped_out = _page_graph(text, page={"/CropBox": [0, 0, 100, 100]})
    assert scanner._classify_graph_text(cropped_out) == ("hidden_proven", 1, 0)
    cropped_in = _page_graph(text, page={"/CropBox": [200, 200, 400, 400]})
    assert scanner._classify_graph_text(cropped_in) == ("visible_only", 0, 0)
    bad_crop = _page_graph(text, page={"/CropBox": [0, 0, 0, 0]})
    assert scanner._classify_graph_text(bad_crop) == ("ambiguous", 0, 1)


def test_paint_away_from_text_leaves_it_visible() -> None:
    """Paint is positional too: only paint that can cover a run, or lie under it,
    is ambiguous. A LibreOffice page draws its table rules after the cell text."""
    scanner = _load_scanner()

    visible = (
        b"BT /F1 12 Tf 100 100 Td (text) Tj ET 0 g 300 300 100 100 re f",
        b"0 g 300 300 100 100 re f BT /F1 12 Tf 100 100 Td (text) Tj ET",
        b"BT /F1 12 Tf 100 100 Td (text) Tj ET 0 G 300 300 100 100 re S",
        b"BT /F1 12 Tf 100 100 Td (cell) Tj ET "
        b"q 0.5 w 70 90 m 530 90 l S Q q 0.5 w 200 60 m 200 400 l S Q",
        b"BT /F1 12 Tf 100 100 Td (underlined) Tj ET 0 g 100 98 60 0.7 re f",
        b"1 g 0 0 612 792 re f 0 g BT /F1 12 Tf 100 100 Td (white fill before text) Tj ET",
        b"BT /F1 12 Tf 100 100 Td (text) Tj ET "
        b"q 100 0 0 100 300 300 cm BI /W 1 /H 1 /BPC 1 /CS /G ID x EI Q",
        b"BT /F1 12 Tf 100 100 Td (a) Tj ET q 0 0 100 100 re W n 0 g 0 0 612 792 re f Q",
    )
    for stream in visible:
        assert scanner.classify_text_visibility(stream) == ("visible_only", 0, 0), stream

    ambiguous = (
        b"BT /F1 12 Tf 100 100 Td (covered) Tj ET 0 g 90 90 100 30 re f",
        b"BT /F1 12 Tf 100 100 Td (white cover after) Tj ET 1 g 90 90 100 30 re f",
        b"BT /F1 12 Tf 100 100 Td (run reaches right) Tj ET 0 g 150 95 20 20 re f",
        b"BT /F1 12 Tf 100 100 Td [(kerned right) -3000 (x)] TJ ET 0 g 175 95 20 20 re f",
        b"0 g 90 90 100 30 re f BT /F1 12 Tf 100 100 Td (on paint) Tj ET",
        b"BT /F1 12 Tf 100 100 Td (thick stroke) Tj ET 0 G 8 w 90 95 m 200 95 l S",
        b"BT /F1 12 Tf 100 100 Td (miter) Tj ET 0 G 2 w 80 80 m 90 130 l 100 80 l S",
        b"BT /F1 12 Tf 100 100 Td (image over) Tj ET "
        b"q 100 0 0 100 50 50 cm BI /W 1 /H 1 /BPC 1 /CS /G ID x EI Q",
        b"BT /F1 12 Tf 0 1 -1 0 100 100 Tm (rotated run, extent unknown) Tj ET "
        b"0 g 300 300 10 10 re f",
        b"BT /F1 12 Tf 100 100 Td (stroke with unknown colour) Tj ET "
        b"/Pattern CS /P1 SCN 90 95 m 200 95 l S",
        b"BT /F1 12 Tf 100 100 Td (a) Tj ET 0 0 612 792 re W n 0 0 100 100 re sh",
    )
    for stream in ambiguous:
        assert scanner.classify_text_visibility(stream) == ("ambiguous", 0, 1), stream

    hairlines = b"".join(b"0 g 95 %d 60 0.5 re f " % (96 + index) for index in range(9))
    assert scanner.classify_text_visibility(
        b"BT /F1 12 Tf 100 100 Td (stacked) Tj ET " + hairlines
    ) == ("ambiguous", 0, 1)
    assert scanner.classify_text_visibility(
        hairlines + b" BT /F1 12 Tf 100 100 Td (on stacked) Tj ET"
    ) == ("ambiguous", 0, 1)


def test_interior_tj_kerning_does_not_inflate_the_run_past_a_bordering_rule() -> None:
    """26.9. T11 acceptance: a LibreOffice table cell packs many single-glyph
    hex strings with small positive (tightening) TJ numbers between them --
    ordinary per-glyph kerning, not a leading offset. Summing every interior
    adjustment into one leftward shift (the old model) pushed the run's box
    into the table's border rule next to it and made the page
    PDF_TEXT_VISIBILITY_AMBIGUOUS, even though no glyph ink reaches the
    border. Interior kerning must not move ink past the naive glyph box;
    only a leading/trailing adjustment that pushes real ink beyond it may."""
    scanner = _load_scanner()

    interior_kerning = (
        b"BT /F1 11.5 Tf 100 100 Td [(a)40(b)40(c)40(d)40(e)] TJ ET "
        b"0 g 98.5 90 3.1 30 re f"
    )
    assert scanner.classify_text_visibility(interior_kerning) == ("visible_only", 0, 0)

    leading_offset = (
        b"BT /F1 12 Tf 100 100 Td [200 (a)] TJ ET 0 g 88 90 15 30 re f"
    )
    assert scanner.classify_text_visibility(leading_offset) == ("ambiguous", 0, 1)


def test_form_paint_is_related_to_page_text_by_position() -> None:
    scanner = _load_scanner()

    away = _form_graph(
        b"BT /F1 12 Tf 100 100 Td (page) Tj ET /Fm Do BT /F1 12 Tf 100 200 Td (after) Tj ET",
        b"0 g 300 300 100 100 re f",
    )
    over = _form_graph(
        b"BT /F1 12 Tf 100 100 Td (page) Tj ET /Fm Do",
        b"1 g 90 90 100 30 re f",
    )
    under = _form_graph(
        b"/Fm Do BT /F1 12 Tf 100 100 Td (page) Tj ET",
        b"0 g 90 90 100 30 re f",
    )
    form_text_on_page_paint = _form_graph(
        b"0 g 90 90 100 30 re f /Fm Do",
        b"BT /F1 12 Tf 100 100 Td (form) Tj ET",
    )
    image = {"obj:6 0 R": _stream(b"x", **{"/Subtype": "/Image", "/Width": 1, "/Height": 1})}
    image_away = _form_graph(
        b"BT /F1 12 Tf 100 100 Td (page) Tj ET q 100 0 0 100 300 300 cm /Im Do Q",
        b"",
        page_xobjects={"/Im": "6 0 R"},
        extra_objects=[image],
    )
    image_over = _form_graph(
        b"BT /F1 12 Tf 100 100 Td (page) Tj ET q 100 0 0 100 50 50 cm /Im Do Q",
        b"",
        page_xobjects={"/Im": "6 0 R"},
        extra_objects=[image],
    )

    assert scanner._classify_graph_text(away) == ("visible_only", 0, 0)
    assert scanner._classify_graph_text(over) == ("ambiguous", 0, 1)
    assert scanner._classify_graph_text(under) == ("ambiguous", 0, 1)
    assert scanner._classify_graph_text(form_text_on_page_paint) == ("ambiguous", 0, 1)
    assert scanner._classify_graph_text(image_away) == ("visible_only", 0, 0)
    assert scanner._classify_graph_text(image_over) == ("ambiguous", 0, 1)


def test_synthetic_page_with_full_page_clip_and_positioned_text_is_visible() -> None:
    """Synthetic coverage of a full-page `re W* n` clip, positioned text,
    interior TJ kerning and table paint outside the text's ink bounds."""
    scanner = _load_scanner()
    graph = json.loads(LIBREOFFICE_GRAPH.read_text())

    assert scanner._classify_graph_text(graph) == ("visible_only", 0, 0)
    # The upper line is at y=800: treating this tall page as Letter hides it.
    graph["qpdf"][1]["obj:3 0 R"]["value"]["/MediaBox"] = LETTER
    assert scanner._classify_graph_text(graph) == ("hidden_proven", 1, 0)

    graph = json.loads(LIBREOFFICE_GRAPH.read_text())
    stream = graph["qpdf"][2]["obj:4 0 R"]["stream"]
    content = base64.b64decode(stream["data"])
    stream["data"] = base64.b64encode(
        content.replace(b"0 0 600 840 re W* n", b"0 0 60 840 re W* n")
    ).decode("ascii")
    # Both origins are at x=71, outside the narrowed clip; hidden is a summary flag.
    assert scanner._classify_graph_text(graph) == ("hidden_proven", 1, 0)


def test_synthetic_table_is_visible_with_narrow_glyph_widths() -> None:
    """Synthetic narrow glyphs end before a table divider; the one-em fallback
    would cross the divider. No source PDF content or embedded font is needed."""
    scanner = _load_scanner()
    graph = json.loads(KNOWN_NITRA_ACCEPTANCE_GRAPH.read_text())

    assert scanner._classify_graph_text(graph) == ("visible_only", 0, 0)
    # At 12 pt, 25 synthetic 300-unit glyphs end at x=190, before x=220.
    # Without widths the 25-em fallback reaches x=400 and crosses the divider.
    del graph["qpdf"][3]["obj:5 0 R"]["value"]["/Widths"]
    assert scanner._classify_graph_text(graph) == ("ambiguous", 0, 1)


@pytest.mark.parametrize("fixture", [LIBREOFFICE_GRAPH, KNOWN_NITRA_ACCEPTANCE_GRAPH])
def test_published_scanner_fixtures_contain_only_neutral_synthetic_data(fixture: Path) -> None:
    """Keep reconstructed source pages, font programs and metadata out of fixtures."""
    graph = json.loads(fixture.read_text())
    assert set(graph) == {"qpdf"}
    assert graph["qpdf"][0] == {"jsonversion": 2}
    assert sum(len(entry) for entry in graph["qpdf"][1:]) == 3
    objects = {key: value for entry in graph["qpdf"][1:] for key, value in entry.items()}
    assert set(objects) == {"obj:3 0 R", "obj:4 0 R", "obj:5 0 R"}
    assert set(objects["obj:3 0 R"]) == {"value"}
    assert set(objects["obj:4 0 R"]) == {"stream"}
    assert set(objects["obj:5 0 R"]) == {"value"}
    page = objects["obj:3 0 R"]["value"]
    assert set(page) == {"/Type", "/MediaBox", "/Contents", "/Resources"}
    assert page["/Type"] == "/Page"
    assert page["/Contents"] == "4 0 R"
    assert page["/Resources"] == {"/Font": {"/F1": "5 0 R"}}
    font = objects["obj:5 0 R"]["value"]
    assert set(font) <= {"/Type", "/Subtype", "/BaseFont", "/FirstChar", "/Widths"}
    assert font["/Type"] == "/Font"
    assert font["/Subtype"] in {"/Type1", "/TrueType"}
    assert font.get("/BaseFont", "/Helvetica") == "/Helvetica"
    assert type(font.get("/FirstChar", 0)) is int
    assert all(type(width) in {int, float} and 0 < width <= 1000
               for width in font.get("/Widths", []))
    stream = objects["obj:4 0 R"]["stream"]
    assert set(stream) == {"dict", "data"}
    assert stream["dict"] == {}
    content = base64.b64decode(stream["data"], validate=True)
    assert content.isascii()
    text = re.findall(rb"\(([^()]*)\)", content)
    assert b"".join(text) in {
        b"Synthetic scanner fixture",
        b"Synthetic scanner fixtureSynthetic scanner fixture",
    }
    syntax = re.sub(rb"\([^()]*\)", b" ", content).replace(b"[", b" ").replace(b"]", b" ")
    operators = {b"q", b"Q", b"re", b"W*", b"n", b"g", b"G", b"BT", b"ET", b"Tf",
                 b"Td", b"TJ", b"Tj", b"w", b"m", b"l", b"S", b"f", b"/F1"}
    assert all(token in operators or re.fullmatch(rb"-?\d+(?:\.\d+)?", token)
               for token in syntax.split())


def _simple_font_graph(content: bytes, *, first_char: int, widths: list[int]) -> dict[str, object]:
    font = {"/Subtype": "/TrueType", "/FirstChar": first_char, "/Widths": widths}
    return _page_graph(
        content,
        page={"/Resources": {"/Font": {"/F1": "5 0 R"}}},
        extra_objects=[{"obj:5 0 R": {"value": font}}],
    )


def test_real_narrow_glyph_widths_keep_a_run_out_of_a_neighbouring_border() -> None:
    """The v10 defect's shape, minimally: four narrow glyphs (width 300, not
    the naive 1000) must not reach a border placed just past their real end."""
    scanner = _load_scanner()
    content = b"BT /F1 12 Tf 100 100 Td (AAAA) Tj ET 0 g 120 95 20 20 re f"
    resolved = _simple_font_graph(content, first_char=65, widths=[300])

    assert scanner._classify_graph_text(resolved) == ("visible_only", 0, 0)


def test_font_without_widths_still_falls_back_to_the_one_em_estimate() -> None:
    """Same geometry as above, but the font carries no `/Widths` (e.g. a
    standard-14 font): falls back to one em per glyph, so the same border
    is still (correctly, conservatively) ambiguous."""
    scanner = _load_scanner()
    content = b"BT /F1 12 Tf 100 100 Td (AAAA) Tj ET 0 g 120 95 20 20 re f"
    font = {"/Subtype": "/TrueType", "/BaseFont": "/Helvetica"}
    unresolved = _page_graph(
        content,
        page={"/Resources": {"/Font": {"/F1": "5 0 R"}}},
        extra_objects=[{"obj:5 0 R": {"value": font}}],
    )

    assert scanner._classify_graph_text(unresolved) == ("ambiguous", 0, 1)


def test_identity_h_composite_font_uses_real_cid_widths() -> None:
    """A Type0/Identity-H font's 2-byte CIDs get real `/W` widths, keeping a
    narrow run out of a neighbouring border the same way a simple font does."""
    scanner = _load_scanner()
    content = b"BT /F1 12 Tf 100 100 Td <000A000A000A000A> Tj ET 0 g 120 95 20 20 re f"
    font = {
        "/Subtype": "/Type0",
        "/Encoding": "/Identity-H",
        "/DescendantFonts": ["6 0 R"],
    }
    descendant = {"/Subtype": "/CIDFontType2", "/DW": 1000, "/W": [10, [300]]}
    graph = _page_graph(
        content,
        page={"/Resources": {"/Font": {"/F1": "5 0 R"}}},
        extra_objects=[
            {"obj:5 0 R": {"value": font}},
            {"obj:6 0 R": {"value": descendant}},
        ],
    )

    assert scanner._classify_graph_text(graph) == ("visible_only", 0, 0)


def test_hidden_text_stays_hidden_with_resolved_real_widths() -> None:
    """Real glyph widths must not weaken the anti-hidden-text controls: an
    invisible render mode (`3 Tr`) stays hidden regardless of font metrics."""
    scanner = _load_scanner()
    content = b"BT /F1 12 Tf 3 Tr 100 100 Td (AAAA) Tj ET"
    graph = _simple_font_graph(content, first_char=65, widths=[300])

    assert scanner._classify_graph_text(graph) == ("hidden_proven", 1, 0)


def test_word_spacing_extends_the_run_only_with_a_resolved_simple_font() -> None:
    """Tw applies once per code-32 byte (PDF spec 9.3.3); with the font's
    byte width known, a large Tw correctly pushes the run into paint that a
    zero Tw over the same text never reaches."""
    scanner = _load_scanner()
    widths = [200] + [0] * 32 + [300]  # code 32 (space) -> 200, code 65 ('A') -> 300
    paint = b"0 g 150 95 20 20 re f"

    zero_tw = _simple_font_graph(
        b"BT /F1 12 Tf 100 100 Td (A A) Tj ET " + paint, first_char=32, widths=widths
    )
    assert scanner._classify_graph_text(zero_tw) == ("visible_only", 0, 0)

    wide_tw = _simple_font_graph(
        b"BT /F1 12 Tf 500 Tw 100 100 Td (A A) Tj ET " + paint, first_char=32, widths=widths
    )
    assert scanner._classify_graph_text(wide_tw) == ("ambiguous", 0, 1)


def test_scanner_q_and_Q_restore_geometry_and_reject_malformed_state() -> None:
    scanner = _load_scanner()

    assert scanner.classify_text_visibility(
        b"q 2 0 0 2 0 0 cm Q BT /F1 12 Tf (visible) Tj ET"
    ) == ("visible_only", 0, 0)
    assert scanner.classify_text_visibility(b"Q BT (unclassifiable) Tj ET") == (
        "ambiguous",
        0,
        1,
    )


def test_scanner_report_hidden_text_is_informational() -> None:
    scanner = _load_scanner()
    report = scanner._empty_report()
    report.update({"text_visibility": "hidden_proven", "hidden_text_objects": 1})

    assert scanner._feature_reason(report) is None


def test_text_color_must_have_proven_contrast_with_the_background() -> None:
    scanner = _load_scanner()

    hidden = (
        b"1 g BT /F1 12 Tf (white on default white) Tj ET",
        b"1 1 1 rg BT /F1 12 Tf (rgb white) Tj ET",
        b"0 0 0 0 k BT /F1 12 Tf (cmyk white) Tj ET",
        b"1 G BT /F1 12 Tf 1 Tr (white stroke) Tj ET",
        b"1 g 1 G BT /F1 12 Tf 2 Tr (white fill and stroke) Tj ET",
    )
    for stream in hidden:
        assert scanner.classify_text_visibility(stream) == ("hidden_proven", 1, 0)

    visible = (
        b"BT /F1 12 Tf (default black) Tj ET",
        b"0.25 g BT /F1 12 Tf (gray) Tj ET",
        b"1 0 0 rg BT /F1 12 Tf (red) Tj ET",
        b"0 1 1 0 k BT /F1 12 Tf (cmyk red) Tj ET",
        b"/DeviceRGB cs 1 0 0 sc BT /F1 12 Tf (space red) Tj ET",
        b"0 G BT /F1 12 Tf 1 Tr (black stroke) Tj ET",
        b"1 g 0 G BT /F1 12 Tf 2 Tr (contrasting stroke) Tj ET",
    )
    for stream in visible:
        assert scanner.classify_text_visibility(stream) == ("visible_only", 0, 0)


def test_painted_or_unknown_background_contrast_fails_closed() -> None:
    scanner = _load_scanner()

    ambiguous = (
        b"0 g 0 0 100 100 re f BT /F1 12 Tf (black on painted black) Tj ET",
        b"0 G 0 0 100 100 re S BT /F1 12 Tf (possibly over stroke) Tj ET",
        b"0 0 100 100 re sh BT /F1 12 Tf (shaded background) Tj ET",
        b"q 100 0 0 100 0 0 cm BI /W 1 /H 1 /BPC 1 /CS /G ID x EI Q BT /F1 12 Tf (image) Tj ET",
        b"BT /F1 12 Tf (possibly covered) Tj ET 0 g 0 0 100 100 re f",
        b"BT /F1 12 Tf (possibly covered) Tj ET 1 g 0 0 100 100 re f",
        b"BT /F1 12 Tf (possibly covered) Tj ET 0 0 100 100 re sh",
    )
    for stream in ambiguous:
        assert scanner.classify_text_visibility(stream) == ("ambiguous", 0, 1), stream

    assert scanner.classify_text_visibility(
        b"1 g 0 0 100 100 re f 0 g BT /F1 12 Tf (black on unchanged white) Tj ET"
    ) == ("visible_only", 0, 0)
    # glyph ink never hides glyph ink: a page of many text runs stays visible
    assert scanner.classify_text_visibility(
        b"BT /F1 12 Tf (first run) Tj (second run) Tj ET q BT 10 10 Td (third) Tj ET Q"
    ) == ("visible_only", 0, 0)
    assert scanner.classify_text_visibility(
        b"BT /F1 12 Tf (black) Tj ET 1 g BT (white over black is still white on white) Tj ET"
    ) == ("hidden_proven", 1, 0)
    assert scanner.classify_text_visibility(
        b"q 0 g 0 0 100 100 re f Q BT /F1 12 Tf (paint survives Q) Tj ET"
    ) == ("ambiguous", 0, 1)


def test_color_state_is_validated_and_unknown_spaces_fail_closed() -> None:
    scanner = _load_scanner()

    ambiguous = (
        b"NaN g BT /F1 12 Tf (bad gray) Tj ET",
        b"2 g BT /F1 12 Tf (gray out of range) Tj ET",
        b"1 1 rg BT /F1 12 Tf (malformed rgb) Tj ET",
        b"/Pattern cs 1 sc BT /F1 12 Tf (unknown space) Tj ET",
        b"/DeviceRGB cs 1 sc BT /F1 12 Tf (wrong components) Tj ET",
        b"/DeviceRGB cs /P1 scn BT /F1 12 Tf (unknown pattern) Tj ET",
        b"1 FutureColor BT /F1 12 Tf (unknown operator) Tj ET",
    )
    for stream in ambiguous:
        assert scanner.classify_text_visibility(stream) == ("ambiguous", 0, 1)


def test_q_Q_and_forms_inherit_and_restore_color_state() -> None:
    scanner = _load_scanner()

    assert scanner.classify_text_visibility(
        b"q 1 g BT /F1 12 Tf (hidden) Tj ET Q BT /F1 12 Tf (black restored) Tj ET"
    ) == ("hidden_proven", 1, 0)
    inherited = _form_graph(
        b"1 g /Fm Do",
        b"BT /F1 12 Tf (inherited white) Tj ET",
    )
    restored = _form_graph(
        b"/Fm Do BT /F1 12 Tf (page black) Tj ET",
        b"1 g BT /F1 12 Tf (form white) Tj ET",
    )

    assert scanner._classify_graph_text(inherited) == ("hidden_proven", 1, 0)
    assert scanner._classify_graph_text(restored) == ("hidden_proven", 1, 0)


def test_form_paint_makes_following_page_text_contrast_ambiguous() -> None:
    scanner = _load_scanner()
    before_text = _form_graph(
        b"/Fm Do BT /F1 12 Tf (possibly over form paint) Tj ET",
        b"0 g 0 0 100 100 re f",
    )
    after_text = _form_graph(
        b"BT /F1 12 Tf (possibly covered by form) Tj ET /Fm Do",
        b"0 g 0 0 100 100 re f",
    )

    assert scanner._classify_graph_text(before_text) == ("ambiguous", 0, 1)
    assert scanner._classify_graph_text(after_text) == ("ambiguous", 0, 1)


def test_form_xobject_matrix_makes_nested_text_visibility_ambiguous() -> None:
    scanner = _load_scanner()

    for matrix in (
        [1, 0, 0, 1, 100000, 100000],
        [0, 0, 0, 0, 0, 0],
        [10**1000, 0, 0, 1, 0, 0],
    ):
        graph = _form_graph(
            b"/Fm Do",
            b"BT /F1 12 Tf (transformed text) Tj ET",
            form_dictionary={"/Matrix": matrix},
        )
        assert scanner._classify_graph_text(graph) == ("ambiguous", 0, 1)


def test_form_xobject_bbox_clipping_is_never_silently_ignored() -> None:
    scanner = _load_scanner()

    inside = _form_graph(
        b"/Fm Do",
        b"BT /F1 12 Tf 20 20 Td (inside the bbox) Tj ET",
        form_dictionary={"/BBox": [0, 0, 100, 100]},
    )
    outside = _form_graph(
        b"/Fm Do",
        b"BT /F1 12 Tf 200 200 Td (outside the bbox) Tj ET",
        form_dictionary={"/BBox": [0, 0, 100, 100]},
    )
    moved = _form_graph(
        b"1 0 0 1 300 300 cm /Fm Do",
        b"BT /F1 12 Tf 20 20 Td (bbox follows the ctm) Tj ET",
        form_dictionary={"/BBox": [0, 0, 100, 100]},
    )
    malformed = _form_graph(
        b"/Fm Do",
        b"BT /F1 12 Tf (invalid clip) Tj ET",
        form_dictionary={"/BBox": [0, 0, 0, 0]},
    )
    tiny = _form_graph(
        b"/Fm Do",
        b"BT /F1 12 Tf 1 1 Td (bbox smaller than the glyphs) Tj ET",
        form_dictionary={"/BBox": [0, 0, 5, 5]},
    )

    assert scanner._classify_graph_text(inside) == ("visible_only", 0, 0)
    assert scanner._classify_graph_text(outside) == ("hidden_proven", 1, 0)
    assert scanner._classify_graph_text(moved) == ("visible_only", 0, 0)
    assert scanner._classify_graph_text(malformed) == ("ambiguous", 0, 1)
    assert scanner._classify_graph_text(tiny) == ("ambiguous", 0, 1)


def test_form_xobject_inherits_text_rendering_mode_from_invoking_stream() -> None:
    scanner = _load_scanner()
    graph = _form_graph(b"BT 3 Tr ET /Fm Do", b"BT (inherited hidden text) Tj ET")

    assert scanner._classify_graph_text(graph) == ("hidden_proven", 1, 0)


def test_form_xobject_inherits_and_restores_text_font_size() -> None:
    scanner = _load_scanner()

    inherited = _form_graph(
        b"BT /F1 0 Tf ET /Fm Do",
        b"BT (inherited zero size) Tj ET",
    )
    restored = _form_graph(
        b"BT /F1 12 Tf ET /Fm Do BT (visible after form) Tj ET",
        b"BT /F1 0 Tf ET",
    )

    assert scanner._classify_graph_text(inherited) == ("hidden_proven", 1, 0)
    assert scanner._classify_graph_text(restored) == ("visible_only", 0, 0)


def test_form_font_size_mutations_fail_closed() -> None:
    scanner = _load_scanner()
    mutations = (
        b"BT /F1 0 Tf (zero nested size) Tj ET",
        b"BT /F1 -1 Tf (negative nested size) Tj ET",
        b"BT /F1 NaN Tf (non-finite nested size) Tj ET",
        b"BT /F1 Tf (malformed nested size) Tj ET",
        b"BT (unresolved nested size) Tj ET",
        b"BT /F1 12 Tf 0 0 0 0 0 0 Tm (degenerate nested size) Tj ET",
    )

    for form_data in mutations:
        graph = _form_graph(b"/Fm Do", form_data)
        result = scanner._classify_graph_text(graph)
        if b" 0 Tf" in form_data:
            assert result == ("hidden_proven", 1, 0)
        else:
            assert result == ("ambiguous", 0, 1)


def test_nested_forms_inherit_and_restore_text_font_size() -> None:
    scanner = _load_scanner()
    inner = {
        "obj:6 0 R": _stream(
            b"BT (inner text) Tj ET",
            **{"/Type": "/XObject", "/Subtype": "/Form"},
        )
    }
    inherited = _form_graph(
        b"/Outer Do",
        b"BT /F1 0 Tf ET /Inner Do",
        page_xobjects={"/Outer": "5 0 R", "/Inner": "6 0 R"},
        extra_objects=[inner],
    )
    restored = _form_graph(
        b"BT /F1 12 Tf ET /Outer Do BT (page text) Tj ET",
        b"/Inner Do",
        page_xobjects={"/Outer": "5 0 R", "/Inner": "6 0 R"},
        extra_objects=[
            {
                "obj:6 0 R": _stream(
                    b"BT /F1 0 Tf ET",
                    **{"/Type": "/XObject", "/Subtype": "/Form"},
                )
            }
        ],
    )

    assert scanner._classify_graph_text(inherited) == ("hidden_proven", 1, 0)
    assert scanner._classify_graph_text(restored) == ("visible_only", 0, 0)


def test_form_xobject_inherits_clip_optional_content_and_extgstate() -> None:
    scanner = _load_scanner()
    mutations = (
        b"0 0 m 10 0 l 5 10 l h W n /Fm Do",
        b"/OC /Layer BDC /Fm Do EMC",
        b"/GS1 gs /Fm Do",
    )

    for invocation in mutations:
        graph = _form_graph(invocation, b"BT /F1 12 Tf 100 100 Td (nested text) Tj ET")
        assert scanner._classify_graph_text(graph) == ("ambiguous", 0, 1), invocation
    clipped = _form_graph(
        b"0 0 10 10 re W n /Fm Do", b"BT /F1 12 Tf 100 100 Td (nested text) Tj ET"
    )
    assert scanner._classify_graph_text(clipped) == ("hidden_proven", 1, 0)


def test_nested_form_uses_inherited_resources_and_finds_hidden_text() -> None:
    scanner = _load_scanner()
    graph = _form_graph(
        b"/Outer Do",
        b"/Inner Do",
        page_xobjects={"/Outer": "5 0 R", "/Inner": "6 0 R"},
        extra_objects=[
            {
                "obj:6 0 R": _stream(
                    b"BT 3 Tr (hidden nested text) Tj ET",
                    **{"/Type": "/XObject", "/Subtype": "/Form"},
                )
            }
        ],
    )

    assert scanner._classify_graph_text(graph) == ("hidden_proven", 1, 0)


def test_cyclic_form_xobject_graph_fails_closed() -> None:
    scanner = _load_scanner()
    graph = _form_graph(
        b"/Loop Do",
        b"/Loop Do",
        page_xobjects={"/Loop": "5 0 R"},
    )

    assert scanner._classify_graph_text(graph) == ("ambiguous", 0, 1)


def test_form_xobject_depth_limit_fails_closed() -> None:
    scanner = _load_scanner()
    xobjects = {f"/F{index}": f"{index + 5} 0 R" for index in range(33)}
    forms = [
        {
            f"obj:{index + 5} 0 R": _stream(
                f"/F{index + 1} Do".encode() if index < 32 else b"BT (deep text) Tj ET",
                **{"/Type": "/XObject", "/Subtype": "/Form"},
            )
        }
        for index in range(33)
    ]
    graph = _form_graph(
        b"/F0 Do",
        b"/F1 Do",
        page_xobjects=xobjects,
        extra_objects=forms[1:],
    )

    assert scanner._classify_graph_text(graph) == ("ambiguous", 0, 1)


def _counts(scanner, objects: dict) -> dict:
    entries = [{f"obj:{k}": {"value": v}} for k, v in objects.items()]
    graph = {"qpdf": [{"jsonversion": 2}, *entries]}
    return scanner.classify_object_graph(graph, max_image_width=6000, max_image_height=6000)


def test_scanner_does_not_count_tagged_structure_attributes_or_page_destinations_as_actions():
    """26.9. T11 on prod: every LibreOffice/Word PDF was refused as
    PDF_ACTIVE_CONTENT because a tagged PDF's structure elements carry `/A`
    (attribute dictionaries, ISO 32000 14.7.6) and the catalog's `/OpenAction`
    is a page destination array. Neither is an action."""
    scanner = _load_scanner()
    counts = _counts(scanner, {
        "1 0 R": {"/Type": "/Catalog", "/OpenAction": ["3 0 R", "/XYZ", 0, 792, 0]},
        "2 0 R": {"/Type": "/StructElem", "/S": "/TH", "/A": {"/O": "/Table", "/ColSpan": 2}},
        "4 0 R": {"/Type": "/StructElem", "/S": "/Table", "/A": "5 0 R"},
        "5 0 R": {"/O": "/Layout", "/Placement": "/Block"},
        "6 0 R": {"/Type": "/Catalog", "/OpenAction": "7 0 R"},
        "7 0 R": {"/S": "/GoTo", "/D": ["3 0 R", "/Fit"]},
    })
    assert counts["javascript_action_objects"] == 0


def test_scanner_still_counts_real_actions_and_refuses_unresolvable_references():
    scanner = _load_scanner()
    assert _counts(scanner, {"1 0 R": {"/Type": "/Catalog", "/OpenAction": "2 0 R"},
                             "2 0 R": {"/S": "/JavaScript", "/JS": "app.alert(1)"}})[
        "javascript_action_objects"] == 2
    assert _counts(scanner, {"1 0 R": {"/Type": "/Annot", "/Subtype": "/Link",
                                        "/A": {"/S": "/Launch", "/F": "cmd.exe"}}})[
        "javascript_action_objects"] == 1
    assert _counts(scanner, {"1 0 R": {"/Type": "/Page", "/AA": {"/O": "9 0 R"}}})[
        "javascript_action_objects"] == 1
    assert _counts(scanner, {"1 0 R": {"/Type": "/Catalog", "/OpenAction": "99 0 R"}})[
        "javascript_action_objects"] == 1  # unresolvable reference: fail closed
    assert _counts(scanner, {"1 0 R": {"/Type": "/StructElem", "/A": "99 0 R"}})[
        "javascript_action_objects"] == 1


def test_scanner_link_uri_action_is_an_annotation_not_a_script():
    scanner = _load_scanner()
    counts = _counts(scanner, {"1 0 R": {"/Type": "/Annot", "/Subtype": "/Link",
                                          "/A": {"/S": "/URI", "/URI": "https://example.org"}}})
    assert counts["javascript_action_objects"] == 0
    assert counts["annotation_objects"] == 1


@pytest.mark.parametrize(("content", "extra_objects", "visibility"), [
    (b"BT /F1 12 Tf (visible) Tj ET",
     [{"obj:5 0 R": {"value": {"/Type": "/Annot", "/Subtype": "/Link",
       "/A": {"/S": "/URI", "/URI": "https://example.org"}}}}], "visible_only"),
    (b"BT 3 Tr (private OCR) Tj ET", [], "hidden_proven"),
    (b"BT 4 Tr (clipping text) Tj ET", [], "ambiguous"),
])
def test_inspect_pdf_visible_raster_profile_accepts_features(
    monkeypatch, content: bytes, extra_objects: list[dict], visibility: str
) -> None:
    scanner = _load_scanner()
    _stub_inspection(monkeypatch, scanner, _page_graph(content, extra_objects=extra_objects))

    report = scanner.inspect_pdf(Path("source.pdf"))

    assert report["status"] == "accepted"
    assert report["reason"] is None
    assert report["text_visibility"] == visibility
    assert "private OCR" not in repr(report)


def _stub_inspection(monkeypatch, scanner, graph: object, *, json_rc: int = 0,
                     check_rc: int = 0, stderr: bytes = b"") -> None:
    limits = {"RASTER_DPI": 300, "MAX_PAGE_WIDTH_POINTS": 1440,
              "MAX_PAGE_HEIGHT_POINTS": 1440, "MAX_RASTER_WIDTH_PIXELS": 6000,
              "MAX_RASTER_HEIGHT_PIXELS": 6000, "MAX_RASTER_PIXELS": 36000000,
              "MAX_PDF_PAGES": 100}
    for key, value in limits.items():
        monkeypatch.setenv(key, str(value))

    def run(command: list[str]):
        if command[0] == "pdfinfo":
            output = b"Encrypted: no\nPages: 1\nPage 1 size: 612 x 792 pts\nPage 1 rot: 0\n"
            return subprocess.CompletedProcess(command, 0, output)
        if command[1] == "--check":
            return subprocess.CompletedProcess(command, check_rc, b"")
        return subprocess.CompletedProcess(command, json_rc, json.dumps(graph).encode(), stderr)

    monkeypatch.setattr(scanner, "_run", run)


@pytest.mark.parametrize(("feature", "reason"), [
    ({"/Type": "/Catalog", "/Names": {"/EmbeddedFiles": 42}}, "PDF_EMBEDDED_FILE"),
    ({"/Type": "/Annot", "/A": {"/S": "/JavaScript", "/JS": "secret()"}},
     "PDF_ACTIVE_CONTENT"),
])
def test_inspect_pdf_warning_json_keeps_feature_refusal(monkeypatch, feature, reason) -> None:
    scanner = _load_scanner()
    graph = _page_graph(b"", extra_objects=[{"obj:5 0 R": {"value": feature}}])
    _stub_inspection(monkeypatch, scanner, graph, json_rc=3)

    report = scanner.inspect_pdf(Path("source.pdf"))

    assert report["status"] == "rejected"
    assert report["reason"] == reason


@pytest.mark.parametrize("graph", [
    {}, {"qpdf": []}, {"qpdf": [{"jsonversion": 2}, {"obj:3 0 R": {}}]},
    _page_graph(b"", page={"/Contents": "99 0 R"}),
])
def test_inspect_pdf_warning_unusable_graph_rejected(monkeypatch, graph) -> None:
    scanner = _load_scanner()
    _stub_inspection(monkeypatch, scanner, graph, json_rc=3)

    assert scanner.inspect_pdf(Path("source.pdf"))["reason"] == "PDF_MALFORMED"


def test_inspect_pdf_warning_without_governed_refusal_rejected(monkeypatch) -> None:
    scanner = _load_scanner()
    _stub_inspection(monkeypatch, scanner, _page_graph(b""), json_rc=3)

    assert scanner.inspect_pdf(Path("source.pdf"))["reason"] == "PDF_MALFORMED"


@pytest.mark.parametrize(("check_rc", "json_rc"), [(3, 0), (2, 0), (0, 2)])
def test_inspect_pdf_failed_qpdf_check_or_json_rejected(monkeypatch, check_rc, json_rc) -> None:
    scanner = _load_scanner()
    _stub_inspection(monkeypatch, scanner, _page_graph(b""),
                     check_rc=check_rc, json_rc=json_rc)

    assert scanner.inspect_pdf(Path("source.pdf"))["reason"] == "PDF_MALFORMED"



def _c2pa_graph(*, spelling: str = "/Filespec", medail_stream: bool = False) -> dict:
    stream_dict = ({"/F": {"/Length": 4, "/Subtype": "u:application/c2pa"}}
                   if medail_stream else {"/Type": "/EmbeddedFile"})
    return _page_graph(b"", extra_objects=[
        {"obj:5 0 R": {"value": {"/Type": "/Catalog", "/AF": ["6 0 R"],
         "/Names": {"/EmbeddedFiles": {"/Names": ["u:Content Credentials", "6 0 R"]}}}}},
        {"obj:6 0 R": {"value": {"/Type": spelling,
         "/AFRelationship": "/C2PA_Manifest", "/F": "u:Content Credentials",
         "/Subtype": "u:application/c2pa", "/EF": {"/F": "7 0 R"}}}},
        {"obj:7 0 R": _stream(b"manifest", **stream_dict)},
    ])


def _c2pa_warning(*, object_number: int = 6) -> bytes:
    return (f"WARNING: source.pdf, object {object_number} 0 at offset 123: "
            "Embedded file object's type is not /Filespec\n"
            "qpdf: operation succeeded with warnings\n").encode()


@pytest.mark.parametrize(("spelling", "medail_stream", "json_rc", "stderr"), [
    ("/Filespec", False, 0, b""),
    ("/FileSpec", True, 3, _c2pa_warning()),
])
def test_inspect_pdf_declared_c2pa_metadata_admitted_as_pixels(
    monkeypatch, spelling, medail_stream, json_rc, stderr
) -> None:
    scanner = _load_scanner()
    graph = _c2pa_graph(spelling=spelling, medail_stream=medail_stream)
    _stub_inspection(monkeypatch, scanner, graph, json_rc=json_rc, stderr=stderr)

    report = scanner.inspect_pdf(Path("source.pdf"))

    assert report["status"] == "accepted"
    assert report["attachment_objects"] == 3
    assert report["c2pa_metadata_objects"] == 3
    assert "manifest" not in repr(report)


@pytest.mark.parametrize("mutation", ["ordinary", "mismarked", "unreferenced", "shared",
                                       "script", "wrong_stream"])
def test_inspect_pdf_c2pa_exception_does_not_admit_other_features(monkeypatch, mutation) -> None:
    scanner = _load_scanner()
    graph = _c2pa_graph()
    spec = graph["qpdf"][4]["obj:6 0 R"]["value"]
    if mutation == "ordinary":
        graph["qpdf"].append({"obj:8 0 R": {"value": {"/Type": "/Filespec",
                                 "/EF": {"/F": "7 0 R"}}}})
    elif mutation == "mismarked":
        spec["/AFRelationship"] = "/Data"
    elif mutation == "unreferenced":
        del graph["qpdf"][3]["obj:5 0 R"]["value"]["/AF"]
        del graph["qpdf"][3]["obj:5 0 R"]["value"]["/Names"]
    elif mutation == "shared":
        graph["qpdf"][1]["obj:3 0 R"]["value"]["/Contents"] = "7 0 R"
    elif mutation == "wrong_stream":
        spec["/EF"]["/F"] = "4 0 R"
    else:
        spec["/A"] = {"/S": "/JavaScript", "/JS": "danger()"}
    _stub_inspection(monkeypatch, scanner, graph)

    report = scanner.inspect_pdf(Path("source.pdf"))

    assert report["status"] == "rejected"
    assert report["reason"] == ("PDF_ACTIVE_CONTENT" if mutation == "script"
                                else "PDF_EMBEDDED_FILE")


@pytest.mark.parametrize("stderr", [b"", _c2pa_warning(object_number=99),
    _c2pa_warning() + b"WARNING: another issue\n",
    _c2pa_warning().replace(b"at offset 123", b"at offset unknown"),
])
def test_inspect_pdf_c2pa_unmatched_warning_rejected(monkeypatch, stderr) -> None:
    scanner = _load_scanner()
    _stub_inspection(monkeypatch, scanner, _c2pa_graph(spelling="/FileSpec"),
                     json_rc=3, stderr=stderr)

    assert scanner.inspect_pdf(Path("source.pdf"))["reason"] == "PDF_MALFORMED"



@pytest.mark.parametrize("nested", [
    {"/Type": "/Filespec", "/F": "u:ordinary.txt"},
    {"/Subtype": "/FileAttachment", "/FS": "99 0 R"},
])
def test_inspect_pdf_nested_attachment_cannot_hide_in_metadata(monkeypatch, nested) -> None:
    scanner = _load_scanner()
    graph = _c2pa_graph()
    graph["qpdf"][4]["obj:6 0 R"]["value"]["/Unexpected"] = nested
    _stub_inspection(monkeypatch, scanner, graph)

    assert scanner.inspect_pdf(Path("source.pdf"))["reason"] == "PDF_EMBEDDED_FILE"


@pytest.mark.parametrize("declaration", ["file_annotation", "indirect_tree"])
def test_inspect_pdf_c2pa_supported_declarations_admitted(monkeypatch, declaration) -> None:
    scanner = _load_scanner()
    graph = _c2pa_graph()
    catalog = graph["qpdf"][3]["obj:5 0 R"]["value"]
    del catalog["/AF"]
    if declaration == "file_annotation":
        del catalog["/Names"]
        graph["qpdf"].append({"obj:8 0 R": {"value": {"/Type": "/Annot",
                                  "/Subtype": "/FileAttachment", "/FS": "6 0 R"}}})
    else:
        catalog["/Names"] = "8 0 R"
        graph["qpdf"].extend([
            {"obj:8 0 R": {"value": {"/EmbeddedFiles": "9 0 R"}}},
            {"obj:9 0 R": {"value": {"/Kids": ["10 0 R"]}}},
            {"obj:10 0 R": {"value": {"/Names": ["u:manifest.c2pa", "6 0 R"]}}},
        ])
    _stub_inspection(monkeypatch, scanner, graph)

    assert scanner.inspect_pdf(Path("source.pdf"))["status"] == "accepted"

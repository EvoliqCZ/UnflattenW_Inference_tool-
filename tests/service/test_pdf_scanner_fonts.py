from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

FONTS_PATH = Path(__file__).parents[2] / "deploy" / "pdf-sandbox" / "scanner_fonts.py"


def _load_fonts():
    sys.path.insert(0, str(FONTS_PATH.parent))
    spec = importlib.util.spec_from_file_location("pdf_admission_scanner_fonts", FONTS_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass field resolution needs this registered
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _resolve(refs: dict[str, object]):
    def resolve(value: object) -> object:
        return refs.get(value, value) if isinstance(value, str) else value

    return resolve


def test_simple_font_uses_widths_first_char_and_missing_width() -> None:
    fonts = _load_fonts()
    font = {
        "/Subtype": "/TrueType",
        "/FirstChar": 65,
        "/Widths": [722, 667, 611],
        "/FontDescriptor": {"/MissingWidth": 250},
    }
    table = fonts.build_width_table(font, _resolve({}))
    assert table is not None
    assert table.byte_width == 1
    assert table.lookup(65) == 722
    assert table.lookup(66) == 667
    assert table.lookup(67) == 611
    assert table.lookup(68) == 250  # outside FirstChar..LastChar -> MissingWidth
    assert table.lookup(32) == 250


def test_simple_font_missing_width_defaults_to_zero() -> None:
    fonts = _load_fonts()
    font = {"/Subtype": "/Type1", "/FirstChar": 32, "/Widths": [278]}
    table = fonts.build_width_table(font, _resolve({}))
    assert table is not None
    assert table.lookup(99) == 0.0  # no FontDescriptor/MissingWidth -> spec default 0


def test_simple_font_resolves_indirect_widths_and_first_char() -> None:
    fonts = _load_fonts()
    font = {"/Subtype": "/Type1", "/FirstChar": "3 0 R", "/Widths": "4 0 R"}
    resolve = _resolve({"3 0 R": 65, "4 0 R": [500, 600]})
    table = fonts.build_width_table(font, resolve)
    assert table is not None
    assert table.lookup(65) == 500
    assert table.lookup(66) == 600


def test_simple_font_without_widths_array_is_unresolved() -> None:
    """Standard-14 fonts (Helvetica, Times, ...) without an explicit /Widths
    array fall back to the one-em estimate -- no AFM tables are shipped."""
    fonts = _load_fonts()
    font = {"/Subtype": "/Type1", "/BaseFont": "/Helvetica"}
    assert fonts.build_width_table(font, _resolve({})) is None


def test_simple_font_with_malformed_widths_is_unresolved() -> None:
    fonts = _load_fonts()
    malformed = (
        {"/Subtype": "/Type1", "/FirstChar": 32, "/Widths": [500, "bad"]},
        {"/Subtype": "/Type1", "/FirstChar": 32, "/Widths": "not-a-list"},
        {"/Subtype": "/Type1", "/FirstChar": "not-an-int", "/Widths": [500]},
        {"/Subtype": "/Type1", "/FirstChar": 32, "/Widths": []},
        {"/Subtype": "/Type1", "/FirstChar": 32, "/Widths": [-5]},
    )
    for font in malformed:
        assert fonts.build_width_table(font, _resolve({})) is None, font


def test_type3_font_scales_widths_by_font_matrix() -> None:
    fonts = _load_fonts()
    font = {
        "/Subtype": "/Type3",
        "/FirstChar": 65,
        "/Widths": [500],
        "/FontMatrix": [0.002, 0, 0, 0.002, 0, 0],
    }
    table = fonts.build_width_table(font, _resolve({}))
    assert table is not None
    assert table.lookup(65) == 1000.0  # 500 glyph-space units * 0.002 * 1000


def test_type3_font_with_skewed_matrix_is_unresolved() -> None:
    fonts = _load_fonts()
    font = {
        "/Subtype": "/Type3",
        "/FirstChar": 65,
        "/Widths": [500],
        "/FontMatrix": [0.001, 0.1, 0, 0.001, 0, 0],
    }
    assert fonts.build_width_table(font, _resolve({})) is None


def test_composite_identity_h_uses_w_array_ranges_and_dw() -> None:
    fonts = _load_fonts()
    descendant = {
        "/Subtype": "/CIDFontType2",
        "/DW": 1000,
        "/W": [10, [200, 300, 400], 50, 60, 500],
    }
    font = {
        "/Subtype": "/Type0",
        "/Encoding": "/Identity-H",
        "/DescendantFonts": [descendant],
    }
    table = fonts.build_width_table(font, _resolve({}))
    assert table is not None
    assert table.byte_width == 2
    assert table.lookup(10) == 200
    assert table.lookup(11) == 300
    assert table.lookup(12) == 400
    assert table.lookup(55) == 500  # inside the 50..60 range
    assert table.lookup(9999) == 1000  # unlisted CID -> /DW


def test_composite_font_defaults_dw_to_1000_when_absent() -> None:
    fonts = _load_fonts()
    descendant = {"/Subtype": "/CIDFontType0", "/W": []}
    font = {
        "/Subtype": "/Type0",
        "/Encoding": "/Identity-H",
        "/DescendantFonts": [descendant],
    }
    table = fonts.build_width_table(font, _resolve({}))
    assert table is not None
    assert table.lookup(1) == 1000.0


def test_composite_font_non_identity_encoding_is_unresolved() -> None:
    fonts = _load_fonts()
    descendant = {"/Subtype": "/CIDFontType2", "/W": []}
    font = {
        "/Subtype": "/Type0",
        "/Encoding": "/UniGB-UCS2-H",
        "/DescendantFonts": [descendant],
    }
    assert fonts.build_width_table(font, _resolve({})) is None


def test_composite_font_with_malformed_w_array_is_unresolved() -> None:
    fonts = _load_fonts()
    malformed_w = (
        [10, [200, "bad"]],
        [10, 20],  # range missing the trailing width
        ["not-a-number", [200]],
        [10, 5, 100],  # end < start
    )
    for w in malformed_w:
        descendant = {"/Subtype": "/CIDFontType2", "/W": w}
        font = {
            "/Subtype": "/Type0",
            "/Encoding": "/Identity-H",
            "/DescendantFonts": [descendant],
        }
        assert fonts.build_width_table(font, _resolve({})) is None, w


def test_unknown_subtype_is_unresolved() -> None:
    fonts = _load_fonts()
    assert fonts.build_width_table({"/Subtype": "/CIDFontType2"}, _resolve({})) is None
    assert fonts.build_width_table({}, _resolve({})) is None

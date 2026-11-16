"""Font glyph-width resolution for the admission scanner (PDF spec 9.2.4,
9.6.2, 9.7.4.3): real per-glyph advances where a font unambiguously supports
them, replacing the one-em-per-glyph estimate that overstates a text run's
reach into neighbouring paint. A font this module cannot resolve returns
`None` -- callers fall back to the one-em estimate, never guess a width."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

Resolve = Callable[[object], object]

_SIMPLE_SUBTYPES = frozenset({"/Type1", "/MMType1", "/TrueType", "/Type3"})
_CID_SUBTYPES = frozenset({"/CIDFontType0", "/CIDFontType2"})
_MAX_CODE = 0xFFFF


@dataclass(frozen=True)
class WidthTable:
    """Resolved per-code advance widths, in thousandths of an em."""

    byte_width: int  # 1 for a simple font, 2 for an Identity-H composite font
    default: float
    widths: Mapping[int, float]

    def lookup(self, code: int) -> float:
        return self.widths.get(code, self.default)


def shown_extent(state: object) -> tuple[float, float]:
    """Ink span `[min, max]` (thousandths of an em) of a text-show operator:
    resolved per-glyph widths (spec 9.2.4) plus Tc/Tw when known, else the
    one-em-per-glyph fallback."""
    operands = state.operands  # type: ignore[attr-defined]
    items = operands[0].value if operands[0].kind == "array" else operands
    graphics = state.graphics  # type: ignore[attr-defined]
    font_size = graphics.font_size or 1.0  # font_size 0 is already "hidden"
    char_spacing = graphics.char_spacing * 1000.0 / font_size
    word_spacing = graphics.word_spacing * 1000.0 / font_size
    apply_word_spacing = word_spacing != 0 and word_spacing_resolved(graphics.font_widths)
    position = min_position = max_position = 0.0
    for item in items:
        if item.kind != "string":
            position -= float(item.value)
            continue
        for code, width in glyph_widths(item, graphics.font_widths):
            min_position = min(min_position, position)
            position += width
            max_position = max(max_position, position)
            position += char_spacing
            if apply_word_spacing and code == 32:
                position += word_spacing
    return min_position, max_position


def glyph_widths(item: object, widths: WidthTable | None) -> list[tuple[int | None, float]]:
    """Per-glyph `(code, width)` pairs for one shown string operand; `code`
    is `None` and `width` is one em (1000) whenever the font or the string's
    codes cannot be resolved -- the always-safe fallback estimate."""
    count = int(item.value)  # type: ignore[attr-defined]
    codes = item.codes  # type: ignore[attr-defined]
    if widths is None or len(codes) != count or count % widths.byte_width:
        return [(None, 1000.0)] * count
    step = widths.byte_width
    result = []
    for index in range(0, count, step):
        code = int.from_bytes(codes[index : index + step], "big")
        result.append((code, widths.lookup(code)))
    return result


def word_spacing_resolved(widths: WidthTable | None) -> bool:
    """Tw only ever applies to single-byte code 32 (PDF spec 9.3.3); we can
    only tell which glyph that is once the font's byte width is known."""
    return widths is not None and widths.byte_width == 1


def build_width_table(font: Mapping[str, object], resolve: Resolve) -> WidthTable | None:
    subtype = font.get("/Subtype")
    if subtype == "/Type0":
        return _composite_width_table(font, resolve)
    if subtype in _SIMPLE_SUBTYPES:
        return _simple_width_table(font, subtype, resolve)
    return None


def _simple_width_table(
    font: Mapping[str, object], subtype: object, resolve: Resolve
) -> WidthTable | None:
    first_char = resolve(font.get("/FirstChar"))
    raw_widths = resolve(font.get("/Widths"))
    if type(first_char) is not int or not isinstance(raw_widths, list) or not raw_widths:
        return None
    values = [_number(resolve(item)) for item in raw_widths]
    if any(value is None or value < 0 for value in values):
        return None
    scale = 1.0
    if subtype == "/Type3":
        scale = _type3_scale(font, resolve)
        if scale is None:
            return None
    widths = {
        first_char + index: value * scale
        for index, value in enumerate(values)
        if 0 <= first_char + index <= _MAX_CODE
    }
    return WidthTable(byte_width=1, default=_missing_width(font, resolve), widths=widths)


def _type3_scale(font: Mapping[str, object], resolve: Resolve) -> float | None:
    matrix = resolve(font.get("/FontMatrix"))
    if not isinstance(matrix, list) or len(matrix) != 6:
        return None
    parsed = [_number(resolve(item)) for item in matrix]
    if any(item is None for item in parsed):
        return None
    a, b, c, *_rest = parsed
    if b != 0.0 or c != 0.0:
        return None  # skewed/rotated glyph space: outside this model
    return a * 1000.0  # type: ignore[operator]


def _missing_width(font: Mapping[str, object], resolve: Resolve) -> float:
    descriptor = resolve(font.get("/FontDescriptor"))
    if isinstance(descriptor, Mapping):
        value = _number(resolve(descriptor.get("/MissingWidth")))
        if value is not None and value >= 0:
            return value
    return 0.0  # PDF spec 9.6.2.1 default when /MissingWidth is absent


def _composite_width_table(font: Mapping[str, object], resolve: Resolve) -> WidthTable | None:
    if font.get("/Encoding") != "/Identity-H":
        return None  # any other CMap needs decoding this module deliberately avoids
    descendants = resolve(font.get("/DescendantFonts"))
    if not isinstance(descendants, list) or len(descendants) != 1:
        return None
    descendant = resolve(descendants[0])
    if not isinstance(descendant, Mapping) or descendant.get("/Subtype") not in _CID_SUBTYPES:
        return None
    default_width = _number(resolve(descendant.get("/DW")))
    if default_width is None:
        default_width = 1000.0  # PDF spec 9.7.4.3 default
    elif default_width < 0:
        return None
    widths = _parse_cid_widths(resolve(descendant.get("/W")), resolve)
    if widths is None:
        return None
    return WidthTable(byte_width=2, default=default_width, widths=widths)


def _parse_cid_widths(entries: object, resolve: Resolve) -> dict[int, float] | None:
    if entries is None:
        return {}
    if not isinstance(entries, list):
        return None
    widths: dict[int, float] = {}
    index = 0
    while index < len(entries):
        start = _integer(resolve(entries[index]))
        index += 1
        if start is None or index >= len(entries):
            return None
        second = resolve(entries[index])
        index += 1
        if isinstance(second, list):
            for offset, item in enumerate(second):
                value = _number(resolve(item))
                if value is None or value < 0:
                    return None
                if 0 <= start + offset <= _MAX_CODE:
                    widths[start + offset] = value
            continue
        end = _integer(second)
        if end is None or end < start or index >= len(entries):
            return None
        value = _number(resolve(entries[index]))
        index += 1
        if value is None or value < 0:
            return None
        for code in range(start, min(end, _MAX_CODE) + 1):
            widths[code] = value
    return widths


def _integer(value: object) -> int | None:
    number = _number(value)
    return int(number) if number is not None and float(number).is_integer() else None


def _number(value: object) -> float | None:
    if type(value) not in {int, float}:
        return None
    try:
        parsed = float(value)
    except OverflowError:
        return None
    return parsed if math.isfinite(parsed) else None

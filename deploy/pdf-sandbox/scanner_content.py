from __future__ import annotations

import math
from collections.abc import Callable

import scanner_fonts as fonts
import scanner_geometry as geometry
import scanner_paint as paint
from scanner_lexer import ContentLexer
from scanner_state import ContentState, FontMetricsProvider, GraphicsState, Operand, Visibility

_MAX_TEXT_VALUE = 1_000_000.0
_MAX_SCALE = 10_000.0
_MIN_EFFECTIVE_SCALE = 0.000001
_MIN_LEGIBLE_SCALE = 1.0  # device points; smaller glyphs cannot be proven to show at 300 DPI
_FormInvoker = Callable[[str, GraphicsState, bool], Visibility]

_FIXED_SAFE_OPERATORS = {
    "J": ("number",), "j": ("number",), "ri": ("name",), "i": ("number",),
    "MP": ("name",), "DP": ("name", "any"), "BX": (), "EX": (),
}
_PATH_CONSTRUCTION_OPERANDS = {
    "m": ("number", "number"), "l": ("number", "number"), "c": ("number",) * 6,
    "v": ("number",) * 4, "y": ("number",) * 4, "h": (), "re": ("number",) * 4, "n": (),
}


def classify_text_visibility(data: bytes) -> tuple[str, int, int]:
    return scan_content(data, GraphicsState(), False, None).result()


def scan_content(
    data: bytes,
    graphics: GraphicsState,
    inherited_optional: bool,
    invoke_form: _FormInvoker | None,
    font_metrics: FontMetricsProvider | None = None,
) -> Visibility:
    state = ContentState(graphics, inherited_optional, font_metrics=font_metrics)
    lexer = ContentLexer(data)
    try:
        for item in lexer:
            if isinstance(item, Operand):
                state.operands.append(item)
                continue
            if state.path_open and item not in geometry.PATH_OPERATORS:
                state.visibility.ambiguous = True
            if item == "BI":
                if not _require(state, ()):
                    state.visibility.ambiguous = True
                lexer.skip_inline_image()
                paint.record_paint(state, geometry.image_paint_region(state.graphics))
            else:
                _apply_operator(state, item, invoke_form)
            state.operands.clear()
    except (ValueError, OverflowError):
        state.visibility.ambiguous = True
    if (
        state.operands
        or state.in_text
        or state.graphics_stack
        or state.optional_content
        or state.path_open
        or state.pending_clip
    ):
        state.visibility.ambiguous = True
    return state.visibility


def _apply_operator(state: ContentState, token: str, invoke_form: _FormInvoker | None) -> None:
    if token == "q":
        if _require(state, ()):
            state.graphics_stack.append(state.graphics.copy())
        else:
            state.visibility.ambiguous = True
    elif token == "Q":
        if not _require(state, ()) or not state.graphics_stack:
            state.visibility.ambiguous = True
        else:
            state.graphics.restore(state.graphics_stack.pop())
    elif token == "Do":
        _invoke_xobject(state, invoke_form)
    elif _apply_text_operator(state, token):
        return
    elif _apply_path_operator(state, token):
        return
    elif _apply_graphics_operator(state, token):
        return
    elif _apply_marked_content_operator(state, token):
        return
    elif not _apply_safe_operator(state, token):
        state.visibility.ambiguous = True


def _invoke_xobject(state: ContentState, invoke_form: _FormInvoker | None) -> None:
    if not _require(state, ("name",)) or invoke_form is None:
        state.visibility.ambiguous = True
        return
    name = str(state.operands[0].value)
    optional = state.inherited_optional or any(state.optional_content)
    nested = invoke_form(name, state.graphics.copy(), optional)
    paint.merge_nested(state, nested)


def _apply_text_operator(state: ContentState, token: str) -> bool:
    if token == "BT":
        valid = _require(state, ()) and not state.in_text
        state.visibility.ambiguous |= not valid
        state.in_text = True
        geometry.begin_text(state)
    elif token == "ET":
        valid = _require(state, ()) and state.in_text
        state.visibility.ambiguous |= not valid
        state.in_text = False
    elif token in {"Tc", "Tw", "Tz", "TL", "Tr", "Ts"}:
        _apply_text_state(state, token)
    elif token == "Tf":
        _apply_text_font(state)
    elif token in {"Tm", "Td", "TD", "T*"}:
        _apply_text_position(state, token)
    elif token in {"Tj", "TJ", "'", '"'}:
        _apply_text_show(state, token)
    else:
        return False
    return True


def _apply_text_state(state: ContentState, token: str) -> None:
    if not state.in_text or not _require(state, ("number",)):
        state.visibility.ambiguous = True
        return
    value = _number(state.operands[0])
    if value is None or abs(value) > _MAX_TEXT_VALUE:
        state.visibility.ambiguous = True
        return
    if token == "Tr":
        if not value.is_integer() or int(value) not in range(8):
            state.visibility.ambiguous = True
        else:
            state.graphics.rendering_mode = int(value)
    elif token == "Tz":
        if value < 0 or value > _MAX_SCALE:
            state.visibility.ambiguous = True
        else:
            state.graphics.horizontal_scale = value
    else:
        setattr(
            state.graphics,
            {"Tc": "char_spacing", "Tw": "word_spacing", "TL": "leading", "Ts": "rise"}[token],
            value,
        )


def _apply_text_font(state: ContentState) -> None:
    if not state.in_text or not _require(state, ("name", "number")):
        state.visibility.ambiguous = True
        return
    size = _number(state.operands[1])
    name = state.operands[0].value
    if (
        size is None
        or size < 0
        or size > _MAX_TEXT_VALUE
        or not isinstance(name, str)
        or len(name) < 2
    ):
        state.visibility.ambiguous = True
        return
    state.graphics.font_size = size
    state.graphics.font_widths = state.font_metrics(name) if state.font_metrics else None


def _apply_text_position(state: ContentState, token: str) -> None:
    kinds = () if token == "T*" else (("number",) * (6 if token == "Tm" else 2))
    if not state.in_text or not _require(state, kinds):
        state.visibility.ambiguous = True
        return
    values = tuple(_number(item) for item in state.operands)
    if any(value is None or abs(value) > _MAX_TEXT_VALUE for value in values):
        state.visibility.ambiguous = True
        return
    numeric = tuple(float(value) for value in values if value is not None)
    if token == "Tm":
        placed = geometry.set_text_matrix(state, numeric)  # type: ignore[arg-type]
    elif token == "T*":
        placed = geometry.next_text_line(state)
    else:
        if token == "TD":
            state.graphics.leading = -numeric[1]
        placed = geometry.move_text_line(state, numeric[0], numeric[1])
    state.visibility.ambiguous |= not placed


def _apply_text_show(state: ContentState, token: str) -> None:
    if not state.in_text:
        state.visibility.ambiguous = True
        return
    if token == "Tj":
        valid = _require(state, ("string",))
    elif token == "TJ":
        valid = _valid_tj_array(state)
    elif token == "'":
        valid = _require(state, ("string",)) and geometry.next_text_line(state)
    else:
        valid = _require(state, ("number", "number", "string"))
        first, second = (_number(item) for item in state.operands[:2])
        valid &= first is not None and second is not None
        if valid:
            state.graphics.word_spacing = float(first)
            state.graphics.char_spacing = float(second)
            valid = geometry.next_text_line(state)
    if not valid:
        state.visibility.ambiguous = True
        return
    graphics = state.graphics
    placement = geometry.text_placement(state)
    effective_scale = None
    run = None
    if graphics.font_size is not None and placement is not None:
        effective_scale = graphics.font_size * placement[1] * graphics.horizontal_scale / 100
        origin, _, rendering = placement
        run = paint.text_run(state, rendering, origin, effective_scale, *fonts.shown_extent(state))
    hidden = (
        graphics.rendering_mode in {3, 7}
        or graphics.font_size == 0
        or graphics.horizontal_scale == 0
        or (effective_scale is not None and effective_scale < _MIN_EFFECTIVE_SCALE)
    )
    if not hidden and graphics.rendering_mode in {0, 1, 2} and placement is not None:
        hidden = not geometry.origin_can_show(state, placement[0])
    uncertain = (
        state.inherited_optional
        or any(state.optional_content)
        or placement is None
        or effective_scale is None
        or effective_scale < _MIN_LEGIBLE_SCALE
        or (graphics.clip is not None and not geometry.covers(graphics.clip, effective_scale))
        or graphics.clip_ambiguous
        or graphics.external_state_ambiguous
        or run is None
        or not paint.background_known(state, run)
        or graphics.rendering_mode in {4, 5, 6}
        or (graphics.word_spacing != 0 and not fonts.word_spacing_resolved(graphics.font_widths))
        or graphics.horizontal_scale != 100
        or graphics.rise != 0
    )
    if hidden:
        state.visibility.hidden = True
    elif not uncertain:
        colors = _text_paint_colors(graphics)
        if any(color is None for color in colors):
            state.visibility.ambiguous = True
        elif all(color == graphics.background_color for color in colors):
            state.visibility.hidden = True
        elif run is not None:
            state.visibility.visible = True
            paint.record_run(state, run)  # glyph ink never hides glyph ink
    else:
        state.visibility.ambiguous = True


def _valid_tj_array(state: ContentState) -> bool:
    if not _require(state, ("array",)):
        return False
    values = state.operands[0].value
    if not isinstance(values, tuple):
        return False
    for item in values:
        if not isinstance(item, Operand) or item.kind not in {"string", "number"}:
            return False
        if item.kind == "number" and (
            _number(item) is None or abs(float(item.value)) > _MAX_TEXT_VALUE
        ):
            return False
    return True


def _apply_path_operator(state: ContentState, token: str) -> bool:
    kinds = _PATH_CONSTRUCTION_OPERANDS.get(token)
    if kinds is None and token not in {"W", "W*"}:
        return False
    if not _require(state, kinds or ()):
        state.visibility.ambiguous = True
        return True
    values = tuple(_number(item) for item in state.operands)
    if any(value is None or abs(value) > _MAX_TEXT_VALUE for value in values):
        state.visibility.ambiguous = True
        return True
    numbers = tuple(float(value) for value in values if value is not None)
    geometry.apply_path_operator(state, token, numbers)
    return True


def _apply_graphics_operator(state: ContentState, token: str) -> bool:
    if token == "cm":
        if not _require(state, ("number",) * 6):
            state.visibility.ambiguous = True
        else:
            values = tuple(_number(item) for item in state.operands)
            if any(value is None or abs(value) > _MAX_TEXT_VALUE for value in values):
                state.visibility.ambiguous = True
            else:
                matrix = tuple(float(value) for value in values if value is not None)
                geometry.apply_ctm(state, matrix)  # type: ignore[arg-type]
    elif token == "gs":
        if _require(state, ("name",)):
            state.graphics.external_state_ambiguous = True
        else:
            state.visibility.ambiguous = True
    elif token in {"w", "M"}:
        value = _number(state.operands[0]) if _require(state, ("number",)) else None
        if value is None or value < 0 or value > _MAX_TEXT_VALUE:
            state.visibility.ambiguous = True
        else:
            setattr(state.graphics, "line_width" if token == "w" else "miter_limit", value)
    elif token == "d":
        if not _require(state, ("array", "number")):
            state.visibility.ambiguous = True
    elif token in {"G", "g", "RG", "rg", "K", "k"}:
        _apply_device_color(state, token)
    elif token in {"CS", "cs"}:
        _apply_color_space(state, token)
    elif token in {"SC", "sc"}:
        _apply_color_components(state, token)
    elif token in {"SCN", "scn"}:
        state.visibility.ambiguous = True
        if token == "SCN":
            state.graphics.stroke_color = None
        else:
            state.graphics.fill_color = None
    elif token in {"S", "s"}:
        _paint_path(state, stroke=True, fill=False)
    elif token in {"f", "F", "f*"}:
        _paint_path(state, stroke=False, fill=True)
    elif token in {"B", "B*", "b", "b*"}:
        _paint_path(state, stroke=True, fill=True)
    elif token == "sh":
        if not _require(state, ("name",)):
            state.visibility.ambiguous = True
        paint.record_paint(state, state.graphics.clip)  # a shading fills the whole clip
    else:
        return False
    return True


def _apply_device_color(state: ContentState, token: str) -> None:
    component_count = {"G": 1, "g": 1, "RG": 3, "rg": 3, "K": 4, "k": 4}[token]
    stroke = token[0].isupper()
    color = _parse_color(state, component_count)
    space = {1: "/DeviceGray", 3: "/DeviceRGB", 4: "/DeviceCMYK"}[component_count]
    setattr(state.graphics, "stroke_color_space" if stroke else "fill_color_space", space)
    setattr(state.graphics, "stroke_color" if stroke else "fill_color", color)


def _apply_color_space(state: ContentState, token: str) -> None:
    stroke = token == "CS"
    if not _require(state, ("name",)):
        state.visibility.ambiguous = True
        return
    raw = str(state.operands[0].value)
    aliases = {"/G": "/DeviceGray", "/RGB": "/DeviceRGB", "/CMYK": "/DeviceCMYK"}
    space = aliases.get(raw, raw)
    known = space in {"/DeviceGray", "/DeviceRGB", "/DeviceCMYK"}
    if not known:
        state.visibility.ambiguous = True
    setattr(state.graphics, "stroke_color_space" if stroke else "fill_color_space", space)
    setattr(state.graphics, "stroke_color" if stroke else "fill_color", None)


def _apply_color_components(state: ContentState, token: str) -> None:
    stroke = token == "SC"
    space = getattr(state.graphics, "stroke_color_space" if stroke else "fill_color_space")
    counts = {"/DeviceGray": 1, "/DeviceRGB": 3, "/DeviceCMYK": 4}
    count = counts.get(space)
    color = None if count is None else _parse_color(state, count)
    if count is None:
        state.visibility.ambiguous = True
    setattr(state.graphics, "stroke_color" if stroke else "fill_color", color)


def _parse_color(state: ContentState, count: int) -> tuple[float, float, float] | None:
    if not _require(state, ("number",) * count):
        state.visibility.ambiguous = True
        return None
    values = tuple(_number(item) for item in state.operands)
    if any(value is None or not 0 <= value <= 1 for value in values):
        state.visibility.ambiguous = True
        return None
    components = tuple(float(value) for value in values if value is not None)
    if count == 1:
        return components * 3
    if count == 3:
        return components
    cyan, magenta, yellow, black = components
    return (
        1.0 - min(1.0, cyan + black),
        1.0 - min(1.0, magenta + black),
        1.0 - min(1.0, yellow + black),
    )


def _text_paint_colors(graphics: GraphicsState) -> tuple[tuple[float, float, float] | None, ...]:
    if graphics.rendering_mode == 0:
        return (graphics.fill_color,)
    if graphics.rendering_mode == 1:
        return (graphics.stroke_color,)
    return (graphics.fill_color, graphics.stroke_color)


def _paint_path(state: ContentState, *, stroke: bool, fill: bool) -> None:
    if not _require(state, ()):
        state.visibility.ambiguous = True
        return
    painted = state.path_bbox is not None or state.path_unbounded
    region = geometry.path_paint_region(state, stroke=stroke)
    geometry.end_path(state)
    if not painted:
        return
    colors = tuple(
        color
        for enabled, color in (
            (fill, state.graphics.fill_color),
            (stroke, state.graphics.stroke_color),
        )
        if enabled
    )
    background = all(color == state.graphics.background_color for color in colors)
    paint.record_paint(state, region, background=background)


def _apply_marked_content_operator(state: ContentState, token: str) -> bool:
    if token == "BDC":
        valid = _require(state, ("name", "any"))
        state.visibility.ambiguous |= not valid
        if valid:
            state.optional_content.append(state.operands[0].value == "/OC")
    elif token == "BMC":
        valid = _require(state, ("name",))
        state.visibility.ambiguous |= not valid
        if valid:
            state.optional_content.append(False)
    elif token == "EMC":
        if not _require(state, ()) or not state.optional_content:
            state.visibility.ambiguous = True
        else:
            state.optional_content.pop()
    else:
        return False
    return True


def _apply_safe_operator(state: ContentState, token: str) -> bool:
    kinds = _FIXED_SAFE_OPERATORS.get(token)
    if kinds is None:
        return False
    if not _require(state, kinds):
        state.visibility.ambiguous = True
    return True


def _require(state: ContentState, kinds: tuple[str, ...]) -> bool:
    return len(state.operands) == len(kinds) and all(
        expected == "any"
        or (
            operand.kind == expected
            and (expected != "number" or _number(operand) is not None)
            and (
                expected != "name"
                or isinstance(operand.value, str)
                and len(operand.value) > 1
            )
        )
        for operand, expected in zip(state.operands, kinds, strict=True)
    )


def _number(operand: Operand) -> float | None:
    value = operand.value
    if operand.kind != "number" or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None

"""Affine geometry for the admission scanner: CTM, text matrix, rectangular clips.

Every rectangle handled here is in device space. A `None` clip means unclipped; a
`None` page box means no outer bound is known (stream-only classification).
"""

from __future__ import annotations

import math

from scanner_state import ContentState, GraphicsState

Matrix = tuple[float, float, float, float, float, float]
Rect = tuple[float, float, float, float]

IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
_MIN_DETERMINANT = 1e-12
_PATH_CONSTRUCTION = {"m", "l", "c", "v", "y", "h", "re"}
_PATH_PAINTING = {"n", "S", "s", "f", "F", "f*", "B", "B*", "b", "b*"}
PATH_OPERATORS = _PATH_CONSTRUCTION | _PATH_PAINTING | {"W", "W*"}


def multiply(first: Matrix, second: Matrix) -> Matrix | None:
    """`first` applied before `second`; None when the product is not finite."""
    a1, b1, c1, d1, e1, f1 = first
    a2, b2, c2, d2, e2, f2 = second
    product = (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    )
    return product if all(math.isfinite(value) for value in product) else None


def invertible(matrix: Matrix) -> bool:
    determinant = matrix[0] * matrix[3] - matrix[1] * matrix[2]
    return math.isfinite(determinant) and abs(determinant) > _MIN_DETERMINANT


def translation(x: float, y: float) -> Matrix:
    return (1.0, 0.0, 0.0, 1.0, x, y)


def transform_point(matrix: Matrix, x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    return (a * x + c * y + e, b * x + d * y + f)


def x_scale(matrix: Matrix) -> float:
    return math.hypot(matrix[0], matrix[1])


def normalized_rect(x0: float, y0: float, x1: float, y1: float) -> Rect:
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def transform_rect(rect: Rect, matrix: Matrix) -> Rect | None:
    """The rectangle under an axis-aligned matrix; None for rotation or skew."""
    if matrix[1] != 0 or matrix[2] != 0:
        return None
    x0, y0 = transform_point(matrix, rect[0], rect[1])
    x1, y1 = transform_point(matrix, rect[2], rect[3])
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        return None
    return normalized_rect(x0, y0, x1, y1)


def intersect(current: Rect | None, other: Rect) -> Rect:
    if current is None:
        return other
    return (
        max(current[0], other[0]),
        max(current[1], other[1]),
        min(current[2], other[2]),
        min(current[3], other[3]),
    )


def contains(rect: Rect, x: float, y: float) -> bool:
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def covers(rect: Rect, extent: float) -> bool:
    """Whether a glyph of `extent` device units can fit inside the rectangle."""
    return rect[2] - rect[0] >= extent and rect[3] - rect[1] >= extent


def apply_ctm(state: ContentState, matrix: Matrix) -> None:
    composed = multiply(matrix, state.graphics.ctm) if invertible(matrix) else None
    if composed is None:
        state.graphics.ctm_ambiguous = True
    else:
        state.graphics.ctm = composed


def clip_to_rect(state: ContentState, rect: Rect | None) -> None:
    """Intersect the clip with a device rectangle; an unknown one fails closed."""
    if rect is None or state.graphics.ctm_ambiguous:
        state.graphics.clip_ambiguous = True
    else:
        state.graphics.clip = intersect(state.graphics.clip, rect)


def apply_path_operator(state: ContentState, token: str, numbers: tuple[float, ...]) -> None:
    """Track the current path: its device bound, and whether it is one rectangle."""
    if token == "re":
        state.path_open = True
        x, y, width, height = numbers
        rect = None
        if not state.graphics.ctm_ambiguous:
            rect = transform_rect(normalized_rect(x, y, x + width, y + height), state.graphics.ctm)
        if rect is None:
            state.path_nonrect = True
        else:
            state.path_rects.append(rect)
        _extend_path_bound(state, (x, y), (x + width, y), (x, y + height), (x + width, y + height))
    elif token in _PATH_CONSTRUCTION:
        state.path_open = True
        state.path_nonrect = True
        if token != "m":
            state.path_segments += 1
        points = tuple(zip(numbers[::2], numbers[1::2], strict=True))
        _extend_path_bound(state, *points)
    elif token in {"W", "W*"}:
        state.pending_clip = True
    elif token in _PATH_PAINTING:
        end_path(state)


def _extend_path_bound(state: ContentState, *points: tuple[float, float]) -> None:
    if state.graphics.ctm_ambiguous:
        state.path_unbounded = True
        return
    for x, y in points:
        dx, dy = transform_point(state.graphics.ctm, x, y)
        bound = state.path_bbox
        state.path_bbox = (
            (dx, dy, dx, dy)
            if bound is None
            else (min(bound[0], dx), min(bound[1], dy), max(bound[2], dx), max(bound[3], dy))
        )


def path_paint_region(state: ContentState, *, stroke: bool) -> Rect | None:
    """Device bound of the ink a painting operator can lay down; None when unknown."""
    if state.path_unbounded or state.path_bbox is None:
        return None
    x0, y0, x1, y1 = state.path_bbox
    if stroke:
        graphics = state.graphics
        width = graphics.line_width * max(x_scale(graphics.ctm), math.hypot(*graphics.ctm[2:4]))
        margin = width * (1.0 if state.path_segments <= 1 else max(1.0, graphics.miter_limit / 2))
        x0, y0, x1, y1 = x0 - margin, y0 - margin, x1 + margin, y1 + margin
    region = (x0, y0, x1, y1)
    return region if state.graphics.clip is None else intersect(state.graphics.clip, region)


def image_paint_region(graphics: GraphicsState) -> Rect | None:
    """Device bound of an image drawn into the unit square; None when unknown."""
    if graphics.ctm_ambiguous:
        return None
    xs, ys = zip(
        *(transform_point(graphics.ctm, x, y) for x in (0.0, 1.0) for y in (0.0, 1.0)),
        strict=True,
    )
    region = (min(xs), min(ys), max(xs), max(ys))
    return region if graphics.clip is None else intersect(graphics.clip, region)


def end_path(state: ContentState) -> None:
    if state.pending_clip:
        single = not state.path_nonrect and len(state.path_rects) == 1
        clip_to_rect(state, state.path_rects[0] if single else None)
    state.path_open = False
    state.path_nonrect = False
    state.path_rects.clear()
    state.path_bbox = None
    state.path_unbounded = False
    state.path_segments = 0
    state.pending_clip = False


def begin_text(state: ContentState) -> None:
    state.text_matrix = IDENTITY
    state.line_matrix = IDENTITY


def set_text_matrix(state: ContentState, matrix: Matrix) -> bool:
    if not invertible(matrix):
        return False
    state.text_matrix = matrix
    state.line_matrix = matrix
    return True


def move_text_line(state: ContentState, x: float, y: float) -> bool:
    moved = multiply(translation(x, y), state.line_matrix)
    if moved is None:
        return False
    state.text_matrix = moved
    state.line_matrix = moved
    return True


def next_text_line(state: ContentState) -> bool:
    return move_text_line(state, 0.0, -state.graphics.leading)


def text_placement(
    state: ContentState,
) -> tuple[tuple[float, float], float, Matrix] | None:
    """Device origin of the next glyph, the x-scale of text space there, and the matrix."""
    if state.graphics.ctm_ambiguous:
        return None
    rendering = multiply(state.text_matrix, state.graphics.ctm)
    if rendering is None:
        return None
    origin = transform_point(rendering, 0.0, 0.0)
    if not all(math.isfinite(value) for value in origin):
        return None
    return (origin, x_scale(rendering), rendering)


def origin_can_show(state: ContentState, origin: tuple[float, float]) -> bool:
    x, y = origin
    graphics = state.graphics
    if graphics.page_box is not None and not contains(graphics.page_box, x, y):
        return False
    return graphics.clip is None or contains(graphics.clip, x, y)
